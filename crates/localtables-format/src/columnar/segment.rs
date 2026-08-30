//! Writing and reading one segment.
//!
//! A segment holds a fixed set of rows and is never changed after it is
//! written. It is laid out as every column's buffers back to back, each padded
//! to the Arrow alignment, followed by a metadata frame:
//!
//! ```text
//! segment start (4096-aligned, so it can be mapped on its own)
//!   column 0: validity, offsets, values ...   each padded to 64 bytes
//!   column 1: ...
//!   SegmentMeta frame
//! segment end
//! ```
//!
//! Because the segment starts on a page boundary and every buffer starts on a
//! 64-byte boundary inside it, a mapped buffer is aligned for Arrow's widest
//! SIMD path without anything being moved.

use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{Schema, SchemaRef};
use std::sync::Arc;

use crate::columnar::bloom::BloomFilter;
use crate::columnar::trigram;
use crate::columnar::zonemap::ZoneMap;
use crate::columnar::decode::{decode_column, decode_column_rows, BufferSource, SegmentBytes};
use crate::columnar::encode::{compress_buffers, encode_column, EncodedColumn};
use crate::columnar::page::{
    ArchivedSegmentMeta, BufferRole, BufferSpec, Codec, ColumnChunk, SegmentMeta,
};
use crate::config::TableOptions;
use crate::io::buf::SharedBuf;
use crate::layout::frame::{self, tag};
use crate::layout::schema::SchemaLayout;
use crate::layout::{align_up, checksum, Extent, BUFFER_ALIGN};
use crate::{Error, Result};

/// A segment serialized into memory, ready to be placed in the file.
pub struct BuiltSegment {
    /// The bytes, from the segment's first buffer to the end of its metadata.
    pub bytes: Vec<u8>,
    /// Where the metadata frame sits, relative to the segment start.
    pub meta_extent: Extent,
    pub row_count: u64,
    pub meta: SegmentMeta,
}

impl BuiltSegment {
    pub fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The segment's extents once it is placed at `offset` in the file.
    pub fn placed(&self, offset: u64) -> (Extent, Extent) {
        (
            Extent::new(offset, self.len()),
            Extent::new(offset + self.meta_extent.offset, self.meta_extent.len),
        )
    }
}

/// Serialize `batches` into one segment.
///
/// All batches must share `schema`. They are encoded column by column, so a
/// column is only ever held in one piece.
pub fn build_segment(
    segment_id: u64,
    schema: &SchemaRef,
    schema_fingerprint: u64,
    batches: &[RecordBatch],
    options: &TableOptions,
) -> Result<BuiltSegment> {
    for batch in batches {
        if batch.schema().fields() != schema.fields() {
            return Err(Error::SchemaMismatch(format!(
                "a batch has schema {:?}, the table expects {:?}",
                batch.schema(),
                schema
            )));
        }
    }

    let row_count: usize = batches.iter().map(|b| b.num_rows()).sum();
    let columns = concat_columns(schema, batches, row_count)?;

    let mut bytes: Vec<u8> = Vec::new();
    let mut chunks = Vec::with_capacity(columns.len());

    for (index, array) in columns.iter().enumerate() {
        let encoded = encode_column(array.as_ref(), options)?;
        // Chosen per column, because whether a codec pays depends on what the
        // column holds: text compresses several times over, scattered numbers
        // not at all, and a codec that gains nothing still costs the column its
        // zero-copy read path.
        let codec = options
            .compression
            .codec_for(schema.field(index).data_type());
        // Built here rather than inside the encoder, because whether a column
        // gets a filter is a question about its name, not about its values.
        let name = schema.field(index).name();
        let bloom = if options.bloom_filters.covers(name) {
            BloomFilter::build(array.as_ref(), options.bloom_bits_per_value)?
        } else {
            None
        };
        let trigram = if options.trigram_filters.covers(name) {
            // The column's own stored size is the budget, so a filter never
            // outweighs the data it describes.
            trigram::build(
                array.as_ref(),
                options.bloom_bits_per_value,
                encoded.byte_len(),
            )?
        } else {
            None
        };
        // Bounds for each row range inside the segment, so a scan can skip the
        // ranges a predicate rules out rather than hand on the whole segment.
        let pages = page_zones(array.as_ref(), options.page_rows);
        chunks.push(write_blocked_chunk(
            &mut bytes,
            array.as_ref(),
            encoded,
            codec,
            options,
            bloom.as_ref(),
            trigram.as_ref(),
            pages.as_deref(),
        )?);
    }

    let meta = SegmentMeta {
        segment_id,
        row_count: row_count as u64,
        schema_fingerprint,
        // A segment with one page of bounds has none worth keeping: they would
        // repeat the chunk's own zone map.
        page_rows: if page_count(row_count, options.page_rows) > 1 {
            options.page_rows as u64
        } else {
            0
        },
        columns: chunks,
    };

    pad_to_alignment(&mut bytes);
    let meta_offset = bytes.len() as u64;
    let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&meta)?;
    let meta_frame = frame::encode(tag::SEGMENT, &payload);
    bytes.extend_from_slice(&meta_frame);

    Ok(BuiltSegment {
        meta_extent: Extent::new(meta_offset, meta_frame.len() as u64),
        row_count: row_count as u64,
        meta,
        bytes,
    })
}

/// How many pages of bounds a segment of `rows` rows holds.
fn page_count(rows: usize, page_rows: usize) -> usize {
    if page_rows == 0 {
        return 0;
    }
    rows.div_ceil(page_rows)
}

/// Bounds for each page of one column.
///
/// `None` when the segment is one page or fewer, where these would only repeat
/// the chunk's own zone map.
fn page_zones(array: &dyn Array, page_rows: usize) -> Option<Vec<ZoneMap>> {
    if page_count(array.len(), page_rows) < 2 {
        return None;
    }
    let mut zones = Vec::with_capacity(page_count(array.len(), page_rows));
    let mut start = 0;
    while start < array.len() {
        let len = page_rows.min(array.len() - start);
        zones.push(ZoneMap::build(&array.slice(start, len)));
        start += len;
    }
    Some(zones)
}

/// A column of nulls for rows written before the column existed.
fn absent_column(field: &arrow_schema::Field, rows: usize) -> ArrayRef {
    arrow_array::new_null_array(field.data_type(), rows)
}

/// Bring each column together across the batches.
///
/// One batch is the common case and costs nothing. Several are concatenated
/// once per column, which is what a segment is: a column-at-a-time rewrite of
/// row-at-a-time input.
fn concat_columns(
    schema: &SchemaRef,
    batches: &[RecordBatch],
    row_count: usize,
) -> Result<Vec<ArrayRef>> {
    (0..schema.fields().len())
        .map(|index| -> Result<ArrayRef> {
            match batches.len() {
                0 => Ok(arrow_array::new_empty_array(
                    schema.field(index).data_type(),
                )),
                1 => Ok(batches[0].column(index).clone()),
                _ => {
                    let parts: Vec<&dyn Array> =
                        batches.iter().map(|b| b.column(index).as_ref()).collect();
                    let joined = arrow_select::concat::concat(&parts)?;
                    debug_assert_eq!(joined.len(), row_count);
                    Ok(joined)
                }
            }
        })
        .collect()
}

/// Append one encoded column to the segment, padding each buffer.
/// Write a column, cut into independently compressed blocks where that helps.
///
/// Only a compressed column is cut. Splitting an uncompressed one would cost it
/// the zero-copy read path — a range spanning blocks has to be concatenated —
/// and buy nothing, since there is nothing to decompress.
#[allow(clippy::too_many_arguments)]
fn write_blocked_chunk(
    bytes: &mut Vec<u8>,
    array: &dyn Array,
    encoded: EncodedColumn,
    codec: Codec,
    options: &TableOptions,
    bloom: Option<&BloomFilter>,
    trigram: Option<&BloomFilter>,
    pages: Option<&[ZoneMap]>,
) -> Result<ColumnChunk> {
    let block_rows = options.compression_block_rows;
    let rows = array.len();
    if codec == Codec::None || block_rows == 0 || rows <= block_rows {
        return write_chunk(bytes, encoded, codec, bloom, trigram, pages);
    }

    let mut blocks = Vec::with_capacity(rows.div_ceil(block_rows));
    let mut start = 0;
    while start < rows {
        let len = block_rows.min(rows - start);
        let piece = array.slice(start, len);
        let piece = encode_column(piece.as_ref(), options)?;
        blocks.push(write_chunk(bytes, piece, codec, None, None, None)?);
        start += len;
    }

    // The outer chunk keeps what describes the column as a whole and holds no
    // buffers of its own; the blocks hold the data.
    let mut outer = write_chunk(
        bytes,
        EncodedColumn {
            buffers: Vec::new(),
            children: Vec::new(),
            ..encoded
        },
        codec,
        bloom,
        trigram,
        pages,
    )?;
    outer.block_rows = block_rows as u64;
    // The outer chunk holds no buffers, so `write_chunk` had nothing to judge
    // the codec from. What the column is actually stored with is what its
    // blocks are stored with.
    outer.codec = if blocks.iter().any(|block| block.codec != Codec::None) {
        codec
    } else {
        Codec::None
    };
    outer.blocks = blocks;
    Ok(outer)
}

fn write_chunk(
    bytes: &mut Vec<u8>,
    encoded: EncodedColumn,
    codec: Codec,
    bloom: Option<&BloomFilter>,
    trigram: Option<&BloomFilter>,
    pages: Option<&[ZoneMap]>,
) -> Result<ColumnChunk> {
    let stored = compress_buffers(codec, &encoded.buffers)?;

    // A chunk is only zero-copy if every one of its buffers stayed raw. When
    // some shrank and others did not, record the codec and let the reader
    // decide per buffer by comparing stored and uncompressed lengths.
    let chunk_codec = if stored.iter().any(|(_, b)| b.codec != Codec::None) {
        codec
    } else {
        Codec::None
    };

    let mut specs = Vec::with_capacity(stored.len());
    for (role, buffer) in stored {
        pad_to_alignment(bytes);
        let offset = bytes.len() as u64;
        let slice = buffer.as_slice();
        bytes.extend_from_slice(slice);
        specs.push(BufferSpec {
            role,
            extent: Extent::new(offset, slice.len() as u64),
            uncompressed_len: buffer.uncompressed_len,
            checksum: checksum(slice),
        });
    }

    // Stored raw and after the loop above, so they take no part in the codec
    // decision. A filter is close to random bits, which no codec shrinks.
    let mut side: Vec<(BufferRole, Vec<u8>)> = Vec::new();
    if let Some(filter) = bloom {
        side.push((BufferRole::Bloom, filter.to_bytes()));
    }
    if let Some(filter) = trigram {
        side.push((BufferRole::Trigram, filter.to_bytes()));
    }
    if let Some(pages) = pages {
        side.push((
            BufferRole::PageZones,
            rkyv::to_bytes::<rkyv::rancor::Error>(&pages.to_vec())?.to_vec(),
        ));
    }
    for (role, stored) in side {
        pad_to_alignment(bytes);
        let offset = bytes.len() as u64;
        bytes.extend_from_slice(&stored);
        specs.push(BufferSpec {
            role,
            extent: Extent::new(offset, stored.len() as u64),
            uncompressed_len: stored.len() as u64,
            checksum: checksum(&stored),
        });
    }

    let children = encoded
        .children
        .into_iter()
        .map(|child| write_chunk(bytes, child, codec, None, None, None))
        .collect::<Result<Vec<_>>>()?;

    Ok(ColumnChunk {
        encoding: encoded.encoding,
        codec: chunk_codec,
        len: encoded.len,
        null_count: encoded.null_count,
        dict_len: encoded.dict_len,
        run_count: encoded.run_count,
        offset: encoded.offset,
        buffers: specs,
        children,
        zone: encoded.zone,
        block_rows: 0,
        blocks: Vec::new(),
    })
}

fn pad_to_alignment(bytes: &mut Vec<u8>) {
    let padded = align_up(bytes.len() as u64, BUFFER_ALIGN) as usize;
    bytes.resize(padded, 0);
}

/// A segment's bytes plus its validated metadata.
///
/// Holding this keeps the mapping alive, which is what lets arrays decoded from
/// it point straight at the file.
pub struct SegmentReader {
    bytes: SharedBuf,
    /// Offset of the metadata frame inside `bytes`.
    meta_offset: usize,
    meta_len: usize,
    schema: SchemaRef,
}

impl SegmentReader {
    /// Validate a segment's metadata and keep its bytes for decoding.
    ///
    /// `bytes` covers the whole segment; `meta` locates the metadata frame
    /// inside the file, and is converted to a segment-relative offset here.
    pub fn new(
        bytes: SharedBuf,
        segment_start: u64,
        meta: Extent,
        schema: SchemaRef,
        layout: &SchemaLayout,
    ) -> Result<Self> {
        let meta_offset = meta.offset.checked_sub(segment_start).ok_or_else(|| {
            Error::corrupt(format!(
                "segment metadata at {} sits before the segment start {segment_start}",
                meta.offset
            ))
        })? as usize;
        let meta_len = meta.len as usize;
        if meta_offset + meta_len > bytes.len() {
            return Err(Error::corrupt(format!(
                "segment metadata at {meta_offset}..{} runs past the {}-byte segment",
                meta_offset + meta_len,
                bytes.len()
            )));
        }

        let reader = Self {
            bytes,
            meta_offset,
            meta_len,
            schema,
        };

        // Check now, so nothing downstream has to wonder whether it was
        // checked. A segment may hold fewer columns than the schema, which is
        // what a column added since it was written looks like; it may not hold
        // different ones, and the fingerprint is what tells those apart.
        let meta = reader.meta()?;
        let columns = meta.columns.len();
        if !layout.accepts(columns, meta.schema_fingerprint.to_native()) {
            return Err(Error::SchemaMismatch(format!(
                "a {columns}-column segment stamped {:#018x} does not match the first \
                 {columns} of this {}-column schema",
                meta.schema_fingerprint.to_native(),
                layout.columns()
            )));
        }
        Ok(reader)
    }

    /// The segment's metadata, read in place.
    ///
    /// The archive is validated on every call rather than cached, because
    /// caching a reference into `bytes` would make this struct self-referential
    /// for no measurable gain: validation is a bounds walk over a few hundred
    /// bytes, and a scan calls it once per segment.
    pub fn meta(&self) -> Result<&ArchivedSegmentMeta> {
        let frame = &self.bytes.as_slice()[self.meta_offset..self.meta_offset + self.meta_len];
        let payload = frame::decode(frame, tag::SEGMENT, "segment metadata")?;
        rkyv::access::<ArchivedSegmentMeta, rkyv::rancor::Error>(payload).map_err(Error::from)
    }

    pub fn row_count(&self) -> Result<u64> {
        Ok(self.meta()?.row_count.to_native())
    }

    /// The membership filter for one column, when the segment stored one.
    ///
    /// Reads only the filter's bytes, not the column's. A segment written
    /// before filters were asked for simply has none, and reports `None`.
    pub fn bloom_filter(&self, index: usize) -> Result<Option<BloomFilter>> {
        self.filter(index, BufferRole::Bloom)
    }

    /// The trigram filter for one column, when the segment stored one.
    pub fn trigram_filter(&self, index: usize) -> Result<Option<BloomFilter>> {
        self.filter(index, BufferRole::Trigram)
    }

    /// The row range one page covers, clamped to the segment.
    pub fn page_range(&self, page: usize) -> Result<Option<(usize, usize)>> {
        let meta = self.meta()?;
        let page_rows = meta.page_rows.to_native() as usize;
        if page_rows == 0 {
            return Ok(None);
        }
        let rows = meta.row_count.to_native() as usize;
        let start = page.checked_mul(page_rows).filter(|start| *start < rows);
        Ok(start.map(|start| (start, page_rows.min(rows - start))))
    }

    /// Rows each page of bounds covers, or zero when the segment has none.
    pub fn page_rows(&self) -> Result<u64> {
        Ok(self.meta()?.page_rows.to_native())
    }

    /// Bounds for each page of one column, when the segment stored them.
    ///
    /// Reads only the bounds, not the column. A segment written without them,
    /// or one small enough that its own zone map says as much, reports `None`
    /// and rules nothing out.
    pub fn page_zones(&self, index: usize) -> Result<Option<Vec<ZoneMap>>> {
        let Some(stored) = self.side_buffer(index, BufferRole::PageZones)? else {
            return Ok(None);
        };
        let zones = rkyv::from_bytes::<Vec<ZoneMap>, rkyv::rancor::Error>(stored)?;
        Ok(Some(zones))
    }

    fn filter(&self, index: usize, role: BufferRole) -> Result<Option<BloomFilter>> {
        let Some(stored) = self.side_buffer(index, role)? else {
            return Ok(None);
        };
        BloomFilter::from_bytes(stored).map(Some)
    }

    /// The bytes of a buffer that is not Arrow's, checked against its checksum.
    fn side_buffer(&self, index: usize, role: BufferRole) -> Result<Option<&[u8]>> {
        let meta = self.meta()?;
        let Some(chunk) = meta.columns.get(index) else {
            return Ok(None);
        };
        let Some(spec) = chunk.buffer(role) else {
            return Ok(None);
        };

        let start = spec.extent.offset.to_native() as usize;
        let end = start
            .checked_add(spec.extent.len.to_native() as usize)
            .ok_or_else(|| Error::corrupt(format!("a {role:?} buffer's extent overflows")))?;
        let all = self.bytes.as_slice();
        if end > all.len() {
            return Err(Error::corrupt(format!(
                "a {role:?} buffer at {start}..{end} runs past the {}-byte segment",
                all.len()
            )));
        }

        let stored = &all[start..end];
        if checksum(stored) != spec.checksum.to_native() {
            return Err(Error::corrupt(format!(
                "a {role:?} buffer failed its checksum"
            )));
        }
        Ok(Some(stored))
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// True when arrays decoded from this segment can avoid copying.
    pub fn is_zero_copy(&self) -> bool {
        self.bytes.is_zero_copy()
    }

    /// Decode one column by its position in the table schema.
    ///
    /// A column the segment predates reads as nulls. That is what a column
    /// added without a rewrite means: the rows are older than the column, so
    /// they have no value for it.
    pub fn column(&self, index: usize) -> Result<ArrayRef> {
        let meta = self.meta()?;
        let field = self.schema.fields().get(index).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "column {index} is out of range for a {}-column schema",
                self.schema.fields().len()
            ))
        })?;
        let Some(chunk) = meta.columns.get(index) else {
            return Ok(absent_column(field, meta.row_count.to_native() as usize));
        };
        let source = SegmentBytes::new(self.bytes.clone());
        decode_column(chunk, field.data_type(), &source)
    }

    /// Decode a range of one column's rows.
    pub fn column_rows(&self, index: usize, start: usize, len: usize) -> Result<ArrayRef> {
        let meta = self.meta()?;
        let field = self.schema.fields().get(index).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "column {index} is out of range for a {}-column schema",
                self.schema.fields().len()
            ))
        })?;
        let Some(chunk) = meta.columns.get(index) else {
            return Ok(absent_column(field, len));
        };
        let source = SegmentBytes::new(self.bytes.clone());
        decode_column_rows(chunk, field.data_type(), &source, start, len)
    }

    /// Decode the named columns into a batch.
    ///
    /// Columns outside `projection` are never touched: their bytes are not
    /// read, decompressed, or decoded.
    pub fn read(&self, projection: Option<&[usize]>) -> Result<RecordBatch> {
        let rows = self.meta()?.row_count.to_native() as usize;
        self.read_rows(projection, 0, rows)
    }

    /// Decode the named columns for a range of rows.
    ///
    /// Only these rows are expanded. For a column stored as a dictionary or as
    /// runs, that is the difference between paying for the segment and paying
    /// for the range.
    pub fn read_rows(
        &self,
        projection: Option<&[usize]>,
        start: usize,
        len: usize,
    ) -> Result<RecordBatch> {
        let indices: Vec<usize> = match projection {
            Some(indices) => indices.to_vec(),
            None => (0..self.schema.fields().len()).collect(),
        };

        let fields: Vec<_> = indices
            .iter()
            .map(|&i| {
                self.schema.fields().get(i).cloned().ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "projected column {i} is out of range for a {}-column schema",
                        self.schema.fields().len()
                    ))
                })
            })
            .collect::<Result<_>>()?;

        // Read the metadata once for the whole batch. Reaching for it per
        // column re-checksums and re-validates the same frame each time, which
        // a scan pays once per column per segment for nothing.
        let meta = self.meta()?;
        let source = SegmentBytes::new(self.bytes.clone());
        let columns: Vec<ArrayRef> = indices
            .iter()
            .map(|&i| {
                let field = self.schema.field(i);
                let Some(chunk) = meta.columns.get(i) else {
                    // Older than the column: no value, so null.
                    return Ok(absent_column(field, len));
                };
                decode_column_rows(chunk, field.data_type(), &source, start, len)
            })
            .collect::<Result<_>>()?;

        let projected = Arc::new(Schema::new(fields));
        if columns.is_empty() {
            // A count-only scan projects nothing; the batch still has to carry
            // the row count.
            let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(len));
            return RecordBatch::try_new_with_options(projected, columns, &options)
                .map_err(Error::from);
        }
        RecordBatch::try_new(projected, columns).map_err(Error::from)
    }

    /// Every byte range the projected columns occupy.
    ///
    /// A backend that can batch reads uses this to fetch a whole projection in
    /// one submission.
    pub fn projected_extents(&self, projection: Option<&[usize]>) -> Result<Vec<Extent>> {
        let meta = self.meta()?;
        let indices: Vec<usize> = match projection {
            Some(indices) => indices.to_vec(),
            None => (0..meta.columns.len()).collect(),
        };
        let mut out = Vec::new();
        for index in indices {
            let chunk = meta
                .columns
                .get(index)
                .ok_or_else(|| Error::InvalidArgument(format!("column {index} is out of range")))?;
            let owned: ColumnChunk = rkyv::deserialize::<_, rkyv::rancor::Error>(chunk)?;
            out.extend(owned.extents());
        }
        Ok(out)
    }
}

impl std::fmt::Debug for SegmentReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentReader")
            .field("bytes", &self.bytes.len())
            .field("meta_offset", &self.meta_offset)
            .field("columns", &self.schema.fields().len())
            .field("zero_copy", &self.bytes.is_zero_copy())
            .finish()
    }
}

/// Read the metadata of a segment held in memory, for tests and diagnostics.
pub fn read_meta(bytes: &[u8], meta_extent: Extent) -> Result<SegmentMeta> {
    let frame = &bytes[meta_extent.range()];
    let payload = frame::decode(frame, tag::SEGMENT, "segment metadata")?;
    let archived = rkyv::access::<ArchivedSegmentMeta, rkyv::rancor::Error>(payload)?;
    rkyv::deserialize::<_, rkyv::rancor::Error>(archived).map_err(Error::from)
}

/// Placate the unused-import lint when only some features are on.
#[allow(dead_code)]
fn _uses(source: &dyn BufferSource) -> Result<SharedBuf> {
    source.fetch(Extent::EMPTY)
}
