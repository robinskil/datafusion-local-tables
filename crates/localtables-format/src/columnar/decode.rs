//! Turning stored buffers back into Arrow arrays.
//!
//! This is the hot path. A plain, uncompressed column becomes an Arrow array by
//! pointing at the bytes where they already are: the mapping stays alive
//! because the Arrow buffer holds it, and nothing is copied or allocated
//! beyond a few reference counts.
//!
//! Compressed or re-encoded columns cost one pass each, and only for the
//! columns a query actually projects.

use arrow_array::{make_array, Array, ArrayRef};
use arrow_buffer::{Buffer, NullBuffer};
use arrow_data::{ArrayData, ArrayDataBuilder};
use arrow_schema::{DataType, Field};
use std::sync::Arc;

use crate::columnar::codec;
use crate::columnar::page::{ArchivedBufferSpec, ArchivedColumnChunk, BufferRole, Codec, Encoding};
use crate::io::buf::SharedBuf;
use crate::layout::{verify_checksum, Extent};
use crate::{Error, Result};

/// Supplies the bytes of one stored buffer.
///
/// The scan fetches a segment once and hands out windows of it, so this
/// normally costs a pointer adjustment rather than a read.
pub trait BufferSource {
    /// Bytes for `extent`, which is relative to the start of the segment.
    fn fetch(&self, extent: Extent) -> Result<SharedBuf>;
}

/// Serves buffers out of one mapped or read segment.
pub struct SegmentBytes {
    bytes: SharedBuf,
}

impl SegmentBytes {
    pub fn new(bytes: SharedBuf) -> Self {
        Self { bytes }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// True when reads off this segment cost no copy.
    pub fn is_zero_copy(&self) -> bool {
        self.bytes.is_zero_copy()
    }
}

impl BufferSource for SegmentBytes {
    fn fetch(&self, extent: Extent) -> Result<SharedBuf> {
        let end = extent.offset as usize + extent.len as usize;
        if end > self.bytes.len() {
            return Err(Error::corrupt(format!(
                "buffer at {extent:?} runs past the {}-byte segment",
                self.bytes.len()
            )));
        }
        Ok(self.bytes.slice(extent.offset as usize..end))
    }
}

/// Read one buffer, verify it, and decompress it if it was compressed.
fn load(spec: &ArchivedBufferSpec, codec: Codec, source: &dyn BufferSource) -> Result<Buffer> {
    let stored = source.fetch(spec.extent.to_native())?;
    verify_checksum(
        "column buffer",
        stored.as_slice(),
        spec.checksum.to_native(),
    )?;

    let uncompressed_len = spec.uncompressed_len.to_native() as usize;
    if codec.is_none() || stored.len() == uncompressed_len {
        // Either nothing was compressed, or this buffer did not shrink and was
        // stored raw. Both go straight to Arrow with no copy.
        return Ok(stored.into_arrow_buffer());
    }
    let expanded = codec::decompress(codec, stored.as_slice(), uncompressed_len)?;
    Ok(SharedBuf::from_owned(expanded).into_arrow_buffer())
}

/// Rebuild one column as an Arrow array of `data_type`.
///
/// Every encoding goes through the same generic rebuild; the encodings differ
/// only in what type the stored array has and what has to happen afterwards.
pub fn decode_column(
    chunk: &ArchivedColumnChunk,
    data_type: &DataType,
    source: &dyn BufferSource,
) -> Result<ArrayRef> {
    let rows = chunk.len.to_native() as usize;
    decode_column_rows(chunk, data_type, source, 0, rows)
}

/// Decode a range of a column's rows.
///
/// The stored array is always assembled whole, because assembling it is buffer
/// wrapping and costs nothing: no byte of a range nobody asked for is read, and
/// on a mapped file those pages are never faulted in.
///
/// What the range does change is the expansion. A dictionary or run-length
/// chunk has to be turned back into the type the schema declares, and that is
/// work proportional to the rows expanded. Slicing before the expansion rather
/// than after is the difference between paying for the segment and paying for
/// the rows.
pub fn decode_column_rows(
    chunk: &ArchivedColumnChunk,
    data_type: &DataType,
    source: &dyn BufferSource,
    start: usize,
    len: usize,
) -> Result<ArrayRef> {
    let stored_rows = chunk.len.to_native() as usize;
    if start > stored_rows || start + len > stored_rows {
        return Err(Error::InvalidArgument(format!(
            "rows {start}..{} are outside a {stored_rows}-row chunk",
            start + len
        )));
    }

    // A chunk cut into blocks holds no buffers of its own. Only the blocks the
    // range touches are read, which for a compressed column is the whole point:
    // the rest are never decompressed.
    if !chunk.blocks.is_empty() {
        return decode_blocks(chunk, data_type, source, start, len);
    }

    let whole = start == 0 && len == stored_rows;

    match chunk.encoding.to_native() {
        // Stored as the column's own type. Nothing to undo, and a slice of it
        // shares the same buffers.
        Encoding::Plain => {
            let array = make_array(rebuild(chunk, data_type, source)?);
            Ok(if whole { array } else { array.slice(start, len) })
        }

        // Stored as a dictionary over the column's type, then cast back. The
        // cast is the price of the smaller file; a column declared as a
        // dictionary in the schema takes the Plain path instead and pays
        // nothing.
        //
        // Slicing first costs nothing — a dictionary array slices its keys and
        // keeps its values — and leaves the cast expanding `len` rows instead
        // of the whole chunk.
        Encoding::Dictionary => {
            let stored =
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(data_type.clone()));
            let array = make_array(rebuild(chunk, &stored, source)?);
            let array = if whole { array } else { array.slice(start, len) };
            arrow_cast::cast(&array, data_type)
                .map_err(|e| Error::corrupt(format!("a dictionary chunk failed to expand: {e}")))
        }

        Encoding::RunLength => {
            let (run_ends, values) = run_end_fields(data_type);
            let stored = DataType::RunEndEncoded(run_ends, values);
            let array = make_array(rebuild(chunk, &stored, source)?);
            let array = if whole { array } else { array.slice(start, len) };
            arrow_cast::cast(&array, data_type)
                .map_err(|e| Error::corrupt(format!("a run-length chunk failed to expand: {e}")))
        }
    }
}

/// Decode a range out of a chunk that is stored in blocks.
///
/// A range inside one block costs one block. A range spanning several is
/// concatenated, which copies; that is the price of being able to decompress
/// the parts independently, and it is why an uncompressed chunk is never cut.
fn decode_blocks(
    chunk: &ArchivedColumnChunk,
    data_type: &DataType,
    source: &dyn BufferSource,
    start: usize,
    len: usize,
) -> Result<ArrayRef> {
    let block_rows = chunk.block_rows.to_native() as usize;
    if block_rows == 0 {
        return Err(Error::corrupt("a chunk holds blocks but no block size"));
    }
    if len == 0 {
        return Ok(arrow_array::new_empty_array(data_type));
    }

    let first = start / block_rows;
    let last = (start + len - 1) / block_rows;
    let mut parts: Vec<ArrayRef> = Vec::with_capacity(last - first + 1);

    for index in first..=last {
        let block = chunk.blocks.get(index).ok_or_else(|| {
            Error::corrupt(format!(
                "block {index} is missing from a chunk of {}",
                chunk.blocks.len()
            ))
        })?;
        let block_start = index * block_rows;
        let block_len = block.len.to_native() as usize;

        // Where this block overlaps the range asked for.
        let from = start.max(block_start) - block_start;
        let to = (start + len).min(block_start + block_len) - block_start;
        parts.push(decode_column_rows(
            block,
            data_type,
            source,
            from,
            to - from,
        )?);
    }

    if parts.len() == 1 {
        return Ok(parts.pop().expect("one part"));
    }
    let refs: Vec<&dyn arrow_array::Array> = parts.iter().map(|p| p.as_ref()).collect();
    arrow_select::concat::concat(&refs)
        .map_err(|e| Error::corrupt(format!("blocks of a column failed to join: {e}")))
}

/// Rebuild the Arrow array a chunk stores, whatever its type.
///
/// The buffers go back to Arrow in the order they were taken, and the children
/// are rebuilt the same way. Nothing here knows what any particular buffer
/// means — `build` validates, so a chunk whose bytes do not fit the type is
/// reported rather than handed on as data.
fn rebuild(
    chunk: &ArchivedColumnChunk,
    data_type: &DataType,
    source: &dyn BufferSource,
) -> Result<ArrayData> {
    let len = chunk.len.to_native() as usize;
    let codec = chunk.codec.to_native();

    let mut builder = ArrayDataBuilder::new(data_type.clone())
        .len(len)
        .offset(chunk.offset.to_native() as usize)
        .nulls(decode_nulls(chunk, len, source)?);

    for spec in chunk.data_buffers() {
        builder = builder.add_buffer(load(spec, codec, source)?);
    }

    let child_types = child_data_types(data_type);
    if chunk.children.len() != child_types.len() {
        return Err(Error::corrupt(format!(
            "a {data_type} column holds {} children, the type has {}",
            chunk.children.len(),
            child_types.len()
        )));
    }
    for (child, child_type) in chunk.children.iter().zip(child_types) {
        builder = builder.add_child_data(rebuild(child, &child_type, source)?);
    }

    builder
        .build()
        .map_err(|e| Error::corrupt(format!("a {data_type} column failed Arrow's checks: {e}")))
}

/// The types of a data type's child arrays, in Arrow's own order.
fn child_data_types(data_type: &DataType) -> Vec<DataType> {
    match data_type {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::Map(field, _) => vec![field.data_type().clone()],
        DataType::Struct(fields) => fields.iter().map(|f| f.data_type().clone()).collect(),
        DataType::Union(fields, _) => fields.iter().map(|(_, f)| f.data_type().clone()).collect(),
        DataType::Dictionary(_, values) => vec![values.as_ref().clone()],
        DataType::RunEndEncoded(run_ends, values) => {
            vec![run_ends.data_type().clone(), values.data_type().clone()]
        }
        _ => Vec::new(),
    }
}

/// The null bitmap, when the chunk stores one.
fn decode_nulls(
    chunk: &ArchivedColumnChunk,
    len: usize,
    source: &dyn BufferSource,
) -> Result<Option<NullBuffer>> {
    let Some(spec) = chunk.buffer(BufferRole::Validity) else {
        return Ok(None);
    };
    let buffer = load(spec, chunk.codec.to_native(), source)?;
    let required = len.div_ceil(8);
    if buffer.len() < required {
        return Err(Error::corrupt(format!(
            "a {len}-row chunk needs {required} bitmap bytes, found {}",
            buffer.len()
        )));
    }
    Ok(Some(NullBuffer::new(arrow_buffer::BooleanBuffer::new(
        buffer, 0, len,
    ))))
}

/// The field a run-length chunk's values belong to. Kept for symmetry with the
/// encoder, which names the same two fields when it builds the target type.
pub fn run_end_fields(value_type: &DataType) -> (Arc<Field>, Arc<Field>) {
    (
        Arc::new(Field::new("run_ends", DataType::Int32, false)),
        Arc::new(Field::new("values", value_type.clone(), true)),
    )
}

/// True when this array's buffers point into `source` rather than a copy.
///
/// Only meaningful right after a decode, and only used by tests and
/// diagnostics to prove the fast path stayed fast.
pub fn borrows_from(array: &dyn Array, source: &SharedBuf) -> bool {
    let base = source.as_slice().as_ptr_range();
    array
        .to_data()
        .buffers()
        .iter()
        .all(|b| base.contains(&b.as_ptr()))
}
