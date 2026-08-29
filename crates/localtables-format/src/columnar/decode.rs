//! Turning stored buffers back into Arrow arrays.
//!
//! This is the hot path. A plain, uncompressed column becomes an Arrow array by
//! pointing at the bytes where they already are: the mapping stays alive
//! because the Arrow buffer holds it, and nothing is copied or allocated
//! beyond a few reference counts.
//!
//! Compressed or re-encoded columns cost one pass each, and only for the
//! columns a query actually projects.

use arrow_array::types::Int32Type;
use arrow_array::{make_array, Array, ArrayRef, DictionaryArray, Int32Array, RunArray};
use arrow_buffer::{Buffer, NullBuffer};
use arrow_data::ArrayDataBuilder;
use arrow_schema::{DataType, Field};
use std::sync::Arc;

use crate::columnar::codec;
use crate::columnar::encode::fixed_width;
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

/// Fetch a buffer by role, or report what the chunk should have carried.
fn load_role(
    chunk: &ArchivedColumnChunk,
    role: BufferRole,
    source: &dyn BufferSource,
) -> Result<Buffer> {
    let spec = chunk.buffer(role).ok_or_else(|| {
        Error::corrupt(format!(
            "a {:?} chunk is missing its {role:?} buffer",
            chunk.encoding
        ))
    })?;
    load(spec, chunk.codec.to_native(), source)
}

/// Rebuild one column as an Arrow array of `data_type`.
pub fn decode_column(
    chunk: &ArchivedColumnChunk,
    data_type: &DataType,
    source: &dyn BufferSource,
) -> Result<ArrayRef> {
    match chunk.encoding.to_native() {
        Encoding::Plain => decode_plain(chunk, data_type, source),
        Encoding::Dictionary => decode_dictionary(chunk, data_type, source),
        Encoding::RunLength => decode_run_length(chunk, data_type, source),
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

/// Rebuild a plainly encoded column. This is the path that copies nothing.
fn decode_plain(
    chunk: &ArchivedColumnChunk,
    data_type: &DataType,
    source: &dyn BufferSource,
) -> Result<ArrayRef> {
    let len = chunk.len.to_native() as usize;
    let nulls = decode_nulls(chunk, len, source)?;

    let mut builder = ArrayDataBuilder::new(data_type.clone())
        .len(len)
        .nulls(nulls);

    match data_type {
        DataType::Null => {}

        DataType::Boolean => {
            let values = load_role(chunk, BufferRole::Values, source)?;
            check_len("boolean values", values.len(), len.div_ceil(8))?;
            builder = builder.add_buffer(values);
        }

        DataType::Utf8 | DataType::Binary => {
            let offsets = load_role(chunk, BufferRole::Offsets, source)?;
            check_len("offsets", offsets.len(), (len + 1) * 4)?;
            builder = builder.add_buffer(offsets).add_buffer(load_role(
                chunk,
                BufferRole::Values,
                source,
            )?);
        }
        DataType::LargeUtf8 | DataType::LargeBinary => {
            let offsets = load_role(chunk, BufferRole::Offsets, source)?;
            check_len("offsets", offsets.len(), (len + 1) * 8)?;
            builder = builder.add_buffer(offsets).add_buffer(load_role(
                chunk,
                BufferRole::Values,
                source,
            )?);
        }

        other => {
            let width = fixed_width(other).ok_or_else(|| {
                Error::Unsupported(format!("{other} columns are not supported yet"))
            })?;
            let values = load_role(chunk, BufferRole::Values, source)?;
            check_len("values", values.len(), len * width)?;
            builder = builder.add_buffer(values);
        }
    }

    // `build` validates offsets, utf8 and buffer sizes. The bytes came off
    // disk, so they are checked rather than trusted, even though every buffer
    // already passed its checksum.
    let data = builder
        .build()
        .map_err(|e| Error::corrupt(format!("a {data_type} column failed Arrow's checks: {e}")))?;
    Ok(make_array(data))
}

fn check_len(what: &'static str, found: usize, needed: usize) -> Result<()> {
    if found < needed {
        return Err(Error::corrupt(format!(
            "{what} buffer holds {found} bytes, {needed} needed"
        )));
    }
    Ok(())
}

/// Rebuild a dictionary-encoded column and expand it back to `data_type`.
fn decode_dictionary(
    chunk: &ArchivedColumnChunk,
    data_type: &DataType,
    source: &dyn BufferSource,
) -> Result<ArrayRef> {
    let len = chunk.len.to_native() as usize;
    let child = chunk
        .children
        .first()
        .ok_or_else(|| Error::corrupt("a dictionary chunk is missing its values"))?;
    let values = decode_column(child, data_type, source)?;

    let keys_buffer = load_role(chunk, BufferRole::DictKeys, source)?;
    check_len("dictionary keys", keys_buffer.len(), len * 4)?;
    let nulls = decode_nulls(chunk, len, source)?;

    let keys_data = ArrayDataBuilder::new(DataType::Int32)
        .len(len)
        .nulls(nulls)
        .add_buffer(keys_buffer)
        .build()
        .map_err(|e| Error::corrupt(format!("dictionary keys failed Arrow's checks: {e}")))?;
    let keys = Int32Array::from(keys_data);

    // `try_new` range-checks every key against the dictionary, so a damaged
    // index cannot read past the values array.
    let dict = DictionaryArray::<Int32Type>::try_new(keys, values)
        .map_err(|e| Error::corrupt(format!("dictionary chunk is inconsistent: {e}")))?;

    arrow_cast::cast(&dict, data_type)
        .map_err(|e| Error::corrupt(format!("dictionary chunk failed to expand: {e}")))
}

/// Rebuild a run-length-encoded column and expand it back to `data_type`.
fn decode_run_length(
    chunk: &ArchivedColumnChunk,
    data_type: &DataType,
    source: &dyn BufferSource,
) -> Result<ArrayRef> {
    let run_count = chunk.run_count.to_native() as usize;
    let child = chunk
        .children
        .first()
        .ok_or_else(|| Error::corrupt("a run-length chunk is missing its values"))?;
    let values = decode_column(child, data_type, source)?;

    let ends_buffer = load_role(chunk, BufferRole::RunEnds, source)?;
    check_len("run ends", ends_buffer.len(), run_count * 4)?;
    let ends_data = ArrayDataBuilder::new(DataType::Int32)
        .len(run_count)
        .add_buffer(ends_buffer)
        .build()
        .map_err(|e| Error::corrupt(format!("run ends failed Arrow's checks: {e}")))?;
    let run_ends = Int32Array::from(ends_data);

    // `try_new` checks that run ends rise and match the value count, so a
    // damaged chunk cannot produce an array that reads out of bounds.
    let runs = RunArray::<Int32Type>::try_new(&run_ends, values.as_ref())
        .map_err(|e| Error::corrupt(format!("run-length chunk is inconsistent: {e}")))?;

    let expected = chunk.len.to_native() as usize;
    if runs.len() != expected {
        return Err(Error::corrupt(format!(
            "run-length chunk covers {} rows, metadata says {expected}",
            runs.len()
        )));
    }

    arrow_cast::cast(&runs, data_type)
        .map_err(|e| Error::corrupt(format!("run-length chunk failed to expand: {e}")))
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
