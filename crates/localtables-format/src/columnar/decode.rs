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
    match chunk.encoding.to_native() {
        // Stored as the column's own type. Nothing to undo.
        Encoding::Plain => Ok(make_array(rebuild(chunk, data_type, source)?)),

        // Stored as a dictionary over the column's type, then cast back. The
        // cast is the price of the smaller file; a column declared as a
        // dictionary in the schema takes the Plain path instead and pays
        // nothing.
        Encoding::Dictionary => {
            let stored =
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(data_type.clone()));
            let array = make_array(rebuild(chunk, &stored, source)?);
            arrow_cast::cast(&array, data_type)
                .map_err(|e| Error::corrupt(format!("a dictionary chunk failed to expand: {e}")))
        }

        Encoding::RunLength => {
            let (run_ends, values) = run_end_fields(data_type);
            let stored = DataType::RunEndEncoded(run_ends, values);
            let array = make_array(rebuild(chunk, &stored, source)?);
            arrow_cast::cast(&array, data_type)
                .map_err(|e| Error::corrupt(format!("a run-length chunk failed to expand: {e}")))
        }
    }
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
