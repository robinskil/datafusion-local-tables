//! Serializing a `RecordBatch` without Arrow IPC.
//!
//! The write-ahead log stores batches, and so could Arrow's IPC format. It is
//! not used, for the same reason segments do not use it: an IPC message has to
//! be decoded as a whole before any one column can be looked at. Here each
//! column is a separate archived record holding its own buffers, so replay can
//! decode one column, or skip one, without touching the rest.
//!
//! The buffers themselves are raw Arrow bytes, exactly as a segment stores
//! them. rkyv describes where they are; it never wraps the values.

use arrow_array::{make_array, Array, ArrayRef, RecordBatch};
use arrow_buffer::{Buffer, NullBuffer};
use arrow_data::{ArrayData, ArrayDataBuilder};
use arrow_schema::{DataType, SchemaRef};
use rkyv::{Archive, Deserialize, Serialize};

use crate::io::buf::{IoBuf, SharedBuf};
use crate::{Error, Result};

/// One column, mirroring Arrow's own array layout.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[rkyv(derive(Debug))]
#[rkyv(serialize_bounds(
    __S: rkyv::ser::Writer + rkyv::ser::Allocator,
    __S::Error: rkyv::rancor::Source,
))]
#[rkyv(deserialize_bounds(__D::Error: rkyv::rancor::Source))]
#[rkyv(bytecheck(bounds(
    __C: rkyv::validation::ArchiveContext,
    __C::Error: rkyv::rancor::Source,
)))]
pub struct ColumnData {
    pub len: u64,
    pub null_count: u64,
    /// Null bitmap, starting at bit zero. Absent when nothing is null.
    pub validity: Option<Vec<u8>>,
    /// Arrow's own buffers for this array, in Arrow's own order.
    pub buffers: Vec<Vec<u8>>,
    /// Child arrays, for the nested types.
    #[rkyv(omit_bounds)]
    pub children: Vec<ColumnData>,
}

/// A batch, column by column.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[rkyv(derive(Debug))]
pub struct BatchData {
    pub row_count: u64,
    pub columns: Vec<ColumnData>,
}

/// Serialize a batch.
///
/// The array's own buffers are copied once into the record. There is no way
/// around one copy here: the bytes have to end up contiguous in the log.
pub fn encode(batch: &RecordBatch) -> BatchData {
    BatchData {
        row_count: batch.num_rows() as u64,
        columns: batch.columns().iter().map(encode_array).collect(),
    }
}

fn encode_array(array: &ArrayRef) -> ColumnData {
    // A sliced array keeps its parent's buffers and an offset into them.
    // Compacting first means the record holds this array's rows starting at
    // index zero, so the decoder needs no offset, and a small slice of a large
    // batch does not log the whole parent.
    let compacted;
    let array = if array.offset() == 0 {
        array
    } else {
        compacted = compact(array);
        &compacted
    };

    let data = array.to_data();
    debug_assert_eq!(data.offset(), 0, "a compacted array starts at index zero");

    ColumnData {
        len: array.len() as u64,
        null_count: array.null_count() as u64,
        validity: array
            .nulls()
            .filter(|n| n.null_count() > 0)
            .map(|n| n.inner().sliced().as_slice().to_vec()),
        buffers: data
            .buffers()
            .iter()
            .map(|b| b.as_slice().to_vec())
            .collect(),
        children: data
            .child_data()
            .iter()
            .map(|c| encode_array(&make_array(c.clone())))
            .collect(),
    }
}

/// Rebuild a batch from an archived record.
///
/// Only the columns in `projection` are decoded; the rest are not touched. A
/// `None` projection decodes every column.
pub fn decode(
    archived: &ArchivedBatchData,
    schema: &SchemaRef,
    projection: Option<&[usize]>,
) -> Result<RecordBatch> {
    let row_count = archived.row_count.to_native() as usize;
    if archived.columns.len() != schema.fields().len() {
        return Err(Error::corrupt(format!(
            "logged batch holds {} columns, the schema has {}",
            archived.columns.len(),
            schema.fields().len()
        )));
    }

    let indices: Vec<usize> = match projection {
        Some(indices) => indices.to_vec(),
        None => (0..schema.fields().len()).collect(),
    };

    let mut fields = Vec::with_capacity(indices.len());
    let mut columns = Vec::with_capacity(indices.len());
    for index in indices {
        let field = schema.fields().get(index).ok_or_else(|| {
            Error::InvalidArgument(format!("projected column {index} is out of range"))
        })?;
        fields.push(field.clone());
        columns.push(decode_array(
            &archived.columns[index],
            field.data_type(),
            row_count,
        )?);
    }

    let projected = std::sync::Arc::new(arrow_schema::Schema::new(fields));
    if columns.is_empty() {
        let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(row_count));
        return RecordBatch::try_new_with_options(projected, columns, &options)
            .map_err(Error::from);
    }
    RecordBatch::try_new(projected, columns).map_err(Error::from)
}

fn decode_array(
    archived: &ArchivedColumnData,
    data_type: &DataType,
    expected_rows: usize,
) -> Result<ArrayRef> {
    let len = archived.len.to_native() as usize;
    if len != expected_rows {
        return Err(Error::corrupt(format!(
            "a logged column holds {len} rows, the batch holds {expected_rows}"
        )));
    }
    Ok(make_array(build_data(archived, data_type, len)?))
}

fn build_data(
    archived: &ArchivedColumnData,
    data_type: &DataType,
    len: usize,
) -> Result<ArrayData> {
    let nulls = match archived.validity.as_ref() {
        None => None,
        Some(bytes) => {
            let required = len.div_ceil(8);
            if bytes.len() < required {
                return Err(Error::corrupt(format!(
                    "a logged {len}-row column needs {required} bitmap bytes, found {}",
                    bytes.len()
                )));
            }
            Some(NullBuffer::new(arrow_buffer::BooleanBuffer::new(
                aligned(bytes),
                0,
                len,
            )))
        }
    };

    let mut builder = ArrayDataBuilder::new(data_type.clone())
        .len(len)
        .nulls(nulls);
    for buffer in archived.buffers.iter() {
        builder = builder.add_buffer(aligned(buffer));
    }

    let child_types = child_data_types(data_type);
    if archived.children.len() != child_types.len() {
        return Err(Error::corrupt(format!(
            "a logged {data_type} column holds {} children, the type has {}",
            archived.children.len(),
            child_types.len()
        )));
    }
    for (child, child_type) in archived.children.iter().zip(child_types) {
        let child_len = child.len.to_native() as usize;
        builder = builder.add_child_data(build_data(child, &child_type, child_len)?);
    }

    builder.build().map_err(|e| {
        Error::corrupt(format!(
            "a logged {data_type} column failed Arrow's checks: {e}"
        ))
    })
}

/// Copy an array into fresh buffers holding only its own rows.
///
/// `concat` of a single array short-circuits to a slice, which is exactly what
/// needs undoing here, so an empty array of the same type goes in front to take
/// the general path. It works for every Arrow type, including nested ones.
fn compact(array: &ArrayRef) -> ArrayRef {
    let empty = arrow_array::new_empty_array(array.data_type());
    arrow_select::concat::concat(&[empty.as_ref(), array.as_ref()])
        .expect("concat of two arrays of the same type cannot fail")
}

/// Copy archived bytes into an aligned Arrow buffer.
///
/// The bytes inside an rkyv archive are only byte-aligned, and Arrow wants 64,
/// so this is the one copy replay cannot avoid. It is bounded by the size of
/// the log, which a flush empties.
fn aligned(bytes: &[u8]) -> Buffer {
    SharedBuf::from_owned(IoBuf::copy_from(bytes)).into_arrow_buffer()
}

/// The types of a data type's child arrays, in Arrow's own order.
fn child_data_types(data_type: &DataType) -> Vec<DataType> {
    match data_type {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        BooleanArray, Float64Array, Int32Array, Int64Array, StringArray, StructArray,
    };
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;

    fn round_trip(batch: &RecordBatch) -> RecordBatch {
        let encoded = encode(batch);
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&encoded).unwrap();
        let archived = rkyv::access::<ArchivedBatchData, rkyv::rancor::Error>(&bytes).unwrap();
        decode(archived, &batch.schema(), None).unwrap()
    }

    fn wide_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i32", DataType::Int32, true),
            Field::new("i64", DataType::Int64, false),
            Field::new("f64", DataType::Float64, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("b", DataType::Boolean, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![Some(1), None, Some(-3)])),
                Arc::new(Int64Array::from(vec![i64::MIN, 0, i64::MAX])),
                Arc::new(Float64Array::from(vec![Some(1.5), Some(-0.25), None])),
                Arc::new(StringArray::from(vec![Some("alpha"), None, Some("")])),
                Arc::new(BooleanArray::from(vec![Some(true), Some(false), None])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn a_batch_round_trips() {
        let batch = wide_batch();
        assert_eq!(round_trip(&batch), batch);
    }

    #[test]
    fn an_empty_batch_round_trips() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(Vec::<Option<i32>>::new()))],
        )
        .unwrap();
        assert_eq!(round_trip(&batch), batch);
    }

    #[test]
    fn nested_columns_round_trip() {
        let inner = Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])) as ArrayRef;
        let names = Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef;
        let struct_array = StructArray::from(vec![
            (Arc::new(Field::new("n", DataType::Int32, true)), inner),
            (Arc::new(Field::new("s", DataType::Utf8, false)), names),
        ]);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "st",
            struct_array.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(struct_array)]).unwrap();

        assert_eq!(round_trip(&batch), batch);
    }

    #[test]
    fn a_projection_decodes_only_what_it_names() {
        let batch = wide_batch();
        let encoded = encode(&batch);
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&encoded).unwrap();
        let archived = rkyv::access::<ArchivedBatchData, rkyv::rancor::Error>(&bytes).unwrap();

        let projected = decode(archived, &batch.schema(), Some(&[3, 1])).unwrap();
        assert_eq!(projected.num_columns(), 2);
        assert_eq!(projected.schema().field(0).name(), "s");
        assert_eq!(projected.column(0), batch.column(3));
        assert_eq!(projected.column(1), batch.column(1));
    }

    #[test]
    fn an_empty_projection_keeps_the_row_count() {
        let batch = wide_batch();
        let encoded = encode(&batch);
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&encoded).unwrap();
        let archived = rkyv::access::<ArchivedBatchData, rkyv::rancor::Error>(&bytes).unwrap();

        let projected = decode(archived, &batch.schema(), Some(&[])).unwrap();
        assert_eq!(projected.num_columns(), 0);
        assert_eq!(projected.num_rows(), 3);
    }

    #[test]
    fn decoded_buffers_are_aligned_for_arrow() {
        let batch = round_trip(&wide_batch());
        for column in batch.columns() {
            for buffer in column.to_data().buffers() {
                assert_eq!(
                    buffer.as_ptr() as usize % crate::layout::BUFFER_ALIGN as usize,
                    0
                );
            }
        }
    }

    #[test]
    fn a_column_count_mismatch_is_reported() {
        let batch = wide_batch();
        let encoded = encode(&batch);
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&encoded).unwrap();
        let archived = rkyv::access::<ArchivedBatchData, rkyv::rancor::Error>(&bytes).unwrap();

        let narrower = Arc::new(Schema::new(vec![Field::new("i32", DataType::Int32, true)]));
        let err = decode(archived, &narrower, None).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)), "got {err:?}");
    }

    #[test]
    fn a_sliced_batch_round_trips() {
        let batch = wide_batch().slice(1, 2);
        // Arrow keeps the parent buffers and an offset behind a slice; the
        // decoded batch must still hold exactly the sliced rows.
        let decoded = round_trip(&batch);
        assert_eq!(decoded.num_rows(), 2);
        assert_eq!(decoded, batch);
    }
}
