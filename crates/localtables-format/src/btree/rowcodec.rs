//! One row, packed into bytes.
//!
//! The columnar table stores columns; a b-tree stores rows, because a point
//! lookup wants one row and not one column of every row. The layout is a null
//! bitmap, then the fixed-width fields, then the variable-width ones with their
//! lengths in front:
//!
//! ```text
//! [ null bitmap: ceil(columns / 8) bytes ]
//! [ fixed-width values, in schema order  ]
//! [ (u32 length, bytes) per varlen value ]
//! ```
//!
//! Nothing here is rkyv: a row is read one field at a time by a lookup that
//! already knows the schema, so a description of the layout would cost more
//! than it saves.

use arrow_array::builder::*;
use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, SchemaRef, TimeUnit};

use crate::{Error, Result};

/// Bytes a fixed-width type occupies in a row, or `None` when it is varlen.
fn fixed_width(data_type: &DataType) -> Option<usize> {
    Some(match data_type {
        DataType::Boolean | DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int32
        | DataType::UInt32
        | DataType::Float32
        | DataType::Date32
        | DataType::Time32(_) => 4,
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Date64
        | DataType::Time64(_)
        | DataType::Timestamp(_, _) => 8,
        _ => return None,
    })
}

/// True when a row of this schema can be packed.
pub fn is_encodable(schema: &SchemaRef) -> bool {
    schema.fields().iter().all(|field| {
        fixed_width(field.data_type()).is_some()
            || matches!(
                field.data_type(),
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary
            )
    })
}

/// Pack row `row` of `batch`.
pub fn encode(batch: &RecordBatch, row: usize) -> Result<Vec<u8>> {
    let schema = batch.schema();
    let mut out = vec![0u8; schema.fields().len().div_ceil(8)];

    // Fixed-width fields first, so a lookup reaching one of them can jump
    // straight to it rather than walking the varlen ones.
    for (index, column) in batch.columns().iter().enumerate() {
        if column.is_null(row) {
            out[index / 8] |= 1 << (index % 8);
        }
        if fixed_width(column.data_type()).is_some() {
            push_fixed(&mut out, column, row)?;
        }
    }
    for column in batch.columns() {
        if fixed_width(column.data_type()).is_none() {
            push_varlen(&mut out, column, row)?;
        }
    }
    Ok(out)
}

/// Append a fixed-width value, or its zeroed placeholder when null.
///
/// Null slots still take their space, so the fixed section has a constant
/// layout the reader can index into.
fn push_fixed(out: &mut Vec<u8>, column: &ArrayRef, row: usize) -> Result<()> {
    let width = fixed_width(column.data_type()).expect("caller checked");
    if column.is_null(row) {
        out.extend(std::iter::repeat_n(0u8, width));
        return Ok(());
    }

    macro_rules! primitive {
        ($ty:ty) => {
            out.extend_from_slice(&column.as_primitive::<$ty>().value(row).to_le_bytes())
        };
    }

    match column.data_type() {
        DataType::Boolean => out.push(u8::from(column.as_boolean().value(row))),
        DataType::Int8 => primitive!(Int8Type),
        DataType::Int16 => primitive!(Int16Type),
        DataType::Int32 => primitive!(Int32Type),
        DataType::Int64 => primitive!(Int64Type),
        DataType::UInt8 => primitive!(UInt8Type),
        DataType::UInt16 => primitive!(UInt16Type),
        DataType::UInt32 => primitive!(UInt32Type),
        DataType::UInt64 => primitive!(UInt64Type),
        DataType::Float32 => primitive!(Float32Type),
        DataType::Float64 => primitive!(Float64Type),
        DataType::Date32 => primitive!(Date32Type),
        DataType::Date64 => primitive!(Date64Type),
        DataType::Time32(TimeUnit::Second) => primitive!(Time32SecondType),
        DataType::Time32(TimeUnit::Millisecond) => primitive!(Time32MillisecondType),
        DataType::Time64(TimeUnit::Microsecond) => primitive!(Time64MicrosecondType),
        DataType::Time64(TimeUnit::Nanosecond) => primitive!(Time64NanosecondType),
        DataType::Timestamp(TimeUnit::Second, _) => primitive!(TimestampSecondType),
        DataType::Timestamp(TimeUnit::Millisecond, _) => primitive!(TimestampMillisecondType),
        DataType::Timestamp(TimeUnit::Microsecond, _) => primitive!(TimestampMicrosecondType),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => primitive!(TimestampNanosecondType),
        other => {
            return Err(Error::Unsupported(format!(
                "{other} cannot be stored in a b-tree row"
            )))
        }
    }
    Ok(())
}

/// Append a variable-width value, length first.
fn push_varlen(out: &mut Vec<u8>, column: &ArrayRef, row: usize) -> Result<()> {
    let bytes: &[u8] = if column.is_null(row) {
        &[]
    } else {
        match column.data_type() {
            DataType::Utf8 => column.as_string::<i32>().value(row).as_bytes(),
            DataType::LargeUtf8 => column.as_string::<i64>().value(row).as_bytes(),
            DataType::Binary => column.as_binary::<i32>().value(row),
            DataType::LargeBinary => column.as_binary::<i64>().value(row),
            other => {
                return Err(Error::Unsupported(format!(
                    "{other} cannot be stored in a b-tree row"
                )))
            }
        }
    };
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

/// Reads packed rows back into Arrow arrays, one row at a time.
///
/// A lookup builds one of these, pushes the rows it found, and finishes with a
/// batch. Rebuilding column by column is what makes a point lookup cheap: the
/// row bytes are read once and never held.
pub struct RowDecoder {
    schema: SchemaRef,
    builders: Vec<Box<dyn ArrayBuilder>>,
    rows: usize,
}

impl RowDecoder {
    pub fn new(schema: SchemaRef) -> Result<Self> {
        if !is_encodable(&schema) {
            return Err(Error::Unsupported(
                "this schema holds a type a b-tree row cannot store".into(),
            ));
        }
        let builders = schema
            .fields()
            .iter()
            .map(|field| arrow_array::builder::make_builder(field.data_type(), 16))
            .collect();
        Ok(Self {
            schema,
            builders,
            rows: 0,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Unpack one row and append it.
    pub fn push(&mut self, bytes: &[u8]) -> Result<()> {
        let fields = self.schema.fields().len();
        let bitmap_len = fields.div_ceil(8);
        if bytes.len() < bitmap_len {
            return Err(Error::corrupt(format!(
                "a row of {fields} columns needs {bitmap_len} bitmap bytes, found {}",
                bytes.len()
            )));
        }
        let (bitmap, mut rest) = bytes.split_at(bitmap_len);
        let is_null = |index: usize| bitmap[index / 8] & (1 << (index % 8)) != 0;

        // Fixed-width fields, in schema order, then the varlen ones.
        for (index, field) in self.schema.fields().iter().enumerate() {
            let Some(width) = fixed_width(field.data_type()) else {
                continue;
            };
            if rest.len() < width {
                return Err(Error::corrupt(format!(
                    "row ran out of bytes reading column {index}"
                )));
            }
            let (value, tail) = rest.split_at(width);
            rest = tail;
            take_fixed(
                self.builders[index].as_mut(),
                field.data_type(),
                value,
                is_null(index),
            )?;
        }

        for (index, field) in self.schema.fields().iter().enumerate() {
            if fixed_width(field.data_type()).is_some() {
                continue;
            }
            if rest.len() < 4 {
                return Err(Error::corrupt(format!(
                    "row ran out of bytes reading the length of column {index}"
                )));
            }
            let (len, tail) = rest.split_at(4);
            let len = u32::from_le_bytes(len.try_into().unwrap()) as usize;
            if tail.len() < len {
                return Err(Error::corrupt(format!(
                    "column {index} claims {len} bytes, {} remain",
                    tail.len()
                )));
            }
            let (value, tail) = tail.split_at(len);
            rest = tail;
            take_varlen(
                self.builders[index].as_mut(),
                field.data_type(),
                value,
                is_null(index),
            )?;
        }

        self.rows += 1;
        Ok(())
    }

    /// Finish the rows pushed so far as one batch.
    pub fn finish(mut self) -> Result<RecordBatch> {
        let columns: Vec<ArrayRef> = self
            .builders
            .iter_mut()
            .map(|builder| builder.finish())
            .collect();
        if columns.is_empty() {
            let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(self.rows));
            return RecordBatch::try_new_with_options(self.schema, columns, &options)
                .map_err(Error::from);
        }
        RecordBatch::try_new(self.schema, columns).map_err(Error::from)
    }
}

/// Append one fixed-width value to its builder.
fn take_fixed(
    builder: &mut dyn ArrayBuilder,
    data_type: &DataType,
    bytes: &[u8],
    null: bool,
) -> Result<()> {
    macro_rules! primitive {
        ($ty:ty, $native:ty) => {{
            let builder = downcast::<PrimitiveBuilder<$ty>>(builder, data_type)?;
            if null {
                builder.append_null();
            } else {
                let raw: [u8; std::mem::size_of::<$native>()] = bytes.try_into().map_err(|_| {
                    Error::corrupt(format!("a {data_type} value is the wrong size"))
                })?;
                builder.append_value(<$native>::from_le_bytes(raw));
            }
        }};
    }

    match data_type {
        DataType::Boolean => {
            let builder = downcast::<BooleanBuilder>(builder, data_type)?;
            if null {
                builder.append_null();
            } else {
                builder.append_value(bytes[0] != 0);
            }
        }
        DataType::Int8 => primitive!(Int8Type, i8),
        DataType::Int16 => primitive!(Int16Type, i16),
        DataType::Int32 => primitive!(Int32Type, i32),
        DataType::Int64 => primitive!(Int64Type, i64),
        DataType::UInt8 => primitive!(UInt8Type, u8),
        DataType::UInt16 => primitive!(UInt16Type, u16),
        DataType::UInt32 => primitive!(UInt32Type, u32),
        DataType::UInt64 => primitive!(UInt64Type, u64),
        DataType::Float32 => primitive!(Float32Type, f32),
        DataType::Float64 => primitive!(Float64Type, f64),
        DataType::Date32 => primitive!(Date32Type, i32),
        DataType::Date64 => primitive!(Date64Type, i64),
        DataType::Time32(TimeUnit::Second) => primitive!(Time32SecondType, i32),
        DataType::Time32(TimeUnit::Millisecond) => primitive!(Time32MillisecondType, i32),
        DataType::Time64(TimeUnit::Microsecond) => primitive!(Time64MicrosecondType, i64),
        DataType::Time64(TimeUnit::Nanosecond) => primitive!(Time64NanosecondType, i64),
        DataType::Timestamp(TimeUnit::Second, _) => primitive!(TimestampSecondType, i64),
        DataType::Timestamp(TimeUnit::Millisecond, _) => primitive!(TimestampMillisecondType, i64),
        DataType::Timestamp(TimeUnit::Microsecond, _) => primitive!(TimestampMicrosecondType, i64),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => primitive!(TimestampNanosecondType, i64),
        other => {
            return Err(Error::Unsupported(format!(
                "{other} cannot be read from a b-tree row"
            )))
        }
    }
    Ok(())
}

/// Append one variable-width value to its builder.
fn take_varlen(
    builder: &mut dyn ArrayBuilder,
    data_type: &DataType,
    bytes: &[u8],
    null: bool,
) -> Result<()> {
    macro_rules! string {
        ($offset:ty) => {{
            let builder = downcast::<GenericStringBuilder<$offset>>(builder, data_type)?;
            if null {
                builder.append_null();
            } else {
                let text = std::str::from_utf8(bytes)
                    .map_err(|e| Error::corrupt(format!("a stored string is not utf8: {e}")))?;
                builder.append_value(text);
            }
        }};
    }
    macro_rules! binary {
        ($offset:ty) => {{
            let builder = downcast::<GenericBinaryBuilder<$offset>>(builder, data_type)?;
            if null {
                builder.append_null();
            } else {
                builder.append_value(bytes);
            }
        }};
    }

    match data_type {
        DataType::Utf8 => string!(i32),
        DataType::LargeUtf8 => string!(i64),
        DataType::Binary => binary!(i32),
        DataType::LargeBinary => binary!(i64),
        other => {
            return Err(Error::Unsupported(format!(
                "{other} cannot be read from a b-tree row"
            )))
        }
    }
    Ok(())
}

fn downcast<'a, T: ArrayBuilder>(
    builder: &'a mut dyn ArrayBuilder,
    data_type: &DataType,
) -> Result<&'a mut T> {
    builder
        .as_any_mut()
        .downcast_mut::<T>()
        .ok_or_else(|| Error::Corrupt(format!("the builder for {data_type} has the wrong type")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        BinaryArray, BooleanArray, Float64Array, Int32Array, Int64Array, StringArray,
    };
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;

    fn wide_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i32", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("b", DataType::Boolean, true),
            Field::new("f", DataType::Float64, true),
            Field::new("bin", DataType::Binary, true),
            Field::new("i64", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![Some(1), None, Some(-3)])),
                Arc::new(StringArray::from(vec![Some("alpha"), Some(""), None])),
                Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])),
                Arc::new(Float64Array::from(vec![Some(1.5), Some(-0.25), None])),
                Arc::new(BinaryArray::from(vec![
                    Some(&b"\x00\x01"[..]),
                    None,
                    Some(&b""[..]),
                ])),
                Arc::new(Int64Array::from(vec![i64::MIN, 0, i64::MAX])),
            ],
        )
        .unwrap()
    }

    /// Pack every row and unpack them back into a batch.
    fn round_trip(batch: &RecordBatch) -> RecordBatch {
        let mut decoder = RowDecoder::new(batch.schema()).unwrap();
        for row in 0..batch.num_rows() {
            decoder.push(&encode(batch, row).unwrap()).unwrap();
        }
        decoder.finish().unwrap()
    }

    #[test]
    fn rows_round_trip() {
        let batch = wide_batch();
        assert_eq!(round_trip(&batch), batch);
    }

    #[test]
    fn a_single_row_round_trips() {
        let batch = wide_batch();
        let mut decoder = RowDecoder::new(batch.schema()).unwrap();
        decoder.push(&encode(&batch, 1).unwrap()).unwrap();
        assert_eq!(decoder.finish().unwrap(), batch.slice(1, 1));
    }

    #[test]
    fn no_rows_gives_an_empty_batch() {
        let decoder = RowDecoder::new(wide_batch().schema()).unwrap();
        assert!(decoder.is_empty());
        assert_eq!(decoder.finish().unwrap().num_rows(), 0);
    }

    #[test]
    fn a_null_fixed_field_takes_the_same_space_as_a_value() {
        // Only fixed-width columns, so the row length depends on nothing else.
        // A null must still occupy its slot, or the layout would shift and a
        // reader could not index into it.
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Float64, true),
            Field::new("c", DataType::Boolean, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![Some(1), None])),
                Arc::new(Float64Array::from(vec![Some(2.5), None])),
                Arc::new(BooleanArray::from(vec![Some(true), None])),
            ],
        )
        .unwrap();

        assert_eq!(
            encode(&batch, 0).unwrap().len(),
            encode(&batch, 1).unwrap().len()
        );
        assert_eq!(round_trip(&batch), batch);
    }

    #[test]
    fn a_truncated_row_is_reported_not_decoded() {
        let batch = wide_batch();
        let bytes = encode(&batch, 0).unwrap();
        let mut decoder = RowDecoder::new(batch.schema()).unwrap();

        let err = decoder.push(&bytes[..bytes.len() / 2]).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)), "got {err:?}");
    }

    #[test]
    fn an_empty_row_is_reported() {
        let batch = wide_batch();
        let mut decoder = RowDecoder::new(batch.schema()).unwrap();
        assert!(decoder.push(&[]).is_err());
    }

    #[test]
    fn a_length_that_runs_past_the_row_is_reported() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let mut decoder = RowDecoder::new(schema).unwrap();

        // One bitmap byte, then a length claiming far more than follows.
        let mut bytes = vec![0u8];
        bytes.extend_from_slice(&1000u32.to_le_bytes());
        bytes.extend_from_slice(b"short");

        let err = decoder.push(&bytes).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)), "got {err:?}");
    }

    #[test]
    fn invalid_utf8_is_reported_rather_than_handed_to_arrow() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let mut decoder = RowDecoder::new(schema).unwrap();

        let mut bytes = vec![0u8];
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&[0xff, 0xfe]);

        let err = decoder.push(&bytes).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)), "got {err:?}");
    }

    #[test]
    fn an_unsupported_schema_is_refused_up_front() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "list",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        )]));
        assert!(!is_encodable(&schema));
        assert!(RowDecoder::new(schema).is_err());
    }

    #[test]
    fn many_rows_round_trip_in_order() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let ids: Vec<i64> = (0..500).collect();
        let names: Vec<Option<String>> = ids
            .iter()
            .map(|i| (i % 3 != 0).then(|| format!("name-{i}")))
            .collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap();

        assert_eq!(round_trip(&batch), batch);
    }
}
