//! Random columns must survive the segment round trip untouched.
//!
//! The hand-written round-trip tests cover the cases worth naming. These cover
//! the ones nobody thought to name: odd row counts that fall between bitmap
//! bytes, columns that are all null or none null, strings that are empty or
//! long enough to force a truncated zone map, and the encoding choices those
//! shapes provoke.

use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    UInt8Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use proptest::prelude::*;

use localtables_format::columnar::segment::{build_segment, SegmentReader};
use localtables_format::config::{Compression, Durability, IoBackend, TableOptions};
use localtables_format::io::open_backend;
use localtables_format::layout::{schema as schema_codec, SEGMENT_ALIGN};

/// A column of one type, as a name, a field and the values.
#[derive(Debug, Clone)]
enum Column {
    Bool(Vec<Option<bool>>),
    U8(Vec<Option<u8>>),
    I32(Vec<Option<i32>>),
    I64(Vec<Option<i64>>),
    F64(Vec<Option<f64>>),
    Utf8(Vec<Option<String>>),
}

impl Column {
    fn len(&self) -> usize {
        match self {
            Column::Bool(v) => v.len(),
            Column::U8(v) => v.len(),
            Column::I32(v) => v.len(),
            Column::I64(v) => v.len(),
            Column::F64(v) => v.len(),
            Column::Utf8(v) => v.len(),
        }
    }

    fn field(&self, name: &str) -> Field {
        let data_type = match self {
            Column::Bool(_) => DataType::Boolean,
            Column::U8(_) => DataType::UInt8,
            Column::I32(_) => DataType::Int32,
            Column::I64(_) => DataType::Int64,
            Column::F64(_) => DataType::Float64,
            Column::Utf8(_) => DataType::Utf8,
        };
        Field::new(name, data_type, true)
    }

    fn array(&self) -> ArrayRef {
        match self {
            Column::Bool(v) => Arc::new(BooleanArray::from(v.clone())),
            Column::U8(v) => Arc::new(UInt8Array::from(v.clone())),
            Column::I32(v) => Arc::new(Int32Array::from(v.clone())),
            Column::I64(v) => Arc::new(Int64Array::from(v.clone())),
            Column::F64(v) => Arc::new(Float64Array::from(v.clone())),
            Column::Utf8(v) => Arc::new(StringArray::from(v.clone())),
        }
    }
}

/// Columns of every supported shape.
///
/// Some variants draw from a small pool so values repeat, which is what pushes
/// the encoder onto the dictionary and run-length paths; others draw freely so
/// it stays on the plain one.
fn column_strategy(rows: usize) -> impl Strategy<Value = Column> {
    prop_oneof![
        prop::collection::vec(prop::option::of(any::<bool>()), rows).prop_map(Column::Bool),
        prop::collection::vec(prop::option::of(0u8..6), rows).prop_map(Column::U8),
        prop::collection::vec(prop::option::of(any::<i32>()), rows).prop_map(Column::I32),
        prop::collection::vec(prop::option::of(-4i64..4), rows).prop_map(Column::I64),
        prop::collection::vec(prop::option::of(any::<i64>()), rows).prop_map(Column::I64),
        // Floats without NaN: NaN never equals itself, so batch equality would
        // fail for reasons that have nothing to do with storage.
        prop::collection::vec(prop::option::of(-1e6f64..1e6), rows).prop_map(Column::F64),
        prop::collection::vec(
            prop::option::of(prop::sample::select(vec![
                String::new(),
                "a".to_string(),
                "beta".to_string(),
                "\u{1f600} unicode".to_string(),
                "x".repeat(100),
            ])),
            rows
        )
        .prop_map(Column::Utf8),
        // Long enough to force a truncated zone map bound.
        prop::collection::vec(prop::option::of("[a-z\u{00e9}\u{1f600}]{0,90}"), rows)
            .prop_map(Column::Utf8),
    ]
}

fn case_strategy() -> impl Strategy<Value = (usize, Vec<Column>)> {
    // Row counts around the bitmap byte boundary catch off-by-one bit handling.
    prop::sample::select(vec![0usize, 1, 7, 8, 9, 63, 64, 65, 300]).prop_flat_map(|rows| {
        prop::collection::vec(column_strategy(rows), 1..4).prop_map(move |cols| (rows, cols))
    })
}

fn options(compression: Compression, encodings: bool) -> TableOptions {
    TableOptions {
        compression,
        dictionary_encoding: encodings,
        rle_encoding: encodings,
        durability: Durability::None,
        io_backend: IoBackend::Mmap,
        ..TableOptions::default()
    }
}

/// Write the batch as a segment in a real file and read it back.
fn round_trip(schema: &SchemaRef, batch: &RecordBatch, opts: &TableOptions) -> RecordBatch {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("segment.lt");
        let io = open_backend(&path, opts.io_backend, opts.durability, false).unwrap();

        let layout = schema_codec::SchemaLayout::of(schema);
        let built =
            build_segment(0, schema, layout.current(), std::slice::from_ref(batch), opts).unwrap();

        io.set_len(SEGMENT_ALIGN).await.unwrap();
        let offset = io.append(&[&built.bytes]).await.unwrap();
        let (data, meta) = built.placed(offset);

        let bytes = io.read_immutable(data).await.unwrap();
        let reader = SegmentReader::new(bytes, offset, meta, schema.clone(), &layout).unwrap();
        reader.read(None).unwrap()
    })
}

fn build_batch(rows: usize, columns: &[Column]) -> (SchemaRef, RecordBatch) {
    let fields: Vec<Field> = columns
        .iter()
        .enumerate()
        .map(|(i, c)| c.field(&format!("c{i}")))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let arrays: Vec<ArrayRef> = columns.iter().map(|c| c.array()).collect();

    let batch = if arrays.is_empty() || rows == 0 {
        let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(rows));
        RecordBatch::try_new_with_options(schema.clone(), arrays, &options).unwrap()
    } else {
        RecordBatch::try_new(schema.clone(), arrays).unwrap()
    };
    (schema, batch)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

    /// Plain, uncompressed storage must be byte-exact.
    #[test]
    fn plain_storage_round_trips((rows, columns) in case_strategy()) {
        prop_assume!(columns.iter().all(|c| c.len() == rows));
        let (schema, batch) = build_batch(rows, &columns);
        let read = round_trip(&schema, &batch, &options(Compression::None, false));
        prop_assert_eq!(read, batch);
    }

    /// Choosing an encoding must never change what comes back.
    #[test]
    fn re_encoded_storage_round_trips((rows, columns) in case_strategy()) {
        prop_assume!(columns.iter().all(|c| c.len() == rows));
        let (schema, batch) = build_batch(rows, &columns);
        let read = round_trip(&schema, &batch, &options(Compression::None, true));
        prop_assert_eq!(read, batch);
    }

    /// Nor must compressing it.
    #[test]
    fn compressed_storage_round_trips((rows, columns) in case_strategy()) {
        prop_assume!(columns.iter().all(|c| c.len() == rows));
        let (schema, batch) = build_batch(rows, &columns);
        let read = round_trip(&schema, &batch, &options(Compression::Lz4, true));
        prop_assert_eq!(read, batch);
    }

    /// Reading a subset of columns must give the same arrays as reading all of
    /// them and picking the same ones.
    #[test]
    fn projection_matches_a_full_read((rows, columns) in case_strategy()) {
        prop_assume!(columns.iter().all(|c| c.len() == rows));
        let (schema, batch) = build_batch(rows, &columns);
        let opts = options(Compression::Lz4, true);

        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        runtime.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("segment.lt");
            let io = open_backend(&path, opts.io_backend, opts.durability, false).unwrap();
            let layout = schema_codec::SchemaLayout::of(&schema);
            let built = build_segment(0, &schema, layout.current(), std::slice::from_ref(&batch), &opts).unwrap();
            io.set_len(SEGMENT_ALIGN).await.unwrap();
            let offset = io.append(&[&built.bytes]).await.unwrap();
            let (data, meta) = built.placed(offset);
            let bytes = io.read_immutable(data).await.unwrap();
            let reader = SegmentReader::new(bytes, offset, meta, schema.clone(), &layout).unwrap();

            for index in 0..columns.len() {
                let projected = reader.read(Some(&[index])).unwrap();
                assert_eq!(projected.num_columns(), 1);
                assert_eq!(projected.num_rows(), rows);
                assert_eq!(projected.column(0), batch.column(index), "column {index}");
            }
        });
    }
}

// Zone map bounds must actually bound the data. A bound that excludes a value
// present in the column would let a scan skip rows that match.
proptest! {
    #![proptest_config(ProptestConfig { cases: 96, ..ProptestConfig::default() })]

    #[test]
    fn zone_bounds_contain_every_value(
        values in prop::collection::vec(prop::option::of(any::<i64>()), 1..200)
    ) {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
        let array = Int64Array::from(values.clone());
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();

        let opts = options(Compression::None, false);
        let layout = schema_codec::SchemaLayout::of(&schema);
        let built = build_segment(0, &schema, layout.current(), &[batch], &opts).unwrap();
        let zone = built.meta.columns[0].zone.clone();

        let present: Vec<i64> = values.iter().flatten().copied().collect();
        prop_assert_eq!(zone.null_count as usize, values.len() - present.len());

        if present.is_empty() {
            prop_assert!(zone.is_unknown(), "no values means no bounds");
        } else {
            let min = zone.min_array(&DataType::Int64).unwrap();
            let max = zone.max_array(&DataType::Int64).unwrap();
            let min = min.as_any().downcast_ref::<Int64Array>().unwrap().value(0);
            let max = max.as_any().downcast_ref::<Int64Array>().unwrap().value(0);

            for value in present {
                prop_assert!(value >= min, "{value} is below the stated minimum {min}");
                prop_assert!(value <= max, "{value} is above the stated maximum {max}");
            }
        }
    }

    /// Truncated string bounds must stay sound: the stored minimum never
    /// exceeds a real value, and the stored maximum is never below one.
    #[test]
    fn string_zone_bounds_stay_sound(
        values in prop::collection::vec("[a-z]{0,120}", 1..60)
    ) {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let array = StringArray::from(values.clone());
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();

        let opts = options(Compression::None, false);
        let layout = schema_codec::SchemaLayout::of(&schema);
        let built = build_segment(0, &schema, layout.current(), &[batch], &opts).unwrap();
        let zone = built.meta.columns[0].zone.clone();

        if let Some(min) = zone.min_array(&DataType::Utf8) {
            let min = min.as_any().downcast_ref::<StringArray>().unwrap().value(0).to_string();
            for value in &values {
                prop_assert!(
                    value.as_bytes() >= min.as_bytes(),
                    "{value:?} is below the stated minimum {min:?}"
                );
            }
        }
        if let Some(max) = zone.max_array(&DataType::Utf8) {
            let max = max.as_any().downcast_ref::<StringArray>().unwrap().value(0).to_string();
            for value in &values {
                prop_assert!(
                    value.as_bytes() <= max.as_bytes(),
                    "{value:?} is above the stated maximum {max:?}"
                );
            }
        }
    }
}
