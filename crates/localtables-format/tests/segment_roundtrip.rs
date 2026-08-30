//! Segments must give back exactly what was written, for every supported type,
//! every encoding, and every codec.
//!
//! These run against a real file rather than an in-memory buffer, so they also
//! cover the alignment rules the zero-copy read path depends on.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Float32Array, Float64Array,
    Int16Array, Int32Array, Int64Array, Int8Array, LargeBinaryArray, LargeStringArray, NullArray,
    RecordBatch, StringArray, TimestampMicrosecondArray, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};

use localtables_format::columnar::page::{Codec, Encoding};
use localtables_format::columnar::segment::{build_segment, SegmentReader};
use localtables_format::config::{Compression, Durability, IoBackend, TableOptions};
use localtables_format::io::open_backend;
use localtables_format::layout::{schema as schema_codec, Extent, BUFFER_ALIGN, SEGMENT_ALIGN};

/// Write a segment to a real file at a page-aligned offset and read it back.
///
/// Returns the reader plus whether the read path was zero-copy.
async fn round_trip(
    schema: &SchemaRef,
    batches: &[RecordBatch],
    options: &TableOptions,
) -> (tempfile::TempDir, SegmentReader) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("segment.lt");
    let io = open_backend(&path, options.io_backend, Durability::None, false).unwrap();

    let layout = schema_codec::SchemaLayout::of(schema);
    let built = build_segment(0, schema, layout.current(), batches, options).unwrap();

    // Place the segment where a real table would: on a page boundary.
    io.set_len(SEGMENT_ALIGN).await.unwrap();
    let offset = io.append(&[&built.bytes]).await.unwrap();
    assert_eq!(
        offset % SEGMENT_ALIGN,
        0,
        "segments must start page-aligned"
    );

    let (data, meta) = built.placed(offset);
    let bytes = io.read_immutable(data).await.unwrap();
    let reader = SegmentReader::new(bytes, offset, meta, schema.clone(), &layout).unwrap();
    (dir, reader)
}

fn options(compression: Compression, encodings: bool) -> TableOptions {
    TableOptions {
        compression,
        dictionary_encoding: encodings,
        rle_encoding: encodings,
        durability: Durability::None,
        ..TableOptions::default()
    }
}

fn batch(schema: &SchemaRef, columns: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(schema.clone(), columns).unwrap()
}

/// One column of each supported type, with nulls where the type allows them.
fn wide_case() -> (SchemaRef, RecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("bool", DataType::Boolean, true),
        Field::new("i8", DataType::Int8, true),
        Field::new("i16", DataType::Int16, false),
        Field::new("i32", DataType::Int32, true),
        Field::new("i64", DataType::Int64, false),
        Field::new("u8", DataType::UInt8, true),
        Field::new("u32", DataType::UInt32, false),
        Field::new("u64", DataType::UInt64, true),
        Field::new("f32", DataType::Float32, true),
        Field::new("f64", DataType::Float64, false),
        Field::new("utf8", DataType::Utf8, true),
        Field::new("large_utf8", DataType::LargeUtf8, false),
        Field::new("binary", DataType::Binary, true),
        Field::new("large_binary", DataType::LargeBinary, false),
        Field::new("date32", DataType::Date32, true),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
        Field::new("null", DataType::Null, true),
    ]));

    let columns: Vec<ArrayRef> = vec![
        Arc::new(BooleanArray::from(vec![
            Some(true),
            None,
            Some(false),
            Some(true),
        ])),
        Arc::new(Int8Array::from(vec![Some(-1), Some(2), None, Some(127)])),
        Arc::new(Int16Array::from(vec![1, -2, 3, -4])),
        Arc::new(Int32Array::from(vec![Some(10), None, Some(-30), Some(40)])),
        Arc::new(Int64Array::from(vec![i64::MIN, 0, i64::MAX, 7])),
        Arc::new(UInt8Array::from(vec![Some(0), Some(255), None, Some(9)])),
        Arc::new(UInt32Array::from(vec![1, 2, 3, u32::MAX])),
        Arc::new(UInt64Array::from(vec![
            Some(u64::MAX),
            None,
            Some(0),
            Some(5),
        ])),
        Arc::new(Float32Array::from(vec![
            Some(1.5),
            None,
            Some(-0.0),
            Some(f32::MAX),
        ])),
        Arc::new(Float64Array::from(vec![1.0, -2.5, f64::MIN, 1e300])),
        Arc::new(StringArray::from(vec![
            Some("alpha"),
            None,
            Some(""),
            Some("a longer string with unicode: \u{1f600}"),
        ])),
        Arc::new(LargeStringArray::from(vec!["a", "bb", "ccc", "dddd"])),
        Arc::new(BinaryArray::from(vec![
            Some(&b"\x00\x01\x02"[..]),
            None,
            Some(&b""[..]),
            Some(&b"\xff\xfe"[..]),
        ])),
        Arc::new(LargeBinaryArray::from(vec![
            &b"x"[..],
            &b"yy"[..],
            &b"zzz"[..],
            &b""[..],
        ])),
        Arc::new(Date32Array::from(vec![
            Some(0),
            Some(19000),
            None,
            Some(-1),
        ])),
        Arc::new(
            TimestampMicrosecondArray::from(vec![
                Some(0),
                None,
                Some(1_700_000_000_000_000),
                Some(-5),
            ])
            .with_data_type(DataType::Timestamp(
                TimeUnit::Microsecond,
                Some("UTC".into()),
            )),
        ),
        Arc::new(NullArray::new(4)),
    ];

    let batch = batch(&schema, columns);
    (schema, batch)
}

#[tokio::test]
async fn every_supported_type_round_trips() {
    for compression in [Compression::None, Compression::Lz4, Compression::Zstd] {
        for encodings in [false, true] {
            let (schema, original) = wide_case();
            let opts = options(compression, encodings);
            let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&original), &opts).await;

            let read = reader.read(None).unwrap();
            assert_eq!(
                read, original,
                "compression {compression:?}, encodings {encodings}"
            );
        }
    }
}

#[tokio::test]
async fn an_uncompressed_plain_segment_reads_without_copying() {
    let (schema, original) = wide_case();
    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&original), &opts).await;

    assert!(reader.is_zero_copy(), "the mmap backend must map, not copy");
    let meta = reader.meta().unwrap();
    for (index, chunk) in meta.columns.iter().enumerate() {
        assert!(
            chunk.is_zero_copy(),
            "column {} ({}) fell off the zero-copy path",
            index,
            schema.field(index).name()
        );
    }
    assert_eq!(reader.read(None).unwrap(), original);
}

#[tokio::test]
async fn column_buffers_land_on_the_arrow_alignment() {
    let (schema, original) = wide_case();
    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&original), &opts).await;

    let meta = reader.meta().unwrap();
    for chunk in meta.columns.iter() {
        for buffer in chunk.buffers.iter() {
            assert_eq!(
                buffer.extent.offset.to_native() % BUFFER_ALIGN,
                0,
                "a buffer that is not {BUFFER_ALIGN}-byte aligned cannot be mapped into Arrow"
            );
        }
    }
}

#[tokio::test]
async fn decoded_arrays_point_into_the_mapped_file() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let values: Vec<i64> = (0..10_000).collect();
    let original = batch(&schema, vec![Arc::new(Int64Array::from(values.clone()))]);
    let opts = options(Compression::None, false);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("segment.lt");
    let io = open_backend(&path, IoBackend::Mmap, Durability::None, false).unwrap();
    let layout = schema_codec::SchemaLayout::of(&schema);
    let built = build_segment(0, &schema, layout.current(), &[original], &opts).unwrap();
    io.set_len(SEGMENT_ALIGN).await.unwrap();
    let offset = io.append(&[&built.bytes]).await.unwrap();
    let (data, meta) = built.placed(offset);

    let bytes = io.read_immutable(data).await.unwrap();
    let base = bytes.as_slice().as_ptr_range();
    let reader = SegmentReader::new(bytes, offset, meta, schema, &layout).unwrap();

    let column = reader.column(0).unwrap();
    let value_ptr = column.to_data().buffers()[0].as_ptr();
    assert!(
        base.contains(&value_ptr),
        "the array must borrow the mapping, not a copy of it"
    );

    // The mapping outlives the reader, because the Arrow buffer holds it.
    drop(reader);
    let array = column.as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(array.values(), &values[..]);
}

#[tokio::test]
async fn projection_reads_only_the_columns_asked_for() {
    let (schema, original) = wide_case();
    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&original), &opts).await;

    let projection = [10usize, 4]; // utf8 then i64, out of order on purpose
    let read = reader.read(Some(&projection)).unwrap();

    assert_eq!(read.num_columns(), 2);
    assert_eq!(read.schema().field(0).name(), "utf8");
    assert_eq!(read.schema().field(1).name(), "i64");
    assert_eq!(read.column(0), original.column(10));
    assert_eq!(read.column(1), original.column(4));

    // A projection's byte ranges cover only its own columns.
    let all = reader.projected_extents(None).unwrap();
    let some = reader.projected_extents(Some(&projection)).unwrap();
    assert!(
        some.len() < all.len(),
        "a two-column projection must fetch fewer ranges than the whole segment"
    );
}

#[tokio::test]
async fn an_empty_projection_still_reports_the_row_count() {
    let (schema, original) = wide_case();
    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&original), &opts).await;

    let read = reader.read(Some(&[])).unwrap();
    assert_eq!(read.num_columns(), 0);
    assert_eq!(read.num_rows(), original.num_rows());
}

#[tokio::test]
async fn many_batches_become_one_segment() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));

    let batches: Vec<RecordBatch> = (0..5)
        .map(|i| {
            batch(
                &schema,
                vec![
                    Arc::new(Int32Array::from(vec![i * 3, i * 3 + 1, i * 3 + 2])),
                    Arc::new(StringArray::from(vec![
                        Some(format!("row{i}a")),
                        None,
                        Some(format!("row{i}c")),
                    ])),
                ],
            )
        })
        .collect();

    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, &batches, &opts).await;

    assert_eq!(reader.row_count().unwrap(), 15);
    let read = reader.read(None).unwrap();
    assert_eq!(read.num_rows(), 15);

    let ids = read
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ids.values(), &(0..15).collect::<Vec<i32>>()[..]);
    let names = read
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "row0a");
    assert!(names.is_null(1));
    assert_eq!(names.value(14), "row4c");
}

#[tokio::test]
async fn a_segment_with_no_rows_round_trips() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let empty = batch(
        &schema,
        vec![
            Arc::new(Int32Array::from(Vec::<i32>::new())),
            Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        ],
    );

    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, &[empty], &opts).await;

    assert_eq!(reader.row_count().unwrap(), 0);
    assert_eq!(reader.read(None).unwrap().num_rows(), 0);
}

#[tokio::test]
async fn an_all_null_column_round_trips() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)]));
    let original = batch(
        &schema,
        vec![Arc::new(Int32Array::from(vec![None, None, None]))],
    );

    for encodings in [false, true] {
        let opts = options(Compression::None, encodings);
        let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&original), &opts).await;
        assert_eq!(reader.read(None).unwrap(), original);

        let meta = reader.meta().unwrap();
        assert_eq!(meta.columns[0].null_count.to_native(), 3);
        assert!(meta.columns[0].zone.to_native().is_unknown());
    }
}

#[tokio::test]
async fn a_repetitive_column_is_re_encoded_and_still_exact() {
    let schema = Arc::new(Schema::new(vec![Field::new("tag", DataType::Utf8, true)]));
    let values: Vec<Option<String>> = (0..4000)
        .map(|i| match i % 5 {
            0 => None,
            n => Some(format!("tag{n}")),
        })
        .collect();
    let original = batch(&schema, vec![Arc::new(StringArray::from(values))]);

    let plain = options(Compression::None, false);
    let (_d1, plain_reader) = round_trip(&schema, std::slice::from_ref(&original), &plain).await;
    let plain_bytes = plain_reader.meta().unwrap().columns[0]
        .buffers
        .iter()
        .map(|b| b.extent.len.to_native())
        .sum::<u64>();

    let encoded = options(Compression::None, true);
    let (_d2, reader) = round_trip(&schema, std::slice::from_ref(&original), &encoded).await;

    let meta = reader.meta().unwrap();
    assert_ne!(
        meta.columns[0].encoding.to_native(),
        Encoding::Plain,
        "a column with five distinct values must not be stored plainly"
    );
    assert!(
        reader.read(None).unwrap() == original,
        "re-encoding must be exact"
    );

    let encoded_bytes = meta.columns[0]
        .buffers
        .iter()
        .map(|b| b.extent.len.to_native())
        .sum::<u64>();
    assert!(encoded_bytes < plain_bytes, "re-encoding must save space");
}

#[tokio::test]
async fn compression_shrinks_a_compressible_column() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let original = batch(
        &schema,
        vec![Arc::new(Int64Array::from(vec![42i64; 20_000]))],
    );

    let plain = options(Compression::None, false);
    let (_d1, plain_reader) = round_trip(&schema, std::slice::from_ref(&original), &plain).await;
    // Whole column, because a compressed one may be stored in blocks.
    let plain_size = plain_reader.meta().unwrap().columns[0].stored_bytes();

    let zstd = options(Compression::Zstd, false);
    let (_d2, reader) = round_trip(&schema, std::slice::from_ref(&original), &zstd).await;
    let meta = reader.meta().unwrap();

    assert_eq!(meta.columns[0].codec.to_native(), Codec::Zstd);
    let packed = meta.columns[0].stored_bytes();
    assert!(
        packed < plain_size / 10,
        "{packed} is not much smaller than {plain_size}"
    );
    assert_eq!(reader.read(None).unwrap(), original);
}

#[tokio::test]
async fn zone_maps_bound_every_ordered_column() {
    let (schema, original) = wide_case();
    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&original), &opts).await;
    let meta = reader.meta().unwrap();

    let i64_zone = meta.columns[4].zone.to_native();
    let min = i64_zone.min_array(&DataType::Int64).unwrap();
    let max = i64_zone.max_array(&DataType::Int64).unwrap();
    assert_eq!(
        min.as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        i64::MIN
    );
    assert_eq!(
        max.as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        i64::MAX
    );

    let utf8_zone = meta.columns[10].zone.to_native();
    assert_eq!(utf8_zone.null_count, 1);
    let min = utf8_zone.min_array(&DataType::Utf8).unwrap();
    assert_eq!(
        min.as_any().downcast_ref::<StringArray>().unwrap().value(0),
        "",
        "the empty string is the smallest value present"
    );
}

#[tokio::test]
async fn a_damaged_column_buffer_is_caught_not_decoded() {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let original = batch(
        &schema,
        vec![Arc::new(Int64Array::from((0..1000i64).collect::<Vec<_>>()))],
    );

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("segment.lt");
    let opts = options(Compression::None, false);
    let layout = schema_codec::SchemaLayout::of(&schema);
    let built = build_segment(0, &schema, layout.current(), &[original], &opts).unwrap();

    let io = open_backend(&path, IoBackend::Pread, Durability::None, false).unwrap();
    io.set_len(SEGMENT_ALIGN).await.unwrap();
    let offset = io.append(&[&built.bytes]).await.unwrap();
    let (data, meta) = built.placed(offset);

    // Flip a bit inside the values buffer.
    let values = built.meta.columns[0]
        .buffer(localtables_format::columnar::BufferRole::Data)
        .unwrap()
        .extent;
    let target = offset + values.offset + values.len / 2;
    let mut byte = io.read_at(target, 1).await.unwrap().as_slice()[0];
    byte ^= 0x08;
    io.write_at(target, &[byte]).await.unwrap();

    let bytes = io.read_immutable(data).await.unwrap();
    let reader = SegmentReader::new(bytes, offset, meta, schema, &layout).unwrap();

    let err = reader.column(0).unwrap_err();
    assert!(
        matches!(err, localtables_format::Error::Checksum { .. }),
        "damage must be reported, not handed to Arrow as data: got {err:?}"
    );
}

#[tokio::test]
async fn a_segment_written_for_another_schema_is_refused() {
    let (schema, original) = wide_case();
    let opts = options(Compression::None, false);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("segment.lt");
    let io = open_backend(&path, IoBackend::Mmap, Durability::None, false).unwrap();
    let built = build_segment(0, &schema, 0xaaaa, &[original], &opts).unwrap();
    io.set_len(SEGMENT_ALIGN).await.unwrap();
    let offset = io.append(&[&built.bytes]).await.unwrap();
    let (data, meta) = built.placed(offset);

    let bytes = io.read_immutable(data).await.unwrap();
    let layout = schema_codec::SchemaLayout::of(&schema);
    let err = SegmentReader::new(bytes, offset, meta, schema, &layout).unwrap_err();
    assert!(
        matches!(err, localtables_format::Error::SchemaMismatch(_)),
        "got {err:?}"
    );
}

#[tokio::test]
async fn a_truncated_segment_is_refused_rather_than_read() {
    let (schema, original) = wide_case();
    let opts = options(Compression::None, false);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("segment.lt");
    let io = open_backend(&path, IoBackend::Pread, Durability::None, false).unwrap();

    let layout = schema_codec::SchemaLayout::of(&schema);
    let built = build_segment(0, &schema, layout.current(), &[original], &opts).unwrap();
    io.set_len(SEGMENT_ALIGN).await.unwrap();
    let offset = io.append(&[&built.bytes]).await.unwrap();
    let (_data, meta) = built.placed(offset);

    // Hand the reader fewer bytes than the segment needs.
    let short = Extent::new(offset, built.len() / 2);
    let bytes = io.read_immutable(short).await.unwrap();
    let err = SegmentReader::new(bytes, offset, meta, schema, &layout).unwrap_err();
    assert!(
        matches!(err, localtables_format::Error::Corrupt(_)),
        "got {err:?}"
    );
}

/// A dictionary column must come back as a dictionary, not expanded.
///
/// This is what lets a group by hash indices rather than values, which is the
/// whole reason to declare the column that way.
#[tokio::test]
async fn a_dictionary_column_stays_a_dictionary() {
    use arrow_array::types::Int32Type;
    use arrow_array::DictionaryArray;

    let data_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    let schema = Arc::new(Schema::new(vec![
        Field::new("category", data_type.clone(), true),
        Field::new("value", DataType::Int64, false),
    ]));

    let keys = Int32Array::from(vec![Some(0), Some(1), None, Some(2), Some(0)]);
    let values = Arc::new(StringArray::from(vec!["alpha", "beta", "gamma"]));
    let dict = DictionaryArray::<Int32Type>::try_new(keys, values).unwrap();

    let original = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(dict),
            Arc::new(Int64Array::from(vec![1i64, 2, 3, 4, 5])),
        ],
    )
    .unwrap();

    for compression in [Compression::None, Compression::Lz4] {
        let opts = options(compression, true);
        let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&original), &opts).await;
        let read = reader.read(None).unwrap();

        assert_eq!(
            read.column(0).data_type(),
            &data_type,
            "the column must keep the type the schema declares"
        );
        assert_eq!(read, original, "compression {compression:?}");

        // The dictionary itself survives, not just the logical values.
        let restored = read
            .column(0)
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .expect("a dictionary column decodes to a dictionary array");
        assert_eq!(restored.values().len(), 3, "the distinct values are intact");
        assert!(restored.is_null(2));
    }
}

/// A dictionary column stored uncompressed reads with no copy, like any other.
#[tokio::test]
async fn a_dictionary_column_reads_without_copying() {
    use arrow_array::types::Int32Type;
    use arrow_array::DictionaryArray;

    let data_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    let schema = Arc::new(Schema::new(vec![Field::new("c", data_type, false)]));

    let keys = Int32Array::from((0..1000).map(|i| i % 4).collect::<Vec<i32>>());
    let values = Arc::new(StringArray::from(vec!["w", "x", "y", "z"]));
    let dict = DictionaryArray::<Int32Type>::try_new(keys, values).unwrap();
    let original = RecordBatch::try_new(schema.clone(), vec![Arc::new(dict)]).unwrap();

    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&original), &opts).await;

    let meta = reader.meta().unwrap();
    assert!(
        meta.columns[0].is_zero_copy(),
        "a dictionary column is stored plainly, so it maps like any other"
    );
    assert_eq!(reader.read(None).unwrap(), original);
}

/// Nested and parameterised types round-trip, because the format stores what
/// Arrow lays out rather than a list of types it knows about.
#[tokio::test]
async fn nested_and_exotic_types_round_trip() {
    use arrow_array::builder::{ListBuilder, StringBuilder};
    use arrow_array::types::Int32Type;
    use arrow_array::{
        Decimal128Array, DictionaryArray, DurationMicrosecondArray, FixedSizeBinaryArray,
        IntervalMonthDayNanoArray, MapArray, StructArray, Time64NanosecondArray, UInt16Array,
    };
    use arrow_buffer::i256;

    // A list of strings, with a null list and an empty one.
    let mut lists = ListBuilder::new(StringBuilder::new());
    lists.values().append_value("a");
    lists.values().append_value("bb");
    lists.append(true);
    lists.append(false);
    lists.append(true);
    lists.values().append_null();
    lists.values().append_value("c");
    lists.append(true);
    let list = lists.finish();

    // A struct of two differently typed fields.
    let struct_array = StructArray::from(vec![
        (
            Arc::new(Field::new("n", DataType::Int32, true)),
            Arc::new(Int32Array::from(vec![Some(1), None, Some(3), Some(4)])) as ArrayRef,
        ),
        (
            Arc::new(Field::new("s", DataType::Utf8, false)),
            Arc::new(StringArray::from(vec!["w", "x", "y", "z"])) as ArrayRef,
        ),
    ]);

    let dict = DictionaryArray::<Int32Type>::try_new(
        Int32Array::from(vec![Some(0), Some(1), None, Some(0)]),
        Arc::new(StringArray::from(vec!["red", "blue"])),
    )
    .unwrap();

    let decimal = Decimal128Array::from(vec![Some(1234i128), None, Some(-9), Some(0)])
        .with_precision_and_scale(10, 2)
        .unwrap();

    let decimal256 = arrow_array::Decimal256Array::from(vec![
        Some(i256::from_i128(1)),
        Some(i256::from_i128(-2)),
        None,
        Some(i256::MAX),
    ])
    .with_precision_and_scale(40, 4)
    .unwrap();

    let fixed = FixedSizeBinaryArray::try_from_iter(
        vec![vec![1u8, 2, 3], vec![4, 5, 6], vec![7, 8, 9], vec![0, 0, 0]].into_iter(),
    )
    .unwrap();

    // A map is a list of structs, so it exercises two levels of nesting.
    let map = {
        let keys = StringArray::from(vec!["k1", "k2", "k3"]);
        let values = Int32Array::from(vec![10, 20, 30]);
        let entries = StructArray::from(vec![
            (
                Arc::new(Field::new("keys", DataType::Utf8, false)),
                Arc::new(keys) as ArrayRef,
            ),
            (
                Arc::new(Field::new("values", DataType::Int32, true)),
                Arc::new(values) as ArrayRef,
            ),
        ]);
        let offsets = arrow_buffer::OffsetBuffer::new(vec![0, 1, 1, 2, 3].into());
        let field = Arc::new(Field::new("entries", entries.data_type().clone(), false));
        MapArray::new(field, offsets, entries, None, false)
    };

    let schema = Arc::new(Schema::new(vec![
        Field::new("list", list.data_type().clone(), true),
        Field::new("st", struct_array.data_type().clone(), false),
        Field::new("dict", dict.data_type().clone(), true),
        Field::new("dec", decimal.data_type().clone(), true),
        Field::new("dec256", decimal256.data_type().clone(), true),
        Field::new("fixed", fixed.data_type().clone(), false),
        Field::new("map", map.data_type().clone(), false),
        Field::new("u16", DataType::UInt16, true),
        Field::new("dur", DataType::Duration(TimeUnit::Microsecond), true),
        Field::new("time", DataType::Time64(TimeUnit::Nanosecond), true),
        Field::new(
            "interval",
            DataType::Interval(arrow_schema::IntervalUnit::MonthDayNano),
            true,
        ),
    ]));

    let original = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(list),
            Arc::new(struct_array),
            Arc::new(dict),
            Arc::new(decimal),
            Arc::new(decimal256),
            Arc::new(fixed),
            Arc::new(map),
            Arc::new(UInt16Array::from(vec![Some(1), None, Some(65535), Some(0)])),
            Arc::new(DurationMicrosecondArray::from(vec![
                Some(1i64),
                None,
                Some(-1),
                Some(i64::MAX),
            ])),
            Arc::new(Time64NanosecondArray::from(vec![
                Some(0i64),
                Some(1),
                None,
                Some(86_399_999_999_999),
            ])),
            Arc::new(IntervalMonthDayNanoArray::from(vec![
                arrow_buffer::IntervalMonthDayNano::new(1, 2, 3),
                arrow_buffer::IntervalMonthDayNano::new(0, 0, 0),
                arrow_buffer::IntervalMonthDayNano::new(-1, -2, -3),
                arrow_buffer::IntervalMonthDayNano::new(12, 30, 1_000),
            ])),
        ],
    )
    .unwrap();

    for compression in [Compression::None, Compression::Lz4] {
        let opts = options(compression, true);
        let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&original), &opts).await;
        assert_eq!(
            reader.read(None).unwrap(),
            original,
            "compression {compression:?}"
        );
    }
}

/// An extension type survives, because its identity lives in the schema's
/// field metadata and its data is an ordinary array.
#[tokio::test]
async fn an_extension_type_keeps_its_metadata() {
    use std::collections::HashMap;

    let metadata = HashMap::from([
        (
            "ARROW:extension:name".to_string(),
            "myorg.point".to_string(),
        ),
        (
            "ARROW:extension:metadata".to_string(),
            "{\"srid\":4326}".to_string(),
        ),
    ]);
    let field =
        Field::new("point", DataType::FixedSizeBinary(8), false).with_metadata(metadata.clone());
    let schema = Arc::new(Schema::new(vec![field]));

    let storage = arrow_array::FixedSizeBinaryArray::try_from_iter(
        vec![vec![1u8; 8], vec![2u8; 8], vec![3u8; 8]].into_iter(),
    )
    .unwrap();
    let original = RecordBatch::try_new(schema.clone(), vec![Arc::new(storage)]).unwrap();

    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&original), &opts).await;
    let read = reader.read(None).unwrap();

    assert_eq!(read, original);
    assert_eq!(
        read.schema().field(0).metadata(),
        &metadata,
        "an extension type is its storage plus its field metadata; both must survive"
    );
}

/// A sliced batch stores only the rows it owns, whatever the types involved.
#[tokio::test]
async fn a_sliced_batch_does_not_store_its_parents_rows() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    let text: Vec<String> = (0..10_000).map(|i| format!("value-number-{i}")).collect();
    let whole = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from((0..10_000i64).collect::<Vec<_>>())),
            Arc::new(StringArray::from(text)),
        ],
    )
    .unwrap();

    let sliced = whole.slice(5_000, 3);
    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&sliced), &opts).await;

    assert_eq!(reader.read(None).unwrap(), sliced);
    let meta = reader.meta().unwrap();
    let stored: u64 = meta
        .columns
        .iter()
        .map(|c| {
            c.buffers
                .iter()
                .map(|b| b.extent.len.to_native())
                .sum::<u64>()
        })
        .sum();
    assert!(
        stored < 1_000,
        "three rows should not carry ten thousand: stored {stored} bytes"
    );
}

/// View types round-trip, including the long values that live in a separate
/// data buffer rather than inline in the view.
///
/// A view array has a variable number of buffers — one of views, then as many
/// data buffers as it needs — and each view for a long value carries the index
/// of the buffer holding it. Storing the buffers in Arrow's order is what keeps
/// those indices meaning what they meant.
#[tokio::test]
async fn view_types_round_trip() {
    use arrow_array::{BinaryViewArray, StringViewArray};

    // Twelve bytes or fewer live inline in the view; longer ones do not.
    let short = "cat-3";
    let long = "a string comfortably past the inline limit of a view";
    let longer = "another long value, so the data buffer holds more than one";

    let strings = StringViewArray::from(vec![
        Some(short),
        None,
        Some(long),
        Some(""),
        Some(longer),
        Some(short),
    ]);
    let binaries = BinaryViewArray::from(vec![
        Some(&b"short"[..]),
        Some(long.as_bytes()),
        None,
        Some(&b""[..]),
        Some(longer.as_bytes()),
        Some(&b"\x00\xff\x00"[..]),
    ]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("s", DataType::Utf8View, true),
        Field::new("b", DataType::BinaryView, true),
        Field::new("n", DataType::Int64, false),
    ]));
    let original = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(strings),
            Arc::new(binaries),
            Arc::new(Int64Array::from(vec![1i64, 2, 3, 4, 5, 6])),
        ],
    )
    .unwrap();

    for compression in [Compression::None, Compression::Lz4, Compression::Zstd] {
        for encodings in [false, true] {
            let opts = options(compression, encodings);
            let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&original), &opts).await;
            let read = reader.read(None).unwrap();

            assert_eq!(
                read, original,
                "compression {compression:?}, encodings {encodings}"
            );
            assert_eq!(
                read.column(0).data_type(),
                &DataType::Utf8View,
                "a view column must come back as a view, not converted"
            );

            // The long value is what proves the data buffers and their indices
            // survived; a short one would pass even if they had not.
            let strings = read
                .column(0)
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap();
            assert_eq!(strings.value(2), long);
            assert_eq!(strings.value(4), longer);
            assert!(strings.is_null(1));
        }
    }
}

/// A sliced view array stores only its own rows.
///
/// This is the case view types make awkward: slicing narrows the views buffer
/// but leaves the data buffers whole, so a two-row slice of a large array still
/// points into all of its data.
#[tokio::test]
async fn a_sliced_view_column_does_not_store_its_parents_rows() {
    use arrow_array::StringViewArray;

    // Long enough that every value lives in a data buffer rather than inline.
    let values: Vec<String> = (0..5_000)
        .map(|i| format!("a value long enough to need a data buffer, number {i}"))
        .collect();
    let whole = StringViewArray::from(values.iter().map(|v| Some(v.as_str())).collect::<Vec<_>>());

    let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8View, true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(whole)]).unwrap();
    let sliced = batch.slice(2_500, 2);

    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&sliced), &opts).await;

    assert_eq!(reader.read(None).unwrap(), sliced);

    let meta = reader.meta().unwrap();
    let stored: u64 = meta.columns[0]
        .buffers
        .iter()
        .map(|b| b.extent.len.to_native())
        .sum();
    assert!(
        stored < 1_000,
        "two rows should not carry five thousand: stored {stored} bytes"
    );
}

/// Reading a segment as a schema that has gained a column since it was
/// written. This is what an added column looks like from below: the rows are
/// older than the column, so they have no value for it.
#[tokio::test]
async fn a_segment_written_before_a_column_existed_reads_it_as_null() {
    let before: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let after: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("score", DataType::Float64, true),
    ]));

    let batch = RecordBatch::try_new(
        before.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();

    let opts = options(Compression::None, false);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("segment.lt");
    let io = open_backend(&path, IoBackend::Mmap, Durability::None, false).unwrap();

    // Written when the schema had two columns.
    let written = schema_codec::SchemaLayout::of(&before);
    let built = build_segment(0, &before, written.current(), &[batch], &opts).unwrap();
    io.set_len(SEGMENT_ALIGN).await.unwrap();
    let offset = io.append(&[&built.bytes]).await.unwrap();
    let (data, meta) = built.placed(offset);
    let bytes = io.read_immutable(data).await.unwrap();

    // Read now that the schema has three.
    let layout = schema_codec::SchemaLayout::of(&after);
    let reader = SegmentReader::new(bytes, offset, meta, after.clone(), &layout).unwrap();

    let full = reader.read(None).unwrap();
    assert_eq!(full.num_columns(), 3);
    assert_eq!(full.num_rows(), 3);
    assert_eq!(full.schema().field(2).name(), "score");
    assert_eq!(
        full.column(2).null_count(),
        3,
        "rows older than the column have no value for it"
    );
    // The columns that were written are untouched.
    assert_eq!(
        full.column(0).as_any().downcast_ref::<Int32Array>().unwrap(),
        &Int32Array::from(vec![1, 2, 3])
    );

    // Projecting only the new column works, and so does mixing old and new.
    let only_new = reader.read(Some(&[2])).unwrap();
    assert_eq!(only_new.num_rows(), 3);
    assert_eq!(only_new.column(0).null_count(), 3);

    let mixed = reader.read(Some(&[2, 0])).unwrap();
    assert_eq!(mixed.num_columns(), 2);
    assert_eq!(mixed.column(0).null_count(), 3);
    assert_eq!(
        mixed.column(1).as_any().downcast_ref::<Int32Array>().unwrap(),
        &Int32Array::from(vec![1, 2, 3])
    );

    // And one column at a time.
    assert_eq!(reader.column(2).unwrap().null_count(), 3);
    assert_eq!(reader.column(0).unwrap().len(), 3);
}

/// A segment holding columns the schema no longer has is refused rather than
/// misread, which is why dropping a column rewrites.
#[tokio::test]
async fn a_segment_with_columns_the_schema_lost_is_refused() {
    let before: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let after: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "id",
        DataType::Int32,
        false,
    )]));

    let batch = RecordBatch::try_new(
        before.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(StringArray::from(vec!["a"])),
        ],
    )
    .unwrap();

    let opts = options(Compression::None, false);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("segment.lt");
    let io = open_backend(&path, IoBackend::Mmap, Durability::None, false).unwrap();

    let written = schema_codec::SchemaLayout::of(&before);
    let built = build_segment(0, &before, written.current(), &[batch], &opts).unwrap();
    io.set_len(SEGMENT_ALIGN).await.unwrap();
    let offset = io.append(&[&built.bytes]).await.unwrap();
    let (data, meta) = built.placed(offset);
    let bytes = io.read_immutable(data).await.unwrap();

    let layout = schema_codec::SchemaLayout::of(&after);
    let err = SegmentReader::new(bytes, offset, meta, after, &layout).unwrap_err();
    assert!(
        matches!(err, localtables_format::Error::SchemaMismatch(_)),
        "got {err:?}"
    );
}

/// Bounds for each row range inside a segment, so a scan can skip the ranges a
/// predicate rules out rather than hand on the whole segment.
#[tokio::test]
async fn a_segment_records_bounds_for_each_of_its_pages() {
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "id",
        DataType::Int32,
        false,
    )]));
    // Ten pages of a hundred, with ids running in order so each page covers a
    // narrow, distinct range.
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from((0..1000).collect::<Vec<i32>>()))],
    )
    .unwrap();

    let opts = TableOptions {
        page_rows: 100,
        ..options(Compression::None, false)
    };
    let (_dir, reader) = round_trip(&schema, &[batch], &opts).await;

    assert_eq!(reader.page_rows().unwrap(), 100);
    let zones = reader.page_zones(0).unwrap().expect("bounds per page");
    assert_eq!(zones.len(), 10);

    for (page, zone) in zones.iter().enumerate() {
        let low = zone.min_array(&DataType::Int32).unwrap();
        let high = zone.max_array(&DataType::Int32).unwrap();
        let low = low.as_any().downcast_ref::<Int32Array>().unwrap().value(0);
        let high = high.as_any().downcast_ref::<Int32Array>().unwrap().value(0);
        assert_eq!(low, page as i32 * 100);
        assert_eq!(high, page as i32 * 100 + 99);
        assert_eq!(reader.page_range(page).unwrap(), Some((page * 100, 100)));
    }
    assert_eq!(reader.page_range(10).unwrap(), None);
}

/// A segment that is one page or fewer records none, because they would only
/// repeat the chunk's own zone map.
#[tokio::test]
async fn a_segment_of_one_page_records_no_page_bounds() {
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "id",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from((0..50).collect::<Vec<i32>>()))],
    )
    .unwrap();

    let opts = TableOptions {
        page_rows: 100,
        ..options(Compression::None, false)
    };
    let (_dir, reader) = round_trip(&schema, &[batch], &opts).await;

    assert_eq!(reader.page_rows().unwrap(), 0);
    assert!(reader.page_zones(0).unwrap().is_none());
    assert_eq!(reader.page_range(0).unwrap(), None);
}

#[tokio::test]
async fn switching_pages_off_records_none() {
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "id",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from((0..1000).collect::<Vec<i32>>()))],
    )
    .unwrap();

    let opts = TableOptions {
        page_rows: 0,
        ..options(Compression::None, false)
    };
    let (_dir, reader) = round_trip(&schema, &[batch], &opts).await;
    assert_eq!(reader.page_rows().unwrap(), 0);
    assert!(reader.page_zones(0).unwrap().is_none());
}

/// A last page shorter than the rest is still covered.
#[tokio::test]
async fn a_ragged_last_page_is_covered() {
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "id",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from((0..250).collect::<Vec<i32>>()))],
    )
    .unwrap();

    let opts = TableOptions {
        page_rows: 100,
        ..options(Compression::None, false)
    };
    let (_dir, reader) = round_trip(&schema, &[batch], &opts).await;

    let zones = reader.page_zones(0).unwrap().unwrap();
    assert_eq!(zones.len(), 3);
    assert_eq!(reader.page_range(2).unwrap(), Some((200, 50)));
    let high = zones[2].max_array(&DataType::Int32).unwrap();
    assert_eq!(
        high.as_any().downcast_ref::<Int32Array>().unwrap().value(0),
        249
    );
}

/// Page bounds must not disturb the data. They are a buffer the decoder skips.
#[tokio::test]
async fn page_bounds_do_not_change_what_a_segment_returns() {
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let ids: Vec<i32> = (0..500).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("name-{i}")).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
        ],
    )
    .unwrap();

    let with = TableOptions {
        page_rows: 100,
        ..options(Compression::None, false)
    };
    let without = TableOptions {
        page_rows: 0,
        ..options(Compression::None, false)
    };
    let (_a, paged) = round_trip(&schema, std::slice::from_ref(&batch), &with).await;
    let (_b, plain) = round_trip(&schema, std::slice::from_ref(&batch), &without).await;

    assert_eq!(paged.read(None).unwrap(), batch);
    assert_eq!(plain.read(None).unwrap(), batch);
    assert!(
        paged.is_zero_copy(),
        "page bounds must not cost the zero-copy read path"
    );
}

/// Decoding a range must give exactly what decoding everything and slicing
/// would. This is the property the page-skipping read path rests on, and it has
/// to hold for every encoding, because each undoes something different.
#[tokio::test]
async fn decoding_a_range_matches_decoding_everything_and_slicing() {
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        // Plain: buffers handed back as they are.
        Field::new("id", DataType::Int64, false),
        // Dictionary: few distinct values, expanded on read.
        Field::new("category", DataType::Utf8, true),
        // Runs: long stretches of one value.
        Field::new("flag", DataType::Int32, false),
        // Variable width, so offsets have to be sliced correctly too.
        Field::new("body", DataType::Utf8, true),
    ]));

    let rows = 1000i64;
    let ids: Vec<i64> = (0..rows).collect();
    let categories: Vec<String> = ids.iter().map(|i| format!("cat-{}", i % 4)).collect();
    let flags: Vec<i32> = ids.iter().map(|i| (i / 100) as i32).collect();
    let bodies: Vec<Option<String>> = ids
        .iter()
        .map(|i| (i % 7 != 0).then(|| format!("body-{i}")))
        .collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(categories)),
            Arc::new(Int32Array::from(flags)),
            Arc::new(StringArray::from(bodies)),
        ],
    )
    .unwrap();

    // Encodings on, so the dictionary and run-length paths are the ones taken.
    let opts = TableOptions {
        dictionary_encoding: true,
        rle_encoding: true,
        ..options(Compression::None, false)
    };
    let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&batch), &opts).await;

    let whole = reader.read(None).unwrap();
    assert_eq!(whole, batch);

    for (start, len) in [
        (0, 1),
        (0, 1000),
        (1, 1),
        (250, 500),
        (999, 1),
        (500, 0),
        (37, 313),
    ] {
        let range = reader.read_rows(None, start, len).unwrap();
        assert_eq!(range.num_rows(), len, "rows {start}..{}", start + len);
        for column in 0..schema.fields().len() {
            let expected = whole.column(column).slice(start, len);
            assert_eq!(
                range.column(column),
                &expected,
                "column {} over rows {start}..{}",
                schema.field(column).name(),
                start + len
            );
            // And one column at a time takes the same path.
            assert_eq!(
                &reader.column_rows(column, start, len).unwrap(),
                &expected,
                "column_rows for {}",
                schema.field(column).name()
            );
        }
    }
}

#[tokio::test]
async fn a_range_outside_the_segment_is_refused() {
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "id",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from((0..100).collect::<Vec<i32>>()))],
    )
    .unwrap();
    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, &[batch], &opts).await;

    assert!(reader.read_rows(None, 0, 101).is_err());
    assert!(reader.read_rows(None, 100, 1).is_err());
    assert!(reader.read_rows(None, 101, 0).is_err());
    assert!(reader.read_rows(None, 100, 0).is_ok(), "the empty tail is a valid range");
}

/// A projection of nothing still has to carry the range's row count, or a
/// count-only query over skipped pages would come out wrong.
#[tokio::test]
async fn a_count_only_read_of_a_range_carries_its_rows() {
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "id",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from((0..100).collect::<Vec<i32>>()))],
    )
    .unwrap();
    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, &[batch], &opts).await;

    let counted = reader.read_rows(Some(&[]), 10, 25).unwrap();
    assert_eq!(counted.num_columns(), 0);
    assert_eq!(counted.num_rows(), 25);
}

/// A compressed chunk is decompressed whole however small the range, because
/// compression covers a buffer rather than a page. The answer must still be
/// right; only the saving is absent.
#[tokio::test]
async fn a_range_of_a_compressed_chunk_is_still_correct() {
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from((0..1000i64).collect::<Vec<i64>>()))],
    )
    .unwrap();
    let opts = options(Compression::Lz4, false);
    let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&batch), &opts).await;

    let whole = reader.read(None).unwrap();
    let range = reader.read_rows(None, 400, 200).unwrap();
    assert_eq!(range.column(0), &whole.column(0).slice(400, 200));
}

/// A compressed column is cut into blocks so a range can be decompressed on its
/// own. The answer must not depend on where the block boundaries fall.
#[tokio::test]
async fn a_blocked_column_returns_the_same_rows_as_an_unblocked_one() {
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "body",
        DataType::Utf8,
        true,
    )]));
    let bodies: Vec<Option<String>> = (0..5_000)
        .map(|i| (i % 11 != 0).then(|| format!("user{i}@example.com some padding text")))
        .collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StringArray::from(bodies))],
    )
    .unwrap();

    let unblocked = TableOptions {
        compression: Compression::Lz4,
        compression_block_rows: 0,
        ..options(Compression::Lz4, false)
    };
    let blocked = TableOptions {
        compression: Compression::Lz4,
        compression_block_rows: 512,
        ..options(Compression::Lz4, false)
    };

    let (_a, whole) = round_trip(&schema, std::slice::from_ref(&batch), &unblocked).await;
    let (_b, split) = round_trip(&schema, std::slice::from_ref(&batch), &blocked).await;

    let meta = split.meta().unwrap();
    assert_eq!(meta.columns[0].block_rows.to_native(), 512);
    assert_eq!(meta.columns[0].blocks.len(), 10);
    assert!(
        meta.columns[0].buffers.is_empty(),
        "a blocked chunk holds its data in its blocks"
    );
    assert_eq!(whole.meta().unwrap().columns[0].blocks.len(), 0);

    assert_eq!(split.read(None).unwrap(), batch);
    assert_eq!(whole.read(None).unwrap(), batch);

    // Ranges inside one block, spanning two, and spanning many.
    for (start, len) in [
        (0, 1),
        (0, 512),
        (100, 50),
        (500, 24),
        (511, 2),
        (1000, 1500),
        (4999, 1),
        (0, 5000),
    ] {
        let expected = batch.column(0).slice(start, len);
        assert_eq!(
            split.read_rows(None, start, len).unwrap().column(0),
            &expected,
            "rows {start}..{}",
            start + len
        );
    }
}

/// An uncompressed column is never cut, whatever the block size says. Cutting
/// it would cost the zero-copy read path and save nothing.
#[tokio::test]
async fn an_uncompressed_column_is_never_blocked() {
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from((0..5_000i64).collect::<Vec<i64>>()))],
    )
    .unwrap();

    let opts = TableOptions {
        compression_block_rows: 512,
        ..options(Compression::None, false)
    };
    let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&batch), &opts).await;

    assert_eq!(reader.meta().unwrap().columns[0].blocks.len(), 0);
    assert!(reader.is_zero_copy(), "blocking must not cost zero-copy");
    assert_eq!(reader.read(None).unwrap(), batch);
}

/// The block size is separate from the page size, so pruning can be finer than
/// decompression or the other way round.
#[tokio::test]
async fn block_size_and_page_size_are_independent() {
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "body",
        DataType::Utf8,
        false,
    )]));
    let bodies: Vec<String> = (0..4_000).map(|i| format!("row {i} padding")).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StringArray::from(bodies))],
    )
    .unwrap();

    let opts = TableOptions {
        compression: Compression::Lz4,
        page_rows: 250,
        compression_block_rows: 1000,
        ..options(Compression::Lz4, false)
    };
    let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&batch), &opts).await;

    assert_eq!(reader.page_rows().unwrap(), 250, "16 pages of bounds");
    assert_eq!(reader.page_zones(0).unwrap().unwrap().len(), 16);
    assert_eq!(
        reader.meta().unwrap().columns[0].blocks.len(),
        4,
        "but only 4 blocks to decompress"
    );
    assert_eq!(reader.read(None).unwrap(), batch);
    assert_eq!(
        reader.read_rows(None, 250, 250).unwrap().column(0),
        &batch.column(0).slice(250, 250)
    );
}

/// An uncompressed variable-width column is cut into blocks, and a fixed-width
/// one is not.
///
/// The reason is Arrow's, not this format's: handing it a variable-width
/// column's buffers makes it walk every offset, and for text check every byte,
/// however few rows are wanted. Building one block costs one block. A
/// fixed-width column has no offsets to walk and gains nothing from being cut.
#[tokio::test]
async fn uncompressed_variable_width_columns_are_blocked_and_numbers_are_not() {
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("body", DataType::Utf8, false),
        Field::new("blob", DataType::Binary, false),
    ]));
    let ids: Vec<i64> = (0..4_000).collect();
    let bodies: Vec<String> = ids.iter().map(|i| format!("row {i} padding")).collect();
    let blobs: Vec<Vec<u8>> = ids.iter().map(|i| format!("blob {i}").into_bytes()).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(bodies)),
            Arc::new(BinaryArray::from(
                blobs.iter().map(|b| b.as_slice()).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();

    let opts = TableOptions {
        compression_block_rows: 1000,
        ..options(Compression::None, false)
    };
    let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&batch), &opts).await;
    let meta = reader.meta().unwrap();

    assert_eq!(meta.columns[0].blocks.len(), 0, "a number needs no blocks");
    assert_eq!(meta.columns[1].blocks.len(), 4, "text is cut");
    assert_eq!(meta.columns[2].blocks.len(), 4, "binary is cut");
    assert_eq!(reader.block_rows().unwrap(), 1000);

    // Blocked or not, the answer is the same, and a block is still zero-copy.
    assert_eq!(reader.read(None).unwrap(), batch);
    assert!(
        reader.is_zero_copy(),
        "cutting an uncompressed column must not cost the zero-copy path"
    );
    for (start, len) in [(0, 1), (999, 2), (1000, 1000), (0, 4000), (3999, 1)] {
        assert_eq!(
            reader.read_rows(None, start, len).unwrap().column(1),
            &batch.column(1).slice(start, len),
            "rows {start}..{}",
            start + len
        );
    }
}

/// Switching blocks off leaves a column whole, which is what a caller who wants
/// one contiguous array should get.
#[tokio::test]
async fn a_block_size_of_zero_leaves_every_column_whole() {
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new(
        "body",
        DataType::Utf8,
        false,
    )]));
    let bodies: Vec<String> = (0..4_000).map(|i| format!("row {i}")).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StringArray::from(bodies))],
    )
    .unwrap();

    let opts = TableOptions {
        compression_block_rows: 0,
        ..options(Compression::None, false)
    };
    let (_dir, reader) = round_trip(&schema, std::slice::from_ref(&batch), &opts).await;
    assert_eq!(reader.meta().unwrap().columns[0].blocks.len(), 0);
    assert_eq!(reader.block_rows().unwrap(), 0);
    assert_eq!(reader.read(None).unwrap(), batch);
}
