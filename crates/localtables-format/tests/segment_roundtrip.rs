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

    let fingerprint = schema_codec::fingerprint(schema);
    let built = build_segment(0, schema, fingerprint, batches, options).unwrap();

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
    let reader = SegmentReader::new(bytes, offset, meta, schema.clone(), fingerprint).unwrap();
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
    let fingerprint = schema_codec::fingerprint(&schema);
    let built = build_segment(0, &schema, fingerprint, &[original], &opts).unwrap();
    io.set_len(SEGMENT_ALIGN).await.unwrap();
    let offset = io.append(&[&built.bytes]).await.unwrap();
    let (data, meta) = built.placed(offset);

    let bytes = io.read_immutable(data).await.unwrap();
    let base = bytes.as_slice().as_ptr_range();
    let reader = SegmentReader::new(bytes, offset, meta, schema, fingerprint).unwrap();

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
    let plain_size: u64 = plain_reader.meta().unwrap().columns[0]
        .buffers
        .iter()
        .map(|b| b.extent.len.to_native())
        .sum();

    let zstd = options(Compression::Zstd, false);
    let (_d2, reader) = round_trip(&schema, std::slice::from_ref(&original), &zstd).await;
    let meta = reader.meta().unwrap();

    assert_eq!(meta.columns[0].codec.to_native(), Codec::Zstd);
    let packed: u64 = meta.columns[0]
        .buffers
        .iter()
        .map(|b| b.extent.len.to_native())
        .sum();
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
    let fingerprint = schema_codec::fingerprint(&schema);
    let built = build_segment(0, &schema, fingerprint, &[original], &opts).unwrap();

    let io = open_backend(&path, IoBackend::Pread, Durability::None, false).unwrap();
    io.set_len(SEGMENT_ALIGN).await.unwrap();
    let offset = io.append(&[&built.bytes]).await.unwrap();
    let (data, meta) = built.placed(offset);

    // Flip a bit inside the values buffer.
    let values = built.meta.columns[0]
        .buffer(localtables_format::columnar::BufferRole::Values)
        .unwrap()
        .extent;
    let target = offset + values.offset + values.len / 2;
    let mut byte = io.read_at(target, 1).await.unwrap().as_slice()[0];
    byte ^= 0x08;
    io.write_at(target, &[byte]).await.unwrap();

    let bytes = io.read_immutable(data).await.unwrap();
    let reader = SegmentReader::new(bytes, offset, meta, schema, fingerprint).unwrap();

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
    let err = SegmentReader::new(bytes, offset, meta, schema, 0xbbbb).unwrap_err();
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

    let fingerprint = schema_codec::fingerprint(&schema);
    let built = build_segment(0, &schema, fingerprint, &[original], &opts).unwrap();
    io.set_len(SEGMENT_ALIGN).await.unwrap();
    let offset = io.append(&[&built.bytes]).await.unwrap();
    let (_data, meta) = built.placed(offset);

    // Hand the reader fewer bytes than the segment needs.
    let short = Extent::new(offset, built.len() / 2);
    let bytes = io.read_immutable(short).await.unwrap();
    let err = SegmentReader::new(bytes, offset, meta, schema, fingerprint).unwrap_err();
    assert!(
        matches!(err, localtables_format::Error::Corrupt(_)),
        "got {err:?}"
    );
}
