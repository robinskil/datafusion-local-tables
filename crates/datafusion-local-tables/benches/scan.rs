//! How fast a local table reads, against DataFusion's parquet reader.
//!
//! The claim this crate makes is that a local file can be read faster than one
//! behind an object store abstraction, because the segment can be mapped and
//! handed to Arrow without copying. These benchmarks are how that claim is
//! checked rather than asserted.
//!
//! The same data is written to both, and the same queries are run through the
//! same DataFusion session, so what is measured is the storage layer.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use datafusion::prelude::{SessionConfig, SessionContext};
use parquet::arrow::ArrowWriter;

use datafusion_local_tables::ColumnarTableProvider;
use localtables_format::{ColumnarTable, Durability, IoBackend, TableOptions};

const ROWS: i64 = 500_000;
const ROWS_PER_SEGMENT: i64 = 50_000;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("payload", DataType::Int64, false),
    ]))
}

/// A batch whose columns exercise different read paths: an ordered integer
/// (which zone maps prune well), a low-cardinality string (which the encoder
/// will dictionary-encode), and two wide columns.
fn batch(ids: std::ops::Range<i64>) -> RecordBatch {
    let ids: Vec<i64> = ids.collect();
    let categories: Vec<String> = ids.iter().map(|i| format!("cat-{}", i % 8)).collect();
    let values: Vec<f64> = ids.iter().map(|i| *i as f64 * 1.5).collect();
    let payload: Vec<i64> = ids.iter().map(|i| i.wrapping_mul(2_654_435_761)).collect();

    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(categories)),
            Arc::new(Float64Array::from(values)),
            Arc::new(Int64Array::from(payload)),
        ],
    )
    .unwrap()
}

fn batches() -> Vec<RecordBatch> {
    (0..ROWS / ROWS_PER_SEGMENT)
        .map(|segment| {
            let start = segment * ROWS_PER_SEGMENT;
            batch(start..start + ROWS_PER_SEGMENT)
        })
        .collect()
}

/// Write the data as a local table, one segment per batch.
///
/// `re_encode` decides whether the writer may choose dictionary or run-length
/// encoding. Those make the file smaller but make a read pay to expand them
/// back to the schema's type, so both settings are measured.
async fn build_local(dir: &std::path::Path, name: &str, re_encode: bool) -> ColumnarTable {
    let table = ColumnarTable::create(
        &dir.join(format!("{name}.lt")),
        schema(),
        TableOptions {
            durability: Durability::None,
            io_backend: IoBackend::Mmap,
            memtable_max_bytes: 64 * 1024 * 1024,
            dictionary_encoding: re_encode,
            rle_encoding: re_encode,
            ..TableOptions::default()
        },
    )
    .await
    .unwrap();

    for batch in batches() {
        table.insert(&[batch]).await.unwrap();
        // One segment per batch, so both formats have the same row-group
        // granularity to prune with.
        table.flush().await.unwrap();
    }
    table
}

/// Write the same data as parquet, with matching row groups.
fn build_parquet(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("bench.parquet");
    let file = std::fs::File::create(&path).unwrap();
    let properties = parquet::file::properties::WriterProperties::builder()
        .set_max_row_group_row_count(Some(ROWS_PER_SEGMENT as usize))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema(), Some(properties)).unwrap();
    for batch in batches() {
        writer.write(&batch).unwrap();
    }
    writer.close().unwrap();
    path
}

/// A session holding every table the benchmarks compare.
async fn session(dir: &std::path::Path) -> SessionContext {
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));

    let local = build_local(dir, "local", true).await;
    ctx.register_table("local", Arc::new(ColumnarTableProvider::new(local)))
        .unwrap();

    let plain = build_local(dir, "plain", false).await;
    ctx.register_table("plain", Arc::new(ColumnarTableProvider::new(plain)))
        .unwrap();

    let parquet = build_parquet(dir);
    ctx.register_parquet(
        "pq",
        parquet.to_str().unwrap(),
        datafusion::prelude::ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    ctx
}

fn run(ctx: &SessionContext, runtime: &tokio::runtime::Runtime, sql: &str) -> usize {
    runtime.block_on(async {
        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        batches.iter().map(|b| b.num_rows()).sum()
    })
}

/// The same data with the string column declared as a dictionary.
///
/// This is the shape that lets a group by hash indices rather than values, and
/// the point of measuring it is to see whether declaring it that way closes the
/// gap against parquet.
fn dict_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "category",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        ),
    ]))
}

fn dict_batch(ids: std::ops::Range<i64>) -> RecordBatch {
    let ids: Vec<i64> = ids.collect();
    let categories: Vec<String> = ids.iter().map(|i| format!("cat-{}", i % 8)).collect();
    let plain = StringArray::from(categories);
    let encoded = arrow::compute::cast(
        &plain,
        &DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
    )
    .unwrap();

    RecordBatch::try_new(
        dict_schema(),
        vec![Arc::new(Int64Array::from(ids)), encoded],
    )
    .unwrap()
}

/// Group by a dictionary column, in both stores.
fn dictionary_group_by(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();

    let ctx = runtime.block_on(async {
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));

        let table = ColumnarTable::create(
            &dir.path().join("dict.lt"),
            dict_schema(),
            TableOptions {
                durability: Durability::None,
                io_backend: IoBackend::Mmap,
                memtable_max_bytes: 64 * 1024 * 1024,
                // The column is already a dictionary; re-encoding it would only
                // add a layer to undo.
                dictionary_encoding: false,
                rle_encoding: false,
                ..TableOptions::default()
            },
        )
        .await
        .unwrap();

        let batches: Vec<RecordBatch> = (0..ROWS / ROWS_PER_SEGMENT)
            .map(|segment| {
                let start = segment * ROWS_PER_SEGMENT;
                dict_batch(start..start + ROWS_PER_SEGMENT)
            })
            .collect();
        for batch in &batches {
            table.insert(std::slice::from_ref(batch)).await.unwrap();
            table.flush().await.unwrap();
        }
        ctx.register_table("dict_local", Arc::new(ColumnarTableProvider::new(table)))
            .unwrap();

        let path = dir.path().join("dict.parquet");
        let file = std::fs::File::create(&path).unwrap();
        let properties = parquet::file::properties::WriterProperties::builder()
            .set_max_row_group_row_count(Some(ROWS_PER_SEGMENT as usize))
            .build();
        let mut writer = ArrowWriter::try_new(file, dict_schema(), Some(properties)).unwrap();
        for batch in &batches {
            writer.write(batch).unwrap();
        }
        writer.close().unwrap();

        ctx.register_parquet(
            "dict_pq",
            path.to_str().unwrap(),
            datafusion::prelude::ParquetReadOptions::default(),
        )
        .await
        .unwrap();

        ctx
    });

    let mut group = c.benchmark_group("group by (dictionary column)");
    group.throughput(Throughput::Elements(ROWS as u64));
    for table in ["dict_local", "dict_pq"] {
        let sql = format!("SELECT category, count(*) FROM {table} GROUP BY category");
        group.bench_function(table, |b| b.iter(|| run(&ctx, &runtime, &sql)));
    }
    group.finish();
}

/// Is parquet's advantage on a string group by about the storage, or about the
/// type DataFusion's parquet reader hands back?
///
/// `schema_force_view_types` defaults on, so that reader turns a `Utf8` column
/// into `Utf8View`, and a short string lives inline in a view rather than
/// behind an offset. This measures the same file read both ways, which is the
/// only way to tell the two explanations apart.
/// The string group by, with the column declared three ways.
///
/// Parquet's reader turns a `Utf8` column into `Utf8View`, which is where its
/// advantage on this query comes from. The format stores whatever Arrow type
/// the schema declares, so the same column can simply be declared `Utf8View`
/// and never converted at all. This measures whether that closes the gap.
fn string_group_by(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();

    let ctx = runtime.block_on(async {
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));

        for (name, string_type) in [
            ("utf8_local", DataType::Utf8),
            ("view_local", DataType::Utf8View),
        ] {
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("category", string_type.clone(), false),
            ]));
            let table = ColumnarTable::create(
                &dir.path().join(format!("{name}.lt")),
                schema.clone(),
                TableOptions {
                    durability: Durability::None,
                    io_backend: IoBackend::Mmap,
                    memtable_max_bytes: 64 * 1024 * 1024,
                    // Re-encoding a column only to expand it again on read is
                    // exactly what this measurement is about avoiding.
                    dictionary_encoding: false,
                    rle_encoding: false,
                    ..TableOptions::default()
                },
            )
            .await
            .unwrap();

            for segment in 0..ROWS / ROWS_PER_SEGMENT {
                let start = segment * ROWS_PER_SEGMENT;
                let ids: Vec<i64> = (start..start + ROWS_PER_SEGMENT).collect();
                let categories: Vec<String> =
                    ids.iter().map(|i| format!("cat-{}", i % 8)).collect();
                let plain = StringArray::from(categories);
                let column: ArrayRef = if string_type == DataType::Utf8View {
                    arrow::compute::cast(&plain, &DataType::Utf8View).unwrap()
                } else {
                    Arc::new(plain)
                };
                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int64Array::from(ids)), column],
                )
                .unwrap();
                table.insert(&[batch]).await.unwrap();
                table.flush().await.unwrap();
            }
            ctx.register_table(name, Arc::new(ColumnarTableProvider::new(table)))
                .unwrap();
        }

        let path = build_parquet(dir.path());
        ctx.register_parquet(
            "pq",
            path.to_str().unwrap(),
            datafusion::prelude::ParquetReadOptions::default(),
        )
        .await
        .unwrap();
        ctx
    });

    let mut group = c.benchmark_group("string group by");
    group.throughput(Throughput::Elements(ROWS as u64));
    for table in ["utf8_local", "view_local", "pq"] {
        let sql = format!("SELECT category, count(*) FROM {table} GROUP BY category");
        group.bench_function(table, |b| b.iter(|| run(&ctx, &runtime, &sql)));
    }
    group.finish();
}

fn parquet_view_types(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = build_parquet(dir.path());

    let with_views = runtime.block_on(async {
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
        ctx.register_parquet(
            "pq",
            path.to_str().unwrap(),
            datafusion::prelude::ParquetReadOptions::default(),
        )
        .await
        .unwrap();
        ctx
    });

    let without_views = runtime.block_on(async {
        let mut config = SessionConfig::new().with_target_partitions(4);
        config
            .options_mut()
            .execution
            .parquet
            .schema_force_view_types = false;
        let ctx = SessionContext::new_with_config(config);
        ctx.register_parquet(
            "pq",
            path.to_str().unwrap(),
            datafusion::prelude::ParquetReadOptions::default(),
        )
        .await
        .unwrap();
        ctx
    });

    let sql = "SELECT category, count(*) FROM pq GROUP BY category";
    let mut group = c.benchmark_group("parquet string group by");
    group.throughput(Throughput::Elements(ROWS as u64));
    group.bench_function("utf8view (default)", |b| {
        b.iter(|| run(&with_views, &runtime, sql))
    });
    group.bench_function("utf8", |b| b.iter(|| run(&without_views, &runtime, sql)));
    group.finish();
}

fn scans(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let ctx = runtime.block_on(session(dir.path()));

    // Each query is the same but for the table it names, so the three results
    // in a group are directly comparable.
    let queries: &[(&str, &str)] = &[
        ("full scan", "SELECT sum(payload) FROM {}"),
        ("one column", "SELECT sum(value) FROM {}"),
        ("point lookup", "SELECT * FROM {} WHERE id = 372145"),
        (
            "narrow range",
            "SELECT sum(value) FROM {} WHERE id >= 200000 AND id < 210000",
        ),
        (
            "group by",
            "SELECT category, count(*) FROM {} GROUP BY category",
        ),
    ];

    for (name, template) in queries {
        let mut group = c.benchmark_group(*name);
        group.throughput(Throughput::Elements(ROWS as u64));
        for table in ["local", "plain", "pq"] {
            let sql = template.replace("{}", table);
            group.bench_function(table, |b| b.iter(|| run(&ctx, &runtime, &sql)));
        }
        group.finish();
    }
}

/// How long it takes to make rows durable, one small insert at a time.
///
/// This is what the write-ahead log exists for: a handful of rows should cost
/// one sync, not a segment write.
fn small_writes(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("small insert");
    group.throughput(Throughput::Elements(100));
    group.bench_function("100 rows", |b| {
        b.iter_batched(
            || {
                let dir = tempfile::tempdir().unwrap();
                let table = runtime
                    .block_on(ColumnarTable::create(
                        &dir.path().join("w.lt"),
                        schema(),
                        TableOptions {
                            durability: Durability::None,
                            memtable_max_bytes: 64 * 1024 * 1024,
                            ..TableOptions::default()
                        },
                    ))
                    .unwrap();
                (dir, table)
            },
            |(dir, table)| {
                runtime.block_on(async {
                    for i in 0..100i64 {
                        table.insert(&[batch(i..i + 1)]).await.unwrap();
                    }
                });
                drop(dir);
            },
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

criterion_group!(
    benches,
    scans,
    dictionary_group_by,
    string_group_by,
    parquet_view_types,
    small_writes
);
criterion_main!(benches);
