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
use localtables_format::{BloomFilters, ColumnarTable, Durability, IoBackend, TableOptions};

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

/// What does dividing a table into more segments cost, and what does it buy?
///
/// A segment is the unit a scan hands to a partition, so more of them means
/// more to divide — and more per-segment work: a mapping, a metadata frame to
/// check, a set of zone maps. Both sides are measured in one run, because
/// timings on a shared machine drift far too much between runs to compare.
fn parallel_scan(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();

    // The same 500k rows, written in one flush, cut into different numbers of
    // segments by pinning the row group size.
    let tables: Vec<(usize, ColumnarTable)> = [4usize, 8, 16, 64]
        .into_iter()
        .map(|wanted| {
            let rows_per_group = (ROWS as usize).div_ceil(wanted);
            let table = runtime.block_on(async {
                let table = ColumnarTable::create(
                    &dir.path().join(format!("seg{wanted}.lt")),
                    schema(),
                    TableOptions {
                        durability: Durability::None,
                        io_backend: IoBackend::Mmap,
                        memtable_max_bytes: 512 * 1024 * 1024,
                        row_group_rows: rows_per_group,
                        min_row_group_rows: rows_per_group,
                        ..TableOptions::default()
                    },
                )
                .await
                .unwrap();
                for batch in batches() {
                    table.insert(&[batch]).await.unwrap();
                }
                table.flush().await.unwrap();
                table
            });
            // Batches are kept whole inside a group, so a group can close under
            // the limit and the count land near what was asked for rather than
            // on it. The label reports what was actually built.
            let segments = table.snapshot().manifest.segments.len();
            (segments, table)
        })
        .collect();

    for threads in [1usize, 4, 8] {
        let mut group = c.benchmark_group(format!("scan with {threads} partitions"));
        group.throughput(Throughput::Elements(ROWS as u64));
        for (segments, table) in &tables {
            let ctx = SessionContext::new_with_config(
                SessionConfig::new().with_target_partitions(threads),
            );
            ctx.register_table("t", Arc::new(ColumnarTableProvider::new(table.clone())))
                .unwrap();
            group.bench_function(format!("{segments} segments"), |b| {
                b.iter(|| run(&ctx, &runtime, "SELECT sum(payload) FROM t"))
            });
        }
        group.finish();
    }
}

/// Uneven segments, which is what taking work from a shared queue is for.
///
/// One segment holding most of the rows and many holding few is the shape a
/// static split handles worst: whichever partition draws the big one is still
/// working when the others have finished. Taking from a queue cannot fix the
/// big segment itself — it is one piece of work either way — but it stops the
/// small ones from being dealt out badly on top of it.
fn skewed_scan(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();

    let table = runtime.block_on(async {
        let table = ColumnarTable::create(
            &dir.path().join("skew.lt"),
            schema(),
            TableOptions {
                durability: Durability::None,
                io_backend: IoBackend::Mmap,
                memtable_max_bytes: 512 * 1024 * 1024,
                // Pinned, so the shape below is what the flushes produce.
                row_group_rows: ROWS as usize,
                min_row_group_rows: 1,
                ..TableOptions::default()
            },
        )
        .await
        .unwrap();

        // Half the rows in one segment, the rest in twenty small ones.
        table.insert(&[batch(0..ROWS / 2)]).await.unwrap();
        table.flush().await.unwrap();
        let small = (ROWS / 2) / 20;
        for i in 0..20 {
            let start = ROWS / 2 + i * small;
            table.insert(&[batch(start..start + small)]).await.unwrap();
            table.flush().await.unwrap();
        }
        table
    });

    let mut group = c.benchmark_group("skewed segments");
    group.throughput(Throughput::Elements(ROWS as u64));
    for threads in [1usize, 4, 8] {
        let ctx =
            SessionContext::new_with_config(SessionConfig::new().with_target_partitions(threads));
        ctx.register_table("t", Arc::new(ColumnarTableProvider::new(table.clone())))
            .unwrap();
        group.bench_function(format!("{threads} partitions"), |b| {
            b.iter(|| run(&ctx, &runtime, "SELECT sum(payload) FROM t"))
        });
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

/// Point lookups on a column whose values are scattered.
///
/// The `scans` fixture uses an ordered id, which zone maps already prune down
/// to one segment, so a membership filter has nothing left to do there. This
/// one writes a permutation of the same range instead: every segment holds keys
/// from across the whole table, every segment's minimum and maximum span it,
/// and no range test rules any of them out. It is the case membership filters
/// exist for, measured against parquet with its own filters switched on.
fn scattered_point_lookup(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();

    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("payload", DataType::Int64, false),
    ]));

    // Knuth's multiplier shares no factor with ROWS, so this is a permutation:
    // every key in 0..ROWS appears exactly once, in scattered order.
    let scattered = |i: i64| i.wrapping_mul(2_654_435_761).rem_euclid(ROWS);

    let make_batch = |range: std::ops::Range<i64>| {
        let keys: Vec<i64> = range.map(scattered).collect();
        let values: Vec<f64> = keys.iter().map(|k| *k as f64 * 1.5).collect();
        let payload: Vec<i64> = keys.iter().map(|k| k.wrapping_mul(7)).collect();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(keys)),
                Arc::new(Float64Array::from(values)),
                Arc::new(Int64Array::from(payload)),
            ],
        )
        .unwrap()
    };
    let all_batches = || {
        (0..ROWS / ROWS_PER_SEGMENT)
            .map(|segment| {
                let start = segment * ROWS_PER_SEGMENT;
                make_batch(start..start + ROWS_PER_SEGMENT)
            })
            .collect::<Vec<_>>()
    };

    let build = |name: &'static str, filters: BloomFilters| {
        let schema = schema.clone();
        let path = dir.path().join(format!("{name}.lt"));
        runtime.block_on(async move {
            let table = ColumnarTable::create(
                &path,
                schema,
                TableOptions {
                    durability: Durability::None,
                    io_backend: IoBackend::Mmap,
                    memtable_max_bytes: 64 * 1024 * 1024,
                    bloom_filters: filters,
                    ..TableOptions::default()
                },
            )
            .await
            .unwrap();
            for batch in all_batches() {
                table.insert(&[batch]).await.unwrap();
                table.flush().await.unwrap();
            }
            table
        })
    };

    let build_pq = |name: &str, blooms: bool| {
        let path = dir.path().join(format!("{name}.parquet"));
        let file = std::fs::File::create(&path).unwrap();
        let properties = parquet::file::properties::WriterProperties::builder()
            .set_max_row_group_row_count(Some(ROWS_PER_SEGMENT as usize))
            .set_bloom_filter_enabled(blooms)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(properties)).unwrap();
        for batch in all_batches() {
            writer.write(&batch).unwrap();
        }
        writer.close().unwrap();
        path
    };

    // Every variant is built in one process, so they are compared against each
    // other and never against a number from another run.
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_target_partitions(4)
            .set_bool("datafusion.execution.parquet.bloom_filter_on_read", true),
    );
    ctx.register_table(
        "no_filter",
        Arc::new(ColumnarTableProvider::new(build("no_filter", BloomFilters::None))),
    )
    .unwrap();
    ctx.register_table(
        "filter",
        Arc::new(ColumnarTableProvider::new(build(
            "filter",
            BloomFilters::Columns(vec!["key".to_string()]),
        ))),
    )
    .unwrap();
    for (name, blooms) in [("pq", false), ("pq_filter", true)] {
        let path = build_pq(name, blooms);
        runtime
            .block_on(ctx.register_parquet(
                name,
                path.to_str().unwrap(),
                datafusion::prelude::ParquetReadOptions::default(),
            ))
            .unwrap();
    }

    let mut group = c.benchmark_group("scattered point lookup");
    for table in ["no_filter", "filter", "pq", "pq_filter"] {
        let sql = format!("SELECT * FROM {table} WHERE key = 372145");
        group.bench_function(table, |b| b.iter(|| run(&ctx, &runtime, &sql)));
    }
    group.finish();
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
    parallel_scan,
    skewed_scan,
    parquet_view_types,
    scattered_point_lookup,
    small_writes
);
criterion_main!(benches);
