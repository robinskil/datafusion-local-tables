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
use localtables_format::{
    BloomFilters, ColumnarTable, Compression, Durability, IoBackend, TableOptions,
};

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







/// What compression and blocking cost to *write*.
///
/// `build_segment` is the whole encode path with no file in it: choose an
/// encoding per column, build the zone maps and page bounds, compress, and lay
/// the bytes out. Everything a flush does before it touches the disk.
///
/// Bytes written are reported alongside, because a codec that costs processor
/// time also hands the disk less to do.
fn write_cost(c: &mut Criterion) {
    use localtables_format::columnar::segment::build_segment;
    use localtables_format::layout::schema::SchemaLayout;

    let text_schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("body", DataType::Utf8, false),
    ]));
    let numeric_schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ]));

    let text = RecordBatch::try_new(
        text_schema.clone(),
        vec![
            Arc::new(Int64Array::from((0..ROWS).collect::<Vec<i64>>())),
            Arc::new(StringArray::from(
                (0..ROWS)
                    .map(|i| format!("user{i}.{}@mail-{}.example.com", i * 7 % 100_000, i % 32))
                    .collect::<Vec<String>>(),
            )),
        ],
    )
    .unwrap();
    let numeric = RecordBatch::try_new(
        numeric_schema.clone(),
        vec![
            Arc::new(Int64Array::from((0..ROWS).collect::<Vec<i64>>())),
            Arc::new(Float64Array::from(
                (0..ROWS).map(|i| i as f64 * 1.5).collect::<Vec<f64>>(),
            )),
        ],
    )
    .unwrap();

    let settings: Vec<(&str, Compression, usize)> = vec![
        ("raw, unblocked", Compression::None, 0),
        ("raw, blocked", Compression::None, 8192),
        ("lz4, blocked", Compression::Lz4, 8192),
        ("lz4, unblocked", Compression::Lz4, 0),
        ("zstd, blocked", Compression::Zstd, 8192),
        ("zstd, unblocked", Compression::Zstd, 0),
        ("zstd, 512-row blocks", Compression::Zstd, 512),
    ];

    for (label, schema, batch) in [
        ("text", &text_schema, &text),
        ("numbers", &numeric_schema, &numeric),
    ] {
        let layout = SchemaLayout::of(schema);
        let mut group = c.benchmark_group(format!("write: {label}"));
        group.throughput(Throughput::Elements(ROWS as u64));
        for (name, compression, block_rows) in &settings {
            let options = TableOptions {
                durability: Durability::None,
                compression: *compression,
                compression_block_rows: *block_rows,
                ..TableOptions::default()
            };
            // Report what a flush would hand the disk, so the processor cost
            // and the bytes it saves can be read together.
            let built =
                build_segment(0, schema, layout.current(), std::slice::from_ref(batch), &options)
                    .unwrap();
            println!("write {label:>8} {name:<22} {:>6} KiB", built.bytes.len() / 1024);

            group.bench_function(*name, |b| {
                b.iter(|| {
                    build_segment(
                        0,
                        schema,
                        layout.current(),
                        std::slice::from_ref(batch),
                        &options,
                    )
                    .unwrap()
                })
            });
        }
        group.finish();
    }
}

/// What a smaller compression block buys, and what it costs, on a table whose
/// bulk is text that does not dictionary encode.
///
/// The benchmark fixture elsewhere in this file has only a low-cardinality
/// string column, which is stored as a dictionary and so has a tiny values
/// buffer; it cannot show any of this. Here every value is distinct.
fn page_size_tradeoff(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();

    let text_schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("body", DataType::Utf8, false),
    ]));
    let make = |sorted: bool| {
        let mut bodies: Vec<String> = (0..ROWS)
            .map(|i| format!("user{i}.{}@mail-{}.example.com", i * 7 % 100_000, i % 32))
            .collect();
        if sorted {
            bodies.sort();
        }
        RecordBatch::try_new(
            text_schema.clone(),
            vec![
                Arc::new(Int64Array::from((0..ROWS).collect::<Vec<i64>>())),
                Arc::new(StringArray::from(bodies)),
            ],
        )
        .unwrap()
    };

    let build = |name: String, compression: Compression, block_rows: usize, sorted: bool| {
        let schema = text_schema.clone();
        let path = dir.path().join(format!("{name}.lt"));
        let batch = make(sorted);
        runtime.block_on(async move {
            let table = ColumnarTable::create(
                &path,
                schema,
                TableOptions {
                    durability: Durability::None,
                    io_backend: IoBackend::Mmap,
                    memtable_max_bytes: 512 * 1024 * 1024,
                    compression,
                    compression_block_rows: block_rows,
                    // One segment, so only blocks decide what is decompressed.
                    min_row_group_rows: ROWS as usize,
                    row_group_rows: ROWS as usize,
                    ..TableOptions::default()
                },
            )
            .await
            .unwrap();
            table.insert(&[batch]).await.unwrap();
            table.flush().await.unwrap();
            table
        })
    };

    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));

    let cases: Vec<(String, Compression, usize, bool)> = vec![
        ("raw_unblocked".to_string(), Compression::None, 0, false),
        ("raw".to_string(), Compression::None, 8192, false),
        ("lz4".to_string(), Compression::Lz4, 8192, false),
        ("zstd_whole".to_string(), Compression::Zstd, 0, false),
        ("zstd_8192".to_string(), Compression::Zstd, 8192, false),
        ("zstd_2048".to_string(), Compression::Zstd, 2048, false),
        ("zstd_512".to_string(), Compression::Zstd, 512, false),
        ("zstd_8192_sorted".to_string(), Compression::Zstd, 8192, true),
    ];
    for (name, compression, block_rows, sorted) in &cases {
        ctx.register_table(
            name.as_str(),
            Arc::new(ColumnarTableProvider::new(build(
                name.clone(),
                *compression,
                *block_rows,
                *sorted,
            ))),
        )
        .unwrap();
        let size = std::fs::metadata(dir.path().join(format!("{name}.lt")))
            .unwrap()
            .len();
        println!("{name:>18} {:>6} KiB", size / 1024);
    }

    // `id only` reads no text, so the gap between it and `one row` is what
    // building the string array costs.
    for (label, sql) in [
        ("one row", "SELECT * FROM {} WHERE id = 372145"),
        ("one row, id only", "SELECT id FROM {} WHERE id = 372145"),
        ("every row", "SELECT max(character_length(body)) FROM {}"),
    ] {
        let mut group = c.benchmark_group(format!("block size: {label}"));
        for (name, ..) in &cases {
            let sql = sql.replace("{}", name);
            group.bench_function(name.as_str(), |b| b.iter(|| run(&ctx, &runtime, &sql)));
        }
        group.finish();
    }
}

/// What compressing a column actually costs a query, and what it saves.
///
/// The read path is zero-copy only while a column is stored raw, so a codec is
/// never free: it trades the mapped bytes for a buffer the reader owns and a
/// pass to fill it. Auto compresses text and binary and leaves everything else
/// alone, which is the shape the codec measurements point at.
fn compression_choice(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();

    let build = |name: &'static str, compression: Compression, block_rows: usize| {
        let path = dir.path().join(format!("{name}.lt"));
        runtime.block_on(async move {
            let table = ColumnarTable::create(
                &path,
                schema(),
                TableOptions {
                    durability: Durability::None,
                    io_backend: IoBackend::Mmap,
                    memtable_max_bytes: 256 * 1024 * 1024,
                    compression,
                    compression_block_rows: block_rows,
                    ..TableOptions::default()
                },
            )
            .await
            .unwrap();
            for batch in batches() {
                table.insert(&[batch]).await.unwrap();
                table.flush().await.unwrap();
            }
            table
        })
    };

    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    // `whole` compresses each column as one block, which is what the format did
    // before blocks existed; the others cut it so a range can be decompressed
    // on its own.
    for (name, compression, block_rows) in [
        ("raw", Compression::None, 0),
        ("auto", Compression::Auto, 8 * 1024),
        ("auto_whole", Compression::Auto, 0),
        ("lz4", Compression::Lz4, 8 * 1024),
        ("zstd", Compression::Zstd, 8 * 1024),
        ("zstd_whole", Compression::Zstd, 0),
    ] {
        ctx.register_table(
            name,
            Arc::new(ColumnarTableProvider::new(build(name, compression, block_rows))),
        )
        .unwrap();
        let size: u64 = std::fs::metadata(dir.path().join(format!("{name}.lt")))
            .unwrap()
            .len();
        println!("{name:>5} file {:>6} KiB", size / 1024);
    }

    for (label, sql) in [
        ("full scan", "SELECT sum(payload) FROM {}"),
        ("string group by", "SELECT category, count(*) FROM {} GROUP BY category"),
        ("point lookup", "SELECT * FROM {} WHERE id = 372145"),
    ] {
        let mut group = c.benchmark_group(format!("compression: {label}"));
        for table in ["raw", "auto", "auto_whole", "lz4", "zstd", "zstd_whole"] {
            let sql = sql.replace("{}", table);
            group.bench_function(table, |b| b.iter(|| run(&ctx, &runtime, &sql)));
        }
        group.finish();
    }
}

/// A point lookup with nothing but page bounds to prune with.
///
/// The whole table is one segment, so its zone map rules nothing out and every
/// row would otherwise be handed to the filter above. That is the regime page
/// bounds exist for. Parquet's own page index does the same job, so it is
/// measured with the feature on and off too.
fn page_pruning(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();

    let build = |name: &'static str, page_rows: usize, re_encode: bool| {
        let path = dir.path().join(format!("{name}.lt"));
        runtime.block_on(async move {
            let table = ColumnarTable::create(
                &path,
                schema(),
                TableOptions {
                    durability: Durability::None,
                    io_backend: IoBackend::Mmap,
                    memtable_max_bytes: 512 * 1024 * 1024,
                    // One segment holding everything.
                    min_row_group_rows: ROWS as usize,
                    row_group_rows: ROWS as usize,
                    page_rows,
                    dictionary_encoding: re_encode,
                    rle_encoding: re_encode,
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
        })
    };

    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    // With and without page bounds, and with and without the encodings that
    // make a column cost something to decode. A zero-copy column is nearly free
    // to build whether or not its rows are wanted; a dictionary column is
    // expanded for the whole segment before any of it is sliced.
    for (name, page_rows, re_encode) in [
        ("no_pages", 0, true),
        ("pages", 8 * 1024, true),
        ("no_pages_plain", 0, false),
        ("pages_plain", 8 * 1024, false),
    ] {
        ctx.register_table(
            name,
            Arc::new(ColumnarTableProvider::new(build(name, page_rows, re_encode))),
        )
        .unwrap();
    }

    for (name, index) in [("pq", false), ("pq_index", true)] {
        let path = dir.path().join(format!("{name}.parquet"));
        let file = std::fs::File::create(&path).unwrap();
        let properties = parquet::file::properties::WriterProperties::builder()
            .set_max_row_group_row_count(Some(ROWS as usize))
            .set_statistics_enabled(if index {
                parquet::file::properties::EnabledStatistics::Page
            } else {
                parquet::file::properties::EnabledStatistics::Chunk
            })
            .build();
        let mut writer = ArrowWriter::try_new(file, schema(), Some(properties)).unwrap();
        for batch in batches() {
            writer.write(&batch).unwrap();
        }
        writer.close().unwrap();
        runtime
            .block_on(ctx.register_parquet(
                name,
                path.to_str().unwrap(),
                datafusion::prelude::ParquetReadOptions::default(),
            ))
            .unwrap();
    }

    let mut group = c.benchmark_group("point lookup inside a segment");
    for table in [
        "no_pages",
        "pages",
        "no_pages_plain",
        "pages_plain",
        "pq",
        "pq_index",
    ] {
        let sql = format!("SELECT * FROM {table} WHERE id = 372145");
        group.bench_function(table, |b| b.iter(|| run(&ctx, &runtime, &sql)));
    }
    group.finish();
}

/// A query on a column the insert order does not follow.
///
/// Rows arrive ordered by `y`, so a zone map prunes `y` perfectly and `x` not
/// at all. Clustering interleaves the two, which costs `y` some of its
/// selectivity and gives `x` most of what it lacked.
fn clustered_layout(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();

    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, false),
        Field::new("y", DataType::Int64, false),
        Field::new("payload", DataType::Int64, false),
    ]));

    // A 707 by 707 grid is close to ROWS, written one row of it at a time.
    let side = 707i64;
    let make_batch = |y: i64| {
        let xs: Vec<i64> = (0..side).collect();
        let ys = vec![y; side as usize];
        let payload: Vec<i64> = xs.iter().map(|x| x * side + y).collect();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(xs)),
                Arc::new(Int64Array::from(ys)),
                Arc::new(Int64Array::from(payload)),
            ],
        )
        .unwrap()
    };

    let build = |name: &'static str, cluster: Vec<String>| {
        let schema = schema.clone();
        let path = dir.path().join(format!("{name}.lt"));
        runtime.block_on(async move {
            let table = ColumnarTable::create(
                &path,
                schema,
                TableOptions {
                    durability: Durability::None,
                    io_backend: IoBackend::Mmap,
                    memtable_max_bytes: 256 * 1024 * 1024,
                    cluster_by: cluster,
                    ..TableOptions::default()
                },
            )
            .await
            .unwrap();
            let batches: Vec<RecordBatch> = (0..side).map(make_batch).collect();
            table.insert(&batches).await.unwrap();
            table.flush().await.unwrap();
            table
        })
    };

    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    ctx.register_table(
        "plain",
        Arc::new(ColumnarTableProvider::new(build("plain", Vec::new()))),
    )
    .unwrap();
    ctx.register_table(
        "clustered",
        Arc::new(ColumnarTableProvider::new(build(
            "clustered",
            vec!["x".to_string(), "y".to_string()],
        ))),
    )
    .unwrap();

    for (name, predicate) in [
        ("x only", "x = 354"),
        ("y only", "y = 354"),
        ("both", "x = 354 AND y = 354"),
    ] {
        let mut group = c.benchmark_group(format!("clustered scan ({name})"));
        for table in ["plain", "clustered"] {
            let sql = format!("SELECT count(*) FROM {table} WHERE {predicate}");
            group.bench_function(table, |b| b.iter(|| run(&ctx, &runtime, &sql)));
        }
        group.finish();
    }
}

/// A substring search over a text column.
///
/// Zone maps cannot prune this at all: a minimum and a maximum say nothing
/// about what a value contains. Parquet has no substring index either, so it is
/// here as the reference for what reading everything costs.
fn substring_search(c: &mut Criterion) {
    let dir = tempfile::tempdir().unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();

    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("body", DataType::Utf8, false),
    ]));

    // Each segment gets a term of its own, so one segment holds the match and
    // the rest hold text that merely looks similar.
    let make_batch = |segment: i64| {
        let ids: Vec<i64> = (0..ROWS_PER_SEGMENT)
            .map(|r| segment * ROWS_PER_SEGMENT + r)
            .collect();
        let bodies: Vec<String> = ids
            .iter()
            .map(|i| format!("shard{segment} record{i} common filler text for padding"))
            .collect();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(bodies)),
            ],
        )
        .unwrap()
    };
    let all_batches = || (0..ROWS / ROWS_PER_SEGMENT).map(make_batch).collect::<Vec<_>>();

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
                    memtable_max_bytes: 256 * 1024 * 1024,
                    trigram_filters: filters,
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

    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    ctx.register_table(
        "no_filter",
        Arc::new(ColumnarTableProvider::new(build("no_filter", BloomFilters::None))),
    )
    .unwrap();
    ctx.register_table(
        "filter",
        Arc::new(ColumnarTableProvider::new(build(
            "filter",
            BloomFilters::Columns(vec!["body".to_string()]),
        ))),
    )
    .unwrap();

    let path = dir.path().join("substring.parquet");
    let file = std::fs::File::create(&path).unwrap();
    let properties = parquet::file::properties::WriterProperties::builder()
        .set_max_row_group_row_count(Some(ROWS_PER_SEGMENT as usize))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(properties)).unwrap();
    for batch in all_batches() {
        writer.write(&batch).unwrap();
    }
    writer.close().unwrap();
    runtime
        .block_on(ctx.register_parquet(
            "pq",
            path.to_str().unwrap(),
            datafusion::prelude::ParquetReadOptions::default(),
        ))
        .unwrap();

    let mut group = c.benchmark_group("substring search");
    for table in ["no_filter", "filter", "pq"] {
        let sql = format!("SELECT count(*) FROM {table} WHERE body LIKE '%shard7 record%'");
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
    page_pruning,
    compression_choice,
    page_size_tradeoff,
    write_cost,
    substring_search,
    clustered_layout,
    small_writes
);
criterion_main!(benches);
