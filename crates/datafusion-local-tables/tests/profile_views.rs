//! Why a `Utf8View` column reads slower here than a `Utf8` one.
//!
//! A measuring tool, not a regression test: it prints sizes and timings and
//! asserts almost nothing, because timing assertions on a shared machine are
//! noise. It is ignored by default. Run it with
//!
//! ```text
//! cargo test --release -p datafusion-local-tables --test profile_views -- --ignored --nocapture
//! ```
//!
//! and note the `--release`: under the dev profile the relative costs invert,
//! which is how this investigation first went wrong.
//!
//! What it establishes, on the machine it was last run on:
//!
//! * a `Utf8View` column is 1.78x the size of a `Utf8` one for five-byte
//!   values, because a view is a flat sixteen bytes per row while an offset
//!   plus five bytes of data is nine;
//! * `Utf8View` costs about twice as much to decode as its size alone explains,
//!   and `BinaryView` — the same layout, the same bytes, no UTF-8 to check —
//!   decodes 2.8x faster than `Utf8View`, which is where that cost is;
//! * views do make grouping cheaper, but only while values are short enough to
//!   live inline in the view; at forty bytes they are slower at that too.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::{SessionConfig, SessionContext};

use datafusion_local_tables::ColumnarTableProvider;
use localtables_format::{ColumnarTable, Durability, IoBackend, TableOptions};

const ROWS: i64 = 500_000;
const PER_SEGMENT: i64 = 50_000;

fn options() -> TableOptions {
    TableOptions {
        durability: Durability::None,
        io_backend: IoBackend::Mmap,
        memtable_max_bytes: 64 * 1024 * 1024,
        dictionary_encoding: false,
        rle_encoding: false,
        ..TableOptions::default()
    }
}

fn schema(string_type: DataType) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("category", string_type, false),
    ]))
}

/// Build a table whose `category` column has the given type.
///
/// `width` is how long each value is. Short values live inline in a view;
/// longer ones do not, which changes how the two layouts compare in size.
async fn build(
    dir: &std::path::Path,
    name: &str,
    string_type: DataType,
    width: usize,
) -> ColumnarTable {
    let schema = schema(string_type.clone());
    let table = ColumnarTable::create(&dir.join(format!("{name}.lt")), schema.clone(), options())
        .await
        .unwrap();

    for segment in 0..ROWS / PER_SEGMENT {
        let start = segment * PER_SEGMENT;
        let ids: Vec<i64> = (start..start + PER_SEGMENT).collect();
        let categories: Vec<String> = ids
            .iter()
            .map(|i| {
                let base = format!("cat-{}", i % 8);
                // Pad to the requested width, keeping eight distinct values.
                format!("{base:-<width$}")
            })
            .collect();
        let plain = StringArray::from(categories);
        let column: ArrayRef = if string_type == DataType::Utf8 {
            Arc::new(plain)
        } else {
            arrow::compute::cast(&plain, &string_type).unwrap()
        };
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(ids)), column],
        )
        .unwrap();
        table.insert(&[batch]).await.unwrap();
        table.flush().await.unwrap();
    }
    table
}

/// Bytes the `category` column occupies across all segments, and how it is laid
/// out in the first one.
async fn column_bytes(table: &ColumnarTable) -> (u64, String) {
    let snapshot = table.snapshot();
    let mut total = 0u64;
    let mut shape = String::new();

    for (index, entry) in snapshot.manifest.segments.iter().enumerate() {
        let reader = table.segment_reader(entry).await.unwrap();
        let meta = reader.meta().unwrap();
        let chunk = &meta.columns[1];
        total += chunk
            .buffers
            .iter()
            .map(|b| b.extent.len.to_native())
            .sum::<u64>();

        if index == 0 {
            let sizes: Vec<u64> = chunk
                .buffers
                .iter()
                .map(|b| b.extent.len.to_native())
                .collect();
            shape = format!("{} buffers {sizes:?}", chunk.buffers.len());
        }
    }
    (total, shape)
}

/// Time decoding the column straight out of the segments, with no query engine
/// involved. This separates what this crate costs from what DataFusion does
/// with the array afterwards.
async fn raw_decode_ms(table: &ColumnarTable, runs: usize) -> f64 {
    let snapshot = table.snapshot();
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let mut rows = 0usize;
        for entry in snapshot.manifest.segments.iter() {
            let reader = table.segment_reader(entry).await.unwrap();
            rows += reader.read(Some(&[1])).unwrap().num_rows();
        }
        assert_eq!(rows, ROWS as usize);
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// Median wall time over several runs, so one slow run does not decide it.
async fn time(ctx: &SessionContext, sql: &str, runs: usize) -> f64 {
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert!(rows > 0, "{sql} returned nothing");
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// Short values (inline in a view) and long ones (not), so the size difference
/// between the two layouts varies and its effect can be seen.
#[ignore = "a measuring tool; run explicitly with --ignored --nocapture --release"]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn where_does_the_view_column_lose_its_time() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));

    // 5 bytes fits inline in a view; 40 does not.
    for (label, width) in [("short", 5usize), ("long", 40)] {
        eprintln!("\n################ {label} values ({width} bytes) ################");

        let mut sizes = Vec::new();
        let mut decodes = Vec::new();
        for (kind, string_type) in [
            ("utf8", DataType::Utf8),
            ("view", DataType::Utf8View),
            // Same layout as Utf8View, but Arrow does not have to check that
            // every value is valid UTF-8 when it rebuilds the array. If this
            // decodes much cheaper than the view, that check is the cost.
            ("bview", DataType::BinaryView),
            // And the same question for the offset layout.
            ("binary", DataType::Binary),
        ] {
            let name = format!("{kind}_{label}");
            let table = build(dir.path(), &name, string_type, width).await;
            let (bytes, shape) = column_bytes(&table).await;
            let decode = raw_decode_ms(&table, 9).await;
            eprintln!("  {kind:5} stored {bytes:>9} bytes   decode {decode:6.3} ms   {shape}");
            sizes.push(bytes);
            decodes.push(decode);
            ctx.register_table(&name, Arc::new(ColumnarTableProvider::new(table)))
                .unwrap();
        }
        eprintln!(
            "  relative to utf8 — view: {:.2}x size {:.2}x decode | \
             binary-view: {:.2}x size {:.2}x decode | binary: {:.2}x size {:.2}x decode",
            sizes[1] as f64 / sizes[0] as f64,
            decodes[1] / decodes[0],
            sizes[2] as f64 / sizes[0] as f64,
            decodes[2] / decodes[0],
            sizes[3] as f64 / sizes[0] as f64,
            decodes[3] / decodes[0],
        );

        eprintln!("  timings (median of 9, ms):");
        let mut reads = Vec::new();
        for kind in ["utf8", "view"] {
            let name = format!("{kind}_{label}");
            // Reads the id column only: the baseline both share.
            let id_only = time(&ctx, &format!("SELECT count(id) FROM {name}"), 9).await;
            // Reads the string column with almost no work done to it.
            let scan = time(&ctx, &format!("SELECT count(category) FROM {name}"), 9).await;
            // Reads it and groups by it.
            let group = time(
                &ctx,
                &format!("SELECT category, count(*) FROM {name} GROUP BY category"),
                9,
            )
            .await;
            reads.push(scan - id_only);
            eprintln!(
                "    {kind:5} read column {:6.3}   grouping {:6.3}   total {group:6.3}",
                scan - id_only,
                group - scan
            );
        }
        eprintln!(
            "  reading the view column costs {:.2}x reading the utf8 one",
            reads[1] / reads[0]
        );
    }
}
