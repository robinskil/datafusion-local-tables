//! Reading a local table through SQL.
//!
//! These run real queries against a real file. They check both that the answers
//! are right and that the scan did the work it claims: pruned the segments a
//! predicate rules out, read only the projected columns, and stopped at a
//! limit.

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::{SessionConfig, SessionContext};

use datafusion_local_tables::ColumnarTableProvider;
use localtables_format::columnar::table::ColumnarTable;
use localtables_format::config::{Durability, IoBackend, TableOptions};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("score", DataType::Int32, true),
    ]))
}

fn batch(ids: std::ops::Range<i64>) -> RecordBatch {
    let ids: Vec<i64> = ids.collect();
    let names: Vec<Option<String>> = ids
        .iter()
        .map(|i| (i % 7 != 0).then(|| format!("name-{i}")))
        .collect();
    let scores: Vec<Option<i32>> = ids.iter().map(|i| Some((*i % 100) as i32)).collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(Int32Array::from(scores)),
        ],
    )
    .unwrap()
}

fn options() -> TableOptions {
    TableOptions {
        durability: Durability::None,
        io_backend: IoBackend::Mmap,
        // Large, so a test controls flushes itself rather than tripping one.
        memtable_max_bytes: 64 * 1024 * 1024,
        ..TableOptions::default()
    }
}

/// A table with `segments` flushed segments of 100 rows each, plus `in_memory`
/// unflushed rows after them.
async fn table(
    dir: &tempfile::TempDir,
    segments: i64,
    in_memory: i64,
) -> (ColumnarTable, SessionContext) {
    let table = ColumnarTable::create(&dir.path().join("t.lt"), schema(), options())
        .await
        .unwrap();

    for segment in 0..segments {
        let start = segment * 100;
        table.insert(&[batch(start..start + 100)]).await.unwrap();
        table.flush().await.unwrap();
    }
    if in_memory > 0 {
        let start = segments * 100;
        table
            .insert(&[batch(start..start + in_memory)])
            .await
            .unwrap();
    }

    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    ctx.register_table("t", Arc::new(ColumnarTableProvider::new(table.clone())))
        .unwrap();
    (table, ctx)
}

async fn query(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    ctx.sql(sql).await.unwrap().collect().await.unwrap()
}

fn count(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

fn ids(batches: &[RecordBatch]) -> Vec<i64> {
    let mut out: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    out.sort_unstable();
    out
}

/// The single scalar a one-row, one-column result holds.
fn scalar_i64(batches: &[RecordBatch]) -> i64 {
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0)
}

#[tokio::test]
async fn a_select_returns_every_row() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, 3, 0).await;

    let rows = query(&ctx, "SELECT * FROM t").await;
    assert_eq!(count(&rows), 300);
    assert_eq!(ids(&rows), (0..300).collect::<Vec<i64>>());
}

#[tokio::test]
async fn rows_still_in_memory_are_part_of_the_answer() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = table(&dir, 2, 50).await;
    assert_eq!(table.memtable_rows().await, 50);

    let rows = query(&ctx, "SELECT * FROM t").await;
    assert_eq!(count(&rows), 250);
    assert_eq!(ids(&rows), (0..250).collect::<Vec<i64>>());

    assert_eq!(
        scalar_i64(&query(&ctx, "SELECT count(*) FROM t").await),
        250
    );
}

#[tokio::test]
async fn a_filter_returns_only_matching_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, 5, 30).await;

    let rows = query(&ctx, "SELECT id FROM t WHERE id >= 250 AND id < 260").await;
    assert_eq!(ids(&rows), (250..260).collect::<Vec<i64>>());
}

#[tokio::test]
async fn a_filter_prunes_the_segments_it_cannot_match() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, 10, 0).await;

    // Ids run 0..1000 in ten segments of 100. Only one holds 550.
    let plan = ctx
        .sql("EXPLAIN SELECT * FROM t WHERE id = 550")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let text = plan_text(&plan);

    assert!(
        text.contains("pruned=9"),
        "one segment can hold id 550, so nine should have been pruned:\n{text}"
    );
    assert_eq!(
        count(&query(&ctx, "SELECT * FROM t WHERE id = 550").await),
        1
    );
}

#[tokio::test]
async fn a_filter_that_matches_nothing_prunes_every_segment() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, 6, 0).await;

    let plan = ctx
        .sql("EXPLAIN SELECT * FROM t WHERE id > 100000")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert!(
        plan_text(&plan).contains("segments=0"),
        "no segment can match, so none should be read:\n{}",
        plan_text(&plan)
    );
    assert!(query(&ctx, "SELECT * FROM t WHERE id > 100000")
        .await
        .iter()
        .all(|b| b.num_rows() == 0));
}

#[tokio::test]
async fn a_filter_that_spans_everything_prunes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, 4, 0).await;

    let plan = ctx
        .sql("EXPLAIN SELECT * FROM t WHERE id >= 0")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert!(
        plan_text(&plan).contains("pruned=0"),
        "every segment can match:\n{}",
        plan_text(&plan)
    );
}

#[tokio::test]
async fn pruning_never_drops_a_matching_row() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, 8, 40).await;

    // Every id from 0 to 839, asked for one segment's worth at a time.
    for start in (0..840).step_by(37) {
        let end = start + 37;
        let sql = format!("SELECT id FROM t WHERE id >= {start} AND id < {end}");
        let rows = query(&ctx, &sql).await;
        let expected: Vec<i64> = (start..end.min(840)).collect();
        assert_eq!(ids(&rows), expected, "{sql}");
    }
}

#[tokio::test]
async fn a_projection_reads_only_the_named_columns() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, 3, 0).await;

    let plan = ctx
        .sql("EXPLAIN SELECT name FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let text = plan_text(&plan);
    assert!(
        text.contains("projection=[name]"),
        "the scan should read one column:\n{text}"
    );

    let rows = query(&ctx, "SELECT name FROM t").await;
    assert_eq!(count(&rows), 300);
    assert_eq!(rows[0].num_columns(), 1);
}

#[tokio::test]
async fn a_limit_stops_the_scan() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, 10, 0).await;

    for limit in [1usize, 5, 100, 250, 1000, 5000] {
        let rows = query(&ctx, &format!("SELECT id FROM t LIMIT {limit}")).await;
        assert_eq!(count(&rows), limit.min(1000), "limit {limit}");
    }
}

#[tokio::test]
async fn aggregates_run_over_the_whole_table() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, 4, 25).await;

    assert_eq!(
        scalar_i64(&query(&ctx, "SELECT count(*) FROM t").await),
        425
    );
    assert_eq!(scalar_i64(&query(&ctx, "SELECT min(id) FROM t").await), 0);
    assert_eq!(scalar_i64(&query(&ctx, "SELECT max(id) FROM t").await), 424);
    assert_eq!(
        scalar_i64(&query(&ctx, "SELECT sum(id) FROM t").await),
        (0..425i64).sum::<i64>()
    );
}

#[tokio::test]
async fn nulls_survive_the_trip_through_sql() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, 2, 0).await;

    // Every seventh id has no name.
    let expected = (0..200i64).filter(|i| i % 7 == 0).count();
    assert_eq!(
        scalar_i64(&query(&ctx, "SELECT count(*) FROM t WHERE name IS NULL").await) as usize,
        expected
    );

    let rows = query(&ctx, "SELECT name FROM t WHERE id = 7").await;
    let names = rows[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert!(names.is_null(0));
}

#[tokio::test]
async fn a_query_sees_the_table_as_it_stood_when_it_started() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = table(&dir, 2, 0).await;

    // The snapshot is pinned when the physical plan is built, not when the SQL
    // is parsed: `sql` only produces a logical plan, which names the table
    // rather than any particular commit of it.
    let physical = ctx
        .sql("SELECT count(*) FROM t")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();

    table.insert(&[batch(1000..1100)]).await.unwrap();
    table.flush().await.unwrap();

    let rows = datafusion::physical_plan::collect(physical, ctx.task_ctx())
        .await
        .unwrap();
    assert_eq!(
        scalar_i64(&rows),
        200,
        "the plan pinned the table before the write, so running it must not see it"
    );

    // A new query does see it.
    assert_eq!(
        scalar_i64(&query(&ctx, "SELECT count(*) FROM t").await),
        300
    );
}

#[tokio::test]
async fn an_empty_table_answers_queries() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, 0, 0).await;

    assert_eq!(scalar_i64(&query(&ctx, "SELECT count(*) FROM t").await), 0);
    assert_eq!(count(&query(&ctx, "SELECT * FROM t").await), 0);
    assert_eq!(count(&query(&ctx, "SELECT * FROM t WHERE id > 5").await), 0);
}

#[tokio::test]
async fn a_join_between_two_local_tables_works() {
    let dir = tempfile::tempdir().unwrap();
    let (_left, ctx) = table(&dir, 2, 0).await;

    let other = ColumnarTable::create(&dir.path().join("u.lt"), schema(), options())
        .await
        .unwrap();
    other.insert(&[batch(50..150)]).await.unwrap();
    other.flush().await.unwrap();
    ctx.register_table("u", Arc::new(ColumnarTableProvider::new(other)))
        .unwrap();

    let rows = query(&ctx, "SELECT t.id FROM t JOIN u ON t.id = u.id").await;
    assert_eq!(ids(&rows), (50..150).collect::<Vec<i64>>());
}

#[tokio::test]
async fn a_table_reopened_after_a_restart_queries_the_same() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    {
        let table = ColumnarTable::create(&path, schema(), options())
            .await
            .unwrap();
        table.insert(&[batch(0..100)]).await.unwrap();
        table.flush().await.unwrap();
        // These stay in the log: the restart has to replay them.
        table.insert(&[batch(100..150)]).await.unwrap();
    }

    let table = ColumnarTable::open(&path, options()).await.unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(ColumnarTableProvider::new(table)))
        .unwrap();

    assert_eq!(
        scalar_i64(&query(&ctx, "SELECT count(*) FROM t").await),
        150
    );
    assert_eq!(
        ids(&query(&ctx, "SELECT id FROM t").await),
        (0..150).collect::<Vec<i64>>()
    );
}

#[tokio::test]
async fn work_is_spread_across_partitions() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, 8, 0).await;

    let plan = ctx
        .sql("EXPLAIN SELECT * FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let text = plan_text(&plan);
    assert!(
        text.contains("partitions=4"),
        "eight segments and four target partitions should give four:\n{text}"
    );
}

#[tokio::test]
async fn fewer_segments_than_partitions_leaves_no_empty_partition() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, 2, 0).await;

    let plan = ctx
        .sql("EXPLAIN SELECT * FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let text = plan_text(&plan);
    assert!(
        text.contains("partitions=2"),
        "two segments cannot fill four partitions:\n{text}"
    );
}

/// The rendered plan, as one string.
fn plan_text(batches: &[RecordBatch]) -> String {
    let mut out = String::new();
    for batch in batches {
        let column = batch
            .column(batch.num_columns() - 1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            out.push_str(column.value(row));
            out.push('\n');
        }
    }
    out
}

/// A table written in one flush should still scan in parallel.
///
/// Parquet splits a single file by row group; a segment is this format's row
/// group, so a flush that makes one enormous segment leaves nothing to split.
#[tokio::test]
async fn one_large_flush_still_scans_in_parallel() {
    let dir = tempfile::tempdir().unwrap();
    let table = ColumnarTable::create(&dir.path().join("t.lt"), schema(), options())
        .await
        .unwrap();

    // One flush, well past the row-group limit.
    for chunk in (0..500_000i64).collect::<Vec<_>>().chunks(50_000) {
        table
            .insert(&[batch(chunk[0]..chunk[chunk.len() - 1] + 1)])
            .await
            .unwrap();
    }
    table.flush().await.unwrap();

    let segments = table.snapshot().manifest.segments.len();
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
    ctx.register_table("t", Arc::new(ColumnarTableProvider::new(table)))
        .unwrap();

    let plan = ctx
        .sql("EXPLAIN SELECT sum(id) FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let text = plan_text(&plan);

    assert!(
        text.contains("partitions=4"),
        "500k rows should split across the four target partitions, \
         but the flush made {segments} segment(s):\n{text}"
    );
}

/// Partitions take work from a shared queue, so the thing that could go wrong
/// is a segment being handed out twice or skipped. Every row must appear
/// exactly once however many partitions are pulling.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn every_row_is_read_exactly_once_however_many_partitions() {
    let dir = tempfile::tempdir().unwrap();
    let table = ColumnarTable::create(&dir.path().join("t.lt"), schema(), options())
        .await
        .unwrap();

    // Deliberately uneven: a big flush, then several small ones, so the pieces
    // of work differ in cost and a static split would leave partitions idle.
    table.insert(&[batch(0..40_000)]).await.unwrap();
    table.flush().await.unwrap();
    for round in 0..7i64 {
        let start = 40_000 + round * 300;
        table.insert(&[batch(start..start + 300)]).await.unwrap();
        table.flush().await.unwrap();
    }
    // And some rows still in memory, which are work items too.
    table.insert(&[batch(42_100..42_500)]).await.unwrap();

    let expected: Vec<i64> = (0..42_100).chain(42_100..42_500).collect();

    for partitions in [1usize, 2, 3, 5, 8, 16] {
        let ctx = SessionContext::new_with_config(
            SessionConfig::new().with_target_partitions(partitions),
        );
        ctx.register_table("t", Arc::new(ColumnarTableProvider::new(table.clone())))
            .unwrap();

        let rows = query(&ctx, "SELECT id FROM t").await;
        assert_eq!(
            ids(&rows),
            expected,
            "with {partitions} partitions, the rows read were not the rows stored"
        );

        // count(*) goes through a different path than materialising the rows.
        assert_eq!(
            scalar_i64(&query(&ctx, "SELECT count(*) FROM t").await) as usize,
            expected.len(),
            "with {partitions} partitions"
        );
    }
}

/// A limit is a budget shared across partitions, and so is the work queue.
/// Together they must still stop at exactly the limit.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_limit_holds_while_partitions_share_the_queue() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, 20, 30).await;

    for limit in [1usize, 7, 100, 999, 2030, 5000] {
        let rows = query(&ctx, &format!("SELECT id FROM t LIMIT {limit}")).await;
        assert_eq!(count(&rows), limit.min(2030), "limit {limit}");
    }
}

/// Row groups grow with the table rather than staying at one size.
#[tokio::test]
async fn row_groups_grow_with_the_table() {
    let dir = tempfile::tempdir().unwrap();

    // A small table: groups are held at the floor, so it still divides.
    let small = ColumnarTable::create(&dir.path().join("small.lt"), schema(), options())
        .await
        .unwrap();
    small.insert(&[batch(0..20_000)]).await.unwrap();
    small.flush().await.unwrap();
    let small_segments = small.snapshot().manifest.segments.len();

    // A larger one written the same way: bigger groups, and more of them.
    let large = ColumnarTable::create(&dir.path().join("large.lt"), schema(), options())
        .await
        .unwrap();
    for chunk in (0..400_000i64).collect::<Vec<_>>().chunks(50_000) {
        large
            .insert(&[batch(chunk[0]..chunk[chunk.len() - 1] + 1)])
            .await
            .unwrap();
    }
    large.flush().await.unwrap();
    let large_rows: Vec<u64> = large
        .snapshot()
        .manifest
        .segments
        .iter()
        .map(|s| s.row_count)
        .collect();
    let small_rows: Vec<u64> = small
        .snapshot()
        .manifest
        .segments
        .iter()
        .map(|s| s.row_count)
        .collect();

    assert!(
        small_segments > 1,
        "even 20k rows should divide: {small_rows:?}"
    );
    assert!(
        large_rows.iter().max() > small_rows.iter().max(),
        "a bigger table should use bigger row groups: {small_rows:?} then {large_rows:?}"
    );
    assert!(
        large_rows.len() >= localtables_format::config::TARGET_ROW_GROUPS,
        "and still hold enough of them to divide: {} groups",
        large_rows.len()
    );
}
