//! Querying a b-tree table through SQL.
//!
//! The answers must be right, and the plan must show the key bound was used:
//! a point lookup should seek to one key, not read the tree.

use std::sync::Arc;

use arrow::array::{Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;

use datafusion_local_tables::BTreeTableProvider;
use localtables_format::btree::BTreeTable;
use localtables_format::config::{Durability, IoBackend, TableOptions};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

fn batch(ids: std::ops::Range<i64>) -> RecordBatch {
    let ids: Vec<i64> = ids.collect();
    let names: Vec<Option<String>> = ids.iter().map(|i| Some(format!("name-{i}"))).collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(names)),
        ],
    )
    .unwrap()
}

fn options() -> TableOptions {
    TableOptions {
        durability: Durability::None,
        io_backend: IoBackend::Mmap,
        memtable_max_bytes: 64 * 1024 * 1024,
        ..TableOptions::default()
    }
}

async fn fixture(dir: &tempfile::TempDir, rows: i64) -> (BTreeTable, SessionContext) {
    let table = BTreeTable::create(&dir.path().join("t.ltb"), schema(), &["id"], options())
        .await
        .unwrap();
    table.insert(&[batch(0..rows)]).await.unwrap();
    table.flush().await.unwrap();

    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(BTreeTableProvider::new(table.clone())))
        .unwrap();
    (table, ctx)
}

async fn query(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    ctx.sql(sql).await.unwrap().collect().await.unwrap()
}

fn ids(batches: &[RecordBatch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect()
}

fn count(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

async fn plan_text(ctx: &SessionContext, sql: &str) -> String {
    let batches = query(ctx, &format!("EXPLAIN {sql}")).await;
    let mut out = String::new();
    for batch in &batches {
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

#[tokio::test]
async fn a_select_returns_every_row_in_key_order() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = fixture(&dir, 200).await;

    let rows = query(&ctx, "SELECT id FROM t").await;
    assert_eq!(ids(&rows), (0..200).collect::<Vec<i64>>());
}

#[tokio::test]
async fn a_point_lookup_seeks_to_one_key() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = fixture(&dir, 1000).await;

    let sql = "SELECT id, name FROM t WHERE id = 742";
    let text = plan_text(&ctx, sql).await;
    assert!(
        text.contains("keys=[") && !text.contains("keys=all"),
        "the plan should seek rather than scan:\n{text}"
    );

    let rows = query(&ctx, sql).await;
    assert_eq!(ids(&rows), vec![742]);
}

#[tokio::test]
async fn a_range_query_bounds_the_scan() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = fixture(&dir, 1000).await;

    let sql = "SELECT id FROM t WHERE id >= 100 AND id < 110";
    assert!(
        !plan_text(&ctx, sql).await.contains("keys=all"),
        "both bounds should narrow the scan"
    );
    assert_eq!(
        ids(&query(&ctx, sql).await),
        (100..110).collect::<Vec<i64>>()
    );
}

#[tokio::test]
async fn every_comparison_returns_the_right_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = fixture(&dir, 50).await;

    for (sql, expected) in [
        ("SELECT id FROM t WHERE id = 10", vec![10i64]),
        ("SELECT id FROM t WHERE id > 47", vec![48, 49]),
        ("SELECT id FROM t WHERE id >= 48", vec![48, 49]),
        ("SELECT id FROM t WHERE id < 2", vec![0, 1]),
        ("SELECT id FROM t WHERE id <= 1", vec![0, 1]),
        ("SELECT id FROM t WHERE 47 < id", vec![48, 49]),
        ("SELECT id FROM t WHERE 1 >= id", vec![0, 1]),
    ] {
        assert_eq!(ids(&query(&ctx, sql).await), expected, "{sql}");
    }
}

#[tokio::test]
async fn a_predicate_that_matches_nothing_returns_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = fixture(&dir, 50).await;

    assert_eq!(
        count(&query(&ctx, "SELECT id FROM t WHERE id = 999").await),
        0
    );
    assert_eq!(
        count(&query(&ctx, "SELECT id FROM t WHERE id > 1000").await),
        0
    );
    assert_eq!(
        count(&query(&ctx, "SELECT id FROM t WHERE id > 40 AND id < 10").await),
        0,
        "contradictory bounds hold nothing"
    );
}

#[tokio::test]
async fn a_predicate_on_another_column_still_returns_the_right_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = fixture(&dir, 50).await;

    // Not a key bound, so the scan reads the tree and the filter runs above it.
    let rows = query(&ctx, "SELECT id FROM t WHERE name = 'name-7'").await;
    assert_eq!(ids(&rows), vec![7]);
}

#[tokio::test]
async fn a_key_bound_and_another_predicate_combine() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = fixture(&dir, 100).await;

    let rows = query(
        &ctx,
        "SELECT id FROM t WHERE id >= 10 AND id < 20 AND name = 'name-15'",
    )
    .await;
    assert_eq!(ids(&rows), vec![15]);
}

#[tokio::test]
async fn pending_writes_are_part_of_the_answer() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir, 10).await;

    // In the log and the overlay, not yet in the tree.
    table.insert(&[batch(100..103)]).await.unwrap();
    assert!(table.pending_changes().await > 0);

    assert_eq!(
        ids(&query(&ctx, "SELECT id FROM t WHERE id >= 100").await),
        vec![100, 101, 102]
    );
    assert_eq!(count(&query(&ctx, "SELECT id FROM t").await), 13);
}

#[tokio::test]
async fn a_pending_delete_hides_a_row() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir, 10).await;

    let doomed = table.key_of(&batch(5..6), 0).unwrap();
    table.delete_keys(&[doomed]).await.unwrap();

    assert_eq!(
        count(&query(&ctx, "SELECT id FROM t WHERE id = 5").await),
        0
    );
    assert_eq!(
        ids(&query(&ctx, "SELECT id FROM t").await),
        (0..10).filter(|i| *i != 5).collect::<Vec<i64>>()
    );
}

#[tokio::test]
async fn a_projection_returns_only_the_named_columns() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = fixture(&dir, 20).await;

    let rows = query(&ctx, "SELECT name FROM t WHERE id = 3").await;
    assert_eq!(rows[0].num_columns(), 1);
    assert_eq!(
        rows[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "name-3"
    );
}

#[tokio::test]
async fn a_limit_stops_the_scan() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = fixture(&dir, 500).await;

    for limit in [1usize, 10, 499, 500, 5000] {
        let rows = query(&ctx, &format!("SELECT id FROM t LIMIT {limit}")).await;
        assert_eq!(count(&rows), limit.min(500), "limit {limit}");
    }
}

#[tokio::test]
async fn aggregates_run_over_the_whole_table() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = fixture(&dir, 100).await;

    let rows = query(&ctx, "SELECT count(*), min(id), max(id) FROM t").await;
    let counts = rows[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(counts.value(0), 100);
}

#[tokio::test]
async fn a_join_against_a_columnar_table_works() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = fixture(&dir, 100).await;

    let columnar =
        localtables_format::ColumnarTable::create(&dir.path().join("c.lt"), schema(), options())
            .await
            .unwrap();
    columnar.insert(&[batch(50..150)]).await.unwrap();
    columnar.flush().await.unwrap();
    ctx.register_table(
        "c",
        Arc::new(datafusion_local_tables::ColumnarTableProvider::new(
            columnar,
        )),
    )
    .unwrap();

    let rows = query(
        &ctx,
        "SELECT t.id FROM t JOIN c ON t.id = c.id ORDER BY t.id",
    )
    .await;
    assert_eq!(ids(&rows), (50..100).collect::<Vec<i64>>());
}

#[tokio::test]
async fn a_reopened_table_queries_the_same() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.ltb");
    {
        let table = BTreeTable::create(&path, schema(), &["id"], options())
            .await
            .unwrap();
        table.insert(&[batch(0..100)]).await.unwrap();
        table.flush().await.unwrap();
        // These stay in the log for the restart to replay.
        table.insert(&[batch(100..120)]).await.unwrap();
    }

    let table = BTreeTable::open(&path, &["id"], options()).await.unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(BTreeTableProvider::new(table)))
        .unwrap();

    assert_eq!(count(&query(&ctx, "SELECT id FROM t").await), 120);
    assert_eq!(
        ids(&query(&ctx, "SELECT id FROM t WHERE id = 110").await),
        vec![110]
    );
}

#[tokio::test]
async fn a_string_key_seeks_too() {
    let dir = tempfile::tempdir().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("score", DataType::Int64, true),
    ]));
    let table = BTreeTable::create(
        &dir.path().join("s.ltb"),
        schema.clone(),
        &["name"],
        options(),
    )
    .await
    .unwrap();

    let rows = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["alpha", "beta", "gamma", "delta"])),
            Arc::new(Int64Array::from(vec![1i64, 2, 3, 4])),
        ],
    )
    .unwrap();
    table.insert(&[rows]).await.unwrap();
    table.flush().await.unwrap();

    let ctx = SessionContext::new();
    ctx.register_table("s", Arc::new(BTreeTableProvider::new(table)))
        .unwrap();

    let found = query(&ctx, "SELECT score FROM s WHERE name = 'gamma'").await;
    assert_eq!(
        found[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        3
    );

    let ranged = query(&ctx, "SELECT name FROM s WHERE name >= 'b' AND name < 'e'").await;
    let names: Vec<String> = ranged
        .iter()
        .flat_map(|b| {
            let column = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            (0..b.num_rows())
                .map(|row| column.value(row).to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(names, vec!["beta", "delta"]);
}

/// A key bound is reported exact, so DataFusion drops its own filter. If the
/// bound were ever wrong, this is where rows would go missing.
#[tokio::test]
async fn an_exact_bound_returns_exactly_the_matching_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = fixture(&dir, 300).await;

    for start in (0..300).step_by(17) {
        let end = start + 17;
        let sql = format!("SELECT id FROM t WHERE id >= {start} AND id < {end}");
        assert_eq!(
            ids(&query(&ctx, &sql).await),
            (start..end.min(300)).collect::<Vec<i64>>(),
            "{sql}"
        );
    }
}
