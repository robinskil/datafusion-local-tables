//! Writing to a local table through SQL.
//!
//! These check the answers and the durability: after every statement the rows
//! are what the statement says they should be, and reopening the table from
//! disk gives the same rows again.

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;

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

fn options() -> TableOptions {
    TableOptions {
        durability: Durability::None,
        io_backend: IoBackend::Mmap,
        memtable_max_bytes: 64 * 1024 * 1024,
        ..TableOptions::default()
    }
}

fn batch(ids: std::ops::Range<i64>) -> RecordBatch {
    let ids: Vec<i64> = ids.collect();
    let names: Vec<Option<String>> = ids.iter().map(|i| Some(format!("name-{i}"))).collect();
    let scores: Vec<Option<i32>> = ids.iter().map(|i| Some(*i as i32 * 10)).collect();
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

async fn fixture(dir: &tempfile::TempDir) -> (ColumnarTable, SessionContext) {
    let table = ColumnarTable::create(&dir.path().join("t.lt"), schema(), options())
        .await
        .unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(ColumnarTableProvider::new(table.clone())))
        .unwrap();
    (table, ctx)
}

async fn run(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    ctx.sql(sql).await.unwrap().collect().await.unwrap()
}

/// The count a DML statement reports.
fn affected(batches: &[RecordBatch]) -> u64 {
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap()
        .value(0)
}

async fn ids(ctx: &SessionContext) -> Vec<i64> {
    let rows = run(ctx, "SELECT id FROM t ORDER BY id").await;
    rows.iter()
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

async fn scores(ctx: &SessionContext) -> Vec<(i64, Option<i32>)> {
    let rows = run(ctx, "SELECT id, score FROM t ORDER BY id").await;
    let mut out = Vec::new();
    for batch in &rows {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let scores = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for row in 0..batch.num_rows() {
            out.push((
                ids.value(row),
                (!scores.is_null(row)).then(|| scores.value(row)),
            ));
        }
    }
    out
}

/// Reopen the table from disk and read it back, to prove the change was durable.
///
/// Both handles have to go first: the session holds its own clone of the
/// table, and the writer lock lives until the last one drops.
async fn reopen(
    dir: &tempfile::TempDir,
    table: ColumnarTable,
    ctx: SessionContext,
) -> SessionContext {
    drop(table);
    drop(ctx);

    let table = ColumnarTable::open(&dir.path().join("t.lt"), options())
        .await
        .unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(ColumnarTableProvider::new(table)))
        .unwrap();
    ctx
}

async fn reopened_ids(
    dir: &tempfile::TempDir,
    table: ColumnarTable,
    ctx: SessionContext,
) -> Vec<i64> {
    let ctx = reopen(dir, table, ctx).await;
    ids(&ctx).await
}

#[tokio::test]
async fn insert_values_adds_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir).await;

    let result = run(
        &ctx,
        "INSERT INTO t VALUES (1, 'one', 10), (2, 'two', 20), (3, 'three', 30)",
    )
    .await;
    assert_eq!(affected(&result), 3);
    assert_eq!(ids(&ctx).await, vec![1, 2, 3]);
    assert_eq!(reopened_ids(&dir, table, ctx).await, vec![1, 2, 3]);
}

#[tokio::test]
async fn insert_select_copies_rows_between_tables() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = fixture(&dir).await;

    let source = ColumnarTable::create(&dir.path().join("src.lt"), schema(), options())
        .await
        .unwrap();
    source.insert(&[batch(0..50)]).await.unwrap();
    source.flush().await.unwrap();
    ctx.register_table("src", Arc::new(ColumnarTableProvider::new(source)))
        .unwrap();

    let result = run(&ctx, "INSERT INTO t SELECT * FROM src WHERE id >= 20").await;
    assert_eq!(affected(&result), 30);
    assert_eq!(ids(&ctx).await, (20..50).collect::<Vec<i64>>());
}

#[tokio::test]
async fn inserting_no_rows_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = fixture(&dir).await;
    run(&ctx, "INSERT INTO t VALUES (1, 'one', 10)").await;

    let result = run(&ctx, "INSERT INTO t SELECT * FROM t WHERE id > 1000").await;
    assert_eq!(affected(&result), 0);
    assert_eq!(ids(&ctx).await, vec![1]);
}

#[tokio::test]
async fn delete_with_a_predicate_removes_only_matching_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir).await;
    table.insert(&[batch(0..20)]).await.unwrap();
    table.flush().await.unwrap();

    let result = run(&ctx, "DELETE FROM t WHERE id >= 5 AND id < 10").await;
    assert_eq!(affected(&result), 5);
    assert_eq!(
        ids(&ctx).await,
        (0..20)
            .filter(|i| !(5..10).contains(i))
            .collect::<Vec<i64>>()
    );
}

#[tokio::test]
async fn delete_without_a_predicate_empties_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir).await;
    table.insert(&[batch(0..30)]).await.unwrap();
    table.flush().await.unwrap();
    table.insert(&[batch(30..40)]).await.unwrap();

    let result = run(&ctx, "DELETE FROM t").await;
    assert_eq!(affected(&result), 40);
    assert!(ids(&ctx).await.is_empty());
    assert!(reopened_ids(&dir, table, ctx).await.is_empty());
}

#[tokio::test]
async fn delete_reaches_rows_that_are_still_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir).await;
    table.insert(&[batch(0..10)]).await.unwrap();
    table.flush().await.unwrap();
    // These are in the log, not in a segment.
    table.insert(&[batch(10..20)]).await.unwrap();
    assert_eq!(table.memtable_rows().await, 10);

    let result = run(&ctx, "DELETE FROM t WHERE id >= 8 AND id < 12").await;
    assert_eq!(
        affected(&result),
        4,
        "the range spans both a segment and the memtable"
    );
    assert_eq!(
        ids(&ctx).await,
        (0..20)
            .filter(|i| !(8..12).contains(i))
            .collect::<Vec<i64>>()
    );
}

#[tokio::test]
async fn a_delete_that_matches_nothing_reports_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir).await;
    table.insert(&[batch(0..10)]).await.unwrap();
    table.flush().await.unwrap();

    let result = run(&ctx, "DELETE FROM t WHERE id > 1000").await;
    assert_eq!(affected(&result), 0);
    assert_eq!(ids(&ctx).await, (0..10).collect::<Vec<i64>>());
}

#[tokio::test]
async fn deleting_the_same_rows_twice_reports_them_once() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir).await;
    table.insert(&[batch(0..10)]).await.unwrap();
    table.flush().await.unwrap();

    assert_eq!(affected(&run(&ctx, "DELETE FROM t WHERE id < 5").await), 5);
    assert_eq!(
        affected(&run(&ctx, "DELETE FROM t WHERE id < 5").await),
        0,
        "rows already gone are not deleted again"
    );
    assert_eq!(ids(&ctx).await, (5..10).collect::<Vec<i64>>());
}

#[tokio::test]
async fn a_delete_survives_a_reopen_without_a_flush() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir).await;
    table.insert(&[batch(0..10)]).await.unwrap();
    table.flush().await.unwrap();

    run(&ctx, "DELETE FROM t WHERE id % 2 = 0").await;
    let expected: Vec<i64> = (0..10).filter(|i| i % 2 != 0).collect();
    assert_eq!(ids(&ctx).await, expected);
    assert_eq!(reopened_ids(&dir, table, ctx).await, expected);
}

#[tokio::test]
async fn update_changes_the_columns_it_names() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir).await;
    table.insert(&[batch(0..5)]).await.unwrap();
    table.flush().await.unwrap();

    let result = run(&ctx, "UPDATE t SET score = 999 WHERE id < 3").await;
    assert_eq!(affected(&result), 3);

    let mut scores = scores(&ctx).await;
    scores.sort();
    assert_eq!(
        scores,
        vec![
            (0, Some(999)),
            (1, Some(999)),
            (2, Some(999)),
            (3, Some(30)),
            (4, Some(40)),
        ]
    );
}

#[tokio::test]
async fn update_can_read_the_old_value() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir).await;
    table.insert(&[batch(1..4)]).await.unwrap();
    table.flush().await.unwrap();

    run(&ctx, "UPDATE t SET score = score + 1 WHERE id = 2").await;

    let mut scores = scores(&ctx).await;
    scores.sort();
    assert_eq!(scores, vec![(1, Some(10)), (2, Some(21)), (3, Some(30))]);
}

#[tokio::test]
async fn update_without_a_predicate_touches_every_row() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir).await;
    table.insert(&[batch(0..6)]).await.unwrap();
    table.flush().await.unwrap();

    let result = run(&ctx, "UPDATE t SET score = 0").await;
    assert_eq!(affected(&result), 6);
    assert!(scores(&ctx).await.iter().all(|(_, s)| *s == Some(0)));
}

#[tokio::test]
async fn update_can_set_a_column_to_null() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir).await;
    table.insert(&[batch(0..3)]).await.unwrap();
    table.flush().await.unwrap();

    run(&ctx, "UPDATE t SET score = NULL WHERE id = 1").await;

    let mut scores = scores(&ctx).await;
    scores.sort();
    assert_eq!(scores, vec![(0, Some(0)), (1, None), (2, Some(20))]);
}

#[tokio::test]
async fn an_update_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir).await;
    table.insert(&[batch(0..5)]).await.unwrap();
    table.flush().await.unwrap();
    run(&ctx, "UPDATE t SET name = 'changed' WHERE id = 3").await;

    let ctx = reopen(&dir, table, ctx).await;

    let rows = run(&ctx, "SELECT name FROM t WHERE id = 3").await;
    let names = rows[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "changed");
    assert_eq!(ids(&ctx).await, (0..5).collect::<Vec<i64>>());
}

#[tokio::test]
async fn an_update_never_changes_the_row_count() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir).await;
    table.insert(&[batch(0..20)]).await.unwrap();
    table.flush().await.unwrap();
    table.insert(&[batch(20..25)]).await.unwrap();

    run(&ctx, "UPDATE t SET score = 1 WHERE id % 3 = 0").await;
    assert_eq!(
        ids(&ctx).await,
        (0..25).collect::<Vec<i64>>(),
        "an update replaces rows, it does not add or lose any"
    );
}

#[tokio::test]
async fn an_update_that_matches_nothing_reports_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir).await;
    table.insert(&[batch(0..5)]).await.unwrap();
    table.flush().await.unwrap();

    let result = run(&ctx, "UPDATE t SET score = 1 WHERE id > 1000").await;
    assert_eq!(affected(&result), 0);
    assert_eq!(scores(&ctx).await.len(), 5);
}

#[tokio::test]
async fn statements_in_sequence_leave_the_table_consistent() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = fixture(&dir).await;

    run(
        &ctx,
        "INSERT INTO t VALUES (1, 'a', 1), (2, 'b', 2), (3, 'c', 3)",
    )
    .await;
    run(&ctx, "UPDATE t SET score = 100 WHERE id = 2").await;
    run(&ctx, "DELETE FROM t WHERE id = 1").await;
    run(&ctx, "INSERT INTO t VALUES (4, 'd', 4)").await;
    table.flush().await.unwrap();
    run(&ctx, "DELETE FROM t WHERE score < 5").await;

    let mut scores = scores(&ctx).await;
    scores.sort();
    assert_eq!(scores, vec![(2, Some(100))]);
    assert_eq!(reopened_ids(&dir, table, ctx).await, vec![2]);
}
