//! A schema change seen through SQL.
//!
//! The provider reads the table's schema when it is asked, so a change reaches
//! a query without the table being re-registered. These check that, and that
//! the pruning and DML built on the schema keep working across one.

use std::sync::Arc;

use arrow::array::{Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;

use datafusion_local_tables::ColumnarTableProvider;
use localtables_format::columnar::table::ColumnarTable;
use localtables_format::config::{BloomFilters, Durability, TableOptions};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

fn options() -> TableOptions {
    TableOptions {
        durability: Durability::None,
        min_row_group_rows: 100,
        row_group_rows: 100,
        bloom_filters: BloomFilters::All,
        ..TableOptions::default()
    }
}

async fn table(dir: &tempfile::TempDir) -> (ColumnarTable, SessionContext) {
    let table = ColumnarTable::create(&dir.path().join("t.lt"), schema(), options())
        .await
        .unwrap();
    for group in 0..5 {
        let ids: Vec<i32> = (group * 100..group * 100 + 100).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("name-{i}")).collect();
        table
            .insert(&[RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int32Array::from(ids)),
                    Arc::new(StringArray::from(names)),
                ],
            )
            .unwrap()])
            .await
            .unwrap();
        table.flush().await.unwrap();
    }

    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(ColumnarTableProvider::new(table.clone())))
        .unwrap();
    (table, ctx)
}

async fn rows(ctx: &SessionContext, sql: &str) -> usize {
    ctx.sql(sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
        .iter()
        .map(|b| b.num_rows())
        .sum()
}

async fn one_int(ctx: &SessionContext, sql: &str) -> i64 {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let column = batches[0].column(0);
    match column.data_type() {
        DataType::Int64 => column.as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        other => panic!("unexpected {other}"),
    }
}

fn pruned(plan: &str) -> usize {
    let at = plan.find("pruned=").expect("the scan reports what it pruned");
    plan[at + "pruned=".len()..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .unwrap()
}

async fn plan_of(ctx: &SessionContext, sql: &str) -> String {
    let plan = ctx
        .sql(&format!("EXPLAIN {sql}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    arrow::util::pretty::pretty_format_batches(&plan)
        .unwrap()
        .to_string()
}

#[tokio::test]
async fn an_added_column_is_queryable_without_re_registering() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = table(&dir).await;

    table
        .add_column(Arc::new(Field::new("score", DataType::Float64, true)))
        .await
        .unwrap();

    assert_eq!(one_int(&ctx, "SELECT count(*) FROM t").await, 500);
    assert_eq!(
        one_int(&ctx, "SELECT count(score) FROM t").await,
        0,
        "no stored row has a value for it"
    );
    assert_eq!(rows(&ctx, "SELECT score FROM t").await, 500);
}

#[tokio::test]
async fn a_renamed_column_is_queryable_by_its_new_name() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = table(&dir).await;
    table.rename_column("name", "label").await.unwrap();

    assert_eq!(one_int(&ctx, "SELECT count(label) FROM t").await, 500);
    assert!(ctx.sql("SELECT name FROM t").await.is_err());
}

#[tokio::test]
async fn a_dropped_column_disappears_from_queries() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = table(&dir).await;
    table.drop_column("name").await.unwrap();

    assert_eq!(one_int(&ctx, "SELECT count(*) FROM t").await, 500);
    assert!(ctx.sql("SELECT name FROM t").await.is_err());
}

/// The reason a cast rewrites. Afterwards the column has one type everywhere,
/// so predicates on it prune exactly as they did before.
#[tokio::test]
async fn pruning_still_works_after_a_cast() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = table(&dir).await;

    let sql = "SELECT * FROM t WHERE id = 250";
    let before = pruned(&plan_of(&ctx, sql).await);
    assert!(before >= 4, "zone maps prune the ordered id: {before}");

    table.cast_column("id", DataType::Int64).await.unwrap();

    let after = pruned(&plan_of(&ctx, sql).await);
    assert!(
        after >= 4,
        "pruning should survive the cast: {before} then {after}"
    );
    assert_eq!(rows(&ctx, sql).await, 1);
    assert_eq!(one_int(&ctx, "SELECT count(*) FROM t").await, 500);
}

/// Writes through SQL keep working, and land in the new shape.
#[tokio::test]
async fn sql_writes_work_after_a_schema_change() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = table(&dir).await;
    table
        .add_column(Arc::new(Field::new("score", DataType::Float64, true)))
        .await
        .unwrap();

    ctx.sql("INSERT INTO t VALUES (9999, 'fresh', 1.5)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert_eq!(one_int(&ctx, "SELECT count(*) FROM t").await, 501);
    assert_eq!(one_int(&ctx, "SELECT count(score) FROM t").await, 1);
    assert_eq!(rows(&ctx, "SELECT * FROM t WHERE id = 9999").await, 1);

    ctx.sql("DELETE FROM t WHERE id = 9999")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(one_int(&ctx, "SELECT count(*) FROM t").await, 500);
}

#[tokio::test]
async fn a_table_reopened_after_a_change_reads_the_new_schema() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    {
        let (table, _ctx) = table(&dir).await;
        table.cast_column("id", DataType::Int64).await.unwrap();
        table
            .add_column(Arc::new(Field::new("score", DataType::Float64, true)))
            .await
            .unwrap();
    }

    let table = ColumnarTable::open(&path, options()).await.unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(ColumnarTableProvider::new(table)))
        .unwrap();

    assert_eq!(one_int(&ctx, "SELECT count(*) FROM t").await, 500);
    assert_eq!(rows(&ctx, "SELECT id, name, score FROM t").await, 500);
    assert_eq!(one_int(&ctx, "SELECT sum(id) FROM t").await, (0..500i64).sum::<i64>());
}
