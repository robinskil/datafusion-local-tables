//! Changing a table's indexes and layout after it holds data.
//!
//! Filters and row order are decided when a segment is written, so a table
//! changes them by being rewritten. These check that reopening with different
//! options and rewriting actually moves what the scan prunes, and that no row
//! moves with it.

use std::sync::Arc;

use arrow::array::{Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;

use datafusion_local_tables::ColumnarTableProvider;
use localtables_format::columnar::table::ColumnarTable;
use localtables_format::config::{BloomFilters, Durability, TableOptions};

const SEGMENTS: i64 = 10;
const PER_SEGMENT: i64 = 100;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("body", DataType::Utf8, false),
    ]))
}

/// Groups small enough that a rewrite leaves segments to prune between.
fn options() -> TableOptions {
    TableOptions {
        durability: Durability::None,
        min_row_group_rows: PER_SEGMENT as usize,
        row_group_rows: PER_SEGMENT as usize,
        ..TableOptions::default()
    }
}

/// Segment `k` holds keys scattered across the whole range, so no zone map
/// rules any of them out, and text of its own, so a substring search can.
async fn seed(path: &std::path::Path) {
    let table = ColumnarTable::create(path, schema(), options())
        .await
        .unwrap();
    for segment in 0..SEGMENTS {
        let keys: Vec<i64> = (0..PER_SEGMENT).map(|r| segment + r * SEGMENTS).collect();
        let bodies: Vec<String> = keys
            .iter()
            .map(|k| format!("shard{segment} record{k} filler"))
            .collect();
        let batch = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(keys)),
                Arc::new(StringArray::from(bodies)),
            ],
        )
        .unwrap();
        table.insert(&[batch]).await.unwrap();
        table.flush().await.unwrap();
    }
}

fn pruned(plan: &str) -> usize {
    let at = plan
        .find("pruned=")
        .expect("the scan reports what it pruned");
    plan[at + "pruned=".len()..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .unwrap()
}

fn session(table: &ColumnarTable) -> SessionContext {
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(ColumnarTableProvider::new(table.clone())))
        .unwrap();
    ctx
}

async fn prunes(table: &ColumnarTable, sql: &str) -> usize {
    let ctx = session(table);
    let plan = ctx
        .sql(&format!("EXPLAIN {sql}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    pruned(
        &arrow::util::pretty::pretty_format_batches(&plan)
            .unwrap()
            .to_string(),
    )
}

async fn rows(table: &ColumnarTable, sql: &str) -> usize {
    session(table)
        .sql(sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
        .iter()
        .map(|b| b.num_rows())
        .sum()
}

const POINT: &str = "SELECT * FROM t WHERE key = 550";
const SUBSTRING: &str = "SELECT * FROM t WHERE body LIKE '%shard7 %'";

/// Reopening alone changes nothing, because the segments already on disk were
/// written without the filters. Rewriting is what applies them.
#[tokio::test]
async fn a_table_gains_filters_by_being_rewritten_not_by_being_reopened() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    seed(&path).await;

    let wanted = TableOptions {
        bloom_filters: BloomFilters::All,
        trigram_filters: BloomFilters::All,
        ..options()
    };
    let table = ColumnarTable::open(&path, wanted).await.unwrap();

    assert_eq!(prunes(&table, POINT).await, 0, "reopening applies nothing");
    assert_eq!(prunes(&table, SUBSTRING).await, 0);

    let rewritten = table.rewrite_all().await.unwrap();
    assert_eq!(rewritten, (SEGMENTS * PER_SEGMENT) as u64);

    assert!(prunes(&table, POINT).await >= 8, "the filter now prunes");
    assert!(prunes(&table, SUBSTRING).await >= 8);

    // And every row is still there, in the right place.
    assert_eq!(rows(&table, POINT).await, 1);
    assert_eq!(rows(&table, SUBSTRING).await, PER_SEGMENT as usize);
    assert_eq!(
        rows(&table, "SELECT * FROM t").await,
        (SEGMENTS * PER_SEGMENT) as usize
    );
}

/// The reverse. A table that has filters sheds them the same way.
#[tokio::test]
async fn a_table_sheds_filters_by_being_rewritten_without_them() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    {
        let table = ColumnarTable::create(
            &path,
            schema(),
            TableOptions {
                bloom_filters: BloomFilters::All,
                ..options()
            },
        )
        .await
        .unwrap();
        for segment in 0..SEGMENTS {
            let keys: Vec<i64> = (0..PER_SEGMENT).map(|r| segment + r * SEGMENTS).collect();
            let bodies: Vec<String> = keys.iter().map(|k| format!("record{k}")).collect();
            table
                .insert(&[RecordBatch::try_new(
                    schema(),
                    vec![
                        Arc::new(Int64Array::from(keys)),
                        Arc::new(StringArray::from(bodies)),
                    ],
                )
                .unwrap()])
                .await
                .unwrap();
            table.flush().await.unwrap();
        }
        assert!(prunes(&table, POINT).await >= 8);
    }

    let table = ColumnarTable::open(&path, options()).await.unwrap();
    table.rewrite_all().await.unwrap();
    assert_eq!(prunes(&table, POINT).await, 0, "the filters are gone");
    assert_eq!(rows(&table, POINT).await, 1, "the rows are not");
}

/// Clustering is applied by the same rewrite, and reorders rows that are
/// already stored.
#[tokio::test]
async fn a_table_gains_a_clustered_order_by_being_rewritten() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    seed(&path).await;

    let table = ColumnarTable::open(
        &path,
        TableOptions {
            cluster_by: vec!["key".to_string()],
            ..options()
        },
    )
    .await
    .unwrap();

    assert_eq!(
        prunes(&table, POINT).await,
        0,
        "keys are scattered as written"
    );
    table.rewrite_all().await.unwrap();
    assert!(
        prunes(&table, POINT).await >= 8,
        "clustering puts each key in one segment"
    );
    assert_eq!(rows(&table, POINT).await, 1);
    assert_eq!(
        rows(&table, "SELECT * FROM t").await,
        (SEGMENTS * PER_SEGMENT) as usize
    );
}

/// Clustering by one column costs another its locality. Worth stating, because
/// it is the trade a rewrite silently makes.
#[tokio::test]
async fn clustering_by_one_column_scatters_another() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    seed(&path).await;

    let table = ColumnarTable::open(
        &path,
        TableOptions {
            cluster_by: vec!["key".to_string()],
            trigram_filters: BloomFilters::All,
            ..options()
        },
    )
    .await
    .unwrap();
    table.rewrite_all().await.unwrap();

    // Each shard's rows are now spread across every segment, so the trigram
    // filter is present and correct and rules nothing out.
    assert_eq!(prunes(&table, SUBSTRING).await, 0);
    assert_eq!(rows(&table, SUBSTRING).await, PER_SEGMENT as usize);
}

#[tokio::test]
async fn rewriting_an_empty_table_does_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    let table = ColumnarTable::create(&path, schema(), options())
        .await
        .unwrap();
    assert_eq!(table.rewrite_all().await.unwrap(), 0);
}

/// A rewrite survives a reopen, because it is an ordinary commit.
#[tokio::test]
async fn a_rewrite_is_durable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    seed(&path).await;

    let wanted = TableOptions {
        bloom_filters: BloomFilters::All,
        ..options()
    };
    {
        let table = ColumnarTable::open(&path, wanted.clone()).await.unwrap();
        table.rewrite_all().await.unwrap();
    }

    let table = ColumnarTable::open(&path, wanted).await.unwrap();
    assert!(prunes(&table, POINT).await >= 8);
    assert_eq!(rows(&table, POINT).await, 1);
}
