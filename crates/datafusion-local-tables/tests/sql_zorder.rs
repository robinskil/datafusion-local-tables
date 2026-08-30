//! Clustered row order, through SQL.
//!
//! A table is written once in insert order and once ordered by a z-order key
//! over two columns, and the same queries run against both. What changes is how
//! many segments a zone map can rule out.

use std::sync::Arc;

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;

use datafusion_local_tables::ColumnarTableProvider;
use localtables_format::columnar::table::ColumnarTable;
use localtables_format::config::{Durability, IoBackend, TableOptions};

/// A square grid, written one row of it at a time. `x` cycles through the whole
/// range inside every value of `y`, which is the layout that leaves `x`
/// impossible to prune.
const SIDE: i64 = 128;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, false),
        Field::new("y", DataType::Int64, false),
        Field::new("payload", DataType::Int64, false),
    ]))
}

fn grid() -> RecordBatch {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for y in 0..SIDE {
        for x in 0..SIDE {
            xs.push(x);
            ys.push(y);
        }
    }
    let payload: Vec<i64> = xs.iter().zip(&ys).map(|(x, y)| x * SIDE + y).collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(xs)),
            Arc::new(Int64Array::from(ys)),
            Arc::new(Int64Array::from(payload)),
        ],
    )
    .unwrap()
}

async fn table(dir: &tempfile::TempDir, name: &str, cluster: bool) -> SessionContext {
    let options = TableOptions {
        durability: Durability::None,
        io_backend: IoBackend::Mmap,
        memtable_max_bytes: 64 * 1024 * 1024,
        // Small groups, so a 16,384-row grid divides into enough segments for
        // pruning to be visible at all.
        min_row_group_rows: 1024,
        row_group_rows: 2048,
        cluster_by: if cluster {
            vec!["x".to_string(), "y".to_string()]
        } else {
            Vec::new()
        },
        ..TableOptions::default()
    };

    let table = ColumnarTable::create(&dir.path().join(format!("{name}.lt")), schema(), options)
        .await
        .unwrap();
    table.insert(&[grid()]).await.unwrap();
    table.flush().await.unwrap();

    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(ColumnarTableProvider::new(table)))
        .unwrap();
    ctx
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

/// The trade, stated in segments. Written as it arrives, `y` prunes perfectly
/// and `x` not at all. Clustered, both prune well and neither perfectly.
#[tokio::test]
async fn clustering_trades_one_perfect_column_for_two_good_ones() {
    let dir = tempfile::tempdir().unwrap();
    let plain = table(&dir, "plain", false).await;
    let clustered = table(&dir, "clustered", true).await;

    let on_x = "SELECT * FROM t WHERE x = 64";
    let on_y = "SELECT * FROM t WHERE y = 64";

    let plain_x = pruned(&plan_of(&plain, on_x).await);
    let plain_y = pruned(&plan_of(&plain, on_y).await);
    let clustered_x = pruned(&plan_of(&clustered, on_x).await);
    let clustered_y = pruned(&plan_of(&clustered, on_y).await);

    // Eight segments. Written as it arrives: x 0, y 7.
    // Clustered: x 6, y 4.
    assert_eq!(
        plain_x, 0,
        "every segment holds every x when written in order"
    );
    assert!(plain_y > 0, "y is the column the insert order follows");

    assert!(
        clustered_x > 0,
        "clustering should make x prunable: pruned {clustered_x}"
    );
    assert!(
        clustered_y > 0,
        "clustering should keep y prunable: pruned {clustered_y}"
    );
    assert!(
        clustered_x + clustered_y > plain_x + plain_y,
        "two good columns should beat one perfect and one useless: \
         plain {plain_x}+{plain_y}, clustered {clustered_x}+{clustered_y}"
    );
}

/// Reordering rows must not change a single answer.
#[tokio::test]
async fn clustering_changes_no_answer() {
    let dir = tempfile::tempdir().unwrap();
    let plain = table(&dir, "plain", false).await;
    let clustered = table(&dir, "clustered", true).await;

    for sql in [
        "SELECT count(*) FROM t",
        "SELECT sum(payload) FROM t",
        "SELECT count(*) FROM t WHERE x = 64",
        "SELECT count(*) FROM t WHERE y = 64",
        "SELECT sum(payload) FROM t WHERE x > 100 AND y < 20",
        "SELECT count(*) FROM t WHERE x = 7 AND y = 9",
    ] {
        let left = ctx_value(&plain, sql).await;
        let right = ctx_value(&clustered, sql).await;
        assert_eq!(left, right, "{sql}");
    }
}

async fn ctx_value(ctx: &SessionContext, sql: &str) -> i64 {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let column = batches[0].column(0);
    match column.data_type() {
        DataType::Int64 => column
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        other => panic!("unexpected {other}"),
    }
}

#[tokio::test]
async fn every_row_is_still_there_after_clustering() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = table(&dir, "clustered", true).await;
    assert_eq!(rows(&ctx, "SELECT * FROM t").await, (SIDE * SIDE) as usize);
}

/// A point query over both clustered columns is the case this layout is for.
#[tokio::test]
async fn a_query_on_both_columns_prunes_hardest() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = table(&dir, "clustered", true).await;

    let both = pruned(&plan_of(&ctx, "SELECT * FROM t WHERE x = 64 AND y = 64").await);
    let one = pruned(&plan_of(&ctx, "SELECT * FROM t WHERE x = 64").await);
    assert!(
        both >= one,
        "two bounds should not prune less than one: {both} against {one}"
    );
    assert_eq!(
        rows(&ctx, "SELECT * FROM t WHERE x = 64 AND y = 64").await,
        1
    );
}

#[tokio::test]
async fn a_column_that_is_not_there_is_refused_at_open() {
    let dir = tempfile::tempdir().unwrap();
    let options = TableOptions {
        durability: Durability::None,
        cluster_by: vec!["absent".to_string()],
        ..TableOptions::default()
    };
    let err = ColumnarTable::create(&dir.path().join("t.lt"), schema(), options)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("absent"), "got {err}");
}
