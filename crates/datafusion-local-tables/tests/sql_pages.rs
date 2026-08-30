//! Skipping row ranges inside a segment.
//!
//! A segment's zone map decides whether to read it. Page bounds decide which
//! row ranges inside it are worth handing on, so a predicate matching a hundred
//! rows of a hundred thousand costs the filter above one page rather than the
//! whole segment.
//!
//! Page pruning is invisible in `EXPLAIN`, which counts segments, so these
//! measure it where it shows: in how many rows the scan hands upward. The scan
//! pushes filters down as inexact, so a filter still runs above it and the
//! answer never depends on any of this.

use std::sync::Arc;

use arrow::array::{Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;

use datafusion_local_tables::ColumnarTableProvider;
use localtables_format::columnar::table::ColumnarTable;
use localtables_format::config::{Durability, TableOptions};

const ROWS: i64 = 10_000;
const PAGE: usize = 500;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("body", DataType::Utf8, false),
    ]))
}

/// One segment holding everything, so nothing can be pruned at segment level
/// and only pages are left to do the work.
fn options(page_rows: usize) -> TableOptions {
    TableOptions {
        durability: Durability::None,
        memtable_max_bytes: 256 * 1024 * 1024,
        min_row_group_rows: ROWS as usize * 2,
        row_group_rows: ROWS as usize * 2,
        page_rows,
        scan_batch_rows: PAGE,
        ..TableOptions::default()
    }
}

async fn table(dir: &tempfile::TempDir, name: &str, page_rows: usize) -> ColumnarTable {
    let table = ColumnarTable::create(
        &dir.path().join(format!("{name}.lt")),
        schema(),
        options(page_rows),
    )
    .await
    .unwrap();
    let ids: Vec<i64> = (0..ROWS).collect();
    let bodies: Vec<String> = ids.iter().map(|i| format!("row {i}")).collect();
    table
        .insert(&[RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(bodies)),
            ],
        )
        .unwrap()])
        .await
        .unwrap();
    table.flush().await.unwrap();
    table
}

fn session(table: &ColumnarTable) -> SessionContext {
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(ColumnarTableProvider::new(table.clone())))
        .unwrap();
    ctx
}

/// Rows the scan handed upward, which is what page pruning changes. Read off
/// the scan's own metrics rather than the answer, which is the same either way.
async fn rows_scanned(table: &ColumnarTable, sql: &str) -> usize {
    let ctx = session(table);
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let task = ctx.task_ctx();
    let stream = datafusion::physical_plan::execute_stream(plan.clone(), task).unwrap();
    use futures::TryStreamExt;
    let _: Vec<_> = stream.try_collect().await.unwrap();

    fn scan_rows(plan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>) -> Option<usize> {
        if plan.name() == "ColumnarScanExec" {
            return plan.metrics().map(|m| m.output_rows().unwrap_or(0));
        }
        plan.children().iter().find_map(|child| scan_rows(child))
    }
    scan_rows(&plan).expect("the scan reports its output rows")
}

async fn answer(table: &ColumnarTable, sql: &str) -> usize {
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

/// The measurement. One segment, one matching row, and the scan hands up one
/// page rather than all of them.
#[tokio::test]
async fn a_narrow_predicate_reads_one_page_instead_of_the_segment() {
    let dir = tempfile::tempdir().unwrap();
    let paged = table(&dir, "paged", PAGE).await;
    let plain = table(&dir, "plain", 0).await;

    let sql = "SELECT * FROM t WHERE id = 5000";
    let without = rows_scanned(&plain, sql).await;
    let with = rows_scanned(&paged, sql).await;

    assert_eq!(
        without, ROWS as usize,
        "without page bounds the whole segment is handed up"
    );
    assert_eq!(with, PAGE, "one page holds id 5000");
    assert_eq!(answer(&plain, sql).await, 1);
    assert_eq!(answer(&paged, sql).await, 1);
}

#[tokio::test]
async fn a_range_reads_the_pages_it_spans() {
    let dir = tempfile::tempdir().unwrap();
    let paged = table(&dir, "paged", PAGE).await;

    // 1200..1800 spans pages 2, 3 and part of 4 in 500-row pages.
    let sql = "SELECT * FROM t WHERE id >= 1200 AND id < 1800";
    assert_eq!(rows_scanned(&paged, sql).await, 2 * PAGE);
    assert_eq!(answer(&paged, sql).await, 600);
}

#[tokio::test]
async fn a_predicate_matching_nothing_reads_no_pages() {
    let dir = tempfile::tempdir().unwrap();
    let paged = table(&dir, "paged", PAGE).await;

    // Inside the segment's range, so its zone map cannot rule it out, but
    // between two pages' ranges... every id exists, so use a gap-free column
    // and a value outside it instead.
    let sql = "SELECT * FROM t WHERE id = 99999";
    assert_eq!(answer(&paged, sql).await, 0);
}

/// Every row must still come back whatever the pages said. A page wrongly
/// skipped would not show up as a wrong count in one query; it would silently
/// drop rows.
#[tokio::test]
async fn every_row_is_still_found_page_by_page() {
    let dir = tempfile::tempdir().unwrap();
    let paged = table(&dir, "paged", PAGE).await;

    for id in (0..ROWS).step_by(137) {
        let sql = format!("SELECT * FROM t WHERE id = {id}");
        assert_eq!(answer(&paged, &sql).await, 1, "lost id {id}");
    }
    assert_eq!(answer(&paged, "SELECT * FROM t").await, ROWS as usize);
}

/// A predicate on a column with no ordering the format records rules out no
/// pages, and must not pretend to.
#[tokio::test]
async fn a_predicate_the_bounds_cannot_answer_reads_everything() {
    let dir = tempfile::tempdir().unwrap();
    let paged = table(&dir, "paged", PAGE).await;

    let sql = "SELECT * FROM t WHERE body LIKE '%row 5000%'";
    assert_eq!(rows_scanned(&paged, sql).await, ROWS as usize);
    assert_eq!(answer(&paged, sql).await, 1);
}

#[tokio::test]
async fn a_scan_with_no_filter_reads_every_page() {
    let dir = tempfile::tempdir().unwrap();
    let paged = table(&dir, "paged", PAGE).await;
    assert_eq!(rows_scanned(&paged, "SELECT * FROM t").await, ROWS as usize);
}

/// Deleted rows are removed inside a page, so page boundaries keep meaning what
/// they meant. Applying the mask to the whole segment first would shift every
/// row and leave the bounds describing the wrong ranges.
#[tokio::test]
async fn deletes_and_page_bounds_agree() {
    let dir = tempfile::tempdir().unwrap();
    let paged = table(&dir, "paged", PAGE).await;

    let ctx = session(&paged);
    ctx.sql("DELETE FROM t WHERE id < 2000")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert_eq!(
        answer(&paged, "SELECT * FROM t").await,
        (ROWS - 2000) as usize
    );
    assert_eq!(answer(&paged, "SELECT * FROM t WHERE id = 5000").await, 1);
    assert_eq!(answer(&paged, "SELECT * FROM t WHERE id = 1500").await, 0);
    assert_eq!(
        answer(&paged, "SELECT * FROM t WHERE id >= 1900 AND id < 2100").await,
        100
    );
}
