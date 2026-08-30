//! Membership filters, through SQL.
//!
//! The case these exist for is a column whose values are scattered rather than
//! clustered. Zone maps prune by range, so when every segment's range covers
//! the value being looked for, they rule nothing out and the scan reads the
//! whole table to return one row. The tests below build exactly that table and
//! measure what each kind of statistic prunes on it.

use std::sync::Arc;

use arrow::array::{Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::SessionContext;

use datafusion_local_tables::ColumnarTableProvider;
use localtables_format::columnar::table::ColumnarTable;
use localtables_format::config::{BloomFilters, Durability, IoBackend, TableOptions};

const SEGMENTS: i64 = 10;
const PER_SEGMENT: i64 = 100;
const ROWS: i64 = SEGMENTS * PER_SEGMENT;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("email", DataType::Utf8, true),
    ]))
}

fn options(filters: BloomFilters) -> TableOptions {
    TableOptions {
        durability: Durability::None,
        io_backend: IoBackend::Mmap,
        memtable_max_bytes: 64 * 1024 * 1024,
        bloom_filters: filters,
        ..TableOptions::default()
    }
}

/// Keys for one segment, spread across the whole range.
///
/// Segment `k` holds `k`, `k + 10`, `k + 20` and so on, so its smallest and
/// largest keys are nearly the table's own. Every segment therefore survives a
/// zone map test for any key at all, which is what makes this table the
/// interesting one.
fn keys_of(segment: i64) -> Vec<i64> {
    (0..PER_SEGMENT).map(|i| segment + i * SEGMENTS).collect()
}

async fn table(dir: &tempfile::TempDir, filters: BloomFilters) -> (ColumnarTable, SessionContext) {
    let path = dir.path().join("t.lt");
    let table = ColumnarTable::create(&path, schema(), options(filters))
        .await
        .unwrap();

    for segment in 0..SEGMENTS {
        let keys = keys_of(segment);
        let emails: Vec<String> = keys
            .iter()
            .map(|k| format!("user{k}@example.com"))
            .collect();
        let batch = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(keys)),
                Arc::new(StringArray::from(emails)),
            ],
        )
        .unwrap();
        table.insert(&[batch]).await.unwrap();
        table.flush().await.unwrap();
    }

    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(ColumnarTableProvider::new(table.clone())))
        .unwrap();
    (table, ctx)
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

/// Segments the plan says it ruled out.
///
/// Asserted as a floor rather than an exact figure: a membership filter admits
/// false positives, so a segment that holds none of the values can still fail
/// to be ruled out. That costs a read and never a row, and which values it
/// happens to affect is a property of the hash, not of the data.
fn pruned(plan: &str) -> usize {
    let at = plan
        .find("pruned=")
        .expect("the scan reports what it pruned");
    plan[at + "pruned=".len()..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .expect("a number after pruned=")
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

/// The measurement this feature exists for: the same query on the same rows,
/// pruning nothing without filters and almost everything with them.
#[tokio::test]
async fn a_filter_prunes_where_a_zone_map_cannot() {
    let bare = tempfile::tempdir().unwrap();
    let (_bare_table, without) = table(&bare, BloomFilters::None).await;
    let filtered = tempfile::tempdir().unwrap();
    let (_filtered_table, with) = table(&filtered, BloomFilters::All).await;

    let sql = "SELECT * FROM t WHERE key = 550";
    let bare_plan = plan_of(&without, sql).await;
    assert_eq!(
        pruned(&bare_plan),
        0,
        "every segment's range covers key 550, so zone maps prune none:\n{bare_plan}"
    );

    let filtered_plan = plan_of(&with, sql).await;
    assert!(
        pruned(&filtered_plan) >= 8,
        "one segment holds key 550, so the rest should have gone:\n{filtered_plan}"
    );

    assert_eq!(rows(&without, sql).await, 1);
    assert_eq!(rows(&with, sql).await, 1);
}

/// The property a filter must never break. A false negative here would not
/// show up as a wrong count in one query; it would silently drop a row.
#[tokio::test]
async fn every_key_is_still_found_with_filters_on() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, BloomFilters::All).await;

    for key in 0..ROWS {
        let found = rows(&ctx, &format!("SELECT * FROM t WHERE key = {key}")).await;
        assert_eq!(found, 1, "key {key} was lost");
    }
}

#[tokio::test]
async fn a_string_column_prunes_by_membership_too() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, BloomFilters::All).await;

    let sql = "SELECT * FROM t WHERE email = 'user550@example.com'";
    let plan = plan_of(&ctx, sql).await;
    assert!(
        pruned(&plan) >= 8,
        "one segment holds that address:\n{plan}"
    );
    assert_eq!(rows(&ctx, sql).await, 1);
}

#[tokio::test]
async fn a_value_no_segment_holds_prunes_all_of_them() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, BloomFilters::All).await;

    let sql = "SELECT * FROM t WHERE key = 999999";
    // A zone map already rules this out, being outside every range; the point
    // is that adding a filter does not make it worse.
    assert_eq!(rows(&ctx, sql).await, 0);

    // This one is inside every range, so only a filter can rule it out.
    let inside = "SELECT * FROM t WHERE email = 'nobody@example.com'";
    let plan = plan_of(&ctx, inside).await;
    assert!(
        pruned(&plan) >= SEGMENTS as usize - 1,
        "no segment holds that address:\n{plan}"
    );
    assert_eq!(rows(&ctx, inside).await, 0);
}

#[tokio::test]
async fn a_set_of_values_prunes_to_the_segments_that_hold_them() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, BloomFilters::All).await;

    // Keys 3 and 7 sit in segments 3 and 7, so eight segments hold neither.
    let sql = "SELECT key FROM t WHERE key IN (3, 7)";
    let plan = plan_of(&ctx, sql).await;
    assert!(pruned(&plan) >= 7, "two segments hold these keys:\n{plan}");
    assert_eq!(rows(&ctx, sql).await, 2);
}

#[tokio::test]
async fn naming_one_column_leaves_the_other_without_a_filter() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, BloomFilters::Columns(vec!["key".to_string()])).await;

    let keyed = plan_of(&ctx, "SELECT * FROM t WHERE key = 550").await;
    assert!(pruned(&keyed) >= 8, "{keyed}");

    let unkeyed = plan_of(&ctx, "SELECT * FROM t WHERE email = 'user550@example.com'").await;
    assert_eq!(
        pruned(&unkeyed),
        0,
        "email was not asked for, so nothing prunes it:\n{unkeyed}"
    );
    assert_eq!(
        rows(&ctx, "SELECT * FROM t WHERE email = 'user550@example.com'").await,
        1
    );
}

/// A range still goes through zone maps, which know nothing here. A filter
/// answers equality only, and must not be consulted for anything else.
#[tokio::test]
async fn a_range_predicate_is_unaffected() {
    let dir = tempfile::tempdir().unwrap();
    let (_table, ctx) = table(&dir, BloomFilters::All).await;

    let sql = "SELECT key FROM t WHERE key > 100 AND key < 110";
    assert_eq!(rows(&ctx, sql).await, 9);
}

/// Rows still in the memtable have no segment and no filter. They must be
/// returned regardless of what the filters on the flushed segments say.
#[tokio::test]
async fn unflushed_rows_are_never_pruned_away() {
    let dir = tempfile::tempdir().unwrap();
    let (table, ctx) = table(&dir, BloomFilters::All).await;

    let batch = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(vec![ROWS + 1])),
            Arc::new(StringArray::from(vec!["fresh@example.com"])),
        ],
    )
    .unwrap();
    table.insert(&[batch]).await.unwrap();

    assert_eq!(
        rows(&ctx, &format!("SELECT * FROM t WHERE key = {}", ROWS + 1)).await,
        1
    );
}
