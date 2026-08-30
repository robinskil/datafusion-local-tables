//! Substring pruning through SQL.
//!
//! A trigram filter rules out the segments that cannot contain a search term.
//! Half of these tests measure that it does. The other half measure that it
//! declines to, for the predicate shapes where acting would lose rows.

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

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("body", DataType::Utf8, true),
    ]))
}

/// Segment `k` gets its own vocabulary, so a term from one segment appears in
/// no other and pruning has something to find.
fn body_of(segment: i64, row: i64) -> String {
    format!("segment{segment} entry{row} common-filler-text")
}

async fn table(dir: &tempfile::TempDir, filters: BloomFilters) -> SessionContext {
    let path = dir.path().join("t.lt");
    let table = ColumnarTable::create(
        &path,
        schema(),
        TableOptions {
            durability: Durability::None,
            io_backend: IoBackend::Mmap,
            memtable_max_bytes: 64 * 1024 * 1024,
            trigram_filters: filters,
            ..TableOptions::default()
        },
    )
    .await
    .unwrap();

    for segment in 0..SEGMENTS {
        let ids: Vec<i64> = (0..PER_SEGMENT)
            .map(|r| segment * PER_SEGMENT + r)
            .collect();
        let bodies: Vec<String> = (0..PER_SEGMENT).map(|r| body_of(segment, r)).collect();
        let batch = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(bodies)),
            ],
        )
        .unwrap();
        table.insert(&[batch]).await.unwrap();
        table.flush().await.unwrap();
    }

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

/// The measurement: the same query, pruning nothing without filters and nearly
/// everything with them.
#[tokio::test]
async fn a_trigram_filter_prunes_a_substring_search() {
    let bare = tempfile::tempdir().unwrap();
    let without = table(&bare, BloomFilters::None).await;
    let filtered = tempfile::tempdir().unwrap();
    let with = table(&filtered, BloomFilters::All).await;

    let sql = "SELECT * FROM t WHERE body LIKE '%segment7 %'";

    assert_eq!(
        pruned(&plan_of(&without, sql).await),
        0,
        "nothing prunes a substring without a trigram filter"
    );
    assert!(
        pruned(&plan_of(&with, sql).await) >= 8,
        "only one segment holds that term"
    );

    assert_eq!(rows(&without, sql).await, PER_SEGMENT as usize);
    assert_eq!(rows(&with, sql).await, PER_SEGMENT as usize);
}

/// The property that must never break. A row that matches must come back,
/// whatever the filter said.
#[tokio::test]
async fn every_matching_row_is_still_returned() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = table(&dir, BloomFilters::All).await;

    for segment in 0..SEGMENTS {
        for row in [0, 37, PER_SEGMENT - 1] {
            let term = format!("entry{row} common");
            let sql = format!("SELECT * FROM t WHERE body LIKE '%segment{segment} {term}%'");
            assert_eq!(rows(&ctx, &sql).await, 1, "lost segment{segment} row{row}");
        }
    }
}

#[tokio::test]
async fn a_term_no_row_holds_prunes_every_segment() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = table(&dir, BloomFilters::All).await;

    let sql = "SELECT * FROM t WHERE body LIKE '%zzzqqqxxx%'";
    assert!(pruned(&plan_of(&ctx, sql).await) >= SEGMENTS as usize - 1);
    assert_eq!(rows(&ctx, sql).await, 0);
}

/// A term shared by every row rules nothing out, and must not pretend to.
#[tokio::test]
async fn a_term_every_row_holds_prunes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = table(&dir, BloomFilters::All).await;

    let sql = "SELECT * FROM t WHERE body LIKE '%common-filler%'";
    assert_eq!(pruned(&plan_of(&ctx, sql).await), 0);
    assert_eq!(rows(&ctx, sql).await, (SEGMENTS * PER_SEGMENT) as usize);
}

/// Every piece of the term must be present, so a term split across wildcards
/// prunes on both halves.
#[tokio::test]
async fn both_halves_of_a_split_pattern_must_be_present() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = table(&dir, BloomFilters::All).await;

    let sql = "SELECT * FROM t WHERE body LIKE '%common%zzzqqq%'";
    assert!(pruned(&plan_of(&ctx, sql).await) >= SEGMENTS as usize - 1);
    assert_eq!(rows(&ctx, sql).await, 0);
}

/// A term shorter than a trigram gives nothing to probe with.
#[tokio::test]
async fn a_term_shorter_than_a_piece_prunes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = table(&dir, BloomFilters::All).await;

    let sql = "SELECT * FROM t WHERE body LIKE '%q%'";
    assert_eq!(pruned(&plan_of(&ctx, sql).await), 0);
    assert_eq!(rows(&ctx, sql).await, 0, "no row holds a q");
}

/// `NOT LIKE` asks the opposite question. Requiring the term's pieces would
/// drop exactly the segments that answer it.
#[tokio::test]
async fn a_negated_like_prunes_nothing_and_answers_correctly() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = table(&dir, BloomFilters::All).await;

    let sql = "SELECT * FROM t WHERE body NOT LIKE '%segment7 %'";
    assert_eq!(pruned(&plan_of(&ctx, sql).await), 0);
    assert_eq!(
        rows(&ctx, sql).await,
        ((SEGMENTS - 1) * PER_SEGMENT) as usize
    );
}

/// The filter holds the bytes as written, so a case-insensitive term would
/// probe for pieces that were never stored.
#[tokio::test]
async fn a_case_insensitive_like_prunes_nothing_and_answers_correctly() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = table(&dir, BloomFilters::All).await;

    let sql = "SELECT * FROM t WHERE body ILIKE '%SEGMENT7 %'";
    assert_eq!(pruned(&plan_of(&ctx, sql).await), 0);
    assert_eq!(rows(&ctx, sql).await, PER_SEGMENT as usize);
}

/// An escape makes `%` an ordinary character, so splitting on it would invent
/// pieces the pattern never required.
#[tokio::test]
async fn an_escaped_pattern_prunes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = table(&dir, BloomFilters::All).await;

    let sql = r"SELECT * FROM t WHERE body LIKE '%seg\%ment%' ESCAPE '\'";
    assert_eq!(pruned(&plan_of(&ctx, sql).await), 0);
    assert_eq!(rows(&ctx, sql).await, 0);
}

/// Either branch of an OR can carry the match, so neither branch's pieces are
/// required of the segment.
#[tokio::test]
async fn an_or_prunes_nothing_and_answers_correctly() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = table(&dir, BloomFilters::All).await;

    let sql = "SELECT * FROM t WHERE body LIKE '%segment7 %' OR body LIKE '%segment2 %'";
    assert_eq!(pruned(&plan_of(&ctx, sql).await), 0);
    assert_eq!(rows(&ctx, sql).await, (2 * PER_SEGMENT) as usize);
}

/// Both sides of an AND must hold, so both sides may prune.
#[tokio::test]
async fn an_and_prunes_on_both_sides() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = table(&dir, BloomFilters::All).await;

    let sql = "SELECT * FROM t WHERE body LIKE '%segment7 %' AND body LIKE '%zzzqqq%'";
    assert!(pruned(&plan_of(&ctx, sql).await) >= SEGMENTS as usize - 1);
    assert_eq!(rows(&ctx, sql).await, 0);
}

/// A prefix pattern has no leading wildcard but still names a run of bytes.
#[tokio::test]
async fn a_prefix_pattern_prunes_too() {
    let dir = tempfile::tempdir().unwrap();
    let ctx = table(&dir, BloomFilters::All).await;

    let sql = "SELECT * FROM t WHERE body LIKE 'segment7 %'";
    assert!(pruned(&plan_of(&ctx, sql).await) >= 8);
    assert_eq!(rows(&ctx, sql).await, PER_SEGMENT as usize);
}

#[tokio::test]
async fn unflushed_rows_are_never_pruned_away() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    let table = ColumnarTable::create(
        &path,
        schema(),
        TableOptions {
            durability: Durability::None,
            trigram_filters: BloomFilters::All,
            ..TableOptions::default()
        },
    )
    .await
    .unwrap();
    let batch = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(vec![1i64])),
            Arc::new(StringArray::from(vec!["unflushed marker text"])),
        ],
    )
    .unwrap();
    table.insert(&[batch]).await.unwrap();

    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(ColumnarTableProvider::new(table)))
        .unwrap();
    assert_eq!(
        rows(&ctx, "SELECT * FROM t WHERE body LIKE '%marker%'").await,
        1
    );
}
