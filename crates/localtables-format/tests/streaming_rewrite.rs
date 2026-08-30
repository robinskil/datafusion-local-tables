//! Rewrites must not read the whole table into memory.
//!
//! Reading every live row before writing any is simple and unbounded: a table
//! larger than memory could then never be compacted, and its schema could never
//! change. Instead the work is cut into runs of bounded source bytes.
//!
//! Memory is not directly observable from a test, so these check the two things
//! that follow from the bounding and are observable: compaction commits once
//! per run rather than once overall, and both paths still return every row with
//! a budget far smaller than the table.

use std::sync::Arc;

use arrow_array::{Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use localtables_format::columnar::table::ColumnarTable;
use localtables_format::config::{Durability, TableOptions};

const SEGMENTS: i32 = 8;
const PER_SEGMENT: i32 = 500;
const ROWS: i32 = SEGMENTS * PER_SEGMENT;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("body", DataType::Utf8, false),
    ]))
}

/// A budget small enough that every segment is its own run.
fn options(compaction_max_bytes: u64) -> TableOptions {
    TableOptions {
        durability: Durability::None,
        min_row_group_rows: PER_SEGMENT as usize,
        row_group_rows: PER_SEGMENT as usize,
        compaction_max_bytes,
        ..TableOptions::default()
    }
}

async fn table(dir: &tempfile::TempDir, options: TableOptions) -> ColumnarTable {
    let table = ColumnarTable::create(&dir.path().join("t.lt"), schema(), options)
        .await
        .unwrap();
    for segment in 0..SEGMENTS {
        let ids: Vec<i32> = (segment * PER_SEGMENT..(segment + 1) * PER_SEGMENT).collect();
        let bodies: Vec<String> = ids
            .iter()
            .map(|i| format!("row {i} with some padding to give the segment a size"))
            .collect();
        table
            .insert(&[RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int32Array::from(ids)),
                    Arc::new(StringArray::from(bodies)),
                ],
            )
            .unwrap()])
            .await
            .unwrap();
        table.flush().await.unwrap();
    }
    table
}

async fn ids(table: &ColumnarTable) -> Vec<i64> {
    let snapshot = table.snapshot();
    let batches = table.scan(&snapshot, None).await.unwrap();
    let mut out = Vec::new();
    for batch in &batches {
        let column = batch.column_by_name("id").unwrap();
        match column.data_type() {
            DataType::Int32 => {
                let values = column.as_any().downcast_ref::<Int32Array>().unwrap();
                out.extend((0..values.len()).map(|r| values.value(r) as i64));
            }
            DataType::Int64 => {
                let values = column.as_any().downcast_ref::<Int64Array>().unwrap();
                out.extend((0..values.len()).map(|r| values.value(r)));
            }
            other => panic!("unexpected {other}"),
        }
    }
    out.sort_unstable();
    out
}

fn all_ids() -> Vec<i64> {
    (0..ROWS as i64).collect()
}

fn segment_ids(table: &ColumnarTable) -> Vec<u64> {
    table
        .snapshot()
        .live_segments()
        .map(|entry| entry.segment_id)
        .collect()
}

/// One run per segment means one commit per segment, which is what keeps the
/// memory bounded and the writer lock short.
#[tokio::test]
async fn a_small_budget_compacts_in_several_commits() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir, options(1)).await;
    let before = table.snapshot().txn_id;

    let rewritten = table.rewrite_all().await.unwrap();
    assert_eq!(rewritten, ROWS as u64);

    let commits = table.snapshot().txn_id - before;
    assert_eq!(
        commits, SEGMENTS as u64,
        "a budget of one byte should put every segment in its own run"
    );
    assert_eq!(ids(&table).await, all_ids());
}

/// A budget larger than the table is one run and one commit, which is what the
/// old behaviour was.
#[tokio::test]
async fn a_large_budget_compacts_in_one_commit() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir, options(u64::MAX)).await;
    let before = table.snapshot().txn_id;

    table.rewrite_all().await.unwrap();

    assert_eq!(table.snapshot().txn_id - before, 1);
    assert_eq!(ids(&table).await, all_ids());
}

/// Between those two, runs hold several segments each.
#[tokio::test]
async fn a_middling_budget_gives_fewer_runs_than_segments() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    let table = table(&dir, options(u64::MAX)).await;
    let segment_bytes = std::fs::metadata(&path).unwrap().len() / SEGMENTS as u64;
    drop(table);

    let table = ColumnarTable::open(&path, options(segment_bytes * 3))
        .await
        .unwrap();
    let before = table.snapshot().txn_id;
    table.rewrite_all().await.unwrap();

    let commits = table.snapshot().txn_id - before;
    assert!(
        commits > 1 && commits < SEGMENTS as u64,
        "runs should hold several segments each, got {commits} commits"
    );
    assert_eq!(ids(&table).await, all_ids());
}

/// A schema change cannot be split across commits, because a half-converted
/// table has segments of two types and nothing can read it. It bounds what it
/// holds instead.
#[tokio::test]
async fn a_cast_under_a_small_budget_is_still_one_commit() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir, options(1)).await;
    let before = table.snapshot().txn_id;

    table.cast_column("id", DataType::Int64).await.unwrap();

    let commits = table.snapshot().txn_id - before;
    assert_eq!(
        commits, 1,
        "the conversion and the schema must land together"
    );
    assert_eq!(table.schema().field(0).data_type(), &DataType::Int64);
    assert_eq!(ids(&table).await, all_ids());
}

#[tokio::test]
async fn a_drop_under_a_small_budget_keeps_every_row() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir, options(1)).await;

    table.drop_column("body").await.unwrap();

    assert_eq!(table.schema().fields().len(), 1);
    assert_eq!(ids(&table).await, all_ids());
}

/// A run that fails leaves the runs before it done and the rest untouched, and
/// running again finishes the job. That is the price of committing per run, and
/// it is why the table stays valid throughout rather than only at the ends.
#[tokio::test]
async fn compacting_a_subset_leaves_the_rest_alone() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir, options(1)).await;

    let all = segment_ids(&table);
    let half = &all[..all.len() / 2];
    table.compact_segments(half).await.unwrap();

    let after = segment_ids(&table);
    for untouched in &all[all.len() / 2..] {
        assert!(after.contains(untouched), "segment {untouched} was rewritten");
    }
    assert_eq!(ids(&table).await, all_ids());
}

#[tokio::test]
async fn a_segment_larger_than_the_budget_is_still_rewritten() {
    let dir = tempfile::tempdir().unwrap();
    // One segment holding everything, and a budget far below its size.
    let table = ColumnarTable::create(
        &dir.path().join("t.lt"),
        schema(),
        TableOptions {
            durability: Durability::None,
            compaction_max_bytes: 1,
            ..TableOptions::default()
        },
    )
    .await
    .unwrap();
    let ids_in: Vec<i32> = (0..ROWS).collect();
    let bodies: Vec<String> = ids_in.iter().map(|i| format!("row {i}")).collect();
    table
        .insert(&[RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(ids_in)),
                Arc::new(StringArray::from(bodies)),
            ],
        )
        .unwrap()])
        .await
        .unwrap();
    table.flush().await.unwrap();

    assert_eq!(table.rewrite_all().await.unwrap(), ROWS as u64);
    assert_eq!(ids(&table).await, all_ids());
}
