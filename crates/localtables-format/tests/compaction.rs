//! Compaction rewrites segments that deletes have hollowed out.
//!
//! The rows a scan returns must not change. What changes is the space: a
//! deleted row keeps its bytes until the segment is rewritten without it.

use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use localtables_format::columnar::table::ColumnarTable;
use localtables_format::config::{Durability, IoBackend, TableOptions};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

fn batch(ids: std::ops::Range<i64>) -> RecordBatch {
    let ids: Vec<i64> = ids.collect();
    let names: Vec<Option<String>> = ids
        .iter()
        .map(|i| (i % 4 != 0).then(|| format!("name-{i}")))
        .collect();
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

async fn table(dir: &tempfile::TempDir) -> ColumnarTable {
    ColumnarTable::create(&dir.path().join("t.lt"), schema(), options())
        .await
        .unwrap()
}

async fn ids(table: &ColumnarTable) -> Vec<i64> {
    let snapshot = table.snapshot();
    let mut out: Vec<i64> = table
        .scan(&snapshot, None)
        .await
        .unwrap()
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

/// Every row must still carry the name its id implies.
async fn check_rows_intact(table: &ColumnarTable) {
    let snapshot = table.snapshot();
    for batch in table.scan(&snapshot, None).await.unwrap() {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let names = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let id = ids.value(row);
            if id % 4 == 0 {
                assert!(names.is_null(row), "row {id} lost its null");
            } else {
                assert_eq!(
                    names.value(row),
                    format!("name-{id}"),
                    "row {id} is damaged"
                );
            }
        }
    }
}

#[tokio::test]
async fn compaction_drops_the_deleted_rows_and_keeps_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(0..100)]).await.unwrap();
    table.flush().await.unwrap();

    let segment = table.snapshot().manifest.segments[0].segment_id;
    let doomed: Vec<u32> = (0..60).collect();
    table.delete_positions(&[(segment, doomed)]).await.unwrap();

    let before = ids(&table).await;
    assert_eq!(before, (60..100).collect::<Vec<i64>>());

    assert_eq!(table.compact(0.5).await.unwrap(), 40);
    assert_eq!(
        ids(&table).await,
        before,
        "compaction must not change the rows"
    );
    check_rows_intact(&table).await;

    let snapshot = table.snapshot();
    assert_eq!(snapshot.manifest.segments.len(), 1);
    assert_eq!(
        snapshot.manifest.segments[0].row_count, 40,
        "the rewritten segment holds only the live rows"
    );
    assert!(
        snapshot.manifest.segments[0].deletes.is_none(),
        "a rewritten segment starts with nothing deleted"
    );
}

#[tokio::test]
async fn compaction_frees_the_bytes_the_old_segment_held() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(0..5000)]).await.unwrap();
    table.flush().await.unwrap();

    let old = table.snapshot().manifest.segments[0].clone();
    table
        .delete_positions(&[(old.segment_id, (0..4900).collect())])
        .await
        .unwrap();

    table.compact(0.5).await.unwrap();

    let snapshot = table.snapshot();
    assert!(
        snapshot
            .manifest
            .free_extents
            .iter()
            .any(|f| f.extent == old.data),
        "the rewritten segment's bytes must become reclaimable"
    );
    assert!(
        snapshot.manifest.segments[0].data.len < old.data.len,
        "a segment of 100 rows should be smaller than one of 5000"
    );
}

#[tokio::test]
async fn compaction_merges_several_segments_into_one() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    for round in 0..4i64 {
        table
            .insert(&[batch(round * 50..round * 50 + 50)])
            .await
            .unwrap();
        table.flush().await.unwrap();
    }

    // Hollow out every segment.
    let segments: Vec<u64> = table
        .snapshot()
        .manifest
        .segments
        .iter()
        .map(|s| s.segment_id)
        .collect();
    for segment in &segments {
        table
            .delete_positions(&[(*segment, (0..30).collect())])
            .await
            .unwrap();
    }

    let before = ids(&table).await;
    assert_eq!(table.compact(0.5).await.unwrap(), 80);

    assert_eq!(ids(&table).await, before);
    assert_eq!(
        table.snapshot().manifest.segments.len(),
        1,
        "four hollowed segments become one"
    );
}

#[tokio::test]
async fn compaction_leaves_healthy_segments_alone() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(0..100)]).await.unwrap();
    table.flush().await.unwrap();
    table.insert(&[batch(100..200)]).await.unwrap();
    table.flush().await.unwrap();

    let hollow = table.snapshot().manifest.segments[0].segment_id;
    table
        .delete_positions(&[(hollow, (0..90).collect())])
        .await
        .unwrap();

    let before = ids(&table).await;
    table.compact(0.5).await.unwrap();

    assert_eq!(ids(&table).await, before);
    let segments = &table.snapshot().manifest.segments;
    assert_eq!(segments.len(), 2, "only the hollow segment was rewritten");
    assert!(
        segments.iter().any(|s| s.row_count == 100),
        "the untouched segment keeps its rows"
    );
    assert!(segments.iter().any(|s| s.row_count == 10));
}

#[tokio::test]
async fn compaction_with_nothing_to_do_does_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(0..50)]).await.unwrap();
    table.flush().await.unwrap();

    let before = table.snapshot().txn_id;
    assert_eq!(table.compact(0.5).await.unwrap(), 0);
    assert_eq!(
        table.snapshot().txn_id,
        before,
        "nothing to rewrite means nothing to commit"
    );
}

#[tokio::test]
async fn compacting_an_entirely_deleted_segment_removes_it() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(0..20)]).await.unwrap();
    table.flush().await.unwrap();

    let segment = table.snapshot().manifest.segments[0].segment_id;
    table
        .delete_positions(&[(segment, (0..20).collect())])
        .await
        .unwrap();

    assert_eq!(table.compact(0.5).await.unwrap(), 0);
    assert!(table.snapshot().manifest.segments.is_empty());
    assert!(ids(&table).await.is_empty());
    assert_eq!(table.row_count(), 0);
}

#[tokio::test]
async fn a_reader_pinned_before_compaction_still_reads_its_rows() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(0..200)]).await.unwrap();
    table.flush().await.unwrap();

    let segment = table.snapshot().manifest.segments[0].segment_id;
    table
        .delete_positions(&[(segment, (0..150).collect())])
        .await
        .unwrap();

    // Pin, then rewrite everything the reader is holding.
    let pinned = table.snapshot();
    table.compact(0.5).await.unwrap();

    let pinned_ids: Vec<i64> = table
        .scan(&pinned, None)
        .await
        .unwrap()
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
    assert_eq!(
        pinned_ids,
        (150..200).collect::<Vec<i64>>(),
        "the pinned snapshot must still read the segment that was rewritten"
    );
    assert_eq!(ids(&table).await, (150..200).collect::<Vec<i64>>());
}

#[tokio::test]
async fn a_compacted_table_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    let expected: Vec<i64>;
    {
        let table = ColumnarTable::create(&path, schema(), options())
            .await
            .unwrap();
        table.insert(&[batch(0..100)]).await.unwrap();
        table.flush().await.unwrap();
        let segment = table.snapshot().manifest.segments[0].segment_id;
        table
            .delete_positions(&[(segment, (0..80).collect())])
            .await
            .unwrap();
        table.compact(0.5).await.unwrap();
        expected = ids(&table).await;
    }

    let table = ColumnarTable::open(&path, options()).await.unwrap();
    assert_eq!(ids(&table).await, expected);
    check_rows_intact(&table).await;
}

#[tokio::test]
async fn compaction_leaves_the_memtable_alone() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(0..100)]).await.unwrap();
    table.flush().await.unwrap();
    // These are in the log, not in a segment.
    table.insert(&[batch(100..120)]).await.unwrap();

    let segment = table.snapshot().manifest.segments[0].segment_id;
    table
        .delete_positions(&[(segment, (0..70).collect())])
        .await
        .unwrap();

    let before = ids(&table).await;
    table.compact(0.5).await.unwrap();

    assert_eq!(ids(&table).await, before);
    assert_eq!(
        table.memtable_rows().await,
        20,
        "compaction rewrites segments; it does not flush"
    );
}

#[tokio::test]
async fn compacting_an_unknown_segment_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    let err = table.compact_segments(&[999]).await.unwrap_err();
    assert!(
        matches!(err, localtables_format::Error::InvalidArgument(_)),
        "got {err:?}"
    );
}
