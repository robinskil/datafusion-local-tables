//! Tests for the columnar table.

use super::write::split_row_groups;
use super::*;
use arrow_array::{Int32Array, StringArray};
use arrow_schema::{DataType, Field, Schema};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

fn batch(ids: &[i32]) -> RecordBatch {
    let names: Vec<Option<String>> = ids
        .iter()
        .map(|i| {
            if i % 3 == 0 {
                None
            } else {
                Some(format!("row{i}"))
            }
        })
        .collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(ids.to_vec())),
            Arc::new(StringArray::from(names)),
        ],
    )
    .unwrap()
}

fn options() -> TableOptions {
    TableOptions {
        durability: crate::config::Durability::None,
        ..TableOptions::default()
    }
}

async fn table(dir: &tempfile::TempDir) -> ColumnarTable {
    ColumnarTable::create(&dir.path().join("t.lt"), schema(), options())
        .await
        .unwrap()
}

fn ids(batches: &[RecordBatch]) -> Vec<i32> {
    batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect()
}

async fn read(table: &ColumnarTable) -> Vec<i32> {
    let snapshot = table.snapshot();
    ids(&table.scan(&snapshot, None).await.unwrap())
}

/// Rows in the groups, in order, so nothing is lost or reordered.
fn grouped_ids(groups: &[Vec<RecordBatch>]) -> Vec<Vec<i32>> {
    groups
        .iter()
        .map(|group| {
            group
                .iter()
                .flat_map(|b| {
                    b.column(0)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values()
                        .to_vec()
                })
                .collect()
        })
        .collect()
}

#[test]
fn splitting_nothing_gives_no_groups() {
    assert!(split_row_groups(Vec::new(), 100).is_empty());
    assert!(split_row_groups(Vec::new(), 0).is_empty());
}

#[test]
fn a_limit_of_zero_means_one_group() {
    let groups = split_row_groups(vec![batch(&[1, 2]), batch(&[3])], 0);
    assert_eq!(grouped_ids(&groups), vec![vec![1, 2, 3]]);
}

#[test]
fn batches_that_fit_stay_whole_and_together() {
    let groups = split_row_groups(vec![batch(&[1, 2]), batch(&[3, 4])], 10);
    assert_eq!(
        grouped_ids(&groups),
        vec![vec![1, 2, 3, 4]],
        "nothing needs slicing, so nothing is copied"
    );
}

#[test]
fn a_group_closes_rather_than_overfilling() {
    // Three batches of two, limit three: the second cannot fit in the first
    // group without slicing, so the group closes at two instead.
    let groups = split_row_groups(vec![batch(&[1, 2]), batch(&[3, 4]), batch(&[5, 6])], 3);
    assert_eq!(
        grouped_ids(&groups),
        vec![vec![1, 2], vec![3, 4], vec![5, 6]]
    );
    assert!(groups
        .iter()
        .all(|g| g.iter().map(|b| b.num_rows()).sum::<usize>() <= 3));
}

#[test]
fn an_exact_multiple_splits_evenly() {
    let groups = split_row_groups(vec![batch(&[1, 2]), batch(&[3, 4])], 2);
    assert_eq!(grouped_ids(&groups), vec![vec![1, 2], vec![3, 4]]);
}

#[test]
fn a_batch_larger_than_a_group_is_sliced() {
    let groups = split_row_groups(vec![batch(&[1, 2, 3, 4, 5])], 2);
    assert_eq!(grouped_ids(&groups), vec![vec![1, 2], vec![3, 4], vec![5]]);
}

#[test]
fn splitting_never_loses_or_reorders_a_row() {
    let rows: Vec<i32> = (0..97).collect();
    for limit in [1usize, 2, 5, 10, 96, 97, 98, 1000] {
        let batches = rows.chunks(7).map(batch).collect::<Vec<_>>();
        let groups = split_row_groups(batches, limit);
        let flat: Vec<i32> = grouped_ids(&groups).into_iter().flatten().collect();
        assert_eq!(flat, rows, "limit {limit}");
        for group in &groups {
            let count: usize = group.iter().map(|b| b.num_rows()).sum();
            assert!(count <= limit, "limit {limit}: a group held {count}");
        }
    }
}

#[tokio::test]
async fn a_new_table_is_empty() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;

    assert_eq!(table.row_count(), 0);
    assert!(read(&table).await.is_empty());
}

#[tokio::test]
async fn inserted_rows_are_visible_before_any_flush() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;

    assert_eq!(table.insert(&[batch(&[1, 2, 3])]).await.unwrap(), 3);
    assert_eq!(table.row_count(), 3);
    assert_eq!(read(&table).await, vec![1, 2, 3]);
    assert_eq!(
        table.snapshot().manifest.segments.len(),
        0,
        "a small insert must not write a segment"
    );
    assert_eq!(table.memtable_rows().await, 3);
}

#[tokio::test]
async fn many_inserts_stay_in_one_memtable() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;

    for i in 0..20 {
        table.insert(&[batch(&[i])]).await.unwrap();
    }
    assert_eq!(read(&table).await, (0..20).collect::<Vec<i32>>());
    assert_eq!(table.snapshot().manifest.segments.len(), 0);
}

#[tokio::test]
async fn flushing_moves_the_rows_into_a_segment() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(&[1, 2])]).await.unwrap();
    table.insert(&[batch(&[3])]).await.unwrap();

    assert_eq!(table.flush().await.unwrap(), 3);
    assert_eq!(table.snapshot().manifest.segments.len(), 1);
    assert_eq!(table.memtable_rows().await, 0);
    assert_eq!(table.wal_bytes().await, 0, "a flush empties the log");
    assert_eq!(read(&table).await, vec![1, 2, 3]);
}

#[tokio::test]
async fn flushing_nothing_does_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    let before = table.snapshot().txn_id;

    assert_eq!(table.flush().await.unwrap(), 0);
    assert_eq!(table.snapshot().txn_id, before);
}

#[tokio::test]
async fn rows_written_before_and_after_a_flush_read_as_one_table() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;

    table.insert(&[batch(&[1, 2])]).await.unwrap();
    table.flush().await.unwrap();
    table.insert(&[batch(&[3, 4])]).await.unwrap();

    assert_eq!(read(&table).await, vec![1, 2, 3, 4]);
    assert_eq!(table.row_count(), 4);
}

#[tokio::test]
async fn unflushed_rows_survive_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    {
        let table = ColumnarTable::create(&path, schema(), options())
            .await
            .unwrap();
        table.insert(&[batch(&[1, 2])]).await.unwrap();
        table.insert(&[batch(&[3])]).await.unwrap();
        // No flush: the rows exist only in the log.
    }

    let table = ColumnarTable::open(&path, options()).await.unwrap();
    assert_eq!(read(&table).await, vec![1, 2, 3]);
    assert_eq!(
        table.memtable_rows().await,
        3,
        "replay puts the rows back in memory, not in a segment"
    );
}

#[tokio::test]
async fn flushed_rows_are_not_replayed_twice() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    {
        let table = ColumnarTable::create(&path, schema(), options())
            .await
            .unwrap();
        table.insert(&[batch(&[1, 2])]).await.unwrap();
        table.flush().await.unwrap();
        table.insert(&[batch(&[3])]).await.unwrap();
    }

    let table = ColumnarTable::open(&path, options()).await.unwrap();
    assert_eq!(read(&table).await, vec![1, 2, 3]);
    assert_eq!(table.row_count(), 3);
    assert_eq!(table.memtable_rows().await, 1);
}

#[tokio::test]
async fn repeated_reopens_do_not_multiply_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    {
        let table = ColumnarTable::create(&path, schema(), options())
            .await
            .unwrap();
        table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
    }
    for _ in 0..5 {
        let table = ColumnarTable::open(&path, options()).await.unwrap();
        assert_eq!(read(&table).await, vec![1, 2, 3]);
    }
}

#[tokio::test]
async fn memtable_rows_can_be_deleted_before_they_are_flushed() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(&[1, 2, 3, 4])]).await.unwrap();

    let seqnos = table.memtable_seqnos().await;
    assert_eq!(
        table
            .delete_memtable_rows(&[seqnos[1], seqnos[3]])
            .await
            .unwrap(),
        2
    );

    assert_eq!(read(&table).await, vec![1, 3]);
    assert_eq!(table.row_count(), 2);
}

#[tokio::test]
async fn a_deleted_memtable_row_stays_deleted_after_a_flush() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
    let seqnos = table.memtable_seqnos().await;
    table.delete_memtable_rows(&[seqnos[0]]).await.unwrap();

    table.flush().await.unwrap();
    assert_eq!(read(&table).await, vec![2, 3]);
}

#[tokio::test]
async fn a_deleted_memtable_row_stays_deleted_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    {
        let table = ColumnarTable::create(&path, schema(), options())
            .await
            .unwrap();
        table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
        let seqnos = table.memtable_seqnos().await;
        table.delete_memtable_rows(&[seqnos[1]]).await.unwrap();
    }

    let table = ColumnarTable::open(&path, options()).await.unwrap();
    assert_eq!(
        read(&table).await,
        vec![1, 3],
        "a delete logged against a memtable row must find it again after replay"
    );
}

#[tokio::test]
async fn segment_rows_can_be_deleted() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(&[1, 2, 3, 4, 5])]).await.unwrap();
    table.flush().await.unwrap();

    let segment = table.snapshot().manifest.segments[0].segment_id;
    assert_eq!(
        table
            .delete_positions(&[(segment, vec![1, 3])])
            .await
            .unwrap(),
        2
    );
    assert_eq!(read(&table).await, vec![1, 3, 5]);
    assert_eq!(table.row_count(), 3);
}

#[tokio::test]
async fn a_segment_delete_survives_a_reopen_without_a_flush() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    {
        let table = ColumnarTable::create(&path, schema(), options())
            .await
            .unwrap();
        table.insert(&[batch(&[1, 2, 3, 4])]).await.unwrap();
        table.flush().await.unwrap();
        let segment = table.snapshot().manifest.segments[0].segment_id;
        table
            .delete_positions(&[(segment, vec![0, 2])])
            .await
            .unwrap();
        // The delete is in the log, not yet in the file.
    }

    let table = ColumnarTable::open(&path, options()).await.unwrap();
    assert_eq!(read(&table).await, vec![2, 4]);
}

#[tokio::test]
async fn a_flush_writes_logged_deletes_into_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    {
        let table = ColumnarTable::create(&path, schema(), options())
            .await
            .unwrap();
        table.insert(&[batch(&[1, 2, 3, 4])]).await.unwrap();
        table.flush().await.unwrap();
        let segment = table.snapshot().manifest.segments[0].segment_id;
        table.delete_positions(&[(segment, vec![0])]).await.unwrap();
        table.flush().await.unwrap();

        assert!(
            table.snapshot().manifest.segments[0].deletes.is_some(),
            "a flush must record the bitmap in the commit"
        );
        assert_eq!(table.wal_bytes().await, 0);
    }

    let table = ColumnarTable::open(&path, options()).await.unwrap();
    assert_eq!(read(&table).await, vec![2, 3, 4]);
}

#[tokio::test]
async fn deleting_the_same_row_twice_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
    table.flush().await.unwrap();
    let segment = table.snapshot().manifest.segments[0].segment_id;

    assert_eq!(
        table.delete_positions(&[(segment, vec![1])]).await.unwrap(),
        1
    );
    let after_first = table.wal_bytes().await;
    assert_eq!(
        table.delete_positions(&[(segment, vec![1])]).await.unwrap(),
        0
    );
    assert_eq!(
        table.wal_bytes().await,
        after_first,
        "a delete that changes nothing must not reach the log"
    );
}

#[tokio::test]
async fn positions_past_the_end_of_a_segment_are_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
    table.flush().await.unwrap();
    let segment = table.snapshot().manifest.segments[0].segment_id;

    assert_eq!(
        table
            .delete_positions(&[(segment, vec![99])])
            .await
            .unwrap(),
        0
    );
    assert_eq!(table.row_count(), 3);
}

#[tokio::test]
async fn deleting_an_unknown_segment_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    let err = table.delete_positions(&[(42, vec![0])]).await.unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "got {err:?}");
}

#[tokio::test]
async fn a_batch_with_the_wrong_schema_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    let other = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
    let wrong = RecordBatch::try_new(
        other,
        vec![Arc::new(arrow_array::Int64Array::from(vec![1i64]))],
    )
    .unwrap();

    let err = table.insert(&[wrong]).await.unwrap_err();
    assert!(matches!(err, Error::SchemaMismatch(_)), "got {err:?}");
    assert_eq!(table.row_count(), 0);
}

#[tokio::test]
async fn inserting_nothing_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;

    assert_eq!(table.insert(&[]).await.unwrap(), 0);
    assert_eq!(table.insert(&[batch(&[])]).await.unwrap(), 0);
    assert_eq!(table.wal_bytes().await, 0);
}

#[tokio::test]
async fn the_memtable_flushes_itself_once_it_grows_too_large() {
    let dir = tempfile::tempdir().unwrap();
    let mut opts = options();
    opts.memtable_max_bytes = 8 * 1024;
    let table = ColumnarTable::create(&dir.path().join("t.lt"), schema(), opts)
        .await
        .unwrap();

    let rows: Vec<i32> = (0..2000).collect();
    for chunk in rows.chunks(100) {
        table.insert(&[batch(chunk)]).await.unwrap();
    }

    assert!(
        !table.snapshot().manifest.segments.is_empty(),
        "growth past the limit must trigger a flush"
    );
    assert_eq!(read(&table).await, rows);
}

#[tokio::test]
async fn dropping_a_segment_reclaims_its_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
    table.flush().await.unwrap();
    table.insert(&[batch(&[4, 5, 6])]).await.unwrap();
    table.flush().await.unwrap();

    let first = table.snapshot().manifest.segments[0].clone();
    assert_eq!(table.delete_segments(&[first.segment_id]).await.unwrap(), 3);

    let snapshot = table.snapshot();
    assert_eq!(snapshot.manifest.segments.len(), 1);
    assert_eq!(read(&table).await, vec![4, 5, 6]);
    assert!(
        snapshot
            .manifest
            .free_extents
            .iter()
            .any(|f| f.extent == first.data),
        "the dropped segment's bytes must become reclaimable"
    );
}

#[tokio::test]
async fn a_pinned_snapshot_keeps_reading_what_it_pinned() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(&[1, 2, 3, 4])]).await.unwrap();
    table.flush().await.unwrap();

    let pinned = table.snapshot();
    let segment = pinned.manifest.segments[0].segment_id;
    table
        .delete_positions(&[(segment, vec![0, 1])])
        .await
        .unwrap();

    assert_eq!(
        ids(&table.scan(&pinned, None).await.unwrap()),
        vec![1, 2, 3, 4],
        "the pinned snapshot must not see the later delete"
    );
    assert_eq!(read(&table).await, vec![3, 4]);
}

#[tokio::test]
async fn a_pinned_snapshot_keeps_the_memtable_rows_it_saw() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(&[1, 2, 3])]).await.unwrap();

    let pinned = table.snapshot();
    table.insert(&[batch(&[4])]).await.unwrap();
    table.flush().await.unwrap();

    assert_eq!(
        ids(&table.scan(&pinned, None).await.unwrap()),
        vec![1, 2, 3],
        "a snapshot taken before the fourth row must not show it"
    );
    assert_eq!(read(&table).await, vec![1, 2, 3, 4]);
}

#[tokio::test]
async fn projection_reads_only_what_was_asked_for() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
    table.flush().await.unwrap();
    table.insert(&[batch(&[4])]).await.unwrap();

    let snapshot = table.snapshot();
    let read = table.scan(&snapshot, Some(&[1])).await.unwrap();
    assert!(read.iter().all(|b| b.num_columns() == 1));
    assert!(read.iter().all(|b| b.schema().field(0).name() == "name"));
    assert_eq!(read.iter().map(|b| b.num_rows()).sum::<usize>(), 4);
}

#[tokio::test]
async fn scans_are_cut_into_batches() {
    let dir = tempfile::tempdir().unwrap();
    let mut opts = options();
    opts.scan_batch_rows = 100;
    let table = ColumnarTable::create(&dir.path().join("t.lt"), schema(), opts)
        .await
        .unwrap();

    let rows: Vec<i32> = (0..250).collect();
    table.insert(&[batch(&rows)]).await.unwrap();
    table.flush().await.unwrap();

    let snapshot = table.snapshot();
    let read = table.scan(&snapshot, None).await.unwrap();
    assert_eq!(read.len(), 3);
    assert_eq!(read[0].num_rows(), 100);
    assert_eq!(read[2].num_rows(), 50);
    assert_eq!(ids(&read), rows);
}

#[tokio::test]
async fn compaction_candidates_appear_once_deletes_dominate() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table
        .insert(&[batch(&(0..10).collect::<Vec<i32>>())])
        .await
        .unwrap();
    table.flush().await.unwrap();
    let segment = table.snapshot().manifest.segments[0].segment_id;

    table
        .delete_positions(&[(segment, vec![0, 1, 2, 3])])
        .await
        .unwrap();
    assert!(
        table.compaction_candidates(0.5).is_empty(),
        "four of ten is not half"
    );

    table.delete_positions(&[(segment, vec![4])]).await.unwrap();
    assert_eq!(table.compaction_candidates(0.5), vec![segment]);
}

#[tokio::test]
async fn a_second_writer_is_refused_while_the_table_is_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    let _held = ColumnarTable::create(&path, schema(), options())
        .await
        .unwrap();

    let err = ColumnarTable::open(&path, options()).await.unwrap_err();
    assert!(matches!(err, Error::WriterLocked(_)), "got {err:?}");
}

#[tokio::test]
async fn the_log_files_sit_beside_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(&[1])]).await.unwrap();

    assert!(dir.path().join("t.lt.wal0").exists());
    assert!(dir.path().join("t.lt.wal1").exists());
}
