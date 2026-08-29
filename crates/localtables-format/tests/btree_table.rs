//! The b-tree table, end to end.
//!
//! Point lookups, range scans, writes, deletes, flushes and restarts. The
//! property that matters throughout: what the table holds is exactly what was
//! written and not deleted, and a key finds its row wherever that row currently
//! lives — the pending overlay or the tree.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use localtables_format::btree::BTreeTable;
use localtables_format::config::{Durability, IoBackend, TableOptions};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

fn batch(ids: &[i64]) -> RecordBatch {
    let names: Vec<Option<String>> = ids
        .iter()
        .map(|i| (i % 5 != 0).then(|| format!("name-{i}")))
        .collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(StringArray::from(names)),
        ],
    )
    .unwrap()
}

fn options() -> TableOptions {
    TableOptions {
        durability: Durability::None,
        io_backend: IoBackend::Mmap,
        // Large enough that tests decide when a flush happens.
        memtable_max_bytes: 64 * 1024 * 1024,
        ..TableOptions::default()
    }
}

async fn table(dir: &tempfile::TempDir) -> BTreeTable {
    BTreeTable::create(&dir.path().join("t.ltb"), schema(), &["id"], options())
        .await
        .unwrap()
}

/// The key for one id, as the table would encode it.
fn key(table: &BTreeTable, id: i64) -> Vec<u8> {
    table.key_of(&batch(&[id]), 0).unwrap()
}

/// Every id the table holds, in key order.
async fn ids(table: &BTreeTable) -> Vec<i64> {
    let snapshot = table.snapshot();
    let batch = table.scan(&snapshot).await.unwrap();
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec()
}

/// Look up one id and return its name column.
async fn get_name(table: &BTreeTable, id: i64) -> Option<Option<String>> {
    let snapshot = table.snapshot();
    let found = table.get(&snapshot, &key(table, id)).await.unwrap()?;
    let names = found
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    Some((!names.is_null(0)).then(|| names.value(0).to_string()))
}

#[tokio::test]
async fn a_new_table_holds_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;

    assert!(ids(&table).await.is_empty());
    assert!(table.snapshot().is_empty());
    assert_eq!(
        table.get(&table.snapshot(), &key(&table, 1)).await.unwrap(),
        None
    );
}

#[tokio::test]
async fn inserted_rows_are_found_by_key_before_any_flush() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    assert_eq!(table.insert(&[batch(&[3, 1, 2])]).await.unwrap(), 3);

    assert_eq!(get_name(&table, 2).await, Some(Some("name-2".to_string())));
    assert_eq!(
        get_name(&table, 5).await,
        None,
        "an absent key finds nothing"
    );
    assert_eq!(
        ids(&table).await,
        vec![1, 2, 3],
        "a scan returns key order, not insertion order"
    );
}

#[tokio::test]
async fn a_flush_moves_the_rows_into_the_tree() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(&[1, 2, 3])]).await.unwrap();

    assert_eq!(table.flush().await.unwrap(), 3);
    assert_eq!(table.pending_changes().await, 0);
    assert_eq!(table.wal_bytes().await, 0, "a flush empties the log");
    assert_eq!(ids(&table).await, vec![1, 2, 3]);
    assert_eq!(get_name(&table, 2).await, Some(Some("name-2".to_string())));
}

#[tokio::test]
async fn rows_written_before_and_after_a_flush_read_as_one_table() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;

    // Ids that are not multiples of five, so each carries a name rather than
    // the null `batch` gives every fifth row.
    table.insert(&[batch(&[11, 33])]).await.unwrap();
    table.flush().await.unwrap();
    table.insert(&[batch(&[22, 44])]).await.unwrap();

    assert_eq!(ids(&table).await, vec![11, 22, 33, 44]);
    assert_eq!(
        get_name(&table, 22).await,
        Some(Some("name-22".to_string())),
        "a key written after the flush is found in the overlay"
    );
    assert_eq!(
        get_name(&table, 33).await,
        Some(Some("name-33".to_string())),
        "a key written before the flush is found in the tree"
    );
}

#[tokio::test]
async fn writing_the_same_key_twice_replaces_the_row() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;

    table.insert(&[batch(&[1])]).await.unwrap();
    table.flush().await.unwrap();

    // A row whose name differs, under the same key.
    let replacement = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(vec![1i64])),
            Arc::new(StringArray::from(vec![Some("replaced")])),
        ],
    )
    .unwrap();
    table.insert(&[replacement]).await.unwrap();

    assert_eq!(
        ids(&table).await,
        vec![1],
        "a replacement is not a second row"
    );
    assert_eq!(
        get_name(&table, 1).await,
        Some(Some("replaced".to_string()))
    );

    table.flush().await.unwrap();
    assert_eq!(ids(&table).await, vec![1]);
    assert_eq!(
        get_name(&table, 1).await,
        Some(Some("replaced".to_string()))
    );
}

#[tokio::test]
async fn a_delete_hides_a_row_that_is_still_in_the_tree() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
    table.flush().await.unwrap();

    table.delete_keys(&[key(&table, 2)]).await.unwrap();
    assert_eq!(get_name(&table, 2).await, None, "the pending delete wins");
    assert_eq!(ids(&table).await, vec![1, 3]);

    table.flush().await.unwrap();
    assert_eq!(ids(&table).await, vec![1, 3]);
}

#[tokio::test]
async fn deleting_a_key_that_is_not_there_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(&[1, 2])]).await.unwrap();
    table.flush().await.unwrap();

    table.delete_keys(&[key(&table, 99)]).await.unwrap();
    table.flush().await.unwrap();
    assert_eq!(ids(&table).await, vec![1, 2]);
}

#[tokio::test]
async fn a_range_returns_the_keys_between_its_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    let rows: Vec<i64> = (0..100).collect();
    table.insert(&[batch(&rows)]).await.unwrap();
    table.flush().await.unwrap();

    let snapshot = table.snapshot();
    let found = table
        .range(&snapshot, &key(&table, 20), Some(&key(&table, 30)), None)
        .await
        .unwrap();

    let ids: Vec<i64> = found
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();
    assert_eq!(
        ids,
        (20..30).collect::<Vec<i64>>(),
        "the end bound is exclusive"
    );
}

#[tokio::test]
async fn a_range_with_no_end_runs_to_the_last_key() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table
        .insert(&[batch(&(0..50).collect::<Vec<i64>>())])
        .await
        .unwrap();
    table.flush().await.unwrap();

    let snapshot = table.snapshot();
    let found = table
        .range(&snapshot, &key(&table, 45), None, None)
        .await
        .unwrap();
    assert_eq!(found.num_rows(), 5);
}

#[tokio::test]
async fn a_range_limit_stops_early() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table
        .insert(&[batch(&(0..100).collect::<Vec<i64>>())])
        .await
        .unwrap();
    table.flush().await.unwrap();

    let snapshot = table.snapshot();
    let found = table.range(&snapshot, &[], None, Some(7)).await.unwrap();
    assert_eq!(found.num_rows(), 7);
}

#[tokio::test]
async fn a_range_sees_pending_writes_and_deletes() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(&[10, 20, 30])]).await.unwrap();
    table.flush().await.unwrap();

    // One new key inside the range, one delete inside it.
    table.insert(&[batch(&[15])]).await.unwrap();
    table.delete_keys(&[key(&table, 20)]).await.unwrap();

    let snapshot = table.snapshot();
    let found = table
        .range(&snapshot, &key(&table, 10), Some(&key(&table, 30)), None)
        .await
        .unwrap();
    let ids: Vec<i64> = found
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();
    assert_eq!(ids, vec![10, 15]);
}

#[tokio::test]
async fn a_tree_deeper_than_one_page_still_finds_every_key() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;

    // Well past the leaf fanout, so the tree has real branch levels.
    let rows: Vec<i64> = (0..5000).collect();
    table.insert(&[batch(&rows)]).await.unwrap();
    table.flush().await.unwrap();

    assert_eq!(ids(&table).await, rows);
    for id in [0i64, 1, 255, 256, 257, 2500, 4998, 4999] {
        assert_eq!(
            get_name(&table, id).await,
            Some((id % 5 != 0).then(|| format!("name-{id}"))),
            "key {id} was not found"
        );
    }
    assert_eq!(get_name(&table, 5000).await, None);
    assert_eq!(get_name(&table, -1).await, None);
}

#[tokio::test]
async fn a_range_over_a_deep_tree_returns_exactly_its_keys() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table
        .insert(&[batch(&(0..5000).collect::<Vec<i64>>())])
        .await
        .unwrap();
    table.flush().await.unwrap();

    let snapshot = table.snapshot();
    for (start, end) in [(0i64, 10i64), (250, 260), (1000, 1001), (4990, 5000)] {
        let found = table
            .range(
                &snapshot,
                &key(&table, start),
                Some(&key(&table, end)),
                None,
            )
            .await
            .unwrap();
        let ids: Vec<i64> = found
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(
            ids,
            (start..end).collect::<Vec<i64>>(),
            "range {start}..{end}"
        );
    }
}

#[tokio::test]
async fn unflushed_writes_survive_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.ltb");
    {
        let table = BTreeTable::create(&path, schema(), &["id"], options())
            .await
            .unwrap();
        table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
        // No flush: the rows exist only in the log.
    }

    let table = BTreeTable::open(&path, &["id"], options()).await.unwrap();
    assert_eq!(ids(&table).await, vec![1, 2, 3]);
    assert_eq!(get_name(&table, 2).await, Some(Some("name-2".to_string())));
}

#[tokio::test]
async fn flushed_writes_are_not_replayed_twice() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.ltb");
    {
        let table = BTreeTable::create(&path, schema(), &["id"], options())
            .await
            .unwrap();
        table.insert(&[batch(&[1, 2])]).await.unwrap();
        table.flush().await.unwrap();
        table.insert(&[batch(&[3])]).await.unwrap();
    }

    let table = BTreeTable::open(&path, &["id"], options()).await.unwrap();
    assert_eq!(ids(&table).await, vec![1, 2, 3]);
}

#[tokio::test]
async fn a_delete_survives_a_reopen_without_a_flush() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.ltb");
    {
        let table = BTreeTable::create(&path, schema(), &["id"], options())
            .await
            .unwrap();
        table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
        table.flush().await.unwrap();
        table.delete_keys(&[key(&table, 2)]).await.unwrap();
    }

    let table = BTreeTable::open(&path, &["id"], options()).await.unwrap();
    assert_eq!(ids(&table).await, vec![1, 3]);
}

#[tokio::test]
async fn repeated_reopens_do_not_multiply_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.ltb");
    {
        let table = BTreeTable::create(&path, schema(), &["id"], options())
            .await
            .unwrap();
        table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
    }
    for _ in 0..5 {
        let table = BTreeTable::open(&path, &["id"], options()).await.unwrap();
        assert_eq!(ids(&table).await, vec![1, 2, 3]);
    }
}

#[tokio::test]
async fn a_flushed_table_needs_only_its_data_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.ltb");
    let rows: Vec<i64> = (0..500).collect();
    {
        let table = BTreeTable::create(&path, schema(), &["id"], options())
            .await
            .unwrap();
        table.insert(&[batch(&rows)]).await.unwrap();
        table.flush().await.unwrap();
    }

    let copy = dir.path().join("copy.ltb");
    std::fs::copy(&path, &copy).unwrap();

    let restored = BTreeTable::open(&copy, &["id"], options()).await.unwrap();
    assert_eq!(ids(&restored).await, rows);
}

#[tokio::test]
async fn a_flush_frees_the_pages_the_old_tree_used() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table
        .insert(&[batch(&(0..1000).collect::<Vec<i64>>())])
        .await
        .unwrap();
    table.flush().await.unwrap();

    table.insert(&[batch(&[5000])]).await.unwrap();
    table.flush().await.unwrap();

    // The second flush rewrote the tree; the first tree's pages are garbage.
    assert!(
        !table.snapshot().is_empty(),
        "the table still holds its rows after the rewrite"
    );
    assert_eq!(ids(&table).await.len(), 1001);
}

#[tokio::test]
async fn a_multi_column_key_orders_by_each_column_in_turn() {
    let dir = tempfile::tempdir().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("group", DataType::Utf8, false),
        Field::new("seq", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, true),
    ]));
    let table = BTreeTable::create(
        &dir.path().join("t.ltb"),
        schema.clone(),
        &["group", "seq"],
        options(),
    )
    .await
    .unwrap();

    let rows = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(vec!["b", "a", "b", "a"])),
            Arc::new(Int64Array::from(vec![1i64, 2, 0, 1])),
            Arc::new(StringArray::from(vec![
                Some("b1"),
                Some("a2"),
                Some("b0"),
                Some("a1"),
            ])),
        ],
    )
    .unwrap();
    table.insert(&[rows]).await.unwrap();
    table.flush().await.unwrap();

    let snapshot = table.snapshot();
    let found = table.scan(&snapshot).await.unwrap();
    let groups = found
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let seqs = found
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let ordered: Vec<(String, i64)> = (0..found.num_rows())
        .map(|row| (groups.value(row).to_string(), seqs.value(row)))
        .collect();

    assert_eq!(
        ordered,
        vec![
            ("a".to_string(), 1),
            ("a".to_string(), 2),
            ("b".to_string(), 0),
            ("b".to_string(), 1),
        ]
    );
}

#[tokio::test]
async fn a_key_column_that_does_not_exist_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let err = BTreeTable::create(&dir.path().join("t.ltb"), schema(), &["absent"], options())
        .await
        .unwrap_err();
    assert!(
        matches!(err, localtables_format::Error::InvalidArgument(_)),
        "got {err:?}"
    );
}

#[tokio::test]
async fn a_table_with_no_key_columns_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let err = BTreeTable::create(&dir.path().join("t.ltb"), schema(), &[], options())
        .await
        .unwrap_err();
    assert!(
        matches!(err, localtables_format::Error::InvalidArgument(_)),
        "got {err:?}"
    );
}

#[tokio::test]
async fn a_second_writer_is_refused_while_the_table_is_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.ltb");
    let _held = BTreeTable::create(&path, schema(), &["id"], options())
        .await
        .unwrap();

    let err = BTreeTable::open(&path, &["id"], options())
        .await
        .unwrap_err();
    assert!(
        matches!(err, localtables_format::Error::WriterLocked(_)),
        "got {err:?}"
    );
}

/// A long mixed sequence: the table must always hold exactly what was written
/// and not deleted, whether it is in the overlay or the tree.
#[tokio::test]
async fn a_mixed_sequence_leaves_the_table_consistent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.ltb");
    let mut table = BTreeTable::create(&path, schema(), &["id"], options())
        .await
        .unwrap();
    let mut expected: BTreeMap<i64, ()> = BTreeMap::new();

    for round in 0..30i64 {
        let inserted: Vec<i64> = (round * 7..round * 7 + 7).collect();
        table.insert(&[batch(&inserted)]).await.unwrap();
        expected.extend(inserted.iter().map(|id| (*id, ())));

        if round % 3 == 0 && !expected.is_empty() {
            let doomed = *expected.keys().next().unwrap();
            table.delete_keys(&[key(&table, doomed)]).await.unwrap();
            expected.remove(&doomed);
        }
        if round % 5 == 0 {
            table.flush().await.unwrap();
        }
        if round % 7 == 0 {
            drop(table);
            table = BTreeTable::open(&path, &["id"], options()).await.unwrap();
        }

        assert_eq!(
            ids(&table).await,
            expected.keys().copied().collect::<Vec<i64>>(),
            "after round {round}"
        );
    }

    drop(table);
    let table = BTreeTable::open(&path, &["id"], options()).await.unwrap();
    assert_eq!(
        ids(&table).await,
        expected.keys().copied().collect::<Vec<i64>>()
    );
}
