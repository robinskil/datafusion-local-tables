//! What the table promises across a crash.
//!
//! A write returns only once its record is durable. So after a crash, every
//! write that returned must still be there, and no write that returned may
//! have turned into something else. These tests run random sequences of
//! inserts, deletes and flushes, cut the process off at a random point, and
//! check that what comes back is exactly the writes that had been
//! acknowledged.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_array::{Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use proptest::prelude::*;

use localtables_format::columnar::table::ColumnarTable;
use localtables_format::config::{Compression, Durability, IoBackend, TableOptions};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

fn batch(ids: &[i32]) -> RecordBatch {
    let names: Vec<Option<String>> = ids
        .iter()
        .map(|i| (i % 5 != 0).then(|| format!("name-{i}")))
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
        durability: Durability::None,
        io_backend: IoBackend::Mmap,
        compression: Compression::None,
        // Small enough that a run of inserts triggers real flushes.
        memtable_max_bytes: 16 * 1024,
        ..TableOptions::default()
    }
}

/// Every id the table currently holds, and a check that the row is intact.
async fn read_ids(table: &ColumnarTable) -> BTreeSet<i32> {
    let snapshot = table.snapshot();
    let batches = table.scan(&snapshot, None).await.unwrap();

    let scanned: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
    assert_eq!(
        scanned,
        snapshot.live_rows(),
        "the row count and the rows themselves disagree"
    );

    let mut ids = BTreeSet::new();
    for batch in &batches {
        let id_column = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let names = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for row in 0..batch.num_rows() {
            let id = id_column.value(row);
            if id % 5 == 0 {
                assert!(names.is_null(row), "row {id} lost its null");
            } else {
                assert_eq!(
                    names.value(row),
                    format!("name-{id}"),
                    "row {id} is damaged"
                );
            }
            assert!(ids.insert(id), "row {id} came back twice");
        }
    }
    ids
}

/// One step in a random write sequence.
#[derive(Debug, Clone)]
enum Op {
    /// Append this many rows.
    Insert(usize),
    /// Delete the row at this position among the live rows.
    DeleteOne(usize),
    Flush,
    /// Close the table and open it again, as a clean restart would.
    Reopen,
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        6 => (1usize..40).prop_map(Op::Insert),
        3 => (0usize..1000).prop_map(Op::DeleteOne),
        2 => Just(Op::Flush),
        2 => Just(Op::Reopen),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 40, ..ProptestConfig::default() })]

    /// Whatever order writes, flushes and restarts happen in, the table holds
    /// exactly the rows that were inserted and not deleted.
    #[test]
    fn a_table_holds_exactly_what_was_written(ops in prop::collection::vec(op_strategy(), 1..40)) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        runtime.block_on(async {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("t.lt");
            let mut table = ColumnarTable::create(&path, schema(), options()).await.unwrap();

            // What the sequence says the table should hold.
            let mut expected: BTreeSet<i32> = BTreeSet::new();
            let mut next_id = 0i32;

            for op in &ops {
                match op {
                    Op::Insert(count) => {
                        let ids: Vec<i32> = (next_id..next_id + *count as i32).collect();
                        next_id += *count as i32;
                        table.insert(&[batch(&ids)]).await.unwrap();
                        expected.extend(ids);
                    }
                    Op::DeleteOne(position) => {
                        if expected.is_empty() {
                            continue;
                        }
                        // Delete whichever row sits at this position, using the
                        // engine's own addressing.
                        let snapshot = table.snapshot();
                        let seqnos = table.memtable_seqnos().await;
                        let segment_rows: Vec<(u64, u32)> = snapshot
                            .live_segments()
                            .flat_map(|entry| {
                                let deletes = snapshot.deletes_for(entry.segment_id).cloned();
                                (0..entry.row_count as u32)
                                    .filter(move |p| {
                                        deletes.as_ref().is_none_or(|dv| !dv.is_deleted(*p))
                                    })
                                    .map(move |p| (entry.segment_id, p))
                            })
                            .collect();

                        let total = segment_rows.len() + seqnos.len();
                        if total == 0 {
                            continue;
                        }
                        let index = position % total;
                        let removed = if index < segment_rows.len() {
                            let (segment_id, row) = segment_rows[index];
                            let value = row_value_in_segment(&table, &snapshot, segment_id, row).await;
                            table
                                .delete_positions(&[(segment_id, vec![row])])
                                .await
                                .unwrap();
                            value
                        } else {
                            let seqno = seqnos[index - segment_rows.len()];
                            let value = memtable_row_value(&table, &snapshot, index - segment_rows.len()).await;
                            table.delete_memtable_rows(&[seqno]).await.unwrap();
                            value
                        };
                        expected.remove(&removed);
                    }
                    Op::Flush => {
                        table.flush().await.unwrap();
                    }
                    Op::Reopen => {
                        drop(table);
                        table = ColumnarTable::open(&path, options()).await.unwrap();
                    }
                }

                let actual = read_ids(&table).await;
                assert_eq!(actual, expected, "after {op:?}");
            }

            // A final restart must change nothing.
            drop(table);
            let table = ColumnarTable::open(&path, options()).await.unwrap();
            assert_eq!(read_ids(&table).await, expected, "after a final restart");
        });
    }
}

/// The id stored at one position of a segment.
async fn row_value_in_segment(
    table: &ColumnarTable,
    snapshot: &localtables_format::Snapshot,
    segment_id: u64,
    row: u32,
) -> i32 {
    let entry = snapshot
        .manifest
        .segments
        .iter()
        .find(|e| e.segment_id == segment_id)
        .expect("segment is in the snapshot");
    let reader = table.segment_reader(entry).await.unwrap();
    let batch = reader.read(Some(&[0])).unwrap();
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .value(row as usize)
}

/// The id of the nth live memtable row.
async fn memtable_row_value(
    _table: &ColumnarTable,
    snapshot: &localtables_format::Snapshot,
    index: usize,
) -> i32 {
    let mut seen = 0usize;
    for batch in snapshot.memtable.iter() {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        if index < seen + batch.num_rows() {
            return ids.value(index - seen);
        }
        seen += batch.num_rows();
    }
    panic!("memtable row {index} is out of range");
}

/// A restart with unflushed writes must not lose or duplicate them.
#[tokio::test]
async fn restarting_repeatedly_never_multiplies_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    let mut expected: BTreeSet<i32> = BTreeSet::new();

    {
        let table = ColumnarTable::create(&path, schema(), options())
            .await
            .unwrap();
        table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
        expected.extend([1, 2, 3]);
    }

    for round in 0..10i32 {
        let table = ColumnarTable::open(&path, options()).await.unwrap();
        assert_eq!(read_ids(&table).await, expected, "round {round}");

        table.insert(&[batch(&[100 + round])]).await.unwrap();
        expected.insert(100 + round);
        if round % 3 == 0 {
            table.flush().await.unwrap();
        }
        assert_eq!(
            read_ids(&table).await,
            expected,
            "round {round} after write"
        );
    }

    let table = ColumnarTable::open(&path, options()).await.unwrap();
    assert_eq!(read_ids(&table).await, expected);
}

/// Rows written but never flushed must come back, because the insert returned.
#[tokio::test]
async fn an_unflushed_write_survives_because_it_was_acknowledged() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    let rows: Vec<i32> = (0..500).collect();

    {
        let table = ColumnarTable::create(&path, schema(), options())
            .await
            .unwrap();
        for chunk in rows.chunks(7) {
            table.insert(&[batch(chunk)]).await.unwrap();
        }
        // No flush, no clean shutdown: just drop the handle.
    }

    let table = ColumnarTable::open(&path, options()).await.unwrap();
    assert_eq!(read_ids(&table).await, rows.iter().copied().collect());
}

/// A crash mid-record leaves the log with a partial tail.
///
/// Cutting the log at every byte stands in for the crash landing anywhere in a
/// record. What comes back must always be a prefix of the writes: some run of
/// the earliest inserts, never a gap, never a damaged row, never a row that was
/// never written.
#[tokio::test]
async fn a_torn_log_recovers_a_prefix_of_the_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    // A fresh table appends to its first log and only leaves it on a flush.
    // The limits below are large enough that no flush happens on its own, so
    // every record lands in this one file.
    let wal = dir.path().join("t.lt.wal0");
    let options = || TableOptions {
        memtable_max_bytes: 64 * 1024 * 1024,
        ..self::options()
    };

    // Ten inserts, none flushed, so the log holds all of them.
    {
        let table = ColumnarTable::create(&path, schema(), options())
            .await
            .unwrap();
        for i in 0..10i32 {
            table.insert(&[batch(&[i * 10, i * 10 + 1])]).await.unwrap();
        }
        assert!(table.wal_bytes().await > 0);
    }
    assert!(
        std::fs::metadata(&wal).unwrap().len() > 64,
        "the records should be in the first log, with nothing having rotated it"
    );

    let intact = std::fs::read(&wal).unwrap();
    let mut recovered_counts = BTreeSet::new();

    let cuts = (0..intact.len()).step_by(13).chain([intact.len()]);
    for cut in cuts {
        std::fs::write(&wal, &intact[..cut]).unwrap();
        // The other log stays as it was; only this one is damaged.

        let table = match ColumnarTable::open(&path, options()).await {
            Ok(table) => table,
            Err(e) => {
                // Cutting inside the file header is not a torn record: it
                // destroys the log's identity, and refusing is correct.
                assert!(
                    cut < 64,
                    "a cut at byte {cut} should have recovered, but failed: {e}"
                );
                continue;
            }
        };

        let ids = read_ids(&table).await;
        // The rows come in pairs, oldest first, so a prefix of the writes is a
        // prefix of the ids.
        let expected: Vec<i32> = (0..10i32)
            .flat_map(|i| [i * 10, i * 10 + 1])
            .take(ids.len())
            .collect();
        assert_eq!(
            ids,
            expected.into_iter().collect::<BTreeSet<_>>(),
            "a cut at byte {cut} recovered something that is not a prefix"
        );
        recovered_counts.insert(ids.len());

        // The recovered table must still take writes.
        table.insert(&[batch(&[999])]).await.unwrap();
        assert!(read_ids(&table).await.contains(&999), "cut at byte {cut}");
    }

    assert!(
        recovered_counts.len() > 2,
        "the cuts should have recovered several different prefixes, got {recovered_counts:?}"
    );
    assert!(
        recovered_counts.contains(&0),
        "no cut landed before the first record"
    );
    assert!(
        recovered_counts.contains(&20),
        "no cut left the whole log intact"
    );
}

/// A log left behind by a table that was flushed cleanly holds nothing, so the
/// data file alone is a complete copy.
#[tokio::test]
async fn a_flushed_table_needs_only_its_data_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    let rows: Vec<i32> = (0..200).collect();

    {
        let table = ColumnarTable::create(&path, schema(), options())
            .await
            .unwrap();
        table.insert(&[batch(&rows)]).await.unwrap();
        table.flush().await.unwrap();
        assert_eq!(table.wal_bytes().await, 0);
    }

    // Copy the data file on its own, as a backup would.
    let copy = dir.path().join("copy.lt");
    std::fs::copy(&path, &copy).unwrap();

    let restored = ColumnarTable::open(&copy, options()).await.unwrap();
    assert_eq!(read_ids(&restored).await, rows.iter().copied().collect());
}
