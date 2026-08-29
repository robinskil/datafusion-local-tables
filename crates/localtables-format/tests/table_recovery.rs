//! A table that survives a crash must contain either the write or not, never
//! half of it, and must still be writable afterwards.
//!
//! The commit protocol is tested at the file layer already. These tests check
//! the layer above: that a torn insert or delete leaves the *table* consistent,
//! with segments, delete vectors and row counts all telling the same story.

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use localtables_format::columnar::table::ColumnarTable;
use localtables_format::config::{Durability, IoBackend, TableOptions};
use localtables_format::io::fault::FaultIo;
use localtables_format::io::{open_backend, FileIo};
use localtables_format::layout::TableKind;
use localtables_format::table_file::TableFile;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

fn batch(ids: &[i32]) -> RecordBatch {
    let names: Vec<Option<String>> = ids.iter().map(|i| Some(format!("n{i}"))).collect();
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
        ..TableOptions::default()
    }
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

/// Open the table file through a backend that stops writing after `budget`
/// bytes, so the next commit tears at a chosen point.
async fn faulty_file(path: &std::path::Path, budget: u64) -> TableFile {
    let io = open_backend(path, IoBackend::Mmap, Durability::None, false).unwrap();
    let mut file = TableFile::open(path, TableKind::Columnar, options())
        .await
        .unwrap();
    file.set_io(Arc::new(FaultIo::with_budget(io, budget)) as Arc<dyn FileIo>);
    file
}

/// Every row a scan returns must also be counted by the manifest, and the two
/// must agree with what the segments actually hold.
async fn check_consistent(table: &ColumnarTable) -> Vec<i32> {
    let snapshot = table.snapshot();
    let batches = table.scan(&snapshot, None).await.unwrap();
    let scanned: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
    assert_eq!(
        scanned,
        snapshot.live_rows(),
        "the manifest and the segments disagree on how many rows exist"
    );
    ids(&batches)
}

/// Crash an insert at every byte boundary and check what the table recovers to.
#[tokio::test]
async fn an_insert_is_all_or_nothing() {
    const MAX_BUDGET: u64 = 9000;

    let mut saw_before = false;
    let mut saw_after = false;

    for budget in (0..MAX_BUDGET).step_by(53) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.lt");

        {
            let table = ColumnarTable::create(&path, schema(), options())
                .await
                .unwrap();
            table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
        }

        // Crash partway through a second insert.
        {
            let mut file = faulty_file(&path, budget).await;
            let mut manifest = file.manifest().clone();
            let segment_id = manifest.next_segment_id;
            let built = localtables_format::columnar::segment::build_segment(
                segment_id,
                &schema(),
                localtables_format::layout::schema::fingerprint(&schema()),
                &[batch(&[4, 5, 6])],
                &options(),
            )
            .unwrap();

            let write = async {
                let data = file
                    .write_allocated(
                        &mut manifest,
                        &built.bytes,
                        localtables_format::layout::SEGMENT_ALIGN,
                        u64::MAX,
                    )
                    .await?;
                let (_, meta) = built.placed(data.offset);
                manifest.next_segment_id += 1;
                manifest
                    .segments
                    .push(localtables_format::layout::manifest::SegmentEntry {
                        segment_id,
                        data,
                        meta,
                        row_count: built.row_count,
                        deleted_count: 0,
                        deletes: None,
                    });
                file.commit(manifest, u64::MAX).await.map(|_| ())
            };
            let _ = write.await;
        }

        let table = ColumnarTable::open(&path, options())
            .await
            .unwrap_or_else(|e| panic!("budget {budget}: the table did not reopen: {e}"));
        let rows = check_consistent(&table).await;

        match rows.len() {
            3 => {
                saw_before = true;
                assert_eq!(rows, vec![1, 2, 3], "budget {budget}");
            }
            6 => {
                saw_after = true;
                assert_eq!(rows, vec![1, 2, 3, 4, 5, 6], "budget {budget}");
            }
            other => panic!("budget {budget}: recovered {other} rows, expected 3 or 6"),
        }

        // The recovered table must still take writes.
        table.insert(&[batch(&[9])]).await.unwrap();
        let after = check_consistent(&table).await;
        assert_eq!(*after.last().unwrap(), 9, "budget {budget}");
    }

    assert!(saw_before, "no crash landed before the insert took effect");
    assert!(saw_after, "no crash landed after the insert took effect");
}

/// A delete that never committed must leave every row readable.
#[tokio::test]
async fn a_torn_delete_leaves_the_rows_in_place() {
    const MAX_BUDGET: u64 = 4000;

    let mut saw_before = false;
    let mut saw_after = false;

    for budget in (0..MAX_BUDGET).step_by(37) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.lt");
        {
            let table = ColumnarTable::create(&path, schema(), options())
                .await
                .unwrap();
            table.insert(&[batch(&[1, 2, 3, 4, 5])]).await.unwrap();
        }

        {
            let mut file = faulty_file(&path, budget).await;
            let mut manifest = file.manifest().clone();
            manifest.txn_id = file.meta().txn_id + 1;
            let dv = localtables_format::columnar::DeleteVector::from_iter([1u32, 3]);

            let attempt = async {
                let extent = file
                    .write_allocated(
                        &mut manifest,
                        &dv.to_frame()?,
                        localtables_format::layout::BUFFER_ALIGN,
                        u64::MAX,
                    )
                    .await?;
                let entry = manifest.segments.first_mut().unwrap();
                entry.deletes = Some(extent);
                entry.deleted_count = dv.len();
                file.commit(manifest, u64::MAX).await.map(|_| ())
            };
            let _ = attempt.await;
        }

        let table = ColumnarTable::open(&path, options())
            .await
            .unwrap_or_else(|e| panic!("budget {budget}: the table did not reopen: {e}"));
        let rows = check_consistent(&table).await;

        match rows.len() {
            5 => {
                saw_before = true;
                assert_eq!(rows, vec![1, 2, 3, 4, 5], "budget {budget}");
            }
            3 => {
                saw_after = true;
                assert_eq!(rows, vec![1, 3, 5], "budget {budget}");
            }
            other => panic!("budget {budget}: recovered {other} rows, expected 5 or 3"),
        }
    }

    assert!(saw_before, "no crash landed before the delete took effect");
    assert!(saw_after, "no crash landed after the delete took effect");
}

/// Bytes written past the last commit are garbage, and reopening must reclaim
/// the space rather than trip over it.
#[tokio::test]
async fn a_reopened_table_keeps_committing_after_repeated_crashes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    {
        ColumnarTable::create(&path, schema(), options())
            .await
            .unwrap();
    }

    let mut expected: Vec<i32> = Vec::new();
    for round in 0..15i32 {
        // Alternate a crashed write with a clean one.
        {
            let mut file = faulty_file(&path, 200 + round as u64 * 71).await;
            let manifest = file.manifest().clone();
            let _ = file.commit(manifest, u64::MAX).await;
        }

        let table = ColumnarTable::open(&path, options())
            .await
            .unwrap_or_else(|e| panic!("round {round}: the table did not reopen: {e}"));
        table.insert(&[batch(&[round])]).await.unwrap();
        expected.push(round);

        assert_eq!(check_consistent(&table).await, expected, "round {round}");
    }
}
