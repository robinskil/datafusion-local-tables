//! Readers and a writer sharing one table.
//!
//! The engine promises that a reader holding a snapshot sees one commit and
//! nothing else, whatever the writer does meanwhile, and that the writer never
//! hands a reader's bytes to a later write. These tests run both at once on a
//! real multi-threaded runtime, because that is the only way those promises
//! can actually break.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use arrow_array::{Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

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
        .map(|i| (i % 4 != 0).then(|| format!("name-{i}")))
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

fn options(backend: IoBackend) -> TableOptions {
    TableOptions {
        durability: Durability::None,
        io_backend: backend,
        compression: Compression::None,
        scan_batch_rows: 512,
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

/// Every row must carry the name its id implies, whichever commit it came from.
fn check_rows_are_intact(batches: &[RecordBatch]) {
    for batch in batches {
        let ids = batch
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
            let id = ids.value(row);
            if id % 4 == 0 {
                assert!(names.is_null(row), "id {id} should have no name");
            } else {
                assert_eq!(
                    names.value(row),
                    format!("name-{id}"),
                    "id {id} carries the wrong name"
                );
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn readers_see_one_commit_while_the_writer_works() {
    for backend in [IoBackend::Mmap, IoBackend::Pread] {
        let dir = tempfile::tempdir().unwrap();
        let table = ColumnarTable::create(&dir.path().join("t.lt"), schema(), options(backend))
            .await
            .unwrap();

        table
            .insert(&[batch(&(0..1000).collect::<Vec<i32>>())])
            .await
            .unwrap();

        let stop = Arc::new(AtomicBool::new(false));
        let writer = {
            let table = table.clone();
            let stop = stop.clone();
            tokio::spawn(async move {
                for round in 1..=20i32 {
                    let rows: Vec<i32> = (round * 1000..round * 1000 + 500).collect();
                    table.insert(&[batch(&rows)]).await.unwrap();
                }
                stop.store(true, Ordering::Release);
            })
        };

        let reader = {
            let table = table.clone();
            let stop = stop.clone();
            tokio::spawn(async move {
                let mut reads = 0;
                while !stop.load(Ordering::Acquire) {
                    // Pin once, then read the whole snapshot. Its row count must
                    // not move under us, whatever the writer commits meanwhile.
                    let snapshot = table.snapshot();
                    let expected = snapshot.live_rows();
                    let batches = table.scan(&snapshot, None).await.unwrap();
                    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
                    assert_eq!(
                        rows as u64, expected,
                        "a pinned snapshot changed size mid-scan on {backend:?}"
                    );
                    check_rows_are_intact(&batches);
                    reads += 1;
                    tokio::task::yield_now().await;
                }
                reads
            })
        };

        writer.await.unwrap();
        let reads = reader.await.unwrap();
        assert!(reads > 0, "the reader never got a scan in on {backend:?}");
        assert_eq!(table.row_count(), 1000 + 20 * 500);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_long_scan_is_unaffected_by_deletes_and_drops_behind_it() {
    let dir = tempfile::tempdir().unwrap();
    let table = ColumnarTable::create(&dir.path().join("t.lt"), schema(), options(IoBackend::Mmap))
        .await
        .unwrap();

    for round in 0..8i32 {
        let rows: Vec<i32> = (round * 500..round * 500 + 500).collect();
        table.insert(&[batch(&rows)]).await.unwrap();
        // One segment per round, so the writer below has segments to drop.
        // An insert on its own only reaches the log and the memtable.
        table.flush().await.unwrap();
    }

    // Pin, then let the writer delete everything the reader is holding.
    let pinned = table.snapshot();
    let expected: Vec<i32> = (0..4000).collect();

    let segments: Vec<u64> = pinned
        .manifest
        .segments
        .iter()
        .map(|s| s.segment_id)
        .collect();
    let writer = {
        let table = table.clone();
        tokio::spawn(async move {
            for id in segments {
                table.delete_segments(&[id]).await.unwrap();
            }
            // Write new data, which the allocator may try to place in the
            // bytes the pinned reader is still mapping.
            for round in 100..110i32 {
                let rows: Vec<i32> = (round * 500..round * 500 + 500).collect();
                table.insert(&[batch(&rows)]).await.unwrap();
                table.flush().await.unwrap();
            }
        })
    };

    writer.await.unwrap();

    let batches = table.scan(&pinned, None).await.unwrap();
    assert_eq!(
        ids(&batches),
        expected,
        "the pinned snapshot must still read its own rows after everything it \
         pointed at was deleted and new data was written"
    );
    check_rows_are_intact(&batches);

    drop(pinned);
    assert_eq!(
        table.row_count(),
        5000,
        "the live table holds only the new rows"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn arrow_buffers_outlive_the_snapshot_that_produced_them() {
    let dir = tempfile::tempdir().unwrap();
    let table = ColumnarTable::create(&dir.path().join("t.lt"), schema(), options(IoBackend::Mmap))
        .await
        .unwrap();
    table
        .insert(&[batch(&(0..2000).collect::<Vec<i32>>())])
        .await
        .unwrap();
    // Into a segment, so the scan below maps the file. Rows still in the
    // memtable are held in memory and would not test anything about mappings.
    table.flush().await.unwrap();

    let batches = {
        let snapshot = table.snapshot();
        table.scan(&snapshot, None).await.unwrap()
    };

    // The snapshot is gone; the mapping is not, because the batches hold it.
    let segment = table.snapshot().manifest.segments[0].segment_id;
    table.delete_segments(&[segment]).await.unwrap();
    for round in 0..5i32 {
        table
            .insert(&[batch(
                &(round * 2000..round * 2000 + 2000).collect::<Vec<i32>>(),
            )])
            .await
            .unwrap();
        table.flush().await.unwrap();
    }

    assert_eq!(ids(&batches), (0..2000).collect::<Vec<i32>>());
    check_rows_are_intact(&batches);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_readers_share_one_table() {
    let dir = tempfile::tempdir().unwrap();
    let table = ColumnarTable::create(&dir.path().join("t.lt"), schema(), options(IoBackend::Mmap))
        .await
        .unwrap();
    table
        .insert(&[batch(&(0..5000).collect::<Vec<i32>>())])
        .await
        .unwrap();

    let readers: Vec<_> = (0..8)
        .map(|_| {
            let table = table.clone();
            tokio::spawn(async move {
                for _ in 0..10 {
                    let snapshot = table.snapshot();
                    let batches = table.scan(&snapshot, Some(&[0])).await.unwrap();
                    assert_eq!(ids(&batches).len(), 5000);
                }
            })
        })
        .collect();

    for reader in readers {
        reader.await.unwrap();
    }

    // Once every reader is done, nothing but the current snapshot is pinned.
    assert_eq!(table.active_snapshots(), 1);
}
