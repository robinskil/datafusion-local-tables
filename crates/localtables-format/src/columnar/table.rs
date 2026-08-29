//! The columnar table.
//!
//! One writer, many readers, one file. Writers take a mutex and commit; readers
//! load the current snapshot with no lock at all and hold it for as long as
//! they need it. A snapshot pins the bytes it reads, so the allocator will not
//! hand those bytes to a later write while a query is still looking at them.

use std::path::Path;
use std::sync::Arc;

use arc_swap::ArcSwap;
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use tokio::sync::Mutex;

use crate::columnar::delete_vector::DeleteVector;
use crate::columnar::segment::{build_segment, SegmentReader};
use crate::config::TableOptions;
use crate::io::FileIo;
use crate::layout::manifest::{Manifest, SegmentEntry, SegmentId};
use crate::layout::{schema as schema_codec, Extent, TableKind, BUFFER_ALIGN, SEGMENT_ALIGN};
use crate::snapshot::{Snapshot, SnapshotRegistry};
use crate::table_file::TableFile;
use crate::{Error, Result};

/// Rewrite a segment once this share of its rows is deleted.
pub const DEFAULT_COMPACTION_DEAD_RATIO: f64 = 0.5;

struct Inner {
    /// Serialises writers. One process holds the file lock, and inside that
    /// process this mutex makes commits one at a time.
    file: Mutex<TableFile>,
    /// Read path. Cloned out of the file so readers never wait on the writer.
    io: Arc<dyn FileIo>,
    /// The newest committed snapshot. Readers load it without locking.
    current: ArcSwap<Snapshot>,
    registry: SnapshotRegistry,
    schema: SchemaRef,
    schema_fingerprint: u64,
    options: TableOptions,
}

/// A table stored in one local file, read column at a time.
#[derive(Clone)]
pub struct ColumnarTable {
    inner: Arc<Inner>,
}

impl ColumnarTable {
    /// Create a table. Fails when the path already exists.
    pub async fn create(path: &Path, schema: SchemaRef, options: TableOptions) -> Result<Self> {
        let file = TableFile::create(path, TableKind::Columnar, schema, options.clone()).await?;
        Self::from_file(file).await
    }

    /// Open an existing table.
    pub async fn open(path: &Path, options: TableOptions) -> Result<Self> {
        let file = TableFile::open(path, TableKind::Columnar, options).await?;
        Self::from_file(file).await
    }

    /// Open the table, creating it when the file is absent.
    pub async fn open_or_create(
        path: &Path,
        schema: SchemaRef,
        options: TableOptions,
    ) -> Result<Self> {
        let file = TableFile::open_or_create(path, TableKind::Columnar, schema, options).await?;
        Self::from_file(file).await
    }

    async fn from_file(file: TableFile) -> Result<Self> {
        let schema = file.schema().clone();
        let schema_fingerprint = schema_codec::fingerprint(&schema);
        let io = file.io().clone();
        let options = file.options().clone();

        let snapshot = build_snapshot(&file, io.as_ref(), &schema).await?;
        let registry = SnapshotRegistry::new();

        Ok(Self {
            inner: Arc::new(Inner {
                current: ArcSwap::from(registry.publish(snapshot)),
                file: Mutex::new(file),
                io,
                registry,
                schema,
                schema_fingerprint,
                options,
            }),
        })
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.inner.schema
    }

    pub fn options(&self) -> &TableOptions {
        &self.inner.options
    }

    /// Pin the table as it stands now.
    ///
    /// Hold the returned snapshot for the length of a query. Everything it
    /// points at stays readable until it is dropped, even while writers commit.
    /// Pinning is just an `Arc` clone: the registry already follows every
    /// published snapshot, so reading costs no lock and leaves nothing behind.
    pub fn snapshot(&self) -> Arc<Snapshot> {
        self.inner.current.load_full()
    }

    /// Rows a scan would return right now.
    pub fn row_count(&self) -> u64 {
        self.inner.current.load().live_rows()
    }

    /// Append rows as a new segment.
    ///
    /// Phase 3 writes one segment per call and commits immediately. The
    /// write-ahead log and memtable that make small inserts cheap arrive next.
    pub async fn insert(&self, batches: &[RecordBatch]) -> Result<u64> {
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        if rows == 0 {
            return Ok(0);
        }

        let mut file = self.inner.file.lock().await;
        let mut manifest = file.manifest().clone();
        let segment_id = manifest.next_segment_id;

        let built = build_segment(
            segment_id,
            &self.inner.schema,
            self.inner.schema_fingerprint,
            batches,
            &self.inner.options,
        )?;

        let min_active = self.inner.registry.min_active_txn();
        let data = file
            .write_allocated(&mut manifest, &built.bytes, SEGMENT_ALIGN, min_active)
            .await?;
        let (_, meta) = built.placed(data.offset);

        manifest.next_segment_id += 1;
        manifest.segments.push(SegmentEntry {
            segment_id,
            data,
            meta,
            row_count: built.row_count,
            deleted_count: 0,
            deletes: None,
        });

        self.commit(&mut file, manifest).await?;
        Ok(built.row_count)
    }

    /// Mark rows deleted by their position inside a segment.
    ///
    /// Returns how many rows this call newly deleted. Positions already
    /// deleted, or past the end of the segment, are ignored.
    pub async fn delete_positions(&self, deletions: &[(SegmentId, Vec<u32>)]) -> Result<u64> {
        if deletions.is_empty() {
            return Ok(0);
        }

        let mut file = self.inner.file.lock().await;
        let mut manifest = file.manifest().clone();
        let min_active = self.inner.registry.min_active_txn();
        let mut deleted = 0u64;
        let mut freed: Vec<Extent> = Vec::new();

        for (segment_id, positions) in deletions {
            let Some(entry) = manifest.segment(*segment_id).cloned() else {
                return Err(Error::InvalidArgument(format!(
                    "segment {segment_id} is not in this table"
                )));
            };

            let mut dv = self.load_deletes(&entry).await?;
            let before = dv.len();
            dv.delete_all(
                positions
                    .iter()
                    .copied()
                    .filter(|p| (*p as u64) < entry.row_count),
            );
            if dv.len() == before {
                continue;
            }
            deleted += dv.len() - before;

            // The old bitmap is garbage once the new one is committed. It is
            // recorded as freed, not overwritten, because a reader pinned to an
            // older snapshot may still be reading it.
            if let Some(old) = entry.deletes {
                freed.push(old);
            }
            let extent = file
                .write_allocated(&mut manifest, &dv.to_frame()?, BUFFER_ALIGN, min_active)
                .await?;

            let entry = manifest.segment_mut(*segment_id).expect("checked above");
            entry.deletes = Some(extent);
            entry.deleted_count = dv.len();
        }

        if deleted == 0 {
            return Ok(0);
        }
        // Freeing happens after the txn is stamped, so the quarantine records
        // the commit that made these bytes garbage.
        manifest.txn_id = file.meta().txn_id + 1;
        for extent in freed {
            manifest.free(extent);
        }

        self.commit(&mut file, manifest).await?;
        Ok(deleted)
    }

    /// Delete every row of the named segments.
    pub async fn delete_segments(&self, segment_ids: &[SegmentId]) -> Result<u64> {
        let mut file = self.inner.file.lock().await;
        let mut manifest = file.manifest().clone();
        manifest.txn_id = file.meta().txn_id + 1;

        let mut deleted = 0u64;
        let mut kept = Vec::with_capacity(manifest.segments.len());
        for entry in std::mem::take(&mut manifest.segments) {
            if segment_ids.contains(&entry.segment_id) {
                deleted += entry.live_rows();
                // Nothing points at the segment or its bitmap any more.
                manifest.free(entry.data);
                if let Some(dv) = entry.deletes {
                    manifest.free(dv);
                }
            } else {
                kept.push(entry);
            }
        }
        manifest.segments = kept;

        if deleted == 0 {
            return Ok(0);
        }
        self.commit(&mut file, manifest).await?;
        Ok(deleted)
    }

    /// Publish a manifest and swap in the snapshot readers will see next.
    async fn commit(&self, file: &mut TableFile, manifest: Manifest) -> Result<()> {
        let min_active = self.inner.registry.min_active_txn();
        file.commit(manifest, min_active).await?;

        let snapshot = build_snapshot(file, self.inner.io.as_ref(), &self.inner.schema).await?;
        self.inner
            .current
            .store(self.inner.registry.publish(snapshot));
        Ok(())
    }

    /// Read a segment's deletions, or an empty vector when it has none.
    async fn load_deletes(&self, entry: &SegmentEntry) -> Result<DeleteVector> {
        load_deletes(self.inner.io.as_ref(), entry).await
    }

    /// Open a segment for reading, keeping its bytes alive through the reader.
    pub async fn segment_reader(&self, entry: &SegmentEntry) -> Result<SegmentReader> {
        let bytes = self.inner.io.read_immutable(entry.data).await?;
        SegmentReader::new(
            bytes,
            entry.data.offset,
            entry.meta,
            self.inner.schema.clone(),
            self.inner.schema_fingerprint,
        )
    }

    /// Read one segment as batches, with deleted rows removed.
    ///
    /// Batches are capped at the configured scan size, so a large segment does
    /// not arrive as one enormous batch.
    pub async fn read_segment(
        &self,
        snapshot: &Snapshot,
        entry: &SegmentEntry,
        projection: Option<&[usize]>,
    ) -> Result<Vec<RecordBatch>> {
        let reader = self.segment_reader(entry).await?;
        let full = reader.read(projection)?;

        let filtered = match snapshot.deletes_for(entry.segment_id) {
            // No deletions is the common case, and it costs nothing.
            None => full,
            Some(dv) if dv.is_empty() => full,
            Some(dv) => {
                let mask = dv.keep_mask(full.num_rows());
                arrow_select::filter::filter_record_batch(&full, &mask)?
            }
        };

        Ok(slice_batches(filtered, self.inner.options.scan_batch_rows))
    }

    /// Read the whole table as batches.
    ///
    /// A convenience for tests and small tables. The streaming, partitioned
    /// scan lives in the DataFusion provider.
    pub async fn scan(
        &self,
        snapshot: &Snapshot,
        projection: Option<&[usize]>,
    ) -> Result<Vec<RecordBatch>> {
        let mut out = Vec::new();
        for entry in snapshot.live_segments() {
            out.extend(self.read_segment(snapshot, entry, projection).await?);
        }
        Ok(out)
    }

    /// Segments worth rewriting, because deletes now dominate them.
    pub fn compaction_candidates(&self, dead_ratio: f64) -> Vec<SegmentId> {
        self.inner
            .current
            .load()
            .manifest
            .segments
            .iter()
            .filter(|s| s.is_compaction_candidate(dead_ratio))
            .map(|s| s.segment_id)
            .collect()
    }

    /// How many snapshots are pinned. For tests and diagnostics.
    pub fn active_snapshots(&self) -> usize {
        self.inner.registry.active_count()
    }

    /// The oldest commit a reader is still pinned to.
    pub fn min_active_txn(&self) -> u64 {
        self.inner.registry.min_active_txn()
    }
}

impl std::fmt::Debug for ColumnarTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let snapshot = self.inner.current.load();
        f.debug_struct("ColumnarTable")
            .field("txn_id", &snapshot.txn_id)
            .field("segments", &snapshot.manifest.segments.len())
            .field("rows", &snapshot.live_rows())
            .finish()
    }
}

/// Read a segment's deletions off disk.
async fn load_deletes(io: &dyn FileIo, entry: &SegmentEntry) -> Result<DeleteVector> {
    match entry.deletes {
        None => Ok(DeleteVector::new()),
        Some(extent) => {
            let bytes = io.read_immutable(extent).await?;
            DeleteVector::from_frame(bytes.as_slice())
        }
    }
}

/// Build the snapshot for the file's current commit.
async fn build_snapshot(
    file: &TableFile,
    io: &dyn FileIo,
    schema: &SchemaRef,
) -> Result<Arc<Snapshot>> {
    let manifest = file.manifest().clone();
    let mut deletes = Vec::new();
    for entry in &manifest.segments {
        if entry.deletes.is_some() {
            deletes.push((entry.segment_id, Arc::new(load_deletes(io, entry).await?)));
        }
    }

    Ok(Arc::new(Snapshot {
        txn_id: file.meta().txn_id,
        schema: schema.clone(),
        manifest: Arc::new(manifest),
        deletes,
    }))
}

/// Cut a batch into pieces of at most `rows` rows.
fn slice_batches(batch: RecordBatch, rows: usize) -> Vec<RecordBatch> {
    if batch.num_rows() == 0 {
        return Vec::new();
    }
    if rows == 0 || batch.num_rows() <= rows {
        return vec![batch];
    }
    (0..batch.num_rows())
        .step_by(rows)
        .map(|start| batch.slice(start, rows.min(batch.num_rows() - start)))
        .collect()
}

#[cfg(test)]
mod tests {
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

    #[tokio::test]
    async fn a_new_table_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let table = table(&dir).await;

        assert_eq!(table.row_count(), 0);
        let snapshot = table.snapshot();
        assert!(table.scan(&snapshot, None).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn inserted_rows_come_back() {
        let dir = tempfile::tempdir().unwrap();
        let table = table(&dir).await;

        assert_eq!(table.insert(&[batch(&[1, 2, 3])]).await.unwrap(), 3);
        assert_eq!(table.row_count(), 3);

        let snapshot = table.snapshot();
        let read = table.scan(&snapshot, None).await.unwrap();
        assert_eq!(ids(&read), vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn each_insert_becomes_its_own_segment() {
        let dir = tempfile::tempdir().unwrap();
        let table = table(&dir).await;

        table.insert(&[batch(&[1, 2])]).await.unwrap();
        table.insert(&[batch(&[3, 4])]).await.unwrap();
        table.insert(&[batch(&[5])]).await.unwrap();

        let snapshot = table.snapshot();
        assert_eq!(snapshot.manifest.segments.len(), 3);
        assert_eq!(
            ids(&table.scan(&snapshot, None).await.unwrap()),
            vec![1, 2, 3, 4, 5]
        );
    }

    #[tokio::test]
    async fn inserting_nothing_does_not_commit() {
        let dir = tempfile::tempdir().unwrap();
        let table = table(&dir).await;
        let before = table.snapshot().txn_id;

        assert_eq!(table.insert(&[]).await.unwrap(), 0);
        assert_eq!(table.insert(&[batch(&[])]).await.unwrap(), 0);
        assert_eq!(
            table.snapshot().txn_id,
            before,
            "an empty insert is not a commit"
        );
    }

    #[tokio::test]
    async fn a_table_survives_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.lt");
        {
            let table = ColumnarTable::create(&path, schema(), options())
                .await
                .unwrap();
            table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
            table.insert(&[batch(&[4, 5])]).await.unwrap();
        }

        let table = ColumnarTable::open(&path, options()).await.unwrap();
        assert_eq!(table.row_count(), 5);
        let snapshot = table.snapshot();
        assert_eq!(
            ids(&table.scan(&snapshot, None).await.unwrap()),
            vec![1, 2, 3, 4, 5]
        );
    }

    #[tokio::test]
    async fn deleted_rows_disappear_from_scans() {
        let dir = tempfile::tempdir().unwrap();
        let table = table(&dir).await;
        table.insert(&[batch(&[1, 2, 3, 4, 5])]).await.unwrap();

        let segment = table.snapshot().manifest.segments[0].segment_id;
        assert_eq!(
            table
                .delete_positions(&[(segment, vec![1, 3])])
                .await
                .unwrap(),
            2
        );

        assert_eq!(table.row_count(), 3);
        let snapshot = table.snapshot();
        assert_eq!(
            ids(&table.scan(&snapshot, None).await.unwrap()),
            vec![1, 3, 5]
        );
    }

    #[tokio::test]
    async fn deletes_survive_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.lt");
        {
            let table = ColumnarTable::create(&path, schema(), options())
                .await
                .unwrap();
            table.insert(&[batch(&[1, 2, 3, 4])]).await.unwrap();
            let segment = table.snapshot().manifest.segments[0].segment_id;
            table
                .delete_positions(&[(segment, vec![0, 2])])
                .await
                .unwrap();
        }

        let table = ColumnarTable::open(&path, options()).await.unwrap();
        let snapshot = table.snapshot();
        assert_eq!(ids(&table.scan(&snapshot, None).await.unwrap()), vec![2, 4]);
        assert_eq!(table.row_count(), 2);
    }

    #[tokio::test]
    async fn deleting_the_same_row_twice_changes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let table = table(&dir).await;
        table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
        let segment = table.snapshot().manifest.segments[0].segment_id;

        assert_eq!(
            table.delete_positions(&[(segment, vec![1])]).await.unwrap(),
            1
        );
        let after_first = table.snapshot().txn_id;

        assert_eq!(
            table.delete_positions(&[(segment, vec![1])]).await.unwrap(),
            0
        );
        assert_eq!(
            table.snapshot().txn_id,
            after_first,
            "a delete that changes nothing must not commit"
        );
    }

    #[tokio::test]
    async fn positions_past_the_end_of_a_segment_are_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let table = table(&dir).await;
        table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
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
    async fn dropping_a_segment_reclaims_its_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let table = table(&dir).await;
        table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
        table.insert(&[batch(&[4, 5, 6])]).await.unwrap();

        let first = table.snapshot().manifest.segments[0].clone();
        assert_eq!(table.delete_segments(&[first.segment_id]).await.unwrap(), 3);

        let snapshot = table.snapshot();
        assert_eq!(snapshot.manifest.segments.len(), 1);
        assert_eq!(
            ids(&table.scan(&snapshot, None).await.unwrap()),
            vec![4, 5, 6]
        );
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

        // Pin before deleting.
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
        let now = table.snapshot();
        assert_eq!(ids(&table.scan(&now, None).await.unwrap()), vec![3, 4]);
    }

    #[tokio::test]
    async fn a_pinned_snapshot_blocks_reuse_of_the_bytes_it_reads() {
        let dir = tempfile::tempdir().unwrap();
        let table = table(&dir).await;
        table.insert(&[batch(&[1, 2, 3])]).await.unwrap();

        let pinned = table.snapshot();
        let dropped = pinned.manifest.segments[0].data;
        table
            .delete_segments(&[pinned.manifest.segments[0].segment_id])
            .await
            .unwrap();

        // While the reader holds its snapshot, the freed extent stays put.
        table.insert(&[batch(&[7, 8, 9])]).await.unwrap();
        let after = table.snapshot();
        let reused = after
            .manifest
            .segments
            .iter()
            .any(|s| s.data.overlaps(&dropped));
        assert!(
            !reused,
            "a new segment landed on bytes a pinned reader may still be mapping"
        );

        // The pinned snapshot still reads its own rows.
        assert_eq!(
            ids(&table.scan(&pinned, None).await.unwrap()),
            vec![1, 2, 3]
        );
    }

    #[tokio::test]
    async fn freed_bytes_are_reused_once_no_reader_holds_them() {
        let dir = tempfile::tempdir().unwrap();
        let table = table(&dir).await;

        // A large segment, so the freed extent is worth reusing.
        let wide: Vec<i32> = (0..5000).collect();
        table.insert(&[batch(&wide)]).await.unwrap();
        let dropped = table.snapshot().manifest.segments[0].data;
        let segment_id = table.snapshot().manifest.segments[0].segment_id;
        table.delete_segments(&[segment_id]).await.unwrap();

        // No reader is pinned to a commit older than the one that freed it.
        assert_eq!(
            table.active_snapshots(),
            1,
            "only the current snapshot is live"
        );
        table.insert(&[batch(&[1])]).await.unwrap();
        table.insert(&[batch(&wide)]).await.unwrap();

        let after = table.snapshot();
        assert!(
            after
                .manifest
                .segments
                .iter()
                .any(|s| s.data.overlaps(&dropped)),
            "with no reader pinned, the freed bytes should be handed out again"
        );
    }

    #[tokio::test]
    async fn projection_reads_only_what_was_asked_for() {
        let dir = tempfile::tempdir().unwrap();
        let table = table(&dir).await;
        table.insert(&[batch(&[1, 2, 3])]).await.unwrap();

        let snapshot = table.snapshot();
        let read = table.scan(&snapshot, Some(&[1])).await.unwrap();
        assert_eq!(read[0].num_columns(), 1);
        assert_eq!(read[0].schema().field(0).name(), "name");
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
        let rows: Vec<i32> = (0..10).collect();
        table.insert(&[batch(&rows)]).await.unwrap();
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
}
