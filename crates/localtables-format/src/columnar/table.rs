//! The columnar table.
//!
//! One writer, many readers, one file. Writers take a mutex; readers load the
//! current snapshot with no lock at all and hold it for as long as they need
//! it. A snapshot pins the bytes it reads, so the allocator will not hand those
//! bytes to a later write while a query is still looking at them.
//!
//! A write does not build a segment. It appends a record to the write-ahead
//! log, waits for one sync, and lands in the memtable, where scans can already
//! see it. A flush later turns the accumulated rows into one segment, commits
//! it, and empties the log. That is what keeps a three-row insert from costing
//! a whole segment write.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arc_swap::ArcSwap;
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use tokio::sync::Mutex;

use crate::columnar::delete_vector::DeleteVector;
use crate::columnar::memtable::Memtable;
use crate::columnar::segment::{build_segment, SegmentReader};
use crate::config::TableOptions;
use crate::io::FileIo;
use crate::layout::manifest::{Manifest, SegmentEntry, SegmentId};
use crate::layout::{schema as schema_codec, TableKind, BUFFER_ALIGN, SEGMENT_ALIGN};
use crate::snapshot::{Snapshot, SnapshotRegistry};
use crate::table_file::TableFile;
use crate::wal::{encode_record, Lsn, SegmentDeletes, WalPair, WalPaths, WalRecord};
use crate::{Error, Result};

/// Rewrite a segment once this share of its rows is deleted.
pub const DEFAULT_COMPACTION_DEAD_RATIO: f64 = 0.5;

/// Everything only the writer touches.
struct Writer {
    file: TableFile,
    wal: WalPair,
    memtable: Memtable,
    /// Deleted rows per segment, and the authority on them: a delete is durable
    /// in the log long before a flush records it in a commit.
    deletes: HashMap<SegmentId, Arc<DeleteVector>>,
    /// Segments whose deletions are in the log but not yet in the file. The
    /// next flush writes these out.
    dirty_deletes: HashSet<SegmentId>,
    /// The sequence number the next record will be given.
    next_lsn: Lsn,
}

impl Writer {
    fn take_lsn(&mut self) -> Lsn {
        let lsn = self.next_lsn;
        self.next_lsn += 1;
        lsn
    }

    fn deletes_for(&self, segment_id: SegmentId) -> DeleteVector {
        self.deletes
            .get(&segment_id)
            .map(|dv| dv.as_ref().clone())
            .unwrap_or_default()
    }

    /// Deletions in the shape a snapshot holds them.
    fn delete_list(&self) -> Vec<(SegmentId, Arc<DeleteVector>)> {
        self.deletes
            .iter()
            .filter(|(_, dv)| !dv.is_empty())
            .map(|(id, dv)| (*id, dv.clone()))
            .collect()
    }
}

struct Inner {
    /// Serialises writers. One process holds the file lock, and inside that
    /// process this mutex makes writes one at a time.
    writer: Mutex<Writer>,
    /// Read path. Cloned out of the file so readers never wait on the writer.
    io: Arc<dyn FileIo>,
    /// The newest published snapshot. Readers load it without locking.
    current: ArcSwap<Snapshot>,
    registry: SnapshotRegistry,
    path: PathBuf,
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

    /// Open an existing table, replaying whatever the log still holds.
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
        let path = file.path().to_path_buf();

        // Deletions recorded in the last commit. Replay may add to these.
        let mut deletes = HashMap::new();
        for entry in &file.manifest().segments {
            if entry.deletes.is_some() {
                deletes.insert(
                    entry.segment_id,
                    Arc::new(load_deletes(io.as_ref(), entry).await?),
                );
            }
        }

        let wal = WalPair::open(
            &WalPaths::for_table(&path),
            file.table_uuid(),
            options.durability,
        )?;
        let checkpoint_lsn = file.meta().checkpoint_lsn;
        let memtable = Memtable::new(schema.clone(), file.manifest().next_seqno);

        let mut writer = Writer {
            file,
            wal,
            memtable,
            deletes,
            dirty_deletes: HashSet::new(),
            next_lsn: checkpoint_lsn + 1,
        };
        replay(&mut writer, checkpoint_lsn)?;

        let registry = SnapshotRegistry::new();
        let snapshot = build_snapshot(&writer)?;
        let table = Self {
            inner: Arc::new(Inner {
                current: ArcSwap::from(registry.publish(snapshot)),
                writer: Mutex::new(writer),
                io,
                registry,
                path,
                schema,
                schema_fingerprint,
                options,
            }),
        };

        // Replay may have restored more than the log should hold, if the crash
        // happened long after the last flush.
        if table.should_flush(&*table.inner.writer.lock().await) {
            table.flush().await?;
        }
        Ok(table)
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.inner.schema
    }

    pub fn options(&self) -> &TableOptions {
        &self.inner.options
    }

    pub fn path(&self) -> &Path {
        &self.inner.path
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

    /// Append rows.
    ///
    /// The rows are durable when this returns, and visible to the next scan,
    /// but they are in the log and the memtable rather than in a segment.
    pub async fn insert(&self, batches: &[RecordBatch]) -> Result<u64> {
        let rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
        if rows == 0 {
            return Ok(0);
        }
        for batch in batches {
            if batch.schema().fields() != self.inner.schema.fields() {
                return Err(Error::SchemaMismatch(format!(
                    "a batch has schema {:?}, the table expects {:?}",
                    batch.schema(),
                    self.inner.schema
                )));
            }
        }

        let mut writer = self.inner.writer.lock().await;

        // Build every record before writing any of them, so one sync covers
        // the whole insert.
        let mut frames = Vec::with_capacity(batches.len());
        let mut planned = Vec::with_capacity(batches.len());
        let mut base_seqno = writer.memtable.next_seqno();
        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }
            let record = WalRecord::Insert {
                lsn: writer.take_lsn(),
                base_seqno,
                batch: crate::layout::batchcodec::encode(batch),
            };
            frames.push(encode_record(&record)?);
            planned.push((base_seqno, batch.clone()));
            base_seqno += batch.num_rows() as u64;
        }

        writer.wal.append_group(&frames)?;
        // Only now are the rows durable, so only now do they become visible.
        for (seqno, batch) in planned {
            writer.memtable.insert_at(seqno, batch);
        }

        self.publish(&writer)?;
        let needs_flush = self.should_flush(&writer);
        drop(writer);

        if needs_flush {
            self.flush().await?;
        }
        Ok(rows)
    }

    /// Mark rows deleted by their position inside a segment.
    ///
    /// Returns how many rows this call newly deleted. Positions already
    /// deleted, or past the end of the segment, are ignored.
    pub async fn delete_positions(&self, deletions: &[(SegmentId, Vec<u32>)]) -> Result<u64> {
        self.delete(deletions, &[]).await
    }

    /// Mark memtable rows deleted, by the sequence numbers they were given.
    pub async fn delete_memtable_rows(&self, seqnos: &[u64]) -> Result<u64> {
        self.delete(&[], seqnos).await
    }

    /// Delete rows in segments and in the memtable, as one durable record.
    pub async fn delete(
        &self,
        segment_deletions: &[(SegmentId, Vec<u32>)],
        memtable_seqnos: &[u64],
    ) -> Result<u64> {
        if segment_deletions.is_empty() && memtable_seqnos.is_empty() {
            return Ok(0);
        }
        let mut writer = self.inner.writer.lock().await;

        let planned = plan_deletes(&writer, segment_deletions, memtable_seqnos)?;
        if planned.is_empty() {
            return Ok(0);
        }

        let record = WalRecord::Delete {
            lsn: writer.take_lsn(),
            segments: planned.logged_segments()?,
            memtable_rows: planned.memtable.clone(),
        };
        writer.wal.append_group(&[encode_record(&record)?])?;

        let deleted = apply_deletes(&mut writer, planned);
        self.publish(&writer)?;
        Ok(deleted)
    }

    /// Replace rows: delete the ones named, and append their replacements.
    ///
    /// Both halves go into one log record, so a crash can never leave the old
    /// rows gone and the new ones missing. Returns how many rows were replaced.
    pub async fn update(
        &self,
        segment_deletions: &[(SegmentId, Vec<u32>)],
        memtable_seqnos: &[u64],
        replacements: &[RecordBatch],
    ) -> Result<u64> {
        let replacement_rows: u64 = replacements.iter().map(|b| b.num_rows() as u64).sum();
        if segment_deletions.is_empty() && memtable_seqnos.is_empty() && replacement_rows == 0 {
            return Ok(0);
        }
        for batch in replacements {
            if batch.schema().fields() != self.inner.schema.fields() {
                return Err(Error::SchemaMismatch(format!(
                    "a replacement batch has schema {:?}, the table expects {:?}",
                    batch.schema(),
                    self.inner.schema
                )));
            }
        }

        let mut writer = self.inner.writer.lock().await;
        let planned = plan_deletes(&writer, segment_deletions, memtable_seqnos)?;
        if planned.is_empty() && replacement_rows == 0 {
            return Ok(0);
        }

        // The replacements are concatenated so the record carries one batch,
        // which keeps the delete and the insert inseparable.
        let merged = match replacements.len() {
            0 => None,
            1 => Some(replacements[0].clone()),
            _ => Some(arrow_select::concat::concat_batches(
                &self.inner.schema,
                replacements,
            )?),
        };

        let base_seqno = writer.memtable.next_seqno();
        let record = WalRecord::Update {
            lsn: writer.take_lsn(),
            segments: planned.logged_segments()?,
            memtable_rows: planned.memtable.clone(),
            base_seqno,
            batch: match &merged {
                Some(batch) => crate::layout::batchcodec::encode(batch),
                None => crate::layout::batchcodec::encode(&RecordBatch::new_empty(
                    self.inner.schema.clone(),
                )),
            },
        };
        writer.wal.append_group(&[encode_record(&record)?])?;

        // Delete first, then insert: the replacements are new rows and must not
        // be caught by the deletion that made room for them.
        let deleted = apply_deletes(&mut writer, planned);
        if let Some(batch) = merged {
            writer.memtable.insert_at(base_seqno, batch);
        }

        self.publish(&writer)?;
        let needs_flush = self.should_flush(&writer);
        drop(writer);
        if needs_flush {
            self.flush().await?;
        }
        Ok(deleted.max(replacement_rows))
    }

    /// Delete every row of the named segments.
    ///
    /// This drops the segments outright rather than filling a bitmap, so it
    /// commits straight away instead of going through the log.
    pub async fn delete_segments(&self, segment_ids: &[SegmentId]) -> Result<u64> {
        let mut writer = self.inner.writer.lock().await;
        let mut manifest = writer.file.manifest().clone();
        manifest.txn_id = writer.file.meta().txn_id + 1;

        let mut deleted = 0u64;
        let mut kept = Vec::with_capacity(manifest.segments.len());
        for entry in std::mem::take(&mut manifest.segments) {
            if segment_ids.contains(&entry.segment_id) {
                let already = writer
                    .deletes
                    .get(&entry.segment_id)
                    .map_or(0, |dv| dv.len().min(entry.row_count));
                deleted += entry.row_count - already;
                // Nothing points at the segment or its bitmap any more.
                manifest.free(entry.data);
                if let Some(dv) = entry.deletes {
                    manifest.free(dv);
                }
                writer.deletes.remove(&entry.segment_id);
                writer.dirty_deletes.remove(&entry.segment_id);
            } else {
                kept.push(entry);
            }
        }
        manifest.segments = kept;

        if deleted == 0 {
            return Ok(0);
        }
        self.commit(&mut writer, manifest).await?;
        Ok(deleted)
    }

    /// Write the memtable out as a segment and empty the log.
    ///
    /// After this returns the table file alone holds everything: copying it is
    /// a complete backup.
    pub async fn flush(&self) -> Result<u64> {
        let mut writer = self.inner.writer.lock().await;
        self.flush_locked(&mut writer).await
    }

    async fn flush_locked(&self, writer: &mut Writer) -> Result<u64> {
        let frozen = writer.memtable.freeze()?;
        let dirty: Vec<SegmentId> = writer.dirty_deletes.drain().collect();
        if frozen.is_empty() && dirty.is_empty() {
            return Ok(0);
        }

        // Everything logged so far is about to be in the file, so a later
        // recovery must not replay it. Records appended after this point go to
        // the log this rotation switches to.
        let checkpoint_lsn = writer.next_lsn.saturating_sub(1);
        let retired = writer.wal.rotate()?;

        let mut manifest = writer.file.manifest().clone();
        manifest.txn_id = writer.file.meta().txn_id + 1;
        manifest.checkpoint_lsn = checkpoint_lsn;
        manifest.next_seqno = frozen.next_seqno.max(manifest.next_seqno);
        let min_active = self.inner.registry.min_active_txn();

        if !frozen.is_empty() {
            let segment_id = manifest.next_segment_id;
            let built = build_segment(
                segment_id,
                &self.inner.schema,
                self.inner.schema_fingerprint,
                &frozen.batches,
                &self.inner.options,
            )?;
            let data = writer
                .file
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
        }

        for segment_id in dirty {
            let Some(dv) = writer.deletes.get(&segment_id).cloned() else {
                continue;
            };
            let Some(previous) = manifest.segment(segment_id).and_then(|e| e.deletes) else {
                // The segment may have been dropped since the delete was logged.
                if manifest.segment(segment_id).is_none() {
                    continue;
                }
                let extent = writer
                    .file
                    .write_allocated(&mut manifest, &dv.to_frame()?, BUFFER_ALIGN, min_active)
                    .await?;
                let entry = manifest.segment_mut(segment_id).expect("checked above");
                entry.deletes = Some(extent);
                entry.deleted_count = dv.len();
                continue;
            };

            let extent = writer
                .file
                .write_allocated(&mut manifest, &dv.to_frame()?, BUFFER_ALIGN, min_active)
                .await?;
            // The bitmap this replaces is garbage, but only as of this commit:
            // a reader on an older snapshot may still be reading it.
            manifest.free(previous);
            let entry = manifest.segment_mut(segment_id).expect("checked above");
            entry.deletes = Some(extent);
            entry.deleted_count = dv.len();
        }

        self.commit(writer, manifest).await?;
        // The retired log's records are now inside the file.
        writer.wal.truncate(retired)?;
        Ok(frozen.rows)
    }

    /// True when the memtable or the log has grown past its limit.
    fn should_flush(&self, writer: &Writer) -> bool {
        writer.memtable.bytes() >= self.inner.options.memtable_max_bytes
            || writer.wal.active_len() >= self.inner.options.wal_max_bytes
    }

    /// Publish a manifest and the snapshot readers will see next.
    async fn commit(&self, writer: &mut Writer, manifest: Manifest) -> Result<()> {
        let min_active = self.inner.registry.min_active_txn();
        writer.file.commit(manifest, min_active).await?;
        self.publish(writer)
    }

    /// Swap in a snapshot of the writer's current state.
    fn publish(&self, writer: &Writer) -> Result<()> {
        let snapshot = build_snapshot(writer)?;
        self.inner
            .current
            .store(self.inner.registry.publish(snapshot));
        Ok(())
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

    /// Read the whole table as batches, segments first and then the rows still
    /// held in memory.
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
        for batch in snapshot.memtable.iter() {
            let batch = match projection {
                Some(indices) => batch.project(indices)?,
                None => batch.clone(),
            };
            out.extend(slice_batches(batch, self.inner.options.scan_batch_rows));
        }
        Ok(out)
    }

    /// Rewrite the segments deletes have hollowed out.
    ///
    /// A deleted row still occupies its bytes until the segment is rewritten
    /// without it. Compaction reads the live rows of the chosen segments,
    /// writes them as one new segment, and frees the old ones. Readers pinned
    /// to an older snapshot keep reading the old bytes; the allocator will not
    /// hand them out until those readers are gone.
    ///
    /// Returns how many rows the new segment holds. Nothing to do returns zero
    /// without committing.
    pub async fn compact(&self, dead_ratio: f64) -> Result<u64> {
        let targets = self.compaction_candidates(dead_ratio);
        if targets.is_empty() {
            return Ok(0);
        }
        self.compact_segments(&targets).await
    }

    /// Rewrite the named segments into one, dropping their deleted rows.
    pub async fn compact_segments(&self, segment_ids: &[SegmentId]) -> Result<u64> {
        if segment_ids.is_empty() {
            return Ok(0);
        }

        // Read outside the writer lock: the segments are immutable, and a
        // rewrite of a large table should not block writes while it reads.
        let snapshot = self.snapshot();
        let mut live: Vec<RecordBatch> = Vec::new();
        let mut sources = Vec::with_capacity(segment_ids.len());
        for segment_id in segment_ids {
            let Some(entry) = snapshot.manifest.segment(*segment_id).cloned() else {
                return Err(Error::InvalidArgument(format!(
                    "segment {segment_id} is not in this table"
                )));
            };
            live.extend(self.read_segment(&snapshot, &entry, None).await?);
            sources.push(entry);
        }

        let mut writer = self.inner.writer.lock().await;

        // A delete may have landed while the rows above were being read. That
        // makes the rewrite stale, so it is abandoned rather than resurrecting
        // rows the delete removed.
        for entry in &sources {
            let current = writer
                .deletes
                .get(&entry.segment_id)
                .map_or(0, |dv| dv.len());
            let when_read = snapshot
                .deletes_for(entry.segment_id)
                .map_or(0, |dv| dv.len());
            if current != when_read {
                return Ok(0);
            }
        }

        let mut manifest = writer.file.manifest().clone();
        manifest.txn_id = writer.file.meta().txn_id + 1;
        let min_active = self.inner.registry.min_active_txn();
        let rows: u64 = live.iter().map(|b| b.num_rows() as u64).sum();

        if rows > 0 {
            let segment_id = manifest.next_segment_id;
            let built = build_segment(
                segment_id,
                &self.inner.schema,
                self.inner.schema_fingerprint,
                &live,
                &self.inner.options,
            )?;
            let data = writer
                .file
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
        }

        // The rewritten segments and their bitmaps are garbage as of this
        // commit, not before it.
        manifest.segments.retain(|entry| {
            if !segment_ids.contains(&entry.segment_id) {
                return true;
            }
            false
        });
        for entry in &sources {
            manifest.free(entry.data);
            if let Some(dv) = entry.deletes {
                manifest.free(dv);
            }
            writer.deletes.remove(&entry.segment_id);
            writer.dirty_deletes.remove(&entry.segment_id);
        }

        self.commit(&mut writer, manifest).await?;
        Ok(rows)
    }

    /// Segments worth rewriting, because deletes now dominate them.
    pub fn compaction_candidates(&self, dead_ratio: f64) -> Vec<SegmentId> {
        let snapshot = self.inner.current.load();
        snapshot
            .manifest
            .segments
            .iter()
            .filter(|entry| {
                let deleted = entry.row_count - snapshot.live_rows_in(entry);
                entry.row_count > 0 && (deleted as f64 / entry.row_count as f64) >= dead_ratio
            })
            .map(|entry| entry.segment_id)
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

    /// Bytes of records the log currently holds. For tests and diagnostics.
    pub async fn wal_bytes(&self) -> u64 {
        self.inner.writer.lock().await.wal.active_len()
    }

    /// Rows held in memory but not yet in a segment.
    pub async fn memtable_rows(&self) -> u64 {
        self.inner.writer.lock().await.memtable.live_rows()
    }

    /// Sequence numbers of the memtable rows a scan would return, in the order
    /// a scan returns them. A delete names rows by these.
    pub async fn memtable_seqnos(&self) -> Vec<u64> {
        self.inner.writer.lock().await.memtable.live_seqnos()
    }
}

impl std::fmt::Debug for ColumnarTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let snapshot = self.inner.current.load();
        f.debug_struct("ColumnarTable")
            .field("txn_id", &snapshot.txn_id)
            .field("segments", &snapshot.manifest.segments.len())
            .field("rows", &snapshot.live_rows())
            .field("in_memory", &snapshot.memtable_rows())
            .finish()
    }
}

/// Put the log's records back, in order, skipping what a flush already folded
/// into the file.
fn replay(writer: &mut Writer, checkpoint_lsn: Lsn) -> Result<()> {
    let records = writer.wal.recover()?;
    let mut highest = checkpoint_lsn;

    for record in records {
        if record.lsn() <= checkpoint_lsn {
            // Already inside a segment. Replaying it would duplicate rows.
            continue;
        }
        highest = highest.max(record.lsn());

        match record {
            WalRecord::Insert {
                base_seqno, batch, ..
            } => {
                let batch = decode_batch(&batch, writer.memtable.schema())?;
                writer.memtable.insert_at(base_seqno, batch);
            }
            WalRecord::Delete {
                segments,
                memtable_rows,
                ..
            } => {
                apply_logged_deletes(writer, segments)?;
                writer.memtable.delete(memtable_rows);
            }
            WalRecord::Update {
                segments,
                memtable_rows,
                base_seqno,
                batch,
                ..
            } => {
                apply_logged_deletes(writer, segments)?;
                writer.memtable.delete(memtable_rows);
                let batch = decode_batch(&batch, writer.memtable.schema())?;
                writer.memtable.insert_at(base_seqno, batch);
            }
        }
    }

    writer.next_lsn = highest + 1;
    Ok(())
}

/// Restore a logged batch to Arrow.
fn decode_batch(
    data: &crate::layout::batchcodec::BatchData,
    schema: &SchemaRef,
) -> Result<RecordBatch> {
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(data)?;
    let archived =
        rkyv::access::<crate::layout::batchcodec::ArchivedBatchData, rkyv::rancor::Error>(&bytes)?;
    crate::layout::batchcodec::decode(archived, schema, None)
}

/// Put logged segment deletions back into the writer's in-memory state.
fn apply_logged_deletes(writer: &mut Writer, segments: Vec<SegmentDeletes>) -> Result<()> {
    for logged in segments {
        // A segment dropped since the delete was logged has nothing to mark.
        if writer.file.manifest().segment(logged.segment_id).is_none() {
            continue;
        }
        let dv = DeleteVector::from_bitmap_bytes(&logged.bitmap)?;
        writer.deletes.insert(logged.segment_id, Arc::new(dv));
        writer.dirty_deletes.insert(logged.segment_id);
    }
    Ok(())
}

/// The deletions a statement will actually make.
///
/// Worked out before anything is logged, so a delete that removes nothing
/// writes nothing.
#[derive(Debug, Default)]
struct PlannedDeletes {
    /// The new bitmap for each segment that changes.
    segments: Vec<(SegmentId, DeleteVector)>,
    /// Memtable rows that are not already deleted.
    memtable: Vec<u64>,
    /// Rows this removes that were not already gone.
    count: u64,
}

impl PlannedDeletes {
    fn is_empty(&self) -> bool {
        self.segments.is_empty() && self.memtable.is_empty()
    }

    /// The segment bitmaps, in the shape a log record carries them.
    fn logged_segments(&self) -> Result<Vec<SegmentDeletes>> {
        self.segments
            .iter()
            .map(|(id, dv)| {
                Ok(SegmentDeletes {
                    segment_id: *id,
                    bitmap: dv.to_bitmap_bytes()?,
                })
            })
            .collect()
    }
}

/// Work out which rows a deletion would actually remove.
fn plan_deletes(
    writer: &Writer,
    segment_deletions: &[(SegmentId, Vec<u32>)],
    memtable_seqnos: &[u64],
) -> Result<PlannedDeletes> {
    let mut planned = PlannedDeletes::default();

    for (segment_id, positions) in segment_deletions {
        let Some(entry) = writer.file.manifest().segment(*segment_id).cloned() else {
            return Err(Error::InvalidArgument(format!(
                "segment {segment_id} is not in this table"
            )));
        };
        let mut dv = writer.deletes_for(*segment_id);
        let before = dv.len();
        dv.delete_all(
            positions
                .iter()
                .copied()
                .filter(|p| (*p as u64) < entry.row_count),
        );
        if dv.len() > before {
            planned.count += dv.len() - before;
            planned.segments.push((*segment_id, dv));
        }
    }

    planned.memtable = memtable_seqnos
        .iter()
        .copied()
        .filter(|seqno| !writer.memtable.is_deleted(*seqno))
        .collect();

    Ok(planned)
}

/// Apply planned deletions to the writer's in-memory state.
///
/// Called only after the record describing them is durable.
fn apply_deletes(writer: &mut Writer, planned: PlannedDeletes) -> u64 {
    let mut deleted = planned.count;
    for (segment_id, dv) in planned.segments {
        writer.deletes.insert(segment_id, Arc::new(dv));
        writer.dirty_deletes.insert(segment_id);
    }
    deleted += writer.memtable.delete(planned.memtable);
    deleted
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

/// Build the snapshot readers should see for the writer's current state.
fn build_snapshot(writer: &Writer) -> Result<Arc<Snapshot>> {
    Ok(Arc::new(Snapshot {
        txn_id: writer.file.meta().txn_id,
        schema: writer.file.schema().clone(),
        manifest: Arc::new(writer.file.manifest().clone()),
        deletes: writer.delete_list(),
        memtable: Arc::new(writer.memtable.batches(None)?),
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

    async fn read(table: &ColumnarTable) -> Vec<i32> {
        let snapshot = table.snapshot();
        ids(&table.scan(&snapshot, None).await.unwrap())
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
}
