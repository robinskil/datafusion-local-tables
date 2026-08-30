//! The columnar table.
//!
//! One writer, many readers, one file. A writer takes a mutex. A reader loads
//! the current snapshot with no lock, and holds it as long as it needs.
//!
//! A snapshot pins the bytes it reads. The allocator will not give those bytes
//! to a later write while a query still reads them.
//!
//! A write does not build a segment. It appends a record to the write-ahead
//! log, waits for one sync, and lands in the memtable. Scans see it there at
//! once.
//!
//! A later flush turns the collected rows into segments, commits them, and
//! empties the log. That is what stops a three-row insert from costing a whole
//! segment write.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arc_swap::ArcSwap;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
use tokio::sync::Mutex;

use crate::columnar::delete_vector::DeleteVector;
use crate::columnar::memtable::Memtable;
use crate::columnar::segment::{build_segment, SegmentReader};
use crate::columnar::zorder;
use crate::layout::schema::SchemaLayout;
use crate::config::TableOptions;
use crate::io::FileIo;
use crate::layout::manifest::{Manifest, SegmentEntry, SegmentId};
use crate::layout::{TableKind, BUFFER_ALIGN, SEGMENT_ALIGN};
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
    pub(super) fn take_lsn(&mut self) -> Lsn {
        let lsn = self.next_lsn;
        self.next_lsn += 1;
        lsn
    }

    pub(super) fn deletes_for(&self, segment_id: SegmentId) -> DeleteVector {
        self.deletes
            .get(&segment_id)
            .map(|dv| dv.as_ref().clone())
            .unwrap_or_default()
    }

    /// Deletions in the shape a snapshot holds them.
    pub(super) fn delete_list(&self) -> Vec<(SegmentId, Arc<DeleteVector>)> {
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
    /// The schema in force. Swapped as a whole when a schema change commits,
    /// so a reader never sees a schema and a layout that disagree.
    schema: ArcSwap<TableSchema>,
    options: TableOptions,
}

/// The schema in force, with everything derived from it.
///
/// These travel together because they must agree: a layout built from one
/// schema and cluster columns resolved against another would write segments
/// nothing could read back.
#[derive(Debug)]
pub(super) struct TableSchema {
    pub(super) schema: SchemaRef,
    /// What a segment's bytes must look like to be read as this schema, for
    /// every prefix of it. A segment written before a column was added holds
    /// one of those prefixes.
    pub(super) layout: SchemaLayout,
    /// Columns whose bits order the rows a flush writes, resolved to positions.
    pub(super) cluster_columns: Vec<usize>,
}

impl TableSchema {
    pub(super) fn new(schema: SchemaRef, cluster_by: &[String]) -> Result<Self> {
        Ok(Self {
            cluster_columns: zorder::resolve(&schema, cluster_by)?,
            layout: SchemaLayout::of(&schema),
            schema,
        })
    }
}

/// A table stored in one local file, read column at a time.
#[derive(Clone)]
pub struct ColumnarTable {
    inner: Arc<Inner>,
}

mod evolve;
mod maintain;
mod read;
mod write;

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

    pub(super) async fn from_file(file: TableFile) -> Result<Self> {
        let schema = file.schema().clone();
        let table_schema = Arc::new(TableSchema::new(schema.clone(), &file.options().cluster_by)?);

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
        write::replay(&mut writer, checkpoint_lsn)?;

        let registry = SnapshotRegistry::new();
        let snapshot = build_snapshot(&writer)?;
        let table = Self {
            inner: Arc::new(Inner {
                current: ArcSwap::from(registry.publish(snapshot)),
                writer: Mutex::new(writer),
                io,
                registry,
                path,
                schema: ArcSwap::from(table_schema),
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

    pub fn schema(&self) -> SchemaRef {
        self.inner.schema.load().schema.clone()
    }

    /// The schema in force with everything derived from it, taken as one.
    pub(super) fn table_schema(&self) -> Arc<TableSchema> {
        self.inner.schema.load_full()
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

#[cfg(test)]
mod tests;
