//! The b-tree table.
//!
//! Built for point lookups and key ranges. Rows are stored whole, ordered by a
//! memcomparable key, so finding one row costs one page read per level rather
//! than a scan.
//!
//! Writes go to the same write-ahead log the columnar table uses, buffered in a
//! sorted overlay. A flush merges the overlay into the tree and writes a new
//! one; the old tree stays where it is until no reader is pinned to it. That is
//! what makes a copy-on-write tree cheap to make crash-safe: there is no torn
//! page to recover, only a root that points at one whole tree or the other.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arc_swap::ArcSwap;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;
use tokio::sync::Mutex;

use crate::btree::keycodec;
use crate::btree::rowcodec::{self, RowDecoder};
use crate::btree::tree::{self, Change, Overlay, TreeRoot};
use crate::config::TableOptions;
use crate::layout::manifest::TreeRootEntry;
use crate::layout::{schema as schema_codec, TableKind};
use crate::table_file::TableFile;
use crate::wal::{encode_record, Lsn, WalPair, WalPaths, WalRecord};
use crate::{Error, Result};

/// The table as it stood at one commit, plus the writes not yet merged in.
#[derive(Debug)]
pub struct BTreeSnapshot {
    pub txn_id: u64,
    pub schema: SchemaRef,
    /// Which columns make up the key, in order.
    pub key_columns: Vec<usize>,
    root: TreeRoot,
    /// Changes durable in the log but not yet in the tree. A read consults
    /// these first, because they are newer than anything the tree holds.
    overlay: Arc<Overlay>,
}

impl BTreeSnapshot {
    /// Rows a scan would return.
    ///
    /// The overlay's puts for keys already in the tree replace rather than add,
    /// so this is exact only once the overlay is empty; between flushes it is
    /// an upper bound, and `count_rows` walks the data when the exact number
    /// matters.
    pub fn approximate_rows(&self) -> u64 {
        self.root.rows + self.overlay.len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_empty() && self.overlay.is_empty()
    }
}

struct Writer {
    file: TableFile,
    wal: WalPair,
    /// Changes durable in the log, waiting for a flush to merge them.
    overlay: Overlay,
    /// Roughly how much memory the overlay holds.
    overlay_bytes: usize,
    next_lsn: Lsn,
}

struct Inner {
    writer: Mutex<Writer>,
    current: ArcSwap<BTreeSnapshot>,
    path: PathBuf,
    schema: SchemaRef,
    key_columns: Vec<usize>,
    options: TableOptions,
}

/// A table stored in one local file, ordered by key.
#[derive(Clone)]
pub struct BTreeTable {
    inner: Arc<Inner>,
}

impl BTreeTable {
    /// Create a table keyed on the named columns.
    ///
    /// The key columns must come from the schema and must be of types that
    /// have an order this format can encode.
    pub async fn create(
        path: &Path,
        schema: SchemaRef,
        key_columns: &[&str],
        options: TableOptions,
    ) -> Result<Self> {
        let keys = resolve_key_columns(&schema, key_columns)?;
        let file = TableFile::create(path, TableKind::BTree, schema, options).await?;
        Self::from_file(file, keys).await
    }

    /// Open an existing table.
    ///
    /// The key columns are given again rather than stored: they are part of how
    /// the caller uses the table, and a mismatch shows up as a lookup that
    /// finds nothing rather than as corruption, so it is worth checking.
    pub async fn open(path: &Path, key_columns: &[&str], options: TableOptions) -> Result<Self> {
        let file = TableFile::open(path, TableKind::BTree, options).await?;
        let keys = resolve_key_columns(file.schema(), key_columns)?;
        Self::from_file(file, keys).await
    }

    pub async fn open_or_create(
        path: &Path,
        schema: SchemaRef,
        key_columns: &[&str],
        options: TableOptions,
    ) -> Result<Self> {
        if path.exists() {
            Self::open(path, key_columns, options).await
        } else {
            Self::create(path, schema, key_columns, options).await
        }
    }

    async fn from_file(file: TableFile, key_columns: Vec<usize>) -> Result<Self> {
        let schema = file.schema().clone();
        if !rowcodec::is_encodable(&schema) {
            return Err(Error::Unsupported(
                "this schema holds a type a b-tree row cannot store".into(),
            ));
        }
        let options = file.options().clone();
        let path = file.path().to_path_buf();

        let wal = WalPair::open(
            &WalPaths::for_table(&path),
            file.table_uuid(),
            options.durability,
        )?;
        let checkpoint_lsn = file.meta().checkpoint_lsn;

        let mut writer = Writer {
            file,
            wal,
            overlay: Overlay::new(),
            overlay_bytes: 0,
            next_lsn: checkpoint_lsn + 1,
        };
        replay(&mut writer, checkpoint_lsn, &schema)?;

        let table = Self {
            inner: Arc::new(Inner {
                current: ArcSwap::from(build_snapshot(&writer, &schema, &key_columns)),
                writer: Mutex::new(writer),
                path,
                schema,
                key_columns,
                options,
            }),
        };

        // Replay may have restored more than the log should hold.
        if table.should_flush(&*table.inner.writer.lock().await) {
            table.flush().await?;
        }
        Ok(table)
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.inner.schema
    }

    pub fn key_columns(&self) -> &[usize] {
        &self.inner.key_columns
    }

    pub fn path(&self) -> &Path {
        &self.inner.path
    }

    pub fn snapshot(&self) -> Arc<BTreeSnapshot> {
        self.inner.current.load_full()
    }

    /// Encode the key of one row of a batch.
    pub fn key_of(&self, batch: &RecordBatch, row: usize) -> Result<Vec<u8>> {
        let columns: Vec<ArrayRef> = self
            .inner
            .key_columns
            .iter()
            .map(|index| batch.column(*index).clone())
            .collect();
        keycodec::encode_row(&columns, row)
    }

    /// Insert or replace rows.
    ///
    /// A row whose key is already present replaces it, which is what a keyed
    /// table means by insert.
    pub async fn insert(&self, batches: &[RecordBatch]) -> Result<u64> {
        let mut changes = Overlay::new();
        for batch in batches {
            if batch.schema().fields() != self.inner.schema.fields() {
                return Err(Error::SchemaMismatch(format!(
                    "a batch has schema {:?}, the table expects {:?}",
                    batch.schema(),
                    self.inner.schema
                )));
            }
            for row in 0..batch.num_rows() {
                changes.insert(
                    self.key_of(batch, row)?,
                    Change::Put(rowcodec::encode(batch, row)?),
                );
            }
        }
        self.apply(changes).await
    }

    /// Remove the rows with these keys. Keys that are not there are ignored.
    pub async fn delete_keys(&self, keys: &[Vec<u8>]) -> Result<u64> {
        let changes: Overlay = keys
            .iter()
            .map(|key| (key.clone(), Change::Delete))
            .collect();
        self.apply(changes).await
    }

    /// Log a set of changes, then make them visible.
    async fn apply(&self, changes: Overlay) -> Result<u64> {
        if changes.is_empty() {
            return Ok(0);
        }
        let count = changes.len() as u64;
        let mut writer = self.inner.writer.lock().await;

        let record = WalRecord::BTree {
            lsn: writer.take_lsn(),
            changes: changes
                .iter()
                .map(|(key, change)| crate::wal::KeyChange {
                    key: key.clone(),
                    // An absent row is a delete; the log needs no second field.
                    row: match change {
                        Change::Put(row) => Some(row.clone()),
                        Change::Delete => None,
                    },
                })
                .collect(),
        };
        writer.wal.append_group(&[encode_record(&record)?])?;

        // Only now are the changes durable, so only now do they become visible.
        for (key, change) in changes {
            writer.overlay_bytes += key.len()
                + match &change {
                    Change::Put(row) => row.len(),
                    Change::Delete => 0,
                };
            writer.overlay.insert(key, change);
        }

        self.publish(&writer);
        let needs_flush = self.should_flush(&writer);
        drop(writer);
        if needs_flush {
            self.flush().await?;
        }
        Ok(count)
    }

    /// Merge the pending changes into the tree and empty the log.
    pub async fn flush(&self) -> Result<u64> {
        let mut writer = self.inner.writer.lock().await;
        if writer.overlay.is_empty() {
            return Ok(0);
        }

        let checkpoint_lsn = writer.next_lsn.saturating_sub(1);
        let retired = writer.wal.rotate()?;

        let old_root = tree_root(writer.file.manifest().tree);
        let overlay = std::mem::take(&mut writer.overlay);
        writer.overlay_bytes = 0;

        let mut manifest = writer.file.manifest().clone();
        manifest.txn_id = writer.file.meta().txn_id + 1;
        manifest.checkpoint_lsn = checkpoint_lsn;

        // The old tree's pages become garbage as of this commit, not before it:
        // a reader pinned to the old root is still walking them.
        let stale = tree::all_pages(writer.file.io().as_ref(), old_root).await?;
        let new_root =
            tree::write_merged(&writer.file, &mut manifest, old_root, &overlay, u64::MAX).await?;

        manifest.tree = TreeRootEntry {
            root: new_root.root,
            rows: new_root.rows,
            height: new_root.height,
        };
        for extent in stale {
            manifest.free(extent);
        }

        writer.file.commit(manifest, u64::MAX).await?;
        self.publish(&writer);
        writer.wal.truncate(retired)?;
        Ok(new_root.rows)
    }

    fn should_flush(&self, writer: &Writer) -> bool {
        writer.overlay_bytes >= self.inner.options.memtable_max_bytes
            || writer.wal.active_len() >= self.inner.options.wal_max_bytes
    }

    fn publish(&self, writer: &Writer) {
        self.inner.current.store(build_snapshot(
            writer,
            &self.inner.schema,
            &self.inner.key_columns,
        ));
    }

    /// Read the row stored under `key`.
    ///
    /// The overlay is consulted first: it holds writes newer than the tree.
    pub async fn get(&self, snapshot: &BTreeSnapshot, key: &[u8]) -> Result<Option<RecordBatch>> {
        let row = match snapshot.overlay.get(key) {
            Some(Change::Put(row)) => Some(row.clone()),
            // A pending delete hides whatever the tree still holds.
            Some(Change::Delete) => None,
            None => {
                let io = self.inner.writer.lock().await.file.io().clone();
                tree::lookup(io.as_ref(), snapshot.root, key).await?
            }
        };

        let Some(row) = row else {
            return Ok(None);
        };
        let mut decoder = RowDecoder::new(self.inner.schema.clone())?;
        decoder.push(&row)?;
        Ok(Some(decoder.finish()?))
    }

    /// Read every row with a key in `start..end`, in key order.
    pub async fn range(
        &self,
        snapshot: &BTreeSnapshot,
        start: &[u8],
        end: Option<&[u8]>,
        limit: Option<usize>,
    ) -> Result<RecordBatch> {
        let io = self.inner.writer.lock().await.file.io().clone();
        // Read the tree without a limit, because the overlay can remove rows
        // from the result and add others ahead of them.
        let from_tree = tree::range(io.as_ref(), snapshot.root, start, end, None).await?;

        // The overlay is newer, so it decides every key it mentions.
        let merged = tree::merge(from_tree, &overlay_slice(&snapshot.overlay, start, end));

        let mut decoder = RowDecoder::new(self.inner.schema.clone())?;
        for (_, row) in merged.iter().take(limit.unwrap_or(usize::MAX)) {
            decoder.push(row)?;
        }
        decoder.finish()
    }

    /// Read the whole table in key order.
    pub async fn scan(&self, snapshot: &BTreeSnapshot) -> Result<RecordBatch> {
        self.range(snapshot, &[], None, None).await
    }

    /// Rows a scan would return, counted exactly.
    pub async fn count_rows(&self, snapshot: &BTreeSnapshot) -> Result<u64> {
        Ok(self.scan(snapshot).await?.num_rows() as u64)
    }

    /// Bytes of records the log currently holds. For tests and diagnostics.
    pub async fn wal_bytes(&self) -> u64 {
        self.inner.writer.lock().await.wal.active_len()
    }

    /// Changes waiting for a flush. For tests and diagnostics.
    pub async fn pending_changes(&self) -> usize {
        self.inner.writer.lock().await.overlay.len()
    }
}

impl Writer {
    fn take_lsn(&mut self) -> Lsn {
        let lsn = self.next_lsn;
        self.next_lsn += 1;
        lsn
    }
}

impl std::fmt::Debug for BTreeTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let snapshot = self.inner.current.load();
        f.debug_struct("BTreeTable")
            .field("txn_id", &snapshot.txn_id)
            .field("rows", &snapshot.root.rows)
            .field("pending", &snapshot.overlay.len())
            .finish()
    }
}

/// The overlay entries whose keys fall in a range.
fn overlay_slice(overlay: &Overlay, start: &[u8], end: Option<&[u8]>) -> Overlay {
    overlay
        .range(start.to_vec()..)
        .take_while(|(key, _)| end.is_none_or(|end| key.as_slice() < end))
        .map(|(key, change)| (key.clone(), change.clone()))
        .collect()
}

/// Turn the manifest's record of the root into the tree's own form.
fn tree_root(entry: TreeRootEntry) -> TreeRoot {
    TreeRoot {
        root: entry.root,
        rows: entry.rows,
        height: entry.height,
    }
}

fn build_snapshot(
    writer: &Writer,
    schema: &SchemaRef,
    key_columns: &[usize],
) -> Arc<BTreeSnapshot> {
    Arc::new(BTreeSnapshot {
        txn_id: writer.file.meta().txn_id,
        schema: schema.clone(),
        key_columns: key_columns.to_vec(),
        root: tree_root(writer.file.manifest().tree),
        overlay: Arc::new(writer.overlay.clone()),
    })
}

/// Put the log's changes back, skipping what a flush already merged.
fn replay(writer: &mut Writer, checkpoint_lsn: Lsn, _schema: &SchemaRef) -> Result<()> {
    let records = writer.wal.recover()?;
    let mut highest = checkpoint_lsn;

    for record in records {
        if record.lsn() <= checkpoint_lsn {
            continue;
        }
        highest = highest.max(record.lsn());

        let WalRecord::BTree { changes, .. } = record else {
            return Err(Error::corrupt(
                "a b-tree table's log holds a columnar record",
            ));
        };
        for change in changes {
            let entry = match change.row {
                Some(row) => Change::Put(row),
                None => Change::Delete,
            };
            writer.overlay_bytes += change.key.len();
            writer.overlay.insert(change.key, entry);
        }
    }

    writer.next_lsn = highest + 1;
    Ok(())
}

/// Turn key column names into their positions, checking each is usable.
fn resolve_key_columns(schema: &SchemaRef, names: &[&str]) -> Result<Vec<usize>> {
    if names.is_empty() {
        return Err(Error::InvalidArgument(
            "a b-tree table needs at least one key column".into(),
        ));
    }
    names
        .iter()
        .map(|name| {
            let index = schema.index_of(name).map_err(|_| {
                Error::InvalidArgument(format!("the schema has no column named {name}"))
            })?;
            let data_type = schema.field(index).data_type();
            if !keycodec::is_encodable(data_type) {
                return Err(Error::Unsupported(format!(
                    "{name} is a {data_type}, which has no key order this format can store"
                )));
            }
            Ok(index)
        })
        .collect()
}

/// Placate the unused-import lint: the schema codec is used by `TableFile`.
#[allow(dead_code)]
fn _uses(schema: &SchemaRef) -> u64 {
    schema_codec::fingerprint(schema)
}

/// A map from key to change, exposed so callers can build one directly.
pub type Changes = BTreeMap<Vec<u8>, Change>;
