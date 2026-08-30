//! The file layer both table kinds share.
//!
//! Owns the header, the two meta page slots, the manifest, and the extent
//! allocator. A commit runs in three steps:
//!
//! 1. append the new bytes and the new manifest, then sync;
//! 2. overwrite the meta slot holding the *older* txn, then sync;
//! 3. hand the new manifest to the caller, which publishes it to readers.
//!
//! A crash before step 2 leaves bytes past the old manifest that nothing points
//! at. The next open reclaims them. A crash during step 2 tears at most the
//! older slot, and the newer slot still describes a complete table.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_schema::Schema;

use crate::config::TableOptions;
use crate::io::lock::FileLock;
use crate::io::{open_backend, FileIo, SharedBuf};
use crate::layout::frame::{self, tag};
use crate::layout::header::{ArchivedFileHeader, ArchivedMetaPage, FileHeader, MetaPage, MetaSlot};
use crate::layout::manifest::{ArchivedManifest, Manifest};
use crate::layout::{
    align_up, schema as schema_codec, Extent, TableKind, BUFFER_ALIGN, DATA_START, HEADER_SIZE,
    META_PAGE_SIZE, SEGMENT_ALIGN,
};
use crate::{Error, Result};

/// The winning meta page and the manifest it points at.
#[derive(Debug, Clone)]
pub struct Committed {
    pub meta: MetaPage,
    pub slot: MetaSlot,
    pub manifest: Manifest,
}

/// A table file opened for reading, and for writing when the lock was taken.
pub struct TableFile {
    io: Arc<dyn FileIo>,
    path: PathBuf,
    schema: Arc<Schema>,
    header: FileHeader,
    options: TableOptions,
    /// Held for the lifetime of the handle. Dropping it lets another writer in.
    _lock: FileLock,
    /// The slot the next commit overwrites: the one holding the older txn.
    next_slot: MetaSlot,
    /// The manifest `next_slot` points at. The next commit frees it, because
    /// after that commit no slot refers to it any more.
    next_slot_manifest: Extent,
    committed: Committed,
}

/// Read a schema blob from the extent that names it.
async fn read_schema(io: &dyn FileIo, extent: Extent) -> Result<Schema> {
    let framed = io.read_at(extent.offset, extent.len as usize).await?;
    schema_codec::decode(frame::decode(&framed, tag::SCHEMA, "schema")?)
}

impl TableFile {
    /// Create a table file. Fails when the path already exists.
    pub async fn create(
        path: &Path,
        kind: TableKind,
        schema: Arc<Schema>,
        options: TableOptions,
    ) -> Result<Self> {
        if options.read_only {
            return Err(Error::InvalidArgument(
                "cannot create a table through a read-only handle".into(),
            ));
        }
        if path.exists() {
            return Err(Error::InvalidArgument(format!(
                "{} already exists",
                path.display()
            )));
        }

        let io = open_backend(path, options.io_backend, options.durability, false)?;
        let lock = take_lock(path, false)?;

        // Reserve the header and both meta pages before anything is appended.
        io.set_len(DATA_START).await?;

        // The schema lands first in the data region and is never freed, so the
        // header can point at a fixed extent that outlives every commit.
        let schema_bytes = schema_codec::encode(&schema);
        let schema_frame = frame::encode(tag::SCHEMA, &schema_bytes);
        let schema_offset = io.append(&[&schema_frame]).await?;
        debug_assert_eq!(schema_offset, DATA_START);
        let schema_extent = Extent::new(schema_offset, schema_frame.len() as u64);

        let header = FileHeader::new(
            kind,
            new_table_uuid(path, schema_extent),
            schema_extent,
            schema_codec::fingerprint(&schema),
        );
        write_region(
            io.as_ref(),
            0,
            HEADER_SIZE,
            tag::HEADER,
            &rkyv::to_bytes::<rkyv::rancor::Error>(&header)?,
            "header",
        )
        .await?;

        // Seed both slots with a complete commit, so the "one slot always
        // survives" rule holds from the first byte on. Each slot needs its own
        // manifest, because a meta page and the manifest it points at must
        // agree on the txn.
        let mut seed = Manifest::empty(align_up(io.len()?, SEGMENT_ALIGN));
        seed.schema = schema_extent;
        let seed_extent = write_manifest(io.as_ref(), &mut seed).await?;

        let mut manifest = seed.clone();
        manifest.txn_id = 1;
        let manifest_extent = write_manifest(io.as_ref(), &mut manifest).await?;
        io.sync_data().await?;

        let meta = MetaPage {
            txn_id: 1,
            manifest: manifest_extent,
            checkpoint_lsn: 0,
            next_lsn: 1,
            file_len: io.len()?,
        };
        write_meta(
            io.as_ref(),
            MetaSlot::B,
            &MetaPage {
                txn_id: 0,
                manifest: seed_extent,
                ..meta
            },
        )
        .await?;
        write_meta(io.as_ref(), MetaSlot::A, &meta).await?;
        io.sync_data().await?;

        Ok(Self {
            io,
            path: path.to_path_buf(),
            schema,
            header,
            options,
            _lock: lock,
            next_slot: MetaSlot::B,
            next_slot_manifest: seed_extent,
            committed: Committed {
                meta,
                slot: MetaSlot::A,
                manifest,
            },
        })
    }

    /// Open an existing table file and recover the newest complete commit.
    pub async fn open(path: &Path, kind: TableKind, options: TableOptions) -> Result<Self> {
        let io = open_backend(
            path,
            options.io_backend,
            options.durability,
            options.read_only,
        )?;
        let lock = take_lock(path, options.read_only)?;

        let header_bytes = io.read_at(0, HEADER_SIZE as usize).await?;
        let payload = frame::decode(&header_bytes, tag::HEADER, "header")?;
        let archived = rkyv::access::<ArchivedFileHeader, rkyv::rancor::Error>(payload)?;
        archived.validate(kind)?;
        let header: FileHeader =
            rkyv::deserialize::<_, rkyv::rancor::Error>(archived).map_err(Error::from)?;

        // The manifest before the schema, because the manifest is what says
        // which schema is in force. The header's is only the one the table was
        // created with, and a table that has since changed it must not be read
        // through the old one.
        let (committed, next_slot_manifest) = read_committed(io.as_ref()).await?;
        let next_slot = committed.slot.other();

        let extent = if committed.manifest.schema.is_empty() {
            header.schema
        } else {
            committed.manifest.schema
        };
        let schema = Arc::new(read_schema(io.as_ref(), extent).await?);

        Ok(Self {
            io,
            path: path.to_path_buf(),
            schema,
            header,
            options,
            _lock: lock,
            next_slot,
            next_slot_manifest,
            committed,
        })
    }

    /// Record a schema as the one in force, after a commit has made it so.
    ///
    /// Snapshots take their schema from here, so a change that did not reach
    /// this would commit to disk and stay invisible to every reader.
    pub fn set_schema(&mut self, schema: Arc<Schema>) {
        self.schema = schema;
    }

    /// Store a schema and return where it landed.
    ///
    /// Schema blobs are appended and never freed. They are small next to the
    /// data, and one stays reachable for every commit a reader might still be
    /// pinned to, which is what the free list would otherwise have to reason
    /// about.
    pub async fn write_schema(&self, schema: &Schema) -> Result<Extent> {
        let bytes = schema_codec::encode(schema);
        let framed = frame::encode(tag::SCHEMA, &bytes);
        let offset = self.io.append(&[&framed]).await?;
        Ok(Extent::new(offset, framed.len() as u64))
    }

    /// Open the table, creating it when the file is absent.
    pub async fn open_or_create(
        path: &Path,
        kind: TableKind,
        schema: Arc<Schema>,
        options: TableOptions,
    ) -> Result<Self> {
        if path.exists() {
            let file = Self::open(path, kind, options).await?;
            // Against the schema in force, not the one the file was created
            // with: a table that has added a column is still the same table.
            let want = schema_codec::fingerprint(&schema);
            let holds = schema_codec::fingerprint(file.schema());
            if holds != want {
                return Err(Error::SchemaMismatch(format!(
                    "{} holds schema {holds:#018x}, caller supplied {want:#018x}",
                    path.display(),
                )));
            }
            Ok(file)
        } else {
            Self::create(path, kind, schema, options).await
        }
    }

    pub fn io(&self) -> &Arc<dyn FileIo> {
        &self.io
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    pub fn options(&self) -> &TableOptions {
        &self.options
    }

    pub fn kind(&self) -> TableKind {
        self.header.table_kind
    }

    pub fn table_uuid(&self) -> [u8; 16] {
        self.header.table_uuid
    }

    pub fn manifest(&self) -> &Manifest {
        &self.committed.manifest
    }

    pub fn meta(&self) -> &MetaPage {
        &self.committed.meta
    }

    /// Reserve `len` bytes for a write.
    ///
    /// Reuses a freed extent when one fits and every snapshot that could still
    /// read it is gone; otherwise grows the file. `min_active_txn` is the
    /// oldest txn any live reader is pinned to. Pass `u64::MAX` when no reader
    /// can exist.
    pub fn allocate(manifest: &mut Manifest, len: u64, align: u64, min_active_txn: u64) -> Extent {
        if len == 0 {
            return Extent::EMPTY;
        }

        // Best fit among extents no live snapshot can still be reading.
        let mut best: Option<(usize, u64)> = None;
        for (i, free) in manifest.free_extents.iter().enumerate() {
            if free.freed_txn >= min_active_txn {
                continue; // a reader may still hold buffers mapped onto it
            }
            let start = align_up(free.extent.offset, align);
            let waste = start - free.extent.offset;
            if free.extent.len < waste + len {
                continue;
            }
            let slack = free.extent.len - waste - len;
            if best.is_none_or(|(_, best_slack)| slack < best_slack) {
                best = Some((i, slack));
            }
        }

        if let Some((index, _)) = best {
            let free = manifest.free_extents.remove(index);
            let start = align_up(free.extent.offset, align);
            let head = start - free.extent.offset;
            // Return the unusable head and the unused tail to the free list,
            // keeping the same txn so the quarantine still applies.
            if head > 0 {
                manifest
                    .free_extents
                    .push(crate::layout::manifest::FreeExtent {
                        extent: Extent::new(free.extent.offset, head),
                        freed_txn: free.freed_txn,
                    });
            }
            let tail_offset = start + len;
            if tail_offset < free.extent.end() {
                manifest
                    .free_extents
                    .push(crate::layout::manifest::FreeExtent {
                        extent: Extent::new(tail_offset, free.extent.end() - tail_offset),
                        freed_txn: free.freed_txn,
                    });
            }
            return Extent::new(start, len);
        }

        let start = align_up(manifest.file_len, align);
        manifest.file_len = start + len;
        Extent::new(start, len)
    }

    /// Write `bytes` at a freshly allocated extent and return it.
    pub async fn write_allocated(
        &self,
        manifest: &mut Manifest,
        bytes: &[u8],
        align: u64,
        min_active_txn: u64,
    ) -> Result<Extent> {
        let extent = Self::allocate(manifest, bytes.len() as u64, align, min_active_txn);
        if extent.is_empty() {
            return Ok(extent);
        }
        // Grow first so a write never lands past the end of the file.
        if extent.end() > self.io.len()? {
            self.io.set_len(extent.end()).await?;
        }
        self.io.write_at(extent.offset, bytes).await?;
        Ok(extent)
    }

    /// Publish `manifest` as the newest commit.
    ///
    /// The caller has already written every byte the manifest refers to. This
    /// appends the manifest itself, syncs, flips the older meta slot, and syncs
    /// again. On return the commit is durable.
    pub async fn commit(
        &mut self,
        mut manifest: Manifest,
        min_active_txn: u64,
    ) -> Result<&Committed> {
        let _ = min_active_txn; // segments reuse free extents; manifests never do
        manifest.txn_id = self.committed.meta.txn_id + 1;

        // Free the manifest belonging to the slot this commit overwrites, not
        // the current one. The current manifest stays live in the other slot
        // and is what recovery falls back to if this commit's meta write tears.
        manifest.free(self.next_slot_manifest);

        let extent = write_manifest(self.io.as_ref(), &mut manifest).await?;
        // Step 1: every byte the new meta page will point at is on disk.
        self.io.sync_data().await?;

        let meta = MetaPage {
            txn_id: manifest.txn_id,
            manifest: extent,
            checkpoint_lsn: manifest.checkpoint_lsn,
            next_lsn: self
                .committed
                .meta
                .next_lsn
                .max(manifest.checkpoint_lsn + 1),
            file_len: self.io.len()?,
        };
        // Step 2: flip the older slot. The newer one stays readable throughout.
        write_meta(self.io.as_ref(), self.next_slot, &meta).await?;
        self.io.sync_data().await?;

        // The manifest that was current becomes the fallback, so the slot the
        // next commit overwrites is the one now holding it.
        self.next_slot_manifest = self.committed.meta.manifest;
        self.committed = Committed {
            meta,
            slot: self.next_slot,
            manifest,
        };
        self.next_slot = self.next_slot.other();
        Ok(&self.committed)
    }

    /// Record the next LSN the WAL will hand out, without a full commit.
    pub fn set_next_lsn(&mut self, next_lsn: u64) {
        self.committed.meta.next_lsn = self.committed.meta.next_lsn.max(next_lsn);
    }

    /// Read a sealed extent, zero-copy where the backend allows it.
    pub async fn read_sealed(&self, extent: Extent) -> Result<SharedBuf> {
        self.io.read_immutable(extent).await
    }

    /// Swap the IO backend, so a test can interpose a failing one.
    #[cfg(any(test, feature = "testing"))]
    pub fn set_io(&mut self, io: Arc<dyn FileIo>) {
        self.io = io;
    }
}

impl std::fmt::Debug for TableFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableFile")
            .field("path", &self.path)
            .field("kind", &self.header.table_kind)
            .field("txn_id", &self.committed.meta.txn_id)
            .field("slot", &self.committed.slot)
            .field("segments", &self.committed.manifest.segments.len())
            .field("backend", &self.io.backend())
            .finish()
    }
}

/// Take the writer lock, or a reader lock for a read-only handle.
fn take_lock(path: &Path, read_only: bool) -> Result<FileLock> {
    if read_only {
        FileLock::try_shared(path)
    } else {
        FileLock::try_exclusive(path)
    }
}

/// Derive a table identity without pulling in a uuid dependency.
///
/// The value only has to be unique enough to stop a WAL sidecar attaching to a
/// different table, so a hash of the path, the creation time and the schema
/// extent is enough.
fn new_table_uuid(path: &Path, schema: Extent) -> [u8; 16] {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut seed = Vec::new();
    seed.extend_from_slice(path.to_string_lossy().as_bytes());
    seed.extend_from_slice(&nanos.to_le_bytes());
    seed.extend_from_slice(&schema.offset.to_le_bytes());
    seed.extend_from_slice(&(std::process::id() as u64).to_le_bytes());

    let high = crate::layout::checksum(&seed);
    seed.extend_from_slice(&high.to_le_bytes());
    let low = crate::layout::checksum(&seed);

    let mut uuid = [0u8; 16];
    uuid[..8].copy_from_slice(&high.to_le_bytes());
    uuid[8..].copy_from_slice(&low.to_le_bytes());
    uuid
}

/// Slack reserved around a manifest record, so its own `file_len` field can be
/// filled in after the size is known without changing the size again.
const MANIFEST_SLACK: u64 = 256;

/// Serialize a manifest, place it at the end of the file, and write it.
///
/// A manifest records the file length that includes the manifest itself, so
/// the record and the value inside it have to agree. Reserving a slot slightly
/// larger than the record breaks the circularity: the slot size is fixed before
/// the final serialization, and the frame header carries its own payload
/// length, so the unused tail of the slot is simply padding.
///
/// Manifests always go at the end of the file, never into a free extent. Old
/// manifests join the free list and are reused by segments instead.
///
/// Updates `manifest.file_len` and returns the extent the record occupies.
async fn write_manifest(io: &dyn FileIo, manifest: &mut Manifest) -> Result<Extent> {
    let start = align_up(manifest.file_len.max(io.len()?), BUFFER_ALIGN);

    let probe = rkyv::to_bytes::<rkyv::rancor::Error>(&*manifest)?;
    let slot = align_up(
        frame::frame_len(probe.len()) as u64 + MANIFEST_SLACK,
        BUFFER_ALIGN,
    );

    manifest.file_len = start + slot;
    let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&*manifest)?;
    let bytes = frame::encode(tag::MANIFEST, &payload);
    if bytes.len() as u64 > slot {
        // Only reachable if serialization grew by more than the slack, which
        // would mean the second pass changed something other than `file_len`.
        return Err(Error::corrupt(format!(
            "manifest grew to {} bytes, past its {slot}-byte slot",
            bytes.len()
        )));
    }

    let extent = Extent::new(start, slot);
    if extent.end() > io.len()? {
        io.set_len(extent.end()).await?;
    }
    io.write_at(extent.offset, &bytes).await?;
    Ok(extent)
}

/// Write a framed rkyv payload into a fixed-size region, zero padded.
async fn write_region(
    io: &dyn FileIo,
    offset: u64,
    region_len: u64,
    tag: u64,
    payload: &[u8],
    region: &'static str,
) -> Result<()> {
    let frame = frame::encode(tag, payload);
    if frame.len() as u64 > region_len {
        return Err(Error::Unsupported(format!(
            "{region} needs {} bytes, the region holds {region_len}",
            frame.len()
        )));
    }
    let mut page = vec![0u8; region_len as usize];
    page[..frame.len()].copy_from_slice(&frame);
    io.write_at(offset, &page).await
}

async fn write_meta(io: &dyn FileIo, slot: MetaSlot, meta: &MetaPage) -> Result<()> {
    let payload = rkyv::to_bytes::<rkyv::rancor::Error>(meta)?;
    write_region(
        io,
        slot.offset(),
        META_PAGE_SIZE,
        tag::META,
        &payload,
        "meta page",
    )
    .await
}

/// Read one meta slot. A torn or absent slot reports `None`, not an error.
async fn read_meta(io: &dyn FileIo, slot: MetaSlot) -> Result<Option<MetaPage>> {
    let bytes = match io.read_at(slot.offset(), META_PAGE_SIZE as usize).await {
        Ok(bytes) => bytes,
        Err(_) => return Ok(None),
    };
    let Ok(payload) = frame::decode(&bytes, tag::META, "meta page") else {
        return Ok(None);
    };
    let Ok(archived) = rkyv::access::<ArchivedMetaPage, rkyv::rancor::Error>(payload) else {
        return Ok(None);
    };
    Ok(Some(archived.to_native()))
}

/// Read the manifest a meta page points at.
async fn read_manifest(io: &dyn FileIo, meta: &MetaPage) -> Result<Manifest> {
    let bytes = io
        .read_at(meta.manifest.offset, meta.manifest.len as usize)
        .await?;
    let payload = frame::decode(&bytes, tag::MANIFEST, "manifest")?;
    let archived = rkyv::access::<ArchivedManifest, rkyv::rancor::Error>(payload)?;
    Ok(archived.to_native())
}

/// Pick the newest slot whose manifest reads back cleanly.
///
/// Returns the winning commit and the manifest extent the losing slot points
/// at. That second extent must stay allocated until the next commit overwrites
/// its slot, because until then it is what a torn write recovers to.
async fn read_committed(io: &dyn FileIo) -> Result<(Committed, Extent)> {
    let mut slots: Vec<(MetaSlot, MetaPage)> = Vec::with_capacity(2);
    for slot in [MetaSlot::A, MetaSlot::B] {
        if let Some(meta) = read_meta(io, slot).await? {
            slots.push((slot, meta));
        }
    }
    if slots.is_empty() {
        return Err(Error::corrupt(
            "both meta pages are unreadable; the table cannot be recovered",
        ));
    }
    // Newest first, so a torn newer slot falls back to the older one.
    slots.sort_by_key(|(_, meta)| std::cmp::Reverse(meta.txn_id));

    let mut last_err = None;
    for (index, (slot, meta)) in slots.iter().enumerate() {
        match read_manifest(io, meta).await {
            Ok(manifest) => {
                if manifest.txn_id != meta.txn_id {
                    last_err = Some(Error::corrupt(format!(
                        "meta slot {slot:?} claims txn {} but its manifest holds txn {}",
                        meta.txn_id, manifest.txn_id
                    )));
                    continue;
                }
                // A slot that lost, or that failed to read, still owns whatever
                // extent it names. Treat an unreadable slot as owning nothing:
                // the next commit overwrites it either way.
                let loser = slots
                    .iter()
                    .enumerate()
                    .find(|(i, _)| *i != index)
                    .map(|(_, (_, other))| other.manifest)
                    .unwrap_or(Extent::EMPTY);
                return Ok((
                    Committed {
                        meta: *meta,
                        slot: *slot,
                        manifest,
                    },
                    loser,
                ));
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| Error::corrupt("no meta slot points at a readable manifest")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Durability;
    use crate::layout::manifest::SegmentEntry;
    use arrow_schema::{DataType, Field};

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn options() -> TableOptions {
        TableOptions::default().with_durability(Durability::None)
    }

    async fn create(dir: &tempfile::TempDir) -> TableFile {
        TableFile::create(
            &dir.path().join("table.lt"),
            TableKind::Columnar,
            schema(),
            options(),
        )
        .await
        .unwrap()
    }

    fn segment(id: u64, offset: u64) -> SegmentEntry {
        SegmentEntry {
            segment_id: id,
            data: Extent::new(offset, 4096),
            meta: Extent::new(offset + 3000, 1096),
            row_count: 1000,
            deleted_count: 0,
            deletes: None,
        }
    }

    #[tokio::test]
    async fn a_fresh_table_opens_empty() {
        let dir = tempfile::tempdir().unwrap();
        let file = create(dir_ref(&dir)).await;

        assert_eq!(file.manifest().segments.len(), 0);
        assert_eq!(file.meta().txn_id, 1);
        assert_eq!(file.kind(), TableKind::Columnar);
        assert_eq!(file.schema().as_ref(), schema().as_ref());
    }

    fn dir_ref(dir: &tempfile::TempDir) -> &tempfile::TempDir {
        dir
    }

    #[tokio::test]
    async fn creating_over_an_existing_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let _file = create(&dir).await;
        let err = TableFile::create(
            &dir.path().join("table.lt"),
            TableKind::Columnar,
            schema(),
            options(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn commits_survive_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("table.lt");
        {
            let mut file = create(&dir).await;
            let mut manifest = file.manifest().clone();
            manifest.segments.push(segment(0, 1 << 20));
            manifest.next_segment_id = 1;
            manifest.file_len = (1 << 20) + 4096;
            file.commit(manifest, u64::MAX).await.unwrap();
            assert_eq!(file.meta().txn_id, 2);
        }

        let file = TableFile::open(&path, TableKind::Columnar, options())
            .await
            .unwrap();
        assert_eq!(file.meta().txn_id, 2);
        assert_eq!(file.manifest().segments.len(), 1);
        assert_eq!(file.manifest().segments[0].segment_id, 0);
    }

    #[tokio::test]
    async fn commits_alternate_between_the_two_slots() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = create(&dir).await;
        assert_eq!(file.committed.slot, MetaSlot::A);

        for expected in [MetaSlot::B, MetaSlot::A, MetaSlot::B] {
            let manifest = file.manifest().clone();
            file.commit(manifest, u64::MAX).await.unwrap();
            assert_eq!(file.committed.slot, expected);
        }
        assert_eq!(file.meta().txn_id, 4);
    }

    #[tokio::test]
    async fn a_torn_newer_slot_falls_back_to_the_older_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("table.lt");
        {
            let mut file = create(&dir).await;
            let mut manifest = file.manifest().clone();
            manifest.segments.push(segment(0, 1 << 20));
            manifest.file_len = (1 << 20) + 4096;
            file.commit(manifest, u64::MAX).await.unwrap();
            // txn 2 sits in slot B; txn 1 in slot A describes an empty table.
            assert_eq!(file.committed.slot, MetaSlot::B);
        }

        // Simulate a write that tore halfway through slot B.
        {
            use std::os::unix::fs::FileExt;
            let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.write_all_at(&[0xa5; 64], MetaSlot::B.offset() + 8)
                .unwrap();
        }

        let file = TableFile::open(&path, TableKind::Columnar, options())
            .await
            .unwrap();
        assert_eq!(
            file.meta().txn_id,
            1,
            "recovery fell back to the intact slot"
        );
        assert_eq!(file.manifest().segments.len(), 0);
    }

    #[tokio::test]
    async fn losing_both_slots_is_reported_not_papered_over() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("table.lt");
        create(&dir).await;
        {
            use std::os::unix::fs::FileExt;
            let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.write_all_at(&[0u8; 8192], HEADER_SIZE).unwrap();
        }
        let err = TableFile::open(&path, TableKind::Columnar, options())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn open_or_create_rejects_a_different_schema() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("table.lt");
        create(&dir).await;

        let other = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let err = TableFile::open_or_create(&path, TableKind::Columnar, other, options())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::SchemaMismatch(_)), "got {err:?}");

        TableFile::open_or_create(&path, TableKind::Columnar, schema(), options())
            .await
            .expect("the original schema still opens");
    }

    #[tokio::test]
    async fn a_second_writer_cannot_open_the_table() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("table.lt");
        let _held = create(&dir).await;

        let err = TableFile::open(&path, TableKind::Columnar, options())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::WriterLocked(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn a_reader_cannot_open_while_a_writer_holds_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("table.lt");
        let _held = create(&dir).await;

        let err = TableFile::open(&path, TableKind::Columnar, options().read_only())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::WriterLocked(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn readers_share_a_closed_table() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("table.lt");
        create(&dir).await;

        let a = TableFile::open(&path, TableKind::Columnar, options().read_only())
            .await
            .unwrap();
        let b = TableFile::open(&path, TableKind::Columnar, options().read_only())
            .await
            .unwrap();
        assert_eq!(a.meta().txn_id, b.meta().txn_id);
    }

    #[test]
    fn allocation_grows_the_file_when_nothing_is_free() {
        let mut manifest = Manifest::empty(DATA_START);
        let a = TableFile::allocate(&mut manifest, 100, BUFFER_ALIGN, u64::MAX);
        let b = TableFile::allocate(&mut manifest, 100, BUFFER_ALIGN, u64::MAX);

        assert_eq!(a.offset, DATA_START);
        assert_eq!(b.offset, align_up(a.end(), BUFFER_ALIGN));
        assert!(!a.overlaps(&b));
        assert_eq!(manifest.file_len, b.end());
    }

    #[test]
    fn allocation_honours_segment_alignment() {
        let mut manifest = Manifest::empty(DATA_START + 17);
        let extent = TableFile::allocate(&mut manifest, 4096, SEGMENT_ALIGN, u64::MAX);
        assert_eq!(extent.offset % SEGMENT_ALIGN, 0);
    }

    #[test]
    fn a_freed_extent_is_reused_once_no_reader_can_hold_it() {
        let mut manifest = Manifest::empty(DATA_START);
        manifest.txn_id = 5;
        manifest.free(Extent::new(1 << 20, 8192));

        // A reader pinned at txn 5 could still be reading those bytes.
        let grown = TableFile::allocate(&mut manifest, 4096, BUFFER_ALIGN, 5);
        assert_eq!(grown.offset, DATA_START, "quarantine forced a fresh extent");
        assert_eq!(manifest.free_extents.len(), 1);

        // Once the oldest reader moved past txn 5, the extent is fair game.
        let reused = TableFile::allocate(&mut manifest, 4096, BUFFER_ALIGN, 6);
        assert_eq!(reused.offset, 1 << 20);
        assert_eq!(reused.len, 4096);
    }

    #[test]
    fn reuse_returns_the_unused_tail_to_the_free_list() {
        let mut manifest = Manifest::empty(DATA_START);
        manifest.txn_id = 1;
        manifest.free(Extent::new(1 << 20, 8192));

        let taken = TableFile::allocate(&mut manifest, 1000, BUFFER_ALIGN, 2);
        assert_eq!(taken, Extent::new(1 << 20, 1000));
        assert_eq!(manifest.free_extents.len(), 1);
        assert_eq!(
            manifest.free_extents[0].extent,
            Extent::new((1 << 20) + 1000, 7192)
        );
        assert_eq!(manifest.free_extents[0].freed_txn, 1, "quarantine is kept");
    }

    #[test]
    fn allocation_prefers_the_tightest_free_extent() {
        let mut manifest = Manifest::empty(DATA_START);
        manifest.txn_id = 1;
        manifest.free(Extent::new(1 << 20, 1 << 16));
        manifest.free(Extent::new(2 << 20, 4096));

        let taken = TableFile::allocate(&mut manifest, 4000, BUFFER_ALIGN, 2);
        assert_eq!(
            taken.offset,
            2 << 20,
            "the snug extent wins over the roomy one"
        );
    }

    #[test]
    fn zero_length_allocations_take_no_space() {
        let mut manifest = Manifest::empty(DATA_START);
        assert_eq!(
            TableFile::allocate(&mut manifest, 0, BUFFER_ALIGN, u64::MAX),
            Extent::EMPTY
        );
        assert_eq!(manifest.file_len, DATA_START);
    }

    #[tokio::test]
    async fn a_commit_frees_only_the_manifest_no_slot_points_at() {
        let dir = tempfile::tempdir().unwrap();
        let mut file = create(&dir).await;
        let overwritten = file.next_slot_manifest;
        let fallback = file.meta().manifest;

        let manifest = file.manifest().clone();
        file.commit(manifest, u64::MAX).await.unwrap();

        let freed = |e: Extent| file.manifest().free_extents.iter().any(|f| f.extent == e);
        assert!(
            freed(overwritten),
            "the overwritten slot's manifest is garbage"
        );
        assert!(
            !freed(fallback),
            "the other slot still points at this manifest; freeing it would let a \
             later commit overwrite what recovery falls back to"
        );
        assert_eq!(file.next_slot_manifest, fallback);
    }

    #[tokio::test]
    async fn the_fallback_manifest_is_never_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("table.lt");

        // Commit repeatedly, so the allocator has plenty of freed manifests to
        // reuse, then tear the newest slot and check the fallback still reads.
        let torn_slot = {
            let mut file = create(&dir).await;
            for i in 0..20u64 {
                let mut manifest = file.manifest().clone();
                manifest.next_segment_id = i + 1;
                manifest.segments.push(segment(i, (1 << 20) + i * 8192));
                manifest.file_len = manifest.file_len.max((1 << 20) + (i + 1) * 8192);
                file.commit(manifest, u64::MAX).await.unwrap();
            }
            file.committed.slot
        };

        {
            use std::os::unix::fs::FileExt;
            let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.write_all_at(&[0x5a; 128], torn_slot.offset() + 8)
                .unwrap();
        }

        let file = TableFile::open(&path, TableKind::Columnar, options())
            .await
            .unwrap();
        assert_eq!(file.meta().txn_id, 20, "fell back one commit, not further");
        assert_eq!(file.manifest().segments.len(), 19);
    }

    #[tokio::test]
    async fn many_commits_keep_the_file_readable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("table.lt");
        {
            let mut file = create(&dir).await;
            for i in 0..50u64 {
                let mut manifest = file.manifest().clone();
                manifest.next_segment_id = i + 1;
                file.commit(manifest, u64::MAX).await.unwrap();
            }
            assert_eq!(file.meta().txn_id, 51);
        }
        let file = TableFile::open(&path, TableKind::Columnar, options())
            .await
            .unwrap();
        assert_eq!(file.meta().txn_id, 51);
        assert_eq!(file.manifest().next_segment_id, 50);
    }

    /// Crash at every byte boundary of a commit and check what survives.
    ///
    /// A commit must be all or nothing: reopening after a crash gives either
    /// the state before the commit or the state after it, and never a mix, a
    /// panic, or an unreadable file.
    #[tokio::test]
    async fn a_commit_is_atomic_at_every_crash_point() {
        use crate::io::fault::FaultIo;

        // Enough writes that the sweep covers the manifest record, both syncs
        // and the meta page flip.
        const MAX_BUDGET: u64 = 6000;

        let mut torn_before = 0;
        let mut torn_after = 0;

        for budget in (0..MAX_BUDGET).step_by(37) {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("table.lt");

            // A table with one commit already in it, so "before" is a state
            // worth distinguishing from "after".
            {
                let mut file = create(&dir).await;
                let mut manifest = file.manifest().clone();
                manifest.segments.push(segment(0, 1 << 20));
                manifest.next_segment_id = 1;
                manifest.file_len = (1 << 20) + 4096;
                file.commit(manifest, u64::MAX).await.unwrap();
            }

            // Reopen, then crash partway through the second commit.
            {
                let mut file = TableFile::open(&path, TableKind::Columnar, options())
                    .await
                    .unwrap();
                let inner = file.io.clone();
                file.set_io(Arc::new(FaultIo::with_budget(inner, budget)));

                let mut manifest = file.manifest().clone();
                manifest.segments.push(segment(1, 2 << 20));
                manifest.next_segment_id = 2;
                manifest.file_len = (2 << 20) + 4096;
                let _ = file.commit(manifest, u64::MAX).await;
            }

            let recovered = TableFile::open(&path, TableKind::Columnar, options())
                .await
                .unwrap_or_else(|e| panic!("budget {budget}: table did not reopen: {e}"));

            let segments = recovered.manifest().segments.len();
            match segments {
                1 => {
                    torn_before += 1;
                    assert_eq!(recovered.meta().txn_id, 2, "budget {budget}");
                }
                2 => {
                    torn_after += 1;
                    assert_eq!(recovered.meta().txn_id, 3, "budget {budget}");
                    assert_eq!(recovered.manifest().segments[1].segment_id, 1);
                }
                other => panic!("budget {budget}: recovered {other} segments, expected 1 or 2"),
            }

            // Whatever it recovered to, the table must still take a commit.
            let mut recovered = recovered;
            let manifest = recovered.manifest().clone();
            recovered
                .commit(manifest, u64::MAX)
                .await
                .unwrap_or_else(|e| panic!("budget {budget}: recovered table cannot commit: {e}"));
        }

        assert!(
            torn_before > 0,
            "no crash landed before the commit took effect"
        );
        assert!(
            torn_after > 0,
            "no crash landed after the commit took effect"
        );
    }

    #[tokio::test]
    async fn write_allocated_lands_where_the_manifest_says() {
        let dir = tempfile::tempdir().unwrap();
        let file = create(&dir).await;
        let mut manifest = file.manifest().clone();

        let payload = vec![0xab; 5000];
        let extent = file
            .write_allocated(&mut manifest, &payload, SEGMENT_ALIGN, u64::MAX)
            .await
            .unwrap();

        assert_eq!(extent.offset % SEGMENT_ALIGN, 0);
        let read = file.read_sealed(extent).await.unwrap();
        assert_eq!(read.as_slice(), &payload[..]);
    }
}
