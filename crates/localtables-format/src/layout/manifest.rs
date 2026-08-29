//! The manifest: the full list of live segments at one commit.
//!
//! Each commit appends a fresh manifest frame and points the new meta page at
//! it. The previous manifest becomes a free extent, quarantined until every
//! snapshot that could still read it is gone.

use rkyv::{Archive, Deserialize, Serialize};

use crate::layout::Extent;

/// Identifies a segment inside one table. Never reused.
pub type SegmentId = u64;

/// One immutable columnar segment.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[rkyv(derive(Debug))]
pub struct SegmentEntry {
    pub segment_id: SegmentId,
    /// The whole segment: column pages followed by the segment meta frame.
    /// Mapped as one unit, and freed as one unit.
    pub data: Extent,
    /// The segment meta frame, inside `data`.
    pub meta: Extent,
    /// Rows written into the segment, before deletes.
    pub row_count: u64,
    /// Rows marked deleted in `deletes`. Cached so pruning avoids a bitmap read.
    pub deleted_count: u64,
    /// Frame holding the serialized delete bitmap. Absent when nothing is deleted.
    pub deletes: Option<Extent>,
}

impl SegmentEntry {
    /// Rows a scan still returns.
    pub fn live_rows(&self) -> u64 {
        self.row_count.saturating_sub(self.deleted_count)
    }

    /// True once deletes dominate the segment, so compaction should rewrite it.
    pub fn is_compaction_candidate(&self, dead_ratio: f64) -> bool {
        self.row_count > 0 && (self.deleted_count as f64 / self.row_count as f64) >= dead_ratio
    }
}

/// A byte range no live commit references any more.
///
/// `freed_txn` records when it became garbage. The allocator hands it out only
/// after every snapshot older than that txn is dropped, because those snapshots
/// may hold Arrow buffers mapped straight onto these bytes.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[rkyv(derive(Debug))]
pub struct FreeExtent {
    pub extent: Extent,
    pub freed_txn: u64,
}

/// The root of a b-tree, as a manifest records it.
///
/// A columnar table leaves this empty; a b-tree table leaves `segments` empty.
/// One manifest type serves both, so the commit protocol, the free list and
/// the recovery path are written once rather than twice.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[rkyv(derive(Debug))]
pub struct TreeRootEntry {
    /// The root page. Empty when the tree holds nothing.
    pub root: Extent,
    pub rows: u64,
    /// Levels between the root and a leaf. Zero when the root is a leaf.
    pub height: u32,
}

/// The complete state of a table at one commit.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[rkyv(derive(Debug))]
pub struct Manifest {
    pub txn_id: u64,
    /// WAL records at or below this LSN are durable inside the segments.
    pub checkpoint_lsn: u64,
    pub next_segment_id: SegmentId,
    /// Next row sequence number the memtable will hand out.
    pub next_seqno: u64,
    pub segments: Vec<SegmentEntry>,
    /// The b-tree's root, for a b-tree table. Empty for a columnar one.
    pub tree: TreeRootEntry,
    pub free_extents: Vec<FreeExtent>,
    /// End of the allocated region. New extents start here when no free extent fits.
    pub file_len: u64,
}

impl Manifest {
    /// The manifest of a table that holds no data yet.
    pub fn empty(file_len: u64) -> Self {
        Self {
            txn_id: 0,
            checkpoint_lsn: 0,
            next_segment_id: 0,
            next_seqno: 0,
            segments: Vec::new(),
            tree: TreeRootEntry::default(),
            free_extents: Vec::new(),
            file_len,
        }
    }

    pub fn total_rows(&self) -> u64 {
        self.segments.iter().map(|s| s.row_count).sum()
    }

    pub fn live_rows(&self) -> u64 {
        self.segments.iter().map(|s| s.live_rows()).sum()
    }

    pub fn segment(&self, id: SegmentId) -> Option<&SegmentEntry> {
        self.segments.iter().find(|s| s.segment_id == id)
    }

    pub fn segment_mut(&mut self, id: SegmentId) -> Option<&mut SegmentEntry> {
        self.segments.iter_mut().find(|s| s.segment_id == id)
    }

    /// Mark `extent` as garbage as of this manifest's txn.
    pub fn free(&mut self, extent: Extent) {
        if !extent.is_empty() {
            self.free_extents.push(FreeExtent {
                extent,
                freed_txn: self.txn_id,
            });
        }
    }
}

impl ArchivedManifest {
    /// Copy the archive into an owned manifest. A commit mutates the owned form
    /// and writes a new frame; readers use the archive in place.
    pub fn to_native(&self) -> Manifest {
        Manifest {
            txn_id: self.txn_id.to_native(),
            checkpoint_lsn: self.checkpoint_lsn.to_native(),
            next_segment_id: self.next_segment_id.to_native(),
            next_seqno: self.next_seqno.to_native(),
            segments: self
                .segments
                .iter()
                .map(|s| SegmentEntry {
                    segment_id: s.segment_id.to_native(),
                    data: s.data.to_native(),
                    meta: s.meta.to_native(),
                    row_count: s.row_count.to_native(),
                    deleted_count: s.deleted_count.to_native(),
                    deletes: s.deletes.as_ref().map(|e| e.to_native()),
                })
                .collect(),
            tree: TreeRootEntry {
                root: self.tree.root.to_native(),
                rows: self.tree.rows.to_native(),
                height: self.tree.height.to_native(),
            },
            free_extents: self
                .free_extents
                .iter()
                .map(|f| FreeExtent {
                    extent: f.extent.to_native(),
                    freed_txn: f.freed_txn.to_native(),
                })
                .collect(),
            file_len: self.file_len.to_native(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(row_count: u64, deleted_count: u64) -> SegmentEntry {
        SegmentEntry {
            segment_id: 1,
            data: Extent::new(4096, 1024),
            meta: Extent::new(4096 + 900, 124),
            row_count,
            deleted_count,
            deletes: None,
        }
    }

    #[test]
    fn live_rows_subtracts_deletes() {
        assert_eq!(entry(100, 30).live_rows(), 70);
        assert_eq!(entry(100, 200).live_rows(), 0, "saturates, never wraps");
    }

    #[test]
    fn compaction_triggers_at_the_dead_ratio() {
        assert!(entry(100, 50).is_compaction_candidate(0.5));
        assert!(!entry(100, 49).is_compaction_candidate(0.5));
        assert!(
            !entry(0, 0).is_compaction_candidate(0.5),
            "empty is not a candidate"
        );
    }

    #[test]
    fn free_records_the_current_txn_and_skips_empty_extents() {
        let mut manifest = Manifest::empty(crate::layout::DATA_START);
        manifest.txn_id = 7;
        manifest.free(Extent::new(8192, 512));
        manifest.free(Extent::EMPTY);
        assert_eq!(manifest.free_extents.len(), 1);
        assert_eq!(manifest.free_extents[0].freed_txn, 7);
    }

    #[test]
    fn manifest_round_trips_through_rkyv() {
        let mut manifest = Manifest::empty(crate::layout::DATA_START);
        manifest.txn_id = 42;
        manifest.checkpoint_lsn = 9;
        manifest.next_segment_id = 3;
        manifest.segments.push(entry(1000, 4));
        manifest.segments.push(SegmentEntry {
            segment_id: 2,
            deletes: Some(Extent::new(9999, 64)),
            ..entry(500, 500)
        });
        manifest.free(Extent::new(1 << 20, 4096));

        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&manifest).unwrap();
        let archived = rkyv::access::<ArchivedManifest, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(archived.to_native(), manifest);
        assert_eq!(archived.to_native().live_rows(), 996);
    }
}
