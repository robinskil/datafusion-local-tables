//! Snapshots, and the rule that keeps their bytes alive.
//!
//! A reader pins a snapshot for the length of a query. It sees the table
//! exactly as it stood at one commit.
//!
//! A scan gives out Arrow buffers that point straight into the mapped file. So
//! a snapshot is more than a list of segments. It is a claim on the bytes those
//! segments occupy.
//!
//! A freed extent therefore cannot go out again at once. The registry tracks
//! which commits readers still hold. The allocator reuses an extent only after
//! every reader that could read it is gone.

use std::sync::{Arc, Weak};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use parking_lot::Mutex;

use crate::columnar::delete_vector::DeleteVector;
use crate::layout::manifest::{Manifest, SegmentEntry, SegmentId};

/// The table as it stood at one commit.
///
/// Cloning the `Arc` is what pins it. Dropping the last clone releases the
/// commit's bytes for reuse.
#[derive(Debug)]
pub struct Snapshot {
    /// The commit this snapshot reads.
    pub txn_id: u64,
    pub schema: SchemaRef,
    pub manifest: Arc<Manifest>,
    /// Deleted rows per segment.
    ///
    /// These stay in memory, rather than come back from disk per snapshot. They
    /// are also fresher than the counts in the manifest: a delete is durable in
    /// the log well before a flush records it in a commit.
    ///
    /// A segment with nothing deleted is absent, so the common case needs no
    /// lookup.
    pub deletes: Vec<(SegmentId, Arc<DeleteVector>)>,
    /// Rows durable in the write-ahead log but not yet written to a segment.
    /// A scan reads these alongside the segments, so a row is visible as soon
    /// as its insert returns.
    pub memtable: Arc<Vec<RecordBatch>>,
}

impl Snapshot {
    pub fn deletes_for(&self, segment_id: SegmentId) -> Option<&Arc<DeleteVector>> {
        self.deletes
            .iter()
            .find(|(id, _)| *id == segment_id)
            .map(|(_, dv)| dv)
    }

    /// Rows still readable in one segment.
    pub fn live_rows_in(&self, entry: &SegmentEntry) -> u64 {
        let deleted = self
            .deletes_for(entry.segment_id)
            .map_or(0, |dv| dv.len().min(entry.row_count));
        entry.row_count - deleted
    }

    /// Rows held in memory but not yet in a segment.
    pub fn memtable_rows(&self) -> u64 {
        self.memtable.iter().map(|b| b.num_rows() as u64).sum()
    }

    /// Rows a scan of this snapshot returns.
    pub fn live_rows(&self) -> u64 {
        self.manifest
            .segments
            .iter()
            .map(|entry| self.live_rows_in(entry))
            .sum::<u64>()
            + self.memtable_rows()
    }

    /// Segments with at least one row left to read.
    pub fn live_segments(&self) -> impl Iterator<Item = &SegmentEntry> {
        self.manifest
            .segments
            .iter()
            .filter(|entry| self.live_rows_in(entry) > 0)
    }
}

/// Tracks which commits readers are still pinned to.
///
/// Holds weak references, so a snapshot the last reader dropped disappears from
/// the registry without anyone having to say so.
#[derive(Debug, Default)]
pub struct SnapshotRegistry {
    live: Mutex<Vec<Weak<Snapshot>>>,
}

impl SnapshotRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Track a newly committed snapshot.
    ///
    /// Call this once per commit, not once per reader. A reader pins a snapshot
    /// by cloning the `Arc`, which the weak reference already follows; adding an
    /// entry per reader would leave one behind for every read of a snapshot
    /// that is still current.
    pub fn publish(&self, snapshot: Arc<Snapshot>) -> Arc<Snapshot> {
        let mut live = self.live.lock();
        live.retain(|weak| weak.strong_count() > 0);
        debug_assert!(
            !live.iter().any(|weak| weak
                .upgrade()
                .is_some_and(|existing| Arc::ptr_eq(&existing, &snapshot))),
            "a snapshot was published twice"
        );
        live.push(Arc::downgrade(&snapshot));
        snapshot
    }

    /// The oldest commit any reader is still pinned to.
    ///
    /// `u64::MAX` when nothing is pinned, which lets the allocator reuse every
    /// freed extent. That is the right answer, not a sentinel hack: with no
    /// readers, no bytes are being looked at.
    pub fn min_active_txn(&self) -> u64 {
        let mut live = self.live.lock();
        live.retain(|weak| weak.strong_count() > 0);
        live.iter()
            .filter_map(|weak| weak.upgrade())
            .map(|snapshot| snapshot.txn_id)
            .min()
            .unwrap_or(u64::MAX)
    }

    /// How many snapshots are still pinned. For tests and diagnostics.
    pub fn active_count(&self) -> usize {
        let mut live = self.live.lock();
        live.retain(|weak| weak.strong_count() > 0);
        live.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema};

    fn snapshot(txn_id: u64) -> Arc<Snapshot> {
        Arc::new(Snapshot {
            txn_id,
            schema: Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)])),
            manifest: Arc::new(Manifest::empty(crate::layout::DATA_START)),
            deletes: Vec::new(),
            memtable: Arc::new(Vec::new()),
        })
    }

    #[test]
    fn nothing_pinned_means_every_extent_is_reusable() {
        let registry = SnapshotRegistry::new();
        assert_eq!(registry.min_active_txn(), u64::MAX);
        assert_eq!(registry.active_count(), 0);
    }

    #[test]
    fn the_oldest_pinned_commit_holds_the_line() {
        let registry = SnapshotRegistry::new();
        let old = registry.publish(snapshot(5));
        let new = registry.publish(snapshot(9));

        assert_eq!(registry.min_active_txn(), 5);
        assert_eq!(registry.active_count(), 2);

        drop(old);
        assert_eq!(
            registry.min_active_txn(),
            9,
            "once the old reader leaves, its bytes are free"
        );
        drop(new);
        assert_eq!(registry.min_active_txn(), u64::MAX);
    }

    #[test]
    fn dropped_snapshots_leave_no_trace() {
        let registry = SnapshotRegistry::new();
        for txn in 0..100 {
            drop(registry.publish(snapshot(txn)));
        }
        assert_eq!(registry.active_count(), 0, "weak entries must be pruned");
    }

    #[test]
    fn deletes_are_found_by_segment() {
        let mut snapshot = Snapshot {
            txn_id: 1,
            schema: Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)])),
            manifest: Arc::new(Manifest::empty(crate::layout::DATA_START)),
            deletes: Vec::new(),
            memtable: Arc::new(Vec::new()),
        };
        snapshot
            .deletes
            .push((7, Arc::new(DeleteVector::from_iter([1u32, 2]))));

        assert_eq!(snapshot.deletes_for(7).unwrap().len(), 2);
        assert!(
            snapshot.deletes_for(8).is_none(),
            "a segment with nothing deleted must not carry an empty bitmap"
        );
    }
}
