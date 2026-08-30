//! Compaction: rewrite segments to drop deleted rows or apply new options.
//!
//! Work is cut into runs of bounded size. Each run commits on its own, so a
//! table larger than memory can still be compacted.

use super::*;

impl ColumnarTable {
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

    /// Rewrite every segment with the options this handle holds.
    ///
    /// A segment fixes its filters, its order and its encodings when the writer
    /// writes it. A table gains or loses them by a rewrite, never by a reopen.
    ///
    /// Open the table with the options you want, then call this. It is how a
    /// table takes on a membership filter, a trigram filter or a z-order it was
    /// not created with. It is also how a table sheds one.
    ///
    /// Returns the rows rewritten. Run this as maintenance, not under load.
    pub async fn rewrite_all(&self) -> Result<u64> {
        let segment_ids: Vec<SegmentId> = self
            .snapshot()
            .live_segments()
            .map(|entry| entry.segment_id)
            .collect();
        self.compact_segments(&segment_ids).await
    }

    /// Rewrite the named segments, dropping their deleted rows.
    ///
    /// The work is cut into runs of at most
    /// [`TableOptions::compaction_max_bytes`] of source data, and **each run is
    /// its own commit**. Reading every row first would be simpler and would
    /// mean a table larger than memory could never be compacted at all.
    ///
    /// Committing per run also keeps the writer lock short, and leaves a valid
    /// table at every point: a run that fails leaves the runs before it
    /// compacted and the rest untouched, and running again finishes the job.
    /// The count returned covers the runs that committed.
    pub async fn compact_segments(&self, segment_ids: &[SegmentId]) -> Result<u64> {
        if segment_ids.is_empty() {
            return Ok(0);
        }

        let snapshot = self.snapshot();
        let mut rows = 0;
        for run in self.runs(&snapshot, segment_ids)? {
            rows += self.compact_run(&run).await?;
        }
        Ok(rows)
    }

    /// Cut segments into runs that each fit the memory budget.
    ///
    /// A run always holds at least one segment, so a segment larger than the
    /// budget is a run on its own rather than an error: one segment is the
    /// smallest unit a rewrite can work in.
    pub(super) fn runs(
        &self,
        snapshot: &Snapshot,
        segment_ids: &[SegmentId],
    ) -> Result<Vec<Vec<SegmentId>>> {
        let budget = self.inner.options.compaction_max_bytes.max(1);
        let mut runs: Vec<Vec<SegmentId>> = Vec::new();
        let mut bytes = 0u64;

        for segment_id in segment_ids {
            let Some(entry) = snapshot.manifest.segment(*segment_id) else {
                return Err(Error::InvalidArgument(format!(
                    "segment {segment_id} is not in this table"
                )));
            };
            let size = entry.data.len;
            match runs.last_mut() {
                Some(run) if bytes + size <= budget => {
                    run.push(*segment_id);
                    bytes += size;
                }
                _ => {
                    runs.push(vec![*segment_id]);
                    bytes = size;
                }
            }
        }
        Ok(runs)
    }

    pub(super) async fn compact_run(&self, segment_ids: &[SegmentId]) -> Result<u64> {
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

        // Rewriting uses the same sizing a flush would, so a compaction cannot
        // undo a table's divisibility by merging everything into one segment.
        let group_rows = self
            .inner
            .options
            .row_group_size_for(manifest.total_rows().max(rows));
        let current = self.table_schema();
        for group in self.row_groups(live, group_rows, &current)? {
            self.write_segment(&writer.file, &mut manifest, &group, &current, min_active)
                .await?;
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
}
