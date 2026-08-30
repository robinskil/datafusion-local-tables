//! Rows in: insert, delete, update, and the flush that makes them durable.
//!
//! One writer at a time holds the mutex. Each change goes to the log first,
//! then to the memtable. A flush turns the memtable into segments.

use super::*;

impl ColumnarTable {
    /// Append rows.
    ///
    /// The rows are durable when this returns, and the next scan sees them.
    /// They sit in the log and the memtable, not yet in a segment.
    pub async fn insert(&self, batches: &[RecordBatch]) -> Result<u64> {
        let rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
        if rows == 0 {
            return Ok(0);
        }
        for batch in batches {
            if batch.schema().fields() != self.schema().fields() {
                return Err(Error::SchemaMismatch(format!(
                    "a batch has schema {:?}, the table expects {:?}",
                    batch.schema(),
                    self.schema()
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
            if batch.schema().fields() != self.schema().fields() {
                return Err(Error::SchemaMismatch(format!(
                    "a replacement batch has schema {:?}, the table expects {:?}",
                    batch.schema(),
                    self.schema()
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
                &self.schema(),
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
                None => crate::layout::batchcodec::encode(&RecordBatch::new_empty(self.schema())),
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

    pub(super) async fn flush_locked(&self, writer: &mut Writer) -> Result<u64> {
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

        // One segment per row group, not one per flush. A segment is the unit a
        // scan gives a partition, and the unit a zone map covers. One enormous
        // segment would leave a reader nothing to divide and nothing to prune.
        //
        // The size comes from what the table holds once this flush lands. A
        // small table gets small groups it can still divide. A large table gets
        // groups near the cap, not thousands of tiny ones.
        let total_rows = manifest.total_rows() + frozen.rows;
        let group_rows = self.inner.options.row_group_size_for(total_rows);
        let current = self.table_schema();
        for group in self.row_groups(frozen.batches, group_rows, &current)? {
            self.write_segment(&writer.file, &mut manifest, &group, &current, min_active)
                .await?;
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

    /// Build one segment from these batches and record it in the manifest.
    /// Cut batches into the row groups a flush or a compaction will write.
    ///
    /// With clustering off this keeps batches whole wherever they fit, so an
    /// unsliced batch goes straight to disk from Arrow's own buffers. With it
    /// on the rows have to be reordered, which copies them once; the groups
    /// come back already gathered, so the copy happens once rather than twice.
    pub(super) fn row_groups(
        &self,
        batches: Vec<RecordBatch>,
        group_rows: usize,
        current: &TableSchema,
    ) -> Result<Vec<Vec<RecordBatch>>> {
        if current.cluster_columns.is_empty() {
            return Ok(split_row_groups(batches, group_rows));
        }
        zorder::cluster(
            &batches,
            &current.schema,
            &current.cluster_columns,
            group_rows,
        )
    }

    pub(super) async fn write_segment(
        &self,
        file: &TableFile,
        manifest: &mut Manifest,
        batches: &[RecordBatch],
        current: &TableSchema,
        min_active_txn: u64,
    ) -> Result<u64> {
        let rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
        if rows == 0 {
            return Ok(0);
        }

        let segment_id = manifest.next_segment_id;
        let built = build_segment(
            segment_id,
            &current.schema,
            current.layout.current(),
            batches,
            &self.inner.options,
        )?;
        let data = file
            .write_allocated(manifest, &built.bytes, SEGMENT_ALIGN, min_active_txn)
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
        Ok(built.row_count)
    }

    /// True when the memtable or the log has grown past its limit.
    pub(super) fn should_flush(&self, writer: &Writer) -> bool {
        writer.memtable.bytes() >= self.inner.options.memtable_max_bytes
            || writer.wal.active_len() >= self.inner.options.wal_max_bytes
    }

    /// Publish a manifest and the snapshot readers will see next.
    pub(super) async fn commit(&self, writer: &mut Writer, manifest: Manifest) -> Result<()> {
        let min_active = self.inner.registry.min_active_txn();
        writer.file.commit(manifest, min_active).await?;
        self.publish(writer)
    }

    /// Swap in a snapshot of the writer's current state.
    pub(super) fn publish(&self, writer: &Writer) -> Result<()> {
        let snapshot = build_snapshot(writer, &self.table_schema())?;
        self.inner
            .current
            .store(self.inner.registry.publish(snapshot));
        Ok(())
    }
}

/// Put the log's records back, in order, skipping what a flush already folded
/// into the file.
pub(super) fn replay(writer: &mut Writer, checkpoint_lsn: Lsn) -> Result<()> {
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
pub(super) fn decode_batch(
    data: &crate::layout::batchcodec::BatchData,
    schema: &SchemaRef,
) -> Result<RecordBatch> {
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(data)?;
    let archived =
        rkyv::access::<crate::layout::batchcodec::ArchivedBatchData, rkyv::rancor::Error>(&bytes)?;
    crate::layout::batchcodec::decode(archived, schema, None)
}

/// Put logged segment deletions back into the writer's in-memory state.
pub(super) fn apply_logged_deletes(
    writer: &mut Writer,
    segments: Vec<SegmentDeletes>,
) -> Result<()> {
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
pub(super) struct PlannedDeletes {
    /// The new bitmap for each segment that changes.
    segments: Vec<(SegmentId, DeleteVector)>,
    /// Memtable rows that are not already deleted.
    memtable: Vec<u64>,
    /// Rows this removes that were not already gone.
    count: u64,
}

impl PlannedDeletes {
    pub(super) fn is_empty(&self) -> bool {
        self.segments.is_empty() && self.memtable.is_empty()
    }

    /// The segment bitmaps, in the shape a log record carries them.
    pub(super) fn logged_segments(&self) -> Result<Vec<SegmentDeletes>> {
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
pub(super) fn plan_deletes(
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
pub(super) fn apply_deletes(writer: &mut Writer, planned: PlannedDeletes) -> u64 {
    let mut deleted = planned.count;
    for (segment_id, dv) in planned.segments {
        writer.deletes.insert(segment_id, Arc::new(dv));
        writer.dirty_deletes.insert(segment_id);
    }
    deleted += writer.memtable.delete(planned.memtable);
    deleted
}

/// Group batches into row groups of at most `max_rows` rows each.
///
/// Whole batches are kept together where they fit, because a batch that is not
/// sliced is stored straight from Arrow's buffers with nothing copied. A single
/// batch larger than the limit is sliced, which costs a copy for that batch
/// alone.
///
/// A limit of zero means one group, however large.
pub(super) fn split_row_groups(
    batches: Vec<RecordBatch>,
    max_rows: usize,
) -> Vec<Vec<RecordBatch>> {
    if max_rows == 0 {
        return if batches.is_empty() {
            Vec::new()
        } else {
            vec![batches]
        };
    }

    let mut groups: Vec<Vec<RecordBatch>> = Vec::new();
    let mut current: Vec<RecordBatch> = Vec::new();
    let mut rows = 0usize;

    for batch in batches {
        let mut batch = batch;
        // Close the open group rather than overfill it, so whole batches stay
        // whole and only an oversized one gets sliced below.
        if rows > 0 && batch.num_rows() > max_rows - rows {
            groups.push(std::mem::take(&mut current));
            rows = 0;
        }
        while batch.num_rows() > max_rows {
            let head = batch.slice(0, max_rows);
            batch = batch.slice(max_rows, batch.num_rows() - max_rows);
            groups.push(vec![head]);
        }
        if batch.num_rows() == 0 {
            continue;
        }
        rows += batch.num_rows();
        current.push(batch);
        if rows >= max_rows {
            groups.push(std::mem::take(&mut current));
            rows = 0;
        }
    }

    if !current.is_empty() {
        groups.push(current);
    }
    groups
}
