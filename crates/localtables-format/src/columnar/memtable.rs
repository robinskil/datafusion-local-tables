//! Rows that are durable in the log but not yet in a segment.
//!
//! The memtable is Arrow-native. Rows arrive as batches and leave as a segment,
//! so holding them as batches means no conversion in either direction. Deleting
//! a memtable row records a tombstone rather than rebuilding the batch, exactly
//! as a segment records a delete rather than rewriting itself.
//!
//! Rows are addressed by sequence number, a counter that keeps rising across
//! flushes. A delete can therefore name a row logged before a crash and still
//! find it after replay.

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use roaring::RoaringTreemap;

use crate::Result;

/// Coalesce small batches once there are more than this many.
///
/// Scanning is one pass per batch, so a memtable full of single-row batches
/// would make reads slower and slower. Merging them keeps that flat.
const COALESCE_THRESHOLD: usize = 256;

/// Rows held in memory, with the sequence number of their first row.
#[derive(Debug, Clone)]
struct Chunk {
    base_seqno: u64,
    batch: RecordBatch,
}

impl Chunk {
    fn contains(&self, seqno: u64) -> bool {
        seqno >= self.base_seqno && seqno < self.base_seqno + self.batch.num_rows() as u64
    }
}

/// Rows written but not yet flushed to a segment.
#[derive(Debug)]
pub struct Memtable {
    schema: SchemaRef,
    chunks: Vec<Chunk>,
    /// Deleted rows, by sequence number.
    tombstones: RoaringTreemap,
    /// The sequence number the next inserted row will get.
    next_seqno: u64,
    bytes: usize,
    rows: u64,
}

impl Memtable {
    pub fn new(schema: SchemaRef, next_seqno: u64) -> Self {
        Self {
            schema,
            chunks: Vec::new(),
            tombstones: RoaringTreemap::new(),
            next_seqno,
            bytes: 0,
            rows: 0,
        }
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// The sequence number the next inserted row will get.
    pub fn next_seqno(&self) -> u64 {
        self.next_seqno
    }

    /// Rows held, deleted ones included.
    pub fn total_rows(&self) -> u64 {
        self.rows
    }

    /// Rows a scan would return.
    pub fn live_rows(&self) -> u64 {
        self.rows - self.tombstones.len()
    }

    /// Roughly how much memory the held batches occupy.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.live_rows() == 0
    }

    /// Add a batch, and return the sequence number of its first row.
    pub fn insert(&mut self, batch: RecordBatch) -> u64 {
        let rows = batch.num_rows() as u64;
        if rows == 0 {
            return self.next_seqno;
        }
        let base_seqno = self.next_seqno;

        self.bytes += batch.get_array_memory_size();
        self.rows += rows;
        self.next_seqno += rows;
        self.chunks.push(Chunk { base_seqno, batch });

        if self.chunks.len() > COALESCE_THRESHOLD {
            self.coalesce();
        }
        base_seqno
    }

    /// Insert a batch that was read back from the log, at its original
    /// sequence numbers.
    ///
    /// Replay has to preserve them, because a logged delete names rows by the
    /// sequence number they had when they were written.
    pub fn insert_at(&mut self, base_seqno: u64, batch: RecordBatch) {
        let rows = batch.num_rows() as u64;
        if rows == 0 {
            return;
        }
        self.bytes += batch.get_array_memory_size();
        self.rows += rows;
        self.next_seqno = self.next_seqno.max(base_seqno + rows);
        self.chunks.push(Chunk { base_seqno, batch });
    }

    /// Mark rows deleted. Returns how many were not already deleted.
    pub fn delete(&mut self, seqnos: impl IntoIterator<Item = u64>) -> u64 {
        seqnos
            .into_iter()
            .filter(|seqno| self.holds(*seqno) && self.tombstones.insert(*seqno))
            .count() as u64
    }

    /// True when some chunk covers this sequence number.
    fn holds(&self, seqno: u64) -> bool {
        self.chunks.iter().any(|chunk| chunk.contains(seqno))
    }

    pub fn is_deleted(&self, seqno: u64) -> bool {
        self.tombstones.contains(seqno)
    }

    /// Sequence numbers of the rows a scan would return, in insertion order.
    pub fn live_seqnos(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.live_rows() as usize);
        for chunk in &self.chunks {
            for row in 0..chunk.batch.num_rows() as u64 {
                let seqno = chunk.base_seqno + row;
                if !self.tombstones.contains(seqno) {
                    out.push(seqno);
                }
            }
        }
        out
    }

    /// The rows a scan should return, with deleted ones removed.
    ///
    /// A chunk with nothing deleted is handed back untouched, so the ordinary
    /// case costs no filtering and no copying.
    pub fn batches(&self, projection: Option<&[usize]>) -> Result<Vec<RecordBatch>> {
        let mut out = Vec::with_capacity(self.chunks.len());
        for chunk in &self.chunks {
            let rows = chunk.batch.num_rows();
            let end = chunk.base_seqno + rows as u64;
            let deleted = self.tombstones.range_cardinality(chunk.base_seqno..end) as usize;

            let batch = if deleted == 0 {
                chunk.batch.clone()
            } else if deleted == rows {
                continue;
            } else {
                let mut builder = arrow_buffer::BooleanBufferBuilder::new(rows);
                builder.append_n(rows, true);
                for row in 0..rows {
                    if self.tombstones.contains(chunk.base_seqno + row as u64) {
                        builder.set_bit(row, false);
                    }
                }
                let mask = arrow_array::BooleanArray::new(builder.finish(), None);
                arrow_select::filter::filter_record_batch(&chunk.batch, &mask)?
            };

            out.push(match projection {
                Some(indices) => batch.project(indices)?,
                None => batch,
            });
        }
        Ok(out)
    }

    /// Merge the held batches into one, dropping deleted rows.
    ///
    /// This is what a flush writes as a segment.
    pub fn coalesce(&mut self) {
        let Ok(batches) = self.batches(None) else {
            // Filtering cannot fail for a mask this code builds itself, but if
            // it somehow did, leaving the memtable as it stands is correct.
            return;
        };
        let merged = match batches.len() {
            0 => {
                self.chunks.clear();
                self.tombstones = RoaringTreemap::new();
                self.bytes = 0;
                self.rows = 0;
                return;
            }
            1 => batches.into_iter().next().expect("length checked"),
            _ => match arrow_select::concat::concat_batches(&self.schema, &batches) {
                Ok(merged) => merged,
                Err(_) => return,
            },
        };

        // Merging renumbers the rows, which is safe only because coalescing
        // drops the deleted ones: nothing outside can still be holding a
        // sequence number for a row that survives.
        let rows = merged.num_rows() as u64;
        let base_seqno = self.next_seqno;
        self.next_seqno += rows;
        self.bytes = merged.get_array_memory_size();
        self.rows = rows;
        self.tombstones = RoaringTreemap::new();
        self.chunks = vec![Chunk {
            base_seqno,
            batch: merged,
        }];
    }

    /// Take everything out, leaving an empty memtable that keeps counting from
    /// where this one stopped.
    ///
    /// A flush freezes the memtable this way and writes the result as a
    /// segment while new writes carry on into the replacement.
    pub fn freeze(&mut self) -> Result<FrozenMemtable> {
        let batches = self.batches(None)?;
        let rows = self.live_rows();
        let next_seqno = self.next_seqno;

        self.chunks.clear();
        self.tombstones = RoaringTreemap::new();
        self.bytes = 0;
        self.rows = 0;

        Ok(FrozenMemtable {
            batches,
            rows,
            next_seqno,
        })
    }
}

/// A memtable's contents, taken out and ready to become a segment.
#[derive(Debug)]
pub struct FrozenMemtable {
    pub batches: Vec<RecordBatch>,
    pub rows: u64,
    /// The sequence number the memtable had reached when it was frozen.
    pub next_seqno: u64,
}

impl FrozenMemtable {
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]))
    }

    fn batch(values: &[i32]) -> RecordBatch {
        RecordBatch::try_new(schema(), vec![Arc::new(Int32Array::from(values.to_vec()))]).unwrap()
    }

    fn values(batches: &[RecordBatch]) -> Vec<i32> {
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

    fn memtable() -> Memtable {
        Memtable::new(schema(), 0)
    }

    #[test]
    fn a_new_memtable_holds_nothing() {
        let table = memtable();
        assert!(table.is_empty());
        assert_eq!(table.live_rows(), 0);
        assert_eq!(table.next_seqno(), 0);
        assert!(table.batches(None).unwrap().is_empty());
    }

    #[test]
    fn sequence_numbers_run_on_across_batches() {
        let mut table = memtable();
        assert_eq!(table.insert(batch(&[1, 2, 3])), 0);
        assert_eq!(table.insert(batch(&[4, 5])), 3);
        assert_eq!(table.next_seqno(), 5);
        assert_eq!(table.live_rows(), 5);
    }

    #[test]
    fn an_empty_batch_takes_no_sequence_numbers() {
        let mut table = memtable();
        table.insert(batch(&[1]));
        assert_eq!(table.insert(batch(&[])), 1);
        assert_eq!(table.next_seqno(), 1);
    }

    #[test]
    fn reading_gives_back_what_was_inserted() {
        let mut table = memtable();
        table.insert(batch(&[1, 2]));
        table.insert(batch(&[3]));
        assert_eq!(values(&table.batches(None).unwrap()), vec![1, 2, 3]);
    }

    #[test]
    fn deleted_rows_disappear_from_reads() {
        let mut table = memtable();
        table.insert(batch(&[10, 11, 12]));
        table.insert(batch(&[13, 14]));

        assert_eq!(table.delete([1, 4]), 2);
        assert_eq!(table.live_rows(), 3);
        assert_eq!(values(&table.batches(None).unwrap()), vec![10, 12, 13]);
    }

    #[test]
    fn deleting_the_same_row_twice_counts_once() {
        let mut table = memtable();
        table.insert(batch(&[1, 2, 3]));
        assert_eq!(table.delete([1]), 1);
        assert_eq!(table.delete([1]), 0);
        assert_eq!(table.live_rows(), 2);
    }

    #[test]
    fn deleting_a_sequence_number_no_row_has_changes_nothing() {
        let mut table = memtable();
        table.insert(batch(&[1, 2]));
        assert_eq!(table.delete([99]), 0);
        assert_eq!(table.live_rows(), 2);
    }

    #[test]
    fn a_fully_deleted_batch_is_skipped_rather_than_filtered() {
        let mut table = memtable();
        table.insert(batch(&[1, 2]));
        table.insert(batch(&[3, 4]));
        table.delete([0, 1]);

        let batches = table.batches(None).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(values(&batches), vec![3, 4]);
    }

    #[test]
    fn an_untouched_batch_is_handed_back_as_it_stands() {
        let mut table = memtable();
        let original = batch(&[1, 2, 3]);
        table.insert(original.clone());

        let batches = table.batches(None).unwrap();
        assert_eq!(
            batches[0].column(0).as_ref() as *const _,
            original.column(0).as_ref() as *const _,
            "a batch with nothing deleted must not be rebuilt"
        );
    }

    #[test]
    fn projection_narrows_what_is_returned() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let mut table = Memtable::new(schema.clone(), 0);
        table.insert(
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int32Array::from(vec![1, 2])),
                    Arc::new(Int32Array::from(vec![10, 20])),
                ],
            )
            .unwrap(),
        );

        let batches = table.batches(Some(&[1])).unwrap();
        assert_eq!(batches[0].num_columns(), 1);
        assert_eq!(values(&batches), vec![10, 20]);
    }

    #[test]
    fn live_sequence_numbers_skip_the_deleted_ones() {
        let mut table = memtable();
        table.insert(batch(&[1, 2, 3]));
        table.delete([1]);
        assert_eq!(table.live_seqnos(), vec![0, 2]);
    }

    #[test]
    fn replayed_rows_keep_the_sequence_numbers_they_were_given() {
        let mut table = Memtable::new(schema(), 0);
        table.insert_at(100, batch(&[1, 2]));
        table.insert_at(102, batch(&[3]));

        assert_eq!(table.next_seqno(), 103);
        // A delete logged before the crash still finds its row.
        assert_eq!(table.delete([101]), 1);
        assert_eq!(values(&table.batches(None).unwrap()), vec![1, 3]);
    }

    #[test]
    fn coalescing_merges_batches_and_drops_deleted_rows() {
        let mut table = memtable();
        for i in 0..10 {
            table.insert(batch(&[i]));
        }
        table.delete([0, 9]);

        table.coalesce();
        let batches = table.batches(None).unwrap();
        assert_eq!(batches.len(), 1, "coalescing must leave one batch");
        assert_eq!(values(&batches), vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(table.live_rows(), 8);
    }

    #[test]
    fn coalescing_an_entirely_deleted_memtable_empties_it() {
        let mut table = memtable();
        table.insert(batch(&[1, 2]));
        table.delete([0, 1]);

        table.coalesce();
        assert!(table.is_empty());
        assert!(table.batches(None).unwrap().is_empty());
    }

    #[test]
    fn many_small_inserts_do_not_pile_up_batches() {
        let mut table = memtable();
        for i in 0..(COALESCE_THRESHOLD as i32 * 3) {
            table.insert(batch(&[i]));
        }
        assert!(
            table.batches(None).unwrap().len() <= COALESCE_THRESHOLD,
            "small batches must be merged, or scanning gets slower with every insert"
        );
        assert_eq!(table.live_rows(), COALESCE_THRESHOLD as u64 * 3);
    }

    #[test]
    fn freezing_takes_the_rows_and_leaves_an_empty_table() {
        let mut table = memtable();
        table.insert(batch(&[1, 2, 3]));
        table.insert(batch(&[4]));
        table.delete([2]);

        let frozen = table.freeze().unwrap();
        assert_eq!(frozen.rows, 3);
        assert_eq!(values(&frozen.batches), vec![1, 2, 4]);
        assert_eq!(frozen.next_seqno, 4);

        assert!(table.is_empty());
        assert_eq!(
            table.next_seqno(),
            4,
            "numbering must carry on, so a logged delete cannot hit a reused number"
        );
    }

    #[test]
    fn rows_added_after_a_freeze_get_fresh_numbers() {
        let mut table = memtable();
        table.insert(batch(&[1, 2]));
        table.freeze().unwrap();

        assert_eq!(table.insert(batch(&[3])), 2);
        assert_eq!(values(&table.batches(None).unwrap()), vec![3]);
    }

    #[test]
    fn memory_use_tracks_what_is_held() {
        let mut table = memtable();
        assert_eq!(table.bytes(), 0);
        table.insert(batch(&[1, 2, 3]));
        assert!(table.bytes() > 0);
        table.freeze().unwrap();
        assert_eq!(table.bytes(), 0);
    }
}
