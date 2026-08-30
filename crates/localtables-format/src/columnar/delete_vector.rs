//! Which rows of a segment have been deleted.
//!
//! A segment never changes once written. So a delete records a row position
//! and rewrites no data.
//!
//! The positions live in a roaring bitmap. That costs a few bytes for a
//! scattered handful of deletes, and a few bytes again for a whole segment.
//!
//! A scan applies the bitmap as a mask. Compaction eventually rewrites the
//! segment without the deleted rows and drops the bitmap.

use roaring::RoaringBitmap;

use crate::layout::frame::{self, tag};
use crate::{Error, Result};

/// Deleted row positions inside one segment.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeleteVector {
    rows: RoaringBitmap,
}

impl DeleteVector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_bitmap(rows: RoaringBitmap) -> Self {
        Self { rows }
    }

    /// Mark row `position` deleted. Returns false when it already was.
    pub fn delete(&mut self, position: u32) -> bool {
        self.rows.insert(position)
    }

    /// Mark every row in `positions` deleted. Returns how many were new.
    pub fn delete_all(&mut self, positions: impl IntoIterator<Item = u32>) -> u64 {
        positions
            .into_iter()
            .filter(|p| self.rows.insert(*p))
            .count() as u64
    }

    /// Mark every row of a `row_count`-row segment deleted.
    pub fn delete_range(&mut self, row_count: u64) -> u64 {
        let before = self.rows.len();
        self.rows.insert_range(0..row_count as u32);
        self.rows.len() - before
    }

    pub fn is_deleted(&self, position: u32) -> bool {
        self.rows.contains(position)
    }

    pub fn len(&self) -> u64 {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn bitmap(&self) -> &RoaringBitmap {
        &self.rows
    }

    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.rows.iter()
    }

    /// Add every deletion from `other`. Returns how many rows this newly
    /// deleted, so a caller can keep a running count without a second pass.
    pub fn union(&mut self, other: &DeleteVector) -> u64 {
        let before = self.rows.len();
        self.rows |= &other.rows;
        self.rows.len() - before
    }

    /// True when every row of a `row_count`-row segment is deleted.
    pub fn covers_all(&self, row_count: u64) -> bool {
        self.rows.len() >= row_count
    }

    /// A mask with `true` for the rows a scan should keep.
    ///
    /// Arrow's filter kernel takes the mask directly, so a segment with no
    /// deletions never builds one.
    pub fn keep_mask(&self, row_count: usize) -> arrow_array::BooleanArray {
        let mut builder = arrow_buffer::BooleanBufferBuilder::new(row_count);
        builder.append_n(row_count, true);
        for position in self.rows.iter() {
            let position = position as usize;
            if position < row_count {
                builder.set_bit(position, false);
            }
        }
        arrow_array::BooleanArray::new(builder.finish(), None)
    }

    /// The bitmap on its own, with no frame around it.
    ///
    /// Used inside a log record, which already carries its own frame.
    pub fn to_bitmap_bytes(&self) -> Result<Vec<u8>> {
        let mut payload = Vec::with_capacity(self.rows.serialized_size());
        self.rows
            .serialize_into(&mut payload)
            .map_err(|e| Error::corrupt(format!("delete vector failed to serialize: {e}")))?;
        Ok(payload)
    }

    /// Read back bytes written by [`DeleteVector::to_bitmap_bytes`].
    pub fn from_bitmap_bytes(bytes: &[u8]) -> Result<Self> {
        let rows = RoaringBitmap::deserialize_from(bytes)
            .map_err(|e| Error::corrupt(format!("delete vector is malformed: {e}")))?;
        Ok(Self { rows })
    }

    /// Serialize into a frame ready to be written to the file.
    pub fn to_frame(&self) -> Result<Vec<u8>> {
        Ok(frame::encode(tag::DELETES, &self.to_bitmap_bytes()?))
    }

    /// Read back a frame written by [`DeleteVector::to_frame`].
    pub fn from_frame(bytes: &[u8]) -> Result<Self> {
        let payload = frame::decode(bytes, tag::DELETES, "delete vector")?;
        Self::from_bitmap_bytes(payload)
    }
}

impl FromIterator<u32> for DeleteVector {
    fn from_iter<I: IntoIterator<Item = u32>>(iter: I) -> Self {
        Self {
            rows: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;

    #[test]
    fn deleting_the_same_row_twice_counts_once() {
        let mut dv = DeleteVector::new();
        assert!(dv.delete(5));
        assert!(!dv.delete(5), "a repeated delete is not a new one");
        assert_eq!(dv.len(), 1);
        assert!(dv.is_deleted(5));
        assert!(!dv.is_deleted(6));
    }

    #[test]
    fn delete_all_reports_only_the_new_rows() {
        let mut dv = DeleteVector::from_iter([1u32, 3]);
        assert_eq!(dv.delete_all([3u32, 4, 5]), 2);
        assert_eq!(dv.len(), 4);
    }

    #[test]
    fn a_whole_segment_can_be_deleted_at_once() {
        let mut dv = DeleteVector::new();
        assert_eq!(dv.delete_range(1000), 1000);
        assert!(dv.covers_all(1000));
        assert!(!dv.covers_all(1001));
    }

    #[test]
    fn union_reports_what_it_added() {
        let mut a = DeleteVector::from_iter([1u32, 2, 3]);
        let b = DeleteVector::from_iter([3u32, 4]);
        assert_eq!(a.union(&b), 1);
        assert_eq!(a.len(), 4);
    }

    #[test]
    fn the_mask_keeps_everything_that_was_not_deleted() {
        let dv = DeleteVector::from_iter([0u32, 3, 4]);
        let mask = dv.keep_mask(6);

        assert_eq!(mask.len(), 6);
        assert_eq!(mask.null_count(), 0, "a mask must never carry nulls");
        let kept: Vec<bool> = (0..6).map(|i| mask.value(i)).collect();
        assert_eq!(kept, vec![false, true, true, false, false, true]);
    }

    #[test]
    fn positions_past_the_row_count_do_not_disturb_the_mask() {
        // A stale bitmap must not panic or shorten the mask.
        let dv = DeleteVector::from_iter([1u32, 99]);
        let mask = dv.keep_mask(3);
        assert_eq!(mask.len(), 3);
        assert!(!mask.value(1));
    }

    #[test]
    fn an_empty_vector_keeps_every_row() {
        let dv = DeleteVector::new();
        assert!(dv.is_empty());
        let mask = dv.keep_mask(4);
        assert!((0..4).all(|i| mask.value(i)));
    }

    #[test]
    fn frames_round_trip() {
        for dv in [
            DeleteVector::new(),
            DeleteVector::from_iter([0u32]),
            DeleteVector::from_iter([7u32, 9, 100_000, u32::MAX]),
            {
                let mut dense = DeleteVector::new();
                dense.delete_range(100_000);
                dense
            },
        ] {
            let frame = dv.to_frame().unwrap();
            assert_eq!(DeleteVector::from_frame(&frame).unwrap(), dv);
        }
    }

    #[test]
    fn a_dense_vector_stays_small() {
        let mut dv = DeleteVector::new();
        dv.delete_range(1_000_000);
        let frame = dv.to_frame().unwrap();
        assert!(
            frame.len() < 1024,
            "a million contiguous deletes took {} bytes",
            frame.len()
        );
    }

    #[test]
    fn a_damaged_frame_is_rejected() {
        let dv = DeleteVector::from_iter([1u32, 2, 3]);
        let mut frame = dv.to_frame().unwrap();
        let last = frame.len() - 1;
        frame[last] ^= 0xff;

        let err = DeleteVector::from_frame(&frame).unwrap_err();
        assert!(matches!(err, Error::Checksum { .. }), "got {err:?}");
    }
}
