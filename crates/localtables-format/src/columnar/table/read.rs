//! Rows out: open a segment, decode the rows a scan asks for, hand them back.
//!
//! A read takes a snapshot and never blocks a writer. It decodes only the
//! pages a predicate leaves and cuts them at block boundaries.

use super::*;

impl ColumnarTable {
    /// Open a segment for reading, keeping its bytes alive through the reader.
    pub async fn segment_reader(&self, entry: &SegmentEntry) -> Result<SegmentReader> {
        self.segment_reader_as(entry, &self.table_schema()).await
    }

    /// Open a segment under a schema that is not necessarily the current one.
    ///
    /// A rewrite reads under the schema the segments carry. It writes under the
    /// schema that replaces it, in the same pass. So neither schema can come
    /// from the handle.
    pub(super) async fn segment_reader_as(
        &self,
        entry: &SegmentEntry,
        schema: &TableSchema,
    ) -> Result<SegmentReader> {
        let bytes = self.inner.io.read_immutable(entry.data).await?;
        SegmentReader::new(
            bytes,
            entry.data.offset,
            entry.meta,
            schema.schema.clone(),
            &schema.layout,
        )
    }

    /// Read one segment as batches, with deleted rows removed.
    pub async fn read_segment(
        &self,
        snapshot: &Snapshot,
        entry: &SegmentEntry,
        projection: Option<&[usize]>,
    ) -> Result<Vec<RecordBatch>> {
        self.read_segment_as(snapshot, entry, projection, &self.table_schema(), None)
            .await
    }

    /// Read a segment, keeping only the pages a caller still wants.
    ///
    /// `keep_pages` names the pages of the segment worth handing on, as decided
    /// by a predicate against the segment's page bounds. `None` keeps them all.
    /// A page nobody keeps costs the filter above nothing, and on a cold file
    /// its bytes are never faulted in.
    pub async fn read_segment_pages(
        &self,
        snapshot: &Snapshot,
        entry: &SegmentEntry,
        projection: Option<&[usize]>,
        keep_pages: Option<&[bool]>,
    ) -> Result<Vec<RecordBatch>> {
        self.read_segment_as(
            snapshot,
            entry,
            projection,
            &self.table_schema(),
            keep_pages,
        )
        .await
    }

    pub(super) async fn read_segment_as(
        &self,
        snapshot: &Snapshot,
        entry: &SegmentEntry,
        projection: Option<&[usize]>,
        schema: &TableSchema,
        keep_pages: Option<&[bool]>,
    ) -> Result<Vec<RecordBatch>> {
        let reader = self.segment_reader_as(entry, schema).await?;
        let rows = reader.row_count()? as usize;

        // The mask covers the segment's own row positions, so it has to be
        // applied to a range before that range is compacted; filtering first
        // would shift every row and leave the page boundaries meaning nothing.
        let mask = match snapshot.deletes_for(entry.segment_id) {
            None => None,
            Some(dv) if dv.is_empty() => None,
            Some(dv) => Some(dv.keep_mask(rows)),
        };

        let ranges = match keep_pages {
            None => vec![(0, rows)],
            Some(keep) => kept_ranges(&reader, keep)?,
        };
        // Cut at block boundaries, so decoding a range never has to join two
        // blocks together. Each piece then comes out of exactly one block, and
        // an uncompressed one comes out without a copy.
        let ranges = split_at_blocks(ranges, reader.block_rows()? as usize);

        let mut out = Vec::new();
        for (start, len) in ranges {
            // Decoded a range at a time, so a column stored as a dictionary or
            // as runs is expanded only for the rows being handed on.
            let page = reader.read_rows(projection, start, len)?;
            let page = match &mask {
                None => page,
                Some(mask) => {
                    arrow_select::filter::filter_record_batch(&page, &mask.slice(start, len))?
                }
            };
            if page.num_rows() == 0 {
                continue;
            }
            out.extend(slice_batches(page, self.inner.options.scan_batch_rows));
        }
        Ok(out)
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
}

/// The row ranges a page selection asks for, with neighbours joined.
///
/// Pages that sit next to each other become one range. A scan that keeps most
/// of a segment then decodes it in a few passes, not one per page. A scan that
/// keeps all of it decodes it in one.
pub(super) fn kept_ranges(reader: &SegmentReader, keep: &[bool]) -> Result<Vec<(usize, usize)>> {
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for (page, wanted) in keep.iter().enumerate() {
        if !wanted {
            continue;
        }
        let Some((start, len)) = reader.page_range(page)? else {
            continue;
        };
        match ranges.last_mut() {
            Some((at, taken)) if *at + *taken == start => *taken += len,
            _ => ranges.push((start, len)),
        }
    }
    Ok(ranges)
}

/// Cut ranges so none of them spans two blocks.
///
/// Decoding a range that spans blocks means joining the results, which copies.
/// Handing back one piece per block avoids that: the scan returns several
/// batches instead of one, which it was going to do anyway.
pub(super) fn split_at_blocks(
    ranges: Vec<(usize, usize)>,
    block_rows: usize,
) -> Vec<(usize, usize)> {
    if block_rows == 0 {
        return ranges;
    }
    let mut out = Vec::with_capacity(ranges.len());
    for (start, len) in ranges {
        let mut at = start;
        let end = start + len;
        while at < end {
            // The end of the block `at` falls in, or the end of the range.
            let boundary = (at / block_rows + 1) * block_rows;
            let piece = end.min(boundary) - at;
            out.push((at, piece));
            at += piece;
        }
    }
    out
}

/// Cut a batch into pieces of at most `rows` rows.
pub(super) fn slice_batches(batch: RecordBatch, rows: usize) -> Vec<RecordBatch> {
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
