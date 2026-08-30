//! Row order that makes zone maps selective on several columns at once.
//!
//! A zone map prunes a column well when the rows follow that column's order. A
//! segment then covers a narrow range of it. Only one column can have that.
//! Sort by `ts`, and a segment covers a minute of time and the whole range of
//! every other column.
//!
//! A z-order interleaves the bits of several columns into one sort key. A
//! segment then covers a compact box in all of them, not a narrow slice of one.
//! No column prunes as well as a single sort key gives it. Every column prunes
//! far better than nothing.
//!
//! This is a layout, not an index. It writes the same rows in a different
//! order and stores no extra bytes.
//!
//! Zone maps still come from the values written, so nothing here can make
//! pruning unsound. A poor key costs reads, never rows. That is why the key
//! below is free to approximate.

use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;

use crate::valuecodec;
use crate::{Error, Result};

/// Bytes each column contributes to a key.
///
/// Eight covers an integer, a timestamp, or the leading characters of a
/// string. Values that agree in their first eight bytes land together, which
/// is what clustering wants anyway.
const DIM_BYTES: usize = 8;
const DIM_BITS: usize = DIM_BYTES * 8;

/// The fixed-width form of one value.
///
/// Zero for a null, so nulls cluster at one end. A value whose encoding is all
/// zeros lands with them, which costs nothing: this decides an order, not an
/// answer.
fn dimension_bytes(
    array: &dyn Array,
    row: usize,
    scratch: &mut Vec<u8>,
) -> Result<[u8; DIM_BYTES]> {
    let mut out = [0u8; DIM_BYTES];
    if array.is_null(row) {
        return Ok(out);
    }

    scratch.clear();
    valuecodec::encode_value(scratch, array, row)?;
    // The first byte is the tag that keeps nulls ordered first. It is the same
    // for every non-null value, so it would spend a whole byte of every
    // dimension saying nothing. Skip it and keep eight bytes of the value.
    let value = scratch.get(1..).unwrap_or(&[]);
    let take = value.len().min(DIM_BYTES);
    out[..take].copy_from_slice(&value[..take]);
    Ok(out)
}

/// Interleave one row's dimensions into its key.
///
/// The most significant bit of every column comes first, then every column's
/// second bit, and so on. That is what makes the key cluster in all of them:
/// two rows share a long key prefix only when they agree in the leading bits
/// of every column, not just of one.
fn interleave_into(dimensions: &[[u8; DIM_BYTES]], key: &mut [u8]) {
    key.fill(0);
    let count = dimensions.len();
    for (dimension, bytes) in dimensions.iter().enumerate() {
        for bit in 0..DIM_BITS {
            if bytes[bit / 8] & (0x80 >> (bit % 8)) == 0 {
                continue;
            }
            // Column `dimension` owns every `count`-th bit of the output.
            let at = bit * count + dimension;
            key[at / 8] |= 0x80 >> (at % 8);
        }
    }
}

/// Check that a table can cluster by these columns, before it accepts writes.
///
/// The table runs this check when it opens, not when it flushes. A wrong name
/// is then an error the caller sees at once. A later flush would fail with rows
/// already accepted.
pub fn resolve(schema: &SchemaRef, names: &[String]) -> Result<Vec<usize>> {
    let mut indices = Vec::with_capacity(names.len());
    for name in names {
        let index = schema.index_of(name).map_err(|_| {
            Error::InvalidArgument(format!(
                "cannot cluster by {name}: the table has no such column"
            ))
        })?;
        let data_type = schema.field(index).data_type();
        if !valuecodec::is_encodable(data_type) {
            return Err(Error::InvalidArgument(format!(
                "cannot cluster by {name}: {data_type} has no byte form that orders"
            )));
        }
        if indices.contains(&index) {
            return Err(Error::InvalidArgument(format!(
                "cannot cluster by {name} twice"
            )));
        }
        indices.push(index);
    }
    Ok(indices)
}

/// Order rows by their z-order key and cut them into row groups.
///
/// This returns groups, not reordered batches. The reorder must copy either
/// way. To copy straight into the groups costs one pass instead of two.
pub fn cluster(
    batches: &[RecordBatch],
    schema: &SchemaRef,
    columns: &[usize],
    group_rows: usize,
) -> Result<Vec<Vec<RecordBatch>>> {
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    if columns.is_empty() || total == 0 {
        return Ok(Vec::new());
    }

    // Where each global row number lives, so a sorted key can name a row.
    let mut at: Vec<(usize, usize)> = Vec::with_capacity(total);
    for (index, batch) in batches.iter().enumerate() {
        at.extend((0..batch.num_rows()).map(|row| (index, row)));
    }

    let width = DIM_BYTES * columns.len();
    let mut keys = vec![0u8; width * total];
    let mut dimensions = vec![[0u8; DIM_BYTES]; columns.len()];
    let mut scratch = Vec::with_capacity(32);

    let mut row_at = 0usize;
    for batch in batches {
        for row in 0..batch.num_rows() {
            for (slot, &column) in dimensions.iter_mut().zip(columns) {
                *slot = dimension_bytes(batch.column(column).as_ref(), row, &mut scratch)?;
            }
            interleave_into(&dimensions, &mut keys[row_at * width..(row_at + 1) * width]);
            row_at += 1;
        }
    }

    let mut order: Vec<u32> = (0..total as u32).collect();
    order.sort_unstable_by(|&left, &right| {
        let left = left as usize * width;
        let right = right as usize * width;
        keys[left..left + width].cmp(&keys[right..right + width])
    });

    let per_group = if group_rows == 0 { total } else { group_rows };
    let mut groups = Vec::with_capacity(total.div_ceil(per_group));
    for chunk in order.chunks(per_group) {
        let picks: Vec<(usize, usize)> = chunk.iter().map(|&row| at[row as usize]).collect();
        groups.push(vec![gather(batches, schema, &picks)?]);
    }
    Ok(groups)
}

/// Build one batch from rows named across several batches.
fn gather(
    batches: &[RecordBatch],
    schema: &SchemaRef,
    picks: &[(usize, usize)],
) -> Result<RecordBatch> {
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for index in 0..schema.fields().len() {
        let arrays: Vec<&dyn Array> = batches.iter().map(|b| b.column(index).as_ref()).collect();
        columns.push(arrow_select::interleave::interleave(&arrays, picks)?);
    }
    Ok(RecordBatch::try_new(schema.clone(), columns)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, true),
            Field::new("y", DataType::Int64, true),
            Field::new("label", DataType::Utf8, true),
        ]))
    }

    fn batch(xs: Vec<i64>, ys: Vec<i64>) -> RecordBatch {
        let labels: Vec<String> = xs.iter().map(|x| format!("row-{x}")).collect();
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(xs)),
                Arc::new(Int64Array::from(ys)),
                Arc::new(StringArray::from(labels)),
            ],
        )
        .unwrap()
    }

    fn columns() -> Vec<usize> {
        resolve(&schema(), &["x".to_string(), "y".to_string()]).unwrap()
    }

    /// The measurement clustering exists for: how wide a range of each column a
    /// group ends up covering. Smaller is better, and a zone map prunes on it.
    fn widest_span(groups: &[Vec<RecordBatch>], column: usize) -> i64 {
        groups
            .iter()
            .map(|group| {
                let values = group[0].column(column);
                let values = values.as_any().downcast_ref::<Int64Array>().unwrap();
                let (mut low, mut high) = (i64::MAX, i64::MIN);
                for row in 0..values.len() {
                    low = low.min(values.value(row));
                    high = high.max(values.value(row));
                }
                high - low
            })
            .max()
            .unwrap_or(0)
    }

    /// A grid written row by row: `x` runs 0..16 within each `y`.
    fn grid() -> RecordBatch {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for y in 0..16 {
            for x in 0..16 {
                xs.push(x);
                ys.push(y);
            }
        }
        batch(xs, ys)
    }

    /// How many groups a zone map cannot rule out for `column = value`.
    fn survivors(groups: &[Vec<RecordBatch>], column: usize, value: i64) -> usize {
        groups
            .iter()
            .filter(|group| {
                let values = group[0].column(column);
                let values = values.as_any().downcast_ref::<Int64Array>().unwrap();
                (0..values.len()).any(|row| values.value(row) == value) || {
                    let (low, high) = (0..values.len())
                        .fold((i64::MAX, i64::MIN), |(lo, hi), row| {
                            (lo.min(values.value(row)), hi.max(values.value(row)))
                        });
                    (low..=high).contains(&value)
                }
            })
            .count()
    }

    /// The trade clustering makes, stated as what a zone map can prune.
    ///
    /// A 16 by 16 grid written row by row prunes `y` perfectly and `x` not at
    /// all, because every group holds every value of `x`. Clustered, neither
    /// column prunes perfectly and both prune well. That is the whole point:
    /// one sort key can only ever serve one column.
    #[test]
    fn clustering_trades_one_perfect_column_for_two_good_ones() {
        let rows = grid();
        let clustered = cluster(std::slice::from_ref(&rows), &schema(), &columns(), 32).unwrap();
        let row_major: Vec<Vec<RecordBatch>> = (0..rows.num_rows() / 32)
            .map(|group| vec![rows.slice(group * 32, 32)])
            .collect();

        assert_eq!(row_major.len(), 8);
        assert_eq!(clustered.len(), 8);

        // Row by row: x is hopeless, y is perfect.
        assert_eq!(survivors(&row_major, 0, 5), 8, "every group holds every x");
        assert_eq!(survivors(&row_major, 1, 5), 1, "one group holds y = 5");

        // Clustered: both columns rule out most groups.
        assert_eq!(survivors(&clustered, 0, 5), 2);
        assert_eq!(survivors(&clustered, 1, 5), 4);

        // Each group is a 4 by 8 box, which is exactly its 32 rows.
        assert_eq!(widest_span(&clustered, 0), 3);
        assert_eq!(widest_span(&clustered, 1), 7);
    }

    #[test]
    fn every_row_survives_the_reorder() {
        let rows = grid();
        let groups = cluster(std::slice::from_ref(&rows), &schema(), &columns(), 32).unwrap();

        let total: usize = groups.iter().flatten().map(|b| b.num_rows()).sum();
        assert_eq!(total, rows.num_rows());

        let mut seen: Vec<(i64, i64)> = groups
            .iter()
            .flatten()
            .flat_map(|batch| {
                let xs = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let ys = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                (0..batch.num_rows())
                    .map(|row| (xs.value(row), ys.value(row)))
                    .collect::<Vec<_>>()
            })
            .collect();
        seen.sort_unstable();

        let mut expected: Vec<(i64, i64)> =
            (0..16).flat_map(|y| (0..16).map(move |x| (x, y))).collect();
        expected.sort_unstable();
        assert_eq!(seen, expected);
    }

    #[test]
    fn a_row_keeps_its_other_columns() {
        let groups = cluster(&[grid()], &schema(), &columns(), 32).unwrap();
        for batch in groups.iter().flatten() {
            let xs = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let labels = batch
                .column(2)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for row in 0..batch.num_rows() {
                assert_eq!(labels.value(row), format!("row-{}", xs.value(row)));
            }
        }
    }

    #[test]
    fn rows_are_gathered_across_batches() {
        let groups = cluster(
            &[batch(vec![3, 1], vec![3, 1]), batch(vec![2, 0], vec![2, 0])],
            &schema(),
            &columns(),
            4,
        )
        .unwrap();
        assert_eq!(groups.len(), 1);
        let xs = groups[0][0].column(0);
        let xs = xs.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(xs.values(), &[0, 1, 2, 3], "the key orders the diagonal");
    }

    #[test]
    fn nulls_cluster_at_one_end() {
        let batch = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(vec![Some(5), None, Some(9), None])),
                Arc::new(Int64Array::from(vec![Some(5), None, Some(9), None])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
            ],
        )
        .unwrap();
        let groups = cluster(&[batch], &schema(), &columns(), 4).unwrap();
        let xs = groups[0][0].column(0);
        let xs = xs.as_any().downcast_ref::<Int64Array>().unwrap();
        assert!(xs.is_null(0) && xs.is_null(1), "nulls sort first");
    }

    #[test]
    fn no_rows_makes_no_groups() {
        assert!(cluster(&[], &schema(), &columns(), 32).unwrap().is_empty());
    }

    #[test]
    fn a_group_size_of_zero_makes_one_group() {
        let groups = cluster(&[grid()], &schema(), &columns(), 0).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0][0].num_rows(), 256);
    }

    #[test]
    fn a_column_that_is_not_there_is_refused() {
        let err = resolve(&schema(), &["absent".to_string()]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "got {err:?}");
    }

    #[test]
    fn a_column_with_no_order_is_refused() {
        let nested = Arc::new(Schema::new(vec![Field::new(
            "items",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        )]));
        let err = resolve(&nested, &["items".to_string()]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "got {err:?}");
    }

    #[test]
    fn the_same_column_twice_is_refused() {
        let err = resolve(&schema(), &["x".to_string(), "x".to_string()]).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "got {err:?}");
    }

    #[test]
    fn one_column_still_works_and_simply_sorts() {
        let columns = resolve(&schema(), &["x".to_string()]).unwrap();
        let groups = cluster(
            &[batch(vec![3, 1, 2], vec![0, 0, 0])],
            &schema(),
            &columns,
            3,
        )
        .unwrap();
        let xs = groups[0][0].column(0);
        let xs = xs.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(xs.values(), &[1, 2, 3]);
    }
}
