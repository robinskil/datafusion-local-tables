//! Zone maps, in the shape DataFusion's pruning wants them.
//!
//! Each segment is one "container". DataFusion asks for the minimum, maximum,
//! null count and row count of a column across all containers, works out which
//! ones a predicate cannot match, and the scan skips those without reading a
//! byte of their data.
//!
//! The interval reasoning is DataFusion's own. This file only reports what the
//! zone maps know, and reports nothing where they do not know: a null entry
//! means "no information", which costs a segment read, while a wrong entry
//! would lose rows.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, UInt64Array};
use arrow::datatypes::{DataType, SchemaRef};
use datafusion::common::pruning::PruningStatistics;
use datafusion::common::{Column, ScalarValue};

use localtables_format::columnar::segment::SegmentReader;
use localtables_format::columnar::zonemap::ZoneMap;

/// One segment's bounds for every column, read once when the scan is planned.
#[derive(Debug, Clone)]
pub struct SegmentZoneMaps {
    /// One entry per column of the table schema.
    pub columns: Vec<ZoneMap>,
    pub row_count: u64,
}

impl SegmentZoneMaps {
    /// Read the zone maps out of a segment's metadata.
    pub fn from_reader(reader: &SegmentReader) -> localtables_format::Result<Self> {
        let meta = reader.meta()?;
        Ok(Self {
            columns: meta.columns.iter().map(|c| c.zone.to_native()).collect(),
            row_count: meta.row_count.to_native(),
        })
    }
}

/// Bounds across the segments a scan is considering.
#[derive(Debug)]
pub struct SegmentStatistics {
    schema: SchemaRef,
    segments: Vec<SegmentZoneMaps>,
}

impl SegmentStatistics {
    pub fn new(schema: SchemaRef, segments: Vec<SegmentZoneMaps>) -> Self {
        Self { schema, segments }
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Position and type of a column, or `None` when the table has no such
    /// column. A predicate over an unknown column prunes nothing.
    fn column(&self, column: &Column) -> Option<(usize, &DataType)> {
        let index = self.schema.index_of(&column.name).ok()?;
        Some((index, self.schema.field(index).data_type()))
    }

    /// Gather one bound per segment into a single array.
    ///
    /// Segments that do not know their bound contribute a null, which
    /// DataFusion reads as "could match". Returning `None` when no segment
    /// knows saves it the work of asking.
    fn bounds(
        &self,
        column: &Column,
        pick: impl Fn(&ZoneMap, &DataType) -> Option<ArrayRef>,
    ) -> Option<ArrayRef> {
        let (index, data_type) = self.column(column)?;

        let mut any_known = false;
        let mut parts: Vec<ArrayRef> = Vec::with_capacity(self.segments.len());
        for segment in &self.segments {
            let bound = segment
                .columns
                .get(index)
                .and_then(|zone| pick(zone, data_type));
            match bound {
                Some(array) => {
                    any_known = true;
                    parts.push(array);
                }
                None => parts.push(arrow::array::new_null_array(data_type, 1)),
            }
        }

        if !any_known {
            return None;
        }
        let refs: Vec<&dyn arrow::array::Array> = parts.iter().map(|a| a.as_ref()).collect();
        arrow::compute::concat(&refs).ok()
    }
}

impl PruningStatistics for SegmentStatistics {
    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bounds(column, |zone, data_type| zone.min_array(data_type))
    }

    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bounds(column, |zone, data_type| zone.max_array(data_type))
    }

    fn num_containers(&self) -> usize {
        self.segments.len()
    }

    fn null_counts(&self, column: &Column) -> Option<ArrayRef> {
        let (index, _) = self.column(column)?;
        let counts: Vec<Option<u64>> = self
            .segments
            .iter()
            .map(|segment| segment.columns.get(index).map(|zone| zone.null_count))
            .collect();
        Some(Arc::new(UInt64Array::from(counts)))
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        Some(Arc::new(UInt64Array::from(
            self.segments
                .iter()
                .map(|segment| segment.row_count)
                .collect::<Vec<u64>>(),
        )))
    }

    /// Whether a segment holds only values from a given set.
    ///
    /// Zone maps record a range, not a membership, so this never claims to
    /// know. Answering it would need per-segment dictionaries or filters,
    /// which the format does not keep.
    fn contained(&self, _column: &Column, _values: &HashSet<ScalarValue>) -> Option<BooleanArray> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{Field, Schema};

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    /// Zone maps for one segment, built from the values it would hold.
    fn segment(ids: Vec<Option<i32>>, names: Vec<Option<&str>>) -> SegmentZoneMaps {
        let row_count = ids.len() as u64;
        SegmentZoneMaps {
            columns: vec![
                ZoneMap::build(&Int32Array::from(ids)),
                ZoneMap::build(&StringArray::from(names)),
            ],
            row_count,
        }
    }

    fn column(name: &str) -> Column {
        Column::new_unqualified(name)
    }

    #[test]
    fn bounds_come_back_one_row_per_segment() {
        let stats = SegmentStatistics::new(
            schema(),
            vec![
                segment(vec![Some(1), Some(5)], vec![Some("a"), Some("c")]),
                segment(vec![Some(10), Some(20)], vec![Some("x"), Some("z")]),
            ],
        );

        assert_eq!(stats.num_containers(), 2);
        let min = stats.min_values(&column("id")).unwrap();
        let max = stats.max_values(&column("id")).unwrap();
        assert_eq!(min.len(), 2);
        assert_eq!(max.len(), 2);

        let min = min.as_any().downcast_ref::<Int32Array>().unwrap();
        let max = max.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!((min.value(0), max.value(0)), (1, 5));
        assert_eq!((min.value(1), max.value(1)), (10, 20));
    }

    #[test]
    fn a_segment_with_no_bound_reports_null_rather_than_a_guess() {
        let stats = SegmentStatistics::new(
            schema(),
            vec![
                segment(vec![Some(1)], vec![Some("a")]),
                segment(vec![None, None], vec![None, None]),
            ],
        );

        let min = stats.min_values(&column("id")).unwrap();
        assert_eq!(min.len(), 2);
        assert!(!min.is_null(0));
        assert!(
            min.is_null(1),
            "an all-null segment must say it does not know, not claim a bound"
        );
    }

    #[test]
    fn a_column_no_segment_knows_reports_nothing_at_all() {
        let stats = SegmentStatistics::new(
            schema(),
            vec![
                segment(vec![None], vec![None]),
                segment(vec![None], vec![None]),
            ],
        );
        assert!(stats.min_values(&column("id")).is_none());
        assert!(stats.max_values(&column("name")).is_none());
    }

    #[test]
    fn an_unknown_column_prunes_nothing() {
        let stats = SegmentStatistics::new(schema(), vec![segment(vec![Some(1)], vec![Some("a")])]);
        assert!(stats.min_values(&column("absent")).is_none());
        assert!(stats.null_counts(&column("absent")).is_none());
    }

    #[test]
    fn null_and_row_counts_line_up_with_the_segments() {
        let stats = SegmentStatistics::new(
            schema(),
            vec![
                segment(vec![Some(1), None, Some(3)], vec![Some("a"), None, None]),
                segment(vec![Some(4)], vec![Some("b")]),
            ],
        );

        let nulls = stats.null_counts(&column("name")).unwrap();
        let nulls = nulls.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(nulls.value(0), 2);
        assert_eq!(nulls.value(1), 0);

        let rows = stats.row_counts().unwrap();
        let rows = rows.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(rows.value(0), 3);
        assert_eq!(rows.value(1), 1);
    }

    #[test]
    fn set_membership_is_never_claimed() {
        let stats = SegmentStatistics::new(schema(), vec![segment(vec![Some(1)], vec![Some("a")])]);
        assert!(
            stats
                .contained(&column("id"), &HashSet::from([ScalarValue::Int32(Some(1))]))
                .is_none(),
            "a range says nothing about membership"
        );
    }

    #[test]
    fn string_bounds_are_reported_as_strings() {
        let stats = SegmentStatistics::new(
            schema(),
            vec![segment(vec![Some(1)], vec![Some("pear"), Some("apple")])],
        );
        let min = stats.min_values(&column("name")).unwrap();
        let min = min.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(min.value(0), "apple");
    }
}
