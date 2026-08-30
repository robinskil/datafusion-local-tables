//! Zone maps, in the shape DataFusion's pruning wants them.
//!
//! Each segment is one "container". DataFusion asks for the minimum, maximum,
//! null count and row count of a column across all containers. It then works
//! out which containers a predicate cannot match. The scan skips those and
//! reads none of their data.
//!
//! Equality is the case zone maps handle worst. On a column of scattered
//! values, every segment's range spans the value. Nothing is ruled out.
//!
//! Where a segment carries a membership filter, [`contained`] answers that case
//! instead.
//!
//! [`contained`]: PruningStatistics::contained
//!
//! The interval reasoning is DataFusion's own. This file reports what the zone
//! maps know, and nothing where they do not know.
//!
//! A null entry means "no information". It costs a segment read. A wrong entry
//! would lose rows.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, UInt64Array};
use arrow::datatypes::{DataType, SchemaRef};
use datafusion::common::pruning::PruningStatistics;
use datafusion::common::{Column, ScalarValue};
use datafusion::logical_expr::{BinaryExpr, Expr, Like, Operator};

use localtables_format::columnar::bloom::BloomFilter;
use localtables_format::columnar::segment::SegmentReader;
use localtables_format::columnar::trigram;
use localtables_format::columnar::zonemap::ZoneMap;

/// One segment's bounds for every column, read once when the scan is planned.
#[derive(Debug, Clone)]
pub struct SegmentZoneMaps {
    /// One entry per column of the table schema.
    pub columns: Vec<ZoneMap>,
    /// Membership filters, by column position. Empty when none were loaded.
    pub blooms: Vec<(usize, BloomFilter)>,
    /// Trigram filters, by column position. Empty when none were loaded.
    pub trigrams: Vec<(usize, BloomFilter)>,
    pub row_count: u64,
}

impl SegmentZoneMaps {
    /// Read the zone maps out of a segment's metadata.
    ///
    /// `bloom_columns` names the columns whose membership filter is worth
    /// loading: a filter lives outside the metadata frame and is far larger
    /// than one, so only the columns a predicate actually mentions are read.
    /// A column with no filter, or a segment written before filters were asked
    /// for, contributes nothing and prunes nothing.
    pub fn from_reader(
        reader: &SegmentReader,
        bloom_columns: &[usize],
        trigram_columns: &[usize],
    ) -> localtables_format::Result<Self> {
        let meta = reader.meta()?;
        let columns: Vec<ZoneMap> = meta.columns.iter().map(|c| c.zone.to_native()).collect();
        let row_count = meta.row_count.to_native();

        let mut blooms = Vec::new();
        for &index in bloom_columns {
            // A damaged filter must not fail the scan: it is an optimisation,
            // and the column's own bytes still carry the truth.
            if let Ok(Some(filter)) = reader.bloom_filter(index) {
                blooms.push((index, filter));
            }
        }

        let mut trigrams = Vec::new();
        for &index in trigram_columns {
            if let Ok(Some(filter)) = reader.trigram_filter(index) {
                trigrams.push((index, filter));
            }
        }

        Ok(Self {
            columns,
            blooms,
            trigrams,
            row_count,
        })
    }

    fn bloom(&self, index: usize) -> Option<&BloomFilter> {
        find(&self.blooms, index)
    }

    /// Whether this segment may hold a value matching a substring predicate.
    ///
    /// False only when the segment carries a trigram filter and that filter is
    /// sure one of the required pieces is absent. No filter means no
    /// information, which is `true`: read the segment.
    pub fn may_match(&self, requirement: &SubstringRequirement) -> bool {
        let Some(filter) = find(&self.trigrams, requirement.column) else {
            return true;
        };
        requirement
            .trigrams
            .iter()
            .all(|piece| filter.may_contain_hash(trigram::hash(piece)))
    }
}

fn find(filters: &[(usize, BloomFilter)], index: usize) -> Option<&BloomFilter> {
    filters
        .iter()
        .find(|(at, _)| *at == index)
        .map(|(_, filter)| filter)
}

/// Bounds for the pages inside one segment, in the shape pruning wants.
///
/// This is [`SegmentStatistics`] one level down. There a container is a
/// segment, and the answer is whether to read it. Here a container is a row
/// range inside a segment, and the answer is whether to hand it on.
///
/// A page with no bounds gives a null, which reads as "could match".
#[derive(Debug)]
pub struct PageStatistics {
    schema: SchemaRef,
    /// Bounds per column, each an entry per page. Absent where a column has no
    /// page bounds at all.
    columns: Vec<Option<Vec<ZoneMap>>>,
    pages: usize,
    /// Rows in each page, the last one usually shorter.
    rows: Vec<u64>,
}

impl PageStatistics {
    pub fn new(schema: SchemaRef, columns: Vec<Option<Vec<ZoneMap>>>, rows: Vec<u64>) -> Self {
        Self {
            pages: rows.len(),
            schema,
            columns,
            rows,
        }
    }

    fn zones(&self, column: &Column) -> Option<(&[ZoneMap], &DataType)> {
        let index = self.schema.index_of(&column.name).ok()?;
        let zones = self.columns.get(index)?.as_deref()?;
        if zones.len() != self.pages {
            return None;
        }
        Some((zones, self.schema.field(index).data_type()))
    }

    fn bounds(
        &self,
        column: &Column,
        pick: impl Fn(&ZoneMap, &DataType) -> Option<ArrayRef>,
    ) -> Option<ArrayRef> {
        let (zones, data_type) = self.zones(column)?;

        let mut any_known = false;
        let mut parts: Vec<ArrayRef> = Vec::with_capacity(zones.len());
        for zone in zones {
            match pick(zone, data_type) {
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

impl PruningStatistics for PageStatistics {
    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bounds(column, |zone, data_type| zone.min_array(data_type))
    }

    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bounds(column, |zone, data_type| zone.max_array(data_type))
    }

    fn num_containers(&self) -> usize {
        self.pages
    }

    fn null_counts(&self, column: &Column) -> Option<ArrayRef> {
        let (zones, _) = self.zones(column)?;
        Some(Arc::new(UInt64Array::from(
            zones.iter().map(|zone| zone.null_count).collect::<Vec<u64>>(),
        )))
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        Some(Arc::new(UInt64Array::from(self.rows.clone())))
    }

    /// Membership filters are per column chunk, not per page, so this knows
    /// nothing a page at a time.
    fn contained(&self, _column: &Column, _values: &HashSet<ScalarValue>) -> Option<BooleanArray> {
        None
    }
}

/// Pieces a `LIKE` predicate needs a segment to contain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubstringRequirement {
    /// Position of the column in the table schema.
    pub column: usize,
    pub trigrams: Vec<[u8; trigram::SIZE]>,
}

/// Read the `LIKE` predicates a trigram filter can act on.
///
/// DataFusion's pruning does not route `LIKE` anywhere, so this reads the
/// filter expressions itself. It is deliberately narrow, because every shape it
/// accepts wrongly would prune a segment that holds matching rows:
///
/// * `NOT LIKE` says a value must *not* contain the term, which the filter
///   cannot show;
/// * `ILIKE` matches text the filter never saw, since the filter holds the
///   bytes as they were written;
/// * an escape character makes `%` and `_` ordinary, so splitting on them
///   would invent pieces the pattern never required;
/// * `OR` lets a row match through the other branch, so nothing either branch
///   requires is required of the segment.
///
/// Anything else contributes no requirement, which prunes nothing.
pub fn substring_requirements(filters: &[Expr], schema: &SchemaRef) -> Vec<SubstringRequirement> {
    let mut found = Vec::new();
    for filter in filters {
        collect_substrings(filter, schema, &mut found);
    }
    found
}

fn collect_substrings(expr: &Expr, schema: &SchemaRef, found: &mut Vec<SubstringRequirement>) {
    match expr {
        // Both sides of an AND must hold, so both sides' requirements hold.
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::And,
            right,
        }) => {
            collect_substrings(left, schema, found);
            collect_substrings(right, schema, found);
        }
        Expr::Like(Like {
            negated: false,
            expr,
            pattern,
            escape_char: None,
            case_insensitive: false,
        }) => {
            let Expr::Column(column) = expr.as_ref() else {
                return;
            };
            let Expr::Literal(value, _) = pattern.as_ref() else {
                return;
            };
            let pattern = match value {
                ScalarValue::Utf8(Some(text))
                | ScalarValue::LargeUtf8(Some(text))
                | ScalarValue::Utf8View(Some(text)) => text,
                _ => return,
            };
            let Ok(index) = schema.index_of(&column.name) else {
                return;
            };
            let trigrams = trigram::required(pattern);
            // A pattern with no run of three bytes says nothing usable.
            if !trigrams.is_empty() {
                found.push(SubstringRequirement {
                    column: index,
                    trigrams,
                });
            }
        }
        _ => {}
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

    pub fn segments(&self) -> &[SegmentZoneMaps] {
        &self.segments
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
    /// A membership filter answers half of this question. It can show that a
    /// segment holds *none* of the values, which is `false` and prunes the
    /// segment; it can never show that a segment holds *only* them, which is
    /// what `true` would mean, so `true` is never returned. Anything else is
    /// null: unknown, read the segment.
    ///
    /// The filter admits false positives, so "may hold one of these" is not
    /// evidence of anything and reports null. It admits no false negatives,
    /// which is what makes `false` safe to act on.
    fn contained(&self, column: &Column, values: &HashSet<ScalarValue>) -> Option<BooleanArray> {
        let (index, data_type) = self.column(column)?;
        if values.is_empty() || self.segments.iter().all(|s| s.bloom(index).is_none()) {
            return None;
        }

        // Each literal becomes a one-row array once, not once per segment.
        let mut probes = Vec::with_capacity(values.len());
        for value in values {
            probes.push(value.to_array().ok()?);
        }

        let mut any_known = false;
        let mut verdicts = Vec::with_capacity(self.segments.len());
        for segment in &self.segments {
            let verdict = match segment.bloom(index) {
                Some(filter) if !probes.iter().any(|p| filter.may_contain(p, data_type)) => {
                    any_known = true;
                    Some(false)
                }
                _ => None,
            };
            verdicts.push(verdict);
        }

        if !any_known {
            return None;
        }
        Some(BooleanArray::from(verdicts))
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
            blooms: Vec::new(),
            trigrams: Vec::new(),
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
