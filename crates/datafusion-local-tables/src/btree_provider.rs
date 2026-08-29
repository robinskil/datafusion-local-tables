//! The DataFusion table provider for a b-tree table.
//!
//! The job here is turning a `WHERE` clause into a key range. A predicate that
//! bounds the first key column becomes a byte range the tree can seek to, which
//! is the difference between a point lookup and a full scan.

use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{Result, ScalarValue, Statistics};
use datafusion::logical_expr::{
    BinaryExpr, Expr, Operator, TableProviderFilterPushDown, TableType,
};
use datafusion::physical_plan::ExecutionPlan;

use localtables_format::btree::keycodec;
use localtables_format::btree::BTreeTable;

use crate::btree_exec::{BTreeScanExec, KeyRange};

/// Exposes a [`BTreeTable`] to DataFusion.
#[derive(Debug, Clone)]
pub struct BTreeTableProvider {
    table: BTreeTable,
}

impl BTreeTableProvider {
    pub fn new(table: BTreeTable) -> Self {
        Self { table }
    }

    pub fn table(&self) -> &BTreeTable {
        &self.table
    }

    /// The name of the first key column, which is the one bounds apply to.
    fn leading_key(&self) -> Option<&str> {
        let index = *self.table.key_columns().first()?;
        Some(self.table.schema().field(index).name())
    }
}

impl From<BTreeTable> for BTreeTableProvider {
    fn from(table: BTreeTable) -> Self {
        Self::new(table)
    }
}

#[async_trait]
impl TableProvider for BTreeTableProvider {
    fn schema(&self) -> SchemaRef {
        self.table.schema().clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// A bound on the leading key column is exact; anything else is not.
    ///
    /// The tree seeks to a byte range, and a range derived from a bound on the
    /// first key column contains exactly the rows that bound selects. Every
    /// other predicate still has to be evaluated above the scan.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        let Some(key) = self.leading_key() else {
            return Ok(vec![
                TableProviderFilterPushDown::Unsupported;
                filters.len()
            ]);
        };
        Ok(filters
            .iter()
            .map(|filter| match key_bound(filter, key) {
                Some(_) => TableProviderFilterPushDown::Exact,
                None => TableProviderFilterPushDown::Inexact,
            })
            .collect())
    }

    fn statistics(&self) -> Option<Statistics> {
        let snapshot = self.table.snapshot();
        let mut statistics = Statistics::new_unknown(&self.schema());
        statistics.num_rows =
            datafusion::common::stats::Precision::Inexact(snapshot.approximate_rows() as usize);
        Some(statistics)
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let snapshot = self.table.snapshot();
        let range = self.key_range(filters)?;

        Ok(Arc::new(BTreeScanExec::new(
            self.table.clone(),
            snapshot,
            range,
            projection.cloned(),
            limit,
        )?))
    }
}

impl BTreeTableProvider {
    /// Narrow the scan to the keys the filters allow.
    ///
    /// Filters combine with AND, so each one can only tighten the range.
    fn key_range(&self, filters: &[Expr]) -> Result<KeyRange> {
        let mut range = KeyRange::all();
        let Some(key) = self.leading_key() else {
            return Ok(range);
        };
        let index = self.table.key_columns()[0];
        let data_type = self.table.schema().field(index).data_type().clone();

        for filter in filters {
            let Some((operator, value)) = key_bound(filter, key) else {
                continue;
            };
            let Some(encoded) = encode_bound(&value, &data_type) else {
                continue;
            };
            apply_bound(&mut range, operator, encoded);
        }
        Ok(range)
    }
}

/// Tighten a range with one bound on the key.
///
/// The encoded value is a prefix of every full key with that leading column, so
/// `>` and `<=` step past the whole prefix while `>=` and `<` stop at its start.
fn apply_bound(range: &mut KeyRange, operator: Operator, encoded: Vec<u8>) {
    let past_prefix = || keycodec::prefix_upper_bound(&encoded);

    match operator {
        Operator::Eq => {
            raise_start(range, encoded.clone());
            if let Some(above) = past_prefix() {
                lower_end(range, above);
            }
        }
        Operator::Gt => {
            if let Some(above) = past_prefix() {
                raise_start(range, above);
            }
        }
        Operator::GtEq => raise_start(range, encoded),
        Operator::Lt => lower_end(range, encoded),
        Operator::LtEq => {
            if let Some(above) = past_prefix() {
                lower_end(range, above);
            }
        }
        _ => {}
    }
}

/// Move the start up, never down: an extra filter can only narrow.
fn raise_start(range: &mut KeyRange, bound: Vec<u8>) {
    if bound > range.start {
        range.start = bound;
    }
}

/// Move the end down, never up.
fn lower_end(range: &mut KeyRange, bound: Vec<u8>) {
    match &range.end {
        Some(current) if *current <= bound => {}
        _ => range.end = Some(bound),
    }
}

/// The comparison a filter makes against the key column, if it makes one.
///
/// Recognises `key <op> literal` and the mirrored `literal <op> key`.
fn key_bound(filter: &Expr, key: &str) -> Option<(Operator, ScalarValue)> {
    let Expr::BinaryExpr(BinaryExpr { left, op, right }) = filter else {
        return None;
    };

    let (operator, value) = match (left.as_ref(), right.as_ref()) {
        (Expr::Column(column), Expr::Literal(value, _)) if column.name == key => {
            (*op, value.clone())
        }
        (Expr::Literal(value, _), Expr::Column(column)) if column.name == key => {
            (mirror(*op)?, value.clone())
        }
        _ => return None,
    };

    // A comparison with null is never true, so it bounds nothing this code can
    // express; leave it to the filter above the scan.
    if value.is_null() {
        return None;
    }
    matches!(
        operator,
        Operator::Eq | Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq
    )
    .then_some((operator, value))
}

/// The operator with its operands swapped: `5 < id` is `id > 5`.
fn mirror(operator: Operator) -> Option<Operator> {
    Some(match operator {
        Operator::Eq => Operator::Eq,
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        _ => return None,
    })
}

/// Encode a literal as the key prefix its column would produce.
///
/// A literal of a different type is cast first; one that cannot be cast bounds
/// nothing, and the filter above the scan handles it.
fn encode_bound(value: &ScalarValue, data_type: &arrow::datatypes::DataType) -> Option<Vec<u8>> {
    let array: ArrayRef = value.to_array().ok()?;
    let array = if array.data_type() == data_type {
        array
    } else {
        arrow::compute::cast(&array, data_type).ok()?
    };

    let mut out = Vec::new();
    keycodec::encode_value(&mut out, array.as_ref(), 0).ok()?;
    Some(out)
}

/// Open a b-tree table and register it with a session.
pub async fn register_btree_table(
    ctx: &datafusion::prelude::SessionContext,
    name: &str,
    path: &std::path::Path,
    key_columns: &[&str],
    options: localtables_format::TableOptions,
) -> Result<BTreeTable> {
    let table = BTreeTable::open(path, key_columns, options)
        .await
        .map_err(crate::columnar_exec::to_df_error)?;
    ctx.register_table(name, Arc::new(BTreeTableProvider::new(table.clone())))?;
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::{col, lit};

    fn bound(filter: &Expr) -> Option<(Operator, ScalarValue)> {
        key_bound(filter, "id")
    }

    #[test]
    fn a_comparison_against_the_key_is_recognised() {
        let (operator, value) = bound(&col("id").gt(lit(5i64))).unwrap();
        assert_eq!(operator, Operator::Gt);
        assert_eq!(value, ScalarValue::Int64(Some(5)));
    }

    #[test]
    fn a_mirrored_comparison_is_flipped() {
        let (operator, value) = bound(&lit(5i64).lt(col("id"))).unwrap();
        assert_eq!(operator, Operator::Gt, "5 < id means id > 5");
        assert_eq!(value, ScalarValue::Int64(Some(5)));
    }

    #[test]
    fn a_comparison_against_another_column_is_not_a_key_bound() {
        assert!(bound(&col("name").gt(lit(5i64))).is_none());
    }

    #[test]
    fn a_comparison_with_null_bounds_nothing() {
        assert!(bound(&col("id").eq(lit(ScalarValue::Int64(None)))).is_none());
    }

    #[test]
    fn an_operator_the_tree_cannot_seek_on_is_not_a_bound() {
        assert!(bound(&col("id").not_eq(lit(5i64))).is_none());
        assert!(bound(&col("id").is_null()).is_none());
    }

    #[test]
    fn equality_narrows_to_the_keys_with_that_prefix() {
        let mut range = KeyRange::all();
        apply_bound(&mut range, Operator::Eq, b"key".to_vec());

        assert_eq!(range.start, b"key".to_vec());
        assert_eq!(range.end, Some(b"kez".to_vec()));
        assert!(!range.is_empty());
    }

    #[test]
    fn greater_than_starts_past_the_whole_prefix() {
        let mut range = KeyRange::all();
        apply_bound(&mut range, Operator::Gt, b"abc".to_vec());
        assert_eq!(
            range.start,
            b"abd".to_vec(),
            "every key beginning abc must be excluded"
        );
        assert!(range.end.is_none());
    }

    #[test]
    fn greater_or_equal_starts_at_the_prefix() {
        let mut range = KeyRange::all();
        apply_bound(&mut range, Operator::GtEq, b"abc".to_vec());
        assert_eq!(range.start, b"abc".to_vec());
    }

    #[test]
    fn less_than_ends_at_the_prefix() {
        let mut range = KeyRange::all();
        apply_bound(&mut range, Operator::Lt, b"abc".to_vec());
        assert_eq!(range.end, Some(b"abc".to_vec()));
    }

    #[test]
    fn less_or_equal_ends_past_the_prefix() {
        let mut range = KeyRange::all();
        apply_bound(&mut range, Operator::LtEq, b"abc".to_vec());
        assert_eq!(
            range.end,
            Some(b"abd".to_vec()),
            "every key beginning abc must be included"
        );
    }

    #[test]
    fn two_bounds_narrow_rather_than_widen() {
        let mut range = KeyRange::all();
        apply_bound(&mut range, Operator::GtEq, b"c".to_vec());
        apply_bound(&mut range, Operator::GtEq, b"a".to_vec());
        assert_eq!(range.start, b"c".to_vec(), "the tighter start wins");

        apply_bound(&mut range, Operator::Lt, b"z".to_vec());
        apply_bound(&mut range, Operator::Lt, b"zz".to_vec());
        assert_eq!(range.end, Some(b"z".to_vec()), "the tighter end wins");
    }

    #[test]
    fn contradictory_bounds_give_an_empty_range() {
        let mut range = KeyRange::all();
        apply_bound(&mut range, Operator::GtEq, b"m".to_vec());
        apply_bound(&mut range, Operator::Lt, b"a".to_vec());
        assert!(range.is_empty(), "nothing satisfies both");
    }
}
