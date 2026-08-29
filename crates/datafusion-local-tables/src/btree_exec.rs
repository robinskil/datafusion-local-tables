//! The scan plan for a b-tree table.
//!
//! One partition: the tree returns rows in key order, and splitting that across
//! partitions would throw the ordering away for no gain, since a point lookup
//! or a narrow range is already cheap.

use std::fmt;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DataFusionError, Result, Statistics};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};

use localtables_format::btree::{BTreeSnapshot, BTreeTable};

use crate::columnar_exec::to_df_error;

/// The span of keys a scan reads.
///
/// `start` is inclusive and `end` exclusive, matching how the tree reads them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KeyRange {
    pub start: Vec<u8>,
    pub end: Option<Vec<u8>>,
}

impl KeyRange {
    /// The whole tree.
    pub fn all() -> Self {
        Self::default()
    }

    /// True when this range covers everything, so nothing was narrowed.
    pub fn is_unbounded(&self) -> bool {
        self.start.is_empty() && self.end.is_none()
    }

    /// A range that cannot hold anything.
    ///
    /// A predicate like `id > 5 AND id < 3` produces one; the scan then reads
    /// nothing rather than reading the tree to find nothing.
    pub fn is_empty(&self) -> bool {
        self.end
            .as_ref()
            .is_some_and(|end| self.start.as_slice() >= end.as_slice())
    }
}

/// A scan of one b-tree table at one snapshot.
pub struct BTreeScanExec {
    table: BTreeTable,
    snapshot: Arc<BTreeSnapshot>,
    range: KeyRange,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    projected_schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl BTreeScanExec {
    pub fn new(
        table: BTreeTable,
        snapshot: Arc<BTreeSnapshot>,
        range: KeyRange,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
    ) -> Result<Self> {
        let projected_schema = match &projection {
            Some(indices) => Arc::new(snapshot.schema.project(indices)?),
            None => snapshot.schema.clone(),
        };
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(projected_schema.clone()),
            datafusion::physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            table,
            snapshot,
            range,
            projection,
            limit,
            projected_schema,
            properties,
        })
    }
}

impl fmt::Debug for BTreeScanExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BTreeScanExec")
    }
}

impl DisplayAs for BTreeScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BTreeScanExec: ")?;
        if self.range.is_empty() {
            write!(f, "keys=none")?;
        } else if self.range.is_unbounded() {
            write!(f, "keys=all")?;
        } else {
            write!(
                f,
                "keys=[{}..{})",
                hex(&self.range.start),
                self.range
                    .end
                    .as_deref()
                    .map(hex)
                    .unwrap_or_else(|| "end".to_string())
            )?;
        }
        if let Some(projection) = &self.projection {
            let names: Vec<&str> = projection
                .iter()
                .map(|i| self.snapshot.schema.field(*i).name().as_str())
                .collect();
            write!(f, ", projection=[{}]", names.join(", "))?;
        }
        if let Some(limit) = self.limit {
            write!(f, ", limit={limit}")?;
        }
        Ok(())
    }
}

/// A short, readable form of a key bound for EXPLAIN.
fn hex(bytes: &[u8]) -> String {
    let shown: String = bytes.iter().take(8).map(|b| format!("{b:02x}")).collect();
    if bytes.len() > 8 {
        format!("{shown}..")
    } else {
        shown
    }
}

impl ExecutionPlan for BTreeScanExec {
    fn name(&self) -> &str {
        "BTreeScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        // The key bounds are bytes by this point, not expressions.
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "a b-tree scan runs in one partition, asked for {partition}"
            )));
        }

        let table = self.table.clone();
        let snapshot = self.snapshot.clone();
        let range = self.range.clone();
        let projection = self.projection.clone();
        let limit = self.limit;
        let schema = self.projected_schema.clone();

        let stream = futures::stream::once(async move {
            if range.is_empty() {
                return Ok(arrow::record_batch::RecordBatch::new_empty(schema));
            }
            let batch = table
                .range(&snapshot, &range.start, range.end.as_deref(), limit)
                .await
                .map_err(to_df_error)?;
            match projection {
                Some(indices) => batch.project(&indices).map_err(DataFusionError::from),
                None => Ok(batch),
            }
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.projected_schema.clone(),
            stream,
        )))
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Arc<Statistics>> {
        let mut statistics = Statistics::new_unknown(&self.projected_schema);
        if self.range.is_empty() {
            statistics.num_rows = datafusion::common::stats::Precision::Exact(0);
        } else if self.range.is_unbounded() {
            // Exact only once the overlay is merged, so this is an estimate.
            statistics.num_rows = datafusion::common::stats::Precision::Inexact(
                self.snapshot.approximate_rows() as usize,
            );
        }
        Ok(Arc::new(statistics))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unbounded_range_covers_everything() {
        let range = KeyRange::all();
        assert!(range.is_unbounded());
        assert!(!range.is_empty());
    }

    #[test]
    fn a_range_whose_end_is_below_its_start_holds_nothing() {
        let range = KeyRange {
            start: b"m".to_vec(),
            end: Some(b"a".to_vec()),
        };
        assert!(range.is_empty());
    }

    #[test]
    fn a_range_whose_bounds_meet_holds_nothing() {
        let range = KeyRange {
            start: b"m".to_vec(),
            end: Some(b"m".to_vec()),
        };
        assert!(range.is_empty(), "the end bound is exclusive");
    }

    #[test]
    fn a_normal_range_is_neither_empty_nor_unbounded() {
        let range = KeyRange {
            start: b"a".to_vec(),
            end: Some(b"m".to_vec()),
        };
        assert!(!range.is_empty());
        assert!(!range.is_unbounded());
    }

    #[test]
    fn a_bound_is_rendered_short_enough_to_read() {
        assert_eq!(hex(&[0x01, 0xab]), "01ab");
        assert_eq!(hex(&[0xff; 20]), "ffffffffffffffff..");
    }
}
