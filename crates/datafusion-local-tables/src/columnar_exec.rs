//! The scan plan for a columnar table.
//!
//! Work is split into partitions when the plan is built: surviving segments go
//! round-robin across them, and the rows still held in memory go with the
//! first. Each partition then streams its own work, so nothing coordinates at
//! run time beyond the shared row budget a `LIMIT` imposes.

use std::fmt;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DataFusionError, Result, Statistics};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::stream::{self, TryStreamExt};

use localtables_format::columnar::table::ColumnarTable;
use localtables_format::layout::manifest::SegmentEntry;
use localtables_format::Snapshot;

/// One piece of work for a partition.
#[derive(Debug, Clone)]
enum Work {
    /// Read a segment off the file.
    Segment(SegmentEntry),
    /// Emit rows already in memory.
    Batch(RecordBatch),
}

/// Turn a storage error into one DataFusion can carry.
pub(crate) fn to_df_error(error: localtables_format::Error) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}

/// A scan of one columnar table at one snapshot.
pub struct ColumnarScanExec {
    table: ColumnarTable,
    /// Pinned for the life of the plan, so the bytes it reads stay put.
    snapshot: Arc<Snapshot>,
    projection: Option<Arc<Vec<usize>>>,
    /// Work per partition, decided when the plan was built.
    partitions: Vec<Vec<Work>>,
    /// Rows still to emit across all partitions, when a limit applies.
    limit: Option<usize>,
    /// Segments the zone maps ruled out. Reported in EXPLAIN, so a query that
    /// prunes nothing is visible rather than merely slow.
    pruned_segments: usize,
    projected_schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl ColumnarScanExec {
    /// Build a scan over the segments and in-memory rows a plan should read.
    pub fn new(
        table: ColumnarTable,
        snapshot: Arc<Snapshot>,
        segments: Vec<SegmentEntry>,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        target_partitions: usize,
        pruned_segments: usize,
    ) -> Result<Self> {
        let projected_schema = match &projection {
            Some(indices) => Arc::new(snapshot.schema.project(indices)?),
            None => snapshot.schema.clone(),
        };

        // One partition per segment at most: splitting further would hand a
        // partition no work, and DataFusion counts empty partitions as real.
        let batches: Vec<RecordBatch> = snapshot.memtable.as_ref().clone();
        let work_units = segments.len() + usize::from(!batches.is_empty());
        let partition_count = target_partitions.clamp(1, work_units.max(1));

        let mut partitions: Vec<Vec<Work>> = vec![Vec::new(); partition_count];
        for (index, entry) in segments.into_iter().enumerate() {
            partitions[index % partition_count].push(Work::Segment(entry));
        }
        if !batches.is_empty() {
            // In-memory rows go to the partition holding the least work, so a
            // single large memtable does not land on top of a busy partition.
            let target = partitions
                .iter()
                .enumerate()
                .min_by_key(|(_, work)| work.len())
                .map(|(index, _)| index)
                .unwrap_or(0);
            partitions[target].extend(batches.into_iter().map(Work::Batch));
        }

        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(projected_schema.clone()),
            datafusion::physical_plan::Partitioning::UnknownPartitioning(partition_count),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));

        Ok(Self {
            table,
            snapshot,
            projection: projection.map(Arc::new),
            partitions,
            limit,
            pruned_segments,
            projected_schema,
            properties,
        })
    }

    /// Rows the scan will return, before any filter above it.
    fn row_estimate(&self) -> u64 {
        let rows: u64 = self
            .partitions
            .iter()
            .flatten()
            .map(|work| match work {
                Work::Segment(entry) => self.snapshot.live_rows_in(entry),
                Work::Batch(batch) => batch.num_rows() as u64,
            })
            .sum();
        match self.limit {
            Some(limit) => rows.min(limit as u64),
            None => rows,
        }
    }
}

impl fmt::Debug for ColumnarScanExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ColumnarScanExec")
    }
}

impl DisplayAs for ColumnarScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let segments: usize = self
            .partitions
            .iter()
            .flatten()
            .filter(|w| matches!(w, Work::Segment(_)))
            .count();
        let in_memory: usize = self
            .partitions
            .iter()
            .flatten()
            .filter(|w| matches!(w, Work::Batch(_)))
            .count();

        write!(
            f,
            "ColumnarScanExec: segments={segments}, pruned={}, in_memory_batches={in_memory}, partitions={}",
            self.pruned_segments,
            self.partitions.len()
        )?;
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

impl ExecutionPlan for ColumnarScanExec {
    fn name(&self) -> &str {
        "ColumnarScanExec"
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
        // A scan carries no physical expressions: filters stay above it,
        // because zone maps prune segments rather than rows.
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // A scan has no children, so there is nothing to replace.
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let work = self.partitions.get(partition).cloned().ok_or_else(|| {
            DataFusionError::Internal(format!(
                "partition {partition} is out of range for a {}-partition scan",
                self.partitions.len()
            ))
        })?;

        let table = self.table.clone();
        let snapshot = self.snapshot.clone();
        let projection = self.projection.clone();
        // Every partition draws from one budget, so a LIMIT stops the whole
        // scan rather than each partition returning a limit's worth.
        let budget = self
            .limit
            .map(|limit| Arc::new(AtomicI64::new(limit as i64)));

        let stream = stream::iter(work.into_iter().map(Ok::<Work, DataFusionError>))
            .and_then(move |item| {
                let table = table.clone();
                let snapshot = snapshot.clone();
                let projection = projection.clone();
                async move {
                    match item {
                        Work::Segment(entry) => table
                            .read_segment(&snapshot, &entry, projection.as_deref().map(|p| &p[..]))
                            .await
                            .map_err(to_df_error),
                        Work::Batch(batch) => match projection.as_deref() {
                            Some(indices) => Ok(vec![batch.project(indices)?]),
                            None => Ok(vec![batch]),
                        },
                    }
                }
            })
            .map_ok(|batches| stream::iter(batches.into_iter().map(Ok)))
            .try_flatten()
            .try_filter_map(move |batch| {
                let budget = budget.clone();
                async move { Ok(apply_budget(budget.as_deref(), batch)) }
            });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.projected_schema.clone(),
            stream,
        )))
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Arc<Statistics>> {
        // Row counts are exact: the manifest knows how many rows each segment
        // holds and the snapshot knows how many are deleted.
        let mut statistics = Statistics::new_unknown(&self.projected_schema);
        statistics.num_rows =
            datafusion::common::stats::Precision::Exact(self.row_estimate() as usize);
        Ok(Arc::new(statistics))
    }
}

/// Take a batch's rows from the shared budget.
///
/// Returns the part of the batch that fits, or `None` once the budget is gone.
fn apply_budget(budget: Option<&AtomicI64>, batch: RecordBatch) -> Option<RecordBatch> {
    let Some(budget) = budget else {
        return Some(batch);
    };
    let rows = batch.num_rows() as i64;
    if rows == 0 {
        return None;
    }

    let remaining = budget.fetch_sub(rows, Ordering::AcqRel);
    if remaining <= 0 {
        None
    } else if remaining >= rows {
        Some(batch)
    } else {
        // This batch crosses the limit: emit the rows that still fit.
        Some(batch.slice(0, remaining as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};

    fn batch(rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(
                (0..rows as i32).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    #[test]
    fn no_budget_lets_everything_through() {
        assert_eq!(apply_budget(None, batch(10)).unwrap().num_rows(), 10);
    }

    #[test]
    fn a_budget_stops_the_stream_once_it_runs_out() {
        let budget = AtomicI64::new(15);
        assert_eq!(
            apply_budget(Some(&budget), batch(10)).unwrap().num_rows(),
            10
        );
        assert_eq!(
            apply_budget(Some(&budget), batch(10)).unwrap().num_rows(),
            5,
            "the batch that crosses the limit is cut to fit"
        );
        assert!(apply_budget(Some(&budget), batch(10)).is_none());
        assert!(apply_budget(Some(&budget), batch(1)).is_none());
    }

    #[test]
    fn a_zero_budget_emits_nothing() {
        let budget = AtomicI64::new(0);
        assert!(apply_budget(Some(&budget), batch(10)).is_none());
    }

    #[test]
    fn an_empty_batch_never_consumes_budget() {
        let budget = AtomicI64::new(5);
        assert!(apply_budget(Some(&budget), batch(0)).is_none());
        assert_eq!(budget.load(Ordering::Acquire), 5);
    }
}
