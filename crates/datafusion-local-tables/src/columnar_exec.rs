//! The scan plan for a columnar table.
//!
//! Every surviving segment is one piece of work. So is every batch still in
//! memory.
//!
//! Partitions take pieces from a shared queue. The plan does not hand out a
//! fixed share. A partition that draws a cheap piece comes back for another,
//! rather than finish early.

use std::fmt;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DataFusionError, Result, Statistics};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::stream::{self, TryStreamExt};

use datafusion::physical_optimizer::pruning::PruningPredicate;

use crate::pruning::PageStatistics;
use localtables_format::columnar::segment::SegmentReader;
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

/// The work a scan has left to do, shared by every partition.
///
/// Partitions take from this queue. The plan does not hand out a fixed share.
///
/// A fixed split is only as good as its guess at how long each piece takes, and
/// the pieces are not equal. The last row group of a flush is a partial one.
/// Compaction leaves uneven ones behind. A compressed segment costs more to
/// decode than a mapped one.
///
/// Whichever partition finishes first takes the next piece. None of that then
/// matters.
#[derive(Debug)]
struct Morsels {
    work: Vec<Work>,
    next: AtomicUsize,
}

impl Morsels {
    fn new(work: Vec<Work>) -> Self {
        Self {
            work,
            next: AtomicUsize::new(0),
        }
    }

    /// The next piece of work, or `None` once they are all taken.
    fn take(&self) -> Option<&Work> {
        let index = self.next.fetch_add(1, Ordering::Relaxed);
        self.work.get(index)
    }

    fn len(&self) -> usize {
        self.work.len()
    }
}

/// What a scan tests each page against.
#[derive(Debug, Default)]
pub struct PageFilters {
    predicates: Vec<Arc<PruningPredicate>>,
    /// Columns the predicates mention, so a scan reads only those bounds.
    columns: Vec<usize>,
}

impl PageFilters {
    pub fn new(predicates: Vec<Arc<PruningPredicate>>, columns: Vec<usize>) -> Self {
        Self {
            predicates,
            columns,
        }
    }

    fn is_empty(&self) -> bool {
        self.predicates.is_empty() || self.columns.is_empty()
    }

    /// Which pages of a segment are worth handing on.
    ///
    /// `None` means all of them. That is the answer when the segment has no
    /// page bounds, when the predicates name nothing it records, and when
    /// anything goes wrong.
    ///
    /// Pruning is an optimisation. A failure here costs a read, never a row.
    fn keep_pages(&self, schema: &SchemaRef, reader: &SegmentReader) -> Option<Vec<bool>> {
        if self.is_empty() {
            return None;
        }
        let page_rows = reader.page_rows().ok()?;
        if page_rows == 0 {
            return None;
        }
        let rows = reader.row_count().ok()?;
        let pages = rows.div_ceil(page_rows) as usize;
        let sizes: Vec<u64> = (0..pages)
            .map(|page| page_rows.min(rows - page as u64 * page_rows))
            .collect();

        let mut columns = vec![None; schema.fields().len()];
        let mut any = false;
        for &index in &self.columns {
            if let Ok(Some(zones)) = reader.page_zones(index) {
                if zones.len() == pages {
                    columns[index] = Some(zones);
                    any = true;
                }
            }
        }
        if !any {
            return None;
        }

        let statistics = PageStatistics::new(schema.clone(), columns, sizes);
        let mut keep = vec![true; pages];
        let mut ruled_out = false;
        for predicate in &self.predicates {
            let Ok(verdicts) = predicate.prune(&statistics) else {
                continue;
            };
            if verdicts.len() != pages {
                continue;
            }
            for (slot, verdict) in keep.iter_mut().zip(verdicts) {
                if !verdict {
                    ruled_out = true;
                }
                *slot &= verdict;
            }
        }
        ruled_out.then_some(keep)
    }
}

/// What one partition needs to turn a piece of work into batches.
///
/// This exists so the stream below captures one value instead of six. The
/// stream clones it once per step, and every field inside is cheap to clone.
#[derive(Clone)]
struct Scanner {
    table: ColumnarTable,
    snapshot: Arc<Snapshot>,
    projection: Option<Arc<Vec<usize>>>,
    pruning: Arc<Pruning>,
    /// The table's schema, not the projected one. Pruning names columns by
    /// their position in the table.
    schema: SchemaRef,
    pages_pruned: Count,
}

impl Scanner {
    /// Turn one piece of work into batches.
    async fn run(&self, item: &Work) -> Result<Vec<RecordBatch>> {
        match item {
            Work::Segment(entry) => self.read(entry).await,
            Work::Batch(batch) => self.project(batch),
        }
    }

    /// Read the pages of a segment that a predicate leaves.
    async fn read(&self, entry: &SegmentEntry) -> Result<Vec<RecordBatch>> {
        let keep = self.keep_pages(entry).await?;
        self.table
            .read_segment_pages(
                &self.snapshot,
                entry,
                self.projection.as_deref().map(|p| &p[..]),
                keep.as_deref(),
            )
            .await
            .map_err(to_df_error)
    }

    /// Which pages of a segment to read, or `None` for all of them.
    ///
    /// This opens the segment once. The bounds that choose the pages and the
    /// bytes those pages hold come from the same reader.
    async fn keep_pages(&self, entry: &SegmentEntry) -> Result<Option<Vec<bool>>> {
        if self.pruning.pages.is_empty() {
            return Ok(None);
        }
        let reader = self
            .table
            .segment_reader(entry)
            .await
            .map_err(to_df_error)?;
        let keep = self.pruning.pages.keep_pages(&self.schema, &reader);
        if let Some(keep) = &keep {
            self.pages_pruned.add(keep.iter().filter(|k| !**k).count());
        }
        Ok(keep)
    }

    /// Cut an in-memory batch down to the projected columns.
    fn project(&self, batch: &RecordBatch) -> Result<Vec<RecordBatch>> {
        Ok(match self.projection.as_deref() {
            Some(indices) => vec![batch.project(indices)?],
            None => vec![batch.clone()],
        })
    }
}

/// What pruning settled when the plan was built, and what it left to the scan.
#[derive(Debug, Default)]
pub struct Pruning {
    /// Segments the zone maps ruled out. Reported in EXPLAIN, so a query that
    /// prunes nothing is visible rather than merely slow.
    pub segments_pruned: usize,
    /// Tested against each surviving segment's page bounds as it is read.
    pub pages: PageFilters,
}

/// A scan of one columnar table at one snapshot.
pub struct ColumnarScanExec {
    table: ColumnarTable,
    /// Pinned for the life of the plan, so the bytes it reads stay put.
    snapshot: Arc<Snapshot>,
    projection: Option<Arc<Vec<usize>>>,
    /// The work, taken by whichever partition is free.
    morsels: Arc<Morsels>,
    /// How many partitions the plan advertises.
    partition_count: usize,
    /// Rows still to emit across all partitions, when a limit applies.
    limit: Option<usize>,
    /// What pruning settled, and what it left for the scan to settle per page.
    ///
    /// Segment pruning happened when the plan was built. Page pruning has to
    /// happen here, because it depends on bounds stored inside a segment that
    /// only a reader of that segment has.
    pruning: Arc<Pruning>,
    projected_schema: SchemaRef,
    properties: Arc<PlanProperties>,
    /// Rows handed upward and pages skipped, for `EXPLAIN ANALYZE`.
    ///
    /// Segment pruning shows in `EXPLAIN` because it is settled when the plan
    /// is built. Page pruning is settled while the scan runs, so the only place
    /// it can show is here.
    metrics: ExecutionPlanMetricsSet,
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
        pruning: Arc<Pruning>,
    ) -> Result<Self> {
        let projected_schema = match &projection {
            Some(indices) => Arc::new(snapshot.schema.project(indices)?),
            None => snapshot.schema.clone(),
        };

        let batches: Vec<RecordBatch> = snapshot.memtable.as_ref().clone();
        let mut work: Vec<Work> = segments.into_iter().map(Work::Segment).collect();
        work.extend(batches.into_iter().map(Work::Batch));

        // No more partitions than pieces of work: an extra one would find the
        // queue empty on its first look, and DataFusion counts empty partitions
        // as real ones.
        let partition_count = target_partitions.clamp(1, work.len().max(1));
        let morsels = Arc::new(Morsels::new(work));

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
            morsels,
            partition_count,
            limit,
            pruning,
            projected_schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// Rows the scan will return, before any filter above it.
    fn row_estimate(&self) -> u64 {
        let rows: u64 = self
            .morsels
            .work
            .iter()
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
        let segments = self
            .morsels
            .work
            .iter()
            .filter(|w| matches!(w, Work::Segment(_)))
            .count();
        let in_memory = self.morsels.len() - segments;

        write!(
            f,
            "ColumnarScanExec: segments={segments}, pruned={}, in_memory_batches={in_memory}, partitions={}",
            self.pruning.segments_pruned, self.partition_count
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

    /// Rows handed upward and pages skipped.
    ///
    /// Segment pruning shows in `EXPLAIN`, because it is settled when the plan
    /// is built. Page pruning is settled while the scan runs, so `EXPLAIN
    /// ANALYZE` is the only place it can show.
    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
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
        if partition >= self.partition_count {
            return Err(DataFusionError::Internal(format!(
                "partition {partition} is out of range for a {}-partition scan",
                self.partition_count
            )));
        }

        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let scanner = Scanner {
            table: self.table.clone(),
            snapshot: self.snapshot.clone(),
            projection: self.projection.clone(),
            pruning: self.pruning.clone(),
            schema: self.table.schema(),
            pages_pruned: MetricBuilder::new(&self.metrics).counter("pages_pruned", partition),
        };

        // Every partition draws from one budget, so a LIMIT stops the whole
        // scan. Each partition does not return a limit's worth of its own.
        let budget = self
            .limit
            .map(|limit| Arc::new(AtomicI64::new(limit as i64)));

        // Each step takes the next piece of work still going. A partition that
        // draws a cheap segment comes straight back for another.
        let stream = stream::try_unfold(
            (scanner, self.morsels.clone()),
            |(scanner, morsels)| async move {
                let Some(item) = morsels.take() else {
                    return Ok::<_, DataFusionError>(None);
                };
                let batches = scanner.run(item).await?;
                Ok(Some((batches, (scanner, morsels))))
            },
        )
        .map_ok(|batches| stream::iter(batches.into_iter().map(Ok)))
        .try_flatten()
        .try_filter_map(move |batch| {
            let budget = budget.clone();
            async move { Ok(apply_budget(budget.as_deref(), batch)) }
        })
        .inspect_ok(move |batch: &RecordBatch| baseline.record_output(batch.num_rows()));

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
