//! The DataFusion table provider for a columnar table.
//!
//! Planning a scan does three things: pin a snapshot, drop the segments the
//! filters cannot match, and split what is left across partitions.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{Result, Statistics};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_optimizer::pruning::PruningPredicateBuilder;
use datafusion::physical_plan::ExecutionPlan;

use localtables_format::columnar::table::ColumnarTable;
use localtables_format::layout::manifest::SegmentEntry;

use crate::columnar_exec::{to_df_error, ColumnarScanExec};
use crate::pruning::{SegmentStatistics, SegmentZoneMaps};

/// Exposes a [`ColumnarTable`] to DataFusion.
#[derive(Debug, Clone)]
pub struct ColumnarTableProvider {
    table: ColumnarTable,
}

impl ColumnarTableProvider {
    pub fn new(table: ColumnarTable) -> Self {
        Self { table }
    }

    /// The underlying table, for writes and maintenance the SQL layer does not
    /// cover.
    pub fn table(&self) -> &ColumnarTable {
        &self.table
    }
}

impl From<ColumnarTable> for ColumnarTableProvider {
    fn from(table: ColumnarTable) -> Self {
        Self::new(table)
    }
}

#[async_trait]
impl TableProvider for ColumnarTableProvider {
    fn schema(&self) -> SchemaRef {
        self.table.schema().clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// Filters are reported as inexact, so DataFusion keeps its own filter
    /// above the scan.
    ///
    /// Zone maps rule out whole segments; they say nothing about individual
    /// rows, so a segment that survives still holds rows the predicate
    /// rejects. Claiming exactness here would return them.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    fn statistics(&self) -> Option<Statistics> {
        let snapshot = self.table.snapshot();
        let mut statistics = Statistics::new_unknown(&self.schema());
        statistics.num_rows =
            datafusion::common::stats::Precision::Exact(snapshot.live_rows() as usize);
        Some(statistics)
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Pin now, so the plan reads one commit however long it runs and the
        // writer cannot reuse the bytes underneath it.
        let snapshot = self.table.snapshot();
        let schema = self.schema();

        let candidates: Vec<SegmentEntry> = snapshot.live_segments().cloned().collect();
        let total = candidates.len();
        let segments = self
            .prune(state, &schema, filters, &candidates)
            .await
            .unwrap_or(candidates);
        let pruned = total - segments.len();

        Ok(Arc::new(ColumnarScanExec::new(
            self.table.clone(),
            snapshot,
            segments,
            projection.cloned(),
            limit,
            state.config().target_partitions(),
            pruned,
        )?))
    }
}

impl ColumnarTableProvider {
    /// Drop the segments the filters cannot match.
    ///
    /// Returns `None` when nothing could be ruled out, which the caller reads
    /// as "read them all". Pruning is an optimisation: a failure to build the
    /// predicate, or a segment whose bounds are unknown, costs a read and never
    /// loses a row.
    async fn prune(
        &self,
        state: &dyn Session,
        schema: &SchemaRef,
        filters: &[Expr],
        candidates: &[SegmentEntry],
    ) -> Option<Vec<SegmentEntry>> {
        if filters.is_empty() || candidates.is_empty() {
            return None;
        }

        // Read every candidate's zone maps. This touches segment metadata
        // only, not column data.
        let mut zone_maps = Vec::with_capacity(candidates.len());
        for entry in candidates {
            let reader = self.table.segment_reader(entry).await.ok()?;
            zone_maps.push(SegmentZoneMaps::from_reader(&reader).ok()?);
        }
        let statistics = SegmentStatistics::new(schema.clone(), zone_maps);

        let mut keep = vec![true; candidates.len()];
        let df_schema = datafusion::common::DFSchema::try_from(schema.as_ref().clone()).ok()?;
        for filter in filters {
            let Ok(physical) = state.create_physical_expr(filter.clone(), &df_schema) else {
                continue;
            };
            // `build` returns nothing when the predicate is trivially true
            // or cannot be turned into one, both of which prune nothing.
            let Some(predicate) = PruningPredicateBuilder::new()
                .with_file_schema(schema.clone())
                .build(physical)
            else {
                continue;
            };
            let Ok(verdicts) = predicate.prune(&statistics) else {
                continue;
            };
            // Filters combine with AND, so one predicate ruling a segment out
            // is enough.
            for (slot, verdict) in keep.iter_mut().zip(verdicts) {
                *slot &= verdict;
            }
        }

        if keep.iter().all(|k| *k) {
            return None;
        }
        Some(
            candidates
                .iter()
                .zip(keep)
                .filter(|(_, keep)| *keep)
                .map(|(entry, _)| entry.clone())
                .collect(),
        )
    }
}

/// Open a columnar table and register it with a session.
///
/// A convenience for the common case; nothing stops a caller building the
/// provider directly.
pub async fn register_columnar_table(
    ctx: &datafusion::prelude::SessionContext,
    name: &str,
    path: &std::path::Path,
    options: localtables_format::TableOptions,
) -> Result<ColumnarTable> {
    let table = ColumnarTable::open(path, options)
        .await
        .map_err(to_df_error)?;
    ctx.register_table(name, Arc::new(ColumnarTableProvider::new(table.clone())))?;
    Ok(table)
}
