//! The DataFusion table provider for a columnar table.
//!
//! Planning a scan does three things: pin a snapshot, drop the segments the
//! filters cannot match, and split what is left across partitions.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{DataFusionError, Result, Statistics};
use datafusion::datasource::sink::DataSinkExec;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_optimizer::pruning::{PruningPredicate, PruningPredicateBuilder};
use datafusion::physical_plan::ExecutionPlan;

use localtables_format::columnar::table::ColumnarTable;
use localtables_format::layout::manifest::SegmentEntry;

use crate::columnar_exec::{to_df_error, ColumnarScanExec};
use crate::columnar_exec::{PageFilters, Pruning};
use crate::dml::{compile_assignments, compile_predicate, ColumnarDataSink, DmlExec};
use crate::pruning::{substring_requirements, SegmentStatistics, SegmentZoneMaps};

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

        // The same predicates that chose the segments, handed on to test the
        // pages inside them. Building them twice would be the same work and
        // could disagree.
        let pruning = Arc::new(Pruning {
            segments_pruned: pruned,
            pages: PageFilters::new(
                self.predicates(state, &schema, filters),
                referenced_columns(filters, &schema),
            ),
        });

        Ok(Arc::new(ColumnarScanExec::new(
            self.table.clone(),
            snapshot,
            segments,
            projection.cloned(),
            limit,
            state.config().target_partitions(),
            pruning,
        )?))
    }

    /// `INSERT INTO`.
    ///
    /// Rows stream into the write-ahead log; each batch is durable before the
    /// next is read, and queryable as soon as it lands.
    async fn insert_into(
        &self,
        _state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match insert_op {
            InsertOp::Append => {}
            other => {
                // Overwrite and replace would have to decide what to do with
                // rows a concurrent reader is pinned to; appending is the only
                // one this engine implements today.
                return Err(DataFusionError::NotImplemented(format!(
                    "{other:?} is not supported; only INSERT ... VALUES / SELECT"
                )));
            }
        }

        Ok(Arc::new(DataSinkExec::new(
            input,
            Arc::new(ColumnarDataSink::new(self.table.clone())),
            None,
        )))
    }

    /// `DELETE FROM`.
    ///
    /// With no `WHERE`, every row goes. With one, the rows it selects are found
    /// by reading the segments the zone maps cannot rule out.
    async fn delete_from(
        &self,
        state: &dyn Session,
        filters: Vec<Expr>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let predicate = compile_predicate(state, &self.schema(), &filters)?;
        Ok(Arc::new(DmlExec::delete(self.table.clone(), predicate)))
    }

    /// `UPDATE`.
    ///
    /// Matching rows are deleted and their replacements appended, as one
    /// durable record.
    async fn update(
        &self,
        state: &dyn Session,
        assignments: Vec<(String, Expr)>,
        filters: Vec<Expr>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = self.schema();
        let predicate = compile_predicate(state, &schema, &filters)?;
        let assignments = compile_assignments(state, &schema, &assignments)?;
        Ok(Arc::new(DmlExec::update(
            self.table.clone(),
            predicate,
            assignments,
        )))
    }
}

impl ColumnarTableProvider {
    /// Turn the filters into predicates that can be tested against bounds.
    ///
    /// A filter that cannot become one, or that is trivially true, is left out
    /// rather than treated as an error: it simply prunes nothing.
    fn predicates(
        &self,
        state: &dyn Session,
        schema: &SchemaRef,
        filters: &[Expr],
    ) -> Vec<Arc<PruningPredicate>> {
        let Ok(df_schema) = datafusion::common::DFSchema::try_from(schema.as_ref().clone()) else {
            return Vec::new();
        };
        filters
            .iter()
            .filter_map(|filter| {
                let physical = state
                    .create_physical_expr(filter.clone(), &df_schema)
                    .ok()?;
                PruningPredicateBuilder::new()
                    .with_file_schema(schema.clone())
                    .build(physical)
            })
            .collect()
    }

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

        // Columns the filters mention. A membership filter is far larger than
        // a zone map and lives outside the metadata frame, so only these are
        // read; a filter on a column nobody asked about would prune nothing.
        let bloom_columns = referenced_columns(filters, schema);

        // The LIKE predicates a trigram filter could act on. Read first,
        // because it decides which trigram filters are worth loading, and they
        // are the largest thing pruning reads.
        let substrings = substring_requirements(filters, schema);
        let mut trigram_columns: Vec<usize> = substrings.iter().map(|s| s.column).collect();
        trigram_columns.sort_unstable();
        trigram_columns.dedup();

        // Read every candidate's zone maps. This touches segment metadata
        // only, not column data.
        let mut zone_maps = Vec::with_capacity(candidates.len());
        for entry in candidates {
            let reader = self.table.segment_reader(entry).await.ok()?;
            zone_maps.push(
                SegmentZoneMaps::from_reader(&reader, &bloom_columns, &trigram_columns).ok()?,
            );
        }
        let statistics = SegmentStatistics::new(schema.clone(), zone_maps);

        let mut keep = vec![true; candidates.len()];
        for predicate in self.predicates(state, schema, filters) {
            let Ok(verdicts) = predicate.prune(&statistics) else {
                continue;
            };
            // Filters combine with AND, so one predicate ruling a segment out
            // is enough.
            for (slot, verdict) in keep.iter_mut().zip(verdicts) {
                *slot &= verdict;
            }
        }

        // Substring pruning, which DataFusion's predicate does not cover.
        // Every requirement must hold, the same way the filters above combine.
        for requirement in &substrings {
            for (slot, segment) in keep.iter_mut().zip(statistics.segments()) {
                *slot &= segment.may_match(requirement);
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

/// Positions of the columns a set of filters mentions.
///
/// Pruning reads bounds and filters for these and no others, so a scan does not
/// pay for statistics on columns nobody asked about.
fn referenced_columns(filters: &[Expr], schema: &SchemaRef) -> Vec<usize> {
    let mut columns: Vec<usize> = filters
        .iter()
        .flat_map(|filter| filter.column_refs())
        .filter_map(|column| schema.index_of(&column.name).ok())
        .collect();
    columns.sort_unstable();
    columns.dedup();
    columns
}
