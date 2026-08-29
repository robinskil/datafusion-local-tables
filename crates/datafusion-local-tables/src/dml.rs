//! Writing to a local table through SQL.
//!
//! `INSERT` streams its input into the table's write-ahead log. `DELETE` and
//! `UPDATE` first work out which rows match, then apply the change as one
//! durable record, so a crash can never leave a delete half done or an update
//! with its rows gone and their replacements missing.
//!
//! Finding the matching rows means reading them. There is no secondary index,
//! so a `DELETE` with a predicate scans the segments the zone maps cannot rule
//! out, exactly as the equivalent `SELECT` would.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DFSchema, DataFusionError, Result, Statistics};
use datafusion::datasource::sink::DataSink;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::Expr;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::MetricsSet;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::StreamExt;

use localtables_format::columnar::table::ColumnarTable;
use localtables_format::layout::manifest::SegmentId;
use localtables_format::Snapshot;

use crate::columnar_exec::to_df_error;

/// The one-column schema DataFusion expects a DML plan to return.
pub(crate) fn count_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::UInt64,
        false,
    )]))
}

/// A batch holding just the row count a DML statement affected.
fn count_batch(count: u64) -> Result<RecordBatch> {
    RecordBatch::try_new(
        count_schema(),
        vec![Arc::new(UInt64Array::from(vec![count]))],
    )
    .map_err(DataFusionError::from)
}

/// Where `INSERT` sends its rows.
pub struct ColumnarDataSink {
    table: ColumnarTable,
    schema: SchemaRef,
}

impl ColumnarDataSink {
    pub fn new(table: ColumnarTable) -> Self {
        let schema = table.schema().clone();
        Self { table, schema }
    }
}

impl fmt::Debug for ColumnarDataSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ColumnarDataSink")
    }
}

impl DisplayAs for ColumnarDataSink {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ColumnarDataSink")
    }
}

#[async_trait]
impl DataSink for ColumnarDataSink {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn metrics(&self) -> Option<MetricsSet> {
        None
    }

    async fn write_all(
        &self,
        mut data: SendableRecordBatchStream,
        _context: &Arc<TaskContext>,
    ) -> Result<u64> {
        let mut written = 0u64;
        while let Some(batch) = data.next().await {
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }
            // One record per batch, each durable before the next is read. The
            // rows are queryable as soon as this returns.
            written += self
                .table
                .insert(std::slice::from_ref(&batch))
                .await
                .map_err(to_df_error)?;
        }
        Ok(written)
    }
}

/// What a `DELETE` or `UPDATE` should change.
#[derive(Debug, Default)]
struct Matches {
    /// Row positions to delete, per segment.
    segments: Vec<(SegmentId, Vec<u32>)>,
    /// Memtable rows to delete, by sequence number.
    memtable: Vec<u64>,
    /// How many rows matched in total.
    count: u64,
}

/// Find every row a predicate selects, in segments and in memory.
///
/// `None` for the predicate means every row, which is what a bare
/// `DELETE FROM t` asks for.
async fn find_matches(
    table: &ColumnarTable,
    snapshot: &Snapshot,
    predicate: Option<&Arc<dyn PhysicalExpr>>,
    mut collect: impl FnMut(&RecordBatch, &BooleanArray) -> Result<()>,
) -> Result<Matches> {
    let mut matches = Matches::default();

    for entry in snapshot.live_segments() {
        let deletes = snapshot.deletes_for(entry.segment_id);
        let reader = table.segment_reader(entry).await.map_err(to_df_error)?;
        let batch = reader.read(None).map_err(to_df_error)?;

        // Positions are relative to the segment as written, so the mask covers
        // rows that are already deleted too; those are skipped rather than
        // counted twice.
        let mask = evaluate(predicate, &batch)?;
        let mut positions = Vec::new();
        for row in 0..batch.num_rows() {
            if !mask.value(row) || mask.is_null(row) {
                continue;
            }
            let position = row as u32;
            if deletes.is_some_and(|dv| dv.is_deleted(position)) {
                continue;
            }
            positions.push(position);
        }

        if !positions.is_empty() {
            matches.count += positions.len() as u64;
            collect(&batch, &row_mask(batch.num_rows(), &positions))?;
            matches.segments.push((entry.segment_id, positions));
        }
    }

    // The memtable's live rows, in the order its sequence numbers list them.
    let seqnos = table.memtable_seqnos().await;
    let mut offset = 0usize;
    for batch in snapshot.memtable.iter() {
        let mask = evaluate(predicate, batch)?;
        let mut hits = Vec::new();
        for row in 0..batch.num_rows() {
            if mask.value(row) && !mask.is_null(row) {
                if let Some(seqno) = seqnos.get(offset + row) {
                    hits.push(*seqno);
                }
            }
        }
        if !hits.is_empty() {
            matches.count += hits.len() as u64;
            let positions: Vec<u32> = (0..batch.num_rows() as u32)
                .filter(|row| mask.value(*row as usize) && !mask.is_null(*row as usize))
                .collect();
            collect(batch, &row_mask(batch.num_rows(), &positions))?;
            matches.memtable.extend(hits);
        }
        offset += batch.num_rows();
    }

    Ok(matches)
}

/// Evaluate a predicate over a batch, or select every row when there is none.
fn evaluate(
    predicate: Option<&Arc<dyn PhysicalExpr>>,
    batch: &RecordBatch,
) -> Result<BooleanArray> {
    let Some(predicate) = predicate else {
        return Ok(BooleanArray::from(vec![true; batch.num_rows()]));
    };
    let value = predicate.evaluate(batch)?;
    let array = value.into_array(batch.num_rows())?;
    array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .cloned()
        .ok_or_else(|| {
            DataFusionError::Internal(format!(
                "a WHERE clause produced {}, not a boolean",
                array.data_type()
            ))
        })
}

/// A mask that is true at each of `positions`.
fn row_mask(rows: usize, positions: &[u32]) -> BooleanArray {
    let mut builder = arrow::array::builder::BooleanBufferBuilder::new(rows);
    builder.append_n(rows, false);
    for position in positions {
        builder.set_bit(*position as usize, true);
    }
    BooleanArray::new(builder.finish(), None)
}

/// A `DELETE`, and the `UPDATE` that reuses it.
///
/// The work happens when the plan is executed, not when it is built, so a
/// statement that is planned but never run changes nothing.
pub struct DmlExec {
    table: ColumnarTable,
    /// The rows to change, or every row when absent.
    predicate: Option<Arc<dyn PhysicalExpr>>,
    /// Column assignments, for an `UPDATE`. Empty for a `DELETE`.
    assignments: Vec<(usize, Arc<dyn PhysicalExpr>)>,
    operation: &'static str,
    properties: Arc<PlanProperties>,
}

impl DmlExec {
    /// Build a `DELETE` plan.
    pub fn delete(table: ColumnarTable, predicate: Option<Arc<dyn PhysicalExpr>>) -> Self {
        Self::new(table, predicate, Vec::new(), "delete")
    }

    /// Build an `UPDATE` plan.
    pub fn update(
        table: ColumnarTable,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        assignments: Vec<(usize, Arc<dyn PhysicalExpr>)>,
    ) -> Self {
        Self::new(table, predicate, assignments, "update")
    }

    fn new(
        table: ColumnarTable,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        assignments: Vec<(usize, Arc<dyn PhysicalExpr>)>,
        operation: &'static str,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(count_schema()),
            datafusion::physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            table,
            predicate,
            assignments,
            operation,
            properties,
        }
    }

    /// Do the work and report how many rows changed.
    async fn run(
        table: ColumnarTable,
        predicate: Option<Arc<dyn PhysicalExpr>>,
        assignments: Vec<(usize, Arc<dyn PhysicalExpr>)>,
    ) -> Result<u64> {
        let snapshot = table.snapshot();

        // An update also needs the matching rows themselves, to build their
        // replacements. A delete throws them away.
        let mut replacements: Vec<RecordBatch> = Vec::new();
        let matches = find_matches(&table, &snapshot, predicate.as_ref(), |batch, mask| {
            if assignments.is_empty() {
                return Ok(());
            }
            let matched = arrow::compute::filter_record_batch(batch, mask)?;
            replacements.push(apply_assignments(&matched, &assignments)?);
            Ok(())
        })
        .await?;

        if matches.count == 0 {
            return Ok(0);
        }

        if assignments.is_empty() {
            return table
                .delete(&matches.segments, &matches.memtable)
                .await
                .map_err(to_df_error);
        }

        // Delete and re-insert as one durable record, so a crash cannot leave
        // the old rows gone and the new ones missing.
        table
            .update(&matches.segments, &matches.memtable, &replacements)
            .await
            .map_err(to_df_error)
    }
}

/// Replace the assigned columns of a batch with their new values.
fn apply_assignments(
    batch: &RecordBatch,
    assignments: &[(usize, Arc<dyn PhysicalExpr>)],
) -> Result<RecordBatch> {
    let mut columns: Vec<Arc<dyn Array>> = batch.columns().to_vec();
    for (index, expr) in assignments {
        let value = expr.evaluate(batch)?.into_array(batch.num_rows())?;
        // SET can produce a wider type than the column holds; the stored type
        // is what the table promises, so the value is cast back to it.
        let field = batch.schema().field(*index).clone();
        columns[*index] = if value.data_type() == field.data_type() {
            value
        } else {
            arrow::compute::cast(&value, field.data_type())?
        };
    }
    RecordBatch::try_new(batch.schema(), columns).map_err(DataFusionError::from)
}

impl fmt::Debug for DmlExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DmlExec: {}", self.operation)
    }
}

impl DisplayAs for DmlExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ColumnarDmlExec: op={}", self.operation)?;
        match &self.predicate {
            Some(predicate) => write!(f, ", predicate={predicate}")?,
            None => write!(f, ", predicate=all rows")?,
        }
        if !self.assignments.is_empty() {
            write!(f, ", assignments={}", self.assignments.len())?;
        }
        Ok(())
    }
}

impl ExecutionPlan for DmlExec {
    fn name(&self) -> &str {
        "ColumnarDmlExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        if let Some(predicate) = &self.predicate {
            if f(predicate)? == TreeNodeRecursion::Stop {
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        for (_, expr) in &self.assignments {
            if f(expr)? == TreeNodeRecursion::Stop {
                return Ok(TreeNodeRecursion::Stop);
            }
        }
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
                "{} runs in one partition, asked for {partition}",
                self.name()
            )));
        }

        let table = self.table.clone();
        let predicate = self.predicate.clone();
        let assignments = self.assignments.clone();
        let stream = futures::stream::once(async move {
            let count = Self::run(table, predicate, assignments).await?;
            count_batch(count)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            count_schema(),
            stream,
        )))
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Arc<Statistics>> {
        // One row, holding the count. How large that count is, is exactly what
        // running the statement decides.
        let mut statistics = Statistics::new_unknown(&count_schema());
        statistics.num_rows = datafusion::common::stats::Precision::Exact(1);
        Ok(Arc::new(statistics))
    }
}

/// Compile a `WHERE` clause into something that can be evaluated per batch.
///
/// `None` means the statement had no clause, which selects every row.
pub(crate) fn compile_predicate(
    state: &dyn Session,
    schema: &SchemaRef,
    filters: &[Expr],
) -> Result<Option<Arc<dyn PhysicalExpr>>> {
    if filters.is_empty() {
        return Ok(None);
    }
    let combined = filters
        .iter()
        .cloned()
        .reduce(datafusion::logical_expr::and)
        .expect("filters is not empty");
    let df_schema = DFSchema::try_from(schema.as_ref().clone())?;
    Ok(Some(state.create_physical_expr(combined, &df_schema)?))
}

/// Compile `SET` assignments, resolving each column name to its position.
pub(crate) fn compile_assignments(
    state: &dyn Session,
    schema: &SchemaRef,
    assignments: &[(String, Expr)],
) -> Result<Vec<(usize, Arc<dyn PhysicalExpr>)>> {
    let df_schema = DFSchema::try_from(schema.as_ref().clone())?;
    assignments
        .iter()
        .map(|(name, expr)| {
            // A qualified name like `t.score` still names one column here.
            let column = name.rsplit('.').next().unwrap_or(name);
            let index = schema.index_of(column).map_err(|_| {
                DataFusionError::Plan(format!("the table has no column named {name}"))
            })?;
            let physical = state.create_physical_expr(expr.clone(), &df_schema)?;
            Ok((index, physical))
        })
        .collect()
}

/// Keeps the unused-import checker honest about `Any`, which the trait bounds
/// on `DataSink` require but no method here names.
const _: fn() = || {
    fn assert_any<T: Any>() {}
    assert_any::<ColumnarDataSink>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_count_batch_holds_one_row() {
        let batch = count_batch(42).unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.schema().field(0).name(), "count");
        assert_eq!(
            batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(0),
            42
        );
    }

    #[test]
    fn a_row_mask_is_true_only_where_asked() {
        let mask = row_mask(5, &[0, 3]);
        assert_eq!(mask.len(), 5);
        assert_eq!(mask.null_count(), 0);
        let values: Vec<bool> = (0..5).map(|i| mask.value(i)).collect();
        assert_eq!(values, vec![true, false, false, true, false]);
    }

    #[test]
    fn an_empty_mask_selects_nothing() {
        let mask = row_mask(3, &[]);
        assert!((0..3).all(|i| !mask.value(i)));
    }
}
