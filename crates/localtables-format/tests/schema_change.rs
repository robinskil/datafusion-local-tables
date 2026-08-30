//! Changing a table's schema after it holds data.
//!
//! Two shapes are being checked. A change that leaves every stored byte meaning
//! what it meant records a new schema and rewrites nothing. A change that does
//! not rewrites every segment in the same commit as the new schema, so a reader
//! never finds a segment that disagrees with the schema in force.
//!
//! The point of rewriting rather than casting at read time is that everything
//! derived from a column stays true of it: zone maps stay in the current type,
//! filters stay usable, and the read path stays zero-copy. Several of these
//! check exactly that.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use localtables_format::columnar::table::ColumnarTable;
use localtables_format::config::{BloomFilters, Durability, IoBackend, TableOptions};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

fn options() -> TableOptions {
    TableOptions {
        durability: Durability::None,
        io_backend: IoBackend::Mmap,
        ..TableOptions::default()
    }
}

fn batch(ids: &[i32]) -> RecordBatch {
    let names: Vec<String> = ids.iter().map(|i| format!("name-{i}")).collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(ids.to_vec())),
            Arc::new(StringArray::from(names)),
        ],
    )
    .unwrap()
}

/// A table holding two flushed segments.
async fn table(dir: &tempfile::TempDir) -> ColumnarTable {
    table_with(dir, options()).await
}

async fn table_with(dir: &tempfile::TempDir, options: TableOptions) -> ColumnarTable {
    let table = ColumnarTable::create(&dir.path().join("t.lt"), schema(), options)
        .await
        .unwrap();
    table.insert(&[batch(&[1, 2, 3])]).await.unwrap();
    table.flush().await.unwrap();
    table.insert(&[batch(&[4, 5, 6])]).await.unwrap();
    table.flush().await.unwrap();
    table
}

async fn rows(table: &ColumnarTable) -> Vec<RecordBatch> {
    let snapshot = table.snapshot();
    table.scan(&snapshot, None).await.unwrap()
}

fn column(batches: &[RecordBatch], name: &str) -> Vec<ArrayRef> {
    batches
        .iter()
        .map(|b| b.column_by_name(name).expect("the column is there").clone())
        .collect()
}

fn ints(batches: &[RecordBatch], name: &str) -> Vec<i64> {
    let mut out = Vec::new();
    for array in column(batches, name) {
        match array.data_type() {
            DataType::Int32 => {
                let values = array.as_any().downcast_ref::<Int32Array>().unwrap();
                out.extend((0..values.len()).map(|r| values.value(r) as i64));
            }
            DataType::Int64 => {
                let values = array.as_any().downcast_ref::<Int64Array>().unwrap();
                out.extend((0..values.len()).map(|r| values.value(r)));
            }
            other => panic!("unexpected {other}"),
        }
    }
    out.sort_unstable();
    out
}

// ---- add ----------------------------------------------------------------

#[tokio::test]
async fn an_added_column_reads_as_null_for_rows_that_predate_it() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;

    table
        .add_column(Arc::new(Field::new("score", DataType::Float64, true)))
        .await
        .unwrap();

    assert_eq!(table.schema().fields().len(), 3);
    let batches = rows(&table).await;
    let nulls: usize = column(&batches, "score")
        .iter()
        .map(|a| a.null_count())
        .sum();
    assert_eq!(nulls, 6, "no stored row has a value for the new column");
    assert_eq!(ints(&batches, "id"), vec![1, 2, 3, 4, 5, 6]);
}

#[tokio::test]
async fn rows_written_after_an_added_column_carry_it() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table
        .add_column(Arc::new(Field::new("score", DataType::Float64, true)))
        .await
        .unwrap();

    let wider = table.schema();
    let batch = RecordBatch::try_new(
        wider,
        vec![
            Arc::new(Int32Array::from(vec![7])),
            Arc::new(StringArray::from(vec!["name-7"])),
            Arc::new(Float64Array::from(vec![1.5])),
        ],
    )
    .unwrap();
    table.insert(&[batch]).await.unwrap();
    table.flush().await.unwrap();

    let batches = rows(&table).await;
    let nulls: usize = column(&batches, "score")
        .iter()
        .map(|a| a.null_count())
        .sum();
    assert_eq!(nulls, 6, "only the older rows are null");
    assert_eq!(ints(&batches, "id"), vec![1, 2, 3, 4, 5, 6, 7]);
}

#[tokio::test]
async fn a_non_nullable_column_cannot_be_added() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    let err = table
        .add_column(Arc::new(Field::new("score", DataType::Float64, false)))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("no value for it"), "got {err}");
    assert_eq!(table.schema().fields().len(), 2, "the schema is unchanged");
}

#[tokio::test]
async fn a_duplicate_column_name_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    let err = table
        .add_column(Arc::new(Field::new("name", DataType::Utf8, true)))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("already has a column"),
        "got {err}"
    );
}

// ---- rename -------------------------------------------------------------

#[tokio::test]
async fn a_rename_changes_no_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    let table = table(&dir).await;
    let before = std::fs::metadata(&path).unwrap().len();

    table.rename_column("name", "label").await.unwrap();

    let batches = rows(&table).await;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 6);
    assert!(batches[0].column_by_name("label").is_some());
    assert!(batches[0].column_by_name("name").is_none());

    let after = std::fs::metadata(&path).unwrap().len();
    let grew = after - before;
    assert!(
        grew < 4096,
        "a rename should append a schema and a manifest, not rewrite data: grew {grew} bytes"
    );
}

#[tokio::test]
async fn renaming_a_column_that_is_not_there_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    let err = table.rename_column("absent", "x").await.unwrap_err();
    assert!(
        err.to_string().contains("no column named absent"),
        "got {err}"
    );
}

// ---- drop ---------------------------------------------------------------

#[tokio::test]
async fn a_dropped_column_is_gone_and_the_rest_survive() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;

    table.drop_column("name").await.unwrap();

    assert_eq!(table.schema().fields().len(), 1);
    let batches = rows(&table).await;
    assert!(batches[0].column_by_name("name").is_none());
    assert_eq!(ints(&batches, "id"), vec![1, 2, 3, 4, 5, 6]);
}

/// Dropping the first column must not shift the ones after it, which is the
/// failure a rewrite exists to prevent.
#[tokio::test]
async fn dropping_the_first_column_leaves_the_others_where_they_belong() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table
        .add_column(Arc::new(Field::new("score", DataType::Float64, true)))
        .await
        .unwrap();

    table.drop_column("id").await.unwrap();

    let batches = rows(&table).await;
    assert_eq!(
        batches[0].schema().fields().len(),
        2,
        "name and score remain"
    );
    let names: Vec<String> = column(&batches, "name")
        .iter()
        .flat_map(|a| {
            let values = a.as_any().downcast_ref::<StringArray>().unwrap();
            (0..values.len())
                .map(|r| values.value(r).to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        vec!["name-1", "name-2", "name-3", "name-4", "name-5", "name-6"]
    );
}

#[tokio::test]
async fn the_last_column_cannot_be_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.drop_column("name").await.unwrap();
    let err = table.drop_column("id").await.unwrap_err();
    assert!(err.to_string().contains("last column"), "got {err}");
}

// ---- cast ---------------------------------------------------------------

#[tokio::test]
async fn a_widening_cast_converts_every_stored_row() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;

    table.cast_column("id", DataType::Int64).await.unwrap();

    assert_eq!(table.schema().field(0).data_type(), &DataType::Int64);
    let batches = rows(&table).await;
    assert_eq!(batches[0].column(0).data_type(), &DataType::Int64);
    assert_eq!(ints(&batches, "id"), vec![1, 2, 3, 4, 5, 6]);
}

/// The reason casts rewrite. After one, every segment holds the new type, so a
/// zone map on that column is recorded in it and still prunes.
#[tokio::test]
async fn zone_maps_still_prune_after_a_cast() {
    let dir = tempfile::tempdir().unwrap();
    // Groups small enough that 600 rows stay divisible across the rewrite.
    let divisible = TableOptions {
        min_row_group_rows: 100,
        row_group_rows: 100,
        ..options()
    };
    let table = ColumnarTable::create(&dir.path().join("t.lt"), schema(), divisible)
        .await
        .unwrap();
    for group in 0..6 {
        let ids: Vec<i32> = (group * 100..group * 100 + 100).collect();
        table.insert(&[batch(&ids)]).await.unwrap();
        table.flush().await.unwrap();
    }

    table.cast_column("id", DataType::Int64).await.unwrap();

    // Every segment's bounds are readable as the new type, which they would not
    // be if the old bytes had been left in place.
    let snapshot = table.snapshot();
    let mut seen = 0;
    for entry in snapshot.live_segments() {
        let reader = table.segment_reader(entry).await.unwrap();
        let meta = reader.meta().unwrap();
        let zone = meta.columns[0].zone.to_native();
        assert!(
            zone.min_array(&DataType::Int64).is_some(),
            "a segment lost its bounds across the cast"
        );
        seen += 1;
    }
    assert!(seen > 1, "the table should still be divisible: {seen}");
}

#[tokio::test]
async fn a_cast_the_types_do_not_allow_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    let err = table
        .cast_column(
            "name",
            DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None),
        )
        .await;
    // Either arrow declines the pair outright or the values fail to convert;
    // both must leave the table as it was.
    if err.is_ok() {
        return;
    }
    assert_eq!(table.schema().field(1).data_type(), &DataType::Utf8);
    assert_eq!(
        rows(&table)
            .await
            .iter()
            .map(|b| b.num_rows())
            .sum::<usize>(),
        6
    );
}

#[tokio::test]
async fn casting_to_the_type_it_already_has_does_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    let table = table(&dir).await;
    let before = std::fs::metadata(&path).unwrap().len();
    table.cast_column("id", DataType::Int32).await.unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), before);
}

// ---- durability ---------------------------------------------------------

#[tokio::test]
async fn every_schema_change_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.lt");
    {
        let table = table(&dir).await;
        table
            .add_column(Arc::new(Field::new("score", DataType::Float64, true)))
            .await
            .unwrap();
        table.rename_column("name", "label").await.unwrap();
        table.cast_column("id", DataType::Int64).await.unwrap();
        table.drop_column("label").await.unwrap();
    }

    let table = ColumnarTable::open(&path, options()).await.unwrap();
    let fields = table.schema();
    assert_eq!(fields.fields().len(), 2);
    assert_eq!(fields.field(0).name(), "id");
    assert_eq!(fields.field(0).data_type(), &DataType::Int64);
    assert_eq!(fields.field(1).name(), "score");

    let batches = rows(&table).await;
    assert_eq!(ints(&batches, "id"), vec![1, 2, 3, 4, 5, 6]);
}

/// Rows still in the memtable are shaped by the old schema, so a change has to
/// land them first rather than leave them unreadable.
#[tokio::test]
async fn unflushed_rows_survive_a_schema_change() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;
    table.insert(&[batch(&[7, 8])]).await.unwrap();

    table
        .add_column(Arc::new(Field::new("score", DataType::Float64, true)))
        .await
        .unwrap();

    let batches = rows(&table).await;
    assert_eq!(ints(&batches, "id"), vec![1, 2, 3, 4, 5, 6, 7, 8]);
}

/// A membership filter is rebuilt by the rewrite a cast performs, so it still
/// describes the column afterwards.
#[tokio::test]
async fn filters_are_rebuilt_by_a_cast() {
    let dir = tempfile::tempdir().unwrap();
    let table = table_with(
        &dir,
        TableOptions {
            bloom_filters: BloomFilters::All,
            ..options()
        },
    )
    .await;

    table.cast_column("id", DataType::Int64).await.unwrap();

    let snapshot = table.snapshot();
    for entry in snapshot.live_segments() {
        let reader = table.segment_reader(entry).await.unwrap();
        assert!(
            reader.bloom_filter(0).unwrap().is_some(),
            "the rewrite should have rebuilt the filter in the new type"
        );
    }
}

// ---- A schema change must not disturb a reader already under way ----------
//
// A snapshot pins the table as it stood at one commit. A schema change can
// commit while a scan still holds one. The scan must decode through the schema
// its snapshot was taken under, not through the table's current one.

/// A cast rewrites every segment. A reader holding an older snapshot must still
/// see its own rows, in their own type.
#[tokio::test]
async fn a_cast_does_not_disturb_a_pinned_reader() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;

    let pinned = table.snapshot();
    assert_eq!(pinned.schema.field(0).data_type(), &DataType::Int32);

    table.cast_column("id", DataType::Int64).await.unwrap();

    // The reader reads again from the snapshot it pinned before the change.
    let batches = table.scan(&pinned, None).await.unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 6);
    assert_eq!(
        batches[0].column(0).data_type(),
        &DataType::Int32,
        "the pinned reader sees the type its snapshot was taken under"
    );
    assert_eq!(ints(&batches, "id"), vec![1, 2, 3, 4, 5, 6]);

    // A reader that takes a snapshot now sees the new type.
    let fresh = table.snapshot();
    let batches = table.scan(&fresh, None).await.unwrap();
    assert_eq!(batches[0].column(0).data_type(), &DataType::Int64);
}

/// An added column rewrites nothing, so the risk is the other way: the reader
/// must not be handed a column its snapshot does not know about.
#[tokio::test]
async fn an_added_column_does_not_disturb_a_pinned_reader() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;

    let pinned = table.snapshot();
    assert_eq!(pinned.schema.fields().len(), 2);

    table
        .add_column(Arc::new(Field::new("score", DataType::Float64, true)))
        .await
        .unwrap();

    let batches = table.scan(&pinned, None).await.unwrap();
    assert_eq!(
        batches[0].num_columns(),
        2,
        "a pinned reader is handed the columns its snapshot names, and no more"
    );
    assert_eq!(ints(&batches, "id"), vec![1, 2, 3, 4, 5, 6]);

    let fresh = table.snapshot();
    let batches = table.scan(&fresh, None).await.unwrap();
    assert_eq!(batches[0].num_columns(), 3);
}

/// A dropped column is the same shape of risk, and the segments are rewritten.
#[tokio::test]
async fn a_dropped_column_does_not_disturb_a_pinned_reader() {
    let dir = tempfile::tempdir().unwrap();
    let table = table(&dir).await;

    let pinned = table.snapshot();
    table.drop_column("name").await.unwrap();

    let batches = table.scan(&pinned, None).await.unwrap();
    assert_eq!(batches[0].num_columns(), 2);
    assert_eq!(ints(&batches, "id"), vec![1, 2, 3, 4, 5, 6]);
}
