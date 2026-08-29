//! Querying a table that holds GeoArrow geometries.
//!
//! Storing an extension type is only half of it. This checks the other half:
//! that a geometry column survives a table, a flush, a reopen and a SQL query,
//! keeping both its nested values and the metadata that says what it is.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float64Array, Int64Array, ListArray, RecordBatch};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::prelude::SessionContext;

use datafusion_local_tables::ColumnarTableProvider;
use localtables_format::{ColumnarTable, Durability, IoBackend, TableOptions};

/// `geoarrow.point`, interleaved.
fn point_type() -> DataType {
    DataType::FixedSizeList(Arc::new(Field::new("xy", DataType::Float64, false)), 2)
}

/// `geoarrow.linestring`: a list of points.
fn route_type() -> DataType {
    DataType::List(Arc::new(Field::new("vertices", point_type(), false)))
}

fn geo_metadata(name: &str) -> HashMap<String, String> {
    HashMap::from([
        ("ARROW:extension:name".to_string(), name.to_string()),
        (
            "ARROW:extension:metadata".to_string(),
            r#"{"crs":{"id":{"authority":"EPSG","code":4326}}}"#.to_string(),
        ),
    ])
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("city", DataType::Utf8, false),
        Field::new("location", point_type(), true).with_metadata(geo_metadata("geoarrow.point")),
        Field::new("route", route_type(), true).with_metadata(geo_metadata("geoarrow.linestring")),
    ]))
}

fn points(coords: &[Option<(f64, f64)>]) -> FixedSizeListArray {
    let mut values = Vec::new();
    let mut valid = Vec::new();
    for coord in coords {
        let (x, y) = coord.unwrap_or((0.0, 0.0));
        values.push(x);
        values.push(y);
        valid.push(coord.is_some());
    }
    FixedSizeListArray::new(
        Arc::new(Field::new("xy", DataType::Float64, false)),
        2,
        Arc::new(Float64Array::from(values)),
        Some(NullBuffer::from(valid)),
    )
}

fn routes(lines: &[Vec<(f64, f64)>]) -> ListArray {
    let vertices: Vec<Option<(f64, f64)>> = lines.iter().flatten().map(|xy| Some(*xy)).collect();
    let mut offsets = vec![0i32];
    let mut cursor = 0i32;
    for line in lines {
        cursor += line.len() as i32;
        offsets.push(cursor);
    }
    ListArray::new(
        Arc::new(Field::new("vertices", point_type(), false)),
        OffsetBuffer::new(offsets.into()),
        Arc::new(points(&vertices)),
        None,
    )
}

fn batch(start: i64) -> RecordBatch {
    let cities = ["Amsterdam", "London", "Paris"];
    let coords = [Some((4.9041, 52.3676)), None, Some((2.3522, 48.8566))];
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(vec![start, start + 1, start + 2])),
            Arc::new(arrow::array::StringArray::from(cities.to_vec())),
            Arc::new(points(&coords)),
            Arc::new(routes(&[
                vec![(0.0, 0.0), (1.0, 1.0)],
                vec![],
                vec![(2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
            ])),
        ],
    )
    .unwrap()
}

fn options() -> TableOptions {
    TableOptions {
        durability: Durability::None,
        io_backend: IoBackend::Mmap,
        memtable_max_bytes: 64 * 1024 * 1024,
        ..TableOptions::default()
    }
}

async fn query(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    ctx.sql(sql).await.unwrap().collect().await.unwrap()
}

#[tokio::test]
async fn geometries_survive_a_table_a_flush_and_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cities.lt");

    {
        let table = ColumnarTable::create(&path, schema(), options())
            .await
            .unwrap();
        table.insert(&[batch(1)]).await.unwrap();
        table.flush().await.unwrap();
        // These stay in the log, so the reopen has to replay a geometry too.
        table.insert(&[batch(10)]).await.unwrap();
    }

    let table = ColumnarTable::open(&path, options()).await.unwrap();
    let ctx = SessionContext::new();
    ctx.register_table("cities", Arc::new(ColumnarTableProvider::new(table)))
        .unwrap();

    let rows = query(
        &ctx,
        "SELECT id, city, location, route FROM cities ORDER BY id",
    )
    .await;
    let total: usize = rows.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 6, "three rows from the segment, three from the log");

    // The metadata that makes these columns geometries is still on the schema.
    let read_schema = rows[0].schema();
    assert_eq!(
        read_schema.field(2).metadata()["ARROW:extension:name"],
        "geoarrow.point"
    );
    assert_eq!(
        read_schema.field(3).metadata()["ARROW:extension:name"],
        "geoarrow.linestring"
    );

    // And the coordinates are intact, two levels down.
    let locations = rows[0]
        .column(2)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    let first = locations.value(0);
    let coords = first.as_any().downcast_ref::<Float64Array>().unwrap();
    assert_eq!((coords.value(0), coords.value(1)), (4.9041, 52.3676));
    assert!(locations.is_null(1), "a missing location stays missing");
}

#[tokio::test]
async fn a_query_can_filter_and_project_around_a_geometry_column() {
    let dir = tempfile::tempdir().unwrap();
    let table = ColumnarTable::create(&dir.path().join("cities.lt"), schema(), options())
        .await
        .unwrap();
    table.insert(&[batch(1)]).await.unwrap();
    table.flush().await.unwrap();

    let ctx = SessionContext::new();
    ctx.register_table("cities", Arc::new(ColumnarTableProvider::new(table)))
        .unwrap();

    // Projecting past the geometry: it is not read at all.
    let rows = query(&ctx, "SELECT city FROM cities WHERE id = 3").await;
    assert_eq!(rows[0].num_columns(), 1);
    assert_eq!(
        rows[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap()
            .value(0),
        "Paris"
    );

    // Projecting only the geometry.
    let rows = query(&ctx, "SELECT location FROM cities WHERE id = 1").await;
    assert_eq!(rows[0].num_columns(), 1);
    assert_eq!(rows[0].schema().field(0).data_type(), &point_type());

    // Counting rows where a geometry is absent, which is an ordinary null.
    let rows = query(&ctx, "SELECT count(*) FROM cities WHERE location IS NULL").await;
    assert_eq!(
        rows[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        1
    );
}

#[tokio::test]
async fn a_geometry_column_can_be_deleted_and_updated_around() {
    let dir = tempfile::tempdir().unwrap();
    let table = ColumnarTable::create(&dir.path().join("cities.lt"), schema(), options())
        .await
        .unwrap();
    table.insert(&[batch(1)]).await.unwrap();
    table.flush().await.unwrap();

    let ctx = SessionContext::new();
    ctx.register_table("cities", Arc::new(ColumnarTableProvider::new(table)))
        .unwrap();

    query(&ctx, "DELETE FROM cities WHERE id = 2").await;
    let rows = query(&ctx, "SELECT id FROM cities ORDER BY id").await;
    let ids: Vec<i64> = rows
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(ids, vec![1, 3]);

    // An update rewrites whole rows, so the geometry has to make the round trip
    // through the log and back.
    query(&ctx, "UPDATE cities SET city = 'Lutetia' WHERE id = 3").await;
    let rows = query(&ctx, "SELECT city, location FROM cities WHERE id = 3").await;
    assert_eq!(
        rows[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap()
            .value(0),
        "Lutetia"
    );
    let locations = rows[0]
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    let coords = locations.value(0);
    let coords = coords.as_any().downcast_ref::<Float64Array>().unwrap();
    assert_eq!(
        (coords.value(0), coords.value(1)),
        (2.3522, 48.8566),
        "the geometry must survive being rewritten by an update"
    );
}
