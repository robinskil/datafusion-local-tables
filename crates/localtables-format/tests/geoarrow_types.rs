//! Storing GeoArrow geometries.
//!
//! GeoArrow is a good test of the claim that this format is generic over what
//! it stores, because it is demanding in both ways that matter: its geometries
//! are Arrow extension types, whose identity lives entirely in field metadata,
//! and their storage types nest up to four levels deep — a multipolygon is a
//! list of polygons, each a list of rings, each a list of points, each a fixed
//! size list of coordinates.
//!
//! The arrays here are built to the GeoArrow specification rather than with the
//! `geoarrow` crate, which is pinned to arrow 58 while this workspace is on
//! arrow 59 (the version DataFusion requires) and those do not interoperate.
//! What is under test is the storage layer, so the types and metadata are what
//! matter, and both follow the spec: interleaved coordinates, the child field
//! names the spec mandates, and an `ARROW:extension:name` of `geoarrow.*` with
//! a PROJJSON `crs` in `ARROW:extension:metadata`.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BinaryArray, FixedSizeListArray, Float64Array, ListArray, RecordBatch,
    StructArray,
};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use localtables_format::columnar::segment::{build_segment, SegmentReader};
use localtables_format::config::{Compression, Durability, IoBackend, TableOptions};
use localtables_format::io::open_backend;
use localtables_format::layout::{schema as schema_codec, SEGMENT_ALIGN};

/// A coordinate pair, or a gap where a geometry is absent.
type Xy = (f64, f64);

// ---------------------------------------------------------------- storage types

/// `geoarrow.point`, interleaved: a fixed size list of two doubles.
fn point_type() -> DataType {
    DataType::FixedSizeList(Arc::new(Field::new("xy", DataType::Float64, false)), 2)
}

/// `geoarrow.linestring`: a list of points.
fn linestring_type() -> DataType {
    DataType::List(Arc::new(Field::new("vertices", point_type(), false)))
}

/// `geoarrow.polygon`: a list of rings, each a list of points.
fn polygon_type() -> DataType {
    DataType::List(Arc::new(Field::new("rings", linestring_type(), false)))
}

/// `geoarrow.multipolygon`: a list of polygons. Four levels of nesting.
fn multipolygon_type() -> DataType {
    DataType::List(Arc::new(Field::new("polygons", polygon_type(), false)))
}

/// `geoarrow.point` in the separated layout: a struct of coordinate columns.
fn separated_point_type() -> DataType {
    DataType::Struct(
        vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ]
        .into(),
    )
}

// ---------------------------------------------------------------- array builders

/// A point array. `None` is a missing geometry, which is a null in the list.
fn points(coords: &[Option<Xy>]) -> FixedSizeListArray {
    let mut values = Vec::with_capacity(coords.len() * 2);
    let mut valid = Vec::with_capacity(coords.len());
    for coord in coords {
        // A null geometry still occupies its slot; the coordinates under it are
        // unspecified, which is what Arrow says about values beneath a null.
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

/// Wrap child geometries in a list, one entry per row.
fn nest(
    child_field: &str,
    child_type: DataType,
    groups: &[Option<usize>],
    child: ArrayRef,
) -> ListArray {
    let mut offsets = Vec::with_capacity(groups.len() + 1);
    let mut valid = Vec::with_capacity(groups.len());
    let mut cursor = 0i32;
    offsets.push(0);
    for group in groups {
        cursor += group.unwrap_or(0) as i32;
        offsets.push(cursor);
        valid.push(group.is_some());
    }
    ListArray::new(
        Arc::new(Field::new(child_field, child_type, false)),
        OffsetBuffer::new(offsets.into()),
        child,
        Some(NullBuffer::from(valid)),
    )
}

/// Line strings, given as the vertices of each.
fn linestrings(lines: &[Option<Vec<Xy>>]) -> ListArray {
    let vertices: Vec<Option<Xy>> = lines
        .iter()
        .flatten()
        .flatten()
        .map(|xy| Some(*xy))
        .collect();
    let counts: Vec<Option<usize>> = lines.iter().map(|l| l.as_ref().map(|v| v.len())).collect();
    nest(
        "vertices",
        point_type(),
        &counts,
        Arc::new(points(&vertices)),
    )
}

/// Polygons, given as the rings of each and the vertices of each ring.
fn polygons(shapes: &[Option<Vec<Vec<Xy>>>]) -> ListArray {
    let rings: Vec<Option<Vec<Xy>>> = shapes
        .iter()
        .flatten()
        .flatten()
        .map(|ring| Some(ring.clone()))
        .collect();
    let counts: Vec<Option<usize>> = shapes.iter().map(|s| s.as_ref().map(|r| r.len())).collect();
    nest(
        "rings",
        linestring_type(),
        &counts,
        Arc::new(linestrings(&rings)),
    )
}

/// Multipolygons: a list of polygons per row.
fn multipolygons(shapes: &[Option<Vec<Vec<Vec<Xy>>>>]) -> ListArray {
    let parts: Vec<Option<Vec<Vec<Xy>>>> = shapes
        .iter()
        .flatten()
        .flatten()
        .map(|p| Some(p.clone()))
        .collect();
    let counts: Vec<Option<usize>> = shapes.iter().map(|s| s.as_ref().map(|p| p.len())).collect();
    nest(
        "polygons",
        polygon_type(),
        &counts,
        Arc::new(polygons(&parts)),
    )
}

// ---------------------------------------------------------------- extension metadata

/// A realistic PROJJSON coordinate reference system, as GeoArrow carries it.
const CRS: &str = r#"{"type":"GeographicCRS","name":"WGS 84","datum":{"type":"GeodeticReferenceFrame","name":"World Geodetic System 1984","ellipsoid":{"name":"WGS 84","semi_major_axis":6378137,"inverse_flattening":298.257223563}},"id":{"authority":"EPSG","code":4326}}"#;

/// The metadata that makes a storage type a GeoArrow geometry.
fn geoarrow_metadata(name: &str) -> HashMap<String, String> {
    HashMap::from([
        ("ARROW:extension:name".to_string(), name.to_string()),
        (
            "ARROW:extension:metadata".to_string(),
            format!(r#"{{"crs":{CRS},"crs_type":"projjson","edges":"spherical"}}"#),
        ),
    ])
}

fn geometry_field(name: &str, extension: &str, storage: DataType) -> Field {
    Field::new(name, storage, true).with_metadata(geoarrow_metadata(extension))
}

// ---------------------------------------------------------------- the fixture

/// One column per GeoArrow geometry type this test covers.
fn geo_batch() -> (SchemaRef, RecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        geometry_field("location", "geoarrow.point", point_type()),
        geometry_field("route", "geoarrow.linestring", linestring_type()),
        geometry_field("parcel", "geoarrow.polygon", polygon_type()),
        geometry_field("region", "geoarrow.multipolygon", multipolygon_type()),
        geometry_field("raw", "geoarrow.wkb", DataType::Binary),
        geometry_field("split", "geoarrow.point", separated_point_type()),
    ]));

    let location = points(&[
        Some((4.9041, 52.3676)), // Amsterdam
        None,
        Some((-0.1276, 51.5072)), // London
    ]);

    let route = linestrings(&[
        Some(vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]),
        Some(vec![]), // an empty geometry is not a missing one
        None,
    ]);

    // A square, then a square with a hole in it.
    let square = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
    let hole = vec![(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8), (0.2, 0.2)];
    let parcel = polygons(&[
        Some(vec![square.clone()]),
        None,
        Some(vec![square.clone(), hole.clone()]),
    ]);

    let region = multipolygons(&[
        Some(vec![
            vec![square.clone()],
            vec![square.clone(), hole.clone()],
        ]),
        Some(vec![]),
        None,
    ]);

    // Well-known binary: a point, in little-endian.
    let mut wkb_point = vec![0x01u8, 0x01, 0x00, 0x00, 0x00];
    wkb_point.extend_from_slice(&4.9041f64.to_le_bytes());
    wkb_point.extend_from_slice(&52.3676f64.to_le_bytes());
    let raw = BinaryArray::from(vec![Some(wkb_point.as_slice()), None, Some(&[0x01][..])]);

    let split = StructArray::new(
        match separated_point_type() {
            DataType::Struct(fields) => fields,
            _ => unreachable!("the separated layout is a struct"),
        },
        vec![
            Arc::new(Float64Array::from(vec![4.9041, 0.0, -0.1276])) as ArrayRef,
            Arc::new(Float64Array::from(vec![52.3676, 0.0, 51.5072])) as ArrayRef,
        ],
        Some(NullBuffer::from(vec![true, false, true])),
    );

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(arrow_array::Int64Array::from(vec![1i64, 2, 3])),
            Arc::new(location),
            Arc::new(route),
            Arc::new(parcel),
            Arc::new(region),
            Arc::new(raw),
            Arc::new(split),
        ],
    )
    .unwrap();
    (schema, batch)
}

fn options(compression: Compression, encodings: bool) -> TableOptions {
    TableOptions {
        compression,
        dictionary_encoding: encodings,
        rle_encoding: encodings,
        durability: Durability::None,
        io_backend: IoBackend::Mmap,
        ..TableOptions::default()
    }
}

/// Write the batch as a segment in a real file and read it back.
async fn round_trip(
    schema: &SchemaRef,
    batch: &RecordBatch,
    opts: &TableOptions,
) -> (tempfile::TempDir, SegmentReader) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("geo.lt");
    let io = open_backend(&path, opts.io_backend, opts.durability, false).unwrap();

    let layout = schema_codec::SchemaLayout::of(schema);
    let built = build_segment(
        0,
        schema,
        layout.current(),
        std::slice::from_ref(batch),
        opts,
    )
    .unwrap();

    io.set_len(SEGMENT_ALIGN).await.unwrap();
    let offset = io.append(&[&built.bytes]).await.unwrap();
    let (data, meta) = built.placed(offset);

    let bytes = io.read_immutable(data).await.unwrap();
    let reader = SegmentReader::new(bytes, offset, meta, schema.clone(), &layout).unwrap();
    (dir, reader)
}

// ---------------------------------------------------------------- tests

#[tokio::test]
async fn geoarrow_geometries_round_trip() {
    let (schema, original) = geo_batch();

    for compression in [Compression::None, Compression::Lz4, Compression::Zstd] {
        for encodings in [false, true] {
            let opts = options(compression, encodings);
            let (_dir, reader) = round_trip(&schema, &original, &opts).await;
            assert_eq!(
                reader.read(None).unwrap(),
                original,
                "compression {compression:?}, encodings {encodings}"
            );
        }
    }
}

#[tokio::test]
async fn a_geometry_keeps_the_metadata_that_makes_it_a_geometry() {
    let (schema, original) = geo_batch();
    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, &original, &opts).await;
    let read = reader.read(None).unwrap();

    for (index, field) in schema.fields().iter().enumerate() {
        assert_eq!(
            read.schema().field(index).metadata(),
            field.metadata(),
            "column {} lost its extension metadata",
            field.name()
        );
    }

    // Spot-check that the metadata really is what GeoArrow needs, rather than
    // an empty map matching an empty map.
    let location = read.schema().field(1).metadata().clone();
    assert_eq!(location["ARROW:extension:name"], "geoarrow.point");
    assert!(
        location["ARROW:extension:metadata"].contains("\"EPSG\""),
        "the coordinate reference system must survive intact"
    );
}

#[tokio::test]
async fn nested_geometry_coordinates_come_back_exactly() {
    let (schema, original) = geo_batch();
    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, &original, &opts).await;
    let read = reader.read(None).unwrap();

    // Walk a multipolygon down all four levels and check a coordinate at the
    // bottom. Equality on the batch already covers this, but a failure there
    // says only "the batches differ"; this says which level went wrong.
    let regions = read
        .column(4)
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("multipolygon is a list");
    assert_eq!(regions.len(), 3);
    assert!(regions.is_null(2), "the third region is absent");
    assert_eq!(regions.value(1).len(), 0, "the second region is empty");

    let polygons = regions.value(0);
    let polygons = polygons
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("a polygon is a list of rings");
    assert_eq!(polygons.len(), 2, "the first region holds two polygons");

    let rings = polygons.value(1);
    let rings = rings
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("a ring is a list of points");
    assert_eq!(
        rings.len(),
        2,
        "the second polygon has an outer ring and a hole"
    );

    let hole = rings.value(1);
    let hole = hole
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("a point is a fixed size list");
    let first = hole.value(0);
    let coords = first
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("coordinates are doubles");
    assert_eq!(
        (coords.value(0), coords.value(1)),
        (0.2, 0.2),
        "the first vertex of the hole"
    );
}

#[tokio::test]
async fn an_empty_geometry_is_not_a_missing_one() {
    let (schema, original) = geo_batch();
    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, &original, &opts).await;
    let read = reader.read(None).unwrap();

    let routes = read.column(2).as_any().downcast_ref::<ListArray>().unwrap();
    assert!(!routes.is_null(1), "an empty line string is present");
    assert_eq!(routes.value(1).len(), 0, "and holds no vertices");
    assert!(routes.is_null(2), "the third route is genuinely absent");

    assert_eq!(read.column(2), original.column(2));
}

/// Geometry columns have no order this format can bound, so they prune nothing
/// rather than claiming a bound that could drop a row.
#[tokio::test]
async fn geometry_columns_report_no_zone_map() {
    let (schema, original) = geo_batch();
    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, &original, &opts).await;
    let meta = reader.meta().unwrap();

    for index in 1..schema.fields().len() {
        let zone = meta.columns[index].zone.to_native();
        // Binary has an order, so the WKB column is bounded; the rest are not.
        if schema.field(index).data_type() == &DataType::Binary {
            assert!(!zone.is_unknown(), "well-known binary compares as bytes");
        } else {
            assert!(
                zone.is_unknown(),
                "{} claims a bound its type has no order for",
                schema.field(index).name()
            );
        }
    }

    // The id column still prunes, so the segment is not simply unbounded.
    assert!(!meta.columns[0].zone.to_native().is_unknown());
}

/// A geometry column stored uncompressed still reads with no copy.
#[tokio::test]
async fn geometry_columns_stay_on_the_zero_copy_path() {
    let (schema, original) = geo_batch();
    let opts = options(Compression::None, false);
    let (_dir, reader) = round_trip(&schema, &original, &opts).await;

    assert!(reader.is_zero_copy());
    let meta = reader.meta().unwrap();
    for (index, chunk) in meta.columns.iter().enumerate() {
        assert!(
            chunk.is_zero_copy(),
            "column {} fell off the zero-copy path",
            schema.field(index).name()
        );
    }
}
