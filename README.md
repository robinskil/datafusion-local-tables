# datafusion-local-tables

DataFusion table providers backed by a single local file.

Local files permit things an object store cannot: memory mapping, io_uring, and
handing the page cache straight to a query with no copy in between.

The table is parquet-like: zone maps, membership filters, dictionary and
run-length encodings, per-page compression. It supports inserts, deletes and
updates, and is built for fast, memory-efficient scans.

Scans divide across threads at segment granularity, which is this format's row
group — the same unit a parquet reader divides a file by.

Arrow is the in-memory model. Metadata is stored as [rkyv](https://rkyv.org)
archives and read in place. Column data is stored as raw Arrow buffers, so a
scan can map a segment and build a `RecordBatch` over it without copying, and
can decode one column without touching the others.

A column can hold any Arrow type. The format stores what Arrow lays out — a
null bitmap, the array's buffers, its child arrays — so nested types,
dictionaries and extension types work without being enumerated anywhere.
GeoArrow geometries round-trip, four levels of nesting and extension metadata
included. Zone maps stay type-specific: a type with no order this format can
record simply prunes nothing.

Zone maps prune by range, which leaves two cases they cannot help with. A column
of scattered values has every segment's range covering the value being looked
for, and a substring search is not a range at all. A column can opt into a
membership filter for the first and a trigram filter for the second.

Row order is a third lever. A zone map is selective on a column only when the
rows follow that column's order, and only one column can have that; writing rows
in z-order interleaves several columns so a segment covers a box in all of them.

## Layout

| crate | contents |
| --- | --- |
| `localtables-format` | on-disk format, IO backends, WAL, storage engine |
| `datafusion-local-tables` | the DataFusion `TableProvider` implementations |

## Status

Under construction.

The table works end to end: `SELECT`, `INSERT`, `DELETE` and `UPDATE` through
SQL, with zone-map, membership-filter and trigram pruning, optional z-order
clustering, projection and limit pushdown, crash-safe commits, a write-ahead
log, and compaction.

Filters and clustering are per-segment, so a table changes them by being
rewritten: reopen with the options you want and call `rewrite_all`.

The schema can change too. Adding a nullable column and renaming one are single
small commits; dropping a column and changing its type rewrite every segment, in
the same commit as the schema, so a segment always matches the schema it is read
under and zone maps and filters stay usable.

Three IO backends: mmap (default), positional reads, and io_uring on Linux.
The io_uring backend compiles for Linux but has not been run. See
`docs/format.md` for the on-disk layout.

An earlier version carried a second table kind, a copy-on-write b-tree for
point lookups. It was removed: a file written by it cannot be opened by this
build.

## Example

```rust
use datafusion::prelude::SessionContext;
use datafusion_local_tables::ColumnarTableProvider;
use localtables_format::{BloomFilters, ColumnarTable, TableOptions};
use std::sync::Arc;

async fn example() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = SessionContext::new();

    // `user_id` is looked up by equality, so it gets a membership filter.
    let options = TableOptions::default()
        .with_bloom_filters(BloomFilters::Columns(vec!["user_id".to_string()]));

    let events = ColumnarTable::open("events.lt".as_ref(), options).await?;
    ctx.register_table("events", Arc::new(ColumnarTableProvider::new(events)))?;

    // The range prunes by zone map, the equality by membership filter.
    ctx.sql(
        "SELECT count(*) FROM events \
         WHERE ts > 1700000000 AND user_id = 8143",
    )
    .await?
    .show()
    .await?;

    Ok(())
}
```

## Benchmarks

```bash
cargo bench -p datafusion-local-tables
```

Runs the same queries against a local table and an equivalent parquet file
through the same DataFusion session, so what is measured is the storage layer
rather than the query engine.

Scans come out about twice as fast as parquet, which is the zero-copy read path
doing what it exists for. Group-by on a string column is slower, because
DataFusion's parquet reader turns the column into `Utf8View` and this crate
returns the type the schema declares. See `docs/performance.md` for the numbers
and what they do and do not establish.

## Format stability

The on-disk format is not stable yet. rkyv does not guarantee that archived
layouts survive a version bump, so the file header carries a format version and
an rkyv upgrade counts as a format change. Until version 1.0, expect to rewrite
tables rather than migrate them.

## Licence

Apache-2.0.
