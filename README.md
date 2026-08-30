# datafusion-local-tables

DataFusion table providers backed by a single local file.

A local file permits what an object store cannot. It permits memory maps,
io_uring, and the page cache given straight to a query with no copy between.

The table is parquet-like. It has zone maps, membership filters, dictionary and
run-length encodings, and per-column compression. It takes inserts, deletes and
updates. It is built for fast scans that use little memory.

A scan divides across threads at segment granularity. A segment is this format's
row group, the same unit a parquet reader divides a file by.

Arrow is the in-memory model. Metadata sits in [rkyv](https://rkyv.org) archives
and reads in place. Column data sits in raw Arrow buffers. A scan can map a
segment and build a `RecordBatch` over it with no copy. It can also decode one
column and leave the others alone.

A column can hold any Arrow type. The format stores what Arrow lays out: a null
bitmap, the array's buffers, and its child arrays. Nested types, dictionaries
and extension types all work, and nothing enumerates them.

GeoArrow geometries round-trip, with four levels of nesting and their extension
metadata. Zone maps stay type-specific. A type with no order this format can
record prunes nothing.

Zone maps prune by range. That leaves two cases they cannot help with. On a
column of scattered values, every segment's range covers the value. A substring
search is not a range at all.

A column can ask for a membership filter for the first case. It can ask for a
trigram filter for the second.

Row order is a third lever. A zone map is selective on a column only when the
rows follow that column's order, and only one column can have that. A z-order
interleaves several columns, so a segment covers a box in all of them.

## Layout

| crate | contents |
| --- | --- |
| `localtables-format` | on-disk format, IO backends, WAL, storage engine |
| `datafusion-local-tables` | the DataFusion `TableProvider` implementations |

## Status

Under construction.

The table works end to end. It takes `SELECT`, `INSERT`, `DELETE` and `UPDATE`
through SQL. It prunes by zone map, membership filter and trigram filter. It
also offers z-order clustering, projection and limit pushdown, crash-safe
commits, a write-ahead log, and compaction.

A segment fixes its filters and its row order when the writer writes it. A table
changes them by a rewrite. Open the table with the options you want, then call
`rewrite_all`.

A rewrite is cut into runs of bounded size. It does not read the whole table
into memory, so a table larger than memory can still be compacted.

The schema can change too. To add a nullable column, or to rename one, is one
small commit. To drop a column, or to change its type, rewrites every segment in
the same commit as the schema. A segment therefore always matches the schema it
is read under, and zone maps and filters stay usable.

There are three IO backends: mmap by default, positional reads, and io_uring on
Linux. The io_uring backend compiles for Linux. Nobody has run it. See
`docs/format.md` for the on-disk layout.

An earlier version carried a second table kind: a copy-on-write b-tree for point
lookups. It was removed. This build cannot open a file that version wrote.

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

This runs the same queries against a local table and an equivalent parquet
file, in one DataFusion session. The measurement is of the storage layer, not
the query engine.

Scans come out about twice as fast as parquet. That is the zero-copy read path
doing what it exists for.

A group-by on a string column is slower. DataFusion's parquet reader turns the
column into `Utf8View`, and this crate returns the type the schema declares. See
`docs/performance.md` for the numbers, and for what they do and do not
establish.

## Format stability

The on-disk format is not stable yet. rkyv does not promise that an archived
layout survives a version bump. So the file header carries a format version, and
an rkyv upgrade counts as a format change.

Before version 1.0, rewrite tables. Do not expect to migrate them.

## Licence

Apache-2.0.
