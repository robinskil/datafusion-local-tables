# datafusion-local-tables

DataFusion table providers backed by a single local file.

Local files permit things an object store cannot: memory mapping, io_uring, and
handing the page cache straight to a query with no copy in between. This crate
builds two table shapes on that idea.

* **Columnar table** — parquet-like. Zone maps, dictionary and run-length
  encodings, per-page compression. Supports inserts, deletes and updates.
  Built for fast, memory-efficient scans.
* **B-tree table** — copy-on-write B+tree. Built for point lookups and key
  range scans.

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

## Layout

| crate | contents |
| --- | --- |
| `localtables-format` | on-disk format, IO backends, WAL, storage engines |
| `datafusion-local-tables` | the DataFusion `TableProvider` implementations |

## Status

Under construction.

The **columnar table** works end to end: `SELECT`, `INSERT`, `DELETE` and
`UPDATE` through SQL, with zone-map pruning, projection and limit pushdown,
crash-safe commits, a write-ahead log, and compaction.

The **b-tree table** works through SQL for reads: point lookups and range
queries push a key bound into the tree, so `WHERE id = 742` seeks rather than
scans. Writes and deletes go through its Rust API. Underneath is a
copy-on-write tree with the same crash-safe commit protocol.

Three IO backends: mmap (default), positional reads, and io_uring on Linux.
Still to come: SQL writes for the b-tree table, and nested Arrow types. See
`docs/format.md` for the on-disk layout.

## Example

```rust
use datafusion::prelude::SessionContext;
use datafusion_local_tables::{BTreeTableProvider, ColumnarTableProvider};
use localtables_format::{BTreeTable, ColumnarTable, TableOptions};
use std::sync::Arc;

async fn example() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = SessionContext::new();

    let events = ColumnarTable::open("events.lt".as_ref(), TableOptions::default()).await?;
    ctx.register_table("events", Arc::new(ColumnarTableProvider::new(events)))?;

    let users = BTreeTable::open("users.ltb".as_ref(), &["id"], TableOptions::default()).await?;
    ctx.register_table("users", Arc::new(BTreeTableProvider::new(users)))?;

    // Prunes segments by zone map; seeks the b-tree by key.
    ctx.sql(
        "SELECT u.name, count(*) FROM events e \
         JOIN users u ON e.user_id = u.id \
         WHERE e.ts > 1700000000 GROUP BY u.name",
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
