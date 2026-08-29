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

Arrow is the in-memory model. Metadata is stored as [rkyv](https://rkyv.org)
archives and read in place. Column data is stored as raw Arrow buffers, so a
scan can map a segment and build a `RecordBatch` over it without copying, and
can decode one column without touching the others.

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

The **b-tree table** works through its Rust API: point lookups, range scans,
writes and deletes, on a copy-on-write tree with the same crash-safe commit
protocol. Its DataFusion provider is not written yet.

Three IO backends: mmap (default), positional reads, and io_uring on Linux.
Still to come: the b-tree table's DataFusion provider, and nested Arrow types.
See `docs/format.md` for the on-disk layout.

## Format stability

The on-disk format is not stable yet. rkyv does not guarantee that archived
layouts survive a version bump, so the file header carries a format version and
an rkyv upgrade counts as a format change. Until version 1.0, expect to rewrite
tables rather than migrate them.

## Licence

Apache-2.0.
