# Measured performance

Numbers from `cargo bench -p datafusion-local-tables`, on an Apple M-series
laptop, 500,000 rows in ten segments of 50,000. The same data is written to a
local table and to a parquet file with matching row groups, and every query
runs through the same DataFusion session, so what is compared is the storage
layer rather than the query engine.

Three tables appear in each result:

* **local** — a local table with the writer's default encodings, which means it
  may choose dictionary or run-length encoding when they make a column smaller;
* **plain** — the same data with re-encoding switched off, so every column is
  stored as raw Arrow buffers;
* **pq** — parquet, read by DataFusion's own reader.

| query | local | plain | parquet |
| --- | --- | --- | --- |
| full scan (`sum(payload)`) | **304 µs** | 305 µs | 609 µs |
| one column (`sum(value)`) | 324 µs | **317 µs** | 619 µs |
| narrow range (10k of 500k rows) | **445 µs** | 450 µs | 589 µs |
| point lookup (`id = 372145`) | 553 µs | **426 µs** | 509 µs |
| group by a string column | 1.85 ms | 1.45 ms | **1.13 ms** |

## What the numbers say

**Scans are about twice as fast as parquet.** That is the zero-copy read path
doing what it exists for: the segment is mapped, and the Arrow buffers a query
receives point into the page cache. Nothing is decompressed, decoded, or
copied.

**Re-encoding costs reads.** Compare `local` against `plain`: dictionary
encoding a low-cardinality string column makes the file much smaller, but the
point lookup pays 30% for it and the group-by pays 28%. The column has to be
expanded back to the type the schema declares, and that expansion is a full
pass over the column. The writer chooses an encoding purely on the size it
saves; it does not know what a read will cost. Switching re-encoding off with
`TableOptions { dictionary_encoding: false, rle_encoding: false, .. }` is the
right choice for a read-heavy table that fits comfortably on disk.

**Group-by on a string column is slower than parquet, even stored plainly.**
Parquet's reader hands DataFusion a dictionary-encoded array for a
low-cardinality string column, and the group-by exploits that directly, hashing
dictionary indices rather than strings. This crate always produces the type the
schema declares, so a `Utf8` column arrives as 500,000 separate strings.

The principled fix is to let a column be declared `Dictionary(Int32, Utf8)` in
the schema and stay dictionary-encoded end to end, at which point no expansion
happens and the group-by gets the same array parquet's reader would give it.
The segment format already stores nested chunks, which is what that needs; the
encoder does not yet accept a dictionary-typed column. Until then, a column
this crate groups on frequently is better stored plainly than dictionary
encoded.

## Small writes

`small insert/100 rows` measures a hundred separate one-row inserts, each
durable before the next: about 1.2 ms in total, or 12 µs per insert. That is
the write-ahead log doing its job — a one-row insert appends a record and
returns, rather than building a segment.

Note that these benchmarks run with `Durability::None`, so no `fsync` is
involved. With a real durability setting the cost per insert is dominated by
the disk, and the group-commit path — several rows in one call, one sync —
matters far more than anything in this crate.

## Reading the caveats

* One machine, one shape of data. A column that compresses differently, a wider
  schema, or a spinning disk would all move these numbers.
* Everything here is warm: the file is in the page cache. The mmap backend's
  advantage is largest exactly there, and smallest on a cold read, where both
  formats wait for the disk.
* The io_uring backend is not in this table. It is Linux-only and was not run
  on the machine these numbers came from.
