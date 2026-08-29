# Measured performance

Numbers from `cargo bench -p datafusion-local-tables`, on an Apple M-series
laptop, 500,000 rows in ten segments of 50,000. The same data is written to a
local table and to a parquet file with matching row groups, and every query
runs through the same DataFusion session, so what is compared is the storage
layer rather than the query engine.

Only compare numbers within one table below. Each table is one benchmark run;
absolute timings drift by tens of percent between runs depending on what else
the machine is doing, so cross-table comparisons are not meaningful.

## Scans

Three tables appear in each result:

* **local** — a local table with the writer's default encodings, which means it
  may choose dictionary or run-length encoding when they make a column smaller;
* **plain** — the same data with re-encoding switched off, so every column is
  stored as raw Arrow buffers;
* **pq** — parquet, read by DataFusion's own reader.

| query | local | plain | parquet |
| --- | --- | --- | --- |
| full scan (`sum(payload)`) | **340 µs** | 351 µs | 732 µs |
| one column (`sum(value)`) | **348 µs** | 364 µs | 760 µs |
| narrow range (10k of 500k rows) | **524 µs** | 543 µs | 687 µs |
| point lookup (`id = 372145`) | 666 µs | **517 µs** | 604 µs |
| group by a string column | 2.07 ms | 1.69 ms | **1.21 ms** |

**Scans are about twice as fast as parquet.** That is the zero-copy read path
doing what it exists for: the segment is mapped, and the Arrow buffers a query
receives point into the page cache. Nothing is decompressed, decoded, or
copied.

**Re-encoding costs reads.** Compare `local` against `plain`: dictionary
encoding a low-cardinality string column makes the file smaller, but the point
lookup pays 29% for it and the group by pays 23%. The column has to be expanded
back to the type the schema declares, and that expansion is a full pass over it.
The writer chooses an encoding purely on the size it saves; it does not know
what a read will cost. Switching re-encoding off with
`TableOptions { dictionary_encoding: false, rle_encoding: false, .. }` is the
right choice for a read-heavy table that fits comfortably on disk.

## Where parquet wins the string group by

Parquet is faster on the group by, and it is worth being precise about why,
because the obvious explanation is wrong.

DataFusion's parquet reader has `schema_force_view_types` on by default, which
turns a `Utf8` column into `Utf8View`. A short string like `cat-3` lives inline
in a view rather than behind an offset, and grouping is much cheaper for it.
Reading the same file both ways, in one run:

| parquet reads the column as | time |
| --- | --- |
| `Utf8View` (the default) | **1.62 ms** |
| `Utf8` | 2.59 ms |

So the reader's choice of type accounts for a 1.6× difference on its own — more
than the whole gap against this crate.

Declaring the column `Utf8View` here does not close it, though. In one run:

| table | column type | time |
| --- | --- | --- |
| local | `Utf8` | **2.29 ms** |
| local | `Utf8View` | 2.74 ms |
| parquet | `Utf8` read as `Utf8View` | 1.92 ms |
| parquet | `Utf8` read as `Utf8` | 2.59 ms |

Two things follow. Against a parquet reader not given view types, this crate is
faster (2.29 against 2.59), which is consistent with the scan results. And
storing the column as `Utf8View` makes it *slower* here rather than faster, so
whatever parquet's reader gains from views, simply handing DataFusion a view
array does not reproduce it. Why is not established, and this document will not
guess again.

## What dictionary columns did and did not do

A column can be declared `Dictionary(Int32, Utf8)` and is then stored and
returned as one, never expanded. That was added on the theory that parquet's
group by advantage came from handing DataFusion a dictionary array. Measured,
in one run:

| table | time |
| --- | --- |
| local, dictionary column | **2.71 ms** |
| parquet, dictionary column | 2.99 ms |

On a dictionary column this crate is now slightly faster than parquet. But both
are slower than the same query over a plain `Utf8` column, so declaring a column
as a dictionary is not a way to make a group by faster — it is a way to make the
file smaller. The theory it was added on was wrong; the capability is still
worth having, because before it a dictionary column could not be stored at all.

## Small writes

`small insert/100 rows` measures a hundred separate one-row inserts, each
durable before the next: about 0.83 ms in total, or 8 µs per insert. That is
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
* The io_uring backend is not in these tables. It is Linux-only and was not run
  on the machine these numbers came from.
