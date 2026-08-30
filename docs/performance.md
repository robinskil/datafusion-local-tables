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

Against a parquet reader not given view types, this crate is faster (2.29
against 2.59), which is consistent with the scan results. But storing the column
as `Utf8View` makes it slower here rather than faster, which is worth
explaining. `tests/profile_views.rs` takes it apart.

### Why storing a view column is slower

Three effects, measured separately. Values are five bytes unless stated.

**The column is bigger.** A view is a flat sixteen bytes per row whatever the
value; an offset plus five bytes of data is nine. So the `Utf8View` column is
8.0 MB against 4.5 MB — 1.78x — and every read pays for it. At forty-byte
values the gap narrows to 1.27x, and the read cost narrows with it, which is
what says the size is doing the work.

**Validating UTF-8 costs more for the view layout.** Decoding checks what came
off disk rather than trusting it, and for a string column that includes UTF-8.
For `Utf8` that is one pass over a contiguous buffer; for `Utf8View` it is a
separate check per value. Comparing types that differ in nothing else:

| column type | stored bytes | decode |
| --- | --- | --- |
| `Utf8` | 4,500,040 | 1.14 ms |
| `Binary` | 4,500,040 | 0.59 ms |
| `Utf8View` | 8,000,000 | 2.47 ms |
| `BinaryView` | 8,000,000 | **0.89 ms** |

`BinaryView` and `Utf8View` hold byte-identical data and differ only in whether
Arrow has to check it is UTF-8. `BinaryView` decodes 2.8x faster. That check is
the whole of the cost that size does not explain.

**Views do make grouping cheaper — while values stay short.** Grouping cost
alone: 0.58 ms for `Utf8View` against 0.92 ms for `Utf8`. At forty-byte values,
where nothing fits inline in a view any more, it reverses: 1.59 ms against
1.13 ms.

Added up, the first two exceed the third, and the view column loses. Parquet
does not face this trade because its stored representation and the type it
hands back are unrelated: it stores compact, dictionary-encoded pages and
materialises views in memory, so it takes the grouping win without paying for
sixteen bytes a row on disk. Doing the same here would mean choosing the output
type at scan time rather than storing what the schema says — a real design
option, and not one this crate takes today.

There is also a lever on the validation cost: every buffer already carries an
xxh3 checksum that is verified before decode, so Arrow's revalidation is
redundant for a file this crate wrote. Skipping it with `build_unchecked` would
recover most of that 2.8x. It is not taken, and should not be taken lightly: a
checksum proves the bytes are the bytes that were written, not that they were
ever valid, and unchecked construction turns a bad file into undefined
behaviour rather than an error.

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

## Membership filters

The point lookup in the table above is on an ordered id, which zone maps already
prune to one segment. A column of scattered values is the hard case: every
segment's range covers the key, no range test rules any of them out, and the
lookup reads the whole table.

The same 500,000 rows, keyed by a permutation of `0..500000` so every segment
spans the range, all four variants built and measured in one run:

| variant | `WHERE key = 372145` |
| --- | --- |
| local, no filter | 717 µs |
| local, filter | **463 µs** |
| parquet, no filter | 1.68 ms |
| parquet, filter | 696 µs |

**A filter is worth about a third of the query.** 717 to 463 µs here, from
ruling out nine segments of ten. Parquet gains more from its own filters, 1.68
ms to 696 µs, because it had further to fall: without them its reader was doing
more work per surviving row group than this one.

What is left after filtering is one segment of 50,000 rows, read in full to find
one row. That bound is segment granularity, not the filter, and page-level zone
maps are what would lower it.

This is the only comparison here where both formats were given the same feature.
It is worth more than the others for that reason: `bloom_filter_on_read` is on
by default in DataFusion, so parquet was reading its filters, and both files
were written with matching row groups.

## Substring search

A zone map says nothing about what a value contains, so `LIKE` prunes nothing
without a trigram filter. 500,000 rows in ten segments, one of which holds the
term, all three built in one run:

| variant | `body LIKE '%shard7 record%'` |
| --- | --- |
| local, no filter | 2.65 ms |
| local, trigram filter | **996 µs** |
| parquet | 4.12 ms |

**A trigram filter is worth 2.7 times on this query.** Parquet is the reference
for reading everything, since it has no substring index to switch on: this is
not a like-for-like comparison the way the point lookup above is.

The filter costs little on text and more on identifiers, because it is sized by
distinct trigrams and prose repeats its own heavily. Whole-file sizes over
100,000 rows: a small-vocabulary column grows 3487 to 3497 KiB, and a column of
high-entropy ids grows 3577 to 4137 KiB. `docs/format.md` covers the bound that
keeps a pathological column from writing a filter at all.

## Clustered row order

A 707 by 707 grid, written one row at a time, so the rows arrive ordered by `y`:

| query | as written | z-ordered on `x`, `y` |
| --- | --- | --- |
| `x = 354` | 634 µs | **508 µs** |
| `y = 354` | **538 µs** | 598 µs |
| both | **694 µs** | 713 µs |

**This is a trade and the timings show it as one.** `x` gains 20%, `y` loses
11%, and querying both is a wash. Segment pruning moves much further than the
clock does: for `x` it goes from 0 of 8 segments to 6 of 8. At half a million
rows a segment is cheap enough that ruling out six of them saves only a fifth of
the query.

That gap is the useful part of this measurement. Clustering pays in proportion
to what a segment costs to read, so it should matter more on wider rows, larger
tables, and reads that miss the page cache. **None of those are measured here**,
and this benchmark is a warm, memory-mapped, half-million-row table. Do not read
the 20% as the number for a larger one in either direction.

Clustering also costs write time, since the rows have to be reordered. That is
not measured either.

## Page bounds inside a segment

Bounds recorded for each row range inside a segment let a scan decode and hand
on one page rather than the whole segment. Measured on 500,000 rows in a single
segment, so a segment zone map rules nothing out and page bounds are the only
thing left to prune with. All six built in one run:

| variant | `WHERE id = 372145` |
| --- | --- |
| local, encodings on, no page bounds | 3.92 ms |
| local, encodings on, page bounds | **1.48 ms** |
| local, plain columns, no page bounds | 2.58 ms |
| local, plain columns, page bounds | 2.04 ms |
| parquet | 2.86 ms |
| parquet, page index | 0.61 ms |

**Page bounds are worth 2.65x on a table with encodings on.** They are worth
much less on plain columns, 1.27x, and the difference between those two rows is
the whole mechanism: what a page skips is the work of *expanding* it. A plain
column is buffer wrapping and costs almost nothing to skip; a dictionary column
has to be turned back into the type the schema declares, and that is work
proportional to the rows expanded.

That only became true when the decoder learned to build a range. Before it did,
a segment was decoded whole and then sliced, and these same bounds bought
nothing at all: 6.04 ms against 5.87 ms unpruned, in a run of their own. Slicing
a dictionary array before the expansion rather than after is the entire
difference, and it is four lines.

Parquet's page index is still ahead, 0.61 against 1.48 ms. The gap was 6.5x
before the decoder changed and is 2.4x now.

**These numbers cannot be compared with the ones in the paragraph above them.**
The machine drifts tens of percent between runs: parquet measured 6.20 ms in the
run that produced the old figures and 2.86 ms in this one, on an identical file.
Only the rows inside one table are comparable with each other. An earlier
version of this section reported page bounds ahead by 14% on encoded columns
before the decoder changed; that was noise across exactly this kind of gap.

## Which columns are worth compressing

A codec is never free here: the read path is zero-copy only while a column is
stored raw. Compressing it trades the mapped bytes for a buffer the reader owns
and a pass to fill it. So the question is not how small a codec makes a column
but whether that column had anything to gain.

Ratios over 500,000 rows, uncompressed over compressed:

| column | lz4 | zstd |
| --- | --- | --- |
| random u64 | 1.00x | 1.00x |
| sequential i64 | 2.0x | 7.7x |
| text | 4.4x | 24x |

**Scattered numbers do not compress, with either codec.** The tidy figures for
sequential integers are a property of the fixture, not of numeric data.

End to end, same 500,000 rows, four settings built in one run:

| setting | file | full scan | string group by | read every column |
| --- | --- | --- | --- | --- |
| raw | 13808 KiB | 311 us | 1.81 ms | 407 us |
| **auto** | **11866 KiB** | 312 us | 1.78 ms | 416 us |
| lz4 everywhere | 7976 KiB | 315 us | 1.78 ms | 580 us |
| zstd everywhere | 4309 KiB | 1.17 ms | 1.79 ms | 1.23 ms |

`Auto` compresses text and binary and leaves the rest raw. It is 14% smaller
than raw and 2% slower on the worst of the three queries, which is why it is the
default. Compressing everything reaches 42% and 69% smaller and costs 42% and
202% on a read of every column.

Two things worth noticing. The string group by is unmoved by any setting,
because that column is low cardinality and already dictionary encoded, so its
values buffer is tiny whatever wraps it. And lz4 everywhere costs almost nothing
on the full scan, because the column it reads did not shrink and the writer
therefore stored it raw: a codec that fails to earn its place is dropped per
buffer, which protects the zero-copy path without anyone choosing anything.

### Blocks, and what they cost

A compressed column is cut into `compression_block_rows` blocks so a scan can
decompress a range without decompressing the column. The size that costs depends
entirely on the codec, because it depends on how far back the codec looks. lz4
looks 64 KiB whatever it is given; zstd looks much further and loses more when
cut off.

| setting | file, one block per column | file, 8192-row blocks | point lookup, whole | blocked |
| --- | --- | --- | --- | --- |
| auto (lz4 on text) | 11866 KiB | 11893 KiB | 406 us | 421 us |
| zstd everywhere | 4309 KiB | 4556 KiB | 1.24 ms | **500 us** |

**Blocking costs the default 0.23% and buys zstd 2.5x.** It is close to free for
lz4, which is why it is on by default, and it is what makes zstd usable at all
for a selective query: whole-column zstd was three times slower than storing
raw, and in blocks it is 21% slower at a third of the size.

The block size is separate from `page_rows` on purpose. A zone map costs bytes
and no processor time, so pruning can be finer than decompression; the defaults
happen to match, so a page a predicate rules out also costs nothing to
decompress.

### How small a block is worth making

A smaller block means a selective query decompresses less, and it means the
codec sees less at a time. Both were measured on a table whose bulk is text that
does not dictionary encode, 500,000 distinct values in one segment:

| setting | file | one row | every row |
| --- | --- | --- | --- |
| raw | 23153 KiB | 3.41 ms | **2.91 ms** |
| lz4, 8192-row blocks | 9942 KiB | **333 us** | 7.54 ms |
| zstd, one block | 3297 KiB | 14.31 ms | 11.55 ms |
| zstd, 8192-row blocks | 4028 KiB | 470 us | 13.43 ms |
| zstd, 2048-row blocks | 4205 KiB | 542 us | 13.91 ms |
| zstd, 512-row blocks | 4558 KiB | 870 us | 17.75 ms |

**Blocking is worth 30x on a selective query and costs 22% of the file.** That
is the whole case for it: whole-column zstd takes 14.31 ms to return one row and
470 us in blocks.

**Below the page size, smaller blocks are worse on both counts.** Going from
8192 to 512 rows makes the file 13% larger *and* the query 85% slower. Pruning
selects whole pages, so a block finer than a page only means decompressing
several of them and joining the results. Block size should not go below
`page_rows`, and the defaults match for that reason.

**Compression is a trade against full scans, not a free win.** Reading every row
costs 2.91 ms raw, 7.54 ms with lz4 and 13.43 ms with zstd. A workload of
selective queries wants compression; a workload of full scans does not.

### Building an unblocked variable-width column is not free

The raw row in that table is the odd one: 3.41 ms to return one row, ten times
what lz4 takes. Projecting only the integer column costs 423 us, so the string
column accounts for about 3 ms of it — to produce one 8192-row page.

The cost is Arrow's validation. `ArrayData::build` checks a variable-width
array's offsets across the whole buffer, and for `Utf8` checks the bytes are
valid text as well, and both are proportional to the whole column rather than
the range being read. Storing the same bytes as `Binary`, which has no text
rule, costs 2.27 ms instead of 2.95: the text check is about a quarter of it and
offset validation is the rest.

A blocked column escapes this by accident, because only the block being read is
built. It means the reasoning behind leaving uncompressed columns unblocked —
that blocking would cost them their zero-copy path and buy nothing — is wrong
for variable-width types, which are paying an O(column) cost per read today.
Fixed-width columns are unaffected: their validation is O(1).

### Compression dictionaries, and why there are none

A dictionary is the usual answer to small blocks compressing badly: give the
codec content it can reference as though it were prior history. It was measured
and it does not work here.

500,000 distinct email-shaped strings, 16 MiB, against compressing the column as
one block:

| block | zstd alone | + content dict | + trained dict | lz4 alone | + content dict |
| --- | --- | --- | --- | --- | --- |
| 4 KiB | +89.5% | +112.0% | +156.1% | +8.6% | +4.7% |
| 17 KiB | +72.9% | +93.9% | +96.9% | +1.6% | −1.3% |
| 70 KiB | +74.5% | +86.2% | +78.4% | +1.5% | +0.5% |
| 280 KiB | +56.5% | +61.4% | +54.8% | +0.4% | +0.1% |

**A dictionary makes zstd worse at every block size tried**, and the trained one
is worse than the raw content one at the sizes where it matters. It reaches
break-even only at 280 KiB blocks, where blocking costs least and a dictionary
is needed least.

For lz4 a content dictionary does help, and the help is worth 0.3% at the block
size this format uses, while costing 30% on decompression: 49.6 us per block
becomes 64.6 us. That is not a trade worth any machinery.

An earlier note in this file said a trained dictionary got within 4.6% of
whole-column compression. That was measured on a fixture whose text cycled
through eight values, which a trained dictionary captures almost perfectly and
real text does not. The table above is the corrected figure and the synthetic
one should not be relied on.

The reason is that a dictionary earns its place when a block has no history of
its own to reference. Blocks here are sized in rows, so even 512 rows of text is
17 KiB and already carries plenty. Dictionaries belong to formats whose blocks
are single-digit kilobytes.

One thing the table does say: **zstd loses far more to blocking on real text
than the benchmark fixture suggests.** That fixture's only string column is low
cardinality and dictionary encoded, so its values buffer is tiny; blocking cost
it 5.7%. A column of genuinely distinct text costs 56.5% at the default block
size. A table with text columns that wants zstd should raise
`compression_block_rows`, and give up some of the skipping to get it back.

## Parallel scans

A segment is the unit a scan gives to a partition. Partitions take segments from
a shared queue rather than being dealt a fixed share, so one that draws a cheap
segment comes back for another.

How many segments a table should hold is a real trade, and both sides of it are
measurable. Scanning the same 500,000 rows cut different ways, all in one run:

| segments | 1 partition | 4 partitions | 8 partitions |
| --- | --- | --- | --- |
| 5 | 428 µs | 317 µs | 338 µs |
| 10 | 461 µs | **303 µs** | **301 µs** |
| 20 | 532 µs | 324 µs | 348 µs |
| 70 | 736 µs | 458 µs | 572 µs |

A segment is not free: a mapping, a metadata frame to verify, a set of zone
maps, about five microseconds each. Read down the one-partition column and that
cost is plain — 428 µs to 736 µs for the same rows. Read across and the benefit
is plain too, until there are so many segments that their overhead swamps it.
The flat region here is five to twenty; `TARGET_ROW_GROUPS` is set to eight to
land in it.

More partitions than four buys nothing on this query, and that is not about
segments. Fitting `total = fixed + work / threads` puts roughly 280 µs of it in
costs that do not divide — planning, the final aggregation, spawning the tasks —
and only about 175 µs in the scan. The scan part divides as expected; it is the
smaller half of a query this cheap.

### Uneven segments

The shape a shared queue is meant for: half the rows in one segment and the rest
spread over twenty small ones.

| target partitions | time |
| --- | --- |
| 1 | 531 µs |
| 4 | **359 µs** |
| 8 | 406 µs |

It divides, but this does not establish that taking from a queue beat the fixed
split it replaced — that would need both implementations measured side by side,
and only one of them still exists. The argument for it is that the pieces are
genuinely unequal, so a split fixed before any of them is read is guessing; the
evidence here is only that the result is correct and not slower.

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
