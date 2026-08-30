# On-disk format

One table lives in one file, plus two write-ahead log sidecars named after it
(`table.lt.wal0`, `table.lt.wal1`). After a flush both logs are empty and the
data file alone is a complete copy of the table: copying it is a backup.

## File layout

```
offset 0      FileHeader   4 KiB   magic, format version, table kind, table
                                   uuid, schema extent, schema fingerprint,
                                   alignment rules. Written once, never changed.
offset 4096   MetaPage A   4 KiB   commit slot
offset 8192   MetaPage B   4 KiB   commit slot
offset 12288  data region          schema blob, segments, delete vectors,
                                   manifests
```

Every structure on disk sits inside a frame:

```
[ 0.. 8) tag           region magic, catches a misdirected read
[ 8..16) len           payload length
[16..24) payload_xxh3  checksum of the payload
[24..32) frame_xxh3    checksum of bytes [0..24)
[32..32+len) payload   rkyv archive
```

The header is 32 bytes, so a payload starts on a 16-byte boundary whenever the
frame does. rkyv needs that alignment to read an archive in place.

## Commit protocol

The two meta pages alternate. A commit overwrites the slot holding the *older*
transaction id, so the newer slot stays intact if the write tears.

1. Write the new segments, delete vectors and manifest. Sync.
2. Overwrite the older meta slot with the new transaction id and the manifest
   extent. Sync.
3. Publish the new manifest to readers.

A crash before step 2 leaves bytes past the previous manifest that nothing
points at. The next open reclaims them. A crash during step 2 damages at most
the older slot, and the newer slot still describes a complete table.

Recovery reads both slots, discards any that fail a checksum, and takes the
highest transaction id whose manifest reads back cleanly and agrees with the
slot on its transaction id.

`a_commit_is_atomic_at_every_crash_point` in `table_file.rs` proves this by
sweeping a write budget across a whole commit and checking that every crash
point recovers to either the old state or the new one.

## Space reclaim

A freed byte range joins the manifest's free list tagged with the transaction
that freed it. The allocator hands it out again only once every snapshot older
than that transaction is gone, because those snapshots may hold Arrow buffers
mapped directly onto those bytes.

Two extents are never freed early:

* the schema blob, which the header points at for the life of the table;
* the manifest the *other* meta slot points at, which is what recovery falls
  back to after a torn commit.

Manifests are always placed at the end of the file, never in a reused extent.
A manifest records the file length that includes itself, so it reserves a slot
slightly larger than the record; the frame header carries the real payload
length and the tail is padding.

## Write-ahead log

A small insert should not have to write a segment. It appends a record to the
active log, waits for one sync, and lands in the memtable, where scans see it
immediately. A flush turns the accumulated rows into one segment and empties
the log.

A log file is a 64-byte header followed by framed records:

```
[ header 64 bytes: magic, format version, generation, table uuid, checksum ]
[ frame | frame | frame ... ]        each frame wraps one rkyv WalRecord
```

Three record shapes: `Insert`, `Delete`, `Update`. An update is one record, not
a delete followed by an insert, because a crash between two records must never
leave rows gone and their replacements missing.

Batch payloads use a per-column rkyv record (`layout/batchcodec.rs`), not Arrow
IPC. An IPC message has to be decoded whole before any one column can be read;
here each column is separately addressable, so replay can decode or skip one
column without touching the rest.

Rows are addressed by sequence number, a counter that keeps rising across
flushes. A logged delete names memtable rows by these, so it still finds the
right rows after replay.

### Why two logs

A flush freezes the memtable and switches appends to the other log, so it can
truncate the log it just made durable while new writes carry on. With one file,
truncation would only be safe when nothing was being written — exactly when it
is least needed. The higher generation number marks the active log, and both
opening and recovering must use that same rule: if they disagree, an append can
land in the log a flush is about to truncate.

### Recovery

1. Read the manifest, which carries `checkpoint_lsn`: every record at or below
   it is already inside a segment.
2. Read both logs in generation order. Check each frame; stop at the first that
   fails and truncate the file back to the last good record. A torn tail is the
   normal way a log ends after a crash, not an error.
3. Replay records above the checkpoint, restoring the memtable and the
   deletions at their original sequence numbers.

A damaged record hides every record after it, because their order is what makes
them mean anything.

## What a column can hold

Any Arrow type. A column chunk stores what Arrow lays out — a null bitmap, the
array's buffers in Arrow's order, and its child arrays — and hands them back the
same way. Nothing in the encoder or decoder enumerates types, so nested types,
dictionaries, and extension types all work without being named anywhere.

Extension types need no special handling at all: an extension type is a storage
type plus field metadata, the storage is an ordinary array, and the schema is
stored as Arrow IPC, which preserves metadata. `tests/geoarrow_types.rs` covers
this with GeoArrow geometries, whose multipolygons nest four levels deep.

Two things are still type-specific, and deliberately:

* **Zone maps** are built only for types with an order this format can record —
  the numbers, strings, binary, and the date and time types. Everything else
  reports no bound and prunes nothing, which costs a read and never loses a row.
* **Dictionary and run-length encoding** are only *chosen* for the flat types
  Arrow can cast, and only when they make the column smaller. A column the
  schema already declares as a dictionary is stored that way and never expanded,
  which is what lets a group by hash indices rather than values.

A sliced array is compacted before it is stored, so a batch holding three rows
of a million-row parent stores three rows. Whether an array needs compacting is
decided by comparing the bytes its rows need against the bytes its buffers hold
— a comparison Arrow can make for most types, so this is generic too.

View types are the exception, and the only place the encoder names a type.
Arrow's size accounting counts a view array's views and not the data buffers
behind them, so it reports the same figure for a whole array and for a two-row
slice of one; and concatenating a view array keeps its data buffers as they are,
so it cannot reclaim them either. Both are handled with the operations Arrow
provides for the purpose: `total_buffer_bytes_used` to measure what the views
actually reference, and `gc` to rebuild the buffers around it.

## Segments are row groups

A flush produces segments sized for the table, not one segment per flush. The
size is `total rows / TARGET_ROW_GROUPS`, clamped between `min_row_group_rows`
and `row_group_rows`: a small table gets small groups it can still divide, and a
large one settles at the cap and gains groups rather than bigger ones. A segment is the unit a scan hands to a partition and
the unit a zone map covers, so a flush that made one enormous segment would
leave a reader nothing to divide and nothing to prune — a table written in one
go would scan on a single thread however many were available. Compaction is
bounded the same way, so rewriting cannot undo it.

Batches are kept whole inside a group wherever they fit, because an unsliced
batch is stored straight from Arrow's buffers with nothing copied; a group
closes early rather than slicing one. Only a batch larger than a whole group is
sliced, and only that batch pays a copy.

This is the same granularity parquet readers work at: DataFusion divides a
parquet file by row group, and divides a local table by segment.

Segments are not free, so more is not better: each costs a mapping, a metadata
frame to verify, and a set of zone maps. `docs/performance.md` measures where
that stops paying.

A scan takes segments from a shared queue rather than being dealt a fixed share
when the plan is built. The pieces are not equal — the last group of a flush is
a partial one, compaction leaves uneven ones, a compressed segment costs more to
decode than a mapped one — so a split decided before any of them is read is
guessing.

## Clustered row order

A zone map prunes a column well when the rows are written in that column's
order. Only one column can have that. Sort by `ts` and a segment covers a minute
of time and the whole range of everything else.

A z-order interleaves the leading bits of several columns into one sort key, so
a segment covers a compact box in all of them instead of a narrow slice of one.
Rows are reordered before they are cut into row groups, at flush and at
compaction alike.

A 128 by 128 grid written one row at a time, in eight segments:

| pruned by | written as it arrives | z-ordered on `x`, `y` |
| --- | --- | --- |
| `x = 64` | 0 of 8 | 6 of 8 |
| `y = 64` | 7 of 8 | 4 of 8 |

That is the whole trade. `y` gets worse, because the insert order already
followed it perfectly. `x` goes from unprunable to mostly pruned. A query that
touches either column, or both, reads less.

This is a layout and not an index. It stores no extra bytes, and zone maps are
still built from the values actually written, so a poor key costs reads and
never rows. The key is free to approximate for that reason: each column
contributes eight bytes of its order-preserving encoding, and nulls sort to one
end.

Clustering costs write time, because the rows have to be reordered and that
copies them. Groups come back already gathered, so it is one copy rather than
the two that reordering and then slicing would take.

## Page bounds

A segment's zone map decides whether to read it at all. Page bounds decide which
row ranges inside it a scan hands on: `page_rows` rows per page, recorded per
column, stored as a buffer the decoder skips and read only when a predicate
mentions that column.

A segment of one page or fewer records none, since they would repeat the chunk's
own zone map. Bounds cost roughly a tenth of a percent of a segment.

The delete mask is applied inside a page rather than across the segment, because
filtering the segment first would shift every row and leave the page boundaries
describing the wrong ranges.

Page pruning happens while the scan runs, not when the plan is built, because it
needs bounds stored inside a segment that only a reader of that segment has. It
therefore cannot show in `EXPLAIN`; `EXPLAIN ANALYZE` reports `pages_pruned`
alongside the scan's output rows.

**What this skips is handing a page upward, not decoding it.** A segment is
decoded whole and then sliced, so a page nobody wants still costs whatever its
columns cost to decode. On plain columns that is buffer wrapping and near
nothing; on a dictionary column the whole segment is expanded first.
`docs/performance.md` measures what that leaves on the table.

## Membership filters

A zone map prunes `col = x` only when `x` sits outside a segment's minimum and
maximum. That works for a column written in order and fails for one whose values
are scattered: every segment's range covers the value, nothing is ruled out, and
a point lookup reads the whole table.

A column can carry a membership filter instead, which answers a narrower
question: is this value definitely absent? It never says absent when a value is
present, so acting on that answer cannot lose a row. It does say "may be
present" for values that are not, which costs a segment read.

The layout is the split-block filter parquet uses. All eight bits for one value
land in a single 32-byte block, so a lookup touches one cache line. Measured
false positive rates over 65,536 values:

| bits per value | false positives | filter size |
| --- | --- | --- |
| 6 | 9.9% | 48 KiB |
| 10 | 1.2% | 80 KiB |
| 16 | 0.13% | 128 KiB |

Ten is the default and the knee. Below eight the rate climbs steeply, because a
value sets eight bits and there is no longer room for them; that is what the
single-cache-line layout charges for its locality.

Measured on 500,000 scattered keys, a point lookup goes from 717 to 463 us with
a filter; parquet, given its own filters on the same data, goes from 1.68 ms to
696 us. `docs/performance.md` has the run.

Filters are off unless a column asks for one, because they cost bits for every
value stored. They pay on a column looked up by equality with many distinct
values, and not on a low-cardinality column, where a zone map or a dictionary
already answers.

A filter is stored as a buffer the decoder skips, not inside the metadata
frame, so it costs nothing to open a segment that has one. Pruning reads only
the filters for columns a predicate actually mentions.

Values hash through `valuecodec`, the same canonical byte form for a value in a
column and a literal in a predicate. A literal of a different type is cast
first; one that cannot be cast, or that casts to null, is unknown rather than
absent.

## Trigram filters

A membership filter holds whole values, so it says nothing about
`col LIKE '%ell%'`. A trigram filter holds three-byte pieces of every value
instead:

```text
'hello'  ->  'hel', 'ell', 'llo'
```

A search term is cut the same way, and every one of its pieces must be present
for any row to contain it. One absent piece rules the segment out.

Pieces are bytes rather than characters. UTF-8 is self-synchronising, so a valid
sequence never starts part-way through another one, and byte containment and
text containment agree.

It cannot prove a match, only rule one out. A segment can hold `hel`, `ell` and
`llo` in three different rows and hold `hello` nowhere, and every probe still
passes. That is a second source of false positives on top of the filter's own,
and no number of bits removes it. The scan's own filter decides the answer.

### What it costs

Filters are sized by *distinct* pieces, not by pieces produced. Text repeats its
trigrams heavily, and a filter sized for every repeat would be many times larger
for nothing, since inserting the same item twice adds no information. 100,000
rows, whole-file sizes:

| column | no filter | membership | trigram |
| --- | --- | --- | --- |
| small vocabulary | 3487 KiB | 3732 KiB | 3497 KiB |
| high-entropy ids | 3577 KiB | 3817 KiB | 4137 KiB |

Prose is close to free. Identifiers cost about a sixth of the file, because
nearly every trigram of their alphabet appears.

There are only 2^24 possible trigrams, and a column of near-random bytes
approaches all of them: roughly 21 MiB of filter per chunk, ruling out almost
nothing, since any search term's pieces are then present. A filter that would
come out larger than the column it describes is not written at all.

### What it declines to prune

Each of these would drop segments holding matching rows, so each is refused:

* `NOT LIKE`, which asks whether a value does *not* contain the term;
* `ILIKE`, which matches text the filter never saw, since the filter holds the
  bytes as written;
* a pattern with an `ESCAPE`, where `%` and `_` are ordinary characters and
  splitting on them would invent requirements;
* either side of an `OR`, where a row can match through the other branch.

Both sides of an `AND` are used, since both must hold. A pattern with no run of
three bytes requires nothing and prunes nothing.

DataFusion routes no `LIKE` to its pruning statistics, so this reads the filter
expressions itself. Filters are pushed down as inexact, so the scan keeps a
filter above it and an answer never depends on any of this being right.

## Deletes and compaction

Segments never change once written, so a delete records row positions in a
roaring bitmap rather than rewriting data. A scan applies the bitmap as a mask,
and skips building one at all when nothing is deleted.

Deleted rows still occupy their bytes. Compaction reads the live rows of the
hollowed-out segments, writes them as one new segment, and frees the old ones.
It reads outside the writer lock, then rechecks under it: a delete that landed
meanwhile makes the rewrite stale, so it is abandoned rather than resurrecting
rows the delete removed.

An `UPDATE` is a delete and an insert in one log record, so a crash can never
leave the old rows gone and the new ones missing.

## Changing a table

Filters, clustering and encodings are properties of a *segment*, not of the
file. A segment records what it was written with, and a reader that finds no
filter treats it as no information rather than as an error. So these change by
rewriting, not by any change to the file's header:

```rust
let table = ColumnarTable::open(path, options_you_want).await?;
table.rewrite_all().await?;
```

`rewrite_all` is compaction over every live segment. It applies the current
options to data already stored, which is how a table acquires a membership
filter, a trigram filter or a z-order it was not created with, and how it sheds
one.

### What a rewrite holds

Every rewrite reads stored rows back before writing them out again. Reading all
of them first is simple and unbounded: a table larger than memory could then
never be compacted, and its schema could never change.

Instead the work is cut into runs whose source segments total no more than
`compaction_max_bytes`, measured as the bytes they occupy on disk. A run always
holds at least one segment, so one segment larger than the budget is a run on
its own: a segment is the smallest unit a rewrite can work in.

Compaction commits **once per run**. That keeps the writer lock short and leaves
a valid table at every point: a run that fails leaves the runs before it
compacted and the rest untouched, and running again finishes the job.

A schema change cannot be split that way. A half-converted table would hold
segments of two different types and nothing could read it, so the conversion and
the new schema go into one commit however large the table is. What it bounds is
what it holds while it works, streaming a run at a time into the same commit.
Its reads happen under the writer lock, unlike compaction's, which costs nothing
because a schema change already refuses to run alongside a write.

Clustering is applied within a run, so a table larger than the budget comes out
clustered in runs rather than as a whole. Raising the budget trades memory for a
better layout.

Rewriting to cluster by one column costs another its locality, since there is
only one row order. A trigram filter on text that was grouped by segment before
the rewrite can come out pruning nothing afterwards, still correct and no longer
useful.

### Changing the schema

The header records the schema a table was created with and is never rewritten.
The manifest records the one in force, and every commit rewrites that, so a
schema change is an ordinary commit.

Changes come in two shapes:

| change | cost | why |
| --- | --- | --- |
| add a nullable column | one small commit | old segments hold a prefix of the schema and read the new column as null |
| rename a column | one small commit | a segment's bytes mean what its column *types* say, and names take no part in that |
| drop a column | rewrites every segment | a segment addresses columns by position, so a drop would shift the ones after it |
| change a column's type | rewrites every segment | see below |

A column is added at the end, because anywhere else would move the columns after
it, which is the same problem a drop has.

**A cast rewrites rather than converting at read time.** Converting lazily would
work, and it would cost the column its zero-copy path on every scan for as long
as the table lived, and it would leave every zone map on that column recorded in
the old type and so unusable for pruning. Rewriting pays once and leaves
everything downstream true of the column: bounds, membership filters, trigram
filters and the buffers themselves are all in the current type.

The rewrite and the schema go into **one commit**. There is never an instant
where the manifest says `Int64` and a segment holds `Int32`, so a reader can
always trust that a segment matches the schema it is being read under, and a
crash part-way through leaves the old type and the old data.

A change flushes first, so the memtable and the log never hold rows shaped by a
schema that is no longer in force.

## Alignment

| boundary | value | reason |
| --- | --- | --- |
| Arrow buffers, frames | 64 bytes | widest SIMD register Arrow targets; also satisfies rkyv's 16 |
| segment starts | 4096 bytes | a segment is mapped on its own |


## IO backends

| backend | reads | notes |
| --- | --- | --- |
| `mmap` (default) | map the segment, no syscall, no copy | Arrow buffers point into the page cache |
| `pread` | positional reads on a blocking pool | portable; the reference the others are checked against |
| `uring` | one submission per segment projection | Linux only, `uring` feature |

The io_uring backend exists for one thing: a scan of a segment needs every
projected column's byte ranges, which is one syscall each through `pread` and
one submission for all of them through io_uring. Its ring is owned by a
dedicated thread rather than integrated with tokio's driver, which keeps the
unsafe part small. Writes go through positional writes in every backend,
because the write path is sequential appends followed by a sync.

Asking for a backend this build cannot provide is an error, never a silent
downgrade to another one.

## Locking

The writer takes an exclusive advisory lock on the data file; read-only handles
take a shared one. The lock owns its own file handle, because an advisory lock
belongs to the open file description rather than the descriptor: a lock taken
through a clone of another handle would not release until that handle closed
too.
