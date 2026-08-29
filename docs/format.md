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
