# On-disk format

One table lives in one file. A columnar table may keep two WAL sidecars while
it holds unflushed writes; after a flush the sidecars are empty and the data
file alone is a complete copy of the table.

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

## Alignment

| boundary | value | reason |
| --- | --- | --- |
| Arrow buffers, frames | 64 bytes | widest SIMD register Arrow targets; also satisfies rkyv's 16 |
| segment starts | 4096 bytes | a segment is mapped on its own |

## Locking

The writer takes an exclusive advisory lock on the data file; read-only handles
take a shared one. The lock owns its own file handle, because an advisory lock
belongs to the open file description rather than the descriptor: a lock taken
through a clone of another handle would not release until that handle closed
too.
