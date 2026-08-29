//! Segment metadata: what each column chunk holds and where its bytes are.
//!
//! A segment is written once and never changed. Its bytes look like this:
//!
//! ```text
//! segment start (4096-aligned)
//!   column 0 buffers, each padded to 64 bytes
//!   column 1 buffers
//!   ...
//!   SegmentMeta frame (64-aligned)
//! segment end
//! ```
//!
//! The metadata describes byte ranges; it never wraps the data. That is what
//! lets a scan read one column without touching the others, and lets an
//! uncompressed column become an Arrow array with no copy at all.

use rkyv::{Archive, Deserialize, Serialize};

use crate::layout::Extent;

/// How a column chunk's values are laid out on disk.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[rkyv(derive(Debug), compare(PartialEq))]
#[repr(u8)]
pub enum Encoding {
    /// Arrow's own buffers, byte for byte. The only encoding a scan can read
    /// with no copy at all.
    #[default]
    Plain = 0,
    /// Distinct values once, plus an index per row. Pays off when a column
    /// repeats a small set of values.
    Dictionary = 1,
    /// Run ends plus one value per run. Pays off when equal values are adjacent.
    RunLength = 2,
}

/// Per-buffer compression.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[rkyv(derive(Debug), compare(PartialEq))]
#[repr(u8)]
pub enum Codec {
    /// Stored as is. Keeps the zero-copy read path.
    #[default]
    None = 0,
    Lz4 = 1,
    Zstd = 2,
}

impl Codec {
    pub fn is_none(self) -> bool {
        matches!(self, Codec::None)
    }
}

impl ArchivedEncoding {
    pub fn to_native(&self) -> Encoding {
        match self {
            ArchivedEncoding::Plain => Encoding::Plain,
            ArchivedEncoding::Dictionary => Encoding::Dictionary,
            ArchivedEncoding::RunLength => Encoding::RunLength,
        }
    }
}

impl ArchivedCodec {
    pub fn to_native(&self) -> Codec {
        match self {
            ArchivedCodec::None => Codec::None,
            ArchivedCodec::Lz4 => Codec::Lz4,
            ArchivedCodec::Zstd => Codec::Zstd,
        }
    }
}

impl From<crate::config::Compression> for Codec {
    fn from(value: crate::config::Compression) -> Self {
        match value {
            crate::config::Compression::None => Codec::None,
            crate::config::Compression::Lz4 => Codec::Lz4,
            crate::config::Compression::Zstd => Codec::Zstd,
        }
    }
}

/// What a stored buffer means to the decoder.
///
/// A chunk holds the null bitmap, then Arrow's own buffers for the array in
/// Arrow's own order. What each of those buffers means depends on the type —
/// offsets for a string, values for an integer, indices for a dictionary — and
/// the decoder does not need to know: it hands them back to Arrow in the same
/// order, and Arrow's own validation decides whether they make sense.
///
/// That is what makes the format generic over the type stored. Naming the
/// bitmap separately is still worth it, because it is the one buffer Arrow
/// keeps outside `ArrayData::buffers`.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[rkyv(derive(Debug), compare(PartialEq))]
#[repr(u8)]
pub enum BufferRole {
    /// Null bitmap, one bit per row, starting at bit zero.
    Validity = 0,
    /// One of Arrow's buffers for this array, in Arrow's order.
    Data = 1,
}

/// One stored buffer.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[rkyv(derive(Debug))]
pub struct BufferSpec {
    pub role: BufferRole,
    /// Where the stored bytes live, relative to the start of the segment.
    pub extent: Extent,
    /// Size once decompressed. Equal to `extent.len` when the codec is `None`.
    pub uncompressed_len: u64,
    /// Checksum of the stored bytes, compressed or not.
    pub checksum: u64,
}

/// One column's data inside a segment.
///
/// A chunk can hold chunks, so a dictionary column carries its distinct values
/// and a run-length column its run values. rkyv needs the recursive field
/// exempted from its generated bounds, and the container bounds restated, or
/// the derive recurses forever.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[rkyv(derive(Debug))]
#[rkyv(serialize_bounds(
    __S: rkyv::ser::Writer + rkyv::ser::Allocator,
    __S::Error: rkyv::rancor::Source,
))]
#[rkyv(deserialize_bounds(__D::Error: rkyv::rancor::Source))]
#[rkyv(bytecheck(bounds(
    __C: rkyv::validation::ArchiveContext,
    __C::Error: rkyv::rancor::Source,
)))]
pub struct ColumnChunk {
    pub encoding: Encoding,
    pub codec: Codec,
    /// Rows in the chunk. Always the segment's row count.
    pub len: u64,
    pub null_count: u64,
    /// Distinct values, when the chunk is dictionary encoded. Informational.
    pub dict_len: u64,
    /// Runs, when the chunk is run-length encoded. Informational.
    pub run_count: u64,
    /// Where this array's rows start inside its buffers.
    ///
    /// Normally zero: a sliced array is compacted before it is stored. It is
    /// recorded anyway for the types Arrow cannot compact, where the parent's
    /// buffers are stored as they stand and the offset is what makes them
    /// readable.
    pub offset: u64,
    pub buffers: Vec<BufferSpec>,
    /// Nested chunks. A dictionary chunk holds its distinct values here, a
    /// run-length chunk its run values. Plain chunks have none today; nested
    /// Arrow types will use the same slot.
    #[rkyv(omit_bounds)]
    pub children: Vec<ColumnChunk>,
    pub zone: super::zonemap::ZoneMap,
}

impl ColumnChunk {
    pub fn buffer(&self, role: BufferRole) -> Option<&BufferSpec> {
        self.buffers.iter().find(|b| b.role == role)
    }

    /// Arrow's own buffers, in Arrow's order.
    pub fn data_buffers(&self) -> impl Iterator<Item = &BufferSpec> {
        self.buffers.iter().filter(|b| b.role == BufferRole::Data)
    }

    /// Bytes this chunk occupies on disk, children included.
    pub fn stored_bytes(&self) -> u64 {
        self.buffers.iter().map(|b| b.extent.len).sum::<u64>()
            + self.children.iter().map(|c| c.stored_bytes()).sum::<u64>()
    }

    /// True when the chunk can become an Arrow array with no copy.
    pub fn is_zero_copy(&self) -> bool {
        self.encoding == Encoding::Plain && self.codec.is_none()
    }

    /// Every byte range a scan must fetch to read this chunk.
    pub fn extents(&self) -> Vec<Extent> {
        let mut out = Vec::new();
        self.collect_extents(&mut out);
        out
    }

    fn collect_extents(&self, out: &mut Vec<Extent>) {
        out.extend(self.buffers.iter().map(|b| b.extent));
        for child in &self.children {
            child.collect_extents(out);
        }
    }
}

impl ArchivedColumnChunk {
    pub fn buffer(&self, role: BufferRole) -> Option<&ArchivedBufferSpec> {
        self.buffers.iter().find(|b| b.role == role)
    }

    /// Arrow's own buffers, in Arrow's order.
    pub fn data_buffers(&self) -> impl Iterator<Item = &ArchivedBufferSpec> {
        self.buffers.iter().filter(|b| b.role == BufferRole::Data)
    }

    pub fn is_zero_copy(&self) -> bool {
        self.encoding == Encoding::Plain && self.codec == Codec::None
    }
}

/// The footer of a segment: one entry per column of the table schema.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[rkyv(derive(Debug))]
pub struct SegmentMeta {
    pub segment_id: u64,
    pub row_count: u64,
    /// Matched against the table header, so a segment cannot be read with the
    /// wrong schema after a botched recovery.
    pub schema_fingerprint: u64,
    pub columns: Vec<ColumnChunk>,
}

impl SegmentMeta {
    /// Bytes the column data occupies, excluding this footer.
    pub fn data_bytes(&self) -> u64 {
        self.columns.iter().map(|c| c.stored_bytes()).sum()
    }

    /// True when every column can be read with no copy.
    pub fn is_zero_copy(&self) -> bool {
        self.columns.iter().all(|c| c.is_zero_copy())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::columnar::zonemap::ZoneMap;

    fn spec(role: BufferRole, offset: u64, len: u64) -> BufferSpec {
        BufferSpec {
            role,
            extent: Extent::new(offset, len),
            uncompressed_len: len,
            checksum: 0,
        }
    }

    fn chunk(encoding: Encoding, codec: Codec) -> ColumnChunk {
        ColumnChunk {
            encoding,
            codec,
            len: 100,
            null_count: 3,
            dict_len: 0,
            run_count: 0,
            offset: 0,
            buffers: vec![
                spec(BufferRole::Validity, 0, 16),
                spec(BufferRole::Data, 64, 800),
            ],
            children: Vec::new(),
            zone: ZoneMap::unknown(3),
        }
    }

    #[test]
    fn only_plain_uncompressed_chunks_are_zero_copy() {
        assert!(chunk(Encoding::Plain, Codec::None).is_zero_copy());
        assert!(!chunk(Encoding::Plain, Codec::Lz4).is_zero_copy());
        assert!(!chunk(Encoding::Dictionary, Codec::None).is_zero_copy());
        assert!(!chunk(Encoding::RunLength, Codec::None).is_zero_copy());
    }

    #[test]
    fn buffers_are_found_by_role() {
        let chunk = chunk(Encoding::Plain, Codec::None);
        assert_eq!(chunk.buffer(BufferRole::Data).unwrap().extent.len, 800);
        assert_eq!(chunk.data_buffers().count(), 1);
        assert_eq!(chunk.stored_bytes(), 816);
        assert_eq!(chunk.extents().len(), 2);
    }

    #[test]
    fn a_chunk_accounts_for_its_children() {
        let mut parent = chunk(Encoding::Dictionary, Codec::None);
        parent.buffers = vec![spec(BufferRole::Data, 0, 400)];
        parent.children = vec![chunk(Encoding::Plain, Codec::None)];

        assert_eq!(parent.stored_bytes(), 400 + 816);
        assert_eq!(
            parent.extents().len(),
            3,
            "a scan must fetch the child's buffers too"
        );
    }

    #[test]
    fn segment_meta_round_trips_through_rkyv() {
        let meta = SegmentMeta {
            segment_id: 7,
            row_count: 100,
            schema_fingerprint: 0xdead_beef,
            columns: vec![
                chunk(Encoding::Plain, Codec::None),
                chunk(Encoding::Dictionary, Codec::Zstd),
            ],
        };

        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&meta).unwrap();
        let archived = rkyv::access::<ArchivedSegmentMeta, rkyv::rancor::Error>(&bytes).unwrap();

        assert_eq!(archived.row_count.to_native(), 100);
        assert_eq!(archived.columns.len(), 2);
        assert!(archived.columns[0].is_zero_copy());
        assert!(!archived.columns[1].is_zero_copy());
        assert_eq!(
            archived.columns[0]
                .buffer(BufferRole::Data)
                .unwrap()
                .extent
                .len
                .to_native(),
            800
        );

        let restored: SegmentMeta = rkyv::deserialize::<_, rkyv::rancor::Error>(archived).unwrap();
        assert_eq!(restored, meta);
    }
}
