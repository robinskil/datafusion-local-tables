//! Shared on-disk primitives: magic numbers, alignment rules, byte extents.

pub mod batchcodec;
pub mod frame;
pub mod header;
pub mod manifest;
pub mod schema;

use rkyv::{Archive, Deserialize, Serialize};

/// File magic. Spells `DFLT` followed by a format tag.
pub const MAGIC: u64 = u64::from_le_bytes(*b"DFLT\0\0\0\x01");

/// Bump on any change that makes older readers wrong.
pub const FORMAT_VERSION: u32 = 3;

/// Size of the header and of each meta page.
pub const HEADER_SIZE: u64 = 4096;
pub const META_PAGE_SIZE: u64 = 4096;

/// Byte offset of meta page slot A.
pub const META_A_OFFSET: u64 = HEADER_SIZE;
/// Byte offset of meta page slot B.
pub const META_B_OFFSET: u64 = HEADER_SIZE + META_PAGE_SIZE;
/// First byte available for segments, delete vectors and manifests.
pub const DATA_START: u64 = HEADER_SIZE + 2 * META_PAGE_SIZE;

/// Alignment for any byte range Arrow reads directly.
///
/// 64 bytes covers the widest SIMD register Arrow targets and satisfies rkyv's
/// 16-byte archive requirement at the same time.
pub const BUFFER_ALIGN: u64 = 64;

/// Alignment for the start of a segment, so it can be mapped on its own.
pub const SEGMENT_ALIGN: u64 = 4096;

/// Which engine wrote this file.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[rkyv(derive(Debug), compare(PartialEq))]
#[repr(u8)]
pub enum TableKind {
    Columnar = 0,
}

/// A contiguous byte range inside the table file.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[rkyv(derive(Debug), compare(PartialEq))]
pub struct Extent {
    pub offset: u64,
    pub len: u64,
}

impl Extent {
    pub const EMPTY: Extent = Extent { offset: 0, len: 0 };

    pub fn new(offset: u64, len: u64) -> Self {
        Self { offset, len }
    }

    pub fn end(&self) -> u64 {
        self.offset + self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Byte range as `usize`, for slicing a mapping.
    pub fn range(&self) -> std::ops::Range<usize> {
        self.offset as usize..(self.offset + self.len) as usize
    }

    /// Byte range relative to `base`, for slicing inside a segment mapping.
    pub fn range_from(&self, base: u64) -> std::ops::Range<usize> {
        let start = (self.offset - base) as usize;
        start..start + self.len as usize
    }

    pub fn overlaps(&self, other: &Extent) -> bool {
        self.offset < other.end() && other.offset < self.end()
    }
}

impl ArchivedExtent {
    pub fn to_native(&self) -> Extent {
        Extent {
            offset: self.offset.into(),
            len: self.len.into(),
        }
    }
}

/// Round `value` up to the next multiple of `align`. `align` must be a power of two.
#[inline]
pub const fn align_up(value: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

/// Bytes of padding needed to reach the next `align` boundary.
#[inline]
pub const fn padding_to(value: u64, align: u64) -> u64 {
    align_up(value, align) - value
}

/// Hash bytes with the checksum this format uses everywhere.
#[inline]
pub fn checksum(bytes: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(bytes)
}

/// Compare a stored checksum against the bytes it covers.
pub fn verify_checksum(region: &'static str, bytes: &[u8], expected: u64) -> crate::Result<()> {
    let found = checksum(bytes);
    if found != expected {
        return Err(crate::Error::Checksum {
            region,
            expected,
            found,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_rounds_to_boundary() {
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
        assert_eq!(padding_to(65, 64), 63);
        assert_eq!(padding_to(64, 64), 0);
    }

    #[test]
    fn data_start_clears_header_and_meta_pages() {
        assert_eq!(META_A_OFFSET, 4096);
        assert_eq!(META_B_OFFSET, 8192);
        assert_eq!(DATA_START, 12288);
        assert_eq!(align_up(DATA_START, SEGMENT_ALIGN), DATA_START);
    }

    #[test]
    fn extent_overlap_is_symmetric() {
        let a = Extent::new(0, 100);
        let b = Extent::new(50, 100);
        let c = Extent::new(100, 100);
        assert!(a.overlaps(&b) && b.overlaps(&a));
        assert!(!a.overlaps(&c) && !c.overlaps(&a));
    }

    #[test]
    fn checksum_detects_a_single_flipped_bit() {
        let mut bytes = vec![7u8; 512];
        let before = checksum(&bytes);
        bytes[300] ^= 1;
        assert_ne!(before, checksum(&bytes));
    }
}
