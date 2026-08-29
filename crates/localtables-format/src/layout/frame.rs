//! Self-describing envelope around every rkyv payload on disk.
//!
//! Layout:
//!
//! ```text
//! [ 0.. 8) tag: u64          region magic, catches a misdirected read
//! [ 8..16) len: u64          payload length in bytes
//! [16..24) payload_xxh3: u64
//! [24..32) frame_xxh3: u64   checksum over bytes [0..24)
//! [32..32+len) payload       rkyv archive, 16-byte aligned when the frame is
//! ```
//!
//! The frame header is 32 bytes, so a payload sits at a 16-byte boundary
//! whenever the frame itself starts on one. rkyv needs that alignment to read
//! an archive in place.

use crate::layout::checksum;
use crate::{Error, Result};

/// Bytes the header occupies ahead of the payload.
pub const FRAME_HEADER_LEN: usize = 32;

/// Region tags. Each on-disk structure uses its own, so a read that lands on
/// the wrong offset fails loudly instead of decoding garbage.
pub mod tag {
    pub const HEADER: u64 = u64::from_le_bytes(*b"LTHEADER");
    pub const META: u64 = u64::from_le_bytes(*b"LTMETAPG");
    pub const MANIFEST: u64 = u64::from_le_bytes(*b"LTMANIFE");
    pub const SEGMENT: u64 = u64::from_le_bytes(*b"LTSEGMET");
    pub const DELETES: u64 = u64::from_le_bytes(*b"LTDELVEC");
    pub const SCHEMA: u64 = u64::from_le_bytes(*b"LTSCHEMA");
    pub const WAL_FILE: u64 = u64::from_le_bytes(*b"LTWALHDR");
    pub const WAL_REC: u64 = u64::from_le_bytes(*b"LTWALREC");
}

/// Wrap `payload` in a frame. Returns the full frame bytes.
pub fn encode(tag: u64, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(FRAME_HEADER_LEN + payload.len());
    out.extend_from_slice(&tag.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&checksum(payload).to_le_bytes());
    let frame_hash = checksum(&out[0..24]);
    out.extend_from_slice(&frame_hash.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Total on-disk size of a frame around a payload of `payload_len` bytes.
pub const fn frame_len(payload_len: usize) -> usize {
    FRAME_HEADER_LEN + payload_len
}

/// Read the payload length of a frame without validating the payload itself.
///
/// Use this to learn how many bytes to fetch before reading the whole frame.
pub fn peek_len(bytes: &[u8], expect_tag: u64, region: &'static str) -> Result<usize> {
    if bytes.len() < FRAME_HEADER_LEN {
        return Err(Error::corrupt(format!(
            "{region}: frame header needs {FRAME_HEADER_LEN} bytes, found {}",
            bytes.len()
        )));
    }
    let tag = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let len = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let stored_frame_hash = u64::from_le_bytes(bytes[24..32].try_into().unwrap());

    let found = checksum(&bytes[0..24]);
    if found != stored_frame_hash {
        return Err(Error::Checksum {
            region,
            expected: stored_frame_hash,
            found,
        });
    }
    if tag != expect_tag {
        return Err(Error::BadMagic(format!(
            "{region}: expected tag {expect_tag:#018x}, found {tag:#018x}"
        )));
    }
    Ok(len as usize)
}

/// Validate a frame and return its payload bytes.
///
/// Checks the frame header checksum, the tag, and the payload checksum. The
/// returned slice borrows `bytes`, so a mapped frame stays zero-copy.
pub fn decode<'a>(bytes: &'a [u8], expect_tag: u64, region: &'static str) -> Result<&'a [u8]> {
    let len = peek_len(bytes, expect_tag, region)?;
    let end = FRAME_HEADER_LEN
        .checked_add(len)
        .ok_or_else(|| Error::corrupt(format!("{region}: frame length overflows")))?;
    if bytes.len() < end {
        return Err(Error::corrupt(format!(
            "{region}: frame claims {len} payload bytes, only {} available",
            bytes.len().saturating_sub(FRAME_HEADER_LEN)
        )));
    }
    let payload = &bytes[FRAME_HEADER_LEN..end];
    let stored = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let found = checksum(payload);
    if found != stored {
        return Err(Error::Checksum {
            region,
            expected: stored,
            found,
        });
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_returns_the_payload() {
        let frame = encode(tag::META, b"hello world");
        assert_eq!(frame.len(), frame_len(11));
        assert_eq!(decode(&frame, tag::META, "meta").unwrap(), b"hello world");
    }

    #[test]
    fn payload_starts_aligned_for_rkyv() {
        assert_eq!(FRAME_HEADER_LEN % 16, 0);
    }

    #[test]
    fn wrong_tag_is_rejected() {
        let frame = encode(tag::META, b"payload");
        let err = decode(&frame, tag::MANIFEST, "manifest").unwrap_err();
        assert!(matches!(err, Error::BadMagic(_)), "got {err:?}");
    }

    #[test]
    fn a_flipped_payload_bit_fails_the_checksum() {
        let mut frame = encode(tag::MANIFEST, &[3u8; 200]);
        frame[FRAME_HEADER_LEN + 50] ^= 0x20;
        let err = decode(&frame, tag::MANIFEST, "manifest").unwrap_err();
        assert!(matches!(err, Error::Checksum { .. }), "got {err:?}");
    }

    #[test]
    fn a_torn_header_fails_before_the_payload_is_read() {
        let mut frame = encode(tag::META, &[1u8; 64]);
        frame[10] ^= 0xff; // corrupt the length field
        let err = decode(&frame, tag::META, "meta").unwrap_err();
        assert!(matches!(err, Error::Checksum { .. }), "got {err:?}");
    }

    #[test]
    fn a_truncated_frame_is_rejected() {
        let frame = encode(tag::META, &[9u8; 128]);
        let err = decode(&frame[..80], tag::META, "meta").unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)), "got {err:?}");
    }

    #[test]
    fn an_empty_payload_round_trips() {
        let frame = encode(tag::SCHEMA, &[]);
        assert!(decode(&frame, tag::SCHEMA, "schema").unwrap().is_empty());
    }
}
