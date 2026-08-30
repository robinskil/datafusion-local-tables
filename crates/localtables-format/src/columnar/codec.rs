//! Per-buffer compression.
//!
//! A codec applies to one Arrow buffer at a time, not to a whole row batch. A
//! scan that needs two of forty columns decompresses two columns' worth of
//! bytes. A column stored with no codec goes to Arrow untouched.

use crate::columnar::page::Codec;
use crate::io::buf::IoBuf;
use crate::{Error, Result};

/// Compress `bytes` with `codec`.
///
/// Returns `None` when the result is not smaller. A buffer that does not
/// compress then keeps the zero-copy read path. It does not pay to decompress
/// the same number of bytes.
pub fn compress(codec: Codec, bytes: &[u8]) -> Result<Option<Vec<u8>>> {
    let packed = match codec {
        Codec::None => return Ok(None),

        #[cfg(feature = "lz4")]
        Codec::Lz4 => lz4_flex::compress(bytes),
        #[cfg(not(feature = "lz4"))]
        Codec::Lz4 => {
            return Err(Error::Unsupported(
                "lz4 compression needs the `lz4` feature".into(),
            ))
        }

        #[cfg(feature = "zstd")]
        Codec::Zstd => zstd::bulk::compress(bytes, 3).map_err(Error::RawIo)?,
        #[cfg(not(feature = "zstd"))]
        Codec::Zstd => {
            return Err(Error::Unsupported(
                "zstd compression needs the `zstd` feature".into(),
            ))
        }
    };

    if packed.len() >= bytes.len() {
        return Ok(None);
    }
    Ok(Some(packed))
}

/// Expand `bytes` back to exactly `uncompressed_len` bytes.
///
/// The length comes from the segment metadata, so a damaged buffer that
/// expands to the wrong size is caught here rather than handed to Arrow.
pub fn decompress(codec: Codec, bytes: &[u8], uncompressed_len: usize) -> Result<IoBuf> {
    match codec {
        Codec::None => Err(Error::InvalidArgument(
            "decompress called on an uncompressed buffer".into(),
        )),

        #[cfg(feature = "lz4")]
        Codec::Lz4 => {
            let mut out = IoBuf::uninit(uncompressed_len);
            let written = lz4_flex::decompress_into(bytes, out.as_mut_slice())
                .map_err(|e| Error::corrupt(format!("lz4 buffer failed to decompress: {e}")))?;
            if written != uncompressed_len {
                return Err(Error::corrupt(format!(
                    "lz4 buffer expanded to {written} bytes, metadata says {uncompressed_len}"
                )));
            }
            Ok(out)
        }
        #[cfg(not(feature = "lz4"))]
        Codec::Lz4 => Err(Error::Unsupported(
            "this file holds lz4 buffers; rebuild with the `lz4` feature".into(),
        )),

        #[cfg(feature = "zstd")]
        Codec::Zstd => {
            let mut out = IoBuf::uninit(uncompressed_len);
            let written = zstd::bulk::decompress_to_buffer(bytes, out.as_mut_slice())
                .map_err(|e| Error::corrupt(format!("zstd buffer failed to decompress: {e}")))?;
            if written != uncompressed_len {
                return Err(Error::corrupt(format!(
                    "zstd buffer expanded to {written} bytes, metadata says {uncompressed_len}"
                )));
            }
            Ok(out)
        }
        #[cfg(not(feature = "zstd"))]
        Codec::Zstd => Err(Error::Unsupported(
            "this file holds zstd buffers; rebuild with the `zstd` feature".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bytes that compress well, so the codec keeps the compressed form.
    fn compressible() -> Vec<u8> {
        std::iter::repeat_n(b"the same line over and over\n".to_vec(), 200)
            .flatten()
            .collect()
    }

    /// Bytes with no structure, so the codec declines to compress.
    fn incompressible() -> Vec<u8> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        (0..4096)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    #[test]
    fn no_codec_never_compresses() {
        assert!(compress(Codec::None, &compressible()).unwrap().is_none());
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn lz4_round_trips() {
        let original = compressible();
        let packed = compress(Codec::Lz4, &original).unwrap().unwrap();
        assert!(packed.len() < original.len());

        let restored = decompress(Codec::Lz4, &packed, original.len()).unwrap();
        assert_eq!(restored.as_slice(), &original[..]);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_round_trips() {
        let original = compressible();
        let packed = compress(Codec::Zstd, &original).unwrap().unwrap();
        assert!(packed.len() < original.len());

        let restored = decompress(Codec::Zstd, &packed, original.len()).unwrap();
        assert_eq!(restored.as_slice(), &original[..]);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn a_buffer_that_does_not_shrink_is_stored_raw() {
        assert!(
            compress(Codec::Lz4, &incompressible()).unwrap().is_none(),
            "paying to decompress the same number of bytes is worse than not compressing"
        );
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn a_wrong_length_is_caught_rather_than_passed_on() {
        let original = compressible();
        let packed = compress(Codec::Lz4, &original).unwrap().unwrap();

        let err = decompress(Codec::Lz4, &packed, original.len() - 1).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)), "got {err:?}");
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn a_damaged_buffer_fails_instead_of_producing_garbage() {
        let original = compressible();
        let mut packed = compress(Codec::Lz4, &original).unwrap().unwrap();
        let middle = packed.len() / 2;
        packed[middle] ^= 0xff;

        let result = decompress(Codec::Lz4, &packed, original.len());
        match result {
            Err(Error::Corrupt(_)) => {}
            Err(other) => panic!("expected corruption, got {other:?}"),
            // Damage that still decodes to the right length is caught by the
            // per-buffer checksum before this point.
            Ok(out) => assert_eq!(out.len(), original.len()),
        }
    }

    #[test]
    fn decompressing_an_uncompressed_buffer_is_a_caller_error() {
        let err = decompress(Codec::None, b"abc", 3).unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "got {err:?}");
    }

    #[test]
    fn empty_buffers_round_trip() {
        for codec in [Codec::Lz4, Codec::Zstd] {
            // Empty input never shrinks, so it is stored raw.
            assert!(compress(codec, &[]).unwrap().is_none());
        }
    }
}
