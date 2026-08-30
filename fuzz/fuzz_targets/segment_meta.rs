//! Segment metadata is an rkyv archive read in place, after a checksum.
//!
//! The property: any bytes give metadata or an error, never a panic. rkyv's
//! validation is what stands between a damaged file and a bad pointer, so this
//! is the target that matters most.
#![no_main]

use libfuzzer_sys::fuzz_target;
use localtables_format::columnar::segment::read_meta;
use localtables_format::layout::Extent;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    // The first two bytes choose where the metadata frame sits, so the fuzzer
    // can put it anywhere in the input, including past the end.
    let offset = u16::from_le_bytes([data[0], data[1]]) as u64;
    let len = u16::from_le_bytes([data[2], data[3]]) as u64;
    let bytes = &data[4..];

    let _ = read_meta(bytes, Extent::new(offset, len));
    // And the common case: the frame is the whole input.
    let _ = read_meta(bytes, Extent::new(0, bytes.len() as u64));
});
