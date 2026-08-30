//! Every structure on disk sits inside a frame. This is the first thing that
//! looks at bytes off the media, so it is the first thing to prove safe.
//!
//! The property: a frame decoder given any bytes returns a payload or an error.
//! It never panics.
#![no_main]

use libfuzzer_sys::fuzz_target;
use localtables_format::layout::frame;
use localtables_format::layout::frame::tag;

fuzz_target!(|data: &[u8]| {
    // Every tag the format uses, so a mismatched tag is covered too.
    for expect in [tag::HEADER, tag::META, tag::SCHEMA, tag::SEGMENT, tag::MANIFEST] {
        if let Ok(payload) = frame::decode(data, expect, "fuzz") {
            // A payload the decoder accepted must lie inside the input.
            assert!(payload.len() <= data.len());
        }
    }
});
