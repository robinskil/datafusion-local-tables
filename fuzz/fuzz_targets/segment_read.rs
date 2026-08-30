//! A whole segment, read as a table would read it.
//!
//! This goes further than the metadata target: it decodes the columns, which
//! means Arrow builds arrays over byte ranges the file chose. A damaged file
//! must produce an error, never a panic and never an array pointing outside
//! its buffers.
#![no_main]

use std::sync::Arc;

use libfuzzer_sys::fuzz_target;
use localtables_format::columnar::segment::SegmentReader;
use localtables_format::io::buf::{IoBuf, SharedBuf};
use localtables_format::layout::schema::SchemaLayout;
use localtables_format::layout::Extent;

use arrow_schema::{DataType, Field, Schema, SchemaRef};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let offset = u16::from_le_bytes([data[0], data[1]]) as u64;
    let len = u16::from_le_bytes([data[2], data[3]]) as u64;
    let bytes = data[4..].to_vec();

    let schema = schema();
    let layout = SchemaLayout::of(&schema);
    let buf = SharedBuf::from_owned(IoBuf::copy_from(&bytes));

    let Ok(reader) = SegmentReader::new(buf, 0, Extent::new(offset, len), schema, &layout) else {
        return;
    };
    // The reader accepted it. Decoding must still not panic.
    let _ = reader.read(None);
    let _ = reader.read(Some(&[0]));
    let _ = reader.read(Some(&[1]));
    let _ = reader.bloom_filter(0);
    let _ = reader.trigram_filter(1);
    let _ = reader.page_zones(0);
});
