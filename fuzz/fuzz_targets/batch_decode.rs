//! Batch payloads inside a log record.
//!
//! A record holds one column at a time, as raw Arrow buffers. Recovery rebuilds
//! arrays from them, so damaged bytes reach Arrow's own validation. Any bytes
//! must give a batch or an error.
#![no_main]

use std::sync::Arc;

use libfuzzer_sys::fuzz_target;
use localtables_format::layout::batchcodec::{self, ArchivedBatchData};

use arrow_schema::{DataType, Field, Schema, SchemaRef};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

fuzz_target!(|data: &[u8]| {
    let Ok(archived) = rkyv::access::<ArchivedBatchData, rkyv::rancor::Error>(data) else {
        return;
    };
    let schema = schema();
    let _ = batchcodec::decode(archived, &schema, None);
    let _ = batchcodec::decode(archived, &schema, Some(&[0]));
    let _ = batchcodec::decode(archived, &schema, Some(&[1]));
});
