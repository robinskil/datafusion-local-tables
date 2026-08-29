//! Schema storage.
//!
//! The Arrow schema is the one place the format borrows Arrow's IPC encoding.
//! It is written once at creation, read once at open, and never sits on a hot
//! path, so the flatbuffer round trip costs nothing that matters. Column data
//! never touches IPC: it is stored as raw Arrow buffers so a scan can decode
//! one column without touching the others.

use arrow_ipc::convert::{fb_to_schema, IpcSchemaEncoder};
use arrow_ipc::writer::DictionaryTracker;
use arrow_schema::Schema;

use crate::layout::checksum;
use crate::{Error, Result};

/// Encode a schema to flatbuffer bytes.
pub fn encode(schema: &Schema) -> Vec<u8> {
    // Dictionary fields need a tracker to hand out dictionary ids; arrow-ipc
    // panics without one. The ids never leave this buffer, so a fresh tracker
    // per call keeps encoding deterministic.
    let mut tracker = DictionaryTracker::new(false);
    IpcSchemaEncoder::new()
        .with_dictionary_tracker(&mut tracker)
        .schema_to_fb(schema)
        .finished_data()
        .to_vec()
}

/// Decode schema bytes written by [`encode`].
///
/// The bytes come off disk and may be damaged, so this treats them as
/// untrusted. `root_as_schema` verifies the flatbuffer structure, and the
/// unwind guard catches the places inside arrow-ipc that still assert on
/// structurally valid but semantically impossible input.
pub fn decode(bytes: &[u8]) -> Result<Schema> {
    let decoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let root = arrow_ipc::root_as_schema(bytes)
            .map_err(|e| Error::corrupt(format!("schema flatbuffer is malformed: {e}")))?;
        if root.fields().is_none() {
            return Err(Error::corrupt("schema flatbuffer carries no fields"));
        }
        Ok(fb_to_schema(root))
    }));
    match decoded {
        Ok(result) => result,
        Err(_) => Err(Error::corrupt("schema flatbuffer failed to decode")),
    }
}

/// A stable hash of a schema, used to reject a mismatched open.
///
/// Two schemas that encode to the same bytes share a fingerprint. Field order,
/// nullability, types and metadata all take part.
pub fn fingerprint(schema: &Schema) -> u64 {
    checksum(&encode(schema))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, TimeUnit};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn wide_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("score", DataType::Float64, true),
            Field::new("flag", DataType::Boolean, false),
            Field::new("blob", DataType::Binary, true),
            Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
            Field::new(
                "tags",
                DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
                true,
            ),
            Field::new(
                "code",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
        ])
    }

    #[test]
    fn round_trip_preserves_every_field() {
        let schema = wide_schema();
        assert_eq!(decode(&encode(&schema)).unwrap(), schema);
    }

    #[test]
    fn round_trip_preserves_metadata() {
        let schema = Schema::new_with_metadata(
            vec![Field::new("a", DataType::Int32, false)],
            HashMap::from([("owner".to_string(), "robin".to_string())]),
        );
        assert_eq!(decode(&encode(&schema)).unwrap(), schema);
    }

    #[test]
    fn fingerprints_separate_different_schemas() {
        let a = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let b = Schema::new(vec![Field::new("a", DataType::Int64, false)]);
        let c = Schema::new(vec![Field::new("a", DataType::Int32, true)]);
        let reordered = Schema::new(vec![
            Field::new("b", DataType::Int32, false),
            Field::new("a", DataType::Int32, false),
        ]);

        assert_eq!(fingerprint(&a), fingerprint(&a.clone()));
        assert_ne!(fingerprint(&a), fingerprint(&b), "type change");
        assert_ne!(fingerprint(&a), fingerprint(&c), "nullability change");
        assert_ne!(fingerprint(&a), fingerprint(&reordered), "field order");
    }

    #[test]
    fn a_corrupt_flatbuffer_is_rejected_without_panicking() {
        assert!(decode(&[0u8; 32]).is_err());
        assert!(decode(&[]).is_err());

        let mut bytes = encode(&wide_schema());
        for i in (0..bytes.len()).step_by(7) {
            bytes[i] ^= 0xff;
        }
        // Either a clean error or a schema that differs; never a panic.
        let _ = decode(&bytes);
    }
}
