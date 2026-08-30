//! Schema storage.
//!
//! The Arrow schema is the one place the format borrows Arrow's IPC encoding.
//! The writer writes it once at creation. A reader reads it once at open. It
//! never sits on a hot path, so the flatbuffer round trip costs nothing.
//!
//! Column data never touches IPC. It sits in raw Arrow buffers, so a scan can
//! decode one column and leave the others alone.

use arrow_ipc::convert::{fb_to_schema, IpcSchemaEncoder};
use arrow_ipc::writer::DictionaryTracker;
use arrow_schema::{FieldRef, Schema};
use std::sync::Arc;

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

/// What a segment's bytes must look like to be read as a given schema.
///
/// A segment holds the columns the schema had when it was written, which is
/// always a prefix of what the schema has now: a column is only ever added at
/// the end without a rewrite, and every change that would move a column
/// rewrites every segment. So a segment with `n` columns is readable exactly
/// when its fingerprint matches the schema's first `n`.
///
/// Names take no part. A segment stores buffers whose meaning comes from a
/// column's type, not its name, so renaming a column must not invalidate data
/// already written. Types, nullability, order and field metadata all do count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaLayout {
    /// Fingerprint of the first `n` columns, indexed by `n`.
    prefixes: Vec<u64>,
}

impl SchemaLayout {
    pub fn of(schema: &Schema) -> Self {
        let fields = schema.fields();
        let prefixes = (0..=fields.len())
            .map(|count| {
                // Names replaced by position, so only the shape is hashed.
                let anonymous: Vec<FieldRef> = fields[..count]
                    .iter()
                    .enumerate()
                    .map(|(at, field)| {
                        Arc::new(field.as_ref().clone().with_name(at.to_string()))
                    })
                    .collect();
                fingerprint(&Schema::new(anonymous))
            })
            .collect();
        Self { prefixes }
    }

    /// The fingerprint a segment written against the whole schema carries.
    pub fn current(&self) -> u64 {
        *self.prefixes.last().expect("a layout covers zero columns")
    }

    pub fn columns(&self) -> usize {
        self.prefixes.len() - 1
    }

    /// Whether a segment holding `columns` columns and stamped `fingerprint`
    /// can be read as this schema.
    pub fn accepts(&self, columns: usize, fingerprint: u64) -> bool {
        self.prefixes.get(columns) == Some(&fingerprint)
    }
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

#[cfg(test)]
mod layout_tests {
    use super::*;
    use arrow_schema::{DataType, Field};

    fn schema(fields: Vec<Field>) -> Schema {
        Schema::new(fields)
    }

    fn two() -> Schema {
        schema(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ])
    }

    #[test]
    fn a_segment_written_against_the_whole_schema_is_accepted() {
        let layout = SchemaLayout::of(&two());
        assert!(layout.accepts(2, layout.current()));
        assert_eq!(layout.columns(), 2);
    }

    /// The case a new column creates: old segments hold a prefix.
    #[test]
    fn a_segment_holding_a_prefix_is_accepted() {
        let before = SchemaLayout::of(&two());
        let after = SchemaLayout::of(&schema(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("score", DataType::Float64, true),
        ]));

        assert!(
            after.accepts(2, before.current()),
            "a segment written before the column was added must still read"
        );
        assert!(after.accepts(3, after.current()));
    }

    /// The case a rename must not create.
    #[test]
    fn renaming_a_column_does_not_invalidate_a_segment() {
        let before = SchemaLayout::of(&two());
        let renamed = SchemaLayout::of(&schema(vec![
            Field::new("identifier", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        assert_eq!(before.current(), renamed.current());
    }

    #[test]
    fn changing_a_type_invalidates_a_segment() {
        let before = SchemaLayout::of(&two());
        let widened = SchemaLayout::of(&schema(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::LargeUtf8, true),
        ]));
        assert!(!widened.accepts(2, before.current()));
    }

    #[test]
    fn changing_nullability_invalidates_a_segment() {
        let before = SchemaLayout::of(&two());
        let tightened = SchemaLayout::of(&schema(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        assert!(!tightened.accepts(2, before.current()));
    }

    #[test]
    fn reordering_columns_invalidates_a_segment() {
        let before = SchemaLayout::of(&two());
        let swapped = SchemaLayout::of(&schema(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("id", DataType::Int64, false),
        ]));
        assert!(!swapped.accepts(2, before.current()));
    }

    /// Dropping the last column leaves old segments holding one column too
    /// many, which is refused: a drop rewrites.
    #[test]
    fn a_segment_with_more_columns_than_the_schema_is_refused() {
        let before = SchemaLayout::of(&two());
        let shorter = SchemaLayout::of(&schema(vec![Field::new("id", DataType::Int64, false)]));
        assert!(!shorter.accepts(2, before.current()));
    }

    #[test]
    fn a_prefix_fingerprint_does_not_match_the_wrong_length() {
        let layout = SchemaLayout::of(&two());
        assert!(!layout.accepts(1, layout.current()));
        assert!(!layout.accepts(0, layout.current()));
    }

    #[test]
    fn an_empty_schema_has_one_prefix() {
        let layout = SchemaLayout::of(&Schema::empty());
        assert_eq!(layout.columns(), 0);
        assert!(layout.accepts(0, layout.current()));
    }
}
