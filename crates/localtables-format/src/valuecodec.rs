//! One canonical byte form per Arrow value.
//!
//! Membership filters hash values, and a value has to hash the same whether it
//! arrived as a row of a column or as a literal inside a predicate. That is
//! what this provides: one value, one byte string, whatever route it took.
//!
//! The encoding also orders: the bytes compare with `memcmp` the way the values
//! compare. That takes work per type — signed integers need their sign bit
//! flipped, floats need their sign and magnitude rearranged, strings need an
//! escape so a shorter string never looks larger than one it prefixes — and
//! nothing depends on it today. It is kept because it is what makes the
//! encoding canonical in the first place: an order-preserving encoding cannot
//! give one value two spellings, which is exactly the property a filter needs.
//!
//! Every encoding here is reversible.

use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, TimeUnit};

use crate::{Error, Result};

/// Tag byte in front of every value, so null sorts before everything.
const NULL_TAG: u8 = 0x00;
const VALUE_TAG: u8 = 0x01;

/// Escapes inside a variable-length value.
///
/// A zero byte becomes `00 FF`, and the value ends with `00 00`. That makes the
/// terminator the smallest thing that can follow, so `"ab"` sorts before
/// `"abc"` as it must.
const ESCAPE: u8 = 0x00;
const ESCAPED: u8 = 0xFF;
const TERMINATOR: u8 = 0x00;

/// Append one column's value at `row` to `out`.
pub fn encode_value(out: &mut Vec<u8>, array: &dyn Array, row: usize) -> Result<()> {
    if array.is_null(row) {
        out.push(NULL_TAG);
        return Ok(());
    }
    out.push(VALUE_TAG);

    macro_rules! signed {
        ($ty:ty, $native:ty) => {{
            let value = array.as_primitive::<$ty>().value(row);
            // Flipping the sign bit maps the signed range onto the unsigned one
            // in order, so big-endian bytes then compare correctly.
            let biased = (value as $native).wrapping_sub(<$native>::MIN)
                as <$native as SignedToUnsigned>::Unsigned;
            out.extend_from_slice(&biased.to_be_bytes());
        }};
    }

    macro_rules! unsigned {
        ($ty:ty) => {{
            out.extend_from_slice(&array.as_primitive::<$ty>().value(row).to_be_bytes());
        }};
    }

    match array.data_type() {
        DataType::Boolean => out.push(u8::from(array.as_boolean().value(row))),

        DataType::Int8 => signed!(Int8Type, i8),
        DataType::Int16 => signed!(Int16Type, i16),
        DataType::Int32 => signed!(Int32Type, i32),
        DataType::Int64 => signed!(Int64Type, i64),
        DataType::UInt8 => unsigned!(UInt8Type),
        DataType::UInt16 => unsigned!(UInt16Type),
        DataType::UInt32 => unsigned!(UInt32Type),
        DataType::UInt64 => unsigned!(UInt64Type),

        DataType::Date32 => signed!(Date32Type, i32),
        DataType::Date64 => signed!(Date64Type, i64),
        DataType::Time32(TimeUnit::Second) => signed!(Time32SecondType, i32),
        DataType::Time32(TimeUnit::Millisecond) => signed!(Time32MillisecondType, i32),
        DataType::Time64(TimeUnit::Microsecond) => signed!(Time64MicrosecondType, i64),
        DataType::Time64(TimeUnit::Nanosecond) => signed!(Time64NanosecondType, i64),
        DataType::Timestamp(TimeUnit::Second, _) => signed!(TimestampSecondType, i64),
        DataType::Timestamp(TimeUnit::Millisecond, _) => signed!(TimestampMillisecondType, i64),
        DataType::Timestamp(TimeUnit::Microsecond, _) => signed!(TimestampMicrosecondType, i64),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => signed!(TimestampNanosecondType, i64),

        DataType::Float32 => {
            let bits = order_preserving_f32(array.as_primitive::<Float32Type>().value(row));
            out.extend_from_slice(&bits.to_be_bytes());
        }
        DataType::Float64 => {
            let bits = order_preserving_f64(array.as_primitive::<Float64Type>().value(row));
            out.extend_from_slice(&bits.to_be_bytes());
        }

        DataType::Utf8 => encode_bytes(out, array.as_string::<i32>().value(row).as_bytes()),
        DataType::LargeUtf8 => encode_bytes(out, array.as_string::<i64>().value(row).as_bytes()),
        DataType::Binary => encode_bytes(out, array.as_binary::<i32>().value(row)),
        DataType::LargeBinary => encode_bytes(out, array.as_binary::<i64>().value(row)),

        other => {
            return Err(Error::Unsupported(format!(
                "{other} cannot be part of a key"
            )))
        }
    }
    Ok(())
}

/// Build the key for one row of the key columns.
pub fn encode_row(columns: &[ArrayRef], row: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(columns.len() * 9);
    for column in columns {
        encode_value(&mut out, column.as_ref(), row)?;
    }
    Ok(out)
}

/// Write a variable-length value with its escape and terminator.
fn encode_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    for byte in bytes {
        out.push(*byte);
        if *byte == ESCAPE {
            out.push(ESCAPED);
        }
    }
    out.push(ESCAPE);
    out.push(TERMINATOR);
}

/// Map a float onto an unsigned integer that compares the same way.
///
/// Positive floats already compare correctly as big-endian bits once the sign
/// bit is set; negative ones compare backwards, so every bit is flipped.
///
/// Negative zero is normalised to positive zero first. The two are equal as
/// numbers but differ in their bits, and a key that told them apart would let a
/// lookup for `0.0` miss a row stored as `-0.0`.
fn order_preserving_f32(value: f32) -> u32 {
    let bits = if value == 0.0 { 0f32 } else { value }.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

fn order_preserving_f64(value: f64) -> u64 {
    let bits = if value == 0.0 { 0f64 } else { value }.to_bits();
    if bits & 0x8000_0000_0000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000_0000_0000
    }
}

/// The unsigned counterpart of a signed integer type.
trait SignedToUnsigned {
    type Unsigned;
}
macro_rules! signed_to_unsigned {
    ($signed:ty, $unsigned:ty) => {
        impl SignedToUnsigned for $signed {
            type Unsigned = $unsigned;
        }
    };
}
signed_to_unsigned!(i8, u8);
signed_to_unsigned!(i16, u16);
signed_to_unsigned!(i32, u32);
signed_to_unsigned!(i64, u64);

/// True when this type can be part of a key.
pub fn is_encodable(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
    )
}

/// The smallest key that is greater than every key starting with `prefix`.
///
/// A range scan over a key prefix runs from the prefix to this bound.
pub fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut out = prefix.to_vec();
    while let Some(last) = out.last_mut() {
        if *last < u8::MAX {
            *last += 1;
            return Some(out);
        }
        out.pop();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        BooleanArray, Float64Array, Int32Array, Int64Array, StringArray, UInt32Array,
    };
    use std::sync::Arc;

    /// Encode every row of an array, in order.
    fn keys(array: &dyn Array) -> Vec<Vec<u8>> {
        (0..array.len())
            .map(|row| {
                let mut out = Vec::new();
                encode_value(&mut out, array, row).unwrap();
                out
            })
            .collect()
    }

    /// The encoded order must match the order given.
    fn assert_sorted(keys: &[Vec<u8>]) {
        for pair in keys.windows(2) {
            assert!(
                pair[0] < pair[1],
                "{:?} should sort before {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn signed_integers_sort_across_zero() {
        let array = Int32Array::from(vec![i32::MIN, -1000, -1, 0, 1, 1000, i32::MAX]);
        assert_sorted(&keys(&array));
    }

    #[test]
    fn unsigned_integers_sort() {
        let array = UInt32Array::from(vec![0, 1, 1000, u32::MAX]);
        assert_sorted(&keys(&array));
    }

    #[test]
    fn floats_sort_across_zero_and_infinity() {
        let array = Float64Array::from(vec![
            f64::NEG_INFINITY,
            f64::MIN,
            -1.5,
            0.0,
            1.5,
            f64::MAX,
            f64::INFINITY,
        ]);
        assert_sorted(&keys(&array));
    }

    #[test]
    fn negative_zero_gets_the_same_key_as_zero() {
        // Equal values must get equal keys, or a lookup for one misses a row
        // stored under the other.
        let array = Float64Array::from(vec![-0.0, 0.0]);
        let encoded = keys(&array);
        assert_eq!(encoded[0], encoded[1]);

        let array = arrow_array::Float32Array::from(vec![-0.0f32, 0.0]);
        let encoded = keys(&array);
        assert_eq!(encoded[0], encoded[1]);
    }

    #[test]
    fn booleans_sort_false_before_true() {
        assert_sorted(&keys(&BooleanArray::from(vec![false, true])));
    }

    #[test]
    fn strings_sort_lexicographically() {
        let array = StringArray::from(vec!["", "a", "ab", "abc", "b", "ba", "z"]);
        assert_sorted(&keys(&array));
    }

    #[test]
    fn a_prefix_sorts_before_what_extends_it() {
        // This is what the escape and terminator are for: without them, a
        // shorter key could look larger than a longer one it prefixes.
        let array = StringArray::from(vec!["ab", "ab\u{0}", "abc"]);
        assert_sorted(&keys(&array));
    }

    #[test]
    fn embedded_zero_bytes_do_not_end_a_value() {
        let array =
            arrow_array::BinaryArray::from(vec![&b"a\x00b"[..], &b"a\x00c"[..], &b"ab"[..]]);
        assert_sorted(&keys(&array));
    }

    #[test]
    fn nulls_sort_before_every_value() {
        let array = Int32Array::from(vec![None, Some(i32::MIN), Some(0), Some(i32::MAX)]);
        assert_sorted(&keys(&array));
    }

    #[test]
    fn a_multi_column_key_sorts_by_each_column_in_turn() {
        let a: ArrayRef = Arc::new(Int32Array::from(vec![1, 1, 1, 2, 2]));
        let b: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c", "a", "b"]));
        let columns = vec![a, b];

        let keys: Vec<Vec<u8>> = (0..5)
            .map(|row| encode_row(&columns, row).unwrap())
            .collect();
        assert_sorted(&keys);
    }

    #[test]
    fn a_shorter_first_column_still_orders_the_second() {
        let a: ArrayRef = Arc::new(StringArray::from(vec!["a", "a", "ab"]));
        let b: ArrayRef = Arc::new(Int32Array::from(vec![2, 10, 1]));
        let columns = vec![a, b];

        let keys: Vec<Vec<u8>> = (0..3)
            .map(|row| encode_row(&columns, row).unwrap())
            .collect();
        assert_sorted(&keys);
    }

    #[test]
    fn an_unsupported_type_is_refused_rather_than_mis_sorted() {
        let values = Int32Array::from(vec![1, 2, 3]);
        let field = Arc::new(arrow_schema::Field::new("item", DataType::Int32, true));
        let offsets = arrow_buffer::OffsetBuffer::new(vec![0, 1, 3].into());
        let list = arrow_array::ListArray::new(field, offsets, Arc::new(values), None);

        let mut out = Vec::new();
        assert!(encode_value(&mut out, &list, 0).is_err());
        assert!(!is_encodable(list.data_type()));
    }

    #[test]
    fn timestamps_and_dates_are_encodable() {
        for data_type in [
            DataType::Date32,
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Time64(TimeUnit::Nanosecond),
        ] {
            assert!(is_encodable(&data_type), "{data_type} should be encodable");
        }
    }

    #[test]
    fn timestamp_keys_sort_across_the_epoch() {
        let array = arrow_array::TimestampMicrosecondArray::from(vec![
            i64::MIN,
            -1_000_000,
            0,
            1_000_000,
            i64::MAX,
        ]);
        assert_sorted(&keys(&array));
    }

    #[test]
    fn a_prefix_bound_covers_every_key_that_starts_with_it() {
        let bound = prefix_upper_bound(b"abc").unwrap();
        assert!(bound.as_slice() > &b"abc"[..]);
        assert!(bound.as_slice() > &b"abcz"[..]);
        assert!(bound.as_slice() < &b"abd"[..] || bound == b"abd".to_vec());

        assert_eq!(prefix_upper_bound(b"ab\xff"), Some(b"ac".to_vec()));
        assert_eq!(prefix_upper_bound(b"\xff\xff"), None);
        assert_eq!(prefix_upper_bound(b""), None);
    }

    #[test]
    fn i64_keys_sort_across_the_whole_range() {
        let array = Int64Array::from(vec![
            i64::MIN,
            i64::MIN + 1,
            -1,
            0,
            1,
            i64::MAX - 1,
            i64::MAX,
        ]);
        assert_sorted(&keys(&array));
    }
}
