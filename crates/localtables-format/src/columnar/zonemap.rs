//! Per-column bounds, used to skip segments a query cannot match.
//!
//! A zone map holds the smallest and largest value in a column chunk. It also
//! holds the null count. A predicate outside those bounds matches no row in the
//! chunk. The scan then skips the chunk and reads none of its data.
//!
//! Bounds are stored as raw value bytes and are read back against the column's
//! declared type. Long strings and binary are truncated, which changes what the
//! bounds mean:
//!
//! * A truncated minimum is still a valid lower bound. To cut bytes off the
//!   end of a string can only make it smaller or equal.
//! * A truncated maximum is *not* a valid upper bound. The zone map reports it
//!   as unknown, unless the truncated form rounds up to one.
//!
//! Reporting an unknown bound costs a segment read. Reporting a wrong one loses
//! rows, so the rule never guesses.

use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{Array, ArrayRef, BooleanArray, PrimitiveArray};
use arrow_schema::{DataType, TimeUnit};
use rkyv::{Archive, Deserialize, Serialize};
use std::sync::Arc;

/// Longest min or max the format stores. Longer values are truncated.
pub const MAX_BOUND_BYTES: usize = 64;

/// Bounds and null count for one column chunk.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
#[rkyv(derive(Debug))]
pub struct ZoneMap {
    pub null_count: u64,
    /// Smallest value, as raw bytes. Absent when every row is null, or when
    /// the type has no ordering this format records.
    pub min: Option<Vec<u8>>,
    /// Largest value, as raw bytes. Absent for the same reasons, and also when
    /// truncation left no sound upper bound.
    pub max: Option<Vec<u8>>,
    /// The stored minimum was cut short. It is still a valid lower bound.
    pub min_truncated: bool,
}

impl ZoneMap {
    /// A zone map that rules nothing out.
    pub fn unknown(null_count: u64) -> Self {
        Self {
            null_count,
            min: None,
            max: None,
            min_truncated: false,
        }
    }

    /// True when neither bound is known, so this map can never prune.
    pub fn is_unknown(&self) -> bool {
        self.min.is_none() && self.max.is_none()
    }

    /// Compute bounds over `array`.
    pub fn build(array: &dyn Array) -> Self {
        let null_count = array.null_count() as u64;
        let (min, max) = bounds(array);

        let (min, min_truncated) = match min {
            Some(bytes) if bytes.len() > MAX_BOUND_BYTES => {
                // Cutting bytes off the end can only lower a value, so the
                // prefix stays a valid lower bound.
                (Some(bytes[..MAX_BOUND_BYTES].to_vec()), true)
            }
            other => (other, false),
        };

        let max = match max {
            Some(bytes) if bytes.len() > MAX_BOUND_BYTES => {
                // A prefix is smaller than the value it came from, so it is not
                // an upper bound. Rounding the prefix up gives one back; when
                // every byte is 0xff there is nothing to round up to, and the
                // bound is dropped.
                increment_prefix(&bytes[..MAX_BOUND_BYTES])
            }
            other => other,
        };

        Self {
            null_count,
            min,
            max,
            min_truncated,
        }
    }

    /// The minimum as a one-element array of `data_type`.
    ///
    /// Returns `None` when the bound is unknown. It also returns `None` when
    /// the bound cannot be read back as this type. A caller must read `None` as
    /// "no information", never as a match.
    pub fn min_array(&self, data_type: &DataType) -> Option<ArrayRef> {
        decode_bound(self.min.as_deref()?, data_type)
    }

    /// The maximum as a one-element array of `data_type`.
    pub fn max_array(&self, data_type: &DataType) -> Option<ArrayRef> {
        decode_bound(self.max.as_deref()?, data_type)
    }
}

impl ArchivedZoneMap {
    /// Copy the archive into an owned zone map. Pruning reads bounds once per
    /// segment, so this is off the hot path.
    pub fn to_native(&self) -> ZoneMap {
        ZoneMap {
            null_count: self.null_count.to_native(),
            min: self.min.as_ref().map(|v| v.to_vec()),
            max: self.max.as_ref().map(|v| v.to_vec()),
            min_truncated: self.min_truncated,
        }
    }
}

/// Round a truncated prefix up to the smallest value that is greater than
/// every string starting with it.
///
/// Increments the last byte below 0xff and drops the trailing 0xff run. All
/// 0xff means no such value fits in the same length, so there is no bound.
fn increment_prefix(prefix: &[u8]) -> Option<Vec<u8>> {
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

/// Smallest and largest value in `array`, as raw bytes.
fn bounds(array: &dyn Array) -> (Option<Vec<u8>>, Option<Vec<u8>>) {
    use arrow_arith::aggregate;

    macro_rules! primitive {
        ($ty:ty) => {{
            let typed: &PrimitiveArray<$ty> = array.as_primitive::<$ty>();
            (
                aggregate::min(typed).map(|v| v.to_le_bytes().to_vec()),
                aggregate::max(typed).map(|v| v.to_le_bytes().to_vec()),
            )
        }};
    }

    match array.data_type() {
        DataType::Boolean => {
            let typed: &BooleanArray = array.as_boolean();
            (
                aggregate::min_boolean(typed).map(|v| vec![v as u8]),
                aggregate::max_boolean(typed).map(|v| vec![v as u8]),
            )
        }

        DataType::Int8 => primitive!(Int8Type),
        DataType::Int16 => primitive!(Int16Type),
        DataType::Int32 => primitive!(Int32Type),
        DataType::Int64 => primitive!(Int64Type),
        DataType::UInt8 => primitive!(UInt8Type),
        DataType::UInt16 => primitive!(UInt16Type),
        DataType::UInt32 => primitive!(UInt32Type),
        DataType::UInt64 => primitive!(UInt64Type),
        DataType::Float32 => primitive!(Float32Type),
        DataType::Float64 => primitive!(Float64Type),
        DataType::Date32 => primitive!(Date32Type),
        DataType::Date64 => primitive!(Date64Type),
        DataType::Time32(TimeUnit::Second) => primitive!(Time32SecondType),
        DataType::Time32(TimeUnit::Millisecond) => primitive!(Time32MillisecondType),
        DataType::Time64(TimeUnit::Microsecond) => primitive!(Time64MicrosecondType),
        DataType::Time64(TimeUnit::Nanosecond) => primitive!(Time64NanosecondType),
        DataType::Timestamp(TimeUnit::Second, _) => primitive!(TimestampSecondType),
        DataType::Timestamp(TimeUnit::Millisecond, _) => primitive!(TimestampMillisecondType),
        DataType::Timestamp(TimeUnit::Microsecond, _) => primitive!(TimestampMicrosecondType),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => primitive!(TimestampNanosecondType),
        DataType::Duration(TimeUnit::Second) => primitive!(DurationSecondType),
        DataType::Duration(TimeUnit::Millisecond) => primitive!(DurationMillisecondType),
        DataType::Duration(TimeUnit::Microsecond) => primitive!(DurationMicrosecondType),
        DataType::Duration(TimeUnit::Nanosecond) => primitive!(DurationNanosecondType),

        DataType::Utf8 => {
            let typed = array.as_string::<i32>();
            (
                aggregate::min_string(typed).map(|v| v.as_bytes().to_vec()),
                aggregate::max_string(typed).map(|v| v.as_bytes().to_vec()),
            )
        }
        DataType::LargeUtf8 => {
            let typed = array.as_string::<i64>();
            (
                aggregate::min_string(typed).map(|v| v.as_bytes().to_vec()),
                aggregate::max_string(typed).map(|v| v.as_bytes().to_vec()),
            )
        }
        DataType::Binary => {
            let typed = array.as_binary::<i32>();
            (
                aggregate::min_binary(typed).map(|v| v.to_vec()),
                aggregate::max_binary(typed).map(|v| v.to_vec()),
            )
        }
        DataType::LargeBinary => {
            let typed = array.as_binary::<i64>();
            (
                aggregate::min_binary(typed).map(|v| v.to_vec()),
                aggregate::max_binary(typed).map(|v| v.to_vec()),
            )
        }

        // A dictionary's bounds come from its values. A value the keys never
        // reference widens the bounds; it never narrows them. The result still
        // holds every value present, which is all a bound must do.
        DataType::Dictionary(_, _) => {
            let values = array.to_data().child_data()[0].clone();
            bounds(arrow_array::make_array(values).as_ref())
        }

        // Anything else prunes nothing rather than risking a wrong bound.
        _ => (None, None),
    }
}

/// Rebuild a stored bound as a one-element array.
fn decode_bound(bytes: &[u8], data_type: &DataType) -> Option<ArrayRef> {
    macro_rules! primitive {
        ($ty:ty, $native:ty) => {{
            let raw: [u8; std::mem::size_of::<$native>()] = bytes.try_into().ok()?;
            let value = <$native>::from_le_bytes(raw);
            Some(Arc::new(PrimitiveArray::<$ty>::from(vec![value])) as ArrayRef)
        }};
    }

    /// Same, but keeps the type's parameters (time zone, unit) from the schema.
    macro_rules! primitive_typed {
        ($ty:ty, $native:ty) => {{
            let raw: [u8; std::mem::size_of::<$native>()] = bytes.try_into().ok()?;
            let value = <$native>::from_le_bytes(raw);
            let array = PrimitiveArray::<$ty>::from(vec![value]).with_data_type(data_type.clone());
            Some(Arc::new(array) as ArrayRef)
        }};
    }

    match data_type {
        DataType::Boolean => {
            let value = *bytes.first()? != 0;
            Some(Arc::new(BooleanArray::from(vec![value])) as ArrayRef)
        }

        DataType::Int8 => primitive!(Int8Type, i8),
        DataType::Int16 => primitive!(Int16Type, i16),
        DataType::Int32 => primitive!(Int32Type, i32),
        DataType::Int64 => primitive!(Int64Type, i64),
        DataType::UInt8 => primitive!(UInt8Type, u8),
        DataType::UInt16 => primitive!(UInt16Type, u16),
        DataType::UInt32 => primitive!(UInt32Type, u32),
        DataType::UInt64 => primitive!(UInt64Type, u64),
        DataType::Float32 => primitive!(Float32Type, f32),
        DataType::Float64 => primitive!(Float64Type, f64),
        DataType::Date32 => primitive!(Date32Type, i32),
        DataType::Date64 => primitive!(Date64Type, i64),
        DataType::Time32(TimeUnit::Second) => primitive!(Time32SecondType, i32),
        DataType::Time32(TimeUnit::Millisecond) => primitive!(Time32MillisecondType, i32),
        DataType::Time64(TimeUnit::Microsecond) => primitive!(Time64MicrosecondType, i64),
        DataType::Time64(TimeUnit::Nanosecond) => primitive!(Time64NanosecondType, i64),
        DataType::Timestamp(TimeUnit::Second, _) => primitive_typed!(TimestampSecondType, i64),
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            primitive_typed!(TimestampMillisecondType, i64)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            primitive_typed!(TimestampMicrosecondType, i64)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            primitive_typed!(TimestampNanosecondType, i64)
        }
        DataType::Duration(TimeUnit::Second) => primitive!(DurationSecondType, i64),
        DataType::Duration(TimeUnit::Millisecond) => primitive!(DurationMillisecondType, i64),
        DataType::Duration(TimeUnit::Microsecond) => primitive!(DurationMicrosecondType, i64),
        DataType::Duration(TimeUnit::Nanosecond) => primitive!(DurationNanosecondType, i64),

        DataType::Utf8 => {
            let value = std::str::from_utf8(bytes).ok()?;
            Some(Arc::new(arrow_array::StringArray::from(vec![value])) as ArrayRef)
        }
        DataType::LargeUtf8 => {
            let value = std::str::from_utf8(bytes).ok()?;
            Some(Arc::new(arrow_array::LargeStringArray::from(vec![value])) as ArrayRef)
        }
        // A dictionary's bounds are values, so they are read back as values.
        // Pruning compares them against literals of the value type.
        DataType::Dictionary(_, value_type) => decode_bound(bytes, value_type),

        DataType::Binary => Some(Arc::new(arrow_array::BinaryArray::from(vec![bytes])) as ArrayRef),
        DataType::LargeBinary => {
            Some(Arc::new(arrow_array::LargeBinaryArray::from(vec![bytes])) as ArrayRef)
        }

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, Int32Array, StringArray, TimestampMicrosecondArray};

    fn min_max_i32(zone: &ZoneMap) -> (Option<i32>, Option<i32>) {
        let get = |a: Option<ArrayRef>| a.map(|a| a.as_primitive::<Int32Type>().value(0));
        (
            get(zone.min_array(&DataType::Int32)),
            get(zone.max_array(&DataType::Int32)),
        )
    }

    #[test]
    fn integer_bounds_ignore_nulls() {
        let array = Int32Array::from(vec![Some(5), None, Some(-3), Some(11), None]);
        let zone = ZoneMap::build(&array);

        assert_eq!(zone.null_count, 2);
        assert_eq!(min_max_i32(&zone), (Some(-3), Some(11)));
        assert!(!zone.min_truncated);
    }

    #[test]
    fn an_all_null_column_has_no_bounds() {
        let array = Int32Array::from(vec![None, None, None]);
        let zone = ZoneMap::build(&array);

        assert_eq!(zone.null_count, 3);
        assert!(zone.is_unknown(), "no value means no bound to state");
        assert!(zone.min_array(&DataType::Int32).is_none());
    }

    #[test]
    fn an_empty_column_has_no_bounds() {
        let zone = ZoneMap::build(&Int32Array::from(Vec::<i32>::new()));
        assert!(zone.is_unknown());
        assert_eq!(zone.null_count, 0);
    }

    #[test]
    fn float_bounds_survive_the_round_trip() {
        let array = Float64Array::from(vec![1.5, -0.25, 1e300]);
        let zone = ZoneMap::build(&array);

        let min = zone.min_array(&DataType::Float64).unwrap();
        let max = zone.max_array(&DataType::Float64).unwrap();
        assert_eq!(min.as_primitive::<Float64Type>().value(0), -0.25);
        assert_eq!(max.as_primitive::<Float64Type>().value(0), 1e300);
    }

    #[test]
    fn boolean_bounds_are_the_two_ends() {
        let zone = ZoneMap::build(&BooleanArray::from(vec![true, false, true]));
        let min = zone.min_array(&DataType::Boolean).unwrap();
        let max = zone.max_array(&DataType::Boolean).unwrap();
        assert!(!min.as_boolean().value(0));
        assert!(max.as_boolean().value(0));
    }

    #[test]
    fn string_bounds_compare_lexicographically() {
        let array = StringArray::from(vec![Some("pear"), None, Some("apple"), Some("quince")]);
        let zone = ZoneMap::build(&array);

        let min = zone.min_array(&DataType::Utf8).unwrap();
        let max = zone.max_array(&DataType::Utf8).unwrap();
        assert_eq!(min.as_string::<i32>().value(0), "apple");
        assert_eq!(max.as_string::<i32>().value(0), "quince");
        assert_eq!(zone.null_count, 1);
    }

    #[test]
    fn timestamp_bounds_keep_the_declared_time_zone() {
        let data_type = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
        let array = TimestampMicrosecondArray::from(vec![100i64, 900, 500])
            .with_data_type(data_type.clone());
        let zone = ZoneMap::build(&array);

        let min = zone.min_array(&data_type).unwrap();
        assert_eq!(min.data_type(), &data_type, "a bound must match the column");
        assert_eq!(min.as_primitive::<TimestampMicrosecondType>().value(0), 100);
    }

    #[test]
    fn a_truncated_minimum_stays_a_valid_lower_bound() {
        let long = "a".repeat(MAX_BOUND_BYTES + 20);
        let zone = ZoneMap::build(&StringArray::from(vec![long.as_str(), "zzz"]));

        assert!(zone.min_truncated);
        let min = zone.min_array(&DataType::Utf8).unwrap();
        let stored = min.as_string::<i32>().value(0);
        assert_eq!(stored.len(), MAX_BOUND_BYTES);
        assert!(
            stored <= long.as_str(),
            "the prefix must not exceed the value"
        );
    }

    #[test]
    fn a_truncated_maximum_is_rounded_up_not_reported_short() {
        let long = format!("{}b", "a".repeat(MAX_BOUND_BYTES + 20));
        let zone = ZoneMap::build(&StringArray::from(vec!["aaa", long.as_str()]));

        let max = zone.max_array(&DataType::Utf8).unwrap();
        let stored = max.as_string::<i32>().value(0);
        assert!(
            stored.as_bytes() > long.as_bytes(),
            "a truncated max that is smaller than a real value would lose rows"
        );
    }

    #[test]
    fn a_maximum_that_cannot_be_rounded_up_is_dropped() {
        let all_high = [0xffu8; MAX_BOUND_BYTES + 5];
        let array = arrow_array::BinaryArray::from(vec![&all_high[..], b"\x00"]);
        let zone = ZoneMap::build(&array);

        assert!(
            zone.max.is_none(),
            "no sound upper bound exists, so none is claimed"
        );
        assert!(zone.min.is_some(), "the lower bound is unaffected");
    }

    #[test]
    fn increment_prefix_rounds_up_and_gives_up_on_all_high_bytes() {
        assert_eq!(increment_prefix(b"abc"), Some(b"abd".to_vec()));
        assert_eq!(increment_prefix(b"ab\xff"), Some(b"ac".to_vec()));
        assert_eq!(increment_prefix(b"\xff\xff"), None);
        assert_eq!(increment_prefix(b""), None);
    }

    #[test]
    fn a_dictionary_column_is_bounded_by_its_values() {
        let keys = Int32Array::from(vec![Some(0), Some(2), None, Some(1)]);
        let values = Arc::new(StringArray::from(vec!["banana", "cherry", "apple"]));
        let dict = arrow_array::DictionaryArray::<Int32Type>::try_new(keys, values).unwrap();

        let zone = ZoneMap::build(&dict);
        assert_eq!(zone.null_count, 1);

        // Read back as the value type, which is what a predicate compares to.
        let value_type = DataType::Utf8;
        let min = zone.min_array(&value_type).unwrap();
        let max = zone.max_array(&value_type).unwrap();
        assert_eq!(min.as_string::<i32>().value(0), "apple");
        assert_eq!(max.as_string::<i32>().value(0), "cherry");
    }

    #[test]
    fn a_dictionary_bound_reads_back_through_its_own_type_too() {
        let keys = Int32Array::from(vec![0, 1]);
        let values = Arc::new(StringArray::from(vec!["a", "z"]));
        let dict = arrow_array::DictionaryArray::<Int32Type>::try_new(keys, values).unwrap();

        let zone = ZoneMap::build(&dict);
        let declared = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let min = zone.min_array(&declared).unwrap();
        assert_eq!(
            min.data_type(),
            &DataType::Utf8,
            "a bound is a value, not an index into a dictionary"
        );
        assert_eq!(min.as_string::<i32>().value(0), "a");
    }

    #[test]
    fn an_unsupported_type_prunes_nothing() {
        let values = Int32Array::from(vec![1, 2, 3]);
        let field = Arc::new(arrow_schema::Field::new("item", DataType::Int32, true));
        let offsets = arrow_buffer::OffsetBuffer::new(vec![0, 1, 3].into());
        let list = arrow_array::ListArray::new(field, offsets, Arc::new(values), None);

        let zone = ZoneMap::build(&list);
        assert!(
            zone.is_unknown(),
            "unknown beats a bound that might be wrong"
        );
    }

    #[test]
    fn a_bound_read_back_as_the_wrong_type_is_refused() {
        let zone = ZoneMap::build(&Int32Array::from(vec![7]));
        assert!(zone.min_array(&DataType::Int32).is_some());
        assert!(
            zone.min_array(&DataType::Int64).is_none(),
            "four bytes cannot be read as an i64 bound"
        );
    }
}
