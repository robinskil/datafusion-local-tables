//! Encoded keys must sort exactly as the values they came from.
//!
//! This is the property the whole b-tree rests on: the tree compares bytes, so
//! if the byte order and the value order ever disagree, lookups miss rows that
//! are there. Random values across the full range of each type are the only way
//! to be confident of it.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, StringArray, UInt64Array,
};
use proptest::prelude::*;

use localtables_format::valuecodec::{encode_row, encode_value, prefix_upper_bound};

/// Encode one value of a single-column key.
fn key(array: &ArrayRef, row: usize) -> Vec<u8> {
    let mut out = Vec::new();
    encode_value(&mut out, array.as_ref(), row).unwrap();
    out
}

/// Byte order and value order must agree for every pair.
fn check_pairwise<T: PartialOrd + std::fmt::Debug>(values: &[Option<T>], array: ArrayRef) {
    for (i, left) in values.iter().enumerate() {
        for (j, right) in values.iter().enumerate() {
            let expected = match (left, right) {
                // Null sorts before every value, and ties with itself.
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Less,
                (Some(_), None) => Ordering::Greater,
                (Some(a), Some(b)) => match a.partial_cmp(b) {
                    Some(ordering) => ordering,
                    // Values that do not compare (NaN) are excluded upstream.
                    None => continue,
                },
            };
            let found = key(&array, i).cmp(&key(&array, j));
            assert_eq!(
                found, expected,
                "{left:?} vs {right:?}: bytes say {found:?}, values say {expected:?}"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

    #[test]
    fn i32_keys_sort_like_their_values(
        values in prop::collection::vec(prop::option::of(any::<i32>()), 1..12)
    ) {
        let array: ArrayRef = Arc::new(Int32Array::from(values.clone()));
        check_pairwise(&values, array);
    }

    #[test]
    fn i64_keys_sort_like_their_values(
        values in prop::collection::vec(prop::option::of(any::<i64>()), 1..12)
    ) {
        let array: ArrayRef = Arc::new(Int64Array::from(values.clone()));
        check_pairwise(&values, array);
    }

    #[test]
    fn u64_keys_sort_like_their_values(
        values in prop::collection::vec(prop::option::of(any::<u64>()), 1..12)
    ) {
        let array: ArrayRef = Arc::new(UInt64Array::from(values.clone()));
        check_pairwise(&values, array);
    }

    #[test]
    fn bool_keys_sort_like_their_values(
        values in prop::collection::vec(prop::option::of(any::<bool>()), 1..8)
    ) {
        let array: ArrayRef = Arc::new(BooleanArray::from(values.clone()));
        check_pairwise(&values, array);
    }

    /// Floats without NaN, which has no order to preserve.
    #[test]
    fn f64_keys_sort_like_their_values(
        values in prop::collection::vec(
            prop::option::of(prop_oneof![
                Just(f64::NEG_INFINITY),
                Just(f64::INFINITY),
                Just(0.0f64),
                Just(-0.0f64),
                -1e300f64..1e300,
            ]),
            1..12,
        )
    ) {
        // Negative and positive zero are equal as numbers, so the encoding has
        // to give them the same key; anything else would let a lookup for one
        // miss a row stored under the other.
        let array: ArrayRef = Arc::new(Float64Array::from(values.clone()));
        check_pairwise(&values, array);
    }

    /// Strings, including ones that prefix each other and ones holding zero
    /// bytes, which is what the escape exists for.
    #[test]
    fn string_keys_sort_like_their_values(
        values in prop::collection::vec(
            prop::option::of("[a-c\u{0}\u{1}\u{ff}]{0,6}"),
            1..12,
        )
    ) {
        let array: ArrayRef = Arc::new(StringArray::from(values.clone()));
        check_pairwise(&values, array);
    }

    /// A multi-column key must sort by each column in turn, exactly as a tuple
    /// comparison would.
    #[test]
    fn multi_column_keys_sort_like_their_tuples(
        rows in prop::collection::vec(
            (
                prop::option::of(-5i32..5),
                prop::option::of("[a-c]{0,3}"),
                prop::option::of(any::<bool>()),
            ),
            1..10,
        )
    ) {
        let ints: ArrayRef = Arc::new(Int32Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        ));
        let strings: ArrayRef = Arc::new(StringArray::from(
            rows.iter().map(|r| r.1.clone()).collect::<Vec<_>>(),
        ));
        let bools: ArrayRef = Arc::new(BooleanArray::from(
            rows.iter().map(|r| r.2).collect::<Vec<_>>(),
        ));
        let columns = vec![ints, strings, bools];

        /// Null sorts first, matching the encoding's null tag.
        fn rank<T: Ord>(value: &Option<T>) -> (u8, Option<&T>) {
            match value {
                None => (0, None),
                Some(v) => (1, Some(v)),
            }
        }

        for (i, left) in rows.iter().enumerate() {
            for (j, right) in rows.iter().enumerate() {
                let expected = rank(&left.0)
                    .cmp(&rank(&right.0))
                    .then_with(|| rank(&left.1).cmp(&rank(&right.1)))
                    .then_with(|| rank(&left.2).cmp(&rank(&right.2)));
                let found = encode_row(&columns, i)
                    .unwrap()
                    .cmp(&encode_row(&columns, j).unwrap());
                prop_assert_eq!(
                    found, expected,
                    "row {:?} vs {:?}", left, right
                );
            }
        }
    }

    /// The bound a prefix scan runs to must exclude nothing that starts with
    /// the prefix, and include nothing that does not.
    #[test]
    fn a_prefix_bound_covers_exactly_the_prefix(
        prefix in prop::collection::vec(0u8..=255, 1..6),
        suffix in prop::collection::vec(0u8..=255, 0..6),
    ) {
        let Some(bound) = prefix_upper_bound(&prefix) else {
            // All 0xff: no bound of the same length exists, and the scan runs
            // to the end of the tree instead.
            prop_assert!(prefix.iter().all(|b| *b == 0xff));
            return Ok(());
        };

        let mut extended = prefix.clone();
        extended.extend_from_slice(&suffix);
        prop_assert!(extended < bound, "{extended:?} should be below {bound:?}");
        prop_assert!(prefix < bound);
    }
}
