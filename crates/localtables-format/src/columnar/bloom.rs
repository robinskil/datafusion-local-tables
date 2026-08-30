//! Per-column membership filters, for the predicates zone maps cannot help.
//!
//! A zone map prunes `col = x` only when `x` falls outside a segment's minimum
//! and maximum. Take a column of scattered values: an id, a hash, a name. Every
//! segment spans nearly the whole range. Every segment survives, and the scan
//! reads all of them to find one row.
//!
//! A membership filter answers a different question. *Is this value definitely
//! absent?* It answers no when it is sure. It shrugs otherwise. To be sure is
//! enough to skip a segment.
//!
//! The filter never reports a value absent when that value is present. That
//! direction would lose rows. The tests hammer that property hardest.
//!
//! The layout is the split-block filter parquet uses. The bits for one value
//! all sit in one 32-byte block, so a lookup touches one cache line.

use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;

use crate::valuecodec;
use crate::{Error, Result};

/// Words in a block. Eight 32-bit words is 32 bytes: one cache line.
const WORDS_PER_BLOCK: usize = 8;
const BITS_PER_BLOCK: usize = WORDS_PER_BLOCK * 32;

/// Odd constants that spread one hash across the eight words of a block.
///
/// These are the parquet specification's, kept so the arithmetic here is the
/// arithmetic a reader of that specification would expect.
const SALT: [u32; WORDS_PER_BLOCK] = [
    0x47b6_137b,
    0x4497_4d91,
    0x8824_ad5b,
    0xa2b7_289d,
    0x7054_95c7,
    0x2df1_424b,
    0x9efc_4947,
    0x5c6b_fb31,
];

/// Bits per value when the caller does not say.
///
/// Measured false positive rates, 200,000 probes against values not stored:
///
/// | bits per value | 65,536 rows | filter size |
/// | --- | --- | --- |
/// | 6 | 9.9% | 48 KiB |
/// | 10 | 1.2% | 80 KiB |
/// | 16 | 0.13% | 128 KiB |
///
/// Ten is the knee. Below eight the rate climbs steeply, because a value sets
/// eight bits and there is no longer room for them: at six bits the filter is
/// worse than a classical one of the same size, which is the price the
/// single-cache-line layout charges. A false positive costs a segment read and
/// never a row.
pub const DEFAULT_BITS_PER_VALUE: usize = 10;

/// A membership filter over one column chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BloomFilter {
    /// Blocks of eight words each.
    words: Vec<u32>,
}

impl BloomFilter {
    /// A filter sized for `values` values at `bits_per_value`.
    pub fn with_capacity(values: usize, bits_per_value: usize) -> Self {
        let bits = values.max(1) * bits_per_value.max(1);
        let blocks = bits.div_ceil(BITS_PER_BLOCK).max(1);
        Self {
            words: vec![0u32; blocks * WORDS_PER_BLOCK],
        }
    }

    /// Read a filter back from its stored bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if !bytes.len().is_multiple_of(WORDS_PER_BLOCK * 4) || bytes.is_empty() {
            return Err(Error::corrupt(format!(
                "a membership filter is {} bytes, which is not whole blocks",
                bytes.len()
            )));
        }
        let words = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        Ok(Self { words })
    }

    /// The stored form.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.words.len() * 4);
        for word in &self.words {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out
    }

    pub fn byte_len(&self) -> usize {
        self.words.len() * 4
    }

    fn blocks(&self) -> usize {
        self.words.len() / WORDS_PER_BLOCK
    }

    /// Where in the filter a hash lands, and which bits it sets there.
    fn locate(&self, hash: u64) -> (usize, [u32; WORDS_PER_BLOCK]) {
        // The high half picks the block, scaled rather than masked so the block
        // count need not be a power of two.
        let block = (((hash >> 32) as u128 * self.blocks() as u128) >> 32) as usize;
        let low = hash as u32;

        let mut mask = [0u32; WORDS_PER_BLOCK];
        for (word, salt) in mask.iter_mut().zip(SALT) {
            // The top five bits of each salted hash choose a bit in that word.
            *word = 1u32 << (low.wrapping_mul(salt) >> 27);
        }
        (block * WORDS_PER_BLOCK, mask)
    }

    pub fn insert_hash(&mut self, hash: u64) {
        let (base, mask) = self.locate(hash);
        for (word, bit) in self.words[base..base + WORDS_PER_BLOCK]
            .iter_mut()
            .zip(mask)
        {
            *word |= bit;
        }
    }

    /// True when the hash may be present. False means it certainly is not.
    pub fn may_contain_hash(&self, hash: u64) -> bool {
        let (base, mask) = self.locate(hash);
        self.words[base..base + WORDS_PER_BLOCK]
            .iter()
            .zip(mask)
            .all(|(word, bit)| word & bit != 0)
    }

    /// Build a filter over an array's non-null values.
    ///
    /// Nulls are left out: `col = x` is never true of a null, so a filter that
    /// knew about them would only be larger.
    ///
    /// Sized by the number of *distinct* values, not the number of rows.
    /// Inserting the same value twice tells a filter nothing, so a column of a
    /// thousand rows holding eight values needs room for eight. That matters
    /// most for the column where a filter is useless anyway: if every segment
    /// holds every value, the filter can never rule one out, and sizing it by
    /// rows would spend real bytes saying so.
    pub fn build(array: &dyn Array, bits_per_value: usize) -> Result<Option<Self>> {
        if !supports(array.data_type()) || array.is_empty() {
            return Ok(None);
        }

        let mut hashes = std::collections::HashSet::new();
        let mut bytes = Vec::with_capacity(32);
        for row in 0..array.len() {
            if array.is_null(row) {
                continue;
            }
            bytes.clear();
            valuecodec::encode_value(&mut bytes, array, row)?;
            hashes.insert(crate::layout::checksum(&bytes));
        }
        if hashes.is_empty() {
            return Ok(None);
        }

        let mut filter = Self::with_capacity(hashes.len(), bits_per_value);
        for hash in hashes {
            filter.insert_hash(hash);
        }
        Ok(Some(filter))
    }

    /// True when this value may be present in the chunk.
    ///
    /// The value is compared as the column's own type, so a literal of another
    /// type is cast first; one that cannot be cast is treated as unknown rather
    /// than absent, because a failed cast says nothing about what is stored.
    pub fn may_contain(&self, value: &ArrayRef, data_type: &DataType) -> bool {
        let Some(hash) = hash_value(value, data_type) else {
            return true;
        };
        self.may_contain_hash(hash)
    }
}

/// The hash of a one-element array's value, as the column's type.
fn hash_value(value: &ArrayRef, data_type: &DataType) -> Option<u64> {
    if value.len() != 1 || value.is_null(0) {
        return None;
    }
    let cast;
    let array = if value.data_type() == data_type {
        value
    } else {
        cast = arrow_cast::cast(value, data_type).ok()?;
        &cast
    };
    // A cast that cannot represent the value gives a null, not an error. So the
    // null needs a check here as well as above. A null says nothing about what
    // the column holds, and to prune on it would lose rows.
    if array.is_null(0) {
        return None;
    }

    let mut bytes = Vec::with_capacity(32);
    valuecodec::encode_value(&mut bytes, array.as_ref(), 0).ok()?;
    Some(crate::layout::checksum(&bytes))
}

/// True for the types a value can be hashed canonically as.
pub fn supports(data_type: &DataType) -> bool {
    valuecodec::is_encodable(data_type)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, Int64Array, StringArray};
    use std::sync::Arc;

    fn filter_over(array: &dyn Array) -> BloomFilter {
        BloomFilter::build(array, DEFAULT_BITS_PER_VALUE)
            .unwrap()
            .expect("a filter for a supported type")
    }

    /// Every value that went in must be reported as possibly present. A miss
    /// here would silently drop rows from a query.
    #[test]
    fn a_value_that_was_inserted_is_never_reported_absent() {
        let values: Vec<i64> = (0..5_000).map(|i| i * 7919).collect();
        let array = Int64Array::from(values.clone());
        let filter = filter_over(&array);

        for (row, value) in values.iter().enumerate() {
            let mut bytes = Vec::new();
            valuecodec::encode_value(&mut bytes, &array, row).unwrap();
            assert!(
                filter.may_contain_hash(crate::layout::checksum(&bytes)),
                "value {value} was reported absent"
            );
        }
    }

    #[test]
    fn strings_that_were_inserted_are_never_reported_absent() {
        let values: Vec<String> = (0..2_000).map(|i| format!("value-{i}")).collect();
        let array = StringArray::from(values.iter().map(|v| Some(v.as_str())).collect::<Vec<_>>());
        let filter = filter_over(&array);

        for row in 0..array.len() {
            let mut bytes = Vec::new();
            valuecodec::encode_value(&mut bytes, &array, row).unwrap();
            assert!(filter.may_contain_hash(crate::layout::checksum(&bytes)));
        }
    }

    #[test]
    fn most_values_that_were_not_inserted_are_reported_absent() {
        let array = Int64Array::from((0..10_000i64).collect::<Vec<_>>());
        let filter = filter_over(&array);

        let mut false_positives = 0;
        let absent = 10_000;
        for value in 1_000_000..1_000_000 + absent {
            let probe: ArrayRef = Arc::new(Int64Array::from(vec![value]));
            if filter.may_contain(&probe, &DataType::Int64) {
                false_positives += 1;
            }
        }
        // Ten bits per value is about one in a hundred; allow generous slack so
        // the test measures the design rather than one machine's hash luck.
        assert!(
            false_positives * 20 < absent,
            "{false_positives} of {absent} absent values looked present"
        );
    }

    #[test]
    fn a_literal_of_another_type_is_cast_before_it_is_hashed() {
        let array = Int32Array::from(vec![1, 2, 3]);
        let filter = filter_over(&array);

        // An i64 literal against an i32 column must still be found.
        let probe: ArrayRef = Arc::new(Int64Array::from(vec![2i64]));
        assert!(filter.may_contain(&probe, &DataType::Int32));
    }

    #[test]
    fn a_literal_that_cannot_be_cast_is_treated_as_unknown() {
        let array = Int32Array::from(vec![1, 2, 3]);
        let filter = filter_over(&array);

        let probe: ArrayRef = Arc::new(StringArray::from(vec!["not a number at all"]));
        assert!(
            filter.may_contain(&probe, &DataType::Int32),
            "a failed cast says nothing about what is stored, so it must not prune"
        );
    }

    #[test]
    fn nulls_are_left_out_but_do_not_stop_a_filter_being_built() {
        let array = Int64Array::from(vec![Some(1), None, Some(3)]);
        let filter = filter_over(&array);

        let present: ArrayRef = Arc::new(Int64Array::from(vec![1i64]));
        assert!(filter.may_contain(&present, &DataType::Int64));
        let null: ArrayRef = Arc::new(Int64Array::from(vec![None::<i64>]));
        assert!(
            filter.may_contain(&null, &DataType::Int64),
            "a null probe is unknown, not absent"
        );
    }

    #[test]
    fn an_all_null_column_gets_no_filter() {
        let array = Int64Array::from(vec![None, None, None] as Vec<Option<i64>>);
        assert!(BloomFilter::build(&array, DEFAULT_BITS_PER_VALUE)
            .unwrap()
            .is_none());
    }

    #[test]
    fn an_empty_column_gets_no_filter() {
        let array = Int64Array::from(Vec::<i64>::new());
        assert!(BloomFilter::build(&array, DEFAULT_BITS_PER_VALUE)
            .unwrap()
            .is_none());
    }

    /// View arrays hold the same text behind a different layout, so they must
    /// hash to the same thing. DataFusion asks for view types by default, so
    /// without this a string column would often have no filter at all.
    #[test]
    fn a_view_column_gets_a_filter_that_matches_its_plain_form() {
        use arrow_array::{BinaryViewArray, StringViewArray};

        let values = ["alpha", "bravo", "charlie"];
        let plain = StringArray::from(values.to_vec());
        let view = StringViewArray::from(values.to_vec());
        assert!(supports(view.data_type()));

        let plain_filter = filter_over(&plain);
        let view_filter = filter_over(&view);
        assert_eq!(
            plain_filter, view_filter,
            "the same text must give the same filter whatever the layout"
        );

        // And a probe of either type finds it.
        let probe: ArrayRef = Arc::new(StringArray::from(vec!["bravo"]));
        assert!(view_filter.may_contain(&probe, &DataType::Utf8View));
        let absent: ArrayRef = Arc::new(StringArray::from(vec!["zulu"]));
        assert!(!view_filter.may_contain(&absent, &DataType::Utf8View));

        let blobs = BinaryViewArray::from(vec![&b"one"[..], b"two"]);
        assert!(supports(blobs.data_type()));
        assert!(BloomFilter::build(&blobs, DEFAULT_BITS_PER_VALUE)
            .unwrap()
            .is_some());
    }

    #[test]
    fn a_type_with_no_canonical_encoding_gets_no_filter() {
        let values = Int32Array::from(vec![1, 2, 3]);
        let field = Arc::new(arrow_schema::Field::new("item", DataType::Int32, true));
        let offsets = arrow_buffer::OffsetBuffer::new(vec![0, 1, 3].into());
        let list = arrow_array::ListArray::new(field, offsets, Arc::new(values), None);

        assert!(BloomFilter::build(&list, DEFAULT_BITS_PER_VALUE)
            .unwrap()
            .is_none());
        assert!(!supports(list.data_type()));
    }

    #[test]
    fn a_filter_round_trips_through_its_bytes() {
        let array = Int64Array::from((0..1_000i64).collect::<Vec<_>>());
        let filter = filter_over(&array);

        let restored = BloomFilter::from_bytes(&filter.to_bytes()).unwrap();
        assert_eq!(restored, filter);
        assert_eq!(restored.byte_len(), filter.byte_len());
    }

    #[test]
    fn bytes_that_are_not_whole_blocks_are_refused() {
        assert!(BloomFilter::from_bytes(&[]).is_err());
        assert!(BloomFilter::from_bytes(&[0u8; 7]).is_err());
        assert!(BloomFilter::from_bytes(&[0u8; 32]).is_ok());
    }

    /// A column that repeats itself needs room for what it holds, not for how
    /// often it holds it.
    #[test]
    fn a_filter_is_sized_by_distinct_values() {
        let few: Vec<i64> = (0..10_000).map(|i| i % 8).collect();
        let many: Vec<i64> = (0..10_000).collect();
        let repeated = filter_over(&Int64Array::from(few));
        let distinct = filter_over(&Int64Array::from(many));

        assert!(
            repeated.byte_len() * 50 < distinct.byte_len(),
            "eight values should not need a filter sized for ten thousand: \
             {} against {}",
            repeated.byte_len(),
            distinct.byte_len()
        );
        // And it still answers correctly.
        for value in 0..8i64 {
            let probe: ArrayRef = Arc::new(Int64Array::from(vec![value]));
            assert!(repeated.may_contain(&probe, &DataType::Int64));
        }
    }

    #[test]
    fn a_bigger_filter_makes_fewer_mistakes() {
        let array = Int64Array::from((0..5_000i64).collect::<Vec<_>>());
        let small = BloomFilter::build(&array, 4).unwrap().unwrap();
        let large = BloomFilter::build(&array, 20).unwrap().unwrap();
        assert!(large.byte_len() > small.byte_len());

        let miss = |filter: &BloomFilter| {
            (0..5_000i64)
                .filter(|i| {
                    let probe: ArrayRef = Arc::new(Int64Array::from(vec![1_000_000 + i]));
                    filter.may_contain(&probe, &DataType::Int64)
                })
                .count()
        };
        assert!(
            miss(&large) <= miss(&small),
            "more bits should not mean more false positives"
        );
    }
}
