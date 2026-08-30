//! Substring pruning for text columns.
//!
//! A membership filter holds whole values, so it answers `col = 'hello'` and
//! nothing about `col LIKE '%ell%'`: no stored value equals `ell`.
//!
//! A trigram filter holds fragments instead. Every value is cut into
//! overlapping three-byte pieces, and those go into the filter:
//!
//! ```text
//! 'hello'  ->  'hel', 'ell', 'llo'
//! ```
//!
//! A search term is cut the same way. Every one of its pieces must be present
//! for any row to contain it, so one absent piece rules the segment out. The
//! filter never reports a piece absent when it is present, which is what makes
//! that safe.
//!
//! It cannot prove a match. A segment can hold `hel`, `ell` and `llo` in three
//! different rows and hold `hello` nowhere, and every probe still passes. That
//! is a second source of false positives on top of the filter's own, and no
//! amount of extra bits removes it. The scan's own filter decides the answer.
//!
//! Pieces are bytes, not characters. UTF-8 is self-synchronising, so a valid
//! sequence never appears starting part-way through another one, and byte
//! containment and text containment agree.

use arrow_array::cast::AsArray;
use arrow_array::Array;
use arrow_schema::DataType;

use crate::columnar::bloom::BloomFilter;
use crate::layout::checksum;
use crate::Result;

/// Bytes in a piece.
pub const SIZE: usize = 3;

/// Every trigram of a byte string, in order, with repeats.
pub fn trigrams(text: &[u8]) -> impl Iterator<Item = &[u8]> {
    text.windows(SIZE)
}

/// The hash a trigram is stored and probed under.
pub fn hash(trigram: &[u8]) -> u64 {
    checksum(trigram)
}

/// True for the types whose values can be cut into trigrams.
pub fn supports(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
    )
}

/// Call `visit` with the bytes of every non-null value.
fn for_each_value(array: &dyn Array, mut visit: impl FnMut(&[u8])) -> bool {
    macro_rules! walk {
        ($values:expr) => {{
            let values = $values;
            for row in 0..values.len() {
                if !values.is_null(row) {
                    visit(values.value(row).as_ref());
                }
            }
            true
        }};
    }

    match array.data_type() {
        DataType::Utf8 => walk!(array.as_string::<i32>()),
        DataType::LargeUtf8 => walk!(array.as_string::<i64>()),
        DataType::Utf8View => walk!(array.as_string_view()),
        DataType::Binary => walk!(array.as_binary::<i32>()),
        DataType::LargeBinary => walk!(array.as_binary::<i64>()),
        DataType::BinaryView => walk!(array.as_binary_view()),
        _ => false,
    }
}

/// Every trigram is one of 2^24, so a bitmap of that many bits records which
/// ones a column holds. Two mebibytes, held only while a chunk is encoded.
const SPACE: usize = 1 << (SIZE * 8);

fn index_of(trigram: &[u8]) -> usize {
    (trigram[0] as usize) << 16 | (trigram[1] as usize) << 8 | trigram[2] as usize
}

/// Build a trigram filter over an array's non-null values.
///
/// The filter is sized by the number of *distinct* trigrams, not the number
/// produced. A column of prose repeats its trigrams heavily. The same piece
/// inserted twice tells the filter nothing new. A filter sized for every repeat
/// would be many times larger for no gain.
///
/// That keeps the cost near zero on ordinary text and moderate on identifiers.
///
/// `max_bytes` abandons the filter when it would exceed that size. Only 2^24
/// trigrams exist. A column of near-random bytes approaches all of them: about
/// 21 MiB of filter per chunk.
///
/// Such a filter rules out almost nothing, because a search term's pieces are
/// then always present. Callers pass the size of the column itself, so a filter
/// never outweighs the data it describes.
///
/// No filter prunes nothing. That costs reads, never rows.
pub fn build(
    array: &dyn Array,
    bits_per_value: usize,
    max_bytes: usize,
) -> Result<Option<BloomFilter>> {
    if !supports(array.data_type()) || array.is_empty() {
        return Ok(None);
    }

    let mut seen = vec![0u64; SPACE / 64];
    let mut distinct = 0usize;
    let found = for_each_value(array, |value| {
        for trigram in trigrams(value) {
            let at = index_of(trigram);
            let word = &mut seen[at / 64];
            let bit = 1u64 << (at % 64);
            if *word & bit == 0 {
                *word |= bit;
                distinct += 1;
            }
        }
    });
    if !found || distinct == 0 {
        return Ok(None);
    }

    let mut filter = BloomFilter::with_capacity(distinct, bits_per_value);
    if filter.byte_len() > max_bytes {
        return Ok(None);
    }
    for (index, word) in seen.iter().enumerate() {
        let mut bits = *word;
        while bits != 0 {
            let at = index * 64 + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let trigram = [(at >> 16) as u8, (at >> 8) as u8, at as u8];
            filter.insert_hash(hash(&trigram));
        }
    }
    Ok(Some(filter))
}

/// The literal runs of a `LIKE` pattern, with the wildcards removed.
///
/// `%` and `_` both end a run: `%` stands for any text and `_` for any one
/// character, so neither says anything about the bytes on either side of it.
pub fn fragments(pattern: &str) -> Vec<&str> {
    pattern
        .split(['%', '_'])
        .filter(|run| !run.is_empty())
        .collect()
}

/// The trigrams a value must contain to match this pattern.
///
/// Empty means the pattern says nothing a filter can use, which is the answer
/// for any pattern whose runs are all shorter than a trigram. A caller that
/// gets an empty list must prune nothing.
pub fn required(pattern: &str) -> Vec<[u8; SIZE]> {
    let mut out = Vec::new();
    for fragment in fragments(pattern) {
        for trigram in trigrams(fragment.as_bytes()) {
            let piece = [trigram[0], trigram[1], trigram[2]];
            if !out.contains(&piece) {
                out.push(piece);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, LargeStringArray, StringArray, StringViewArray};

    const BITS: usize = 10;
    /// No practical budget, for the tests that are not about the budget.
    const ROOM: usize = usize::MAX;

    fn strings(values: &[&str]) -> StringArray {
        StringArray::from(values.to_vec())
    }

    fn filter_over(values: &[&str]) -> BloomFilter {
        build(&strings(values), BITS, ROOM)
            .unwrap()
            .expect("a filter")
    }

    fn holds(filter: &BloomFilter, pattern: &str) -> bool {
        let required = required(pattern);
        assert!(!required.is_empty(), "{pattern} gives no trigrams");
        required.iter().all(|t| filter.may_contain_hash(hash(t)))
    }

    #[test]
    fn a_word_is_cut_into_overlapping_pieces() {
        let pieces: Vec<&[u8]> = trigrams(b"hello").collect();
        assert_eq!(pieces, vec![&b"hel"[..], b"ell", b"llo"]);
    }

    #[test]
    fn a_string_shorter_than_a_piece_gives_none() {
        assert_eq!(trigrams(b"ab").count(), 0);
        assert_eq!(trigrams(b"").count(), 0);
        assert_eq!(trigrams(b"abc").count(), 1);
    }

    /// The property the pruning rests on. A substring that is present must
    /// never be reported absent.
    #[test]
    fn every_substring_of_a_stored_value_is_reported_present() {
        let values: Vec<String> = (0..500).map(|i| format!("user{i}@example.com")).collect();
        let refs: Vec<&str> = values.iter().map(String::as_str).collect();
        let filter = filter_over(&refs);

        for value in &values {
            for start in 0..value.len().saturating_sub(SIZE) {
                for end in (start + SIZE)..=value.len() {
                    let substring = &value[start..end];
                    assert!(
                        holds(&filter, &format!("%{substring}%")),
                        "{substring} was reported absent"
                    );
                }
            }
        }
    }

    #[test]
    fn a_substring_no_value_holds_is_reported_absent() {
        let filter = filter_over(&["alpha", "bravo", "charlie"]);
        assert!(!holds(&filter, "%zzz%"));
        assert!(!holds(&filter, "%qqqq%"));
    }

    /// A longer term is sharper, because more pieces must all be present.
    #[test]
    fn a_term_built_from_pieces_of_different_values_can_still_pass() {
        let filter = filter_over(&["hel", "ell", "llo"]);
        // No value holds "hello", yet all three of its pieces are stored, so
        // the filter cannot rule it out. This is the design, not a defect.
        assert!(holds(&filter, "%hello%"));
    }

    #[test]
    fn wildcards_split_a_pattern_into_runs() {
        assert_eq!(fragments("%hello%"), vec!["hello"]);
        assert_eq!(fragments("abc%def"), vec!["abc", "def"]);
        assert_eq!(fragments("a_bcd"), vec!["a", "bcd"]);
        assert_eq!(fragments("%%%"), Vec::<&str>::new());
        assert_eq!(fragments("plain"), vec!["plain"]);
    }

    #[test]
    fn a_pattern_with_no_run_long_enough_requires_nothing() {
        assert!(required("%ab%").is_empty());
        assert!(required("%a%b%").is_empty());
        assert!(required("%").is_empty());
        assert!(required("_").is_empty());
        assert!(!required("%abc%").is_empty());
    }

    #[test]
    fn every_run_contributes_its_pieces() {
        let required = required("%abc%xyz%");
        assert_eq!(required.len(), 2);
        assert!(required.contains(b"abc"));
        assert!(required.contains(b"xyz"));
    }

    #[test]
    fn a_repeated_piece_is_required_once() {
        // "abcabc" yields abc, bca, cab, abc: four pieces, three distinct.
        assert_eq!(required("%abcabc%").len(), 3);
    }

    #[test]
    fn both_halves_of_a_split_pattern_must_be_present() {
        let filter = filter_over(&["alpha"]);
        // "alp" is there and "zzz" is not, so the pattern as a whole fails.
        assert!(!holds(&filter, "%alp%zzz%"));
        assert!(holds(&filter, "%alp%pha%"));
    }

    #[test]
    fn the_large_and_view_string_types_work_too() {
        let large = LargeStringArray::from(vec!["hello world"]);
        let view = StringViewArray::from(vec!["hello world"]);
        for array in [&large as &dyn Array, &view as &dyn Array] {
            let filter = build(array, BITS, ROOM).unwrap().unwrap();
            assert!(holds(&filter, "%world%"));
            assert!(!holds(&filter, "%zzz%"));
        }
    }

    #[test]
    fn multibyte_text_matches_on_bytes() {
        let filter = filter_over(&["naïve café", "日本語のテキスト"]);
        assert!(holds(&filter, "%café%"));
        assert!(holds(&filter, "%日本語%"));
        assert!(!holds(&filter, "%zzzz%"));
    }

    #[test]
    fn a_column_of_short_values_gets_no_filter() {
        assert!(build(&strings(&["a", "bc", ""]), BITS, ROOM)
            .unwrap()
            .is_none());
    }

    #[test]
    fn a_type_with_no_text_gets_no_filter() {
        let numbers = Int64Array::from(vec![1, 2, 3]);
        assert!(build(&numbers, BITS, ROOM).unwrap().is_none());
        assert!(!supports(numbers.data_type()));
    }

    #[test]
    fn nulls_are_skipped() {
        let array = StringArray::from(vec![Some("hello"), None, Some("world")]);
        let filter = build(&array, BITS, ROOM).unwrap().unwrap();
        assert!(holds(&filter, "%hello%"));
        assert!(holds(&filter, "%world%"));
    }

    #[test]
    fn an_empty_column_gets_no_filter() {
        assert!(build(&strings(&[]), BITS, ROOM).unwrap().is_none());
    }

    /// Repeats must not inflate the filter, which is the point of sizing by
    /// distinct pieces.
    #[test]
    fn a_filter_larger_than_its_budget_is_abandoned() {
        let values: Vec<String> = (0..500).map(|i| format!("value number {i} here")).collect();
        let refs: Vec<&str> = values.iter().map(String::as_str).collect();
        let full = build(&strings(&refs), BITS, ROOM).unwrap().unwrap();

        assert!(build(&strings(&refs), BITS, full.byte_len())
            .unwrap()
            .is_some());
        assert!(
            build(&strings(&refs), BITS, full.byte_len() - 1)
                .unwrap()
                .is_none(),
            "a filter that does not fit its budget is not written at all"
        );
    }

    #[test]
    fn repeating_a_value_does_not_grow_the_filter() {
        let once = build(&strings(&["hello world"]), BITS, ROOM)
            .unwrap()
            .unwrap();
        let many: Vec<&str> = std::iter::repeat_n("hello world", 1_000).collect();
        let repeated = build(&strings(&many), BITS, ROOM).unwrap().unwrap();
        assert_eq!(once.byte_len(), repeated.byte_len());
    }
}
