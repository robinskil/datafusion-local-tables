//! Turning an Arrow array into the buffers a segment stores.
//!
//! Plain encoding is the default and the fast one: the stored bytes *are*
//! Arrow's buffers, so reading them back costs no copy. Dictionary and
//! run-length encoding trade that away for a smaller file, and are only chosen
//! when they actually produce one.

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{make_array, Array, ArrayRef};
use arrow_buffer::Buffer;
use arrow_schema::{DataType, Field};
use std::sync::Arc;

use crate::columnar::page::{BufferRole, Codec, Encoding};
use crate::columnar::zonemap::ZoneMap;
use crate::config::TableOptions;
use crate::Result;

/// One column, encoded and ready to write.
#[derive(Debug, Clone)]
pub struct EncodedColumn {
    pub encoding: Encoding,
    pub len: u64,
    pub null_count: u64,
    /// Distinct values, when dictionary encoded.
    pub dict_len: u64,
    /// Runs, when run-length encoded.
    pub run_count: u64,
    /// Where the rows start inside the stored buffers. Normally zero.
    pub offset: u64,
    pub zone: ZoneMap,
    /// Buffers in write order. Each is either a slice of the input array, which
    /// costs nothing, or a rebuilt buffer where the input could not be used as
    /// it stood.
    pub buffers: Vec<(BufferRole, Buffer)>,
    pub children: Vec<EncodedColumn>,
}

impl EncodedColumn {
    /// What describes a column as a whole, without encoding any of it.
    ///
    /// For a column stored in blocks: the blocks carry the data, and the chunk
    /// above them carries only the counts and the bounds. Encoding it whole as
    /// well would build buffers nobody keeps.
    pub fn describing(array: &dyn Array) -> Self {
        Self {
            encoding: Encoding::Plain,
            len: array.len() as u64,
            null_count: array.null_count() as u64,
            dict_len: 0,
            run_count: 0,
            offset: 0,
            zone: ZoneMap::build(array),
            buffers: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Bytes this column contributes to the segment, before compression.
    pub fn byte_len(&self) -> usize {
        self.buffers.iter().map(|(_, b)| b.len()).sum::<usize>()
            + self.children.iter().map(|c| c.byte_len()).sum::<usize>()
    }
}

/// Encode one column, choosing the encoding that stores it smallest.
///
/// Plain wins ties, because it is the only encoding a scan can read without
/// copying.
pub fn encode_column(array: &dyn Array, options: &TableOptions) -> Result<EncodedColumn> {
    let zone = ZoneMap::build(array);
    let plain = encode_plain(array, zone.clone())?;

    let mut best = plain;
    if options.dictionary_encoding {
        if let Some(candidate) = try_dictionary(array, &zone)? {
            if candidate.byte_len() < best.byte_len() {
                best = candidate;
            }
        }
    }
    if options.rle_encoding {
        if let Some(candidate) = try_run_length(array, &zone)? {
            if candidate.byte_len() < best.byte_len() {
                best = candidate;
            }
        }
    }
    Ok(best)
}

/// Encode an array exactly as Arrow lays it out.
///
/// This is generic over the type stored, and deliberately so. An Arrow array is
/// a null bitmap, a list of buffers, and a list of child arrays; what those
/// buffers mean is the type's business, not this function's. Storing that shape
/// verbatim means every Arrow type works — nested types, dictionaries, and
/// extension types, whose storage is an ordinary array and whose identity lives
/// in the schema's field metadata.
///
/// Nothing is copied for an array that starts at row zero, which is the common
/// case: the stored bytes are Arrow's own buffers.
pub fn encode_plain(array: &dyn Array, zone: ZoneMap) -> Result<EncodedColumn> {
    // A sliced array's buffers belong to its parent and carry rows this column
    // does not own. Compacting rebuilds them around just these rows. Where
    // Arrow cannot compact a type, the buffers are stored as they stand and the
    // offset is recorded: larger, but still correct.
    let compacted;
    let array = if carries_only_its_own_rows(array) {
        array
    } else {
        match compact(array) {
            Some(rebuilt) => {
                compacted = rebuilt;
                compacted.as_ref()
            }
            None => array,
        }
    };

    let data = array.to_data();
    let mut buffers: Vec<(BufferRole, Buffer)> = Vec::new();

    if let Some(nulls) = array.nulls().filter(|n| n.null_count() > 0) {
        // The bitmap must start at bit zero on disk. `sliced` gives exactly the
        // bits this array covers, and returns the original buffer untouched
        // when the array was never offset.
        buffers.push((BufferRole::Validity, nulls.inner().sliced()));
    }
    for buffer in data.buffers() {
        buffers.push((BufferRole::Data, buffer.clone()));
    }

    let children = data
        .child_data()
        .iter()
        .map(|child| encode_plain(make_array(child.clone()).as_ref(), ZoneMap::unknown(0)))
        .collect::<Result<Vec<_>>>()?;

    Ok(EncodedColumn {
        encoding: Encoding::Plain,
        len: array.len() as u64,
        null_count: array.null_count() as u64,
        dict_len: 0,
        run_count: 0,
        offset: data.offset() as u64,
        zone,
        buffers,
        children,
    })
}

/// True when this array's buffers hold its rows and nothing else.
///
/// A slice can carry more than it owns in two ways: by leaving the buffers
/// alone and setting an offset, or by narrowing one buffer and leaving another
/// whole — which is what Arrow does when it slices a string column, and why
/// checking the offset alone is not enough.
///
/// Arrow can say how many bytes the rows need, and that is compared against
/// the bytes the buffers actually hold. For an array that owns its buffers the
/// two are equal, for every type measured; a slice that carries its parent's
/// values is where they diverge. Buffer *capacity* is deliberately not used:
/// it is rounded up for alignment, which would make every array look wasteful.
fn carries_only_its_own_rows(array: &dyn Array) -> bool {
    if array.offset() != 0 {
        return false;
    }
    // View types need their own measure; see `view_bytes`.
    if let Some((referenced, held)) = view_bytes(array) {
        return held <= referenced;
    }

    let data = array.to_data();
    let Ok(needed) = data.get_slice_memory_size() else {
        // A type Arrow will not measure is stored as it stands.
        return true;
    };
    let held: usize = data.buffers().iter().map(|b| b.len()).sum::<usize>()
        + data.nulls().map_or(0, |n| n.buffer().len())
        + data
            .child_data()
            .iter()
            .map(|child| child.get_slice_memory_size().unwrap_or(0))
            .sum::<usize>();
    held <= needed
}

/// For a view array, the bytes its views actually reference and the bytes its
/// data buffers hold. `None` for every other type.
///
/// View types are the one family the generic measure cannot judge. Arrow's
/// `get_slice_memory_size` counts their views and not the data buffers behind
/// them, so it reports the same figure for a whole array and for a two-row
/// slice of it — which would have this code compacting every view array and
/// still never noticing the one that needed it.
fn view_bytes(array: &dyn Array) -> Option<(usize, usize)> {
    fn measure<T: arrow_array::types::ByteViewType>(
        array: &arrow_array::GenericByteViewArray<T>,
    ) -> (usize, usize) {
        (
            array.total_buffer_bytes_used(),
            array.data_buffers().iter().map(|b| b.len()).sum(),
        )
    }

    match array.data_type() {
        DataType::Utf8View => Some(measure(array.as_string_view())),
        DataType::BinaryView => Some(measure(array.as_binary_view())),
        _ => None,
    }
}

/// Copy an array into fresh buffers holding only its own rows.
///
/// `concat` of a single array short-circuits to a slice, which is exactly what
/// needs undoing, so an empty array of the same type goes in front to take the
/// general path. Returns `None` for the types `concat` does not handle.
///
/// View arrays take a different route: concatenating them keeps the data
/// buffers as they are, so it cannot reclaim anything. `gc` is the operation
/// that rebuilds them around the values the views actually point at.
fn compact(array: &dyn Array) -> Option<ArrayRef> {
    match array.data_type() {
        DataType::Utf8View => return Some(Arc::new(array.as_string_view().gc())),
        DataType::BinaryView => return Some(Arc::new(array.as_binary_view().gc())),
        _ => {}
    }
    let empty = arrow_array::new_empty_array(array.data_type());
    arrow_select::concat::concat(&[empty.as_ref(), array]).ok()
}

/// Bytes per value for the fixed-width types this format stores.
///
/// Used to judge whether a re-encoding is worth trying, not to lay bytes out:
/// the layout is Arrow's.
pub fn fixed_width(data_type: &DataType) -> Option<usize> {
    use DataType::*;
    Some(match data_type {
        Int8 | UInt8 => 1,
        Int16 | UInt16 | Float16 => 2,
        Int32
        | UInt32
        | Float32
        | Date32
        | Time32(_)
        | Interval(arrow_schema::IntervalUnit::YearMonth) => 4,
        Int64
        | UInt64
        | Float64
        | Date64
        | Time64(_)
        | Timestamp(_, _)
        | Duration(_)
        | Interval(arrow_schema::IntervalUnit::DayTime) => 8,
        Decimal128(_, _) | Interval(arrow_schema::IntervalUnit::MonthDayNano) => 16,
        Decimal256(_, _) => 32,
        FixedSizeBinary(width) => *width as usize,
        _ => return None,
    })
}

/// True for types worth trying an alternative encoding on.
///
/// Booleans are already one bit per row and null columns store nothing, so
/// neither can be improved on. A column the schema already declares as a
/// dictionary or a run-length encoding is left alone: it is stored in that form
/// already, and wrapping it again would only add a layer to undo.
fn worth_re_encoding(data_type: &DataType) -> bool {
    !matches!(data_type, DataType::Null | DataType::Boolean)
        && (fixed_width(data_type).is_some()
            || matches!(
                data_type,
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary
            ))
}

/// Encode as distinct values plus one index per row.
///
/// The re-encoded array is stored by the same generic path as any other; only
/// the encoding tag differs, so the reader knows to cast it back.
fn try_dictionary(array: &dyn Array, zone: &ZoneMap) -> Result<Option<EncodedColumn>> {
    if !worth_re_encoding(array.data_type()) || array.is_empty() {
        return Ok(None);
    }

    let target = DataType::Dictionary(
        Box::new(DataType::Int32),
        Box::new(array.data_type().clone()),
    );
    let Ok(encoded) = arrow_cast::cast(array, &target) else {
        return Ok(None);
    };
    let dict = encoded.as_dictionary::<Int32Type>();

    // A dictionary nearly as long as the column saves nothing and only adds an
    // indirection on read.
    if dict.values().len() * 2 >= array.len() {
        return Ok(None);
    }

    let mut column = encode_plain(encoded.as_ref(), zone.clone())?;
    column.encoding = Encoding::Dictionary;
    column.dict_len = dict.values().len() as u64;
    // The counts describe the column, not the form it happens to be stored in:
    // a re-encoded array keeps its nulls somewhere else, and a reader asking
    // how many rows are null means the column's rows.
    column.null_count = array.null_count() as u64;
    Ok(Some(column))
}

/// Encode as run ends plus one value per run.
fn try_run_length(array: &dyn Array, zone: &ZoneMap) -> Result<Option<EncodedColumn>> {
    if !worth_re_encoding(array.data_type()) || array.is_empty() {
        return Ok(None);
    }

    let target = DataType::RunEndEncoded(
        Arc::new(Field::new("run_ends", DataType::Int32, false)),
        Arc::new(Field::new("values", array.data_type().clone(), true)),
    );
    let Ok(encoded) = arrow_cast::cast(array, &target) else {
        return Ok(None);
    };
    let Some(runs) = encoded
        .as_any()
        .downcast_ref::<arrow_array::RunArray<Int32Type>>()
    else {
        return Ok(None);
    };

    // `RunEndBuffer::len` is the logical row count; the runs are in `values`.
    let run_count = runs.run_ends().values().len();
    // One run per row means the column has no repetition to exploit.
    if run_count * 2 >= array.len() {
        return Ok(None);
    }

    let mut column = encode_plain(encoded.as_ref(), zone.clone())?;
    column.encoding = Encoding::RunLength;
    column.run_count = run_count as u64;
    column.null_count = array.null_count() as u64;
    Ok(Some(column))
}

/// Apply `codec` to each buffer, keeping whichever form is smaller.
///
/// Returns, per buffer, the bytes to store and the size they expand to. A
/// buffer that did not shrink is reported as uncompressed, so it keeps the
/// zero-copy read path.
pub fn compress_buffers(
    codec: Codec,
    buffers: &[(BufferRole, Buffer)],
) -> Result<Vec<(BufferRole, StoredBuffer)>> {
    buffers
        .iter()
        .map(|(role, buffer)| {
            let stored = match crate::columnar::codec::compress(codec, buffer.as_slice())? {
                Some(packed) => StoredBuffer {
                    codec,
                    uncompressed_len: buffer.len() as u64,
                    bytes: StoredBytes::Packed(packed),
                },
                None => StoredBuffer {
                    codec: Codec::None,
                    uncompressed_len: buffer.len() as u64,
                    bytes: StoredBytes::Raw(buffer.clone()),
                },
            };
            Ok((*role, stored))
        })
        .collect()
}

/// Bytes as they will land on disk.
#[derive(Debug, Clone)]
pub struct StoredBuffer {
    /// The codec actually applied, which may be `None` even when one was asked
    /// for, because the buffer did not shrink.
    pub codec: Codec,
    pub uncompressed_len: u64,
    pub bytes: StoredBytes,
}

#[derive(Debug, Clone)]
pub enum StoredBytes {
    /// The Arrow buffer itself, stored as it stands.
    Raw(Buffer),
    /// A compressed copy.
    Packed(Vec<u8>),
}

impl StoredBuffer {
    pub fn as_slice(&self) -> &[u8] {
        match &self.bytes {
            StoredBytes::Raw(buffer) => buffer.as_slice(),
            StoredBytes::Packed(bytes) => bytes,
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};

    fn plain_options() -> TableOptions {
        TableOptions {
            dictionary_encoding: false,
            rle_encoding: false,
            ..TableOptions::default()
        }
    }

    fn roles(column: &EncodedColumn) -> Vec<BufferRole> {
        column.buffers.iter().map(|(role, _)| *role).collect()
    }

    #[test]
    fn a_column_without_nulls_stores_no_bitmap() {
        let array = Int32Array::from(vec![1, 2, 3]);
        let column = encode_column(&array, &plain_options()).unwrap();

        assert_eq!(roles(&column), vec![BufferRole::Data]);
        assert_eq!(column.null_count, 0);
        assert_eq!(column.buffers[0].1.len(), 12);
    }

    #[test]
    fn a_column_with_nulls_stores_a_bitmap_first() {
        let array = Int32Array::from(vec![Some(1), None, Some(3)]);
        let column = encode_column(&array, &plain_options()).unwrap();

        assert_eq!(roles(&column), vec![BufferRole::Validity, BufferRole::Data]);
        assert_eq!(column.null_count, 1);
    }

    #[test]
    fn plain_encoding_slices_rather_than_copying() {
        let array = Int32Array::from(vec![1, 2, 3, 4]);
        let source = array.to_data().buffers()[0].clone();
        let column = encode_plain(&array, ZoneMap::unknown(0)).unwrap();

        assert_eq!(
            column.buffers[0].1.as_ptr(),
            source.as_ptr(),
            "an unsliced column must be stored straight from Arrow's own buffer"
        );
    }

    #[test]
    fn a_sliced_column_stores_only_its_own_rows() {
        let array = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
        let sliced = array.slice(2, 3);
        let column = encode_column(&sliced, &plain_options()).unwrap();

        assert_eq!(column.len, 3);
        assert_eq!(column.buffers[0].1.len(), 12, "three values, not six");
        assert_eq!(column.buffers[0].1.typed_data::<i32>(), &[3, 4, 5]);
    }

    #[test]
    fn strings_store_offsets_based_at_zero() {
        let array = StringArray::from(vec!["alpha", "beta", "gamma"]);
        let column = encode_column(&array, &plain_options()).unwrap();

        assert_eq!(roles(&column), vec![BufferRole::Data, BufferRole::Data]);
        let offsets = column.buffers[0].1.typed_data::<i32>();
        assert_eq!(offsets, &[0, 5, 9, 14]);
        assert_eq!(column.buffers[1].1.as_slice(), b"alphabetagamma");
    }

    #[test]
    fn a_sliced_string_column_stores_only_its_own_rows() {
        let array = StringArray::from(vec!["alpha", "beta", "gamma", "delta"]);
        let sliced = array.slice(1, 2);
        let column = encode_column(&sliced, &plain_options()).unwrap();

        assert_eq!(column.offset, 0);
        assert_eq!(
            column.buffers[1].1.as_slice(),
            b"betagamma",
            "the values buffer must not carry rows this column does not hold"
        );
    }

    #[test]
    fn an_unsliced_view_column_is_stored_without_being_copied() {
        use arrow_array::StringViewArray;

        // Long enough that the values live in data buffers rather than inline,
        // which is the case where a needless rebuild would cost something.
        let values: Vec<String> = (0..2_000)
            .map(|i| format!("a value long enough to need a data buffer, number {i}"))
            .collect();
        let array =
            StringViewArray::from(values.iter().map(|v| Some(v.as_str())).collect::<Vec<_>>());
        let source = array.to_data().buffers()[1].as_ptr();

        let column = encode_column(&array, &plain_options()).unwrap();
        assert_eq!(
            column.buffers[1].1.as_ptr(),
            source,
            "a view array that owns its data buffers must not be rebuilt"
        );
    }

    #[test]
    fn an_unsliced_column_is_stored_without_being_copied() {
        let array = StringArray::from(vec!["alpha", "beta", "gamma"]);
        let source = array.to_data().buffers()[1].clone();
        let column = encode_column(&array, &plain_options()).unwrap();

        assert_eq!(
            column.buffers[1].1.as_ptr(),
            source.as_ptr(),
            "an array that owns its buffers must be stored straight from them"
        );
    }

    #[test]
    fn a_repetitive_string_column_picks_dictionary_or_run_length() {
        let values: Vec<&str> = (0..1000)
            .map(|i| if i % 2 == 0 { "yes" } else { "no" })
            .collect();
        let array = StringArray::from(values);

        let plain = encode_column(&array, &plain_options()).unwrap();
        let chosen = encode_column(&array, &TableOptions::default()).unwrap();

        assert_ne!(chosen.encoding, Encoding::Plain);
        assert!(
            chosen.byte_len() < plain.byte_len(),
            "an alternative encoding is only worth choosing when it is smaller"
        );
    }

    #[test]
    fn a_long_run_prefers_run_length() {
        // One value repeated: two runs at most, versus one dictionary index
        // per row.
        let array = StringArray::from(vec!["same"; 2000]);
        let column = encode_column(&array, &TableOptions::default()).unwrap();

        assert_eq!(column.encoding, Encoding::RunLength);
        assert_eq!(column.run_count, 1);
    }

    #[test]
    fn a_column_of_distinct_values_stays_plain() {
        let array = Int32Array::from((0..1000).collect::<Vec<i32>>());
        let column = encode_column(&array, &TableOptions::default()).unwrap();

        assert_eq!(
            column.encoding,
            Encoding::Plain,
            "nothing repeats, so neither alternative can pay for itself"
        );
    }

    #[test]
    fn booleans_are_never_re_encoded() {
        let array = arrow_array::BooleanArray::from(vec![true; 1000]);
        let column = encode_column(&array, &TableOptions::default()).unwrap();
        assert_eq!(
            column.encoding,
            Encoding::Plain,
            "one bit per row is already the floor"
        );
    }

    #[test]
    fn an_empty_column_encodes_to_empty_buffers() {
        let array = Int32Array::from(Vec::<i32>::new());
        let column = encode_column(&array, &TableOptions::default()).unwrap();

        assert_eq!(column.len, 0);
        assert_eq!(column.encoding, Encoding::Plain);
        assert_eq!(column.byte_len(), 0);
    }

    #[test]
    fn compression_keeps_whichever_form_is_smaller() {
        let repetitive = Buffer::from_vec(vec![7u8; 8192]);
        let stored = compress_buffers(Codec::Lz4, &[(BufferRole::Data, repetitive)]).unwrap();
        assert_eq!(stored[0].1.codec, Codec::Lz4);
        assert!(stored[0].1.len() < 8192);
        assert_eq!(stored[0].1.uncompressed_len, 8192);

        let mut state = 12345u64;
        let random: Vec<u8> = (0..8192)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect();
        let stored =
            compress_buffers(Codec::Lz4, &[(BufferRole::Data, Buffer::from_vec(random))]).unwrap();
        assert_eq!(
            stored[0].1.codec,
            Codec::None,
            "a buffer that does not shrink must keep the zero-copy path"
        );
    }
}
