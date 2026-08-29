//! Turning an Arrow array into the buffers a segment stores.
//!
//! Plain encoding is the default and the fast one: the stored bytes *are*
//! Arrow's buffers, so reading them back costs no copy. Dictionary and
//! run-length encoding trade that away for a smaller file, and are only chosen
//! when they actually produce one.

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef};
use arrow_buffer::Buffer;
use arrow_schema::{DataType, Field};
use std::sync::Arc;

use crate::columnar::page::{BufferRole, Codec, Encoding};
use crate::columnar::zonemap::ZoneMap;
use crate::config::TableOptions;
use crate::{Error, Result};

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
    pub zone: ZoneMap,
    /// Buffers in write order. Each is either a slice of the input array, which
    /// costs nothing, or a rebuilt buffer where the input could not be used as
    /// it stood.
    pub buffers: Vec<(BufferRole, Buffer)>,
    pub children: Vec<EncodedColumn>,
}

impl EncodedColumn {
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

/// Encode with Arrow's own layout, slicing rather than copying wherever the
/// input array allows it.
pub fn encode_plain(array: &dyn Array, zone: ZoneMap) -> Result<EncodedColumn> {
    let len = array.len();
    let offset = array.offset();
    let data = array.to_data();
    let mut buffers: Vec<(BufferRole, Buffer)> = Vec::new();

    if let Some(nulls) = array.nulls().filter(|n| n.null_count() > 0) {
        // The bitmap must start at bit zero on disk. `sliced` gives exactly the
        // bits this array covers, and returns the original buffer untouched
        // when the array was never offset.
        buffers.push((BufferRole::Validity, nulls.inner().sliced()));
    }

    match array.data_type() {
        DataType::Null => {}

        DataType::Boolean => {
            let values = array.as_boolean().values();
            buffers.push((BufferRole::Values, values.sliced()));
        }

        DataType::Utf8 | DataType::Binary => {
            push_var_width::<i32>(&data, offset, len, &mut buffers);
        }
        DataType::LargeUtf8 | DataType::LargeBinary => {
            push_var_width::<i64>(&data, offset, len, &mut buffers);
        }

        other => {
            let width = fixed_width(other).ok_or_else(|| {
                Error::Unsupported(format!("{other} columns are not supported yet"))
            })?;
            let values = data.buffers()[0].slice_with_length(offset * width, len * width);
            buffers.push((BufferRole::Values, values));
        }
    }

    Ok(EncodedColumn {
        encoding: Encoding::Plain,
        len: len as u64,
        null_count: array.null_count() as u64,
        dict_len: 0,
        run_count: 0,
        zone,
        buffers,
        children: Vec::new(),
    })
}

/// Push the offsets and values of a variable-width column.
///
/// Offsets are stored based at zero, and values are trimmed to the range the
/// offsets actually cover. An array that already satisfies both is sliced
/// rather than rebuilt.
fn push_var_width<O: arrow_array::OffsetSizeTrait>(
    data: &arrow_data::ArrayData,
    offset: usize,
    len: usize,
    buffers: &mut Vec<(BufferRole, Buffer)>,
) {
    let width = std::mem::size_of::<O>();
    let all_offsets: &[O] = data.buffers()[0].typed_data::<O>();
    let first = all_offsets[offset];
    let last = all_offsets[offset + len];

    let offsets = if first.is_zero() {
        data.buffers()[0].slice_with_length(offset * width, (len + 1) * width)
    } else {
        // Rebase, so the values buffer can start at the first byte this column
        // actually uses instead of carrying everything before it.
        let rebased: Vec<O> = all_offsets[offset..=offset + len]
            .iter()
            .map(|o| *o - first)
            .collect();
        Buffer::from_vec(rebased)
    };

    let values =
        data.buffers()[1].slice_with_length(first.as_usize(), last.as_usize() - first.as_usize());

    buffers.push((BufferRole::Offsets, offsets));
    buffers.push((BufferRole::Values, values));
}

/// Bytes per value for the fixed-width types this format stores.
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
/// Booleans are already one bit per row, and null columns store nothing, so
/// neither can be improved on.
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
/// Returns `None` when the type is unsuitable, or when the encoded form cannot
/// be stored in the shape this format expects.
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
    let values: ArrayRef = dict.values().clone();

    // A dictionary that is nearly as long as the column saves nothing and only
    // adds an indirection on read.
    if values.len() * 2 >= array.len() {
        return Ok(None);
    }

    let keys_plain = encode_plain(dict.keys(), ZoneMap::unknown(0))?;
    let mut buffers = Vec::with_capacity(2);
    for (role, buffer) in keys_plain.buffers {
        buffers.push((
            // The keys carry the column's nulls; their values become the index.
            if role == BufferRole::Values {
                BufferRole::DictKeys
            } else {
                role
            },
            buffer,
        ));
    }

    Ok(Some(EncodedColumn {
        encoding: Encoding::Dictionary,
        len: array.len() as u64,
        null_count: array.null_count() as u64,
        dict_len: values.len() as u64,
        run_count: 0,
        zone: zone.clone(),
        buffers,
        children: vec![encode_plain(values.as_ref(), ZoneMap::unknown(0))?],
    }))
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
    let runs = encoded
        .as_any()
        .downcast_ref::<arrow_array::RunArray<Int32Type>>();
    let Some(runs) = runs else {
        return Ok(None);
    };

    // `RunEndBuffer::len` is the logical row count; the runs themselves are in
    // `values`.
    let ends = runs.run_ends().values();
    let run_count = ends.len();
    // One run per row means the column has no repetition to exploit.
    if run_count * 2 >= array.len() {
        return Ok(None);
    }

    let run_ends = Buffer::from_slice_ref(ends);
    Ok(Some(EncodedColumn {
        encoding: Encoding::RunLength,
        len: array.len() as u64,
        null_count: array.null_count() as u64,
        dict_len: 0,
        run_count: run_count as u64,
        zone: zone.clone(),
        buffers: vec![(BufferRole::RunEnds, run_ends)],
        children: vec![encode_plain(runs.values().as_ref(), ZoneMap::unknown(0))?],
    }))
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

        assert_eq!(roles(&column), vec![BufferRole::Values]);
        assert_eq!(column.null_count, 0);
        assert_eq!(column.buffers[0].1.len(), 12);
    }

    #[test]
    fn a_column_with_nulls_stores_a_bitmap_first() {
        let array = Int32Array::from(vec![Some(1), None, Some(3)]);
        let column = encode_column(&array, &plain_options()).unwrap();

        assert_eq!(
            roles(&column),
            vec![BufferRole::Validity, BufferRole::Values]
        );
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

        assert_eq!(
            roles(&column),
            vec![BufferRole::Offsets, BufferRole::Values]
        );
        let offsets = column.buffers[0].1.typed_data::<i32>();
        assert_eq!(offsets, &[0, 5, 9, 14]);
        assert_eq!(column.buffers[1].1.as_slice(), b"alphabetagamma");
    }

    #[test]
    fn a_sliced_string_column_rebases_its_offsets_and_trims_its_values() {
        let array = StringArray::from(vec!["alpha", "beta", "gamma", "delta"]);
        let sliced = array.slice(1, 2);
        let column = encode_column(&sliced, &plain_options()).unwrap();

        let offsets = column.buffers[0].1.typed_data::<i32>();
        assert_eq!(offsets, &[0, 4, 9], "offsets must start at zero");
        assert_eq!(
            column.buffers[1].1.as_slice(),
            b"betagamma",
            "the values buffer must not carry rows this column does not hold"
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
        let stored = compress_buffers(Codec::Lz4, &[(BufferRole::Values, repetitive)]).unwrap();
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
        let stored = compress_buffers(
            Codec::Lz4,
            &[(BufferRole::Values, Buffer::from_vec(random))],
        )
        .unwrap();
        assert_eq!(
            stored[0].1.codec,
            Codec::None,
            "a buffer that does not shrink must keep the zero-copy path"
        );
    }
}
