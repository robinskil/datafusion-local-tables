//! Tunables for a table handle.

/// Row groups a flush leaves in the table.
///
/// This is a floor on the count, not a target. A larger table gets larger
/// groups, not more of them.
///
/// A scan needs one group per thread, and a few spare. It does not need more.
/// Each segment costs one mapping, one metadata frame, and one set of zone
/// maps. That cost is about five microseconds.
///
/// The same 500,000 rows, cut different ways, at four partitions: 317 us in 5
/// segments, 303 in 10, 324 in 20, 458 in 70. Eight groups sits in the flat
/// part of that curve.
///
/// See `docs/performance.md` for the full measurement.
pub const TARGET_ROW_GROUPS: usize = 8;

/// How hard a commit pushes bytes toward the media.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Durability {
    /// `fdatasync` on Linux, `fsync` on macOS. The drive cache may still hold
    /// the data after the call returns on macOS.
    #[default]
    Os,
    /// `F_FULLFSYNC` on macOS, `fdatasync` elsewhere. Slower, survives power loss.
    Full,
    /// No sync at all. Fast, unsafe. Tests and throwaway tables only.
    None,
}

/// Which [`crate::io::FileIo`] backend a table reads through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IoBackend {
    /// Map segments and read them without syscalls or copies.
    #[default]
    Mmap,
    /// Positional reads on a blocking thread pool. Portable everywhere.
    Pread,
    /// io_uring reactor. Linux only, needs the `uring` feature.
    Uring,
}

/// Which columns get a membership filter.
///
/// A filter answers `col = x` where a zone map cannot. On a column of scattered
/// values, every segment's range spans the value. No segment is ruled out, and
/// the scan reads all of them.
///
/// A filter costs [`TableOptions::bloom_bits_per_value`] bits per distinct
/// value. It is off by default. Ask for it per column, as parquet does.
///
/// It pays on a column with many distinct values that queries compare by
/// equality. It does not pay on a low-cardinality column, where a zone map or a
/// dictionary already answers. It does not pay on a column compared only by
/// range.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum BloomFilters {
    /// No filters. Equality falls back to zone maps.
    #[default]
    None,
    /// Every column whose type has a canonical byte form.
    All,
    /// The named columns, where their type allows it.
    Columns(Vec<String>),
}

impl BloomFilters {
    /// True when this column should get a filter.
    pub fn covers(&self, column: &str) -> bool {
        match self {
            Self::None => false,
            Self::All => true,
            Self::Columns(names) => names.iter().any(|name| name == column),
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// How column bytes are compressed.
///
/// Compression is the one thing that costs a column its zero-copy read path.
/// The reader must decompress a compressed chunk into a buffer it owns. It
/// gives Arrow an uncompressed chunk as the mapped bytes themselves.
///
/// So the question is not only how small a codec makes a column. The question
/// is whether the column had anything to gain.
///
/// Measured over 500,000 rows. The ratio is uncompressed over compressed:
///
/// | column | lz4 | zstd |
/// | --- | --- | --- |
/// | random u64 | 1.00x | 1.00x |
/// | text | 4.4x | 24x |
///
/// A column of scattered numbers does not compress at all, with either codec.
/// Text does, heavily. That is what [`Compression::Auto`] acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compression {
    /// Store raw Arrow bytes. Keeps the zero-copy read path everywhere.
    ///
    /// This is the default. A codec is the one thing that takes that path
    /// away, and the format exists to have it.
    ///
    /// On a table whose bulk is text, a codec costs 2.3x on a full scan and
    /// returns 2.3x the size. On a table whose text is low cardinality it costs
    /// 2%. The data decides which. So the cheap answer is the default, and
    /// [`Compression::Auto`] is one line away.
    #[default]
    None,
    /// Compress the columns that gain by it and leave the rest raw.
    ///
    /// Text and binary get lz4. Every other type stays as it stands.
    ///
    /// lz4 rather than zstd: it decompresses two to three times faster and
    /// still gets most of the size. The text column is usually the bulk of the
    /// file.
    ///
    /// Measured over 500,000 rows, against raw: 14% smaller, and 2% slower on
    /// the worst of three queries. A codec on every column instead reaches 42%
    /// smaller with lz4 and 69% with zstd. It costs 42% and 202% on a read of
    /// every column.
    ///
    /// On a table that is mostly high-cardinality text the trade is sharper:
    /// 2.3x smaller, point lookups a quarter faster, full scans 2.3x slower.
    /// Ask for it when size matters more than scan speed. That is why it is not
    /// the default.
    Auto,
    /// lz4 for every column, whatever it holds.
    Lz4,
    /// zstd for every column.
    Zstd,
}

impl Compression {
    /// The codec to store one column with.
    pub fn codec_for(&self, data_type: &arrow_schema::DataType) -> crate::columnar::page::Codec {
        use crate::columnar::page::Codec;
        match self {
            Self::None => Codec::None,
            Self::Lz4 => Codec::Lz4,
            Self::Zstd => Codec::Zstd,
            Self::Auto if is_text(data_type) => Codec::Lz4,
            Self::Auto => Codec::None,
        }
    }
}

/// True for the types whose bytes are worth compression.
///
/// Text and binary qualify. So does either one inside a dictionary, which is
/// where a low-cardinality string column keeps its values.
fn is_text(data_type: &arrow_schema::DataType) -> bool {
    use arrow_schema::DataType;
    match data_type {
        DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView => true,
        DataType::Dictionary(_, values) => is_text(values),
        _ => false,
    }
}

/// Options a caller passes to open or create a table.
#[derive(Debug, Clone)]
pub struct TableOptions {
    /// Flush the memtable once it holds this many bytes.
    pub memtable_max_bytes: usize,
    /// Flush once the active WAL file grows past this size.
    pub wal_max_bytes: u64,
    /// Largest row group a flush will write.
    ///
    /// A flush aims for [`TARGET_ROW_GROUPS`] groups across the table. It
    /// clamps the result between [`TableOptions::min_row_group_rows`] and this
    /// value. A small table stays divisible. A large table does not collect
    /// metadata for row groups nobody needs.
    pub row_group_rows: usize,
    /// Smallest row group a flush will write.
    ///
    /// Each segment costs one mapping, one metadata frame, and one set of zone
    /// maps. Below this size those costs outweigh what the division buys.
    pub min_row_group_rows: usize,
    /// Batch size the scan emits.
    pub scan_batch_rows: usize,
    pub durability: Durability,
    pub io_backend: IoBackend,
    pub compression: Compression,
    /// Try dictionary encoding when a column chunk has few distinct values.
    pub dictionary_encoding: bool,
    /// Try run-length encoding when a column chunk has long runs.
    pub rle_encoding: bool,
    /// Columns whose bits interleave to order rows before a flush writes them.
    ///
    /// Empty leaves rows in the order they arrived. Zone maps are then
    /// selective on the column that order follows, and on no other.
    ///
    /// A z-order makes them selective on all of these columns at once. It also
    /// makes them less selective on each one than a plain sort would. See
    /// `columnar::zorder`.
    ///
    /// This is a layout, not an index. It stores no extra bytes. It cannot
    /// change what a query returns.
    pub cluster_by: Vec<String>,
    /// Rows covered by each page of bounds inside a segment.
    ///
    /// A segment's own zone map decides whether to read the segment. These
    /// bounds decide which row ranges inside it the scan hands on.
    ///
    /// A predicate that matches a hundred rows of a hundred thousand then costs
    /// the filter above one page, not the whole segment.
    ///
    /// Zero switches the bounds off. They cost about a tenth of a percent of a
    /// segment, so they are on by default.
    pub page_rows: usize,
    /// Rows in each independently compressed block.
    ///
    /// This is separate from [`TableOptions::page_rows`] on purpose. A zone map
    /// costs bytes and no processor time, so a scan can prune finer than it
    /// decompresses. A block is the unit a scan must decompress to reach any
    /// row inside it.
    ///
    /// A compressed column is always cut. A variable-width column is cut even
    /// when it is not compressed: Arrow checks every offset of such a column,
    /// whatever range a reader asks for. A fixed-width column is never cut.
    ///
    /// Small blocks cost compression ratio. How much depends on the codec. lz4
    /// looks back 64 KiB whatever it gets, so 8,192-row blocks cost it about
    /// 3%. zstd looks much further and loses more.
    ///
    /// The default matches `page_rows`. A page that a predicate rules out then
    /// costs nothing to decompress.
    pub compression_block_rows: usize,
    /// Source bytes a rewrite holds in memory at once.
    ///
    /// Compaction and every rewrite read stored rows back, then write them out
    /// again. To read all of them first is simple and unbounded. A table larger
    /// than memory could then never be compacted, and its schema could never
    /// change.
    ///
    /// The work is cut into runs instead. The source segments of one run total
    /// no more than this many bytes on disk. A run always holds at least one
    /// segment, so one segment larger than the budget is the floor.
    ///
    /// A z-order applies within a run. A table larger than the budget comes out
    /// clustered per run, not as a whole. Raise this value to trade memory for
    /// a better layout.
    pub compaction_max_bytes: u64,
    /// Which columns get a membership filter.
    pub bloom_filters: BloomFilters,
    /// Which text columns get a trigram filter, for `LIKE` pruning.
    ///
    /// This filter holds three-byte pieces of every value, not whole values. A
    /// substring search can then rule out the segments that cannot hold it.
    /// See `columnar::trigram`.
    pub trigram_filters: BloomFilters,
    /// Bits a membership filter spends per value.
    ///
    /// More bits give fewer false positives and a larger filter. Ten give about
    /// one in a hundred. A false positive costs a segment read, never a row.
    pub bloom_bits_per_value: usize,
    /// Open read-only. Skips the writer lock, permits many processes.
    pub read_only: bool,
}

impl Default for TableOptions {
    fn default() -> Self {
        Self {
            memtable_max_bytes: 64 * 1024 * 1024,
            wal_max_bytes: 256 * 1024 * 1024,
            row_group_rows: 128 * 1024,
            min_row_group_rows: 8 * 1024,
            scan_batch_rows: 8192,
            durability: Durability::default(),
            io_backend: IoBackend::default(),
            compression: Compression::default(),
            dictionary_encoding: true,
            rle_encoding: true,
            cluster_by: Vec::new(),
            page_rows: 8 * 1024,
            compression_block_rows: 8 * 1024,
            compaction_max_bytes: 256 * 1024 * 1024,
            bloom_filters: BloomFilters::default(),
            trigram_filters: BloomFilters::default(),
            bloom_bits_per_value: crate::columnar::bloom::DEFAULT_BITS_PER_VALUE,
            read_only: false,
        }
    }
}

impl TableOptions {
    /// The row group size to write, given how many rows the table will hold.
    ///
    /// Small tables get small groups so a scan can still divide them; the size
    /// grows with the table until it reaches the maximum, after which the
    /// number of groups grows instead.
    pub fn row_group_size_for(&self, total_rows: u64) -> usize {
        if self.row_group_rows == 0 {
            return 0;
        }
        let even = (total_rows as usize).div_ceil(TARGET_ROW_GROUPS.max(1));
        even.clamp(
            self.min_row_group_rows.min(self.row_group_rows).max(1),
            self.row_group_rows,
        )
    }

    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    pub fn with_durability(mut self, durability: Durability) -> Self {
        self.durability = durability;
        self
    }

    pub fn with_io_backend(mut self, backend: IoBackend) -> Self {
        self.io_backend = backend;
        self
    }

    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    pub fn with_cluster_by<I, S>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.cluster_by = columns.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_bloom_filters(mut self, filters: BloomFilters) -> Self {
        self.bloom_filters = filters;
        self
    }

    pub fn with_trigram_filters(mut self, filters: BloomFilters) -> Self {
        self.trigram_filters = filters;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> TableOptions {
        TableOptions::default()
    }

    #[test]
    fn a_small_table_gets_the_smallest_groups() {
        let options = options();
        // Dividing evenly would give groups far below the floor, so the floor
        // decides and the table simply holds fewer groups.
        assert_eq!(
            options.row_group_size_for(1_000),
            options.min_row_group_rows
        );
        assert_eq!(options.row_group_size_for(0), options.min_row_group_rows);
    }

    #[test]
    fn groups_grow_with_the_table() {
        let options = options();
        let small = options.row_group_size_for(200_000);
        let medium = options.row_group_size_for(1_000_000);
        assert!(
            small < medium,
            "a bigger table should get bigger groups: {small} then {medium}"
        );
        assert_eq!(medium, 1_000_000 / TARGET_ROW_GROUPS);
        assert!(medium <= options.row_group_rows);
    }

    #[test]
    fn growth_stops_at_the_maximum() {
        let options = options();
        assert_eq!(
            options.row_group_size_for(100_000_000),
            options.row_group_rows,
            "past the cap the table gains groups rather than bigger ones"
        );
    }

    #[test]
    fn a_large_table_is_divided_into_at_least_the_target_number() {
        let options = options();
        for rows in [200_000u64, 1_000_000, 4_000_000] {
            let size = options.row_group_size_for(rows);
            let groups = (rows as usize).div_ceil(size);
            assert!(
                groups >= TARGET_ROW_GROUPS,
                "{rows} rows in groups of {size} gives only {groups}"
            );
        }
    }

    #[test]
    fn a_maximum_of_zero_means_one_group() {
        let options = TableOptions {
            row_group_rows: 0,
            ..TableOptions::default()
        };
        assert_eq!(options.row_group_size_for(1_000_000), 0);
    }

    #[test]
    fn filters_are_off_until_they_are_asked_for() {
        assert!(TableOptions::default().bloom_filters.is_none());
        assert!(!TableOptions::default().bloom_filters.covers("id"));
    }

    #[test]
    fn named_columns_get_filters_and_others_do_not() {
        let filters = BloomFilters::Columns(vec!["id".to_string(), "email".to_string()]);
        assert!(filters.covers("id"));
        assert!(filters.covers("email"));
        assert!(!filters.covers("name"));
        assert!(!filters.covers("i"));
    }

    #[test]
    fn asking_for_all_covers_every_name() {
        assert!(BloomFilters::All.covers("anything at all"));
    }

    #[test]
    fn a_floor_above_the_cap_does_not_invert_them() {
        let options = TableOptions {
            row_group_rows: 1_000,
            min_row_group_rows: 100_000,
            ..TableOptions::default()
        };
        assert_eq!(
            options.row_group_size_for(10),
            1_000,
            "the cap wins, so the size never exceeds what was asked for"
        );
    }
}
