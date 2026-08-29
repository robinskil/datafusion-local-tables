//! Tunables for a table handle.

/// Row groups a flush aims to leave the table holding.
///
/// A floor on the count rather than a target: passing it means the groups grow
/// instead. Enough that a scan has something for every thread with room to
/// spare, and no more, because a segment is not free — it costs a mapping, a
/// metadata frame to check and a set of zone maps, measured at roughly five
/// microseconds each. Scanning the same 500,000 rows cut different ways, at
/// four partitions: 317 us in 5 segments, 303 in 10, 324 in 20, 458 in 70.
/// Eight groups puts a table in that flat region without running past it.
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
/// A filter answers `col = x` where a zone map cannot: on a column of scattered
/// values every segment's range spans the value, so no segment is ruled out and
/// the scan reads all of them. It costs
/// [`TableOptions::bloom_bits_per_value`] bits for every non-null value, so it
/// is off by default and asked for per column, the way parquet asks.
///
/// It pays on a column that is looked up by equality and has many distinct
/// values. It does not pay on a low-cardinality column, where a zone map or a
/// dictionary already answers, nor on one only ever compared by range.
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

/// Per-page compression codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compression {
    /// Store raw Arrow bytes. Keeps the zero-copy read path.
    #[default]
    None,
    Lz4,
    Zstd,
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
    /// A flush aims for [`TARGET_ROW_GROUPS`] groups across the table and
    /// clamps the result between [`TableOptions::min_row_group_rows`] and this,
    /// so a small table is still divisible and a large one does not accumulate
    /// metadata for row groups nobody needs.
    pub row_group_rows: usize,
    /// Smallest row group a flush will write.
    ///
    /// Below this the per-segment costs — a mapping, a metadata frame, a set of
    /// zone maps — start to outweigh what dividing the work buys.
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
    /// Columns whose bits are interleaved to order rows before they are
    /// written.
    ///
    /// Empty means rows keep the order they arrived in, which makes zone maps
    /// selective on whatever column that order follows and on nothing else. A
    /// z-order makes them selective on all of these at once, and none of them
    /// as well as a plain sort would. See `columnar::zorder`.
    ///
    /// This is a layout, not an index: it stores no extra bytes and cannot
    /// affect what a query returns.
    pub cluster_by: Vec<String>,
    /// Which columns get a membership filter.
    pub bloom_filters: BloomFilters,
    /// Bits a membership filter spends per value.
    ///
    /// More bits means fewer false positives and a larger filter. Ten gives
    /// roughly one in a hundred, which costs a segment read and never a row.
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
            bloom_filters: BloomFilters::default(),
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
