//! Tunables for a table handle.

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
    /// Target row count per segment row group.
    pub row_group_rows: usize,
    /// Batch size the scan emits.
    pub scan_batch_rows: usize,
    pub durability: Durability,
    pub io_backend: IoBackend,
    pub compression: Compression,
    /// Try dictionary encoding when a column chunk has few distinct values.
    pub dictionary_encoding: bool,
    /// Try run-length encoding when a column chunk has long runs.
    pub rle_encoding: bool,
    /// Open read-only. Skips the writer lock, permits many processes.
    pub read_only: bool,
}

impl Default for TableOptions {
    fn default() -> Self {
        Self {
            memtable_max_bytes: 64 * 1024 * 1024,
            wal_max_bytes: 256 * 1024 * 1024,
            row_group_rows: 128 * 1024,
            scan_batch_rows: 8192,
            durability: Durability::default(),
            io_backend: IoBackend::default(),
            compression: Compression::default(),
            dictionary_encoding: true,
            rle_encoding: true,
            read_only: false,
        }
    }
}

impl TableOptions {
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
}
