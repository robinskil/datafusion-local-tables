//! Storage engine for Arrow tables held in a single local file.
//!
//! Two table shapes sit on one shared format layer:
//!
//! * a columnar table, for scans, with zone maps, encodings and compression;
//! * a b-tree table, for point lookups and key ranges.
//!
//! Both keep their metadata as rkyv archives, so opening a table reads no more
//! than it must, and column data as raw Arrow buffers, so a scan can hand the
//! page cache straight to a query with no copy in between.
//!
//! This crate holds the format and the engine. The DataFusion providers live in
//! the `datafusion-local-tables` crate.

pub mod columnar;
pub mod config;
pub mod error;
pub mod io;
pub mod layout;
pub mod snapshot;
pub mod table_file;
pub mod wal;

pub use columnar::ColumnarTable;
pub use config::{Compression, Durability, IoBackend, TableOptions};
pub use error::{Error, Result};
pub use layout::{Extent, TableKind};
pub use snapshot::{Snapshot, SnapshotRegistry};
pub use table_file::TableFile;
