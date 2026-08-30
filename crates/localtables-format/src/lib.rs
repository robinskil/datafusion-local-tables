//! Storage engine for Arrow tables held in a single local file.
//!
//! One table shape: a columnar table built for scans. It has zone maps,
//! membership filters, per-column encodings and per-column compression.
//!
//! Metadata sits in rkyv archives, so a table open reads no more than it must.
//! Column data sits in raw Arrow buffers, so a scan gives the page cache
//! straight to a query with no copy between.
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
pub mod valuecodec;
pub mod wal;

pub use columnar::ColumnarTable;
pub use config::{BloomFilters, Compression, Durability, IoBackend, TableOptions};
pub use error::{Error, Result};
pub use layout::{Extent, TableKind};
pub use snapshot::{Snapshot, SnapshotRegistry};
pub use table_file::TableFile;
