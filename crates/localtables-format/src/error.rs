//! Error type for the storage engine.

use std::path::PathBuf;

/// Result alias used across the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// All failures the storage engine reports.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("io error: {0}")]
    RawIo(#[from] std::io::Error),

    #[error("arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),

    /// The file is not a table file, or the version does not match.
    #[error("not a valid table file: {0}")]
    BadMagic(String),

    /// A checksum does not match the bytes it covers.
    #[error("checksum mismatch in {region}: expected {expected:#018x}, found {found:#018x}")]
    Checksum {
        region: &'static str,
        expected: u64,
        found: u64,
    },

    /// Structural corruption that no fallback slot can repair.
    #[error("corrupt table file: {0}")]
    Corrupt(String),

    /// Another process holds the writer lock on this table.
    #[error("table {0} is already open for writing by another process")]
    WriterLocked(PathBuf),

    /// The caller supplied data that does not match the table schema.
    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),

    /// A feature is compiled out, or the file needs it to be read.
    #[error("unsupported: {0}")]
    Unsupported(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// A background task (WAL group commit, uring reactor) stopped.
    #[error("background task stopped: {0}")]
    TaskStopped(&'static str),
}

impl Error {
    /// Attach a path to an [`std::io::Error`].
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }

    pub fn corrupt(msg: impl Into<String>) -> Self {
        Error::Corrupt(msg.into())
    }
}

/// Convert an rkyv validation failure into a corruption error.
impl From<rkyv::rancor::Error> for Error {
    fn from(value: rkyv::rancor::Error) -> Self {
        Error::Corrupt(format!("rkyv: {value}"))
    }
}
