//! Advisory file locks.
//!
//! One process at a time may write a table. Readers in other processes take a
//! shared lock, so they never block each other and never block the writer out.
//! The lock lives on the table file itself, so no sidecar lockfile is needed.

use std::fs::File;
use std::path::{Path, PathBuf};

use crate::{Error, Result};

/// A held advisory lock. Dropping it releases the lock.
///
/// The lock owns its own handle on purpose. An advisory lock belongs to the
/// open file description, not to the descriptor.
///
/// A lock taken on a `try_clone` of another handle would outlive this value. It
/// would release only when that other handle closed.
#[derive(Debug)]
pub struct FileLock {
    /// Closing this handle releases the lock. Nothing else refers to it.
    _file: File,
    path: PathBuf,
    exclusive: bool,
}

impl FileLock {
    /// Open `path` and take the writer lock without waiting.
    ///
    /// Returns [`Error::WriterLocked`] when another handle already holds it.
    pub fn try_exclusive(path: &Path) -> Result<Self> {
        let file = Self::open(path)?;
        match file.try_lock() {
            Ok(()) => Ok(Self {
                _file: file,
                path: path.to_path_buf(),
                exclusive: true,
            }),
            Err(std::fs::TryLockError::WouldBlock) => Err(Error::WriterLocked(path.to_path_buf())),
            Err(std::fs::TryLockError::Error(e)) => Err(Error::io(path.to_path_buf(), e)),
        }
    }

    /// Open `path` and take a reader lock without waiting.
    ///
    /// Many readers share the lock. It only fails while a writer holds the file.
    pub fn try_shared(path: &Path) -> Result<Self> {
        let file = Self::open(path)?;
        match file.try_lock_shared() {
            Ok(()) => Ok(Self {
                _file: file,
                path: path.to_path_buf(),
                exclusive: false,
            }),
            Err(std::fs::TryLockError::WouldBlock) => Err(Error::WriterLocked(path.to_path_buf())),
            Err(std::fs::TryLockError::Error(e)) => Err(Error::io(path.to_path_buf(), e)),
        }
    }

    /// Open an existing table file for locking only. Never creates it.
    fn open(path: &Path) -> Result<File> {
        File::open(path).map_err(|e| Error::io(path.to_path_buf(), e))
    }

    pub fn is_exclusive(&self) -> bool {
        self.exclusive
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(dir: &tempfile::TempDir) -> PathBuf {
        let path = dir.path().join("table.lt");
        std::fs::write(&path, b"table bytes").unwrap();
        path
    }

    #[test]
    fn a_second_writer_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = table(&dir);

        let held = FileLock::try_exclusive(&path).unwrap();
        assert!(held.is_exclusive());

        let err = FileLock::try_exclusive(&path).unwrap_err();
        assert!(matches!(err, Error::WriterLocked(_)), "got {err:?}");

        drop(held);
        FileLock::try_exclusive(&path).expect("lock is free after the holder drops");
    }

    #[test]
    fn readers_share_the_lock() {
        let dir = tempfile::tempdir().unwrap();
        let path = table(&dir);

        let first = FileLock::try_shared(&path).unwrap();
        let second = FileLock::try_shared(&path).unwrap();
        assert!(!first.is_exclusive() && !second.is_exclusive());
    }

    #[test]
    fn a_writer_blocks_readers() {
        let dir = tempfile::tempdir().unwrap();
        let path = table(&dir);

        let held = FileLock::try_exclusive(&path).unwrap();
        let err = FileLock::try_shared(&path).unwrap_err();
        assert!(matches!(err, Error::WriterLocked(_)), "got {err:?}");

        drop(held);
        FileLock::try_shared(&path).expect("readers get in once the writer leaves");
    }

    #[test]
    fn readers_block_a_writer() {
        let dir = tempfile::tempdir().unwrap();
        let path = table(&dir);

        let held = FileLock::try_shared(&path).unwrap();
        let err = FileLock::try_exclusive(&path).unwrap_err();
        assert!(matches!(err, Error::WriterLocked(_)), "got {err:?}");
        drop(held);
    }

    #[test]
    fn locking_a_missing_file_reports_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let err = FileLock::try_exclusive(&dir.path().join("absent.lt")).unwrap_err();
        assert!(matches!(err, Error::Io { .. }), "got {err:?}");
    }
}
