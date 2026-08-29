//! Positional reads and writes on a blocking thread pool.
//!
//! This backend works on every platform. It is the fallback when mmap is
//! compiled out, and the reference the other backends are checked against.

use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;

use crate::config::{Durability, IoBackend};
use crate::io::buf::{IoBuf, SharedBuf};
use crate::io::{blocking, sync_file, FileIo};
use crate::layout::Extent;
use crate::{Error, Result};

/// A table file accessed through `pread` and `pwrite`.
pub struct StdFileIo {
    file: Arc<File>,
    path: PathBuf,
    durability: Durability,
    read_only: bool,
    /// Serialises appends so two writers cannot pick the same offset.
    append_lock: Arc<Mutex<()>>,
}

impl StdFileIo {
    /// Open an existing file, or create an empty one.
    pub fn open(path: &Path, durability: Durability, read_only: bool) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(!read_only)
            .create(!read_only)
            .truncate(false)
            .open(path)
            .map_err(|e| Error::io(path.to_path_buf(), e))?;
        Ok(Self::from_file(
            file,
            path.to_path_buf(),
            durability,
            read_only,
        ))
    }

    pub fn from_file(file: File, path: PathBuf, durability: Durability, read_only: bool) -> Self {
        Self {
            file: Arc::new(file),
            path,
            durability,
            read_only,
            append_lock: Arc::new(Mutex::new(())),
        }
    }

    /// The underlying handle, for taking the advisory lock.
    pub fn file(&self) -> &Arc<File> {
        &self.file
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn reject_writes(&self) -> Result<()> {
        if self.read_only {
            return Err(Error::InvalidArgument(format!(
                "{} is open read-only",
                self.path.display()
            )));
        }
        Ok(())
    }

    /// Read into an aligned buffer on the calling thread.
    fn read_exact_blocking(file: &File, path: &Path, offset: u64, len: usize) -> Result<IoBuf> {
        let mut buf = IoBuf::uninit(len);
        if len > 0 {
            file.read_exact_at(buf.as_mut_slice(), offset)
                .map_err(|e| Error::io(path.to_path_buf(), e))?;
        }
        Ok(buf)
    }
}

#[async_trait]
impl FileIo for StdFileIo {
    async fn read_at(&self, offset: u64, len: usize) -> Result<SharedBuf> {
        let file = self.file.clone();
        let path = self.path.clone();
        blocking(move || {
            Ok(SharedBuf::from_owned(Self::read_exact_blocking(
                &file, &path, offset, len,
            )?))
        })
        .await
    }

    async fn read_immutable(&self, extent: Extent) -> Result<SharedBuf> {
        self.read_at(extent.offset, extent.len as usize).await
    }

    async fn read_scattered(&self, extents: &[Extent]) -> Result<Vec<SharedBuf>> {
        let file = self.file.clone();
        let path = self.path.clone();
        let extents = extents.to_vec();
        blocking(move || {
            // One hop to the pool for the whole set, not one per extent.
            extents
                .iter()
                .map(|e| {
                    Self::read_exact_blocking(&file, &path, e.offset, e.len as usize)
                        .map(SharedBuf::from_owned)
                })
                .collect()
        })
        .await
    }

    async fn append(&self, bufs: &[&[u8]]) -> Result<u64> {
        self.reject_writes()?;
        let total: usize = bufs.iter().map(|b| b.len()).sum();
        let joined = {
            let mut joined = Vec::with_capacity(total);
            for b in bufs {
                joined.extend_from_slice(b);
            }
            joined
        };
        let file = self.file.clone();
        let path = self.path.clone();
        let append_lock = self.append_lock.clone();
        blocking(move || {
            // Pick the offset and write it under one lock, so two appends
            // cannot both claim the same end of file.
            let _guard = append_lock.lock();
            let offset = file
                .metadata()
                .map_err(|e| Error::io(path.clone(), e))?
                .len();
            file.write_all_at(&joined, offset)
                .map_err(|e| Error::io(path, e))?;
            Ok(offset)
        })
        .await
    }

    async fn write_at(&self, offset: u64, buf: &[u8]) -> Result<()> {
        self.reject_writes()?;
        let file = self.file.clone();
        let path = self.path.clone();
        let owned = buf.to_vec();
        blocking(move || {
            file.write_all_at(&owned, offset)
                .map_err(|e| Error::io(path, e))
        })
        .await
    }

    async fn set_len(&self, len: u64) -> Result<()> {
        self.reject_writes()?;
        let file = self.file.clone();
        let path = self.path.clone();
        blocking(move || file.set_len(len).map_err(|e| Error::io(path, e))).await
    }

    async fn sync_data(&self) -> Result<()> {
        if self.read_only {
            return Ok(());
        }
        let file = self.file.clone();
        let path = self.path.clone();
        let durability = self.durability;
        blocking(move || sync_file(&file, durability).map_err(|e| Error::io(path, e))).await
    }

    fn len(&self) -> Result<u64> {
        Ok(self
            .file
            .metadata()
            .map_err(|e| Error::io(self.path.clone(), e))?
            .len())
    }

    fn backend(&self) -> IoBackend {
        IoBackend::Pread
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn io(dir: &tempfile::TempDir) -> StdFileIo {
        StdFileIo::open(&dir.path().join("table.lt"), Durability::None, false).unwrap()
    }

    #[tokio::test]
    async fn append_returns_the_offset_it_wrote_at() {
        let dir = tempfile::tempdir().unwrap();
        let io = io(&dir);

        assert_eq!(io.append(&[b"hello"]).await.unwrap(), 0);
        assert_eq!(io.append(&[b" ", b"world"]).await.unwrap(), 5);
        assert_eq!(io.len().unwrap(), 11);
        assert_eq!(io.read_at(0, 11).await.unwrap().as_slice(), b"hello world");
    }

    #[tokio::test]
    async fn write_at_overwrites_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let io = io(&dir);
        io.append(&[b"aaaaaaaa"]).await.unwrap();
        io.write_at(2, b"BB").await.unwrap();
        assert_eq!(io.read_at(0, 8).await.unwrap().as_slice(), b"aaBBaaaa");
    }

    #[tokio::test]
    async fn scattered_reads_return_each_range() {
        let dir = tempfile::tempdir().unwrap();
        let io = io(&dir);
        io.append(&[b"0123456789"]).await.unwrap();

        let bufs = io
            .read_scattered(&[Extent::new(0, 3), Extent::new(7, 3), Extent::new(4, 1)])
            .await
            .unwrap();
        let got: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
        assert_eq!(got, vec![&b"012"[..], &b"789"[..], &b"4"[..]]);
    }

    #[tokio::test]
    async fn reading_past_the_end_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let io = io(&dir);
        io.append(&[b"short"]).await.unwrap();
        assert!(io.read_at(0, 100).await.is_err());
    }

    #[tokio::test]
    async fn read_only_handles_refuse_writes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("table.lt");
        StdFileIo::open(&path, Durability::None, false)
            .unwrap()
            .append(&[b"data"])
            .await
            .unwrap();

        let ro = StdFileIo::open(&path, Durability::None, true).unwrap();
        assert_eq!(ro.read_at(0, 4).await.unwrap().as_slice(), b"data");
        assert!(ro.append(&[b"more"]).await.is_err());
        assert!(ro.write_at(0, b"x").await.is_err());
    }

    #[tokio::test]
    async fn empty_reads_return_an_empty_buffer() {
        let dir = tempfile::tempdir().unwrap();
        let io = io(&dir);
        assert!(io.read_at(0, 0).await.unwrap().is_empty());
    }
}
