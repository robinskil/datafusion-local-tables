//! An IO backend that stops writing partway through, to stand in for a crash.
//!
//! Crash safety is the hardest property to test by accident. The interesting
//! failures happen between two writes.
//!
//! This wrapper makes the failure point a parameter. Give it a byte budget. It
//! passes writes through until the budget runs out. It then tears the write
//! that crosses the line, and fails every write after it.
//!
//! Sweep the budget from zero to the size of a commit. That covers every crash
//! point in that commit.
//!
//! Available under `cfg(test)` and behind the `testing` feature.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;

use crate::config::IoBackend;
use crate::io::buf::SharedBuf;
use crate::io::FileIo;
use crate::layout::Extent;
use crate::{Error, Result};

/// Wraps another backend and cuts writes off after a byte budget.
pub struct FaultIo {
    inner: Arc<dyn FileIo>,
    /// Bytes of writes still allowed. Reaching zero fails every later write.
    budget: AtomicI64,
    /// Write the part of a crossing write that still fits, instead of dropping
    /// it whole. This is what a real torn write looks like.
    tear: bool,
    /// Set once a write has been refused, so a test can tell a clean run from
    /// an interrupted one.
    tripped: AtomicBool,
}

impl FaultIo {
    /// Allow `budget` bytes of writes, then fail.
    pub fn with_budget(inner: Arc<dyn FileIo>, budget: u64) -> Self {
        Self {
            inner,
            budget: AtomicI64::new(budget as i64),
            tear: true,
            tripped: AtomicBool::new(false),
        }
    }

    /// Fail whole writes rather than tearing them.
    pub fn without_tearing(mut self) -> Self {
        self.tear = false;
        self
    }

    /// True once a write was cut short or refused.
    pub fn tripped(&self) -> bool {
        self.tripped.load(Ordering::Acquire)
    }

    /// How many bytes of writes are still allowed.
    pub fn remaining(&self) -> i64 {
        self.budget.load(Ordering::Acquire)
    }

    /// Take up to `len` bytes from the budget.
    ///
    /// Returns how many bytes the caller may write. `None` means the budget is
    /// exhausted and the write must fail outright.
    fn take(&self, len: usize) -> Option<usize> {
        let len = len as i64;
        let mut current = self.budget.load(Ordering::Acquire);
        loop {
            if current <= 0 {
                self.tripped.store(true, Ordering::Release);
                return None;
            }
            let granted = current.min(len);
            match self.budget.compare_exchange_weak(
                current,
                current - granted,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if granted < len {
                        self.tripped.store(true, Ordering::Release);
                        return if self.tear {
                            Some(granted as usize)
                        } else {
                            None
                        };
                    }
                    return Some(granted as usize);
                }
                Err(observed) => current = observed,
            }
        }
    }

    fn crashed() -> Error {
        Error::RawIo(std::io::Error::other(
            "simulated crash: write budget exhausted",
        ))
    }
}

#[async_trait]
impl FileIo for FaultIo {
    async fn read_at(&self, offset: u64, len: usize) -> Result<SharedBuf> {
        self.inner.read_at(offset, len).await
    }

    async fn read_immutable(&self, extent: Extent) -> Result<SharedBuf> {
        self.inner.read_immutable(extent).await
    }

    async fn read_scattered(&self, extents: &[Extent]) -> Result<Vec<SharedBuf>> {
        self.inner.read_scattered(extents).await
    }

    async fn append(&self, bufs: &[&[u8]]) -> Result<u64> {
        let total: usize = bufs.iter().map(|b| b.len()).sum();
        match self.take(total) {
            Some(granted) if granted == total => self.inner.append(bufs).await,
            Some(granted) => {
                let mut joined = Vec::with_capacity(granted);
                for b in bufs {
                    if joined.len() == granted {
                        break;
                    }
                    let take = granted - joined.len();
                    joined.extend_from_slice(&b[..take.min(b.len())]);
                }
                let offset = self.inner.append(&[&joined]).await?;
                let _ = offset;
                Err(Self::crashed())
            }
            None => Err(Self::crashed()),
        }
    }

    async fn write_at(&self, offset: u64, buf: &[u8]) -> Result<()> {
        match self.take(buf.len()) {
            Some(granted) if granted == buf.len() => self.inner.write_at(offset, buf).await,
            Some(granted) => {
                self.inner.write_at(offset, &buf[..granted]).await?;
                Err(Self::crashed())
            }
            None => Err(Self::crashed()),
        }
    }

    async fn set_len(&self, len: u64) -> Result<()> {
        // Growing the file writes no data of its own, so it is not budgeted.
        self.inner.set_len(len).await
    }

    async fn sync_data(&self) -> Result<()> {
        if self.tripped() {
            return Err(Self::crashed());
        }
        self.inner.sync_data().await
    }

    fn len(&self) -> Result<u64> {
        self.inner.len()
    }

    fn backend(&self) -> IoBackend {
        self.inner.backend()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Durability;
    use crate::io::StdFileIo;

    fn backing(dir: &tempfile::TempDir) -> Arc<dyn FileIo> {
        Arc::new(StdFileIo::open(&dir.path().join("f"), Durability::None, false).unwrap())
    }

    #[tokio::test]
    async fn writes_pass_through_while_the_budget_lasts() {
        let dir = tempfile::tempdir().unwrap();
        let io = FaultIo::with_budget(backing(&dir), 10);

        io.append(&[b"12345"]).await.unwrap();
        assert!(!io.tripped());
        assert_eq!(io.remaining(), 5);

        io.append(&[b"67890"]).await.unwrap();
        assert_eq!(io.remaining(), 0);
        assert_eq!(io.read_at(0, 10).await.unwrap().as_slice(), b"1234567890");
    }

    #[tokio::test]
    async fn a_crossing_write_tears_and_then_fails() {
        let dir = tempfile::tempdir().unwrap();
        let io = FaultIo::with_budget(backing(&dir), 3);

        assert!(io.append(&[b"abcdefgh"]).await.is_err());
        assert!(io.tripped());
        assert_eq!(io.len().unwrap(), 3, "the part that fit was written");
        assert_eq!(io.read_at(0, 3).await.unwrap().as_slice(), b"abc");
    }

    #[tokio::test]
    async fn without_tearing_a_crossing_write_lands_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let io = FaultIo::with_budget(backing(&dir), 3).without_tearing();

        assert!(io.append(&[b"abcdefgh"]).await.is_err());
        assert_eq!(io.len().unwrap(), 0);
    }

    #[tokio::test]
    async fn every_write_after_the_budget_fails() {
        let dir = tempfile::tempdir().unwrap();
        let io = FaultIo::with_budget(backing(&dir), 0);

        assert!(io.append(&[b"x"]).await.is_err());
        assert!(io.write_at(0, b"x").await.is_err());
        assert!(io.sync_data().await.is_err());
    }

    #[tokio::test]
    async fn reads_keep_working_after_a_crash() {
        let dir = tempfile::tempdir().unwrap();
        let io = FaultIo::with_budget(backing(&dir), 4);
        io.append(&[b"keep"]).await.unwrap();
        assert!(io.append(&[b"lost"]).await.is_err());

        assert_eq!(io.read_at(0, 4).await.unwrap().as_slice(), b"keep");
    }
}
