//! File IO backends.
//!
//! Every backend implements [`FileIo`]. A read returns a [`SharedBuf`]. That is
//! either an owned allocation or a zero-copy window into a mapping. The decode
//! path stays the same for every backend.
//!
//! The trait splits reads in two, because the two have different contracts:
//!
//! * [`FileIo::read_at`] reads a region that may still change (meta pages, the
//!   WAL tail). It always returns the current bytes.
//! * [`FileIo::read_immutable`] reads a region a commit sealed. The caller
//!   promises those bytes never change again. A backend can then return a
//!   mapping instead of a copy.

pub mod buf;
pub mod lock;
pub mod std_file;

#[cfg(any(test, feature = "testing"))]
pub mod fault;

#[cfg(feature = "mmap")]
pub mod mmap;

#[cfg(all(target_os = "linux", feature = "uring"))]
pub mod uring;

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;

pub use buf::{IoBuf, SharedBuf};
pub use lock::FileLock;
pub use std_file::StdFileIo;

#[cfg(feature = "mmap")]
pub use mmap::{FileMapping, MmapIo};

use crate::config::{Durability, IoBackend};
use crate::layout::Extent;
use crate::{Error, Result};

/// Read and write access to one table file.
// `len` here is the file's size, not a container's element count, so the
// clippy pairing with `is_empty` does not apply.
#[allow(clippy::len_without_is_empty)]
#[async_trait]
pub trait FileIo: Send + Sync + 'static {
    /// Read `len` bytes at `offset` from a region that may still be rewritten.
    ///
    /// Always reflects the latest write. Never returns a mapping, because a
    /// mapping of a rewritten page has no defined contents.
    async fn read_at(&self, offset: u64, len: usize) -> Result<SharedBuf>;

    /// Read a sealed region.
    ///
    /// The caller guarantees these bytes are committed and will not change
    /// while the returned buffer lives. A backend may satisfy this from a
    /// mapping with no syscall and no copy.
    async fn read_immutable(&self, extent: Extent) -> Result<SharedBuf>;

    /// Read many sealed regions, ideally in one submission.
    ///
    /// This is the scan path: one call per segment carrying every projected
    /// column chunk. The default runs the reads one at a time.
    async fn read_scattered(&self, extents: &[Extent]) -> Result<Vec<SharedBuf>> {
        let mut out = Vec::with_capacity(extents.len());
        for extent in extents {
            out.push(self.read_immutable(*extent).await?);
        }
        Ok(out)
    }

    /// Append the buffers back to back at the end of the file.
    ///
    /// Returns the offset the first byte landed at.
    async fn append(&self, bufs: &[&[u8]]) -> Result<u64>;

    /// Overwrite bytes in place. Used for the two meta page slots.
    async fn write_at(&self, offset: u64, buf: &[u8]) -> Result<()>;

    /// Grow the file to `len` bytes, filling with zeros.
    async fn set_len(&self, len: u64) -> Result<()>;

    /// Push writes toward the media, honouring the configured durability.
    async fn sync_data(&self) -> Result<()>;

    /// Current file length in bytes.
    fn len(&self) -> Result<u64>;

    /// Name of this backend, for errors and diagnostics.
    fn backend(&self) -> IoBackend;
}

/// Open a table file with the requested backend.
///
/// Falls back with a clear error when a backend is compiled out or unavailable
/// on this platform, rather than silently substituting another one.
pub fn open_backend(
    path: &Path,
    backend: IoBackend,
    durability: Durability,
    read_only: bool,
) -> Result<Arc<dyn FileIo>> {
    match backend {
        IoBackend::Pread => Ok(Arc::new(StdFileIo::open(path, durability, read_only)?)),

        #[cfg(feature = "mmap")]
        IoBackend::Mmap => Ok(Arc::new(MmapIo::open(path, durability, read_only)?)),
        #[cfg(not(feature = "mmap"))]
        IoBackend::Mmap => Err(Error::Unsupported(
            "the mmap backend needs the `mmap` feature".into(),
        )),

        #[cfg(all(target_os = "linux", feature = "uring"))]
        IoBackend::Uring => Ok(Arc::new(uring::UringIo::open(path, durability, read_only)?)),
        // Asking for a backend this build cannot provide is an error, not a
        // silent downgrade to another one.
        #[cfg(not(all(target_os = "linux", feature = "uring")))]
        IoBackend::Uring => Err(Error::Unsupported(
            "the io_uring backend needs Linux and the `uring` feature".into(),
        )),
    }
}

/// Apply the durability setting to an open file.
pub(crate) fn sync_file(file: &std::fs::File, durability: Durability) -> std::io::Result<()> {
    match durability {
        Durability::None => Ok(()),
        Durability::Os => file.sync_data(),
        Durability::Full => full_sync(file),
    }
}

/// Flush the drive's own write cache, not only the page cache.
#[cfg(target_os = "macos")]
fn full_sync(file: &std::fs::File) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    // F_FULLFSYNC. `fsync` on macOS returns once the data reaches the drive,
    // which may still be a volatile cache; this waits for the platter.
    const F_FULLFSYNC: i32 = 51;
    // Safety: `fcntl` with F_FULLFSYNC takes no extra argument and only reads
    // the descriptor, which stays open for the call.
    let rc = unsafe { libc_fcntl(file.as_raw_fd(), F_FULLFSYNC) };
    if rc == -1 {
        // Some filesystems reject F_FULLFSYNC. A plain sync is the best left.
        return file.sync_data();
    }
    Ok(())
}

#[cfg(target_os = "macos")]
extern "C" {
    #[link_name = "fcntl"]
    fn libc_fcntl(fd: std::os::unix::io::RawFd, cmd: i32, ...) -> i32;
}

#[cfg(not(target_os = "macos"))]
fn full_sync(file: &std::fs::File) -> std::io::Result<()> {
    file.sync_all()
}

/// Run a blocking file operation off the async runtime.
///
/// Falls back to running inline when no runtime is present, so the format layer
/// stays usable from plain synchronous code and tests.
pub(crate) async fn blocking<F, T>(f: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    match tokio::runtime::Handle::try_current() {
        Ok(_) => tokio::task::spawn_blocking(f)
            .await
            .map_err(|e| Error::Corrupt(format!("blocking io task failed: {e}")))?,
        Err(_) => f(),
    }
}
