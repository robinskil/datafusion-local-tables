//! Memory-mapped reads.
//!
//! Sealed regions are mapped once and read with no syscall and no copy. The
//! mapping becomes the owner of every Arrow buffer carved out of it, so a
//! `RecordBatch` handed to a query points straight at the page cache.
//!
//! Writes go through positional writes, not the mapping. Only committed,
//! never-rewritten regions are mapped, so a mapping's contents never change
//! under a reader.

use std::collections::HashMap;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::{Arc, Weak};

use arrow_buffer::alloc::Allocation;
use async_trait::async_trait;
use memmap2::{Mmap, MmapOptions};
use parking_lot::Mutex;

use crate::config::{Durability, IoBackend};
use crate::io::buf::SharedBuf;
use crate::io::{blocking, FileIo, StdFileIo};
use crate::layout::Extent;
use crate::{Error, Result};

/// One mapped window of a table file.
///
/// Arrow buffers hold an `Arc` of this, so the window stays mapped for as long
/// as any array built from it is alive. That is what makes extent reuse
/// dangerous, and why freed extents wait for old snapshots to drop.
pub struct FileMapping {
    map: Mmap,
    extent: Extent,
}

// Safety: `Mmap` derefs to an immutable byte slice and the mapped region is
// sealed before it is mapped, so sharing it across threads is sound.
unsafe impl Send for FileMapping {}
unsafe impl Sync for FileMapping {}

// Arrow's blanket `Allocation` impl covers `FileMapping`. Holding the `Arc`
// keeps the pages mapped for as long as a buffer points into them.
const _: fn() = || {
    fn assert_allocation<T: Allocation>() {}
    assert_allocation::<FileMapping>();
};

impl FileMapping {
    pub fn as_slice(&self) -> &[u8] {
        &self.map
    }

    pub fn extent(&self) -> Extent {
        self.extent
    }

    /// Advise the kernel that a scan will read this region front to back.
    pub fn advise_sequential(&self) {
        let _ = self.map.advise(memmap2::Advice::Sequential);
    }

    /// Advise the kernel that reads will jump around, as point lookups do.
    pub fn advise_random(&self) {
        let _ = self.map.advise(memmap2::Advice::Random);
    }
}

impl std::fmt::Debug for FileMapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileMapping")
            .field("extent", &self.extent)
            .finish()
    }
}

/// A table file read through mappings and written through positional writes.
pub struct MmapIo {
    inner: StdFileIo,
    /// Mappings still in use, keyed by the extent they cover.
    ///
    /// Weak references, so a mapping is unmapped as soon as the last snapshot
    /// and the last Arrow buffer over it are gone.
    cache: Mutex<HashMap<(u64, u64), Weak<FileMapping>>>,
}

impl MmapIo {
    pub fn open(path: &Path, durability: Durability, read_only: bool) -> Result<Self> {
        Ok(Self {
            inner: StdFileIo::open(path, durability, read_only)?,
            cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn file(&self) -> &Arc<std::fs::File> {
        self.inner.file()
    }

    pub fn path(&self) -> &Path {
        self.inner.path()
    }

    /// Map a sealed extent, reusing an existing mapping when one covers it.
    ///
    /// A segment is mapped as one unit, so a scan of many column chunks in one
    /// segment costs a single `mmap`.
    pub fn map_extent(&self, extent: Extent) -> Result<Arc<FileMapping>> {
        let key = (extent.offset, extent.len);
        {
            let cache = self.cache.lock();
            if let Some(existing) = cache.get(&key).and_then(Weak::upgrade) {
                return Ok(existing);
            }
        }

        let file_len = self.inner.len()?;
        if extent.end() > file_len {
            return Err(Error::corrupt(format!(
                "extent {extent:?} runs past the {file_len}-byte file",
                extent = extent
            )));
        }

        // Safety: the extent is committed and immutable. Another process could
        // still truncate the file, which is why the writer lock is exclusive.
        let map = unsafe {
            MmapOptions::new()
                .offset(extent.offset)
                .len(extent.len as usize)
                .map(&**self.inner.file())
        }
        .map_err(|e| Error::io(self.inner.path().to_path_buf(), e))?;

        let mapping = Arc::new(FileMapping { map, extent });

        let mut cache = self.cache.lock();
        // Another thread may have mapped the same extent while this one worked.
        if let Some(existing) = cache.get(&key).and_then(Weak::upgrade) {
            return Ok(existing);
        }
        cache.retain(|_, weak| weak.strong_count() > 0);
        cache.insert(key, Arc::downgrade(&mapping));
        Ok(mapping)
    }

    /// Number of mappings still alive. Tests use it to prove unmapping happens.
    pub fn live_mappings(&self) -> usize {
        let mut cache = self.cache.lock();
        cache.retain(|_, weak| weak.strong_count() > 0);
        cache.len()
    }
}

/// Wrap a whole mapping as a zero-copy buffer.
fn mapping_to_buf(mapping: Arc<FileMapping>) -> SharedBuf {
    let slice = mapping.as_slice();
    let len = slice.len();
    let ptr = NonNull::new(slice.as_ptr() as *mut u8).expect("a mapping is never null");
    // Safety: the mapping owns the pages and is moved into the buffer, so the
    // pointer stays valid and immutable for as long as the buffer lives.
    unsafe { SharedBuf::from_mapped(mapping, ptr, len) }
}

#[async_trait]
impl FileIo for MmapIo {
    async fn read_at(&self, offset: u64, len: usize) -> Result<SharedBuf> {
        // Mutable regions are read, never mapped: a mapping of a page that is
        // rewritten in place has no defined contents.
        self.inner.read_at(offset, len).await
    }

    async fn read_immutable(&self, extent: Extent) -> Result<SharedBuf> {
        if extent.is_empty() {
            return self.inner.read_at(extent.offset, 0).await;
        }
        let mapping = self.map_extent(extent)?;
        Ok(mapping_to_buf(mapping))
    }

    async fn read_scattered(&self, extents: &[Extent]) -> Result<Vec<SharedBuf>> {
        let extents = extents.to_vec();
        let mut out = Vec::with_capacity(extents.len());
        for extent in extents {
            out.push(self.read_immutable(extent).await?);
        }
        Ok(out)
    }

    async fn append(&self, bufs: &[&[u8]]) -> Result<u64> {
        self.inner.append(bufs).await
    }

    async fn write_at(&self, offset: u64, buf: &[u8]) -> Result<()> {
        self.inner.write_at(offset, buf).await
    }

    async fn set_len(&self, len: u64) -> Result<()> {
        self.inner.set_len(len).await
    }

    async fn sync_data(&self) -> Result<()> {
        self.inner.sync_data().await
    }

    fn len(&self) -> Result<u64> {
        self.inner.len()
    }

    fn backend(&self) -> IoBackend {
        IoBackend::Mmap
    }
}

/// Map a sealed extent through any backend, preferring zero copy.
///
/// Callers that want a mapping but hold a `dyn FileIo` use this instead of
/// downcasting.
pub async fn read_sealed(io: &dyn FileIo, extent: Extent) -> Result<SharedBuf> {
    io.read_immutable(extent).await
}

/// Placate the unused-import check when the crate is built without a runtime.
#[allow(dead_code)]
async fn _unused(io: &StdFileIo) -> Result<SharedBuf> {
    blocking(move || Ok(())).await?;
    io.read_at(0, 0).await
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn seeded(dir: &tempfile::TempDir, bytes: &[u8]) -> MmapIo {
        let io = MmapIo::open(&dir.path().join("table.lt"), Durability::None, false).unwrap();
        io.append(&[bytes]).await.unwrap();
        io
    }

    #[tokio::test]
    async fn sealed_reads_are_zero_copy() {
        let dir = tempfile::tempdir().unwrap();
        let io = seeded(&dir, &vec![7u8; 8192]).await;

        let buf = io.read_immutable(Extent::new(4096, 4096)).await.unwrap();
        assert!(
            buf.is_zero_copy(),
            "mmap backend must not copy sealed reads"
        );
        assert_eq!(buf.len(), 4096);
        assert!(buf.iter().all(|&b| b == 7));
    }

    #[tokio::test]
    async fn mutable_reads_see_later_writes() {
        let dir = tempfile::tempdir().unwrap();
        let io = seeded(&dir, &[0u8; 4096]).await;

        io.write_at(0, b"first").await.unwrap();
        assert_eq!(&io.read_at(0, 5).await.unwrap()[..], b"first");
        io.write_at(0, b"secnd").await.unwrap();
        assert_eq!(&io.read_at(0, 5).await.unwrap()[..], b"secnd");
    }

    #[tokio::test]
    async fn the_same_extent_maps_once() {
        let dir = tempfile::tempdir().unwrap();
        let io = seeded(&dir, &vec![1u8; 16384]).await;

        let a = io.map_extent(Extent::new(0, 8192)).unwrap();
        let b = io.map_extent(Extent::new(0, 8192)).unwrap();
        assert!(Arc::ptr_eq(&a, &b), "the cache must hand back one mapping");
        assert_eq!(io.live_mappings(), 1);

        let _c = io.map_extent(Extent::new(8192, 8192)).unwrap();
        assert_eq!(io.live_mappings(), 2);
    }

    #[tokio::test]
    async fn mappings_drop_once_nothing_holds_them() {
        let dir = tempfile::tempdir().unwrap();
        let io = seeded(&dir, &vec![2u8; 8192]).await;

        let mapping = io.map_extent(Extent::new(0, 4096)).unwrap();
        assert_eq!(io.live_mappings(), 1);
        drop(mapping);
        assert_eq!(io.live_mappings(), 0);
    }

    #[tokio::test]
    async fn an_arrow_buffer_outlives_the_mapping_handle() {
        let dir = tempfile::tempdir().unwrap();
        let io = seeded(&dir, &vec![9u8; 4096]).await;

        let buffer = io
            .read_immutable(Extent::new(0, 4096))
            .await
            .unwrap()
            .into_arrow_buffer();
        assert_eq!(io.live_mappings(), 1, "the buffer still holds the mapping");
        assert!(buffer.as_slice().iter().all(|&b| b == 9));

        drop(buffer);
        assert_eq!(io.live_mappings(), 0);
    }

    #[tokio::test]
    async fn mapping_past_the_end_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let io = seeded(&dir, &[0u8; 4096]).await;
        let err = io.read_immutable(Extent::new(0, 8192)).await.unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn unaligned_extents_still_map() {
        let dir = tempfile::tempdir().unwrap();
        let bytes: Vec<u8> = (0..16384u32).map(|i| i as u8).collect();
        let io = seeded(&dir, &bytes).await;

        let buf = io.read_immutable(Extent::new(4097, 100)).await.unwrap();
        assert!(buf.is_zero_copy());
        assert_eq!(buf.as_slice(), &bytes[4097..4197]);
    }
}
