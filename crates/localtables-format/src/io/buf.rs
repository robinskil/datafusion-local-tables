//! Buffers the IO layer hands out.
//!
//! Every read returns a [`SharedBuf`]. It is either an owned allocation or a
//! view into a file mapping. Both convert into an Arrow [`Buffer`] without a
//! copy, so decode code never cares which backend produced the bytes.

use std::alloc::{alloc, dealloc, Layout};
use std::ops::{Deref, Range};
use std::ptr::NonNull;
use std::sync::Arc;

use arrow_buffer::alloc::Allocation;
use arrow_buffer::Buffer;

use crate::layout::BUFFER_ALIGN;

/// An owned, 64-byte aligned byte buffer.
///
/// Arrow prefers 64-byte alignment for its value buffers, and rkyv needs 16.
/// Allocating at 64 satisfies both, so a decoded page can back an Arrow array
/// directly instead of being copied into an Arrow-owned allocation.
pub struct IoBuf {
    ptr: NonNull<u8>,
    len: usize,
    capacity: usize,
}

// Safety: `IoBuf` owns a unique heap allocation and hands out no interior
// pointers that outlive it.
unsafe impl Send for IoBuf {}
unsafe impl Sync for IoBuf {}

impl IoBuf {
    fn layout(capacity: usize) -> Layout {
        Layout::from_size_align(capacity.max(1), BUFFER_ALIGN as usize)
            .expect("buffer capacity overflows the address space")
    }

    /// Allocate `len` zeroed bytes.
    pub fn zeroed(len: usize) -> Self {
        let mut buf = Self::uninit(len);
        buf.as_mut_slice().fill(0);
        buf
    }

    /// Allocate `len` bytes without initialising them.
    ///
    /// The caller must fill the buffer before reading it. Reads go straight
    /// into this memory, so zeroing first would double the write traffic.
    pub fn uninit(len: usize) -> Self {
        let capacity = len;
        let layout = Self::layout(capacity);
        // Safety: the layout has a non-zero size because `layout()` clamps it.
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, len, capacity }
    }

    /// Copy `bytes` into a fresh aligned buffer.
    pub fn copy_from(bytes: &[u8]) -> Self {
        let mut buf = Self::uninit(bytes.len());
        buf.as_mut_slice().copy_from_slice(bytes);
        buf
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        // Safety: `ptr` is valid for `len` initialised-or-caller-filled bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // Safety: `self` is borrowed uniquely, so no other view exists.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Shrink the visible length after a short read.
    ///
    /// Panics when `len` exceeds the allocated capacity.
    pub fn truncate(&mut self, len: usize) {
        assert!(len <= self.capacity, "truncate past the allocation");
        self.len = len;
    }
}

impl Drop for IoBuf {
    fn drop(&mut self) {
        // Safety: the pointer came from `alloc` with this exact layout.
        unsafe { dealloc(self.ptr.as_ptr(), Self::layout(self.capacity)) }
    }
}

impl Deref for IoBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::fmt::Debug for IoBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IoBuf").field("len", &self.len).finish()
    }
}

// Arrow's blanket `Allocation` impl already covers `IoBuf`: it is `Send`,
// `Sync` and unwind-safe, which is all the trait asks for. Holding the `Arc`
// keeps the allocation alive for as long as a buffer points into it.
const _: fn() = || {
    fn assert_allocation<T: Allocation>() {}
    assert_allocation::<IoBuf>();
};

/// Bytes a read produced, owned or mapped.
#[derive(Clone)]
pub enum SharedBuf {
    /// A heap allocation this crate owns.
    Owned(Arc<IoBuf>),
    /// A window into a file mapping. Reading it costs no syscall and no copy.
    Mapped {
        source: Arc<dyn Allocation>,
        ptr: NonNull<u8>,
        len: usize,
    },
}

// Safety: both variants hold `Send + Sync` owners, and the pointer in the
// mapped variant stays valid while that owner lives.
unsafe impl Send for SharedBuf {}
unsafe impl Sync for SharedBuf {}

impl SharedBuf {
    pub fn from_owned(buf: IoBuf) -> Self {
        SharedBuf::Owned(Arc::new(buf))
    }

    /// Wrap a window of an allocation that something else keeps alive.
    ///
    /// # Safety
    /// `ptr..ptr + len` must stay valid and immutable for as long as `source`
    /// lives, and `source` must be what keeps that memory mapped.
    pub unsafe fn from_mapped(source: Arc<dyn Allocation>, ptr: NonNull<u8>, len: usize) -> Self {
        SharedBuf::Mapped { source, ptr, len }
    }

    pub fn len(&self) -> usize {
        match self {
            SharedBuf::Owned(buf) => buf.len(),
            SharedBuf::Mapped { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        match self {
            SharedBuf::Owned(buf) => buf.as_slice(),
            // Safety: the mapped window is valid while `source` is held.
            SharedBuf::Mapped { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(ptr.as_ptr(), *len)
            },
        }
    }

    /// True when the bytes come straight from a mapping, with no copy behind them.
    pub fn is_zero_copy(&self) -> bool {
        matches!(self, SharedBuf::Mapped { .. })
    }

    /// Narrow to a sub-range, keeping the same owner alive.
    ///
    /// Panics when `range` runs past the end of the buffer.
    pub fn slice(&self, range: Range<usize>) -> SharedBuf {
        assert!(
            range.end <= self.len() && range.start <= range.end,
            "slice {range:?} out of bounds for a {}-byte buffer",
            self.len()
        );
        let len = range.end - range.start;
        match self {
            SharedBuf::Owned(buf) => {
                // Safety: the Arc clone keeps the allocation alive, and the
                // pointer stays inside it.
                let ptr = unsafe {
                    NonNull::new_unchecked(buf.as_slice().as_ptr().add(range.start) as *mut u8)
                };
                SharedBuf::Mapped {
                    source: buf.clone(),
                    ptr,
                    len,
                }
            }
            SharedBuf::Mapped { source, ptr, .. } => {
                // Safety: same window, narrowed; the owner is cloned along with it.
                let ptr = unsafe { NonNull::new_unchecked(ptr.as_ptr().add(range.start)) };
                SharedBuf::Mapped {
                    source: source.clone(),
                    ptr,
                    len,
                }
            }
        }
    }

    /// Hand the bytes to Arrow without copying them.
    ///
    /// The returned [`Buffer`] holds the owner, so the mapping or allocation
    /// outlives every array built on top of it.
    pub fn into_arrow_buffer(self) -> Buffer {
        match self {
            SharedBuf::Owned(buf) => {
                let ptr = NonNull::new(buf.as_slice().as_ptr() as *mut u8).unwrap();
                let len = buf.len();
                // Safety: `buf` owns the allocation and is moved into the Buffer.
                unsafe { Buffer::from_custom_allocation(ptr, len, buf) }
            }
            SharedBuf::Mapped { source, ptr, len } => {
                // Safety: `source` keeps the window mapped and is moved into the
                // Buffer, so the pointer stays valid for the Buffer's lifetime.
                unsafe { Buffer::from_custom_allocation(ptr, len, source) }
            }
        }
    }
}

impl Deref for SharedBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::fmt::Debug for SharedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedBuf")
            .field("len", &self.len())
            .field("zero_copy", &self.is_zero_copy())
            .finish()
    }
}

impl From<IoBuf> for SharedBuf {
    fn from(buf: IoBuf) -> Self {
        SharedBuf::from_owned(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocations_meet_arrow_alignment() {
        for len in [1usize, 7, 63, 64, 65, 4096, 100_000] {
            let buf = IoBuf::zeroed(len);
            assert_eq!(buf.len(), len);
            assert_eq!(
                buf.as_slice().as_ptr() as usize % BUFFER_ALIGN as usize,
                0,
                "len {len} is not {BUFFER_ALIGN}-byte aligned"
            );
        }
    }

    #[test]
    fn zeroed_buffers_start_empty() {
        assert!(IoBuf::zeroed(256).iter().all(|&b| b == 0));
    }

    #[test]
    fn copy_from_preserves_bytes() {
        let src: Vec<u8> = (0..=255).collect();
        assert_eq!(IoBuf::copy_from(&src).as_slice(), &src[..]);
    }

    #[test]
    fn truncate_shortens_the_view() {
        let mut buf = IoBuf::zeroed(100);
        buf.truncate(10);
        assert_eq!(buf.len(), 10);
    }

    #[test]
    fn owned_buffers_survive_the_trip_into_arrow() {
        let shared = SharedBuf::from_owned(IoBuf::copy_from(b"arrow bytes"));
        let buffer = shared.into_arrow_buffer();
        assert_eq!(buffer.as_slice(), b"arrow bytes");
        assert_eq!(buffer.as_ptr() as usize % BUFFER_ALIGN as usize, 0);
    }

    #[test]
    fn slicing_keeps_the_owner_alive() {
        let shared = SharedBuf::from_owned(IoBuf::copy_from(b"0123456789"));
        let mid = shared.slice(3..7);
        drop(shared);
        assert_eq!(mid.as_slice(), b"3456");
        assert_eq!(mid.slice(1..3).as_slice(), b"45");
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn slicing_past_the_end_panics() {
        SharedBuf::from_owned(IoBuf::zeroed(8)).slice(4..9);
    }
}
