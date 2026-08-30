//! io_uring reads, on Linux.
//!
//! The point of this backend is [`FileIo::read_scattered`]. A scan of one
//! segment needs the byte ranges of every projected column.
//!
//! Through `pread` that costs one syscall each. Through io_uring it costs one
//! submission for all of them, and the kernel may reorder and overlap them.
//!
//! One dedicated thread owns the ring. It does not join tokio's driver. A
//! caller hands it an operation and awaits a oneshot, which keeps the unsafe
//! part small.
//!
//! Writes use positional writes. The write path is sequential appends followed
//! by a sync, and io_uring does not make that faster.

use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use io_uring::{opcode, types, IoUring};
use tokio::sync::{mpsc, oneshot};

use crate::config::{Durability, IoBackend};
use crate::io::buf::{IoBuf, SharedBuf};
use crate::io::{FileIo, StdFileIo};
use crate::layout::Extent;
use crate::{Error, Result};

/// Entries in the ring. Reads beyond this queue behind the ones in flight.
const RING_ENTRIES: u32 = 256;

/// One read waiting to be submitted.
struct ReadOp {
    offset: u64,
    /// The destination. Its address is handed to the kernel, so the buffer must
    /// not move until the completion arrives; it lives in the slab until then.
    buf: IoBuf,
    reply: oneshot::Sender<Result<IoBuf>>,
}

/// What the reactor thread accepts.
enum Request {
    /// Read every range, replying once all of them are done.
    Scatter { ops: Vec<ReadOp> },
}

/// A read that has been submitted and not yet completed.
struct InFlight {
    buf: IoBuf,
    reply: oneshot::Sender<Result<IoBuf>>,
    /// Bytes still to read, when the kernel returned a short read.
    offset: u64,
    filled: usize,
}

/// A table file read through io_uring.
pub struct UringIo {
    /// Writes, metadata and the file handle the ring reads from.
    inner: StdFileIo,
    /// Hands work to the reactor thread. Dropping it stops the thread.
    requests: mpsc::UnboundedSender<Request>,
    /// Joined on drop, so the thread does not outlive the file it reads.
    reactor: Option<std::thread::JoinHandle<()>>,
}

impl UringIo {
    /// Open a file and start its reactor thread.
    ///
    /// Fails when the kernel has no io_uring, which is the honest answer: a
    /// caller that asked for this backend should hear that it is unavailable
    /// rather than silently get another one.
    pub fn open(path: &Path, durability: Durability, read_only: bool) -> Result<Self> {
        let inner = StdFileIo::open(path, durability, read_only)?;
        let file = inner.file().clone();

        // Build the ring on this thread, so a kernel without io_uring reports
        // the failure to the caller rather than killing the reactor silently.
        let ring = IoUring::new(RING_ENTRIES).map_err(|e| {
            Error::Unsupported(format!("io_uring is not available on this kernel: {e}"))
        })?;

        let (requests, receiver) = mpsc::unbounded_channel();
        let reactor = std::thread::Builder::new()
            .name("localtables-uring".into())
            .spawn(move || reactor_loop(ring, file, receiver))
            .map_err(Error::RawIo)?;

        Ok(Self {
            inner,
            requests,
            reactor: Some(reactor),
        })
    }

    pub fn path(&self) -> &Path {
        self.inner.path()
    }

    /// Send reads to the reactor and wait for all of them.
    async fn submit_reads(&self, extents: &[Extent]) -> Result<Vec<SharedBuf>> {
        if extents.is_empty() {
            return Ok(Vec::new());
        }

        let mut ops = Vec::with_capacity(extents.len());
        let mut replies = Vec::with_capacity(extents.len());
        for extent in extents {
            let (reply, wait) = oneshot::channel();
            ops.push(ReadOp {
                offset: extent.offset,
                buf: IoBuf::uninit(extent.len as usize),
                reply,
            });
            replies.push(wait);
        }

        self.requests
            .send(Request::Scatter { ops })
            .map_err(|_| Error::TaskStopped("io_uring reactor"))?;

        let mut out = Vec::with_capacity(replies.len());
        for wait in replies {
            let buf = wait
                .await
                .map_err(|_| Error::TaskStopped("io_uring reactor"))??;
            out.push(SharedBuf::from_owned(buf));
        }
        Ok(out)
    }
}

impl Drop for UringIo {
    fn drop(&mut self) {
        // Closing the channel tells the reactor to finish; joining it makes
        // sure no read is still touching the file when it closes.
        let (dead, _) = mpsc::unbounded_channel();
        let _ = std::mem::replace(&mut self.requests, dead);
        if let Some(handle) = self.reactor.take() {
            let _ = handle.join();
        }
    }
}

#[async_trait]
impl FileIo for UringIo {
    async fn read_at(&self, offset: u64, len: usize) -> Result<SharedBuf> {
        let mut bufs = self
            .submit_reads(&[Extent::new(offset, len as u64)])
            .await?;
        Ok(bufs.pop().expect("one extent gives one buffer"))
    }

    async fn read_immutable(&self, extent: Extent) -> Result<SharedBuf> {
        self.read_at(extent.offset, extent.len as usize).await
    }

    async fn read_scattered(&self, extents: &[Extent]) -> Result<Vec<SharedBuf>> {
        // The win: every range of a segment's projection goes to the kernel in
        // one submission instead of one syscall each.
        self.submit_reads(extents).await
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
        IoBackend::Uring
    }
}

/// The reactor thread: submit what arrives, reap what completes.
fn reactor_loop(
    mut ring: IoUring,
    file: Arc<std::fs::File>,
    mut requests: mpsc::UnboundedReceiver<Request>,
) {
    let fd = types::Fd(file.as_raw_fd());
    // Operations the kernel is working on, keyed by the user_data of their SQE.
    let mut in_flight: std::collections::HashMap<u64, InFlight> = std::collections::HashMap::new();
    let mut next_id: u64 = 0;

    loop {
        // Take work, blocking only when the kernel has nothing outstanding.
        let request = if in_flight.is_empty() {
            match requests.blocking_recv() {
                Some(request) => Some(request),
                // Every sender is gone and nothing is in flight: done.
                None => return,
            }
        } else {
            requests.try_recv().ok()
        };

        if let Some(Request::Scatter { ops }) = request {
            for op in ops {
                let id = next_id;
                next_id += 1;
                submit(&mut ring, &mut in_flight, fd, id, op);
            }
        }

        if in_flight.is_empty() {
            continue;
        }

        // Wait for at least one completion, so this loop never spins.
        if let Err(e) = ring.submit_and_wait(1) {
            fail_all(&mut in_flight, &e);
            continue;
        }

        let mut resubmit = Vec::new();
        for cqe in ring.completion() {
            let Some(mut op) = in_flight.remove(&cqe.user_data()) else {
                continue;
            };
            let result = cqe.result();

            if result < 0 {
                let _ = op
                    .reply
                    .send(Err(Error::RawIo(std::io::Error::from_raw_os_error(
                        -result,
                    ))));
                continue;
            }

            let read = result as usize;
            op.filled += read;
            if read == 0 && op.filled < op.buf.len() {
                // End of file before the range was filled: the caller asked for
                // bytes that are not there.
                let _ = op.reply.send(Err(Error::corrupt(format!(
                    "read past the end of the file: {} of {} bytes",
                    op.filled,
                    op.buf.len()
                ))));
                continue;
            }

            if op.filled < op.buf.len() {
                // A short read is legal; carry on from where it stopped.
                resubmit.push(op);
            } else {
                let _ = op.reply.send(Ok(op.buf));
            }
        }

        for op in resubmit {
            let id = next_id;
            next_id += 1;
            let offset = op.offset + op.filled as u64;
            let filled = op.filled;
            submit_partial(&mut ring, &mut in_flight, fd, id, op, offset, filled);
        }
    }
}

/// Queue one whole read.
fn submit(
    ring: &mut IoUring,
    in_flight: &mut std::collections::HashMap<u64, InFlight>,
    fd: types::Fd,
    id: u64,
    op: ReadOp,
) {
    let ReadOp { offset, buf, reply } = op;
    let entry = InFlight {
        buf,
        reply,
        offset,
        filled: 0,
    };
    submit_entry(ring, in_flight, fd, id, entry);
}

/// Queue the rest of a read the kernel returned short.
fn submit_partial(
    ring: &mut IoUring,
    in_flight: &mut std::collections::HashMap<u64, InFlight>,
    fd: types::Fd,
    id: u64,
    mut op: InFlight,
    offset: u64,
    filled: usize,
) {
    op.offset = offset - filled as u64;
    submit_entry(ring, in_flight, fd, id, op);
}

fn submit_entry(
    ring: &mut IoUring,
    in_flight: &mut std::collections::HashMap<u64, InFlight>,
    fd: types::Fd,
    id: u64,
    mut op: InFlight,
) {
    let len = op.buf.len() - op.filled;
    if len == 0 {
        let _ = op.reply.send(Ok(op.buf));
        return;
    }

    // Safety: the buffer lives in `in_flight` until its completion arrives, so
    // the pointer the kernel writes through stays valid and unaliased.
    let ptr = unsafe { op.buf.as_mut_slice().as_mut_ptr().add(op.filled) };
    let entry = opcode::Read::new(fd, ptr, len as u32)
        .offset(op.offset + op.filled as u64)
        .build()
        .user_data(id);

    in_flight.insert(id, op);

    // Safety: the entry names a buffer owned by `in_flight`, which outlives the
    // operation. A full queue is retried after the next submit.
    while unsafe { ring.submission().push(&entry) }.is_err() {
        if let Err(e) = ring.submit() {
            if let Some(op) = in_flight.remove(&id) {
                let _ = op.reply.send(Err(Error::RawIo(clone_io_error(&e))));
            }
            return;
        }
    }
}

/// Report a ring-level failure to everything waiting on it.
fn fail_all(in_flight: &mut std::collections::HashMap<u64, InFlight>, error: &std::io::Error) {
    for (_, op) in in_flight.drain() {
        let _ = op.reply.send(Err(Error::RawIo(clone_io_error(error))));
    }
}

/// `io::Error` is not `Clone`, and one failure has to reach several waiters.
fn clone_io_error(error: &std::io::Error) -> std::io::Error {
    match error.raw_os_error() {
        Some(code) => std::io::Error::from_raw_os_error(code),
        None => std::io::Error::other(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Skip on a kernel without io_uring rather than failing: containers and
    /// older kernels are legitimate places to run the rest of the suite.
    fn io(dir: &tempfile::TempDir) -> Option<UringIo> {
        match UringIo::open(&dir.path().join("f.lt"), Durability::None, false) {
            Ok(io) => Some(io),
            Err(Error::Unsupported(_)) => None,
            Err(e) => panic!("unexpected failure: {e}"),
        }
    }

    #[tokio::test]
    async fn a_read_returns_the_bytes_that_were_written() {
        let dir = tempfile::tempdir().unwrap();
        let Some(io) = io(&dir) else { return };

        io.append(&[b"hello world"]).await.unwrap();
        assert_eq!(io.read_at(0, 11).await.unwrap().as_slice(), b"hello world");
        assert_eq!(io.read_at(6, 5).await.unwrap().as_slice(), b"world");
    }

    #[tokio::test]
    async fn a_scatter_read_returns_every_range_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let Some(io) = io(&dir) else { return };

        io.append(&[b"0123456789"]).await.unwrap();
        let bufs = io
            .read_scattered(&[Extent::new(0, 3), Extent::new(7, 3), Extent::new(4, 1)])
            .await
            .unwrap();

        let got: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
        assert_eq!(got, vec![&b"012"[..], &b"789"[..], &b"4"[..]]);
    }

    #[tokio::test]
    async fn a_large_scatter_read_works_past_the_ring_size() {
        let dir = tempfile::tempdir().unwrap();
        let Some(io) = io(&dir) else { return };

        let bytes: Vec<u8> = (0..8192u32).map(|i| i as u8).collect();
        io.append(&[&bytes]).await.unwrap();

        // More ranges than the ring holds, so the reactor has to submit in
        // waves rather than all at once.
        let extents: Vec<Extent> = (0..(RING_ENTRIES as u64 * 2))
            .map(|i| Extent::new(i * 4, 4))
            .collect();
        let bufs = io.read_scattered(&extents).await.unwrap();

        assert_eq!(bufs.len(), extents.len());
        for (index, buf) in bufs.iter().enumerate() {
            let start = index * 4;
            assert_eq!(buf.as_slice(), &bytes[start..start + 4], "range {index}");
        }
    }

    #[tokio::test]
    async fn reading_past_the_end_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let Some(io) = io(&dir) else { return };

        io.append(&[b"short"]).await.unwrap();
        assert!(io.read_at(0, 100).await.is_err());
    }

    #[tokio::test]
    async fn an_empty_read_returns_an_empty_buffer() {
        let dir = tempfile::tempdir().unwrap();
        let Some(io) = io(&dir) else { return };

        io.append(&[b"data"]).await.unwrap();
        assert!(io.read_at(0, 0).await.unwrap().is_empty());
        assert!(io.read_scattered(&[]).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn reads_and_writes_interleave() {
        let dir = tempfile::tempdir().unwrap();
        let Some(io) = io(&dir) else { return };

        io.append(&[b"aaaa"]).await.unwrap();
        assert_eq!(io.read_at(0, 4).await.unwrap().as_slice(), b"aaaa");
        io.write_at(1, b"bb").await.unwrap();
        assert_eq!(io.read_at(0, 4).await.unwrap().as_slice(), b"abba");
        io.sync_data().await.unwrap();
    }
}
