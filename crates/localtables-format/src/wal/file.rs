//! One write-ahead log file: appending records, and reading them back.
//!
//! A log file is a 64-byte header followed by framed records. Recovery reads
//! forward and checks each frame. It stops at the first frame that fails.
//!
//! That frame is where the crash happened. Recovery discards it and everything
//! after it, then truncates the file back to the last good record.
//!
//! A table keeps two of these files and swaps between them. A flush can then
//! truncate the log it made durable while new writes go to the other one.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use rkyv::rancor;

use crate::config::Durability;
use crate::layout::frame::{self, tag, FRAME_HEADER_LEN};
use crate::wal::record::{ArchivedWalRecord, Lsn, WalRecord};
use crate::{Error, Result};

/// Bytes at the start of a log file, before the first record.
pub const WAL_HEADER_LEN: u64 = 64;

/// Identifies a log file and the table it belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalHeader {
    /// Which of the table's two logs this is. The higher generation holds the
    /// newer records, which is how recovery orders them.
    pub generation: u64,
    pub table_uuid: [u8; 16],
}

impl WalHeader {
    fn encode(&self) -> [u8; WAL_HEADER_LEN as usize] {
        let mut out = [0u8; WAL_HEADER_LEN as usize];
        out[0..8].copy_from_slice(&tag::WAL_FILE.to_le_bytes());
        out[8..12].copy_from_slice(&crate::layout::FORMAT_VERSION.to_le_bytes());
        out[16..24].copy_from_slice(&self.generation.to_le_bytes());
        out[24..40].copy_from_slice(&self.table_uuid);
        let checksum = crate::layout::checksum(&out[0..40]);
        out[40..48].copy_from_slice(&checksum.to_le_bytes());
        out
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < WAL_HEADER_LEN as usize {
            return Err(Error::corrupt("write-ahead log header is truncated"));
        }
        let stored = u64::from_le_bytes(bytes[40..48].try_into().unwrap());
        crate::layout::verify_checksum("wal header", &bytes[0..40], stored)?;

        let magic = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        if magic != tag::WAL_FILE {
            return Err(Error::BadMagic("this file is not a write-ahead log".into()));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != crate::layout::FORMAT_VERSION {
            return Err(Error::Unsupported(format!(
                "write-ahead log format version {version}, this build reads {}",
                crate::layout::FORMAT_VERSION
            )));
        }
        Ok(Self {
            generation: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            table_uuid: bytes[24..40].try_into().unwrap(),
        })
    }
}

/// An open log file, positioned at the end.
///
/// Writes go through plain buffered file IO rather than the table's backend.
/// The log is appended sequentially, synced, and thrown away; none of the
/// mapping or scatter-read machinery helps it.
#[derive(Debug)]
pub struct WalFile {
    file: File,
    path: PathBuf,
    header: WalHeader,
    /// Bytes written after the header.
    len: u64,
    durability: Durability,
}

impl WalFile {
    /// Create a log file, replacing any file already at that path.
    pub fn create(path: &Path, header: WalHeader, durability: Durability) -> Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| Error::io(path.to_path_buf(), e))?;
        file.write_all(&header.encode())
            .map_err(|e| Error::io(path.to_path_buf(), e))?;
        crate::io::sync_file(&file, durability).map_err(|e| Error::io(path.to_path_buf(), e))?;

        Ok(Self {
            file,
            path: path.to_path_buf(),
            header,
            len: 0,
            durability,
        })
    }

    /// Open an existing log file for appending.
    pub fn open(path: &Path, durability: Durability) -> Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| Error::io(path.to_path_buf(), e))?;

        let mut header_bytes = [0u8; WAL_HEADER_LEN as usize];
        file.read_exact(&mut header_bytes)
            .map_err(|e| Error::io(path.to_path_buf(), e))?;
        let header = WalHeader::decode(&header_bytes)?;

        let total = file
            .metadata()
            .map_err(|e| Error::io(path.to_path_buf(), e))?
            .len();
        Ok(Self {
            file,
            path: path.to_path_buf(),
            header,
            len: total.saturating_sub(WAL_HEADER_LEN),
            durability,
        })
    }

    pub fn header(&self) -> WalHeader {
        self.header
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Bytes of records in the file.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append framed records, then sync once for all of them.
    ///
    /// Grouping the sync is the point: syncing per record would make every
    /// small insert wait for the disk on its own.
    pub fn append_group(&mut self, frames: &[Vec<u8>]) -> Result<()> {
        if frames.is_empty() {
            return Ok(());
        }
        let total: usize = frames.iter().map(|f| f.len()).sum();
        let mut joined = Vec::with_capacity(total);
        for frame in frames {
            joined.extend_from_slice(frame);
        }

        self.file
            .seek(SeekFrom::Start(WAL_HEADER_LEN + self.len))
            .map_err(|e| Error::io(self.path.clone(), e))?;
        self.file
            .write_all(&joined)
            .map_err(|e| Error::io(self.path.clone(), e))?;
        crate::io::sync_file(&self.file, self.durability)
            .map_err(|e| Error::io(self.path.clone(), e))?;

        self.len += total as u64;
        Ok(())
    }

    /// Drop every record, keeping the file and its header.
    ///
    /// Called after a flush has made these records durable inside a segment.
    pub fn truncate(&mut self) -> Result<()> {
        self.file
            .set_len(WAL_HEADER_LEN)
            .map_err(|e| Error::io(self.path.clone(), e))?;
        crate::io::sync_file(&self.file, self.durability)
            .map_err(|e| Error::io(self.path.clone(), e))?;
        self.len = 0;
        Ok(())
    }

    /// Start a new generation in this file, discarding what it held.
    pub fn rotate(&mut self, generation: u64) -> Result<()> {
        self.header.generation = generation;
        self.file
            .set_len(0)
            .map_err(|e| Error::io(self.path.clone(), e))?;
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|e| Error::io(self.path.clone(), e))?;
        self.file
            .write_all(&self.header.encode())
            .map_err(|e| Error::io(self.path.clone(), e))?;
        crate::io::sync_file(&self.file, self.durability)
            .map_err(|e| Error::io(self.path.clone(), e))?;
        self.len = 0;
        Ok(())
    }

    /// Read every whole record, and truncate the file at the first damaged one.
    ///
    /// A torn tail is the normal way a log ends after a crash, so it is not an
    /// error: the records before it were acknowledged and the ones after it
    /// never were.
    pub fn recover(&mut self) -> Result<RecoveredLog> {
        let mut bytes = Vec::with_capacity(self.len as usize);
        self.file
            .seek(SeekFrom::Start(WAL_HEADER_LEN))
            .map_err(|e| Error::io(self.path.clone(), e))?;
        self.file
            .read_to_end(&mut bytes)
            .map_err(|e| Error::io(self.path.clone(), e))?;

        let mut records = Vec::new();
        let mut offset = 0usize;
        let mut truncated_at = None;

        while offset < bytes.len() {
            let rest = &bytes[offset..];
            let Ok(payload_len) = frame::peek_len(rest, tag::WAL_REC, "wal record") else {
                truncated_at = Some(offset);
                break;
            };
            let frame_len = FRAME_HEADER_LEN + payload_len;
            if offset + frame_len > bytes.len() {
                // The record was still being written when the crash happened.
                truncated_at = Some(offset);
                break;
            }
            let Ok(payload) = frame::decode(&rest[..frame_len], tag::WAL_REC, "wal record") else {
                truncated_at = Some(offset);
                break;
            };
            let Ok(archived) = rkyv::access::<ArchivedWalRecord, rancor::Error>(payload) else {
                truncated_at = Some(offset);
                break;
            };
            let Ok(record) = rkyv::deserialize::<WalRecord, rancor::Error>(archived) else {
                truncated_at = Some(offset);
                break;
            };
            records.push(record);
            offset += frame_len;
        }

        if let Some(good_bytes) = truncated_at {
            // Cut the partial record off, so the next append starts from a
            // record boundary rather than inside one.
            self.file
                .set_len(WAL_HEADER_LEN + good_bytes as u64)
                .map_err(|e| Error::io(self.path.clone(), e))?;
            crate::io::sync_file(&self.file, self.durability)
                .map_err(|e| Error::io(self.path.clone(), e))?;
            self.len = good_bytes as u64;
        }

        Ok(RecoveredLog {
            generation: self.header.generation,
            records,
            truncated: truncated_at.is_some(),
        })
    }
}

/// What one log file held after a crash.
#[derive(Debug)]
pub struct RecoveredLog {
    pub generation: u64,
    pub records: Vec<WalRecord>,
    /// A partial record was found and discarded.
    pub truncated: bool,
}

impl RecoveredLog {
    pub fn highest_lsn(&self) -> Option<Lsn> {
        self.records.last().map(|r| r.lsn())
    }
}

/// Wrap a record in a frame ready to append.
pub fn encode_record(record: &WalRecord) -> Result<Vec<u8>> {
    let payload = rkyv::to_bytes::<rancor::Error>(record)?;
    Ok(frame::encode(tag::WAL_REC, &payload))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::batchcodec;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn header() -> WalHeader {
        WalHeader {
            generation: 1,
            table_uuid: [7u8; 16],
        }
    }

    fn insert(lsn: Lsn, values: Vec<i32>) -> WalRecord {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values))]).unwrap();
        WalRecord::Insert {
            lsn,
            base_seqno: 0,
            batch: batchcodec::encode(&batch),
        }
    }

    fn wal(dir: &tempfile::TempDir) -> WalFile {
        WalFile::create(&dir.path().join("t.lt.wal0"), header(), Durability::None).unwrap()
    }

    #[test]
    fn header_round_trips() {
        let encoded = header().encode();
        assert_eq!(WalHeader::decode(&encoded).unwrap(), header());
    }

    #[test]
    fn a_damaged_header_is_rejected() {
        let mut encoded = header().encode();
        encoded[20] ^= 0xff;
        assert!(WalHeader::decode(&encoded).is_err());
    }

    #[test]
    fn records_come_back_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = wal(&dir);

        let written: Vec<WalRecord> = (1..=5)
            .map(|lsn| insert(lsn, vec![lsn as i32, lsn as i32 + 1]))
            .collect();
        let frames: Vec<Vec<u8>> = written.iter().map(|r| encode_record(r).unwrap()).collect();
        wal.append_group(&frames).unwrap();

        let recovered = wal.recover().unwrap();
        assert!(!recovered.truncated);
        assert_eq!(recovered.records, written);
        assert_eq!(recovered.highest_lsn(), Some(5));
    }

    #[test]
    fn an_empty_log_recovers_to_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = wal(&dir);
        let recovered = wal.recover().unwrap();
        assert!(recovered.records.is_empty());
        assert!(!recovered.truncated);
        assert!(wal.is_empty());
    }

    #[test]
    fn records_survive_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.lt.wal0");
        {
            let mut wal = WalFile::create(&path, header(), Durability::None).unwrap();
            wal.append_group(&[encode_record(&insert(1, vec![1])).unwrap()])
                .unwrap();
        }

        let mut wal = WalFile::open(&path, Durability::None).unwrap();
        assert_eq!(wal.header(), header());
        assert_eq!(wal.recover().unwrap().records.len(), 1);
    }

    #[test]
    fn appending_after_a_reopen_keeps_the_earlier_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.lt.wal0");
        {
            let mut wal = WalFile::create(&path, header(), Durability::None).unwrap();
            wal.append_group(&[encode_record(&insert(1, vec![1])).unwrap()])
                .unwrap();
        }
        {
            let mut wal = WalFile::open(&path, Durability::None).unwrap();
            wal.append_group(&[encode_record(&insert(2, vec![2])).unwrap()])
                .unwrap();
        }

        let mut wal = WalFile::open(&path, Durability::None).unwrap();
        let recovered = wal.recover().unwrap();
        assert_eq!(
            recovered
                .records
                .iter()
                .map(|r| r.lsn())
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn a_torn_tail_is_dropped_and_the_file_is_cut_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.lt.wal0");
        let good_len;
        {
            let mut wal = WalFile::create(&path, header(), Durability::None).unwrap();
            let frames: Vec<Vec<u8>> = (1..=3)
                .map(|lsn| encode_record(&insert(lsn, vec![lsn as i32])).unwrap())
                .collect();
            wal.append_group(&frames[..2]).unwrap();
            good_len = wal.len();
            // Half of the third record reaches the disk.
            let partial = &frames[2][..frames[2].len() / 2];
            wal.append_group(&[partial.to_vec()]).unwrap();
        }

        let mut wal = WalFile::open(&path, Durability::None).unwrap();
        let recovered = wal.recover().unwrap();

        assert!(recovered.truncated, "the partial record must be noticed");
        assert_eq!(
            recovered
                .records
                .iter()
                .map(|r| r.lsn())
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(
            wal.len(),
            good_len,
            "the file must be cut at the last good record"
        );

        // Appending after recovery starts at a record boundary.
        wal.append_group(&[encode_record(&insert(3, vec![3])).unwrap()])
            .unwrap();
        let recovered = WalFile::open(&path, Durability::None)
            .unwrap()
            .recover()
            .unwrap();
        assert_eq!(
            recovered
                .records
                .iter()
                .map(|r| r.lsn())
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn a_damaged_record_stops_recovery_at_that_point() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.lt.wal0");
        let first_len;
        {
            let mut wal = WalFile::create(&path, header(), Durability::None).unwrap();
            let frames: Vec<Vec<u8>> = (1..=3)
                .map(|lsn| encode_record(&insert(lsn, vec![lsn as i32])).unwrap())
                .collect();
            first_len = frames[0].len();
            wal.append_group(&frames).unwrap();
        }

        // Corrupt the second record's payload.
        {
            use std::os::unix::fs::FileExt;
            let file = OpenOptions::new().write(true).open(&path).unwrap();
            let target = WAL_HEADER_LEN + first_len as u64 + FRAME_HEADER_LEN as u64 + 4;
            file.write_all_at(&[0xa5], target).unwrap();
        }

        let mut wal = WalFile::open(&path, Durability::None).unwrap();
        let recovered = wal.recover().unwrap();
        assert!(recovered.truncated);
        assert_eq!(
            recovered.records.iter().map(|r| r.lsn()).collect::<Vec<_>>(),
            vec![1],
            "a damaged record hides everything after it, because their order is what makes them mean anything"
        );
    }

    #[test]
    fn truncating_keeps_the_file_usable() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = wal(&dir);
        wal.append_group(&[encode_record(&insert(1, vec![1])).unwrap()])
            .unwrap();

        wal.truncate().unwrap();
        assert!(wal.is_empty());
        assert!(wal.recover().unwrap().records.is_empty());

        wal.append_group(&[encode_record(&insert(2, vec![2])).unwrap()])
            .unwrap();
        assert_eq!(wal.recover().unwrap().records.len(), 1);
    }

    #[test]
    fn rotating_starts_a_new_generation() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = wal(&dir);
        wal.append_group(&[encode_record(&insert(1, vec![1])).unwrap()])
            .unwrap();

        wal.rotate(9).unwrap();
        assert_eq!(wal.header().generation, 9);
        assert!(wal.is_empty());

        let reopened = WalFile::open(wal.path(), Durability::None).unwrap();
        assert_eq!(reopened.header().generation, 9);
    }

    #[test]
    fn appending_nothing_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = wal(&dir);
        wal.append_group(&[]).unwrap();
        assert!(wal.is_empty());
    }
}
