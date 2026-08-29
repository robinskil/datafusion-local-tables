//! The write-ahead log.
//!
//! A small insert should not have to write a segment. It appends a record here
//! instead, waits for one sync, and lands in the memtable. A flush later turns
//! the accumulated rows into one segment and empties the log.
//!
//! Tables keep two log files and swap between them. A flush freezes the
//! memtable and switches appends to the other file, so it can truncate the log
//! it just made durable while new writes carry on into the fresh one. With one
//! file the truncation could only happen if nothing was being written at that
//! moment, which is exactly when it is least needed.

pub mod file;
pub mod record;

use std::path::{Path, PathBuf};

use crate::config::Durability;
use crate::{Error, Result};

pub use file::{encode_record, RecoveredLog, WalFile, WalHeader, WAL_HEADER_LEN};
pub use record::{KeyChange, Lsn, SegmentDeletes, WalRecord};

/// The two log files belonging to one table.
///
/// Named after the table file, so a directory listing makes the relationship
/// obvious and a stray log cannot attach itself to a different table: the
/// header carries the table's identity, and a mismatch is refused.
#[derive(Debug)]
pub struct WalPaths {
    pub a: PathBuf,
    pub b: PathBuf,
}

impl WalPaths {
    pub fn for_table(table: &Path) -> Self {
        let mut a = table.as_os_str().to_owned();
        a.push(".wal0");
        let mut b = table.as_os_str().to_owned();
        b.push(".wal1");
        Self {
            a: PathBuf::from(a),
            b: PathBuf::from(b),
        }
    }

    pub fn both(&self) -> [&Path; 2] {
        [&self.a, &self.b]
    }

    /// Remove both log files. Used when a table is dropped.
    pub fn remove(&self) -> Result<()> {
        for path in self.both() {
            match std::fs::remove_file(path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(Error::io(path.to_path_buf(), e)),
            }
        }
        Ok(())
    }
}

/// Which of the two logs holds the newest records, and so takes appends.
///
/// A log that was never rotated is generation zero, so both are zero on a fresh
/// table. Either would do there, and the first wins. This is the only rule that
/// decides the active log: opening and recovering must not disagree, or an
/// append could land in the log a flush is about to truncate.
fn newest(files: &[WalFile; 2]) -> usize {
    usize::from(files[1].header().generation > files[0].header().generation)
}

/// Both log files, with one of them taking appends.
#[derive(Debug)]
pub struct WalPair {
    files: [WalFile; 2],
    /// Index of the file appends currently go to.
    active: usize,
    durability: Durability,
}

impl WalPair {
    /// Open both logs, creating either that is missing.
    ///
    /// The one with the higher generation takes new appends, because it holds
    /// the newer records.
    pub fn open(paths: &WalPaths, table_uuid: [u8; 16], durability: Durability) -> Result<Self> {
        let a = Self::open_one(&paths.a, table_uuid, durability)?;
        let b = Self::open_one(&paths.b, table_uuid, durability)?;
        let files = [a, b];
        Ok(Self {
            active: newest(&files),
            files,
            durability,
        })
    }

    fn open_one(path: &Path, table_uuid: [u8; 16], durability: Durability) -> Result<WalFile> {
        if !path.exists() {
            return WalFile::create(
                path,
                WalHeader {
                    generation: 0,
                    table_uuid,
                },
                durability,
            );
        }
        let file = WalFile::open(path, durability)?;
        if file.header().table_uuid != table_uuid {
            return Err(Error::corrupt(format!(
                "{} is a log for a different table",
                path.display()
            )));
        }
        Ok(file)
    }

    /// The file appends go to.
    pub fn active(&mut self) -> &mut WalFile {
        &mut self.files[self.active]
    }

    /// Bytes of records in the active log.
    pub fn active_len(&self) -> u64 {
        self.files[self.active].len()
    }

    pub fn active_generation(&self) -> u64 {
        self.files[self.active].header().generation
    }

    /// Append records to the active log and sync once.
    pub fn append_group(&mut self, frames: &[Vec<u8>]) -> Result<()> {
        self.active().append_group(frames)
    }

    /// Switch appends to the other log, leaving the current one as it is.
    ///
    /// Returns the index of the log that was just retired, which is the one a
    /// flush will truncate once its records are inside a segment.
    pub fn rotate(&mut self) -> Result<usize> {
        let retired = self.active;
        let next_generation = self.active_generation() + 1;
        let next = 1 - self.active;
        self.files[next].rotate(next_generation)?;
        self.active = next;
        Ok(retired)
    }

    /// Drop the records of a retired log, after a flush made them durable.
    pub fn truncate(&mut self, index: usize) -> Result<()> {
        self.files[index].truncate()
    }

    /// Read both logs in generation order, discarding any torn tail.
    ///
    /// Records are returned oldest first, which is the order they have to be
    /// replayed in.
    pub fn recover(&mut self) -> Result<Vec<WalRecord>> {
        let mut logs: Vec<RecoveredLog> = Vec::with_capacity(2);
        for file in &mut self.files {
            logs.push(file.recover()?);
        }
        logs.sort_by_key(|log| log.generation);

        // Appends resume in whichever log holds the newest records, by the same
        // rule `open` used. Recovery must not move them somewhere else.
        self.active = newest(&self.files);

        let mut records: Vec<WalRecord> = logs.into_iter().flat_map(|log| log.records).collect();
        // Generations order the files; LSNs order the records inside them. A
        // sort by LSN covers both, and makes replay independent of how the
        // files happened to be laid out.
        records.sort_by_key(|record| record.lsn());
        Ok(records)
    }

    pub fn durability(&self) -> Durability {
        self.durability
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::batchcodec;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    const UUID: [u8; 16] = [3u8; 16];

    fn insert(lsn: Lsn) -> WalRecord {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![lsn as i32]))])
                .unwrap();
        WalRecord::Insert {
            lsn,
            base_seqno: 0,
            batch: batchcodec::encode(&batch),
        }
    }

    fn pair(dir: &tempfile::TempDir) -> (WalPaths, WalPair) {
        let paths = WalPaths::for_table(&dir.path().join("t.lt"));
        let pair = WalPair::open(&paths, UUID, Durability::None).unwrap();
        (paths, pair)
    }

    fn append(pair: &mut WalPair, lsns: impl IntoIterator<Item = Lsn>) {
        let frames: Vec<Vec<u8>> = lsns
            .into_iter()
            .map(|lsn| encode_record(&insert(lsn)).unwrap())
            .collect();
        pair.append_group(&frames).unwrap();
    }

    fn lsns(records: &[WalRecord]) -> Vec<Lsn> {
        records.iter().map(|r| r.lsn()).collect()
    }

    #[test]
    fn log_paths_sit_beside_the_table() {
        let paths = WalPaths::for_table(Path::new("/data/orders.lt"));
        assert_eq!(paths.a, Path::new("/data/orders.lt.wal0"));
        assert_eq!(paths.b, Path::new("/data/orders.lt.wal1"));
    }

    #[test]
    fn a_fresh_pair_starts_on_the_first_log() {
        let dir = tempfile::tempdir().unwrap();
        let (_paths, mut pair) = pair(&dir);
        assert_eq!(pair.active, 0);
        assert!(pair.recover().unwrap().is_empty());
    }

    #[test]
    fn records_survive_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let (paths, mut pair) = pair(&dir);
        append(&mut pair, [1, 2, 3]);
        drop(pair);

        let mut pair = WalPair::open(&paths, UUID, Durability::None).unwrap();
        assert_eq!(lsns(&pair.recover().unwrap()), vec![1, 2, 3]);
    }

    #[test]
    fn rotation_sends_new_records_to_the_other_log() {
        let dir = tempfile::tempdir().unwrap();
        let (_paths, mut pair) = pair(&dir);
        append(&mut pair, [1, 2]);

        let retired = pair.rotate().unwrap();
        assert_eq!(retired, 0);
        assert_eq!(pair.active, 1);
        assert_eq!(pair.active_len(), 0, "the new log starts empty");

        append(&mut pair, [3, 4]);
        assert_eq!(
            lsns(&pair.recover().unwrap()),
            vec![1, 2, 3, 4],
            "both logs are replayed, oldest generation first"
        );
    }

    #[test]
    fn truncating_a_retired_log_leaves_only_the_new_records() {
        let dir = tempfile::tempdir().unwrap();
        let (_paths, mut pair) = pair(&dir);
        append(&mut pair, [1, 2]);
        let retired = pair.rotate().unwrap();
        append(&mut pair, [3, 4]);

        // A flush has folded lsn 1 and 2 into a segment.
        pair.truncate(retired).unwrap();
        assert_eq!(lsns(&pair.recover().unwrap()), vec![3, 4]);
    }

    #[test]
    fn appends_continue_in_the_newest_log_after_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let (paths, mut pair) = pair(&dir);
        append(&mut pair, [1]);
        let retired = pair.rotate().unwrap();
        append(&mut pair, [2]);
        pair.truncate(retired).unwrap();
        drop(pair);

        let mut pair = WalPair::open(&paths, UUID, Durability::None).unwrap();
        assert_eq!(lsns(&pair.recover().unwrap()), vec![2]);

        append(&mut pair, [3]);
        assert_eq!(
            lsns(&pair.recover().unwrap()),
            vec![2, 3],
            "a reopened pair must append to the log holding the newest records"
        );
    }

    #[test]
    fn many_rotations_keep_the_records_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let (_paths, mut pair) = pair(&dir);

        let mut lsn = 1;
        for _ in 0..10 {
            append(&mut pair, [lsn, lsn + 1]);
            lsn += 2;
            let retired = pair.rotate().unwrap();
            pair.truncate(retired).unwrap();
        }
        append(&mut pair, [lsn]);

        assert_eq!(lsns(&pair.recover().unwrap()), vec![lsn]);
    }

    #[test]
    fn a_log_belonging_to_another_table_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let (paths, mut pair) = pair(&dir);
        append(&mut pair, [1]);
        drop(pair);

        let err = WalPair::open(&paths, [9u8; 16], Durability::None).unwrap_err();
        assert!(matches!(err, Error::Corrupt(_)), "got {err:?}");
    }

    #[test]
    fn removing_the_logs_is_safe_to_repeat() {
        let dir = tempfile::tempdir().unwrap();
        let (paths, _pair) = pair(&dir);
        paths.remove().unwrap();
        paths
            .remove()
            .expect("removing absent logs is not an error");
        assert!(!paths.a.exists() && !paths.b.exists());
    }
}
