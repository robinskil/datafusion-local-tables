//! What a write-ahead log record holds.
//!
//! Three shapes, one per kind of write. An update is one record, not a delete
//! followed by an insert, because a crash between two records must never leave
//! the rows gone and their replacements missing.

use rkyv::{Archive, Deserialize, Serialize};

use crate::layout::batchcodec::BatchData;
use crate::layout::manifest::SegmentId;

/// A log sequence number. Rises by one per record and never repeats.
pub type Lsn = u64;

/// One key's fate: a row to store, or nothing when the key is deleted.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[rkyv(derive(Debug))]
pub struct KeyChange {
    pub key: Vec<u8>,
    /// The packed row, or absent when this change removes the key.
    pub row: Option<Vec<u8>>,
}

/// Rows deleted from one segment, as a serialized roaring bitmap.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[rkyv(derive(Debug))]
pub struct SegmentDeletes {
    pub segment_id: SegmentId,
    pub bitmap: Vec<u8>,
}

/// One durable write.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[rkyv(derive(Debug))]
pub enum WalRecord {
    /// Rows appended to the table.
    Insert {
        lsn: Lsn,
        /// Sequence number the first row was given. Replay puts the rows back
        /// at the same numbers, so a logged delete still names the right rows.
        base_seqno: u64,
        batch: BatchData,
    },
    /// Rows removed, in segments and in the memtable.
    Delete {
        lsn: Lsn,
        segments: Vec<SegmentDeletes>,
        /// Memtable rows, by the sequence number they were given on insert.
        memtable_rows: Vec<u64>,
    },
    /// Rows replaced. Both halves land together or neither does.
    Update {
        lsn: Lsn,
        segments: Vec<SegmentDeletes>,
        memtable_rows: Vec<u64>,
        base_seqno: u64,
        batch: BatchData,
    },
    /// Keys written or removed in a b-tree table.
    ///
    /// One record covers a whole statement, so a crash cannot apply half of it.
    BTree { lsn: Lsn, changes: Vec<KeyChange> },
}

impl WalRecord {
    pub fn lsn(&self) -> Lsn {
        match self {
            WalRecord::Insert { lsn, .. }
            | WalRecord::Delete { lsn, .. }
            | WalRecord::Update { lsn, .. }
            | WalRecord::BTree { lsn, .. } => *lsn,
        }
    }
}

impl ArchivedWalRecord {
    pub fn lsn(&self) -> Lsn {
        match self {
            ArchivedWalRecord::Insert { lsn, .. }
            | ArchivedWalRecord::Delete { lsn, .. }
            | ArchivedWalRecord::Update { lsn, .. }
            | ArchivedWalRecord::BTree { lsn, .. } => lsn.to_native(),
        }
    }

    /// A short description, for errors and diagnostics.
    pub fn kind(&self) -> &'static str {
        match self {
            ArchivedWalRecord::Insert { .. } => "insert",
            ArchivedWalRecord::Delete { .. } => "delete",
            ArchivedWalRecord::Update { .. } => "update",
            ArchivedWalRecord::BTree { .. } => "btree",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch_data() -> BatchData {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
        crate::layout::batchcodec::encode(&batch)
    }

    fn round_trip(record: &WalRecord) -> WalRecord {
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(record).unwrap();
        let archived = rkyv::access::<ArchivedWalRecord, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(archived.lsn(), record.lsn());
        rkyv::deserialize::<_, rkyv::rancor::Error>(archived).unwrap()
    }

    #[test]
    fn every_record_shape_round_trips() {
        let records = vec![
            WalRecord::Insert {
                lsn: 1,
                base_seqno: 0,
                batch: batch_data(),
            },
            WalRecord::Delete {
                lsn: 2,
                segments: vec![SegmentDeletes {
                    segment_id: 7,
                    bitmap: vec![1, 2, 3, 4],
                }],
                memtable_rows: vec![10, 11],
            },
            WalRecord::BTree {
                lsn: 4,
                changes: vec![
                    KeyChange {
                        key: b"a".to_vec(),
                        row: Some(b"row".to_vec()),
                    },
                    KeyChange {
                        key: b"b".to_vec(),
                        row: None,
                    },
                ],
            },
            WalRecord::Update {
                lsn: 3,
                segments: vec![SegmentDeletes {
                    segment_id: 7,
                    bitmap: vec![9],
                }],
                memtable_rows: vec![12],
                base_seqno: 3,
                batch: batch_data(),
            },
        ];

        for record in records {
            assert_eq!(round_trip(&record), record);
        }
    }

    #[test]
    fn the_kind_matches_the_shape() {
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&WalRecord::Delete {
            lsn: 5,
            segments: Vec::new(),
            memtable_rows: Vec::new(),
        })
        .unwrap();
        let archived = rkyv::access::<ArchivedWalRecord, rkyv::rancor::Error>(&bytes).unwrap();
        assert_eq!(archived.kind(), "delete");
        assert_eq!(archived.lsn(), 5);
    }
}
