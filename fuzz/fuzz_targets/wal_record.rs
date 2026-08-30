//! Write-ahead log records, as recovery reads them.
//!
//! Recovery reads a log written before a crash, so a torn tail is the normal
//! way a log ends. Any bytes must give a record or an error.
#![no_main]

use libfuzzer_sys::fuzz_target;
use localtables_format::layout::frame;
use localtables_format::layout::frame::tag;
use localtables_format::wal::record::ArchivedWalRecord;

fuzz_target!(|data: &[u8]| {
    // Recovery frames each record, then reads the archive inside.
    let Ok(payload) = frame::decode(data, tag::WAL_REC, "fuzz") else {
        return;
    };
    if let Ok(record) = rkyv::access::<ArchivedWalRecord, rkyv::rancor::Error>(payload) {
        // A record the validator accepted must answer for itself.
        let _ = record.lsn();
        let _ = record.kind();
    }
});
