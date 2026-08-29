//! The file header and the two meta page slots.
//!
//! The header is immutable after creation. The meta pages alternate: a commit
//! always overwrites the slot holding the older `txn_id`, so the newer slot
//! stays intact if the write tears.

use rkyv::{Archive, Deserialize, Serialize};

use crate::layout::{Extent, TableKind, FORMAT_VERSION, MAGIC};

/// Fixed description of the table, written once at creation.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[rkyv(derive(Debug))]
pub struct FileHeader {
    /// Guards against opening an unrelated file.
    pub magic: u64,
    pub format_version: u32,
    pub table_kind: TableKind,
    /// Identifies this table, so a WAL sidecar cannot attach to the wrong file.
    pub table_uuid: [u8; 16],
    /// Arrow schema as IPC bytes. Lives in the data region and is never freed.
    pub schema: Extent,
    /// Hash of the schema bytes. Open fails when a caller supplies another schema.
    pub schema_fingerprint: u64,
    /// Alignment the writer used for segment starts.
    pub segment_align: u32,
    /// Alignment the writer used for Arrow buffer starts.
    pub buffer_align: u32,
    /// Reserved for forward-compatible flags. Zero today.
    pub flags: u64,
}

impl FileHeader {
    pub fn new(
        table_kind: TableKind,
        table_uuid: [u8; 16],
        schema: Extent,
        schema_fingerprint: u64,
    ) -> Self {
        Self {
            magic: MAGIC,
            format_version: FORMAT_VERSION,
            table_kind,
            table_uuid,
            schema,
            schema_fingerprint,
            segment_align: crate::layout::SEGMENT_ALIGN as u32,
            buffer_align: crate::layout::BUFFER_ALIGN as u32,
            flags: 0,
        }
    }
}

impl ArchivedFileHeader {
    /// Reject files this build cannot read.
    pub fn validate(&self, expect_kind: TableKind) -> crate::Result<()> {
        if self.magic.to_native() != MAGIC {
            return Err(crate::Error::BadMagic(format!(
                "header magic {:#018x} does not match {MAGIC:#018x}",
                self.magic.to_native()
            )));
        }
        if self.format_version.to_native() != FORMAT_VERSION {
            return Err(crate::Error::Unsupported(format!(
                "format version {}, this build reads version {FORMAT_VERSION}",
                self.format_version.to_native()
            )));
        }
        if self.table_kind != expect_kind {
            return Err(crate::Error::BadMagic(format!(
                "file holds a {:?} table, opened as {expect_kind:?}",
                self.table_kind
            )));
        }
        if self.buffer_align.to_native() as u64 != crate::layout::BUFFER_ALIGN {
            return Err(crate::Error::Unsupported(format!(
                "file uses {}-byte buffer alignment, this build uses {}",
                self.buffer_align.to_native(),
                crate::layout::BUFFER_ALIGN
            )));
        }
        Ok(())
    }

    pub fn schema_extent(&self) -> Extent {
        self.schema.to_native()
    }
}

/// One commit slot. Two of these alternate at fixed offsets in the file.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[rkyv(derive(Debug))]
pub struct MetaPage {
    /// Monotonic commit counter. The larger of the two valid slots wins.
    pub txn_id: u64,
    /// Frame holding the manifest for this commit.
    pub manifest: Extent,
    /// WAL records at or below this LSN are already inside the segments.
    pub checkpoint_lsn: u64,
    /// Next LSN the writer will hand out.
    pub next_lsn: u64,
    /// File length at commit time. Bytes past this are crash garbage.
    pub file_len: u64,
}

/// Which of the two slots a commit writes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaSlot {
    A,
    B,
}

impl MetaSlot {
    pub fn offset(self) -> u64 {
        match self {
            MetaSlot::A => crate::layout::META_A_OFFSET,
            MetaSlot::B => crate::layout::META_B_OFFSET,
        }
    }

    pub fn other(self) -> MetaSlot {
        match self {
            MetaSlot::A => MetaSlot::B,
            MetaSlot::B => MetaSlot::A,
        }
    }
}

impl ArchivedMetaPage {
    pub fn to_native(&self) -> MetaPage {
        MetaPage {
            txn_id: self.txn_id.to_native(),
            manifest: self.manifest.to_native(),
            checkpoint_lsn: self.checkpoint_lsn.to_native(),
            next_lsn: self.next_lsn.to_native(),
            file_len: self.file_len.to_native(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> FileHeader {
        FileHeader::new(TableKind::Columnar, [7u8; 16], Extent::new(4096, 128), 0xabcd)
    }

    /// Read a header back the way an open does, so `validate` sees an archive
    /// rather than the struct it was built from.
    fn validate(header: &FileHeader) -> crate::Result<()> {
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(header).unwrap();
        rkyv::access::<ArchivedFileHeader, rkyv::rancor::Error>(&bytes)
            .unwrap()
            .validate(TableKind::Columnar)
    }

    #[test]
    fn a_header_this_build_wrote_is_accepted() {
        validate(&header()).unwrap();
    }

    /// The guard that refuses a file from an older format, including one
    /// written by the b-tree engine that this format no longer has.
    #[test]
    fn an_older_format_version_is_refused() {
        let older = FileHeader {
            format_version: FORMAT_VERSION - 1,
            ..header()
        };
        let err = validate(&older).unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn a_newer_format_version_is_refused_too() {
        let newer = FileHeader {
            format_version: FORMAT_VERSION + 1,
            ..header()
        };
        assert!(validate(&newer).is_err());
    }

    #[test]
    fn the_wrong_magic_is_refused() {
        let wrong = FileHeader {
            magic: MAGIC ^ 1,
            ..header()
        };
        let err = validate(&wrong).unwrap_err();
        assert!(matches!(err, crate::Error::BadMagic(_)), "got {err:?}");
    }
}
