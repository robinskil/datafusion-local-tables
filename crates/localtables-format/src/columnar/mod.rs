//! The columnar table: segments, encodings, zone maps.

pub mod codec;
pub mod decode;
pub mod encode;
pub mod page;
pub mod segment;
pub mod zonemap;

pub use page::{BufferRole, Codec, ColumnChunk, Encoding, SegmentMeta};
pub use segment::{build_segment, BuiltSegment, SegmentReader};
pub use zonemap::ZoneMap;
