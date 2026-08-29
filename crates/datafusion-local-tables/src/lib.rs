//! DataFusion table providers backed by a single local file.
//!
//! This crate wires [`localtables_format`] storage engines into DataFusion.
//! The read path arrives in phase 5, the DML path in phase 6.

pub use localtables_format as format;
