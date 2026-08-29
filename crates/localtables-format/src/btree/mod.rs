//! The b-tree table: keys, rows, and the tree that orders them.

pub mod keycodec;
pub mod node;
pub mod rowcodec;
pub mod table;
pub mod tree;

pub use table::{BTreeSnapshot, BTreeTable};
