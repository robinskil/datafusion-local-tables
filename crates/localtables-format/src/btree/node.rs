//! B-tree pages.
//!
//! A page is one rkyv archive. The tree is copy-on-write, so a change rewrites
//! whole pages anyway; serialising on write and reading the archive in place
//! costs nothing over editing bytes by hand, and it means a damaged page fails
//! a check rather than being walked as if it were fine.
//!
//! Leaves hold keys and packed rows. Branches hold separator keys and the
//! extents of their children: to find the child for a key, take the first
//! separator the key does not exceed.

use rkyv::{Archive, Deserialize, Serialize};

use crate::layout::frame::{self, tag};
use crate::layout::Extent;
use crate::{Error, Result};

/// One key and the row it maps to.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[rkyv(derive(Debug))]
pub struct LeafEntry {
    pub key: Vec<u8>,
    /// The row, packed by `rowcodec`.
    pub row: Vec<u8>,
}

/// A child, and the largest key that can live under it.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[rkyv(derive(Debug))]
pub struct BranchEntry {
    /// Every key under `child` is less than or equal to this.
    pub separator: Vec<u8>,
    pub child: Extent,
}

/// A page of the tree.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[rkyv(derive(Debug))]
pub enum Node {
    Leaf { entries: Vec<LeafEntry> },
    Branch { entries: Vec<BranchEntry> },
}

impl Node {
    pub fn leaf(entries: Vec<LeafEntry>) -> Self {
        Node::Leaf { entries }
    }

    pub fn branch(entries: Vec<BranchEntry>) -> Self {
        Node::Branch { entries }
    }

    /// Keys, or children, this page holds.
    pub fn len(&self) -> usize {
        match self {
            Node::Leaf { entries } => entries.len(),
            Node::Branch { entries } => entries.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The largest key anywhere under this page.
    ///
    /// A branch above it uses this as its separator.
    pub fn high_key(&self) -> Option<&[u8]> {
        match self {
            Node::Leaf { entries } => entries.last().map(|e| e.key.as_slice()),
            Node::Branch { entries } => entries.last().map(|e| e.separator.as_slice()),
        }
    }

    /// Serialize into a frame ready to be written.
    pub fn to_frame(&self) -> Result<Vec<u8>> {
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(self)?;
        Ok(frame::encode(tag::BTREE_NODE, &payload))
    }
}

/// Read a page's archive out of its frame.
pub fn read(bytes: &[u8]) -> Result<&ArchivedNode> {
    let payload = frame::decode(bytes, tag::BTREE_NODE, "b-tree node")?;
    rkyv::access::<ArchivedNode, rkyv::rancor::Error>(payload).map_err(Error::from)
}

impl ArchivedNode {
    pub fn len(&self) -> usize {
        match self {
            ArchivedNode::Leaf { entries } => entries.len(),
            ArchivedNode::Branch { entries } => entries.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self, ArchivedNode::Leaf { .. })
    }

    /// The row stored under `key`, if the page holds it.
    ///
    /// Entries are sorted, so this is a binary search.
    pub fn find(&self, key: &[u8]) -> Option<&[u8]> {
        let ArchivedNode::Leaf { entries } = self else {
            return None;
        };
        let index = entries
            .binary_search_by(|entry| entry.key.as_ref().cmp(key))
            .ok()?;
        Some(entries[index].row.as_ref())
    }

    /// Where the first key at or after `key` sits in this leaf.
    ///
    /// A range scan starts here and walks forward.
    pub fn lower_bound(&self, key: &[u8]) -> usize {
        let ArchivedNode::Leaf { entries } = self else {
            return 0;
        };
        entries.partition_point(|entry| entry.key.as_ref() < key)
    }

    /// The child a key belongs under.
    ///
    /// `None` when the key is past every separator, which means the tree does
    /// not hold it.
    pub fn child_for(&self, key: &[u8]) -> Option<Extent> {
        let ArchivedNode::Branch { entries } = self else {
            return None;
        };
        let index = entries.partition_point(|entry| entry.separator.as_ref() < key);
        entries.get(index).map(|entry| entry.child.to_native())
    }

    /// Every separator and child of a branch, left to right.
    ///
    /// A child's separator is the largest key beneath it, so a scan looking for
    /// keys at or after some bound can skip every child whose separator is
    /// below it.
    pub fn branch_entries(&self) -> Vec<(&[u8], Extent)> {
        match self {
            ArchivedNode::Branch { entries } => entries
                .iter()
                .map(|entry| (entry.separator.as_ref(), entry.child.to_native()))
                .collect(),
            ArchivedNode::Leaf { .. } => Vec::new(),
        }
    }

    /// Every child of a branch, left to right.
    pub fn children(&self) -> Vec<Extent> {
        match self {
            ArchivedNode::Branch { entries } => {
                entries.iter().map(|e| e.child.to_native()).collect()
            }
            ArchivedNode::Leaf { .. } => Vec::new(),
        }
    }

    /// Key and row at `index`, for a scan walking a leaf.
    pub fn entry(&self, index: usize) -> Option<(&[u8], &[u8])> {
        let ArchivedNode::Leaf { entries } = self else {
            return None;
        };
        entries
            .get(index)
            .map(|entry| (entry.key.as_ref(), entry.row.as_ref()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(keys: &[&[u8]]) -> Node {
        Node::leaf(
            keys.iter()
                .map(|key| LeafEntry {
                    key: key.to_vec(),
                    row: format!("row for {key:?}").into_bytes(),
                })
                .collect(),
        )
    }

    fn branch(separators: &[(&[u8], u64)]) -> Node {
        Node::branch(
            separators
                .iter()
                .map(|(separator, offset)| BranchEntry {
                    separator: separator.to_vec(),
                    child: Extent::new(*offset, 100),
                })
                .collect(),
        )
    }

    fn archived(node: &Node) -> Vec<u8> {
        node.to_frame().unwrap()
    }

    #[test]
    fn a_leaf_round_trips() {
        let node = leaf(&[b"a", b"b", b"c"]);
        let frame = archived(&node);
        let read = read(&frame).unwrap();

        assert!(read.is_leaf());
        assert_eq!(read.len(), 3);
        assert_eq!(read.find(b"b").unwrap(), b"row for [98]");
        assert!(read.find(b"z").is_none());
    }

    #[test]
    fn a_branch_round_trips() {
        let node = branch(&[(b"m", 4096), (b"z", 8192)]);
        let frame = archived(&node);
        let read = read(&frame).unwrap();

        assert!(!read.is_leaf());
        assert_eq!(
            read.children(),
            vec![Extent::new(4096, 100), Extent::new(8192, 100)]
        );
    }

    #[test]
    fn a_branch_sends_a_key_to_the_first_separator_it_does_not_exceed() {
        let frame = archived(&branch(&[(b"m", 1), (b"z", 2)]));
        let read = read(&frame).unwrap();

        assert_eq!(read.child_for(b"a").unwrap().offset, 1);
        assert_eq!(
            read.child_for(b"m").unwrap().offset,
            1,
            "a key equal to a separator stays left"
        );
        assert_eq!(read.child_for(b"n").unwrap().offset, 2);
        assert_eq!(read.child_for(b"z").unwrap().offset, 2);
        assert!(
            read.child_for(b"zz").is_none(),
            "a key past every separator is not in the tree"
        );
    }

    #[test]
    fn branch_entries_pair_each_separator_with_its_child() {
        let frame = archived(&branch(&[(b"m", 1), (b"z", 2)]));
        let read = read(&frame).unwrap();

        let pairs = read.branch_entries();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, b"m");
        assert_eq!(pairs[0].1.offset, 1);
        assert_eq!(pairs[1].0, b"z");
        assert_eq!(pairs[1].1.offset, 2);
    }

    #[test]
    fn a_leaf_reports_where_a_range_should_start() {
        let frame = archived(&leaf(&[b"b", b"d", b"f"]));
        let read = read(&frame).unwrap();

        assert_eq!(read.lower_bound(b"a"), 0);
        assert_eq!(read.lower_bound(b"b"), 0);
        assert_eq!(read.lower_bound(b"c"), 1);
        assert_eq!(read.lower_bound(b"f"), 2);
        assert_eq!(read.lower_bound(b"g"), 3, "past the end is the end");
    }

    #[test]
    fn entries_are_readable_in_order() {
        let frame = archived(&leaf(&[b"a", b"b"]));
        let read = read(&frame).unwrap();

        assert_eq!(read.entry(0).unwrap().0, b"a");
        assert_eq!(read.entry(1).unwrap().0, b"b");
        assert!(read.entry(2).is_none());
    }

    #[test]
    fn the_high_key_is_the_largest_key_below() {
        assert_eq!(leaf(&[b"a", b"m"]).high_key(), Some(&b"m"[..]));
        assert_eq!(branch(&[(b"m", 1), (b"z", 2)]).high_key(), Some(&b"z"[..]));
        assert_eq!(Node::leaf(Vec::new()).high_key(), None);
    }

    #[test]
    fn an_empty_leaf_round_trips() {
        let frame = archived(&Node::leaf(Vec::new()));
        let read = read(&frame).unwrap();
        assert!(read.is_empty());
        assert!(read.find(b"a").is_none());
    }

    #[test]
    fn a_damaged_page_is_refused() {
        let mut frame = archived(&leaf(&[b"a", b"b", b"c"]));
        let middle = frame.len() / 2;
        frame[middle] ^= 0xff;

        let err = read(&frame).unwrap_err();
        assert!(matches!(err, Error::Checksum { .. }), "got {err:?}");
    }

    #[test]
    fn a_frame_with_the_wrong_tag_is_refused() {
        let frame = frame::encode(tag::SEGMENT, b"not a node");
        assert!(matches!(read(&frame).unwrap_err(), Error::BadMagic(_)));
    }
}
