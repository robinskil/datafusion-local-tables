//! Building and searching the tree.
//!
//! The tree is copy-on-write and built bottom-up: a write merges the sorted
//! overlay with the existing entries, writes fresh leaves, then fresh branches
//! over them, and commits a new root. Nothing already on disk is overwritten,
//! so a reader on the old root keeps reading a complete tree throughout.
//!
//! Rewriting whole levels costs more per write than editing pages in place, and
//! buys crash safety for free: there is no torn page to recover, only a root
//! that either points at the new tree or the old one.

use std::collections::BTreeMap;

use crate::btree::node::{self, ArchivedNode, BranchEntry, LeafEntry, Node};
use crate::io::FileIo;
use crate::layout::manifest::Manifest;
use crate::layout::{Extent, BUFFER_ALIGN};
use crate::table_file::TableFile;
use crate::{Error, Result};

/// Entries per leaf page.
///
/// Small enough that a lookup reads little, large enough that the tree stays
/// shallow: at 256 per level, three levels hold sixteen million rows.
pub const LEAF_FANOUT: usize = 256;

/// Children per branch page.
pub const BRANCH_FANOUT: usize = 256;

/// What a write does to one key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Change {
    /// Store this row under the key, replacing any row already there.
    Put(Vec<u8>),
    /// Remove the key.
    Delete,
}

/// Pending changes, in key order.
pub type Overlay = BTreeMap<Vec<u8>, Change>;

/// A tree that has been written and is ready to be committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TreeRoot {
    /// The root page, or empty when the tree holds nothing.
    pub root: Extent,
    pub rows: u64,
    /// Levels from the root to a leaf. Zero for an empty tree.
    pub height: u32,
}

impl TreeRoot {
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }
}

/// Read the row stored under `key`.
///
/// Walks one page per level: the branch chain narrows to a leaf, and the leaf
/// is searched. A miss costs the same walk, which is what a b-tree is for.
pub async fn lookup(io: &dyn FileIo, root: TreeRoot, key: &[u8]) -> Result<Option<Vec<u8>>> {
    if root.is_empty() {
        return Ok(None);
    }
    let mut extent = root.root;

    // Bounded by the height, so a cycle in a damaged tree cannot loop forever.
    for _ in 0..=root.height {
        let bytes = io.read_immutable(extent).await?;
        let page = node::read(bytes.as_slice())?;
        match page {
            ArchivedNode::Leaf { .. } => return Ok(page.find(key).map(|row| row.to_vec())),
            ArchivedNode::Branch { .. } => match page.child_for(key) {
                Some(child) => extent = child,
                // Past every separator: no leaf can hold this key.
                None => return Ok(None),
            },
        }
    }
    Err(Error::corrupt(
        "b-tree walk did not reach a leaf within its height",
    ))
}

/// Read every key in `start..end`, in order.
///
/// `end` of `None` runs to the end of the tree.
pub async fn range(
    io: &dyn FileIo,
    root: TreeRoot,
    start: &[u8],
    end: Option<&[u8]>,
    limit: Option<usize>,
) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    if root.is_empty() {
        return Ok(Vec::new());
    }

    // The leaves in order, starting from the one that could hold `start`.
    let leaves = leaves_from(io, root, start).await?;
    let mut out = Vec::new();

    for extent in leaves {
        let bytes = io.read_immutable(extent).await?;
        let page = node::read(bytes.as_slice())?;
        let mut index = page.lower_bound(start);
        // Only the first leaf can start partway in; later ones start at zero.
        while let Some((key, row)) = page.entry(index) {
            if end.is_some_and(|end| key >= end) {
                return Ok(out);
            }
            out.push((key.to_vec(), row.to_vec()));
            if limit.is_some_and(|limit| out.len() >= limit) {
                return Ok(out);
            }
            index += 1;
        }
    }
    Ok(out)
}

/// The leaves that could hold keys at or after `start`, in order.
///
/// A child's separator is the largest key beneath it, so a child whose
/// separator is below `start` holds nothing the scan wants and is skipped
/// without being read.
async fn leaves_from(io: &dyn FileIo, root: TreeRoot, start: &[u8]) -> Result<Vec<Extent>> {
    let mut out = Vec::new();
    descend(io, root.root, root.height, start, &mut out).await?;
    Ok(out)
}

async fn descend(
    io: &dyn FileIo,
    extent: Extent,
    height: u32,
    start: &[u8],
    out: &mut Vec<Extent>,
) -> Result<()> {
    let bytes = io.read_immutable(extent).await?;
    let page = node::read(bytes.as_slice())?;

    if page.is_leaf() {
        out.push(extent);
        return Ok(());
    }
    // The height bounds the recursion, so a damaged tree that points back at
    // itself fails rather than looping.
    if height == 0 {
        return Err(Error::corrupt("a b-tree branch sits below its own height"));
    }

    for (separator, child) in page.branch_entries() {
        if separator < start {
            continue;
        }
        Box::pin(descend(io, child, height - 1, start, out)).await?;
    }
    Ok(())
}

/// Every key and row in the tree, in order.
pub async fn read_all(io: &dyn FileIo, root: TreeRoot) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    range(io, root, &[], None, None).await
}

/// Merge `overlay` into the tree and write a new one.
///
/// The old tree is left where it is: the caller frees its pages once no reader
/// is pinned to them. Returns the new root, or an empty one when nothing is
/// left.
pub async fn write_merged(
    file: &TableFile,
    manifest: &mut Manifest,
    old_root: TreeRoot,
    overlay: &Overlay,
    min_active_txn: u64,
) -> Result<TreeRoot> {
    let existing = read_all(file.io().as_ref(), old_root).await?;
    let merged = merge(existing, overlay);
    build(file, manifest, merged, min_active_txn).await
}

/// Apply an overlay to sorted entries, keeping the result sorted.
///
/// A put replaces what was there; a delete removes it. Both are decided by key,
/// so the merge is one pass over two sorted sequences.
pub fn merge(existing: Vec<(Vec<u8>, Vec<u8>)>, overlay: &Overlay) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut out = Vec::with_capacity(existing.len() + overlay.len());
    let mut changes = overlay.iter().peekable();
    let mut rows = existing.into_iter().peekable();

    loop {
        match (rows.peek(), changes.peek()) {
            (None, None) => break,
            (Some(_), None) => out.push(rows.next().expect("peeked")),
            (None, Some(_)) => {
                let (key, change) = changes.next().expect("peeked");
                if let Change::Put(row) = change {
                    out.push((key.clone(), row.clone()));
                }
            }
            (Some((row_key, _)), Some((change_key, _))) => {
                match row_key.cmp(change_key) {
                    std::cmp::Ordering::Less => out.push(rows.next().expect("peeked")),
                    std::cmp::Ordering::Greater => {
                        let (key, change) = changes.next().expect("peeked");
                        if let Change::Put(row) = change {
                            out.push((key.clone(), row.clone()));
                        }
                    }
                    // The overlay wins: it is the newer write.
                    std::cmp::Ordering::Equal => {
                        rows.next();
                        let (key, change) = changes.next().expect("peeked");
                        if let Change::Put(row) = change {
                            out.push((key.clone(), row.clone()));
                        }
                    }
                }
            }
        }
    }
    out
}

/// Write a whole tree from sorted entries.
///
/// Bottom-up: leaves first, then a level of branches over them, until one page
/// is left. That page is the root.
pub async fn build(
    file: &TableFile,
    manifest: &mut Manifest,
    entries: Vec<(Vec<u8>, Vec<u8>)>,
    min_active_txn: u64,
) -> Result<TreeRoot> {
    if entries.is_empty() {
        return Ok(TreeRoot::default());
    }
    let rows = entries.len() as u64;

    // Leaves.
    let mut level: Vec<(Vec<u8>, Extent)> = Vec::new();
    for chunk in entries.chunks(LEAF_FANOUT) {
        let node = Node::leaf(
            chunk
                .iter()
                .map(|(key, row)| LeafEntry {
                    key: key.clone(),
                    row: row.clone(),
                })
                .collect(),
        );
        let high = node.high_key().expect("a chunk is never empty").to_vec();
        let extent = write_page(file, manifest, &node, min_active_txn).await?;
        level.push((high, extent));
    }

    // Branches, until one page covers everything.
    let mut height = 0u32;
    while level.len() > 1 {
        height += 1;
        let mut next = Vec::with_capacity(level.len().div_ceil(BRANCH_FANOUT));
        for chunk in level.chunks(BRANCH_FANOUT) {
            let node = Node::branch(
                chunk
                    .iter()
                    .map(|(separator, child)| BranchEntry {
                        separator: separator.clone(),
                        child: *child,
                    })
                    .collect(),
            );
            let high = node.high_key().expect("a chunk is never empty").to_vec();
            let extent = write_page(file, manifest, &node, min_active_txn).await?;
            next.push((high, extent));
        }
        level = next;
    }

    Ok(TreeRoot {
        root: level[0].1,
        rows,
        height,
    })
}

/// Place one page in the file.
async fn write_page(
    file: &TableFile,
    manifest: &mut Manifest,
    node: &Node,
    min_active_txn: u64,
) -> Result<Extent> {
    let bytes = node.to_frame()?;
    file.write_allocated(manifest, &bytes, BUFFER_ALIGN, min_active_txn)
        .await
}

/// Every page of a tree, so a commit can free them all.
pub async fn all_pages(io: &dyn FileIo, root: TreeRoot) -> Result<Vec<Extent>> {
    if root.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    collect_pages(io, root.root, root.height, &mut out).await?;
    Ok(out)
}

async fn collect_pages(
    io: &dyn FileIo,
    extent: Extent,
    height: u32,
    out: &mut Vec<Extent>,
) -> Result<()> {
    out.push(extent);
    let bytes = io.read_immutable(extent).await?;
    let page = node::read(bytes.as_slice())?;
    if page.is_leaf() {
        return Ok(());
    }
    if height == 0 {
        return Err(Error::corrupt("a b-tree branch sits below its own height"));
    }
    for child in page.children() {
        Box::pin(collect_pages(io, child, height - 1, out)).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries(keys: &[&str]) -> Vec<(Vec<u8>, Vec<u8>)> {
        keys.iter()
            .map(|key| (key.as_bytes().to_vec(), format!("row-{key}").into_bytes()))
            .collect()
    }

    fn overlay(changes: &[(&str, Option<&str>)]) -> Overlay {
        changes
            .iter()
            .map(|(key, row)| {
                (
                    key.as_bytes().to_vec(),
                    match row {
                        Some(row) => Change::Put(row.as_bytes().to_vec()),
                        None => Change::Delete,
                    },
                )
            })
            .collect()
    }

    fn keys(entries: &[(Vec<u8>, Vec<u8>)]) -> Vec<String> {
        entries
            .iter()
            .map(|(key, _)| String::from_utf8(key.clone()).unwrap())
            .collect()
    }

    #[test]
    fn merging_an_empty_overlay_changes_nothing() {
        let existing = entries(&["a", "b"]);
        assert_eq!(merge(existing.clone(), &Overlay::new()), existing);
    }

    #[test]
    fn a_put_for_a_new_key_lands_in_order() {
        let merged = merge(entries(&["a", "c"]), &overlay(&[("b", Some("row-b"))]));
        assert_eq!(keys(&merged), vec!["a", "b", "c"]);
    }

    #[test]
    fn a_put_for_an_existing_key_replaces_it() {
        let merged = merge(entries(&["a", "b"]), &overlay(&[("b", Some("new"))]));
        assert_eq!(keys(&merged), vec!["a", "b"]);
        assert_eq!(merged[1].1, b"new");
    }

    #[test]
    fn a_delete_removes_a_key() {
        let merged = merge(entries(&["a", "b", "c"]), &overlay(&[("b", None)]));
        assert_eq!(keys(&merged), vec!["a", "c"]);
    }

    #[test]
    fn a_delete_for_a_key_that_is_not_there_changes_nothing() {
        let merged = merge(entries(&["a", "c"]), &overlay(&[("b", None)]));
        assert_eq!(keys(&merged), vec!["a", "c"]);
    }

    #[test]
    fn puts_and_deletes_mix() {
        let merged = merge(
            entries(&["a", "b", "c", "d"]),
            &overlay(&[("a", None), ("c", Some("new-c")), ("e", Some("row-e"))]),
        );
        assert_eq!(keys(&merged), vec!["b", "c", "d", "e"]);
        assert_eq!(merged[1].1, b"new-c");
    }

    #[test]
    fn an_overlay_over_nothing_becomes_the_whole_tree() {
        let merged = merge(Vec::new(), &overlay(&[("b", Some("1")), ("a", Some("2"))]));
        assert_eq!(
            keys(&merged),
            vec!["a", "b"],
            "the overlay is already sorted"
        );
    }

    #[test]
    fn deleting_everything_leaves_nothing() {
        let merged = merge(entries(&["a", "b"]), &overlay(&[("a", None), ("b", None)]));
        assert!(merged.is_empty());
    }

    #[test]
    fn an_empty_tree_root_holds_nothing() {
        assert!(TreeRoot::default().is_empty());
    }
}
