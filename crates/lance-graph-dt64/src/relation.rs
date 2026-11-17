//! The relation socket: a source owner lends a relation's pair plane
//! (D-XLOG-W7-2 step 2, entry `2026-11-17-d-xlog-w7-2-admission.md`).
//!
//! A [`RelationSource`] hands out the [`PairPlane`] it maintains for one of
//! its relations, never a copy. [`relation_rows`] is the one read path: it
//! refuses a relation the owner does not hold (never read as empty), and the
//! plane itself refuses a frame other than the issue it was built under.
//!
//! The trait cannot prove the plane describes the owner's relation; that is
//! the implementer's obligation, as for the lane and presence sockets. What
//! the socket does establish is that the rows come from the owner, under the
//! reader's frame, in one direction, without being rebuilt.

use crate::dr::{Frame, PairDir, PairPlane, PairRefusal};

/// A relation's number within its owner. An address; it carries no pairs.
pub type RelNo = u32;

/// The socket of a relation owner.
pub trait RelationSource {
    /// The owner's own plane for `rel`, or `None` if it holds no such
    /// relation. Must not copy or rebuild.
    fn borrow_relation(&self, rel: RelNo) -> Option<&PairPlane>;
}

/// Why [`relation_rows`] refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelationRefusal {
    /// The owner holds no such relation.
    NoRelation(RelNo),
    /// The plane refused (foreign frame).
    Pair(PairRefusal),
}

/// A relation's rows in one direction, lent by its owner.
#[derive(Clone, Copy, Debug)]
pub struct RelationRows<'s> {
    rows: &'s [u64],
    words: usize,
    count: usize,
}

impl<'s> RelationRows<'s> {
    /// Words per row.
    pub fn words_per_row(&self) -> usize {
        self.words
    }

    /// Rows: the positions of the axis read.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Row `at`, or `None` past the axis.
    pub fn row(&self, at: usize) -> Option<&'s [u64]> {
        (at < self.count).then(|| &self.rows[at * self.words..(at + 1) * self.words])
    }

    /// All rows back to back.
    pub fn as_words(&self) -> &'s [u64] {
        self.rows
    }
}

/// Borrow `rel`'s rows read in `dir`, under `frame`.
pub fn relation_rows<'s, S: RelationSource + ?Sized>(
    source: &'s S,
    rel: RelNo,
    frame: &Frame,
    dir: PairDir,
) -> Result<RelationRows<'s>, RelationRefusal> {
    let plane = source
        .borrow_relation(rel)
        .ok_or(RelationRefusal::NoRelation(rel))?;
    let rows = plane.rows(frame, dir).map_err(RelationRefusal::Pair)?;
    Ok(RelationRows {
        rows,
        words: plane.words_per_row(dir),
        count: plane.rows_in(dir),
    })
}
