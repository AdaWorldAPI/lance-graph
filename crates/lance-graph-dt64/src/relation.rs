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

/// [`ReachBuf::hop`]'s value for a position the reach did not arrive at.
pub const UNREACHED: u16 = u16::MAX;

/// Caller-owned buffers of [`relation_reach`]. Nothing else is written, and
/// nothing is allocated.
#[derive(Debug)]
pub struct ReachBuf<'b> {
    /// One bit per position: the answer. `words_per_row` words.
    pub reached: &'b mut [u64],
    /// The witness: per position, the number of hops after which it was first
    /// reached (`0` for a seed), or [`UNREACHED`]. One entry per position.
    pub hop: &'b mut [u16],
    /// Working space, `2 * words_per_row` words. Contents are not meaningful
    /// after the call.
    pub scratch: &'b mut [u64],
}

/// Which input of [`relation_reach`] or [`reach_path`] had the wrong length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReachPart {
    Seed,
    Target,
    Reached,
    Hop,
    Scratch,
    Path,
}

/// Why [`relation_reach`] or [`reach_path`] refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReachRefusal {
    /// The relation could not be borrowed under the reader's frame.
    Relation(RelationRefusal),
    /// The two axes reserve different numbers of positions, so a reached
    /// position is not a row to continue from.
    NotSquare { src: usize, dst: usize },
    /// An input has the wrong length.
    Length {
        part: ReachPart,
        want: usize,
        got: usize,
    },
    /// More hops than [`ReachBuf::hop`] can number.
    HopOverflow,
    /// [`reach_path`] was asked for a position past the axis.
    OutOfAxis { at: usize, positions: usize },
    /// [`reach_path`] was asked for a position the witness marks unreached.
    NotReached(usize),
    /// The witness names a hop the relation does not hold: it was made over
    /// another relation, direction or state of the plane. Carries the
    /// position whose predecessor is missing.
    WitnessMismatch(usize),
}

/// What [`relation_reach`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReachStats {
    /// Rounds run, the last one finding nothing new.
    pub rounds: u32,
    /// Rows of the relation read.
    pub rows_read: u64,
    /// Positions reached, seeds included.
    pub reached: u64,
}

fn want(part: ReachPart, want: usize, got: usize) -> Result<(), ReachRefusal> {
    if want == got {
        Ok(())
    } else {
        Err(ReachRefusal::Length { part, want, got })
    }
}

fn square<'s, S: RelationSource + ?Sized>(
    source: &'s S,
    rel: RelNo,
    frame: &Frame,
    dir: PairDir,
) -> Result<RelationRows<'s>, ReachRefusal> {
    let rows = relation_rows(source, rel, frame, dir).map_err(ReachRefusal::Relation)?;
    let dst = rows.words_per_row() * 64;
    if rows.len() != dst {
        return Err(ReachRefusal::NotSquare {
            src: rows.len(),
            dst,
        });
    }
    Ok(rows)
}

/// The positions reachable from `seed` along `rel` read in `dir`, every step
/// landing in `target` (all positions when `None`), seeds included.
///
/// The relation is borrowed once through the socket, under `frame`; a frame
/// the plane was not built under is refused before anything is written. The
/// answer and its witness go into `buf`, which the caller owns; nothing is
/// allocated. Semi-naive: each round reads the rows of the positions found in
/// the round before.
pub fn relation_reach<S: RelationSource + ?Sized>(
    source: &S,
    rel: RelNo,
    frame: &Frame,
    dir: PairDir,
    seed: &[u64],
    target: Option<&[u64]>,
    buf: &mut ReachBuf<'_>,
) -> Result<ReachStats, ReachRefusal> {
    let rows = square(source, rel, frame, dir)?;
    let w = rows.words_per_row();
    want(ReachPart::Seed, w, seed.len())?;
    if let Some(t) = target {
        want(ReachPart::Target, w, t.len())?;
    }
    want(ReachPart::Reached, w, buf.reached.len())?;
    want(ReachPart::Hop, rows.len(), buf.hop.len())?;
    want(ReachPart::Scratch, 2 * w, buf.scratch.len())?;

    let all = rows.as_words();
    let (delta, next) = buf.scratch.split_at_mut(w);
    buf.reached.copy_from_slice(seed);
    delta.copy_from_slice(seed);
    buf.hop.fill(UNREACHED);
    let mut stats = ReachStats::default();
    for (wi, &word) in seed.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            buf.hop[wi * 64 + bits.trailing_zeros() as usize] = 0;
            bits &= bits - 1;
            stats.reached += 1;
        }
    }
    loop {
        if delta.iter().all(|&d| d == 0) {
            return Ok(stats);
        }
        stats.rounds += 1;
        let round = u16::try_from(stats.rounds)
            .ok()
            .filter(|&r| r != UNREACHED)
            .ok_or(ReachRefusal::HopOverflow)?;
        next.fill(0);
        for (wi, &word) in delta.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let x = wi * 64 + bits.trailing_zeros() as usize;
                for (n, &r) in next.iter_mut().zip(&all[x * w..(x + 1) * w]) {
                    *n |= r;
                }
                bits &= bits - 1;
                stats.rows_read += 1;
            }
        }
        for i in 0..w {
            let kept = match target {
                Some(t) => next[i] & t[i],
                None => next[i],
            };
            let new = kept & !buf.reached[i];
            delta[i] = new;
            buf.reached[i] |= new;
            let mut bits = new;
            while bits != 0 {
                buf.hop[i * 64 + bits.trailing_zeros() as usize] = round;
                bits &= bits - 1;
                stats.reached += 1;
            }
        }
    }
}

/// One path to `to`, read back from a [`relation_reach`] witness: `path[0]`
/// is a seed, `path[h]` is `to`, and each step is a pair the relation holds.
/// Returns the number of positions written (`hop[to] + 1`).
///
/// `rel`, `frame` and `dir` are those of the reach that made `hop`. The
/// relation is borrowed again and each step is looked up in it, so a witness
/// made over another relation, direction or state is refused with
/// [`ReachRefusal::WitnessMismatch`] and never read as a path. Nothing is
/// allocated.
pub fn reach_path<S: RelationSource + ?Sized>(
    source: &S,
    rel: RelNo,
    frame: &Frame,
    dir: PairDir,
    hop: &[u16],
    to: usize,
    path: &mut [u32],
) -> Result<usize, ReachRefusal> {
    let back = match dir {
        PairDir::Forward => PairDir::Backward,
        PairDir::Backward => PairDir::Forward,
    };
    let rows = square(source, rel, frame, back)?;
    want(ReachPart::Hop, rows.len(), hop.len())?;
    if to >= rows.len() {
        return Err(ReachRefusal::OutOfAxis {
            at: to,
            positions: rows.len(),
        });
    }
    let h = hop[to];
    if h == UNREACHED {
        return Err(ReachRefusal::NotReached(to));
    }
    let len = h as usize + 1;
    if path.len() < len {
        return Err(ReachRefusal::Length {
            part: ReachPart::Path,
            want: len,
            got: path.len(),
        });
    }
    let w = rows.words_per_row();
    let all = rows.as_words();
    let mut at = to;
    path[h as usize] = at as u32;
    for step in (0..h).rev() {
        let row = &all[at * w..(at + 1) * w];
        let mut found = None;
        'row: for (wi, &word) in row.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let p = wi * 64 + bits.trailing_zeros() as usize;
                if hop[p] == step {
                    found = Some(p);
                    break 'row;
                }
                bits &= bits - 1;
            }
        }
        at = found.ok_or(ReachRefusal::WitnessMismatch(at))?;
        path[step as usize] = at as u32;
    }
    Ok(len)
}
