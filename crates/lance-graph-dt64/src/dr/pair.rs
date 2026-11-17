//! Relation presence over two DR axes: one bit per coordinate pair
//! (D-XLOG-W7-2, entry `2026-11-17-d-xlog-w7-2-admission.md`).
//!
//! A [`PairPlane`] holds, for a binary relation placed in one frame, which
//! pairs `(x, y)` are present: `x` a coordinate on the source axis, `y` on the
//! target axis. It keeps both directions as row masks, so a reader takes
//! "every `y` with `(x, y)`" or "every `x` with `(x, y)`" as one row, LSB-first
//! `u64` words in mask-RISC plane order, tail bits past the axis zero.
//!
//! This is not lane presence. [`crate::PresenceSource`] gives one bit per row
//! of a lane (absent vs. zero); a pair plane gives one bit per pair of
//! coordinates. Coordinates are positions, never identities: placing an id is
//! [`super::MapTable::locate`]'s job, done before a pair is inserted.
//!
//! A plane belongs to the frame issue it was built under. Every read takes the
//! reader's frame and refuses any other issue, a newer frame of the same table
//! included: a plane built at one layout is not an answer about another. The
//! owner maintains the contents (insert on add, remove on retire, rebuild when
//! its relation is refolded under a new frame); this type only keeps the two
//! directions equal to each other's transpose.

use super::{Axis, Frame};

/// Which way a row is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairDir {
    /// Row `x` of the source axis: every target `y` with `(x, y)`.
    Forward,
    /// Row `y` of the target axis: every source `x` with `(x, y)`.
    Backward,
}

/// Why a [`PairPlane`] refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairRefusal {
    /// The reader's frame is not the issue the plane was built under.
    ForeignFrame,
    /// A coordinate is past its axis's reserved positions.
    OutOfAxis {
        /// The direction whose axis was exceeded: `Forward` for the source
        /// axis, `Backward` for the target axis.
        axis: PairDir,
        coord: usize,
        positions: usize,
    },
}

/// One relation's presence over a source axis and a target axis.
#[derive(Clone, Debug)]
pub struct PairPlane {
    frame: Frame,
    src: Axis,
    dst: Axis,
    /// `src.reserved()` rows of `fwd_words` words.
    fwd: Vec<u64>,
    /// `dst.reserved()` rows of `bwd_words` words.
    bwd: Vec<u64>,
    fwd_words: usize,
    bwd_words: usize,
    pairs: u64,
}

impl PairPlane {
    /// An empty plane over `src × dst`, owned by `frame`'s issue.
    pub fn new(frame: &Frame, src: Axis, dst: Axis) -> Self {
        let (s, d) = (src.reserved(), dst.reserved());
        let (fwd_words, bwd_words) = (d.div_ceil(64), s.div_ceil(64));
        PairPlane {
            frame: frame.clone(),
            src,
            dst,
            fwd: vec![0; s * fwd_words],
            bwd: vec![0; d * bwd_words],
            fwd_words,
            bwd_words,
            pairs: 0,
        }
    }

    /// The frame the plane was built under.
    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    pub fn src(&self) -> Axis {
        self.src
    }

    pub fn dst(&self) -> Axis {
        self.dst
    }

    /// Present pairs.
    pub fn len(&self) -> u64 {
        self.pairs
    }

    pub fn is_empty(&self) -> bool {
        self.pairs == 0
    }

    /// Words per row read in `dir`: the other axis's positions over 64.
    pub fn words_per_row(&self, dir: PairDir) -> usize {
        match dir {
            PairDir::Forward => self.fwd_words,
            PairDir::Backward => self.bwd_words,
        }
    }

    /// Rows read in `dir`: that axis's positions.
    pub fn rows_in(&self, dir: PairDir) -> usize {
        match dir {
            PairDir::Forward => self.src.reserved(),
            PairDir::Backward => self.dst.reserved(),
        }
    }

    fn check_pair(&self, x: usize, y: usize) -> Result<(), PairRefusal> {
        for (axis, coord, positions) in [
            (PairDir::Forward, x, self.src.reserved()),
            (PairDir::Backward, y, self.dst.reserved()),
        ] {
            if coord >= positions {
                return Err(PairRefusal::OutOfAxis {
                    axis,
                    coord,
                    positions,
                });
            }
        }
        Ok(())
    }

    /// Mark `(x, y)` present in both directions. Returns whether it was new.
    pub fn insert(&mut self, x: usize, y: usize) -> Result<bool, PairRefusal> {
        self.check_pair(x, y)?;
        let f = &mut self.fwd[x * self.fwd_words + y / 64];
        let was = *f >> (y % 64) & 1 == 1;
        *f |= 1 << (y % 64);
        self.bwd[y * self.bwd_words + x / 64] |= 1 << (x % 64);
        if !was {
            self.pairs += 1;
        }
        Ok(!was)
    }

    /// Mark `(x, y)` absent in both directions. Returns whether it was present.
    pub fn remove(&mut self, x: usize, y: usize) -> Result<bool, PairRefusal> {
        self.check_pair(x, y)?;
        let f = &mut self.fwd[x * self.fwd_words + y / 64];
        let was = *f >> (y % 64) & 1 == 1;
        *f &= !(1 << (y % 64));
        self.bwd[y * self.bwd_words + x / 64] &= !(1 << (x % 64));
        if was {
            self.pairs -= 1;
        }
        Ok(was)
    }

    /// Whether `(x, y)` is present, read under `frame`.
    pub fn contains(&self, frame: &Frame, x: usize, y: usize) -> Result<bool, PairRefusal> {
        self.check_frame(frame)?;
        self.check_pair(x, y)?;
        Ok(self.fwd[x * self.fwd_words + y / 64] >> (y % 64) & 1 == 1)
    }

    fn check_frame(&self, frame: &Frame) -> Result<(), PairRefusal> {
        if self.frame.same_issue(frame) {
            Ok(())
        } else {
            Err(PairRefusal::ForeignFrame)
        }
    }

    /// Row `at` read in `dir`, under `frame`. Borrows; allocates nothing.
    pub fn row(&self, frame: &Frame, dir: PairDir, at: usize) -> Result<&[u64], PairRefusal> {
        self.check_frame(frame)?;
        let positions = self.rows_in(dir);
        if at >= positions {
            return Err(PairRefusal::OutOfAxis {
                axis: dir,
                coord: at,
                positions,
            });
        }
        let w = self.words_per_row(dir);
        let plane = match dir {
            PairDir::Forward => &self.fwd,
            PairDir::Backward => &self.bwd,
        };
        Ok(&plane[at * w..(at + 1) * w])
    }

    /// Every row read in `dir`, back to back (`words_per_row(dir)` words
    /// each), under `frame`. Borrows; allocates nothing.
    pub fn rows(&self, frame: &Frame, dir: PairDir) -> Result<&[u64], PairRefusal> {
        self.check_frame(frame)?;
        Ok(match dir {
            PairDir::Forward => &self.fwd,
            PairDir::Backward => &self.bwd,
        })
    }
}
