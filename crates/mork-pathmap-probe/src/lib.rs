//! MORK / PathMap differential probe behind one Lance-owned boundary.
//!
//! STATUS: PROBE (results only). Not a production executor, not wired into
//! Quack or R2IL, and not a fact store. The native arm ([`join`]) runs its
//! intersection as a mask-risc program, as a probe only. Plan:
//! `.claude/plans/mork-pathmap-schnellverschluss-v1.md`.
//!
//! The query is fixed: `Q(x, y, z) :- R(x, y), S(y, z)` — two relations that
//! share one variable. Every arm answers the same question; [`reference`] is the
//! independent oracle that names no upstream type.
//!
//! Ownership, as the module layout states it:
//!
//! - [`RelationView`] is the Lance side: a relation as two lanes (Arrow
//!   buffers, shared by reference count, never copied) plus the generation of
//!   the sealed view they came from. A row is the template over the lanes at
//!   one position; it is never held as a pair.
//! - [`join`] is the native arm: an index map per side and folds over it.
//!   It builds no row (`"{0}{1}" -f $laneA[$i],$laneB[$i]`).
//! - [`boundary`] is the only module that names `mork` or `pathmap`. Replacing
//!   or removing a backend touches that module and `Cargo.toml`, nothing else.
//!   Those two engines store their own encoded copies (a trie of paths) and
//!   answer in rows ([`Answers`]); that is their representation, read only to
//!   compare them against the native folds. A MORK `Expr` is a raw pointer
//!   with no lifetime, so it never leaves the boundary module.

use std::collections::BTreeMap;

pub mod boundary;
pub mod chain;
pub mod join;
pub mod reference;
pub mod rmap;
pub mod tile_gemm;

pub use boundary::{Backend, BoundIndex, Refusal, bind, capability};

/// One column of a relation: Arrow's reference-counted buffer. Cloning a lane
/// shares its bytes; a lane read from Lance is the column buffer itself.
pub type Lane = arrow_buffer::ScalarBuffer<u32>;

/// One binary relation as two lanes and the generation of the sealed view
/// they were read from. Row `i` is `(cols[0][i], cols[1][i])`, read in place
/// at `i`; it is never held as a pair. Row order and duplicates are the
/// caller's: a bag.
#[derive(Clone, Debug)]
pub struct RelationView {
    pub cols: [Lane; 2],
    pub generation: u64,
}

impl RelationView {
    pub fn new(c0: impl Into<Lane>, c1: impl Into<Lane>, generation: u64) -> Self {
        let cols = [c0.into(), c1.into()];
        assert_eq!(
            cols[0].len(),
            cols[1].len(),
            "a relation's lanes have one length"
        );
        RelationView { cols, generation }
    }

    /// Builds the two lanes from authored fixture rows, once. For tests and
    /// examples that write their data as literals; a stored relation arrives
    /// as lanes.
    pub fn from_rows(rows: &[[u32; 2]], generation: u64) -> Self {
        let c0: Vec<u32> = rows.iter().map(|r| r[0]).collect();
        let c1: Vec<u32> = rows.iter().map(|r| r[1]).collect();
        RelationView::new(c0, c1, generation)
    }

    pub fn len(&self) -> usize {
        self.cols[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.cols[0].is_empty()
    }
}

/// The two inputs of `Q(x, y, z) :- R(x, y), S(y, z)`.
#[derive(Clone, Debug)]
pub struct JoinInput {
    pub r: RelationView,
    pub s: RelationView,
}

impl JoinInput {
    pub fn generations(&self) -> (u64, u64) {
        (self.r.generation, self.s.generation)
    }
}

/// One answer row `[x, y, z]`.
pub type Row = [u32; 3];

/// Whether duplicate derivations are counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Semantics {
    /// Each distinct answer once.
    Set,
    /// Each answer as many times as it is derived (`|R row| × |S row|` pairs).
    Bag,
}

/// The upstream engines' answers, in their own representation (rows): what
/// MORK and PathMap return, kept only to compare them against the native
/// folds. The native arm never builds one. Under [`Semantics::Set`] every
/// count is 1.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Answers {
    pub rows: BTreeMap<Row, u32>,
}

impl Answers {
    pub fn to_set(&self) -> Answers {
        Answers {
            rows: self.rows.keys().map(|r| (*r, 1)).collect(),
        }
    }

    pub fn total(&self) -> u64 {
        self.rows.values().map(|&c| u64::from(c)).sum()
    }
}
