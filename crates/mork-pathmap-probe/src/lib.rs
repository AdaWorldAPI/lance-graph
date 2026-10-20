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
//! - [`RelationView`] is the Lance side: borrowed rows plus the generation of the
//!   sealed view they came from. It is the probe's stand-in for a sealed
//!   `RelationOrder` view, which does not exist in code yet (D-RMAP is PROPOSED).
//! - [`boundary`] is the only module that names `mork` or `pathmap`. Replacing
//!   or removing a backend touches that module and `Cargo.toml`, nothing else.
//! - No upstream type crosses the boundary in either direction: callers pass
//!   `[u32; 2]` rows in and get `[u32; 3]` rows out. A MORK `Expr` is a raw
//!   pointer with no lifetime, so it never leaves the boundary module.

use std::collections::BTreeMap;

pub mod boundary;
pub mod join;
pub mod reference;
pub mod rmap;

pub use boundary::{Backend, BoundIndex, Refusal, bind, capability};

/// Borrowed rows of one binary relation and the generation of the sealed view
/// they were read from. Row order and duplicates are the caller's: a bag.
#[derive(Clone, Copy, Debug)]
pub struct RelationView<'a> {
    pub rows: &'a [[u32; 2]],
    pub generation: u64,
}

/// The two inputs of `Q(x, y, z) :- R(x, y), S(y, z)`.
#[derive(Clone, Copy, Debug)]
pub struct JoinInput<'a> {
    pub r: RelationView<'a>,
    pub s: RelationView<'a>,
}

impl JoinInput<'_> {
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

/// Answers with their multiplicities. Under [`Semantics::Set`] every count is 1.
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
