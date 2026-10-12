//! `lance-graph-nal` — NAL over the fold path (D-NALP-1/2,
//! `.claude/plans/nal-polyfill-v1.md`).
//!
//! # Statements are coordinates
//!
//! A belief is not a row with foreign keys. It is an address: the reading
//! `nal-statement-v1` places it on one [`FacetCascade`] key,
//!
//! ```text
//! semantic tiles:  [classid hi, classid lo,  S,  copula,  P,  255 − c,  255 − f,  0]
//! ```
//!
//! and the population is a [`SealedFacetLane`] of those keys, sorted at seal
//! and attested by an [`OrderedLaneWitness`]. Every query below is therefore
//! a coordinate range, found with two `partition_point`s:
//!
//! - `<S copula P>?` is the prefix of depth 5;
//! - "every belief about S" is the prefix of depth 3;
//! - `c ≥ t` within a statement is the start of that statement's range up to
//!   the key with tile 5 = `255 − t`, because confidence is stored inverted
//!   and the range is ordered by it;
//! - the highest-confidence belief of a statement (NAL choice) is the first
//!   row of its range.
//!
//! The confidence lane `c` handed to mask-risc is decoded from the sealed keys
//! themselves, so it is in the sealed order by construction: the obligation
//! `ISS-WITNESSED-RANGE-DOES-NOT-ATTEST-PLANE-ORDER` names cannot be broken
//! here.
//!
//! # Operations are procedures, compiled once
//!
//! Each [`nal`] operation is an R2IL procedure ([`nal::NalOp::lower`]) over
//! [`lance_graph_r2il::FoldDialect`]. Binding a query runs that procedure once
//! and keeps the mask-risc [`Program`] it produced; every later call runs the
//! kept program through [`execute_compiled`]. A call does not parse, lower or
//! allocate.
//!
//! # Not here yet
//!
//! Revision (W3), the per-row truth laws (W4), syllogisms (W5) and the
//! Narsese text front end (W6). The term width is `u16` per tile: a
//! vocabulary larger than 65,536 terms scales through the classid prefix, not
//! by widening the tile.
//!
//! OPEN: R2IL `RANGE` takes its bounds from two `u8` immediates (`ogar-r2il`
//! arity 0), so it cannot name a range past row 255. The procedures emit the
//! bound through [`lance_graph_r2il::FoldDialect::emit_pred`] instead.

#![forbid(unsafe_code)]

pub mod nal;

use lance_graph_contract::facet::FacetCascade;
use lance_graph_contract::ordered_lane::{OrderedLaneWitness, SealedFacetLane, WitnessError};
use lance_graph_contract::temporal_pov::LanceVersion;
use lance_graph_mask_risc::{execute_compiled, ExecError, LaneRef, Program};

/// The NAL copulas this reading stores. NAL-1 and NAL-2 (inheritance,
/// similarity) and NAL-5 (implication, equivalence) statements over atomic
/// terms; compound terms are out of scope (ledger R10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u16)]
pub enum Copula {
    /// `-->`
    Inheritance = 1,
    /// `<->`
    Similarity = 2,
    /// `==>`
    Implication = 3,
    /// `<=>`
    Equivalence = 4,
}

impl Copula {
    /// Every copula, in code order.
    pub const ALL: [Copula; 4] = [
        Copula::Inheritance,
        Copula::Similarity,
        Copula::Implication,
        Copula::Equivalence,
    ];

    fn from_code(c: u16) -> Option<Self> {
        Self::ALL.into_iter().find(|k| *k as u16 == c)
    }
}

/// A statement `<S copula P>`: two term addresses and a copula.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Statement {
    /// Subject term address.
    pub s: u16,
    /// Copula.
    pub copula: Copula,
    /// Predicate term address.
    pub p: u16,
}

/// One belief: a statement with a truth value in `u8` codes (`f`, `c` scaled
/// to `0..=255`, the codes `causal_edge::isa::truth` reads).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Belief {
    /// Subject term address.
    pub s: u16,
    /// Copula.
    pub copula: Copula,
    /// Predicate term address.
    pub p: u16,
    /// Frequency code.
    pub f: u8,
    /// Confidence code.
    pub c: u8,
}

impl Belief {
    /// The statement this belief is about.
    pub fn statement(&self) -> Statement {
        Statement {
            s: self.s,
            copula: self.copula,
            p: self.p,
        }
    }
}

/// Semantic tile positions of the `nal-statement-v1` reading.
const T_S: usize = 2;
const T_COPULA: usize = 3;
const T_P: usize = 4;
const T_INV_C: usize = 5;
const T_INV_F: usize = 6;

fn tiles(class: u32, s: u16, copula: u16, p: u16, inv_c: u16, inv_f: u16, last: u16) -> [u16; 8] {
    [
        (class >> 16) as u16,
        class as u16,
        s,
        copula,
        p,
        inv_c,
        inv_f,
        last,
    ]
}

fn key_of(class: u32, b: &Belief) -> FacetCascade {
    FacetCascade::from_semantic_tiles(tiles(
        class,
        b.s,
        b.copula as u16,
        b.p,
        u16::from(255 - b.c),
        u16::from(255 - b.f),
        0,
    ))
}

fn belief_of(k: &FacetCascade) -> Option<Belief> {
    let t = k.semantic_tiles();
    Some(Belief {
        s: t[T_S],
        copula: Copula::from_code(t[T_COPULA])?,
        p: t[T_P],
        c: 255 - u8::try_from(t[T_INV_C]).ok()?,
        f: 255 - u8::try_from(t[T_INV_F]).ok()?,
    })
}

/// A sealed belief population: the ordered key lane and the confidence lane
/// decoded from it.
#[derive(Debug, Clone)]
pub struct Beliefs {
    class: u32,
    lane: SealedFacetLane,
    pub(crate) c: Vec<i32>,
}

impl Beliefs {
    /// Seal `beliefs` under the statement class `class` at `version`. Sorting
    /// happens here, once; every read after it is a fold.
    ///
    /// # Errors
    ///
    /// [`WitnessError::TooManyRows`] past `u32::MAX` rows.
    pub fn seal(
        class: u32,
        beliefs: &[Belief],
        version: LanceVersion,
    ) -> Result<Self, WitnessError> {
        let lane =
            SealedFacetLane::seal(beliefs.iter().map(|b| key_of(class, b)).collect(), version)?;
        let c = lane
            .keys()
            .iter()
            .map(|k| 255 - i32::from(k.semantic_tiles()[T_INV_C]))
            .collect();
        Ok(Beliefs { class, lane, c })
    }

    /// The witness the lane was sealed with. A planner holds it and hands it
    /// back to [`Beliefs::bind`].
    pub fn witness(&self) -> OrderedLaneWitness {
        self.lane.witness()
    }

    /// Validate `witness` against this lane (O(1)) and return the bound view
    /// the operations run over.
    ///
    /// # Errors
    ///
    /// [`NalError::Witness`] when the witness is from another seal.
    pub fn bind(&self, witness: &OrderedLaneWitness) -> Result<Bound<'_>, NalError> {
        self.lane.validate(witness)?;
        Ok(Bound { beliefs: self })
    }
}

/// A belief population whose ordering witness has been validated.
#[derive(Debug, Clone, Copy)]
pub struct Bound<'a> {
    pub(crate) beliefs: &'a Beliefs,
}

/// Lane index of the confidence lane in the planes every NAL program runs over.
pub(crate) const LANE_C: u16 = 0;

impl<'a> Bound<'a> {
    /// Row count.
    pub fn n_rows(&self) -> usize {
        self.beliefs.c.len()
    }

    fn keys(&self) -> &'a [FacetCascade] {
        self.beliefs.lane.keys()
    }

    /// The belief stored at sealed row `row`.
    pub fn belief(&self, row: usize) -> Option<Belief> {
        self.keys().get(row).and_then(belief_of)
    }

    /// `[lo, hi)` of every key in the closed interval `[lo_t, hi_t]` of
    /// semantic tiles: two `partition_point`s on the witnessed order.
    pub(crate) fn interval(&self, lo_t: [u16; 8], hi_t: [u16; 8]) -> (u32, u32) {
        let (lo_k, hi_k) = (
            FacetCascade::from_semantic_tiles(lo_t),
            FacetCascade::from_semantic_tiles(hi_t),
        );
        let keys = self.keys();
        let lo = keys.partition_point(|k| k.cmp_numeric_projection(&lo_k).is_lt());
        let hi = keys.partition_point(|k| k.cmp_numeric_projection(&hi_k).is_le());
        (lo as u32, hi.max(lo) as u32)
    }

    /// The rows of one statement.
    pub(crate) fn statement(&self, st: Statement) -> (u32, u32) {
        self.statement_at_least(st, 0)
    }

    /// The rows of one statement with confidence `>= t`.
    pub(crate) fn statement_at_least(&self, st: Statement, t: u8) -> (u32, u32) {
        let class = self.beliefs.class;
        let cop = st.copula as u16;
        self.interval(
            tiles(class, st.s, cop, st.p, 0, 0, 0),
            tiles(class, st.s, cop, st.p, u16::from(255 - t), 0xFFFF, 0xFFFF),
        )
    }

    /// The rows of every belief about subject `s`.
    pub(crate) fn subject(&self, s: u16) -> (u32, u32) {
        let class = self.beliefs.class;
        self.interval(
            tiles(class, s, 0, 0, 0, 0, 0),
            tiles(class, s, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF),
        )
    }

    /// Run a kept program over this population.
    pub(crate) fn run(
        &self,
        program: &Program,
        scratch: &mut lance_graph_mask_risc::Scratch<'_>,
        extent: std::ops::Range<usize>,
    ) -> Result<lance_graph_mask_risc::Value, ExecError> {
        let lanes = [LaneRef::I32(&self.beliefs.c)];
        let planes = lance_graph_mask_risc::Planes {
            n_rows: self.n_rows(),
            masks: &[],
            lanes: &lanes,
        };
        let foreign = lance_graph_mask_risc::Foreign {
            planes: &[],
            lanes: &[],
        };
        execute_compiled(
            &program.compile(),
            &planes,
            &foreign,
            scratch,
            lance_graph_mask_risc::Out::None,
            extent,
        )
    }
}

/// Why a NAL operation did not answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NalError {
    /// The ordering witness does not belong to this lane.
    Witness(WitnessError),
    /// The procedure was refused while binding.
    Fold(lance_graph_r2il::FoldError),
    /// mask-risc refused the kept program.
    Exec(ExecError),
    /// The procedure ended without the one scalar it promised.
    NoAnswer,
}

impl From<WitnessError> for NalError {
    fn from(e: WitnessError) -> Self {
        NalError::Witness(e)
    }
}

impl From<lance_graph_r2il::FoldError> for NalError {
    fn from(e: lance_graph_r2il::FoldError) -> Self {
        NalError::Fold(e)
    }
}

impl From<ExecError> for NalError {
    fn from(e: ExecError) -> Self {
        NalError::Exec(e)
    }
}
