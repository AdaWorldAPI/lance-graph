//! `lance-graph-nal` — NAL over the fold path (D-NALP-1/2,
//! `.claude/plans/nal-polyfill-v1.md`).
//!
//! # The NodeGuid is the address and the distinguished name
//!
//! A belief's key is its [`NodeGuid`]: 16 bytes, `classid(4) | 6 × (8:8)`, read
//! here as the reading `nal-statement-v1`,
//!
//! ```text
//! semantic tiles:  [classid hi, classid lo,  S,  copula,  P,  id,  0,  0]
//! ```
//!
//! The key names the belief and nothing else. Its truth value is not part of
//! its name: `f` and `c` are value lanes beside the key, in the key's sealed
//! order.
//!
//! # A selection is the key minus an aperture
//!
//! A query names its rows with an [`Aperture`]: a key and a count `n` of
//! leading bytes that must match; the remaining `16 − n` bytes are free. On the
//! sealed, witnessed order (HHTL Cartesian: classid, then each tile coarse to
//! fine) the matches are one contiguous range, which
//! [`SealedFacetLane::bound_aperture`] finds with two `partition_point`s.
//!
//! - `n = 10`: one statement `<S copula P>` (classid 4 + S 2 + copula 2 + P 2);
//! - `n = 6`: every belief about subject `S`;
//! - `n = 4`: the whole statement class.
//!
//! A query runs over its range as the execution extent; a lane predicate (a
//! confidence threshold) reads only the rows of that range.
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
//! Revision (W3), the per-row truth laws (W4), syllogisms (W5), the Narsese
//! text front end (W6), and an argmax fold (which belief holds the maximum).
//! The term width is `u16` per tile: a vocabulary larger than 65,536 terms
//! scales through the classid prefix, not by widening the tile.
//!
//! OPEN: R2IL `RANGE` takes its bounds from two `u8` immediates (`ogar-r2il`
//! arity 0), so it cannot name a range past row 255. The procedures emit the
//! bound through [`lance_graph_r2il::FoldDialect::emit_pred`] instead.

#![forbid(unsafe_code)]

pub mod nal;

use lance_graph_contract::canonical_node::NodeGuid;
use lance_graph_contract::facet::{FacetCascade, SemanticAperture};
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

/// One belief: its name (statement and identity) and its truth value in `u8`
/// codes (`f`, `c` scaled to `0..=255`, the codes `causal_edge::isa::truth`
/// reads).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Belief {
    /// Subject term address.
    pub s: u16,
    /// Copula.
    pub copula: Copula,
    /// Predicate term address.
    pub p: u16,
    /// Identity of this belief among the beliefs of its statement.
    pub id: u16,
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

/// The NodeGuid of a belief of class `class`.
pub fn guid(class: u32, s: u16, copula: Copula, p: u16, id: u16) -> NodeGuid {
    FacetCascade::from_semantic_tiles([
        (class >> 16) as u16,
        class as u16,
        s,
        copula as u16,
        p,
        id,
        0,
        0,
    ])
    .into()
}

/// A selection: `key` with only its first `bytes` bytes (in semantic order:
/// classid, then tiles coarse to fine) cared, the rest free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Aperture {
    /// The key the cared bytes are read from.
    pub key: NodeGuid,
    /// Leading bytes that must match, `0..=16`.
    pub bytes: u8,
}

impl Aperture {
    /// One statement of class `class` (10 bytes cared).
    pub fn statement(class: u32, st: Statement) -> Self {
        Aperture {
            key: guid(class, st.s, st.copula, st.p, 0),
            bytes: 10,
        }
    }

    /// Every belief about subject `s` in class `class` (6 bytes cared).
    pub fn subject(class: u32, s: u16) -> Self {
        Aperture {
            key: guid(class, s, Copula::Inheritance, 0, 0),
            bytes: 6,
        }
    }

    /// The contract's aperture: care = the leading `bytes` bytes.
    pub fn semantic(self) -> SemanticAperture {
        let bits = 8 * u32::from(self.bytes.min(16));
        let care: u128 = if bits == 0 {
            0
        } else {
            u128::MAX << (128 - bits)
        };
        let tiles = core::array::from_fn(|i| (care >> (112 - 16 * i)) as u16);
        SemanticAperture::new(self.key.facet(), FacetCascade::from_semantic_tiles(tiles))
    }
}

/// A sealed belief population: the ordered NodeGuid lane and the truth lanes
/// in the same order.
#[derive(Debug, Clone)]
pub struct Beliefs {
    class: u32,
    lane: SealedFacetLane,
    pub(crate) f: Vec<i32>,
    pub(crate) c: Vec<i32>,
}

impl Beliefs {
    /// Seal `beliefs` under the statement class `class` at `version`. The
    /// beliefs are sorted by key here, once, and the truth lanes are written in
    /// that order, so key row `i` and truth row `i` are one belief by
    /// construction.
    ///
    /// # Errors
    ///
    /// [`WitnessError::TooManyRows`] past `u32::MAX` rows.
    pub fn seal(
        class: u32,
        beliefs: &[Belief],
        version: LanceVersion,
    ) -> Result<Self, WitnessError> {
        let mut rows: Vec<(FacetCascade, i32, i32)> = beliefs
            .iter()
            .map(|b| {
                (
                    guid(class, b.s, b.copula, b.p, b.id).facet(),
                    i32::from(b.f),
                    i32::from(b.c),
                )
            })
            .collect();
        rows.sort_by(|a, b| a.0.cmp_numeric_projection(&b.0));
        let f = rows.iter().map(|r| r.1).collect();
        let c = rows.iter().map(|r| r.2).collect();
        let lane =
            SealedFacetLane::attest_sorted(rows.into_iter().map(|r| r.0).collect(), version)?;
        Ok(Beliefs { class, lane, f, c })
    }

    /// The statement class.
    pub fn class(&self) -> u32 {
        self.class
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
        Ok(Bound {
            beliefs: self,
            witness: *witness,
        })
    }
}

/// A belief population whose ordering witness has been validated.
#[derive(Debug, Clone, Copy)]
pub struct Bound<'a> {
    pub(crate) beliefs: &'a Beliefs,
    witness: OrderedLaneWitness,
}

/// Lane index of `c` in the planes every NAL program runs over (`f` is lane 0).
pub(crate) const LANE_C: u16 = 1;

impl Bound<'_> {
    /// Row count.
    pub fn n_rows(&self) -> usize {
        self.beliefs.c.len()
    }

    /// The statement class.
    pub fn class(&self) -> u32 {
        self.beliefs.class
    }

    /// The rows `[lo, hi)` an aperture selects.
    ///
    /// # Errors
    ///
    /// [`NalError::Witness`] if the witness no longer validates;
    /// [`NalError::NotAnInterval`] if the aperture's care has a hole (it
    /// cannot: [`Aperture`] cares a leading run of bytes).
    pub fn range(&self, a: Aperture) -> Result<(u32, u32), NalError> {
        self.beliefs
            .lane
            .bound_aperture(&self.witness, &a.semantic())?
            .ok_or(NalError::NotAnInterval)
    }

    /// Run a kept program over `extent`.
    pub(crate) fn run(
        &self,
        program: &Program,
        scratch: &mut lance_graph_mask_risc::Scratch<'_>,
        extent: std::ops::Range<usize>,
    ) -> Result<lance_graph_mask_risc::Value, ExecError> {
        let lanes = [LaneRef::I32(&self.beliefs.f), LaneRef::I32(&self.beliefs.c)];
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
    /// The aperture is not one interval of the sealed order.
    NotAnInterval,
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
