//! The P7a certification reading (D-GSO-7a, #1369), shared by the examples
//! that read it: `relational_certification_probe` (where it is tested) and
//! `epistemic_reading_conflict_probe` (which pins its disagreement with the
//! affordance law against this implementation, not a copy).
//!
//! D-EPI-MIG-0: P7a's codes ARE the certification coordinate of the
//! canonical Cartesian `EpistemicState5` (`raw5 = topology | certification
//! << 2`). Stamping is identity packing of `(declared topology, contract)`
//! into bits 59..63 — no translation table; reading projects the canonical
//! state through the class declaration and takes its certification.
//!
//! Originally moved here unchanged. Each including example uses a subset, hence the
//! `dead_code` allowance.
#![allow(dead_code)]

use causal_edge::edge::CausalEdge64;
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::epistemic_state5::{
    Certification3, Epi5Declarations, Epi5Gen, Epi5ReadError, Epi5Reading, EpistemicState5,
    Topology2,
};
use lance_graph_contract::rail_geometry::RailAxis;

// ── The declared reading ──────────────────────────────────────────────────

/// A class whose edges carry the certification reading.
pub const CERT_CLASS: ClassId = 0x0902;
/// A class that declared a band under the historical reading. It has no
/// canonical declaration.
pub const LEGACY_CLASS: ClassId = 0x0901;
pub const RAIL: RailAxis = RailAxis::Taxonomy;
/// Classes declared under the certification reading. The bits cannot say which
/// reading wrote them, so the class does.
pub const CERTIFICATION_CLASSES: &[ClassId] = &[CERT_CLASS];
/// The grounding P7a's sealed models declare: exposure and outcome are
/// measured on the same units with no intermediate in the model, so the
/// certified relation is `Direct`. A producer statement, not an inference
/// from the bits.
pub const MODEL_GROUNDING: Topology2 = Topology2::Direct;

/// The certified relational contract. Working names; the obligations in the
/// module table are the content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Contract {
    Open,
    Associated,
    Related,
    Contributes,
    CausalCandidate,
    Causes,
}

impl Contract {
    pub const ALL: [Contract; 6] = [
        Contract::Open,
        Contract::Associated,
        Contract::Related,
        Contract::Contributes,
        Contract::CausalCandidate,
        Contract::Causes,
    ];

    pub const fn code(self) -> u8 {
        match self {
            Contract::Open => 0,
            Contract::Associated => 1,
            Contract::Related => 2,
            Contract::Contributes => 3,
            Contract::CausalCandidate => 4,
            Contract::Causes => 5,
        }
    }

    pub fn from_code(raw: u8) -> Result<Self, Refusal> {
        Ok(match raw {
            0 => Contract::Open,
            1 => Contract::Associated,
            2 => Contract::Related,
            3 => Contract::Contributes,
            4 => Contract::CausalCandidate,
            5 => Contract::Causes,
            other => return Err(Refusal::Reserved(other)),
        })
    }

    /// Does holding `self` license asserting `required`? An explicit table,
    /// never a comparison of codes.
    pub const fn entails(self, required: Contract) -> bool {
        use Contract::*;
        matches!(
            (self, required),
            (_, Open)
                | (Associated, Associated)
                | (Related, Associated | Related)
                | (Contributes, Associated | Related | Contributes)
                | (
                    CausalCandidate,
                    Associated | Related | Contributes | CausalCandidate
                )
                | (Causes, _)
        )
    }
}

impl Contract {
    /// The canonical certification coordinate (same code).
    pub fn certification(self) -> Certification3 {
        Certification3::ALL[self.code() as usize]
    }

    pub fn from_certification(c: Certification3) -> Self {
        Contract::ALL[c.code() as usize]
    }
}

/// Why a certification could not be stamped or read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// The class is not declared under the certification reading.
    NotCertificationClass,
    /// A reserved legacy code (6 or 7).
    Reserved(u8),
    /// The canonical projection refused (undeclared, untrusted provenance,
    /// undeclared code).
    Reading(Epi5ReadError),
}

/// The canonical declarations: `CERT_CLASS` reads bits 59..63 as
/// `EpistemicState5` V1. `LEGACY_CLASS` is deliberately absent.
pub fn declarations() -> Epi5Declarations {
    let mut d = Epi5Declarations::new();
    d.declare(
        CERT_CLASS,
        RAIL,
        Epi5Reading {
            generation: Epi5Gen::V1,
        },
    );
    d
}

/// Read the certified contract off an edge: canonical projection, then the
/// legacy certification projection of the state.
pub fn read(
    decl: &Epi5Declarations,
    class: ClassId,
    edge: CausalEdge64,
    provenance: EdgeProvenance,
) -> Result<Contract, Refusal> {
    if !CERTIFICATION_CLASSES.contains(&class) {
        return Err(Refusal::NotCertificationClass);
    }
    let state = decl
        .project_state5(class, RAIL, Epi5Gen::V1, edge.epistemic_raw5(), provenance)
        .map_err(Refusal::Reading)?;
    Ok(Contract::from_certification(state.certification()))
}

/// Does the edge license `required`?
pub fn satisfies(
    decl: &Epi5Declarations,
    class: ClassId,
    edge: CausalEdge64,
    required: Contract,
) -> Result<bool, Refusal> {
    Ok(read(decl, class, edge, EdgeProvenance::V2Stamped)?.entails(required))
}

/// Stamp a certification under a declared topology: identity packing into
/// all of bits 59..63. Nothing else moves. Every contract × topology is a
/// meaningful V1 state.
pub fn stamp(edge: CausalEdge64, contract: Contract, topology: Topology2) -> CausalEdge64 {
    let state = EpistemicState5::new(Epi5Gen::V1, topology, contract.certification());
    edge.with_epistemic_raw5(state.raw())
}
