//! The P7a certification reading (D-GSO-7a, #1369), shared by the examples
//! that read it: `relational_certification_probe` (where it is tested) and
//! `epistemic_reading_conflict_probe` (which pins its disagreement with the
//! affordance law against this implementation, not a copy).
//!
//! Moved here unchanged. Each including example uses a subset, hence the
//! `dead_code` allowance.
#![allow(dead_code)]

use causal_edge::edge::CausalEdge64;
use causal_edge::layout::ReasoningBand;
use lance_graph_contract::band_reading::{
    BandDeclarations, BandPresence, BandReadError, BandReading, EdgeProvenance,
};
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::rail_geometry::RailAxis;

// ── The declared reading ──────────────────────────────────────────────────

/// A class whose edges carry the certification reading.
pub const CERT_CLASS: ClassId = 0x0902;
/// A class that declares a band under the historical reading.
pub const LEGACY_CLASS: ClassId = 0x0901;
pub const RAIL: RailAxis = RailAxis::Taxonomy;
/// Classes declared under the certification reading. The bits cannot say which
/// reading wrote them, so the class does.
pub const CERTIFICATION_CLASSES: &[ClassId] = &[CERT_CLASS];

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

/// Why a band could not be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// The class is not declared under the certification reading.
    NotCertificationClass,
    /// A reserved code (6 or 7).
    Reserved(u8),
    /// The contract refused (undeclared, band absent, untrusted provenance).
    Band(BandReadError),
}

pub fn declarations() -> BandDeclarations {
    let mut d = BandDeclarations::new();
    for class in [CERT_CLASS, LEGACY_CLASS] {
        d.declare(
            class,
            RAIL,
            BandReading {
                band: BandPresence::Present,
                ..BandReading::ZERO_FALLBACK
            },
        );
    }
    d
}

/// Read the certified contract off an edge.
pub fn read(
    decl: &BandDeclarations,
    class: ClassId,
    edge: CausalEdge64,
    provenance: EdgeProvenance,
) -> Result<Contract, Refusal> {
    if !CERTIFICATION_CLASSES.contains(&class) {
        return Err(Refusal::NotCertificationClass);
    }
    let raw = decl
        .project_band(class, RAIL, edge.reasoning_band().to_bits_3(), provenance)
        .map_err(Refusal::Band)?;
    Contract::from_code(raw)
}

/// Does the band on the edge license `required`? Reserved codes refuse.
pub fn satisfies(
    decl: &BandDeclarations,
    class: ClassId,
    edge: CausalEdge64,
    required: Contract,
) -> Result<bool, Refusal> {
    Ok(read(decl, class, edge, EdgeProvenance::V2Stamped)?.entails(required))
}

/// Write a contract into bits 61..63. Only those bits move.
pub fn stamp(edge: CausalEdge64, contract: Contract) -> CausalEdge64 {
    edge.with_reasoning_band(ReasoningBand::from_bits_3(contract.code()))
}
