//! How an XLOG result reads in `CausalEdge64` bits 59..63, the
//! `EpistemicState5` field (`Topology2 × Certification3`).
//!
//! The rule this pins, for any XLOG result that is ever written as a CE64
//! edge:
//!
//! - A base fact (`r(1, 2)`) is `Direct × Open`.
//! - A composed edge (`r ; s`, X to Z through Y) is `IndirectKnown × Open`:
//!   the register keeps the intermediate Y, so it is known.
//! - A join certifies nothing (`Open`): sharing a key is not an association
//!   in a declared population, let alone a cause.
//! - A bag count is a value in the binding (`tests/dto_binding.rs`), never a
//!   certification and never frequency/confidence.
//!
//! `raw5 == 0` is `Direct × Open`, not "unknown", so a composed edge left
//! unstamped silently claims a direct relation. The code is also opaque
//! without a declaration: the XLOG lens has none in production (an OGAR mint
//! decision), and projection refuses until it has one.

use causal_edge::edge::CausalEdge64;
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::epistemic_state5::{
    Certification3, Epi5Declarations, Epi5Gen, Epi5ReadError, Epi5Reading, EpistemicState5,
    Topology2, fact,
};
use lance_graph_contract::rail_geometry::RailAxis;

/// The XLOG lens's view class (lab, as in `tests/dto_binding.rs`).
const XLOG_VIEW: ClassId = 0x7F01;
/// Lab placeholder: which rail an XLOG edge's reading is declared on is part
/// of the OGAR declaration this probe does not make.
const RAIL: RailAxis = RailAxis::Taxonomy;

const BASE: EpistemicState5 =
    EpistemicState5::new(Epi5Gen::V1, Topology2::Direct, Certification3::Open);
const COMPOSED: EpistemicState5 =
    EpistemicState5::new(Epi5Gen::V1, Topology2::IndirectKnown, Certification3::Open);

/// S=0x56, P=0x34, O=0x12, frequency 0xCD, confidence 0xAB; upper half zero.
const EDGE: CausalEdge64 = CausalEdge64(0x0000_00AB_CD12_3456);
const LOW_40: u64 = (1 << 40) - 1;

fn declared() -> Epi5Declarations {
    let mut d = Epi5Declarations::new();
    d.declare(XLOG_VIEW, RAIL, Epi5Reading::default());
    d
}

fn read(d: &Epi5Declarations, e: CausalEdge64) -> Result<EpistemicState5, Epi5ReadError> {
    d.project_state5(
        XLOG_VIEW,
        RAIL,
        Epi5Gen::V1,
        e.epistemic_raw5(),
        EdgeProvenance::V2Stamped,
    )
}

#[test]
fn an_xlog_edge_has_no_reading_until_its_lens_is_declared() {
    let e = EDGE.with_epistemic_raw5(COMPOSED.raw());
    assert_eq!(
        read(&Epi5Declarations::new(), e),
        Err(Epi5ReadError::UndeclaredClass(XLOG_VIEW))
    );
}

#[test]
fn base_facts_and_composed_edges_carry_different_topologies() {
    let d = declared();
    let base = EDGE.with_epistemic_raw5(BASE.raw());
    let composed = EDGE.with_epistemic_raw5(COMPOSED.raw());
    assert_eq!((BASE.raw(), COMPOSED.raw()), (0, 1));
    assert_eq!(read(&d, base), Ok(BASE));
    assert_eq!(read(&d, composed), Ok(COMPOSED));
    assert!(COMPOSED.asserts(fact::INTERMEDIATE_KNOWN));
    assert!(
        !COMPOSED.asserts(fact::ASSOCIATED),
        "a join certifies nothing"
    );
    // Only bits 59..63 moved: palette indices and F/C are untouched.
    assert_eq!(composed.0 & LOW_40, EDGE.0 & LOW_40);
    assert_eq!(composed.0 & !(0b1_1111 << 59), EDGE.0);
}

/// The pitfall: a composed edge written without its state reads as a direct
/// one.
#[test]
fn an_unstamped_composed_edge_reads_as_direct() {
    let unstamped = read(&declared(), EDGE).unwrap();
    assert_eq!(unstamped, BASE);
    assert_ne!(unstamped, COMPOSED);
}

#[test]
fn legacy_provenance_is_refused() {
    let e = EDGE.with_epistemic_raw5(COMPOSED.raw());
    assert_eq!(
        declared().project_state5(
            XLOG_VIEW,
            RAIL,
            Epi5Gen::V1,
            e.epistemic_raw5(),
            EdgeProvenance::V1Legacy,
        ),
        Err(Epi5ReadError::UnknownProvenance(EdgeProvenance::V1Legacy))
    );
}
