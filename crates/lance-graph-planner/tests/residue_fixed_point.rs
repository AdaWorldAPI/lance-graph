//! Residue fixed point: computation alone cannot lower the residue; only new
//! evidence can.
//!
//! The claim under test. The reasoning loop is cheap (`pearl::hydrate` runs
//! per edge over the sealed chain), so it runs eagerly until it reaches a
//! fixed point. What is still open after that is RESIDUE: work the substrate
//! cannot do with the evidence it has. Two things must then hold:
//!
//! 1. More passes over the same evidence change nothing, bit for bit.
//!    If they did, the residue was holding computation the loop should
//!    already have done.
//! 2. The residue falls only when evidence arrives that it did not have
//!    before. Repeating evidence it already had does not count.
//!
//! The residue measured here is the hydration debt in bits 59..63: edges
//! whose topology is still `IndirectUnknown` (an intermediate on the path is
//! not yet known). The surprise level proposed for bits 50..52 is not in the
//! layout yet (bits 50..52 are still plasticity), so it is not tested here.
//!
//! The loop proposes candidate intermediates from the chain itself: for an
//! edge `a → y`, every `b` with a sealed step `a → b` is tried. Proposing a
//! candidate is not evidence; `hydrate` promotes only when both `a → b` and
//! `b → y` are sealed.

use causal_edge::edge::InferenceType;
use causal_edge::{CausalEdge64, CausalMask, PlasticityState};
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::epistemic_state5::{
    Certification3, Epi5Declarations, Epi5Gen, Epi5Reading, EpistemicState5, Topology2,
};
use lance_graph_contract::rail_geometry::RailAxis;
use lance_graph_planner::chain_replay::ChainStep;
use lance_graph_planner::pearl::{hydrate, Reading};

const CLASS: ClassId = 0x0902;
const RAIL: RailAxis = RailAxis::Taxonomy;
const PREDICATE: u8 = 0x91;

/// Bits 61..63: the certification coordinate.
const CERT_BITS: u64 = 0b111 << 61;

fn declarations() -> Epi5Declarations {
    let mut d = Epi5Declarations::new();
    d.declare(
        CLASS,
        RAIL,
        Epi5Reading {
            generation: Epi5Gen::V1,
        },
    );
    d
}

fn reading(d: &Epi5Declarations) -> Reading<'_> {
    Reading {
        declarations: d,
        class: CLASS,
        rail: RAIL,
        provenance: EdgeProvenance::V2Stamped,
    }
}

fn raw_edge(s: u8, o: u8) -> CausalEdge64 {
    CausalEdge64::pack(
        s,
        PREDICATE,
        o,
        200,
        200,
        CausalMask::SO,
        causal_edge::Orientation::from_code(0b101),
        InferenceType::Deduction,
        PlasticityState::S_HOT,
        0,
    )
}

/// An open relation `s → o`: an intermediate is claimed but not known.
fn open_relation(s: u8, o: u8, c: Certification3) -> CausalEdge64 {
    let state = EpistemicState5::new(Epi5Gen::V1, Topology2::IndirectUnknown, c);
    raw_edge(s, o).with_epistemic_raw5(state.raw())
}

/// A sealed binding `s → o`.
fn step(s: u8, o: u8) -> ChainStep {
    (0, raw_edge(s, o))
}

fn topology(e: CausalEdge64) -> Topology2 {
    EpistemicState5::decode(Epi5Gen::V1, e.epistemic_raw5())
        .expect("valid code")
        .topology()
}

/// One pass of the loop over the population: every still-open edge tries each
/// candidate intermediate the chain proposes.
fn pass(edges: &mut [CausalEdge64], sealed: &[ChainStep], d: &Epi5Declarations) {
    for e in edges.iter_mut() {
        if topology(*e) != Topology2::IndirectUnknown {
            continue;
        }
        let (a, y) = (e.s_idx(), e.o_idx());
        let candidates: Vec<u8> = sealed
            .iter()
            .filter(|(_, s)| s.s_idx() == a)
            .map(|(_, s)| s.o_idx())
            .collect();
        for b in candidates {
            let (next, _) = hydrate(*e, (a, b, y), sealed, reading(d)).expect("declared reading");
            *e = next;
        }
    }
}

/// Residue: edges whose intermediate is still unknown.
fn residue(edges: &[CausalEdge64]) -> usize {
    edges
        .iter()
        .filter(|e| topology(**e) == Topology2::IndirectUnknown)
        .count()
}

fn bits(edges: &[CausalEdge64]) -> Vec<u64> {
    edges.iter().map(|e| e.0).collect()
}

/// Population of three open relations, each with different evidence:
/// - `1 → 3`: both bindings sealed (`1 → 2`, `2 → 3`), resolvable now.
/// - `4 → 6`: only `4 → 5` sealed, waiting on `5 → 6`.
/// - `7 → 9`: nothing sealed.
fn population() -> Vec<CausalEdge64> {
    vec![
        open_relation(1, 3, Certification3::Related),
        open_relation(4, 6, Certification3::Supports),
        open_relation(7, 9, Certification3::Associated),
    ]
}

fn initial_chain() -> Vec<ChainStep> {
    vec![step(1, 2), step(2, 3), step(4, 5)]
}

/// Run passes until nothing changes; return how many passes moved anything.
fn run_to_fixed_point(
    edges: &mut [CausalEdge64],
    sealed: &[ChainStep],
    d: &Epi5Declarations,
) -> usize {
    for moving in 0..16 {
        let before = bits(edges);
        pass(edges, sealed, d);
        if bits(edges) == before {
            return moving;
        }
    }
    panic!("the loop did not reach a fixed point in 16 passes");
}

#[test]
fn computation_reaches_a_fixed_point_and_more_passes_change_nothing() {
    let d = declarations();
    let mut edges = population();
    let sealed = initial_chain();

    assert_eq!(
        residue(&edges),
        3,
        "fixture: all three relations start open"
    );

    let moving = run_to_fixed_point(&mut edges, &sealed, &d);
    assert_eq!(moving, 1, "one pass absorbs everything the evidence allows");
    assert_eq!(residue(&edges), 2, "only 1 → 3 had both bindings");
    assert_eq!(topology(edges[0]), Topology2::IndirectKnown);

    // The fixed point holds bit for bit under many more passes.
    let at_fixed_point = bits(&edges);
    for _ in 0..32 {
        pass(&mut edges, &sealed, &d);
    }
    assert_eq!(
        bits(&edges),
        at_fixed_point,
        "extra passes must not move any bit"
    );
}

#[test]
fn repeated_evidence_does_not_lower_the_residue() {
    let d = declarations();
    let mut edges = population();
    let mut sealed = initial_chain();
    run_to_fixed_point(&mut edges, &sealed, &d);
    let at_fixed_point = bits(&edges);

    // Seal the same bindings again, many times. Nothing new is known.
    for _ in 0..8 {
        sealed.push(step(4, 5));
        sealed.push(step(1, 2));
    }
    let moving = run_to_fixed_point(&mut edges, &sealed, &d);

    assert_eq!(moving, 0, "repetition is not evidence");
    assert_eq!(residue(&edges), 2);
    assert_eq!(bits(&edges), at_fixed_point);
}

#[test]
fn only_new_evidence_lowers_the_residue_and_only_where_it_lands() {
    let d = declarations();
    let mut edges = population();
    let mut sealed = initial_chain();
    run_to_fixed_point(&mut edges, &sealed, &d);
    let before = bits(&edges);

    // Evidence that touches no open relation changes nothing.
    sealed.push(step(40, 41));
    assert_eq!(run_to_fixed_point(&mut edges, &sealed, &d), 0);
    assert_eq!(bits(&edges), before);

    // The missing binding for 4 → 6 arrives.
    sealed.push(step(5, 6));
    let moving = run_to_fixed_point(&mut edges, &sealed, &d);

    assert_eq!(moving, 1);
    assert_eq!(
        residue(&edges),
        1,
        "4 → 6 resolved; 7 → 9 still has nothing"
    );
    assert_eq!(topology(edges[1]), Topology2::IndirectKnown);
    assert_eq!(
        edges[0].0, before[0],
        "an already-resolved edge is untouched"
    );
    assert_eq!(
        edges[2].0, before[2],
        "evidence elsewhere does not touch 7 → 9"
    );
}

#[test]
fn the_loop_moves_topology_only_never_certification_or_the_rest() {
    let d = declarations();
    let start = population();
    let mut edges = start.clone();
    let mut sealed = initial_chain();
    run_to_fixed_point(&mut edges, &sealed, &d);
    sealed.push(step(5, 6));
    run_to_fixed_point(&mut edges, &sealed, &d);

    let topology_bits = 0b11u64 << 59;
    for (after, before) in edges.iter().zip(&start) {
        assert_eq!(
            after.0 & !topology_bits,
            before.0 & !topology_bits,
            "only bits 59..60 may change"
        );
        assert_eq!(
            after.0 & CERT_BITS,
            before.0 & CERT_BITS,
            "certification never moves here"
        );
    }
}
