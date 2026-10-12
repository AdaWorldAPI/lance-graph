//! **D-NAL-REV-0 — lance-graph's revision laws against an external NARS
//! reference.**
//!
//! EXPERIMENTAL, test-only. Nothing in production changes.
//!
//! D-RPF-CONF-0 (`.claude/board/entries/2026-10-09-d-rpf-conf-0-confidence-surface.md`)
//! measured how far the live revision laws disagree with EACH OTHER and
//! declared none canonical. This file adds the missing outside anchor: the
//! numbers OpenNARS for Research actually printed.
//!
//! Source of every expected value below (MEASURED, not derived by hand):
//! OpenNARS for Research, commit 4d1bdb0f599ccf6ed44edfc398f02f930713d30f,
//! run through its own `NALTest.testNAL`. Point 1 is upstream's
//! `single_step/nal1.0.nal` (and `nal2.0.nal`, same numbers):
//! `OUT: <bird --> swimmer>. %0.87;0.91%`. Points 2 and 3 come from an audit
//! fixture of the same shape (`<a2 --> b2>`, `<a3 --> b3>`), whose OUT lines
//! read `%0.45;0.77%` and `%0.50;0.95%`. OpenNARS prints truth to two decimals,
//! so that is the equivalence criterion: a law agrees when its output rounds to
//! the printed value. Laws that read and write u8 truth codes get one extra
//! output code (1/255), the most the grid itself can move a result. Law A runs
//! on the same grid and stays within 0.0045 of every printed value, so the
//! allowance is measured, not generous.
//!
//! The law letters are D-RPF-CONF-0's census:
//! A `CausalEdge64::revision`, B1/B16 `NarsTables::build(1|16).revise`,
//! C `ndarray::hpc::nars::nars_revision`, D planner `nars::truth::TruthValue::revise`,
//! E contract `exploration::NarsTruth::revision`, LIN contract
//! `grammar::revise_truth` (confidence used directly as the evidence weight).
//!
//! This pins which laws agree with the reference. It does not declare a law
//! canonical; that stays the operator decision D-RPF-CONF-0 left open (G2).

use causal_edge::edge::CausalEdge64;
use causal_edge::tables::{unpack_c, unpack_f, NarsTables};
use lance_graph_contract::crystal::TruthValue as CrystalTruth;
use lance_graph_contract::exploration::NarsTruth as ContractTruth;
use lance_graph_contract::grammar::revise_truth;
use lance_graph_planner::nars::truth::TruthValue as PlannerTruth;
use ndarray::hpc::nars::{nars_revision, NarsTruth as NdTruth};

/// (premise 1, premise 2, OpenNARS output), all as (frequency, confidence).
/// Premise 1 of point 1 is OpenNARS's default judgment truth %1.00;0.90%.
type Point = ((f32, f32), (f32, f32), (f32, f32));
const ORACLE: [Point; 3] = [
    ((1.0, 0.9), (0.1, 0.6), (0.87, 0.91)),
    ((0.8, 0.5), (0.3, 0.7), (0.45, 0.77)),
    ((0.0, 0.9), (1.0, 0.9), (0.50, 0.95)),
];

/// Print precision of the oracle: agreement is |x - printed| <= half a hundredth.
const PRINT_TOL: f32 = 0.005 + 1e-6;
/// For laws that read u8 truth codes: one further output code.
const GRID_TOL: f32 = PRINT_TOL + 1.0 / 255.0;

fn q(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0).round() as u8
}
fn d(x: u8) -> f32 {
    x as f32 / 255.0
}
fn edge(f: u8, c: u8) -> CausalEdge64 {
    let mut e = CausalEdge64(0);
    e.set_frequency_u8(f);
    e.set_confidence_u8(c);
    e
}

type Law = Box<dyn Fn((f32, f32), (f32, f32)) -> (f32, f32)>;

fn laws() -> Vec<(&'static str, f32, Law)> {
    vec![
        (
            "A",
            GRID_TOL,
            Box::new(|(f1, c1), (f2, c2)| {
                let r = edge(q(f1), q(c1)).revision(edge(q(f2), q(c2)));
                (d(r.frequency_u8()), d(r.confidence_u8()))
            }),
        ),
        (
            "B1",
            GRID_TOL,
            Box::new(|(f1, c1), (f2, c2)| {
                let p = NarsTables::build(1).revise(q(f1), q(c1), q(f2), q(c2));
                (d(unpack_f(p)), d(unpack_c(p)))
            }),
        ),
        (
            "B16",
            GRID_TOL,
            Box::new(|(f1, c1), (f2, c2)| {
                let p = NarsTables::build(16).revise(q(f1), q(c1), q(f2), q(c2));
                (d(unpack_f(p)), d(unpack_c(p)))
            }),
        ),
        (
            "C",
            PRINT_TOL,
            Box::new(|(f1, c1), (f2, c2)| {
                let r = nars_revision(NdTruth::new(f1, c1), NdTruth::new(f2, c2));
                (r.frequency, r.confidence)
            }),
        ),
        (
            "D",
            PRINT_TOL,
            Box::new(|(f1, c1), (f2, c2)| {
                let r = PlannerTruth::new(f1, c1).revise(&PlannerTruth::new(f2, c2));
                (r.frequency, r.confidence)
            }),
        ),
        (
            "E",
            PRINT_TOL,
            Box::new(|(f1, c1), (f2, c2)| {
                let r = ContractTruth::new(f1, c1).revision(&ContractTruth::new(f2, c2));
                (r.frequency, r.confidence)
            }),
        ),
        (
            "LIN",
            PRINT_TOL,
            Box::new(|(f1, c1), (f2, c2)| {
                let r = revise_truth(CrystalTruth::new(f1, c1), f2, c2);
                (r.frequency, r.confidence)
            }),
        ),
    ]
}

/// How many of the three oracle points a law reproduces.
fn agreements(law: &Law, tol: f32) -> usize {
    ORACLE
        .iter()
        .filter(|&&(p1, p2, (fo, co))| {
            let (f, c) = law(p1, p2);
            (f - fo).abs() <= tol && (c - co).abs() <= tol
        })
        .count()
}

#[test]
fn the_evidence_weighted_laws_reproduce_opennars_revision() {
    for (name, tol, law) in laws() {
        if ["A", "C", "D", "E"].contains(&name) {
            assert_eq!(agreements(&law, tol), 3, "{name} must match all 3 points");
        }
    }
}

/// Pinned two-sided: if one of these is fixed, this test goes red and the
/// table above must be re-measured rather than silently becoming wrong.
#[test]
fn b1_b16_and_lin_diverge_from_the_reference() {
    let got: Vec<(&str, usize)> = laws()
        .iter()
        .filter(|(n, _, _)| ["B1", "B16", "LIN"].contains(n))
        .map(|(n, tol, law)| (*n, agreements(law, *tol)))
        .collect();
    assert_eq!(got, vec![("B1", 0), ("B16", 1), ("LIN", 0)]);
}

/// B1 is the law the replay and counterfactual path emits (D-RPF-CONF-0). Its
/// confidence is the same whatever the premises' confidences are.
#[test]
fn b1_confidence_ignores_its_premises() {
    let all = laws();
    let b1 = &all.iter().find(|(n, _, _)| *n == "B1").unwrap().2;
    let cs: Vec<f32> = ORACLE.iter().map(|&(p1, p2, _)| b1(p1, p2).1).collect();
    assert!(cs.windows(2).all(|w| w[0] == w[1]), "{cs:?}");
    // The premises' confidences really do differ, so constancy is not the fixture's.
    assert!(ORACLE.iter().any(|&((_, c1), _, _)| c1 != ORACLE[0].0 .1));
}

/// LIN treats confidence as the evidence weight, so revision can end BELOW the
/// stronger premise. NAL revision adds evidence, so its confidence is never
/// below either premise (law C, the reference-matching one, shown alongside).
#[test]
fn lin_revision_can_lose_confidence() {
    let r = revise_truth(CrystalTruth::new(1.0, 0.9), 0.1, 0.6);
    assert!(r.confidence < 0.9, "got {}", r.confidence);
    let c = nars_revision(NdTruth::new(1.0, 0.9), NdTruth::new(0.1, 0.6));
    assert!(c.confidence >= 0.9, "got {}", c.confidence);
}
