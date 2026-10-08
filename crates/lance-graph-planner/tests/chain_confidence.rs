//! PROBE-CHAIN-CONF and PROBE-STAMP-GATE
//! (`.claude/board/entries/2026-10-08-coresearch-ce64-moore-masking-wiring.md`).
//!
//! `replay_step` fuses each hop's weight into the running truth with NARS
//! REVISION, then overwrites `forward`'s own truth with the revised one. These
//! tests pin what that does to confidence along a chain, against `forward`'s
//! deduction arm as the reference, and what it does when an edge is revised
//! with itself (no evidential-base check exists on this path).

use causal_edge::edge::{CausalEdge64, InferenceType};
use causal_edge::pearl::CausalMask;
use causal_edge::plasticity::PlasticityState;
use causal_edge::tables::NarsTables;
use lance_graph_planner::chain_replay::{replay_step, ComposeTables};

fn keep_left() -> Box<[u8; 256 * 256]> {
    let mut t = vec![0u8; 256 * 256].into_boxed_slice();
    for a in 0..256usize {
        for b in 0..256usize {
            t[a * 256 + b] = a as u8;
        }
    }
    t.try_into().expect("256*256")
}

#[allow(deprecated)] // v2 `pack` ignores temporal; not under test
fn edge(f: u8, c: u8) -> CausalEdge64 {
    CausalEdge64::pack(
        1,
        2,
        3,
        f,
        c,
        CausalMask::SPO,
        0,
        InferenceType::Deduction,
        PlasticityState::ALL_HOT,
        0,
    )
}

/// Confidence after each of `hops` steps, through `step`.
fn trace(hops: usize, step: impl Fn(CausalEdge64, CausalEdge64) -> CausalEdge64) -> Vec<u8> {
    let weight = edge(200, 200);
    let mut running = edge(200, 200);
    let mut out = vec![running.confidence_u8()];
    for _ in 0..hops {
        running = step(running, weight);
        out.push(running.confidence_u8());
    }
    out
}

/// Can-fire arm (reference): `forward`'s own Deduction arm lowers confidence
/// at every hop on this chain. If it did not, the fixture could not tell a
/// rising chain from a falling one.
#[test]
fn forward_deduction_lowers_confidence_along_the_chain() {
    let t = keep_left();
    let c = trace(5, |r, w| r.forward(w, &t, &t, &t).unwrap());
    assert!(c.windows(2).all(|p| p[1] < p[0]), "{c:?}");
}

/// MEASURED (pinned): through `replay_step`, the same 5-hop deduction chain
/// RAISES confidence, 200 -> 224 -> 237, then holds at 237 (the revision
/// table's confidence bins saturate). Each hop is revised into the running
/// truth instead of deduced through it, so confidence never falls.
///
/// FAILS IF: the trace changes (flip this test and record which rule the step
/// now applies).
#[test]
fn replay_step_raises_confidence_along_a_deduction_chain() {
    let t = keep_left();
    let tables = NarsTables::build(16);
    let compose = ComposeTables {
        s: &t,
        p: &t,
        o: &t,
    };
    let c = trace(5, |r, w| replay_step(r, w, &tables, compose).unwrap());
    assert_eq!(c, [200, 224, 237, 237, 237, 237]);
}

/// MEASURED (pinned): revising an edge with ITSELF raises its confidence.
/// `BeliefArena` refuses that (overlapping stamps mean choice, not pooling);
/// this path has no evidential base to check.
///
/// FAILS IF: `replay_step` gains a stamp or self-revision guard.
#[test]
fn self_revision_raises_confidence_on_the_replay_path() {
    let t = keep_left();
    let tables = NarsTables::build(16);
    let compose = ComposeTables {
        s: &t,
        p: &t,
        o: &t,
    };
    let e = edge(200, 128);
    let out = replay_step(e, e, &tables, compose).unwrap();
    assert!(
        out.confidence_u8() > e.confidence_u8(),
        "{} -> {}",
        e.confidence_u8(),
        out.confidence_u8()
    );
}

/// MEASURED (pinned): a weight with NO evidence (c = 0) still raises
/// confidence on the replay path, 128 -> 137. `NarsTables::revise` reads
/// confidence through 16 bins, and bin 0 carries non-zero evidence.
///
/// FAILS IF: a zero-confidence weight stops adding confidence.
#[test]
fn a_zero_confidence_weight_still_adds_confidence_on_the_replay_path() {
    let t = keep_left();
    let tables = NarsTables::build(16);
    let compose = ComposeTables {
        s: &t,
        p: &t,
        o: &t,
    };
    let e = edge(200, 128);
    let out = replay_step(e, edge(200, 0), &tables, compose).unwrap();
    assert_eq!((e.confidence_u8(), out.confidence_u8()), (128, 137));
}

/// Silent arm: `forward`'s own deduction gives zero confidence for a weight
/// with zero confidence, so the reference does not add evidence from nothing.
#[test]
fn forward_deduction_adds_nothing_for_a_zero_confidence_weight() {
    let t = keep_left();
    let out = edge(200, 128).forward(edge(200, 0), &t, &t, &t).unwrap();
    assert_eq!(out.confidence_u8(), 0);
}
