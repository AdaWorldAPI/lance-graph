//! P3b · the SPO 2³ decomposition gives questions AND candidates inherently,
//! amortized all-at-once on the same cached tile data.
//!
//! `SpoDistances::all_projections` reads the three per-plane palette cells for
//! one `(a,b)` pair — `s_dist`, `p_dist`, `o_dist` — and produces **all eight**
//! Pearl projections (the 2³ subsets of {S,P,O}) by masked summation. Three
//! cache reads → eight causal questions. This probe pins two properties the
//! in-crate `test_all_projections` (subset-sum arithmetic) does not:
//!
//! 1. **Amortization** — the 8-vector is a pure function of the 3 scalar plane
//!    distances; no mask costs an extra table read. `all_projections` is the
//!    Morton-tile "2³ all at once on the same data in cache."
//! 2. **Questions AND candidates inherently** — the 8 masks are 8 *distinct*
//!    causal questions (prior / 3 marginals / confounder / association /
//!    intervention / counterfactual), and because they weight the planes
//!    differently, the SAME two candidates can rank **oppositely** under
//!    Association (`SO`) vs Intervention (`PO`). The decomposition is not
//!    redundant: each rung is its own retrieval.
//!
//! Integer-exact. `lance-graph-planner` only; no new deps.

use lance_graph_planner::cache::nars_engine::{
    SpoDistances, SpoHead, ALL_MASKS, MASK_NONE, MASK_O, MASK_P, MASK_PO, MASK_S, MASK_SO,
    MASK_SP, MASK_SPO,
};

fn head(s: u8, p: u8, o: u8) -> SpoHead {
    let mut h = SpoHead::zero();
    h.s_idx = s;
    h.p_idx = p;
    h.o_idx = o;
    h
}

/// Reconstruct the 8 projections from ONLY the 3 scalar plane distances, by the
/// mask's popcount-subset sum — the amortized computation the engine performs
/// once the 3 cells are in cache.
fn expected_from_three(s_d: u32, p_d: u32, o_d: u32) -> [u32; 8] {
    let mut out = [0u32; 8];
    for (i, &mask) in ALL_MASKS.iter().enumerate() {
        let mut d = 0;
        if mask & MASK_S != 0 {
            d += s_d;
        }
        if mask & MASK_P != 0 {
            d += p_d;
        }
        if mask & MASK_O != 0 {
            d += o_d;
        }
        out[i] = d;
    }
    out
}

#[test]
fn p3b_eight_questions_are_a_pure_function_of_three_reads() {
    // One (a,b) pair; three plane cells set.
    let mut d = SpoDistances::new_zero();
    let (s_d, p_d, o_d) = (17u32, 101u32, 255u32);
    d.s_table[0 * 256 + 1] = s_d as u16;
    d.p_table[0 * 256 + 1] = p_d as u16;
    d.o_table[0 * 256 + 1] = o_d as u16;

    let a = head(0, 0, 0);
    let b = head(1, 1, 1);

    let proj = d.all_projections(&a, &b);
    // The whole 8-vector is determined by the 3 scalar reads — amortization.
    assert_eq!(proj, expected_from_three(s_d, p_d, o_d));

    // Named taxonomy sanity: prior=0, the three marginals, and the ladder tops.
    assert_eq!(proj[0], 0, "MASK_NONE = prior");
    assert_eq!(proj[7], s_d + p_d + o_d, "MASK_SPO = counterfactual = full sum");
    let idx = |m: u8| ALL_MASKS.iter().position(|&x| x == m).unwrap();
    assert_eq!(proj[idx(MASK_S)], s_d);
    assert_eq!(proj[idx(MASK_P)], p_d);
    assert_eq!(proj[idx(MASK_O)], o_d);
    assert_eq!(proj[idx(MASK_SP)], s_d + p_d);
    assert_eq!(proj[idx(MASK_SO)], s_d + o_d, "Association");
    assert_eq!(proj[idx(MASK_PO)], p_d + o_d, "Intervention (S projected out)");
}

#[test]
fn p3b_ladder_is_monotone_a_superset_question_never_undercounts() {
    // Adding a plane to a mask can never decrease its projection: the 2³ lattice
    // is monotone, so the counterfactual (SPO) upper-bounds every sub-question.
    let mut d = SpoDistances::new_zero();
    d.s_table[0 * 256 + 1] = 40;
    d.p_table[0 * 256 + 1] = 70;
    d.o_table[0 * 256 + 1] = 25;
    let proj = d.all_projections(&head(0, 0, 0), &head(1, 1, 1));
    let idx = |m: u8| ALL_MASKS.iter().position(|&x| x == m).unwrap();

    // A ⊆ B (as bitsets) ⇒ proj[A] ≤ proj[B]. Spot-check the ladder spine.
    assert!(proj[idx(MASK_NONE)] <= proj[idx(MASK_S)]);
    assert!(proj[idx(MASK_S)] <= proj[idx(MASK_SO)]);
    assert!(proj[idx(MASK_SO)] <= proj[idx(MASK_SPO)]);
    assert!(proj[idx(MASK_O)] <= proj[idx(MASK_PO)]);
    assert!(proj[idx(MASK_PO)] <= proj[idx(MASK_SPO)]);
    assert_eq!(
        proj[idx(MASK_SPO)],
        *proj.iter().max().unwrap(),
        "counterfactual dominates all sub-questions"
    );
}

#[test]
fn p3b_association_and_intervention_rank_candidates_oppositely() {
    // The decomposition yields CANDIDATES inherently: the same two candidates
    // seen from a fixed context rank oppositely under Association (SO) vs
    // Intervention (PO), because PO projects out the Subject plane.
    let mut d = SpoDistances::new_zero();
    let ctx = head(0, 0, 0);
    let x = head(1, 1, 1); // near on S, far on P
    let y = head(2, 2, 2); // far on S, near on P

    // Lookup is table[a_idx*256 + b_idx]; we call all_projections(candidate, ctx=0),
    // so set the (candidate, 0) cells.
    // Subject plane: X near, Y far.
    d.s_table[1 * 256 + 0] = 10;
    d.s_table[2 * 256 + 0] = 90;
    // Predicate plane: X far, Y near.
    d.p_table[1 * 256 + 0] = 90;
    d.p_table[2 * 256 + 0] = 10;
    // Object plane: equal — so it cannot decide the ordering.
    d.o_table[1 * 256 + 0] = 50;
    d.o_table[2 * 256 + 0] = 50;

    let px = d.all_projections(&x, &ctx);
    let py = d.all_projections(&y, &ctx);
    let idx = |m: u8| ALL_MASKS.iter().position(|&x| x == m).unwrap();

    // Association (S+O): X wins (its Subject match dominates).
    assert!(
        px[idx(MASK_SO)] < py[idx(MASK_SO)],
        "under Association, candidate X is nearer"
    );
    // Intervention (P+O, Subject confounder projected out): Y wins.
    assert!(
        py[idx(MASK_PO)] < px[idx(MASK_PO)],
        "under Intervention, candidate Y is nearer — the Subject match no longer counts"
    );
    // Same data, same cache line, opposite winners: the 8 masks are 8 real
    // questions, each with its own candidate answer — not one distance in 8 hats.
}
