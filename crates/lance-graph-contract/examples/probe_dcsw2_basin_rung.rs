// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `probe_dcsw2_basin_rung` — CONTRACT-LEVEL scoping probe for D-CSW-2.
//!
//! Registered claim (plan `.claude/plans/causal-rung-standing-wave-v1.md` §6.3,
//! written BEFORE this file existed): basin co-occupancy
//! ([`PairPalette`], PR #787's certified distance table) joined with rung
//! survival ([`standing_wave_grounded`]) predicts causal-edge candidates on a
//! synthetic AND-gate fixture, beating EITHER signal alone by a registered
//! margin — without repeating the M1-M3 autocorrelation-style confound the
//! same plan's leg-1 audit already caught once.
//!
//! **Scope note — do not overclaim.** This is the contract-level MECHANISM
//! test on a synthetic, deterministic (no rng/clock) fixture. It exercises
//! the REAL `PairPalette`/witness-fabric primitives, not stand-ins, but it is
//! NOT the real-corpus labeled-candidate-set D-CSW-2 the plan's own pass
//! criterion names — that still needs real basins from real data.
//!
//! Ground truth is an AND-gate over two independent axes (4 equal groups):
//!
//! | Group | Co-occupy basin? | Rung survives? | Label |
//! |---|---|---|---|
//! | 1 | YES | YES | TRUE (real candidate) |
//! | 2 | YES | NO  | FALSE (coincidental neighbor) |
//! | 3 | NO  | YES | FALSE (spurious temporal pattern) |
//! | 4 | NO  | NO  | FALSE |
//!
//! A single-ablation score CANNOT recover this label (basin-only conflates
//! groups 1&2; rung-only conflates 1&3) — only the joint (product) score can,
//! which is what makes this a genuine, non-tautological test.

use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_contract::recipe_substrate::PairPalette;
use lance_graph_contract::witness_fabric::{standing_wave_grounded, WaveGrounding};

/// Pairs per group (4 groups → `4 * GROUP_SIZE` total pairs).
const GROUP_SIZE: usize = 25;
/// Registered pass margin (plan §6.3): joint precision@k must exceed BOTH
/// single-ablation precisions@k by at least this much.
const MARGIN: f32 = 0.15;

fn wit(edges: &[(Locus, i8)]) -> CausalWitnessFacet {
    let mut w = CausalWitnessFacet::ZERO;
    for &(l, o) in edges {
        w = w.with(l, o);
    }
    w
}

/// Deterministic 16-centroid axis codebook (index-derived, no rng/clock).
fn axis_codebook(seed: usize) -> Vec<Vec<f32>> {
    (0..16)
        .map(|c| {
            vec![
                ((c * 5 + seed) % 13) as f32 - 6.0,
                ((c * 3 + seed * 2) % 11) as f32 - 5.0,
            ]
        })
        .collect()
}

struct Pair {
    /// `PairPalette::similarity` between the two (basin, identity) points.
    basin_score: f32,
    /// `1.0` iff `standing_wave_grounded` returns `Causal`, else `0.0`.
    rung_score: f32,
    /// Ground truth: co-occupy AND rung-survives.
    label: bool,
}

fn make_pair(co_occupy: bool, rung_survives: bool, idx: usize, pal: &PairPalette) -> Pair {
    // Co-occupancy: identical (basin, identity) point if co_occupy, else
    // maximally separated in the 16-centroid codebook.
    let a = ((idx as u8) % 16, ((idx * 3) as u8) % 16);
    let b = if co_occupy {
        a
    } else {
        ((a.0 + 8) % 16, (a.1 + 8) % 16)
    };
    let basin_score = pal.similarity(a, b);

    // Rung survival: the EXACT fixture shapes from dispatch_guard's own
    // shipped tests (`bound_and_settled_chain_fires` for Causal,
    // `bound_but_non_local_cause_escalates_not_coincidental` for Escalate) —
    // not reinvented, mirrored verbatim from src/dispatch_guard.rs.
    let (window, focal_idx, locus, passes): (Vec<(usize, CausalWitnessFacet)>, usize, Locus, u8) =
        if rung_survives {
            let focal = wit(&[(Locus::Quorum, 1)]);
            let peer = wit(&[(Locus::Temporal, 0)]); // terminal, no Quorum rebind
            (vec![(0, focal), (1, peer)], 0, Locus::Quorum, 4)
        } else {
            let a2 = wit(&[(Locus::Quorum, 7)]);
            let b2 = wit(&[(Locus::Quorum, 7)]); // rebinds → leaves ±8 → escalates
            (vec![(0, a2), (7, b2)], 0, Locus::Quorum, 8)
        };
    let grounding = standing_wave_grounded(focal_idx, &window, locus, passes);
    let rung_score = if grounding == WaveGrounding::Causal {
        1.0
    } else {
        0.0
    };

    Pair {
        basin_score,
        rung_score,
        label: co_occupy && rung_survives,
    }
}

/// Precision among the top-`k` highest-scoring entries (stable sort, so ties
/// resolve deterministically by original insertion order — no randomness).
fn precision_at_k(mut scored: Vec<(f32, bool)>, k: usize) -> f32 {
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let hits = scored.iter().take(k).filter(|&&(_, l)| l).count();
    hits as f32 / k as f32
}

fn main() {
    let pal = PairPalette::new(axis_codebook(1), axis_codebook(2), 100.0);

    let mut pairs = Vec::new();
    for i in 0..GROUP_SIZE {
        pairs.push(make_pair(true, true, i, &pal)); // group 1: TRUE
        pairs.push(make_pair(true, false, i, &pal)); // group 2
        pairs.push(make_pair(false, true, i, &pal)); // group 3
        pairs.push(make_pair(false, false, i, &pal)); // group 4
    }

    let k = GROUP_SIZE; // == |group 1| == |true labels|

    let basin_only: Vec<(f32, bool)> = pairs.iter().map(|p| (p.basin_score, p.label)).collect();
    let rung_only: Vec<(f32, bool)> = pairs.iter().map(|p| (p.rung_score, p.label)).collect();
    let joint: Vec<(f32, bool)> = pairs
        .iter()
        .map(|p| (p.basin_score * p.rung_score, p.label))
        .collect();

    let p_basin = precision_at_k(basin_only, k);
    let p_rung = precision_at_k(rung_only, k);
    let p_joint = precision_at_k(joint, k);

    println!(
        "D-CSW-2 contract-level probe (synthetic AND-gate fixture, N={} pairs, k={})",
        pairs.len(),
        k
    );
    println!("  basin-only precision@k: {p_basin:.3}");
    println!("  rung-only  precision@k: {p_rung:.3}");
    println!("  joint      precision@k: {p_joint:.3}");

    let margin_basin = p_joint - p_basin;
    let margin_rung = p_joint - p_rung;
    println!("  margin over basin-only: {margin_basin:+.3} (registered pass: >= {MARGIN})");
    println!("  margin over rung-only:  {margin_rung:+.3} (registered pass: >= {MARGIN})");

    let pass = margin_basin >= MARGIN && margin_rung >= MARGIN;
    assert!(
        pass,
        "D-CSW-2 contract-level probe KILL: joint did not clear both ablations by the registered {MARGIN} margin (margin_basin={margin_basin:.3}, margin_rung={margin_rung:.3})"
    );
    println!("PASS: joint basin+rung beats both single-signal ablations by >= {MARGIN}");
}
