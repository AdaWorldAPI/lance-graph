//! `rung_divergence_reliability` — the jc battery on the TWO rung scales that the
//! cross-session audit (`E-RECIPE-LOCI-CONVERGENCE-1`) found to diverge, answering
//! the operator's question: does combining them create a **statistical tautology**
//! (they measure the same thing → redundant-identical, no confidence gain) or
//! **higher confidence + real redundancy** (related-but-distinct facets → the
//! combination genuinely reduces error)?
//!
//! The two raters of "recipe depth", over the same 34 recipes:
//!   * `recipe_dispatch::rung`  — inference-escalation COST (Tier + NARS delta, 1..9)
//!   * `recipe_loci::loci_rung` — organ NEED-DEPTH (`max(locus_rung)`, 0..6)
//!
//! Battery (`jc::reliability`): Pearson, Spearman (rank — scale-invariant), ICC(2,1)
//! absolute-agreement, ICC(3,1) consistency, Cronbach α over the 2-item scale.
//! Verdict lens (evolve-not-collapse precedent): α≳0.8 = TAUTOLOGY (one redundant);
//! α≈0.4–0.7 = DISTINCT FACETS (real higher-confidence redundancy); α≲0.2 =
//! ORTHOGONAL (separate axes, keep-both but not a confidence combination) — cf.
//! awareness tenants α 0.448 (keep-both) and part_of:is_a vs palette256² α 0.019
//! (orthogonal).
//!
//! ```sh
//! cargo run -p jc --example rung_divergence_reliability
//! ```

use jc::reliability::{cronbach_alpha, icc, pearson, spearman, IccForm};
use lance_graph_contract::recipe_dispatch::rung as dispatch_rung;
use lance_graph_contract::recipe_loci::loci_rung;

/// z-normalize a vector (mean 0, sd 1) so absolute-agreement ICC + α are not
/// dominated by the raw scale mismatch (1..9 vs 0..6).
fn z(v: &[f64]) -> Vec<f64> {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    let sd = var.sqrt();
    if sd == 0.0 {
        return vec![0.0; v.len()];
    }
    v.iter().map(|x| (x - m) / sd).collect()
}

fn main() {
    println!("rung_divergence_reliability — do the two rung scales combine to tautology or confidence?\n");

    // the two rating vectors over the real 34 recipes (from the shipped fns)
    let a: Vec<f64> = (1..=34u8).map(|id| dispatch_rung(id) as f64).collect();
    let b: Vec<f64> = (1..=34u8).map(|id| loci_rung(id) as f64).collect();

    // rank agreement is scale-invariant → the honest "same order?" coefficient
    let sp = spearman(&a, &b).expect("spearman");
    let pe = pearson(&a, &b).expect("pearson");

    // absolute-agreement + consistency + α on z-normalized scales (fair to the
    // scale mismatch; documented normalization, not a planted transform)
    let za = z(&a);
    let zb = z(&b);
    let ratings: Vec<Vec<f64>> = za.iter().zip(&zb).map(|(&x, &y)| vec![x, y]).collect();
    let icc21 = icc(&ratings, IccForm::Icc2_1).expect("icc21");
    let icc31 = icc(&ratings, IccForm::Icc3_1).expect("icc31");
    let alpha = cronbach_alpha(&[za.clone(), zb.clone()]).expect("alpha");

    // order divergence (the audit's 38%) as a coefficient cross-check
    let inversions = {
        let mut inv = 0usize;
        for i in 0..34 {
            for j in (i + 1)..34 {
                if (a[i] < a[j]) != (b[i] < b[j]) {
                    inv += 1;
                }
            }
        }
        inv
    };
    let disorder = 100.0 * inversions as f64 / (34.0 * 33.0 / 2.0);

    println!("── the two depth raters over 34 recipes ──");
    println!("  Pearson r         {pe:+.3}   (linear)");
    println!("  Spearman ρ        {sp:+.3}   (rank — the scale-invariant 'same order?')");
    println!("  ICC(2,1) abs-agr  {icc21:+.3}   (absolute agreement, z-scaled)");
    println!("  ICC(3,1) consist  {icc31:+.3}   (consistency, rater bias ignored)");
    println!("  Cronbach α        {alpha:+.3}   (internal consistency of the 2-item depth scale)");
    println!("  order disorder    {disorder:.1}%   ({inversions}/561 pairs reordered)");

    // ── the verdict ──
    println!("\n── tautology vs higher-confidence redundancy? ──");
    let verdict = if alpha >= 0.8 && sp >= 0.8 {
        "TAUTOLOGY — the two rungs measure the same depth; combining is redundant-identical (no confidence gain, one is spare)."
    } else if alpha >= 0.35 || sp >= 0.4 {
        "DISTINCT FACETS — related but not identical (inference-COST vs organ-DEPTH); combining is genuine higher-confidence redundancy (each reduces the other's error)."
    } else {
        "ORTHOGONAL — near-independent axes; keep BOTH, but the combination is two separate measurements, not a confidence boost (cf. part_of:is_a vs palette256²)."
    };
    println!("  {verdict}");
    println!(
        "  (precedent: awareness tenants α 0.448 = keep-both distinct facets; part_of:is_a vs palette256² α 0.019 = orthogonal)"
    );

    println!("\n── companion finding: the two GATES (dispatch_guard_redundancy) ──");
    println!("  The GATE combination is NOT symmetric: 109 organ-only catches, 0 scalar-only →");
    println!(
        "  the scalar gate is a TAUTOLOGICAL SUBSET of the organ gate on the projected substrate"
    );
    println!(
        "  (ctx is projected FROM the witness, so the organ is the finer source). The scalar gate"
    );
    println!(
        "  is a cheap coarse pre-filter, the organ the authoritative net — asymmetric subsumption,"
    );
    println!(
        "  not confidence-adding redundancy. So the RUNG and the GATE divergences give DIFFERENT"
    );
    println!("  redundancy verdicts, and both are honest: measure each, never assume symmetry.");

    // ═══ registered gates ═══
    println!("\n── gates ──");
    let mut green = true;

    // G1: the battery is well-posed (all coefficients finite, α in [-1,1]).
    let g1 = [pe, sp, icc21, icc31, alpha].iter().all(|x| x.is_finite())
        && (-1.0..=1.0).contains(&alpha);
    println!(
        "[{}] G1 jc battery well-posed (all coefficients finite)",
        pf(g1)
    );
    green &= g1;

    // G2: the order divergence is real (matches the audit's ~38%) AND the rank
    //     agreement is positive but sub-tautological — the honest "diverge but
    //     correlate" shape the audit reported.
    let g2 = disorder > 20.0 && sp > 0.0 && sp < 0.9;
    println!(
        "[{}] G2 diverge-but-correlate: order disorder >20% AND 0 < ρ < 0.9 (ρ={sp:.3})",
        pf(g2)
    );
    green &= g2;

    // G3: the verdict is the middle regime (distinct facets) — NOT a tautology
    //     (they carry different information: cost vs depth) and NOT orthogonal
    //     (they agree on the apex). This is the load-bearing keep-both result.
    let g3 = !(alpha >= 0.8 && sp >= 0.8) && (alpha >= 0.35 || sp >= 0.4);
    println!(
        "[{}] G3 verdict = DISTINCT FACETS (neither tautology nor orthogonal)",
        pf(g3)
    );
    green &= g3;

    println!(
        "\n{}",
        if green {
            "ALL GATES GREEN"
        } else {
            "GATE FAILURE"
        }
    );
    assert!(green, "rung-divergence reliability gates failed");
}

fn pf(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}
