//! TD-BASE17-FOLD-CEILING-SINGLE-WORD — input-specificity probe.
//!
//! Self-contained, deterministic SYNTHETIC characterization. NO external API,
//! NO real embeddings, NO on-disk cache — every input is generated in-process
//! from a SplitMix64 stream (canon seed `0x9E3779B97F4A7C15`).
//!
//! ## Background (the finding under test)
//!
//! `E-PROBE-CODEBOOK-44-MECHANISM-1` / `TECH_DEBT.md`
//! `TD-BASE17-FOLD-CEILING-SINGLE-WORD` found that on 4096 real single-word
//! `jina-embeddings-v3` vectors, the Base17 17-dim golden-fold PROJECTION
//! ceiling (Base17 L1 distance vs raw cosine, NO codebook — "framing B") caps
//! at ρ = 0.2599, and that the folded Base17 patterns' pairwise-L1 spread has
//! coefficient of variation CV = 0.220 (near-degenerate: distances cluster
//! tightly around a single value). The finding's claim: this ceiling is an
//! INPUT-VARIANCE artifact of low-CV inputs, NOT a hard limit of the Base17
//! fold or the 256-entry codebook itself.
//!
//! ## What this probe measures
//!
//! This probe cannot re-run the real API-derived embeddings (no external API
//! permitted here). Instead it tests the GENERALIZED mechanism directly: does
//! rank-correlation fidelity of a Base17 256-entry codebook degrade as the
//! INPUT pairwise-distance CV drops, independent of any upstream projection?
//!
//! Synthetic Base17 patterns are generated directly in 17-dim i16 space as a
//! cluster mixture (`k_clusters` centers + within-cluster noise). The ratio of
//! cluster separation to within-cluster spread is swept to produce a
//! monotonic CV ladder from near-degenerate (low CV, mimics the real
//! single-word finding) to well-separated (high CV, mimics structured
//! SPO/aerial patterns the 0.965/0.9973 canon anchors were set on).
//!
//! At each CV level: build BOTH a flat-256 (`Palette::build`) and a
//! hierarchical-16×16 (`Palette::build_hierarchical`) codebook on a 75% train
//! split, then measure Spearman ρ of the codebook-RECONSTRUCTED pairwise
//! distance vs the GROUND-TRUTH Base17 L1 distance on a held-out 25% split —
//! the "codebook-fidelity framing" (framing A), the same framing the
//! 0.965/0.9973 canon anchors were measured against in
//! `probe_codebook_44_realdata.rs`.
//!
//! ## Verdict
//!
//! CONFIRMED: ρ rises (roughly monotonically) with CV and clears 0.965 at the
//! high-CV end → the ceiling is input-variance-driven / input-specific, a
//! property of the DATA, not of the Base17 fold or the codebook.
//!
//! FALSIFIED: ρ stays capped even at high CV → the fold/codebook imposes a
//! harder limit than input variance alone (the more valuable, stronger
//! finding — reported honestly if observed).
//!
//! Run:
//!   cargo run --example probe_base17_cv_sweep --manifest-path crates/bgz17/Cargo.toml

use bgz17::base17::Base17;
use bgz17::palette::Palette;
use bgz17::BASE_DIM;

// ── deterministic RNG (SplitMix64, canon seed) ──────────────────────────────

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform f64 in `[-1.0, 1.0)`.
    fn signed_unit(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64); // [0,1)
        u * 2.0 - 1.0
    }
}

// ── Spearman ρ (average-rank, tie-aware) — mirrors probe_codebook_44_realdata.rs ─

fn average_ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1 + j) as f64) / 2.0;
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    ranks
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        0.0
    } else {
        cov / (vx.sqrt() * vy.sqrt())
    }
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&average_ranks(x), &average_ranks(y))
}

fn deterministic_pairs(
    lo: usize,
    hi: usize,
    n_pairs: usize,
    rng: &mut SplitMix64,
) -> Vec<(usize, usize)> {
    let span = hi - lo;
    assert!(span >= 2, "need >= 2 indices to form pairs");
    let mut pairs = Vec::with_capacity(n_pairs);
    while pairs.len() < n_pairs {
        let a = lo + rng.below(span);
        let b = lo + rng.below(span);
        if a != b {
            pairs.push((a.min(b), a.max(b)));
        }
    }
    pairs
}

// ── synthetic Base17 generation ──────────────────────────────────────────────

/// Fixed within-cluster half-range (arbitrary units; only the RATIO to
/// `sep_mult` matters — CV is scale-invariant).
const UNIT: f64 = 300.0;

/// Generate `n` synthetic Base17 patterns as a `k_clusters`-cluster mixture.
///
/// `sep_mult` controls cluster-center spread relative to the fixed
/// within-cluster noise (`UNIT`): `sep_mult -> 0` collapses every cluster
/// into one blob (near-degenerate, LOW pairwise-distance CV — mimics the
/// real single-word dense-embedding fold); large `sep_mult` separates
/// clusters strongly relative to their internal spread (bimodal small/large
/// distances, HIGH CV — mimics structured SPO/aerial patterns).
fn gen_patterns(n: usize, k_clusters: usize, sep_mult: f64, rng: &mut SplitMix64) -> Vec<Base17> {
    let sep_range = sep_mult * UNIT;
    let centers: Vec<[f64; BASE_DIM]> = (0..k_clusters)
        .map(|_| {
            let mut c = [0f64; BASE_DIM];
            for d in c.iter_mut() {
                *d = rng.signed_unit() * sep_range;
            }
            c
        })
        .collect();
    (0..n)
        .map(|_| {
            let cid = rng.below(k_clusters);
            let mut dims = [0i16; BASE_DIM];
            for (d, slot) in dims.iter_mut().enumerate() {
                let v = centers[cid][d] + rng.signed_unit() * UNIT;
                *slot = v.round().clamp(-32768.0, 32767.0) as i16;
            }
            Base17 { dims }
        })
        .collect()
}

/// Pairwise Base17 L1 distance mean / std / coefficient of variation over a
/// sample of pairs — the same CV metric the real-data probe's diagnostic
/// block computes ("Base17 pairwise-L1 spread").
fn distance_cv(patterns: &[Base17], pairs: &[(usize, usize)]) -> (f64, f64, f64) {
    let d: Vec<f64> = pairs
        .iter()
        .map(|&(i, j)| patterns[i].l1(&patterns[j]) as f64)
        .collect();
    let mean = d.iter().sum::<f64>() / d.len() as f64;
    let var = d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / d.len() as f64;
    let std = var.sqrt();
    let cv = if mean > 0.0 { std / mean } else { 0.0 };
    (mean, std, cv)
}

/// Codebook-fidelity ρ (framing A): reconstructed distance via nearest-centroid
/// codes vs the ground-truth Base17 L1 distance. Mirrors
/// `probe_codebook_44_realdata.rs::reconstruct_rho_vs_base17` — the framing the
/// 0.965 (k=128) / 0.9973 canon anchors were measured against.
fn reconstruct_rho_vs_base17(
    palette: &Palette,
    patterns: &[Base17],
    pairs: &[(usize, usize)],
) -> f64 {
    let table = palette.build_distance_table();
    let codes: Vec<u8> = patterns.iter().map(|p| palette.nearest(p)).collect();
    let mut recon = Vec::with_capacity(pairs.len());
    let mut reference = Vec::with_capacity(pairs.len());
    for &(i, j) in pairs {
        recon.push(table.distance(codes[i], codes[j]) as f64);
        reference.push(patterns[i].l1(&patterns[j]) as f64);
    }
    let rho = spearman(&recon, &reference);
    if rho.is_nan() {
        panic!("NaN Spearman ρ — HALT (metric stage)");
    }
    rho
}

/// NaN scan on the Base17 fixed-point dims — certification doctrine: never
/// silently filter a NaN. i16 dims cannot carry NaN by construction (integer
/// storage), so this asserts the invariant explicitly rather than skipping it.
fn nan_scan_patterns(patterns: &[Base17]) {
    for (i, p) in patterns.iter().enumerate() {
        for (d, &v) in p.dims.iter().enumerate() {
            assert!(
                v > i16::MIN,
                "sentinel/overflow value at pattern {i} dim {d}: {v} (i16::MIN is the clamp floor, \
                 should never be hit by a well-formed generator) — HALT"
            );
        }
    }
}

struct SweepRow {
    label: &'static str,
    sep_mult: f64,
    cv_achieved: f64,
    flat_rho: f64,
    hier_rho: f64,
}

fn main() {
    println!("TD-BASE17-FOLD-CEILING-SINGLE-WORD — input-specificity probe");
    println!("================================================================");
    println!("Self-contained SYNTHETIC characterization (NO external API, NO real data).");
    println!("Deterministic SplitMix64, seed 0x9E3779B97F4A7C15.\n");
    println!("Claim under test: is the codebook/Base17-fold fidelity ceiling an");
    println!("INPUT-VARIANCE artifact (input-specific), NOT a hard fold/codebook limit —");
    println!("does rank-correlation fidelity rise toward the 0.965/0.9973 canon anchors");
    println!("as pairwise-distance coefficient-of-variation (CV) rises?\n");
    println!("REAL-DATA REFERENCE (framing B, projection-only ceiling on real single-word");
    println!("jina-embeddings-v3 data, TD-BASE17-FOLD-CEILING-SINGLE-WORD /");
    println!("E-PROBE-CODEBOOK-44-MECHANISM-1):");
    println!(
        "  CV = 0.220  ->  projection-only rho = 0.2599  (Base17 L1 vs raw cosine, NO codebook)"
    );
    println!("  NOTE: framing B (no codebook) is NOT directly comparable to framing A");
    println!("  (codebook fidelity vs Base17 L1) measured below — printed only as a");
    println!("  cross-check anchor for the low-CV end of this sweep.\n");

    let n = 4096usize;
    let n_pairs = 4000usize;
    let max_iter = 20usize;
    let train_n = (n * 3) / 4;

    // (label, k_clusters, sep_mult) sweep. For a k-cluster mixture with a
    // uniformly-random cluster assignment, the pairwise-distance CV is
    // dominated by the two-point-mixture variance formula: same-cluster
    // pairs occur with probability p = 1/k, cross-cluster with (1-p); as
    // separation grows the CV approaches sqrt(p / (1-p)). Solving for the
    // target CVs gives k = {100, 22, 7, 3, 2} (asymptotes ~0.10/0.22/0.41/
    // 0.71/1.00); `sep_mult = 50.0` is held fixed and large enough
    // (sep_range = 50 * UNIT) that the within-cluster and between-cluster
    // distance modes are cleanly separated at every k.
    let sweep_spec: [(&str, usize, f64); 5] = [
        ("0.10", 100, 50.0),
        ("0.22", 22, 50.0),
        ("0.40", 7, 50.0),
        ("0.70", 3, 50.0),
        ("1.00", 2, 50.0),
    ];

    let mut rows: Vec<SweepRow> = Vec::with_capacity(sweep_spec.len());

    for (idx, (label, k_clusters, sep_mult)) in sweep_spec.iter().enumerate() {
        // Independent, deterministic seed per sweep step (index-derived, no
        // float-bit-pattern subtlety).
        let seed_offset = 0x1000_0000_0000_0001u64.wrapping_mul(idx as u64 + 1);
        let mut gen_rng = SplitMix64(0x9E37_79B9_7F4A_7C15 ^ seed_offset);
        let patterns = gen_patterns(n, *k_clusters, *sep_mult, &mut gen_rng);
        nan_scan_patterns(&patterns);

        let mut pair_rng = SplitMix64(0x9E37_79B9_7F4A_7C15);
        let held_pairs = deterministic_pairs(train_n, n, n_pairs, &mut pair_rng);

        let (_, _, cv_achieved) = distance_cv(&patterns, &held_pairs);

        let train_patterns = &patterns[..train_n];
        let flat = Palette::build(train_patterns, 256, max_iter);
        let hp = Palette::build_hierarchical(train_patterns, max_iter);

        let flat_rho = reconstruct_rho_vs_base17(&flat, &patterns, &held_pairs);
        let hier_rho = reconstruct_rho_vs_base17(&hp.leaves, &patterns, &held_pairs);

        rows.push(SweepRow {
            label,
            sep_mult: *sep_mult,
            cv_achieved,
            flat_rho,
            hier_rho,
        });
    }

    println!(
        "{:<8} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "CVtgt", "sepMult", "CVach", "flatRho", "hierRho", "clr0.965", "clr0.9973"
    );
    for r in &rows {
        let clr965 = if r.flat_rho >= 0.965 && r.hier_rho >= 0.965 {
            "yes"
        } else {
            "no"
        };
        let clr9973 = if r.flat_rho >= 0.9973 && r.hier_rho >= 0.9973 {
            "yes"
        } else {
            "no"
        };
        println!(
            "{:<8} {:>9.2} {:>10.4} {:>10.4} {:>10.4} {:>10} {:>10}",
            r.label, r.sep_mult, r.cv_achieved, r.flat_rho, r.hier_rho, clr965, clr9973
        );
    }

    // ── structure-is-free check across the sweep ────────────────────────────
    println!("\nhier vs flat (structure-is-free) across the sweep:");
    for r in &rows {
        let rel = if r.hier_rho >= r.flat_rho - 1e-9 {
            ">="
        } else {
            "<"
        };
        println!(
            "  CVtgt={} (CVach={:.4}): hier {} flat  (hier={:.4} flat={:.4})",
            r.label, r.cv_achieved, rel, r.hier_rho, r.flat_rho
        );
    }
    let structure_is_free_holds = rows.iter().all(|r| r.hier_rho >= r.flat_rho - 0.02);

    // ── extreme-CV diagnostic (k=2, push separation far past the CVtgt=1.00
    //    row) — disambiguates a SOFT asymptote (rho keeps climbing toward the
    //    anchor as separation grows, so the CVtgt=1.00 row was merely
    //    under-separated) from a HARD cap (rho plateaus below 0.965 even as
    //    separation -> very large, i.e. a real codebook/quantization limit
    //    independent of further CV growth). Not part of the 5-row sweep
    //    table above; purely diagnostic context for the verdict below.
    let mut extreme_rows: Vec<SweepRow> = Vec::new();
    for (label, sep_mult) in [("extreme-1", 70.0f64), ("extreme-2", 100.0f64)] {
        let seed_offset = 0x2000_0000_0000_0001u64.wrapping_mul((extreme_rows.len() as u64) + 1);
        let mut gen_rng = SplitMix64(0x9E37_79B9_7F4A_7C15 ^ seed_offset);
        let patterns = gen_patterns(n, 2, sep_mult, &mut gen_rng);
        nan_scan_patterns(&patterns);
        let mut pair_rng = SplitMix64(0x9E37_79B9_7F4A_7C15);
        let held_pairs = deterministic_pairs(train_n, n, n_pairs, &mut pair_rng);
        let (_, _, cv_achieved) = distance_cv(&patterns, &held_pairs);
        let train_patterns = &patterns[..train_n];
        let flat = Palette::build(train_patterns, 256, max_iter);
        let hp = Palette::build_hierarchical(train_patterns, max_iter);
        let flat_rho = reconstruct_rho_vs_base17(&flat, &patterns, &held_pairs);
        let hier_rho = reconstruct_rho_vs_base17(&hp.leaves, &patterns, &held_pairs);
        extreme_rows.push(SweepRow {
            label,
            sep_mult,
            cv_achieved,
            flat_rho,
            hier_rho,
        });
    }
    println!("\nEXTREME-CV DIAGNOSTIC (k=2, separation pushed far past the CVtgt=1.00 row):");
    for r in &extreme_rows {
        println!(
            "  {:<10} sepMult={:>7.1} CVach={:.4}  flatRho={:.4}  hierRho={:.4}",
            r.label, r.sep_mult, r.cv_achieved, r.flat_rho, r.hier_rho
        );
    }
    let sweep_last = rows.last().unwrap();
    let extreme_last = extreme_rows.last().unwrap();
    let extreme_still_rising = extreme_last.flat_rho > sweep_last.flat_rho + 0.02
        && extreme_last.hier_rho > sweep_last.hier_rho + 0.02;
    let extreme_clears_965 = extreme_last.flat_rho >= 0.965 && extreme_last.hier_rho >= 0.965;
    println!(
        "  vs CVtgt=1.00 row (flat={:.4} hier={:.4}): still rising = {extreme_still_rising}, clears 0.965 = {extreme_clears_965}",
        sweep_last.flat_rho, sweep_last.hier_rho
    );

    // ── verdict ───────────────────────────────────────────────────────────
    println!("\nVERDICT");
    println!("=======");
    let first = rows.first().unwrap();
    let last = rows.last().unwrap();
    // Monotonic (allow small noise slack) across BOTH flat and hier.
    let flat_monotonic = rows
        .windows(2)
        .all(|w| w[1].flat_rho >= w[0].flat_rho - 0.03);
    let hier_monotonic = rows
        .windows(2)
        .all(|w| w[1].hier_rho >= w[0].hier_rho - 0.03);
    let rises = last.flat_rho > first.flat_rho + 0.05 && last.hier_rho > first.hier_rho + 0.05;
    let high_cv_clears_965 = last.flat_rho >= 0.965 && last.hier_rho >= 0.965;

    println!(
        "  low-CV end   (CVtgt={}, CVach={:.4}): flat rho={:.4} hier rho={:.4}",
        first.label, first.cv_achieved, first.flat_rho, first.hier_rho
    );
    println!(
        "  high-CV end  (CVtgt={}, CVach={:.4}): flat rho={:.4} hier rho={:.4}",
        last.label, last.cv_achieved, last.flat_rho, last.hier_rho
    );
    println!("  monotonic (± 0.03 slack): flat={flat_monotonic} hier={hier_monotonic}");
    println!("  rises meaningfully low->high CV (> 0.05 gain, both codebooks): {rises}");
    println!("  clears 0.965 at high-CV end (both codebooks): {high_cv_clears_965}");
    println!("  structure-is-free (hier >= flat within noise) holds across sweep: {structure_is_free_holds}");

    if rises && high_cv_clears_965 {
        println!(
            "\n  => CONFIRMED: rho rises with CV and clears the 0.965 anchor at the\n     \
             high-CV end. The fidelity ceiling is an INPUT-VARIANCE artifact —\n     \
             input-specific, NOT a hard limit of the Base17 fold or the codebook."
        );
    } else if !rises {
        println!(
            "\n  => FALSIFIED: rho does NOT rise meaningfully from low to high CV.\n     \
             The fold/codebook imposes a limit independent of input variance —\n     \
             a stronger, more concerning finding than the original debt entry assumed."
        );
    } else {
        println!(
            "\n  => PARTIAL: rho rises with CV (input-variance sensitivity confirmed)\n     \
             but does NOT clear the 0.965 anchor even at the high-CV end tested here.\n     \
             Either a wider/higher CV sweep or a different codebook parameter is needed\n     \
             to reach the anchor — report the numbers, do not over-claim closure."
        );
    }

    println!(
        "\n  Cross-check vs the real-data anchor (framing B, CV=0.220 -> rho=0.2599):\n  \
         at the matched CVtgt=0.22 synthetic point (framing A, CVach={:.4}), this sweep\n  \
         measures flat rho={:.4} hier rho={:.4} — framings differ (A has a codebook,\n  \
         ground truth is Base17 L1 not raw cosine; B has neither), so a strict numeric\n  \
         match is not expected. What is comparable: whether low CV depresses BOTH\n  \
         framings' rho relative to their own high-CV values.",
        rows[1].cv_achieved, rows[1].flat_rho, rows[1].hier_rho
    );
}
