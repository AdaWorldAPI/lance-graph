//! **PROBE-ENTROPY-SURFACE-CENSUS-1** — the gate D-DCR-4 records as
//! "entropy-surface CONSOLIDATION decision recorded first" (plan
//! `dismech-causal-replay-v1.md`).
//!
//! Seven Shannon-entropy surfaces exist in this tree with four different
//! conventions. This probe measures which of them are the SAME estimator
//! and which are not, so the consolidation is decided by a measurement
//! instead of a code read (the D-TEH-3 method: the lift gate is run on a
//! fixture that can distinguish the implementations —
//! `E-JC-IS-THE-HOME-OF-ALL-CALIBRATED-MATH-1`).
//!
//! It changes no library code and lands no consolidation.
//!
//! ```text
//! cargo run -p lance-graph-planner --example entropy_surface_census
//! ```
//!
//! # Live vs. transcribed
//!
//! Form A (`lance_graph_contract::thought_atoms::normalized_entropy`) is
//! IMPORTED and called live — the planner already depends on the contract
//! crate.
//!
//! Forms B through G are **transcribed**, not imported: `B` is a private
//! function inside `lance-graph-planner` itself (transcribed rather than
//! made `pub` — this probe must not change library visibility); `C` lives
//! in `cognitive-shader-driver`; `D` and `E` live in `lance-graph-cognitive`;
//! `F` and `G` live in `thinking-engine`. The last three crates are
//! EXCLUDED from the workspace and the planner does not depend on any of
//! them, so their forms cannot be imported here — each is copied
//! byte-for-byte into [`mod transcribed`] with the exact `file:line` it
//! came from in a doc comment above it, so a future session can diff the
//! transcription against the source. **These are transcriptions, not the
//! live functions — the census is therefore only as current as those
//! citations.** If a cited source function changes, this probe's numbers
//! for that form go stale until re-transcribed.

use lance_graph_contract::thought_atoms::normalized_entropy;

/// The caller's source, pulled in at COMPILE time so the caller census is a
/// checked fact rather than a sentence. If `insight.rs` moves, this breaks the
/// build — which is the point: a census that cannot fail is not a census.
const INSIGHT_SRC: &str = include_str!("../src/nars/insight.rs");

/// Call sites of form A inside the caller's source. Counts `normalized_entropy(`
/// — the `use` line and every doc-comment mention lack the paren, so they do not
/// inflate it, and this comment lives in a different file so it cannot self-match.
fn production_call_sites() -> usize {
    INSIGHT_SRC.matches("normalized_entropy(").count()
}

// ═══════════════════════════════════════════════════════════════════════
// The convention table (printed verbatim, then measured against it)
// ═══════════════════════════════════════════════════════════════════════

const CONVENTION_TABLE: &str = "\
| id | form                                                | normalizes input | base    | normalized output | empty | zero-guard   |
|----|------------------------------------------------------|-------------------|---------|--------------------|-------|--------------|
| A  | contract::thought_atoms::normalized_entropy (LIVE)   | yes \u{f7}\u{3a3}w         | ln      | yes \u{f7}ln(n)          | None  | p > 0.0      |
| B  | planner insight::confidence_entropy (math only)      | yes \u{f7}n          | log2    | yes \u{f7}log2(bins)     | 0.0   | c > 0        |
| C  | cognitive-shader-driver entropy_std (entropy half)   | yes \u{f7}\u{3a3}          | ln      | NO                 | 0.0   | p > 1e-9     |
| D  | lance-graph-cognitive distribution::entropy          | yes \u{f7}total      | log2    | NO                 | 0.0   | count > 0    |
| E  | lance-graph-cognitive features::compute_entropy      | yes \u{f7}total      | ln (f64)| yes \u{f7}ln(W)          | 0.0   | p > 0        |
| F  | thinking-engine qualia::shannon_entropy              | NO                | ln      | NO                 | 0.0   | e > 1e-10    |
| G  | thinking-engine dto::entropy                         | NO                | ln      | NO                 | 0.0   | e > 1e-10    |";

// ═══════════════════════════════════════════════════════════════════════
// mod transcribed — byte-for-byte copies of unimportable/private sources
// ═══════════════════════════════════════════════════════════════════════

/// Transcriptions of six of the seven entropy surfaces (form A is imported
/// live above). Each function's doc comment cites the exact `file:line`
/// range it was copied from at the time this probe was written. **These
/// are NOT the live functions** — `lance-graph-cognitive` and
/// `thinking-engine` are excluded from the workspace, and `confidence_entropy`
/// is a private `fn` in `lance-graph-planner::nars::insight`, so none of
/// the three could be imported. Diff this module against its cited sources
/// before trusting a stale run of this probe.
mod transcribed {
    /// **Form B** — transcribed from
    /// `crates/lance-graph-planner/src/nars/insight.rs:176-198`
    /// (`fn confidence_entropy(arena: &BeliefArena) -> f32`), verified at
    /// that range on read **as of `a1c9488e`**.
    ///
    /// ⊘ **SUPERSEDED IN THE TREE, DELIBERATELY PRESERVED HERE.** The
    /// consolidation this census gated replaced that implementation: the live
    /// `confidence_entropy` now delegates to form A and no longer builds its
    /// own histogram or runs its own log2 loop. This transcription is
    /// therefore a FOSSIL, and it is kept rather than re-transcribed because
    /// C2's falsification result is a statement about the OLD B — re-pointing
    /// it at the new code would silently convert a measured falsification into
    /// a comparison of form A against itself. Read every B row below as
    /// "B as of `a1c9488e`", never as "the planner's current entropy".
    ///
    /// Source shape: builds a fixed `[0usize; 10]` histogram from
    /// `BeliefArena` confidences, then computes normalized Shannon entropy
    /// (log2, ÷ log2(BINS)) over the histogram, normalizing counts by
    /// `n = arena.entries().len()` (the total entry count, i.e. the sum of
    /// the histogram) rather than by the histogram's own length.
    ///
    /// **Generalization (the ONLY change from the source):** the source
    /// takes a `&BeliefArena` and builds a fixed-width `[usize; 10]`
    /// histogram internally; this transcription instead takes the
    /// already-built histogram directly as a `&[f32]` slice (so it can run
    /// over this probe's shared fixture battery) and normalizes by
    /// `hist.len()` in place of the source's own constant bin width
    /// (`BINS = 10`) — i.e. `slice.len()` stands in for `BINS`, and
    /// `slice.iter().sum()` stands in for the source's `n` (which, in the
    /// source, is always exactly the sum of `hist` by construction, since
    /// every entry increments exactly one bin). The zero-guard (`c > 0` in
    /// the source, `count > 0.0` in the fixture-generalized f32 form) and
    /// the log2/÷log2 arithmetic are otherwise identical, in the same
    /// order.
    pub fn b_confidence_entropy(hist: &[f32]) -> f32 {
        if hist.is_empty() {
            return 0.0;
        }
        let n: f32 = hist.iter().sum();
        let mut h = 0.0f32;
        for &count in hist {
            if count > 0.0 {
                let p = count / n;
                h -= p * p.log2();
            }
        }
        h / (hist.len() as f32).log2() // normalize to [0, 1]
    }

    /// **Form C** — transcribed from
    /// `crates/cognitive-shader-driver/src/driver.rs:921-940`
    /// (`fn entropy_std(hits: &[ShaderHit]) -> (f32, f32)`), verified at
    /// that range on read. Per the worker brief, only the ENTROPY half is
    /// transcribed — the source also returns a std-dev of `h.resonance`,
    /// dropped here since it is not a Shannon-entropy form.
    ///
    /// Source shape: `sum = Σ h.resonance`; if `sum <= 0.0` return `0.0`;
    /// else for each hit, `p = resonance / sum`, accumulate `-p * p.ln()`
    /// when `p > 1e-9`. No further normalization (no ÷ln(n)).
    ///
    /// **Generalization:** the source's `&[ShaderHit]` (read via
    /// `h.resonance`) is replaced by a plain `&[f32]` of the same values —
    /// no arithmetic or guard changes.
    pub fn c_entropy(weights: &[f32]) -> f32 {
        if weights.is_empty() {
            return 0.0;
        }
        let sum: f32 = weights.iter().sum();
        if sum <= 0.0 {
            return 0.0;
        }
        let mut ent = 0.0f32;
        for &w in weights {
            let p = w / sum;
            if p > 1e-9 {
                ent -= p * p.ln();
            }
        }
        ent
    }

    /// **Form D** — transcribed from
    /// `crates/lance-graph-cognitive/src/search/distribution.rs:158-169`
    /// (`pub fn entropy(&self) -> f32` on `ClusterDistribution`, reading
    /// `self.histogram_int4: [u16; 16]`), verified at that range on read.
    ///
    /// Source shape: `total = Σ histogram_int4` (as f32); if `total == 0.0`
    /// return `0.0`; else, filtering to `count > 0`, sum
    /// `-p * p.log2()` where `p = count / total`. No further normalization.
    ///
    /// **Generalization:** the source's fixed `[u16; 16]` histogram is
    /// replaced by a `&[f32]` slice (the fixture values stand in for the
    /// counts) and the `count > 0` guard becomes `count > 0.0`; the
    /// arithmetic and order are otherwise identical.
    pub fn d_entropy(hist: &[f32]) -> f32 {
        let total: f32 = hist.iter().sum();
        if total == 0.0 {
            return 0.0;
        }
        hist.iter()
            .filter(|&&count| count > 0.0)
            .map(|&count| {
                let p = count / total;
                -p * p.log2()
            })
            .sum()
    }

    /// **Form E** — transcribed from
    /// `crates/lance-graph-cognitive/src/spectroscopy/features.rs:95-115`
    /// (`fn compute_entropy(pops: &[u32; CONTAINER_WORDS]) -> f32`),
    /// verified at that range on read.
    ///
    /// Source shape: sums `pops` into an `f64` total; if `total == 0.0`
    /// return `0.0`; else, for `p > 0`, accumulate `-prob * prob.ln()` in
    /// `f64` where `prob = p as f64 / total`; finally divide by
    /// `max_entropy = (CONTAINER_WORDS as f64).ln()` (guarded `> 0.0`,
    /// else `0.0`) and cast back to `f32`.
    ///
    /// **Generalization (the ONLY change from the source):** the source's
    /// fixed `&[u32; CONTAINER_WORDS]` becomes a `&[f32]` slice, and the
    /// normalizer `CONTAINER_WORDS` (the array's fixed const width) becomes
    /// `pops.len()` (the slice's own width, `W`) — the f64 accumulation,
    /// the `p > 0` guard (as `p > 0.0`), and the max-entropy guard are
    /// otherwise identical, in the same order.
    pub fn e_compute_entropy(pops: &[f32]) -> f32 {
        let total: f64 = pops.iter().map(|&p| p as f64).sum();
        if total == 0.0 {
            return 0.0;
        }

        let mut entropy: f64 = 0.0;
        for &p in pops.iter() {
            if p > 0.0 {
                let prob = p as f64 / total;
                entropy -= prob * prob.ln();
            }
        }

        // Normalise by maximum possible entropy (uniform distribution).
        let max_entropy = (pops.len() as f64).ln();
        if max_entropy > 0.0 {
            (entropy / max_entropy) as f32
        } else {
            0.0
        }
    }

    /// **Form F** — transcribed from `crates/thinking-engine/src/qualia.rs:688-695`
    /// (`fn shannon_entropy(energy: &[f32]) -> f32`), verified at that
    /// range on read, byte-for-byte (no generalization needed — the source
    /// already takes a `&[f32]`).
    ///
    /// Source shape: no normalization of the input at all (raw `e` values
    /// are used directly, not divided by any sum); for `e > 1e-10`,
    /// accumulate `-e * e.ln()`. No output normalization. An empty slice
    /// returns `0.0` implicitly (the loop never executes, `h` stays at its
    /// initial `0.0`).
    pub fn f_shannon_entropy(energy: &[f32]) -> f32 {
        let mut h = 0.0f32;
        for &e in energy {
            if e > 1e-10 {
                h -= e * e.ln();
            }
        }
        h
    }

    /// **Form G** — transcribed from `crates/thinking-engine/src/dto.rs:113-120`
    /// (`pub fn entropy(&self) -> f32` on the thought-DTO, reading
    /// `self.energy: Vec<f32>`), verified at that range on read,
    /// byte-for-byte (no generalization needed — the source loop already
    /// takes plain `f32` values via `&self.energy`).
    ///
    /// Source shape is IDENTICAL to form F's: no input normalization, for
    /// `e > 1e-10` accumulate `-e * e.ln()`, no output normalization,
    /// empty slice implicitly `0.0`.
    pub fn g_entropy(energy: &[f32]) -> f32 {
        let mut h = 0.0f32;
        for &e in energy {
            if e > 1e-10 {
                h -= e * e.ln();
            }
        }
        h
    }
}

use transcribed::{
    b_confidence_entropy, c_entropy, d_entropy, e_compute_entropy, f_shannon_entropy, g_entropy,
};

// ═══════════════════════════════════════════════════════════════════════
// Fixture battery
// ═══════════════════════════════════════════════════════════════════════

const FIXTURES: &[(&str, &[f32])] = &[
    ("uniform4", &[1.0, 1.0, 1.0, 1.0]),
    ("uniform4_x10", &[10.0, 10.0, 10.0, 10.0]), // same distribution, 10x the mass
    ("peaked4", &[1.0, 0.0, 0.0, 0.0]),
    ("two_point", &[3.0, 1.0]),
    ("already_prob", &[0.25, 0.25, 0.25, 0.25]), // sums to 1 exactly
    ("tiny_mass", &[1e-10, 1.0]),                // separates the p>0 / p>1e-9 / e>1e-10 cutoffs
    ("single", &[5.0]),
    ("all_zero", &[0.0, 0.0, 0.0]),
    ("empty", &[]),
];

/// The log2 twin of form A, written inline for claim C1: same guards
/// (`p > 0.0`), same ÷ln(n)-shaped normalization but with `log2` in both
/// numerator and denominator, over the exact same `n == 1` / `sum <= 0.0`
/// special cases as `normalized_entropy` (mirrored here rather than
/// imported, since the point of C1 is to show the base is inert under
/// normalization, not to reuse the base-ln implementation).
fn a_log2_twin(weights: &[f32]) -> Option<f32> {
    match weights.len() {
        0 => None,
        1 => Some(0.0),
        n => {
            let sum: f32 = weights.iter().sum();
            if sum <= 0.0 {
                return Some(1.0);
            }
            let h: f32 = weights
                .iter()
                .map(|&w| {
                    let p = w / sum;
                    if p > 0.0 {
                        -p * p.log2()
                    } else {
                        0.0
                    }
                })
                .sum();
            Some((h / (n as f32).log2()).clamp(0.0, 1.0))
        }
    }
}

fn fmt_opt(v: Option<f32>) -> String {
    match v {
        Some(x) => format!("{x:.6}"),
        None => "None".to_string(),
    }
}

fn main() {
    println!("PROBE-ENTROPY-SURFACE-CENSUS-1 (D-DCR-4 gate)\n");
    println!("{CONVENTION_TABLE}\n");

    // ── per-fixture, per-form values ──────────────────────────────────
    println!("per-fixture values (A..G), {{:.6}} or \"None\":\n");
    println!(
        "{:<14} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "fixture", "A", "B", "C", "D", "E", "F", "G"
    );

    // Cache A's values per fixture (needed for C1/C2 anti-vacuity + spread).
    let mut a_vals: Vec<Option<f32>> = Vec::with_capacity(FIXTURES.len());
    let mut b_vals: Vec<f32> = Vec::with_capacity(FIXTURES.len());
    let mut c_vals: Vec<f32> = Vec::with_capacity(FIXTURES.len());
    let mut d_vals: Vec<f32> = Vec::with_capacity(FIXTURES.len());
    let mut e_vals: Vec<f32> = Vec::with_capacity(FIXTURES.len());
    let mut f_vals: Vec<f32> = Vec::with_capacity(FIXTURES.len());
    let mut g_vals: Vec<f32> = Vec::with_capacity(FIXTURES.len());

    for &(name, xs) in FIXTURES {
        let a = normalized_entropy(xs);
        let b = b_confidence_entropy(xs);
        let c = c_entropy(xs);
        let d = d_entropy(xs);
        let e = e_compute_entropy(xs);
        let f = f_shannon_entropy(xs);
        let g = g_entropy(xs);
        println!(
            "{:<14} {:>10} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
            name,
            fmt_opt(a),
            b,
            c,
            d,
            e,
            f,
            g
        );
        a_vals.push(a);
        b_vals.push(b);
        c_vals.push(c);
        d_vals.push(d);
        e_vals.push(e);
        f_vals.push(f);
        g_vals.push(g);
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // CLAIM C1 — normalization makes the log base inert
    // ═══════════════════════════════════════════════════════════════
    println!("CLAIM C1 — normalization makes the log base inert");
    let mut c1_pass = true;
    let mut c1_two_point_a = None;
    let mut c1_two_point_twin = None;
    for (i, &(name, xs)) in FIXTURES.iter().enumerate() {
        let non_zero = xs.iter().filter(|&&w| w > 0.0).count();
        if non_zero < 2 {
            continue;
        }
        let a = a_vals[i];
        let twin = a_log2_twin(xs);
        let (Some(a), Some(twin)) = (a, twin) else {
            c1_pass = false;
            continue;
        };
        let diff = (a - twin).abs();
        let ok = diff <= 1e-5;
        c1_pass &= ok;
        println!(
            "  {name:<14} A={a:.6} log2-twin={twin:.6} |diff|={diff:.8} {}",
            if ok { "ok" } else { "MISMATCH" }
        );
        if name == "two_point" {
            c1_two_point_a = Some(a);
            c1_two_point_twin = Some(twin);
        }
    }
    // ANTI-VACUITY: two_point must not be trivially 0.0 or 1.0 for either twin.
    let anti_vacuity_c1 = match (c1_two_point_a, c1_two_point_twin) {
        (Some(a), Some(twin)) => {
            let non_degenerate = |v: f32| (v - 0.0).abs() > 1e-6 && (v - 1.0).abs() > 1e-6;
            non_degenerate(a) && non_degenerate(twin)
        }
        _ => false,
    };
    println!(
        "  anti-vacuity (two_point not 0.0/1.0 for A or its twin): {}",
        if anti_vacuity_c1 { "ok" } else { "FAIL" }
    );
    let c1_verdict = c1_pass && anti_vacuity_c1;
    println!(
        "  C1 verdict: {}\n",
        if c1_verdict { "PASS" } else { "FAIL" }
    );
    assert!(
        c1_pass,
        "C1: A and its log2 twin disagree beyond 1e-5 on some fixture"
    );
    assert!(
        anti_vacuity_c1,
        "C1 anti-vacuity: two_point's A/twin values are trivially 0.0 or 1.0"
    );

    // ═══════════════════════════════════════════════════════════════
    // CLAIM C2 — B is A  ⊘ FALSIFIED AS PRE-REGISTERED (2026-09-03)
    // ═══════════════════════════════════════════════════════════════
    //
    // Pre-registered: "on every non-empty fixture, |B - A| <= 1e-5". The
    // measurement REFUTED it, and the refutation is this probe's finding —
    // the claim is restated to what was measured, never relaxed to pass:
    //
    //   C2a  same estimator on NON-DEGENERATE input (len >= 2, mass > 0):
    //        measured |B - A| = 0.0 exactly on all six such fixtures.
    //   C2b  OPPOSITE conventions on ZERO MASS: A = 1.0 ("zero-sum = uniform",
    //        thought_atoms.rs `normalized_entropy_hand_math`), B = 0.0.
    //        The gap is 1.0 — the entire range of a normalized entropy.
    //   C2c  `empty`: A = None, B = 0.0.
    //   C2d  `single`: B is NaN — an ARTIFACT of this probe's length
    //        generalization (the source's BINS is a const 10 and can never
    //        be 1), pinned here so it cannot change silently. Not a defect
    //        in `confidence_entropy`.
    //
    // Consequence, which is why the gate existed: routing B through A is NOT
    // a drop-in. A code read says "same formula"; the zero-mass case inverts.
    println!("CLAIM C2 — B is A  [PRE-REGISTERED FORM FALSIFIED — see C2a..C2d]");
    let mut c2a_pass = true;
    for (i, &(name, xs)) in FIXTURES.iter().enumerate() {
        if name == "empty" || name == "single" || name == "all_zero" {
            continue; // degenerate: covered by C2b/C2c/C2d below
        }
        let _ = xs;
        let a = a_vals[i];
        let b = b_vals[i];
        match a {
            Some(a) => {
                let diff = (b - a).abs();
                let ok = diff <= 1e-5;
                c2a_pass &= ok;
                println!(
                    "  C2a {name:<14} A={a:.6} B={b:.6} |diff|={diff:.8} {}",
                    if ok { "ok" } else { "MISMATCH" }
                );
            }
            None => {
                c2a_pass = false;
                println!("  C2a {name:<14} A=None (unexpected here) B={b:.6} MISMATCH");
            }
        }
    }

    let idx_of = |want: &str| {
        FIXTURES
            .iter()
            .position(|&(n, _)| n == want)
            .expect("fixture present")
    };

    // C2b — the divergence, asserted two-sided so a future alignment also fails.
    let az = idx_of("all_zero");
    let a_az = a_vals[az].expect("A returns Some on all_zero (zero-sum = uniform)");
    let b_az = b_vals[az];
    let az_gap = (a_az - b_az).abs();
    let c2b_pass = az_gap >= 0.99 && a_az > 0.99 && b_az.abs() < 1e-6;
    println!(
        "  C2b all_zero      A={a_az:.6} B={b_az:.6} |gap|={az_gap:.6} (OPPOSITE conventions) {}",
        if c2b_pass { "ok" } else { "FAIL" }
    );

    // C2c — the empty convention.
    let em = idx_of("empty");
    let a_empty = a_vals[em];
    let b_empty = b_vals[em];
    let c2c_pass = a_empty.is_none() && b_empty.abs() < 1e-9;
    println!(
        "  C2c empty         A={} B={:.6} {}",
        fmt_opt(a_empty),
        b_empty,
        if c2c_pass { "ok" } else { "FAIL" }
    );

    // C2d — the generalization artifact, pinned.
    let sg = idx_of("single");
    let b_single = b_vals[sg];
    let c2d_pass = b_single.is_nan();
    println!(
        "  C2d single        A={} B={b_single:.6} (probe artifact: log2(1)=0 denominator) {}",
        fmt_opt(a_vals[sg]),
        if c2d_pass { "ok" } else { "FAIL" }
    );

    // ANTI-VACUITY: A itself varies across the non-degenerate fixtures.
    let non_degenerate_a: Vec<f32> = FIXTURES
        .iter()
        .zip(a_vals.iter())
        .filter(|(&(name, _), _)| name != "empty" && name != "single" && name != "all_zero")
        .filter_map(|(_, &v)| v)
        .collect();
    let a_max = non_degenerate_a.iter().cloned().fold(f32::MIN, f32::max);
    let a_min = non_degenerate_a.iter().cloned().fold(f32::MAX, f32::min);
    let a_spread = a_max - a_min;
    let anti_vacuity_c2 = a_spread >= 0.5;
    println!(
        "  anti-vacuity (A varies: max={a_max:.6} min={a_min:.6} spread={a_spread:.6} >= 0.5): {}",
        if anti_vacuity_c2 { "ok" } else { "FAIL" }
    );
    let c2_verdict = c2a_pass && c2b_pass && c2c_pass && c2d_pass && anti_vacuity_c2;
    println!(
        "  C2 verdict (restated form): {}\n",
        if c2_verdict { "PASS" } else { "FAIL" }
    );

    // ═══════════════════════════════════════════════════════════════
    // CLAIM C3 — F and G are not entropies of a distribution
    // ═══════════════════════════════════════════════════════════════
    println!("CLAIM C3 — F and G are not entropies of a distribution (mass-invariance)");
    let u4_idx = FIXTURES
        .iter()
        .position(|&(n, _)| n == "uniform4")
        .expect("uniform4 present");
    let u4x10_idx = FIXTURES
        .iter()
        .position(|&(n, _)| n == "uniform4_x10")
        .expect("uniform4_x10 present");

    let a_delta = match (a_vals[u4_idx], a_vals[u4x10_idx]) {
        (Some(x), Some(y)) => (x - y).abs(),
        _ => f32::NAN,
    };
    let b_delta = (b_vals[u4_idx] - b_vals[u4x10_idx]).abs();
    let c_delta = (c_vals[u4_idx] - c_vals[u4x10_idx]).abs();
    let d_delta = (d_vals[u4_idx] - d_vals[u4x10_idx]).abs();
    let e_delta = (e_vals[u4_idx] - e_vals[u4x10_idx]).abs();
    let f_delta = (f_vals[u4_idx] - f_vals[u4x10_idx]).abs();
    let g_delta = (g_vals[u4_idx] - g_vals[u4x10_idx]).abs();

    println!("  measured |f(uniform4) - f(uniform4_x10)| per form:");
    println!("    A={a_delta:.6}  B={b_delta:.6}  C={c_delta:.6}  D={d_delta:.6}  E={e_delta:.6}  F={f_delta:.6}  G={g_delta:.6}");

    let normalizing_invariant =
        a_delta <= 1e-5 && b_delta <= 1e-5 && c_delta <= 1e-5 && d_delta <= 1e-5 && e_delta <= 1e-5;
    let f_g_vary = f_delta > 1.0 && g_delta > 1.0;
    println!(
        "  A,B,C,D,E invariant (each <= 1e-5) [ANTI-VACUITY GUARD]: {}",
        if normalizing_invariant { "ok" } else { "FAIL" }
    );
    println!(
        "  F,G vary (each > 1.0): {}",
        if f_g_vary { "ok" } else { "FAIL" }
    );
    let c3_verdict = normalizing_invariant && f_g_vary;
    println!(
        "  C3 verdict: {}\n",
        if c3_verdict { "PASS" } else { "FAIL" }
    );
    assert!(
        normalizing_invariant,
        "C3 anti-vacuity: a normalizing form (A/B/C/D/E) was NOT mass-invariant — \
         without this guard the F/G-vary assertion alone would pass even if nothing in the tree is invariant"
    );
    assert!(
        f_g_vary,
        "C3: F and/or G did not vary by > 1.0 between uniform4 and uniform4_x10 — \
         they would then behave like a normalizing entropy, contradicting the table"
    );

    // ═══════════════════════════════════════════════════════════════
    // Observations (measured, NOT asserted)
    // ═══════════════════════════════════════════════════════════════
    println!("OBSERVATIONS (measured, not claims — no assertions below this line)\n");

    let tm_idx = FIXTURES
        .iter()
        .position(|&(n, _)| n == "tiny_mass")
        .expect("tiny_mass present");
    let tm_a = a_vals[tm_idx];
    let tm_b = b_vals[tm_idx];
    let tm_c = c_vals[tm_idx];
    let tm_d = d_vals[tm_idx];
    let tm_e = e_vals[tm_idx];
    let tm_f = f_vals[tm_idx];
    let tm_g = g_vals[tm_idx];
    let tm_all: Vec<f32> = [tm_c, tm_d, tm_e, tm_f, tm_g]
        .into_iter()
        .chain(tm_a)
        .chain(std::iter::once(tm_b))
        .collect();
    let tm_max = tm_all.iter().cloned().fold(f32::MIN, f32::max);
    let tm_min = tm_all.iter().cloned().fold(f32::MAX, f32::min);
    println!(
        "  tiny_mass spread across all seven forms (cutoff disagreement p>0.0 vs p>1e-9 vs e>1e-10):"
    );
    println!(
        "    A={} B={:.6} C={:.6} D={:.6} E={:.6} F={:.6} G={:.6}",
        fmt_opt(tm_a),
        tm_b,
        tm_c,
        tm_d,
        tm_e,
        tm_f,
        tm_g
    );
    println!(
        "    max-min spread = {:.9} (expected ~1e-9 and therefore negligible in f32 — reported, not dramatized)",
        tm_max - tm_min
    );

    // The caller census is a MEASUREMENT, so it is ASSERTED, not printed.
    //
    // Two earlier revisions of this block were prose. The first claimed "ZERO
    // callers ... verified by a grep returning nothing" and was wrong twice
    // over: the consolidation it was written to motivate gave form A a
    // production caller, AND the grep had never returned nothing, because THIS
    // FILE imports form A and calls it. The second revision added the commit
    // and the command — real provenance — but was still a bare `println!`, so
    // nothing failed when the count changed. Provenance is not falsifiability.
    //
    // The fix is to source the claim from the tree at COMPILE time and assert
    // it at run time: `PRODUCTION_CALLERS` below breaks the build's own probe
    // the moment a call site is added or removed. Note the self-reference this
    // block cannot escape — the printed command's own text matches the pattern
    // it prints, so a reader re-running it by hand sees hits generated by this
    // very comment. That is why the machine-checked count reads the CALLER's
    // source rather than counting grep lines.
    let production_callers = production_call_sites();
    println!(
        "\n  caller census (ASSERTED below, not merely printed): form A has {} \
         production call site(s) in `nars::insight` — the consolidation this \
         census gated. Before it the count was zero, which is the vacancy the \
         census was run to establish. A by-hand grep also returns this file's \
         import, doc comments, and THIS COMMENT's own copy of the pattern; the \
         asserted number below counts call sites in the caller's source and is \
         immune to that self-match.",
        production_callers
    );

    // ── every claim has now RUN; assert at the very end so no later claim
    // ── is hidden behind an earlier panic (a probe that aborts early hides
    // ── evidence — that is how C3 went unmeasured on the first run).
    assert_eq!(
        production_callers, 1,
        "caller census: form A must have exactly ONE production call site in \
         nars::insight (found {production_callers}). This is the assertion the \
         two prose revisions of this block lacked — if a call site was added or \
         removed, re-measure and re-pin deliberately rather than silencing it."
    );
    assert!(c2a_pass, "C2a: B and A disagree on non-degenerate input");
    assert!(
        c2b_pass,
        "C2b: A and B no longer carry OPPOSITE zero-mass conventions — \
         re-measure and re-pin deliberately, do not silence this"
    );
    assert!(c2c_pass, "C2c: empty convention changed (A=None, B=0.0)");
    assert!(
        c2d_pass,
        "C2d: the length-1 generalization artifact changed (B was NaN)"
    );
    assert!(
        anti_vacuity_c2,
        "C2 anti-vacuity: A does not vary (spread < 0.5)"
    );
    println!("\nC1 PASS · C2 RESTATED (pre-registered form falsified) · C3 PASS.");
}
