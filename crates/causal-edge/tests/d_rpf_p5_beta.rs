//! D-RPF-P5 — the NARS ↔ Beta correspondence, measured on the CE64 surface.
//!
//! The live revision law has evidential horizon k = 1: a confidence code
//! `c` stands for evidence weight `w = c / (255 − c)` and revision adds
//! weights (`d_rpf_conf_0.rs` T1). Under Bernoulli evidence with a
//! Beta(k/2, k/2) prior, the NARS expectation
//! `e = c·(f − ½) + ½` equals the posterior mean `(w⁺ + k/2) / (w + k)`.
//! With k = 1 the prior is Jeffreys, Beta(½, ½).
//!
//! What this file pins, and what it does not claim:
//!
//! - **E1** the identity holds exactly in real arithmetic for k = 1 and fails
//!   for k = 2, and the f32 `expectation()` stays within its rounding.
//! - **E2** a unit observation (w = 1, c = ½) is not representable as a code;
//!   sequential revision of Bernoulli observations therefore drifts from the
//!   conjugate posterior, and the code range caps the evidence it can hold.
//! - **E3** revision cannot tell duplicated evidence from independent
//!   evidence; duplication shrinks the posterior as if it were new data.
//! - **E2b** the stall code depends on observation strength (weak evidence
//!   stalls low: observations at c = 1 stop at c = 75).
//! - **E4** a posterior variance rebuilt from `(f, c)` codes is coarse at
//!   high confidence, because successive codes are far apart in `w`.
//! - **E5** revision of finite codes never produces c = 255.
//!
//! Confidence is not a calibrated credible interval. A variance is derived
//! from the evidence model (E4), never read off the confidence code.
//! There is no decay operation on the CE64 truth fields, so decay is not
//! tested here.
#![cfg(feature = "causal-edge-v2-layout")]

use causal_edge::CausalEdge64;

fn edge(f: u8, c: u8) -> CausalEdge64 {
    let mut e = CausalEdge64(0);
    e.set_frequency_u8(f);
    e.set_confidence_u8(c);
    e
}

/// Evidence weight of a confidence code under horizon k = 1.
fn weight(c: u8) -> f64 {
    f64::from(c) / (255.0 - f64::from(c))
}

/// Beta posterior mean with prior Beta(k/2, k/2).
fn posterior_mean(w_pos: f64, w: f64, k: f64) -> f64 {
    (w_pos + k / 2.0) / (w + k)
}

/// Beta posterior variance with prior Beta(k/2, k/2).
fn posterior_var(w_pos: f64, w: f64, k: f64) -> f64 {
    let a = w_pos + k / 2.0;
    let b = w - w_pos + k / 2.0;
    a * b / ((a + b) * (a + b) * (a + b + 1.0))
}

/// NARS expectation in exact arithmetic from codes.
fn expectation_exact(f: u8, c: u8) -> f64 {
    let (f, c) = (f64::from(f) / 255.0, f64::from(c) / 255.0);
    c * (f - 0.5) + 0.5
}

// ─── E1 ─────────────────────────────────────────────────────────────────

#[test]
fn e1_expectation_is_the_jeffreys_posterior_mean() {
    let mut max_k1 = 0.0_f64;
    let mut min_k2 = f64::INFINITY;
    let mut max_f32 = 0.0_f64;
    for c in 0..255u8 {
        let w = weight(c);
        for f in 0..=255u8 {
            let w_pos = w * f64::from(f) / 255.0;
            let e = expectation_exact(f, c);
            max_k1 = max_k1.max((e - posterior_mean(w_pos, w, 1.0)).abs());
            if c > 0 && f != 128 && f != 127 {
                min_k2 = min_k2.min((e - posterior_mean(w_pos, w, 2.0)).abs());
            }
            max_f32 = max_f32.max((f64::from(edge(f, c).expectation()) - e).abs());
        }
    }
    eprintln!("E1 k=1 max {max_k1:e}  k=2 min {min_k2:e}  f32 max {max_f32:e}");
    assert!(max_k1 < 1e-12, "k = 1: e is the posterior mean");
    assert!(min_k2 > 1e-6, "k = 2 is a different prior; the test sees k");
    assert!(max_f32 < 1e-6, "f32 expectation stays within rounding");
}

// ─── E2 ─────────────────────────────────────────────────────────────────

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

/// Revise `n` Bernoulli(p) unit observations into one edge. Returns, at each
/// step, `(e_code_path, e_conjugate, c_code)`.
fn bernoulli_run(p_num: u64, n: usize, seed: u64) -> Vec<(f64, f64, u8)> {
    let unit_c = 128u8; // round(255 · ½): the nearest code to w = 1
    let mut rng = Lcg(seed);
    let mut acc: Option<CausalEdge64> = None;
    let mut successes = 0.0_f64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let hit = rng.next() % 100 < p_num;
        if hit {
            successes += 1.0;
        }
        let obs = edge(if hit { 255 } else { 0 }, unit_c);
        let next = match acc {
            None => obs,
            Some(a) => a.revision(obs),
        };
        acc = Some(next);
        let e_conj = posterior_mean(successes, (i + 1) as f64, 1.0);
        out.push((
            expectation_exact(next.frequency_u8(), next.confidence_u8()),
            e_conj,
            next.confidence_u8(),
        ));
    }
    out
}

#[test]
fn e2_unit_observation_is_not_a_code_and_the_horizon_is_bounded() {
    // w = 1 needs c = 127.5; the nearest codes carry 127/128 and 128/127.
    assert!((weight(127) - 1.0).abs() > 0.007 && (weight(128) - 1.0).abs() > 0.007);

    let runs: Vec<Vec<(f64, f64, u8)>> = (0..32u64).map(|s| bernoulli_run(70, 400, s)).collect();
    for n in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 400] {
        let mut max_d = 0.0_f64;
        let mut sum_d = 0.0_f64;
        let mut cmin = u8::MAX;
        let mut cmax = 0u8;
        for r in &runs {
            let (e, ec, c) = r[n - 1];
            let d = (e - ec).abs();
            max_d = max_d.max(d);
            sum_d += d;
            cmin = cmin.min(c);
            cmax = cmax.max(c);
        }
        eprintln!(
            "E2 n={n:4}  mean|Δe| {:.4}  max|Δe| {:.4}  c in {cmin}..={cmax}",
            sum_d / runs.len() as f64,
            max_d
        );
    }
    // First step at which any run's confidence code stops rising.
    let mut first_stall = usize::MAX;
    for r in &runs {
        for i in 1..r.len() {
            if r[i].2 <= r[i - 1].2 {
                first_stall = first_stall.min(i + 1);
                break;
            }
        }
    }
    eprintln!("E2 first confidence stall at n = {first_stall}");
    // Measured: every run's confidence stops at code 244 (w ≈ 22.2); the
    // first stall is at n = 21. From there the code path forgets old
    // evidence like a fixed window while the conjugate posterior keeps
    // narrowing, so e drifts.
    assert_eq!(first_stall, 21);
    assert!(runs.iter().all(|r| r[399].2 == 244));
    let mean_d_256: f64 = runs
        .iter()
        .map(|r| (r[255].0 - r[255].1).abs())
        .sum::<f64>()
        / runs.len() as f64;
    let mean_d_8: f64 =
        runs.iter().map(|r| (r[7].0 - r[7].1).abs()).sum::<f64>() / runs.len() as f64;
    assert!(
        mean_d_8 < 0.002,
        "before the stall the code path tracks the posterior"
    );
    assert!(mean_d_256 > 0.05, "after the stall it does not");
}

// ─── E3 ─────────────────────────────────────────────────────────────────

#[test]
fn e3_duplicated_evidence_is_counted_as_new() {
    let one = edge(255, 128);
    let mut dup = one;
    for k in 2..=8u32 {
        dup = dup.revision(one);
        let w = weight(dup.confidence_u8());
        let var_claimed = posterior_var(w, w, 1.0);
        let w1 = weight(128);
        let var_true = posterior_var(w1, w1, 1.0);
        let ratio = (var_true / var_claimed).sqrt();
        eprintln!(
            "E3 copies {k}: c {}  w {:.3}  sd claimed {:.4}  sd true {:.4}  ratio {:.3}",
            dup.confidence_u8(),
            w,
            var_claimed.sqrt(),
            var_true.sqrt(),
            ratio
        );
        // Four copies of one observation claim twice the precision it has.
        if k == 4 {
            assert!((2.0..2.1).contains(&ratio), "k = 4 ratio {ratio}");
        }
    }
    // Independent and duplicated evidence are indistinguishable to revision:
    // the operands carry no provenance, so this is by construction.
    assert_eq!(one.revision(one), edge(255, 128).revision(edge(255, 128)));
}

// ─── E4 ─────────────────────────────────────────────────────────────────

#[test]
fn e4_variance_from_codes_is_coarse_at_high_confidence() {
    for c in [64u8, 128, 192, 224, 240, 248, 252, 253] {
        let w = weight(c);
        let w_next = weight(c + 1);
        let v = posterior_var(0.7 * w, w, 1.0);
        let v_next = posterior_var(0.7 * w_next, w_next, 1.0);
        let step = 1.0 - (v_next / v).sqrt();
        eprintln!(
            "E4 c {c:3}: w {w:8.3} → {w_next:8.3}  sd {:.4} → {:.4}  step {:.1}%",
            v.sqrt(),
            v_next.sqrt(),
            100.0 * step
        );
        if c == 128 {
            assert!(step < 0.005, "mid-range: one code is a fine step");
        }
        if c == 253 {
            assert!(step > 0.25, "top of range: one code moves sd by > 25%");
        }
    }
}

// ─── E2b: the quantization fixed point ──────────────────────────────────

/// Repeatedly revise an edge with observations of confidence code `obs_c`;
/// return the code where confidence stops rising and the step it got there.
fn stall_code(obs_c: u8) -> (u8, u32) {
    let obs = edge(255, obs_c);
    let mut acc = obs;
    for step in 2..10_000u32 {
        let next = acc.revision(obs);
        if next.confidence_u8() <= acc.confidence_u8() {
            return (acc.confidence_u8(), step - 1);
        }
        acc = next;
    }
    (acc.confidence_u8(), 10_000)
}

/// The horizon depends on the strength of each observation: weak evidence
/// stalls at low confidence. Measured and pinned.
#[test]
fn e2b_the_stall_code_depends_on_observation_strength() {
    const PINNED: [(u8, u8, u32); 12] = [
        (1, 75, 75),
        (8, 193, 87),
        (32, 225, 48),
        (64, 236, 34),
        (96, 241, 26),
        (128, 244, 20),
        (160, 247, 16),
        (192, 249, 12),
        (224, 251, 8),
        (240, 252, 5),
        (250, 254, 4),
        (254, 254, 1),
    ];
    for (obs, stall, after) in PINNED {
        assert_eq!(stall_code(obs), (stall, after), "obs c {obs}");
    }
}

/// Revision of two finite codes never reaches 255: the maximum, (254, 254),
/// is 254.499… and rounds to 254. So the singular cell (G1) is entered only
/// by writing 255 directly, never by accumulating evidence.
#[test]
fn e5_revision_never_reaches_the_singular_code() {
    let mut max = 0u8;
    for c1 in 0..255u8 {
        for c2 in 0..255u8 {
            max = max.max(edge(128, c1).revision(edge(128, c2)).confidence_u8());
        }
    }
    assert_eq!(max, 254);
}
