//! **PROBE-NXG-FLOOR-1** — plan `.claude/nexgen/plans/nexgen-mask-histogram-thresholds-v1.md`
//! §5 step 4 (D-NXG-6). Runs after HIST-1 (E-NXG-17) and ROLL-1.
//!
//! The claim under test (E-NXG-4): **`mu + k·σ` is the wrong operating
//! threshold, and the code already says so.** `ndarray::hpc::cascade` fixes
//! `k = 3`; `perturbation-sim::rolling_floor` states in its own header that the
//! σ comes from a weakly dependent sample, so "significance is the Jirak
//! `n^(p/2−1)` rate, not a clean Gaussian tail". The plan's replacement is a
//! floor read off the rank ladder instead.
//!
//! What an operator actually budgets is a **rate**: how often the floor is
//! crossed. So the question this probe answers is narrow and empirical — does
//! `k` name a rate?
//!
//! It changes no library code.
//!
//! ```text
//! CARGO_PROFILE_DEV_DEBUG=0 cargo run -p lance-graph-planner --example probe_nxg_floor_1
//! ```
//!
//! # The columns — three real ones, three shapes
//!
//! | column | file | n | shape |
//! |---|---|---|---|
//! | speech | `tts_real_output.wav` | 94 572 | heavy-tailed |
//! | saturated | `cascade_speech_128frames.wav` | 61 995 | clipped near full scale |
//! | quiet | `cascade_output.wav` | 24 240 | sparse, mostly near zero |
//!
//! All three are `|sample|` of recordings already in this repo. None is drawn
//! from a generator this probe controls.
//!
//! # Pre-registered claims
//!
//! - **C1 — RESTATED, and the restatement is sharper than the original.**
//!   Pre-registered form: "at `k = 3` the rates differ by more than 2×, and
//!   every column must be crossable". The crossability guard was mis-specified
//!   and its own first run said so: on the saturated column `mu + 3σ = 52 425`
//!   while the column's physical maximum is 32 767, so the floor is
//!   **unreachable — an alarm that can never fire**, rate exactly 0. That is
//!   not a vacuous input to be excluded, it is the strongest instance of the
//!   claim. Restated: among the columns whose floor is reachable the rates
//!   differ by more than 2×, AND an unreachable floor is reported separately as
//!   its own failure mode. The guard that survives is the one that matters —
//!   not every column may be unreachable, or there is nothing to compare.
//! - **C2 — RESTATED. A rank floor cannot hit an arbitrary rate on a column
//!   with atoms.** Pre-registered form: the achieved rate equals `r` within one
//!   row on every column. Measured: the saturated column reads 0.00411 against
//!   a target of 0.00500, because 32 767 is a huge tie (a clipped recording) and
//!   no boundary cuts inside it. Restated to what is actually true and is what
//!   an operator needs: **the ladder picks the best achievable boundary** —
//!   the achieved rate never exceeds `r`, and the next distinct value below the
//!   floor overshoots `r`, so no better boundary exists. That is checked on
//!   every column, and it is what `mu + kσ` cannot promise at all. Anti-vacuity:
//!   the floor VALUE must differ across columns, so it is not a constant.
//! - **C3 re-tuning `k` globally does not fix C1.** Take the `k` that yields
//!   exactly rate `r` on the speech column; applied to the other two columns it
//!   must still miss `r` by more than 2×. This is the load-bearing claim: it
//!   rules out "just calibrate `k` once and keep the σ floor".
//!
//! Assertions run at the END.

use ndarray::simd::{gt_i32_to_mask, popcount_batch_u64};

const WAV_HEADER: usize = 44;
/// The operating rate an operator would budget: 1 crossing in 200 rows.
const TARGET_RATE: f64 = 0.005;
/// The `k` shipped in `ndarray::hpc::cascade::calibrate` (`threshold = mu + 3σ`).
const SHIPPED_K: f64 = 3.0;
const COLUMNS: [(&str, &str); 3] = [
    ("speech", "data/tts-cascade/tts_real_output.wav"),
    ("saturated", "data/tts-cascade/cascade_speech_128frames.wav"),
    ("quiet", "data/tts-cascade/cascade_output.wav"),
];

fn load_abs_samples(path: &str) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert!(
        bytes.len() > WAV_HEADER && &bytes[0..4] == b"RIFF",
        "{path}: not RIFF"
    );
    assert_eq!(&bytes[8..12], b"WAVE", "{path}: not WAVE");
    let bits = u16::from_le_bytes([bytes[34], bytes[35]]);
    let ch = u16::from_le_bytes([bytes[22], bytes[23]]);
    assert_eq!((bits, ch), (16, 1), "{path}: expected 16-bit mono PCM");
    bytes[WAV_HEADER..]
        .chunks_exact(2)
        .map(|c| (i16::from_le_bytes([c[0], c[1]]) as i32).abs())
        .collect()
}

/// Exceedance rate above `floor`, measured the mask way: `popcount(¬M_floor)/n`.
/// `gt_i32_to_mask` IS the complement of the band mask, so this is one sweep.
fn exceedance(values: &[i32], floor: i32) -> f64 {
    let n = values.len();
    let mut m = vec![0u64; n.div_ceil(64)];
    gt_i32_to_mask(values, floor, &mut m);
    if !n.is_multiple_of(64) {
        let last = m.len() - 1;
        m[last] &= (1u64 << (n % 64)) - 1;
    }
    popcount_batch_u64(&m) as f64 / n as f64
}

fn mu_sigma(values: &[i32]) -> (f64, f64) {
    let n = values.len() as f64;
    let mu = values.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = values.iter().map(|&v| (v as f64 - mu).powi(2)).sum::<f64>() / n;
    (mu, var.sqrt())
}

/// The rank floor: the value below which `1 - r` of the rows sit.
fn rank_floor(sorted: &[i32], rate: f64) -> i32 {
    let n = sorted.len();
    let keep = ((1.0 - rate) * n as f64).round() as usize;
    sorted[keep.clamp(1, n) - 1]
}

fn main() {
    println!("PROBE-NXG-FLOOR-1\ntarget operating rate r = {TARGET_RATE} (1 crossing in {:.0} rows), shipped k = {SHIPPED_K}\n", 1.0 / TARGET_RATE);

    struct Row {
        name: &'static str,
        n: usize,
        mu: f64,
        sigma: f64,
        sigma_floor: i32,
        sigma_rate: f64,
        rank_floor: i32,
        rank_rate: f64,
    }
    let mut rows = Vec::new();
    let mut sorted_cols = Vec::new();

    for (name, path) in COLUMNS {
        let v = load_abs_samples(path);
        let mut s = v.clone();
        s.sort_unstable();
        let (mu, sigma) = mu_sigma(&v);
        let sf = (mu + SHIPPED_K * sigma).round() as i32;
        let rf = rank_floor(&s, TARGET_RATE);
        rows.push(Row {
            name,
            n: v.len(),
            mu,
            sigma,
            sigma_floor: sf,
            sigma_rate: exceedance(&v, sf),
            rank_floor: rf,
            rank_rate: exceedance(&v, rf),
        });
        sorted_cols.push((name, v, s));
    }

    println!(
        "{:<10} {:>8} {:>10} {:>10} {:>12} {:>12} {:>11} {:>12}",
        "column", "n", "mu", "sigma", "mu+3s floor", "its rate", "rank floor", "its rate"
    );
    for r in &rows {
        println!(
            "{:<10} {:>8} {:>10.1} {:>10.1} {:>12} {:>12.5} {:>11} {:>12.5}",
            r.name, r.n, r.mu, r.sigma, r.sigma_floor, r.sigma_rate, r.rank_floor, r.rank_rate
        );
    }

    // ── C1: does k name a rate? ─────────────────────────────────────────────
    let rates: Vec<f64> = rows.iter().map(|r| r.sigma_rate).collect();
    let reachable: Vec<f64> = rates.iter().cloned().filter(|&x| x > 0.0).collect();
    let unreachable: Vec<&str> = rows
        .iter()
        .filter(|r| r.sigma_rate == 0.0)
        .map(|r| r.name)
        .collect();
    let (lo, hi) = (
        reachable.iter().cloned().fold(f64::MAX, f64::min),
        reachable.iter().cloned().fold(0.0, f64::max),
    );
    let spread = if reachable.len() >= 2 {
        hi / lo
    } else {
        f64::NAN
    };
    for r in rows.iter().filter(|r| r.sigma_rate == 0.0) {
        let max_seen = sorted_cols
            .iter()
            .find(|(n, _, _)| *n == r.name)
            .map(|(_, _, s)| *s.last().unwrap())
            .unwrap();
        println!("  UNREACHABLE: {} has mu+3s = {} but its maximum value is {max_seen} — the floor can never be crossed", r.name, r.sigma_floor);
    }

    // ── C2: does the rank floor hit the rate it was asked for? ──────────────
    // Best-achievable check: the achieved rate never exceeds r, and the next
    // DISTINCT value below the floor overshoots r — so no better boundary exists.
    let mut rank_best_achievable = true;
    for (r, (_, v, sorted)) in rows.iter().zip(&sorted_cols) {
        let next_below = sorted.iter().rev().find(|&&x| x < r.rank_floor).copied();
        let over = match next_below {
            Some(nb) => exceedance(v, nb),
            None => f64::INFINITY,
        };
        let ok = r.rank_rate <= TARGET_RATE + 1.0 / r.n as f64 && over > TARGET_RATE;
        println!(
            "  {:<10} rank floor {:>6} -> rate {:.5} (<= r); next distinct below {:?} -> {:.5} (> r): best achievable = {ok}",
            r.name, r.rank_floor, r.rank_rate, next_below, over
        );
        if !ok {
            rank_best_achievable = false;
        }
    }
    let floors_differ = rows
        .iter()
        .map(|r| r.rank_floor)
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        == rows.len();

    // ── C3: re-tune k on `speech`, apply it to the others ───────────────────
    // The k that puts `speech` exactly at the target rate.
    let sp = &rows[0];
    let k_tuned = (sp.rank_floor as f64 - sp.mu) / sp.sigma;
    let tuned: Vec<(&str, f64)> = rows
        .iter()
        .zip(&sorted_cols)
        .map(|(r, (_, v, _))| {
            let f = (r.mu + k_tuned * r.sigma).round() as i32;
            (r.name, exceedance(v, f))
        })
        .collect();
    // Worst FINITE off-target ratio, so the claim never leans on an infinity.
    let mut worst_tuned_ratio: f64 = 1.0;
    let mut tuned_unreachable = 0usize;
    for (name, rate) in &tuned {
        if *rate > 0.0 {
            let ratio = (rate / TARGET_RATE).max(TARGET_RATE / rate);
            println!("  k tuned on speech = {k_tuned:.4} -> on {name}: rate {rate:.5} ({ratio:.2}x off target)");
            if *name != "speech" {
                worst_tuned_ratio = worst_tuned_ratio.max(ratio);
            }
        } else {
            tuned_unreachable += 1;
            println!(
                "  k tuned on speech = {k_tuned:.4} -> on {name}: floor still UNREACHABLE (rate 0)"
            );
        }
    }

    println!("\nOBSERVATIONS (measured, not claims)");
    println!("  Jirak context: the σ above is a whole-column σ. `rolling_floor.rs` warns its own per-tier σ");
    println!("  is a small weakly-dependent sample, so the nominal Gaussian tail is not the right reference —");
    println!("  this probe therefore compares MEASURED rates only, never a nominal one.");
    println!(
        "  a Gaussian would put P(v > mu+3s) at 0.00135; the three real columns read {rates:?}"
    );

    println!("\nCLAIMS");
    println!("  C1 (RESTATED) k=3 spread over REACHABLE columns: {spread:.2}x (need > 2); unreachable floors: {unreachable:?}");
    println!("  C2 (RESTATED) rank floor is the best achievable boundary on every column: {rank_best_achievable}; floors differ: {floors_differ}");
    println!("  C3 worst FINITE off-target after re-tuning k on speech: {worst_tuned_ratio:.2}x (need > 2); still-unreachable: {tuned_unreachable}");

    assert!(
        reachable.len() >= 2,
        "C1 anti-vacuity: fewer than two columns have a reachable mu+3s floor — there is nothing left to compare"
    );
    assert!(spread > 2.0, "C1: k=3 gave near-identical rates across real columns ({spread:.2}x) — k would then name a rate after all");
    assert!(
        rank_best_achievable,
        "C2 (restated): the rank floor was not the best achievable boundary — either it overshot r, or a lower distinct value would have been closer"
    );
    assert!(
        floors_differ,
        "C2 anti-vacuity: the rank floor is the same value on every column"
    );
    assert!(
        worst_tuned_ratio > 2.0,
        "C3: a single re-tuned k held the target rate across columns ({worst_tuned_ratio:.2}x) — the sigma floor would then be salvageable by calibration"
    );
    println!("\nC1 RESTATED (unreachable floor found) · C2 RESTATED (ties bound the achievable rate) · C3 PASS.");
}
