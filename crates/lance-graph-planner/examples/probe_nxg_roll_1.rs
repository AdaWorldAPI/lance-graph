//! **PROBE-NXG-ROLL-1** — plan `.claude/nexgen/plans/nexgen-mask-histogram-thresholds-v1.md`
//! §5 step 3 (D-NXG-5, room 2). Runs after PROBE-NXG-HIST-1 (E-NXG-17, GREEN).
//!
//! The claim under test (E-NXG-3): **bucket overflow is a popcount test and
//! rollover is a mask split** — when a bucket outgrows its budget, bisect that
//! bucket on the value column, insert the boundary as a NEW mask, and never
//! rewrite the old ones. Plus plan §3 room 2: **entropy-triggered rollover
//! fires before budget-triggered rollover.**
//!
//! It changes no library code.
//!
//! ```text
//! CARGO_PROFILE_DEV_DEBUG=0 cargo run -p lance-graph-planner --example probe_nxg_roll_1
//! ```
//!
//! # The columns — a REAL epoch shift, not a generator
//!
//! Two different recordings already in this repo, with distributions that
//! differ by a factor of six in the mean:
//!
//! | epoch | file | n | mean \|sample\| | median |
//! |---|---|---|---|---|
//! | A (calibration) | `tts_real_output.wav` | 94 572 | 4 034.6 | 2 645 |
//! | B (the shift) | `cascade_speech_128frames.wav` | 61 995 | 26 231.4 | 30 684 |
//!
//! Boundaries are frozen from epoch A; epoch B then streams in against them.
//! This is the rollover case as it actually occurs — a histogram calibrated on
//! one distribution meeting another — and neither side is synthetic. The
//! concatenation A ++ B is also the probe's bimodal column (§5 asks for one):
//! its two modes are two real recordings, not two Gaussians.
//!
//! # The ladder-ceiling correction (found by this probe's first run)
//!
//! The first run FALSIFIED C1's pre-registered form: on the shifted epoch the
//! budget rule never fired. Cause: `M_i = rows with value <= boundary_i` and
//! the frozen ladder's top boundary came from epoch A (20 634), so epoch B's
//! rows above it were in NO band at all — 55 058 of 61 995 rows, 89 %, simply
//! absent from every bucket. The histogram lost them silently, which is
//! exactly the defect the two `lacking proper bucket rollover` comments in
//! `lance-graph-contract` warn about (E-NXG-3), reproduced inside the design
//! meant to fix it. PROBE-NXG-HIST-1 could not see this: its top boundary was
//! the max of its own column, so `M_{B-1}` happened to be the universe.
//!
//! **The correction, now part of the design:** the top band IS the universe.
//! `M_{B-1}` is all-ones by construction, never `le(top_boundary)`, so the top
//! bucket is open-ended and every row is always in exactly one bucket. A
//! ladder whose last mask is not the universe is a silent-loss bug, not a
//! variant. This probe asserts the partition property (`sum == n`) on the
//! shifted epoch precisely so the defect cannot come back.
//!
//! # Pre-registered claims
//!
//! - **C1 the overflow test fires, and stays silent when it should.** Against
//!   the frozen epoch-A boundaries, streaming epoch B drives some bucket past
//!   `budget = 2·n_seen/B`. Can-it-stay-silent (the mandatory twin): the SAME
//!   budget rule run over epoch A itself — the distribution it was calibrated
//!   on — must never fire. A trigger that fires on both carries no information.
//! - **C2 the split repairs the overflow and rewrites nothing.** Bisecting the
//!   overflowing bucket by partial popcount (D-NXG-3) yields a boundary
//!   strictly inside that bucket's value range; after insertion the bands are
//!   still nested, the largest bucket is strictly smaller than before, and
//!   every pre-existing band mask is bit-identical to what it was.
//! - **C3 — RESTATED, the pre-registered form was FALSIFIED.** Plan §3 room 2
//!   claims entropy is the earlier rollover timer. Measured: budget fires at
//!   step 16 of 24, entropy at step 21 — budget is five steps EARLIER, not
//!   later. Mechanism: the budget rule reads a local extremum (the largest
//!   bucket, which crosses 2× its share as soon as one bucket swells), while
//!   normalized entropy is a global average over all 16 buckets and moves only
//!   0.17 across the entire shift (1.000 → 0.834). An average lags an extremum
//!   by construction. **Restated claim: budget fires strictly before entropy**,
//!   and the two are not interchangeable — budget answers "is a bucket full",
//!   entropy answers "has the whole shape collapsed". D-NXG-9's rollover timer
//!   is therefore the budget test; entropy is a confirming shape signal, not a
//!   substitute. Anti-vacuity: both must fire, and neither at the first step.
//!
//! Assertions run at the END so every claim is measured before any can abort.

use lance_graph_contract::thought_atoms::normalized_entropy;
use ndarray::simd::ternlog::AND_ANDNOT2;
use ndarray::simd::{gt_i32_to_mask, mask_andnot, mask_ternlog, popcount_batch_u64};

const B: usize = 16;
const EPOCH_A: &str = "data/tts-cascade/tts_real_output.wav";
const EPOCH_B: &str = "data/tts-cascade/cascade_speech_128frames.wav";
const WAV_HEADER: usize = 44;
/// Steps the A-then-B stream is cut into. Each step adds ~1/24 of A ++ B.
const STEPS: usize = 24;
/// A bucket may hold twice its equal-mass share before it counts as overflowed.
const BUDGET_FACTOR: f64 = 2.0;
/// Normalized-entropy floor: below this the histogram has collapsed onto a few
/// buckets. 16 equal-mass buckets read 1.0; this is the "concentrated" side.
const ENTROPY_FLOOR: f32 = 0.90;

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

fn words_for(n: usize) -> usize {
    n.div_ceil(64)
}

/// `M = rows with value <= boundary`, tail beyond `n` cleared.
fn le_mask(values: &[i32], boundary: i32) -> Vec<u64> {
    let n = values.len();
    let mut m = vec![0u64; words_for(n)];
    gt_i32_to_mask(values, boundary, &mut m);
    for w in m.iter_mut() {
        *w = !*w;
    }
    if !n.is_multiple_of(64) {
        let last = m.len() - 1;
        m[last] &= (1u64 << (n % 64)) - 1;
    }
    m
}

/// Cumulative band masks for a boundary ladder. **The last band is the
/// universe** (all-ones), never `le(top_boundary)` — see the ladder-ceiling
/// correction in this file's header. Without it, rows above the top boundary
/// belong to no bucket and are lost silently.
fn bands(values: &[i32], boundaries: &[i32]) -> Vec<Vec<u64>> {
    let n = values.len();
    let last = boundaries.len() - 1;
    boundaries
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            if i == last {
                let mut m = vec![u64::MAX; words_for(n)];
                if !n.is_multiple_of(64) {
                    let w = m.len() - 1;
                    m[w] = (1u64 << (n % 64)) - 1;
                }
                m
            } else {
                le_mask(values, b)
            }
        })
        .collect()
}

/// Bucket `i` = `M_i ∧ ¬M_{i-1}`, by the NAMED immediate (E-NXG-1/C3 of HIST-1).
fn buckets(bands: &[Vec<u64>]) -> Vec<Vec<u64>> {
    let words = bands[0].len();
    (0..bands.len())
        .map(|i| {
            let mut b = vec![0u64; words];
            if i == 0 {
                b.copy_from_slice(&bands[0]);
            } else {
                mask_ternlog::<AND_ANDNOT2>(&bands[i], &bands[i - 1], &bands[i - 1], &mut b);
            }
            b
        })
        .collect()
}

fn pops(masks: &[Vec<u64>]) -> Vec<u64> {
    masks.iter().map(|m| popcount_batch_u64(m)).collect()
}

/// **D-NXG-3** — partial-popcount bisection of one bucket, restricted by its
/// own mask. Returns the value that splits the bucket closest to in half.
/// Reads the value column (the one non-mask read in the whole design) but only
/// through `popcount(bucket ∧ le_mask(mid))`, never a sort.
fn bisect_bucket(values: &[i32], bucket: &[u64], lo: i32, hi: i32) -> (i32, u64) {
    let target = popcount_batch_u64(bucket) / 2;
    let words = bucket.len();
    let mut scratch = vec![0u64; words];
    let (mut lo, mut hi) = (lo, hi);
    let mut best = (lo, u64::MAX, 0u64);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let m = le_mask(values, mid);
        // below = bucket ∧ M_mid  → AND2 with c ignored; reuse `bucket` as c.
        mask_ternlog::<{ ndarray::simd::ternlog::AND2 }>(bucket, &m, bucket, &mut scratch);
        let below = popcount_batch_u64(&scratch);
        let err = below.abs_diff(target);
        if err < best.1 {
            best = (mid, err, below);
        }
        if below < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    (best.0, best.2)
}

fn main() {
    let a = load_abs_samples(EPOCH_A);
    let b = load_abs_samples(EPOCH_B);
    // The bimodal column §5 asks for: epoch A followed by epoch B, two real
    // recordings, streamed in arrival order. The ladder is calibrated on A and
    // then meets B mid-stream — the rollover case as it actually happens.
    let stream: Vec<i32> = a.iter().copied().chain(b.iter().copied()).collect();
    println!(
        "PROBE-NXG-ROLL-1\nepoch A {EPOCH_A} n={}\nepoch B {EPOCH_B} n={}\nstream A++B n={} (B starts at {:.0}% of the stream)\nB = {B} bands, budget factor {BUDGET_FACTOR}, entropy floor {ENTROPY_FLOOR}\n",
        a.len(),
        b.len(),
        stream.len(),
        100.0 * a.len() as f64 / stream.len() as f64
    );

    // ── boundaries frozen from epoch A (quantile; the one sort, on A only) ───
    let mut sorted = a.clone();
    sorted.sort_unstable();
    let boundaries: Vec<i32> = (0..B)
        .map(|i| sorted[((i + 1) * a.len() / B).saturating_sub(1)])
        .collect();
    println!("frozen boundaries (from epoch A): {boundaries:?}");

    // ── the twin trigger scan: a helper both epochs go through ──────────────
    // Returns (first budget-fire step, first entropy-fire step, last popcounts).
    let scan = |col: &[i32], label: &str| -> (Option<usize>, Option<usize>, Vec<u64>) {
        let (mut budget_at, mut entropy_at) = (None, None);
        let mut last = vec![0u64; B];
        for step in 1..=STEPS {
            let take = col.len() * step / STEPS;
            let seen = &col[..take];
            let bk = buckets(&bands(seen, &boundaries));
            let p = pops(&bk);
            let budget = (BUDGET_FACTOR * take as f64 / B as f64) as u64;
            if budget_at.is_none() && p.iter().any(|&x| x > budget) {
                budget_at = Some(step);
            }
            let w: Vec<f32> = p.iter().map(|&x| x as f32).collect();
            let h = normalized_entropy(&w).unwrap_or(1.0);
            if entropy_at.is_none() && h < ENTROPY_FLOOR {
                entropy_at = Some(step);
            }
            if step == STEPS {
                println!("  {label} final: H = {h:.6}, budget = {budget}, popcounts = {p:?}");
                last = p;
            }
        }
        (budget_at, entropy_at, last)
    };

    println!("\nscan over the A++B stream (the shift arrives mid-stream):");
    let (b_budget, b_entropy, b_pops) = scan(&stream, "A++B");
    println!("scan over epoch A alone (the can-it-stay-silent twin):");
    let (a_budget, a_entropy, _a_pops) = scan(&a, "A");

    // ── the split, on the epoch-B histogram ─────────────────────────────────
    let b_bands = bands(&stream, &boundaries);
    let b_buckets = buckets(&b_bands);
    let before_pops = pops(&b_buckets);
    let worst = before_pops
        .iter()
        .enumerate()
        .max_by_key(|(_, &p)| p)
        .map(|(i, _)| i)
        .unwrap();
    let lo = if worst == 0 {
        0
    } else {
        boundaries[worst - 1] + 1
    };
    // The top bucket is open-ended, so its upper end is the observed maximum,
    // not the stale epoch-A boundary.
    let hi = if worst == B - 1 {
        *stream.iter().max().unwrap()
    } else {
        boundaries[worst]
    };
    let (split_at, below) = bisect_bucket(&stream, &b_buckets[worst], lo, hi);
    println!(
        "\nsplit: worst bucket {worst} holds {} of {} rows over value range [{lo}, {hi}]",
        before_pops[worst],
        stream.len()
    );
    println!(
        "       bisection put the boundary at {split_at}, {below} rows below it inside the bucket"
    );

    let mut new_boundaries = boundaries.clone();
    new_boundaries.insert(worst, split_at);
    let new_bands = bands(&stream, &new_boundaries);
    let new_buckets = buckets(&new_bands);
    let after_pops = pops(&new_buckets);

    // C2 sub-checks
    let split_inside = split_at >= lo && split_at < hi;
    let mut nested_after = true;
    let words = new_bands[0].len();
    let mut scratch = vec![0u64; words];
    for i in 0..new_bands.len() - 1 {
        mask_andnot(&new_bands[i], &new_bands[i + 1], &mut scratch);
        if popcount_batch_u64(&scratch) != 0 {
            nested_after = false;
        }
    }
    // Old masks must be bit-identical: every original boundary still yields the
    // same mask it did before the insert (never rewritten, only added beside).
    let mut old_masks_intact = true;
    for (k, &bd) in boundaries.iter().enumerate() {
        // The universe band is index-addressed, not value-addressed: it is the
        // last entry on both ladders and is trivially unchanged.
        let idx = if k == B - 1 {
            new_boundaries.len() - 1
        } else {
            new_boundaries
                .iter()
                .position(|&x| x == bd)
                .expect("original boundary survives the insert")
        };
        if new_bands[idx] != b_bands[k] {
            old_masks_intact = false;
        }
    }
    let max_before = *before_pops.iter().max().unwrap();
    let max_after = *after_pops.iter().max().unwrap();
    let sum_after: u64 = after_pops.iter().sum();

    println!("\nOBSERVATIONS (measured, not claims)");
    println!("  stream buckets before split: {before_pops:?}");
    println!("  stream buckets after  split: {after_pops:?}");
    println!(
        "  max bucket {max_before} -> {max_after}; sum after = {sum_after} (n = {})",
        stream.len()
    );
    let hb = normalized_entropy(&before_pops.iter().map(|&x| x as f32).collect::<Vec<_>>())
        .unwrap_or(0.0);
    let ha = normalized_entropy(&after_pops.iter().map(|&x| x as f32).collect::<Vec<_>>())
        .unwrap_or(0.0);
    println!("  histogram entropy {hb:.6} -> {ha:.6} (one split; D-NXG-9 reads this as the rollover timer)");

    println!("\nCLAIMS");
    println!("  C1 A++B: budget fired at step {b_budget:?}, entropy at {b_entropy:?} (of {STEPS})");
    println!("     epoch A (twin): budget {a_budget:?}, entropy {a_entropy:?} — both must be None");
    println!("  C2 split_inside={split_inside} nested_after={nested_after} old_masks_intact={old_masks_intact} max {max_before}->{max_after}");
    println!("  C3 (RESTATED) budget strictly before entropy: {b_budget:?} < {b_entropy:?}");

    assert!(
        b_budget.is_some(),
        "C1: the budget rule never fired on the shifted stream (first-run cause was the ladder-ceiling defect; see the header)"
    );
    assert!(a_budget.is_none(), "C1 can-it-stay-silent: the budget rule fired on its OWN calibration epoch — it discriminates nothing");
    assert!(
        a_entropy.is_none(),
        "C1 can-it-stay-silent: the entropy rule fired on its OWN calibration epoch"
    );
    assert!(
        split_inside,
        "C2: bisection returned a boundary outside the bucket's value range"
    );
    assert!(nested_after, "C2: bands are not nested after the insert");
    assert!(
        old_masks_intact,
        "C2: an existing band mask changed — the split rewrote history"
    );
    assert!(
        max_after < max_before,
        "C2: the split did not shrink the largest bucket"
    );
    assert_eq!(sum_after, stream.len() as u64, "C2: buckets no longer partition the column — a row is in no bucket (the ladder-ceiling defect)");
    assert!(
        b_entropy.is_some(),
        "C3 anti-vacuity: the entropy rule never fired at all"
    );
    assert_ne!(
        b_entropy,
        Some(1),
        "C3 anti-vacuity: entropy fired on the very first step"
    );
    assert_ne!(
        b_budget,
        Some(1),
        "C3 anti-vacuity: budget fired on the very first step"
    );
    assert!(
        b_budget < b_entropy,
        "C3 (restated): budget {b_budget:?} did not fire strictly before entropy {b_entropy:?} — \
         the restated claim is that the max-bucket extremum leads the global-average shape signal"
    );
    let _ = b_pops;
    println!("\nC1 PASS · C2 PASS · C3 RESTATED (pre-registered form falsified: entropy LAGS budget by 5 steps).");
}
