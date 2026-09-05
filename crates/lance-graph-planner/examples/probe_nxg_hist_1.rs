//! **PROBE-NXG-HIST-1** — plan `.claude/nexgen/plans/nexgen-mask-histogram-thresholds-v1.md`
//! §5 step 2, the first gate before any of D-NXG-4..12 leaves PROPOSAL.
//!
//! The claim under test (E-NXG-1): a Belichtungsmesser reading over N rows IS
//! a chain of nested row masks `M_0 ⊆ M_1 ⊆ … ⊆ M_{B-1}`; bucket `i` is
//! `M_i ∧ ¬M_{i-1}` = `mask_ternlog::<AND_ANDNOT2>` by NAME (E-NXG-2/3); and a
//! row's Prozentrang is the index of the innermost mask containing it — a
//! partition point over nested masks, never a sort (E-NXG-2).
//!
//! It changes no library code. It builds one `NestedBands` from a REAL value
//! column and asserts three pre-registered claims plus their anti-vacuity
//! guards. Everything else it prints is an OBSERVATION, not a claim.
//!
//! ```text
//! CARGO_PROFILE_DEV_DEBUG=0 cargo run -p lance-graph-planner --example probe_nxg_hist_1 [wav-path]
//! ```
//!
//! # The column
//!
//! `data/tts-cascade/tts_real_output.wav` — 94 572 samples of real 16-bit
//! mono speech already checked into this repo. The probe reads `|sample|` as
//! an `i32` distance-like column. It is real (recorded audio, heavy-tailed,
//! not drawn from any generator this probe controls) — the truth-architect
//! rule against synthetic inputs is satisfied. It is NOT a facet column: the
//! plan's §1 gap "Prozentrang exists only in doctrine" is about the mechanism,
//! and the mechanism is value-type-agnostic. A facet-column rerun is the
//! obvious next arm once one is on disk (none is: the 2026-09-05 harvest found
//! no `.soa`/facet fixture in-tree).
//!
//! # Pre-registered claims
//!
//! - **C1 structure.** The B=16 band masks are nested (`M_i ∧ ¬M_{i+1}` is
//!   empty for every i), the 16 bucket masks are pairwise disjoint, and the
//!   bucket popcounts sum to exactly N.
//! - **C2 rank = partition point.** For EVERY row, the bucket index read off
//!   the masks equals `boundaries.partition_point(|b| b < v)` computed from
//!   the value directly. Anti-vacuity: at least 8 of 16 buckets are non-empty
//!   and no bucket holds more than half the rows — a degenerate column would
//!   make C2 true of a one-bucket histogram.
//! - **C3 bucket is one named immediate.** `mask_ternlog::<AND_ANDNOT2>(M_i,
//!   M_{i-1}, M_{i-1})` is bit-identical to `mask_andnot(M_i, M_{i-1})` and to
//!   the bucket the reference computed. Can-it-fire: the neighbouring
//!   immediate `AND3` on the same operands must DIFFER for every i whose
//!   lower band is non-empty — otherwise the immediate is decoration.
//!
//! Assertions run at the very END so every claim is measured before any can
//! abort the run (the entropy-census lesson: a probe that asserts inline hides
//! evidence).

use lance_graph_contract::thought_atoms::normalized_entropy;
use ndarray::simd::ternlog::{AND3, AND_ANDNOT2};
use ndarray::simd::{gt_i32_to_mask, mask_andnot, mask_ternlog, popcount_batch_u64};

const B: usize = 16;
const DEFAULT_WAV: &str = "data/tts-cascade/tts_real_output.wav";
const WAV_HEADER: usize = 44;

fn load_abs_samples(path: &str) -> Vec<i32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert!(
        bytes.len() > WAV_HEADER && &bytes[0..4] == b"RIFF",
        "{path}: not a RIFF file"
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

/// `M = rows with value <= boundary`, as `!gt` with the tail beyond `n` cleared.
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

fn bit(m: &[u64], row: usize) -> bool {
    (m[row / 64] >> (row % 64)) & 1 == 1
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_WAV.to_string());
    let values = load_abs_samples(&path);
    let n = values.len();
    let words = words_for(n);
    println!(
        "PROBE-NXG-HIST-1\ncolumn: {path}\nrows N = {n}, mask words = {words}, bands B = {B}\n"
    );

    // ── boundaries: empirical quantiles (the ONE sort in this probe; it is the
    // reference the mask walk must reproduce, never the mechanism) ───────────
    let mut sorted = values.clone();
    sorted.sort_unstable();
    let boundaries: Vec<i32> = (0..B)
        .map(|i| sorted[((i + 1) * n / B).saturating_sub(1)])
        .collect();
    println!("boundaries (quantile): {boundaries:?}");

    // ── the nested band masks M_0 ⊆ … ⊆ M_{B-1} ─────────────────────────────
    let bands: Vec<Vec<u64>> = boundaries.iter().map(|&b| le_mask(&values, b)).collect();
    let band_pop: Vec<u64> = bands.iter().map(|m| popcount_batch_u64(m)).collect();

    // ── buckets by the NAMED immediate, and two references ──────────────────
    let mut buckets: Vec<Vec<u64>> = Vec::with_capacity(B);
    let mut ref_andnot: Vec<Vec<u64>> = Vec::with_capacity(B);
    let mut c3_named_eq_andnot = true;
    let mut c3_and3_differs_where_it_can = true;
    for i in 0..B {
        let mut bucket = vec![0u64; words];
        let mut refb = vec![0u64; words];
        if i == 0 {
            bucket.copy_from_slice(&bands[0]);
            refb.copy_from_slice(&bands[0]);
        } else {
            mask_ternlog::<AND_ANDNOT2>(&bands[i], &bands[i - 1], &bands[i - 1], &mut bucket);
            mask_andnot(&bands[i], &bands[i - 1], &mut refb);
            let mut and3 = vec![0u64; words];
            mask_ternlog::<AND3>(&bands[i], &bands[i - 1], &bands[i - 1], &mut and3);
            if band_pop[i - 1] > 0 && and3 == bucket {
                c3_and3_differs_where_it_can = false;
            }
        }
        if bucket != refb {
            c3_named_eq_andnot = false;
        }
        buckets.push(bucket);
        ref_andnot.push(refb);
    }
    let bucket_pop: Vec<u64> = buckets.iter().map(|m| popcount_batch_u64(m)).collect();

    // ── C1: nested, disjoint, sums to N ─────────────────────────────────────
    let mut scratch = vec![0u64; words];
    let mut c1_nested = true;
    for i in 0..B - 1 {
        mask_andnot(&bands[i], &bands[i + 1], &mut scratch);
        if popcount_batch_u64(&scratch) != 0 {
            c1_nested = false;
        }
    }
    let mut c1_disjoint = true;
    for i in 0..B {
        for j in i + 1..B {
            // AND2 = a & b (c ignored); reuse c = a.
            mask_ternlog::<{ ndarray::simd::ternlog::AND2 }>(
                &buckets[i],
                &buckets[j],
                &buckets[i],
                &mut scratch,
            );
            if popcount_batch_u64(&scratch) != 0 {
                c1_disjoint = false;
            }
        }
    }
    let c1_sum: u64 = bucket_pop.iter().sum();

    // ── C2: rank read off the masks == partition point on the value ──────────
    let mut c2_mismatches = 0usize;
    for (row, &v) in values.iter().enumerate() {
        let by_value = boundaries.partition_point(|&b| b < v).min(B - 1);
        let by_mask = (0..B).find(|&i| bit(&buckets[i], row));
        if by_mask != Some(by_value) {
            c2_mismatches += 1;
        }
    }
    let nonempty = bucket_pop.iter().filter(|&&p| p > 0).count();
    let max_bucket = *bucket_pop.iter().max().unwrap_or(&0);

    // ── OBSERVATIONS (measured, not claims) ─────────────────────────────────
    println!("\nOBSERVATIONS (measured, not claims)");
    println!("  band popcounts (cumulative): {band_pop:?}");
    println!("  bucket popcounts:            {bucket_pop:?}");
    let weights: Vec<f32> = bucket_pop.iter().map(|&p| p as f32).collect();
    let h = normalized_entropy(&weights);
    println!("  normalized Shannon entropy of the bucket histogram (D-NXG-9 preview): {h:?}");
    // σ recovered from the histogram (bucket midpoints weighted by popcount)
    // vs σ computed directly — plan §3 room 3, observation only.
    let mids: Vec<f64> = (0..B)
        .map(|i| {
            let lo = if i == 0 {
                0.0
            } else {
                boundaries[i - 1] as f64
            };
            (lo + boundaries[i] as f64) / 2.0
        })
        .collect();
    let mean_h = mids
        .iter()
        .zip(&bucket_pop)
        .map(|(m, &p)| m * p as f64)
        .sum::<f64>()
        / n as f64;
    let var_h = mids
        .iter()
        .zip(&bucket_pop)
        .map(|(m, &p)| (m - mean_h).powi(2) * p as f64)
        .sum::<f64>()
        / n as f64;
    let mean_d = values.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var_d = values
        .iter()
        .map(|&v| (v as f64 - mean_d).powi(2))
        .sum::<f64>()
        / n as f64;
    println!(
        "  sigma from histogram = {:.2}, sigma direct = {:.2}, ratio = {:.4} (room 3: unasserted)",
        var_h.sqrt(),
        var_d.sqrt(),
        var_h.sqrt() / var_d.sqrt()
    );
    println!(
        "  mask bytes hot: {} band masks x {} B = {} B",
        B,
        words * 8,
        B * words * 8
    );

    // ── verdicts, asserted LAST ──────────────────────────────────────────────
    println!("\nCLAIMS");
    println!("  C1 nested={c1_nested} disjoint={c1_disjoint} sum={c1_sum} (N={n})");
    println!("  C2 rank mismatches = {c2_mismatches} / {n}; non-empty buckets = {nonempty}/{B}; max bucket = {max_bucket}");
    println!("  C3 named==andnot: {c3_named_eq_andnot}; AND3 differs where lower band non-empty: {c3_and3_differs_where_it_can}");

    assert!(c1_nested, "C1: bands are not nested");
    assert!(c1_disjoint, "C1: buckets are not pairwise disjoint");
    assert_eq!(c1_sum, n as u64, "C1: bucket popcounts do not sum to N");
    assert!(nonempty >= 8, "C2 anti-vacuity: only {nonempty} of {B} buckets non-empty — the column is degenerate for this probe");
    assert!(
        (max_bucket as usize) * 2 < n,
        "C2 anti-vacuity: one bucket holds >= half the rows"
    );
    assert_eq!(
        c2_mismatches, 0,
        "C2: mask-walk rank != partition-point rank"
    );
    assert!(
        c3_named_eq_andnot,
        "C3: mask_ternlog::<AND_ANDNOT2> != mask_andnot"
    );
    assert!(
        c3_and3_differs_where_it_can,
        "C3 can-it-fire: AND3 gave the same mask as AND_ANDNOT2 — the immediate is inert"
    );
    println!("\nC1 PASS · C2 PASS · C3 PASS.");
}
