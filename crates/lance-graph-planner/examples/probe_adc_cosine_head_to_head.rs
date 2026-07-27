//! PROBE — ADC cosine head-to-head: f32 tables (BEFORE) vs u8-quantized
//! 6×256 LUT (AFTER, the palette256-shaped cosine replacement).
//!
//! Operator directive (2026-07-27): "measure before after head to head —
//! ICC spearman pearson etc." Pass gate is the operator's palette256
//! exactness band: mean Spearman ρ ≥ 0.9973.
//!
//! ## Protocol (certification-officer pattern)
//! - **Real bytes only** (Rule 23): rows come from a bgz7 shard of real
//!   model weights (default: bge-m3-f16.bgz7, SHA256-pinned in
//!   `crates/bgz-tensor/data/manifest.json`, release v0.1.0-bgz-data).
//!   Each `Base17` row is `[i16; 17]` fixed-point ×256 → 17-dim f32.
//! - **Deterministic sampling**: SplitMix64, seed 0x9E3779B97F4A7C15
//!   (the workspace's canonical pair-sampler seed).
//! - **Shared inputs**: one codebook (6 subspaces × 256 real-row centroids,
//!   dims 3+3+3+3+3+2 = 17), one encode pass, identical queries/candidates
//!   for both arms. The delta isolates TABLE quantization only.
//! - **BEFORE**: `ScalarAdc::new(AdcMetric::Cosine)` — the shipped f32 path
//!   consumed hot by `CamPqScanOp` (`cam.rs:260-271` → `distance_batch`).
//! - **AFTER**: the same tables affine-quantized to u8 with ONE shared
//!   (min, span) per query across all 6 subspace tables — preserving
//!   additivity — then integer-summed (u32). This is the 6×256 u8 LUT
//!   shape of the palette256 doctrine.
//! - **Falsifier** (can-it-fire): a 4-bit (16-level) ablation MUST score
//!   strictly worse than 8-bit, or the harness cannot detect degradation.
//!
//! ## Metrics (4 dp): Pearson r, Spearman ρ (average ties), ICC(2,1) on
//! min-max-normalized columns, recall@10, plus wall-clock ns/candidate.
//!
//! Run:
//! ```text
//! cargo run -p lance-graph-planner --example probe_adc_cosine_head_to_head -- <shard.bgz7>
//! ```
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)] // probe-local: stats casts on values whose ranges are verified above each site

use lance_graph_contract::cam::{AdcMetric, DistanceTableProvider, ScalarAdc};
use ndarray::hpc::gguf_indexer::CompressedTensor;
use std::io::Read;
use std::time::Instant;

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const DIM: usize = 17;
const SUB_DIMS: [usize; 6] = [3, 3, 3, 3, 3, 2]; // 6 subspaces over 17 dims
const N_CENTROIDS: usize = 256;
const N_QUERIES: usize = 64;
const N_DB: usize = 4096;
const TOP_K: usize = 10;

/// SplitMix64 — deterministic, no external rng dep.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn main() {
    let shard = std::env::args().nth(1).unwrap_or_else(|| {
        "/tmp/claude-0/-home-user/bcd29cfc-5bae-5b23-b86b-0de9582a87da/scratchpad/bge-m3-f16.bgz7"
            .to_string()
    });
    println!("shard: {shard}");
    // Lenient bgz7 read: the published v0.1.0 bge-m3 asset declares 389
    // tensors but contains 290 complete ones then exact EOF (SHA256 matches
    // the committed manifest — it shipped truncated). Keep every tensor that
    // parses; report declared vs parsed. `read_bgz7_file` would hard-fail.
    let mut reader = std::io::BufReader::new(std::fs::File::open(&shard).expect("open shard"));
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).expect("magic");
    assert_eq!(&magic, b"BGZ7", "bad magic");
    let mut u32_buf = [0u8; 4];
    reader.read_exact(&mut u32_buf).expect("n_tensors");
    let declared = u32::from_le_bytes(u32_buf) as usize;
    let mut tensors: Vec<CompressedTensor> = Vec::with_capacity(declared);
    for _ in 0..declared {
        match CompressedTensor::read_from(&mut reader) {
            Ok(t) => tensors.push(t),
            Err(_) => break, // truncated tail — keep the complete prefix
        }
    }
    println!("declared tensors: {declared}  parsed: {}", tensors.len());
    // Flatten all rows to 17-dim f32 (i16 fixed-point / 256), drop all-zero rows.
    let mut rows: Vec<[f32; DIM]> = Vec::new();
    for t in &tensors {
        for r in &t.rows {
            let mut v = [0f32; DIM];
            let mut nz = false;
            for (i, d) in r.dims.iter().enumerate() {
                v[i] = f32::from(*d) / 256.0;
                nz |= *d != 0;
            }
            if nz {
                rows.push(v);
            }
        }
    }
    println!("tensors: {}  usable rows: {}", tensors.len(), rows.len());
    assert!(
        rows.len() > N_CENTROIDS + N_QUERIES + N_DB,
        "shard too small for disjoint sampling"
    );

    // Disjoint deterministic sample: centroid rows, queries, database.
    let mut rng = SplitMix64(SEED);
    let mut taken = vec![false; rows.len()];
    let draw = |rng: &mut SplitMix64, taken: &mut Vec<bool>| -> usize {
        loop {
            let i = rng.below(taken.len());
            if !taken[i] {
                taken[i] = true;
                return i;
            }
        }
    };
    let centroid_rows: Vec<usize> = (0..N_CENTROIDS)
        .map(|_| draw(&mut rng, &mut taken))
        .collect();
    let query_rows: Vec<usize> = (0..N_QUERIES).map(|_| draw(&mut rng, &mut taken)).collect();
    let db_rows: Vec<usize> = (0..N_DB).map(|_| draw(&mut rng, &mut taken)).collect();

    // Codebook: subspace s = slice s of each of the 256 real rows.
    let mut base = 0usize;
    let mut codebook: Vec<Vec<Vec<f32>>> = Vec::with_capacity(6);
    for sd in SUB_DIMS {
        let sub: Vec<Vec<f32>> = centroid_rows
            .iter()
            .map(|&ri| rows[ri][base..base + sd].to_vec())
            .collect();
        codebook.push(sub);
        base += sd;
    }

    // Encode database rows (shared by both arms): argmin cosine cell per subspace.
    let metric = AdcMetric::Cosine;
    let encode = |v: &[f32; DIM]| -> [u8; 6] {
        let mut code = [0u8; 6];
        let mut b = 0usize;
        for (s, sd) in SUB_DIMS.iter().enumerate() {
            let q = &v[b..b + sd];
            let mut best = (f32::INFINITY, 0usize);
            for (c, cent) in codebook[s].iter().enumerate() {
                let d = metric.cell(q, cent);
                if d < best.0 {
                    best = (d, c);
                }
            }
            code[s] = best.1 as u8;
            b += sd;
        }
        code
    };
    let db_codes: Vec<[u8; 6]> = db_rows.iter().map(|&ri| encode(&rows[ri])).collect();

    // Quantize one query's f32 tables to an n-level integer LUT with ONE
    // shared affine (min, span) across all 6 tables — the sum stays affine
    // in the f32 sum up to rounding, so ranking fidelity is what's measured.
    let quantize = |tables: &[[f32; 256]; 6], levels: u32| -> ([[u16; 256]; 6], f32, f32) {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for t in tables {
            for &c in t {
                if c.is_finite() {
                    lo = lo.min(c);
                    hi = hi.max(c);
                }
            }
        }
        let span = (hi - lo).max(1e-12);
        let maxq = (levels - 1) as f32;
        let mut out = [[0u16; 256]; 6];
        for (s, t) in tables.iter().enumerate() {
            for (c, &cell) in t.iter().enumerate() {
                out[s][c] = if cell.is_finite() {
                    (((cell - lo) / span * maxq).round()) as u16
                } else {
                    (levels - 1) as u16 // unreachable-far, mirrors +INF init
                };
            }
        }
        (out, lo, span)
    };

    // ── stats helpers (probe-local; canonical impls live in the excluded
    //    bgz-tensor lab crate, which the spine deliberately does not import) ──
    fn pearson(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for (a, b) in x.iter().zip(y) {
            let (dx, dy) = (a - mx, b - my);
            sxy += dx * dy;
            sxx += dx * dx;
            syy += dy * dy;
        }
        sxy / (sxx.sqrt() * syy.sqrt()).max(1e-300)
    }
    fn ranks(v: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).expect("finite").then(a.cmp(&b)));
        let mut r = vec![0f64; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && (v[idx[j + 1]] - v[idx[i]]).abs() < 1e-12 {
                j += 1;
            }
            let avg = (i + j) as f64 / 2.0 + 1.0; // average rank for ties
            for k in i..=j {
                r[idx[k]] = avg;
            }
            i = j + 1;
        }
        r
    }
    fn spearman(x: &[f64], y: &[f64]) -> f64 {
        pearson(&ranks(x), &ranks(y))
    }
    /// ICC(2,1) two-way random, absolute agreement, on min-max-normalized cols.
    fn icc_2_1(x: &[f64], y: &[f64]) -> f64 {
        let norm = |v: &[f64]| -> Vec<f64> {
            let (lo, hi) = v
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &a| {
                    (l.min(a), h.max(a))
                });
            let s = (hi - lo).max(1e-300);
            v.iter().map(|&a| (a - lo) / s).collect()
        };
        let (x, y) = (norm(x), norm(y));
        let n = x.len() as f64; // targets
        let k = 2.0; // raters
        let grand = (x.iter().sum::<f64>() + y.iter().sum::<f64>()) / (n * k);
        let (mr1, mr2) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let mut ss_rows = 0.0;
        let mut ss_err = 0.0;
        for (a, b) in x.iter().zip(&y) {
            let mrow = (a + b) / 2.0;
            ss_rows += k * (mrow - grand).powi(2);
            ss_err += (a - mrow).powi(2) + (b - mrow).powi(2);
        }
        let ss_cols = n * ((mr1 - grand).powi(2) + (mr2 - grand).powi(2));
        let ss_e = ss_err - ss_cols; // residual after removing rater effect
        let msr = ss_rows / (n - 1.0);
        let msc = ss_cols / (k - 1.0);
        let mse = (ss_e / ((n - 1.0) * (k - 1.0))).max(0.0);
        (msr - mse) / (msr + (k - 1.0) * mse + k * (msc - mse) / n)
    }
    fn recall_at_k(before: &[f64], after: &[f64], k: usize) -> f64 {
        let top = |v: &[f64]| -> Vec<usize> {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            // deterministic total order: distance, then index (tie-breaker)
            idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).expect("finite").then(a.cmp(&b)));
            idx.truncate(k);
            idx
        };
        let (ta, tb) = (top(before), top(after));
        let hits = ta.iter().filter(|i| tb.contains(i)).count();
        hits as f64 / k as f64
    }

    // ── the head-to-head ──
    let adc = ScalarAdc::new(metric);
    let mut sp8 = Vec::with_capacity(N_QUERIES);
    let mut pr8 = Vec::with_capacity(N_QUERIES);
    let mut ic8 = Vec::with_capacity(N_QUERIES);
    let mut rc8 = Vec::with_capacity(N_QUERIES);
    let mut sp4 = Vec::with_capacity(N_QUERIES);
    let mut ns_f32 = 0u128;
    let mut ns_u8 = 0u128;

    for &qi in &query_rows {
        let q: Vec<f32> = rows[qi].to_vec();
        let tables = adc.precompute(&q, &codebook);

        // BEFORE arm: shipped f32 path (timed).
        let t0 = Instant::now();
        let before_f32 = adc.distance_batch(&tables, &db_codes);
        ns_f32 += t0.elapsed().as_nanos();
        let before: Vec<f64> = before_f32.iter().map(|&d| f64::from(d)).collect();

        // AFTER arm: u8 LUT, integer accumulate (timed).
        let (lut8, _lo, _span) = quantize(&tables, 256);
        let t1 = Instant::now();
        let after_u8: Vec<u32> = db_codes
            .iter()
            .map(|code| {
                let mut s = 0u32;
                for (sub, t) in lut8.iter().enumerate() {
                    s += u32::from(t[code[sub] as usize]);
                }
                s
            })
            .collect();
        ns_u8 += t1.elapsed().as_nanos();
        let after: Vec<f64> = after_u8.iter().map(|&d| f64::from(d)).collect();

        sp8.push(spearman(&before, &after));
        pr8.push(pearson(&before, &after));
        ic8.push(icc_2_1(&before, &after));
        rc8.push(recall_at_k(&before, &after, TOP_K));

        // Falsifier arm: 16 levels must be measurably worse than 256.
        let (lut4, _, _) = quantize(&tables, 16);
        let after4: Vec<f64> = db_codes
            .iter()
            .map(|code| {
                let mut s = 0u32;
                for (sub, t) in lut4.iter().enumerate() {
                    s += u32::from(t[code[sub] as usize]);
                }
                f64::from(s)
            })
            .collect();
        sp4.push(spearman(&before, &after4));
    }

    let stats = |v: &[f64]| -> (f64, f64, f64) {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &a in v {
            lo = lo.min(a);
            hi = hi.max(a);
        }
        (mean, lo, hi)
    };
    let (sp_m, sp_lo, sp_hi) = stats(&sp8);
    let (pr_m, pr_lo, pr_hi) = stats(&pr8);
    let (ic_m, ic_lo, ic_hi) = stats(&ic8);
    let (rc_m, rc_lo, rc_hi) = stats(&rc8);
    let (s4_m, _, s4_hi) = stats(&sp4);
    let per_cand = (N_QUERIES * N_DB) as u128;

    println!("\n== ADC cosine head-to-head: f32 tables vs u8 6x256 LUT ==");
    println!("queries={N_QUERIES} candidates={N_DB} codebook=6x{N_CENTROIDS} (real rows, seed {SEED:#x})");
    println!("Spearman rho  mean {sp_m:.4}  min {sp_lo:.4}  max {sp_hi:.4}");
    println!("Pearson  r    mean {pr_m:.4}  min {pr_lo:.4}  max {pr_hi:.4}");
    println!("ICC(2,1)      mean {ic_m:.4}  min {ic_lo:.4}  max {ic_hi:.4}");
    println!("recall@{TOP_K}     mean {rc_m:.4}  min {rc_lo:.4}  max {rc_hi:.4}");
    println!("ablation 4-bit Spearman mean {s4_m:.4} (max {s4_hi:.4})");
    println!(
        "timing: f32 {} ns/cand   u8 {} ns/cand (debug build; relative only)",
        ns_f32 / per_cand,
        ns_u8 / per_cand
    );

    // Verdicts (operator band 0.9973..0.9995; falsifier must fire).
    let pass_band = sp_m >= 0.9973;
    let falsifier_fires = s4_m < sp_m;
    println!(
        "\nPASS band (mean rho >= 0.9973): {}",
        if pass_band { "YES" } else { "NO" }
    );
    println!(
        "falsifier fires (4-bit < 8-bit): {}",
        if falsifier_fires {
            "YES"
        } else {
            "NO — harness cannot detect degradation; result INVALID"
        }
    );
    assert!(
        falsifier_fires,
        "falsifier must fire for the result to be meaningful"
    );
}
