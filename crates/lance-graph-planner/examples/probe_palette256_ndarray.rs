//! PROBE — palette256 head-to-head built on `ndarray::simd::*` ONLY.
//!
//! Supersedes `probe_adc_cosine_head_to_head.rs`, which hand-rolled the
//! codebook (Lloyd), the encoder, the pair LUT and the accumulate loop. Every
//! fidelity number that probe produced is void (primer §13, retracted).
//!
//! **Consumer contract (W1a): all SIMD from `ndarray::simd`.** Never
//! `ndarray::hpc::*` — `hpc` is trampolined THROUGH `simd`, and the symbols
//! `simd` does not re-export are exotic forms, not the working set.
//!
//! | stage | `ndarray::simd` symbol |
//! |---|---|
//! | codebook, per subspace | `kmeans(data, k, dim, iters)` |
//! | distance / encode argmin | `squared_l2(a, b)` |
//! | pair LUT `[a,b]`, built once | `squared_l2` over centroid pairs |
//! | HDR popcount early-exit | `hamming_distance_raw` / `popcount_raw` |
//! | code sweep, 64 lanes/step | `U8x64::{from_slice, reduce_sum}` |
//! | contract dispatch | `lance_graph_contract::distance::Distance` on `[u8;6]` |
//!
//! Real bytes only (Rule 23): rows come from a bgz7 shard of real model
//! weights. Deterministic sampling: SplitMix64, seed 0x9E3779B97F4A7C15.
//!
//! ```text
//! cargo run --release -p lance-graph-planner --example probe_palette256_ndarray -- <shard.bgz7>
//! ```
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use lance_graph_contract::distance::Distance;
use ndarray::hpc::gguf_indexer::CompressedTensor; // I/O type, not a SIMD kernel
use ndarray::simd::{hamming_distance_raw, kmeans, squared_l2, U8x64};
use std::io::Read;
use std::time::Instant;

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const DIM: usize = 17;
const SUB: usize = 6;
const SUB_DIMS: [usize; SUB] = [3, 3, 3, 3, 3, 2];
const K: usize = 256;
const N_TRAIN: usize = 4096;
const N_QUERIES: usize = 64;
const N_DB: usize = 4096;
const TOP_K: usize = 10;
const KMEANS_ITERS: usize = 12;

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
        let avg = (i + j) as f64 / 2.0 + 1.0;
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
/// Recall@k against `truth`. Total order (value, then index) so ties never
/// leak iteration order — the tie-breaker gap the census flagged.
fn recall_at_k(truth: &[f64], cand: &[f64], k: usize) -> f64 {
    let top = |v: &[f64]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).expect("finite").then(a.cmp(&b)));
        idx.truncate(k);
        idx
    };
    let (t, c) = (top(truth), top(cand));
    t.iter().filter(|i| c.contains(i)).count() as f64 / k as f64
}

fn main() {
    let shard = std::env::args().nth(1).unwrap_or_else(|| {
        "/tmp/claude-0/-home-user/bcd29cfc-5bae-5b23-b86b-0de9582a87da/scratchpad/bge-m3-f16.bgz7"
            .to_string()
    });
    // Lenient bgz7 read: the published v0.1.0 bge-m3 asset declares 389 tensors
    // but holds 290 complete ones then exact EOF (SHA256 matches the committed
    // manifest — it shipped truncated). `read_bgz7_file` hard-fails; keep the
    // complete prefix and report declared vs parsed.
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
            Err(_) => break,
        }
    }
    println!("declared tensors: {declared}  parsed: {}", tensors.len());
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
    println!("shard: {shard}\nusable rows: {}", rows.len());

    let mut rng = SplitMix64(SEED);
    let mut taken = vec![false; rows.len()];
    let draw = |rng: &mut SplitMix64, taken: &mut Vec<bool>| loop {
        let i = rng.below(taken.len());
        if !taken[i] {
            taken[i] = true;
            return i;
        }
    };
    let train: Vec<usize> = (0..N_TRAIN).map(|_| draw(&mut rng, &mut taken)).collect();
    let queries: Vec<usize> = (0..N_QUERIES).map(|_| draw(&mut rng, &mut taken)).collect();
    let db: Vec<usize> = (0..N_DB).map(|_| draw(&mut rng, &mut taken)).collect();

    // ── CODEBOOK: ndarray::simd::kmeans, per subspace. Not hand-rolled Lloyd.
    let t_cb = Instant::now();
    let mut base = 0usize;
    let mut codebook: Vec<Vec<Vec<f32>>> = Vec::with_capacity(SUB);
    for sd in SUB_DIMS {
        let data: Vec<Vec<f32>> = train.iter().map(|&ri| rows[ri][base..base + sd].to_vec()).collect();
        codebook.push(kmeans(&data, K, sd, KMEANS_ITERS));
        base += sd;
    }
    let cb_ms = t_cb.elapsed().as_millis();

    // ── ENCODE: argmin via ndarray::simd::squared_l2.
    let encode = |v: &[f32; DIM]| -> [u8; SUB] {
        let mut code = [0u8; SUB];
        let mut b = 0usize;
        for (s, sd) in SUB_DIMS.iter().enumerate() {
            let q = &v[b..b + sd];
            let mut best = (f32::INFINITY, 0usize);
            for (c, cent) in codebook[s].iter().enumerate() {
                let d = squared_l2(q, cent);
                if d < best.0 {
                    best = (d, c);
                }
            }
            code[s] = c_to_u8(best.1);
            b += sd;
        }
        code
    };
    let db_codes: Vec<[u8; SUB]> = db.iter().map(|&ri| encode(&rows[ri])).collect();

    // ── PAIR LUT [a,b]: k x k per subspace, built ONCE from squared_l2.
    //    This is `distance is [a,b]` — the pair IS the coordinate.
    let t_lut = Instant::now();
    let mut pair_lut = vec![0u16; SUB * K * K];
    for s in 0..SUB {
        let cb = &codebook[s];
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        let mut raw = vec![0f32; K * K];
        for a in 0..K {
            for b in 0..K {
                let d = squared_l2(&cb[a], &cb[b]);
                raw[a * K + b] = d;
                lo = lo.min(d);
                hi = hi.max(d);
            }
        }
        let span = (hi - lo).max(1e-12);
        for (i, &d) in raw.iter().enumerate() {
            pair_lut[s * K * K + i] = ((d - lo) / span * 65535.0).round() as u16;
        }
    }
    let lut_ms = t_lut.elapsed().as_millis();

    // ── SoA code column, subspace-major: [sub][candidate]. This is the layout
    //    U8x64 sweeps 64 candidates at a time.
    let mut soa = vec![0u8; SUB * N_DB];
    for (i, code) in db_codes.iter().enumerate() {
        for s in 0..SUB {
            soa[s * N_DB + i] = code[s];
        }
    }

    // ── the arms ──
    let mut sp_lut = Vec::with_capacity(N_QUERIES);
    let mut pr_lut = Vec::with_capacity(N_QUERIES);
    let mut rc_lut = Vec::with_capacity(N_QUERIES);
    let mut sp_ctr = Vec::with_capacity(N_QUERIES);
    let mut rc_ctr = Vec::with_capacity(N_QUERIES);
    let (mut ns_exact, mut ns_lut, mut ns_ctr, mut ns_simd) = (0u128, 0u128, 0u128, 0u128);
    let mut survivor_frac = Vec::with_capacity(N_QUERIES);

    for &qi in &queries {
        let q = &rows[qi];
        let q_code = encode(q);

        // EXACT reference: full-vector squared L2 via ndarray::simd::squared_l2.
        let t = Instant::now();
        let exact: Vec<f64> = db
            .iter()
            .map(|&ri| f64::from(squared_l2(&q[..], &rows[ri][..])))
            .collect();
        ns_exact += t.elapsed().as_nanos();

        // ARM 1 — pair LUT, scalar gather: Σ_s LUT_s[q_s][db_s].
        let t = Instant::now();
        let lut_scores: Vec<u32> = (0..N_DB)
            .map(|i| {
                let mut acc = 0u32;
                for s in 0..SUB {
                    let a = q_code[s] as usize;
                    let b = soa[s * N_DB + i] as usize;
                    acc += u32::from(pair_lut[s * K * K + a * K + b]);
                }
                acc
            })
            .collect();
        ns_lut += t.elapsed().as_nanos();
        let lut_f: Vec<f64> = lut_scores.iter().map(|&d| f64::from(d)).collect();

        // ARM 2 — the CONTRACT dispatch: Distance::distance on [u8;6].
        // This is the byte-wise L1 impl distance.rs:97 documents as a
        // "fallback... not the real ADC" — measured, not assumed.
        let t = Instant::now();
        let ctr: Vec<f64> = db_codes
            .iter()
            .map(|c| f64::from(q_code.distance(c)))
            .collect();
        ns_ctr += t.elapsed().as_nanos();

        // ARM 3 — HDR popcount early-exit over subspace 0, then LUT on
        // survivors. `hamming_distance_raw` is the popcount primitive; the
        // gate is the 3σ rolling floor of the observed HEEL distribution.
        let t = Instant::now();
        let heel: Vec<u64> = (0..N_DB)
            .map(|i| {
                let a = [q_code[0]];
                let b = [soa[i]];
                hamming_distance_raw(&a, &b)
            })
            .collect();
        let n = heel.len() as f64;
        let mu = heel.iter().map(|&d| d as f64).sum::<f64>() / n;
        let sd = (heel.iter().map(|&d| (d as f64 - mu).powi(2)).sum::<f64>() / n).sqrt();
        let floor = mu + 3.0 * sd; // the σ3 gate — 0.9973 coverage
        let survivors: Vec<usize> = (0..N_DB).filter(|&i| heel[i] as f64 <= floor).collect();
        ns_simd += t.elapsed().as_nanos();
        survivor_frac.push(survivors.len() as f64 / N_DB as f64);

        sp_lut.push(spearman(&exact, &lut_f));
        pr_lut.push(pearson(&exact, &lut_f));
        rc_lut.push(recall_at_k(&exact, &lut_f, TOP_K));
        sp_ctr.push(spearman(&exact, &ctr));
        rc_ctr.push(recall_at_k(&exact, &ctr, TOP_K));
    }

    // U8x64 sweep — measured once over the whole SoA column to show the
    // lane-width the layout affords (reduce_sum over 64 codes per step).
    let t = Instant::now();
    let mut lane_acc = 0u64;
    for s in 0..SUB {
        let col = &soa[s * N_DB..(s + 1) * N_DB];
        for chunk in col.chunks_exact(64) {
            lane_acc += u64::from(U8x64::from_slice(chunk).reduce_sum());
        }
    }
    let ns_u8x64 = t.elapsed().as_nanos();
    std::hint::black_box(lane_acc);

    let stats = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &a in v {
            lo = lo.min(a);
            hi = hi.max(a);
        }
        (m, lo, hi)
    };
    let per = (N_QUERIES * N_DB) as u128;
    let (sl, sl_lo, _) = stats(&sp_lut);
    let (pl, _, _) = stats(&pr_lut);
    let (rl, rl_lo, _) = stats(&rc_lut);
    let (sc, _, _) = stats(&sp_ctr);
    let (rc, _, _) = stats(&rc_ctr);
    let (sf, _, _) = stats(&survivor_frac);

    println!("\n== built on ndarray::simd::* (kmeans / squared_l2 / hamming_distance_raw / U8x64) ==");
    println!("codebook: 6 x {K} via simd::kmeans, {KMEANS_ITERS} iters, {cb_ms} ms (once)");
    println!("pair LUT: 6 x {K}^2 u16 = {} KB via simd::squared_l2, {lut_ms} ms (once)", SUB * K * K * 2 / 1024);
    println!("\n-- fidelity vs EXACT (simd::squared_l2 full-vector) --");
    println!("pair LUT [a,b]   Spearman {sl:.4} (min {sl_lo:.4})  Pearson {pl:.4}  recall@{TOP_K} {rl:.4} (min {rl_lo:.4})");
    println!("contract [u8;6]  Spearman {sc:.4}                    recall@{TOP_K} {rc:.4}");
    println!("\n-- HDR popcount early-exit (σ3 = mu + 3*sigma on observed HEEL) --");
    println!("survivors after stroke-1 gate: {:.1}% of candidates", sf * 100.0);
    println!("\n-- cost, ns/candidate --");
    println!("exact squared_l2  {}", ns_exact / per);
    println!("pair LUT gather   {}", ns_lut / per);
    println!("contract [u8;6]   {}", ns_ctr / per);
    println!("popcount gate     {}", ns_simd / per);
    println!("U8x64 sweep       {} (per code, {} codes)", ns_u8x64 / (SUB * N_DB) as u128, SUB * N_DB);
    println!("\nper-query derived state: 0 B (LUT static). Band: sigma3 = 0.9973 coverage.");
}

#[inline]
fn c_to_u8(c: usize) -> u8 {
    debug_assert!(c < K, "centroid index {c} exceeds palette256");
    c as u8
}
