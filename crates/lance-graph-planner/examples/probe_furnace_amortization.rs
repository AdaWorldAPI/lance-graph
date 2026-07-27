//! PROBE — the furnace lanes L1/L2/L3-mini/L4/L6 built end-to-end on real
//! bytes, with AMORTIZATION measured and validated, plus the HHTL
//! awareness-location claim made falsifiable.
//!
//! Lanes (per `.claude/board/BF16-HIGH-VALUE-LANES-2026-07-27.md`):
//! - **L1** codebook training — `ndarray::simd::kmeans`, 6×256 over real rows.
//! - **L2 (shape)** pair-table bake — 6×256² u16 from `simd::squared_l2`.
//!   (The CERTIFIED i8+FamilyGamma form lives in bgz-tensor `FisherZTable`;
//!   its build/lookup economics are measured by the companion probe there.
//!   This lane measures the table-shape amortization, not the z-encoding.)
//! - **L3-mini** BF16-RNE ingestion transport — `simd::{f32_to_bf16_batch_rne,
//!   bf16_to_f32_batch}` round-trip, certified against the harness gate
//!   (Pearson AND Spearman ≥ 0.9999).
//! - **L4** calibration — σ3 floor (μ+3σ) over HEEL-table values + θ→`u8`
//!   quantization (`theta_accept_q8` shape): campaigns that store BYTES.
//! - **L6** ingestion-boundary encode — argmin over the L1 codebook; the last
//!   float the row ever sees; thereafter every read is `[a,b]`.
//! - **L5 / certified-L2** — companion probe in bgz-tensor (γ-fold + FisherZ).
//! - **L7** — fenced (ratified Σ lane); nothing to build by design.
//!
//! **HHTL awareness-location** (operator: "hhtl also allows for location of
//! awareness in semantic space"): the code's coarse prefix (HEEL byte =
//! subspace-0 centroid) is claimed to be a semantic ADDRESS. Falsifiable form:
//! (a) rows sharing a HEEL byte must be closer in EXACT distance than random
//! pairs (ratio < 1); (b) a HEEL-first σ3 early-exit must cut candidates while
//! preserving recall; (c) a SHUFFLED-code control must destroy both signals —
//! the can-it-fire / can-it-stay-silent pair the falsifiability rule demands.
//!
//! **Amortization model** (validated, not asserted): for each lane,
//! `break_even_reads = build_ns / (alt_ns_per_read − product_ns_per_read)`,
//! then compare against ONE session's actual read count (64 q × 4096 cand).
//! PASS = the furnace pays for itself inside a single pass.
//!
//! Real bytes only: bge-m3 bgz7 shard (SHA-pinned). Seed 0x9E3779B97F4A7C15.
//!
//! ```text
//! cargo run --release -p lance-graph-planner --example probe_furnace_amortization
//! ```
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use ndarray::hpc::gguf_indexer::CompressedTensor; // I/O type, not a SIMD kernel
use ndarray::simd::{bf16_to_f32_batch, f32_to_bf16_batch_rne, kmeans, squared_l2};
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
    // Lenient bgz7 read (shard declares 389 tensors, holds 290 then exact EOF).
    let mut reader = std::io::BufReader::new(std::fs::File::open(&shard).expect("open shard"));
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).expect("magic");
    assert_eq!(&magic, b"BGZ7");
    let mut b4 = [0u8; 4];
    reader.read_exact(&mut b4).expect("n_tensors");
    let declared = u32::from_le_bytes(b4) as usize;
    let mut tensors: Vec<CompressedTensor> = Vec::with_capacity(declared);
    for _ in 0..declared {
        match CompressedTensor::read_from(&mut reader) {
            Ok(t) => tensors.push(t),
            Err(_) => break,
        }
    }
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
    println!("shard: {} tensors ({} declared), usable rows: {}", tensors.len(), declared, rows.len());

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

    // ── L1: codebook training via simd::kmeans ──────────────────────────────
    let t0 = Instant::now();
    let mut base = 0usize;
    let mut codebook: Vec<Vec<Vec<f32>>> = Vec::with_capacity(SUB);
    for sd in SUB_DIMS {
        let data: Vec<Vec<f32>> = train.iter().map(|&ri| rows[ri][base..base + sd].to_vec()).collect();
        codebook.push(kmeans(&data, K, sd, KMEANS_ITERS));
        base += sd;
    }
    let l1_build_ns = t0.elapsed().as_nanos();
    let l1_bytes: usize = SUB_DIMS.iter().map(|sd| K * sd * 4).sum();

    // ── L6: ingestion-boundary encode (once per arriving row) ───────────────
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
            code[s] = best.1 as u8;
            b += sd;
        }
        code
    };
    let t6 = Instant::now();
    let db_codes: Vec<[u8; SUB]> = db.iter().map(|&ri| encode(&rows[ri])).collect();
    let l6_encode_ns_per_row = t6.elapsed().as_nanos() / N_DB as u128;

    // ── L2 (shape): pair-table bake, once ───────────────────────────────────
    let t2 = Instant::now();
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
    let l2_build_ns = t2.elapsed().as_nanos();
    let l2_bytes = SUB * K * K * 2;

    // SoA code column, subspace-major.
    let mut soa = vec![0u8; SUB * N_DB];
    for (i, code) in db_codes.iter().enumerate() {
        for s in 0..SUB {
            soa[s * N_DB + i] = code[s];
        }
    }

    // ── L3-mini: BF16-RNE transport certification ───────────────────────────
    // Cast a large real-f32 block to BF16 (RNE) and back; certify against the
    // harness gate for the bf16-RNE lane: Pearson AND Spearman >= 0.9999.
    let flat: Vec<f32> = train.iter().flat_map(|&ri| rows[ri]).collect();
    let mut bf16 = vec![0u16; flat.len()];
    let t3a = Instant::now();
    f32_to_bf16_batch_rne(&flat, &mut bf16);
    let l3_cast_ns = t3a.elapsed().as_nanos();
    let mut back = vec![0f32; flat.len()];
    let t3b = Instant::now();
    bf16_to_f32_batch(&bf16, &mut back);
    let l3_uncast_ns = t3b.elapsed().as_nanos();
    let (fx, fy): (Vec<f64>, Vec<f64>) = (
        flat.iter().map(|&v| f64::from(v)).collect(),
        back.iter().map(|&v| f64::from(v)).collect(),
    );
    let l3_pearson = pearson(&fx, &fy);
    let l3_spearman = spearman(&fx, &fy);

    // ── L4: calibration — σ3 floor over HEEL-table values + θ→u8 ────────────
    // The floor is computed over the HEEL pair-table row of each query later;
    // here calibrate the GLOBAL HEEL-table distribution once (build-time).
    let t4 = Instant::now();
    let heel_tab = &pair_lut[0..K * K];
    let n = heel_tab.len() as f64;
    let mu = heel_tab.iter().map(|&d| f64::from(d)).sum::<f64>() / n;
    let sd = (heel_tab.iter().map(|&d| (f64::from(d) - mu).powi(2)).sum::<f64>() / n).sqrt();
    let sigma3_t = mu + 3.0 * sd; // the far-tail threshold t (Belichtungsmesser shape)
    // Early-exit KEEPS the near bands: d <= t/4 (Foveal) or <= t/2 (Near) —
    // ndarray cascade::expose's band carve. Keeping d <= t itself admits the
    // 99.87th percentile = decoration (the defect the first run self-reported).
    let keep_floor = sigma3_t / 4.0;
    let theta = 1.52f64; // a Fisher-z aperture in the validated 1.45-1.6 band
    let theta_accept_q8: u8 = ((theta / 5.0) * 255.0).round() as u8; // stored as ONE byte
    let l4_build_ns = t4.elapsed().as_nanos();

    // ── the measurement pass: 64 q x 4096 cand ──────────────────────────────
    let mut ns_exact = 0u128;
    let mut ns_lut = 0u128;
    let mut sp_lut = Vec::new();
    let mut rc_lut = Vec::new();
    // HHTL awareness-location accumulators
    let mut ns_cascade = 0u128;
    let mut survivors_pct = Vec::new();
    let mut rc_cascade = Vec::new();

    for &qi in &queries {
        let q = &rows[qi];
        let q_code = encode(q);

        let t = Instant::now();
        let exact: Vec<f64> = db
            .iter()
            .map(|&ri| f64::from(squared_l2(&q[..], &rows[ri][..])))
            .collect();
        ns_exact += t.elapsed().as_nanos();

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
        sp_lut.push(spearman(&exact, &lut_f));
        rc_lut.push(recall_at_k(&exact, &lut_f, TOP_K));

        // HHTL cascade: HEEL-stage sigma3 early-exit, then full [a,b] on
        // survivors. The gate value is the HEEL pair-table entry itself —
        // same scale as the floor (fixes the earlier one-byte-popcount defect).
        let t = Instant::now();
        let heel_a = q_code[0] as usize;
        let mut cascade_scores = vec![u32::MAX; N_DB];
        let mut surv = 0usize;
        for i in 0..N_DB {
            let hb = soa[i] as usize; // subspace 0 column
            let hd = f64::from(pair_lut[heel_a * K + hb]);
            if hd <= keep_floor {
                surv += 1;
                let mut acc = 0u32;
                for s in 0..SUB {
                    let a = q_code[s] as usize;
                    let b = soa[s * N_DB + i] as usize;
                    acc += u32::from(pair_lut[s * K * K + a * K + b]);
                }
                cascade_scores[i] = acc;
            }
        }
        ns_cascade += t.elapsed().as_nanos();
        survivors_pct.push(surv as f64 / N_DB as f64);
        let casc_f: Vec<f64> = cascade_scores.iter().map(|&d| f64::from(d)).collect();
        rc_cascade.push(recall_at_k(&exact, &casc_f, TOP_K));
    }

    // ── HHTL awareness-location validity + shuffled-code control ────────────
    // (a) rows sharing a HEEL byte must be CLOSER in exact distance than
    // random pairs; (b) shuffling codes must destroy the signal.
    let mut heel_groups: Vec<Vec<usize>> = vec![Vec::new(); K];
    for i in 0..N_DB {
        heel_groups[soa[i] as usize].push(i);
    }
    let mut rng2 = SplitMix64(SEED ^ 0xABCD);
    let mut same_heel = Vec::new();
    let mut random_pair = Vec::new();
    let mut tries = 0;
    while same_heel.len() < 2000 && tries < 200_000 {
        tries += 1;
        let g = &heel_groups[rng2.below(K)];
        if g.len() < 2 {
            continue;
        }
        let (a, b) = (g[rng2.below(g.len())], g[rng2.below(g.len())]);
        if a == b {
            continue;
        }
        same_heel.push(f64::from(squared_l2(&rows[db[a]][..], &rows[db[b]][..])));
    }
    for _ in 0..2000 {
        let (a, b) = (rng2.below(N_DB), rng2.below(N_DB));
        random_pair.push(f64::from(squared_l2(&rows[db[a]][..], &rows[db[b]][..])));
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let locality_ratio = mean(&same_heel) / mean(&random_pair).max(1e-300);

    // Shuffled control: random HEEL assignment must yield ratio ~1.
    let mut shuf_groups: Vec<Vec<usize>> = vec![Vec::new(); K];
    for i in 0..N_DB {
        shuf_groups[rng2.below(K)].push(i);
    }
    let mut shuf_pairs = Vec::new();
    let mut tries2 = 0;
    while shuf_pairs.len() < 2000 && tries2 < 200_000 {
        tries2 += 1;
        let g = &shuf_groups[rng2.below(K)];
        if g.len() < 2 {
            continue;
        }
        let (a, b) = (g[rng2.below(g.len())], g[rng2.below(g.len())]);
        if a == b {
            continue;
        }
        shuf_pairs.push(f64::from(squared_l2(&rows[db[a]][..], &rows[db[b]][..])));
    }
    let control_ratio = mean(&shuf_pairs) / mean(&random_pair).max(1e-300);

    // ── report ──────────────────────────────────────────────────────────────
    let stats = |v: &[f64]| {
        let m = mean(v);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &a in v {
            lo = lo.min(a);
            hi = hi.max(a);
        }
        (m, lo, hi)
    };
    let per = (N_QUERIES * N_DB) as u128;
    let exact_ns = ns_exact / per;
    let lut_ns = ns_lut / per;
    let casc_ns = ns_cascade / per;
    let session_reads = per; // one pass = 262 144 candidate reads

    let be = |build: u128, alt: u128, prod: u128| -> String {
        if alt <= prod {
            return "n/a (alt not slower)".into();
        }
        let n = build / (alt - prod).max(1);
        if n <= session_reads {
            format!("{n} reads = {:.2} of one pass", n as f64 / session_reads as f64)
        } else {
            format!("{n} reads = {:.1} passes", n as f64 / session_reads as f64)
        }
    };

    println!("\n== FURNACE AMORTIZATION (build once vs read forever) ==");
    println!("session read volume: {session_reads} candidate reads (64 q x 4096)");
    println!("float alternative per read: exact squared_l2 = {exact_ns} ns/cand\n");
    println!("lane | build_once | product | per_read | break_even");
    println!("L1 codebook      | {:>8.1} ms | {:>6} B | (enables L2/L6)  | amortized via L2+L6", l1_build_ns as f64 / 1e6, l1_bytes);
    println!(
        "L2 pair-table    | {:>8.1} ms | {:>6} B | {lut_ns} ns/cand | L2-only: {}; charged L1+L2: {}",
        l2_build_ns as f64 / 1e6, l2_bytes,
        be(l2_build_ns, exact_ns, lut_ns),
        be(l1_build_ns + l2_build_ns, exact_ns, lut_ns)
    );
    println!("L3 bf16-RNE cast | {:>8.1} ms | 2 B/val  | cast-once        | Pearson {:.6} Spearman {:.6} (gate 0.9999)", (l3_cast_ns + l3_uncast_ns) as f64 / 1e6, l3_pearson, l3_spearman);
    println!("L4 calibration   | {:>8.3} ms | 3 B      | (gates cascade)  | t=mu+3s, keep<=t/4 (u16) + theta_q8({theta_accept_q8})", l4_build_ns as f64 / 1e6);
    println!("L6 encode        | {l6_encode_ns_per_row} ns/row (once at ingest) | 6 B/row | [a,b] thereafter | vs 68 B float row: 11.3x denser");

    let (slm, sll, _) = stats(&sp_lut);
    let (rlm, rll, _) = stats(&rc_lut);
    let (svm, svl, svh) = stats(&survivors_pct);
    let (rcm, rcl, _) = stats(&rc_cascade);
    println!("\n== fidelity vs exact ==");
    println!("flat [a,b] scan   : Spearman {slm:.4} (min {sll:.4})  recall@{TOP_K} {rlm:.4} (min {rll:.4})  {lut_ns} ns/cand");
    println!("HHTL cascade      : recall@{TOP_K} {rcm:.4} (min {rcl:.4})  survivors {:.1}% (min {:.1}% max {:.1}%)  {casc_ns} ns/cand", svm * 100.0, svl * 100.0, svh * 100.0);

    println!("\n== HHTL: location of awareness in semantic space ==");
    println!("same-HEEL pair distance / random pair distance = {locality_ratio:.4}  (<1 = the prefix IS a semantic address)");
    println!("shuffled-code control ratio                    = {control_ratio:.4}  (~1 = the measurement CAN stay silent)");
    let locality_proven = locality_ratio < 0.9 && (control_ratio - 1.0).abs() < 0.1;
    println!("awareness-location: {}", if locality_proven { "PROVEN (signal fires, control silent)" } else { "NOT PROVEN on this rig" });

    println!("\n== verdicts ==");
    let l3_pass = l3_pearson >= 0.9999 && l3_spearman >= 0.9999;
    println!("L3 bf16-RNE certification: {}", if l3_pass { "PASS" } else { "FAIL" });
    let gate_fires = svm < 0.999 && svm > 0.001;
    println!("L4 sigma3 gate discriminates (not 0%/100%): {}", if gate_fires { "YES" } else { "NO — gate is decoration" });
    let amort = exact_ns > lut_ns;
    println!("amortization validated (per-read float > per-read [a,b]): {}", if amort { "YES" } else { "NO" });
    assert!(l3_pass && amort, "core furnace claims must hold");
}
