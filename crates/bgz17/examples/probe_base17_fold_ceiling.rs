//! PROBE-BASE17-CEILING — quantify `TD-BASE17-FOLD-CEILING-SINGLE-WORD`.
//!
//! **Why this exists.** `PROBE-CODEBOOK-44` (real data) found that NEITHER the
//! hierarchical nor the flat 256-codebook cleared the 0.965/0.9973 anchors on
//! single-word dense embeddings, and attributed the cap to an upstream
//! "Base17 17-dim golden-fold ceiling rho=0.2599" — recorded but never
//! isolated. Until the ceiling is measured, any downstream probe run on folded
//! rows (e.g. the WordNet-ancestry falsifier) risks reading an upstream cap as
//! its own failure.
//!
//! **What is measured.** Spearman rho / Pearson r of an encoded distance
//! against TWO ground truths on the same real pairs:
//!   - `1 - cosine` on the full-dimension f32 rows (the semantic reference), and
//!   - `L1` on the full-dimension f32 rows (the metric-matched reference).
//! Reporting both separates "the FOLD loses information" from "L1-on-means is
//! the wrong metric for a cosine question" — two different defects with two
//! different fixes.
//!
//! **The encoders (all 17 numbers per row, so dimension is held constant):**
//!   1. `golden`   — `Base17` residue-class means at the canon `GOLDEN_STEP=11`.
//!   2. `step-s`   — the same fold at other steps coprime to 17.
//!   3. `block`    — contiguous block means (17 blocks), NOT residue classes.
//!   4. `jl`       — a random Gaussian Johnson-Lindenstrauss projection to 17.
//!
//! `jl` is the reference that makes the result actionable: it is what 17
//! dimensions are WORTH when the projection is chosen well. If `golden` ~ `jl`,
//! the ceiling is dimensional and no fold can beat it. If `jl` >> `golden`, the
//! ceiling is the FOLD, not the dimension — a fixable defect.
//!
//! **Falsifiers (this probe can fail):**
//!   - RELABEL: `GOLDEN_POS[i] = (i*step) % 17` is a permutation, and the fold
//!     buckets index `octave*17 + GOLDEN_POS[i]`, i.e. residue class
//!     `GOLDEN_POS[i]` (mod 17). So the step only permutes bucket LABELS, and
//!     any symmetric readout (L1, sign agreement) MUST be bit-identical across
//!     coprime steps. If steps disagree, this reading of the code is WRONG.
//!   - NON-VACUITY: `jl` at 17 dims must lose something (rho < 0.999) and must
//!     retain something (rho > 0.1). Either bound failing means the harness,
//!     not the fold, is being measured.
//!   - SPREAD: the ground-truth distances must not be near-constant.
//!
//! Real bytes only (Rule 23). Deterministic SplitMix64, seed 0x9E3779B97F4A7C15.
//!
//! Input: `<emb.f32>` = `[u32 n][u32 dim]` + n*dim f32 LE. Any real embedding
//! matrix works; the finding is reported per dimension width so the trend, not
//! one number, is the result. Reproduce the reference input with:
//!
//! ```text
//! curl -sSL -o /tmp/m.safetensors \
//!   https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors
//! # sha256 53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db
//! # then slice tensor `embeddings.word_embeddings.weight` F32 [30522, 384]
//! cargo run --release --manifest-path crates/bgz17/Cargo.toml \
//!   --example probe_base17_fold_ceiling -- <emb.f32>
//! ```
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use bgz17::{BASE_DIM, FP_SCALE};
use std::io::Read;

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const N_ROWS: usize = 4096;
const N_PAIRS: usize = 20_000;
/// Coprime-to-17 steps; 11 is the canon `GOLDEN_STEP`.
const STEPS: [usize; 7] = [1, 2, 3, 5, 7, 11, 13];
/// Truncation widths (all multiples of 17 so no bucket is short-changed).
const WIDTHS: [usize; 6] = [68, 136, 204, 272, 340, 374];

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
    /// Box-Muller standard normal (deterministic, no clock, no rand crate).
    fn normal(&mut self) -> f64 {
        let u1 = ((self.next() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

// ── statistics ───────────────────────────────────────────────────────────────

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
        let mut j = i + 1;
        while j < idx.len() && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1 + j) as f64) / 2.0;
        for &k in &idx[i..j] {
            r[k] = avg;
        }
        i = j;
    }
    r
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&ranks(x), &ranks(y))
}

fn stddev(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    (v.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / n).sqrt()
}

// ── ground truths on the full-width real vectors ─────────────────────────────

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na * nb).sqrt().max(1e-300)
}

fn l1_f32(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| f64::from(*x - *y).abs()).sum()
}

// ── the encoders: every one produces exactly BASE_DIM numbers ────────────────

/// The shipped fold, generalized over `step`. Bucket `i` accumulates source
/// dims `octave*17 + (i*step % 17)` — i.e. one residue class mod 17.
/// Quantized to i16 fixed-point exactly as `Base17::encode` does.
fn fold_residue(v: &[f32], step: usize) -> [f64; BASE_DIM] {
    let mut pos = [0usize; BASE_DIM];
    for (i, p) in pos.iter_mut().enumerate() {
        *p = (i * step) % BASE_DIM;
    }
    let n = v.len();
    let mut sum = [0f64; BASE_DIM];
    let mut cnt = [0u32; BASE_DIM];
    for octave in 0..n.div_ceil(BASE_DIM) {
        for (bi, &p) in pos.iter().enumerate() {
            let d = octave * BASE_DIM + p;
            if d < n {
                sum[bi] += f64::from(v[d]);
                cnt[bi] += 1;
            }
        }
    }
    let mut out = [0f64; BASE_DIM];
    for i in 0..BASE_DIM {
        if cnt[i] > 0 {
            // The i16 fixed-point quantization is part of the encoder under test.
            let q = ((sum[i] / f64::from(cnt[i])) * FP_SCALE)
                .round()
                .clamp(-32768.0, 32767.0);
            out[i] = q;
        }
    }
    out
}

/// The same residue-class fold WITHOUT the i16 fixed-point rounding.
/// Control: separates "17 dims is the cap" from "the STORED i16 is the cap".
/// Embedding coordinates are ~1e-2, so `mean * 256` rounds to single-digit
/// integers — a plausible second suspect that must be ruled out explicitly.
fn fold_residue_exact(v: &[f32], step: usize) -> [f64; BASE_DIM] {
    let mut pos = [0usize; BASE_DIM];
    for (i, p) in pos.iter_mut().enumerate() {
        *p = (i * step) % BASE_DIM;
    }
    let n = v.len();
    let mut sum = [0f64; BASE_DIM];
    let mut cnt = [0u32; BASE_DIM];
    for octave in 0..n.div_ceil(BASE_DIM) {
        for (bi, &p) in pos.iter().enumerate() {
            let d = octave * BASE_DIM + p;
            if d < n {
                sum[bi] += f64::from(v[d]);
                cnt[bi] += 1;
            }
        }
    }
    let mut out = [0f64; BASE_DIM];
    for i in 0..BASE_DIM {
        if cnt[i] > 0 {
            out[i] = sum[i] / f64::from(cnt[i]);
        }
    }
    out
}

/// Contiguous block means — same 17 outputs, different grouping.
fn fold_block(v: &[f32]) -> [f64; BASE_DIM] {
    let n = v.len();
    let w = n.div_ceil(BASE_DIM);
    let mut out = [0f64; BASE_DIM];
    for (i, slot) in out.iter_mut().enumerate() {
        let lo = i * w;
        let hi = ((i + 1) * w).min(n);
        if lo < hi {
            let s: f64 = v[lo..hi].iter().map(|x| f64::from(*x)).sum();
            *slot = ((s / (hi - lo) as f64) * FP_SCALE).round();
        }
    }
    out
}

/// Random Gaussian JL projection to 17 dims — what 17 dims are WORTH.
fn project_jl(v: &[f32], mat: &[f64]) -> [f64; BASE_DIM] {
    let n = v.len();
    let mut out = [0f64; BASE_DIM];
    for (i, slot) in out.iter_mut().enumerate() {
        let row = &mat[i * n..(i + 1) * n];
        *slot = row.iter().zip(v).map(|(m, x)| m * f64::from(*x)).sum();
    }
    out
}

fn l1_17(a: &[f64; BASE_DIM], b: &[f64; BASE_DIM]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!(
                "usage: probe_base17_fold_ceiling <emb.f32>\n\n\
                 Requires REAL embedding bytes (Rule 23 - no synthetic vectors).\n\
                 Format: [u32 n][u32 dim] + n*dim f32 LE.\n\n\
                 Reference input (all-MiniLM-L6-v2 word embeddings, 30522 x 384):\n  \
                 curl -sSL -o /tmp/m.safetensors \\\n    \
                 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors\n  \
                 sha256 53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db\n  \
                 then slice tensor `embeddings.word_embeddings.weight` into the format above."
            );
            std::process::exit(2);
        }
    };

    let mut buf = Vec::new();
    std::fs::File::open(&path)
        .unwrap_or_else(|e| panic!("open {path}: {e}"))
        .read_to_end(&mut buf)
        .expect("read");
    assert!(
        buf.len() > 8,
        "{path}: too short for a [u32 n][u32 dim] header"
    );
    let n = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    assert_eq!(
        buf.len(),
        8 + n * dim * 4,
        "{path}: header says {n} x {dim} f32 but file is {} bytes",
        buf.len()
    );
    let all: Vec<f32> = buf[8..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    println!("source: {path}\n  {n} rows x {dim} dims (real embedding weights)");

    let mut rng = SplitMix64(SEED);
    assert!(
        n >= N_ROWS,
        "need >= {N_ROWS} rows to sample from, file has {n}"
    );
    let mut taken = vec![false; n];
    let mut rows: Vec<usize> = Vec::with_capacity(N_ROWS);
    while rows.len() < N_ROWS {
        let i = rng.below(n);
        if !taken[i] {
            taken[i] = true;
            rows.push(i);
        }
    }
    let pairs: Vec<(usize, usize)> = (0..N_PAIRS)
        .map(|_| {
            let a = rng.below(N_ROWS);
            let mut b = rng.below(N_ROWS);
            while b == a {
                b = rng.below(N_ROWS);
            }
            (rows[a], rows[b])
        })
        .collect();
    println!("  sampled {N_ROWS} rows -> {N_PAIRS} pairs (SplitMix64 {SEED:#x})\n");

    // ── RELABEL FALSIFIER: coprime steps must be bit-identical under L1 ──────
    // If this fires, the golden step DOES carry information and the whole
    // "it only permutes bucket labels" reading below is wrong.
    {
        let v0 = &all[rows[0] * dim..(rows[0] + 1) * dim];
        let v1 = &all[rows[1] * dim..(rows[1] + 1) * dim];
        let base = l1_17(&fold_residue(v0, 11), &fold_residue(v1, 11));
        let mut disagree = Vec::new();
        for s in STEPS {
            let d = l1_17(&fold_residue(v0, s), &fold_residue(v1, s));
            if (d - base).abs() > 0.0 {
                disagree.push((s, d));
            }
        }
        println!("RELABEL falsifier (L1 across coprime steps {STEPS:?}):");
        if disagree.is_empty() {
            println!(
                "  all steps give L1 = {base} EXACTLY -> the golden step is a \
                 bucket RELABEL,\n  not an information-bearing choice. \
                 gcd(step,17)=1 makes it a permutation of\n  residue classes; \
                 L1 sums over all 17, so the labelling cancels.\n"
            );
        } else {
            println!(
                "  DISAGREEMENT at steps {disagree:?} (base {base}) -> relabel claim FALSIFIED\n"
            );
        }
        assert!(
            disagree.is_empty(),
            "coprime steps disagreed; the fold is not a pure relabel"
        );
    }

    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>10}   {:>10} {:>10}",
        "width", "golden", "exact", "block", "jl-17", "gold-vsL1", "CV"
    );
    println!("{}", "-".repeat(76));

    let mut widths: Vec<usize> = WIDTHS.iter().copied().filter(|w| *w <= dim).collect();
    if !widths.contains(&dim) {
        widths.push(dim);
    }

    let mut summary: Vec<(usize, f64, f64, f64)> = Vec::new();

    for w in widths {
        // A JL matrix per width, drawn once and shared by every pair.
        let jl: Vec<f64> = (0..BASE_DIM * w).map(|_| rng.normal()).collect();

        let slice = |r: usize| &all[r * dim..r * dim + w];

        let mut gt_cos = Vec::with_capacity(N_PAIRS);
        let mut gt_l1 = Vec::with_capacity(N_PAIRS);
        let mut d_gold = Vec::with_capacity(N_PAIRS);
        let mut d_exact = Vec::with_capacity(N_PAIRS);
        let mut d_block = Vec::with_capacity(N_PAIRS);
        let mut d_jl = Vec::with_capacity(N_PAIRS);

        for &(a, b) in &pairs {
            let (va, vb) = (slice(a), slice(b));
            gt_cos.push(1.0 - cosine(va, vb));
            gt_l1.push(l1_f32(va, vb));
            d_gold.push(l1_17(&fold_residue(va, 11), &fold_residue(vb, 11)));
            d_exact.push(l1_17(
                &fold_residue_exact(va, 11),
                &fold_residue_exact(vb, 11),
            ));
            d_block.push(l1_17(&fold_block(va), &fold_block(vb)));
            d_jl.push(l1_17(&project_jl(va, &jl), &project_jl(vb, &jl)));
        }

        // SPREAD guard: a near-constant ground truth makes every rho meaningless.
        assert!(
            stddev(&gt_cos) > 1e-3,
            "ground-truth cosine distance is near-constant (sd {:.2e}) at width {w}",
            stddev(&gt_cos)
        );

        // CV of the ground-truth distances places this input on the axis of the
        // existing CV sweep (`probe_base17_cv_sweep.rs`), which found rho rises
        // 0.22->0.856 as CV rises 0.22->1.0. Without it the two findings are
        // not commensurable.
        let cv = stddev(&gt_l1) / (gt_l1.iter().sum::<f64>() / gt_l1.len() as f64);
        let (rg, re, rb, rj) = (
            spearman(&d_gold, &gt_cos),
            spearman(&d_exact, &gt_cos),
            spearman(&d_block, &gt_cos),
            spearman(&d_jl, &gt_cos),
        );
        let rg_l1 = spearman(&d_gold, &gt_l1);

        println!("{w:>6} {rg:>10.4} {re:>10.4} {rb:>10.4} {rj:>10.4}   {rg_l1:>10.4} {cv:>8.4}");
        // QUANTIZATION control: if the stored i16 were the binding cap, the
        // unrounded fold would clear it by a wide margin. Assert it does not.
        assert!(
            re - rg < 0.15,
            "unquantized fold rho {re:.4} beats stored-i16 {rg:.4} by >0.15 at width {w}: \
             the cap is the i16 QUANTIZATION, not the dimension - revisit FP_SCALE"
        );
        summary.push((w, rg, rj, rg_l1));
    }

    // ── NON-VACUITY guards on the JL reference ──────────────────────────────
    let (_, _, jl_full, _) = *summary.last().expect("at least one width");
    assert!(
        jl_full < 0.999,
        "JL-17 rho {jl_full:.4} implies 17 dims lose nothing - harness suspect, not the fold"
    );
    assert!(
        jl_full > 0.10,
        "JL-17 rho {jl_full:.4} implies 17 dims carry nothing - harness suspect, not the fold"
    );

    let (w_full, gold_full, _, gold_l1_full) = *summary.last().expect("at least one width");
    println!("\n--- verdict (width {w_full}) ---");
    println!("  golden fold vs cosine : rho {gold_full:.4}   <- the CEILING under test");
    println!("  golden fold vs L1     : rho {gold_l1_full:.4}   <- same encoder, metric-matched");
    println!("  JL-17       vs cosine : rho {jl_full:.4}   <- what 17 dims are WORTH");
    let ratio = jl_full / gold_full.abs().max(1e-9);
    if ratio > 1.5 {
        println!(
            "\n  JL-17 recovers {ratio:.1}x the golden fold at IDENTICAL width.\n  \
             => the ceiling is the FOLD (residue-class mean pooling), NOT the 17 dims.\n  \
             Mean-pooling ~{} coordinates per bucket concentrates every bucket toward\n  \
             the row mean (sd shrinks ~1/sqrt(m)); a JL row keeps a random-sign\n  \
             combination and so keeps the distance.",
            w_full / BASE_DIM
        );
    } else {
        println!(
            "\n  JL-17 is within {ratio:.2}x of the golden fold at identical width\n  \
             => the ceiling is DIMENSIONAL (17 dims), not the fold's grouping."
        );
    }
    println!(
        "\n  Trend across widths shows how the ceiling moves with source dimension:\n  \
         each bucket averages width/17 coordinates, so a wider source is a LOWER\n  \
         ceiling for the fold and (per JL) a flat-to-better one for a real projection."
    );
}
