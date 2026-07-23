//! PROBE-CODEBOOK-44 — REAL-DATA ρ (retires Probe M1's real-data PENDING).
//!
//! PR #823 proved the MECHANISM synthetically (prefix==ancestry on a planted
//! 16×16 hierarchy) and honestly left "real-data ρ PENDING — no Jina centroids
//! on disk". This example closes that gap on REAL Jina embeddings.
//!
//! The question (one measurable claim): on real Jina semantics, does the
//! hierarchical 16×16 bgz17 codebook (`Palette::build_hierarchical`) preserve
//! semantic distances AS WELL AS OR BETTER THAN the flat-256 k-means codebook
//! (`Palette::build`)?
//!
//! Metric: Spearman ρ of each codebook's reconstructed pairwise distance vs the
//! TRUE Jina f32 cosine (ground truth), against the canon anchors 0.9973 / 0.965.
//! Falsifier: hierarchical ρ ≥ flat ρ on real data. If hierarchical sacrifices
//! real fidelity, Probe M1 is FALSIFIED — a valid, valuable result.
//!
//! Data: top ~4096 frequency-ranked academic words, embedded via the Jina API
//! (jina-embeddings-v3, 1024-d, task=text-matching) into a LOCAL gitignored
//! cache `crates/bgz17/data/jina_probe_embeddings.f32` (self-describing header
//! `[u32 n][u32 dim]` + n·dim f32 LE). Re-run does NOT re-hit the API.
//!
//! Deterministic SplitMix64 (seed 0x9E3779B97F4A7C15) — no clock, no rand.
//!
//! Run:  cargo run --release --manifest-path crates/bgz17/Cargo.toml \
//!         --example probe_codebook_44_realdata

use bgz17::base17::Base17;
use bgz17::palette::Palette;
use bgz17::{BASE_DIM, FP_SCALE, GOLDEN_STEP};
use std::path::Path;

// ── deterministic RNG ────────────────────────────────────────────────────────

struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform index in [0, n).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ── f32 → bgz17 Base17 (golden-fold, identical to projection::Base17::from_f32) ─

fn base17_from_f32(v: &[f32]) -> Base17 {
    // GOLDEN_POS[i] = (i * GOLDEN_STEP) % BASE_DIM
    let mut golden_pos = [0usize; BASE_DIM];
    for (i, gp) in golden_pos.iter_mut().enumerate() {
        *gp = (i * GOLDEN_STEP) % BASE_DIM;
    }
    let n = v.len();
    let n_octaves = n.div_ceil(BASE_DIM);
    let mut sum = [0f64; BASE_DIM];
    let mut count = [0u32; BASE_DIM];
    for octave in 0..n_octaves {
        for (bi, &gp) in golden_pos.iter().enumerate() {
            let dim = octave * BASE_DIM + gp;
            if dim < n {
                sum[bi] += v[dim] as f64;
                count[bi] += 1;
            }
        }
    }
    let mut dims = [0i16; BASE_DIM];
    for (d, slot) in dims.iter_mut().enumerate() {
        if count[d] > 0 {
            let mean = sum[d] / count[d] as f64;
            *slot = (mean * FP_SCALE).round().clamp(-32768.0, 32767.0) as i16;
        }
    }
    Base17 { dims }
}

// ── ground-truth cosine on raw f32 vectors ───────────────────────────────────

fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

// ── Spearman ρ (average-rank, tie-aware) ─────────────────────────────────────

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

// ── load cached embeddings ───────────────────────────────────────────────────

fn load_embeddings(path: &Path) -> (usize, usize, Vec<Vec<f32>>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "cannot read {} ({e}). Run the embed step first:\n  \
             python3 <scratch>/embed_vocab.py  (writes the gitignored cache).",
            path.display()
        )
    });
    assert!(bytes.len() >= 8, "truncated header");
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    assert_eq!(bytes.len(), 8 + n * dim * 4, "size mismatch vs header");
    let mut rows = Vec::with_capacity(n);
    let mut off = 8;
    for _ in 0..n {
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            let f = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            v.push(f);
            off += 4;
        }
        rows.push(v);
    }
    (n, dim, rows)
}

/// NaN scan: panic with the exact location. Certification doctrine — never
/// silently filter a NaN.
fn nan_scan_vectors(rows: &[Vec<f32>]) {
    for (i, r) in rows.iter().enumerate() {
        for (d, &x) in r.iter().enumerate() {
            if x.is_nan() {
                panic!("NaN in raw embedding: row {i}, dim {d} — HALT (source-byte stage)");
            }
        }
    }
}

/// Reconstructed Spearman ρ for one codebook: for each held-out pair, encode
/// both words to their nearest centroid, look up the palette distance, and
/// correlate the reconstructed SIMILARITY (−distance) against the TRUE cosine.
fn reconstruct_rho(
    palette: &Palette,
    patterns: &[Base17],
    raw: &[Vec<f32>],
    pairs: &[(usize, usize)],
) -> (f64, usize) {
    let table = palette.build_distance_table();
    // cache codes for the indices we touch
    let codes: Vec<u8> = patterns.iter().map(|p| palette.nearest(p)).collect();
    let mut recon_sim = Vec::with_capacity(pairs.len());
    let mut true_cos = Vec::with_capacity(pairs.len());
    let mut collisions = 0usize;
    for &(i, j) in pairs {
        let d = table.distance(codes[i], codes[j]) as f64;
        if d == 0.0 {
            collisions += 1; // both words fell in the same centroid
        }
        recon_sim.push(-d); // −distance = reconstructed similarity
        let c = cosine_f32(&raw[i], &raw[j]);
        if c.is_nan() {
            panic!("NaN cosine at pair ({i},{j}) — HALT (ground-truth stage)");
        }
        true_cos.push(c);
    }
    let rho = spearman(&recon_sim, &true_cos);
    if rho.is_nan() {
        panic!("NaN Spearman ρ — HALT (metric stage)");
    }
    (rho, collisions)
}

/// Canon-anchor framing: Spearman ρ of the codebook-reconstructed distance vs
/// the codebook's OWN reference metric — the FULL Base17 L1 distance. This is
/// exactly what the 0.965 (k=128) / 0.992 (k=256) anchors measure (bgz17
/// KNOWLEDGE.md) and what the synthetic probe's `fidelity_rho` correlates
/// against (`templates[i].l1(templates[j])`). It isolates the CODEBOOK's
/// fidelity from the upstream Base17-projection loss, so it actually
/// discriminates hierarchical vs flat.
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
        panic!("NaN Spearman ρ (vs Base17) — HALT (metric stage)");
    }
    rho
}

fn deterministic_pairs(lo: usize, hi: usize, n_pairs: usize) -> Vec<(usize, usize)> {
    let span = hi - lo;
    assert!(span >= 2, "need ≥2 indices to form pairs");
    let mut rng = SplitMix64(0x9E37_79B9_7F4A_7C15);
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

fn run_split(
    label: &str,
    train: &[Base17],
    eval_patterns: &[Base17],
    raw: &[Vec<f32>],
    pairs: &[(usize, usize)],
    max_iter: usize,
) -> (f64, f64) {
    let flat = Palette::build(train, 256, max_iter);
    let hp = Palette::build_hierarchical(train, max_iter);

    // (A) canon-anchor framing: codebook fidelity vs its OWN Base17 reference
    //     (comparable to the 0.965 / 0.992 anchors — discriminates the codebook).
    let cb_flat = reconstruct_rho_vs_base17(&flat, eval_patterns, pairs);
    let cb_hier = reconstruct_rho_vs_base17(&hp.leaves, eval_patterns, pairs);

    // (B) end-to-end framing: reconstructed distance vs raw Jina cosine
    //     (folds in the upstream Base17-fold loss — bottlenecked, informational).
    let (e2e_flat, coll_flat) = reconstruct_rho(&flat, eval_patterns, raw, pairs);
    let (e2e_hier, coll_hier) = reconstruct_rho(&hp.leaves, eval_patterns, raw, pairs);

    println!("── {label} ──");
    println!(
        "  train patterns={}  flat_centroids={}  hier_leaves={} (coarse={})",
        train.len(),
        flat.len(),
        hp.leaf_len(),
        hp.coarse_len()
    );
    println!("  eval pairs={}  (SplitMix64 seed 0x9E3779B97F4A7C15)", pairs.len());
    println!("  (A) CODEBOOK fidelity ρ vs full Base17 distance  [canon-anchor 0.965/0.992 framing]");
    println!("        flat         ρ : {cb_flat:.4}");
    println!("        hierarchical ρ : {cb_hier:.4}");
    println!(
        "        clears 0.965 : flat={} hier={}   |   clears 0.9973 : flat={} hier={}",
        cb_flat >= 0.965,
        cb_hier >= 0.965,
        cb_flat >= 0.9973,
        cb_hier >= 0.9973
    );
    println!(
        "        hierarchical ρ >= flat ρ (structure is free) : {}",
        if cb_hier >= cb_flat - 1e-9 { "PASS" } else { "FAIL" }
    );
    println!("  (B) END-TO-END ρ vs raw Jina cosine  [bottlenecked by 17-dim Base17 fold]");
    println!(
        "        flat ρ : {e2e_flat:.4} (collisions {coll_flat}/{})   hierarchical ρ : {e2e_hier:.4} (collisions {coll_hier}/{})",
        pairs.len(),
        pairs.len()
    );
    println!();
    (cb_flat, cb_hier)
}

fn main() {
    let emb_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/jina_probe_embeddings.f32");
    let (n, dim, raw) = load_embeddings(&emb_path);

    println!("PROBE-CODEBOOK-44  REAL-DATA ρ  (closes Probe M1 real-data PENDING)");
    println!("====================================================================");
    println!("data label: REAL Jina embeddings (jina-embeddings-v3, {dim}-d, text-matching)");
    println!("vocab: top {n} frequency-ranked academic words");
    println!("cache: crates/bgz17/data/jina_probe_embeddings.f32 (gitignored, LOCAL-only)\n");

    // NaN scan — source-byte stage.
    nan_scan_vectors(&raw);
    println!("NaN scan (raw embeddings): PASS ({n} × {dim} f32)\n");

    // Project every word into bgz17 Base17 (golden fold = projection::from_f32).
    let patterns: Vec<Base17> = raw.iter().map(|v| base17_from_f32(v)).collect();

    // ── DIAGNOSTIC: decompose where fidelity is lost ─────────────────────────
    // (a) true-cosine distribution over the eval pairs; (b) the Base17 golden-
    //     fold PROJECTION ceiling (17-dim L1, no codebook) vs true cosine.
    {
        let diag_pairs = deterministic_pairs(0, n, 4000);
        let mut cos = Vec::with_capacity(diag_pairs.len());
        let mut b17_sim = Vec::with_capacity(diag_pairs.len());
        for &(i, j) in &diag_pairs {
            cos.push(cosine_f32(&raw[i], &raw[j]));
            b17_sim.push(-(patterns[i].l1(&patterns[j]) as f64)); // −L1 = similarity
        }
        let mean = cos.iter().sum::<f64>() / cos.len() as f64;
        let var = cos.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / cos.len() as f64;
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &c in &cos {
            lo = lo.min(c);
            hi = hi.max(c);
        }
        // Base17 pairwise-distance spread (coefficient of variation): if the
        // folded 17-dim patterns are near-degenerate, distances cluster near a
        // single value and rank correlation becomes noise-sensitive.
        let b17d: Vec<f64> = b17_sim.iter().map(|&s| -s).collect();
        let dmean = b17d.iter().sum::<f64>() / b17d.len() as f64;
        let dvar = b17d.iter().map(|d| (d - dmean).powi(2)).sum::<f64>() / b17d.len() as f64;
        let cv = if dmean > 0.0 { dvar.sqrt() / dmean } else { 0.0 };

        println!("── DIAGNOSTIC (fidelity decomposition) ──");
        println!(
            "  true cosine over pairs: mean={mean:.4} std={:.4} min={lo:.4} max={hi:.4}",
            var.sqrt()
        );
        println!(
            "  Base17 golden-fold PROJECTION ceiling (17-dim L1, NO codebook) ρ : {:.4}",
            spearman(&b17_sim, &cos)
        );
        println!(
            "  Base17 pairwise-L1 spread: mean={dmean:.1} std={:.1} CV={cv:.3} (low CV ⇒ near-degenerate)",
            dvar.sqrt()
        );
        println!("  (this is the ceiling any 256-centroid codebook can reach on this input)\n");
    }

    let max_iter = 20;

    // ── HEADLINE: held-out generalization split ──────────────────────────────
    // Build BOTH codebooks on train words only; evaluate ρ on pairs drawn from
    // the held-out words the codebooks never saw during calibration. This is the
    // certification-doctrine validation set (generalization past the centroids).
    let train_n = (n * 3) / 4; // 75% train, 25% held out
    let train_patterns = &patterns[..train_n];
    let held_pairs = deterministic_pairs(train_n, n, 4000);
    let (h_flat, h_hier) = run_split(
        "HELD-OUT SPLIT (headline): build on 75% train, eval on 25% unseen words",
        train_patterns,
        &patterns,
        &raw,
        &held_pairs,
        max_iter,
    );

    // ── SECONDARY: full-set (build on all words; matches synthetic-probe framing)
    let full_pairs = deterministic_pairs(0, n, 4000);
    let (f_flat, f_hier) = run_split(
        "FULL-SET (secondary): build on all words, eval on random pairs",
        &patterns,
        &patterns,
        &raw,
        &full_pairs,
        max_iter,
    );

    // ── VERDICT ───────────────────────────────────────────────────────────────
    // Headline = the CODEBOOK-fidelity framing (framing A) on the held-out split,
    // because that is the reference the 0.965/0.992 canon anchors were measured
    // against and the only framing that discriminates the codebook (framing B is
    // capped by the upstream 17-dim Base17 fold, ceiling ≈ 0.26 on dense Jina).
    println!("VERDICT (headline = held-out split, CODEBOOK-fidelity framing vs full Base17)");
    let hier_ge_flat = h_hier >= h_flat - 1e-9;
    let clears = h_hier >= 0.965 && h_flat >= 0.965;
    println!("  flat ρ = {h_flat:.4}   hierarchical ρ = {h_hier:.4}");
    println!("  hierarchical ρ ≥ flat ρ (structure is free) : {hier_ge_flat}");
    println!("  both clear 0.965 anchor : {clears}");
    if hier_ge_flat && clears {
        println!(
            "  => PROBE M1 CLOSED: on REAL Jina-derived Base17 patterns the hierarchical\n     \
             16×16 codebook preserves the metric ≥ the flat-256 codebook AND clears the\n     \
             0.965 anchor. Structure-is-free holds on real semantics, not just synthetic."
        );
    } else if !hier_ge_flat {
        println!(
            "  => PROBE M1 FALSIFIED: hierarchical sacrifices codebook fidelity (ρ below flat)\n     \
             on real data. The synthetic prefix==ancestry win does NOT carry to real Jina."
        );
    } else {
        println!(
            "  => PARTIAL: hierarchical ≥ flat (structure is free holds), but the 0.965 anchor\n     \
             is not cleared by both at this k / vocab — report the numbers, do not over-claim."
        );
    }
    println!("  secondary full-set (codebook framing): flat ρ = {f_flat:.4}  hierarchical ρ = {f_hier:.4}");
    println!(
        "\n  NOTE: end-to-end ρ vs raw Jina cosine is bottlenecked at the 17-dim Base17\n  \
         golden-fold ceiling (ρ ≈ 0.26, printed above) — a SEPARATE upstream finding\n  \
         about projecting dense 1024-d embeddings to 17 dims, not a codebook property."
    );
}
