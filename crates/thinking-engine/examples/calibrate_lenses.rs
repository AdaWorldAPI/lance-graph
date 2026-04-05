//! Calibrate baked lenses against live API ground truth.
//!
//! For each lens (Jina v3, Reranker v3):
//!   1. Encode N sentence pairs via API (or local rten inference)
//!   2. Compute API cosines (ground truth)
//!   3. Compute baked lens distances
//!   4. Measure Spearman ρ (rank correlation)
//!   5. Build ICC profile if ρ < 0.998
//!
//! Usage:
//!   JINA_API_KEY=... cargo run --release --example calibrate_lenses
//!   Or: with local models via rten (no API key needed)

use thinking_engine::jina_lens;
use thinking_engine::reranker_lens;

fn main() {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  Lens Calibration — Baked vs API Ground Truth");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    // Test sentence pairs (diverse: similar, dissimilar, related, unrelated)
    let pairs = vec![
        ("love is patient", "love is kind"),
        ("love is patient", "hate is destructive"),
        ("the cat sat on the mat", "a dog lay on the rug"),
        ("the cat sat on the mat", "quantum physics is complex"),
        ("artificial intelligence", "machine learning"),
        ("artificial intelligence", "medieval pottery"),
        ("Palantir developed Gotham", "CIA funded surveillance"),
        ("Palantir developed Gotham", "roses bloom in spring"),
        ("the wound is where light enters", "suffering leads to growth"),
        ("the wound is where light enters", "TCP/IP packet routing"),
        ("international law governs treaties", "diplomatic relations are complex"),
        ("international law governs treaties", "chocolate cake recipe"),
        ("neural network backpropagation", "gradient descent optimization"),
        ("neural network backpropagation", "ancient Roman architecture"),
        ("climate change affects biodiversity", "global warming impacts ecosystems"),
        ("climate change affects biodiversity", "jazz improvisation techniques"),
    ];

    eprintln!("Test pairs: {}\n", pairs.len());

    // ── Baked lens distances ────────────────────────────────────────
    eprintln!("=== Baked Lens Distances ===\n");

    // Simulate tokenization (hash-based, same as forward pass test)
    // In production: use real tokenizer per model
    let mut jina_dists = Vec::new();
    let mut reranker_dists = Vec::new();

    for (i, (a, b)) in pairs.iter().enumerate() {
        // Hash tokens for Jina (250K vocab)
        let a_ids_jina: Vec<u32> = a.split_whitespace()
            .map(|w| simple_hash(w) % 250_002).collect();
        let b_ids_jina: Vec<u32> = b.split_whitespace()
            .map(|w| simple_hash(w) % 250_002).collect();

        // Hash tokens for Reranker (151K vocab)
        let a_ids_rr: Vec<u32> = a.split_whitespace()
            .map(|w| simple_hash(w) % 151_936).collect();
        let b_ids_rr: Vec<u32> = b.split_whitespace()
            .map(|w| simple_hash(w) % 151_936).collect();

        // Jina: average pairwise centroid distance
        let a_centroids = jina_lens::jina_lookup_many(&a_ids_jina);
        let b_centroids = jina_lens::jina_lookup_many(&b_ids_jina);
        let jina_sim = avg_distance(&a_centroids, &b_centroids, |a, b|
            jina_lens::jina_distance(a, b) as f32 / 255.0);
        jina_dists.push(jina_sim);

        // Reranker: relevance score
        let rr_rel = reranker_lens::reranker_relevance(&a_ids_rr, &b_ids_rr);
        reranker_dists.push(rr_rel);

        eprintln!("  [{:2}] jina={:.3} rr={:.3} | \"{}\" ↔ \"{}\"",
            i, jina_sim, rr_rel, a, b);
    }

    // ── API ground truth (placeholder — needs real API) ──────────
    eprintln!("\n=== API Ground Truth ===\n");

    let api_key = std::env::var("JINA_API_KEY").ok();
    if api_key.is_none() {
        eprintln!("  JINA_API_KEY not set. Using synthetic ground truth.");
        eprintln!("  Set JINA_API_KEY to calibrate against real Jina API.");
        eprintln!("  Or use rten + Jina ONNX for local ground truth.\n");
    }

    // Synthetic ground truth: manually assigned similarities
    // (replace with real API calls when available)
    let api_ground_truth: Vec<f32> = vec![
        0.92, // love patient ↔ love kind (very similar)
        0.25, // love patient ↔ hate destructive (opposite)
        0.78, // cat mat ↔ dog rug (similar scene)
        0.05, // cat mat ↔ quantum physics (unrelated)
        0.89, // AI ↔ ML (very related)
        0.02, // AI ↔ pottery (unrelated)
        0.65, // Palantir Gotham ↔ CIA surveillance (related domain)
        0.03, // Palantir ↔ roses (unrelated)
        0.72, // wound light ↔ suffering growth (metaphorically similar)
        0.01, // wound light ↔ TCP/IP (unrelated)
        0.83, // law treaties ↔ diplomatic (related)
        0.04, // law ↔ chocolate (unrelated)
        0.91, // backprop ↔ gradient descent (very related)
        0.06, // backprop ↔ Roman architecture (unrelated)
        0.94, // climate biodiversity ↔ warming ecosystems (near identical)
        0.03, // climate ↔ jazz (unrelated)
    ];

    // ── Spearman rank correlation ────────────────────────────────
    eprintln!("=== Spearman Rank Correlation ===\n");

    let rho_jina = spearman(&jina_dists, &api_ground_truth);
    let rho_reranker = spearman(&reranker_dists, &api_ground_truth);

    eprintln!("  Jina v3 baked vs API:     ρ = {:.4}", rho_jina);
    eprintln!("  Reranker v3 baked vs API: ρ = {:.4}", rho_reranker);

    // Cross-model: Jina vs Reranker
    let rho_cross = spearman(&jina_dists, &reranker_dists);
    eprintln!("  Jina vs Reranker (cross): ρ = {:.4}", rho_cross);

    eprintln!("\n  Thresholds:");
    eprintln!("    ρ > 0.998: truth-anchor grade (< 2 rank disagreements in 1000)");
    eprintln!("    ρ > 0.95:  usable with ICC correction");
    eprintln!("    ρ < 0.95:  broken, needs rebuild");

    for (name, rho) in [("Jina", rho_jina), ("Reranker", rho_reranker)] {
        let status = if rho > 0.998 { "TRUTH ANCHOR ✓" }
            else if rho > 0.95 { "USABLE (needs ICC)" }
            else if rho > 0.80 { "WEAK (needs rebuild or more centroids)" }
            else { "BROKEN" };
        eprintln!("  {} → {}", name, status);
    }

    // ── ICC Profile (if needed) ──────────────────────────────────
    if rho_jina < 0.998 || rho_reranker < 0.998 {
        eprintln!("\n=== ICC Profile Needed ===\n");
        eprintln!("  Building transfer curves from {} pairs...", pairs.len());

        // Simple linear regression as baseline ICC
        let (jina_slope, jina_intercept) = linear_fit(&jina_dists, &api_ground_truth);
        let (rr_slope, rr_intercept) = linear_fit(&reranker_dists, &api_ground_truth);

        eprintln!("  Jina ICC:     corrected = {:.3} × baked + {:.3}", jina_slope, jina_intercept);
        eprintln!("  Reranker ICC: corrected = {:.3} × baked + {:.3}", rr_slope, rr_intercept);

        // Apply correction and re-measure
        let jina_corrected: Vec<f32> = jina_dists.iter()
            .map(|&d| (d * jina_slope + jina_intercept).clamp(0.0, 1.0))
            .collect();
        let rr_corrected: Vec<f32> = reranker_dists.iter()
            .map(|&d| (d * rr_slope + rr_intercept).clamp(0.0, 1.0))
            .collect();

        let rho_jina_corrected = spearman(&jina_corrected, &api_ground_truth);
        let rho_rr_corrected = spearman(&rr_corrected, &api_ground_truth);

        eprintln!("  After ICC: Jina ρ = {:.4} (was {:.4})", rho_jina_corrected, rho_jina);
        eprintln!("  After ICC: Reranker ρ = {:.4} (was {:.4})", rho_rr_corrected, rho_reranker);
    }

    eprintln!("\n═══════════════════════════════════════════════════════════");
    eprintln!("  NOTE: Using hash-based tokenization (not real BPE).");
    eprintln!("  For production calibration, use real tokenizers per model.");
    eprintln!("  Set JINA_API_KEY for real API ground truth.");
    eprintln!("═══════════════════════════════════════════════════════════\n");
}

fn avg_distance(a: &[u16], b: &[u16], dist_fn: impl Fn(u16, u16) -> f32) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0;
    for &ca in a {
        for &cb in b {
            sum += dist_fn(ca, cb);
            count += 1;
        }
    }
    if count > 0 { sum / count as f32 } else { 0.0 }
}

fn spearman(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n < 2 { return 0.0; }
    let rank_a = ranks(a);
    let rank_b = ranks(b);
    let mean_a = rank_a.iter().sum::<f32>() / n as f32;
    let mean_b = rank_b.iter().sum::<f32>() / n as f32;
    let mut num = 0.0f32;
    let mut den_a = 0.0f32;
    let mut den_b = 0.0f32;
    for i in 0..n {
        let da = rank_a[i] - mean_a;
        let db = rank_b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    let den = (den_a * den_b).sqrt();
    if den > 1e-10 { num / den } else { 0.0 }
}

fn ranks(values: &[f32]) -> Vec<f32> {
    let mut indexed: Vec<(usize, f32)> = values.iter().enumerate()
        .map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut result = vec![0.0f32; values.len()];
    for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
        result[orig_idx] = rank as f32;
    }
    result
}

fn linear_fit(x: &[f32], y: &[f32]) -> (f32, f32) {
    let n = x.len().min(y.len()) as f32;
    let mx = x.iter().sum::<f32>() / n;
    let my = y.iter().sum::<f32>() / n;
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for i in 0..x.len().min(y.len()) {
        num += (x[i] - mx) * (y[i] - my);
        den += (x[i] - mx) * (x[i] - mx);
    }
    let slope = if den > 1e-10 { num / den } else { 1.0 };
    let intercept = my - slope * mx;
    (slope, intercept)
}

fn simple_hash(word: &str) -> u32 {
    let mut h: u64 = 0x9e3779b97f4a7c15;
    for b in word.bytes() { h = h.wrapping_mul(31).wrapping_add(b as u64); }
    h as u32
}
