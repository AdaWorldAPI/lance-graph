//! Calibrate baked lenses against ground truth using REAL tokenizers.
//!
//! Uses HuggingFace tokenizers crate for proper BPE tokenization:
//!   - Jina v3: XLM-RoBERTa tokenizer (250K vocab) via from_pretrained
//!   - Reranker v3: Qwen2 tokenizer (151K vocab) from local file
//!
//! Calibration corpus: Rumi, Tagore, Wittgenstein, STS-B style pairs,
//! OSINT-relevant text, technical paraphrases. 4 similarity tiers.
//!
//! Usage:
//!   cargo run --release --features tokenizer --example calibrate_lenses
//!
//! For local-only (no HuggingFace download):
//!   Place tokenizer.json files in data/ directories and set TOKENIZER_LOCAL=1

use thinking_engine::jina_lens;
use thinking_engine::reranker_lens;

fn main() {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  Lens Calibration — Real Tokenizers + Baked Lenses");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    // ── Load tokenizers ────────────────────────────────────────────
    let jina_tok = load_jina_tokenizer();
    let reranker_tok = load_reranker_tokenizer();

    let jina_ok = jina_tok.is_some();
    let rr_ok = reranker_tok.is_some();

    eprintln!("  Jina v3 tokenizer:   {}", if jina_ok { "LOADED (XLM-RoBERTa 250K)" } else { "FALLBACK (hash)" });
    eprintln!("  Reranker tokenizer:  {}", if rr_ok { "LOADED (Qwen2 151K)" } else { "FALLBACK (hash)" });
    eprintln!();

    // ── Calibration corpus: 4 tiers × 4 pairs = 16 pairs ──────────
    //
    // TIER 1 — Near-identical meaning (expected cos > 0.85)
    // TIER 2 — Metaphorical / thematic similarity (expected cos 0.5-0.8)
    // TIER 3 — Domain-related but different claims (expected cos 0.2-0.5)
    // TIER 4 — Unrelated domains (expected cos < 0.15)
    let pairs = vec![
        // ── TIER 1: paraphrase / near-identical ──
        ("The wound is the place where the light enters you", "Where there is ruin there is hope for a treasure"),
        ("The flower which is single need not envy the thorns that are numerous", "Let your life lightly dance on the edges of time like dew on the tip of a leaf"),
        ("A federal judge in New York ruled the surveillance program unconstitutional", "A US court declared the mass surveillance scheme violated the constitution"),
        ("Gradient descent minimizes the loss function by following the negative gradient", "The optimizer reduces error by stepping in the direction of steepest descent"),

        // ── TIER 2: metaphorical / thematic resonance ──
        ("Out beyond ideas of wrongdoing and rightdoing there is a field I will meet you there", "Love is not a mere impulse it must contain truth which is law"),
        ("Whereof one cannot speak thereof one must be silent", "Any consistent formal system strong enough to encode arithmetic is incomplete"),
        ("Palantir built Gotham for intelligence agencies to map human networks", "Edward Snowden revealed the NSA collected phone metadata of millions of Americans"),
        ("Amyloid plaques accumulate in the brains of Alzheimer patients", "Tau protein tangles disrupt neural communication in neurodegenerative disease"),

        // ── TIER 3: loosely related domain ──
        ("Consciousness arises from integrated information across cortical networks", "The hard problem asks why physical processes give rise to subjective experience"),
        ("The Vienna Convention codifies diplomatic immunity for foreign ambassadors", "Maritime law governs salvage rights for vessels in international waters"),
        ("Newton showed that gravity follows an inverse square law", "Quantum entanglement allows particles to share states across arbitrary distances"),
        ("The butterfly counts not months but moments and has time enough", "Monarch butterfly populations declined forty percent due to habitat fragmentation"),

        // ── TIER 4: unrelated domains ──
        ("You are not a drop in the ocean you are the entire ocean in a drop", "TCP uses a three-way handshake to establish a reliable connection between hosts"),
        ("Where the mind is without fear and the head is held high", "The Federal Reserve raised interest rates by twenty-five basis points in March"),
        ("CRISPR-Cas9 enables precise editing of genomic sequences at targeted loci", "Bach composed the Well-Tempered Clavier as an exploration of all major and minor keys"),
        ("The International Criminal Court prosecutes genocide and crimes against humanity", "Fermentation converts sugars into alcohol and carbon dioxide through anaerobic respiration"),
    ];

    eprintln!("Calibration corpus: {} pairs (4 tiers)\n", pairs.len());

    // ── Tokenize + baked lens distances ────────────────────────────
    eprintln!("=== Baked Lens Distances (real tokenization) ===\n");

    let mut jina_dists = Vec::new();
    let mut reranker_dists = Vec::new();

    for (i, (a, b)) in pairs.iter().enumerate() {
        // Jina v3: XLM-RoBERTa tokenizer → 250K vocab → codebook lookup
        let (a_ids_jina, b_ids_jina) = tokenize_pair(
            a, b, jina_tok.as_ref(), 250_002,
        );
        // Reranker: Qwen2 tokenizer → 151K vocab → codebook lookup
        let (a_ids_rr, b_ids_rr) = tokenize_pair(
            a, b, reranker_tok.as_ref(), 151_936,
        );

        // Jina: average pairwise centroid distance
        let a_centroids = jina_lens::jina_lookup_many(&a_ids_jina);
        let b_centroids = jina_lens::jina_lookup_many(&b_ids_jina);
        let jina_sim = avg_distance(&a_centroids, &b_centroids, |a, b|
            jina_lens::jina_distance(a, b) as f32 / 255.0);
        jina_dists.push(jina_sim);

        // Reranker: relevance score
        let rr_rel = reranker_lens::reranker_relevance(&a_ids_rr, &b_ids_rr);
        reranker_dists.push(rr_rel);

        let tok_type = if jina_ok { "BPE" } else { "hash" };
        eprintln!("  [{:2}] jina={:.3} rr={:.3} [{}] | \"{}...\" ↔ \"{}...\"",
            i, jina_sim, rr_rel, tok_type,
            &a[..a.len().min(40)], &b[..b.len().min(40)]);
    }

    // ── Ground truth (expert-assigned, tiered) ─────────────────────
    eprintln!("\n=== Ground Truth (expert-assigned, tiered) ===\n");

    let api_ground_truth: Vec<f32> = vec![
        // TIER 1 — paraphrase / near-identical
        0.88, 0.72, 0.93, 0.95,
        // TIER 2 — metaphorical / thematic resonance
        0.58, 0.52, 0.68, 0.72,
        // TIER 3 — loosely related domain
        0.42, 0.25, 0.18, 0.30,
        // TIER 4 — unrelated domains
        0.04, 0.03, 0.05, 0.06,
    ];

    eprintln!("  NOTE: Ground truth is expert-assigned.");
    eprintln!("  Replace with Jina v5 ONNX (rten) for machine ground truth.\n");

    // ── Spearman rank correlation ────────────────────────────────
    eprintln!("=== Spearman Rank Correlation ===\n");

    let rho_jina = spearman(&jina_dists, &api_ground_truth);
    let rho_reranker = spearman(&reranker_dists, &api_ground_truth);
    let rho_cross = spearman(&jina_dists, &reranker_dists);

    eprintln!("  Jina v3 baked vs truth:    ρ = {:.4}", rho_jina);
    eprintln!("  Reranker baked vs truth:   ρ = {:.4}", rho_reranker);
    eprintln!("  Jina vs Reranker (cross):  ρ = {:.4}", rho_cross);

    eprintln!("\n  Thresholds:");
    eprintln!("    ρ > 0.998: truth-anchor grade");
    eprintln!("    ρ > 0.95:  usable with ICC correction");
    eprintln!("    ρ > 0.80:  weak (needs more centroids or γ+φ)");
    eprintln!("    ρ < 0.80:  broken");

    for (name, rho) in [("Jina", rho_jina), ("Reranker", rho_reranker)] {
        let status = if rho > 0.998 { "TRUTH ANCHOR" }
            else if rho > 0.95 { "USABLE (needs ICC)" }
            else if rho > 0.80 { "WEAK" }
            else { "BROKEN" };
        eprintln!("  {} → {}", name, status);
    }

    // ── ICC Profile ──────────────────────────────────────────────
    if rho_jina < 0.998 || rho_reranker < 0.998 {
        eprintln!("\n=== ICC Profile ===\n");

        let (jina_slope, jina_intercept) = linear_fit(&jina_dists, &api_ground_truth);
        let (rr_slope, rr_intercept) = linear_fit(&reranker_dists, &api_ground_truth);

        eprintln!("  Jina ICC:     corrected = {:.3} × baked + {:.3}", jina_slope, jina_intercept);
        eprintln!("  Reranker ICC: corrected = {:.3} × baked + {:.3}", rr_slope, rr_intercept);

        let jina_corrected: Vec<f32> = jina_dists.iter()
            .map(|&d| (d * jina_slope + jina_intercept).clamp(0.0, 1.0)).collect();
        let rr_corrected: Vec<f32> = reranker_dists.iter()
            .map(|&d| (d * rr_slope + rr_intercept).clamp(0.0, 1.0)).collect();

        let rho_jina_c = spearman(&jina_corrected, &api_ground_truth);
        let rho_rr_c = spearman(&rr_corrected, &api_ground_truth);

        eprintln!("  After ICC: Jina ρ = {:.4} (was {:.4})", rho_jina_c, rho_jina);
        eprintln!("  After ICC: Reranker ρ = {:.4} (was {:.4})", rho_rr_c, rho_reranker);
    }

    eprintln!("\n═══════════════════════════════════════════════════════════");
    if jina_ok && rr_ok {
        eprintln!("  Real BPE tokenization used for both lenses.");
    } else {
        eprintln!("  Hash fallback used. For real BPE:");
        eprintln!("    cargo run --features tokenizer --example calibrate_lenses");
    }
    eprintln!("═══════════════════════════════════════════════════════════\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// Tokenizer loading
// ═══════════════════════════════════════════════════════════════════════════

/// Load Jina v3 tokenizer (XLM-RoBERTa, 250K vocab).
/// Tries from_pretrained first, falls back to local file, then None.
fn load_jina_tokenizer() -> Option<tokenizers::Tokenizer> {
    // Try local file first (no network)
    let local_path = "crates/thinking-engine/data/jina-v3-hdr/tokenizer.json";
    if let Ok(tok) = tokenizers::Tokenizer::from_file(local_path) {
        eprintln!("  [jina] Loaded from {}", local_path);
        return Some(tok);
    }

    // Try from_pretrained (downloads from HuggingFace)
    eprintln!("  [jina] Downloading tokenizer from jinaai/jina-embeddings-v3...");
    match tokenizers::Tokenizer::from_pretrained("jinaai/jina-embeddings-v3", None) {
        Ok(tok) => {
            eprintln!("  [jina] Downloaded successfully.");
            // Save for next time
            let _ = tok.save(local_path, false);
            Some(tok)
        }
        Err(e) => {
            eprintln!("  [jina] Failed to download: {}. Using hash fallback.", e);
            None
        }
    }
}

/// Load Reranker tokenizer (Qwen2, 151K vocab).
/// Uses the Qwopus tokenizer.json on disk (same Qwen2 BPE).
fn load_reranker_tokenizer() -> Option<tokenizers::Tokenizer> {
    // Qwopus uses same Qwen2 tokenizer as Reranker
    let local_path = "crates/thinking-engine/data/Qwopus3.5-27B-v3-BF16-silu/tokenizer.json";
    if let Ok(tok) = tokenizers::Tokenizer::from_file(local_path) {
        eprintln!("  [reranker] Loaded from {}", local_path);
        return Some(tok);
    }

    // Try from_pretrained
    eprintln!("  [reranker] Downloading tokenizer from jinaai/jina-reranker-v2-base-multilingual...");
    match tokenizers::Tokenizer::from_pretrained("jinaai/jina-reranker-v2-base-multilingual", None) {
        Ok(tok) => {
            eprintln!("  [reranker] Downloaded successfully.");
            Some(tok)
        }
        Err(e) => {
            eprintln!("  [reranker] Failed to download: {}. Using hash fallback.", e);
            None
        }
    }
}

/// Tokenize a pair of texts. Uses real tokenizer if available, hash fallback otherwise.
fn tokenize_pair(
    a: &str,
    b: &str,
    tokenizer: Option<&tokenizers::Tokenizer>,
    vocab_size: u32,
) -> (Vec<u32>, Vec<u32>) {
    if let Some(tok) = tokenizer {
        let enc_a = tok.encode(a, true).expect("tokenize failed");
        let enc_b = tok.encode(b, true).expect("tokenize failed");
        let ids_a: Vec<u32> = enc_a.get_ids().iter()
            .map(|&id| id.min(vocab_size - 1)).collect();
        let ids_b: Vec<u32> = enc_b.get_ids().iter()
            .map(|&id| id.min(vocab_size - 1)).collect();
        (ids_a, ids_b)
    } else {
        // Hash fallback (last resort, gives garbage ρ)
        let ids_a: Vec<u32> = a.split_whitespace()
            .map(|w| simple_hash(w) % vocab_size).collect();
        let ids_b: Vec<u32> = b.split_whitespace()
            .map(|w| simple_hash(w) % vocab_size).collect();
        (ids_a, ids_b)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Distance and statistics
// ═══════════════════════════════════════════════════════════════════════════

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
