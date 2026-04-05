//! End-to-end test: real tokenizer → signed engine → nucleus sampling.
//!
//! Tests whether the full pipeline produces meaningful similarity:
//!   Similar texts (Rumi↔Rumi) should have higher overlap than
//!   unrelated texts (Rumi↔TCP).
//!
//! Uses: real XLM-RoBERTa tokenizer, Jina v3 HDR lens (converted to i8),
//! SignedThinkingEngine with Nucleus pooling (T=0.7, p=0.9).
//!
//! This is the SMOKE TEST before calibration. If this fails,
//! the 7-lane encoding and ONNX ICC are measuring noise.

use thinking_engine::jina_lens::{JINA_HDR_TABLE, jina_lookup_many, JINA_N_CENTROIDS};
use thinking_engine::signed_engine::SignedThinkingEngine;
use thinking_engine::pooling::Pooling;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  END-TO-END: real tokenizer → i8 signed → nucleus");
    println!("═══════════════════════════════════════════════════════════\n");

    // Load real XLM-RoBERTa tokenizer
    let tok = match tokenizers::Tokenizer::from_file(
        "crates/thinking-engine/data/jina-v3-hdr/tokenizer.json"
    ) {
        Ok(t) => t,
        Err(e) => { eprintln!("Tokenizer failed: {}. Aborting.", e); return; }
    };
    println!("Tokenizer: XLM-RoBERTa 250K loaded\n");

    // Build signed engine from Jina HDR table
    let signed_table: Vec<i8> = JINA_HDR_TABLE.iter()
        .map(|&v| (v as i16 - 128) as i8)
        .collect();
    // NOTE: This is from_unsigned (CDF rank relabeling, not true signed).
    // The real i8 path needs from_f32_cosines via stream_signed_lens.
    // But this tests the ENGINE + POOLING pipeline, not the encoding quality.
    let mut engine = SignedThinkingEngine::new(signed_table);

    let pooling = Pooling::Nucleus {
        temperature: 0.7,
        top_p: 0.9,
        seed: Some(42), // deterministic for comparison
    };

    // Calibration pairs (4 tiers)
    let pairs: Vec<(&str, &str, &str)> = vec![
        // TIER 1 — should be MOST similar
        ("The wound is the place where the light enters you",
         "Where there is ruin there is hope for a treasure",
         "Rumi↔Rumi"),
        ("A federal judge ruled the surveillance program unconstitutional",
         "A US court declared the mass surveillance scheme violated the constitution",
         "STS-B paraphrase"),
        // TIER 2 — moderate
        ("Palantir built Gotham for intelligence agencies to map human networks",
         "Edward Snowden revealed the NSA collected phone metadata of millions",
         "Palantir↔Snowden"),
        ("Amyloid plaques accumulate in the brains of Alzheimer patients",
         "Tau protein tangles disrupt neural communication in neurodegenerative disease",
         "Alzheimer↔Tau"),
        // TIER 3 — weak
        ("Newton showed that gravity follows an inverse square law",
         "Quantum entanglement allows particles to share states across arbitrary distances",
         "Newton↔Quantum"),
        // TIER 4 — should be LEAST similar
        ("You are not a drop in the ocean you are the entire ocean in a drop",
         "TCP uses a three-way handshake to establish a reliable connection between hosts",
         "Rumi↔TCP"),
        ("CRISPR-Cas9 enables precise editing of genomic sequences at targeted loci",
         "Bach composed the Well-Tempered Clavier as an exploration of all major and minor keys",
         "CRISPR↔Bach"),
    ];

    println!("  {:>20}  {:>8}  {:>8}  {:>8}  {:>6}  {:>6}",
        "Pair", "Jaccard", "Cos(E)", "TopK∩", "Inhib", "Cycles");
    println!("  {:─>20}  {:─>8}  {:─>8}  {:─>8}  {:─>6}  {:─>6}", "", "", "", "", "", "");

    let mut results: Vec<(String, f32, f32, usize)> = Vec::new();

    for (text_a, text_b, label) in &pairs {
        let enc_a = tok.encode(*text_a, true).unwrap();
        let enc_b = tok.encode(*text_b, true).unwrap();
        let ids_a: Vec<u32> = enc_a.get_ids().to_vec();
        let ids_b: Vec<u32> = enc_b.get_ids().to_vec();

        let centroids_a = jina_lookup_many(&ids_a);
        let centroids_b = jina_lookup_many(&ids_b);

        // Think text A
        engine.reset();
        engine.perturb(&centroids_a);
        engine.think(10);
        let energy_a = engine.energy.clone();
        let pooled_a = pooling.pool(&energy_a);
        let inhib_a = engine.total_inhibitions;

        // Think text B
        engine.reset();
        engine.perturb(&centroids_b);
        engine.think(10);
        let energy_b = engine.energy.clone();
        let pooled_b = pooling.pool(&energy_b);
        let inhib_b = engine.total_inhibitions;

        // Compare: Jaccard of pooled atoms
        let atoms_a: std::collections::HashSet<u16> = pooled_a.atoms.iter()
            .map(|&(idx, _)| idx).collect();
        let atoms_b: std::collections::HashSet<u16> = pooled_b.atoms.iter()
            .map(|&(idx, _)| idx).collect();
        let intersection = atoms_a.intersection(&atoms_b).count();
        let union = atoms_a.union(&atoms_b).count().max(1);
        let jaccard = intersection as f32 / union as f32;

        // Compare: cosine of full energy vectors
        let dot: f32 = energy_a.iter().zip(&energy_b).map(|(a, b)| a * b).sum();
        let na: f32 = energy_a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = energy_b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_e = if na > 1e-10 && nb > 1e-10 { dot / (na * nb) } else { 0.0 };

        // Compare: top-k overlap
        let top_a: Vec<u16> = pooled_a.atoms.iter().take(5).map(|&(idx, _)| idx).collect();
        let top_b: Vec<u16> = pooled_b.atoms.iter().take(5).map(|&(idx, _)| idx).collect();
        let topk_overlap = top_a.iter().filter(|x| top_b.contains(x)).count();

        println!("  {:>20}  {:>8.3}  {:>8.3}  {:>5}/5  {:>6}  {:>3}+{:<3}",
            label, jaccard, cos_e, topk_overlap,
            (inhib_a + inhib_b) / 2,
            pooled_a.atoms.len(), pooled_b.atoms.len());

        results.push((label.to_string(), jaccard, cos_e, topk_overlap));
    }

    // Verdict
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  VERDICT");
    println!("═══════════════════════════════════════════════════════════");

    // Check monotonicity: tier 1 > tier 2 > tier 3 > tier 4
    let tier1_avg = (results[0].2 + results[1].2) / 2.0;
    let tier2_avg = (results[2].2 + results[3].2) / 2.0;
    let tier3_avg = results[4].2;
    let tier4_avg = (results[5].2 + results[6].2) / 2.0;

    println!("  Tier 1 (paraphrase):  cos={:.3}", tier1_avg);
    println!("  Tier 2 (thematic):    cos={:.3}", tier2_avg);
    println!("  Tier 3 (weak):        cos={:.3}", tier3_avg);
    println!("  Tier 4 (unrelated):   cos={:.3}", tier4_avg);
    println!();

    let monotonic = tier1_avg >= tier2_avg && tier2_avg >= tier3_avg && tier3_avg >= tier4_avg;
    if monotonic {
        println!("  → MONOTONIC: tiers decrease correctly. Engine discriminates.");
        println!("  → Ready for 7-lane encoding + ONNX ICC calibration.");
    } else if tier1_avg > tier4_avg {
        println!("  → PARTIALLY DISCRIMINATIVE: tier1 > tier4 but not monotonic.");
        println!("  → Engine sees some signal. May improve with better encoding.");
    } else {
        println!("  → NOT DISCRIMINATIVE: tier1 ≤ tier4. Engine is confused.");
        println!("  → Fix encoding or table granularity before calibration.");
    }
}
