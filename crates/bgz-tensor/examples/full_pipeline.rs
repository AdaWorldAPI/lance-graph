//! Full pipeline comparison: real Jina embeddings through all encoding resolutions,
//! codebook sizes, HDR popcount cascade, and leaf hydration.
//!
//! cargo run --manifest-path crates/bgz-tensor/Cargo.toml --example full_pipeline

fn main() {
    // ─── Load real Jina embeddings ──────────────────────────────────────────
    let texts1 = vec![
        "The cat sat on the mat.", "A cat was sitting on the mat.",
        "Machine learning is a subset of artificial intelligence.", "ML is a branch of AI.",
        "The stock market crashed today.", "Today the stock market experienced a crash.",
        "Python is a programming language.", "JavaScript is used for web development.",
        "The sun rises in the east.", "Sunset paints the western sky.",
        "He ran quickly to the store.", "She walked slowly to the market.",
        "Quantum mechanics describes subatomic particles.", "The recipe calls for two cups of flour.",
        "Mount Everest is the tallest mountain.", "Abstract art challenges traditional aesthetics.",
        "Hello, how are you?", "Hallo, wie geht es dir?",
        "Good morning, world.", "Guten Morgen, Welt.",
    ];
    let texts2 = vec![
        "The Eiffel Tower is in Paris.", "La Tour Eiffel se trouve a Paris.",
        "Water boils at 100 degrees Celsius.", "The boiling point of water is 100C.",
        "Dogs are loyal companions.", "Cats are independent creatures.",
        "The speed of light is approximately 300000 km per second.",
        "Photons travel at about 300000 kilometers every second.",
        "Shakespeare wrote Romeo and Juliet.", "The famous love tragedy was written by Shakespeare.",
        "Bitcoin is a cryptocurrency.", "Ethereum uses smart contracts for decentralized applications.",
        "Neural networks learn from data.", "Deep learning models require large datasets for training.",
        "The Pacific Ocean is the largest ocean.", "Antarctica is the coldest continent on Earth.",
        "Bach composed the Brandenburg Concertos.", "Beethoven wrote nine symphonies.",
        "Rust is a systems programming language.", "Go is designed for concurrent programming.",
    ];

    let mut all_texts: Vec<&str> = Vec::new();
    all_texts.extend(texts1.iter().copied());
    all_texts.extend(texts2.iter().copied());

    let mut raw_vectors: Vec<Vec<f32>> = Vec::new();

    for (path, texts) in &[("/tmp/jina_batch1.json", &texts1), ("/tmp/jina_batch2.json", &texts2)] {
        match std::fs::read_to_string(path) {
            Ok(json) => {
                let text_refs: Vec<&str> = texts.iter().copied().collect();
                match bgz_tensor::jina::parse_jina_response(&json, &text_refs) {
                    Ok(embs) => {
                        println!("Loaded {} embeddings from {}", embs.len(), path);
                        for e in embs { raw_vectors.push(e.vector); }
                    }
                    Err(e) => { eprintln!("Parse error {}: {}", path, e); return; }
                }
            }
            Err(e) => { eprintln!("Cannot read {}: {}", path, e); return; }
        }
    }

    let n = raw_vectors.len();
    let dim = raw_vectors[0].len();
    println!("\n{} vectors, {} dimensions\n", n, dim);

    // ─── Ground truth: f32 cosine pairs ─────────────────────────────────────
    let n_pairs = n * (n - 1) / 2;
    let mut gt_pairs: Vec<(usize, usize, f64)> = Vec::with_capacity(n_pairs);
    for i in 0..n {
        for j in (i + 1)..n {
            gt_pairs.push((i, j, bgz_tensor::stacked_n::cosine_f32_slice(&raw_vectors[i], &raw_vectors[j])));
        }
    }
    let gt_cosines: Vec<f64> = gt_pairs.iter().map(|p| p.2).collect();

    println!("Ground truth: {} pairs", gt_pairs.len());
    println!("  cosine range: [{:.4}, {:.4}]",
        gt_cosines.iter().cloned().fold(f64::INFINITY, f64::min),
        gt_cosines.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    println!("  mean cosine:  {:.4}\n", gt_cosines.iter().sum::<f64>() / gt_cosines.len() as f64);

    // ═══════════════════════════════════════════════════════════════════════
    // PART 1: Encoding resolution comparison
    // ═══════════════════════════════════════════════════════════════════════
    println!("======================================================================");
    println!("PART 1: Encoding resolution (samples per dim)");
    println!("======================================================================\n");

    let sample_counts = [4, 8, 16, 32, 64];
    println!("┌──────┬────────┬──────────────────┬──────────────────┬──────────────────┐");
    println!("│  SPD │ Bytes  │ Stacked cosine   │ Popcount sign    │ Leaf-hydrated    │");
    println!("│      │        │ Pearson/Spearman │ Pearson/Spearman │ Pearson/Spearman │");
    println!("├──────┼────────┼──────────────────┼──────────────────┼──────────────────┤");

    for &spd in &sample_counts {
        let encoded: Vec<bgz_tensor::StackedN> = raw_vectors.iter()
            .map(|v| bgz_tensor::StackedN::from_f32(v, spd))
            .collect();

        let stacked_cos: Vec<f64> = gt_pairs.iter()
            .map(|&(i, j, _)| encoded[i].cosine(&encoded[j]))
            .collect();

        let sign_agree: Vec<f64> = gt_pairs.iter()
            .map(|&(i, j, _)| encoded[i].sign_agreement(&encoded[j]))
            .collect();

        // Leaf hydration: BF16→f32
        let hydrated: Vec<Vec<f32>> = encoded.iter().map(|e| e.hydrate_f32()).collect();
        let leaf_cos: Vec<f64> = gt_pairs.iter()
            .map(|&(i, j, _)| bgz_tensor::stacked_n::cosine_f32_slice(&hydrated[i], &hydrated[j]))
            .collect();

        let bytes = 17 * spd * 2;
        println!("│ {:>4} │ {:>6} │ {:>7.4}/{:>7.4}  │ {:>7.4}/{:>7.4}  │ {:>7.4}/{:>7.4}  │",
            spd, bytes,
            bgz_tensor::quality::pearson(&gt_cosines, &stacked_cos),
            bgz_tensor::quality::spearman(&gt_cosines, &stacked_cos),
            bgz_tensor::quality::pearson(&gt_cosines, &sign_agree),
            bgz_tensor::quality::spearman(&gt_cosines, &sign_agree),
            bgz_tensor::quality::pearson(&gt_cosines, &leaf_cos),
            bgz_tensor::quality::spearman(&gt_cosines, &leaf_cos),
        );
    }

    // Also compare Base17 i16
    let base17: Vec<bgz_tensor::Base17> = raw_vectors.iter()
        .map(|v| bgz_tensor::Base17::from_f32(v))
        .collect();
    let b17_cos: Vec<f64> = gt_pairs.iter()
        .map(|&(i, j, _)| base17[i].cosine(&base17[j]))
        .collect();
    let b17_l1: Vec<f64> = gt_pairs.iter()
        .map(|&(i, j, _)| -(base17[i].l1(&base17[j]) as f64))
        .collect();

    println!("├──────┼────────┼──────────────────┼──────────────────┼──────────────────┤");
    println!("│ i16  │     34 │ {:>7.4}/{:>7.4}  │      n/a/n/a     │      n/a/n/a     │",
        bgz_tensor::quality::pearson(&gt_cosines, &b17_cos),
        bgz_tensor::quality::spearman(&gt_cosines, &b17_cos));
    println!("│i16L1 │     34 │ {:>7.4}/{:>7.4}  │      n/a/n/a     │      n/a/n/a     │",
        bgz_tensor::quality::pearson(&gt_cosines, &b17_l1),
        bgz_tensor::quality::spearman(&gt_cosines, &b17_l1));
    println!("└──────┴────────┴──────────────────┴──────────────────┴──────────────────┘\n");

    // ═══════════════════════════════════════════════════════════════════════
    // PART 2: Codebook size comparison (CLAM cosine)
    // ═══════════════════════════════════════════════════════════════════════
    println!("======================================================================");
    println!("PART 2: Codebook size (CLAM cosine, SPD=16)");
    println!("======================================================================\n");

    let encoded_16: Vec<bgz_tensor::StackedN> = raw_vectors.iter()
        .map(|v| bgz_tensor::StackedN::from_f32(v, 16))
        .collect();

    let codebook_sizes = [8, 16, 32]; // limited by n=40 vectors
    println!("┌──────────┬───────┬──────────┬──────────┬──────────────┐");
    println!("│ Entries  │ Bits  │ KB       │ Pearson  │ Spearman     │");
    println!("├──────────┼───────┼──────────┼──────────┼──────────────┤");

    for &k in &codebook_sizes {
        let cb = bgz_tensor::ClamCodebook::build_cosine(&encoded_16, k);
        let cb_cos: Vec<f64> = gt_pairs.iter()
            .map(|&(i, j, _)| {
                let a = cb.get(cb.assignments[i]);
                let b = cb.get(cb.assignments[j]);
                match (a, b) { (Some(a), Some(b)) => a.stacked.cosine(&b.stacked), _ => 0.0 }
            })
            .collect();

        println!("│ {:>8} │ {:>5} │ {:>8.1} │ {:>8.4} │ {:>12.4} │",
            k, (k as f64).log2().ceil() as u32,
            cb.byte_size() as f64 / 1024.0,
            bgz_tensor::quality::pearson(&gt_cosines, &cb_cos),
            bgz_tensor::quality::spearman(&gt_cosines, &cb_cos));
    }
    println!("└──────────┴───────┴──────────┴──────────┴──────────────┘\n");

    // ═══════════════════════════════════════════════════════════════════════
    // PART 3: HDR Popcount Cascade with Leaf Hydration
    // ═══════════════════════════════════════════════════════════════════════
    println!("======================================================================");
    println!("PART 3: HDR Popcount Cascade + Leaf Hydration (SPD=16)");
    println!("======================================================================\n");

    let cb_16 = bgz_tensor::ClamCodebook::build_cosine(&encoded_16, 16);

    for (label, config) in &[
        ("Permissive", bgz_tensor::HdrCascadeConfig {
            popcount_min_agreement: 0.3,
            codebook_max_distance: 0.9,
            stacked_min_cosine: -0.5,
        }),
        ("Moderate", bgz_tensor::HdrCascadeConfig {
            popcount_min_agreement: 0.4,
            codebook_max_distance: 0.7,
            stacked_min_cosine: 0.0,
        }),
        ("Strict", bgz_tensor::HdrCascadeConfig {
            popcount_min_agreement: 0.5,
            codebook_max_distance: 0.5,
            stacked_min_cosine: 0.3,
        }),
    ] {
        let (results, stats) = bgz_tensor::stacked_n::hdr_cascade(
            &encoded_16, &cb_16, config, gt_pairs.len()
        );
        println!("--- {} cascade ---", label);
        println!("{}", stats.summary());

        // Quality: what fraction of top-K similar pairs survived?
        let mut gt_sorted = gt_pairs.clone();
        gt_sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        let top_k = 20;
        let top_set: std::collections::HashSet<(usize, usize)> = gt_sorted[..top_k]
            .iter().map(|p| (p.0, p.1)).collect();
        let survived_top = results.iter()
            .filter(|r| top_set.contains(&(r.idx_a, r.idx_b)))
            .count();
        println!("Top-{} recall: {}/{} ({:.0}%)", top_k, survived_top, top_k,
            survived_top as f64 / top_k as f64 * 100.0);

        // Leaf cosine vs ground truth for survivors
        if !results.is_empty() {
            let leaf_cosines: Vec<f64> = results.iter().map(|r| r.leaf_cosine).collect();
            let survivor_gt: Vec<f64> = results.iter()
                .map(|r| gt_pairs.iter()
                    .find(|p| p.0 == r.idx_a && p.1 == r.idx_b)
                    .map(|p| p.2).unwrap_or(0.0))
                .collect();
            let leaf_pearson = bgz_tensor::quality::pearson(&survivor_gt, &leaf_cosines);
            println!("Leaf↔GT Pearson: {:.4} (leaf hydration fidelity)", leaf_pearson);

            // Show top 5 leaf results
            let mut sorted_results = results.clone();
            sorted_results.sort_by(|a, b| b.leaf_cosine.partial_cmp(&a.leaf_cosine).unwrap());
            println!("\nTop 5 leaf-hydrated pairs:");
            for (i, r) in sorted_results.iter().take(5).enumerate() {
                let gt = gt_pairs.iter()
                    .find(|p| p.0 == r.idx_a && p.1 == r.idx_b)
                    .map(|p| p.2).unwrap_or(0.0);
                println!("  {}. leaf={:.4} gt={:.4} err={:.4} | {} ↔ {}",
                    i + 1, r.leaf_cosine, gt, (r.leaf_cosine - gt).abs(),
                    truncate(all_texts[r.idx_a], 30), truncate(all_texts[r.idx_b], 30));
            }
        }
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PART 4: Leaf hydration vs API ground truth (the key question)
    // ═══════════════════════════════════════════════════════════════════════
    println!("======================================================================");
    println!("PART 4: Leaf Hydration Fidelity (BF16 > f32 vs API f32)");
    println!("======================================================================\n");

    println!("SPD │ Leaf↔API Pearson │ Leaf↔API Spearman │ Max error │ Mean error");
    println!("────┼─────────────────┼───────────────────┼───────────┼───────────");
    for &spd in &sample_counts {
        let encoded: Vec<bgz_tensor::StackedN> = raw_vectors.iter()
            .map(|v| bgz_tensor::StackedN::from_f32(v, spd))
            .collect();
        let hydrated: Vec<Vec<f32>> = encoded.iter().map(|e| e.hydrate_f32()).collect();

        let leaf_cos: Vec<f64> = gt_pairs.iter()
            .map(|&(i, j, _)| bgz_tensor::stacked_n::cosine_f32_slice(&hydrated[i], &hydrated[j]))
            .collect();

        let errors: Vec<f64> = gt_cosines.iter().zip(leaf_cos.iter())
            .map(|(gt, lf)| (gt - lf).abs())
            .collect();

        println!("{:>3} │ {:>15.4} │ {:>17.4} │ {:>9.4} │ {:>9.4}",
            spd,
            bgz_tensor::quality::pearson(&gt_cosines, &leaf_cos),
            bgz_tensor::quality::spearman(&gt_cosines, &leaf_cos),
            errors.iter().cloned().fold(0.0f64, f64::max),
            errors.iter().sum::<f64>() / errors.len() as f64,
        );
    }

    println!("\n=== PIPELINE COMPLETE ===");
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() } else { format!("{}...", &s[..max.min(s.len()) - 3]) }
}
