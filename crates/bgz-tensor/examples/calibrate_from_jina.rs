//! Calibrate bgz-tensor from real Jina API embeddings.
//!
//! Usage: cargo run --manifest-path crates/bgz-tensor/Cargo.toml --example calibrate_from_jina
//!
//! Reads Jina API JSON responses from /tmp/jina_batch*.json,
//! computes ground truth, calibrates SimilarityTable and Belichtungsmesser.

fn main() {
    let texts_batch1 = vec![
        "The cat sat on the mat.",
        "A cat was sitting on the mat.",
        "Machine learning is a subset of artificial intelligence.",
        "ML is a branch of AI.",
        "The stock market crashed today.",
        "Today the stock market experienced a crash.",
        "Python is a programming language.",
        "JavaScript is used for web development.",
        "The sun rises in the east.",
        "Sunset paints the western sky.",
        "He ran quickly to the store.",
        "She walked slowly to the market.",
        "Quantum mechanics describes subatomic particles.",
        "The recipe calls for two cups of flour.",
        "Mount Everest is the tallest mountain.",
        "Abstract art challenges traditional aesthetics.",
        "Hello, how are you?",
        "Hallo, wie geht es dir?",
        "Good morning, world.",
        "Guten Morgen, Welt.",
    ];

    let texts_batch2 = vec![
        "The Eiffel Tower is in Paris.",
        "La Tour Eiffel se trouve a Paris.",
        "Water boils at 100 degrees Celsius.",
        "The boiling point of water is 100C.",
        "Dogs are loyal companions.",
        "Cats are independent creatures.",
        "The speed of light is approximately 300000 km per second.",
        "Photons travel at about 300000 kilometers every second.",
        "Shakespeare wrote Romeo and Juliet.",
        "The famous love tragedy was written by Shakespeare.",
        "Bitcoin is a cryptocurrency.",
        "Ethereum uses smart contracts for decentralized applications.",
        "Neural networks learn from data.",
        "Deep learning models require large datasets for training.",
        "The Pacific Ocean is the largest ocean.",
        "Antarctica is the coldest continent on Earth.",
        "Bach composed the Brandenburg Concertos.",
        "Beethoven wrote nine symphonies.",
        "Rust is a systems programming language.",
        "Go is designed for concurrent programming.",
    ];

    // Load and parse embeddings
    let mut all_embeddings = Vec::new();

    for (path, texts) in &[
        ("/tmp/jina_batch1.json", &texts_batch1),
        ("/tmp/jina_batch2.json", &texts_batch2),
    ] {
        match std::fs::read_to_string(path) {
            Ok(json) => {
                let text_refs: Vec<&str> = texts.iter().map(|s| *s).collect();
                match bgz_tensor::jina::parse_jina_response(&json, &text_refs) {
                    Ok(embs) => {
                        println!("Loaded {} embeddings from {}", embs.len(), path);
                        all_embeddings.extend(embs);
                    }
                    Err(e) => eprintln!("Parse error for {}: {}", path, e),
                }
            }
            Err(e) => eprintln!("Cannot read {}: {}", path, e),
        }
    }

    if all_embeddings.is_empty() {
        eprintln!("No embeddings loaded. Run Jina API calls first.");
        return;
    }

    println!("\n=== Ground Truth Collection ===");
    println!("Total embeddings: {}", all_embeddings.len());
    println!("Embedding dim: {}", all_embeddings[0].vector.len());

    // Collect ground truth pairs
    let pairs = bgz_tensor::jina::collect_ground_truth(&all_embeddings);
    println!("Total pairs: {} (C({},2))", pairs.len(), all_embeddings.len());

    // Summary statistics
    let summary = bgz_tensor::jina::summarize_ground_truth(&pairs);
    println!("\n=== Summary ===");
    println!("Mean API cosine:      {:.4}", summary.mean_api_cosine);
    println!("Std API cosine:       {:.4}", summary.std_api_cosine);
    println!("Mean Base17 L1:       {:.1}", summary.mean_base17_l1);
    println!("Mean Base17 cosine:   {:.4}", summary.mean_base17_cosine);
    println!("Pearson(neg_L1, cos): {:.4}", summary.pearson_l1_vs_cosine);
    println!("Spearman(neg_L1,cos): {:.4}", summary.spearman_l1_vs_cosine);
    println!("Pearson(b17cos,cos):  {:.4}", summary.pearson_base17cos_vs_apicos);

    // Show top-10 most similar pairs
    println!("\n=== Top 10 Most Similar Pairs ===");
    for (i, p) in pairs.iter().take(10).enumerate() {
        println!("{:2}. cos={:.4} L1={:5} b17cos={:.4} | {} ↔ {}",
            i + 1, p.api_cosine, p.base17_l1, p.base17_cosine,
            truncate(&p.text_a, 40), truncate(&p.text_b, 40));
    }

    // Show bottom-10 (most dissimilar)
    println!("\n=== Bottom 10 Most Dissimilar Pairs ===");
    for (i, p) in pairs.iter().rev().take(10).enumerate() {
        println!("{:2}. cos={:.4} L1={:5} b17cos={:.4} | {} ↔ {}",
            i + 1, p.api_cosine, p.base17_l1, p.base17_cosine,
            truncate(&p.text_a, 40), truncate(&p.text_b, 40));
    }

    // Calibrate SimilarityTable
    println!("\n=== SimilarityTable Calibration ===");
    let cal_pairs = bgz_tensor::jina::to_calibration_pairs(&pairs);
    let table = bgz_tensor::similarity::SimilarityTable::calibrate(&cal_pairs);
    println!("Max L1: {}", table.max_l1);
    println!("Similarity at L1=0:     {:.4}", table.similarity(0));
    println!("Similarity at L1=1000:  {:.4}", table.similarity(1000));
    println!("Similarity at L1=5000:  {:.4}", table.similarity(5000));
    println!("Similarity at L1=10000: {:.4}", table.similarity(10000));

    // Validate table: compute Spearman between table output and API cosine
    let table_sims: Vec<f64> = pairs.iter()
        .map(|p| table.similarity(p.base17_l1) as f64)
        .collect();
    let api_cosines: Vec<f64> = pairs.iter().map(|p| p.api_cosine).collect();
    let table_spearman = bgz_tensor::quality::spearman(&table_sims, &api_cosines);
    println!("Spearman(table_sim, api_cos): {:.4}", table_spearman);

    // Calibrate Belichtungsmesser
    println!("\n=== Belichtungsmesser Calibration ===");
    let l1_distances: Vec<u32> = pairs.iter().map(|p| p.base17_l1).collect();
    let bel = bgz_tensor::belichtungsmesser::Belichtungsmesser::calibrate(&l1_distances);
    println!("{}", bel.summary());

    // Validate: false negative rate
    let val_pairs: Vec<(bgz_tensor::Base17, bgz_tensor::Base17, f64)> = all_embeddings.iter()
        .enumerate()
        .flat_map(|(i, a)| {
            all_embeddings[i + 1..].iter().map(move |b| {
                let cos = bgz_tensor::jina::cosine_f32(&a.vector, &b.vector);
                (a.base17.clone(), b.base17.clone(), cos)
            })
        })
        .collect();

    let (fn_rate, agreement_rate) = bel.validate(&val_pairs, 0.5, 6);
    println!("False negative rate (cosine > 0.5): {:.4} (target < 0.01)", fn_rate);
    println!("Band agreement rate (3-stroke):     {:.4}", agreement_rate);

    // Band distribution of similar vs dissimilar pairs
    println!("\n=== Band Distribution ===");
    let mut band_counts_sim = [0u32; 12];
    let mut band_counts_dis = [0u32; 12];
    for (a, b, cos) in &val_pairs {
        let (band, _) = bel.three_stroke(a, b);
        if *cos > 0.5 {
            band_counts_sim[band as usize] += 1;
        } else {
            band_counts_dis[band as usize] += 1;
        }
    }
    println!("Band | Similar | Dissimilar");
    for b in 0..12 {
        if band_counts_sim[b] > 0 || band_counts_dis[b] > 0 {
            println!("  {:2}  | {:>7} | {:>10}", b, band_counts_sim[b], band_counts_dis[b]);
        }
    }

    println!("\n=== CALIBRATION COMPLETE ===");
    println!("Texts: {}, Pairs: {}", all_embeddings.len(), pairs.len());
    println!("Target Spearman > 0.85: {} (actual: {:.4})",
        if table_spearman > 0.85 { "PASS" } else { "FAIL" }, table_spearman);
    println!("Target FN rate < 0.01:  {} (actual: {:.4})",
        if fn_rate < 0.01 { "PASS" } else if fn_rate < 0.05 { "MARGINAL" } else { "FAIL" }, fn_rate);
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() } else { format!("{}...", &s[..max - 3]) }
}
