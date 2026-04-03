//! Euler-gamma fold on real Jina API embedding clusters.
//!
//! Embeddings cluster naturally (similar texts → similar vectors).
//! This is WHERE the fold works: high intra-family cosine.
//!
//! cargo run --manifest-path crates/bgz-tensor/Cargo.toml --example fold_jina_embeddings

use bgz_tensor::euler_fold::{clam_group, euler_gamma_fold, euler_gamma_unfold, gate_test};
use bgz_tensor::stacked_n::cosine_f32_slice;
use bgz_tensor::neuron_hetero::ThinkingStyleFingerprint;

fn main() {
    println!("=== Euler-Gamma Fold on Real Jina Embedding Clusters ===\n");

    // Load Jina embeddings
    let texts = vec![
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

    let mut raw_vectors: Vec<Vec<f32>> = Vec::new();
    for path in &["/tmp/jina_batch1.json", "/tmp/jina_batch2.json"] {
        let text_slice = if *path == "/tmp/jina_batch1.json" { &texts[..20] } else { &texts[20..] };
        match std::fs::read_to_string(path) {
            Ok(json) => {
                let refs: Vec<&str> = text_slice.iter().copied().collect();
                match bgz_tensor::jina::parse_jina_response(&json, &refs) {
                    Ok(embs) => { for e in embs { raw_vectors.push(e.vector); } }
                    Err(e) => { eprintln!("Parse error: {}", e); return; }
                }
            }
            Err(e) => { eprintln!("Cannot read {}: {}", path, e); return; }
        }
    }

    let n = raw_vectors.len();
    println!("{} embeddings loaded (dim={})\n", n, raw_vectors[0].len());

    // ═══ PART 1: CLAM family grouping ═══════════════════════════════════
    println!("=== PART 1: CLAM Families in Embedding Space ===\n");

    for threshold in [0.50, 0.60, 0.70, 0.80] {
        let families = clam_group(&raw_vectors, threshold);
        let multi: Vec<_> = families.iter().filter(|f| f.member_indices.len() >= 2).collect();
        let sizes: Vec<usize> = multi.iter().map(|f| f.member_indices.len()).collect();

        println!("cos>={:.2}: {} families, {} multi-member, sizes: {:?}",
            threshold, families.len(), multi.len(), sizes);

        // Show which texts cluster together
        if threshold == 0.70 {
            for (fi, fam) in multi.iter().enumerate().take(5) {
                let members: Vec<&str> = fam.member_indices.iter()
                    .map(|&i| texts[i])
                    .collect();
                println!("  Family {}: {:?}", fi, members.iter()
                    .map(|t| if t.len() > 40 { &t[..40] } else { t })
                    .collect::<Vec<_>>());
            }
        }
    }

    // ═══ PART 2: Euler-gamma fold on embedding families ═════════════════
    println!("\n=== PART 2: Euler-Gamma Fold on Embedding Families ===\n");

    let families = clam_group(&raw_vectors, 0.60);
    let multi: Vec<_> = families.iter().filter(|f| f.member_indices.len() >= 2).collect();

    if multi.is_empty() {
        println!("No families with >=2 members at cos=0.60. Trying lower threshold...");
        return;
    }

    for fam in &multi {
        let members: Vec<Vec<f32>> = fam.member_indices.iter()
            .map(|&i| raw_vectors[i].clone())
            .collect();

        let result = gate_test(&members, 32);
        let member_texts: Vec<&str> = fam.member_indices.iter()
            .map(|&i| if texts[i].len() > 35 { &texts[i][..35] } else { texts[i] })
            .collect();

        println!("Family (n={}): mean_rho={:.4}, min_rho={:.4}, ratio={:.1}x",
            members.len(), result.mean_pearson, result.min_pearson, result.compression_ratio);
        println!("  Texts: {:?}", member_texts);
        for (j, &rho) in result.pearson_per_member.iter().enumerate() {
            println!("    [{}] rho={:.4} \"{}\"", j, rho,
                if texts[fam.member_indices[j]].len() > 50 {
                    &texts[fam.member_indices[j]][..50]
                } else { texts[fam.member_indices[j]] });
        }
        println!();
    }

    // ═══ PART 3: ThinkingStyleFingerprint on embeddings ═════════════════
    println!("=== PART 3: Thinking Style from Embedding Vectors ===\n");
    println!("(Treating each embedding as if it were a Gate weight row)\n");

    let fingerprints: Vec<ThinkingStyleFingerprint> = raw_vectors.iter()
        .map(|v| ThinkingStyleFingerprint::from_gate_weights(v))
        .collect();

    // Cluster fingerprints by Hamming distance
    println!("Top 10 most similar fingerprint pairs:");
    let mut pairs: Vec<(usize, usize, u32)> = Vec::new();
    for i in 0..n {
        for j in (i+1)..n {
            pairs.push((i, j, fingerprints[i].bit_disagreements(&fingerprints[j])));
        }
    }
    pairs.sort_by_key(|p| p.2);

    for (rank, &(i, j, d)) in pairs.iter().take(10).enumerate() {
        let cos = cosine_f32_slice(&raw_vectors[i], &raw_vectors[j]);
        println!("  {:>2}. hamming={:>2} cos={:.4} | {} <-> {}",
            rank + 1, d, cos,
            if texts[i].len() > 30 { &texts[i][..30] } else { texts[i] },
            if texts[j].len() > 30 { &texts[j][..30] } else { texts[j] });
    }

    // Correlation: does Hamming distance predict cosine?
    let hamming_dists: Vec<f64> = pairs.iter().map(|p| -(p.2 as f64)).collect();
    let cosines: Vec<f64> = pairs.iter()
        .map(|&(i, j, _)| cosine_f32_slice(&raw_vectors[i], &raw_vectors[j]))
        .collect();
    let pearson = bgz_tensor::quality::pearson(&hamming_dists, &cosines);
    let spearman = bgz_tensor::quality::spearman(&hamming_dists, &cosines);
    println!("\nFingerprint Hamming vs Cosine: Pearson={:.4}, Spearman={:.4}", pearson, spearman);

    println!("\n=== DONE ===");
}
