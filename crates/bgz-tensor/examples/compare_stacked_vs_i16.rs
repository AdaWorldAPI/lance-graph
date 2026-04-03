//! Compare 4096×stacked vs 256×i16 phase resolution.
//!
//! Usage: cargo run --manifest-path crates/bgz-tensor/Cargo.toml --example compare_stacked_vs_i16
//!
//! Generates test vectors with known phase relationships, encodes both ways,
//! and measures which preserves cosine/distance ranking better.

fn main() {
    println!("=== Phase Resolution: 4096×Stacked BF16×4 vs 256×i16 Base17 ===\n");

    // Generate vectors with controlled phase relationships
    let n_vectors = 500;
    let dim = 4096;

    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|i| {
            (0..dim)
                .map(|d| {
                    let freq = 0.01 + (i as f64 * 0.001);
                    let phase = i as f64 * 0.1;
                    ((d as f64 * freq + phase).sin() * 0.5
                        + (d as f64 * freq * 2.3 + phase * 0.7).cos() * 0.3) as f32
                })
                .collect()
        })
        .collect();

    // Compute ground-truth pairwise cosines (sample)
    let n_pairs = 2000;
    let mut pairs: Vec<(usize, usize, f64)> = Vec::with_capacity(n_pairs);
    let mut pair_idx = 0;
    'outer: for i in 0..n_vectors {
        for j in (i + 1)..n_vectors {
            pairs.push((i, j, cosine_f32(&vectors[i], &vectors[j])));
            pair_idx += 1;
            if pair_idx >= n_pairs {
                break 'outer;
            }
        }
    }

    println!("Ground truth: {} pairs, dim={}", pairs.len(), dim);
    let gt_cosines: Vec<f64> = pairs.iter().map(|p| p.2).collect();
    println!(
        "Cosine range: [{:.4}, {:.4}], mean={:.4}\n",
        gt_cosines.iter().cloned().fold(f64::INFINITY, f64::min),
        gt_cosines.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        gt_cosines.iter().sum::<f64>() / gt_cosines.len() as f64,
    );

    // === Encode as Base17 (i16, 34 bytes) ===
    let base17_vecs: Vec<bgz_tensor::Base17> = vectors
        .iter()
        .map(|v| bgz_tensor::Base17::from_f32(v))
        .collect();

    let b17_cosines: Vec<f64> = pairs
        .iter()
        .map(|&(i, j, _)| base17_vecs[i].cosine(&base17_vecs[j]))
        .collect();

    let b17_l1: Vec<f64> = pairs
        .iter()
        .map(|&(i, j, _)| -(base17_vecs[i].l1(&base17_vecs[j]) as f64))
        .collect();

    // === Encode as StackedBF16×4 (136 bytes) ===
    let stacked_vecs: Vec<bgz_tensor::StackedBF16x4> = vectors
        .iter()
        .map(|v| bgz_tensor::StackedBF16x4::from_f32(v))
        .collect();

    let stacked_cosines: Vec<f64> = pairs
        .iter()
        .map(|&(i, j, _)| stacked_vecs[i].cosine(&stacked_vecs[j]))
        .collect();

    let stacked_full: Vec<f64> = pairs
        .iter()
        .map(|&(i, j, _)| -(stacked_vecs[i].full_distance(&stacked_vecs[j]) as f64))
        .collect();

    let stacked_vedic: Vec<f64> = pairs
        .iter()
        .map(|&(i, j, _)| -(stacked_vecs[i].vedic_upper_distance(&stacked_vecs[j]) as f64))
        .collect();

    // === Build 256-palette from Base17 ===
    let palette256 = bgz_tensor::WeightPalette::build(&base17_vecs, 256);
    let palette_indices: Vec<u8> = palette256.assign_all(&base17_vecs);
    let attn_table = bgz_tensor::attention::AttentionTable::build(&palette256);

    let palette_distances: Vec<f64> = pairs
        .iter()
        .map(|&(i, j, _)| -(attn_table.distance(palette_indices[i], palette_indices[j]) as f64))
        .collect();

    // === Build 4096-codebook from stacked ===
    let codebook = bgz_tensor::Codebook4096::build(&stacked_vecs, 64);
    let cb_indices: Vec<bgz_tensor::CodebookIndex> = codebook.assign_all(&stacked_vecs);

    let codebook_distances: Vec<f64> = pairs
        .iter()
        .map(|&(i, j, _)| {
            let a = codebook.lookup(cb_indices[i]);
            let b = codebook.lookup(cb_indices[j]);
            match (a, b) {
                (Some(ea), Some(eb)) => -(ea.full_distance(eb) as f64),
                _ => 0.0,
            }
        })
        .collect();

    // === Compute correlations ===
    println!("┌─────────────────────────────────────┬──────────┬──────────┐");
    println!("│ Encoding                            │  Pearson │ Spearman │");
    println!("├─────────────────────────────────────┼──────────┼──────────┤");

    let encodings: Vec<(&str, &[f64])> = vec![
        ("Base17 i16 cosine (34 B)", &b17_cosines),
        ("Base17 i16 neg-L1 (34 B)", &b17_l1),
        ("Stacked BF16×4 cosine (136 B)", &stacked_cosines),
        ("Stacked BF16×4 neg-full-L1 (136B)", &stacked_full),
        ("Stacked BF16×4 neg-vedic (136 B)", &stacked_vedic),
        ("256-palette distance (1 B)", &palette_distances),
        ("4096-codebook distance (2 B)", &codebook_distances),
    ];

    for (name, values) in &encodings {
        let pearson = bgz_tensor::quality::pearson(&gt_cosines, values);
        let spearman = bgz_tensor::quality::spearman(&gt_cosines, values);
        println!("│ {:<35} │ {:>8.4} │ {:>8.4} │", name, pearson, spearman);
    }
    println!("└─────────────────────────────────────┴──────────┴──────────┘");

    // === Size comparison ===
    println!("\n=== Size Comparison ===");
    println!(
        "Base17 (i16):     {:>6} B/vector, {:>8} B total",
        34,
        34 * n_vectors
    );
    println!(
        "Stacked BF16×4:   {:>6} B/vector, {:>8} B total",
        136,
        136 * n_vectors
    );
    println!(
        "256-palette:      {:>6} B/vector + {:>6} B codebook = {:>8} B",
        1,
        256 * 34,
        n_vectors + 256 * 34
    );
    println!(
        "4096-codebook:    {:>6} B/vector + {:>6} B codebook = {:>8} B",
        2,
        codebook.byte_size(),
        n_vectors * 2 + codebook.byte_size()
    );
    println!("\n{}", codebook.summary());

    // === Search key comparison ===
    println!("\n=== Search Key HEEL Performance ===");
    let search_keys: Vec<bgz_tensor::SearchKey17> = stacked_vecs
        .iter()
        .map(|s| s.search_key())
        .collect();

    // Measure sign agreement vs cosine correlation
    let sign_agreements: Vec<f64> = pairs
        .iter()
        .map(|&(i, j, _)| search_keys[i].sign_agreement(&search_keys[j]) as f64)
        .collect();
    let mag_agreements: Vec<f64> = pairs
        .iter()
        .map(|&(i, j, _)| search_keys[i].magnitude_agreement(&search_keys[j]) as f64)
        .collect();
    let key_l1: Vec<f64> = pairs
        .iter()
        .map(|&(i, j, _)| -(search_keys[i].l1(&search_keys[j]) as f64))
        .collect();

    println!(
        "SearchKey17 (17 B) neg-L1:      Pearson={:.4}, Spearman={:.4}",
        bgz_tensor::quality::pearson(&gt_cosines, &key_l1),
        bgz_tensor::quality::spearman(&gt_cosines, &key_l1),
    );
    println!(
        "SearchKey17 sign agreement:     Pearson={:.4}",
        bgz_tensor::quality::pearson(&gt_cosines, &sign_agreements),
    );
    println!(
        "SearchKey17 magnitude agreement: Pearson={:.4}",
        bgz_tensor::quality::pearson(&gt_cosines, &mag_agreements),
    );

    // === Phase-specific test: vectors differing only in phase ===
    println!("\n=== Phase-Specific Discrimination ===");
    let phase_pairs: Vec<(Vec<f32>, Vec<f32>, f64)> = (0..20)
        .map(|i| {
            let phase_a = 0.0;
            let phase_b = (i as f64 + 1.0) * 0.05; // 0.05 to 1.0 radians
            let a: Vec<f32> = (0..dim)
                .map(|d| ((d as f64 * 0.01 + phase_a).sin() * 0.5) as f32)
                .collect();
            let b: Vec<f32> = (0..dim)
                .map(|d| ((d as f64 * 0.01 + phase_b).sin() * 0.5) as f32)
                .collect();
            let cos = cosine_f32(&a, &b);
            (a, b, cos)
        })
        .collect();

    println!("Phase offset | True cos | Base17 cos | Stacked cos | Stacked err | Base17 err");
    for (i, (a, b, true_cos)) in phase_pairs.iter().enumerate() {
        let b17_a = bgz_tensor::Base17::from_f32(a);
        let b17_b = bgz_tensor::Base17::from_f32(b);
        let b17_cos = b17_a.cosine(&b17_b);

        let st_a = bgz_tensor::StackedBF16x4::from_f32(a);
        let st_b = bgz_tensor::StackedBF16x4::from_f32(b);
        let st_cos = st_a.cosine(&st_b);

        let phase = (i as f64 + 1.0) * 0.05;
        println!(
            "  {:.2} rad    | {:>8.4} | {:>10.4} | {:>11.4} | {:>11.4} | {:>10.4}",
            phase,
            true_cos,
            b17_cos,
            st_cos,
            (st_cos - true_cos).abs(),
            (b17_cos - true_cos).abs(),
        );
    }

    println!("\n=== DONE ===");
}

fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}
