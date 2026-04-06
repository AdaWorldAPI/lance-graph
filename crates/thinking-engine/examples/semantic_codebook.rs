//! Build SEMANTIC codebook: 256 forward passes on centroid representatives.
//!
//! Token embeddings have no semantic structure (ρ=0.54 vs ground truth).
//! The 28-layer forward pass CREATES the structure.
//! Run one forward pass per centroid representative → 256 semantic embeddings
//! → pairwise cosine → semantic distance table (128 KB i16).
//!
//! cargo run --release --features calibration \
//!   --manifest-path crates/thinking-engine/Cargo.toml \
//!   --example semantic_codebook

#[cfg(feature = "calibration")]
fn main() {
    use candle_core::{Device, DType, Tensor, IndexOp};
    use candle_nn::VarBuilder;
    use candle_transformers::models::qwen3;
    use std::time::Instant;

    let device = Device::Cpu;
    let dtype = DType::F32;

    println!("═══════════════════════════════════════════════════════════");
    println!("  SEMANTIC CODEBOOK: 256 forward passes → semantic table");
    println!("═══════════════════════════════════════════════════════════\n");

    // Load config
    let config: qwen3::Config = serde_json::from_str(
        &std::fs::read_to_string("data/jina-v5-onnx/config_candle.json").expect("config")
    ).expect("parse");
    println!("[1] Jina v5: {} layers, {}D hidden", config.num_hidden_layers, config.hidden_size);

    let model_path = "data/jina-v5-onnx/model.safetensors";

    // Load codebook index to find representative tokens
    let idx_data = std::fs::read("/tmp/codebooks/jina-v5-256/codebook_index.u16").expect("idx");
    let codebook_idx: Vec<u16> = idx_data.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

    // Load token embeddings to find best representative per centroid
    let embed_data = std::fs::read("data/jina-v5-onnx/model.safetensors").expect("safetensors");
    let n_cent: usize = 256;
    let hidden = config.hidden_size;
    let vocab = codebook_idx.len();

    // Find representative token for each centroid (token closest to centroid mean)
    // For speed: just use the most common token in each centroid
    println!("[2] Finding 256 centroid representatives...");
    let mut rep_tokens: Vec<u32> = vec![0u32; n_cent];
    let mut best_count: Vec<u32> = vec![0u32; n_cent];

    // Simple: pick a real word token for each centroid
    // Use the centroid index directly — just need ANY token from each centroid
    let mut centroid_token_map: Vec<Vec<u32>> = vec![Vec::new(); n_cent];
    for (tok, &cent) in codebook_idx.iter().enumerate() {
        centroid_token_map[cent as usize].push(tok as u32);
    }
    for c in 0..n_cent {
        // Pick the token closest to index 1000-10000 range (likely real words, not special tokens)
        rep_tokens[c] = centroid_token_map[c].iter()
            .find(|&&t| t > 500 && t < 50000)
            .copied()
            .unwrap_or(centroid_token_map[c].first().copied().unwrap_or(0));
    }

    // Load tokenizer for display
    let tokenizer = tokenizers::Tokenizer::from_file("data/readerlm-v2/tokenizer.json")
        .expect("tokenizer");

    println!("[3] Running 256 forward passes (this takes ~10-15 minutes)...");
    let mut semantic_embeddings: Vec<Vec<f32>> = Vec::with_capacity(n_cent);

    let total_start = Instant::now();
    for c in 0..n_cent {
        let tok_id = rep_tokens[c];
        // Build a minimal context: just the token repeated a few times
        // This gives the model enough context to produce a meaningful embedding
        let input_ids = vec![tok_id; 4]; // 4 copies for minimal context

        // Fresh model per centroid (avoid KV cache contamination)
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device).expect("load")
        }.rename_f(|name| name.strip_prefix("model.").unwrap_or(name).to_string());

        let mut model = qwen3::Model::new(&config, vb).expect("build");

        let input = Tensor::new(&input_ids[..], &device).expect("t").unsqueeze(0).expect("b");
        let output = model.forward(&input, 0).expect("forward");

        // Last-token pooling + L2 normalize
        let last = output.i((0, input_ids.len() - 1)).expect("last");
        let emb: Vec<f32> = last.to_vec1().expect("vec");
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let emb_norm: Vec<f32> = if norm > 1e-12 {
            emb.iter().map(|x| x / norm).collect()
        } else { emb };

        semantic_embeddings.push(emb_norm);

        if c % 16 == 0 || c == n_cent - 1 {
            let elapsed = total_start.elapsed().as_secs_f64();
            let eta = if c > 0 { elapsed / c as f64 * (n_cent - c) as f64 } else { 0.0 };
            let decoded = tokenizer.decode(&[tok_id], false).unwrap_or_default();
            println!("  [{:>3}/{}] token {:>6} = {:>15} ({:.1}s elapsed, ETA {:.0}s)",
                c + 1, n_cent, tok_id, &decoded[..decoded.len().min(15)], elapsed, eta);
        }
    }
    let total_elapsed = total_start.elapsed().as_secs_f64();
    println!("  Done: {} forward passes in {:.1}s ({:.1}s/pass)",
        n_cent, total_elapsed, total_elapsed / n_cent as f64);

    // Build pairwise cosine table
    println!("\n[4] Building semantic distance table...");
    let mut semantic_table = vec![0.0f32; n_cent * n_cent];
    for i in 0..n_cent {
        semantic_table[i * n_cent + i] = 1.0;
        for j in (i + 1)..n_cent {
            let cos: f32 = semantic_embeddings[i].iter()
                .zip(&semantic_embeddings[j])
                .map(|(a, b)| a * b).sum();
            semantic_table[i * n_cent + j] = cos;
            semantic_table[j * n_cent + i] = cos;
        }
    }

    let mut all_cos: Vec<f32> = Vec::new();
    for i in 0..n_cent {
        for j in (i + 1)..n_cent {
            all_cos.push(semantic_table[i * n_cent + j]);
        }
    }
    all_cos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("  Cosine: [{:.4}, {:.4}] mean={:.4}",
        all_cos.first().unwrap(), all_cos.last().unwrap(),
        all_cos.iter().sum::<f32>() / all_cos.len() as f32);

    // Compare with token embedding table
    let token_table_data = std::fs::read("/tmp/codebooks/jina-v5-256/cosine_matrix_256x256.f32").expect("token table");
    let token_table: Vec<f32> = token_table_data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    // Spearman rank correlation
    let n_pairs = all_cos.len();
    let st_ref = &semantic_table;
    let tt_ref = &token_table;
    let sem_vals: Vec<f32> = (0..n_cent).flat_map(|i| (i+1..n_cent).map(move |j| st_ref[i * n_cent + j])).collect();
    let tok_vals: Vec<f32> = (0..n_cent).flat_map(|i| (i+1..n_cent).map(move |j| tt_ref[i * n_cent + j])).collect();

    // Quick Spearman
    let mut sem_rank: Vec<(usize, f32)> = sem_vals.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    let mut tok_rank: Vec<(usize, f32)> = tok_vals.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    sem_rank.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    tok_rank.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut sr = vec![0.0f32; n_pairs];
    let mut tr = vec![0.0f32; n_pairs];
    for (rank, &(idx, _)) in sem_rank.iter().enumerate() { sr[idx] = rank as f32; }
    for (rank, &(idx, _)) in tok_rank.iter().enumerate() { tr[idx] = rank as f32; }
    let mean_s: f32 = sr.iter().sum::<f32>() / n_pairs as f32;
    let mean_t: f32 = tr.iter().sum::<f32>() / n_pairs as f32;
    let num: f32 = sr.iter().zip(&tr).map(|(a, b)| (a - mean_s) * (b - mean_t)).sum();
    let den_s: f32 = sr.iter().map(|a| (a - mean_s).powi(2)).sum::<f32>().sqrt();
    let den_t: f32 = tr.iter().map(|b| (b - mean_t).powi(2)).sum::<f32>().sqrt();
    let rho = if den_s > 0.0 && den_t > 0.0 { num / (den_s * den_t) } else { 0.0 };

    println!("\n[5] Token table vs Semantic table:");
    println!("  Spearman ρ = {:.4}", rho);
    println!("  (ρ < 0.5 = semantic table is DIFFERENT from token table = GOOD)");
    println!("  (ρ ≈ 1.0 = same as token table = no improvement)");

    // Save semantic table
    let outdir = "/tmp/codebooks/jina-v5-semantic-256";
    std::fs::create_dir_all(outdir).ok();

    // f32
    let f32_bytes: Vec<u8> = semantic_table.iter().flat_map(|c| c.to_le_bytes()).collect();
    std::fs::write(format!("{}/cosine_matrix_256x256.f32", outdir), &f32_bytes).ok();

    // i16
    let i16_table: Vec<i16> = semantic_table.iter()
        .map(|&c| (c * 32767.0).round().clamp(-32768.0, 32767.0) as i16).collect();
    let i16_bytes: Vec<u8> = i16_table.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(format!("{}/distance_table_256x256.i16", outdir), &i16_bytes).ok();

    // Copy codebook index
    std::fs::copy("/tmp/codebooks/jina-v5-256/codebook_index.u16",
        format!("{}/codebook_index.u16", outdir)).ok();

    println!("\n[6] Saved to {}", outdir);
    println!("  f32: {} KB", f32_bytes.len() / 1024);
    println!("  i16: {} KB", i16_bytes.len() / 1024);

    // Quick benchmark
    println!("\n[7] Benchmark: softmax T=0.01 on semantic table:");
    let test_atoms: Vec<usize> = (0..20).map(|i| i * 13 % n_cent).collect();
    let inv_t = 100.0f32;
    let mut overlap_5 = 0;
    let mut unique_peaks = std::collections::HashSet::new();

    for &query in &test_atoms {
        let plain: Vec<usize> = {
            let mut idx: Vec<usize> = (0..n_cent).collect();
            idx.sort_by(|&a, &b| semantic_table[query * n_cent + b]
                .partial_cmp(&semantic_table[query * n_cent + a]).unwrap());
            idx
        };

        let mut energy = vec![0.0f32; n_cent];
        energy[query] = 1.0;
        for _ in 0..10 {
            let mut next = vec![0.0f32; n_cent];
            for i in 0..n_cent {
                if energy[i] < 1e-15 { continue; }
                for j in 0..n_cent {
                    next[j] += semantic_table[i * n_cent + j] * energy[i];
                }
            }
            let max_e = next.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for e in &mut next { *e = ((*e - max_e) * inv_t).exp(); exp_sum += *e; }
            if exp_sum > 1e-10 { for e in &mut next { *e /= exp_sum; } }
            energy = next;
        }

        let mut think_idx: Vec<usize> = (0..n_cent).collect();
        think_idx.sort_by(|&a, &b| energy[b].partial_cmp(&energy[a]).unwrap());

        let p5: std::collections::HashSet<usize> = plain[..5].iter().copied().collect();
        let t5: std::collections::HashSet<usize> = think_idx[..5].iter().copied().collect();
        overlap_5 += p5.intersection(&t5).count();
        unique_peaks.insert(think_idx[0]);
    }

    println!("  top5={}/{} = {}%  peaks={}/{}",
        overlap_5, test_atoms.len() * 5,
        overlap_5 * 100 / (test_atoms.len() * 5),
        unique_peaks.len(), test_atoms.len());

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  SEMANTIC CODEBOOK COMPLETE");
    println!("  Token table: lexical only (ρ=0.54 vs semantic truth)");
    println!("  Semantic table: 256 forward passes through 28 layers");
    println!("  Spearman ρ(token, semantic) = {:.4}", rho);
    println!("  If ρ < 0.8: the semantic table captures NEW information");
    println!("═══════════════════════════════════════════════════════════");
}

#[cfg(not(feature = "calibration"))]
fn main() { eprintln!("Requires --features calibration"); }
