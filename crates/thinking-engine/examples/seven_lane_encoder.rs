//! 7-Lane HighHeelBGZ Encoder: safetensors → CLAM → 7 encoding lanes.
//!
//! Lanes:
//!   1. u8 CDF (percentile rank)
//!   2. i8 direct (round(cos × 127), signs preserved)
//!   3. u8 γ+φ (golden ratio redistribution)
//!   4. i8 γ+φ signed
//!   5. SiLU correction deltas (f32)
//!   6. BF16 direct (raw cosines as BF16 — StackedN source precision)
//!   7. Spiral reconstruction error (highheelbgz encode→decode drift)
//!
//! Works from safetensors (on-disk models). For GGUF, use stream_signed_lens.rs.
//!
//! Usage:
//!   cargo run --release --features calibration --example seven_lane_encoder \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- qwen3-vl-embedding
//!
//! Models: qwen3-vl-embedding | jina-v5 | jina-reranker-v3

#[cfg(feature = "calibration")]
fn main() {
    use rayon::prelude::*;

    const N_CENTROIDS: usize = 256;

    let args: Vec<String> = std::env::args().collect();
    let model_key = args.get(1).map(|s| s.as_str()).unwrap_or("qwen3-vl-embedding");

    let (safetensors_path, embed_prefix, hidden_dim_expected) = match model_key {
        "qwen3-vl-embedding" => (
            "crates/thinking-engine/data/qwen3-vl-embedding/model.safetensors",
            "model.language_model.embed_tokens.weight",
            2048usize,
        ),
        "jina-v5" => (
            "crates/thinking-engine/data/jina-v5-onnx/model.safetensors",
            "embed_tokens.weight",
            1024usize,
        ),
        "qwen3-tts" => (
            "/home/user/models/qwen3-tts-0.6b/model.safetensors",
            "talker.model.layers.0.self_attn.q_proj.weight",
            1024usize, // talker hidden_size (rows=2048 × cols=1024)
        ),
        other => {
            eprintln!("Unknown model: {}. Use: qwen3-vl-embedding | jina-v5 | qwen3-tts", other);
            return;
        }
    };

    println!("═══════════════════════════════════════════════════════════");
    println!("  7-LANE HIGHHEELBGZ ENCODER (safetensors)");
    println!("  Model: {}", model_key);
    println!("════════════════════��════════════════════════════════��═════\n");

    // Step 1: Load token embeddings from safetensors
    println!("[1] Loading safetensors...");
    let start = std::time::Instant::now();

    let file_data = std::fs::read(safetensors_path).unwrap_or_else(|e| {
        eprintln!("Cannot read {}: {}", safetensors_path, e);
        std::process::exit(1);
    });
    let tensors = safetensors::SafeTensors::deserialize(&file_data).unwrap_or_else(|e| {
        eprintln!("Cannot parse safetensors: {}", e);
        std::process::exit(1);
    });

    // Find the embedding tensor
    let embed_tensor = tensors.tensor(embed_prefix).unwrap_or_else(|_| {
        eprintln!("Tensor '{}' not found. Available:", embed_prefix);
        for name in tensors.names() {
            if name.contains("embed") {
                eprintln!("  {}", name);
            }
        }
        std::process::exit(1);
    });

    let shape = embed_tensor.shape();
    let vocab_size = shape[0];
    let hidden_dim = shape[1];
    println!("  Tensor: {} shape=[{}, {}] dtype={:?}",
        embed_prefix, vocab_size, hidden_dim, embed_tensor.dtype());
    assert_eq!(hidden_dim, hidden_dim_expected,
        "Hidden dim mismatch: got {} expected {}", hidden_dim, hidden_dim_expected);

    // Convert BF16/F32 → f32
    let embeddings: Vec<f32> = match embed_tensor.dtype() {
        safetensors::Dtype::BF16 => {
            embed_tensor.data().chunks_exact(2).map(|c| {
                f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)
            }).collect()
        }
        safetensors::Dtype::F32 => {
            embed_tensor.data().chunks_exact(4).map(|c| {
                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
            }).collect()
        }
        safetensors::Dtype::F16 => {
            embed_tensor.data().chunks_exact(2).map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                half::f16::from_bits(bits).to_f32()
            }).collect()
        }
        dt => {
            eprintln!("Unsupported dtype: {:?}", dt);
            std::process::exit(1);
        }
    };
    println!("  Loaded {} embeddings in {:.1}s", vocab_size, start.elapsed().as_secs_f64());

    // Step 2: Normalize
    println!("[2] Normalizing {} tokens × {} dims...", vocab_size, hidden_dim);
    let normed: Vec<Vec<f32>> = (0..vocab_size).map(|v| {
        let row = &embeddings[v * hidden_dim..(v + 1) * hidden_dim];
        let norm = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        if norm < 1e-12 { vec![0.0f32; hidden_dim] }
        else { let inv = (1.0 / norm) as f32; row.iter().map(|x| x * inv).collect() }
    }).collect();

    // Step 3: CLAM greedy centroid selection
    println!("[3] CLAM {} centroids...", N_CENTROIDS);
    let start = std::time::Instant::now();
    let mut selected = vec![0usize];
    let mut min_dist = vec![f64::INFINITY; vocab_size];
    for v in 0..vocab_size {
        let dot: f32 = normed[v].iter().zip(&normed[0]).map(|(a, b)| a * b).sum();
        min_dist[v] = 1.0 - dot as f64;
    }
    for _k in 1..N_CENTROIDS.min(vocab_size) {
        let next = min_dist.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        selected.push(next);
        for v in 0..vocab_size {
            let dot: f32 = normed[v].iter().zip(&normed[next]).map(|(a, b)| a * b).sum();
            let d = 1.0 - dot as f64;
            if d < min_dist[v] { min_dist[v] = d; }
        }
    }
    println!("  {} centroids in {:.1}s", selected.len(), start.elapsed().as_secs_f64());

    // Step 4: Assign tokens to centroids
    println!("[4] Assigning {} tokens...", vocab_size);
    let start = std::time::Instant::now();
    let centroid_vecs: Vec<&[f32]> = selected.iter().map(|&i| normed[i].as_slice()).collect();
    let assignments: Vec<u16> = (0..vocab_size).into_par_iter().map(|v| {
        let mut best = 0u16;
        let mut best_dot = f32::NEG_INFINITY;
        for (c, &cen) in centroid_vecs.iter().enumerate() {
            let dot: f32 = normed[v].iter().zip(cen).map(|(a, b)| a * b).sum();
            if dot > best_dot { best_dot = dot; best = c as u16; }
        }
        best
    }).collect();
    println!("  Done in {:.1}s", start.elapsed().as_secs_f64());

    // Step 5: Average centroids + pairwise cosines
    let n_cent = selected.len();
    let mut sums = vec![vec![0.0f64; hidden_dim]; n_cent];
    let mut counts = vec![0u32; n_cent];
    for (v, &c) in assignments.iter().enumerate() {
        counts[c as usize] += 1;
        let row = &embeddings[v * hidden_dim..(v + 1) * hidden_dim];
        for d in 0..hidden_dim { sums[c as usize][d] += row[d] as f64; }
    }
    let centroids_avg: Vec<Vec<f32>> = (0..n_cent).map(|c| {
        if counts[c] == 0 { return vec![0.0f32; hidden_dim]; }
        let n = counts[c] as f64;
        let avg: Vec<f32> = sums[c].iter().map(|&s| (s / n) as f32).collect();
        let norm = avg.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
        if norm < 1e-12 { vec![0.0f32; hidden_dim] }
        else { let inv = (1.0 / norm) as f32; avg.iter().map(|v| v * inv).collect() }
    }).collect();

    let mut all_cos: Vec<f32> = Vec::new();
    let mut raw_cos = vec![0.0f32; n_cent * n_cent];
    for i in 0..n_cent {
        raw_cos[i * n_cent + i] = 1.0;
        for j in (i+1)..n_cent {
            let dot: f32 = centroids_avg[i].iter().zip(&centroids_avg[j]).map(|(a, b)| a * b).sum();
            let cos = dot.clamp(-1.0, 1.0);
            raw_cos[i * n_cent + j] = cos;
            raw_cos[j * n_cent + i] = cos;
            all_cos.push(cos);
        }
    }
    all_cos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cos_min = all_cos.first().copied().unwrap_or(0.0);
    let cos_max = all_cos.last().copied().unwrap_or(1.0);
    let cos_mean: f32 = all_cos.iter().sum::<f32>() / all_cos.len().max(1) as f32;
    println!("[5] Cosine: [{:.4}, {:.4}] mean={:.4}", cos_min, cos_max, cos_mean);

    // ═══ LANE 1: u8 CDF (percentile rank) ═══
    let mut lane1_u8 = vec![0u8; n_cent * n_cent];
    for i in 0..n_cent {
        lane1_u8[i * n_cent + i] = 255;
        for j in (i+1)..n_cent {
            let cos = raw_cos[i * n_cent + j];
            let rank = all_cos.partition_point(|&c| c <= cos);
            let pct = rank as f32 / all_cos.len() as f32;
            let u = (pct * 254.0).round().clamp(0.0, 254.0) as u8;
            lane1_u8[i * n_cent + j] = u;
            lane1_u8[j * n_cent + i] = u;
        }
    }

    // ═══ LANE 2: i8 direct (round(cos × 127)) ═══
    let mut lane2_i8 = vec![0i8; n_cent * n_cent];
    let mut pos_count = 0usize;
    let mut neg_count = 0usize;
    let mut zero_count = 0usize;
    for i in 0..n_cent {
        lane2_i8[i * n_cent + i] = 127;
        for j in (i+1)..n_cent {
            let cos = raw_cos[i * n_cent + j];
            let val = (cos * 127.0).round().clamp(-128.0, 127.0) as i8;
            lane2_i8[i * n_cent + j] = val;
            lane2_i8[j * n_cent + i] = val;
            if val > 0 { pos_count += 1; }
            else if val < 0 { neg_count += 1; }
            else { zero_count += 1; }
        }
    }
    let total_pairs = pos_count + neg_count + zero_count;
    let ei_ratio = if total_pairs > 0 { pos_count as f32 / total_pairs as f32 } else { 0.5 };

    // ═══ LANE 3: u8 γ+φ (golden ratio redistribution) ═══
    let cos_abs_mean: f32 = all_cos.iter().map(|c| c.abs()).sum::<f32>() / all_cos.len().max(1) as f32;
    let cos_abs_max: f32 = all_cos.iter().map(|c| c.abs()).fold(0.0f32, f32::max);
    let role_gamma = cos_abs_mean;
    let phi_scale = cos_abs_max.max(0.01);

    let mut gp_encoded: Vec<f32> = Vec::new();
    for i in 0..n_cent {
        for j in (i+1)..n_cent {
            let cos = raw_cos[i * n_cent + j];
            let sign = cos.signum();
            let mag = cos.abs();
            let gamma_enc = sign * (1.0 + mag / role_gamma.max(1e-8)).ln() * role_gamma;
            let normalized = gamma_enc.abs() / phi_scale.max(1e-8);
            let phi_log = (1.0 + normalized).ln() / (std::f64::consts::GOLDEN_RATIO as f32).ln();
            let gp_val = gamma_enc.signum() * phi_log * phi_scale;
            gp_encoded.push(gp_val);
        }
    }
    let mut gp_sorted = gp_encoded.clone();
    gp_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut lane3_gp_u8 = vec![0u8; n_cent * n_cent];
    let mut gp_idx = 0;
    for i in 0..n_cent {
        lane3_gp_u8[i * n_cent + i] = 255;
        for j in (i+1)..n_cent {
            let val = gp_encoded[gp_idx];
            let rank = gp_sorted.partition_point(|&c| c <= val);
            let pct = rank as f32 / gp_sorted.len() as f32;
            let u = (pct * 254.0).round().clamp(0.0, 254.0) as u8;
            lane3_gp_u8[i * n_cent + j] = u;
            lane3_gp_u8[j * n_cent + i] = u;
            gp_idx += 1;
        }
    }

    // ═══ LANE 4: i8 γ+φ signed ═══
    let mut lane4_gp_i8 = vec![0i8; n_cent * n_cent];
    gp_idx = 0;
    let max_gp = gp_sorted.last().copied().unwrap_or(1.0).abs()
        .max(gp_sorted.first().copied().unwrap_or(-1.0).abs())
        .max(0.01);
    for i in 0..n_cent {
        lane4_gp_i8[i * n_cent + i] = 127;
        for j in (i+1)..n_cent {
            let val = gp_encoded[gp_idx];
            let normalized = val / max_gp;
            let i8_val = (normalized * 127.0).round().clamp(-128.0, 127.0) as i8;
            lane4_gp_i8[i * n_cent + j] = i8_val;
            lane4_gp_i8[j * n_cent + i] = i8_val;
            gp_idx += 1;
        }
    }

    // ═══ LANE 5: SiLU correction deltas (zeros for token_embd) ═══
    let lane5_silu = vec![0.0f32; n_cent * n_cent];

    // ═══ LANE 6: BF16 direct (raw cosines stored as BF16, RNE) ═══
    //
    // RNE (round-to-nearest-even) via `ndarray::simd::f32_to_bf16_batch_rne`
    // from the pure AVX-512-F routine in commit c489d31 (byte-exact vs
    // hardware `_mm512_cvtneps_pbh` on 1M inputs). This replaces the older
    // per-element truncation shift (`(bits >> 16) as u16`) that drifts
    // ~1 ULP from the hardware path on ~50% of values and cannot serve as
    // the atomic-clock lab-BF16 lane.
    //
    // Structural change: the previous loop converted cosines one at a time
    // in a scalar hot loop. The new code passes the full `raw_cos` F32
    // matrix to a single batch call — the workspace-wide "never scalar
    // ever" rule for F32→BF16 mandates SIMD/AMX throughput (500-20000×
    // faster). See lance-graph/CLAUDE.md § Certification Process.
    let mut lane6_bf16 = vec![0u16; n_cent * n_cent];
    ndarray::simd::f32_to_bf16_batch_rne(&raw_cos, &mut lane6_bf16);

    // Roundtrip error measurement also runs via the batch primitive so the
    // whole lane stays SIMD-only. bf16_to_f32 is a trivial lossless shift,
    // so this is for reporting only (it is always zero ± 1 ULP from the
    // RNE truncation, which is the lane's defining loss).
    let mut lane6_roundtrip = vec![0.0f32; n_cent * n_cent];
    ndarray::simd::bf16_to_f32_batch(&lane6_bf16, &mut lane6_roundtrip);
    let bf16_max_err: f32 = raw_cos
        .iter()
        .zip(lane6_roundtrip.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // ═══ LANE 7: Spiral reconstruction error ═══
    // Encode each centroid average vector as a spiral walk, reconstruct, measure drift.
    // The drift per centroid pair = |cos(original) - cos(reconstructed)|
    // This quantifies how much the highheelbgz addressing loses.
    let mut lane7_spiral_drift = vec![0u8; n_cent * n_cent];
    let mut total_drift: f64 = 0.0;
    let mut max_drift: f32 = 0.0;

    // Simple spiral: sample every stride-th dimension, reconstruct via interpolation
    let spiral_stride = 11usize; // GOLDEN_STEP
    let reconstructed: Vec<Vec<f32>> = centroids_avg.iter().map(|centroid| {
        // Sample at golden stride
        let samples: Vec<(usize, f32)> = (0..hidden_dim)
            .step_by(spiral_stride)
            .map(|d| (d, centroid[d]))
            .collect();
        // Reconstruct by linear interpolation between samples
        let mut recon = vec![0.0f32; hidden_dim];
        for w in samples.windows(2) {
            let (d0, v0) = w[0];
            let (d1, v1) = w[1];
            for d in d0..=d1 {
                let t = (d - d0) as f32 / (d1 - d0).max(1) as f32;
                recon[d] = v0 * (1.0 - t) + v1 * t;
            }
        }
        // Fill tail (after last sample)
        if let Some(&(last_d, last_v)) = samples.last() {
            for d in last_d..hidden_dim {
                recon[d] = last_v;
            }
        }
        // Normalize reconstructed
        let norm = recon.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        if norm > 1e-12 {
            let inv = (1.0 / norm) as f32;
            for v in &mut recon { *v *= inv; }
        }
        recon
    }).collect();

    for i in 0..n_cent {
        lane7_spiral_drift[i * n_cent + i] = 0; // zero drift for self
        for j in (i+1)..n_cent {
            let cos_orig = raw_cos[i * n_cent + j];
            let cos_recon: f32 = reconstructed[i].iter().zip(&reconstructed[j])
                .map(|(a, b)| a * b).sum();
            let drift = (cos_orig - cos_recon).abs();
            // Scale drift to u8: 0=no drift, 255=max drift
            // Typical drift is < 0.1, so scale by 10× then clamp
            let u = (drift * 2550.0).round().clamp(0.0, 255.0) as u8;
            lane7_spiral_drift[i * n_cent + j] = u;
            lane7_spiral_drift[j * n_cent + i] = u;
            total_drift += drift as f64;
            if drift > max_drift { max_drift = drift; }
        }
    }
    let n_pairs = n_cent * (n_cent - 1) / 2;
    let avg_drift = total_drift / n_pairs.max(1) as f64;

    // ═══ PRINT SUMMARY ═══
    let t_avg = lane1_u8.iter().map(|&v| v as f64).sum::<f64>() / lane1_u8.len() as f64;
    println!("\n[6] 7-Lane Encoding Summary:");
    println!("  Lane 1: u8 CDF           {:>5} KB  avg={:.1}", lane1_u8.len() / 1024, t_avg);
    println!("  Lane 2: i8 direct        {:>5} KB  E/I={:.1}% pos={} neg={} zero={}",
        lane2_i8.len() / 1024, ei_ratio * 100.0, pos_count, neg_count, zero_count);
    println!("  Lane 3: u8 γ+φ           {:>5} KB  γ={:.4} φ={:.4}",
        lane3_gp_u8.len() / 1024, role_gamma, phi_scale);
    println!("  Lane 4: i8 γ+φ signed    {:>5} KB", lane4_gp_i8.len() / 1024);
    println!("  Lane 5: SiLU deltas      {:>5} KB  (zeros, token_embd)", lane5_silu.len() * 4 / 1024);
    println!("  Lane 6: BF16 direct      {:>5} KB  max_err={:.6}",
        lane6_bf16.len() * 2 / 1024, bf16_max_err);
    println!("  Lane 7: spiral drift     {:>5} KB  stride={} avg={:.4} max={:.4}",
        lane7_spiral_drift.len() / 1024, spiral_stride, avg_drift, max_drift);

    // ═══ SAVE ═══
    let model_name = model_key.replace("/", "-");
    let f32_bytes: Vec<u8> = raw_cos.iter().flat_map(|c| c.to_le_bytes()).collect();
    let silu_bytes: Vec<u8> = lane5_silu.iter().flat_map(|c| c.to_le_bytes()).collect();
    let bf16_bytes: Vec<u8> = lane6_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
    let idx_bytes: Vec<u8> = assignments.iter().flat_map(|&a| a.to_le_bytes()).collect();

    for dir in [
        format!("/tmp/codebooks/{}-7lane", model_name),
        format!("crates/thinking-engine/data/{}-7lane", model_name),
    ] {
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(format!("{}/distance_table_{}x{}.u8", dir, n_cent, n_cent), &lane1_u8).ok();
        std::fs::write(format!("{}/distance_table_{}x{}.i8", dir, n_cent, n_cent),
            unsafe { std::slice::from_raw_parts(lane2_i8.as_ptr() as *const u8, lane2_i8.len()) }).ok();
        std::fs::write(format!("{}/distance_table_{}x{}.gamma_phi.u8", dir, n_cent, n_cent), &lane3_gp_u8).ok();
        std::fs::write(format!("{}/distance_table_{}x{}.gamma_phi.i8", dir, n_cent, n_cent),
            unsafe { std::slice::from_raw_parts(lane4_gp_i8.as_ptr() as *const u8, lane4_gp_i8.len()) }).ok();
        std::fs::write(format!("{}/silu_deltas_{}x{}.f32", dir, n_cent, n_cent), &silu_bytes).ok();
        std::fs::write(format!("{}/distance_table_{}x{}.bf16", dir, n_cent, n_cent), &bf16_bytes).ok();
        std::fs::write(format!("{}/spiral_drift_{}x{}.u8", dir, n_cent, n_cent), &lane7_spiral_drift).ok();
        std::fs::write(format!("{}/cosine_matrix_{}x{}.f32", dir, n_cent, n_cent), &f32_bytes).ok();
        std::fs::write(format!("{}/codebook_index.u16", dir), &idx_bytes).ok();

        let metadata = serde_json::json!({
            "model": model_name,
            "source": "safetensors",
            "source_path": safetensors_path,
            "n_centroids": n_cent,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "cosine_range": [cos_min, cos_max],
            "cosine_mean": cos_mean,
            "encoding_lanes": {
                "lane_1_u8_cdf": { "file": format!("distance_table_{}x{}.u8", n_cent, n_cent), "encoding": "CDF_percentile" },
                "lane_2_i8_direct": { "file": format!("distance_table_{}x{}.i8", n_cent, n_cent), "encoding": "round(cos*127)", "scale": 127.0 },
                "lane_3_u8_gamma_phi": { "file": format!("distance_table_{}x{}.gamma_phi.u8", n_cent, n_cent), "role_gamma": role_gamma, "phi_scale": phi_scale },
                "lane_4_i8_gamma_phi_signed": { "file": format!("distance_table_{}x{}.gamma_phi.i8", n_cent, n_cent), "role_gamma": role_gamma, "phi_scale": phi_scale },
                "lane_5_silu_correction": { "file": format!("silu_deltas_{}x{}.f32", n_cent, n_cent), "note": "zeros (token_embd, no gate)" },
                "lane_6_bf16_direct": { "file": format!("distance_table_{}x{}.bf16", n_cent, n_cent), "max_roundtrip_error": bf16_max_err },
                "lane_7_spiral_drift": { "file": format!("spiral_drift_{}x{}.u8", n_cent, n_cent), "stride": spiral_stride, "avg_drift": avg_drift, "max_drift": max_drift },
            },
            "ei_ratio": ei_ratio,
            "positive_pairs": pos_count,
            "negative_pairs": neg_count,
            "zero_pairs": zero_count,
        });
        std::fs::write(format!("{}/encoding_metadata.json", dir),
            serde_json::to_string_pretty(&metadata).unwrap()).ok();
    }

    println!("\n[7] Saved to /tmp/codebooks/{}-7lane/", model_name);
    println!("═══════════════════════════════════════════════════════════");
}

// Local scalar `f32_to_bf16` and `bf16_to_f32` functions removed — the
// Lane 6 BF16 derivation now uses the batch RNE primitives from
// `ndarray::simd::{f32_to_bf16_batch_rne, bf16_to_f32_batch}` (c489d31)
// per the workspace-wide "never scalar ever" rule for F32→BF16.
// See lance-graph/CLAUDE.md § Certification Process.

#[cfg(not(feature = "calibration"))]
fn main() { eprintln!("Requires --features calibration"); }
