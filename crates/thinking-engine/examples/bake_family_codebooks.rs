//! Bake family codebooks — multipass generator with RaBitQ first-pass
//! bucketing, for Qwen3-0.6B (and any compatible Qwen3 / Jina v5 family).
//!
//! **MVP v1 scope** (this file):
//!   - Single role: `model.embed_tokens.weight` from Qwen3-0.6B safetensors
//!   - Flat K-centroid codebook (no hierarchy yet)
//!   - K selectable via CLI: 256 (L0+L1 depth truncated) or 4096 (full
//!     depth, precision tree target for bgz-hhtl-d Slot D)
//!   - RaBitQ first-pass FPS (furthest-point sampling in Hamming space)
//!     via `bgz17::rabitq_compat` for O(N × D/64) initialization
//!   - K-means refinement in F32 space for 3 iterations (convergence
//!     threshold: < 0.1% centroid movement stops early)
//!   - Writes: `data/qwen3-0.6b-family-{role}-{K}/` with a new JSON
//!     metadata file + the K × D centroid matrix as `centroids.f32`
//!
//! **NOT yet implemented** (follow-up MVPs v2/v3):
//!   - v2: all 8 role families (Embed, Q, K, V, O, Gate, Up, Down)
//!         × 28 layers = ~198 tensors per model
//!   - v3: 16-way hierarchical split matching the Slot D descriptor:
//!           L0  16  HEEL
//!           L1  256 HIP   (16 per HEEL)
//!           L2  4096 TWIG (16 per HIP)
//!           flags 4 bits (polarity + γ-phase)
//!   - Post-bgz-hhtl-d: precision tree bucket optimization + CLAM
//!     cycloid table integration
//!
//! Run:
//! ```sh
//! cargo run --release --features calibration --example bake_family_codebooks \
//!     --manifest-path crates/thinking-engine/Cargo.toml -- 256
//! ```
//!
//! Or for the 4096-centroid run:
//! ```sh
//! cargo run --release --features calibration --example bake_family_codebooks \
//!     --manifest-path crates/thinking-engine/Cargo.toml -- 4096
//! ```

#[cfg(not(feature = "calibration"))]
fn main() { eprintln!("Requires --features calibration"); }

#[cfg(feature = "calibration")]
fn main() {
    use bgz17::rabitq_compat::{OrthogonalMatrix, RaBitQEncoding};
    use bgz17::palette::Palette;
    use rayon::prelude::*;

    // ── Configuration ──
    const SAFETENSORS_PATH: &str = "/home/user/models/qwen3-0.6b.safetensors";
    const TENSOR_NAME: &str = "model.embed_tokens.weight";
    const EXPECTED_VOCAB: usize = 151936;
    const EXPECTED_DIM: usize = 1024;
    const N_REFINE_ITERS: usize = 3;
    const CONVERGENCE_EPS: f64 = 0.001; // 0.1% mean centroid movement

    // Parse K from argv (default 256).
    let args: Vec<String> = std::env::args().collect();
    let k: usize = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    println!("═══════════════════════════════════════════════════════════");
    println!("  BAKE FAMILY CODEBOOKS — MVP v1");
    println!("  Source : {}", SAFETENSORS_PATH);
    println!("  Tensor : {}", TENSOR_NAME);
    println!("  K      : {} centroids", k);
    println!("  Algo   : RaBitQ FPS init + F32 k-means refinement ({} iters)", N_REFINE_ITERS);
    println!("═══════════════════════════════════════════════════════════\n");

    // ── Step 1: Load safetensors via the upstream crate ──
    println!("[1] Loading safetensors from {}", SAFETENSORS_PATH);
    let t_start = std::time::Instant::now();
    let bytes = std::fs::read(SAFETENSORS_PATH).unwrap_or_else(|e| {
        eprintln!("  FAILED to read {}: {}", SAFETENSORS_PATH, e);
        std::process::exit(1);
    });
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap_or_else(|e| {
        eprintln!("  FAILED to parse safetensors: {}", e);
        std::process::exit(1);
    });
    let embed = tensors.tensor(TENSOR_NAME).unwrap_or_else(|e| {
        eprintln!("  Tensor {} not found: {}", TENSOR_NAME, e);
        eprintln!("  Available tensors (first 20):");
        for name in tensors.names().iter().take(20) {
            eprintln!("    {}", name);
        }
        std::process::exit(1);
    });
    let shape = embed.shape();
    let vocab = shape[0];
    let dim = shape[1];
    println!(
        "  {} shape=[{}, {}] dtype={:?} ({:.2} MB) loaded in {:.1}s",
        TENSOR_NAME, vocab, dim, embed.dtype(),
        bytes.len() as f64 / 1e6,
        t_start.elapsed().as_secs_f64()
    );
    assert_eq!(vocab, EXPECTED_VOCAB, "vocab size mismatch");
    assert_eq!(dim, EXPECTED_DIM, "hidden dim mismatch");
    assert_eq!(embed.dtype(), safetensors::Dtype::BF16, "expected BF16 source");

    // ── Step 2: Upcast BF16 → F32 in one batch pass ──
    // This is the only materialized F32 buffer in the bake (vocab × dim × 4
    // bytes ≈ 620 MB for Jina-v5-sized embeddings). Acceptable for a
    // one-time bake run; the certification pipeline then operates on
    // this F32 view without re-materializing.
    println!("\n[2] Upcasting BF16 → F32 ({} elements)", vocab * dim);
    let t_upcast = std::time::Instant::now();
    let mut embeddings: Vec<f32> = Vec::with_capacity(vocab * dim);
    for chunk in embed.data().chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        embeddings.push(f32::from_bits((bits as u32) << 16));
    }
    println!(
        "  {:.2} MB F32 materialized in {:.2}s",
        embeddings.len() as f64 * 4.0 / 1e6,
        t_upcast.elapsed().as_secs_f64()
    );

    // NaN scan stage 1: source bytes post-upcast.
    let nan_in_source = embeddings.iter().filter(|v| v.is_nan()).count();
    if nan_in_source > 0 {
        eprintln!("  NaN in source tensor: {} values. Halting.", nan_in_source);
        std::process::exit(2);
    }

    // ── Step 3: Normalize each row to unit length ──
    // RaBitQ requires unit-normalized vectors. Norms are kept per-row
    // for the unbiased distance reconstruction in the post-encode phase.
    println!("\n[3] Normalizing {} rows to unit length", vocab);
    let t_norm = std::time::Instant::now();
    let mut norms = vec![0.0f32; vocab];
    for v in 0..vocab {
        let row = &mut embeddings[v * dim..(v + 1) * dim];
        let norm = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt() as f32;
        norms[v] = norm;
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for x in row.iter_mut() { *x *= inv; }
        }
    }
    let nonzero_norms = norms.iter().filter(|&&n| n > 1e-12).count();
    println!(
        "  {} non-zero norms of {} total in {:.2}s",
        nonzero_norms, vocab, t_norm.elapsed().as_secs_f64()
    );

    // ── Step 4: RaBitQ encode every row (first-pass binary signature) ──
    // Uses bgz17::rabitq_compat with a Walsh-Hadamard-initialized
    // orthogonal matrix. The encode's rotate() inner loop is currently
    // scalar — this is a known optimization target (see the
    // "SIMD-optimize rabitq rotate" todo entry). For a one-time bake of
    // a 151K × 1024 embedding matrix, scalar runs in ~30s on our
    // hardware; acceptable for v1, will be SIMD-replaced in v2.
    println!("\n[4] RaBitQ binary-encoding {} rows (D = {})", vocab, dim);
    let t_rabitq = std::time::Instant::now();
    let rotation = OrthogonalMatrix::hadamard(dim);
    let empty_palette = Palette { entries: vec![] };
    let normalized_rows: Vec<&[f32]> = (0..vocab)
        .map(|v| &embeddings[v * dim..(v + 1) * dim])
        .collect();
    let rabitq_encodings: Vec<RaBitQEncoding> = normalized_rows
        .par_iter()
        .map(|row| RaBitQEncoding::encode(row, &rotation, &empty_palette))
        .collect();
    println!(
        "  {} RaBitQ codes ({} × u64 words per code) in {:.1}s",
        rabitq_encodings.len(),
        rabitq_encodings[0].binary.len(),
        t_rabitq.elapsed().as_secs_f64()
    );

    // ── Step 5: FPS in Hamming space → K seed centroids ──
    // Starts from token 0, iteratively picks the row with the maximum
    // Hamming distance to the nearest already-selected centroid. This
    // is the RaBitQ-based equivalent of CLAM greedy and gives a
    // well-spread K-point cover of the binary-quantized manifold.
    println!("\n[5] FPS in Hamming space → {} seed centroids", k);
    let t_fps = std::time::Instant::now();
    let mut selected: Vec<usize> = Vec::with_capacity(k);
    selected.push(0);
    let mut min_hamming: Vec<u32> = vec![u32::MAX; vocab];
    for v in 0..vocab {
        min_hamming[v] = rabitq_encodings[v].hamming_distance(&rabitq_encodings[0]);
    }
    while selected.len() < k {
        // Find row with maximum min_hamming distance to any selected centroid.
        let (next_idx, _max_d) = min_hamming
            .iter()
            .enumerate()
            .max_by_key(|&(_, &d)| d)
            .unwrap();
        selected.push(next_idx);
        // Update min_hamming: for each row, if Hamming(row, new_centroid) <
        // current min_hamming, replace it.
        let new_enc = &rabitq_encodings[next_idx];
        min_hamming
            .par_iter_mut()
            .zip(rabitq_encodings.par_iter())
            .for_each(|(md, enc)| {
                let d = enc.hamming_distance(new_enc);
                if d < *md { *md = d; }
            });
    }
    println!("  {} centroids picked in {:.1}s", selected.len(), t_fps.elapsed().as_secs_f64());

    // ── Step 6: Initial assignment (nearest Hamming centroid) ──
    println!("\n[6] Initial bucket assignment by Hamming distance");
    let t_assign = std::time::Instant::now();
    let centroid_encs: Vec<&RaBitQEncoding> = selected.iter().map(|&i| &rabitq_encodings[i]).collect();
    let mut assignments: Vec<u16> = (0..vocab)
        .into_par_iter()
        .map(|v| {
            let mut best = 0u16;
            let mut best_d = u32::MAX;
            for (c, enc) in centroid_encs.iter().enumerate() {
                let d = rabitq_encodings[v].hamming_distance(enc);
                if d < best_d { best_d = d; best = c as u16; }
            }
            best
        })
        .collect();
    println!("  Done in {:.1}s", t_assign.elapsed().as_secs_f64());

    // ── Step 7: K-means refinement in F32 space ──
    // Iteratively: compute F32 mean per bucket → reassign rows by
    // cosine similarity to the new means → repeat until convergence
    // or N_REFINE_ITERS exhausted.
    println!("\n[7] F32 k-means refinement (max {} iterations, ε = {:.4})",
        N_REFINE_ITERS, CONVERGENCE_EPS);
    let mut centroids: Vec<Vec<f32>> = compute_centroid_means(&embeddings, &assignments, k, dim);
    for iter in 0..N_REFINE_ITERS {
        let t_iter = std::time::Instant::now();
        // Reassign by cosine to the new centroids.
        let new_assignments: Vec<u16> = (0..vocab)
            .into_par_iter()
            .map(|v| {
                let row = &embeddings[v * dim..(v + 1) * dim];
                let mut best = 0u16;
                let mut best_sim = f32::NEG_INFINITY;
                for (c, cent) in centroids.iter().enumerate() {
                    let dot: f32 = row.iter().zip(cent.iter()).map(|(a, b)| a * b).sum();
                    if dot > best_sim { best_sim = dot; best = c as u16; }
                }
                best
            })
            .collect();

        // Compute how many assignments changed (convergence measure).
        let changed = assignments
            .iter()
            .zip(new_assignments.iter())
            .filter(|(a, b)| a != b)
            .count();
        let change_rate = changed as f64 / vocab as f64;

        assignments = new_assignments;
        let new_centroids = compute_centroid_means(&embeddings, &assignments, k, dim);

        // Measure mean centroid movement.
        let mut total_move = 0.0f64;
        for (old, new) in centroids.iter().zip(new_centroids.iter()) {
            let move_dist: f64 = old
                .iter()
                .zip(new.iter())
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            total_move += move_dist;
        }
        let mean_move = total_move / k as f64;
        centroids = new_centroids;

        println!(
            "  iter {}: {:.2}% reassigned, mean centroid Δ = {:.5}, time {:.1}s",
            iter + 1,
            change_rate * 100.0,
            mean_move,
            t_iter.elapsed().as_secs_f64()
        );

        if mean_move < CONVERGENCE_EPS {
            println!("  Converged at iter {} (mean Δ < {:.4})", iter + 1, CONVERGENCE_EPS);
            break;
        }
    }

    // ── Step 8: Final bucket statistics ──
    let mut bucket_counts = vec![0u32; k];
    for &a in &assignments { bucket_counts[a as usize] += 1; }
    let min_bucket = *bucket_counts.iter().min().unwrap_or(&0);
    let max_bucket = *bucket_counts.iter().max().unwrap_or(&0);
    let mean_bucket = vocab as f64 / k as f64;
    let var: f64 = bucket_counts
        .iter()
        .map(|&c| (c as f64 - mean_bucket).powi(2))
        .sum::<f64>() / k as f64;
    let std_bucket = var.sqrt();
    let balance_ratio = max_bucket as f64 / mean_bucket;

    println!("\n[8] Bucket statistics:");
    println!("  K = {}, vocab = {}, mean = {:.1}", k, vocab, mean_bucket);
    println!("  min = {}, max = {}, std = {:.1}", min_bucket, max_bucket, std_bucket);
    println!("  balance ratio (max / mean) = {:.2}", balance_ratio);
    if balance_ratio > 5.0 {
        eprintln!("  ⚠ balance ratio > 5 — buckets are highly uneven; tree hierarchy may fail");
    }

    // ── Step 9: Save outputs ──
    let out_dir = format!(
        "crates/thinking-engine/data/qwen3-0.6b-family-embed-{}", k
    );
    std::fs::create_dir_all(&out_dir).ok();

    let centroid_bytes: Vec<u8> = centroids
        .iter()
        .flat_map(|c| c.iter().flat_map(|x| x.to_le_bytes()))
        .collect();
    std::fs::write(
        format!("{}/centroids_f32.bin", out_dir),
        &centroid_bytes,
    ).unwrap();

    let assignments_bytes: Vec<u8> = assignments
        .iter()
        .flat_map(|a| a.to_le_bytes())
        .collect();
    std::fs::write(
        format!("{}/assignments_u16.bin", out_dir),
        &assignments_bytes,
    ).unwrap();

    let metadata = serde_json::json!({
        "model": "qwen3-0.6b",
        "tensor": TENSOR_NAME,
        "vocab_size": vocab,
        "hidden_dim": dim,
        "k_centroids": k,
        "algorithm": {
            "first_pass": "rabitq_hamming_fps",
            "rotation": "hadamard",
            "refinement": "f32_kmeans",
            "iterations": N_REFINE_ITERS,
            "convergence_eps": CONVERGENCE_EPS,
        },
        "bucket_statistics": {
            "min": min_bucket,
            "max": max_bucket,
            "mean": round2(mean_bucket),
            "std": round2(std_bucket),
            "balance_ratio": round2(balance_ratio),
        },
        "files": {
            "centroids": format!("centroids_f32.bin  ({} × {} × f32 = {} bytes)",
                k, dim, centroid_bytes.len()),
            "assignments": format!("assignments_u16.bin  ({} × u16 = {} bytes)",
                vocab, assignments_bytes.len()),
        },
        "provenance": {
            "branch": "claude/risc-thought-engine-TCZw7",
            "mvp_version": "v1",
            "source_file": SAFETENSORS_PATH,
            "todo": "MVP v2: extend to all 8 roles × 28 layers. MVP v3: 16-way hierarchical Slot D.",
        },
    });
    std::fs::write(
        format!("{}/metadata.json", out_dir),
        serde_json::to_string_pretty(&metadata).unwrap(),
    ).unwrap();

    println!("\n[9] Saved to {}", out_dir);
    println!("═══════════════════════════════════════════════════════════");
    println!("  BAKE COMPLETE ({} centroids)", k);
    println!("═══════════════════════════════════════════════════════════");
}

/// Compute the F32 mean of each bucket, re-normalize to unit length.
/// Returns K × D centroids.
#[cfg(feature = "calibration")]
fn compute_centroid_means(
    embeddings: &[f32],
    assignments: &[u16],
    k: usize,
    dim: usize,
) -> Vec<Vec<f32>> {
    let vocab = assignments.len();
    let mut sums = vec![vec![0.0f64; dim]; k];
    let mut counts = vec![0u32; k];
    for (v, &c) in assignments.iter().enumerate() {
        let bucket = c as usize;
        counts[bucket] += 1;
        let row = &embeddings[v * dim..(v + 1) * dim];
        for d in 0..dim { sums[bucket][d] += row[d] as f64; }
    }
    (0..k)
        .map(|c| {
            if counts[c] == 0 {
                return vec![0.0f32; dim];
            }
            let n = counts[c] as f64;
            let mut centroid: Vec<f32> = sums[c].iter().map(|&s| (s / n) as f32).collect();
            let norm: f32 = centroid
                .iter()
                .map(|x| (*x as f64) * (*x as f64))
                .sum::<f64>()
                .sqrt() as f32;
            if norm > 1e-12 {
                let inv = 1.0 / norm;
                for x in centroid.iter_mut() { *x *= inv; }
            }
            centroid
        })
        .collect()
}

#[cfg(feature = "calibration")]
fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
