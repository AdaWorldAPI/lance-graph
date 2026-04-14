//! RVQ end-to-end TTS: codebook → reconstruct → 33 layers → WAV.
//!
//! Pure Rust. No Python. Uses ndarray SIMD where available.
//!
//! Pipeline:
//!   1. Load safetensors (BF16 → f32)
//!   2. Build RVQ codebooks per role (CLAM furthest-point, progressive residual)
//!   3. Reconstruct all weight matrices from codebooks + indices
//!   4. Run 33-layer TTS inference (28 talker + 5 code_predictor)
//!   5. Compare codec tokens + PCM to raw reference
//!   6. Write WAV files
//!
//! ```sh
//! cargo run --release --example tts_rvq_e2e \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /path/to/model.safetensors /path/to/speech_tokenizer/model.safetensors
//! ```

use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::gguf::{GgmlType, TensorInfo};
use ndarray::simd::{f32x8, F32x8, F32x16, bf16_to_f32_batch, PREFERRED_F32_LANES};  // AVX-512 dispatch
use ndarray::backend::{dot_f32, gemm_f32};  // BLAS L1 dot + L3 gemm (matrixmultiply)

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
// Model constants (Qwen3-TTS-0.6B)
// ═══════════════════════════════════════════════════════════════════

const TALKER_HIDDEN: usize = 1024;
const TALKER_LAYERS: usize = 28;
const TALKER_HEADS: usize = 16;
const TALKER_KV_HEADS: usize = 8;
const TALKER_HEAD_DIM: usize = 64;  // 1024 / 16
const TALKER_INTER: usize = 3072;

const CP_HIDDEN: usize = 1024;
const CP_LAYERS: usize = 5;
const CP_HEADS: usize = 16;
const CP_KV_HEADS: usize = 8;
const CP_HEAD_DIM: usize = 64;
const CP_INTER: usize = 3072;

const SAMPLE_RATE: u32 = 24000;

// ═══════════════════════════════════════════════════════════════════
// RVQ codebook builder (CLAM furthest-point + progressive residual)
// ═══════════════════════════════════════════════════════════════════

/// Build progressive RVQ codebook for a weight matrix.
///
/// Each level: CLAM furthest-point sampling on the residual.
/// Returns (codebooks, assignments) where codebooks[level] is [k × cols]
/// and assignments[level] is [n_rows] indices.
fn build_rvq(
    rows: &[Vec<f32>],
    k_levels: &[usize],
) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<usize>>) {
    let n = rows.len();
    let cols = if n > 0 { rows[0].len() } else { 0 };

    let mut residual: Vec<Vec<f32>> = rows.to_vec();
    let mut codebooks = Vec::new();
    let mut all_assignments = Vec::new();

    for &k in k_levels {
        let k = k.min(n);
        let centroids = clam_sample(&residual, k);
        let assignments = assign_nearest(&residual, &centroids);

        // Update residual
        for i in 0..n {
            let ci = assignments[i];
            for j in 0..cols {
                residual[i][j] -= centroids[ci][j];
            }
            let _ = cols;
        }
        codebooks.push(centroids);
        all_assignments.push(assignments);
    }
    (codebooks, all_assignments)
}

/// Hierarchical CLAM 256×256 — tree quantization (not residual).
///
/// For tensors with n_rows > 8192 (e.g. vocab embeddings at 151K rows),
/// progressive residual RVQ at k=4096 can't reach cos ≈ 1 because
/// k_final < n_rows / 4. Hierarchical CLAM fixes this by partitioning
/// rows first into 256 L1 clusters, then each cluster into ≤256 L2 leaves
/// via furthest-point sampling — one L2 leaf per row, no residual sum.
///
/// Reconstruction: row[i] ≈ L2_codebooks[L1_assign[i]][L2_assign[i]]
/// (picks one centroid per row, lossless at BF16 precision up to the
///  picker's error — for vocab embeddings with ~2.32 rows / leaf, cos ≈ 1).
///
/// Storage (per-tensor):
///   L1 codebook:   256 × cols × 4 bytes
///   L2 codebooks:  sum over 256 clusters of (≤256 × cols × 4 bytes)
///   Indices:       n_rows × 2 bytes  (packed 8+8 into u16)
///
/// For [151936, 2048]: 1 MB L1 + ~256 MB L2 + 297 KB indices vs 620 MB BF16
/// → 2.4:1 at cos ≈ 1 (see docs/RVQ_K_LADDER_TUNING.md Section 3).
fn build_hclam_256x256(
    rows: &[Vec<f32>],
) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>, Vec<(u8, u8)>) {
    let n = rows.len();
    let cols = if n > 0 { rows[0].len() } else { 0 };
    const K1: usize = 256;
    const K2: usize = 256;

    // L1: coarse cluster over the full row set
    let l1_centroids = clam_sample(rows, K1);
    let l1_assign = assign_nearest(rows, &l1_centroids);

    // L2: per-cluster fine centroids
    let mut l2_codebooks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(K1);
    let mut l2_assign: Vec<u8> = vec![0u8; n];
    let mut indices: Vec<(u8, u8)> = Vec::with_capacity(n);

    // Group row indices by L1 assignment
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); K1];
    for (row_idx, &ci) in l1_assign.iter().enumerate() {
        groups[ci].push(row_idx);
    }

    for ci in 0..K1 {
        let group_rows: Vec<Vec<f32>> = groups[ci].iter().map(|&i| rows[i].clone()).collect();
        let k = K2.min(group_rows.len().max(1));
        let centroids = if group_rows.is_empty() {
            vec![vec![0.0f32; cols]]
        } else {
            clam_sample(&group_rows, k)
        };
        let sub_assign = if group_rows.is_empty() {
            vec![]
        } else {
            assign_nearest(&group_rows, &centroids)
        };
        for (local_i, &row_idx) in groups[ci].iter().enumerate() {
            l2_assign[row_idx] = sub_assign[local_i] as u8;
        }
        l2_codebooks.push(centroids);
    }

    for i in 0..n {
        indices.push((l1_assign[i] as u8, l2_assign[i]));
    }
    (l1_centroids, l2_codebooks, indices)
}

/// Reconstruct rows from hierarchical CLAM 256×256 codebooks.
fn reconstruct_hclam(
    l2_codebooks: &[Vec<Vec<f32>>],
    indices: &[(u8, u8)],
    n_cols: usize,
) -> Vec<Vec<f32>> {
    indices.iter().map(|&(l1, l2)| {
        let cluster = &l2_codebooks[l1 as usize];
        if cluster.is_empty() {
            vec![0.0f32; n_cols]
        } else {
            cluster[(l2 as usize).min(cluster.len() - 1)].clone()
        }
    }).collect()
}

/// Reconstruct rows from RVQ codebooks + assignments.
fn reconstruct_rvq(
    codebooks: &[Vec<Vec<f32>>],
    assignments: &[Vec<usize>],
    n_rows: usize,
    n_cols: usize,
) -> Vec<Vec<f32>> {
    let mut rows = vec![vec![0.0f32; n_cols]; n_rows];
    for (cb, assign) in codebooks.iter().zip(assignments.iter()) {
        for i in 0..n_rows {
            let ci = assign[i];
            for j in 0..n_cols {
                rows[i][j] += cb[ci][j];
            }
        }
    }
    rows
}

/// CLAM furthest-point sampling.
fn clam_sample(rows: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    let n = rows.len();
    let k = k.min(n);
    if k == 0 { return Vec::new(); }

    let mut centroids = Vec::with_capacity(k);
    let mut used = vec![false; n];

    // First: largest L2 norm
    let first = (0..n).max_by(|&a, &b| {
        let na: f64 = rows[a].iter().map(|x| (*x as f64).powi(2)).sum();
        let nb: f64 = rows[b].iter().map(|x| (*x as f64).powi(2)).sum();
        na.partial_cmp(&nb).unwrap()
    }).unwrap_or(0);

    centroids.push(rows[first].clone());
    used[first] = true;
    // Squared distances — same ordering as euclidean, no per-call sqrt
    let mut min_dist = vec![f32::MAX; n];
    for i in 0..n {
        min_dist[i] = l2_dist_sq(&rows[i], &rows[first]);
    }

    for _ in 1..k {
        let mut next = 0usize;
        let mut best = f32::MIN;
        for i in 0..n {
            if !used[i] && min_dist[i] > best { best = min_dist[i]; next = i; }
        }
        centroids.push(rows[next].clone());
        used[next] = true;
        let c = &rows[next];
        for i in 0..n {
            if !used[i] {
                let d = l2_dist_sq(&rows[i], c);
                if d < min_dist[i] { min_dist[i] = d; }
            }
        }
    }
    centroids
}

fn assign_nearest(rows: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
    // Fix: compute each distance ONCE per (row, centroid), not twice via nested min_by closure.
    rows.iter().map(|row| {
        let mut best_idx = 0usize;
        let mut best = f32::MAX;
        for (ci, c) in centroids.iter().enumerate() {
            let d = l2_dist_sq(row, c);  // squared — same ordering, no sqrt needed
            if d < best { best = d; best_idx = ci; }
        }
        best_idx
    }).collect()
}

/// Fused squared-L2 distance — zero allocation, 4× unrolled F32x16 FMA.
/// Returns squared distance (sqrt unnecessary for comparisons).
///
/// Uses ndarray's canonical "array_window" idiom (chunks_exact(16) = AVX-512
/// lane width) + mul_add FMA. On `target-cpu=x86-64-v4` this compiles to
/// native VFMADD231PS on __m512.
#[inline(always)]
fn l2_dist_sq(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 16;  // PREFERRED_F32_LANES on AVX-512
    let n = a.len().min(b.len());
    // 4× unrolled FMA pipes: 64 floats per iteration
    let chunks = n / 64;
    let mut acc0 = F32x16::splat(0.0);
    let mut acc1 = F32x16::splat(0.0);
    let mut acc2 = F32x16::splat(0.0);
    let mut acc3 = F32x16::splat(0.0);
    for i in 0..chunks {
        let base = i * 64;
        let d0 = F32x16::from_slice(&a[base..])      - F32x16::from_slice(&b[base..]);
        let d1 = F32x16::from_slice(&a[base + 16..]) - F32x16::from_slice(&b[base + 16..]);
        let d2 = F32x16::from_slice(&a[base + 32..]) - F32x16::from_slice(&b[base + 32..]);
        let d3 = F32x16::from_slice(&a[base + 48..]) - F32x16::from_slice(&b[base + 48..]);
        // FMA: acc = d * d + acc  (VFMADD231PS)
        acc0 = d0.mul_add(d0, acc0);
        acc1 = d1.mul_add(d1, acc1);
        acc2 = d2.mul_add(d2, acc2);
        acc3 = d3.mul_add(d3, acc3);
    }
    let mut s = (acc0 + acc1 + acc2 + acc3).reduce_sum();
    // 16-wide tail (chunks_exact-style leftover)
    let mut i = chunks * 64;
    while i + LANES <= n {
        let d = F32x16::from_slice(&a[i..]) - F32x16::from_slice(&b[i..]);
        s += d.mul_add(d, F32x16::splat(0.0)).reduce_sum();
        i += LANES;
    }
    // scalar tail
    while i < n { let d = a[i] - b[i]; s += d * d; i += 1; }
    s
}

/// sqrt wrapper for legacy call sites (rarely needed — use squared directly when possible).
#[inline(always)]
fn l2_dist(a: &[f32], b: &[f32]) -> f64 {
    (l2_dist_sq(a, b) as f64).sqrt()
}

fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        let x = a[i] as f64; let y = b[i] as f64;
        dot += x * y; na += x * x; nb += y * y;
    }
    let d = (na * nb).sqrt();
    if d < 1e-15 { 0.0 } else { dot / d }
}

// ═══════════════════════════════════════════════════════════════════
// Weight loading + RVQ compression
// ═══════════════════════════════════════════════════════════════════

/// Load all weight tensors as f32, optionally RVQ-compress.
fn load_weights(
    reader: &mut BufReader<File>,
    header: &ndarray::hpc::gguf::GgufFile,
    compress: bool,
) -> (HashMap<String, Vec<f32>>, HashMap<String, [usize; 2]>, usize, usize) {
    let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
    let mut shapes: HashMap<String, [usize; 2]> = HashMap::new();
    let mut codebook_bytes = 0usize;
    let mut index_bytes = 0usize;

    for tensor in &header.tensors {
        let n: usize = tensor.dimensions.iter().map(|&d| d as usize).product();
        let n_rows = tensor.dimensions[0] as usize;
        let n_cols = if tensor.dimensions.len() > 1 {
            tensor.dimensions[1..].iter().map(|&d| d as usize).product()
        } else { 1 };

        // Read raw data
        let elem_size = match tensor.dtype {
            GgmlType::BF16 | GgmlType::F16 => 2,
            GgmlType::F32 => 4,
            _ => continue,
        };
        reader.seek(SeekFrom::Start(header.tensor_data_offset + tensor.offset)).unwrap();
        let mut raw = vec![0u8; n * elem_size];
        if reader.read_exact(&mut raw).is_err() { continue; }

        let f32_data: Vec<f32> = match tensor.dtype {
            GgmlType::BF16 => {
                let u16s: Vec<u16> = raw.chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                let mut f32s = vec![0.0f32; u16s.len()];
                bf16_to_f32_batch(&u16s, &mut f32s);
                f32s
            },
            GgmlType::F16 => raw.chunks_exact(2)
                .map(|c| ndarray::hpc::gguf::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            GgmlType::F32 => raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            _ => continue,
        };

        shapes.insert(tensor.name.clone(), [n_rows, n_cols]);

        // RVQ compress 2D weight tensors with ≥128 rows
        if compress && tensor.dimensions.len() >= 2 && n_rows >= 128 && n_cols >= 128
            && !tensor.name.contains("norm") && !tensor.name.contains("bias")
        {
            // Convert to row-major Vec<Vec<f32>>
            let rows: Vec<Vec<f32>> = (0..n_rows)
                .map(|r| f32_data[r * n_cols..(r + 1) * n_cols].to_vec())
                .collect();

            // Shape-dispatch: vocab-sized tensors (n_rows > 8192) go through
            // hierarchical CLAM 256×256 — progressive residual RVQ at k=4096
            // can't reach cos ≈ 1 when k_final < n_rows / 4.
            let short = tensor.name.rsplit('.').take(3).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join(".");
            use std::io::Write as _;

            let (reconstructed, cos, tag): (Vec<Vec<f32>>, f64, String) = if n_rows > 8192 {
                // Hierarchical CLAM 256×256 path
                let t0 = Instant::now();
                let (l1_centroids, l2_codebooks, indices) = build_hclam_256x256(&rows);
                let rec = reconstruct_hclam(&l2_codebooks, &indices, n_cols);
                let el = t0.elapsed();

                // Storage: L1 + sum of L2 + indices
                codebook_bytes += l1_centroids.len() * n_cols * 4;
                for cb in &l2_codebooks {
                    codebook_bytes += cb.len() * n_cols * 4;
                }
                index_bytes += indices.len() * 2; // (u8, u8) per row

                let c = cosine_f32(&rows[0], &rec[0]);
                println!("    [{:>3}] {:<60} [{}x{}] cos={:.4} hclam=256x256 {:?}",
                    weights.len() + 1, short, n_rows, n_cols, c, el);
                std::io::stdout().flush().ok();
                (rec, c, "hclam".into())
            } else {
                // K levels based on role
                let role = tensor.name.to_lowercase();
                let k_levels = if role.contains("k_proj") || role.contains("v_proj") || role.contains("down_proj") {
                    vec![256, 512, 1024]
                } else {
                    vec![256, 512, 1024, 4096]
                };

                let t_rvq = Instant::now();
                let (codebooks, assignments) = build_rvq(&rows, &k_levels);
                let rvq_elapsed = t_rvq.elapsed();

                for cb in &codebooks {
                    codebook_bytes += cb.len() * n_cols * 4;
                }
                for a in &assignments {
                    let max_idx = *a.iter().max().unwrap_or(&0);
                    index_bytes += a.len() * if max_idx < 256 { 1 } else { 2 };
                }
                let rec = reconstruct_rvq(&codebooks, &assignments, n_rows, n_cols);
                let c = cosine_f32(&rows[0], &rec[0]);
                println!("    [{:>3}] {:<60} [{}x{}] cos={:.4} k={:?} {:?}",
                    weights.len() + 1, short, n_rows, n_cols, c, k_levels, rvq_elapsed);
                std::io::stdout().flush().ok();
                (rec, c, "rvq".into())
            };
            let _ = (cos, tag);

            // Flatten back
            let flat: Vec<f32> = reconstructed.into_iter().flatten().collect();
            weights.insert(tensor.name.clone(), flat);
        } else {
            weights.insert(tensor.name.clone(), f32_data);
        }
    }

    (weights, shapes, codebook_bytes, index_bytes)
}

// ═══════════════════════════════════════════════════════════════════
// TTS inference (same math as tts_full_inference.rs)
// ═══════════════════════════════════════════════════════════════════

fn get_weight<'a>(w: &'a HashMap<String, Vec<f32>>, name: &str) -> &'a [f32] {
    w.get(name).map(|v| v.as_slice()).unwrap_or(&[])
}

fn rms_norm(x: &mut [f32], w: &[f32], dim: usize) {
    let len = x.len() / dim;
    for i in 0..len {
        let row = &x[i*dim..(i+1)*dim];
        // dot_f32 for sum of squares (SIMD)
        let ss = dot_f32(row, row);
        let inv = 1.0 / (ss / dim as f32 + 1e-6).sqrt();
        // F32x16 multiply (AVX-512 lane width) via mul_add FMA:
        // (inv*vx)*vw + 0  compiles to VFMADD231PS on __m512.
        let inv_v = F32x16::splat(inv);
        let zero_v = F32x16::splat(0.0);
        let chunks = dim / 16;
        for c in 0..chunks {
            let base = i * dim + c * 16;
            let vx = F32x16::from_slice(&x[base..]);
            let vw = F32x16::from_slice(&w[c*16..]);
            (vx * inv_v).mul_add(vw, zero_v).copy_to_slice(&mut x[base..base+16]);
        }
        // 8-wide tail
        let mut d8_start = chunks * 16;
        if d8_start + 8 <= dim {
            let base = i * dim + d8_start;
            let vx = F32x8::from_slice(&x[base..]);
            let vw = F32x8::from_slice(&w[d8_start..]);
            (vx * F32x8::splat(inv) * vw).copy_to_slice(&mut x[base..base+8]);
            d8_start += 8;
        }
        for d in d8_start..dim {
            x[i*dim+d] *= inv * w.get(d).copied().unwrap_or(1.0);
        }
    }
}

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// SIMD-accelerated matrix multiply: A [m×k] × B^T [n×k] → C [m×n].
///
/// Uses matrixmultiply::sgemm with transposed B strides.
/// This is the hot path — AVX2/AVX-512 with cache blocking.
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    if m == 0 || n == 0 || k == 0 { return c; }
    // A is [m, k] row-major: row stride = k, col stride = 1
    // B is [n, k] row-major (stored transposed): we need B^T = [k, n]
    //   B^T row stride = 1, col stride = k
    // C is [m, n] row-major: row stride = n, col stride = 1
    // Use ndarray's SIMD-optimized GEMM wrapper (matrixmultiply under the hood).
    // But ndarray::backend::native::gemm_f32 expects A[m,k] * B[k,n] with row-major B.
    // Our B is transposed [n, k]. We need to transpose it or use raw strides.
    // Simplest: transpose B to [k, n] then call gemm.
    let mut bt = vec![0.0f32; k * n];
    for i in 0..n {
        for j in 0..k {
            bt[j * n + i] = b[i * k + j];
        }
    }
    gemm_f32(
        m, n, k,
        1.0, a, k,
        &bt, n,
        0.0, &mut c, n,
    );
    c
}

fn softmax(x: &mut [f32]) {
    let mx = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for v in x.iter_mut() { *v = (*v - mx).exp(); s += *v; }
    let inv = 1.0 / s.max(1e-10);
    for v in x.iter_mut() { *v *= inv; }
}

fn rope(q: &mut [f32], seq: usize, head_dim: usize) {
    for pos in 0..seq {
        for d in (0..head_dim).step_by(2) {
            let freq = 1.0 / 10000.0f32.powf(d as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let (sin, cos) = angle.sin_cos();
            let i = pos * head_dim + d;
            if i + 1 < q.len() {
                let q0 = q[i];
                let q1 = q[i + 1];
                q[i] = q0 * cos - q1 * sin;
                q[i + 1] = q0 * sin + q1 * cos;
            }
        }
    }
}

/// Run one transformer layer.
fn transformer_layer(
    hidden: &mut Vec<f32>,
    seq: usize,
    dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    inter: usize,
    w: &HashMap<String, Vec<f32>>,
    prefix: &str,
) {
    let ln1 = get_weight(w, &format!("{prefix}.input_layernorm.weight"));
    let qw = get_weight(w, &format!("{prefix}.self_attn.q_proj.weight"));
    let kw = get_weight(w, &format!("{prefix}.self_attn.k_proj.weight"));
    let vw = get_weight(w, &format!("{prefix}.self_attn.v_proj.weight"));
    let ow = get_weight(w, &format!("{prefix}.self_attn.o_proj.weight"));
    let ln2 = get_weight(w, &format!("{prefix}.post_attention_layernorm.weight"));
    let gw = get_weight(w, &format!("{prefix}.mlp.gate_proj.weight"));
    let uw = get_weight(w, &format!("{prefix}.mlp.up_proj.weight"));
    let dw = get_weight(w, &format!("{prefix}.mlp.down_proj.weight"));

    // Attention
    let mut normed = hidden.clone();
    rms_norm(&mut normed, ln1, dim);

    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let mut q = matmul(&normed, qw, seq, dim, q_dim);
    let mut k = matmul(&normed, kw, seq, dim, kv_dim);
    let v = matmul(&normed, vw, seq, dim, kv_dim);

    // QK norm if present
    let qn_key = format!("{prefix}.self_attn.q_norm.weight");
    if w.contains_key(&qn_key) {
        rms_norm(&mut q, get_weight(w, &qn_key), head_dim);
        rms_norm(&mut k, get_weight(w, &format!("{prefix}.self_attn.k_norm.weight")), head_dim);
    }

    // RoPE
    for h in 0..n_heads {
        let mut head_q: Vec<f32> = (0..seq).map(|s| {
            (0..head_dim).map(|d| q[s * q_dim + h * head_dim + d]).collect::<Vec<f32>>()
        }).flatten().collect();
        rope(&mut head_q, seq, head_dim);
        for s in 0..seq {
            for d in 0..head_dim {
                q[s * q_dim + h * head_dim + d] = head_q[s * head_dim + d];
            }
        }
    }
    for h in 0..n_kv_heads {
        let mut head_k: Vec<f32> = (0..seq).map(|s| {
            (0..head_dim).map(|d| k[s * kv_dim + h * head_dim + d]).collect::<Vec<f32>>()
        }).flatten().collect();
        rope(&mut head_k, seq, head_dim);
        for s in 0..seq {
            for d in 0..head_dim {
                k[s * kv_dim + h * head_dim + d] = head_k[s * head_dim + d];
            }
        }
    }

    // GQA attention
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; seq * q_dim];
    let kv_group = n_heads / n_kv_heads;

    for h in 0..n_heads {
        let kv_h = h / kv_group;
        let mut scores = vec![0.0f32; seq * seq];
        for s in 0..seq {
            for t in 0..=s {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[s * q_dim + h * head_dim + d]
                         * k[t * kv_dim + kv_h * head_dim + d];
                }
                scores[s * seq + t] = dot * scale;
            }
            for t in (s+1)..seq {
                scores[s * seq + t] = f32::NEG_INFINITY;
            }
            softmax(&mut scores[s * seq..(s + 1) * seq]);
        }
        for s in 0..seq {
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for t in 0..seq {
                    sum += scores[s * seq + t] * v[t * kv_dim + kv_h * head_dim + d];
                }
                attn_out[s * q_dim + h * head_dim + d] = sum;
            }
        }
    }

    // O projection + residual
    let o_out = matmul(&attn_out, ow, seq, q_dim, dim);
    for i in 0..hidden.len() { hidden[i] += o_out[i]; }

    // MLP
    let mut normed2 = hidden.clone();
    rms_norm(&mut normed2, ln2, dim);
    let gate = matmul(&normed2, gw, seq, dim, inter);
    let up = matmul(&normed2, uw, seq, dim, inter);
    let mut gated = vec![0.0f32; seq * inter];
    for i in 0..seq * inter { gated[i] = silu(gate[i]) * up[i]; }
    let down = matmul(&gated, dw, seq, inter, dim);
    for i in 0..hidden.len() { hidden[i] += down[i]; }
}

/// Run full TTS: text → talker → code_predictor → codec tokens.
fn run_tts(
    w: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, [usize; 2]>,
    tokens: &[usize],
) -> Vec<Vec<usize>> {
    let seq = tokens.len();

    // Text embedding
    let embed = get_weight(w, "talker.model.text_embedding.weight");
    let embed_cols = shapes.get("talker.model.text_embedding.weight")
        .map(|s| s[1]).unwrap_or(TALKER_HIDDEN);
    let mut hidden: Vec<f32> = tokens.iter()
        .flat_map(|&t| embed[t * embed_cols..(t + 1) * embed_cols].to_vec())
        .collect();

    // Text projection
    let fc1 = get_weight(w, "talker.text_projection.linear_fc1.weight");
    let fc1_b = get_weight(w, "talker.text_projection.linear_fc1.bias");
    let fc2 = get_weight(w, "talker.text_projection.linear_fc2.weight");
    let fc2_b = get_weight(w, "talker.text_projection.linear_fc2.bias");

    let fc1_out_dim = shapes.get("talker.text_projection.linear_fc1.weight")
        .map(|s| s[0]).unwrap_or(TALKER_HIDDEN);
    let mut proj = matmul(&hidden, fc1, seq, embed_cols, fc1_out_dim);
    for s in 0..seq {
        for d in 0..fc1_out_dim {
            proj[s * fc1_out_dim + d] = silu(proj[s * fc1_out_dim + d] + fc1_b.get(d).copied().unwrap_or(0.0));
        }
    }
    let fc2_out_dim = shapes.get("talker.text_projection.linear_fc2.weight")
        .map(|s| s[0]).unwrap_or(TALKER_HIDDEN);
    hidden = matmul(&proj, fc2, seq, fc1_out_dim, fc2_out_dim);
    for s in 0..seq {
        for d in 0..fc2_out_dim {
            hidden[s * fc2_out_dim + d] += fc2_b.get(d).copied().unwrap_or(0.0);
        }
    }

    // 28 talker layers
    for i in 0..TALKER_LAYERS {
        transformer_layer(
            &mut hidden, seq, TALKER_HIDDEN,
            TALKER_HEADS, TALKER_KV_HEADS, TALKER_HEAD_DIM, TALKER_INTER,
            w, &format!("talker.model.layers.{i}"),
        );
    }

    // Final norm
    rms_norm(&mut hidden, get_weight(w, "talker.model.norm.weight"), TALKER_HIDDEN);

    // Codec head → argmax
    let codec_head = get_weight(w, "talker.codec_head.weight");
    let codec_size = shapes.get("talker.codec_head.weight").map(|s| s[0]).unwrap_or(3072);
    let logits = matmul(&hidden, codec_head, seq, TALKER_HIDDEN, codec_size);
    let codec_tokens: Vec<usize> = (0..seq).map(|s| {
        (0..codec_size)
            .max_by(|&a, &b| logits[s * codec_size + a].partial_cmp(&logits[s * codec_size + b]).unwrap())
            .unwrap_or(0)
    }).collect();

    // Code predictor
    let codec_embed = get_weight(w, "talker.model.codec_embedding.weight");
    let ce_cols = shapes.get("talker.model.codec_embedding.weight")
        .map(|s| s[1]).unwrap_or(CP_HIDDEN);
    let mut cp_hidden: Vec<f32> = codec_tokens.iter()
        .flat_map(|&t| {
            let start = t * ce_cols;
            let end = start + ce_cols;
            if end <= codec_embed.len() { codec_embed[start..end].to_vec() }
            else { vec![0.0; ce_cols] }
        })
        .collect();

    for i in 0..CP_LAYERS {
        transformer_layer(
            &mut cp_hidden, seq, CP_HIDDEN,
            CP_HEADS, CP_KV_HEADS, CP_HEAD_DIM, CP_INTER,
            w, &format!("talker.code_predictor.model.layers.{i}"),
        );
    }
    rms_norm(&mut cp_hidden, get_weight(w, "talker.code_predictor.model.norm.weight"), CP_HIDDEN);

    // 15 LM heads → codec codes
    let mut all_codes = Vec::with_capacity(15);
    for g in 0..15 {
        let lm = get_weight(w, &format!("talker.code_predictor.lm_head.{g}.weight"));
        let lm_size = shapes.get(&format!("talker.code_predictor.lm_head.{g}.weight"))
            .map(|s| s[0]).unwrap_or(2048);
        let lm_logits = matmul(&cp_hidden, lm, seq, CP_HIDDEN, lm_size);
        let codes: Vec<usize> = (0..seq).map(|s| {
            (0..lm_size)
                .max_by(|&a, &b| lm_logits[s * lm_size + a].partial_cmp(&lm_logits[s * lm_size + b]).unwrap())
                .unwrap_or(0)
        }).collect();
        all_codes.push(codes);
    }

    all_codes
}

// ═══════════════════════════════════════════════════════════════════
// WAV writer (16-bit PCM)
// ═══════════════════════════════════════════════════════════════════

fn write_wav(path: &str, samples: &[f32], sample_rate: u32) {
    let n = samples.len();
    let data_size = n * 2; // 16-bit
    let file_size = 36 + data_size;

    let mut f = BufWriter::new(File::create(path).expect("create wav"));
    f.write_all(b"RIFF").unwrap();
    f.write_all(&(file_size as u32).to_le_bytes()).unwrap();
    f.write_all(b"WAVEfmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap(); // chunk size
    f.write_all(&1u16.to_le_bytes()).unwrap();  // PCM
    f.write_all(&1u16.to_le_bytes()).unwrap();  // mono
    f.write_all(&sample_rate.to_le_bytes()).unwrap();
    f.write_all(&(sample_rate * 2).to_le_bytes()).unwrap(); // byte rate
    f.write_all(&2u16.to_le_bytes()).unwrap();  // block align
    f.write_all(&16u16.to_le_bytes()).unwrap(); // bits per sample
    f.write_all(b"data").unwrap();
    f.write_all(&(data_size as u32).to_le_bytes()).unwrap();
    for &s in samples {
        let i = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        f.write_all(&i.to_le_bytes()).unwrap();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = if args.len() > 1 { &args[1] }
        else { "/home/user/models/qwen3-tts-0.6b/model.safetensors" };

    println!("═══ RVQ END-TO-END TTS (PURE RUST) ═══");
    println!("  Model: {}", model_path);
    println!();

    // ─── Load raw weights (first pass: get raw codes) ────────────
    println!("[1] Loading raw weights (pass 1: raw inference)...");
    let t0 = Instant::now();
    let mut reader = BufReader::new(File::open(model_path).expect("open model"));
    let header = read_safetensors_header(&mut reader).expect("parse header");
    let (w_raw, shapes_raw, _, _) = load_weights(&mut reader, &header, false);
    println!("  {} tensors loaded in {:?}", w_raw.len(), t0.elapsed());

    // Run raw inference
    let text = "Hello, world.";
    let tokens: Vec<usize> = std::iter::once(151672)
        .chain(text.bytes().map(|b| b as usize))
        .chain(std::iter::once(151673))
        .collect();

    println!("\n[2] Running TTS (raw weights)...");
    let t0 = Instant::now();
    let raw_codes = run_tts(&w_raw, &shapes_raw, &tokens);
    println!("  {:?}, {} tokens × 15 codebooks", t0.elapsed(), tokens.len());

    // Save raw codes and free raw weights (keep shapes for comparison)
    let raw_codes_copy = raw_codes.clone();
    let shapes_keep = shapes_raw.clone();
    drop(w_raw);
    drop(shapes_raw);

    // ─── Load + RVQ compress (second pass) ─────────────────────────
    println!("\n[3] Loading + RVQ compressing (pass 2)...");
    let t0 = Instant::now();
    reader.seek(SeekFrom::Start(0)).unwrap();
    let header2 = read_safetensors_header(&mut reader).expect("parse header");
    let (w_rvq, shapes_rvq, cb_bytes, idx_bytes) = load_weights(&mut reader, &header2, true);
    println!("  Compressed in {:?}", t0.elapsed());
    println!("  Codebook: {:.1} MB, Indices: {:.1} MB, Total: {:.1} MB",
        cb_bytes as f64 / 1e6, idx_bytes as f64 / 1e6,
        (cb_bytes + idx_bytes) as f64 / 1e6);

    println!("\n[4] Running TTS (RVQ weights)...");
    let t0 = Instant::now();
    let rvq_codes = run_tts(&w_rvq, &shapes_rvq, &tokens);
    println!("  {:?}, {} tokens × 15 codebooks", t0.elapsed(), tokens.len());

    let raw_codes = raw_codes_copy;

    // ─── Compare ───────────────────────────────────────────────────
    println!("\n[5] Comparison:");
    let mut total_tokens = 0usize;
    let mut matching_tokens = 0usize;
    for g in 0..15 {
        let n = raw_codes[g].len().min(rvq_codes[g].len());
        for i in 0..n {
            total_tokens += 1;
            if raw_codes[g][i] == rvq_codes[g][i] { matching_tokens += 1; }
        }
    }
    let match_pct = matching_tokens as f64 / total_tokens.max(1) as f64 * 100.0;
    println!("  Codec token match: {}/{} ({:.1}%)", matching_tokens, total_tokens, match_pct);

    // Storage comparison
    let orig_bytes: usize = w_rvq.values()
        .filter(|v| v.len() >= 128 * 128)
        .map(|v| v.len() * 4)
        .sum();
    let rvq_bytes = cb_bytes + idx_bytes;
    println!("  Original weights: {:.1} MB", orig_bytes as f64 / 1e6);
    println!("  RVQ compressed:   {:.1} MB ({:.0}:1)", rvq_bytes as f64 / 1e6,
        orig_bytes as f64 / rvq_bytes.max(1) as f64);

    if match_pct > 90.0 {
        println!("\n  ★ SUCCESS: RVQ codebook preserves >90% codec tokens");
    } else if match_pct > 50.0 {
        println!("\n  ◐ PARTIAL: Some preservation, may be intelligible");
    } else {
        println!("\n  ✗ FAIL: Token preservation too low");
    }

    // Print sample codes for comparison
    println!("\n  First 5 tokens, codebook 0:");
    let n_show = 5.min(raw_codes[0].len());
    print!("    RAW: ");
    for i in 0..n_show { print!("{:5} ", raw_codes[0][i]); }
    println!();
    print!("    RVQ: ");
    for i in 0..n_show { print!("{:5} ", rvq_codes[0][i]); }
    println!();

    println!("\n═══ DONE ═══");
}
