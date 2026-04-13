//! Qwen3-TTS → BGZ speech codebook: hard reality check.
//!
//! Streams the Qwen3-TTS 0.6B safetensors through the existing ndarray
//! BF16→Base17 SIMD projection pipeline, then builds per-role palettes
//! and distance tables. Compares against text-model structure.
//!
//! Uses AVX-512 SIMD via `project_tensor_bf16_simd()` — no scalar.
//!
//! ```sh
//! cargo run --release --example tts_bgz_codebook \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /home/user/models/qwen3-tts-0.6b/model.safetensors
//! ```

use ndarray::hpc::bgz17_bridge::Base17;
use ndarray::hpc::gguf::{GgmlType, TensorInfo};
use ndarray::hpc::gguf_indexer::{project_tensor_bf16_simd, project_row_to_base17};
use ndarray::hpc::safetensors::read_safetensors_header;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

const N_CENTROIDS: usize = 256;

/// TensorRole from name (matches hydrate.rs)
fn detect_role(name: &str) -> &'static str {
    let n = name.to_lowercase();
    if n.contains("q_proj") { "q_proj" }
    else if n.contains("k_proj") { "k_proj" }
    else if n.contains("v_proj") { "v_proj" }
    else if n.contains("o_proj") { "o_proj" }
    else if n.contains("gate_proj") { "gate_proj" }
    else if n.contains("up_proj") { "up_proj" }
    else if n.contains("down_proj") { "down_proj" }
    else if n.contains("embed") { "embedding" }
    else if n.contains("norm") { "norm" }
    else { "other" }
}

fn detect_component(name: &str) -> &'static str {
    if name.contains("code_predictor") { "code_predictor" }
    else if name.contains("talker") { "talker" }
    else if name.contains("speaker_encoder") { "speaker_encoder" }
    else { "other" }
}

/// Stream BF16 tensor rows and project to Base17 via SIMD.
fn stream_project_tensor(
    reader: &mut BufReader<File>,
    tensor: &TensorInfo,
    data_offset: u64,
    octave_stride: usize,
) -> Vec<Base17> {
    let n_rows = tensor.dimensions[0] as usize;
    let n_cols = if tensor.dimensions.len() > 1 { tensor.dimensions[1] as usize } else { 1 };

    // Seek to tensor data
    let tensor_start = data_offset + tensor.offset;
    reader.seek(SeekFrom::Start(tensor_start)).unwrap();

    match tensor.dtype {
        GgmlType::BF16 => {
            // Read entire tensor as u16 BF16 buffer
            let n_elements = n_rows * n_cols;
            let mut buf = vec![0u16; n_elements];
            let mut raw = vec![0u8; n_elements * 2];
            reader.read_exact(&mut raw).unwrap();
            for i in 0..n_elements {
                buf[i] = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
            }
            // SIMD projection: 8 rows at a time
            project_tensor_bf16_simd(&buf, n_rows, n_cols, octave_stride)
        }
        GgmlType::F16 => {
            // F16 → f32 → Base17 (no SIMD shortcut for F16 currently)
            let n_elements = n_rows * n_cols;
            let mut raw = vec![0u8; n_elements * 2];
            reader.read_exact(&mut raw).unwrap();
            let mut result = Vec::with_capacity(n_rows);
            for row_idx in 0..n_rows {
                let start = row_idx * n_cols;
                let mut f32_row = vec![0.0f32; n_cols];
                for c in 0..n_cols {
                    let bits = u16::from_le_bytes([raw[(start + c) * 2], raw[(start + c) * 2 + 1]]);
                    f32_row[c] = ndarray::hpc::gguf::f16_to_f32(bits);
                }
                result.push(project_row_to_base17(&f32_row));
            }
            result
        }
        _ => {
            eprintln!("  Skipping {} (dtype {:?})", tensor.name, tensor.dtype);
            Vec::new()
        }
    }
}

/// Build k-means palette from Base17 rows (L1 distance, deterministic init).
fn build_palette(rows: &[Base17], k: usize, max_iter: usize) -> Vec<Base17> {
    let n = rows.len();
    if n <= k {
        let mut result = rows.to_vec();
        result.resize(k, Base17::zero());
        return result;
    }

    // Deterministic init: evenly-spaced indices
    let mut centroids: Vec<[f64; 17]> = (0..k)
        .map(|i| {
            let idx = i * n / k;
            let mut c = [0.0f64; 17];
            for d in 0..17 { c[d] = rows[idx].dims[d] as f64; }
            c
        })
        .collect();

    for _iter in 0..max_iter {
        // Assign
        let mut sums = vec![[0.0f64; 17]; k];
        let mut counts = vec![0usize; k];

        for row in rows {
            let mut best = 0usize;
            let mut best_d = i64::MAX;
            for c in 0..k {
                let d: i64 = (0..17).map(|d| {
                    (row.dims[d] as i64 - centroids[c][d] as i64).abs()
                }).sum();
                if d < best_d { best_d = d; best = c; }
            }
            for d in 0..17 { sums[best][d] += row.dims[d] as f64; }
            counts[best] += 1;
        }

        // Update
        let mut delta = 0.0f64;
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..17 {
                    let new_val = sums[c][d] / counts[c] as f64;
                    delta += (new_val - centroids[c][d]).abs();
                    centroids[c][d] = new_val;
                }
            }
        }
        if delta < 1.0 { break; }
    }

    centroids.iter().map(|c| {
        let mut dims = [0i16; 17];
        for d in 0..17 { dims[d] = c[d].clamp(-32768.0, 32767.0) as i16; }
        Base17 { dims }
    }).collect()
}

/// Build L1 distance table (k × k, u16).
fn build_distance_table(palette: &[Base17]) -> Vec<u16> {
    let k = palette.len();
    let mut table = vec![0u16; k * k];
    for i in 0..k {
        for j in (i + 1)..k {
            let d = palette[i].l1(&palette[j]);
            let d16 = (d as u64).min(65535) as u16;
            table[i * k + j] = d16;
            table[j * k + i] = d16;
        }
    }
    table
}

/// Shannon entropy of distance table values.
fn table_entropy(table: &[u16]) -> f64 {
    let mut counts: HashMap<u16, usize> = HashMap::new();
    for &v in table { *counts.entry(v).or_insert(0) += 1; }
    let total = table.len() as f64;
    counts.values()
        .map(|&c| { let p = c as f64 / total; -p * p.log2() })
        .sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let st_path = if args.len() > 1 {
        &args[1]
    } else {
        "/home/user/models/qwen3-tts-0.6b/model.safetensors"
    };

    println!("═══ QWEN3-TTS → BGZ SPEECH CODEBOOK (RUST/AVX-512) ═══\n");

    // Step 1: Parse safetensors header
    println!("[1] Parsing safetensors header...");
    let t0 = Instant::now();
    let mut reader = BufReader::new(File::open(st_path).expect("open safetensors"));
    let header = read_safetensors_header(&mut reader).expect("parse header");
    println!("    {} tensors, parsed in {:?}", header.tensors.len(), t0.elapsed());

    // Group by (component, role)
    let mut role_groups: HashMap<(String, String), Vec<&TensorInfo>> = HashMap::new();
    for tensor in &header.tensors {
        if !tensor.name.ends_with("weight") { continue; }
        let role = detect_role(&tensor.name).to_string();
        let comp = detect_component(&tensor.name).to_string();
        if ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            .contains(&role.as_str())
        {
            role_groups.entry((comp, role)).or_default().push(tensor);
        }
    }

    println!("    {} (component, role) groups:", role_groups.len());
    for ((comp, role), tensors) in role_groups.iter() {
        let total: usize = tensors.iter().map(|t| t.dimensions[0] as usize).sum();
        println!("      {}/{}: {} tensors, {} rows", comp, role, tensors.len(), total);
    }

    // Step 2: Stream → SIMD Base17 projection per role
    println!("\n[2] Streaming → AVX-512 Base17 projection...");
    let mut role_base17: HashMap<(String, String), Vec<Base17>> = HashMap::new();
    let octave_stride = 1; // full resolution for first pass

    for ((comp, role), tensors) in role_groups.iter() {
        let t0 = Instant::now();
        let mut all_rows = Vec::new();
        for tensor in tensors {
            let rows = stream_project_tensor(
                &mut reader, tensor, header.tensor_data_offset, octave_stride,
            );
            all_rows.extend(rows);
        }
        let n = all_rows.len();
        role_base17.insert((comp.clone(), role.clone()), all_rows);
        println!("    {}/{}: {} rows → Base17 ({:?})", comp, role, n, t0.elapsed());
    }

    // Step 3: Build palette per role (use first 4096 rows for speed)
    println!("\n[3] Building {}-entry palette per role...", N_CENTROIDS);
    let mut role_palettes: HashMap<(String, String), Vec<Base17>> = HashMap::new();
    for (key, rows) in &role_base17 {
        let t0 = Instant::now();
        let sample = if rows.len() > 4096 { &rows[..4096] } else { &rows[..] };
        let palette = build_palette(sample, N_CENTROIDS, 15);
        let nonzero = palette.iter().filter(|c| *c != &Base17::zero()).count();
        role_palettes.insert(key.clone(), palette);
        println!("    {}/{}: {}/{} active centroids ({:?})",
            key.0, key.1, nonzero, N_CENTROIDS, t0.elapsed());
    }

    // Step 4: Build distance tables
    println!("\n[4] Building L1 distance tables ({}×{})...", N_CENTROIDS, N_CENTROIDS);
    let mut role_tables: HashMap<(String, String), Vec<u16>> = HashMap::new();
    for (key, palette) in &role_palettes {
        let t0 = Instant::now();
        let table = build_distance_table(palette);
        let entropy = table_entropy(&table);
        let nonzero: Vec<&u16> = table.iter().filter(|&&v| v > 0).collect();
        let mean_d = if nonzero.is_empty() { 0.0 }
            else { nonzero.iter().map(|&&v| v as f64).sum::<f64>() / nonzero.len() as f64 };
        let max_d = table.iter().copied().max().unwrap_or(0);
        role_tables.insert(key.clone(), table);
        println!("    {}/{}: entropy={:.1} bits, mean_d={:.0}, max_d={}, ({:?})",
            key.0, key.1, entropy, mean_d, max_d, t0.elapsed());
    }

    // Step 5: Pairwise preservation (50 pairs, Spearman ρ)
    println!("\n[5] Pairwise L1 preservation (50 pairs per role)...");
    let seed = 0x9E3779B97F4A7C15u64;
    for (key, rows) in &role_base17 {
        let palette = &role_palettes[key];
        let table = &role_tables[key];
        let n = rows.len();
        if n < 50 { continue; }

        let mut rng = seed;
        let mut exact = Vec::new();
        let mut quant = Vec::new();
        for _ in 0..50 {
            // SplitMix64
            rng = rng.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            let a = (z as usize) % n;
            rng = rng.wrapping_add(0x9E3779B97F4A7C15);
            z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            let b = (z as usize) % n;

            let exact_d = rows[a].l1(&rows[b]);
            // Find nearest centroids
            let ca = (0..N_CENTROIDS).min_by_key(|&c| rows[a].l1(&palette[c])).unwrap();
            let cb = (0..N_CENTROIDS).min_by_key(|&c| rows[b].l1(&palette[c])).unwrap();
            let quant_d = table[ca * N_CENTROIDS + cb] as u32;
            exact.push(exact_d as f64);
            quant.push(quant_d as f64);
        }

        // Spearman ρ (manual — sort ranks, correlate)
        let rho = spearman_rho(&exact, &quant);
        println!("    {}/{}: Spearman ρ = {:.4}", key.0, key.1, rho);
    }

    println!("\n═══ DONE ═══");
}

/// Manual Spearman rank correlation.
fn spearman_rho(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 { return 0.0; }
    let rx = ranks(x);
    let ry = ranks(y);
    let mean_r = (n as f64 + 1.0) / 2.0;
    let mut num = 0.0f64;
    let mut dx2 = 0.0f64;
    let mut dy2 = 0.0f64;
    for i in 0..n {
        let dx = rx[i] - mean_r;
        let dy = ry[i] - mean_r;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    if dx2 < 1e-15 || dy2 < 1e-15 { return 0.0; }
    num / (dx2.sqrt() * dy2.sqrt())
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = v.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut result = vec![0.0f64; v.len()];
    let mut i = 0;
    while i < indexed.len() {
        let mut j = i;
        while j < indexed.len() && indexed[j].1 == indexed[i].1 { j += 1; }
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based average
        for k in i..j { result[indexed[k].0] = avg_rank; }
        i = j;
    }
    result
}
