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

use ndarray::hpc::gguf::{GgmlType, TensorInfo};
use ndarray::hpc::gguf_indexer::{project_tensor_bf16_simd, project_row_to_base17};
use ndarray::hpc::safetensors::read_safetensors_header;
use bgz_tensor::projection::Base17;
use bgz_tensor::palette::WeightPalette;
use bgz_tensor::hhtl_cache::HhtlCache;

/// Convert ndarray Base17 to bgz-tensor Base17 (identical layout: [i16; 17]).
fn convert_base17(nd: &ndarray::hpc::bgz17_bridge::Base17) -> Base17 {
    Base17 { dims: nd.dims }
}
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
) -> Vec<ndarray::hpc::bgz17_bridge::Base17> {
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
/// Shannon entropy of a u16 distance set.
fn table_entropy_u16(dists: &[u16]) -> f64 {
    let mut counts: HashMap<u16, usize> = HashMap::new();
    for &v in dists { *counts.entry(v).or_insert(0) += 1; }
    let total = dists.len() as f64;
    if total < 1.0 { return 0.0; }
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
        if ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
            "embedding", "norm"]
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
            all_rows.extend(rows.into_iter().map(|r| convert_base17(&r)));
        }
        let n = all_rows.len();
        role_base17.insert((comp.clone(), role.clone()), all_rows);
        println!("    {}/{}: {} rows → Base17 ({:?})", comp, role, n, t0.elapsed());
    }

    // Step 3: Build WeightPalette per role (CLAM furthest-point sampling)
    println!("\n[3] Building CLAM {}-entry palette per role...", N_CENTROIDS);
    let mut role_wp: HashMap<(String, String), WeightPalette> = HashMap::new();
    for (key, rows) in &role_base17 {
        let t0 = Instant::now();
        let sample = if rows.len() > 4096 { &rows[..4096] } else { &rows[..] };
        let wp = WeightPalette::build(sample, N_CENTROIDS);
        let active = wp.counts.iter().filter(|&&c| c > 0).count();
        let max_radius = wp.radii.iter().copied().max().unwrap_or(0);
        println!("    {}/{}: {}/{} active, max_radius={} ({:?})",
            key.0, key.1, active, wp.entries.len(), max_radius, t0.elapsed());
        role_wp.insert(key.clone(), wp);
    }

    // Step 4: Build AttentionTable (distance tables) from palettes
    println!("\n[4] Building L1 distance tables ({}×{})...", N_CENTROIDS, N_CENTROIDS);
    use bgz_tensor::attention::AttentionTable;
    let mut role_tables_at: HashMap<(String, String), AttentionTable> = HashMap::new();
    for (key, wp) in &role_wp {
        let t0 = Instant::now();
        let at = AttentionTable::build(wp);
        // Compute stats from the table
        let k = wp.entries.len();
        let mut dists: Vec<u16> = Vec::new();
        for a in 0..k {
            for b in (a+1)..k {
                dists.push(at.distance(a as u8, b as u8));
            }
        }
        let mean_d = if dists.is_empty() { 0.0 }
            else { dists.iter().map(|&d| d as f64).sum::<f64>() / dists.len() as f64 };
        let max_d = dists.iter().copied().max().unwrap_or(0);
        let entropy = table_entropy_u16(&dists);
        println!("    {}/{}: entropy={:.1} bits, mean_d={:.0}, max_d={}, ({:?})",
            key.0, key.1, entropy, mean_d, max_d, t0.elapsed());
        role_tables_at.insert(key.clone(), at);
    }

    // Step 5: Pairwise preservation (50 pairs, Spearman ρ)
    println!("\n[5] Pairwise L1 preservation (50 pairs per role)...");
    let seed = 0x9E3779B97F4A7C15u64;
    for (key, rows) in &role_base17 {
        let wp = &role_wp[key];
        let at = &role_tables_at[key];
        let palette = &wp.entries;
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
            let quant_d = at.distance(ca as u8, cb as u8) as u32;
            exact.push(exact_d as f64);
            quant.push(quant_d as f64);
        }

        // Spearman ρ (manual — sort ranks, correlate)
        let rho = spearman_rho(&exact, &quant);
        println!("    {}/{}: Spearman ρ = {:.4}", key.0, key.1, rho);
    }

    // Step 6: Build HhtlCache per role and save as HHTL format
    let out_dir = std::path::Path::new(st_path).parent().unwrap().join("codebooks");
    std::fs::create_dir_all(&out_dir).ok();
    println!("\n[6] Building HhtlCache per role + saving HHTL...");

    let mut total_bytes = 0usize;
    for (key, wp) in &role_wp {
        let rows = &role_base17[key];

        // Build HhtlCache: palette + distance table + route table (cascade decisions)
        let t0 = Instant::now();
        let cache = HhtlCache::from_palette(wp.clone());

        let fname = format!("{}_{}_hhtl.bgz", key.0, key.1);
        let fpath = out_dir.join(&fname);
        cache.serialize(fpath.to_str().unwrap()).expect("serialize HHTL");

        // Also save row assignments (not part of HHTL, needed for token→archetype)
        let assignments: Vec<u8> = rows.iter().map(|row| {
            cache.nearest(row).0
        }).collect();
        let assign_path = out_dir.join(format!("{}_{}_assign.bin", key.0, key.1));
        std::fs::write(&assign_path, &assignments).unwrap();

        let hhtl_bytes = std::fs::metadata(&fpath).map(|m| m.len() as usize).unwrap_or(0);
        let assign_bytes = assignments.len();
        total_bytes += hhtl_bytes + assign_bytes;

        // Count route actions
        let k = cache.palette.entries.len();
        let n_skip = (0..k).flat_map(|a| (0..k).map(move |b| (a, b)))
            .filter(|&(a, b)| cache.route(a as u8, b as u8) == bgz_tensor::hhtl_cache::RouteAction::Skip)
            .count();
        let skip_pct = n_skip as f64 / (k * k) as f64 * 100.0;

        println!("    {}: {} bytes HHTL + {} bytes assign, skip={:.0}% ({:?})",
            fname, hhtl_bytes, assign_bytes, skip_pct, t0.elapsed());
    }
    println!("    Total: {} bytes ({:.1} KB)", total_bytes, total_bytes as f64 / 1024.0);

    // Summary
    let st_size = std::fs::metadata(st_path).map(|m| m.len()).unwrap_or(0);
    println!("\n[7] Compression summary:");
    println!("    Original safetensors: {} bytes ({:.1} MB)", st_size, st_size as f64 / 1e6);
    println!("    Codebook total:       {} bytes ({:.1} KB)", total_bytes, total_bytes as f64 / 1024.0);
    println!("    Ratio:                {:.0}:1", st_size as f64 / total_bytes as f64);

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
