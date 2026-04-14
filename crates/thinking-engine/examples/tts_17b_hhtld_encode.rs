//! Qwen3-TTS-1.7B safetensors → BGZ-HHTL-D compressed safetensors.
//!
//! Complete pipeline:
//!   safetensors (BF16, ~3.4 GB)
//!     → highheelbgz::SpiralEncoding (stride=role)
//!       → γ+φ rehydrate → Base17
//!         → WeightPalette (256-entry CLAM)
//!           → HhtlDTensor (Slot D + Slot V, 4 bytes/row)
//!             → safetensors output (compressed, ~4 MB)
//!
//! The output safetensors contains:
//!   - Per-role HHTL-D entries as u8 tensors (4 bytes per row)
//!   - Per-role palette as u8 tensor (Base17 × k entries)
//!   - Per-role distance table as u16 tensor (k × k)
//!   - Per-role route table as u8 tensor (k × k)
//!   - Metadata: original shapes, gamma profiles, compression stats
//!
//! ```sh
//! # From 1.7B Base model:
//! cargo run --release --example tts_17b_hhtld_encode \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /path/to/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors
//!
//! # Output: /path/to/Qwen3-TTS-12Hz-1.7B-Base/model_hhtld.safetensors
//! ```

use ndarray::hpc::gguf::{GgmlType, TensorInfo};
use ndarray::hpc::gguf_indexer::project_row_to_base17;
use ndarray::hpc::safetensors::read_safetensors_header;
use bgz_tensor::projection::Base17;
use bgz_tensor::palette::WeightPalette;
use bgz_tensor::hhtl_cache::HhtlCache;
use bgz_tensor::hhtl_d::{
    HhtlDTensor,
    build_hip_families,
};
use bgz_tensor::attention::AttentionTable;
use highheelbgz::rehydrate::{SpiralEncoding, GammaProfile};

// safetensors format written manually via write_hhtld_safetensors()

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// Qwen3-TTS-1.7B constants
// ═══════════════════════════════════════════════════════════════════════════

const TALKER_HIDDEN: usize = 2048;
const TALKER_LAYERS: usize = 28;
const TALKER_HEADS: usize = 16;
const TALKER_KV_HEADS: usize = 8;
const TALKER_HEAD_DIM: usize = 128; // 2048 / 16
const TALKER_INTER: usize = 6144;

const CP_HIDDEN: usize = 1024;
const CP_LAYERS: usize = 5;
const CP_HEADS: usize = 16;
const CP_KV_HEADS: usize = 8;
const CP_HEAD_DIM: usize = 64; // 1024 / 16
const CP_INTER: usize = 3072;

const N_CENTROIDS: usize = 256;
const SPIRAL_K: usize = 8;
const REHYDRATE_SPD: usize = 32;

// ═══════════════════════════════════════════════════════════════════════════
// Tensor classification
// ═══════════════════════════════════════════════════════════════════════════

fn role_stride(name: &str) -> u32 {
    let n = name.to_lowercase();
    if n.contains("q_proj") || n.contains("k_proj") || n.contains("o_proj") { 3 }
    else if n.contains("v_proj") { 5 }
    else if n.contains("gate_proj") { 8 }
    else if n.contains("up_proj") { 2 }
    else if n.contains("down_proj") { 4 }
    else { 3 }
}

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

/// Weight roles that get HHTL-D encoded (skip norms, embeddings — they're tiny).
const ENCODABLE_ROLES: &[&str] = &[
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
];

// ═══════════════════════════════════════════════════════════════════════════
// Streaming pipeline: safetensors → SpiralEncoding → Base17
// ═══════════════════════════════════════════════════════════════════════════

/// Convert ndarray Base17 to bgz-tensor Base17 (same layout: [i16; 17]).
fn convert_base17(nd: &ndarray::hpc::bgz17_bridge::Base17) -> Base17 {
    Base17 { dims: nd.dims }
}

/// Stream weight rows → SpiralEncoding → rehydrate → Base17.
/// Also returns raw f32 rows for direct HhtlDTensor encoding fallback.
fn stream_to_base17(
    reader: &mut BufReader<File>,
    tensor: &TensorInfo,
    data_offset: u64,
    role_name: &str,
) -> (Vec<Base17>, Vec<Vec<f32>>, GammaProfile) {
    let n_rows = tensor.dimensions[0] as usize;
    let n_cols = if tensor.dimensions.len() > 1 { tensor.dimensions[1] as usize } else { 1 };
    let stride = role_stride(role_name);
    let start = 20u32;

    let tensor_start = data_offset + tensor.offset;
    reader.seek(SeekFrom::Start(tensor_start)).unwrap();

    // Read all rows as f32
    let f32_rows: Vec<Vec<f32>> = match tensor.dtype {
        GgmlType::BF16 => {
            let n_elements = n_rows * n_cols;
            let mut raw = vec![0u8; n_elements * 2];
            reader.read_exact(&mut raw).unwrap();
            (0..n_rows).map(|r| {
                (0..n_cols).map(|c| {
                    let idx = r * n_cols + c;
                    let bits = u16::from_le_bytes([raw[idx * 2], raw[idx * 2 + 1]]);
                    f32::from_bits((bits as u32) << 16)
                }).collect()
            }).collect()
        }
        GgmlType::F16 => {
            let n_elements = n_rows * n_cols;
            let mut raw = vec![0u8; n_elements * 2];
            reader.read_exact(&mut raw).unwrap();
            (0..n_rows).map(|r| {
                (0..n_cols).map(|c| {
                    let idx = r * n_cols + c;
                    let bits = u16::from_le_bytes([raw[idx * 2], raw[idx * 2 + 1]]);
                    ndarray::hpc::gguf::f16_to_f32(bits)
                }).collect()
            }).collect()
        }
        GgmlType::F32 => {
            let n_elements = n_rows * n_cols;
            let mut raw = vec![0u8; n_elements * 4];
            reader.read_exact(&mut raw).unwrap();
            (0..n_rows).map(|r| {
                (0..n_cols).map(|c| {
                    let idx = r * n_cols + c;
                    f32::from_le_bytes([
                        raw[idx * 4], raw[idx * 4 + 1],
                        raw[idx * 4 + 2], raw[idx * 4 + 3],
                    ])
                }).collect()
            }).collect()
        }
        _ => {
            eprintln!("  Skipping {} (dtype {:?})", tensor.name, tensor.dtype);
            return (Vec::new(), Vec::new(), GammaProfile { role_gamma: [0.01; 6], phi_scale: 0.01 });
        }
    };

    // Calibrate GammaProfile
    let row_refs: Vec<&[f32]> = f32_rows.iter().map(|r| r.as_slice()).collect();
    let gamma = GammaProfile::calibrate(&row_refs);

    // Spiral-encode → rehydrate → Base17
    let encodings: Vec<SpiralEncoding> = f32_rows.iter()
        .map(|row| SpiralEncoding::encode(row, start, stride, SPIRAL_K))
        .collect();

    let base17_rows: Vec<Base17> = encodings.iter()
        .map(|enc| {
            let rehydrated = enc.rehydrate_interpolated(REHYDRATE_SPD, &gamma);
            convert_base17(&project_row_to_base17(&rehydrated))
        })
        .collect();

    (base17_rows, f32_rows, gamma)
}

// ═══════════════════════════════════════════════════════════════════════════
// Spearman ρ (for quality validation)
// ═══════════════════════════════════════════════════════════════════════════

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = v.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut result = vec![0.0f64; v.len()];
    let mut i = 0;
    while i < indexed.len() {
        let mut j = i;
        while j < indexed.len() && indexed[j].1 == indexed[i].1 { j += 1; }
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for k in i..j { result[indexed[k].0] = avg_rank; }
        i = j;
    }
    result
}

fn spearman_rho(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 { return 0.0; }
    let rx = ranks(x);
    let ry = ranks(y);
    let mean_r = (n as f64 + 1.0) / 2.0;
    let (mut num, mut dx2, mut dy2) = (0.0f64, 0.0f64, 0.0f64);
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

fn table_entropy_u16(dists: &[u16]) -> f64 {
    let mut counts: HashMap<u16, usize> = HashMap::new();
    for &v in dists { *counts.entry(v).or_insert(0) += 1; }
    let total = dists.len() as f64;
    if total < 1.0 { return 0.0; }
    counts.values()
        .map(|&c| { let p = c as f64 / total; -p * p.log2() })
        .sum()
}

// ═══════════════════════════════════════════════════════════════════════════
// Main pipeline
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let st_path = if args.len() > 1 {
        &args[1]
    } else {
        "/home/user/models/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors"
    };

    let out_path = {
        let p = std::path::Path::new(st_path);
        let stem = p.file_stem().unwrap().to_str().unwrap();
        let dir = p.parent().unwrap();
        dir.join(format!("{}_hhtld.safetensors", stem)).to_string_lossy().to_string()
    };

    println!("═══ QWEN3-TTS-1.7B → BGZ-HHTL-D SAFETENSORS ENCODER ═══");
    println!();
    println!("  Input:  {}", st_path);
    println!("  Output: {}", out_path);
    println!("  Talker: {}×{}, {} layers, {} heads (GQA {})",
        TALKER_HIDDEN, TALKER_INTER, TALKER_LAYERS, TALKER_HEADS, TALKER_KV_HEADS);
    println!("  CodePred: {}×{}, {} layers, {} heads (GQA {})",
        CP_HIDDEN, CP_INTER, CP_LAYERS, CP_HEADS, CP_KV_HEADS);
    println!();

    // ─── Step 1: Parse safetensors header ───────────────────────────────
    println!("[1] Parsing safetensors header...");
    let t0 = Instant::now();
    let mut reader = BufReader::new(File::open(st_path).expect("open safetensors"));
    let header = read_safetensors_header(&mut reader).expect("parse header");
    println!("    {} tensors, parsed in {:?}", header.tensors.len(), t0.elapsed());

    // Group by (component, role) — only encodable weight tensors
    let mut role_groups: HashMap<(String, String), Vec<&TensorInfo>> = HashMap::new();
    let mut skipped_tensors: Vec<String> = Vec::new();
    for tensor in &header.tensors {
        if !tensor.name.ends_with("weight") { continue; }
        let role = detect_role(&tensor.name).to_string();
        let comp = detect_component(&tensor.name).to_string();
        if ENCODABLE_ROLES.contains(&role.as_str()) {
            role_groups.entry((comp, role)).or_default().push(tensor);
        } else {
            skipped_tensors.push(tensor.name.clone());
        }
    }

    println!("    {} (component, role) groups for HHTL-D encoding:", role_groups.len());
    for ((comp, role), tensors) in role_groups.iter() {
        let total_rows: usize = tensors.iter().map(|t| t.dimensions[0] as usize).sum();
        let total_cols: usize = tensors.iter()
            .map(|t| if t.dimensions.len() > 1 { t.dimensions[1] as usize } else { 1 })
            .max().unwrap_or(0);
        println!("      {}/{}: {} tensors, {} rows × {} cols",
            comp, role, tensors.len(), total_rows, total_cols);
    }
    println!("    {} tensors skipped (norms, embeddings, biases)", skipped_tensors.len());

    // ─── Step 2: Stream → SpiralEncoding → Base17 ──────────────────────
    println!("\n[2] Streaming → SpiralEncoding → γ+φ rehydrate → Base17...");
    let mut role_base17: HashMap<(String, String), Vec<Base17>> = HashMap::new();
    let mut role_f32: HashMap<(String, String), Vec<Vec<f32>>> = HashMap::new();
    let mut role_gammas: HashMap<(String, String), GammaProfile> = HashMap::new();
    let mut role_shapes: HashMap<(String, String), [usize; 2]> = HashMap::new();

    for ((comp, role), tensors) in role_groups.iter() {
        let t0 = Instant::now();
        let mut all_b17 = Vec::new();
        let mut all_f32 = Vec::new();
        let mut last_gamma = GammaProfile { role_gamma: [0.01; 6], phi_scale: 0.01 };
        let mut total_rows = 0usize;
        let mut max_cols = 0usize;

        for tensor in tensors {
            let (rows, f32_rows, gamma) = stream_to_base17(
                &mut reader, tensor, header.tensor_data_offset, role,
            );
            total_rows += rows.len();
            if tensor.dimensions.len() > 1 {
                max_cols = max_cols.max(tensor.dimensions[1] as usize);
            }
            all_b17.extend(rows);
            all_f32.extend(f32_rows);
            last_gamma = gamma;
        }

        let stride = role_stride(role);
        println!("    {}/{}: {} rows, stride={}, γ={:.4}, φ={:.4} ({:?})",
            comp, role, total_rows, stride, last_gamma.role_gamma[0],
            last_gamma.phi_scale, t0.elapsed());

        role_shapes.insert((comp.clone(), role.clone()), [total_rows, max_cols]);
        role_gammas.insert((comp.clone(), role.clone()), last_gamma);
        role_base17.insert((comp.clone(), role.clone()), all_b17);
        role_f32.insert((comp.clone(), role.clone()), all_f32);
    }

    // ─── Step 3: Build WeightPalette per role ──────────────────────────
    println!("\n[3] Building {}-entry CLAM palette per role...", N_CENTROIDS);
    let mut role_palettes: HashMap<(String, String), WeightPalette> = HashMap::new();
    let mut role_caches: HashMap<(String, String), HhtlCache> = HashMap::new();

    for (key, rows) in &role_base17 {
        let t0 = Instant::now();
        let sample = if rows.len() > 4096 { &rows[..4096] } else { &rows[..] };
        let wp = WeightPalette::build(sample, N_CENTROIDS);
        let active = wp.counts.iter().filter(|&&c| c > 0).count();
        let max_radius = wp.radii.iter().copied().max().unwrap_or(0);

        // Build HhtlCache from palette
        let mut cache = HhtlCache::from_palette(wp.clone());
        let gamma = &role_gammas[key];
        let role_id = match key.1.as_str() {
            "q_proj" => 0.0, "k_proj" => 1.0, "v_proj" => 2.0,
            "gate_proj" => 3.0, "up_proj" => 4.0, "down_proj" => 5.0,
            "o_proj" => 6.0, _ => 9.0,
        };
        let stride_f = role_stride(&key.1) as f32;
        cache.gamma_meta = [gamma.role_gamma[0], gamma.phi_scale, stride_f, role_id];

        println!("    {}/{}: {}/{} active centroids, max_radius={} ({:?})",
            key.0, key.1, active, N_CENTROIDS, max_radius, t0.elapsed());

        role_palettes.insert(key.clone(), wp);
        role_caches.insert(key.clone(), cache);
    }

    // ─── Step 4: Build HIP families (16-way split) ─────────────────────
    println!("\n[4] Building 16-way HIP families per role...");
    let mut role_hip: HashMap<(String, String), Vec<u8>> = HashMap::new();

    for (key, wp) in &role_palettes {
        let families = build_hip_families(&wp.entries);
        let used: std::collections::HashSet<u8> = families.iter().copied().collect();
        println!("    {}/{}: {} families used of 16", key.0, key.1, used.len());
        role_hip.insert(key.clone(), families);
    }

    // ─── Step 5: Encode to HhtlDTensor ─────────────────────────────────
    println!("\n[5] Encoding weight matrices → HhtlDTensor (Slot D + Slot V)...");
    let mut role_hhtld: HashMap<(String, String), HhtlDTensor> = HashMap::new();

    for (key, f32_rows) in &role_f32 {
        let t0 = Instant::now();
        let cache = &role_caches[key];
        let hip = &role_hip[key];
        let role_name = format!("{}_{}", key.0, key.1);

        let hhtld = HhtlDTensor::encode(&role_name, f32_rows, cache, hip);
        let n_rows = hhtld.entries.len();
        let shape = role_shapes[key];
        let ratio = hhtld.compression_ratio();

        println!("    {}: {} rows → {} bytes (4 B/row), ratio={:.0}:1 ({:?})",
            role_name, n_rows, hhtld.entries_byte_size(), ratio, t0.elapsed());

        role_hhtld.insert(key.clone(), hhtld);
    }

    // ─── Step 6: Validate pairwise distance preservation ───────────────
    println!("\n[6] Pairwise Spearman ρ validation (50 pairs per role)...");
    let seed = 0x9E3779B97F4A7C15u64;

    for (key, rows) in &role_base17 {
        let cache = &role_caches[key];
        let at = AttentionTable::build(&role_palettes[key]);
        let n = rows.len();
        if n < 50 { continue; }

        let mut rng = seed;
        let mut exact = Vec::new();
        let mut quant = Vec::new();

        for _ in 0..50 {
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
            let ca = (0..N_CENTROIDS).min_by_key(|&c| rows[a].l1(&cache.palette.entries[c])).unwrap();
            let cb = (0..N_CENTROIDS).min_by_key(|&c| rows[b].l1(&cache.palette.entries[c])).unwrap();
            let quant_d = at.distance(ca as u8, cb as u8) as u32;
            exact.push(exact_d as f64);
            quant.push(quant_d as f64);
        }

        let rho = spearman_rho(&exact, &quant);
        let dists: Vec<u16> = quant.iter().map(|&d| d as u16).collect();
        let entropy = table_entropy_u16(&dists);
        println!("    {}/{}: ρ={:.4}, entropy={:.1} bits",
            key.0, key.1, rho, entropy);
    }

    // ─── Step 7: Write compressed safetensors ──────────────────────────
    println!("\n[7] Writing BGZ-HHTL-D compressed safetensors...");
    let t0 = Instant::now();

    // Collect all tensors for safetensors output
    let mut tensor_data: HashMap<String, Vec<u8>> = HashMap::new();
    let mut total_entries = 0usize;
    let mut total_entry_bytes = 0usize;

    for (key, hhtld) in &role_hhtld {
        let role_name = format!("{}_{}", key.0, key.1);

        // HHTL-D entries: flat u8 tensor, shape [n_rows, 4]
        let entries_bytes = hhtld.entries_to_bytes();
        let n_rows = hhtld.entries.len();
        total_entries += n_rows;
        total_entry_bytes += entries_bytes.len();
        tensor_data.insert(format!("{}.hhtld_entries", role_name), entries_bytes);

        // Palette: Base17 centroids as flat u8, shape [k, 34]
        let cache = &role_caches[key];
        let k = cache.k();
        let mut palette_bytes = Vec::with_capacity(k * 34);
        for entry in &cache.palette.entries {
            for &dim in &entry.dims {
                palette_bytes.extend_from_slice(&dim.to_le_bytes());
            }
        }
        tensor_data.insert(format!("{}.palette", role_name), palette_bytes);

        // Distance table: k×k u16 as flat u8
        let at = AttentionTable::build(&cache.palette);
        let mut dist_bytes = Vec::with_capacity(k * k * 2);
        for a in 0..k {
            for b in 0..k {
                dist_bytes.extend_from_slice(&at.distance(a as u8, b as u8).to_le_bytes());
            }
        }
        tensor_data.insert(format!("{}.distance_table", role_name), dist_bytes);

        // Route table: k×k u8
        let mut route_bytes = Vec::with_capacity(k * k);
        for a in 0..k {
            for b in 0..k {
                route_bytes.push(cache.route(a as u8, b as u8) as u8);
            }
        }
        tensor_data.insert(format!("{}.route_table", role_name), route_bytes);

        // Gamma metadata: 4 × f32 = 16 bytes
        let mut gamma_bytes = Vec::with_capacity(16);
        for &g in &cache.gamma_meta {
            gamma_bytes.extend_from_slice(&g.to_le_bytes());
        }
        tensor_data.insert(format!("{}.gamma_meta", role_name), gamma_bytes);

        // HIP family assignments: k × u8
        let hip = &role_hip[key];
        tensor_data.insert(format!("{}.hip_families", role_name), hip.clone());

        // Original shape metadata: 2 × u32 = 8 bytes
        let shape = role_shapes[key];
        let mut shape_bytes = Vec::with_capacity(8);
        shape_bytes.extend_from_slice(&(shape[0] as u32).to_le_bytes());
        shape_bytes.extend_from_slice(&(shape[1] as u32).to_le_bytes());
        tensor_data.insert(format!("{}.original_shape", role_name), shape_bytes);

        // Fisher z i8 pairwise cosine table: k×k i8 + 8 bytes gamma
        // Built from centroid-nearest representative f32 rows
        let n_cols = shape[1];
        let reps: Vec<Vec<f32>> = cache.palette.entries.iter()
            .map(|b| b.to_f32(n_cols))
            .collect();
        let fz = bgz_tensor::fisher_z::FisherZTable::build(&reps, k);
        let fz_bytes = fz.to_bytes();
        tensor_data.insert(format!("{}.fisher_z", role_name), fz_bytes);
        let fz_gamma_bytes = fz.gamma.to_le_bytes().to_vec();
        tensor_data.insert(format!("{}.fisher_z_gamma", role_name), fz_gamma_bytes);
    }

    // Also pass through skipped tensors (norms, embeddings) at original precision
    println!("    Passing through {} non-HHTL-D tensors at original precision...", skipped_tensors.len());
    for tensor_name in &skipped_tensors {
        if let Some(tensor) = header.tensors.iter().find(|t| &t.name == tensor_name) {
            let n: usize = tensor.dimensions.iter().map(|&d| d as usize).product();
            let elem_size = match tensor.dtype {
                GgmlType::BF16 | GgmlType::F16 => 2,
                GgmlType::F32 => 4,
                _ => continue,
            };
            reader.seek(SeekFrom::Start(header.tensor_data_offset + tensor.offset)).unwrap();
            let mut raw = vec![0u8; n * elem_size];
            reader.read_exact(&mut raw).unwrap();
            tensor_data.insert(format!("passthrough.{}", tensor.name), raw);
        }
    }

    // Build safetensors-compatible tensors and metadata
    // We write a simple binary format since the safetensors crate API
    // expects TensorView references. Our data is all u8 blobs.
    let total_output_bytes: usize = tensor_data.values().map(|v| v.len()).sum();
    let st_size = std::fs::metadata(st_path).map(|m| m.len()).unwrap_or(0);
    let compression_ratio = st_size as f64 / total_output_bytes as f64;

    // Build metadata
    let mut metadata: HashMap<String, String> = HashMap::new();
    metadata.insert("encoding".into(), "bgz-hhtl-d+fisher-z".into());
    metadata.insert("version".into(), "2".into());
    metadata.insert("original_model".into(), "Qwen3-TTS-12Hz-1.7B-Base".into());
    metadata.insert("palette_k".into(), N_CENTROIDS.to_string());
    metadata.insert("spiral_k".into(), SPIRAL_K.to_string());
    metadata.insert("rehydrate_spd".into(), REHYDRATE_SPD.to_string());
    metadata.insert("total_entries".into(), total_entries.to_string());
    metadata.insert("total_entry_bytes".into(), total_entry_bytes.to_string());
    metadata.insert("total_output_bytes".into(), total_output_bytes.to_string());
    metadata.insert("original_bytes".into(), st_size.to_string());
    metadata.insert("compression_ratio".into(), format!("{:.1}", compression_ratio));
    metadata.insert("n_encoded_roles".into(), role_hhtld.len().to_string());
    metadata.insert("n_passthrough".into(), skipped_tensors.len().to_string());

    // Write as safetensors format
    // Build TensorView list — all tensors stored as U8 with their byte length as shape
    let tensor_views: Vec<(String, Vec<u8>, Vec<usize>)> = tensor_data.into_iter()
        .map(|(name, data)| {
            let len = data.len();
            (name, data, vec![len])
        })
        .collect();

    // Write as safetensors format using custom writer
    write_hhtld_safetensors(&out_path, &tensor_views, &metadata)
        .expect("write safetensors");

    println!("    Written: {} bytes ({:.1} MB)", total_output_bytes, total_output_bytes as f64 / 1e6);
    println!("    Elapsed: {:?}", t0.elapsed());

    // ─── Summary ───────────────────────────────────────────────────────
    println!("\n[8] Compression summary:");
    println!("    Original safetensors:  {} bytes ({:.1} MB)", st_size, st_size as f64 / 1e6);
    println!("    HHTL-D entries only:   {} bytes ({:.1} KB)", total_entry_bytes, total_entry_bytes as f64 / 1024.0);
    println!("    Total output:          {} bytes ({:.1} MB)", total_output_bytes, total_output_bytes as f64 / 1e6);
    println!("    Entry compression:     {:.0}:1 (weight matrices only)", st_size as f64 / total_entry_bytes.max(1) as f64);
    println!("    Overall compression:   {:.1}:1 (including passthrough)", compression_ratio);
    println!();

    // Breakdown by component
    println!("    Per-component breakdown:");
    let mut talker_entries = 0usize;
    let mut cp_entries = 0usize;
    for (key, hhtld) in &role_hhtld {
        let bytes = hhtld.entries_byte_size();
        if key.0 == "talker" { talker_entries += bytes; }
        else if key.0 == "code_predictor" { cp_entries += bytes; }
    }
    println!("      Talker (28 layers):       {} bytes ({:.1} KB)", talker_entries, talker_entries as f64 / 1024.0);
    println!("      Code predictor (5 layers): {} bytes ({:.1} KB)", cp_entries, cp_entries as f64 / 1024.0);

    // Palette overhead
    let palette_overhead = role_caches.len() * (N_CENTROIDS * 34 + N_CENTROIDS * N_CENTROIDS * 2 + N_CENTROIDS * N_CENTROIDS);
    println!("      Palette overhead ({} roles): {} bytes ({:.1} KB)",
        role_caches.len(), palette_overhead, palette_overhead as f64 / 1024.0);

    println!("\n═══ DONE ═══");
}

/// Write BGZ-HHTL-D data as safetensors format.
///
/// safetensors format:
///   8 bytes: header_size (u64 LE)
///   header_size bytes: JSON header {tensor_name: {dtype, shape, data_offsets}, __metadata__: {...}}
///   remaining: concatenated tensor data
fn write_hhtld_safetensors(
    path: &str,
    tensors: &[(String, Vec<u8>, Vec<usize>)],
    metadata: &HashMap<String, String>,
) -> Result<(), String> {
    use serde_json::{json, Value, Map};

    let mut header_map = Map::new();

    // Add metadata
    let meta_value: Value = metadata.iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect::<Map<String, Value>>()
        .into();
    header_map.insert("__metadata__".into(), meta_value);

    // Compute data offsets
    let mut offset = 0usize;
    let mut tensor_entries = Vec::new();

    for (name, data, shape) in tensors {
        let begin = offset;
        let end = offset + data.len();
        offset = end;

        header_map.insert(name.clone(), json!({
            "dtype": "U8",
            "shape": shape,
            "data_offsets": [begin, end]
        }));

        tensor_entries.push((name.as_str(), data.as_slice()));
    }

    let header_json = serde_json::to_string(&Value::Object(header_map))
        .map_err(|e| format!("JSON serialize: {}", e))?;
    let header_bytes = header_json.as_bytes();
    let header_size = header_bytes.len() as u64;

    // Write file
    let mut f = File::create(path)
        .map_err(|e| format!("create file: {}", e))?;

    f.write_all(&header_size.to_le_bytes())
        .map_err(|e| format!("write header size: {}", e))?;
    f.write_all(header_bytes)
        .map_err(|e| format!("write header: {}", e))?;

    for (_name, data) in &tensor_entries {
        f.write_all(data)
            .map_err(|e| format!("write tensor data: {}", e))?;
    }

    f.flush().map_err(|e| format!("flush: {}", e))?;

    Ok(())
}
