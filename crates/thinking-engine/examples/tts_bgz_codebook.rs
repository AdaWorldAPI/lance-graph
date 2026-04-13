//! Qwen3-TTS → BGZ speech codebook via highheelbgz spiral pipeline.
//!
//! Correct pipeline:
//!   safetensors → highheelbgz::SpiralEncoding (BF16 anchors, stride=role)
//!     → highheelbgz::GammaProfile (exact restore metadata)
//!       → bgz-tensor::WeightPalette (CLAM from spiral-encoded rows)
//!         → bgz-tensor::HhtlCache (route table + distance table)
//!
//! The spiral encoding uses stride to encode tensor role:
//!   stride=3 → QK (attention query/key)
//!   stride=5 → V  (content retrieval)
//!   stride=8 → Gate (coarse routing)
//!   stride=2 → Up (fine expansion)
//!   stride=4 → Down (compression)
//!
//! ```sh
//! cargo run --release --example tts_bgz_codebook \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /home/user/models/qwen3-tts-0.6b/model.safetensors
//! ```

use ndarray::hpc::gguf::{GgmlType, TensorInfo};
use ndarray::hpc::gguf_indexer::project_row_to_base17;
use ndarray::hpc::safetensors::read_safetensors_header;
use bgz_tensor::projection::Base17;
use bgz_tensor::palette::WeightPalette;
use bgz_tensor::hhtl_cache::HhtlCache;
use highheelbgz::rehydrate::{SpiralEncoding, GammaProfile};

/// Convert ndarray Base17 to bgz-tensor Base17 (identical layout: [i16; 17]).
fn convert_base17(nd: &ndarray::hpc::bgz17_bridge::Base17) -> Base17 {
    Base17 { dims: nd.dims }
}
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

const N_CENTROIDS: usize = 256;
/// Anchors per spiral dimension (K=8 gives 278 bytes per encoding,
/// good balance between fidelity and size).
const SPIRAL_K: usize = 8;
/// Target samples-per-dim for γ+φ interpolation when rehydrating.
const REHYDRATE_SPD: usize = 32;

/// TensorRole from name → highheelbgz stride.
/// stride IS the role — no separate field needed.
fn role_stride(name: &str) -> u32 {
    let n = name.to_lowercase();
    if n.contains("q_proj") || n.contains("k_proj") || n.contains("o_proj") { 3 }  // QK
    else if n.contains("v_proj") { 5 }   // V: content retrieval
    else if n.contains("gate_proj") { 8 } // Gate: coarse routing
    else if n.contains("up_proj") { 2 }   // Up: fine expansion
    else if n.contains("down_proj") { 4 } // Down: compression
    else { 3 } // default to QK stride
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

/// Stream tensor rows, spiral-encode each via highheelbgz, then rehydrate+project to Base17.
///
/// Pipeline per row:
///   BF16/F16 → f32 → SpiralEncoding::encode(f32, start, stride, k)
///     → rehydrate_interpolated(target_spd, gamma) → project_row_to_base17(f32)
///
/// Returns (base17_rows, spiral_encodings, gamma_profile).
fn stream_spiral_encode(
    reader: &mut BufReader<File>,
    tensor: &TensorInfo,
    data_offset: u64,
    role_name: &str,
) -> (Vec<ndarray::hpc::bgz17_bridge::Base17>, Vec<SpiralEncoding>, GammaProfile) {
    let n_rows = tensor.dimensions[0] as usize;
    let n_cols = if tensor.dimensions.len() > 1 { tensor.dimensions[1] as usize } else { 1 };
    let stride = role_stride(role_name);
    // Start offset: skip degenerate region at beginning of weight vector
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
        _ => {
            eprintln!("  Skipping {} (dtype {:?})", tensor.name, tensor.dtype);
            return (Vec::new(), Vec::new(), GammaProfile { role_gamma: [0.01; 6], phi_scale: 0.01 });
        }
    };

    // Step 1: Calibrate GammaProfile from raw rows (highheelbgz canonical)
    let row_refs: Vec<&[f32]> = f32_rows.iter().map(|r| r.as_slice()).collect();
    let gamma = GammaProfile::calibrate(&row_refs);

    // Step 2: Spiral-encode each row (BF16 anchors at role stride)
    let encodings: Vec<SpiralEncoding> = f32_rows.iter()
        .map(|row| SpiralEncoding::encode(row, start, stride, SPIRAL_K))
        .collect();

    // Step 3: Rehydrate with γ+φ interpolation → project to Base17
    let base17_rows: Vec<ndarray::hpc::bgz17_bridge::Base17> = encodings.iter()
        .map(|enc| {
            let rehydrated = enc.rehydrate_interpolated(REHYDRATE_SPD, &gamma);
            project_row_to_base17(&rehydrated)
        })
        .collect();

    (base17_rows, encodings, gamma)
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

    // Step 2: Stream → highheelbgz SpiralEncoding → GammaProfile → rehydrate → Base17
    println!("\n[2] Streaming → SpiralEncoding (stride=role) → γ+φ rehydrate → Base17...");
    let mut role_base17: HashMap<(String, String), Vec<Base17>> = HashMap::new();
    let mut role_gammas: HashMap<(String, String), GammaProfile> = HashMap::new();
    let mut role_spirals: HashMap<(String, String), Vec<SpiralEncoding>> = HashMap::new();

    for ((comp, role), tensors) in role_groups.iter() {
        let t0 = Instant::now();
        let mut all_rows = Vec::new();
        let mut all_encodings = Vec::new();
        let mut last_gamma = GammaProfile { role_gamma: [0.01; 6], phi_scale: 0.01 };
        let stride = role_stride(role);
        for tensor in tensors {
            let (rows, encodings, gamma) = stream_spiral_encode(
                &mut reader, tensor, header.tensor_data_offset, role,
            );
            all_rows.extend(rows.into_iter().map(|r| convert_base17(&r)));
            all_encodings.extend(encodings);
            last_gamma = gamma;
        }
        let n = all_rows.len();
        let phi = last_gamma.phi_scale;
        let g0 = last_gamma.role_gamma[0];
        println!("    {}/{}: {} rows, stride={}, gamma={:.4}, φ_scale={:.4}, K={} ({:?})",
            comp, role, n, stride, g0, phi, SPIRAL_K, t0.elapsed());
        role_gammas.insert((comp.clone(), role.clone()), last_gamma);
        role_spirals.insert((comp.clone(), role.clone()), all_encodings);
        role_base17.insert((comp.clone(), role.clone()), all_rows);
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

        // Build HhtlCache: palette + distance table + route table + gamma metadata
        let t0 = Instant::now();
        let mut cache = HhtlCache::from_palette(wp.clone());

        // Populate gamma metadata from highheelbgz GammaProfile (canonical source)
        let gamma_profile = &role_gammas[key];
        let role_id = match key.1.as_str() {
            "q_proj" => 0.0, "k_proj" => 1.0, "v_proj" => 2.0,
            "gate_proj" => 3.0, "up_proj" => 4.0, "down_proj" => 5.0,
            "o_proj" => 6.0, "embedding" => 7.0, "norm" => 8.0,
            _ => 9.0,
        };
        // Store: [role_gamma_mean, phi_scale, stride_as_float, role_id]
        let stride_f = role_stride(&key.1) as f32;
        cache.gamma_meta = [gamma_profile.role_gamma[0], gamma_profile.phi_scale, stride_f, role_id];

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
