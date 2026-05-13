//! Three-γ calibration for F16 1:1 role distance tables.
//!
//! Loads raw u8 distance tables from /tmp/codebooks/*-roles-f16/,
//! calibrates RoleGamma + CosineGamma + MetaGamma, re-quantizes each
//! table using the γ-expanded cosine→u8 mapping, and saves calibrated
//! tables alongside a CalibrationProfile JSON.
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml --example calibrate_roles

use bgz_tensor::gamma_calibration::{CalibrationProfile, CosineGamma, MetaGamma, RoleGamma};
fn main() {
    println!("=== Three-γ Role Calibration (F16 1:1 Tables) ===\n");

    // ── Step 1: Discover all *-roles-f16 directories ────────────────────────
    let codebook_root = "/tmp/codebooks";
    let models = discover_models(codebook_root);
    if models.is_empty() {
        eprintln!("No *-roles-f16 directories found in {}", codebook_root);
        return;
    }
    println!("Found {} model(s):", models.len());
    for m in &models {
        println!("  {}", m.name);
    }
    println!();

    // ── Step 2: For each model, load all role tables + meta.json ─────────────
    let mut model_data: Vec<ModelCalibrationData> = Vec::new();

    for model in &models {
        let mut roles: Vec<RoleData> = Vec::new();
        let role_dirs = match std::fs::read_dir(&model.path) {
            Ok(d) => d,
            Err(_) => continue,
        };

        for entry in role_dirs.flatten() {
            let role_path = entry.path();
            if !role_path.is_dir() {
                continue;
            }
            let role_name = entry.file_name().to_string_lossy().to_string();

            // Find distance_table_NxN.u8
            let (table_bytes, n) = match find_and_load_table(&role_path) {
                Some(v) => v,
                None => continue,
            };

            // Load meta.json for cosine range
            let meta = load_meta(&role_path);

            roles.push(RoleData {
                name: role_name,
                table: table_bytes,
                n,
                cos_min: meta.cos_min,
                cos_max: meta.cos_max,
            });
        }

        if roles.is_empty() {
            println!("SKIP {} — no role tables found", model.name);
            continue;
        }

        println!("Model: {} — {} roles loaded", model.name, roles.len());
        for r in &roles {
            println!(
                "  {:<14} {}×{}  cos[{:.4}, {:.4}]",
                r.name, r.n, r.n, r.cos_min, r.cos_max
            );
        }

        model_data.push(ModelCalibrationData {
            name: model.name.clone(),
            src_path: model.path.clone(),
            roles,
        });
    }

    if model_data.is_empty() {
        eprintln!("No usable model data found.");
        return;
    }
    println!();

    // ── Step 3: Calibrate RoleGamma per model (wider range → smaller γ) ─────
    println!("── RoleGamma calibration ──");
    let mut profiles: Vec<(String, RoleGamma, CosineGamma)> = Vec::new();

    for md in &model_data {
        // Build per-role cosine ranges
        // RoleGamma::calibrate expects weight rows, but we only have distance tables.
        // We synthesize role gammas from the cosine ranges: wider range = smaller gamma
        // (more dynamic range to compress → needs more shadow expansion).
        let mut gamma = [0.01f32; 6];
        for role in &md.roles {
            let idx = role_name_to_idx(&role.name);
            if idx >= 6 {
                continue;
            }
            let range = (role.cos_max - role.cos_min) as f32;
            // Wider range → smaller γ (more expansion needed near center)
            // γ = 1 / (1 + range)  scaled to reasonable domain
            gamma[idx] = (1.0 / (1.0 + range * 2.0)).max(0.01);
        }
        let phi_scale = gamma.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
        let role_gamma = RoleGamma { gamma, phi_scale };

        println!(
            "  {} RoleGamma: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]  φ={:.3}",
            md.name,
            role_gamma.gamma[0],
            role_gamma.gamma[1],
            role_gamma.gamma[2],
            role_gamma.gamma[3],
            role_gamma.gamma[4],
            role_gamma.gamma[5],
            role_gamma.phi_scale,
        );

        // ── Step 4: Calibrate CosineGamma from u8 histogram ─────────────────
        // Collect all pairwise cosine values from ALL role tables for this model
        let mut all_cosines: Vec<f64> = Vec::new();
        for role in &md.roles {
            for i in 0..role.n {
                for j in (i + 1)..role.n {
                    let u = role.table[i * role.n + j];
                    // Original linear mapping: u8 → cosine [-1, 1]
                    let cos = (u as f64 / 255.0) * 2.0 - 1.0;
                    all_cosines.push(cos);
                }
            }
        }

        let cosine_gamma = CosineGamma::calibrate(&all_cosines);
        println!(
            "  {} CosineGamma: γ={:.4}, center={:.4}, spread={:.4}  ({} pairs)",
            md.name,
            cosine_gamma.gamma,
            cosine_gamma.center,
            cosine_gamma.spread,
            all_cosines.len(),
        );

        profiles.push((md.name.clone(), role_gamma, cosine_gamma));
    }
    println!();

    // ── Step 5: Build MetaGamma cross-model offsets ─────────────────────────
    println!("── MetaGamma calibration ──");
    let model_cosines_for_meta: Vec<Vec<f64>> = model_data
        .iter()
        .map(|md| {
            let mut cosines = Vec::new();
            for role in &md.roles {
                for i in 0..role.n {
                    for j in (i + 1)..role.n {
                        let u = role.table[i * role.n + j];
                        cosines.push((u as f64 / 255.0) * 2.0 - 1.0);
                    }
                }
            }
            cosines
        })
        .collect();

    let meta_input: Vec<(&str, &[f64])> = model_data
        .iter()
        .zip(model_cosines_for_meta.iter())
        .map(|(md, cos)| (md.name.as_str(), cos.as_slice()))
        .collect();

    let meta_gamma = MetaGamma::calibrate(&meta_input);

    println!("  Baselines:");
    for (name, baseline) in meta_gamma.models.iter().zip(meta_gamma.baselines.iter()) {
        println!("    {:<30} median cosine = {:.4}", name, baseline);
    }
    if meta_gamma.models.len() > 1 {
        println!("  Cross-model offsets:");
        for i in 0..meta_gamma.models.len() {
            for j in 0..meta_gamma.models.len() {
                if i == j {
                    continue;
                }
                let off = meta_gamma.offsets[i * meta_gamma.models.len() + j];
                println!(
                    "    {} → {}: {:.4}",
                    meta_gamma.models[i], meta_gamma.models[j], off
                );
            }
        }
    }
    println!();

    // ── Step 6: Apply calibration — re-quantize tables ──────────────────────
    println!("── Applying γ-calibrated re-quantization ──");
    let mut total_saved = 0usize;

    for (md, (_, _role_gamma, cosine_gamma)) in model_data.iter().zip(profiles.iter()) {
        let out_dir = format!(
            "{}/{}-calibrated",
            codebook_root,
            md.src_path
                .split('/')
                .last()
                .unwrap_or(&md.name)
        );
        std::fs::create_dir_all(&out_dir).ok();

        println!("  Model: {} → {}", md.name, out_dir);

        for role in &md.roles {
            let role_out = format!("{}/{}", out_dir, role.name);
            std::fs::create_dir_all(&role_out).ok();

            // Before: compute histogram entropy on original table
            let entropy_before = u8_histogram_entropy(&role.table);

            // Re-quantize: decode u8→cosine (linear), then re-encode via CosineGamma
            let mut calibrated = vec![0u8; role.n * role.n];
            for i in 0..role.n {
                // Diagonal stays at 255 (self-similarity = max)
                calibrated[i * role.n + i] = 255;
                for j in (i + 1)..role.n {
                    let orig_u8 = role.table[i * role.n + j];
                    // Decode from original linear mapping
                    let cosine = (orig_u8 as f64 / 255.0) * 2.0 - 1.0;
                    // Re-encode with γ-expanded mapping
                    let new_u8 = cosine_gamma.cosine_to_u8(cosine);
                    calibrated[i * role.n + j] = new_u8;
                    calibrated[j * role.n + i] = new_u8;
                }
            }

            let entropy_after = u8_histogram_entropy(&calibrated);

            // Save calibrated table
            let table_path = format!("{}/distance_table_{}x{}.u8", role_out, role.n, role.n);
            std::fs::write(&table_path, &calibrated).ok();

            // Save meta
            let meta_json = format!(
                concat!(
                    "{{",
                    "\"model\":\"{}\",",
                    "\"role\":\"{}\",",
                    "\"calibration\":\"three_gamma\",",
                    "\"cosine_gamma\":{:.6},",
                    "\"cosine_center\":{:.6},",
                    "\"cosine_spread\":{:.6},",
                    "\"entropy_before\":{:.4},",
                    "\"entropy_after\":{:.4},",
                    "\"table_size\":{}",
                    "}}"
                ),
                md.name,
                role.name,
                cosine_gamma.gamma,
                cosine_gamma.center,
                cosine_gamma.spread,
                entropy_before,
                entropy_after,
                role.n * role.n,
            );
            std::fs::write(format!("{}/meta.json", role_out), &meta_json).ok();

            let delta = entropy_after - entropy_before;
            let arrow = if delta > 0.01 {
                "▲"
            } else if delta < -0.01 {
                "▼"
            } else {
                "="
            };
            println!(
                "    {:<14} entropy: {:.4} → {:.4}  ({}{:.4})  {}×{}",
                role.name, entropy_before, entropy_after, arrow, delta.abs(), role.n, role.n,
            );

            total_saved += 1;
        }

        // Save CalibrationProfile as JSON
        let profile_json = format!(
            concat!(
                "{{",
                "\"model\":\"{}\",",
                "\"byte_size\":{},",
                "\"role_gamma\":[{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}],",
                "\"role_phi_scale\":{:.6},",
                "\"cosine_gamma\":{:.6},",
                "\"cosine_center\":{:.6},",
                "\"cosine_spread\":{:.6},",
                "\"meta_baseline\":{:.6}",
                "}}"
            ),
            md.name,
            CalibrationProfile::byte_size(),
            _role_gamma.gamma[0],
            _role_gamma.gamma[1],
            _role_gamma.gamma[2],
            _role_gamma.gamma[3],
            _role_gamma.gamma[4],
            _role_gamma.gamma[5],
            _role_gamma.phi_scale,
            cosine_gamma.gamma,
            cosine_gamma.center,
            cosine_gamma.spread,
            meta_gamma
                .baselines
                .get(
                    meta_gamma
                        .models
                        .iter()
                        .position(|m| m == &md.name)
                        .unwrap_or(0),
                )
                .copied()
                .unwrap_or(0.0),
        );
        std::fs::write(format!("{}/calibration_profile.json", out_dir), &profile_json).ok();
    }

    println!();
    println!("════════════════════════════════════════");
    println!(
        "Calibrated {} role tables across {} model(s)",
        total_saved,
        model_data.len()
    );
    println!("Profile size: {} bytes per model", CalibrationProfile::byte_size());
    println!("Output: /tmp/codebooks/*-roles-f16-calibrated/");
    println!("════════════════════════════════════════");
}

// ─── Data structures ────────────────────────────────────────────────────────

struct ModelInfo {
    name: String,
    path: String,
}

struct ModelCalibrationData {
    name: String,
    src_path: String,
    roles: Vec<RoleData>,
}

struct RoleData {
    name: String,
    table: Vec<u8>,
    n: usize,
    cos_min: f64,
    cos_max: f64,
}

struct MetaJson {
    cos_min: f64,
    cos_max: f64,
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn discover_models(root: &str) -> Vec<ModelInfo> {
    let mut models = Vec::new();
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return models,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if name.ends_with("-roles-f16") && path.is_dir() {
            let model_name = name.trim_end_matches("-roles-f16").to_string();
            models.push(ModelInfo {
                name: model_name,
                path: path.to_string_lossy().to_string(),
            });
        }
    }
    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}

fn find_and_load_table(role_dir: &std::path::Path) -> Option<(Vec<u8>, usize)> {
    let entries = std::fs::read_dir(role_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("distance_table_") && name.ends_with(".u8") {
            // Parse NxN from filename: distance_table_NxN.u8
            let inner = name
                .trim_start_matches("distance_table_")
                .trim_end_matches(".u8");
            let parts: Vec<&str> = inner.split('x').collect();
            if parts.len() == 2 {
                if let (Ok(n1), Ok(n2)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                    if n1 == n2 && n1 > 0 {
                        let data = std::fs::read(entry.path()).ok()?;
                        if data.len() == n1 * n2 {
                            return Some((data, n1));
                        }
                    }
                }
            }
        }
    }
    None
}

fn load_meta(role_dir: &std::path::Path) -> MetaJson {
    let meta_path = role_dir.join("meta.json");
    let default = MetaJson {
        cos_min: -1.0,
        cos_max: 1.0,
    };
    let content = match std::fs::read_to_string(&meta_path) {
        Ok(c) => c,
        Err(_) => return default,
    };
    // Minimal JSON parsing — extract cos_min and cos_max
    let cos_min = extract_json_f64(&content, "cos_min").unwrap_or(-1.0);
    let cos_max = extract_json_f64(&content, "cos_max").unwrap_or(1.0);
    MetaJson { cos_min, cos_max }
}

fn extract_json_f64(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\":", key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim_start();
    let end = after
        .find(|c: char| c == ',' || c == '}')
        .unwrap_or(after.len());
    after[..end].trim().parse().ok()
}

fn role_name_to_idx(name: &str) -> usize {
    match name {
        "attn_q" | "q_proj" => 0,
        "attn_k" | "k_proj" => 1,
        "attn_v" | "v_proj" => 2,
        "ffn_gate" | "gate_proj" => 3,
        "ffn_up" | "up_proj" => 4,
        "ffn_down" | "down_proj" => 5,
        // Combined QKV → map to Q slot
        "attn_qkv" => 0,
        "attn_output" | "o_proj" => 0,
        _ => 6, // unknown → skip
    }
}

/// Shannon entropy of the u8 histogram in bits.
/// Higher entropy = more uniform distribution = more information preserved.
fn u8_histogram_entropy(data: &[u8]) -> f64 {
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let total = data.len() as f64;
    if total == 0.0 {
        return 0.0;
    }
    let mut entropy = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}
