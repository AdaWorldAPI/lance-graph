//! Test whether existing i16 bgz7 files have useful information after to_f32() hydration.
//!
//! The question: is i16[17] (Pearson 0.458 vs ground truth) "mushy blur" or
//! does it still contain discriminative information for cross-model comparison?
//!
//! Tests:
//! 1. Hydrate i16→f32, compute pairwise cosines — do roles separate?
//! 2. Can hydrated data detect cross-model differences (base vs distilled)?
//! 3. What's the effective resolution? How many distinct cosine values?
//!
//! cargo run --release --manifest-path crates/bgz-tensor/Cargo.toml --example bgz7_hydration_quality

use bgz_tensor::projection::Base17;
use bgz_tensor::variance::Role;
use std::collections::HashMap;

fn main() {
    let bgz7_files = vec![
        "/home/user/ndarray/src/hpc/openchat/weights/openchat-3.5-0106.bgz7",
        "/home/user/ndarray/src/hpc/openchat/weights/llama4_scout_shard1.bgz7",
        "/home/user/ndarray/src/hpc/openchat/weights/llama4_scout_shard2.bgz7",
        "/home/user/ndarray/src/hpc/openchat/weights/llama4_scout_shard3.bgz7",
    ];

    println!("=== i16 bgz7 Hydration Quality Test ===\n");

    for path in &bgz7_files {
        if !std::path::Path::new(path).exists() { continue; }
        let model = std::path::Path::new(path).file_stem().unwrap().to_string_lossy();
        println!("============================================================");
        println!("Model: {}", model);
        println!("============================================================\n");

        let data = std::fs::read(path).unwrap();
        if data.len() < 8 || &data[0..4] != b"BGZ7" { continue; }
        let n_tensors = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        // Parse bgz7 → (name, Base17 i16[17]) per row
        let mut role_base17: HashMap<Role, Vec<(String, Base17)>> = HashMap::new();
        let mut pos = 8;

        for _ in 0..n_tensors {
            if pos + 4 > data.len() { break; }
            let name_len = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
            pos += 4;
            if pos + name_len > data.len() { break; }
            let name = String::from_utf8_lossy(&data[pos..pos+name_len]).to_string();
            pos += name_len;
            if pos + 9 > data.len() { break; }
            let _layer_type = data[pos]; pos += 1;
            let n_rows = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
            pos += 4;
            let n_cols = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
            pos += 4;

            let role = Role::from_name(&name);

            for r in 0..n_rows {
                if pos + 34 > data.len() { break; }
                let b17 = Base17::from_bytes(&data[pos..pos+34]);
                pos += 34;
                if let Some(role) = role {
                    if role_base17.entry(role).or_default().len() < 500 {
                        role_base17.entry(role).or_default().push((name.clone(), b17));
                    }
                }
            }
        }

        // === Test 1: i16 raw — do roles separate? ===
        println!("--- Test 1: i16 raw L1 — role separation ---");
        let mut role_centroids: Vec<(Role, Base17)> = Vec::new();
        for role in Role::ALL {
            if let Some(rows) = role_base17.get(&role) {
                if rows.is_empty() { continue; }
                // Compute centroid
                let n = rows.len() as i64;
                let mut sums = [0i64; 17];
                for (_, b) in rows { for d in 0..17 { sums[d] += b.dims[d] as i64; } }
                let mut dims = [0i16; 17];
                for d in 0..17 { dims[d] = (sums[d] / n) as i16; }
                let centroid = Base17 { dims };
                println!("  {:<5}: {} rows, centroid L1 magnitude = {}",
                    role.label(), rows.len(),
                    centroid.dims.iter().map(|d| d.unsigned_abs() as u32).sum::<u32>());
                role_centroids.push((role, centroid));
            }
        }

        // Cross-role centroid L1 at i16 resolution
        println!("\n  Cross-role centroid L1 (i16 raw):");
        for i in 0..role_centroids.len() {
            for j in (i+1)..role_centroids.len() {
                let d = role_centroids[i].1.l1(&role_centroids[j].1);
                println!("    {} ↔ {}: L1 = {}", role_centroids[i].0.label(), role_centroids[j].0.label(), d);
            }
        }

        // === Test 2: Hydrate i16→f32 via to_f32(), then cosine ===
        println!("\n--- Test 2: Hydrated f32 cosine — role separation ---");
        let n_cols_approx = 4096; // approximate; exact doesn't matter for relative comparison

        let mut role_hydrated: HashMap<Role, Vec<Vec<f32>>> = HashMap::new();
        for role in Role::ALL {
            if let Some(rows) = role_base17.get(&role) {
                let hydrated: Vec<Vec<f32>> = rows.iter()
                    .take(100)
                    .map(|(_, b)| b.to_f32(n_cols_approx))
                    .collect();
                if !hydrated.is_empty() {
                    role_hydrated.insert(role, hydrated);
                }
            }
        }

        // Intra-role cosine (within same role)
        for role in Role::ALL {
            if let Some(vecs) = role_hydrated.get(&role) {
                let sample = vecs.len().min(30);
                let mut cosines = Vec::new();
                for i in 0..sample {
                    for j in (i+1)..sample.min(i+5) {
                        cosines.push(cosine(&vecs[i], &vecs[j]));
                    }
                }
                if !cosines.is_empty() {
                    let mean = cosines.iter().sum::<f64>() / cosines.len() as f64;
                    let min = cosines.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = cosines.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    println!("  {:<5} intra-cosine: mean={:.4}, min={:.4}, max={:.4} ({} pairs)",
                        role.label(), mean, min, max, cosines.len());
                }
            }
        }

        // Cross-role cosine (between different roles, same row index)
        println!("\n  Cross-role cosine (hydrated, same row idx):");
        let roles_with_data: Vec<Role> = Role::ALL.iter()
            .filter(|r| role_hydrated.contains_key(r))
            .copied().collect();
        for i in 0..roles_with_data.len() {
            for j in (i+1)..roles_with_data.len() {
                let a_vecs = &role_hydrated[&roles_with_data[i]];
                let b_vecs = &role_hydrated[&roles_with_data[j]];
                let sample = a_vecs.len().min(b_vecs.len()).min(30);
                let mut cosines = Vec::new();
                for k in 0..sample {
                    cosines.push(cosine(&a_vecs[k], &b_vecs[k]));
                }
                if !cosines.is_empty() {
                    let mean = cosines.iter().sum::<f64>() / cosines.len() as f64;
                    println!("    {} ↔ {}: mean_cosine = {:.4}",
                        roles_with_data[i].label(), roles_with_data[j].label(), mean);
                }
            }
        }

        // === Test 3: Resolution — how many distinct cosine values? ===
        println!("\n--- Test 3: Effective resolution ---");
        if let Some(q_vecs) = role_hydrated.get(&Role::Q) {
            let sample = q_vecs.len().min(50);
            let mut all_cosines: Vec<f64> = Vec::new();
            for i in 0..sample {
                for j in (i+1)..sample {
                    all_cosines.push(cosine(&q_vecs[i], &q_vecs[j]));
                }
            }
            all_cosines.sort_by(|a, b| a.partial_cmp(b).unwrap());
            // Count distinct values (within 1e-6)
            let mut distinct = 1;
            for i in 1..all_cosines.len() {
                if (all_cosines[i] - all_cosines[i-1]).abs() > 1e-6 { distinct += 1; }
            }
            println!("  Q role: {} pairs, {} distinct cosine values ({:.1}% unique)",
                all_cosines.len(), distinct,
                distinct as f64 / all_cosines.len() as f64 * 100.0);
            if !all_cosines.is_empty() {
                println!("  cosine range: [{:.6}, {:.6}]", all_cosines[0], all_cosines.last().unwrap());
                println!("  median: {:.6}", all_cosines[all_cosines.len() / 2]);
            }

            // Compare i16 L1 vs hydrated cosine ranking
            if let Some(q_b17) = role_base17.get(&Role::Q) {
                let mut l1_vals: Vec<f64> = Vec::new();
                let mut cos_vals: Vec<f64> = Vec::new();
                let pairs = sample.min(q_b17.len());
                for i in 0..pairs {
                    for j in (i+1)..pairs.min(i+5) {
                        l1_vals.push(-(q_b17[i].1.l1(&q_b17[j].1) as f64));
                        cos_vals.push(cosine(&q_vecs[i], &q_vecs[j]));
                    }
                }
                if l1_vals.len() > 1 {
                    let sp = bgz_tensor::quality::spearman(&l1_vals, &cos_vals);
                    let pr = bgz_tensor::quality::pearson(&l1_vals, &cos_vals);
                    println!("  i16 L1 vs hydrated cosine: Pearson={:.4}, Spearman={:.4}", pr, sp);
                    if sp > 0.7 {
                        println!("  → USEFUL: i16 L1 ranking correlates with hydrated cosine");
                    } else if sp > 0.3 {
                        println!("  → MARGINAL: some signal survives but noisy");
                    } else {
                        println!("  → MUSH: i16 L1 ranking barely correlates with hydrated cosine");
                    }
                }
            }
        }
        println!();
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
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
    if denom < 1e-12 { 0.0 } else { dot / denom }
}
