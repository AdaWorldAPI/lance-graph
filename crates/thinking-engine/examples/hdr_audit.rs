//! HDR Audit: test all baked distance tables + SiLU correction on sixpack.
//!
//! Loads every HDR table, computes quality metrics, applies SiLU correction
//! on Gate×Up from the Qwen 9B sixpack, and cross-checks all models.

use thinking_engine::silu_correction::*;
use std::path::Path;

fn main() {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  HDR Distance Table Audit — All Models + SiLU Correction");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    let data_dir = "crates/thinking-engine/data";

    // ── 1. Audit each baked table ────────────────────────────────────
    let tables = vec![
        ("jina-v3-hdr",                    "Jina v3 (semantic precision)"),
        ("bge-m3-hdr",                     "BGE-M3 (multilingual truth)"),
        ("CompendiumLabs_bge-m3-hdr",      "CompendiumLabs BGE-M3"),
        ("bartowski_reader-lm-1.5b-hdr",   "Reader-LM 1.5B"),
        ("jina-reranker-v3-BF16-hdr",      "Jina Reranker v3 BF16"),
        ("Qwen3-5-4B-BF16-hdr",           "Qwen 3.5 4B BF16"),
        ("Qwen3-5-9B-BF16-hdr",           "Qwen 3.5 9B BF16"),
    ];

    eprintln!("{:<35} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8}",
        "Model", "Mean", "Std", "Min", "Max", "Entpy", "Topology");
    eprintln!("{}", "─".repeat(85));

    let mut all_tables: Vec<(String, Vec<u8>)> = Vec::new();

    for (dir_name, label) in &tables {
        let table_path = format!("{}/{}/distance_table_256x256.u8", data_dir, dir_name);
        match std::fs::read(&table_path) {
            Ok(data) => {
                let stats = table_stats(&data, 256);
                eprintln!("{:<35} {:6.1} {:6.1} {:6} {:6} {:6.2} {:8.1}",
                    label, stats.mean, stats.std, stats.min, stats.max, stats.entropy, stats.topology);
                all_tables.push((label.to_string(), data));
            }
            Err(_) => eprintln!("{:<35} MISSING", label),
        }
    }

    // ── 2. Sixpack: Gate, Up, Down separately ────────────────────────
    eprintln!("\n=== Qwen 3.5 9B Sixpack (L20) ===\n");
    let sixpack_dir = format!("{}/Qwen3-5-9B-BF16-sixpack-l20", data_dir);

    let gate_table = std::fs::read(format!("{}/Gate_256x256.u8", sixpack_dir)).unwrap();
    let up_table = std::fs::read(format!("{}/Up_256x256.u8", sixpack_dir)).unwrap();
    let down_table = std::fs::read(format!("{}/Down_256x256.u8", sixpack_dir)).unwrap();

    let gate_stats = table_stats(&gate_table, 256);
    let up_stats = table_stats(&up_table, 256);
    let down_stats = table_stats(&down_table, 256);

    eprintln!("{:<35} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8}",
        "Role", "Mean", "Std", "Min", "Max", "Entpy", "Topology");
    eprintln!("{}", "─".repeat(85));
    eprintln!("{:<35} {:6.1} {:6.1} {:6} {:6} {:6.2} {:8.1}", "Gate", gate_stats.mean, gate_stats.std, gate_stats.min, gate_stats.max, gate_stats.entropy, gate_stats.topology);
    eprintln!("{:<35} {:6.1} {:6.1} {:6} {:6} {:6.2} {:8.1}", "Up", up_stats.mean, up_stats.std, up_stats.min, up_stats.max, up_stats.entropy, up_stats.topology);
    eprintln!("{:<35} {:6.1} {:6.1} {:6} {:6} {:6.2} {:8.1}", "Down", down_stats.mean, down_stats.std, down_stats.min, down_stats.max, down_stats.entropy, down_stats.topology);

    // ── 3. SiLU correction: Gate × Up ────────────────────────────────
    eprintln!("\n=== SiLU Gate×Up Correction ===\n");

    // The gate table values represent gate-weight cosine distances.
    // We need to convert these to SiLU-correction factors.
    // High gate distance = different gate behavior = large correction needed.
    // Low gate distance = similar gate behavior = small correction.
    let n = 256;
    let mut corrected_up = up_table.clone();

    // Correction strategy: where Gate says two centroids behave differently
    // (gate distance > mean), adjust Up distance proportionally.
    let gate_mean = gate_stats.mean;
    let mut corrections_applied = 0;
    let mut total_correction = 0.0f64;

    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            let gate_dist = gate_table[idx] as f32;
            let up_dist = up_table[idx] as f32;

            // Gate distance > mean → these centroids have different gate behavior
            // → correct Up distance: pull apart if gate disagrees, push together if agrees
            if gate_dist > gate_mean as f32 + gate_stats.std as f32 {
                // Gate says: these behave DIFFERENTLY
                // If Up says similar (low dist) → UP IS WRONG → increase distance
                let correction = (gate_dist - gate_mean as f32) * 0.15;
                corrected_up[idx] = (up_dist + correction).clamp(0.0, 255.0) as u8;
                corrections_applied += 1;
                total_correction += correction.abs() as f64;
            } else if gate_dist < gate_mean as f32 - gate_stats.std as f32 {
                // Gate says: these behave SIMILARLY
                // If Up says different (high dist) → UP IS WRONG → decrease distance
                let correction = (gate_mean as f32 - gate_dist) * 0.1;
                corrected_up[idx] = (up_dist - correction).clamp(0.0, 255.0) as u8;
                corrections_applied += 1;
                total_correction += correction.abs() as f64;
            }
        }
    }

    let corrected_stats = table_stats(&corrected_up, n);
    let avg_correction = if corrections_applied > 0 { total_correction / corrections_applied as f64 } else { 0.0 };

    eprintln!("Corrections applied: {}/{} cells ({:.1}%)",
        corrections_applied, n*n, corrections_applied as f64/(n*n) as f64*100.0);
    eprintln!("Average correction:  {:.2} u8 levels", avg_correction);
    eprintln!("\n{:<35} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8}",
        "", "Mean", "Std", "Min", "Max", "Entpy", "Topology");
    eprintln!("{:<35} {:6.1} {:6.1} {:6} {:6} {:6.2} {:8.1}",
        "Up (raw)", up_stats.mean, up_stats.std, up_stats.min, up_stats.max, up_stats.entropy, up_stats.topology);
    eprintln!("{:<35} {:6.1} {:6.1} {:6} {:6} {:6.2} {:8.1}",
        "Up (gate-corrected)", corrected_stats.mean, corrected_stats.std, corrected_stats.min, corrected_stats.max, corrected_stats.entropy, corrected_stats.topology);

    // ── 4. Cross-check: run same atoms through raw vs corrected ──────
    eprintln!("\n=== ThinkingEngine Cross-Check: Up raw vs Up gate-corrected ===\n");

    let mut engine_raw = thinking_engine::engine::ThinkingEngine::new(up_table.clone());
    let mut engine_cor = thinking_engine::engine::ThinkingEngine::new(corrected_up.clone());

    let test_atoms: Vec<usize> = vec![0, 32, 64, 96, 128, 160, 192, 224, 255];
    let mut total_agreement = 0;
    let mut total_tests = 0;
    let mut total_cos = 0.0f64;

    for &atom in &test_atoms {
        engine_raw.energy = vec![0.0f32; n];
        engine_cor.energy = vec![0.0f32; n];
        engine_raw.energy[atom] = 1.0;
        engine_cor.energy[atom] = 1.0;

        for _ in 0..5 { engine_raw.cycle(); engine_cor.cycle(); }

        let raw_top = top_k(&engine_raw.energy, 5);
        let cor_top = top_k(&engine_cor.energy, 5);
        let agreement = raw_top.iter().filter(|a| cor_top.contains(a)).count();
        total_agreement += agreement;
        total_tests += 5;

        let cos = cosine(&engine_raw.energy, &engine_cor.energy);
        total_cos += cos;

        let verdict = if agreement >= 4 { "SAME" }
            else if agreement >= 2 { "PARTIAL" }
            else { "DIFFERENT" };
        eprintln!("  Atom {:3}: raw={:?} cor={:?} cos={:.4} {}",
            atom, raw_top, cor_top, cos, verdict);
    }

    let avg_cos = total_cos / test_atoms.len() as f64;
    eprintln!("\n  Peak agreement: {}/{} ({:.0}%)", total_agreement, total_tests,
        total_agreement as f64/total_tests as f64*100.0);
    eprintln!("  Average cosine: {:.6}", avg_cos);

    // ── 5. Cross-model distance (how different are the lenses?) ──────
    eprintln!("\n=== Cross-Model Lens Distance ===\n");
    eprintln!("{:<25}", "");
    for (name, _) in &all_tables { eprint!("{:>12}", &name[..name.len().min(11)]); }
    eprintln!();

    for (i, (name_i, table_i)) in all_tables.iter().enumerate() {
        eprint!("{:<25}", &name_i[..name_i.len().min(24)]);
        for (j, (_, table_j)) in all_tables.iter().enumerate() {
            if i == j {
                eprint!("{:>12}", "—");
            } else {
                let dist = table_l1(table_i, table_j);
                eprint!("{:>12.0}", dist);
            }
        }
        eprintln!();
    }

    // ── Verdict ──────────────────────────────────────────────────────
    eprintln!("\n═══════════════════════════════════════════════════════════");
    if avg_cos > 0.995 {
        eprintln!("  GATE CORRECTION: COSMETIC for Up role (dense model)");
    } else if avg_cos > 0.95 {
        eprintln!("  GATE CORRECTION: MATERIAL — changes emphasis within neighborhoods");
    } else if avg_cos > 0.85 {
        eprintln!("  GATE CORRECTION: SIGNIFICANT — different peaks emerge");
    } else {
        eprintln!("  GATE CORRECTION: TRANSFORMATIVE — gate changes everything");
    }
    eprintln!("  Topology gain: {:.1} → {:.1} (gate correction {})",
        up_stats.topology, corrected_stats.topology,
        if corrected_stats.topology > up_stats.topology { "IMPROVES" } else { "no change" });
    eprintln!("═══════════════════════════════════════════════════════════\n");
}

// ── Helpers ──────────────────────────────────────────────────────────────

struct TableStats {
    mean: f64, std: f64, min: u8, max: u8,
    entropy: f64, topology: f64,
}

fn table_stats(data: &[u8], n: usize) -> TableStats {
    let total = data.len() as f64;
    let mean = data.iter().map(|&v| v as f64).sum::<f64>() / total;
    let variance = data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / total;
    let std = variance.sqrt();
    let min = *data.iter().min().unwrap_or(&0);
    let max = *data.iter().max().unwrap_or(&0);

    // Shannon entropy of value distribution
    let mut histogram = [0u32; 256];
    for &v in data { histogram[v as usize] += 1; }
    let entropy = histogram.iter()
        .filter(|&&c| c > 0)
        .map(|&c| { let p = c as f64 / total; -p * p.log2() })
        .sum::<f64>();

    // Topology: std is the primary metric (higher = more structure)
    let topology = std;

    TableStats { mean, std, min, max, entropy, topology }
}

fn top_k(energy: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = energy.iter().enumerate().map(|(i,&e)| (i,e)).collect();
    indexed.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
    indexed.iter().take(k).map(|p| p.0).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

fn table_l1(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    let sum: u64 = a.iter().zip(b.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f64 / n as f64
}
