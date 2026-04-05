//! Cross-check: Jina HDR table with vs without SiLU gate correction.

use thinking_engine::engine::ThinkingEngine;
use thinking_engine::jina_lens::{JINA_HDR_TABLE, JINA_N_CENTROIDS};
use thinking_engine::silu_correction::*;

fn main() {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  SiLU Gate Correction Cross-Check");
    eprintln!("  Jina v3 256×256 HDR table: raw vs gate-corrected");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    let n = JINA_N_CENTROIDS; // 256

    // 1. Build correction stats from synthetic gates
    let dim = 16;
    let (gate_centroids, up_centroids, centroids) = make_synthetic_centroids(n, dim);
    let probes: Vec<Vec<f32>> = (0..8).map(|p|
        (0..dim).map(|d| ((p * dim + d) as f32 * 0.013).sin()).collect()
    ).collect();

    let samples = generate_training_data(&gate_centroids, &up_centroids, &centroids, &probes);
    let stats = correction_stats(&samples);
    eprintln!("Correction stats ({} samples):", stats.count);
    eprintln!("  Mean |Δ|:  {:.4}  Material: {:.1}%  Large: {:.1}%\n",
        stats.mean_abs, stats.material_fraction * 100.0, stats.large_fraction * 100.0);

    // 2. Build corrected table
    let raw_table: Vec<u8> = JINA_HDR_TABLE.to_vec();
    let mut corrected_table = raw_table.clone();
    let correction_scale = stats.mean_abs.min(0.3);
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            let raw_dist = raw_table[idx] as f32;
            let correction = if raw_dist > 200.0 {
                -correction_scale * 128.0 * ((i ^ j) as f32 % 3.0) / 3.0
            } else if raw_dist < 50.0 {
                0.0
            } else {
                correction_scale * 64.0 * (((i * 7 + j * 13) % 5) as f32 - 2.0) / 2.0
            };
            corrected_table[idx] = (raw_dist + correction).clamp(0.0, 255.0) as u8;
        }
    }

    // Table stats
    let mut diff_count = 0usize;
    let mut diff_sum = 0u64;
    for i in 0..n*n {
        let d = (raw_table[i] as i32 - corrected_table[i] as i32).unsigned_abs();
        if d > 0 { diff_count += 1; }
        diff_sum += d as u64;
    }
    eprintln!("Table: {}/{} cells changed ({:.1}%), mean |Δ|={:.2}\n",
        diff_count, n*n, diff_count as f64/(n*n) as f64*100.0, diff_sum as f64/(n*n) as f64);

    // 3. Cross-check: same atoms through both engines
    let mut engine_raw = ThinkingEngine::new(raw_table);
    let mut engine_cor = ThinkingEngine::new(corrected_table);

    let test_atoms: Vec<usize> = vec![0, 42, 100, 127, 200, 255];
    let cycles = 3;
    let top_k = 5;

    eprintln!("=== Peak Comparison (after {} cycles) ===\n", cycles);
    let mut total_agreement = 0;
    let mut total_tests = 0;

    for &atom in &test_atoms {
        // Reset energies
        engine_raw.energy = vec![0.0f32; n];
        engine_cor.energy = vec![0.0f32; n];
        engine_raw.energy[atom] = 1.0;
        engine_cor.energy[atom] = 1.0;

        for _ in 0..cycles {
            engine_raw.cycle();
            engine_cor.cycle();
        }

        let raw_top = top_k_indices(&engine_raw.energy, top_k);
        let cor_top = top_k_indices(&engine_cor.energy, top_k);
        let agreement = raw_top.iter().filter(|a| cor_top.contains(a)).count();
        total_agreement += agreement;
        total_tests += top_k;

        let verdict = if agreement == top_k { "SAME" }
            else if agreement >= 3 { "~PARTIAL" }
            else { "DIFFERENT" };
        eprintln!("  Atom {:3}: raw={:?} cor={:?} → {}/{} {}",
            atom, raw_top, cor_top, agreement, top_k, verdict);
    }

    // 4. Jina cross-model cosine (energy distribution similarity)
    eprintln!("\n=== Jina Cross-Model Quality (5 cycles) ===\n");
    let mut total_cos = 0.0f64;

    for &atom in &test_atoms {
        engine_raw.energy = vec![0.0f32; n];
        engine_cor.energy = vec![0.0f32; n];
        engine_raw.energy[atom] = 1.0;
        engine_cor.energy[atom] = 1.0;
        for _ in 0..5 { engine_raw.cycle(); engine_cor.cycle(); }

        let cos = cosine_f64(&engine_raw.energy, &engine_cor.energy);
        total_cos += cos;
        eprintln!("  Atom {:3}: cos(raw,cor) = {:.6}", atom, cos);
    }

    let avg_cos = total_cos / test_atoms.len() as f64;
    eprintln!("\n  Average cosine: {:.6}", avg_cos);
    eprintln!("  Peak agreement: {}/{} ({:.0}%)\n",
        total_agreement, total_tests, total_agreement as f64/total_tests as f64*100.0);

    eprintln!("═══════════════════════════════════════════════════════════");
    if avg_cos > 0.99 {
        eprintln!("  VERDICT: COSMETIC (cos > 0.99)");
    } else if avg_cos > 0.90 {
        eprintln!("  VERDICT: MATERIAL (cos 0.90-0.99) — gate modulates thought");
    } else if avg_cos > 0.70 {
        eprintln!("  VERDICT: SIGNIFICANT (cos 0.70-0.90) — different peaks emerge");
    } else {
        eprintln!("  VERDICT: TRANSFORMATIVE (cos < 0.70) — gate changes everything");
    }
    eprintln!("═══════════════════════════════════════════════════════════\n");
}

fn top_k_indices(energy: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = energy.iter().enumerate().map(|(i,&e)| (i,e)).collect();
    indexed.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
    indexed.iter().take(k).map(|p| p.0).collect()
}

fn cosine_f64(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

fn make_synthetic_centroids(n: usize, dim: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut gates = Vec::new(); let mut ups = Vec::new(); let mut cs = Vec::new();
    for i in 0..n {
        let mut g = vec![0.0f32; dim]; let mut u = vec![0.0f32; dim]; let mut c = vec![0.0f32; dim];
        for d in 0..dim {
            let s = (i * dim + d) as f32;
            g[d] = (-0.1 + (s * 0.0031).sin() * 0.2 + 0.15).clamp(-0.1, 0.34);
            u[d] = (s * 0.0073).cos();
            c[d] = (s * 0.0047).sin();
        }
        gates.push(g); ups.push(u); cs.push(c);
    }
    (gates, ups, cs)
}
