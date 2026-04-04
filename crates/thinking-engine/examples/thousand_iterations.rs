//! 1000-iteration cascade experiment: 3σ-only vs full context
//! Then feed survivors into 64×64 resonance check.
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
//!   --example thousand_iterations

use thinking_engine::codebook_index::CodebookIndex;
use thinking_engine::engine::ThinkingEngine;
use std::collections::HashMap;

fn main() {
    println!("═══ 1000-ITERATION CASCADE EXPERIMENT ═══\n");

    // Load everything
    let tokenizer = tokenizers::Tokenizer::from_file("/tmp/bge-m3-tokenizer.json")
        .expect("tokenizer");
    let codebook = CodebookIndex::load(
        std::path::Path::new("/tmp/codebooks/bge-m3-roles-f16/codebook_index.u16"),
        1024, "bge-m3".into(),
    ).expect("codebook");
    let table = std::fs::read("/tmp/codebooks/bge-m3-roles-f16/semantic_distance_1024x1024.u8")
        .or_else(|_| std::fs::read("/tmp/codebooks/bge-m3-roles-f16/attn_q/distance_table_1024x1024.u8"))
        .expect("table");
    let n = 1024usize;

    let centroid_counts = codebook.centroid_counts();
    let idf: Vec<f32> = centroid_counts.iter()
        .map(|&c| 1.0 / (1.0 + (c.max(1) as f32).ln()))
        .collect();

    let engine = ThinkingEngine::new(table.clone());
    let floor = engine.floor;
    let table_ref = engine.distance_table_ref();

    // Test sentences
    let sentences = [
        "The wound is the place where the light enters you.",
        "You are not a drop in the ocean. You are the entire ocean in a drop.",
        "The stock market opened at nine.",
    ];

    for (si, text) in sentences.iter().enumerate() {
        println!("━━━ Sentence {} ━━━", si + 1);
        println!("  \"{}\"", text);

        let encoding = tokenizer.encode(*text, true).expect("tokenize");
        let token_ids = encoding.get_ids();
        let centroids = codebook.lookup_many(token_ids);

        // ── EXPERIMENT A: 1000 iterations, 3σ ONLY ──
        println!("\n  [A] 1000 iterations, 3σ focus only:");
        let (visited_3s, edges_3s, top_atoms_3s) = run_cascade(
            &centroids, table_ref, n, floor, &idf, 1000, true,
        );
        println!("    Unique atoms visited: {}", visited_3s.len());
        println!("    Edges discovered: {}", edges_3s.len());
        println!("    Top-10 by visit count:");
        let mut by_visits: Vec<(u16, u32)> = visited_3s.iter().map(|(&k, &v)| (k, v)).collect();
        by_visits.sort_by(|a, b| b.1.cmp(&a.1));
        for (atom, count) in by_visits.iter().take(10) {
            println!("      atom {:>4}: {} visits", atom, count);
        }

        // ── EXPERIMENT B: 1000 iterations, full context ──
        println!("\n  [B] 1000 iterations, full context (no 3σ filter):");
        let (visited_full, edges_full, top_atoms_full) = run_cascade(
            &centroids, table_ref, n, floor, &idf, 1000, false,
        );
        println!("    Unique atoms visited: {}", visited_full.len());
        println!("    Edges discovered: {}", edges_full.len());
        println!("    Top-10 by visit count:");
        let mut by_visits: Vec<(u16, u32)> = visited_full.iter().map(|(&k, &v)| (k, v)).collect();
        by_visits.sort_by(|a, b| b.1.cmp(&a.1));
        for (atom, count) in by_visits.iter().take(10) {
            println!("      atom {:>4}: {} visits", atom, count);
        }

        // ── Compare A vs B ──
        let only_3s: Vec<u16> = visited_3s.keys().filter(|k| !visited_full.contains_key(k)).cloned().collect();
        let only_full: Vec<u16> = visited_full.keys().filter(|k| !visited_3s.contains_key(k)).cloned().collect();
        let shared = visited_3s.keys().filter(|k| visited_full.contains_key(k)).count();
        println!("\n  [A vs B] shared={} only_3σ={} only_full={}",
            shared, only_3s.len(), only_full.len());

        // ── EXPERIMENT C: Feed into 64×64 resonance ──
        println!("\n  [C] 64×64 resonance from top-64 survivors:");

        // Take top-64 most-visited atoms from BOTH experiments
        // Combine: input centroids FIRST (they're the query), then top cascade survivors
        let mut combined: HashMap<u16, u32> = HashMap::new();
        // Input centroids get high priority (1000 visits = always included)
        for &c in &centroids {
            *combined.entry(c).or_insert(0) += 1000;
        }
        for (&atom, &count) in &visited_3s {
            *combined.entry(atom).or_insert(0) += count;
        }
        for (&atom, &count) in &visited_full {
            *combined.entry(atom).or_insert(0) += count;
        }
        let mut top64: Vec<(u16, u32)> = combined.into_iter().collect();
        top64.sort_by(|a, b| b.1.cmp(&a.1));
        top64.truncate(64);

        let atoms64: Vec<u16> = top64.iter().map(|&(a, _)| a).collect();
        println!("    64 atoms selected (by combined visit count)");

        // Build 64×64 distance sub-table from the full 1024×1024
        let mut table_64 = vec![0u8; 64 * 64];
        for i in 0..64 {
            for j in 0..64 {
                let ai = atoms64[i] as usize;
                let aj = atoms64[j] as usize;
                if ai < n && aj < n {
                    table_64[i * 64 + j] = table_ref[ai * n + aj];
                }
            }
        }

        // Table stats for the 64×64
        let min64 = *table_64.iter().min().unwrap();
        let max64 = *table_64.iter().max().unwrap();
        let avg64 = table_64.iter().map(|&v| v as f64).sum::<f64>() / table_64.len() as f64;
        let mut sorted64 = table_64.clone();
        sorted64.sort_unstable();
        let std64 = (table_64.iter().map(|&v| { let d = v as f64 - avg64; d * d }).sum::<f64>() / table_64.len() as f64).sqrt();
        println!("    64×64 table: min={} max={} avg={:.1} std={:.1}", min64, max64, avg64, std64);

        // Run ThinkingEngine on the 64×64 sub-table
        let mut engine64 = ThinkingEngine::new(table_64);
        println!("    64×64 engine floor: {}", engine64.floor);

        // Perturb with the original centroids mapped to 64-space
        let centroid_to_64: HashMap<u16, u16> = atoms64.iter().enumerate()
            .map(|(i, &a)| (a, i as u16)).collect();
        let mapped: Vec<u16> = centroids.iter()
            .filter_map(|c| centroid_to_64.get(c).copied())
            .collect();

        if mapped.is_empty() {
            println!("    ⚠ No input centroids map to the top-64. Skipping resonance.");
        } else {
            engine64.perturb(&mapped);
            println!("    Perturbed {} atoms (of {} tokens)", mapped.len(), centroids.len());

            // Think 10 cycles
            for cycle in 0..10 {
                let prev = engine64.energy.clone();
                engine64.cycle();
                let delta: f32 = engine64.energy.iter().zip(&prev)
                    .map(|(a, b)| (a - b).abs()).sum();
                let active = engine64.energy.iter().filter(|&&e| e > 0.001).count();
                let max_e = engine64.energy.iter().cloned().fold(0.0f32, f32::max);
                let max_idx = engine64.energy.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
                let real_atom = if max_idx < atoms64.len() { atoms64[max_idx] } else { 0 };
                if cycle == 0 || cycle == 4 || cycle == 9 || delta < 0.001 {
                    println!("    cycle {:>2}: delta={:.4} active={:>2}/64 peak=slot {} (atom {}, e={:.4})",
                        cycle + 1, delta, active, max_idx, real_atom, max_e);
                }
                if delta < 0.001 { break; }
            }

            // Top-5 in 64-space mapped back to real atoms
            let mut indexed: Vec<(usize, f32)> = engine64.energy.iter()
                .enumerate().map(|(i, &e)| (i, e)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            println!("    Resonance top-5:");
            for &(slot, e) in indexed.iter().take(5) {
                let real = if slot < atoms64.len() { atoms64[slot] } else { 0 };
                println!("      slot {:>2} → atom {:>4}: {:.4}", slot, real, e);
            }
        }
        println!();
    }

    println!("═══ 1000-ITERATION EXPERIMENT COMPLETE ═══");
}

/// Run N iterations of the cascade, returning visit counts + edges.
/// If `strict_3sigma` is true, only 3σ focus atoms propagate.
/// If false, all above-floor atoms propagate (full context).
fn run_cascade(
    initial: &[u16],
    table: &[u8],
    n: usize,
    floor: u8,
    idf: &[f32],
    iterations: usize,
    strict_3sigma: bool,
) -> (HashMap<u16, u32>, Vec<(u16, u16)>, Vec<u16>) {
    let top_k = 5;
    let mut visited: HashMap<u16, u32> = HashMap::new();
    let mut edges: Vec<(u16, u16)> = Vec::new();
    let mut edge_set: std::collections::HashSet<(u16, u16)> = std::collections::HashSet::new();

    // Initial query
    let mut query: Vec<(u16, f32)> = initial.iter()
        .map(|&c| {
            let w = idf.get(c as usize).copied().unwrap_or(1.0);
            (c, w)
        })
        .collect();
    // Dedup
    let mut merged: HashMap<u16, f32> = HashMap::new();
    for (idx, w) in &query { *merged.entry(*idx).or_insert(0.0) += w; }
    query = merged.into_iter().collect();

    for _iter in 0..iterations {
        let mut neighbors: Vec<(u16, f32)> = Vec::new();

        for &(q_idx, q_energy) in &query {
            if (q_idx as usize) >= n { continue; }
            *visited.entry(q_idx).or_insert(0) += 1;

            let row = &table[q_idx as usize * n..(q_idx as usize + 1) * n];

            let above: Vec<(usize, u8)> = row.iter().enumerate()
                .filter(|(j, &v)| *j != q_idx as usize && v > floor)
                .map(|(j, &v)| (j, v))
                .collect();

            if above.is_empty() { continue; }

            if strict_3sigma {
                // 3σ: only top few
                let mean: f32 = above.iter().map(|(_, v)| *v as f32).sum::<f32>() / above.len() as f32;
                let var: f32 = above.iter().map(|(_, v)| { let d = *v as f32 - mean; d * d }).sum::<f32>() / above.len() as f32;
                let std = var.sqrt().max(0.1);
                let thresh = mean + 3.0 * std;

                for &(j, val) in &above {
                    if (val as f32) >= thresh {
                        let freq = (val - floor) as f32 / (255 - floor) as f32;
                        let visits = visited.get(&(j as u16)).copied().unwrap_or(0);
                        let novelty = 1.0 / (1.0 + visits as f32 * visits as f32);
                        let conf = idf.get(j).copied().unwrap_or(1.0);
                        neighbors.push((j as u16, freq * q_energy * conf * novelty));

                        let edge = (q_idx.min(j as u16), q_idx.max(j as u16));
                        if edge_set.insert(edge) { edges.push(edge); }
                    }
                }
            } else {
                // Full context: all above floor
                for &(j, val) in &above {
                    let freq = (val - floor) as f32 / (255 - floor) as f32;
                    let visits = visited.get(&(j as u16)).copied().unwrap_or(0);
                    let novelty = 1.0 / (1.0 + visits as f32 * visits as f32);
                    let conf = idf.get(j).copied().unwrap_or(1.0);
                    neighbors.push((j as u16, freq * q_energy * conf * novelty));

                    let edge = (q_idx.min(j as u16), q_idx.max(j as u16));
                    if edge_set.insert(edge) { edges.push(edge); }
                }
            }
        }

        // Dedup neighbors, keep top-K
        let mut deduped: HashMap<u16, f32> = HashMap::new();
        for (idx, w) in &neighbors { *deduped.entry(*idx).or_insert(0.0) += w; }
        let mut sorted: Vec<(u16, f32)> = deduped.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted.truncate(top_k);

        if sorted.is_empty() { break; }
        query = sorted;
    }

    let mut top: Vec<(u16, u32)> = visited.iter().map(|(&k, &v)| (k, v)).collect();
    top.sort_by(|a, b| b.1.cmp(&a.1));
    let top_atoms: Vec<u16> = top.iter().take(10).map(|&(a, _)| a).collect();

    (visited, edges, top_atoms)
}
