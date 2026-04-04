//! RAW REALITY CHECK: dump every step of text → thought → qualia
//!
//! No filtering. No prettification. Just raw numbers at every stage.
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml --example reality_check

use thinking_engine::codebook_index::CodebookIndex;
use thinking_engine::engine::ThinkingEngine;
use thinking_engine::qualia::{Qualia17D, DIMS_17D, FAMILY_CENTROIDS};

fn main() {
    println!("═══ REALITY CHECK: text → tokenize → codebook → think → qualia ═══\n");

    // Load tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file("/tmp/bge-m3-tokenizer.json")
        .expect("tokenizer.json not found");
    println!("[tokenizer] vocab_size={}", tokenizer.get_vocab_size(true));

    // Load codebook index
    let codebook = CodebookIndex::load(
        std::path::Path::new("/tmp/codebooks/bge-m3-roles-f16/codebook_index.u16"),
        1024, "bge-m3".into(),
    ).expect("codebook_index.u16 not found");
    println!("[codebook] tokens={} unique_centroids={}", codebook.len(), codebook.unique_centroids());

    // Load distance table
    let table = std::fs::read("/tmp/codebooks/bge-m3-roles-f16/attn_q/distance_table_1024x1024.u8")
        .expect("distance table not found");
    println!("[table] size={}×{} bytes={}", 1024, 1024, table.len());

    // Table statistics
    let min = *table.iter().min().unwrap();
    let max = *table.iter().max().unwrap();
    let sum: u64 = table.iter().map(|&v| v as u64).sum();
    let avg = sum as f64 / table.len() as f64;
    let mut sorted = table.clone();
    sorted.sort_unstable();
    let median = sorted[table.len() / 2];
    let p10 = sorted[table.len() / 10];
    let p90 = sorted[table.len() * 9 / 10];
    println!("[table] min={} max={} avg={:.1} median={} p10={} p90={}", min, max, avg, median, p10, p90);

    // Count values above median
    let above_median = table.iter().filter(|&&v| v > median).count();
    println!("[table] values_above_median={} ({:.1}%)", above_median, above_median as f64 / table.len() as f64 * 100.0);

    // Build engine
    let mut engine = ThinkingEngine::new(table.clone());
    println!("[engine] size={} floor={}", engine.size, engine.floor);
    println!();

    // Test sentences
    let sentences = [
        "The cat sat on the mat.",
        "Quantum entanglement enables faster-than-light correlation.",
        "I feel deeply sad about losing my friend.",
        "The stock market crashed by 30% today.",
        "She laughed with pure joy at the surprise.",
        "Der schnelle braune Fuchs springt über den faulen Hund.",
    ];

    let mut all_results: Vec<(String, Vec<u16>, Vec<(usize, f32)>, Qualia17D)> = Vec::new();

    for (i, text) in sentences.iter().enumerate() {
        println!("━━━ Sentence {} ━━━", i + 1);
        println!("  text: \"{}\"", text);

        // Tokenize
        let encoding = tokenizer.encode(*text, true).expect("tokenize failed");
        let token_ids = encoding.get_ids();
        let tokens_str: Vec<String> = encoding.get_tokens().iter().map(|s| s.to_string()).collect();
        println!("  tokens ({}): {:?}", token_ids.len(), tokens_str);
        println!("  token_ids: {:?}", token_ids);

        // Codebook lookup
        let centroids = codebook.lookup_many(token_ids);
        println!("  centroids: {:?}", centroids);

        // Unique centroids
        let mut unique = centroids.clone();
        unique.sort(); unique.dedup();
        println!("  unique_centroids: {} / {} tokens", unique.len(), centroids.len());

        // Show which distance table rows these map to
        // For each unique centroid, show its row's non-floor values
        println!("  centroid row topology (values > floor={}):", engine.floor);
        for &c in &unique {
            let row = &table[c as usize * 1024..(c as usize + 1) * 1024];
            let above: Vec<(usize, u8)> = row.iter().enumerate()
                .filter(|(_, &v)| v > engine.floor)
                .map(|(j, &v)| (j, v))
                .collect();
            let self_val = row[c as usize];
            let mut top_above = above.clone();
            top_above.sort_by(|a, b| b.1.cmp(&a.1));
            top_above.truncate(3);
            println!("    row {}: self={} above_floor={} top3={:?}",
                c, self_val, above.len(), top_above);
        }

        // Perturb and think
        engine.reset();
        engine.perturb(&centroids);
        println!("  energy after perturb (nonzero):");
        for (j, &e) in engine.energy.iter().enumerate() {
            if e > 1e-10 {
                println!("    atom {:>4}: {:.6}", j, e);
            }
        }

        // Think cycle by cycle
        println!("  thinking (10 cycles):");
        for cycle in 0..10 {
            let prev_energy = engine.energy.clone();
            engine.cycle();
            let delta: f32 = engine.energy.iter().zip(&prev_energy)
                .map(|(a, b)| (a - b).abs()).sum();
            let active = engine.energy.iter().filter(|&&e| e > 0.001).count();
            let max_e = engine.energy.iter().cloned().fold(0.0f32, f32::max);
            let max_idx = engine.energy.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
            println!("    cycle {:>2}: delta={:.6} active={:>4} peak=atom {} ({:.6})",
                cycle + 1, delta, active, max_idx, max_e);
            if delta < 0.001 { break; }
        }

        // Final energy top-10
        let mut indexed: Vec<(usize, f32)> = engine.energy.iter()
            .enumerate().map(|(j, &e)| (j, e)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top10: Vec<(usize, f32)> = indexed.iter().take(10).cloned().collect();
        println!("  top-10 atoms:");
        for &(j, e) in &top10 {
            println!("    atom {:>4}: {:.6}", j, e);
        }

        // Qualia
        let qualia = Qualia17D::from_engine(&engine);
        let (family, dist) = qualia.nearest_family();
        println!("  qualia 17D → nearest family: {} (dist={:.3})", family, dist);
        println!("  qualia dims:");
        for (d, (&name, &val)) in DIMS_17D.iter().zip(qualia.dims.iter()).enumerate() {
            let bar_len = (val.abs() * 20.0).min(20.0) as usize;
            let bar: String = if val >= 0.0 { "█".repeat(bar_len) } else { format!("-{}", "█".repeat(bar_len)) };
            println!("    {:>2}. {:<14} {:>6.3}  {}", d, name, val, bar);
        }

        all_results.push((text.to_string(), centroids, top10, qualia));
        println!();
    }

    // Cross-sentence comparison
    println!("═══ CROSS-SENTENCE COMPARISON ═══\n");
    println!("  {:>2}  {:<40}  {:>6}  {:>6}  {}", "#", "Text", "Peak", "Family", "Unique centroids");
    for (i, (text, centroids, top10, qualia)) in all_results.iter().enumerate() {
        let mut u = centroids.clone(); u.sort(); u.dedup();
        let (family, _) = qualia.nearest_family();
        println!("  {:>2}. {:<40}  {:>6}  {:>6}  {}/{}",
            i + 1,
            &text[..text.len().min(40)],
            top10[0].0,
            family,
            u.len(),
            centroids.len());
    }

    // Do any sentences share centroids?
    println!("\n  Centroid overlap between sentences:");
    for i in 0..all_results.len() {
        for j in (i+1)..all_results.len() {
            let mut a: Vec<u16> = all_results[i].1.clone(); a.sort(); a.dedup();
            let mut b: Vec<u16> = all_results[j].1.clone(); b.sort(); b.dedup();
            let shared: Vec<u16> = a.iter().filter(|x| b.contains(x)).cloned().collect();
            if !shared.is_empty() {
                println!("    S{} ↔ S{}: {} shared centroids {:?}",
                    i+1, j+1, shared.len(), shared);
            }
        }
    }

    // Do any sentences converge to the same peak?
    println!("\n  Peak convergence:");
    let peaks: Vec<usize> = all_results.iter().map(|r| r.2[0].0).collect();
    let unique_peaks: std::collections::HashSet<usize> = peaks.iter().cloned().collect();
    println!("    {} sentences → {} unique peaks: {:?}", peaks.len(), unique_peaks.len(), peaks);
    if unique_peaks.len() == 1 {
        println!("    ⚠ ALL sentences converge to same atom — table topology too uniform");
        println!("    → Need: ffn_down table (wider spread) or HDR grading");
    } else {
        println!("    ✓ Sentences differentiate! Architecture works.");
    }

    println!("\n═══ REALITY CHECK COMPLETE ═══");
}
