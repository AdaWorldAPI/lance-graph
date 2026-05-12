//! Validate F32ThinkingEngine with real-world cosine tables.
//!
//! Uses precomputed f32 cosines from 7-lane encoding (real model data).
//! Tests: does the f32 engine produce meaningful, diverse, stable results?
//!
//! cargo run --release --example validate_f32_engine \
//!   --manifest-path crates/thinking-engine/Cargo.toml

fn main() {
    use std::collections::HashSet;

    const N: usize = 256;

    println!("═══════════════════════════════════════════════════════════");
    println!("  VALIDATE F32 ENGINE WITH REAL TEXT-DERIVED TABLES");
    println!("═══════════════════════════════════════════════════════════\n");

    let base = "crates/thinking-engine/data";
    let models = [
        ("Qwen3-VL (2B, 2048D)", format!("{}/qwen3-vl-embedding-7lane", base)),
        ("Jina-v5 (0.6B, 1024D)", format!("{}/jina-v5-7lane", base)),
        ("Reranker-v3 (cross-encoder)", format!("{}/jina-reranker-v3-BF16-7lane", base)),
    ];

    for (model_name, path) in &models {
        let cos_path = format!("{}/cosine_matrix_{}x{}.f32", path, N, N);
        let cos_data = match std::fs::read(&cos_path) {
            Ok(d) => d,
            Err(e) => { eprintln!("Skip {}: {}", model_name, e); continue; }
        };
        let table: Vec<f32> = cos_data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Load codebook index to map token IDs to centroids
        let idx_path = format!("{}/codebook_index.u16", path);
        let idx_data = std::fs::read(&idx_path).unwrap_or_default();
        let codebook_idx: Vec<u16> = idx_data.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();

        println!("── {} ──", model_name);
        println!("  Table: {}×{} f32, codebook: {} tokens\n", N, N, codebook_idx.len());

        // Build F32 engine
        let engine_template = thinking_engine::f32_engine::F32ThinkingEngine::new(table.clone());

        // ═══ TEST 1: Diverse peaks (no attractor collapse) ═══
        let test_queries: Vec<Vec<u16>> = vec![
            vec![10, 20, 30],      // "arbitrary token set A"
            vec![100, 110, 120],   // "arbitrary token set B"
            vec![200, 210, 220],   // "arbitrary token set C"
            vec![50, 150, 250],    // "spread across codebook"
            vec![5, 5, 5],         // "repeated single token"
        ];

        let mut all_peaks: Vec<u16> = Vec::new();
        let mut unique_peaks = HashSet::new();

        println!("  Test 1: Peak diversity (5 queries, 10 cycles each)");
        for (qi, query) in test_queries.iter().enumerate() {
            // Map to centroid indices (mod N for safety)
            let centroids: Vec<u16> = query.iter().map(|&t| {
                if (t as usize) < codebook_idx.len() {
                    codebook_idx[t as usize]
                } else {
                    t % N as u16
                }
            }).collect();

            let mut eng = thinking_engine::f32_engine::F32ThinkingEngine::new(table.clone());
            eng.perturb(&centroids);
            let resonance = eng.think(10);

            let top = eng.top_k(3);
            let peak = top[0].0;
            unique_peaks.insert(peak);
            all_peaks.push(peak);

            println!("    Q{}: centroids {:?} → peak {} (energy {:.4}) cycles={}",
                qi, centroids, peak, top[0].1, eng.cycles);
        }
        let diversity = unique_peaks.len() as f32 / test_queries.len() as f32;
        println!("  → Unique peaks: {}/{} = {:.0}% diversity {}",
            unique_peaks.len(), test_queries.len(), diversity * 100.0,
            if diversity >= 0.6 { "✓ PASS" } else { "✗ FAIL (attractor collapse)" });

        // ═══ TEST 2: Stability (same input → same output) ═══
        println!("\n  Test 2: Stability (same input, 3 runs)");
        let mut results = Vec::new();
        for _ in 0..3 {
            let mut eng = thinking_engine::f32_engine::F32ThinkingEngine::new(table.clone());
            eng.perturb(&[42, 84, 126]);
            eng.think(10);
            results.push(eng.top_k(3));
        }
        let stable = results[0][0].0 == results[1][0].0 && results[1][0].0 == results[2][0].0;
        println!("  → Peaks: {} {} {} {}",
            results[0][0].0, results[1][0].0, results[2][0].0,
            if stable { "✓ DETERMINISTIC" } else { "✗ NON-DETERMINISTIC" });

        // ═══ TEST 3: Entropy trajectory ═══
        println!("\n  Test 3: Entropy trajectory (30 cycles)");
        let mut eng = thinking_engine::f32_engine::F32ThinkingEngine::new(table.clone());
        eng.perturb(&[10, 50, 200]);
        let initial_entropy = eng.entropy();

        let mut entropies = vec![initial_entropy];
        for _ in 0..30 {
            // Manual single cycle
            let prev_energy = eng.energy().to_vec();
            eng.think(1);
            entropies.push(eng.entropy());
            eng.reset();
            // Restore energy for next cycle
            for (i, &e) in prev_energy.iter().enumerate() {
                if e > 0.0 { eng.energy[i] = e; }
            }
        }

        // Actually just run think(30) properly
        let mut eng = thinking_engine::f32_engine::F32ThinkingEngine::new(table.clone());
        eng.perturb(&[10, 50, 200]);
        let h0 = eng.entropy();
        eng.think(5);
        let h5 = eng.entropy();
        let mut eng2 = thinking_engine::f32_engine::F32ThinkingEngine::new(table.clone());
        eng2.perturb(&[10, 50, 200]);
        eng2.think(30);
        let h30 = eng2.entropy();

        println!("    H(0)={:.3}  H(5)={:.3}  H(30)={:.3}", h0, h5, h30);
        let direction = if h30 < h0 { "CONCENTRATES ✓" } else { "DIFFUSES ✗" };
        println!("  → {}", direction);

        // ═══ TEST 4: Plain cosine vs F32 thinking comparison ═══
        println!("\n  Test 4: Plain cosine vs F32 thinking (20 queries)");
        let test_atoms: Vec<usize> = (0..20).map(|i| i * 13 % N).collect();
        let mut overlap_5 = 0;
        let mut think_entropy_sum = 0.0f64;
        let mut plain_entropy_sum = 0.0f64;

        for &query in &test_atoms {
            // Plain cosine ranking
            let mut plain: Vec<(usize, f32)> = (0..N)
                .filter(|&j| j != query)
                .map(|j| (j, table[query * N + j]))
                .collect();
            plain.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // F32 thinking
            let mut eng = thinking_engine::f32_engine::F32ThinkingEngine::new(table.clone());
            eng.perturb(&[query as u16]);
            eng.think(10);
            let mut think: Vec<(usize, f32)> = (0..N)
                .filter(|&j| j != query)
                .map(|j| (j, eng.energy()[j]))
                .collect();
            think.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Top-5 overlap
            let p5: HashSet<usize> = plain[..5].iter().map(|&(j,_)| j).collect();
            let t5: HashSet<usize> = think[..5].iter().map(|&(j,_)| j).collect();
            overlap_5 += p5.intersection(&t5).count();

            // Entropy
            let ph: f64 = plain.iter()
                .map(|&(_, e)| e.max(0.0) as f64)
                .filter(|&e| e > 1e-10)
                .map(|e| { let t = e / plain.iter().map(|&(_,v)| v.max(0.0) as f64).sum::<f64>().max(1e-10); -t * t.ln() })
                .sum();
            plain_entropy_sum += ph;
            think_entropy_sum += eng.entropy() as f64;
        }

        let n_q = test_atoms.len();
        println!("    Top-5 overlap: {}/{} = {:.0}%",
            overlap_5, n_q * 5, overlap_5 as f64 / (n_q * 5) as f64 * 100.0);
        println!("    Plain entropy avg: {:.3}", plain_entropy_sum / n_q as f64);
        println!("    Think entropy avg: {:.3}", think_entropy_sum / n_q as f64);
        let reduction = 1.0 - think_entropy_sum / plain_entropy_sum.max(1e-10);
        println!("    Entropy change: {:.1}%\n", reduction * 100.0);
    }

    println!("═══════════════════════════════════════════════════════════");
}
