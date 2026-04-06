//! End-to-end benchmark: Does thinking beat plain cosine?
//!
//! Tests the core claim: iterative MatVec on a BF16 distance table
//! produces better semantic discrimination than one-shot cosine similarity.
//!
//! Uses precomputed 7-lane data (f32 cosines + codebook assignments).
//! No model loading needed — works from the codebook files.
//!
//! cargo run --release --example benchmark_thinking \
//!   --manifest-path crates/thinking-engine/Cargo.toml

fn main() {
    use std::time::Instant;

    const N: usize = 256;

    println!("═══════════════════════════════════════════════════════════");
    println!("  BENCHMARK: Thinking Engine vs Plain Cosine");
    println!("═══════════════════════════════════════════════════════════\n");

    // Load raw f32 cosines from 7-lane output
    let base = "crates/thinking-engine/data";
    let models = [
        ("Qwen3-VL", format!("{}/qwen3-vl-embedding-7lane", base)),
        ("Jina-v5", format!("{}/jina-v5-7lane", base)),
        ("Reranker-v3", format!("{}/jina-reranker-v3-BF16-7lane", base)),
    ];

    for (model_name, path) in &models {
        println!("\n── {} ──", model_name);

        // Load raw f32 cosines
        let cos_data = std::fs::read(format!("{}/cosine_matrix_{}x{}.f32", path, N, N))
            .expect("cosine matrix");
        let raw_cos: Vec<f32> = cos_data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // ═══ METHOD 1: Plain cosine (one-shot lookup) ═══
        // Given a query atom, rank all others by cosine similarity
        // This is what FAISS/LanceDB do.

        // ═══ METHOD 2: BF16 ThinkingEngine (iterative) ═══
        // Perturb → 10 cycles → extract peaks
        // This is our claim to novelty.

        // Build BF16 engine from raw cosines
        let bf16_table: Vec<u16> = raw_cos.iter()
            .map(|&c| (c.to_bits() >> 16) as u16)
            .collect();

        // Test: for 20 random "query" atoms, compare rankings
        let test_atoms: Vec<usize> = (0..20).map(|i| i * 13 % N).collect();

        let mut plain_rankings = Vec::new();
        let mut think_rankings = Vec::new();

        let start = Instant::now();
        for &query in &test_atoms {
            // Plain cosine: sort by raw_cos[query][j]
            let mut ranked: Vec<(usize, f32)> = (0..N)
                .filter(|&j| j != query)
                .map(|j| (j, raw_cos[query * N + j]))
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            plain_rankings.push(ranked.iter().map(|&(j, _)| j).collect::<Vec<_>>());
        }
        let plain_time = start.elapsed();

        let start = Instant::now();
        for &query in &test_atoms {
            // ThinkingEngine: perturb + think
            let mut energy = vec![0.0f32; N];
            energy[query] = 1.0;

            // 10 cycles of MatVec
            for _cycle in 0..10 {
                let mut next = vec![0.0f32; N];
                for i in 0..N {
                    if energy[i] < 1e-10 { continue; }
                    for j in 0..N {
                        let bf = bf16_table[i * N + j];
                        let cos = f32::from_bits((bf as u32) << 16);
                        if cos > 0.0 {
                            next[j] += cos * energy[i];
                        }
                    }
                }
                // Normalize
                let total: f32 = next.iter().sum();
                if total > 1e-10 {
                    let inv = 1.0 / total;
                    for e in &mut next { *e *= inv; }
                }
                energy = next;
            }

            // Extract ranking from energy
            let mut ranked: Vec<(usize, f32)> = (0..N)
                .filter(|&j| j != query)
                .map(|j| (j, energy[j]))
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            think_rankings.push(ranked.iter().map(|&(j, _)| j).collect::<Vec<_>>());
        }
        let think_time = start.elapsed();

        // ═══ COMPARE: How different are the rankings? ═══
        let mut rank_agreement_top5 = 0;
        let mut rank_agreement_top10 = 0;
        let mut rank_agreement_top20 = 0;
        let mut total_kendall_tau = 0.0;

        for (plain, think) in plain_rankings.iter().zip(think_rankings.iter()) {
            // Top-K overlap
            let plain5: std::collections::HashSet<_> = plain[..5].iter().collect();
            let think5: std::collections::HashSet<_> = think[..5].iter().collect();
            rank_agreement_top5 += plain5.intersection(&think5).count();

            let plain10: std::collections::HashSet<_> = plain[..10].iter().collect();
            let think10: std::collections::HashSet<_> = think[..10].iter().collect();
            rank_agreement_top10 += plain10.intersection(&think10).count();

            let plain20: std::collections::HashSet<_> = plain[..20].iter().collect();
            let think20: std::collections::HashSet<_> = think[..20].iter().collect();
            rank_agreement_top20 += plain20.intersection(&think20).count();

            // Kendall tau on top-20
            let mut concordant = 0usize;
            let mut discordant = 0usize;
            for i in 0..20.min(plain.len()) {
                for j in (i+1)..20.min(plain.len()) {
                    let p_order = plain.iter().position(|&x| x == plain[i])
                        .unwrap_or(999)
                        .cmp(&plain.iter().position(|&x| x == plain[j]).unwrap_or(999));
                    let t_pos_i = think.iter().position(|&x| x == plain[i]).unwrap_or(999);
                    let t_pos_j = think.iter().position(|&x| x == plain[j]).unwrap_or(999);
                    let t_order = t_pos_i.cmp(&t_pos_j);
                    if p_order == t_order { concordant += 1; }
                    else { discordant += 1; }
                }
            }
            let tau = if concordant + discordant > 0 {
                (concordant as f64 - discordant as f64) / (concordant + discordant) as f64
            } else { 1.0 };
            total_kendall_tau += tau;
        }

        let n_queries = test_atoms.len();
        println!("  Plain cosine:  {:>6.1}ms ({:.0}μs/query)",
            plain_time.as_secs_f64() * 1000.0,
            plain_time.as_secs_f64() * 1_000_000.0 / n_queries as f64);
        println!("  10-cycle think: {:>5.1}ms ({:.0}μs/query)",
            think_time.as_secs_f64() * 1000.0,
            think_time.as_secs_f64() * 1_000_000.0 / n_queries as f64);
        println!();
        println!("  Top-5 overlap:  {:.0}% ({}/{})",
            rank_agreement_top5 as f64 / (n_queries * 5) as f64 * 100.0,
            rank_agreement_top5, n_queries * 5);
        println!("  Top-10 overlap: {:.0}% ({}/{})",
            rank_agreement_top10 as f64 / (n_queries * 10) as f64 * 100.0,
            rank_agreement_top10, n_queries * 10);
        println!("  Top-20 overlap: {:.0}% ({}/{})",
            rank_agreement_top20 as f64 / (n_queries * 20) as f64 * 100.0,
            rank_agreement_top20, n_queries * 20);
        println!("  Kendall τ (top-20): {:.4}",
            total_kendall_tau / n_queries as f64);

        // ═══ KEY QUESTION: Does thinking CHANGE the ranking? ═══
        // If overlap is ~100%, thinking adds nothing.
        // If overlap is ~50% with BETTER results, thinking helps.
        // If overlap is ~50% with WORSE results, thinking hurts.

        // Check: does thinking concentrate energy (lower entropy)?
        let mut plain_entropy_sum = 0.0f64;
        let mut think_entropy_sum = 0.0f64;
        for &query in &test_atoms {
            // Plain: cosine distribution
            let mut probs: Vec<f64> = (0..N)
                .filter(|&j| j != query)
                .map(|j| raw_cos[query * N + j].max(0.0) as f64)
                .collect();
            let total: f64 = probs.iter().sum();
            if total > 1e-10 {
                for p in &mut probs { *p /= total; }
            }
            let h: f64 = probs.iter()
                .filter(|&&p| p > 1e-10)
                .map(|&p| -p * p.ln())
                .sum();
            plain_entropy_sum += h;

            // Think: energy distribution after 10 cycles
            let mut energy = vec![0.0f32; N];
            energy[query] = 1.0;
            for _ in 0..10 {
                let mut next = vec![0.0f32; N];
                for i in 0..N {
                    if energy[i] < 1e-10 { continue; }
                    for j in 0..N {
                        let bf = bf16_table[i * N + j];
                        let cos = f32::from_bits((bf as u32) << 16);
                        if cos > 0.0 { next[j] += cos * energy[i]; }
                    }
                }
                let total: f32 = next.iter().sum();
                if total > 1e-10 { let inv = 1.0 / total; for e in &mut next { *e *= inv; } }
                energy = next;
            }
            let h: f64 = energy.iter()
                .filter(|&&e| e > 1e-10)
                .map(|&e| { let p = e as f64; -p * p.ln() })
                .sum();
            think_entropy_sum += h;
        }
        println!();
        println!("  Plain entropy:  {:.3} (higher = more uniform = less focused)",
            plain_entropy_sum / n_queries as f64);
        println!("  Think entropy:  {:.3} (lower = more focused = better discrimination)",
            think_entropy_sum / n_queries as f64);
        let reduction = 1.0 - think_entropy_sum / plain_entropy_sum;
        println!("  Entropy reduction: {:.1}%", reduction * 100.0);
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  INTERPRETATION:");
    println!("  - Top-K overlap ~100%: thinking agrees with plain cosine");
    println!("  - Entropy reduction > 0: thinking concentrates energy");
    println!("  - If both: thinking is a REFINEMENT (not revolution)");
    println!("  - If entropy up: thinking DIFFUSES (hurts, not helps)");
    println!("═══════════════════════════════════════════════════════════");
}
