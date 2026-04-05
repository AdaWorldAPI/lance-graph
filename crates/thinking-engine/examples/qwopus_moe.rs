//! Qwopus 27B as Mixture of Experts — 4096 experts, top-128 sparse routing.
//!
//! Layer 0: 4096 experts each see the input → top 128 by energy survive
//! Layer 1: 128 survivors run K internal cycles (deep expert processing)
//! Layer 2: expert outputs compose (interference between specialists)
//! Layer 3: composition collapses → next token
//!
//! The gate table IS the router. The per-layer tables ARE the expert internals.
//! Tension drives the loop. Ghost predicts. Free energy measures surprise.

use std::time::Instant;

const N_EXPERTS: usize = 4096;    // total experts (input codebook)
const N_INTERNAL: usize = 256;    // expert internal resolution (per-layer)
const N_LAYERS: usize = 64;       // total layers
const TOP_K: usize = 128;         // experts that fire per token
const EXPERT_DEPTH: usize = 4;    // internal cycles per expert
const MAX_TOKENS: usize = 30;     // max tokens to generate

fn main() {
    let t0 = Instant::now();
    let dd = "crates/thinking-engine/data/Qwopus3.5-27B-v3-BF16-silu";

    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  Qwopus 27B — Mixture of {} Experts, Top-{} Sparse", N_EXPERTS, TOP_K);
    eprintln!("═══════════════════════════════════════════════════════════\n");

    let tokenizer = tokenizers::Tokenizer::from_file(format!("{}/tokenizer.json", dd)).expect("tok");

    // 4096-centroid assignments (expert routing)
    let asgn4k: Vec<u16> = std::fs::read(format!("{}/token_embd_assignments_4096_248320.u16", dd))
        .expect("asgn4k").chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

    // 4096×4096 router table
    let router = std::fs::read(format!("{}/token_embd_4096x4096.u8", dd)).expect("router");
    eprintln!("Router: {}×{}", N_EXPERTS, N_EXPERTS);

    // 256-centroid assignments (for internal expert routing)
    let asgn256: Vec<u16> = std::fs::read(format!("{}/token_embd_assignments_248320.u16", dd))
        .expect("asgn256").chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

    // Layer tables (expert internals)
    let mut layers: Vec<[Vec<u8>; 4]> = Vec::new();
    for l in 0..N_LAYERS {
        let d = format!("{}/layer_{:02}", dd, l);
        layers.push([
            ld(&format!("{}/attn_qkv_256x256.u8", d)),
            ld(&format!("{}/ffn_gate_256x256.u8", d)),
            ldo(&format!("{}/ffn_up_silu_256x256.u8", d))
                .unwrap_or_else(|| ld(&format!("{}/ffn_up_256x256.u8", d))),
            ld(&format!("{}/ffn_down_256x256.u8", d)),
        ]);
    }
    eprintln!("Loaded {} layer tables\n", layers.len());

    let prompts = vec![
        "The meaning of life is",
        "Artificial intelligence will",
        "Once upon a time there was",
    ];

    for prompt in &prompts {
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!("  \"{}\"", prompt);
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let enc = tokenizer.encode(prompt.to_string(), false).expect("enc");
        let prompt_ids: Vec<u32> = enc.get_ids().to_vec();

        let mut context_4k: Vec<usize> = prompt_ids.iter()
            .map(|&id| if (id as usize) < asgn4k.len() { asgn4k[id as usize] as usize } else { 0 })
            .collect();

        let mut ghost = vec![0.0f32; N_EXPERTS];
        let mut generated: Vec<String> = Vec::new();

        eprint!("  {}", prompt);

        for step in 0..MAX_TOKENS {
            // ═══ PHASE 0: EXPERT ROUTING (4096 space) ═══
            // Activate experts based on context
            let mut expert_energy = vec![0.0f32; N_EXPERTS];
            for (pos, &ci) in context_4k.iter().enumerate() {
                let recency = (pos as f32 + 1.0) / context_4k.len() as f32;
                expert_energy[ci % N_EXPERTS] += recency;
            }
            rn_n(&mut expert_energy, N_EXPERTS);

            // Route through 4096×4096 table (who responds to this input?)
            let routed = mv_n(&router, &expert_energy, N_EXPERTS);

            // ═══ PHASE 1: TOP-K EXPERT SELECTION ═══
            let mut ranked: Vec<(usize, f32)> = routed.iter().enumerate()
                .map(|(i, &e)| (i, e)).collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let active_experts: Vec<(usize, f32)> = ranked.iter()
                .take(TOP_K).cloned().collect();

            // ═══ PHASE 2: EXPERT DEEP PROCESSING ═══
            // Each active expert runs EXPERT_DEPTH cycles through its layers
            // Map expert (4096) → internal (256) space
            let mut expert_outputs = vec![0.0f32; N_INTERNAL];

            // Distribute layers among experts:
            // Expert group g gets layers [g*layers_per_group..(g+1)*layers_per_group]
            let groups = 4; // 4 hierarchical meeting points
            let layers_per_group = N_LAYERS / groups;

            for group in 0..groups {
                let layer_start = group * layers_per_group;
                let layer_end = layer_start + layers_per_group;

                // Each expert processes through its assigned layers
                let mut group_output = vec![0.0f32; N_INTERNAL];

                for &(expert_id, expert_weight) in &active_experts {
                    // Map expert_id (4096) → internal position (256)
                    let internal_pos = expert_id % N_INTERNAL;

                    // Expert's internal state: seeded by its activation
                    let mut state = vec![0.0f32; N_INTERNAL];
                    state[internal_pos] = expert_weight;

                    // Run through assigned layers
                    for l in layer_start..layer_end.min(N_LAYERS) {
                        let [ref at, ref gt, ref up, ref dn] = layers[l];

                        // Attention
                        let mut a = state.clone();
                        rn_n(&mut a, N_INTERNAL);
                        a = mv_n(at, &a, N_INTERNAL);
                        for i in 0..N_INTERNAL { state[i] += a[i] * 0.1; }

                        // Gate-modulated FFN
                        let mut f = state.clone();
                        rn_n(&mut f, N_INTERNAL);
                        let g = mv_n(gt, &f, N_INTERNAL);
                        let u = mv_n(up, &f, N_INTERNAL);
                        let mut gd = vec![0.0f32; N_INTERNAL];
                        for i in 0..N_INTERNAL {
                            gd[i] = (g[i] / (1.0 + (-g[i] * 0.01).exp())) * u[i];
                        }
                        let d = mv_n(dn, &gd, N_INTERNAL);
                        for i in 0..N_INTERNAL { state[i] += d[i] * 0.1; }
                    }

                    // Expert contributes its output weighted by its routing score
                    for i in 0..N_INTERNAL {
                        group_output[i] += state[i] * expert_weight;
                    }
                }

                rn_n(&mut group_output, N_INTERNAL);

                // ═══ PHASE 3: EXPERTS MEET ═══
                // Group output feeds into the next group as prior
                for i in 0..N_INTERNAL {
                    expert_outputs[i] += group_output[i] * (1.0 / groups as f32);
                }
            }

            rn_n(&mut expert_outputs, N_INTERNAL);

            // ═══ PHASE 4: COLLAPSE (256 → 4096 → token) ═══
            // Expand internal output back to 4096 expert space
            let mut output_4k = vec![0.0f32; N_EXPERTS];
            for i in 0..N_INTERNAL {
                // Each internal position maps to multiple experts
                let energy = expert_outputs[i];
                for k in 0..(N_EXPERTS / N_INTERNAL) {
                    let eidx = i + k * N_INTERNAL;
                    if eidx < N_EXPERTS { output_4k[eidx] = energy; }
                }
            }
            // Refine with router topology
            output_4k = mv_n(&router, &output_4k, N_EXPERTS);
            rn_n(&mut output_4k, N_EXPERTS);

            // Free energy
            let mut fe = 0.0f32;
            for i in 0..N_EXPERTS { fe += (output_4k[i] - ghost[i]).powi(2); }
            fe = (fe / N_EXPERTS as f32).sqrt();

            // Winner
            let mut peaks: Vec<(usize, f32)> = output_4k.iter().enumerate()
                .map(|(i, &e)| (i, e)).collect();
            peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let winner = peaks[0].0;

            // Token lookup with temperature sampling
            let matching: Vec<u32> = (0..asgn4k.len().min(tokenizer.get_vocab_size(true)))
                .filter(|&t| asgn4k[t] as usize == winner)
                .map(|t| t as u32).collect();

            // Pick token by position in cluster (pseudo-frequency weighting)
            let token_idx = if matching.len() > 1 {
                // Use energy ratio to index into cluster
                let ratio = (peaks[0].1 / (peaks[0].1 + peaks[1].1 + 0.001)).clamp(0.0, 0.99);
                (ratio * matching.len() as f32) as usize
            } else { 0 };

            let tid = matching.get(token_idx).or(matching.first()).copied().unwrap_or(0);
            let tok = tokenizer.id_to_token(tid).unwrap_or_else(|| "?".to_string());
            let display = tok.replace("Ġ", " ").replace("Ċ", "\n");

            eprint!("{}", display);
            generated.push(display.clone());

            // Update ghost + context
            for i in 0..N_EXPERTS { ghost[i] = ghost[i] * 0.7 + output_4k[i] * 0.3; }
            context_4k.push(winner);

            // Stop conditions
            if step % 10 == 0 {
                eprintln!();
                eprintln!("    [step {:2}] fe={:.4} experts={} winner=c{} \"{}\"",
                    step, fe, active_experts.len(), winner, display.trim());
            }

            if (step > 5 && fe < 0.05) || display.contains('\n') || step >= MAX_TOKENS - 1 {
                eprintln!();
                eprintln!("    [STOP at step {}] fe={:.4}", step, fe);
                break;
            }
        }

        eprintln!("\n  Output: {}{}", prompt, generated.join(""));
        eprintln!("  Tokens: {}\n", generated.len());
    }

    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  {:.1}s | {} experts × top-{} × {} internal × {} groups",
        t0.elapsed().as_secs_f64(), N_EXPERTS, TOP_K, N_INTERNAL, 4);
    eprintln!("═══════════════════════════════════════════════════════════\n");
}

fn mv_n(t: &[u8], e: &[f32], n: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; n];
    for i in 0..n { if e[i].abs() < 1e-8 { continue; }
        let r = &t[i*n..(i+1)*n];
        for j in 0..n { o[j] += (r[j] as f32 - 128.0) * e[i]; } }
    o
}
fn rn_n(v: &mut [f32], n: usize) {
    let r = (v[..n].iter().map(|x| x*x).sum::<f32>() / n as f32).sqrt();
    if r > 1e-8 { for x in v[..n].iter_mut() { *x /= r; } }
}
fn ld(p: &str) -> Vec<u8> { std::fs::read(p).unwrap_or_else(|_| vec![128u8; N_INTERNAL*N_INTERNAL]) }
fn ldo(p: &str) -> Option<Vec<u8>> { std::fs::read(p).ok() }
