//! Qwopus 27B LIVING loop — tension-driven autopoietic thought.
//!
//! The 16M compositions per cycle create interference patterns.
//! Peaks = tension that demands resolution.
//! Ghost predicts next state → free energy = prediction error.
//! High free energy → MUST keep composing.
//! Resolution of tension IS the thought.
//! System drives itself until tension settles.

use std::time::Instant;
const N: usize = 256;
const N_LAYERS: usize = 64;
const MAX_THOUGHTS: usize = 50;  // max tokens to generate

fn main() {
    let t0 = Instant::now();
    let dd = "crates/thinking-engine/data/Qwopus3.5-27B-v3-BF16-silu";

    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  Qwopus 27B — Living Thought (Tension-Driven)");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    let tokenizer = tokenizers::Tokenizer::from_file(format!("{}/tokenizer.json", dd)).expect("tok");
    let asgn: Vec<u16> = std::fs::read(format!("{}/token_embd_assignments_248320.u16", dd))
        .expect("asgn").chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

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
    eprintln!("Loaded {} layers\n", layers.len());

    let prompts = vec![
        "The meaning of life is",
        "Artificial intelligence will",
        "Once upon a time there was",
    ];

    for prompt in &prompts {
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        eprintln!("  \"{}\"", prompt);
        eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        // Encode prompt
        let enc = tokenizer.encode(prompt.to_string(), false).expect("enc");
        let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
        let prompt_centroids: Vec<usize> = prompt_ids.iter()
            .map(|&id| if (id as usize) < asgn.len() { asgn[id as usize] as usize } else { 0 })
            .collect();

        // === THE LIVING STATE ===
        let mut context = prompt_centroids.clone(); // growing context window
        let mut ghost = vec![0.0f32; N];            // ghost prediction (what SHOULD come next)
        let mut generated_tokens: Vec<String> = Vec::new();
        let mut total_free_energy = 0.0f32;

        eprint!("  {}", prompt);

        for step in 0..MAX_THOUGHTS {
            // 1. BUILD HIDDEN STATE from full context (not just last token)
            let mut hidden = vec![0.0f32; N];
            // Recency-weighted: recent context contributes more
            for (pos, &ci) in context.iter().enumerate() {
                let recency = (pos as f32 + 1.0) / context.len() as f32; // 0→1
                hidden[ci % N] += recency;
            }
            rn(&mut hidden);

            // 2. FORWARD PASS through 64 layers
            // Gate modulates as NARS truth perturbation
            let mut nars_conf = vec![0.3f32; N]; // start with moderate confidence

            for l in 0..N_LAYERS {
                let [ref at, ref gt, ref up, ref dn] = layers[l];

                // Gate → NARS truth
                for i in 0..N {
                    if hidden[i].abs() < 1e-6 { continue; }
                    let gr = &gt[i*N..(i+1)*N];
                    let mut agree = 0.0f32;
                    let mut cnt = 0;
                    for j in 0..N {
                        if j != i && hidden[j].abs() > 1e-6 {
                            agree += gr[j] as f32 / 255.0;
                            cnt += 1;
                        }
                    }
                    if cnt > 0 {
                        agree /= cnt as f32;
                        nars_conf[i] = (nars_conf[i] + agree * 0.05).clamp(0.0, 0.99);
                    }
                }

                // Attention (NARS-modulated)
                let mut a = hidden.clone();
                rn(&mut a);
                a = mv(at, &a);
                for i in 0..N { a[i] *= nars_conf[i]; }
                for i in 0..N { hidden[i] += a[i] * 0.1; }

                // FFN (confidence-gated)
                let mut f = hidden.clone();
                rn(&mut f);
                let u = mv(up, &f);
                let mut gu = vec![0.0f32; N];
                for i in 0..N { gu[i] = u[i] * nars_conf[i]; }
                let d = mv(dn, &gu);
                for i in 0..N { hidden[i] += d[i] * 0.1; }

                // Confidence decay
                for i in 0..N { nars_conf[i] *= 0.995; }
            }
            rn(&mut hidden);

            // 3. FREE ENERGY = surprise = |hidden - ghost|
            let mut free_energy = 0.0f32;
            for i in 0..N {
                free_energy += (hidden[i] - ghost[i]).powi(2);
            }
            free_energy = (free_energy / N as f32).sqrt();

            // 4. COLLAPSE: pick the winner
            let mut peaks: Vec<(usize, f32)> = hidden.iter().enumerate()
                .map(|(i, &e)| (i, e)).collect();
            peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let winner_centroid = peaks[0].0;
            let winner_energy = peaks[0].1;
            let runner_up_energy = peaks[1].1;
            let collapse_confidence = if runner_up_energy > 0.0 {
                1.0 - (runner_up_energy / winner_energy)
            } else { 1.0 };

            // 5. REVERSE LOOKUP: centroid → token
            // Pick the most likely token in this centroid based on frequency
            let matching_tokens: Vec<u32> = (0..asgn.len().min(tokenizer.get_vocab_size(true)))
                .filter(|&t| asgn[t] as usize == winner_centroid)
                .map(|t| t as u32)
                .collect();

            let token_str = if let Some(&tid) = matching_tokens.first() {
                tokenizer.id_to_token(tid).unwrap_or_else(|| "?".to_string())
            } else {
                "?".to_string()
            };

            // Clean up BPE artifacts for display
            let display = token_str.replace("Ġ", " ").replace("Ċ", "\n");
            eprint!("{}", display);
            generated_tokens.push(display.clone());

            // 6. UPDATE GHOST (prediction for next step)
            // Ghost = exponential moving average of recent hidden states
            for i in 0..N {
                ghost[i] = ghost[i] * 0.7 + hidden[i] * 0.3;
            }

            // 7. FEED BACK: winner enters context
            context.push(winner_centroid);

            // 8. SHOULD WE STOP?
            total_free_energy += free_energy;
            let avg_fe = total_free_energy / (step + 1) as f32;

            // Stop conditions:
            // - Free energy drops below threshold (thought resolved)
            // - Collapse confidence too low (confused, stop)
            // - Generated enough tokens
            let should_stop = (step > 5 && free_energy < avg_fe * 0.3)  // energy settled
                || (step > 3 && collapse_confidence < 0.001)  // can't decide
                || display.contains('\n')  // natural break
                || step >= MAX_THOUGHTS - 1;

            if step % 10 == 0 || should_stop {
                eprintln!();
                eprintln!("    [step {:2}] fe={:.4} avg_fe={:.4} conf={:.4} c={} \"{}\"",
                    step, free_energy, avg_fe, collapse_confidence, winner_centroid, display.trim());
            }

            if should_stop {
                eprintln!("    [STOP] {}", if free_energy < avg_fe * 0.3 {
                    "tension resolved"
                } else if collapse_confidence < 0.001 {
                    "confused (low collapse confidence)"
                } else if display.contains('\n') {
                    "natural break"
                } else {
                    "max tokens"
                });
                break;
            }
        }

        eprintln!();
        eprintln!("  Generated: {}", generated_tokens.join(""));
        eprintln!("  Tokens: {}", generated_tokens.len());
        eprintln!("  Avg free energy: {:.4}", total_free_energy / generated_tokens.len().max(1) as f32);
        eprintln!();
    }

    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  {:.1}s total", t0.elapsed().as_secs_f64());
    eprintln!("═══════════════════════════════════════════════════════════\n");
}

fn mv(t: &[u8], e: &[f32]) -> Vec<f32> {
    let mut o = vec![0.0f32; N];
    for i in 0..N { if e[i].abs() < 1e-8 { continue; }
        let r = &t[i*N..(i+1)*N];
        for j in 0..N { o[j] += (r[j] as f32 - 128.0) * e[i]; } }
    o
}
fn rn(v: &mut [f32]) {
    let r = (v.iter().map(|x| x*x).sum::<f32>() / N as f32).sqrt();
    if r > 1e-8 { for x in v.iter_mut() { *x /= r; } }
}
fn ld(p: &str) -> Vec<u8> { std::fs::read(p).unwrap_or_else(|_| vec![128u8; N*N]) }
fn ldo(p: &str) -> Option<Vec<u8>> { std::fs::read(p).ok() }
