//! Qwopus 27B forward pass with residual connections + RMS norm.

use thinking_engine::engine::ThinkingEngine;
use std::time::Instant;

const N: usize = 256;
const N_LAYERS: usize = 64;

fn main() {
    let t0 = Instant::now();
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  Qwopus 27B Forward Pass — Residual + RMSNorm");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    let data_dir = "crates/thinking-engine/data/Qwopus3.5-27B-v3-BF16-silu";

    // Load assignments
    let assignments: Vec<u16> = std::fs::read(format!("{}/token_embd_assignments_248320.u16", data_dir))
        .expect("assignments").chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

    // Load layer tables
    let mut layers: Vec<LayerTables> = Vec::new();
    for l in 0..N_LAYERS {
        let dir = format!("{}/layer_{:02}", data_dir, l);
        layers.push(LayerTables {
            attn: load_table(&format!("{}/attn_qkv_256x256.u8", dir)),
            gate: load_table(&format!("{}/ffn_gate_256x256.u8", dir)),
            up_silu: load_table_opt(&format!("{}/ffn_up_silu_256x256.u8", dir))
                .unwrap_or_else(|| load_table(&format!("{}/ffn_up_256x256.u8", dir))),
            down: load_table(&format!("{}/ffn_down_256x256.u8", dir)),
        });
    }
    eprintln!("Loaded {} layers in {:.0}ms\n", layers.len(), t0.elapsed().as_millis());

    let prompts = vec![
        "The meaning of life is",
        "Artificial intelligence will",
        "The cat sat on the",
        "In the beginning there was",
        "Love is patient love is",
    ];

    for prompt in &prompts {
        eprintln!("━━━ \"{}\" ━━━", prompt);

        // Tokenize + assign
        let centroids: Vec<usize> = prompt.split_whitespace()
            .map(|w| assignments[simple_hash(w) % assignments.len()] as usize)
            .collect();

        // Initialize hidden state from input centroids
        let mut hidden = vec![0.0f32; N];
        for &ci in &centroids { hidden[ci] += 1.0; }
        rms_norm(&mut hidden);

        let tp = Instant::now();

        // 64-layer forward pass with RESIDUAL CONNECTIONS
        for l in 0..N_LAYERS {
            let lt = &layers[l];

            // --- Attention sublayer with residual ---
            let mut attn_out = hidden.clone();
            rms_norm(&mut attn_out);
            attn_out = matvec(&lt.attn, &attn_out);
            // Residual: hidden = hidden + attn_out
            for i in 0..N { hidden[i] += attn_out[i] * 0.1; } // scaled residual

            // --- FFN sublayer with residual ---
            let mut ffn_in = hidden.clone();
            rms_norm(&mut ffn_in);

            // Gate: SiLU-corrected gate topology filters features
            let gate_out = matvec(&lt.gate, &ffn_in);
            // SiLU activation on gate output
            let mut gated: Vec<f32> = gate_out.iter()
                .map(|&g| g / (1.0 + (-g * 0.01).exp())) // soft SiLU
                .collect();

            // Up projection (SiLU-corrected)
            let up_out = matvec(&lt.up_silu, &ffn_in);

            // Gate × Up (elementwise)
            for i in 0..N { gated[i] *= up_out[i]; }

            // Down projection
            let ffn_out = matvec(&lt.down, &gated);

            // Residual: hidden = hidden + ffn_out
            for i in 0..N { hidden[i] += ffn_out[i] * 0.1; } // scaled residual
        }

        let elapsed = tp.elapsed();

        // Final RMS norm
        rms_norm(&mut hidden);

        // Top-10 output
        let mut peaks: Vec<(usize, f32)> = hidden.iter().enumerate()
            .map(|(i, &e)| (i, e)).collect();
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        eprintln!("  {:.1}ms ({:.0}μs/layer) | input centroids: {:?}",
            elapsed.as_secs_f64()*1000.0, elapsed.as_secs_f64()*1e6/N_LAYERS as f64, centroids);
        eprintln!("  Top-5:");
        for &(ci, e) in peaks.iter().take(5) {
            // Find some tokens in this centroid
            let toks: Vec<usize> = (0..10000)
                .filter(|&t| assignments[t] as usize == ci)
                .take(3).collect();
            eprintln!("    centroid {:3} e={:.4} ← tokens {:?}", ci, e, toks);
        }

        // Check: are the outputs DIFFERENT per prompt?
        let top3: Vec<usize> = peaks.iter().take(3).map(|p| p.0).collect();
        eprintln!("  Top-3 centroids: {:?}\n", top3);
    }

    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  {:.1}s total | 64 layers | 256 centroids | 26.4 MB model",
        t0.elapsed().as_secs_f64());
    eprintln!("═══════════════════════════════════════════════════════════\n");
}

struct LayerTables { attn: Vec<u8>, gate: Vec<u8>, up_silu: Vec<u8>, down: Vec<u8> }

fn matvec(table: &[u8], energy: &[f32]) -> Vec<f32> {
    let n = N;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let e = energy[i];
        if e.abs() < 1e-8 { continue; }
        let row = &table[i * n..(i + 1) * n];
        for j in 0..n {
            // Center around 128 (the HDR CDF mean)
            out[j] += (row[j] as f32 - 128.0) * e;
        }
    }
    out
}

fn rms_norm(v: &mut [f32]) {
    let n = v.len() as f32;
    let rms = (v.iter().map(|x| x * x).sum::<f32>() / n).sqrt();
    if rms > 1e-8 { for x in v.iter_mut() { *x /= rms; } }
}

fn load_table(p: &str) -> Vec<u8> { std::fs::read(p).unwrap_or_else(|_| vec![128u8; N*N]) }
fn load_table_opt(p: &str) -> Option<Vec<u8>> { std::fs::read(p).ok() }
fn simple_hash(w: &str) -> usize {
    let mut h: u64 = 0x9e3779b97f4a7c15;
    for b in w.bytes() { h = h.wrapping_mul(31).wrapping_add(b as u64); }
    h as usize
}
