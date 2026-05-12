//! Qwopus 27B: 4096-centroid input → 64-layer MatVec → token output.

use std::time::Instant;
const N_INPUT: usize = 4096;  // input codebook resolution
const N_LAYER: usize = 256;   // per-layer routing resolution  
const N_LAYERS: usize = 64;

fn main() {
    let t0 = Instant::now();
    let dd = "crates/thinking-engine/data/Qwopus3.5-27B-v3-BF16-silu";
    
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  Qwopus 27B — 4096 centroids × 64 layers");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    // Real tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(format!("{}/tokenizer.json", dd)).expect("tok");
    eprintln!("Tokenizer: {} vocab", tokenizer.get_vocab_size(true));

    // 4096-centroid assignments
    let asgn4k: Vec<u16> = std::fs::read(format!("{}/token_embd_assignments_4096_248320.u16", dd))
        .expect("asgn4k").chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

    // 4096×4096 input distance table
    let input_table = std::fs::read(format!("{}/token_embd_4096x4096.u8", dd)).expect("input table");
    eprintln!("Input table: {}×{}", N_INPUT, N_INPUT);

    // 256-centroid assignments (for layer routing)
    let asgn256: Vec<u16> = std::fs::read(format!("{}/token_embd_assignments_248320.u16", dd))
        .expect("asgn256").chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

    // Layer tables (256×256)
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
    eprintln!("Loaded in {:.0}ms\n", t0.elapsed().as_millis());

    let prompts = vec![
        "The meaning of life is",
        "Artificial intelligence will",
        "The cat sat on the",
        "In the beginning there was",
        "Love is patient love is",
        "Palantir developed Gotham for the",
        "The wound is the place where the light",
    ];

    for prompt in &prompts {
        let enc = tokenizer.encode(prompt.to_string(), false).expect("enc");
        let ids = enc.get_ids();
        let toks: Vec<&str> = enc.get_tokens().iter().map(|s| s.as_str()).collect();

        eprintln!("━━━ \"{}\" ━━━", prompt);
        eprintln!("  BPE: {:?}", toks);

        // Phase 1: 4096-centroid input activation
        let input_centroids: Vec<usize> = ids.iter()
            .map(|&id| if (id as usize) < asgn4k.len() { asgn4k[id as usize] as usize } else { 0 })
            .collect();

        let mut input_energy = vec![0.0f32; N_INPUT];
        for &ci in &input_centroids { input_energy[ci % N_INPUT] += 1.0; }
        rn_n(&mut input_energy, N_INPUT);

        // Input MatVec: spread through 4096-centroid topology
        let mut activated = mv_n(&input_table, &input_energy, N_INPUT);
        rn_n(&mut activated, N_INPUT);

        // Phase 2: Project 4096 → 256 for layer routing
        // Average the 4096 activations into 256 buckets (16:1 fold)
        let mut hidden = vec![0.0f32; N_LAYER];
        for i in 0..N_INPUT {
            hidden[i % N_LAYER] += activated[i];
        }
        rn_n(&mut hidden, N_LAYER);

        // Phase 3: 64-layer residual forward pass (256×256)
        let tp = Instant::now();
        for l in 0..N_LAYERS {
            let [ref at, ref gt, ref up, ref dn] = layers[l];
            let mut a = hidden.clone(); rn_n(&mut a, N_LAYER); a = mv_n(at, &a, N_LAYER);
            for i in 0..N_LAYER { hidden[i] += a[i] * 0.1; }
            let mut f = hidden.clone(); rn_n(&mut f, N_LAYER);
            let g = mv_n(gt, &f, N_LAYER); let u = mv_n(up, &f, N_LAYER);
            let mut gd = vec![0.0f32; N_LAYER];
            for i in 0..N_LAYER { gd[i] = (g[i] / (1.0 + (-g[i]*0.01).exp())) * u[i]; }
            let d = mv_n(dn, &gd, N_LAYER);
            for i in 0..N_LAYER { hidden[i] += d[i] * 0.1; }
        }
        rn_n(&mut hidden, N_LAYER);

        // Phase 4: Expand 256 → 4096 for output token lookup
        let mut output = vec![0.0f32; N_INPUT];
        for i in 0..N_LAYER {
            // Each 256-centroid maps to 16 4096-centroids
            for k in 0..16 {
                let out_idx = i * 16 + k;
                if out_idx < N_INPUT { output[out_idx] = hidden[i]; }
            }
        }
        // Refine with input table topology
        output = mv_n(&input_table, &output, N_INPUT);
        rn_n(&mut output, N_INPUT);

        let fwd = tp.elapsed();

        // Top-10 output centroids → reverse token lookup
        let mut peaks: Vec<(usize, f32)> = output.iter().enumerate()
            .map(|(i, &e)| (i, e)).collect();
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        eprintln!("  {:.0}ms | Top-5 predicted tokens:", fwd.as_secs_f64()*1000.0);
        for &(ci, e) in peaks.iter().take(5) {
            let matching: Vec<String> = (0..asgn4k.len().min(tokenizer.get_vocab_size(true)))
                .filter(|&t| asgn4k[t] as usize == ci)
                .take(8)
                .filter_map(|t| tokenizer.id_to_token(t as u32).map(|s| s.to_string()))
                .collect();
            eprintln!("    c{:4} e={:.3} ← {:?}", ci, e, matching);
        }
        eprintln!();
    }

    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  {:.1}s | 4096 input → 64×256 layers → 4096 output",
        t0.elapsed().as_secs_f64());
    eprintln!("═══════════════════════════════════════════════════════════\n");
}

fn mv_n(t: &[u8], e: &[f32], n: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; n];
    for i in 0..n { if e[i].abs() < 1e-8 { continue; }
        let row = &t[i*n..(i+1)*n];
        for j in 0..n { o[j] += (row[j] as f32 - 128.0) * e[i]; } }
    o
}
fn rn_n(v: &mut [f32], n: usize) {
    let r = (v[..n].iter().map(|x| x*x).sum::<f32>() / n as f32).sqrt();
    if r > 1e-8 { for x in v[..n].iter_mut() { *x /= r; } }
}
fn ld(p: &str) -> Vec<u8> { std::fs::read(p).unwrap_or_else(|_| vec![128u8; N_LAYER*N_LAYER]) }
fn ldo(p: &str) -> Option<Vec<u8>> { std::fs::read(p).ok() }
