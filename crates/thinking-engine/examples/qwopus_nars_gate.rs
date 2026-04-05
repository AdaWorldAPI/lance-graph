//! Qwopus 27B: gate as upstream NARS perturbation modulator.
//!
//! The gate table modulates NARS truth values before routing:
//! - Gate distance small → high trust (freq↑, conf↑) → strengthen path
//! - Gate distance large → low trust (freq↓, conf↓) → weaken path
//! - Gate near decision boundary → uncertain (conf↓) → explore
//!
//! This turns the gate from a multiplicative filter into a
//! truth-value perturbation that modulates which paths the
//! thinking engine trusts.

use std::time::Instant;
const N: usize = 256;
const N_LAYERS: usize = 64;

fn main() {
    let t0 = Instant::now();
    let dd = "crates/thinking-engine/data/Qwopus3.5-27B-v3-BF16-silu";

    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  Qwopus 27B — Gate as NARS Perturbation Modulator");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    let tokenizer = tokenizers::Tokenizer::from_file(format!("{}/tokenizer.json", dd)).expect("tok");
    let asgn: Vec<u16> = std::fs::read(format!("{}/token_embd_assignments_248320.u16", dd))
        .expect("asgn").chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

    // Load layers: attn + gate + up_silu + down
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
    eprintln!("Loaded {} layers in {:.0}ms\n", layers.len(), t0.elapsed().as_millis());

    let prompts = vec![
        ("The meaning of life is", 0.1, 0.01),     // baseline
        ("The meaning of life is", 0.2, 0.05),     // stronger residual + gate
        ("The meaning of life is", 0.05, 0.001),   // weaker
        ("Artificial intelligence will", 0.1, 0.01),
        ("The cat sat on the", 0.1, 0.01),
        ("In the beginning there was", 0.1, 0.01),
        ("Love is patient love is", 0.1, 0.01),
    ];

    // Run three modes: no gate, gate-as-filter, gate-as-NARS
    eprintln!("{:<40} {:>12} {:>12} {:>12}", "Prompt", "No Gate", "Gate×Filter", "Gate→NARS");
    eprintln!("{}", "─".repeat(80));

    for (prompt, residual_scale, silu_temp) in &prompts {
        let enc = tokenizer.encode(prompt.to_string(), false).expect("enc");
        let ids = enc.get_ids();
        let cidx: Vec<usize> = ids.iter()
            .map(|&id| if (id as usize) < asgn.len() { asgn[id as usize] as usize } else { 0 })
            .collect();

        // Mode 1: No gate (just attn + up + down)
        let top_no_gate = forward_no_gate(&layers, &cidx, *residual_scale);
        
        // Mode 2: Gate as multiplicative filter (current approach)
        let top_gate_filter = forward_gate_filter(&layers, &cidx, *residual_scale, *silu_temp);
        
        // Mode 3: Gate as NARS truth perturbation (new approach)
        let top_gate_nars = forward_gate_nars(&layers, &cidx, *residual_scale);

        // Compare top centroids
        let label = format!("{} (r={},t={})", prompt, residual_scale, silu_temp);
        eprintln!("{:<40} {:>12?} {:>12?} {:>12?}", 
            &label[..label.len().min(39)],
            &top_no_gate[..3], &top_gate_filter[..3], &top_gate_nars[..3]);
    }

    // Detailed comparison for one prompt
    let prompt = "The meaning of life is";
    let enc = tokenizer.encode(prompt.to_string(), false).expect("enc");
    let cidx: Vec<usize> = enc.get_ids().iter()
        .map(|&id| if (id as usize) < asgn.len() { asgn[id as usize] as usize } else { 0 })
        .collect();

    eprintln!("\n=== Detailed: \"{}\" ===\n", prompt);

    for mode in [Mode::NoGate, Mode::GateFilter, Mode::GateNars] {
        let (mode_name, top, energy) = forward_detailed(&layers, &cidx, mode);
        eprintln!("  {} — Top-5:", mode_name);
        for &(ci, e) in top.iter().take(5) {
            let toks: Vec<String> = (0..asgn.len().min(tokenizer.get_vocab_size(true)))
                .filter(|&t| asgn[t] as usize == ci).take(5)
                .filter_map(|t| tokenizer.id_to_token(t as u32).map(|s| s.to_string()))
                .collect();
            eprintln!("    c{:3} e={:.3} ← {:?}", ci, e, toks);
        }

        // Entropy of energy distribution (higher = more spread)
        let total: f32 = energy.iter().filter(|e| **e > 0.0).sum();
        let entropy: f32 = if total > 0.0 {
            -energy.iter()
                .filter(|e| **e > 0.0)
                .map(|e| { let p = e / total; p * p.ln() })
                .sum::<f32>()
        } else { 0.0 };
        eprintln!("    Entropy: {:.3} (higher=more diverse)\n", entropy);
    }

    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  {:.1}s total", t0.elapsed().as_secs_f64());
    eprintln!("═══════════════════════════════════════════════════════════\n");
}

#[derive(Clone, Copy)]
enum Mode { NoGate, GateFilter, GateNars }

fn forward_no_gate(layers: &[[Vec<u8>; 4]], cidx: &[usize], rs: f32) -> Vec<usize> {
    let mut h = init_hidden(cidx);
    for l in 0..N_LAYERS {
        let [ref at, _, ref up, ref dn] = layers[l];
        let mut a = h.clone(); rn(&mut a); a = mv(at, &a);
        for i in 0..N { h[i] += a[i] * rs; }
        let mut f = h.clone(); rn(&mut f);
        let u = mv(up, &f);
        let d = mv(dn, &u);
        for i in 0..N { h[i] += d[i] * rs; }
    }
    rn(&mut h);
    top_k(&h, 5)
}

fn forward_gate_filter(layers: &[[Vec<u8>; 4]], cidx: &[usize], rs: f32, st: f32) -> Vec<usize> {
    let mut h = init_hidden(cidx);
    for l in 0..N_LAYERS {
        let [ref at, ref gt, ref up, ref dn] = layers[l];
        let mut a = h.clone(); rn(&mut a); a = mv(at, &a);
        for i in 0..N { h[i] += a[i] * rs; }
        let mut f = h.clone(); rn(&mut f);
        let g = mv(gt, &f); let u = mv(up, &f);
        let mut gd = vec![0.0f32; N];
        for i in 0..N { gd[i] = (g[i] / (1.0 + (-g[i] * st).exp())) * u[i]; }
        let d = mv(dn, &gd);
        for i in 0..N { h[i] += d[i] * rs; }
    }
    rn(&mut h);
    top_k(&h, 5)
}

fn forward_gate_nars(layers: &[[Vec<u8>; 4]], cidx: &[usize], rs: f32) -> Vec<usize> {
    let mut h = init_hidden(cidx);
    // NARS truth per centroid: frequency (how often active) and confidence
    let mut nars_freq = vec![0.5f32; N];
    let mut nars_conf = vec![0.1f32; N];

    for l in 0..N_LAYERS {
        let [ref at, ref gt, ref up, ref dn] = layers[l];

        // Step 1: Gate perturbs NARS truth values
        // For each active centroid, look at gate topology to modulate trust
        for i in 0..N {
            if h[i].abs() < 1e-6 { continue; }
            // Gate row i tells us how this centroid relates to all others via gate
            let gate_row = &gt[i * N..(i + 1) * N];
            // Average gate distance to other active centroids
            let mut gate_agreement = 0.0f32;
            let mut active_count = 0;
            for j in 0..N {
                if j == i || h[j].abs() < 1e-6 { continue; }
                // High table value = similar gate behavior = agreement
                gate_agreement += gate_row[j] as f32 / 255.0;
                active_count += 1;
            }
            if active_count > 0 {
                gate_agreement /= active_count as f32;
                // Gate agreement modulates NARS truth:
                // High agreement → boost freq+conf (this centroid is trusted)
                // Low agreement → lower conf (uncertain, explore)
                nars_freq[i] = (nars_freq[i] * 0.7 + gate_agreement * 0.3).clamp(0.0, 1.0);
                nars_conf[i] = (nars_conf[i] + gate_agreement * 0.1).clamp(0.0, 0.99);
            }
        }

        // Step 2: NARS-modulated attention
        let mut a = h.clone(); rn(&mut a); a = mv(at, &a);
        // Modulate attention output by NARS expectation
        for i in 0..N {
            let expectation = nars_conf[i] * (nars_freq[i] - 0.5) + 0.5;
            a[i] *= expectation;
        }
        for i in 0..N { h[i] += a[i] * rs; }

        // Step 3: FFN (up + down, no separate gate multiply — gate is in NARS)
        let mut f = h.clone(); rn(&mut f);
        let u = mv(up, &f);
        // NARS confidence gates the FFN output
        let mut gated_u = vec![0.0f32; N];
        for i in 0..N {
            gated_u[i] = u[i] * nars_conf[i]; // confidence = gate strength
        }
        let d = mv(dn, &gated_u);
        for i in 0..N { h[i] += d[i] * rs; }

        // Step 4: Decay NARS confidence slightly (prevents crystallization too fast)
        for i in 0..N { nars_conf[i] *= 0.99; }
    }
    rn(&mut h);
    top_k(&h, 5)
}

fn forward_detailed(layers: &[[Vec<u8>; 4]], cidx: &[usize], mode: Mode) -> (&'static str, Vec<(usize, f32)>, Vec<f32>) {
    let mut h = init_hidden(cidx);
    let mut nars_freq = vec![0.5f32; N];
    let mut nars_conf = vec![0.1f32; N];

    for l in 0..N_LAYERS {
        let [ref at, ref gt, ref up, ref dn] = layers[l];
        match mode {
            Mode::NoGate => {
                let mut a = h.clone(); rn(&mut a); a = mv(at, &a);
                for i in 0..N { h[i] += a[i] * 0.1; }
                let mut f = h.clone(); rn(&mut f);
                let d = mv(dn, &mv(up, &f));
                for i in 0..N { h[i] += d[i] * 0.1; }
            }
            Mode::GateFilter => {
                let mut a = h.clone(); rn(&mut a); a = mv(at, &a);
                for i in 0..N { h[i] += a[i] * 0.1; }
                let mut f = h.clone(); rn(&mut f);
                let g = mv(gt, &f); let u = mv(up, &f);
                let mut gd = vec![0.0f32; N];
                for i in 0..N { gd[i] = (g[i] / (1.0 + (-g[i]*0.01).exp())) * u[i]; }
                let d = mv(dn, &gd);
                for i in 0..N { h[i] += d[i] * 0.1; }
            }
            Mode::GateNars => {
                for i in 0..N {
                    if h[i].abs() < 1e-6 { continue; }
                    let gr = &gt[i*N..(i+1)*N];
                    let mut ga = 0.0f32; let mut ac = 0;
                    for j in 0..N { if j != i && h[j].abs() > 1e-6 {
                        ga += gr[j] as f32 / 255.0; ac += 1; } }
                    if ac > 0 { ga /= ac as f32;
                        nars_freq[i] = (nars_freq[i]*0.7 + ga*0.3).clamp(0.0, 1.0);
                        nars_conf[i] = (nars_conf[i] + ga*0.1).clamp(0.0, 0.99); }
                }
                let mut a = h.clone(); rn(&mut a); a = mv(at, &a);
                for i in 0..N { a[i] *= nars_conf[i]*(nars_freq[i]-0.5)+0.5; }
                for i in 0..N { h[i] += a[i] * 0.1; }
                let mut f = h.clone(); rn(&mut f);
                let u = mv(up, &f);
                let mut gu = vec![0.0f32; N];
                for i in 0..N { gu[i] = u[i] * nars_conf[i]; }
                let d = mv(dn, &gu);
                for i in 0..N { h[i] += d[i] * 0.1; }
                for i in 0..N { nars_conf[i] *= 0.99; }
            }
        }
    }
    rn(&mut h);
    let mut pk: Vec<(usize, f32)> = h.iter().enumerate().map(|(i,&e)| (i,e)).collect();
    pk.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
    let name = match mode { Mode::NoGate => "No Gate", Mode::GateFilter => "Gate Filter", Mode::GateNars => "Gate NARS" };
    (name, pk, h)
}

fn init_hidden(cidx: &[usize]) -> Vec<f32> {
    let mut h = vec![0.0f32; N];
    for &c in cidx { h[c % N] += 1.0; }
    rn(&mut h);
    h
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
fn top_k(e: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<(usize, f32)> = e.iter().enumerate().map(|(i,&v)| (i,v)).collect();
    idx.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
    idx.iter().take(k).map(|p| p.0).collect()
}
fn ld(p: &str) -> Vec<u8> { std::fs::read(p).unwrap_or_else(|_| vec![128u8; N*N]) }
fn ldo(p: &str) -> Option<Vec<u8>> { std::fs::read(p).ok() }
