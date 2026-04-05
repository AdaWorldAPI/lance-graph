//! Qwopus 27B forward pass with REAL Qwen BPE tokenizer.

use std::time::Instant;
const N: usize = 256;
const N_LAYERS: usize = 64;

fn main() {
    let t0 = Instant::now();
    let dd = "crates/thinking-engine/data/Qwopus3.5-27B-v3-BF16-silu";

    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  Qwopus 27B — Real Tokenizer + 64-Layer Forward");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    let tokenizer = tokenizers::Tokenizer::from_file(format!("{}/tokenizer.json", dd))
        .expect("tokenizer");
    eprintln!("Tokenizer: {} vocab", tokenizer.get_vocab_size(true));

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
    eprintln!("Loaded {} layers in {:.0}ms\n", layers.len(), t0.elapsed().as_millis());

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
        eprintln!("  Tokens: {:?}", toks);

        let cidx: Vec<usize> = ids.iter()
            .map(|&id| if (id as usize) < asgn.len() { asgn[id as usize] as usize } else { 0 })
            .collect();

        let mut h = vec![0.0f32; N];
        for &c in &cidx { h[c % N] += 1.0; }
        rn(&mut h);

        let tp = Instant::now();
        for l in 0..N_LAYERS {
            let [ref at, ref gt, ref up, ref dn] = layers[l];
            let mut a = h.clone(); rn(&mut a); a = mv(at, &a);
            for i in 0..N { h[i] += a[i] * 0.1; }
            let mut f = h.clone(); rn(&mut f);
            let g = mv(gt, &f); let u = mv(up, &f);
            let mut gd = vec![0.0f32; N];
            for i in 0..N { gd[i] = (g[i] / (1.0 + (-g[i]*0.01).exp())) * u[i]; }
            let d = mv(dn, &gd);
            for i in 0..N { h[i] += d[i] * 0.1; }
        }
        rn(&mut h);

        let mut pk: Vec<(usize, f32)> = h.iter().enumerate().map(|(i,&e)| (i,e)).collect();
        pk.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        eprintln!("  {:.0}ms | Top-5:", tp.elapsed().as_secs_f64()*1000.0);
        for &(ci, e) in pk.iter().take(5) {
            let m: Vec<String> = (0..asgn.len().min(tokenizer.get_vocab_size(true)))
                .filter(|&t| asgn[t] as usize == ci).take(5)
                .filter_map(|t| tokenizer.id_to_token(t as u32).map(|s| s.to_string()))
                .collect();
            eprintln!("    c{:3} e={:.3} <- {:?}", ci, e, m);
        }
        eprintln!();
    }
    eprintln!("Done in {:.1}s\n", t0.elapsed().as_secs_f64());
}

fn mv(t: &[u8], e: &[f32]) -> Vec<f32> {
    let mut o = vec![0.0f32; N];
    for i in 0..N { if e[i].abs() < 1e-8 { continue; }
        let r = &t[i*N..(i+1)*N];
        for j in 0..N { o[j] += (r[j] as f32 - 128.0) * e[i]; } }
    o
}
fn rn(v: &mut [f32]) {
    let r = (v.iter().map(|x| x*x).sum::<f32>() / v.len() as f32).sqrt();
    if r > 1e-8 { for x in v.iter_mut() { *x /= r; } }
}
fn ld(p: &str) -> Vec<u8> { std::fs::read(p).unwrap_or_else(|_| vec![128u8; N*N]) }
fn ldo(p: &str) -> Option<Vec<u8>> { std::fs::read(p).ok() }
