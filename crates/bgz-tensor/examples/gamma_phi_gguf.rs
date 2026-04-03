//! Gamma-phi encoding comparison on real Jina GGUF weights.
//!
//! The key test: Up (magnitude 0.004) vs Gate (magnitude 2.3).
//! Does γ+φ encoding give Up more resolution without destroying Gate?
//!
//! cargo run --release --manifest-path crates/bgz-tensor/Cargo.toml --example gamma_phi_gguf

use bgz_tensor::gamma_phi::*;
use bgz_tensor::variance::Role;
use bgz_tensor::stacked_n::{StackedN, cosine_f32_slice};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/tmp/hf_cache/models--gaianet--jina-embeddings-v3-GGUF/snapshots/d7d998aab1a7f2ea0aa256d8b3f035cbd0af682a/jina-embeddings-v3-Q8_0.gguf".into()
    });

    println!("=== Gamma-Phi Encoding on Real GGUF Weights ===\n");

    let mut file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => { eprintln!("Cannot open: {}", e); return; }
    };
    let header = match parse_gguf_header(&mut file) {
        Ok(h) => h,
        Err(e) => { eprintln!("Parse error: {}", e); return; }
    };

    // Collect f32 weight rows by role
    let mut role_rows: HashMap<Role, Vec<Vec<f32>>> = HashMap::new();
    let mut tensors_done = 0;

    for tensor in &header.tensors {
        let role = match Role::from_name(&tensor.name) { Some(r) => r, None => continue };
        if tensor.n_elements < 1024 { continue; }
        let data = match read_tensor_f32(&mut file, &header, tensor) { Ok(d) => d, Err(_) => continue };
        let (n_rows, n_cols) = if tensor.dims.len() >= 2 {
            (tensor.dims[0] as usize, tensor.dims[1..].iter().map(|&d| d as usize).product())
        } else { (1, data.len()) };

        let entry = role_rows.entry(role).or_default();
        for r in 0..n_rows.min(200) {
            let start = r * n_cols;
            let end = (start + n_cols).min(data.len());
            if end > start { entry.push(data[start..end].to_vec()); }
        }
        tensors_done += 1;
        if tensors_done % 20 == 0 { eprint!("\r{} tensors...", tensors_done); }
    }
    eprintln!("\r{} tensors processed\n", tensors_done);

    // ═══ Part 1: Per-role magnitude distribution ═════════════════════════
    println!("=== Part 1: Per-Role Magnitude Distribution ===\n");
    println!("Role  │ Rows │ Mean |v| │ Std |v|  │ Min |v|   │ Max |v|");
    println!("──────┼──────┼──────────┼──────────┼───────────┼──────────");

    let mut all_role_refs: Vec<(&str, Vec<&[f32]>)> = Vec::new();

    for role in Role::ALL {
        if let Some(rows) = role_rows.get(&role) {
            let mags: Vec<f64> = rows.iter().flat_map(|r| r.iter().map(|v| v.abs() as f64)).collect();
            let mean = mags.iter().sum::<f64>() / mags.len() as f64;
            let var = mags.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / mags.len() as f64;
            let min = mags.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = mags.iter().cloned().fold(0.0f64, f64::max);
            println!("{:<5} │ {:>4} │ {:>8.5} │ {:>8.5} │ {:>9.6} │ {:>8.4}",
                role.label(), rows.len(), mean, var.sqrt(), min, max);

            let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
            all_role_refs.push((role.label(), refs));
        }
    }

    // ═══ Part 2: Calibrate gamma profile ═════════════════════════════════
    println!("\n=== Part 2: Calibrate Gamma Profile ===\n");
    let profile = calibrate_gamma("jina-v3-q8_0", &all_role_refs.iter()
        .map(|(name, refs)| (*name, refs.as_slice())).collect::<Vec<_>>());

    println!("Per-role gamma offsets:");
    for (i, name) in ["Q", "K", "V", "Gate", "Up", "Down"].iter().enumerate() {
        println!("  {:<5}: γ = {:.6}", name, profile.role_gamma[i]);
    }
    println!("  φ_scale = {:.6}", profile.phi_scale);
    println!("  Metadata: {} bytes per model\n", GammaProfile::METADATA_BYTES);

    // ═══ Part 3: Compare strategies per role ═════════════════════════════
    println!("=== Part 3: Strategy Comparison Per Role ===\n");
    println!("The key question: does γ+φ help Up/Down (magnitude 0.004-0.15)");
    println!("without hurting Q/Gate (magnitude 0.3-2.3)?\n");

    for role in Role::ALL {
        if let Some(rows) = role_rows.get(&role) {
            if rows.len() < 10 { continue; }
            let role_idx = role as usize;
            let gamma = profile.role_gamma[role_idx];
            let comp = compare_strategies(rows, gamma, profile.phi_scale);
            println!("--- {} (γ={:.4}) ---", role.label(), gamma);
            println!("{}\n", comp.summary());
        }
    }

    // ═══ Part 4: BF16 quantization after gamma-phi encoding ══════════════
    println!("=== Part 4: BF16 Quantization After Gamma-Phi ===\n");
    println!("The real question: when we truncate to BF16 (7-bit mantissa),");
    println!("does γ+φ pre-encoding preserve more information than linear?\n");

    for role in Role::ALL {
        if let Some(rows) = role_rows.get(&role) {
            if rows.len() < 10 { continue; }
            let role_idx = role as usize;
            let gamma = profile.role_gamma[role_idx];
            let sample = rows.len().min(50);

            // Ground truth: f32 pairwise cosines
            let mut gt = Vec::new();
            let mut linear_bf16 = Vec::new();
            let mut gp_bf16 = Vec::new();

            for i in 0..sample {
                for j in (i+1)..sample.min(i+5) {
                    let cos_gt = cosine_f32_slice(&rows[i], &rows[j]);
                    gt.push(cos_gt);

                    // Linear BF16: f32 → BF16 → f32 → cosine
                    let a_lin: Vec<f32> = rows[i].iter().map(|&v| bf16_roundtrip(v)).collect();
                    let b_lin: Vec<f32> = rows[j].iter().map(|&v| bf16_roundtrip(v)).collect();
                    linear_bf16.push(cosine_f32_slice(&a_lin, &b_lin));

                    // γ+φ BF16: f32 → γ+φ encode → BF16 → f32 → γ+φ decode → cosine
                    let a_gp: Vec<f32> = rows[i].iter()
                        .map(|&v| gamma_phi_decode(bf16_roundtrip(gamma_phi_encode(v, gamma, profile.phi_scale)), gamma, profile.phi_scale))
                        .collect();
                    let b_gp: Vec<f32> = rows[j].iter()
                        .map(|&v| gamma_phi_decode(bf16_roundtrip(gamma_phi_encode(v, gamma, profile.phi_scale)), gamma, profile.phi_scale))
                        .collect();
                    gp_bf16.push(cosine_f32_slice(&a_gp, &b_gp));
                }
            }

            let lin_p = bgz_tensor::quality::pearson(&gt, &linear_bf16);
            let lin_s = bgz_tensor::quality::spearman(&gt, &linear_bf16);
            let gp_p = bgz_tensor::quality::pearson(&gt, &gp_bf16);
            let gp_s = bgz_tensor::quality::spearman(&gt, &gp_bf16);

            let improvement_p = gp_p - lin_p;
            let improvement_s = gp_s - lin_s;

            println!("{:<5}: Linear BF16: P={:.6} S={:.6} │ γ+φ BF16: P={:.6} S={:.6} │ Δ P={:+.6} S={:+.6} {}",
                role.label(), lin_p, lin_s, gp_p, gp_s, improvement_p, improvement_s,
                if improvement_p > 0.001 { "BETTER" } else if improvement_p < -0.001 { "WORSE" } else { "~SAME" });
        }
    }

    // ═══ Part 5: Stacked SPD=32 with gamma-phi pre-encoding ══════════════
    println!("\n=== Part 5: Stacked SPD=32 + Gamma-Phi ===\n");
    println!("Does γ+φ before stacking improve the SPD=32 Pearson of 0.996?\n");

    for role in [Role::Q, Role::Up, Role::Down] {
        if let Some(rows) = role_rows.get(&role) {
            if rows.len() < 10 { continue; }
            let role_idx = role as usize;
            let gamma = profile.role_gamma[role_idx];
            let sample = rows.len().min(50);

            let mut gt = Vec::new();
            let mut stacked_linear = Vec::new();
            let mut stacked_gp = Vec::new();

            // Encode both ways at SPD=32
            let enc_lin: Vec<StackedN> = rows[..sample].iter()
                .map(|r| StackedN::from_f32(r, 32)).collect();
            let enc_gp: Vec<StackedN> = rows[..sample].iter()
                .map(|r| {
                    let gp: Vec<f32> = r.iter().map(|&v| gamma_phi_encode(v, gamma, profile.phi_scale)).collect();
                    StackedN::from_f32(&gp, 32)
                }).collect();

            for i in 0..sample {
                for j in (i+1)..sample.min(i+5) {
                    gt.push(cosine_f32_slice(&rows[i], &rows[j]));

                    // Linear stacked: hydrate → cosine
                    let a = enc_lin[i].hydrate_f32();
                    let b = enc_lin[j].hydrate_f32();
                    stacked_linear.push(cosine_f32_slice(&a, &b));

                    // γ+φ stacked: hydrate → decode → cosine
                    let a_h = enc_gp[i].hydrate_f32();
                    let b_h = enc_gp[j].hydrate_f32();
                    let a_dec: Vec<f32> = a_h.iter().map(|&v| gamma_phi_decode(v, gamma, profile.phi_scale)).collect();
                    let b_dec: Vec<f32> = b_h.iter().map(|&v| gamma_phi_decode(v, gamma, profile.phi_scale)).collect();
                    stacked_gp.push(cosine_f32_slice(&a_dec, &b_dec));
                }
            }

            let lin_p = bgz_tensor::quality::pearson(&gt, &stacked_linear);
            let gp_p = bgz_tensor::quality::pearson(&gt, &stacked_gp);
            println!("{:<5}: Stacked linear P={:.6} │ Stacked γ+φ P={:.6} │ Δ={:+.6} {}",
                role.label(), lin_p, gp_p, gp_p - lin_p,
                if gp_p > lin_p + 0.001 { "BETTER" } else if gp_p < lin_p - 0.001 { "WORSE" } else { "~SAME" });
        }
    }

    println!("\n=== DONE ===");
}

fn bf16_roundtrip(v: f32) -> f32 {
    let bits = (v.to_bits() >> 16) as u16;
    f32::from_bits((bits as u32) << 16)
}

// ═══ GGUF reader (reused) ════════════════════════════════════════════════

struct GgufHeader { tensors: Vec<TensorMeta>, data_offset: u64 }
struct TensorMeta { name: String, dims: Vec<u64>, dtype: u32, offset: u64, n_elements: u64 }

fn parse_gguf_header<R: Read + Seek>(r: &mut R) -> Result<GgufHeader, String> {
    let mut b4 = [0u8; 4]; let mut b8 = [0u8; 8];
    r.read_exact(&mut b4).map_err(|e| e.to_string())?;
    if u32::from_le_bytes(b4) != 0x46554747 { return Err("bad magic".into()); }
    r.read_exact(&mut b4).map_err(|e| e.to_string())?;
    r.read_exact(&mut b8).map_err(|e| e.to_string())?; let nt = u64::from_le_bytes(b8) as usize;
    r.read_exact(&mut b8).map_err(|e| e.to_string())?; let nm = u64::from_le_bytes(b8) as usize;
    for _ in 0..nm { skip_kv(r)?; }
    let mut tensors = Vec::with_capacity(nt);
    for _ in 0..nt {
        r.read_exact(&mut b8).map_err(|e| e.to_string())?; let nl = u64::from_le_bytes(b8) as usize;
        let mut nb = vec![0u8; nl]; r.read_exact(&mut nb).map_err(|e| e.to_string())?;
        let name = String::from_utf8_lossy(&nb).to_string();
        r.read_exact(&mut b4).map_err(|e| e.to_string())?; let nd = u32::from_le_bytes(b4) as usize;
        let mut dims = Vec::with_capacity(nd);
        for _ in 0..nd { r.read_exact(&mut b8).map_err(|e| e.to_string())?; dims.push(u64::from_le_bytes(b8)); }
        r.read_exact(&mut b4).map_err(|e| e.to_string())?; let dtype = u32::from_le_bytes(b4);
        r.read_exact(&mut b8).map_err(|e| e.to_string())?; let offset = u64::from_le_bytes(b8);
        tensors.push(TensorMeta { name, dims: dims.clone(), dtype, offset, n_elements: dims.iter().product() });
    }
    let pos = r.stream_position().map_err(|e| e.to_string())?;
    Ok(GgufHeader { tensors, data_offset: (pos + 31) / 32 * 32 })
}
fn skip_kv<R: Read + Seek>(r: &mut R) -> Result<(), String> {
    let mut b4=[0u8;4]; let mut b8=[0u8;8];
    r.read_exact(&mut b8).map_err(|e|e.to_string())?; let kl=u64::from_le_bytes(b8) as usize;
    let mut kb=vec![0u8;kl]; r.read_exact(&mut kb).map_err(|e|e.to_string())?;
    r.read_exact(&mut b4).map_err(|e|e.to_string())?; skip_val(r, u32::from_le_bytes(b4))
}
fn skip_val<R: Read + Seek>(r: &mut R, vt: u32) -> Result<(), String> {
    let mut b4=[0u8;4]; let mut b8=[0u8;8];
    match vt {
        0|1|7 => { let mut b=[0u8;1]; r.read_exact(&mut b).map_err(|e|e.to_string())?; }
        2|3 => { r.read_exact(&mut [0u8;2]).map_err(|e|e.to_string())?; }
        4|5|6 => { r.read_exact(&mut b4).map_err(|e|e.to_string())?; }
        8 => { r.read_exact(&mut b8).map_err(|e|e.to_string())?; let l=u64::from_le_bytes(b8) as usize; let mut s=vec![0u8;l]; r.read_exact(&mut s).map_err(|e|e.to_string())?; }
        9 => { r.read_exact(&mut b4).map_err(|e|e.to_string())?; let et=u32::from_le_bytes(b4); r.read_exact(&mut b8).map_err(|e|e.to_string())?; let c=u64::from_le_bytes(b8) as usize; for _ in 0..c { skip_val(r,et)?; } }
        10|11|12 => { r.read_exact(&mut b8).map_err(|e|e.to_string())?; }
        _ => return Err(format!("unknown vtype {}", vt)),
    }
    Ok(())
}
fn read_tensor_f32<R: Read + Seek>(r: &mut R, h: &GgufHeader, t: &TensorMeta) -> Result<Vec<f32>, String> {
    r.seek(SeekFrom::Start(h.data_offset + t.offset)).map_err(|e|e.to_string())?;
    let n = t.n_elements as usize;
    match t.dtype {
        0 => { let mut buf=vec![0u8;n*4]; r.read_exact(&mut buf).map_err(|e|e.to_string())?; Ok(buf.chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()) }
        8 => { let nb=(n+31)/32; let bpb=34; let mut buf=vec![0u8;nb*bpb]; r.read_exact(&mut buf).map_err(|e|e.to_string())?;
            let mut res=Vec::with_capacity(n); for b in 0..nb { let o=b*bpb; let sb=u16::from_le_bytes([buf[o],buf[o+1]]); let s=f32::from_bits((sb as u32)<<16);
            for i in 0..32 { if res.len()>=n{break;} res.push(buf[o+2+i] as i8 as f32 * s); } } Ok(res) }
        _ => Err(format!("unsupported dtype {}", t.dtype)),
    }
}
