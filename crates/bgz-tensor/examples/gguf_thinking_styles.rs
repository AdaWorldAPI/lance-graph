//! Cross-model thinking style fingerprint comparison on real GGUF weight data.
//!
//! cargo run --release --manifest-path crates/bgz-tensor/Cargo.toml --example gguf_thinking_styles

use bgz_tensor::neuron_hetero::{ThinkingStyleFingerprint, TransformSpectrum};
use bgz_tensor::variance::Role;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

fn main() {
    let jina_path = "/tmp/hf_cache/models--gaianet--jina-embeddings-v3-GGUF/snapshots/d7d998aab1a7f2ea0aa256d8b3f035cbd0af682a/jina-embeddings-v3-Q8_0.gguf";
    // Also check for OpenChat/Llama4 bgz7 files for cross-model comparison
    let bgz7_paths = vec![
        "/home/user/ndarray/src/hpc/openchat/weights/openchat-3.5-0106.bgz7",
        "/home/user/ndarray/src/hpc/openchat/weights/llama4_scout_shard1.bgz7",
    ];

    println!("=== Cross-Model Thinking Style Comparison ===\n");

    // ═══ Jina v3 GGUF ═══════════════════════════════════════════════════
    if let Ok(mut file) = std::fs::File::open(jina_path) {
        if let Ok(header) = parse_gguf_header(&mut file) {
            println!("--- Jina v3 ({} tensors) ---\n", header.tensors.len());
            let mut role_fingerprints: HashMap<&str, Vec<ThinkingStyleFingerprint>> = HashMap::new();
            let mut role_spectra: HashMap<&str, Vec<TransformSpectrum>> = HashMap::new();
            let mut up_rows: Vec<Vec<f32>> = Vec::new();
            let mut down_rows: Vec<Vec<f32>> = Vec::new();

            for tensor in &header.tensors {
                let role = Role::from_name(&tensor.name);
                if tensor.n_elements < 1024 { continue; }

                let data = match read_tensor_f32(&mut file, &header, tensor) {
                    Ok(d) => d,
                    Err(_) => continue,
                };

                let (n_rows, n_cols) = if tensor.dims.len() >= 2 {
                    (tensor.dims[0] as usize, tensor.dims[1..].iter().map(|&d| d as usize).product())
                } else { (1, data.len()) };

                let limit = n_rows.min(200);
                for r in 0..limit {
                    let start = r * n_cols;
                    let end = (start + n_cols).min(data.len());
                    if end <= start { continue; }
                    let row = &data[start..end];

                    match role {
                        Some(Role::Q) => {
                            role_fingerprints.entry("Q").or_default()
                                .push(ThinkingStyleFingerprint::from_gate_weights(row));
                        }
                        Some(Role::K) => {
                            role_fingerprints.entry("K").or_default()
                                .push(ThinkingStyleFingerprint::from_gate_weights(row));
                        }
                        Some(Role::V) => {
                            role_fingerprints.entry("V").or_default()
                                .push(ThinkingStyleFingerprint::from_gate_weights(row));
                        }
                        Some(Role::Gate) => {
                            role_fingerprints.entry("Gate").or_default()
                                .push(ThinkingStyleFingerprint::from_gate_weights(row));
                        }
                        Some(Role::Up) => {
                            role_fingerprints.entry("Up").or_default()
                                .push(ThinkingStyleFingerprint::from_gate_weights(row));
                            up_rows.push(row.to_vec());
                        }
                        Some(Role::Down) => {
                            role_fingerprints.entry("Down").or_default()
                                .push(ThinkingStyleFingerprint::from_gate_weights(row));
                            down_rows.push(row.to_vec());
                        }
                        _ => {}
                    }
                }
            }

            // Per-role fingerprint statistics
            println!("Per-role fingerprint profiles:");
            for role_name in &["Q", "K", "V", "Gate", "Up", "Down"] {
                if let Some(fps) = role_fingerprints.get(role_name) {
                    if fps.is_empty() { continue; }
                    // Compute mean hamming within role (intra-role diversity)
                    let sample = fps.len().min(100);
                    let mut intra_hamming = Vec::new();
                    for i in 0..sample {
                        for j in (i+1)..sample.min(i+10) {
                            intra_hamming.push(fps[i].bit_disagreements(&fps[j]) as f64);
                        }
                    }
                    let mean_intra = intra_hamming.iter().sum::<f64>() / intra_hamming.len().max(1) as f64;
                    println!("  {:<5}: {} fingerprints, intra-hamming={:.1}, profile: {}",
                        role_name, fps.len(), mean_intra, fps[0].profile());
                }
            }

            // Cross-role hamming (inter-role distance)
            println!("\nCross-role Hamming distances:");
            let role_names: Vec<&&str> = role_fingerprints.keys().collect();
            for i in 0..role_names.len() {
                for j in (i+1)..role_names.len() {
                    let fps_a = &role_fingerprints[role_names[i]];
                    let fps_b = &role_fingerprints[role_names[j]];
                    let sample = fps_a.len().min(fps_b.len()).min(50);
                    if sample == 0 { continue; }
                    let mut inter = Vec::new();
                    for k in 0..sample {
                        inter.push(fps_a[k].bit_disagreements(&fps_b[k]) as f64);
                    }
                    let mean = inter.iter().sum::<f64>() / inter.len() as f64;
                    println!("  {} <-> {}: mean hamming = {:.1}", role_names[i], role_names[j], mean);
                }
            }

            // Up×Down transform spectra
            let shared = up_rows.len().min(down_rows.len()).min(200);
            if shared > 0 {
                println!("\nUp×Down TransformSpectrum ({} pairs):", shared);
                let spectra: Vec<TransformSpectrum> = (0..shared)
                    .map(|i| TransformSpectrum::from_up_down(&up_rows[i], &down_rows[i]))
                    .collect();

                let mean_rank = spectra.iter().map(|s| s.effective_rank as f64).sum::<f64>() / shared as f64;
                let mean_conc = spectra.iter().map(|s| s.energy_concentration as f64 / 255.0).sum::<f64>() / shared as f64;
                let mean_corr = spectra.iter().map(|s| s.up_down_correlation as f64 / 127.0).sum::<f64>() / shared as f64;
                println!("  mean_rank={:.0}, mean_concentration={:.1}%, mean_corr={:.3}",
                    mean_rank, mean_conc * 100.0, mean_corr);
                println!("  sample: {}", spectra[0].summary());
            }
        }
    } else {
        println!("Jina GGUF not found at {}", jina_path);
    }

    // ═══ bgz7 models (OpenChat, Llama4) ═════════════════════════════════
    for bgz7_path in &bgz7_paths {
        if !std::path::Path::new(bgz7_path).exists() { continue; }

        let data = match std::fs::read(bgz7_path) {
            Ok(d) => d,
            Err(_) => continue,
        };

        if data.len() < 8 || &data[0..4] != b"BGZ7" { continue; }
        let n_tensors = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        let model_name = std::path::Path::new(bgz7_path)
            .file_stem().map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        println!("\n--- {} ({} tensors, bgz7) ---\n", model_name, n_tensors);

        // Parse bgz7 and generate fingerprints from Base17 dims
        let mut pos = 8;
        let mut role_fps: HashMap<&str, Vec<ThinkingStyleFingerprint>> = HashMap::new();
        let mut count = 0;

        for _ in 0..n_tensors {
            if pos + 4 > data.len() { break; }
            let name_len = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
            pos += 4;
            if pos + name_len > data.len() { break; }
            let name = String::from_utf8_lossy(&data[pos..pos+name_len]).to_string();
            pos += name_len;
            if pos + 9 > data.len() { break; }
            pos += 1; // layer_type
            let n_rows = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
            pos += 4;
            pos += 4; // n_cols

            let role_label = if name.contains("q_proj") || name.contains("attn_q") { Some("Q") }
                else if name.contains("k_proj") || name.contains("attn_k") { Some("K") }
                else if name.contains("v_proj") || name.contains("attn_v") { Some("V") }
                else if name.contains("gate_proj") || name.contains("ffn_gate") { Some("Gate") }
                else if name.contains("up_proj") || name.contains("ffn_up") { Some("Up") }
                else if name.contains("down_proj") || name.contains("ffn_down") { Some("Down") }
                else { None };

            for _ in 0..n_rows {
                if pos + 34 > data.len() { break; }
                if let Some(rl) = role_label {
                    // Convert Base17 i16[17] to f32[17] for fingerprinting
                    let mut f32_vals = vec![0.0f32; 17];
                    for d in 0..17 {
                        f32_vals[d] = i16::from_le_bytes([data[pos + d*2], data[pos + d*2 + 1]]) as f32;
                    }
                    if count < 500 {
                        role_fps.entry(rl).or_default()
                            .push(ThinkingStyleFingerprint::from_gate_weights(&f32_vals));
                    }
                }
                pos += 34;
                count += 1;
            }
        }

        for role_name in &["Q", "K", "V", "Gate", "Up", "Down"] {
            if let Some(fps) = role_fps.get(role_name) {
                if fps.is_empty() { continue; }
                let sample = fps.len().min(50);
                let mut intra = Vec::new();
                for i in 0..sample {
                    for j in (i+1)..sample.min(i+5) {
                        intra.push(fps[i].bit_disagreements(&fps[j]) as f64);
                    }
                }
                let mean = intra.iter().sum::<f64>() / intra.len().max(1) as f64;
                println!("  {:<5}: {} fps, intra-hamming={:.1}, profile: {}", role_name, fps.len(), mean, fps[0].profile());
            }
        }
    }

    println!("\n=== DONE ===");
}

// Minimal GGUF reader (reused)
struct GgufHeader { tensors: Vec<TensorMeta>, data_offset: u64 }
struct TensorMeta { name: String, dims: Vec<u64>, dtype: u32, offset: u64, n_elements: u64 }

fn parse_gguf_header<R: Read + Seek>(r: &mut R) -> Result<GgufHeader, String> {
    let mut b4 = [0u8; 4]; let mut b8 = [0u8; 8];
    r.read_exact(&mut b4).map_err(|e| e.to_string())?;
    if u32::from_le_bytes(b4) != 0x46554747 { return Err("bad magic".into()); }
    r.read_exact(&mut b4).map_err(|e| e.to_string())?;
    r.read_exact(&mut b8).map_err(|e| e.to_string())?;
    let nt = u64::from_le_bytes(b8) as usize;
    r.read_exact(&mut b8).map_err(|e| e.to_string())?;
    let nm = u64::from_le_bytes(b8) as usize;
    for _ in 0..nm { skip_kv(r)?; }
    let mut tensors = Vec::with_capacity(nt);
    for _ in 0..nt {
        r.read_exact(&mut b8).map_err(|e| e.to_string())?;
        let nl = u64::from_le_bytes(b8) as usize;
        let mut nb = vec![0u8; nl]; r.read_exact(&mut nb).map_err(|e| e.to_string())?;
        let name = String::from_utf8_lossy(&nb).to_string();
        r.read_exact(&mut b4).map_err(|e| e.to_string())?;
        let nd = u32::from_le_bytes(b4) as usize;
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
    r.read_exact(&mut b8).map_err(|e|e.to_string())?;
    let kl=u64::from_le_bytes(b8) as usize;
    let mut kb=vec![0u8;kl]; r.read_exact(&mut kb).map_err(|e|e.to_string())?;
    r.read_exact(&mut b4).map_err(|e|e.to_string())?;
    skip_val(r, u32::from_le_bytes(b4))
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
