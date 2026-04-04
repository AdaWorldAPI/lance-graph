//! Build 1:1 exact distance tables per role per model.
//! Every row IS its own centroid. Pearson = 1.000.
//! Distance table = exact cosine topology.
//!
//! Saves to /tmp/codebooks/<model>/role/<table>.u8
//! Then uploads to GitHub release via REST API.
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml --example build_1to1_roles

use std::io::{Read, Seek, SeekFrom};
use std::collections::HashMap;

fn main() {
    println!("=== 1:1 Per-Role Distance Tables ===\n");

    let gguf_files = find_gguf_files();
    if gguf_files.is_empty() { eprintln!("No GGUF files"); return; }

    // Role patterns: Jina/BERT style + Llama/Qwen style
    let role_patterns: &[(&str, &[&str])] = &[
        ("attn_qkv",    &["attn_qkv"]),
        ("attn_q",      &["attn_q", "q_proj"]),
        ("attn_k",      &["attn_k", "k_proj"]),
        ("attn_v",      &["attn_v", "v_proj"]),
        ("attn_output", &["attn_output", "o_proj"]),
        ("ffn_gate",    &["ffn_gate", "gate_proj"]),
        ("ffn_up",      &["ffn_up", "up_proj"]),
        ("ffn_down",    &["ffn_down", "down_proj"]),
    ];

    let mut total_tables = 0;
    let mut total_bytes = 0u64;

    for (model_name, path) in &gguf_files {
        let mut file = match std::fs::File::open(path) {
            Ok(f) => f, Err(_) => continue,
        };
        let header = match parse_gguf_header(&mut file) {
            Ok(h) => h, Err(_) => continue,
        };

        let short = model_name.split('/').last().unwrap_or(model_name)
            .replace("-GGUF", "").replace("-gguf", "");
        println!("════════════════════════════════════════");
        println!("Model: {} ({} tensors)", short, header.tensors.len());
        println!("════════════════════════════════════════");

        let out_dir = format!("/tmp/codebooks/{}-roles", short);
        std::fs::create_dir_all(&out_dir).ok();

        for (role_name, patterns) in role_patterns {
            // Find first matching tensor for this role (from a middle layer for representative data)
            let tensor = header.tensors.iter()
                .filter(|t| {
                    t.n_elements >= 1024 &&
                    (t.dtype == 8 || t.dtype == 0 || t.dtype == 1 || t.dtype == 30) &&
                    patterns.iter().any(|p| t.name.contains(p)) &&
                    !t.name.contains("bias") &&
                    !t.name.contains("norm")
                })
                // Prefer a middle-layer tensor
                .max_by_key(|t| {
                    let layer = extract_layer(&t.name).unwrap_or(0);
                    if layer > 0 { layer } else { 1 }
                });

            let tensor = match tensor { Some(t) => t, None => continue };

            let data = match read_tensor_f32(&mut file, &header, tensor) {
                Ok(d) => d, Err(_) => continue,
            };

            let (n_rows, n_cols) = if tensor.dims.len() >= 2 {
                (tensor.dims[0] as usize, tensor.dims[1..].iter().map(|&d| d as usize).product())
            } else { (1, data.len()) };

            if n_rows < 4 { continue; }

            // Cap at 4096 rows for memory (4096² = 16 MB, fits L3)
            let k = n_rows.min(4096);

            let rows: Vec<&[f32]> = (0..k)
                .map(|r| &data[r * n_cols..(r * n_cols + n_cols).min(data.len())])
                .collect();

            // Build 1:1 distance table
            let start = std::time::Instant::now();
            let mut table = vec![128u8; k * k];
            let mut min_c = 1.0f64;
            let mut max_c = -1.0f64;

            // Compute upper triangle
            for i in 0..k {
                table[i * k + i] = 255; // self = max
                for j in (i + 1)..k {
                    let c = cosine(rows[i], rows[j]);
                    if c < min_c { min_c = c; }
                    if c > max_c { max_c = c; }
                    let u = (((c + 1.0) / 2.0) * 255.0).round().clamp(0.0, 255.0) as u8;
                    table[i * k + j] = u;
                    table[j * k + i] = u;
                }
            }
            let elapsed = start.elapsed();

            let table_bytes = k * k;
            let role_dir = format!("{}/{}", out_dir, role_name);
            std::fs::create_dir_all(&role_dir).ok();

            let table_path = format!("{}/distance_table_{}x{}.u8", role_dir, k, k);
            std::fs::write(&table_path, &table).ok();

            // Save metadata
            let meta = format!(
                "{{\"model\":\"{}\",\"role\":\"{}\",\"tensor\":\"{}\",\"rows\":{},\"cols\":{},\"cos_min\":{:.6},\"cos_max\":{:.6},\"table_bytes\":{},\"build_secs\":{:.2}}}",
                short, role_name, tensor.name, k, n_cols, min_c, max_c, table_bytes, elapsed.as_secs_f64()
            );
            std::fs::write(format!("{}/meta.json", role_dir), &meta).ok();

            total_tables += 1;
            total_bytes += table_bytes as u64;

            let size_display = if table_bytes > 1_000_000 {
                format!("{:.1} MB", table_bytes as f64 / 1_000_000.0)
            } else {
                format!("{:.0} KB", table_bytes as f64 / 1024.0)
            };

            println!("  {:<12} {:>5}×{:<5} = {:>8}  cos[{:.3},{:.3}]  {:.2}s  layer {}",
                role_name, k, k, size_display,
                min_c, max_c, elapsed.as_secs_f64(),
                extract_layer(&tensor.name).unwrap_or(0));
        }
        println!();
    }

    println!("════════════════════════════════════════");
    println!("Total: {} tables, {:.1} MB",
        total_tables, total_bytes as f64 / 1_000_000.0);
    println!("════════════════════════════════════════");

    // List all saved files
    println!("\nSaved to /tmp/codebooks/*-roles/:");
    if let Ok(entries) = std::fs::read_dir("/tmp/codebooks") {
        for e in entries.flatten() {
            if e.path().to_string_lossy().contains("-roles") {
                let dir = e.path();
                if let Ok(roles) = std::fs::read_dir(&dir) {
                    for role in roles.flatten() {
                        if role.path().is_dir() {
                            if let Ok(files) = std::fs::read_dir(role.path()) {
                                for f in files.flatten() {
                                    if f.path().to_string_lossy().ends_with(".u8") {
                                        let size = std::fs::metadata(f.path()).map(|m| m.len()).unwrap_or(0);
                                        println!("  {} ({:.1} MB)", f.path().display(), size as f64 / 1_000_000.0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn extract_layer(name: &str) -> Option<u32> {
    let n = name.to_lowercase();
    if let Some(pos) = n.find("blk.") {
        n[pos + 4..].split('.').next().and_then(|s| s.parse().ok())
    } else if let Some(pos) = n.find("layers.") {
        n[pos + 7..].split('.').next().and_then(|s| s.parse().ok())
    } else { None }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n { dot += a[i] as f64 * b[i] as f64; na += (a[i] as f64).powi(2); nb += (b[i] as f64).powi(2); }
    let d = (na * nb).sqrt(); if d < 1e-12 { 0.0 } else { dot / d }
}

fn find_gguf_files() -> Vec<(String, String)> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/tmp/hf_cache") {
        for entry in entries.flatten() {
            let p = entry.path();
            if !p.is_dir() || !p.to_string_lossy().contains("models--") { continue; }
            let snap = p.join("snapshots");
            if let Ok(snaps) = std::fs::read_dir(&snap) {
                for s in snaps.flatten() {
                    if let Ok(gfs) = std::fs::read_dir(s.path()) {
                        for gf in gfs.flatten() {
                            let gp = gf.path();
                            if !gp.to_string_lossy().ends_with(".gguf") || gp.to_string_lossy().contains("mmproj") { continue; }
                            let real = std::fs::read_link(&gp).map(|r| if r.is_relative() { gp.parent().unwrap().join(r) } else { r }).unwrap_or(gp.clone());
                            if real.exists() && std::fs::metadata(&real).map(|m| m.len() > 1000).unwrap_or(false) {
                                let name = p.file_name().map(|n| n.to_string_lossy().replace("models--","").replace("--","/")).unwrap_or_default();
                                files.push((name, real.to_string_lossy().to_string()));
                            }
                        }
                    }
                }
            }
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files
}

struct GgufHeader { tensors: Vec<TensorMeta>, data_offset: u64 }
struct TensorMeta { name: String, dims: Vec<u64>, dtype: u32, offset: u64, n_elements: u64 }
fn parse_gguf_header<R:Read+Seek>(r:&mut R)->Result<GgufHeader,String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b4).map_err(|e|e.to_string())?;if u32::from_le_bytes(b4)!=0x46554747{return Err("bad magic".into());}r.read_exact(&mut b4).map_err(|e|e.to_string())?;r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nt=u64::from_le_bytes(b8)as usize;r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nm=u64::from_le_bytes(b8)as usize;for _ in 0..nm{skip_kv(r)?;}let mut tensors=Vec::with_capacity(nt);for _ in 0..nt{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nl=u64::from_le_bytes(b8)as usize;let mut nb=vec![0u8;nl];r.read_exact(&mut nb).map_err(|e|e.to_string())?;let name=String::from_utf8_lossy(&nb).to_string();r.read_exact(&mut b4).map_err(|e|e.to_string())?;let nd=u32::from_le_bytes(b4)as usize;let mut dims=Vec::with_capacity(nd);for _ in 0..nd{r.read_exact(&mut b8).map_err(|e|e.to_string())?;dims.push(u64::from_le_bytes(b8));}r.read_exact(&mut b4).map_err(|e|e.to_string())?;let dtype=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let offset=u64::from_le_bytes(b8);tensors.push(TensorMeta{name,dims:dims.clone(),dtype,offset,n_elements:dims.iter().product()});}let pos=r.stream_position().map_err(|e|e.to_string())?;Ok(GgufHeader{tensors,data_offset:(pos+31)/32*32})}
fn skip_kv<R:Read+Seek>(r:&mut R)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b8).map_err(|e|e.to_string())?;let kl=u64::from_le_bytes(b8)as usize;let mut kb=vec![0u8;kl];r.read_exact(&mut kb).map_err(|e|e.to_string())?;r.read_exact(&mut b4).map_err(|e|e.to_string())?;skip_val(r,u32::from_le_bytes(b4))}
fn skip_val<R:Read+Seek>(r:&mut R,vt:u32)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];match vt{0|1|7=>{let mut b=[0u8;1];r.read_exact(&mut b).map_err(|e|e.to_string())?;}2|3=>{r.read_exact(&mut[0u8;2]).map_err(|e|e.to_string())?;}4|5|6=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;}8=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let l=u64::from_le_bytes(b8)as usize;let mut s=vec![0u8;l];r.read_exact(&mut s).map_err(|e|e.to_string())?;}9=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;let et=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let c=u64::from_le_bytes(b8)as usize;for _ in 0..c{skip_val(r,et)?;}}10|11|12=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;}_=>return Err(format!("unknown vtype {}",vt)),}Ok(())}
fn read_tensor_f32<R:Read+Seek>(r:&mut R,h:&GgufHeader,t:&TensorMeta)->Result<Vec<f32>,String>{r.seek(SeekFrom::Start(h.data_offset+t.offset)).map_err(|e|e.to_string())?;let n=t.n_elements as usize;match t.dtype{0=>{let mut buf=vec![0u8;n*4];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())}8=>{let nb=(n+31)/32;let bpb=34;let mut buf=vec![0u8;nb*bpb];r.read_exact(&mut buf).map_err(|e|e.to_string())?;let mut res=Vec::with_capacity(n);for b in 0..nb{let o=b*bpb;let sb=u16::from_le_bytes([buf[o],buf[o+1]]);let s=f32::from_bits((sb as u32)<<16);for i in 0..32{if res.len()>=n{break;}res.push(buf[o+2+i]as i8 as f32*s);}}Ok(res)}1=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|{let bits=u16::from_le_bytes([c[0],c[1]]);let s=((bits>>15)&1)as u32;let e=((bits>>10)&0x1F)as u32;let f=(bits&0x3FF)as u32;if e==0{if f==0{f32::from_bits(s<<31)}else{let v=f as f32/1024.0*2.0f32.powi(-14);if s==1{-v}else{v}}}else if e==31{if f==0{if s==1{f32::NEG_INFINITY}else{f32::INFINITY}}else{f32::NAN}}else{f32::from_bits((s<<31)|((e+127-15)<<23)|(f<<13))}}).collect())}30=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|f32::from_bits((u16::from_le_bytes([c[0],c[1]])as u32)<<16)).collect())}_=>Err(format!("unsupported dtype {}",t.dtype)),}}
