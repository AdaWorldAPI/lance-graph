//! Stream 1:1 per-role distance tables from HuggingFace F16/BF16 GGUFs.
//! Zero disk — tensors streamed via HttpRangeReader (64 MB segments).
//! Uses ndarray SIMD cosine (F64x8 FMA) + rayon parallel pairs.
//!
//! Usage:
//!   cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
//!     --example stream_1to1_roles -- <repo> <filename>
//!
//! Examples:
//!   # Qwopus 9B v3 BF16 (bartowski)
//!   cargo run --release ... --example stream_1to1_roles -- \
//!     bartowski/Qwen3.5-9B-GGUF Qwen3.5-9B-BF16.gguf
//!
//!   # reader-lm 0.5B F16
//!   cargo run --release ... --example stream_1to1_roles -- \
//!     bartowski/reader-lm-0.5b-GGUF reader-lm-0.5b-f16.gguf

use rayon::prelude::*;
use std::io::{Read, Seek, SeekFrom};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (repo, filename) = if args.len() >= 3 {
        (args[1].as_str(), args[2].as_str())
    } else {
        ("bartowski/Qwen3.5-9B-GGUF", "Qwen3.5-9B-BF16.gguf")
    };

    println!("=== Streaming 1:1 Role Tables (zero disk) ===");
    println!("Repo: {}", repo);
    println!("File: {}\n", filename);

    // Step 1: Resolve HF URL
    println!("[1] Resolving HF URL...");
    let (url, size) = match ndarray::hpc::http_reader::resolve_hf_url(repo, filename) {
        Ok(r) => r,
        Err(e) => { eprintln!("Failed to resolve: {}", e); return; }
    };
    println!("  Size: {:.1} GB", size as f64 / 1e9);

    // Step 2: Create streaming reader (64 MB segments)
    println!("[2] Creating HttpRangeReader (64 MB segments)...");
    let mut reader = ndarray::hpc::http_reader::HttpRangeReader::from_hf(repo, filename, 64 * 1024 * 1024)
        .unwrap_or_else(|_| ndarray::hpc::http_reader::HttpRangeReader::with_chunk_size(url, size, 64 * 1024 * 1024));

    // Step 3: Parse GGUF header
    println!("[3] Parsing GGUF header...");
    let header = match parse_gguf_header(&mut reader) {
        Ok(h) => h,
        Err(e) => { eprintln!("GGUF parse error: {}", e); return; }
    };
    println!("  Total tensors: {}", header.tensors.len());

    // Count dtype distribution
    let mut dtype_counts = std::collections::HashMap::new();
    for t in &header.tensors {
        *dtype_counts.entry(t.dtype).or_insert(0usize) += 1;
    }
    println!("  Dtype distribution: {:?}", dtype_counts);

    // Filter to F16 (dtype=1) and BF16 (dtype=30) weight tensors only
    let fp_tensors: Vec<&TensorMeta> = header.tensors.iter()
        .filter(|t| (t.dtype == 1 || t.dtype == 30) && t.n_elements >= 1024
            && !t.name.contains("bias") && !t.name.contains("norm") && !t.name.contains("embed"))
        .collect();

    if fp_tensors.is_empty() {
        eprintln!("No F16/BF16 weight tensors found. This GGUF may be quantized (Q8_0/Q4_K/etc).");
        eprintln!("Use F16 or BF16 GGUF variants (e.g. from bartowski).");
        return;
    }
    println!("  F16/BF16 weight tensors: {}\n", fp_tensors.len());

    // Role patterns
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

    let model_name = filename.replace(".gguf", "")
        .replace(".", "-").replace(" ", "-");
    let out_base = format!("/tmp/codebooks/{}-roles-f16", model_name);
    std::fs::create_dir_all(&out_base).ok();

    let mut total_tables = 0;
    let mut total_bytes = 0u64;

    println!("[4] Building 1:1 role tables (SIMD cosine + rayon)...\n");

    for (role_name, patterns) in role_patterns {
        // Find the highest-layer F16/BF16 tensor matching this role
        let tensor = fp_tensors.iter()
            .filter(|t| patterns.iter().any(|p| t.name.contains(p)))
            .max_by_key(|t| extract_layer(&t.name).unwrap_or(0).max(1));

        let tensor = match tensor { Some(t) => t, None => continue };

        let dtype_name = if tensor.dtype == 1 { "F16" } else { "BF16" };
        let layer = extract_layer(&tensor.name).unwrap_or(0);

        print!("  {:<12} {} layer {} — streaming {}...",
            role_name, dtype_name, layer, tensor.name);

        // Stream tensor data
        let start = std::time::Instant::now();
        let data = match read_tensor_f32(&mut reader, &header, tensor) {
            Ok(d) => d,
            Err(e) => {
                println!(" ERROR: {}", e);
                continue;
            }
        };
        let stream_secs = start.elapsed().as_secs_f64();

        let (n_rows, n_cols) = if tensor.dims.len() >= 2 {
            (tensor.dims[0] as usize, tensor.dims[1..].iter().map(|&d| d as usize).product())
        } else { (1, data.len()) };

        if n_rows < 4 { println!(" too few rows"); continue; }
        let k = n_rows.min(4096);

        // Pre-normalize rows
        let normalized: Vec<Vec<f32>> = (0..k).map(|r| {
            let row = &data[r * n_cols..(r * n_cols + n_cols).min(data.len())];
            let norm = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
            if norm < 1e-12 {
                vec![0.0f32; row.len()]
            } else {
                let inv = (1.0 / norm) as f32;
                row.iter().map(|v| v * inv).collect()
            }
        }).collect();

        // Build pair indices for rayon
        let pairs: Vec<(usize, usize)> = (0..k)
            .flat_map(|i| ((i + 1)..k).map(move |j| (i, j)))
            .collect();

        let n_pairs = pairs.len();

        // Parallel cosine computation
        let cosines: Vec<(usize, usize, f64)> = pairs.par_iter()
            .map(|&(i, j)| {
                let dot = ndarray::hpc::heel_f64x8::cosine_f32_to_f64_simd(&normalized[i], &normalized[j]);
                (i, j, dot.clamp(-1.0, 1.0))
            })
            .collect();

        let mut table = vec![128u8; k * k];
        let mut min_c = 1.0f64;
        let mut max_c = -1.0f64;

        for i in 0..k { table[i * k + i] = 255; }
        for &(i, j, c) in &cosines {
            if c < min_c { min_c = c; }
            if c > max_c { max_c = c; }
            let u = (((c + 1.0) / 2.0) * 255.0).round().clamp(0.0, 255.0) as u8;
            table[i * k + j] = u;
            table[j * k + i] = u;
        }

        let total_secs = start.elapsed().as_secs_f64();
        let table_bytes = k * k;

        // Save
        let role_dir = format!("{}/{}", out_base, role_name);
        std::fs::create_dir_all(&role_dir).ok();
        std::fs::write(format!("{}/distance_table_{}x{}.u8", role_dir, k, k), &table).ok();

        let meta = format!(
            "{{\"model\":\"{}\",\"role\":\"{}\",\"tensor\":\"{}\",\"dtype\":\"{}\",\"rows\":{},\"cols\":{},\"cos_min\":{:.6},\"cos_max\":{:.6},\"table_bytes\":{},\"stream_secs\":{:.2},\"build_secs\":{:.2},\"pairs\":{},\"simd\":\"f64x8_fma+rayon\",\"source\":\"streaming\"}}",
            model_name, role_name, tensor.name, dtype_name, k, n_cols,
            min_c, max_c, table_bytes, stream_secs, total_secs, n_pairs
        );
        std::fs::write(format!("{}/meta.json", role_dir), &meta).ok();

        total_tables += 1;
        total_bytes += table_bytes as u64;

        let size_display = if table_bytes > 1_000_000 {
            format!("{:.1} MB", table_bytes as f64 / 1_000_000.0)
        } else {
            format!("{:.0} KB", table_bytes as f64 / 1024.0)
        };

        println!(" {}×{} = {} cos[{:.3},{:.3}] stream={:.1}s total={:.1}s",
            k, k, size_display, min_c, max_c, stream_secs, total_secs);
    }

    println!("\n════════════════════════════════════════");
    println!("Total: {} tables, {:.1} MB", total_tables, total_bytes as f64 / 1_000_000.0);
    println!("Output: {}/", out_base);
    println!("Disk used: 0 bytes (streamed from HF)");
    println!("════════════════════════════════════════");
}

fn extract_layer(name: &str) -> Option<u32> {
    let n = name.to_lowercase();
    if let Some(pos) = n.find("blk.") {
        n[pos + 4..].split('.').next().and_then(|s| s.parse().ok())
    } else if let Some(pos) = n.find("layers.") {
        n[pos + 7..].split('.').next().and_then(|s| s.parse().ok())
    } else { None }
}

// ═══ Streaming GGUF parser (same as build_1to1_roles but works with HttpRangeReader) ═══
struct GgufHeader { tensors: Vec<TensorMeta>, data_offset: u64 }
struct TensorMeta { name: String, dims: Vec<u64>, dtype: u32, offset: u64, n_elements: u64 }
fn parse_gguf_header<R:Read+Seek>(r:&mut R)->Result<GgufHeader,String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b4).map_err(|e|e.to_string())?;if u32::from_le_bytes(b4)!=0x46554747{return Err("bad magic".into());}r.read_exact(&mut b4).map_err(|e|e.to_string())?;r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nt=u64::from_le_bytes(b8)as usize;r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nm=u64::from_le_bytes(b8)as usize;for _ in 0..nm{skip_kv(r)?;}let mut tensors=Vec::with_capacity(nt);for _ in 0..nt{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nl=u64::from_le_bytes(b8)as usize;let mut nb=vec![0u8;nl];r.read_exact(&mut nb).map_err(|e|e.to_string())?;let name=String::from_utf8_lossy(&nb).to_string();r.read_exact(&mut b4).map_err(|e|e.to_string())?;let nd=u32::from_le_bytes(b4)as usize;let mut dims=Vec::with_capacity(nd);for _ in 0..nd{r.read_exact(&mut b8).map_err(|e|e.to_string())?;dims.push(u64::from_le_bytes(b8));}r.read_exact(&mut b4).map_err(|e|e.to_string())?;let dtype=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let offset=u64::from_le_bytes(b8);tensors.push(TensorMeta{name,dims:dims.clone(),dtype,offset,n_elements:dims.iter().product()});}let pos=r.stream_position().map_err(|e|e.to_string())?;Ok(GgufHeader{tensors,data_offset:(pos+31)/32*32})}
fn skip_kv<R:Read+Seek>(r:&mut R)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b8).map_err(|e|e.to_string())?;let kl=u64::from_le_bytes(b8)as usize;let mut kb=vec![0u8;kl];r.read_exact(&mut kb).map_err(|e|e.to_string())?;r.read_exact(&mut b4).map_err(|e|e.to_string())?;skip_val(r,u32::from_le_bytes(b4))}
fn skip_val<R:Read+Seek>(r:&mut R,vt:u32)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];match vt{0|1|7=>{let mut b=[0u8;1];r.read_exact(&mut b).map_err(|e|e.to_string())?;}2|3=>{r.read_exact(&mut[0u8;2]).map_err(|e|e.to_string())?;}4|5|6=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;}8=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let l=u64::from_le_bytes(b8)as usize;let mut s=vec![0u8;l];r.read_exact(&mut s).map_err(|e|e.to_string())?;}9=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;let et=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let c=u64::from_le_bytes(b8)as usize;for _ in 0..c{skip_val(r,et)?;}}10|11|12=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;}_=>return Err(format!("unknown vtype {}",vt)),}Ok(())}
fn read_tensor_f32<R:Read+Seek>(r:&mut R,h:&GgufHeader,t:&TensorMeta)->Result<Vec<f32>,String>{r.seek(SeekFrom::Start(h.data_offset+t.offset)).map_err(|e|e.to_string())?;let n=t.n_elements as usize;match t.dtype{0=>{let mut buf=vec![0u8;n*4];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())}1=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|{let bits=u16::from_le_bytes([c[0],c[1]]);let s=((bits>>15)&1)as u32;let e=((bits>>10)&0x1F)as u32;let f=(bits&0x3FF)as u32;if e==0{if f==0{f32::from_bits(s<<31)}else{let v=f as f32/1024.0*2.0f32.powi(-14);if s==1{-v}else{v}}}else if e==31{if f==0{if s==1{f32::NEG_INFINITY}else{f32::INFINITY}}else{f32::NAN}}else{f32::from_bits((s<<31)|((e+127-15)<<23)|(f<<13))}}).collect())}30=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|f32::from_bits((u16::from_le_bytes([c[0],c[1]])as u32)<<16)).collect())}_=>Err(format!("unsupported dtype {} — only F16(1)/BF16(30)/F32(0)",t.dtype)),}}
