//! Six-Pack HDR: project 256 centroids through Q/K/V/Gate/Up/Down roles
//!
//! Instead of one distance table from raw token embeddings, build SIX tables:
//! each showing how the 256 centroids relate THROUGH a specific attention role.
//!
//! "Stock market" through Q = "what it searches for"
//! "Stock market" through K = "what finds it"
//! "Stock market" through V = "what it carries"
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
//!   --example stream_sixpack -- <repo> <file.gguf> [layer]

use rayon::prelude::*;
use std::io::{Read, Seek, SeekFrom};

const N_CENTROIDS: usize = 256;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (repo, filename) = if args.len() >= 3 {
        (args[1].as_str(), args[2].as_str())
    } else {
        ("Jackrong/Qwopus3.5-9B-v3-GGUF", "Qwen3.5-9B.BF16.gguf")
    };
    let target_layer = if args.len() >= 4 {
        args[3].parse::<usize>().unwrap_or(20)
    } else { 20 }; // middle layer

    println!("═══ SIX-PACK HDR (Q/K/V/Gate/Up/Down) ═══");
    println!("Repo: {}  File: {}  Layer: {}\n", repo, filename, target_layer);

    // Stream GGUF
    println!("[1] Resolving...");
    let mut reader = match ndarray::hpc::http_reader::HttpRangeReader::from_hf(repo, filename, 64 * 1024 * 1024) {
        Ok(r) => r,
        Err(e) => {
            let (url, size) = ndarray::hpc::http_reader::resolve_hf_url(repo, filename).expect("resolve");
            ndarray::hpc::http_reader::HttpRangeReader::with_chunk_size(url, size, 64 * 1024 * 1024)
        }
    };

    println!("[2] Parsing header...");
    let header = parse_gguf_header(&mut reader).expect("parse");
    println!("  {} tensors", header.tensors.len());

    // Find token embedding
    let embd = header.tensors.iter()
        .find(|t| t.name.contains("token_embd") && t.name.ends_with("weight") && t.n_elements > 10000)
        .expect("no token_embd");
    let hidden = embd.dims[0].min(embd.dims.get(1).copied().unwrap_or(embd.dims[0])) as usize;
    let vocab = embd.dims[0].max(embd.dims.get(1).copied().unwrap_or(embd.dims[0])) as usize;
    println!("  token_embd: vocab={} hidden={} dtype={}", vocab, hidden, embd.dtype);

    // Find role tensors for target layer
    let role_names = [
        ("Q",    format!("blk.{}.attn_q.weight", target_layer)),
        ("K",    format!("blk.{}.attn_k.weight", target_layer)),
        ("V",    format!("blk.{}.attn_v.weight", target_layer)),
        ("Gate", format!("blk.{}.ffn_gate.weight", target_layer)),
        ("Up",   format!("blk.{}.ffn_up.weight", target_layer)),
        ("Down", format!("blk.{}.ffn_down.weight", target_layer)),
    ];

    println!("  Layer {} roles:", target_layer);
    for (name, tensor_name) in &role_names {
        if let Some(t) = header.tensors.iter().find(|t| t.name == *tensor_name) {
            println!("    {}: {} dims={:?} dtype={}", name, t.name, t.dims, t.dtype);
        } else {
            println!("    {}: NOT FOUND ({})", name, tensor_name);
        }
    }

    // Read token embeddings
    println!("\n[3] Streaming token embeddings ({:.0} MB)...", embd.n_elements as f64 * 2.0 / 1e6);
    let raw_embd = read_tensor_f32(&mut reader, &header, embd).expect("read embd");
    let is_transposed = embd.dims[0] as usize == hidden;
    let embeddings: Vec<f32> = if is_transposed {
        let mut t = vec![0.0f32; vocab * hidden];
        for d in 0..hidden { for v in 0..vocab { t[v * hidden + d] = raw_embd[d * vocab + v]; } }
        t
    } else { raw_embd };

    // Normalize embeddings
    let normed: Vec<Vec<f32>> = (0..vocab).map(|v| {
        let row = &embeddings[v * hidden..(v + 1) * hidden];
        let norm = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        if norm < 1e-12 { vec![0.0f32; hidden] }
        else { let inv = (1.0 / norm) as f32; row.iter().map(|x| x * inv).collect() }
    }).collect();

    // CLAM on raw embeddings (reuse if already built)
    println!("[4] CLAM {} centroids...", N_CENTROIDS);
    let start = std::time::Instant::now();
    let mut selected = vec![0usize];
    let mut min_dist = vec![f64::INFINITY; vocab];
    for v in 0..vocab {
        let dot: f32 = normed[v].iter().zip(&normed[0]).map(|(a, b)| a * b).sum();
        min_dist[v] = 1.0 - dot as f64;
    }
    for k in 1..N_CENTROIDS.min(vocab) {
        let next = min_dist.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        selected.push(next);
        for v in 0..vocab {
            let dot: f32 = normed[v].iter().zip(&normed[next]).map(|(a, b)| a * b).sum();
            let d = 1.0 - dot as f64;
            if d < min_dist[v] { min_dist[v] = d; }
        }
    }
    println!("  {} centroids in {:.1}s", selected.len(), start.elapsed().as_secs_f64());

    // Get centroid embeddings (raw, not normalized — we'll project then normalize)
    let centroid_raw: Vec<Vec<f32>> = selected.iter()
        .map(|&i| embeddings[i * hidden..(i + 1) * hidden].to_vec())
        .collect();

    // For each role: project centroids, compute HDR table
    println!("\n[5] Building six-pack HDR tables:");

    let model_name = filename.replace(".gguf", "").replace(".", "-");
    let out_dir = format!("/tmp/codebooks/{}-sixpack-l{}", model_name, target_layer);
    std::fs::create_dir_all(&out_dir).ok();
    let bake_dir = format!("crates/thinking-engine/data/{}-sixpack-l{}", model_name, target_layer);
    std::fs::create_dir_all(&bake_dir).ok();

    for (role_name, tensor_name) in &role_names {
        let tensor = match header.tensors.iter().find(|t| t.name == *tensor_name) {
            Some(t) => t,
            None => { println!("  {}: SKIP (not found)", role_name); continue; }
        };

        print!("  {}: streaming {}...", role_name, tensor_name);
        let weight = match read_tensor_f32(&mut reader, &header, tensor) {
            Ok(w) => w,
            Err(e) => { println!(" ERROR: {}", e); continue; }
        };

        // Weight matrix dimensions
        let (w_rows, w_cols) = (tensor.dims[0] as usize, tensor.dims.get(1).copied().unwrap_or(tensor.dims[0]) as usize);

        // Project centroids: projected[c] = centroid[c] × weight^T
        // centroid: [hidden], weight: [out_dim, hidden] → projected: [out_dim]
        let out_dim = if w_rows == hidden { w_cols } else { w_rows };
        let projected: Vec<Vec<f32>> = centroid_raw.iter().map(|centroid| {
            let mut proj = vec![0.0f32; out_dim];
            for o in 0..out_dim {
                let mut sum = 0.0f32;
                for h in 0..hidden {
                    // weight layout: [out_dim, hidden] row-major
                    let w_idx = if w_rows == out_dim { o * w_cols + h } else { h * w_rows + o };
                    if w_idx < weight.len() {
                        sum += centroid[h] * weight[w_idx];
                    }
                }
                proj[o] = sum;
            }
            // Normalize
            let norm = proj.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
            if norm > 1e-12 { let inv = (1.0 / norm) as f32; proj.iter_mut().for_each(|x| *x *= inv); }
            proj
        }).collect();

        // Pairwise cosine + HDR
        let mut all_cos: Vec<f32> = Vec::new();
        let mut raw_cos = vec![0.0f32; N_CENTROIDS * N_CENTROIDS];
        for i in 0..N_CENTROIDS {
            raw_cos[i * N_CENTROIDS + i] = 1.0;
            for j in (i+1)..N_CENTROIDS {
                let dot: f32 = projected[i].iter().zip(&projected[j]).map(|(a, b)| a * b).sum();
                let cos = dot.clamp(-1.0, 1.0);
                raw_cos[i * N_CENTROIDS + j] = cos;
                raw_cos[j * N_CENTROIDS + i] = cos;
                all_cos.push(cos);
            }
        }
        all_cos.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cos_min = all_cos.first().copied().unwrap_or(0.0);
        let cos_max = all_cos.last().copied().unwrap_or(1.0);

        let mut table = vec![0u8; N_CENTROIDS * N_CENTROIDS];
        for i in 0..N_CENTROIDS {
            table[i * N_CENTROIDS + i] = 255;
            for j in (i+1)..N_CENTROIDS {
                let cos = raw_cos[i * N_CENTROIDS + j];
                let rank = all_cos.partition_point(|&c| c <= cos);
                let pct = rank as f32 / all_cos.len() as f32;
                let u = (pct * 254.0).round().clamp(0.0, 254.0) as u8;
                table[i * N_CENTROIDS + j] = u;
                table[j * N_CENTROIDS + i] = u;
            }
        }

        let t_std = (table.iter().map(|&v| { let d = v as f64 - 127.5; d * d }).sum::<f64>() / table.len() as f64).sqrt();

        std::fs::write(format!("{}/{}_256x256.u8", out_dir, role_name), &table).ok();
        std::fs::write(format!("{}/{}_256x256.u8", bake_dir, role_name), &table).ok();

        println!(" cos[{:.3},{:.3}] std={:.1} proj_dim={}", cos_min, cos_max, t_std, out_dim);
    }

    // Save codebook index (shared across all roles)
    let assignments: Vec<u16> = (0..vocab).into_par_iter().map(|v| {
        let mut best = 0u16;
        let mut best_dot = f32::NEG_INFINITY;
        for (c, &idx) in selected.iter().enumerate() {
            let dot: f32 = normed[v].iter().zip(&normed[idx]).map(|(a, b)| a * b).sum();
            if dot > best_dot { best_dot = dot; best = c as u16; }
        }
        best
    }).collect();
    let idx_bytes: Vec<u8> = assignments.iter().flat_map(|&a| a.to_le_bytes()).collect();
    std::fs::write(format!("{}/codebook_index.u16", out_dir), &idx_bytes).ok();
    std::fs::write(format!("{}/codebook_index.u16", bake_dir), &idx_bytes).ok();

    println!("\n[6] Saved to {} + {}", out_dir, bake_dir);
    println!("  6 × 64 KB tables = 384 KB total");
    println!("  + {} KB codebook index", idx_bytes.len() / 1024);
    println!("\n═══ SIX-PACK COMPLETE ═══");
}

// ═══ GGUF parser ═══
struct GgufHeader { tensors: Vec<TensorMeta>, data_offset: u64 }
struct TensorMeta { name: String, dims: Vec<u64>, dtype: u32, offset: u64, n_elements: u64 }
fn parse_gguf_header<R:Read+Seek>(r:&mut R)->Result<GgufHeader,String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b4).map_err(|e|e.to_string())?;if u32::from_le_bytes(b4)!=0x46554747{return Err("bad magic".into());}r.read_exact(&mut b4).map_err(|e|e.to_string())?;r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nt=u64::from_le_bytes(b8)as usize;r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nm=u64::from_le_bytes(b8)as usize;for _ in 0..nm{skip_kv(r)?;}let mut tensors=Vec::with_capacity(nt);for _ in 0..nt{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nl=u64::from_le_bytes(b8)as usize;let mut nb=vec![0u8;nl];r.read_exact(&mut nb).map_err(|e|e.to_string())?;let name=String::from_utf8_lossy(&nb).to_string();r.read_exact(&mut b4).map_err(|e|e.to_string())?;let nd=u32::from_le_bytes(b4)as usize;let mut dims=Vec::with_capacity(nd);for _ in 0..nd{r.read_exact(&mut b8).map_err(|e|e.to_string())?;dims.push(u64::from_le_bytes(b8));}r.read_exact(&mut b4).map_err(|e|e.to_string())?;let dtype=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let offset=u64::from_le_bytes(b8);tensors.push(TensorMeta{name,dims:dims.clone(),dtype,offset,n_elements:dims.iter().product()});}let pos=r.stream_position().map_err(|e|e.to_string())?;Ok(GgufHeader{tensors,data_offset:(pos+31)/32*32})}
fn skip_kv<R:Read+Seek>(r:&mut R)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b8).map_err(|e|e.to_string())?;let kl=u64::from_le_bytes(b8)as usize;let mut kb=vec![0u8;kl];r.read_exact(&mut kb).map_err(|e|e.to_string())?;r.read_exact(&mut b4).map_err(|e|e.to_string())?;skip_val(r,u32::from_le_bytes(b4))}
fn skip_val<R:Read+Seek>(r:&mut R,vt:u32)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];match vt{0|1|7=>{let mut b=[0u8;1];r.read_exact(&mut b).map_err(|e|e.to_string())?;}2|3=>{r.read_exact(&mut[0u8;2]).map_err(|e|e.to_string())?;}4|5|6=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;}8=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let l=u64::from_le_bytes(b8)as usize;let mut s=vec![0u8;l];r.read_exact(&mut s).map_err(|e|e.to_string())?;}9=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;let et=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let c=u64::from_le_bytes(b8)as usize;for _ in 0..c{skip_val(r,et)?;}}10|11|12=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;}_=>return Err(format!("unknown vtype {}",vt)),}Ok(())}
fn read_tensor_f32<R:Read+Seek>(r:&mut R,h:&GgufHeader,t:&TensorMeta)->Result<Vec<f32>,String>{r.seek(SeekFrom::Start(h.data_offset+t.offset)).map_err(|e|e.to_string())?;let n=t.n_elements as usize;match t.dtype{0=>{let mut buf=vec![0u8;n*4];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())}1=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|{let bits=u16::from_le_bytes([c[0],c[1]]);let s=((bits>>15)&1)as u32;let e=((bits>>10)&0x1F)as u32;let f=(bits&0x3FF)as u32;if e==0{if f==0{f32::from_bits(s<<31)}else{let v=f as f32/1024.0*2.0f32.powi(-14);if s==1{-v}else{v}}}else if e==31{if f==0{if s==1{f32::NEG_INFINITY}else{f32::INFINITY}}else{f32::NAN}}else{f32::from_bits((s<<31)|((e+127-15)<<23)|(f<<13))}}).collect())}30=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|f32::from_bits((u16::from_le_bytes([c[0],c[1]])as u32)<<16)).collect())}_=>Err(format!("unsupported dtype {}",t.dtype)),}}
