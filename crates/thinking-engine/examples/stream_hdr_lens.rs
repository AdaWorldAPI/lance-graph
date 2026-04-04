//! Stream BF16 GGUF from HuggingFace → HDR lens (zero disk)
//!
//! Uses HttpRangeReader to stream only the token_embd tensor.
//! Builds CLAM 256 centroids + HDR CDF table without downloading the full GGUF.
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
//!   --example stream_hdr_lens -- jinaai/jina-reranker-v3-GGUF jina-reranker-v3-BF16.gguf

use rayon::prelude::*;
use std::io::{Read, Seek, SeekFrom};

const N_CENTROIDS: usize = 256;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (repo, filename) = if args.len() >= 3 {
        (args[1].as_str(), args[2].as_str())
    } else {
        ("jinaai/jina-reranker-v3-GGUF", "jina-reranker-v3-BF16.gguf")
    };

    println!("═══ STREAM HDR LENS (zero disk) ═══");
    println!("Repo: {}", repo);
    println!("File: {}\n", filename);

    // Step 1: Resolve + create reader
    println!("[1] Resolving HF URL...");
    let (url, size) = match ndarray::hpc::http_reader::resolve_hf_url(repo, filename) {
        Ok(r) => r,
        Err(e) => { eprintln!("Failed: {}", e); return; }
    };
    println!("  Size: {:.1} GB", size as f64 / 1e9);

    let mut reader = ndarray::hpc::http_reader::HttpRangeReader::from_hf(repo, filename, 64 * 1024 * 1024)
        .unwrap_or_else(|_| ndarray::hpc::http_reader::HttpRangeReader::with_chunk_size(url, size, 64 * 1024 * 1024));

    // Step 2: Parse GGUF header
    println!("[2] Parsing GGUF header...");
    let header = match parse_gguf_header(&mut reader) {
        Ok(h) => h,
        Err(e) => { eprintln!("Parse error: {}", e); return; }
    };
    println!("  {} tensors", header.tensors.len());

    // Find token embedding
    let embd = match header.tensors.iter()
        .find(|t| (t.name.contains("token_embd") || t.name.contains("token_embed"))
            && t.name.ends_with("weight") && t.n_elements > 10000) {
        Some(t) => t,
        None => {
            eprintln!("No token embedding found. Available:");
            for t in &header.tensors {
                if t.name.contains("embd") || t.name.contains("embed") {
                    eprintln!("  {} dtype={} dims={:?}", t.name, t.dtype, t.dims);
                }
            }
            return;
        }
    };
    println!("  {} dtype={} dims={:?}", embd.name, embd.dtype, embd.dims);

    let hidden_dim = embd.dims[0].min(embd.dims.get(1).copied().unwrap_or(embd.dims[0])) as usize;
    let vocab_size = embd.dims[0].max(embd.dims.get(1).copied().unwrap_or(embd.dims[0])) as usize;
    let is_transposed = embd.dims[0] as usize == hidden_dim && hidden_dim != vocab_size;
    println!("  vocab={} hidden={} transposed={}", vocab_size, hidden_dim, is_transposed);

    // Step 3: Stream token embeddings
    println!("[3] Streaming token embeddings ({:.0} MB)...", embd.n_elements as f64 * 2.0 / 1e6);
    let start = std::time::Instant::now();
    let raw = match read_tensor_f32(&mut reader, &header, embd) {
        Ok(d) => d,
        Err(e) => { eprintln!("Read error: {}", e); return; }
    };
    println!("  Streamed {} floats in {:.1}s", raw.len(), start.elapsed().as_secs_f64());

    let embeddings: Vec<f32> = if is_transposed {
        print!("  Transposing...");
        let mut t = vec![0.0f32; vocab_size * hidden_dim];
        for d in 0..hidden_dim {
            for v in 0..vocab_size {
                t[v * hidden_dim + d] = raw[d * vocab_size + v];
            }
        }
        println!(" done");
        t
    } else { raw };

    // Step 4: Normalize
    let normed: Vec<Vec<f32>> = (0..vocab_size).map(|v| {
        let row = &embeddings[v * hidden_dim..(v + 1) * hidden_dim];
        let norm = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        if norm < 1e-12 { vec![0.0f32; hidden_dim] }
        else { let inv = (1.0 / norm) as f32; row.iter().map(|x| x * inv).collect() }
    }).collect();

    // Step 5: CLAM
    println!("[4] CLAM {} centroids from {} tokens...", N_CENTROIDS, vocab_size);
    let start = std::time::Instant::now();
    let mut selected = vec![0usize];
    let mut min_dist = vec![f64::INFINITY; vocab_size];
    for v in 0..vocab_size {
        let dot: f32 = normed[v].iter().zip(&normed[0]).map(|(a, b)| a * b).sum();
        min_dist[v] = 1.0 - dot as f64;
    }
    for k in 1..N_CENTROIDS.min(vocab_size) {
        let next = min_dist.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        selected.push(next);
        for v in 0..vocab_size {
            let dot: f32 = normed[v].iter().zip(&normed[next]).map(|(a, b)| a * b).sum();
            let d = 1.0 - dot as f64;
            if d < min_dist[v] { min_dist[v] = d; }
        }
    }
    println!("  {} centroids in {:.1}s", selected.len(), start.elapsed().as_secs_f64());

    // Step 6: Assign
    println!("[5] Assigning {} tokens...", vocab_size);
    let start = std::time::Instant::now();
    let centroid_vecs: Vec<&[f32]> = selected.iter().map(|&i| normed[i].as_slice()).collect();
    let assignments: Vec<u16> = (0..vocab_size).into_par_iter().map(|v| {
        let mut best = 0u16;
        let mut best_dot = f32::NEG_INFINITY;
        for (c, &cen) in centroid_vecs.iter().enumerate() {
            let dot: f32 = normed[v].iter().zip(cen).map(|(a, b)| a * b).sum();
            if dot > best_dot { best_dot = dot; best = c as u16; }
        }
        best
    }).collect();
    println!("  Done in {:.1}s", start.elapsed().as_secs_f64());

    // Step 7: Average + HDR
    let n_cent = selected.len();
    let mut sums = vec![vec![0.0f64; hidden_dim]; n_cent];
    let mut counts = vec![0u32; n_cent];
    for (v, &c) in assignments.iter().enumerate() {
        counts[c as usize] += 1;
        let row = &embeddings[v * hidden_dim..(v + 1) * hidden_dim];
        for d in 0..hidden_dim { sums[c as usize][d] += row[d] as f64; }
    }
    let centroids_avg: Vec<Vec<f32>> = (0..n_cent).map(|c| {
        if counts[c] == 0 { return vec![0.0f32; hidden_dim]; }
        let n = counts[c] as f64;
        let avg: Vec<f32> = sums[c].iter().map(|&s| (s / n) as f32).collect();
        let norm = avg.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
        if norm < 1e-12 { vec![0.0f32; hidden_dim] }
        else { let inv = (1.0 / norm) as f32; avg.iter().map(|v| v * inv).collect() }
    }).collect();

    // Pairwise + HDR CDF
    let mut all_cos: Vec<f32> = Vec::new();
    let mut raw_cos = vec![0.0f32; n_cent * n_cent];
    for i in 0..n_cent {
        raw_cos[i * n_cent + i] = 1.0;
        for j in (i+1)..n_cent {
            let dot: f32 = centroids_avg[i].iter().zip(&centroids_avg[j]).map(|(a, b)| a * b).sum();
            let cos = dot.clamp(-1.0, 1.0);
            raw_cos[i * n_cent + j] = cos;
            raw_cos[j * n_cent + i] = cos;
            all_cos.push(cos);
        }
    }
    all_cos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cos_min = all_cos.first().copied().unwrap_or(0.0);
    let cos_max = all_cos.last().copied().unwrap_or(1.0);
    let cos_mean: f32 = all_cos.iter().sum::<f32>() / all_cos.len().max(1) as f32;
    println!("[6] Cosine: [{:.3}, {:.3}] mean={:.3}", cos_min, cos_max, cos_mean);

    let mut table = vec![0u8; n_cent * n_cent];
    for i in 0..n_cent {
        table[i * n_cent + i] = 255;
        for j in (i+1)..n_cent {
            let cos = raw_cos[i * n_cent + j];
            let rank = all_cos.partition_point(|&c| c <= cos);
            let pct = rank as f32 / all_cos.len() as f32;
            let u = (pct * 254.0).round().clamp(0.0, 254.0) as u8;
            table[i * n_cent + j] = u;
            table[j * n_cent + i] = u;
        }
    }

    let t_avg = table.iter().map(|&v| v as f64).sum::<f64>() / table.len() as f64;
    let t_std = (table.iter().map(|&v| { let d = v as f64 - t_avg; d * d }).sum::<f64>() / table.len() as f64).sqrt();
    println!("  HDR table: {}×{} avg={:.1} std={:.1}", n_cent, n_cent, t_avg, t_std);

    // Save
    let model_name = filename.replace(".gguf", "").replace(".", "-");
    let out_dir = format!("/tmp/codebooks/{}-hdr", model_name);
    std::fs::create_dir_all(&out_dir).ok();
    std::fs::write(format!("{}/distance_table_{}x{}.u8", out_dir, n_cent, n_cent), &table).ok();
    let idx_bytes: Vec<u8> = assignments.iter().flat_map(|&a| a.to_le_bytes()).collect();
    std::fs::write(format!("{}/codebook_index.u16", out_dir), &idx_bytes).ok();

    // Also to crate data
    let bake = format!("crates/thinking-engine/data/{}-hdr", model_name);
    std::fs::create_dir_all(&bake).ok();
    std::fs::write(format!("{}/distance_table_{}x{}.u8", bake, n_cent, n_cent), &table).ok();
    std::fs::write(format!("{}/codebook_index.u16", bake), &idx_bytes).ok();

    println!("\n[7] Saved: {} + {}", out_dir, bake);
    println!("  Table: {} KB, Index: {} KB", table.len() / 1024, idx_bytes.len() / 1024);
    println!("  Disk used: 0 (streamed from HF)");
    println!("\n═══ STREAM HDR LENS COMPLETE ═══");
}

// ═══ GGUF parser for streaming ═══
struct GgufHeader { tensors: Vec<TensorMeta>, data_offset: u64 }
struct TensorMeta { name: String, dims: Vec<u64>, dtype: u32, offset: u64, n_elements: u64 }
fn parse_gguf_header<R:Read+Seek>(r:&mut R)->Result<GgufHeader,String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b4).map_err(|e|e.to_string())?;if u32::from_le_bytes(b4)!=0x46554747{return Err("bad magic".into());}r.read_exact(&mut b4).map_err(|e|e.to_string())?;r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nt=u64::from_le_bytes(b8)as usize;r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nm=u64::from_le_bytes(b8)as usize;for _ in 0..nm{skip_kv(r)?;}let mut tensors=Vec::with_capacity(nt);for _ in 0..nt{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nl=u64::from_le_bytes(b8)as usize;let mut nb=vec![0u8;nl];r.read_exact(&mut nb).map_err(|e|e.to_string())?;let name=String::from_utf8_lossy(&nb).to_string();r.read_exact(&mut b4).map_err(|e|e.to_string())?;let nd=u32::from_le_bytes(b4)as usize;let mut dims=Vec::with_capacity(nd);for _ in 0..nd{r.read_exact(&mut b8).map_err(|e|e.to_string())?;dims.push(u64::from_le_bytes(b8));}r.read_exact(&mut b4).map_err(|e|e.to_string())?;let dtype=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let offset=u64::from_le_bytes(b8);tensors.push(TensorMeta{name,dims:dims.clone(),dtype,offset,n_elements:dims.iter().product()});}let pos=r.stream_position().map_err(|e|e.to_string())?;Ok(GgufHeader{tensors,data_offset:(pos+31)/32*32})}
fn skip_kv<R:Read+Seek>(r:&mut R)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b8).map_err(|e|e.to_string())?;let kl=u64::from_le_bytes(b8)as usize;let mut kb=vec![0u8;kl];r.read_exact(&mut kb).map_err(|e|e.to_string())?;r.read_exact(&mut b4).map_err(|e|e.to_string())?;skip_val(r,u32::from_le_bytes(b4))}
fn skip_val<R:Read+Seek>(r:&mut R,vt:u32)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];match vt{0|1|7=>{let mut b=[0u8;1];r.read_exact(&mut b).map_err(|e|e.to_string())?;}2|3=>{r.read_exact(&mut[0u8;2]).map_err(|e|e.to_string())?;}4|5|6=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;}8=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let l=u64::from_le_bytes(b8)as usize;let mut s=vec![0u8;l];r.read_exact(&mut s).map_err(|e|e.to_string())?;}9=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;let et=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let c=u64::from_le_bytes(b8)as usize;for _ in 0..c{skip_val(r,et)?;}}10|11|12=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;}_=>return Err(format!("unknown vtype {}",vt)),}Ok(())}
fn read_tensor_f32<R:Read+Seek>(r:&mut R,h:&GgufHeader,t:&TensorMeta)->Result<Vec<f32>,String>{r.seek(SeekFrom::Start(h.data_offset+t.offset)).map_err(|e|e.to_string())?;let n=t.n_elements as usize;match t.dtype{0=>{let mut buf=vec![0u8;n*4];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())}1=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|{let bits=u16::from_le_bytes([c[0],c[1]]);let s=((bits>>15)&1)as u32;let e=((bits>>10)&0x1F)as u32;let f=(bits&0x3FF)as u32;if e==0{if f==0{f32::from_bits(s<<31)}else{let v=f as f32/1024.0*2.0f32.powi(-14);if s==1{-v}else{v}}}else if e==31{if f==0{if s==1{f32::NEG_INFINITY}else{f32::INFINITY}}else{f32::NAN}}else{f32::from_bits((s<<31)|((e+127-15)<<23)|(f<<13))}}).collect())}30=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|f32::from_bits((u16::from_le_bytes([c[0],c[1]])as u32)<<16)).collect())}_=>Err(format!("unsupported dtype {}",t.dtype)),}}
