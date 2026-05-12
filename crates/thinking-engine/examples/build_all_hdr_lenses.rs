//! Build HDR precision lenses for ALL available F16 GGUFs.
//!
//! For each model: token_embd.weight → CLAM 256 → HDR CDF → binary files
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
//!   --example build_all_hdr_lenses

use rayon::prelude::*;
use std::io::{Read, Seek, SeekFrom};

const N_CENTROIDS: usize = 256;

fn main() {
    println!("═══ BUILD ALL HDR PRECISION LENSES ═══\n");

    let models = find_f16_ggufs();
    if models.is_empty() {
        eprintln!("No F16 GGUFs found in /tmp/hf_cache");
        return;
    }
    println!("Found {} F16 GGUFs:\n", models.len());
    for (name, path) in &models {
        println!("  {}: {}", name, &path[path.len().saturating_sub(50)..]);
    }
    println!();

    for (name, path) in &models {
        println!("════════════════════════════════════════");
        println!("  Model: {}", name);
        println!("════════════════════════════════════════");

        match build_hdr_lens(name, path) {
            Ok((table_std, n_centroids, vocab)) => {
                println!("  ✓ HDR table: {}×{} std={:.1} vocab={}\n",
                    n_centroids, n_centroids, table_std, vocab);
            }
            Err(e) => {
                println!("  ✗ Failed: {}\n", e);
            }
        }
    }

    // Summary
    println!("═══ ALL LENSES BUILT ═══");
    let out_dir = "/tmp/codebooks";
    for (name, _) in &models {
        let dir = format!("{}/{}-hdr", out_dir, name);
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let files: Vec<String> = entries.flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect();
            println!("  {}: {:?}", name, files);
        }
    }
}

fn build_hdr_lens(name: &str, gguf_path: &str) -> Result<(f64, usize, usize), String> {
    let mut file = std::fs::File::open(gguf_path).map_err(|e| e.to_string())?;
    let header = parse_gguf_header(&mut file)?;

    // Find token embedding tensor
    let embd = header.tensors.iter()
        .find(|t| (t.name.contains("token_embd") || t.name.contains("token_embed"))
            && t.name.ends_with("weight") && t.n_elements > 10000)
        .ok_or_else(|| "No token embedding tensor found".to_string())?;

    println!("  Tensor: {} dtype={} dims={:?}", embd.name, embd.dtype, embd.dims);

    let hidden_dim = embd.dims[0].min(embd.dims.get(1).copied().unwrap_or(embd.dims[0])) as usize;
    let vocab_size = embd.dims[0].max(embd.dims.get(1).copied().unwrap_or(embd.dims[0])) as usize;
    let is_transposed = embd.dims[0] as usize == hidden_dim && hidden_dim != vocab_size;

    println!("  vocab={} hidden={} transposed={}", vocab_size, hidden_dim, is_transposed);

    // Read embeddings
    let raw = read_tensor_f32(&mut file, &header, embd)?;

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

    // Normalize
    let normed: Vec<Vec<f32>> = (0..vocab_size).map(|v| {
        let row = &embeddings[v * hidden_dim..(v + 1) * hidden_dim];
        let norm = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        if norm < 1e-12 { vec![0.0f32; hidden_dim] }
        else { let inv = (1.0 / norm) as f32; row.iter().map(|x| x * inv).collect() }
    }).collect();

    // CLAM furthest-point
    println!("  CLAM {} centroids from {} tokens...", N_CENTROIDS, vocab_size);
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
    let n_cent = selected.len();
    println!("  {} centroids in {:.1}s", n_cent, start.elapsed().as_secs_f64());

    // Assign tokens
    println!("  Assigning {} tokens...", vocab_size);
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
    println!("  Assigned in {:.1}s", start.elapsed().as_secs_f64());

    // Average per centroid
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

    let empty = counts.iter().filter(|&&c| c == 0).count();
    println!("  Centroids: {} (empty: {})", n_cent, empty);

    // Pairwise cosine + HDR CDF encoding
    let mut all_cos: Vec<f32> = Vec::with_capacity(n_cent * (n_cent - 1) / 2);
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
    let cos_std = (all_cos.iter().map(|c| { let d = c - cos_mean; d * d }).sum::<f32>() / all_cos.len().max(1) as f32).sqrt();
    println!("  Cosine: [{:.3}, {:.3}] mean={:.3} std={:.3}", cos_min, cos_max, cos_mean, cos_std);

    // HDR: CDF percentile mapping
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
    println!("  HDR table: avg={:.1} std={:.1}", t_avg, t_std);

    // Save
    let out_dir = format!("/tmp/codebooks/{}-hdr", name);
    std::fs::create_dir_all(&out_dir).ok();
    std::fs::write(format!("{}/distance_table_{}x{}.u8", out_dir, n_cent, n_cent), &table)
        .map_err(|e| e.to_string())?;
    let idx_bytes: Vec<u8> = assignments.iter().flat_map(|&a| a.to_le_bytes()).collect();
    std::fs::write(format!("{}/codebook_index.u16", out_dir), &idx_bytes)
        .map_err(|e| e.to_string())?;

    // Also copy to crate data for baking
    let bake_dir = format!("crates/thinking-engine/data/{}-hdr", name);
    std::fs::create_dir_all(&bake_dir).ok();
    std::fs::write(format!("{}/distance_table_{}x{}.u8", bake_dir, n_cent, n_cent), &table).ok();
    std::fs::write(format!("{}/codebook_index.u16", bake_dir), &idx_bytes).ok();
    println!("  Saved to {} + {}", out_dir, bake_dir);

    Ok((t_std, n_cent, vocab_size))
}

fn find_f16_ggufs() -> Vec<(String, String)> {
    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/tmp/hf_cache") {
        for entry in entries.flatten() {
            let dir_name = entry.file_name().to_string_lossy().to_string();
            if !dir_name.starts_with("models--") { continue; }
            let snap = entry.path().join("snapshots");
            if let Ok(snaps) = std::fs::read_dir(&snap) {
                for s in snaps.flatten() {
                    if let Ok(files) = std::fs::read_dir(s.path()) {
                        for f in files.flatten() {
                            let fname = f.file_name().to_string_lossy().to_string();
                            if fname.ends_with(".gguf") && (fname.contains("f16") || fname.contains("F16")) {
                                let real = std::fs::read_link(f.path())
                                    .map(|r| if r.is_relative() { f.path().parent().unwrap().join(r) } else { r })
                                    .unwrap_or(f.path());
                                let model_name = dir_name.replace("models--", "")
                                    .replace("--", "_")
                                    .replace("-GGUF", "").replace("-gguf", "");
                                models.push((model_name, real.to_string_lossy().to_string()));
                            }
                        }
                    }
                }
            }
        }
    }
    models.sort_by(|a, b| a.0.cmp(&b.0));
    models
}

// ═══ GGUF parser (same as other examples) ═══
struct GgufHeader { tensors: Vec<TensorMeta>, data_offset: u64 }
struct TensorMeta { name: String, dims: Vec<u64>, dtype: u32, offset: u64, n_elements: u64 }
fn parse_gguf_header<R:Read+Seek>(r:&mut R)->Result<GgufHeader,String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b4).map_err(|e|e.to_string())?;if u32::from_le_bytes(b4)!=0x46554747{return Err("bad magic".into());}r.read_exact(&mut b4).map_err(|e|e.to_string())?;r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nt=u64::from_le_bytes(b8)as usize;r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nm=u64::from_le_bytes(b8)as usize;for _ in 0..nm{skip_kv(r)?;}let mut tensors=Vec::with_capacity(nt);for _ in 0..nt{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nl=u64::from_le_bytes(b8)as usize;let mut nb=vec![0u8;nl];r.read_exact(&mut nb).map_err(|e|e.to_string())?;let name=String::from_utf8_lossy(&nb).to_string();r.read_exact(&mut b4).map_err(|e|e.to_string())?;let nd=u32::from_le_bytes(b4)as usize;let mut dims=Vec::with_capacity(nd);for _ in 0..nd{r.read_exact(&mut b8).map_err(|e|e.to_string())?;dims.push(u64::from_le_bytes(b8));}r.read_exact(&mut b4).map_err(|e|e.to_string())?;let dtype=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let offset=u64::from_le_bytes(b8);tensors.push(TensorMeta{name,dims:dims.clone(),dtype,offset,n_elements:dims.iter().product()});}let pos=r.stream_position().map_err(|e|e.to_string())?;Ok(GgufHeader{tensors,data_offset:(pos+31)/32*32})}
fn skip_kv<R:Read+Seek>(r:&mut R)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b8).map_err(|e|e.to_string())?;let kl=u64::from_le_bytes(b8)as usize;let mut kb=vec![0u8;kl];r.read_exact(&mut kb).map_err(|e|e.to_string())?;r.read_exact(&mut b4).map_err(|e|e.to_string())?;skip_val(r,u32::from_le_bytes(b4))}
fn skip_val<R:Read+Seek>(r:&mut R,vt:u32)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];match vt{0|1|7=>{let mut b=[0u8;1];r.read_exact(&mut b).map_err(|e|e.to_string())?;}2|3=>{r.read_exact(&mut[0u8;2]).map_err(|e|e.to_string())?;}4|5|6=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;}8=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let l=u64::from_le_bytes(b8)as usize;let mut s=vec![0u8;l];r.read_exact(&mut s).map_err(|e|e.to_string())?;}9=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;let et=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let c=u64::from_le_bytes(b8)as usize;for _ in 0..c{skip_val(r,et)?;}}10|11|12=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;}_=>return Err(format!("unknown vtype {}",vt)),}Ok(())}
fn read_tensor_f32<R:Read+Seek>(r:&mut R,h:&GgufHeader,t:&TensorMeta)->Result<Vec<f32>,String>{r.seek(SeekFrom::Start(h.data_offset+t.offset)).map_err(|e|e.to_string())?;let n=t.n_elements as usize;match t.dtype{0=>{let mut buf=vec![0u8;n*4];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())}1=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|{let bits=u16::from_le_bytes([c[0],c[1]]);let s=((bits>>15)&1)as u32;let e=((bits>>10)&0x1F)as u32;let f=(bits&0x3FF)as u32;if e==0{if f==0{f32::from_bits(s<<31)}else{let v=f as f32/1024.0*2.0f32.powi(-14);if s==1{-v}else{v}}}else if e==31{if f==0{if s==1{f32::NEG_INFINITY}else{f32::INFINITY}}else{f32::NAN}}else{f32::from_bits((s<<31)|((e+127-15)<<23)|(f<<13))}}).collect())}30=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|f32::from_bits((u16::from_le_bytes([c[0],c[1]])as u32)<<16)).collect())}_=>Err(format!("unsupported dtype {}",t.dtype)),}}
