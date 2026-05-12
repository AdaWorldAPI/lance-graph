//! Stream codebook build from HuggingFace — zero disk, 256 MB peak RAM.
//!
//! Uses ndarray's HttpRangeReader to stream GGUF tensors directly from HF.
//! Never downloads the full file. Each tensor read via HTTP range request.
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml --example stream_codebook -- Jackrong/Qwopus3.5-27B-v3-GGUF Qwopus3.5-27B-v3-Q2_K.gguf

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (repo, filename) = if args.len() >= 3 {
        (args[1].as_str(), args[2].as_str())
    } else {
        // Default: Qwopus 27B v3 Q4_K_M (smaller than Q8_0, better than Q2_K)
        ("Jackrong/Qwopus3.5-27B-v3-GGUF", "Qwopus3.5-27B-v3-Q4_K_M.gguf")
    };

    println!("=== Streaming Codebook Build (zero disk) ===");
    println!("Repo: {}", repo);
    println!("File: {}\n", filename);

    // Step 1: Resolve HF URL + get file size
    println!("[1] Resolving HF URL...");
    let (url, size) = match ndarray::hpc::http_reader::resolve_hf_url(repo, filename) {
        Ok(r) => r,
        Err(e) => { eprintln!("Failed to resolve: {}", e); return; }
    };
    println!("  URL: {}...{}", &url[..50.min(url.len())], &url[url.len().saturating_sub(30)..]);
    println!("  Size: {:.1} GB\n", size as f64 / 1e9);

    // Step 2: Create streaming reader (256 MB chunks)
    println!("[2] Creating HttpRangeReader (256 MB window)...");
    let mut reader = ndarray::hpc::http_reader::HttpRangeReader::with_chunk_size(
        url, size, 256 * 1024 * 1024
    );

    // Step 3: Parse GGUF header (first ~100 KB)
    println!("[3] Parsing GGUF header...");
    use std::io::{Read, Seek, SeekFrom};

    let header = match parse_gguf_header_streaming(&mut reader) {
        Ok(h) => h,
        Err(e) => { eprintln!("GGUF parse error: {}", e); return; }
    };
    println!("  Tensors: {}", header.tensors.len());

    // Step 4: Find weight tensors with Q8_0 dtype
    let weight_tensors: Vec<&TensorMeta> = header.tensors.iter()
        .filter(|t| t.n_elements >= 1024 && (t.dtype == 8 || t.dtype == 0 || t.dtype == 1))
        .collect();
    println!("  Weight tensors (Q8_0/F32/F16): {}", weight_tensors.len());

    if weight_tensors.is_empty() {
        // Try Q4_K or Q2_K (we don't have dequant for these yet)
        let all_types: std::collections::HashMap<u32, usize> = header.tensors.iter()
            .fold(std::collections::HashMap::new(), |mut m, t| {
                *m.entry(t.dtype).or_insert(0) += 1; m
            });
        println!("  Tensor dtype distribution: {:?}", all_types);
        println!("  No supported dtype found. Need Q2_K/Q4_K dequant support.");
        return;
    }

    // Pick a representative tensor (prefer ffn_down, avoid embeddings)
    let target = weight_tensors.iter()
        .find(|t| t.name.contains("ffn_down") && !t.name.contains("embed"))
        .or_else(|| weight_tensors.iter().find(|t| !t.name.contains("embed")))
        .unwrap_or(&weight_tensors[0]);

    println!("  Target: {} ({:?}, {} elements)", target.name,
        target.dims, target.n_elements);

    // Step 5: Stream tensor data via range request
    println!("\n[4] Streaming tensor data...");
    let start = std::time::Instant::now();
    let f32_data = match read_tensor_f32_streaming(&mut reader, &header, target) {
        Ok(d) => d,
        Err(e) => { eprintln!("Read error: {}", e); return; }
    };
    println!("  Streamed {} floats in {:.2}s", f32_data.len(), start.elapsed().as_secs_f64());

    let (n_rows, n_cols) = if target.dims.len() >= 2 {
        (target.dims[0] as usize, target.dims[1..].iter().map(|&d| d as usize).product())
    } else { (1, f32_data.len()) };

    // Step 6: CLAM codebook build (64 centroids)
    println!("\n[5] CLAM furthest-point sampling (64 centroids)...");
    let rows: Vec<Vec<f32>> = (0..n_rows.min(4096))
        .map(|r| {
            let s = r * n_cols;
            let e = (s + n_cols).min(f32_data.len());
            f32_data[s..e].to_vec()
        })
        .filter(|v| !v.is_empty())
        .collect();

    let start = std::time::Instant::now();
    let k = 64.min(rows.len());
    let centroids = clam_sample(&rows, k);
    println!("  {} centroids in {:.2}s", centroids.len(), start.elapsed().as_secs_f64());

    // Step 7: Build distance table
    let table = build_cosine_table(&centroids);
    let mean: f64 = table.iter().map(|&v| v as f64).sum::<f64>() / table.len() as f64;
    let min = table.iter().copied().min().unwrap_or(0);
    let max = table.iter().copied().max().unwrap_or(0);
    println!("  Distance table: {}×{}, min={}, max={}, mean={:.1}", k, k, min, max, mean);

    // Step 8: Save
    let model_name = filename.replace(".gguf", "").replace(".", "-");
    let out_dir = format!("/tmp/codebooks/{}", model_name);
    std::fs::create_dir_all(&out_dir).ok();
    std::fs::write(format!("{}/distance_table_{}x{}.u8", out_dir, k, k), &table).ok();
    println!("\n  Saved to {}/", out_dir);
    println!("  Peak RAM: ~{} MB (one tensor + centroids)",
        (f32_data.len() * 4 + centroids.len() * centroids[0].len() * 4) / (1024 * 1024));
    println!("  Disk used: 0 bytes (streamed from HF)");

    println!("\n=== Done ===");
}

fn clam_sample(vectors: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let k = k.min(n);
    let mut selected = Vec::with_capacity(k);
    let mut max_dist = vec![f64::INFINITY; n];
    selected.push(0);
    for i in 0..n { max_dist[i] = 1.0 - cosine(&vectors[i], &vectors[0]); }
    for _ in 1..k {
        let next = max_dist.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        selected.push(next);
        for i in 0..n {
            let d = 1.0 - cosine(&vectors[i], &vectors[next]);
            if d < max_dist[i] { max_dist[i] = d; }
        }
    }
    selected.iter().map(|&i| vectors[i].clone()).collect()
}

fn build_cosine_table(centroids: &[Vec<f32>]) -> Vec<u8> {
    let k = centroids.len();
    let mut raw = vec![0.0f64; k * k];
    let mut min_c = 1.0f64; let mut max_c = -1.0f64;
    for i in 0..k {
        raw[i * k + i] = 1.0;
        for j in (i+1)..k {
            let c = cosine(&centroids[i], &centroids[j]);
            raw[i*k+j] = c; raw[j*k+i] = c;
            if c < min_c { min_c = c; } if c > max_c { max_c = c; }
        }
    }
    let range = (max_c - min_c).max(1e-10);
    (0..k*k).map(|i| (((raw[i] - min_c) / range) * 255.0).round().clamp(0.0, 255.0) as u8).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n { dot += a[i] as f64 * b[i] as f64; na += (a[i] as f64).powi(2); nb += (b[i] as f64).powi(2); }
    let d = (na * nb).sqrt(); if d < 1e-12 { 0.0 } else { dot / d }
}

// ═══ Streaming GGUF parser ═══
struct GgufHeader { tensors: Vec<TensorMeta>, data_offset: u64 }
struct TensorMeta { name: String, dims: Vec<u64>, dtype: u32, offset: u64, n_elements: u64 }

fn parse_gguf_header_streaming<R: std::io::Read + std::io::Seek>(r: &mut R) -> Result<GgufHeader, String> {
    let mut b4=[0u8;4]; let mut b8=[0u8;8];
    r.read_exact(&mut b4).map_err(|e|e.to_string())?;
    if u32::from_le_bytes(b4) != 0x46554747 { return Err("bad GGUF magic".into()); }
    r.read_exact(&mut b4).map_err(|e|e.to_string())?;
    r.read_exact(&mut b8).map_err(|e|e.to_string())?; let nt=u64::from_le_bytes(b8) as usize;
    r.read_exact(&mut b8).map_err(|e|e.to_string())?; let nm=u64::from_le_bytes(b8) as usize;
    for _ in 0..nm { skip_kv(r)?; }
    let mut tensors=Vec::with_capacity(nt);
    for _ in 0..nt {
        r.read_exact(&mut b8).map_err(|e|e.to_string())?; let nl=u64::from_le_bytes(b8) as usize;
        let mut nb=vec![0u8;nl]; r.read_exact(&mut nb).map_err(|e|e.to_string())?;
        let name=String::from_utf8_lossy(&nb).to_string();
        r.read_exact(&mut b4).map_err(|e|e.to_string())?; let nd=u32::from_le_bytes(b4) as usize;
        let mut dims=Vec::with_capacity(nd);
        for _ in 0..nd { r.read_exact(&mut b8).map_err(|e|e.to_string())?; dims.push(u64::from_le_bytes(b8)); }
        r.read_exact(&mut b4).map_err(|e|e.to_string())?; let dtype=u32::from_le_bytes(b4);
        r.read_exact(&mut b8).map_err(|e|e.to_string())?; let offset=u64::from_le_bytes(b8);
        tensors.push(TensorMeta{name,dims:dims.clone(),dtype,offset,n_elements:dims.iter().product()});
    }
    let pos=r.stream_position().map_err(|e|e.to_string())?;
    Ok(GgufHeader{tensors,data_offset:(pos+31)/32*32})
}

fn skip_kv<R:std::io::Read+std::io::Seek>(r:&mut R)->Result<(),String>{
    let mut b4=[0u8;4];let mut b8=[0u8;8];
    r.read_exact(&mut b8).map_err(|e|e.to_string())?;let kl=u64::from_le_bytes(b8)as usize;
    let mut kb=vec![0u8;kl];r.read_exact(&mut kb).map_err(|e|e.to_string())?;
    r.read_exact(&mut b4).map_err(|e|e.to_string())?;skip_val(r,u32::from_le_bytes(b4))
}

fn skip_val<R:std::io::Read+std::io::Seek>(r:&mut R,vt:u32)->Result<(),String>{
    let mut b4=[0u8;4];let mut b8=[0u8;8];
    match vt{0|1|7=>{let mut b=[0u8;1];r.read_exact(&mut b).map_err(|e|e.to_string())?;}
    2|3=>{r.read_exact(&mut[0u8;2]).map_err(|e|e.to_string())?;}
    4|5|6=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;}
    8=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let l=u64::from_le_bytes(b8)as usize;let mut s=vec![0u8;l];r.read_exact(&mut s).map_err(|e|e.to_string())?;}
    9=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;let et=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let c=u64::from_le_bytes(b8)as usize;for _ in 0..c{skip_val(r,et)?;}}
    10|11|12=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;}
    _=>return Err(format!("unknown vtype {}",vt)),}Ok(())
}

fn read_tensor_f32_streaming<R:std::io::Read+std::io::Seek>(r:&mut R,h:&GgufHeader,t:&TensorMeta)->Result<Vec<f32>,String>{
    use std::io::SeekFrom;
    r.seek(SeekFrom::Start(h.data_offset+t.offset)).map_err(|e|e.to_string())?;
    let n=t.n_elements as usize;
    match t.dtype{
    0=>{let mut buf=vec![0u8;n*4];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())}
    8=>{let nb=(n+31)/32;let bpb=34;let mut buf=vec![0u8;nb*bpb];r.read_exact(&mut buf).map_err(|e|e.to_string())?;
        let mut res=Vec::with_capacity(n);for b in 0..nb{let o=b*bpb;let sb=u16::from_le_bytes([buf[o],buf[o+1]]);let s=f32::from_bits((sb as u32)<<16);
        for i in 0..32{if res.len()>=n{break;}res.push(buf[o+2+i]as i8 as f32*s);}}Ok(res)}
    1=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;
        Ok(buf.chunks_exact(2).map(|c|f32::from_bits((u16::from_le_bytes([c[0],c[1]]) as u32)<<16)).collect())}
    _=>Err(format!("unsupported dtype {} — need Q2_K/Q4_K dequant for this model",t.dtype)),}
}
