//! Measure Pearson correlation per model: codebook cosine vs f32 ground truth.
//!
//! For each model's GGUF:
//! 1. Load weight rows (f32 ground truth)
//! 2. Assign to codebook centroids
//! 3. Compare: codebook_cosine(i,j) vs f32_cosine(i,j)
//! 4. Report Pearson + Spearman per model, per role
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml --example codebook_pearson

use std::io::{Read, Seek, SeekFrom};
use std::collections::HashMap;

fn main() {
    println!("=== Codebook Pearson: Quality per Model per Role ===\n");

    let gguf_files = find_gguf_files();
    if gguf_files.is_empty() { eprintln!("No GGUF files found"); return; }

    println!("┌────────────────────────┬──────────┬──────────┬──────────┬───────┬───────────┐");
    println!("│ Model                  │ Role     │ Rows     │ Pearson  │ Spear │ Centroids │");
    println!("├────────────────────────┼──────────┼──────────┼──────────┼───────┼───────────┤");

    for (model_name, path) in &gguf_files {
        let mut file = match std::fs::File::open(path) {
            Ok(f) => f, Err(_) => continue,
        };
        let header = match parse_gguf_header(&mut file) {
            Ok(h) => h, Err(_) => continue,
        };

        // Collect rows by role
        let roles = ["attn_qkv", "attn_q", "q_proj",
                      "attn_k", "k_proj",
                      "attn_v", "v_proj",
                      "attn_output", "o_proj",
                      "gate_proj", "ffn_gate",
                      "up_proj", "ffn_up",
                      "down_proj", "ffn_down"];

        let mut role_rows: HashMap<String, Vec<Vec<f32>>> = HashMap::new();

        for tensor in &header.tensors {
            if tensor.n_elements < 1024 { continue; }
            let role = roles.iter().find(|r| tensor.name.contains(**r));
            let role_name = match role { Some(r) => *r, None => continue };

            let data = match read_tensor_f32(&mut file, &header, tensor) {
                Ok(d) => d, Err(_) => continue,
            };

            let (n_rows, n_cols) = if tensor.dims.len() >= 2 {
                (tensor.dims[0] as usize, tensor.dims[1..].iter().map(|&d| d as usize).product())
            } else { (1, data.len()) };

            let entry = role_rows.entry(role_name.to_string()).or_default();
            for r in 0..n_rows.min(200) {
                let start = r * n_cols;
                let end = (start + n_cols).min(data.len());
                if end > start { entry.push(data[start..end].to_vec()); }
            }
        }

        // For each role: build 32-centroid codebook, measure Pearson
        let mut any_printed = false;
        for role_name in &roles {
            let rows = match role_rows.get(*role_name) {
                Some(r) if r.len() >= 10 => r,
                _ => continue,
            };

            let k = 32.min(rows.len());
            let centroids = clam_sample(rows, k);

            // Assign each row to nearest centroid
            let assignments: Vec<usize> = rows.iter()
                .map(|row| nearest_centroid(row, &centroids))
                .collect();

            // Sample pairwise: ground truth f32 cosine vs codebook cosine
            let sample = rows.len().min(50);
            let mut gt_cosines = Vec::new();
            let mut cb_cosines = Vec::new();

            for i in 0..sample {
                for j in (i + 1)..sample.min(i + 10) {
                    let gt = cosine(&rows[i], &rows[j]);
                    let cb = cosine(&centroids[assignments[i]], &centroids[assignments[j]]);
                    gt_cosines.push(gt);
                    cb_cosines.push(cb);
                }
            }

            if gt_cosines.len() < 2 { continue; }

            let pearson = pearson_corr(&gt_cosines, &cb_cosines);
            let spearman = spearman_corr(&gt_cosines, &cb_cosines);

            let display_model = if any_printed { "" } else { model_name.as_str() };
            println!("│ {:<22} │ {:<8} │ {:>8} │ {:>8.4} │ {:>5.3} │ {:>9} │",
                display_model, role_name, rows.len(), pearson, spearman, k);
            any_printed = true;
        }

        if any_printed {
            println!("├────────────────────────┼──────────┼──────────┼──────────┼───────┼───────────┤");
        }
    }
    println!("└────────────────────────┴──────────┴──────────┴──────────┴───────┴───────────┘");
}

fn find_gguf_files() -> Vec<(String, String)> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/tmp/hf_cache") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.to_string_lossy().contains("models--") {
                // Walk into snapshots
                let snap_dir = path.join("snapshots");
                if let Ok(snaps) = std::fs::read_dir(&snap_dir) {
                    for snap in snaps.flatten() {
                        if let Ok(gguf_files) = std::fs::read_dir(snap.path()) {
                            for gf in gguf_files.flatten() {
                                let gf_path = gf.path();
                                if gf_path.to_string_lossy().ends_with(".gguf") &&
                                   !gf_path.to_string_lossy().contains("mmproj") {
                                    let model_name = path.file_name()
                                        .map(|n| n.to_string_lossy().replace("models--", "").replace("--", "/"))
                                        .unwrap_or_default();
                                    // Follow symlink to get actual file
                                    let real_path = std::fs::read_link(&gf_path)
                                        .map(|p| {
                                            if p.is_relative() { gf_path.parent().unwrap().join(p) } else { p }
                                        })
                                        .unwrap_or(gf_path.clone());
                                    if real_path.exists() && std::fs::metadata(&real_path).map(|m| m.len() > 1000).unwrap_or(false) {
                                        files.push((model_name, real_path.to_string_lossy().to_string()));
                                    }
                                }
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

fn nearest_centroid(row: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids.iter().enumerate()
        .max_by(|(_, a), (_, b)| cosine(row, a).partial_cmp(&cosine(row, b)).unwrap())
        .map(|(i, _)| i).unwrap_or(0)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n { dot += a[i] as f64 * b[i] as f64; na += (a[i] as f64).powi(2); nb += (b[i] as f64).powi(2); }
    let d = (na * nb).sqrt(); if d < 1e-12 { 0.0 } else { dot / d }
}

fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 { return 0.0; }
    let mx = x[..n].iter().sum::<f64>() / n as f64;
    let my = y[..n].iter().sum::<f64>() / n as f64;
    let (mut cov, mut vx, mut vy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n { let dx = x[i]-mx; let dy = y[i]-my; cov += dx*dy; vx += dx*dx; vy += dy*dy; }
    let d = (vx*vy).sqrt(); if d < 1e-12 { 0.0 } else { cov / d }
}

fn spearman_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 { return 0.0; }
    let rx = ranks(&x[..n]); let ry = ranks(&y[..n]);
    pearson_corr(&rx, &ry)
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut r = vec![0.0f64; n]; let mut i = 0;
    while i < n { let mut j = i+1; while j < n && (idx[j].1-idx[i].1).abs() < 1e-12 { j += 1; }
        let avg = (i+j) as f64 / 2.0 + 0.5; for k in i..j { r[idx[k].0] = avg; } i = j; }
    r
}

// GGUF reader (reused)
struct GgufHeader { tensors: Vec<TensorMeta>, data_offset: u64 }
struct TensorMeta { name: String, dims: Vec<u64>, dtype: u32, offset: u64, n_elements: u64 }
fn parse_gguf_header<R: Read + Seek>(r: &mut R) -> Result<GgufHeader, String> {
    let mut b4=[0u8;4]; let mut b8=[0u8;8];
    r.read_exact(&mut b4).map_err(|e|e.to_string())?;
    if u32::from_le_bytes(b4)!=0x46554747{return Err("bad magic".into());}
    r.read_exact(&mut b4).map_err(|e|e.to_string())?;
    r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nt=u64::from_le_bytes(b8)as usize;
    r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nm=u64::from_le_bytes(b8)as usize;
    for _ in 0..nm{skip_kv(r)?;}
    let mut tensors=Vec::with_capacity(nt);
    for _ in 0..nt{
        r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nl=u64::from_le_bytes(b8)as usize;
        let mut nb=vec![0u8;nl];r.read_exact(&mut nb).map_err(|e|e.to_string())?;
        let name=String::from_utf8_lossy(&nb).to_string();
        r.read_exact(&mut b4).map_err(|e|e.to_string())?;let nd=u32::from_le_bytes(b4)as usize;
        let mut dims=Vec::with_capacity(nd);
        for _ in 0..nd{r.read_exact(&mut b8).map_err(|e|e.to_string())?;dims.push(u64::from_le_bytes(b8));}
        r.read_exact(&mut b4).map_err(|e|e.to_string())?;let dtype=u32::from_le_bytes(b4);
        r.read_exact(&mut b8).map_err(|e|e.to_string())?;let offset=u64::from_le_bytes(b8);
        tensors.push(TensorMeta{name,dims:dims.clone(),dtype,offset,n_elements:dims.iter().product()});
    }
    let pos=r.stream_position().map_err(|e|e.to_string())?;
    Ok(GgufHeader{tensors,data_offset:(pos+31)/32*32})
}
fn skip_kv<R:Read+Seek>(r:&mut R)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b8).map_err(|e|e.to_string())?;let kl=u64::from_le_bytes(b8)as usize;let mut kb=vec![0u8;kl];r.read_exact(&mut kb).map_err(|e|e.to_string())?;r.read_exact(&mut b4).map_err(|e|e.to_string())?;skip_val(r,u32::from_le_bytes(b4))}
fn skip_val<R:Read+Seek>(r:&mut R,vt:u32)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];match vt{0|1|7=>{let mut b=[0u8;1];r.read_exact(&mut b).map_err(|e|e.to_string())?;}2|3=>{r.read_exact(&mut[0u8;2]).map_err(|e|e.to_string())?;}4|5|6=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;}8=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let l=u64::from_le_bytes(b8)as usize;let mut s=vec![0u8;l];r.read_exact(&mut s).map_err(|e|e.to_string())?;}9=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;let et=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let c=u64::from_le_bytes(b8)as usize;for _ in 0..c{skip_val(r,et)?;}}10|11|12=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;}_=>return Err(format!("unknown vtype {}",vt)),}Ok(())}
fn read_tensor_f32<R:Read+Seek>(r:&mut R,h:&GgufHeader,t:&TensorMeta)->Result<Vec<f32>,String>{r.seek(SeekFrom::Start(h.data_offset+t.offset)).map_err(|e|e.to_string())?;let n=t.n_elements as usize;match t.dtype{0=>{let mut buf=vec![0u8;n*4];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())}8=>{let nb=(n+31)/32;let bpb=34;let mut buf=vec![0u8;nb*bpb];r.read_exact(&mut buf).map_err(|e|e.to_string())?;let mut res=Vec::with_capacity(n);for b in 0..nb{let o=b*bpb;let sb=u16::from_le_bytes([buf[o],buf[o+1]]);let s=f32::from_bits((sb as u32)<<16);for i in 0..32{if res.len()>=n{break;}res.push(buf[o+2+i]as i8 as f32*s);}}Ok(res)}1=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|{let bits=u16::from_le_bytes([c[0],c[1]]);let s=((bits>>15)&1)as u32;let e=((bits>>10)&0x1F)as u32;let f=(bits&0x3FF)as u32;if e==0{if f==0{f32::from_bits(s<<31)}else{let v=f as f32/1024.0*2.0f32.powi(-14);if s==1{-v}else{v}}}else if e==31{if f==0{if s==1{f32::NEG_INFINITY}else{f32::INFINITY}}else{f32::NAN}}else{f32::from_bits((s<<31)|((e+127-15)<<23)|(f<<13))}}).collect())}30=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|f32::from_bits((u16::from_le_bytes([c[0],c[1]])as u32)<<16)).collect())}_=>Err(format!("unsupported dtype {}",t.dtype)),}}
