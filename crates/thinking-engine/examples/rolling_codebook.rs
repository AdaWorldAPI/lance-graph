//! Rolling stride codebook: 1:1 exact through stride 2/4/8.
//! Measures Pearson at each stride to find the topology preservation knee.
//! γ offset calibrated at each scale.
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml --example rolling_codebook

use std::io::{Read, Seek, SeekFrom};
use std::collections::HashMap;

fn main() {
    println!("=== Rolling Stride Codebook: 1:1 → stride 2/4/8 ===\n");

    let gguf_files = find_gguf_files();
    if gguf_files.is_empty() { eprintln!("No GGUF files"); return; }

    let strides = [1, 2, 4, 8];

    for (model_name, path) in &gguf_files {
        let mut file = match std::fs::File::open(path) {
            Ok(f) => f, Err(_) => continue,
        };
        let header = match parse_gguf_header(&mut file) {
            Ok(h) => h, Err(_) => continue,
        };

        // Find a good tensor (prefer ffn_down, avoid embed)
        let target = header.tensors.iter()
            .filter(|t| t.n_elements >= 1024 && (t.dtype == 8 || t.dtype == 0 || t.dtype == 1 || t.dtype == 30))
            .find(|t| t.name.contains("ffn_down") && !t.name.contains("embed"))
            .or_else(|| header.tensors.iter()
                .filter(|t| t.n_elements >= 1024 && (t.dtype == 8 || t.dtype == 0 || t.dtype == 1 || t.dtype == 30))
                .find(|t| !t.name.contains("embed") && !t.name.contains("norm")));

        let target = match target { Some(t) => t, None => continue };

        let data = match read_tensor_f32(&mut file, &header, target) {
            Ok(d) => d, Err(_) => continue,
        };

        let (n_rows, n_cols) = if target.dims.len() >= 2 {
            (target.dims[0] as usize, target.dims[1..].iter().map(|&d| d as usize).product())
        } else { (1, data.len()) };

        let rows: Vec<&[f32]> = (0..n_rows)
            .map(|r| &data[r * n_cols..(r * n_cols + n_cols).min(data.len())])
            .filter(|r| !r.is_empty())
            .collect();

        if rows.len() < 16 { continue; }

        let short_name = model_name.split('/').last().unwrap_or(model_name);
        println!("════════════════════════════════════════════════════════════");
        println!("Model: {} ({} rows × {} dims)", short_name, rows.len(), n_cols);
        println!("Tensor: {}", target.name);
        println!("════════════════════════════════════════════════════════════\n");

        println!("Stride │ Centroids │ Table Size │ Pearson  │ Spearman │ γ_cosine");
        println!("───────┼───────────┼────────────┼──────────┼──────────┼─────────");

        for &stride in &strides {
            let centroid_indices: Vec<usize> = (0..rows.len()).step_by(stride).collect();
            let k = centroid_indices.len();

            if k < 4 { continue; }

            // Build distance table at this stride
            // Sample pairwise cosines for Pearson measurement
            let sample = rows.len().min(100);
            let mut gt_cosines = Vec::new();
            let mut cb_cosines = Vec::new();

            // For each sampled pair: gt = f32 cosine, cb = nearest-centroid cosine
            for i in 0..sample {
                for j in (i + 1)..sample.min(i + 5) {
                    let gt = cosine(rows[i], rows[j]);

                    // Assign to nearest centroid (from strided set)
                    let ci = nearest(&centroid_indices, rows[i], &rows);
                    let cj = nearest(&centroid_indices, rows[j], &rows);

                    let cb = cosine(rows[centroid_indices[ci]], rows[centroid_indices[cj]]);
                    gt_cosines.push(gt);
                    cb_cosines.push(cb);
                }
            }

            if gt_cosines.len() < 2 { continue; }

            let pearson = pearson_corr(&gt_cosines, &cb_cosines);
            let spearman = spearman_corr(&gt_cosines, &cb_cosines);

            // γ_cosine calibration for this stride
            let mut all_cosines = Vec::new();
            let cos_sample = k.min(200);
            for i in 0..cos_sample {
                for j in (i + 1)..cos_sample.min(i + 10) {
                    all_cosines.push(cosine(
                        rows[centroid_indices[i]],
                        rows[centroid_indices[j]]
                    ));
                }
            }
            all_cosines.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let center = if all_cosines.is_empty() { 0.0 }
                else { all_cosines[all_cosines.len() / 2] };
            let q1 = if all_cosines.len() > 4 { all_cosines[all_cosines.len() / 4] } else { -1.0 };
            let q3 = if all_cosines.len() > 4 { all_cosines[3 * all_cosines.len() / 4] } else { 1.0 };
            let gamma_cosine = ((q3 - q1) / 2.0).max(0.01);

            let table_bytes = k * k;
            let table_display = if table_bytes > 1_000_000 {
                format!("{:.1} MB", table_bytes as f64 / 1_000_000.0)
            } else {
                format!("{:.0} KB", table_bytes as f64 / 1024.0)
            };

            println!("{:>6} │ {:>9} │ {:>10} │ {:>8.4} │ {:>8.4} │ {:>7.4}",
                stride, k, table_display, pearson, spearman, gamma_cosine);
        }
        println!();

        // Save 1:1 distance table
        let k = rows.len();
        if k <= 8192 { // Only save if table fits reasonably
            let out_dir = format!("/tmp/codebooks/{}-1to1", short_name.replace(".gguf", ""));
            std::fs::create_dir_all(&out_dir).ok();

            let start = std::time::Instant::now();
            let mut table = vec![128u8; k * k];
            let mut min_c = 1.0f64;
            let mut max_c = -1.0f64;
            let mut raw = vec![0.0f64; k * k];

            for i in 0..k {
                raw[i * k + i] = 1.0;
                for j in (i + 1)..k {
                    let c = cosine(rows[i], rows[j]);
                    raw[i * k + j] = c;
                    raw[j * k + i] = c;
                    if c < min_c { min_c = c; }
                    if c > max_c { max_c = c; }
                }
            }
            let range = (max_c - min_c).max(1e-10);
            for idx in 0..k * k {
                table[idx] = (((raw[idx] - min_c) / range) * 255.0).round().clamp(0.0, 255.0) as u8;
            }

            let elapsed = start.elapsed();
            std::fs::write(format!("{}/distance_table_{}x{}.u8", out_dir, k, k), &table).ok();
            println!("  1:1 table saved: {}×{} = {:.1} MB in {:.2}s",
                k, k, (k * k) as f64 / 1_000_000.0, elapsed.as_secs_f64());
            println!("  Cosine range: [{:.4}, {:.4}]", min_c, max_c);
        }
        println!();
    }
}

fn nearest(centroid_indices: &[usize], query: &[f32], all_rows: &[&[f32]]) -> usize {
    centroid_indices.iter().enumerate()
        .max_by(|(_, &a), (_, &b)| cosine(query, all_rows[a]).partial_cmp(&cosine(query, all_rows[b])).unwrap())
        .map(|(i, _)| i).unwrap_or(0)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n { dot += a[i] as f64 * b[i] as f64; na += (a[i] as f64).powi(2); nb += (b[i] as f64).powi(2); }
    let d = (na * nb).sqrt(); if d < 1e-12 { 0.0 } else { dot / d }
}

fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()); if n < 2 { return 0.0; }
    let mx = x[..n].iter().sum::<f64>() / n as f64;
    let my = y[..n].iter().sum::<f64>() / n as f64;
    let (mut c, mut vx, mut vy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n { let dx=x[i]-mx; let dy=y[i]-my; c+=dx*dy; vx+=dx*dx; vy+=dy*dy; }
    let d = (vx*vy).sqrt(); if d < 1e-12 { 0.0 } else { c / d }
}

fn spearman_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()); if n < 2 { return 0.0; }
    let rx = ranks(&x[..n]); let ry = ranks(&y[..n]); pearson_corr(&rx, &ry)
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<(usize, f64)> = v.iter().copied().enumerate().collect();
    idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut r = vec![0.0f64; n]; let mut i = 0;
    while i < n { let mut j=i+1; while j<n&&(idx[j].1-idx[i].1).abs()<1e-12{j+=1;}
        let avg=(i+j)as f64/2.0+0.5; for k in i..j{r[idx[k].0]=avg;} i=j; } r
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

// GGUF reader
struct GgufHeader { tensors: Vec<TensorMeta>, data_offset: u64 }
struct TensorMeta { name: String, dims: Vec<u64>, dtype: u32, offset: u64, n_elements: u64 }
fn parse_gguf_header<R:Read+Seek>(r:&mut R)->Result<GgufHeader,String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b4).map_err(|e|e.to_string())?;if u32::from_le_bytes(b4)!=0x46554747{return Err("bad magic".into());}r.read_exact(&mut b4).map_err(|e|e.to_string())?;r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nt=u64::from_le_bytes(b8)as usize;r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nm=u64::from_le_bytes(b8)as usize;for _ in 0..nm{skip_kv(r)?;}let mut tensors=Vec::with_capacity(nt);for _ in 0..nt{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nl=u64::from_le_bytes(b8)as usize;let mut nb=vec![0u8;nl];r.read_exact(&mut nb).map_err(|e|e.to_string())?;let name=String::from_utf8_lossy(&nb).to_string();r.read_exact(&mut b4).map_err(|e|e.to_string())?;let nd=u32::from_le_bytes(b4)as usize;let mut dims=Vec::with_capacity(nd);for _ in 0..nd{r.read_exact(&mut b8).map_err(|e|e.to_string())?;dims.push(u64::from_le_bytes(b8));}r.read_exact(&mut b4).map_err(|e|e.to_string())?;let dtype=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let offset=u64::from_le_bytes(b8);tensors.push(TensorMeta{name,dims:dims.clone(),dtype,offset,n_elements:dims.iter().product()});}let pos=r.stream_position().map_err(|e|e.to_string())?;Ok(GgufHeader{tensors,data_offset:(pos+31)/32*32})}
fn skip_kv<R:Read+Seek>(r:&mut R)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b8).map_err(|e|e.to_string())?;let kl=u64::from_le_bytes(b8)as usize;let mut kb=vec![0u8;kl];r.read_exact(&mut kb).map_err(|e|e.to_string())?;r.read_exact(&mut b4).map_err(|e|e.to_string())?;skip_val(r,u32::from_le_bytes(b4))}
fn skip_val<R:Read+Seek>(r:&mut R,vt:u32)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];match vt{0|1|7=>{let mut b=[0u8;1];r.read_exact(&mut b).map_err(|e|e.to_string())?;}2|3=>{r.read_exact(&mut[0u8;2]).map_err(|e|e.to_string())?;}4|5|6=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;}8=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let l=u64::from_le_bytes(b8)as usize;let mut s=vec![0u8;l];r.read_exact(&mut s).map_err(|e|e.to_string())?;}9=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;let et=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let c=u64::from_le_bytes(b8)as usize;for _ in 0..c{skip_val(r,et)?;}}10|11|12=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;}_=>return Err(format!("unknown vtype {}",vt)),}Ok(())}
fn read_tensor_f32<R:Read+Seek>(r:&mut R,h:&GgufHeader,t:&TensorMeta)->Result<Vec<f32>,String>{r.seek(SeekFrom::Start(h.data_offset+t.offset)).map_err(|e|e.to_string())?;let n=t.n_elements as usize;match t.dtype{0=>{let mut buf=vec![0u8;n*4];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())}8=>{let nb=(n+31)/32;let bpb=34;let mut buf=vec![0u8;nb*bpb];r.read_exact(&mut buf).map_err(|e|e.to_string())?;let mut res=Vec::with_capacity(n);for b in 0..nb{let o=b*bpb;let sb=u16::from_le_bytes([buf[o],buf[o+1]]);let s=f32::from_bits((sb as u32)<<16);for i in 0..32{if res.len()>=n{break;}res.push(buf[o+2+i]as i8 as f32*s);}}Ok(res)}1=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|{let bits=u16::from_le_bytes([c[0],c[1]]);let s=((bits>>15)&1)as u32;let e=((bits>>10)&0x1F)as u32;let f=(bits&0x3FF)as u32;if e==0{if f==0{f32::from_bits(s<<31)}else{let v=f as f32/1024.0*2.0f32.powi(-14);if s==1{-v}else{v}}}else if e==31{if f==0{if s==1{f32::NEG_INFINITY}else{f32::INFINITY}}else{f32::NAN}}else{f32::from_bits((s<<31)|((e+127-15)<<23)|(f<<13))}}).collect())}30=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|f32::from_bits((u16::from_le_bytes([c[0],c[1]])as u32)<<16)).collect())}_=>Err(format!("unsupported dtype {}",t.dtype)),}}
