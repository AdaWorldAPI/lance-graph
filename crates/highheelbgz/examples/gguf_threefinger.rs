//! Three-finger validation on real Jina GGUF weights.
//!
//! Key questions:
//! 1. Does Foveal cosine >> Reject cosine on real weights? (spatial locality)
//! 2. What % of pairs does HEEL reject with zero data access?
//! 3. Does stride-as-role match the actual tensor naming?
//! 4. What's the optimal (start, stride) for this model?
//!
//! cargo run --release --manifest-path crates/highheelbgz/Cargo.toml --example gguf_threefinger

use highheelbgz::*;
use std::io::{Read, Seek, SeekFrom};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/tmp/hf_cache/models--gaianet--jina-embeddings-v3-GGUF/snapshots/d7d998aab1a7f2ea0aa256d8b3f035cbd0af682a/jina-embeddings-v3-Q8_0.gguf".into()
    });

    println!("=== Three-Finger Validation on Real GGUF ===\n");

    let mut file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => { eprintln!("Cannot open: {}", e); return; }
    };
    let header = match parse_gguf_header(&mut file) {
        Ok(h) => h,
        Err(e) => { eprintln!("Parse error: {}", e); return; }
    };
    println!("Model: {} tensors\n", header.tensors.len());

    // Collect f32 weight rows by role (from tensor name)
    let mut role_rows: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
    let mut tensors_done = 0;

    for tensor in &header.tensors {
        if tensor.n_elements < 1024 { continue; }
        let data = match read_tensor_f32(&mut file, &header, tensor) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let (n_rows, n_cols) = if tensor.dims.len() >= 2 {
            (tensor.dims[0] as usize, tensor.dims[1..].iter().map(|&d| d as usize).product())
        } else { (1, data.len()) };

        let mut rows = Vec::new();
        for r in 0..n_rows.min(100) {
            let start = r * n_cols;
            let end = (start + n_cols).min(data.len());
            if end > start { rows.push(data[start..end].to_vec()); }
        }
        if !rows.is_empty() {
            role_rows.push((tensor.name.clone(), rows));
        }
        tensors_done += 1;
        if tensors_done % 20 == 0 { eprint!("\r{} tensors...", tensors_done); }
    }
    eprintln!("\r{} tensors, {} with rows\n", tensors_done, role_rows.len());

    // ═══ Part 1: Spatial locality — does adjacent offset give higher cosine? ═══
    println!("=== Part 1: Spatial Locality (Foveal vs Reject cosine gap) ===\n");

    let test_configs = [
        ("Foveal: offset=1, stride=8",  20u32, 21u32, 8u32, 8u32),
        ("Near:   offset=5, stride=8",  20,    25,    8,    8),
        ("Maybe:  offset=20, stride=8", 20,    40,    8,    8),
        ("Reject: same pos, stride 8v2",20,    20,    8,    2),
        ("Reject: disjoint, stride=8",  20,    200,   8,    8),
    ];

    // Pick 5 tensors with enough rows
    let test_tensors: Vec<&(String, Vec<Vec<f32>>)> = role_rows.iter()
        .filter(|(_, rows)| rows.len() >= 20 && rows[0].len() >= 512)
        .take(5)
        .collect();

    for (tname, rows) in &test_tensors {
        let short_name = tname.split('.').last().unwrap_or(tname);
        println!("--- {} ({} rows, dim={}) ---", short_name, rows.len(), rows[0].len());

        for (label, s1, s2, st1, st2) in &test_configs {
            let addr_a = SpiralAddress::new(*s1, *st1, 4);
            let addr_b = SpiralAddress::new(*s2, *st2, 4);
            let band = addr_a.coarse_band(&addr_b);

            // Compute actual walk cosine on several row pairs
            let n_sample = rows.len().min(20);
            let mut cosines = Vec::new();
            for i in 0..n_sample {
                let wa = SpiralWalk::execute(&addr_a, &rows[i]);
                let wb = SpiralWalk::execute(&addr_b, &rows[i]);
                cosines.push(wa.cosine(&wb));
            }
            let mean_cos = cosines.iter().sum::<f64>() / cosines.len() as f64;
            let min_cos = cosines.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_cos = cosines.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            println!("  {:>35} → {:<8?}  cos: mean={:.4} [{:.4}, {:.4}]",
                label, band, mean_cos, min_cos, max_cos);
        }
        println!();
    }

    // ═══ Part 2: HEEL rejection rate ═══════════════════════════════════════
    println!("=== Part 2: HEEL Rejection Rate (three-finger, zero data access) ===\n");

    // Generate addresses: one per row per tensor, with role-appropriate stride
    let mut all_addresses: Vec<(SpiralAddress, String)> = Vec::new();
    for (tname, rows) in &role_rows {
        let stride = if tname.contains("q_proj") || tname.contains("attn_q") { 3 }
            else if tname.contains("k_proj") || tname.contains("attn_k") { 3 }
            else if tname.contains("v_proj") || tname.contains("attn_v") { 5 }
            else if tname.contains("gate") || tname.contains("ffn_gate") { 8 }
            else if tname.contains("up_proj") || tname.contains("ffn_up") { 2 }
            else if tname.contains("down_proj") || tname.contains("ffn_down") { 4 }
            else { 6 }; // default

        for (r, _) in rows.iter().enumerate().take(20) {
            let addr = SpiralAddress::new(20 + r as u32, stride, 4);
            all_addresses.push((addr, tname.clone()));
        }
    }

    let n_addrs = all_addresses.len();
    let max_pairs = 5000;
    let mut foveal = 0u32;
    let mut near = 0u32;
    let mut maybe = 0u32;
    let mut reject = 0u32;
    let mut total = 0u32;

    let mut pair_count = 0;
    'outer: for i in 0..n_addrs {
        for j in (i+1)..n_addrs {
            if pair_count >= max_pairs { break 'outer; }
            pair_count += 1;
            total += 1;
            match all_addresses[i].0.coarse_band(&all_addresses[j].0) {
                CoarseBand::Foveal => foveal += 1,
                CoarseBand::Near => near += 1,
                CoarseBand::Maybe => maybe += 1,
                CoarseBand::Reject => reject += 1,
            }
        }
    }

    let pct = |n: u32| n as f64 / total as f64 * 100.0;
    println!("{} pairs evaluated:", total);
    println!("  Foveal: {:>5} ({:.1}%)", foveal, pct(foveal));
    println!("  Near:   {:>5} ({:.1}%)", near, pct(near));
    println!("  Maybe:  {:>5} ({:.1}%)", maybe, pct(maybe));
    println!("  Reject: {:>5} ({:.1}%)", reject, pct(reject));
    println!("  HEEL elimination: {:.1}% (Reject only)", pct(reject));
    println!("  HEEL+Near skip:   {:.1}% (Reject + Maybe)", pct(reject + maybe));

    // ═══ Part 3: Calibrate optimal address for this model ═══════════════
    println!("\n=== Part 3: Calibrate Optimal Address ===\n");

    // Use first tensor with enough rows
    if let Some((tname, rows)) = role_rows.iter().find(|(_, r)| r.len() >= 30 && r[0].len() >= 512) {
        println!("Calibrating on: {} ({} rows, dim={})", tname, rows.len(), rows[0].len());
        let (best, sp) = calibrate(rows, 5..40, 2..16, &[4, 8]);
        println!("  Best: start={}, stride={}, length={}", best.start, best.stride, best.length);
        println!("  Spearman vs f32 ground truth: {:.4}", sp);

        // Compare with default (20, 8, 4)
        let default_addr = SpiralAddress::new(20, 8, 4);
        let walks_default: Vec<SpiralWalk> = rows[..30].iter()
            .map(|r| SpiralWalk::execute(&default_addr, r)).collect();
        let walks_best: Vec<SpiralWalk> = rows[..30].iter()
            .map(|r| SpiralWalk::execute(&best, r)).collect();

        let mut gt = Vec::new();
        let mut cos_default = Vec::new();
        let mut cos_best = Vec::new();
        for i in 0..30 { for j in (i+1)..30 {
            gt.push(cosine_f32(&rows[i], &rows[j]));
            cos_default.push(walks_default[i].cosine(&walks_default[j]));
            cos_best.push(walks_best[i].cosine(&walks_best[j]));
        }}
        println!("  Default (20,8,4) Spearman: {:.4}", spearman_local(&gt, &cos_default));
        println!("  Best ({},{},{}) Spearman: {:.4}", best.start, best.stride, best.length, spearman_local(&gt, &cos_best));
    }

    // ═══ Part 4: Cross-role three-finger validation ═══════════════════════
    println!("\n=== Part 4: Cross-Role Three-Finger ===\n");
    println!("Same-role pairs should be Near/Maybe. Cross-role should be Reject.\n");

    let role_names = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"];
    let role_strides = [3u32, 3, 5, 8, 2, 4];

    for i in 0..role_names.len() {
        for j in (i+1)..role_names.len() {
            let a = SpiralAddress::new(20, role_strides[i], 4);
            let b = SpiralAddress::new(20, role_strides[j], 4);
            let band = a.coarse_band(&b);
            let same_stride = role_strides[i] == role_strides[j];
            println!("  {} ↔ {}: stride {}v{} → {:?} {}",
                role_names[i], role_names[j],
                role_strides[i], role_strides[j],
                band,
                if same_stride { "(SAME stride → comparable)" } else { "(DIFFERENT stride → reject)" });
        }
    }

    println!("\n=== DONE ===");
}

// ═══ Helpers ═════════════════════════════════════════════════════════════

fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    for i in 0..n { dot += a[i] as f64 * b[i] as f64; na += (a[i] as f64).powi(2); nb += (b[i] as f64).powi(2); }
    let d = (na * nb).sqrt(); if d < 1e-12 { 0.0 } else { dot / d }
}

fn spearman_local(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 { return 0.0; }
    let rx = ranks(x); let ry = ranks(y);
    let mx = rx.iter().sum::<f64>() / n as f64;
    let my = ry.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0f64; let mut vx = 0.0f64; let mut vy = 0.0f64;
    for i in 0..n { let dx = rx[i]-mx; let dy = ry[i]-my; cov += dx*dy; vx += dx*dx; vy += dy*dy; }
    let d = (vx*vy).sqrt(); if d < 1e-12 { 0.0 } else { cov / d }
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0f64; n]; let mut i = 0;
    while i < n { let mut j = i+1; while j < n && (idx[j].1-idx[i].1).abs() < 1e-12 { j += 1; }
        let avg = (i+j) as f64 / 2.0 + 0.5; for k in i..j { r[idx[k].0] = avg; } i = j; }
    r
}

// ═══ GGUF reader ═════════════════════════════════════════════════════════

struct GgufHeader { tensors: Vec<TensorMeta>, data_offset: u64 }
struct TensorMeta { name: String, dims: Vec<u64>, dtype: u32, offset: u64, n_elements: u64 }

fn parse_gguf_header<R: Read + Seek>(r: &mut R) -> Result<GgufHeader, String> {
    let mut b4=[0u8;4]; let mut b8=[0u8;8];
    r.read_exact(&mut b4).map_err(|e|e.to_string())?;
    if u32::from_le_bytes(b4) != 0x46554747 { return Err("bad magic".into()); }
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
fn skip_kv<R:Read+Seek>(r:&mut R)->Result<(),String>{
    let mut b4=[0u8;4];let mut b8=[0u8;8];
    r.read_exact(&mut b8).map_err(|e|e.to_string())?;let kl=u64::from_le_bytes(b8)as usize;
    let mut kb=vec![0u8;kl];r.read_exact(&mut kb).map_err(|e|e.to_string())?;
    r.read_exact(&mut b4).map_err(|e|e.to_string())?;skip_val(r,u32::from_le_bytes(b4))
}
fn skip_val<R:Read+Seek>(r:&mut R,vt:u32)->Result<(),String>{
    let mut b4=[0u8;4];let mut b8=[0u8;8];
    match vt{0|1|7=>{let mut b=[0u8;1];r.read_exact(&mut b).map_err(|e|e.to_string())?;}
    2|3=>{r.read_exact(&mut[0u8;2]).map_err(|e|e.to_string())?;}
    4|5|6=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;}
    8=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let l=u64::from_le_bytes(b8)as usize;let mut s=vec![0u8;l];r.read_exact(&mut s).map_err(|e|e.to_string())?;}
    9=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;let et=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let c=u64::from_le_bytes(b8)as usize;for _ in 0..c{skip_val(r,et)?;}}
    10|11|12=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;}
    _=>return Err(format!("unknown vtype {}",vt)),}Ok(())
}
fn read_tensor_f32<R:Read+Seek>(r:&mut R,h:&GgufHeader,t:&TensorMeta)->Result<Vec<f32>,String>{
    r.seek(SeekFrom::Start(h.data_offset+t.offset)).map_err(|e|e.to_string())?;
    let n=t.n_elements as usize;
    match t.dtype{
    0=>{let mut buf=vec![0u8;n*4];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())}
    8=>{let nb=(n+31)/32;let bpb=34;let mut buf=vec![0u8;nb*bpb];r.read_exact(&mut buf).map_err(|e|e.to_string())?;
        let mut res=Vec::with_capacity(n);for b in 0..nb{let o=b*bpb;let sb=u16::from_le_bytes([buf[o],buf[o+1]]);let s=f32::from_bits((sb as u32)<<16);
        for i in 0..32{if res.len()>=n{break;}res.push(buf[o+2+i]as i8 as f32*s);}}Ok(res)}
    _=>Err(format!("unsupported dtype {}",t.dtype)),}
}
