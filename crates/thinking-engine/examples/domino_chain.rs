//! Domino chain: Q1→K1→V1 → Q2→K2→V2 → ... perturbation chain.
//!
//! Tests whether perturbation through one role's codebook propagates
//! meaningfully through the distance table to affect other roles.
//!
//! The "domino effect": perturb Q at layer 0, observe which K/V/Gate/Up/Down
//! entries resonate in the distance table. Then feed those into layer 1's Q.
//! Does structure emerge? Or does it flatten to noise?
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml --example domino_chain

use std::io::{Read, Seek, SeekFrom};
use std::collections::HashMap;

fn main() {
    let gguf_path = "/tmp/hf_cache/models--gaianet--jina-embeddings-v3-GGUF/snapshots/d7d998aab1a7f2ea0aa256d8b3f035cbd0af682a/jina-embeddings-v3-Q8_0.gguf";

    if !std::path::Path::new(gguf_path).exists() {
        eprintln!("Jina GGUF not found"); return;
    }

    println!("=== Domino Chain: Q→K→V Perturbation Through Distance Table ===\n");

    let mut file = std::fs::File::open(gguf_path).unwrap();
    let header = parse_gguf_header(&mut file).unwrap();

    // ═══ Step 1: Build per-role codebooks (64 centroids each) ═══

    // Jina uses: attn_qkv (fused Q/K/V), attn_output, ffn_down, ffn_up
    // Llama uses: q_proj, k_proj, v_proj, gate_proj, up_proj, down_proj
    // Match both naming conventions
    let roles = ["attn_qkv", "attn_output", "ffn_down", "ffn_up",
                 "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"];
    let mut role_centroids: HashMap<&str, Vec<Vec<f32>>> = HashMap::new();
    let mut role_rows: HashMap<&str, Vec<Vec<f32>>> = HashMap::new();

    println!("Step 1: Building per-role codebooks...\n");

    for tensor in &header.tensors {
        if tensor.n_elements < 1024 { continue; }
        let role = roles.iter().find(|r| tensor.name.contains(**r));
        let role = match role { Some(r) => *r, None => continue };

        let data = match read_tensor_f32(&mut file, &header, tensor) {
            Ok(d) => d, Err(_) => continue
        };

        let (n_rows, n_cols) = if tensor.dims.len() >= 2 {
            (tensor.dims[0] as usize, tensor.dims[1..].iter().map(|&d| d as usize).product())
        } else { (1, data.len()) };

        let entry = role_rows.entry(role).or_default();
        for r in 0..n_rows.min(200) {
            let start = r * n_cols;
            let end = (start + n_cols).min(data.len());
            if end > start { entry.push(data[start..end].to_vec()); }
        }
    }

    // Build 32-centroid codebook per role (small for speed)
    let k = 32;
    for &role in &roles {
        if let Some(rows) = role_rows.get(role) {
            if rows.len() < k { continue; }
            let centroids = clam_sample(rows, k);
            println!("  {:<10}: {} rows → {} centroids (dim={})",
                role, rows.len(), centroids.len(),
                centroids.first().map_or(0, |c| c.len()));
            role_centroids.insert(role, centroids);
        }
    }

    // ═══ Step 2: Build per-role distance tables ═══

    println!("\nStep 2: Building per-role distance tables...\n");

    let mut role_tables: HashMap<&str, Vec<u8>> = HashMap::new();
    for &role in &roles {
        if let Some(centroids) = role_centroids.get(role) {
            let table = build_cosine_table_u8(centroids);
            println!("  {:<10}: {}×{} table, mean={:.1}",
                role, k, k,
                table.iter().map(|&v| v as f64).sum::<f64>() / table.len() as f64);
            role_tables.insert(role, table);
        }
    }

    // ═══ Step 3: Domino chain — Q perturb → resonance → K→V→Gate→Up→Down ═══

    println!("\nStep 3: Domino chain Q→K→V...\n");

    // Use first available attention role (attn_qkv for Jina, q_proj for Llama)
    let attn_role = role_tables.keys()
        .find(|k| k.contains("attn") || k.contains("q_proj"))
        .copied();
    let q_table = match attn_role.and_then(|r| role_tables.get(r)) {
        Some(t) => t,
        None => { eprintln!("No attention table found"); return; }
    };
    println!("  Using attention role: {:?}", attn_role.unwrap_or("unknown"));

    // Q perturbation: entry 5 fires
    let q_perturb_idx = 5;
    println!("  Perturbing Q[{}]:", q_perturb_idx);

    // Find which Q entries resonate with entry 5
    let q_resonance: Vec<(usize, u8)> = (0..k)
        .map(|j| (j, q_table[q_perturb_idx * k + j]))
        .filter(|&(_, sim)| sim > 160) // high similarity threshold
        .collect();
    println!("    Q resonance (sim > 160): {:?}", q_resonance);

    // Domino: attention resonance → which FFN entries activate?
    // Each role table is a O(1) nested hashtable: table[i][j] = similarity
    let ffn_role = role_tables.keys()
        .find(|k| k.contains("ffn_down") || k.contains("down_proj"))
        .copied();
    if let Some(k_table) = ffn_role.and_then(|r| role_tables.get(r)) {
        println!("    Domino to FFN role: {:?}", ffn_role.unwrap_or("?"));
        // Cross-role: use the Q centroid to find nearest K centroid
        // (In the real engine, Q and K share the distance table — they're in the same matrix)
        // For now: the domino effect is that Q's active indices map to the same row indices in K
        let k_activated: Vec<(usize, u8)> = q_resonance.iter()
            .flat_map(|&(q_idx, _)| {
                // Each Q entry at row idx maps to same-row K entry
                // The K table similarity shows how K[q_idx] relates to other K entries
                (0..k).map(move |j| (j, k_table[q_idx * k + j]))
                    .filter(|&(_, sim)| sim > 160)
            })
            .collect();
        println!("    K activated (via Q domino): {} entries", k_activated.len());

        // Second domino: FFN down → FFN up
        let up_role = role_tables.keys()
            .find(|k| k.contains("ffn_up") || k.contains("up_proj"))
            .copied();
        if let Some(v_table) = up_role.and_then(|r| role_tables.get(r)) {
            println!("    Domino to Up role: {:?}", up_role.unwrap_or("?"));
            let v_activated: Vec<(usize, u8)> = k_activated.iter()
                .flat_map(|&(k_idx, _)| {
                    (0..k).map(move |j| (j, v_table[k_idx * k + j]))
                        .filter(|&(_, sim)| sim > 160)
                })
                .collect();
            println!("    V activated (via K domino): {} entries", v_activated.len());

            // Third domino: Up → output
            let out_role = role_tables.keys()
                .find(|k| k.contains("attn_output") || k.contains("o_proj"))
                .copied();
            if let Some(gate_table) = out_role.and_then(|r| role_tables.get(r)) {
                println!("    Domino to Output role: {:?}", out_role.unwrap_or("?"));
                let gate_activated: Vec<(usize, u8)> = v_activated.iter()
                    .take(10) // limit fan-out
                    .flat_map(|&(v_idx, _)| {
                        (0..k).map(move |j| (j, gate_table[v_idx * k + j]))
                            .filter(|&(_, sim)| sim > 160)
                    })
                    .collect();
                println!("    Gate activated (via V domino): {} entries", gate_activated.len());
            }
        }
    }

    // ═══ Step 4: Energy evolution — full ThinkingEngine cycle on Q table ═══

    println!("\nStep 4: Energy evolution on Q distance table...\n");

    let mut energy = vec![0.0f64; k];
    energy[q_perturb_idx] = 1.0;

    for cycle in 0..10 {
        let mut next = vec![0.0f64; k];
        for i in 0..k {
            if energy[i] < 1e-10 { continue; }
            for j in 0..k {
                next[j] += (q_table[i * k + j] as f64 / 255.0) * energy[i];
            }
        }
        let total: f64 = next.iter().sum();
        if total > 0.0 { for e in &mut next { *e /= total; } }
        energy = next;

        let entropy: f64 = energy.iter()
            .filter(|&&e| e > 1e-15)
            .map(|&e| -e * e.ln())
            .sum();
        let active = energy.iter().filter(|&&e| e > 0.01).count();
        let max_idx = energy.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);

        println!("  Cycle {:2}: entropy={:.3}, active={:2}, peak=Q[{}] ({:.4})",
            cycle, entropy, active, max_idx, energy[max_idx]);
    }

    // ═══ Step 5: Cross-role correlation ═══

    println!("\nStep 5: Cross-role table correlation...\n");

    for i in 0..roles.len() {
        for j in (i + 1)..roles.len() {
            if let (Some(t1), Some(t2)) = (role_tables.get(roles[i]), role_tables.get(roles[j])) {
                // Pearson correlation between the two tables
                let n = t1.len().min(t2.len());
                let mean1 = t1[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
                let mean2 = t2[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
                let mut cov = 0.0f64;
                let mut var1 = 0.0f64;
                let mut var2 = 0.0f64;
                for idx in 0..n {
                    let d1 = t1[idx] as f64 - mean1;
                    let d2 = t2[idx] as f64 - mean2;
                    cov += d1 * d2;
                    var1 += d1 * d1;
                    var2 += d2 * d2;
                }
                let denom = (var1 * var2).sqrt();
                let pearson = if denom > 0.0 { cov / denom } else { 0.0 };
                println!("  {:<10} ↔ {:<10}: Pearson = {:.4}", roles[i], roles[j], pearson);
            }
        }
    }

    println!("\n=== DONE ===");
}

fn clam_sample(vectors: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let k = k.min(n);
    let mut selected = Vec::with_capacity(k);
    let mut max_dist = vec![f64::INFINITY; n];
    selected.push(0);
    for i in 0..n {
        max_dist[i] = 1.0 - cosine(&vectors[i], &vectors[0]);
    }
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

fn build_cosine_table_u8(centroids: &[Vec<f32>]) -> Vec<u8> {
    let k = centroids.len();
    let mut table = vec![128u8; k * k];
    let mut min_cos = 1.0f64;
    let mut max_cos = -1.0f64;
    let mut raw = vec![0.0f64; k * k];
    for i in 0..k {
        raw[i * k + i] = 1.0;
        for j in (i + 1)..k {
            let c = cosine(&centroids[i], &centroids[j]);
            raw[i * k + j] = c;
            raw[j * k + i] = c;
            if c < min_cos { min_cos = c; }
            if c > max_cos { max_cos = c; }
        }
    }
    let range = (max_cos - min_cos).max(1e-10);
    for i in 0..k * k {
        table[i] = (((raw[i] - min_cos) / range) * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    table
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    for i in 0..n { dot += a[i] as f64 * b[i] as f64; na += (a[i] as f64).powi(2); nb += (b[i] as f64).powi(2); }
    let d = (na * nb).sqrt(); if d < 1e-12 { 0.0 } else { dot / d }
}

// ═══ GGUF reader (reused) ═══
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
    1=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;
        Ok(buf.chunks_exact(2).map(|c|{let bits=u16::from_le_bytes([c[0],c[1]]);
        let s=((bits>>15)&1)as u32;let e=((bits>>10)&0x1F)as u32;let f=(bits&0x3FF)as u32;
        if e==0{if f==0{f32::from_bits(s<<31)}else{let v=f as f32/1024.0*2.0f32.powi(-14);if s==1{-v}else{v}}}
        else if e==31{if f==0{if s==1{f32::NEG_INFINITY}else{f32::INFINITY}}else{f32::NAN}}
        else{f32::from_bits((s<<31)|((e+127-15)<<23)|(f<<13))}}).collect())}
    _=>Err(format!("unsupported dtype {}",t.dtype)),}
}
