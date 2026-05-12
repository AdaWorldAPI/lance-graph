//! Build SEMANTIC distance table from token embedding centroids.
//!
//! The attn_q weight table has σ=8.4 (near-orthogonal, no semantic signal).
//! This builds a table from the ACTUAL token embedding clusters:
//!   1. Load codebook_index.u16 (token → centroid mapping)
//!   2. Load token_embd.weight from GGUF (250K × 1024 F16)
//!   3. Average all token embeddings per centroid → 1024 cluster centers
//!   4. Compute pairwise cosine between 1024 centers → 1024² u8 table
//!
//! This table reflects SEMANTIC relationships:
//!   "cat" cluster close to "dog" cluster, far from "quantum" cluster.
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
//!   --example build_semantic_table

use rayon::prelude::*;
use std::io::{Read, Seek, SeekFrom};

const VOCAB_SIZE: usize = 250002;
const HIDDEN_DIM: usize = 1024;
const N_CENTROIDS: usize = 1024;

fn main() {
    println!("═══ BUILD SEMANTIC DISTANCE TABLE ═══\n");

    // Step 1: Load codebook index
    println!("[1] Loading codebook index...");
    let index_data = std::fs::read("/tmp/codebooks/bge-m3-roles-f16/codebook_index.u16")
        .expect("codebook_index.u16 not found — run build_codebook_index first");
    let indices: Vec<u16> = index_data.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(indices.len(), VOCAB_SIZE);
    println!("  {} tokens → {} centroids", indices.len(), N_CENTROIDS);

    // Step 2: Load token embeddings from GGUF
    println!("[2] Loading token embeddings from BGE-M3 GGUF...");
    let gguf_path = find_bge_m3_gguf();
    let mut file = std::fs::File::open(&gguf_path).expect("GGUF not found");
    let header = parse_gguf_header(&mut file).expect("GGUF parse error");

    let embd = header.tensors.iter()
        .find(|t| t.name.contains("token_embd") && t.name.contains("weight"))
        .expect("token_embd.weight not found");
    println!("  tensor: {} dtype={} dims={:?}", embd.name, embd.dtype, embd.dims);

    let embd_data = read_tensor_f32(&mut file, &header, embd).expect("read failed");
    let is_transposed = embd.dims[0] as usize == HIDDEN_DIM;

    // Transpose if needed
    let embd_data = if is_transposed {
        println!("  Transposing {}×{} → {}×{}...", HIDDEN_DIM, VOCAB_SIZE, VOCAB_SIZE, HIDDEN_DIM);
        let mut t = vec![0.0f32; VOCAB_SIZE * HIDDEN_DIM];
        for d in 0..HIDDEN_DIM {
            for v in 0..VOCAB_SIZE {
                t[v * HIDDEN_DIM + d] = embd_data[d * VOCAB_SIZE + v];
            }
        }
        t
    } else {
        embd_data
    };
    println!("  {} embeddings × {} dims", VOCAB_SIZE, HIDDEN_DIM);

    // Step 3: Compute centroid averages
    println!("[3] Computing centroid averages ({} centroids)...", N_CENTROIDS);
    let start = std::time::Instant::now();

    let mut centroid_sums = vec![vec![0.0f64; HIDDEN_DIM]; N_CENTROIDS];
    let mut centroid_counts = vec![0u32; N_CENTROIDS];

    for (tok_id, &centroid) in indices.iter().enumerate() {
        let c = centroid as usize;
        if c >= N_CENTROIDS { continue; }
        centroid_counts[c] += 1;
        let emb = &embd_data[tok_id * HIDDEN_DIM..(tok_id + 1) * HIDDEN_DIM];
        for d in 0..HIDDEN_DIM {
            centroid_sums[c][d] += emb[d] as f64;
        }
    }

    // Average and normalize
    let mut centroids_f32: Vec<Vec<f32>> = Vec::with_capacity(N_CENTROIDS);
    let mut empty = 0;
    for c in 0..N_CENTROIDS {
        if centroid_counts[c] == 0 {
            centroids_f32.push(vec![0.0f32; HIDDEN_DIM]);
            empty += 1;
            continue;
        }
        let n = centroid_counts[c] as f64;
        let avg: Vec<f32> = centroid_sums[c].iter().map(|&s| (s / n) as f32).collect();
        // Normalize
        let norm = avg.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
        if norm < 1e-12 {
            centroids_f32.push(vec![0.0f32; HIDDEN_DIM]);
            empty += 1;
        } else {
            let inv = (1.0 / norm) as f32;
            centroids_f32.push(avg.iter().map(|v| v * inv).collect());
        }
    }
    println!("  Done in {:.1}s. {} empty centroids.", start.elapsed().as_secs_f64(), empty);

    // Show some centroid stats
    let min_count = centroid_counts.iter().filter(|&&c| c > 0).min().copied().unwrap_or(0);
    let max_count = centroid_counts.iter().max().copied().unwrap_or(0);
    let avg_count = VOCAB_SIZE as f64 / N_CENTROIDS as f64;
    println!("  Tokens/centroid: min={} max={} avg={:.0}", min_count, max_count, avg_count);

    // Step 4: Build pairwise cosine distance table
    println!("[4] Building {}×{} semantic distance table (rayon)...", N_CENTROIDS, N_CENTROIDS);
    let start = std::time::Instant::now();

    let pairs: Vec<(usize, usize)> = (0..N_CENTROIDS)
        .flat_map(|i| ((i + 1)..N_CENTROIDS).map(move |j| (i, j)))
        .collect();

    let cosines: Vec<(usize, usize, f32)> = pairs.par_iter()
        .map(|&(i, j)| {
            let dot: f32 = centroids_f32[i].iter().zip(&centroids_f32[j])
                .map(|(a, b)| a * b).sum();
            (i, j, dot.clamp(-1.0, 1.0))
        })
        .collect();

    let mut table = vec![128u8; N_CENTROIDS * N_CENTROIDS];
    let mut min_c = 1.0f32;
    let mut max_c = -1.0f32;

    for i in 0..N_CENTROIDS {
        table[i * N_CENTROIDS + i] = 255;
    }
    for &(i, j, c) in &cosines {
        if c < min_c { min_c = c; }
        if c > max_c { max_c = c; }
        let u = (((c + 1.0) / 2.0) * 255.0).round().clamp(0.0, 255.0) as u8;
        table[i * N_CENTROIDS + j] = u;
        table[j * N_CENTROIDS + i] = u;
    }
    println!("  Done in {:.1}s. cos[{:.3}, {:.3}]", start.elapsed().as_secs_f64(), min_c, max_c);

    // Table statistics
    let vals: Vec<u8> = table.clone();
    let avg_val = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
    let std_val = (vals.iter().map(|&v| { let d = v as f64 - avg_val; d * d }).sum::<f64>() / vals.len() as f64).sqrt();
    let mut sorted = vals.clone();
    sorted.sort_unstable();
    let median = sorted[vals.len() / 2];
    let p75 = sorted[vals.len() * 3 / 4];
    let p90 = sorted[vals.len() * 9 / 10];
    println!("  Table stats: avg={:.1} std={:.1} median={} p75={} p90={}", avg_val, std_val, median, p75, p90);

    // Compare with attn_q table
    if let Ok(attn_q) = std::fs::read("/tmp/codebooks/bge-m3-roles-f16/attn_q/distance_table_1024x1024.u8") {
        let aq_avg = attn_q.iter().map(|&v| v as f64).sum::<f64>() / attn_q.len() as f64;
        let aq_std = (attn_q.iter().map(|&v| { let d = v as f64 - aq_avg; d * d }).sum::<f64>() / attn_q.len() as f64).sqrt();
        println!("\n  COMPARISON:");
        println!("    attn_q (weight):   avg={:.1} std={:.1}  ← too uniform", aq_avg, aq_std);
        println!("    semantic (embeds):  avg={:.1} std={:.1}  ← should be higher", avg_val, std_val);
        if std_val > aq_std * 1.5 {
            println!("    ✓ Semantic table has {:.1}× more variance!", std_val / aq_std);
        } else {
            println!("    ⚠ Semantic table variance not much better");
        }
    }

    // Step 5: Save
    let out_path = "/tmp/codebooks/bge-m3-roles-f16/semantic_distance_1024x1024.u8";
    std::fs::write(out_path, &table).expect("save failed");
    println!("\n[5] Saved: {} ({:.1} MB)", out_path, table.len() as f64 / 1_000_000.0);

    // Step 6: Quick differentiation test
    println!("\n[6] Quick differentiation test...");
    // Check if different centroids have different neighbor sets
    let mut row_tops: Vec<Vec<u16>> = Vec::new();
    for r in 0..N_CENTROIDS {
        let row = &table[r * N_CENTROIDS..(r + 1) * N_CENTROIDS];
        let mut indexed: Vec<(usize, u8)> = row.iter().enumerate().map(|(j, &v)| (j, v)).collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
        let top5: Vec<u16> = indexed.iter().skip(1).take(5).map(|&(j, _)| j as u16).collect();
        row_tops.push(top5);
    }
    // Check overlap between some centroids
    let test_pairs = [(39, 269), (74, 822), (303, 608), (39, 822), (74, 269)];
    for (a, b) in test_pairs {
        let shared: Vec<u16> = row_tops[a].iter().filter(|x| row_tops[b].contains(x)).cloned().collect();
        println!("  centroid {} ↔ {}: {} shared top-5 neighbors", a, b, shared.len());
    }

    println!("\n═══ SEMANTIC TABLE BUILT ═══");
}

fn find_bge_m3_gguf() -> String {
    for entry in std::fs::read_dir("/tmp/hf_cache").expect("no hf_cache") {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.contains("bge-m3") { continue; }
        let snap = entry.path().join("snapshots");
        if let Ok(snaps) = std::fs::read_dir(&snap) {
            for s in snaps.flatten() {
                if let Ok(files) = std::fs::read_dir(s.path()) {
                    for f in files.flatten() {
                        let fp = f.path();
                        let fname = fp.to_string_lossy().to_string();
                        if fname.ends_with(".gguf") && fname.contains("f16") {
                            let real = std::fs::read_link(&fp).map(|r| if r.is_relative() { fp.parent().unwrap().join(r) } else { r }).unwrap_or(fp);
                            return real.to_string_lossy().to_string();
                        }
                    }
                }
            }
        }
    }
    panic!("BGE-M3 F16 GGUF not found in /tmp/hf_cache");
}

struct GgufHeader { tensors: Vec<TensorMeta>, data_offset: u64 }
struct TensorMeta { name: String, dims: Vec<u64>, dtype: u32, offset: u64, n_elements: u64 }
fn parse_gguf_header<R:Read+Seek>(r:&mut R)->Result<GgufHeader,String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b4).map_err(|e|e.to_string())?;if u32::from_le_bytes(b4)!=0x46554747{return Err("bad magic".into());}r.read_exact(&mut b4).map_err(|e|e.to_string())?;r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nt=u64::from_le_bytes(b8)as usize;r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nm=u64::from_le_bytes(b8)as usize;for _ in 0..nm{skip_kv(r)?;}let mut tensors=Vec::with_capacity(nt);for _ in 0..nt{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nl=u64::from_le_bytes(b8)as usize;let mut nb=vec![0u8;nl];r.read_exact(&mut nb).map_err(|e|e.to_string())?;let name=String::from_utf8_lossy(&nb).to_string();r.read_exact(&mut b4).map_err(|e|e.to_string())?;let nd=u32::from_le_bytes(b4)as usize;let mut dims=Vec::with_capacity(nd);for _ in 0..nd{r.read_exact(&mut b8).map_err(|e|e.to_string())?;dims.push(u64::from_le_bytes(b8));}r.read_exact(&mut b4).map_err(|e|e.to_string())?;let dtype=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let offset=u64::from_le_bytes(b8);tensors.push(TensorMeta{name,dims:dims.clone(),dtype,offset,n_elements:dims.iter().product()});}let pos=r.stream_position().map_err(|e|e.to_string())?;Ok(GgufHeader{tensors,data_offset:(pos+31)/32*32})}
fn skip_kv<R:Read+Seek>(r:&mut R)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b8).map_err(|e|e.to_string())?;let kl=u64::from_le_bytes(b8)as usize;let mut kb=vec![0u8;kl];r.read_exact(&mut kb).map_err(|e|e.to_string())?;r.read_exact(&mut b4).map_err(|e|e.to_string())?;skip_val(r,u32::from_le_bytes(b4))}
fn skip_val<R:Read+Seek>(r:&mut R,vt:u32)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];match vt{0|1|7=>{let mut b=[0u8;1];r.read_exact(&mut b).map_err(|e|e.to_string())?;}2|3=>{r.read_exact(&mut[0u8;2]).map_err(|e|e.to_string())?;}4|5|6=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;}8=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let l=u64::from_le_bytes(b8)as usize;let mut s=vec![0u8;l];r.read_exact(&mut s).map_err(|e|e.to_string())?;}9=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;let et=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let c=u64::from_le_bytes(b8)as usize;for _ in 0..c{skip_val(r,et)?;}}10|11|12=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;}_=>return Err(format!("unknown vtype {}",vt)),}Ok(())}
fn read_tensor_f32<R:Read+Seek>(r:&mut R,h:&GgufHeader,t:&TensorMeta)->Result<Vec<f32>,String>{r.seek(SeekFrom::Start(h.data_offset+t.offset)).map_err(|e|e.to_string())?;let n=t.n_elements as usize;match t.dtype{0=>{let mut buf=vec![0u8;n*4];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())}1=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|{let bits=u16::from_le_bytes([c[0],c[1]]);let s=((bits>>15)&1)as u32;let e=((bits>>10)&0x1F)as u32;let f=(bits&0x3FF)as u32;if e==0{if f==0{f32::from_bits(s<<31)}else{let v=f as f32/1024.0*2.0f32.powi(-14);if s==1{-v}else{v}}}else if e==31{if f==0{if s==1{f32::NEG_INFINITY}else{f32::INFINITY}}else{f32::NAN}}else{f32::from_bits((s<<31)|((e+127-15)<<23)|(f<<13))}}).collect())}30=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|f32::from_bits((u16::from_le_bytes([c[0],c[1]])as u32)<<16)).collect())}_=>Err(format!("unsupported dtype {}",t.dtype)),}}
