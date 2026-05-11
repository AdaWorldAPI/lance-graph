//! Jina F16 GGUF → token embedding centroids → HDR-encoded distance table
//!
//! The pipeline:
//! 1. Read Jina F16 GGUF token_embd.weight (vocab × 1024)
//! 2. Build codebook: CLAM furthest-point sampling → 256 centroids
//! 3. Assign tokens to nearest centroid
//! 4. Average embeddings per centroid → 256 semantic cluster centers
//! 5. Pairwise cosine → HDR encode (Belichtungsmesser ¼σ bands)
//! 6. Save as const-ready binary
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
//!   --example jina_hdr_table

use rayon::prelude::*;
use std::io::{Read, Seek, SeekFrom};

const HIDDEN_DIM: usize = 1024;
const N_CENTROIDS: usize = 256; // L2 tier: 256² = 64 KB, L2-cache resident

fn main() {
    println!("═══ JINA F16 → HDR SEMANTIC TABLE ═══\n");

    // ── Step 1: Read Jina token embeddings ──
    let args: Vec<String> = std::env::args().collect();
    let gguf_path = if args.len() > 1 {
        args[1].clone()
    } else {
        find_jina_f16_gguf()
    };
    println!("[1] Reading Jina F16 GGUF: {}", &gguf_path[gguf_path.len().saturating_sub(40)..]);
    let mut file = std::fs::File::open(&gguf_path).expect("open");
    let header = parse_gguf_header(&mut file).expect("parse");
    println!("  {} tensors", header.tensors.len());

    // List embedding-related tensors
    for t in &header.tensors {
        if t.name.contains("embd") || t.name.contains("embed") {
            println!("  candidate: {} dtype={} dims={:?} elems={}", t.name, t.dtype, t.dims, t.n_elements);
        }
    }
    let embd = header.tensors.iter()
        .find(|t| (t.name.contains("token_embd") || t.name.contains("token_embed"))
            && t.name.ends_with("weight") && t.n_elements > 10000)
        .expect("token embedding weight not found — check tensor names above");
    println!("  {} dtype={} dims={:?}", embd.name, embd.dtype, embd.dims);

    let raw = read_tensor_f32(&mut file, &header, embd).expect("read");
    let vocab_size = if embd.dims[0] as usize == HIDDEN_DIM {
        embd.dims[1] as usize
    } else {
        embd.dims[0] as usize
    };
    let is_transposed = embd.dims[0] as usize == HIDDEN_DIM;
    println!("  vocab={} hidden={} transposed={}", vocab_size, HIDDEN_DIM, is_transposed);

    // Transpose if needed
    let embeddings: Vec<f32> = if is_transposed {
        print!("  Transposing...");
        let mut t = vec![0.0f32; vocab_size * HIDDEN_DIM];
        for d in 0..HIDDEN_DIM {
            for v in 0..vocab_size {
                t[v * HIDDEN_DIM + d] = raw[d * vocab_size + v];
            }
        }
        println!(" done");
        t
    } else { raw };

    // ── Step 2: CLAM furthest-point sampling → 256 centroids ──
    println!("[2] CLAM furthest-point: {} centroids from {} tokens...", N_CENTROIDS, vocab_size);
    let start = std::time::Instant::now();

    // Normalize all embeddings first
    let normed: Vec<Vec<f32>> = (0..vocab_size).map(|v| {
        let row = &embeddings[v * HIDDEN_DIM..(v + 1) * HIDDEN_DIM];
        let norm = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        if norm < 1e-12 { vec![0.0f32; HIDDEN_DIM] }
        else { let inv = (1.0 / norm) as f32; row.iter().map(|x| x * inv).collect() }
    }).collect();

    // CLAM: start from token 0, repeatedly pick the farthest point
    let mut selected = vec![0usize];
    let mut min_dist = vec![f64::INFINITY; vocab_size];

    // Compute initial distances to token 0
    for v in 0..vocab_size {
        let dot: f32 = normed[v].iter().zip(&normed[0]).map(|(a, b)| a * b).sum();
        min_dist[v] = 1.0 - dot as f64;
    }

    for k in 1..N_CENTROIDS {
        // Find the farthest point
        let next = min_dist.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        selected.push(next);

        // Update min distances
        for v in 0..vocab_size {
            let dot: f32 = normed[v].iter().zip(&normed[next]).map(|(a, b)| a * b).sum();
            let d = 1.0 - dot as f64;
            if d < min_dist[v] { min_dist[v] = d; }
        }

        if k % 50 == 0 { eprint!("  {}/{}\r", k, N_CENTROIDS); }
    }
    println!("  {} centroids in {:.1}s", selected.len(), start.elapsed().as_secs_f64());

    // ── Step 3: Assign all tokens to nearest centroid ──
    println!("[3] Assigning {} tokens to {} centroids (rayon)...", vocab_size, N_CENTROIDS);
    let start = std::time::Instant::now();

    let centroid_vecs: Vec<&[f32]> = selected.iter()
        .map(|&i| normed[i].as_slice()).collect();

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

    // ── Step 4: Average embeddings per centroid ──
    println!("[4] Averaging embeddings per centroid...");
    let mut centroid_sums = vec![vec![0.0f64; HIDDEN_DIM]; N_CENTROIDS];
    let mut counts = vec![0u32; N_CENTROIDS];
    for (v, &c) in assignments.iter().enumerate() {
        counts[c as usize] += 1;
        let row = &embeddings[v * HIDDEN_DIM..(v + 1) * HIDDEN_DIM];
        for d in 0..HIDDEN_DIM {
            centroid_sums[c as usize][d] += row[d] as f64;
        }
    }
    let centroids_avg: Vec<Vec<f32>> = (0..N_CENTROIDS).map(|c| {
        if counts[c] == 0 { return vec![0.0f32; HIDDEN_DIM]; }
        let n = counts[c] as f64;
        let avg: Vec<f32> = centroid_sums[c].iter().map(|&s| (s / n) as f32).collect();
        let norm = avg.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
        if norm < 1e-12 { vec![0.0f32; HIDDEN_DIM] }
        else { let inv = (1.0 / norm) as f32; avg.iter().map(|v| v * inv).collect() }
    }).collect();

    let empty = counts.iter().filter(|&&c| c == 0).count();
    let min_c = counts.iter().filter(|&&c| c > 0).min().copied().unwrap_or(0);
    let max_c = counts.iter().max().copied().unwrap_or(0);
    println!("  tokens/centroid: min={} max={} empty={}", min_c, max_c, empty);

    // ── Step 5: Pairwise cosine → HDR encode ──
    println!("[5] Building {}×{} distance table with HDR encoding...", N_CENTROIDS, N_CENTROIDS);
    let start = std::time::Instant::now();

    // First pass: collect all cosine values
    let mut raw_cosines = vec![0.0f32; N_CENTROIDS * N_CENTROIDS];
    let mut all_cos: Vec<f32> = Vec::with_capacity(N_CENTROIDS * (N_CENTROIDS - 1) / 2);
    for i in 0..N_CENTROIDS {
        raw_cosines[i * N_CENTROIDS + i] = 1.0;
        for j in (i+1)..N_CENTROIDS {
            let dot: f32 = centroids_avg[i].iter().zip(&centroids_avg[j]).map(|(a, b)| a * b).sum();
            let cos = dot.clamp(-1.0, 1.0);
            raw_cosines[i * N_CENTROIDS + j] = cos;
            raw_cosines[j * N_CENTROIDS + i] = cos;
            all_cos.push(cos);
        }
    }

    // HDR Belichtungsmesser: compute distribution statistics
    all_cos.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n_pairs = all_cos.len();
    let cos_min = all_cos[0];
    let cos_max = all_cos[n_pairs - 1];
    let cos_median = all_cos[n_pairs / 2];
    let cos_mean: f32 = all_cos.iter().sum::<f32>() / n_pairs as f32;
    let cos_std = (all_cos.iter().map(|c| { let d = c - cos_mean; d * d }).sum::<f32>() / n_pairs as f32).sqrt();
    let cos_p25 = all_cos[n_pairs / 4];
    let cos_p75 = all_cos[n_pairs * 3 / 4];
    let iqr = cos_p75 - cos_p25;

    println!("  Cosine distribution:");
    println!("    min={:.3} p25={:.3} median={:.3} p75={:.3} max={:.3}", cos_min, cos_p25, cos_median, cos_p75, cos_max);
    println!("    mean={:.3} std={:.3} iqr={:.3}", cos_mean, cos_std, iqr);

    // HDR encoding: 12 ¼σ bands
    // Map cosine to u8 using the ACTUAL distribution, not linear.
    // Values in the dense center get more u8 levels (higher resolution).
    // Values in the sparse tails get fewer levels (less resolution needed).
    let mut table = vec![0u8; N_CENTROIDS * N_CENTROIDS];
    for i in 0..N_CENTROIDS {
        table[i * N_CENTROIDS + i] = 255; // self
        for j in (i+1)..N_CENTROIDS {
            let cos = raw_cosines[i * N_CENTROIDS + j];
            // HDR: map via CDF position (percentile rank)
            let rank = all_cos.partition_point(|&c| c <= cos);
            let percentile = rank as f32 / n_pairs as f32;
            // Map percentile [0,1] → u8 [0,254] (255 reserved for self)
            let u = (percentile * 254.0).round().clamp(0.0, 254.0) as u8;
            table[i * N_CENTROIDS + j] = u;
            table[j * N_CENTROIDS + i] = u;
        }
    }

    // HDR table stats
    let t_avg = table.iter().map(|&v| v as f64).sum::<f64>() / table.len() as f64;
    let t_std = (table.iter().map(|&v| { let d = v as f64 - t_avg; d * d }).sum::<f64>() / table.len() as f64).sqrt();
    let mut t_sorted = table.clone();
    t_sorted.sort_unstable();
    let t_p75 = t_sorted[table.len() * 3 / 4];
    println!("  HDR table: avg={:.1} std={:.1} p75={}", t_avg, t_std, t_p75);
    println!("  Compare: linear attn_q std=8.4, linear semantic std=6.8");
    println!("  Done in {:.1}s", start.elapsed().as_secs_f64());

    // ── Step 6: Save ──
    let out_dir = "/tmp/codebooks/jina-v3-hdr";
    std::fs::create_dir_all(out_dir).ok();

    let table_path = format!("{}/distance_table_{}x{}.u8", out_dir, N_CENTROIDS, N_CENTROIDS);
    std::fs::write(&table_path, &table).expect("save table");

    let index_path = format!("{}/codebook_index.u16", out_dir);
    let index_bytes: Vec<u8> = assignments.iter().flat_map(|&a| a.to_le_bytes()).collect();
    std::fs::write(&index_path, &index_bytes).expect("save index");

    println!("\n[6] Saved:");
    println!("  {} ({:.1} KB)", table_path, table.len() as f64 / 1024.0);
    println!("  {} ({:.1} KB)", index_path, index_bytes.len() as f64 / 1024.0);

    // ── Step 7: Quick cascade test ──
    println!("\n[7] Quick cascade test on HDR table:");
    let engine = thinking_engine::engine::ThinkingEngine::new(table.clone());
    println!("  floor={}", engine.floor);

    let cascade = thinking_engine::domino::DominoCascade::new(&engine, &counts);

    // Map some test words through the codebook
    let test = ["wound", "light", "ocean", "fire", "grief", "joy", "cat", "stock market"];
    // Use first token of each word via the Jina tokenizer (if available) or hash
    for word in &test {
        let hash = word.bytes().fold(5381u64, |h, b| h.wrapping_mul(33).wrapping_add(b as u64));
        let fake_centroid = (hash % N_CENTROIDS as u64) as u16;
        let (dom, stages, dis) = cascade.think(&[fake_centroid]);
        let chain_len = stages.len();
        let staunen_max = stages.iter().map(|s| s.markers.staunen).fold(0.0f32, f32::max);
        let wisdom_max = stages.iter().map(|s| s.markers.wisdom).fold(0.0f32, f32::max);
        println!("  {:<15} → centroid {:>3} → dom {:>3}  stages={}  dis={:.2}  ✨{:.2} 🦉{:.2}",
            word, fake_centroid, dom, chain_len, dis.total_dissonance, staunen_max, wisdom_max);
    }

    println!("\n═══ JINA HDR TABLE COMPLETE ═══");
    println!("  {} centroids, CLAM sampled, HDR encoded", N_CENTROIDS);
    println!("  Table std={:.1} (vs linear 6.8) — HDR redistributes u8 levels to topology", t_std);
}

// ═══ GGUF helpers (reused from build_1to1_roles) ═══

fn find_jina_f16_gguf() -> String {
    for entry in std::fs::read_dir("/tmp/hf_cache").expect("no hf_cache") {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.contains("jina") { continue; }
        let snap = entry.path().join("snapshots");
        if let Ok(snaps) = std::fs::read_dir(&snap) {
            for s in snaps.flatten() {
                if let Ok(files) = std::fs::read_dir(s.path()) {
                    for f in files.flatten() {
                        let fp = f.path();
                        let fname = fp.to_string_lossy().to_string();
                        if fname.ends_with(".gguf") && fname.contains("f16") {
                            let real = std::fs::read_link(&fp)
                                .map(|r| if r.is_relative() { fp.parent().unwrap().join(r) } else { r })
                                .unwrap_or(fp);
                            return real.to_string_lossy().to_string();
                        }
                    }
                }
            }
        }
    }
    panic!("Jina F16 GGUF not found in /tmp/hf_cache");
}

struct GgufHeader { tensors: Vec<TensorMeta>, data_offset: u64 }
struct TensorMeta { name: String, dims: Vec<u64>, dtype: u32, offset: u64, n_elements: u64 }
fn parse_gguf_header<R:Read+Seek>(r:&mut R)->Result<GgufHeader,String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b4).map_err(|e|e.to_string())?;if u32::from_le_bytes(b4)!=0x46554747{return Err("bad magic".into());}r.read_exact(&mut b4).map_err(|e|e.to_string())?;r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nt=u64::from_le_bytes(b8)as usize;r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nm=u64::from_le_bytes(b8)as usize;for _ in 0..nm{skip_kv(r)?;}let mut tensors=Vec::with_capacity(nt);for _ in 0..nt{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let nl=u64::from_le_bytes(b8)as usize;let mut nb=vec![0u8;nl];r.read_exact(&mut nb).map_err(|e|e.to_string())?;let name=String::from_utf8_lossy(&nb).to_string();r.read_exact(&mut b4).map_err(|e|e.to_string())?;let nd=u32::from_le_bytes(b4)as usize;let mut dims=Vec::with_capacity(nd);for _ in 0..nd{r.read_exact(&mut b8).map_err(|e|e.to_string())?;dims.push(u64::from_le_bytes(b8));}r.read_exact(&mut b4).map_err(|e|e.to_string())?;let dtype=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let offset=u64::from_le_bytes(b8);tensors.push(TensorMeta{name,dims:dims.clone(),dtype,offset,n_elements:dims.iter().product()});}let pos=r.stream_position().map_err(|e|e.to_string())?;Ok(GgufHeader{tensors,data_offset:(pos+31)/32*32})}
fn skip_kv<R:Read+Seek>(r:&mut R)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];r.read_exact(&mut b8).map_err(|e|e.to_string())?;let kl=u64::from_le_bytes(b8)as usize;let mut kb=vec![0u8;kl];r.read_exact(&mut kb).map_err(|e|e.to_string())?;r.read_exact(&mut b4).map_err(|e|e.to_string())?;skip_val(r,u32::from_le_bytes(b4))}
fn skip_val<R:Read+Seek>(r:&mut R,vt:u32)->Result<(),String>{let mut b4=[0u8;4];let mut b8=[0u8;8];match vt{0|1|7=>{let mut b=[0u8;1];r.read_exact(&mut b).map_err(|e|e.to_string())?;}2|3=>{r.read_exact(&mut[0u8;2]).map_err(|e|e.to_string())?;}4|5|6=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;}8=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;let l=u64::from_le_bytes(b8)as usize;let mut s=vec![0u8;l];r.read_exact(&mut s).map_err(|e|e.to_string())?;}9=>{r.read_exact(&mut b4).map_err(|e|e.to_string())?;let et=u32::from_le_bytes(b4);r.read_exact(&mut b8).map_err(|e|e.to_string())?;let c=u64::from_le_bytes(b8)as usize;for _ in 0..c{skip_val(r,et)?;}}10|11|12=>{r.read_exact(&mut b8).map_err(|e|e.to_string())?;}_=>return Err(format!("unknown vtype {}",vt)),}Ok(())}
fn read_tensor_f32<R:Read+Seek>(r:&mut R,h:&GgufHeader,t:&TensorMeta)->Result<Vec<f32>,String>{r.seek(SeekFrom::Start(h.data_offset+t.offset)).map_err(|e|e.to_string())?;let n=t.n_elements as usize;match t.dtype{0=>{let mut buf=vec![0u8;n*4];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())}1=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|{let bits=u16::from_le_bytes([c[0],c[1]]);let s=((bits>>15)&1)as u32;let e=((bits>>10)&0x1F)as u32;let f=(bits&0x3FF)as u32;if e==0{if f==0{f32::from_bits(s<<31)}else{let v=f as f32/1024.0*2.0f32.powi(-14);if s==1{-v}else{v}}}else if e==31{if f==0{if s==1{f32::NEG_INFINITY}else{f32::INFINITY}}else{f32::NAN}}else{f32::from_bits((s<<31)|((e+127-15)<<23)|(f<<13))}}).collect())}30=>{let mut buf=vec![0u8;n*2];r.read_exact(&mut buf).map_err(|e|e.to_string())?;Ok(buf.chunks_exact(2).map(|c|f32::from_bits((u16::from_le_bytes([c[0],c[1]])as u32)<<16)).collect())}_=>Err(format!("unsupported dtype {}",t.dtype)),}}
