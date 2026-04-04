//! Build real token → centroid codebook assignment for BGE-M3.
//!
//! For each of 250,002 token embeddings, finds the nearest weight row
//! in blk.23.attn_q.weight (1024 rows) by cosine similarity.
//! Saves as codebook_index.u16 (250,002 × 2 bytes = ~500 KB).
//!
//! Requires BGE-M3 F16 GGUF in /tmp/hf_cache (already cached).
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
//!   --example build_codebook_index

use std::io::{Read, Seek, SeekFrom};
use rayon::prelude::*;
use ndarray::hpc::heel_f64x8::cosine_f32_to_f64_simd;
use thinking_engine::codebook_index::CodebookIndex;

const VOCAB_SIZE: usize = 250_002;
const HIDDEN_DIM: usize = 1024;
const TABLE_ROWS: usize = 1024; // blk.23.attn_q.weight rows

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  BUILD CODEBOOK INDEX: BGE-M3 token → centroid");
    println!("═══════════════════════════════════════════════════════\n");

    // ── Step 1: Find and open BGE-M3 GGUF ─────────────────────────────────
    println!("[1] Locating BGE-M3 F16 GGUF in /tmp/hf_cache ...");
    let gguf_path = find_bge_m3_gguf()
        .expect("BGE-M3 F16 GGUF not found in /tmp/hf_cache");
    println!("  Found: {}", gguf_path);

    let mut file = std::fs::File::open(&gguf_path)
        .expect("Failed to open GGUF file");
    let header = parse_gguf_header(&mut file)
        .expect("Failed to parse GGUF header");
    println!("  Tensors: {}", header.tensors.len());

    // ── Step 2: Read blk.23.attn_q.weight (1024 × 1024, F16) ─────────────
    println!("\n[2] Reading blk.23.attn_q.weight ({} × {}) ...", TABLE_ROWS, HIDDEN_DIM);
    let attn_q = header.tensors.iter()
        .find(|t| t.name.contains("blk.23") && t.name.contains("attn_q") && t.name.contains("weight"))
        .expect("blk.23.attn_q.weight not found in GGUF");
    println!("  Tensor: {} (dtype={}, dims={:?})", attn_q.name, attn_q.dtype, attn_q.dims);
    assert!(
        attn_q.dtype == 1 || attn_q.dtype == 30,
        "Expected F16 (1) or BF16 (30), got dtype={}",
        attn_q.dtype
    );

    let attn_q_data = read_tensor_f32(&mut file, &header, attn_q)
        .expect("Failed to read attn_q tensor");
    let attn_q_rows = attn_q.dims[0] as usize;
    let attn_q_cols: usize = attn_q.dims[1..].iter().map(|&d| d as usize).product();
    println!("  Shape: {} rows × {} cols, {} floats total",
        attn_q_rows, attn_q_cols, attn_q_data.len());
    assert_eq!(attn_q_rows, TABLE_ROWS, "Expected {} rows", TABLE_ROWS);

    // ── Step 3: Read token_embd.weight (250002 × 1024, F16) ──────────────
    println!("\n[3] Reading token_embd.weight ({} × {}) ...", VOCAB_SIZE, HIDDEN_DIM);
    let embd = header.tensors.iter()
        .find(|t| t.name.contains("token_embd") && t.name.contains("weight"))
        .expect("token_embd.weight not found in GGUF");
    println!("  Tensor: {} (dtype={}, dims={:?})", embd.name, embd.dtype, embd.dims);

    let embd_data = read_tensor_f32(&mut file, &header, embd)
        .expect("Failed to read token_embd tensor");

    // GGUF stores embedding as [hidden_dim, vocab_size] — need to handle both layouts
    let (embd_rows, embd_cols) = if embd.dims[0] as usize == VOCAB_SIZE {
        (embd.dims[0] as usize, embd.dims[1] as usize)
    } else if embd.dims.len() >= 2 && embd.dims[1] as usize == VOCAB_SIZE {
        // Transposed: [hidden_dim, vocab_size] → treat as [vocab_size, hidden_dim]
        (embd.dims[1] as usize, embd.dims[0] as usize)
    } else if embd.dims[0] as usize == HIDDEN_DIM {
        // [1024, 250002] layout — vocab is dim[1]
        (embd.dims[1] as usize, embd.dims[0] as usize)
    } else {
        panic!("Unexpected embedding dims: {:?}", embd.dims);
    };
    println!("  Logical shape: {} tokens × {} hidden_dim, {} floats total",
        embd_rows, embd_cols, embd_data.len());
    assert_eq!(embd_rows, VOCAB_SIZE, "Expected {} tokens", VOCAB_SIZE);
    assert_eq!(embd_cols, HIDDEN_DIM, "Expected {} hidden_dim", HIDDEN_DIM);

    // If transposed [hidden_dim, vocab_size], transpose in-memory to [vocab_size, hidden_dim]
    // This is O(N) once and makes the parallel search cache-friendly
    let is_transposed = embd.dims[0] as usize == HIDDEN_DIM;
    let embd_data = if is_transposed {
        println!("  Transposing embedding in-memory ({} × {} → {} × {}) ...",
            HIDDEN_DIM, VOCAB_SIZE, VOCAB_SIZE, HIDDEN_DIM);
        let start = std::time::Instant::now();
        let mut transposed = vec![0.0f32; VOCAB_SIZE * HIDDEN_DIM];
        for d in 0..HIDDEN_DIM {
            for t in 0..VOCAB_SIZE {
                transposed[t * HIDDEN_DIM + d] = embd_data[d * VOCAB_SIZE + t];
            }
        }
        println!("  Transposed in {:.1}s", start.elapsed().as_secs_f64());
        transposed
    } else {
        embd_data
    };

    // ── Step 4: Pre-normalize centroid rows (attn_q) ──────────────────────
    println!("\n[4] Pre-normalizing {} centroid rows ...", TABLE_ROWS);
    let start = std::time::Instant::now();
    let centroids: Vec<Vec<f32>> = (0..TABLE_ROWS).map(|r| {
        let row = &attn_q_data[r * attn_q_cols..(r + 1) * attn_q_cols];
        let norm = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
        if norm < 1e-12 {
            vec![0.0f32; attn_q_cols]
        } else {
            let inv = (1.0 / norm) as f32;
            row.iter().map(|v| v * inv).collect()
        }
    }).collect();
    println!("  Done in {:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    // ── Step 5: For each token embedding, find nearest centroid ────────────
    println!("\n[5] Finding nearest centroid for each of {} tokens (chunked, 2 threads) ...", VOCAB_SIZE);
    let start = std::time::Instant::now();

    // Process in chunks to show progress and avoid process kill
    let chunk_size = 25000;
    let mut indices = vec![0u16; embd_rows];

    for chunk_start in (0..embd_rows).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(embd_rows);
        let chunk_indices: Vec<u16> = (chunk_start..chunk_end)
            .into_par_iter()
            .map(|tok_id| {
                let tok_row = &embd_data[tok_id * embd_cols..(tok_id + 1) * embd_cols];

                // Pre-normalize
                let norm = tok_row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();
                if norm < 1e-12 { return 0u16; }
                let inv = (1.0 / norm) as f32;
                let tok_normed: Vec<f32> = tok_row.iter().map(|v| v * inv).collect();

                // Find nearest centroid by f32 dot product (both normalized → dot = cosine)
                let mut best_idx = 0u16;
                let mut best_dot = f32::NEG_INFINITY;
                for (c_idx, centroid) in centroids.iter().enumerate() {
                    let dot: f32 = tok_normed.iter().zip(centroid.iter())
                        .map(|(a, b)| a * b).sum();
                    if dot > best_dot {
                        best_dot = dot;
                        best_idx = c_idx as u16;
                    }
                }
                best_idx
            })
            .collect();

        for (i, &idx) in chunk_indices.iter().enumerate() {
            indices[chunk_start + i] = idx;
        }
        let elapsed = start.elapsed().as_secs_f64();
        let pct = (chunk_end as f64 / embd_rows as f64) * 100.0;
        println!("  [{:>6}/{:>6}] {:.0}%  {:.1}s", chunk_end, embd_rows, pct, elapsed);
    }

    let elapsed = start.elapsed();
    println!("  Done in {:.2}s ({:.0} tokens/sec)",
        elapsed.as_secs_f64(),
        VOCAB_SIZE as f64 / elapsed.as_secs_f64());

    // ── Step 6: Build CodebookIndex and save ──────────────────────────────
    println!("\n[6] Building CodebookIndex and saving ...");
    let codebook = CodebookIndex::new(indices, TABLE_ROWS as u16, "bge-m3".into());

    let out_dir = "/tmp/codebooks/bge-m3-roles-f16";
    let out_path = std::path::Path::new(out_dir).join("codebook_index.u16");
    codebook.save(&out_path).expect("Failed to save codebook index");

    let file_size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    println!("  Saved: {} ({} bytes, {:.1} KB)",
        out_path.display(), file_size, file_size as f64 / 1024.0);

    // ── Step 7: Print stats ───────────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════");
    println!("  CODEBOOK INDEX STATS");
    println!("═══════════════════════════════════════════════════════\n");

    let unique = codebook.unique_centroids();
    println!("  Vocab size:       {}", codebook.len());
    println!("  Table rows:       {}", codebook.table_size());
    println!("  Unique centroids: {} / {} ({:.1}%)",
        unique, TABLE_ROWS, unique as f64 / TABLE_ROWS as f64 * 100.0);

    let counts = codebook.centroid_counts();
    let max_count = *counts.iter().max().unwrap_or(&0);
    let min_count = *counts.iter().min().unwrap_or(&0);
    let empty_centroids = counts.iter().filter(|&&c| c == 0).count();
    let mean_count = codebook.len() as f64 / TABLE_ROWS as f64;

    println!("  Tokens/centroid:  min={}, max={}, mean={:.1}",
        min_count, max_count, mean_count);
    println!("  Empty centroids:  {} ({:.1}%)",
        empty_centroids, empty_centroids as f64 / TABLE_ROWS as f64 * 100.0);

    // Histogram: bucket sizes
    let mut histogram = std::collections::BTreeMap::new();
    for &c in &counts {
        let bucket = match c {
            0 => "      0",
            1..=10 => "   1-10",
            11..=50 => "  11-50",
            51..=100 => " 51-100",
            101..=500 => "101-500",
            501..=1000 => "501-1K",
            _ => "  1K+",
        };
        *histogram.entry(bucket).or_insert(0u32) += 1;
    }
    println!("\n  Distribution of centroid cluster sizes:");
    for (bucket, count) in &histogram {
        let bar_len = (*count as usize).min(60);
        let bar: String = std::iter::repeat('#').take(bar_len).collect();
        println!("    {} tokens: {:>4} centroids  {}", bucket, count, bar);
    }

    // Top-10 most popular centroids
    let mut indexed_counts: Vec<(usize, u32)> = counts.iter().enumerate()
        .map(|(i, &c)| (i, c))
        .collect();
    indexed_counts.sort_by(|a, b| b.1.cmp(&a.1));
    println!("\n  Top-10 most popular centroids:");
    for (i, (centroid, count)) in indexed_counts.iter().take(10).enumerate() {
        println!("    #{}: centroid {:>4} → {} tokens", i + 1, centroid, count);
    }

    // Bottom-10 (least used, excluding empty)
    let mut nonempty: Vec<(usize, u32)> = indexed_counts.iter()
        .filter(|(_, c)| *c > 0)
        .copied()
        .collect();
    nonempty.sort_by(|a, b| a.1.cmp(&b.1));
    println!("\n  Bottom-10 least popular (non-empty) centroids:");
    for (i, (centroid, count)) in nonempty.iter().take(10).enumerate() {
        println!("    #{}: centroid {:>4} → {} tokens", i + 1, centroid, count);
    }

    println!("\n═══════════════════════════════════════════════════════");
    println!("  DONE: {} → {}", gguf_path, out_path.display());
    println!("═══════════════════════════════════════════════════════");
}

// ── GGUF Parsing (reused from build_1to1_roles.rs) ────────────────────────

fn find_bge_m3_gguf() -> Option<String> {
    let base = "/tmp/hf_cache";
    let entries = std::fs::read_dir(base).ok()?;
    for entry in entries.flatten() {
        let p = entry.path();
        if !p.is_dir() { continue; }
        let name = p.file_name()?.to_string_lossy().to_string();
        if !name.contains("models--") { continue; }
        // Look for bge-m3 (case insensitive)
        let lower = name.to_lowercase();
        if !lower.contains("bge") || !lower.contains("m3") { continue; }

        let snap = p.join("snapshots");
        if let Ok(snaps) = std::fs::read_dir(&snap) {
            for s in snaps.flatten() {
                if let Ok(files) = std::fs::read_dir(s.path()) {
                    for f in files.flatten() {
                        let fp = f.path();
                        let fname = fp.to_string_lossy().to_string();
                        if !fname.ends_with(".gguf") { continue; }
                        // Skip quantized — we need F16
                        if fname.contains("Q8_0") || fname.contains("Q4_K")
                            || fname.contains("Q2_K") || fname.contains("Q5_K") {
                            continue;
                        }
                        let real = std::fs::read_link(&fp)
                            .map(|r| if r.is_relative() { fp.parent().unwrap().join(r) } else { r })
                            .unwrap_or(fp.clone());
                        if real.exists() && std::fs::metadata(&real).map(|m| m.len() > 1000).unwrap_or(false) {
                            return Some(real.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

struct GgufHeader {
    tensors: Vec<TensorMeta>,
    data_offset: u64,
}

struct TensorMeta {
    name: String,
    dims: Vec<u64>,
    dtype: u32,
    offset: u64,
    #[allow(dead_code)]
    n_elements: u64,
}

fn parse_gguf_header<R: Read + Seek>(r: &mut R) -> Result<GgufHeader, String> {
    let mut b4 = [0u8; 4];
    let mut b8 = [0u8; 8];
    r.read_exact(&mut b4).map_err(|e| e.to_string())?;
    if u32::from_le_bytes(b4) != 0x46554747 {
        return Err("bad magic".into());
    }
    r.read_exact(&mut b4).map_err(|e| e.to_string())?; // version
    r.read_exact(&mut b8).map_err(|e| e.to_string())?;
    let nt = u64::from_le_bytes(b8) as usize;
    r.read_exact(&mut b8).map_err(|e| e.to_string())?;
    let nm = u64::from_le_bytes(b8) as usize;
    for _ in 0..nm {
        skip_kv(r)?;
    }
    let mut tensors = Vec::with_capacity(nt);
    for _ in 0..nt {
        r.read_exact(&mut b8).map_err(|e| e.to_string())?;
        let nl = u64::from_le_bytes(b8) as usize;
        let mut nb = vec![0u8; nl];
        r.read_exact(&mut nb).map_err(|e| e.to_string())?;
        let name = String::from_utf8_lossy(&nb).to_string();
        r.read_exact(&mut b4).map_err(|e| e.to_string())?;
        let nd = u32::from_le_bytes(b4) as usize;
        let mut dims = Vec::with_capacity(nd);
        for _ in 0..nd {
            r.read_exact(&mut b8).map_err(|e| e.to_string())?;
            dims.push(u64::from_le_bytes(b8));
        }
        r.read_exact(&mut b4).map_err(|e| e.to_string())?;
        let dtype = u32::from_le_bytes(b4);
        r.read_exact(&mut b8).map_err(|e| e.to_string())?;
        let offset = u64::from_le_bytes(b8);
        let n_elements: u64 = dims.iter().product();
        tensors.push(TensorMeta { name, dims, dtype, offset, n_elements });
    }
    let pos = r.stream_position().map_err(|e| e.to_string())?;
    Ok(GgufHeader { tensors, data_offset: (pos + 31) / 32 * 32 })
}

fn skip_kv<R: Read + Seek>(r: &mut R) -> Result<(), String> {
    let mut b4 = [0u8; 4];
    let mut b8 = [0u8; 8];
    r.read_exact(&mut b8).map_err(|e| e.to_string())?;
    let kl = u64::from_le_bytes(b8) as usize;
    let mut kb = vec![0u8; kl];
    r.read_exact(&mut kb).map_err(|e| e.to_string())?;
    r.read_exact(&mut b4).map_err(|e| e.to_string())?;
    skip_val(r, u32::from_le_bytes(b4))
}

fn skip_val<R: Read + Seek>(r: &mut R, vt: u32) -> Result<(), String> {
    let mut b4 = [0u8; 4];
    let mut b8 = [0u8; 8];
    match vt {
        0 | 1 | 7 => {
            r.read_exact(&mut [0u8; 1]).map_err(|e| e.to_string())?;
        }
        2 | 3 => {
            r.read_exact(&mut [0u8; 2]).map_err(|e| e.to_string())?;
        }
        4 | 5 | 6 => {
            r.read_exact(&mut b4).map_err(|e| e.to_string())?;
        }
        8 => {
            r.read_exact(&mut b8).map_err(|e| e.to_string())?;
            let l = u64::from_le_bytes(b8) as usize;
            let mut s = vec![0u8; l];
            r.read_exact(&mut s).map_err(|e| e.to_string())?;
        }
        9 => {
            r.read_exact(&mut b4).map_err(|e| e.to_string())?;
            let et = u32::from_le_bytes(b4);
            r.read_exact(&mut b8).map_err(|e| e.to_string())?;
            let c = u64::from_le_bytes(b8) as usize;
            for _ in 0..c {
                skip_val(r, et)?;
            }
        }
        10 | 11 | 12 => {
            r.read_exact(&mut b8).map_err(|e| e.to_string())?;
        }
        _ => return Err(format!("unknown vtype {}", vt)),
    }
    Ok(())
}

fn read_tensor_f32<R: Read + Seek>(r: &mut R, h: &GgufHeader, t: &TensorMeta) -> Result<Vec<f32>, String> {
    r.seek(SeekFrom::Start(h.data_offset + t.offset)).map_err(|e| e.to_string())?;
    let n = t.dims.iter().product::<u64>() as usize;
    match t.dtype {
        0 => {
            // F32
            let mut buf = vec![0u8; n * 4];
            r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        }
        1 => {
            // F16
            let mut buf = vec![0u8; n * 2];
            r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf.chunks_exact(2).map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                let s = ((bits >> 15) & 1) as u32;
                let e = ((bits >> 10) & 0x1F) as u32;
                let f = (bits & 0x3FF) as u32;
                if e == 0 {
                    if f == 0 { f32::from_bits(s << 31) }
                    else {
                        let v = f as f32 / 1024.0 * 2.0f32.powi(-14);
                        if s == 1 { -v } else { v }
                    }
                } else if e == 31 {
                    if f == 0 { if s == 1 { f32::NEG_INFINITY } else { f32::INFINITY } }
                    else { f32::NAN }
                } else {
                    f32::from_bits((s << 31) | ((e + 127 - 15) << 23) | (f << 13))
                }
            }).collect())
        }
        30 => {
            // BF16
            let mut buf = vec![0u8; n * 2];
            r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf.chunks_exact(2).map(|c| {
                f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)
            }).collect())
        }
        _ => Err(format!("unsupported dtype {} — only F16(1)/BF16(30)/F32(0)", t.dtype)),
    }
}
