//! Build CLAM codebooks from GGUF model files for the ThinkingEngine.
//!
//! For each GGUF model:
//!   1. Parse GGUF header, index tensors
//!   2. Dequantize Q8_0 rows to f32
//!   3. Furthest-point sampling: select 64 centroids
//!   4. Assign each row to nearest centroid (u16 index)
//!   5. Compute 64x64 distance table (cosine -> u8)
//!   6. Save codebook + distance table + indices to /tmp/
//!
//! Run: cargo run --release --manifest-path crates/thinking-engine/Cargo.toml --example build_codebooks

use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;
use std::path::Path;

const N_CENTROIDS: usize = 64;

fn main() {
    let models: Vec<(&str, &str)> = vec![
        ("jina-v3", "/tmp/hf_cache/models--gaianet--jina-embeddings-v3-GGUF/snapshots/d7d998aab1a7f2ea0aa256d8b3f035cbd0af682a/jina-embeddings-v3-Q8_0.gguf"),
        ("bge-m3", ""),      // filled in dynamically
        ("reader-lm-1.5b", ""), // filled in dynamically
        ("gpt2", ""),          // filled in dynamically
    ];

    // Discover downloaded GGUFs
    let mut paths: Vec<(String, String)> = Vec::new();

    // 1. Jina v3 (known path)
    let jina_path = models[0].1;
    if Path::new(jina_path).exists() {
        paths.push(("jina-v3".into(), jina_path.into()));
    } else {
        eprintln!("[SKIP] Jina v3: not found at {}", jina_path);
    }

    // 2-4. Search /tmp/hf_cache for other GGUFs
    for (tag, pattern) in &[
        ("bge-m3", "bge-m3"),
        ("reader-lm-1.5b", "reader-lm"),
        ("gpt2", "pt-2"),  // matches GPT-2 or gpt2
    ] {
        if let Some(p) = find_gguf_in_cache(pattern) {
            paths.push((tag.to_string(), p));
        } else {
            eprintln!("[SKIP] {}: no GGUF found in /tmp/hf_cache matching '{}'", tag, pattern);
        }
    }

    if paths.is_empty() {
        eprintln!("No GGUF models found. Nothing to do.");
        return;
    }

    println!("=== CLAM Codebook Builder ({} centroids) ===\n", N_CENTROIDS);

    for (name, path) in &paths {
        println!("────────────────────────────────────────────────────────");
        println!("Model: {}  ({})", name, path);
        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        println!("File size: {:.1} MB", file_size as f64 / 1e6);
        println!("────────────────────────────────────────────────────────");

        let t0 = Instant::now();
        match process_model(name, path) {
            Ok(()) => println!("  Total time: {:.2}s\n", t0.elapsed().as_secs_f64()),
            Err(e) => {
                eprintln!("  ERROR processing {}: {}\n", name, e);
                continue;
            }
        }
    }

    println!("=== Done ===");
}

fn find_gguf_in_cache(pattern: &str) -> Option<String> {
    let cache = Path::new("/tmp/hf_cache");
    if !cache.exists() { return None; }
    find_gguf_recursive(cache, pattern)
}

fn find_gguf_recursive(dir: &Path, pattern: &str) -> Option<String> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_gguf_recursive(&path, pattern) {
                return Some(found);
            }
        } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            let name_lower = name.to_lowercase();
            let pattern_lower = pattern.to_lowercase();
            if name_lower.contains(&pattern_lower)
                && name_lower.contains("q8_0")
                && name_lower.ends_with(".gguf")
            {
                return Some(path.to_string_lossy().into());
            }
        }
    }
    None
}

fn process_model(name: &str, path: &str) -> Result<(), String> {
    // Step 1: Parse GGUF header
    let t1 = Instant::now();
    let index = GgufIndex::open(path)?;
    println!("  [1] GGUF index: {} tensors, {:.2}s",
        index.tensors.len(), t1.elapsed().as_secs_f64());

    // Filter to 2D weight tensors with reasonable dimensions
    let weight_tensors: Vec<&TensorLocation> = index.tensors.iter()
        .filter(|t| t.n_rows() >= 4 && t.n_cols >= 32 && t.dtype == 8) // Q8_0 only
        .collect();

    if weight_tensors.is_empty() {
        return Err("no Q8_0 2D tensors found".into());
    }

    println!("  Q8_0 weight tensors: {} (of {} total)",
        weight_tensors.len(), index.tensors.len());

    // Step 2: Collect all rows from largest tensor (for codebook)
    // Pick the largest tensor by row count for representative sampling
    let target = weight_tensors.iter()
        .max_by_key(|t| t.n_rows() * t.n_cols)
        .unwrap();

    println!("  Target tensor: {} ({}x{}, {} elements)",
        target.name, target.n_rows(), target.n_cols, target.n_elements);

    let t2 = Instant::now();
    let n_rows = target.n_rows();
    let n_cols = target.n_cols;

    // Read all rows
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n_rows);
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    for r in 0..n_rows {
        let row = hydrate_q8_0_row(&mut file, target, r)?;
        rows.push(row);
    }
    println!("  [2] Dequantized {} rows x {} cols, {:.2}s",
        n_rows, n_cols, t2.elapsed().as_secs_f64());

    // Step 3: CLAM furthest-point sampling -> 64 centroids
    let t3 = Instant::now();
    let centroid_indices = furthest_point_sampling(&rows, N_CENTROIDS);
    let centroids: Vec<Vec<f32>> = centroid_indices.iter()
        .map(|&i| rows[i].clone())
        .collect();
    println!("  [3] FPS selected {} centroids, {:.2}s",
        centroids.len(), t3.elapsed().as_secs_f64());

    // Step 4: Assign each row to nearest centroid
    let t4 = Instant::now();
    let assignments: Vec<u16> = rows.iter()
        .map(|row| {
            let mut best_idx = 0u16;
            let mut best_sim = f32::NEG_INFINITY;
            for (c, centroid) in centroids.iter().enumerate() {
                let sim = cosine_f32(row, centroid);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = c as u16;
                }
            }
            best_idx
        })
        .collect();
    println!("  [4] Assigned {} rows to centroids, {:.2}s",
        assignments.len(), t4.elapsed().as_secs_f64());

    // Verify assignment distribution
    let mut counts = vec![0u32; N_CENTROIDS];
    for &a in &assignments {
        counts[a as usize] += 1;
    }
    let min_count = counts.iter().copied().min().unwrap_or(0);
    let max_count = counts.iter().copied().max().unwrap_or(0);
    let nonempty = counts.iter().filter(|&&c| c > 0).count();
    println!("  Assignment stats: {}/{} centroids used, min={}, max={} rows/centroid",
        nonempty, N_CENTROIDS, min_count, max_count);

    // Step 5: Build 64x64 distance table (cosine -> u8)
    let t5 = Instant::now();
    let k = centroids.len();
    let mut distance_table = vec![128u8; k * k];
    for i in 0..k {
        distance_table[i * k + i] = 255; // self = max similarity
        for j in (i + 1)..k {
            let cos = cosine_f32(&centroids[i], &centroids[j]);
            let u = ((cos + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
            distance_table[i * k + j] = u;
            distance_table[j * k + i] = u;
        }
    }
    println!("  [5] Built {}x{} distance table, {:.2}s",
        k, k, t5.elapsed().as_secs_f64());

    // Distance table stats
    let dt_min = *distance_table.iter().min().unwrap_or(&0);
    let dt_max = *distance_table.iter().max().unwrap_or(&0);
    let dt_mean: f64 = distance_table.iter().map(|&v| v as f64).sum::<f64>()
        / distance_table.len() as f64;
    println!("  Distance table stats: min={}, max={}, mean={:.1}", dt_min, dt_max, dt_mean);

    // Step 6: Save everything
    let t6 = Instant::now();
    let out_dir = format!("/tmp/codebooks/{}", name);
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;

    // Save centroids as flat f32 binary
    let centroids_path = format!("{}/centroids_{}x{}.f32", out_dir, k, n_cols);
    let centroids_flat: Vec<u8> = centroids.iter()
        .flat_map(|c| c.iter().flat_map(|v| v.to_le_bytes()))
        .collect();
    std::fs::write(&centroids_path, &centroids_flat).map_err(|e| e.to_string())?;

    // Save distance table
    let table_path = format!("{}/distance_table_{}x{}.u8", out_dir, k, k);
    std::fs::write(&table_path, &distance_table).map_err(|e| e.to_string())?;

    // Save assignments
    let assign_path = format!("{}/assignments_{}.u16", out_dir, n_rows);
    let assign_bytes: Vec<u8> = assignments.iter()
        .flat_map(|&a| a.to_le_bytes())
        .collect();
    std::fs::write(&assign_path, &assign_bytes).map_err(|e| e.to_string())?;

    println!("  [6] Saved to {}, {:.2}s", out_dir, t6.elapsed().as_secs_f64());
    println!("    Centroids: {} ({} bytes)", centroids_path, centroids_flat.len());
    println!("    Table:     {} ({} bytes)", table_path, distance_table.len());
    println!("    Indices:   {} ({} bytes)", assign_path, assign_bytes.len());

    // Summary
    let total_saved = centroids_flat.len() + distance_table.len() + assign_bytes.len();
    println!("  SUMMARY: {} tensors, {} rows, codebook={} bytes, table={} bytes, total={} bytes",
        weight_tensors.len(), n_rows, centroids_flat.len(),
        distance_table.len(), total_saved);

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// CLAM Furthest-Point Sampling
// ═══════════════════════════════════════════════════════════════════════════

fn furthest_point_sampling(rows: &[Vec<f32>], k: usize) -> Vec<usize> {
    let n = rows.len();
    if n <= k {
        return (0..n).collect();
    }

    let mut selected = Vec::with_capacity(k);
    let mut min_dist = vec![f32::INFINITY; n]; // min distance to any selected point

    // Start with row 0
    selected.push(0);

    // Update distances from first selected point
    for i in 0..n {
        let d = 1.0 - cosine_f32(&rows[0], &rows[i]); // cosine distance
        if d < min_dist[i] { min_dist[i] = d; }
    }

    for _ in 1..k {
        // Find the point farthest from all selected points
        let mut best_idx = 0;
        let mut best_dist = f32::NEG_INFINITY;
        for i in 0..n {
            if min_dist[i] > best_dist {
                best_dist = min_dist[i];
                best_idx = i;
            }
        }

        selected.push(best_idx);
        min_dist[best_idx] = 0.0; // mark as selected

        // Update min distances
        for i in 0..n {
            let d = 1.0 - cosine_f32(&rows[best_idx], &rows[i]);
            if d < min_dist[i] { min_dist[i] = d; }
        }
    }

    selected
}

// ═══════════════════════════════════════════════════════════════════════════
// GGUF Reader (reused from highheelbgz/src/source.rs pattern)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
struct TensorLocation {
    name: String,
    data_offset: u64,
    dtype: u32,
    dims: Vec<u64>,
    n_elements: u64,
    n_cols: usize,
}

impl TensorLocation {
    fn n_rows(&self) -> usize {
        if self.dims.len() >= 2 { self.dims[0] as usize } else { 1 }
    }
}

struct GgufIndex {
    tensors: Vec<TensorLocation>,
}

impl GgufIndex {
    fn open(path: &str) -> Result<Self, String> {
        let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {}", path, e))?;
        let header = parse_gguf_header(&mut file)?;

        let tensors: Vec<TensorLocation> = header.tensors.into_iter().map(|t| {
            let n_cols = if t.dims.len() >= 2 {
                t.dims[1..].iter().map(|&d| d as usize).product()
            } else { t.n_elements as usize };

            TensorLocation {
                name: t.name,
                data_offset: header.data_offset + t.offset,
                dtype: t.dtype,
                dims: t.dims,
                n_elements: t.n_elements,
                n_cols,
            }
        }).collect();

        Ok(GgufIndex { tensors })
    }
}

fn hydrate_q8_0_row<R: Read + Seek>(
    file: &mut R,
    tensor: &TensorLocation,
    row_idx: usize,
) -> Result<Vec<f32>, String> {
    let n_cols = tensor.n_cols;
    let blocks_per_row = (n_cols + 31) / 32;
    let bytes_per_block = 34; // 2 (f16 scale) + 32 (int8 values)
    let row_offset = tensor.data_offset + (row_idx * blocks_per_row * bytes_per_block) as u64;
    file.seek(SeekFrom::Start(row_offset)).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; blocks_per_row * bytes_per_block];
    file.read_exact(&mut buf).map_err(|e| e.to_string())?;

    let mut result = Vec::with_capacity(n_cols);
    for b in 0..blocks_per_row {
        let o = b * bytes_per_block;
        let scale_bits = u16::from_le_bytes([buf[o], buf[o + 1]]);
        let scale = f16_to_f32(scale_bits);
        for i in 0..32 {
            if result.len() >= n_cols { break; }
            result.push(buf[o + 2 + i] as i8 as f32 * scale);
        }
    }
    Ok(result)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 { f32::from_bits(sign << 31) }
        else {
            let f = frac as f32 / 1024.0 * 2.0f32.powi(-14);
            if sign == 1 { -f } else { f }
        }
    } else if exp == 31 {
        if frac == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else { f32::NAN }
    } else {
        f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GGUF header parser
// ═══════════════════════════════════════════════════════════════════════════

struct ParsedHeader { tensors: Vec<ParsedTensor>, data_offset: u64 }
struct ParsedTensor { name: String, dims: Vec<u64>, dtype: u32, offset: u64, n_elements: u64 }

fn parse_gguf_header<R: Read + Seek>(r: &mut R) -> Result<ParsedHeader, String> {
    let mut b4 = [0u8; 4];
    let mut b8 = [0u8; 8];

    r.read_exact(&mut b4).map_err(|e| e.to_string())?;
    if u32::from_le_bytes(b4) != 0x46554747 {
        return Err("bad GGUF magic".into());
    }

    r.read_exact(&mut b4).map_err(|e| e.to_string())?; // version
    r.read_exact(&mut b8).map_err(|e| e.to_string())?;
    let nt = u64::from_le_bytes(b8) as usize;
    r.read_exact(&mut b8).map_err(|e| e.to_string())?;
    let nm = u64::from_le_bytes(b8) as usize;

    for _ in 0..nm { skip_kv(r)?; }

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

        tensors.push(ParsedTensor {
            name,
            dims: dims.clone(),
            dtype,
            offset,
            n_elements: dims.iter().product(),
        });
    }

    let pos = r.stream_position().map_err(|e| e.to_string())?;
    Ok(ParsedHeader { tensors, data_offset: (pos + 31) / 32 * 32 })
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
        0 | 1 | 7 => { let mut b = [0u8; 1]; r.read_exact(&mut b).map_err(|e| e.to_string())?; }
        2 | 3 => { r.read_exact(&mut [0u8; 2]).map_err(|e| e.to_string())?; }
        4 | 5 | 6 => { r.read_exact(&mut b4).map_err(|e| e.to_string())?; }
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
            for _ in 0..c { skip_val(r, et)?; }
        }
        10 | 11 | 12 => { r.read_exact(&mut b8).map_err(|e| e.to_string())?; }
        _ => return Err(format!("unknown GGUF vtype {}", vt)),
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Cosine similarity
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-12 { 0.0 } else { (dot / denom) as f32 }
}
