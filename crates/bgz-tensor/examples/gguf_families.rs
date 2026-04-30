//! GGUF → StackedN → CLAM Family Buckets pipeline.
//!
//! Reads a GGUF model file, dequantizes to f32, encodes as StackedN at
//! multiple SPD resolutions, builds CLAM cosine family buckets, and
//! measures compression ratio + cosine fidelity.
//!
//! Family buckets: HEEL centroid stored once per family, members hold index.
//! Many weight rows are similar within the same role+layer → massive reuse.
//!
//! Usage: cargo run --manifest-path crates/bgz-tensor/Cargo.toml --example gguf_families -- /path/to/model.gguf

use bgz_tensor::stacked_n::{bf16_to_f32, cosine_f32_slice, f32_to_bf16, ClamCodebook, StackedN};
use bgz_tensor::variance::Role;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        // Default: Jina v3 Q8_0
        "/tmp/hf_cache/models--gaianet--jina-embeddings-v3-GGUF/snapshots/d7d998aab1a7f2ea0aa256d8b3f035cbd0af682a/jina-embeddings-v3-Q8_0.gguf".into()
    });

    println!("=== GGUF → StackedN → CLAM Family Buckets ===\n");
    println!("Model: {}\n", path);

    let mut file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Cannot open {}: {}", path, e);
            return;
        }
    };

    // ─── Parse GGUF header ──────────────────────────────────────────────
    let header = match parse_gguf_header(&mut file) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("GGUF parse error: {}", e);
            return;
        }
    };

    println!("Tensors: {}", header.tensors.len());
    println!("Data offset: {}\n", header.data_offset);

    // ─── Process tensors: dequant → StackedN ────────────────────────────
    let spd_values = [4, 16, 32, 64];
    let mut role_vectors: HashMap<Role, Vec<Vec<f32>>> = HashMap::new();
    let mut total_rows = 0usize;
    let mut total_params = 0usize;
    let mut tensors_processed = 0;

    for tensor in &header.tensors {
        let role = match Role::from_name(&tensor.name) {
            Some(r) => r,
            None => continue, // skip non-Q/K/V/Gate/Up/Down tensors
        };

        if tensor.n_elements < 1024 {
            continue;
        } // skip tiny

        // Read and dequantize to f32
        let f32_data = match read_tensor_f32(&mut file, &header, tensor) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  Skip {}: {}", tensor.name, e);
                continue;
            }
        };

        // Reshape into rows
        let (n_rows, n_cols) = if tensor.dims.len() >= 2 {
            (
                tensor.dims[0] as usize,
                tensor.dims[1..].iter().map(|&d| d as usize).product(),
            )
        } else {
            (1, f32_data.len())
        };

        // Collect row vectors for this role
        let entry = role_vectors.entry(role).or_default();
        for r in 0..n_rows {
            let start = r * n_cols;
            let end = (start + n_cols).min(f32_data.len());
            if end > start {
                entry.push(f32_data[start..end].to_vec());
            }
        }

        total_rows += n_rows;
        total_params += f32_data.len();
        tensors_processed += 1;

        if tensors_processed % 20 == 0 {
            eprint!(
                "\rProcessed {} tensors, {} rows...",
                tensors_processed, total_rows
            );
        }
    }
    eprintln!(
        "\rProcessed {} tensors, {} rows, {} params total",
        tensors_processed, total_rows, total_params
    );

    // ─── Per-role summary ───────────────────────────────────────────────
    println!("\n=== Per-Role Row Counts ===");
    for role in Role::ALL {
        let count = role_vectors.get(&role).map_or(0, |v| v.len());
        println!("  {:<5}: {} rows", role.label(), count);
    }

    // ─── Build StackedN at each SPD, then family buckets ────────────────
    println!("\n=== Family Bucket Compression ===\n");

    for &spd in &spd_values {
        println!("--- SPD={} ({} bytes/vector) ---", spd, 17 * spd * 2);

        let mut total_raw_bytes = 0usize;
        let mut total_family_bytes = 0usize;
        let mut total_vectors = 0usize;
        let mut total_families = 0usize;
        let mut cosine_errors: Vec<f64> = Vec::new();

        for role in Role::ALL {
            let rows = match role_vectors.get(&role) {
                Some(r) if !r.is_empty() => r,
                _ => continue,
            };

            // Limit to 2000 rows per role for speed
            let limit = rows.len().min(2000);
            let rows = &rows[..limit];

            // Encode as StackedN
            let encoded: Vec<StackedN> = rows.iter().map(|v| StackedN::from_f32(v, spd)).collect();

            let bytes_per_vec = 17 * spd * 2;
            let raw_bytes = encoded.len() * bytes_per_vec;

            // Build CLAM cosine family buckets
            // Target: ~32-64 families per role (5-6 bit index)
            let n_families = (encoded.len() / 20).clamp(4, 64);
            let cb = ClamCodebook::build_cosine(&encoded, n_families);

            // Family bucket storage:
            //   - n_families HEELs at full resolution
            //   - n_vectors assignments (2 bytes each)
            let family_bytes = cb.byte_size() + encoded.len() * 2;

            // Measure cosine fidelity: sample pairs, compare ground truth
            let n_sample = encoded.len().min(200);
            for i in 0..n_sample {
                for j in (i + 1)..n_sample.min(i + 10) {
                    // Ground truth: f32 cosine
                    let gt = cosine_f32_slice(&rows[i], &rows[j]);
                    // Family bucket: cosine between HEEL centroids
                    let ai = cb.assignments[i];
                    let aj = cb.assignments[j];
                    let family_cos = if ai == aj {
                        // Same family → use stacked cosine as proxy
                        encoded[i].cosine(&encoded[j])
                    } else {
                        let ea = cb.get(ai);
                        let eb = cb.get(aj);
                        match (ea, eb) {
                            (Some(a), Some(b)) => a.stacked.cosine(&b.stacked),
                            _ => 0.0,
                        }
                    };
                    cosine_errors.push((gt - family_cos).abs());
                }
            }

            let ratio = raw_bytes as f64 / family_bytes.max(1) as f64;
            println!(
                "  {:<5}: {} vecs → {} families, raw={:.0}KB family={:.0}KB ratio={:.1}×",
                role.label(),
                encoded.len(),
                cb.entries.len(),
                raw_bytes as f64 / 1024.0,
                family_bytes as f64 / 1024.0,
                ratio
            );

            total_raw_bytes += raw_bytes;
            total_family_bytes += family_bytes;
            total_vectors += encoded.len();
            total_families += cb.entries.len();
        }

        let total_ratio = total_raw_bytes as f64 / total_family_bytes.max(1) as f64;
        let mean_err = if cosine_errors.is_empty() {
            0.0
        } else {
            cosine_errors.iter().sum::<f64>() / cosine_errors.len() as f64
        };
        let max_err = cosine_errors.iter().cloned().fold(0.0f64, f64::max);

        println!(
            "  TOTAL: {} vecs, {} families",
            total_vectors, total_families
        );
        println!(
            "  Raw: {:.0} KB, Family: {:.0} KB, Ratio: {:.1}×",
            total_raw_bytes as f64 / 1024.0,
            total_family_bytes as f64 / 1024.0,
            total_ratio
        );
        println!("  Cosine error: mean={:.4}, max={:.4}\n", mean_err, max_err);
    }

    // ─── Leaf hydration test: pick a few vectors, encode→hydrate→compare ─
    println!("=== Leaf Hydration: BF16→f32 Fidelity ===\n");
    if let Some(q_rows) = role_vectors.get(&Role::Q) {
        let sample = q_rows.len().min(50);
        for &spd in &[32, 64] {
            let mut errors = Vec::new();
            for i in 0..sample {
                for j in (i + 1)..sample.min(i + 5) {
                    let gt = cosine_f32_slice(&q_rows[i], &q_rows[j]);
                    let a = StackedN::from_f32(&q_rows[i], spd);
                    let b = StackedN::from_f32(&q_rows[j], spd);
                    let ha = a.hydrate_f32();
                    let hb = b.hydrate_f32();
                    let leaf = cosine_f32_slice(&ha, &hb);
                    errors.push((gt - leaf).abs());
                }
            }
            let mean = errors.iter().sum::<f64>() / errors.len().max(1) as f64;
            let max = errors.iter().cloned().fold(0.0f64, f64::max);
            println!(
                "  SPD={}: leaf↔gt mean_err={:.6}, max_err={:.6} ({} pairs)",
                spd,
                mean,
                max,
                errors.len()
            );
        }
    }

    println!("\n=== PIPELINE COMPLETE ===");
}

// ═══════════════════════════════════════════════════════════════════════════
// Minimal GGUF reader (self-contained, no ndarray dependency needed)
// ═══════════════════════════════════════════════════════════════════════════

struct GgufHeader {
    tensors: Vec<TensorMeta>,
    data_offset: u64,
}

struct TensorMeta {
    name: String,
    dims: Vec<u64>,
    dtype: u32,
    offset: u64,
    n_elements: u64,
}

fn parse_gguf_header<R: Read + Seek>(r: &mut R) -> Result<GgufHeader, String> {
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    // Magic
    r.read_exact(&mut buf4).map_err(|e| e.to_string())?;
    let magic = u32::from_le_bytes(buf4);
    if magic != 0x46554747 {
        return Err(format!("bad magic: {:#x}", magic));
    }

    // Version
    r.read_exact(&mut buf4).map_err(|e| e.to_string())?;
    let version = u32::from_le_bytes(buf4);
    if version < 2 || version > 3 {
        return Err(format!("unsupported version: {}", version));
    }

    // Tensor count
    r.read_exact(&mut buf8).map_err(|e| e.to_string())?;
    let n_tensors = u64::from_le_bytes(buf8) as usize;

    // Metadata count
    r.read_exact(&mut buf8).map_err(|e| e.to_string())?;
    let n_metadata = u64::from_le_bytes(buf8) as usize;

    // Skip metadata key-value pairs
    for _ in 0..n_metadata {
        skip_gguf_kv(r, version)?;
    }

    // Read tensor infos
    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        // Name
        r.read_exact(&mut buf8).map_err(|e| e.to_string())?;
        let name_len = u64::from_le_bytes(buf8) as usize;
        let mut name_buf = vec![0u8; name_len];
        r.read_exact(&mut name_buf).map_err(|e| e.to_string())?;
        let name = String::from_utf8_lossy(&name_buf).to_string();

        // n_dims
        r.read_exact(&mut buf4).map_err(|e| e.to_string())?;
        let n_dims = u32::from_le_bytes(buf4) as usize;

        // dims
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            r.read_exact(&mut buf8).map_err(|e| e.to_string())?;
            dims.push(u64::from_le_bytes(buf8));
        }

        // dtype
        r.read_exact(&mut buf4).map_err(|e| e.to_string())?;
        let dtype = u32::from_le_bytes(buf4);

        // offset
        r.read_exact(&mut buf8).map_err(|e| e.to_string())?;
        let offset = u64::from_le_bytes(buf8);

        let n_elements: u64 = dims.iter().product();
        tensors.push(TensorMeta {
            name,
            dims,
            dtype,
            offset,
            n_elements,
        });
    }

    // Data starts at next alignment boundary (default: 32 bytes)
    let pos = r.stream_position().map_err(|e| e.to_string())?;
    let align = 32u64;
    let data_offset = (pos + align - 1) / align * align;

    Ok(GgufHeader {
        tensors,
        data_offset,
    })
}

fn skip_gguf_kv<R: Read + Seek>(r: &mut R, _version: u32) -> Result<(), String> {
    let mut buf8 = [0u8; 8];
    let mut buf4 = [0u8; 4];

    // Key (string)
    r.read_exact(&mut buf8).map_err(|e| e.to_string())?;
    let key_len = u64::from_le_bytes(buf8) as usize;
    let mut key_buf = vec![0u8; key_len];
    r.read_exact(&mut key_buf).map_err(|e| e.to_string())?;

    // Value type
    r.read_exact(&mut buf4).map_err(|e| e.to_string())?;
    let vtype = u32::from_le_bytes(buf4);

    skip_gguf_value(r, vtype)?;
    Ok(())
}

fn skip_gguf_value<R: Read + Seek>(r: &mut R, vtype: u32) -> Result<(), String> {
    let mut buf8 = [0u8; 8];
    let mut buf4 = [0u8; 4];
    match vtype {
        0 | 1 | 7 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
        } // uint8/int8/bool
        2 | 3 => {
            r.read_exact(&mut [0u8; 2]).map_err(|e| e.to_string())?;
        } // uint16/int16
        4 | 5 | 6 => {
            r.read_exact(&mut buf4).map_err(|e| e.to_string())?;
        } // uint32/int32/float32
        8 => {
            // string
            r.read_exact(&mut buf8).map_err(|e| e.to_string())?;
            let len = u64::from_le_bytes(buf8) as usize;
            let mut s = vec![0u8; len];
            r.read_exact(&mut s).map_err(|e| e.to_string())?;
        }
        9 => {
            // array
            r.read_exact(&mut buf4).map_err(|e| e.to_string())?;
            let elem_type = u32::from_le_bytes(buf4);
            r.read_exact(&mut buf8).map_err(|e| e.to_string())?;
            let count = u64::from_le_bytes(buf8) as usize;
            for _ in 0..count {
                skip_gguf_value(r, elem_type)?;
            }
        }
        10 | 11 | 12 => {
            r.read_exact(&mut buf8).map_err(|e| e.to_string())?;
        } // uint64/int64/float64
        _ => return Err(format!("unknown GGUF value type: {}", vtype)),
    }
    Ok(())
}

fn read_tensor_f32<R: Read + Seek>(
    r: &mut R,
    header: &GgufHeader,
    tensor: &TensorMeta,
) -> Result<Vec<f32>, String> {
    let abs_offset = header.data_offset + tensor.offset;
    r.seek(SeekFrom::Start(abs_offset))
        .map_err(|e| e.to_string())?;

    let n = tensor.n_elements as usize;

    match tensor.dtype {
        0 => {
            // F32
            let mut buf = vec![0u8; n * 4];
            r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        1 => {
            // F16
            let mut buf = vec![0u8; n * 2];
            r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f16_to_f32(bits)
                })
                .collect())
        }
        8 => {
            // Q8_0: blocks of 32 int8 values + f16 scale
            let block_size = 32;
            let n_blocks = (n + block_size - 1) / block_size;
            let bytes_per_block = 2 + 32; // f16 scale + 32 int8 values
            let mut buf = vec![0u8; n_blocks * bytes_per_block];
            r.read_exact(&mut buf).map_err(|e| e.to_string())?;

            let mut result = Vec::with_capacity(n);
            for block in 0..n_blocks {
                let off = block * bytes_per_block;
                let scale_bits = u16::from_le_bytes([buf[off], buf[off + 1]]);
                let scale = f16_to_f32(scale_bits);
                for i in 0..block_size {
                    if result.len() >= n {
                        break;
                    }
                    let val = buf[off + 2 + i] as i8;
                    result.push(val as f32 * scale);
                }
            }
            Ok(result)
        }
        30 => {
            // BF16
            let mut buf = vec![0u8; n * 2];
            r.read_exact(&mut buf).map_err(|e| e.to_string())?;
            Ok(buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    bf16_to_f32(bits)
                })
                .collect())
        }
        _ => Err(format!(
            "unsupported dtype {} for {}",
            tensor.dtype, tensor.name
        )),
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            let f = frac as f32 / 1024.0 * 2.0f32.powi(-14);
            if sign == 1 {
                -f
            } else {
                f
            }
        }
    } else if exp == 31 {
        if frac == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
    }
}
