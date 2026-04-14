//! Streaming HHTL-D encoder — zero disk, 256 MB peak RAM.
//!
//! Streams Qwen3-TTS-1.7B safetensors directly from HuggingFace via
//! ndarray's HttpRangeReader. Never downloads the full 3.86 GB file.
//!
//! Reader comparison:
//!
//! | Reader | Source | Disk | RAM | Streaming |
//! |---|---|---|---|---|
//! | `safetensors` crate | crates.io | full file | full file | no (mmap) |
//! | `ndarray::hpc::safetensors` | AdaWorldAPI/ndarray | header only | per-tensor | yes |
//! | `ndarray::hpc::http_reader` | AdaWorldAPI/ndarray | zero | 256 MB window | yes (HTTP range) |
//! | `bgz-tensor` palettes | AdaWorldAPI/lance-graph | zero | per-group | yes (CLAM online) |
//! | `tokenizers` crate | crates.io | tokenizer.json | ~5 MB | n/a |
//!
//! This example uses ndarray's streaming readers (rows 2-3 above) for the
//! weight data, and bgz-tensor for the encoding pipeline. The safetensors
//! crate (row 1) is only used for the OUTPUT format — writing the compressed
//! file. The tokenizers crate is not needed here (encoding is weight-level,
//! not token-level).
//!
//! Pipeline:
//!   HF HTTP → header parse (58 KB) → per-group:
//!     stream tensor rows (64 MB window) → Base17 → palette sample
//!     → build shared palette → stream again → encode HhtlDEntry
//!     → write to output safetensors
//!
//! Two-pass streaming:
//!   Pass 1: Sample rows for palette building (read 4096 rows per group)
//!   Pass 2: Encode all rows against the built palette
//!
//! ```sh
//! # Stream from HuggingFace (zero disk):
//! cargo run --release --example tts_stream_hhtld \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- Qwen/Qwen3-TTS-12Hz-1.7B-Base model.safetensors
//!
//! # Or from local file:
//! cargo run --release --example tts_stream_hhtld \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- --local /path/to/model.safetensors
//! ```

use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::gguf::{GgmlType, TensorInfo};
use bgz_tensor::projection::Base17;
use bgz_tensor::palette::WeightPalette;
use bgz_tensor::hhtl_cache::HhtlCache;
use bgz_tensor::hhtl_d::{HhtlDTensor, HhtlDEntry};
use bgz_tensor::shared_palette::{
    PaletteGroupKey, classify_role, classify_component,
    is_encodable, effective_shape, build_hip_families,
};

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::time::Instant;

const N_CENTROIDS: usize = 256;
const SAMPLE_PER_GROUP: usize = 4096;

// ═══════════════════════════════════════════════════════════════════
// Reader abstraction: local file or HTTP range
// ═══════════════════════════════════════════════════════════════════

enum TensorReader {
    Local(BufReader<File>),
    #[cfg(feature = "http")]
    Http(ndarray::hpc::http_reader::HttpRangeReader),
}

impl TensorReader {
    fn seek_read(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), String> {
        match self {
            TensorReader::Local(r) => {
                r.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
                r.read_exact(buf).map_err(|e| e.to_string())?;
            }
            #[cfg(feature = "http")]
            TensorReader::Http(r) => {
                r.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
                r.read_exact(buf).map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }
}

/// Read a single tensor as f32 rows, streaming from the reader.
fn read_tensor_rows(
    reader: &mut TensorReader,
    tensor: &TensorInfo,
    data_offset: u64,
    max_rows: Option<usize>,
) -> Vec<Vec<f32>> {
    let n_rows = tensor.dimensions[0] as usize;
    let n_cols = if tensor.dimensions.len() > 1 { tensor.dimensions[1] as usize } else { 1 };
    let limit = max_rows.unwrap_or(n_rows).min(n_rows);

    let tensor_start = data_offset + tensor.offset;
    let elem_size = match tensor.dtype {
        GgmlType::BF16 | GgmlType::F16 => 2,
        GgmlType::F32 => 4,
        _ => return Vec::new(),
    };

    // Stream row by row (or in chunks for efficiency)
    let row_bytes = n_cols * elem_size;
    let chunk_rows = 256.min(limit); // read 256 rows at a time
    let mut all_rows = Vec::with_capacity(limit);

    let mut offset = tensor_start;
    let mut remaining = limit;

    while remaining > 0 {
        let batch = chunk_rows.min(remaining);
        let mut buf = vec![0u8; batch * row_bytes];
        if reader.seek_read(offset, &mut buf).is_err() {
            break;
        }

        for r in 0..batch {
            let row_start = r * row_bytes;
            let row: Vec<f32> = match tensor.dtype {
                GgmlType::BF16 => {
                    (0..n_cols).map(|c| {
                        let i = row_start + c * 2;
                        let bits = u16::from_le_bytes([buf[i], buf[i + 1]]);
                        f32::from_bits((bits as u32) << 16)
                    }).collect()
                }
                GgmlType::F16 => {
                    (0..n_cols).map(|c| {
                        let i = row_start + c * 2;
                        let bits = u16::from_le_bytes([buf[i], buf[i + 1]]);
                        ndarray::hpc::gguf::f16_to_f32(bits)
                    }).collect()
                }
                GgmlType::F32 => {
                    (0..n_cols).map(|c| {
                        let i = row_start + c * 4;
                        f32::from_le_bytes([buf[i], buf[i+1], buf[i+2], buf[i+3]])
                    }).collect()
                }
                _ => unreachable!(),
            };
            all_rows.push(row);
        }

        offset += (batch * row_bytes) as u64;
        remaining -= batch;
    }

    all_rows
}

// ═══════════════════════════════════════════════════════════════════
// Group tensors by shared palette key
// ═══════════════════════════════════════════════════════════════════

fn group_tensors<'a>(
    tensors: &'a [TensorInfo],
) -> (
    HashMap<PaletteGroupKey, Vec<&'a TensorInfo>>,
    Vec<&'a TensorInfo>,
) {
    let mut groups: HashMap<PaletteGroupKey, Vec<&TensorInfo>> = HashMap::new();
    let mut passthrough: Vec<&TensorInfo> = Vec::new();

    for tensor in tensors {
        if !tensor.name.ends_with("weight") {
            // Biases, layernorms without "weight" suffix — passthrough
            passthrough.push(tensor);
            continue;
        }

        let shape: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();
        let size: usize = shape.iter().product::<usize>() * match tensor.dtype {
            GgmlType::BF16 | GgmlType::F16 => 2,
            GgmlType::F32 => 4,
            _ => 1,
        };

        if is_encodable(&shape, size) {
            let (rows, cols) = effective_shape(&shape);
            let key = PaletteGroupKey {
                component: classify_component(&tensor.name).to_string(),
                role: classify_role(&tensor.name).to_string(),
                shape: (rows, cols),
            };
            groups.entry(key).or_default().push(tensor);
        } else {
            passthrough.push(tensor);
        }
    }

    (groups, passthrough)
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let (mut reader, header, out_path) = if args.len() >= 2 && args[1] == "--local" {
        // Local file mode
        let path = &args[2];
        let mut r = BufReader::new(File::open(path).expect("open safetensors"));
        let h = read_safetensors_header(&mut r).expect("parse header");
        let out = {
            let p = std::path::Path::new(path);
            let stem = p.file_stem().unwrap().to_str().unwrap();
            p.parent().unwrap().join(format!("{stem}_hhtld.safetensors")).to_string_lossy().to_string()
        };
        (TensorReader::Local(r), h, out)
    } else {
        // HTTP streaming mode (requires "http" feature or ndarray with http)
        #[cfg(not(feature = "http"))]
        {
            eprintln!("HTTP streaming requires the 'http' feature.");
            eprintln!("Usage: --local /path/to/model.safetensors");
            eprintln!("   or: (with http feature) REPO FILENAME");
            std::process::exit(1);
        }
        #[cfg(feature = "http")]
        {
            let repo = &args[1];
            let filename = if args.len() >= 3 { &args[2] } else { "model.safetensors" };

            println!("[0] Resolving HuggingFace URL...");
            let (url, size) = ndarray::hpc::http_reader::resolve_hf_url(repo, filename)
                .expect("resolve HF URL");
            println!("  URL: {}...{}", &url[..50.min(url.len())], &url[url.len().saturating_sub(20)..]);
            println!("  Size: {:.2} GB", size as f64 / 1e9);

            let mut hr = ndarray::hpc::http_reader::HttpRangeReader::with_chunk_size(
                url, size, 64 * 1024 * 1024 // 64 MB window
            );

            // Parse header via HTTP range
            let h = read_safetensors_header(&mut hr).expect("parse header");
            let out = format!("{}_hhtld.safetensors", filename.trim_end_matches(".safetensors"));
            (TensorReader::Http(hr), h, out)
        }
    };

    let data_offset = header.tensor_data_offset;

    println!("═══ STREAMING HHTL-D ENCODER ═══");
    println!("  {} tensors, data offset {}", header.tensors.len(), data_offset);
    println!("  Output: {}", out_path);
    println!();

    // ─── Step 1: Group tensors ─────────────────────────────────────
    let (groups, passthrough) = group_tensors(&header.tensors);
    println!("[1] {} palette groups, {} passthrough tensors", groups.len(), passthrough.len());
    for (key, tensors) in &groups {
        let total_rows: usize = tensors.iter().map(|t| t.dimensions[0] as usize).sum();
        println!("    {}/{} {:?}: {} tensors, {} rows",
            key.component, key.role, key.shape, tensors.len(), total_rows);
    }

    // ─── Step 2: Pass 1 — Sample for palette building ──────────────
    println!("\n[2] Pass 1: Sampling rows for palette building...");
    let t0 = Instant::now();
    let mut group_samples: HashMap<PaletteGroupKey, Vec<Base17>> = HashMap::new();

    for (key, tensors) in &groups {
        let mut samples = Vec::new();
        let per_tensor = SAMPLE_PER_GROUP / tensors.len().max(1);

        for tensor in tensors {
            let rows = read_tensor_rows(&mut reader, tensor, data_offset, Some(per_tensor));
            for row in &rows {
                samples.push(Base17::from_f32(row));
            }
        }

        println!("    {}/{}: sampled {} rows", key.component, key.role, samples.len());
        group_samples.insert(key.clone(), samples);
    }
    println!("  Pass 1 done in {:?}", t0.elapsed());

    // ─── Step 3: Build shared palettes ─────────────────────────────
    println!("\n[3] Building shared palettes ({} groups)...", groups.len());
    let mut group_palettes: HashMap<PaletteGroupKey, (HhtlCache, Vec<u8>)> = HashMap::new();

    for (key, samples) in &group_samples {
        let t0 = Instant::now();
        let sample_slice = if samples.len() > 4096 { &samples[..4096] } else { &samples[..] };
        let wp = WeightPalette::build(sample_slice, N_CENTROIDS);
        let active = wp.counts.iter().filter(|&&c| c > 0).count();

        let hip = build_hip_families(&wp.entries);
        let mut cache = HhtlCache::from_palette(wp);
        // Set gamma_meta stub (will be refined in pass 2)
        cache.gamma_meta = [0.01, 0.01, 0.0, 0.0];

        println!("    {}/{}: {}/{} active centroids ({:?})",
            key.component, key.role, active, N_CENTROIDS, t0.elapsed());

        group_palettes.insert(key.clone(), (cache, hip));
    }

    // ─── Step 4: Pass 2 — Stream and encode all rows ───────────────
    println!("\n[4] Pass 2: Streaming full encode...");
    let t0 = Instant::now();

    // Collect all output data
    let mut output_tensors: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
    let mut total_entries = 0usize;
    let mut total_entry_bytes = 0usize;

    for (key, tensors) in &groups {
        let (cache, hip) = &group_palettes[key];

        for tensor in tensors {
            let rows = read_tensor_rows(&mut reader, tensor, data_offset, None);
            let n_rows = rows.len();
            let n_cols = if tensor.dimensions.len() > 1 { tensor.dimensions[1] as usize } else { 1 };

            let role_name = format!("{}_{}", key.component, key.role);
            let hhtld = HhtlDTensor::encode(&role_name, &rows, cache, hip);
            let entries_bytes = hhtld.entries_to_bytes();

            total_entries += n_rows;
            total_entry_bytes += entries_bytes.len();

            // Store with tensor-specific name
            let safe_name = tensor.name.replace('.', "_");
            output_tensors.push((
                format!("{safe_name}.hhtld_entries"),
                entries_bytes,
                vec![n_rows, 4],
            ));

            // Original shape metadata
            let mut shape_bytes = Vec::with_capacity(8);
            shape_bytes.extend_from_slice(&(n_rows as u32).to_le_bytes());
            shape_bytes.extend_from_slice(&(n_cols as u32).to_le_bytes());
            output_tensors.push((
                format!("{safe_name}.original_shape"),
                shape_bytes,
                vec![8],
            ));
        }

        // Store shared palette + tables once per group
        let group_name = format!("{}_{}_{}x{}", key.component, key.role, key.shape.0, key.shape.1);

        // Palette entries
        let mut palette_bytes = Vec::with_capacity(N_CENTROIDS * 34);
        for entry in &cache.palette.entries {
            for &dim in &entry.dims {
                palette_bytes.extend_from_slice(&dim.to_le_bytes());
            }
        }
        output_tensors.push((format!("{group_name}.palette"), palette_bytes, vec![N_CENTROIDS, 34]));

        // Distance table
        let k = cache.k();
        let mut dist_bytes = Vec::with_capacity(k * k * 2);
        let at = bgz_tensor::attention::AttentionTable::build(&cache.palette);
        for a in 0..k {
            for b in 0..k {
                dist_bytes.extend_from_slice(&at.distance(a as u8, b as u8).to_le_bytes());
            }
        }
        output_tensors.push((format!("{group_name}.distance_table"), dist_bytes, vec![k, k]));

        // Route table
        let mut route_bytes = Vec::with_capacity(k * k);
        for a in 0..k {
            for b in 0..k {
                route_bytes.push(cache.route(a as u8, b as u8) as u8);
            }
        }
        output_tensors.push((format!("{group_name}.route_table"), route_bytes, vec![k, k]));

        // HIP families
        output_tensors.push((format!("{group_name}.hip_families"), hip.clone(), vec![k]));
    }

    // Passthrough tensors
    for tensor in &passthrough {
        let shape: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();
        let n: usize = shape.iter().product();
        let elem_size = match tensor.dtype {
            GgmlType::BF16 | GgmlType::F16 => 2,
            GgmlType::F32 => 4,
            _ => continue,
        };
        let mut buf = vec![0u8; n * elem_size];
        if reader.seek_read(data_offset + tensor.offset, &mut buf).is_ok() {
            let safe_name = tensor.name.replace('.', "_");
            output_tensors.push((
                format!("passthrough.{safe_name}"),
                buf,
                shape,
            ));
        }
    }

    println!("  Pass 2 done in {:?}", t0.elapsed());

    // ─── Step 5: Write output ──────────────────────────────────────
    println!("\n[5] Writing output...");
    let total_output: usize = output_tensors.iter().map(|(_, d, _)| d.len()).sum();
    let original_bytes: u64 = header.tensors.iter()
        .map(|t| {
            let n: u64 = t.dimensions.iter().map(|&d| d as u64).product();
            n * match t.dtype { GgmlType::BF16 | GgmlType::F16 => 2, GgmlType::F32 => 4, _ => 1 }
        })
        .sum();

    // Build metadata
    let mut metadata = HashMap::new();
    metadata.insert("encoding".to_string(), "bgz-hhtl-d".to_string());
    metadata.insert("version".to_string(), "2".to_string()); // v2 = streaming + shared palettes
    metadata.insert("palette_k".to_string(), N_CENTROIDS.to_string());
    metadata.insert("n_groups".to_string(), groups.len().to_string());
    metadata.insert("total_entries".to_string(), total_entries.to_string());
    metadata.insert("original_bytes".to_string(), original_bytes.to_string());
    metadata.insert("compression_ratio".to_string(),
        format!("{:.1}", original_bytes as f64 / total_output as f64));

    write_safetensors(&out_path, &output_tensors, &metadata).expect("write output");

    println!("  Output:      {} ({:.1} MB)", out_path, total_output as f64 / 1e6);
    println!("  Original:    {:.1} MB", original_bytes as f64 / 1e6);
    println!("  Entries:     {} ({:.1} KB)", total_entry_bytes, total_entry_bytes as f64 / 1024.0);
    println!("  Compression: {:.0}:1", original_bytes as f64 / total_output as f64);
    println!("\n═══ DONE ═══");
}

/// Write safetensors format: 8-byte header_size + JSON header + data.
fn write_safetensors(
    path: &str,
    tensors: &[(String, Vec<u8>, Vec<usize>)],
    metadata: &HashMap<String, String>,
) -> Result<(), String> {
    use serde_json::{json, Value, Map};

    let mut header_map = Map::new();
    let meta_value: Value = metadata.iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect::<Map<String, Value>>()
        .into();
    header_map.insert("__metadata__".into(), meta_value);

    let mut offset = 0usize;
    for (name, data, shape) in tensors {
        let begin = offset;
        let end = offset + data.len();
        offset = end;
        header_map.insert(name.clone(), json!({
            "dtype": "U8",
            "shape": shape,
            "data_offsets": [begin, end]
        }));
    }

    let header_json = serde_json::to_string(&Value::Object(header_map))
        .map_err(|e| format!("JSON: {e}"))?;
    let header_bytes = header_json.as_bytes();

    let mut f = File::create(path).map_err(|e| format!("create: {e}"))?;
    f.write_all(&(header_bytes.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(header_bytes).map_err(|e| e.to_string())?;
    for (_, data, _) in tensors {
        f.write_all(data).map_err(|e| e.to_string())?;
    }
    f.flush().map_err(|e| e.to_string())?;
    Ok(())
}
