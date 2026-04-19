//! Fractal probe — cheap gating test for orthogonal fractal decomposition.
//!
//! Streams 100 k_proj + 100 gate_proj BF16 rows from a Qwen3 GGUF on
//! HuggingFace, Hadamard-rotates each, computes MFDFA fractal descriptor,
//! and reports CoV(w_mfs) per tensor. If CoV > 0.3, fractal leaf has signal.
//!
//! Usage:
//!   cargo run --release --example fractal_probe
//!
//! Uses ndarray::hpc::{http_reader, gguf_indexer, fft} + bgz_tensor::fractal_descriptor.
//! No f32 Vec for the full tensor — reads row batches, converts BF16→f32 inline.

use ndarray::hpc::fft::wht_f32;
use ndarray::hpc::http_reader::HttpRangeReader;

use bgz_tensor::fractal_descriptor::{compute_mfdfa_descriptor, FractalDescriptor};

use std::io::{Read, Seek, SeekFrom};

const PROBE_ROWS: usize = 100;
const QWEN3_REPO: &str = "Qwen/Qwen3-8B";
const QWEN3_GGUF: &str = "qwen3-8b-bf16.gguf";

fn bf16_to_f32_scalar(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn bf16_row_to_f32(bf16: &[u16], out: &mut [f32]) {
    debug_assert_eq!(bf16.len(), out.len());
    let n = bf16.len();
    let mut i = 0;
    // Process 16 elements at a time for autovectorization.
    while i + 16 <= n {
        for j in 0..16 {
            out[i + j] = bf16_to_f32_scalar(bf16[i + j]);
        }
        i += 16;
    }
    while i < n {
        out[i] = bf16_to_f32_scalar(bf16[i]);
        i += 1;
    }
}

fn next_pow2(n: usize) -> usize {
    let mut v = n;
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v + 1
}

/// Parse the GGUF header to find tensor name, offset, and dimensions.
/// Returns (tensor_data_offset, Vec<(name, offset, [rows, cols])>).
fn find_gguf_tensors<R: Read + Seek>(
    reader: &mut R,
) -> Result<(u64, Vec<(String, u64, usize, usize)>), String> {
    // Use ndarray's GGUF header parser.
    let header = ndarray::hpc::gguf::read_gguf_header(reader)
        .map_err(|e| format!("GGUF parse: {e}"))?;

    let mut tensors = Vec::new();
    for t in &header.tensors {
        let ndim = t.dimensions.len();
        if ndim < 2 {
            continue;
        }
        let n_rows = t.dimensions[0] as usize;
        let n_cols: usize = t.dimensions[1..].iter().map(|&d| d as usize).product();
        tensors.push((t.name.clone(), t.offset, n_rows, n_cols));
    }
    Ok((header.tensor_data_offset, tensors))
}

fn probe_tensor<R: Read + Seek>(
    reader: &mut R,
    data_offset: u64,
    tensor_offset: u64,
    n_rows: usize,
    n_cols: usize,
    max_rows: usize,
) -> Vec<FractalDescriptor> {
    let rows_to_read = max_rows.min(n_rows);
    let abs_offset = data_offset + tensor_offset;
    reader
        .seek(SeekFrom::Start(abs_offset))
        .expect("seek to tensor");

    // Pad to power-of-2 for WHT.
    let padded = next_pow2(n_cols);

    // Reusable buffers on the stack/heap once.
    let mut bf16_buf = vec![0u16; n_cols];
    let mut f32_buf = vec![0.0f32; padded];
    let mut descriptors = Vec::with_capacity(rows_to_read);

    let byte_buf = unsafe {
        std::slice::from_raw_parts_mut(bf16_buf.as_mut_ptr() as *mut u8, n_cols * 2)
    };

    for _ in 0..rows_to_read {
        reader.read_exact(byte_buf).expect("read BF16 row");

        // BF16 → f32 (autovectorizable).
        bf16_row_to_f32(&bf16_buf, &mut f32_buf[..n_cols]);
        // Zero-pad remainder for WHT.
        for j in n_cols..padded {
            f32_buf[j] = 0.0;
        }

        // Hadamard rotate in-place (existing SIMD butterfly).
        wht_f32(&mut f32_buf);

        // MFDFA on rotated coefficients.
        let desc = compute_mfdfa_descriptor(&f32_buf);
        descriptors.push(desc);
    }
    descriptors
}

fn compute_cov(values: &[f32]) -> f32 {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let std = var.sqrt();
    if mean.abs() < 1e-10 {
        0.0
    } else {
        std / mean.abs()
    }
}

fn report(name: &str, descriptors: &[FractalDescriptor]) {
    let w_values: Vec<f32> = descriptors.iter().map(|d| d.w_mfs_f32()).collect();
    let h_values: Vec<f32> = descriptors.iter().map(|d| d.h_hurst_f32()).collect();
    let d_values: Vec<f32> = descriptors.iter().map(|d| d.d_local_f32()).collect();

    let cov_w = compute_cov(&w_values);
    let cov_h = compute_cov(&h_values);
    let cov_d = compute_cov(&d_values);

    let mean_w = w_values.iter().sum::<f32>() / w_values.len() as f32;
    let mean_h = h_values.iter().sum::<f32>() / h_values.len() as f32;
    let mean_d = d_values.iter().sum::<f32>() / d_values.len() as f32;

    eprintln!("  {name}:");
    eprintln!("    w_mfs   mean={mean_w:.4}  CoV={cov_w:.4}  {}",
        if cov_w > 0.3 { "✓ SIGNAL" } else { "✗ flat" });
    eprintln!("    H_hurst mean={mean_h:.4}  CoV={cov_h:.4}");
    eprintln!("    D_local mean={mean_d:.4}  CoV={cov_d:.4}");
}

fn main() {
    eprintln!("Fractal probe — orthogonal MFDFA on Hadamard-rotated weight rows");
    eprintln!("Connecting to HuggingFace: {QWEN3_REPO}/{QWEN3_GGUF}");

    let mut reader = HttpRangeReader::from_hf(QWEN3_REPO, QWEN3_GGUF, 64 * 1024 * 1024)
        .expect("failed to connect to HF");

    eprintln!("File size: {:.1} GB", reader.total_size() as f64 / 1e9);
    eprintln!("Parsing GGUF header...");

    let (data_offset, tensors) = find_gguf_tensors(&mut reader)
        .expect("failed to parse GGUF");

    eprintln!("Found {} tensors", tensors.len());

    // Find first k_proj and first gate_proj.
    let k_proj = tensors.iter().find(|(name, _, _, _)| name.contains("k_proj"));
    let gate_proj = tensors.iter().find(|(name, _, _, _)| name.contains("gate_proj"));

    if let Some((name, offset, n_rows, n_cols)) = k_proj {
        eprintln!("\nProbing {name} ({n_rows} rows × {n_cols} cols, {PROBE_ROWS} sampled)...");
        let descs = probe_tensor(&mut reader, data_offset, *offset, *n_rows, *n_cols, PROBE_ROWS);
        report("k_proj L0", &descs);
    } else {
        eprintln!("WARNING: no k_proj tensor found");
    }

    if let Some((name, offset, n_rows, n_cols)) = gate_proj {
        eprintln!("\nProbing {name} ({n_rows} rows × {n_cols} cols, {PROBE_ROWS} sampled)...");
        let descs = probe_tensor(&mut reader, data_offset, *offset, *n_rows, *n_cols, PROBE_ROWS);
        report("gate_proj L0", &descs);
    } else {
        eprintln!("WARNING: no gate_proj tensor found");
    }

    eprintln!("\nGate: CoV(w_mfs) > 0.3 → fractal leaf has signal.");
    eprintln!("Downloaded {:.1} MB from HF", reader.bytes_downloaded() as f64 / 1e6);
}
