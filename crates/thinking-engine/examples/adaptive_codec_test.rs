//! adaptive_codec_test — CLAM-adaptive precision + argmax validation.
//!
//! Tests the CHAODA-driven codec on the tensors that failed uniform k=64.

use bgz_tensor::adaptive_codec::AdaptiveCodecTensor;
use bgz_tensor::xor_adaptive::XorAdaptiveTensor;
use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::heel_f64x8::cosine_f32_to_f64_simd;
use ndarray::simd::bf16_to_f32_batch;

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

const N_SAMPLE: usize = 512;

fn load_tensor(path: &str, substr: &str) -> Option<(Vec<Vec<f32>>, String, usize, usize)> {
    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let header = read_safetensors_header(&mut reader).ok()?;
    let t = header.tensors.iter().find(|t| t.name.contains(substr))?;
    let n_rows = t.dimensions[0] as usize;
    let n_cols: usize = t.dimensions.iter().skip(1).map(|&d| d as usize).product();
    let sample = N_SAMPLE.min(n_rows);
    let stride = n_rows.max(1) / sample;
    reader.seek(SeekFrom::Start(header.tensor_data_offset + t.offset)).ok()?;
    let mut raw = vec![0u8; n_rows * n_cols * 2];
    reader.read_exact(&mut raw).ok()?;
    let u16s: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let mut f32_data = vec![0.0f32; u16s.len()];
    bf16_to_f32_batch(&u16s, &mut f32_data);
    let rows: Vec<Vec<f32>> = (0..sample)
        .map(|i| {
            let ri = (i * stride).min(n_rows - 1);
            f32_data[ri * n_cols..(ri + 1) * n_cols].to_vec()
        }).collect();
    Some((rows, t.name.clone(), n_rows, n_cols))
}

fn matmul_row(x: &[f32], weight_rows: &[Vec<f32>]) -> Vec<f32> {
    weight_rows.iter().map(|w| {
        x.iter().zip(w.iter()).map(|(a, b)| a * b).sum()
    }).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
}

fn main() {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "/home/user/models/qwen3-tts-0.6b/model.safetensors".into());

    println!("# CLAM-Adaptive Codec — CHAODA-driven Argmax Test");
    println!("Model: `{}`", path);
    println!();

    let test_cases = vec![
        ("speaker_encoder.fc.weight", false, "speaker_encoder.fc (HARD — was 25% at k=64)"),
        ("mlp.down_proj.weight", false, "MLP down_proj (was 69% at k=64)"),
        ("self_attn.k_proj.weight", true, "Attention k_proj (KV sensitive)"),
        ("self_attn.q_proj.weight", false, "Attention q_proj"),
        ("mlp.gate_proj.weight", false, "MLP gate_proj"),
    ];

    println!("| Tensor | CLAM-adaptive | XOR-adaptive |");
    println!("|---|---|---|");

    let n_test = 32;

    for (substr, is_kv, label) in &test_cases {
        let Some((rows, name, full_n, n_cols)) = load_tensor(&path, substr) else {
            println!("| {} | — | — | — | — | — | — | — |", label);
            continue;
        };
        let n = rows.len();

        let t0 = Instant::now();
        let tensor = AdaptiveCodecTensor::encode(&name, &rows, 64, *is_kv, None);
        let encode_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let recon = tensor.reconstruct_all();

        // Also test XOR-adaptive
        let t1 = Instant::now();
        let xor_tensor = XorAdaptiveTensor::encode(&name, &rows, 64);
        let xor_ms = t1.elapsed().as_secs_f32() * 1000.0;
        let xor_recon = xor_tensor.reconstruct_all();

        let mut match_count = 0usize;
        let mut xor_match = 0usize;
        let mut cos_sum = 0.0f64;
        let mut xor_cos = 0.0f64;
        for t in 0..n_test {
            let x: Vec<f32> = (0..n_cols).map(|d| {
                ((d * 97 + t * 31 + 17) as f64 * 0.618).sin() as f32 * 0.1
            }).collect();
            let y_orig = matmul_row(&x, &rows);
            let y_recon = matmul_row(&x, &recon);
            if argmax(&y_orig) == argmax(&y_recon) { match_count += 1; }
            cos_sum += cosine_f32_to_f64_simd(&y_orig, &y_recon);
            let y_xor = matmul_row(&x, &xor_recon);
            if argmax(&y_orig) == argmax(&y_xor) { xor_match += 1; }
            xor_cos += cosine_f32_to_f64_simd(&y_orig, &y_xor);
        }

        let match_pct = match_count as f64 / n_test as f64 * 100.0;
        let avg_cos = cos_sum / n_test as f64;

        let xor_pct = xor_match as f64 / n_test as f64 * 100.0;
        let xor_avg = xor_cos / n_test as f64;
        println!("| {} | CLAM: {:.0}% cos={:.4} | XOR: {:.0}% cos={:.4} flip={:.1}% bpr={:.0} |",
            label, match_pct, avg_cos, xor_pct, xor_avg,
            xor_tensor.avg_flipped_ratio() * 100.0,
            xor_tensor.bytes_per_row_avg());
    }
}
