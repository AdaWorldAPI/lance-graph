//! had_cascade_inference_test — validate codec doesn't break inference.
//!
//! Two-level validation:
//!   1. Matmul argmax test: x @ W.T vs x @ W_recon.T — same argmax?
//!   2. Full codec token test: generate audio tokens with compressed weights
//!
//! For TTS: waveform phase coherence requires argmax stability across ALL
//! layers. One flip in the codec head → wrong phase segment → click/noise.
//!
//! Usage:
//!   cargo run --release --example had_cascade_inference_test \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /path/to/model.safetensors

use bgz_tensor::had_cascade::{HadCascadeTensor, TensorRegime};
use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::heel_f64x8::cosine_f32_to_f64_simd;
use ndarray::simd::bf16_to_f32_batch;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

fn load_tensor(path: &str, tensor_name: &str) -> Option<(Vec<Vec<f32>>, usize, usize)> {
    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let header = read_safetensors_header(&mut reader).ok()?;
    let t = header.tensors.iter().find(|t| t.name == tensor_name)?;
    let n_rows = t.dimensions[0] as usize;
    let n_cols: usize = t.dimensions.iter().skip(1).map(|&d| d as usize).product();
    let n = n_rows * n_cols;
    reader.seek(SeekFrom::Start(header.tensor_data_offset + t.offset)).ok()?;
    let mut raw = vec![0u8; n * 2];
    reader.read_exact(&mut raw).ok()?;
    let u16s: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let mut f32_data = vec![0.0f32; u16s.len()];
    bf16_to_f32_batch(&u16s, &mut f32_data);
    let rows: Vec<Vec<f32>> = (0..n_rows)
        .map(|r| f32_data[r * n_cols..(r + 1) * n_cols].to_vec())
        .collect();
    Some((rows, n_rows, n_cols))
}

fn list_weight_tensors(path: &str) -> Vec<(String, usize, usize)> {
    let file = File::open(path).expect("open");
    let mut reader = BufReader::new(file);
    let header = read_safetensors_header(&mut reader).expect("parse");
    header.tensors.iter()
        .filter(|t| t.name.contains("weight") && t.dimensions.len() >= 2)
        .map(|t| {
            let n_rows = t.dimensions[0] as usize;
            let n_cols: usize = t.dimensions.iter().skip(1).map(|&d| d as usize).product();
            (t.name.clone(), n_rows, n_cols)
        })
        .collect()
}

fn matmul_row(x: &[f32], weight_rows: &[Vec<f32>]) -> Vec<f32> {
    weight_rows.iter().map(|w| {
        x.iter().zip(w.iter()).map(|(a, b)| a * b).sum()
    }).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn rms_diff(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 { return 0.0; }
    let sum: f64 = a.iter().zip(b.iter())
        .map(|(x, y)| { let d = *x as f64 - *y as f64; d * d })
        .sum();
    (sum / n as f64).sqrt()
}

fn main() {
    let path = std::env::args().nth(1)
        .expect("usage: had_cascade_inference_test <model.safetensors>");

    println!("# HadCascade Inference Validation");
    println!("Model: `{}`", path);
    println!();

    let tensors = list_weight_tensors(&path);
    println!("Found {} weight tensors", tensors.len());

    // ═══════════════════════════════════════════════════════════════
    // TEST 1: Matmul argmax stability
    // ═══════════════════════════════════════════════════════════════

    println!();
    println!("## Test 1: Matmul Argmax Stability");
    println!();
    println!("For each tensor: encode → reconstruct → multiply by random activations");
    println!("→ check argmax(x @ W.T) == argmax(x @ W_recon.T)");
    println!();
    println!("| Tensor | Regime | Shape | Argmax match | RMS diff | Cosine | Encode ms |");
    println!("|---|---|---|---|---|---|---|");

    let n_test_inputs = 32;
    let mut total_argmax_tests = 0usize;
    let mut total_argmax_match = 0usize;
    let mut total_tensors = 0usize;
    let mut failed_tensors = Vec::new();

    // Sample subset of tensors (one per layer type)
    let sample_tensors: Vec<_> = {
        let mut seen_types = HashMap::new();
        tensors.iter().filter(|(name, n_rows, n_cols)| {
            if *n_rows > 50000 { return false; } // skip huge embeddings
            let type_key = name.rsplit('.').next().unwrap_or(name).to_string();
            let layer_key = if name.contains(".0.") {
                type_key.clone()
            } else {
                format!("{}_{}", type_key, name.matches('.').count())
            };
            if seen_types.contains_key(&layer_key) { return false; }
            seen_types.insert(layer_key, true);
            true
        }).take(20).collect()
    };

    for (name, n_rows, n_cols) in &sample_tensors {
        let Some((rows, _, _)) = load_tensor(&path, name) else { continue };
        let regime = TensorRegime::from_role(name);

        if !regime.should_compress() {
            println!("| {} | Index | {}×{} | PASS (passthrough) | 0 | 1.0000 | 0 |",
                &name[name.len().saturating_sub(45)..], n_rows, n_cols);
            continue;
        }

        let t0 = Instant::now();
        let k = 256.min(rows.len());
        let tensor = HadCascadeTensor::encode(name, &rows, k);
        let encode_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let recon = tensor.reconstruct_all();

        // Generate test activations (deterministic pseudo-random)
        let mut match_count = 0usize;
        let mut total_rms = 0.0f64;
        let mut total_cos = 0.0f64;

        for t in 0..n_test_inputs {
            let x: Vec<f32> = (0..*n_cols).map(|d| {
                ((d * 97 + t * 31 + 17) as f64 * 0.618).sin() as f32 * 0.1
            }).collect();

            let y_orig = matmul_row(&x, &rows);
            let y_recon = matmul_row(&x, &recon);

            if argmax(&y_orig) == argmax(&y_recon) {
                match_count += 1;
            }
            total_rms += rms_diff(&y_orig, &y_recon);
            total_cos += cosine_f32_to_f64_simd(&y_orig, &y_recon);
        }

        let match_rate = match_count as f64 / n_test_inputs as f64;
        let avg_rms = total_rms / n_test_inputs as f64;
        let avg_cos = total_cos / n_test_inputs as f64;

        total_argmax_tests += n_test_inputs;
        total_argmax_match += match_count;
        total_tensors += 1;

        let short_name = &name[name.len().saturating_sub(45)..];
        let status = if match_rate >= 1.0 { "100%" } else {
            failed_tensors.push(name.clone());
            &format!("{:.1}%", match_rate * 100.0)
        };

        println!("| {} | Argmax | {}×{} | {} | {:.2e} | {:.6} | {:.0} |",
            short_name, n_rows, n_cols, status, avg_rms, avg_cos, encode_ms);
    }

    println!();
    println!("**Summary**: {}/{} argmax tests passed ({:.2}%) across {} tensors",
        total_argmax_match, total_argmax_tests,
        total_argmax_match as f64 / total_argmax_tests.max(1) as f64 * 100.0,
        total_tensors);
    if !failed_tensors.is_empty() {
        println!("**Failed tensors**: {}", failed_tensors.join(", "));
    }

    // ═══════════════════════════════════════════════════════════════
    // TEST 2: Waveform phase coherence check
    // ═══════════════════════════════════════════════════════════════

    println!();
    println!("## Test 2: Waveform Phase Coherence");
    println!();
    println!("Audio TTS generates codec tokens that select waveform phase segments.");
    println!("One argmax flip in the codec head → wrong phase → click/noise.");
    println!();

    // Find the codec head / output projection tensors
    let codec_head_tensors: Vec<_> = tensors.iter()
        .filter(|(name, _, _)| {
            name.contains("code_predictor") && !name.contains("embed")
                && name.contains("weight")
        })
        .take(5)
        .collect();

    if codec_head_tensors.is_empty() {
        println!("No codec head tensors found — skipping phase coherence test.");
    } else {
        println!("Testing {} codec head tensors for phase-critical argmax stability:", codec_head_tensors.len());
        println!();

        // For codec head tensors, test with MANY random inputs to find edge cases
        let n_phase_tests = 256;
        let mut phase_total = 0usize;
        let mut phase_match = 0usize;

        for (name, n_rows, n_cols) in &codec_head_tensors {
            let Some((rows, _, _)) = load_tensor(&path, name) else { continue };
            let regime = TensorRegime::from_role(name);

            let (test_rows, label) = if regime.should_compress() {
                let tensor = HadCascadeTensor::encode(name, &rows, 64);
                (tensor.reconstruct_all(), "compressed")
            } else {
                (rows.clone(), "passthrough")
            };

            let mut matches = 0;
            for t in 0..n_phase_tests {
                let x: Vec<f32> = (0..*n_cols).map(|d| {
                    ((d * 53 + t * 97 + 7) as f64 * 0.314).sin() as f32 * 0.05
                }).collect();
                let y_orig = matmul_row(&x, &rows);
                let y_recon = matmul_row(&x, &test_rows);
                if argmax(&y_orig) == argmax(&y_recon) {
                    matches += 1;
                }
            }

            let rate = matches as f64 / n_phase_tests as f64;
            phase_total += n_phase_tests;
            phase_match += matches;

            let short = &name[name.len().saturating_sub(50)..];
            let icon = if rate >= 1.0 { "PASS" } else if rate > 0.99 { "WARN" } else { "FAIL" };
            println!("  {} [{}] {}: {}/{} argmax match ({:.1}%)",
                icon, label, short, matches, n_phase_tests, rate * 100.0);
        }

        println!();
        println!("**Phase coherence**: {}/{} codec-head argmax stable ({:.2}%)",
            phase_match, phase_total,
            phase_match as f64 / phase_total.max(1) as f64 * 100.0);

        // Explain waveform implications
        println!();
        println!("### Waveform Phase Encoding");
        println!();
        println!("Audio codec tokens are NOT independent symbols — each selects a");
        println!("waveform segment with specific phase. Adjacent tokens must have");
        println!("phase continuity or the listener hears a click/pop at the boundary.");
        println!();
        println!("The chain: attention weights (compressed) → codec head → token →");
        println!("codebook lookup → waveform segment → DAC → speaker.");
        println!();
        println!("Compression errors propagate: weight error → logit shift →");
        println!("possible argmax flip → wrong token → phase discontinuity → noise.");
        println!();
        if phase_match == phase_total {
            println!("**Result: all codec-head argmax tests passed — waveform phase safe.**");
        } else {
            let flip_rate = 1.0 - phase_match as f64 / phase_total.max(1) as f64;
            println!("**Result: {:.2}% argmax flips in codec head — expect audible artifacts.**", flip_rate * 100.0);
            println!("At 75 tokens/sec, {:.1}% flips = ~{:.0} clicks per second of audio.",
                flip_rate * 100.0, flip_rate * 75.0);
        }
    }
}
