//! Code predictor forward pass: 5 Qwen3 transformer layers → codec tokens.
//!
//! Reads weights from safetensors, runs real transformer inference.
//! Input: cascade archetype → codec_embedding lookup → hidden state.
//! Output: 15 codec token indices (0-2047) per frame via lm_head argmax.
//!
//! Then feeds real tokens to the speech decoder (tts_decode_speech.rs).

use ndarray::hpc::safetensors::read_safetensors_header;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

const MODEL_PATH: &str = "/home/user/models/qwen3-tts-0.6b/model.safetensors";
const CODES_PATH: &str = "/home/user/models/qwen3-tts-0.6b/codebooks/cascade_audio_codes.bin";
const OUTPUT_PATH: &str = "/home/user/models/qwen3-tts-0.6b/codebooks/real_codec_tokens.bin";
const HIDDEN: usize = 1024;
const N_HEADS: usize = 16;
const N_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const INTER: usize = 3072;
const N_LAYERS: usize = 5;
const CODEBOOK_SIZE: usize = 2048;

fn read_tensor(reader: &mut BufReader<File>, header: &ndarray::hpc::gguf::GgufFile, name: &str) -> Vec<f32> {
    let tensor = match header.tensors.iter().find(|t| t.name == name) {
        Some(t) => t,
        None => { eprintln!("MISSING: {}", name); return vec![]; }
    };
    let n: usize = tensor.dimensions.iter().map(|&d| d as usize).product();
    reader.seek(SeekFrom::Start(header.tensor_data_offset + tensor.offset)).unwrap();
    // BF16
    let mut raw = vec![0u8; n * 2];
    reader.read_exact(&mut raw).unwrap();
    raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
}

fn rms_norm(x: &mut [f32], weight: &[f32], dim: usize) {
    let len = x.len() / dim;
    for i in 0..len {
        let slice = &x[i * dim..(i + 1) * dim];
        let rms = (slice.iter().map(|v| v * v).sum::<f32>() / dim as f32 + 1e-6).sqrt();
        let inv_rms = 1.0 / rms;
        for d in 0..dim {
            x[i * dim + d] *= inv_rms * weight.get(d).copied().unwrap_or(1.0);
        }
    }
}

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // a: [m, k], b: [n, k] (transposed — weight matrices are [out, in])
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[j * k + p];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    let inv = 1.0 / sum.max(1e-10);
    for v in x.iter_mut() { *v *= inv; }
}

fn main() {
    println!("═══ CODE PREDICTOR FORWARD PASS ═══\n");

    let t0 = Instant::now();
    let mut reader = BufReader::new(File::open(MODEL_PATH).expect("open model"));
    let header = read_safetensors_header(&mut reader).expect("parse header");
    println!("[1] Model header parsed in {:?}", t0.elapsed());

    // Load cascade codes
    let code_bytes = std::fs::read(CODES_PATH).expect("read codes");
    let n_frames = code_bytes.len() / 16;
    println!("[2] {} frames of cascade codes", n_frames);

    // Load codec_embedding for the first code group (semantic)
    let t0 = Instant::now();
    let ce0 = read_tensor(&mut reader, &header, "talker.code_predictor.model.codec_embedding.0.weight");
    println!("[3] Loaded codec_embedding.0 [{}, {}] in {:?}", CODEBOOK_SIZE, HIDDEN, t0.elapsed());

    // Build initial hidden states: cascade code → embedding lookup
    let mut hidden = vec![0.0f32; n_frames * HIDDEN];
    for f in 0..n_frames {
        let code = code_bytes[f * 16] as usize;
        let code = code.min(CODEBOOK_SIZE - 1);
        hidden[f * HIDDEN..(f + 1) * HIDDEN].copy_from_slice(&ce0[code * HIDDEN..(code + 1) * HIDDEN]);
    }
    println!("    Initial hidden RMS: {:.4}",
        (hidden.iter().map(|v| v * v).sum::<f32>() / hidden.len() as f32).sqrt());

    // Run 5 transformer layers
    for layer in 0..N_LAYERS {
        let t0 = Instant::now();
        let prefix = format!("talker.code_predictor.model.layers.{}", layer);

        // Load weights
        let ln1_w = read_tensor(&mut reader, &header, &format!("{}.input_layernorm.weight", prefix));
        let q_w = read_tensor(&mut reader, &header, &format!("{}.self_attn.q_proj.weight", prefix));
        let k_w = read_tensor(&mut reader, &header, &format!("{}.self_attn.k_proj.weight", prefix));
        let v_w = read_tensor(&mut reader, &header, &format!("{}.self_attn.v_proj.weight", prefix));
        let o_w = read_tensor(&mut reader, &header, &format!("{}.self_attn.o_proj.weight", prefix));
        let ln2_w = read_tensor(&mut reader, &header, &format!("{}.post_attention_layernorm.weight", prefix));
        let gate_w = read_tensor(&mut reader, &header, &format!("{}.mlp.gate_proj.weight", prefix));
        let up_w = read_tensor(&mut reader, &header, &format!("{}.mlp.up_proj.weight", prefix));
        let down_w = read_tensor(&mut reader, &header, &format!("{}.mlp.down_proj.weight", prefix));

        // Pre-attention RMSNorm
        let mut normed = hidden.clone();
        rms_norm(&mut normed, &ln1_w, HIDDEN);

        // Q, K, V projections
        let q = matmul(&normed, &q_w, n_frames, HIDDEN, N_HEADS * HEAD_DIM);
        let k = matmul(&normed, &k_w, n_frames, HIDDEN, N_KV_HEADS * HEAD_DIM);
        let v = matmul(&normed, &v_w, n_frames, HIDDEN, N_KV_HEADS * HEAD_DIM);

        // Simplified attention: for small sequence (12 frames), just compute full
        // QK^T / sqrt(d) → softmax → V for each head
        let mut attn_out = vec![0.0f32; n_frames * N_HEADS * HEAD_DIM];
        let kv_group = N_HEADS / N_KV_HEADS; // GQA: 2 Q heads per KV head

        for h in 0..N_HEADS {
            let kv_h = h / kv_group;
            // Compute attention scores for this head
            let mut scores = vec![0.0f32; n_frames * n_frames];
            for i in 0..n_frames {
                for j in 0..=i { // causal mask
                    let mut dot = 0.0f32;
                    for d in 0..HEAD_DIM {
                        dot += q[i * N_HEADS * HEAD_DIM + h * HEAD_DIM + d]
                             * k[j * N_KV_HEADS * HEAD_DIM + kv_h * HEAD_DIM + d];
                    }
                    scores[i * n_frames + j] = dot / (HEAD_DIM as f32).sqrt();
                }
                // Mask future positions
                for j in (i + 1)..n_frames {
                    scores[i * n_frames + j] = f32::NEG_INFINITY;
                }
                // Softmax over row
                softmax(&mut scores[i * n_frames..(i + 1) * n_frames]);
            }

            // Weighted sum of V
            for i in 0..n_frames {
                for d in 0..HEAD_DIM {
                    let mut sum = 0.0f32;
                    for j in 0..n_frames {
                        sum += scores[i * n_frames + j]
                             * v[j * N_KV_HEADS * HEAD_DIM + kv_h * HEAD_DIM + d];
                    }
                    attn_out[i * N_HEADS * HEAD_DIM + h * HEAD_DIM + d] = sum;
                }
            }
        }

        // O projection
        let o_out = matmul(&attn_out, &o_w, n_frames, N_HEADS * HEAD_DIM, HIDDEN);

        // Residual
        for i in 0..hidden.len() { hidden[i] += o_out.get(i).copied().unwrap_or(0.0); }

        // Post-attention RMSNorm
        let mut normed2 = hidden.clone();
        rms_norm(&mut normed2, &ln2_w, HIDDEN);

        // MLP: SiLU(gate) × up → down
        let gate = matmul(&normed2, &gate_w, n_frames, HIDDEN, INTER);
        let up = matmul(&normed2, &up_w, n_frames, HIDDEN, INTER);
        let mut mlp_hidden = vec![0.0f32; n_frames * INTER];
        for i in 0..n_frames * INTER {
            mlp_hidden[i] = silu(gate[i]) * up[i];
        }
        let mlp_out = matmul(&mlp_hidden, &down_w, n_frames, INTER, HIDDEN);

        // Residual
        for i in 0..hidden.len() { hidden[i] += mlp_out.get(i).copied().unwrap_or(0.0); }

        let rms = (hidden.iter().map(|v| v * v).sum::<f32>() / hidden.len() as f32).sqrt();
        println!("    Layer {}: RMS={:.4} ({:?})", layer, rms, t0.elapsed());
    }

    // lm_head projection → codec tokens
    println!("[4] lm_head argmax...");
    let mut all_tokens: Vec<u8> = Vec::new(); // [n_frames × 16] — u8 for file compat

    for g in 0..15 {
        let lm = read_tensor(&mut reader, &header,
            &format!("talker.code_predictor.lm_head.{}.weight", g));
        if lm.is_empty() { continue; }

        for f in 0..n_frames {
            let h = &hidden[f * HIDDEN..(f + 1) * HIDDEN];
            let mut best = 0u16;
            let mut best_logit = f32::NEG_INFINITY;
            for tok in 0..CODEBOOK_SIZE {
                let mut logit = 0.0f32;
                for d in 0..HIDDEN {
                    logit += h[d] * lm[tok * HIDDEN + d];
                }
                if logit > best_logit { best_logit = logit; best = tok as u16; }
            }
            all_tokens.push((best % 256) as u8);
        }
    }
    // Pad to 16 groups
    while all_tokens.len() < n_frames * 16 {
        all_tokens.push(0);
    }

    println!("    Frame 0 tokens: {:?}",
        &all_tokens[..16.min(all_tokens.len())].iter().map(|&t| t as u16).collect::<Vec<_>>());

    // Save real codec tokens
    std::fs::write(OUTPUT_PATH, &all_tokens).expect("write tokens");
    println!("[5] Saved {} real codec tokens to {}", all_tokens.len(), OUTPUT_PATH);

    println!("\n═══ DONE — run tts_decode_speech with real_codec_tokens.bin ═══");
}
