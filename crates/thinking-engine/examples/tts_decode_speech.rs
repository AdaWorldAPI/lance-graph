//! Speech tokenizer decoder: RVQ codebook lookup → conv1d stack → PCM.
//!
//! Loads the decoder conv weights from speech_tokenizer/model.safetensors
//! and runs the forward pass on RVQ embeddings from cascade codes.
//!
//! The decoder is pure conv1d — no attention, no matmul, 52M params.
//! Architecture: input_proj(conv7) → 4 upsample blocks → output_conv → PCM.
//!
//! ```sh
//! cargo run --release --example tts_decode_speech \
//!     --manifest-path crates/thinking-engine/Cargo.toml
//! ```

use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::audio::synth;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

const ST_PATH: &str = "/home/user/models/qwen3-tts-0.6b/speech_tokenizer/model.safetensors";
const CODES_PATH: &str = "/home/user/models/qwen3-tts-0.6b/codebooks/cascade_audio_codes.bin";
const WAV_PATH: &str = "/home/user/models/cascade_speech.wav";
const SAMPLE_RATE: u32 = 24000;

/// Simple 1D convolution: output[n] = sum_k(input[n*stride + k] * kernel[k]) + bias
fn conv1d(input: &[f32], in_ch: usize, out_ch: usize, kernel_size: usize,
           weight: &[f32], bias: &[f32], stride: usize, padding: usize) -> Vec<f32> {
    let in_len = input.len() / in_ch;
    let out_len = (in_len + 2 * padding - kernel_size) / stride + 1;
    let mut output = vec![0.0f32; out_len * out_ch];

    for oc in 0..out_ch {
        for n in 0..out_len {
            let mut sum = bias.get(oc).copied().unwrap_or(0.0);
            for ic in 0..in_ch {
                for k in 0..kernel_size {
                    let in_pos = n * stride + k;
                    let in_pos = if in_pos >= padding { in_pos - padding } else { continue };
                    if in_pos >= in_len { continue; }
                    // weight layout: [out_ch, in_ch, kernel_size]
                    let w_idx = oc * in_ch * kernel_size + ic * kernel_size + k;
                    let i_idx = in_pos * in_ch + ic;
                    sum += input.get(i_idx).copied().unwrap_or(0.0)
                         * weight.get(w_idx).copied().unwrap_or(0.0);
                }
            }
            output[n * out_ch + oc] = sum;
        }
    }
    output
}

/// Snake activation: x + (1/alpha) * sin²(alpha * x)
fn snake_activation(x: &mut [f32], alpha: &[f32], beta: &[f32], channels: usize) {
    let len = x.len() / channels;
    for n in 0..len {
        for c in 0..channels {
            let idx = n * channels + c;
            let a = alpha.get(c).copied().unwrap_or(1.0);
            let b = beta.get(c).copied().unwrap_or(1.0);
            let v = x[idx];
            let sin_val = (b * v).sin();
            x[idx] = v + sin_val * sin_val / a.max(1e-8);
        }
    }
}

/// Read f32 tensor from safetensors by name.
fn read_tensor(reader: &mut BufReader<File>, header: &ndarray::hpc::gguf::GgufFile, name: &str) -> Option<Vec<f32>> {
    let tensor = header.tensors.iter().find(|t| t.name == name)?;
    let n_elements: usize = tensor.dimensions.iter().map(|&d| d as usize).product();
    reader.seek(SeekFrom::Start(header.tensor_data_offset + tensor.offset)).ok()?;
    let mut raw = vec![0u8; n_elements * 4];
    reader.read_exact(&mut raw).ok()?;
    Some(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn main() {
    println!("═══ TTS SPEECH DECODER (conv1d from safetensors) ═══\n");

    // Step 1: Load RVQ codebook embeddings
    let t0 = Instant::now();
    let mut reader = BufReader::new(File::open(ST_PATH).expect("open speech tokenizer"));
    let header = read_safetensors_header(&mut reader).expect("parse header");
    println!("[1] Speech tokenizer: {} tensors", header.tensors.len());

    // Load RVQ codebooks: embedding_sum / cluster_usage → normalized codebook
    // rvq_first (semantic, 1 layer) + rvq_rest (acoustic, 15 layers) = 16 total
    let mut codebooks: Vec<Vec<f32>> = Vec::new(); // 16 × [2048 × 256]
    let codebook_dim = 256usize;
    let codebook_size = 2048usize;

    // Semantic quantizer
    if let (Some(emb), Some(usage)) = (
        read_tensor(&mut reader, &header, "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"),
        read_tensor(&mut reader, &header, "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"),
    ) {
        let mut cb = vec![0.0f32; codebook_size * codebook_dim];
        for i in 0..codebook_size {
            let u = usage[i].max(1.0);
            for d in 0..codebook_dim {
                cb[i * codebook_dim + d] = emb[i * codebook_dim + d] / u;
            }
        }
        codebooks.push(cb);
    }

    // 15 acoustic quantizers
    for layer in 0..15 {
        let emb_name = format!("decoder.quantizer.rvq_rest.vq.layers.{}._codebook.embedding_sum", layer);
        let usage_name = format!("decoder.quantizer.rvq_rest.vq.layers.{}._codebook.cluster_usage", layer);
        if let (Some(emb), Some(usage)) = (read_tensor(&mut reader, &header, &emb_name),
                                            read_tensor(&mut reader, &header, &usage_name)) {
            let mut cb = vec![0.0f32; codebook_size * codebook_dim];
            for i in 0..codebook_size {
                let u = usage[i].max(1.0);
                for d in 0..codebook_dim {
                    cb[i * codebook_dim + d] = emb[i * codebook_dim + d] / u;
                }
            }
            codebooks.push(cb);
        }
    }
    println!("    Loaded {} RVQ codebooks in {:?}", codebooks.len(), t0.elapsed());

    // Step 2: Load cascade audio codes
    let code_bytes = std::fs::read(CODES_PATH).expect("read cascade codes");
    let n_frames = code_bytes.len() / 16;
    println!("[2] {} frames of cascade codes", n_frames);

    // Step 3: RVQ lookup → sum embeddings → latent sequence
    let t0 = Instant::now();
    let n_cb = codebooks.len().min(16);
    let mut latent = vec![0.0f32; n_frames * codebook_dim]; // [n_frames × 256]

    for frame in 0..n_frames {
        for g in 0..n_cb {
            let code = code_bytes[frame * 16 + g] as usize;
            let code = code.min(codebook_size - 1);
            for d in 0..codebook_dim {
                latent[frame * codebook_dim + d] += codebooks[g][code * codebook_dim + d];
            }
        }
    }
    println!("[3] RVQ lookup: {} frames × {} dim latent in {:?}",
        n_frames, codebook_dim, t0.elapsed());

    // Check latent is non-trivial
    let latent_rms: f32 = (latent.iter().map(|v| v * v).sum::<f32>() / latent.len() as f32).sqrt();
    println!("    Latent RMS: {:.4}", latent_rms);

    // Step 4: Project through output_proj (256→512)
    let t0 = Instant::now();
    let output_proj_w = read_tensor(&mut reader, &header,
        "decoder.quantizer.rvq_first.output_proj.weight")
        .expect("output_proj weight");
    // output_proj is conv1d [512, 256, 1] — pointwise projection
    let proj_out_ch = 512;
    let mut projected = vec![0.0f32; n_frames * proj_out_ch];
    for frame in 0..n_frames {
        for oc in 0..proj_out_ch {
            let mut sum = 0.0f32;
            for ic in 0..codebook_dim {
                sum += latent[frame * codebook_dim + ic]
                     * output_proj_w[oc * codebook_dim + ic]; // [out, in, 1] kernel=1
            }
            projected[frame * proj_out_ch + oc] = sum;
        }
    }
    println!("[4] Projected 256→512 in {:?}", t0.elapsed());

    // Step 5: Full decoder conv stack
    // Block 0: conv1d [1536, 1024, 7] — input projection
    // Input is 512-dim, decoder expects 1024. Duplicate channels.
    let t0 = Instant::now();
    let din = 1024;
    let mut decoder_in = vec![0.0f32; n_frames * din];
    for f in 0..n_frames {
        for d in 0..proj_out_ch {
            decoder_in[f * din + d] = projected[f * proj_out_ch + d];
            decoder_in[f * din + proj_out_ch + d] = projected[f * proj_out_ch + d];
        }
    }

    let w = read_tensor(&mut reader, &header, "decoder.decoder.0.conv.weight").expect("d0.w");
    let b = read_tensor(&mut reader, &header, "decoder.decoder.0.conv.bias").expect("d0.b");
    let mut x = conv1d(&decoder_in, din, 1536, 7, &w, &b, 1, 3);
    let mut ch = 1536usize;
    let mut len = x.len() / ch;
    println!("[5] Block 0: {}→{} ch, {} frames ({:?})", din, ch, len, t0.elapsed());

    // Blocks 1-4: snake → transposed_conv (upsample) → 3 residual blocks
    let upsample_config: [(usize, usize, usize, usize); 4] = [
        // (block_idx, out_ch, kernel_size, stride)
        (1, 768, 16, 8),
        (2, 384, 10, 5),
        (3, 192, 8, 4),
        (4, 96, 6, 3),
    ];

    for &(blk, out_ch, kern, stride) in &upsample_config {
        let t0 = Instant::now();
        // Snake activation
        let alpha = read_tensor(&mut reader, &header, &format!("decoder.decoder.{}.block.0.alpha", blk)).unwrap_or_else(|| vec![1.0; ch]);
        let beta = read_tensor(&mut reader, &header, &format!("decoder.decoder.{}.block.0.beta", blk)).unwrap_or_else(|| vec![1.0; ch]);
        snake_activation(&mut x, &alpha, &beta, ch);

        // Transposed conv (upsample): nearest-neighbor + conv
        // Real transposed conv inserts stride-1 zeros between samples then convolves.
        // Approximation: repeat each frame stride× then apply regular conv.
        let up_len = len * stride;
        let mut upsampled = vec![0.0f32; up_len * ch];
        for n in 0..len {
            for s in 0..stride {
                for c in 0..ch {
                    upsampled[(n * stride + s) * ch + c] = x[n * ch + c];
                }
            }
        }

        // Apply the transposed conv weight as regular conv on upsampled data
        let tw = read_tensor(&mut reader, &header, &format!("decoder.decoder.{}.block.1.conv.weight", blk)).expect("upsample weight");
        let tb = read_tensor(&mut reader, &header, &format!("decoder.decoder.{}.block.1.conv.bias", blk)).expect("upsample bias");
        x = conv1d(&upsampled, ch, out_ch, kern, &tw, &tb, 1, kern / 2);
        ch = out_ch;
        len = x.len() / ch;

        // 3 residual blocks: each has conv1(7) + conv2(1)
        for res in 2..=4 {
            let a1 = read_tensor(&mut reader, &header, &format!("decoder.decoder.{}.block.{}.act1.alpha", blk, res)).unwrap_or_else(|| vec![1.0; ch]);
            let b1 = read_tensor(&mut reader, &header, &format!("decoder.decoder.{}.block.{}.act1.beta", blk, res)).unwrap_or_else(|| vec![1.0; ch]);
            let mut residual = x.clone();
            snake_activation(&mut residual, &a1, &b1, ch);

            let rw1 = read_tensor(&mut reader, &header, &format!("decoder.decoder.{}.block.{}.conv1.conv.weight", blk, res)).expect("res conv1 w");
            let rb1 = read_tensor(&mut reader, &header, &format!("decoder.decoder.{}.block.{}.conv1.conv.bias", blk, res)).expect("res conv1 b");
            residual = conv1d(&residual, ch, ch, 7, &rw1, &rb1, 1, 3);

            let a2 = read_tensor(&mut reader, &header, &format!("decoder.decoder.{}.block.{}.act2.alpha", blk, res)).unwrap_or_else(|| vec![1.0; ch]);
            let b2 = read_tensor(&mut reader, &header, &format!("decoder.decoder.{}.block.{}.act2.beta", blk, res)).unwrap_or_else(|| vec![1.0; ch]);
            snake_activation(&mut residual, &a2, &b2, ch);

            let rw2 = read_tensor(&mut reader, &header, &format!("decoder.decoder.{}.block.{}.conv2.conv.weight", blk, res)).expect("res conv2 w");
            let rb2 = read_tensor(&mut reader, &header, &format!("decoder.decoder.{}.block.{}.conv2.conv.bias", blk, res)).expect("res conv2 b");
            residual = conv1d(&residual, ch, ch, 1, &rw2, &rb2, 1, 0);

            // Add residual
            let rlen = x.len().min(residual.len());
            for i in 0..rlen { x[i] += residual[i]; }
        }

        println!("    Block {}: {}ch, {} frames, upsample {}× ({:?})", blk, ch, len, stride, t0.elapsed());
    }

    // Block 5: final snake activation
    let a5 = read_tensor(&mut reader, &header, "decoder.decoder.5.alpha").unwrap_or_else(|| vec![1.0; ch]);
    let b5 = read_tensor(&mut reader, &header, "decoder.decoder.5.beta").unwrap_or_else(|| vec![1.0; ch]);
    snake_activation(&mut x, &a5, &b5, ch);

    // Block 6: output conv [1, 96, 7] → mono PCM
    let w6 = read_tensor(&mut reader, &header, "decoder.decoder.6.conv.weight").expect("output weight");
    let b6 = read_tensor(&mut reader, &header, "decoder.decoder.6.conv.bias").expect("output bias");
    let pcm_raw = conv1d(&x, ch, 1, 7, &w6, &b6, 1, 3);
    let pcm: Vec<f32> = pcm_raw.iter().map(|&v| v.tanh()).collect(); // tanh output activation

    let pcm_rms = (pcm.iter().map(|v| v * v).sum::<f32>() / pcm.len() as f32).sqrt();
    println!("[6] PCM: {} samples ({:.2}s at {}Hz), RMS={:.4}",
        pcm.len(), pcm.len() as f64 / SAMPLE_RATE as f64, SAMPLE_RATE, pcm_rms);

    // Write WAV
    let wav = synth::write_wav(&pcm, SAMPLE_RATE);
    std::fs::write(WAV_PATH, &wav).expect("write WAV");
    println!("[7] WAV: {} ({} bytes)", WAV_PATH, wav.len());

    println!("\n═══ DONE ═══");
}
