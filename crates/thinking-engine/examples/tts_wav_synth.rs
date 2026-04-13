//! Minimal WAV synthesizer from cascade audio codes.
//!
//! Reads the 16 RVQ codebooks from speech tokenizer safetensors,
//! looks up the cascade-generated audio codes, sums embeddings,
//! and synthesizes a WAV file via simple spectral → PCM conversion.
//!
//! This is the minimum viable audio output — proves the cascade
//! produces something audible, not a production-quality decoder.
//!
//! ```sh
//! cargo run --release --example tts_wav_synth \
//!     --manifest-path crates/thinking-engine/Cargo.toml
//! ```

use ndarray::hpc::safetensors::read_safetensors_header;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::time::Instant;

const ST_PATH: &str = "/home/user/models/qwen3-tts-0.6b/speech_tokenizer/model.safetensors";
const CODES_PATH: &str = "/home/user/models/qwen3-tts-0.6b/codebooks/cascade_audio_codes.bin";
const WAV_PATH: &str = "/home/user/models/cascade_output.wav";
const SAMPLE_RATE: u32 = 24000;
const UPSAMPLE: usize = 1920; // samples per frame (80ms at 24kHz)
const CODEBOOK_DIM: usize = 256;
const CODEBOOK_SIZE: usize = 2048;

/// Read a f32 tensor from safetensors by name.
fn read_f32_tensor(
    reader: &mut BufReader<File>,
    header: &ndarray::hpc::gguf::GgufFile,
    name: &str,
) -> Option<Vec<f32>> {
    let tensor = header.tensors.iter().find(|t| t.name == name)?;
    let n_elements: usize = tensor.dimensions.iter().map(|&d| d as usize).product();
    let byte_offset = header.tensor_data_offset + tensor.offset;

    reader.seek(SeekFrom::Start(byte_offset)).ok()?;

    // This tokenizer uses f32 (not BF16)
    let mut raw = vec![0u8; n_elements * 4];
    reader.read_exact(&mut raw).ok()?;

    let data: Vec<f32> = raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some(data)
}

/// Write a WAV file header + PCM data.
fn write_wav(path: &str, samples: &[i16], sample_rate: u32) {
    let mut f = File::create(path).expect("create WAV");
    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;

    // RIFF header
    f.write_all(b"RIFF").unwrap();
    f.write_all(&file_size.to_le_bytes()).unwrap();
    f.write_all(b"WAVE").unwrap();

    // fmt chunk
    f.write_all(b"fmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap(); // chunk size
    f.write_all(&1u16.to_le_bytes()).unwrap(); // PCM format
    f.write_all(&1u16.to_le_bytes()).unwrap(); // mono
    f.write_all(&sample_rate.to_le_bytes()).unwrap();
    f.write_all(&(sample_rate * 2).to_le_bytes()).unwrap(); // byte rate
    f.write_all(&2u16.to_le_bytes()).unwrap(); // block align
    f.write_all(&16u16.to_le_bytes()).unwrap(); // bits per sample

    // data chunk
    f.write_all(b"data").unwrap();
    f.write_all(&data_size.to_le_bytes()).unwrap();
    for &s in samples {
        f.write_all(&s.to_le_bytes()).unwrap();
    }
}

fn main() {
    println!("═══ TTS WAV SYNTHESIZER ═══\n");

    // Step 1: Load cascade audio codes
    let code_bytes = std::fs::read(CODES_PATH).expect("read audio codes");
    let n_frames = code_bytes.len() / 16;
    println!("[1] {} frames of audio codes loaded", n_frames);

    // Step 2: Load RVQ codebooks from speech tokenizer
    let t0 = Instant::now();
    let mut reader = BufReader::new(File::open(ST_PATH).expect("open speech tokenizer"));
    let header = read_safetensors_header(&mut reader).expect("parse header");

    // Load 16 codebooks: rvq_first (1 semantic) + rvq_rest (15 acoustic)
    let mut codebooks: Vec<Vec<f32>> = Vec::new(); // 16 × [2048 × 256]
    let mut usages: Vec<Vec<f32>> = Vec::new(); // 16 × [2048]

    // First: semantic quantizer
    if let Some(emb) = read_f32_tensor(&mut reader, &header,
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum") {
        let usage = read_f32_tensor(&mut reader, &header,
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage")
            .unwrap_or_else(|| vec![1.0; CODEBOOK_SIZE]);
        codebooks.push(emb);
        usages.push(usage);
    }

    // Rest: 15 acoustic quantizers
    for i in 0..15 {
        let name = format!("decoder.quantizer.rvq_rest.vq.layers.{}._codebook.embedding_sum", i);
        let usage_name = format!("decoder.quantizer.rvq_rest.vq.layers.{}._codebook.cluster_usage", i);
        if let Some(emb) = read_f32_tensor(&mut reader, &header, &name) {
            let usage = read_f32_tensor(&mut reader, &header, &usage_name)
                .unwrap_or_else(|| vec![1.0; CODEBOOK_SIZE]);
            codebooks.push(emb);
            usages.push(usage);
        }
    }
    println!("[2] Loaded {} codebooks in {:?}", codebooks.len(), t0.elapsed());

    // Normalize codebooks: embedding = embedding_sum / cluster_usage
    let mut normalized_cb: Vec<Vec<f32>> = Vec::new();
    for (cb, usage) in codebooks.iter().zip(usages.iter()) {
        let mut norm = vec![0.0f32; CODEBOOK_SIZE * CODEBOOK_DIM];
        for i in 0..CODEBOOK_SIZE {
            let u = usage[i].max(1.0); // avoid div by zero
            for d in 0..CODEBOOK_DIM {
                norm[i * CODEBOOK_DIM + d] = cb[i * CODEBOOK_DIM + d] / u;
            }
        }
        normalized_cb.push(norm);
    }

    // Step 3: Synthesize PCM from audio codes
    let t0 = Instant::now();
    let mut pcm_samples: Vec<i16> = Vec::with_capacity(n_frames * UPSAMPLE);
    let n_cb = normalized_cb.len().min(16);

    for frame_idx in 0..n_frames {
        // Sum embeddings from all 16 code groups
        let mut embedding = vec![0.0f32; CODEBOOK_DIM];
        for g in 0..n_cb {
            let code = code_bytes[frame_idx * 16 + g] as usize;
            let code_clamped = code.min(CODEBOOK_SIZE - 1);
            let cb = &normalized_cb[g];
            for d in 0..CODEBOOK_DIM {
                embedding[d] += cb[code_clamped * CODEBOOK_DIM + d];
            }
        }

        // Simple synthesis: treat embedding as spectral magnitudes,
        // generate PCM by summing cosine waves at those frequencies.
        // Each of the 256 dims maps to a frequency band.
        for s in 0..UPSAMPLE {
            let t = s as f64 / SAMPLE_RATE as f64;
            let mut sample = 0.0f64;
            // Use first 128 dims as frequency magnitudes (symmetric spectrum)
            for d in 0..128.min(CODEBOOK_DIM / 2) {
                let freq = (d + 1) as f64 * (SAMPLE_RATE as f64 / 2.0 / 128.0); // linear spacing
                let mag = embedding[d] as f64;
                sample += mag * (2.0 * std::f64::consts::PI * freq * t).cos();
            }
            // Normalize
            let clamped = (sample * 100.0).clamp(-32768.0, 32767.0) as i16;
            pcm_samples.push(clamped);
        }
    }
    println!("[3] Synthesized {} samples in {:?}", pcm_samples.len(), t0.elapsed());
    println!("    Duration: {:.1}s at {}Hz", pcm_samples.len() as f64 / SAMPLE_RATE as f64, SAMPLE_RATE);

    // Step 4: Write WAV
    write_wav(WAV_PATH, &pcm_samples, SAMPLE_RATE);
    let wav_size = std::fs::metadata(WAV_PATH).map(|m| m.len()).unwrap_or(0);
    println!("[4] WAV written: {} ({} bytes)", WAV_PATH, wav_size);

    println!("\n═══ DONE ═══");
}
