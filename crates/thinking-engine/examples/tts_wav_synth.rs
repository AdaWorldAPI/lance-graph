//! TTS WAV synthesizer: cascade audio codes → AudioFrame → iMDCT → PCM.
//!
//! Correct pipeline (replaces bits-as-vectors mistake):
//!   cascade audio codes (16 × u8 per frame)
//!     → map to band energies (via archetype palette + highheelbgz spiral)
//!       → AudioFrame (48 bytes: 21 BF16 band energies + 6B PVQ summary)
//!         → decode_coarse() → iMDCT → PCM → WAV
//!
//! HHTL routing via PVQ summary:
//!   HEEL: PVQ summary bytes 0-1 (sign pattern → spectral category)
//!   HIP:  band energies (BF16 gain → L1 distance)
//!   TWIG: PVQ summary bytes 4-5 (harmonic detail)
//!   LEAF: full iMDCT decode
//!
//! ```sh
//! cargo run --release --example tts_wav_synth \
//!     --manifest-path crates/thinking-engine/Cargo.toml
//! ```

use ndarray::hpc::audio::bands;
use ndarray::hpc::audio::voice::{VoiceArchetype, VoiceCodebook, VoiceFrame, RvqFrame};
use ndarray::hpc::audio::phase::PhaseDescriptor;
use ndarray::hpc::audio::synth;
use bgz_tensor::hhtl_cache::HhtlCache;
use std::time::Instant;

const CODES_PATH: &str = "/home/user/models/qwen3-tts-0.6b/codebooks/cascade_audio_codes.bin";
const CODEBOOK_DIR: &str = "/home/user/models/qwen3-tts-0.6b/codebooks";
const WAV_PATH: &str = "/home/user/models/cascade_output.wav";
const SAMPLE_RATE: u32 = 24000;


fn main() {
    println!("═══ TTS WAV SYNTHESIZER (synth.rs pipeline) ═══\n");

    // Step 1: Load HHTL cache → build coarse centroids from palette
    let hhtl_path = format!("{}/code_predictor_gate_proj_hhtl.bgz", CODEBOOK_DIR);
    let palette_cache = match HhtlCache::deserialize(&hhtl_path) {
        Ok(c) => {
            println!("[1] HHTL cache loaded: k={}, gamma=[{:.4},{:.4},{:.4}]",
                c.k(), c.gamma_meta[0], c.gamma_meta[1], c.gamma_meta[2]);
            c
        }
        Err(e) => {
            eprintln!("[1] Cannot load {}: {}", hhtl_path, e);
            return;
        }
    };

    // Build coarse_centroids from HHTL palette: Base17 → 21 BF16 band energies
    let gamma = ((palette_cache.gamma_meta[0] + palette_cache.gamma_meta[1]) / 2.0).max(0.001);
    let mut coarse_centroids = [[0u16; bands::N_BANDS]; 256];
    for (idx, entry) in palette_cache.palette.entries.iter().enumerate().take(256) {
        let mut energies = [0.0f32; bands::N_BANDS];
        for band in 0..bands::N_BANDS {
            let center = (bands::CELT_BANDS_48K[band] + bands::CELT_BANDS_48K[band + 1]) / 2;
            let dim = (center * 17 / 480).min(16);
            energies[band] = (entry.dims[dim] as f32).abs() * gamma;
        }
        coarse_centroids[idx] = bands::energies_to_bf16(&energies);
    }
    println!("    Built 256 coarse centroids from palette Base17 × gamma");

    // Step 2: Load cascade audio codes
    let code_bytes = match std::fs::read(CODES_PATH) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[2] Cannot read {}: {}", CODES_PATH, e);
            return;
        }
    };
    let n_frames = code_bytes.len() / 16;
    println!("[2] {} frames of audio codes loaded", n_frames);

    // Step 3: Convert 16-byte cascade codes → VoiceFrame (21 bytes)
    // Cascade: [16 × u8] → RvqFrame { archetype, coarse[8], fine[8] } + PhaseDescriptor
    let codebook = VoiceCodebook { entries: (0..256).map(|_| VoiceArchetype::zero()).collect() };
    let voice_frames: Vec<VoiceFrame> = (0..n_frames).map(|i| {
        let offset = i * 16;
        let codes = &code_bytes[offset..offset + 16];
        // Map: code[0] = archetype, codes[1..9] = coarse, codes[9..16]+pad = fine
        let mut coarse = [0u8; 8];
        let mut fine = [0u8; 8];
        coarse.copy_from_slice(&codes[0..8]);
        fine[..7].copy_from_slice(&codes[8..15]);
        fine[7] = codes[15];

        VoiceFrame {
            rvq: RvqFrame {
                archetype: codes[0],
                coarse,
                fine,
            },
            phase: PhaseDescriptor {
                bytes: [codes[1], codes[5], codes[9], codes[13]],
            },
        }
    }).collect();
    println!("[3] Converted to {} VoiceFrames", voice_frames.len());

    // Step 4: Synthesize using synth.rs (overlap-add, phase modulation, the works)
    let t0 = Instant::now();
    let pcm = synth::synthesize(&voice_frames, &codebook, &coarse_centroids, SAMPLE_RATE);
    println!("[4] Synthesized {} samples in {:?} ({:.2}s at {}Hz)",
        pcm.len(), t0.elapsed(), pcm.len() as f64 / SAMPLE_RATE as f64, SAMPLE_RATE);

    // Step 5: Write WAV using synth.rs (proper normalization + header)
    let wav = synth::write_wav(&pcm, SAMPLE_RATE);
    std::fs::write(WAV_PATH, &wav).expect("write WAV");
    println!("[5] WAV written: {} ({} bytes, {:.1} KB)", WAV_PATH, wav.len(), wav.len() as f64 / 1024.0);

    // Validate
    if let Ok((sr, n)) = synth::validate_wav(&wav) {
        println!("    Validated: {}Hz, {} samples", sr, n);
    }

    println!("\n═══ DONE ═══");
}
