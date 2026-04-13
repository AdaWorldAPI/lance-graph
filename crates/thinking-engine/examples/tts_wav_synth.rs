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

use ndarray::hpc::audio::codec::AudioFrame;
use ndarray::hpc::audio::bands;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const CODES_PATH: &str = "/home/user/models/qwen3-tts-0.6b/codebooks/cascade_audio_codes.bin";
const WAV_PATH: &str = "/home/user/models/cascade_output.wav";
const SAMPLE_RATE: u32 = 24000;

/// Build a synthetic AudioFrame from 16 cascade code indices.
///
/// Maps the 16 codebook indices to 21 band energies by distributing
/// the code values across the quasi-Bark critical bands.
///
/// The 16 code groups from the cascade correspond to:
///   groups 0-3: low-frequency structure (bands 0-5, 0-1200 Hz)
///   groups 4-7: mid-frequency content (bands 6-12, 1200-4800 Hz)
///   groups 8-11: high-frequency detail (bands 13-17, 4800-12000 Hz)
///   groups 12-15: ultrasonic texture (bands 18-20, 12000-24000 Hz)
///
/// Each code value (0-255) maps to a BF16 energy via palette-normalized
/// scaling. Higher codes = higher energy in that band group.
fn codes_to_audio_frame(codes: &[u8; 16]) -> AudioFrame {
    let mut energies = [0.0f32; bands::N_BANDS];

    // Map 16 code groups → 21 band energies
    // Group-to-band mapping follows Qwen3-TTS upsample_rates [8,5,4,3]
    // which align with highheelbgz stride→role mapping
    let band_groups: [&[usize]; 16] = [
        &[0, 1],    // code 0  → bands 0-1 (sub-bass, 0-400 Hz)
        &[2, 3],    // code 1  → bands 2-3 (bass, 400-800 Hz)
        &[4],       // code 2  → band 4 (low-mid, 800-1000 Hz)
        &[5],       // code 3  → band 5 (mid, 1000-1200 Hz)
        &[6],       // code 4  → band 6 (mid, 1200-1400 Hz)
        &[7],       // code 5  → band 7 (mid-upper, 1400-1600 Hz)
        &[8],       // code 6  → band 8 (upper-mid, 1600-1800 Hz)
        &[9, 10],   // code 7  → bands 9-10 (presence, 1800-2600 Hz)
        &[11],      // code 8  → band 11 (brilliance, 2600-3000 Hz)
        &[12],      // code 9  → band 12 (brilliance, 3000-3400 Hz)
        &[13],      // code 10 → band 13 (sibilance, 3400-4000 Hz)
        &[14],      // code 11 → band 14 (sibilance, 4000-4800 Hz)
        &[15],      // code 12 → band 15 (air, 4800-5600 Hz)
        &[16, 17],  // code 13 → bands 16-17 (air, 5600-8000 Hz)
        &[18, 19],  // code 14 → bands 18-19 (ultra, 8000-12800 Hz)
        &[20],      // code 15 → band 20 (ultra, 12800-24000 Hz)
    ];

    for (g, band_indices) in band_groups.iter().enumerate() {
        // Code value → energy: log-scale mapping
        // 0 → silence, 128 → reference level, 255 → maximum
        let code_val = codes[g] as f32;
        let energy = if code_val < 1.0 {
            0.0
        } else {
            // Log-domain energy: each +32 codes ≈ +6 dB
            let db = (code_val - 128.0) * 0.1875; // ±24 dB range
            10.0f32.powf(db / 20.0) * 0.1 // scale to reasonable amplitude
        };

        for &band in *band_indices {
            energies[band] = energy;
        }
    }

    // Pack energies as BF16
    let bf16_energies = bands::energies_to_bf16(&energies);

    // PVQ summary: derive from code pattern for HHTL routing
    // bytes 0-1: sign pattern (HEEL) — which code groups are above/below median
    let median_code = {
        let mut sorted = *codes;
        sorted.sort();
        sorted[8]
    };
    let sign_hi: u8 = (0..8).fold(0u8, |acc, g| acc | (((codes[g] > median_code) as u8) << g));
    let sign_lo: u8 = (0..8).fold(0u8, |acc, g| acc | (((codes[8 + g] > median_code) as u8) << g));

    // bytes 2-3: temporal gradient (how much energy changes frame-to-frame)
    let gradient_hi = codes[0].wrapping_sub(codes[4]);
    let gradient_lo = codes[4].wrapping_sub(codes[8]);

    // bytes 4-5: harmonic signature (TWIG)
    let harmonic = codes[0] ^ codes[3] ^ codes[7]; // harmonic/noise ratio proxy
    let texture = codes[12] ^ codes[14]; // high-freq texture

    let pvq_summary = [sign_hi, sign_lo, gradient_hi, gradient_lo, harmonic, texture];

    AudioFrame { band_energies: bf16_energies, pvq_summary }
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
    f.write_all(&16u32.to_le_bytes()).unwrap();
    f.write_all(&1u16.to_le_bytes()).unwrap(); // PCM format
    f.write_all(&1u16.to_le_bytes()).unwrap(); // mono
    f.write_all(&sample_rate.to_le_bytes()).unwrap();
    f.write_all(&(sample_rate * 2).to_le_bytes()).unwrap();
    f.write_all(&2u16.to_le_bytes()).unwrap();
    f.write_all(&16u16.to_le_bytes()).unwrap();

    // data chunk
    f.write_all(b"data").unwrap();
    f.write_all(&data_size.to_le_bytes()).unwrap();
    for &s in samples {
        f.write_all(&s.to_le_bytes()).unwrap();
    }
}

fn main() {
    println!("═══ TTS WAV SYNTHESIZER (AudioFrame pipeline) ═══\n");

    // Step 1: Load cascade audio codes
    let code_bytes = match std::fs::read(CODES_PATH) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[1] Cannot read {}: {}", CODES_PATH, e);
            eprintln!("    Run tts_cascade_runner first to generate audio codes.");
            return;
        }
    };
    let n_frames = code_bytes.len() / 16;
    println!("[1] {} frames of audio codes loaded ({} bytes)", n_frames, code_bytes.len());

    // Step 2: Convert each frame to AudioFrame → decode via iMDCT → PCM
    let t0 = Instant::now();
    let mut all_pcm: Vec<f32> = Vec::with_capacity(n_frames * 960);

    for frame_idx in 0..n_frames {
        let offset = frame_idx * 16;
        let mut codes = [0u8; 16];
        codes.copy_from_slice(&code_bytes[offset..offset + 16]);

        // Build AudioFrame from cascade codes
        let audio_frame = codes_to_audio_frame(&codes);

        // Decode: band energies → flat spectral shape → iMDCT → PCM
        let pcm = audio_frame.decode_coarse();
        all_pcm.extend_from_slice(&pcm);
    }

    // Normalize and convert to i16
    let peak = all_pcm.iter().map(|s| s.abs()).fold(0.0f32, f32::max).max(1e-10);
    let scale = 30000.0 / peak; // leave headroom
    let pcm_i16: Vec<i16> = all_pcm.iter()
        .map(|&s| (s * scale).clamp(-32768.0, 32767.0) as i16)
        .collect();

    println!("[2] Decoded {} frames → {} samples in {:?}",
        n_frames, pcm_i16.len(), t0.elapsed());
    println!("    Duration: {:.2}s at {}Hz (MDCT frame size, not padded to 24kHz)",
        pcm_i16.len() as f64 / SAMPLE_RATE as f64, SAMPLE_RATE);
    println!("    Peak amplitude: {:.4}, scale factor: {:.1}", peak, scale);

    // Step 3: Show AudioFrame stats
    println!("\n[3] AudioFrame analysis:");
    if n_frames > 0 {
        let mut codes = [0u8; 16];
        codes.copy_from_slice(&code_bytes[0..16]);
        let frame0 = codes_to_audio_frame(&codes);
        let e = bands::bf16_to_energies(&frame0.band_energies);
        println!("    Frame 0 band energies (first 5): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
            e[0], e[1], e[2], e[3], e[4]);
        println!("    Frame 0 PVQ summary: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
            frame0.pvq_summary[0], frame0.pvq_summary[1],
            frame0.pvq_summary[2], frame0.pvq_summary[3],
            frame0.pvq_summary[4], frame0.pvq_summary[5]);

        // HHTL routing info from PVQ summary
        println!("    HEEL sign pattern: hi={:08b} lo={:08b}", frame0.pvq_summary[0], frame0.pvq_summary[1]);
        println!("    TWIG harmonic: {:02x}, texture: {:02x}", frame0.pvq_summary[4], frame0.pvq_summary[5]);
    }

    // Step 4: Write WAV
    write_wav(WAV_PATH, &pcm_i16, SAMPLE_RATE);
    let wav_size = std::fs::metadata(WAV_PATH).map(|m| m.len()).unwrap_or(0);
    println!("\n[4] WAV written: {} ({} bytes, {:.1} KB)", WAV_PATH, wav_size, wav_size as f64 / 1024.0);

    println!("\n═══ DONE ═══");
}
