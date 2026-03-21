//! Strategy C: Hybrid Encoding
//!
//! Both paths simultaneously:
//!   Path 1: per-frame scent → neighborhoods.lance (temporal navigation)
//!   Path 2: streaming accumulator → crystallize → cognitive_nodes.lance (identity)
//!
//! The scent stream IS the P-frame sequence.
//! The crystallized identity IS the I-frame.
//! The Diamond Markov epiphanies ARE the scene changes.

use crate::{AudioFrame, CrystallizedComponent, SpectralAccumulator,
            BARK_BANDS, SAMPLES_PER_FRAME, FRAME_RATE};
use crate::transform::{mdct, coeffs_to_band_energies, psychoacoustic_mask, sine_window};
use crate::bands::{pack_bands, bf16_band_distance};
use crate::perframe::encode_perframe;
use crate::accumulator::decode_crystallized;

/// Hybrid encoding result: per-frame stream + crystallized components.
#[derive(Clone, Debug)]
pub struct HybridEncoding {
    /// Per-frame data (Strategy A): for temporal navigation.
    pub frames: Vec<AudioFrame>,
    /// Crystallized components (Strategy B): for content identity.
    pub components: Vec<CrystallizedComponent>,
    /// Scent bytes: byte 0 of ZeckF64 edge to previous frame.
    /// This is the L1 search index — 1 byte per frame.
    pub scent_stream: Vec<u8>,
    /// ZeckF64 edges to previous frame (full 8 bytes).
    pub edge_stream: Vec<u64>,
}

/// Encode with both strategies simultaneously.
///
/// window_frames: how many frames per accumulator window (e.g., 75 = 1 second).
/// crystallization_threshold: controls quality/bitrate tradeoff.
pub fn encode_hybrid(
    samples: &[f32],
    window_frames: usize,
    crystallization_threshold: i16,
) -> HybridEncoding {
    // Path 1: per-frame encoding
    let frames = encode_perframe(samples);

    // Path 2: streaming accumulator in windows
    let mut components = Vec::new();
    let mut acc = SpectralAccumulator::new();
    let mut window_start = 0u64;

    for (i, frame) in frames.iter().enumerate() {
        acc.accumulate_frame(&frame.bands);

        // End of window: crystallize
        if acc.frame_count as usize >= window_frames {
            let mut component = acc.crystallize(crystallization_threshold);
            component.start_frame = window_start;
            component.end_frame = i as u64;
            components.push(component);

            // Reset for next window
            acc.reset();
            window_start = i as u64 + 1;
        }
    }

    // Final partial window
    if acc.frame_count > 0 {
        let mut component = acc.crystallize(crystallization_threshold);
        component.start_frame = window_start;
        component.end_frame = frames.len() as u64;
        components.push(component);
    }

    // Compute scent and edge streams
    let mut scent_stream = Vec::with_capacity(frames.len());
    let mut edge_stream = Vec::with_capacity(frames.len());
    let mut prev_bands = [0u16; BARK_BANDS];

    for frame in &frames {
        // ZeckF64-like edge: BF16 band distance to previous frame
        let ds = bf16_band_distance(&frame.bands, &prev_bands);
        let dp = bf16_band_distance(&frame.temporal, &[0u16; BARK_BANDS]);
        let d_o = bf16_band_distance(&frame.harmonic, &[0u16; BARK_BANDS]);

        // Pack as simplified ZeckF64 (scent byte)
        let max_d = 1128u32; // max possible bf16_band_distance
        let s_close = (ds < max_d / 2) as u8;
        let p_close = (dp < max_d / 2) as u8;
        let o_close = (d_o < max_d / 2) as u8;
        let sp_close = s_close & p_close;
        let so_close = s_close & o_close;
        let po_close = p_close & o_close;
        let spo_close = sp_close & so_close & po_close;

        let scent = s_close | (p_close << 1) | (o_close << 2) | (sp_close << 3)
            | (so_close << 4) | (po_close << 5) | (spo_close << 6);

        scent_stream.push(scent);

        // Full edge (simplified — real implementation would use ZeckF64 quantiles)
        let edge = (scent as u64)
            | ((quantile(ds, max_d) as u64) << 8)
            | ((quantile(dp, max_d) as u64) << 16)
            | ((quantile(d_o, max_d) as u64) << 24);
        edge_stream.push(edge);

        prev_bands = frame.bands;
    }

    HybridEncoding {
        frames,
        components,
        scent_stream,
        edge_stream,
    }
}

/// Simple linear quantile mapping to [0, 255].
#[inline]
fn quantile(distance: u32, max_distance: u32) -> u8 {
    ((distance as u64 * 255) / max_distance as u64).min(255) as u8
}

/// Decode hybrid: use crystallized components for identity,
/// per-frame data for temporal detail.
pub fn decode_hybrid(encoding: &HybridEncoding) -> Vec<f32> {
    // Use per-frame decoder for full quality reconstruction
    crate::perframe::decode_perframe(&encoding.frames)
}

/// Decode hybrid at scent-only quality (1 byte per frame).
/// Uses crystallized component as the baseline and scent stream
/// to track temporal changes.
pub fn decode_hybrid_scent_only(encoding: &HybridEncoding) -> Vec<f32> {
    crate::perframe::decode_perframe(&encoding.frames)
    // TODO: implement scent-only progressive decode using
    // crystallized component as I-frame and scent as P-frame deltas
}

/// Compute bitrate breakdown for hybrid encoding.
pub fn bitrate_breakdown(encoding: &HybridEncoding, duration_seconds: f32) -> HybridBitrateBreakdown {
    let frame_bits = encoding.frames.len() * BARK_BANDS * 16; // full frames
    let scent_bits = encoding.scent_stream.len() * 8; // 1 byte per frame
    let component_bits = encoding.components.len() * BARK_BANDS * 16; // crystallized
    let edge_bits = encoding.edge_stream.len() * 64; // full ZeckF64 edges

    HybridBitrateBreakdown {
        total_bps: (frame_bits + component_bits) as f64 / duration_seconds as f64,
        frame_bps: frame_bits as f64 / duration_seconds as f64,
        scent_only_bps: scent_bits as f64 / duration_seconds as f64,
        component_bps: component_bits as f64 / duration_seconds as f64,
        edge_bps: edge_bits as f64 / duration_seconds as f64,
        frame_count: encoding.frames.len(),
        component_count: encoding.components.len(),
    }
}

#[derive(Clone, Debug)]
pub struct HybridBitrateBreakdown {
    pub total_bps: f64,
    pub frame_bps: f64,
    pub scent_only_bps: f64,
    pub component_bps: f64,
    pub edge_bps: f64,
    pub frame_count: usize,
    pub component_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_samples(freq_hz: f32, duration_s: f32) -> Vec<f32> {
        let n = (duration_s * 48000.0) as usize;
        (0..n).map(|i| {
            (2.0 * std::f32::consts::PI * freq_hz * i as f32 / 48000.0).sin() * 0.5
        }).collect()
    }

    #[test]
    fn test_hybrid_encode_basic() {
        let samples = sine_samples(440.0, 2.0);
        let encoding = encode_hybrid(&samples, FRAME_RATE as usize, 10);

        assert!(!encoding.frames.is_empty());
        assert!(!encoding.components.is_empty());
        assert_eq!(encoding.scent_stream.len(), encoding.frames.len());
        assert_eq!(encoding.edge_stream.len(), encoding.frames.len());

        // Should have ~2 components (2 seconds / 1 second window)
        assert!(encoding.components.len() >= 1 && encoding.components.len() <= 3,
            "Expected 1-3 components for 2s audio, got {}", encoding.components.len());
    }

    #[test]
    fn test_hybrid_scent_stream_valid() {
        let samples = sine_samples(440.0, 1.0);
        let encoding = encode_hybrid(&samples, FRAME_RATE as usize, 10);

        for &scent in &encoding.scent_stream {
            // Scent byte should have legal lattice pattern (bit 7 = sign, unused)
            let bands = scent & 0x7F;
            let s = bands & 1;
            let p = (bands >> 1) & 1;
            let o = (bands >> 2) & 1;
            let sp = (bands >> 3) & 1;
            let so = (bands >> 4) & 1;
            let po = (bands >> 5) & 1;
            let spo = (bands >> 6) & 1;

            // Lattice: compound implies components
            if sp == 1 { assert!(s == 1 && p == 1, "SP implies S and P"); }
            if so == 1 { assert!(s == 1 && o == 1, "SO implies S and O"); }
            if po == 1 { assert!(p == 1 && o == 1, "PO implies P and O"); }
            if spo == 1 { assert!(sp == 1 && so == 1 && po == 1, "SPO implies all pairs"); }
        }
    }

    #[test]
    fn test_hybrid_bitrate_breakdown() {
        let samples = sine_samples(440.0, 3.0);
        let encoding = encode_hybrid(&samples, FRAME_RATE as usize, 10);
        let breakdown = bitrate_breakdown(&encoding, 3.0);

        println!("Hybrid bitrate breakdown for 3s 440Hz sine:");
        println!("  Total:      {:.0} bps", breakdown.total_bps);
        println!("  Frames:     {:.0} bps ({} frames)", breakdown.frame_bps, breakdown.frame_count);
        println!("  Scent only: {:.0} bps", breakdown.scent_only_bps);
        println!("  Components: {:.0} bps ({} components)", breakdown.component_bps, breakdown.component_count);
        println!("  Edges:      {:.0} bps", breakdown.edge_bps);

        // Scent-only should be much less than full frames
        assert!(breakdown.scent_only_bps < breakdown.frame_bps / 10.0,
            "Scent should be <10% of frame bitrate");
    }
}
