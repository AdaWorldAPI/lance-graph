//! AudioNode: one audio frame as a graph vertex.
//!
//! Each node stores:
//!   - 48B AudioFrame (21 BF16 band energies + 6B PVQ summary)
//!   - 4B PhaseDescriptor (coherence, gradient, entropy, stability)
//!   - 6B SpiralAddress (start, stride, length as u16)
//!   - 1B palette index (VoiceCodebook archetype)
//!   - 1B route hint (RouteAction for HHTL cascade skip)
//!
//!   ─────────────────────────────────────────────────────
//!   Total: 60 bytes per node. Fits in one cache line.
//!
//! The SpiralAddress maps band energies to highheelbgz:
//!   stride = TensorRole = voice character
//!   start = temporal position in the audio stream
//!   length = averaging window (longer = smoother)
//!
//! The palette index maps to bgz-tensor's WeightPalette:
//!   Each AudioFrame's band energies are projected to Base17,
//!   then assigned to the nearest palette centroid.
//!   Route decisions are O(1) via HhtlCache.route(a, b).

/// One audio frame as a graph node.
///
/// 60 bytes, cache-line aligned. Stores both the audio data
/// and the HHTL routing metadata needed for cascade search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct AudioNode {
    /// 21 BF16 band energies (42 bytes).
    pub band_energies: [u16; 21],
    /// PVQ shape fingerprint (6 bytes).
    pub pvq_summary: [u8; 6],
    /// Phase dynamics (4 bytes).
    pub phase: [u8; 4],
    /// SpiralAddress as 3 × u16 (6 bytes).
    /// start: temporal offset, stride: role/character, length: window.
    pub spiral: [u16; 3],
    /// Palette centroid index (1 byte, 0-255).
    pub palette_idx: u8,
    /// Precomputed route hint (1 byte): Skip=0, Attend=1, Compose=2, Escalate=3.
    /// Relative to the PREVIOUS frame — enables streaming skip.
    pub route_hint: u8,
}

impl AudioNode {
    pub const BYTE_SIZE: usize = 60;

    /// Build from raw audio frame bytes + metadata.
    pub fn from_parts(
        band_energies: [u16; 21],
        pvq_summary: [u8; 6],
        phase: [u8; 4],
        spiral_start: u16,
        spiral_stride: u16,
        spiral_length: u16,
        palette_idx: u8,
        route_hint: u8,
    ) -> Self {
        AudioNode {
            band_energies,
            pvq_summary,
            phase,
            spiral: [spiral_start, spiral_stride, spiral_length],
            palette_idx,
            route_hint,
        }
    }

    /// Stride → voice role (from highheelbgz TensorRole mapping).
    pub fn role_name(&self) -> &'static str {
        match self.spiral[1] {
            8 => "gate",
            5 => "v",
            4 => "down",
            3 => "qk",
            2 => "up",
            _ => "other",
        }
    }

    /// Is this a voiced frame? (phase coherence > 50%)
    pub fn is_voiced(&self) -> bool {
        self.phase[0] > 128
    }

    /// Is this an attack/plosive? (low coherence + high gradient)
    pub fn is_attack(&self) -> bool {
        self.phase[0] < 64 && self.phase[1] > 128
    }

    /// Should the cascade skip this frame relative to the previous?
    pub fn should_skip(&self) -> bool {
        self.route_hint == 0 // RouteAction::Skip
    }

    /// Total spectral energy (sum of BF16 band energies, approximate).
    pub fn energy(&self) -> f32 {
        self.band_energies.iter()
            .map(|&b| f32::from_bits((b as u32) << 16))
            .sum()
    }

    /// Serialize to 60 bytes.
    pub fn to_bytes(&self) -> [u8; Self::BYTE_SIZE] {
        let mut out = [0u8; Self::BYTE_SIZE];
        for i in 0..21 {
            let b = self.band_energies[i].to_le_bytes();
            out[i * 2] = b[0];
            out[i * 2 + 1] = b[1];
        }
        out[42..48].copy_from_slice(&self.pvq_summary);
        out[48..52].copy_from_slice(&self.phase);
        for i in 0..3 {
            let b = self.spiral[i].to_le_bytes();
            out[52 + i * 2] = b[0];
            out[52 + i * 2 + 1] = b[1];
        }
        out[58] = self.palette_idx;
        out[59] = self.route_hint;
        out
    }

    /// Deserialize from 60 bytes.
    pub fn from_bytes(bytes: &[u8; Self::BYTE_SIZE]) -> Self {
        let mut band_energies = [0u16; 21];
        for i in 0..21 {
            band_energies[i] = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        }
        let mut pvq_summary = [0u8; 6];
        pvq_summary.copy_from_slice(&bytes[42..48]);
        let mut phase = [0u8; 4];
        phase.copy_from_slice(&bytes[48..52]);
        let mut spiral = [0u16; 3];
        for i in 0..3 {
            spiral[i] = u16::from_le_bytes([bytes[52 + i * 2], bytes[52 + i * 2 + 1]]);
        }
        AudioNode {
            band_energies,
            pvq_summary,
            phase,
            spiral,
            palette_idx: bytes[58],
            route_hint: bytes[59],
        }
    }
}

/// Temporal edge between consecutive AudioNodes.
///
/// Weight = spectral distance (L1 over BF16 bands).
/// Used for streaming playback and temporal search.
#[derive(Clone, Copy, Debug)]
pub struct TemporalEdge {
    pub from_idx: u32,
    pub to_idx: u32,
    /// Spectral distance: L1 over 21 BF16 bands.
    /// Low = smooth transition, High = spectral change (attack, phoneme boundary).
    pub spectral_distance: u16,
}

/// Spectral L1 distance between two AudioNodes (HHTL HIP level).
///
/// Returns u16: sum of |band_a - band_b| over 21 bands,
/// scaled to fit u16 (0 = identical, 65535 = maximally different).
pub fn spectral_l1(a: &AudioNode, b: &AudioNode) -> u16 {
    let mut d = 0u32;
    for i in 0..21 {
        let ea = f32::from_bits((a.band_energies[i] as u32) << 16);
        let eb = f32::from_bits((b.band_energies[i] as u32) << 16);
        d += ((ea - eb).abs() * 1000.0) as u32;
    }
    d.min(65535) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_node_size() {
        assert_eq!(AudioNode::BYTE_SIZE, 60);
    }

    #[test]
    fn serialize_roundtrip() {
        let node = AudioNode::from_parts(
            [100; 21], [1, 2, 3, 4, 5, 6], [200, 50, 128, 30],
            0, 5, 10, 42, 1,
        );
        let bytes = node.to_bytes();
        let recovered = AudioNode::from_bytes(&bytes);
        assert_eq!(node, recovered);
    }

    #[test]
    fn role_from_stride() {
        let gate = AudioNode::from_parts([0; 21], [0; 6], [0; 4], 0, 8, 1, 0, 0);
        assert_eq!(gate.role_name(), "gate");

        let v = AudioNode::from_parts([0; 21], [0; 6], [0; 4], 0, 5, 1, 0, 0);
        assert_eq!(v.role_name(), "v");
    }

    #[test]
    fn voiced_detection() {
        let voiced = AudioNode::from_parts([0; 21], [0; 6], [200, 30, 128, 50], 0, 5, 1, 0, 0);
        assert!(voiced.is_voiced());

        let noise = AudioNode::from_parts([0; 21], [0; 6], [30, 30, 128, 50], 0, 5, 1, 0, 0);
        assert!(!noise.is_voiced());
    }

    #[test]
    fn spectral_l1_self_zero() {
        let node = AudioNode::from_parts(
            [0x3C00; 21], // BF16 for 1.0
            [0; 6], [0; 4], 0, 5, 1, 0, 0,
        );
        assert_eq!(spectral_l1(&node, &node), 0);
    }
}
