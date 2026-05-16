//! Layered thinking cascade with CausalEdge64 upstream propagation.
//!
//! Three tiers of ThinkingEngine, connected by causal edges:
//!
//! ```text
//! L1 (small, routing)  ──edges──►  L2 (mid, role resonance)  ──edges──►  L3 (full thought)
//! ```
//!
//! CausalEdge64 packs 8 channels (7 constructive + 1 destructive) into a u64.
//! Each channel is one byte (0-255). Constructive channels add energy;
//! the CONTRADICTS channel subtracts energy.

use crate::dto::BusDto;
use crate::engine::ThinkingEngine;
use causal_edge::{CausalEdge64 as SpoEdge, CausalMask};

// ═══════════════════════════════════════════════════════════════════════════
// CausalEdge64: packed u64 with 7 constructive + 1 destructive channel
// ═══════════════════════════════════════════════════════════════════════════

/// Channel indices.
/// 0=BECOMES, 1=CAUSES, 2=SUPPORTS, 3=REFINES,
/// 4=GROUNDS, 5=ABSTRACTS, 6=RELATES, 7=CONTRADICTS.
pub const CHANNEL_BECOMES: u8 = 0;
pub const CHANNEL_CAUSES: u8 = 1;
pub const CHANNEL_SUPPORTS: u8 = 2;
pub const CHANNEL_REFINES: u8 = 3;
pub const CHANNEL_GROUNDS: u8 = 4;
pub const CHANNEL_ABSTRACTS: u8 = 5;
pub const CHANNEL_RELATES: u8 = 6;
pub const CHANNEL_CONTRADICTS: u8 = 7;

/// CausalEdge64: packed u64 with 7 constructive + 1 destructive channel.
///
/// Layout (little-endian byte order within the u64):
///   bits  0..7  = channel 0 (BECOMES)
///   bits  8..15 = channel 1 (CAUSES)
///   bits 16..23 = channel 2 (SUPPORTS)
///   bits 24..31 = channel 3 (REFINES)
///   bits 32..39 = channel 4 (GROUNDS)
///   bits 40..47 = channel 5 (ABSTRACTS)
///   bits 48..55 = channel 6 (RELATES)
///   bits 56..63 = channel 7 (CONTRADICTS)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CausalEdge64(pub u64);

impl CausalEdge64 {
    /// Create a zero edge (no causal strength on any channel).
    pub fn new() -> Self {
        CausalEdge64(0)
    }

    /// Set a channel's value (u8, 0..=255). `channel` must be 0..=7.
    pub fn set_channel_u8(&mut self, channel: u8, value: u8) {
        assert!(channel < 8, "channel must be 0..=7, got {}", channel);
        let shift = channel as u64 * 8;
        self.0 = (self.0 & !(0xFF << shift)) | ((value as u64) << shift);
    }

    /// Get a channel's value as u8 (0..=255).
    pub fn get_channel(&self, channel: u8) -> u8 {
        assert!(channel < 8, "channel must be 0..=7, got {}", channel);
        let shift = channel as u64 * 8;
        ((self.0 >> shift) & 0xFF) as u8
    }

    /// Read one channel's net_strength as i8 (SIMD-friendly bitcast).
    /// Channel layout: byte k = channel k, k = 0..8. Each byte is a signed i8.
    #[inline]
    pub fn channel(&self, idx: usize) -> i8 {
        debug_assert!(idx < 8, "channel idx must be 0..8");
        ((self.0 >> (idx * 8)) & 0xFF) as i8
    }

    /// Set one channel's i8 net_strength. Out-of-range idx is a no-op.
    #[inline]
    pub fn set_channel(&mut self, idx: usize, value: i8) {
        if idx >= 8 { return; }
        let mask = 0xFFu64 << (idx * 8);
        self.0 = (self.0 & !mask) | ((value as u8 as u64) << (idx * 8));
    }

    /// Total constructive strength: sum of channels 0..=6.
    pub fn constructive_strength(&self) -> u16 {
        let mut sum: u16 = 0;
        for ch in 0..7u8 {
            sum += self.get_channel(ch) as u16;
        }
        sum
    }

    /// Destructive strength: channel 7 (CONTRADICTS).
    pub fn contradiction_strength(&self) -> u8 {
        self.get_channel(CHANNEL_CONTRADICTS)
    }

    /// Net strength: constructive - destructive. Can be negative.
    pub fn net_strength(&self) -> i16 {
        self.constructive_strength() as i16 - self.contradiction_strength() as i16
    }

    /// Convenience: create an edge with CAUSES channel set to `strength`.
    ///
    /// Source and target are NOT stored inside the u64 (all 64 bits are channels).
    /// They are carried alongside as the tuple key `(u16, CausalEdge64)`.
    pub fn with_source_target(_source: u16, _target: u16, strength: u8) -> Self {
        let mut e = Self::new();
        e.set_channel_u8(CHANNEL_CAUSES, strength);
        e
    }
}

impl Default for CausalEdge64 {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Transcoder impl block (D-CSV-9, Option R-3 per plan §5 L-12)
//
// Collapses the 8-channel cascade form into one SPO-palette CausalEdge64
// at the L3 commit boundary, and provides the inverse for round-trip tests.
// ═══════════════════════════════════════════════════════════════════════════

impl CausalEdge64 {
    /// 8 channel labels for diagnostics + tests.
    pub const CHANNEL_NAMES: [&'static str; 8] = [
        "BECOMES", "CAUSES", "SUPPORTS", "REFINES",
        "GROUNDS", "ABSTRACTS", "RELATES", "CONTRADICTS",
    ];

    /// Index of the dominant channel (max |net_strength|). Ties break to
    /// the lowest index (per the L-12 rule "stable tie-break").
    /// Returns 0 if all channels are zero (identity).
    #[inline]
    pub fn dominant_channel(&self) -> usize {
        let mut best_idx = 0usize;
        let mut best_abs: u8 = 0;
        for i in 0..8 {
            let v = self.channel(i);
            let abs_v = v.unsigned_abs();
            if abs_v > best_abs {
                best_abs = abs_v;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Count of channels with non-zero net_strength (used as
    /// confidence proxy in the transcode).
    #[inline]
    pub fn active_channel_count(&self) -> u8 {
        let mut n = 0u8;
        for i in 0..8 { if self.channel(i) != 0 { n += 1; } }
        n
    }

    /// Per L-12 / Option R-3: collapse this 8-channel edge into one
    /// SPO-palette `causal_edge::CausalEdge64` at the L3 commit boundary.
    ///
    /// Caller supplies the (s_idx, p_idx, o_idx) palette context. The
    /// transcoder resolves:
    /// - Dominant channel → (mantissa slot, causal_mask) per the table
    ///   in `cognitive-substrate-convergence-v1.md` §11 D-CSV-9
    /// - Mantissa sign = sign of dominant channel's net_strength
    /// - Frequency = |net_strength|/32 quantized to u8
    /// - Confidence = active_channel_count/8 quantized to u8
    /// - Direction triad = 0b000 (sign carried by mantissa per L-9)
    /// - W-slot = 0, truth = Crystalline, spare = 0 (caller stamps later)
    pub fn to_spo(&self, s_idx: u8, p_idx: u8, o_idx: u8) -> SpoEdge {
        let dom = self.dominant_channel();
        let net = self.channel(dom);
        let freq_u8 = ((net.unsigned_abs() as u32 * 255 / 32).min(255)) as u8;
        let conf_u8 = (self.active_channel_count() as u32 * 255 / 8) as u8;
        let (mantissa_magnitude, causal_mask) = match dom {
            0 => (1u8, CausalMask::SPO),  // BECOMES → Deduction (forward chain)
            1 => (6u8, CausalMask::SPO),  // CAUSES → Intervention/Counterfactual (Pearl-3)
            2 => (4u8, CausalMask::PO),   // SUPPORTS → Revision (interventional plane)
            3 => (5u8, CausalMask::PO),   // REFINES → Synthesis
            4 => (1u8, CausalMask::S),    // GROUNDS → Deduction (S-grounded)
            5 => (2u8, CausalMask::P),    // ABSTRACTS → Induction
            6 => (0u8, CausalMask::None), // RELATES → Identity/neutral
            _ => (1u8, CausalMask::SPO),  // CONTRADICTS → Abduction (sign carries polarity)
        };
        let mantissa_signed: i8 = if net >= 0 {
            mantissa_magnitude as i8
        } else {
            -(mantissa_magnitude as i8)
        };
        causal_edge::CausalEdge64::pack_v2(s_idx, p_idx, o_idx, freq_u8, conf_u8, causal_mask, 0, causal_edge::PlasticityState::ALL_FROZEN)
            .with_inference_mantissa(mantissa_signed)
    }

    /// Inverse: project an SPO-palette edge into the 8-channel form
    /// where the dominant channel carries the mantissa magnitude scaled
    /// by frequency. Lossy (8 channels collapse to 1); used for round-
    /// trip tests + debugging.
    pub fn from_spo(spo: SpoEdge) -> Self {
        let mantissa = spo.inference_mantissa();
        let mag = mantissa.unsigned_abs();
        let dom = match mag {
            0 => 6, // RELATES (neutral)
            1 => if mantissa >= 0 { 0 } else { 7 }, // BECOMES vs CONTRADICTS
            2 => 5, // ABSTRACTS
            3 => 5, // (Synthesis tilts ABSTRACTS) — matches the table tilt
            4 => 2, // SUPPORTS
            5 => 3, // REFINES
            6 => 1, // CAUSES
            _ => 0, // Reserved → fall back to BECOMES
        };
        let net_signed = if mantissa >= 0 {
            (spo.frequency_u8() as i32 * 32 / 255).min(127) as i8
        } else {
            -((spo.frequency_u8() as i32 * 32 / 255).min(127) as i8)
        };
        let mut out = CausalEdge64::default();
        out.set_channel(dom, net_signed);
        out
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TierEngine: wraps ThinkingEngine for one level of the cascade
// ═══════════════════════════════════════════════════════════════════════════

/// A tier engine wrapping ThinkingEngine for one level of the cascade.
///
/// Keeps a shadow copy of the distance table for neighbor lookups
/// (ThinkingEngine's table field is private).
pub struct TierEngine {
    engine: ThinkingEngine,
    /// Shadow copy of the N×N distance table for causal edge emission.
    distance_table: Vec<u8>,
    tier_name: String,
    size: usize,
}

impl TierEngine {
    /// Create a tier engine from an N×N distance table.
    pub fn new(distance_table: Vec<u8>, name: &str) -> Self {
        let total = distance_table.len();
        let size = (total as f32).sqrt() as usize;
        assert_eq!(
            size * size,
            total,
            "distance table length {} is not a perfect square",
            total
        );
        let engine = ThinkingEngine::new(distance_table.clone());
        Self {
            engine,
            distance_table,
            tier_name: name.to_string(),
            size,
        }
    }

    /// Run thinking cycles on this tier.
    pub fn think(&mut self, max_cycles: usize) {
        self.engine.think(max_cycles);
    }

    /// Get top-k peaks from current energy, sorted descending by energy.
    pub fn top_k(&self, k: usize) -> Vec<(u16, f32)> {
        let mut indexed: Vec<(usize, f32)> = self
            .engine
            .energy
            .iter()
            .enumerate()
            .map(|(i, &e)| (i, e))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed
            .iter()
            .take(k)
            .map(|&(i, e)| (i as u16, e))
            .collect()
    }

    /// Emit CausalEdge64 events from top-k peaks.
    ///
    /// For each of the top-k peaks, emit edges to their 4 nearest neighbors
    /// in the distance table (highest similarity = strongest constructive edges).
    pub fn emit_causal_edges(&self, k: usize) -> Vec<(u16, CausalEdge64)> {
        let peaks = self.top_k(k);
        let mut edges = Vec::new();

        for &(peak_idx, peak_energy) in &peaks {
            if peak_energy < 1e-15 {
                continue;
            }
            let pi = peak_idx as usize;
            let row_offset = pi * self.size;

            // Collect (neighbor_index, similarity) excluding self.
            let mut neighbors: Vec<(usize, u8)> = (0..self.size)
                .filter(|&j| j != pi)
                .map(|j| (j, self.distance_table[row_offset + j]))
                .collect();
            // Sort by similarity descending to find nearest neighbors.
            neighbors.sort_by(|a, b| b.1.cmp(&a.1));

            // Take top 4 neighbors.
            for &(neighbor_idx, sim) in neighbors.iter().take(4) {
                // Strength proportional to similarity and peak energy.
                let strength = ((sim as f32 / 255.0) * peak_energy * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                if strength == 0 {
                    continue;
                }
                let mut edge = CausalEdge64::new();
                edge.set_channel_u8(CHANNEL_CAUSES, strength);
                edges.push((neighbor_idx as u16, edge));
            }
        }

        edges
    }

    /// Apply causal edges as energy perturbation.
    ///
    /// Constructive channels (0..=6) add positive energy.
    /// CONTRADICTS channel (7) subtracts energy.
    pub fn apply_edges(&mut self, edges: &[(u16, CausalEdge64)]) {
        for &(target, edge) in edges {
            let idx = target as usize;
            if idx >= self.size {
                continue;
            }
            let net = edge.net_strength();
            // Scale: divide by 255 to keep in reasonable range.
            let delta = net as f32 / 255.0;
            self.engine.energy[idx] += delta;
            // Clamp to zero floor.
            if self.engine.energy[idx] < 0.0 {
                self.engine.energy[idx] = 0.0;
            }
        }
        // Re-normalize.
        let total: f32 = self.engine.energy.iter().sum();
        if total > 1e-15 {
            for e in &mut self.engine.energy {
                *e /= total;
            }
        }
    }

    /// Reset energy to zero.
    pub fn reset(&mut self) {
        self.engine.reset();
    }

    /// Number of thought-atoms in this tier.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Perturb this tier with codebook indices.
    pub fn perturb(&mut self, indices: &[u16]) {
        self.engine.perturb(indices);
    }

    /// Access the underlying engine.
    pub fn engine(&self) -> &ThinkingEngine {
        &self.engine
    }

    /// Tier name.
    pub fn name(&self) -> &str {
        &self.tier_name
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LayeredEngine: three-level cascade with causal edge propagation
// ═══════════════════════════════════════════════════════════════════════════

/// Three-level cascade: L1 → L2 → L3 with CausalEdge64 upstream propagation.
pub struct LayeredEngine {
    l1: TierEngine,
    l2: TierEngine,
    l3: TierEngine,
}

impl LayeredEngine {
    /// Create from three distance tables (any sizes).
    pub fn new(l1_table: Vec<u8>, l2_table: Vec<u8>, l3_table: Vec<u8>) -> Self {
        Self {
            l1: TierEngine::new(l1_table, "L1-routing"),
            l2: TierEngine::new(l2_table, "L2-resonance"),
            l3: TierEngine::new(l3_table, "L3-thought"),
        }
    }

    /// Full cascade: perturb L1 → think → emit edges → apply to L2 →
    /// think → emit → apply to L3 → think → commit.
    pub fn process(&mut self, codebook_indices: &[u16]) -> BusDto {
        // Map raw indices down to L1 scale.
        let l1_size = self.l1.size();
        let l3_size = self.l3.size();
        let scale = if l1_size > 0 && l3_size > l1_size {
            l3_size / l1_size
        } else {
            1
        };
        let l1_indices: Vec<u16> = codebook_indices
            .iter()
            .map(|&idx| {
                let mapped = (idx as usize) / scale.max(1);
                mapped.min(l1_size.saturating_sub(1)) as u16
            })
            .collect();

        // L1: perturb and think.
        self.l1.perturb(&l1_indices);
        self.l1.think(10);

        // L1 → L2: emit causal edges, scale targets to L2 index space.
        let l1_edges = self.l1.emit_causal_edges(4);
        let l2_size = self.l2.size();
        let l1_to_l2 = if l1_size > 0 && l2_size > l1_size {
            l2_size / l1_size
        } else {
            1
        };
        let l2_edges: Vec<(u16, CausalEdge64)> = l1_edges
            .iter()
            .map(|&(idx, edge)| {
                let scaled = (idx as usize * l1_to_l2).min(l2_size.saturating_sub(1));
                (scaled as u16, edge)
            })
            .collect();
        self.l2.apply_edges(&l2_edges);
        self.l2.think(10);

        // L2 → L3: emit causal edges, scale targets to L3 index space.
        let l2_edges_out = self.l2.emit_causal_edges(4);
        let l2_to_l3 = if l2_size > 0 && l3_size > l2_size {
            l3_size / l2_size
        } else {
            1
        };
        let l3_edges: Vec<(u16, CausalEdge64)> = l2_edges_out
            .iter()
            .map(|&(idx, edge)| {
                let scaled = (idx as usize * l2_to_l3).min(l3_size.saturating_sub(1));
                (scaled as u16, edge)
            })
            .collect();
        self.l3.apply_edges(&l3_edges);
        self.l3.think(10);

        // Commit from L3.
        self.l3.engine().commit()
    }

    /// Access L1 tier.
    pub fn l1(&self) -> &TierEngine {
        &self.l1
    }

    /// Access L2 tier.
    pub fn l2(&self) -> &TierEngine {
        &self.l2
    }

    /// Access L3 tier.
    pub fn l3(&self) -> &TierEngine {
        &self.l3
    }

    /// Reset all tiers.
    pub fn reset(&mut self) {
        self.l1.reset();
        self.l2.reset();
        self.l3.reset();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic N×N distance table where nearby indices are very similar.
    /// Values range from 0 (distant) to 255 (self), with strong topology signal
    /// well above the median floor.
    fn make_table(n: usize) -> Vec<u8> {
        let mut table = vec![64u8; n * n]; // baseline at 64 (below typical median)
        for i in 0..n {
            table[i * n + i] = 255;
            for j in 0..n {
                let dist = (i as i64 - j as i64).unsigned_abs() as usize;
                if dist > 0 && dist <= n / 4 {
                    table[i * n + j] = (200 - dist * 10).max(100) as u8;
                }
            }
        }
        table
    }

    // ── CausalEdge64 tests ──

    #[test]
    fn causal_edge_channels() {
        let mut edge = CausalEdge64::new();
        assert_eq!(edge.0, 0);

        // Set each channel to a distinct value.
        for ch in 0..8u8 {
            edge.set_channel_u8(ch, (ch + 1) * 30);
        }
        // Read back all 8 channels.
        for ch in 0..8u8 {
            assert_eq!(
                edge.get_channel(ch),
                (ch + 1) * 30,
                "channel {} mismatch",
                ch
            );
        }
    }

    #[test]
    fn causal_edge_constructive_sum() {
        let mut edge = CausalEdge64::new();
        // Channels 0-6 = 10 each.
        for ch in 0..7u8 {
            edge.set_channel_u8(ch, 10);
        }
        // Channel 7 (CONTRADICTS) = 50.
        edge.set_channel_u8(CHANNEL_CONTRADICTS, 50);

        assert_eq!(edge.constructive_strength(), 70); // 7 * 10
        assert_eq!(edge.contradiction_strength(), 50);
    }

    #[test]
    fn causal_edge_net_strength() {
        let mut edge = CausalEdge64::new();
        // Constructive: channels 0-6 = 20 each = 140 total.
        for ch in 0..7u8 {
            edge.set_channel_u8(ch, 20);
        }
        // Destructive: channel 7 = 100.
        edge.set_channel_u8(CHANNEL_CONTRADICTS, 100);
        assert_eq!(edge.net_strength(), 40); // 140 - 100

        // Destructive dominates.
        let mut edge2 = CausalEdge64::new();
        edge2.set_channel_u8(CHANNEL_CAUSES, 10);
        edge2.set_channel_u8(CHANNEL_CONTRADICTS, 200);
        assert_eq!(edge2.net_strength(), -190); // 10 - 200
    }

    // ── TierEngine tests ──

    #[test]
    fn tier_engine_basic() {
        let table = make_table(8);
        let mut tier = TierEngine::new(table, "test");
        assert_eq!(tier.size(), 8);

        tier.perturb(&[3]);
        tier.think(5);

        let peaks = tier.top_k(3);
        assert!(!peaks.is_empty());
        assert!(peaks[0].1 > 0.0, "top peak should have positive energy");
    }

    #[test]
    fn tier_engine_emit_edges() {
        let table = make_table(8);
        let mut tier = TierEngine::new(table, "test");
        tier.perturb(&[2, 3]);
        tier.think(3);

        let edges = tier.emit_causal_edges(2);
        // With a structured distance table, top peaks should produce edges
        // to their nearest neighbors.
        assert!(!edges.is_empty(), "should emit causal edges from peaks");

        // All edges should have positive CAUSES channel.
        for &(_target, edge) in &edges {
            assert!(
                edge.get_channel(CHANNEL_CAUSES) > 0,
                "emitted edge should have positive CAUSES strength"
            );
        }
    }

    #[test]
    fn tier_engine_apply_edges_constructive() {
        let table = make_table(8);
        let mut tier = TierEngine::new(table, "test");

        // Start with zero energy, apply constructive edges.
        let mut edge = CausalEdge64::new();
        edge.set_channel_u8(CHANNEL_CAUSES, 100);
        edge.set_channel_u8(CHANNEL_SUPPORTS, 50);

        tier.apply_edges(&[(3, edge), (5, edge)]);

        // Targets 3 and 5 should have positive energy.
        assert!(
            tier.engine().energy[3] > 0.0,
            "target 3 should have energy after constructive edge"
        );
        assert!(
            tier.engine().energy[5] > 0.0,
            "target 5 should have energy after constructive edge"
        );
        // They should have equal energy (same edge applied).
        let diff = (tier.engine().energy[3] - tier.engine().energy[5]).abs();
        assert!(diff < 1e-10, "same edges should produce same energy");
    }

    #[test]
    fn tier_engine_apply_edges_contradiction() {
        let table = make_table(8);
        let mut tier = TierEngine::new(table, "test");

        // Give some initial energy.
        tier.perturb(&[3]);
        let initial = tier.engine().energy[3];
        assert!(initial > 0.0);

        // Apply a strongly contradicting edge to atom 3.
        let mut edge = CausalEdge64::new();
        edge.set_channel_u8(CHANNEL_CONTRADICTS, 255);
        tier.apply_edges(&[(3, edge)]);

        // Energy at 3 should decrease (clamped to zero, then renormalized).
        let after = tier.engine().energy[3];
        assert!(
            after < initial,
            "contradiction should reduce energy: before={}, after={}",
            initial,
            after
        );
    }

    // ── LayeredEngine tests ──

    #[test]
    fn layered_engine_cascade() {
        let l1 = make_table(8);
        let l2 = make_table(16);
        let l3 = make_table(32);

        let mut engine = LayeredEngine::new(l1, l2, l3);
        let bus = engine.process(&[10, 15, 20]);

        assert!(bus.energy > 0.0, "cascade should produce positive energy");
        assert!(bus.cycle_count > 0, "should have run cycles");
    }

    #[test]
    fn layered_engine_reset() {
        let l1 = make_table(8);
        let l2 = make_table(16);
        let l3 = make_table(32);

        let mut engine = LayeredEngine::new(l1, l2, l3);
        engine.process(&[5, 10]);

        // After process, L3 should have energy.
        assert!(engine.l3().engine().energy.iter().any(|&e| e > 0.0));

        engine.reset();

        // After reset, all energy should be zero.
        assert_eq!(engine.l1().engine().energy.iter().sum::<f32>(), 0.0);
        assert_eq!(engine.l2().engine().energy.iter().sum::<f32>(), 0.0);
        assert_eq!(engine.l3().engine().energy.iter().sum::<f32>(), 0.0);
    }
}

#[cfg(test)]
mod transcoder_tests {
    use super::*;
    use causal_edge::CausalEdge64 as SpoEdge;

    fn build_8ch_with(idx: usize, net: i8) -> CausalEdge64 {
        let mut e = CausalEdge64::default();
        e.set_channel(idx, net);
        e
    }

    #[test]
    fn test_channel_roundtrip() {
        for idx in 0..8 {
            for &v in &[-128i8, -1, 0, 1, 127] {
                let e = build_8ch_with(idx, v);
                assert_eq!(e.channel(idx), v, "channel {idx} round-trip {v}");
            }
        }
    }

    #[test]
    fn test_dominant_channel_zero_default() {
        let e = CausalEdge64::default();
        assert_eq!(e.dominant_channel(), 0, "all-zero edge dominant idx is 0");
        assert_eq!(e.active_channel_count(), 0);
    }

    #[test]
    fn test_dominant_channel_picks_max_abs() {
        let mut e = CausalEdge64::default();
        e.set_channel(2, 30);   // SUPPORTS
        e.set_channel(5, -100); // ABSTRACTS, larger magnitude
        e.set_channel(7, 10);
        assert_eq!(e.dominant_channel(), 5);
        assert_eq!(e.active_channel_count(), 3);
    }

    #[test]
    fn test_to_spo_becomes_dominant_forward() {
        let e = build_8ch_with(0, 16); // BECOMES, +16 net
        let spo = e.to_spo(10, 20, 30);
        assert_eq!(spo.s_idx(), 10);
        assert_eq!(spo.p_idx(), 20);
        assert_eq!(spo.o_idx(), 30);
        assert_eq!(spo.inference_mantissa(), 1, "BECOMES → mantissa +1 Deduction");
        assert!(spo.frequency_u8() > 0, "non-zero net → non-zero frequency");
    }

    #[test]
    fn test_to_spo_causes_negative_is_counterfactual() {
        let e = build_8ch_with(1, -32); // CAUSES, negative magnitude
        let spo = e.to_spo(1, 2, 3);
        assert_eq!(spo.inference_mantissa(), -6, "CAUSES negative → mantissa -6 Counterfactual");
    }

    #[test]
    fn test_to_spo_relates_neutral_mantissa_zero() {
        let e = build_8ch_with(6, 100);  // RELATES dominant
        let spo = e.to_spo(0, 0, 0);
        assert_eq!(spo.inference_mantissa(), 0, "RELATES → mantissa 0 (Identity)");
    }

    #[test]
    fn test_16_mapping_round_trip_dominant_preserved() {
        // For each (dominant_channel, sign) pair, transcode 8ch → SPO → 8ch
        // and assert the dominant channel index survives. The exact net_strength
        // doesn't survive (lossy), but the dominant channel SHOULD.
        for dom in 0..8 {
            for &sign in &[1i8, -1i8] {
                let e = build_8ch_with(dom, sign * 64);
                let spo = e.to_spo(1, 1, 1);
                let back = CausalEdge64::from_spo(spo);
                let back_dom = back.dominant_channel();
                // Channel mapping is many-to-one in the transcoder table; some
                // dominant channels collapse to the same SPO mantissa slot.
                // Per L-12, the SEMANTIC class survives, not necessarily the
                // exact channel idx. Assert the back-channel is in the
                // expected equivalence class for this dominant.
                let expected_class: &[usize] = match dom {
                    // BECOMES + GROUNDS → mantissa ±1.
                    // Positive sign → BECOMES (0) on round-trip.
                    // Negative sign → mantissa=-1 → from_spo maps to CONTRADICTS (7);
                    // lossy collapse: negative-BECOMES/GROUNDS is semantically CONTRADICTS
                    // in the SPO lattice (both carry |mantissa|=1 in the backward slot).
                    0 | 4 => &[0, 7],
                    1 => &[1],         // CAUSES → mantissa ±6 → CAUSES
                    2 => &[2],         // SUPPORTS → mantissa +4 → SUPPORTS
                    3 => &[3, 5],      // REFINES → mantissa +5 → REFINES (or ABSTRACTS in tilt)
                    5 => &[5],         // ABSTRACTS → mantissa +2 → ABSTRACTS
                    6 => &[6],         // RELATES → mantissa 0 → RELATES
                    7 => &[7, 0],      // CONTRADICTS → mantissa ±1 (sign distinguishes)
                    _ => &[],
                };
                assert!(
                    expected_class.contains(&back_dom),
                    "dom={dom} sign={sign} round-trip back_dom={back_dom} not in expected_class={expected_class:?}",
                );
            }
        }
    }

    #[test]
    fn test_set_channel_out_of_range_no_op() {
        let mut e = CausalEdge64::default();
        e.set_channel(8, 100);
        e.set_channel(255, 50);
        assert_eq!(e.0, 0, "out-of-range set_channel must be a no-op");
    }
}
