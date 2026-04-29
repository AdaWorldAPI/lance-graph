//! Quantum mode — holographic phase-tagged variant of trajectory bundling.
//!
//! PR #279 outlook E8: same 16384-dim substrate as Crystal mode (Markov SPO
//! bundling), but with a phase-tag field on Trajectory and a 4th
//! WeightingKernel variant `Holographic`. Holographic mode trades structured
//! recoverability for higher-capacity superposition.
//!
//! Crystal vs Quantum is a knob, not a separate stack.
//!
//! META-AGENT: `pub mod quantum_mode;` in deepnsm/lib.rs.

/// 128-bit phase tag for holographic addressing.
/// See ladybug-rs hologram types + cross-repo harvest doc H7.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PhaseTag(pub u128);

impl PhaseTag {
    pub fn pi() -> Self {
        PhaseTag(u128::MAX / 2)
    }

    pub fn from_angle(theta: f32) -> Self {
        // Normalize to [0, 1) first.
        let normalized = ((theta / std::f32::consts::TAU).rem_euclid(1.0)) as f64;
        // f64 has ~15 digits — enough headroom for u64 precision; we cast
        // through u64 (u128::MAX as f32 overflows to infinity).
        let scaled = (normalized * (u64::MAX as f64)) as u128;
        // Place the u64 value in the low half, leave high half zero.
        // For higher resolution, a future PR can fold an additional 64-bit
        // entropy source into the upper half.
        PhaseTag(scaled)
    }

    pub fn to_angle(self) -> f32 {
        // Use the low 64 bits (the high 64 are reserved for future precision).
        let low = (self.0 & u64::MAX as u128) as u64;
        let normalized = (low as f64) / (u64::MAX as f64);
        (normalized * std::f64::consts::TAU as f64) as f32
    }

    pub fn distance(self, other: Self) -> u32 {
        // Hamming on the 128-bit tag = phase distance proxy.
        (self.0 ^ other.0).count_ones()
    }
}

/// Holographic kernel variant. Use this when you want phase-coherent
/// superposition rather than amplitude-bundled accumulation.
#[derive(Debug, Clone, Copy)]
pub enum HolographicMode {
    /// Single-phase carrier — one phase tag per trajectory.
    SinglePhase,
    /// Multi-phase per-role — each role slice carries its own phase.
    /// Future: when role-keys grow phase-tagged variants.
    PerRole,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pi_phase_is_half_max() {
        let p = PhaseTag::pi();
        assert!(p.0 > u128::MAX / 4 && p.0 < 3 * (u128::MAX / 4));
    }

    #[test]
    fn phase_distance_zero_for_self() {
        let p = PhaseTag(12345);
        assert_eq!(p.distance(p), 0);
    }

    #[test]
    fn from_angle_round_trips_approximately() {
        let theta = 1.5f32;
        let p = PhaseTag::from_angle(theta);
        let recovered = p.to_angle();
        // f64 intermediate gives sub-1e-3 round-trip; f32 final cast caps
        // precision around 1e-6 of TAU (~6e-6 absolute).
        let diff = (recovered - theta).abs();
        assert!(diff < 0.001, "round-trip diff {} exceeds tolerance 0.001", diff);
    }

    #[test]
    fn default_is_zero_phase() {
        let p: PhaseTag = Default::default();
        assert_eq!(p.0, 0);
    }

    #[test]
    fn holographic_mode_is_copy() {
        // Smoke test: enum is Copy so we can pass by value freely.
        let m = HolographicMode::SinglePhase;
        let m2 = m;
        let _ = (m, m2);
    }
}
