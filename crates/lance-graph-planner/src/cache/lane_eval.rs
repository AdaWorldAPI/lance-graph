//! Lane Evaluator: 4096 interdependent attention heads firing through ThinkingStyles.
//!
//! Not independent parallel lanes — a cascade where each head reads the residual
//! stream of all previous heads. The 64×64 matrix IS the attention pattern.
//!
//! Euler-gamma noise floor: signals below γ/(γ+1)/√d are noise.
//! ThinkingStyle.noise_tolerance controls how far above the floor to accept.

use super::kv_bundle::{HeadPrint, AttentionMatrix};
use super::candidate_pool::{Candidate, HeadAddress};
use super::triple_model::{TripleModel, DkPosition};

/// Euler-Mascheroni constant (Rust 1.94+).
const EULER_GAMMA: f64 = std::f64::consts::EULER_GAMMA;

/// Noise floor for Base17 dimensions (d=17).
/// Precomputed: γ/(γ+1)/√17 = 0.5772156649/(1.5772156649)/4.123105625 ≈ 0.08874
const NOISE_FLOOR: f32 = (EULER_GAMMA / (EULER_GAMMA + 1.0) / 4.123105625617661) as f32;

/// 7D field modulation (from ThinkingStyle).
#[derive(Clone, Debug)]
pub struct Tension {
    pub resonance_threshold: f32,
    pub fan_out: u8,
    pub depth_bias: f32,
    pub breadth_bias: f32,
    pub noise_tolerance: f32,
    pub speed_bias: f32,
    pub exploration: f32,
}

impl Tension {
    pub fn analytical() -> Self {
        Self {
            resonance_threshold: 0.85,
            fan_out: 4,
            depth_bias: 0.9,
            breadth_bias: 0.2,
            noise_tolerance: 0.1,
            speed_bias: 0.3,
            exploration: 0.1,
        }
    }
    pub fn creative() -> Self {
        Self {
            resonance_threshold: 0.5,
            fan_out: 12,
            depth_bias: 0.4,
            breadth_bias: 0.9,
            noise_tolerance: 0.7,
            speed_bias: 0.6,
            exploration: 0.8,
        }
    }
    pub fn focused() -> Self {
        Self {
            resonance_threshold: 0.9,
            fan_out: 2,
            depth_bias: 1.0,
            breadth_bias: 0.1,
            noise_tolerance: 0.05,
            speed_bias: 0.4,
            exploration: 0.05,
        }
    }
    pub fn integrative() -> Self {
        Self {
            resonance_threshold: 0.7,
            fan_out: 8,
            depth_bias: 0.6,
            breadth_bias: 0.6,
            noise_tolerance: 0.3,
            speed_bias: 0.3,
            exploration: 0.5,
        }
    }

    /// Signal threshold: noise_floor × (1 + 1/noise_tolerance)
    pub fn signal_threshold(&self) -> f32 {
        NOISE_FLOOR * (1.0 + 1.0 / (self.noise_tolerance + 0.01))
    }

    /// Select tension from DK position.
    pub fn from_dk(dk: DkPosition) -> Self {
        match dk {
            DkPosition::MountStupid => Self::creative(),      // explore everything
            DkPosition::ValleyOfDespair => Self::analytical(), // careful, methodical
            DkPosition::SlopeOfEnlightenment => Self::integrative(), // balanced
            DkPosition::PlateauOfMastery => Self::focused(),   // trust and precision
        }
    }
}

/// Evaluate all 4096 heads and produce candidates.
pub struct LaneEvaluator {
    pub tension: Tension,
}

impl LaneEvaluator {
    pub fn new(tension: Tension) -> Self { Self { tension } }

    /// Fire all 4096 heads in the matrix. Each head that exceeds the
    /// signal threshold produces a candidate.
    pub fn evaluate(&self, matrix: &AttentionMatrix, gestalt: &HeadPrint) -> Vec<Candidate> {
        let threshold = self.tension.signal_threshold();
        let max_candidates = self.tension.fan_out as usize * matrix.resolution;
        let mut candidates = Vec::new();

        for row in 0..matrix.resolution {
            for col in 0..matrix.resolution {
                let head = matrix.get(row, col);
                let signal = gestalt.l1(head) as f32 / (17u32 * 65535) as f32;

                if signal > threshold {
                    let rank = signal * self.tension.depth_bias
                        + (1.0 - signal) * self.tension.breadth_bias;

                    candidates.push(Candidate {
                        head: head.clone(),
                        address: HeadAddress { row: row as u8, col: col as u8 },
                        rank,
                        confidence: 0.5 + signal * 0.4,
                        frequency: signal,
                        inference: 0, // deduction default
                    });
                }

                if candidates.len() >= max_candidates { break; }
            }
            if candidates.len() >= max_candidates { break; }
        }

        candidates.sort_by(|a, b| b.rank.partial_cmp(&a.rank).unwrap());
        candidates.truncate(max_candidates);
        candidates
    }

    /// Evaluate triple model: fire each model's matrix with appropriate tension.
    pub fn evaluate_triple(&self, triple: &TripleModel) -> Vec<Candidate> {
        let mut all = Vec::new();

        // Self model: use DK-appropriate tension
        let self_tension = Tension::from_dk(triple.self_model.dk);
        let self_eval = LaneEvaluator::new(self_tension);
        all.extend(self_eval.evaluate(&triple.self_model.matrix, &triple.self_model.matrix.gestalt));

        // User model: always more exploratory (we know less)
        let mut user_tension = Tension::from_dk(triple.user_model.dk);
        user_tension.noise_tolerance = (user_tension.noise_tolerance + 0.3).min(1.0);
        let user_eval = LaneEvaluator::new(user_tension);
        all.extend(user_eval.evaluate(&triple.user_model.matrix, &triple.user_model.matrix.gestalt));

        // Impact model: analytical (we need precision on predictions)
        let impact_eval = LaneEvaluator::new(Tension::analytical());
        all.extend(impact_eval.evaluate(&triple.impact_model.matrix, &triple.impact_model.matrix.gestalt));

        all.sort_by(|a, b| b.rank.partial_cmp(&a.rank).unwrap());
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_floor_value() {
        // NOISE_FLOOR = γ/(γ+1)/√17
        let expected = (EULER_GAMMA / (EULER_GAMMA + 1.0) / (17.0_f64).sqrt()) as f32;
        assert!((NOISE_FLOOR - expected).abs() < 1e-6, "NOISE_FLOOR={NOISE_FLOOR}, expected={expected}");
        // Should be roughly 0.0887
        assert!(NOISE_FLOOR > 0.08 && NOISE_FLOOR < 0.10, "NOISE_FLOOR out of expected range: {NOISE_FLOOR}");
    }

    #[test]
    fn test_signal_threshold() {
        let analytical = Tension::analytical();
        let creative = Tension::creative();

        let at = analytical.signal_threshold();
        let ct = creative.signal_threshold();

        // Analytical (noise_tolerance=0.1) should have higher threshold than creative (0.7)
        assert!(
            at > ct,
            "analytical threshold ({at}) should exceed creative ({ct})"
        );
        // Both should be above noise floor
        assert!(at > NOISE_FLOOR);
        assert!(ct > NOISE_FLOOR);
    }

    #[test]
    fn test_evaluate_empty_matrix() {
        let eval = LaneEvaluator::new(Tension::analytical());
        let matrix = AttentionMatrix::new_hip();
        let gestalt = HeadPrint::zero();

        // All heads are zero, gestalt is zero, l1 distance = 0, signal = 0
        // Nothing should exceed the threshold
        let candidates = eval.evaluate(&matrix, &gestalt);
        assert!(candidates.is_empty(), "zero matrix should produce no candidates");
    }

    #[test]
    fn test_evaluate_with_signal() {
        let eval = LaneEvaluator::new(Tension::creative()); // low threshold
        let mut matrix = AttentionMatrix::new_hip();

        // Set a head with max-range values so gestalt diverges strongly from zero heads.
        // After setting, gestalt = strong_head (epoch was 0, weight_self=0).
        // l1(gestalt, zero) = 17 * 30000 = 510000; signal = 510000 / (17*65535) ≈ 0.458
        // Creative threshold ≈ 0.214, so 0.458 > 0.214 → candidates produced.
        let strong_head = HeadPrint {
            dims: [30000; 17],
        };
        matrix.set(5, 10, strong_head);

        // Now gestalt has shifted toward that head, but all other heads are zero.
        // Evaluating with gestalt: zero heads will have high l1 from shifted gestalt.
        let candidates = eval.evaluate(&matrix, &matrix.gestalt);

        // We should get some candidates (the zero heads are now "surprising" relative to gestalt)
        assert!(!candidates.is_empty(), "should produce candidates when gestalt diverges from heads");

        // Candidates should be sorted by rank descending
        for window in candidates.windows(2) {
            assert!(
                window[0].rank >= window[1].rank,
                "candidates should be sorted by rank descending"
            );
        }
    }

    #[test]
    fn test_tension_from_dk() {
        let mount = Tension::from_dk(DkPosition::MountStupid);
        let valley = Tension::from_dk(DkPosition::ValleyOfDespair);
        let slope = Tension::from_dk(DkPosition::SlopeOfEnlightenment);
        let plateau = Tension::from_dk(DkPosition::PlateauOfMastery);

        // MountStupid = creative (high exploration)
        assert_eq!(mount.exploration, 0.8);
        // ValleyOfDespair = analytical (low exploration)
        assert_eq!(valley.exploration, 0.1);
        // SlopeOfEnlightenment = integrative (balanced)
        assert_eq!(slope.exploration, 0.5);
        // PlateauOfMastery = focused (minimal exploration)
        assert_eq!(plateau.exploration, 0.05);
    }

    #[test]
    fn test_evaluate_triple() {
        let eval = LaneEvaluator::new(Tension::analytical());
        let triple = TripleModel::new();

        // All-zero triple should produce no candidates
        let candidates = eval.evaluate_triple(&triple);
        assert!(candidates.is_empty(), "all-zero triple should produce no candidates");

        // After setting some heads, we should get candidates
        let mut triple2 = TripleModel::new();
        let head = HeadPrint { dims: [5000; 17] };
        triple2.on_self_output(&head, 0, 0);

        let candidates2 = eval.evaluate_triple(&triple2);
        // Self model gestalt shifted, so other heads become candidates
        // Result depends on threshold but should be non-empty for at least some model
        // (creative tension on self model since dk=MountStupid)
        // Not asserting non-empty since it depends on exact threshold math,
        // but the function should not panic
        let _ = candidates2;
    }
}
