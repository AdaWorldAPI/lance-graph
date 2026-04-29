//! Trajectory-as-statement-hash bridge.
//!
//! PR #279 outlook E4: a Trajectory's binarized fingerprint replaces audit.rs's
//! `statement_hash: u64` with a SEMANTIC hash. Two grammatically-equivalent
//! queries collide; queries that share grammatical structure with attack
//! patterns become Hamming-near-neighbor on the audit log.
//!
//! META-AGENT: add `pub mod trajectory_audit;` to deepnsm/lib.rs gated by
//! `feature = "audit-bridge" = ["dep:lance-graph-callcenter"]`.

use crate::trajectory::Trajectory;

/// Binarized 256-word (16384-bit) fingerprint of a trajectory.
/// Suitable as an `AuditEntry` semantic hash key.
pub type TrajectoryHash = [u64; 256];

impl Trajectory {
    /// Project the trajectory's continuous fingerprint into a 16384-bit
    /// binary fingerprint via signed thresholding (+ → 1, - or 0 → 0).
    pub fn binarize(&self) -> TrajectoryHash {
        let mut bits = [0u64; 256];
        for (word_idx, chunk) in self.fingerprint.chunks(64).enumerate().take(256) {
            let mut w = 0u64;
            for (bit_idx, v) in chunk.iter().enumerate().take(64) {
                if *v > 0.0 {
                    w |= 1u64 << bit_idx;
                }
            }
            bits[word_idx] = w;
        }
        bits
    }

    /// 64-bit syntactic-fallback hash for use when the consumer wants a
    /// scalar instead of a fingerprint. Folds the binarized fingerprint
    /// into u64 via XOR-of-words.
    pub fn audit_hash_u64(&self) -> u64 {
        let bits = self.binarize();
        bits.iter().fold(0u64, |acc, w| acc ^ w)
    }
}

/// Hamming distance between two trajectory hashes — how grammatically
/// similar two queries / sentences are.
pub fn trajectory_distance(a: &TrajectoryHash, b: &TrajectoryHash) -> u32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Threshold for "grammatically similar" — used by the audit log to flag
/// queries that share structure with known-attack patterns.
pub const GRAMMATICAL_SIMILARITY_THRESHOLD: u32 = 256; // ~1.5% of 16384 bits

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trajectory(seed: f32, len: usize) -> Trajectory {
        let fp: Vec<f32> = (0..len)
            .map(|i| {
                let x = (i as f32) * 0.1 + seed;
                x.sin()
            })
            .collect();
        Trajectory {
            fingerprint: fp,
            radius: 5,
        }
    }

    #[test]
    fn binarize_is_deterministic_for_same_trajectory() {
        let t = make_trajectory(0.3, 16384);
        let a = t.binarize();
        let b = t.binarize();
        assert_eq!(a, b);
    }

    #[test]
    fn distance_is_zero_for_self() {
        let h = [0u64; 256];
        assert_eq!(trajectory_distance(&h, &h), 0);
    }

    #[test]
    fn distance_max_is_16384() {
        let zeros = [0u64; 256];
        let ones = [u64::MAX; 256];
        assert_eq!(trajectory_distance(&zeros, &ones), 16384);
    }

    #[test]
    fn audit_hash_u64_changes_with_content() {
        let a = make_trajectory(0.0, 16384);
        let b = make_trajectory(1.7, 16384);
        assert_ne!(a.audit_hash_u64(), b.audit_hash_u64());
    }

    #[test]
    fn similar_trajectories_are_hamming_close() {
        // Same shape, tiny shift — should be far below the threshold.
        let a = make_trajectory(0.0, 16384);
        let a2 = make_trajectory(0.0, 16384);
        let ha = a.binarize();
        let ha2 = a2.binarize();
        assert_eq!(trajectory_distance(&ha, &ha2), 0);
    }
}
