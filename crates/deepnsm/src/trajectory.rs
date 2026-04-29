//! META-AGENT: add `pub mod trajectory;` to lib.rs.

use crate::markov_bundle::GrammaticalRole;

#[derive(Debug, Clone)]
pub struct Trajectory {
    pub fingerprint: Vec<f32>,
    pub radius: u32,
}

impl Trajectory {
    /// Slice the role band out of the trajectory's fingerprint.
    pub fn role_bundle(&self, role: GrammaticalRole) -> Vec<f32> {
        let slice = role.slice();
        self.role_bundle_range(slice.start, slice.stop)
    }

    /// Lower-level slice helper retained for test fixtures + callers
    /// that pre-compute (start, stop) by hand. Prefer `role_bundle(role)`.
    pub fn role_bundle_range(&self, start: usize, stop: usize) -> Vec<f32> {
        let stop = stop.min(self.fingerprint.len());
        if start >= stop {
            return Vec::new();
        }
        self.fingerprint[start..stop].to_vec()
    }

    /// Score the codebook against the role's bundle, filter by
    /// `threshold` (cosine ≥ threshold), sort descending, truncate to
    /// `top_k`.
    ///
    /// `threshold` and `top_k` are explicit so callers tune them per
    /// style / per role band — no hidden 0.5 / 5 defaults baked into
    /// the carrier. See `role_candidates_default` for the
    /// backwards-compat shim with the previous (0.5, 5) values.
    pub fn role_candidates(
        &self,
        role: GrammaticalRole,
        codebook: &[Vec<f32>],
        threshold: f32,
        top_k: usize,
    ) -> Vec<Candidate> {
        let bundle = self.role_bundle(role);
        let mut scored: Vec<Candidate> = codebook
            .iter()
            .enumerate()
            .map(|(i, entry)| Candidate {
                codebook_index: i,
                score: cosine(&bundle, entry),
            })
            .filter(|c| c.score >= threshold)
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        scored
    }

    /// Backwards-compat shim with the previous signature
    /// (threshold = 0.5, top_k = 5).
    #[deprecated(note = "use role_candidates with explicit threshold + top_k")]
    pub fn role_candidates_default(
        &self,
        role: GrammaticalRole,
        codebook: &[Vec<f32>],
    ) -> Vec<Candidate> {
        self.role_candidates(role, codebook, 0.5, 5)
    }
}

/// Cosine similarity. **Panics** on length mismatch — the carrier
/// guarantees role-aligned slices and a length mismatch is a wiring
/// bug, not a runtime input error. Callers that need a fallible
/// variant should length-check before invoking.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "cosine: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    if a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub codebook_index: usize,
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_carrier(value: f32) -> Vec<f32> {
        vec![value; 16_384]
    }

    #[test]
    fn role_bundle_returns_subject_band() {
        let t = Trajectory {
            fingerprint: full_carrier(1.0),
            radius: 5,
        };
        let bundle = t.role_bundle(GrammaticalRole::Subject);
        let _slice = GrammaticalRole::Subject.slice();
        let start = _slice.start;
        let stop = _slice.stop;
        assert_eq!(bundle.len(), stop - start);
    }

    #[test]
    fn role_bundle_range_empty_when_inverted() {
        let t = Trajectory {
            fingerprint: full_carrier(1.0),
            radius: 5,
        };
        assert_eq!(t.role_bundle_range(50, 30).len(), 0);
    }

    #[test]
    fn role_candidates_filters_by_threshold() {
        // Codebook of 5 with similarities [0.9, 0.8, 0.4, 0.3, 0.1];
        // threshold 0.5 → only the first two pass.
        let _slice = GrammaticalRole::Subject.slice();
        let start = _slice.start;
        let stop = _slice.stop;
        let n = stop - start;
        let mut fingerprint = vec![0.0_f32; 16_384];
        for v in fingerprint[start..stop].iter_mut() {
            *v = 1.0;
        }
        let t = Trajectory {
            fingerprint,
            radius: 5,
        };
        // Build codebook entries where each entry has `value` in the
        // subject band and 0.0 elsewhere — cosine vs the all-1 bundle
        // becomes deterministic.
        let make_entry = |scale: f32| -> Vec<f32> { vec![scale; n] };
        // Cosine of all-ones bundle vs scaled all-ones → 1.0 regardless
        // of scale (when scale > 0). Use the sign + zeroing of
        // individual positions to engineer specific cosines.
        // Concretely: an entry that has the first `k` positions = 1 and
        // the rest = 0 has cosine = sqrt(k/n) against the all-ones
        // bundle.
        let make_partial = |k: usize| -> Vec<f32> {
            let mut e = vec![0.0_f32; n];
            for v in e.iter_mut().take(k) {
                *v = 1.0;
            }
            e
        };
        // Choose k so cosines are roughly [0.9, 0.8, 0.4, 0.3, 0.1].
        let codebook: Vec<Vec<f32>> = vec![
            make_partial((0.9_f32 * 0.9 * n as f32) as usize),
            make_partial((0.8_f32 * 0.8 * n as f32) as usize),
            make_partial((0.4_f32 * 0.4 * n as f32) as usize),
            make_partial((0.3_f32 * 0.3 * n as f32) as usize),
            make_partial((0.1_f32 * 0.1 * n as f32) as usize),
        ];
        let _ = make_entry; // silence unused if-the-build-is-aggressive
        let cands = t.role_candidates(GrammaticalRole::Subject, &codebook, 0.5, 10);
        assert_eq!(cands.len(), 2, "threshold 0.5 should keep 2 of 5 entries");
        // Sorted descending — the 0.9-cosine entry comes first.
        assert_eq!(cands[0].codebook_index, 0);
        assert_eq!(cands[1].codebook_index, 1);
    }

    #[test]
    fn role_candidates_top_k_truncation() {
        // Codebook of 10 entries all above threshold — top_k=3 must
        // return exactly 3.
        let _slice = GrammaticalRole::Subject.slice();
        let start = _slice.start;
        let stop = _slice.stop;
        let n = stop - start;
        let mut fingerprint = vec![0.0_f32; 16_384];
        for v in fingerprint[start..stop].iter_mut() {
            *v = 1.0;
        }
        let t = Trajectory {
            fingerprint,
            radius: 5,
        };
        let codebook: Vec<Vec<f32>> = (0..10).map(|_| vec![1.0_f32; n]).collect();
        let cands = t.role_candidates(GrammaticalRole::Subject, &codebook, 0.5, 3);
        assert_eq!(cands.len(), 3);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn cosine_panics_on_length_mismatch() {
        let a = vec![1.0_f32; 10];
        let b = vec![1.0_f32; 11];
        let _ = cosine(&a, &b);
    }
}
