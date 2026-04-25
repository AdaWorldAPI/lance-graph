//! Markov ±5 context chain with replay.
//!
//! The chain carries `2 × MARKOV_RADIUS + 1 = 11` fingerprints of the
//! sentences neighboring a focal point. When a Wechsel (or any
//! ambiguity) is marked, the chain is **replayed** — re-scanned with
//! each candidate branch pinned — and whichever branch preserves
//! coherent NARS confidence across the window wins.

use crate::crystal::fingerprint::CrystalFingerprint;

pub const MARKOV_RADIUS: usize = 5;
pub const CHAIN_LEN: usize = 2 * MARKOV_RADIUS + 1;

/// Counterfactual margin below which we escalate to an LLM.
pub const DISAMBIGUATION_MARGIN_THRESHOLD: f32 = 0.1;

/// Max Hamming distance across a full Binary16K fingerprint: 256 u64 × 64 bits.
const MAX_HAMMING_BITS: u32 = 256 * 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayDirection {
    Forward,
    Backward,
    BothAndCompare,
}

/// A request to replay the chain with a specific branch pinned.
#[derive(Debug, Clone)]
pub struct ReplayRequest {
    /// Which token's ambiguity to resolve.
    pub token_index: u16,
    /// Candidate role id the replay should assume.
    pub candidate_id: u8,
    pub direction: ReplayDirection,
}

/// A ±5 Markov context chain of fingerprints.
#[derive(Debug, Clone)]
pub struct ContextChain {
    /// 11 fingerprints. Position `MARKOV_RADIUS` (index 5) is the focal
    /// sentence; 0..5 are preceding (oldest first); 6..11 are following.
    pub fingerprints: Vec<Option<CrystalFingerprint>>,
}

/// Result of a counterfactual disambiguation: the chosen candidate, its
/// coherence, the margin to second place, the full ranked alternatives,
/// and whether the caller should escalate to an LLM.
#[derive(Debug, Clone)]
pub struct DisambiguationResult {
    pub chosen: CrystalFingerprint,
    pub coherence: f32,
    /// `chosen.coherence - second_place.coherence`. Zero if only one candidate.
    pub margin: f32,
    /// All candidates with their scores, sorted descending by coherence.
    pub alternatives: Vec<(CrystalFingerprint, f32)>,
    /// True if `margin < DISAMBIGUATION_MARGIN_THRESHOLD` (ambiguous, escalate).
    pub escalate_to_llm: bool,
}

/// Weighting kernel for temporal position in the Markov chain.
/// Mexican-hat emphasizes focal, de-emphasizes distant positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightingKernel {
    Uniform,
    MexicanHat,
    Gaussian,
}

impl WeightingKernel {
    /// Weight for a position at distance `d` from focal (0 = focal, 5 = edge).
    pub fn weight(&self, d: usize) -> f32 {
        match self {
            Self::Uniform => 1.0,
            Self::MexicanHat => {
                // Peak at focal (d=0), smooth fall-off, slight negative at edge.
                let x = d as f32 / (MARKOV_RADIUS as f32);
                (1.0 - 2.0 * x * x) * (-x * x * 2.0).exp()
            }
            Self::Gaussian => {
                let x = d as f32 / (MARKOV_RADIUS as f32);
                (-x * x * 2.0).exp()
            }
        }
    }
}

impl ContextChain {
    pub fn new() -> Self {
        Self { fingerprints: (0..CHAIN_LEN).map(|_| None).collect() }
    }

    pub fn focal_index() -> usize { MARKOV_RADIUS }

    /// Count of filled positions in the chain.
    pub fn filled(&self) -> usize {
        self.fingerprints.iter().filter(|f| f.is_some()).count()
    }

    /// Whether the chain is saturated on both sides of the focal.
    pub fn is_saturated(&self) -> bool {
        self.filled() == CHAIN_LEN
    }

    pub fn focal(&self) -> Option<&CrystalFingerprint> {
        self.fingerprints[Self::focal_index()].as_ref()
    }

    /// Coherence of position `i` against the rest of the chain.
    /// Computed as `1 - hamming(fp[i], bundled_excluding_i) / max_hamming`.
    /// Returns `0.0` if position `i` is empty or out of range, or if no
    /// other Binary16K fingerprints are present to bundle against.
    pub fn coherence_at(&self, i: usize) -> f32 {
        if i >= self.fingerprints.len() {
            return 0.0;
        }
        let fp_i = match self.fingerprints[i].as_ref() {
            Some(fp) => fp,
            None => return 0.0,
        };
        let fp_i_bits = match binary16k_bits(fp_i) {
            Some(b) => b,
            None => return 0.0,
        };

        // Bundle all other Binary16K positions via naive majority-XOR:
        // a bit is set in the bundle if it is set in the majority of the
        // contributing fingerprints. With an even number of contributors
        // and a tie, we leave the bit unset (conservative).
        let mut counts = [0u16; MAX_HAMMING_BITS as usize];
        let mut contributors: u16 = 0;
        for (j, slot) in self.fingerprints.iter().enumerate() {
            if j == i { continue; }
            if let Some(fp) = slot {
                if let Some(bits) = binary16k_bits(fp) {
                    contributors += 1;
                    for (w, &word) in bits.iter().enumerate() {
                        let base = w * 64;
                        for b in 0..64 {
                            if (word >> b) & 1 == 1 {
                                counts[base + b] += 1;
                            }
                        }
                    }
                }
            }
        }
        if contributors == 0 {
            return 0.0;
        }

        let threshold = (contributors + 1) / 2; // majority: strictly more than half
        let mut bundle = [0u64; 256];
        for (w, slot) in bundle.iter_mut().enumerate() {
            let mut word: u64 = 0;
            let base = w * 64;
            for b in 0..64 {
                if counts[base + b] >= threshold
                    && counts[base + b] * 2 > contributors
                {
                    word |= 1u64 << b;
                }
            }
            *slot = word;
        }

        let dist = hamming_256(fp_i_bits, &bundle);
        1.0 - (dist as f32 / MAX_HAMMING_BITS as f32)
    }

    /// Mean coherence across all filled positions. Returns `0.0` if the
    /// chain is entirely empty.
    pub fn total_coherence(&self) -> f32 {
        let mut sum = 0.0f32;
        let mut n = 0u32;
        for i in 0..self.fingerprints.len() {
            if self.fingerprints[i].is_some() {
                sum += self.coherence_at(i);
                n += 1;
            }
        }
        if n == 0 { 0.0 } else { sum / n as f32 }
    }

    /// Returns a new chain where position `i` has been replaced with `alt`.
    /// Used for counterfactual disambiguation testing.
    /// Second return value is the `total_coherence` of the modified chain.
    /// If `i` is out of range, returns a clone of the chain untouched and
    /// its current total coherence.
    pub fn replay_with_alternative(
        &self,
        i: usize,
        alt: CrystalFingerprint,
    ) -> (Self, f32) {
        let mut cloned = self.clone();
        if i < cloned.fingerprints.len() {
            cloned.fingerprints[i] = Some(alt);
        }
        let coh = cloned.total_coherence();
        (cloned, coh)
    }

    /// Counterfactual disambiguation: try each candidate at position `i`,
    /// return the one with highest coherence and the decision margin.
    ///
    /// Edge cases:
    /// - Empty candidate list: returns a result with a placeholder zero
    ///   fingerprint and `escalate_to_llm = true`.
    /// - Single candidate: `margin = 0.0`, `escalate_to_llm = true`.
    pub fn disambiguate<I>(
        &self,
        i: usize,
        candidates: I,
    ) -> DisambiguationResult
    where
        I: IntoIterator<Item = CrystalFingerprint>,
    {
        let mut scored: Vec<(CrystalFingerprint, f32)> = candidates
            .into_iter()
            .map(|cand| {
                let (_chain, coh) = self.replay_with_alternative(i, cand.clone());
                (cand, coh)
            })
            .collect();

        if scored.is_empty() {
            // Placeholder: caller should check `escalate_to_llm`.
            return DisambiguationResult {
                chosen: CrystalFingerprint::Binary16K(Box::new([0u64; 256])),
                coherence: 0.0,
                margin: 0.0,
                alternatives: Vec::new(),
                escalate_to_llm: true,
            };
        }

        // Sort descending by coherence; ties resolved by insertion order
        // (stable sort + NaN-safe partial_cmp fallback to Equal).
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        let (chosen, coherence) = scored[0].clone();
        let margin = if scored.len() >= 2 {
            scored[0].1 - scored[1].1
        } else {
            0.0
        };
        let escalate_to_llm =
            scored.len() < 2 || margin < DISAMBIGUATION_MARGIN_THRESHOLD;

        DisambiguationResult {
            chosen,
            coherence,
            margin,
            alternatives: scored,
            escalate_to_llm,
        }
    }
}

impl Default for ContextChain {
    fn default() -> Self { Self::new() }
}

// ── Internal helpers ───────────────────────────────────────────────────

/// Extract the 256-word bit array from a Binary16K fingerprint.
/// Returns `None` for other variants (contract crate is zero-dep and
/// does not carry the math surface to bridge Structured5x5 / Vsa10k here).
#[inline]
fn binary16k_bits(fp: &CrystalFingerprint) -> Option<&[u64; 256]> {
    match fp {
        CrystalFingerprint::Binary16K(bits) => Some(bits),
        _ => None,
    }
}

/// Popcount of XOR across 256 u64 words. Returns the total number of
/// differing bits in [0, 16_384].
#[inline]
fn hamming_256(a: &[u64; 256], b: &[u64; 256]) -> u32 {
    let mut d: u32 = 0;
    for w in 0..256 {
        d += (a[w] ^ b[w]).count_ones();
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_fp(pattern: u64) -> CrystalFingerprint {
        let mut bits = Box::new([0u64; 256]);
        for i in 0..256 {
            bits[i] = pattern.wrapping_mul(i as u64 + 1);
        }
        CrystalFingerprint::Binary16K(bits)
    }

    fn fill_chain_with(fp: &CrystalFingerprint) -> ContextChain {
        let mut c = ContextChain::new();
        for i in 0..CHAIN_LEN {
            c.fingerprints[i] = Some(fp.clone());
        }
        c
    }

    #[test]
    fn chain_length_is_eleven() {
        let c = ContextChain::new();
        assert_eq!(c.fingerprints.len(), CHAIN_LEN);
        assert_eq!(CHAIN_LEN, 11);
    }

    #[test]
    fn focal_is_center() {
        assert_eq!(ContextChain::focal_index(), 5);
    }

    #[test]
    fn coherence_zero_for_empty_chain() {
        let c = ContextChain::new();
        assert_eq!(c.total_coherence(), 0.0);
        for i in 0..CHAIN_LEN {
            assert_eq!(c.coherence_at(i), 0.0);
        }
        // Out of range is also zero.
        assert_eq!(c.coherence_at(CHAIN_LEN + 10), 0.0);
    }

    #[test]
    fn coherence_high_for_self_chain() {
        // Fill the chain with the same fingerprint → bundle should
        // equal the fingerprint itself and coherence should be ~1.0.
        let fp = mk_fp(0xDEAD_BEEF_CAFE_BABE);
        let c = fill_chain_with(&fp);
        let total = c.total_coherence();
        assert!(total > 0.99, "expected near-1.0 self-coherence, got {total}");
        for i in 0..CHAIN_LEN {
            let k = c.coherence_at(i);
            assert!(k > 0.99, "position {i} coherence {k} should be near 1.0");
        }
    }

    #[test]
    fn replay_preserves_other_positions() {
        let base = mk_fp(0xAAAA_AAAA_AAAA_AAAA);
        let alt = mk_fp(0x5555_5555_5555_5555);
        let c = fill_chain_with(&base);
        let (replayed, _coh) = c.replay_with_alternative(3, alt.clone());
        // Position 3 should equal alt.
        match (&replayed.fingerprints[3], &alt) {
            (Some(CrystalFingerprint::Binary16K(a)),
             CrystalFingerprint::Binary16K(b)) => {
                assert_eq!(**a, **b, "position 3 should be the alt fingerprint");
            }
            _ => panic!("position 3 missing or wrong variant"),
        }
        // All other positions should equal `base`.
        for i in 0..CHAIN_LEN {
            if i == 3 { continue; }
            match (&replayed.fingerprints[i], &base) {
                (Some(CrystalFingerprint::Binary16K(a)),
                 CrystalFingerprint::Binary16K(b)) => {
                    assert_eq!(**a, **b,
                        "position {i} was mutated by replay");
                }
                _ => panic!("position {i} unexpectedly empty or wrong variant"),
            }
        }
        // The original chain must not have been modified.
        match &c.fingerprints[3] {
            Some(CrystalFingerprint::Binary16K(a)) => {
                match &base {
                    CrystalFingerprint::Binary16K(b) => {
                        assert_eq!(**a, **b,
                            "original chain should be untouched");
                    }
                    _ => unreachable!(),
                }
            }
            _ => panic!("original position 3 missing"),
        }
    }

    #[test]
    fn disambiguate_picks_most_coherent() {
        // Chain full of `base`; candidates include base (perfect),
        // a near-miss, and a far-miss. Base must win.
        let base = mk_fp(0x1111_2222_3333_4444);
        let mut c = fill_chain_with(&base);
        // Blank out position 3 so we can replay into it.
        c.fingerprints[3] = None;

        // Near-miss: flip a single bit.
        let near = {
            let mut bits = Box::new([0u64; 256]);
            for i in 0..256 {
                bits[i] = 0x1111_2222_3333_4444u64.wrapping_mul(i as u64 + 1);
            }
            bits[0] ^= 1;
            CrystalFingerprint::Binary16K(bits)
        };
        // Far-miss: fully inverted.
        let far = {
            let mut bits = Box::new([0u64; 256]);
            for i in 0..256 {
                bits[i] =
                    !(0x1111_2222_3333_4444u64.wrapping_mul(i as u64 + 1));
            }
            CrystalFingerprint::Binary16K(bits)
        };

        let res = c.disambiguate(
            3,
            vec![far.clone(), near.clone(), base.clone()],
        );
        // Base should win (it matches the surrounding bundle perfectly).
        match (&res.chosen, &base) {
            (CrystalFingerprint::Binary16K(a),
             CrystalFingerprint::Binary16K(b)) => {
                assert_eq!(**a, **b, "disambiguate should pick base");
            }
            _ => panic!("wrong variant in chosen"),
        }
        assert_eq!(res.alternatives.len(), 3);
        // Monotone order: first ≥ second ≥ third.
        assert!(res.alternatives[0].1 >= res.alternatives[1].1);
        assert!(res.alternatives[1].1 >= res.alternatives[2].1);
        // Margin between base and near-miss should be tiny but positive;
        // the margin between base and far-miss is huge. With 3 candidates,
        // `margin` is top-vs-second which is near-miss.
        assert!(res.margin >= 0.0);
    }

    #[test]
    fn disambiguate_escalates_on_tie() {
        // Two identical candidates → margin = 0 → escalate.
        let base = mk_fp(0xF00D_F00D_F00D_F00D);
        let c = fill_chain_with(&base);
        let cand_a = mk_fp(0x1234_5678_9ABC_DEF0);
        let cand_b = cand_a.clone();
        let res = c.disambiguate(3, vec![cand_a, cand_b]);
        assert_eq!(res.alternatives.len(), 2);
        assert!(res.margin.abs() < 1e-6,
            "two identical candidates should produce zero margin, got {}",
            res.margin);
        assert!(res.escalate_to_llm,
            "zero margin must trigger LLM escalation");
    }

    #[test]
    fn mexican_hat_weights_monotone() {
        // Mexican-hat: peak at d=0, monotone decrease through d=1..5.
        let k = WeightingKernel::MexicanHat;
        let w0 = k.weight(0);
        let w1 = k.weight(1);
        let w2 = k.weight(2);
        let w3 = k.weight(3);
        let w4 = k.weight(4);
        let w5 = k.weight(5);
        assert!(w0 > w1, "w(0)={w0} should exceed w(1)={w1}");
        assert!(w1 > w2, "w(1)={w1} should exceed w(2)={w2}");
        assert!(w2 > w3, "w(2)={w2} should exceed w(3)={w3}");
        assert!(w3 > w4, "w(3)={w3} should exceed w(4)={w4}");
        assert!(w4 > w5, "w(4)={w4} should exceed w(5)={w5}");
        // Uniform and Gaussian sanity checks.
        assert_eq!(WeightingKernel::Uniform.weight(0), 1.0);
        assert_eq!(WeightingKernel::Uniform.weight(5), 1.0);
        let g0 = WeightingKernel::Gaussian.weight(0);
        let g5 = WeightingKernel::Gaussian.weight(5);
        assert!(g0 > g5, "gaussian should also decay: g(0)={g0}, g(5)={g5}");
    }
}
