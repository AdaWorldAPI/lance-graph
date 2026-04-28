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
///
/// D4 (2026-04 worker B2) extended this with `winner_index`, an alias
/// `winner`, `dispersion` across the top-3 candidates, and a
/// `candidate_count`. `chosen` and `winner` are equal by construction
/// (`winner` is the canonical D4 name; `chosen` is preserved for
/// existing consumers).
///
/// Empty-candidates contract: returns a sentinel result with
/// `candidate_count = 0`, `winner_index = usize::MAX`, a zero
/// `Binary16K` placeholder fingerprint, and `escalate_to_llm = true`.
/// Callers should check `candidate_count == 0` (or `escalate_to_llm`)
/// before reading `winner` / `chosen`.
#[derive(Debug, Clone)]
pub struct DisambiguationResult {
    pub chosen: CrystalFingerprint,
    pub coherence: f32,
    /// `chosen.coherence - second_place.coherence`. Zero if only one
    /// candidate. `> DISAMBIGUATION_MARGIN_THRESHOLD` (~0.1) means
    /// the winner is confidently above the runner-up.
    pub margin: f32,
    /// All candidates with their scores, sorted descending by coherence.
    pub alternatives: Vec<(CrystalFingerprint, f32)>,
    /// True if `margin < DISAMBIGUATION_MARGIN_THRESHOLD` (ambiguous, escalate).
    pub escalate_to_llm: bool,

    // ── D4 reasoning-operator extensions ──────────────────────────
    /// Index of the winner in the original candidate iterator (0-based).
    /// `usize::MAX` if the candidate iterator was empty.
    pub winner_index: usize,
    /// Best candidate's fingerprint. Equal to `chosen`; provided under the
    /// canonical D4 name for new callers.
    pub winner: CrystalFingerprint,
    /// Mean pairwise normalized Hamming distance across the top-3
    /// candidates' Binary16K fingerprints. High value (close to 0.5)
    /// indicates the alternatives spread out — "no clear winner."
    /// Zero if fewer than two top candidates carry comparable
    /// `Binary16K` fingerprints.
    pub dispersion: f32,
    /// Total candidates evaluated (length of the input iterator).
    pub candidate_count: usize,
}

/// Weighting kernel for temporal position in the Markov chain.
/// Mexican-hat emphasizes focal, de-emphasizes distant positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WeightingKernel {
    /// All positions weighted equally.
    Uniform,
    /// Mexican-hat (DoG) — focal positive, near-neighbors decay through
    /// zero-crossing into a small negative tail. Captures
    /// anticipation / surprise on context shift. Default kernel.
    #[default]
    MexicanHat,
    /// Standard Gaussian decay from focal.
    Gaussian,
}

impl WeightingKernel {
    /// Weight at signed offset `delta` (signed) for window `radius`.
    /// Returns f32 in roughly `[-1, 1]`.
    ///
    /// `delta = 0` is the focal position; `|delta| = radius` is the edge of
    /// the window. The kernel is symmetric in `delta` for all variants
    /// (uniform / mexican-hat / gaussian).
    ///
    /// Approximate ricker wavelet: `(1 - d²) · exp(-d²/2)` where
    /// `d = |delta| / max(radius, 1)`. Gaussian uses `exp(-d²/2)`.
    pub fn weight(&self, delta: i32, radius: u32) -> f32 {
        let r = radius.max(1) as f32;
        match self {
            Self::Uniform => 1.0,
            Self::MexicanHat => {
                let d = delta.unsigned_abs() as f32 / r;
                let dd = d * d;
                (1.0 - dd) * (-dd / 2.0).exp()
            }
            Self::Gaussian => {
                // delta is signed but the kernel is symmetric; using the
                // absolute value avoids relying on signed-cast semantics.
                let d = delta.unsigned_abs() as f32 / r;
                (-(d * d) / 2.0).exp()
            }
        }
    }

    /// Convenience: weight at unsigned distance `d` from focal under the
    /// chain's default radius (`MARKOV_RADIUS`). Equivalent to
    /// `self.weight(d as i32, MARKOV_RADIUS as u32)`.
    pub fn weight_at_distance(&self, d: usize) -> f32 {
        self.weight(d as i32, MARKOV_RADIUS as u32)
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

        let threshold = contributors.div_ceil(2); // majority: strictly more than half
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
    /// Each candidate is scored by the `total_coherence` of the chain
    /// after replacing position `i` with that candidate. The result
    /// also carries `winner_index` (position in the input iterator),
    /// `dispersion` (mean pairwise Binary16K Hamming distance across
    /// the top-3 candidates), and `candidate_count`.
    ///
    /// Edge cases:
    /// - **Empty candidate iterator**: returns the documented sentinel
    ///   (`candidate_count = 0`, `winner_index = usize::MAX`,
    ///   placeholder `Binary16K` fingerprint, `escalate_to_llm = true`).
    ///   Does *not* panic — keeping the API total simplifies caller
    ///   code in the cypher bridge.
    /// - **Single candidate**: `margin = 0.0`, `dispersion = 0.0`,
    ///   `escalate_to_llm = true`.
    pub fn disambiguate<I>(
        &self,
        i: usize,
        candidates: I,
    ) -> DisambiguationResult
    where
        I: IntoIterator<Item = CrystalFingerprint>,
    {
        // Score with original input index preserved so we can report
        // `winner_index` in the iterator's order.
        let mut scored: Vec<(usize, CrystalFingerprint, f32)> = candidates
            .into_iter()
            .enumerate()
            .map(|(idx, cand)| {
                let (_chain, coh) = self.replay_with_alternative(i, cand.clone());
                (idx, cand, coh)
            })
            .collect();

        let candidate_count = scored.len();

        if scored.is_empty() {
            // Documented sentinel — never panic; callers gate on
            // `escalate_to_llm` or `candidate_count == 0`.
            let placeholder =
                CrystalFingerprint::Binary16K(Box::new([0u64; 256]));
            return DisambiguationResult {
                chosen: placeholder.clone(),
                coherence: 0.0,
                margin: 0.0,
                alternatives: Vec::new(),
                escalate_to_llm: true,
                winner_index: usize::MAX,
                winner: placeholder,
                dispersion: 0.0,
                candidate_count: 0,
            };
        }

        // Sort descending by coherence; ties resolved by insertion order
        // (stable sort + NaN-safe partial_cmp fallback to Equal).
        scored.sort_by(|a, b| {
            b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
        });

        let winner_index = scored[0].0;
        let chosen = scored[0].1.clone();
        let coherence = scored[0].2;
        let margin = if scored.len() >= 2 {
            scored[0].2 - scored[1].2
        } else {
            0.0
        };
        let escalate_to_llm =
            scored.len() < 2 || margin < DISAMBIGUATION_MARGIN_THRESHOLD;

        // Dispersion: mean pairwise normalized Hamming distance over the
        // top-3 candidates. Only Binary16K pairs contribute; if fewer
        // than two contribute, dispersion is 0.0 (cannot say).
        let top_n = scored.len().min(3);
        let mut pair_sum: f32 = 0.0;
        let mut pair_count: u32 = 0;
        for a_idx in 0..top_n {
            for b_idx in (a_idx + 1)..top_n {
                let a_bits = binary16k_bits(&scored[a_idx].1);
                let b_bits = binary16k_bits(&scored[b_idx].1);
                if let (Some(a), Some(b)) = (a_bits, b_bits) {
                    let d = hamming_256(a, b) as f32 / MAX_HAMMING_BITS as f32;
                    pair_sum += d;
                    pair_count += 1;
                }
            }
        }
        let dispersion = if pair_count == 0 {
            0.0
        } else {
            pair_sum / pair_count as f32
        };

        let alternatives: Vec<(CrystalFingerprint, f32)> =
            scored.into_iter().map(|(_, fp, c)| (fp, c)).collect();

        DisambiguationResult {
            chosen: chosen.clone(),
            coherence,
            margin,
            alternatives,
            escalate_to_llm,
            winner_index,
            winner: chosen,
            dispersion,
            candidate_count,
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
        // Test through the convenience helper for compactness; the
        // primary API is `weight(delta: i32, radius: u32)`.
        let k = WeightingKernel::MexicanHat;
        let w0 = k.weight_at_distance(0);
        let w1 = k.weight_at_distance(1);
        let w2 = k.weight_at_distance(2);
        let w3 = k.weight_at_distance(3);
        let w4 = k.weight_at_distance(4);
        let w5 = k.weight_at_distance(5);
        assert!(w0 > w1, "w(0)={w0} should exceed w(1)={w1}");
        assert!(w1 > w2, "w(1)={w1} should exceed w(2)={w2}");
        assert!(w2 > w3, "w(2)={w2} should exceed w(3)={w3}");
        assert!(w3 > w4, "w(3)={w3} should exceed w(4)={w4}");
        assert!(w4 > w5, "w(4)={w4} should exceed w(5)={w5}");
        // Uniform and Gaussian sanity checks.
        assert_eq!(WeightingKernel::Uniform.weight_at_distance(0), 1.0);
        assert_eq!(WeightingKernel::Uniform.weight_at_distance(5), 1.0);
        let g0 = WeightingKernel::Gaussian.weight_at_distance(0);
        let g5 = WeightingKernel::Gaussian.weight_at_distance(5);
        assert!(g0 > g5, "gaussian should also decay: g(0)={g0}, g(5)={g5}");
    }

    // ── D4 reasoning-operator tests (worker B2, 2026-04) ────────────────

    /// 1. `Uniform` returns 1.0 at every offset.
    #[test]
    fn d4_uniform_kernel_is_constant() {
        let k = WeightingKernel::Uniform;
        for delta in -10i32..=10 {
            for radius in 1u32..=5 {
                let w = k.weight(delta, radius);
                assert!(
                    (w - 1.0).abs() < f32::EPSILON,
                    "Uniform({delta}, {radius}) = {w}, expected 1.0"
                );
            }
        }
    }

    /// 2. `MexicanHat` is symmetric: w(-d, r) == w(+d, r).
    #[test]
    fn d4_mexican_hat_symmetric() {
        let k = WeightingKernel::MexicanHat;
        for radius in [1u32, 2, 3, 5, 8] {
            for d in 1i32..=10 {
                let w_pos = k.weight(d, radius);
                let w_neg = k.weight(-d, radius);
                assert!(
                    (w_pos - w_neg).abs() < 1e-6,
                    "MexicanHat asymmetric at d={d} r={radius}: \
                     w(+d)={w_pos} w(-d)={w_neg}"
                );
            }
        }
        // Gaussian also symmetric (same code path via |delta|).
        let g = WeightingKernel::Gaussian;
        for radius in [1u32, 5] {
            for d in 1i32..=5 {
                assert!(
                    (g.weight(d, radius) - g.weight(-d, radius)).abs() < 1e-6,
                    "Gaussian should also be symmetric"
                );
            }
        }
    }

    /// 3. `MexicanHat` weight is monotone-decreasing in `|delta|` over the
    ///    radius. Crosses zero at `|delta| ≈ radius` (where d² = 1) and
    ///    stays in the negative tail past the radius — that is the
    ///    Mexican-hat shape.
    #[test]
    fn d4_mexican_hat_monotone_and_zero_crossing() {
        let k = WeightingKernel::MexicanHat;
        let radius: u32 = 5;
        // Monotone decrease in [0, radius].
        let mut prev = k.weight(0, radius);
        assert!(prev > 0.99, "focal weight should be ~1.0, got {prev}");
        for d in 1i32..=radius as i32 {
            let cur = k.weight(d, radius);
            assert!(
                cur < prev,
                "MexicanHat not monotone at d={d}: prev={prev} cur={cur}"
            );
            prev = cur;
        }
        // At |delta| = radius, d² = 1 → (1 - 1) · exp(-0.5) = 0.
        let edge = k.weight(radius as i32, radius);
        assert!(
            edge.abs() < 1e-6,
            "MexicanHat zero-crossing should be at |delta|=radius, got {edge}"
        );
        // Beyond the radius the kernel goes negative (the "hat brim").
        let beyond = k.weight((radius as i32) + 1, radius);
        assert!(
            beyond < 0.0,
            "MexicanHat should be negative beyond radius, got {beyond}"
        );
    }

    /// 4. `coherence_at` on a chain of identical fingerprints is ~1.0.
    ///    (Existing `coherence_high_for_self_chain` covers this; this
    ///    test re-verifies the D4 contract independently of the
    ///    pre-existing test.)
    #[test]
    fn d4_coherence_self_chain_is_one() {
        let fp = mk_fp(0x0102_0304_0506_0708);
        let chain = fill_chain_with(&fp);
        for i in 0..CHAIN_LEN {
            let c = chain.coherence_at(i);
            assert!(
                c > 0.99,
                "self-chain coherence at {i} should be ~1.0, got {c}"
            );
        }
        let total = chain.total_coherence();
        assert!(
            total > 0.99,
            "self-chain total_coherence should be ~1.0, got {total}"
        );
    }

    /// 5. `disambiguate` with two candidates where one matches the
    ///    surrounding chain → that one wins with non-zero margin and
    ///    `winner_index` points at it.
    #[test]
    fn d4_disambiguate_picks_matching_candidate() {
        let base = mk_fp(0x9999_AAAA_BBBB_CCCC);
        let mut chain = fill_chain_with(&base);
        // Blank position 4 so we can replay alternatives in.
        chain.fingerprints[4] = None;

        // Far-miss: fully inverted vs. base.
        let far = match &base {
            CrystalFingerprint::Binary16K(bits) => {
                let mut inv = Box::new([0u64; 256]);
                for (i, w) in bits.iter().enumerate() {
                    inv[i] = !w;
                }
                CrystalFingerprint::Binary16K(inv)
            }
            _ => unreachable!(),
        };

        // Order: [far, base] → if base wins, winner_index must be 1.
        let res = chain.disambiguate(4, vec![far, base.clone()]);
        assert_eq!(res.candidate_count, 2);
        assert_eq!(res.winner_index, 1, "base was at iterator index 1");
        assert!(
            res.margin > 0.0,
            "matching candidate should have non-zero margin, got {}",
            res.margin
        );
        // `winner` and `chosen` agree by construction.
        match (&res.winner, &res.chosen) {
            (CrystalFingerprint::Binary16K(a),
             CrystalFingerprint::Binary16K(b)) => {
                assert_eq!(**a, **b, "winner and chosen must agree");
            }
            _ => panic!("unexpected fingerprint variants"),
        }
        // Winner equals base.
        match (&res.winner, &base) {
            (CrystalFingerprint::Binary16K(a),
             CrystalFingerprint::Binary16K(b)) => {
                assert_eq!(**a, **b, "winner must be the matching base");
            }
            _ => panic!("unexpected fingerprint variants"),
        }
    }

    /// 6. `disambiguate` with an empty candidate iterator returns the
    ///    documented sentinel result (no panic). `candidate_count = 0`,
    ///    `winner_index = usize::MAX`, `escalate_to_llm = true`.
    #[test]
    fn d4_disambiguate_empty_returns_sentinel() {
        let chain = fill_chain_with(&mk_fp(0x1));
        let res: DisambiguationResult =
            chain.disambiguate(0, Vec::<CrystalFingerprint>::new());
        assert_eq!(res.candidate_count, 0);
        assert_eq!(res.winner_index, usize::MAX);
        assert!(res.escalate_to_llm, "empty must escalate");
        assert!(res.alternatives.is_empty());
        assert_eq!(res.coherence, 0.0);
        assert_eq!(res.margin, 0.0);
        assert_eq!(res.dispersion, 0.0);
        // The placeholder fingerprint is a zeroed Binary16K.
        match &res.winner {
            CrystalFingerprint::Binary16K(bits) => {
                assert!(bits.iter().all(|&w| w == 0),
                    "sentinel placeholder should be all-zero");
            }
            _ => panic!("sentinel must be Binary16K placeholder"),
        }
    }

    /// `WeightingKernel::default()` is `MexicanHat` (D4 chose this as
    /// the canonical kernel — focal-emphasizing with anticipation tail).
    #[test]
    fn d4_default_kernel_is_mexican_hat() {
        let k: WeightingKernel = Default::default();
        assert_eq!(k, WeightingKernel::MexicanHat);
    }
}
