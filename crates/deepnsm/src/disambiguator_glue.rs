//! Bridge: deepnsm trajectory bundles → contract `ContextChain` disambiguation.
//!
//! ## Why this module exists
//!
//! The "real fingerprint" path on `ContextChain::disambiguate_with` accepts a
//! `Option<CrystalFingerprint>` as `sentinel_fp` for the empty-candidates
//! sentinel branch. Until this module shipped, the contract crate had no
//! caller that produced a real (non-zero, non-placeholder) fingerprint —
//! the parameter was effectively dead-letter typing. The "real fp" honesty
//! gap came from advertising support for `Some(fp)` without anyone wiring
//! a real `MarkovBundler::role_bundle()` projection through it.
//!
//! `disambiguate_with_trajectory` closes that gap. It sign-binarizes a
//! 16,384-dim f32 trajectory bundle into a `Binary16K` fingerprint and
//! threads it as the sentinel fingerprint. This way, when the chain has
//! no in-window evidence (empty candidates), the result still carries a
//! signal — the bundled-trajectory fingerprint of where the parser
//! actually was — instead of an all-zero placeholder.
//!
//! ## Why sign-binarize?
//!
//! `Binary16K` is the 2 KB Hamming-compare format
//! (`Box<[u64; 256]>`, 256 × 64 = 16,384 bits, one bit per dim). The
//! lossless route from Vsa16kF32 to Binary16K is a sign threshold:
//! `dim ≥ 0 → bit set`. This preserves bipolar VSA semantics for
//! Hamming-distance comparison; magnitude is discarded by design (the
//! magnitude lives in Vsa16kF32).
//!
//! ## Workspace iron rule consistency
//!
//! Per `I-VSA-IDENTITIES`: the trajectory bundle is an IDENTITY
//! superposition of role-bound content fingerprints, not bitpacked
//! content. Sign-binarizing it for Hamming compare is the canonical
//! switchboard hop (Vsa16kF32 → Binary16K) called out in
//! `vsa-switchboard-architecture.md`.

use lance_graph_contract::crystal::fingerprint::CrystalFingerprint;
use lance_graph_contract::grammar::context_chain::{
    ContextChain, DisambiguateOpts, DisambiguationResult,
};

/// Number of bits in a `Binary16K` fingerprint (16,384).
const BINARY16K_BITS: usize = 16_384;

/// Number of u64 words in a `Binary16K` fingerprint (256).
const BINARY16K_WORDS: usize = 256;

/// Disambiguate at position `i` against `candidates`, with the sentinel
/// fingerprint sourced from a real trajectory `bundle` (typically
/// `MarkovBundler::role_bundle(role)` or the full role-superposed
/// trajectory). The bundle is sign-binarized into a `Binary16K`
/// fingerprint and threaded through `DisambiguateOpts::sentinel_fp`.
///
/// On the empty-candidates sentinel path the result's `winner` /
/// `chosen` carry the bundled-trajectory fingerprint instead of the
/// zero placeholder — closing the "real fp" honesty gap that prior
/// PR-G3 work left open (the contract accepted the option but no
/// caller produced a real value).
///
/// `bundle` lengths shorter than 16,384 are zero-padded to 16,384;
/// longer bundles are truncated. Both edge cases are intentionally
/// silent — the deepnsm carrier is by-construction
/// 16,384 dims (`Vsa16kF32`-shaped), so any deviation is a wiring
/// bug at the call site, not user input.
pub fn disambiguate_with_trajectory<I>(
    chain: &ContextChain,
    i: usize,
    candidates: I,
    bundle: &[f32],
) -> DisambiguationResult
where
    I: IntoIterator<Item = CrystalFingerprint>,
{
    let bits = sign_binarize_to_binary16k(bundle);
    chain.disambiguate_with(
        i,
        candidates,
        DisambiguateOpts {
            kernel: None,
            sentinel_fp: Some(CrystalFingerprint::Binary16K(bits)),
        },
    )
}

/// Sign-binarize a 16,384-dim f32 bundle into a `Binary16K` payload.
///
/// Bit `i` is set iff `bundle[i] >= 0.0`. Bundle entries beyond
/// `bundle.len()` (when shorter than 16,384) are treated as `< 0.0`
/// (bit clear). Bundle entries past 16,384 are ignored.
///
/// Returns the boxed `[u64; 256]` shape that the
/// `CrystalFingerprint::Binary16K` variant wraps.
pub fn sign_binarize_to_binary16k(bundle: &[f32]) -> Box<[u64; BINARY16K_WORDS]> {
    let mut out = Box::new([0u64; BINARY16K_WORDS]);
    for (i, &v) in bundle.iter().take(BINARY16K_BITS).enumerate() {
        if v >= 0.0 {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::crystal::fingerprint::CrystalFingerprint;
    use lance_graph_contract::grammar::context_chain::ContextChain;

    fn empty_chain() -> ContextChain {
        ContextChain::new()
    }

    /// Loose-end-#2 failing-test-first witness (the "real fp" honesty
    /// gap closer): on the empty-candidates sentinel path,
    /// `disambiguate_with_trajectory` with an all-positive bundle must
    /// produce a `Binary16K` fingerprint that has at least one bit
    /// set — i.e. the result is *not* the zero-sentinel.
    ///
    /// This test would fail (compile-but-no-such-function) without
    /// `disambiguate_with_trajectory`, which is the whole point of
    /// the bridge: the contract accepted `Option<CrystalFingerprint>`
    /// as a passthrough but no caller produced a real value.
    #[test]
    fn test_caller_constructs_real_fingerprint_not_zero() {
        let chain = empty_chain();
        let bundle_of_all_ones = vec![1.0_f32; BINARY16K_BITS];

        let result = disambiguate_with_trajectory(
            &chain,
            0,
            std::iter::empty::<CrystalFingerprint>(),
            &bundle_of_all_ones,
        );

        // Sentinel path was taken (no candidates).
        assert_eq!(result.candidate_count, 0);
        assert_eq!(result.winner_index, usize::MAX);
        assert!(result.escalate_to_llm);

        // Critical assertion: chosen carries a non-zero fingerprint.
        match &result.chosen {
            CrystalFingerprint::Binary16K(bits) => {
                let any_set = bits.iter().any(|&w| w != 0);
                assert!(
                    any_set,
                    "real-fp caller must NOT produce a zero-sentinel: \
                     all-positive bundle → all bits set (saw all-zero)"
                );
                // Strong form: an all-positive bundle should set ALL bits.
                assert!(
                    bits.iter().all(|&w| w == u64::MAX),
                    "all-positive bundle should sign-binarize to all-ones"
                );
            }
            _ => panic!("expected Binary16K sentinel"),
        }
    }

    /// Sign-binarize unit test: an all-positive bundle becomes all-1s.
    #[test]
    fn sign_binarize_all_positive_yields_all_ones() {
        let bundle = vec![0.5_f32; BINARY16K_BITS];
        let bits = sign_binarize_to_binary16k(&bundle);
        for (i, &w) in bits.iter().enumerate() {
            assert_eq!(
                w,
                u64::MAX,
                "word {i} should be all-ones for all-positive bundle"
            );
        }
    }

    /// Sign-binarize complementary test: an all-negative bundle stays
    /// all-zeros (bit set requires `>= 0.0`). Anchors the contract.
    #[test]
    fn sign_binarize_all_negative_yields_all_zeros() {
        let bundle = vec![-1.0_f32; BINARY16K_BITS];
        let bits = sign_binarize_to_binary16k(&bundle);
        for (i, &w) in bits.iter().enumerate() {
            assert_eq!(w, 0u64, "word {i} should be zero for all-negative bundle");
        }
    }

    /// Round-trip: two distinct bundles (all-positive vs sign-flipped
    /// at every other dim) drive `disambiguate_with_trajectory` to two
    /// different sentinel fingerprints. Confirms the bundle actually
    /// flows into the result, not a constant.
    #[test]
    fn round_trip_different_bundles_produce_different_fingerprints() {
        let chain = empty_chain();

        let bundle_a: Vec<f32> = (0..BINARY16K_BITS).map(|_| 1.0).collect();
        let bundle_b: Vec<f32> = (0..BINARY16K_BITS)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        let res_a = disambiguate_with_trajectory(
            &chain,
            0,
            std::iter::empty::<CrystalFingerprint>(),
            &bundle_a,
        );
        let res_b = disambiguate_with_trajectory(
            &chain,
            0,
            std::iter::empty::<CrystalFingerprint>(),
            &bundle_b,
        );

        let bits_a = match &res_a.chosen {
            CrystalFingerprint::Binary16K(b) => b.clone(),
            _ => panic!("expected Binary16K"),
        };
        let bits_b = match &res_b.chosen {
            CrystalFingerprint::Binary16K(b) => b.clone(),
            _ => panic!("expected Binary16K"),
        };

        // The two bundles must produce distinct sentinel fingerprints.
        assert_ne!(
            *bits_a, *bits_b,
            "different bundles must produce different sentinel fingerprints"
        );
        // Bundle B has alternating signs → exactly half the bits set.
        let popcount_b: u32 = bits_b.iter().map(|w| w.count_ones()).sum();
        assert_eq!(
            popcount_b,
            (BINARY16K_BITS / 2) as u32,
            "alternating-sign bundle should set every other bit"
        );
    }

    /// Truncation contract: a bundle longer than 16,384 silently keeps
    /// only the first 16,384 dims; tail is ignored.
    #[test]
    fn sign_binarize_truncates_oversized_bundle() {
        let mut bundle = vec![1.0_f32; BINARY16K_BITS];
        bundle.extend(std::iter::repeat(-1.0_f32).take(100));
        let bits = sign_binarize_to_binary16k(&bundle);
        // First 16,384 bits → all-positive → all-ones.
        for &w in bits.iter() {
            assert_eq!(w, u64::MAX);
        }
    }
}
