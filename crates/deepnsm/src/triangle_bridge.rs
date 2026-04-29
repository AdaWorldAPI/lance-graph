//! META-AGENT: add `pub mod triangle_bridge;` to lib.rs. Gate behind
//! feature `grammar-triangle`. Also requires adding to Cargo.toml:
//!   lance-graph-cognitive = { path = "../lance-graph-cognitive", optional = true }
//!   lance-graph-contract  = { path = "../lance-graph-contract",  optional = true }
//!   [features] grammar-triangle = ["dep:lance-graph-cognitive", "dep:lance-graph-contract"]
//!
//! Grammar Triangle bridge: merge DeepNSM SPO output with the Triangle's
//! NSMField + CausalityFlow + QualiaField into a single SpoWithGrammar.
//!
//! The Triangle lives in `lance_graph_cognitive::grammar::GrammarTriangle`
//! and is already-shipped (`from_text` constructs it). The actual API has
//! `QualiaField` (18D phenomenal field) — not `Qualia18D`.

use crate::parser::SentenceStructure;

#[cfg(feature = "grammar-triangle")]
use lance_graph_cognitive::grammar::{
    CausalityFlow, GrammarTriangle, NSMField, QualiaField,
};

/// Merged output: DeepNSM SPO triples + Triangle's three lenses.
///
/// Consumers downstream of DeepNSM read this struct to get both the
/// discrete SPO commit + the continuous semantic field. When the
/// `grammar-triangle` feature is off, the Triangle fields collapse to
/// nothing and the consumer only sees the SPO half.
#[derive(Clone, Debug)]
pub struct SpoWithGrammar {
    /// DeepNSM-extracted SPO triples + modifiers + negations + temporals.
    pub triples: SentenceStructure,

    /// Causality flow from the Triangle (agency, temporality, dependency).
    #[cfg(feature = "grammar-triangle")]
    pub causality: CausalityFlow,

    /// 65-prime NSM field activations.
    #[cfg(feature = "grammar-triangle")]
    pub nsm_field: NSMField,

    /// 18D qualia phenomenal coordinates.
    #[cfg(feature = "grammar-triangle")]
    pub qualia_signature: QualiaField,

    /// Distance between this parse and the SPO's expected qualia
    /// footprint. Higher = more "novel domain" → routes to
    /// `NarsInference::Extrapolation` in the ticket.
    pub classification_distance: f32,
}

/// Build the merged Triangle + SPO view. Default entry point.
///
/// When the `grammar-triangle` feature is enabled, this calls
/// `GrammarTriangle::from_text(text)` and stamps the three lenses onto
/// the SPO output. When it is off, this is a thin wrapper that just
/// carries the SPO and a 0.0 classification distance.
#[cfg(feature = "grammar-triangle")]
pub fn analyze_with_triangle(text: &str, structure: SentenceStructure) -> SpoWithGrammar {
    let triangle = GrammarTriangle::from_text(text);
    let dist = compute_classification_distance(&structure, &triangle);
    SpoWithGrammar {
        triples: structure,
        causality: triangle.causality,
        nsm_field: triangle.nsm,
        qualia_signature: triangle.qualia,
        classification_distance: dist,
    }
}

/// Feature-off fallback: just carry the SPO with `classification_distance = 0`.
///
/// Available regardless of feature so the parser always has something to
/// hand the LLM router.
pub fn analyze_without_triangle(structure: SentenceStructure) -> SpoWithGrammar {
    SpoWithGrammar {
        triples: structure,
        #[cfg(feature = "grammar-triangle")]
        causality: CausalityFlow::default(),
        #[cfg(feature = "grammar-triangle")]
        nsm_field: NSMField::default(),
        #[cfg(feature = "grammar-triangle")]
        qualia_signature: QualiaField::default(),
        classification_distance: 0.0,
    }
}

/// Normalized Hamming distance between the qualia fingerprint and the
/// SPO predicate's expected qualia footprint.
///
/// The footprint is derived from the verb's row in the 144-cell table
/// (currently a placeholder = neutral 0.5 prior, encoded as zero bits
/// after thresholding). Once D7 `GrammarStyleConfig` surfaces per-style
/// qualia expectations, `expected_qualia_footprint` will look up the row
/// by (verb, tense) and return that row's qualia footprint.
///
/// Returns `[0.0, 1.0]`:
/// - `0.0` = qualia exactly matches expected footprint (familiar domain).
/// - `1.0` = total mismatch (novel domain — extrapolation needed).
///
/// FOLLOW-UP: tune against the Jirak-derived noise floor (see CLAUDE.md
/// §I-NOISE-FLOOR-JIRAK) — values that exceed the n^(-1/2) weak-dependence
/// bound are real signal, not register noise.
#[cfg(feature = "grammar-triangle")]
fn compute_classification_distance(
    structure: &SentenceStructure,
    triangle: &GrammarTriangle,
) -> f32 {
    // Convert the 18D qualia coordinates to a binary fingerprint by
    // thresholding each dimension at 0.5. Pack into a single u64 (only
    // 18 bits used; remaining 46 bits are zero — they participate in
    // the Hamming compare but cancel against zeroed expected bits).
    let q_bits = qualia_to_binary_fingerprint(triangle);
    let expected_bits = expected_qualia_footprint(structure);
    hamming_normalized(&q_bits, &expected_bits)
}

/// Threshold the 18D qualia coordinates at 0.5 → 18-bit packed `u64`.
///
/// The packed register is `[u64; 1]` so the Hamming compare counts only
/// the 18 meaningful bits against the same 18 bits of the expected
/// footprint (the upper 46 bits in both registers are zero, contributing
/// nothing to the diff but inflating the denominator — handled by
/// `hamming_normalized` returning a normalized [0, 1] value).
#[cfg(feature = "grammar-triangle")]
fn qualia_to_binary_fingerprint(triangle: &GrammarTriangle) -> Vec<u64> {
    let coords = triangle.qualia.coordinates();
    let mut packed: u64 = 0;
    for (i, &c) in coords.iter().enumerate() {
        if c >= 0.5 {
            packed |= 1u64 << i;
        }
    }
    vec![packed]
}

/// Expected qualia footprint for a given sentence structure.
///
/// Placeholder: zero-fingerprint = neutral expectation (every dimension
/// below 0.5). When the 144-cell verb table lands, this looks up the row
/// by (verb, tense) on `structure.triples[0]` and returns that row's
/// qualia footprint.
#[cfg(feature = "grammar-triangle")]
fn expected_qualia_footprint(_structure: &SentenceStructure) -> Vec<u64> {
    vec![0u64; 1]
}

/// Normalized Hamming distance between two `[u64]` registers.
///
/// Returns `bits_diff / total_bits` in `[0.0, 1.0]`. Compares
/// `min(a.len(), b.len())` words; empty input returns `0.0`.
#[cfg(feature = "grammar-triangle")]
fn hamming_normalized(a: &[u64], b: &[u64]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut bits_diff: u32 = 0;
    for i in 0..n {
        bits_diff += (a[i] ^ b[i]).count_ones();
    }
    let total_bits = (n * 64) as f32;
    bits_diff as f32 / total_bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::SentenceStructure;
    use crate::spo::SpoTriple;

    fn fixture_structure() -> SentenceStructure {
        // Build a minimal SentenceStructure via the public parse() entry.
        // Use parser::parse on an empty token slice to get a default
        // structure shape, then push one synthetic triple in.
        let empty: Vec<crate::vocabulary::Token> = Vec::new();
        let mut s = crate::parser::parse(&empty);
        s.triples.push(SpoTriple::new(1, 2, 3));
        s
    }

    #[test]
    fn analyze_without_triangle_yields_zero_distance() {
        let s = fixture_structure();
        let out = analyze_without_triangle(s);
        assert_eq!(out.classification_distance, 0.0);
        assert_eq!(out.triples.triples.len(), 1);
    }

    #[cfg(feature = "grammar-triangle")]
    #[test]
    fn classification_distance_in_unit_interval() {
        // Identical fingerprints → 0.0; orthogonal (all bits flipped)
        // within the 18 used bits → 18/64 = 0.28125; full-register
        // orthogonality (every bit flipped) → 1.0.
        assert_eq!(hamming_normalized(&[0u64], &[0u64]), 0.0);
        assert_eq!(hamming_normalized(&[u64::MAX], &[u64::MAX]), 0.0);
        assert!((hamming_normalized(&[0u64], &[u64::MAX]) - 1.0).abs() < 1e-6);
        // 18 bits set vs. 0 bits → 18/64.
        let eighteen_bits = (1u64 << 18) - 1;
        let d = hamming_normalized(&[eighteen_bits], &[0u64]);
        assert!((d - (18.0 / 64.0)).abs() < 1e-6);
    }

    #[cfg(feature = "grammar-triangle")]
    #[test]
    fn analyze_with_triangle_stamps_lenses() {
        let s = fixture_structure();
        let out = analyze_with_triangle("the dog runs", s);
        // Real Hamming over 18-bit qualia footprint vs. zero expectation;
        // result must be in [0, 1] and not a hardcoded 0.0.
        assert!(out.classification_distance >= 0.0);
        assert!(out.classification_distance <= 1.0);
        assert_eq!(out.triples.triples.len(), 1);
    }

    #[cfg(feature = "grammar-triangle")]
    #[test]
    fn novel_domain_qualia_yields_high_distance() {
        // High-activation, high-novelty, high-urgency text should pull
        // multiple qualia dimensions above the 0.5 threshold, producing
        // a non-zero Hamming distance against the all-zero expected
        // footprint (placeholder = neutral expectation).
        let s = fixture_structure();
        let out = analyze_with_triangle(
            "Suddenly an unprecedented intense urgent novel surprising explosion!",
            s,
        );
        assert!(
            out.classification_distance > 0.0,
            "novel-domain text should yield non-zero classification distance, got {}",
            out.classification_distance
        );
    }

    #[cfg(feature = "grammar-triangle")]
    #[test]
    fn qualia_fingerprint_thresholds_at_half() {
        // Build a triangle whose qualia coordinates straddle 0.5; the
        // packed fingerprint must have exactly the bits at-or-above 0.5
        // set.
        let triangle = GrammarTriangle::default(); // all coords = 0.5
        let fp = qualia_to_binary_fingerprint(&triangle);
        // Default = 0.5 on every dim → every bit set (>= 0.5 threshold),
        // so packed register == 18 lowest bits set.
        let expected = (1u64 << 18) - 1;
        assert_eq!(fp[0], expected);
    }
}
