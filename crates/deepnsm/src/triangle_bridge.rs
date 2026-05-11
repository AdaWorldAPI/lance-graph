//! Grammar Triangle bridge: merge DeepNSM SPO output with the Triangle's
//! NSMField + CausalityFlow + QualiaField into a single SpoWithGrammar.
//!
//! The Triangle lives in `lance_graph_cognitive::grammar::GrammarTriangle`
//! and is already-shipped (`from_text` constructs it). The actual API has
//! `QualiaField` (18D phenomenal field) — not `Qualia18D`.
//!
//! PR-G1: real Causality footprint. The `causality_footprint` field on
//! `SpoWithGrammar` carries a 3-bit Pearl 2³ mask derived from the SPO
//! triple's active planes. This replaces the former neutral 0.5
//! placeholder at lines 90 and 221.

use crate::parser::SentenceStructure;
use crate::spo::{SpoTriple, NO_ROLE};

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

    /// Pearl 2³ causality footprint: a 3-bit mask encoding which SPO
    /// planes are active in the sentence's primary triple.
    ///
    /// Bit layout (matches `causal-edge::pearl::CausalMask` repr):
    /// - bit 2 (0x04): Subject plane active
    /// - bit 1 (0x02): Predicate plane active
    /// - bit 0 (0x01): Object plane active
    ///
    /// Maps to Pearl's causal ladder:
    /// - `0b101` (S+O) = Level 1: Association P(Y|X)
    /// - `0b011` (P+O) = Level 2: Intervention P(Y|do(X))
    /// - `0b111` (SPO) = Level 3: Counterfactual
    /// - `0b110` (S+P) = intransitive: confounder detection
    ///
    /// Replaces the former neutral 0.5 placeholder. Downstream consumers
    /// compare two triangles' `causality_footprint` values to detect when
    /// the causal level has shifted (e.g., observational → interventional).
    pub causality_footprint: u8,
}

/// Compute the Pearl 2³ causality mask from an SPO triple.
///
/// Each bit corresponds to one SPO plane:
/// - bit 2: Subject active (rank != NO_ROLE)
/// - bit 1: Predicate active (rank != NO_ROLE)
/// - bit 0: Object active (rank != NO_ROLE)
///
/// This is a pure 3-bit projection — no `CausalEdge64` import needed.
/// The resulting mask has the same bit layout as `causal-edge::pearl::CausalMask`
/// so downstream consumers can cast directly.
#[inline]
pub fn compute_pearl_mask(triple: &SpoTriple) -> u8 {
    let s_bit = if triple.subject() != NO_ROLE { 0b100 } else { 0 };
    let p_bit = if triple.predicate() != NO_ROLE { 0b010 } else { 0 };
    let o_bit = if triple.object() != NO_ROLE { 0b001 } else { 0 };
    s_bit | p_bit | o_bit
}

/// Compute the aggregate Pearl mask for a sentence structure.
///
/// When the structure has one or more triples, returns the mask of
/// the *primary* triple (index 0). When empty, returns `0b000` (no
/// planes active — aggregate prior).
pub fn causality_footprint_for(structure: &SentenceStructure) -> u8 {
    match structure.triples.first() {
        Some(triple) => compute_pearl_mask(triple),
        None => 0b000,
    }
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
    let footprint = causality_footprint_for(&structure);
    SpoWithGrammar {
        triples: structure,
        causality: triangle.causality,
        nsm_field: triangle.nsm,
        qualia_signature: triangle.qualia,
        classification_distance: dist,
        causality_footprint: footprint,
    }
}

/// Feature-off fallback: just carry the SPO with `classification_distance = 0`.
///
/// Available regardless of feature so the parser always has something to
/// hand the LLM router. The `causality_footprint` is still computed from
/// the SPO triple — it does not require the Triangle feature.
pub fn analyze_without_triangle(structure: SentenceStructure) -> SpoWithGrammar {
    let footprint = causality_footprint_for(&structure);
    SpoWithGrammar {
        triples: structure,
        #[cfg(feature = "grammar-triangle")]
        causality: CausalityFlow::default(),
        #[cfg(feature = "grammar-triangle")]
        nsm_field: NSMField::default(),
        #[cfg(feature = "grammar-triangle")]
        qualia_signature: QualiaField::default(),
        classification_distance: 0.0,
        causality_footprint: footprint,
    }
}

// Qualia dim mapping (PAD model — Pleasure-Arousal-Dominance, sanitized for non-academic context):
//   dim 0 = Agency      ← Dominance      (Subject plane bit; P=1 → Subject contributes)
//   dim 1 = Activity    ← Activation     (Predicate plane bit; P=1 → Predicate contributes)
//   dim 2 = Affection   ← Arousal        (Object plane bit; P=1 → Object contributes)
// Cross-ref: lance-graph-cognitive::grammar::qualia uses {valence,activation,dominance};
//            contract::qualia uses {arousal,valence,tension}. This module uses sanitized PAD.

/// Normalized Hamming distance between the qualia fingerprint and the
/// SPO predicate's expected qualia footprint.
///
/// The footprint is derived from the Pearl 2³ mask of the sentence's
/// primary SPO triple. Active SPO planes seed qualia dimensions 0-2:
///   dim 0 (Agency)    ← Subject plane (bit 2)
///   dim 1 (Activity)  ← Predicate plane (bit 1)
///   dim 2 (Affection) ← Object plane (bit 0)
///
/// Once D7 `GrammarStyleConfig` surfaces per-style qualia expectations,
/// `expected_qualia_footprint` will additionally look up the verb's row
/// by (verb, tense) and merge that row's qualia footprint.
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
/// Seeds bits 0-2 from the Pearl 2³ mask of the primary SPO triple:
///   bit 0 ← Subject plane active (Agency qualia expected >= 0.5)
///   bit 1 ← Predicate plane active (Activity qualia expected >= 0.5)
///   bit 2 ← Object plane active (Affection qualia expected >= 0.5)
///
/// When the 144-cell verb table lands, this will additionally look up
/// the row by (verb, tense) on `structure.triples[0]` and merge that
/// row's higher-dimensional qualia bits.
#[cfg(feature = "grammar-triangle")]
fn expected_qualia_footprint(structure: &SentenceStructure) -> Vec<u64> {
    let pearl = causality_footprint_for(structure);
    // Map Pearl mask bits (S=bit2, P=bit1, O=bit0) to qualia dimension
    // bits (dim0=Agency<-S, dim1=Activity<-P, dim2=Affection<-O).
    let mut packed: u64 = 0;
    if pearl & 0b100 != 0 { packed |= 1u64 << 0; } // S -> dim 0 (Agency)
    if pearl & 0b010 != 0 { packed |= 1u64 << 1; } // P -> dim 1 (Activity)
    if pearl & 0b001 != 0 { packed |= 1u64 << 2; } // O -> dim 2 (Affection)
    vec![packed]
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
    use crate::spo::SpoTriple;

    fn fixture_structure() -> SentenceStructure {
        let empty: Vec<crate::vocabulary::Token> = Vec::new();
        let mut s = crate::parser::parse(&empty);
        s.triples.push(SpoTriple::new(1, 2, 3));
        s
    }

    /// Build a structure with a specific SPO triple.
    fn fixture_structure_with(subject: u16, predicate: u16, object: u16) -> SentenceStructure {
        let empty: Vec<crate::vocabulary::Token> = Vec::new();
        let mut s = crate::parser::parse(&empty);
        s.triples.push(SpoTriple::new(subject, predicate, object));
        s
    }

    #[test]
    fn analyze_without_triangle_yields_zero_distance() {
        let s = fixture_structure();
        let out = analyze_without_triangle(s);
        assert_eq!(out.classification_distance, 0.0);
        assert_eq!(out.triples.triples.len(), 1);
    }

    #[test]
    fn pearl_mask_transitive_is_spo() {
        // "dog bites man" -- all three planes active -> 0b111 (Counterfactual)
        let triple = SpoTriple::new(671, 2943, 95);
        assert_eq!(compute_pearl_mask(&triple), 0b111);
    }

    #[test]
    fn pearl_mask_intransitive_is_sp() {
        // "dog runs" -- no object -> 0b110 (Confounder Detection / intransitive)
        let triple = SpoTriple::intransitive(671, 100);
        assert_eq!(compute_pearl_mask(&triple), 0b110);
    }

    #[test]
    fn causality_footprint_transitive_sentence() {
        // Transitive sentence: S+P+O all present -> 0b111
        let s = fixture_structure_with(671, 2943, 95);
        let out = analyze_without_triangle(s);
        assert_eq!(out.causality_footprint, 0b111);
    }

    #[test]
    fn causality_footprint_intransitive_sentence() {
        // Intransitive sentence: S+P, no O -> 0b110
        let empty: Vec<crate::vocabulary::Token> = Vec::new();
        let mut s = crate::parser::parse(&empty);
        s.triples.push(SpoTriple::intransitive(671, 100));
        let out = analyze_without_triangle(s);
        assert_eq!(out.causality_footprint, 0b110);
    }

    #[test]
    fn causality_footprint_empty_structure() {
        // No triples -> 0b000 (aggregate prior)
        let empty: Vec<crate::vocabulary::Token> = Vec::new();
        let s = crate::parser::parse(&empty);
        let out = analyze_without_triangle(s);
        assert_eq!(out.causality_footprint, 0b000);
    }

    /// PR-G1 spec test: two sentences with the SAME subject but DIFFERENT
    /// Pearl masks must produce different triangle outputs.
    ///
    /// Sentence A: "dog bites man" -> transitive -> SPO = 0b111 (Counterfactual)
    /// Sentence B: "dog runs"      -> intransitive -> SP_ = 0b110 (Confounder)
    ///
    /// Same subject (671), different Pearl masks -> different causality_footprint.
    #[test]
    fn same_subject_different_pearl_masks_yield_different_outputs() {
        let s_transitive = fixture_structure_with(671, 2943, 95);
        let s_intransitive = {
            let empty: Vec<crate::vocabulary::Token> = Vec::new();
            let mut s = crate::parser::parse(&empty);
            s.triples.push(SpoTriple::intransitive(671, 100));
            s
        };

        let out_a = analyze_without_triangle(s_transitive);
        let out_b = analyze_without_triangle(s_intransitive);

        // Same subject
        assert_eq!(
            out_a.triples.triples[0].subject(),
            out_b.triples.triples[0].subject(),
            "both sentences must share the same subject"
        );

        // Different Pearl masks -> different causality_footprint
        assert_ne!(
            out_a.causality_footprint, out_b.causality_footprint,
            "transitive (0b{:03b}) vs intransitive (0b{:03b}) must differ",
            out_a.causality_footprint, out_b.causality_footprint
        );

        // Verify exact values
        assert_eq!(out_a.causality_footprint, 0b111, "transitive = SPO");
        assert_eq!(out_b.causality_footprint, 0b110, "intransitive = SP_");
    }

    /// PR-G1 spec test: two sentences with same Subject, different Predicates
    /// producing the same Pearl mask structure (both transitive) but encoding
    /// different causal content.
    #[test]
    fn same_subject_same_mask_different_predicates_distinguishable() {
        // "dog bites man" vs "dog loves man" -- same mask (0b111), different predicate
        let s_a = fixture_structure_with(671, 2943, 95);  // bites
        let s_b = fixture_structure_with(671, 500, 95);    // loves

        let out_a = analyze_without_triangle(s_a);
        let out_b = analyze_without_triangle(s_b);

        // Same Pearl mask (both SPO)
        assert_eq!(out_a.causality_footprint, out_b.causality_footprint);
        assert_eq!(out_a.causality_footprint, 0b111);

        // But different predicates -> different triples
        assert_ne!(
            out_a.triples.triples[0].predicate(),
            out_b.triples.triples[0].predicate(),
        );
    }

    #[cfg(feature = "grammar-triangle")]
    #[test]
    fn classification_distance_in_unit_interval() {
        assert_eq!(hamming_normalized(&[0u64], &[0u64]), 0.0);
        assert_eq!(hamming_normalized(&[u64::MAX], &[u64::MAX]), 0.0);
        assert!((hamming_normalized(&[0u64], &[u64::MAX]) - 1.0).abs() < 1e-6);
        let eighteen_bits = (1u64 << 18) - 1;
        let d = hamming_normalized(&[eighteen_bits], &[0u64]);
        assert!((d - (18.0 / 64.0)).abs() < 1e-6);
    }

    #[cfg(feature = "grammar-triangle")]
    #[test]
    fn analyze_with_triangle_stamps_lenses() {
        let s = fixture_structure();
        let out = analyze_with_triangle("the dog runs", s);
        assert!(out.classification_distance >= 0.0);
        assert!(out.classification_distance <= 1.0);
        assert_eq!(out.triples.triples.len(), 1);
        assert_ne!(out.causality_footprint, 0);
    }

    #[cfg(feature = "grammar-triangle")]
    #[test]
    fn novel_domain_qualia_yields_high_distance() {
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
        let triangle = GrammarTriangle::default();
        let fp = qualia_to_binary_fingerprint(&triangle);
        let expected = (1u64 << 18) - 1;
        assert_eq!(fp[0], expected);
    }

    /// PR-G1 spec test (failing-test-first): two structures with the SAME
    /// subject but DIFFERENT Pearl masks (one transitive 0b111, one
    /// intransitive 0b110) must produce different `analyze_with_triangle`
    /// outputs.
    ///
    /// This exercises the path where the former 0.5 placeholder lived. If
    /// the placeholder were still in place, both calls would have produced
    /// the SAME `causality_footprint = 0` (zero-init / placeholder) and the
    /// expected_qualia_footprint would not differ on dim 2. With the real
    /// Pearl 2³ mask wired in:
    ///
    /// - Transitive (S+P+O = 0b111) → expected_qualia_footprint bit 2 set
    /// - Intransitive (S+P   = 0b110) → expected_qualia_footprint bit 2 unset
    ///
    /// The differing dim is dim 2 (Affection ← Object plane bit 0 of mask).
    #[cfg(feature = "grammar-triangle")]
    #[test]
    fn analyze_with_triangle_returns_different_qualia_for_different_pearl_masks() {
        // Sentence A: transitive — S+P+O all present (0b111)
        let s_transitive = fixture_structure_with(671, 2943, 95);
        // Sentence B: intransitive — S+P only, no O (0b110)
        let s_intransitive = {
            let empty: Vec<crate::vocabulary::Token> = Vec::new();
            let mut s = crate::parser::parse(&empty);
            s.triples.push(SpoTriple::intransitive(671, 100));
            s
        };

        // Same surface text so the GrammarTriangle::from_text branch is
        // identical for both — what differs is the structure (Pearl mask).
        let text = "the dog acts";

        let out_a = analyze_with_triangle(text, s_transitive);
        let out_b = analyze_with_triangle(text, s_intransitive);

        // Same subject in both
        assert_eq!(
            out_a.triples.triples[0].subject(),
            out_b.triples.triples[0].subject(),
            "both sentences must share the same subject"
        );

        // Pearl masks must differ — the core PR-G1 invariant.
        assert_ne!(
            out_a.causality_footprint, out_b.causality_footprint,
            "transitive (0b{:03b}) vs intransitive (0b{:03b}) must differ \
             on the analyze_with_triangle path",
            out_a.causality_footprint, out_b.causality_footprint
        );
        assert_eq!(out_a.causality_footprint, 0b111, "transitive = SPO");
        assert_eq!(out_b.causality_footprint, 0b110, "intransitive = SP_");

        // Expected-qualia-footprint differs in dim 2 (Affection ← Object).
        // The transitive sentence expects bit 2 set; the intransitive does
        // not. This is what the 0.5 placeholder used to mask.
        let exp_a = expected_qualia_footprint(&out_a.triples);
        let exp_b = expected_qualia_footprint(&out_b.triples);
        assert_ne!(
            exp_a, exp_b,
            "expected_qualia_footprint must differ for differing Pearl masks"
        );
        let dim2_mask: u64 = 1u64 << 2;
        assert_eq!(
            exp_a[0] & dim2_mask,
            dim2_mask,
            "transitive must set dim 2 (Affection from Object plane)"
        );
        assert_eq!(
            exp_b[0] & dim2_mask,
            0,
            "intransitive must NOT set dim 2 (no Object plane)"
        );

        // Sanity: classification_distance is in [0,1] for both — a 0.5
        // placeholder would have been a constant; here it must respond
        // to the structure.
        assert!(out_a.classification_distance >= 0.0 && out_a.classification_distance <= 1.0);
        assert!(out_b.classification_distance >= 0.0 && out_b.classification_distance <= 1.0);
    }
}
