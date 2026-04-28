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

/// Hamming-style distance between the SPO's expected qualia footprint
/// and the Triangle's actual qualia signature.
///
/// **Stub**: returns 0.0 today. The expected footprint is currently a
/// fixed prior; once D7 GrammarStyleConfig surfaces per-style qualia
/// expectations, this lookup becomes "compare actual qualia to the
/// style-specific footprint and emit a normalized distance."
///
/// FOLLOW-UP: tune against the Jirak-derived noise floor (see CLAUDE.md
/// §I-NOISE-FLOOR-JIRAK) — values that exceed the n^(-1/2) weak-dependence
/// bound are real signal, not register noise.
#[cfg(feature = "grammar-triangle")]
fn compute_classification_distance(
    _structure: &SentenceStructure,
    _triangle: &GrammarTriangle,
) -> f32 {
    0.0
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
    fn analyze_with_triangle_stamps_lenses() {
        let s = fixture_structure();
        let out = analyze_with_triangle("the dog runs", s);
        // Stub returns 0.0 today — until D7 footprint lookup lands.
        assert_eq!(out.classification_distance, 0.0);
        assert_eq!(out.triples.triples.len(), 1);
    }
}
