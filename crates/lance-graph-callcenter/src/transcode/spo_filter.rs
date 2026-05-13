//! SQL filter ↔ SPO triple-lookup translator.
//!
//! Given a flat list of `(column, op, literal)` terms — what DataFusion
//! pushdown hands us — produce an [`SpoLookup`] the SPO store can
//! evaluate. This is the **read-side bridge** between the outer
//! ontology's SQL surface and the inner ontology's triple store.
//!
//! Domain-agnostic: any [`Ontology`] resolves entity_type names through
//! [`entity_type_id`]; predicate names hash to fingerprints through the
//! contract's canonical [`fnv1a`] (so the encode and decode sides agree
//! without sharing a code-loaded codebook).

use lance_graph_contract::hash::fnv1a;
use lance_graph_contract::ontology::{entity_type_id, EntityTypeId, Ontology};

/// Stable u64 fingerprint of a predicate string.
pub fn predicate_fingerprint(predicate: &str) -> u64 {
    fnv1a(predicate.as_bytes())
}

/// Op codes the bridge translates today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Eq,
    NotEq,
    Gt,
    Gte,
    Lt,
    Lte,
}

/// Single literal-typed comparison: `column op literal`.
#[derive(Debug, Clone)]
pub struct FilterTerm {
    pub column: String,
    pub op: Op,
    pub literal: Literal,
}

#[derive(Debug, Clone)]
pub enum Literal {
    Utf8(String),
    UInt64(u64),
    Float32(f32),
}

/// Resolved lookup against the inner-ontology SPO store. Carries every-
/// thing the store needs to evaluate a single-table filter; nothing it
/// doesn't.
#[derive(Debug, Clone, Default)]
pub struct SpoLookup {
    pub entity_type_id: Option<EntityTypeId>,
    pub predicate_fp: Option<u64>,
    pub predicate_fp_excluded: Option<u64>,
    pub min_frequency: Option<f32>,
    pub min_confidence: Option<f32>,
    pub entity_id: Option<u64>,
}

/// Translates `FilterTerm`s into an `SpoLookup`. Unknown columns are
/// **silently left as residual** — the table provider hands them back to
/// DataFusion. This keeps the bridge's surface tight and avoids silent
/// over-rejection.
#[derive(Debug)]
pub struct SpoFilterTranslator<'a> {
    ontology: &'a Ontology,
}

impl<'a> SpoFilterTranslator<'a> {
    pub fn new(ontology: &'a Ontology) -> Self {
        Self { ontology }
    }

    pub fn translate(&self, terms: &[FilterTerm]) -> SpoLookup {
        let mut out = SpoLookup::default();
        for t in terms {
            match (t.column.as_str(), t.op, &t.literal) {
                ("entity_type", Op::Eq, Literal::Utf8(s)) => {
                    let id = entity_type_id(self.ontology, s);
                    if id != 0 {
                        out.entity_type_id = Some(id);
                    }
                }
                ("entity_id", Op::Eq, Literal::UInt64(n)) => {
                    out.entity_id = Some(*n);
                }
                ("predicate", Op::Eq, Literal::Utf8(s)) => {
                    out.predicate_fp = Some(predicate_fingerprint(s));
                }
                ("predicate", Op::NotEq, Literal::Utf8(s)) => {
                    out.predicate_fp_excluded = Some(predicate_fingerprint(s));
                }
                ("nars_frequency", Op::Gt | Op::Gte, Literal::Float32(x)) => {
                    out.min_frequency = Some(*x);
                }
                ("nars_confidence", Op::Gt | Op::Gte, Literal::Float32(x)) => {
                    out.min_confidence = Some(*x);
                }
                _ => {}
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::property::Schema;

    fn ontology_with_patient() -> Ontology {
        Ontology::builder("Test")
            .schema(Schema::builder("Patient").required("name").build())
            .build()
    }

    #[test]
    fn translates_entity_type_eq_to_id() {
        let ont = ontology_with_patient();
        let look = SpoFilterTranslator::new(&ont).translate(&[FilterTerm {
            column: "entity_type".into(),
            op: Op::Eq,
            literal: Literal::Utf8("Patient".into()),
        }]);
        assert_eq!(look.entity_type_id, Some(1));
    }

    #[test]
    fn unknown_entity_type_eq_drops_to_none() {
        let ont = ontology_with_patient();
        let look = SpoFilterTranslator::new(&ont).translate(&[FilterTerm {
            column: "entity_type".into(),
            op: Op::Eq,
            literal: Literal::Utf8("Unknown".into()),
        }]);
        assert!(look.entity_type_id.is_none());
    }

    #[test]
    fn predicate_fingerprint_uses_canonical_fnv1a() {
        let h = predicate_fingerprint("name");
        assert_eq!(h, fnv1a(b"name"));
    }

    #[test]
    fn nars_frequency_gt_lifts_threshold() {
        let ont = ontology_with_patient();
        let look = SpoFilterTranslator::new(&ont).translate(&[FilterTerm {
            column: "nars_frequency".into(),
            op: Op::Gt,
            literal: Literal::Float32(0.7),
        }]);
        assert_eq!(look.min_frequency, Some(0.7));
    }

    #[test]
    fn unrecognised_terms_silently_become_residual() {
        let ont = ontology_with_patient();
        let look = SpoFilterTranslator::new(&ont).translate(&[FilterTerm {
            column: "weird".into(),
            op: Op::Eq,
            literal: Literal::Utf8("x".into()),
        }]);
        assert!(look.entity_type_id.is_none());
        assert!(look.predicate_fp.is_none());
    }
}
