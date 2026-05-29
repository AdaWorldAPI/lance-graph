// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Action guard/effect emitter — composes SPO ontology edges per function into
//! Foundry-shape `ActionSpec` records (`requires{}` / `effects{}` blocks).
//!
//! # Why this exists
//!
//! The SPO ontology stores the Foundry-shape graph as triples:
//!
//! - `(field, emitted_by, function)` — which function writes this field
//! - `(field, depends_on, dep)` — which fields this field needs as input
//! - `(function, raises, exc:Type)` — which exceptions this function raises
//! - `(function, reads_field, field)` — which fields this function reads
//! - `(function, traverses_relation, rel)` — which relations this function walks
//!
//! Foundry's Action model needs these composed into a single record per
//! function: **what guards must hold** (`requires`) and **what state mutates**
//! (`effects`), plus the dependency closure that drives recompute.
//!
//! This module is the deterministic compose step. Input: parsed triples.
//! Output: one [`ActionSpec`] per function, suitable for downstream askama
//! emitters (per-bucket SoC templates) or direct Elixir-surface dispatch.
//!
//! # Iron rule
//!
//! No similarity, no LLM, no inference. Set indexing + reverse lookup over
//! the triple set. The mapping is a graph projection, gate-able by
//! `codegen_spine::TripletProjection::roundtrip_eq` (the action_emitter's
//! output is lossy — it drops triple-level truth values — but the
//! `(function, field)` identity set must round-trip).
//!
//! # Provenance
//!
//! Consumes the output of `parse_triples(odoo_ontology.spo.ndjson)`
//! (see `odoo_ontology.rs`). 3 328 functions in the shipped data file
//! (per the module-level docstring of `odoo_ontology.rs`) → 3 328
//! `ActionSpec` records expected.

use std::collections::{BTreeMap, BTreeSet};

use super::odoo_ontology::OntologyTriple;

// ---------------------------------------------------------------------------
// ActionSpec — the Foundry-shape per-function record
// ---------------------------------------------------------------------------

/// One Foundry-shape Action specification, composed from SPO triples for a
/// single function subject (e.g. `odoo:account_move._compute_amount`).
///
/// Fields are sorted (BTreeSet → Vec) for deterministic output: two emit
/// runs over the same triples produce byte-identical specs. Identity sets
/// only — truth values from the source triples are not preserved at this
/// layer (the action spec is a coarser projection; round-trip equality
/// holds only at the `(function, field)` identity level).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ActionSpec {
    /// Fully-qualified function IRI (e.g. `odoo:account_move._compute_amount`).
    pub id: String,
    /// Family the function belongs to (`account_move` for
    /// `odoo:account_move._compute_amount`).
    pub family: String,
    /// Fields this function writes (sourced from reverse `emitted_by`).
    pub effects: Vec<String>,
    /// Direct dependency fields of every effect field (sourced from forward
    /// `depends_on` of each effect). Transitive closure NOT taken — the
    /// dependency graph is already the source of truth for transitivity.
    pub inputs: Vec<String>,
    /// Exception types this function raises (sourced from forward `raises`).
    /// These are the guard signals — a `requires{}` block in Foundry terms.
    pub raises: Vec<String>,
    /// Fields this function reads in its body (sourced from forward
    /// `reads_field`). Distinct from `inputs`: `inputs` is the
    /// `@api.depends`-declared dependency set, `reads` is body-inferred.
    pub reads: Vec<String>,
    /// Relations this function traverses (sourced from forward
    /// `traverses_relation`).
    pub traverses: Vec<String>,
}

impl ActionSpec {
    /// Whether this action has any structural content — false if the function
    /// has no effects, no raises, no reads, and no traversals (a degenerate
    /// node, typically a method body the harvester couldn't decode).
    #[must_use]
    pub fn is_trivial(&self) -> bool {
        self.effects.is_empty()
            && self.raises.is_empty()
            && self.reads.is_empty()
            && self.traverses.is_empty()
    }

    /// Whether this action behaves as a pure guard (raises but writes no
    /// fields). Maps to a Foundry "validation only" action.
    #[must_use]
    pub fn is_pure_guard(&self) -> bool {
        !self.raises.is_empty() && self.effects.is_empty()
    }

    /// Whether this action behaves as a pure compute (writes fields but
    /// raises nothing). Maps to a Foundry "derive only" action.
    #[must_use]
    pub fn is_pure_compute(&self) -> bool {
        self.raises.is_empty() && !self.effects.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Emitter — the deterministic projection from triples to ActionSpec set
// ---------------------------------------------------------------------------

/// Compose [`ActionSpec`] records from a parsed triple set.
///
/// One spec per function (subject of `(function, rdf:type, ogit:Function)`).
/// All edges are indexed in a single pass; per-function lookups are then
/// constant-time on the indices. Total cost O(|triples| log |triples|) from
/// the BTreeSet sort.
///
/// # Determinism
///
/// All output collections (effects, inputs, raises, reads, traverses) are
/// sorted ascending. Two runs over the same input produce identical output
/// (byte-identical when serialised). This is the contract the codegen_spine
/// `TripletProjection::roundtrip_eq` gate expects.
///
/// # Non-functions are skipped
///
/// Only subjects with `(s, rdf:type, ogit:Function)` produce a spec.
/// Object types and properties have their own emitters (the widget emitter
/// per family, the column emitter per property).
#[must_use]
pub fn emit_actions(triples: &[OntologyTriple]) -> Vec<ActionSpec> {
    let idx = TripleIndex::build(triples);
    let mut specs: Vec<ActionSpec> = idx
        .functions
        .iter()
        .map(|fn_id| compose_spec(fn_id, &idx))
        .collect();
    specs.sort_by(|a, b| a.id.cmp(&b.id));
    specs
}

/// Filter [`emit_actions`] output to non-trivial specs only.
///
/// Trivial = no effects, no raises, no reads, no traversals (see
/// [`ActionSpec::is_trivial`]). Useful when downstream codegen wants to
/// skip stub methods the harvester couldn't decode.
#[must_use]
pub fn emit_non_trivial_actions(triples: &[OntologyTriple]) -> Vec<ActionSpec> {
    emit_actions(triples)
        .into_iter()
        .filter(|s| !s.is_trivial())
        .collect()
}

// ---------------------------------------------------------------------------
// Internal: triple indexing
// ---------------------------------------------------------------------------

/// Indexed view of the triple set — built once per emit_actions call.
///
/// Edges are stored as `BTreeMap<subject, BTreeSet<object>>` (forward) or
/// `BTreeMap<object, BTreeSet<subject>>` (reverse), keyed by predicate kind.
/// The function-id set is a `BTreeSet<String>` for sorted, deduplicated
/// iteration.
struct TripleIndex {
    /// Set of function IRIs (subjects of `rdf:type ogit:Function`).
    functions: BTreeSet<String>,
    /// `(function → fields)` map from reverse `emitted_by` (field-S, fn-O →
    /// indexed by fn-O for lookup).
    emits_by_fn: BTreeMap<String, BTreeSet<String>>,
    /// `(field → deps)` map from forward `depends_on`.
    deps_by_field: BTreeMap<String, BTreeSet<String>>,
    /// `(function → exceptions)` map from forward `raises`.
    raises_by_fn: BTreeMap<String, BTreeSet<String>>,
    /// `(function → fields-read)` map from forward `reads_field`.
    reads_by_fn: BTreeMap<String, BTreeSet<String>>,
    /// `(function → relations)` map from forward `traverses_relation`.
    traverses_by_fn: BTreeMap<String, BTreeSet<String>>,
}

impl TripleIndex {
    fn build(triples: &[OntologyTriple]) -> Self {
        let mut functions: BTreeSet<String> = BTreeSet::new();
        let mut emits_by_fn: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut deps_by_field: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut raises_by_fn: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut reads_by_fn: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut traverses_by_fn: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

        for t in triples {
            match t.p.as_str() {
                "rdf:type" if t.o == "ogit:Function" => {
                    functions.insert(t.s.clone());
                }
                "emitted_by" => {
                    emits_by_fn
                        .entry(t.o.clone())
                        .or_default()
                        .insert(t.s.clone());
                }
                "depends_on" => {
                    deps_by_field
                        .entry(t.s.clone())
                        .or_default()
                        .insert(t.o.clone());
                }
                "raises" => {
                    raises_by_fn
                        .entry(t.s.clone())
                        .or_default()
                        .insert(t.o.clone());
                }
                "reads_field" => {
                    reads_by_fn
                        .entry(t.s.clone())
                        .or_default()
                        .insert(t.o.clone());
                }
                "traverses_relation" => {
                    traverses_by_fn
                        .entry(t.s.clone())
                        .or_default()
                        .insert(t.o.clone());
                }
                _ => {}
            }
        }

        Self {
            functions,
            emits_by_fn,
            deps_by_field,
            raises_by_fn,
            reads_by_fn,
            traverses_by_fn,
        }
    }
}

fn compose_spec(fn_id: &str, idx: &TripleIndex) -> ActionSpec {
    let family = family_of(fn_id);

    // `effects` must materialize as a Vec at the end, but we also iterate it
    // to drive `inputs` below — borrow the source set, then collect once.
    let effects_set: &BTreeSet<String> = idx.emits_by_fn.get(fn_id).unwrap_or(&EMPTY_SET);

    let mut inputs: BTreeSet<String> = BTreeSet::new();
    for effect in effects_set {
        if let Some(deps) = idx.deps_by_field.get(effect) {
            inputs.extend(deps.iter().cloned());
        }
    }

    ActionSpec {
        id: fn_id.to_string(),
        family,
        effects: collect_sorted(idx.emits_by_fn.get(fn_id)),
        inputs: inputs.into_iter().collect(),
        raises: collect_sorted(idx.raises_by_fn.get(fn_id)),
        reads: collect_sorted(idx.reads_by_fn.get(fn_id)),
        traverses: collect_sorted(idx.traverses_by_fn.get(fn_id)),
    }
}

/// Collect an optional `BTreeSet` reference into an ascending `Vec`, cloning
/// each element once (vs `.cloned().unwrap_or_default()` which clones the
/// entire tree before re-collecting).
fn collect_sorted(set: Option<&BTreeSet<String>>) -> Vec<String> {
    set.map(|s| s.iter().cloned().collect()).unwrap_or_default()
}

/// Singleton empty set so `compose_spec` can borrow a reference for the
/// missing-key path without allocating per call.
static EMPTY_SET: BTreeSet<String> = BTreeSet::new();

/// Extract the family from a function IRI: `odoo:account_move._compute_amount`
/// → `account_move`. Returns the IRI itself (sans `odoo:` prefix) if the
/// format does not match (defensive — the extractor is expected to always
/// produce dotted IRIs).
#[must_use]
fn family_of(fn_id: &str) -> String {
    // Strip `odoo:` prefix if present; then take the segment before the first dot.
    let after_prefix = fn_id.strip_prefix("odoo:").unwrap_or(fn_id);
    match after_prefix.find('.') {
        Some(dot) => after_prefix[..dot].to_string(),
        None => after_prefix.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests — verify the projection on a known fixture + on shipped ontology
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::spo::odoo_ontology::parse_triples;

    /// The shipped Odoo ontology (test-only embed).
    const ONTOLOGY: &str = include_str!("odoo_ontology.spo.ndjson");

    fn triple(s: &str, p: &str, o: &str) -> OntologyTriple {
        OntologyTriple {
            s: s.into(),
            p: p.into(),
            o: o.into(),
            f: 1.0,
            c: 1.0,
        }
    }

    fn fixture() -> Vec<OntologyTriple> {
        vec![
            // _compute_amount is a Function that writes amount_total and amount_residual,
            // depending on line_ids.balance and line_ids.amount_residual; reads currency_id;
            // traverses line_ids; raises UserError.
            triple(
                "odoo:account_move._compute_amount",
                "rdf:type",
                "ogit:Function",
            ),
            triple(
                "odoo:account_move.amount_total",
                "emitted_by",
                "odoo:account_move._compute_amount",
            ),
            triple(
                "odoo:account_move.amount_residual",
                "emitted_by",
                "odoo:account_move._compute_amount",
            ),
            triple(
                "odoo:account_move.amount_total",
                "depends_on",
                "odoo:account_move.line_ids.balance",
            ),
            triple(
                "odoo:account_move.amount_residual",
                "depends_on",
                "odoo:account_move.line_ids.amount_residual",
            ),
            triple(
                "odoo:account_move._compute_amount",
                "reads_field",
                "odoo:account_move.currency_id",
            ),
            triple(
                "odoo:account_move._compute_amount",
                "traverses_relation",
                "odoo:account_move.line_ids",
            ),
            triple(
                "odoo:account_move._compute_amount",
                "raises",
                "exc:UserError",
            ),
            // A pure-guard function: only raises, no emits.
            triple(
                "odoo:account_move._check_balanced",
                "rdf:type",
                "ogit:Function",
            ),
            triple(
                "odoo:account_move._check_balanced",
                "raises",
                "exc:ValidationError",
            ),
        ]
    }

    #[test]
    fn compose_account_move_compute_amount() {
        let specs = emit_actions(&fixture());
        let cm = specs
            .iter()
            .find(|s| s.id == "odoo:account_move._compute_amount")
            .expect("compute_amount spec missing");

        assert_eq!(cm.family, "account_move");
        assert_eq!(
            cm.effects,
            vec![
                "odoo:account_move.amount_residual".to_string(),
                "odoo:account_move.amount_total".to_string(),
            ]
        );
        assert_eq!(
            cm.inputs,
            vec![
                "odoo:account_move.line_ids.amount_residual".to_string(),
                "odoo:account_move.line_ids.balance".to_string(),
            ]
        );
        assert_eq!(cm.raises, vec!["exc:UserError".to_string()]);
        assert_eq!(cm.reads, vec!["odoo:account_move.currency_id".to_string()]);
        assert_eq!(cm.traverses, vec!["odoo:account_move.line_ids".to_string()]);
        assert!(
            !cm.is_pure_compute(),
            "raises non-empty disqualifies pure compute"
        );
        assert!(
            !cm.is_pure_guard(),
            "effects non-empty disqualifies pure guard"
        );
        assert!(!cm.is_trivial());
    }

    #[test]
    fn pure_guard_classification() {
        let specs = emit_actions(&fixture());
        let cb = specs
            .iter()
            .find(|s| s.id == "odoo:account_move._check_balanced")
            .expect("check_balanced spec missing");
        assert!(cb.is_pure_guard());
        assert!(!cb.is_pure_compute());
        assert!(!cb.is_trivial());
    }

    #[test]
    fn output_is_sorted_deterministic() {
        let specs1 = emit_actions(&fixture());
        let specs2 = emit_actions(&fixture());
        assert_eq!(specs1, specs2, "emit_actions must be deterministic");

        // ID order ascending.
        for window in specs1.windows(2) {
            assert!(window[0].id < window[1].id, "specs not sorted by id");
        }
    }

    #[test]
    fn emit_non_trivial_drops_empties() {
        let mut t = fixture();
        // Add a function with no edges at all.
        t.push(triple(
            "odoo:account_move._stub",
            "rdf:type",
            "ogit:Function",
        ));

        let all = emit_actions(&t);
        let non_trivial = emit_non_trivial_actions(&t);

        assert!(
            all.iter().any(|s| s.id == "odoo:account_move._stub"),
            "stub should appear in full output"
        );
        assert!(
            !non_trivial
                .iter()
                .any(|s| s.id == "odoo:account_move._stub"),
            "stub should be filtered from non_trivial output"
        );
    }

    #[test]
    fn shipped_ontology_produces_expected_function_count() {
        let triples = parse_triples(ONTOLOGY);
        let specs = emit_actions(&triples);

        // 3 328 functions per the module-level docstring of `odoo_ontology.rs`
        // (line ~47) — counted at extraction time by `emit_ontology2.py`.
        // The function count equals the number of `(s, rdf:type, ogit:Function)`
        // triples in the data file; this is a stable property of the corpus.
        assert_eq!(
            specs.len(),
            3328,
            "function count drifted from data file extraction"
        );
    }

    #[test]
    fn shipped_ontology_compute_amount_has_real_dependencies() {
        let triples = parse_triples(ONTOLOGY);
        let specs = emit_actions(&triples);

        let cm = specs
            .iter()
            .find(|s| s.id == "odoo:account_move._compute_amount")
            .expect("real compute_amount missing from shipped ontology");

        // Verified in odoo_ontology.rs::emitted_by_edge_is_present:
        // account_move.amount_total emitted_by _compute_amount.
        assert!(
            cm.effects
                .iter()
                .any(|e| e == "odoo:account_move.amount_total"),
            "amount_total must be in _compute_amount.effects"
        );
        // The compute pulls from line_ids — verify at least one input exists.
        assert!(
            !cm.inputs.is_empty(),
            "_compute_amount must have non-empty dependency closure"
        );
    }

    #[test]
    fn family_of_handles_dotted_iri() {
        assert_eq!(
            family_of("odoo:account_move._compute_amount"),
            "account_move"
        );
        assert_eq!(family_of("odoo:res_partner.name"), "res_partner");
        assert_eq!(family_of("odoo:standalone"), "standalone");
        // No `odoo:` prefix — IRI returned as-is up to the first dot.
        assert_eq!(family_of("bare.dotted"), "bare");
        // Empty input degenerates to empty family (defensive; the extractor
        // never emits empty IRIs).
        assert_eq!(family_of(""), "");
    }

    #[test]
    fn empty_triples_produce_empty_specs() {
        let specs = emit_actions(&[]);
        assert!(specs.is_empty(), "no triples ⇒ no specs");
        let non_trivial = emit_non_trivial_actions(&[]);
        assert!(non_trivial.is_empty());
    }

    #[test]
    fn function_with_no_emits_has_empty_inputs() {
        // Pure-guard-style function: it raises something but writes no fields,
        // so the dependency closure has nothing to pull from.
        let triples = vec![
            triple("odoo:m._guard", "rdf:type", "ogit:Function"),
            triple("odoo:m._guard", "raises", "exc:ValidationError"),
            // depends_on on some OTHER field — must NOT leak into _guard.inputs.
            triple("odoo:m.unrelated", "depends_on", "odoo:m.something"),
        ];
        let specs = emit_actions(&triples);
        let g = specs
            .iter()
            .find(|s| s.id == "odoo:m._guard")
            .expect("guard");
        assert!(g.effects.is_empty(), "no emitted_by ⇒ no effects");
        assert!(
            g.inputs.is_empty(),
            "empty effects ⇒ empty dependency closure (no leakage from other fields)"
        );
        assert_eq!(g.raises, vec!["exc:ValidationError".to_string()]);
    }
}
