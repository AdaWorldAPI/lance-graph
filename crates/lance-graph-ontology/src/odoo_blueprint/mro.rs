// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `virtually_overrides` — the ClassView method-resolution relation, **computed**
//! from the manifest, never harvested.
//!
//! # Why this is NOT `spo_enrich` predicate #6 (Core-first, 2026-06-18)
//!
//! The Core-first correction (`EPIPHANIES.md` E-ODOO-CORE-FIRST-STRUCTURAL)
//! drew the line: a structural fact that an AST pass *reads* belongs in the
//! typed Core, and a relation that the ClassView *derives* belongs in the
//! resolver — neither is a new flat-ndjson harvest predicate.
//!
//! `virtually_overrides` is the second kind. Per the doctrine, the
//! `(has_function / inherits_from / virtually_overrides)` triad is the
//! **ClassView method-resolution manifest**: `has_function` (which methods a
//! class declares) and `inherits_from` (the `_inherit` chain) are *facts*;
//! `virtually_overrides` is the **derivation** — model `M`'s method `m`
//! virtually-overrides base `B`'s `m` iff `M` declares `m`, `B` is reachable
//! up `M`'s `_inherit` chain, and `B` also declares `m`. This module computes
//! that relation. Adding a `virtually_overrides` AST pass to `spo_enrich.py`
//! would be the drift; computing it here is the Core-correct move.
//!
//! # Manifest source is the caller's choice; the resolver is pure
//!
//! [`resolve_overrides`] takes an abstract manifest (`methods_of` +
//! `bases_of`) so the same resolver serves either source:
//!
//! - the **typed Core** ([`manifest_from_curated_core`]) — authoritative, but
//!   resolves an override only where *both* the child and the shadowed base
//!   are curated `OdooEntity`s. Today most curated models inherit *uncurated*
//!   mixins (`mail.activity.mixin`, `sequence.mixin`), so the curated-only
//!   manifest resolves a small set — honest, not fabricated.
//! - the **SPO harvest manifest** — a consumer holding the corpus builds the
//!   same two maps from `has_function` + `inherits_from` triples (388
//!   ObjectTypes incl. mixins) and gets the full resolution. The resolver
//!   code is identical; only the manifest breadth differs.
//!
//! # Precedence
//!
//! The shadowed base is the **nearest** one up a breadth-first walk of the
//! `_inherit` chain (sorted tie-break for determinism). For the *linear*
//! mixin chains Odoo uses in practice this equals Python's C3 MRO; for
//! diamonds it is the documented breadth-first approximation, not full C3.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::structural::SpoTriple;

/// One resolved method-override: `child_model.method` virtually-overrides the
/// same-named method on `base_model`, reached up the `_inherit` chain. All
/// names are underscored (corpus IRI local-part convention).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MethodOverride {
    /// Overriding model (underscored, e.g. `account_move`).
    pub child_model: String,
    /// Method name shared by child and base (e.g. `_post`).
    pub method: String,
    /// Nearest shadowed base model (underscored, e.g. `sequence_mixin`).
    pub base_model: String,
}

/// The method-resolution manifest, keyed by underscored model name:
/// `methods_of[m]` = the set of method names model `m` declares
/// (`has_function`); `bases_of[m]` = model `m`'s ordered `_inherit` bases
/// (`inherits_from`).
pub type MethodsOf = BTreeMap<String, BTreeSet<String>>;
/// See [`MethodsOf`].
pub type BasesOf = BTreeMap<String, Vec<String>>;

/// Compute `virtually_overrides` from the manifest. For every model `M` with
/// methods and every method `m` it declares, walk `M`'s `_inherit` chain
/// breadth-first and emit one override against the **nearest** base that also
/// declares `m`. Cycle-guarded; deterministic (sorted output).
#[must_use]
pub fn resolve_overrides(methods_of: &MethodsOf, bases_of: &BasesOf) -> Vec<MethodOverride> {
    let mut out: Vec<MethodOverride> = Vec::new();

    for (child, child_methods) in methods_of {
        for method in child_methods {
            if let Some(base) = nearest_base_declaring(child, method, methods_of, bases_of) {
                out.push(MethodOverride {
                    child_model: child.clone(),
                    method: method.clone(),
                    base_model: base,
                });
            }
        }
    }

    out.sort();
    out.dedup();
    out
}

/// BFS up `start`'s `_inherit` chain; return the first base (sorted tie-break
/// within a level) that declares `method`. `None` if no base declares it.
fn nearest_base_declaring(
    start: &str,
    method: &str,
    methods_of: &MethodsOf,
    bases_of: &BasesOf,
) -> Option<String> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    seen.insert(start.to_string());
    // Frontier holds (model) at increasing chain distance; a BTreeSet per level
    // gives the sorted tie-break, then feeds the next level.
    let mut frontier: VecDeque<String> =
        bases_of.get(start).into_iter().flatten().cloned().collect();

    while let Some(node) = frontier.pop_front() {
        if !seen.insert(node.clone()) {
            continue; // cycle / diamond re-visit guard
        }
        if methods_of.get(&node).is_some_and(|ms| ms.contains(method)) {
            return Some(node);
        }
        // enqueue this node's bases (sorted for determinism), depth-extending
        if let Some(bs) = bases_of.get(&node) {
            let mut next: Vec<&String> = bs.iter().collect();
            next.sort();
            for b in next {
                if !seen.contains(b) {
                    frontier.push_back(b.clone());
                }
            }
        }
    }
    None
}

/// Project resolved overrides into SPO triples
/// `(odoo:<child>.<method>, virtually_overrides, odoo:<base>.<method>)`.
#[must_use]
pub fn project_virtually_overrides(overrides: &[MethodOverride]) -> Vec<SpoTriple> {
    overrides
        .iter()
        .map(|o| {
            (
                format!("odoo:{}.{}", o.child_model, o.method),
                "virtually_overrides",
                format!("odoo:{}.{}", o.base_model, o.method),
            )
        })
        .collect()
}

/// Build the manifest from the **typed Core** — `curated_entities()` methods
/// (`has_function`) + [`super::structural::INHERITS`] (`inherits_from`).
/// Authoritative but narrow: resolves an override only where both ends are
/// curated. Model + base names are underscored to match the corpus convention.
#[must_use]
pub fn manifest_from_curated_core() -> (MethodsOf, BasesOf) {
    let underscore = |s: &str| s.replace('.', "_");

    let mut methods_of: MethodsOf = BTreeMap::new();
    for ent in super::class_signature::curated_entities() {
        let m = underscore(ent.model_name);
        let set = methods_of.entry(m).or_default();
        for meth in ent.methods {
            set.insert(meth.name.to_string());
        }
    }

    let mut bases_of: BasesOf = BTreeMap::new();
    for row in super::structural::INHERITS {
        bases_of.insert(
            underscore(row.model),
            row.bases.iter().map(|b| underscore(b)).collect(),
        );
    }

    (methods_of, bases_of)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn methods(pairs: &[(&str, &[&str])]) -> MethodsOf {
        pairs
            .iter()
            .map(|(m, ms)| (m.to_string(), ms.iter().map(|s| s.to_string()).collect()))
            .collect()
    }
    fn bases(pairs: &[(&str, &[&str])]) -> BasesOf {
        pairs
            .iter()
            .map(|(m, bs)| (m.to_string(), bs.iter().map(|s| s.to_string()).collect()))
            .collect()
    }

    #[test]
    fn child_overrides_direct_base() {
        let m = methods(&[("child", &["_post", "own"]), ("base", &["_post"])]);
        let b = bases(&[("child", &["base"])]);
        let ov = resolve_overrides(&m, &b);
        assert_eq!(
            ov,
            vec![MethodOverride {
                child_model: "child".into(),
                method: "_post".into(),
                base_model: "base".into(),
            }]
        );
    }

    #[test]
    fn method_only_on_child_is_not_an_override() {
        // `own` exists only on child → no base shadows it → no override.
        let m = methods(&[("child", &["own"]), ("base", &["_post"])]);
        let b = bases(&[("child", &["base"])]);
        assert!(resolve_overrides(&m, &b).is_empty());
    }

    #[test]
    fn nearest_base_wins_over_farther_base() {
        // child → mid → far ; all declare _x. Override targets `mid` (nearest).
        let m = methods(&[("child", &["_x"]), ("mid", &["_x"]), ("far", &["_x"])]);
        let b = bases(&[("child", &["mid"]), ("mid", &["far"])]);
        let ov = resolve_overrides(&m, &b);
        assert_eq!(ov.len(), 2, "child→mid and mid→far both resolve");
        // child's override target is the nearer `mid`, not `far`.
        let child = ov.iter().find(|o| o.child_model == "child").unwrap();
        assert_eq!(child.base_model, "mid");
    }

    #[test]
    fn transitive_skip_when_nearest_lacks_the_method() {
        // child → mid → far ; only `far` declares _x. child overrides far.
        let m = methods(&[("child", &["_x"]), ("mid", &["other"]), ("far", &["_x"])]);
        let b = bases(&[("child", &["mid"]), ("mid", &["far"])]);
        let ov = resolve_overrides(&m, &b);
        let child = ov.iter().find(|o| o.child_model == "child").unwrap();
        assert_eq!(child.base_model, "far");
    }

    #[test]
    fn cycle_in_chain_is_guarded() {
        // a → b → a (pathological). Must terminate; a._x overrides b._x.
        let m = methods(&[("a", &["_x"]), ("b", &["_x"])]);
        let b = bases(&[("a", &["b"]), ("b", &["a"])]);
        let ov = resolve_overrides(&m, &b); // must not hang
        assert!(ov
            .iter()
            .any(|o| o.child_model == "a" && o.base_model == "b"));
    }

    #[test]
    fn projection_shape_is_method_iri_both_sides() {
        let ov = vec![MethodOverride {
            child_model: "account_move".into(),
            method: "_post".into(),
            base_model: "sequence_mixin".into(),
        }];
        let triples = project_virtually_overrides(&ov);
        assert_eq!(
            triples,
            vec![(
                "odoo:account_move._post".to_string(),
                "virtually_overrides",
                "odoo:sequence_mixin._post".to_string(),
            )]
        );
    }

    #[test]
    fn typed_core_manifest_builds_and_resolver_runs() {
        // Smoke: the curated-core manifest builds and the resolver terminates.
        // The resolved set is honestly small/empty today (curated models mostly
        // inherit UNCURATED mixins, so few overrides have both ends present) —
        // a fuller SPO manifest resolves more. We assert the manifest is
        // non-trivial and the call is total, not that overrides exist.
        let (m, b) = manifest_from_curated_core();
        assert!(!m.is_empty(), "curated core declares methods");
        assert!(
            b.contains_key("account_move"),
            "account_move has _inherit bases"
        );
        let _ = resolve_overrides(&m, &b); // terminates, deterministic
    }
}
