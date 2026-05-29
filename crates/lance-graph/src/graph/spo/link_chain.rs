// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Link-chain splitter — decomposes flat dotted `depends_on` paths into
//! per-hop sequences for Foundry Link traversal.
//!
//! # The problem
//!
//! The harvester emits paths like
//! `odoo:account_move.line_ids.matched_debit_ids.debit_move_id.move_id.line_ids.amount_residual`
//! as a single dotted object string in `depends_on` triples. That string is
//! actually a 5-hop link traversal: each segment (`line_ids`,
//! `matched_debit_ids`, …) is a Many2one / One2many / Many2many relation
//! that walks to a different ObjectType, and the final segment
//! (`amount_residual`) is the leaf field.
//!
//! Foundry's compute graph needs these as **per-hop link triples** —
//! `(ObjectType, link, ObjectType)` — so the dependency walk is a typed
//! graph traversal, not a substring search.
//!
//! # What this module does
//!
//! Pure string-shape decomposition. Takes a dotted path and the source
//! family, returns the structured [`LinkChain`] record:
//!
//! ```text
//!   "odoo:account_move.line_ids.matched_debit_ids.debit_move_id.move_id.line_ids.amount_residual"
//! ─→ LinkChain {
//!       source_family: "account_move",
//!       hops: ["line_ids", "matched_debit_ids", "debit_move_id", "move_id", "line_ids"],
//!       leaf: "amount_residual",
//!    }
//! ```
//!
//! # What this module does NOT do
//!
//! **Target ObjectType resolution.** Knowing that
//! `account_move.line_ids → account.move.line` requires the
//! `OdooEntity::fields[*].target` table in `lance-graph-ontology`. Adding a
//! dep on that crate from `lance-graph` would create an upward edge in the
//! crate graph; instead, target resolution is the next layer's job
//! (a separate emitter in the consumer crate that has access to both the
//! chains and the ontology).
//!
//! # Iron rule
//!
//! Deterministic; no inference; no fallback heuristics for malformed paths.
//! A path that doesn't fit the expected shape returns `None` so the caller
//! sees the schema violation immediately rather than getting silent bad data.

use std::collections::BTreeMap;

use super::odoo_ontology::OntologyTriple;

// ---------------------------------------------------------------------------
// LinkChain — the structured per-hop view of a dotted path
// ---------------------------------------------------------------------------

/// One decomposed dependency path: the source family, its hops, and the leaf.
///
/// All fields are owned `String`s — at the codegen/ASCII layer, identity is
/// the IRI segment, not a fingerprint. Cloning is cheap relative to the
/// triple-set scan that produced it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct LinkChain {
    /// The family the chain starts from — the Odoo model name on which the
    /// `depends_on` declaration lives (e.g. `account_move`).
    pub source_family: String,
    /// Ordered link-relation segments. For a single-hop path
    /// `account_move.partner_id.name`, this is `["partner_id"]`. For the
    /// 0-hop case (direct field on the source — `account_move.amount_total`
    /// depends on `account_move.balance`), this is empty.
    pub hops: Vec<String>,
    /// The terminal field name (e.g. `amount_residual`). Always present;
    /// a path with no leaf is rejected by [`split_chain`] as malformed.
    pub leaf: String,
}

impl LinkChain {
    /// Total length of the chain in hops + the leaf: `hops.len() + 1`.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.hops.len() + 1
    }

    /// Whether this chain is a direct field reference (no link hops — the
    /// leaf lives on the source family).
    #[must_use]
    pub fn is_direct(&self) -> bool {
        self.hops.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Splitter — the deterministic decomposition
// ---------------------------------------------------------------------------

/// Decompose a dotted dependency IRI into a [`LinkChain`].
///
/// Returns `None` if the path is not in the expected `odoo:<family>.<seg>.<…>.<leaf>`
/// shape (no dots, empty segments, or a leading/trailing dot). The `odoo:`
/// prefix is optional — paths without it are treated as raw dotted IRIs.
///
/// # Examples
///
/// - `odoo:account_move.amount_total` → 0 hops, leaf `amount_total`
///   (a direct field reference on `account_move`).
/// - `odoo:account_move.partner_id.name` → 1 hop `partner_id`, leaf `name`.
/// - `odoo:account_move.line_ids.balance` → 1 hop `line_ids`, leaf `balance`.
///
/// # Edge cases
///
/// - Empty string → `None`.
/// - `"odoo:"` with nothing after → `None`.
/// - `"odoo:family"` with no leaf (just the family) → `None`.
/// - Any empty segment (`"a..b"`, `".a"`, `"a."`) → `None`.
#[must_use]
pub fn split_chain(path: &str) -> Option<LinkChain> {
    let body = path.strip_prefix("odoo:").unwrap_or(path);
    if body.is_empty() {
        return None;
    }

    // Single pass: collect segments and reject any empty fragment at the
    // same time. `split('.')` on `"a..b"` yields `["a","","b"]`, on `".a"`
    // yields `["","a"]`, on `"a."` yields `["a",""]` — all rejected.
    let mut segments: Vec<&str> = Vec::new();
    for seg in body.split('.') {
        if seg.is_empty() {
            return None;
        }
        segments.push(seg);
    }
    if segments.len() < 2 {
        // family-only — no leaf to extract.
        return None;
    }

    // `split_first` avoids the O(n) shift of `Vec::remove(0)`; `pop` is O(1).
    let (head, rest) = segments.split_first().expect("len >= 2");
    let source_family = (*head).to_string();
    let (last, middle) = rest.split_last().expect("len >= 1 after head");
    let leaf = (*last).to_string();
    let hops: Vec<String> = middle.iter().map(|s| (*s).to_string()).collect();

    Some(LinkChain {
        source_family,
        hops,
        leaf,
    })
}

/// Decompose every `depends_on` triple in a triple set into per-source
/// [`LinkChain`] sets.
///
/// Returns `BTreeMap<source_field_IRI, Vec<LinkChain>>` — keyed by the
/// SUBJECT of the `depends_on` triple (the dependent field), using the
/// subject string as-harvested (no normalization). Each chain in the Vec
/// is one of that field's decomposed dependency paths.
///
/// Malformed paths are dropped silently from the output; the surrounding
/// triple is still scanned, but produces no entry. Use [`compute_stats`]
/// over the same triple set to detect drift (it counts well-formed vs
/// malformed separately).
///
/// # Determinism
///
/// BTreeMap ordering + chain Vec is sorted ascending by `(source_family,
/// hops, leaf)` and deduplicated. Re-running on the same input produces
/// byte-identical output.
#[must_use]
pub fn split_all_depends_on(triples: &[OntologyTriple]) -> BTreeMap<String, Vec<LinkChain>> {
    let mut by_source: BTreeMap<String, Vec<LinkChain>> = BTreeMap::new();
    for t in triples {
        if t.p != "depends_on" {
            continue;
        }
        if let Some(chain) = split_chain(&t.o) {
            by_source.entry(t.s.clone()).or_default().push(chain);
        }
    }
    for chains in by_source.values_mut() {
        chains.sort();
        chains.dedup();
    }
    by_source
}

/// Statistics about a decomposition pass — useful for verifying the
/// extractor's output matches the splitter's expectations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SplitStats {
    /// Total `depends_on` triples seen in the input.
    pub depends_on_total: usize,
    /// Paths that successfully decomposed into a [`LinkChain`].
    pub well_formed: usize,
    /// Paths that returned `None` from [`split_chain`] (schema violations).
    pub malformed: usize,
    /// Maximum hop depth observed (link-traversal complexity ceiling).
    pub max_depth: usize,
    /// Direct (0-hop) field references — a `depends_on` between two fields
    /// of the same family.
    pub direct_refs: usize,
}

/// Compute aggregate decomposition statistics over a triple set.
///
/// Order-independent: the output depends only on the multiset of
/// `depends_on` triples in the input, not their order.
#[must_use]
pub fn compute_stats(triples: &[OntologyTriple]) -> SplitStats {
    let mut s = SplitStats::default();
    for t in triples {
        if t.p != "depends_on" {
            continue;
        }
        s.depends_on_total += 1;
        match split_chain(&t.o) {
            Some(chain) => {
                s.well_formed += 1;
                if chain.is_direct() {
                    s.direct_refs += 1;
                }
                if chain.depth() > s.max_depth {
                    s.max_depth = chain.depth();
                }
            }
            None => s.malformed += 1,
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::spo::odoo_ontology::parse_triples;

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

    #[test]
    fn split_direct_field_reference() {
        let chain = split_chain("odoo:account_move.amount_total").expect("well-formed");
        assert_eq!(chain.source_family, "account_move");
        assert!(chain.hops.is_empty());
        assert_eq!(chain.leaf, "amount_total");
        assert!(chain.is_direct());
        assert_eq!(chain.depth(), 1);
    }

    #[test]
    fn split_single_hop() {
        let chain = split_chain("odoo:account_move.partner_id.name").expect("well-formed");
        assert_eq!(chain.source_family, "account_move");
        assert_eq!(chain.hops, vec!["partner_id"]);
        assert_eq!(chain.leaf, "name");
        assert!(!chain.is_direct());
        assert_eq!(chain.depth(), 2);
    }

    #[test]
    fn split_five_hop_real_path() {
        // The example from the foundry-workshop evaluation: 5-hop chain.
        let chain = split_chain(
            "odoo:account_move.line_ids.matched_debit_ids.debit_move_id.move_id.line_ids.amount_residual",
        )
        .expect("well-formed");
        assert_eq!(chain.source_family, "account_move");
        assert_eq!(
            chain.hops,
            vec![
                "line_ids",
                "matched_debit_ids",
                "debit_move_id",
                "move_id",
                "line_ids",
            ]
        );
        assert_eq!(chain.leaf, "amount_residual");
        assert_eq!(chain.depth(), 6);
    }

    #[test]
    fn split_handles_missing_prefix() {
        // Without odoo: prefix, treated as a raw dotted path.
        let chain = split_chain("res_partner.name").expect("well-formed without prefix");
        assert_eq!(chain.source_family, "res_partner");
        assert_eq!(chain.leaf, "name");
    }

    #[test]
    fn split_rejects_malformed() {
        assert!(split_chain("").is_none(), "empty");
        assert!(split_chain("odoo:").is_none(), "prefix only");
        assert!(split_chain("odoo:standalone").is_none(), "no leaf");
        assert!(split_chain("odoo:a..b").is_none(), "empty segment");
        assert!(split_chain("a..b").is_none(), "empty segment, no prefix");
        assert!(split_chain(".").is_none(), "lone dot");
        assert!(split_chain("odoo:.").is_none(), "lone dot after prefix");
        assert!(split_chain("odoo:a.").is_none(), "trailing dot");
        assert!(split_chain("odoo:.a").is_none(), "leading dot");
    }

    #[test]
    fn split_all_depends_on_indexes_by_subject() {
        let triples = vec![
            triple(
                "odoo:account_move.amount_total",
                "depends_on",
                "odoo:account_move.line_ids.balance",
            ),
            triple(
                "odoo:account_move.amount_total",
                "depends_on",
                "odoo:account_move.line_ids.amount_residual",
            ),
            triple(
                "odoo:account_move.amount_residual",
                "depends_on",
                "odoo:account_move.line_ids.amount_residual",
            ),
            // Non-depends_on edge — must be ignored.
            triple(
                "odoo:account_move.amount_total",
                "emitted_by",
                "odoo:account_move._compute_amount",
            ),
        ];

        let by_subj = split_all_depends_on(&triples);
        assert_eq!(by_subj.len(), 2);

        let amt_total = by_subj
            .get("odoo:account_move.amount_total")
            .expect("amount_total deps missing");
        assert_eq!(amt_total.len(), 2);
        // Sorted ascending: amount_residual < balance.
        assert_eq!(amt_total[0].leaf, "amount_residual");
        assert_eq!(amt_total[1].leaf, "balance");
    }

    #[test]
    fn split_all_depends_on_dedups() {
        let dup = triple(
            "odoo:account_move.amount_total",
            "depends_on",
            "odoo:account_move.line_ids.balance",
        );
        let triples = vec![dup.clone(), dup];
        let by_subj = split_all_depends_on(&triples);
        let chains = by_subj.get("odoo:account_move.amount_total").unwrap();
        assert_eq!(chains.len(), 1, "duplicate triples must dedup post-sort");
    }

    #[test]
    fn compute_stats_counts_each_category() {
        // Synthetic triple set with a known distribution so every field of
        // SplitStats is asserted (the shipped-ontology test below covers the
        // aggregate counters but does not pin direct_refs to a known value).
        let triples = vec![
            // Direct (0-hop): leaf lives on the source family.
            triple(
                "odoo:account_move.amount_total",
                "depends_on",
                "odoo:account_move.balance",
            ),
            // 1-hop chain.
            triple(
                "odoo:account_move.amount_total",
                "depends_on",
                "odoo:account_move.partner_id.name",
            ),
            // 3-hop chain → max_depth = 4 (3 hops + leaf).
            triple(
                "odoo:account_move.amount_total",
                "depends_on",
                "odoo:account_move.line_ids.partner_id.country_id.code",
            ),
            // Malformed — empty segment.
            triple(
                "odoo:account_move.amount_total",
                "depends_on",
                "odoo:account_move..broken",
            ),
            // Non-depends_on edge — ignored entirely.
            triple(
                "odoo:account_move.amount_total",
                "emitted_by",
                "odoo:account_move._compute_amount",
            ),
        ];

        let s = compute_stats(&triples);
        assert_eq!(s.depends_on_total, 4, "ignores non-depends_on");
        assert_eq!(s.well_formed, 3);
        assert_eq!(s.malformed, 1);
        assert_eq!(s.direct_refs, 1, "exactly one 0-hop chain");
        assert_eq!(s.max_depth, 4, "3 hops + leaf");
    }

    #[test]
    fn shipped_ontology_decomposes_cleanly() {
        let triples = parse_triples(ONTOLOGY);
        let stats = compute_stats(&triples);

        // depends_on count must match the extraction histogram
        // (odoo_ontology.rs::predicate_histogram_matches_extraction → 6309).
        assert_eq!(stats.depends_on_total, 6309);

        // Well-formed + malformed must sum to the total — no triples lost.
        assert_eq!(stats.well_formed + stats.malformed, stats.depends_on_total);

        // The shipped data is harvester-validated; expect zero malformed
        // (if this fails, the extractor regressed, not the splitter).
        assert_eq!(
            stats.malformed, 0,
            "shipped ontology must have zero malformed depends_on paths"
        );

        // Sanity: at least some multi-hop chains exist (the 5-hop example
        // is real in the data).
        assert!(
            stats.max_depth >= 3,
            "expected multi-hop chains in shipped ontology, got max_depth={}",
            stats.max_depth
        );
    }

    #[test]
    fn shipped_ontology_amount_total_chains_present() {
        let triples = parse_triples(ONTOLOGY);
        let by_subj = split_all_depends_on(&triples);

        let chains = by_subj
            .get("odoo:account_move.amount_total")
            .expect("amount_total has no depends_on chains in shipped data");
        assert!(
            !chains.is_empty(),
            "amount_total must have ≥1 dependency chain"
        );
        // Every chain must have account_move as the source family (the
        // depends_on declarations live on the account_move model).
        for c in chains {
            assert_eq!(
                c.source_family, "account_move",
                "amount_total dependency chain points to wrong source family: {:?}",
                c
            );
        }
    }
}
