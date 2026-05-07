//! `NamespaceRegistry` — IRI ↔ `ontology_context_id: u32` allocation table.
//!
//! Sidecar to [`crate::OntologyRegistry`]. The cascade plan
//! (`.claude/plans/ogit-cascade-supabase-callcenter-v1.md` §Pillar 1) and the
//! RDF/FMA/SNOMED plan (`.claude/plans/lance-graph-rdf-fma-snomed-v1.md`
//! §Core types, open question 2 of the cascade plan) ratified **dense,
//! deterministic, persistent across hydrate** allocation.
//!
//! The seed table below is the canonical assignment for the v1 push. Wave 2
//! `agent-bioportal-stubs` reserves ids `10..=19` for `Medical/<sub>`
//! (ICD10CM=10, RxNorm=11, LOINC=12, FMA=13, RadLex=14, SNOMED=15, MONDO=16,
//! HPO=17, DRON=18, CHEBI=19) — that range is allocated densely so the BioPortal
//! TTL stubs can declare `ogit:contextId 10` (etc.) at file authoring time
//! without coordinating with this allocator.
//!
//! `SMB = 0` is **export-only** per v5 ratification (smb-bridge surfaces under
//! `lance-graph-callcenter::ontology_dto` and is consumed by Zone 3 transcode
//! only); seeding it as 0 means callers that omit the context id default to
//! the export-only context, matching back-compat. `WorkOrder = 1`,
//! `Healthcare = 2`, `Network = 3` are the three "live cognitive" namespaces.
//!
//! Carrier-method doctrine: methods on the registry, not free functions.

use std::collections::HashMap;

/// Sidecar in-memory mapping `namespace IRI → ontology_context_id`. Matches
/// `lance-graph-rdf::NamedGraphRegistry` shape but lives in the ontology
/// crate so the registry's hot path (`SchemaPtr::ontology_context_id()`) can
/// resolve without pulling the rdf crate.
#[derive(Clone, Debug, Default)]
pub struct NamespaceRegistry {
    ids: HashMap<String, u32>,
}

impl NamespaceRegistry {
    /// Empty registry. Use [`NamespaceRegistry::seed_defaults`] for the v1
    /// canonical assignment.
    pub fn new() -> Self {
        Self::default()
    }

    /// V1-canonical allocation. Dense + deterministic; safe to call on every
    /// hydrate (calls into `allocate` are idempotent via `entry`).
    ///
    /// | Namespace | Context id | Source |
    /// |---|---|---|
    /// | `SMB` | 0 | export-only per v5 ratification |
    /// | `WorkOrder` | 1 | OGIT/NTO/WorkOrder (already shipped) |
    /// | `Healthcare` | 2 | OGIT/NTO/Healthcare (delegated to lance-graph-rdf) |
    /// | `Network` | 3 | OGIT/NTO/Network |
    /// | `EmailCorrespondance` | 4 | OGIT/NTO/EmailCorrespondance (spear/stalwart/SharePoint) |
    /// | `Medical/ICD10CM` | 10 | BioPortal stub (Wave 2 agent-bioportal-stubs) |
    /// | `Medical/RxNorm` | 11 | BioPortal stub |
    /// | `Medical/LOINC` | 12 | BioPortal stub |
    /// | `Medical/FMA` | 13 | BioPortal stub |
    /// | `Medical/RadLex` | 14 | BioPortal stub |
    /// | `Medical/SNOMED` | 15 | BioPortal stub (license-gated load) |
    /// | `Medical/MONDO` | 16 | BioPortal stub |
    /// | `Medical/HPO` | 17 | BioPortal stub |
    /// | `Medical/DRON` | 18 | BioPortal stub |
    /// | `Medical/CHEBI` | 19 | BioPortal stub |
    pub fn seed_defaults() -> Self {
        let mut ids = HashMap::with_capacity(15);
        // Live cognitive namespaces.
        ids.insert("SMB".to_string(), 0); // export-only per v5 ratification
        ids.insert("WorkOrder".to_string(), 1);
        ids.insert("Healthcare".to_string(), 2);
        ids.insert("Network".to_string(), 3);
        // Mail orchestration namespace (spear / stalwart / SharePoint).
        ids.insert("EmailCorrespondance".to_string(), 4);
        // Medical/<sub> reserved range 10..=19, dense.
        ids.insert("Medical/ICD10CM".to_string(), 10);
        ids.insert("Medical/RxNorm".to_string(), 11);
        ids.insert("Medical/LOINC".to_string(), 12);
        ids.insert("Medical/FMA".to_string(), 13);
        ids.insert("Medical/RadLex".to_string(), 14);
        ids.insert("Medical/SNOMED".to_string(), 15);
        ids.insert("Medical/MONDO".to_string(), 16);
        ids.insert("Medical/HPO".to_string(), 17);
        ids.insert("Medical/DRON".to_string(), 18);
        ids.insert("Medical/CHEBI".to_string(), 19);
        Self { ids }
    }

    /// Look up the context id for `namespace_iri`. Returns `None` for
    /// unregistered namespaces; consumers needing automatic allocation use
    /// [`NamespaceRegistry::allocate`].
    pub fn get(&self, namespace_iri: &str) -> Option<u32> {
        self.ids.get(namespace_iri).copied()
    }

    /// Get-or-allocate. Returns the existing id if registered, otherwise
    /// assigns the next free dense id (skipping reserved seed ranges). This
    /// preserves the dense + deterministic property: the same hydrate run
    /// from a clean registry always produces the same id sequence.
    pub fn allocate(&mut self, namespace_iri: &str) -> u32 {
        if let Some(id) = self.ids.get(namespace_iri) {
            return *id;
        }
        let next = self.next_free_id();
        self.ids.insert(namespace_iri.to_string(), next);
        next
    }

    /// Number of registered namespaces.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// True iff no namespaces are registered.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Iterate every (namespace_iri, context_id) pair. Order is unspecified
    /// (HashMap iteration); callers needing deterministic order should sort.
    pub fn iter(&self) -> impl Iterator<Item = (&str, u32)> {
        self.ids.iter().map(|(k, v)| (k.as_str(), *v))
    }

    /// First context id that is not currently in use. Skips the seed ranges
    /// to keep allocations dense within their family (v1 ids 0..=3 + 10..=19
    /// occupied by `seed_defaults`; first dynamic id therefore lands at 20).
    fn next_free_id(&self) -> u32 {
        let mut candidate: u32 = 0;
        let used: std::collections::BTreeSet<u32> = self.ids.values().copied().collect();
        while used.contains(&candidate) {
            candidate = candidate.wrapping_add(1);
        }
        candidate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_defaults_has_fifteen_entries() {
        let r = NamespaceRegistry::seed_defaults();
        assert_eq!(r.len(), 15);
    }

    #[test]
    fn seed_defaults_assigns_canonical_ids() {
        let r = NamespaceRegistry::seed_defaults();
        // Live cognitive namespaces.
        assert_eq!(r.get("SMB"), Some(0));
        assert_eq!(r.get("WorkOrder"), Some(1));
        assert_eq!(r.get("Healthcare"), Some(2));
        assert_eq!(r.get("Network"), Some(3));
        assert_eq!(r.get("EmailCorrespondance"), Some(4));
        // Medical/<sub> reserved range 10..=19.
        assert_eq!(r.get("Medical/ICD10CM"), Some(10));
        assert_eq!(r.get("Medical/CHEBI"), Some(19));
    }

    #[test]
    fn allocate_skips_to_first_unused_id() {
        let mut r = NamespaceRegistry::seed_defaults();
        // 0..=4 and 10..=19 are taken; first free id is 5.
        let id = r.allocate("CallCenter");
        assert_eq!(id, 5);
        // Idempotent: re-allocate returns the same id.
        assert_eq!(r.allocate("CallCenter"), 5);
        // Next allocation skips again.
        assert_eq!(r.allocate("Splat"), 6);
    }
}
