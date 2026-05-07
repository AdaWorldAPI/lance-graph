//! Producer-side DTO and dictionary row types.
//!
//! `MappingProposal` is what TTL hydration and (future) MySQL/MSSQL scanners
//! emit. `MappingRow` is what the registry stores. `HydrationReport` is the
//! summary returned from a hydration run. `MappingHandle` is an opaque
//! receipt for an appended proposal.
//!
//! Carrier-method doctrine: methods on these types describe what they do.

use crate::namespace::{NamespaceId, OgitUri, SchemaKind, SchemaPtr};
use lance_graph_contract::property::{LinkSpec, Marking, Schema, SemanticType};

/// A single producer-side proposal. One TTL file → typically one proposal
/// (an entity TTL). Schema scanners may emit one proposal per discovered
/// table; customer admin forms may emit one per row.
#[derive(Clone, Debug)]
pub struct MappingProposal {
    /// Producer-facing public name. For OGIT-direct: the OGIT URI itself
    /// (e.g. `ogit.Network:IPAddress`). For tenant bridges: the bridge's
    /// public name (e.g. `Customer`, `WorkOrder`).
    pub public_name: String,
    /// Bridge id this proposal is registered under. `"ogit"` for raw OGIT.
    /// `"woa"`, `"medcare"`, etc. for tenant bridges. The same OGIT URI may
    /// appear under multiple bridge ids with different public names.
    pub bridge_id: String,
    /// Canonical OGIT URI. Required: every mapping must resolve to a URI.
    pub ogit_uri: OgitUri,
    /// Namespace of the OGIT URI (e.g. "Network", "WorkOrder"). The
    /// registry uses this to assign / look up the `NamespaceId` (G).
    pub namespace: String,
    /// What kind of mapping this is.
    pub kind: MappingProposalKind,
    /// Default marking. PII / Financial / Restricted overrides come from
    /// the TTL annotation or the schema scanner.
    pub marking: Marking,
    /// Confidence — 1.0 for canonical TTL hydration; <1.0 for scanner-
    /// suggested mappings awaiting review; 0.0 for guesses.
    pub confidence: f32,
    /// Where this proposal came from. Free text, intended for audit.
    pub source_uri: String,
    /// SHA256 of the source fragment (TTL file body, scanner output, etc.).
    /// Used for idempotent re-hydration.
    pub checksum: String,
    /// Who/what produced this proposal. `"ogit_hydrator_v1"`,
    /// `"mysql_scanner_v1"`, `"admin:user@..."`.
    pub created_by: String,
}

/// What kind of mapping this proposal carries. Entity mappings carry a
/// `Schema`; edge mappings carry a `LinkSpec`; attribute mappings carry a
/// single `SemanticType` annotation.
#[derive(Clone, Debug)]
pub enum MappingProposalKind {
    Entity {
        schema: Schema,
    },
    Edge {
        link: LinkSpec,
    },
    Attribute {
        predicate: String,
        semantic_type: SemanticType,
    },
}

impl MappingProposal {
    pub fn schema_kind(&self) -> SchemaKind {
        match self.kind {
            MappingProposalKind::Entity { .. } => SchemaKind::Entity,
            MappingProposalKind::Edge { .. } => SchemaKind::Edge,
            MappingProposalKind::Attribute { .. } => SchemaKind::Attribute,
        }
    }
}

/// What the registry stores. `MappingRow` mirrors the
/// `ontology_dictionary` Lance table schema column-for-column. Adding a
/// new column means adding a field here AND extending the Lance writer
/// (under `lance-cache`) AND bumping the registry's append path.
#[derive(Clone, Debug)]
pub struct MappingRow {
    pub bridge_id: String,
    pub public_name: String,
    pub ogit_uri: OgitUri,
    pub namespace_id: NamespaceId,
    pub schema_ptr: SchemaPtr,
    pub kind: SchemaKind,
    pub semantic_type: SemanticType,
    pub marking: Marking,
    pub confidence: f32,
    pub created_at_us: i64,
    pub created_by: String,
    pub source_uri: String,
    pub active: bool,
    pub checksum: String,
}

impl MappingRow {
    pub fn schema_ptr(&self) -> SchemaPtr {
        self.schema_ptr
    }
}

/// Per-attribute provenance — sibling structure to [`MappingRow`].
///
/// After OGIT #2 every attribute predicate in an entity TTL carries its own
/// `dcterms:source` literal (e.g. `ogit.WorkOrder:fahrtKm` is sourced from
/// `"AdaWorldAPI/WoA/models.py:Customer.fahrt_km"`). The TTL parser walks
/// these triples and emits one `AttributeProvenance` per `(predicate,
/// dcterms:source)` pair so column-level provenance can ride alongside the
/// entity-level `source_uri` on `MappingRow` without modifying that struct's
/// shape (Wave 3 owns the MappingRow column extension; this is the Wave 1
/// extraction half).
///
/// `predicate_iri` is the canonical OGIT URI of the attribute (e.g.
/// `"ogit.WorkOrder:fahrtKm"`). `source_uri` is the literal verbatim from
/// the TTL's `dcterms:source` triple. Closes TTL-PROBE-5 (the data is now
/// available; persisting it into the dictionary row is a follow-up).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttributeProvenance {
    pub predicate_iri: String,
    pub source_uri: String,
}

/// Per-entity provenance bundle — what the TTL parser exposes alongside
/// `MappingProposal` lists. Carries the entity URI plus every attribute
/// predicate's `(predicate_iri, dcterms:source)` pair the parser saw.
///
/// Empty `entity_source_uri` and empty `attribute_sources` together mean the
/// TTL had no `dcterms:source` triples at all (legacy / pre-OGIT-#2 files).
#[derive(Clone, Debug, Default)]
pub struct ProvenanceBundle {
    /// OGIT URI of the entity / verb / attribute subject this bundle is
    /// keyed against, e.g. `"ogit.WorkOrder:Customer"`.
    pub entity_uri: String,
    /// Entity-level `dcterms:source` literal (verbatim) — empty when the
    /// TTL did not declare one for this subject.
    pub entity_source_uri: String,
    /// Per-attribute `(predicate_iri, dcterms:source)` pairs collected from
    /// every attribute subject under this entity that carried its own
    /// `dcterms:source` literal in the TTL.
    pub attribute_sources: Vec<AttributeProvenance>,
}

impl ProvenanceBundle {
    /// Number of per-attribute `dcterms:source` pairs recorded.
    pub fn attribute_count(&self) -> usize {
        self.attribute_sources.len()
    }

    /// Look up the `dcterms:source` literal for a given predicate IRI.
    pub fn source_for(&self, predicate_iri: &str) -> Option<&str> {
        self.attribute_sources
            .iter()
            .find(|p| p.predicate_iri == predicate_iri)
            .map(|p| p.source_uri.as_str())
    }
}

/// Opaque receipt for an appended proposal. Carries the assigned
/// `SchemaPtr` and the dictionary index where the row landed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MappingHandle {
    pub schema_ptr: SchemaPtr,
    pub row_index: u32,
}

/// Summary of a `hydrate_once` run.
#[derive(Clone, Debug, Default)]
pub struct HydrationReport {
    pub registered: u32,
    pub skipped_idempotent: u32,
    pub failed: u32,
    pub failures: Vec<HydrationFailure>,
    pub namespaces_seen: Vec<String>,
    pub from_cache: bool,
}

#[derive(Clone, Debug)]
pub struct HydrationFailure {
    pub source: String,
    pub reason: String,
}

impl HydrationReport {
    pub fn total(&self) -> u32 {
        self.registered + self.skipped_idempotent + self.failed
    }

    pub fn is_clean(&self) -> bool {
        self.failed == 0
    }
}
