//! Outer ↔ inner ontology transcode — reusable Foundry primitives.
//!
//! This module group is the **mapper** between two ontology surfaces:
//!
//! - **Outer ontology** — the shape every external consumer sees. Object
//!   types (`Schema`), link types (`LinkSpec`), action types (`ActionSpec`),
//!   bilingual labels (`Locale` + `Label`), and the wire DTOs in
//!   [`crate::ontology_dto`]. Foundry's "Object Type / Link Type" surface.
//!
//! - **Inner ontology** — the shape the internal SoA actually stores.
//!   `BindSpace` columns (`FingerprintColumns`, `EdgeColumn`,
//!   `QualiaColumn`, `MetaColumn`, `entity_type`), CAM-PQ-encoded
//!   compressed columns when persisted to a Lance dataset, and SPO
//!   triples produced by [`SchemaExpander`].
//!
//! The pieces below are **domain-agnostic**. None of them reference
//! medcare, smb, callcenter, or any specific ontology — they operate on
//! whatever `Ontology` is handed in. Domain-specific schemas live where
//! they belong: `medcare_ontology()` in [`crate::ontology_dto`],
//! `smb_ontology()` in [`crate::ontology_dto`], future verticals in their
//! own factory.
//!
//! ## Modules
//!
//! - [`zerocopy`] — owned-column → Arrow `RecordBatch` mapping. The
//!   canonical zerocopy path: `Vec<T>` → `Buffer` is an `O(1)`
//!   reinterpretation. Wraps an `Ontology`-derived schema, refuses
//!   undeclared columns at the boundary.
//! - [`cam_pq_decode`] — codec dispatch for persistent SoA columns. The
//!   `CamPqDecoder` trait + `PassthroughDecoder` cover the
//!   `Skip`/`Passthrough` `CodecRoute` lanes; `CamPq` decode-on-read
//!   plumbs once a codebook handle is wired (see `ROADMAP` below).
//! - [`ontology_table`] — DataFusion `TableProvider` over an
//!   `(Ontology, entity_type)` pair. Schema reflection works today;
//!   filter pushdown to the SPO store is the canonical Phase 2 lift.
//! - [`spo_filter`] — SQL filter → SPO lookup translator. Recognises
//!   `entity_type`/`predicate`/`entity_id`/`nars_frequency`/
//!   `nars_confidence` against any `Ontology`.
//! - [`parallelbetrieb`] — **the one deliberate transition bandaid**.
//!   MySQL ↔ DataFusion ↔ SPO reconciler. Necessary as ground truth
//!   during F1/F2; documented as transitional, not as Foundry primitive.
//!
//! ## What this module is NOT
//!
//! - A new "transcode crate". After PR #73 framed `lance-graph-callcenter`
//!   itself as the Foundry / supabase-realtime transcode crate, the right
//!   home for these helpers is here, alongside `ontology_dto` and
//!   `version_watcher`. A sibling crate would create a competing framing.
//! - A duplicate of `ontology_dto`. The DTO surface (`OntologyDto`,
//!   `EntityTypeDto`, etc.) stays canonical in `crate::ontology_dto`.
//!   This module group consumes that surface; it does not redefine it.
//! - A duplicate of `version_watcher`. Realtime fan-out belongs to the
//!   existing `LanceVersionWatcher`. This module group does not introduce
//!   a second channel primitive.
//!
//! ## Feature gating
//!
//! The whole subtree is behind the `transcode` feature so consumers that
//! don't need DataFusion/Arrow can still depend on the rest of
//! `lance-graph-callcenter`. Each submodule then layers on the feature
//! it strictly needs (`query-lite` for DataFusion, `arrow` for the
//! zerocopy mapping, etc.).

pub mod cam_pq_decode;
pub mod parallelbetrieb;
pub mod spo_filter;
pub mod zerocopy;

#[cfg(feature = "query-lite")]
pub mod ontology_table;

// Re-export the outer-ontology DTO types so consumers reach the whole
// transcode surface from one import path.
pub use crate::ontology_dto::{
    ActionTypeDto, EntityTypeDto, LinkTypeDto, OntologyDto, PropertyDto,
};
pub use lance_graph_contract::ontology::{
    EntityTypeId, ExpandedTriple, Label, Locale, Ontology, SchemaExpander,
};

// ── Cached ontology bundle ───────────────────────────────────────────────────

use std::collections::HashMap;
use std::sync::Arc;

/// Bundle that caches the bilingual DTO projections of one `Ontology`.
///
/// `OntologyDto::from_ontology(_, locale)` walks every schema +
/// link + action and is `O(properties + links + actions)`. Per-call
/// rebuilds in a hot path waste cycles. Round-2 of the transcode crate
/// extracts the cached pattern that medcare-rs's `MedcareOntology` and
/// smb-office-rs's session ontology both grew independently — one place,
/// one bug to fix, identical semantics for both consumers.
///
/// Construction is `O(work)`; every subsequent `dto(locale)` call is a
/// `HashMap` hit. The DTOs are cloned-cheap (`Arc<OntologyDto>`).
///
/// # Example
///
/// ```rust,ignore
/// use lance_graph_callcenter::transcode::CachedOntology;
/// use lance_graph_callcenter::ontology_dto::medcare_ontology;
/// use lance_graph_contract::ontology::Locale;
///
/// let cached = CachedOntology::new(medcare_ontology());
/// let de = cached.dto(Locale::De); // O(1) cache hit
/// let en = cached.dto(Locale::En); // O(1) cache hit
/// assert_eq!(cached.inner().name, "medcare");
/// ```
#[derive(Debug)]
pub struct CachedOntology {
    inner: Arc<Ontology>,
    dtos: HashMap<Locale, Arc<OntologyDto>>,
}

impl CachedOntology {
    /// Build a `CachedOntology` for the given inner ontology, eagerly
    /// projecting it to every supported locale (`De`, `En`).
    pub fn new(ontology: Ontology) -> Self {
        let inner = Arc::new(ontology);
        let mut dtos = HashMap::with_capacity(2);
        for locale in [Locale::De, Locale::En] {
            dtos.insert(locale, Arc::new(OntologyDto::from_ontology(&inner, locale)));
        }
        Self { inner, dtos }
    }

    /// Return a shared reference to the underlying inner ontology.
    /// Use this when you need to consume the canonical Schema /
    /// LinkSpec / ActionSpec entries directly (e.g. inside
    /// `OntologyTableProvider::new`).
    pub fn inner(&self) -> &Arc<Ontology> {
        &self.inner
    }

    /// Look up the DTO projection for one locale. Returns the cached
    /// `Arc<OntologyDto>` — clone is cheap.
    ///
    /// Panics only if the locale wasn't projected at construction
    /// time, which can't happen with the present implementation
    /// (constructor projects all variants of `Locale`). The panic
    /// would surface as a fast-failure indicator if the enum grows.
    pub fn dto(&self, locale: Locale) -> Arc<OntologyDto> {
        self.dtos
            .get(&locale)
            .cloned()
            .expect("Locale must be projected at CachedOntology construction time")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::property::Schema;

    fn small_ontology() -> Ontology {
        Ontology::builder("Small")
            .label(Label::new("small", "Small Test", "Kleiner Test"))
            .schema(Schema::builder("Patient").required("name").build())
            .build()
    }

    #[test]
    fn cached_ontology_projects_every_locale_at_construction() {
        let cached = CachedOntology::new(small_ontology());
        let de = cached.dto(Locale::De);
        let en = cached.dto(Locale::En);
        // Just verifying the cache is populated and clones are cheap.
        assert_eq!(de.locale, Locale::De);
        assert_eq!(en.locale, Locale::En);
    }

    #[test]
    fn cached_ontology_clones_are_arc_cheap() {
        let cached = CachedOntology::new(small_ontology());
        let a = cached.dto(Locale::De);
        let b = cached.dto(Locale::De);
        // Both Arcs point at the same cached projection.
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn cached_ontology_inner_round_trips() {
        let cached = CachedOntology::new(small_ontology());
        assert_eq!(cached.inner().name, "Small");
    }
}
