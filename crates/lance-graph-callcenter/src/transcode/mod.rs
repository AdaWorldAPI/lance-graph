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
