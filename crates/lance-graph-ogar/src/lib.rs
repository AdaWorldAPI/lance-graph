//! `lance-graph-ogar` — the OGAR (Open Graph of Active Record) activation crate.
//!
//! The lance-graph-side **re-export + activation** of OGAR's full Active-Record
//! surface, the OGAR half of the clean separation (operator, 2026-06-20):
//!
//! ```text
//!   lance-graph-ontology  =  OGIT   (TTL/RDF hydration spine — the ontology SOURCE)
//!   lance-graph-ogar      =  OGAR   (Active-Record Class / ClassView / adapters)
//! ```
//!
//! # OGAR is the Active-Record Core, and it already speaks the contract
//!
//! OGAR is **not** "just a codebook" — the unit is the **`Class`** and its
//! **`ClassView`** (the active-record shape: identity, state, relations,
//! composition). The codebook `u16` is one *facet* of a `Class`'s identity.
//!
//! - [`ogar_vocab::Class`] — the calcified AR shape: canonical concept + typed
//!   attributes + family-edge `Association`s. `canonical_concept_id` == the
//!   contract [`ClassId`](lance_graph_contract::class_view::ClassId).
//! - [`ogar_class_view::OgarClassView`] — **`impl lance_graph_contract::ClassView`**:
//!   builds an `ObjectView` per promoted concept, keyed by `ClassId`, exposing the
//!   whole 32-concept AR set through the contract's runtime projection trait
//!   (`render_rows(id, mask)`).
//! - [`ogar_ontology`] — prefix conventions + NiblePath identity routing.
//! - [`ogar_adapter_surrealql`] — `emit(Class) -> SurrealQL DDL` (the DO arm);
//!   the `unmap(SurrealQL) -> Class` parser half is behind `surrealql-parser`.
//!
//! OGAR depending on `lance-graph-contract` (the **zero-dep** trait crate) is
//! *not* "needing lance-graph" — contract is the compile-time handshake (the
//! "contracts compile types, never serialize" principle). OGAR stays fully
//! **headless-capable**: a build without this crate uses the contract's zero-dep
//! [`ogar_codebook`](lance_graph_contract::ogar_codebook) mirror + the bare
//! `ClassView` trait; OGAR's own crates never depend on the lance-graph engine.
//!
//! # Auto-activation = Cargo presence (no runtime detection)
//!
//! A build graph that pulls THIS crate (the golden image via `symbiont`, or any
//! AR-aware consumer — q2, medcare, …) gets the **real** OGAR `Class`/`ClassView`/
//! codebook (including [`ogar_vocab`]'s full curator-alias normalizer, so OGAR is
//! never dumbed down) **plus** the [`parity`] guard that fails the build if the
//! contract's lean `ogar_codebook` mirror ever drifts from OGAR's authoritative
//! `class_ids::ALL`. One contract source: this crate and `ogar-class-view` both
//! resolve `lance-graph-contract` to git `AdaWorldAPI/lance-graph#main` (which
//! carries `ogar_codebook` since PR #563), so the `OgarClassView` `impl ClassView`
//! is for the SAME contract the guard checks.
//!
//! # The OGIT ↔ OGAR seam
//!
//! `lance-graph-ontology` (OGIT) hydrates classes from TTL; OGAR mints the
//! calcified canonical concepts (`class_ids::ALL`) keyed by the same `ClassId`
//! space. They meet at the codebook id == `NodeGuid.classid` low u16 — the
//! `0xDDCC` domain layout the [`parity`] guard pins. Reconciling an OGIT-hydrated
//! TTL class against an OGAR-promoted concept is a `ClassId` lookup, not a parse.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

// ── Full re-export of the OGAR Active-Record crates under stable names ──
pub use ogar_adapter_surrealql;
pub use ogar_class_view;
pub use ogar_ontology;
pub use ogar_vocab;

// ── The contract surface OGAR implements + the wire mirror the guard checks ──
pub use lance_graph_contract as contract;

/// The OGAR active-record `ClassView` projection (`impl
/// lance_graph_contract::ClassView`) — the one-stop entry point a renderer holds.
pub use ogar_class_view::OgarClassView;
/// The calcified canonical AR shape (attributes + family `Association`s).
pub use ogar_vocab::Class;

/// Codebook parity-guard — the drift fuse between OGAR's authoritative codebook
/// (`ogar_vocab::class_ids::ALL`) and the contract's zero-dep wire mirror
/// (`lance_graph_contract::ogar_codebook::CODEBOOK`). Compiled (and tested) only
/// when this crate is in the build graph, so any OGAR-present build (the golden
/// image, AR-aware consumers) fails fast on divergence; OGAR-absent builds carry
/// the mirror alone and never need the check.
pub mod parity {
    use lance_graph_contract::ogar_codebook as mirror;

    /// Whether OGAR's domain for `id` agrees with the contract mirror's. Both
    /// enums are structurally identical (`id >> 8` discriminant); compared by a
    /// total match so a new OGAR domain variant trips this (`#[non_exhaustive]`).
    #[must_use]
    pub fn domains_agree(id: u16) -> bool {
        use lance_graph_contract::ogar_codebook::ConceptDomain as C;
        use ogar_vocab::ConceptDomain as O;
        matches!(
            (
                ogar_vocab::canonical_concept_domain(id),
                mirror::canonical_concept_domain(id)
            ),
            (O::Reserved, C::Reserved)
                | (O::ProjectMgmt, C::ProjectMgmt)
                | (O::Commerce, C::Commerce)
                | (O::Osint, C::Osint)
                | (O::Ocr, C::Ocr)
                | (O::Health, C::Health)
                | (O::Unassigned, C::Unassigned)
        )
    }

    /// Assert the mirror is a faithful, complete copy of OGAR's codebook —
    /// forward (mirror ⊆ OGAR), reverse (OGAR ⊆ mirror), and domain agreement.
    /// Returns the number of concepts checked. Panics on any divergence.
    pub fn assert_codebook_parity() -> usize {
        // Forward: every mirror entry resolves identically through OGAR's API.
        for &(concept, id) in mirror::CODEBOOK {
            assert_eq!(
                ogar_vocab::canonical_concept_id(concept),
                Some(id),
                "contract mirror has {concept}={id:#06x} but OGAR disagrees",
            );
            assert!(
                domains_agree(id),
                "domain disagreement for {concept} ({id:#06x})"
            );
        }
        // Reverse: every OGAR canonical concept is present in the mirror with the
        // same id (no OGAR concept silently missing from the wire mirror).
        for &(concept, id) in ogar_vocab::class_ids::ALL {
            assert_eq!(
                mirror::canonical_concept_id(concept),
                Some(id),
                "OGAR has {concept}={id:#06x} but contract mirror is missing/wrong",
            );
        }
        ogar_vocab::class_ids::ALL.len()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn mirror_is_a_faithful_copy_of_ogar_codebook() {
            let n = assert_codebook_parity();
            assert!(n >= 32, "expected ≥32 promoted concepts, got {n}");
        }

        #[test]
        fn classid_low_u16_is_the_codebook_id() {
            // The contract NodeGuid.classid low u16 IS the OGAR codebook id — the
            // wire identity the whole separation rests on.
            use lance_graph_contract::NodeGuid;
            let project_id = ogar_vocab::canonical_concept_id("project").unwrap();
            let guid = NodeGuid::new(u32::from(project_id), 0, 0, 0, 0, 0);
            assert_eq!(guid.classid() as u16, project_id);
            // and it routes to the ProjectMgmt domain on both sides
            assert!(domains_agree(project_id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ogar_class_view_implements_contract_class_view() {
        // The activation in one line: an OgarClassView IS a contract ClassView,
        // so a consumer holding `&dyn ClassView` can be handed the real OGAR AR
        // surface. (Compile-time proof; constructing it walks the 32 class fns.)
        use lance_graph_contract::class_view::ClassView;
        let view = OgarClassView::new();
        let _as_trait: &dyn ClassView = &view;
    }
}
