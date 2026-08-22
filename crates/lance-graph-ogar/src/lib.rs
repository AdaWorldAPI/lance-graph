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
//! never dumbed down) **plus** the [`parity`] check.
//!
//! ## Plug-and-play, not a fuse (operator, 2026-08-14)
//!
//! This used to carry a **compile-time length fuse** — a `const` assert that the
//! contract mirror and `ogar_vocab::class_ids::ALL` held the same number of
//! concepts, firing in ANY build. **It is removed.** The reasoning, and why the
//! removal is a strengthening rather than a loosening:
//!
//! A hand-maintained mirror plus a global equality assert is the opposite of
//! plug-and-play. It is a device that works only if you also patch the host's
//! driver table by hand, in another repo, in another PR — and the assert can
//! only ever *detect* the omission, never prevent or resolve it. Worse, it
//! detects it in the wrong place: this crate is workspace-excluded, so its own
//! tests are not the main CI gate, while every AR-aware CONSUMER (q2, medcare, …)
//! compiles it. A producer-side bookkeeping lapse therefore broke consumers'
//! builds, not the producer's.
//!
//! That is not hypothetical. On 2026-08-14 `osm_street_node` (`0x0F0B`) was minted
//! in OGAR and its mirror row landed in a separate, unopened PR; the fuse panicked
//! at const-eval (`E0080`) and killed a production deploy at COMPILE — for a
//! concept that deploy never used.
//!
//! **The inversion:** the device announces, the host enumerates and binds. A
//! consumer declares one [`lance_graph_contract::hotplug::HotPlug`] naming the
//! classids it actually plugs, and [`OgarAuthority`] resolves exactly those
//! against the authority — returning a named [`lance_graph_contract::hotplug::ActivationDrift`]
//! (`UnknownClassid`, `NoCapabilitiesFor`, …) for the ids the consumer USES. A
//! concept nobody plugs cannot break anyone's build, so minting one is a single
//! PR again.
//!
//! What still catches genuine drift, in the right place and at the right blast
//! radius:
//! - [`parity::assert_codebook_parity`] — the **runtime full bijection** (forward,
//!   reverse, and domain agreement). It strictly CONTAINS the length check the
//!   fuse performed, so nothing is lost by deleting the fuse; it is asserted by
//!   this crate's tests and callable at consumer startup.
//! - **Hot-plug activation** — per-consumer, per-classid, at the moment of use.
//!
//! Prefer resolving through the authority ([`OgarAuthority`]) over reading the
//! mirror: the mirror is the BBB-safe fallback for a consumer that cannot depend
//! on OGAR at all, and a stale mirror is a test failure here rather than a
//! silent mis-resolution there.
//!
//! One contract source: this crate path-deps `lance-graph-contract` (the canonical
//! in-repo copy) and a `[patch]` folds `ogar-class-view`'s transitive *git*
//! contract onto the SAME path copy, so the `OgarClassView` `impl ClassView` is for
//! the contract the guard checks (an in-repo workspace root adding this crate must
//! repeat that patch — see the manifest CONSUMER REQUIREMENT note).
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

/// DEPRECATED IR arm — only under the existing `surrealql-parser` feature.
///
/// The SurrealQL branch is not in use (operator, 2026-08-22; measured zero
/// consumers in lance-graph, MedCare-rs and OGAR). The re-export is not
/// deleted, so the work stays reachable — it is simply no longer
/// unconditional, so no default build pulls the crate.
#[cfg(feature = "surrealql-parser")]
pub use ogar_adapter_surrealql;
pub use ogar_class_view;
pub use ogar_ontology;
pub use ogar_vocab;

pub use lance_graph_contract as contract;

/// The OGAR active-record `ClassView` projection (`impl
/// lance_graph_contract::ClassView`) — the one-stop entry point a renderer holds.
pub use ogar_class_view::OgarClassView;
/// The calcified canonical AR shape (attributes + family `Association`s).
pub use ogar_vocab::Class;

pub mod bridges;

pub mod actions;
pub use actions::OgarActionProvider;
pub mod rbac_impl;

pub use bridges::{
    HealthcarePort, OdooPort, OpenProjectPort, RedminePort, SmbPort, UnifiedBridge, WoaPort,
};
#[allow(deprecated)]
pub use bridges::{
    MedcareBridge, OdooBridge, OpenProjectBridge, RedmineBridge, SmbBridge, WoaBridge,
};

/// Codebook parity-guard — the drift fuse between OGAR's authoritative codebook
/// (`ogar_vocab::class_ids::ALL`) and the contract's zero-dep wire mirror
/// (`lance_graph_contract::ogar_codebook::CODEBOOK`).
///
/// **The compile-time `COUNT_FUSE` was REMOVED 2026-08-14** in favour of
/// plug-and-play activation — see this crate's module docs for the incident and
/// the reasoning. Two facts made the deletion safe rather than a loosening:
///
/// 1. [`assert_codebook_parity`] already checks the FULL bijection (forward,
///    reverse, domain agreement), which strictly contains the length equality the
///    fuse asserted. The fuse detected a subset of what the test detects.
/// 2. The fuse's one unique property — firing during `cargo build` — is precisely
///    what made it harmful: this crate is workspace-excluded, so the fuse did not
///    gate the PRODUCER's CI, only every CONSUMER's build. It converted a
///    producer-side bookkeeping lapse into a downstream outage, for concepts the
///    consumer did not use.
///
/// Drift now surfaces where it can be acted on: [`assert_codebook_parity`] in
/// tests / at consumer startup, and [`super::OgarAuthority`] at the point a
/// consumer actually plugs a classid.
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
                | (O::Weather, C::Weather)
                | (O::Osint, C::Osint)
                | (O::Ocr, C::Ocr)
                | (O::Health, C::Health)
                | (O::Anatomy, C::Anatomy)
                | (O::Auth, C::Auth)
                | (O::Automation, C::Automation)
                | (O::HR, C::HR)
                | (O::Genetics, C::Genetics)
                | (O::Geo, C::Geo)
                | (O::Ontology, C::Ontology)
                | (O::Blocks, C::Blocks)
                | (O::JavaRuntime, C::JavaRuntime)
                | (O::Analytics, C::Analytics)
                | (O::BinaryLifting, C::BinaryLifting)
                | (O::Unassigned, C::Unassigned)
        )
    }

    /// Assert the mirror is a faithful, complete copy of OGAR's codebook —
    /// forward (mirror ⊆ OGAR), reverse (OGAR ⊆ mirror), and domain agreement.
    /// Returns the number of concepts checked. Panics on any divergence.
    pub fn assert_codebook_parity() -> usize {
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
        fn reserved_empty_domains_agree_across_the_mirror() {
            // The codebook-parity walk above only visits ids that carry
            // CONCEPT ROWS, so a reserved-EMPTY domain (Blocks, the C-band)
            // added to one enum but not the other would slip past it — the
            // exact drift this pairing exists to catch. Pin one id per
            // reserved/new domain, an id from a POPULATED new domain
            // (0x0333, the DisMech mints in Ontology), the deliberate
            // C2-C3 gap, and the 0x0C/0xC0 digit-swap hazard two-sided.
            for id in [
                0x0300u16, 0x0333, // Ontology (populated: dismech)
                0x1701, 0x17FF, // Blocks
                0xC000, 0xC0FF, // JavaRuntime (Panama FFM alone)
                0xC100, // Analytics
                0xC400, // BinaryLifting
                0xC200, 0xC300, // the deliberate gap (both Unassigned)
                0xBF00, 0xC500, // band edges (both Unassigned)
                0x0C01, 0xC001, // Automation vs JavaRuntime, transposed
            ] {
                assert!(domains_agree(id), "domain drift at {id:#06x}");
            }
        }

        #[test]
        fn classid_low_u16_is_the_codebook_id() {
            use lance_graph_contract::NodeGuid;
            let project_id = ogar_vocab::canonical_concept_id("project").unwrap();
            let guid = NodeGuid::new(u32::from(project_id), 0, 0, 0, 0, 0);
            assert_eq!(guid.classid() as u16, project_id);
            assert!(domains_agree(project_id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authoritative_ocr_table_roundtrip_is_green() {
        use ogar_vocab::ocr_actions;

        assert!(
            !ocr_actions::OCR_EXPECTED_EXECUTORS.is_empty(),
            "authority declared no expected executor for the OCR table"
        );
        let mirror: std::collections::BTreeMap<&str, u16> =
            lance_graph_contract::ogar_codebook::CODEBOOK
                .iter()
                .copied()
                .collect();
        for &id in ocr_actions::OCR_SUBJECT_CLASSIDS {
            assert!(
                mirror.values().any(|&v| v == id),
                "OCR table subject 0x{id:04X} missing from the wire mirror"
            );
        }
        let names = ocr_actions::OCR_ACTION_NAMES;
        assert!(!names.is_empty());
        let set: std::collections::BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(set.len(), names.len(), "duplicate capability names");
    }

    #[test]
    fn ogar_class_view_implements_contract_class_view() {
        use lance_graph_contract::class_view::ClassView;
        let view = OgarClassView::new();
        let _as_trait: &dyn ClassView = &view;
    }
}

/// The generic hot-plug bridge (operator, 2026-07-07): "lance-graph-contract
/// pulling into OGAR with a generic activation." The contract defines the
/// zero-dep SOCKET ([`lance_graph_contract::hotplug`]); OGAR owns the data
/// resolution (`ogar_vocab::capability_registry::resolve_hotplug`); THIS
/// workspace-excluded crate is where the two meet in one binary. A consumer
/// declares a [`lance_graph_contract::hotplug::HotPlug`] const and activates
/// it through this authority (or calls `resolve_hotplug` directly — same
/// data path, same drift arms).
pub struct OgarAuthority;

impl lance_graph_contract::hotplug::CapabilityAuthority for OgarAuthority {
    fn activate(
        &self,
        plug: &lance_graph_contract::hotplug::HotPlug,
    ) -> Result<
        lance_graph_contract::hotplug::Activation,
        lance_graph_contract::hotplug::ActivationDrift,
    > {
        use lance_graph_contract::hotplug::{Activation, ActivationDrift};
        use ogar_vocab::capability_registry::{resolve_hotplug, HotplugDrift};

        match resolve_hotplug(plug.consumer, plug.classids, plug.covered) {
            Ok((concepts, capabilities)) => {
                let concepts: Vec<(String, u16)> = concepts
                    .into_iter()
                    .map(|(name, id)| (name.to_string(), id))
                    .collect();
                if let Some(drift) = lance_graph_contract::hotplug::verify_against_mirror(&concepts)
                {
                    return Err(drift);
                }
                Ok(Activation {
                    concepts,
                    capabilities,
                })
            }
            Err(HotplugDrift::UnknownClassid(id)) => Err(ActivationDrift::UnknownClassid(id)),
            Err(HotplugDrift::NoCapabilitiesFor(id)) => Err(ActivationDrift::NoCapabilitiesFor(id)),
            Err(HotplugDrift::UnexpectedConsumer(c)) => Err(ActivationDrift::UnexpectedConsumer(c)),
            Err(HotplugDrift::Uncovered(c)) => Err(ActivationDrift::Uncovered(c)),
            Err(HotplugDrift::Undeclared(c)) => Err(ActivationDrift::Undeclared(c)),
        }
    }
}

#[cfg(test)]
mod hotplug_bridge_tests {
    use lance_graph_contract::hotplug::{CapabilityAuthority, HotPlug};

    #[test]
    fn ocr_hotplug_activates_through_the_contract_socket() {
        let plug = HotPlug {
            consumer: "tesseract-ogar",
            classids: &[0x0805, 0x0808, 0x0809],
            covered: &[
                "recognize_line",
                "recognize_page",
                "extract_text_layer",
                "extract_page_image",
                "recognize_page_words",
                "recognize_document",
                "segment_page",
                "detect_halftone_regions",
                "render_text",
                "render_tsv",
                "render_hocr",
                "render_searchable_pdf",
            ],
        };
        let auth: &dyn CapabilityAuthority = &super::OgarAuthority;
        let act = auth.activate(&plug).expect("green activation");
        assert_eq!(act.concepts.len(), 3);
        assert!(act.concepts.contains(&("textline".to_string(), 0x0805)));
        assert_eq!(act.capabilities.len(), 12);
    }

    #[test]
    fn an_unknown_classid_drifts_at_the_plug_not_at_the_build() {
        use lance_graph_contract::hotplug::ActivationDrift;

        let auth: &dyn CapabilityAuthority = &super::OgarAuthority;
        let bogus = HotPlug {
            consumer: "tesseract-ogar",
            classids: &[0xDEAD],
            covered: &[],
        };
        assert!(
            matches!(
                auth.activate(&bogus),
                Err(ActivationDrift::UnknownClassid(0xDEAD))
            ),
            "a plugged classid that is not minted must drift by NAME"
        );

        assert_eq!(
            ogar_vocab::canonical_concept_id("osm_street_node"),
            Some(0x0F0B),
            "the regression fixture must remain a minted OGAR concept"
        );

        let ocr_only = HotPlug {
            consumer: "tesseract-ogar",
            classids: &[0x0805],
            covered: &["recognize_line"],
        };
        let act = auth
            .activate(&ocr_only)
            .expect("plugging one id must not be affected by concepts elsewhere in the codebook");
        assert_eq!(act.concepts, vec![("textline".to_string(), 0x0805)]);
    }
}
