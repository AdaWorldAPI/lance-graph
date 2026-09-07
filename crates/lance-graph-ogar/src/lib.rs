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
pub mod recipe_vocab;

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
        // DERIVED, never enumerated. This used to be a 19-arm
        // `matches!((O::X, C::X) | ...)` listing every domain pair by hand —
        // which is a LOCK: a hand-maintained mirror of the authority that has
        // to be bumped every time the authority mints a domain, and whose
        // omission shows up as a parity failure that looks like real drift.
        // (Measured: the 0xC6 `Mmio` mint failed here for exactly that reason,
        // and the first fix attempted was to add a 20th arm — answering a
        // stale pin with another pin.)
        //
        // The two enums are wire-compatible by construction (same `id >> 8`
        // discriminant, same variant names — the contract's own doc comment
        // says so), so the agreement to check is that both sides NAME the id's
        // domain identically. `Debug` gives the variant name as data, so a new
        // domain minted on both sides agrees with no edit here, and one minted
        // on only one side still fails: the authority reports its new name
        // while the mirror reports `Unassigned`.
        format!("{:?}", ogar_vocab::canonical_concept_domain(id))
            == format!("{:?}", mirror::canonical_concept_domain(id))
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

    /// Assert the contract's zero-dep predicate-palette mirror
    /// ([`lance_graph_contract::dismech_evidence::DISMECH_PREDICATES`]) is a
    /// faithful, complete copy of the authority (`ogar_dismech::RELATIONS`) —
    /// forward, reverse, and the position-lookup agreement. Returns the number
    /// of predicates checked. Panics on any divergence.
    ///
    /// **This is the membrane half of D-DCR-1 (W1).** The replay core
    /// (`lance_graph_planner::dismech_replay`) addresses each recorded step by
    /// a plain `u8` ordinal — it is in-workspace and cannot reach OGAR, which
    /// lives in this excluded armed tier. So the claim *"these ordinals ARE
    /// the dismech palette"* is proved HERE, against the real palette, rather
    /// than asserted in the core's prose.
    ///
    /// Both directions are load-bearing and catch different drift: forward
    /// catches a mirror row the palette no longer mints (a stale copy);
    /// reverse catches a newly minted predicate the mirror never learned about
    /// (the silent one — a replay would refuse a legitimate ordinal, and
    /// nothing in the planner could tell that from a corrupt chain).
    pub fn assert_dismech_palette_parity() -> usize {
        use lance_graph_contract::dismech_evidence as mirror;

        for &(ord, name, curie) in mirror::DISMECH_PREDICATES {
            let authority = ogar_dismech::by_index(ogar_loco::FnIndex(ord)).unwrap_or_else(|| {
                panic!("mirror has {ord:#04x} but the palette does not mint it")
            });
            assert_eq!(
                authority.name, name,
                "{ord:#04x}: mirror says {name:?}, palette says {:?}",
                authority.name
            );
            assert_eq!(authority.curie, curie, "{ord:#04x}: CURIE disagreement",);
        }
        for p in ogar_dismech::RELATIONS {
            let row = mirror::dismech_predicate(p.index.0).unwrap_or_else(|| {
                panic!(
                    "palette mints {} ({:#04x}) but the contract mirror is missing it",
                    p.name, p.index.0
                )
            });
            assert_eq!(row.1, p.name, "{:#04x}: reverse name mismatch", p.index.0);
            assert_eq!(row.2, p.curie, "{:#04x}: reverse CURIE mismatch", p.index.0);
        }
        assert_eq!(
            mirror::DISMECH_PREDICATE_FLOOR,
            ogar_dismech::CAUSES.0,
            "the mirror's band floor is not the palette's",
        );
        assert_eq!(
            mirror::DISMECH_PREDICATES.len(),
            ogar_dismech::RELATIONS.len(),
            "palette and mirror mint a different number of predicates",
        );
        ogar_dismech::RELATIONS.len()
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
            // CONCEPT ROWS, so a reserved-EMPTY domain (Ontology, Blocks, the
            // C-band) added to one enum but not the other would slip past it
            // — the exact drift this pairing exists to catch. Pin one id per
            // reserved domain, the deliberate C2-C3 gap, and the 0x0C/0xC0
            // digit-swap hazard two-sided.
            //
            // Ontology (0x03XX) is reserved+row-free on BOTH sides by design
            // (OBO reference concepts live in the producer crate `ogar-obo`,
            // never in the shared codebook) — no populated id exists to pin
            // here, unlike ProjectMgmt/Commerce/etc.
            for id in [
                0x0300u16, 0x0333, // Ontology (reserved, zero rows)
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
        fn the_contract_mirror_is_a_faithful_copy_of_the_dismech_palette() {
            let n = assert_dismech_palette_parity();
            // Anti-vacuity: a parity walk over an EMPTY mirror passes both
            // directions trivially. The palette is the closed measured set of
            // 19 upstream causal predicates; if that number ever moves, this
            // is the line that forces someone to look at why.
            assert_eq!(n, 19, "the DisMech palette mints 19 causal predicates");
        }

        #[test]
        fn the_search_band_is_not_swallowed_by_the_predicate_mirror() {
            // The palette mints a SECOND band immediately above the
            // predicates (`CANDIDATES` = 0xA3 and up — the Sudoku search ops,
            // deliberately kept out of the closed measured predicate set).
            // Contiguity makes an off-by-one in the mirror's bound both easy
            // and silent: it would resolve a search op AS a causal predicate,
            // and a replay would then travel a step under a verb that is not
            // a relation at all.
            use lance_graph_contract::dismech_evidence as mirror;
            assert!(
                mirror::dismech_predicate(ogar_dismech::CANDIDATES.0).is_none(),
                "the mirror resolved a SEARCH op as a causal predicate",
            );
            // ...and the authority agrees it is a real slot, so this test is
            // pinning a boundary rather than a byte that means nothing.
            assert!(
                ogar_dismech::search_op_by_index(ogar_dismech::CANDIDATES).is_some(),
                "0xA3 must be a real search op, or this test proves nothing",
            );
            assert!(ogar_dismech::by_index(ogar_dismech::CANDIDATES).is_none());
        }

        #[test]
        fn a_replay_ordinal_resolves_to_the_palette_predicate_it_names() {
            // The seam D-DCR-1 exists to pin. `lance_graph_planner`'s
            // `ChainStep = (u8, CausalEdge64)` carries the predicate as a bare
            // byte because the planner is in-workspace and cannot reach OGAR.
            // Here — where the real palette IS reachable — walk the same
            // ordinals a recorded chain would carry and prove each one names
            // the predicate the plan says it does.
            use lance_graph_contract::dismech_evidence as mirror;
            for (ordinal, expected) in [
                (0x90u8, "causes"),
                (0x94, "predisposes_to"),
                (0x9D, "perturbs"),
                (0xA2, "variant_of"),
            ] {
                let via_mirror = mirror::dismech_predicate(ordinal)
                    .unwrap_or_else(|| panic!("{ordinal:#04x} unresolved in the mirror"));
                let via_palette = ogar_dismech::by_index(ogar_loco::FnIndex(ordinal))
                    .unwrap_or_else(|| panic!("{ordinal:#04x} unresolved in the palette"));
                assert_eq!(via_mirror.1, expected);
                assert_eq!(via_palette.name, expected);
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

use lance_graph_contract::canonical_node::{EdgeCodecFlavor, ReadMode, TailVariant, ValueSchema};
use lance_graph_contract::hotplug::HotPlug;

/// The reading for every classid a plug declares — **derived, not tabulated**.
///
/// Operator, 2026-09-07: *"plug and play already has all the domains, you
/// could simply make the global schema for all appids already in plug-and-play
/// pattern activate V3 and be silent about all others"*, and *"local quad
/// usage must mint any V3 settings in plug and play, regardless of the
/// settings here."*
///
/// **What this replaced, and why it was wrong.** The first cut held a
/// `const LOCO_READ_MODES` with ONE ROW per consumer seat (`0x1717` alone),
/// answered only when the plug covered every declared seat, and returned
/// `&[]` otherwise. Three faults, in increasing order of seriousness:
///
/// 1. Adding a frontend meant adding a row to a shared table — the central
///    lockstep `D-BLOCKS-HOTPLUG-1` retired, rebuilt one level up.
/// 2. Every consumer outside `0x17XX` — `medcare-rs`'s six Health classids
///    among them — got NO reading at all, so a session asking for one hit
///    `NoReadingFor` and had to go hunting for a setting.
/// 3. Both failures are silent and REMOTE. A missing row does not break a
///    build; it surfaces much later as a V1 tail where V3 was expected —
///    *"otherwise I will spend weeks until I remember that we changed it
///    here wondering why quad 4x24 stopped working."*
///
/// **Now: being plugged in IS the declaration.** Every classid in the plug
/// gets [`ReadMode::PLUG_AND_PLAY_V3`], except where this authority
/// deliberately overrides one concept. Anything NOT plugged gets no entry —
/// silent, and `Activation::read_mode_for` bangs rather than defaulting.
///
/// The quad (`LegacyOutlier::WideTriple`, G2 `4 × u24`) rides on this: it is a
/// carving of the 12-byte content-blind payload, which exists as such only
/// under a V3 tail. A plugged consumer therefore keeps its quad without
/// knowing this function exists.
fn plug_readings(plug: &HotPlug) -> Vec<(u16, ReadMode)> {
    plug.classids
        .iter()
        .map(|&id| {
            (
                id,
                concept_override(id).unwrap_or(ReadMode::PLUG_AND_PLAY_V3),
            )
        })
        .collect()
}

/// Per-concept deviations from [`ReadMode::PLUG_AND_PLAY_V3`].
///
/// An override changes ONLY what a class genuinely reads differently; the
/// tail stays V3 for everything plugged, which is the ruling. `blockly-rs`'s
/// palette seat materialises no value tenants (it stores an `ogar-loco`
/// function body in the slab, not cognitive columns), so it reads
/// `Bootstrap` rather than `Full`.
///
/// This is a short list of exceptions, not a registry of participants: a
/// consumer absent from it is not absent from plug-and-play, it simply has
/// nothing unusual to say. That asymmetry is the point — the failure mode of
/// a forgotten entry is "no override", never "no reading".
const fn concept_override(classid: u16) -> Option<ReadMode> {
    match classid {
        0x1717 => Some(ReadMode {
            tail_variant: TailVariant::V3,
            value_schema: ValueSchema::Bootstrap,
            edge_codec: EdgeCodecFlavor::CoarseOnly,
        }),
        _ => None,
    }
}

/// `true` when a classid is a consumer palette seat in the `ogar-loco`
/// `0x17` domain (`0x1717` and up; `0x1701`/`0x1702` are the substrate's own
/// node shapes and `0x1703`-`0x1716` its reserved headroom).
///
/// A palette plug carries no capabilities, so it must not reach the
/// capability join — `resolve_hotplug` is pinned to answer `UnknownClassid`
/// for these ids by design.
const fn is_palette_seat(classid: u16) -> bool {
    classid >= 0x1717 && (classid >> 8) == 0x17
}

/// The consumer a palette seat is KNOWN to belong to, when one is known.
///
/// Deliberately a short list of CLAIMED seats, not a roster of participants.
/// `None` means "no consumer has claimed this seat", and an unclaimed seat is
/// plug-and-play for whoever plugs it — a new frontend at `0x1718` activates
/// and reads V3 with no edit here, which is the ruling.
///
/// What the list still buys is the ownership guard codex flagged on #1207: a
/// seat somebody HAS claimed cannot be activated by a different consumer, so
/// the reading cannot be obtained by impersonating one. Without it "who owns
/// `0x1717`" would be answerable by simply asking.
const fn palette_seat_owner(classid: u16) -> Option<&'static str> {
    match classid {
        0x1717 => Some("blockly-abi"),
        _ => None,
    }
}

/// The first plugged seat whose declared owner is somebody other than
/// `consumer`, if any.
fn impersonated_seat(consumer: &str, classids: &[u16]) -> Option<u16> {
    classids
        .iter()
        .copied()
        .find(|&id| matches!(palette_seat_owner(id), Some(owner) if owner != consumer))
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

        // The loco arm runs BEFORE the capability join, and must: a palette
        // classid deliberately does NOT resolve as a capability concept.
        // `ogar-vocab`'s own `a_palette_classid_does_not_resolve_as_a_hot_plug`
        // pins `resolve_hotplug("blockly-abi", &[0x1717], &[])` to
        // `UnknownClassid` — *"this is not a capability-authority concept at
        // all, in any build"* — and `no_0x17xx_row_reached_the_globally_
        // mirrored_codebook` keeps every `0x17XX` row out of `class_ids::ALL`
        // on purpose. So the join can never succeed here, and a reading placed
        // only in the `Ok` arm below would be unreachable for the one domain
        // it was written for.
        //
        // A STORAGE READING is a different seam from a CAPABILITY, exactly as
        // vocabulary routing is (the palette test says so in as many words).
        // This authority can therefore answer "no concepts, no capabilities,
        // and here is how your rows are read" without contradicting either
        // test — both are on `resolve_hotplug`, which is left untouched.
        //
        // PARTITION, do not choose (codex P2 on #1216). An earlier cut took
        // the palette arm only when EVERY id was `0x17XX`, so a mixed plug —
        // an unclaimed palette seat alongside ordinary capability ids, e.g.
        // `[0x1718, 0x0901]` — went whole to `resolve_hotplug` and came back
        // `UnknownClassid(0x1718)`. A consumer that legitimately has both
        // could not activate at all, which contradicts the ruling this arm
        // exists to implement: EVERY plugged appid reads V3.
        //
        // The two id kinds answer to different authorities, so they are
        // routed separately and the results merged; the reading covers the
        // whole plug either way, because it is derived from the plug.
        let (palette, capability): (Vec<u16>, Vec<u16>) =
            plug.classids.iter().partition(|&&id| is_palette_seat(id));

        // A CLAIMED seat is its owner's, whichever arm it arrives on. An
        // unclaimed one is plug-and-play for whoever plugs it.
        if impersonated_seat(plug.consumer, &palette).is_some() {
            return Err(ActivationDrift::UnexpectedConsumer(plug.consumer.into()));
        }

        // Pure-palette plug: no capability ids at all, so the join is not
        // consulted (it is pinned to refuse `0x17XX`).
        if !palette.is_empty() && capability.is_empty() {
            // Fails closed on a lie: a palette plug has no capabilities to
            // cover, so claiming one is drift, not an empty-set no-op.
            if let Some(cap) = plug.covered.first() {
                return Err(ActivationDrift::Undeclared((*cap).into()));
            }
            return Ok(Activation::new(Vec::new(), Vec::new(), plug_readings(plug)));
        }

        // Capability ids go to the join; palette seats were removed above so
        // they cannot make it refuse the whole plug.
        match resolve_hotplug(plug.consumer, &capability, plug.covered) {
            Ok((concepts, capabilities)) => {
                let concepts: Vec<(String, u16)> = concepts
                    .into_iter()
                    .map(|(name, id)| (name.to_string(), id))
                    .collect();
                if let Some(drift) = lance_graph_contract::hotplug::verify_against_mirror(&concepts)
                {
                    return Err(drift);
                }
                // EVERY plugged classid gets its reading here, not just the
                // `0x17XX` ones. A capability consumer (medcare-rs's six
                // Health ids, tesseract-rs's OCR seats) is as plugged in as a
                // palette, so it reads V3 too — that is the ruling, and it is
                // what keeps a quad working without its author having to find
                // this file.
                Ok(Activation::new(concepts, capabilities, plug_readings(plug)))
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

/// D-BLOCKS-HOTPLUG-1: the storage READING rides the activation, scoped to
/// `ogar-loco` (`0x17XX`) for now.
#[cfg(test)]
mod plug_and_play_reading {
    use lance_graph_contract::canonical_node::{
        classid_read_mode, ReadMode, TailVariant, ValueSchema,
    };
    use lance_graph_contract::hotplug::{ActivationDrift, CapabilityAuthority, HotPlug};

    const BLOCKLY: HotPlug = HotPlug {
        consumer: "blockly-abi",
        classids: &[0x1717],
        covered: &[],
    };

    /// CAN FIRE: blockly plugs in and is handed its reading — through the very
    /// call that `resolve_hotplug` refuses, which is the whole point.
    #[test]
    fn a_palette_plug_activates_with_a_reading_and_no_capabilities() {
        let act = super::OgarAuthority
            .activate(&BLOCKLY)
            .expect("palette plug activates");
        let mode = act.read_mode_for(0x1717).expect("its seat has a reading");
        assert_eq!(mode.tail_variant, TailVariant::V3);
        // blockly is the one declared override: it stores an ogar-loco body in
        // the slab, not cognitive tenants.
        assert_eq!(mode.value_schema, ValueSchema::Bootstrap);

        // A palette is not a capability concept; both arms are empty and that
        // is the correct answer, not a partial resolution.
        assert!(act.concepts.is_empty());
        assert!(act.capabilities.is_empty());

        // Anti-vacuity: the SAME ids through the capability join still fail,
        // so this cannot be passing because the join started succeeding.
        assert!(matches!(
            ogar_vocab::capability_registry::resolve_hotplug("blockly-abi", &[0x1717], &[]),
            Err(ogar_vocab::capability_registry::HotplugDrift::UnknownClassid(0x1717))
        ));
    }

    /// The reading did NOT come from the canon registry — the class stays
    /// unknown to `BUILTIN_READ_MODES`, which is what keeps the lockstep gone.
    #[test]
    fn the_canon_registry_still_does_not_know_the_palette_class() {
        assert_eq!(classid_read_mode(0x1717_1000), ReadMode::DEFAULT);
        assert_ne!(ReadMode::DEFAULT.tail_variant, TailVariant::V3);
    }

    /// THE RULING: a seat nobody has claimed is plug-and-play. A new frontend
    /// activates and reads V3 with NO edit to this file.
    ///
    /// This is the case the previous design got wrong: it held one hard-coded
    /// row per seat and refused everything else, so a second frontend silently
    /// lost its V3 tail until somebody remembered to add a row.
    #[test]
    fn an_unclaimed_seat_is_plug_and_play_for_whoever_plugs_it() {
        let newcomer = HotPlug {
            consumer: "some-future-frontend",
            classids: &[0x1718],
            covered: &[],
        };
        let act = super::OgarAuthority
            .activate(&newcomer)
            .expect("an unclaimed seat activates");
        assert_eq!(
            act.read_mode_for(0x1718).expect("reads V3"),
            ReadMode::PLUG_AND_PLAY_V3
        );
        assert_eq!(ReadMode::PLUG_AND_PLAY_V3.tail_variant, TailVariant::V3);

        // Anti-vacuity: 0x1718 appears in NO table in this crate. If the
        // reading came from a lookup rather than from being plugged, this
        // could not resolve.
        assert!(super::palette_seat_owner(0x1718).is_none());
        assert!(super::concept_override(0x1718).is_none());
    }

    /// …and a CLAIMED seat still belongs to its owner. Restores the ownership
    /// guard codex flagged on #1207, which the derive-don't-tabulate rewrite
    /// would otherwise have dropped.
    #[test]
    fn another_consumer_cannot_activate_a_claimed_seat() {
        let impostor = HotPlug {
            consumer: "scratch-abi",
            ..BLOCKLY
        };
        assert!(matches!(
            super::OgarAuthority.activate(&impostor),
            Err(ActivationDrift::UnexpectedConsumer(c)) if c == "scratch-abi"
        ));
        // Two-sided on the same seat: its real owner still activates, so the
        // guard discriminates rather than refusing 0x1717 outright.
        assert!(super::OgarAuthority.activate(&BLOCKLY).is_ok());
    }

    /// THE MEDCARE CASE — the one this rewrite exists for.
    ///
    /// A capability consumer outside `0x17XX` is as plugged in as a palette,
    /// so it reads V3 too. Under the previous design its activation carried NO
    /// reading and a session asking for one hit `NoReadingFor` — the footgun
    /// that surfaces weeks later as "why did quad 4x24 stop working", because
    /// the G2 `4 × u24` carving is a reading of the 12-byte payload that only
    /// exists under a V3 tail.
    #[test]
    fn a_capability_consumer_outside_the_loco_domain_also_reads_v3() {
        let medcare = HotPlug {
            consumer: "medcare-rs",
            classids: &[0x0901, 0x0902],
            covered: &[
                "register_patient",
                "get_patient_record",
                "list_patients",
                "update_patient_access",
                "add_diagnosis",
                "get_diagnosis",
                "list_diagnoses",
                "delete_diagnosis",
            ],
        };
        let act = super::OgarAuthority
            .activate(&medcare)
            .expect("the health plug activates");

        for id in [0x0901u16, 0x0902] {
            assert_eq!(
                act.read_mode_for(id).expect("every plugged id reads"),
                ReadMode::PLUG_AND_PLAY_V3,
                "0x{id:04X} must read V3 by being plugged in"
            );
        }
        // It really did go through the capability join, so this is the
        // ordinary path and not a palette short-circuit.
        assert!(!act.capabilities.is_empty());
        assert!(!act.concepts.is_empty());
    }

    /// A MIXED plug — an unclaimed palette seat alongside capability ids —
    /// activates, and every id gets its reading.
    ///
    /// Codex P2 on #1216. The arm used to take the palette path only when
    /// EVERY id was `0x17XX`, so a mixed plug went whole to
    /// `resolve_hotplug`, which is pinned to refuse a palette id — the
    /// consumer got `UnknownClassid` and could not activate at all. That
    /// contradicts the ruling this arm implements: every plugged appid reads
    /// V3.
    ///
    /// Anti-vacuity: the palette id is asserted to be one, and the capability
    /// ids are asserted NOT to be, so the fixture provably straddles the
    /// partition rather than being a capability-only plug in disguise.
    #[test]
    fn a_mixed_palette_and_capability_plug_activates_and_reads_v3_for_both() {
        let mixed = HotPlug {
            consumer: "medcare-rs",
            classids: &[0x1718, 0x0901, 0x0902],
            covered: &[
                "register_patient",
                "get_patient_record",
                "list_patients",
                "update_patient_access",
                "add_diagnosis",
                "get_diagnosis",
                "list_diagnoses",
                "delete_diagnosis",
            ],
        };
        assert!(
            super::is_palette_seat(0x1718),
            "fixture straddles the split"
        );
        assert!(!super::is_palette_seat(0x0901));
        assert!(!super::is_palette_seat(0x0902));

        let act = super::OgarAuthority
            .activate(&mixed)
            .expect("a mixed plug must activate, not bang on its palette half");

        for id in [0x1718u16, 0x0901, 0x0902] {
            assert_eq!(
                act.read_mode_for(id).expect("every plugged id reads"),
                ReadMode::PLUG_AND_PLAY_V3,
                "0x{id:04X} must read V3 by being plugged in"
            );
        }
        // The capability half really went through the join, so the palette id
        // was removed from it rather than the join being skipped wholesale.
        assert!(!act.capabilities.is_empty());
        assert!(act.concepts.iter().all(|(_, id)| *id != 0x1718));
    }

    /// …and a CLAIMED seat is still refused on the mixed path, so the
    /// partition did not open a hole around the ownership guard.
    #[test]
    fn a_mixed_plug_carrying_someone_elses_seat_is_still_refused() {
        let impostor = HotPlug {
            consumer: "medcare-rs",
            classids: &[0x1717, 0x0901],
            covered: &["register_patient"],
        };
        assert!(matches!(
            super::OgarAuthority.activate(&impostor),
            Err(ActivationDrift::UnexpectedConsumer(c)) if c == "medcare-rs"
        ));
    }

    /// CAN STAY SILENT: an UNPLUGGED concept still bangs.
    ///
    /// "V3 for all appids in plug-and-play" must not become "V3 for
    /// everything". Without this half the lookup would answer for any id, and
    /// a guard that fires on everything carries no information.
    #[test]
    fn an_unplugged_concept_is_silent_not_v3() {
        let act = super::OgarAuthority.activate(&BLOCKLY).expect("activates");
        assert_eq!(
            act.read_mode_for(0x0901),
            Err(ActivationDrift::NoReadingFor(0x0901)),
            "a concept this plug never declared has no reading"
        );
        assert_eq!(act.declared_readings().len(), 1);
    }

    /// Fails closed on a lie: a palette plug claiming a capability is drift.
    #[test]
    fn a_palette_plug_claiming_a_capability_bangs() {
        let lying = HotPlug {
            covered: &["lower_script"],
            ..BLOCKLY
        };
        assert!(matches!(
            super::OgarAuthority.activate(&lying),
            Err(ActivationDrift::Undeclared(c)) if c == "lower_script"
        ));
    }
}
