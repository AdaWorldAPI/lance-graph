//! Per-family codebook table — inline label + schema + verbs per OGIT slot
//! (per `.claude/plans/super-domain-rbac-tenancy-v1.md` §3.3 / D-SDR-3).
//!
//! Each OGIT family (basin) carries its own `OgitFamilyTable` — a sparse
//! map indexed by `OwlIdentity::slot()` (u16, full registry width). Each
//! occupied slot holds the label URI, schema kind (Entity / Edge /
//! Attribute), OWL property characteristics, DOLCE upper marker, axiom
//! blob, dcterms:source provenance, and outgoing-verb slot list
//! **inline** — no sidecar table, no join, one cache-line per slot.
//!
//! Per the spec's brutal-honest correction (§16.5 + §17.4): inline storage
//! is the Foundry-parity-shaped surface; the earlier sidecar sketch was
//! Neo4j-shaped and rejected.
//!
//! The original D-SDR-3 sketch used `[Option<FamilyEntry>; 256]`; PR #364
//! review surfaced that registry IDs allocate globally as u16, so a dense
//! 256-slot table would alias slot collisions across distinct entities.
//! The dense array is therefore replaced by a `HashMap<u16, FamilyEntry>`
//! preserving O(1) lookup while honoring the full slot domain.
//!
//! ## Hot path
//!
//! ```text
//! caller's OwlIdentity (family u8 + slot u16)
//!     │
//!     ▼ table.lookup(owl)
//! &FamilyEntry  ← one hash probe (`entries.get(&owl.slot())`)
//!     │
//!     ├─ label_uri:     &'static str    (display / audit)
//!     ├─ kind:          SchemaKind      (Entity / Edge / Attribute)
//!     ├─ owl_chars:     OwlCharacteristics (1 byte bitfield)
//!     ├─ dolce_marker:  DolceMarker     (1 byte enum)
//!     ├─ axiom_blob:    &'static [u8]   (subClassOf / equivalentClass / ...)
//!     ├─ provenance:    &'static str    (dcterms:source — off-label lineage)
//!     └─ verbs:         &'static [u8]   (outgoing verb slots within this family)
//! ```
//!
//! D-SDR-3 scope: type system + lookup method. Bake-time population from TTL
//! hydration is D-SDR-3b (lance-graph-ontology side). `PerFamilyCodebook` is
//! a placeholder for the per-family CAM-PQ centroids (D-SDR-3c).

use std::collections::HashMap;

use crate::super_domain::DolceMarker;
use crate::unified_bridge::{OgitFamily, OwlIdentity};

// Reuse the canonical SchemaKind from lance-graph-ontology; FamilyEntry tags
// each slot with one of its three variants so a caller knows whether the
// slot refers to an Entity / Edge / Attribute without re-querying the
// registry.
pub use lance_graph_ontology::namespace::SchemaKind;

// ═══════════════════════════════════════════════════════════════════════════
// OwlCharacteristics — 1-byte bitfield (§3.3 / GLUE_LAYER_OGIT_TO_OWL_SPEC.md)
// ═══════════════════════════════════════════════════════════════════════════

/// 1 byte. OWL property characteristics bitfield per the
/// `GLUE_LAYER_OGIT_TO_OWL_SPEC.md` §3 layout. Lives inline in
/// `FamilyEntry`; consultant-shaped axiom refinement that consumers can
/// ignore unless they need it.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct OwlCharacteristics(pub u8);

impl OwlCharacteristics
{
    pub const FUNCTIONAL: u8 = 1 << 0;
    pub const INVERSE_FUNCTIONAL: u8 = 1 << 1;
    pub const TRANSITIVE: u8 = 1 << 2;
    pub const SYMMETRIC: u8 = 1 << 3;
    pub const ASYMMETRIC: u8 = 1 << 4;
    pub const REFLEXIVE: u8 = 1 << 5;
    pub const IRREFLEXIVE: u8 = 1 << 6;
    /// Reserved / DOLCE-specific extension bit.
    pub const RESERVED: u8 = 1 << 7;

    pub const EMPTY: Self = Self(0);

    #[inline]
    pub const fn from_bits(bits: u8) -> Self
    {
        Self(bits)
    }

    #[inline]
    pub const fn bits(self) -> u8
    {
        self.0
    }

    #[inline]
    pub const fn is_functional(self) -> bool
    {
        self.0 & Self::FUNCTIONAL != 0
    }

    #[inline]
    pub const fn is_inverse_functional(self) -> bool
    {
        self.0 & Self::INVERSE_FUNCTIONAL != 0
    }

    #[inline]
    pub const fn is_transitive(self) -> bool
    {
        self.0 & Self::TRANSITIVE != 0
    }

    #[inline]
    pub const fn is_symmetric(self) -> bool
    {
        self.0 & Self::SYMMETRIC != 0
    }

    #[inline]
    pub const fn is_asymmetric(self) -> bool
    {
        self.0 & Self::ASYMMETRIC != 0
    }

    #[inline]
    pub const fn is_reflexive(self) -> bool
    {
        self.0 & Self::REFLEXIVE != 0
    }

    #[inline]
    pub const fn is_irreflexive(self) -> bool
    {
        self.0 & Self::IRREFLEXIVE != 0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FamilyEntry — one slot in OgitFamilyTable (label + schema + verbs inline)
// ═══════════════════════════════════════════════════════════════════════════

/// One slot's worth of OGIT data. Variable size depending on axiom blob;
/// typical ~80-200 bytes when populated. The struct is `'static` because
/// these tables are baked at hydration time (TTL → `OgitFamilyTable`) and
/// then read-only at runtime.
#[derive(Clone, Copy, Debug)]
pub struct FamilyEntry
{
    /// e.g. `"ogit.Network:IPAddress"` — the canonical OGIT URI.
    pub label_uri: &'static str,
    /// Entity / Edge / Attribute discriminant (reuses
    /// `lance_graph_ontology::namespace::SchemaKind`).
    pub kind: SchemaKind,
    /// OWL property characteristics bitfield (Functional / Transitive / ...).
    pub owl_characteristics: OwlCharacteristics,
    /// DOLCE upper marker (Endurant / Perdurant / Quality / Abstract).
    pub dolce_marker: DolceMarker,
    /// Optional OWL axiom blob — subClassOf / equivalentClass / ... encoded
    /// as TTL bytes. Most slots have `&[]`; only consultant-shaped refined
    /// classes carry axioms.
    pub axiom_blob: &'static [u8],
    /// `dcterms:source` — carries off-label lineage (e.g.
    /// `"OSLC-perfmon v3.0 § 4.2 (off-label fit: ...)"`). Per the §5 OSLC
    /// absorption decision: the off-label-ness rides here.
    pub provenance: &'static str,
    /// Outgoing verb slots (within the same family). Empty for attributes
    /// and most entities; populated when this entry IS an edge type.
    pub verbs: &'static [u8],
}

impl FamilyEntry
{
    /// Construct a minimal `FamilyEntry` for a plain entity — no OWL axioms,
    /// no DOLCE refinement, no provenance. Used by tests and the simplest
    /// hydration paths.
    pub const fn plain_entity(label_uri: &'static str) -> Self
    {
        Self {
            label_uri,
            kind: SchemaKind::Entity,
            owl_characteristics: OwlCharacteristics::EMPTY,
            dolce_marker: DolceMarker::Unknown,
            axiom_blob: &[],
            provenance: "",
            verbs: &[],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PerFamilyCodebook — per-basin compression substrate (§3.3 placeholder)
// ═══════════════════════════════════════════════════════════════════════════

/// Per-family compression substrate — placeholder for D-SDR-3c (per-basin
/// CAM-PQ centroids + Base17 head + scent). For D-SDR-3 minimum the table
/// works without it; concrete fields land alongside the
/// `lance-graph-contract::cam` codec wiring.
#[derive(Clone, Copy, Debug, Default)]
pub struct PerFamilyCodebook;

// ═══════════════════════════════════════════════════════════════════════════
// OgitFamilyTable — the 256-slot per-basin codebook
// ═══════════════════════════════════════════════════════════════════════════

/// One table per OGIT family. Lives in static memory after hydration —
/// sparse `HashMap<u16, FamilyEntry>` keyed by `OwlIdentity::slot()`.
/// Each occupied slot holds a `FamilyEntry` (label + schema + verbs
/// inline).
///
/// Typical size scales with occupied-slot count, not the u16 keyspace:
/// ~50-200 KB per family depending on entry count + axiom blob sizes.
/// With ~75 active basins on a hydrated registry, the resident set is
/// ~5-15 MB. Lookups are O(1) hash probes (sub-microsecond).
pub struct OgitFamilyTable
{
    pub family: OgitFamily,
    pub entries: HashMap<u16, FamilyEntry>,
    pub codebook: PerFamilyCodebook,
}

impl OgitFamilyTable
{
    /// Construct an empty table for the given family. Hydration populates
    /// `entries` as TTL classes / properties are discovered.
    pub fn empty(family: OgitFamily) -> Self
    {
        Self {
            family,
            entries: HashMap::new(),
            codebook: PerFamilyCodebook,
        }
    }

    /// Hot-path lookup: O(1) hash probe. Sub-microsecond.
    ///
    /// `debug_assert`s that the `OwlIdentity` belongs to this family — in
    /// release builds the assertion is elided, so callers MUST ensure the
    /// table was selected via `FAMILY_TO_SUPER_DOMAIN`-style routing first.
    #[inline]
    pub fn lookup(&self, owl: OwlIdentity) -> Option<&FamilyEntry>
    {
        debug_assert_eq!(
            owl.family().raw(),
            self.family.raw(),
            "OwlIdentity family {} does not match table family {}",
            owl.family().raw(),
            self.family.raw(),
        );
        self.entries.get(&owl.slot())
    }

    /// Resolve to the canonical label URI for a slot, e.g.
    /// `"ogit.Network:IPAddress"`. Returns `None` if the slot is empty.
    #[inline]
    pub fn label(&self, owl: OwlIdentity) -> Option<&str>
    {
        self.lookup(owl).map(|e| e.label_uri)
    }

    /// What kind of dictionary entry this is (Entity / Edge / Attribute).
    #[inline]
    pub fn kind(&self, owl: OwlIdentity) -> Option<SchemaKind>
    {
        self.lookup(owl).map(|e| e.kind)
    }

    /// Does this slot carry the `Functional` OWL characteristic? Cheap
    /// helper for the MUL planner's veto path.
    #[inline]
    pub fn is_functional(&self, owl: OwlIdentity) -> bool
    {
        self.lookup(owl)
            .is_some_and(|e| e.owl_characteristics.is_functional())
    }

    /// Does this slot carry the `Transitive` OWL characteristic? Used by
    /// the planner's closure-expansion hint.
    #[inline]
    pub fn is_transitive(&self, owl: OwlIdentity) -> bool
    {
        self.lookup(owl)
            .is_some_and(|e| e.owl_characteristics.is_transitive())
    }

    /// Insert / overwrite a slot. Used by hydration; runtime code stays
    /// read-only against `&OgitFamilyTable`.
    pub fn set(&mut self, slot: u16, entry: FamilyEntry)
    {
        self.entries.insert(slot, entry);
    }

    /// Drop a slot's entry. Used by retraction during re-hydration.
    pub fn clear(&mut self, slot: u16)
    {
        self.entries.remove(&slot);
    }

    /// Number of occupied slots in this table. O(1).
    pub fn len(&self) -> usize
    {
        self.entries.len()
    }

    /// True when no slots are occupied.
    pub fn is_empty(&self) -> bool
    {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    const TEST_FAMILY: OgitFamily = OgitFamily(7);

    #[test]
    fn empty_table_has_no_entries()
    {
        let t = OgitFamilyTable::empty(TEST_FAMILY);
        assert_eq!(t.family.raw(), 7);
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn lookup_returns_inserted_entry()
    {
        let mut t = OgitFamilyTable::empty(TEST_FAMILY);
        let entry = FamilyEntry::plain_entity("ogit.Test:Patient");
        t.set(42, entry);

        let owl = OwlIdentity::new(TEST_FAMILY, 42);
        let got = t.lookup(owl).expect("slot 42 should be populated");
        assert_eq!(got.label_uri, "ogit.Test:Patient");
        assert_eq!(got.kind, SchemaKind::Entity);
        assert!(got.axiom_blob.is_empty());
    }

    #[test]
    fn lookup_returns_none_for_empty_slot()
    {
        let t = OgitFamilyTable::empty(TEST_FAMILY);
        let owl = OwlIdentity::new(TEST_FAMILY, 0);
        assert!(t.lookup(owl).is_none());
    }

    #[test]
    fn label_kind_helpers_match_lookup()
    {
        let mut t = OgitFamilyTable::empty(TEST_FAMILY);
        t.set(5, FamilyEntry::plain_entity("ogit.Healthcare:Diagnose"));

        let owl = OwlIdentity::new(TEST_FAMILY, 5);
        assert_eq!(t.label(owl), Some("ogit.Healthcare:Diagnose"));
        assert_eq!(t.kind(owl), Some(SchemaKind::Entity));
    }

    #[test]
    fn owl_characteristics_bits_round_trip()
    {
        let chars = OwlCharacteristics::from_bits(
            OwlCharacteristics::FUNCTIONAL | OwlCharacteristics::TRANSITIVE,
        );
        assert!(chars.is_functional());
        assert!(chars.is_transitive());
        assert!(!chars.is_symmetric());
        assert!(!chars.is_inverse_functional());
        assert_eq!(chars.bits(), 0b0000_0101);
    }

    #[test]
    fn is_functional_helper_reads_slot()
    {
        let mut t = OgitFamilyTable::empty(TEST_FAMILY);
        let entry = FamilyEntry {
            label_uri: "ogit.Network:hostname",
            kind: SchemaKind::Attribute,
            owl_characteristics: OwlCharacteristics::from_bits(OwlCharacteristics::FUNCTIONAL),
            dolce_marker: DolceMarker::Quality,
            axiom_blob: &[],
            provenance: "",
            verbs: &[],
        };
        t.set(12, entry);

        let owl = OwlIdentity::new(TEST_FAMILY, 12);
        assert!(t.is_functional(owl));
        assert!(!t.is_transitive(owl));
    }

    #[test]
    fn clear_removes_entry()
    {
        let mut t = OgitFamilyTable::empty(TEST_FAMILY);
        t.set(99, FamilyEntry::plain_entity("ogit.Test:Removable"));
        assert_eq!(t.len(), 1);
        t.clear(99);
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
    }

    #[test]
    fn len_counts_populated_slots_only()
    {
        let mut t = OgitFamilyTable::empty(TEST_FAMILY);
        for slot in [0, 1, 5, 17, 200] {
            t.set(slot, FamilyEntry::plain_entity("ogit.Test:Multi"));
        }
        assert_eq!(t.len(), 5);
    }

    #[test]
    fn slot_keyspace_distinguishes_high_ids()
    {
        // PR #364 review: registry IDs allocate globally as u16, so slots
        // that differ by 256 used to alias under the old u8 truncation.
        // Lock that two slots in the upper half of the keyspace stay
        // distinct after the widening.
        let mut t = OgitFamilyTable::empty(TEST_FAMILY);
        t.set(7, FamilyEntry::plain_entity("ogit.Test:Low"));
        t.set(7 + 256, FamilyEntry::plain_entity("ogit.Test:High"));
        let low = t.lookup(OwlIdentity::new(TEST_FAMILY, 7));
        let high = t.lookup(OwlIdentity::new(TEST_FAMILY, 7 + 256));
        assert_eq!(low.unwrap().label_uri, "ogit.Test:Low");
        assert_eq!(high.unwrap().label_uri, "ogit.Test:High");
        assert_eq!(t.len(), 2);
    }

    #[test]
    #[should_panic(expected = "does not match table family")]
    fn lookup_panics_on_wrong_family_in_debug()
    {
        let t = OgitFamilyTable::empty(TEST_FAMILY);
        // Different family — debug_assert should fire.
        let wrong = OwlIdentity::new(OgitFamily(99), 0);
        let _ = t.lookup(wrong);
    }
}
