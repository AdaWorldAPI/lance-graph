//! `cognitive_palette` — the **226-atom palette256 FROZEN value codebook**.
//!
//! # What this is (operator ruling 2026-07-18)
//!
//! *"226 ARE the frozen; anything else needs 6×2×8bit (12 slots for an
//! Orchestration for v3 substrate replayability)."*
//!
//! The autopoiesis-triangle lanes ([`StyleLane`](crate::soa_view::StyleLane), on
//! `NodeRow` value tenants / `MailboxSoA` columns) are each a 12-byte
//! **content-blind register** — the same 96 bits the LE contract carves as
//! `6×(8:8)` / `4×(8:8:8)` / `3×(8:8:8:8)` / `12×8` (`.claude/v3/soa_layout/le-contract.md`
//! §3). This module names the **`12×u8` FROZEN reading**: each of the 12 slots
//! (a [`StyleFamily`](crate::style_family::StyleFamily) ordinal) holds ONE
//! **`AtomId`** — a palette256 index into this 226-atom catalogue.
//!
//! **The reading is ClassView-selected PER ROW/CLASS, never per lane** (the
//! le-contract §3 content-blind-register rule). Within a **policy / thinking-class**
//! row, ALL THREE triangle lanes — Frozen, Learned, Explore — are read as `12×u8`
//! palette atoms into THIS catalogue; that uniformity is what makes the autopoiesis
//! promotion `learned[f] → frozen[f]` a coherent `AtomId` copy (both operands are
//! palette atoms, never a byte reinterpreted across representations). The **other**
//! reading of the same 12 bytes — `6×(8:8)` (le-contract §3 L1/L4, replayable per
//! `E-H268-REPLAYABLE-TILE-1`) — is what an **orchestration-class** row selects for
//! ALL its lanes; it does NOT index this catalogue. One register, two ClassView-
//! selected readings (the triangle plan
//! `.claude/plans/triangle-tenants-gestalt-separation-v1.md` §4 "12 families | 12
//! template steps") — the discriminant is the ROW's class, so Frozen/Learned/Explore
//! never disagree on representation.
//!
//! # This is an ADDRESSING table, not a content store
//!
//! Per `I-VSA-IDENTITIES` (Layer 2 = domain role catalogues, Layer 3 = content
//! stores): this module owns only the **address space** — which palette index
//! resolves to which catalogue and local index. It is deliberately **zero-dep**
//! and does NOT import the catalogues' concrete types. The content each atom
//! points to lives in its own registry, resolved by the caller:
//!
//! | Catalogue | count | content registry (where the atom RESOLVES) |
//! |---|---|---|
//! | Verb    | 144 | `holograph::dntree` `DnVerb` (0..=143; 6 categories × 24) |
//! | Recipe  |  34 | `crate::recipes` `RECIPES` (the 34 NARS tactic runbooks) |
//! | Persona |  36 | `crate::thinking` `ThinkingStyle::ALL` (36 styles / 6 clusters) |
//! | Family  |  12 | `crate::style_family` `StyleFamily::ALL` (the 12 abstract families) |
//!
//! # Layout (operator-locked composition, RESERVE-DON'T-RECLAIM)
//!
//! The composition (`144 verb ∥ 34 recipe ∥ 36 persona ∥ 12 family; 30 reserved`)
//! is fixed by the triangle plan §1. Offsets are **permanent** once shipped — a
//! consumer stores a bare `u8` atom, so reordering a sub-range would silently
//! reinterpret every persisted lane. The catalogue is append-only into the 29
//! reserved slots; existing ranges never move.
//!
//! ```text
//!   0          NULL       (atom 0 = the null default — a zeroed lane reads all-null)
//!   1  ..= 144 Verb       (144; local = palette - 1)
//! 145  ..= 178 Recipe     ( 34; local = palette - 145)
//! 179  ..= 214 Persona    ( 36; local = palette - 179)
//! 215  ..= 226 Family     ( 12; local = palette - 215; local == StyleFamily ordinal)
//! 227  ..= 255 Reserved   ( 29; the append margin — never free real estate)
//! ```

/// A palette256 index — one byte, `0..=255`. Value `0` is the null default
/// (`AtomId::NULL`); `1..=226` address the four catalogues; `227..=255` are the
/// append-only reserve. Stored bare (`u8`) in a `12×u8` FROZEN lane slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AtomId(pub u8);

/// The null default — an un-populated FROZEN lane slot reads this (never a wrong
/// policy). Matches the `canonical_node` "atom 0 = null default" convention.
impl AtomId {
    /// The null / unassigned atom (palette index `0`).
    pub const NULL: AtomId = AtomId(0);
}

// ── Range layout (operator-locked; RESERVE-DON'T-RECLAIM) ──

/// Count of Verb atoms (`holograph::dntree` `DnVerb`, 6 categories × 24).
pub const VERB_COUNT: u8 = 144;
/// Count of Recipe atoms (the 34 NARS tactic runbooks, `crate::recipes`).
pub const RECIPE_COUNT: u8 = 34;
/// Count of Persona atoms (`crate::thinking` `ThinkingStyle::ALL`, 36 styles).
pub const PERSONA_COUNT: u8 = 36;
/// Count of Family atoms (`crate::style_family` `StyleFamily::ALL`, 12 families).
pub const FAMILY_COUNT: u8 = 12;

/// First palette index of the Verb range (`1`; index `0` is `NULL`).
pub const VERB_BASE: u8 = 1;
/// First palette index of the Recipe range.
pub const RECIPE_BASE: u8 = VERB_BASE + VERB_COUNT; // 145
/// First palette index of the Persona range.
pub const PERSONA_BASE: u8 = RECIPE_BASE + RECIPE_COUNT; // 179
/// First palette index of the Family range.
pub const FAMILY_BASE: u8 = PERSONA_BASE + PERSONA_COUNT; // 215
/// First palette index of the reserved append-margin range.
pub const RESERVED_BASE: u8 = FAMILY_BASE + FAMILY_COUNT; // 227

/// Total addressed atoms (`226` — the "226 ARE the frozen" catalogue).
pub const ATOM_COUNT: u16 =
    VERB_COUNT as u16 + RECIPE_COUNT as u16 + PERSONA_COUNT as u16 + FAMILY_COUNT as u16;

// Compile-time proof the layout is exactly the operator-locked 256-slot carve:
// null(1) + 226 atoms + reserved = 256, and the sub-ranges are contiguous.
const _: () = assert!(ATOM_COUNT == 226);
const _: () = assert!(RESERVED_BASE == 227);
const _: () = assert!(
    256 - RESERVED_BASE as u16 == 29,
    "29 reserved slots (227..=255)"
);

/// Which catalogue (and local index within it) a palette [`AtomId`] resolves to.
/// The `u8` payload is the **local** index inside that catalogue's registry
/// (Verb `0..=143`, Recipe `0..=33`, Persona `0..=35`, Family `0..=11`), NOT the
/// palette index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomCatalogue {
    /// The null default (palette index `0`).
    Null,
    /// A `holograph::dntree` `DnVerb`, local `0..=143`.
    Verb(u8),
    /// A `crate::recipes` recipe, local `0..=33`.
    Recipe(u8),
    /// A `crate::thinking` `ThinkingStyle`, local `0..=35`.
    Persona(u8),
    /// A `crate::style_family` `StyleFamily`, local `0..=11` (== the ordinal).
    Family(u8),
    /// An append-margin reserved slot (palette `227..=255`); carries the offset
    /// past [`RESERVED_BASE`] so a future promotion can be placed deterministically.
    Reserved(u8),
}

impl AtomId {
    /// Resolve this palette index to its catalogue + local index. Total function
    /// over all `256` byte values (reserved / null included) — never panics.
    #[must_use]
    pub const fn resolve(self) -> AtomCatalogue {
        let p = self.0;
        if p == 0 {
            AtomCatalogue::Null
        } else if p < RECIPE_BASE {
            AtomCatalogue::Verb(p - VERB_BASE)
        } else if p < PERSONA_BASE {
            AtomCatalogue::Recipe(p - RECIPE_BASE)
        } else if p < FAMILY_BASE {
            AtomCatalogue::Persona(p - PERSONA_BASE)
        } else if p < RESERVED_BASE {
            AtomCatalogue::Family(p - FAMILY_BASE)
        } else {
            AtomCatalogue::Reserved(p - RESERVED_BASE)
        }
    }

    /// The palette [`AtomId`] for Verb `local` (`0..=143`), or `None` if out of range.
    #[inline]
    #[must_use]
    pub const fn verb(local: u8) -> Option<AtomId> {
        if local < VERB_COUNT {
            Some(AtomId(VERB_BASE + local))
        } else {
            None
        }
    }

    /// The palette [`AtomId`] for Recipe `local` (`0..=33`), or `None` if out of range.
    #[inline]
    #[must_use]
    pub const fn recipe(local: u8) -> Option<AtomId> {
        if local < RECIPE_COUNT {
            Some(AtomId(RECIPE_BASE + local))
        } else {
            None
        }
    }

    /// The palette [`AtomId`] for Persona `local` (`0..=35`), or `None` if out of range.
    #[inline]
    #[must_use]
    pub const fn persona(local: u8) -> Option<AtomId> {
        if local < PERSONA_COUNT {
            Some(AtomId(PERSONA_BASE + local))
        } else {
            None
        }
    }

    /// The palette [`AtomId`] for Family `local` (`0..=11`, == the `StyleFamily`
    /// ordinal), or `None` if out of range.
    #[inline]
    #[must_use]
    pub const fn family(local: u8) -> Option<AtomId> {
        if local < FAMILY_COUNT {
            Some(AtomId(FAMILY_BASE + local))
        } else {
            None
        }
    }

    /// Whether this is the null default.
    #[inline]
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_are_contiguous_and_total_226() {
        // The four sub-ranges tile [1, 227) with no gap and no overlap.
        assert_eq!(VERB_BASE, 1);
        assert_eq!(RECIPE_BASE, 145);
        assert_eq!(PERSONA_BASE, 179);
        assert_eq!(FAMILY_BASE, 215);
        assert_eq!(RESERVED_BASE, 227);
        assert_eq!(ATOM_COUNT, 226);
        // 29 reserved slots fill 227..=255.
        assert_eq!(256 - RESERVED_BASE as u16, 29);
    }

    #[test]
    fn resolve_is_total_and_every_byte_lands_in_exactly_one_catalogue() {
        // Walk all 256 byte values; count each catalogue; assert the census.
        let (mut null, mut verb, mut recipe, mut persona, mut family, mut reserved) =
            (0, 0, 0, 0, 0, 0);
        for p in 0u16..=255 {
            match AtomId(p as u8).resolve() {
                AtomCatalogue::Null => null += 1,
                AtomCatalogue::Verb(l) => {
                    assert!(l < VERB_COUNT);
                    verb += 1;
                }
                AtomCatalogue::Recipe(l) => {
                    assert!(l < RECIPE_COUNT);
                    recipe += 1;
                }
                AtomCatalogue::Persona(l) => {
                    assert!(l < PERSONA_COUNT);
                    persona += 1;
                }
                AtomCatalogue::Family(l) => {
                    assert!(l < FAMILY_COUNT);
                    family += 1;
                }
                AtomCatalogue::Reserved(_) => reserved += 1,
            }
        }
        assert_eq!(null, 1);
        assert_eq!(verb, 144);
        assert_eq!(recipe, 34);
        assert_eq!(persona, 36);
        assert_eq!(family, 12);
        assert_eq!(reserved, 29);
        assert_eq!(null + verb + recipe + persona + family + reserved, 256);
    }

    #[test]
    fn constructors_round_trip_through_resolve() {
        for l in 0..VERB_COUNT {
            assert_eq!(AtomId::verb(l).unwrap().resolve(), AtomCatalogue::Verb(l));
        }
        for l in 0..RECIPE_COUNT {
            assert_eq!(
                AtomId::recipe(l).unwrap().resolve(),
                AtomCatalogue::Recipe(l)
            );
        }
        for l in 0..PERSONA_COUNT {
            assert_eq!(
                AtomId::persona(l).unwrap().resolve(),
                AtomCatalogue::Persona(l)
            );
        }
        for l in 0..FAMILY_COUNT {
            assert_eq!(
                AtomId::family(l).unwrap().resolve(),
                AtomCatalogue::Family(l)
            );
        }
    }

    #[test]
    fn constructors_reject_out_of_range() {
        assert_eq!(AtomId::verb(VERB_COUNT), None);
        assert_eq!(AtomId::recipe(RECIPE_COUNT), None);
        assert_eq!(AtomId::persona(PERSONA_COUNT), None);
        assert_eq!(AtomId::family(FAMILY_COUNT), None);
    }

    #[test]
    fn null_default_matches_the_zeroed_lane_convention() {
        assert!(AtomId::NULL.is_null());
        assert_eq!(AtomId::NULL.resolve(), AtomCatalogue::Null);
        // A zeroed FROZEN lane byte reads as the null atom, never a wrong policy.
        assert_eq!(AtomId(0).resolve(), AtomCatalogue::Null);
        assert!(!AtomId::family(0).unwrap().is_null());
    }

    #[test]
    fn family_local_is_the_style_family_ordinal() {
        // Family atom `f` resolves to Family(f) where f is exactly the
        // StyleFamily ordinal — so triangle slot `f` (indexed by family ordinal)
        // and the family atom's local index agree.
        assert_eq!(AtomId::family(0).unwrap(), AtomId(FAMILY_BASE));
        assert_eq!(AtomId::family(11).unwrap(), AtomId(226));
        assert_eq!(AtomId(226).resolve(), AtomCatalogue::Family(11));
    }
}
