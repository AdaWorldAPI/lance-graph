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
//! | Tsv     |  29 | `crate::atoms` `CANONICAL_ATOMS` minus the 4 overlaps ([`TSV_OVERLAPS`]) |
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
//! 227  ..= 255 Tsv        ( 29; the locked-TSV lanes with no prior home)
//! ```
//!
//! # The append margin is SPENT (operator ruling, 2026-09-01)
//!
//! `227..=255` was the append margin. It now holds the **Tsv** block, and the
//! palette is **full**: every one of the 256 byte values resolves to a named
//! catalogue entry, and there is no room left for a future promotion.
//!
//! This was ruled deliberately, with the cost stated first. The reason the fit
//! is exact — 29 homeless atoms into 29 free slots — is arithmetic coincidence,
//! not design: `29 = 256 − (1 + 144 + 34 + 36 + 12)` on one side and
//! `29 = 33 − 4 overlaps` on the other, and the second number moves if the
//! overlap adjudication moves (28 if `imagine_counterfactual` is read as the
//! ICR recipe; 33 on exact string match alone). [`TSV_OVERLAPS`] pins the
//! adjudication so the count cannot drift silently.
//!
//! **Growth from here is `ogar-loco`, not this table.** A palette slot is a
//! call of arity zero — `12×u8` is the `6×(u8:u8)` `function:value` carving
//! with every operand discarded. The Tsv block inherits that limit: `rung_r1`
//! … `rung_r9` are nine slots for what is one op with a depth operand, and the
//! Σ chain and the Meta knobs are magnitudes stored as needles. A vocabulary
//! in the `ogar-loco` domain range (`0x90..=0xFF`, classid-selected, 112 slots
//! **per vocabulary** rather than 29 shared globally) expresses those as
//! `Call { function, values }` without spending anyone's margin. Treat this
//! block as the closing entry in a legacy table, not as the pattern to repeat.

/// A palette256 index — one byte, `0..=255`. Value `0` is the null default
/// (`AtomId::NULL`); `1..=226` address the original four catalogues; `227..=255`
/// address the [`TSV_ATOMS`] block. The palette is **full** — there is no
/// remaining append margin. Stored bare (`u8`) in a `12×u8` FROZEN lane slot.
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
/// Count of Tsv atoms — the locked 33-lane TSV basis minus the 4 entries that
/// already had a palette home ([`TSV_OVERLAPS`]). Fills the former append margin.
pub const TSV_COUNT: u8 = 29;

/// First palette index of the Verb range (`1`; index `0` is `NULL`).
pub const VERB_BASE: u8 = 1;
/// First palette index of the Recipe range.
pub const RECIPE_BASE: u8 = VERB_BASE + VERB_COUNT; // 145
/// First palette index of the Persona range.
pub const PERSONA_BASE: u8 = RECIPE_BASE + RECIPE_COUNT; // 179
/// First palette index of the Family range.
pub const FAMILY_BASE: u8 = PERSONA_BASE + PERSONA_COUNT; // 215
/// First palette index of the Tsv range (the former append margin, now spent).
pub const TSV_BASE: u8 = FAMILY_BASE + FAMILY_COUNT; // 227

/// The former append-margin base. Retained as the historical name for
/// [`TSV_BASE`]: the range it named is allocated, so this is where the margin
/// *was*, not free real estate.
pub const RESERVED_BASE: u8 = TSV_BASE;

/// Total addressed atoms (`255` — the four legacy catalogues plus the Tsv block;
/// with `NULL` that is all 256 byte values).
pub const ATOM_COUNT: u16 = VERB_COUNT as u16
    + RECIPE_COUNT as u16
    + PERSONA_COUNT as u16
    + FAMILY_COUNT as u16
    + TSV_COUNT as u16;

// Compile-time proof the layout tiles all 256 slots: null(1) + 255 atoms, with
// contiguous sub-ranges and NO remaining margin.
const _: () = assert!(ATOM_COUNT == 255);
const _: () = assert!(TSV_BASE == 227);
const _: () = assert!(
    TSV_BASE as u16 + TSV_COUNT as u16 == 256,
    "the Tsv block closes the palette at 256 — no append margin remains"
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
    /// A locked-TSV atom, local `0..=28` — an index into [`TSV_ATOMS`], NOT a
    /// `crate::atoms` canonical dim (use [`TsvAtom::dim`] for that). Occupies the
    /// former append margin (`227..=255`).
    Tsv(u8),
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
        } else if p < TSV_BASE {
            AtomCatalogue::Family(p - FAMILY_BASE)
        } else {
            AtomCatalogue::Tsv(p - TSV_BASE)
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

    /// The palette [`AtomId`] for Tsv `local` (`0..=28`, an index into
    /// [`TSV_ATOMS`]), or `None` if out of range.
    #[inline]
    #[must_use]
    pub const fn tsv(local: u8) -> Option<AtomId> {
        if local < TSV_COUNT {
            Some(AtomId(TSV_BASE + local))
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

// ── The Tsv block (227..=255) — the locked TSV lanes with no prior home ──

/// One Tsv-block entry: its palette-local index, the `crate::atoms` canonical
/// lane it stands for, and that lane's locked name.
///
/// `dim` is the join back to `crate::atoms::CANONICAL_ATOMS` — the local index
/// is a palette address and deliberately does NOT equal the canonical dim,
/// because the 4 overlapping lanes ([`TSV_OVERLAPS`]) are skipped here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TsvAtom {
    /// Local index inside the Tsv block, `0..=28` (palette = [`TSV_BASE`] + this).
    pub local: u8,
    /// Canonical lane index in `crate::atoms::CANONICAL_ATOMS`, `0..=32`.
    pub dim: u8,
    /// The locked lane name, verbatim from the atom catalogue.
    pub name: &'static str,
}

/// The 29 locked-TSV lanes that had no palette home, in canonical `dim` order.
///
/// This is the complement of [`TSV_OVERLAPS`] within the 33-lane basis:
/// `TSV_ATOMS ⊎ TSV_OVERLAPS` covers `dim 0..33` exactly once each (pinned by
/// [`tests::every_canonical_lane_is_either_minted_or_annotated_exactly_once`]).
///
/// # These are needles standing in for axes
///
/// Nine `rung_r*` entries address what is one operation with a depth operand;
/// the five `sigma_*` entries are a chain; `confidence_threshold`,
/// `preflight_depth`, `exploration` and `verbosity` are continuous knobs stored
/// as a categorical pick. That is the arity-zero limit of a palette slot, and it
/// is the reason the module doc points growth at an `ogar-loco` vocabulary
/// (`function : value`) rather than at more entries here.
pub const TSV_ATOMS: [TsvAtom; TSV_COUNT as usize] = [
    TsvAtom {
        local: 0,
        dim: 0,
        name: "see_association",
    },
    TsvAtom {
        local: 1,
        dim: 1,
        name: "do_intervention",
    },
    TsvAtom {
        local: 2,
        dim: 2,
        name: "imagine_counterfactual",
    },
    TsvAtom {
        local: 3,
        dim: 3,
        name: "rung_r1",
    },
    TsvAtom {
        local: 4,
        dim: 4,
        name: "rung_r2",
    },
    TsvAtom {
        local: 5,
        dim: 5,
        name: "rung_r3",
    },
    TsvAtom {
        local: 6,
        dim: 6,
        name: "rung_r4",
    },
    TsvAtom {
        local: 7,
        dim: 7,
        name: "rung_r5",
    },
    TsvAtom {
        local: 8,
        dim: 8,
        name: "rung_r6",
    },
    TsvAtom {
        local: 9,
        dim: 9,
        name: "rung_r7",
    },
    TsvAtom {
        local: 10,
        dim: 10,
        name: "rung_r8",
    },
    TsvAtom {
        local: 11,
        dim: 11,
        name: "rung_r9",
    },
    TsvAtom {
        local: 12,
        dim: 12,
        name: "sigma_omega",
    },
    TsvAtom {
        local: 13,
        dim: 13,
        name: "sigma_delta",
    },
    TsvAtom {
        local: 14,
        dim: 14,
        name: "sigma_phi",
    },
    TsvAtom {
        local: 15,
        dim: 15,
        name: "sigma_theta",
    },
    TsvAtom {
        local: 16,
        dim: 16,
        name: "sigma_lambda",
    },
    TsvAtom {
        local: 17,
        dim: 20,
        name: "synthesize",
    },
    TsvAtom {
        local: 18,
        dim: 21,
        name: "preflight",
    },
    TsvAtom {
        local: 19,
        dim: 22,
        name: "escalate",
    },
    TsvAtom {
        local: 20,
        dim: 24,
        name: "model_other",
    },
    TsvAtom {
        local: 21,
        dim: 25,
        name: "authentic",
    },
    TsvAtom {
        local: 22,
        dim: 26,
        name: "performance",
    },
    TsvAtom {
        local: 23,
        dim: 27,
        name: "protective",
    },
    TsvAtom {
        local: 24,
        dim: 28,
        name: "absent",
    },
    TsvAtom {
        local: 25,
        dim: 29,
        name: "confidence_threshold",
    },
    TsvAtom {
        local: 26,
        dim: 30,
        name: "preflight_depth",
    },
    TsvAtom {
        local: 27,
        dim: 31,
        name: "exploration",
    },
    TsvAtom {
        local: 28,
        dim: 32,
        name: "verbosity",
    },
];

/// One annotated overlap: a locked-TSV lane that was NOT given a Tsv slot
/// because an existing palette entry already names the same thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TsvOverlap {
    /// Canonical lane index in `crate::atoms::CANONICAL_ATOMS`, `0..=32`.
    pub dim: u8,
    /// The locked lane name.
    pub name: &'static str,
    /// The pre-existing palette index this lane resolves to instead.
    pub palette: AtomId,
    /// The pre-existing entry's own name, as spelled in ITS registry.
    pub existing: &'static str,
}

/// The 4 locked-TSV lanes that already had a palette home — annotated, not minted.
///
/// Each is the same concept under a different inflection, so giving it a second
/// palette index would make two bytes mean one thing. `dim` is the join back to
/// `crate::atoms::CANONICAL_ATOMS`; `palette` is where callers should address it.
///
/// # This adjudication is a JUDGEMENT and it moves the block size
///
/// Three of the four are the same word in a different grammatical person
/// (`abduct`/`abduces`) — strong. `transcend`/`transcendent` is a verb/adjective
/// pair — the weak one; reading it as distinct makes the block 30 and the palette
/// overflows 256. A fifth candidate was REJECTED: `imagine_counterfactual` shares
/// only the word "counterfactual" with the ICR recipe (`Iterative Counterfactual
/// Reasoning`), and a Pearl ladder rung is not a named tactic — reading it as a
/// match would make the block 28 and leave one slot free. The block size is
/// therefore contingent on exactly these four calls; the table exists so that
/// contingency is visible instead of buried in a count.
pub const TSV_OVERLAPS: [TsvOverlap; 4] = [
    // Verb local 84 (`ABDUCES`) → palette 1 + 84 = 85.
    TsvOverlap {
        dim: 17,
        name: "abduct",
        palette: AtomId(VERB_BASE + 84),
        existing: "ABDUCES",
    },
    // Verb local 82 (`DEDUCES`) → palette 1 + 82 = 83.
    TsvOverlap {
        dim: 18,
        name: "deduce",
        palette: AtomId(VERB_BASE + 82),
        existing: "DEDUCES",
    },
    // Verb local 83 (`INDUCES`) → palette 1 + 83 = 84.
    TsvOverlap {
        dim: 19,
        name: "induce",
        palette: AtomId(VERB_BASE + 83),
        existing: "INDUCES",
    },
    // Persona local 34 (`ThinkingStyle::Transcendent`) → palette 179 + 34 = 213.
    TsvOverlap {
        dim: 23,
        name: "transcend",
        palette: AtomId(PERSONA_BASE + 34),
        existing: "ThinkingStyle::Transcendent",
    },
];

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
        assert_eq!(TSV_BASE, 227);
        assert_eq!(ATOM_COUNT, 255);
        // The Tsv block closes the palette: 227 + 29 == 256, no margin left.
        assert_eq!(TSV_BASE as u16 + TSV_COUNT as u16, 256);
        assert_eq!(RESERVED_BASE, TSV_BASE);
    }

    /// The partition law: every one of the 33 locked lanes is EITHER minted in
    /// [`TSV_ATOMS`] XOR annotated in [`TSV_OVERLAPS`] — never both, never
    /// neither. This is what makes the block size 29 a consequence rather than
    /// a constant: add an overlap without removing its `TsvAtom` and this fails.
    #[test]
    fn every_canonical_lane_is_either_minted_or_annotated_exactly_once() {
        let mut seen = [0u8; 33];
        for a in TSV_ATOMS {
            seen[a.dim as usize] += 1;
        }
        for o in TSV_OVERLAPS {
            seen[o.dim as usize] += 1;
        }
        for (dim, n) in seen.iter().enumerate() {
            assert_eq!(*n, 1, "lane {dim} covered {n}x, expected exactly once");
        }
        assert_eq!(TSV_ATOMS.len() + TSV_OVERLAPS.len(), 33);
        assert_eq!(TSV_ATOMS.len(), TSV_COUNT as usize);
    }

    /// `local` is a palette address, not a canonical dim — and for at least one
    /// entry the two genuinely differ. Without this the table could renumber
    /// `local` to equal `dim` and nothing would notice.
    #[test]
    fn tsv_locals_are_dense_and_diverge_from_canonical_dims() {
        for (i, a) in TSV_ATOMS.iter().enumerate() {
            assert_eq!(a.local as usize, i, "local must be dense 0..29");
            assert_eq!(AtomId::tsv(a.local), Some(AtomId(TSV_BASE + a.local)));
        }
        let diverging = TSV_ATOMS.iter().filter(|a| a.local != a.dim).count();
        assert!(
            diverging >= 12,
            "the 4 skipped overlaps must shift at least the 12 lanes above dim 17; got {diverging}"
        );
    }

    /// Each annotated overlap points at a REAL pre-existing entry — one that
    /// resolves into the catalogue the annotation claims, and never into the
    /// Tsv block itself (which would be a self-reference, not an overlap).
    #[test]
    fn annotated_overlaps_resolve_into_their_pre_existing_catalogue() {
        for o in TSV_OVERLAPS {
            assert!(
                o.palette.0 < TSV_BASE,
                "{} must point below the Tsv block, got {}",
                o.name,
                o.palette.0
            );
            match o.palette.resolve() {
                AtomCatalogue::Verb(l) => assert!(l < VERB_COUNT),
                AtomCatalogue::Persona(l) => assert!(l < PERSONA_COUNT),
                other => panic!("{} resolved to {other:?}, expected Verb or Persona", o.name),
            }
        }
        // The specific bytes, pinned: renumbering a verb must break this.
        assert_eq!(TSV_OVERLAPS[0].palette, AtomId(85)); // abduct  -> ABDUCES
        assert_eq!(TSV_OVERLAPS[1].palette, AtomId(83)); // deduce  -> DEDUCES
        assert_eq!(TSV_OVERLAPS[2].palette, AtomId(84)); // induce  -> INDUCES
        assert_eq!(TSV_OVERLAPS[3].palette, AtomId(213)); // transcend -> Transcendent
    }

    /// No byte is a Reserved slot any more — the whole palette is allocated.
    #[test]
    fn no_byte_value_is_unallocated() {
        for p in 0u16..=255 {
            let r = AtomId(p as u8).resolve();
            if let AtomCatalogue::Tsv(l) = r {
                assert!(l < TSV_COUNT, "Tsv local {l} out of range");
            }
        }
        assert!(AtomId::tsv(TSV_COUNT).is_none(), "29 is out of range");
        assert_eq!(AtomId::tsv(28), Some(AtomId(255)));
    }

    #[test]
    fn resolve_is_total_and_every_byte_lands_in_exactly_one_catalogue() {
        // Walk all 256 byte values; count each catalogue; assert the census.
        let (mut null, mut verb, mut recipe, mut persona, mut family, mut tsv) = (0, 0, 0, 0, 0, 0);
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
                AtomCatalogue::Tsv(l) => {
                    assert!(l < TSV_COUNT);
                    tsv += 1;
                }
            }
        }
        assert_eq!(null, 1);
        assert_eq!(verb, 144);
        assert_eq!(recipe, 34);
        assert_eq!(persona, 36);
        assert_eq!(family, 12);
        assert_eq!(tsv, 29);
        assert_eq!(null + verb + recipe + persona + family + tsv, 256);
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
