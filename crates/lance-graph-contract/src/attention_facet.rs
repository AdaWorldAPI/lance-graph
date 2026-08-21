// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `attention_facet` — the **Attention** reading of the content-blind
//! `6×(8:8)` register, and the sparse container over it (`D-ACR-1`).
//!
//! This is a **reading, not a layout** — the discipline
//! [`crate::awareness_facet::SpoFacet`] and [`crate::tekamolo_facet`] already
//! follow. It re-labels the 12 payload bytes a
//! [`FacetCascade`](crate::facet::FacetCascade) already holds; it reserves,
//! moves and stores **nothing**. Zero new bytes enter the substrate.
//!
//! # The basis decision (`D-ACR-1`), and why it is not six vertical slots
//!
//! The attention atom is the SHIPPED atom: `FacetCascade` =
//! `facet_classid(4) | 6×(8:8) = 16 B`, read under
//! [`CascadeShape::G6D2`](crate::facet::CascadeShape::G6D2) — `6 groups ×
//! 2 levels`, `group_of` a pure shift. Nothing is minted.
//!
//! The tempting alternative was a **matrix**: six *vertical* rows of the same
//! 12-byte atom, one per rung/layer/timestep. It is rejected, and the reason is
//! the plan's own §3 (`alpha-channel-rung-overlay-v1.md`):
//!
//! > a second-order thought at node `g` is the **same** `NodeGuid` … a
//! > **different thinking-table row** — a different `(classid, rail)` pairing
//! > resolved by the ClassView … occupancy **sparse**.
//!
//! So the vertical dimension is **already addressed** — it is the
//! `facet_classid` selecting which thinking-table row a focus belongs to.
//! Stacking six fixed rows inside one atom would (a) mint a second addressing
//! system for a dimension the classid already carries, which HTT §2.2
//! explicitly withdrew (*"one fabric, several ClassView-resolved readings,
//! never alternative addressing systems"*), and (b) force an occupancy of
//! exactly six where §3 requires sparse. **One atom, one `(classid, rail)`;
//! the vertical stack is a set of atoms, not a field inside one.**
//!
//! # The six axes are CONTENT-BLIND — this type names none of them
//!
//! [`FacetCascade`]'s own contract is that *"the substrate is ALWAYS 8:8 …
//! only the CONSUMER projects meaning onto the bytes"*. This module keeps that
//! property: [`FocusAxis`] is `Axis0..Axis5`, a **position**, and
//! [`AttentionFocusFacet`] exposes only `coarse`/`fine` per axis. What an axis
//! MEANS is a **reading contract** resolved by `classid → ClassView`, never a
//! constant here.
//!
//! Two readings are sketched in the tests, over the SAME unchanged bytes, to
//! prove the projection is free:
//!
//! | reading | axis 0 … axis 5 |
//! |---|---|
//! | **cascade** (the key's own tiers) | HEEL · HIP · TWIG · LEAF · family · identity |
//! | **ontology scope** (a domain adapter's) | disease/phenotype · anatomy · process/mechanism · substance/intervention · evidence/provenance · context/time/population |
//!
//! Neither list is a type in this crate. The cascade reading is the canon's
//! own tier ladder; the ontology-scope reading is a *candidate* domain
//! adapter's projection (operator hypothesis, 2026-08-21) and belongs to that
//! adapter — a consumer stamps it via its ClassView. Baking either into
//! [`FocusAxis`] would make one domain's schema the substrate's, which is the
//! collision `E-ATTENTION-MASK-IS-A-RENAME-REGISTER-FILE-NOT-A-RESIDUE-CARRIER-1`
//! has this arc's fourth instance of.
//!
//! Each axis carries the always-8:8 pair — **level 0 = `hi` = coarse**,
//! **level 1 = `lo` = fine** (the shipped
//! [`FacetCascade::tier_bytes`](crate::facet::FacetCascade::tier_bytes) order,
//! `[t0.hi, t0.lo, t1.hi, …]`). The `u8`s bound **axis resolution** — 256
//! centroids per level — and say nothing about how many graph rows a focus
//! covers.
//!
//! # Composition is PREFIX CONTAINMENT, never a blind OR
//!
//! `FieldMask` / `WideFieldMask` / `StepMask` are bitsets over **field
//! positions**, and their `union` is a bit-`OR`. **This type does not do
//! that**, and must not be made to: OR-ing two addresses yields a third
//! address neither side ever visited. The composition here is the one the
//! canon already uses for ancestry —
//! [`NiblePath::is_ancestor_of`](crate::hhtl::NiblePath::is_ancestor_of) — a
//! coarse→fine **prefix** test:
//!
//! ```text
//! a covers b   ⟺   a.classid == b.classid
//!                  ∧ a.depth  <= b.depth
//!                  ∧ the first a.depth cascade units agree
//! ```
//!
//! A focus at depth 0 covers its whole class; at depth 12 it is one exact
//! address. **That is also how a focus over more than 256 rows is
//! representable**: one shallow facet is a wildcard over an unbounded subtree,
//! not 10⁶ bits. The `u8` per level caps the *branching a single level can
//! name*, never the population a facet covers.
//!
//! # Depth is explicit, never inferred from zero bytes
//!
//! [`AttentionFocusFacet`] carries `depth` **outside** the 12 bytes, exactly as
//! [`NiblePath`](crate::hhtl::NiblePath) carries its own. Inferring "the
//! wildcard starts where the bytes go zero" would collide with the
//! zero-fallback ladder, where `0` is a legitimate *dormant tier* rather than a
//! terminator — two meanings for one byte value, the ambiguity `NiblePath`
//! already refused. The wire shape stays exactly `6 × 2 × u8`; `depth` is a
//! **composition parameter**, not a stored byte.
//!
//! # What this module does NOT touch
//!
//! `cognitive-shader-driver`'s `attention_mask.rs` / `attention_mask_actor.rs`
//! are **not** used, and structurally cannot be: this is
//! `lance-graph-contract`, the zero-dep crate `cognitive-shader-driver` itself
//! depends on — the dependency edge runs only the other way, so the exclusion
//! is a compile-time property, not a convention. Per `D-ACR-0`
//! (`.claude/ATTENTION_MASK_AUDIT_2026_08_21.md`) those types are a finished
//! **rename register file** for a different contract (wide identity → scarce
//! narrow slot, LRU because slots are scarce), keyed by `MailboxId`, with no
//! address, no mask algebra and no trajectory. Do not build on them.

use crate::facet::{CascadeShape, FacetCascade, FacetTier, CASCADE_UNITS};

/// The number of axes an [`AttentionFocusFacet`] carries — `CASCADE_UNITS / 2`
/// under [`CascadeShape::G6D2`]. Derived, never a second literal.
pub const FOCUS_AXES: usize = CASCADE_UNITS / 2;

/// One of the six axes — a **position, not a meaning**.
///
/// `FacetCascade`'s contract is that the substrate is content-blind and only the
/// consumer projects meaning; this enum keeps that. It names slots `Axis0..Axis5`
/// on purpose. A cascade reading calls axis 0 "HEEL"; an ontology-scope adapter
/// may call it "disease/phenotype". Both are ClassView-resolved projections of
/// the same byte pair — see the module docs' two-reading table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum FocusAxis {
    /// Axis 0 — cascade units 0 (coarse) and 1 (fine).
    Axis0 = 0,
    /// Axis 1 — cascade units 2 and 3.
    Axis1 = 1,
    /// Axis 2 — cascade units 4 and 5.
    Axis2 = 2,
    /// Axis 3 — cascade units 6 and 7.
    Axis3 = 3,
    /// Axis 4 — cascade units 8 and 9.
    Axis4 = 4,
    /// Axis 5 — cascade units 10 and 11.
    Axis5 = 5,
}

impl FocusAxis {
    /// All six axes, ascending.
    pub const ALL: [FocusAxis; FOCUS_AXES] = [
        FocusAxis::Axis0,
        FocusAxis::Axis1,
        FocusAxis::Axis2,
        FocusAxis::Axis3,
        FocusAxis::Axis4,
        FocusAxis::Axis5,
    ];

    /// The `CascadeShape::G6D2` group index this axis addresses.
    #[inline]
    #[must_use]
    pub const fn group(self) -> u8 {
        self as u8
    }

    /// Axis for a group index, or `None` past the sixth — a loud refusal rather
    /// than a silent clamp (the `WideFieldMask` cap discipline).
    #[inline]
    #[must_use]
    pub const fn from_group(g: u8) -> Option<FocusAxis> {
        match g {
            0 => Some(FocusAxis::Axis0),
            1 => Some(FocusAxis::Axis1),
            2 => Some(FocusAxis::Axis2),
            3 => Some(FocusAxis::Axis3),
            4 => Some(FocusAxis::Axis4),
            5 => Some(FocusAxis::Axis5),
            _ => None,
        }
    }
}

/// **Where focus landed or was projected** — an [`FacetCascade`] read as
/// attention, plus the explicit prefix `depth` that makes it a wildcard.
///
/// The bytes are the shipped atom, untouched. `depth ∈ 0..=CASCADE_UNITS`
/// counts significant cascade units coarse→fine; it lives OUTSIDE the 12 bytes
/// (see the module docs' depth section).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionFocusFacet {
    /// The content-blind carrier — `facet_classid(4) | 6×(8:8)`.
    pub facet: FacetCascade,
    /// Significant cascade units, coarse→fine, `0..=CASCADE_UNITS`.
    /// `0` = the whole class; `CASCADE_UNITS` = one exact address.
    depth: u8,
}

impl AttentionFocusFacet {
    /// The carving this reading uses. `G6D2` — six axes, coarse/fine, and
    /// `group_of` is the shift `i >> 1`.
    pub const SHAPE: CascadeShape = CascadeShape::G6D2;

    /// An exact focus: all 12 units significant.
    #[inline]
    #[must_use]
    pub const fn exact(facet: FacetCascade) -> Self {
        Self {
            facet,
            depth: CASCADE_UNITS as u8,
        }
    }

    /// A **wildcard** focus: only the first `depth` cascade units are
    /// significant, the rest are "not consulted". Returns `None` if
    /// `depth > CASCADE_UNITS` — a loud refusal, never a silent clamp.
    #[inline]
    #[must_use]
    pub const fn prefix(facet: FacetCascade, depth: u8) -> Option<Self> {
        if depth as usize > CASCADE_UNITS {
            None
        } else {
            Some(Self { facet, depth })
        }
    }

    /// The whole class — depth 0, the broadest focus expressible.
    #[inline]
    #[must_use]
    pub const fn whole_class(facet_classid: u32) -> Self {
        Self {
            facet: FacetCascade {
                facet_classid,
                tiers: [FacetTier { lo: 0, hi: 0 }; FOCUS_AXES],
            },
            depth: 0,
        }
    }

    /// Significant cascade units, coarse→fine.
    #[inline]
    #[must_use]
    pub const fn depth(self) -> u8 {
        self.depth
    }

    /// The class this focus is scoped to. Focus never crosses classes —
    /// [`covers`](Self::covers) requires equality.
    #[inline]
    #[must_use]
    pub const fn classid(self) -> u32 {
        self.facet.facet_classid
    }

    /// The 12 payload bytes, coarse→fine per axis (`[a0.hi, a0.lo, a1.hi, …]`)
    /// — the shipped [`FacetCascade::tier_bytes`] order, unchanged.
    #[inline]
    #[must_use]
    pub const fn payload_bytes(self) -> [u8; CASCADE_UNITS] {
        self.facet.tier_bytes()
    }

    /// The coarse byte of `axis` (`hi`, level 0), or `None` if the axis lies
    /// past this focus's [`depth`](Self::depth) — an unconsulted axis has no
    /// value, and reporting `0` would conflate "not consulted" with "centroid
    /// zero".
    #[inline]
    #[must_use]
    pub fn coarse(self, axis: FocusAxis) -> Option<u8> {
        let unit = Self::SHAPE.index(axis.group(), 0);
        (unit < self.depth as usize).then(|| self.payload_bytes()[unit])
    }

    /// The fine byte of `axis` (`lo`, level 1), or `None` if past
    /// [`depth`](Self::depth).
    #[inline]
    #[must_use]
    pub fn fine(self, axis: FocusAxis) -> Option<u8> {
        let unit = Self::SHAPE.index(axis.group(), 1);
        (unit < self.depth as usize).then(|| self.payload_bytes()[unit])
    }

    /// Both bytes of `axis` as the shipped 8:8 tile, or `None` unless BOTH
    /// levels are within [`depth`](Self::depth).
    #[inline]
    #[must_use]
    pub fn axis(self, axis: FocusAxis) -> Option<FacetTier> {
        match (self.coarse(axis), self.fine(axis)) {
            (Some(hi), Some(lo)) => Some(FacetTier { lo, hi }),
            _ => None,
        }
    }

    /// **The composition primitive: prefix containment.** `self` covers `other`
    /// iff they share a class, `self` is no deeper, and their first
    /// `self.depth` cascade units agree coarse→fine.
    ///
    /// This is [`NiblePath::is_ancestor_of`](crate::hhtl::NiblePath::is_ancestor_of)'s
    /// rule on the 12-unit ladder — reflexive (a focus covers itself), unlike
    /// `NiblePath`'s empty-path case. It is **not** a bit test and has no
    /// `OR`-like counterpart; see [`RowFocusMask`].
    #[must_use]
    pub fn covers(self, other: Self) -> bool {
        if self.facet.facet_classid != other.facet.facet_classid || self.depth > other.depth {
            return false;
        }
        let (a, b) = (self.payload_bytes(), other.payload_bytes());
        a[..self.depth as usize] == b[..self.depth as usize]
    }

    /// The **meet**: the deepest focus that covers both — their shared
    /// coarse→fine prefix. `None` across classes (no common ancestor exists;
    /// a focus never crosses a class).
    ///
    /// This is what replaces a union: two focuses generalise to the subtree
    /// containing both, never to an invented address. It is lossy on purpose —
    /// [`RowFocusMask::union`] keeps both entries instead when exactness
    /// matters.
    #[must_use]
    pub fn common_prefix(self, other: Self) -> Option<Self> {
        if self.facet.facet_classid != other.facet.facet_classid {
            return None;
        }
        let (a, b) = (self.payload_bytes(), other.payload_bytes());
        let limit = self.depth.min(other.depth) as usize;
        let mut shared = 0usize;
        while shared < limit && a[shared] == b[shared] {
            shared += 1;
        }
        let mut facet = self.facet;
        // Units past the shared prefix are not consulted; zero them so the
        // carrier cannot be misread as an exact address by a reader that
        // ignores `depth`.
        let mut unit = shared;
        while unit < CASCADE_UNITS {
            let (g, l) = (Self::SHAPE.group_of(unit), Self::SHAPE.level_of(unit));
            let tier = &mut facet.tiers[g as usize];
            if l == 0 {
                tier.hi = 0;
            } else {
                tier.lo = 0;
            }
            unit += 1;
        }
        Some(Self {
            facet,
            depth: shared as u8,
        })
    }
}

/// A **sparse set of focuses** — the container half of `D-ACR-1`.
///
/// # Why this is not a bitmask
///
/// A row population is unbounded; `FieldMask` tops out at 64 positions and
/// `WideFieldMask` at 256, both by construction. Neither cardinality is
/// inherited here, because this container does not index rows at all — it
/// holds **wildcards over address subtrees**. One [`AttentionFocusFacet`] at a
/// shallow depth covers an arbitrarily large population; membership is
/// [`AttentionFocusFacet::covers`], not a bit lookup.
///
/// # The set operations, named
///
/// | op | meaning | NOT |
/// |---|---|---|
/// | [`union`](Self::union) | both focuses retained; entries covered by a shallower entry are absorbed | never a bit-`OR` of addresses |
/// | [`intersect`](Self::intersect) | the deeper of each covering pair | never `AND` of bytes |
/// | [`difference`](Self::difference) | entries not covered by the other side | never `ANDNOT` of bytes |
///
/// `union` is **absorbing, not accumulating**: inserting a focus already
/// covered by a member is a no-op, and inserting one that covers members
/// replaces them. That keeps the set minimal (an antichain) without ever
/// inventing an address.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RowFocusMask {
    entries: Vec<AttentionFocusFacet>,
}

impl RowFocusMask {
    /// The empty focus — covers nothing. Distinct from a depth-0 focus, which
    /// covers a whole class (the can-stay-silent / can-fire pair).
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Is this focus empty? An empty mask covers nothing at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of retained entries — the antichain size, NOT a row count. A
    /// single entry may cover an unbounded population.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The retained entries, in insertion order after absorption.
    #[must_use]
    pub fn entries(&self) -> &[AttentionFocusFacet] {
        &self.entries
    }

    /// Insert a focus, keeping the set a minimal antichain: a no-op if already
    /// covered; otherwise it replaces every entry it covers. Returns `true` iff
    /// the set changed.
    pub fn insert(&mut self, f: AttentionFocusFacet) -> bool {
        if self.entries.iter().any(|e| e.covers(f)) {
            return false;
        }
        self.entries.retain(|e| !f.covers(*e));
        self.entries.push(f);
        true
    }

    /// Does any entry cover `f`? — membership, by containment.
    #[must_use]
    pub fn contains(&self, f: AttentionFocusFacet) -> bool {
        self.entries.iter().any(|e| e.covers(f))
    }

    /// Every focus of either side, absorbed to a minimal antichain.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let mut out = self.clone();
        for e in &other.entries {
            out.insert(*e);
        }
        out
    }

    /// The overlap: for each covering pair, the **deeper** focus — the region
    /// both sides actually agree on. Disjoint pairs contribute nothing.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        let mut out = Self::empty();
        for a in &self.entries {
            for b in &other.entries {
                if a.covers(*b) {
                    out.insert(*b);
                } else if b.covers(*a) {
                    out.insert(*a);
                }
            }
        }
        out
    }

    /// Entries of `self` not covered by any entry of `other`.
    ///
    /// **Deliberately conservative:** an entry only PARTIALLY overlapped is
    /// kept whole, because subtracting a subtree from a prefix would require
    /// enumerating siblings — inventing addresses the focus never visited.
    /// Splitting is deferred (see the module's deferred note).
    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        Self {
            entries: self
                .entries
                .iter()
                .filter(|a| !other.entries.iter().any(|b| b.covers(**a)))
                .copied()
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLASS: u32 = 0x0301_0000;

    fn facet_of(bytes: [u8; CASCADE_UNITS], classid: u32) -> FacetCascade {
        // `tier_bytes()` is [t0.hi, t0.lo, t1.hi, t1.lo, …] — coarse first.
        let mut tiers = [FacetTier { lo: 0, hi: 0 }; FOCUS_AXES];
        for (g, tier) in tiers.iter_mut().enumerate() {
            tier.hi = bytes[g * 2];
            tier.lo = bytes[g * 2 + 1];
        }
        FacetCascade {
            facet_classid: classid,
            tiers,
        }
    }

    fn exact(bytes: [u8; CASCADE_UNITS]) -> AttentionFocusFacet {
        AttentionFocusFacet::exact(facet_of(bytes, CLASS))
    }

    // ── the atom: exact 12-byte round-trip ────────────────────────────────

    #[test]
    fn twelve_byte_payload_round_trips_exactly() {
        let bytes: [u8; CASCADE_UNITS] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let f = exact(bytes);
        assert_eq!(
            f.payload_bytes(),
            bytes,
            "the 12 payload bytes are the atom"
        );
        // And the carrier is the shipped 16-byte facet, unchanged.
        assert_eq!(
            FacetCascade::from_bytes(&f.facet.to_bytes()),
            f.facet,
            "the FacetCascade carrier round-trips through its own 16 bytes"
        );
        assert_eq!(core::mem::size_of::<FacetCascade>(), 16);
    }

    #[test]
    fn per_axis_coarse_and_fine_round_trip() {
        let bytes: [u8; CASCADE_UNITS] = [10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61];
        let f = exact(bytes);
        for (g, axis) in FocusAxis::ALL.into_iter().enumerate() {
            assert_eq!(f.coarse(axis), Some(bytes[g * 2]), "coarse = hi, level 0");
            assert_eq!(f.fine(axis), Some(bytes[g * 2 + 1]), "fine = lo, level 1");
            let tile = f.axis(axis).expect("exact focus has every axis");
            assert_eq!((tile.hi, tile.lo), (bytes[g * 2], bytes[g * 2 + 1]));
        }
    }

    #[test]
    fn an_axis_past_the_prefix_depth_reads_none_not_zero() {
        // depth 4 ⇒ axes 0 and 1 significant, axes 2..6 not consulted.
        let f = AttentionFocusFacet::prefix(facet_of([7; CASCADE_UNITS], CLASS), 4).unwrap();
        assert_eq!(f.coarse(FocusAxis::Axis1), Some(7));
        assert_eq!(
            f.coarse(FocusAxis::Axis2),
            None,
            "unconsulted axis must not report 0 — that is centroid zero's value"
        );
        assert_eq!(f.axis(FocusAxis::Axis2), None);
    }

    #[test]
    fn prefix_refuses_out_of_range_depth_loudly() {
        let c = facet_of([0; CASCADE_UNITS], CLASS);
        assert!(AttentionFocusFacet::prefix(c, CASCADE_UNITS as u8).is_some());
        assert!(
            AttentionFocusFacet::prefix(c, CASCADE_UNITS as u8 + 1).is_none(),
            "past the ladder is a refusal, never a silent clamp"
        );
    }

    // ── composition: containment, and demonstrably NOT a blind OR ─────────

    #[test]
    fn covers_is_prefix_containment_and_is_reflexive() {
        let deep = exact([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let shallow =
            AttentionFocusFacet::prefix(facet_of([1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0], CLASS), 4)
                .unwrap();
        assert!(shallow.covers(deep), "a prefix covers what extends it");
        assert!(!deep.covers(shallow), "and containment is asymmetric");
        assert!(deep.covers(deep), "reflexive");
    }

    #[test]
    fn a_focus_never_covers_across_classes() {
        let a = AttentionFocusFacet::whole_class(CLASS);
        let b = AttentionFocusFacet::exact(facet_of([1; CASCADE_UNITS], 0x0302_0000));
        assert!(
            !a.covers(b),
            "depth 0 covers its OWN class only — focus never crosses a class"
        );
    }

    /// The load-bearing negative: bitwise-OR of two focuses would produce an
    /// address neither visited, and the container must not behave that way.
    #[test]
    fn composition_is_not_a_blind_or() {
        let a = exact([0b0000_0001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let b = exact([0b0000_0010, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        // What a bit-OR WOULD produce — a third, never-visited address.
        let or_byte = 0b0000_0001u8 | 0b0000_0010u8;
        let phantom = exact([or_byte, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_ne!(or_byte, 0b0000_0001);
        assert_ne!(or_byte, 0b0000_0010);

        let u = RowFocusMask::empty().union(&{
            let mut m = RowFocusMask::empty();
            m.insert(a);
            m.insert(b);
            m
        });

        assert!(
            u.contains(a) && u.contains(b),
            "both originals are retained"
        );
        assert!(
            !u.contains(phantom),
            "the OR-address was never visited and must not become a member"
        );
        assert_eq!(u.len(), 2, "union keeps both, it does not fuse them");
    }

    #[test]
    fn common_prefix_generalises_and_never_invents_an_address() {
        let a = exact([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let b = exact([1, 2, 3, 99, 0, 0, 0, 0, 0, 0, 0, 0]);
        let m = a.common_prefix(b).expect("same class");
        assert_eq!(m.depth(), 3, "they agree on exactly three cascade units");
        assert!(m.covers(a) && m.covers(b), "the meet covers both");
        assert_eq!(
            &m.payload_bytes()[3..],
            &[0u8; CASCADE_UNITS - 3],
            "units past the shared prefix are zeroed, not left as stale bytes"
        );
        assert!(
            exact([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
                .common_prefix(AttentionFocusFacet::exact(facet_of(
                    [1; CASCADE_UNITS],
                    0x0999_0000
                )))
                .is_none(),
            "no meet across classes"
        );
    }

    // ── the container: cardinality is NOT inherited ───────────────────────

    #[test]
    fn container_holds_far_more_than_the_fieldmask_and_widefieldmask_caps() {
        // FieldMask::MAX_FIELDS == 64 and WideFieldMask's u8 universe caps at
        // 256. Neither bound applies here: this is not a position bitset.
        let mut m = RowFocusMask::empty();
        for i in 0..1000u32 {
            let b = i.to_le_bytes();
            // Distinct in the FINE units so no entry absorbs another.
            m.insert(exact([0, 0, 0, 0, b[0], b[1], b[2], b[3], 0, 0, 0, 0]));
        }
        assert_eq!(m.len(), 1000, "1000 > 256 > 64 — no cap inherited");
    }

    #[test]
    fn one_shallow_focus_covers_an_unbounded_population() {
        // The >256-rows requirement: coverage comes from the PREFIX, not from
        // one bit per row. A depth-2 focus covers every address sharing axis 0.
        let wildcard =
            AttentionFocusFacet::prefix(facet_of([9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], CLASS), 2)
                .unwrap();
        let mut covered = 0usize;
        for i in 0..=u16::MAX {
            let b = i.to_le_bytes();
            let addr = exact([9, 9, b[0], b[1], 0, 0, 0, 0, 0, 0, 0, 0]);
            if wildcard.covers(addr) {
                covered += 1;
            }
        }
        assert_eq!(
            covered, 65_536,
            "one 12-byte atom covers 2^16 distinct addresses — and that is only \
             the two units varied here, not the ceiling"
        );
    }

    #[test]
    fn insert_absorbs_into_a_minimal_antichain() {
        let mut m = RowFocusMask::empty();
        let deep = exact([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert!(m.insert(deep));
        // A shallower focus covering it replaces it.
        let shallow =
            AttentionFocusFacet::prefix(facet_of([1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], CLASS), 2)
                .unwrap();
        assert!(m.insert(shallow));
        assert_eq!(m.len(), 1, "the covered entry was absorbed");
        assert!(m.contains(deep), "and is still a member by containment");
        assert!(!m.insert(deep), "re-inserting a covered focus is a no-op");
    }

    #[test]
    fn empty_covers_nothing_but_depth_zero_covers_its_class() {
        // The can-fire / can-stay-silent pair on NON-trivial input.
        let some = exact([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert!(
            !RowFocusMask::empty().contains(some),
            "an empty focus must stay silent"
        );
        let mut all = RowFocusMask::empty();
        all.insert(AttentionFocusFacet::whole_class(CLASS));
        assert!(all.contains(some), "and a depth-0 focus must fire");
        // Distinguishable from each other AND from a real, narrow focus.
        let mut narrow = RowFocusMask::empty();
        narrow.insert(some);
        assert!(!narrow.contains(exact([9; CASCADE_UNITS])));
        assert_ne!(narrow, all);
        assert_ne!(narrow, RowFocusMask::empty());
    }

    // ── two readings, ONE unchanged 12-byte atom ─────────────────────────

    /// The operator's 2026-08-21 hypothesis: the six axes may also carry six
    /// **ontology scopes** in interplay. This test is the whole argument that
    /// the hypothesis costs nothing — it is a *projection*, so both readings
    /// address the identical bytes and neither is a type in this crate.
    ///
    /// A domain adapter would resolve these labels through its ClassView; they
    /// appear here as local `const`s precisely to prove they need not exist in
    /// the substrate.
    #[test]
    fn the_same_atom_reads_as_cascade_and_as_ontology_scope_without_changing_a_byte() {
        // One adapter's projection (a candidate MedCare-shaped scope reading).
        const SCOPE: [&str; FOCUS_AXES] = [
            "disease/phenotype",
            "anatomy",
            "process/mechanism",
            "substance/intervention",
            "evidence/provenance",
            "context/time/population",
        ];
        // The canon's own reading of the very same positions.
        const CASCADE: [&str; FOCUS_AXES] = ["HEEL", "HIP", "TWIG", "LEAF", "family", "identity"];

        let bytes: [u8; CASCADE_UNITS] = [3, 30, 4, 40, 5, 50, 6, 60, 7, 70, 8, 80];
        let f = exact(bytes);
        let before = f.facet.to_bytes();

        // Read it as ontology scope: (scope centroid, local refinement).
        let scoped: Vec<(&str, u8, u8)> = FocusAxis::ALL
            .into_iter()
            .map(|a| {
                (
                    SCOPE[a.group() as usize],
                    f.coarse(a).unwrap(),
                    f.fine(a).unwrap(),
                )
            })
            .collect();
        assert_eq!(scoped[1], ("anatomy", 4, 40));
        assert_eq!(scoped[4], ("evidence/provenance", 7, 70));

        // Read the SAME facet as the cascade tiers.
        let cascaded: Vec<(&str, u8, u8)> = FocusAxis::ALL
            .into_iter()
            .map(|a| {
                (
                    CASCADE[a.group() as usize],
                    f.coarse(a).unwrap(),
                    f.fine(a).unwrap(),
                )
            })
            .collect();
        assert_eq!(cascaded[1], ("HIP", 4, 40));

        // The projections differ only in the LABEL; every byte is identical.
        for (s, c) in scoped.iter().zip(cascaded.iter()) {
            assert_ne!(s.0, c.0, "the two readings name the axis differently");
            assert_eq!((s.1, s.2), (c.1, c.2), "and read the identical bytes");
        }
        assert_eq!(
            f.facet.to_bytes(),
            before,
            "reading a facet never mutates it — a projection is free"
        );

        // Containment is label-blind too: a scope-shaped wildcard ("any
        // anatomy under disease centroid 3") is an ordinary depth-2 prefix.
        let any_under_disease_3 =
            AttentionFocusFacet::prefix(facet_of([3, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], CLASS), 2)
                .unwrap();
        assert!(any_under_disease_3.covers(f));
    }

    #[test]
    fn intersect_and_difference_are_containment_shaped() {
        let deep = exact([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let shallow =
            AttentionFocusFacet::prefix(facet_of([1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], CLASS), 2)
                .unwrap();
        let other = exact([9; CASCADE_UNITS]);

        let (mut a, mut b) = (RowFocusMask::empty(), RowFocusMask::empty());
        a.insert(deep);
        a.insert(other);
        b.insert(shallow);

        let i = a.intersect(&b);
        assert_eq!(i.len(), 1, "only the covering pair contributes");
        assert!(i.contains(deep), "and it yields the DEEPER side");
        assert!(!i.contains(other), "the disjoint entry contributes nothing");

        let d = a.difference(&b);
        assert!(!d.contains(deep), "covered entry removed");
        assert!(d.contains(other), "uncovered entry kept");
    }
}
