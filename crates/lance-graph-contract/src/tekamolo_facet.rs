//! `tekamolo_facet` — a TEKAMOLO **reading** of a [`FacetCascade`], for testing
//! the cognitive readers with TYPED relations (operator, 2026-07-23: "add a
//! tekamolo tenant … temporal 256:256:256 / causal / modal / local, and the
//! qualia if not already").
//!
//! # What it is (and is NOT)
//!
//! This is a **ClassView naming** over the existing 12-byte content-blind facet
//! payload — the [`CascadeShape::G4D3`] carving (`4 groups × 3 levels`) with its
//! four groups NAMED as the four adverbial roles of TEKAMOLO:
//!
//! | group | role  | German | question | lane |
//! |-------|-------|--------|----------|------|
//! | 0     | **Te**mporal | *temporal* | when  | `256:256:256` |
//! | 1     | **Ka**usal   | *kausal*   | why   | `256:256:256` |
//! | 2     | **Mo**dal    | *modal*    | how   | `256:256:256` |
//! | 3     | **Lo**kal    | *lokal*    | where | `256:256:256` |
//!
//! Each lane is a **3-byte cascade** — three tiers of a 256-entry codebook
//! (`256:256:256`), coarse→fine, exactly the `cascade_byte(G4D3, group, level)`
//! bytes the facet already exposes. So this type carries **no bytes of its own**:
//! it is a `#[repr(transparent)]`-shaped newtype whose accessors rename the
//! canonical carving. **No `ENVELOPE_LAYOUT_VERSION` bump, no new tenant lane** —
//! the value slab is unchanged; a `classid → ClassView` chooses to *read* a facet
//! as TEKAMOLO (per `le-contract.md` §3, "the ClassView picks the carving").
//!
//! # Status: EXPERIMENTAL reading — not yet in the operator-locked §3 catalogue
//!
//! `le-contract.md` §3 sanctions the `G4D3` (`4×3`) carving as **L5 = SPO
//! triplets**. The TEKAMOLO four-adverbial-role naming is a *different* semantic
//! reading of the SAME byte carving — legitimate under the slot-purity /
//! "ClassView picks the carving" doctrine (no bytes move), operator-green-lit
//! 2026-07-23 for testing typed relations, but **not yet registered as a
//! sanctioned §3 reading**. Two consequences a consumer must respect
//! (`v3-envelope-auditor` verdict, `E-TEKAMOLO-FACET-IS-A-G4D3-READING-1`):
//! - each lane's three bytes STRADDLE tier boundaries (group 0 = `t0.hi, t0.lo,
//!   t1.hi` — the G4D3 "divide" shape, `is_byte_aligned() == false`); the
//!   `256:256:256` framing is a naming over ladder positions, not a clean
//!   per-tier hierarchy (inherent to `G4D3`, same as the L5 triplets);
//! - a consumer that begins *trusting* a TEKAMOLO reading owes the §3b jc-pillar
//!   (ICC / Spearman / Cronbach) certification before backing any downstream
//!   claim on it. Until registered in §3, treat this as experimental.
//!
//! # Qualia — already present, not added here
//!
//! The signed-nibble qualia lane already exists as **value tenant #1**
//! (`QualiaI4_16D`, 16×i4; `tenants.md` §2). TEKAMOLO is the *when/why/how/where*
//! ADDRESS of a node; qualia is its *felt tone*. They are orthogonal tenants —
//! this module does not touch qualia.
//!
//! # Why it matters (the D-SCI-1 frontier)
//!
//! The `insight_read` House-text falsifier showed the readers need TYPED, sparse
//! relations (not dense word-adjacency) to find a relational centre. TEKAMOLO is
//! that type system: a `temporal` edge (*"during that winter"*), a `causal` edge
//! (*"whenever the key moved, the clock recovered"*), a `modal` edge, a `local`
//! edge. Typing reasoning edges by adverbial role gives the ablation centre-finder
//! a graph with real articulation structure. This facet is the address side of
//! that; the edge-typing side rides `Copula::Rel(verb)` in the reasoning layer.

use crate::facet::{CascadeShape, FacetCascade, FacetTier};

/// The four TEKAMOLO adverbial roles, in canonical **Te-Ka-Mo-Lo** order — the
/// group index into the [`CascadeShape::G4D3`] carving.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TekamoloRole {
    /// when — group 0.
    Temporal = 0,
    /// why (German *kausal*) — group 1.
    Kausal = 1,
    /// how — group 2.
    Modal = 2,
    /// where (German *lokal*) — group 3.
    Lokal = 3,
}

impl TekamoloRole {
    /// All four roles in Te-Ka-Mo-Lo order.
    pub const ALL: [TekamoloRole; 4] = [
        TekamoloRole::Temporal,
        TekamoloRole::Kausal,
        TekamoloRole::Modal,
        TekamoloRole::Lokal,
    ];

    /// The G4D3 group index (0..4) this role names.
    #[inline]
    #[must_use]
    pub const fn group(self) -> u8 {
        self as u8
    }
}

/// A TEKAMOLO reading of a [`FacetCascade`] — the [`CascadeShape::G4D3`] carving
/// with its four groups named Temporal / Kausal / Modal / Lokal, each a 3-byte
/// `256:256:256` cascade. Carries no bytes of its own (wraps the facet); every
/// accessor is the canonical `cascade_byte(G4D3, group, level)` lookup renamed.
///
/// `#[repr(transparent)]` over [`FacetCascade`]: this reading never grows a
/// backing store (const-asserted 16 B below), so it is safe to reinterpret a
/// `&FacetCascade` as a `&TekamoloFacet` where the ClassView selects the reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct TekamoloFacet(pub FacetCascade);

const _: () = assert!(
    core::mem::size_of::<TekamoloFacet>() == core::mem::size_of::<FacetCascade>(),
    "TekamoloFacet is a pure reading — it must never grow a backing store beyond the facet"
);

impl TekamoloFacet {
    /// The carving this reading always uses.
    pub const SHAPE: CascadeShape = CascadeShape::G4D3;

    /// Wrap a facet as a TEKAMOLO reading (borrow, no copy of the payload).
    #[inline]
    #[must_use]
    pub const fn new(facet: FacetCascade) -> Self {
        Self(facet)
    }

    /// Build a facet FROM the four named lanes. Each lane is a coarse→fine
    /// `[tier0, tier1, tier2]` 256-cascade. Packs into the canonical
    /// [`FacetCascade::tier_bytes`] ladder so `cascade_byte(G4D3, …)` reads them
    /// back byte-for-byte (proven by [`tests::roundtrip`]).
    #[inline]
    #[must_use]
    pub const fn from_lanes(
        facet_classid: u32,
        temporal: [u8; 3],
        causal: [u8; 3],
        modal: [u8; 3],
        local: [u8; 3],
    ) -> Self {
        // The G4D3 ladder is [temporal(3) ++ causal(3) ++ modal(3) ++ local(3)];
        // `tier_bytes()` is [t0.hi, t0.lo, t1.hi, t1.lo, …], so tier i takes
        // ladder[2i] as hi (coarse) and ladder[2i+1] as lo (fine).
        let l = [
            temporal[0],
            temporal[1],
            temporal[2], // 0,1,2
            causal[0],
            causal[1],
            causal[2], // 3,4,5
            modal[0],
            modal[1],
            modal[2], // 6,7,8
            local[0],
            local[1],
            local[2], // 9,10,11
        ];
        let tiers = [
            FacetTier { hi: l[0], lo: l[1] },
            FacetTier { hi: l[2], lo: l[3] },
            FacetTier { hi: l[4], lo: l[5] },
            FacetTier { hi: l[6], lo: l[7] },
            FacetTier { hi: l[8], lo: l[9] },
            FacetTier {
                hi: l[10],
                lo: l[11],
            },
        ];
        Self(FacetCascade {
            facet_classid,
            tiers,
        })
    }

    /// The wrapped facet.
    #[inline]
    #[must_use]
    pub const fn facet(&self) -> &FacetCascade {
        &self.0
    }

    /// The `classid` naming this reading.
    #[inline]
    #[must_use]
    pub const fn facet_classid(&self) -> u32 {
        self.0.facet_classid
    }

    /// One byte of a role's cascade — `cascade_byte(G4D3, role, level)`.
    /// `level` 0 = coarse, 2 = fine.
    #[inline]
    #[must_use]
    pub const fn role_byte(&self, role: TekamoloRole, level: u8) -> u8 {
        self.0.cascade_byte(Self::SHAPE, role.group(), level)
    }

    /// A role's whole 3-byte `256:256:256` cascade, coarse→fine.
    #[inline]
    #[must_use]
    pub const fn lane(&self, role: TekamoloRole) -> [u8; 3] {
        [
            self.role_byte(role, 0),
            self.role_byte(role, 1),
            self.role_byte(role, 2),
        ]
    }

    /// The temporal (when) lane.
    #[inline]
    #[must_use]
    pub const fn temporal(&self) -> [u8; 3] {
        self.lane(TekamoloRole::Temporal)
    }

    /// The causal (why / *kausal*) lane.
    #[inline]
    #[must_use]
    pub const fn causal(&self) -> [u8; 3] {
        self.lane(TekamoloRole::Kausal)
    }

    /// The modal (how) lane.
    #[inline]
    #[must_use]
    pub const fn modal(&self) -> [u8; 3] {
        self.lane(TekamoloRole::Modal)
    }

    /// The local (where / *lokal*) lane.
    #[inline]
    #[must_use]
    pub const fn local(&self) -> [u8; 3] {
        self.lane(TekamoloRole::Lokal)
    }

    /// Coarse→fine shared-prefix length (`0..=3`) of ONE role between two facets —
    /// how deep two nodes share a temporal / causal / modal / local ancestry.
    /// This is the per-axis prefix-routing readout (`cascade_group_shared`): two
    /// events that share the same coarse temporal tier are "near in time",
    /// independent of where or why. `3` ⇒ that whole axis agrees.
    #[inline]
    #[must_use]
    pub const fn shared(&self, other: &Self, role: TekamoloRole) -> u8 {
        self.0
            .cascade_group_shared(other.0, Self::SHAPE, role.group())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> TekamoloFacet {
        TekamoloFacet::from_lanes(
            0x0042_1000,
            [10, 11, 12], // temporal
            [20, 21, 22], // causal
            [30, 31, 32], // modal
            [40, 41, 42], // local
        )
    }

    /// The four named lanes read back exactly what `from_lanes` packed — the
    /// naming is a lossless reading of the G4D3 carving.
    #[test]
    fn roundtrip() {
        let t = sample();
        assert_eq!(t.temporal(), [10, 11, 12]);
        assert_eq!(t.causal(), [20, 21, 22]);
        assert_eq!(t.modal(), [30, 31, 32]);
        assert_eq!(t.local(), [40, 41, 42]);
        assert_eq!(t.facet_classid(), 0x0042_1000);
    }

    /// The named accessors ARE the canonical `cascade_byte(G4D3, group, level)` —
    /// byte-for-byte parity with the facet algebra (no private reinterpretation).
    #[test]
    fn parity_with_cascade_byte() {
        let t = sample();
        for role in TekamoloRole::ALL {
            for level in 0..3u8 {
                assert_eq!(
                    t.role_byte(role, level),
                    t.facet()
                        .cascade_byte(CascadeShape::G4D3, role.group(), level),
                    "role {role:?} level {level} must equal the canonical carving byte"
                );
            }
        }
    }

    /// Field-isolation (I-LEGACY-API-FEATURE-GATED discipline for a facet
    /// reading): rewriting ONE lane leaves the other three byte-identical.
    #[test]
    fn lane_isolation() {
        let base = sample();
        let changed = TekamoloFacet::from_lanes(
            base.facet_classid(),
            [99, 98, 97], // temporal changed
            base.causal(),
            base.modal(),
            base.local(),
        );
        assert_eq!(changed.temporal(), [99, 98, 97]);
        assert_eq!(changed.causal(), base.causal(), "causal untouched");
        assert_eq!(changed.modal(), base.modal(), "modal untouched");
        assert_eq!(changed.local(), base.local(), "local untouched");
    }

    /// Per-role shared-prefix: two facets that agree on the WHOLE temporal axis
    /// but diverge on the local axis share time deeply and place shallowly —
    /// the per-axis locality readout, independent across roles.
    #[test]
    fn shared_prefix_is_per_role() {
        let a = sample();
        // Same temporal + causal + modal; local diverges at the coarse tier.
        let b = TekamoloFacet::from_lanes(
            a.facet_classid(),
            a.temporal(),
            a.causal(),
            a.modal(),
            [99, 41, 42], // local: coarse tier differs immediately
        );
        assert_eq!(
            a.shared(&b, TekamoloRole::Temporal),
            3,
            "whole time axis agrees"
        );
        assert_eq!(
            a.shared(&b, TekamoloRole::Kausal),
            3,
            "whole cause axis agrees"
        );
        assert_eq!(
            a.shared(&b, TekamoloRole::Modal),
            3,
            "whole mode axis agrees"
        );
        assert_eq!(
            a.shared(&b, TekamoloRole::Lokal),
            0,
            "place diverges at the coarse tier"
        );
    }

    /// A facet built the ordinary way reads sensibly as TEKAMOLO (the reading is
    /// just a rename — any facet is a valid TEKAMOLO facet).
    #[test]
    fn any_facet_reads_as_tekamolo() {
        let raw = FacetCascade::from_bytes(&[
            0, 0, 0, 0, // classid
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, // 6 tiers
        ]);
        let t = TekamoloFacet::new(raw);
        // Every byte the reading returns is a byte of the underlying facet.
        for role in TekamoloRole::ALL {
            for level in 0..3u8 {
                let _ = t.role_byte(role, level); // no panic, in-bounds
            }
        }
        assert_eq!(t.facet(), &raw);
    }
}
