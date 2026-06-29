//! `facet` — the content-blind **8:8 facet** substrate (a reusable 16-byte primitive).
//!
//! A [`FacetCascade`] is `facet_classid(4) | 6×(8:8) = 16 B` — one 128-bit register.
//! The substrate is **ALWAYS 8:8** (each tier is two opaque bytes `hi:lo`); only the
//! CONSUMER projects meaning onto the bytes — `(part_of:is_a)`, a `256:256` palette
//! (CAM-PQ) centroid pair, `(group:member)`, `(mixin:identity)`, `(column:row)`, a
//! `(Y:Z)` coordinate, or a concatenated `u16`. The producer bakes in nothing
//! (AGI-as-glove: the SoA is content-blind, the reader interprets).
//!
//! It carries **no value-slab offset** — it is a *reading* over a borrowed `[u8; 16]`,
//! so it never touches the operator-LOCKED 480-byte node layout. The
//! `classid → ClassView` wiring that picks which 16 value bytes it reads is a separate
//! step (`soa-value-tenant-migration-v1-harvest.md` §5.1, §5–§6).
//!
//! ## One register, four lanes
//!
//! The same 16 bytes are addressable at four granularities, each a single SIMD op —
//! pick the lens by the operation (measured; the redout is granularity-free):
//!
//! | lens | unit | accessor | hardware op |
//! |---|---|---|---|
//! | **row** | 4× `u32` | [`FacetCascade::rows`] / [`row_match_mask`](FacetCascade::row_match_mask) | `vpcmpeqd` + `vmovmskps` |
//! | **tile** | 8× `u16` (the 8:8) | [`tiers`](FacetCascade::tiers) / [`hi_chain`](FacetCascade::hi_chain) | `vpcmpeqw` / `pshufb` |
//! | **prefix** | bit (LCP) | [`prefix_distance`](FacetCascade::prefix_distance) | `vpxor` + `tzcnt` (granularity-free) |
//! | **nibble** | 32× `[4]` (Morton) | [`FacetTier::morton`] | GFNI `vgf2p8affineqb` (AVX-512) |
//!
//! Row 0 is the `facet_classid` (`{domain}{schema}`); rows 1–3 are the 6 cascade
//! tiers paired coarse→fine (`HEEL:HIP` / `TWIG:LEAF` / `family:identity`). The layout
//! is transpose-native: 4 facets → `_MM_TRANSPOSE4` → SoA columns for a batch sweep.

/// One **8:8 tile** of a [`FacetCascade`] — ALWAYS exactly two bytes, `hi` and `lo`.
/// The substrate is **content-blind**: only the CONSUMER (the
/// [`FacetCascade::facet_classid`]'s ClassView) decides what the 8:8 *means*
/// (`(part_of:is_a)`, a `256:256` palette centroid, `(group:member)`, `(column:row)`,
/// a concatenated `u16`, …). `hi` is the coarse-side byte, `lo` the fine-side byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct FacetTier {
    /// Low byte of the LE 8:8 tile (is_a / member / row / centroid-lo / …).
    pub lo: u8,
    /// High byte of the LE 8:8 tile (part_of / group / column / centroid-hi / …).
    pub hi: u8,
}

impl FacetTier {
    /// The two bytes as the LE `u16 = (hi << 8) | lo` — the "consumer reads the 8:8
    /// as one concatenated 16-bit value" projection.
    #[inline]
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        ((self.hi as u16) << 8) | self.lo as u16
    }

    /// The `hi:lo` pair **Morton-interleaved** into a `u16` Z-order code (`lo` on
    /// even bits, `hi` on odd) — the amortization benefit of the always-8:8
    /// substrate: every nibble of the result is a **2 bit × 2 bit Morton tile**, so a
    /// nibble prefix is a quad-tree quadrant in BOTH bytes at once (`256 = 4⁴`
    /// hierarchical ancestry). Whatever the consumer decides the 8:8 means, it ALWAYS
    /// amortizes to this one Morton tile cascade — uniform prefix routing.
    #[inline]
    #[must_use]
    pub const fn morton(self) -> u16 {
        Self::spread8(self.lo) | (Self::spread8(self.hi) << 1)
    }

    /// Spread a byte's 8 bits to the even positions `0,2,…,14` of a `u16` (the Morton
    /// building block).
    const fn spread8(x: u8) -> u16 {
        let mut v = x as u16; // ........ abcdefgh
        v = (v | (v << 4)) & 0x0F0F; // ....abcd ....efgh
        v = (v | (v << 2)) & 0x3333; // ..ab..cd ..ef..gh
        v = (v | (v << 1)) & 0x5555; // .a.b.c.d .e.f.g.h
        v
    }
}

/// The **FacetCascade** — a content-blind 16-byte facet: `facet_classid(4) | 6×(8:8)`.
///
/// **ALWAYS 8:8.** Six tiers, each two opaque bytes (`hi:lo`); the `facet_classid`'s
/// ClassView decides the interpretation (see [`FacetTier`]). Both bytes of every tier
/// are carried (lossless): the `hi` chain prefix-routes one hierarchy, the `lo` chain
/// the orthogonal one. The full 6-tier facet does NOT fit the 64-bit key `NiblePath`
/// (which carries only the 4-tier HHTL routing prefix,
/// [`crate::hhtl::NiblePath::from_guid_prefix_v3`]) — the complete address lives here.
///
/// A *reading* over a borrowed `[u8; 16]`: NO value-slab offset, does not touch the
/// LOCKED 480-byte layout. `#[repr(C, align(16))]` makes it a 128-bit register value
/// byte-identical to `[u8; 16]`, so decode is a **reinterpret no-op** — see
/// [`ref_from_bytes`](Self::ref_from_bytes) / [`as_bytes`](Self::as_bytes). The
/// compiler reads fields/lanes straight from the backing store; nothing materializes.
/// See the module docs for the one-register / four-lane design.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C, align(16))]
pub struct FacetCascade {
    /// The facet's own class id — `{domain}{schema}`, row 0; which ClassView
    /// interprets the 6 tiers' 8:8.
    pub facet_classid: u32,
    /// 6 tiers coarse→fine: `HEEL·HIP·TWIG·LEAF·family·identity`, each an 8:8 tile.
    pub tiers: [FacetTier; 6],
}

const _: () = assert!(core::mem::size_of::<FacetTier>() == 2, "one 8:8 tile");
const _: () = assert!(
    core::mem::size_of::<FacetCascade>() == 16,
    "facet_classid(4) | 6×(8:8)=12 = 16B (harvest §5.1)"
);

impl FacetCascade {
    /// Decode from the 16 facet bytes (LE): `facet_classid` in `[0..4)`, then 6 tiers,
    /// each an LE `u16 = (hi << 8) | lo` — on the wire `[lo, hi]` (the `converge.rs`
    /// `tier(hi, lo)` byte order, matching the key tiers).
    #[inline]
    #[must_use]
    pub const fn from_bytes(b: &[u8; 16]) -> Self {
        FacetCascade {
            facet_classid: u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            tiers: [
                FacetTier { lo: b[4], hi: b[5] },
                FacetTier { lo: b[6], hi: b[7] },
                FacetTier { lo: b[8], hi: b[9] },
                FacetTier {
                    lo: b[10],
                    hi: b[11],
                },
                FacetTier {
                    lo: b[12],
                    hi: b[13],
                },
                FacetTier {
                    lo: b[14],
                    hi: b[15],
                },
            ],
        }
    }

    /// Encode to the 16 facet bytes (LE), the inverse of [`from_bytes`](Self::from_bytes).
    #[inline]
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 16] {
        let c = self.facet_classid.to_le_bytes();
        let t = &self.tiers;
        [
            c[0], c[1], c[2], c[3], t[0].lo, t[0].hi, t[1].lo, t[1].hi, t[2].lo, t[2].hi, t[3].lo,
            t[3].hi, t[4].lo, t[4].hi, t[5].lo, t[5].hi,
        ]
    }

    /// The whole facet as one LE `u128` — the single-register view (the `vmovdqu`
    /// load). Use for the bit-level redout ([`prefix_distance`](Self::prefix_distance))
    /// and for SIMD batch.
    #[inline]
    #[must_use]
    pub const fn as_u128(self) -> u128 {
        u128::from_le_bytes(self.to_bytes())
    }

    /// Build from the single-register LE `u128` — inverse of [`as_u128`](Self::as_u128).
    #[inline]
    #[must_use]
    pub const fn from_u128(v: u128) -> Self {
        Self::from_bytes(&v.to_le_bytes())
    }

    /// Zero-cost view of the facet AS its 16 LE bytes — a **reinterpret no-op**
    /// (`repr(C, align(16))`, byte-identical to `[u8; 16]`); the compiler emits no
    /// conversion. Companion to [`ref_from_bytes`](Self::ref_from_bytes).
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 16] {
        // SAFETY: FacetCascade is #[repr(C, align(16))], size_of == 16, byte-identical
        // to [u8; 16] and strictly more-aligned (16 ≥ 1). The bytes ARE the facet's own
        // backing store — a pure pointer reinterpret, lifetime tied to `&self`.
        unsafe { &*(self as *const Self).cast::<[u8; 16]>() }
    }

    /// **Zero-copy borrow** of 16 slab bytes AS a facet — the literal no-op decode: the
    /// compiler reads fields/lanes straight from the slab, nothing materializes. Returns
    /// `None` if `b` is not 16-byte aligned (then copy via [`from_bytes`](Self::from_bytes)).
    /// Mirrors `node_rows_from_le_bytes`'s checked reinterpret.
    #[inline]
    #[must_use]
    pub fn ref_from_bytes(b: &[u8; 16]) -> Option<&Self> {
        if !(b.as_ptr() as usize).is_multiple_of(core::mem::align_of::<Self>()) {
            return None;
        }
        // SAFETY: 16-byte alignment checked above; FacetCascade is #[repr(C,
        // align(16))], size_of == 16 == the array, byte-identical layout — a pure
        // reinterpret of the borrow, lifetime tied to `b`.
        Some(unsafe { &*(b.as_ptr().cast::<Self>()) })
    }

    /// The 4 **dword rows** (the 4×4 lane): `[facet_classid, HEEL:HIP, TWIG:LEAF,
    /// family:identity]`. `rows()[0] == facet_classid`. Compares as `vpcmpeqd`.
    #[inline]
    #[must_use]
    pub const fn rows(self) -> [u32; 4] {
        let b = self.to_bytes();
        [
            u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
            u32::from_le_bytes([b[8], b[9], b[10], b[11]]),
            u32::from_le_bytes([b[12], b[13], b[14], b[15]]),
        ]
    }

    /// The `hi`-byte chain, coarse→fine — one hierarchy (part_of / group / column /
    /// centroid-hi, per the consumer).
    #[inline]
    #[must_use]
    pub const fn hi_chain(self) -> [u8; 6] {
        let t = &self.tiers;
        [t[0].hi, t[1].hi, t[2].hi, t[3].hi, t[4].hi, t[5].hi]
    }

    /// The `lo`-byte chain, coarse→fine — the orthogonal hierarchy (is_a / member /
    /// row / centroid-lo, per the consumer).
    #[inline]
    #[must_use]
    pub const fn lo_chain(self) -> [u8; 6] {
        let t = &self.tiers;
        [t[0].lo, t[1].lo, t[2].lo, t[3].lo, t[4].lo, t[5].lo]
    }

    /// Shared coarse→fine prefix length (0..=6) of two 6-byte chains.
    const fn shared6(a: [u8; 6], b: [u8; 6]) -> u8 {
        let mut n = 0u8;
        while (n as usize) < 6 && a[n as usize] == b[n as usize] {
            n += 1;
        }
        n
    }

    /// `hi`-chain distance: `6 − shared hi-prefix` — locality along the `hi` hierarchy,
    /// orthogonal to [`lo_distance`](Self::lo_distance).
    #[inline]
    #[must_use]
    pub const fn hi_distance(self, other: Self) -> u8 {
        6 - Self::shared6(self.hi_chain(), other.hi_chain())
    }

    /// `lo`-chain distance: `6 − shared lo-prefix` — locality along the orthogonal `lo`
    /// hierarchy, on the SAME facet.
    #[inline]
    #[must_use]
    pub const fn lo_distance(self, other: Self) -> u8 {
        6 - Self::shared6(self.lo_chain(), other.lo_chain())
    }

    /// Number of fully-matching low **tiles** (0..=8, classid tiles 0–1 first, then the
    /// 6 cascade tiers) — the granularity-free LCP redout: `(xor).trailing_zeros() / 16`.
    /// `8` ⇒ identical. The whole-facet prefix over class + cascade in one `vpxor`+`tzcnt`.
    #[inline]
    #[must_use]
    pub const fn shared_prefix_tiles(self, other: Self) -> u8 {
        let x = self.as_u128() ^ other.as_u128();
        if x == 0 {
            8
        } else {
            (x.trailing_zeros() / 16) as u8
        }
    }

    /// `8 − shared_prefix_tiles` — the coarse→fine tile distance over the whole facet
    /// (class first, then the cascade). `0` ⇒ identical.
    #[inline]
    #[must_use]
    pub const fn prefix_distance(self, other: Self) -> u8 {
        8 - self.shared_prefix_tiles(other)
    }

    /// 4-bit mask: bit `i` set iff [`row`](Self::rows) `i` matches `other` — the
    /// dword-lane "which of `{class, HEEL:HIP, TWIG:LEAF, family:identity}` agree"
    /// (`vpcmpeqd` + `vmovmskps`).
    #[inline]
    #[must_use]
    pub const fn row_match_mask(self, other: Self) -> u8 {
        let (a, b) = (self.rows(), other.rows());
        let mut m = 0u8;
        let mut i = 0;
        while i < 4 {
            if a[i] == b[i] {
                m |= 1 << i;
            }
            i += 1;
        }
        m
    }

    /// The 12 cascade tier-bytes as **one coarse→fine ladder** — `hi` then `lo`
    /// per tier, tiers in order:
    /// `[t0.hi, t0.lo, t1.hi, t1.lo, … t5.hi, t5.lo]`. Excludes the 4-byte
    /// [`facet_classid`](Self::facet_classid). This is the input the
    /// [`CascadeShape`] algebra re-carves; it is byte-for-byte the same 12-unit
    /// ladder a 12-field class exposes, so a ClassView addresses *either* the
    /// facet bytes or its own fields with the identical `(group, level)` math.
    /// (Distinct from [`hi_chain`](Self::hi_chain)/[`lo_chain`](Self::lo_chain),
    /// which group by *axis* across tiers; this groups by *position* in the
    /// ladder — the `(1:2)`/`(1:2:3)`/`(1:2:3:4)` view.)
    #[inline]
    #[must_use]
    pub const fn tier_bytes(self) -> [u8; CASCADE_UNITS] {
        let t = &self.tiers;
        [
            t[0].hi, t[0].lo, t[1].hi, t[1].lo, t[2].hi, t[2].lo, t[3].hi, t[3].lo, t[4].hi,
            t[4].lo, t[5].hi, t[5].lo,
        ]
    }

    /// The cascade byte at `(group, level)` under `shape` —
    /// `tier_bytes()[shape.index(group, level)]`. The same lookup whether the
    /// ClassView reads the facet as `6×2`, `4×3`, or `3×4`; the bytes never move.
    #[inline]
    #[must_use]
    pub const fn cascade_byte(self, shape: CascadeShape, group: u8, level: u8) -> u8 {
        self.tier_bytes()[shape.index(group, level)]
    }

    /// Coarse→fine shared-prefix length (`0..=D`) of one group `g` between two
    /// facets under `shape` — the **per-group LCP redout**; `D` ⇒ that group's
    /// whole `(1:…:D)` hierarchy agrees. The per-carving refinement of the
    /// whole-facet [`shared_prefix_tiles`](Self::shared_prefix_tiles): pick the
    /// carving, then read locality one group (one axis of meaning) at a time.
    #[inline]
    #[must_use]
    pub const fn cascade_group_shared(self, other: Self, shape: CascadeShape, group: u8) -> u8 {
        let (a, b) = (self.tier_bytes(), other.tier_bytes());
        let d = shape.levels();
        let base = shape.index(group, 0);
        let mut n = 0u8;
        while n < d && a[base + n as usize] == b[base + n as usize] {
            n += 1;
        }
        n
    }
}

/// The number of cascade tier-bytes a [`FacetCascade`] carries (excludes the
/// 4-byte [`FacetCascade::facet_classid`]): `6 tiers × 2 bytes`. Equivalently
/// the field-count of a 12-field class — the cascade algebra is unit-agnostic,
/// so the same `G·D = CASCADE_UNITS` invariant binds bytes and fields alike.
pub const CASCADE_UNITS: usize = 12;

/// **One cascade algebra; carvings are VIEW rotations, not function layouts.**
/// The 12 cascade units (the facet's [`tier_bytes`](FacetCascade::tier_bytes),
/// or a 12-field class's fields) are read as `G groups × D levels` with
/// `G·D = CASCADE_UNITS` *always*. The units never move — a `ClassView` picks
/// the carving (its **rotation**) and the same index math (`i = g·D + l`)
/// addresses any of them. A ClassView can **always rotate** — read the SAME
/// bytes under a different carving — per class.
///
/// **This addresses the VIEW only.** Functions (behaviour) are NOT a facet
/// carving — they are reached by switching the classid to the
/// [`ClassArm::Functions`] arm (the OGAR THINK/DO split). Never slice the
/// tier-bytes to reach a function.
///
/// Two carvings are byte-**aligned** ([`ALIGNED`](Self::ALIGNED) — `group_of`
/// is a shift); the third, [`G4D3`](Self::G4D3), **straddles** tier boundaries
/// (`group_of` must DIVIDE) and is the **worst case** — *prevented as a
/// default*, kept only as the **rare rotation / escape hatch** (some Odoo
/// classes may need it to relieve **classid-stacking entropy** — rotate the
/// reading instead of minting another classid):
///
/// | shape | G×D | notation | role | `group_of` |
/// |---|---|---|---|---|
/// | [`G6D2`](Self::G6D2) | 6 × 2 | `6×(1:2)` | aligned default (native `hi:lo`) | `i >> 1` (shift) |
/// | [`G3D4`](Self::G3D4) | 3 × 4 | `3×(1:2:3:4)` | aligned default (tier-pair super-groups) | `i >> 2` (shift) |
/// | [`G4D3`](Self::G4D3) | 4 × 3 | `4×(1:2:3)` | **worst case** — straddle; rare rotation only | `i / 3` (**divide**) |
///
/// This is the OGAR GUID's `3×4`-vs-`4×3` debate generalized from nibble-units
/// to byte/field-units — and it lands on the canon's verdict: the aligned (3×4)
/// carving is the default; the straddling (4×3) is the worst case, kept legal
/// only so a rotation can reach it deliberately and a guard can reject it
/// elsewhere. With 12 fields it is often cheaper to map a sub-range as a
/// hierarchy and stack **nested** ClassViews into constructors before
/// materializing the `32×GUID` SoA — see `docs/OGAR-TRANSPILE-SUBSTRATE.md` §1.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CascadeShape {
    /// 6 groups × 2 levels — native `(hi:lo)` per tier (`6×(1:2)`). Aligned
    /// default (`group_of` = `i >> 1`).
    G6D2,
    /// 4 groups × 3 levels (`4×(1:2:3)`) — the **straddle**: `group_of` must
    /// divide (`i / 3`), crossing tier boundaries. The **worst case**, NOT a
    /// default — kept only as the rare rotation / escape hatch (e.g. an Odoo
    /// class avoiding classid-stacking entropy). Use [`is_byte_aligned`](Self::is_byte_aligned)
    /// to reject it on the common path.
    G4D3,
    /// 3 groups × 4 levels — tier-pair super-groups (`3×(1:2:3:4)`). Aligned
    /// default (`group_of` = `i >> 2`).
    G3D4,
}

impl CascadeShape {
    /// The full **rotation set** — every carving a `ClassView` may rotate the
    /// 12-unit ladder through (`G·D = 12` each), group-count ascending. A
    /// ClassView can always rotate (read the same facet bytes under a different
    /// grouping) per class; this is the relief valve for **classid-stacking
    /// entropy** (rare, e.g. some Odoo classes — rotate the reading instead of
    /// minting another classid). Includes the straddle [`G4D3`](Self::G4D3) so a
    /// rotation can reach it deliberately; prefer [`ALIGNED`](Self::ALIGNED) on
    /// the common path.
    pub const ROTATIONS: [CascadeShape; 3] =
        [CascadeShape::G3D4, CascadeShape::G4D3, CascadeShape::G6D2];

    /// The byte-**ALIGNED** carvings — the default rotations. `group_of` is a
    /// pure shift for both ([`shift`](Self::shift) is `Some`) — the canon's
    /// "tier-of-level is a shift, never a branch". The straddle [`G4D3`](Self::G4D3)
    /// is deliberately EXCLUDED — it is the worst case
    /// ([`is_byte_aligned`](Self::is_byte_aligned) is `false`), available only as
    /// the rare rotation, never a default.
    pub const ALIGNED: [CascadeShape; 2] = [CascadeShape::G3D4, CascadeShape::G6D2];

    /// `G` — number of groups (axes of meaning). `groups() · levels() == CASCADE_UNITS`.
    #[inline]
    #[must_use]
    pub const fn groups(self) -> u8 {
        match self {
            CascadeShape::G6D2 => 6,
            CascadeShape::G4D3 => 4,
            CascadeShape::G3D4 => 3,
        }
    }

    /// `D` — levels per group: the depth of the `(1:2:…:D)` coarse→fine ladder.
    /// `groups() · levels() == CASCADE_UNITS`.
    #[inline]
    #[must_use]
    pub const fn levels(self) -> u8 {
        match self {
            CascadeShape::G6D2 => 2,
            CascadeShape::G4D3 => 3,
            CascadeShape::G3D4 => 4,
        }
    }

    /// Linear unit index of `(group, level)`: `group · D + level` — groups laid
    /// out in order, coarse→fine within each. The single shared addressing rule
    /// for facet bytes *and* class fields.
    #[inline]
    #[must_use]
    pub const fn index(self, group: u8, level: u8) -> usize {
        (group * self.levels() + level) as usize
    }

    /// Inverse of [`index`](Self::index): which group linear unit `i` belongs to
    /// (`i / D`). For the [`ALIGNED`](Self::ALIGNED) carvings this is a pure
    /// shift (see [`shift`](Self::shift)); for the straddle [`G4D3`](Self::G4D3)
    /// it is a real divide — the worst case. Prefer dispatching on
    /// [`shift`](Self::shift) so the divide path is opt-in (the rare rotation),
    /// never silently on the hot path.
    #[inline]
    #[must_use]
    pub const fn group_of(self, unit: usize) -> u8 {
        (unit / self.levels() as usize) as u8
    }

    /// Inverse of [`index`](Self::index): the within-group level of unit `i`
    /// (`i % D`).
    #[inline]
    #[must_use]
    pub const fn level_of(self, unit: usize) -> u8 {
        (unit % self.levels() as usize) as u8
    }

    /// The bit-shift that implements [`group_of`](Self::group_of) for a
    /// byte-aligned carving — `Some(1)` for [`G6D2`](Self::G6D2) (`i >> 1`),
    /// `Some(2)` for [`G3D4`](Self::G3D4) (`i >> 2`) — or `None` for the straddle
    /// [`G4D3`](Self::G4D3), which has NO shift (`group_of` must divide). `None`
    /// IS the "this rotation is the rare escape hatch, not a default" signal: the
    /// canon's "tier-of-level is a shift, never a branch" holds exactly for the
    /// `Some` carvings.
    #[inline]
    #[must_use]
    pub const fn shift(self) -> Option<u32> {
        match self {
            CascadeShape::G6D2 => Some(1),
            CascadeShape::G3D4 => Some(2),
            CascadeShape::G4D3 => None,
        }
    }

    /// A carving is byte-aligned iff its group boundaries fall on tier
    /// boundaries — [`group_of`](Self::group_of) is a shift, not a divide. True
    /// for the [`ALIGNED`](Self::ALIGNED) defaults; **false** for the straddle
    /// [`G4D3`](Self::G4D3). A default layout MUST assert this; the straddle is a
    /// worst case to PREVENT on the common path, reached only as a deliberate
    /// rare rotation (classid-stacking-entropy relief). Functions are never
    /// reached by any carving — see [`ClassArm`].
    #[inline]
    #[must_use]
    pub const fn is_byte_aligned(self) -> bool {
        self.shift().is_some()
    }
}

/// The classid is an **additional switch**, not only a data address: resolving a
/// classid selects one of two ARMS — the same THINK/DO split the OGAR AST draws
/// (`docs/OGAR-AST-CONTRACT.md`).
///
/// - [`View`](Self::View) — the THINK arm: the class's **data layout**, read by
///   a `ClassView` over the facet bytes carved/rotated per [`CascadeShape`]
///   (byte-aligned on the common path).
/// - [`Functions`](Self::Functions) — the DO arm: the class's **behaviour**
///   (`ActionDef` / `KausalSpec`) on the Core node the classid resolves to.
///
/// **Functions are NOT a facet carving.** Behaviour is reached by switching the
/// classid to the `Functions` arm — never by slicing the facet's tier-bytes (a
/// straddling carve like the worst-case [`CascadeShape::G4D3`] is exactly what
/// that mistake looks like). The carving addresses the VIEW; this switch reaches
/// the functions. (Canon: neither u16 half of the classid carries behaviour —
/// behaviour is a property of the resolved node, *selected* by this arm; see
/// `OGAR/docs/OGAR-CONSUMER-BEST-PRACTICES.md`.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClassArm {
    /// THINK arm — data layout (a `ClassView` over the carved/rotated facet bytes).
    View,
    /// DO arm — behaviour (`ActionDef` / `KausalSpec` on the resolved Core node).
    Functions,
}

impl ClassArm {
    /// The two arms the classid switch selects, `View` first (the
    /// prerender-from-key default; canon "THE GUID IS THE KEY OF KEY-VALUE").
    pub const BOTH: [ClassArm; 2] = [ClassArm::View, ClassArm::Functions];

    /// Whether this arm reaches behaviour (the DO arm). `false` for `View`.
    #[inline]
    #[must_use]
    pub const fn is_functions(self) -> bool {
        matches!(self, ClassArm::Functions)
    }
}

const _: () = assert!(
    CascadeShape::G6D2.groups() as usize * CascadeShape::G6D2.levels() as usize == CASCADE_UNITS,
    "6×2 = 12"
);
const _: () = assert!(
    CascadeShape::G4D3.groups() as usize * CascadeShape::G4D3.levels() as usize == CASCADE_UNITS,
    "4×3 = 12"
);
const _: () = assert!(
    CascadeShape::G3D4.groups() as usize * CascadeShape::G3D4.levels() as usize == CASCADE_UNITS,
    "3×4 = 12"
);

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> [u8; 16] {
        [
            0xEF, 0xBE, 0xAD, 0xDE, // facet_classid = 0xDEAD_BEEF (LE)
            0x01, 0xAB, // tier0 lo=01 hi=AB
            0x02, 0xCD, // tier1
            0x03, 0xEF, // tier2
            0x04, 0x12, // tier3
            0x05, 0x34, // tier4
            0x06, 0x56, // tier5
        ]
    }

    #[test]
    fn always_8_8_consumer_neutral_roundtrip_and_lanes() {
        assert_eq!(core::mem::size_of::<FacetCascade>(), 16);
        assert_eq!(core::mem::size_of::<FacetTier>(), 2);

        let b = sample();
        let f = FacetCascade::from_bytes(&b);
        assert_eq!(f.facet_classid, 0xDEAD_BEEF);
        assert_eq!(f.to_bytes(), b, "round-trip is exact (8:8 stored verbatim)");

        // u128 single-register view round-trips.
        assert_eq!(FacetCascade::from_u128(f.as_u128()), f);
        assert_eq!(f.as_u128(), u128::from_le_bytes(b));

        // The two orthogonal chains (content-neutral hi/lo).
        assert_eq!(f.hi_chain(), [0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56]);
        assert_eq!(f.lo_chain(), [0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);

        // The 4 dword rows; row 0 IS the classid.
        let r = f.rows();
        assert_eq!(r[0], 0xDEAD_BEEF);
        assert_eq!(r[0], f.facet_classid);
        assert_eq!(r[1], u32::from_le_bytes([0x01, 0xAB, 0x02, 0xCD]));

        // Tier projections: concatenated u16 + Morton tile (2bit×2bit).
        assert_eq!(f.tiers[0].as_u16(), 0xAB01);
        assert_eq!(
            f.tiers[0].morton() & 0x5555,
            FacetTier { lo: 0x01, hi: 0 }.morton()
        );
    }

    #[test]
    fn redout_is_granularity_free_and_orthogonal() {
        let f = FacetCascade::from_bytes(&sample());

        // identical ⇒ all 8 tiles shared, distance 0.
        assert_eq!(f.shared_prefix_tiles(f), 8);
        assert_eq!(f.prefix_distance(f), 0);
        assert_eq!(f.row_match_mask(f), 0b1111);

        // Differ only in tier0's is_a (lo) byte ⇒ hi chain intact, lo chain diverges
        // at tier0; the whole-facet prefix breaks after the 2 classid tiles (tile 2).
        let mut b = sample();
        b[4] = 0x99; // tier0 lo
        let g = FacetCascade::from_bytes(&b);
        assert_eq!(f.hi_distance(g), 0, "hi chain unchanged");
        assert!(f.lo_distance(g) > 0, "lo chain diverges at tier0");
        assert_eq!(
            f.shared_prefix_tiles(g),
            2,
            "class (tiles 0-1) shared, tile 2 differs"
        );
        // row 1 (HEEL:HIP, holds tier0) differs; rows 0/2/3 match.
        assert_eq!(f.row_match_mask(g), 0b1101);

        // Differ in the classid (row 0) ⇒ diverge at the very first tile.
        let h = FacetCascade::from_u128(f.as_u128() ^ 1);
        assert_eq!(h.shared_prefix_tiles(f), 0);
        assert_eq!(h.row_match_mask(f), 0b1110);
    }

    #[test]
    fn cascade_rotations_are_total_but_only_aligned_are_defaults() {
        // The rotation set is exhaustive; every rotation covers all 12 units.
        assert_eq!(CascadeShape::ROTATIONS.len(), 3);
        for s in CascadeShape::ROTATIONS {
            assert_eq!(
                s.groups() as usize * s.levels() as usize,
                CASCADE_UNITS,
                "every rotation covers all 12 units exactly"
            );
            // index / group_of / level_of are mutual inverses over the whole ladder.
            for unit in 0..CASCADE_UNITS {
                let (g, l) = (s.group_of(unit), s.level_of(unit));
                assert!(g < s.groups() && l < s.levels());
                assert_eq!(s.index(g, l), unit, "{s:?}: index∘(group,level) = id");
            }
        }

        // ALIGNED is the default subset: group_of IS a shift (canon "shift, not
        // a branch"); shift() is Some and matches.
        assert_eq!(
            CascadeShape::ALIGNED,
            [CascadeShape::G3D4, CascadeShape::G6D2]
        );
        for s in CascadeShape::ALIGNED {
            assert!(s.is_byte_aligned(), "{s:?} is an aligned default");
            let sh = s.shift().expect("aligned carving has a shift");
            for unit in 0..CASCADE_UNITS {
                assert_eq!(
                    s.group_of(unit) as usize,
                    unit >> sh,
                    "{s:?} group_of is a shift"
                );
            }
        }
        assert_eq!(CascadeShape::G6D2.shift(), Some(1));
        assert_eq!(CascadeShape::G3D4.shift(), Some(2));

        // G4D3 is the WORST CASE, not a default: it straddles, has NO shift
        // (group_of DIVIDES), is excluded from ALIGNED, but stays in ROTATIONS
        // as the rare escape hatch (classid-stacking-entropy relief, msg 2).
        assert!(!CascadeShape::G4D3.is_byte_aligned());
        assert_eq!(CascadeShape::G4D3.shift(), None);
        assert!(!CascadeShape::ALIGNED.contains(&CascadeShape::G4D3));
        assert!(CascadeShape::ROTATIONS.contains(&CascadeShape::G4D3));
        assert_eq!(CascadeShape::G4D3.group_of(2) as usize, 2 / 3); // a divide — the bad example
    }

    #[test]
    fn classid_switch_separates_view_from_functions() {
        // The classid is an additional (functions, view) switch; functions are
        // NOT a facet carving — they are the DO arm, reached by this switch.
        assert_eq!(CascadeShape::ROTATIONS.len(), 3); // carvings address the VIEW only
        assert_eq!(ClassArm::BOTH, [ClassArm::View, ClassArm::Functions]);
        assert_ne!(ClassArm::View, ClassArm::Functions);
        assert!(!ClassArm::View.is_functions(), "View is the THINK/data arm");
        assert!(
            ClassArm::Functions.is_functions(),
            "Functions is the DO/behaviour arm"
        );
    }

    #[test]
    fn tier_bytes_ladder_and_per_carving_grouping() {
        let f = FacetCascade::from_bytes(&sample());

        // The 12-unit ladder: hi then lo per tier, coarse→fine. (hi = sample odd
        // bytes 0xAB..0x56; lo = even bytes 0x01..0x06.)
        assert_eq!(
            f.tier_bytes(),
            [0xAB, 0x01, 0xCD, 0x02, 0xEF, 0x03, 0x12, 0x04, 0x34, 0x05, 0x56, 0x06]
        );

        // 6×2: group g == tier g's (hi, lo) — the native pairing.
        for g in 0..6u8 {
            assert_eq!(
                f.cascade_byte(CascadeShape::G6D2, g, 0),
                f.tiers[g as usize].hi
            );
            assert_eq!(
                f.cascade_byte(CascadeShape::G6D2, g, 1),
                f.tiers[g as usize].lo
            );
        }
        // 3×4: group 0 spans tiers 0–1 (4 bytes), byte-aligned super-group.
        assert_eq!(
            [
                f.cascade_byte(CascadeShape::G3D4, 0, 0),
                f.cascade_byte(CascadeShape::G3D4, 0, 1),
                f.cascade_byte(CascadeShape::G3D4, 0, 2),
                f.cascade_byte(CascadeShape::G3D4, 0, 3),
            ],
            [0xAB, 0x01, 0xCD, 0x02]
        );
        // 4×3 (the worst-case rare rotation): group 0 straddles tier 0 fully +
        // tier 1's hi (the 1.5-tier cost) — shown only to demonstrate why it is
        // NOT a default, not to endorse it.
        assert!(!CascadeShape::G4D3.is_byte_aligned());
        assert_eq!(
            [
                f.cascade_byte(CascadeShape::G4D3, 0, 0),
                f.cascade_byte(CascadeShape::G4D3, 0, 1),
                f.cascade_byte(CascadeShape::G4D3, 0, 2),
            ],
            [0xAB, 0x01, 0xCD]
        );
    }

    #[test]
    fn cascade_group_shared_is_per_group_lcp() {
        let f = FacetCascade::from_bytes(&sample());

        // identical ⇒ every group's whole ladder agrees (== D), for every rotation.
        for s in CascadeShape::ROTATIONS {
            for g in 0..s.groups() {
                assert_eq!(f.cascade_group_shared(f, s, g), s.levels());
            }
        }

        // Perturb tier1's hi byte (ladder unit 2). Under 6×2 that is group 1,
        // level 0 ⇒ group 1 diverges immediately (shared 0); group 0 untouched.
        let mut b = sample();
        b[7] = 0x99; // tier1.hi
        let g = FacetCascade::from_bytes(&b);
        assert_eq!(
            f.cascade_group_shared(g, CascadeShape::G6D2, 0),
            2,
            "tier0 intact"
        );
        assert_eq!(
            f.cascade_group_shared(g, CascadeShape::G6D2, 1),
            0,
            "tier1.hi differs first"
        );

        // Under 3×4 unit 2 is group 0, level 2 ⇒ group 0 shares its first 2
        // levels then breaks; group 1 (tiers 2–3) fully intact.
        assert_eq!(f.cascade_group_shared(g, CascadeShape::G3D4, 0), 2);
        assert_eq!(f.cascade_group_shared(g, CascadeShape::G3D4, 1), 4);
    }

    #[test]
    fn reinterpret_is_a_no_op() {
        // align(16) ⇒ the facet's own bytes are 16-aligned, so the zero-copy borrow
        // round-trips: bytes → &FacetCascade reads straight from the same store.
        let f = FacetCascade::from_bytes(&sample());
        let bytes: &[u8; 16] = f.as_bytes();
        assert_eq!(bytes, &f.to_bytes());
        assert_eq!(
            bytes.as_ptr() as usize,
            &f as *const _ as usize,
            "as_bytes is a pointer reinterpret, no copy"
        );
        let g = FacetCascade::ref_from_bytes(bytes).expect("a facet's own bytes are 16-aligned");
        assert_eq!(*g, f);
        assert_eq!(
            g as *const FacetCascade as usize,
            bytes.as_ptr() as usize,
            "ref_from_bytes is a borrow reinterpret, no decode"
        );
        assert_eq!(core::mem::align_of::<FacetCascade>(), 16);
    }
}
