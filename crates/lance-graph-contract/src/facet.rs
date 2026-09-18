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
//! | **prefix** | semantic tile (LCP, canon→custom→tiers) | [`prefix_distance`](FacetCascade::prefix_distance) | `vpxor` + `tzcnt` (+ classid tile swap) |
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

// ⊘ NARROWED. This guard was introduced (codex P2 on #1246) because
// `NodeRow::edges` WAS a `FacetCascade`, so this struct's in-memory image was
// the canonical stored row image — and `facet_classid` is a native-endian
// `u32`, so on a big-endian target `as_bytes` (a reinterpret) and `from_bytes`
// (an explicit `u32::from_le_bytes`) would disagree on `[0..4)` and silently
// byte-swap a class id through serialization. **That storage dependency is
// GONE:** `edges` is now the byte-backed `EdgeFacet([u8; 16])`, so the whole
// 512-byte row is `[u8;16] | [u8;16] | [u8;480]` with no native-endian integer
// anywhere in it, and nothing typed is stored.
//
// What the guard still protects is this type AS A COMPUTE LENS: `as_bytes` /
// `ref_from_bytes` remain pointer reinterprets, and that is deliberate — the
// reinterpret IS the fast path (`examples/facet_axis_lcp_probe.rs` measures the
// byte-chain LCP at 1.72 ns precisely because nothing is materialized). So the
// `reinterpret == encode` identity is still assumed, and is still pinned by
// `le_byte_image_round_trips_with_a_non_zero_classid` below — but a violation
// can now only mis-read a value in flight, never corrupt a row at rest.
//
// The doctrine, in one line: BYTES ARE STORED, INTEGERS ARE PROJECTED. Storage
// is byte-agnostic (any target, any bit pattern); little-endian is a COMPUTE
// superpower and lives on this side of the projection only.
// (`ISS-EDGE-BLOCK-WAS-A-SECOND-TYPE-FOR-THE-SAME-FACET`.)
const _: () = assert!(
    cfg!(target_endian = "little"),
    "FacetCascade's reinterpret-based LE byte image assumes a little-endian target"
);

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

    /// Mutable twin of [`as_bytes`](Self::as_bytes): write the facet's own 16
    /// backing bytes in place. Every byte pattern is a valid facet (it is
    /// content-blind by construction), so no invariant can be broken through
    /// this view.
    #[inline]
    #[must_use]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        // SAFETY: as for `as_bytes`; `&mut self` guarantees exclusivity, and
        // all bit patterns of [u8; 16] are valid `FacetCascade` values.
        unsafe { &mut *(self as *mut Self).cast::<[u8; 16]>() }
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

    /// Shared coarse→fine prefix length (0..=6) along one axis — the early-exit
    /// byte chain fold, comparing tier `t`'s axis byte straight out of each
    /// facet's backing bytes.
    ///
    /// **This is a PEEK, not a mask** (`E-THREE-CARRIERS-THREE-FOLDS-1`). The
    /// cascade is byte-addressed: tier `t`'s axis byte sits at the compile-time
    /// constant offset `4 + 2t` (lo) or `5 + 2t` (hi), so each step lowers to a
    /// `movzbl` plus a `cmp` against the other facet's memory and a `jne` that
    /// exits at the first divergence. Nothing is gathered — LLVM never
    /// materializes the `[u8; 6]`. The single-register masked readout (`xor`,
    /// mask, `tzcnt`, offset-correct) performs the SAME byte loads, then pays
    /// to reassemble them and cannot exit early; measured 2026-09-17 by
    /// `examples/facet_axis_lcp_probe.rs` at **3.64 ns vs 1.72 ns** on 64K
    /// random pairs, and slower at every divergence depth 0..5.
    ///
    /// The masked form is retained as this fold's test oracle
    /// (`masked_axis_oracle`), the same license a raw intrinsic gets under
    /// `#[cfg(test)]`. This is the pre-#1242 `shared6` verbatim — restored, not
    /// rewritten, so the probe's arm A and the shipped path are one function.
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

    /// Number of fully-matching **semantic tiles** (0..=8), coarse→fine: **canon**
    /// (the concept — the HIGH `u16` of `facet_classid`), then **custom** (the app
    /// prefix — the LOW `u16`), then the 6 cascade tiers. `8` ⇒ identical. The
    /// whole-facet prefix over class + cascade, still one `vpxor` + `tzcnt`.
    ///
    /// ⊘ CORRECTED 2026-09-18 — D-DIAMOND-1 R1
    /// (`ISS-SHARED-PREFIX-TILES-CLASSID-INVERSION`). The stored LE image holds
    /// `custom` at bytes `[0..2)` and `canon` at `[2..4)`
    /// (`ClassidOrder::CanonHigh => (canon << 16) | custom`), so the previous
    /// form — `trailing_zeros` straight over `as_u128()` — counted the APP before
    /// the CONCEPT: same concept / different app shared 0 tiles, same app /
    /// different concept shared 1. Fine-before-coarse at exactly the boundary the
    /// canon-high flip exists for. **The stored image is unchanged**; this
    /// projection swaps the two classid tiles of the XOR (a fixed 4-op fixup on
    /// the low 32 bits) before counting, so tile 0 IS canon. Zero callers when
    /// corrected; latent until whole-facet traversal used the lens. Proven by
    /// `diamond_tests::f1_le_byte_order_is_not_semantic_tuple_order`.
    #[inline]
    #[must_use]
    pub const fn shared_prefix_tiles(self, other: Self) -> u8 {
        let x = self.as_u128() ^ other.as_u128();
        // Semantic tile order: canon (image bytes 2..4) is tile 0, custom (image
        // bytes 0..2) is tile 1. Swap the two 16-bit halves of the classid XOR.
        let cls = x as u32;
        let cls = cls.rotate_left(16);
        let x = (x & !0xFFFF_FFFFu128) | cls as u128;
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

    /// The 8 **semantic** tiles, coarse→fine, as their numeric projections:
    /// `[canon, custom, tiers[0].as_u16(), …, tiers[5].as_u16()]` — `canon` is the
    /// HIGH `u16` of [`facet_classid`](Self::facet_classid) (the shared concept),
    /// `custom` the LOW `u16` (the app prefix), per the canon-high flip
    /// (`ogar_codebook::ClassidOrder::CanonHigh`).
    ///
    /// Lexicographic unsigned order over this array is the **normative lane
    /// order** (D-DIAMOND-1 R2) — see
    /// [`cmp_numeric_projection`](Self::cmp_numeric_projection), which compares
    /// the tuple `(facet_classid, tiers[i].as_u16() …)` literally and is proven
    /// equal to this array's order in the tests. Note the two are the same
    /// because comparing the `u32` classid numerically already puts `canon`
    /// (its high half) first; nothing here re-orders the stored bytes.
    #[inline]
    #[must_use]
    pub const fn semantic_tiles(self) -> [u16; 8] {
        let t = &self.tiers;
        [
            (self.facet_classid >> 16) as u16,
            (self.facet_classid & 0xFFFF) as u16,
            t[0].as_u16(),
            t[1].as_u16(),
            t[2].as_u16(),
            t[3].as_u16(),
            t[4].as_u16(),
            t[5].as_u16(),
        ]
    }

    /// Inverse of [`semantic_tiles`](Self::semantic_tiles): rebuild the facet
    /// from its 8 semantic tiles. Round-trips exactly (tested).
    #[inline]
    #[must_use]
    pub const fn from_semantic_tiles(t: [u16; 8]) -> Self {
        const fn tier(v: u16) -> FacetTier {
            FacetTier {
                lo: (v & 0xFF) as u8,
                hi: (v >> 8) as u8,
            }
        }
        FacetCascade {
            facet_classid: ((t[0] as u32) << 16) | t[1] as u32,
            tiers: [
                tier(t[2]),
                tier(t[3]),
                tier(t[4]),
                tier(t[5]),
                tier(t[6]),
                tier(t[7]),
            ],
        }
    }

    /// **The normative lane order (D-DIAMOND-1 R2):** lexicographic unsigned
    /// order over the numeric projections
    /// `(facet_classid, tiers[0].as_u16(), …, tiers[5].as_u16())` — *numeric
    /// projection order over the canonical LE image*. `facet_classid` is compared
    /// as its projected `u32`, which preserves canon-high semantics (the concept
    /// is the high half, so it decides first).
    ///
    /// This is NOT byte-wise order over the stored image: the LE image holds
    /// `custom` at bytes `[0..2)` and each tile's fine byte before its coarse
    /// byte, so a `memcmp` of the image orders fine-before-coarse. The
    /// projections are what carry the hierarchy; the bytes only store it.
    /// The 8 semantic tiles packed into two `u64` planes, coarse tile in the
    /// HIGH bits: `hi = t0<<48 | t1<<32 | t2<<16 | t3`, `lo = t4<<48 | … | t7`.
    /// Numeric order over `(hi, lo)` equals
    /// [`cmp_numeric_projection`](Self::cmp_numeric_projection), so a prefix of
    /// `d` tiles is a `MatchU64` with `care = u64::MAX << (64 - 16·d)` on `hi`
    /// (and on `lo` for `d > 4`) — the SWEEP form a prefix lowers to when no
    /// ordering witness is available (`ordered_lane`).
    #[inline]
    #[must_use]
    pub const fn semantic_u64_halves(self) -> (u64, u64) {
        let t = self.semantic_tiles();
        (
            ((t[0] as u64) << 48) | ((t[1] as u64) << 32) | ((t[2] as u64) << 16) | t[3] as u64,
            ((t[4] as u64) << 48) | ((t[5] as u64) << 32) | ((t[6] as u64) << 16) | t[7] as u64,
        )
    }

    #[must_use]
    pub fn cmp_numeric_projection(&self, other: &Self) -> core::cmp::Ordering {
        self.facet_classid.cmp(&other.facet_classid).then_with(|| {
            let mut i = 0;
            while i < 6 {
                match self.tiers[i].as_u16().cmp(&other.tiers[i].as_u16()) {
                    core::cmp::Ordering::Equal => i += 1,
                    o => return o,
                }
            }
            core::cmp::Ordering::Equal
        })
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

/// A **semantic prefix** over a [`FacetCascade`]: the first `depth` of its 8
/// semantic tiles (see [`FacetCascade::semantic_tiles`]), coarse→fine. `depth`
/// is `0..=8`; `0` matches every facet, `8` matches exactly one key value.
///
/// On a lane in [numeric projection order](FacetCascade::cmp_numeric_projection)
/// the facets matching a prefix are CONTIGUOUS and bracketed by
/// [`lo_key`](Self::lo_key) / [`hi_key`](Self::hi_key) — which is what lets a
/// prefix predicate lower to a bound (`lower_bound + upper_bound +
/// mask_set_range`) instead of a sweep. On an unordered lane it is just a
/// predicate with holes; that is why the lowering is gated on an ordering
/// witness (`ordered_lane::OrderedLaneWitness`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SemanticPrefix {
    tiles: [u16; 8],
    depth: u8,
}

impl SemanticPrefix {
    /// The first `depth` semantic tiles of `f`. `depth` is clamped to 8.
    #[must_use]
    pub const fn of(f: FacetCascade, depth: u8) -> Self {
        let depth = if depth > 8 { 8 } else { depth };
        let src = f.semantic_tiles();
        let mut tiles = [0u16; 8];
        let mut i = 0;
        while i < depth as usize {
            tiles[i] = src[i];
            i += 1;
        }
        SemanticPrefix { tiles, depth }
    }

    /// Number of leading semantic tiles this prefix fixes (`0..=8`).
    #[must_use]
    pub const fn depth(self) -> u8 {
        self.depth
    }

    /// The fixed tiles; entries at index `>= depth()` are zero.
    #[must_use]
    pub const fn tiles(self) -> [u16; 8] {
        self.tiles
    }

    /// Does `f` carry this prefix? — its first `depth` semantic tiles equal ours.
    /// Equivalent to `shared_prefix_tiles(lo_key()) >= depth`, spelled directly.
    #[must_use]
    pub const fn matches(self, f: FacetCascade) -> bool {
        let t = f.semantic_tiles();
        let mut i = 0;
        while i < self.depth as usize {
            if t[i] != self.tiles[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    /// The smallest key carrying this prefix (unfixed tiles `= 0`).
    #[must_use]
    pub const fn lo_key(self) -> FacetCascade {
        FacetCascade::from_semantic_tiles(self.tiles)
    }

    /// The largest key carrying this prefix (unfixed tiles `= 0xFFFF`).
    #[must_use]
    pub const fn hi_key(self) -> FacetCascade {
        let mut t = self.tiles;
        let mut i = self.depth as usize;
        while i < 8 {
            t[i] = 0xFFFF;
            i += 1;
        }
        FacetCascade::from_semantic_tiles(t)
    }
}

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
/// **The shape is class-conditioned, not locked.** A `ClassView` is *mapped
/// from the class's inherited format* and selected by `classid` (the filter), so
/// a framework picks the carving its schema implies — **Rails → `6×2`, other
/// frameworks → `4×3`, the canonical GUID → `3×4`** (all `G·D = 12`, 8-bit
/// tiers; the per-group depth `D ∈ {2,3,4}` is the per-class knob, see
/// [`from_levels`](Self::from_levels)). Each is legitimate for the class that
/// needs it; none is restated or locked here.
///
/// | shape | G×D | notation | framework | `group_of` |
/// |---|---|---|---|---|
/// | [`G6D2`](Self::G6D2) | 6 × 2 | `6×(1:2)` | Rails (native `hi:lo`) | `i >> 1` (shift) |
/// | [`G4D3`](Self::G4D3) | 4 × 3 | `4×(1:2:3)` | other frameworks | `i / 3` (divide) |
/// | [`G3D4`](Self::G3D4) | 3 × 4 | `3×(1:2:3:4)` | canonical GUID (tier-pair super-groups) | `i >> 2` (shift) |
///
/// `G6D2`/`G3D4` carve on tier boundaries so `group_of` is a pure shift
/// ([`ALIGNED`](Self::ALIGNED) — the canon's "tier-of-level is a shift, never a
/// branch"); `G4D3` straddles, so its `group_of` divides — a **per-class cost a
/// class opts into when its schema needs `4×3`**, not a prohibition. This is the
/// OGAR GUID `3×4`-vs-`4×3` debate generalized from nibble-units to byte/field-
/// units: `3×4` is the GUID default, the others are class-conditioned. With 12
/// fields a class may also map a sub-range as a hierarchy and stack **nested**
/// ClassViews into constructors before materializing the `32×GUID` SoA — see
/// `docs/OGAR-TRANSPILE-SUBSTRATE.md` §1.5.
///
/// **Clean / SoC over packed.** What stays the last resort is cramming two
/// *distinct concerns* into one facet (independent of shape): a node has
/// [`GUIDS_PER_NODE`](crate::canonical_node::GUIDS_PER_NODE) = 32 sixteen-byte
/// slots, so the cheap move is to *Tetris* each concern into its own slot
/// (separation-of-concerns) rather than bit-pack. (`G4D3`'s divide is a per-class
/// *shape* cost, separate from this — a class whose schema needs `4×3` is clean.)
///
/// **Encoding-lane scope.** These byte-shapes (8-bit tiers) are the **transpile /
/// ClassView field-grouping** lane. A *separate* `G2×48bit` lane reads the same 12
/// tier-bytes as the two 48-bit chains ([`hi_chain`](FacetCascade::hi_chain) /
/// [`lo_chain`](FacetCascade::lo_chain), cf. the CAM-PQ `6×256` path code) — for
/// **helix** (location) and **CAM-PQ** (centroid) encoding. That lane is **not
/// required by transpile** and is never dragged into ClassView shape selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CascadeShape {
    /// 6 groups × 2 levels — native `(hi:lo)` per tier (`6×(1:2)`). The Rails
    /// shape; `group_of` is the shift `i >> 1`.
    G6D2,
    /// 4 groups × 3 levels (`4×(1:2:3)`) — the **class-conditioned** shape for
    /// frameworks whose inherited schema implies `4×3` ("other frameworks"; not
    /// the GUID default `3×4`, not Rails' `6×2`). It straddles tier boundaries so
    /// `group_of` divides (`i / 3`) — the per-class *cost* the class opts into,
    /// not a prohibition. [`is_byte_aligned`](Self::is_byte_aligned) is `false`
    /// (it distinguishes the divide shape from the shift shapes — not a "reject").
    G4D3,
    /// 3 groups × 4 levels — tier-pair super-groups (`3×(1:2:3:4)`). The
    /// canonical GUID shape; `group_of` is the shift `i >> 2`.
    G3D4,
}

impl CascadeShape {
    /// Every shape, group-count ascending — the full set a class may be
    /// carved/rotated through (`G·D = 12` each). Which one a class uses is
    /// **class-conditioned**: `classid` selects it from the inherited schema
    /// (Rails `6×2`, other frameworks `4×3`, the GUID `3×4`) — see
    /// [`from_levels`](Self::from_levels). A ClassView can also always rotate
    /// (read the same bytes under a different grouping) per class.
    pub const ROTATIONS: [CascadeShape; 3] =
        [CascadeShape::G3D4, CascadeShape::G4D3, CascadeShape::G6D2];

    /// The byte-**aligned** shapes — `group_of` is a pure shift
    /// ([`shift`](Self::shift) is `Some`), the canon's "tier-of-level is a shift,
    /// never a branch". `G6D2` (Rails) and `G3D4` (GUID). [`G4D3`](Self::G4D3)
    /// (other frameworks) is excluded because its `group_of` *divides*, not
    /// because it is forbidden — it is a legitimate class-conditioned shape.
    pub const ALIGNED: [CascadeShape; 2] = [CascadeShape::G3D4, CascadeShape::G6D2];

    /// Select the shape by its per-group depth `D` — the **class-conditioned
    /// knob** (operator 2026-06-29): a framework/class picks `D ∈ {2,3,4}` from
    /// its inherited format and the shape follows — `2 → G6D2` (Rails),
    /// `3 → G4D3` (other frameworks), `4 → G3D4` (the GUID default). `None` for
    /// any other `D` (only 2/3/4 divide the 12-unit ladder). This is the
    /// inverse of [`levels`](Self::levels); the classid resolves `D`, not a lock.
    #[inline]
    #[must_use]
    pub const fn from_levels(d: u8) -> Option<CascadeShape> {
        match d {
            2 => Some(CascadeShape::G6D2),
            3 => Some(CascadeShape::G4D3),
            4 => Some(CascadeShape::G3D4),
            _ => None,
        }
    }

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
    ///
    /// **Precondition:** `group < groups()` and `level < levels()` (the result is
    /// then in `0..CASCADE_UNITS`). The multiply/add is done in `usize` (widen
    /// first), so an out-of-range argument cannot wrap a `u8` — and a
    /// `debug_assert` catches the misuse in debug builds.
    #[inline]
    #[must_use]
    pub const fn index(self, group: u8, level: u8) -> usize {
        debug_assert!(
            group < self.groups() && level < self.levels(),
            "CascadeShape::index: (group, level) out of range for this shape"
        );
        group as usize * self.levels() as usize + level as usize
    }

    /// Inverse of [`index`](Self::index): which group linear unit `i` belongs to
    /// (`i / D`). For the [`ALIGNED`](Self::ALIGNED) shapes this is a pure shift
    /// (see [`shift`](Self::shift)); for [`G4D3`](Self::G4D3) it is a real divide
    /// — the per-class cost of the `4×3` shape. Dispatch on [`shift`](Self::shift)
    /// when you want the shift fast-path for the aligned shapes.
    ///
    /// **Precondition:** `unit < CASCADE_UNITS` — the inverse identity
    /// `index(group_of(u), level_of(u)) == u` holds only on the 12-unit ladder
    /// (`debug_assert`-checked).
    #[inline]
    #[must_use]
    pub const fn group_of(self, unit: usize) -> u8 {
        debug_assert!(
            unit < CASCADE_UNITS,
            "CascadeShape::group_of: unit out of range"
        );
        (unit / self.levels() as usize) as u8
    }

    /// Inverse of [`index`](Self::index): the within-group level of unit `i`
    /// (`i % D`). **Precondition:** `unit < CASCADE_UNITS` (`debug_assert`-checked;
    /// the inverse identity holds only on the 12-unit ladder).
    #[inline]
    #[must_use]
    pub const fn level_of(self, unit: usize) -> u8 {
        debug_assert!(
            unit < CASCADE_UNITS,
            "CascadeShape::level_of: unit out of range"
        );
        (unit % self.levels() as usize) as u8
    }

    /// The bit-shift that implements [`group_of`](Self::group_of) for a
    /// byte-aligned shape — `Some(1)` for [`G6D2`](Self::G6D2) (`i >> 1`),
    /// `Some(2)` for [`G3D4`](Self::G3D4) (`i >> 2`) — or `None` for
    /// [`G4D3`](Self::G4D3), whose `group_of` divides. `Some` carvings satisfy the
    /// canon's "tier-of-level is a shift, never a branch"; `None` marks the
    /// `4×3` shape's per-class divide cost (a property, not a verdict).
    #[inline]
    #[must_use]
    pub const fn shift(self) -> Option<u32> {
        match self {
            CascadeShape::G6D2 => Some(1),
            CascadeShape::G3D4 => Some(2),
            CascadeShape::G4D3 => None,
        }
    }

    /// A shape is byte-aligned iff its group boundaries fall on tier boundaries —
    /// [`group_of`](Self::group_of) is a shift, not a divide. True for the
    /// [`ALIGNED`](Self::ALIGNED) shapes (`6×2`/`3×4`); **false** for
    /// [`G4D3`](Self::G4D3) (`4×3`), whose `group_of` divides. This distinguishes
    /// the shift fast-path from the divide shape — it is not a "prevent" gate;
    /// `4×3` is a legitimate class-conditioned shape. Functions are never reached
    /// by any carving — see [`ClassArm`].
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

    /// The **reinterpret** view and the **explicit LE codec** must agree byte for
    /// byte on a non-zero `facet_classid`.
    ///
    /// These are two different mechanisms, and only one of them was pinned before:
    /// `to_bytes` encodes with `u32::to_le_bytes`, while `as_bytes` (and
    /// `ref_from_bytes`) reinterpret the struct's own memory — which is what
    /// `NodeRowPacket::as_le_bytes` and `row_bytes` serialize, now that a facet is
    /// a STORED row field (`NodeRow::edges`). A zero class id byte-swaps to itself
    /// and would hide the divergence, so the fixture's `0xDEAD_BEEF` is doing the
    /// work here. The compile-time `target_endian` guard above is what keeps this
    /// property from being merely asserted on the one target that satisfies it.
    /// (codex P2 on PR #1246.)
    #[test]
    fn le_byte_image_round_trips_with_a_non_zero_classid() {
        let b = sample();
        let f = FacetCascade::from_bytes(&b);
        assert_ne!(f.facet_classid, 0, "a zero class id cannot detect a swap");
        assert_ne!(
            f.facet_classid.swap_bytes(),
            f.facet_classid,
            "the fixture class id must be byte-order sensitive"
        );

        assert_eq!(
            f.as_bytes(),
            &f.to_bytes(),
            "the reinterpret view must equal the explicit LE encoding"
        );
        assert_eq!(f.as_bytes(), &b, "and both must equal the source bytes");

        // The borrowed no-op decode reads the same image.
        let aligned = FacetCascade::from_bytes(&b);
        assert_eq!(
            FacetCascade::ref_from_bytes(aligned.as_bytes()),
            Some(&aligned)
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

        // Differ in the classid (row 0). Bit 0 of the image is the LOW half of the
        // classid = `custom` (the app prefix), which is SEMANTIC tile 1, not 0.
        // ⊘ 2026-09-18 (D-DIAMOND-1 R1): this asserted `0` while the lens counted
        // the raw LE tile order; the concept (canon, semantic tile 0) is shared
        // here, so the corrected lens reports 1. Flipping a canon bit reports 0.
        let h = FacetCascade::from_u128(f.as_u128() ^ 1);
        assert_eq!(h.shared_prefix_tiles(f), 1, "custom differs, canon shared");
        assert_eq!(h.row_match_mask(f), 0b1110);
        let h2 = FacetCascade::from_u128(f.as_u128() ^ (1 << 16));
        assert_eq!(
            h2.shared_prefix_tiles(f),
            0,
            "canon differs ⇒ nothing shared"
        );
    }

    /// The shipped byte-chain fold against the **masked single-register oracle**
    /// it is measured faster than, at every divergence position and on the
    /// identical case.
    ///
    /// Direction reversed 2026-09-17 (`E-THREE-CARRIERS-THREE-FOLDS-1`): the
    /// masked readout is now the `#[cfg(test)]` oracle and the chain is what
    /// ships, so this stays a real differential rather than becoming a
    /// restatement of the implementation. The `−32` in the oracle is the
    /// classid's 32 bits sitting below the tiers in the LE `u128`: the mask
    /// removes the classid's *bits*, never its *offset*. Disable-verified
    /// 2026-09-16 in its prior direction: swapping the hi/lo masks fails
    /// `hi flip at tier 0`; it still does.
    #[test]
    fn folded_axis_prefix_matches_the_loop_at_every_position() {
        /// The retired single-register readout, kept as the oracle: xor the whole
        /// facet, mask to one axis's six bytes, `tzcnt`, subtract the classid's
        /// 32-bit offset, divide by the 16-bit tier stride.
        const fn masked_axis_oracle(a: u128, b: u128, axis_off: u32) -> u8 {
            let mut mask = 0u128;
            let mut t = 0;
            while t < 6 {
                mask |= 0xFF << (8 * (4 + 2 * t + axis_off));
                t += 1;
            }
            let x = (a ^ b) & mask;
            if x == 0 {
                6
            } else {
                ((x.trailing_zeros() - 32) / 16) as u8
            }
        }
        let f = FacetCascade::from_bytes(&sample());
        let base = sample();
        // identical: both axes fully shared (the oracle's xor == 0 clamp).
        assert_eq!(f.hi_distance(f), 0);
        assert_eq!(f.lo_distance(f), 0);
        // flip exactly tier `t`'s hi byte, then its lo byte: prefix must be `t` on
        // that axis and 6 on the other, and equal the loop's answer.
        for t in 0..6usize {
            for (axis_off, is_hi) in [(1usize, true), (0usize, false)] {
                let mut b = base;
                b[4 + 2 * t + axis_off] ^= 0x80;
                let g = FacetCascade::from_bytes(&b);
                let (sh, sl) = (6 - f.hi_distance(g) as usize, 6 - f.lo_distance(g) as usize);
                let (xf, xg) = (f.as_u128(), g.as_u128());
                assert_eq!(sh, masked_axis_oracle(xf, xg, 1) as usize, "hi t={t}");
                assert_eq!(sl, masked_axis_oracle(xf, xg, 0) as usize, "lo t={t}");
                if is_hi {
                    assert_eq!((sh, sl), (t, 6), "hi flip at tier {t}");
                } else {
                    assert_eq!((sh, sl), (6, t), "lo flip at tier {t}");
                }
            }
        }
    }

    #[test]
    fn cascade_shapes_are_total_and_class_conditioned() {
        // Every shape covers all 12 units; index/group_of/level_of are inverses.
        assert_eq!(CascadeShape::ROTATIONS.len(), 3);
        for s in CascadeShape::ROTATIONS {
            assert_eq!(s.groups() as usize * s.levels() as usize, CASCADE_UNITS);
            for unit in 0..CASCADE_UNITS {
                let (g, l) = (s.group_of(unit), s.level_of(unit));
                assert!(g < s.groups() && l < s.levels());
                assert_eq!(s.index(g, l), unit, "{s:?}: index∘(group,level) = id");
            }
        }

        // The shape is CLASS-CONDITIONED, selected by depth D (the classid knob):
        // 2 → G6D2 (Rails), 3 → G4D3 (other frameworks), 4 → G3D4 (GUID). Round-
        // trips with levels(); only 2/3/4 divide 12.
        assert_eq!(CascadeShape::from_levels(2), Some(CascadeShape::G6D2));
        assert_eq!(CascadeShape::from_levels(3), Some(CascadeShape::G4D3));
        assert_eq!(CascadeShape::from_levels(4), Some(CascadeShape::G3D4));
        assert_eq!(CascadeShape::from_levels(1), None);
        assert_eq!(CascadeShape::from_levels(5), None);
        for s in CascadeShape::ROTATIONS {
            assert_eq!(CascadeShape::from_levels(s.levels()), Some(s));
        }

        // The aligned shapes (Rails 6×2, GUID 3×4) have a shift group_of; G4D3
        // (other frameworks, 4×3) divides — a per-class cost, NOT a prohibition.
        assert_eq!(
            CascadeShape::ALIGNED,
            [CascadeShape::G3D4, CascadeShape::G6D2]
        );
        for s in CascadeShape::ALIGNED {
            let sh = s.shift().expect("aligned shape has a shift");
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
        // G4D3 is legitimate but not a shift shape: divide group_of, not in ALIGNED.
        assert!(!CascadeShape::G4D3.is_byte_aligned());
        assert_eq!(CascadeShape::G4D3.shift(), None);
        assert!(!CascadeShape::ALIGNED.contains(&CascadeShape::G4D3));
        assert!(CascadeShape::ROTATIONS.contains(&CascadeShape::G4D3));
        assert_eq!(CascadeShape::G4D3.group_of(2) as usize, 2 / 3); // a divide, its per-class cost
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

/// D-DIAMOND-1 contract tests (R1 semantic projection, R2 normative order, F1,
/// F5). Kept as their own module so the falsifiers read as one block.
#[cfg(test)]
mod diamond_tests {
    use super::*;
    use core::cmp::Ordering;

    /// The pre-R1 lens, verbatim: `trailing_zeros` over the raw LE image. Kept
    /// ONLY as the thing F1 proves wrong at the classid boundary.
    fn raw_le_prefix_tiles(a: FacetCascade, b: FacetCascade) -> u8 {
        let x = a.as_u128() ^ b.as_u128();
        if x == 0 {
            8
        } else {
            (x.trailing_zeros() / 16) as u8
        }
    }

    fn key(canon: u16, custom: u16, tiers: [u16; 6]) -> FacetCascade {
        FacetCascade::from_semantic_tiles([
            canon, custom, tiers[0], tiers[1], tiers[2], tiers[3], tiers[4], tiers[5],
        ])
    }

    /// F1 — LE byte order is not semantic tuple order at the classid boundary.
    ///
    /// Same concept, different app: semantically ONE tile shared (canon), but the
    /// raw LE scan says ZERO because `custom` sits at bytes 0..2. Same app,
    /// different concept: semantically ZERO shared, raw says ONE. The corrected
    /// projection reverses both.
    #[test]
    fn f1_le_byte_order_is_not_semantic_tuple_order() {
        let t = [0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666];
        let account_move_odoo = key(0x0202, 0x0001, t);
        let account_move_medcare = key(0x0202, 0x0002, t);
        let res_partner_odoo = key(0x0303, 0x0001, t);

        // The stored image is what it always was: custom first, canon second.
        let b = account_move_odoo.to_bytes();
        assert_eq!(&b[0..2], &[0x01, 0x00], "custom at bytes 0..2 (LE)");
        assert_eq!(&b[2..4], &[0x02, 0x02], "canon at bytes 2..4 (LE)");

        // Raw LE scan: app before concept — the inversion.
        assert_eq!(
            raw_le_prefix_tiles(account_move_odoo, account_move_medcare),
            0
        );
        assert_eq!(raw_le_prefix_tiles(account_move_odoo, res_partner_odoo), 1);

        // Corrected semantic projection: concept before app.
        assert_eq!(
            account_move_odoo.shared_prefix_tiles(account_move_medcare),
            1,
            "same concept, different app ⇒ canon tile shared"
        );
        assert_eq!(
            account_move_odoo.shared_prefix_tiles(res_partner_odoo),
            0,
            "same app, different concept ⇒ nothing shared"
        );
        // The stored bytes did not move.
        assert_eq!(account_move_odoo.to_bytes(), b);
    }

    #[test]
    fn semantic_tiles_round_trip_and_name_the_halves() {
        let f = key(0xBEEF, 0xDEAD, [1, 2, 3, 4, 5, 6]);
        assert_eq!(f.facet_classid, 0xBEEF_DEAD, "canon high, custom low");
        assert_eq!(f.semantic_tiles(), [0xBEEF, 0xDEAD, 1, 2, 3, 4, 5, 6]);
        assert_eq!(FacetCascade::from_semantic_tiles(f.semantic_tiles()), f);
        // And through the LE image.
        assert_eq!(
            FacetCascade::from_bytes(&f.to_bytes()).semantic_tiles(),
            f.semantic_tiles()
        );
    }

    /// R2 — the normative tuple compare equals lexicographic order over the
    /// semantic tiles, on a deliberately adversarial set (every tile position
    /// decides at least one pair, and the classid halves disagree with byte order).
    #[test]
    fn r2_numeric_projection_order_is_semantic_tile_lexicographic() {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut keys = Vec::new();
        for _ in 0..512 {
            let mut t = [0u16; 8];
            for x in &mut t {
                // small alphabets so ties at each position are common
                *x = (next() % 4) as u16;
            }
            keys.push(FacetCascade::from_semantic_tiles(t));
        }
        let mut decided_at = [0usize; 8];
        for a in &keys {
            for b in &keys {
                let by_tuple = a.cmp_numeric_projection(b);
                let by_tiles = a.semantic_tiles().cmp(&b.semantic_tiles());
                assert_eq!(by_tuple, by_tiles);
                if by_tuple != Ordering::Equal {
                    let (ta, tb) = (a.semantic_tiles(), b.semantic_tiles());
                    let i = (0..8).find(|&i| ta[i] != tb[i]).unwrap();
                    decided_at[i] += 1;
                }
            }
        }
        assert!(
            decided_at.iter().all(|&n| n > 0),
            "every position must decide some pair: {decided_at:?}"
        );

        // And the thing R2 forbids as normative: byte-wise image order differs.
        let by_bytes_differs = keys
            .iter()
            .zip(keys.iter().skip(1))
            .any(|(a, b)| a.to_bytes().cmp(&b.to_bytes()) != a.cmp_numeric_projection(b));
        assert!(
            by_bytes_differs,
            "byte-wise LE order must NOT coincide with the normative order"
        );
    }

    /// F5 — every prefix depth brackets exactly the matching keys on a sorted
    /// lane: 0, the canon boundary (1), the custom boundary (2), every cascade
    /// tier (3..=7), full-depth exact match (8). Checked against the `matches`
    /// oracle AND against `shared_prefix_tiles >= depth`.
    #[test]
    fn f5_prefix_lo_hi_bracket_exactly_the_matching_keys_at_every_depth() {
        let mut keys: Vec<FacetCascade> = Vec::new();
        for canon in [0x0100u16, 0x0200] {
            for custom in [1u16, 2] {
                for t0 in [0u16, 7] {
                    for t5 in [0u16, 1, 0xFFFF] {
                        keys.push(key(canon, custom, [t0, 3, 3, 3, 3, t5]));
                    }
                }
            }
        }
        keys.sort_by(FacetCascade::cmp_numeric_projection);
        let probe = key(0x0200, 1, [7, 3, 3, 3, 3, 1]);
        for depth in 0..=8u8 {
            let p = SemanticPrefix::of(probe, depth);
            let lo =
                keys.partition_point(|k| k.cmp_numeric_projection(&p.lo_key()) == Ordering::Less);
            let hi = keys
                .partition_point(|k| k.cmp_numeric_projection(&p.hi_key()) != Ordering::Greater);
            let expect: Vec<usize> = (0..keys.len()).filter(|&i| p.matches(keys[i])).collect();
            assert_eq!((lo..hi).collect::<Vec<_>>(), expect, "depth {depth}");
            for (i, k) in keys.iter().enumerate() {
                assert_eq!(
                    p.matches(*k),
                    probe.shared_prefix_tiles(*k) >= depth,
                    "depth {depth} row {i}: matches must agree with the tzcnt lens"
                );
            }
            match depth {
                0 => assert_eq!(hi - lo, keys.len(), "depth 0 is the whole lane"),
                8 => assert_eq!(hi - lo, 1, "full depth is exactly one key"),
                _ => assert!(
                    hi - lo > 0 && hi - lo < keys.len(),
                    "depth {depth} non-trivial"
                ),
            }
        }
    }

    /// The two-`u64` sweep projection orders exactly as the normative tuple —
    /// so the sweep and the bound lower the SAME predicate.
    #[test]
    fn semantic_u64_halves_order_equals_numeric_projection_order() {
        let ks = [
            key(1, 2, [3, 4, 5, 6, 7, 8]),
            key(1, 2, [3, 4, 5, 6, 7, 9]),
            key(1, 3, [0, 0, 0, 0, 0, 0]),
            key(2, 0, [0, 0, 0, 0, 0, 0]),
            key(0xFFFF, 0xFFFF, [0xFFFF; 6]),
        ];
        for a in &ks {
            for b in &ks {
                assert_eq!(
                    a.semantic_u64_halves().cmp(&b.semantic_u64_halves()),
                    a.cmp_numeric_projection(b)
                );
            }
        }
        let (h, l) = key(0xAAAA, 0xBBBB, [1, 2, 3, 4, 5, 6]).semantic_u64_halves();
        assert_eq!(h, 0xAAAA_BBBB_0001_0002);
        assert_eq!(l, 0x0003_0004_0005_0006);
    }
}
