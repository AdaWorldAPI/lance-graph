//! Canonical SoA node — LOCKED minimal layout + zero-fallback ladder.
//!
//! Decisions pinned here (everything else comes after):
//!   * key byte/print order: classid · HEEL · HIP · TWIG · family · identity (LE)
//!   * family + identity are the CONTIGUOUS TRAILING 6 BYTES → the basin-local
//!     key you can use alone after an HHTL radix walk (skip the prefix).
//!   * edge block = 12 in-family + 4 out-of-family, one byte per slot (canonical,
//!     not mandatory — always reserved, never shrunk; opt-out is registry-resolved).
//!   * node = 4096 bit = 512 byte = key(16) | edges(16) | value(480).
//!
//! ## Zero-fallback ladder (monotonic: zero = fall through to the broader default)
//!   * classid  == 0x0000_0000  → default class,  no prefix routing   (dormant)
//!   * family   == 0x00_0000     → default basin,  no neighborhood grouping (dormant)
//!   * ⇒ while both are zero, `identity` (3 bytes / 24 bits) ALONE discriminates.
//!
//! RESERVE, DON'T RECLAIM: a zero tier means "not consulted", never "compacted
//! away". classid(4B) and family(3B) keep their fixed offsets so a non-zero mint
//! later wakes routing/basin binding with ZERO layout change.
//!
//! No UUID ceremony: no version nibble, no variant bits, no namespace/kind framing.
//! Little-endian throughout so the trailing-6-byte local key is a single masked load.

/// 16-byte canonical instance key.
///
/// ```text
///   0..4   classid   (u32)   ← 8 hex, prefix-routable; default 0x0000_0000
///   4..6   HEEL      (u16)   ┐
///   6..8   HIP       (u16)   ├ 3 cascade tiers (HHTL path)
///   8..10  TWIG      (u16)   ┘
///  10..13  family    (u24)   ┐ trailing 6 bytes = basin-local key
///  13..16  identity  (u24)   ┘ (usable alone once the prefix is trie-resolved)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C, align(16))]
pub struct NodeGuid([u8; 16]);

impl NodeGuid {
    /// Reserved canonical default class (implicit fallback; no prefix routing).
    pub const CLASSID_DEFAULT: u32 = 0x0000_0000;
    /// Reserved canonical default basin (implicit fallback; no neighborhood grouping).
    pub const FAMILY_DEFAULT: u32 = 0x00_0000;

    /// OGAR class for the **OSINT / Palantir-Gotham** domain — the neo4j-emulation
    /// entity graph (people / orgs / systems / events, family-grouped). Resolves
    /// to [`ReadMode::OSINT`] (hot `Cognitive` value + `CoarseOnly` adjacency edges).
    pub const CLASSID_OSINT: u32 = 0x0000_0007;
    /// OGAR class for the **FMA anatomy** domain — the Foundational Model of
    /// Anatomy (~70k structural entities, family = body region, bones = stability
    /// anchors). Resolves to [`ReadMode::FMA`] (cold `Compressed` reference value +
    /// `CoarseOnly` part-of adjacency).
    pub const CLASSID_FMA: u32 = 0x0000_0008;

    /// Construct from the six canonical groups. `family`/`identity` use their low 3 bytes.
    ///
    /// Panics (incl. const-eval) when `family` or `identity` exceed 24 bits — the
    /// silent-truncation footgun: distinct u32 inputs would otherwise collapse
    /// to the same stored key.
    pub const fn new(
        classid: u32,
        heel: u16,
        hip: u16,
        twig: u16,
        family: u32,
        identity: u32,
    ) -> Self {
        assert!(family <= 0x00FF_FFFF, "family must fit in 24 bits");
        assert!(identity <= 0x00FF_FFFF, "identity must fit in 24 bits");
        let c = classid.to_le_bytes();
        let h = heel.to_le_bytes();
        let p = hip.to_le_bytes();
        let t = twig.to_le_bytes();
        let f = family.to_le_bytes(); // low 3 bytes
        let i = identity.to_le_bytes(); // low 3 bytes
        Self([
            c[0], c[1], c[2], c[3], //  0..4  classid
            h[0], h[1], //  4..6  HEEL
            p[0], p[1], //  6..8  HIP
            t[0], t[1], //  8..10 TWIG
            f[0], f[1], f[2], // 10..13 family
            i[0], i[1], i[2], // 13..16 identity
        ])
    }

    /// Default-class, default-basin node: only `identity` discriminates.
    /// This is the bootstrap address while classid and family are zero.
    pub const fn local(identity: u32) -> Self {
        Self::new(
            Self::CLASSID_DEFAULT,
            0,
            0,
            0,
            Self::FAMILY_DEFAULT,
            identity,
        )
    }

    #[inline]
    pub const fn classid(&self) -> u32 {
        u32::from_le_bytes([self.0[0], self.0[1], self.0[2], self.0[3]])
    }

    #[inline]
    pub const fn family(&self) -> u32 {
        u32::from_le_bytes([self.0[10], self.0[11], self.0[12], 0])
    }

    #[inline]
    pub const fn identity(&self) -> u32 {
        u32::from_le_bytes([self.0[13], self.0[14], self.0[15], 0])
    }

    /// HEEL — HHT cascade tier 1 (bytes 4..6, LE `u16`).
    #[inline]
    pub const fn heel(&self) -> u16 {
        u16::from_le_bytes([self.0[4], self.0[5]])
    }

    /// HIP — HHT cascade tier 2 (bytes 6..8, LE `u16`).
    #[inline]
    pub const fn hip(&self) -> u16 {
        u16::from_le_bytes([self.0[6], self.0[7]])
    }

    /// TWIG — HHT cascade tier 3 (bytes 8..10, LE `u16`).
    #[inline]
    pub const fn twig(&self) -> u16 {
        u16::from_le_bytes([self.0[8], self.0[9]])
    }

    /// Decode the whole key in one read — every canon group as its native
    /// LE-decoded integer. This is the "read the GUID as a GUID" surface: a
    /// consumer or OGAR gets `classid + HHT (HEEL/HIP/TWIG) + family + identity`
    /// from one call instead of re-deriving each group from raw bytes. The six
    /// fields ARE the canon print order — nothing invented, nothing dropped (cf.
    /// [`Display`](NodeGuid#impl-Display-for-NodeGuid), which renders the same six).
    #[inline]
    pub const fn decode(&self) -> GuidParts {
        GuidParts {
            classid: self.classid(),
            heel: self.heel(),
            hip: self.hip(),
            twig: self.twig(),
            family: self.family(),
            identity: self.identity(),
        }
    }

    /// The [`ReadMode`] this node's `classid` resolves to — which value tenants
    /// to materialise + how to read the edge block. The carrier-method form (the
    /// object speaks for itself): a consumer reads `guid.read_mode()`, OGAR reads
    /// [`classid_read_mode`]`(guid.classid())`; both inherit the SAME answer from
    /// the one [`LazyLock`] registry, so the LE interpretation of the node's bytes
    /// is single-sourced. Not `const` — it consults the runtime registry.
    #[inline]
    pub fn read_mode(&self) -> ReadMode {
        classid_read_mode(self.classid())
    }

    /// Basin-local key: trailing 6 bytes (family ++ identity), zero-padded to u64.
    /// After an HHTL radix walk has bound classid+HEEL+HIP+TWIG, this is the only
    /// part that still discriminates — a single masked load, no gather.
    #[inline]
    pub const fn local_key(&self) -> u64 {
        u64::from_le_bytes([
            self.0[10], self.0[11], self.0[12], self.0[13], self.0[14], self.0[15], 0, 0,
        ])
    }

    // ── fallback-ladder dispatch guards ─────────────────────────────────────
    /// `true` while the classid is the implicit default (no prefix routing).
    #[inline]
    pub const fn is_default_class(&self) -> bool {
        self.classid() == Self::CLASSID_DEFAULT
    }
    /// `true` while the family is the implicit default basin (no grouping).
    #[inline]
    pub const fn is_unbasined(&self) -> bool {
        self.family() == Self::FAMILY_DEFAULT
    }
    /// `true` when both tiers fall through and only `identity` discriminates.
    #[inline]
    pub const fn is_bootstrap_address(&self) -> bool {
        self.is_default_class() && self.is_unbasined()
    }

    #[inline]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Mint-path guard: while in the default basin, `identity` (24 bits) is the
    /// ONLY discriminator, so the mint path MUST guarantee its uniqueness. Call
    /// on insert with whatever set/bitmap the mint path keeps; this centralises
    /// the invariant so it can't be forgotten while family is still a no-op.
    #[inline]
    pub fn debug_assert_identity_unique(&self, already_present: bool) {
        if self.is_bootstrap_address() {
            debug_assert!(
                !already_present,
                "identity collision in default basin: 24-bit identity space exhausted \
                 or reused — mint a non-zero family to expand before this fires in prod"
            );
        }
    }
}

// ── GUID v2 tail (leaf·family·identity, 3×u16) — D-GV2-1, feature-gated ────────
//
// The v2 basin tail repartitions bytes 10..16: leaf(u16) 10..12 (the 4th HHTL
// tier), family(u16) 12..14 (the basin / episodic hub), identity(u16) 14..16
// (the instance). Bytes 0..10 (classid·HEEL·HIP·TWIG) are IDENTICAL to v1.
// Additive and NON-breaking: v1 `new`/`family`/`identity` are untouched; these
// v2 accessors coexist behind `guid-v2-tail` until cutover (D-GV2-5). Per
// I-LEGACY-API-FEATURE-GATED the v2 names are distinct (`leaf`/`*_v2`), so no
// function silently changes semantics, and `GUID_TAIL_LAYOUT_VERSION_V2` is the
// version gate marking a v2-tail packet.
#[cfg(feature = "guid-v2-tail")]
impl NodeGuid {
    /// Construct a v2-tail GUID: `classid·HEEL·HIP·TWIG` identical to v1, then the
    /// 3×u16 basin tail `leaf·family·identity`. Each tail field is a full `u16` —
    /// no 24-bit truncation footgun (the point of v2).
    #[allow(clippy::too_many_arguments)]
    pub const fn new_v2(
        classid: u32,
        heel: u16,
        hip: u16,
        twig: u16,
        leaf: u16,
        family: u16,
        identity: u16,
    ) -> Self {
        let c = classid.to_le_bytes();
        let h = heel.to_le_bytes();
        let p = hip.to_le_bytes();
        let t = twig.to_le_bytes();
        let l = leaf.to_le_bytes();
        let f = family.to_le_bytes();
        let i = identity.to_le_bytes();
        Self([
            c[0], c[1], c[2], c[3], //  0..4  classid
            h[0], h[1], //  4..6  HEEL
            p[0], p[1], //  6..8  HIP
            t[0], t[1], //  8..10 TWIG
            l[0], l[1], // 10..12 leaf   (4th HHTL tier)
            f[0], f[1], // 12..14 family (basin / episodic hub)
            i[0], i[1], // 14..16 identity (instance)
        ])
    }

    /// v2 `leaf` — bytes 10..12, the 4th HHTL routing tier (cascade terminal).
    #[inline]
    pub const fn leaf(&self) -> u16 {
        u16::from_le_bytes([self.0[10], self.0[11]])
    }

    /// v2 `family` — bytes 12..14, the basin / episodic-hub tier (the codebook
    /// selector). Distinct from v1 [`family`](NodeGuid::family) (u24 at 10..13):
    /// different name, different bytes — no silent semantic swap.
    #[inline]
    pub const fn family_v2(&self) -> u16 {
        u16::from_le_bytes([self.0[12], self.0[13]])
    }

    /// v2 `identity` — bytes 14..16, the instance tier (full `u16`).
    #[inline]
    pub const fn identity_v2(&self) -> u16 {
        u16::from_le_bytes([self.0[14], self.0[15]])
    }

    /// v2 basin-local key: trailing 4 bytes (family ++ identity), zero-padded to
    /// `u32` — the discriminator once the HHTL prefix (incl. leaf) is bound.
    #[inline]
    pub const fn local_key_v2(&self) -> u32 {
        u32::from_le_bytes([self.0[12], self.0[13], self.0[14], self.0[15]])
    }

    /// v2 decode — every tier (`classid·HEEL·HIP·TWIG·leaf·family·identity`) as a
    /// native integer. The "read the GUID as a GUID" surface for v2.
    #[inline]
    pub const fn decode_v2(&self) -> GuidPartsV2 {
        GuidPartsV2 {
            classid: self.classid(),
            heel: self.heel(),
            hip: self.hip(),
            twig: self.twig(),
            leaf: self.leaf(),
            family: self.family_v2(),
            identity: self.identity_v2(),
        }
    }

    /// v2 self-describing hex: `classid-heel-hip-twig-leaf-family-identity`,
    /// uniform 4-hex groups (classid as 8) — the v2 Display shape.
    pub fn to_hex_v2(&self) -> String {
        let p = self.decode_v2();
        format!(
            "{:08x}-{:04x}-{:04x}-{:04x}-{:04x}-{:04x}-{:04x}",
            p.classid, p.heel, p.hip, p.twig, p.leaf, p.family, p.identity
        )
    }
}

/// The v2-tail GUID decoded — `classid · HEEL · HIP · TWIG · leaf · family ·
/// identity`, every tier a native integer (no `u24`). The v2 counterpart of
/// [`GuidParts`]. (D-GV2-1; feature `guid-v2-tail`.)
#[cfg(feature = "guid-v2-tail")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GuidPartsV2 {
    /// 0..4 — prefix-routable class id.
    pub classid: u32,
    /// 4..6 — HEEL (HHT cascade tier 1).
    pub heel: u16,
    /// 6..8 — HIP (HHT cascade tier 2).
    pub hip: u16,
    /// 8..10 — TWIG (HHT cascade tier 3).
    pub twig: u16,
    /// 10..12 — leaf, the 4th HHTL tier.
    pub leaf: u16,
    /// 12..14 — family, the basin / episodic hub.
    pub family: u16,
    /// 14..16 — identity, the instance.
    pub identity: u16,
}

/// v2 layout-version marker: a v2-tail packet is layout version 2. A v1 reader
/// MUST refuse a v2 blob (and vice-versa) — the version gate per
/// `I-LEGACY-API-FEATURE-GATED`. Wired into the `SoaEnvelope` version at cutover
/// (D-GV2-5).
#[cfg(feature = "guid-v2-tail")]
pub const GUID_TAIL_LAYOUT_VERSION_V2: u16 = 2;

/// The whole canonical key decoded in one shot — `classid · HEEL · HIP · TWIG ·
/// family · identity`, each as its native LE-decoded integer.
///
/// This is the "read the GUID as a GUID and return classid + HHT + Leaf +
/// identity" contract: one decode, six fields, in canon print order. It invents
/// nothing — it is exactly [`NodeGuid::decode`] of the existing 16-byte key, the
/// same six groups [`NodeGuid`]'s `Display` renders. `family` is the basin
/// "Leaf" and `family ++ identity` is the trailing-6-byte [`NodeGuid::local_key`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GuidParts {
    /// 0..4 — prefix-routable class id (default `0x0000_0000`).
    pub classid: u32,
    /// 4..6 — HEEL (HHT cascade tier 1).
    pub heel: u16,
    /// 6..8 — HIP (HHT cascade tier 2).
    pub hip: u16,
    /// 8..10 — TWIG (HHT cascade tier 3).
    pub twig: u16,
    /// 10..13 — family (u24, the basin "Leaf").
    pub family: u32,
    /// 13..16 — identity (u24).
    pub identity: u32,
}

/// Canonical self-describing print: `classid-HEEL-HIP-TWIG-family·identity`.
///
/// The dash-groups ARE the semantic delimiters — every printed GUID is
/// self-describing at sight (OGAR canon, P0). `{:08x}-{:04x}-{:04x}-{:04x}-{:06x}{:06x}`
/// renders the canonical 8-4-4-4-12 hex layout regardless of in-memory byte
/// order (the field accessors fold LE bytes into u32/u16/u24 first).
impl core::fmt::Display for NodeGuid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:08x}-{:04x}-{:04x}-{:04x}-{:06x}{:06x}",
            self.classid(),
            self.heel(),
            self.hip(),
            self.twig(),
            self.family(),
            self.identity(),
        )
    }
}

/// 16-byte canonical edge block: 12 in-family + 4 out-of-family.
///
/// Canonical, not mandatory: the 16 bytes are ALWAYS reserved (zeroed when unused).
/// A class never shrinks this block — opting out of edges is resolved via
/// classid → ClassView in the registry, never by changing the row stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C, align(16))]
pub struct EdgeBlock {
    /// 12 local adjacency slots (basin-local), one byte each.
    pub in_family: [u8; 12],
    /// 4 inherited adapter slots (out-of-family interfaces), one byte each.
    pub out_family: [u8; 4],
}

/// Which edge-codec flavor a class uses to *read* its node's edge block.
///
/// The flavor is an INTERPRETATION of the canonical 16-byte [`EdgeBlock`] (plus
/// an optional value-slab residue), selected per class via
/// [`ClassView::edge_codec_flavor`](crate::class_view::ClassView::edge_codec_flavor)
/// — never a change to [`NodeRow`]'s 512-byte layout. Every variant leaves
/// [`NODE_ROW_STRIDE`] untouched (the canon "registry-resolved via
/// `classid → ClassView`" rule), so adopting a flavor needs NO
/// `ENVELOPE_LAYOUT_VERSION` bump.
///
/// Encode/reconstruct kernels live in `ndarray::hpc::edge_codec`; per-flavor
/// fidelity is measured by `ndarray::hpc::reliability` (see the
/// `edge_codec_compare` example — CoarseResidue dominates on agreement, Pq32x4
/// preserves rank but not absolute distance). Default is [`CoarseOnly`], the
/// zero-fallback reading that matches the canon all-zero bootstrap default.
///
/// [`CoarseOnly`]: EdgeCodecFlavor::CoarseOnly
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum EdgeCodecFlavor {
    /// 1 byte/vector: each edge byte is a palette/centroid index — the
    /// [`EdgeBlock`] read literally. The canon zero-fallback default.
    #[default]
    CoarseOnly = 0,
    /// 1 + ⌈D/2⌉ bytes: coarse index + a per-dimension signed-4-bit residue
    /// carried in the reserved value slab. Highest fidelity / agreement.
    CoarseResidue = 1,
    /// 16 bytes: the edge block read as 32 × 4-bit product-quantizer codes (the
    /// turbovec PQ model). Preserves neighbour *rank* better than absolute
    /// distance (low ICC, decent Spearman).
    Pq32x4 = 2,
}

impl EdgeCodecFlavor {
    /// Per-vector byte cost for dimensionality `dim` (D even for the residue's
    /// nibble packing; `⌈D/2⌉` is used so odd D rounds up).
    #[inline]
    pub const fn bytes_per_vector(self, dim: usize) -> usize {
        match self {
            EdgeCodecFlavor::CoarseOnly => 1,
            EdgeCodecFlavor::CoarseResidue => 1 + dim.div_ceil(2),
            EdgeCodecFlavor::Pq32x4 => 16,
        }
    }

    /// Every flavor re-interprets the SAME 512-byte node row — none changes
    /// [`NODE_ROW_STRIDE`], so no flavor requires a layout-version bump. This is
    /// the canon invariant, encoded so a regression test can assert it.
    #[inline]
    pub const fn is_layout_preserving(self) -> bool {
        true
    }
}

/// One node = 4096 bit = 512 byte: key(16) | edges(16) | value(480).
///
/// The 480-byte value is deferred — energy/meta/qualia/entity_type, materialized
/// CausalEdge64, helix residue, fingerprint, class extensions all land here later,
/// Lance-compressible. This is the row the MailboxSoA owns and the MailboxSoaView reads.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct NodeRow {
    pub key: NodeGuid,    //  0..16
    pub edges: EdgeBlock, // 16..32
    pub value: [u8; 480], // 32..512  (reserved — comes after)
}

// Sizes are part of the lock.
const _: () = assert!(core::mem::size_of::<NodeGuid>() == 16);
const _: () = assert!(core::mem::size_of::<EdgeBlock>() == 16);
const _: () = assert!(core::mem::size_of::<NodeRow>() == 512);

// ── SoaEnvelope binding for [NodeRow] ────────────────────────────────────────

use crate::class_view::FieldMask;
use crate::soa_envelope::{ColumnDescriptor, ColumnKind, SoaEnvelope};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Stable column-id ordinals for [`NodeRow`]'s three top-level slots.
/// `name_id` in the [`ColumnDescriptor`] table; the registry-resolved value
/// carve-out (per `classid → ClassView`) lives *inside* `Value` and is not
/// surfaced as its own envelope column — the canon contract is at this level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum NodeRowColumn {
    Key = 0,
    Edges = 1,
    Value = 2,
}

/// Canonical [`ColumnDescriptor`] table for [`NodeRow`].
///
/// Three columns, all `ColumnKind::U8` byte-arrays (their internal structure
/// is canon-described elsewhere — `NodeGuid` decomposes the key, `EdgeBlock`
/// the edges, registry `ClassView` carves the value side). The envelope
/// contract is at the row-stride level: bytes 0..16 are the key, 16..32 are
/// the edges, 32..512 are the class-resolved value slab. Sum = 512 = stride.
pub const NODE_ROW_COLUMNS: &[ColumnDescriptor] = &[
    ColumnDescriptor {
        name_id: NodeRowColumn::Key as u16,
        kind: ColumnKind::U8,
        elems_per_row: 16,
        row_offset: 0,
    },
    ColumnDescriptor {
        name_id: NodeRowColumn::Edges as u16,
        kind: ColumnKind::U8,
        elems_per_row: 16,
        row_offset: 16,
    },
    ColumnDescriptor {
        name_id: NodeRowColumn::Value as u16,
        kind: ColumnKind::U8,
        elems_per_row: 480,
        row_offset: 32,
    },
];

/// Row stride for [`NodeRow`] in bytes — equal to `size_of::<NodeRow>()`.
pub const NODE_ROW_STRIDE: usize = 512;

// ── Value-slab schema presets: which tenants a class materialises ─────────────

/// Full-row byte offset of the value slab (key 16 + edges 16).
pub const VALUE_SLAB_ROW_OFFSET: usize = 32;
/// Bytes available in the [`NodeRow::value`] slab.
pub const VALUE_SLAB_LEN: usize = 480;

/// A named tenant of the 480-byte [`NodeRow::value`] slab.
///
/// Stable, append-only positions — the canon "reserve, don't reclaim" rule and
/// the [`FieldMask`] N3 contract: a tenant's presence bit and its byte offset in
/// [`VALUE_TENANTS`] never move once instances persist, and retired tenants are
/// never reused. **The discriminant IS the [`FieldMask`] bit position** and the
/// index into [`VALUE_TENANTS`] (asserted at compile time below).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ValueTenant {
    /// `MetaWord` — thinking / awareness / NARS / free-energy bits.
    Meta = 0,
    /// `QualiaI4_16D` — 16 signed-4-bit chroma channels.
    Qualia = 1,
    /// The 4 out-of-family edges materialised as full `CausalEdge64`.
    MaterializedEdges = 2,
    /// `Fingerprint<256>` — 32-byte identity print.
    Fingerprint = 3,
    /// helix golden-spiral Place/Residue — signed full-sphere `Signed360`,
    /// 48-bit = 6 B (2× the 24-bit equal-area hemisphere; produced by the `helix`
    /// crate's `Signed360`, written here zero-copy).
    HelixResidue = 4,
    /// turbovec PQ residue ([`EdgeCodecFlavor::Pq32x4`], 16 B).
    TurbovecResidue = 5,
    /// Spatio-temporal accumulator (`f32`).
    Energy = 6,
    /// Hebbian plasticity counter + last-active stamp.
    Plasticity = 7,
    /// OGIT entity-type / class discriminator (`u16`).
    EntityType = 8,
}

impl ValueTenant {
    /// This tenant's byte offset **within the 480-byte value slab** (its row
    /// offset minus [`VALUE_SLAB_ROW_OFFSET`]). The companion to its
    /// [`VALUE_TENANTS`] descriptor — lets a transcode write into
    /// [`NodeRow::value`] without hardcoding the carve. Not a new property: a
    /// derived accessor over the already-locked, compile-asserted carve.
    #[inline]
    pub const fn value_offset(self) -> usize {
        VALUE_TENANTS[self as usize].row_offset as usize - VALUE_SLAB_ROW_OFFSET
    }

    /// This tenant's byte length in the slab (from its [`VALUE_TENANTS`] descriptor).
    #[inline]
    pub const fn byte_len(self) -> usize {
        VALUE_TENANTS[self as usize].col_bytes_per_row()
    }
}

/// Stable byte carve of the value slab. Offsets are **row-relative** (within one
/// row packet, in the value region `[32, 512)`) — consistent with
/// [`NODE_ROW_COLUMNS`], one level finer. Contiguous, in [`ValueTenant`]
/// discriminant order, no gaps; the Full set fits the slab (all asserted at
/// compile time below). This is the per-class carve the canon defers to
/// `ClassView`; it is NOT surfaced as its own top-level envelope column.
pub const VALUE_TENANTS: &[ColumnDescriptor] = &[
    ColumnDescriptor {
        name_id: ValueTenant::Meta as u16,
        kind: ColumnKind::U64,
        elems_per_row: 1,
        row_offset: 32,
    },
    ColumnDescriptor {
        name_id: ValueTenant::Qualia as u16,
        kind: ColumnKind::U64,
        elems_per_row: 1,
        row_offset: 40,
    },
    ColumnDescriptor {
        name_id: ValueTenant::MaterializedEdges as u16,
        kind: ColumnKind::U64,
        elems_per_row: 4,
        row_offset: 48,
    },
    ColumnDescriptor {
        name_id: ValueTenant::Fingerprint as u16,
        kind: ColumnKind::U8,
        elems_per_row: 32,
        row_offset: 80,
    },
    ColumnDescriptor {
        name_id: ValueTenant::HelixResidue as u16,
        kind: ColumnKind::U8,
        // 6 B = 48 bit = 2× the 24-bit equal-area hemisphere (helix `Signed360`,
        // signed full sphere). Was 48 B — a bits→bytes slip; right-sized 2026-06-15.
        elems_per_row: 6,
        row_offset: 112,
    },
    ColumnDescriptor {
        name_id: ValueTenant::TurbovecResidue as u16,
        kind: ColumnKind::U8,
        elems_per_row: 16,
        row_offset: 118,
    },
    ColumnDescriptor {
        name_id: ValueTenant::Energy as u16,
        kind: ColumnKind::F32,
        elems_per_row: 1,
        row_offset: 134,
    },
    ColumnDescriptor {
        name_id: ValueTenant::Plasticity as u16,
        kind: ColumnKind::U32,
        elems_per_row: 1,
        row_offset: 138,
    },
    ColumnDescriptor {
        name_id: ValueTenant::EntityType as u16,
        kind: ColumnKind::U16,
        elems_per_row: 1,
        row_offset: 142,
    },
];

// Compile-time canon: VALUE_TENANTS is discriminant-ordered, contiguous within the
// value slab, and the Full carve fits the 480-byte slab.
const _: () = {
    let mut i = 0usize;
    let mut prev_end = VALUE_SLAB_ROW_OFFSET;
    while i < VALUE_TENANTS.len() {
        let c = &VALUE_TENANTS[i];
        assert!(
            c.name_id as usize == i,
            "ValueTenant discriminant must equal its VALUE_TENANTS index"
        );
        assert!(
            c.row_offset as usize == prev_end,
            "VALUE_TENANTS must be contiguous within the value slab (no gaps/overlap)"
        );
        prev_end = c.row_offset as usize + c.col_bytes_per_row();
        i += 1;
    }
    assert!(
        prev_end <= NODE_ROW_STRIDE,
        "value tenants must fit within the 512-byte row"
    );
    assert!(
        prev_end - VALUE_SLAB_ROW_OFFSET <= VALUE_SLAB_LEN,
        "value tenants must fit the 480-byte slab"
    );
};

/// Which value-slab schema a class materialises — the value-side analog of
/// [`EdgeCodecFlavor`]. A preset is a presence [`FieldMask`] over [`ValueTenant`]
/// positions; a class selects it via
/// [`ClassView::value_schema`](crate::class_view::ClassView::value_schema).
///
/// **Layout-preserving:** every preset carves WITHIN the reserved 480-byte value
/// slab, so the choice never changes [`NODE_ROW_STRIDE`] (no
/// `ENVELOPE_LAYOUT_VERSION` bump — canon "registry-resolved via
/// `classid → ClassView`", never a stride change). [`Bootstrap`] is the
/// zero-fallback default: value all zero, only key + edges meaningful.
///
/// [`Bootstrap`]: ValueSchema::Bootstrap
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum ValueSchema {
    /// Empty value slab — the canon zero-fallback (key + edges only).
    #[default]
    Bootstrap = 0,
    /// Hot self-thinking set: Meta + Qualia + Fingerprint + Energy + Plasticity +
    /// EntityType. No materialised edges, no codec residues.
    Cognitive = 1,
    /// Cold / compressed codec stack: Fingerprint + Helix `Signed360` (6 B) +
    /// turbovec residue + EntityType. No hot lifecycle columns.
    Compressed = 2,
    /// Every [`ValueTenant`] materialised — the densest node.
    Full = 3,
}

impl ValueSchema {
    /// The presence [`FieldMask`] over [`ValueTenant`] positions for this preset.
    pub const fn field_mask(self) -> FieldMask {
        match self {
            ValueSchema::Bootstrap => FieldMask::EMPTY,
            ValueSchema::Cognitive => FieldMask::from_positions(&[
                ValueTenant::Meta as u8,
                ValueTenant::Qualia as u8,
                ValueTenant::Fingerprint as u8,
                ValueTenant::Energy as u8,
                ValueTenant::Plasticity as u8,
                ValueTenant::EntityType as u8,
            ]),
            ValueSchema::Compressed => FieldMask::from_positions(&[
                ValueTenant::Fingerprint as u8,
                ValueTenant::HelixResidue as u8,
                ValueTenant::TurbovecResidue as u8,
                ValueTenant::EntityType as u8,
            ]),
            ValueSchema::Full => FieldMask::from_positions(&[
                ValueTenant::Meta as u8,
                ValueTenant::Qualia as u8,
                ValueTenant::MaterializedEdges as u8,
                ValueTenant::Fingerprint as u8,
                ValueTenant::HelixResidue as u8,
                ValueTenant::TurbovecResidue as u8,
                ValueTenant::Energy as u8,
                ValueTenant::Plasticity as u8,
                ValueTenant::EntityType as u8,
            ]),
        }
    }

    /// Does this preset materialise `tenant`?
    #[inline]
    pub const fn has(self, tenant: ValueTenant) -> bool {
        self.field_mask().has(tenant as u8)
    }

    /// Total bytes this preset occupies in the value slab (Σ present tenants).
    pub const fn tenant_bytes(self) -> usize {
        let mask = self.field_mask();
        let mut total = 0usize;
        let mut i = 0usize;
        while i < VALUE_TENANTS.len() {
            let c = &VALUE_TENANTS[i];
            if mask.has(c.name_id as u8) {
                total += c.col_bytes_per_row();
            }
            i += 1;
        }
        total
    }

    /// Every preset carves within the reserved 480-byte slab — none changes
    /// [`NODE_ROW_STRIDE`], so none forces an `ENVELOPE_LAYOUT_VERSION` bump.
    #[inline]
    pub const fn is_layout_preserving(self) -> bool {
        true
    }
}

// Compile-time canon: the densest preset fits the slab; Full covers every tenant;
// Bootstrap is empty.
const _: () = assert!(ValueSchema::Full.tenant_bytes() <= VALUE_SLAB_LEN);
const _: () = assert!(ValueSchema::Full.field_mask().count() as usize == VALUE_TENANTS.len());
const _: () = assert!(ValueSchema::Bootstrap.field_mask().is_empty());

// ── classid → read-mode: the LE contract both the consumer and OGAR inherit ────

/// The **read mode** a `classid` resolves to: the pair of *already-existing*
/// read-mode axes — [`ValueSchema`] (which value tenants to materialise) and
/// [`EdgeCodecFlavor`] (how to read the 16-byte edge block).
///
/// It is NOT a new node property and NOT a SoA column — nothing is stored on the
/// row. This is the *resolution result* (the lens): the value-side analog of
/// "which XSD parses this document". §0 anti-invention — it bundles the two
/// read-mode enums that already exist, adding zero new fields to the node.
///
/// Both consumers and OGAR resolve `classid → ReadMode` through the one
/// [`LazyLock`] registry ([`classid_read_mode`]), so the LE interpretation of a
/// node's bytes is single-sourced: a consumer transcoding a [`NodeRow`] and OGAR
/// minting/projecting the same class read the identical schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadMode {
    /// Which value-slab tenants this class materialises.
    pub value_schema: ValueSchema,
    /// How this class reads its 16-byte edge block.
    pub edge_codec: EdgeCodecFlavor,
}

impl ReadMode {
    /// The zero-fallback / POC default an *unconfigured* classid resolves to.
    ///
    /// **TEMPORARY (2026-06-15 POC):** `value_schema = Full` mirrors the
    /// [`ClassView::value_schema`](crate::class_view::ClassView::value_schema)
    /// POC default so an unconfigured class materialises the whole slab for
    /// transcode; `edge_codec = CoarseOnly` is the canon zero-fallback edge
    /// reading. When the POC ends, flip `value_schema` back to
    /// [`ValueSchema::Bootstrap`] HERE and in `ClassView` together (one revert,
    /// two sites — the test `read_mode_default_is_full_poc` guards the pairing).
    pub const DEFAULT: ReadMode = ReadMode {
        value_schema: ValueSchema::Full,
        edge_codec: EdgeCodecFlavor::CoarseOnly,
    };

    /// The **OSINT / Palantir-Gotham** read-mode ([`NodeGuid::CLASSID_OSINT`]):
    /// a *hot* entity graph — [`ValueSchema::Cognitive`] (Meta + Qualia +
    /// Fingerprint + Energy + Plasticity + EntityType, for live NARS reasoning)
    /// over [`EdgeCodecFlavor::CoarseOnly`] adjacency (the 12 in-family + 4
    /// out-of-family slots read literally as the neo4j-emulation edges).
    pub const OSINT: ReadMode = ReadMode {
        value_schema: ValueSchema::Cognitive,
        edge_codec: EdgeCodecFlavor::CoarseOnly,
    };

    /// The **FMA anatomy** read-mode ([`NodeGuid::CLASSID_FMA`]): a *cold*
    /// structural reference graph — [`ValueSchema::Compressed`] (Fingerprint +
    /// Helix + Turbovec + EntityType; no hot lifecycle columns, it is static
    /// reference data) over [`EdgeCodecFlavor::CoarseOnly`] part-of adjacency.
    pub const FMA: ReadMode = ReadMode {
        value_schema: ValueSchema::Compressed,
        edge_codec: EdgeCodecFlavor::CoarseOnly,
    };

    /// Both axes are layout-preserving (a preset/flavor re-interprets reserved
    /// bytes, never a stride change), so adopting any read-mode needs no
    /// `ENVELOPE_LAYOUT_VERSION` bump.
    #[inline]
    pub const fn is_layout_preserving(self) -> bool {
        self.value_schema.is_layout_preserving() && self.edge_codec.is_layout_preserving()
    }
}

/// Builtin `classid → ReadMode` registry, built once on first use.
///
/// Immutable after init — the canon "already-immutable ontology registry" shape,
/// the same [`LazyLock`] pattern `lance-graph-ontology` uses for its seed
/// namespace registry. Holds only the canon builtins; a minted class's read-mode
/// is layered in by OGAR one level up. Any classid NOT in the map falls through
/// to [`ReadMode::DEFAULT`] — the same zero-fallback ladder as the key itself
/// (`classid 0 ⇒ default class`).
static BUILTIN_READ_MODES: LazyLock<HashMap<u32, ReadMode>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    // The canon default class materialises the POC-Full slab (see ReadMode::DEFAULT).
    m.insert(NodeGuid::CLASSID_DEFAULT, ReadMode::DEFAULT);
    // OSINT/Gotham (hot entity graph) + FMA anatomy (cold structural reference) —
    // the two registered graph domains (see `soa_graph`). Both read edges as
    // CoarseOnly adjacency; they differ in the value schema (hot vs cold).
    m.insert(NodeGuid::CLASSID_OSINT, ReadMode::OSINT);
    m.insert(NodeGuid::CLASSID_FMA, ReadMode::FMA);
    m
});

/// Resolve a `classid` to its [`ReadMode`] — the single source both consumers
/// and OGAR inherit. Reads the [`BUILTIN_READ_MODES`] registry, falling through
/// to [`ReadMode::DEFAULT`] for any unconfigured classid (the key's own
/// zero-fallback ladder). [`NodeGuid::read_mode`] is the carrier-method form.
#[inline]
pub fn classid_read_mode(classid: u32) -> ReadMode {
    BUILTIN_READ_MODES
        .get(&classid)
        .copied()
        .unwrap_or(ReadMode::DEFAULT)
}

/// Zero-copy [`SoaEnvelope`] wrapper over a contiguous slice of [`NodeRow`].
///
/// `NodeRow` is `#[repr(C, align(64))]` with the locked 16/16/480 byte
/// layout, so a `&[NodeRow]` IS already a row-strided LE packet at stride
/// 512 — no allocation, no copy. This wrapper just attaches the cycle stamp
/// and exposes the slice through the [`SoaEnvelope`] trait so Lance's
/// columnar I/O reads it directly.
///
/// The envelope's column table ([`NODE_ROW_COLUMNS`]) names the three
/// top-level slots (key / edges / value). Internal structure within each
/// slot is the canon's concern (`NodeGuid` for the key, `EdgeBlock` for the
/// edges, registry `ClassView` for the value carve-out).
#[derive(Clone, Copy)]
pub struct NodeRowPacket<'a> {
    rows: &'a [NodeRow],
    cycle: u32,
}

impl<'a> core::fmt::Debug for NodeRowPacket<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NodeRowPacket")
            .field("n_rows", &self.rows.len())
            .field("cycle", &self.cycle)
            .field("row_stride", &NODE_ROW_STRIDE)
            .finish()
    }
}

impl<'a> NodeRowPacket<'a> {
    /// Wrap a contiguous slice of [`NodeRow`] with a cycle stamp.
    #[inline]
    pub const fn new(rows: &'a [NodeRow], cycle: u32) -> Self {
        Self { rows, cycle }
    }

    /// The underlying rows.
    #[inline]
    pub const fn rows(&self) -> &'a [NodeRow] {
        self.rows
    }
}

impl<'a> SoaEnvelope for NodeRowPacket<'a> {
    fn columns(&self) -> &[ColumnDescriptor] {
        NODE_ROW_COLUMNS
    }
    fn row_stride(&self) -> usize {
        NODE_ROW_STRIDE
    }
    fn n_rows(&self) -> usize {
        self.rows.len()
    }
    fn cycle(&self) -> u32 {
        self.cycle
    }
    fn as_le_bytes(&self) -> &[u8] {
        // SAFETY: NodeRow is #[repr(C, align(64))] with size_of::<NodeRow>() ==
        // 512 (checked by the const _: () asserts above). A &[NodeRow] is a
        // contiguous array of #[repr(C)] structs; viewing it as &[u8] of
        // length len * 512 is a standard column-store packing operation, and
        // every byte position is valid for reads (no padding past size_of,
        // alignment of NodeRow (64) ⊇ alignment of u8 (1)).
        //
        // The NodeGuid and EdgeBlock fields hold their bytes in canon-LE
        // order (NodeGuid::new uses to_le_bytes; EdgeBlock is plain [u8;_]),
        // so the resulting byte slice IS the envelope's LE packet — no
        // translation needed at the boundary.
        unsafe {
            core::slice::from_raw_parts(
                self.rows.as_ptr().cast::<u8>(),
                self.rows.len() * NODE_ROW_STRIDE,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_zero_and_bootstrap() {
        let g = NodeGuid::local(0x00_00CD);
        assert_eq!(g.classid(), 0x0000_0000);
        assert_eq!(g.family(), 0x00_0000);
        assert!(g.is_default_class());
        assert!(g.is_unbasined());
        assert!(g.is_bootstrap_address());
    }

    #[test]
    fn nonzero_family_wakes_basin_binding() {
        let g = NodeGuid::new(0, 0, 0, 0, 0x00_00AB, 0x00_00CD);
        assert!(g.is_default_class());
        assert!(!g.is_unbasined()); // family != 0 ⇒ basin binding active
        assert!(!g.is_bootstrap_address());
    }

    #[test]
    fn family_identity_are_the_trailing_six_bytes() {
        let g = NodeGuid::new(0xDEAD_BEEF, 0x1111, 0x2222, 0x3333, 0x00_00AB, 0x00_00CD);
        assert_eq!(g.family(), 0x00_00AB);
        assert_eq!(g.identity(), 0x00_00CD);
        let lk = g.local_key();
        assert_eq!(lk & 0xFF_FFFF, 0x00_00AB);
        assert_eq!((lk >> 24) & 0xFF_FFFF, 0x00_00CD);
        assert_eq!(&g.as_bytes()[10..16], &[0xAB, 0x00, 0x00, 0xCD, 0x00, 0x00]);
    }

    #[test]
    fn edge_block_is_twelve_plus_four() {
        let e = EdgeBlock::default();
        assert_eq!(e.in_family.len(), 12);
        assert_eq!(e.out_family.len(), 4);
        assert_eq!(core::mem::size_of_val(&e), 16);
    }

    #[test]
    fn edge_codec_flavor_default_is_coarse_only() {
        // Zero-fallback default: the all-zero reading is the canon bootstrap.
        assert_eq!(EdgeCodecFlavor::default(), EdgeCodecFlavor::CoarseOnly);
        assert_eq!(EdgeCodecFlavor::CoarseOnly as u8, 0);
    }

    #[test]
    fn edge_codec_flavor_byte_costs() {
        // D = 128: coarse 1 B, residue 1 + 64 = 65 B, PQ fixed 16 B.
        assert_eq!(EdgeCodecFlavor::CoarseOnly.bytes_per_vector(128), 1);
        assert_eq!(EdgeCodecFlavor::CoarseResidue.bytes_per_vector(128), 65);
        assert_eq!(EdgeCodecFlavor::Pq32x4.bytes_per_vector(128), 16);
        // Odd D rounds the residue nibble count up.
        assert_eq!(EdgeCodecFlavor::CoarseResidue.bytes_per_vector(7), 1 + 4);
    }

    #[test]
    fn every_flavor_preserves_node_layout() {
        // The canon invariant: a flavor is an interpretation, never a stride
        // change — so no flavor forces an ENVELOPE_LAYOUT_VERSION bump.
        for f in [
            EdgeCodecFlavor::CoarseOnly,
            EdgeCodecFlavor::CoarseResidue,
            EdgeCodecFlavor::Pq32x4,
        ] {
            assert!(f.is_layout_preserving());
        }
        assert_eq!(NODE_ROW_STRIDE, core::mem::size_of::<NodeRow>());
    }

    #[test]
    fn uniqueness_guard_is_noop_outside_bootstrap() {
        // family != 0 ⇒ no longer the bootstrap address: the guard is a no-op
        // even when `already_present` is true.
        let g = NodeGuid::new(0, 0, 0, 0, 0x00_0001, 0x00_0001);
        g.debug_assert_identity_unique(true);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "identity collision in default basin")]
    fn uniqueness_guard_panics_on_bootstrap_collision() {
        let g = NodeGuid::local(1);
        g.debug_assert_identity_unique(true);
    }

    #[test]
    #[should_panic(expected = "family must fit in 24 bits")]
    fn new_panics_on_family_overflow() {
        let _ = NodeGuid::new(0, 0, 0, 0, 0x0100_0000, 0);
    }

    #[test]
    #[should_panic(expected = "identity must fit in 24 bits")]
    fn new_panics_on_identity_overflow() {
        let _ = NodeGuid::new(0, 0, 0, 0, 0, 0x0100_0000);
    }

    #[test]
    fn display_is_canonical_self_describing() {
        // Canon (OGAR P0): xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (8-4-4-4-12 hex);
        // groups = classid · HEEL · HIP · TWIG · family·identity.
        let g = NodeGuid::new(0xDEAD_BEEF, 0x1111, 0x2222, 0x3333, 0x00_00AB, 0x00_00CD);
        let s = g.to_string();
        assert_eq!(s, "deadbeef-1111-2222-3333-0000ab0000cd");
        assert_eq!(s.len(), 36, "8-4-4-4-12 + 4 hyphens");
        for i in [8usize, 13, 18, 23] {
            assert_eq!(s.as_bytes()[i], b'-', "hyphen at {i}");
        }
    }

    #[test]
    fn display_zero_default_is_all_zeros() {
        // Zero-fallback ladder visible at sight: classid + family == 0 prints
        // as ...0...-...0... with identity-only discrimination.
        let g = NodeGuid::local(0x00_00CD);
        assert_eq!(g.to_string(), "00000000-0000-0000-0000-0000000000cd");
    }

    // ── SoaEnvelope binding for NodeRowPacket ────────────────────────────────

    fn sample_row(classid: u32, identity: u32) -> NodeRow {
        NodeRow {
            key: NodeGuid::new(classid, 0x1111, 0x2222, 0x3333, 0x00_00AB, identity),
            edges: EdgeBlock::default(),
            value: [0u8; 480],
        }
    }

    #[test]
    fn node_row_column_table_sums_to_row_stride() {
        let total: usize = NODE_ROW_COLUMNS.iter().map(|c| c.col_bytes_per_row()).sum();
        assert_eq!(total, NODE_ROW_STRIDE);
        assert_eq!(NODE_ROW_STRIDE, core::mem::size_of::<NodeRow>());
    }

    #[test]
    fn node_row_column_table_is_in_offset_order_without_gaps() {
        // The contract: columns are contiguous (key 0..16, edges 16..32,
        // value 32..512) — no gaps, no overlap, in offset order.
        let mut prev_end = 0usize;
        for c in NODE_ROW_COLUMNS {
            assert_eq!(c.row_offset as usize, prev_end, "no gap before {c:?}");
            prev_end = c.row_offset as usize + c.col_bytes_per_row();
        }
        assert_eq!(prev_end, NODE_ROW_STRIDE);
    }

    #[test]
    fn empty_packet_verifies() {
        let rows: &[NodeRow] = &[];
        let pkt = NodeRowPacket::new(rows, 0);
        assert_eq!(pkt.n_rows(), 0);
        assert_eq!(pkt.as_le_bytes().len(), 0);
        assert!(pkt.verify_layout().is_ok(), "empty packet must verify");
    }

    #[test]
    fn single_row_packet_verifies_and_byte_view_is_zero_copy() {
        let rows = [sample_row(0xDEAD_BEEF, 0x00_00CD)];
        let pkt = NodeRowPacket::new(&rows, 7);
        assert_eq!(pkt.n_rows(), 1);
        assert_eq!(pkt.cycle(), 7);
        assert_eq!(pkt.row_stride(), 512);
        assert_eq!(pkt.as_le_bytes().len(), 512);
        // Zero-copy: the byte view's pointer is the slice's pointer.
        assert_eq!(
            pkt.as_le_bytes().as_ptr() as usize,
            rows.as_ptr() as usize,
            "as_le_bytes must be zero-copy"
        );
        assert!(pkt.verify_layout().is_ok());
    }

    #[test]
    fn multi_row_packet_byte_length_is_stride_times_rows() {
        let rows = [
            sample_row(0xDEAD_BEEF, 0x00_00CD),
            sample_row(0xCAFE_BABE, 0x00_0001),
            sample_row(0x0000_0000, 0x00_0042),
        ];
        let pkt = NodeRowPacket::new(&rows, 42);
        assert_eq!(pkt.n_rows(), 3);
        assert_eq!(pkt.as_le_bytes().len(), 3 * 512);
        assert!(pkt.verify_layout().is_ok());
    }

    #[test]
    fn row_le_view_returns_one_full_row() {
        let rows = [sample_row(1, 2), sample_row(3, 4), sample_row(5, 6)];
        let pkt = NodeRowPacket::new(&rows, 0);
        for (i, row) in rows.iter().enumerate() {
            let row_bytes = pkt.row_le(i).expect("row in range");
            assert_eq!(row_bytes.len(), 512);
            // First 4 bytes are the classid in canon-LE order.
            assert_eq!(
                u32::from_le_bytes(row_bytes[..4].try_into().unwrap()),
                row.key.classid()
            );
        }
        assert!(pkt.row_le(3).is_none(), "out of range");
    }

    #[test]
    fn column_le_view_returns_the_named_slot() {
        // Place a recognisable byte pattern in the value side; verify the
        // value column-view picks it up at the right offset.
        let mut row = sample_row(0xDEAD_BEEF, 0x00_00CD);
        row.value[0] = 0xAB;
        row.value[479] = 0xCD;
        let rows = [row];
        let pkt = NodeRowPacket::new(&rows, 0);
        let value_col = pkt
            .column_le(0, &NODE_ROW_COLUMNS[NodeRowColumn::Value as usize])
            .expect("value column in range");
        assert_eq!(value_col.len(), 480);
        assert_eq!(value_col[0], 0xAB);
        assert_eq!(value_col[479], 0xCD);
        // Key column is at offset 0, length 16 — first byte = LE byte 0 of
        // classid = 0xEF (low byte of 0xDEAD_BEEF).
        let key_col = pkt
            .column_le(0, &NODE_ROW_COLUMNS[NodeRowColumn::Key as usize])
            .expect("key column in range");
        assert_eq!(key_col.len(), 16);
        assert_eq!(key_col[0], 0xEF);
        assert_eq!(key_col[3], 0xDE);
    }

    #[test]
    fn key_bytes_in_canon_le_order() {
        // Round-trip: pack a NodeRow with known fields, read the bytes back
        // through the envelope, parse each canon group by its LE byte range,
        // confirm values match. Proves the SoA envelope view stays canon-LE
        // end-to-end without any field-accessor intermediation.
        let row = sample_row(0xDEAD_BEEF, 0x00_00CD);
        let rows = [row];
        let pkt = NodeRowPacket::new(&rows, 0);
        let bytes = pkt.as_le_bytes();
        // Per OGAR/CLAUDE.md P0: classid · HEEL · HIP · TWIG · family · identity.
        assert_eq!(
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            0xDEAD_BEEF,
            "classid at [0..4]"
        );
        assert_eq!(
            u16::from_le_bytes([bytes[4], bytes[5]]),
            0x1111,
            "HEEL at [4..6]"
        );
        assert_eq!(
            u16::from_le_bytes([bytes[6], bytes[7]]),
            0x2222,
            "HIP at [6..8]"
        );
        assert_eq!(
            u16::from_le_bytes([bytes[8], bytes[9]]),
            0x3333,
            "TWIG at [8..10]"
        );
        // family is u24 LE in bytes [10..13]: 0xAB, 0x00, 0x00.
        assert_eq!(&bytes[10..13], &[0xAB, 0x00, 0x00], "family at [10..13]");
        // identity is u24 LE in bytes [13..16]: 0xCD, 0x00, 0x00.
        assert_eq!(&bytes[13..16], &[0xCD, 0x00, 0x00], "identity at [13..16]");
    }

    #[test]
    fn envelope_layout_version_matches_envelope_default() {
        // The wrapper does not override LAYOUT_VERSION, so verify_layout
        // checks against the envelope-crate default (ENVELOPE_LAYOUT_VERSION).
        let rows = [sample_row(0, 1)];
        let pkt = NodeRowPacket::new(&rows, 0);
        assert_eq!(
            <NodeRowPacket<'_> as SoaEnvelope>::LAYOUT_VERSION,
            crate::soa_envelope::ENVELOPE_LAYOUT_VERSION
        );
        // verify_layout exercises that gate.
        assert!(pkt.verify_layout().is_ok());
    }

    // ── Value-slab schema presets ────────────────────────────────────────────

    #[test]
    fn value_tenants_contiguous_within_slab() {
        let mut prev_end = VALUE_SLAB_ROW_OFFSET;
        for (i, c) in VALUE_TENANTS.iter().enumerate() {
            assert_eq!(c.name_id as usize, i, "discriminant == index");
            assert_eq!(c.row_offset as usize, prev_end, "no gap before {c:?}");
            prev_end = c.row_offset as usize + c.col_bytes_per_row();
        }
        assert!(prev_end <= NODE_ROW_STRIDE);
        assert_eq!(
            prev_end - VALUE_SLAB_ROW_OFFSET,
            112,
            "current Full carve uses 112 of 480 B (helix right-sized 48→6)"
        );
        assert!(prev_end - VALUE_SLAB_ROW_OFFSET <= VALUE_SLAB_LEN);
    }

    #[test]
    fn value_schema_default_is_bootstrap_empty() {
        assert_eq!(ValueSchema::default(), ValueSchema::Bootstrap);
        assert!(ValueSchema::Bootstrap.field_mask().is_empty());
        assert_eq!(ValueSchema::Bootstrap.tenant_bytes(), 0);
    }

    #[test]
    fn value_schema_full_covers_every_tenant() {
        let full = ValueSchema::Full;
        assert_eq!(full.field_mask().count() as usize, VALUE_TENANTS.len());
        for t in [
            ValueTenant::Meta,
            ValueTenant::Qualia,
            ValueTenant::MaterializedEdges,
            ValueTenant::Fingerprint,
            ValueTenant::HelixResidue,
            ValueTenant::TurbovecResidue,
            ValueTenant::Energy,
            ValueTenant::Plasticity,
            ValueTenant::EntityType,
        ] {
            assert!(full.has(t), "Full must materialise {t:?}");
        }
    }

    #[test]
    fn value_schema_byte_budgets_are_locked() {
        assert_eq!(ValueSchema::Bootstrap.tenant_bytes(), 0);
        assert_eq!(ValueSchema::Cognitive.tenant_bytes(), 58);
        assert_eq!(ValueSchema::Compressed.tenant_bytes(), 56);
        assert_eq!(ValueSchema::Full.tenant_bytes(), 112);
        for s in [
            ValueSchema::Bootstrap,
            ValueSchema::Cognitive,
            ValueSchema::Compressed,
            ValueSchema::Full,
        ] {
            assert!(s.tenant_bytes() <= VALUE_SLAB_LEN);
            assert!(s.is_layout_preserving());
        }
    }

    #[test]
    fn value_schema_presets_carry_expected_tenants() {
        // Cognitive: hot columns, no codec residues, no materialised edges.
        let c = ValueSchema::Cognitive;
        assert!(c.has(ValueTenant::Meta) && c.has(ValueTenant::Qualia));
        assert!(c.has(ValueTenant::Energy) && c.has(ValueTenant::EntityType));
        assert!(!c.has(ValueTenant::HelixResidue));
        assert!(!c.has(ValueTenant::TurbovecResidue));
        assert!(!c.has(ValueTenant::MaterializedEdges));
        // Compressed: codec residues, no hot lifecycle.
        let z = ValueSchema::Compressed;
        assert!(z.has(ValueTenant::HelixResidue) && z.has(ValueTenant::TurbovecResidue));
        assert!(z.has(ValueTenant::Fingerprint));
        assert!(!z.has(ValueTenant::Energy) && !z.has(ValueTenant::Meta));
    }

    #[test]
    fn value_schema_preserves_node_stride() {
        // A preset is an interpretation of the reserved value slab, never a
        // stride change — same canon invariant as EdgeCodecFlavor.
        assert_eq!(NODE_ROW_STRIDE, core::mem::size_of::<NodeRow>());
        assert_eq!(VALUE_SLAB_ROW_OFFSET + VALUE_SLAB_LEN, NODE_ROW_STRIDE);
    }

    // ── GUID decode + classid → read-mode (the keystone) ─────────────────────

    #[test]
    fn decode_returns_all_six_canon_groups() {
        // One read yields classid + HHT (HEEL/HIP/TWIG) + family + identity, in
        // canon print order — the "read the GUID as a GUID" contract.
        let g = NodeGuid::new(0xDEAD_BEEF, 0x1111, 0x2222, 0x3333, 0x00_00AB, 0x00_00CD);
        let p = g.decode();
        assert_eq!(p.classid, 0xDEAD_BEEF);
        assert_eq!(p.heel, 0x1111);
        assert_eq!(p.hip, 0x2222);
        assert_eq!(p.twig, 0x3333);
        assert_eq!(p.family, 0x00_00AB);
        assert_eq!(p.identity, 0x00_00CD);
        // decode() is exactly the field accessors, no field invented/dropped.
        assert_eq!(p.classid, g.classid());
        assert_eq!(p.family, g.family());
        assert_eq!(p.identity, g.identity());
    }

    #[test]
    fn hht_accessors_match_display_groups() {
        // The new HEEL/HIP/TWIG accessors fold the same LE bytes Display renders.
        let g = NodeGuid::new(0xDEAD_BEEF, 0xA1B2, 0xC3D4, 0xE5F6, 0x12_3456, 0x78_9ABC);
        assert_eq!(g.heel(), 0xA1B2);
        assert_eq!(g.hip(), 0xC3D4);
        assert_eq!(g.twig(), 0xE5F6);
        // Display's middle three groups are exactly heel-hip-twig in hex.
        let s = g.to_string();
        let groups: Vec<&str> = s.split('-').collect();
        assert_eq!(groups[1], format!("{:04x}", g.heel()));
        assert_eq!(groups[2], format!("{:04x}", g.hip()));
        assert_eq!(groups[3], format!("{:04x}", g.twig()));
    }

    #[test]
    fn read_mode_default_is_full_poc() {
        // The default classid resolves to the POC read-mode: Full value slab +
        // CoarseOnly edges. This GUARDS the ClassView pairing — ReadMode::DEFAULT
        // .value_schema MUST equal the ClassView POC default (Full). When the POC
        // ends, both flip to Bootstrap together and this test flips with them.
        let rm = classid_read_mode(NodeGuid::CLASSID_DEFAULT);
        assert_eq!(rm, ReadMode::DEFAULT);
        assert_eq!(rm.value_schema, ValueSchema::Full);
        assert_eq!(rm.edge_codec, EdgeCodecFlavor::CoarseOnly);
        assert!(rm.is_layout_preserving());
    }

    #[test]
    fn read_mode_zero_fallback_for_unconfigured_classid() {
        // Any classid NOT in the builtin registry falls through to DEFAULT — the
        // key's own zero-fallback ladder (classid 0 ⇒ default class), extended to
        // read-mode resolution.
        assert_eq!(classid_read_mode(0xDEAD_BEEF), ReadMode::DEFAULT);
        assert_eq!(classid_read_mode(0x0000_0001), ReadMode::DEFAULT);
        assert_eq!(classid_read_mode(u32::MAX), ReadMode::DEFAULT);
    }

    #[test]
    fn guid_read_mode_method_delegates_to_registry() {
        // The carrier method (guid.read_mode()) and the free resolver
        // (classid_read_mode(classid)) are the SAME answer — consumer and OGAR
        // inherit one source.
        let g = NodeGuid::new(0xCAFE_BABE, 1, 2, 3, 0x00_0001, 0x00_0002);
        assert_eq!(g.read_mode(), classid_read_mode(g.classid()));
        // A default-class node reads the Full POC slab.
        assert_eq!(NodeGuid::local(0x00_00CD).read_mode(), ReadMode::DEFAULT);
    }

    #[test]
    fn default_class_node_materialises_full_slab() {
        // End-to-end connect: a bootstrap NodeRow → its classid resolves to Full →
        // the Full preset covers every tenant and uses the locked 112-byte carve.
        let row = sample_row(NodeGuid::CLASSID_DEFAULT, 0x00_00CD);
        let rm = row.key.read_mode();
        assert_eq!(rm.value_schema, ValueSchema::Full);
        assert_eq!(
            rm.value_schema.field_mask().count() as usize,
            VALUE_TENANTS.len(),
            "Full read-mode materialises every value tenant"
        );
        assert_eq!(rm.value_schema.tenant_bytes(), 112);
        // The slab has room (112 ≤ 480) and the choice never grows the stride.
        assert!(rm.value_schema.tenant_bytes() <= VALUE_SLAB_LEN);
        assert!(rm.is_layout_preserving());
    }

    #[test]
    fn osint_and_fma_classids_resolve_to_their_read_modes() {
        // The two registered graph domains (see `soa_graph`): OSINT/Gotham is a
        // hot entity graph (Cognitive value), FMA anatomy is a cold structural
        // reference (Compressed value); both read edges as CoarseOnly adjacency.
        let osint = classid_read_mode(NodeGuid::CLASSID_OSINT);
        assert_eq!(osint, ReadMode::OSINT);
        assert_eq!(osint.value_schema, ValueSchema::Cognitive);
        assert_eq!(osint.edge_codec, EdgeCodecFlavor::CoarseOnly);

        let fma = classid_read_mode(NodeGuid::CLASSID_FMA);
        assert_eq!(fma, ReadMode::FMA);
        assert_eq!(fma.value_schema, ValueSchema::Compressed);
        assert_eq!(fma.edge_codec, EdgeCodecFlavor::CoarseOnly);

        // The classids are the OGAR-confirmed 0x0007 (OSINT) and 0x0008 (FMA);
        // both are layout-preserving and carrier-method-consistent.
        assert_eq!(NodeGuid::CLASSID_OSINT, 0x0000_0007);
        assert_eq!(NodeGuid::CLASSID_FMA, 0x0000_0008);
        assert_eq!(
            NodeGuid::new(NodeGuid::CLASSID_OSINT, 1, 2, 3, 0xAB, 0xCD).read_mode(),
            ReadMode::OSINT
        );
        assert!(osint.is_layout_preserving() && fma.is_layout_preserving());
    }

    // ── GUID v2 tail (D-GV2-1) — field-isolation matrix + coexistence ─────────

    #[cfg(feature = "guid-v2-tail")]
    #[test]
    fn v2_field_isolation_matrix() {
        // Each tier carries a distinct value; every accessor reads back exactly
        // its own, and varying ONE tier changes ONLY that accessor (the
        // mandatory layout-bit-boundary test for a reclaim, I-LEGACY).
        let base = NodeGuid::new_v2(0x1111_2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777, 0x8888);
        assert_eq!(base.classid(), 0x1111_2222);
        assert_eq!(base.heel(), 0x3333);
        assert_eq!(base.hip(), 0x4444);
        assert_eq!(base.twig(), 0x5555);
        assert_eq!(base.leaf(), 0x6666);
        assert_eq!(base.family_v2(), 0x7777);
        assert_eq!(base.identity_v2(), 0x8888);

        // vary ONLY leaf
        let l = NodeGuid::new_v2(0x1111_2222, 0x3333, 0x4444, 0x5555, 0xAAAA, 0x7777, 0x8888);
        assert_eq!(l.leaf(), 0xAAAA);
        assert_eq!(l.family_v2(), base.family_v2());
        assert_eq!(l.identity_v2(), base.identity_v2());
        assert_eq!(l.twig(), base.twig());
        // vary ONLY family
        let f = NodeGuid::new_v2(0x1111_2222, 0x3333, 0x4444, 0x5555, 0x6666, 0xBBBB, 0x8888);
        assert_eq!(f.family_v2(), 0xBBBB);
        assert_eq!(f.leaf(), base.leaf());
        assert_eq!(f.identity_v2(), base.identity_v2());
        // vary ONLY identity
        let i = NodeGuid::new_v2(0x1111_2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777, 0xCCCC);
        assert_eq!(i.identity_v2(), 0xCCCC);
        assert_eq!(i.leaf(), base.leaf());
        assert_eq!(i.family_v2(), base.family_v2());

        // local_key_v2 = family ++ identity (LE)
        assert_eq!(base.local_key_v2(), 0x8888_7777);
        // decode_v2 round-trips the tail
        let d = base.decode_v2();
        assert_eq!((d.leaf, d.family, d.identity), (0x6666, 0x7777, 0x8888));
        // Display is uniform 4-hex groups (classid 8).
        assert_eq!(base.to_hex_v2(), "11112222-3333-4444-5555-6666-7777-8888");
    }

    #[cfg(feature = "guid-v2-tail")]
    #[test]
    fn v1_and_v2_share_prefix_differ_in_tail() {
        // v1 and v2 agree on the prefix (classid·HEEL·HIP·TWIG)…
        let v1 = NodeGuid::new(0xDEAD_BEEF, 0x1111, 0x2222, 0x3333, 0x00_00AB, 0x00_00CD);
        let v2 = NodeGuid::new_v2(0xDEAD_BEEF, 0x1111, 0x2222, 0x3333, 0, 0xABCD, 0);
        assert_eq!(v1.classid(), v2.classid());
        assert_eq!(v1.heel(), v2.heel());
        assert_eq!(v1.hip(), v2.hip());
        assert_eq!(v1.twig(), v2.twig());
        // …but the tail bytes are interpreted differently — which is exactly why
        // the version gate is mandatory before reading a tail.
        assert_eq!(GUID_TAIL_LAYOUT_VERSION_V2, 2);
        // v1 accessors remain UNTOUCHED under the feature (additive, non-breaking).
        assert_eq!(v1.family(), 0x00_00AB);
        assert_eq!(v1.identity(), 0x00_00CD);
    }
}
