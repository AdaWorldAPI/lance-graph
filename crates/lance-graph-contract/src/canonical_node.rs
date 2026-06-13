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

    /// Construct from the six canonical groups. `family`/`identity` use their low 3 bytes.
    pub const fn new(classid: u32, heel: u16, hip: u16, twig: u16, family: u32, identity: u32) -> Self {
        let c = classid.to_le_bytes();
        let h = heel.to_le_bytes();
        let p = hip.to_le_bytes();
        let t = twig.to_le_bytes();
        let f = family.to_le_bytes();   // low 3 bytes
        let i = identity.to_le_bytes(); // low 3 bytes
        Self([
            c[0], c[1], c[2], c[3], //  0..4  classid
            h[0], h[1],             //  4..6  HEEL
            p[0], p[1],             //  6..8  HIP
            t[0], t[1],             //  8..10 TWIG
            f[0], f[1], f[2],       // 10..13 family
            i[0], i[1], i[2],       // 13..16 identity
        ])
    }

    /// Default-class, default-basin node: only `identity` discriminates.
    /// This is the bootstrap address while classid and family are zero.
    pub const fn local(identity: u32) -> Self {
        Self::new(Self::CLASSID_DEFAULT, 0, 0, 0, Self::FAMILY_DEFAULT, identity)
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

/// Mint-path guard: while in the default basin, `identity` (24 bits) is the ONLY
/// discriminator, so the mint path MUST guarantee its uniqueness. Call on insert.
/// `seen` is whatever set/bitmap the mint path keeps; this just centralises the
/// invariant so it can't be forgotten when family is still a no-op.
#[inline]
pub fn debug_assert_identity_unique(guid: &NodeGuid, already_present: bool) {
    if guid.is_bootstrap_address() {
        debug_assert!(
            !already_present,
            "identity collision in default basin: 24-bit identity space exhausted \
             or reused — mint a non-zero family to expand before this fires in prod"
        );
    }
}

// Sizes are part of the lock.
const _: () = assert!(core::mem::size_of::<NodeGuid>() == 16);
const _: () = assert!(core::mem::size_of::<EdgeBlock>() == 16);
const _: () = assert!(core::mem::size_of::<NodeRow>() == 512);

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
}
