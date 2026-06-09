//! # `identity` — the structured 128-bit node identity (UUIDv8).
//!
//! A [`NodeGuid`] is the HHTL nibble-address **formalized + namespaced** into a
//! standards-compliant UUIDv8 (RFC 9562). It is the workspace's first *stable
//! binary instance identity*: the cold path keys nodes by `node_id:u32` today,
//! the SPO hot path by a `u64` content `dn_hash` — neither is a stable,
//! globally-referenceable id. `NodeGuid` fills that gap.
//!
//! ## Compose, don't re-invent (the iron mandate)
//!
//! Every field is an existing committed scalar; `NodeGuid` is their composition,
//! never a parallel re-pack (which would duplicate the ratified `OD-CLASSID-WIDTH`
//! discriminator + `I-VSA-IDENTITIES`):
//! - `namespace:u8 | entity_type:u16 | kind:u8` = the `SchemaPtr.packed` u32
//!   convention (`lance-graph-ontology::namespace`).
//! - `niblepath_prefix` = a truncated [`NiblePath`](crate::hhtl::NiblePath) (the
//!   HHTL tree address; full depth resolves from `entity_type`).
//! - `shape_hash` = a truncated `StructuralSignature` (the D-CLS shape witness).
//! - `local` = an instance index.
//!
//! ## Eineindeutigkeit — `entity_type` canonical, `NiblePath` the derived view
//!
//! `entity_type:u16` is the **exact, bijective** class identity (fixed-width, no
//! truncation; the registry mints it 1:1 with the class's `NiblePath`). The
//! `niblepath_prefix` in the GUID is a *derived routing cache* — coarse (≤4
//! nibbles), equal to `niblepath_of(entity_type)` truncated to the prefix. A
//! *truncated* NiblePath CANNOT be the bijective identity (two deep classes
//! collide past the prefix — see `hhtl.rs`); so the exact identity is the dense
//! `entity_type`, and the prefix only accelerates `is_ancestor_of` routing.
//!
//! ## The five readings (register reads of one immutable key)
//!
//! - **resolve** `entity_type()` → `ClassView` (class-from-address, O(1)).
//! - **route**   `niblepath()` → delegate switch (HHTL bit-shift).
//! - **witness** the frozen 16 bytes + the merkle chain (immutable, in place).
//! - **ground-truth** `shape_hash()` vs `resolve(addr).shape_hash_now` → drift.
//! - **dispatch-to-store** `as_bytes()` → `EntityKey` → consumer store.
//!
//! ## Immutability law
//!
//! Write-once. `entity_type` never updates (the lineage id — re-resolved from the
//! address for free). Drift *repair* is a new immutable version (Lance is
//! versioned), never an in-place mutation. `I-VSA-IDENTITIES` Test 0: a register
//! key that POINTS TO content, never VSA-bundled.

use crate::hhtl::NiblePath;

/// Layout version of [`NodeGuid`]'s byte geometry. Stamped into the GUID so a
/// future reader refuses to decode a layout it does not understand
/// (`I-LEGACY-API-FEATURE-GATED`). Bump on any field-bit change.
pub const IDENTITY_LAYOUT_VERSION: u8 = 1;

/// RFC 9562 UUID version nibble — `8` = custom / application-defined layout.
const UUID_VERSION: u8 = 8;
/// RFC 9562 variant bits — `0b10`.
const UUID_VARIANT: u8 = 0b10;

/// Bit-width of the `shape_hash` field (truncated `StructuralSignature`).
pub const SHAPE_HASH_BITS: u32 = 22;
/// Bit-width of the `local` instance-index field.
pub const LOCAL_BITS: u32 = 24;

const SHAPE_HASH_MASK: u32 = (1u32 << SHAPE_HASH_BITS) - 1;
const LOCAL_MASK: u32 = (1u32 << LOCAL_BITS) - 1;

/// A 128-bit immutable structured node identity (UUIDv8).
///
/// Octet layout (canonical UUID byte order, octets 0-15):
/// ```text
///   0      : namespace        (u8)            ─┐
///   1..=2  : entity_type      (u16 BE)         ├ SchemaPtr.packed u32 convention
///   3      : kind             (u8)            ─┘
///   4..=5  : niblepath_prefix (u16 BE, ≤4 nibbles routing cache)
///   6      : [hi nibble: UUID version=8][lo nibble: niblepath_depth 0..=4]
///   7      : shape_hash[21:14]
///   8      : [hi 2 bits: UUID variant=0b10][lo 6 bits: shape_hash[13:8]]
///   9      : shape_hash[7:0]                  (shape_hash = 22 bits total)
///   10..=12: local            (u24 BE)
///   13     : layout_version   (u8)
///   14..=15: spare            (zero)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C, align(16))]
pub struct NodeGuid([u8; 16]);

impl NodeGuid {
    /// Routing-cache prefix depth carried in the GUID (the full path resolves
    /// from `entity_type`; this is the coarse, branchless-routing prefix).
    pub const PREFIX_NIBBLES: u8 = 4;

    /// Compose a `NodeGuid` from its existing-field parts. `entity_type` is exact
    /// (the bijective identity); `niblepath` is truncated to the routing prefix;
    /// `shape_hash`/`local` are masked to their field widths. Sets the UUIDv8
    /// version/variant + the layout-version stamp.
    pub fn new(
        namespace: u8,
        entity_type: u16,
        kind: u8,
        niblepath: NiblePath,
        shape_hash: u32,
        local: u32,
    ) -> Self {
        let (prefix, depth) = Self::truncate_prefix(niblepath);
        let sh = shape_hash & SHAPE_HASH_MASK;
        let loc = local & LOCAL_MASK;
        let et = entity_type.to_be_bytes();
        let np = prefix.to_be_bytes();
        Self([
            namespace,
            et[0],
            et[1],
            kind,
            np[0],
            np[1],
            (UUID_VERSION << 4) | (depth & 0x0F),
            ((sh >> 14) & 0xFF) as u8,
            (UUID_VARIANT << 6) | ((sh >> 8) & 0x3F) as u8,
            (sh & 0xFF) as u8,
            ((loc >> 16) & 0xFF) as u8,
            ((loc >> 8) & 0xFF) as u8,
            (loc & 0xFF) as u8,
            IDENTITY_LAYOUT_VERSION,
            0,
            0,
        ])
    }

    /// Keep the top `PREFIX_NIBBLES` nibbles of a full path (root-first, so we
    /// drop the lowest `depth - keep` nibbles). The result `is_ancestor_of` the
    /// full path — a true coarse routing prefix.
    fn truncate_prefix(path: NiblePath) -> (u16, u8) {
        let (full, depth) = path.packed();
        let keep = if depth < Self::PREFIX_NIBBLES {
            depth
        } else {
            Self::PREFIX_NIBBLES
        };
        let drop = depth - keep;
        let prefix = (full >> (4 * drop as u32)) as u16;
        (prefix, keep)
    }

    // ── accessors (the five readings start here) ──

    /// The raw 16 bytes — feed straight to `EntityKey(&[u8])` (dispatch-to-store).
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Reconstruct from raw bytes (e.g. read back from a Lance column / EntityKey).
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// OGIT namespace ordinal (`NamespaceId`).
    #[must_use]
    pub const fn namespace(&self) -> u8 {
        self.0[0]
    }

    /// The **exact, bijective** class identity (`entity_type_id` / `class_id`).
    /// No truncation — this is the canonical class lineage id (resolve here).
    #[must_use]
    pub const fn entity_type(&self) -> u16 {
        u16::from_be_bytes([self.0[1], self.0[2]])
    }

    /// `SchemaPtr` kind discriminator.
    #[must_use]
    pub const fn kind(&self) -> u8 {
        self.0[3]
    }

    /// Raw routing-prefix bits (≤4 nibbles). Prefer [`niblepath`](Self::niblepath).
    #[must_use]
    pub const fn niblepath_prefix(&self) -> u16 {
        u16::from_be_bytes([self.0[4], self.0[5]])
    }

    /// Depth (nibble count) of the stored routing prefix (0..=4).
    #[must_use]
    pub const fn niblepath_depth(&self) -> u8 {
        self.0[6] & 0x0F
    }

    /// The routing prefix as a (truncated) [`NiblePath`] — the coarse delegate
    /// switch. The EXACT class is [`entity_type`](Self::entity_type); this only
    /// accelerates `is_ancestor_of`.
    #[must_use]
    pub fn niblepath(&self) -> Option<NiblePath> {
        NiblePath::from_packed(u64::from(self.niblepath_prefix()), self.niblepath_depth())
    }

    /// The shape_hash drift witness (truncated `StructuralSignature`, 22 bits).
    #[must_use]
    pub const fn shape_hash(&self) -> u32 {
        ((self.0[7] as u32) << 14) | (((self.0[8] & 0x3F) as u32) << 8) | (self.0[9] as u32)
    }

    /// The instance index (24 bits).
    #[must_use]
    pub const fn local(&self) -> u32 {
        ((self.0[10] as u32) << 16) | ((self.0[11] as u32) << 8) | (self.0[12] as u32)
    }

    /// The stamped [`IDENTITY_LAYOUT_VERSION`].
    #[must_use]
    pub const fn layout_version(&self) -> u8 {
        self.0[13]
    }

    /// RFC 9562 version nibble (must be `8`).
    #[must_use]
    pub const fn version(&self) -> u8 {
        self.0[6] >> 4
    }

    /// RFC 9562 variant bits (must be `0b10`).
    #[must_use]
    pub const fn variant(&self) -> u8 {
        self.0[8] >> 6
    }

    /// Is this a well-formed UUIDv8 (version `8`, variant `0b10`)? The
    /// self-explanatory-to-any-consumer property holds only when this is true.
    #[must_use]
    pub const fn is_valid_uuid_v8(&self) -> bool {
        self.version() == UUID_VERSION && self.variant() == UUID_VARIANT
    }
}

impl core::fmt::Display for NodeGuid {
    /// Canonical UUID string `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let b = &self.0;
        write!(
            f,
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deep example path: basin 0x2 → 0x5 → 0xA → 0x3 → 0x7 (depth 5, deeper
    /// than the 4-nibble GUID prefix).
    fn deep_path() -> NiblePath {
        NiblePath::root(0x2)
            .child(0x5)
            .child(0xA)
            .child(0x3)
            .child(0x7)
    }

    fn sample() -> NodeGuid {
        NodeGuid::new(0x07, 0xABCD, 0x03, deep_path(), 0x2F_1A3, 0x12_3456)
    }

    #[test]
    fn fields_round_trip() {
        let g = sample();
        assert_eq!(g.namespace(), 0x07);
        assert_eq!(
            g.entity_type(),
            0xABCD,
            "entity_type is EXACT (no truncation)"
        );
        assert_eq!(g.kind(), 0x03);
        assert_eq!(g.shape_hash(), 0x2F_1A3 & SHAPE_HASH_MASK);
        assert_eq!(g.local(), 0x12_3456 & LOCAL_MASK);
        assert_eq!(g.layout_version(), IDENTITY_LAYOUT_VERSION);
    }

    #[test]
    fn uuid_v8_version_and_variant_are_reserved() {
        let g = sample();
        assert_eq!(g.version(), 8, "UUIDv8");
        assert_eq!(g.variant(), 0b10, "RFC 9562 variant");
        assert!(g.is_valid_uuid_v8());
        // The reserved bits sit in octets 6 (hi nibble) and 8 (hi 2 bits).
        let b = g.as_bytes();
        assert_eq!(b[6] >> 4, 8);
        assert_eq!(b[8] >> 6, 0b10);
    }

    #[test]
    fn shape_hash_and_local_saturate_to_field_width() {
        // Over-wide inputs are masked, never overflow into neighbours.
        let g = NodeGuid::new(0, 0, 0, NiblePath::EMPTY, u32::MAX, u32::MAX);
        assert_eq!(g.shape_hash(), SHAPE_HASH_MASK);
        assert_eq!(g.local(), LOCAL_MASK);
        // The version/variant survived the over-wide shape_hash.
        assert!(g.is_valid_uuid_v8());
    }

    #[test]
    fn field_isolation_each_field_is_independent() {
        // The I-LEGACY-API field-isolation matrix: vary one field, assert all
        // OTHERS are unchanged (and the UUIDv8 reserved bits never move).
        let base = sample();
        let probes = [
            NodeGuid::new(0xFF, 0xABCD, 0x03, deep_path(), 0x2F_1A3, 0x12_3456), // namespace
            NodeGuid::new(0x07, 0x0000, 0x03, deep_path(), 0x2F_1A3, 0x12_3456), // entity_type
            NodeGuid::new(0x07, 0xABCD, 0xFF, deep_path(), 0x2F_1A3, 0x12_3456), // kind
            NodeGuid::new(0x07, 0xABCD, 0x03, deep_path(), 0x00000, 0x12_3456),  // shape_hash
            NodeGuid::new(0x07, 0xABCD, 0x03, deep_path(), 0x2F_1A3, 0x000000),  // local
        ];
        for p in probes {
            assert!(p.is_valid_uuid_v8(), "reserved version/variant bits intact");
            assert_eq!(p.layout_version(), IDENTITY_LAYOUT_VERSION);
        }
        // namespace probe: only namespace differs.
        let p = probes[0];
        assert_ne!(p.namespace(), base.namespace());
        assert_eq!(p.entity_type(), base.entity_type());
        assert_eq!(p.kind(), base.kind());
        assert_eq!(p.shape_hash(), base.shape_hash());
        assert_eq!(p.local(), base.local());
        // entity_type probe: only entity_type differs.
        let p = probes[1];
        assert_ne!(p.entity_type(), base.entity_type());
        assert_eq!(p.namespace(), base.namespace());
        assert_eq!(p.local(), base.local());
        assert_eq!(p.shape_hash(), base.shape_hash());
    }

    #[test]
    fn niblepath_prefix_is_a_true_ancestor_of_the_full_path() {
        // The routing cache is coarse but SOUND: the stored prefix is an
        // ancestor-or-equal of the full path (so is_ancestor_of routing is valid).
        let full = deep_path(); // depth 5
        let g = NodeGuid::new(0, 0, 0, full, 0, 0);
        assert_eq!(
            g.niblepath_depth(),
            NodeGuid::PREFIX_NIBBLES,
            "truncated to 4"
        );
        let prefix = g.niblepath().expect("prefix reconstructs");
        assert!(
            prefix.is_ancestor_of(full),
            "the GUID prefix must be an ancestor of the full path"
        );
        assert_eq!(prefix.basin(), full.basin(), "same DOLCE basin");
    }

    #[test]
    fn shallow_path_is_carried_whole() {
        // A path shallower than the prefix budget round-trips exactly.
        let shallow = NiblePath::root(0x1).child(0x2); // depth 2
        let g = NodeGuid::new(0, 0, 0, shallow, 0, 0);
        assert_eq!(g.niblepath_depth(), 2);
        assert_eq!(g.niblepath(), Some(shallow));
    }

    #[test]
    fn empty_path_is_the_no_route_sentinel() {
        let g = NodeGuid::new(0, 0, 0, NiblePath::EMPTY, 0, 0);
        assert_eq!(g.niblepath_depth(), 0);
        assert_eq!(g.niblepath(), Some(NiblePath::EMPTY));
    }

    #[test]
    fn as_bytes_from_bytes_round_trip() {
        let g = sample();
        let g2 = NodeGuid::from_bytes(*g.as_bytes());
        assert_eq!(g, g2);
    }

    #[test]
    fn display_is_a_canonical_uuid_string() {
        let g = sample();
        let s = g.to_string();
        assert_eq!(s.len(), 36, "8-4-4-4-12 + 4 hyphens");
        // Canonical UUID hyphen positions.
        for i in [8usize, 13, 18, 23] {
            assert_eq!(s.as_bytes()[i], b'-', "hyphen at {i}");
        }
        // Version nibble is the 13th hex digit (group 3, first char) = '8'.
        assert_eq!(s.chars().nth(14).unwrap(), '8');
        // entity_type 0xABCD is octets 1-2 → hex digits 2..6 = "abcd".
        assert_eq!(&s[2..6], "abcd");
        // namespace 0x07 → digits 0..2, kind 0x03 → digits 6..8.
        assert_eq!(&s[0..2], "07");
        assert_eq!(&s[6..8], "03");
    }

    #[test]
    fn size_and_alignment_are_16() {
        assert_eq!(core::mem::size_of::<NodeGuid>(), 16);
        assert_eq!(core::mem::align_of::<NodeGuid>(), 16);
    }
}
