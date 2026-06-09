//! # `hhtl` — the 16ⁿ nibble bucket router (the Abstammung tree axis).
//!
//! `wikidata-hhtl-load.md` §"HHTL = the cheap bucket router (16^n)": the ONE tree
//! axis is the `subClassOf` (P279) path, addressed as a fixed-fan-out-16 nibble
//! sequence — *"bucket path = nibble sequence → routing is bit-shift, not hash
//! lookup. O(1) arithmetic (super billig)."* This is the **downstream bucket
//! router** PR #438 (D-ARM-14 Phase 1) names but did not build: aerial discovers
//! the OWL/DOLCE skeleton + basins; this routes an entity into its 16ⁿ bucket.
//!
//! **Domain-agnostic by construction.** The router takes a `basin` nibble
//! (`0x0..=0xF`) and child nibbles; it does NOT know DOLCE. The DOLCE→basin
//! binding is resolved THROUGH the ontology cache (OD-DOLCE ratification, #441
//! `b31464d` "DOLCE-from-cache, dissolves 6v4") — never a hard-coded enum here.
//! So the duplicated `DolceCategory` (arm-discovery discovery-side *vs* ontology
//! cache-side) is dissolved at the **resolution layer**, not by a third copy in
//! contract: the structural router has zero DOLCE knowledge.
//!
//! The DOLCE top facets seed basins `0..3` by the cache's stable `dolce_id`
//! ordering — `ENDURANT=0`, `PERDURANT=1`, `QUALITY=2`, `ABSTRACT=3` (#441
//! `class_resolver::dolce_id`) — which is ALSO the order of arm-discovery's
//! discovery-side `DolceCategory::basin()` (#438). Both sides of the firewall
//! therefore agree on the nibble without either embedding the enum here; the
//! remaining `0x4..=0xF` basins are reserved (append-only) for finer top axes.
//! The Wikidata "D-CLS triple" `(class_id, shape_hash, presence_bitmask)` is
//! `(ClassId, StructuralSignature, FieldMask)` from #441; this path is its
//! addressing.
//!
//! **One tree axis only (`wikidata-hhtl-load.md`:46).** Multi-parent
//! ("flying-family") is NOT a second nibble path — it is an orthogonal facet bit
//! in the SAME [`FieldMask`](crate::class_view::FieldMask). *"Bat = mammal-path +
//! flight-bit, not two paths."* This keeps 16ⁿ a clean tree (cheap nibble
//! addressing) AND keeps multi-parent dedup.
//!
//! **mask-inherits-as-delta.** Walking DOWN the path is IS-A inheritance: a
//! child's presence mask is the parent's OR its own delta
//! ([`FieldMask::inherit`](crate::class_view::FieldMask::inherit)). N3 stable
//! positions mean the parent's bits never move; the child only adds.

/// Fixed HHTL fan-out: 16 children per level (one nibble). `wikidata-hhtl-load.md`:44.
pub const FAN_OUT: u8 = 16;

/// Max depth addressable in a single `u64` path (16 nibbles × 4 bits = 64).
/// Beyond this the bit-budget discipline says switch to a ref, not a deeper
/// nibble (`wikidata-hhtl-load.md`:71 "grows unbounded → path/ref").
pub const MAX_DEPTH: u8 = 16;

/// A path in the 16ⁿ Abstammung tree — a nibble sequence, root-first, packed into
/// a `u64`. Routing is bit-shift, not hash (O(1) arithmetic).
///
/// Layout: the root (basin) nibble occupies the highest *used* nibble; each
/// [`child`](NiblePath::child) shifts the accumulated path left 4 and ORs the new
/// leaf nibble into the low 4 bits. [`depth`](NiblePath::depth) counts the nibbles
/// used, so a partially-filled `u64` is unambiguous (leading zero nibbles are
/// "not yet routed", not basin 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct NiblePath {
    path: u64,
    depth: u8,
}

impl NiblePath {
    /// The empty path — no basin routed yet.
    pub const EMPTY: Self = Self { path: 0, depth: 0 };

    /// Start a path at a `basin` nibble — the DOLCE top facet, resolved UPSTREAM
    /// through the ontology cache (not decided here). An out-of-range
    /// `basin >= FAN_OUT` (16) returns [`EMPTY`](NiblePath::EMPTY), the "no route"
    /// sentinel — NOT a silent fold onto a valid basin (which would misroute
    /// ancestry — CodeRabbit #442). Mirrors [`child`](NiblePath::child)'s
    /// out-of-range no-op and `FieldMask`'s ignore-don't-fold discipline.
    #[must_use]
    pub const fn root(basin: u8) -> Self {
        if basin >= FAN_OUT {
            Self::EMPTY
        } else {
            Self {
                path: basin as u64,
                depth: 1,
            }
        }
    }

    /// Route one level deeper to child `nibble`. **Saturating:** returns `self`
    /// UNCHANGED once [`MAX_DEPTH`] is reached or `nibble >= FAN_OUT` (out of range —
    /// never folded onto a valid child, mirroring
    /// [`FieldMask`](crate::class_view::FieldMask)'s out-of-range discipline).
    ///
    /// At [`MAX_DEPTH`] the silent saturation means two *distinct* deeper paths would
    /// collide on this address — so a real-scale caller MUST gate on
    /// [`is_full`](NiblePath::is_full) or use [`try_child`](NiblePath::try_child),
    /// which signal the ceiling instead of colliding (D-ARM-14 review of #442).
    #[must_use]
    pub const fn child(self, nibble: u8) -> Self {
        if self.depth >= MAX_DEPTH || nibble >= FAN_OUT {
            self
        } else {
            Self {
                path: (self.path << 4) | (nibble as u64),
                depth: self.depth + 1,
            }
        }
    }

    /// Has this path reached [`MAX_DEPTH`] — i.e. [`child`](NiblePath::child) can no
    /// longer descend within the `u64`? When `true`, the bit-budget discipline
    /// (`wikidata-hhtl-load.md`:71 "grows unbounded → path/ref") says switch to a
    /// ref for deeper addressing: descending anyway via [`child`] is a SILENT no-op,
    /// so two distinct deeper classes would collide on this same path. The deferred
    /// 115M loader gates each descent on this (D-ARM-14 review of #442).
    #[must_use]
    pub const fn is_full(self) -> bool {
        self.depth >= MAX_DEPTH
    }

    /// Route one level deeper, returning `None` instead of silently saturating when
    /// the path [`is_full`](NiblePath::is_full) or `nibble >= FAN_OUT`. The explicit
    /// counterpart to [`child`](NiblePath::child) for callers that must NOT collide
    /// distinct deep paths (the real-scale loader).
    #[must_use]
    pub const fn try_child(self, nibble: u8) -> Option<Self> {
        if self.depth >= MAX_DEPTH || nibble >= FAN_OUT {
            None
        } else {
            Some(Self {
                path: (self.path << 4) | (nibble as u64),
                depth: self.depth + 1,
            })
        }
    }

    /// The basin (root) nibble — the DOLCE top facet this path lives under.
    /// `None` for the empty path.
    #[must_use]
    pub const fn basin(self) -> Option<u8> {
        if self.depth == 0 {
            None
        } else {
            Some(((self.path >> (4 * (self.depth as u32 - 1))) & 0x0F) as u8)
        }
    }

    /// The leaf (deepest) nibble. `None` for the empty path.
    #[must_use]
    pub const fn leaf(self) -> Option<u8> {
        if self.depth == 0 {
            None
        } else {
            Some((self.path & 0x0F) as u8)
        }
    }

    /// The parent path (one level shallower). `None` at the basin/empty — the
    /// basin has no parent in this tree (it IS the DOLCE top facet).
    #[must_use]
    pub const fn parent(self) -> Option<Self> {
        if self.depth <= 1 {
            None
        } else {
            Some(Self {
                path: self.path >> 4,
                depth: self.depth - 1,
            })
        }
    }

    /// Depth (number of nibbles routed).
    #[must_use]
    pub const fn depth(self) -> u8 {
        self.depth
    }

    /// Is this path a prefix of (ancestor-or-equal of) `other`? — the cheap
    /// arithmetic reachability test that replaces a P279\* graph walk. An empty
    /// path is an ancestor of nothing (there is no basin to share).
    #[must_use]
    pub const fn is_ancestor_of(self, other: Self) -> bool {
        if self.depth == 0 || self.depth > other.depth {
            false
        } else {
            // Align `other` down to self.depth, then compare the shared prefix.
            (other.path >> (4 * (other.depth as u32 - self.depth as u32))) == self.path
        }
    }

    /// The raw packed `(path, depth)` — for SoA facet-column storage / CAM key.
    #[must_use]
    pub const fn packed(self) -> (u64, u8) {
        (self.path, self.depth)
    }

    /// Reconstruct a path from its raw packed `(path, depth)` — the inverse of
    /// [`packed`](NiblePath::packed). Used by `identity::NodeGuid` to round-trip
    /// the routing-prefix it stores.
    ///
    /// Returns `None` if `depth > MAX_DEPTH`, or if `path` has bits set above the
    /// `depth` nibbles (an inconsistent pack — leading nibbles must be the route,
    /// trailing high bits must be zero). `from_packed(0, 0)` is [`EMPTY`](NiblePath::EMPTY).
    #[must_use]
    pub const fn from_packed(path: u64, depth: u8) -> Option<Self> {
        if depth > MAX_DEPTH {
            return None;
        }
        // `path` must fit in `depth` nibbles (4·depth bits); higher bits must be 0.
        // At MAX_DEPTH (16 nibbles = 64 bits) the whole u64 is usable — skip the
        // shift (a `>> 64` would be UB).
        let used_bits = 4 * depth as u32;
        if used_bits < 64 && (path >> used_bits) != 0 {
            return None;
        }
        Some(Self { path, depth })
    }

    /// Is this path a descendant-or-equal of `other`? — the symmetric form of
    /// [`is_ancestor_of`]. `self.is_descendant_of(other)` is equivalent to
    /// `other.is_ancestor_of(self)` BUT the form is sometimes more natural at
    /// the call site (e.g. iterating over candidate ancestors).
    ///
    /// Like [`is_ancestor_of`], the empty path is never a descendant of
    /// anything.
    #[must_use]
    pub const fn is_descendant_of(self, other: Self) -> bool {
        other.is_ancestor_of(self)
    }

    /// Are `self` and `other` siblings — distinct paths that share the SAME
    /// parent (and thus the same depth)? Returns `false` if either is the
    /// basin (depth 1 — basins have no parent in this tree), if the depths
    /// differ, or if the paths are equal.
    ///
    /// Together with [`is_ancestor_of`] / [`is_descendant_of`] this exposes
    /// the three structural relations the Pearl-junction classifier
    /// (`crate::pearl_junction`) needs without forcing the caller to do its
    /// own bit-shift arithmetic.
    #[must_use]
    pub const fn is_sibling_of(self, other: Self) -> bool {
        if self.depth != other.depth || self.depth <= 1 || self.path == other.path {
            return false;
        }
        // Same depth + same parent ⇔ matching top (depth−1) nibbles ⇔
        // matching all bits except the low 4 (the leaf nibble).
        const LEAF_MASK: u64 = !0x0F_u64;
        (self.path & LEAF_MASK) == (other.path & LEAF_MASK)
    }

    /// The longest common ancestor path — the longest prefix shared by
    /// `self` and `other`. `None` if the two paths share no basin (they
    /// live in disjoint DOLCE-facet subtrees, OR either is the empty path).
    ///
    /// Symmetric in its arguments: `a.common_ancestor(b) == b.common_ancestor(a)`.
    ///
    /// O(depth) — at most `MAX_DEPTH` nibble-shifts in the worst case.
    #[must_use]
    pub const fn common_ancestor(self, other: Self) -> Option<Self> {
        if self.depth == 0 || other.depth == 0 {
            return None;
        }
        // Align both paths to the shallower depth, then walk up until the
        // packed prefixes agree. Once we reach depth 0 without a match,
        // the two paths share no basin.
        let mut a_path = self.path;
        let mut a_depth = self.depth;
        let mut b_path = other.path;
        let mut b_depth = other.depth;
        while a_depth > b_depth {
            a_path >>= 4;
            a_depth -= 1;
        }
        while b_depth > a_depth {
            b_path >>= 4;
            b_depth -= 1;
        }
        // Same depth now. Walk up until the bits match.
        while a_path != b_path {
            if a_depth <= 1 {
                // Reaching depth 0 means the paths share no basin; reaching
                // depth 1 with no match means the basins themselves differ.
                if a_depth == 1 {
                    return None;
                }
                a_path >>= 4;
                b_path >>= 4;
                a_depth -= 1;
                continue;
            }
            a_path >>= 4;
            b_path >>= 4;
            a_depth -= 1;
        }
        if a_depth == 0 {
            None
        } else {
            Some(Self {
                path: a_path,
                depth: a_depth,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::class_view::FieldMask;

    #[test]
    fn root_child_basin_leaf_roundtrip_is_bitshift_exact() {
        // basin 0x2 (say DOLCE Quality) → child 0x5 → child 0xA.
        let p = NiblePath::root(0x2).child(0x5).child(0xA);
        assert_eq!(p.depth(), 3);
        assert_eq!(
            p.basin(),
            Some(0x2),
            "basin = root nibble, stable down the path"
        );
        assert_eq!(p.leaf(), Some(0xA), "leaf = deepest nibble");
        // parent walks back up exactly (bit-shift inverse of child).
        let pp = p.parent().unwrap();
        assert_eq!(pp, NiblePath::root(0x2).child(0x5));
        assert_eq!(pp.leaf(), Some(0x5));
        assert_eq!(p.parent().unwrap().parent().unwrap(), NiblePath::root(0x2));
        assert_eq!(NiblePath::root(0x2).parent(), None, "basin has no parent");
        assert_eq!(NiblePath::EMPTY.basin(), None);
    }

    #[test]
    fn from_packed_validates_depth_high_bits_and_roundtrips() {
        // (0, 0) is the EMPTY sentinel.
        assert_eq!(NiblePath::from_packed(0, 0), Some(NiblePath::EMPTY));

        // A well-formed (path, depth) reconstructs exactly what `child` builds.
        assert_eq!(
            NiblePath::from_packed(0x12, 2),
            Some(NiblePath::root(0x1).child(0x2)),
        );

        // depth > MAX_DEPTH is rejected.
        assert_eq!(NiblePath::from_packed(0, MAX_DEPTH + 1), None);

        // Bits set above the 4·depth route nibbles are an inconsistent pack.
        // depth = 2 ⇒ only the low 8 bits may be set; 0x112 has a 9th.
        assert_eq!(NiblePath::from_packed(0x112, 2), None);

        // Boundary: at MAX_DEPTH the whole u64 is usable (the `used_bits < 64`
        // guard skips a `>> 64` UB), so even all-ones round-trips.
        let max = NiblePath::from_packed(u64::MAX, MAX_DEPTH);
        assert_eq!(max.map(NiblePath::packed), Some((u64::MAX, MAX_DEPTH)));

        // packed ∘ from_packed is identity on every valid path.
        for p in [
            NiblePath::EMPTY,
            NiblePath::root(0x3),
            NiblePath::root(0x3).child(0x5).child(0xA),
        ] {
            let (path, depth) = p.packed();
            assert_eq!(NiblePath::from_packed(path, depth), Some(p));
        }
    }

    #[test]
    fn depth_caps_at_max_and_rejects_out_of_range_nibble() {
        // Fill to MAX_DEPTH, then one more child is a no-op (not a wrap/overflow).
        let mut p = NiblePath::root(0x1);
        while p.depth() < MAX_DEPTH {
            p = p.child(0xF);
        }
        assert_eq!(p.depth(), MAX_DEPTH);
        assert_eq!(
            p.child(0x3),
            p,
            "child past MAX_DEPTH is a no-op, never wraps"
        );
        // Out-of-range nibble (>= FAN_OUT) is ignored, NOT folded onto a valid child.
        assert_eq!(NiblePath::root(0x1).child(16), NiblePath::root(0x1));
        assert_eq!(NiblePath::root(0x1).child(99), NiblePath::root(0x1));
        // root() rejects an out-of-range basin to EMPTY — never folds 16 → basin 0
        // (which would misroute ancestry; CodeRabbit #442).
        assert_eq!(NiblePath::root(16), NiblePath::EMPTY);
        assert_eq!(NiblePath::root(99), NiblePath::EMPTY);
        assert_eq!(
            NiblePath::root(16).basin(),
            None,
            "bad basin must not alias to basin 0"
        );
    }

    #[test]
    fn is_ancestor_of_is_cheap_prefix_reachability() {
        let mammal = NiblePath::root(0x0).child(0x3); // Endurant → …mammal
        let bat = mammal.child(0x7);
        let dog = mammal.child(0x8);
        assert!(mammal.is_ancestor_of(bat), "mammal is an ancestor of bat");
        assert!(mammal.is_ancestor_of(dog));
        assert!(
            mammal.is_ancestor_of(mammal),
            "ancestor-or-EQUAL (reflexive)"
        );
        assert!(
            !bat.is_ancestor_of(mammal),
            "child is not an ancestor of its parent"
        );
        assert!(
            !bat.is_ancestor_of(dog),
            "siblings are not ancestors of each other"
        );
        // A different basin shares no prefix.
        let process = NiblePath::root(0x1).child(0x3);
        assert!(
            !mammal.is_ancestor_of(process),
            "different basin → not reachable"
        );
        assert!(
            !NiblePath::EMPTY.is_ancestor_of(bat),
            "empty path is an ancestor of nothing"
        );
    }

    #[test]
    fn multi_parent_is_a_facet_bit_not_a_second_path() {
        // "Bat = mammal-path + flight-bit, not two paths" (wikidata-hhtl-load.md:46).
        // ONE nibble path (the mammal Abstammung), the flight capability is an
        // orthogonal facet bit in the SAME FieldMask — never a second NiblePath.
        let bat_path = NiblePath::root(0x0).child(0x3).child(0x7); // mammal → bat
                                                                   // declared mammal fields (positions 0,1,2) + the flight facet bit (40).
        let mammal_mask = FieldMask::from_positions(&[0, 1, 2]);
        let flight_facet = FieldMask::EMPTY.with(40);
        let bat_mask = mammal_mask.inherit(flight_facet);

        assert_eq!(bat_path.depth(), 3, "bat is reached by ONE path, not two");
        assert!(
            bat_mask.has(0) && bat_mask.has(1) && bat_mask.has(2),
            "inherits mammal fields"
        );
        assert!(
            bat_mask.has(40),
            "carries the flight facet bit in the same mask"
        );
        assert_eq!(bat_mask.count(), 4);
    }

    #[test]
    fn is_full_and_try_child_signal_depth_exhaustion() {
        // child() saturates silently at MAX_DEPTH; is_full()/try_child() expose the
        // ceiling so the deferred loader switches to a ref instead of colliding two
        // distinct deep paths (D-ARM-14 review of #442).
        let mut p = NiblePath::root(0x1);
        assert!(!p.is_full());
        while !p.is_full() {
            p = p.try_child(0xF).expect("descends while not full");
        }
        assert_eq!(p.depth(), MAX_DEPTH);
        assert!(p.is_full());
        assert_eq!(
            p.try_child(0x2),
            None,
            "try_child signals exhaustion, not a silent collision"
        );
        assert_eq!(
            p.child(0x2),
            p,
            "child() still saturates (the convenience path)"
        );
        assert_eq!(
            NiblePath::root(0x1).try_child(16),
            None,
            "out-of-range nibble is None too"
        );
    }

    #[test]
    fn is_descendant_of_inverse_of_is_ancestor_of() {
        let mammal = NiblePath::root(0x1);
        let dog = NiblePath::root(0x1).child(0x1);
        let cat = NiblePath::root(0x2);
        assert!(dog.is_descendant_of(mammal));
        assert!(!mammal.is_descendant_of(dog));
        assert!(!dog.is_descendant_of(cat));
        // empty path is never a descendant of anything
        assert!(!NiblePath::EMPTY.is_descendant_of(mammal));
    }

    #[test]
    fn is_sibling_of_requires_same_parent_distinct_paths() {
        let dog = NiblePath::root(0x1).child(0x1);
        let cat = NiblePath::root(0x1).child(0x2);
        let lance = NiblePath::root(0x1).child(0x1);
        // siblings: same parent (mammal), distinct leaf nibbles
        assert!(dog.is_sibling_of(cat));
        assert!(cat.is_sibling_of(dog));
        // not siblings: equal paths
        assert!(!dog.is_sibling_of(lance));
        // not siblings: different depth
        let mammal = NiblePath::root(0x1);
        assert!(!dog.is_sibling_of(mammal));
        // not siblings: different parent
        let plant = NiblePath::root(0x2).child(0x1);
        assert!(!dog.is_sibling_of(plant));
        // basins themselves are not siblings (depth 1, no parent)
        let b1 = NiblePath::root(0x1);
        let b2 = NiblePath::root(0x2);
        assert!(!b1.is_sibling_of(b2));
    }

    #[test]
    fn common_ancestor_returns_longest_shared_prefix() {
        // (1)(2)(3)(4) and (1)(2)(5)(6) share (1)(2)
        let a = NiblePath::root(0x1).child(0x2).child(0x3).child(0x4);
        let b = NiblePath::root(0x1).child(0x2).child(0x5).child(0x6);
        let lca = a.common_ancestor(b).unwrap();
        assert_eq!(lca.depth(), 2);
        assert_eq!(lca.basin(), Some(0x1));
        assert_eq!(lca.leaf(), Some(0x2));
        // symmetric
        assert_eq!(b.common_ancestor(a), Some(lca));
    }

    #[test]
    fn common_ancestor_handles_different_depths() {
        // (1)(2) is an ancestor of (1)(2)(3); LCA should be (1)(2)
        let shallow = NiblePath::root(0x1).child(0x2);
        let deep = NiblePath::root(0x1).child(0x2).child(0x3);
        assert_eq!(shallow.common_ancestor(deep), Some(shallow));
        assert_eq!(deep.common_ancestor(shallow), Some(shallow));
    }

    #[test]
    fn common_ancestor_disjoint_basins_returns_none() {
        // different basins → no common ancestor in this tree
        let a = NiblePath::root(0x1).child(0x2);
        let b = NiblePath::root(0x3).child(0x4);
        assert_eq!(a.common_ancestor(b), None);
        assert_eq!(b.common_ancestor(a), None);
    }

    #[test]
    fn common_ancestor_empty_path_returns_none() {
        let a = NiblePath::root(0x1);
        assert_eq!(a.common_ancestor(NiblePath::EMPTY), None);
        assert_eq!(NiblePath::EMPTY.common_ancestor(a), None);
    }
}
