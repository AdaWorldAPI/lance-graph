//! `prefix_select` — compile a (classid, cascade-prefix, field-mask) SELECTION
//! into **key ranges × column masks**, so a columnar scanner (Lance) reads ONLY
//! the masked columns over ONLY the addressed rows. This is the structural
//! pushdown GraphQL cannot express: it lowers an *address prefix* to a contiguous
//! byte range over the canonical [`NodeGuid`](crate::canonical_node::NodeGuid) key
//! and pairs it with the [`WideFieldMask`] column projection.
//!
//! # The comparator (READ THIS — it is the load-bearing subtlety)
//!
//! The 16-byte key is stored **little-endian** (see
//! [`canonical_node`](crate::canonical_node)): bytes `0..4` are the `classid`
//! `u32` LE, `4..6` HEEL `u16` LE, `6..8` HIP, `8..10` TWIG, `10..16` the tail
//! (family `u24` ++ identity `u24`). A RANGE scan needs a **total order in which
//! "shares a prefix" ⟺ "occupies a contiguous run"** — and raw stored-byte
//! lexicographic order does NOT provide it. Example: `classid 0x0000_0001` stores
//! as `[01,00,00,00]` and `classid 0x0000_0100` as `[00,01,00,00]`; comparing the
//! stored bytes lexicographically yields `0x0000_0100 < 0x0000_0001`, which is
//! numerically backwards. A range compiled under that wrong order silently returns
//! the wrong row set.
//!
//! The comparator this module defines and every consumer MUST use is
//! [`scan_cmp`]: it compares two stored keys by rendering each field
//! **big-endian** and comparing the tuple `(classid, HEEL, HIP, TWIG, family,
//! identity)` numerically — equivalently, lexicographic order over the
//! big-endian rendering [`scan_key`] produces. This is exactly the coarse→fine
//! cascade order the existing [`hhtl::NiblePath`](crate::hhtl) lowering already
//! uses (`canon · HEEL · HIP · TWIG`, most-significant nibble first), so a range
//! prefix here is an [`hhtl`](crate::hhtl) ancestor there — one order across the
//! spine.
//!
//! **Bounds are stored keys.** [`KeyRange::lo`]/[`KeyRange::hi`] are ordinary
//! 16-byte `NodeGuid` byte patterns (the least/greatest key of the addressed set
//! *under [`scan_cmp`]*), so a consumer feeds `node.key.as_bytes()` straight into
//! [`contains`] with no transform. The scanner is only obligated to compare keys
//! with [`scan_cmp`] (or, equivalently, to pre-render each key with [`scan_key`]
//! and compare the results lexicographically).
//!
//! **Why the cascade prefix is contiguous.** A prefix fixes the top `levels`
//! nibbles of the 12-nibble `HEEL·HIP·TWIG` sequence (`level >> 2` selects the
//! tier — a shift, never a division; the coarsest nibble is HEEL's most-
//! significant). In big-endian order the fixed nibbles are the leading bits, so
//! the addressed set is `[prefix‖0…0 , prefix‖F…F]` — one contiguous run. The
//! Morton `x/y` nibble-interleave (256 = 4⁴ centroid hierarchy) is a *semantic
//! lens* on which axis each fixed nibble names; it never re-orders the nibbles, so
//! it does not affect range contiguity.
//!
//! # classid: exact (`u32`) vs concept (`u16`)
//!
//! The key's `classid` is a `u32`; the canon-high split (see
//! [`ogar_codebook`](crate::ogar_codebook)) puts the **shared concept in the high
//! `u16`** (`classid >> 16`) and the **app render prefix in the low `u16`**. Two
//! selections follow, both provably a *single* contiguous [`ScanUnit`]:
//!
//! * [`compile`] of a [`ClassSelect::Exact`] prefix — fixes all 32 classid bits,
//!   then the cascade prefix. One range.
//! * [`compile_concept`] — fixes only the high `u16` (the concept); the low `u16`
//!   (every app prefix) is free. Because the concept occupies the *most-
//!   significant* 16 bits of the big-endian classid, "all app prefixes of a
//!   concept" is `[concept‖0x0000 , concept‖0xFFFF]` — still **one contiguous
//!   range**, not a `Vec<ScanUnit>`.
//!
//! A concept selection carries **no** cascade path: the free app-prefix dimension
//! sorts *between* the concept and the HHT nibbles, so a deep cascade under a bare
//! concept would not be one contiguous range. Deep addressing always uses an exact
//! classid ([`ClassSelect::Exact`]); a concept selection is concept-wide only.

use crate::class_view::WideFieldMask;
use core::cmp::Ordering;

/// How the `classid` half of a selection is pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClassSelect {
    /// An exact 32-bit `classid` — one concept AND one app render prefix. The
    /// deep-addressing form: compiles with any cascade `levels`.
    Exact(u32),
    /// A shared **concept** (the canon-high `u16`, `classid >> 16`) across ALL
    /// app render prefixes (the low `u16`, free). Concept-wide only — no cascade
    /// path (see the module docs).
    Concept(u16),
}

/// A selection prefix: a [`ClassSelect`] plus `0..=12` cascade path levels of the
/// `HEEL·HIP·TWIG` address. Only the top `levels` nibbles of `heel`/`hip`/`twig`
/// are significant; deeper nibbles are free (spanned by the compiled range).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CascadePrefix {
    /// Which classid(s) the selection pins.
    pub class: ClassSelect,
    /// Cascade path depth in nibbles, `0..=12` (3 tiers × 4 nibbles). Values
    /// above 12 are clamped by the constructors and [`key_range`](CascadePrefix::key_range).
    pub levels: u8,
    /// HEEL tier (cascade levels 0..4). Only the top `min(levels, 4)` nibbles matter.
    pub heel: u16,
    /// HIP tier (cascade levels 4..8). Only the top `clamp(levels-4, 0, 4)` nibbles matter.
    pub hip: u16,
    /// TWIG tier (cascade levels 8..12). Only the top `clamp(levels-8, 0, 4)` nibbles matter.
    pub twig: u16,
}

/// The greatest addressable cascade depth: 3 tiers × 4 nibbles.
pub const MAX_LEVELS: u8 = 12;

impl CascadePrefix {
    /// An exact-classid selection with no cascade path — the whole classid, every
    /// HEEL/HIP/TWIG. One contiguous range over exactly that classid.
    #[inline]
    #[must_use]
    pub const fn exact(classid: u32) -> Self {
        Self {
            class: ClassSelect::Exact(classid),
            levels: 0,
            heel: 0,
            hip: 0,
            twig: 0,
        }
    }

    /// A concept-wide selection: every app render prefix of `concept` (the
    /// canon-high `u16`). No cascade path (see the module docs).
    #[inline]
    #[must_use]
    pub const fn concept(concept: u16) -> Self {
        Self {
            class: ClassSelect::Concept(concept),
            levels: 0,
            heel: 0,
            hip: 0,
            twig: 0,
        }
    }

    /// An exact-classid selection with a cascade path of `levels` nibbles (clamped
    /// to [`MAX_LEVELS`]). Only the top `levels` nibbles of `heel`/`hip`/`twig`
    /// are pinned.
    #[inline]
    #[must_use]
    pub const fn with_path(classid: u32, levels: u8, heel: u16, hip: u16, twig: u16) -> Self {
        Self {
            class: ClassSelect::Exact(classid),
            levels: if levels > MAX_LEVELS {
                MAX_LEVELS
            } else {
                levels
            },
            heel,
            hip,
            twig,
        }
    }

    /// Lower this prefix to its [`KeyRange`] under the [`scan_cmp`] order.
    ///
    /// `lo`/`hi` are stored-LE `NodeGuid` byte patterns: `lo` fixes the prefix
    /// nibbles and zero-fills the rest; `hi` fixes the same prefix nibbles and
    /// `0xF`-fills the rest (incl. the whole tail). For [`ClassSelect::Concept`]
    /// the cascade path is forced free regardless of `levels` (a bare concept
    /// spans the free app-prefix dimension, which sorts between the concept and
    /// the HHT nibbles — a deeper cascade would not be one contiguous range; a
    /// `debug_assert` guards misuse).
    #[must_use]
    pub fn key_range(&self) -> KeyRange {
        let (cid_lo, cid_hi, hht_free) = match self.class {
            ClassSelect::Exact(c) => (c, c, false),
            ClassSelect::Concept(concept) => {
                debug_assert!(
                    self.levels == 0,
                    "concept selection is concept-wide only: the free app-prefix \
                     dimension sorts between the concept and the cascade path, so a \
                     cascade under a bare concept is not one contiguous range — use \
                     ClassSelect::Exact for deep addressing"
                );
                let base = (concept as u32) << 16;
                (base, base | 0xFFFF, true)
            }
        };
        let levels = if hht_free {
            0
        } else if self.levels > MAX_LEVELS {
            MAX_LEVELS
        } else {
            self.levels
        };
        let (hf, pf, tf) = tier_fixed(levels);
        let (h_lo, h_hi) = u16_bounds(self.heel, hf);
        let (p_lo, p_hi) = u16_bounds(self.hip, pf);
        let (t_lo, t_hi) = u16_bounds(self.twig, tf);
        KeyRange {
            lo: stored_key(cid_lo, h_lo, p_lo, t_lo, 0x00),
            hi: stored_key(cid_hi, h_hi, p_hi, t_hi, 0xFF),
        }
    }
}

/// A contiguous key range under the [`scan_cmp`] order: `[lo, hi]` inclusive.
///
/// `lo` and `hi` are stored-LE 16-byte [`NodeGuid`](crate::canonical_node::NodeGuid)
/// keys — the least and greatest key of the addressed set *as ordered by
/// [`scan_cmp`]*. A scanner must compare with [`scan_cmp`]; comparing the raw
/// bytes lexicographically is WRONG (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyRange {
    /// Least key in the range (inclusive), stored-LE.
    pub lo: [u8; 16],
    /// Greatest key in the range (inclusive), stored-LE.
    pub hi: [u8; 16],
}

impl KeyRange {
    /// Does `key` (a stored-LE 16-byte key) fall inside this range under [`scan_cmp`]?
    #[inline]
    #[must_use]
    pub fn contains(&self, key: &[u8; 16]) -> bool {
        contains(self, key)
    }

    /// Is this range a (non-strict) subset of `outer`? See [`narrows`].
    #[inline]
    #[must_use]
    pub fn narrows(&self, outer: &KeyRange) -> bool {
        narrows(self, outer)
    }

    /// Do this range and `other` share no key? See [`disjoint`].
    #[inline]
    #[must_use]
    pub fn is_disjoint(&self, other: &KeyRange) -> bool {
        disjoint(self, other)
    }
}

/// A compiled scan unit: WHICH rows ([`range`](ScanUnit::range)) × WHICH columns
/// ([`columns`](ScanUnit::columns)). The columnar scanner reads only the masked
/// columns over only the addressed rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanUnit {
    /// The addressed row range under [`scan_cmp`].
    pub range: KeyRange,
    /// The column projection — passed through from the selection untouched.
    pub columns: WideFieldMask,
}

/// Compile an **exact-classid** selection prefix + column mask into one
/// [`ScanUnit`]. Always a single contiguous range (classid fully pinned, then the
/// cascade prefix). The `columns` mask is carried through verbatim.
///
/// A [`ClassSelect::Concept`] prefix is accepted but only well-formed at
/// `levels == 0` (see [`CascadePrefix::key_range`]); prefer [`compile_concept`]
/// for the concept case.
#[inline]
#[must_use]
pub fn compile(prefix: &CascadePrefix, columns: WideFieldMask) -> ScanUnit {
    ScanUnit {
        range: prefix.key_range(),
        columns,
    }
}

/// Compile a **concept-wide** selection: every app render prefix of `concept`
/// (the canon-high `u16`) + column mask, into one [`ScanUnit`].
///
/// This is provably one contiguous range: the concept occupies the most-
/// significant 16 bits of the big-endian classid, so all its app prefixes form
/// `[concept‖0x0000 , concept‖0xFFFF]` under [`scan_cmp`] — not a `Vec<ScanUnit>`.
/// The `columns` mask is carried through verbatim.
#[inline]
#[must_use]
pub fn compile_concept(concept: u16, columns: WideFieldMask) -> ScanUnit {
    compile(&CascadePrefix::concept(concept), columns)
}

/// Render a stored-LE 16-byte key into its **canonical big-endian scan key** —
/// the byte string whose plain lexicographic order IS the [`scan_cmp`] order.
///
/// Each field is emitted most-significant-byte first: `classid` (4), HEEL (2),
/// HIP (2), TWIG (2), family (3), identity (3). Lexicographic comparison of two
/// scan keys therefore compares `(classid, HEEL, HIP, TWIG, family, identity)`
/// numerically, coarse→fine — the same order the [`hhtl`](crate::hhtl) cascade
/// uses.
#[inline]
#[must_use]
pub const fn scan_key(g: &[u8; 16]) -> [u8; 16] {
    [
        g[3], g[2], g[1], g[0], // classid u32, big-endian
        g[5], g[4], // HEEL u16, big-endian
        g[7], g[6], // HIP u16, big-endian
        g[9], g[8], // TWIG u16, big-endian
        g[12], g[11], g[10], // family u24, big-endian
        g[15], g[14], g[13], // identity u24, big-endian
    ]
}

/// The canonical scan comparator over two stored-LE 16-byte keys.
///
/// Compares `(classid, HEEL, HIP, TWIG, family, identity)` numerically by rendering
/// each key big-endian ([`scan_key`]) and comparing lexicographically. This is the
/// ONE order every consumer must use — comparing stored bytes directly is wrong
/// (module docs). A total order, so it is a valid sort key.
#[inline]
#[must_use]
pub fn scan_cmp(a: &[u8; 16], b: &[u8; 16]) -> Ordering {
    scan_key(a).cmp(&scan_key(b))
}

/// Does `key` (stored-LE) fall within `range` (inclusive) under [`scan_cmp`]?
#[inline]
#[must_use]
pub fn contains(range: &KeyRange, key: &[u8; 16]) -> bool {
    scan_cmp(&range.lo, key) != Ordering::Greater && scan_cmp(key, &range.hi) != Ordering::Greater
}

/// Is `deeper` a (non-strict) subset of `shallower`? True iff
/// `shallower.lo <= deeper.lo` AND `deeper.hi <= shallower.hi` under [`scan_cmp`]
/// — the containment both a deeper cascade level and an exact classid under a
/// concept satisfy relative to their parent selection.
#[inline]
#[must_use]
pub fn narrows(deeper: &KeyRange, shallower: &KeyRange) -> bool {
    scan_cmp(&shallower.lo, &deeper.lo) != Ordering::Greater
        && scan_cmp(&deeper.hi, &shallower.hi) != Ordering::Greater
}

/// Do two ranges share no key? True iff one ends strictly before the other begins
/// under [`scan_cmp`] — the sibling-disjointness check (two same-depth prefixes
/// that differ in a fixed nibble never overlap).
#[inline]
#[must_use]
pub fn disjoint(a: &KeyRange, b: &KeyRange) -> bool {
    scan_cmp(&a.hi, &b.lo) == Ordering::Less || scan_cmp(&b.hi, &a.lo) == Ordering::Less
}

// ── internals ────────────────────────────────────────────────────────────────

/// How many nibbles each tier pins for a cascade depth of `levels` (`0..=12`).
/// `level >> 2` selects the tier — arithmetic by clamp/subtract, never division.
#[inline]
const fn tier_fixed(levels: u8) -> (u8, u8, u8) {
    let heel = if levels > 4 { 4 } else { levels };
    let hip = if levels <= 4 {
        0
    } else if levels - 4 > 4 {
        4
    } else {
        levels - 4
    };
    let twig = if levels <= 8 {
        0
    } else if levels - 8 > 4 {
        4
    } else {
        levels - 8
    };
    (heel, hip, twig)
}

/// The `(lo, hi)` bounds of a `u16` tier when its top `fixed` (`0..=4`) nibbles
/// are pinned to `v`'s and the rest are free: `lo` zero-fills, `hi` `0xF`-fills.
#[inline]
const fn u16_bounds(v: u16, fixed: u8) -> (u16, u16) {
    if fixed == 0 {
        return (0, 0xFFFF);
    }
    if fixed >= 4 {
        return (v, v);
    }
    let free_bits = 16 - 4 * fixed as u32; // low bits left free
    let keep = (!0u16) << free_bits; // top `fixed` nibbles
    let lo = v & keep;
    (lo, lo | !keep)
}

/// Assemble a stored-LE 16-byte key from its fields; the 6 tail bytes are all set
/// to `tail` (`0x00` for a range floor, `0xFF` for a ceiling).
#[inline]
const fn stored_key(classid: u32, heel: u16, hip: u16, twig: u16, tail: u8) -> [u8; 16] {
    let c = classid.to_le_bytes();
    let h = heel.to_le_bytes();
    let p = hip.to_le_bytes();
    let t = twig.to_le_bytes();
    [
        c[0], c[1], c[2], c[3], // classid LE
        h[0], h[1], // HEEL LE
        p[0], p[1], // HIP LE
        t[0], t[1], // TWIG LE
        tail, tail, tail, tail, tail, tail, // family ++ identity
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_node::NodeGuid;

    fn key(classid: u32, heel: u16, hip: u16, twig: u16, family: u32, identity: u32) -> [u8; 16] {
        *NodeGuid::new(classid, heel, hip, twig, family, identity).as_bytes()
    }

    // ── (a) a classid range contains exactly keys of that classid ──────────────

    #[test]
    fn exact_classid_contains_exactly_that_classid() {
        let cid = 0x0700_0000u32;
        let unit = compile(&CascadePrefix::exact(cid), WideFieldMask::from(0b1011));

        // any HEEL/HIP/TWIG/tail with THIS classid is inside
        assert!(unit
            .range
            .contains(&key(cid, 0x1234, 0x5678, 0x9abc, 0x00_0001, 0x00_0002)));
        // both extremes of the free suffix are inside
        assert!(unit.range.contains(&key(cid, 0, 0, 0, 0, 0)));
        assert!(unit
            .range
            .contains(&key(cid, 0xFFFF, 0xFFFF, 0xFFFF, 0xFF_FFFF, 0xFF_FFFF)));

        // classid ± 1 is rejected (classid is the primary sort field)
        assert!(!unit.range.contains(&key(cid + 1, 0, 0, 0, 0, 0)));
        assert!(!unit
            .range
            .contains(&key(cid - 1, 0xFFFF, 0xFFFF, 0xFFFF, 0xFF_FFFF, 0xFF_FFFF)));
    }

    // ── (b) each added level STRICTLY narrows (containment both directions) ─────

    #[test]
    fn each_cascade_level_strictly_narrows() {
        let cid = 0x0A01_0000u32;
        let (heel, hip, twig) = (0xABCDu16, 0xEF01u16, 0x2345u16);

        let ranges: Vec<KeyRange> = (0..=MAX_LEVELS)
            .map(|l| CascadePrefix::with_path(cid, l, heel, hip, twig).key_range())
            .collect();

        for l in 1..=MAX_LEVELS as usize {
            let deeper = &ranges[l];
            let shallower = &ranges[l - 1];
            // deeper ⊆ shallower
            assert!(
                narrows(deeper, shallower),
                "level {l} must ⊆ level {}",
                l - 1
            );
            // shallower ⊄ deeper (strictly wider): a nibble freed at `shallower`
            // is pinned at `deeper`, so shallower is not contained in deeper
            assert!(
                !narrows(shallower, deeper),
                "level {} must be strictly wider than level {l}",
                l - 1
            );
        }

        // subset proven by sampled keys: the exact key at (heel,hip,twig) is in
        // EVERY level; a key whose top HEEL nibble differs sits in level 0 but not
        // level 1.
        let exact = key(cid, heel, hip, twig, 0x00_0007, 0x00_0009);
        for r in &ranges {
            assert!(r.contains(&exact));
        }
        let flip_top = heel ^ 0xF000; // change level-0 nibble
        let off = key(cid, flip_top, hip, twig, 0, 0);
        assert!(ranges[0].contains(&off));
        assert!(!ranges[1].contains(&off));
    }

    // ── (c) same-depth sibling prefixes are disjoint ───────────────────────────

    #[test]
    fn same_depth_siblings_are_disjoint() {
        let cid = 0x0100_0000u32;
        // differ at cascade level 1 (top HEEL nibble): 0xA… vs 0xB…
        let a = CascadePrefix::with_path(cid, 1, 0xA000, 0, 0).key_range();
        let b = CascadePrefix::with_path(cid, 1, 0xB000, 0, 0).key_range();
        assert!(disjoint(&a, &b));
        assert!(disjoint(&b, &a));

        // differ at a deeper level (first HIP nibble), levels 5
        let c = CascadePrefix::with_path(cid, 5, 0xABCD, 0x1000, 0).key_range();
        let d = CascadePrefix::with_path(cid, 5, 0xABCD, 0x2000, 0).key_range();
        assert!(disjoint(&c, &d));
        // a range is never disjoint from itself
        assert!(!disjoint(&c, &c));
    }

    // ── (d) THE COMPARATOR TEST: compiled range covers a contiguous sorted run ──

    #[test]
    fn compiled_range_is_a_contiguous_run_under_scan_cmp() {
        // classids chosen so that stored-LE byte order and numeric order DISAGREE
        // — this is the byte-order falsifier.
        let classids = [0x0000_0001u32, 0x0000_0100, 0x0001_0000, 0x0100_0000];
        let mut keys: Vec<[u8; 16]> = Vec::new();
        for &c in &classids {
            for hh in [0x0000u16, 0x8000, 0xFFFF] {
                keys.push(key(c, hh, 0x1111, 0x2222, 0x00_0003, 0x00_0004));
            }
        }
        keys.sort_by(scan_cmp);

        // exact(0x0000_0100) must select exactly the classid-0x0000_0100 keys,
        // and they must form a contiguous run in the sorted list.
        let unit = compile(&CascadePrefix::exact(0x0000_0100), WideFieldMask::EMPTY);
        let hits: Vec<usize> = keys
            .iter()
            .enumerate()
            .filter(|(_, k)| unit.range.contains(k))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(hits.len(), 3, "three HEEL variants of that classid");
        // contiguous: indices are consecutive
        assert_eq!(
            hits.last().unwrap() - hits.first().unwrap(),
            hits.len() - 1,
            "selected keys must be a contiguous run in scan order"
        );
        // and every key in that index span is a hit (no interlopers)
        for k in &keys[*hits.first().unwrap()..=*hits.last().unwrap()] {
            assert!(unit.range.contains(k));
        }
    }

    #[test]
    fn scan_cmp_disagrees_with_raw_byte_order() {
        // Proof we did NOT use stored-byte lexicographic order: for these two
        // classids the numeric order and the raw stored-byte order are OPPOSITE.
        let a = key(0x0000_0001, 0, 0, 0, 0, 0);
        let b = key(0x0000_0100, 0, 0, 0, 0, 0);
        assert_eq!(scan_cmp(&a, &b), Ordering::Less); // 1 < 256, numerically
        assert_eq!(a.cmp(&b), Ordering::Greater); // stored-LE bytes: [01,..] > [00,01,..]
    }

    // ── (e) column mask passes through untouched ───────────────────────────────

    #[test]
    fn column_mask_passes_through_untouched() {
        let mask = WideFieldMask::from(0b1010_1100u64).with(70); // wide (>64)
        let unit = compile(&CascadePrefix::exact(0x0202_0000), mask.clone());
        assert_eq!(unit.columns, mask);

        let cmask = WideFieldMask::full_for(9);
        let cunit = compile_concept(0x0202, cmask.clone());
        assert_eq!(cunit.columns, cmask);
    }

    // ── (f) concept() covers exactly all app-prefix classids of the concept ────

    #[test]
    fn concept_covers_all_app_prefixes_of_the_concept() {
        let concept = 0x0700u16;
        let unit = compile_concept(concept, WideFieldMask::EMPTY);

        // every app render prefix (the low u16) resolves inside
        for app in [0x0000u16, 0x0001, 0x0005, 0x1000, 0xFFFF] {
            let classid = ((concept as u32) << 16) | app as u32;
            assert!(
                unit.range.contains(&key(classid, 0x1234, 0, 0, 0, 5)),
                "app prefix {app:#06x} of concept {concept:#06x} must be inside"
            );
        }

        // a different concept is entirely outside, at both app extremes
        for other in [concept - 1, concept + 1] {
            for app in [0x0000u16, 0xFFFF] {
                let classid = ((other as u32) << 16) | app as u32;
                assert!(
                    !unit.range.contains(&key(classid, 0, 0, 0, 0, 0)),
                    "concept {other:#06x} must be outside concept {concept:#06x}"
                );
            }
        }

        // the concept range is exactly [concept:0000 .. concept:FFFF]
        assert_eq!(unit.range.lo, key((concept as u32) << 16, 0, 0, 0, 0, 0));
        assert_eq!(
            unit.range.hi,
            key(
                ((concept as u32) << 16) | 0xFFFF,
                0xFFFF,
                0xFFFF,
                0xFFFF,
                0xFF_FFFF,
                0xFF_FFFF
            )
        );
    }

    // ── extra: an exact classid narrows a concept (u16 ⊇ u32 reconciliation) ───

    #[test]
    fn exact_classid_narrows_its_concept() {
        let concept = 0x0202u16;
        let concept_range = compile_concept(concept, WideFieldMask::EMPTY).range;
        for app in [0x0000u16, 0x0002, 0x1000] {
            let classid = ((concept as u32) << 16) | app as u32;
            let exact_range = compile(&CascadePrefix::exact(classid), WideFieldMask::EMPTY).range;
            assert!(
                narrows(&exact_range, &concept_range),
                "exact classid {classid:#010x} must ⊆ its concept {concept:#06x}"
            );
        }
        // an exact classid of a DIFFERENT concept does not narrow this concept
        let other = compile(
            &CascadePrefix::exact(((concept as u32 + 1) << 16) | 0x0001),
            WideFieldMask::EMPTY,
        )
        .range;
        assert!(!narrows(&other, &concept_range));
    }

    // ── tier arithmetic sanity ─────────────────────────────────────────────────

    #[test]
    fn tier_fixed_partitions_levels_by_shift() {
        assert_eq!(tier_fixed(0), (0, 0, 0));
        assert_eq!(tier_fixed(3), (3, 0, 0));
        assert_eq!(tier_fixed(4), (4, 0, 0));
        assert_eq!(tier_fixed(6), (4, 2, 0));
        assert_eq!(tier_fixed(8), (4, 4, 0));
        assert_eq!(tier_fixed(11), (4, 4, 3));
        assert_eq!(tier_fixed(12), (4, 4, 4));
    }

    #[test]
    fn u16_bounds_pins_top_nibbles() {
        assert_eq!(u16_bounds(0xABCD, 0), (0x0000, 0xFFFF));
        assert_eq!(u16_bounds(0xABCD, 1), (0xA000, 0xAFFF));
        assert_eq!(u16_bounds(0xABCD, 2), (0xAB00, 0xABFF));
        assert_eq!(u16_bounds(0xABCD, 3), (0xABC0, 0xABCF));
        assert_eq!(u16_bounds(0xABCD, 4), (0xABCD, 0xABCD));
    }
}
