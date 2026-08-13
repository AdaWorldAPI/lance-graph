//! Rail-trie geometry — deterministic node placement from the rail registers.
//!
//! **"The address IS the coordinate" — with enough address.** The OGAR canon
//! already says the key prerenders nodes with zero value decode; this module
//! is that sentence made executable for the *rail* registers: a node's rail
//! path (child-index-within-parent per level) maps to a drawing position by
//! pure arithmetic. Level = ring (or column); the slot byte at each level =
//! position within the parent's arc, in the bake's own linearisation order.
//! Two loads render identically; there is no solver, no simulation, no scene
//! model — the geometry is a projection of the row, cacheable like one.
//!
//! ## Two carvings, resolved per class — never assumed
//!
//! The register bytes have more than one sanctioned reading, and which one a
//! row uses is a property of its BAKE, resolved through
//! [`ClassView::rail_carving`](crate::class_view::ClassView::rail_carving) —
//! the same registry-resolution pattern as
//! [`edge_codec_flavor`](crate::class_view::ClassView::edge_codec_flavor):
//!
//! | carving | levels | stride | where |
//! |---|---|---|---|
//! | [`RailCarving::InterleavedPairs`] | 6 | 2 | the key's facet payload, `X:Y` axis pairs |
//! | [`RailCarving::AxisSlab`] | 12 (+12 cont) | 1 | one axis per register, in the value slab |
//!
//! The slab variant is not decoration: a consumer bake **measured the pair
//! reading and rejected it** for its hierarchy (44.25 % of paths fit, vs
//! 99.62 % in twelve per-axis levels). A pair byte is two SEPARATE bytes —
//! never widened to u16; a widened word has no axis.
//!
//! ## What this is NOT
//!
//! Not a renderer. Askama-SVG, a2ui-paint, or any other glove reads ONE
//! resolved placement from here; none of them re-derive it. Not a distance
//! either — the CLAM-side geodesic lives with the compute crate; this module
//! only answers *where a node sits*, from bytes it can name.

/// Which hierarchy axis a rail register carries.
///
/// Generic on purpose: which VERB the taxonomy axis means for a given class
/// (`is_a`, `parent`, `containment`, `refines`, …) is the class's business,
/// resolved where the ClassView lives — never encoded here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RailAxis {
    /// The subsumption chain (is_a-shaped, whatever the class's verb is).
    Taxonomy,
    /// The composition chain (part_of-shaped).
    Mereology,
}

/// Levels per interleaved-pair register (the key facet payload).
pub const RAIL_PAIR_LEVELS: usize = 6;
/// Levels per axis-slab register.
pub const RAIL_SLAB_LEVELS: usize = 12;
/// Maximum addressable depth: a slab register plus its continuation.
pub const RAIL_MAX_DEPTH: usize = 2 * RAIL_SLAB_LEVELS;

/// How a class's rail register is carved — the ClassView's reading, portable.
///
/// Offsets are **row-relative** (`0..512`), because both carvings exist:
/// pairs live in the key (`4..16`), slabs in the value. Selection only —
/// every carving reads within the existing row, so the choice never changes
/// `NODE_ROW_STRIDE` (canon: registry-resolved via `classid → ClassView`,
/// never a stride change).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RailCarving {
    /// Six `(X:Y)` axis pairs, stride 2. `axis_byte` selects the pair lane
    /// (0 = `X`, 1 = `Y`) — two separate bytes, never a widened word.
    InterleavedPairs { reg: usize, axis_byte: u8 },
    /// One axis per register: twelve contiguous level bytes at `reg`, with an
    /// optional (possibly discontiguous) continuation register of twelve more.
    AxisSlab { reg: usize, cont: Option<usize> },
}

impl RailCarving {
    /// The canon zero-fallback: the key facet's interleaved pairs at `4..16`,
    /// Taxonomy on the pair's first byte, Mereology on its second. Correct for
    /// a class whose ClassView has not said otherwise — and ONLY for such a
    /// class; a bake that measured its way to slabs overrides this.
    #[must_use]
    pub const fn zero_fallback(axis: RailAxis) -> Self {
        RailCarving::InterleavedPairs {
            reg: 4,
            axis_byte: match axis {
                RailAxis::Taxonomy => 0,
                RailAxis::Mereology => 1,
            },
        }
    }

    /// Maximum depth this carving can express.
    #[must_use]
    pub const fn max_depth(&self) -> usize {
        match self {
            RailCarving::InterleavedPairs { .. } => RAIL_PAIR_LEVELS,
            RailCarving::AxisSlab { cont: Some(_), .. } => RAIL_MAX_DEPTH,
            RailCarving::AxisSlab { cont: None, .. } => RAIL_SLAB_LEVELS,
        }
    }

    /// The level byte at `i`, or 0 when out of range. Position is the
    /// information: level index maps to a byte offset and nothing else.
    #[must_use]
    fn level(&self, row: &[u8], i: usize) -> u8 {
        let at = match *self {
            RailCarving::InterleavedPairs { reg, axis_byte } => {
                if i >= RAIL_PAIR_LEVELS {
                    return 0;
                }
                reg + 2 * i + axis_byte as usize
            }
            RailCarving::AxisSlab { reg, cont } => {
                if i < RAIL_SLAB_LEVELS {
                    reg + i
                } else if i < RAIL_MAX_DEPTH {
                    match cont {
                        Some(c) => c + (i - RAIL_SLAB_LEVELS),
                        None => return 0,
                    }
                } else {
                    return 0;
                }
            }
        };
        if at < row.len() {
            row[at]
        } else {
            0
        }
    }

    /// Read the occupied leading path out of a row. Stops at the first zero:
    /// a hole ends the chain, and a value after a hole is not ancestry
    /// (`[1, 0, 7]` is depth 1, never 2).
    #[must_use]
    pub fn read_path(&self, row: &[u8]) -> RailPath {
        let mut p = RailPath {
            len: 0,
            slots: [0u8; RAIL_MAX_DEPTH],
        };
        for i in 0..self.max_depth() {
            let v = self.level(row, i);
            if v == 0 {
                break;
            }
            p.slots[i] = v;
            p.len += 1;
        }
        p
    }
}

/// A read rail path: `len` occupied levels, slot byte `i` = child index at
/// level `i` (stored as `1 + index`; 0 is the hole, never a slot).
///
/// The empty path is the **dominant root of the lane** — the zero-fallback
/// reading, not "unknown".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RailPath {
    len: u8,
    slots: [u8; RAIL_MAX_DEPTH],
}

impl RailPath {
    /// Depth = occupied levels; the ring/column the node draws on.
    #[must_use]
    pub fn depth(&self) -> u32 {
        u32::from(self.len)
    }

    /// The occupied slots, root-first.
    #[must_use]
    pub fn slots(&self) -> &[u8] {
        &self.slots[..self.len as usize]
    }

    /// Prefix containment — the trie's own ancestry test.
    #[must_use]
    pub fn is_ancestor_of(&self, other: &RailPath) -> bool {
        self.len <= other.len && self.slots() == &other.slots()[..self.len as usize]
    }

    /// The node's **arc coordinate** in `[0, 1)`: the radix fraction
    /// `Σ slots[i] / 256^(i+1)`, accumulated Horner-style.
    ///
    /// This single number is the neo4j-shaped layout invariant, provable and
    /// tested rather than styled: a child's arc always lands inside its
    /// parent's half-open interval `[arc, arc + 256^-depth)`, siblings order
    /// by slot byte, and the whole placement is a pure function of the row —
    /// two loads render identically.
    ///
    /// **Honest precision boundary:** `f64` carries 52 mantissa bits, so arcs
    /// are exact through level 6 (48 bits) and remain ORDER-preserving but may
    /// stop discriminating between deep siblings past that. A glove that needs
    /// deeper discrimination reads [`slots`](RailPath::slots) directly; the
    /// arc is the cheap common case, not the whole truth.
    #[must_use]
    pub fn arc(&self) -> f64 {
        let mut a = 0.0f64;
        for &s in self.slots().iter().rev() {
            a = (a + f64::from(s)) / 256.0;
        }
        a
    }

    /// The full placement: ring from depth, arc from the slots.
    #[must_use]
    pub fn placement(&self) -> TriePlacement {
        TriePlacement {
            ring: self.depth(),
            arc: self.arc(),
        }
    }
}

/// Where a node draws: `ring` (= trie depth; a radial layout reads it as the
/// ring index, a layered one as the column) and `arc` in `[0, 1)` (angle
/// fraction, or x within the band). One resolved placement, many gloves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriePlacement {
    pub ring: u32,
    pub arc: f64,
}

/// The two-register composition: primary axis places, secondary overlays.
///
/// The reusable pattern for a `2 × 6×2×8bit` (or `2 ×` slab) drawing: the
/// PRIMARY axis (usually Taxonomy) yields the node's placement; the SECONDARY
/// axis (usually Mereology) yields a second path whose edges are drawn OVER
/// that placement — overlay/bundling, never a second coordinate system. Two
/// hierarchies on one canvas stay legible exactly because only one of them
/// places.
#[must_use]
pub fn dual_rail_placement(
    row: &[u8],
    primary: RailCarving,
    secondary: RailCarving,
) -> (TriePlacement, RailPath) {
    (primary.read_path(row).placement(), secondary.read_path(row))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair_row(levels: &[u8], axis_byte: usize) -> Vec<u8> {
        let mut r = vec![0u8; 512];
        for (i, &v) in levels.iter().enumerate() {
            r[4 + 2 * i + axis_byte] = v;
        }
        r
    }

    #[test]
    fn the_hole_rule_ends_the_chain() {
        let c = RailCarving::zero_fallback(RailAxis::Taxonomy);
        assert_eq!(c.read_path(&pair_row(&[1, 2, 3], 0)).depth(), 3);
        assert_eq!(
            c.read_path(&pair_row(&[1, 0, 7], 0)).depth(),
            1,
            "nach dem Loch ist keine Ahnenschaft"
        );
        assert_eq!(
            c.read_path(&pair_row(&[], 0)).depth(),
            0,
            "leerer Pfad = dominante Wurzel"
        );
    }

    #[test]
    fn the_pair_axes_are_two_separate_bytes() {
        let tax = RailCarving::zero_fallback(RailAxis::Taxonomy);
        let mer = RailCarving::zero_fallback(RailAxis::Mereology);
        let mut r = pair_row(&[2, 3], 0);
        r[5] = 7; // Mereology level 0 — a DIFFERENT chain
        assert_eq!(tax.read_path(&r).depth(), 2);
        assert_eq!(mer.read_path(&r).depth(), 1);
        let mut s = r.clone();
        s[5] = 9; // change ONLY the other axis
        assert_eq!(
            tax.read_path(&r),
            tax.read_path(&s),
            "Taxonomy darf Mereology nicht sehen"
        );
        assert_ne!(mer.read_path(&r), mer.read_path(&s));
    }

    /// The slab carving with a discontiguous continuation — the shape a
    /// consumer bake measured its way to. The register BETWEEN the two
    /// (another axis's slab) must be invisible.
    #[test]
    fn the_axis_slab_reads_stride_one_with_discontiguous_continuation() {
        let c = RailCarving::AxisSlab {
            reg: 76,
            cont: Some(100),
        };
        assert_eq!(c.max_depth(), 24);
        let mut r = vec![0u8; 512];
        for i in 0..12 {
            r[76 + i] = 1;
        }
        r[100] = 3;
        r[101] = 5;
        assert_eq!(c.read_path(&r).depth(), 14);
        let mut s = r.clone();
        s[92] = 9; // between the registers — a foreign axis's byte
        assert_eq!(
            c.read_path(&r),
            c.read_path(&s),
            "fremde Register sind unsichtbar"
        );
    }

    /// THE neo4j-shaped invariant, proven not styled: a child's arc lands
    /// inside its parent's half-open interval, and siblings order by slot.
    #[test]
    fn a_childs_arc_lands_inside_its_parents_interval() {
        let c = RailCarving::zero_fallback(RailAxis::Taxonomy);
        let parent = c.read_path(&pair_row(&[3, 5], 0));
        let child = c.read_path(&pair_row(&[3, 5, 2], 0));
        let sib = c.read_path(&pair_row(&[3, 5, 4], 0));
        let stranger = c.read_path(&pair_row(&[9], 0));

        let span = 256f64.powi(-(parent.depth() as i32));
        assert!(parent.arc() <= child.arc() && child.arc() < parent.arc() + span);
        assert!(parent.arc() <= sib.arc() && sib.arc() < parent.arc() + span);
        assert!(child.arc() < sib.arc(), "Geschwister ordnen nach Slot");
        assert!(
            stranger.arc() < parent.arc() || stranger.arc() >= parent.arc() + span,
            "ein Fremder faellt nie ins Intervall"
        );
        assert!(parent.is_ancestor_of(&child) && !parent.is_ancestor_of(&stranger));
    }

    /// Determinism is the whole point: same row, same placement, always.
    #[test]
    fn placement_is_a_pure_function_of_the_row() {
        let c = RailCarving::AxisSlab {
            reg: 76,
            cont: Some(100),
        };
        let mut r = vec![0u8; 512];
        for (i, b) in r.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let (p1, s1) = dual_rail_placement(
            &r,
            c,
            RailCarving::AxisSlab {
                reg: 88,
                cont: None,
            },
        );
        let (p2, s2) = dual_rail_placement(
            &r,
            c,
            RailCarving::AxisSlab {
                reg: 88,
                cont: None,
            },
        );
        assert_eq!(p1, p2);
        assert_eq!(s1, s2);
        assert!(p1.arc >= 0.0 && p1.arc < 1.0);
    }

    /// The documented f64 boundary, pinned as a passing test: exact and
    /// discriminating through level 6; order-preserving at level 7+ for THIS
    /// construction (monotone Horner), with discrimination not guaranteed.
    #[test]
    fn the_arc_is_exact_through_level_six() {
        let c = RailCarving::AxisSlab {
            reg: 76,
            cont: None,
        };
        let mut a = vec![0u8; 512];
        let mut b = vec![0u8; 512];
        for i in 0..6 {
            a[76 + i] = 1;
            b[76 + i] = 1;
        }
        b[76 + 5] = 2; // differ at level 6 exactly
        let (pa, pb) = (c.read_path(&a).arc(), c.read_path(&b).arc());
        assert!(pa < pb, "Level 6 diskriminiert noch exakt");
        assert!((pb - pa - 256f64.powi(-6)).abs() < f64::EPSILON);
    }
}
