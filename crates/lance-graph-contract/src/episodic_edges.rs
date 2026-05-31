//! # `episodic_edges` — AriGraph episodic edges, RISC-encoded (zero-dep).
//!
//! EW64 is **AriGraph's episodic edges** — a mailbox(=episode) is a *basin* with
//! *multiple* edges (NOT a lens over one `CausalEdge64`). This is the witness/
//! relational concern, SoC'd from both the temporal arc (a basin one HHTL level up)
//! and frozen identity (CAM/OGIT).
//!
//! ## Cost model (grounded by the #444 locality probe)
//! - **~98.6% intra-basin** (probe): an edge stays in the row's own family, which is
//!   **inherited** from the HHTL/`class_id` path → ~0 extra bits. `EdgeRef::family == 0`.
//! - **~1.4% cross-family** (the crossover): a **4-bit nibble** (16 families,
//!   `family ∈ 1..=15`) indexes the **OGIT-class-inherited cross-family palette** — a
//!   CAM_PQ facet code whose codebook is the class's declared closed range
//!   (`owl:disjointWith` ⇒ collision-free). The 16 *identities* live in the class,
//!   **never on the edge** (`I-VSA-IDENTITIES`: point, don't copy). Probe fan-out
//!   ≤ 3 ⇒ 4 bits (16) has headroom.
//!
//! ## Layout — `EpisodicEdges64(u64)` = 4 × `u16` slots
//! Each slot: `0x0000` = empty; else `[bits 12-15: family nibble][bits 0-11: local]`.
//! `local` is a **1-based within-family index** (`1..=4095`); the resolved family is
//! the row's own basin (`family == 0`, inherited) or `class.cross_family_palette[family]`
//! (`1..=15`). Cross-session reach is a *separate* 16-bit episode-store column, not this
//! word. Identity resolution flies ABOVE the row (the OGIT class), as `class_view` does.

// The slot pack/unpack does intentional nibble extraction (slot>>12 ∈ 0..=15) and
// low-16-bit reads (u64 -> u16); both are provably-bounded narrowings.
#![allow(clippy::cast_possible_truncation)]

/// One episodic edge: a `(family, local)` reference in the episodic basin space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeRef {
    /// Cross-family selector. `0` = **intra-basin** (the row's own family, inherited
    /// from the HHTL/`class_id` path — the ~98.6% common case). `1..=15` = a
    /// cross-family index into the OGIT-class-inherited palette (the ~1.4% crossover).
    pub family: u8,
    /// 1-based within-family local index (`1..=4095`); `0` is the empty-slot sentinel.
    pub local: u16,
}

impl EdgeRef {
    /// Family count addressable by the 4-bit nibble (probe fan-out ≤ 3 ⇒ headroom).
    pub const FAMILIES: u8 = 16;
    /// Max 1-based within-family local index (12 bits).
    pub const MAX_LOCAL: u16 = 0x0FFF;

    /// A validated edge, or `None` if `family ≥ 16` or `local ∉ 1..=4095`.
    #[must_use]
    pub const fn new(family: u8, local: u16) -> Option<Self> {
        if family < Self::FAMILIES && local >= 1 && local <= Self::MAX_LOCAL {
            Some(Self { family, local })
        } else {
            None
        }
    }

    /// An **intra-basin** edge (`family == 0`, the inherited common case).
    #[must_use]
    pub const fn intra(local: u16) -> Option<Self> {
        Self::new(0, local)
    }

    /// A **cross-family** edge into palette index `family ∈ 1..=15`.
    #[must_use]
    pub const fn cross(family: u8, local: u16) -> Option<Self> {
        if family == 0 {
            None
        } else {
            Self::new(family, local)
        }
    }

    /// Does this edge cross to another family (vs. staying intra-basin)?
    #[must_use]
    pub const fn is_cross(self) -> bool {
        self.family != 0
    }

    const fn to_slot(self) -> u16 {
        (u16::from(self.family) << 12) | (self.local & Self::MAX_LOCAL)
    }

    const fn from_slot(slot: u16) -> Option<Self> {
        if slot == 0 {
            None
        } else {
            Some(Self { family: (slot >> 12) as u8, local: slot & Self::MAX_LOCAL })
        }
    }
}

/// Up to 4 AriGraph episodic edges packed into one `u64` (4 × 16-bit slots).
///
/// The witness/relational column of the per-row SoA: which other basin members this
/// episode touched. Agnostic — the nibble's *meaning* resolves in the OGIT class, not
/// here. `Default` / [`EpisodicEdges64::empty`] is the no-edge word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct EpisodicEdges64(pub u64);

impl EpisodicEdges64 {
    /// Edge slots per word (4 × 16 bits = 64).
    pub const CAPACITY: usize = 4;

    /// The empty (no-edge) word.
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// The edge in slot `i` (`0..4`), or `None` if the slot is empty / out of range.
    #[must_use]
    pub const fn edge(self, i: usize) -> Option<EdgeRef> {
        if i >= Self::CAPACITY {
            return None;
        }
        EdgeRef::from_slot((self.0 >> (i * 16)) as u16)
    }

    /// How many slots carry an edge.
    #[must_use]
    pub fn count(self) -> usize {
        (0..Self::CAPACITY).filter(|&i| self.edge(i).is_some()).count()
    }

    /// All 4 slots full?
    #[must_use]
    pub fn is_full(self) -> bool {
        self.count() == Self::CAPACITY
    }

    /// Place `e` into the first empty slot; `None` if the word is already full.
    #[must_use]
    pub fn push(self, e: EdgeRef) -> Option<Self> {
        let mut i = 0;
        while i < Self::CAPACITY {
            if self.edge(i).is_none() {
                let shift = i * 16;
                let cleared = self.0 & !(0xFFFF_u64 << shift);
                return Some(Self(cleared | (u64::from(e.to_slot()) << shift)));
            }
            i += 1;
        }
        None
    }

    /// Iterate the present edges in slot order.
    pub fn iter(self) -> impl Iterator<Item = EdgeRef> {
        (0..Self::CAPACITY).filter_map(move |i| self.edge(i))
    }

    /// Count of cross-family edges (the crossover load — the ~1.4% the probe measured).
    #[must_use]
    pub fn cross_count(self) -> usize {
        self.iter().filter(|e| e.is_cross()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edgeref_new_validates_family_and_local() {
        assert!(EdgeRef::new(0, 1).is_some());
        assert!(EdgeRef::new(15, 4095).is_some());
        assert!(EdgeRef::new(16, 1).is_none());
        assert!(EdgeRef::new(0, 0).is_none());
        assert!(EdgeRef::new(0, 4096).is_none());
        assert!(EdgeRef::cross(0, 5).is_none());
        assert_eq!(EdgeRef::intra(5).unwrap().family, 0);
    }

    #[test]
    fn slot_roundtrip_and_empty_sentinel() {
        for family in 0..16u8 {
            for &local in &[1u16, 2, 100, 4095] {
                let e = EdgeRef::new(family, local).unwrap();
                assert_eq!(EdgeRef::from_slot(e.to_slot()).unwrap(), e);
            }
        }
        assert_eq!(EdgeRef::from_slot(0), None);
    }

    #[test]
    fn push_count_and_full() {
        let mut w = EpisodicEdges64::empty();
        assert_eq!(w.count(), 0);
        for k in 1..=4u16 {
            w = w.push(EdgeRef::intra(k).unwrap()).expect("fits");
        }
        assert_eq!(w.count(), 4);
        assert!(w.is_full());
        assert!(w.push(EdgeRef::intra(5).unwrap()).is_none());
    }

    #[test]
    fn edge_index_out_of_range_is_none() {
        let w = EpisodicEdges64::empty().push(EdgeRef::intra(7).unwrap()).unwrap();
        assert_eq!(w.edge(0), EdgeRef::intra(7));
        assert_eq!(w.edge(1), None);
        assert_eq!(w.edge(EpisodicEdges64::CAPACITY), None);
    }

    #[test]
    fn intra_is_cheap_default_cross_is_the_nibble() {
        let w = EpisodicEdges64::empty()
            .push(EdgeRef::intra(10).unwrap())
            .unwrap()
            .push(EdgeRef::intra(11).unwrap())
            .unwrap()
            .push(EdgeRef::intra(12).unwrap())
            .unwrap()
            .push(EdgeRef::cross(3, 7).unwrap())
            .unwrap();
        assert_eq!(w.count(), 4);
        assert_eq!(w.cross_count(), 1);
        assert!(!w.edge(0).unwrap().is_cross());
        assert!(w.edge(3).unwrap().is_cross());
        assert_eq!(w.edge(3).unwrap().family, 3);
    }

    #[test]
    fn iter_yields_present_edges_in_order() {
        let w = EpisodicEdges64::empty()
            .push(EdgeRef::intra(1).unwrap())
            .unwrap()
            .push(EdgeRef::cross(2, 9).unwrap())
            .unwrap();
        let got: Vec<_> = w.iter().collect();
        assert_eq!(got, vec![EdgeRef::intra(1).unwrap(), EdgeRef::cross(2, 9).unwrap()]);
    }

    #[test]
    fn word_is_exactly_64_bits() {
        assert_eq!(core::mem::size_of::<EpisodicEdges64>(), 8);
    }
}
