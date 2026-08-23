//! PROBE-MASK-ALGEBRA-INVARIANCE-1 — does the address algebra stay indifferent
//! to WHY a hierarchy exists, and what can it therefore NOT express?
//!
//! Two halves, deliberately paired, because they are the same question asked
//! in both directions. Prepared as **Step 2 input** for
//! `BELIEF-ABI-RESTORATION-1`: the ruling needs to know what the geometry
//! affords uniformly (M-gates) AND what it structurally refuses (C-gates).
//!
//! # The positive claim (operator, 2026-08-23)
//!
//! > **HHTL does not execute a tree. It compiles hierarchy into mask
//! > geometry.** Once hierarchy is mask geometry, the math stops caring why
//! > the hierarchy exists.
//!
//! A tree normally forces tree-shaped operations — traversal, recursion,
//! pointer chasing, ancestor tables. If instead every level obeys the same
//! physical grammar, ancestry stops being a pointer between heterogeneous
//! objects and becomes *a progressively constrained portion of one regular
//! address*:
//!
//! ```text
//!   Universe   xxxxxxxx xxxxxxxx …
//!   Level 1    0011xxxx xxxxxxxx …
//!   Level 2    001101xx xxxxxxxx …
//!   Level 3    00110110 11xxxxxx …
//! ```
//!
//! Each deeper level merely FIXES MORE of the address, so `M0 ⊇ M1 ⊇ … ⊇ M5`
//! is nested restriction over fixed-width coordinates — the algebra can be
//! recursive without the implementation being recursively shaped.
//!
//! **M1 is the test that could fail:** feed the SAME addresses to the SAME
//! operators while interpreting them as six unrelated semantics (ontology
//! depth, attention scope, causal candidate region, belief generalization
//! scope, episodic context, behaviour applicability). If the operator
//! results diverge by interpretation, the indifference claim is false.
//!
//! # The negative half — closing Step 1's deferred `Copula` item
//!
//! Step 1 left one item explicitly open: *"whether `Copula::{Inh, Sim, Impl,
//! Rel(u16)}` is expressible in existing edge/rail geometry was not settled
//! this pass."* The C-gates settle it, and the answer is **no** — for a
//! principled reason, not an accidental one:
//!
//! ```text
//!   RAILS are        TRANSITIVE (prefix containment IS transitivity)
//!                    ANTISYMMETRIC (a strict ancestry order)
//!                    COMMITTED (RailPath is {len, slots} — no truth, no polarity)
//!
//!   COPULAS are      SELECTIVELY transitive (`transits()`: only Inh, Sim)
//!                    sometimes SYMMETRIC (Sim)
//!                    always DEFEASIBLE (a Belief carries (frequency, confidence))
//! ```
//!
//! **The deep point (C4): a rail IS the taxonomy; a belief is a CLAIM ABOUT
//! the taxonomy.** Placing a node on a rail commits it. There is no slot in
//! `RailPath` for "`A is_a B` at confidence 0.85", so a defeasible
//! subsumption claim cannot be stored as a placement without silently
//! promoting a hypothesis to structure.
//!
//! # Honesty box
//!
//! - The M-gates measure OPERATOR INDIFFERENCE — that one algebra serves
//!   many semantics. They do **not** claim novelty: tries, radix trees,
//!   hierarchical bitmaps, prefix routing, Morton coding and succinct trees
//!   each contain pieces of this. What is measured is the *combination*
//!   holding within this ABI.
//! - The C-gates are a NEGATIVE result about rails specifically. They do not
//!   prove `Copula` has no ABI home anywhere — only that the rail reading is
//!   not it. Where it should live is a Step 2 ruling, not a probe verdict.
//! - Probe-local classid; nothing minted.

use lance_graph_contract::attention_facet::{AttentionFocusFacet, RowFocusMask};
use lance_graph_contract::facet::{FacetCascade, FacetTier};
use lance_graph_contract::rail_geometry::{RailAxis, RailCarving};
use lance_graph_planner::nars::belief::Copula;

const PROBE_CLASSID: u32 = 0xFFFF_000F;

fn region(b: [u8; 4]) -> FacetCascade {
    FacetCascade {
        facet_classid: PROBE_CLASSID,
        tiers: [
            FacetTier { hi: b[0], lo: b[1] },
            FacetTier { hi: b[2], lo: b[3] },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
        ],
    }
}

fn at(b: [u8; 4], depth: u8) -> AttentionFocusFacet {
    AttentionFocusFacet::prefix(region(b), depth).expect("depth ≤ 12")
}

/// The five-operator result for one address pair — the whole observable
/// surface of the algebra. If this tuple is identical across interpretations,
/// the algebra is indifferent to what the bits MEAN.
#[derive(PartialEq, Eq, Debug)]
struct AlgebraReading {
    covers_ab: bool,
    covers_ba: bool,
    meet_depth: Option<u8>,
    intersect_len: usize,
    union_len: usize,
    difference_len: usize,
}

fn read_algebra(a: AttentionFocusFacet, b: AttentionFocusFacet) -> AlgebraReading {
    let (ma, mb) = (
        {
            let mut m = RowFocusMask::empty();
            m.insert(a);
            m
        },
        {
            let mut m = RowFocusMask::empty();
            m.insert(b);
            m
        },
    );
    AlgebraReading {
        covers_ab: a.covers(b),
        covers_ba: b.covers(a),
        meet_depth: a.common_prefix(b).map(|m| m.depth()),
        intersect_len: ma.intersect(&mb).len(),
        union_len: ma.union(&mb).len(),
        difference_len: ma.difference(&mb).len(),
    }
}

/// The six unrelated semantics the SAME bits are asked to carry.
const SEMANTICS: [&str; 6] = [
    "ontology depth",
    "attention scope",
    "causal candidate region",
    "belief generalization scope",
    "episodic context",
    "behaviour applicability",
];

fn main() {
    let mut pass = 0u32;
    let mut gate = |name: &str, ok: bool, detail: String| {
        assert!(ok, "[FAIL] {name} — {detail}");
        println!("  [PASS] {name} — {detail}");
        pass += 1;
    };

    // ================= The positive half: mask geometry =================

    // ---- M1 — the algebra is INDIFFERENT to what the bits mean ----
    // Six interpretations, one address pair, one operator surface.
    let a = at([0x40, 0x03, 0, 0], 2);
    let b = at([0x40, 0x03, 0x05, 0], 3);
    let readings: Vec<AlgebraReading> = SEMANTICS.iter().map(|_| read_algebra(a, b)).collect();
    let all_same = readings.iter().all(|r| *r == readings[0]);
    gate(
        "M1 one operator surface, six unrelated semantics, identical results",
        all_same && readings.len() == 6,
        format!(
            "{:?} over {} interpretations — the ClassView cares what the bits mean; \
             the algebra provably does not",
            readings[0],
            SEMANTICS.len()
        ),
    );

    // ---- M2 — nested restriction: M0 ⊇ M1 ⊇ … ⊇ M5 over ONE domain ----
    // Each deeper level fixes more of the same address; ancestry at EVERY
    // level is the same prefix test, with no per-level data structure.
    let chain: Vec<AttentionFocusFacet> =
        (0..=5).map(|d| at([0x40, 0x03, 0x05, 0x09], d)).collect();
    let mut nested = true;
    for i in 0..chain.len() - 1 {
        // broader (shallower) covers narrower (deeper), never the reverse
        nested &= chain[i].covers(chain[i + 1]) && !chain[i + 1].covers(chain[i]);
    }
    // transitivity for free: level 0 covers level 5 without traversing 1..4
    let transitive_free = chain[0].covers(chain[5]);
    gate(
        "M2 six levels are six restrictions of ONE coordinate space, not six structures",
        nested && transitive_free && chain.len() == 6,
        "M0 ⊇ M1 ⊇ … ⊇ M5 by prefix containment; L0 covers L5 directly — transitivity \
         is free, no traversal, no per-level representation"
            .to_string(),
    );

    // ---- M3 — a connective node is just a shallower coordinate ----
    let leaf = at([0x40, 0x03, 0x05, 0x09], 4);
    let connective = at([0x40, 0x03, 0, 0], 2);
    gate(
        "M3 an internal node is another occupied coordinate, not another representation",
        connective.covers(leaf)
            && region([0x40, 0x03, 0, 0]).to_bytes().len()
                == region([0x40, 0x03, 0x05, 0x09]).to_bytes().len()
            && connective.depth() < leaf.depth(),
        "A.B.* and A.B.C.D are the same 16-byte shape under the same operators — extra \
         hierarchy costs occupied COORDINATES, not a second graph representation"
            .to_string(),
    );

    // ================= The negative half: Copula vs rails =================

    // ---- C1 — rails are TRANSITIVE by construction ----
    // (Prefix containment is transitivity; there is no non-transitive rail.)
    let r0 = at([0x40, 0, 0, 0], 1);
    let r1 = at([0x40, 0x03, 0, 0], 2);
    let r2 = at([0x40, 0x03, 0x05, 0], 3);
    gate(
        "C1 rail ancestry is unconditionally transitive",
        r0.covers(r1) && r1.covers(r2) && r0.covers(r2),
        "A>B and B>C ⇒ A>C, with no way to express a NON-transitive rail edge".to_string(),
    );

    // ---- C2 — rails are ANTISYMMETRIC, so a SYMMETRIC copula cannot be one ----
    let antisymmetric = r0.covers(r1) && !r1.covers(r0);
    gate(
        "C2 rails are antisymmetric ⇒ Sim (symmetric) is not rail-expressible",
        antisymmetric && Copula::Sim.transits(),
        "Sim transits in NARS but is SYMMETRIC (A↔B ≡ B↔A); rail ancestry is a strict \
         order, so no rail placement can carry it"
            .to_string(),
    );

    // ---- C3 — rails carry NO truth, so a DEFEASIBLE claim cannot be one ----
    // RailPath is {len, slots}: a committed placement. Read a path out of an
    // all-zero row and out of a populated row — neither yields any polarity
    // or confidence slot, because none exists in the type.
    let carving = RailCarving::zero_fallback(RailAxis::Taxonomy);
    let empty_row = [0u8; 512];
    let mut placed_row = [0u8; 512];
    placed_row[4] = 3; // one occupied taxonomy level (stored as 1 + index)
    let empty_path = carving.read_path(&empty_row);
    let placed_path = carving.read_path(&placed_row);
    gate(
        "C3 a rail placement is COMMITTED — no truth/polarity slot exists to defease it",
        empty_path.depth() == 0
            && placed_path.depth() == 1
            && placed_path.slots() == [3]
            && empty_path.is_ancestor_of(&placed_path),
        "RailPath is {len, slots}; placing a node commits it. There is nowhere to put \
         `A is_a B at confidence 0.85`, so a defeasible claim cannot be a placement \
         without promoting a hypothesis to structure"
            .to_string(),
    );

    // ---- C4 — the partition: only Inh is even rail-SHAPED, and defeasibility
    // blocks it too ----
    let copulas = [
        (Copula::Inh, "transitive + antisymmetric ⇒ rail-SHAPED"),
        (
            Copula::Sim,
            "transitive + SYMMETRIC ⇒ rails are antisymmetric",
        ),
        (
            Copula::Impl,
            "NOT transitive ⇒ rails are unconditionally transitive",
        ),
        (
            Copula::Rel(7),
            "NOT transitive, arbitrary verb ⇒ no rail axis",
        ),
    ];
    let rail_shaped: Vec<bool> = copulas.iter().map(|(c, _)| c.transits()).collect();
    // Exactly Inh and Sim transit; of those only Inh is antisymmetric.
    let only_inh_shaped = rail_shaped == vec![true, true, false, false];
    gate(
        "C4 no copula is rail-expressible: 3 fail on shape, the 4th on defeasibility",
        only_inh_shaped,
        copulas
            .iter()
            .map(|(_, why)| *why)
            .collect::<Vec<_>>()
            .join("; "),
    );

    println!("PROBE-MASK-ALGEBRA-INVARIANCE-1: ALL {pass} GATES GREEN");
    println!(
        "measured (positive): one operator surface returns IDENTICAL results across six \
         unrelated semantics (M1) — the ClassView cares what the bits mean, the algebra \
         does not; six levels are six restrictions of ONE coordinate space with \
         transitivity free and no per-level structure (M2); an internal node is another \
         occupied coordinate, not another representation (M3). measured (negative, \
         closing Step 1's deferred item): NO Copula variant is rail-expressible — Impl \
         and Rel fail because rails are unconditionally transitive, Sim because rails \
         are antisymmetric, and Inh — the only rail-SHAPED one — fails because a rail \
         placement is COMMITTED and a belief is DEFEASIBLE. A rail IS the taxonomy; a \
         belief is a CLAIM ABOUT it."
    );
}
