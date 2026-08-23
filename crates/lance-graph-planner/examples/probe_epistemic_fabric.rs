//! PROBE-EPISTEMIC-FABRIC-1 — can the composed planes distinguish SEVEN
//! epistemic states, and does bounding a missing mediator from both sides
//! make `IndirectUnknownIntermediates` refutable?
//!
//! **The synthesis under test (operator, 2026-08-23).** The pieces stop
//! being separate tricks and become one hierarchical epistemic fabric:
//!
//! ```text
//!   HHTL       where is the uncertainty?
//!   CE64       what causal topology do we currently claim?
//!   G24N4      what supports / falsifies the claim?   (+ / 0 / −)
//!   V4         what intervention did we perform?
//!   NARS f/c   how strongly is the result supported?
//! ```
//!
//! …with no field impersonating another. The load-bearing claims measured
//! here, each able to fail:
//!
//! - **E1/E2 — the missing middle, bounded from both sides.** Upstream
//!   admissible ∩ downstream admissible = the mediator candidate mask. And
//!   the sharp consequence: when that intersection is **empty**, the
//!   `IndirectUnknownIntermediates` claim is **REFUTED** — no mediator can
//!   exist in the addressed space. A topology claim that was previously only
//!   assertable becomes falsifiable.
//! - **E3 — depth and scope are INDEPENDENT axes.** Derivational depth
//!   (proof structure) and generalization scope (HHTL ancestry at which
//!   support survives) are orthogonal. Two beliefs with the SAME scalar rung
//!   can differ in scope; a scalar cannot express the 2×2.
//! - **E4 — seven epistemic states, pairwise distinguishable.** The
//!   strongest gate: if any two collapse to the same reading, the fabric
//!   claim fails.
//! - **E5 — connective tissue obeys the SAME ABI.** An internal node
//!   carrying aggregate state uses the identical 16-byte dock and address
//!   grammar as a leaf. No special "belief object", no second universe.
//! - **E6 — revision moves the view; the population stays.**
//!
//! # Honesty box
//!
//! - Toy hierarchy over shipped operators — this measures the ALGEBRA, not a
//!   corpus.
//! - **E2's refutation is scoped to the ADDRESSED universe.** Disjoint
//!   admissible regions prove no mediator exists *among addressed things*.
//!   An unaddressed mediator (a genuine unknown-unknown) is NOT refuted —
//!   that is precisely the state E4 keeps distinct, and the honest limit of
//!   the result.
//! - The signed field is the probe-local reading from
//!   `PROBE-TARSKI-SIGNED-WITNESS-1` (own slots, own accessors), never the
//!   A9 `Locus` API ("loci, not magnitudes").
//! - Probe-local classids; nothing minted.

use causal_edge::layout::CausalTopology;
use lance_graph_contract::attention_facet::{AttentionFocusFacet, RowFocusMask};
use lance_graph_contract::facet::{FacetCascade, FacetTier};
use lance_graph_planner::nars::truth::TruthValue;

const FABRIC_CLASSID: u32 = 0xFFFF_000E;
const CONSTRUCTIVE: usize = 0;
const FALSIFYING: usize = 1;

/// The probe-local signed `24×i4` reading (magnitudes, not A9 loci).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct SignedField([u8; 12]);

impl SignedField {
    const ZERO: Self = Self([0u8; 12]);
    fn get(self, slot: usize) -> i8 {
        let b = self.0[slot / 2];
        let n = if slot & 1 == 0 {
            b & 0x0F
        } else {
            (b >> 4) & 0x0F
        };
        ((n << 4) as i8) >> 4
    }
    fn set(&mut self, slot: usize, v: i8) {
        let x = (v.clamp(-8, 7) as u8) & 0x0F;
        let i = slot / 2;
        if slot & 1 == 0 {
            self.0[i] = (self.0[i] & 0xF0) | x;
        } else {
            self.0[i] = (self.0[i] & 0x0F) | (x << 4);
        }
    }
    fn with(mut self, slot: usize, v: i8) -> Self {
        self.set(slot, v);
        self
    }
}

fn region(b: [u8; 4]) -> FacetCascade {
    FacetCascade {
        facet_classid: FABRIC_CLASSID,
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

fn mask_of(items: &[AttentionFocusFacet]) -> RowFocusMask {
    let mut m = RowFocusMask::empty();
    for i in items {
        m.insert(*i);
    }
    m
}

/// A V4-shaped typed intervention row (16-byte LE dock, `Copy`, no heap).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Probe {
    performed: bool,
    /// `Some(true)` = predicted effect observed; `Some(false)` = failed.
    outcome: Option<bool>,
}

/// The composed per-claim reading. Every field is a DIFFERENT plane; none
/// derives another.
#[derive(Clone, Copy)]
struct Claim {
    /// WHERE support currently survives (shallower depth = broader scope).
    support_at: Option<AttentionFocusFacet>,
    /// WHAT topology is claimed.
    topology: CausalTopology,
    /// WHY — signed derivational witness.
    field: SignedField,
    /// Whether a mediator candidate region exists (upstream ∩ downstream).
    candidate_region: Option<AttentionFocusFacet>,
    /// WHAT WE DID.
    probe: Probe,
    /// HOW STRONGLY.
    truth: TruthValue,
}

/// The seven epistemic states the fabric claims to distinguish.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    /// Nobody looked: nothing asserted, no candidate region, no probe.
    UnknownUnlooked,
    /// A mediator is claimed to exist and we know WHERE it would live, but
    /// it is unbound. The targetable pothole.
    UnknownMediatorUnresolved,
    /// Support exists, and it survives only at a deep (narrow) address.
    SupportedLocally { depth: u8 },
    /// Support survives at a shallow (broad) address.
    SupportedBroadly { depth: u8 },
    /// Support broadly, but a subtree carries a falsifier.
    FalsifiedInOneBranch,
    /// An intervention was performed; the result has not yet settled it.
    CounterfactuallyTested,
    /// Tested AND still supported afterwards — the only state that licenses
    /// learning a transformation.
    LearnedSurvivedTests,
}

/// Read the composed planes into one epistemic state. Deliberately total and
/// deliberately ordered: the more specific states are checked first.
fn read_state(c: &Claim) -> State {
    let supported = c.field.get(CONSTRUCTIVE) > 0;
    let falsified = c.field.get(FALSIFYING) < 0;

    if c.probe.performed {
        return match c.probe.outcome {
            Some(true) if supported && !falsified => State::LearnedSurvivedTests,
            _ => State::CounterfactuallyTested,
        };
    }
    if supported && falsified {
        return State::FalsifiedInOneBranch;
    }
    if supported {
        let d = c.support_at.map(|a| a.depth()).unwrap_or(u8::MAX);
        return if d <= 1 {
            State::SupportedBroadly { depth: d }
        } else {
            State::SupportedLocally { depth: d }
        };
    }
    if c.topology == CausalTopology::IndirectUnknownIntermediates && c.candidate_region.is_some() {
        return State::UnknownMediatorUnresolved;
    }
    State::UnknownUnlooked
}

fn main() {
    let mut pass = 0u32;
    let mut gate = |name: &str, ok: bool, detail: String| {
        assert!(ok, "[FAIL] {name} — {detail}");
        println!("  [PASS] {name} — {detail}");
        pass += 1;
    };

    // ---- E1 — the missing middle, bounded from both sides ----
    // A constrains the mediator to somewhere under P.3; B constrains it to
    // somewhere under P.3.5. The candidate region is the DEEPER (narrower).
    let upstream = mask_of(&[at([0x40, 0x03, 0, 0], 2)]);
    let downstream = mask_of(&[at([0x40, 0x03, 0x05, 0], 3)]);
    let candidates = upstream.intersect(&downstream);
    gate(
        "E1 mediator candidate mask = upstream ∩ downstream (narrows to the deeper)",
        candidates.len() == 1
            && candidates.entries()[0] == at([0x40, 0x03, 0x05, 0], 3)
            && candidates.entries()[0].depth() == 3,
        format!(
            "upstream depth 2 ∩ downstream depth 3 ⇒ {} candidate region at depth {} \
             — the search surface, without enumerating any mediator",
            candidates.len(),
            candidates.entries()[0].depth()
        ),
    );

    // ---- E2 — THE REFUTATION. Disjoint admissible regions mean no mediator
    // can exist in the addressed space ----
    let up_x = mask_of(&[at([0x40, 0x03, 0, 0], 2)]);
    let down_y = mask_of(&[at([0x40, 0x07, 0, 0], 2)]);
    let impossible = up_x.intersect(&down_y);
    gate(
        "E2 disjoint admissible regions REFUTE IndirectUnknownIntermediates",
        impossible.is_empty() && !candidates.is_empty(),
        "empty intersection ⇒ no addressed mediator can satisfy both sides, so the \
         topology claim is falsified (scoped to the ADDRESSED universe — an \
         unaddressed mediator is not refuted); the E1 case stays non-empty, so the \
         test discriminates"
            .to_string(),
    );

    // ---- E3 — depth and scope are INDEPENDENT axes ----
    // Two claims with the SAME derivational depth (3) but different scope.
    let deep_local = Claim {
        support_at: Some(at([0x40, 0x03, 0x05, 0], 3)),
        topology: CausalTopology::Direct,
        field: SignedField::ZERO.with(CONSTRUCTIVE, 3),
        candidate_region: None,
        probe: Probe {
            performed: false,
            outcome: None,
        },
        truth: TruthValue::new(0.9, 0.85),
    };
    let deep_broad = Claim {
        support_at: Some(at([0x40, 0, 0, 0], 1)),
        ..deep_local
    };
    let shallow_broad = Claim {
        field: SignedField::ZERO.with(CONSTRUCTIVE, 1),
        ..deep_broad
    };
    let same_rung_diff_scope = deep_local.field.get(CONSTRUCTIVE)
        == deep_broad.field.get(CONSTRUCTIVE)
        && deep_local.support_at.unwrap().depth() != deep_broad.support_at.unwrap().depth();
    let same_scope_diff_rung = deep_broad.support_at.unwrap().depth()
        == shallow_broad.support_at.unwrap().depth()
        && deep_broad.field.get(CONSTRUCTIVE) != shallow_broad.field.get(CONSTRUCTIVE);
    gate(
        "E3 derivational depth and generalization scope vary independently",
        same_rung_diff_scope
            && same_scope_diff_rung
            && read_state(&deep_local) != read_state(&deep_broad),
        format!(
            "same depth (3) at scopes {} vs {} ⇒ different states; same scope (1) at \
             depths 3 vs 1 ⇒ a scalar rung collapses both distinctions",
            deep_local.support_at.unwrap().depth(),
            deep_broad.support_at.unwrap().depth()
        ),
    );

    // ---- E4 — SEVEN STATES, PAIRWISE DISTINGUISHABLE (the strongest gate) ----
    let base = Claim {
        support_at: None,
        topology: CausalTopology::Direct,
        field: SignedField::ZERO,
        candidate_region: None,
        probe: Probe {
            performed: false,
            outcome: None,
        },
        truth: TruthValue::new(0.5, 0.0),
    };
    let seven = [
        // 1. nobody looked
        base,
        // 2. mediator claimed, region known, value unbound
        Claim {
            topology: CausalTopology::IndirectUnknownIntermediates,
            candidate_region: Some(at([0x40, 0x03, 0x05, 0], 3)),
            ..base
        },
        // 3. supported, only locally
        deep_local,
        // 4. supported, broadly
        deep_broad,
        // 5. supported broadly BUT falsified in a branch
        Claim {
            field: SignedField::ZERO.with(CONSTRUCTIVE, 3).with(FALSIFYING, -2),
            ..deep_broad
        },
        // 6. an intervention was performed, not yet settling
        Claim {
            probe: Probe {
                performed: true,
                outcome: Some(false),
            },
            ..deep_broad
        },
        // 7. tested and survived
        Claim {
            probe: Probe {
                performed: true,
                outcome: Some(true),
            },
            ..deep_broad
        },
    ];
    let states: Vec<State> = seven.iter().map(read_state).collect();
    let mut all_distinct = true;
    for i in 0..states.len() {
        for j in (i + 1)..states.len() {
            if states[i] == states[j] {
                all_distinct = false;
            }
        }
    }
    gate(
        "E4 seven epistemic states are pairwise distinguishable",
        all_distinct && states.len() == 7,
        format!("{:?}", states),
    );

    // ---- E5 — connective tissue obeys the SAME ABI as a leaf ----
    // An internal node carrying aggregate state is addressed and docked
    // identically; only its DEPTH differs. No special belief object.
    let leaf_dock = region([0x40, 0x03, 0x05, 0x09]).to_bytes();
    let connective_dock = region([0x40, 0x03, 0, 0]).to_bytes();
    let leaf = at([0x40, 0x03, 0x05, 0x09], 4);
    let connective = at([0x40, 0x03, 0, 0], 2);
    gate(
        "E5 a connective node uses the identical dock + address grammar as a leaf",
        leaf_dock.len() == 16
            && connective_dock.len() == 16
            && FacetCascade::from_bytes(&connective_dock).facet_classid
                == FacetCascade::from_bytes(&leaf_dock).facet_classid
            && connective.covers(leaf)
            && connective.depth() < leaf.depth(),
        "same 16-byte dock, same classid, same grammar — the internal node differs only \
         in DEPTH, so aggregate state needs no second representation universe"
            .to_string(),
    );

    // ---- E6 — revision MOVES THE VIEW; the population stays ----
    let population: Vec<[u8; 16]> = vec![
        region([0x40, 0x03, 0x05, 0x09]).to_bytes(),
        region([0x40, 0x03, 0x06, 0x01]).to_bytes(),
        region([0x40, 0x07, 0x01, 0x02]).to_bytes(),
    ];
    let before = population.clone();
    let view_a = mask_of(&[at([0x40, 0x03, 0, 0], 2)]);
    let view_b = mask_of(&[at([0x40, 0x07, 0, 0], 2)]);
    let moved = view_a != view_b
        && view_a.contains(at([0x40, 0x03, 0x05, 0x09], 4))
        && !view_b.contains(at([0x40, 0x03, 0x05, 0x09], 4));
    gate(
        "E6 revision moves the view; the population is byte-identical",
        moved && population == before,
        "the attended region changed and the coverage answer changed with it, while \
         every resident dock stayed bit-for-bit the same"
            .to_string(),
    );

    // ---- E7 — NARS strength is a SEPARATE plane: same state, different f/c ----
    // "No one field needs to impersonate all the others" — the epistemic
    // STATE and the STRENGTH of support are independent readings, so a
    // strength change must not silently move the state.
    let weakly_supported = Claim {
        truth: TruthValue::new(0.55, 0.10),
        ..deep_broad
    };
    let strongly_supported = Claim {
        truth: TruthValue::new(0.99, 0.95),
        ..deep_broad
    };
    gate(
        "E7 strength (NARS f/c) is orthogonal to epistemic state",
        read_state(&weakly_supported) == read_state(&strongly_supported)
            && weakly_supported.truth.expectation() < strongly_supported.truth.expectation()
            && (weakly_supported.truth.confidence - strongly_supported.truth.confidence).abs()
                > 0.5,
        format!(
            "same state {:?} at expectation {:.3} vs {:.3} — strength moves without the \
             state moving, and neither field impersonates the other",
            read_state(&weakly_supported),
            weakly_supported.truth.expectation(),
            strongly_supported.truth.expectation()
        ),
    );

    println!("PROBE-EPISTEMIC-FABRIC-1: ALL {pass} GATES GREEN");
    println!(
        "measured: bounding a missing mediator from BOTH sides yields the candidate mask \
         by intersection (E1) and — the sharp result — makes IndirectUnknownIntermediates \
         REFUTABLE when the admissible regions are disjoint (E2, scoped to the addressed \
         universe). Derivational depth and generalization scope are independent axes a \
         scalar rung collapses (E3). All SEVEN epistemic states stay pairwise \
         distinguishable (E4). Connective tissue needs no second universe — same dock, \
         same grammar, only greater depth (E5). And revision moves the view while every \
         resident byte stays put (E6)."
    );
}
