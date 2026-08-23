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
//!   admissible ∩ downstream admissible = the mediator candidate mask. An
//!   empty intersection makes `IndirectUnknownIntermediates`
//!   **CONDITIONALLY falsifiable** — but ONLY under a completeness warrant
//!   on BOTH masks. **E2c proves the premise is load-bearing** by
//!   constructing incomplete masks whose intersection is empty while the
//!   COMPLETE masks overlap: absence of overlap is evidence of absence only
//!   when the search universe is proven closed.
//! - **E3 — depth and scope are INDEPENDENT axes.** Derivational depth
//!   (proof structure) and generalization scope (HHTL ancestry at which
//!   support survives) are orthogonal. Two beliefs with the SAME scalar rung
//!   can differ in scope; a scalar cannot express the 2×2.
//! - **E4 — the planes ENCODE seven pairwise-distinguishable readings.** If
//!   any two collapsed, the separation claim would fail. This is a
//!   representation-separation result, NOT a canonical state machine — see
//!   the honesty box.
//! - **E5 — connective tissue needs no new ADDRESS universe.** An internal
//!   node reuses the identical 16-byte dock and address grammar as a leaf;
//!   the aggregate planes attach to that same identity. No special "belief
//!   object" — and no claim they all fit inside the 16 bytes.
//! - **E6 — revision moves the view; the population stays.**
//!
//! # Honesty box
//!
//! - Toy hierarchy over shipped operators — this measures the ALGEBRA, not a
//!   corpus.
//! - **E2's refutation has TWO preconditions, not one.** (a) It is scoped to
//!   the ADDRESSED universe — an unaddressed mediator (a genuine
//!   unknown-unknown) is never refuted, which is why E4 keeps
//!   `UnknownUnlooked` distinct. (b) It requires a **closure receipt** on
//!   each admissible mask: the mask must enumerate EVERY addressed candidate
//!   under the declared universe, not merely the ones found so far. A merely
//!   SOUND mask is not enough — see E2c.
//! - **E4 is a representation-separation result, not a state machine.** It
//!   shows the planes carry enough independent information to encode and
//!   distinguish seven readings. It does NOT show the seven are exhaustive
//!   or canonical, nor that production transitions reach them:
//!   `LearnedSurvivedTests` is reached because the fixture sets the probe
//!   fields, which proves an ADMISSION PREDICATE is expressible, not that a
//!   behavioural learner exists.
//! - **E5 proves address reuse, not payload collapse.** A connective node
//!   needs no new address universe; the aggregate planes ATTACH to that same
//!   canonical identity. It does not claim truth + witness + provenance +
//!   coverage all physically fit inside one 16-byte dock.
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

/// **A closure receipt** — the warrant that an admissible mask enumerates
/// EVERY addressed candidate under `universe`, not merely those found so far.
///
/// This is the premise that separates a falsifier from a search accelerator.
/// A mask can be perfectly SOUND (every entry really is admissible) and still
/// be INCOMPLETE (admissible regions it never visited), and an empty
/// intersection of two incomplete masks says nothing about existence.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ClosureReceipt {
    universe: AttentionFocusFacet,
    /// Enumeration over `universe` is proven complete.
    complete: bool,
}

/// What an EMPTY intersection is allowed to mean.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AbsenceVerdict {
    /// Both masks closed over the declared universe ⇒ absence IS evidence.
    TopologyRefuted,
    /// Empty, but at least one side lacks closure ⇒ absence is only a
    /// pothole: gather more evidence, do not conclude.
    PotholeGatherMore,
}

/// The knife edge: an empty region is a falsifier only when the search
/// universe is proven closed enough for absence to mean absence.
fn absence_verdict(
    intersection: &RowFocusMask,
    up: ClosureReceipt,
    down: ClosureReceipt,
) -> Option<AbsenceVerdict> {
    if !intersection.is_empty() {
        return None; // not an absence claim at all
    }
    if up.complete && down.complete {
        Some(AbsenceVerdict::TopologyRefuted)
    } else {
        Some(AbsenceVerdict::PotholeGatherMore)
    }
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

    // ---- E2 — CONDITIONAL refutation. Disjoint admissible regions refute
    // the topology claim ONLY under a closure warrant on both masks ----
    let declared_universe = at([0x40, 0, 0, 0], 1);
    let closed = ClosureReceipt {
        universe: declared_universe,
        complete: true,
    };
    let open = ClosureReceipt {
        universe: declared_universe,
        complete: false,
    };

    let up_x = mask_of(&[at([0x40, 0x03, 0, 0], 2)]);
    let down_y = mask_of(&[at([0x40, 0x07, 0, 0], 2)]);
    let impossible = up_x.intersect(&down_y);

    // E2a — CAN-FIRE: both masks closed ⇒ absence is evidence.
    gate(
        "E2a empty intersection + BOTH closure receipts ⇒ topology REFUTED",
        impossible.is_empty()
            && absence_verdict(&impossible, closed, closed)
                == Some(AbsenceVerdict::TopologyRefuted)
            && absence_verdict(&candidates, closed, closed).is_none(),
        "with enumeration proven complete on both sides, no addressed mediator can \
         satisfy both ⇒ the claim is falsified; the non-empty E1 case yields no absence \
         verdict at all, so the test discriminates"
            .to_string(),
    );

    // E2b — CAN-STAY-SILENT: the SAME empty intersection, one receipt missing.
    gate(
        "E2b the SAME empty intersection without closure is only a POTHOLE",
        absence_verdict(&impossible, open, closed) == Some(AbsenceVerdict::PotholeGatherMore)
            && absence_verdict(&impossible, closed, open)
                == Some(AbsenceVerdict::PotholeGatherMore)
            && absence_verdict(&impossible, open, open) == Some(AbsenceVerdict::PotholeGatherMore),
        "identical geometry, different verdict — the closure receipt is load-bearing, \
         not decoration: drop it on EITHER side and the refutation degrades to \
         gather-more-evidence"
            .to_string(),
    );

    // E2c — WHY the premise is required, demonstrated rather than asserted.
    // Construct COMPLETE masks that DO overlap, then sound-but-incomplete
    // subsets of each whose intersection is empty. Reading that emptiness as
    // refutation would be a false negative about a mediator that exists.
    let up_complete = mask_of(&[at([0x40, 0x03, 0, 0], 2), at([0x40, 0x07, 0, 0], 2)]);
    let down_complete = mask_of(&[at([0x40, 0x07, 0, 0], 2), at([0x40, 0x09, 0, 0], 2)]);
    let truth_overlap = up_complete.intersect(&down_complete); // {P.7} — it EXISTS
    let up_partial = mask_of(&[at([0x40, 0x03, 0, 0], 2)]); // sound ⊂ complete
    let down_partial = mask_of(&[at([0x40, 0x09, 0, 0], 2)]); // sound ⊂ complete
    let partial_overlap = up_partial.intersect(&down_partial); // ∅ — but WRONGLY so
    gate(
        "E2c incomplete masks can be EMPTY while the complete masks OVERLAP",
        truth_overlap.len() == 1
            && truth_overlap.entries()[0] == at([0x40, 0x07, 0, 0], 2)
            && partial_overlap.is_empty()
            && absence_verdict(&partial_overlap, open, open)
                == Some(AbsenceVerdict::PotholeGatherMore),
        format!(
            "complete U ∩ D = {} region (a mediator EXISTS at P.7), yet sound-but-\
             incomplete subsets intersect to ∅ — so `U' ∩ D' = ∅` does NOT entail \
             `U ∩ D = ∅`, and without closure the correct verdict is {:?}",
            truth_overlap.len(),
            AbsenceVerdict::PotholeGatherMore
        ),
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
        "E4 the planes ENCODE seven pairwise-distinguishable readings \
         (representation separation, not an exhaustive state machine)",
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
        "E5 a connective node needs NO new address universe (planes attach to \
         the same canonical identity)",
        leaf_dock.len() == 16
            && connective_dock.len() == 16
            && FacetCascade::from_bytes(&connective_dock).facet_classid
                == FacetCascade::from_bytes(&leaf_dock).facet_classid
            && connective.covers(leaf)
            && connective.depth() < leaf.depth(),
        "same 16-byte dock, same classid, same grammar — the internal node differs only \
         in DEPTH. The aggregate planes (truth / witness / provenance / coverage) ATTACH \
         to that identity; this does NOT claim they all fit inside the 16 bytes"
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
         by intersection without enumerating any mediator (E1). An empty intersection \
         refutes IndirectUnknownIntermediates ONLY under closure receipts on BOTH masks \
         (E2a); the SAME emptiness without them is a pothole (E2b); and E2c shows why — \
         sound-but-incomplete masks intersect to ∅ while the COMPLETE masks overlap, so \
         absence of overlap is evidence of absence only when the universe is proven \
         closed. Depth and scope are independent axes a scalar rung collapses (E3). The \
         planes ENCODE seven pairwise-distinguishable readings (E4 — representation \
         separation, not an exhaustive state machine). Connective nodes need no new \
         address universe (E5). Revision moves the view while every resident byte stays \
         put (E6), and strength moves without the state moving (E7)."
    );
}
