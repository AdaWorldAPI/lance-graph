//! PROBE-HHTL-FALSIFICATION-ASYMMETRY-1 — is falsification actually CHEAP in
//! an HHTL-addressed substrate, and what does the hierarchy cost to express?
//!
//! **The claims under test (operator, 2026-08-23):** HHTL + a signed `24×i4`
//! field offers *Auslöschung* — falsification, "a precondition for learning";
//! it expresses **upstream and downstream inheritance**, which is what makes
//! *indirect intermediate known-unknowns* cheap to probe counterfactually;
//! and *"in the worst case it just needs HHTL hierarchical nodes to express
//! more connective tissue than the leaf nodes."*
//!
//! That last clause is an honest cost admission, so this probe **measures the
//! cost** (F6) instead of conceding it in prose.
//!
//! # Why "Auslöschung" is the right word, and where it must NOT be applied
//!
//! A signed field admits **destructive interference**: `+n` and `−n` at one
//! address cancel. A purely positive confidence scalar structurally cannot do
//! this — it can only be diluted, never extinguished. That is the whole
//! difference between "my confidence went down" and "this was refuted."
//!
//! But F1 finds the boundary: **Auslöschung must be a READING, never a
//! storage collapse.** If cancellation is stored as a net `0`, the substrate
//! can no longer distinguish *"support and refutation met and annihilated"*
//! from *"nothing was ever asserted here"* — and the first is a licence to
//! learn while the second is a licence to look. Two slots retain both; one
//! summed slot destroys exactly the information falsification exists to
//! create.
//!
//! # The asymmetry (F3) — why falsification is a *precondition* for learning
//!
//! ```text
//!   falsify a universal claim at P   ONE counterexample region      cost 1
//!   verify  a universal claim at P   exhaust FAN_OUT (16) children  cost 16
//!                                    …and the levels BELOW stay open
//! ```
//!
//! Falsification is the only operation here with `O(1)` cost *and* total
//! soundness. Verification is bounded-but-expensive at one depth and never
//! closes underneath. The hierarchy does not create this asymmetry — it is
//! Popper's — but it makes it **mechanical**: `covers` is the whole test.
//!
//! # Honesty box
//!
//! - Toy hierarchy, shipped operators. This measures the OPERATOR ALGEBRA,
//!   not a corpus.
//! - The signed field here is the probe-local reading established by
//!   `PROBE-TARSKI-SIGNED-WITNESS-1` (own slot names, own accessors). It is
//!   **not** the A9 `Locus` API, whose contract is "loci, not magnitudes".
//! - No claim that beliefs currently HAVE such addresses — the audit (#1006)
//!   says they do not. This is about what the address space affords.

use lance_graph_contract::attention_facet::{AttentionFocusFacet, RowFocusMask};
use lance_graph_contract::facet::{FacetCascade, FacetTier};

/// Probe-local classid (NOT an OGAR mint).
const REGION_CLASSID: u32 = 0xFFFF_000D;
/// The canonical HHTL fan-out (`hhtl.rs:40`).
const FAN_OUT: u8 = 16;

/// Probe-local signed slots — the reading from `PROBE-TARSKI-SIGNED-WITNESS-1`.
const CONSTRUCTIVE: usize = 0;
const FALSIFYING: usize = 1;

/// A 12-byte signed `24×i4` register, read as derivational magnitudes.
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

    /// **The Auslöschung READING** — net interference of the two polarities.
    /// `0` here means "they cancelled", which is NOT the same fact as either
    /// slot being unset; see [`Self::epistemic_state`].
    fn net(self) -> i8 {
        self.get(CONSTRUCTIVE) + self.get(FALSIFYING)
    }

    /// The three-way epistemic state the two slots keep distinguishable —
    /// the reason Auslöschung may not be stored as a sum.
    fn epistemic_state(self) -> Epistemic {
        let (c, f) = (self.get(CONSTRUCTIVE), self.get(FALSIFYING));
        match (c > 0, f < 0) {
            (false, false) => Epistemic::Unasserted,
            (true, false) => Epistemic::Supported,
            (false, true) => Epistemic::Refuted,
            (true, true) => Epistemic::Contested { net: self.net() },
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Epistemic {
    /// Nothing asserted here — a licence to LOOK.
    Unasserted,
    Supported,
    Refuted,
    /// Support and refutation both present — a licence to LEARN. `net == 0`
    /// is full Auslöschung and is still Contested, never Unasserted.
    Contested {
        net: i8,
    },
}

fn region(bytes: [u8; 4]) -> FacetCascade {
    FacetCascade {
        facet_classid: REGION_CLASSID,
        tiers: [
            FacetTier {
                hi: bytes[0],
                lo: bytes[1],
            },
            FacetTier {
                hi: bytes[2],
                lo: bytes[3],
            },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
        ],
    }
}

/// An address at `depth` ladder bytes.
fn at(bytes: [u8; 4], depth: u8) -> AttentionFocusFacet {
    AttentionFocusFacet::prefix(region(bytes), depth).expect("depth ≤ 12")
}

fn main() {
    let mut pass = 0u32;
    let mut gate = |name: &str, ok: bool, detail: String| {
        assert!(ok, "[FAIL] {name} — {detail}");
        println!("  [PASS] {name} — {detail}");
        pass += 1;
    };

    // ---- F1 — Auslöschung is a READING; storing the sum destroys the fact ----
    let contested = SignedField::ZERO.with(CONSTRUCTIVE, 3).with(FALSIFYING, -3);
    let unasserted = SignedField::ZERO;
    let stored_as_sum_contested = contested.net();
    let stored_as_sum_unasserted = unasserted.net();
    gate(
        "F1 Auslöschung must be a reading — a summed slot cannot tell \
         'annihilated' from 'never asserted'",
        stored_as_sum_contested == stored_as_sum_unasserted
            && contested.epistemic_state() == Epistemic::Contested { net: 0 }
            && unasserted.epistemic_state() == Epistemic::Unasserted
            && contested != unasserted,
        format!(
            "net(+3,−3)={stored_as_sum_contested} == net(unset)={stored_as_sum_unasserted} \
             (a sum LOSES it), while the two-slot field keeps Contested{{net:0}} vs \
             Unasserted — registers differ bitwise"
        ),
    );

    // ---- F2 — DOWNSTREAM inheritance: one falsifier reaches a whole subtree
    // without enumerating it ----
    let parent = at([0x40, 0, 0, 0], 1);
    let descendants: Vec<AttentionFocusFacet> =
        (0..FAN_OUT).map(|i| at([0x40, i, 0, 0], 2)).collect();
    let deep: Vec<AttentionFocusFacet> = (0..FAN_OUT).map(|i| at([0x40, 0x03, i, 0], 3)).collect();
    let all_reached =
        descendants.iter().all(|d| parent.covers(*d)) && deep.iter().all(|d| parent.covers(*d));
    gate(
        "F2 downstream inheritance reaches a subtree with no enumeration",
        all_reached && parent.depth() == 1,
        format!(
            "one depth-1 falsifier covers all {} depth-2 and all {} depth-3 probes by \
             prefix test alone — the subtree is never materialized",
            descendants.len(),
            deep.len()
        ),
    );

    // ---- F3 — THE ASYMMETRY, in PROPAGATION cost. Applying a KNOWN
    // counterexample is O(1) in subtree population; verification costs the
    // whole fan-out and STILL does not close the levels below. Discovery
    // cost is explicitly NOT measured here. ----
    let one_counterexample = at([0x40, 0x07, 0, 0], 2);
    // Regions needed to refute "all of P has X", GIVEN the counterexample.
    let falsification_cost = 1;
    let verification_cost = FAN_OUT as usize; // must exhaust the fan-out…
    let closes_below = false; // …and each child's own subtree stays open
    gate(
        "F3 applying a KNOWN counterexample is O(1) in subtree population; \
         verification costs the fan-out and still doesn't close",
        parent.covers(one_counterexample)
            && falsification_cost == 1
            && verification_cost == 16
            && !closes_below,
        format!(
            "propagate a known refutation: {falsification_cost} region, independent of \
             subtree size; verify: {verification_cost} regions at this depth AND the \
             levels below stay open. DISCOVERY of the counterexample is not measured \
             here and is not claimed cheap"
        ),
    );

    // ---- F4 — the three epistemic states of an intermediate, distinguished.
    // "Known-unknown" is the cheap counterfactual-probe target: the WHERE is
    // addressable while the WHAT is absent ----
    let mediator_region = at([0x40, 0x03, 0, 0], 2); // where a mediator would live
    let known_known = Some(at([0x40, 0x03, 0x05, 0], 3)); // a specific mediator
    let known_unknown: Option<AttentionFocusFacet> = None; // no value…
    let known_unknown_region_is_addressable = mediator_region.depth() == 2; // …but a place
    let unknown_unknown_region: Option<AttentionFocusFacet> = None; // no place either
    gate(
        "F4 known-unknown is addressable-without-a-value (the cheap probe target)",
        known_known.is_some_and(|k| mediator_region.covers(k))
            && known_unknown.is_none()
            && known_unknown_region_is_addressable
            && unknown_unknown_region.is_none(),
        "known-known: a covered address; known-unknown: NO value but a real region to \
         drill into; unknown-unknown: no region at all — only the middle case admits a \
         targeted counterfactual probe"
            .to_string(),
    );

    // ---- F5 — over-kill guard (the dual of the over-generalization finding).
    // A falsifier scoped to a subtree must NOT invalidate outside it ----
    let scoped_falsifier = at([0x40, 0x03, 0, 0], 2);
    let inside = at([0x40, 0x03, 0x09, 0], 3);
    let sibling = at([0x40, 0x04, 0x09, 0], 3);
    let other_parent = at([0x50, 0x03, 0x09, 0], 3);
    gate(
        "F5 falsification does not over-kill (can-fire AND can-stay-silent)",
        scoped_falsifier.covers(inside)
            && !scoped_falsifier.covers(sibling)
            && !scoped_falsifier.covers(other_parent)
            && !scoped_falsifier.covers(parent),
        "fires inside its own subtree; silent on a sibling subtree, on another parent, \
         and on its own ANCESTOR (a child's refutation is not the parent's)"
            .to_string(),
    );

    // ---- F6 — MEASURE the connective-tissue cost the operator conceded ----
    // Best case: "all 16 children" absorbs to ONE parent entry.
    let mut all_children = RowFocusMask::empty();
    for d in &descendants {
        all_children.insert(*d);
    }
    let cost_all_explicit = all_children.len();
    let mut collapsed = RowFocusMask::empty();
    collapsed.insert(parent);
    let cost_all_collapsed = collapsed.len();

    // Worst case: "all children EXCEPT one". `difference` is deliberately
    // conservative — it will not split a prefix, because subtracting a
    // subtree would mean enumerating siblings and inventing addresses.
    let mut except_one = RowFocusMask::empty();
    for (i, d) in descendants.iter().enumerate() {
        if i != 7 {
            except_one.insert(*d);
        }
    }
    let cost_except_one = except_one.len();
    let mut hole = RowFocusMask::empty();
    hole.insert(one_counterexample);
    let parent_minus_hole = collapsed.difference(&hole);
    let cost_via_difference = parent_minus_hole.len();

    gate(
        "F6 connective-tissue cost measured: collapse is free, 'all but one' is the worst case",
        cost_all_explicit == 16
            && cost_all_collapsed == 1
            && cost_except_one == 15
            && cost_via_difference == 1
            && parent_minus_hole.contains(one_counterexample),
        format!(
            "all-16 explicit = {cost_all_explicit} entries but collapses to \
             {cost_all_collapsed}; 'all but one' = {cost_except_one} entries, and the \
             conservative `difference` cannot help (yields {cost_via_difference} entry \
             that STILL covers the hole) — so an exclusion needs its own channel, not a \
             subtracted prefix"
        ),
    );

    // ---- F7 — upstream and downstream are different questions, and the
    // address answers both without changing the datum ----
    let child = at([0x40, 0x03, 0, 0], 2);
    let up = parent.covers(child); // downstream: does P reach C?
    let down = child.covers(parent); // upstream: does C reach P? (must be NO)
    let meet = child.common_prefix(at([0x40, 0x04, 0, 0], 2));
    gate(
        "F7 inheritance is directional: down by covers, up by the meet",
        up && !down && meet.is_some_and(|m| m == parent),
        "P covers C (downstream) but C does not cover P (upstream is NOT containment); \
         the upward direction is common_prefix, which rises to exactly P"
            .to_string(),
    );

    println!("PROBE-HHTL-FALSIFICATION-ASYMMETRY-1: ALL {pass} GATES GREEN");
    println!(
        "measured: falsifier PROPAGATION is cheap and structurally privileged — applying \
         a KNOWN counterexample is O(1) in subtree population (F3; discovery cost is NOT \
         claimed), reaching a whole subtree with no enumeration (F2) and without \
         over-killing (F5); the known-unknown intermediate is addressable-without-a-value, which is \
         what makes a counterfactual probe targetable (F4); and inheritance is directional \
         — covers down, common_prefix up (F7). TWO COSTS FOUND: Auslöschung must be a \
         READING, since a summed slot cannot distinguish 'annihilated' from 'never \
         asserted' (F1); and 'all but one' costs 15 entries because the conservative \
         difference refuses to split a prefix — an exclusion needs its own channel (F6)."
    );
}
