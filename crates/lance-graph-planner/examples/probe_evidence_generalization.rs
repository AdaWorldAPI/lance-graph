//! PROBE-EVIDENCE-RISES-BY-GENERALIZATION-1 — does "globality" fall out of
//! HHTL geometry, or does geometry alone over-generalize?
//!
//! **The law under test (operator, 2026-08-23):**
//!
//! > Evidence rises only as high in the HHTL tree as its independent support
//! > generalizes.
//!
//! The appeal is that it removes a whole metadata system: no `enum Scope
//! { Local, Regional, Global }`, no global-concern score, no scheduler
//! deciding to "promote" a belief. A conclusion supported in one basin lives
//! at that basin's address; a conclusion independently supported across
//! sibling basins can live at their common ancestor. **Globality becomes
//! geometry.**
//!
//! This probe measures whether that actually holds on shipped types — and
//! finds a real boundary: **it does not hold on geometry alone.** G3 is the
//! load-bearing negative result.
//!
//! # The two-part decomposition being tested
//!
//! ```text
//!   NARS frequency   proportion / direction of evidence   (an estimate)
//!   NARS confidence  evidential weight behind it          (an evidence mass)
//!   HHTL address     the scope at which it holds
//!   HHTL ancestry    how far that support generalizes
//! ```
//!
//! Grounded, not assumed: `TruthValue::revise` (`nars/truth.rs:57`) pools by
//! `evidence_weight() = c/(1−c)` — so confidence IS the evidence-mass side,
//! and revision is weight-weighted pooling. `spo::truth`'s sibling revision
//! is documented *"combine two truth values with **independent** evidence"* —
//! the independence precondition is already written down in shipped code.
//!
//! # What this probe uses, all shipped
//!
//! - [`AttentionFocusFacet::common_prefix`] — the MEET (deepest focus
//!   covering both). This is the "rise to the common ancestor" operator; it
//!   never invents an address.
//! - [`AttentionFocusFacet::covers`] — prefix containment (ancestor test).
//! - [`RowFocusMask`] — the antichain, with absorbing `union` and
//!   deliberately-conservative `difference`.
//! - `TruthValue::revise` — NARS evidence pooling.
//! - `Stamp::{disjoint, union}` — the shipped independence machinery.
//!
//! # Honesty box
//!
//! - Six regions of a toy hierarchy. This tests the OPERATOR ALGEBRA
//!   (does the meet + revision behave as the law claims), not a corpus.
//! - `BeliefArena` is not used at all here — this probe is about the
//!   addressing law, not about the arena. No claim is made that beliefs
//!   currently HAVE such addresses; the audit (#1006) says they do not.
//! - "Parent learns" is modelled as a DERIVED READING computed at the
//!   parent's address, never a mutation, copy, or move of the children.
//!   G5 asserts the children are byte-identical afterward.

use lance_graph_contract::attention_facet::{AttentionFocusFacet, RowFocusMask};
use lance_graph_contract::facet::{FacetCascade, FacetTier};
use lance_graph_planner::nars::belief::Stamp;
use lance_graph_planner::nars::truth::TruthValue;

/// Probe-local classid for the toy hierarchy (NOT an OGAR mint).
const REGION_CLASSID: u32 = 0xFFFF_000C;

/// A facet whose first ladder byte is `parent` and second is `child`.
/// (`tier_bytes()` is `[t0.hi, t0.lo, t1.hi, …]`, and `covers` compares the
/// first `depth` ladder bytes — so depth 1 = the parent prefix, depth 2 =
/// the child.)
fn region(parent: u8, child: u8) -> FacetCascade {
    FacetCascade {
        facet_classid: REGION_CLASSID,
        tiers: [
            FacetTier {
                hi: parent,
                lo: child,
            },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
            FacetTier { hi: 0, lo: 0 },
        ],
    }
}

/// The exact address of one child basin (depth 2 = parent byte + child byte).
fn child_at(parent: u8, child: u8) -> AttentionFocusFacet {
    AttentionFocusFacet::prefix(region(parent, child), 2).expect("depth 2 ≤ 12")
}

/// The parent prefix (depth 1 = the parent byte alone).
fn parent_at(parent: u8) -> AttentionFocusFacet {
    AttentionFocusFacet::prefix(region(parent, 0), 1).expect("depth 1 ≤ 12")
}

/// One basin's local support: an address, a truth, and its evidential base.
#[derive(Clone, Copy)]
struct Support {
    at: AttentionFocusFacet,
    truth: TruthValue,
    stamp: Stamp,
}

/// **The rise operator, provenance-aware.** Fold a set of local supports into
/// the highest address their INDEPENDENT support justifies:
///
/// - the address rises by `common_prefix` (the shipped meet) — never by
///   inventing an ancestor;
/// - truth pools by `TruthValue::revise` ONLY across supports whose stamps
///   are disjoint; an overlapping stamp contributes its address but NOT a
///   second count of its evidence (the shipped `Stamp::disjoint` rule).
///
/// Returns `None` if the supports do not share a class (no common ancestor
/// exists — a focus never crosses a class).
fn rise(supports: &[Support], provenance_aware: bool) -> Option<(AttentionFocusFacet, TruthValue)> {
    let first = supports.first()?;
    let mut at = first.at;
    let mut truth = first.truth;
    let mut pooled = first.stamp;

    for s in &supports[1..] {
        at = at.common_prefix(s.at)?;
        if !provenance_aware || pooled.disjoint(s.stamp) {
            truth = truth.revise(&s.truth);
            pooled = pooled.union(s.stamp);
        }
        // provenance-aware + overlapping ⇒ the address still rises (the
        // claim IS supported there), but the evidence is not counted twice.
    }
    Some((at, truth))
}

fn main() {
    let mut pass = 0u32;
    let mut gate = |name: &str, ok: bool, detail: String| {
        assert!(ok, "[FAIL] {name} — {detail}");
        println!("  [PASS] {name} — {detail}");
        pass += 1;
    };

    const P: u8 = 0x40;
    let (a, b, c, d) = (
        child_at(P, 0x01),
        child_at(P, 0x02),
        child_at(P, 0x03),
        child_at(P, 0x04),
    );
    let parent = parent_at(P);
    let strong = TruthValue::new(0.90, 0.85);

    // ---- G1 — a single basin's support STAYS LOCAL ----
    // Nothing promotes it; there is no second basin for a meet to rise to.
    let only_a = [Support {
        at: a,
        truth: strong,
        stamp: Stamp::source(1),
    }];
    let (at1, t1) = rise(&only_a, true).expect("same class");
    gate(
        "G1 single-basin support stays at its own address",
        at1 == a && at1 != parent && !at1.covers(parent) && t1 == strong,
        format!(
            "address stays depth {} (the child), truth untouched f={:.2} c={:.2}",
            at1.depth(),
            t1.frequency,
            t1.confidence
        ),
    );

    // ---- G2 — independent sibling support RISES to the common ancestor,
    // and pooling genuinely increases evidential weight ----
    let indep = [
        Support {
            at: a,
            truth: strong,
            stamp: Stamp::source(1),
        },
        Support {
            at: b,
            truth: strong,
            stamp: Stamp::source(2),
        },
        Support {
            at: c,
            truth: strong,
            stamp: Stamp::source(3),
        },
    ];
    let (at2, t2) = rise(&indep, true).expect("same class");
    gate(
        "G2 independent sibling support rises to the common ancestor",
        at2 == parent
            && at2.depth() == 1
            && at2.covers(a)
            && at2.covers(b)
            && at2.covers(c)
            && t2.confidence > strong.confidence,
        format!(
            "rose to depth 1 (covers A/B/C); pooled c={:.4} > each child's c={:.2}",
            t2.confidence, strong.confidence
        ),
    );

    // ---- G3 — THE FALSIFIER. Geometry ALONE over-generalizes. ----
    // The SAME original observation seen through three sibling basins: the
    // addresses are still siblings, so the meet still rises — but the
    // evidence must NOT pool. A geometry-only fold inflates confidence;
    // the provenance-aware fold does not.
    let same_source = Stamp::source(7);
    let correlated = [
        Support {
            at: a,
            truth: strong,
            stamp: same_source,
        },
        Support {
            at: b,
            truth: strong,
            stamp: same_source,
        },
        Support {
            at: c,
            truth: strong,
            stamp: same_source,
        },
    ];
    let (at_naive, t_naive) = rise(&correlated, false).expect("same class");
    let (at_prov, t_prov) = rise(&correlated, true).expect("same class");
    gate(
        "G3 geometry alone over-generalizes; provenance is REQUIRED",
        at_naive == parent
            && at_prov == parent
            && t_naive.confidence > t_prov.confidence
            && (t_prov.confidence - strong.confidence).abs() < 1e-6
            && (t_naive.confidence - t2.confidence).abs() < 1e-6,
        format!(
            "same source via 3 basins: naive pools to c={:.4} (indistinguishable from 3 \
             INDEPENDENT sources, c={:.4}) while provenance-aware holds c={:.2}",
            t_naive.confidence, t2.confidence, t_prov.confidence
        ),
    );

    // ---- G4 — a falsifying sibling does not collapse the parent ----
    // D disagrees. The parent keeps its generalized support; the
    // disagreement is retained AS AN ADDRESS to descend into, not folded
    // into a single scalar that forgets where the exception lives.
    let weak_d = TruthValue::new(0.05, 0.85);
    let with_d = [
        Support {
            at: a,
            truth: strong,
            stamp: Stamp::source(1),
        },
        Support {
            at: b,
            truth: strong,
            stamp: Stamp::source(2),
        },
        Support {
            at: c,
            truth: strong,
            stamp: Stamp::source(3),
        },
        Support {
            at: d,
            truth: weak_d,
            stamp: Stamp::source(4),
        },
    ];
    let (at4, t4) = rise(&with_d, true).expect("same class");
    // The exception's address survives as a scoped difference: the supported
    // antichain minus the region that agrees.
    let mut supported = RowFocusMask::empty();
    for s in &with_d {
        if s.truth.expectation() > 0.5 {
            supported.insert(s.at);
        }
    }
    let mut dissenting = RowFocusMask::empty();
    for s in &with_d {
        if s.truth.expectation() <= 0.5 {
            dissenting.insert(s.at);
        }
    }
    let exception_regions = dissenting.difference(&supported);
    gate(
        "G4 a dissenting sibling is retained as an ADDRESS, not averaged away",
        at4 == parent
            && t4.frequency < strong.frequency
            && t4.frequency > weak_d.frequency
            && exception_regions.len() == 1
            && exception_regions.entries()[0] == d
            && !supported.contains(d),
        format!(
            "parent still generalizes (f={:.3}, between {:.2} and {:.2}); the exception \
             remains addressable at exactly 1 region (D)",
            t4.frequency, weak_d.frequency, strong.frequency
        ),
    );

    // ---- G5 — CHILDREN STAY, PARENT LEARNS ----
    // The rise is a derived reading at an address that already exists. It
    // copies nothing upward and mutates nothing.
    let before: Vec<[u8; 12]> = with_d.iter().map(|s| s.at.payload_bytes()).collect();
    let _ = rise(&with_d, true);
    let after: Vec<[u8; 12]> = with_d.iter().map(|s| s.at.payload_bytes()).collect();
    gate(
        "G5 children stay byte-identical; the parent acquires a reading",
        before == after && parent.covers(a) && parent.covers(d) && parent != a,
        "the fold produced a value AT an existing address; no child moved or was copied"
            .to_string(),
    );

    // ---- G6 — the rise is BOUNDED by where support actually is ----
    // A support in a DIFFERENT parent region drags the meet up to a shallower
    // prefix — it cannot smuggle the conclusion sideways into P.
    const Q: u8 = 0x50;
    let cross = [
        Support {
            at: a,
            truth: strong,
            stamp: Stamp::source(1),
        },
        Support {
            at: child_at(Q, 0x01),
            truth: strong,
            stamp: Stamp::source(2),
        },
    ];
    let (at6, _) = rise(&cross, true).expect("same class");
    gate(
        "G6 cross-region support rises HIGHER (coarser), never sideways",
        at6.depth() == 0 && at6.covers(a) && at6.covers(child_at(Q, 0x01)) && at6 != parent,
        format!(
            "meet of two different parent regions is depth {} (the whole class), not P",
            at6.depth()
        ),
    );

    // ---- G7 — anti-vacuity: the rise operator can REFUSE ----
    // A focus never crosses a class, so supports in different classes have no
    // common ancestor and the fold returns None rather than inventing one.
    let other_class = AttentionFocusFacet::prefix(
        FacetCascade {
            facet_classid: REGION_CLASSID ^ 0xFF,
            tiers: [FacetTier { hi: P, lo: 0x01 }; 6],
        },
        2,
    )
    .expect("depth 2");
    let cross_class = [
        Support {
            at: a,
            truth: strong,
            stamp: Stamp::source(1),
        },
        Support {
            at: other_class,
            truth: strong,
            stamp: Stamp::source(2),
        },
    ];
    gate(
        "G7 the rise refuses across classes (can say NO)",
        rise(&cross_class, true).is_none() && rise(&only_a, true).is_some(),
        "no common ancestor across classes ⇒ None; the same operator still folds within one"
            .to_string(),
    );

    println!("PROBE-EVIDENCE-RISES-BY-GENERALIZATION-1: ALL {pass} GATES GREEN");
    println!(
        "measured: the address side of the law holds on shipped operators — support rises by \
         `common_prefix` exactly as far as it generalizes (G1/G2/G6), refuses across classes \
         (G7), keeps a dissenting region addressable instead of averaging it away (G4), and \
         moves no child (G5). BUT G3 shows geometry ALONE over-generalizes: three views of ONE \
         source are geometrically indistinguishable from three independent ones. \
         `globality = geometry` is TRUE ONLY WITH PROVENANCE."
    );
}
