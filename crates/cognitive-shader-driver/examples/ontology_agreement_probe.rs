//! D-GSO-3 (P3): global ontology agreement / disagreement probe.
//!
//! Plan: `.claude/plans/2026-10-06-global-sudoku-replayable-orchestration-v1.md`
//! §9 and §18 P3. Claim under test: support, opposition and unknown can be
//! measured upward and downward through a hierarchy, folded immediately, with
//! no population of path objects, and local agreement can be told apart from
//! accumulated global agreement.
//!
//! What it reuses, and what it adds:
//!
//! - **Hierarchy:** [`NiblePath`] addresses. Ancestor, descendant and sibling
//!   are key arithmetic (`is_ancestor_of`, `is_descendant_of`, `is_sibling_of`).
//! - **Contribution and fold:** [`Quorum`] / [`SourceVerdict`] from
//!   `ontology_warrant`. Each observation is one verdict, folded with
//!   `Quorum::observe` the moment it is read. Silence stays abstention.
//! - **The only new piece:** where an observation counts. For a claim
//!   `(node, property)`:
//!   - **local** = the node itself and its siblings (same parent);
//!   - **global up** = strict ancestors of the node;
//!   - **global down** = strict descendants of the node.
//!
//!   Anything else was not consulted for this claim.
//!
//! Tension is the binary Shannon entropy of the speaking split of a quorum
//! (`corroborating : conflicting`). With nobody speaking it is unknown, not
//! zero. Per plan §10 entropy is search pressure, never a truth score; this
//! probe only classifies where the two scales disagree.
//!
//! One pass over the observation list per claim; the three quorums are `Copy`
//! counters. No ancestor list, no descendant list and no path object is built.
//!
//! Run: `cargo run -p cognitive-shader-driver --example ontology_agreement_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example ontology_agreement_probe`

use lance_graph_contract::hhtl::NiblePath;
use lance_graph_contract::ontology_warrant::{Quorum, SourceVerdict};

/// The properties the fixture observes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Property {
    HasFur,
    LaysEggs,
    Flies,
    GivesMilk,
}

/// One observation: a source looked at a class and said something about one
/// property. This is evidence, so it stays a population (plan §3).
#[derive(Debug, Clone, Copy)]
struct Observation {
    class: NiblePath,
    property: Property,
    verdict: SourceVerdict,
}

/// The three quorums for one claim, built in one pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Agreement {
    local: Quorum,
    up: Quorum,
    down: Quorum,
}

impl Agreement {
    /// Ancestors and descendants together: the accumulated basin.
    fn global(self) -> Quorum {
        Quorum::new(
            self.up.corroborating + self.down.corroborating,
            self.up.silent + self.down.silent,
            self.up.conflicting + self.down.conflicting,
        )
    }
}

/// Fold every observation that bears on `(node, property)` into the scale it
/// belongs to. One pass; nothing is collected.
fn agreement(observations: &[Observation], node: NiblePath, property: Property) -> Agreement {
    let mut a = Agreement::default();
    for o in observations.iter().filter(|o| o.property == property) {
        if o.class == node || o.class.is_sibling_of(node) {
            a.local = a.local.observe(o.verdict);
        } else if o.class.is_ancestor_of(node) {
            a.up = a.up.observe(o.verdict);
        } else if o.class.is_descendant_of(node) {
            a.down = a.down.observe(o.verdict);
        }
    }
    a
}

/// Binary Shannon entropy (bits) of the speaking split, or `None` when no
/// source spoke. Silence does not enter.
fn tension(q: Quorum) -> Option<f32> {
    if !q.has_evidence() {
        return None;
    }
    let p = f32::from(q.corroborating) / f32::from(q.speaking());
    let h = |x: f32| if x <= 0.0 { 0.0 } else { -x * x.log2() };
    Some(h(p) + h(1.0 - p))
}

/// Where local and global tension fall relative to `high` (plan §9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Diagnosis {
    /// Both scales calm: nothing to inspect.
    Settled,
    /// Global calm, local split: an exception, a missing relation or a boundary.
    LocalException,
    /// Local calm, global split: locally plausible, inconsistent in the basin.
    BasinConflict,
    /// Both split: a hotspot worth another bounded experiment.
    Hotspot,
    /// At least one scale has no speaking source.
    Unknown,
}

/// `HIGH` is a policy pin for this fixture, not a measured constant.
const HIGH: f32 = 0.5;

fn diagnose(a: Agreement, high: f32) -> Diagnosis {
    match (tension(a.local), tension(a.global())) {
        (Some(l), Some(g)) => match (l >= high, g >= high) {
            (false, false) => Diagnosis::Settled,
            (true, false) => Diagnosis::LocalException,
            (false, true) => Diagnosis::BasinConflict,
            (true, true) => Diagnosis::Hotspot,
        },
        _ => Diagnosis::Unknown,
    }
}

/// The synthetic mammal family (plan §18 P3).
struct Family {
    mammal: NiblePath,
    monotreme: NiblePath,
    platypus: NiblePath,
    echidna: NiblePath,
    dog: NiblePath,
    cat: NiblePath,
    whale: NiblePath,
    bat: NiblePath,
    fruit_bat: NiblePath,
    vampire_bat: NiblePath,
    bird: NiblePath,
    penguin: NiblePath,
}

fn family() -> Family {
    let mammal = NiblePath::root(0);
    let monotreme = mammal.child(0);
    let placental = mammal.child(1);
    let bat = placental.child(3);
    let bird = NiblePath::root(1);
    Family {
        mammal,
        monotreme,
        platypus: monotreme.child(0),
        echidna: monotreme.child(1),
        dog: placental.child(0),
        cat: placental.child(1),
        whale: placental.child(2),
        bat,
        fruit_bat: bat.child(0),
        vampire_bat: bat.child(1),
        bird,
        penguin: bird.child(0),
    }
}

fn observations(f: &Family) -> Vec<Observation> {
    use Property::*;
    use SourceVerdict::*;
    let o = |class, property, verdict| Observation {
        class,
        property,
        verdict,
    };
    vec![
        // Fur: the basin says yes, the whale says no.
        o(f.mammal, HasFur, Corroborates),
        o(f.dog, HasFur, Corroborates),
        o(f.cat, HasFur, Corroborates),
        o(f.bat, HasFur, Corroborates),
        o(f.whale, HasFur, Conflicts),
        o(f.whale, HasFur, Silent),
        // Eggs: monotremes say yes, the mammal basin says no.
        o(f.mammal, LaysEggs, Conflicts),
        o(f.monotreme, LaysEggs, Corroborates),
        o(f.platypus, LaysEggs, Corroborates),
        o(f.echidna, LaysEggs, Corroborates),
        // Flight: split locally among placentals and globally around bats.
        o(f.mammal, Flies, Conflicts),
        o(f.bat, Flies, Corroborates),
        o(f.dog, Flies, Conflicts),
        o(f.cat, Flies, Conflicts),
        o(f.whale, Flies, Conflicts),
        o(f.fruit_bat, Flies, Corroborates),
        o(f.vampire_bat, Flies, Corroborates),
        // Milk: everyone in the placental neighbourhood and the basin agrees.
        o(f.mammal, GivesMilk, Corroborates),
        o(f.dog, GivesMilk, Corroborates),
        o(f.cat, GivesMilk, Corroborates),
        o(f.whale, GivesMilk, Corroborates),
        o(f.bat, GivesMilk, Corroborates),
        // A bird source that is consulted and silent on milk.
        o(f.bird, GivesMilk, Silent),
    ]
}

fn main() {
    let f = family();
    let obs = observations(&f);
    let claims = [
        ("whale  has_fur   ", f.whale, Property::HasFur),
        ("platypus lays_eggs", f.platypus, Property::LaysEggs),
        ("bat    flies     ", f.bat, Property::Flies),
        ("dog    gives_milk ", f.dog, Property::GivesMilk),
        ("penguin gives_milk", f.penguin, Property::GivesMilk),
    ];
    println!(
        "D-GSO-3 ontology agreement probe ({} observations)",
        obs.len()
    );
    for (name, node, property) in claims {
        let a = agreement(&obs, node, property);
        println!(
            "  {name}  local {:?} up {:?} down {:?}  H_local {:?} H_global {:?}  -> {:?}",
            (a.local.corroborating, a.local.conflicting),
            (a.up.corroborating, a.up.conflicting),
            (a.down.corroborating, a.down.conflicting),
            tension(a.local),
            tension(a.global()),
            diagnose(a, HIGH),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(node: NiblePath, property: Property) -> Agreement {
        agreement(&observations(&family()), node, property)
    }

    /// FAILS IF: any of the four populated cells is not reached, or a claim
    /// lands in the wrong one. Each claim is built to hit exactly one cell.
    #[test]
    fn each_claim_lands_in_its_cell() {
        let f = family();
        let cases = [
            (f.whale, Property::HasFur, Diagnosis::LocalException),
            (f.platypus, Property::LaysEggs, Diagnosis::BasinConflict),
            (f.bat, Property::Flies, Diagnosis::Hotspot),
            (f.dog, Property::GivesMilk, Diagnosis::Settled),
            (f.penguin, Property::GivesMilk, Diagnosis::Unknown),
        ];
        for (node, property, want) in cases {
            assert_eq!(diagnose(check(node, property), HIGH), want, "{property:?}");
        }
    }

    /// FAILS IF: an observation is counted on the wrong scale. The exact
    /// counts per scale are pinned for each claim.
    #[test]
    fn scales_count_exactly() {
        let f = family();
        // Whale fur: local = whale + 3 siblings; up = placental (none) + mammal.
        let a = check(f.whale, Property::HasFur);
        assert_eq!(a.local, Quorum::new(3, 1, 1));
        assert_eq!(a.up, Quorum::new(1, 0, 0));
        assert_eq!(a.down, Quorum::default());
        // Platypus eggs: local = platypus + echidna; up = monotreme + mammal.
        let a = check(f.platypus, Property::LaysEggs);
        assert_eq!(a.local, Quorum::new(2, 0, 0));
        assert_eq!(a.up, Quorum::new(1, 0, 1));
        // Bat flight: down = the two bat species.
        let a = check(f.bat, Property::Flies);
        assert_eq!(a.local, Quorum::new(1, 0, 3));
        assert_eq!(a.up, Quorum::new(0, 0, 1));
        assert_eq!(a.down, Quorum::new(2, 0, 0));
        // Penguin milk: the bird basin is consulted but silent.
        let a = check(f.penguin, Property::GivesMilk);
        assert_eq!(a.up, Quorum::new(0, 1, 0));
        assert!(!a.global().has_evidence());
    }

    /// FAILS IF: local and global are not genuinely different scales.
    ///
    /// Pooled into one quorum, the whale (local exception) and the platypus
    /// (basin conflict) are the same kind of claim: both split, both above the
    /// threshold. Kept apart, they get different diagnoses, because the split
    /// sits on a different scale.
    #[test]
    fn two_scales_separate_what_one_pooled_scale_merges() {
        let f = family();
        let pooled = |a: Agreement| {
            let g = a.global();
            Quorum::new(
                a.local.corroborating + g.corroborating,
                0,
                a.local.conflicting + g.conflicting,
            )
        };
        let whale = check(f.whale, Property::HasFur);
        let platypus = check(f.platypus, Property::LaysEggs);
        assert!(tension(pooled(whale)).unwrap() >= HIGH);
        assert!(tension(pooled(platypus)).unwrap() >= HIGH);
        assert_eq!(diagnose(whale, HIGH), Diagnosis::LocalException);
        assert_eq!(diagnose(platypus, HIGH), Diagnosis::BasinConflict);
        assert_eq!(tension(platypus.local), Some(0.0));
        assert_eq!(tension(platypus.global()), Some(1.0));
    }

    /// FAILS IF: silence moves tension, or a claim nobody spoke on reads as
    /// calm (0.0) instead of unknown.
    #[test]
    fn silence_is_not_evidence_and_nobody_is_unknown() {
        let a = check(family().whale, Property::HasFur);
        let without_silence = Quorum::new(a.local.corroborating, 0, a.local.conflicting);
        assert_eq!(tension(a.local), tension(without_silence));
        assert_eq!(tension(Quorum::new(0, 9, 0)), None);
    }

    /// FAILS IF: the threshold does nothing. Raising it above any possible
    /// entropy silences every split cell; lowering it to zero makes every
    /// speaking claim a hotspot.
    #[test]
    fn the_threshold_is_not_inert() {
        let f = family();
        let split = [
            (f.whale, Property::HasFur),
            (f.platypus, Property::LaysEggs),
            (f.bat, Property::Flies),
        ];
        for (node, property) in split {
            let a = check(node, property);
            assert_eq!(diagnose(a, 1.01), Diagnosis::Settled);
            assert_eq!(diagnose(a, 0.0), Diagnosis::Hotspot);
        }
    }

    /// FAILS IF: classes outside the node's line count. A bird observation
    /// never reaches a mammal claim, and a cousin (other subtree under the same
    /// grandparent) is not local.
    #[test]
    fn unrelated_classes_are_not_consulted() {
        let f = family();
        let a = check(f.dog, Property::LaysEggs);
        // Platypus, echidna and monotreme are cousins and an uncle: not local,
        // not ancestors, not descendants. Only the mammal basin speaks.
        assert_eq!(a.local, Quorum::default());
        assert_eq!(a.up, Quorum::new(0, 0, 1));
        assert_eq!(a.down, Quorum::default());
    }
}
