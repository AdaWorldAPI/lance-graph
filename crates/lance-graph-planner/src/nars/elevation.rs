//! `elevation` — the S11 field-elevation RESPONSE (D-DIA-V3-B): the
//! mass-induction sweep that [`super::dissolution::should_elevate`] TRIGGERS
//! (`.claude/plans/dialectic-engine-v1.md` §1 S11, operator pillar 5: "mass
//! induction mints the parent floor; HHTL grows upward").
//!
//! When the rung tissue is dissolving (novelty influx outrunning
//! crystallization), the correct response is NOT per-thought churn — it is a
//! FIELD-scale sweep that lifts groups of subjects sharing a predicate `M`
//! (`S_i is_a M` for many `S_i`, no abstract parent grouping them yet) into a
//! freshly-minted abstract parent `G`: `G is_a M` (the shared property lifted
//! to the abstraction) plus `S_i is_a G` for every child (the children
//! rehomed under the abstraction). A flat `k`-fact flood becomes a 2-level
//! hierarchy.
//!
//! Elevation invents NO abstraction on structureless noise (the honest
//! guard): predicates with fewer than `min_cluster` distinct subjects are
//! left untouched.
//!
//! The payoff (why HHTL grows upward): once the hierarchy exists, a LATER
//! fact about the abstraction (`G is_a N`) propagates to ALL of its children
//! via [`BeliefArena::close_transitive`] — one observed fact reaches `k`
//! children, which a flat structure could never do.
//!
//! This takes `&mut BeliefArena` because it is a GROWTH/write-back step
//! (minting new concepts and beliefs), not a compute path — a *builder*, not
//! a *compute* function, per the workspace's data-flow invariants (no `&mut
//! self` during pure computation; growth/construction steps are the sanctioned
//! exception).

use super::belief::{BeliefArena, CStmt, Copula, Stamp};
use super::truth::TruthValue;
use std::collections::BTreeMap;

/// The outcome of one field-elevation sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Elevation {
    /// The freshly-minted abstract parent term ids (one per lifted cluster).
    pub minted_parents: Vec<u16>,
    /// How many shared-predicate clusters were lifted into an abstraction.
    pub clusters_lifted: usize,
    /// Total children rehomed under a minted parent (Σ cluster sizes).
    pub children_rehomed: usize,
}

/// Mass-induction field-elevation (S11 response). Scans OBSERVED `is_a`
/// beliefs (`rung == 0`), groups subjects by shared predicate `M`, and for
/// every predicate with at least `min_cluster` distinct subjects mints a
/// fresh abstract parent `G`, asserting `G is_a M` + `S_i is_a G` for each
/// child. Predicates below the threshold are left untouched (no false
/// abstraction on noise — the honest guard). Returns what was minted.
///
/// Idempotency: the grouping snapshot is taken BEFORE any minting, so a
/// predicate that is itself a parent minted THIS sweep is never re-lifted
/// (no runaway abstraction within one call).
#[must_use]
pub fn elevate_field(arena: &mut BeliefArena, min_cluster: usize) -> Elevation {
    // 1. Fresh term id floor: one past the max term id in use anywhere.
    let mut max_id: u32 = 0;
    for b in arena.entries() {
        max_id = max_id.max(u32::from(b.stmt.s)).max(u32::from(b.stmt.p));
    }
    let mut next_id: u32 = max_id + 1;
    if next_id > u32::from(u16::MAX) {
        // No room to mint a fresh term id — do nothing rather than panic.
        return Elevation {
            minted_parents: Vec::new(),
            clusters_lifted: 0,
            children_rehomed: 0,
        };
    }

    // 2. Group OBSERVED `is_a` subjects by shared predicate, deterministically
    // (BTreeMap sorts predicate keys; subject vecs are sorted below).
    let mut groups: BTreeMap<u16, Vec<u16>> = BTreeMap::new();
    for b in arena.entries() {
        if b.rung == 0 && b.stmt.cop == Copula::Inh {
            let subjects = groups.entry(b.stmt.p).or_default();
            if !subjects.contains(&b.stmt.s) {
                subjects.push(b.stmt.s);
            }
        }
    }
    for subjects in groups.values_mut() {
        subjects.sort_unstable();
    }

    // Collect the qualifying clusters into an OWNED Vec before touching the
    // arena again — `arena.observe` below needs `&mut`, so no borrow from
    // `arena.entries()` may still be live at that point.
    let clusters: Vec<(u16, Vec<u16>)> = groups
        .into_iter()
        .filter(|(_, subjects)| subjects.len() >= min_cluster)
        .collect();

    // 3. Mint one abstract parent per qualifying cluster.
    let mut minted_parents = Vec::new();
    let mut clusters_lifted = 0usize;
    let mut children_rehomed = 0usize;
    let mut child_counter: u32 = 0;

    for (m, subjects) in clusters {
        if next_id > u32::from(u16::MAX) {
            break;
        }
        let g = next_id as u16;
        next_id += 1;

        arena.observe(
            inh(g, m),
            TruthValue::new(0.9, 0.9),
            Stamp::source(50_000 + clusters_lifted as u32),
        );
        for &s in &subjects {
            arena.observe(
                inh(s, g),
                TruthValue::new(0.9, 0.9),
                Stamp::source(51_000 + child_counter),
            );
            child_counter += 1;
        }

        minted_parents.push(g);
        clusters_lifted += 1;
        children_rehomed += subjects.len();
    }

    Elevation {
        minted_parents,
        clusters_lifted,
        children_rehomed,
    }
}

fn inh(s: u16, p: u16) -> CStmt {
    CStmt {
        s,
        cop: Copula::Inh,
        p,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nars::{BeliefArena, CStmt, Copula, Stamp, TruthValue};

    fn inh(s: u16, p: u16) -> CStmt {
        CStmt {
            s,
            cop: Copula::Inh,
            p,
        }
    }

    /// The honest guard / null: a shared-predicate cluster gets lifted into a
    /// minted abstraction; a structureless flood (every subject its own
    /// distinct predicate — no shared structure at all) mints nothing.
    #[test]
    fn elevate_lifts_structure_not_noise() {
        // STRUCTURED: 5 subjects share predicate 100.
        let mut arena = BeliefArena::new();
        for s in 10u16..15 {
            arena.observe(
                inh(s, 100),
                TruthValue::new(0.9, 0.9),
                Stamp::source(s.into()),
            );
        }
        let e = elevate_field(&mut arena, 3);
        assert_eq!(e.clusters_lifted, 1, "one shared-predicate cluster lifted");
        assert_eq!(e.children_rehomed, 5, "all 5 subjects rehomed");
        assert_eq!(e.minted_parents.len(), 1);

        let g = e.minted_parents[0];
        assert_eq!(g, 101, "next fresh id past max existing id 100");
        assert!(
            arena.get(inh(g, 100)).is_some(),
            "G is_a M lifted (the shared property lifted to the abstraction)"
        );
        assert!(
            arena.get(inh(10, g)).is_some(),
            "child rehomed under the minted abstraction"
        );

        // STRUCTURELESS: every subject a DISTINCT predicate — no shared
        // structure to lift.
        let mut noisy_arena = BeliefArena::new();
        for (s, p) in [(10u16, 100u16), (11, 101), (12, 102), (13, 103), (14, 104)] {
            noisy_arena.observe(
                inh(s, p),
                TruthValue::new(0.9, 0.9),
                Stamp::source(s.into()),
            );
        }
        let e_noise = elevate_field(&mut noisy_arena, 3);
        assert_eq!(
            e_noise.clusters_lifted, 0,
            "elevation invents no abstraction on structureless noise"
        );
        assert!(e_noise.minted_parents.is_empty());
    }

    /// The payoff a hierarchy enables: once the abstraction exists, a single
    /// LATER fact about it propagates to every child via `close_transitive` —
    /// something no flat structure could do with one fact.
    #[test]
    fn minted_parent_propagates_to_children() {
        let mut arena = BeliefArena::new();
        for s in 10u16..15 {
            arena.observe(
                inh(s, 100),
                TruthValue::new(0.9, 0.9),
                Stamp::source(s.into()),
            );
        }
        let e = elevate_field(&mut arena, 3);
        let g = e.minted_parents[0];
        assert_eq!(g, 101);

        // A single NEW fact about the abstraction, never about any child.
        arena.observe(inh(g, 200), TruthValue::new(0.9, 0.9), Stamp::source(9));
        arena.close_transitive(64);

        // Without the minted parent, this one `g is_a 200` fact could not
        // reach any child at all — the hierarchy is what makes one fact
        // propagate to five.
        let children = [10u16, 11, 12, 13, 14];
        let mut derived_count = 0;
        for &s in &children {
            let belief = arena
                .get(inh(s, 200))
                .unwrap_or_else(|| panic!("child {s} must have inherited is_a 200"));
            assert!(
                belief.rung >= 1,
                "child {s}'s is_a 200 must be DERIVED (never observed directly)"
            );
            derived_count += 1;
        }
        assert_eq!(
            derived_count, 5,
            "all 5 children received the propagated fact"
        );
    }

    /// `min_cluster` is a hard threshold: below it, nothing is minted; at it,
    /// the cluster lifts.
    #[test]
    fn min_cluster_threshold_holds() {
        let mut arena = BeliefArena::new();
        arena.observe(inh(10, 100), TruthValue::new(0.9, 0.9), Stamp::source(0));
        arena.observe(inh(11, 100), TruthValue::new(0.9, 0.9), Stamp::source(1));

        let e3 = elevate_field(&mut arena, 3);
        assert_eq!(e3.clusters_lifted, 0, "2 subjects < min_cluster 3");

        let e2 = elevate_field(&mut arena, 2);
        assert_eq!(e2.clusters_lifted, 1, "2 subjects >= min_cluster 2");
    }
}
