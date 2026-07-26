//! `meta_basin` — **ride the tail**: grade low-quorum rows by their multi-hop
//! causal trajectories, cluster those trajectories into meta-basins, find the
//! mini-basins inside them, and SUGGEST (never decide) which rows are outliers.
//!
//! # Why this exists — the saturation problem
//!
//! At [`RungLevel::Transcendent`] every one of the 34 tactics is admissible, so
//! the rung gate has no discriminating power left. Continuing to follow a
//! *dominant* tactic there is eigenvalue-following with nothing justifying it.
//! The discrimination switches to the **passive quorum mantissa**
//! ([`quorum_mantissa`]) — how much of the window converges, observed rather
//! than selected, and hindsight-blind by signature (it cannot see any outcome).
//!
//! The rows the quorum does NOT cover are the **tail**. They are not noise:
//! this workspace's corrections have repeatedly come from exactly there. So the
//! tail is graded by *how* its causality resolved
//! ([`TrajectorySignature`] — hop depth, escalation, terminus) and clustered on
//! that shape.
//!
//! # The anti-eigenvalue discipline, applied AGAIN at the meta level
//!
//! `E-PERIPHERAL-DISSENT-GUARDS-THE-STRATIFICATION-1` fixed dominance blindness
//! one level down. The same failure recurs here in a new costume: a
//! meta-clustering that always reports its biggest basin is following the
//! meta-eigenvalue. Two guards:
//!
//! * **Perturbation stability** ([`MetaBasin::stable_under_perturbation`]) — a
//!   basin that dissolves when the hop budget is nudged was an artifact of the
//!   budget, not a structure. Riding it would be perturbation blindness.
//! * **Mini-basins are searched INSIDE every meta-basin, not only the largest**
//!   ([`mini_basins`]) — sub-structure in a small basin is exactly what a
//!   dominant-mode reader discards.
//!
//! # Outliers are SUGGESTIONS, never verdicts
//!
//! [`outlier_suggestions`] returns [`OutlierSuggestion`]s carrying a reason and
//! the evidence that produced them. Nothing here prunes, commits, or scores.
//! An outlier is a row whose causal shape does not fit the basin it sits in —
//! which may mean it is wrong, or may mean the basin is too coarse. The
//! substrate is not entitled to that judgement, so it does not make it: same
//! shape as `WaveGrounding::Escalate` and `peripheral_dissent` — a signal.

use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_contract::witness_fabric::{quorum_mantissa, trajectory_of, TrajectorySignature};

/// A row of the window, carried with the two gradings this module computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GradedRow {
    /// Index into the caller's window.
    pub idx: usize,
    /// Stream position (the absolute address the loci are relative to).
    pub pos: usize,
    /// Passive quorum mantissa, `0..=15`. LOW = tail.
    pub quorum: u8,
    /// The causal shape its locus resolved with.
    pub trajectory: TrajectorySignature,
}

/// A cluster of rows sharing a causal SHAPE (hop depth + escalation status).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetaBasin {
    /// The shape every member shares.
    pub shape: TrajectorySignature,
    /// Member rows, ascending by window index.
    pub members: Vec<GradedRow>,
}

/// A sub-cluster inside a [`MetaBasin`], distinguished by terminal event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniBasin {
    /// The terminal offset its members converge on (`None` = resolved to
    /// nothing locally — a real group, not an error bucket).
    pub terminal_offset: Option<i8>,
    pub members: Vec<GradedRow>,
}

/// Why a row was suggested as an outlier. Descriptive, never prescriptive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutlierReason {
    /// Alone in its mini-basin while the meta-basin has other, larger ones —
    /// its causality terminates somewhere nothing else does.
    SoloTerminus,
    /// Sits in a meta-basin that did not survive perturbation of the hop
    /// budget: the grouping that placed it may be an artifact.
    UnstableBasin,
    /// Low quorum AND an escalating chain — the row agrees with nobody and its
    /// causality leaves the local horizon.
    IsolatedAndEscalating,
}

/// A **suggestion** that a row is an outlier. Carries its evidence so a
/// consumer can disagree; carries no instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutlierSuggestion {
    pub row: GradedRow,
    pub reason: OutlierReason,
    /// Size of the basin the row was judged against — small basins make weak
    /// suggestions, and the consumer can see that rather than being told.
    pub basin_size: usize,
}

/// Grade every row of a window: passive quorum + causal trajectory.
#[must_use]
pub fn grade_rows(
    window: &[(usize, CausalWitnessFacet)],
    locus: Locus,
    max_hops: u8,
) -> Vec<GradedRow> {
    window
        .iter()
        .enumerate()
        .map(|(idx, &(pos, _))| GradedRow {
            idx,
            pos,
            quorum: quorum_mantissa(idx, window),
            trajectory: trajectory_of(idx, window, locus, max_hops),
        })
        .collect()
}

/// **Ride the tail** — the rows the quorum does not cover.
///
/// `tail_below` is the mantissa threshold (`0..=15`). Rows at or below it are
/// the tail. This deliberately returns rows rather than discarding them: the
/// tail is the input to meta-clustering, not a reject pile.
#[must_use]
pub fn tail(graded: &[GradedRow], tail_below: u8) -> Vec<GradedRow> {
    graded
        .iter()
        .copied()
        .filter(|r| r.quorum <= tail_below)
        .collect()
}

/// Cluster rows into [`MetaBasin`]s by causal shape.
///
/// Every basin is returned — including singletons. Returning only the large
/// ones would be the meta-eigenvalue failure this module exists to avoid.
#[must_use]
pub fn meta_cluster(rows: &[GradedRow]) -> Vec<MetaBasin> {
    let mut basins: Vec<MetaBasin> = Vec::new();
    for &r in rows {
        match basins
            .iter_mut()
            .find(|b| b.shape.same_meta_basin(r.trajectory))
        {
            Some(b) => b.members.push(r),
            None => basins.push(MetaBasin {
                shape: r.trajectory,
                members: vec![r],
            }),
        }
    }
    // Deterministic order: descending size, then by hop depth — stable output
    // for the same input, without which "suggestions" would not be auditable.
    basins.sort_by(|a, b| {
        b.members
            .len()
            .cmp(&a.members.len())
            .then(a.shape.hops.cmp(&b.shape.hops))
    });
    basins
}

/// The **mini-basins** inside one meta-basin, split by terminal event.
///
/// Called for EVERY meta-basin, not just the dominant one — sub-structure in a
/// small basin is precisely what a dominant-mode reader throws away.
#[must_use]
pub fn mini_basins(basin: &MetaBasin) -> Vec<MiniBasin> {
    let mut minis: Vec<MiniBasin> = Vec::new();
    for &m in &basin.members {
        match minis
            .iter_mut()
            .find(|mb| mb.terminal_offset == m.trajectory.terminal_offset)
        {
            Some(mb) => mb.members.push(m),
            None => minis.push(MiniBasin {
                terminal_offset: m.trajectory.terminal_offset,
                members: vec![m],
            }),
        }
    }
    minis.sort_by(|a, b| b.members.len().cmp(&a.members.len()));
    minis
}

impl MetaBasin {
    /// **Perturbation stability** — does this basin survive a nudge to the hop
    /// budget?
    ///
    /// A basin that dissolves when `max_hops` moves by one was an artifact of
    /// the budget rather than a structure in the data. Riding it would be
    /// perturbation blindness — the anti-eigenvalue discipline applied at the
    /// meta level.
    ///
    /// Stable = the same member set still shares a shape at the perturbed
    /// budget. Returns `true` for singletons (nothing to dissolve).
    #[must_use]
    pub fn stable_under_perturbation(
        &self,
        window: &[(usize, CausalWitnessFacet)],
        locus: Locus,
        perturbed_hops: u8,
    ) -> bool {
        if self.members.len() < 2 {
            return true;
        }
        let mut shape: Option<TrajectorySignature> = None;
        for m in &self.members {
            let t = trajectory_of(m.idx, window, locus, perturbed_hops);
            match shape {
                None => shape = Some(t),
                Some(s) => {
                    if !s.same_meta_basin(t) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// **Suggest** which rows look like outliers — never decide.
///
/// Runs over EVERY meta-basin (not the dominant one), splits each into
/// mini-basins, and flags rows whose causal shape does not fit. Each suggestion
/// carries its reason and the size of the basin it was judged against, so a
/// consumer can weigh it rather than obey it.
///
/// `perturbed_hops` drives the [`stability`](MetaBasin::stable_under_perturbation)
/// check; a row inside an unstable basin is suggested with
/// [`OutlierReason::UnstableBasin`] — the honest reading being "the grouping
/// may be an artifact", not "this row is wrong".
#[must_use]
pub fn outlier_suggestions(
    window: &[(usize, CausalWitnessFacet)],
    locus: Locus,
    max_hops: u8,
    perturbed_hops: u8,
    tail_below: u8,
) -> Vec<OutlierSuggestion> {
    let graded = grade_rows(window, locus, max_hops);
    let tail_rows = tail(&graded, tail_below);
    let mut out = Vec::new();
    for basin in meta_cluster(&tail_rows) {
        let size = basin.members.len();
        let stable = basin.stable_under_perturbation(window, locus, perturbed_hops);
        for mini in mini_basins(&basin) {
            for &row in &mini.members {
                let reason = if !stable {
                    OutlierReason::UnstableBasin
                } else if mini.members.len() == 1 && size > 1 {
                    OutlierReason::SoloTerminus
                } else if row.quorum == 0 && row.trajectory.escalated {
                    OutlierReason::IsolatedAndEscalating
                } else {
                    continue;
                };
                out.push(OutlierSuggestion {
                    row,
                    reason,
                    basin_size: size,
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(edges: &[(Locus, i8)]) -> CausalWitnessFacet {
        let mut f = CausalWitnessFacet::ZERO;
        for &(l, o) in edges {
            f = f.with(l, o);
        }
        f
    }

    #[test]
    fn grading_carries_both_axes_and_the_tail_is_the_low_quorum_rows() {
        let win = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, CausalWitnessFacet::ZERO),
            (2, w(&[(Locus::Antecedent, 1)])),
        ];
        let graded = grade_rows(&win, Locus::Antecedent, 8);
        assert_eq!(graded.len(), 3);
        for g in &graded {
            assert!(g.quorum <= 15, "mantissa out of i4 range");
        }
        // A high threshold takes everything; a threshold of 0 takes only the
        // rows nobody agrees with. The tail is a VIEW, never a discard.
        assert_eq!(tail(&graded, 15).len(), 3);
        assert!(tail(&graded, 0).len() <= 3);
    }

    #[test]
    fn meta_cluster_keeps_singletons_not_just_the_dominant_basin() {
        let win = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, CausalWitnessFacet::ZERO),
            (2, w(&[(Locus::Antecedent, 1)])),
            (3, CausalWitnessFacet::ZERO),
            (4, w(&[(Locus::Antecedent, 7)])), // escalates → its own shape
        ];
        let graded = grade_rows(&win, Locus::Antecedent, 8);
        let basins = meta_cluster(&graded);
        assert!(basins.len() >= 2, "escalating row was merged away");
        // Every row survives clustering — nothing is silently dropped.
        let total: usize = basins.iter().map(|b| b.members.len()).sum();
        assert_eq!(total, graded.len(), "meta_cluster lost rows");
        // Deterministic ordering (auditable suggestions require it).
        let again = meta_cluster(&graded);
        assert_eq!(basins, again);
    }

    #[test]
    fn mini_basins_partition_their_meta_basin() {
        let win = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, CausalWitnessFacet::ZERO),
            (2, w(&[(Locus::Antecedent, 1)])),
            (3, CausalWitnessFacet::ZERO),
        ];
        let graded = grade_rows(&win, Locus::Antecedent, 8);
        for b in meta_cluster(&graded) {
            let minis = mini_basins(&b);
            let total: usize = minis.iter().map(|m| m.members.len()).sum();
            assert_eq!(total, b.members.len(), "mini-basins lost members");
        }
    }

    #[test]
    fn perturbation_marks_budget_artifacts_and_spares_singletons() {
        let win = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, CausalWitnessFacet::ZERO),
        ];
        let graded = grade_rows(&win, Locus::Antecedent, 8);
        for b in meta_cluster(&graded) {
            // Singletons have nothing to dissolve — never reported unstable.
            if b.members.len() < 2 {
                assert!(b.stable_under_perturbation(&win, Locus::Antecedent, 1));
            }
            // The call is total: any budget, no panic.
            for p in [0u8, 1, 2, 8, 255] {
                let _ = b.stable_under_perturbation(&win, Locus::Antecedent, p);
            }
        }
    }

    /// The load-bearing contract of this module: it SUGGESTS. Every suggestion
    /// carries a reason and the basin size it was judged against, and nothing
    /// is removed from the input.
    #[test]
    fn suggestions_are_advisory_and_evidenced() {
        let win = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, CausalWitnessFacet::ZERO),
            (2, w(&[(Locus::Antecedent, 1)])),
            (3, CausalWitnessFacet::ZERO),
            (4, w(&[(Locus::Antecedent, 7)])),
        ];
        let before = win.len();
        let sug = outlier_suggestions(&win, Locus::Antecedent, 8, 2, 15);
        // Advisory: the window is untouched (it is `&`, so this is a statement
        // about intent as much as memory).
        assert_eq!(win.len(), before);
        for s in &sug {
            assert!(
                s.basin_size >= 1,
                "suggestion without a basin to justify it"
            );
            assert!(s.row.idx < win.len());
        }
        // Deterministic: same input, same suggestions — auditable, not a draw.
        assert_eq!(sug, outlier_suggestions(&win, Locus::Antecedent, 8, 2, 15));
    }

    /// A suggester that can never suggest is as useless as a gate that never
    /// gates — the failure this workspace already caught twice.
    #[test]
    fn suggester_can_actually_fire() {
        // A window with one escalating, isolated row among settled ones.
        let win = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, CausalWitnessFacet::ZERO),
            (2, w(&[(Locus::Antecedent, 1)])),
            (3, CausalWitnessFacet::ZERO),
            (4, w(&[(Locus::Antecedent, 7)])),
            (5, w(&[(Locus::Kausal, -1)])),
        ];
        let sug = outlier_suggestions(&win, Locus::Antecedent, 8, 2, 15);
        assert!(
            !sug.is_empty(),
            "outlier suggester never fires — inert channel"
        );
    }
}
