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
//!
//! # The metric upgrade — from exact match to density
//!
//! `E-SATURATION-SWITCHES-TO-PASSIVE-QUORUM-1` named its own floor: clustering
//! by exact causal-shape equality is a proxy, not a metric — two trajectories
//! one hop apart were as "far" as two five hops apart, and [`mini_basins`] split
//! on terminal EQUALITY. [`trajectory_distance`] supplies the missing metric and
//! [`density_scores`] a CHAODA-flavoured relative-density anomaly over it: a
//! row's score is its own local sparsity divided by its neighbours', so an
//! anomaly is *relative to its neighbourhood* rather than to a global cut.
//!
//! The metric is a strict generalization, not a replacement: at distance zero it
//! agrees with `same_meta_basin` + terminal equality, so the coarse path
//! ([`outlier_suggestions`]) keeps its exact prior behaviour and the metric path
//! ([`ranked_outlier_suggestions`]) only RANKS what the coarse path could merely
//! list.

use lance_graph_contract::causal_witness::Locus;
use lance_graph_contract::witness_fabric::{
    quorum_mantissa_lens, trajectory_of_lens, TrajectorySignature, WitnessLens,
};

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

// ── The metric over trajectory space ──────────────────────────────────────
//
// Integer weights, integer distance: floating point would make the ranking's
// tie structure depend on rounding, and a suggestion that reorders between runs
// is not auditable. Every weight below is a modelling choice, so each one states
// what it claims.

/// Cost of one hop of depth difference.
///
/// Hop count is genuinely ORDINAL — three hops is further from one hop than two
/// is — so it is the one axis where a difference is a magnitude.
pub const HOP_WEIGHT: u32 = 4;

/// Cost of disagreeing about escalation.
///
/// `escalated` is CATEGORICAL, not a continuous axis: it says the chain left the
/// horizon, which is a different *mode* of resolution rather than more of the
/// same one. So it contributes a flat cost on mismatch and nothing on agreement
/// — never a scaled difference. Its size (3 hops) says a mode change is a real
/// separation but not an incommensurable one; that magnitude is hand-tuned and
/// is stated as such (`I-NOISE-FLOOR-JIRAK` — a threshold without a derivation
/// must admit it).
pub const ESCALATION_WEIGHT: u32 = 12;

/// Cost of one terminus being `None` while the other is `Some`.
///
/// `terminal_offset: None` means "resolved to nothing locally" — a REAL group,
/// never a missing value. Imputing it (as 0, as a mean, as the nearest offset)
/// would place every locally-unresolved row on top of the rows that resolved at
/// the focal, which is the exact confusion the tail exists to avoid. It is
/// therefore treated as its own category: zero distance to another `None`, a
/// flat [`TERMINUS_KIND_WEIGHT`] to any `Some`, and no numeric relationship to
/// the offset axis at all. The value equals the widest gap two termini can have
/// inside the ±8 window, so "resolved nowhere" sits at the far edge of the
/// terminus axis — not beyond it, which would let terminus kind outvote depth.
pub const TERMINUS_KIND_WEIGHT: u32 = 16;

/// Truncation applied to the offset difference.
///
/// Needed for the triangle inequality: without it a pair of far-apart `Some`
/// offsets could exceed the two-hop route through `None`
/// (`2 · TERMINUS_KIND_WEIGHT`) and [`trajectory_distance`] would not be a
/// metric. Inside the ±8 window the cap is unreachable, so it changes no real
/// reading — it only makes the metric claim true for every representable `i8`.
pub const OFFSET_CAP: u32 = 2 * TERMINUS_KIND_WEIGHT;

/// Fixed-point unit for [`DensityScore::anomaly`]. `1000` = exactly as dense as
/// the neighbourhood; above = sparser, i.e. more anomalous.
pub const DENSITY_SCALE: u32 = 1_000;

/// Saturation ceiling for [`DensityScore::anomaly`], so an isolated row in a
/// perfectly-collapsed neighbourhood reports "maximal" rather than overflowing.
pub const ANOMALY_CEILING: u32 = 1_000_000;

/// Distance between two causal trajectories.
///
/// A true metric (symmetric, zero iff identical, triangle inequality) — proven
/// by `metric_axioms_hold_over_the_sampled_space`. Being a metric is what lets
/// [`density_scores`] mean anything: a "local neighbourhood" is only well-defined
/// if closeness composes.
///
/// The three axes compose additively because they answer independent questions:
/// how deep (hops), in what mode (escalated), ending where (terminus).
#[must_use]
pub fn trajectory_distance(a: TrajectorySignature, b: TrajectorySignature) -> u32 {
    let hop = HOP_WEIGHT * u32::from(a.hops.abs_diff(b.hops));
    let esc = if a.escalated == b.escalated {
        0
    } else {
        ESCALATION_WEIGHT
    };
    let term = match (a.terminal_offset, b.terminal_offset) {
        (None, None) => 0,
        (Some(x), Some(y)) => u32::from(x.abs_diff(y)).min(OFFSET_CAP),
        // Categorical mismatch: never imputed, never scaled.
        _ => TERMINUS_KIND_WEIGHT,
    };
    hop + esc + term
}

/// Neighbourhood size and flag threshold for the density pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensityConfig {
    /// Neighbours per row. Clamped to `len - 1`; `0` when a row is alone.
    pub k: usize,
    /// Anomaly at or above which [`ranked_outlier_suggestions`] will SUGGEST a
    /// row the coarse path did not flag. `1500` = "half again as sparse as its
    /// own neighbourhood" — hand-tuned, not bound-derived, and said so.
    pub anomaly_threshold: u32,
}

impl Default for DensityConfig {
    fn default() -> Self {
        Self {
            k: 3,
            anomaly_threshold: 1_500,
        }
    }
}

/// One row's local density and the anomaly derived from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensityScore {
    pub row: GradedRow,
    /// Summed distance to the `k` nearest rows — LOW = dense, HIGH = sparse.
    pub reach: u32,
    /// `reach` relative to the mean `reach` of those same `k` neighbours, in
    /// [`DENSITY_SCALE`] units. This is the CHAODA move: anomaly is measured
    /// against the local manifold, not a global cut, so a legitimately sparse
    /// region does not report every one of its members as an outlier.
    pub anomaly: u32,
    /// How many neighbours the score was computed against — `0` means the row
    /// was alone and the score is the neutral [`DENSITY_SCALE`], never a flag.
    pub neighbours: usize,
}

/// Local density + relative anomaly for every row, over [`trajectory_distance`].
///
/// Deterministic end to end: integer arithmetic only, and neighbour selection
/// breaks distance ties on window index so the same input yields the same
/// neighbourhoods in the same order.
///
/// Never mutates its input and never removes a row — every input row gets a
/// score, including the ones the score will rank lowest.
#[must_use]
pub fn density_scores(rows: &[GradedRow], cfg: DensityConfig) -> Vec<DensityScore> {
    let n = rows.len();
    if n == 0 {
        return Vec::new();
    }
    let k = cfg.k.min(n - 1);

    // Pass 1 — k-nearest neighbours and the reach (summed distance) they imply.
    let mut nbrs: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut reach: Vec<u32> = Vec::with_capacity(n);
    for (i, ri) in rows.iter().enumerate() {
        let mut d: Vec<(u32, usize)> = rows
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(j, rj)| (trajectory_distance(ri.trajectory, rj.trajectory), j))
            .collect();
        // Explicit tie-break on position: equal distances must not reorder.
        d.sort_unstable();
        let take: Vec<usize> = d.iter().take(k).map(|&(_, j)| j).collect();
        reach.push(d.iter().take(k).map(|&(dist, _)| dist).sum());
        nbrs.push(take);
    }

    // Pass 2 — relative density. A row is anomalous when it is sparser than the
    // rows it is nearest to, which is a different claim from "far from the mean".
    rows.iter()
        .enumerate()
        .map(|(i, &row)| {
            let denom: u64 = nbrs[i].iter().map(|&j| u64::from(reach[j])).sum();
            let anomaly = if nbrs[i].is_empty() || denom == 0 {
                if reach[i] == 0 {
                    DENSITY_SCALE
                } else {
                    ANOMALY_CEILING
                }
            } else {
                let num = u64::from(reach[i]) * nbrs[i].len() as u64 * u64::from(DENSITY_SCALE);
                u32::try_from((num / denom).min(u64::from(ANOMALY_CEILING)))
                    .unwrap_or(ANOMALY_CEILING)
            };
            DensityScore {
                row,
                reach: reach[i],
                anomaly,
                neighbours: nbrs[i].len(),
            }
        })
        .collect()
}

/// How much of a perturbation SWEEP a basin survived.
///
/// A single nudge answers "did it survive THIS budget"; a fraction answers "how
/// budget-dependent is it", which is the question the anti-eigenvalue discipline
/// actually asks. `1000/1000` is a structure; `200/1000` is mostly an artifact
/// of the budget; the consumer decides where the line is, because the substrate
/// is not entitled to that judgement either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stability {
    /// Budgets at which the basin still shared one causal shape.
    pub stable: usize,
    /// Budgets probed.
    pub probed: usize,
}

impl Stability {
    /// Stable fraction in [`DENSITY_SCALE`] units (`1000` = survived every
    /// budget). An empty sweep reports `1000` — nothing falsified it.
    #[must_use]
    pub fn fraction_milli(&self) -> u32 {
        if self.probed == 0 {
            return DENSITY_SCALE;
        }
        u32::try_from(self.stable as u64 * u64::from(DENSITY_SCALE) / self.probed as u64)
            .unwrap_or(DENSITY_SCALE)
    }
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
    /// Sparser in trajectory space than the rows nearest to it — the metric
    /// reading the exact-match path cannot express. Only [`ranked_outlier_suggestions`]
    /// emits it; the coarse path is unchanged.
    DensityAnomaly,
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
    /// The row's [`DensityScore::anomaly`] within the tail it was judged in.
    ///
    /// This RANKS a suggestion; it never promotes one to a decision. A consumer
    /// reading only `reason` gets exactly the prior behaviour; a consumer
    /// reading `anomaly` gets an ordering over the same suggestions.
    pub anomaly: u32,
}

/// Grade every visible row of a lens: passive quorum + causal trajectory.
///
/// **Zero-copy (`zero-copy-lens-law.md`).** The row array IS the projection, so
/// this reads registers through [`WitnessLens::at`] — a bounds-checked cast into
/// each row's own bytes — instead of taking a caller-gathered
/// `&[(usize, CausalWitnessFacet)]` slab. No register byte is copied out.
///
/// The visible domain is `{ pos ∈ 0..lens.len() | visible(pos) }`, visited
/// **ascending**: [`GradedRow::idx`] is the dense counter over that set and
/// [`GradedRow::pos`] is the absolute row position. `visible` is what carries a
/// SPARSE selection — the gathered form could hold positions `[0, 1, 2, 10, 20]`
/// with no rows between, and the lens expresses exactly that as a row array with
/// `visible` false on the gaps.
#[must_use]
pub fn grade_rows(
    lens: &WitnessLens<'_>,
    visible: &impl Fn(usize) -> bool,
    locus: Locus,
    max_hops: u8,
) -> Vec<GradedRow> {
    (0..lens.len())
        .filter(|&pos| visible(pos))
        .enumerate()
        .map(|(idx, pos)| GradedRow {
            idx,
            pos,
            quorum: quorum_mantissa_lens(pos, lens, visible),
            trajectory: trajectory_of_lens(pos, lens, visible, locus, max_hops),
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
    minis.sort_by_key(|m| std::cmp::Reverse(m.members.len()));
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
    /// **Stable means this basin's exact member-index SET survives
    /// re-clustering the COMPLETE window at the perturbed budget** — not
    /// merely that the members it already had still agree with each other.
    /// Comparing only `self.members` (the earlier shape of this function)
    /// cannot see a MERGE: a row that sat OUTSIDE the basin can converge onto
    /// the same post-perturbation shape while every original member still
    /// agrees with every other original member, and an internal-only check
    /// reports "stable" for a basin whose true membership changed underneath
    /// it (CodeRabbit review, PR #852). So this re-derives the complete
    /// meta-clustering of `window` at `perturbed_hops` and requires one of its
    /// basins to carry EXACTLY this basin's index set — a merge (an outside
    /// row joined) and a split (a member left) are both reported `false`.
    ///
    /// Returns `true` for singletons (nothing to merge into or split from).
    ///
    /// This is the ONE-DIRECTION convenience wrapper over
    /// [`stability_sweep`](MetaBasin::stability_sweep). Testing a single nudge
    /// cannot distinguish "structure" from "survived the one budget I happened
    /// to pick"; prefer the sweep and read its fraction.
    #[must_use]
    pub fn stable_under_perturbation(
        &self,
        lens: &WitnessLens<'_>,
        visible: &impl Fn(usize) -> bool,
        locus: Locus,
        perturbed_hops: u8,
    ) -> bool {
        if self.members.len() < 2 {
            return true;
        }
        let mut want: Vec<usize> = self.members.iter().map(|m| m.idx).collect();
        want.sort_unstable();

        // Re-grade and re-cluster EVERY row of the window at the perturbed
        // budget — not just this basin's members — so a row that joins from
        // outside is visible. `quorum` is irrelevant to shape-clustering
        // (`meta_cluster` only reads `.trajectory`), so it is left at `0`.
        let reperturbed: Vec<GradedRow> = (0..lens.len())
            .filter(|&pos| visible(pos))
            .enumerate()
            .map(|(idx, pos)| GradedRow {
                idx,
                pos,
                quorum: 0,
                trajectory: trajectory_of_lens(pos, lens, visible, locus, perturbed_hops),
            })
            .collect();

        meta_cluster(&reperturbed).into_iter().any(|b| {
            let mut got: Vec<usize> = b.members.iter().map(|m| m.idx).collect();
            got.sort_unstable();
            got == want
        })
    }

    /// **Perturbation SWEEP** — survival across a range of hop budgets, as a
    /// fraction rather than a verdict.
    ///
    /// One nudge tests one budget; a basin can survive that nudge and dissolve
    /// at every other. Sweeping reports how budget-dependent the grouping is,
    /// which is the quantity the anti-eigenvalue guard actually wants. Budgets
    /// are probed in the order given and each is independent, so the result is
    /// deterministic.
    ///
    /// Singletons report full stability at every budget — there is nothing in a
    /// one-member basin that a budget could dissolve.
    #[must_use]
    pub fn stability_sweep(
        &self,
        lens: &WitnessLens<'_>,
        visible: &impl Fn(usize) -> bool,
        locus: Locus,
        budgets: &[u8],
    ) -> Stability {
        Stability {
            stable: budgets
                .iter()
                .filter(|&&p| self.stable_under_perturbation(lens, visible, locus, p))
                .count(),
            probed: budgets.len(),
        }
    }

    /// The sweep's default range: `max_hops.saturating_sub(8) ..=
    /// max_hops.saturating_add(2)` — a window CENTRED ON the caller's own
    /// budget, always including it. The old range (`0..=min(max_hops+2, 16)`)
    /// never probed the caller's own budget once `max_hops` exceeded 14, so a
    /// caller running at `max_hops = 255` got a fraction answering "how
    /// budget-dependent is this basin near budget 0-16", not "near the budget
    /// I actually use" — silently contradicting this doc's own claim. Naturally
    /// bounded to at most 11 probes without an extra cap, so it stays cheap at
    /// any `max_hops`.
    #[must_use]
    pub fn stability_around(
        &self,
        lens: &WitnessLens<'_>,
        visible: &impl Fn(usize) -> bool,
        locus: Locus,
        max_hops: u8,
    ) -> Stability {
        let budgets: Vec<u8> = stability_around_window(max_hops).collect();
        self.stability_sweep(lens, visible, locus, &budgets)
    }
}

/// The budget window [`MetaBasin::stability_around`] sweeps — factored out so
/// a test can assert the caller's own `max_hops` is actually inside it
/// (`stability_around_probes_the_callers_own_budget_even_at_255`) rather than
/// only asserting the call does not panic.
fn stability_around_window(max_hops: u8) -> std::ops::RangeInclusive<u8> {
    let lo = max_hops.saturating_sub(8);
    let hi = max_hops.saturating_add(2);
    lo..=hi
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
    lens: &WitnessLens<'_>,
    visible: &impl Fn(usize) -> bool,
    locus: Locus,
    max_hops: u8,
    perturbed_hops: u8,
    tail_below: u8,
) -> Vec<OutlierSuggestion> {
    let graded = grade_rows(lens, visible, locus, max_hops);
    let tail_rows = tail(&graded, tail_below);
    let scores = density_scores(&tail_rows, DensityConfig::default());
    coarse_flags(lens, visible, locus, perturbed_hops, &tail_rows)
        .into_iter()
        .map(|(row, reason, basin_size)| OutlierSuggestion {
            row,
            reason,
            basin_size,
            anomaly: anomaly_of(&scores, row.idx),
        })
        .collect()
}

/// The exact-match flagging the coarse path has always done, factored out so the
/// metric path can reuse it verbatim rather than restate it (a restated rule
/// drifts; a reused one cannot).
fn coarse_flags(
    lens: &WitnessLens<'_>,
    visible: &impl Fn(usize) -> bool,
    locus: Locus,
    perturbed_hops: u8,
    tail_rows: &[GradedRow],
) -> Vec<(GradedRow, OutlierReason, usize)> {
    let mut out = Vec::new();
    for basin in meta_cluster(tail_rows) {
        let size = basin.members.len();
        let stable = basin.stable_under_perturbation(lens, visible, locus, perturbed_hops);
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
                out.push((row, reason, size));
            }
        }
    }
    out
}

/// Neutral [`DENSITY_SCALE`] when a row has no score — a missing score is not
/// evidence of anomaly.
fn anomaly_of(scores: &[DensityScore], idx: usize) -> u32 {
    scores
        .iter()
        .find(|s| s.row.idx == idx)
        .map_or(DENSITY_SCALE, |s| s.anomaly)
}

/// **Suggest and RANK** — the metric path.
///
/// Subsumes [`outlier_suggestions`] and adds two things exact matching cannot
/// give: a [`OutlierReason::DensityAnomaly`] for rows that are sparse in
/// trajectory space without tripping any exact-match rule, and a deterministic
/// ORDER over all suggestions (descending anomaly, then window index — an
/// explicit tie-break, because a ranking that reorders between runs is not
/// auditable).
///
/// Exactly one suggestion per row: the coarse reason wins when a row has one, so
/// upgrading a caller from [`outlier_suggestions`] never silently reclassifies a
/// row it was already told about.
///
/// Still a SUGGESTION. The score orders the list; it does not license acting on
/// it, and nothing here prunes, commits, or mutates the window.
#[must_use]
pub fn ranked_outlier_suggestions(
    lens: &WitnessLens<'_>,
    visible: &impl Fn(usize) -> bool,
    locus: Locus,
    max_hops: u8,
    perturbed_hops: u8,
    tail_below: u8,
    cfg: DensityConfig,
) -> Vec<OutlierSuggestion> {
    let graded = grade_rows(lens, visible, locus, max_hops);
    let tail_rows = tail(&graded, tail_below);
    let scores = density_scores(&tail_rows, cfg);
    let coarse = coarse_flags(lens, visible, locus, perturbed_hops, &tail_rows);

    let mut out: Vec<OutlierSuggestion> = coarse
        .iter()
        .map(|&(row, reason, basin_size)| OutlierSuggestion {
            row,
            reason,
            basin_size,
            anomaly: anomaly_of(&scores, row.idx),
        })
        .collect();

    // Density-only suggestions: a row the exact-match rules never reached, whose
    // neighbourhood says it does not belong to it. `neighbours == 0` is excluded
    // — a row alone has no neighbourhood to be anomalous against.
    for s in &scores {
        if s.neighbours > 0
            && s.anomaly >= cfg.anomaly_threshold
            && !coarse.iter().any(|&(r, _, _)| r.idx == s.row.idx)
        {
            out.push(OutlierSuggestion {
                row: s.row,
                reason: OutlierReason::DensityAnomaly,
                basin_size: s.neighbours + 1,
                anomaly: s.anomaly,
            });
        }
    }

    out.sort_by(|a, b| b.anomaly.cmp(&a.anomaly).then(a.row.idx.cmp(&b.row.idx)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    use lance_graph_contract::canonical_node::{EdgeBlock, NodeGuid, NodeRow};
    use lance_graph_contract::causal_witness::CausalWitnessFacet;
    use lance_graph_contract::witness_fabric::{quorum_mantissa, trajectory_of};

    fn w(edges: &[(Locus, i8)]) -> CausalWitnessFacet {
        let mut f = CausalWitnessFacet::ZERO;
        for &(l, o) in edges {
            f = f.with(l, o);
        }
        f
    }

    /// A sparse `(pos, facet)` fixture rendered as the dense row array the lens
    /// projects. Positions the fixture skips exist as rows (a row array has no
    /// holes) but are excluded by [`vis_of`], which is how the lens expresses
    /// the sparse selection the gathered form carried in its position column.
    fn rows_from(regs: &[(usize, CausalWitnessFacet)]) -> Vec<NodeRow> {
        let max_pos = regs.iter().map(|&(p, _)| p).max().unwrap_or(0);
        let mut rows: Vec<NodeRow> = (0..=max_pos)
            .map(|_| NodeRow {
                key: NodeGuid::local(1),
                edges: EdgeBlock::default(),
                value: [0u8; 480],
            })
            .collect();
        for &(pos, facet) in regs {
            WitnessLens::write_register(&mut rows[pos], &facet);
        }
        rows
    }

    /// The visibility predicate for a sparse fixture: exactly the positions it
    /// names, so the lens domain equals the gathered window's position set.
    fn vis_of(regs: &[(usize, CausalWitnessFacet)]) -> impl Fn(usize) -> bool + '_ {
        move |p| regs.iter().any(|&(q, _)| q == p)
    }

    /// The PRE-MIGRATION gathered body, kept verbatim as the oracle the lens
    /// form is proven against. Retaining it is the point: an equivalence test
    /// whose reference is a paraphrase proves the paraphrase, not the migration.
    fn grade_rows_gathered(
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

    #[test]
    fn grading_carries_both_axes_and_the_tail_is_the_low_quorum_rows() {
        let win = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, CausalWitnessFacet::ZERO),
            (2, w(&[(Locus::Antecedent, 1)])),
        ];
        let rows = rows_from(&win);
        let lens = WitnessLens::new(&rows);
        let vis = vis_of(&win);
        let graded = grade_rows(&lens, &vis, Locus::Antecedent, 8);
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
        let rows = rows_from(&win);
        let lens = WitnessLens::new(&rows);
        let vis = vis_of(&win);
        let graded = grade_rows(&lens, &vis, Locus::Antecedent, 8);
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
        let rows = rows_from(&win);
        let lens = WitnessLens::new(&rows);
        let vis = vis_of(&win);
        let graded = grade_rows(&lens, &vis, Locus::Antecedent, 8);
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
        let rows = rows_from(&win);
        let lens = WitnessLens::new(&rows);
        let vis = vis_of(&win);
        let graded = grade_rows(&lens, &vis, Locus::Antecedent, 8);
        for b in meta_cluster(&graded) {
            // Singletons have nothing to dissolve — never reported unstable.
            if b.members.len() < 2 {
                assert!(b.stable_under_perturbation(&lens, &vis, Locus::Antecedent, 1));
            }
            // The call is total: any budget, no panic.
            for p in [0u8, 1, 2, 8, 255] {
                let _ = b.stable_under_perturbation(&lens, &vis, Locus::Antecedent, p);
            }
        }
    }

    /// **Falsifies the internal-only check a MERGE hides from** (CodeRabbit
    /// review, PR #852, finding 1 + finding 3): a basin whose ORIGINAL members
    /// still agree with EACH OTHER after perturbation, while a row that sat
    /// OUTSIDE the basin converges onto the very same post-perturbation shape.
    /// A `stable_under_perturbation` that compares only `self.members` cannot
    /// see the outside row and reports `true`; the fixed version re-clusters
    /// the whole window and must report `false`, because the basin's true
    /// member SET grew.
    ///
    /// Fixture: two independent 2-hop `Antecedent` chains (`0->1->2`,
    /// `10->11->12`) land in one meta-basin at `max_hops = 8`
    /// (`hops = 1, escalated = false` — each chain's first hop is bound, so it
    /// takes budget ≥ 2 to walk past it to the unbound terminal). A THIRD,
    /// unrelated 3-hop chain (`20->21->22->23`) sits in a DIFFERENT meta-basin
    /// at that budget (`hops = 2, escalated = false`). At the perturbed budget
    /// of `1`, every chain's first hop now exhausts the budget mid-walk, so
    /// ALL THREE chains — including the row starting the outside chain — read
    /// `hops = 1, escalated = true`: the outside row joins the basin's
    /// post-perturbation shape. This is a real, falsifiable dissolve (a
    /// merge), not an artifact of the fixture.
    #[test]
    fn stable_under_perturbation_catches_a_merge_the_internal_check_misses() {
        let win = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, w(&[(Locus::Antecedent, 1)])),
            (2, CausalWitnessFacet::ZERO),
            (10, w(&[(Locus::Antecedent, 1)])),
            (11, w(&[(Locus::Antecedent, 1)])),
            (12, CausalWitnessFacet::ZERO),
            (20, w(&[(Locus::Antecedent, 1)])),
            (21, w(&[(Locus::Antecedent, 1)])),
            (22, w(&[(Locus::Antecedent, 1)])),
            (23, CausalWitnessFacet::ZERO),
        ];
        let rows = rows_from(&win);
        let lens = WitnessLens::new(&rows);
        let vis = vis_of(&win);
        let graded = grade_rows(&lens, &vis, Locus::Antecedent, 8);
        let basins = meta_cluster(&graded);
        let basin = basins
            .iter()
            .find(|b| b.shape.hops == 1 && !b.shape.escalated)
            .expect("fixture must produce a hops=1, non-escalated meta-basin at budget 8");
        assert!(
            basin.members.len() >= 2,
            "fixture basin needs ≥2 members to exercise the internal-only false positive \
             (a single member trivially 'agrees with itself')"
        );

        // What the pre-fix, internal-only loop would have concluded: every
        // original member, taken pairwise, still shares a shape at the
        // perturbed budget. This is TRUE for this fixture (all three
        // original members truncate to the identical hops=1/escalated shape)
        // — which is exactly why the internal-only check is fooled.
        let mut shapes = basin
            .members
            .iter()
            .map(|m| trajectory_of(m.idx, &win, Locus::Antecedent, 1));
        let first = shapes.next().expect("basin has members");
        assert!(
            shapes.all(|s| first.same_meta_basin(s)),
            "fixture invariant broken: original members must still agree with EACH OTHER \
             post-perturbation for this to exercise the internal-only blind spot"
        );

        // The correct answer: unstable, because an outside row (the start of
        // the 3-hop chain) now shares that same post-perturbation shape too —
        // the basin's true membership grew.
        assert!(
            !basin.stable_under_perturbation(&lens, &vis, Locus::Antecedent, 1),
            "a row outside the basin converged onto its post-perturbation shape — \
             this is a merge, not stability, and must be reported false"
        );
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
        let rows = rows_from(&win);
        let lens = WitnessLens::new(&rows);
        let vis = vis_of(&win);
        let before = win.len();
        let sug = outlier_suggestions(&lens, &vis, Locus::Antecedent, 8, 2, 15);
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
        assert_eq!(
            sug,
            outlier_suggestions(&lens, &vis, Locus::Antecedent, 8, 2, 15)
        );
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
        let rows = rows_from(&win);
        let lens = WitnessLens::new(&rows);
        let vis = vis_of(&win);
        let sug = outlier_suggestions(&lens, &vis, Locus::Antecedent, 8, 2, 15);
        assert!(
            !sug.is_empty(),
            "outlier suggester never fires — inert channel"
        );
    }

    // ── the metric ───────────────────────────────────────────────────────

    /// A representative sweep of the trajectory space, including both terminus
    /// kinds and both escalation states.
    fn sample_space() -> Vec<TrajectorySignature> {
        let mut v = Vec::new();
        for hops in [0u8, 1, 3, 8, 255] {
            for escalated in [false, true] {
                for terminal_offset in [None, Some(-8i8), Some(0), Some(7), Some(127)] {
                    v.push(TrajectorySignature {
                        hops,
                        escalated,
                        terminal_offset,
                    });
                }
            }
        }
        v
    }

    /// Density scoring is only meaningful if closeness composes — so the claim
    /// that this IS a metric is asserted, not asserted-in-prose.
    #[test]
    fn metric_axioms_hold_over_the_sampled_space() {
        let s = sample_space();
        for &a in &s {
            assert_eq!(trajectory_distance(a, a), 0, "identity of indiscernibles");
            for &b in &s {
                assert_eq!(trajectory_distance(a, b), trajectory_distance(b, a));
                assert_eq!(
                    trajectory_distance(a, b) == 0,
                    a == b,
                    "zero distance must mean identical, not merely similar"
                );
                for &c in &s {
                    assert!(
                        trajectory_distance(a, c)
                            <= trajectory_distance(a, b) + trajectory_distance(b, c),
                        "triangle inequality violated: {a:?} {b:?} {c:?}"
                    );
                }
            }
        }
    }

    /// `None` is "resolved to nothing locally" — a real group. If it were ever
    /// imputed to `0`, this row would collapse onto the rows that resolved at
    /// the focal, which is the confusion the tail exists to keep apart.
    #[test]
    fn none_terminus_is_a_group_never_an_imputed_zero() {
        let none = TrajectorySignature {
            hops: 1,
            escalated: false,
            terminal_offset: None,
        };
        let zero = TrajectorySignature {
            terminal_offset: Some(0),
            ..none
        };
        let far = TrajectorySignature {
            terminal_offset: Some(7),
            ..none
        };
        assert_eq!(
            trajectory_distance(none, none),
            0,
            "None clusters with None"
        );
        assert_eq!(trajectory_distance(none, zero), TERMINUS_KIND_WEIGHT);
        // Categorical: the distance from None does NOT vary with the offset.
        assert_eq!(
            trajectory_distance(none, zero),
            trajectory_distance(none, far),
            "None was treated as a point on the offset axis"
        );
    }

    /// `escalated` is a MODE, not a magnitude: a flat cost, never scaled by the
    /// other axes.
    #[test]
    fn escalation_is_categorical_not_a_continuous_axis() {
        for hops in [0u8, 5, 200] {
            let a = TrajectorySignature {
                hops,
                escalated: false,
                terminal_offset: Some(1),
            };
            let b = TrajectorySignature {
                escalated: true,
                ..a
            };
            assert_eq!(trajectory_distance(a, b), ESCALATION_WEIGHT);
        }
    }

    /// The metric generalizes the shipped exact-match rule rather than replacing
    /// it: same shape ⟺ zero distance on the two shape axes.
    #[test]
    fn zero_shape_distance_agrees_with_same_meta_basin() {
        let s = sample_space();
        for &a in &s {
            for &b in &s {
                let shape_only = trajectory_distance(
                    TrajectorySignature {
                        terminal_offset: None,
                        ..a
                    },
                    TrajectorySignature {
                        terminal_offset: None,
                        ..b
                    },
                );
                assert_eq!(shape_only == 0, a.same_meta_basin(b));
            }
        }
    }

    // ── the density score ────────────────────────────────────────────────

    fn row(idx: usize, quorum: u8, hops: u8, escalated: bool, term: Option<i8>) -> GradedRow {
        GradedRow {
            idx,
            pos: idx,
            quorum,
            trajectory: TrajectorySignature {
                hops,
                escalated,
                terminal_offset: term,
            },
        }
    }

    /// A cluster plus one planted far row. A score that cannot separate them is
    /// as useless as a gate that never gates — the failure this workspace has
    /// now caught four times.
    #[test]
    fn density_score_discriminates_a_planted_outlier() {
        let mut rows: Vec<GradedRow> = (0..6)
            .map(|i| row(i, 1, 1 + u8::from(i % 2 == 0), false, Some(1)))
            .collect();
        // The plant: deep, escalating, resolved nowhere — far on all three axes.
        rows.push(row(6, 1, 40, true, None));

        let scores = density_scores(&rows, DensityConfig::default());
        let planted = scores.iter().find(|s| s.row.idx == 6).unwrap();
        for s in scores.iter().filter(|s| s.row.idx != 6) {
            assert!(
                planted.anomaly > s.anomaly,
                "planted outlier ({}) did not outscore cluster member {} ({})",
                planted.anomaly,
                s.row.idx,
                s.anomaly
            );
        }
        assert!(
            planted.anomaly > DENSITY_SCALE,
            "planted outlier scored as dense as its neighbourhood"
        );
    }

    /// Relative, not absolute: a uniformly-spread set has no outlier, because
    /// every row is exactly as sparse as its neighbours. A score that reported
    /// one anyway would be measuring size, not structure.
    #[test]
    fn a_uniform_spread_reports_no_anomaly() {
        let rows: Vec<GradedRow> = (0..6).map(|i| row(i, 1, i as u8, false, Some(0))).collect();
        for s in density_scores(&rows, DensityConfig::default()) {
            assert!(
                s.anomaly <= DensityConfig::default().anomaly_threshold,
                "uniform row {} flagged at {}",
                s.row.idx,
                s.anomaly
            );
        }
    }

    #[test]
    fn density_scores_are_deterministic_and_never_mutate_their_input() {
        let rows: Vec<GradedRow> = (0..7)
            .map(|i| row(i, i as u8 % 3, i as u8, i % 3 == 0, Some(i as i8 - 3)))
            .collect();
        let before = rows.clone();
        let a = density_scores(&rows, DensityConfig::default());
        let b = density_scores(&rows, DensityConfig::default());
        assert_eq!(a, b, "density scoring is not deterministic");
        assert_eq!(rows, before, "density scoring mutated its input");
        assert_eq!(a.len(), rows.len(), "a row was dropped from the scoring");
    }

    /// A lone row has no neighbourhood to be anomalous against — it must read
    /// neutral, never "maximally strange because it is alone".
    #[test]
    fn a_lone_row_is_neutral_not_anomalous() {
        let s = density_scores(&[row(0, 0, 3, true, None)], DensityConfig::default());
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].neighbours, 0);
        assert_eq!(s[0].anomaly, DENSITY_SCALE);
        assert!(density_scores(&[], DensityConfig::default()).is_empty());
    }

    // ── the perturbation sweep ───────────────────────────────────────────

    #[test]
    fn stability_is_a_fraction_and_the_bool_wrapper_still_agrees() {
        let win = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, CausalWitnessFacet::ZERO),
            (2, w(&[(Locus::Antecedent, 1)])),
            (3, w(&[(Locus::Antecedent, 7)])),
        ];
        let rows = rows_from(&win);
        let lens = WitnessLens::new(&rows);
        let vis = vis_of(&win);
        let graded = grade_rows(&lens, &vis, Locus::Antecedent, 8);
        let budgets: Vec<u8> = (0..=10).collect();
        for b in meta_cluster(&graded) {
            let sweep = b.stability_sweep(&lens, &vis, Locus::Antecedent, &budgets);
            assert_eq!(sweep.probed, budgets.len());
            assert!(sweep.stable <= sweep.probed);
            assert!(sweep.fraction_milli() <= DENSITY_SCALE);
            // The fraction is the sweep, not a restatement of one nudge: the
            // count must equal the number of budgets the bool wrapper accepts.
            let by_wrapper = budgets
                .iter()
                .filter(|&&p| b.stable_under_perturbation(&lens, &vis, Locus::Antecedent, p))
                .count();
            assert_eq!(sweep.stable, by_wrapper);
            // Singletons have nothing to dissolve at ANY budget.
            if b.members.len() < 2 {
                assert_eq!(sweep.fraction_milli(), DENSITY_SCALE);
            }
            // Deterministic, and total over the default range.
            assert_eq!(
                sweep,
                b.stability_sweep(&lens, &vis, Locus::Antecedent, &budgets)
            );
            let _ = b.stability_around(&lens, &vis, Locus::Antecedent, 255);
        }
        // An empty sweep falsifies nothing, so it claims full stability.
        let lone = MetaBasin {
            shape: TrajectorySignature::default(),
            members: vec![],
        };
        assert_eq!(
            lone.stability_sweep(&lens, &vis, Locus::Antecedent, &[])
                .fraction_milli(),
            DENSITY_SCALE
        );
    }

    /// **Falsifies the pre-fix `stability_around` range** (CodeRabbit review,
    /// PR #852, finding 2): the doc claimed the sweep "centres on the caller's
    /// own budget", but the old range was `0..=min(max_hops+2, 16)` — a
    /// caller running at `max_hops = 255` never had its OWN budget probed at
    /// all (255 is nowhere in `0..=16`). Under the old code this assertion
    /// fails; under the fixed windowing (`max_hops.saturating_sub(8) ..=
    /// max_hops.saturating_add(2)`) the caller's own budget is always inside.
    #[test]
    fn stability_around_probes_the_callers_own_budget_even_at_255() {
        let window = stability_around_window(255);
        assert!(
            window.contains(&255),
            "the caller's own budget (255) must be inside its own stability window, got {window:?}"
        );
        // Every budget in the window must be a legal probe, and the window
        // must stay bounded (never sweep the whole u8 range) even at the
        // largest possible max_hops.
        assert!(
            window.clone().count() <= 11,
            "the window around max_hops must stay small, got {} budgets",
            window.count()
        );
        // Sanity at the other end: a small max_hops still probes budget 0
        // (nothing below it is representable), matching the old behaviour
        // there.
        let small = stability_around_window(1);
        assert!(small.contains(&0) && small.contains(&1));
    }

    // ── the ranked surface ───────────────────────────────────────────────

    #[test]
    fn ranked_suggestions_fire_are_ordered_and_stay_advisory() {
        let win = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, CausalWitnessFacet::ZERO),
            (2, w(&[(Locus::Antecedent, 1)])),
            (3, CausalWitnessFacet::ZERO),
            (4, w(&[(Locus::Antecedent, 7)])),
            (5, w(&[(Locus::Kausal, -1)])),
        ];
        let rows = rows_from(&win);
        let lens = WitnessLens::new(&rows);
        let vis = vis_of(&win);
        let before = win.clone();
        let cfg = DensityConfig::default();
        let sug = ranked_outlier_suggestions(&lens, &vis, Locus::Antecedent, 8, 2, 15, cfg);
        assert!(
            !sug.is_empty(),
            "ranked suggester never fires — inert channel"
        );
        assert_eq!(win, before, "the ranked path mutated its window");
        // Descending anomaly with an explicit index tie-break.
        for pair in sug.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            assert!(
                a.anomaly > b.anomaly || (a.anomaly == b.anomaly && a.row.idx < b.row.idx),
                "ranking is not totally ordered"
            );
        }
        // One suggestion per row, each still evidenced.
        let mut seen: Vec<usize> = sug.iter().map(|s| s.row.idx).collect();
        seen.sort_unstable();
        let mut dedup = seen.clone();
        dedup.dedup();
        assert_eq!(seen, dedup, "a row was suggested twice");
        for s in &sug {
            assert!(s.basin_size >= 1);
            assert!(s.row.idx < win.len());
        }
        assert_eq!(
            sug,
            ranked_outlier_suggestions(&lens, &vis, Locus::Antecedent, 8, 2, 15, cfg),
            "ranking is not deterministic"
        );
    }

    /// The coarse path keeps its exact prior classification — the metric only
    /// adds; it never reclassifies a row a caller was already told about.
    #[test]
    fn ranked_path_subsumes_the_coarse_path_without_reclassifying() {
        let win = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, CausalWitnessFacet::ZERO),
            (2, w(&[(Locus::Antecedent, 1)])),
            (3, CausalWitnessFacet::ZERO),
            (4, w(&[(Locus::Antecedent, 7)])),
            (5, w(&[(Locus::Kausal, -1)])),
        ];
        let rows = rows_from(&win);
        let lens = WitnessLens::new(&rows);
        let vis = vis_of(&win);
        let coarse = outlier_suggestions(&lens, &vis, Locus::Antecedent, 8, 2, 15);
        let ranked = ranked_outlier_suggestions(
            &lens,
            &vis,
            Locus::Antecedent,
            8,
            2,
            15,
            DensityConfig::default(),
        );
        for c in &coarse {
            let r = ranked
                .iter()
                .find(|r| r.row.idx == c.row.idx)
                .expect("ranked path dropped a coarse suggestion");
            assert_eq!(r.reason, c.reason, "coarse reason was reclassified");
            assert_eq!(r.basin_size, c.basin_size);
        }
        assert!(ranked.len() >= coarse.len());
    }

    /// **The migration's proof** — the lens form reproduces the PRE-MIGRATION
    /// gathered body exactly, over two fixtures chosen so that BOTH graded axes
    /// are actually exercised.
    ///
    /// The risk the sparse fixture pins is SPARSITY. `resolve_chain` walks hops
    /// by absolute stream POSITION (`cur_pos + off`), so a gathered window could
    /// name positions `[0,1,2, 10,11,12, 20,21,22,23]` with nothing in between;
    /// the lens instead indexes a dense row array, and the gaps must be excluded
    /// by `visible` rather than by simply not existing. If those two notions of
    /// "not addressable" ever diverged, that fixture is where it would show.
    ///
    /// But a sparse window has NO quorum: no two rows are close enough to agree
    /// on an absolute target, so every row grades `quorum = 0` and the quorum
    /// half of the comparison is vacuous. (That is not a hypothesis — the
    /// anti-vacuity assert below caught exactly this while the migration was
    /// being written, on a version of this test that used the sparse fixture
    /// alone.) The dense fixture supplies the agreement the sparse one cannot.
    #[test]
    fn lens_grading_matches_the_gathered_oracle_on_sparse_and_dense_windows() {
        // Sparse: three chains with gaps between them; every chain hops into a
        // position the window does not name.
        let sparse = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (1, w(&[(Locus::Antecedent, 1)])),
            (2, CausalWitnessFacet::ZERO),
            (10, w(&[(Locus::Antecedent, 1)])),
            (11, w(&[(Locus::Antecedent, 1)])),
            (12, CausalWitnessFacet::ZERO),
            (20, w(&[(Locus::Antecedent, 1)])),
            (21, w(&[(Locus::Antecedent, 1)])),
            (22, w(&[(Locus::Antecedent, 1)])),
            (23, CausalWitnessFacet::ZERO),
        ];
        // Dense: rows 0 and 1 converge on absolute position 2 across FOUR content
        // loci, so they grade a non-zero quorum; row 2 agrees with nobody and
        // grades 0. Four loci is not decoration — `quorum_mantissa` scales
        // `agreed * 15 / (peers * 14)` and rounds DOWN, so a single agreeing
        // locus floors to 0 and the axis would still be untested.
        let dense = vec![
            (
                0,
                w(&[
                    (Locus::Temporal, 2),
                    (Locus::Kausal, 2),
                    (Locus::Modal, 2),
                    (Locus::Lokal, 2),
                ]),
            ),
            (
                1,
                w(&[
                    (Locus::Temporal, 1),
                    (Locus::Kausal, 1),
                    (Locus::Modal, 1),
                    (Locus::Lokal, 1),
                ]),
            ),
            (2, w(&[(Locus::Antecedent, 1)])),
        ];

        let mut saw_escalation = false;
        let mut quorums = std::collections::BTreeSet::new();
        let mut saw_sparsity = false;

        for (label, win) in [("sparse", &sparse), ("dense", &dense)] {
            let rows = rows_from(win);
            let lens = WitnessLens::new(&rows);
            let vis = vis_of(win);
            saw_sparsity |= rows.len() > win.len();

            for hops in [0u8, 1, 2, 3, 8, 255] {
                let gathered = grade_rows_gathered(win, Locus::Antecedent, hops);
                let lensed = grade_rows(&lens, &vis, Locus::Antecedent, hops);
                assert_eq!(
                    gathered.len(),
                    lensed.len(),
                    "{label}: row count diverged at max_hops={hops}"
                );
                for (g, l) in gathered.iter().zip(lensed.iter()) {
                    assert_eq!(g, l, "{label}: grading diverged at max_hops={hops}");
                    saw_escalation |= g.trajectory.escalated;
                    quorums.insert(g.quorum);
                }
            }
        }

        // Anti-vacuity, one clause per axis the comparison claims to cover. An
        // all-identical grading would make the equality pass for reasons that
        // have nothing to do with the migration.
        assert!(
            saw_sparsity,
            "no fixture was actually sparse — the gap-exclusion path went unchecked"
        );
        assert!(
            saw_escalation,
            "no fixture escalated — the escalation axis went unchecked"
        );
        assert!(
            quorums.len() > 1,
            "every row got the same quorum ({quorums:?}) — the quorum axis went unchecked"
        );
    }

    /// **Cost characterization (Codex P2 on #868).** Pins the complexity SHAPE
    /// of the lens form so a regression — or an improvement — is visible rather
    /// than argued.
    ///
    /// `visible` is invoked once per candidate position, so counting its calls
    /// measures the scan exactly and deterministically (a wall-clock assert
    /// would be flaky and would measure the machine).
    ///
    /// The honest trade, both directions:
    /// * **quorum got worse.** `quorum_mantissa_lens` scans `0..lens.len()`
    ///   where the gathered `quorum_mantissa` scanned the `k`-entry window, so
    ///   peer work goes Θ(k²) → Θ(N·k).
    /// * **trajectory got better.** Gathered `resolve_chain` resolved each hop
    ///   with `window.iter().position(..)`, a linear O(k) scan PER HOP;
    ///   `resolve_chain_lens` uses `lens.at(pos)`, which is O(1). So hop work
    ///   goes Θ(hops·k) → Θ(hops).
    ///
    /// Net per graded row: gathered `k·(1 + hops)` vs lens `N + hops`. The lens
    /// WINS whenever `N < k·(1 + hops) - hops` (dense windows, deep chains) and
    /// LOSES when a small window is viewed through a large row array — which is
    /// exactly the case Codex flagged. Tracked as
    /// `TD-LENS-QUORUM-SCANS-THE-WHOLE-LENS`.
    #[test]
    fn grading_cost_scales_with_lens_length_not_window_size() {
        use std::cell::Cell;

        const N: usize = 512;
        const K: usize = 8;
        let positions: Vec<usize> = (0..K).map(|i| i * (N / K)).collect();
        let win: Vec<(usize, CausalWitnessFacet)> = positions
            .iter()
            .map(|&p| (p, w(&[(Locus::Antecedent, 1)])))
            .collect();

        let mut rows = rows_from(&win);
        rows.resize_with(N, || NodeRow {
            key: NodeGuid::local(1),
            edges: EdgeBlock::default(),
            value: [0u8; 480],
        });
        let lens = WitnessLens::new(&rows);
        assert_eq!(lens.len(), N, "the lens must span the whole row array");

        let calls = Cell::new(0usize);
        let vis = |p: usize| {
            calls.set(calls.get() + 1);
            positions.contains(&p)
        };
        let graded = grade_rows(&lens, &vis, Locus::Antecedent, 8);
        assert_eq!(graded.len(), K, "only the visible rows are graded");

        // The shape, not a magic number: at least one full sweep per graded row
        // (each `quorum_mantissa_lens` call) plus the outer scan.
        let observed = calls.get();
        assert!(
            observed >= N * K,
            "expected the documented Theta(N*k) scan, saw {observed} for N={N} k={K}"
        );
        // ...and NOT quadratic-in-N, which would be a different defect entirely.
        assert!(
            observed < N * N,
            "scan is superlinear in N ({observed} for N={N}) — that is not the \
             documented shape and needs investigating, not re-baselining"
        );
    }

    /// A position the fixture SKIPS must read as unaddressable, exactly as a
    /// position absent from a gathered window did. This is the half the
    /// equivalence test above cannot state directly: it proves the gaps are
    /// excluded because `visible` says so, not because the rows are empty.
    #[test]
    fn an_invisible_gap_is_excluded_even_though_the_row_exists() {
        let win = vec![
            (0, w(&[(Locus::Antecedent, 1)])),
            (2, CausalWitnessFacet::ZERO),
        ];
        let rows = rows_from(&win);
        let lens = WitnessLens::new(&rows);
        let vis = vis_of(&win);

        assert_eq!(
            lens.len(),
            3,
            "row 1 must EXIST for the exclusion to mean anything"
        );
        assert!(lens.at(1).is_some(), "row 1 is addressable by the lens");
        assert!(!vis(1), "row 1 must be invisible");

        let graded = grade_rows(&lens, &vis, Locus::Antecedent, 8);
        assert_eq!(graded.len(), 2, "the invisible row leaked into the grading");
        assert_eq!(
            graded.iter().map(|g| g.pos).collect::<Vec<_>>(),
            vec![0, 2],
            "positions must be the visible ones, ascending"
        );
        // `idx` is the dense counter over the VISIBLE set, not the position.
        assert_eq!(graded.iter().map(|g| g.idx).collect::<Vec<_>>(), vec![0, 1]);
    }
}
