//! **Stage 2.5 — the consumer-filter census.** Measurement only; compiled out
//! of every non-test build (`#[cfg(test)]` at the `mod` line).
//!
//! Stage 2 shipped a maturity clause in both dissent channels' eligibility
//! predicate and reported one sentence: *the filter changes which watchers are
//! sampled but changed no verdict across 5,760 configurations.* Stage 2.5
//! exists to characterise that sentence quantitatively — and, specifically, to
//! keep its two halves apart. **"Filter changes the watcher sample" and "filter
//! changes the verdict" are two dependent variables, not one**; collapsing them
//! is how a coverage fix gets written up as "no effect".
//!
//! # What is measured, and what is NOT
//!
//! Every configuration `x = (style, rung, k, tol)` is a PAIRED observation:
//! the OFF arm samples on the mechanism clause alone, the ON arm adds the
//! maturity clause, and both arms then run the SAME shipped verdict body
//! ([`StyleStrategy::dissent_over`]). Nothing here reimplements the channel —
//! that is the whole reason the body was extracted, and the extraction is
//! pinned verdict-identical (0/5,760) by the Stage-2 arc.
//!
//! **No inferential test is run over the enumeration, on purpose.** The census
//! is EXHAUSTIVE and DETERMINISTIC: every cell of the design is visited exactly
//! once and the same input always produces the same output, so there is no
//! sampling distribution for a p-value to be a statement about. `jc::stats` is
//! used here for what it can honestly say on a census — cross-tabulation
//! ([`binary_association`], which carries the marginals and κ), variance
//! DECOMPOSITION ([`eta_squared`], [`multiple_r_squared`]), and rank/linear
//! association ([`spearman`], [`pearson`]). `t_test_*` / `anova_one_way`'s
//! p-values are deliberately not reported: manufacturing significance where the
//! statistic is degenerate is the failure this module is supposed to avoid, not
//! commit.
//!
//! The one genuinely inferential quantity IS reported, because it is the
//! question worth asking: **given zero observed verdict flips, how large could
//! the true flip rate still plausibly be?** That is a one-sided exact binomial
//! (Clopper-Pearson) upper limit, which at zero events has the closed form
//! `1 − α^(1/n)` and needs no incomplete-beta evaluation — and it is reported as
//! a LADDER over clustering assumptions, because the 5,760 rows are repeated
//! measures, not 5,760 independent trials (one style contributes 160 of them).
//!
//! # What is NOT available, and was not added
//!
//! Part C of the Stage-2.5 brief asks for a pre-verdict numeric — the
//! `|tc.confidence − admitted|` margin the channel compares against `tol`. That
//! value is local to [`StyleStrategy::dissent_over`] and is not returned.
//! **It was not exposed for this measurement**, per the brief's own guardrail.
//! What IS naturally available is finer than the boolean and is measured here:
//! the elevation `RungLevel` (which watcher objected) and, on the cross-family
//! channel, the reported `Mechanism`. Four paired outcome surfaces in total, not
//! one.

use std::collections::BTreeSet;

use lance_graph_contract::cognitive_shader::RungLevel;
use lance_graph_contract::recipes::Mechanism;
use lance_graph_contract::thinking::ThinkingStyle;

use super::StyleStrategy;
use crate::traits::PlanContext;

/// The `k` budgets swept. Chosen to span the sampler's regimes: `k = 1` takes
/// only the stride's first element, `k = 8` is what every shipped call site and
/// every Stage-2 test uses.
const KS: [usize; 5] = [1, 2, 3, 4, 8];

/// The tolerances swept. `0.0` is "any movement at all counts"; `0.2` is well
/// past the largest single-kernel confidence step in the carved population.
const TOLS: [f32; 8] = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2];

/// The rungs with a non-empty periphery. `Transcendent` excludes nothing, so it
/// has no watchers to filter and would contribute only tautological rows —
/// `no_periphery_no_dissent` already pins that separately.
const RUNGS: [RungLevel; 4] = [
    RungLevel::Surface,
    RungLevel::Shallow,
    RungLevel::Contextual,
    RungLevel::Analogical,
];

/// One paired observation: the same configuration measured with the maturity
/// clause OFF and ON.
#[derive(Clone, Debug)]
struct Pair {
    style: ThinkingStyle,
    style_idx: usize,
    rung: RungLevel,
    k: usize,
    tol: f32,
    /// Sampled watcher recipe ids, same-family channel.
    same_off: BTreeSet<u8>,
    same_on: BTreeSet<u8>,
    /// Sampled watcher recipe ids, cross-family channel.
    cross_off: BTreeSet<u8>,
    cross_on: BTreeSet<u8>,
    /// Verdicts — the elevation target, which is strictly finer than fired/not.
    v_same_off: Option<RungLevel>,
    v_same_on: Option<RungLevel>,
    v_cross_off: Option<(RungLevel, Mechanism)>,
    v_cross_on: Option<(RungLevel, Mechanism)>,
}

/// Set-comparison summary for one channel of one pair.
#[derive(Clone, Copy, Debug)]
struct SetDelta {
    n_off: usize,
    n_on: usize,
    intersection: usize,
    symmetric_difference: usize,
    added: usize,
    removed: usize,
}

impl SetDelta {
    fn of(off: &BTreeSet<u8>, on: &BTreeSet<u8>) -> Self {
        let inter = off.intersection(on).count();
        Self {
            n_off: off.len(),
            n_on: on.len(),
            intersection: inter,
            symmetric_difference: off.symmetric_difference(on).count(),
            added: on.difference(off).count(),
            removed: off.difference(on).count(),
        }
    }
    fn union(&self) -> usize {
        self.n_off + self.n_on - self.intersection
    }
    /// Jaccard similarity `|A∩B| / |A∪B|`. Two EMPTY samples are defined as
    /// identical (`1.0`) rather than `0/0`: "the filter changed nothing here"
    /// is the true statement about a cell with no watchers on either side, and
    /// scoring it `0` would report maximal change where none occurred.
    fn jaccard(&self) -> f64 {
        let u = self.union();
        if u == 0 {
            1.0
        } else {
            self.intersection as f64 / u as f64
        }
    }
    fn jaccard_distance(&self) -> f64 {
        1.0 - self.jaccard()
    }
    /// Fraction of the OFF sample that survived into ON.
    fn retention(&self) -> f64 {
        if self.n_off == 0 {
            1.0
        } else {
            self.intersection as f64 / self.n_off as f64
        }
    }
}

/// Enumerate the full paired census. Deterministic; no I/O.
fn census() -> Vec<Pair> {
    let ctx = probe_ctx();
    let mut out =
        Vec::with_capacity(ThinkingStyle::ALL.len() * RUNGS.len() * KS.len() * TOLS.len());
    for (style_idx, &style) in ThinkingStyle::ALL.iter().enumerate() {
        let want = StyleStrategy::cluster_mechanism(style.cluster());
        for rung in RUNGS {
            for k in KS {
                // The four samples depend only on (rung, want, k) — hoisted out
                // of the tol loop so the census cannot accidentally measure
                // sampler nondeterminism it does not have.
                let ids = |same: bool, maturity: bool| -> BTreeSet<u8> {
                    rung.peripheral_sample_where(k, |r| {
                        (r.mechanism == want) == same
                            && (!maturity || StyleStrategy::watcher_can_dissent(r.id))
                    })
                    .map(|r| r.id)
                    .collect()
                };
                let same_off = ids(true, false);
                let same_on = ids(true, true);
                let cross_off = ids(false, false);
                let cross_on = ids(false, true);

                for tol in TOLS {
                    let admitted = StyleStrategy::reliability_at(style, &ctx, rung);
                    let run = |set: &BTreeSet<u8>| {
                        StyleStrategy::dissent_over(
                            style,
                            &ctx,
                            rung,
                            tol,
                            admitted,
                            recipes_by_id(set),
                        )
                    };
                    out.push(Pair {
                        style,
                        style_idx,
                        rung,
                        k,
                        tol,
                        v_same_off: run(&same_off).map(|w| w.min_rung()),
                        v_same_on: run(&same_on).map(|w| w.min_rung()),
                        v_cross_off: run(&cross_off).map(|w| (w.min_rung(), w.mechanism)),
                        v_cross_on: run(&cross_on).map(|w| (w.min_rung(), w.mechanism)),
                        same_off: same_off.clone(),
                        same_on: same_on.clone(),
                        cross_off: cross_off.clone(),
                        cross_on: cross_on.clone(),
                    });
                }
            }
        }
    }
    out
}

/// Resolve ids back to recipes in the sampler's own ascending-id order.
///
/// The channels iterate their sample in whatever order `peripheral_sample_where`
/// yields, which for a stride over an ascending list IS ascending; a `BTreeSet`
/// preserves that, so the census feeds `dissent_over` the same order production
/// does. (Order matters: the body returns the FIRST objector.)
fn recipes_by_id(ids: &BTreeSet<u8>) -> Vec<&'static lance_graph_contract::recipes::Recipe> {
    ids.iter()
        .filter_map(|&id| {
            lance_graph_contract::recipes::RECIPES
                .iter()
                .find(|r| r.id == id)
        })
        .collect()
}

/// The probe context. ONE fixed context for the whole census: the experiment
/// varies the FILTER, and holding everything else constant is what makes the
/// pairs paired.
///
/// The same shape the Stage-2 dissent tests use (`ctx_with(style_vec(0.9, 0,
/// 0))` — an analytical-dominant 23-D style vector). Built here rather than
/// reached for across the test module boundary, so the census does not depend
/// on a test helper's visibility.
fn probe_ctx() -> PlanContext {
    let mut style = vec![0.0; 23];
    style[4] = 0.9; // analytical
    PlanContext {
        query: "MATCH (n:Person) RETURN n".into(),
        features: crate::traits::QueryFeatures::default(),
        free_will_modifier: 0.7,
        thinking_style: Some(style),
        nars_hint: None,
        witness: None,
    }
}

// ── statistics ───────────────────────────────────────────────────────────────

/// One-sided exact (Clopper-Pearson) upper limit on the event probability when
/// **zero** events were observed in `n` trials.
///
/// At `k = 0` the general Clopper-Pearson limit collapses to a closed form:
/// the upper limit `p_u` solves `P(X = 0 | p_u) = (1 − p_u)^n = α`, hence
/// `p_u = 1 − α^(1/n)`. No incomplete-beta evaluation is needed, which is why
/// this is three lines here rather than a gap in `jc::stats` (JC carries no
/// binomial interval, and adding one for a degenerate case is not the job).
fn zero_event_upper_bound(n: usize, alpha: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    1.0 - alpha.powf(1.0 / n as f64)
}

/// Group a value column by an integer stratum key, for `jc::stats::eta_squared`.
fn group_by(keys: &[usize], values: &[f64]) -> Vec<Vec<f64>> {
    let n_groups = keys.iter().copied().max().map_or(0, |m| m + 1);
    let mut groups = vec![Vec::new(); n_groups];
    for (&kk, &v) in keys.iter().zip(values) {
        groups[kk].push(v);
    }
    groups.retain(|g| !g.is_empty());
    groups
}

// ── the report ───────────────────────────────────────────────────────────────

struct ChannelStats {
    label: &'static str,
    deltas: Vec<SetDelta>,
    jaccard_distance: Vec<f64>,
    fired_off: Vec<bool>,
    fired_on: Vec<bool>,
    /// Elevation target as a nominal label (`0` = no dissent, else `rung + 1`).
    label_off: Vec<usize>,
    label_on: Vec<usize>,
}

fn channel_stats(pairs: &[Pair], same_family: bool) -> ChannelStats {
    let mut cs = ChannelStats {
        label: if same_family {
            "same-family (peripheral_dissent)"
        } else {
            "cross-family (cross_family_dissent)"
        },
        deltas: Vec::new(),
        jaccard_distance: Vec::new(),
        fired_off: Vec::new(),
        fired_on: Vec::new(),
        label_off: Vec::new(),
        label_on: Vec::new(),
    };
    for p in pairs {
        let (off, on) = if same_family {
            (&p.same_off, &p.same_on)
        } else {
            (&p.cross_off, &p.cross_on)
        };
        let d = SetDelta::of(off, on);
        cs.jaccard_distance.push(d.jaccard_distance());
        cs.deltas.push(d);
        let (vo, vn): (Option<usize>, Option<usize>) = if same_family {
            (
                p.v_same_off.map(|r| r as usize + 1),
                p.v_same_on.map(|r| r as usize + 1),
            )
        } else {
            (
                p.v_cross_off.map(|(r, _)| r as usize + 1),
                p.v_cross_on.map(|(r, _)| r as usize + 1),
            )
        };
        cs.fired_off.push(vo.is_some());
        cs.fired_on.push(vn.is_some());
        cs.label_off.push(vo.unwrap_or(0));
        cs.label_on.push(vn.unwrap_or(0));
    }
    cs
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jc::reliability::spearman;
    use jc::stats::{binary_association, cohen_kappa, eta_squared, multiple_r_squared};

    /// The census is the size the design says it is. A silently-shrunk
    /// enumeration would make every number below look better than it is.
    #[test]
    fn stage25_census_covers_the_full_design() {
        let pairs = census();
        assert_eq!(
            pairs.len(),
            ThinkingStyle::ALL.len() * RUNGS.len() * KS.len() * TOLS.len()
        );
        assert_eq!(pairs.len(), 5760, "the design is 36 x 4 x 5 x 8");
    }

    /// **The headline, both halves, pinned.**
    ///
    /// This is the falsifier for the Stage-2 claim itself: the filter's watcher
    /// effect must be non-trivial (or the coverage argument is decoration) AND
    /// the verdict effect must be exactly zero (or the "no behaviour change"
    /// documentation at the call site is false). Either half moving forces a
    /// deliberate re-measure rather than a silent drift.
    #[test]
    fn stage25_headline_watcher_delta_is_real_and_verdict_delta_is_zero() {
        let pairs = census();

        for same_family in [true, false] {
            let cs = channel_stats(&pairs, same_family);

            // (a) the watcher sample really moves
            let changed = cs
                .deltas
                .iter()
                .filter(|d| d.symmetric_difference > 0)
                .count();
            assert!(
                changed > 0,
                "{}: the maturity clause changed no watcher sample anywhere — the \
                 coverage claim is decoration",
                cs.label
            );

            // (b) and no verdict moves, on the FINE label (elevation target),
            //     which is strictly stronger than fired/not-fired
            let discordant = cs
                .label_off
                .iter()
                .zip(&cs.label_on)
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                discordant,
                0,
                "{}: {discordant} of {} paired configurations changed verdict — the \
                 call-site documentation says none do",
                cs.label,
                pairs.len()
            );
        }
    }

    /// Cohen's κ is **defined** here — both outcome categories occur — so the
    /// perfect agreement is a real 1.0 and not the degenerate `p_e == 1` case
    /// `jc::stats::cohen_kappa` correctly refuses to score.
    ///
    /// Without this the zero-discordance result above could be an artifact of a
    /// constant column, which is exactly the "a guard that fires on everything
    /// carries as much information as one that never fires" trap.
    #[test]
    fn stage25_agreement_is_perfect_and_not_degenerate() {
        let pairs = census();
        for same_family in [true, false] {
            let cs = channel_stats(&pairs, same_family);
            let t = binary_association(&cs.fired_off, &cs.fired_on)
                .expect("non-empty, equal-length columns");
            assert_eq!(
                (t.n10, t.n01),
                (0, 0),
                "{}: off-diagonal must be empty",
                cs.label
            );
            assert!(
                t.n11 > 0 && t.n00 > 0,
                "{}: both outcomes must occur or agreement is vacuous (n11={}, n00={})",
                cs.label,
                t.n11,
                t.n00
            );
            let k = t
                .kappa
                .expect("kappa is defined when both categories occur");
            assert!((k - 1.0).abs() < 1e-12, "{}: kappa = {k}", cs.label);

            // ...and the same on the finer elevation-target label.
            let k_fine = cohen_kappa(&cs.label_off, &cs.label_on)
                .expect("kappa defined on the fine label too");
            assert!(
                (k_fine - 1.0).abs() < 1e-12,
                "{}: fine kappa = {k_fine}",
                cs.label
            );
        }
    }

    /// **The tolerance cannot move the watcher sample.** Structural — the
    /// sample depends only on `(rung, want, k)` — and pinned because the
    /// report leans on it when it declines to read `tol`'s eta-squared of
    /// exactly 0.0000 as a finding about tolerance. If the two ever couple,
    /// the census's factor table becomes misleading and this fails first.
    #[test]
    fn stage25_tolerance_cannot_move_the_watcher_sample() {
        let pairs = census();
        // Group by everything EXCEPT tol; every group must be constant.
        for chunk in pairs.chunks(TOLS.len()) {
            let head = &chunk[0];
            for p in chunk {
                assert_eq!(p.style_idx, head.style_idx, "chunking assumption broken");
                assert_eq!(p.k, head.k, "chunking assumption broken");
                assert_eq!(
                    (&p.same_off, &p.same_on, &p.cross_off, &p.cross_on),
                    (
                        &head.same_off,
                        &head.same_on,
                        &head.cross_off,
                        &head.cross_on
                    ),
                    "tol moved the watcher sample at style {} rung {:?} k {}",
                    p.style_idx,
                    p.rung,
                    p.k
                );
            }
        }
        // Anti-vacuity: the sweep must contain more than one tolerance, or the
        // constancy above is a statement about a single column.
        assert!(TOLS.len() > 1 && pairs.len() > TOLS.len());
    }

    /// The zero-event bound is the exact Clopper-Pearson limit and is
    /// monotone in `n` — pinned so the ladder in the report cannot be read
    /// upside down.
    #[test]
    fn stage25_zero_event_bound_is_exact_and_monotone() {
        // (1 - p)^n = alpha  =>  p = 1 - alpha^(1/n)
        let p = zero_event_upper_bound(5760, 0.05);
        assert!(
            ((1.0 - p).powi(5760) - 0.05).abs() < 1e-9,
            "not the exact limit: {p}"
        );
        assert!(
            zero_event_upper_bound(36, 0.05) > zero_event_upper_bound(5760, 0.05),
            "fewer independent units must give a WEAKER (larger) bound"
        );
        assert_eq!(
            zero_event_upper_bound(0, 0.05),
            1.0,
            "no trials bound nothing"
        );
    }

    /// **Write the artifacts.** Ignored by default — it is a report generator,
    /// not a gate; the gates are the three tests above.
    ///
    /// `cargo test -p lance-graph-planner --lib stage25_census -- --ignored --nocapture`
    #[test]
    #[ignore = "artifact generator; run explicitly"]
    fn stage25_write_artifacts() {
        let pairs = census();
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/probes");
        std::fs::create_dir_all(dir).expect("create docs/probes");

        // ── machine-readable, and actually compact ──
        //
        // Two files, split along the census's own structure rather than dumped
        // as one row-per-cell table (that was 11,520 rows / ~1 MB, and 7 of
        // every 8 rows were byte-identical in every column that varies).
        //
        // A-surface: collapsed over `tol`. **Lossless** — the sampled watcher
        // set depends only on `(rung, want, k)`, pinned by
        // `stage25_tolerance_cannot_move_the_watcher_sample`, so the eight
        // tolerance rows of a cell carry one observation between them.
        let mut csv = String::from(
            "style_idx,style,cluster,rung,k,channel,n_off,n_on,count_delta,intersection,\
             sym_diff,added,removed,jaccard,jaccard_distance,retention,ids_off,ids_on\n",
        );
        let mut seen: Vec<(usize, RungLevel, usize)> = Vec::new();
        for p in &pairs {
            let key = (p.style_idx, p.rung, p.k);
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            for same_family in [true, false] {
                let (off, on) = if same_family {
                    (&p.same_off, &p.same_on)
                } else {
                    (&p.cross_off, &p.cross_on)
                };
                let d = SetDelta::of(off, on);
                let ids = |set: &BTreeSet<u8>| {
                    set.iter().map(u8::to_string).collect::<Vec<_>>().join(" ")
                };
                csv.push_str(&format!(
                    "{},{:?},{:?},{:?},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{},{}\n",
                    p.style_idx,
                    p.style,
                    p.style.cluster(),
                    p.rung,
                    p.k,
                    if same_family { "same" } else { "cross" },
                    d.n_off,
                    d.n_on,
                    d.n_on as i64 - d.n_off as i64,
                    d.intersection,
                    d.symmetric_difference,
                    d.added,
                    d.removed,
                    d.jaccard(),
                    d.jaccard_distance(),
                    d.retention(),
                    ids(off),
                    ids(on),
                ));
            }
        }
        std::fs::write(format!("{dir}/stage25-consumer-filter-census.csv"), csv)
            .expect("write census csv");

        // B-surface: ONE row per DISCORDANT pair. Empty (header only) is the
        // result, not a missing file — and it is the shape that stays useful if
        // a future run does flip something, where a 11,520-row all-concordant
        // dump would bury the one row that mattered.
        let mut disc = String::from("style_idx,style,rung,k,tol,channel,verdict_off,verdict_on\n");
        for p in &pairs {
            for same_family in [true, false] {
                let (vo, vn) = if same_family {
                    (
                        p.v_same_off.map(|r| r as usize + 1).unwrap_or(0),
                        p.v_same_on.map(|r| r as usize + 1).unwrap_or(0),
                    )
                } else {
                    (
                        p.v_cross_off.map(|(r, _)| r as usize + 1).unwrap_or(0),
                        p.v_cross_on.map(|(r, _)| r as usize + 1).unwrap_or(0),
                    )
                };
                if vo != vn {
                    disc.push_str(&format!(
                        "{},{:?},{:?},{},{},{},{vo},{vn}\n",
                        p.style_idx,
                        p.style,
                        p.rung,
                        p.k,
                        p.tol,
                        if same_family { "same" } else { "cross" },
                    ));
                }
            }
        }
        std::fs::write(
            format!("{dir}/stage25-consumer-filter-verdict-discordance.csv"),
            disc,
        )
        .expect("write discordance csv");

        std::fs::write(
            format!("{dir}/stage25-consumer-filter-census.md"),
            render_report(&pairs),
        )
        .expect("write census report");
    }

    fn render_report(pairs: &[Pair]) -> String {
        let n = pairs.len();
        let mut r = String::new();
        r.push_str("# Stage 2.5 — consumer-filter census\n\n");
        r.push_str(
            "> Generated by `strategy::stage25_census::tests::stage25_write_artifacts`\n\
             > (`cargo test -p lance-graph-planner --lib stage25_census -- --ignored`).\n\
             > Data: `stage25-consumer-filter-census.csv` (watcher-sample surface, one row per\n             > style x rung x k x channel — collapsed over `tol`, losslessly, per the pinned\n             > invariant that the sample cannot depend on it) and\n             > `stage25-consumer-filter-verdict-discordance.csv` (one row per verdict flip;\n             > **header-only is the result**).\n\n",
        );
        r.push_str(&format!(
            "**Design.** {} paired configurations = {} styles x {} rungs x {} budgets x {} \
             tolerances. Each pair holds everything fixed and toggles ONE thing: whether the \
             dissent channels' eligibility predicate carries the maturity clause. Both arms then \
             run the same shipped verdict body (`StyleStrategy::dissent_over`) — nothing here \
             reimplements the channel.\n\n",
            n,
            ThinkingStyle::ALL.len(),
            RUNGS.len(),
            KS.len(),
            TOLS.len()
        ));
        r.push_str(
            "**Statistics.** `jc::stats` / `jc::reliability`. The census is exhaustive and \
             deterministic, so no inferential test is run over it — there is no sampling \
             distribution for a p-value to describe. What is reported is cross-tabulation \
             (`binary_association`, carrying kappa and the marginals), variance decomposition \
             (`eta_squared`, `multiple_r_squared`), rank association (`spearman`), and ONE \
             genuinely inferential quantity: the exact one-sided binomial bound on an unseen \
             flip rate.\n\n",
        );

        // ── A. watcher-sample effect ──
        r.push_str("## A. Watcher-sample effect\n\n");
        r.push_str("| channel | pairs changed | mean Jaccard dist | mean retention | mean |Δcount| | max sym-diff |\n");
        r.push_str("|---|---|---|---|---|---|\n");
        for same_family in [true, false] {
            let cs = channel_stats(pairs, same_family);
            let changed = cs
                .deltas
                .iter()
                .filter(|d| d.symmetric_difference > 0)
                .count();
            let mean_abs_dcount = mean(
                &cs.deltas
                    .iter()
                    .map(|d| (d.n_on as f64 - d.n_off as f64).abs())
                    .collect::<Vec<_>>(),
            );
            let max_sd = cs
                .deltas
                .iter()
                .map(|d| d.symmetric_difference)
                .max()
                .unwrap_or(0);
            r.push_str(&format!(
                "| {} | {changed} / {n} ({:.1}%) | {:.4} | {:.4} | {:.3} | {max_sd} |\n",
                cs.label,
                100.0 * changed as f64 / n as f64,
                mean(&cs.jaccard_distance),
                mean(
                    &cs.deltas
                        .iter()
                        .map(SetDelta::retention)
                        .collect::<Vec<_>>()
                ),
                mean_abs_dcount,
            ));
        }

        // ── D. stratification ──
        r.push_str(
            "\n## D. Stratification of the watcher-sample effect (descriptive eta-squared)\n\n",
        );
        r.push_str(
            "Proportion of the Jaccard-distance variance attributable to each factor, over the \
             exhaustive design. **Descriptive, not inferential** — a variance decomposition of a \
             census, never a test of a hypothesis about a population.\n\n",
        );
        r.push_str("| channel | style | rung | k | tol | joint R² (rung,k,tol,style) |\n");
        r.push_str("|---|---|---|---|---|---|\n");
        for same_family in [true, false] {
            let cs = channel_stats(pairs, same_family);
            let y = &cs.jaccard_distance;
            let e = |keys: Vec<usize>| -> String {
                eta_squared(&group_by(&keys, y)).map_or("n/a".into(), |v| format!("{v:.4}"))
            };
            let style_k: Vec<usize> = pairs.iter().map(|p| p.style_idx).collect();
            let rung_k: Vec<usize> = pairs.iter().map(|p| p.rung as usize).collect();
            let k_k: Vec<usize> = pairs
                .iter()
                .map(|p| KS.iter().position(|&x| x == p.k).unwrap())
                .collect();
            let tol_k: Vec<usize> = pairs
                .iter()
                .map(|p| TOLS.iter().position(|&x| x == p.tol).unwrap())
                .collect();
            let preds: Vec<Vec<f64>> = vec![
                rung_k.iter().map(|&v| v as f64).collect(),
                k_k.iter().map(|&v| v as f64).collect(),
                tol_k.iter().map(|&v| v as f64).collect(),
                style_k.iter().map(|&v| v as f64).collect(),
            ];
            let joint = multiple_r_squared(y, &preds).map_or("n/a".into(), |v| format!("{v:.4}"));
            r.push_str(&format!(
                "| {} | {} | {} | {} | {} | {joint} |\n",
                cs.label,
                e(style_k),
                e(rung_k),
                e(k_k),
                e(tol_k)
            ));
        }
        r.push_str(
            "\n`tol` is **exactly** 0.0000 in both rows, and that is structural rather than a \
             finding about tolerance: the sampled watcher set depends only on `(rung, want, k)`, \
             so the tolerance cannot move it. It is reported because a NON-zero value there would \
             mean the census had accidentally coupled the two, and `stage25_tolerance_cannot_\
             move_the_watcher_sample` pins it.\n\n\
             The joint `R²` column is **not comparable to the eta-squared columns** and is \
             routinely SMALLER than the largest of them. `multiple_r_squared` fits the factors as \
             LINEAR predictors over their integer codes, while eta-squared groups them \
             nominally; a factor whose effect is non-monotone in its code (style, most \
             obviously — the code is an enum position, not a quantity) is largely invisible to \
             the linear fit. Read the joint figure as \"how much a linear read of the design \
             explains\", never as a ceiling on the design's total effect.\n\n\
             Strongest and weakest strata by mean Jaccard distance:\n\n",
        );
        for same_family in [true, false] {
            let cs = channel_stats(pairs, same_family);
            let mut by_cell: Vec<((RungLevel, usize), Vec<f64>)> = Vec::new();
            for (p, d) in pairs.iter().zip(&cs.jaccard_distance) {
                let key = (p.rung, p.k);
                match by_cell.iter_mut().find(|(kk, _)| *kk == key) {
                    Some((_, v)) => v.push(*d),
                    None => by_cell.push((key, vec![*d])),
                }
            }
            let mut rows: Vec<((RungLevel, usize), f64)> =
                by_cell.into_iter().map(|(k, v)| (k, mean(&v))).collect();
            rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top = &rows[..rows.len().min(3)];
            let bot = &rows[rows.len().saturating_sub(3)..];
            r.push_str(&format!("- **{}**\n", cs.label));
            for (label, set) in [("strongest", top), ("weakest", bot)] {
                let listed: Vec<String> = set
                    .iter()
                    .map(|((rg, k), v)| format!("{rg:?}/k={k}: {v:.4}"))
                    .collect();
                r.push_str(&format!("  - {label}: {}\n", listed.join(" · ")));
            }
        }
        r.push_str("\nRank association between the budget `k` and the Jaccard distance:\n\n");
        for same_family in [true, false] {
            let cs = channel_stats(pairs, same_family);
            let kf: Vec<f64> = pairs.iter().map(|p| p.k as f64).collect();
            let rho =
                spearman(&kf, &cs.jaccard_distance).map_or("n/a".into(), |v| format!("{v:+.4}"));
            r.push_str(&format!("- {}: Spearman rho = {rho}\n", cs.label));
        }

        // ── B. verdict effect ──
        r.push_str("\n## B. Verdict effect\n\n");
        for same_family in [true, false] {
            let cs = channel_stats(pairs, same_family);
            let t = binary_association(&cs.fired_off, &cs.fired_on).expect("columns");
            let discordant = cs
                .label_off
                .iter()
                .zip(&cs.label_on)
                .filter(|(a, b)| a != b)
                .count();
            r.push_str(&format!("### {}\n\n", cs.label));
            r.push_str("OFF → ON transition matrix (fired / did not fire):\n\n");
            r.push_str("| | ON: silent | ON: fired |\n|---|---|---|\n");
            r.push_str(&format!("| **OFF: silent** | {} | {} |\n", t.n00, t.n01));
            r.push_str(&format!("| **OFF: fired** | {} | {} |\n\n", t.n10, t.n11));
            r.push_str(&format!(
                "- concordant {} / {n}, discordant **{}** / {n}\n",
                n - discordant,
                discordant
            ));
            r.push_str(&format!(
                "- observed verdict-change rate: **{:.6}**\n",
                discordant as f64 / n as f64
            ));
            r.push_str(&format!(
                "- Cohen's kappa = {} (defined: both categories occur, n11={} n00={})\n",
                t.kappa.map_or("undefined".into(), |k| format!("{k:.6}")),
                t.n11,
                t.n00
            ));
            r.push_str(
                "- **McNemar is NOT reported**: it is a test on the discordant cells and there \
                 are none, so the statistic is degenerate rather than significant. Reporting it \
                 would be manufacturing a result out of an empty table.\n\n",
            );
        }

        // ── the bound ──
        r.push_str("### Exact bound on an unseen flip rate\n\n");
        r.push_str(
            "One-sided Clopper-Pearson upper limit at zero observed events, `1 - alpha^(1/n)`, \
             alpha = 0.05. **The 5,760 rows are repeated measures, not independent trials** — one \
             style contributes 160 of them — so the bound is a LADDER over clustering \
             assumptions, and the honest number to quote is the one whose independence \
             assumption you are willing to defend.\n\n",
        );
        r.push_str("| independent unit | n | upper bound on flip rate |\n|---|---|---|\n");
        // `StyleCluster` is not `Ord`, so count distinct by linear scan rather
        // than reaching for a set — 36 styles, and a derive added for a report
        // would be production surface bent to an instrument.
        let mut seen = Vec::new();
        for st in ThinkingStyle::ALL {
            let c = st.cluster();
            if !seen.contains(&c) {
                seen.push(c);
            }
        }
        let n_clusters = seen.len();
        for (label, m) in [
            (
                "configuration cell (assumes full independence — optimistic)",
                n,
            ),
            ("style x rung cell", ThinkingStyle::ALL.len() * RUNGS.len()),
            ("style", ThinkingStyle::ALL.len()),
            ("style cluster (most conservative)", n_clusters),
        ] {
            r.push_str(&format!(
                "| {label} | {m} | {:.3e} |\n",
                zero_event_upper_bound(m, 0.05)
            ));
        }

        // ── C. latent effect ──
        r.push_str("\n## C. Pre-verdict numeric\n\n");
        r.push_str(
            "The margin the channel actually thresholds — `|tc.confidence - admitted|` against \
             `tol` — is local to `dissent_over` and is not returned. **It was not exposed for \
             this measurement.** What IS naturally available is finer than fired/not-fired and is \
             measured above: the elevation `RungLevel` (which watcher objected) and, on the \
             cross-family channel, the reported `Mechanism`. Both agree exactly, so the zero \
             discordance is not a coarse-label artifact — it survives the finest outcome the \
             harness exposes without adding one.\n\n",
        );

        // ── headline ──
        //
        // Reported PER CHANNEL, not pooled. Pooling put the mean at 0.1932 and
        // produced the single word "weakly" for two channels that differ by a
        // factor of two — a threshold artifact, and exactly the kind of
        // smoothing this census exists to avoid. The rule is stated so the word
        // is checkable rather than editorial: >= 0.20 mean Jaccard distance
        // over the design, or >= a third of configurations changed, is
        // "materially"; any non-zero effect below that is "weakly".
        r.push_str("## Headline\n\n");
        for same_family in [true, false] {
            let cs = channel_stats(pairs, same_family);
            let changed = cs
                .deltas
                .iter()
                .filter(|d| d.symmetric_difference > 0)
                .count();
            let jd = mean(&cs.jaccard_distance);
            let frac = changed as f64 / n as f64;
            let strength = if jd >= 0.20 || frac >= 1.0 / 3.0 {
                "materially"
            } else if jd > 0.0 {
                "weakly"
            } else {
                "not at all"
            };
            r.push_str(&format!(
                "- **{}: consumer filtering {strength} changes watcher sampling** — \
                 {changed}/{n} configurations differ ({:.1}%), mean Jaccard distance {jd:.4}, \
                 mean retention {:.4}. **Verdict change: 0/{n}.**\n",
                cs.label,
                100.0 * frac,
                mean(
                    &cs.deltas
                        .iter()
                        .map(SetDelta::retention)
                        .collect::<Vec<_>>()
                ),
            ));
        }
        r.push('\n');
        r.push_str(
            "Scope, stated so it is not over-read: this is a statement about the measured \
             Stage-2 surface only. It is NOT \"the filter has no effect\" (it demonstrably \
             changes watcher membership) and it is NOT \"the filter is semantically irrelevant \
             forever\" (a different tolerance regime, a different admitted set, or a carve that \
             changes which kernels are production all move the input to this measurement).\n",
        );
        r
    }
}
