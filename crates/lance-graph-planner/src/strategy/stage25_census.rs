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
                p.v_cross_off.map(cross_label),
                p.v_cross_on.map(cross_label),
            )
        };
        cs.fired_off.push(vo.is_some());
        cs.fired_on.push(vn.is_some());
        cs.label_off.push(vo.unwrap_or(0));
        cs.label_on.push(vn.unwrap_or(0));
    }
    cs
}

/// Encode a cross-family verdict as ONE nominal label carrying **both** tuple
/// components.
///
/// The production verdict is `(RungLevel, Mechanism)`, and reducing it to the
/// rung would call a pair concordant whenever the filter swapped the first
/// objector for one of a DIFFERENT mechanism at the SAME `min_rung` — silently
/// supporting the report's claim that the reported mechanism agrees, using a
/// measurement that never looked at it. (Codex review, PR #971.)
///
/// `0` stays reserved for "no dissent", so the encoding starts at 1.
fn cross_label((rung, mech): (RungLevel, Mechanism)) -> usize {
    const N_MECH: usize = 4; // ParallelIndependence | TruthAware | StructuralDivergence | Infrastructure
    let m = match mech {
        Mechanism::ParallelIndependence => 0,
        Mechanism::TruthAwareInference => 1,
        Mechanism::StructuralDivergence => 2,
        Mechanism::Infrastructure => 3,
    };
    1 + (rung as usize) * N_MECH + m
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

    /// **The headline, re-pinned — and the re-pin is the finding.**
    ///
    /// The first version of this test asserted `discordant == 0` on both
    /// channels, and it passed, because the predicate it was measuring was
    /// wrong: `watcher_can_dissent` filtered on `maturity().is_production()`,
    /// which admits 31 of 34 kernels while only **14** can move `delta_conf` —
    /// the only quantity either channel compares. The filter was excluding 3
    /// mute watchers and admitting 17 more (codex review, PR #971).
    ///
    /// With the honest predicate the answer inverts: the filter changes **1,098
    /// of 5,760** same-family verdicts and **384** cross-family. Pinned as
    /// EQUALITIES, not bounds — this is a deterministic census, so any movement
    /// is a real behavioural change that deserves a deliberate re-measure
    /// rather than a widened assertion.
    #[test]
    fn stage25_headline_watcher_delta_and_verdict_delta_are_both_real() {
        let pairs = census();
        for (same_family, expect_changed, expect_discordant) in
            [(true, 4080usize, 1098usize), (false, 4224, 384)]
        {
            let cs = channel_stats(&pairs, same_family);
            let changed = cs
                .deltas
                .iter()
                .filter(|d| d.symmetric_difference > 0)
                .count();
            let discordant = cs
                .label_off
                .iter()
                .zip(&cs.label_on)
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                changed, expect_changed,
                "{}: watcher-sample delta",
                cs.label
            );
            assert_eq!(
                discordant,
                expect_discordant,
                "{}: verdict delta over {} pairs",
                cs.label,
                pairs.len()
            );
        }
    }

    /// **The filter's effect has a DIRECTION, and on the same-family channel it
    /// is one-way.**
    ///
    /// `n10 == 0` there: no configuration that dissented with the filter OFF
    /// goes silent with it ON. That is the coverage argument turned into a
    /// measurement — replacing structurally-mute watchers with capable ones can
    /// only ADD objections, never remove them, because the removed watchers
    /// could not have raised one.
    ///
    /// The cross-family channel is *almost* one-way: 18 of 5,760 go the other
    /// way, and the reason is the SAMPLER, not the filter.
    /// `peripheral_sample_where` strides `k` picks over the eligible list, so
    /// shrinking that list changes WHICH capable watchers are picked — a
    /// capable dissenter selected under OFF can fall off the stride under ON.
    /// Pinned exactly so it stays visible instead of being rounded to "one-way".
    #[test]
    fn stage25_the_filter_adds_dissent_and_on_one_channel_never_removes_it() {
        let pairs = census();
        let same = binary_association(
            &channel_stats(&pairs, true).fired_off,
            &channel_stats(&pairs, true).fired_on,
        )
        .expect("columns");
        assert_eq!(
            same.n10, 0,
            "same-family: {} configurations LOST a dissent — the filter is \
             supposed to be unable to remove one",
            same.n10
        );
        assert_eq!(same.n01, 1098, "same-family: dissents gained");

        let cross = binary_association(
            &channel_stats(&pairs, false).fired_off,
            &channel_stats(&pairs, false).fired_on,
        )
        .expect("columns");
        assert_eq!(cross.n01, 366, "cross-family: dissents gained");
        assert_eq!(
            cross.n10, 18,
            "cross-family: dissents lost to the sampler stride"
        );
        assert!(
            cross.n01 > cross.n10,
            "the net direction must still be toward MORE dissent"
        );
    }

    /// Agreement is now **partial and measured**, not perfect — and κ is
    /// defined in every case, so the numbers are real rather than the
    /// degenerate constant-column case `jc::stats::cohen_kappa` refuses to
    /// score.
    ///
    /// Re-pinned from `κ == 1.0` when the watcher predicate was corrected. The
    /// old assertion was true of a measurement of the wrong thing.
    #[test]
    fn stage25_agreement_is_partial_and_kappa_is_defined() {
        let pairs = census();
        for (same_family, lo, hi) in [(true, 0.60, 0.75), (false, 0.78, 0.92)] {
            let cs = channel_stats(&pairs, same_family);
            let t = binary_association(&cs.fired_off, &cs.fired_on).expect("columns");
            assert!(
                t.n11 > 0 && t.n00 > 0,
                "{}: both outcomes must occur or agreement is vacuous",
                cs.label
            );
            let k = t
                .kappa
                .expect("kappa is defined when both categories occur");
            assert!(
                (lo..=hi).contains(&k),
                "{}: kappa {k:.4} outside the pinned window [{lo}, {hi}]",
                cs.label
            );
            assert!(
                k < 1.0,
                "{}: kappa 1.0 would mean the filter changed nothing",
                cs.label
            );

            // The FINE label (elevation target, and for cross-family the
            // Mechanism too) must also be scoreable — a `None` here would mean
            // one column collapsed to a single category and the discordance
            // counts above were measured against a constant.
            let k_fine = cohen_kappa(&cs.label_off, &cs.label_on)
                .expect("kappa defined on the fine label too");
            assert!(k_fine < 1.0, "{}: fine kappa = {k_fine}", cs.label);
        }
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
                    // Same `cross_label` the statistics use — the artifact and
                    // the numbers must not disagree about what "changed" means.
                    (
                        p.v_cross_off.map(cross_label).unwrap_or(0),
                        p.v_cross_on.map(cross_label).unwrap_or(0),
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
             > Data: `stage25-consumer-filter-census.csv` (watcher-sample surface, one row\n\
             > per style x rung x k x channel — collapsed over `tol`, losslessly, per the\n\
             > pinned invariant that the sample cannot depend on it) and\n\
             > `stage25-consumer-filter-verdict-discordance.csv` (one row per verdict flip).\n\n",
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
        r.push_str(
            "| channel | pairs changed | mean Jaccard dist | mean retention | \
             mean \\|Δcount\\| | max sym-diff |\n",
        );
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
            r.push_str(&format!(
                "- fired {} -> {} on this channel\n",
                t.n10 + t.n11,
                t.n01 + t.n11
            ));
            r.push_str(&format!(
                "- **direction**: {} silence -> dissent, {} dissent -> silence\n",
                t.n01, t.n10
            ));
            r.push_str(
                "- **McNemar is NOT computed.** `jc` carries no implementation, and hand-rolling \
                 a test statistic into an exhaustive deterministic census — which has no \
                 sampling distribution for its p-value to describe — is the failure this report \
                 exists to avoid. The transition matrix above carries the same information \
                 without the borrowed authority.\n\n",
            );
        }

        // ── the bound ──
        r.push_str("### Exact bound on the one surface that IS still zero\n\n");
        r.push_str(
            "The verdict surface has events now, so a zero-event bound no longer applies to it. \
             What remains exactly zero is the DIRECTION on the same-family channel: **no \
             configuration lost a dissent** (`n10 = 0` of 5,760). That is the coverage claim in \
             its sharpest form — a filter that removes only structurally-mute watchers should be \
             incapable of removing an objection — so that is the quantity worth bounding.\n\n\
             One-sided Clopper-Pearson upper limit at zero observed events, `1 - alpha^(1/n)`, \
             with **alpha = 0.05** (a 95 % one-sided confidence level). **The 5,760 rows are \
             repeated measures, not independent trials** — one style contributes 160 of them — so \
             the bound is a LADDER over clustering assumptions and each row names the unit it \
             assumes; quote the one whose independence you are willing to defend.\n\n",
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
                "one (style, rung, k, tol) cell — every cell its own trial; OPTIMISTIC",
                n,
            ),
            (
                "one (style, rung) cell — k and tol treated as within-cell repeats",
                ThinkingStyle::ALL.len() * RUNGS.len(),
            ),
            (
                "one style — a style's whole 160-cell block as one observation",
                ThinkingStyle::ALL.len(),
            ),
            (
                "one style cluster — styles in a cluster share a Mechanism; most conservative",
                n_clusters,
            ),
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
             what the numbers above are computed over: the elevation `RungLevel` (which watcher \
             objected) and, on the cross-family channel, the reported `Mechanism` — both encoded \
             into ONE nominal label by `cross_label`, so a swap to a different mechanism at the \
             same rung counts as a change rather than passing as agreement.\n\n\
             That encoding is a correction, not a flourish. The first version of this census \
             reduced the cross-family verdict to its rung and would have reported such a swap as \
             concordant while the report claimed the mechanism agreed — a measurement that never \
             looked at the thing it certified.\n\n",
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
            // COMPUTED, never literal. The first version of this line wrote
            // `Verdict change: 0/{n}` as a hardcoded string — so the report
            // would have printed zero no matter what the census measured, and
            // did, for one revision after the predicate was corrected. A report
            // that states its conclusion instead of deriving it is not a
            // measurement; `stage25_headline_watcher_delta_and_verdict_delta_are_both_real`
            // is what caught it.
            let t = binary_association(&cs.fired_off, &cs.fired_on).expect("columns");
            let discordant = cs
                .label_off
                .iter()
                .zip(&cs.label_on)
                .filter(|(a, b)| a != b)
                .count();
            r.push_str(&format!(
                "- **{}: consumer filtering {strength} changes watcher sampling** — \
                 {changed}/{n} configurations differ ({:.1}%), mean Jaccard distance {jd:.4}, \
                 mean retention {:.4}. **Verdict change: {discordant}/{n}**, direction \
                 {} silence->dissent vs {} dissent->silence.\n",
                cs.label,
                100.0 * frac,
                mean(
                    &cs.deltas
                        .iter()
                        .map(SetDelta::retention)
                        .collect::<Vec<_>>()
                ),
                t.n01,
                t.n10,
            ));
        }
        r.push('\n');
        r.push_str(
            "\n**This headline replaces an earlier one that read `0/5,760` on both channels.** \
             That number was real but measured the wrong predicate: the first consumer filter \
             tested `maturity().is_production()`, which admits 31 of 34 kernels, while only 14 \
             can move `delta_conf` — the only quantity either channel compares. The filter was \
             removing 3 mute watchers and admitting 17 more, so it changed the sample and could \
             not change the answer. Corrected after codex review on PR #971.\n\n\
             Scope, stated so it is not over-read: this is a statement about the measured \
             Stage-2 surface only, under one fixed `PlanContext`. It says nothing about whether \
             the ADDED dissent is correct — only that the periphery now contains instruments \
             that can object, where before a majority of the sampled watchers structurally could \
             not.\n",
        );
        r
    }
}
