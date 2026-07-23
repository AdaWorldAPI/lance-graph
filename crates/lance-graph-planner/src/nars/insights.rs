//! `insights` — D-SCI-INSIGHT: surface the ranked main insights of a reasoned
//! concept-KG, the no-LLM insight-extraction surface. Each [`MainInsight`]
//! carries its provenance ladder (rung + premises — the derivation chain, a
//! learnable signal) and an explained [`InsightReason`] (why it is
//! insightful, never a black-box score). This is the insight-SURFACING half
//! of "extract the main insights of a paper" — D-SCI-1 feeds it real paper
//! concepts as a reasoned [`BeliefArena`]; this module composes the three
//! shipped signals (epiphany attractors, high-expectation derivations, shared
//! non-hub middle terms) into one ranked output.
//!
//! Cross-ref: [`super::epiphany`] (CoreTheme reuses `rank_epiphany_attractors`
//! directly — no re-derivation of density), [`super::insight`] (the S10
//! per-step insight notion this module is NOT — this is basin/derivation/
//! predicate-level, not per-step), plan D-SCI.

use super::belief::{BeliefArena, CStmt, Copula};
use super::epiphany::rank_epiphany_attractors;
use std::collections::{BTreeMap, BTreeSet};

/// Which kind of main insight this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsightKind {
    /// A concept the KG reasons DENSELY about (a top epiphany attractor).
    CoreTheme,
    /// A strong DERIVED claim (high-expectation inference).
    Conclusion,
    /// A concept bridging otherwise-separate claims (a shared middle term).
    Bridge,
}

/// WHY an insight is insightful — the explained (auditable) reason, never a
/// black-box score. This is what makes the extraction testable and learnable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InsightReason {
    /// Densest epiphany basin: `epiphanies` derived of `attempts` total at this concept.
    DenseBasin { epiphanies: usize, attempts: usize },
    /// A strong derivation: its NARS expectation.
    StrongDerivation { expectation: f32 },
    /// A middle term shared by `bridges` distinct subjects (connects them).
    MiddleTerm { bridges: usize },
}

/// One ranked main insight, carrying its provenance LADDER and explained reason.
#[derive(Debug, Clone, PartialEq)]
pub struct MainInsight {
    /// What kind of insight.
    pub kind: InsightKind,
    /// The focal statement (for `Conclusion`) or a marker `is_a` on the focal
    /// concept (for `CoreTheme`/`Bridge`, the concept sits in `focus.s`).
    pub focus: CStmt,
    /// The ranking strength (rate-/expectation-normalized; comparable within a kind).
    pub strength: f32,
    /// Ladder depth: 0 = observed, N = derived through N inference steps.
    pub rung: u32,
    /// The provenance ladder: arena indices of the beliefs that composed to
    /// produce this (empty for an observed theme). The learnable derivation shape.
    pub premises: Vec<u32>,
    /// The explained reason this is insightful.
    pub reason: InsightReason,
}

/// Configuration for main-insight extraction.
#[derive(Debug, Clone, Copy)]
pub struct InsightConfig {
    /// Min beliefs at a subject for it to be a CoreTheme candidate.
    pub min_theme_attempts: usize,
    /// Min distinct subjects sharing a predicate for it to be a Bridge.
    pub min_bridge: usize,
    /// A predicate with in-degree ABOVE this is a hub (too generic) — not a Bridge.
    pub hub_ceiling: usize,
    /// How many insights to return (top-k by strength across kinds).
    pub top_k: usize,
}

impl Default for InsightConfig {
    fn default() -> Self {
        Self {
            min_theme_attempts: 2,
            min_bridge: 2,
            hub_ceiling: 32,
            top_k: 16,
        }
    }
}

/// Surface the ranked main insights of a reasoned concept-KG. Composes three
/// shipped signals — epiphany attractors (CoreTheme), high-expectation
/// derivations (Conclusion), shared non-hub middle terms (Bridge) — each
/// insight carrying its ladder (rung + premises) and explained reason. Returns
/// up to `cfg.top_k`, sorted by `strength` DESC (ties → `focus.s` ASC).
#[must_use]
pub fn extract_main_insights(arena: &BeliefArena, cfg: &InsightConfig) -> Vec<MainInsight> {
    let mut out: Vec<MainInsight> = Vec::new();

    // 1. CoreTheme — top epiphany attractors (reuse the shipped ranker).
    for a in rank_epiphany_attractors(arena, cfg.min_theme_attempts) {
        if a.epiphanies == 0 {
            continue; // a basin with no derived belief is not (yet) a theme
        }
        out.push(MainInsight {
            kind: InsightKind::CoreTheme,
            focus: inh(a.subject, a.subject), // self-marker: the concept is focus.s
            strength: a.rate,
            rung: 0,
            premises: Vec::new(),
            reason: InsightReason::DenseBasin {
                epiphanies: a.epiphanies,
                attempts: a.attempts,
            },
        });
    }

    // 2. Conclusion — derived beliefs ranked by expectation, carrying their ladder.
    //    Only surface derivations with a REAL provenance ladder: a rung>=1 belief
    //    admitted with empty premises (e.g. an `elevate_field` mint, or any
    //    `admit_derived(.., &[], ..)`) has no auditable chain, and this API's
    //    promise is auditability — an unexplainable "conclusion" is worse than
    //    none. Empty-premise derivations are skipped here (Codex P2, E-SCI-INSIGHT).
    for b in arena.entries() {
        // Skip self-conclusions (`s == p`, minted by `close_transitive` on cyclic
        // KGs — real text is full of cycles): `river → river` is trivially true
        // and carries no insight, only noise in the ranked output (D-SCI-1 real-
        // text finding; same spirit as the basin self-loop dedup).
        if b.rung >= 1 && !b.premises.is_empty() && b.stmt.s != b.stmt.p {
            out.push(MainInsight {
                kind: InsightKind::Conclusion,
                focus: b.stmt,
                strength: b.truth.expectation(),
                rung: b.rung,
                premises: b.premises.clone(),
                reason: InsightReason::StrongDerivation {
                    expectation: b.truth.expectation(),
                },
            });
        }
    }

    // 3. Bridge — predicates shared by ≥min_bridge distinct subjects, below the hub ceiling.
    let mut indeg: BTreeMap<u16, BTreeSet<u16>> = BTreeMap::new();
    let mut subjects_all: BTreeSet<u16> = BTreeSet::new();
    for b in arena.entries() {
        if b.stmt.cop == Copula::Inh {
            indeg.entry(b.stmt.p).or_default().insert(b.stmt.s);
            subjects_all.insert(b.stmt.s);
        }
    }
    // Normalize bridge strength by the TOTAL distinct subjects, not the predicate
    // count: `d` (subjects sharing a predicate) is a subset of all subjects, so
    // `d / total_subjects` is a fraction in [0,1] — dimensionally a subject-share,
    // on the SAME scale as a Conclusion's expectation. Dividing by the predicate
    // count (the prior bug) is unbounded — a predicate shared by 2 subjects in a
    // 1-predicate KG scored 2.0 and out-ranked a max-confidence conclusion in the
    // cross-kind top_k sort (Codex P2, E-SCI-INSIGHT).
    let total_subjects = subjects_all.len().max(1) as f32;
    for (m, subjects) in &indeg {
        let d = subjects.len();
        if d >= cfg.min_bridge && d <= cfg.hub_ceiling {
            out.push(MainInsight {
                kind: InsightKind::Bridge,
                focus: inh(*m, *m), // the bridging concept sits in focus.s
                strength: d as f32 / total_subjects, // subject-share, bounded [0,1]
                rung: 0,
                premises: Vec::new(),
                reason: InsightReason::MiddleTerm { bridges: d },
            });
        }
    }

    // Rank by strength DESC, ties by focus.s ASC (deterministic), take top_k.
    out.sort_by(|x, y| {
        y.strength
            .total_cmp(&x.strength)
            .then_with(|| x.focus.s.cmp(&y.focus.s))
    });
    out.truncate(cfg.top_k);
    out
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

    /// Builds the "paper" KG: a theme cluster (subjects 1,2,3 all `is_a` the
    /// shared bridge predicate 100) plus a chain `1→2→3→4` that
    /// `close_transitive` derives into `1→3`, `1→4`, `2→4`.
    fn structured_kg() -> BeliefArena {
        let mut a = BeliefArena::new();
        a.observe(inh(1, 100), TruthValue::new(0.9, 0.9), Stamp::source(0));
        a.observe(inh(2, 100), TruthValue::new(0.9, 0.9), Stamp::source(1));
        a.observe(inh(3, 100), TruthValue::new(0.9, 0.9), Stamp::source(2));
        a.observe(inh(1, 2), TruthValue::new(0.9, 0.9), Stamp::source(3));
        a.observe(inh(2, 3), TruthValue::new(0.9, 0.9), Stamp::source(4));
        a.observe(inh(3, 4), TruthValue::new(0.9, 0.9), Stamp::source(5));
        a.close_transitive(64);
        a
    }

    #[test]
    fn structured_kg_surfaces_its_insights() {
        let a = structured_kg();
        let insights = extract_main_insights(&a, &InsightConfig::default());

        assert!(!insights.is_empty());

        let bridge = insights
            .iter()
            .find(|i| i.kind == InsightKind::Bridge && i.focus.s == 100)
            .expect("predicate 100 shared by subjects 1,2,3 must surface as a Bridge");
        assert_eq!(bridge.reason, InsightReason::MiddleTerm { bridges: 3 });

        let conclusion = insights
            .iter()
            .find(|i| i.kind == InsightKind::Conclusion && i.rung >= 1 && !i.premises.is_empty())
            .expect("a derived statement (e.g. 1→3 or 1→4) must surface as a Conclusion with a recovered ladder");
        assert!(conclusion.rung >= 1);
        assert!(!conclusion.premises.is_empty());

        assert!(
            insights.iter().any(|i| i.kind == InsightKind::CoreTheme),
            "a dense basin must surface as a CoreTheme"
        );
    }

    /// The size-matched falsifier: a KG with the SAME number of observed
    /// beliefs (6) but fully disjoint (no shared predicate, no chain) must
    /// stay quiet — no Bridge, no Conclusion — and must produce strictly
    /// fewer insights than the structured KG (E-BASIN-WIDTH).
    #[test]
    fn noise_kg_stays_quiet() {
        let mut noise = BeliefArena::new();
        for (s, p) in [
            (10u16, 110u16),
            (11, 111),
            (12, 112),
            (13, 113),
            (14, 114),
            (15, 115),
        ] {
            noise.observe(
                inh(s, p),
                TruthValue::new(0.9, 0.9),
                Stamp::source(s as u32),
            );
        }
        noise.close_transitive(64);

        let noise_insights = extract_main_insights(&noise, &InsightConfig::default());

        assert!(
            !noise_insights.iter().any(|i| i.kind == InsightKind::Bridge),
            "no predicate is shared by >=2 subjects in the disjoint KG"
        );
        assert!(
            !noise_insights
                .iter()
                .any(|i| i.kind == InsightKind::Conclusion),
            "nothing is derived in the disjoint KG"
        );

        let structured = structured_kg();
        let structured_insights = extract_main_insights(&structured, &InsightConfig::default());

        assert!(
            structured_insights.len() > noise_insights.len(),
            "the extractor must surface structure and stay quiet on noise (E-BASIN-WIDTH): \
             structured={} noise={}",
            structured_insights.len(),
            noise_insights.len()
        );
    }

    #[test]
    fn every_conclusion_carries_its_ladder() {
        let a = structured_kg();
        let insights = extract_main_insights(&a, &InsightConfig::default());

        for i in insights
            .iter()
            .filter(|i| i.kind == InsightKind::Conclusion)
        {
            assert!(i.rung >= 1, "every surfaced conclusion must be derived");
            assert!(
                !i.premises.is_empty(),
                "every surfaced conclusion must carry a non-empty provenance ladder"
            );
        }
    }

    /// Codex P2 (#832): a derived belief admitted with EMPTY premises (e.g. an
    /// `elevate_field` mint, or any `admit_derived(.., &[], ..)`) has no auditable
    /// ladder and must NOT surface as a `Conclusion` — this API promises every
    /// conclusion carries its provenance chain.
    #[test]
    fn empty_premise_derivation_is_not_surfaced_as_conclusion() {
        let mut a = BeliefArena::new();
        // A synthetic parent edge with high truth but NO premises (the elevate mint shape).
        a.admit_derived(inh(7, 8), TruthValue::new(0.95, 0.95), &[], 1);
        let insights = extract_main_insights(&a, &InsightConfig::default());
        assert!(
            !insights
                .iter()
                .any(|i| i.kind == InsightKind::Conclusion && i.focus == inh(7, 8)),
            "an empty-premise derivation must not be surfaced as an auditable Conclusion"
        );
    }

    /// Codex P2 (#832): bridge strength must stay in [0,1] (a subject-share), on
    /// the same scale as a Conclusion's expectation — so a bridge in a
    /// few-predicate KG cannot out-rank a max-confidence conclusion in the
    /// cross-kind `top_k` sort.
    #[test]
    fn bridge_strength_is_bounded_and_comparable() {
        // 2 subjects share predicate 100; only 1 predicate in the KG — the prior
        // `d/total_preds` bug would have scored this bridge 2.0.
        let mut a = BeliefArena::new();
        a.observe(inh(1, 100), TruthValue::new(0.9, 0.9), Stamp::source(0));
        a.observe(inh(2, 100), TruthValue::new(0.9, 0.9), Stamp::source(1));
        // A derived conclusion with a real ladder (so a Conclusion exists to compare).
        a.observe(inh(1, 2), TruthValue::new(0.9, 0.9), Stamp::source(2));
        a.observe(inh(2, 3), TruthValue::new(0.9, 0.9), Stamp::source(3));
        a.close_transitive(64);

        let insights = extract_main_insights(&a, &InsightConfig::default());
        for b in insights.iter().filter(|i| i.kind == InsightKind::Bridge) {
            assert!(
                (0.0..=1.0).contains(&b.strength),
                "bridge strength must be bounded [0,1], got {}",
                b.strength
            );
        }
    }

    /// D-SCI-1 real-text finding: a self-conclusion (`s == p`, minted by
    /// `close_transitive` closing a cycle) must NOT surface as an insight — it is
    /// trivially true and pure noise. Real prose is cyclic, so this fires often.
    #[test]
    fn self_conclusion_is_not_surfaced() {
        // A 2-cycle 1→2, 2→1 closes into self-loops 1→1 and 2→2.
        let mut a = BeliefArena::new();
        a.observe(inh(1, 2), TruthValue::new(0.9, 0.9), Stamp::source(0));
        a.observe(inh(2, 1), TruthValue::new(0.9, 0.9), Stamp::source(1));
        a.close_transitive(64);
        assert!(
            a.get(inh(1, 1)).is_some(),
            "the cycle must mint a self-loop"
        );

        let insights = extract_main_insights(&a, &InsightConfig::default());
        assert!(
            !insights
                .iter()
                .any(|i| i.kind == InsightKind::Conclusion && i.focus.s == i.focus.p),
            "a self-conclusion (river→river) must never surface as an insight"
        );
    }

    #[test]
    fn top_k_bounds_output() {
        let a = structured_kg();
        let cfg = InsightConfig {
            top_k: 1,
            ..InsightConfig::default()
        };
        let insights = extract_main_insights(&a, &cfg);
        assert!(insights.len() <= 1);
        if let Some(top) = insights.first() {
            let full = extract_main_insights(&a, &InsightConfig::default());
            assert_eq!(top.strength, full[0].strength);
        }
    }
}
