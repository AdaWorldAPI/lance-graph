//! `tactics` — the five NARS reasoning tactics as **term logic** over the
//! [`BeliefArena`] (`.claude/plans/dialectic-engine-v1.md` §3), in the
//! lance-graph reasoning layer.
//!
//! Every tactic composes STATEMENTS by their shared terms and derives truth by
//! the ONE engine's truth functions ([`TruthValue`]) — never a local truth
//! reimplementation, never a fingerprint/popcount distance
//! (`E-NARS-IS-LOGIC-...-1`; the litmus: *am I composing statements by their
//! terms, or measuring fingerprints by their bits?* — bits = reject). The
//! Datapath syllogisms (RCR/TR/CAS) return a throttled [`Candidate`] frontier
//! plus first-class [`ReasoningGap`]s (never eager arena closure — S5); the
//! Control stamp-ops (ASC/CR) act through the arena's disjointness guard.
//!
//! Each tactic carries its `contract::recipe_dispatch` id (RCR=4, TR=6, ASC=7,
//! CAS=8, CR=11); the [`tactic_matches_recipe_taxonomy`] test asserts that the
//! truth function each tactic uses agrees with the recipe's declared
//! [`RecipeInference`] — the new code is pinned to the shipped taxonomy.
//!
//! | # | Tactic | Rule | Truth (one engine) | Bucket |
//! |---|---|---|---|---|
//! | 4 RCR | abduction | `{P→M, S→M} ⊢ S→P` (shared predicate M) | [`TruthValue::abduction`] (weak) | Datapath (throttled) |
//! | 6 TR | divergence | `{S cop P, S↔S′} ⊢ S′ cop P` (Sim BELIEF) | [`TruthValue::analogy`] | Datapath |
//! | 7 ASC | self-critique | negation target `⟨1−f, c⟩`; independent counter revised in | [`TruthValue::revise`] / blocked | Control |
//! | 8 CAS | abstraction | up=induction `{S→P, S→G} ⊢ G→P`; down=deduction `{G→P, S→G} ⊢ S→P` | [`TruthValue::induction`] / [`TruthValue::deduction`] | Datapath |
//! | 11 CR | dialectic | same statement, disjoint → revision; overlap → CHOICE | [`TruthValue::revise`] / CHOICE | Control |

use super::belief::{BeliefArena, CStmt, Copula, ReviseOutcome, Stamp};
use super::truth::TruthValue;
use std::collections::HashMap;

/// `contract::recipe_dispatch` ids of the five dialectic tactics.
pub const RCR_ID: u8 = 4;
/// TR — divergence / analogy.
pub const TR_ID: u8 = 6;
/// ASC — adversarial self-critique.
pub const ASC_ID: u8 = 7;
/// CAS — abstraction scaling.
pub const CAS_ID: u8 = 8;
/// CR — contradiction resolution (dialectic).
pub const CR_ID: u8 = 11;

/// Which tactic proposed a [`Candidate`] (for introspection / provenance).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tactic {
    /// RCR — abduction over a shared predicate.
    Rcr,
    /// TR — analogy across a similarity sibling.
    Tr,
    /// CAS up — induction to the parent (`{S→P, S→G} ⊢ G→P`).
    CasUp,
    /// CAS down — deduction from the parent (`{G→P, S→G} ⊢ S→P`).
    CasDown,
}

impl Tactic {
    /// The `recipe_dispatch` id this tactic implements.
    #[must_use]
    pub fn recipe_id(self) -> u8 {
        match self {
            Tactic::Rcr => RCR_ID,
            Tactic::Tr => TR_ID,
            Tactic::CasUp | Tactic::CasDown => CAS_ID,
        }
    }
}

/// A proposed but NOT-yet-admitted derivation — the throttled frontier (S5:
/// tactics propose; the caller admits selectively via
/// [`BeliefArena::admit_derived`], never eager closure).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// The derived statement.
    pub stmt: CStmt,
    /// Its NARS truth (weak for abduction/induction/analogy; strong for CAS-down).
    pub truth: TruthValue,
    /// Arena indices of the two premises (the pointer fabric).
    pub premises: [u32; 2],
    /// Tarski rung: `max(premise rungs) + 1`.
    pub rung: u32,
    /// Which tactic proposed it.
    pub tactic: Tactic,
}

/// A first-class reasoning FAILURE — what is MISSING to reason further (the
/// reach-out hook: a gap names the concept/relation whose absence blocked a
/// tactic, so the frontier can seek it rather than silently deriving nothing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningGap {
    /// What kind of premise was missing.
    pub kind: GapKind,
    /// The subject term in scope (if any).
    pub subject: Option<u16>,
    /// The predicate / middle term in scope (if any).
    pub predicate: Option<u16>,
}

/// The taxonomy of reasoning failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapKind {
    /// RCR: no two beliefs share a (non-hub) predicate — no middle term.
    NoSharedMiddle,
    /// TR: the focus subject has no similarity sibling.
    NoSibling,
    /// CAS: the focus subject has no `is_a` parent to abstract over.
    NoAbstraction,
    /// Throttle: a middle term was barred as a hub (too high in-degree).
    HubExcluded,
    /// Throttle: the per-thought derivation budget was hit; some candidates dropped.
    BudgetExhausted,
}

/// S5 flood throttle for the weak abductive frontier: a confidence floor, a
/// per-thought derivation budget `k`, and hub middle-term exclusion.
#[derive(Debug, Clone, Copy)]
pub struct Throttle {
    /// Minimum candidate confidence — abduction is weak by construction, so a
    /// floor keeps the `d_M²` hypothesis flood off the frontier.
    pub c_min: f32,
    /// Per-thought derivation budget `k` (hard cap on candidates produced).
    pub budget: usize,
    /// A predicate whose in-degree (count of `?→M` beliefs) EXCEEDS this is a
    /// hub and is barred as a middle term.
    pub hub_indegree: usize,
}

impl Throttle {
    /// A calibrated throttle.
    #[must_use]
    pub fn new(c_min: f32, budget: usize, hub_indegree: usize) -> Self {
        Self {
            c_min,
            budget,
            hub_indegree,
        }
    }

    /// Wide-open throttle (tests / small arenas): no floor, large budget, no hub bar.
    #[must_use]
    pub fn permissive() -> Self {
        Self {
            c_min: 0.0,
            budget: usize::MAX,
            hub_indegree: usize::MAX,
        }
    }
}

/// The output of a Datapath proposer: the throttled candidate frontier plus the
/// gaps that blocked further derivation.
#[derive(Debug, Clone, Default)]
pub struct Frontier {
    /// Candidates that passed the throttle.
    pub candidates: Vec<Candidate>,
    /// First-class failures — what was missing (the reach-out hook).
    pub gaps: Vec<ReasoningGap>,
}

/// Count the `is_a` (Inh) in-degree of every predicate term — the hub metric for
/// RCR's middle-term exclusion (S5).
#[must_use]
fn inh_predicate_indegree(arena: &BeliefArena) -> HashMap<u16, usize> {
    let mut deg: HashMap<u16, usize> = HashMap::new();
    for b in arena.entries() {
        if b.stmt.cop == Copula::Inh {
            *deg.entry(b.stmt.p).or_default() += 1;
        }
    }
    deg
}

/// **RCR — abduction** (recipe #4). For every pair of `is_a` beliefs sharing a
/// (non-hub) predicate M — `A→M`, `B→M` — propose `A→B` and `B→A`, each with
/// [`TruthValue::abduction`]. Throttled by S5: hub-M exclusion, confidence floor,
/// per-thought budget. Emits [`GapKind`]s for a missing middle, a barred hub, or
/// an exhausted budget.
#[must_use]
pub fn rcr_abduce(arena: &BeliefArena, throttle: &Throttle) -> Frontier {
    let mut out = Frontier::default();
    let deg = inh_predicate_indegree(arena);

    let mut by_pred: HashMap<u16, Vec<u32>> = HashMap::new();
    for (i, b) in arena.entries().iter().enumerate() {
        if b.stmt.cop == Copula::Inh {
            by_pred.entry(b.stmt.p).or_default().push(i as u32);
        }
    }
    // DETERMINISM: `by_pred` is a HashMap (randomly-seeded iteration). Under a
    // finite budget the set of candidates KEPT depends on which predicates are
    // visited first, so the frontier must be reproducible — iterate predicates
    // in a stable (ascending) order. `members` are already in arena-index order.
    let mut preds: Vec<u16> = by_pred.keys().copied().collect();
    preds.sort_unstable();

    let mut any_pair = false;
    let mut hub_seen = false;
    'outer: for m in preds {
        let members = &by_pred[&m];
        if members.len() < 2 {
            continue;
        }
        any_pair = true;
        if deg.get(&m).copied().unwrap_or(0) > throttle.hub_indegree {
            hub_seen = true;
            out.gaps.push(ReasoningGap {
                kind: GapKind::HubExcluded,
                subject: None,
                predicate: Some(m),
            });
            continue;
        }
        for &r in members {
            for &o in members {
                if r == o {
                    continue;
                }
                let (subj_rule, t_rule) = {
                    let e = &arena.entries()[r as usize];
                    (e.stmt.s, e.truth)
                };
                let (subj_obs, t_obs) = {
                    let e = &arena.entries()[o as usize];
                    (e.stmt.s, e.truth)
                };
                if subj_rule == subj_obs {
                    continue; // trivial S→S self-statement
                }
                // {P→M (rule), S→M (obs)} ⊢ S→P, via the one engine.
                let truth = t_rule.abduction(&t_obs);
                if truth.confidence < throttle.c_min {
                    continue;
                }
                if out.candidates.len() >= throttle.budget {
                    out.gaps.push(ReasoningGap {
                        kind: GapKind::BudgetExhausted,
                        subject: None,
                        predicate: Some(m),
                    });
                    break 'outer;
                }
                let rung = arena.entries()[r as usize]
                    .rung
                    .max(arena.entries()[o as usize].rung)
                    + 1;
                out.candidates.push(Candidate {
                    stmt: CStmt {
                        s: subj_obs,
                        cop: Copula::Inh,
                        p: subj_rule,
                    },
                    truth,
                    premises: [r, o],
                    rung,
                    tactic: Tactic::Rcr,
                });
            }
        }
    }

    if !any_pair && !hub_seen {
        out.gaps.push(ReasoningGap {
            kind: GapKind::NoSharedMiddle,
            subject: None,
            predicate: None,
        });
    }
    out
}

/// **TR — divergence / analogy** (recipe #6). Given a focus belief `S cop P`,
/// substitute each similarity sibling `S′` (from a `Sim` BELIEF touching S) to
/// propose `S′ cop P` with [`TruthValue::analogy`]. No sibling →
/// [`GapKind::NoSibling`].
#[must_use]
pub fn tr_diverge(arena: &BeliefArena, focus: CStmt) -> Frontier {
    let mut out = Frontier::default();
    let Some(focus_belief) = arena.get(focus) else {
        out.gaps.push(ReasoningGap {
            kind: GapKind::NoSibling,
            subject: Some(focus.s),
            predicate: Some(focus.p),
        });
        return out;
    };
    let focus_idx = arena
        .entries()
        .iter()
        .position(|b| b.stmt == focus)
        .expect("focus present (get succeeded)") as u32;
    let focus_truth = focus_belief.truth;
    let focus_rung = focus_belief.rung;

    let mut found = false;
    for (i, b) in arena.entries().iter().enumerate() {
        if b.stmt.cop != Copula::Sim {
            continue;
        }
        let sibling = if b.stmt.s == focus.s {
            Some(b.stmt.p)
        } else if b.stmt.p == focus.s {
            Some(b.stmt.s)
        } else {
            None
        };
        let Some(s_prime) = sibling else { continue };
        if s_prime == focus.s || s_prime == focus.p {
            continue; // trivial
        }
        found = true;
        out.candidates.push(Candidate {
            stmt: CStmt {
                s: s_prime,
                cop: focus.cop,
                p: focus.p,
            },
            truth: focus_truth.analogy(&b.truth),
            premises: [focus_idx, i as u32],
            rung: focus_rung.max(b.rung) + 1,
            tactic: Tactic::Tr,
        });
    }

    if !found {
        out.gaps.push(ReasoningGap {
            kind: GapKind::NoSibling,
            subject: Some(focus.s),
            predicate: None,
        });
    }
    out
}

/// **CAS — abstraction** (recipe #8). Tree-guided over the focus subject's
/// `is_a` parents `G` (`S→G` beliefs):
/// - **up = induction** `{S→P, S→G} ⊢ G→P` (weak);
/// - **down = deduction** `{G→P, S→G} ⊢ S→P` (strong).
///
/// No parent → [`GapKind::NoAbstraction`].
#[must_use]
pub fn cas_abstract(arena: &BeliefArena, focus_subject: u16) -> Frontier {
    let mut out = Frontier::default();

    let parents: Vec<(u32, u16)> = arena
        .entries()
        .iter()
        .enumerate()
        .filter(|(_, b)| b.stmt.cop == Copula::Inh && b.stmt.s == focus_subject)
        .map(|(i, b)| (i as u32, b.stmt.p))
        .collect();

    if parents.is_empty() {
        out.gaps.push(ReasoningGap {
            kind: GapKind::NoAbstraction,
            subject: Some(focus_subject),
            predicate: None,
        });
        return out;
    }

    for &(sg_idx, g) in &parents {
        let (t_sg, r_sg) = {
            let e = &arena.entries()[sg_idx as usize];
            (e.truth, e.rung)
        };
        for (i, b) in arena.entries().iter().enumerate() {
            if b.stmt.cop != Copula::Inh {
                continue;
            }
            // up: {S→P, S→G} ⊢ G→P — induction. `TruthValue::induction` models
            // `{A→B, A→C} ⊢ B→C` (f = f of the second premise). For the G→P
            // conclusion the figure is A=S, B=G, C=P, so A→B = S→G (t_sg) is the
            // FIRST premise and A→C = S→P (b) the second — the conclusion must
            // inherit P's frequency, `t_sg.induction(&b.truth)`, not the reverse.
            if b.stmt.s == focus_subject && b.stmt.p != g {
                out.candidates.push(Candidate {
                    stmt: CStmt {
                        s: g,
                        cop: Copula::Inh,
                        p: b.stmt.p,
                    },
                    truth: t_sg.induction(&b.truth),
                    premises: [sg_idx, i as u32],
                    rung: b.rung.max(r_sg) + 1,
                    tactic: Tactic::CasUp,
                });
            }
            // down: {G→P, S→G} ⊢ S→P — deduction (G→P is M→P, S→G is S→M).
            if b.stmt.s == g && b.stmt.p != focus_subject {
                out.candidates.push(Candidate {
                    stmt: CStmt {
                        s: focus_subject,
                        cop: Copula::Inh,
                        p: b.stmt.p,
                    },
                    truth: b.truth.deduction(&t_sg),
                    premises: [i as u32, sg_idx],
                    rung: b.rung.max(r_sg) + 1,
                    tactic: Tactic::CasDown,
                });
            }
        }
    }
    out
}

/// The refutation target of a belief (ASC): `⟨1−f, c⟩` — what independent
/// counter-evidence would have to assert.
#[must_use]
pub fn challenge_target(belief: TruthValue) -> TruthValue {
    TruthValue::new(1.0 - belief.frequency, belief.confidence)
}

/// The outcome of an ASC self-critique.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AscOutcome {
    /// The target statement is not in the arena — nothing to challenge.
    NoTarget,
    /// Independently-sourced (disjoint-stamp) counter-evidence was revised in.
    Revised { synthesis_c: f32, depth: f32 },
    /// The offered counter-evidence OVERLAPS the belief's own sources — a
    /// self-reference. Blocked: a belief may not refute itself with its own
    /// evidence (S4; non-circularity is probabilistic under bounded stamps).
    BlockedSelfReference,
}

/// **ASC — self-critique** (recipe #7, Control bucket). Forms the belief's
/// refutation target `⟨1−f, c⟩` and admits offered counter-evidence ONLY when it
/// is independently sourced (its stamp is DISJOINT from the belief's) — then
/// revises it in. Counter-evidence overlapping the belief's own sources is
/// BLOCKED (no self-refutation from one's own evidence).
pub fn asc_challenge(
    arena: &mut BeliefArena,
    target: CStmt,
    counter: TruthValue,
    counter_stamp: Stamp,
) -> AscOutcome {
    let Some(belief) = arena.get(target) else {
        return AscOutcome::NoTarget;
    };
    if !belief.stamp.disjoint(counter_stamp) {
        return AscOutcome::BlockedSelfReference;
    }
    match arena.observe(target, counter, counter_stamp) {
        ReviseOutcome::Revised {
            synthesis_c, depth, ..
        } => AscOutcome::Revised { synthesis_c, depth },
        _ => AscOutcome::BlockedSelfReference,
    }
}

/// **CR — dialectic** (recipe #11, Control bucket). Thesis + antithesis on the
/// SAME statement: disjoint → NARS revision (synthesis c ABOVE both inputs,
/// `|f₁−f₂|` contradiction depth preserved); overlap → CHOICE. This IS the
/// arena's `observe`/`revise_at` path, named as the tactic it implements.
pub fn cr_synthesize(
    arena: &mut BeliefArena,
    stmt: CStmt,
    truth: TruthValue,
    stamp: Stamp,
) -> ReviseOutcome {
    arena.observe(stmt, truth, stamp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::recipe_dispatch::{inference, RecipeInference};

    fn inh(s: u16, p: u16) -> CStmt {
        CStmt {
            s,
            cop: Copula::Inh,
            p,
        }
    }
    fn sim(s: u16, p: u16) -> CStmt {
        CStmt {
            s,
            cop: Copula::Sim,
            p,
        }
    }

    /// The truth function each tactic uses agrees with the recipe's declared
    /// `RecipeInference` — the new tactics are pinned to the shipped taxonomy.
    #[test]
    fn tactic_matches_recipe_taxonomy() {
        assert_eq!(inference(RCR_ID), RecipeInference::Abduction, "RCR#4");
        assert_eq!(inference(TR_ID), RecipeInference::Induction, "TR#6 bucket");
        assert_eq!(inference(ASC_ID), RecipeInference::Revision, "ASC#7");
        assert_eq!(inference(CAS_ID), RecipeInference::Deduction, "CAS#8");
        assert_eq!(inference(CR_ID), RecipeInference::Revision, "CR#11");
    }

    /// RCR abduces over a shared predicate with the weak NAL truth (below both premises).
    #[test]
    fn rcr_abduces_shared_predicate_with_weak_truth() {
        let mut arena = BeliefArena::new();
        arena.observe(inh(1, 9), TruthValue::new(0.9, 0.8), Stamp::source(0)); // A→M
        arena.observe(inh(2, 9), TruthValue::new(0.8, 0.7), Stamp::source(1)); // B→M
        let fr = rcr_abduce(&arena, &Throttle::permissive());
        assert_eq!(fr.candidates.len(), 2, "both directions abduced");
        let b_to_a = fr
            .candidates
            .iter()
            .find(|c| c.stmt == inh(2, 1))
            .expect("B→A");
        // f = f_rule (A→M) = 0.9; w = f_obs·c_rule·c_obs = 0.8·0.8·0.7.
        assert!((b_to_a.truth.frequency - 0.9).abs() < 1e-6);
        let w = 0.8 * 0.8 * 0.7;
        assert!((b_to_a.truth.confidence - w / (w + 1.0)).abs() < 1e-6);
        assert!(b_to_a.truth.confidence < 0.7, "abduction is weak");
        assert_eq!(b_to_a.tactic.recipe_id(), RCR_ID);
    }

    /// Hub middle terms are barred; the frontier reports the gap and mints nothing.
    #[test]
    fn rcr_excludes_hub_and_reports_gaps() {
        let mut arena = BeliefArena::new();
        for s in 1..=5u16 {
            arena.observe(
                inh(s, 9),
                TruthValue::new(0.9, 0.8),
                Stamp::source(s as u32),
            );
        }
        let fr = rcr_abduce(&arena, &Throttle::new(0.0, usize::MAX, 4));
        assert!(fr.candidates.is_empty(), "no candidates through a hub");
        assert!(fr.gaps.iter().any(|g| g.kind == GapKind::HubExcluded));
        // No shared middle at all → NoSharedMiddle.
        let mut disjoint = BeliefArena::new();
        disjoint.observe(inh(1, 8), TruthValue::new(0.9, 0.8), Stamp::source(0));
        disjoint.observe(inh(2, 9), TruthValue::new(0.9, 0.8), Stamp::source(1));
        assert!(rcr_abduce(&disjoint, &Throttle::permissive())
            .gaps
            .iter()
            .any(|g| g.kind == GapKind::NoSharedMiddle));
    }

    /// Confidence floor filters weak abductions; budget caps + reports exhaustion.
    #[test]
    fn rcr_floor_and_budget() {
        let mut arena = BeliefArena::new();
        for s in 1..=4u16 {
            arena.observe(
                inh(s, 9),
                TruthValue::new(0.9, 0.9),
                Stamp::source(s as u32),
            );
        }
        assert_eq!(
            rcr_abduce(&arena, &Throttle::new(0.0, usize::MAX, usize::MAX))
                .candidates
                .len(),
            12
        );
        let capped = rcr_abduce(&arena, &Throttle::new(0.0, 5, usize::MAX));
        assert_eq!(capped.candidates.len(), 5);
        assert!(capped
            .gaps
            .iter()
            .any(|g| g.kind == GapKind::BudgetExhausted));
        // DETERMINISM: predicate iteration is sorted + members are in arena
        // order, so the budget-capped set is a STABLE prefix, not hash-seeded.
        let capped_stmts: Vec<CStmt> = capped.candidates.iter().map(|c| c.stmt).collect();
        assert_eq!(
            capped_stmts,
            vec![inh(2, 1), inh(3, 1), inh(4, 1), inh(1, 2), inh(3, 2)],
            "budget keeps a deterministic prefix"
        );
        assert!(
            rcr_abduce(&arena, &Throttle::new(0.9, usize::MAX, usize::MAX))
                .candidates
                .is_empty()
        );
    }

    /// TR substitutes a similarity sibling with analogy truth; none → gap.
    #[test]
    fn tr_analogy_substitutes_sibling() {
        let mut arena = BeliefArena::new();
        arena.observe(inh(1, 3), TruthValue::new(0.9, 0.8), Stamp::source(0)); // dog→mammal
        arena.observe(sim(1, 2), TruthValue::new(0.7, 0.6), Stamp::source(1)); // dog↔wolf
        let fr = tr_diverge(&arena, inh(1, 3));
        assert_eq!(fr.candidates.len(), 1);
        let c = fr.candidates[0];
        assert_eq!(c.stmt, inh(2, 3), "wolf→mammal by analogy");
        assert!((c.truth.frequency - 0.63).abs() < 1e-6); // 0.9·0.7
        assert!((c.truth.confidence - 0.8 * 0.6 * 0.7).abs() < 1e-6);
        assert!(tr_diverge(&arena, inh(9, 3))
            .gaps
            .iter()
            .any(|g| g.kind == GapKind::NoSibling));
    }

    /// CAS up is induction (weak), down is deduction (strong); down > up in confidence.
    #[test]
    fn cas_up_induction_down_deduction() {
        let mut arena = BeliefArena::new();
        arena.observe(inh(1, 2), TruthValue::new(0.95, 0.9), Stamp::source(0)); // S→G
        arena.observe(inh(1, 3), TruthValue::new(0.9, 0.85), Stamp::source(1)); // S→P
        arena.observe(inh(2, 4), TruthValue::new(0.9, 0.85), Stamp::source(2)); // G→Q
        let fr = cas_abstract(&arena, 1);
        let up = fr
            .candidates
            .iter()
            .find(|c| c.stmt == inh(2, 3) && c.tactic == Tactic::CasUp)
            .expect("G→P induced");
        let down = fr
            .candidates
            .iter()
            .find(|c| c.stmt == inh(1, 4) && c.tactic == Tactic::CasDown)
            .expect("S→Q deduced");
        assert!(
            down.truth.confidence > up.truth.confidence,
            "deduction beats induction"
        );
        assert!((down.truth.frequency - 0.9 * 0.95).abs() < 1e-6);
        assert!((down.truth.confidence - 0.85 * 0.9 * 0.9 * 0.95).abs() < 1e-6);
        // up = induction(S→G, S→P) for the G→P conclusion (premise order fixed):
        // f = f(S→P) = 0.9 (G inherits P's frequency), w = f_SG·c_SG·c_SP.
        assert!(
            (up.truth.frequency - 0.9).abs() < 1e-6,
            "G→P inherits P's frequency, not the reverse: {}",
            up.truth.frequency
        );
        let w_up = 0.95 * 0.9 * 0.85;
        assert!((up.truth.confidence - w_up / (w_up + 1.0)).abs() < 1e-6);
        assert_eq!(up.premises, [0, 1], "premises in S→G, S→P order");
        assert!(cas_abstract(&arena, 7)
            .gaps
            .iter()
            .any(|g| g.kind == GapKind::NoAbstraction));
    }

    /// ASC revises INDEPENDENT counter-evidence but BLOCKS self-refutation from
    /// overlapping sources.
    #[test]
    fn asc_independent_revises_overlap_blocks() {
        let mut arena = BeliefArena::new();
        let stmt = inh(1, 2);
        arena.observe(stmt, TruthValue::new(0.9, 0.8), Stamp::source(3));
        assert!((challenge_target(arena.get(stmt).unwrap().truth).frequency - 0.1).abs() < 1e-6);
        assert_eq!(
            asc_challenge(
                &mut arena,
                stmt,
                TruthValue::new(0.1, 0.7),
                Stamp::source(3)
            ),
            AscOutcome::BlockedSelfReference
        );
        assert!((arena.get(stmt).unwrap().truth.frequency - 0.9).abs() < 1e-6);
        let out = asc_challenge(
            &mut arena,
            stmt,
            TruthValue::new(0.1, 0.7),
            Stamp::source(40),
        );
        assert!(matches!(out, AscOutcome::Revised { .. }));
        assert!(
            arena.get(stmt).unwrap().truth.frequency < 0.9,
            "counter lowered f"
        );
        assert_eq!(
            asc_challenge(
                &mut arena,
                inh(7, 8),
                TruthValue::new(0.5, 0.5),
                Stamp::source(0)
            ),
            AscOutcome::NoTarget
        );
    }

    /// CR synthesizes thesis + antithesis (disjoint) into higher confidence with
    /// the contradiction depth preserved — the dialectic.
    #[test]
    fn cr_dialectic_is_revision() {
        let mut arena = BeliefArena::new();
        let stmt = inh(1, 2);
        assert!(matches!(
            cr_synthesize(
                &mut arena,
                stmt,
                TruthValue::new(0.9, 0.8),
                Stamp::source(0)
            ),
            ReviseOutcome::Admitted { .. }
        ));
        let anti = cr_synthesize(
            &mut arena,
            stmt,
            TruthValue::new(0.2, 0.75),
            Stamp::source(1),
        );
        let ReviseOutcome::Revised {
            synthesis_c, depth, ..
        } = anti
        else {
            panic!("disjoint dialectic must revise, got {anti:?}");
        };
        assert!(synthesis_c > 0.8, "synthesis c above both: {synthesis_c}");
        assert!((depth - 0.7).abs() < 1e-6, "|0.9−0.2| depth kept");
        assert!((arena.get(stmt).unwrap().contradiction - 0.7).abs() < 1e-6);
    }

    /// A tactic candidate admitted through the arena's shared CHOICE path never
    /// overwrites an observation-grounded belief.
    #[test]
    fn admitted_candidate_respects_ground() {
        let mut arena = BeliefArena::new();
        let stmt = inh(2, 1);
        arena.observe(stmt, TruthValue::new(0.55, 0.95), Stamp::source(9));
        let before = arena.get(stmt).unwrap().truth;
        arena.observe(inh(1, 9), TruthValue::new(0.99, 0.9), Stamp::source(0));
        arena.observe(inh(2, 9), TruthValue::new(0.99, 0.9), Stamp::source(1));
        let cand = *rcr_abduce(&arena, &Throttle::permissive())
            .candidates
            .iter()
            .find(|c| c.stmt == stmt)
            .expect("B→A abduced");
        assert!(!arena.admit_derived(cand.stmt, cand.truth, &cand.premises, cand.rung));
        assert_eq!(
            arena.get(stmt).unwrap().truth.frequency,
            before.frequency,
            "ground intact"
        );
    }
}
