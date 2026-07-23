//! `belief` — the V0 falsifying slice of the dialectic engine
//! (`.claude/plans/dialectic-engine-v1.md` §4): a Belief-carrying arena where
//! **revision merges truth in place on the unique statement** (triple-keyed
//! dedup — the termination proof of `reason.rs` preserved verbatim) under a
//! **fixed-width observation-source stamp** with the disjointness guard the
//! shipped `nars_revision` lacks.
//!
//! Synthesis decisions embodied (S2, S3, S4 of the plan):
//! - **S2:** a statement exists ONCE; revision updates `(truth, stamp)` in
//!   place at its EXISTING rung; only genuinely-new statements get
//!   `max(premise rungs)+1`. Closure-internal duplicates resolve by CHOICE
//!   (higher expectation), never a stamped duplicate entry.
//! - **S3:** `Copula { Inh, Sim, Impl, Rel }` — **only `Inh` and `Sim`
//!   auto-transit**; `Rel` (FSM verbs) NEVER freely composes. This fixes the
//!   latent unsoundness of the blanket same-predicate closure
//!   ("dog bit man" + "man bit sandwich" ⊬ "dog bit sandwich").
//! - **S4:** `Stamp(u64)` — a bitset over a bounded horizon of OBSERVATION
//!   sources (never derivation ancestry: unbounded, serializing). Disjoint →
//!   NARS revision (evidence pooling, synthesis confidence above both inputs,
//!   contradiction depth |f₁−f₂| preserved). Overlap → CHOICE, no double count.
//!
//! NARS is treated as LOGIC here (`E-NARS-IS-LOGIC-...-1`): statements compose
//! by shared terms; truth moves only by truth functions. No fingerprints.

use lance_graph_contract::exploration::NarsTruth;
use std::collections::HashMap;

/// Fixed-width evidential stamp: bit *i* = observation source *i* (bounded
/// horizon of 64 sources; sources beyond the horizon fold by modulo, which is
/// CONSERVATIVE — folding can only create false overlap, never false
/// disjointness, so the no-double-count guarantee survives the bound).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stamp(pub u64);

impl Stamp {
    /// The stamp of a single observation source.
    #[must_use]
    pub fn source(id: u32) -> Self {
        Stamp(1u64 << (id % 64))
    }
    /// Two stamps share no evidence.
    #[must_use]
    pub fn disjoint(self, other: Self) -> bool {
        self.0 & other.0 == 0
    }
    /// Pooled evidence base.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Stamp(self.0 | other.0)
    }
}

/// The copula of a concept-level statement (S3). `Rel` carries an arbitrary
/// relational term (an FSM verb id) — stored, queryable, NEVER transitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Copula {
    /// Inheritance `→` (is_a). Transitive: composes by shared middle term.
    Inh,
    /// Similarity `↔`. Transitive (resemblance chains, weaker).
    Sim,
    /// Implication `⇒` (reasoned causality). NOT auto-transitive in V0 —
    /// causal chaining goes through explicit tactics (V1), never blanket closure.
    Impl,
    /// An arbitrary relational term (verb). NEVER transits.
    Rel(u16),
}

impl Copula {
    /// May the blanket transitive closure compose two statements of this copula?
    #[must_use]
    pub fn transits(self) -> bool {
        matches!(self, Copula::Inh | Copula::Sim)
    }
}

/// A concept-level statement: subject –copula– predicate (terms are interned
/// dense concept ids; `NodeGuid` remains the addressable identity outside the
/// reasoning index — never a 16-byte pivot key).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CStmt {
    /// Subject concept id.
    pub s: u16,
    /// The copula (carries the verb id when `Rel`).
    pub cop: Copula,
    /// Predicate concept id.
    pub p: u16,
}

/// One belief: a unique statement with its NARS truth, evidential stamp,
/// Tarski rung, and premise pointers (empty = observed).
#[derive(Debug, Clone)]
pub struct Belief {
    /// The statement (UNIQUE in the arena — S2).
    pub stmt: CStmt,
    /// Current NARS truth (moves only via revision/CHOICE).
    pub truth: NarsTruth,
    /// Evidential base (bounded source bitset — S4).
    pub stamp: Stamp,
    /// Tarski rung: 0 observed; derived = max(premise rungs)+1, fixed at
    /// creation — revision does NOT change it (S2 / rung-inflation fix).
    pub rung: u32,
    /// Arena indices of premises (derived beliefs only; always earlier).
    pub premises: Vec<u32>,
    /// Preserved dialectic depth: max |f₁−f₂| seen across revisions of this
    /// statement (the contradiction is committed, not erased).
    pub contradiction: f32,
}

/// Outcome of offering evidence to the arena (observe/revise path).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReviseOutcome {
    /// New statement admitted (was absent).
    Admitted { id: u32 },
    /// Same statement, DISJOINT stamps → NARS revision pooled the evidence in
    /// place; `synthesis_c` is the pooled confidence (≥ both inputs),
    /// `depth` the |f₁−f₂| dialectic gap that was preserved.
    Revised {
        id: u32,
        synthesis_c: f32,
        depth: f32,
    },
    /// Same statement, OVERLAPPING stamps → CHOICE kept the higher-confidence
    /// truth; no evidence was double-counted.
    Chosen { id: u32, kept_existing: bool },
}

/// The dialectic belief arena: triple-keyed (statement-keyed) — a statement
/// exists exactly once, so the `reason.rs` finite-closure termination argument
/// applies verbatim (the derivable set ⊆ terms × copulas × terms, finite; each
/// statement admitted at most once).
#[derive(Debug, Default)]
pub struct BeliefArena {
    entries: Vec<Belief>,
    index: HashMap<CStmt, u32>,
    /// Closure passes run by the last `close_transitive` call.
    pub passes: u32,
    /// Whether the last closure reached a true fixed point.
    pub reached_fixed_point: bool,
}

impl BeliefArena {
    /// Empty arena.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// All beliefs, in admission order.
    #[must_use]
    pub fn entries(&self) -> &[Belief] {
        &self.entries
    }

    /// Look up a statement's belief.
    #[must_use]
    pub fn get(&self, stmt: CStmt) -> Option<&Belief> {
        self.index.get(&stmt).map(|&i| &self.entries[i as usize])
    }

    /// Offer evidence for a statement (rung 0 observation path). Admits it if
    /// absent; otherwise routes through the S4 guard: disjoint → revision
    /// in place; overlap → CHOICE. The arena NEVER grows a second entry for an
    /// existing statement — the termination invariant.
    pub fn observe(&mut self, stmt: CStmt, truth: NarsTruth, stamp: Stamp) -> ReviseOutcome {
        match self.index.get(&stmt) {
            None => {
                let id = self.entries.len() as u32;
                self.entries.push(Belief {
                    stmt,
                    truth,
                    stamp,
                    rung: 0,
                    premises: Vec::new(),
                    contradiction: 0.0,
                });
                self.index.insert(stmt, id);
                ReviseOutcome::Admitted { id }
            }
            Some(&id) => self.revise_at(id, truth, stamp),
        }
    }

    /// The S4 revision guard on an existing belief. Disjoint stamps → NARS
    /// evidence pooling (`NarsTruth::revision`) + stamp union + preserved
    /// |f₁−f₂| depth, IN PLACE (rung untouched). Overlapping stamps → CHOICE:
    /// keep the higher-confidence truth, count nothing twice.
    pub fn revise_at(&mut self, id: u32, new: NarsTruth, stamp: Stamp) -> ReviseOutcome {
        let b = &mut self.entries[id as usize];
        if b.stamp.disjoint(stamp) {
            let depth = (b.truth.frequency - new.frequency).abs();
            b.contradiction = b.contradiction.max(depth);
            b.truth = b.truth.revision(&new);
            b.stamp = b.stamp.union(stamp);
            ReviseOutcome::Revised {
                id,
                synthesis_c: b.truth.confidence,
                depth,
            }
        } else {
            let kept_existing = b.truth.confidence >= new.confidence;
            if !kept_existing {
                b.truth = new;
                b.stamp = b.stamp.union(stamp);
            }
            ReviseOutcome::Chosen { id, kept_existing }
        }
    }

    /// Copula-gated transitive closure (S3): compose `{A cop B, B cop C} ⊢
    /// A cop C` ONLY for `cop.transits()` (Inh, Sim), with NARS deduction
    /// truth `f=f₁f₂, c=f₁f₂c₁c₂` computed per ordered premise pair (truth is
    /// NOT an mxm quantity — S1; this scalar path IS the second-pass form at
    /// V0 scale). A statement exists exactly once (S2); every derivation of
    /// that statement — including multiple paths to the same conclusion (a weak
    /// A→B→C vs a strong A→D→C) — is resolved by **CHOICE on `expectation()`**,
    /// NOT by which path was walked first. Concretely, per pass:
    /// - all derivations are collected keyed by the derived statement, keeping
    ///   the maximum-`expectation()` candidate (order-independent — the Codex
    ///   P2 fix; the earlier `continue`-on-duplicate made stored truth depend
    ///   on iteration order);
    /// - an ABSENT statement is admitted rung-stamped `max(premise rungs)+1`;
    /// - an OBSERVATION-GROUNDED statement (non-empty stamp — whether observed
    ///   before it was ever derived, rung 0, or derived-then-observed, rung ≥ 1)
    ///   is ground and a pure derivation never overrides it;
    /// - a PURE-DERIVED statement (empty stamp) is updated only when the winning
    ///   derivation's `expectation()` strictly exceeds the stored one (`EPS`).
    ///
    /// Termination: each stored expectation only ever increases and is bounded
    /// (deduction confidence is a product of confidences < 1, so a longer path
    /// can never exceed a shorter one), so the number of `changed` passes is
    /// finite — the `reason.rs` finiteness argument holds and the closure
    /// reaches a true fixed point (`max_passes` is only a backstop).
    pub fn close_transitive(&mut self, max_passes: u32) {
        self.passes = 0;
        self.reached_fixed_point = false;
        // A CHOICE update fires only on a MEANINGFUL expectation gain, so float
        // churn cannot re-trigger passes indefinitely.
        const EPS: f32 = 1e-6;
        while self.passes < max_passes {
            self.passes += 1;
            // Pivot: (subject, copula) → entry ids, over the CURRENT arena.
            let mut by_sc: HashMap<(u16, Copula), Vec<u32>> = HashMap::new();
            for (i, b) in self.entries.iter().enumerate() {
                if b.stmt.cop.transits() {
                    by_sc
                        .entry((b.stmt.s, b.stmt.cop))
                        .or_default()
                        .push(i as u32);
                }
            }
            // Collect EVERY derivation this pass, keyed by the derived
            // statement, keeping the maximum-`expectation()` candidate. Two
            // paths to the same conclusion no longer race on insertion order.
            let mut derived: HashMap<CStmt, (NarsTruth, u32, [u32; 2])> = HashMap::new();
            for i in 0..self.entries.len() {
                let (a, cop, b_term, t1, r1) = {
                    let e = &self.entries[i];
                    (e.stmt.s, e.stmt.cop, e.stmt.p, e.truth, e.rung)
                };
                if !cop.transits() {
                    continue;
                }
                let Some(js) = by_sc.get(&(b_term, cop)) else {
                    continue;
                };
                for &j in js {
                    let (c_term, t2, r2) = {
                        let e = &self.entries[j as usize];
                        (e.stmt.p, e.truth, e.rung)
                    };
                    let stmt = CStmt {
                        s: a,
                        cop,
                        p: c_term,
                    };
                    // NARS deduction truth, per ordered pair (S1 second-pass form).
                    let f = t1.frequency * t2.frequency;
                    let c = f * t1.confidence * t2.confidence;
                    let truth = NarsTruth::new(f, c);
                    let cand = (truth, r1.max(r2) + 1, [i as u32, j]);
                    match derived.get(&stmt) {
                        Some((best, ..)) if best.expectation() >= truth.expectation() => {}
                        _ => {
                            derived.insert(stmt, cand);
                        }
                    }
                }
            }
            // Apply each winning derivation through CHOICE against the arena.
            // Absent → admit; observation-grounded → untouched; pure derivation →
            // update only on a strict expectation gain. Order-independent: at
            // most one candidate per statement, and admit/gain-update never
            // interfere across distinct statements.
            let mut changed = false;
            for (stmt, (truth, rung, premises)) in derived {
                match self.index.get(&stmt) {
                    Some(&id) => {
                        let e = &mut self.entries[id as usize];
                        // Any belief carrying observation evidence — a non-empty
                        // stamp — is GROUND: a pure derivation never overwrites
                        // it. This catches BOTH a purely-observed belief (rung 0)
                        // AND a derived-then-observed one (rung ≥ 1, but its stamp
                        // was unioned in by `revise_at` — a derived belief's zero
                        // stamp is disjoint from every observation, so observing
                        // it always pools). The `rung == 0` guard missed the
                        // latter and would drop the observed evidence.
                        if e.stamp != Stamp::default() {
                            continue; // observation dominates a derivation
                        }
                        if truth.expectation() > e.truth.expectation() + EPS {
                            e.truth = truth;
                            e.rung = rung;
                            e.premises = premises.to_vec();
                            changed = true;
                        }
                    }
                    None => {
                        let id = self.entries.len() as u32;
                        self.entries.push(Belief {
                            stmt,
                            truth,
                            stamp: Stamp::default(), // derived: no observation sources of its own
                            rung,
                            premises: premises.to_vec(),
                            contradiction: 0.0,
                        });
                        self.index.insert(stmt, id);
                        changed = true;
                    }
                }
            }
            if !changed {
                self.reached_fixed_point = true;
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inh(s: u16, p: u16) -> CStmt {
        CStmt {
            s,
            cop: Copula::Inh,
            p,
        }
    }

    /// REGISTERED V0 GATE 1 (`dialectic-engine-v1` §4): disjoint-stamp
    /// revision pools evidence per the NARS formula (synthesis confidence
    /// above both inputs, dialectic depth preserved) AND the closure over a
    /// ~10-concept KG still reaches a true fixed point — the P0s falsified
    /// cheaply if this reds.
    #[test]
    fn revision_disjoint_stamps_moves_truth_and_still_terminates() {
        let mut arena = BeliefArena::new();
        // A 10-concept is_a chain with a cycle at the tail (worst case for
        // termination) …
        for k in 0..9u16 {
            arena.observe(
                inh(k, k + 1),
                NarsTruth::new(0.9, 0.9),
                Stamp::source(k as u32),
            );
        }
        arena.observe(inh(9, 0), NarsTruth::new(0.9, 0.9), Stamp::source(9)); // cycle
                                                                              // … and the dialectic: thesis vs antithesis on the SAME statement from
                                                                              // DISJOINT sources (the CR worked example: w₁=9, w₂=3).
        let stmt = inh(0, 1);
        let out = arena.observe(stmt, NarsTruth::new(0.2, 0.75), Stamp::source(40));
        let ReviseOutcome::Revised {
            synthesis_c, depth, ..
        } = out
        else {
            panic!("disjoint stamps must revise, got {out:?}");
        };
        // Synthesis: f=(9·0.9+3·0.2)/12=0.725, c=12/13≈0.923 — ABOVE both inputs.
        let b = arena.get(stmt).unwrap();
        assert!(
            (b.truth.frequency - 0.725).abs() < 1e-3,
            "pooled f: {}",
            b.truth.frequency
        );
        // Synthesis confidence exceeds BOTH inputs (0.9 is the max, so > 0.9
        // subsumes > 0.75 — the pooling property, c=(w₁+w₂)/(w₁+w₂+1)≈0.923).
        assert!(
            synthesis_c > 0.9,
            "synthesis c above both inputs: {synthesis_c}"
        );
        assert!(
            (depth - 0.7).abs() < 1e-6,
            "dialectic depth |0.9−0.2| preserved"
        );
        assert!(
            (b.contradiction - 0.7).abs() < 1e-6,
            "committed, not erased"
        );
        // Termination: the closure over the cyclic 10-concept chain reaches a
        // TRUE fixed point (finite statement set + statement-keyed dedup).
        arena.close_transitive(1024);
        assert!(arena.reached_fixed_point, "closure must terminate (S2)");
        // Full cyclic closure: all 10×10 ordered pairs (incl. self-loops).
        assert_eq!(arena.entries().len(), 100, "finite closure of the 10-cycle");
        // Revision after closure STILL terminates arena growth: same statement,
        // new disjoint source → in-place, no new entry.
        let n = arena.entries().len();
        arena.observe(stmt, NarsTruth::new(0.8, 0.5), Stamp::source(41));
        assert_eq!(arena.entries().len(), n, "revision never mints an entry");
    }

    /// REGISTERED V0 GATE 2: overlapping stamps are REJECTED from revision —
    /// CHOICE keeps the higher-confidence truth and counts nothing twice (the
    /// guard the shipped nars_revision lacks; see TECH_DEBT).
    #[test]
    fn revision_overlapping_stamp_is_rejected() {
        let mut arena = BeliefArena::new();
        let stmt = inh(1, 2);
        arena.observe(stmt, NarsTruth::new(0.9, 0.8), Stamp::source(7));
        let before = arena.get(stmt).unwrap().truth;
        // Same source (overlap) offering "more of the same evidence".
        let out = arena.observe(stmt, NarsTruth::new(0.9, 0.6), Stamp::source(7));
        assert_eq!(
            out,
            ReviseOutcome::Chosen {
                id: 0,
                kept_existing: true
            },
            "overlap → CHOICE, not revision"
        );
        let after = arena.get(stmt).unwrap().truth;
        assert_eq!(
            (after.frequency, after.confidence),
            (before.frequency, before.confidence),
            "no double count: truth unchanged by overlapping evidence"
        );
        // Overlap with HIGHER confidence: choice swaps, still no pooling.
        let out = arena.observe(stmt, NarsTruth::new(0.4, 0.95), Stamp::source(7));
        assert_eq!(
            out,
            ReviseOutcome::Chosen {
                id: 0,
                kept_existing: false
            }
        );
        let after = arena.get(stmt).unwrap().truth;
        assert!(
            (after.confidence - 0.95).abs() < 1e-6,
            "choice keeps higher c"
        );
        assert!(after.confidence < 0.96, "…but never ABOVE it (no pooling)");
    }

    /// S3 soundness gate: verbs (Rel) never transit — the latent unsoundness
    /// of the blanket same-predicate closure is closed at the copula.
    #[test]
    fn verbs_do_not_transit() {
        let mut arena = BeliefArena::new();
        let bit = Copula::Rel(77); // "bit"
        arena.observe(
            CStmt {
                s: 1,
                cop: bit,
                p: 2,
            }, // dog bit man
            NarsTruth::new(1.0, 0.9),
            Stamp::source(0),
        );
        arena.observe(
            CStmt {
                s: 2,
                cop: bit,
                p: 3,
            }, // man bit sandwich
            NarsTruth::new(1.0, 0.9),
            Stamp::source(1),
        );
        arena.close_transitive(16);
        assert!(arena.reached_fixed_point);
        assert_eq!(arena.entries().len(), 2, "dog bit sandwich must NOT derive");
        assert!(arena
            .get(CStmt {
                s: 1,
                cop: bit,
                p: 3
            })
            .is_none());
        // Control: the same shape under Inh DOES transit, with deduction truth.
        let mut inh_arena = BeliefArena::new();
        inh_arena.observe(inh(1, 2), NarsTruth::new(0.9, 0.8), Stamp::source(0));
        inh_arena.observe(inh(2, 3), NarsTruth::new(1.0, 0.95), Stamp::source(1));
        inh_arena.close_transitive(16);
        let d = inh_arena.get(inh(1, 3)).expect("Inh transits");
        assert!((d.truth.frequency - 0.9).abs() < 1e-6); // f = 0.9·1.0
        assert!((d.truth.confidence - 0.9 * 0.8 * 0.95).abs() < 1e-6); // c = f·c₁c₂
        assert_eq!(d.rung, 1);
        assert_eq!(d.premises.len(), 2, "premise pointers carried");
    }

    /// S2 rung semantics: revision is an in-place truth UPDATE — the rung does
    /// not move (no rung inflation; one statement, one rung).
    #[test]
    fn revision_keeps_rung_in_place() {
        let mut arena = BeliefArena::new();
        arena.observe(inh(1, 2), NarsTruth::new(0.9, 0.8), Stamp::source(0));
        arena.observe(inh(2, 3), NarsTruth::new(0.9, 0.8), Stamp::source(1));
        arena.close_transitive(16);
        let derived = inh(1, 3);
        assert_eq!(arena.get(derived).unwrap().rung, 1);
        // Independent observation of the derived statement revises it in place…
        let out = arena.observe(derived, NarsTruth::new(0.7, 0.6), Stamp::source(9));
        assert!(matches!(out, ReviseOutcome::Revised { .. }));
        // …at its EXISTING rung.
        assert_eq!(
            arena.get(derived).unwrap().rung,
            1,
            "rung fixed at creation"
        );
    }

    /// Close a two-path arena (weak A→B→C vs strong A→D→C) in a chosen
    /// insertion order and return the derived truth of A→C. A=0, B=1, C=2, D=3.
    fn close_two_paths_to_ac(strong_first: bool) -> NarsTruth {
        let weak: [(CStmt, NarsTruth); 2] = [
            (inh(0, 1), NarsTruth::new(0.6, 0.5)), // A→B (weak)
            (inh(1, 2), NarsTruth::new(0.6, 0.5)), // B→C (weak)
        ];
        let strong: [(CStmt, NarsTruth); 2] = [
            (inh(0, 3), NarsTruth::new(0.95, 0.9)), // A→D (strong)
            (inh(3, 2), NarsTruth::new(0.95, 0.9)), // D→C (strong)
        ];
        let order: [&[(CStmt, NarsTruth)]; 2] = if strong_first {
            [&strong, &weak]
        } else {
            [&weak, &strong]
        };
        let mut arena = BeliefArena::new();
        let mut src = 0u32;
        for group in order {
            for &(stmt, truth) in group {
                arena.observe(stmt, truth, Stamp::source(src));
                src += 1;
            }
        }
        arena.close_transitive(64);
        assert!(arena.reached_fixed_point, "closure must terminate");
        arena.get(inh(0, 2)).expect("A→C must derive").truth
    }

    /// CODEX P2 REGRESSION (`close_transitive` CHOICE-for-duplicates fix): when
    /// two paths reach the same conclusion (a weak A→B→C and a strong A→D→C),
    /// the stored truth of A→C is decided by CHOICE on `expectation()` and is
    /// therefore INDEPENDENT of which path is walked first. Before the fix the
    /// first-encountered derivation was `continue`-skipped and permanently won,
    /// making stored truth insertion-order-dependent (violating S2).
    #[test]
    fn transitive_duplicate_resolves_by_choice_not_insertion_order() {
        // Deduction truth of the STRONG path A→C: f=0.95², c=f·0.9² — the higher
        // expectation, so CHOICE must keep it in BOTH orderings.
        let want_f = 0.95f32 * 0.95;
        let want_c = want_f * 0.9 * 0.9;

        let strong_first = close_two_paths_to_ac(true);
        let weak_first = close_two_paths_to_ac(false);

        for (label, t) in [("strong-first", strong_first), ("weak-first", weak_first)] {
            assert!(
                (t.frequency - want_f).abs() < 1e-6 && (t.confidence - want_c).abs() < 1e-6,
                "{label}: CHOICE must keep the strong-path truth (f={want_f}, c={want_c}), \
                 got f={}, c={}",
                t.frequency,
                t.confidence,
            );
        }
        // The two orderings agree exactly — the order-independence Codex asked for.
        assert!((strong_first.frequency - weak_first.frequency).abs() < 1e-9);
        assert!((strong_first.confidence - weak_first.confidence).abs() < 1e-9);
    }

    /// CODEX P2 REGRESSION (the second, subtler one): a belief that is DERIVED
    /// first and OBSERVED later keeps its rung (≥ 1) but now carries real
    /// observation evidence (its stamp was unioned in by revision). A later
    /// closure that finds a higher-expectation pure derivation must NOT
    /// overwrite it — the `rung == 0` guard missed this; the stamp guard catches
    /// it. Scenario: derive A→C from weak A→B→C, observe A→C, add strong
    /// A→D→C, re-close — the observed/revised truth must survive intact.
    #[test]
    fn closure_does_not_overwrite_a_derived_then_observed_belief() {
        let mut arena = BeliefArena::new();
        // 1) Derive A→C weakly from A→B→C (rung 1, empty stamp).
        arena.observe(inh(0, 1), NarsTruth::new(0.6, 0.5), Stamp::source(0));
        arena.observe(inh(1, 2), NarsTruth::new(0.6, 0.5), Stamp::source(1));
        arena.close_transitive(64);
        assert_eq!(arena.get(inh(0, 2)).unwrap().rung, 1, "A→C derived");
        // 2) Observe A→C directly (disjoint source): revision pools evidence in
        //    place at rung 1 — the belief now carries a non-empty stamp. Chosen
        //    so its expectation stays BELOW the strong path's, proving the guard
        //    (not the expectation test) is what protects it.
        let out = arena.observe(inh(0, 2), NarsTruth::new(0.55, 0.9), Stamp::source(2));
        assert!(
            matches!(out, ReviseOutcome::Revised { .. }),
            "disjoint → revision (derived belief's zero stamp is disjoint from any source)"
        );
        let observed = arena.get(inh(0, 2)).unwrap().truth;
        // 3) Add a STRONGER path A→D→C (deduction expectation ≈ 0.79, above the
        //    revised A→C) and re-close. The pre-fix code would overwrite A→C with
        //    the pure derivation, dropping the observation.
        arena.observe(inh(0, 3), NarsTruth::new(0.95, 0.9), Stamp::source(3));
        arena.observe(inh(3, 2), NarsTruth::new(0.95, 0.9), Stamp::source(4));
        arena.close_transitive(64);
        let after = arena.get(inh(0, 2)).unwrap().truth;
        assert_eq!(
            (after.frequency, after.confidence),
            (observed.frequency, observed.confidence),
            "closure must not overwrite an observation-grounded belief with a pure derivation"
        );
        // It did NOT become the pure strong derivation (f = 0.95²) — observation survived.
        assert!(
            (after.frequency - 0.95 * 0.95).abs() > 1e-2,
            "observed evidence survived, not replaced by the strong derivation"
        );
    }
}
