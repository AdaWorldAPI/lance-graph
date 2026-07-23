//! `belief` — the statement-keyed **Belief arena**: the dialectic engine's
//! reasoning substrate, in the lance-graph reasoning layer.
//!
//! This is the reasoning home ruled by
//! `E-DEEPNSM-V2-IS-INBOUND-LEG-REASONING-LIVES-IN-LANCE-GRAPH-1`: the arena
//! composes concept-level STATEMENTS by their shared terms and moves truth ONLY
//! by the one engine's truth functions ([`TruthValue`]) — never a local truth
//! reimplementation, never a fingerprint/popcount distance
//! (`E-NARS-IS-LOGIC-...-1`). `deepnsm-v2` (the inbound leg) emits the
//! belief/SPO stream that feeds this arena; the arena + the tactics
//! ([`super::tactics`]) are the reasoning.
//!
//! Synthesis decisions embodied (`dialectic-engine-v1.md` §1):
//! - **S2:** a statement exists ONCE; revision updates `(truth, stamp)` in place
//!   at its EXISTING rung; only genuinely-new statements get
//!   `max(premise rungs)+1`; closure-internal duplicates resolve by CHOICE on
//!   `expectation()` (order-independent — the Codex `close_transitive` fix).
//! - **S3:** `Copula { Inh, Sim, Impl, Rel }` — only `Inh`/`Sim` auto-transit;
//!   `Rel` (FSM verbs) NEVER freely composes.
//! - **S4:** `Stamp(u64)` bounded observation-source bitset. Disjoint → NARS
//!   revision (evidence pooling, synthesis c above both, `|f₁−f₂|` kept);
//!   overlap → CHOICE, no double count.

use super::truth::TruthValue;
use std::collections::HashMap;

/// Fixed-width evidential stamp: bit *i* = observation source *i* (bounded
/// horizon of 64; sources beyond fold by modulo — CONSERVATIVE: folding can only
/// create false overlap, never false disjointness, so no-double-count survives).
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
    /// Implication `⇒` (reasoned causality). NOT auto-transitive — causal
    /// chaining goes through explicit tactics, never blanket closure.
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

/// A concept-level statement: subject –copula– predicate (dense interned concept
/// ids; the addressable `NodeGuid` identity lives outside the reasoning index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CStmt {
    /// Subject concept id.
    pub s: u16,
    /// The copula (carries the verb id when `Rel`).
    pub cop: Copula,
    /// Predicate concept id.
    pub p: u16,
}

/// One belief: a unique statement with its NARS truth, evidential stamp, Tarski
/// rung, and premise pointers (empty = observed).
#[derive(Debug, Clone)]
pub struct Belief {
    /// The statement (UNIQUE in the arena — S2).
    pub stmt: CStmt,
    /// Current NARS truth (moves only via revision/CHOICE/derivation).
    pub truth: TruthValue,
    /// Evidential base (bounded source bitset — S4).
    pub stamp: Stamp,
    /// Tarski rung: 0 observed; derived = `max(premise rungs)+1`, fixed at
    /// creation — revision does NOT change it (S2 / rung-inflation fix).
    pub rung: u32,
    /// Arena indices of premises (derived beliefs only; always earlier).
    pub premises: Vec<u32>,
    /// Preserved dialectic depth: max `|f₁−f₂|` across revisions (the
    /// contradiction is committed, not erased).
    pub contradiction: f32,
}

/// Outcome of offering evidence to the arena (observe/revise path).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReviseOutcome {
    /// New statement admitted (was absent).
    Admitted { id: u32 },
    /// Same statement, DISJOINT stamps → NARS revision pooled the evidence in
    /// place; `synthesis_c` is the pooled confidence (≥ both inputs), `depth`
    /// the `|f₁−f₂|` dialectic gap preserved.
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
/// exists exactly once, so the finite-closure termination argument holds (the
/// derivable set ⊆ terms × copulas × terms, finite; each statement admitted at
/// most once).
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
    /// absent; otherwise routes through the S4 guard: disjoint → revision in
    /// place; overlap → CHOICE. The arena NEVER grows a second entry for an
    /// existing statement — the termination invariant.
    pub fn observe(&mut self, stmt: CStmt, truth: TruthValue, stamp: Stamp) -> ReviseOutcome {
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

    /// The S4 revision guard on an existing belief. A NON-EMPTY incoming stamp
    /// that is disjoint from the belief's own sources → NARS evidence pooling
    /// ([`TruthValue::revise`]) + stamp union + preserved `|f₁−f₂|` depth, IN
    /// PLACE (rung untouched). Otherwise (overlap OR an EMPTY incoming stamp) →
    /// CHOICE: keep the higher-confidence truth, count nothing twice.
    ///
    /// The empty-stamp guard is load-bearing: `Stamp::default()` (the "no
    /// observation source" sentinel) is disjoint from EVERY stamp, so treating
    /// it as independent evidence would let a repeated zero-stamped observation
    /// pool into itself and inflate confidence without bound (`union` also never
    /// records overlap). Unsourced evidence cannot pool — it competes by CHOICE.
    pub fn revise_at(&mut self, id: u32, new: TruthValue, stamp: Stamp) -> ReviseOutcome {
        let b = &mut self.entries[id as usize];
        if stamp != Stamp::default() && b.stamp.disjoint(stamp) {
            let depth = (b.truth.frequency - new.frequency).abs();
            b.contradiction = b.contradiction.max(depth);
            b.truth = b.truth.revise(&new);
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

    /// Admit a DERIVED candidate — a conclusion with no observation source of its
    /// own — through the CHOICE discipline shared with `close_transitive`. This
    /// is the throttled-frontier admission path (S5): a tactic PROPOSES
    /// candidates; the caller admits selected ones here, never eager closure.
    ///
    /// - ABSENT statement → admit at `rung`, with `premises`, empty stamp;
    /// - OBSERVATION-GROUNDED statement (non-empty stamp — observed, or derived-
    ///   then-observed) → untouched (a pure derivation never overrides ground);
    /// - PURE-DERIVED statement (empty stamp) → truth/rung/premises replaced only
    ///   when the candidate's `expectation()` strictly exceeds the stored one.
    ///
    /// Returns whether the arena changed. Monotone: a pure-derived belief's
    /// expectation only ever increases, bounding closure.
    pub fn admit_derived(
        &mut self,
        stmt: CStmt,
        truth: TruthValue,
        premises: &[u32],
        rung: u32,
    ) -> bool {
        // A CHOICE update fires only on a MEANINGFUL expectation gain, so float
        // churn cannot re-trigger closure passes indefinitely.
        const EPS: f32 = 1e-6;
        match self.index.get(&stmt) {
            Some(&id) => {
                let e = &mut self.entries[id as usize];
                // Any belief carrying observation evidence (non-empty stamp) is
                // GROUND — a pure derivation never overwrites it. This catches
                // both purely-observed (rung 0) and derived-then-observed
                // (rung ≥ 1, stamp unioned by `revise_at`) beliefs; keying on the
                // stamp (evidence provenance), never the rung, is the Codex fix.
                if e.stamp != Stamp::default() {
                    return false;
                }
                if truth.expectation() > e.truth.expectation() + EPS {
                    e.truth = truth;
                    e.rung = rung;
                    e.premises = premises.to_vec();
                    return true;
                }
                false
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
                true
            }
        }
    }

    /// Copula-gated transitive closure (S3): compose `{A cop B, B cop C} ⊢
    /// A cop C` ONLY for `cop.transits()` (Inh, Sim), with NARS deduction truth
    /// ([`TruthValue::deduction`]) per ordered premise pair. Every derivation of
    /// a statement — including multiple paths to the same conclusion — is
    /// resolved by CHOICE on `expectation()` (order-independent). Termination:
    /// each stored expectation only increases and is bounded (deduction
    /// confidence is a product of confidences < 1), so the closure reaches a true
    /// fixed point (`max_passes` is only a backstop).
    pub fn close_transitive(&mut self, max_passes: u32) {
        self.passes = 0;
        self.reached_fixed_point = false;
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
            // Collect EVERY derivation this pass, keyed by the derived statement,
            // keeping the maximum-`expectation()` candidate.
            let mut derived: HashMap<CStmt, (TruthValue, u32, [u32; 2])> = HashMap::new();
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
                    let truth = t1.deduction(&t2);
                    let cand = (truth, r1.max(r2) + 1, [i as u32, j]);
                    match derived.get(&stmt) {
                        Some((best, ..)) if best.expectation() >= truth.expectation() => {}
                        _ => {
                            derived.insert(stmt, cand);
                        }
                    }
                }
            }
            let mut changed = false;
            for (stmt, (truth, rung, premises)) in derived {
                changed |= self.admit_derived(stmt, truth, &premises, rung);
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

    /// Disjoint-stamp revision pools evidence per NARS (synthesis c above both
    /// inputs, `|f₁−f₂|` kept) AND the closure over a cyclic KG still terminates.
    #[test]
    fn revision_disjoint_moves_truth_and_terminates() {
        let mut arena = BeliefArena::new();
        for k in 0..9u16 {
            arena.observe(
                inh(k, k + 1),
                TruthValue::new(0.9, 0.9),
                Stamp::source(k as u32),
            );
        }
        arena.observe(inh(9, 0), TruthValue::new(0.9, 0.9), Stamp::source(9)); // cycle
        let stmt = inh(0, 1);
        let out = arena.observe(stmt, TruthValue::new(0.2, 0.75), Stamp::source(40));
        let ReviseOutcome::Revised {
            synthesis_c, depth, ..
        } = out
        else {
            panic!("disjoint must revise, got {out:?}");
        };
        assert!(
            synthesis_c > 0.9,
            "synthesis c above both inputs: {synthesis_c}"
        );
        assert!((depth - 0.7).abs() < 1e-6, "|0.9−0.2| depth kept");
        arena.close_transitive(1024);
        assert!(arena.reached_fixed_point, "closure terminates (S2)");
        assert_eq!(
            arena.entries().len(),
            100,
            "full cyclic closure of the 10-cycle"
        );
    }

    /// Verbs (Rel) never transit; the same shape under Inh does, with deduction truth.
    #[test]
    fn verbs_do_not_transit() {
        let mut arena = BeliefArena::new();
        let bit = Copula::Rel(77);
        arena.observe(
            CStmt {
                s: 1,
                cop: bit,
                p: 2,
            },
            TruthValue::new(1.0, 0.9),
            Stamp::source(0),
        );
        arena.observe(
            CStmt {
                s: 2,
                cop: bit,
                p: 3,
            },
            TruthValue::new(1.0, 0.9),
            Stamp::source(1),
        );
        arena.close_transitive(16);
        assert_eq!(arena.entries().len(), 2, "dog bit sandwich must NOT derive");

        let mut inh_arena = BeliefArena::new();
        inh_arena.observe(inh(1, 2), TruthValue::new(0.9, 0.8), Stamp::source(0));
        inh_arena.observe(inh(2, 3), TruthValue::new(1.0, 0.95), Stamp::source(1));
        inh_arena.close_transitive(16);
        let d = inh_arena.get(inh(1, 3)).expect("Inh transits");
        assert!((d.truth.frequency - 0.9).abs() < 1e-6);
        assert!((d.truth.confidence - 0.9 * 0.8 * 0.95).abs() < 1e-6);
        assert_eq!(d.rung, 1);
    }

    /// Observation-grounded beliefs are never overwritten by a derivation, keyed
    /// on the stamp (the Codex fix, now one shared path for closure and tactics).
    #[test]
    fn admit_derived_respects_observation_ground() {
        let mut arena = BeliefArena::new();
        let stmt = inh(2, 1);
        arena.observe(stmt, TruthValue::new(0.55, 0.95), Stamp::source(9));
        let before = arena.get(stmt).unwrap().truth;
        let changed = arena.admit_derived(stmt, TruthValue::new(0.99, 0.9), &[0, 0], 1);
        assert!(!changed, "ground belief not overwritten");
        assert_eq!(
            (
                arena.get(stmt).unwrap().truth.frequency,
                arena.get(stmt).unwrap().truth.confidence
            ),
            (before.frequency, before.confidence),
        );
    }

    /// An EMPTY incoming stamp (`Stamp::default()`, the no-source sentinel) must
    /// NOT pool as independent evidence — repeated zero-stamped observations
    /// would otherwise inflate confidence without bound. It routes through CHOICE
    /// (the Codex empty-stamp guard).
    #[test]
    fn empty_incoming_stamp_does_not_pool() {
        let mut arena = BeliefArena::new();
        let stmt = inh(1, 2);
        arena.observe(stmt, TruthValue::new(0.8, 0.5), Stamp::source(3));
        let c0 = arena.get(stmt).unwrap().truth.confidence;
        // Same statement offered with an EMPTY stamp, repeatedly.
        for _ in 0..10 {
            let out = arena.observe(stmt, TruthValue::new(0.8, 0.5), Stamp::default());
            assert!(
                matches!(out, ReviseOutcome::Chosen { .. }),
                "empty stamp → CHOICE, got {out:?}"
            );
        }
        let c1 = arena.get(stmt).unwrap().truth.confidence;
        assert!(
            (c1 - c0).abs() < 1e-6,
            "empty-stamped evidence never pooled: {c0} → {c1}"
        );
        // A REAL disjoint source still pools (the guard is only for empty stamps).
        let out = arena.observe(stmt, TruthValue::new(0.8, 0.5), Stamp::source(7));
        assert!(
            matches!(out, ReviseOutcome::Revised { .. }),
            "real source still revises"
        );
        assert!(
            arena.get(stmt).unwrap().truth.confidence > c0,
            "real evidence pools"
        );
    }
}
