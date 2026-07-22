//! `reason` — the derivation-pointer fabric: **the graph reasoning about
//! itself as the SAME pointer geometry, one level up over the triple stream.**
//!
//! A parse tree is a pointer fabric over the *word* stream (each word points to
//! its attachment site; `E-GRAMMAR-TREE-IS-POINTER-FABRIC-1`). A **proof tree is
//! the identical fabric over the *triple* stream**: each derived triple points to
//! the PREMISE triples it was composed from. The tree never materializes as an
//! object — the pointers ARE the tree (Manning & Carpenter p.153; Moore §7's
//! 2-field back-pointer reconstructs every parse). This module is the
//! self-reasoning keystone `D-SRS-1` of `self-reasoning-substrate-v1`.
//!
//! ## The one sound rule (anti-runaway)
//!
//! Inference is **per-predicate transitive composition ONLY**: from `(A, p, B)`
//! and `(B, p, C)` sharing the SAME predicate `p`, derive `(A, p, C)`. Composing
//! edges with DIFFERENT predicates is the `TD-INFER-DEDUCTIONS-RELATION-BLIND`
//! runaway (`E-SELF-DIRECTED-GRAPH-1`) — forbidden here. Keeping `p` constant is
//! the sound `is_a`-style rule.
//!
//! ## Why the gate holds by construction (Tarski stratification)
//!
//! Base (observed) triples sit at **rung 0**; a derived triple is stamped
//! `max(premise rungs) + 1`. Two invariants then fall out *by construction* and
//! are re-checked explicitly (never assumed) by [`DerivationArena::gate`]:
//!
//! - **Resolvability** — every premise pointer indexes an EARLIER arena entry
//!   (composition only ever cites already-present entries), so 0 pointers dangle.
//! - **Acyclicity** — a derived triple's rung strictly exceeds every premise's
//!   rung, so the derivation graph cannot contain a cycle (not even an
//!   equal-rung one — that is exactly why the stamp is `+1`, not `≥`).
//!
//! Termination is guaranteed independently by **dedup**: the derivable set is a
//! subset of `entities × {p} × entities`, which is finite, and each distinct
//! triple is admitted at most once — so the fixed-point closure halts even when
//! the *base* relation is cyclic.

use crate::spo::Spo;
use std::collections::{HashMap, HashSet};

/// One entry in the derivation arena: a triple, its Tarski rung, and pointers to
/// the premises it was composed from (empty for an observed base triple).
///
/// The `premises` are arena indices — the pointer fabric. They are ALWAYS
/// strictly less than this entry's own index and cite strictly-lower rungs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Derivation {
    /// The triple this entry asserts.
    pub triple: Spo,
    /// Tarski rung: 0 for observed base facts, `max(premise rungs) + 1` for
    /// derived facts.
    pub rung: u32,
    /// Arena indices of the premise entries (empty ⇔ observed base fact).
    pub premises: Vec<usize>,
}

/// The result of checking the `D-SRS-1` gate over an arena — the three metrics
/// the pre-registered gate pins, computed by verification (not assumed).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateReport {
    /// Number of observed base triples (rung 0).
    pub base: usize,
    /// Number of derived triples (rung ≥ 1).
    pub derived: usize,
    /// Percent of derived triples whose every premise pointer resolves to an
    /// earlier, existing arena entry. PASS requires exactly `100.0`.
    pub resolvability_pct: f64,
    /// Whether every premise cites a strictly-lower rung (no equal/higher edge,
    /// no cycle). PASS requires `true`.
    pub acyclic: bool,
    /// Whether the fixed-point closure reached a fixed point within the pass cap
    /// (a pass that added nothing). PASS requires `true`.
    pub terminated: bool,
    /// Closure passes actually run (for reporting; not a gate value).
    pub passes: u32,
}

impl GateReport {
    /// The `D-SRS-1` PASS condition: 100% resolvable AND acyclic AND terminated.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.resolvability_pct == 100.0 && self.acyclic && self.terminated
    }
}

/// A defensive cap on closure passes. The closure terminates by dedup long
/// before this on any real KG; hitting it is reported as `terminated = false`
/// (a KILL), never silently ignored.
const MAX_PASSES: u32 = 1_024;

/// The append-only derivation arena: base triples first, then the transitive
/// closure, every derived entry carrying its premise pointers.
#[derive(Debug, Clone)]
pub struct DerivationArena {
    entries: Vec<Derivation>,
    base_len: usize,
    passes: u32,
    reached_fixed_point: bool,
}

impl DerivationArena {
    /// Build the arena from observed base triples and close it under
    /// per-predicate transitive composition.
    ///
    /// Base triples are deduplicated (first occurrence kept) and stamped rung 0;
    /// the closure then repeatedly composes `(A,p,B) + (B,p,C) → (A,p,C)` for a
    /// shared predicate `p`, stamping each new triple `max(premise rungs)+1` and
    /// recording its two premise pointers, until a pass adds nothing.
    #[must_use]
    pub fn derive_transitive(base: &[Spo]) -> Self {
        // Base: dedup, rung 0, no premises.
        let mut seen: HashSet<Spo> = HashSet::new();
        let mut entries: Vec<Derivation> = Vec::new();
        for &t in base {
            if seen.insert(t) {
                entries.push(Derivation {
                    triple: t,
                    rung: 0,
                    premises: Vec::new(),
                });
            }
        }
        let base_len = entries.len();

        // Closure. Each pass rebuilds the pivot index `(subject, predicate) →
        // [arena idx]`, then composes every `(A,p,B)` with every `(B,p,C)`.
        let mut passes = 0u32;
        let mut reached_fixed_point = false;
        while passes < MAX_PASSES {
            passes += 1;

            // Pivot index over the CURRENT arena.
            let mut by_sp: HashMap<(u16, u16), Vec<usize>> = HashMap::new();
            for (idx, d) in entries.iter().enumerate() {
                by_sp
                    .entry((d.triple.subject, d.triple.predicate))
                    .or_default()
                    .push(idx);
            }

            // Collect new triples this pass (do not mutate `entries` while
            // iterating it).
            let mut additions: Vec<Derivation> = Vec::new();
            for i in 0..entries.len() {
                let (a, p, b) = {
                    let d = &entries[i];
                    (d.triple.subject, d.triple.predicate, d.triple.object)
                };
                // Find (B, p, C): same predicate, subject == this object.
                let Some(js) = by_sp.get(&(b, p)) else {
                    continue;
                };
                for &j in js {
                    let c = entries[j].triple.object;
                    let composed = Spo::new(a, p, c);
                    // Dedup against base + prior derivations + this pass.
                    if seen.contains(&composed) {
                        continue;
                    }
                    if additions.iter().any(|d| d.triple == composed) {
                        continue;
                    }
                    let rung = entries[i].rung.max(entries[j].rung) + 1;
                    additions.push(Derivation {
                        triple: composed,
                        rung,
                        premises: vec![i, j],
                    });
                }
            }

            if additions.is_empty() {
                reached_fixed_point = true;
                break;
            }
            for d in additions {
                seen.insert(d.triple);
                entries.push(d);
            }
        }

        Self {
            entries,
            base_len,
            passes,
            reached_fixed_point,
        }
    }

    /// All arena entries (base then derived), in append order.
    #[must_use]
    pub fn entries(&self) -> &[Derivation] {
        &self.entries
    }

    /// The derived entries only (rung ≥ 1) — the proof-fabric slice.
    #[must_use]
    pub fn derived(&self) -> &[Derivation] {
        &self.entries[self.base_len..]
    }

    /// Resolve a premise pointer back to its triple (the round-trip the gate
    /// requires). `None` if the pointer dangles.
    #[must_use]
    pub fn resolve(&self, premise: usize) -> Option<Spo> {
        self.entries.get(premise).map(|d| d.triple)
    }

    /// Verify the `D-SRS-1` gate over this arena — computes the three pinned
    /// metrics by explicit checking, never by assumption.
    #[must_use]
    pub fn gate(&self) -> GateReport {
        // Resolvability: every premise indexes an EARLIER, existing entry.
        let derived = self.entries.len() - self.base_len;
        let mut resolvable = 0usize;
        for (idx, d) in self.entries.iter().enumerate().skip(self.base_len) {
            if d.premises
                .iter()
                .all(|&pmz| pmz < idx && pmz < self.entries.len())
            {
                resolvable += 1;
            }
        }
        let resolvability_pct = if derived == 0 {
            100.0
        } else {
            100.0 * resolvable as f64 / derived as f64
        };

        // Acyclicity: every premise cites a strictly-lower rung.
        let acyclic = self.entries.iter().all(|d| {
            d.premises
                .iter()
                .all(|&pmz| self.entries[pmz].rung < d.rung)
        });

        GateReport {
            base: self.base_len,
            derived,
            resolvability_pct,
            acyclic,
            terminated: self.reached_fixed_point,
            passes: self.passes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3-link chain `1→2→3→4` under one predicate closes to its full
    /// transitive set, every derived triple carries resolvable pointers, and the
    /// rungs stratify (`1→3`,`2→4` at rung 1; `1→4` at rung 2).
    #[test]
    fn transitive_chain_closes_with_stratified_pointers() {
        let p = 7;
        let base = [Spo::new(1, p, 2), Spo::new(2, p, 3), Spo::new(3, p, 4)];
        let arena = DerivationArena::derive_transitive(&base);
        let g = arena.gate();

        assert!(g.passed(), "gate must pass: {g:?}");
        assert_eq!(g.base, 3);
        // derived = {1→3, 2→4, 1→4}
        assert_eq!(g.derived, 3);
        assert_eq!(g.resolvability_pct, 100.0);
        assert!(g.acyclic);
        assert!(g.terminated);

        // Every premise round-trips to a real earlier triple.
        for (idx, d) in arena.entries().iter().enumerate() {
            for &pmz in &d.premises {
                assert!(pmz < idx, "premise must be earlier");
                assert!(arena.resolve(pmz).is_some(), "premise must resolve");
                assert!(
                    arena.entries()[pmz].rung < d.rung,
                    "premise must be strictly-lower rung"
                );
            }
        }

        // The transitive `1→4` sits at rung 2 (composed from a rung-1 premise).
        let one_four = arena
            .derived()
            .iter()
            .find(|d| d.triple == Spo::new(1, p, 4))
            .expect("1→4 derived");
        assert_eq!(one_four.rung, 2);
        assert_eq!(one_four.premises.len(), 2);
    }

    /// Different predicates never compose — the anti-runaway rule.
    #[test]
    fn different_predicate_does_not_compose() {
        let base = [Spo::new(1, 7, 2), Spo::new(2, 9, 3)];
        let arena = DerivationArena::derive_transitive(&base);
        let g = arena.gate();
        assert_eq!(g.derived, 0, "cross-predicate composition is forbidden");
        assert!(g.passed());
    }

    /// A cyclic BASE relation still terminates (dedup bounds it) and the
    /// DERIVATION graph stays acyclic (rung stamping) — the two-cycles case
    /// D-SRS-1's KILL clause guards against.
    #[test]
    fn cyclic_base_terminates_and_derivation_stays_acyclic() {
        let p = 3;
        let base = [Spo::new(1, p, 2), Spo::new(2, p, 1)];
        let arena = DerivationArena::derive_transitive(&base);
        let g = arena.gate();

        assert!(g.terminated, "dedup must halt the cyclic base");
        assert!(
            g.acyclic,
            "derivation graph must be acyclic despite base cycle"
        );
        assert_eq!(g.resolvability_pct, 100.0);
        assert!(g.passed());
        // Closure adds exactly the two self-edges {1→1, 2→2}.
        let mut derived: Vec<_> = arena.derived().iter().map(|d| d.triple).collect();
        derived.sort_by_key(|t| t.pack());
        assert_eq!(derived, vec![Spo::new(1, p, 1), Spo::new(2, p, 2)]);
    }

    /// A lone self-loop `(1,p,1)` composes only with itself → already present →
    /// no explosion.
    #[test]
    fn self_loop_base_does_not_explode() {
        let arena = DerivationArena::derive_transitive(&[Spo::new(1, 5, 1)]);
        let g = arena.gate();
        assert_eq!(g.derived, 0);
        assert!(g.terminated);
        assert!(g.passed());
    }

    /// Duplicate base triples are deduplicated to one rung-0 entry.
    #[test]
    fn duplicate_base_is_deduplicated() {
        let t = Spo::new(1, 2, 3);
        let arena = DerivationArena::derive_transitive(&[t, t, t]);
        assert_eq!(arena.gate().base, 1);
    }

    /// No shared pivot ⇒ no derivation, gate still passes vacuously.
    #[test]
    fn disjoint_edges_derive_nothing() {
        let base = [Spo::new(1, 7, 2), Spo::new(3, 7, 4)];
        let arena = DerivationArena::derive_transitive(&base);
        assert_eq!(arena.gate().derived, 0);
        assert!(arena.gate().passed());
    }
}
