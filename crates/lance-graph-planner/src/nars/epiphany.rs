//! `epiphany` — epiphany attractors ranked by RATE, never count (D-DIA-S9,
//! plan §1 S9, `E-DOOMSCROLL`).
//!
//! Plan §1 S9 (E-DOOMSCROLL, third confirmation): thought promotion/prune and
//! `WisdomMarker` neighborhood bias use normalized RATES, size-normalized —
//! a count-ranked field collapses into its largest basin. This module applies
//! that discipline to EPIPHANY ATTRACTORS: rank subject-basins by epiphany
//! DENSITY (the local closure rate — derived beliefs ÷ total beliefs for that
//! subject), NOT by raw epiphany count. A basin where reasoning closes
//! DENSELY (high rate) is a genuine attractor even if small; the largest
//! basin wins on raw count merely by being large, so count-ranking collapses
//! the field into it. Rate-ranking surfaces the dense basin the
//! count-ranking buries.
//!
//! Grounding note: "epiphany" here is proxied by a landed derivation — a
//! `rung >= 1` belief — at that subject; local closure density = epiphany
//! density, consistent with the shipped global `coherence = derived/total`
//! in [`super::insight`]. This is the density proxy, not a per-step S10
//! insight attribution.

use super::belief::BeliefArena;
use std::collections::BTreeMap;

/// One epiphany attractor: a subject-basin ranked by its epiphany DENSITY.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpiphanyAttractor {
    /// The subject term identifying the basin.
    pub subject: u16,
    /// Epiphanies at this subject = derived (`rung >= 1`) beliefs with this subject.
    pub epiphanies: usize,
    /// Attempts = total beliefs with this subject (the size normalizer).
    pub attempts: usize,
    /// The size-normalized rate `epiphanies / attempts` — the S9 ranking key.
    pub rate: f32,
}

/// Rank subject-basins as epiphany attractors by RATE, never count (S9,
/// E-DOOMSCROLL). For every subject that has at least `min_attempts` beliefs,
/// compute its epiphany density (`derived / total`) and return the attractors
/// sorted by rate DESC, ties broken by subject id ASC (deterministic). A
/// count-ranked field would surface the largest basin; this surfaces the
/// densest, so the field never collapses into its largest basin.
#[must_use]
pub fn rank_epiphany_attractors(
    arena: &BeliefArena,
    min_attempts: usize,
) -> Vec<EpiphanyAttractor> {
    // (subject) -> (epiphanies, attempts)
    let mut basins: BTreeMap<u16, (usize, usize)> = BTreeMap::new();
    for b in arena.entries() {
        let e = basins.entry(b.stmt.s).or_insert((0, 0));
        e.1 += 1;
        if b.rung >= 1 {
            e.0 += 1;
        }
    }
    let mut out: Vec<EpiphanyAttractor> = basins
        .into_iter()
        .filter(|&(_, (_, attempts))| attempts >= min_attempts)
        .map(|(subject, (epiphanies, attempts))| EpiphanyAttractor {
            subject,
            epiphanies,
            attempts,
            rate: epiphanies as f32 / attempts as f32,
        })
        .collect();
    // Rate DESC, then subject ASC for determinism. Use total_cmp on the f32.
    out.sort_by(|a, b| {
        b.rate
            .total_cmp(&a.rate)
            .then_with(|| a.subject.cmp(&b.subject))
    });
    out
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

    /// The S9 / E-DOOMSCROLL falsifier: rate-ranking surfaces the dense small
    /// basin that count-ranking buries under the large sparse basin.
    #[test]
    fn rate_ranking_beats_count_ranking() {
        let mut a = BeliefArena::default();

        // LARGE sparse basin, subject 1: 20 observed + 5 derived.
        // epiphanies 5, attempts 25, rate 0.20.
        for p in 100..120u16 {
            a.observe(
                inh(1, p),
                TruthValue::new(0.9, 0.9),
                Stamp::source(p as u32),
            );
        }
        for p in 200..205u16 {
            assert!(a.admit_derived(inh(1, p), TruthValue::new(0.9, 0.9), &[], 1));
        }

        // SMALL dense basin, subject 2: 1 observed + 3 derived.
        // epiphanies 3, attempts 4, rate 0.75.
        a.observe(inh(2, 300), TruthValue::new(0.9, 0.9), Stamp::source(300));
        for p in 301..304u16 {
            assert!(a.admit_derived(inh(2, p), TruthValue::new(0.9, 0.9), &[], 1));
        }

        let ranked = rank_epiphany_attractors(&a, 2);

        assert_eq!(
            ranked[0].subject, 2,
            "rate-ranking must surface the dense small basin, not the large sparse one"
        );
        assert!(ranked[0].rate > ranked[1].rate);

        // Build the COUNT ranking for contrast.
        let mut by_count = ranked.clone();
        by_count.sort_by(|x, y| {
            y.epiphanies
                .cmp(&x.epiphanies)
                .then(x.subject.cmp(&y.subject))
        });
        assert_eq!(
            by_count[0].subject, 1,
            "count-ranking buries the dense basin under the large one"
        );

        assert_ne!(
            ranked[0].subject, by_count[0].subject,
            "S9 / E-DOOMSCROLL: rate-ranking and count-ranking must diverge — \
             the field must not collapse into its largest basin"
        );
    }

    #[test]
    fn min_attempts_filters_thin_basins() {
        let mut a = BeliefArena::default();

        // Subject 1: 3 beliefs (clears min_attempts = 2).
        a.observe(inh(1, 10), TruthValue::new(0.9, 0.9), Stamp::source(10));
        a.observe(inh(1, 11), TruthValue::new(0.9, 0.9), Stamp::source(11));
        assert!(a.admit_derived(inh(1, 12), TruthValue::new(0.9, 0.9), &[], 1));

        // Subject 2: 1 belief (below min_attempts = 2).
        a.observe(inh(2, 20), TruthValue::new(0.9, 0.9), Stamp::source(20));

        let ranked = rank_epiphany_attractors(&a, 2);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].subject, 1);
    }

    #[test]
    fn ordering_is_deterministic_on_ties() {
        let mut a = BeliefArena::default();

        // Subject 5: 1 derived / 2 total = rate 0.5.
        a.observe(inh(5, 50), TruthValue::new(0.9, 0.9), Stamp::source(50));
        assert!(a.admit_derived(inh(5, 51), TruthValue::new(0.9, 0.9), &[], 1));

        // Subject 3: 1 derived / 2 total = rate 0.5 (same rate).
        a.observe(inh(3, 30), TruthValue::new(0.9, 0.9), Stamp::source(30));
        assert!(a.admit_derived(inh(3, 31), TruthValue::new(0.9, 0.9), &[], 1));

        let ranked = rank_epiphany_attractors(&a, 2);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].subject, 3, "tie broken by subject id ASC");
        assert_eq!(ranked[1].subject, 5);
    }
}
