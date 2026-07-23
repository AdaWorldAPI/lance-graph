//! `basin_resonance` — "measure the basins that CLICK most over the standing
//! wave" (operator, D-SCI-INSIGHT reframe). The honest, single-measure answer
//! to "what are the main insights" — where [`super::insights`] glued three
//! separate signals, this reads ONE resonance field off the closed
//! [`BeliefArena`] and lets the two basin kinds fall out of the same measure.
//!
//! ## The standing wave
//! `close_transitive` drives the arena to a fixed point (`reached_fixed_point`).
//! That fixed point IS the standing wave: some concepts accumulate a dense,
//! self-reinforcing set of strong DERIVED beliefs (they stay lit); most do not.
//! A **basin** is the neighbourhood of beliefs touching one concept. We do not
//! ask the LLM which concepts matter — the paper's own belief structure
//! resonates, and we read it.
//!
//! ## The click = Staunen × Wisdom (the canonical magnitude)
//! `.claude/CLAUDE.md`: *"Magnitude = Contradiction depth from Staunen × Wisdom
//! qualia."* Per basin (all rates, never counts — `E-DOOMSCROLL`; all bounded
//! [0,1] — the #832 P2 bounding lesson):
//! - **wisdom** = the derived-conclusion strength: mean [`TruthValue::expectation`]
//!   contributed by the basin's DERIVED (`rung ≥ 1`) beliefs. High = the KG
//!   reasoned its way INTO this concept (logical coherence).
//! - **evidence** = the observed-confidence mass: mean confidence contributed by
//!   the basin's OBSERVED (evidence-stamped) beliefs. High = the concept is
//!   asserted densely (an evidence cluster).
//! - **staunen** = the stakes = wonder: mean distance of the basin's truths from
//!   the neutral 0.5 prior, `2·|expectation − 0.5| = 2·|f − 0.5|·c` — exactly
//!   [`TruthValue::surprise`]`(0.5)` rescaled to [0,1]. This is "the qualia of
//!   the text feeling the stakes of embodied truth": a basin whose truths sit at
//!   indifference (0.5) has no stakes and cannot click.
//! - **resonance** = `staunen × wisdom` ∈ [0,1] — the click. A basin resonates
//!   only when it holds STRONG conclusions (wisdom) that also carry STAKES
//!   (staunen). Trivially-coherent-but-indifferent → no click; high-stakes-but-
//!   unreasoned noise → no click.
//!
//! ## Two basin kinds, one measure
//! [`BasinKind::Coherence`] when `wisdom ≥ evidence` (reasoning-dominant — the
//! logical-coherence basin), [`BasinKind::Evidence`] otherwise (observation-
//! dominant — the evidence cluster). The operator's distinction is not two
//! detectors; it is which pole dominates the SAME basin.
//!
//! ## The null (`E-BASIN-WIDTH`)
//! A size-matched disjoint noise KG composes nothing, so every basin's derived
//! weight is 0 → wisdom 0 → resonance 0: the measure stays silent on noise and
//! lights up only on structure. This is the mandatory falsifier, not a nicety.
//!
//! Cross-ref: [`super::insights`] (the categorized catalog — CoreTheme /
//! Conclusion / Bridge; this module is the resonance RANKER behind it),
//! [`super::dissolution`] (`staunen`/`wisdom` at whole-arena granularity; here
//! they are per-basin), [`super::insight`] (the S10 per-step notion).

use super::belief::{BeliefArena, Copula, Stamp};
use super::truth::TruthValue;
use std::collections::BTreeMap;

/// Which pole dominates a basin — the operator's coherence-vs-evidence split,
/// derived from the SAME resonance measure (not two detectors).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasinKind {
    /// Reasoning-dominant (`wisdom ≥ evidence`): the KG derived its way into this
    /// concept — a logical-coherence basin.
    Coherence,
    /// Observation-dominant (`wisdom < evidence`): the concept is asserted
    /// densely — an evidence cluster basin.
    Evidence,
}

/// One resonant basin: a concept, its click, and the poles that produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Basin {
    /// The concept the basin is anchored at.
    pub concept: u16,
    /// Which pole dominates (coherence vs evidence).
    pub kind: BasinKind,
    /// The click = `staunen × wisdom` ∈ [0,1] — the ranking signal.
    pub resonance: f32,
    /// Stakes / wonder ∈ [0,1]: mean distance of the basin's truths from 0.5.
    pub staunen: f32,
    /// Derived-conclusion strength ∈ [0,1]: mean expectation from derived beliefs.
    pub wisdom: f32,
    /// Observed-confidence mass ∈ [0,1]: mean confidence from observed beliefs.
    pub evidence: f32,
    /// How many DERIVED (`rung ≥ 1`) beliefs touch the concept.
    pub derived: usize,
    /// How many OBSERVED (evidence-stamped) beliefs touch the concept.
    pub observed: usize,
}

/// Configuration for basin resonance ranking.
#[derive(Debug, Clone, Copy)]
pub struct ResonanceConfig {
    /// Minimum beliefs touching a concept for it to count as a basin (a lone
    /// belief has no neighbourhood to resonate).
    pub min_basin: usize,
    /// How many basins to return (top-k by resonance).
    pub top_k: usize,
}

impl Default for ResonanceConfig {
    fn default() -> Self {
        Self {
            min_basin: 2,
            top_k: 16,
        }
    }
}

/// Per-concept accumulator over the basin's transitive-copula beliefs.
#[derive(Default, Clone, Copy)]
struct Acc {
    n: usize,
    n_obs: usize,
    n_der: usize,
    /// Σ confidence over OBSERVED (evidence-stamped) beliefs.
    obs_weight: f32,
    /// Σ expectation over DERIVED (`rung ≥ 1`) beliefs.
    der_weight: f32,
    /// Σ stakes `2·|expectation − 0.5|` over ALL the basin's beliefs.
    stakes: f32,
}

/// Stakes of a single truth: its distance from the neutral 0.5 prior, rescaled
/// to [0,1]. Equals `2·|expectation − 0.5| = 2·surprise(0.5)`.
fn stakes(t: TruthValue) -> f32 {
    (2.0 * (t.expectation() - 0.5).abs()).clamp(0.0, 1.0)
}

/// Rank the concept-KG's basins by RESONANCE (`staunen × wisdom`), the "does it
/// click" measure. Each basin carries both poles and its kind (coherence vs
/// evidence). Returns up to `cfg.top_k`, sorted by resonance DESC (ties →
/// `concept` ASC, deterministic). Silent on noise by construction
/// (`E-BASIN-WIDTH`): with no derivations, every wisdom is 0 and no basin clicks.
#[must_use]
pub fn rank_basins(arena: &BeliefArena, cfg: &ResonanceConfig) -> Vec<Basin> {
    // One pass: fold every transitive-copula belief into the basins of BOTH the
    // concepts it touches (subject and predicate) — a basin is a neighbourhood.
    let mut acc: BTreeMap<u16, Acc> = BTreeMap::new();
    for b in arena.entries() {
        if !b.stmt.cop.transits() && b.stmt.cop != Copula::Impl {
            // Rel (verbs) never join a reasoning basin; Inh/Sim/Impl do.
            continue;
        }
        let is_evidence = b.stamp != Stamp::default();
        let is_derived = b.rung >= 1;
        let conf = b.truth.confidence;
        let exp = b.truth.expectation();
        let st = stakes(b.truth);
        for concept in [b.stmt.s, b.stmt.p] {
            let a = acc.entry(concept).or_default();
            a.n += 1;
            a.stakes += st;
            if is_evidence {
                a.n_obs += 1;
                a.obs_weight += conf;
            }
            if is_derived {
                a.n_der += 1;
                a.der_weight += exp;
            }
        }
    }

    let mut out: Vec<Basin> = Vec::new();
    for (concept, a) in &acc {
        if a.n < cfg.min_basin {
            continue;
        }
        let n = a.n as f32;
        // All three poles are per-basin MEAN rates in [0,1] (E-DOOMSCROLL):
        // beliefs that are not evidence contribute 0 to obs_weight, not derived
        // contribute 0 to der_weight — so the means are over the whole basin.
        let evidence = a.obs_weight / n;
        let wisdom = a.der_weight / n;
        let staunen = a.stakes / n;
        let resonance = (staunen * wisdom).clamp(0.0, 1.0);
        let kind = if wisdom >= evidence {
            BasinKind::Coherence
        } else {
            BasinKind::Evidence
        };
        out.push(Basin {
            concept: *concept,
            kind,
            resonance,
            staunen,
            wisdom,
            evidence,
            derived: a.n_der,
            observed: a.n_obs,
        });
    }

    out.sort_by(|x, y| {
        y.resonance
            .total_cmp(&x.resonance)
            .then_with(|| x.concept.cmp(&y.concept))
    });
    out.truncate(cfg.top_k);
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

    /// The "paper" KG: subjects 1,2,3 all `is_a` shared predicate 100 (an
    /// evidence cluster), plus a chain `1→2→3→4` that closes into derived
    /// conclusions `1→3, 1→4, 2→4` (logical coherence around the chain heads).
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

    /// Every reported pole and the click stay in [0,1] (the bounding discipline).
    #[test]
    fn resonance_and_poles_are_bounded() {
        let a = structured_kg();
        for b in rank_basins(&a, &ResonanceConfig::default()) {
            for (name, v) in [
                ("resonance", b.resonance),
                ("staunen", b.staunen),
                ("wisdom", b.wisdom),
                ("evidence", b.evidence),
            ] {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "{name} must be in [0,1], got {v} at concept {}",
                    b.concept
                );
            }
        }
    }

    /// The mandatory null (`E-BASIN-WIDTH`): a size-matched disjoint noise KG
    /// composes nothing → every wisdom is 0 → NOTHING clicks. The structured KG
    /// must have a strictly positive top resonance the noise KG lacks.
    #[test]
    fn noise_does_not_click() {
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

        let noise_basins = rank_basins(&noise, &ResonanceConfig::default());
        assert!(
            noise_basins.iter().all(|b| b.resonance == 0.0),
            "a disjoint KG composes nothing — no basin may click"
        );

        let structured = rank_basins(&structured_kg(), &ResonanceConfig::default());
        let top = structured
            .first()
            .expect("structured KG has basins meeting min_basin");
        assert!(
            top.resonance > 0.0,
            "the structured KG's top basin must click, got {}",
            top.resonance
        );
    }

    /// The two kinds fall out of the one measure: the shared predicate 100 (only
    /// ever OBSERVED into, never derived) is an Evidence basin; a chain head that
    /// the KG derives strong new conclusions FROM is a Coherence basin.
    #[test]
    fn coherence_and_evidence_kinds_separate() {
        let a = structured_kg();
        let basins = rank_basins(&a, &ResonanceConfig::default());

        let p100 = basins
            .iter()
            .find(|b| b.concept == 100)
            .expect("predicate 100 is a basin (3 subjects assert it)");
        assert_eq!(
            p100.kind,
            BasinKind::Evidence,
            "100 is only observed into (no derivation), so evidence-dominant: \
             wisdom={} evidence={}",
            p100.wisdom,
            p100.evidence
        );
        assert_eq!(p100.derived, 0, "nothing derives 100 as a conclusion");

        assert!(
            basins
                .iter()
                .any(|b| b.kind == BasinKind::Coherence && b.derived > 0),
            "at least one chain-head must surface as a Coherence basin with derivations"
        );
    }

    /// Resonance is a RATE, not a count (`E-DOOMSCROLL`): duplicating the whole
    /// KG onto a disjoint concept range (same shapes, twice the beliefs) does not
    /// inflate the top basin's resonance — it stays a mean.
    #[test]
    fn resonance_is_rate_not_count() {
        let small = rank_basins(&structured_kg(), &ResonanceConfig::default());
        let small_top = small.first().unwrap().resonance;

        // Two disjoint copies of the same structure: identical ratios, 2× beliefs.
        let mut big = BeliefArena::new();
        for base in [0u16, 200u16] {
            let s = |x: u16| x + base;
            let pred = 100 + base;
            big.observe(
                inh(s(1), pred),
                TruthValue::new(0.9, 0.9),
                Stamp::source(base as u32),
            );
            big.observe(
                inh(s(2), pred),
                TruthValue::new(0.9, 0.9),
                Stamp::source(base as u32 + 1),
            );
            big.observe(
                inh(s(3), pred),
                TruthValue::new(0.9, 0.9),
                Stamp::source(base as u32 + 2),
            );
            big.observe(
                inh(s(1), s(2)),
                TruthValue::new(0.9, 0.9),
                Stamp::source(base as u32 + 3),
            );
            big.observe(
                inh(s(2), s(3)),
                TruthValue::new(0.9, 0.9),
                Stamp::source(base as u32 + 4),
            );
            big.observe(
                inh(s(3), s(4)),
                TruthValue::new(0.9, 0.9),
                Stamp::source(base as u32 + 5),
            );
        }
        big.close_transitive(64);
        let big_top = rank_basins(&big, &ResonanceConfig::default())
            .first()
            .unwrap()
            .resonance;

        assert!(
            (small_top - big_top).abs() < 1e-6,
            "resonance is a per-basin rate: small={small_top} big={big_top}"
        );
    }

    /// `top_k` bounds the output and the top basin is stable across the cut.
    #[test]
    fn top_k_bounds_output() {
        let a = structured_kg();
        let cfg = ResonanceConfig {
            top_k: 1,
            ..ResonanceConfig::default()
        };
        let one = rank_basins(&a, &cfg);
        assert!(one.len() <= 1);
        if let Some(top) = one.first() {
            let full = rank_basins(&a, &ResonanceConfig::default());
            assert_eq!(top.resonance, full[0].resonance);
            assert_eq!(top.concept, full[0].concept);
        }
    }
}
