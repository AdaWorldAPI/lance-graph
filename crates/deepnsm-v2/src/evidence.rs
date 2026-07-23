//! `evidence` — the D-SRS-3b **evidence-composite** basin uncertainty
//! (the operator-corrected instrument).
//!
//! D-SRS-3's Cam96 code-spread width was GEOMETRY with no evidence semantics —
//! and it fell to the label-shuffle null ("bullshit in, bullshit out",
//! operator ruling 2026-07-23). The corrected instrument composes the
//! EVIDENCE-BEARING signals the substrate already carries — exactly the ones
//! D-SRS-4 proved read faithfully:
//!
//! - `u_conf` — **NARS Truth × frequency**: `1 − mean(nᵢ/(nᵢ+1))` over the
//!   basin's distinct beliefs. Singleton-heavy neighborhoods = thin evidence.
//! - `u_contra` — contradiction density: share of predicates under the subject
//!   carrying > 1 distinct object.
//! - `u_rung` — **rung-ladder** derived share: fraction of the subject's
//!   triples in the derivation arena at rung ≥ 1 (inferred, not observed).
//!
//! `U = (u_conf + u_contra + u_rung)/3` (registered equal weights, never
//! tuned). MUL mapping: `competence = 1 − U`, `curiosity = U`.
//!
//! ## The forward gate (G-SRS3b-1, active-inference reading)
//!
//! Reported uncertainty must predict where SURPRISE actually arrives:
//! `U` computed on the FIRST half of the stream is Spearman-correlated with
//! second-half **novelty** (never-before-seen `(p,o)` occurrence share),
//! against a deterministic size-preserving null (belief records redealt
//! across basins, per-basin distinct-belief count preserved) and a
//! frequency-only activity baseline. Verdicts are REPORTED, never panicked.

use crate::basin::spearman;
use lance_graph_contract::kanban::KanbanColumn;
use lance_graph_contract::mul::GateDecision;

/// One distinct belief under a subject: `(predicate, object, occurrence_count)`.
pub type BeliefRecord = (u16, u16, usize);
/// A basin's evidence as `(subject, its distinct belief records)`.
pub type BasinBeliefs = (u16, Vec<BeliefRecord>);

/// One basin's evidence-composite self-measurement (first-half evidence only).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvidenceBasin {
    /// The subject entity anchoring this basin.
    pub subject: u16,
    /// NARS×frequency thinness: `1 − mean(nᵢ/(nᵢ+1))` over distinct beliefs.
    pub u_conf: f32,
    /// Contradiction density: predicates with > 1 distinct object / predicates.
    pub u_contra: f32,
    /// Rung-ladder derived share: rung ≥ 1 triples / all triples for subject.
    pub u_rung: f32,
    /// Distinct `(p, o)` beliefs (first half).
    pub beliefs: usize,
    /// Total `(p, o)` occurrences (first half) — the activity baseline input.
    pub occurrences: usize,
}

impl EvidenceBasin {
    /// The registered composite `U = (u_conf + u_contra + u_rung)/3` ∈ [0,1].
    #[must_use]
    pub fn uncertainty(&self) -> f32 {
        (self.u_conf + self.u_contra + self.u_rung) / 3.0
    }

    /// MUL competence = `1 − U` (the `CompassNeedles` self-measurement).
    #[must_use]
    pub fn competence(&self) -> f32 {
        (1.0 - self.uncertainty()).clamp(0.0, 1.0)
    }

    /// MUL curiosity = `U` — the graph is drawn to its thin-evidence regions.
    #[must_use]
    pub fn curiosity(&self) -> f32 {
        self.uncertainty().clamp(0.0, 1.0)
    }

    /// The **kanban gate** this basin's evidence drives — the MUL→phase seam
    /// ([`KanbanColumn::advance_on_gate`]). This is what makes the instrument
    /// KANBANSTEP-DRIVEN rather than a printed report: the composite maps to a
    /// [`GateDecision`], and the gate advances (or vetoes) the basin's
    /// exploration lifecycle on the Rubicon DAG. A-priori tertile mapping —
    /// registered as CONVENTION (thirds), never tuned to data:
    /// - `U < 1/3` → [`GateDecision::Flow`] — evidence thick, proceed
    ///   (`Planning → CognitiveWork`: explore here next).
    /// - `U ≥ 2/3` → [`GateDecision::Block`] — evidence too thin/contradicted
    ///   to act on (the Libet veto: `Planning → Prune` until new evidence).
    /// - else → [`GateDecision::Hold`] — stay in place, gather evidence.
    #[must_use]
    pub fn gate(&self) -> GateDecision {
        let u = self.uncertainty();
        if u < 1.0 / 3.0 {
            GateDecision::Flow
        } else if u >= 2.0 / 3.0 {
            GateDecision::Block {
                reason: format!(
                    "evidence too thin: U={u:.2} (conf {:.2}, contra {:.2}, rung {:.2})",
                    self.u_conf, self.u_contra, self.u_rung
                ),
            }
        } else {
            GateDecision::Hold {
                reason: format!("gathering evidence: U={u:.2}"),
            }
        }
    }

    /// Drive one kanban lifecycle step from `col` under this basin's gate —
    /// the `E-KANBANSTEP-IS-THE-TRIGGER` shape: the step (not the report) is
    /// what advances exploration. `None` = Hold in place (re-evaluate next
    /// cycle) or no legal edge.
    #[must_use]
    pub fn advance(&self, col: KanbanColumn) -> Option<KanbanColumn> {
        col.advance_on_gate(&self.gate())
    }
}

/// Build one basin's evidence composite from its DISTINCT beliefs
/// (`(predicate, object, count)`, count ≥ 1) and its rung-ladder derived share
/// (computed by the caller from the derivation arena). Returns `None` for an
/// empty belief set.
#[must_use]
pub fn evidence_basin(
    subject: u16,
    beliefs: &[BeliefRecord],
    derived_share: f32,
) -> Option<EvidenceBasin> {
    if beliefs.is_empty() {
        return None;
    }
    // u_conf: mean NARS confidence n/(n+1) over distinct beliefs, inverted.
    let mean_c = beliefs
        .iter()
        .map(|&(_, _, n)| n as f32 / (n as f32 + 1.0))
        .sum::<f32>()
        / beliefs.len() as f32;
    // u_contra: predicates with >1 distinct object. Beliefs are distinct
    // (p,o), so counting per-p distinct objects = counting entries per p.
    let mut preds: Vec<u16> = beliefs.iter().map(|&(p, _, _)| p).collect();
    preds.sort_unstable();
    let n_preds = {
        let mut u = preds.clone();
        u.dedup();
        u.len()
    };
    let multi = {
        let mut m = 0usize;
        let mut i = 0;
        while i < preds.len() {
            let mut j = i + 1;
            while j < preds.len() && preds[j] == preds[i] {
                j += 1;
            }
            if j - i > 1 {
                m += 1;
            }
            i = j;
        }
        m
    };
    let u_contra = if n_preds == 0 {
        0.0
    } else {
        multi as f32 / n_preds as f32
    };
    Some(EvidenceBasin {
        subject,
        u_conf: 1.0 - mean_c,
        u_contra,
        u_rung: derived_share.clamp(0.0, 1.0),
        beliefs: beliefs.len(),
        occurrences: beliefs.iter().map(|&(_, _, n)| n).sum(),
    })
}

/// The rung ladder's **open questions** for a subject and their forward
/// **yield** (G-SRS3b-2): a first-half derived-but-not-observed triple is the
/// graph PREDICTING `(A,p,C)` by transitivity — the open question is "does the
/// text later confirm it?". `yield = |OpenQ ∩ second-half-base| / |OpenQ|`.
///
/// `open_q` = the subject's first-half inferences (rung ≥ 1, NOT already
/// observed), as `(predicate, object)` pairs. `second_half_base` = the
/// distinct `(predicate, object)` base facts observed under the subject in the
/// second half. Returns `None` if there are no open questions (nothing to
/// resolve). This is the operator-corrected forward target — question
/// RESOLUTION, not raw novelty (the doom-scroll trap).
#[must_use]
pub fn open_question_yield(open_q: &[(u16, u16)], second_half_base: &[(u16, u16)]) -> Option<f32> {
    if open_q.is_empty() {
        return None;
    }
    let observed: std::collections::HashSet<(u16, u16)> =
        second_half_base.iter().copied().collect();
    let resolved = open_q.iter().filter(|po| observed.contains(po)).count();
    Some(resolved as f32 / open_q.len() as f32)
}

/// Second-half **novelty**: the share of `(p, o)` occurrences (with repeats)
/// that were never seen in the subject's first-half distinct-belief set.
/// Independent ground truth for the forward gate — computed by direct
/// membership, no window API, no codebook.
#[must_use]
pub fn novelty_rate(first_half_beliefs: &[(u16, u16)], second_half_occ: &[(u16, u16)]) -> f32 {
    if second_half_occ.is_empty() {
        return 0.0;
    }
    let seen: std::collections::HashSet<(u16, u16)> = first_half_beliefs.iter().copied().collect();
    let novel = second_half_occ
        .iter()
        .filter(|po| !seen.contains(po))
        .count();
    novel as f32 / second_half_occ.len() as f32
}

/// Deterministic SplitMix64 sequence (no rng/clock) — the same generator the
/// D-SRS-3 null used.
fn splitmix_seq(len: usize) -> impl FnMut() -> u64 {
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15 ^ (len as u64);
    move || {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The size-preserving evidence null: pool every basin's belief records
/// `(p, o, n)` (caller supplies basins in DETERMINISTIC sorted order),
/// Fisher-Yates shuffle the pool (SplitMix64), redeal preserving each basin's
/// DISTINCT-BELIEF COUNT. Preserves the n-artifact (how many beliefs a basin
/// has); destroys WHICH evidence it holds.
#[must_use]
pub fn shuffle_beliefs_null(basins: &[BasinBeliefs]) -> Vec<BasinBeliefs> {
    let mut pool: Vec<(u16, u16, usize)> = basins.iter().flat_map(|(_, b)| b.clone()).collect();
    let mut next = splitmix_seq(pool.len());
    for i in (1..pool.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        pool.swap(i, j);
    }
    let mut off = 0usize;
    basins
        .iter()
        .map(|(s, b)| {
            let g = pool[off..off + b.len()].to_vec();
            off += b.len();
            (*s, g)
        })
        .collect()
}

/// Shuffle the per-basin `u_rung` scalars across basins (deterministic,
/// distinct seed lane) — the rung half of the evidence null (rung shares are
/// per-basin scalars, not belief records, so they are permuted directly).
#[must_use]
pub fn shuffle_rungs_null(rungs: &[f32]) -> Vec<f32> {
    let mut v = rungs.to_vec();
    let mut next = splitmix_seq(v.len().wrapping_mul(3) + 1);
    for i in (1..v.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

/// **Spearman partial correlation** ρ(x, y | z): rank-transform all three,
/// linearly residualize `rank(x)` and `rank(y)` on `rank(z)`, and correlate the
/// residuals. Removes the confounding covariate `z` (here: basin size). `0.0`
/// for < 3 points or a degenerate (zero-variance) side. The G-SRS3b-3 primitive
/// — "does the composite predict yield AFTER size is partialled out?"
#[must_use]
pub fn partial_spearman(x: &[f32], y: &[f32], z: &[f32]) -> f32 {
    if x.len() != y.len() || x.len() != z.len() || x.len() < 3 {
        return 0.0;
    }
    let (rx, ry, rz) = (ranks_f(x), ranks_f(y), ranks_f(z));
    let ex = residualize(&rx, &rz);
    let ey = residualize(&ry, &rz);
    pearson_local(&ex, &ey)
}

/// Average ranks (1-based, ties averaged) — local copy so this module is
/// self-contained for the partial-correlation path.
fn ranks_f(v: &[f32]) -> Vec<f32> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1 + j) as f32) / 2.0;
        for &k in &idx[i..j] {
            r[k] = avg;
        }
        i = j;
    }
    r
}

/// Residuals of `a` after linear regression on `b` (`a − (α + β·b)`).
fn residualize(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let mut sbb = 0.0f32;
    let mut sab = 0.0f32;
    for (&av, &bv) in a.iter().zip(b) {
        sbb += (bv - mb) * (bv - mb);
        sab += (av - ma) * (bv - mb);
    }
    let beta = if sbb <= 0.0 { 0.0 } else { sab / sbb };
    let alpha = ma - beta * mb;
    a.iter()
        .zip(b)
        .map(|(&av, &bv)| av - (alpha + beta * bv))
        .collect()
}

/// Pearson correlation (local, for the residual path). `0.0` on zero variance.
fn pearson_local(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    let (mx, my) = (x.iter().sum::<f32>() / n, y.iter().sum::<f32>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for (&a, &b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    let d = (sxx * syy).sqrt();
    if d <= 0.0 {
        0.0
    } else {
        sxy / d
    }
}

/// The G-SRS3b-1 forward-gate report: real vs null vs activity-baseline
/// Spearman against second-half novelty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardGateReport {
    /// Eligible basins correlated.
    pub basins: usize,
    /// Spearman ρ(U_real, novelty).
    pub real_rho: f32,
    /// Spearman ρ(U_null, novelty) — the size-preserving evidence null.
    pub null_rho: f32,
    /// Spearman ρ(first-half total occurrences, novelty) — frequency-only
    /// baseline (REPORTED, not gated).
    pub baseline_rho: f32,
}

impl ForwardGateReport {
    /// Registered separation: real − null.
    #[must_use]
    pub fn separation(&self) -> f32 {
        self.real_rho - self.null_rho
    }
    /// PASS: real ρ ≥ 0.25 AND separation ≥ 0.15 (and enough basins).
    #[must_use]
    pub fn passed(&self) -> bool {
        self.basins >= 3 && self.real_rho >= 0.25 && self.separation() >= 0.15
    }
    /// KILL: separation ≤ 0.05 — no signal beyond structure-free chance.
    #[must_use]
    pub fn killed(&self) -> bool {
        self.basins >= 3 && self.separation() <= 0.05
    }
}

/// Compute the forward gate from aligned per-basin vectors.
#[must_use]
pub fn forward_gate(
    u_real: &[f32],
    u_null: &[f32],
    activity: &[f32],
    novelty: &[f32],
) -> ForwardGateReport {
    ForwardGateReport {
        basins: novelty.len(),
        real_rho: spearman(u_real, novelty),
        null_rho: spearman(u_null, novelty),
        baseline_rho: spearman(activity, novelty),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Singleton-heavy evidence reads thinner (higher u_conf) than repeated
    /// evidence — the NARS×frequency component.
    #[test]
    fn thin_evidence_reads_more_uncertain() {
        let thin = evidence_basin(1, &[(5, 10, 1), (6, 11, 1), (7, 12, 1)], 0.0).unwrap();
        let thick = evidence_basin(2, &[(5, 10, 9), (6, 11, 9), (7, 12, 9)], 0.0).unwrap();
        assert!(thin.u_conf > thick.u_conf);
        assert!(thin.uncertainty() > thick.uncertainty());
    }

    /// Contradiction density counts multi-object predicates over distinct
    /// beliefs; single-object predicates contribute zero.
    #[test]
    fn contradiction_density_counts_multi_object_predicates() {
        let clash = evidence_basin(1, &[(5, 10, 2), (5, 11, 1), (6, 12, 3)], 0.0).unwrap();
        assert!((clash.u_contra - 0.5).abs() < 1e-6); // p=5 clashes, p=6 clean
        let clean = evidence_basin(1, &[(5, 10, 2), (6, 12, 3)], 0.0).unwrap();
        assert_eq!(clean.u_contra, 0.0);
    }

    /// Composite and MUL mappings are bounded and complementary; rung share
    /// clamps into [0,1].
    #[test]
    fn composite_is_bounded_and_complementary() {
        let b = evidence_basin(1, &[(5, 10, 1), (5, 11, 1)], 2.0).unwrap();
        assert_eq!(b.u_rung, 1.0, "derived_share clamps");
        let u = b.uncertainty();
        assert!((0.0..=1.0).contains(&u));
        assert!((b.competence() + b.curiosity() - 1.0).abs() < 1e-6);
        assert!(evidence_basin(1, &[], 0.0).is_none());
    }

    /// Novelty rate is the never-seen occurrence share, by direct membership.
    #[test]
    fn novelty_rate_counts_unseen_occurrences() {
        let first = [(5u16, 10u16), (6, 11)];
        let second = [(5u16, 10u16), (7, 12), (7, 12), (6, 11)];
        // novel: the two (7,12) occurrences of 4 total.
        assert!((novelty_rate(&first, &second) - 0.5).abs() < 1e-6);
        assert_eq!(novelty_rate(&first, &[]), 0.0);
    }

    /// The null redeal preserves each basin's distinct-belief count and is a
    /// permutation of the pooled records; it is deterministic.
    #[test]
    fn belief_null_preserves_counts_and_is_deterministic() {
        let basins = vec![
            (1u16, vec![(5u16, 10u16, 3usize), (6, 11, 1)]),
            (2u16, vec![(7u16, 12u16, 2usize)]),
            (3u16, vec![(8u16, 13u16, 5usize), (9, 14, 1), (9, 15, 1)]),
        ];
        let null = shuffle_beliefs_null(&basins);
        for ((_, real), (_, n)) in basins.iter().zip(&null) {
            assert_eq!(real.len(), n.len(), "per-basin count preserved");
        }
        let mut all_real: Vec<_> = basins.iter().flat_map(|(_, b)| b.clone()).collect();
        let mut all_null: Vec<_> = null.iter().flat_map(|(_, b)| b.clone()).collect();
        all_real.sort_unstable();
        all_null.sort_unstable();
        assert_eq!(all_real, all_null, "null is a permutation of the pool");
        assert_eq!(null, shuffle_beliefs_null(&basins), "deterministic");
        assert_eq!(
            shuffle_rungs_null(&[0.1, 0.2, 0.3]),
            shuffle_rungs_null(&[0.1, 0.2, 0.3]),
            "rung null deterministic"
        );
    }

    /// Open-question yield: the share of first-half inferences the second half
    /// confirms (the operator-corrected forward target — resolution, not novelty).
    #[test]
    fn open_question_yield_counts_confirmed_inferences() {
        let open_q = [(7u16, 4u16), (7, 9), (7, 12), (8, 3)]; // 4 inferences
        let sh_base = [(7u16, 4u16), (7, 12), (5, 5)]; // confirms 2 of them
        assert!((open_question_yield(&open_q, &sh_base).unwrap() - 0.5).abs() < 1e-6);
        assert!(open_question_yield(&[], &sh_base).is_none()); // no questions
        assert_eq!(open_question_yield(&open_q, &[]), Some(0.0)); // none confirmed
    }

    /// The kanban seam: the evidence composite DRIVES the Rubicon lifecycle
    /// via GateDecision → advance_on_gate. Thick evidence Flows Planning →
    /// CognitiveWork; thin evidence Blocks Planning → Prune (Libet veto);
    /// mid evidence Holds in place. The step is the trigger, not the report.
    #[test]
    fn evidence_gate_drives_kanban_lifecycle() {
        // Thick: repeated evidence, no contradiction, fully observed.
        let thick = evidence_basin(1, &[(5, 10, 9), (6, 11, 9), (7, 12, 9)], 0.0).unwrap();
        assert_eq!(thick.gate().to_disc(), 0, "thick evidence → Flow");
        assert_eq!(
            thick.advance(KanbanColumn::Planning),
            Some(KanbanColumn::CognitiveWork),
            "Flow advances Planning → CognitiveWork (explore here)"
        );
        // Thin: all singletons, every predicate contradicted, fully derived.
        let thin = evidence_basin(2, &[(5, 10, 1), (5, 11, 1)], 1.0).unwrap();
        assert_eq!(thin.gate().to_disc(), 2, "thin evidence → Block");
        assert_eq!(
            thin.advance(KanbanColumn::Planning),
            Some(KanbanColumn::Prune),
            "Block vetoes Planning → Prune (Libet free-won't)"
        );
        // Mid: singletons (u_conf 0.5) + one contradicted predicate (u_contra
        // 0.5) + moderate rung 0.3 ⇒ U ≈ 0.43 → Hold → stay in place.
        let mid = evidence_basin(3, &[(5, 10, 1), (5, 11, 1), (6, 12, 1)], 0.3).unwrap();
        assert_eq!(mid.gate().to_disc(), 1, "mid evidence → Hold");
        assert_eq!(mid.advance(KanbanColumn::Planning), None, "Hold stays");
        // Block mid-CognitiveWork has no veto edge (contract DAG) → None.
        assert_eq!(thin.advance(KanbanColumn::CognitiveWork), None);
    }

    /// Partial correlation removes a shared confound: when x and y are
    /// correlated ONLY through z, the partial ρ collapses to ≈0; when x has a
    /// genuine z-independent tie to y, it survives.
    #[test]
    fn partial_spearman_removes_shared_confound() {
        // x and y both = z + tiny independent noise ⇒ correlation is all z.
        let z = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x: Vec<f32> = z.iter().map(|&v| v + 0.01 * (v * 1.7).sin()).collect();
        let y: Vec<f32> = z.iter().map(|&v| v + 0.01 * (v * 2.3).cos()).collect();
        let raw = spearman(&x, &y);
        let partial = partial_spearman(&x, &y, &z);
        assert!(raw > 0.9, "raw corr high (both track z): {raw}");
        assert!(
            partial.abs() < raw,
            "partial removes the shared z: {partial}"
        );
        // Degenerate guards.
        assert_eq!(partial_spearman(&[1.0, 2.0], &[1.0, 2.0], &[1.0, 2.0]), 0.0);
    }

    /// Forward gate mechanism: when U tracks novelty monotonically and the
    /// null does not, the report passes; a no-separation report kills.
    #[test]
    fn forward_gate_mechanism() {
        let novelty = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let u_real = [0.15f32, 0.22, 0.33, 0.41, 0.52, 0.63]; // monotone with novelty
        let u_null = [0.5f32, 0.1, 0.4, 0.2, 0.6, 0.3]; // scrambled
        let activity = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let g = forward_gate(&u_real, &u_null, &activity, &novelty);
        assert!(g.passed(), "monotone U must pass: {g:?}");
        assert!(!g.killed());
        let dead = forward_gate(&u_null, &u_null, &activity, &novelty);
        assert!(dead.killed(), "zero separation must kill: {dead:?}");
    }
}
