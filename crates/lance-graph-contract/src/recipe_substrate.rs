// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `recipe_substrate` — wire the REAL tenants into the recipe layer's input.
//!
//! `recipe_claim_audit` proved the 34 [`recipe_kernels`](crate::recipe_kernels)
//! run on a lightweight [`ThoughtCtx`] of hand-set scalar markers — none read a
//! real organ. This module closes that gap: a [`SubstrateView`] carries the three
//! real awareness tenants and **projects** them into the marker basis, so the
//! kernels run on tenant-derived input instead of synthetic constants.
//!
//! # The three tenants
//!
//! * **SPO / AriGraph** — [`SpoFacet`](crate::awareness_facet::SpoFacet) (A1): the
//!   semantic triple + episodic-witness triple grounding *what S/P/O mean here*.
//! * **causalwitness** — [`CausalWitnessFacet`](crate::causal_witness::CausalWitnessFacet)
//!   (A9): the **24 edges** — signed loci placing each awareness dimension in the
//!   `±8` Markov window. The quorum / contradiction / kausal edges are the logical
//!   core.
//! * **qualia** — [`QualiaI4_16D`](crate::qualia::QualiaI4_16D): the affective
//!   texture.
//!
//! # Two separations the operator locked
//!
//! 1. **Qualia is an ADDITIVE factor** — it is *added* to the affective marker
//!    ([`affective_temperature`]), never multiplied in and never used to gate.
//! 2. **Qualia is stakes, not logic** — the *logical* markers (confidence,
//!    surprise, dissonance) are derived ONLY from the SPO tenant + the 24 witness
//!    edges. Qualia describes the **stakes** of a thought — how much it matters,
//!    the felt arousal/tension — never whether a conclusion is true or
//!    contradictory. You do not describe an epiphany's logical quality with a
//!    qualia about how the epiphany feels.
//! 3. **Causality is NOT a qualia** — the cause is the witness [`Locus::Kausal`]
//!    edge (logical, in A9), never a qualia dimension.
//!
//! So: **logic + causality ← SPO + witness edges; stakes ← qualia (additive,
//! temperature only).**

use crate::awareness_facet::SpoFacet;
use crate::causal_witness::{CausalWitnessFacet, Locus};
use crate::qualia::{QualiaI4_16D, QUALIA_I4_LABELS};
use crate::recipe_kernels::ThoughtCtx;

/// A row's real awareness tenants — the substrate a recipe reasons over.
#[derive(Debug, Clone, Copy, Default)]
pub struct SubstrateView {
    /// SPO / AriGraph tenant (A1) — semantic + episodic-witness triple.
    pub spo: SpoFacet,
    /// causalwitness tenant (A9) — the 24 window-loci edges.
    pub witness: CausalWitnessFacet,
    /// qualia tenant — affective texture (additive; temperature only).
    pub qualia: QualiaI4_16D,
}

/// Palette256² grid similarity between two `(basin, identity)` centroids, in
/// `[0.0, 1.0]` (1 = identical). A structural stand-in for the ndarray 256×256
/// CAM-PQ distance table (unavailable in the zero-dep contract) — deterministic
/// L1-in-grid; the exact table binding is an ndarray-side follow-up.
#[inline]
#[must_use]
pub fn pair_similarity(a: (u8, u8), b: (u8, u8)) -> f32 {
    let d = (a.0 as i32 - b.0 as i32).unsigned_abs() + (a.1 as i32 - b.1 as i32).unsigned_abs();
    1.0 - (d as f32 / 510.0)
}

/// Index of a named qualia i4 dimension.
fn q_dim(label: &str) -> usize {
    QUALIA_I4_LABELS
        .iter()
        .position(|&l| l == label)
        .expect("qualia label present")
}

/// The affective temperature marker — qualia as an **additive** factor. Base
/// explore/exploit `0.5` plus the felt arousal & tension, minus groundedness
/// (settled → cooler). Purely affective: it never touches the logical markers.
#[must_use]
pub fn affective_temperature(qualia: QualiaI4_16D) -> f32 {
    // i4 values are −8..+7; normalize each additive contribution to ~[−0.1,0.1].
    let arousal = qualia.get(q_dim("arousal")) as f32 / 70.0;
    let tension = qualia.get(q_dim("tension")) as f32 / 70.0;
    let grounded = qualia.get(q_dim("groundedness")) as f32 / 70.0;
    (0.5 + arousal + tension - grounded).clamp(0.0, 1.0)
}

impl SubstrateView {
    /// Build from the three tenant registers.
    #[inline]
    #[must_use]
    pub const fn new(spo: SpoFacet, witness: CausalWitnessFacet, qualia: QualiaI4_16D) -> Self {
        Self {
            spo,
            witness,
            qualia,
        }
    }

    /// **Logical confidence** — from the SPO tenant + witness edges ONLY (no
    /// qualia). Rises when a **quorum** (agreeing) peer is bound and when the
    /// semantic triple agrees with its episodic witness; the contradiction edge
    /// pulls it down.
    #[must_use]
    pub fn logical_confidence(&self) -> f32 {
        // SPO semantic ↔ episodic-witness agreement (real: does the triple's
        // meaning match what was witnessed).
        let sw = (pair_similarity(self.spo.subject, self.spo.ew_subject)
            + pair_similarity(self.spo.predicate, self.spo.ew_predicate)
            + pair_similarity(self.spo.object, self.spo.ew_object))
            / 3.0;
        let quorum = if self.witness.is_bound(Locus::Quorum) {
            0.2
        } else {
            0.0
        };
        let contra = if self.witness.is_bound(Locus::Contradiction) {
            0.2
        } else {
            0.0
        };
        (0.5 * sw + 0.3 + quorum - contra).clamp(0.0, 1.0)
    }

    /// **Logical surprise (free energy)** — from the witness edges ONLY. A bound
    /// contradiction edge raises surprise; a richly-bound window (many loci
    /// placed) lowers it (the row is well-situated). No qualia.
    #[must_use]
    pub fn logical_surprise(&self) -> f32 {
        let contra = if self.witness.is_bound(Locus::Contradiction) {
            0.4
        } else {
            0.0
        };
        let situated = self.witness.bound_count() as f32 / crate::causal_witness::NAMED_LOCI as f32;
        (0.5 + contra - 0.4 * situated).clamp(0.0, 1.0)
    }

    /// **Logical dissonance** — the contradiction edge is bound AND points to a
    /// DIFFERENT event than the quorum edge (a preserved, real disagreement). No
    /// qualia. Magnitude scales with the offset gap between the two peers.
    #[must_use]
    pub fn logical_dissonance(&self) -> f32 {
        let q = self.witness.quorum();
        let c = self.witness.contradiction();
        if c != 0 && c != q {
            ((q - c).unsigned_abs() as f32 / 15.0).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Candidate scores from the **bound witness edges** — the window's placed
    /// loci as a real vector (their similarities to the quorum peer), the
    /// substrate the parallel/prune/filter recipes actually search. Logical.
    #[must_use]
    pub fn logical_candidates(&self) -> Vec<f32> {
        let q = self.witness.quorum();
        Locus::ALL
            .iter()
            .filter(|&&l| self.witness.is_bound(l))
            .map(|&l| {
                // closeness of this edge's placement to the quorum peer (∈[0,1])
                let off = self.witness.at(l);
                1.0 - ((off - q).unsigned_abs() as f32 / 15.0)
            })
            .collect()
    }

    /// The meaning-depth rung, from the witness **meaning-level** edge magnitude
    /// (1..=9). Logical.
    #[must_use]
    pub fn logical_rung(&self) -> u8 {
        let m = self.witness.at(Locus::MeaningLevel).unsigned_abs();
        (m.min(8) + 1).min(9)
    }

    /// Belief set grounded in the SPO tenant: the semantic triple as one belief
    /// keyed by the subject centroid, with frequency = S↔O relatedness and
    /// confidence = the logical confidence. Logical.
    #[must_use]
    pub fn logical_beliefs(&self) -> Vec<(u32, f32, f32)> {
        let topic = u32::from(self.spo.subject.0) << 8 | u32::from(self.spo.subject.1);
        let freq = pair_similarity(self.spo.subject, self.spo.object);
        vec![(topic, freq, self.logical_confidence())]
    }

    /// Is the SPO tenant present (not the all-zero default)?
    #[inline]
    #[must_use]
    pub fn spo_present(&self) -> bool {
        self.spo != SpoFacet::default()
    }

    /// Is the causalwitness tenant present (at least one bound edge)?
    #[inline]
    #[must_use]
    pub fn witness_present(&self) -> bool {
        self.witness.bound_count() > 0
    }

    /// Project all three tenants into a [`ThoughtCtx`] the 34 kernels consume.
    ///
    /// LOGICAL markers ← SPO + witness edges; the AFFECTIVE `temperature` ←
    /// qualia (additive). A marker a **missing** tenant cannot ground is emitted
    /// as **NaN** (an f32) or **empty** (a Vec) — the honest "undefined input"
    /// that a recipe's NaN-disqualifier gate ([`crate::recipe_dispatch::nan_disqualifier`])
    /// then reads to skip that recipe. `temperature` is always grounded (qualia
    /// defaults are valid stakes).
    #[must_use]
    pub fn project(&self) -> ThoughtCtx {
        let candidates = self.logical_candidates(); // empty if no bound edges
                                                    // free-energy (surprise about truth) needs the witness edges; ungrounded → NaN.
        let free_energy = if self.witness_present() {
            self.logical_surprise()
        } else {
            f32::NAN
        };
        // confidence is grounded by SPO agreement OR an agreement/contradiction edge.
        let conf_grounded = self.spo_present()
            || self.witness.is_bound(Locus::Quorum)
            || self.witness.is_bound(Locus::Contradiction);
        let confidence = if conf_grounded {
            self.logical_confidence()
        } else {
            f32::NAN
        };
        let beliefs = if self.spo_present() {
            self.logical_beliefs()
        } else {
            Vec::new()
        };
        // dissonance is always computable from the witness (0.0 = no contradiction edge).
        let dissonance = self.logical_dissonance();
        // sd = dispersion of the candidate edges (0 if <2 edges).
        let sd = if candidates.len() >= 2 {
            let m = candidates.iter().sum::<f32>() / candidates.len() as f32;
            (candidates.iter().map(|&v| (v - m).powi(2)).sum::<f32>() / candidates.len() as f32)
                .sqrt()
        } else {
            0.25
        };
        ThoughtCtx {
            sd,
            free_energy,
            dissonance,
            temperature: affective_temperature(self.qualia), // qualia: additive, affective
            confidence,
            rung: self.logical_rung(),
            candidates,
            beliefs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qualia_hot() -> QualiaI4_16D {
        QualiaI4_16D::ZERO
            .with(q_dim("arousal"), 6)
            .with(q_dim("tension"), 5)
    }
    fn qualia_cool() -> QualiaI4_16D {
        QualiaI4_16D::ZERO.with(q_dim("groundedness"), 6)
    }

    #[test]
    fn qualia_is_additive_and_affective_only() {
        // Two rows with IDENTICAL logic (same SPO + witness) but different qualia
        // must differ ONLY in temperature — never in any logical marker.
        let spo = SpoFacet::from_register([10, 20, 10, 20, 10, 20, 11, 21, 11, 21, 11, 21]);
        let witness = CausalWitnessFacet::ZERO
            .with(Locus::Quorum, 2)
            .with(Locus::Kausal, -3);
        let hot = SubstrateView::new(spo, witness, qualia_hot()).project();
        let cool = SubstrateView::new(spo, witness, qualia_cool()).project();

        assert!(
            hot.temperature > cool.temperature,
            "qualia adds to temperature"
        );
        // logic identical — qualia touched nothing logical
        assert_eq!(
            hot.confidence, cool.confidence,
            "qualia must not touch confidence"
        );
        assert_eq!(
            hot.free_energy, cool.free_energy,
            "qualia must not touch surprise"
        );
        assert_eq!(
            hot.dissonance, cool.dissonance,
            "qualia must not touch dissonance"
        );
        assert_eq!(
            hot.candidates, cool.candidates,
            "qualia must not touch candidates"
        );
        assert_eq!(hot.rung, cool.rung, "qualia must not touch rung");
    }

    #[test]
    fn contradiction_edge_drives_logical_dissonance_and_surprise() {
        let spo = SpoFacet::default();
        let peaceful = SubstrateView::new(
            spo,
            CausalWitnessFacet::ZERO.with(Locus::Quorum, 2),
            QualiaI4_16D::ZERO,
        );
        let conflicted = SubstrateView::new(
            spo,
            CausalWitnessFacet::ZERO
                .with(Locus::Quorum, 2)
                .with(Locus::Contradiction, -4), // a preserved dissenting peer
            QualiaI4_16D::ZERO,
        );
        assert_eq!(peaceful.logical_dissonance(), 0.0);
        assert!(
            conflicted.logical_dissonance() > 0.0,
            "contradiction edge → dissonance"
        );
        assert!(
            conflicted.logical_surprise() > peaceful.logical_surprise(),
            "contradiction edge → higher surprise"
        );
    }

    #[test]
    fn quorum_edge_raises_logical_confidence() {
        let spo = SpoFacet::default();
        let alone = SubstrateView::new(spo, CausalWitnessFacet::ZERO, QualiaI4_16D::ZERO);
        let corroborated = SubstrateView::new(
            spo,
            CausalWitnessFacet::ZERO.with(Locus::Quorum, 3),
            QualiaI4_16D::ZERO,
        );
        assert!(
            corroborated.logical_confidence() > alone.logical_confidence(),
            "an agreeing peer edge raises confidence"
        );
    }

    #[test]
    fn candidates_come_from_bound_edges() {
        let spo = SpoFacet::default();
        let w = CausalWitnessFacet::ZERO
            .with(Locus::Quorum, 2)
            .with(Locus::Kausal, -3)
            .with(Locus::SMeaning, 1);
        let v = SubstrateView::new(spo, w, QualiaI4_16D::ZERO);
        assert_eq!(v.logical_candidates().len(), 3, "one score per bound edge");
    }
}
