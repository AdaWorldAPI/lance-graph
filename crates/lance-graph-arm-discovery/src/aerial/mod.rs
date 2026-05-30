// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Aerial+ rule discovery — the **deterministic codebook-probe** backend.
//!
//! Transcode of Karabulut, Groth, Degeler, *Neurosymbolic Association Rule
//! Mining from Tabular Data* (arXiv 2504.19354v1), with the paper's float
//! autoencoder **replaced by an integer codebook distance oracle**. The
//! reconstruction probe is a nearest-neighbour query; this substrate answers
//! it exactly via the palette256 distance table (ρ=0.9973 vs cosine), so no
//! `f32`, no SGD, no seed enter the proposer.
//!
//! - [`codebook::CodebookDistance`] — the injected integer similarity oracle
//!   (palette256 / BLASGraph splat / HDR-popcount; `MatrixDistance` in tests).
//! - [`extract::extract_rules`] — Algorithm 1 as a codebook top-k + data
//!   confirmation.
//! - [`AerialProposer`] — wires them into the [`crate::rule::Proposer`]
//!   contract.
//!
//! # No determinism firewall needed
//!
//! The autoencoder transcode was a *nondeterministic fan-in* that had to stay
//! behind the ratification gate and out of the compile path. The codebook
//! probe is **bitwise-deterministic by construction** — same data + same
//! oracle + same `theta` ⇒ identical rules on every target. It can sit in the
//! deterministic trunk beside pair-stats (D-ARM-3); the ratification gate
//! still governs *promotion to the SPO store*, but no longer because of any
//! nondeterminism here.

pub mod codebook;
pub mod extract;
pub mod ontology;

pub use codebook::{antecedent_distance, CodebookDistance, MatrixDistance, TopKDistance};
pub use extract::{extract_rules, ExtractParams};
pub use ontology::{DolceCategory, OntologyProjector};

use crate::encode::Dataset;
use crate::rule::{CandidateRule, Proposer};

/// Configuration for an [`AerialProposer`] — the extraction thresholds.
/// (No autoencoder hyper-parameters and no seed: the backend is deterministic.)
pub type AerialParams = ExtractParams;

/// An Aerial+ codebook-probe proposer over one data window + a distance oracle.
#[derive(Debug)]
pub struct AerialProposer<D: CodebookDistance> {
    data: Dataset,
    oracle: D,
    params: AerialParams,
    drained: bool,
}

impl<D: CodebookDistance> AerialProposer<D> {
    /// Build a proposer over a window and an injected codebook oracle.
    /// No training step — the oracle is the frozen co-occurrence model.
    #[must_use]
    pub fn new(data: Dataset, oracle: D, params: AerialParams) -> Self {
        Self {
            data,
            oracle,
            params,
            drained: false,
        }
    }

    /// Run the codebook probe once, regardless of the drained flag.
    #[must_use]
    pub fn mine(&self) -> Vec<CandidateRule> {
        extract_rules(&self.oracle, &self.data, &self.params)
    }
}

impl<D: CodebookDistance> Proposer for AerialProposer<D> {
    fn next_batch(&mut self) -> Vec<CandidateRule> {
        if self.drained {
            return Vec::new();
        }
        self.drained = true;
        self.mine()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::FeatureSpec;
    use crate::ndjson::to_ndjson;
    use crate::rule::Item;
    use crate::translator::{CandidateTriple, DebugProjector, NARS_PERSONALITY_K};

    fn fixture() -> (Dataset, MatrixDistance) {
        let spec = FeatureSpec::new(vec![2, 2, 2]);
        // Initialise every off-diagonal pair to "far" (50), diagonal to 0, then
        // mark only the near pairs — so an unset cell can never read as nearest.
        let mut table = vec![50u32; 36];
        for d in 0..6 {
            table[d * 6 + d] = 0;
        }
        let set = |a: usize, b: usize, v: u32, t: &mut Vec<u32>| {
            t[a * 6 + b] = v;
            t[b * 6 + a] = v;
        };
        // f0c0(0)~f1c0(2) near; f0c1(1)~f1c1(3) near. f2 (4,5) stays far.
        set(0, 2, 1, &mut table);
        set(1, 3, 1, &mut table);
        let rows = (0..600)
            .map(|i| {
                let a = (i % 2) as u32;
                let c = ((i / 2) % 2) as u32;
                vec![a, a, c]
            })
            .collect();
        (Dataset::new(spec.clone(), rows), MatrixDistance::new(&spec, table))
    }

    #[test]
    fn proposer_yields_then_drains() {
        let (data, dist) = fixture();
        let params = AerialParams {
            theta: 2,
            max_antecedent: 1,
            min_support_ppm: 50_000,
            min_confidence_ppm: 700_000,
        };
        let mut p = AerialProposer::new(data, dist, params);
        assert!(!p.next_batch().is_empty());
        assert!(p.next_batch().is_empty());
    }

    #[test]
    fn mined_rules_serialise_to_spo_ndjson() {
        let (data, dist) = fixture();
        let params = AerialParams { theta: 2, max_antecedent: 1, min_support_ppm: 50_000, min_confidence_ppm: 700_000 };
        let p = AerialProposer::new(data, dist, params);
        let rules = p.mine();
        assert!(!rules.is_empty());
        let proj = DebugProjector::default();
        let triples: Vec<CandidateTriple> = rules
            .iter()
            .map(|r| CandidateTriple::from_rule(r, &proj, NARS_PERSONALITY_K))
            .collect();
        let ndjson = to_ndjson(&triples);
        for line in ndjson.lines() {
            assert!(line.starts_with("{\"s\":\""), "line: {line}");
            assert!(line.contains("\"p\":\"implies\""));
        }
    }

    #[test]
    fn reproducible_no_seed_needed() {
        let (d1, dist1) = fixture();
        let (d2, dist2) = fixture();
        let p = AerialParams { theta: 2, max_antecedent: 1, min_support_ppm: 50_000, min_confidence_ppm: 700_000 };
        let r1 = AerialProposer::new(d1, dist1, p).mine();
        let r2 = AerialProposer::new(d2, dist2, p).mine();
        assert_eq!(r1, r2);
        let _ = Item::new(0, 0);
    }
}
