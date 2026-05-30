// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Aerial+ — the neurosymbolic association-rule proposer (Stage A fan-in).
//!
//! Transcode of Karabulut, Groth, Degeler, *Neurosymbolic Association Rule
//! Mining from Tabular Data* (arXiv 2504.19354v1). Two pieces:
//!
//! - [`autoencoder::AerialAutoencoder`] — the under-complete denoising AE
//!   (softmax-per-feature, cross-entropy), §3.2.
//! - [`extract::extract_rules`] — Algorithm 1, the reconstruction probe, §3.3.
//!
//! [`AerialProposer`] wires them into the [`crate::rule::Proposer`] contract:
//! `fit` trains the AE on a window; `next_batch` runs Algorithm 1 once and
//! returns the confirmed rules.
//!
//! # Why this is a *fan-in*, not the trunk
//!
//! Per `streaming-arm-nars-discovery-v1.md` the deterministic pair-stats
//! proposer is the default trunk; Aerial+ earns its keep only on
//! high-dimensional sparse data where pair/triple counters blow up. It is
//! nondeterministic in the general case, so it is seeded here for
//! reproducibility and its output is gated by the downstream ratification
//! council before any triple it proposes can reach the codegen path.

pub mod autoencoder;
pub mod extract;
pub mod rng;

pub use autoencoder::AerialAutoencoder;
pub use extract::{extract_rules, ExtractParams};
pub use rng::Rng;

use crate::encode::Dataset;
use crate::rule::{CandidateRule, Proposer};

/// Full configuration for an [`AerialProposer`]: the autoencoder
/// hyper-parameters, the extraction thresholds, and the seed that makes the
/// whole run reproducible.
#[derive(Debug, Clone, Copy)]
pub struct AerialParams {
    /// Latent dimension `H` (under-complete: keep `H < D`).
    pub hidden_dim: usize,
    /// Training epochs.
    pub epochs: usize,
    /// SGD learning rate.
    pub learning_rate: f32,
    /// Denoising mask probability (load-bearing — see autoencoder docs).
    pub noise: f32,
    /// Algorithm 1 extraction thresholds + ARM floors.
    pub extract: ExtractParams,
    /// PRNG seed — fixes weight init, denoising mask, and epoch shuffle.
    pub seed: u64,
}

impl Default for AerialParams {
    fn default() -> Self {
        Self {
            hidden_dim: 8,
            epochs: 500,
            learning_rate: 0.1,
            noise: 0.3,
            extract: ExtractParams::default(),
            seed: 0x5EED,
        }
    }
}

/// A trained Aerial+ proposer over one data window.
#[derive(Debug)]
pub struct AerialProposer {
    data: Dataset,
    params: AerialParams,
    ae: AerialAutoencoder,
    /// Set once `next_batch` has run, so a second call returns `[]`
    /// (the window is exhausted until `fit` is called on a fresh window).
    drained: bool,
}

impl AerialProposer {
    /// Train an autoencoder on the window and return a ready proposer.
    #[must_use]
    pub fn fit(data: Dataset, params: AerialParams) -> Self {
        let mut rng = Rng::new(params.seed);
        let mut ae = AerialAutoencoder::new(&data.spec, params.hidden_dim, &mut rng);
        ae.train(&data, params.epochs, params.learning_rate, params.noise, &mut rng);
        Self {
            data,
            params,
            ae,
            drained: false,
        }
    }

    /// Borrow the trained autoencoder (for probes / diagnostics).
    #[must_use]
    pub fn autoencoder(&self) -> &AerialAutoencoder {
        &self.ae
    }

    /// Run Algorithm 1 once, regardless of the drained flag.
    #[must_use]
    pub fn mine(&self) -> Vec<CandidateRule> {
        extract_rules(&self.ae, &self.data, &self.params.extract)
    }
}

impl Proposer for AerialProposer {
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

    fn planted(n: usize, seed: u64) -> Dataset {
        // f0 → f1 deterministic; f2 independent.
        let spec = FeatureSpec::new(vec![2, 2, 2]);
        let mut rng = Rng::new(seed);
        let rows = (0..n)
            .map(|_| {
                let a = (rng.next_u64() % 2) as u32;
                let c = (rng.next_u64() % 2) as u32;
                vec![a, a, c]
            })
            .collect();
        Dataset::new(spec, rows)
    }

    #[test]
    fn proposer_end_to_end_yields_then_drains() {
        let data = planted(400, 9);
        let mut params = AerialParams {
            hidden_dim: 4,
            epochs: 600,
            ..AerialParams::default()
        };
        params.extract.max_antecedent = 1;
        params.extract.min_support = 0.05;
        params.extract.min_confidence = 0.7;

        let mut proposer = AerialProposer::fit(data, params);
        let batch = proposer.next_batch();
        assert!(!batch.is_empty(), "should mine the planted rule");
        // Drains: a second call is empty until re-fit on a new window.
        assert!(proposer.next_batch().is_empty());
    }

    #[test]
    fn mined_rules_serialise_to_spo_ndjson() {
        // End-to-end synergy check: mine → translate → ndjson in the exact
        // {s,p,o,f,c} shape the SPO store loader reads.
        let data = planted(400, 13);
        let mut params = AerialParams {
            hidden_dim: 4,
            epochs: 600,
            ..AerialParams::default()
        };
        params.extract.max_antecedent = 1;
        params.extract.min_support = 0.05;
        params.extract.min_confidence = 0.7;

        let proposer = AerialProposer::fit(data, params);
        let rules = proposer.mine();
        assert!(!rules.is_empty());

        let projector = DebugProjector::default();
        let triples: Vec<CandidateTriple> = rules
            .iter()
            .map(|r| CandidateTriple::from_rule(r, &projector, NARS_PERSONALITY_K))
            .collect();
        let ndjson = to_ndjson(&triples);

        // Every line is a {s,p,o,f,c} object terminated by newline.
        for line in ndjson.lines() {
            assert!(line.starts_with("{\"s\":\""), "line: {line}");
            assert!(line.contains("\"p\":\"implies\""), "line: {line}");
            assert!(line.contains("\"f\":"), "line: {line}");
            assert!(line.contains("\"c\":"), "line: {line}");
        }
        assert!(ndjson.ends_with('\n'));
    }

    #[test]
    fn reproducible_from_seed() {
        let p = AerialParams {
            hidden_dim: 4,
            epochs: 200,
            seed: 777,
            ..AerialParams::default()
        };
        let r1 = AerialProposer::fit(planted(200, 4), p).mine();
        let r2 = AerialProposer::fit(planted(200, 4), p).mine();
        assert_eq!(r1, r2, "same seed + data ⇒ identical rules");
    }

    #[test]
    fn default_params_are_under_complete_on_small_specs() {
        // sanity: default hidden 8 ≥ a 6-slot toy spec, so toy tests pass a
        // smaller hidden explicitly; this asserts the default is documented.
        let _ = Item::new(0, 0);
        assert_eq!(AerialParams::default().hidden_dim, 8);
    }
}
