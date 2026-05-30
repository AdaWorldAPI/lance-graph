// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Aerial+ **Algorithm 1** — rule extraction by reconstruction probing
//! (paper §3.3).
//!
//! The idea, verbatim: to test antecedent `X`, build an input where `X`'s
//! categories are marked at probability 1 and **every other feature is left
//! uniform** ("unknown"); forward-pass the trained autoencoder; then
//!
//! 1. **antecedent test** — every marked category must still come back with
//!    probability `≥ τ_a` (`ant_similarity`); a self-inconsistent antecedent
//!    the network cannot hold is rejected, and
//! 2. **consequent test** — for every *other* feature, if its top category
//!    `Y` reconstructs with probability `≥ τ_c` (`cons_similarity`), the rule
//!    `X → Y` is proposed.
//!
//! Each proposed rule's `support`/`confidence` are then measured **on the
//! actual data** (`encode::Dataset`) and gated by the classical ARM floors.
//! This is the double gate that makes Aerial+ robust: the network *proposes*
//! a rule from its learned conditional structure; the data *confirms* it.
//! An independent feature whose marginal happens to clear `τ_c` is dropped
//! at the confidence floor because `P(Y|X) ≈ P(Y)` for independent `Y`.

use crate::aerial::autoencoder::AerialAutoencoder;
use crate::encode::Dataset;
use crate::rule::{CandidateRule, Item};

/// Thresholds + bounds for [`extract_rules`].
#[derive(Debug, Clone, Copy)]
pub struct ExtractParams {
    /// `τ_a` — minimum reconstructed probability of each marked antecedent
    /// category for the antecedent to be considered coherent.
    pub ant_similarity: f32,
    /// `τ_c` — minimum reconstructed probability of a consequent category.
    pub cons_similarity: f32,
    /// Maximum antecedent size `|X|` (paper default 2; pair-stats trunk caps
    /// here, the Aerial fan-in can go higher).
    pub max_antecedent: usize,
    /// Classical minimum support floor, measured on the data.
    pub min_support: f32,
    /// Classical minimum confidence floor, measured on the data.
    pub min_confidence: f32,
}

impl Default for ExtractParams {
    fn default() -> Self {
        Self {
            ant_similarity: 0.5,
            cons_similarity: 0.6,
            max_antecedent: 2,
            min_support: 0.01,
            min_confidence: 0.5,
        }
    }
}

/// Run Algorithm 1 against a trained autoencoder and the data it was trained
/// on. Returns deterministically-sorted candidate rules.
#[must_use]
pub fn extract_rules(
    ae: &AerialAutoencoder,
    data: &Dataset,
    params: &ExtractParams,
) -> Vec<CandidateRule> {
    let spec = &data.spec;
    let n = data.len() as u32;
    if n == 0 {
        return Vec::new();
    }

    // Frequent single items (support floor) — the apriori-style prune that
    // bounds the antecedent search, grouped by feature.
    let mut frequent_by_feature: Vec<Vec<Item>> = vec![Vec::new(); spec.num_features()];
    for (f, bucket) in frequent_by_feature.iter_mut().enumerate() {
        for cat in 0..spec.cardinality(f) {
            let item = Item::new(f as u32, cat);
            if data.support(&[item]) >= params.min_support {
                bucket.push(item);
            }
        }
    }
    let candidate_features: Vec<usize> = (0..spec.num_features())
        .filter(|&f| !frequent_by_feature[f].is_empty())
        .collect();

    // The neutral probe: every block uniform (= "unknown").
    let neutral = neutral_probe(spec);

    let mut rules: Vec<CandidateRule> = Vec::new();

    let max_ant = params.max_antecedent.max(1);
    for size in 1..=max_ant {
        for feature_combo in feature_combinations(&candidate_features, size) {
            // Cartesian product of the frequent items of each chosen feature.
            for antecedent in item_product(&feature_combo, &frequent_by_feature) {
                probe_antecedent(ae, spec, &neutral, &antecedent, &feature_combo, data, params, &mut rules);
            }
        }
    }

    rules.sort_by(|a, b| {
        a.antecedent
            .cmp(&b.antecedent)
            .then_with(|| a.consequent.cmp(&b.consequent))
    });
    rules.dedup_by(|a, b| a.antecedent == b.antecedent && a.consequent == b.consequent);
    rules
}

/// Probe one antecedent and append every confirmed rule it yields.
#[allow(clippy::too_many_arguments)]
fn probe_antecedent(
    ae: &AerialAutoencoder,
    spec: &crate::encode::FeatureSpec,
    neutral: &[f32],
    antecedent: &[Item],
    antecedent_features: &[usize],
    data: &Dataset,
    params: &ExtractParams,
    out: &mut Vec<CandidateRule>,
) {
    // Build the probe: neutral, then mark the antecedent blocks one-hot.
    let mut probe = neutral.to_vec();
    for &it in antecedent {
        let (s, e) = spec.block(it.feature as usize);
        probe[s..e].fill(0.0);
        probe[spec.slot(it)] = 1.0;
    }

    let p = ae.reconstruct(&probe);

    // Antecedent test: every marked category must survive at ≥ τ_a.
    for &it in antecedent {
        if p[spec.slot(it)] < params.ant_similarity {
            return;
        }
    }

    // Consequent test: for each feature NOT in the antecedent, take its top
    // category; if it clears τ_c, propose X → (g, w) and confirm on data.
    let n = data.len() as u32;
    for g in 0..spec.num_features() {
        if antecedent_features.contains(&g) {
            continue;
        }
        let (s, e) = spec.block(g);
        let (mut best_cat, mut best_p) = (0usize, f32::NEG_INFINITY);
        for (cat, &prob) in p[s..e].iter().enumerate() {
            if prob > best_p {
                best_p = prob;
                best_cat = cat;
            }
        }
        if best_p < params.cons_similarity {
            continue;
        }
        let consequent = vec![Item::new(g as u32, best_cat as u32)];
        let support = {
            let mut both = antecedent.to_vec();
            both.extend_from_slice(&consequent);
            data.support(&both)
        };
        let confidence = data.confidence(antecedent, &consequent);
        let rule = CandidateRule {
            antecedent: antecedent.to_vec(),
            consequent,
            support,
            confidence,
            n,
        };
        if rule.passes(params.min_support, params.min_confidence) {
            out.push(rule);
        }
    }
}

/// The neutral probe vector: every feature block set to its uniform
/// distribution `1/cardinality`.
fn neutral_probe(spec: &crate::encode::FeatureSpec) -> Vec<f32> {
    let mut v = vec![0.0f32; spec.dim()];
    for f in 0..spec.num_features() {
        let (s, e) = spec.block(f);
        let u = 1.0 / (e - s) as f32;
        v[s..e].fill(u);
    }
    v
}

/// All size-`k` combinations of the given feature indices (order preserved,
/// ascending), produced deterministically.
fn feature_combinations(features: &[usize], k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut combo = Vec::with_capacity(k);
    fn recurse(
        features: &[usize],
        start: usize,
        k: usize,
        combo: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if combo.len() == k {
            out.push(combo.clone());
            return;
        }
        for i in start..features.len() {
            combo.push(features[i]);
            recurse(features, i + 1, k, combo, out);
            combo.pop();
        }
    }
    recurse(features, 0, k, &mut combo, &mut out);
    out
}

/// Cartesian product of the frequent items of each feature in `combo`,
/// yielding one antecedent (sorted by feature) per element.
fn item_product(combo: &[usize], frequent_by_feature: &[Vec<Item>]) -> Vec<Vec<Item>> {
    let mut result: Vec<Vec<Item>> = vec![Vec::new()];
    for &f in combo {
        let items = &frequent_by_feature[f];
        let mut next = Vec::with_capacity(result.len() * items.len());
        for prefix in &result {
            for &item in items {
                let mut ext = prefix.clone();
                ext.push(item);
                next.push(ext);
            }
        }
        result = next;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aerial::rng::Rng;
    use crate::encode::FeatureSpec;

    /// Build a dataset with a planted rule (f0=0 ⇒ f1=0, f0=1 ⇒ f1=1) plus an
    /// independent feature f2, with a little label noise on f1.
    fn planted(n: usize, seed: u64, noise: f32) -> Dataset {
        let spec = FeatureSpec::new(vec![2, 2, 2]);
        let mut rng = Rng::new(seed);
        let rows = (0..n)
            .map(|_| {
                let a = (rng.next_u64() % 2) as u32;
                let b = if rng.bernoulli(noise) { 1 - a } else { a };
                let c = (rng.next_u64() % 2) as u32;
                vec![a, b, c]
            })
            .collect();
        Dataset::new(spec, rows)
    }

    #[test]
    fn feature_combinations_are_correct() {
        assert_eq!(
            feature_combinations(&[0, 1, 2], 2),
            vec![vec![0, 1], vec![0, 2], vec![1, 2]]
        );
        assert_eq!(feature_combinations(&[0, 1, 2], 1), vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn item_product_enumerates_category_assignments() {
        let freq = vec![
            vec![Item::new(0, 0), Item::new(0, 1)],
            vec![Item::new(1, 0)],
        ];
        let prod = item_product(&[0, 1], &freq);
        assert_eq!(prod.len(), 2);
        assert!(prod.contains(&vec![Item::new(0, 0), Item::new(1, 0)]));
        assert!(prod.contains(&vec![Item::new(0, 1), Item::new(1, 0)]));
    }

    #[test]
    fn neutral_probe_is_uniform_per_block() {
        let spec = FeatureSpec::new(vec![2, 4]);
        let v = neutral_probe(&spec);
        assert!((v[0] - 0.5).abs() < 1e-6);
        assert!((v[2] - 0.25).abs() < 1e-6);
        assert!((v[0..2].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((v[2..6].iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recovers_planted_rule_and_rejects_independent_feature() {
        let data = planted(500, 11, 0.05); // 5% label noise on f1
        let mut rng = Rng::new(42);
        let mut ae = AerialAutoencoder::new(&data.spec, 4, &mut rng);
        ae.train(&data, 600, 0.1, 0.3, &mut rng);

        let params = ExtractParams {
            ant_similarity: 0.5,
            cons_similarity: 0.6,
            max_antecedent: 1,
            min_support: 0.05,
            min_confidence: 0.7,
        };
        let rules = extract_rules(&ae, &data, &params);

        // The planted dependency f0 → f1 must surface (both directions of the
        // category mapping are valid rules).
        let has_f0_implies_f1 = rules.iter().any(|r| {
            r.antecedent == vec![Item::new(0, 0)] && r.consequent == vec![Item::new(1, 0)]
        });
        assert!(has_f0_implies_f1, "planted f0=0 ⇒ f1=0 not recovered: {rules:#?}");

        // No rule should have feature 2 (independent) as a consequent of f0.
        let spurious = rules.iter().any(|r| {
            r.antecedent == vec![Item::new(0, 0)] && r.consequent[0].feature == 2
        });
        assert!(!spurious, "independent feature 2 leaked as a consequent: {rules:#?}");

        // Recovered rule confidence reflects the data (≈0.95 with 5% noise).
        let rule = rules
            .iter()
            .find(|r| r.antecedent == vec![Item::new(0, 0)] && r.consequent == vec![Item::new(1, 0)])
            .unwrap();
        assert!(rule.confidence >= 0.7, "confidence {} below floor", rule.confidence);
    }

    #[test]
    fn empty_dataset_yields_no_rules() {
        let spec = FeatureSpec::new(vec![2, 2]);
        let data = Dataset::new(spec, Vec::new());
        let mut rng = Rng::new(1);
        let ae = AerialAutoencoder::new(&data.spec, 3, &mut rng);
        assert!(extract_rules(&ae, &data, &ExtractParams::default()).is_empty());
    }
}
