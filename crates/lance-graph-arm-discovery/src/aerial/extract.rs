// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Aerial+ rule extraction — the **codebook-probe** backend (float-free).
//!
//! Aerial+'s reconstruction probe (paper §3.3) marks an antecedent, leaves
//! the rest "unknown", and reads off the consequents the network reconstructs
//! with high probability. Mechanically that is: *which other items are
//! nearest to the antecedent in the learned co-occurrence space?* This backend
//! answers it deterministically with an integer [`CodebookDistance`] oracle
//! (the palette256 table) instead of a float autoencoder:
//!
//! 1. **probe** — for every feature not in the antecedent, take the category
//!    whose codebook distance to the antecedent is minimal; if it is within
//!    `theta`, propose it as a consequent (the τ_c analogue, integer);
//! 2. **confirm** — measure `support`/`confidence` on the actual data as
//!    integer counts and gate by the classical ARM floors (ppm).
//!
//! Double gate, exactly as the paper: the codebook *proposes* (near pairs),
//! the data *confirms* (co-occurrence). An independent feature whose nearest
//! category clears `theta` is dropped at the confidence floor because
//! `P(Y|X) ≈ P(Y)` for independent `Y`. No softmax, no seed, no `f32`.

use crate::aerial::codebook::{antecedent_distance, CodebookDistance};
use crate::encode::Dataset;
use crate::rule::{CandidateRule, Item};

/// Thresholds + bounds for [`extract_rules`]. All integer.
#[derive(Debug, Clone, Copy)]
pub struct ExtractParams {
    /// Maximum codebook distance for a category to be a candidate consequent
    /// (the τ_c analogue, in the oracle's integer units — e.g. palette256
    /// distance). `u32::MAX` = no codebook prune (always take the nearest
    /// category, let the data gates decide). Tune to your distance scale.
    pub theta: u32,
    /// Maximum antecedent size `|X|`.
    pub max_antecedent: usize,
    /// Classical minimum support floor, parts-per-million.
    pub min_support_ppm: u32,
    /// Classical minimum confidence floor, parts-per-million.
    pub min_confidence_ppm: u32,
}

impl Default for ExtractParams {
    fn default() -> Self {
        Self {
            theta: u32::MAX,
            max_antecedent: 2,
            min_support_ppm: 10_000,   // 1%
            min_confidence_ppm: 500_000, // 50%
        }
    }
}

/// Run the codebook probe against a distance oracle and the data, returning
/// deterministically-sorted candidate rules.
#[must_use]
pub fn extract_rules(
    oracle: &dyn CodebookDistance,
    data: &Dataset,
    params: &ExtractParams,
) -> Vec<CandidateRule> {
    let spec = &data.spec;
    let n = data.len() as u32;
    if n == 0 {
        return Vec::new();
    }

    // Frequent single items (support floor) — the apriori prune that bounds
    // the antecedent search, grouped by feature. Integer ppm.
    let mut frequent_by_feature: Vec<Vec<Item>> = vec![Vec::new(); spec.num_features()];
    for (f, bucket) in frequent_by_feature.iter_mut().enumerate() {
        for cat in 0..spec.cardinality(f) {
            let item = Item::new(f as u32, cat);
            let count = data.count_matching(&[item]);
            let support_ppm = ((count as u64 * crate::rule::PPM) / n as u64) as u32;
            if support_ppm >= params.min_support_ppm {
                bucket.push(item);
            }
        }
    }
    let candidate_features: Vec<usize> = (0..spec.num_features())
        .filter(|&f| !frequent_by_feature[f].is_empty())
        .collect();

    let mut rules: Vec<CandidateRule> = Vec::new();
    let max_ant = params.max_antecedent.max(1);
    for size in 1..=max_ant {
        for feature_combo in feature_combinations(&candidate_features, size) {
            for antecedent in item_product(&feature_combo, &frequent_by_feature) {
                probe(oracle, data, &antecedent, &feature_combo, params, &mut rules);
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

/// Probe one antecedent: for each other feature, take the codebook-nearest
/// category; if within `theta`, confirm on data and emit if it clears the
/// ARM floors.
fn probe(
    oracle: &dyn CodebookDistance,
    data: &Dataset,
    antecedent: &[Item],
    antecedent_features: &[usize],
    params: &ExtractParams,
    out: &mut Vec<CandidateRule>,
) {
    let spec = &data.spec;
    let n = data.len() as u32;
    let antecedent_count = data.count_matching(antecedent);
    if antecedent_count == 0 {
        return;
    }

    for g in 0..spec.num_features() {
        if antecedent_features.contains(&g) {
            continue;
        }
        // Codebook-nearest category of feature g to the antecedent.
        let (mut best_cat, mut best_dist) = (0u32, u32::MAX);
        for cat in 0..spec.cardinality(g) {
            let d = antecedent_distance(oracle, antecedent, Item::new(g as u32, cat));
            if d < best_dist {
                best_dist = d;
                best_cat = cat;
            }
        }
        if best_dist > params.theta {
            continue;
        }

        let consequent = vec![Item::new(g as u32, best_cat)];
        let mut both = antecedent.to_vec();
        both.extend_from_slice(&consequent);
        let cooccur = data.count_matching(&both);

        let rule = CandidateRule {
            antecedent: antecedent.to_vec(),
            consequent,
            cooccur,
            antecedent_count,
            window: n,
        };
        if rule.passes(params.min_support_ppm, params.min_confidence_ppm) {
            out.push(rule);
        }
    }
}

/// All size-`k` combinations of the given feature indices (ascending,
/// deterministic).
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
    use crate::aerial::codebook::MatrixDistance;
    use crate::encode::FeatureSpec;

    /// 3 features × 2 cats. Codebook places (f*,c0) near each other and
    /// (f*,c1) near each other; feature 2 is far from features 0/1 (independent
    /// in codebook space). Data plants f0 == f1 with a little noise; f2 random.
    fn fixture(noise_every: usize) -> (Dataset, MatrixDistance) {
        let spec = FeatureSpec::new(vec![2, 2, 2]); // dim 6
        // slots: f0c0=0 f0c1=1 | f1c0=2 f1c1=3 | f2c0=4 f2c1=5
        // near = 1, mid = 5, far = 50.
        let near = 1u32;
        let mid = 5u32;
        let far = 50u32;
        let mut table = vec![0u32; 36];
        let set = |t: &mut Vec<u32>, a: usize, b: usize, v: u32| {
            t[a * 6 + b] = v;
            t[b * 6 + a] = v;
        };
        // f0c0(0) ~ f1c0(2) near; f0c1(1) ~ f1c1(3) near; cross pairs mid.
        set(&mut table, 0, 2, near);
        set(&mut table, 1, 3, near);
        set(&mut table, 0, 3, mid);
        set(&mut table, 1, 2, mid);
        set(&mut table, 0, 1, mid);
        set(&mut table, 2, 3, mid);
        // feature 2 (slots 4,5) far from everything.
        for a in 0..4 {
            set(&mut table, a, 4, far);
            set(&mut table, a, 5, far);
        }
        set(&mut table, 4, 5, mid);

        let rows: Vec<Vec<u32>> = (0..600)
            .map(|i| {
                let a = (i % 2) as u32;
                let b = if noise_every != 0 && i % noise_every == 0 { 1 - a } else { a };
                let c = ((i / 2) % 2) as u32;
                vec![a, b, c]
            })
            .collect();
        (Dataset::new(spec.clone(), rows), MatrixDistance::new(&spec, table))
    }

    #[test]
    fn recovers_planted_rule_and_rejects_independent_feature() {
        let (data, dist) = fixture(20); // 5% label noise on f1
        let params = ExtractParams {
            theta: 2, // codebook prune: only "near" (dist ≤ 2) consequents
            max_antecedent: 1,
            min_support_ppm: 50_000,    // 5%
            min_confidence_ppm: 700_000, // 70%
        };
        let rules = extract_rules(&dist, &data, &params);

        // planted f0=0 ⇒ f1=0 (codebook-near AND data-confirmed)
        let has = rules.iter().any(|r| {
            r.antecedent == vec![Item::new(0, 0)] && r.consequent == vec![Item::new(1, 0)]
        });
        assert!(has, "planted f0=0 ⇒ f1=0 not recovered: {rules:#?}");

        // feature 2 is codebook-far → pruned by theta → never a consequent of f0
        let spurious = rules.iter().any(|r| {
            r.antecedent == vec![Item::new(0, 0)] && r.consequent[0].feature == 2
        });
        assert!(!spurious, "independent feature 2 leaked: {rules:#?}");

        let rule = rules
            .iter()
            .find(|r| r.antecedent == vec![Item::new(0, 0)] && r.consequent == vec![Item::new(1, 0)])
            .unwrap();
        assert!(rule.confidence_ppm() >= 700_000, "confidence below floor");
    }

    #[test]
    fn fully_deterministic_no_seed() {
        let (data, dist) = fixture(20);
        let p = ExtractParams { theta: 2, max_antecedent: 1, min_support_ppm: 50_000, min_confidence_ppm: 700_000 };
        let a = extract_rules(&dist, &data, &p);
        let b = extract_rules(&dist, &data, &p);
        assert_eq!(a, b, "codebook probe is bitwise-deterministic by construction");
    }

    #[test]
    fn theta_prune_blocks_far_consequents() {
        // With a generous theta the far feature is *tested* (then data-gated);
        // with a tight theta it is pruned before the data even sees it.
        let (data, dist) = fixture(0); // no noise: f1==f0 exactly
        let tight = ExtractParams { theta: 2, max_antecedent: 1, min_support_ppm: 50_000, min_confidence_ppm: 600_000 };
        let rules = extract_rules(&dist, &data, &tight);
        assert!(rules.iter().all(|r| r.consequent[0].feature != 2));
    }

    #[test]
    fn feature_combinations_are_correct() {
        assert_eq!(
            feature_combinations(&[0, 1, 2], 2),
            vec![vec![0, 1], vec![0, 2], vec![1, 2]]
        );
    }

    #[test]
    fn empty_dataset_yields_no_rules() {
        let spec = FeatureSpec::new(vec![2, 2]);
        let data = Dataset::new(spec.clone(), Vec::new());
        let dist = MatrixDistance::new(&spec, vec![0u32; 16]);
        assert!(extract_rules(&dist, &data, &ExtractParams::default()).is_empty());
    }
}
