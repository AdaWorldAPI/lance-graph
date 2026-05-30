// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One-hot categorical encoding — the input layer of the Aerial+ transcode.
//!
//! Aerial+ (paper §3.1) one-hot encodes each row into a concatenation of
//! per-feature blocks: feature `i` with `k_i` categories occupies a block of
//! `k_i` slots, exactly one of which is `1.0` (the observed category). The
//! full input vector has dimension `D = Σ k_i`.
//!
//! Numerical columns must be discretised into bins *before* they reach this
//! layer (the paper bins numerics; [`FeatureSpec::bin`] is the helper).

use crate::rule::Item;

/// The schema for a one-hot encoding: the ordered list of features and how
/// many categories each has.
///
/// Block offsets are derived once and cached so encode/decode is a pointer
/// add, not a scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSpec {
    /// Number of categories per feature, in feature order.
    cardinalities: Vec<u32>,
    /// Prefix-sum offsets: `offsets[i]` is the first input slot of feature `i`.
    /// Length is `cardinalities.len() + 1`; the last entry is the total dim.
    offsets: Vec<usize>,
}

impl FeatureSpec {
    /// Build a spec from per-feature cardinalities.
    ///
    /// # Panics
    /// If any cardinality is zero (a feature with no categories cannot be
    /// one-hot encoded).
    #[must_use]
    pub fn new(cardinalities: Vec<u32>) -> Self {
        assert!(
            cardinalities.iter().all(|&k| k > 0),
            "every feature needs ≥1 category"
        );
        let mut offsets = Vec::with_capacity(cardinalities.len() + 1);
        let mut acc = 0usize;
        offsets.push(0);
        for &k in &cardinalities {
            acc += k as usize;
            offsets.push(acc);
        }
        Self {
            cardinalities,
            offsets,
        }
    }

    /// Total input dimension `D = Σ k_i`.
    #[must_use]
    pub fn dim(&self) -> usize {
        *self.offsets.last().expect("offsets always has ≥1 entry")
    }

    /// Number of features (columns).
    #[must_use]
    pub fn num_features(&self) -> usize {
        self.cardinalities.len()
    }

    /// Category count for a feature.
    #[must_use]
    pub fn cardinality(&self, feature: usize) -> u32 {
        self.cardinalities[feature]
    }

    /// The `[start, end)` input-slot range owned by a feature's block.
    #[must_use]
    pub fn block(&self, feature: usize) -> (usize, usize) {
        (self.offsets[feature], self.offsets[feature + 1])
    }

    /// The absolute input slot for an `(feature, category)` item.
    #[must_use]
    pub fn slot(&self, item: Item) -> usize {
        self.offsets[item.feature as usize] + item.category as usize
    }

    /// One-hot encode a row given as one category index per feature.
    ///
    /// `row[i]` is the observed category of feature `i`. Returns a `D`-length
    /// vector with one `1.0` per block.
    #[must_use]
    pub fn encode(&self, row: &[u32]) -> Vec<f32> {
        assert_eq!(row.len(), self.num_features(), "row arity mismatch");
        let mut v = vec![0.0f32; self.dim()];
        for (feature, &cat) in row.iter().enumerate() {
            debug_assert!(cat < self.cardinalities[feature], "category out of range");
            v[self.offsets[feature] + cat as usize] = 1.0;
        }
        v
    }

    /// Discretise a numeric value into `[0, bins)` over `[lo, hi]` (uniform
    /// width). Values at or below `lo` land in bin 0; at or above `hi` in the
    /// last bin. This is the pre-encoding step for numerical columns.
    #[must_use]
    pub fn bin(value: f32, lo: f32, hi: f32, bins: u32) -> u32 {
        debug_assert!(bins > 0 && hi > lo, "bad binning parameters");
        let t = ((value - lo) / (hi - lo)).clamp(0.0, 1.0);
        let idx = (t * bins as f32) as u32;
        idx.min(bins - 1)
    }
}

/// A window of rows in category-index form, sharing one [`FeatureSpec`].
///
/// This is the unit Aerial+ trains on and the unit support/confidence are
/// counted over. `rows.len()` is the `n` that the NARS confidence mapping
/// uses as evidential mass.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// The shared schema.
    pub spec: FeatureSpec,
    /// One `Vec<u32>` per row; each is one category index per feature.
    pub rows: Vec<Vec<u32>>,
}

impl Dataset {
    /// Wrap rows in a dataset, validating arity against the spec.
    ///
    /// # Panics
    /// If any row's length differs from the spec's feature count.
    #[must_use]
    pub fn new(spec: FeatureSpec, rows: Vec<Vec<u32>>) -> Self {
        for r in &rows {
            assert_eq!(r.len(), spec.num_features(), "row arity mismatch");
        }
        Self { spec, rows }
    }

    /// Window size `n`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the window is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Count of rows containing every item in `items` (an AND over items).
    #[must_use]
    pub fn count_matching(&self, items: &[Item]) -> u32 {
        self.rows
            .iter()
            .filter(|row| {
                items
                    .iter()
                    .all(|it| row[it.feature as usize] == it.category)
            })
            .count() as u32
    }

    /// Classical support of an itemset: `count_matching / n`.
    #[must_use]
    pub fn support(&self, items: &[Item]) -> f32 {
        if self.rows.is_empty() {
            return 0.0;
        }
        self.count_matching(items) as f32 / self.rows.len() as f32
    }

    /// Classical confidence of `antecedent → consequent`:
    /// `count(antecedent ∪ consequent) / count(antecedent)`.
    /// Returns `0.0` when the antecedent never occurs.
    #[must_use]
    pub fn confidence(&self, antecedent: &[Item], consequent: &[Item]) -> f32 {
        let ant = self.count_matching(antecedent);
        if ant == 0 {
            return 0.0;
        }
        let mut both: Vec<Item> = antecedent.to_vec();
        both.extend_from_slice(consequent);
        self.count_matching(&both) as f32 / ant as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_and_dim() {
        let spec = FeatureSpec::new(vec![2, 3, 1]);
        assert_eq!(spec.dim(), 6);
        assert_eq!(spec.num_features(), 3);
        assert_eq!(spec.block(0), (0, 2));
        assert_eq!(spec.block(1), (2, 5));
        assert_eq!(spec.block(2), (5, 6));
    }

    #[test]
    fn one_hot_sets_one_per_block() {
        let spec = FeatureSpec::new(vec![2, 3]);
        let v = spec.encode(&[1, 2]);
        assert_eq!(v, vec![0.0, 1.0, 0.0, 0.0, 1.0]);
        // exactly one 1.0 per block
        assert_eq!(v[0..2].iter().sum::<f32>(), 1.0);
        assert_eq!(v[2..5].iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn slot_addressing() {
        let spec = FeatureSpec::new(vec![2, 3]);
        assert_eq!(spec.slot(Item::new(0, 1)), 1);
        assert_eq!(spec.slot(Item::new(1, 2)), 4);
    }

    #[test]
    fn binning_clamps_edges() {
        assert_eq!(FeatureSpec::bin(-5.0, 0.0, 10.0, 5), 0);
        assert_eq!(FeatureSpec::bin(0.0, 0.0, 10.0, 5), 0);
        assert_eq!(FeatureSpec::bin(9.99, 0.0, 10.0, 5), 4);
        assert_eq!(FeatureSpec::bin(100.0, 0.0, 10.0, 5), 4);
        assert_eq!(FeatureSpec::bin(5.0, 0.0, 10.0, 5), 2);
    }

    #[test]
    fn support_and_confidence_counts() {
        let spec = FeatureSpec::new(vec![2, 2]);
        // rows: (0,0),(0,0),(0,1),(1,1)
        let data = Dataset::new(
            spec,
            vec![vec![0, 0], vec![0, 0], vec![0, 1], vec![1, 1]],
        );
        // support of feature0=0 : 3/4
        assert!((data.support(&[Item::new(0, 0)]) - 0.75).abs() < 1e-6);
        // confidence (f0=0 -> f1=0): count(0,0)=2 / count(f0=0)=3
        let c = data.confidence(&[Item::new(0, 0)], &[Item::new(1, 0)]);
        assert!((c - 2.0 / 3.0).abs() < 1e-6);
        // confidence on never-seen antecedent is 0
        assert_eq!(
            data.confidence(&[Item::new(0, 5)], &[Item::new(1, 0)]),
            0.0
        );
    }
}
