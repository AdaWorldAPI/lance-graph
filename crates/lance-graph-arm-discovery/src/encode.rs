// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Codebook items + the data window — **integer only**.
//!
//! Each row is a list of category indices, one per feature (a tuple of
//! codebook codes). There is no one-hot float vector and no embedding: the
//! codebook-probe backend ([`crate::aerial`]) reasons over category indices
//! and integer co-occurrence counts, and the similarity it needs comes from
//! the injected [`crate::aerial::CodebookDistance`] oracle. Numeric columns
//! must be discretised into category indices *before* they reach this layer
//! (that quantisation belongs to the CAM/palette codebook, not here).

use crate::rule::Item;

/// The schema for a window: the ordered features and how many categories each
/// has. Block offsets are derived once and cached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSpec {
    cardinalities: Vec<u32>,
    /// Prefix-sum offsets: `offsets[i]` is the first code slot of feature `i`;
    /// the last entry is the total code space size.
    offsets: Vec<usize>,
}

impl FeatureSpec {
    /// Build a spec from per-feature cardinalities.
    ///
    /// # Panics
    /// If any cardinality is zero.
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

    /// Total code space `Σ kᵢ` (the side length of a per-item distance table).
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

    /// The `[start, end)` code range owned by a feature's block.
    #[must_use]
    pub fn block(&self, feature: usize) -> (usize, usize) {
        (self.offsets[feature], self.offsets[feature + 1])
    }

    /// The absolute code slot for an `(feature, category)` item.
    #[must_use]
    pub fn slot(&self, item: Item) -> usize {
        self.offsets[item.feature as usize] + item.category as usize
    }
}

/// A window of rows in category-index form, sharing one [`FeatureSpec`].
/// `rows.len()` is the `n` the evidence counts are measured over.
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
    /// The one and only evidence primitive — integer, exact.
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
    fn slot_addressing() {
        let spec = FeatureSpec::new(vec![2, 3]);
        assert_eq!(spec.slot(Item::new(0, 1)), 1);
        assert_eq!(spec.slot(Item::new(1, 2)), 4);
    }

    #[test]
    fn count_matching_is_an_and_over_items() {
        let spec = FeatureSpec::new(vec![2, 2]);
        // rows: (0,0),(0,0),(0,1),(1,1)
        let data = Dataset::new(spec, vec![vec![0, 0], vec![0, 0], vec![0, 1], vec![1, 1]]);
        assert_eq!(data.count_matching(&[Item::new(0, 0)]), 3);
        assert_eq!(data.count_matching(&[Item::new(0, 0), Item::new(1, 0)]), 2);
        assert_eq!(data.count_matching(&[Item::new(0, 5)]), 0); // absent category
        assert_eq!(data.len(), 4);
    }
}
