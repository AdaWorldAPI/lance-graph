// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Row-bitset SoA — the columnar form the codebook probe counts over.
//!
//! `count_matching` on the AoS `Dataset` is an `O(rows × items)` scan. The
//! probe calls it once per candidate, so the window is rescanned thousands of
//! times. [`RowMasks`] transposes the window once into one `u64` bitset per
//! `(feature, category)` item (bit `r` set iff row `r` has that category);
//! then every candidate's support/co-occurrence count is an `AND` + popcount
//! over `&[u64]` — the [`crate::simd`] primitive that routes through
//! `ndarray::simd::U64x8` under the `ndarray-simd` feature.

use crate::encode::Dataset;
use crate::rule::Item;
use crate::simd::{and_popcount, popcount};

/// One `u64` bitset per item slot, over the window's rows.
#[derive(Debug, Clone)]
pub struct RowMasks {
    /// Rows in the window.
    n: usize,
    /// `u64` words per mask = `ceil(n / 64)`.
    words: usize,
    /// Per-feature code offset (`offsets[feature] + category` = item slot).
    offsets: Vec<usize>,
    /// Flat `dim × words` storage; mask of slot `s` is `[s·words .. (s+1)·words)`.
    masks: Vec<u64>,
}

impl RowMasks {
    /// Transpose a dataset into row bitsets (one pass over the window).
    #[must_use]
    pub fn build(data: &Dataset) -> Self {
        let spec = &data.spec;
        let dim = spec.dim();
        let n = data.len();
        let words = n.div_ceil(64).max(1);
        let offsets: Vec<usize> = (0..spec.num_features()).map(|f| spec.block(f).0).collect();
        let mut masks = vec![0u64; dim * words];
        for (r, row) in data.rows.iter().enumerate() {
            let w = r / 64;
            let bit = 1u64 << (r % 64);
            for (f, &cat) in row.iter().enumerate() {
                let slot = offsets[f] + cat as usize;
                masks[slot * words + w] |= bit;
            }
        }
        Self {
            n,
            words,
            offsets,
            masks,
        }
    }

    /// Window size `n`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the window is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The bitset of an item.
    #[must_use]
    fn mask(&self, item: Item) -> &[u64] {
        let slot = self.offsets[item.feature as usize] + item.category as usize;
        &self.masks[slot * self.words..(slot + 1) * self.words]
    }

    /// `|{rows ⊇ {item}}|` — single-item support count.
    #[must_use]
    pub fn support_count(&self, item: Item) -> u32 {
        popcount(self.mask(item))
    }

    /// `|{rows ⊇ items}|` — conjunction count (the evidence primitive).
    /// The 1- and 2-item cases (the probe's hot path: a 1–2 item antecedent
    /// plus one consequent) avoid any allocation.
    #[must_use]
    pub fn and_count(&self, items: &[Item]) -> u32 {
        match items {
            [] => self.n as u32,
            [a] => self.support_count(*a),
            [a, b] => and_popcount(self.mask(*a), self.mask(*b)),
            [a, rest @ ..] => {
                let mut acc = self.mask(*a).to_vec();
                for &it in rest {
                    for (x, &y) in acc.iter_mut().zip(self.mask(it)) {
                        *x &= y;
                    }
                }
                popcount(&acc)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::FeatureSpec;

    fn data() -> Dataset {
        // rows: (0,0),(0,0),(0,1),(1,1)
        let spec = FeatureSpec::new(vec![2, 2]);
        Dataset::new(spec, vec![vec![0, 0], vec![0, 0], vec![0, 1], vec![1, 1]])
    }

    #[test]
    fn masks_match_aos_count() {
        let d = data();
        let m = RowMasks::build(&d);
        // single-item: feature0=0 in 3 rows
        assert_eq!(m.support_count(Item::new(0, 0)), 3);
        assert_eq!(
            m.support_count(Item::new(0, 0)),
            d.count_matching(&[Item::new(0, 0)])
        );
        // conjunction f0=0 ∧ f1=0 → 2 rows
        let both = [Item::new(0, 0), Item::new(1, 0)];
        assert_eq!(m.and_count(&both), 2);
        assert_eq!(m.and_count(&both), d.count_matching(&both));
        // empty conjunction = n
        assert_eq!(m.and_count(&[]), 4);
    }

    #[test]
    fn three_item_conjunction_folds() {
        let spec = FeatureSpec::new(vec![2, 2, 2]);
        let d = Dataset::new(spec, vec![vec![0, 0, 0], vec![0, 0, 1], vec![0, 0, 0]]);
        let m = RowMasks::build(&d);
        let items = [Item::new(0, 0), Item::new(1, 0), Item::new(2, 0)];
        assert_eq!(m.and_count(&items), 2);
        assert_eq!(m.and_count(&items), d.count_matching(&items));
    }

    #[test]
    fn spans_multiple_words() {
        // 130 rows → 3 u64 words per mask; every row has feature0=0.
        let spec = FeatureSpec::new(vec![1, 2]);
        let rows: Vec<Vec<u32>> = (0..130).map(|i| vec![0, (i % 2) as u32]).collect();
        let d = Dataset::new(spec, rows);
        let m = RowMasks::build(&d);
        assert_eq!(m.support_count(Item::new(0, 0)), 130);
        assert_eq!(m.support_count(Item::new(1, 0)), 65);
    }
}
