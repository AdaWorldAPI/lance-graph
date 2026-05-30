// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The deterministic similarity oracle — `CodebookDistance`.
//!
//! This is the float-free replacement for Aerial+'s autoencoder. The paper's
//! reconstruction probe ("mark the antecedent, read off the high-probability
//! consequents") is, mechanically, a **nearest-neighbour query in the learned
//! co-occurrence space** — and this substrate already answers that exactly,
//! in integers, via the **palette256 distance table** (`[a,b] → u32`, ρ=0.9973
//! vs cosine). So the probe ranks consequents by integer codebook distance,
//! never by a softmax over float weights.
//!
//! The oracle is a **trait**, not a concrete table, so this crate stays
//! zero-dep and standalone. The production table is **built and certified
//! offline by `crates/jc` (Jirak-Cartan)** — `ewa_sandwich` certifies the
//! Gaussian-splat Σ-push-forward (the 10000² BLASGraph spatial top-k that
//! constructs it), and `sigma_codebook_probe` measures the 256-codebook viable
//! at ρ=0.9973. That offline build may use float (k-means, SPD math); the
//! frozen `[u32; dim²]` table it emits is consumed here as integer through
//! [`MatrixDistance`]. Float to BUILD the codebook, integer to USE it — the
//! CAM-PQ doctrine. See `.claude/knowledge/splat-codebook-aerial-wikidata-compression.md`.
//! [`MatrixDistance`] is also the in-crate reference impl used by tests.
//!
//! Invariant (per `faiss-homology-cam-pq.md`): this distance is **discovery /
//! shape-family only**, never identity addressing. Identity is the exact CAM
//! hash over canonicalized `(s,p,o)`. Same triples, two indexes, never swapped.

use crate::encode::FeatureSpec;
use crate::rule::Item;

/// A deterministic, integer distance between two codebook items. Lower means
/// more associated (nearer). No float, no seed — bitwise-identical on every
/// target.
pub trait CodebookDistance {
    /// Distance between two items' codebook codes. Lower = nearer.
    fn distance(&self, a: Item, b: Item) -> u32;
}

/// Reference [`CodebookDistance`] over a flat `dim × dim` `u32` table keyed by
/// the [`FeatureSpec`] slot of each item. Zero-dep; the in-crate stand-in for
/// `bgz17::PaletteDistanceTable`.
#[derive(Debug, Clone)]
pub struct MatrixDistance {
    dim: usize,
    offsets: Vec<usize>,
    table: Vec<u32>,
}

impl MatrixDistance {
    /// Build from a spec and a flat `dim × dim` row-major distance table.
    ///
    /// # Panics
    /// If `table.len() != spec.dim()²`.
    #[must_use]
    pub fn new(spec: &FeatureSpec, table: Vec<u32>) -> Self {
        let dim = spec.dim();
        assert_eq!(table.len(), dim * dim, "distance table must be dim × dim");
        let offsets: Vec<usize> = (0..spec.num_features())
            .map(|f| spec.block(f).0)
            .collect();
        Self { dim, offsets, table }
    }

    /// The flat code (slot index) of an item. Validates bounds so an invalid
    /// `(feature, category)` fails fast rather than aliasing another feature's
    /// block and returning a real-but-wrong distance.
    #[must_use]
    fn code(&self, it: Item) -> usize {
        let feature = it.feature as usize;
        let start = *self
            .offsets
            .get(feature)
            .expect("item feature out of range for distance table");
        let end = self.offsets.get(feature + 1).copied().unwrap_or(self.dim);
        let category = it.category as usize;
        assert!(
            category < end - start,
            "item category {category} out of range for feature {feature} block"
        );
        start + category
    }
}

impl CodebookDistance for MatrixDistance {
    fn distance(&self, a: Item, b: Item) -> u32 {
        self.table[self.code(a) * self.dim + self.code(b)]
    }
}

/// Aggregate the distance from a (multi-item) antecedent to a candidate
/// consequent item: the **nearest** antecedent item wins (min). For a
/// single-item antecedent this is just `distance(a, item)`.
#[must_use]
pub fn antecedent_distance(
    oracle: &dyn CodebookDistance,
    antecedent: &[Item],
    item: Item,
) -> u32 {
    antecedent
        .iter()
        .map(|&a| oracle.distance(a, item))
        .min()
        .unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_distance_is_slot_addressed_and_symmetric_if_built_so() {
        let spec = FeatureSpec::new(vec![2, 2]); // dim 4
        // 4×4 table; (f0,c0)=slot0 near (f1,c0)=slot2 (dist 1), far from slot3.
        #[rustfmt::skip]
        let table = vec![
            0, 5, 1, 9,
            5, 0, 9, 1,
            1, 9, 0, 5,
            9, 1, 5, 0,
        ];
        let d = MatrixDistance::new(&spec, table);
        assert_eq!(d.distance(Item::new(0, 0), Item::new(1, 0)), 1);
        assert_eq!(d.distance(Item::new(0, 0), Item::new(1, 1)), 9);
        assert_eq!(d.distance(Item::new(0, 0), Item::new(0, 0)), 0);
    }

    #[test]
    fn antecedent_distance_takes_the_nearest_item() {
        let spec = FeatureSpec::new(vec![2, 2]);
        #[rustfmt::skip]
        let table = vec![
            0, 5, 1, 9,
            5, 0, 9, 1,
            1, 9, 0, 5,
            9, 1, 5, 0,
        ];
        let d = MatrixDistance::new(&spec, table);
        // antecedent {(f0,c0)=slot0, (f0,c1)=slot1} to (f1,c0)=slot2: min(1, 9) = 1
        let ant = [Item::new(0, 0), Item::new(0, 1)];
        assert_eq!(antecedent_distance(&d, &ant, Item::new(1, 0)), 1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn code_rejects_out_of_range_category() {
        let spec = FeatureSpec::new(vec![2, 2]);
        let d = MatrixDistance::new(&spec, vec![0u32; 16]);
        // category 5 ≥ feature-0 block width 2 — must fail, not alias feature 1.
        let _ = d.distance(Item::new(0, 5), Item::new(1, 0));
    }
}
