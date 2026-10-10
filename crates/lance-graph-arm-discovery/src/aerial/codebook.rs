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
        let offsets: Vec<usize> = (0..spec.num_features()).map(|f| spec.block(f).0).collect();
        Self {
            dim,
            offsets,
            table,
        }
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

/// A [`CodebookDistance`] backed by a **sparse top-k neighbour list per item** —
/// the shape the BLASGraph Gaussian-splat spatial top-k (10000²) actually
/// emits. You keep only the `k` nearest neighbours per node (certified by jc's
/// EWA-sandwich Σ-push-forward), never a dense `dim²` table. `distance(a, b)`
/// returns the stored splat distance if `b ∈ topk(a)`, else `miss` (far).
///
/// This is the production oracle shape for D-ARM-14 (the [`MatrixDistance`]
/// dense table is the test/reference impl). The splat that *builds* the
/// neighbour lists is the consumer-side `crates/jc` / blasgraph step; this crate
/// stays zero-dep and just consumes the frozen integer lists.
#[derive(Debug, Clone)]
pub struct TopKDistance {
    spec: FeatureSpec,
    miss: u32,
    /// `neighbours[slot]` = ascending `(neighbour_slot, distance)`, dedup'd to
    /// the nearest distance per neighbour. Binary-searchable.
    neighbours: Vec<Vec<(u32, u32)>>,
}

impl TopKDistance {
    /// Build from undirected splat edges `(a, b, distance)`. `miss` is the
    /// distance returned for pairs not in each other's top-k (i.e. "far").
    #[must_use]
    pub fn new(spec: FeatureSpec, miss: u32, edges: &[(Item, Item, u32)]) -> Self {
        let dim = spec.dim();
        let mut neighbours: Vec<Vec<(u32, u32)>> = vec![Vec::new(); dim];
        for &(a, b, d) in edges {
            let (ca, cb) = (spec.checked_slot(a), spec.checked_slot(b));
            neighbours[ca].push((cb as u32, d));
            neighbours[cb].push((ca as u32, d)); // splat distance is symmetric
        }
        for list in &mut neighbours {
            // sort by (neighbour, distance) so dedup_by_key keeps the nearest.
            list.sort_unstable();
            list.dedup_by_key(|&mut (n, _)| n);
        }
        Self {
            spec,
            miss,
            neighbours,
        }
    }
}

impl CodebookDistance for TopKDistance {
    fn distance(&self, a: Item, b: Item) -> u32 {
        let ca = self.spec.checked_slot(a);
        let cb = self.spec.checked_slot(b) as u32;
        if ca as u32 == cb {
            return 0;
        }
        match self.neighbours[ca].binary_search_by_key(&cb, |&(n, _)| n) {
            Ok(i) => self.neighbours[ca][i].1,
            Err(_) => self.miss,
        }
    }
}

/// Aggregate the distance from a (multi-item) antecedent to a candidate
/// consequent item: the **nearest** antecedent item wins (min). For a
/// single-item antecedent this is just `distance(a, item)`.
#[must_use]
pub fn antecedent_distance(oracle: &dyn CodebookDistance, antecedent: &[Item], item: Item) -> u32 {
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

    #[test]
    fn topk_distance_is_sparse_and_symmetric() {
        let spec = FeatureSpec::new(vec![2, 2]);
        // one splat edge (f0,c0)~(f1,c0) at distance 3; all else "far" (99).
        let edges = [(Item::new(0, 0), Item::new(1, 0), 3)];
        let d = TopKDistance::new(spec, 99, &edges);
        assert_eq!(d.distance(Item::new(0, 0), Item::new(1, 0)), 3);
        assert_eq!(d.distance(Item::new(1, 0), Item::new(0, 0)), 3, "symmetric");
        assert_eq!(
            d.distance(Item::new(0, 0), Item::new(1, 1)),
            99,
            "non-neighbour = far"
        );
        assert_eq!(d.distance(Item::new(0, 0), Item::new(0, 0)), 0, "self = 0");
    }

    #[test]
    fn topk_keeps_the_nearest_on_duplicate_edges() {
        let spec = FeatureSpec::new(vec![2, 2]);
        let edges = [
            (Item::new(0, 0), Item::new(1, 0), 9),
            (Item::new(0, 0), Item::new(1, 0), 2), // nearer duplicate wins
        ];
        let d = TopKDistance::new(spec, 99, &edges);
        assert_eq!(d.distance(Item::new(0, 0), Item::new(1, 0)), 2);
    }
}
