//! `space` — the semantic-distance layer, on the certified palette256² tables.
//!
//! ## What changed from DeepNSM v1
//!
//! v1 baked a `4096²` u8 distance matrix (16 MB) — one byte per word pair — from
//! a CAM-PQ codebook, then looked distances up by `(id_a, id_b)`. v2 keeps the
//! *look-it-up* shape but the table is the **certified-exact palette256²
//! distance** from `lance-graph-contract`: a word's `(basin, identity)` pair
//! ([`crate::vocab`]) indexes two 256-centroid axis codebooks, and the distance
//! is the real centroid distance, not a byte-grid approximation.
//!
//! Two consumers of the contract, no reimplementation:
//!
//! - [`SemanticSpace`] wraps [`recipe_substrate::PairPalette`] — the palette256²
//!   `(u8, u8)` pair distance. This is the direct replacement for the `4096²`
//!   matrix (a `256×256` tile is `65_536` cells vs `4096²`, and it is
//!   *computed* from the codebook, not stored).
//! - [`AdcSpace`] wraps [`cam::ScalarAdc`] — the `6×256` ADC path, i.e. the
//!   "whole-work-is-one-tile" 6-byte CAM code (a work of `≤64k` SPO is one
//!   `256×256` centroid tile; the 6 subspaces are that tile's CAM-PQ code).
//!
//! ## Honest scope (the certified-table debt applies)
//!
//! Both spaces need a **trained codebook** to carry real semantics. Providing
//! that from real embeddings is the ndarray-side producer named in
//! `TD-CERTIFIED-DISTANCE-TABLE-UNCONSUMED` — it does not exist yet. So this
//! module ships:
//! - `demo()` constructors with a **deterministic, index-derived** codebook, so
//!   the crate compiles, tests, and runs standalone — but the demo distances are
//!   a placeholder, NOT real semantics;
//! - `from_axis_codebooks()` / `from_codebook()` constructors that take REAL
//!   trained codebooks when a producer supplies them.

use lance_graph_contract::cam::{AdcMetric, DistanceTableProvider, ScalarAdc};
use lance_graph_contract::recipe_substrate::PairPalette;

/// A `256`-centroid axis codebook: `centroid[i]` is that axis-byte's vector.
pub type AxisCodebook = Vec<Vec<f32>>;

/// The palette256² semantic space — the `(basin, identity)` pair-distance table.
///
/// Wraps [`PairPalette`]; `similarity` reads the certified `256×256` LUT.
#[derive(Debug, Clone)]
pub struct SemanticSpace {
    palette: PairPalette,
}

impl SemanticSpace {
    /// Build from two REAL trained axis codebooks (`basin`, `identity`), each
    /// `≤256` centroids. `d_max` normalizes the raw squared-L2 into a `[0, 1]`
    /// similarity — pass the codebook diameter (a non-positive value is clamped
    /// to `1.0` by the contract).
    #[must_use]
    pub fn from_axis_codebooks(basin: AxisCodebook, identity: AxisCodebook, d_max: f32) -> Self {
        Self {
            palette: PairPalette::new(basin, identity, d_max),
        }
    }

    /// A DETERMINISTIC demo space (no rng/clock) so the crate runs standalone.
    /// `dim` is the centroid dimension. The distances are a placeholder — real
    /// semantics need [`from_axis_codebooks`](Self::from_axis_codebooks) with a
    /// trained codebook.
    #[must_use]
    pub fn demo(dim: usize) -> Self {
        Self::from_axis_codebooks(
            demo_axis(1, dim),
            demo_axis(2, dim),
            // A generous normalizer so demo similarities spread across [0,1].
            (dim as f32) * 64.0,
        )
    }

    /// Semantic similarity ∈ `[0, 1]` between two words, addressed by their
    /// `(basin, identity)` palette pairs (from [`crate::vocab::PaletteVocab::pair`]).
    /// `1.0` = identical.
    #[must_use]
    pub fn similarity(&self, a: (u8, u8), b: (u8, u8)) -> f32 {
        self.palette.similarity(a, b)
    }

    /// Raw squared-L2 distance between two words' reconstructed points (the
    /// pre-normalization value `similarity` is `1 − d/d_max` of).
    #[must_use]
    pub fn distance(&self, a: (u8, u8), b: (u8, u8)) -> f32 {
        self.palette.distance(a, b)
    }
}

/// The `6×256` ADC space — the whole-work centroid-tile CAM path.
///
/// Wraps [`ScalarAdc`]: a query vector precomputes `6×256` tables against a
/// 6-subspace codebook, and each candidate is a 6-byte CAM code scored by
/// summing 6 lookups. This is the exact, additive-decomposition SSD path (a
/// code's distance to its own reconstruction is `0`).
#[derive(Debug, Clone)]
pub struct AdcSpace {
    adc: ScalarAdc,
    /// `codebook[subspace][centroid]` — 6 subspaces, `≤256` centroids each.
    codebook: Vec<AxisCodebook>,
}

impl AdcSpace {
    /// Build from a REAL 6-subspace trained codebook.
    #[must_use]
    pub fn from_codebook(codebook: Vec<AxisCodebook>) -> Self {
        Self {
            adc: ScalarAdc::new(AdcMetric::SquaredL2),
            codebook,
        }
    }

    /// A DETERMINISTIC demo ADC space (6 subspaces × `k` centroids × `dim`).
    /// Placeholder distances — real semantics need a trained codebook.
    #[must_use]
    pub fn demo(k: usize, dim: usize) -> Self {
        let codebook: Vec<AxisCodebook> = (0..6)
            .map(|s| {
                (0..k.min(256))
                    .map(|c| {
                        (0..dim)
                            .map(|d| ((s * 31 + c * 7 + d * 3) % 17) as f32 - 8.0)
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Self::from_codebook(codebook)
    }

    /// Precompute the `6×256` distance tables for a query vector (length =
    /// `6 × subspace_dim`). Reuse the tables across many candidate CAM codes.
    #[must_use]
    pub fn precompute(&self, query: &[f32]) -> [[f32; 256]; 6] {
        self.adc.precompute(query, &self.codebook)
    }

    /// ADC distance from the precomputed `tables` to one 6-byte CAM code.
    #[must_use]
    pub fn distance(&self, tables: &[[f32; 256]; 6], cam: &[u8; 6]) -> f32 {
        self.adc.distance(tables, cam)
    }

    /// The full query vector a code reconstructs to (concatenated centroids) —
    /// the "decode" side, useful for the exactness property in tests.
    #[must_use]
    pub fn reconstruct(&self, cam: &[u8; 6]) -> Vec<f32> {
        let mut v = Vec::new();
        for (s, subspace) in self.codebook.iter().enumerate() {
            if let Some(c) = subspace.get(cam[s] as usize) {
                v.extend_from_slice(c);
            }
        }
        v
    }
}

/// Deterministic `256`-centroid axis codebook (index-derived; demo only).
fn demo_axis(seed: usize, dim: usize) -> AxisCodebook {
    (0..256)
        .map(|c| {
            (0..dim)
                .map(|d| ((c * 5 + seed * 3 + d * 2) % 13) as f32 - 6.0)
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_words_are_maximally_similar() {
        let space = SemanticSpace::demo(4);
        assert!((space.similarity((7, 42), (7, 42)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn similarity_is_bounded_and_symmetric() {
        let space = SemanticSpace::demo(4);
        let a = (3, 9);
        let b = (200, 4);
        let s = space.similarity(a, b);
        assert!((0.0..=1.0).contains(&s));
        assert!((space.similarity(a, b) - space.similarity(b, a)).abs() < 1e-6);
    }

    #[test]
    fn adc_code_to_its_own_reconstruction_is_zero() {
        // The exactness property the contract proves, exercised through the space.
        let space = AdcSpace::demo(8, 4);
        let code = [1u8, 3, 0, 5, 2, 7];
        let query = space.reconstruct(&code);
        let tables = space.precompute(&query);
        assert!(space.distance(&tables, &code) < 1e-4);
    }

    #[test]
    fn adc_orders_near_before_far() {
        let space = AdcSpace::demo(8, 4);
        let near_code = [1u8, 1, 1, 1, 1, 1];
        let query = space.reconstruct(&near_code);
        let tables = space.precompute(&query);
        let near = space.distance(&tables, &near_code);
        let far = space.distance(&tables, &[7, 7, 7, 7, 7, 7]);
        assert!(near <= far);
    }
}
