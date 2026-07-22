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

/// A 96-bit CAM-PQ **DISTRIBUTION** code: `6×(u8:u8)` = 12 bytes = 6 rails,
/// each a `palette256:palette256` pair (a cosine²). This is deepnsm_v2's word /
/// meaning code — the granular upgrade of deepnsm's 48-bit CAM-PQ **POINT**
/// (`6×256`, [`AdcSpace`]). Same 6 subspaces; each subspace gains a second axis
/// (point → distribution).
pub type Cam96 = [u8; 12];

/// The CAM-PQ **96-bit DISTRIBUTION** space: 12 axis codebooks (2 per rail × 6
/// rails), ≤256 centroids each. [`encode`](Self::encode) quantizes a length-`12·d`
/// vector into the 12-byte code (nearest centroid per axis);
/// [`distance`](Self::distance) sums the 12 per-axis centroid squared-L2 — the
/// additive-decomposition exactness (`‖q−c‖² = Σ_k ‖q_k−c_k‖²` over disjoint
/// subspaces) — normalized by `d_max`. No cosine call: the normalized `[x;y]`
/// coordinate distance carries the ordering directly.
///
/// Measured (`probes/`, Jina-v3 96-d ground truth, HELD-OUT protocol —
/// codebooks fit on a disjoint train split): this 96-bit distribution preserves
/// **ρ 0.766** of Jina's meaning ordering vs the 48-bit point's **ρ 0.624**
/// (+22.7%), at 39% lower reconstruction error — the point→distribution
/// ladder, out-of-sample. Real semantics need a **trained** codebook
/// ([`from_axis_codebooks`](Self::from_axis_codebooks)); [`demo`](Self::demo)
/// ships a deterministic placeholder so the crate runs standalone.
#[derive(Debug, Clone)]
pub struct Cam96Space {
    /// 12 axis codebooks (rail `r` = axes `2r`, `2r+1`), each ≤256 centroids.
    axes: Vec<AxisCodebook>,
    d_max: f32,
}

impl Cam96Space {
    /// Build from 12 REAL trained axis codebooks (2 per rail × 6 rails).
    /// `d_max` normalizes the summed squared-L2 into a `[0, 1]` similarity;
    /// a non-positive value is clamped to `1.0`.
    ///
    /// # Panics
    /// Panics if `axes.len() != 12` (the 96-bit facet is exactly 6 rails × 2).
    #[must_use]
    pub fn from_axis_codebooks(axes: Vec<AxisCodebook>, d_max: f32) -> Self {
        assert_eq!(axes.len(), 12, "CAM-PQ 96 is 12 axes (6 rails × 2)");
        let d_max = if d_max.is_finite() && d_max > 0.0 {
            d_max
        } else {
            1.0
        };
        Self { axes, d_max }
    }

    /// A DETERMINISTIC demo space (12 axes × `dim`-d centroids). Placeholder
    /// distances — real semantics need [`from_axis_codebooks`](Self::from_axis_codebooks).
    #[must_use]
    pub fn demo(dim: usize) -> Self {
        let axes = (0..12).map(|s| demo_axis(s + 1, dim)).collect();
        Self::from_axis_codebooks(axes, (dim as f32) * 12.0 * 64.0)
    }

    /// The per-axis CENTROID dimension (each of the 12 axes owns `axis_dim()`
    /// consecutive components of an input vector). Read from the first
    /// centroid's length — NOT the outer `Vec` length, which is the centroid
    /// COUNT (≤256); conflating the two collapsed every encode to centroid 0
    /// (caught independently by two reviewers on PR #801).
    #[must_use]
    pub fn axis_dim(&self) -> usize {
        self.axes
            .first()
            .and_then(|axis| axis.first())
            .map_or(0, Vec::len)
    }

    /// Quantize a length-`12·axis_dim` vector into a 12-byte [`Cam96`] code:
    /// each axis picks its nearest of ≤256 centroids. A short/empty axis yields
    /// centroid `0`.
    #[must_use]
    pub fn encode(&self, v: &[f32]) -> Cam96 {
        let d = self.axis_dim();
        let mut code = [0u8; 12];
        for (k, axis) in self.axes.iter().enumerate() {
            let chunk = v.get(k * d..(k + 1) * d).unwrap_or(&[]);
            let mut best = 0u8;
            let mut best_d = f32::INFINITY;
            for (ci, c) in axis.iter().enumerate() {
                let dist: f32 = chunk
                    .iter()
                    .zip(c.iter())
                    .map(|(&x, &y)| (x - y) * (x - y))
                    .sum();
                if dist < best_d {
                    best_d = dist;
                    best = ci as u8;
                }
            }
            code[k] = best;
        }
        code
    }

    /// Squared-L2 distance between two codes = `Σ_k ‖axis_k[a_k] − axis_k[b_k]‖²`
    /// (additive over the 12 disjoint axes — the exactness property). An
    /// out-of-range centroid on either side reads as `+∞` (absent ≠ match).
    #[must_use]
    pub fn distance(&self, a: &Cam96, b: &Cam96) -> f32 {
        let mut total = 0.0f32;
        for (k, axis) in self.axes.iter().enumerate() {
            match (axis.get(a[k] as usize), axis.get(b[k] as usize)) {
                (Some(ca), Some(cb)) => {
                    total += ca
                        .iter()
                        .zip(cb.iter())
                        .map(|(&x, &y)| (x - y) * (x - y))
                        .sum::<f32>();
                }
                _ => return f32::INFINITY,
            }
        }
        total
    }

    /// Similarity ∈ `[0, 1]` = `1 − distance/d_max`, clamped. `1.0` = identical.
    #[must_use]
    pub fn similarity(&self, a: &Cam96, b: &Cam96) -> f32 {
        let d = self.distance(a, b);
        if !d.is_finite() {
            return 0.0;
        }
        (1.0 - d / self.d_max).clamp(0.0, 1.0)
    }

    /// The 6-rail `[(u8, u8); 6]` view of a code — the `6×palette256:palette256`
    /// reading (rail `r` = `(code[2r], code[2r+1])`).
    #[must_use]
    pub fn rails(code: &Cam96) -> [(u8, u8); 6] {
        std::array::from_fn(|r| (code[2 * r], code[2 * r + 1]))
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

    #[test]
    fn cam96_is_twelve_bytes_six_rails() {
        let s = Cam96Space::demo(4);
        let code: Cam96 = s.encode(&[0.0; 48]); // 12 axes × 4-d
        assert_eq!(code.len(), 12);
        assert_eq!(Cam96Space::rails(&code).len(), 6); // 6×(u8:u8)
    }

    #[test]
    fn cam96_identical_codes_are_maximally_similar() {
        let s = Cam96Space::demo(4);
        let code = [3u8, 9, 200, 4, 0, 255, 1, 2, 3, 4, 5, 6];
        assert!((s.similarity(&code, &code) - 1.0).abs() < 1e-6);
        assert!(s.distance(&code, &code).abs() < 1e-6);
    }

    #[test]
    fn cam96_axis_dim_is_centroid_dimension_not_count() {
        // Regression for the PR #801 review finding: axis_dim() must be the
        // per-centroid dimension (`dim`), never the centroid COUNT (256).
        let s = Cam96Space::demo(4);
        assert_eq!(s.axis_dim(), 4);
    }

    #[test]
    fn cam96_encode_recovers_exact_centroids() {
        // A vector built by concatenating known centroids re-encodes to exactly
        // those indices. Indices are kept < 13 because demo_axis is mod-13
        // periodic (centroid c ≡ c+13), so below 13 recovery is unambiguous.
        // This is the STRONG test whose earlier failure was misdiagnosed as
        // tie-collisions when it was actually the axis_dim count/dim bug.
        let dim = 4;
        let s = Cam96Space::demo(dim);
        let target: Cam96 = [2, 5, 0, 9, 12, 1, 7, 3, 11, 6, 4, 8];
        let mut v = Vec::new();
        for (axis, &centroid) in target.iter().enumerate() {
            v.extend_from_slice(&demo_axis(axis + 1, dim)[centroid as usize]);
        }
        assert_eq!(s.encode(&v), target);
        // and a non-centroid vector no longer collapses to all-zeros:
        let noisy: Vec<f32> = (0..12 * dim)
            .map(|i| (i as f32 * 0.37).sin() * 5.0)
            .collect();
        let code = s.encode(&noisy);
        assert!(
            code.iter().any(|&c| c != 0),
            "encode must not collapse to centroid 0"
        );
    }

    #[test]
    fn cam96_similarity_is_symmetric_and_bounded() {
        let s = Cam96Space::demo(4);
        let a = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let b = [200u8, 199, 198, 197, 196, 195, 194, 193, 192, 191, 190, 189];
        let sim = s.similarity(&a, &b);
        assert!((0.0..=1.0).contains(&sim));
        assert!((s.similarity(&a, &b) - s.similarity(&b, &a)).abs() < 1e-6);
    }
}
