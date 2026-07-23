//! PaletteSemiring: algebraic structure over palette indices.
//!
//! The 256×256 distance matrix defines a metric space. The compose table
//! defines a multiplication (path composition via XOR bind in Base17 space).
//! Together they form a semiring compatible with Session A's TypedGraph.
//!
//! compose_table[a * k + b] = palette index of `palette[a].xor_bind(palette[b])`.
//! This is the "path through a then b" operation — XOR bind in Base17 is
//! self-inverse and associative (bitwise XOR on i16 dims).

use crate::base17::Base17;
use crate::distance_matrix::DistanceMatrix;
use crate::palette::Palette;
use crate::BASE_DIM;

/// Semiring over palette indices: distance (metric) + compose (path algebra).
#[derive(Clone, Debug)]
pub struct PaletteSemiring {
    /// Precomputed pairwise distance matrix.
    pub distance_matrix: DistanceMatrix,
    /// compose_table[a * k + b] = palette index of path(a → b).
    /// Size: k × k bytes.
    pub compose_table: Vec<u8>,
    /// Palette size.
    pub k: usize,
}

impl PaletteSemiring {
    /// Build from a palette: compute distance matrix and compose table.
    pub fn build(palette: &Palette) -> Self {
        let dm = DistanceMatrix::build(palette);
        let compose_table = build_compose(palette);
        PaletteSemiring {
            k: palette.len(),
            distance_matrix: dm,
            compose_table,
        }
    }

    /// Look up composed path: a → b.
    #[inline]
    pub fn compose(&self, a: u8, b: u8) -> u8 {
        self.compose_table[a as usize * self.k + b as usize]
    }

    /// Look up distance between two palette indices.
    #[inline]
    pub fn distance(&self, a: u8, b: u8) -> u16 {
        self.distance_matrix.distance(a, b)
    }

    /// Identity element: the palette entry closest to Base17::zero().
    pub fn identity(&self, palette: &Palette) -> u8 {
        palette.nearest(&Base17::zero())
    }

    /// Total byte size: distance matrix + compose table.
    pub fn byte_size(&self) -> usize {
        self.distance_matrix.byte_size() + self.compose_table.len()
    }
}

/// Build the compose table from a palette's Base17 entries.
///
/// For each pair (a, b), compute `palette[a].xor_bind(palette[b])`,
/// then find the nearest palette entry to the result.
pub fn build_compose(palette: &Palette) -> Vec<u8> {
    let k = palette.len();
    let mut table = vec![0u8; k * k];
    for a in 0..k {
        for b in 0..k {
            let composed = palette.entries[a].xor_bind(&palette.entries[b]);
            table[a * k + b] = palette.nearest(&composed);
        }
    }
    table
}

/// Three PaletteSemirings: one per S/P/O plane.
#[derive(Clone, Debug)]
pub struct SpoPaletteSemiring {
    pub subject: PaletteSemiring,
    pub predicate: PaletteSemiring,
    pub object: PaletteSemiring,
}

impl SpoPaletteSemiring {
    /// Build from three palettes.
    pub fn build(s_pal: &Palette, p_pal: &Palette, o_pal: &Palette) -> Self {
        SpoPaletteSemiring {
            subject: PaletteSemiring::build(s_pal),
            predicate: PaletteSemiring::build(p_pal),
            object: PaletteSemiring::build(o_pal),
        }
    }
}

/// D-DIA-V4 rung 3 — the `PremultipliedOver` palette composite.
///
/// Sort→reduce alpha-over depth compositing (3DGS-style: `T_i = T_{i-1}(1-α_i)`,
/// see `EPIPHANIES.md` `E-FOVEATED-HHTL-TRIE-FIELD-SEARCH-1`) is non-commutative
/// in its raw form because each `α_i` is scaled by the *running* transmittance
/// `T_i`, which depends on evaluation order. **Premultiplying** the transmittance
/// into the weight *before* calling this function (`weight_i = round(T_i · α_i ·
/// 255)`, computed once by the caller's depth-sort) turns the composite into a
/// plain weighted sum — a commutative, associative monoid — which is what makes
/// the operation safe to fold as an unordered GraphBLAS-style `mxv` reduce
/// instead of a strictly-ordered scan.
///
/// Carrier: bgz17's PALETTE world. `contribs` is a slice of
/// `(palette_code: u8, premultiplied_weight: u16)` pairs — the `SpoFacet`
/// palette-index side of a depth-sorted contribution list. `value(code)` is the
/// palette centroid (`palette.entries[code].dims`, the same `Base17` archetype
/// `PaletteSemiring::compose`/`distance` already index into) — the composite
/// reduces `Σ_i weight_i · value(code_i)` per Base17 dimension.
///
/// Accumulates in `i64` (not `u32`): `Base17` dims are *signed* `i16`, and
/// `weight (u16, ≤65535) × value (i16, ≤32767)` summed over up to 256
/// contributions needs headroom a `u32` can't give a signed value — `i64` has
/// no realistic overflow risk for palette-scale contribution lists.
///
/// Commutativity is the headline property under test: because plain integer
/// multiply-then-add is exactly commutative and associative (no floating-point
/// rounding), `premultiplied_over` is order-independent by construction once
/// the weights are premultiplied — the sort→reduce split.
pub fn premultiplied_over(palette: &Palette, contribs: &[(u8, u16)]) -> [i64; BASE_DIM] {
    let mut acc = [0i64; BASE_DIM];
    for &(code, weight) in contribs {
        let entry = &palette.entries[code as usize];
        for d in 0..BASE_DIM {
            acc[d] += weight as i64 * entry.dims[d] as i64;
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base17::Base17;
    use crate::BASE_DIM;

    fn make_palette(k: usize) -> Palette {
        let entries = (0..k)
            .map(|i| {
                let mut dims = [0i16; BASE_DIM];
                for d in 0..BASE_DIM {
                    dims[d] = ((i * 97 + d * 31) % 512) as i16 - 256;
                }
                Base17 { dims }
            })
            .collect();
        Palette { entries }
    }

    #[test]
    fn test_compose_identity() {
        // Build a palette that includes an actual zero entry for exact identity
        let mut entries: Vec<Base17> = (0..31)
            .map(|i| {
                let mut dims = [0i16; BASE_DIM];
                for d in 0..BASE_DIM {
                    dims[d] = ((i * 97 + d * 31) % 512) as i16 - 256;
                }
                Base17 { dims }
            })
            .collect();
        entries.push(Base17::zero()); // entry 31 is exact zero
        let pal = Palette { entries };

        let sr = PaletteSemiring::build(&pal);
        let id = sr.identity(&pal);
        assert_eq!(id, 31, "identity should be the zero entry");

        // compose(a, identity) = a when identity is exact zero
        for a in 0..32u8 {
            let composed = sr.compose(a, id);
            assert_eq!(
                composed, a,
                "compose({}, identity={}) should equal {} but got {}",
                a, id, a, composed
            );
        }
    }

    #[test]
    fn test_compose_self_inverse() {
        let pal = make_palette(32);
        let sr = PaletteSemiring::build(&pal);
        // xor_bind is self-inverse, so compose(compose(a, b), b) ≈ a
        for a in 0..8u8 {
            for b in 0..8u8 {
                let ab = sr.compose(a, b);
                let abb = sr.compose(ab, b);
                // Should be close to a (quantization error possible)
                let dist = sr.distance(a, abb);
                // Allow some quantization slack (palette approximation)
                assert!(
                    dist < 10000,
                    "compose(compose({}, {}), {}) = {}, dist to {} = {}",
                    a,
                    b,
                    b,
                    abb,
                    a,
                    dist
                );
            }
        }
    }

    #[test]
    fn test_compose_table_size() {
        let pal = make_palette(64);
        let sr = PaletteSemiring::build(&pal);
        assert_eq!(sr.compose_table.len(), 64 * 64);
        assert_eq!(sr.k, 64);
    }

    #[test]
    fn test_spo_semiring_build() {
        let s_pal = make_palette(16);
        let p_pal = make_palette(32);
        let o_pal = make_palette(24);
        let spo = SpoPaletteSemiring::build(&s_pal, &p_pal, &o_pal);
        assert_eq!(spo.subject.k, 16);
        assert_eq!(spo.predicate.k, 32);
        assert_eq!(spo.object.k, 24);
    }

    #[test]
    fn test_byte_size() {
        let pal = make_palette(256);
        let sr = PaletteSemiring::build(&pal);
        // distance matrix: 256*256*2 = 128KB
        // compose table: 256*256 = 64KB
        assert_eq!(sr.byte_size(), 256 * 256 * 2 + 256 * 256);
    }

    // ── D-DIA-V4 rung 3 — `premultiplied_over` gates ───────────────────────

    /// Deterministic SplitMix64 (seed 0x9E3779B97F4A7C15) — no clock/rand.
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform index in `[0, n)`.
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Deterministic Fisher-Yates shuffle over a SplitMix64 stream.
    fn shuffled<T: Clone>(items: &[T], seed: u64) -> Vec<T> {
        let mut v = items.to_vec();
        let mut rng = SplitMix64(seed);
        for i in (1..v.len()).rev() {
            let j = rng.below(i + 1);
            v.swap(i, j);
        }
        v
    }

    /// Deterministic (code, weight) contribution list over a k-entry palette.
    fn make_contribs(k: u8) -> Vec<(u8, u16)> {
        (0..k)
            .map(|code| (code, 1 + (code as u16) * 37 % 4000))
            .collect()
    }

    #[test]
    fn premultiplied_over_is_commutative() {
        // The headline gate: premultiplied weights make the sort→reduce
        // composite order-independent — plain (code, weight) reordering must
        // not change the result.
        let pal = make_palette(32);
        let contribs = make_contribs(32);
        let baseline = premultiplied_over(&pal, &contribs);

        let shuffled_contribs = shuffled(&contribs, 0x9E37_79B9_7F4A_7C15);
        // Sanity: the shuffle actually reorders (not a no-op permutation).
        assert_ne!(
            contribs, shuffled_contribs,
            "shuffle must actually permute the contribution order"
        );

        let reordered = premultiplied_over(&pal, &shuffled_contribs);
        assert_eq!(
            baseline, reordered,
            "premultiplied_over must be order-independent (commutative reduce)"
        );

        // A second, independently-derived shuffle (different seed) must also
        // agree — commutativity should hold for ANY permutation, not just one.
        let shuffled_again = shuffled(&contribs, 0xD1B5_4A32_D192_ED03);
        let reordered_again = premultiplied_over(&pal, &shuffled_again);
        assert_eq!(baseline, reordered_again);
    }

    #[test]
    fn premultiplied_over_is_linear_in_weight() {
        // Doubling every contribution's premultiplied weight must double the
        // reduce (plain weighted-sum linearity — no saturation, no clamping).
        let pal = make_palette(24);
        let contribs = make_contribs(24);
        let doubled: Vec<(u8, u16)> = contribs.iter().map(|&(c, w)| (c, w * 2)).collect();

        let base = premultiplied_over(&pal, &contribs);
        let scaled = premultiplied_over(&pal, &doubled);

        for d in 0..BASE_DIM {
            assert_eq!(
                scaled[d],
                base[d] * 2,
                "dim {d}: doubling weights must double the reduce ({} vs {})",
                scaled[d],
                base[d] * 2
            );
        }
    }

    #[test]
    fn premultiplied_over_zero_weight_is_a_no_op() {
        // A zero-weight contribution (fully occluded / T_i*alpha_i == 0) must
        // not perturb the result — appending one is a no-op.
        let pal = make_palette(16);
        let contribs = make_contribs(16);
        let base = premultiplied_over(&pal, &contribs);

        let mut with_zero = contribs.clone();
        with_zero.push((7, 0));
        let with_zero_result = premultiplied_over(&pal, &with_zero);
        assert_eq!(
            base, with_zero_result,
            "a zero-weight contribution must be a no-op"
        );

        // An all-zero contribution list must reduce to the additive identity.
        let all_zero: Vec<(u8, u16)> = contribs.iter().map(|&(c, _)| (c, 0)).collect();
        assert_eq!(
            premultiplied_over(&pal, &all_zero),
            [0i64; BASE_DIM],
            "an all-zero-weight contribution list must reduce to zero"
        );
    }

    #[test]
    fn premultiplied_over_empty_contribs_is_zero() {
        let pal = make_palette(8);
        assert_eq!(premultiplied_over(&pal, &[]), [0i64; BASE_DIM]);
    }
}
