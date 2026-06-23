//! Full **16-bit-per-tier** spatial cascade key — the OGAR production form of
//! the HHTL address (`OGAR/CLAUDE.md` P0; `ndarray .../guid-prefix-shape-routing.md`
//! §4 "the key selects the grid").
//!
//! [`crate::hhtl::HhtlKey`] derives the cascade address by **binary** Cheeger
//! bisection — its own doc note is honest that this only fills the *low bit* of
//! each tier (an 8-leaf tree), "**not** that full encoding". This module realizes
//! the deferred form: each cascade tier is a full **16-bit 256×256 centroid tile**
//! (two byte-axes, nibble-interleaved Morton — [`crate::splat::morton2`]), built
//! from the bus's position in the [`spectral_embedding`] (electrical coordinates,
//! NOT geography — "topology IS the key"). Three tiers ⇒ a 24-bit-per-axis Morton
//! address over the spectral plane, perfectly aligned with the cascade.
//!
//! The three 16-bit roles map coarse→fine onto the canonical GUID cascade tiers:
//!
//! | role         | bits | GUID tier | meaning                                  |
//! |--------------|------|-----------|------------------------------------------|
//! | [`family`]   | 16   | **HEEL**  | coarsest 256×256 tile — the broad basin  |
//! | [`leaf`]     | 16   | **HIP**   | mid tile — the leaf basin in the family  |
//! | [`identity`] | 16   | **TWIG**  | finest tile — the per-bus identity cell  |
//!
//! [`family`]: CascadeKey::family
//! [`leaf`]: CascadeKey::leaf
//! [`identity`]: CascadeKey::identity
//!
//! ## The six lenses this one key proves
//!
//! - **location** — [`CascadeKey::tile`] decodes the key back to the quantized
//!   spectral tile the bus sits in (the address *is* the position).
//! - **math** — [`CascadeKey::cascade_distance`] is Morton-prefix containment:
//!   `3 − shared_prefix_tiers`, an O(1) tier compare, no value decode.
//! - **learning** — the blackout footprint is **prefix-local**: co-failing buses
//!   share a `family`/`leaf` prefix (proven in the `learning_*` test against a
//!   random baseline), so field-perturbation placement *learns* the basin tree.
//! - **representation** — [`CascadeKey::to_guid_tiers`] is exactly the canonical
//!   `(HEEL, HIP, TWIG)` `u16` triple; the dash-groups are the only semantics.
//! - **substrate** — three `u16` are bit-for-bit the canonical `NodeGuid` cascade
//!   tiers at byte offsets 4..6 / 6..8 / 8..10; [`CascadeKey::morton48`] is the
//!   packed key the SoA node carries with zero value decode.
//! - **thinking** — the outage cascade ([`crate::cascade::simulate_outage`])
//!   traverses the same key arithmetic: its epicentre is a low-`cascade_distance`
//!   neighbourhood.

use crate::basin::spectral_embedding;
use crate::graph::Grid;
use crate::splat::morton2;

/// A node's full-resolution HHTL cascade address: three 16-bit centroid tiles,
/// coarse→fine. Each tier interleaves one byte of each spectral axis (a 256×256
/// tile); the whole key is a 48-bit Morton code over the 2-D embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CascadeKey {
    /// **HEEL** — coarsest 256×256 tile: the broad basin / family.
    pub family: u16,
    /// **HIP** — mid 256×256 tile: the leaf basin within the family.
    pub leaf: u16,
    /// **TWIG** — finest 256×256 tile: the per-bus identity cell.
    pub identity: u16,
}

/// Interleave one byte of each axis into a 16-bit centroid tile (the per-tier
/// 256×256 Morton code). `morton2` of two ≤255 values fits in the low 16 bits.
#[inline]
fn tile(x_byte: u8, y_byte: u8) -> u16 {
    morton2(x_byte as u16, y_byte as u16) as u16
}

/// Map a normalized coordinate `t ∈ [0,1]` to a 24-bit axis index `[0, 2^24)`.
/// Linear (not rank) so a nibble prefix = a quad-tree quadrant — the
/// `256 = 4⁴` hierarchical-ancestry condition (`guid-prefix-shape-routing.md` §6).
#[inline]
fn axis24(t: f64) -> u32 {
    // (1<<24) - 1 keeps t==1.0 in-range (closed interval, no overflow to bit 24).
    (t.clamp(0.0, 1.0) * ((1u32 << 24) - 1) as f64).round() as u32
}

impl CascadeKey {
    /// Build the key from a bus's spectral coordinate `(u, v)` given the per-axis
    /// `(min, max)` bounds of the embedding. Coarsest tier = the top byte of each
    /// 24-bit axis index; finest = the low byte.
    pub fn from_spectral(u: f64, v: f64, ub: (f64, f64), vb: (f64, f64)) -> Self {
        let norm = |x: f64, (lo, hi): (f64, f64)| {
            let w = hi - lo;
            if w.abs() < 1e-300 {
                0.0
            } else {
                (x - lo) / w
            }
        };
        let xi = axis24(norm(u, ub));
        let yi = axis24(norm(v, vb));
        CascadeKey {
            family: tile((xi >> 16) as u8, (yi >> 16) as u8),
            leaf: tile((xi >> 8) as u8, (yi >> 8) as u8),
            identity: tile(xi as u8, yi as u8),
        }
    }

    /// The canonical `(HEEL, HIP, TWIG)` `u16` triple — the cascade tiers a
    /// `NodeGuid` stores at byte offsets 4..6 / 6..8 / 8..10. The key IS the GUID
    /// cascade path; this is the representation lens.
    #[inline]
    pub fn to_guid_tiers(self) -> (u16, u16, u16) {
        (self.family, self.leaf, self.identity)
    }

    /// The 48-bit Morton code `family<<32 | leaf<<16 | identity` — the packed
    /// cascade key the SoA node carries (prerenders/routes with zero value
    /// decode). The substrate lens.
    #[inline]
    pub fn morton48(self) -> u64 {
        ((self.family as u64) << 32) | ((self.leaf as u64) << 16) | (self.identity as u64)
    }

    /// How many tiers agree from the coarsest down (0..=3): `family`, then
    /// `leaf`, then `identity`. Stops at the first divergence (prefix property).
    #[inline]
    pub fn shared_prefix_tiers(self, other: Self) -> u8 {
        if self.family != other.family {
            0
        } else if self.leaf != other.leaf {
            1
        } else if self.identity != other.identity {
            2
        } else {
            3
        }
    }

    /// Morton-containment cascade distance: `3 − shared_prefix_tiers`. `0` =
    /// identical cell, `3` = diverge at the coarsest tier. O(1), zero value
    /// decode — the math lens.
    #[inline]
    pub fn cascade_distance(self, other: Self) -> u8 {
        3 - self.shared_prefix_tiers(other)
    }

    /// The quantized spectral tile this key addresses, as `(x24, y24)` axis
    /// indices in `[0, 2^24)` — decode of [`from_spectral`](Self::from_spectral)'s
    /// placement (the location lens). Inverse of the per-tier Morton interleave.
    pub fn tile(self) -> (u32, u32) {
        // De-interleave a 16-bit tile back to its two byte-axes.
        fn unmorton(t: u16) -> (u8, u8) {
            fn compact(mut v: u32) -> u8 {
                v &= 0x5555_5555;
                v = (v | (v >> 1)) & 0x3333_3333;
                v = (v | (v >> 2)) & 0x0F0F_0F0F;
                v = (v | (v >> 4)) & 0x00FF_00FF;
                v = (v | (v >> 8)) & 0x0000_FFFF;
                v as u8
            }
            let t = t as u32;
            (compact(t), compact(t >> 1))
        }
        let (fx, fy) = unmorton(self.family);
        let (lx, ly) = unmorton(self.leaf);
        let (ix, iy) = unmorton(self.identity);
        let x = ((fx as u32) << 16) | ((lx as u32) << 8) | ix as u32;
        let y = ((fy as u32) << 16) | ((ly as u32) << 8) | iy as u32;
        (x, y)
    }
}

/// Assign every bus its full 16-bit-per-tier [`CascadeKey`] from the 2-D spectral
/// embedding of the live grid. Deterministic: the embedding is a pure function of
/// the Laplacian spectrum, and the Morton quantization is fixed — same grid ⇒ same
/// keys (the representation lens). Min/max normalization makes the address fill
/// the cascade space, so spectral adjacency becomes a shared prefix.
pub fn cascade_keys(grid: &Grid, alive: &[bool]) -> Vec<CascadeKey> {
    let emb = spectral_embedding(grid, alive, 2);
    if emb.is_empty() {
        return Vec::new();
    }
    let axis = |k: usize| {
        emb.iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), e| {
                let x = e.get(k).copied().unwrap_or(0.0);
                (lo.min(x), hi.max(x))
            })
    };
    let ub = axis(0);
    let vb = axis(1);
    emb.iter()
        .map(|e| {
            CascadeKey::from_spectral(
                e.first().copied().unwrap_or(0.0),
                e.get(1).copied().unwrap_or(0.0),
                ub,
                vb,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cascade::{simulate_outage, CascadeConfig};
    use crate::graph::{Edge, Grid};

    /// Three weakly-bridged 4-cliques — a clean 3-basin transmission topology
    /// (the regional-grid regime this crate targets). Seeded faults stay local.
    fn three_region_grid() -> Grid {
        let mut e = Vec::new();
        for blk in 0..3 {
            let b = blk * 4;
            for (a, c) in [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3)] {
                e.push(Edge::new(b + a, b + c, 1.0, 1e6));
            }
        }
        // weak inter-region tie-lines
        e.push(Edge::new(3, 4, 0.01, 1e6));
        e.push(Edge::new(7, 8, 0.01, 1e6));
        Grid::new(12, e)
    }

    fn keys() -> Vec<CascadeKey> {
        let g = three_region_grid();
        cascade_keys(&g, &vec![true; g.edges.len()])
    }

    #[test]
    fn representation_is_deterministic() {
        // Same grid ⇒ same keys: the address is a pure function of the spectrum.
        assert_eq!(keys(), keys());
    }

    #[test]
    fn location_tile_round_trips_the_morton_interleave() {
        // Each tier de-interleaves back to its byte-axes; morton48 packs the
        // canonical (HEEL,HIP,TWIG) at 32/16/0 bit-exact (substrate layout).
        let k = CascadeKey::from_spectral(0.3, 0.7, (0.0, 1.0), (0.0, 1.0));
        let (x, y) = k.tile();
        // 0.3*16777215 ≈ 5033164 (0x4CCCCC); 0.7 ≈ 11744150 (0xB33333).
        assert_eq!(x >> 16, 0x4C, "top byte of x-axis recovered");
        assert_eq!(y >> 16, 0xB3, "top byte of y-axis recovered");
        let (h, hp, t) = k.to_guid_tiers();
        assert_eq!(
            k.morton48(),
            ((h as u64) << 32) | ((hp as u64) << 16) | t as u64
        );
    }

    #[test]
    fn math_distance_is_a_prefix_metric() {
        let a = CascadeKey {
            family: 7,
            leaf: 3,
            identity: 1,
        };
        assert_eq!(a.cascade_distance(a), 0); // identical cell
        assert_eq!(
            a.cascade_distance(CascadeKey {
                family: 7,
                leaf: 3,
                identity: 9
            }),
            1 // share family+leaf, differ at identity
        );
        assert_eq!(
            a.cascade_distance(CascadeKey {
                family: 7,
                leaf: 8,
                identity: 1
            }),
            2 // share family, differ at leaf
        );
        assert_eq!(
            a.cascade_distance(CascadeKey {
                family: 9,
                leaf: 3,
                identity: 1
            }),
            3 // diverge at the coarsest tier
        );
    }

    #[test]
    fn representation_distinct_basins_get_distinct_addresses() {
        // The three electrically-separated cliques do not all collapse to one
        // address — the embedding spreads them across the cascade space.
        let k = keys();
        let distinct: std::collections::HashSet<_> = k.iter().map(|c| c.morton48()).collect();
        assert!(
            distinct.len() >= 3,
            "≥3 distinct cascade addresses for 3 basins, got {}",
            distinct.len()
        );
    }

    #[test]
    fn learning_blackout_footprint_is_prefix_local() {
        // THE learning lens: a seeded outage's epicentre clusters in the cascade
        // tree — the impacted buses share a coarse prefix, so their mean pairwise
        // cascade_distance is strictly below the all-pairs (random) baseline.
        // Placement LEARNS the basin tree; the cascade traverses the same key.
        let g = three_region_grid();
        let k = cascade_keys(&g, &vec![true; g.edges.len()]);
        // Inject power at region-0, draw at region-2 → flow stresses the ties.
        let mut p = vec![0.0; g.n];
        p[0] = 1.0;
        p[10] = -1.0;
        let res = simulate_outage(
            &g,
            &p,
            g.edges.len() - 1, // trip a tie-line as the seed
            CascadeConfig {
                overload_factor: 1.0,
                max_rounds: 16,
                rel_tol: 1e-12,
            },
        );
        let epi: Vec<usize> = res.shape.epicentre(4).into_iter().map(|(b, _)| b).collect();
        assert!(epi.len() >= 2, "need an epicentre to measure locality");

        let mean_dist = |buses: &[usize]| {
            let mut s = 0u32;
            let mut n = 0u32;
            for i in 0..buses.len() {
                for j in (i + 1)..buses.len() {
                    s += k[buses[i]].cascade_distance(k[buses[j]]) as u32;
                    n += 1;
                }
            }
            if n == 0 {
                0.0
            } else {
                s as f64 / n as f64
            }
        };
        let all: Vec<usize> = (0..g.n).collect();
        let epi_local = mean_dist(&epi);
        let baseline = mean_dist(&all);
        assert!(
            epi_local < baseline,
            "epicentre cascade-distance {epi_local} must beat the random baseline {baseline} \
             (the footprint is prefix-local — placement learns the tree)"
        );
    }
}
