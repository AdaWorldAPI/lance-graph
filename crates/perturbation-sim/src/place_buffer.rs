//! Location / impulse-permeability split — the operator's buffer-substrate
//! contract (2026-06-24).
//!
//! `cascade_key` (V1/V2/V3) derives `place` from the **live spectral embedding**,
//! which IS the Laplacian impulse-response (effective resistance / Fiedler). So
//! "location" was **conflated with impulse permeability** — it flips globally
//! under any line trip (the substrate became the ketchup yield it measures;
//! measured: absolute-cell ICC ≈ 0.14, every bus flips). The conflation re-fused
//! two axes #509/#511 had measured orthogonal (`Spearman(λ₂, inertia) ≈ 0`).
//!
//! The fix is two orthogonal axes:
//!
//! - **LOCATION** — the helix **Place** convention: equal-area golden-spiral
//!   placement (`r = √u`, `θ = i·golden-angle` — the `helix::HemispherePoint`
//!   math) addressed by a **stable node index**. Pure geometry; it never reads
//!   the Laplacian, so it cannot carry the dynamics, is deterministic, and is
//!   perturbation-invariant by construction. (Inlined to keep `perturbation-sim`
//!   zero-dep — `crates/helix` is the canonical codec but carries a mandatory
//!   `ndarray` dependency, disproportionate for the √u formula. A future
//!   `helix-place` feature swaps in `helix::HemispherePoint` verbatim.)
//! - **BUFFER** — the BF16 residue: the 3×3 Moore-stencil (8-neighbour) impulse
//!   **permeability** (conductance to the 8 nearest buses), live, recomputed per
//!   perturbation. The responsive axis where the ketchup yield is the SIGNAL,
//!   orthogonal to location. Each Umspannwerk is modelled as the `[3×3]` stencil
//!   the operator drew (centre = the located node, the 8 cells = the couplings).
//!
//! Measured (`examples/location_buffer_split.rs`, 24-bus 4-region grid, 36
//! line-trip perturbations): location ICC **0.14 → 1.00** (deterministic — stable
//! identity), location ρ-vs-R_eff **0.46 → ≈0** (it is *not* the dynamics — correct
//! demotion), buffer ICC **0.51** (responsive — it moves; its motion is the
//! ketchup signal). (The buffer is a *node-summary* permeability, so comparing it
//! *pairwise* against R_eff — a pairwise coupling — is the wrong shape and gives a
//! meaningless ≈0; the buffer's role is shown by its motion under perturbation,
//! not a pairwise ρ.)

use crate::basin::effective_resistance;
use crate::splat::morton2;

/// The golden angle `π(3 − √5)` — the Vogel/sunflower spiral increment used by
/// the helix `HemispherePoint` placement (equidistributing, Weyl-low-discrepancy).
pub const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653_3;

/// The **helix Place** of node `index` of `n`: equal-area golden-spiral placement
/// (`r = √((i+½)/n)`, `θ = i·GOLDEN_ANGLE`) on the unit disc, quantized to a
/// 24-bit Morton cell → 3 place octets (coarse→fine, the HEEL/HIP/TWIG high
/// bytes). A pure function of `(index, n)` — it never touches the grid, so it is
/// deterministic, graph-independent, and perturbation-invariant: location is
/// *just* location.
pub fn helix_place(index: usize, n: usize) -> [u8; 3] {
    let n = n.max(1);
    let r = (((index as f64) + 0.5) / n as f64).sqrt(); // equal-area √u
    let th = index as f64 * GOLDEN_ANGLE;
    let x = (0.5 + 0.5 * r * th.cos()).clamp(0.0, 1.0);
    let y = (0.5 + 0.5 * r * th.sin()).clamp(0.0, 1.0);
    let xi = (x * 4095.0).round() as u16; // 12-bit per axis
    let yi = (y * 4095.0).round() as u16;
    let m = morton2(xi, yi); // 24-bit Z-order
    [(m >> 16) as u8, (m >> 8) as u8, m as u8]
}

/// `f32 → bf16` bits: the standard truncated-to-top-16-bits view (round-toward-
/// zero on the mantissa). Lossy by 8 mantissa bits; deterministic.
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// `bf16 bits → f32` (zero-extend the dropped mantissa). Inverse of
/// [`f32_to_bf16`] up to the truncated low bits.
#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// One node's impulse-permeability buffer: BF16 conductance (`1 / R_eff`) to its
/// 8 nearest buses — the 3×3 Moore-stencil couplings the operator drew. This is
/// the **live dynamical axis** (recompute per perturbation); higher conductance =
/// more permeable = the perturbation flows there. Stored as 8 BF16 lanes (one 3×3
/// stencil); a real model stacks `× slots` for {susceptance, flow, angle-Δ,
/// inertia} — the same 8-lane shape per quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferResidue {
    /// 8 BF16 conductances to the nearest neighbours (descending permeability),
    /// zero-padded if the grid has < 9 buses.
    pub lanes: [u16; 8],
}

impl BufferResidue {
    /// Mean permeability (decoded f32) across the live lanes — a scalar summary of
    /// the node's coupling to its neighbourhood.
    pub fn mean_permeability(self) -> f32 {
        let (mut s, mut c) = (0.0f32, 0u32);
        for l in self.lanes {
            if l != 0 {
                s += bf16_to_f32(l);
                c += 1;
            }
        }
        if c == 0 {
            0.0
        } else {
            s / c as f32
        }
    }
}

/// Compute node `node`'s [`BufferResidue`] from the Laplacian pseudo-inverse:
/// effective resistance to every other bus, take the 8 nearest, store BF16
/// conductance (`1/(R+ε)`) descending. `l_plus` is `laplacian_pinv(grid, alive, …)`.
pub fn buffer_residue(l_plus: &[f64], n: usize, node: usize) -> BufferResidue {
    let mut d: Vec<f64> = (0..n)
        .filter(|&j| j != node)
        .map(|j| effective_resistance(l_plus, n, node, j))
        .collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut lanes = [0u16; 8];
    for (k, &r) in d.iter().take(8).enumerate() {
        lanes[k] = f32_to_bf16((1.0 / (r + 1e-9)) as f32); // permeability = conductance
    }
    BufferResidue { lanes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basin::laplacian_pinv;
    use crate::graph::{Edge, Grid};

    fn two_block_bridge() -> Grid {
        let mut e = Vec::new();
        for (a, b) in [(0, 1), (0, 2), (1, 3), (2, 3)] {
            e.push(Edge::new(a, b, 1.0, 1.0));
        }
        for (a, b) in [(4, 5), (4, 6), (5, 7), (6, 7)] {
            e.push(Edge::new(a, b, 1.0, 1.0));
        }
        e.push(Edge::new(3, 4, 0.01, 1.0)); // weak bridge
        Grid::new(8, e)
    }

    #[test]
    fn location_is_deterministic_and_graph_independent() {
        // The helix place of a node depends ONLY on (index, n) — never on the
        // grid. So it is perturbation-invariant by construction: location is just
        // location, it cannot carry the dynamics.
        assert_eq!(helix_place(3, 8), helix_place(3, 8));
        assert_ne!(
            helix_place(3, 8),
            helix_place(4, 8),
            "distinct indices place apart"
        );
        // Same index/n ⇒ same place whether or not any line is "tripped" — there
        // is no grid argument, so nothing dynamical can leak in.
    }

    #[test]
    fn bf16_round_trips_within_truncation() {
        for &x in &[0.0f32, 1.0, 0.5, 12.34, 1e-3, 9876.0] {
            let r = bf16_to_f32(f32_to_bf16(x));
            let tol = x.abs() * (1.0 / 128.0) + 1e-6; // 7-bit mantissa
            assert!((r - x).abs() <= tol, "bf16 {x} -> {r}");
        }
    }

    #[test]
    fn buffer_is_the_dynamical_axis_it_moves_when_a_line_trips() {
        // The buffer (permeability) is SUPPOSED to change under perturbation —
        // that motion is the signal. Drop the bridge and the cross-block
        // permeability collapses.
        let g = two_block_bridge();
        let alive = vec![true; g.edges.len()];
        let lp0 = laplacian_pinv(&g, &alive, 1e-12);
        let b0 = buffer_residue(&lp0, g.n, 3); // bus 3 touches the bridge

        let mut a2 = alive.clone();
        a2[8] = false; // trip the weak bridge (last edge)
        let lp1 = laplacian_pinv(&g, &a2, 1e-12);
        let b1 = buffer_residue(&lp1, g.n, 3);

        assert_ne!(
            b0.lanes, b1.lanes,
            "buffer must move when the topology is perturbed (it is the dynamics)"
        );
        // ...while the location of bus 3 is identical (it never saw the grid):
        assert_eq!(helix_place(3, g.n), helix_place(3, g.n));
    }
}
