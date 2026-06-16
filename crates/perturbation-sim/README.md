# perturbation-sim

Spectral + edge-propagation **outage perturbation-shape** simulator. Models the
footprint a cascading power-grid failure lights up on a topology view, by
composing the two halves of the eigenvalue-perturbation + edge-propagation
method:

| Half | What it does | Module |
|---|---|---|
| **Spectral perturbation** | A line trip is a rank-1 perturbation `E` of the weighted Laplacian `L` (`‖E‖₂ = 2·b_k`). Certifies **Weyl** `|λᵢ(L')−λᵢ(L)| ≤ ‖E‖₂`, reports **Davis–Kahan** Fiedler rotation `sinθ ≤ ‖E‖₂/gap`, and tracks **algebraic connectivity** `λ₂` (its drop toward 0 = fragmentation precursor). | `perturbation.rs`, `eigen.rs` |
| **Edge propagation** | DC power flow `θ = L⁺p`, `f_e = b_e(θ_a−θ_b)`; a trip redistributes flow, overloaded lines trip in turn (the cascade), recomputed exactly each round. | `flow.rs`, `cascade.rs`, `graph.rs` |
| **Basin / HHTL field tier** | Kron reduction (Schur complement — basin→super-node, cross-border), effective-resistance metric + spectral embedding (electrical Morton/HHTL coords), Cheeger sweep (`μ₂/2 ≤ h ≤ √(2μ₂)` — the field↔cut exchange rate), and the Go-meta `infight_vs_raumgewinn` regime. | `basin.rs` |
| **Fast-sketch synergy** (PROTOTYPE) | Spielman–Srivastava resistance sketch via random ±1 (`vsa_bundle`) projections + Walsh/Morton pyramid coarse↔fine collapse screen. The VSA/Hamming side of the field tier. | `sketch.rs` |
| **Gaussian-splat magnitude side** (PROTOTYPE) | anisotropic `Σ` fit to the electrical neighbourhood + EWA pyramid coarsen (Morton-seam anti-alias) + `morton2`. The magnitude algebra complementing the Walsh sign side. | `splat.rs` |
| **Data-shaped scoping** | run on today's data; `assess_capability` gates which outputs are valid; missing variables modeled as uniform constants (provably free for relative results); `AgeModel` = Uniform null vs DensityProxy Gegenhypothese (topology-only) vs ModernizationSpend (official planning data). | `model.rs` |

> **SIMD:** the Morton/Walsh pyramid transform optionally routes through
> `ndarray::simd::wht_f32` (AVX-512/AMX) via `--features ndarray-simd`
> (`RUSTFLAGS='-C target-cpu=x86-64-v4'`); default is the zero-dep scalar path.
> All SIMD comes from `ndarray::simd` (workspace rule). See `METHODS.md §10`.

> **Methods & math grounding:** see [`METHODS.md`](METHODS.md) — the one-operator
> grounding that connects all four, the anti-dilution distinctions (combinatorial
> `λ₂` vs normalized `μ₂`; geography vs electrical distance; infight vs
> Raumgewinn), and the statistics design (the four methods as a measurement
> battery → ICC / Pearson / Spearman / Cronbach, mutual control variables via
> partial correlation, Jirak-correct significance).

Output: [`PerturbationShape`] — a per-bus angle-deviation field + per-line
flow-shift field + the trip footprint (which lines tripped, in which round).

## Run it

```sh
cargo test  --manifest-path crates/perturbation-sim/Cargo.toml
cargo run   --manifest-path crates/perturbation-sim/Cargo.toml --example simulate
```

The example builds a 4×4 transmission lattice, stresses it to its limits, trips
the most-loaded line, and prints the spectral analysis + cascade + shape (e.g.
"Weyl HOLDS, connectivity loss 27%, 15/24 lines tripped, islanded into 7
components").

## Use it

```rust
use perturbation_sim::{Grid, Edge, simulate_outage, CascadeConfig};

let grid = Grid::new(n_buses, edges);          // edges carry susceptance + limit
let injections = /* balanced ∑p = 0 */;
let r = simulate_outage(&grid, &injections, seed_line, CascadeConfig::default());

r.spectral.weyl_satisfied;          // Weyl bound held
r.spectral.connectivity_loss();     // fractional λ₂ collapse
r.shape.node_field;                 // per-bus perturbation magnitude
r.shape.epicentre(3);               // top-3 buses by perturbation
r.islanded; r.components_final;     // did the grid fragment?
```

## Where it sits

Standalone, zero-dep, deterministic — the same proof-in-code pattern as
`crates/jc`, `crates/sigker`, `crates/helix` (excluded from the workspace;
build via `--manifest-path`). It is the **applied companion to `jc`** and closes
the gap noted in `ada-docs/research/JIRAK_MATH_THEOREMS_HARVEST.md`:

- `jc::weyl` is Hermann Weyl's **equidistribution** theorem (golden-ratio
  low-discrepancy sampling), **not** the eigenvalue-perturbation inequality.
  This crate supplies the genuine spectral-perturbation result.
- `jc::ewa_sandwich` is a covariance Σ-push-forward along multi-hop edge paths —
  the *uncertainty-propagation* sibling of the deterministic flow cascade here.

## Statistical hand-off

`shape.node_field` is a per-node magnitude vector ready to be correlated
(Pearson / Spearman / **ICC** via `ndarray::hpc::reliability`) against an
*observed* outage footprint — predicted-shape-vs-observed-shape validity.
Significance of any such correlation must use the **Jirak 2016** weak-dependence
rate `n^(p/2−1)`, not classical IID Berry–Esseen (the `I-NOISE-FLOOR-JIRAK`
doctrine).

## Honest scope

A **DC** model (linearized, lossless, reactance-only) — the standard
first-order contingency screen, not a full AC power-flow / transient-stability
solver. Islanding is detected (zero-eigenvalue multiplicity = component count)
and treated as terminal rather than fabricating per-island balanced flows.
Targets regional graphs (`n` up to a few hundred buses); the Jacobi eigensolver
is O(n³) per recompute.
