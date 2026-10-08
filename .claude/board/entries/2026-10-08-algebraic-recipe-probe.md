# 2026-10-08 — Algebraic recipe table × percentile profiles × Wankel phasors (D-ART-1)

**Status:** MEASURED. Probe only; no production code changes.
- **Probe:** `crates/lance-graph-mask-risc/examples/algebraic_recipe_probe.rs`.
- **Revision:** lance-graph `origin/main` 7f63ae9d.
- **Host:** Intel Xeon @ 2.8 GHz, 4 vCPU, `avx2` build, release, `debug = 0`, rustc 1.98.1. Medians of 5–31 runs.

```text
CARGO_PROFILE_RELEASE_DEBUG=0 cargo run --release -p lance-graph-mask-risc --example algebraic_recipe_probe
```

## Question

Given a concept (a shape family and its parameters), a chain of transformations and a requested terminal, how much geometric and trigonometric work can a small, guarded recipe table remove before anything is evaluated? And does it know when it cannot?

## Inventory (read, not assumed)

| item | state |
|---|---|
| `PowerSums`, `CrossPowerSums` (ndarray `simd_masking_ops.rs`, re-exported in `simd`) | **SHIPPED** as integer folds (`n`, `Σ`, `Σ²`, `Σxy`) with `checked_merge`. **MISSING:** any shift, scale or affine method; any mean/variance terminal on these types. |
| `MomentsU32` (ndarray `hpc/statistics.rs`) | **SHIPPED**, unsigned, with `mean`/`variance`. **SEMANTICALLY DIFFERENT** from `PowerSums`; not interchangeable. |
| Float terminals over power sums | **SHIPPED** consumer-side in lance-graph `crates/jc/src/stats.rs` (ANOVA, Pearson, covariance, regression). |
| `GroupPowerSumsI32`, `GroupCrossPowerSumsI32` | **SHIPPED** as mask-risc terminals (tiled path only). **PARTIALLY AVAILABLE** end to end: no Quack `Agg`/`GroupAgg` variant reaches them. |
| `RollingFloor`, `EmpiricalShape` (ndarray `hpc/rolling_floor.rs`) | **SHIPPED**, integer. The location-scale quantile transform `Q = μ + (σ/σ₀)(Q₀ − μ₀)`, Euclidean floor division, clamped to `u32`. Nearest-rank floor `rank_per_10000`; this probe uses that convention. |
| Arithmetic expression IR / algebraic simplifier | **MISSING.** Quack's typed IR is Boolean predicates only. The rewrites that exist (SQL three-valued logic, survivor-gate hoisting, ternlog fusion) are all Boolean. The planner's ten `OptimizerRule`s all return `Unchanged`. DataFusion is in its grace period. |
| Rotation / complex / phasor type | **MISSING** in ndarray `hpc` (only `GivensRotation`, no apply). D-PHT-1's LUT is a probe; ndarray PR #343 (`hpc::phase`) is open. |
| NARS inference recipes | **SEMANTICALLY INCOMPATIBLE.** Not touched; mathematical identities are not inference rules. |

## What the probe holds

- **Concepts:** a regular `m`-fold family `c + r·e^{i(θ + 2πk/m)}`, with either a Wankel centre `e·e^{i3θ}` or a fixed centre. No points are stored.
- **Canonicaliser:** one left fold of the chain.
  - Rotations stay exact `u32` turns; `R(a)R(b) = R(a + b)` is a wrapping add.
  - A negative similarity becomes a half turn plus a positive scale.
  - Translations stay lazy, each tagged with the rotation and scale in force; trigonometry is spent only if a terminal reads the translation.
  - The class drops back down `Affine ⊃ Similarity ⊃ Rotation ⊃ Identity` when scales multiply to 1 and rotations cancel.
  - It is linear in the chain length, so it terminates; there is no search and no cycle.
- **Recipe table:** `static [[[Rid; 9]; 4]; 2]` over (shape, linear class, terminal). The key only proposes. Guards (`m ≥ 3` for isotropy, class, translation present) run at evaluation, and a failed guard falls through to `Materialize`.
- **Oracle:** materialises the `m` vertices and applies the raw chain to each point, independently of the canonical form.

## Results

### 1. The Wankel concept under `R(+1/8) · T(5, −3) · R(−1/8) · S(2) · S(0.5)`

Canonical form: the identity plus one lazy translation (3 rewrites). The oracle spends 10 transcendental calls and materialises 3 vertices for every query.

| query | recipe | trig | vertices | remaining work |
|---|---|---|---|---|
| A count | Const | 0 | 0 | `3` |
| B Σ positions, centroid | CenterPhasor | 2 | 0 | the centre phasor `3θ`, plus one rotation of the lazy translation |
| C centred Σ\|z − c\|², centred Σ(x−x̄)(y−ȳ) | IsoInvariant | 0 | 0 | `3R²`, `0` |
| D Σ\|z\|², translated | OriginNorm | 2 | 0 | centred invariant + `3\|c'\|²` |
| D′ Σ\|z\|² under `R · S(−1.5)`, untranslated | OriginNorm | 0 | 0 | `3s²(R² + e²)` |
| E apex 1 | VertexPhasor | 3 | 0 | one vertex phasor |
| E′ x-quantile, p = 0.5 | SectorQuantile | 3 | 0 | centre, translation, one cosine |
| F collision with (110, 0), d = 5 | Materialize | 10 | 3 | refused: no summary decides it |

All equal the oracle.

### 2. Guards, 36,000 random (concept, chain, terminal) cases

- m ∈ {2, 3, 4, 5, 8, 16, 256}; chains of 0–4 rotations, signed scales, exact zero scales, translations and general linear maps.
- All equal the oracle within 1e-9 of the compared values plus an absolute floor of 1e-13 of the case's Σ|p|² + m (for results that cancel to zero).
- A translation applied after a zero scale survives it: `S(0) · T(5, −3)` has centroid (5, −3).

| recipe | cases | trig/case | oracle trig/case | exactness |
|---|---|---|---|---|
| Const | 4,000 | 0.17 | 51.4 | real identity |
| CenterPhasor | 8,000 | 0.99 | 51.4 | real identity |
| IsoInvariant | 7,000 | 0.16 | 58.2 | real identity |
| OriginNorm | 3,500 | 0.64 | 58.2 | real identity |
| SectorQuantile | 4,000 | 1.99 | 51.4 | real identity |
| VertexPhasor | 4,000 | 2.19 | 51.4 | real identity |
| Materialize | 5,500 | 38.5 | 38.4 | reference |

The nonzero trig on Const and IsoInvariant is the canonicaliser leaving the similarity form (a general linear map materialises `L` once).

**Refusals, measured:**
- `m = 2` under `diag(2, 1)`: the isotropic formula gives 500.00, the truth is 777.16. The `m ≥ 3` guard refuses, and the recipe falls back to `Materialize`.
- A shear `[1 1.5; 0 1]` gives Σ|z − c|² = 637.50, not the rotation invariant 300.00. The class is `Affine`, and `‖L‖_F²` carries it.

### 3. Populations: affine chains over `CrossPowerSums`

T3∘T2∘T1 integer affine maps (composed `A = [0 −2; −1 −2]`, `t = (20, 4)`). Coordinates are in [−100, 100].

| rows | A: materialise each step + ndarray fold | A2: composed map per row + scalar fold | B: ndarray fold + 3 summary steps | C: ndarray fold + 1 composed step |
|---|---|---|---|---|
| 1,000 | 12.0 | 3.6 | 4.8 | 4.8 |
| 65,536 | 18.8 | 3.7 | 4.9 | 4.8 |
| 1,000,000 | 20.5 | 3.9 | 5.4 | 5.7 |

ns per row. All four are **bitwise equal** (`i128`). A warm composed summary step costs **13.7 ns**, independent of the row count.

- **f64:** the same identity is not bitwise. Max relative difference 1.5e-14; 1 of 6 fields bitwise equal.
- **Moments are not sufficient:** {0, 3, 3} and {1, 1, 4} on `y = 0` share `n, Σx, Σy, Σx², Σy², Σxy` = (3, 6, 0, 18, 0, 0). Their maxima are 3 vs 4, and a collision with (4, 0) is false vs true. The table maps Population × {quantile, vertex, collide} to `Materialize` for every class.
- **Group keys:** with the key `x ≥ 0` and the map `x + 50`, pushing the fold below the transform is wrong (group sizes 2,052 vs 3,059). With a key on an untouched column (id parity), it is exact.

### 4. One projected order statistic per instance

**Symmetry, measured.**
- `Q(θ + 1/m) = Q(θ) = Q(−θ)` to 2.6e-15 on 20,000 phases.
- Ties between `cos(θ + 2πk/m)` occur only at `θ ≡ 0 mod 1/(2m)` turn. So inside each half-sector the rank → vertex map is **fixed**, and a phase-conditioned percentile profile collapses to two permutations of `m` entries plus one cosine (arm E).
- 1/3 turn is not representable in `u32` turns (2^32 mod 3 = 1). Threefold symmetry holds in reals, and to one ulp-turn in `u32`.

**Arms.** Error is the max |error| on R = 100 over 200,003 phases × every one of the `m` ranks (the oracle sorts all `m` projections once per phase). Timings are ns per instance at 1M instances; 65,536 instances agree within 10 %.

| m | A generate + select | B analytic (m = 3) | C profile LUT, 64 buckets | E sector permutation + 1 `cos` | C table | E table | C error, 64 buckets | C error, 63 buckets | E error |
|---|---|---|---|---|---|---|---|---|---|
| 3 (Wankel) | 108 | 39 | 34 | 50 | 780 B | 24 B | 1.3e-2 | **1.4** | 2.0e-13 |
| 16 | 383 | — | 10.6 | 42 | 4.2 KB | 128 B | 4.7e-4 | 3.1e-1 | 2.1e-13 |
| 256 | 5,037 | — | 10.9 | 42 | 66.6 KB | 2 KB | 7.7e-6 | 1.9e-2 | 2.0e-13 |

- B is one `sin_cos`, ±120° by constants, `cos 3θ = 4c³ − 3c`, and a three-element sorting network.
- E's time is two `f64` cosines: the value and the Wankel or fixed centre.
- A query that needs no profile (Σ|z − c|²) costs 1.2–3 ns.

### 5. Lookup and rewrite cost

- **Recipe lookup over 4,096 random keys:** static array 1.2 ns, `match` 2.0 ns, `HashMap` (SipHash) 27 ns.
- A `HashMap` whose hasher sends every key to one bucket still returns the right recipe (106 ns): structural key equality, not the hash, decides.
- **8-step chain:** canonicalising costs 49–72 ns; canonicalise + lookup + evaluate Σ|z − c|² costs 62–98 ns, against 301–496 ns for the oracle (3 vertices × 8 raw steps). Ranges are two runs.

### 6. 1-D affine quantile

`Q(aX + b)` is `b + a·X[rank]` for `a ≥ 0` and `b + a·X[n − 1 − rank]` for `a < 0`. It equals the sorted oracle on 2,000 random cases.

## Disable runs

Each disable ran after the commit, turned red, and was restored afterwards.

| disable | effect |
|---|---|
| isotropy guard `m ≥ 3` → `true` | case 0, `m = 2` under a linear map: recipe ≠ oracle |
| negative scale not folded into a half turn | case 6, `m = 16`, Σ positions sign-flipped |
| 1-D quantile rank reversal for `a < 0` dropped | `a −2, b 3, p 2270` ≠ sorted oracle |
| sector permutation without the half-sector split | case 3, `m = 4` quantile ≠ oracle |
| Population × quantile mapped to a summary | the table assertion fires |
| `‖L‖_F²` of a general map treated as a similarity's | case 2, `m = 3` Σ\|z − c\|² 43,271 vs 136,940 |
| translation multiplier not updated by later scales | case 11, `T · S(0) · R`: Σ positions ≠ oracle |
| the first version's `scale / scale_at` translation factor | case 14, `S(0) · T`: Σ positions (0, 0) vs the oracle's translated sum |

## Findings

1. **The terminal decides the representation, and most terminals need no vertices.**
   - Count, centred second moments and an untranslated Σ|z|² need no phase at all.
   - Sum and centroid need one centre phasor; a vertex needs one phasor; an order statistic needs one cosine.
   - Only collision falls back to materialisation, and the table refuses it rather than approximating.
2. **The centred second moment of any regular family with `m ≥ 3` is `(m r²/2) · ‖L‖_F²` for every linear map `L`, not only rotations.**
   - It is phase-free even under shear.
   - `m = 2` is a rank-one dyad, and the guard is load-bearing.
3. **The percentile profile is two permutations, not a table.**
   - Ranks only change at `1/(2m)` turn, so the phase-conditioned profile reduces to a fixed vertex per (half-sector, rank).
   - A bucketed LUT is fast (10 ns) but must align its bucket edges with the tie points: 63 buckets put a kink inside a bucket and the error jumps from 1e-2 to 1.4 (m = 3).
   - For m = 3 no table pays: the analytic arm B is within 1.15× of the LUT, and exact.
4. **Summary transforms are exact and O(1), but the fold of the base population is the floor.**
   - Folding once and transforming the summary removes every materialisation (2.5–4× against A).
   - A fused per-row composed transform with a scalar fold (A2, 3.7 ns/row) is faster than the shipped grouped `masked_group_cross_power_sums_i32` without a transform (4.8 ns/row).
   - The recipe's real value is the warm path: one fold, any number of affine queries at 14 ns each.
5. **Hashing costs more than many of the recipes it selects.** Σ|z − c|² is one multiply-add, and a SipHash lookup is 27 ns. A static array indexed by enum discriminants is 1.2 ns and cannot collide.

## Recommendations

| item | verdict | correctness contract | measurement |
|---|---|---|---|
| Affine transform of `CrossPowerSums` (`S' = AS + nt`, `M' = AMAᵀ + AStᵀ + tSᵀAᵀ + nttᵀ`), checked `i128` | **ADOPT** (a small ndarray PR, as a method on `CrossPowerSums`) | bitwise equal to materialise-and-fold within the documented widths; refuses when the group key reads a transformed column | 14 ns per summary step against 5–20 ns per row |
| Isotropic invariant `(m r²/2)·‖L‖_F²` for regular families | **ADOPT as a recipe** where such a concept exists | real identity; `m ≥ 3` guard | 0 trig against 56–64 for the oracle |
| Sector-permutation order statistic | **ADOPT as the method** for projected order statistics of `m`-fold families | exact up to `f64` rounding (2e-13) | 42 ns, flat in `m`; A is 5 µs at m = 256 |
| Sector permutation with the phase LUT from ndarray #343 instead of `f64` `cos` | **PROBE** | the LUT's stated bound times `r·ρ` | expected ≈ C's 10 ns; not measured |
| Phase-conditioned profile LUT | **PROBE** for m ≥ 16 only, bucket count a multiple of 2 per sector | bounded; edges must sit on the tie points | 10 ns, 4–67 KB |
| Profile LUT for the Wankel triangle (m = 3) | **REJECT** | — | the analytic arm is within 1.15× and exact |
| Static recipe table + linear canonicaliser as a production layer | **PARK** | — | no arithmetic expression IR exists to host it; building one is not justified until a consumer asks concept + terminal queries |
| `HashMap`-keyed recipe lookup | **REJECT** | — | 27 ns against 1.2 ns |
| e-graph or search-based rewriting | **REJECT** for this rule set | — | a single linear fold terminates and is stable |
| `RollingFloor` / `EmpiricalShape` for deterministic geometry | **REJECT the role** | — | static geometry needs no statistical frame |
| `RollingFloor` / `EmpiricalShape` for learned, drifting profiles | **PARK** | — | the location-scale transform matches, but no learned-shape workload exists here |
| NARS inference recipes | untouched | — | — |

## OPEN

- One host class only.
- The fused arm A2 beating the shipped grouped fold is one measurement; whether the group addressing or the mask walk costs the difference is not isolated.
- Multiple quantiles per instance (all `m` order statistics) were not timed. E then needs `m` cosines and the permutations, but no sort.
- Reflection symmetry of a projection along a general direction (the affine `β`) costs one `atan2`. It is counted, not removed.
