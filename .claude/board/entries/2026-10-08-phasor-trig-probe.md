# 2026-10-08 — Phasors without transcendental calls: sin_cos vs phase LUT vs CORDIC vs recurrence (D-PHT-1)

**Status:** MEASURED. Probe only, no production primitive.
- **Probe:** `crates/lance-graph-mask-risc/examples/phasor_trig_probe.rs`.
- **Revision:** lance-graph `origin/main` d127f7dc.
- **Host:** Intel Xeon @ 2.8 GHz, 4 vCPU.
- **Build:** `x86-64-v3`, release, `debug = 0`, rustc 1.98.1.
- **Timings:** medians of 5 runs.

```text
CARGO_PROFILE_RELEASE_DEBUG=0 cargo run --release -p lance-graph-mask-risc --example phasor_trig_probe
```

## Question

Can two observables that need only phasors avoid every transcendental call in the hot path?

1. **Wankel apex coordinates:** `z_k(θ) = e·e^{i3θ} + R·e^{i(θ + 2πk/3)}`. This is the ideal apex triangle, not the rotor flanks and not sealing.
2. **Coherent interference:** `I(P) = |Σ A_i e^{iφ_i(P)}|²`, with an amplitude-bounded exact early exit.

The phase is a `u32` in turns, so `3θ` is `phase.wrapping_mul(3)` and wraparound is free.

## Inventory

- **MISSING:** CORDIC, a phase LUT, or a SIMD `sin`/`cos` anywhere in ndarray or lance-graph.
- **SHIPPED, but not what it looks like:** `ndarray::hpc::vml::vscos`/`vssin` exist, but they call scalar `cos`/`sin` per lane inside an `F32x16` wrapper (`hpc/vml.rs`). That is not a SIMD transcendental.

## Wankel invariants (f64 reference)

All hold, checked on 100,000 angles in ±1000 rad, for (R, e) ∈ {(100, 14), (100, 1e-9), (100, 0), (1, 0.15)}:
- every apex lies on the housing curve at `θ + 2πk/3`;
- the triangle side stays `R√3`;
- the rotor centre stays at distance `e`;
- threefold symmetry holds.

Maximum error is 5.8e-12 for R = 100. The 3:1 law as `wrapping_mul(3)` equals `3θ mod 2π` on 100,000 phases.

## Results

### Wankel apex: 1M random phases

ns per apex; error in housing units, with R = 100.

| arm | ns | max error | table |
|---|---|---|---|
| native f64 `sin_cos` ×2 | 66.8 | reference | — |
| native f32 `sin_cos` ×2 | 25.4 | 5.1e-5 | — |
| **L12 LUT, nearest** | **3.1** | 8.7e-2 | 32 KB |
| L16 LUT, nearest | 4.0 | 5.5e-3 | 512 KB |
| **L12i LUT, linear interpolation** | **5.6** | 4.6e-5 | 32 KB |
| L10i LUT, linear interpolation | 5.5 | 5.5e-4 | 8 KB |
| CORDIC Q30, 24 iterations | 258 | 1.4e-5 | 192 B |
| ndarray `vml` (4 calls) | 50.9 | 3.4e-5 | materialises 24 MB of arrays |

The 64K run gives the same ordering: f64 61.9, f32 25.2, L12 2.5, L12i 5.8, `vml` 45.6.

### Trajectory, 1M equal steps

| arm | max error |
|---|---|
| f64 complex recurrence, never renormalised | 4.8e-9 |
| f32 recurrence, never renormalised | 1.95 |
| f32 recurrence, renormalised every 1,024 steps | 1.2e-2 |
| integer `u32` phase accumulator + L12i | no drift by construction; 3.8e-5, the table's own error |

The integer step differs from the requested Δθ by 5.7e-10 rad. That is a frequency choice, not accumulation.

### Two waves: 65,536 detectors

| arm | ns | error / peak |
|---|---|---|
| two `sin_cos` + `\|Σ\|²` | 80.7 | reference |
| one `cos(Δφ)`: `I = A1² + A2² + 2A1A2 cos Δφ` | 28.9 | 1.7e-14 (an identity) |
| one L12i lookup of Δφ | 13.4 | 1.7e-7 |

### 1,000 sources × 65,536 detectors

ns per (source, detector) term:

| arm | ns | max \|ΔI\| / peak I |
|---|---|---|
| native f64 `sin_cos` | 45.9 | reference |
| native f32 `sin_cos` | 36.4 | 3.5e-6 |
| **L12 LUT, nearest** | **9.95** | 4.5e-4 |
| L10i LUT, linear interpolation | 12.9 | 6.0e-6 |
| CORDIC 24 | 156.7 | 6.1e-8 |
| path length only (one `√`) | 2.36 | the floor every arm pays |

### Amplitude-bounded early exit for `I ≥ T`

Sources are visited in descending amplitude. The bound is `max(0, |S| − R)² ≤ I ≤ (|S| + R)²`.

**Correctness, two arms:**
- **f64 arm:** asserted equal to the full fold on every detector.
- **L12 arm:** carries its own per-term error `ε = π/2^12 + 2·f32::EPSILON` in the bound. Any detector still undecided after all terms falls back to the f64 reference.

**Terms evaluated** (of 1,000; mean, p95 in brackets):

| T quantile | uniform A ∈ [0.5, 1] | `A_i = i^-1.5` | L12 fallbacks (uniform / heavy) |
|---|---|---|---|
| 50 % | 983 (999) | 122 (550) | 2,679 / 278 |
| 90 % | 968 (996) | 76 (390) | 961 / 160 |
| 99 % | 941 (980) | 28 (100) | 124 / 35 |

## Falsifiers and disable runs

Each disable run was made after the commit, turned red, and was restored afterwards.

| disable | effect |
|---|---|
| early exit ignores the unevaluated amplitude | `exact early exit disagrees at detector 4` |
| LUT bound ignores the table's own error | `decided wrongly at detector 7088` |
| `as u64` instead of `as i64` for a phase difference | two-wave LUT error 2.77 of peak, assertion fires |

**This probe's own first run had two bugs, both caught by its assertions:**
- `as u64` saturates a negative phase difference to 0, giving a 99 % error.
- Deciding at `R = 0` through `√` then a square disagreed with the full fold at the threshold detector. The fix decides on `re² + im²` exactly as the fold computes it, and absorbs the `√` rounding in a relative margin of 1e-12 before that.

## Findings

1. **A phase LUT removes the transcendental cost.**
   - L12i matches f32 `sin_cos` accuracy (4.6e-5 vs 5.1e-5 on R = 100) at about a quarter of the time (5.6 vs 25.4 ns), and is about 12× faster than f64.
   - Nearest L12 is 20× faster than f64 at 8.7e-4 relative error.
   - The `u32` turns representation makes `3θ` one `wrapping_mul` and wraparound free.
2. **CORDIC loses on this CPU.** It is 2.6–4.8× slower than native f64 here (16–30 iterations): a serial, branchy shift-add loop against hardware multiply. It is the right tool only without a multiplier or without table space.
3. **ndarray `vml` is not a SIMD `sin`/`cos`.** It runs as fast as native calls and, used through arrays, materialises its inputs and outputs.
4. **Algebra beats lookup where it applies.** For two waves, the phase difference turns two complex phasors into one cosine (2.8×), and a LUT on Δφ makes it 6×.
5. **The bound early exit is a random-walk problem.**
   - With N comparable amplitudes, `|S| ~ √N·A` while the remaining bound is `~N·A`, so the bound only closes at the very end: 94–98 % of the terms are still needed.
   - With heavy-tailed amplitudes it saves 8–36× (28–122 of 1,000 terms).
   - Whether it pays is a property of the amplitude distribution, not of the bound.
6. **After the LUT, the path length is the next floor.** One `√` per term is 2.36 ns against ~10 ns for the LUT arm. The rest is table access and the multiply-add.
7. **Incremental trajectories:**
   - an integer phase accumulator plus a LUT has zero drift by construction;
   - an f64 recurrence drifts 4.8e-9 in 1M steps, negligible;
   - an f32 recurrence is unusable without renormalisation, and still 1e-2 with it.

## Recommendations

| item | verdict | benefit | falsifier |
|---|---|---|---|
| `u32`-turns phase + 2^12 `(cos, sin)` LUT with linear interpolation | **ADOPT** (separate PR, ndarray, exposed through `ndarray::simd`) | ~4.5× vs f32 `sin_cos`, ~12× vs f64, same accuracy as f32 | max error ≤ 5e-5 relative on 1M phases; `3θ = wrapping_mul(3)` exact |
| Closed form for two waves, `cos Δφ` | **ADOPT** where exactly two coherent terms meet | 2.8× (6× with LUT) | identity to 1e-14 |
| Amplitude-bounded early exit | **PROBE**, only for heavy-tailed amplitude sets | 8–36× fewer terms when heavy-tailed; none when uniform | equals the full fold on every detector; dropping R fails |
| LUT + error-inflated bound with exact fallback | **PROBE** | exact decisions on top of the LUT speed | dropping ε fails; fallback count reported |
| Integer phase accumulator for trajectories | **ADOPT**, together with the LUT | no drift | error equals the table bound after 1M steps |
| CORDIC | **REJECT** on CPU | — | 2.6–4.8× slower than native f64 |
| f32 complex recurrence | **REJECT** without renormalisation | — | 1.95 units after 1M steps |
| ndarray `vml` sin/cos as a SIMD path | **REJECT the equivalence** | — | per-lane scalar; no speed-up |

## Not done here

- **SIMD gather of the LUT** (AVX2 `vpgatherdd`) was not measured.
- **The Wankel rotor as a moving mask** was not built: containment, `point_inside_rotor`, rotating aperture × wavefront, visibility.
- **VSA bind/bundle against complex multiply/add** was not compared.
- **Log₂(r²) overlap buckets** were not built. D-MHB-1 counterexample 5 already shows that distance buckets cannot see phase, and log buckets are coarser still.

## OPEN

- One host class only.
- The LUT's place in ndarray, and whether its interpolation should be `f32` or fixed point, are open.
