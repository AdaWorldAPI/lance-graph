# KNOWLEDGE.md — Place / Residue Encoding (golden-spiral hemisphere, Fisher-Z aligned)

> **READ BY:** family-codec-smith, palette-engineer, certification-officer,
> truth-architect — anyone touching `crates/helix`, the residue codec, or the
> HHTL place/residue split. Companion to `bgz17/KNOWLEDGE.md` (metric-safety
> contract, palette/LUT) and `jc::weyl` (the 1-D φ-stride proof).
>
> **Status:** crate `helix` v0.1.0 implements this document. The body below is
> the concept-preservation spec (code is downstream); the two appended sections
> — **Implementation Map** and **Overlap & Consolidation** — record how the code
> realises it and how it relates to the workspace's existing (certified)
> primitives. Sections are labelled FINDING / CONJECTURE per the insight cycle.

---

## What This Is

The orthogonal-residue half of the substrate. **HHTL is the deterministic
PLACE** (the trie address — *where*). **This algorithm is the RESIDUE** (the
orthogonal edge at that place — the hemispheric angle the place itself does
not capture). It is the discrete 2-D companion to `crates/jc/src/weyl.rs`:
where `weyl.rs` proves the 1-D `{k·φ⁻¹ mod 1}` golden stride has minimal
star-discrepancy, this defines the 2-D hemispheric residue that rides the
same φ skeleton and lands in the same int8 / 256-palette / L1 arithmetic.

Headline property: **8K resolution at Super-8 cost.** Full neighbour-
discrimination (the curve is regenerated from a template, not stored), at
3-bytes-per-edge storage and O(1) 256×256-LUT distance. The resolution lives
in the deterministic template (free, regenerable); the cost is only the
endpoint pair.

## The Curve-Ruler Principle (the core idea)

> Every curve is determined by start and end on the curve-ruler.

A draughtsman's curve-ruler (French curve) has a fixed shape. Mark two points
— start and end — and the whole curve between them is determined, because the
template fixes the form. You do not store the curve. You store
`(template-ref, start, end)` and regenerate.

The φ-low-discrepancy sequence (the `weyl.rs` golden stride) **is the
template**. Once you know "it is the φ-spiral" and "index a to index b", every
point between is determined by `(offset + 4·k) mod 17`. Endpoints are stored;
the interior regenerates. This is the `bgz17` 680× compression lifted to the
*curve* level: do not compress the points — recognise they lie on a template,
keep only the endpoints.

Distance between two curves becomes distance between their endpoints on the
same template. Endpoints in a linear index order have **L1 distance** →
triangle inequality free → metric-safe for CAKES. This is the property Scent
does NOT have (Hamming on 7-bit lattice) and Palette/Base17 DO have
(L1 on i16[17]).

## The Three Constants (three jobs, do not conflate)

```
GOLDEN_RATIO   φ   = 1.618_033_988_749_895   (φ⁻¹ = 0.618_033_988_749_895)
EULER_GAMMA    γ   = 0.577_215_664_901_532_9   (Euler–Mascheroni)
E              e   = 2.718_281_828_459_045     (Euler's number)
```

Their values sit close enough to invite confusion; their roles are disjoint.

- **GOLDEN_RATIO (φ) — PLACES.** The stride/placement constant; the discrete
  stride-4-over-17 is its integer counterpart; lowest star-discrepancy among
  all irrationals (continued fraction `[1;1,1,1,…]`, Ostrowski bound `2/N`). φ
  decides *where* points fall, with minimal aliasing.
- **EULER_GAMMA (γ) — CORRECTS.** The harmonic-to-log bridge:
  `γ = lim(Σ_{k=1}^{n} 1/k − ln n)`. It is exactly the difference between a
  discrete (rank-like) sum and the continuous (scale-like) logarithm — precisely
  the term that reconciles a discrete index rank with the continuous φ scale. γ
  is what makes a non-φ distribution "fit", or makes it pay.
- **E (e) — DESCRIBES GROWTH.** The natural base for the exponential growth of
  the golden-lattice spacing, and the base of `simd_ln_f32` / `simd_exp_f32`,
  used when the Fisher-Z / arctanh step leaves the LUT and is computed.

## The Founding Bet (why the inbound tax works at all)

> Natural distributions converge to the golden ratio. Everything else is made
> to fit with EULER_GAMMA, or it pays the inbound tax.

1. **Natural distributions converge to φ.** Phyllotaxis, growth spirals, the
   convergence of any recursive Fibonacci-like ratio → φ. The golden ratio is
   the *attractor* self-similar growing structures converge to, *because* it is
   the most irrational (slowest approximation → no resonance → stablest packing).
2. **Everything else is corrected by γ, or pays.** What does not arrive already
   φ-distributed is pulled into alignment by the Euler-Gamma offset. What is so
   far from φ that γ cannot align it pays the full inbound tax — it packs into
   suboptimal buckets, costs resolution or storage.

This is self-selecting efficiency: data that follows the natural φ tendency
flows cheaply; data that works against the attractor pays. It dissolves the
per-palette distribution problem — one normalisation, one attractor, one
correction term, applied at entry, not per-palette in the interior.

## The Residue Algorithm

Orthogonal to the HHTL place. The residue need NOT arrive perfectly
distributed — the distribution is reached afterward by Euler. The algorithm
only places the raw residue into the index order; γ aligns it before the
256 buckets fall.

```
RESIDUE-ENCODE  (orthogonal to the HHTL place)

  in:  hhtl_address (the place), raw_residue n, total N

  1. PLACEMENT  (tomato-rose hemisphere lift)
       u = (n + 0.5) / N            // midpoint rule, no edge bias
       r = sqrt(u)                  // equal-area disk radius
       Y = sqrt(1 - u) = sqrt(1-r^2)// hemispheric lift (pole distance)
       phi = n * GOLDEN_RATIO       // golden-angle azimuth
       // X = r·sin(phi), Z = r·cos(phi)  (only if Cartesian needed)

  2. PLACE COUPLING  (stride arc from the HHTL offset)
       start = hhtl_to_offset(hhtl_address)   // the place sets WHERE on the ruler
       idx   = (start + 4*k) mod 17           // stride 4, modulus 17 (prime)
       // gcd(4,17)=1 -> full permutation of all 17 residues

  3. FISHER-Z ALIGNMENT  (= hyperbolic depth — one function, two meanings)
       z = arctanh(similarity) = 0.5*(ln(1+s) - ln(1-s))   // via simd_ln_f32
       // identical to hyperbolic depth rho = 2*arctanh(r) up to the factor 2

  4. EULER HAND-OFF  (distribution is reached afterward)
       aligned = z + EULER_GAMMA * (rank(n)/N - ln(17))   // constant gamma shove
       palette_idx = quantize_256(aligned, floor_state)    // -> U8, L1 metric-safe

  out: (start_idx, end_idx) endpoint pair
       -> L1 distance, 256x256 LUT, U8x64 hot path
```

### Fixed parameters

- **Modulus M = 17, stride = 4.** 17 is prime → every coprime stride is a full
  permutation; 4 is coprime to 17. 17 is the zeck17 / Base17 number running
  through the stack (i16[17]), so the residue aligns to the same structure the
  palette uses and inherits its metric safety. `ln_term = ln(17)`, a constant
  γ shove per calibration (not per-rank).
- **Start offset 17 or 20.** The first ~17-20 indices are the bad-discrepancy
  transient (Ostrowski `2/N` is large for small N). Starting at 17/20 skips the
  transient and begins in the regular, low-discrepancy regime. 17 aligns to
  Base17; 20 sits just below F₈=21.

## Geometry: Spherical, NOT Hyperbolic (a naming guard)

"Poincaré-inspired" here refers to the **encoding intuition** — a bounded disk
where centre = early/coarse and rim = late/fine, the whole structure graspable
at once (the "tomato-rose": a flat spiral slice curled up into a dome) — NOT to
hyperbolic curvature. The lift is **spherical (equal-area)**:

```
r = sqrt(u),  Y = sqrt(1 - r^2)     // equal-AREA hemisphere (what we use)
NOT:  Y = 1 - u                      // equal-area surface DENSITY (different)
```

Chosen for explainability and cheap `sqrt`-only computation. The hyperbolic
*metric* feeling (rim densification) is supplied separately and for free by the
arctanh / Fisher-Z step (stage 3), not by the placement geometry. Two layers:
spherical placement (the grid), hyperbolic-flavoured depth (the alignment).

## Layer Discipline (inherited from bgz17/KNOWLEDGE.md)

The residue distance rides the same metric-safety contract:

- **Metric-safe (L1 on the endpoint index order):** usable for CAKES pruning,
  triangle inequality holds by construction.
- **Heuristic only:** any angular/periodic measure on raw `phi` is NOT a metric
  (the 2π wrap, same failure mode as Scent's 7-bit lattice). Use the linear
  index-order endpoints for bounds, never the raw azimuth.

`distance_adaptive()` returns the metric-safe endpoint L1.
`distance_heuristic()` may return a raw angular pre-filter — caller must NOT
use it for CAKES bounds.

## Sampling Grid vs. Data Distribution (a layer guard)

Weyl/low-discrepancy guarantees the **sampling order** is optimally
equidistributed. It does NOT guarantee the **data values** are equidistributed
— that is why the rolling-floor adaptation and the γ hand-off exist. The optimal
grid makes observation of the (non-optimal) data as clean as possible; the two
are different layers. Do not conflate "the grid is φ-optimal" with "the data is
φ-distributed."

## Calibration: Rolling Floor with Occupancy Drift Detection

The 256 buckets are simultaneously their own monitoring instrument. Under a
stable calibration the bucket occupancy follows an expected shape; when the
underlying distribution shifts, buckets at a *different* place start filling —
visible immediately, because the occupancy IS the distribution estimate.

- **Rolling floor:** bucket bounds glide slowly after a drifting distribution —
  not frozen (would stale), not per-value (would jitter). Inertia ignores
  short-term noise, follows real drift.
- **Drift vs. noise threshold:** expected per-bucket occupancy variance is
  multinomial (`√(np(1-p))`); deviation beyond several such SDs is real drift,
  below is sampling noise. Principled, distribution-free.
- **Determinism under adaptation:** "same value → same int8" holds only WITHIN a
  stable floor state. Each quantised batch carries a **floor-version stamp**;
  oracle differential tests compare "same value under same floor → same int8".

## Quantisation Cost (honest, quantified)

- **256 uniform buckets**, each `1/256 = 0.390625%` wide.
- **3σ span**: 99.73% of mass inside ±3σ, 0.27% outside. Since one bucket
  (0.39%) is WIDER than the tail (0.27%), the entire tail saturates into the
  two rim buckets — no inner bucket is spent on outliers.
- **Inside 3σ**: error = ±½ bucket = ±0.195% of the span, uniform everywhere.
- **Outside 3σ**: saturates to the rim bucket (loss, by design).
- Net: NOT lossless. Loss-uniform in the 99.73% informative range at ±0.195%,
  controlled-saturating in the 0.27% tail. A stronger, quantifiable claim than
  "lossless" — the error is a stated number.
- 256 is also int8 (one byte, `U8x64`-aligned). The "bucket width > tail mass"
  constraint and the int8/register-alignment constraint point at the same number.

## The Fisher-Z / Hyperbolic-Depth Identity

The depth scale `rho = 2·arctanh(r)` is exactly twice the Fisher-Z transform
`z = arctanh(r)`. Same `arctanh` core. Not "close to" — identical up to the
factor 2 (geometry keeps the 2 as hyperbolic arc length `∫ 2/(1-t²)`; statistics
drops it for variance stabilisation `Var(z) ≈ 1/(n-3)`). Both map a bounded
`[-1,1]` quantity onto an unbounded scale so equal steps mean equal amounts.
For neighbour similarities near the upper rim (ρ=0.937, 0.982), Fisher-Z
stretches the rim-near differences before quantisation, putting the 256-palette
resolution where the real neighbour distinctions happen.

## Where This Sits in the Stack

```
HHTL address (the PLACE)              -> WHERE: deterministic trie position
     |  hhtl_to_offset
     v
RESIDUE (this algorithm, the EDGE)    -> orthogonal hemispheric angle at the place
     |  tomato-rose placement (r=√u, Y=√(1-r²), φ=n·GOLDEN_RATIO)
     |  stride-4-over-17 arc from the offset
     |  Fisher-Z / arctanh alignment (= hyperbolic depth, one function)
     |  EULER_GAMMA hand-off (distribution reached afterward)
     v
ENDPOINT PAIR (start_idx, end_idx)    -> L1 metric-safe, 256-palette
     v
256×256 LUT, U8x64 hot path           -> O(1) distance, Super-8 cost
```

## Open Items (from the original spec)

1. **2-D discrepancy proof.** `weyl.rs` proves the 1-D case. The 2-D
   golden-spiral-on-disk equidistribution is a separate result.
2. **Operation order in the γ hand-off is non-commutative.** `(cut × stride) +
   γ_offset → normalise` differs from `(cut + γ_offset) × stride → normalise`.
   The implementation must fix one exact order, bit-for-bit.
3. **Start offset 17 vs 20** is a calibration choice (Base17 vs F₈=21 proximity).

*This document is concept-preservation, not implementation. Code is downstream.*

---

# Implementation Map (helix v0.1.0)

How the code realises the spec above. Modules → pipeline stages:

| Stage | Module | Carrier / entry | Notes |
|---|---|---|---|
| 1 Placement | `placement.rs` | `HemispherePoint::lift(n, N)` | `r=√u`, `Y=√(1−u)`, azimuth `n·φ`; `rim()` returns `r` |
| 2 Coupling | `curve_ruler.rs` | `CurveRuler::from_place / from_hhtl` | `index(k)=(start+4k) mod 17`, full permutation |
| 3 Fisher-Z | `fisher_z.rs` | `Similarity(s).fisher_z()` | `½(ln(1+s)−ln(1−s))`; `hyperbolic_depth = 2·fisher_z` |
| 4 Euler+quant | `quantize.rs` | `RollingFloor::quantize / observe / roll` | 256 buckets, occupancy drift, version stamp |
| out | `residue.rs` | `ResidueEncoder::encode → ResidueEdge` | 3-byte endpoint pair `(start_idx, end_idx, floor_version)` |
| distance | `distance.rs` | `DistanceLut::{linear, from_floor}` | 256×256 L1, metric-safe |
| proof | `prove.rs` | `prove() → ProofResult` | the 2-D discrepancy companion (Open Item #1) |
| accel | `simd.rs` | `batch_fisher_z`, `batch_l1_u8` | always ndarray `simd_ln_f32` / `U8x64` (mandatory dep; ndarray does its own AVX-512/AVX2/scalar dispatch) |

**Pipeline decisions (the latitude the spec grants — "code is downstream"):**

- **Fisher-Z input is the radius `r = √u`** (FINDING: the spec writes
  `ρ = 2·arctanh(r)` with `r` the radius, so `similarity := r`). With the
  midpoint rule `u ∈ (0,1)`, `r ∈ (0,1)` and `arctanh(r)` is finite; the extreme
  rim is clamped by `Similarity::CLAMP_EPS = 1e-9`.
- **Endpoint pair semantics:** `start_idx` is the quantised value of the PLACE
  anchor (`r0 = start_offset/17` through the same pipeline) — *where the arc
  begins on the ruler*; `end_idx` is the quantised value of the residue point
  `n` — *where the arc ends*. The pair regenerates the whole curve.
- **Operation order — RESOLVED (Open Item #2).** Fixed bit-for-bit as
  `aligned = (z × STRIDE) + γ·(rank/N − ln 17)`, then `quantize`, i.e. the
  `(cut × stride) + γ_offset → normalise` order the spec's confirmation note
  selected. `rank/N = n/N`; `ln 17` is the constant per-calibration shove.
- **Floor auto-init:** `ResidueEncoder::new(N)` seeds `RollingFloor` bounds from
  the pipeline value at `n=0` and `n≈0.99·N`, so the bulk lands in-range and the
  top ~1% rim saturates into bucket 255 (the intended controlled-saturation tail).
- **`const::simd` correction (FINDING):** the spec references
  `const::simd::{GOLDEN_RATIO, EULER_GAMMA, E}`; that path does not exist. The
  canonical source is `std::f64::consts` (Rust ≥1.94); ndarray does not wrap
  them. `constants.rs` defines local consts (mirroring `std`, like `jc::weyl`'s
  `PHI_INV`) to stay toolchain-robust (independent of the mandatory `ndarray` link).
- **ndarray is a MANDATORY git dependency (FINDING — codex P2 #460 + directive
  "ndarray is mandatory for lance-graph"):** the `simd.rs` batch path runs on
  `ndarray::simd`. ndarray is sourced by `git` (`AdaWorldAPI/ndarray @ master`),
  NOT a local path — an optional/local *path* dep forces Cargo to read the sibling
  manifest at resolution, failing a clean checkout; a non-optional git dep resolves
  remotely and is a hard dep (no feature gate). The fork is self-contained
  (internal subcrates only, no lance-graph back-dependency) → no import cycle.

**Metric-safety (enforced):** `ResidueEdge::distance_adaptive` = L1 over the
256×256 LUT — a metric, safe for CAKES/CLAM bounds (regression-tested:
`distance::linear_satisfies_triangle_inequality`, zero violations).
`ResidueEdge::distance_heuristic` = a byte-Hamming pre-filter returning
`(d, below_threshold)` — NOT a metric; pre-filter only.

# Overlap & Consolidation (placement check, 2026-06-03)

**FINDING (placement check).** ~80% of this pipeline already exists in the
workspace; some of it is certified. helix is a deliberate **clean-room
re-derivation** — it re-derives the math rather than reusing those primitives (per
the directive "scoped only to crate, self-resolving" and
the curve-ruler "regenerable from template" ethos). The genuinely novel pieces
are the equal-area `√u` hemisphere placement and the PLACE/RESIDUE doctrine.

| helix piece | Pre-existing (in places CERTIFIED) | Location |
|---|---|---|
| Fisher-Z/arctanh → i8/i16 | `Base17Fz`; `FamilyGamma` (CERTIFIED ρ≥0.999, 21 roles) | `bgz-tensor/src/projection.rs`, `fisher_z.rs` |
| golden-spiral azimuth proof | `weyl::prove()` (1-D φ-stride, Ostrowski 2/N) | `jc/src/weyl.rs` |
| stride coupling | stride-4 family-zipper; golden-step `(i·11)%17` | `thinking-engine/reencode_safety.rs`, `bgz-tensor/projection.rs` |
| EULER_GAMMA hand-off | γ+φ preconditioner; euler-fold; gamma-calibration | `jc/precond.rs`, `bgz-tensor/euler_fold.rs` |
| 256-palette / L1 endpoints | `Palette` (256), `PaletteEdge` (3 B, metric-safe) | `bgz17/src/palette.rs` |

**Consolidation path (when helix graduates from clean-room to integrated):**
1. Replace `fisher_z.rs` with a thin re-export of `bgz-tensor::fisher_z::FamilyGamma`
   (the CERTIFIED i8 table); keep the scalar `Similarity` as the per-element path
   (the batch SIMD path already requires the now-mandatory `ndarray` dep).
2. Route the stride coupling through the established `(i·11)%17` golden-step or
   the stride-4 family-zipper rather than a second implementation.
3. Feed `ResidueEdge` endpoints into the existing HIP/TWIG CAKES path
   (`PaletteEdge`-equivalent tier) rather than a parallel `DistanceLut`.
4. Lift the PLACE/RESIDUE doctrine + `√u` hemisphere (the new parts) upstream as
   a Layer-2 role catalogue, not a parallel codec.

**Iron rules respected.** Stride-4-over-17 is coprime → full permutation; the
**banned** Fibonacci-mod-17 (misses {6,7,10,11}) is NOT used.

**Gate (MEASURE — probe pending).** The encoding-ecosystem naive-u8 floor gate
requires ≥ 0.9980 Pearson vs ground truth to justify a new int8 encoding's
existence. helix's endpoint fidelity vs the certified `Base17Fz` is **CONJECTURE
— probe NOT RUN**; the `prove_residue` example is the discrepancy probe, but a
fidelity-vs-ground-truth probe is future work before helix is promoted past
clean-room status.

**Name note.** `helix` also appears in `.claude/plans/palantir-parity-cascade-v2.md`
for a planner "Helix" (Foundry time-series histogram) — a plan-doc discussion
that leaned *away* from a crate, so `crates/helix` is free, but a future reader
should not conflate the two.
