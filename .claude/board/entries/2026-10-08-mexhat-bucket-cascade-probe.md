# 2026-10-08 — Mexican hat × popcount stacking × early exit, without a raster (D-MHB-1)

**Status:** MEASURED. Probe only, no production primitive.
- **Probe:** `crates/lance-graph-mask-risc/examples/mexhat_bucket_probe.rs`.
- **Revisions:** lance-graph `origin/main` d127f7dc; ndarray `origin/master` 30ce119.

```text
CARGO_PROFILE_RELEASE_DEBUG=0 cargo run --release -p lance-graph-mask-risc --example mexhat_bucket_probe
```

**The question.** Can popcount stacking, a quantised Mexican hat and an exact
early exit answer a centre-surround query without per-point geometry and
without a materialised raster? If so, is a Rolling-Floor controller worth
adding on top?

**The query:**

```text
S(c) = Σ_{p ∈ P, |p − c|² ≤ R²} w(|p − c|²)
```

- `P` is a presence bitmap over a row-major `W × H` grid.
- `w` is a normalised DoG, quantised to integers with `w(0) = 2^12`.
- The kernel is σc 3, σs 6, R 18: a 37 × 37 window holding 1,009 disk cells.

## Inventory

### SHIPPED (VERIFIED-IN-CODE)

**ndarray:**
- **`RollingFloor`** (`hpc/rolling_floor.rs`):
  - exact `(n, Σx, Σx²)` moments, deterministic reservoir;
  - a quarter-σ lattice, thresholds derived on demand;
  - drift read at the 1σ/3σ cuts.
  - Lower is stronger (`shade`, l. 755).
  - Not re-exported from `ndarray::simd`.
- **`hamming_distance_within`** (`bitwise.rs`): an exact monotone early exit per 256-byte block. Re-exported from `ndarray::simd`.
- **`Cascade`** (`hpc/cascade.rs`):
  - Stroke 2 has an exact budget early exit (l. 346).
  - **Stroke 1 is a statistical prune with no exact fallback.** A prefix whose estimate exceeds `threshold + 3σ` is dropped (l. 329–337). An unresolved candidate disappears instead of falling through to an exact check.

**lance-graph:**
- **mask-risc** fused `Count` / `Any` over resident planes.
- **`MaskedSumI32`** with `execute_extent`.

### PROPOSED, or present in name only

- **`styles/lsi.rs` `mexican_hat`** is a five-step percentile function in f32 (1 / 0.5 / 0 / −0.5 / −1). The percentiles come from a normal fit. It is not a normalised DoG.
- **`pillar/mexican_hat.rs` (Pillar-15) is DEFERRED.** `prove_pillar_15` runs nothing and returns `passed: true`. No DoG kernel exists in ndarray.

### MISSING

- A DoG kernel in ndarray.
- A 2-D window or stencil operand in mask-risc: planes are 1-D row spaces.

### Observation 6, reproduced

`examples/rolling_floor_probe` (ndarray) still prints the same result:

| phase | FIXED reject | SHIPPED reject | EWMA reject |
|---|---|---|---|
| 2 | 97.55 % | 97.55 % | 0.12 % |
| 4 | 64.10 % | 64.10 % | 0.07 % |

FIXED and SHIPPED are identical because `Cascade`'s `ShiftAlert` never fires on a per-sample feed. That probe exercises **`Cascade`**, not `RollingFloor`. `RollingFloor`'s checkpoint drift rule is untested by it.

## Semantics, kept apart

- **Continuous** (arm A, f64): the reference for the quantisation error.
- **Quantised** (integer `w(q)` per lattice offset): arms Ascan, D, F, G and B must agree, and the probe **asserts exact equality** on every checked centre:
  - 1,024 centres × 4 densities × 2 grid sizes;
  - B on every timed query.

  Its distance from the continuous reference, as a mean of |S|:

  | ρ | quantised vs continuous |
  |---|---|
  | 0.01 | 0.07 % |
  | 0.1 | 0.18 % |
  | 0.5 | 0.5 % |
  | 0.9 | 1.3 % |

- **Bucketed** (arm E: K equal-q rings, one mean weight each): a further approximation, measured against the quantised answer, never asserted equal.

## Kernel falsifiers

All six `(σc, κ)` pairs pass, for κ ∈ {1.5, 2, 3}:
- a positive centre;
- a negative surround;
- **exactly one sign change**, at the analytic radius² `2σc²σs² ln(σs²/σc²)/(σs² − σc²)` within one q-step (e.g. 59.15 lands between lattice q 59 and 60);
- **one annular minimum**: no false extremum after quantisation;
- the 8 square symmetries of the template;
- rings that partition the disk;
- no overflow of the worst-case bounds.

## Results

**Setup:** median of 10 runs.
- Host: Intel Xeon @ 2.8 GHz, 4 vCPU (AVX-512 on the host).
- Build: `x86-64-v3` (AVX2, POPCNT), release, `debug = 0`, rustc 1.98.1.
- Single runs scatter up to 2×, which is why medians are reported.

**ns per query:**

| ρ (1M grid) | A f64 | Ascan | **D** window + LUT | E K=8 | E K=32 | **F** bit-sliced | G per-row | B mask-risc |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 529 | 125 k | **246** | 489 | 1378 | 624 | 267 | 61.7 k |
| 0.1 | 2812 | 499 k | 619 | 464 | 1376 | **592** | 620 | 62.4 k |
| 0.5 | 11.8 k | 2.0 M | 1166 | 489 | 1413 | **624** | 894 | 80.4 k |
| 0.9 | 21.0 k | 3.5 M | 1516 | 473 | 1374 | **604** | 679 | 88.9 k |
| wavefront ring + 1 % noise | 560 | 135 k | **252** | 477 | 1334 | 591 | 270 | 64.0 k |

The 64K grid gives the same orderings (D 230 / 620 / 1070 / 1516; F ≈ 600).

| arm | work per query | memory |
|---|---|---|
| D | ~1.4 ns per present disk cell + ~210 ns base | |
| F | constant: 418 popcounts | |
| E, F, D | read about 57 `u64` words (the row-major window) | 0 allocations per query |
| B | | writes a 37.9 KB (64K) / 151.6 KB (1M) weight lane per query, over a 4 MB resident lane |

## Exact-bound early exit (arm H)

**The decision:** `S ≥ T`, using F's rows, centre rows first.

**The bounds:** each row's suffix bound is `[Σ negative, Σ positive]` of the rows not yet visited. The early decision is:

| condition | decision |
|---|---|
| `S + U < T` | reject |
| `S + L ≥ T` | accept |
| otherwise | visit the next row |

**Correctness:** the decision is asserted equal to the full answer for every query: 4,096 queries × 4 thresholds × 5 fixtures.

**Rows visited** (of 37; mean, with p95 in brackets):

| T quantile | ρ 0.01 | ρ 0.5 | ρ 0.9 |
|---|---|---|---|
| 10 % | 30.3 | 21.6 | 15.8 |
| 50 % | 24.3 | 24.5 | 24.0 |
| 90 % | 14.2 | 21.7 | 26.1 |
| 99 % | 9.9 (p95 14) | 17.8 | 24.4 |

**Timing at the median T:** H 500–534 ns against F 628–651 ns, about −20 %. On sparse input, D (~250 ns) beats H.

## Counterexamples

1. **The running sum peaks above T, then ends below it.** A positive core plus a late negative surround: peak 143,154 above T = 100,777, final S = 58,400. H stays undecided while the sum is above T and rejects after 18 of 37 rows.
2. **Cancellation.** A ring placed exactly at the zero crossing gives S = 116. Deciding `S ≥ 1` needs 35 of 37 rows.
3. **Dense input with T at the median:** 24.2 of 37 rows. The bound cannot help here.
4. **Disable run, in place.** An upper bound that drops the unvisited rows decides 52 of 512 queries wrongly, so the bound is load-bearing.
   A second disable run, made after the commit: shifting F's negative planes one bit too far fails `F differs from D` at the first checked centre. The file was restored afterwards.
5. **Geometry buckets cannot see phase.** Two coherent sources, λ = 8, detector cells bucketed by `(r1, r2)`:
   - buckets of width λ/2: 738 of 1,003 occupied buckets hold both a dark (I < 0.1) and a bright (I > 0.9) cell;
   - buckets of width λ/8: 0 of 8,032 do.

   A geometric prune coarser than about λ/8 drops dark fringes with the noise.
6. **Locality.** A 37 × 37 window touches 56.7 row-major words and 30.2 Morton words. Morton touches fewer words, but its bits are not row-shaped. Arms D/E/F extract row windows, so Morton needs different (8 × 8 tile) templates. Word counts only, not timed.

## Findings

1. **The exact answer is already cheap; the bucketed one is wrong.** Equal-q rings (E):
   - K = 8 is fast (~470 ns, constant) but wrong by 950–5,600 on |S| in the thousands;
   - K = 32 is still wrong by 255–1,830 and slower than exact F.

   A popcount ring fold buys nothing an exact path does not.
2. **Bit-slicing makes popcount stacking exact.** For a static integer weight template:

   ```text
   Σ_p w(p) = Σ_b 2^b · (popcount(P ∧ Pos_b) − popcount(P ∧ Neg_b))
   ```

   - It costs 2 × 13 planes per row, independent of density, at ~600 ns.
   - It is identical to direct geometry: asserted on every centre.
   - It needs no value lane, no square root and no `exp`.
3. **Crossover at ρ ≈ 0.1** (≈ 100 present cells in a 1,009-cell disk):
   - below it, direct per-point geometry D wins: 246 vs 624 ns at ρ 0.01, and on the wavefront fixture;
   - above it, F wins: 624 vs 1,166 ns at ρ 0.5, 604 vs 1,516 at 0.9.
4. **Per-row adaptivity (G) does not reach min(D, F).** Choosing per row with one popcount is exact, but it loses up to 43 % at mid density (894 vs 624 ns at ρ 0.5). Choose per **query** instead, from the window's own popcount (37 popcounts ≈ 50 ns). That is an exact count, not a statistic.
5. **The shipped mask-risc path (B) is 26–40× slower than F on the 64K grid and 100–150× on the 1M grid** (the band is full width) and materialises a weight lane per query. The cause is that mask-risc has no 2-D window operand.
6. **Exact early exit is a modest, safe win.** About −20 % on a threshold decision over F, up to ~4× fewer rows at extreme quantiles on sparse input. It is never wrong, by construction and as asserted.

## Recommendations

| item | verdict | why |
|---|---|---|
| Bit-sliced weight fold (F) | **ADOPT as the dense path** (a separate PR, routed through `ndarray::simd` popcount) | exact, constant cost, no lane, no sqrt/exp |
| Direct window geometry (D) | **ADOPT as the sparse path** | fastest below ρ ≈ 0.1 |
| Per-query D/F choice from the window popcount | **PROBE** | exact count, ~50 ns; should reach min(D, F), not measured yet |
| Exact-bound early exit (H) | **PROBE** | −20 % at median T; needs a D-path variant for sparse input |
| Ring-bucket Mexican hat (E) | **REJECT** | wrong at every K tried and not faster than exact F |
| Rolling-Floor-controlled cascade | **PARK** | the deciding quantity (window density) is an exact popcount; a statistical controller has nothing to decide here |
| Statistical early exit + exact fallback | **PARK** | the exact bound already decides without error |
| mask-risc `MaskedSumI32` over a per-query weight lane (B) | **REJECT for this query** | 26–150× slower, materialises a lane |
| `lsi.rs` `mexican_hat` as a DoG | **REJECT the equivalence** | a percentile step function, not a DoG |
| Pillar-15 activation | **PARK** (separate from any planner change) | needs a real DoG kernel in ndarray first; the deferred stub reports `passed: true` |

## Proposed rewrite (not implemented)

**R-MHB-1, bit-sliced weighted fold.**
- **When:** a masked sum whose weights come from a static, narrow integer template.
- **Rewrite:** `MaskedSumI32` over that template → `Σ_b 2^b · (Count(P ∧ Pos_b) − Count(P ∧ Neg_b))` over resident planes.
- This turns a value-lane fold into fused `Count`s (D-RPF-9 finding 1: no slot, no bitmap).
- **Exact** for the quantised semantics.
- **Applies only** when the template planes are resident and aligned with `P`. For a moving window that needs the missing 2-D window operand.

## Boundaries

- No wavefield or raster is built. The only per-query state is a running sum and the window words.
- An interference intensity `|E1 + E2|²` stays a phase computation. A centre-surround score is not a fringe classifier (counterexample 5).
- Every popcount in the probe is `u64::count_ones`. A production path goes through `ndarray::simd`.

## OPEN

- Per-query D/F selection: not yet measured.
- An early exit on the D path for sparse input: not yet measured.
- Morton-shaped (8 × 8) templates: not timed.
- `RollingFloor`'s own drift rule under drifting density streams: untested.
- `Cascade` Stroke 1 silently drops unresolved candidates. It is an ndarray behaviour, reported here, not changed.
- One host class only. The ρ ≈ 0.1 crossover is a pin for this host.
