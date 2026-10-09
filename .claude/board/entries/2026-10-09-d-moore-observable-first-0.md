# 2026-10-09 — D-MOORE-OBSERVABLE-FIRST-0: provably correct decisions before the field converges

**Status:** MEASURED. Research probe only; no production code changed.
**Harness:** `crates/perturbation-sim/examples/moore_observable_first.rs`
(feature `moore-probe`, 8 tests, CI step in `rust-test.yml` member-tests).
**Parent:** #1427 (`MoorePertubationsfeld`), which reported correlation per
sweep budget. Correlation does not say whether a decision read off an
unconverged iterate is right; this probe measures that directly.

## Question

Can the Moore relaxation make a decision that is **provably** correct before
its numerical field converges? Field accuracy, observable (line-flow)
accuracy and decision correctness are reported separately.

## Setup

- **Decisions** per (outage, line), single-line N-1:
  - overload: `|f_l| > R_l`;
  - the sign of the flow shift;
  - per outage, which line moves most.
- **Ratings** `R_l = max(|f0_l|, 0.2·max|f0|) / 0.8`, i.e. 80 % intact
  loading. This is a policy pin, not data.
- **Near-tie adversary:** 40 ratings are set to an exact post-outage flow, so
  no finite bound can decide them.
- **Oracles**, independent of the Moore arm and of each other:
  - dense Cholesky of the grounded post-outage system;
  - closed-form LODF from one intact inverse;
  - the spectral pseudoinverse (`symmetric_eigen`), sampled.
  - Maximum disagreement 4.2e-13 relative to max|f0|.
- **Fixtures:**
  - #1427's 4×4;
  - `hetero-16x16`: weights log-uniform over 10⁻²..10²;
  - `strip-64x2`: slow mixing;
  - `weak-tie-16x8`: two blocks joined by two weak ties, near-islanding when
    one trips;
  - `bipartite-12x12`: no diagonals, Jacobi oscillates.
- **Arms:**
  - Jacobi from zero (#1427's setup);
  - Jacobi warm-started from the exact intact field;
  - diagonally preconditioned CG from the same start;
  - exact baselines: LODF and dense Cholesky.

## Two certificates

Bus 0 is the reference (ground); `A` is the reduced Laplacian, which is a
nonsingular M-matrix, and `r = p − A x` is the residual.

1. **Field certificate.**
   - If a vector `w` satisfies `A w ≥ 1`, verified by one stencil pass, then
     `|θ_i − x_i| ≤ ‖r‖∞ · w_i`.
   - No eigenvalue is needed.
   - After an outage only two rows of `A w` change, so the intact `w` can be
     rechecked in O(1). When that recheck fails, warm PCG on `A' w = 1`
     produces a new `w`, which passes the same verification.
2. **L1 flow certificate.**
   - The bound: `|f_l − f̂_l| ≤ ‖r‖₁` for every live line.
   - The flow error is `b_l Σ_k (G_ik − G_jk) r_k`. By reciprocity
     `G_ik − G_jk` is the potential at bus k of a unit dipole i→j, with bus 0
     held at 0.
   - By the maximum principle that potential lies between its values at the
     two poles, and so does bus 0's value of 0. Hence
     `|G_ik − G_jk| ≤ R_eff(i,j) ≤ 1/b_l`.
   - Needs: no `w`, no eigenvalue, no conditioning assumption.
   - Cost: Jacobi computes the residual during the sweep anyway.

Both bounds carry a first-order floating-point margin. This is not interval
arithmetic.

## Results

**Soundness:** 0 certified-wrong decisions and 0 bound violations, over every
fixture, arm, budget (0 to 1024) and decision kind.

**Bound tightness** (median of worst-case bound over worst actual error):
- field certificate: 10¹ to 10⁵;
- L1 certificate, warm-started Jacobi: 1.4–25× on the 4×4, hetero, strip and
  weak-tie grids, rising to about 130× on the oscillating bipartite grid.

Halving the L1 bound breaks soundness, so it is within 2× of tight somewhere.

**Iterations until every overload decision of an outage is certified, L1
bound** (median / p95 / max / outages not certified within 20 000):

| fixture | jac-cold | jac-warm | pcg-warm |
|---|---|---|---|
| 1427-4x4 | 30 / 132 / 211 / 0 | 8 / 158 / 203 / 0 | 4 / 12 / 12 / 0 |
| hetero-16x16 | 12620 / 16974 / 19980 / 84 | **0** / 940 / 19034 / 4 | **0** / 97 / 128 / 0 |
| strip-64x2 | 14683 / 18856 / 19999 / 83 | 245 / 8515 / 18233 / 2 | 34 / 66 / 66 / 0 |
| weak-tie-16x8 | 17895 / 19625 / 19997 / 242 | 28 / 2855 / 18957 / 2 | 10 / 36 / 42 / 0 |
| bipartite-12x12 | 4700 / 6997 / 9023 / 0 | 3007 / 5217 / 7949 / 0 | 42 / 50 / 57 / 0 |

With the field certificate instead, cold-start Jacobi certifies no outage at
all on hetero, strip or weak-tie.

**Field vs observable vs decision** (hetero-16x16, warm Jacobi, 1024
sweeps):
- the angle-field delta is still 88 % wrong (relative L2);
- the flow delta is 43 % wrong;
- yet 99.9 % of the 863,970 overload decisions are certified correct.

**What gets certified early is the absence of an overload.** True overloads
certified (L1), out of the total:

| fixture | true | jac-warm @0 / 64 / 1024 | pcg-warm @64 / 128 / 256 |
|---|---|---|---|
| hetero-16x16 | 225 | 0 / 51 / 179 | 134 / 212 / 225 |
| strip-64x2 | 442 | 0 / 135 / 380 | 114 / 442 / 442 |
| weak-tie-16x8 | 467 | 0 / 286 / 399 | 467 / 467 / 467 |
| bipartite-12x12 | 556 | 0 / 0 / 10 | 556 / 556 / 556 |

**Naive early decisions fail in the dangerous direction.** Read off the
iterate without a bound, warm Jacobi:
- misses every true overload at sweep 0;
- on hetero, still gets 61 decisions wrong after 64 sweeps and 11 after 1024;
  with near-tie ratings, 23 after 1024.

**Screening all outages, median of 5, ms, quiet machine:**

| fixture | LODF (incl. intact inverse) | dense Cholesky | certified PCG (L1) | certified Jacobi (L1) |
|---|---|---|---|---|
| hetero-16x16 | 16.6 | 2129 | 172 | 2747 |
| strip-64x2 | 2.0 | 93 | 61 | 1388 |
| weak-tie-16x8 | 2.3 | 167 | 30 | 783 |
| bipartite-12x12 | 2.6 | 105 | 33 | 1869 |

The Jacobi times are dominated by the few outages that run to the
20 000-sweep cap.

## What this falsified

- **"Moore can decide early because flows converge faster than phase."**
  - Flows do converge faster.
  - The reason a flow decision is provable early is different: the flow error
    is bounded by `‖r‖₁` regardless of how badly conditioned the system is.
    Warm-starting makes the initial residual exactly `±f0_k` at the two
    endpoints of the tripped line.
- **"Jacobi is the arm to use."** The L1 certificate works with any iterate.
  PCG reaches certification 16× faster than Jacobi on hetero and 57× faster
  on bipartite. Jacobi's only advantage is that it needs no global dot
  products, and nothing here measured that as an advantage.
- **"An iterative method beats a direct one for N-1 screening."** It does not
  on a fixed grid. LODF from one intact inverse is about 10× faster than
  certified PCG.
- **"Correlation tells you when a field is good enough to decide on."** At
  flow r ≈ 0.97 the naive decisions are still wrong.
- **"The cheap O(1) reuse of the field certificate after an outage is
  enough."** The two-row recheck fails on 26–92 % of outages.

## Gaps this probe exposed

- **Sign decisions.**
  - The L1 bound is uniform across lines, while the shifts on distant lines
    decay with distance.
  - So their signs need a locality-aware bound. Measured: hetero, warm
    Jacobi, 1024 sweeps certifies only 8.6 % of signs.
- **Shifts smaller than the residual floor.**
  - On the strip, even fully converged PCG certifies only 50.1 % of signs:
    the shifts on distant lines decay below the ~1e-13 residual floor, and
    some fall below the oracle's own precision, where the true sign is
    undefined.
  - The certificate correctly abstains on those. The naive count there
    (1,153 sign "errors" at full convergence) is oracle noise, not error.
- **Exactly mirrored lines** make "which line moves most" undecidable. The
  certificate abstains there, which is correct.
- **Outside this probe's scope:**
  - multi-outage (N-2) contingencies;
  - grids beyond about 10³ buses, where the honest baseline is sparse
    Cholesky, not dense;
  - AC flow, where the linear certificate does not apply.

## Disable runs (all red as required)

- Field bound halved → `certified_decisions_are_never_wrong` fails.
- L1 bound halved → that test plus `l1_flow_bound_holds_without_a_certificate`
  fail.
- The two-row recheck skipped (intact `w` reused unconditionally) → 4 tests
  fail.
- Certifying on the point estimate (bound ignored) → 3 tests fail.

One test was a guess and was replaced before commit: it assumed zero
certificate fallbacks on `hetero-6x6`, and the measurement showed 30.

```
STATUS: measured | OUTCOME: Moore/Jacobi can certify most decisions before
convergence, but the enabling piece is the solver-agnostic ||r||_1 flow bound
plus warm start; early certificates prove absence of overload, not presence;
PCG and LODF dominate Jacobi for N-1 screening | OPEN: locality-aware bound
for sign decisions; N-2; sparse baseline at scale
```

MIRROR
- BLIND SPOT: all grids are lattices of at most 256 buses with synthetic
  injections. Real transmission topology and loading may move every
  percentage.
- BIAS CHECK: did the #1427 framing anchor the question on Jacobi? The
  answer came from adding arms (warm start, PCG, LODF) that the original
  framing did not include.
- STILL OPEN: whether locality (no global reductions) ever pays for itself.
  Only a distributed or streaming setting can measure that.
