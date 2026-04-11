# Invariant Matrix — Research Note (additive, no code changes)

> **Status:** research only. No production code touched. This document replaces the
> "pick the winning encoding" framing with "which lens covers which invariant."
>
> **Companion notes:**
> - `ONE_FORTIETH_SIGMA_LENS.md` — 1/40 σ lens as a matrix column
> - `RING_PERTURBATION_PROPAGATION.md` — NARS+ONNX+SiLU+ReLU runtime mechanism
> - `PHASE_5_12_LAB_PROTOCOL.md` — kill criteria and test order
> - `CONSTANTS_DISCIPLINE.md` — when to use φ vs e vs π vs exact integers

---

## The reframing

We kept asking "which encoding wins." Every answer produced a narrative that held
until the next measurement killed it. The right question is instead:

> **Which invariant does each encoding preserve, and which pair of encodings is
> complementary enough to cover each other's glitches?**

Every "glitch" we found is a signature telling us *which axis that encoding does
not carry*. That's not a bug — it's the negative space that tells us what the
other lens has to cover.

Examples from our own measurements:

- **u8 CDF** destroys magnitude and sign, keeps rank. Signature: quantile-only lens.
- **γ+φ rotation** proved ρ=1.000 vs plain CDF. Signature: monotone transform
  before rank = identity on rank. It carries nothing the base didn't already carry.
- **K=256 attractor collapse** (pre-softmax). Signature: ReLU destroys the local
  gradient structure that the semantic neighborhood relied on.
- **"thank thank thank"** at K=4096. Signature: high-frequency verb monotony —
  a bucketing invariant we never built.

Each glitch isolates exactly one axis. That's the matrix.

---

## The invariant axes

Eight axes, none of them optional, none of them substitutable for each other:

| # | Invariant | What it preserves | What destroys it |
|---|---|---|---|
| 1 | **Magnitude** | absolute value, energy, contrast | quantile mapping, rank-only |
| 2 | **Sign / direction** | inhibition, cancellation, polarity | unsigned encodings, abs() |
| 3 | **Rank / order** | top-k queries, monotone operations | noise floor, tie-breaking loss |
| 4 | **Pair identity** | (i,j) locality, adjacency, zipper | global mixing, random permutation |
| 5 | **Manifold curvature** | knee-point, local density, geodesic | flat distance, isotropic sampling |
| 6 | **Trajectory / stride** | sequence, start-offset, 128-step cache | frame-independent hashing |
| 7 | **Sparse structure** | zero-pattern, support set, hole | dense reconstruction |
| 8 | **Phase / rotation** | relative angle, interference | magnitude-only, rank-only |

Note: axes 1–3 are what the proven substrate already covers (i16 direct). Axes
4–8 are where the current work lives, and where every unresolved debate sits.

---

## Encoding candidates as rows

Each row declares which axis it *claims* to carry and which it *measurably* carries.
`✓` = measured. `?` = claimed but unmeasured. `✗` = measured to fail. `–` = doesn't
apply / doesn't claim.

| Encoding | Mag | Sign | Rank | Pair | Manif | Traj | Sparse | Phase |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **i16 direct** (proven lossless, ρ=0.997) | ✓ | ✓ | ✓ | ? | – | – | – | – |
| **u8 CDF** | ✗ | ✗ | ✓ | ? | – | – | – | – |
| **u8 + γ+φ rotation** (no-op, ρ=1.000 vs CDF) | ✗ | ✗ | ✓ | ? | – | – | – | ✗ |
| **11/17 prime modular (zipper)** | – | – | – | ? | – | ? | – | – |
| **Zeckendorf / Fibonacci** | – | – | – | – | – | – | ? | – |
| **CLAM tree** (ndarray has 46 tests) | – | – | – | – | ? | – | – | – |
| **CHAODA outlier graph** (on top of CLAM) | – | – | – | – | ? | – | – | – |
| **1/40 σ ring lens** (see ONE_FORTIETH_SIGMA_LENS.md) | ? | – | ? | – | ? | – | – | ? |
| **Sparse + zipper** | – | – | – | ? | – | – | ? | – |
| **128-step forward cache** (372K tok/s u8, Python/Rust contradiction open) | ? | ? | ? | – | – | ? | – | – |
| **Hole + Dish (SiLU radial field)** | – | – | – | – | ? | – | ? | ? |
| **7-lane HighHeelBGZ** (L1 broken, L2 lossless) | ✓ | ✓ | ✓ | – | – | – | – | ✗ |
| **Semantic codebook (256 Jina v5 forward passes)** | ✓ | – | ✓ | – | ? | – | – | – |
| **Multi-role composite (4 layers × 5 roles)** | ? | ? | ? | – | – | – | – | ? |
| **BF16 calibration reference** | ✓ | ✓ | ✓ | – | – | – | – | – |

Observations that fall out of this table *without running any new test*:

1. **Axes 4 and 5 (pair identity, manifold curvature) have no `✓` entries yet.**
   Every candidate that claims them is unmeasured. That's the biggest blind spot.

2. **CLAM is the only candidate for axis 5 that doesn't carry narrative debt** —
   it's already implemented in ndarray (46 tests), it has a published algorithm,
   and the knee-delta gives us a measurable scalar. Everything else on axis 5
   is speculative.

3. **i16 direct is the only row with three hard `✓`s.** It's the floor. Every
   other encoding has to earn its place by carrying an axis i16 direct doesn't.

4. **γ+φ has a `✗` in phase** — we measured it carries nothing. It should be
   removed from production roles, kept only as a documented negative result.

5. **u8 CDF has `✗` in magnitude and sign** — this isn't a bug, it's the
   *definition* of quantile mapping. The question isn't "fix u8 CDF." The
   question is "what do we pair it with to cover magnitude and sign?"
   Answer: i16 direct, at the cost of 2× storage. That's the honest trade.

6. **The 7-lane HighHeelBGZ row shows the current production paradox:** lanes
   that are individually broken (L1 u8 CDF broken, γ+φ no-op) are stacked
   together without a declaration of which axis each lane covers. That's the
   frame-confusion we kept hitting. Once lanes are declared by invariant, the
   stack becomes legible: L2 covers magnitude, BF16 calibration covers reference,
   and the remaining lanes either carry a specific axis or they're dead weight.

---

## What the matrix lets us decide

Instead of "which encoding is best," the architectural question becomes:

> **Pick the minimum set of encodings whose `✓`s cover all the invariants you need
> for the current task.**

Task-specific coverage needs:

| Task | Required invariants | Minimum lens set |
|---|---|---|
| SPO extraction | mag + rank + pair | i16 direct + (zipper or CLAM, once measured) |
| Contradiction detection (NARS) | rank + manifold + phase | CLAM + 1/40 σ + multi-role composite |
| 128-step forward cache | mag + trajectory | i16 direct + 128-step cache (needs Python/Rust reconciliation) |
| Semantic grounding (Wikidata hydration) | mag + rank + manifold | i16 direct + semantic codebook + CLAM |
| Contrastive learning | rank + pair + sparse | 11/17 zipper + sparse-zipper + rank |
| Hole + Dish focus field | sparse + manifold + phase | sparse-zipper + CLAM + 1/40 σ |

The point: **no single encoding is the answer.** The architecture is always a
small set of lenses whose `✓`s union to cover the task.

This also means we can **retire encodings** when a stronger lens covers the
same axis. γ+φ goes first (nothing to retire from — it covers nothing). u8 CDF
stays only as long as we don't have a cheaper rank-only encoding.

---

## What the matrix does NOT let us decide

Three things the matrix deliberately refuses to answer:

1. **Which axis is "more important."** That depends on the task. Ranking axes
   globally is the mistake we keep making when we call one encoding "central."
2. **Whether an encoding is elegant.** Elegance is aesthetic. The matrix only
   tracks what's preserved and what's destroyed.
3. **Whether an encoding will work at runtime performance targets.** That's a
   separate measurement (SIMD-friendliness, cache behavior, allocation pattern).
   Invariant preservation and runtime speed are orthogonal and should never be
   conflated in a single cell.

---

## Next additive steps (no production code)

1. **Invariant measurement harness** — one new example file,
   `crates/thinking-engine/examples/invariant_matrix_bench.rs`, that takes
   existing codebooks and existing lenses and emits a CSV of
   `(encoding, invariant, score)`. Uses only public APIs. Touches no library code.
2. **CLAM knee-delta probe** — one new example file,
   `crates/thinking-engine/examples/clam_knee_probe.rs`, that runs the existing
   ndarray CLAM tree over the semantic codebook and reports the knee-point at
   each depth. Measures axis 5 for the first time.
3. **Pair-identity test** — one new example file,
   `crates/thinking-engine/examples/pair_identity_probe.rs`, that measures
   mutual information between adjacent centroids under the 11/17 zipper vs a
   random permutation. Measures axis 4 for the first time.
4. **Proposed but not written yet:** `1_40_sigma_probe.rs` and
   `ring_perturbation_probe.rs`. See companion notes for what they'd measure.

All four probe files would be **new examples**, not edits to library code. They
would compile against the existing workspace without modifying any existing file.

---

## Honest scope

- **What this note gives us:** a frame for distinguishing measured from
  unmeasured claims, and a language for deciding which encodings to add, keep,
  or retire.
- **What this note does NOT give us:** any new measurement. Every `?` in the
  table is still a `?`. The matrix is only as good as the probes we actually run.
- **What would collapse this note:** if the probes come back and say every
  axis is explained by i16 direct + a single corrective lane. Then the matrix
  was overbuilt and we should simplify. That's a good failure mode — it means
  the architecture is simpler than we thought.

---

## One sentence that should survive any refactor

> Every encoding is a lens on a subset of invariants; the architecture picks
> complementary lenses, not winners.
