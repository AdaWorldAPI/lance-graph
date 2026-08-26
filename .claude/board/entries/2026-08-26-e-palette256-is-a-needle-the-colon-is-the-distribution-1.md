## 2026-08-26 — E-PALETTE256-IS-A-NEEDLE-THE-COLON-IS-THE-DISTRIBUTION-1 — one index finds a point; only a PAIR carries a distribution, which is why the Fisher-z diagonal returns a constant and why that is the table being honest rather than broken

**Status:** FINDING [MEASURED] for the diagonal numbers and the 6.26×
resolution arithmetic (probe run 2026-08-26, public API only, outside
the repo — no crate was mutated); **operator-stated** for the framing
and for the production provenance (*"we use it since a year or so"*).
Carries a **correction to a conclusion already merged on `main`**
(Q7b, `E-Q7-…-COMPLEMENTARY-NOT-COMPETING-1`).
**Confidence:** High. The framing is not a metaphor — it predicts the
measured diagonal behaviour and the gamma-exclusion arithmetic, both
of which were measured after the framing was stated.

### The split

> **palette256 can find a point like a needle in a haystack.
> For distribution it needs pairwise.** (operator, 2026-08-26)

| layer | the question it answers | carrier |
|---|---|---|
| `palette256` index | **which one** — needle | the byte itself, exact, O(1); prefix-routable under the canon's 4⁴-hierarchical condition, so the needle is found by radix descent, not search |
| `palette256`**`:`**`palette256` | **how related** — distribution | the Fisher-z i8 k×k LUT (`bgz-tensor::fisher_z::FisherZTable`) |
| self-similarity | neither — **1.0 by definition of the address** | the addressing layer, never a lookup |

**The colon IS the pairwise-ness.** `6×(u8:u8)` is six *relations*, not
twelve indices — which is exactly why the canon never writes
`palette65536`. A single index says *which*; it cannot say *how far*,
*how spread*, or *how related*, because those are relational by
construction and need `(a,b) → value`.

### What this relocates: the diagonal is a CATEGORY ERROR at the call site

Measured on four fixtures through the public API. `lookup_f32(a,a)`
**never returns 1.0** — it returns the largest off-diagonal cosine, and
returns the *same* value for every centroid:

| fixture | off-diagonal cosines | `lookup_f32(a,a)` | error vs 1.0 | off-diagonal round-trip err |
|---|---|---|---|---|
| tight cluster k=64 | 0.961 … 0.985 | 0.9847 | **1.5 %** | 0.00013 |
| hand-built spread | 0.100 … 0.912 | 0.9117 | **8.8 %** | 0.00396 |
| k=256 pseudo-random | −0.454 … 0.477 | 0.4769 | **52.3 %** | 0.00397 |
| orthogonal basis | all 0.000 | 0.0000 | **100 %** | 0.00000 |

`diagonal == max off-diagonal` was **true in all four**;
`every diagonal entry identical` was **true in all four**. The
off-diagonal path is sound (max abs error 0.00013–0.00397), so the
certified ρ≥0.999 is **not** contradicted — the effect is diagonal-only.

Read through the split, the constant is not a defect in the table: it
is the **distribution carrier correctly reporting that it holds no
point information.** The error is asking it a needle question.
`morton_cascade/legacy.rs:23` does exactly that — `coh +=
fz.lookup_f32(a, a)` for a bare index with no `b`, its own comment
calling it *"the self-cosine"* — so that sum is `n × constant`, a
counter in the costume of a coherence measure. `legacy` is **live**
(`Backend::Legacy`; `morton_cascade/mod.rs:109` runs v3 and legacy
together), not deprecated.

### Excluding the diagonal from gamma is NECESSARY, not sloppy

`FisherZTable::build` fits `FamilyGamma` from off-diagonals only
(`if i < j`). That is load-bearing. On the tight-cluster fixture
off-diagonals span `z ∈ [1.954, 2.433]`, range **0.479**. Including the
diagonal forces `z_max = atanh(0.9999) = 4.952`, so the range becomes
**2.997 — 6.26× wider, hence 6.26× coarser quantisation on the
certified path** — spent to encode a value that is 1.0 for every
centroid. Fitting the distribution to a constant would wreck the very
ρ the table exists for.

**So the fix is not in `build()`.** Self-similarity is a property of
*being the same address*; `if a == b { return 1.0 }` is right precisely
because it answers from the addressing layer. (Not applied here: any
consumer calibrated against today's constant shifts when the diagonal
becomes real — a behaviour change in a live path, needing its own
measurement.)

### ⊘ CORRECTION to `E-Q7-…-COMPLEMENTARY-NOT-COMPETING-1` (merged, #1038)

Q7b ran `PAL` as exact whole-chain matching with **no distance table
anywhere** — that is a **needle** test, not a distribution test. It is
why PAL saturated to a 1.0 hit-rate, and why its 0.99–1.00 transfer is
unsurprising: exact addresses transfer when the address space is
shared. The Q7b *measurements* stand for the carrier actually built;
what does not stand is reading them as evidence about
`palette256:palette256`-as-distribution. Q7b measured palette256 as a
needle and reported it as though it spoke to the pairwise rail.

### Universality (operator-stated; the boundary is mine)

> *"the amazing thing is you can convert everything into a Fisher-z
> LUT — apples, oranges, cider, the age of your grandmother, the
> beginning of the universe."*

The carrier is **domain-blind by construction**, and three independent
properties make that work: a **codebook** makes anything addressable
(≤256 centroids); **arctanh** makes the tails resolvable where raw
values crowd; **per-family gamma** (`z_min`, `z_range`, fitted per
family) makes it **scale-free** — a family of years and a family of
cosines each fill the whole i8 range regardless of absolute magnitude.
Nothing in the table knows what its axis means.

**The honest boundary:** this holds *given a bounded similarity in
[−1,1]*. Choosing that similarity for a new domain is the modelling
step, and it is where the domain re-enters — it does not disappear, it
moves to the ClassView. Which is the same three-layer separation as
above: **address** in the byte, **relation** in the LUT, **meaning** in
the ClassView, none contaminating the others.

Per `I-NOISE-FLOOR-JIRAK` the arctanh **tail-stretch** is unconditional
(pure reparameterisation — what earns the ρ); the **variance
stabilisation** (SE `1/√(n−3)`) assumes IID sampling and does not hold
for weakly-dependent bits. The shipped code stays on the right side:
gamma normalises by min/max **range**, never by a variance, so it never
implicitly claims the IID SE. z here is a *quantisation* device, never
an *inference* device.

Cross-ref: `bgz-tensor::fisher_z` (the table),
`lance-graph-contract::distance` / `awareness_facet` (*"Fisher-z
cosine-replacement, never a float"*), `helix::fisher_z`
(`hyperbolic_depth = 2·arctanh(r)` — the **spatial** flavour, twice the
statistical one, same `ln` core; `Signed360` carries it as rim+polar+
azimuth), `E-Q7-…-1` (corrected above), `I-NOISE-FLOOR-JIRAK`.

