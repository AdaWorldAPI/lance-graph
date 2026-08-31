# pillar11-signature-certification-unification-v1

**Status:** ACTIVE — W0 green-lit and executing (operator, 2026-08-31);
Q1–Q3 ruled same day: **Q1 = a new jc pillar slot** (W3's home), **Q2 =
cross-repo sigker call** (W4's kernel source; an f32 port waits for W5's
trigger), **Q3 = docs only** (the ndarray rename touches no registry slot
id). Original line preserved below.
~~PROPOSED (plan-only — no wave runs without operator green-light).~~
**Scope:** lance-graph `crates/jc/src/hambly_lyons.rs` × ndarray
`src/hpc/pillar/signature.rs` × sigker × the two signature-kernel papers.
**Born from:** the 2026-08-31 D-SK arc (E-LEVY-AREA-COEFFICIENT-BEATS-
REFINEMENT-1, E-ORIENTATION-BIT-PARTIAL-NIBBLE-SUFFICES-1,
E-MONOTONE-STREAM-LEVEL2-IS-DISCRIMINATION-NOT-MAGNITUDE-1) and the
same-session census that found two "Pillar 11"s certifying different things.

## 1. What exists, and what each one MEASURES today

| | lance-graph jc `hambly_lyons.rs` | ndarray `hpc/pillar/signature.rs` |
|---|---|---|
| name | "Pillar 11 — Hambly-Lyons: signature uniqueness on tree-quotient" | "Pillar-11 — Hambly–Lyons signature transform (B7)" |
| activation | feature `hambly-lyons` (default OFF, deferred; pulls sigker) | always in the pillar battery (`prove_pillar_11`) |
| object | `sigker::signature_truncated`, d=3, depth 2, f64 | own `signature_d2_deg3`, d=2, degree 3, f32, Chen accumulation |
| claim certified | the UNIQUENESS theorem's operational content: out-and-back ≈ identity (tree-forward, ε=1e-9), triangle ≉ identity (converse, δ=0.05), discrimination ratio > 1e6 over 100 random pairs | kernel STABILITY: PSD Gram over 1000 Brownian paths (sampled Sylvester) + self-kernel concentration across halves |
| what it does NOT touch | the Goursat PDE solver (explicitly avoided, citing a pre-#350 divergence) | uniqueness, tree-quotient, the PDE solver, d>2, depth>3 |

**The two batteries certify DISJOINT properties of the same mathematical
object under ONE shared name.** Neither is redundant; both are partial.

## 2. Overlaps and frictions (the census findings, now actionable)

- **F-1 name collision.** Two "Pillar 11"s across repos, different content.
  A session reading "Pillar 11 is green" cannot know WHICH claim held.
- **F-2 misnomer.** ndarray's `sigker_hl` labels a plain truncated
  signature kernel "Hambly–Lyons kernel". HL 2010 is a uniqueness theorem,
  not a kernel construction; the label imports authority the code does not
  carry.
- **F-3 stale divergence note.** jc's module doc says
  `signature_kernel_pde` "diverges from the true signature kernel at
  moderate inner products (PR #350 documents the corrected form)". Measured
  this session on the CURRENT solver: rel err 6.25e-5 (d=3 linear anchor,
  N=256) and 4.53e-4 (d=24) — the note describes the pre-fix state and now
  steers readers away from a solver that is a valid reference.
- **F-4 computation duplication with no parity bridge.** sigker
  `signature_truncated` (reference, f64, any d/depth) and ndarray
  `signature_d2_deg3` (hardware, f32, fixed shape) implement the same
  iterated integrals with zero cross-checks. The workspace's own
  architecture rule (ndarray = hardware, lance-graph = thinking) blesses
  the split but demands the parity test it never got.

## 3. What the papers say these batteries COULD measure — and do not yet

Papers: **PowerSig** (arXiv 2502.20392 — tile-local Neumann/power-series
Goursat solves, O(ℓ·P) memory, ℓ ≥ 10⁶) and the **rough higher-order
solver** (OpenReview 1fycT4ZRf1 — increment-only coefficients are
first-order; a PDE system whose cell coefficients carry higher-order
iterated integrals has a unique solution with quantitative error bounds).

- **M-1 Solver-order certification (rough paper §error bounds).** Nothing
  certifies the Goursat solver's CONVERGENCE ORDER. The D-SK probes
  measured it ad hoc (first-order plateau under aliasing; >10× recovery
  from level-2 coefficients in the super-period regime; regime boundary =
  window vs oscillation period). The paper supplies the quantitative bound
  the empirical slope should match — a falsifiable pillar: measured order
  within tolerance of the proven order, plus the aliasing-plateau
  can-stay-silent half.
- **M-2 Depth-∞ uniqueness (HL's actual statement).** jc certifies
  tree-quotient at DEPTH-2 TRUNCATION. Hambly-Lyons is a depth-∞ theorem;
  the PDE kernel gives depth-∞ access without materialization:
  tree-equivalent paths must satisfy K(x,x)=K(x,y)=K(y,y) to solver
  tolerance, non-tree perturbations must not. Blocked only by F-3's stale
  note.
- **M-3 PSD at depth-∞.** ndarray's PSD certification covers the truncated
  d=2/deg-3 kernel only. The PDE kernel's Gram PSD-ness (the property
  kernel machines actually rely on) is uncertified everywhere.
- **M-4 Higher-order coefficient fidelity as a BATTERY.** The D-SK-A/B′
  gates (area-domain RMS ordering; the kernel-scalar cancellation trap —
  "never gate carrier fidelity on kernel-scalar error near the
  discretization floor") live only in probe examples. Neither battery
  knows the trap; a future pillar comparing kernels could silently step
  into it.
- **M-5 Scalability regime (PowerSig).** Nothing measures long-path
  behaviour (memory O(ℓ²) today). DEFERRED BY DESIGN: no workload above
  ℓ ≈ a few hundred windows exists yet — measure-then-pin says this wave
  waits for the workload, not the paper.

## 4. The waves (each gated, falsifiability pairs mandatory)

**W0 — hygiene, zero behaviour change (lance-graph + ndarray, 2 small PRs).**
Fix F-1/F-2/F-3: jc's pillar keeps the HL name (it certifies the HL
theorem); ndarray's doc header renames its claim to what it is
("truncated signature-kernel stability battery", keeping the B7/Pillar-11
slot id and noting the cross-repo disambiguation); `sigker_hl` doc-comment
drops the HL attribution for the kernel itself; jc's divergence note is
re-dated as pre-#350 history with this session's measured anchors
(6.25e-5 / 4.53e-4). Gate: docs-only diff; both suites untouched-green.

**W1 — the parity bridge (F-4).** One test, home ndarray (hardware side
proves itself against the reference): random d=2 paths → `signature_d2_deg3`
vs sigker `signature_truncated(d=2, depth=3)`, f32-tolerance parity, plus
the anti-vacuity half (a deliberately wrong Chen accumulation fails).
Cross-repo dep is test-only and sibling-checked-out, the jc pattern.
Gate: parity ≤ f32 eps bound over ≥1000 paths; red-then-green on the
sabotaged accumulator.

**W2 — depth-∞ uniqueness leg in jc (M-2, unblocked by W0).** Extend the
`hambly-lyons` feature: same 100 pairs, PDE-kernel leg —
|K(x,y)/√(K(x,x)K(y,y)) − 1| < tol for out-and-back vs base, ≫ tol for
triangle loops. Falsifier pair inherited from the existing legs.

> **⊘ AMENDED 2026-08-31 (post-merge codex P2 pair — both correct, both
> accepted):**
> 1. **The raw 3-point fixture is NOT tree-invariant under the
>    first-order scheme.** For an out-and-back loop with increment `u`
>    and `a = ‖u‖²`, the discrete recurrence gives `K(x,x) = 1 + a²`
>    against 1 for the constant base — normalized `1/√(1+a²)`, a
>    DISCRETIZATION artifact, not a uniqueness failure. The leg therefore
>    RESAMPLES each segment to `N` points and gates against a
>    resolution-dependent tolerance `ε(N)` derived from a pre-registered
>    refinement sweep (measure the convergence slope first, then pin
>    `ε(N)`; a fixed grid-blind tolerance would fail typical pairs for
>    reasons that have nothing to do with the theorem).
> 2. **The original anchor was void.** The depth-2 forward fixture
>    cancels EXACTLY under Chen accumulation — `hambly_lyons.rs` maps the
>    zero forward distance to `f64::INFINITY` — so "within 10× of the
>    depth-2 discrimination ratio" compared against infinity. Replaced:
>    the gate binds on FINITE absolute statistics — forward
>    `< ε(N)`, converse `> δ` (δ = 0.05, the existing leg's threshold),
>    and converse/forward `> 1e3` at the sweep's chosen `N`. Nothing is
>    anchored to the truncated leg's ratio.

Gate (amended): refinement sweep pre-registered; forward < ε(N);
converse > δ; converse/forward > 1e3 at the pinned N.

**W3 — solver-order + carrier-fidelity battery (M-1, M-4).** Promote the
D-SK probe gates into a repeatable battery. Home: jc (a new pillar slot,
NOT another "11"), feature-gated like `hambly-lyons`, pulling sigker.
Contents: (a) empirical convergence order of increment-only vs
level-2-augmented solves on the oscillatory fixture family, gated against
the rough paper's bound shape (order gap > 0.5 in the super-period
regime; aliasing plateau as the silence half); (b) the carrier-fidelity
rules as executable law — area-domain RMS ordering gate + a test that
DEMONSTRATES the kernel-scalar cancellation trap (a coarser carrier
measurably "beating" a finer one near the floor) so the trap is a red
test, not a prose warning. Gate: all D-SK numbers reproduced within
noise; the trap demonstration fires.

**W4 — PSD at depth-∞ (M-3).** ndarray side (it owns the PSD method):
`prove_pillar_11`'s Gram machinery re-run over `signature_kernel_pde`
values (computed via the W1 bridge or a local Goursat port — decision
point, see Q2). Gate: PSD + concentration at depth-∞ over the same
Brownian pool.

**W5 — PowerSig scalability (M-5). DEFERRED** until a workload with
ℓ ≳ 10⁴ windows exists. Entry criterion written down now so it is a
trigger, not a judgment call: first real stream whose Goursat solve
exceeds 1 GiB or 10 s.

## 5. Open questions (operator input before any wave beyond W0)

- **Q1:** W3's home — a new jc pillar slot vs a sigker-internal battery?
  jc keeps the certification franchise in one place; sigker keeps the
  zero-dep constitution clean either way (jc already depends on sigker
  under the feature).
- **Q2:** W4's kernel source — cross-repo call into sigker (test-only
  sibling dep, jc precedent) vs a minimal f32 Goursat port into ndarray
  (hardware-side, SIMD-able later, but a second implementation needing its
  own parity gate)? Default recommendation: cross-repo first, port only
  when W5's trigger fires.
- **Q3:** does W0's ndarray rename touch the pillar REGISTRY slot id
  (B7/11) or docs only? Recommendation: docs only — slot ids are stored
  history.

## 6. Falsifiability discipline carried over

Every wave inherits the D-SK method findings as law: kernel-scalar error
near the discretization floor never gates carrier fidelity (gate in the
area/feature domain); every filter/guard ships its can-fire AND
can-stay-silent halves with non-trivial inputs; convergence-order gates
bind in the regime the theory names (W relative to period), never on a
fixed absolute range.
