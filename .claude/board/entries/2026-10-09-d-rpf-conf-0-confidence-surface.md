# 2026-10-09 — D-RPF-CONF-0: the confidence revision surface, measured before any policy

**Status:** MEASURED. Tests only; no production semantics changed, no policy
chosen, no quantizer unified, no contract declaration added (T5 stays blocked).
**Harnesses:** `crates/causal-edge/tests/d_rpf_conf_0.rs` (T1, singular, T3,
R1, NarsTables invariance) and `crates/lance-graph-planner/tests/d_rpf_conf_0_r3.rs`
(R3). Both run in under a second in release.
**Parent:** `2026-10-08-coresearch-ce64-bf16-for-thinking.md` (T1, T3, R1, R3,
G4 items of its TO-DO / RESEARCH / GAP list).

Rule held throughout: defined somewhere ≠ live ≠ numerically distinct.

## T1 — Rust confidence surface vs the exact integer form

Reference: `c = 255·N/D`, `N = c1(255−c2) + c2(255−c1)`, `D = 255² − c1·c2`,
rounded half up, computed in integers. 65,535 ordinary cells compared (the
singular cell excluded, see below).

- **3 deviations, all by 1 code:** (45,45) Rust 76 vs exact 77, and
  (146,148)/(148,146) Rust 187 vs exact 186.
- The numpy hypothesis `{(45,45),(85,165),(165,85)}` was right on one cell of
  three. (85,165) is an exact tie that Rust rounds correctly; (146,148) is a
  near-tie (exact 186.499988) that f32 rounds the wrong way.
- **8 exact ties** on the grid, pinned: (3,75), (45,45), (75,3), (85,85),
  (85,165), (153,225), (165,85), (225,153). One of them, (45,45), is a deviation.
- Monotonicity violations 0; symmetry violations 0.
- Confidence output does not depend on frequency (all 256 f pairs tested).
- f side, sampled on c ∈ {1,64,128,200,254}²: 6,932 deviations of 1 code. All
  6,932 sit on exact .5 ties (163,840 ties in the sample). There is no
  deviation away from a tie.

Pinned in `T1_DEVIATIONS` / `T1_TIES`. The falsifier flips one reference cell
and the deviation set changes.

## Singular cell (255,255)

Kept out of the ordinary domain. `w = f32::MAX` on both sides, `ws = inf`, the
pooled frequency is `inf/inf = NaN` and the confidence is `inf/inf = NaN`; both
encode to 0. **Measured: (255,255) → (0,0) for every f pair.** The test is named
`t1_singular_cell_is_recorded_not_endorsed`: this is what the code does, not a
value anyone chose.

## T3 — zero-weight identity and null operands

Over all (f,c), with the other operand g ∈ {0,128,255}, in both orders:

- **255 drift cells**, all with c = 0 and f ≠ 128. Each becomes (128, 0).
  max delta 128.
- Neither g nor operand order leaks into the result.
- There is no drift for c > 0: revision with a zero-weight partner returns the
  input exactly.

So the drift is semantic, not arithmetic: no evidence on either side resolves
to (0.5, 0), and the original frequency is dropped. `forward` with both weights
zero faults with `IsaFault::Unsupported { mantissa: 0 }`.

## R1 — c=255 policy candidates, side by side (no choice made)

Columns: NaN on the c surface / NaN on the f boundary (c ∈ 253..255) /
monotonicity violations (singular included) / symmetry / c cells off the exact
form / their max error / cells changed vs status quo / the singular cell /
one-sided f `c1=255, c2<255`: off exact, max error, changed.

| policy | NaN c | NaN f | mono | sym | off exact | max | changed | (255,255) | 1-sided off | max | changed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| status quo | 1 | 65,536 | 1 | 0 | 3 | 1 | 0 | (0,0) | 0 | 0 | 0 |
| cap before weight (0.9999) | 0 | 0 | 0 | 0 | 3 | 1 | 1 | (128,255) | 101,900 | 6 | 101,900 |
| cap input at 254 | 0 | 0 | 0 | 0 | 513 | 1 | 511 | (128,254) | 194,671 | 128 | 194,671 |
| saturate at 254 | 0 | 0 | 0 | 0 | 513 | 1 | 511 | (128,254) | 194,671 | 128 | 194,671 |
| singular only | 0 | 0 | 0 | 0 | 3 | 1 | 1 | (128,255) | 0 | 0 | 0 |

Reading, without a recommendation:
- The status quo is **exact on the one-sided boundary** (one dogmatic premise
  wins outright). Every cap or saturate candidate breaks that: cap-before-weight
  by up to 6 codes on 101,900 cells, the 254 variants by up to 128 codes on
  194,671 cells.
- "Singular only" (status quo except (255,255) → (mean f, 255)) removes all NaN
  and the one monotonicity violation and changes exactly one cell. That makes
  G1 a single, explicit decision about one cell, not a policy over a surface.
- The f-boundary NaN count (65,536) is the f surface at (255,255) — every f
  pair there is NaN.

The candidates are replicas of the production f32 path; `r1_replica_of_status_quo_is_faithful`
checks the status-quo replica against `CausalEdge64::revision` on every cell.

## G4 — census of revision laws

Closed search: `fn (revise|revision)\b` over `crates/**/*.rs`. **N = 24
definitions in 20 files.** (The earlier "37" counted files containing the words,
not definitions.) Every call site used for a live/not-live claim was read.

- **9 homonyms, not NARS revision:** Gadamer interpretive revision
  (`contract/revision.rs:308`, `:327` and three example wrappers), kanban column
  routing (`kanban.rs:231`), a counterfactual trait declaration and its test
  stub, `planner/pearl.rs:373`.
- **Delegations, not new laws:** arigraph orchestrator → `spo::TruthValue`;
  `CausalEdge64::revision` → `isa::truth::revision`;
  `GrammarStyleAwareness::revise` → `revise_truth`.
- **Distinct laws reaching production in some crate:** A (ISA f32), B(L)
  (NarsTables), C (ndarray evidence split), D (planner `TruthValue`), E
  (contract `NarsTruth`, cap 0.99), F (`spo::TruthValue`, `+EPS`), G (holograph
  u16), J (`PathTruth`, raw weight argument), LIN (c used as weight; three
  copies). H (`learning`) is private behind `wip` and cannot be called.
- **Planner path, read in code:** `NarsEngine::new` builds
  `NarsTables::build(1)` (`cache/nars_engine.rs:451`). `replay_step`
  (`chain_replay.rs:196-207`) calls `tables.revise` and **overwrites** the
  truth `forward` computed with law A. That chain is reached through
  `replay_chain` ← `counterfactual_replay` ← `pearl::reason`, all public
  production functions. Every in-tree caller passes `build(1)`, except
  `tests/chain_confidence.rs` (16). So on the replay/counterfactual path the
  emitted truth is **B(1)**.

  > ⊘ Corrected by D-RPF-TABLE-1 (`2026-10-09-d-rpf-table-1-fast-path-cost.md`):
  > the replay path emits whatever tables its caller passes, and every caller
  > in the tree is a test, example or probe. No production code constructs a
  > `CutContext` or reads `NarsEngine::tables`, so B(1) changes no production
  > decision today.
- Not in the census, found while verifying it: `contract/high_heel.rs:240`
  `revise_truth` is another LIN copy with a 0.99 cap. The census sweep of
  differently named pooling functions was incidental, not exhaustive, so this
  is unsurprising; it is recorded so the count is not read as closed.

**NarsTables `build(1)` is confidence-inert, measured:** one distinct output per
f pair over the whole c grid; c is always 170 and f is the mean. `build(16)`
varies (92 distinct c outputs at f = (128,128)).

## R3 — disagreement matrix over the distinct live laws only

Laws: A, B1, B16, C, D, E, LIN, each through its real entry point. F and G are
off the planner path and unreachable from it; J needs a declared c→weight rule
that no caller fixes; H is not callable. f32 laws are quantized with the
`set_*_u8` rule; A and B keep their native encoding.

Domain: f ∈ {0,51,128,204,255}², c over the full 256×256 grid (1,638,400 cells).
"Broad" = neither c in {0, 254, 255}.

Distinct output c at f = (128,128): A 256, B1 **1**, B16 92, C 256, D 256, E 253, LIN 171.

| pair | cells differing | of those broad | broad max dF | broad max dC | mean dC |
|---|---|---|---|---|---|
| A–D | 0 | 0 | 0 | 0 | 0 |
| A–C | 391 | 130 | 1 | 1 | 0.004 |
| C–D | 391 | 130 | 1 | 1 | 0.004 |
| A–E | 38,897 | 13,327 | 15 | 2 | 0.051 |
| C–E | 39,025 | 13,455 | 15 | 2 | 0.047 |
| A–B16 | 1,585,273 | 1,547,346 | 113 | 14 | 3.27 |
| A–B1 | 1,636,550 | 1,598,385 | 128 | 168 | 47.6 |
| A–LIN | 1,636,238 | 1,598,618 | 107 | 126 | 59.6 |
| B1–B16 | 1,630,720 | 1,592,545 | 128 | 155 | 47.3 |
| E–LIN | 1,636,218 | 1,598,618 | 105 | 125 | 59.6 |

(Full 21-pair matrix pinned in `MATRIX`.)

The falsifier: a pair whose broad cells never differ by more than one code is a
cosmetic difference. Measured, and pinned from the measurement:

- **Cosmetic: A–C, A–D, C–D.** A and D are byte-identical on this domain (D's
  `MAX` weight is reached only at c = 1.0, which on the u8 grid is code 255,
  the same single code as A's `c ≥ 0.999`). C differs from both only at the
  boundary and by one code in the broad domain. Unifying A, C and D would be
  cosmetic in the broad domain; their real difference is the c = 255 policy.
- **E differs only through its 0.99 cap:** broad dC ≤ 2; the dF of 15 comes
  from c = 253, which the cap moves from w ≈ 126 to 99.
- **Not cosmetic: B1, B16, LIN** against everything. B1 differs from A on 99.9%
  of cells by up to 168 confidence codes. **B1 is the law the replay and
  counterfactual path emits.**

So M = 7 laws compared, which collapses to 4 numerical behaviours on the u8
grid away from the boundary: {A, C, D} (E is this one plus a cap), B16, B1, LIN.

## Disable runs (all red as required)

- T1: exact rounding changed to truncation → pins fail.
- R1: "singular only" singular c changed to 254 → table pin fails.
- NarsTables: `build(1)` changed to `build(2)` → invariance pin fails.
- T3: identity measured with `rev(f,c,0,1)` → drift pin fails.
- R3: law D replaced by a constant-c bucket → distinct-c, matrix and cosmetic
  pins fail.
- R3: the broad/boundary split disabled → matrix pin fails.

One test was vacuous and was fixed: the first `r3_cosmetic_pairs_are_pinned`
filtered the pinned `MATRIX` constant, so it passed under the law-D disable. It
now filters the measured matrix and fails under that disable.

## What this does not decide

- G1 (the (255,255) cell) and G2 (whether replay should emit B1) are operator
  decisions. This entry gives each one its measured cost.
- T5 (rounding/saturation declarations in `isa::contracts`) stays blocked.
- No law is declared canonical.

```
STATUS: measured | OUTCOME: T1 3 one-code f32 deviations; (255,255)->(0,0);
T3 drift only at c=0; R1 table (singular-only changes 1 cell, caps change
up to 194,671); G4 24 defs -> 7 compared laws -> 4 broad behaviours;
replay path emits confidence-inert B(1) | OPEN: G1, G2, T5; J needs a c->w
rule; F/G unmeasured on the u8 grid
```

MIRROR
- BLIND SPOT: R3 samples five f values; the f side of the matrix is not
  exhaustive, and the "broad" boundary at {0,254,255} is a chosen cut.
- BIAS CHECK: did the numpy prediction anchor the T1 search? The Rust run was
  compared against the integer form, not against the prediction, and the
  prediction was wrong on two of three cells.
- STILL OPEN: whether B1 on the replay path was intended (a fast path) or an
  accident; nothing in the code says which.
