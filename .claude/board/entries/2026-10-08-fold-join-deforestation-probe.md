# 2026-10-08 — Fold-Join deforestation probe (D-RPF-9)

**Status:** MEASURED. Probe only, no production primitive. Branch
`ccr-b2e415d9-4jfvyk-fold-join`, commit `1c43faa5`:
`crates/lance-graph-quack/examples/fold_join_probe.rs`.

```text
CARGO_PROFILE_RELEASE_DEBUG=0 cargo run --release -p lance-graph-quack --example fold_join_probe
```

The question: when two address-aligned membership sources meet in one
terminal, which shipped lowering already avoids the intermediate, and what does
each arm actually write? Every arm runs the shipped mask-risc executor or
quack's lowering into it. Each one is checked against a per-row oracle on
65,536 rows.

## Capability inventory (VERIFIED-IN-CODE)

| shape | where it lives | intermediate |
|---|---|---|
| Boolean chain over ≤ 6 resident planes → `Count`/`Any` | `Program::fused_ternlog` / `fused_tern2` / `fused_tern3` (`mask-risc/src/ir.rs`) → `mask_ternlog_popcount` | none: no slot, no bitmap |
| the same chain → `Keep` | `fused_keep`, `Tern2/Tern3` with `Out::Mask` | the demanded bitmap only |
| any `Pred` or `Gather` in the chain | `Lowering::Tiled` (`exec.rs`, `TILE_WORDS = 256`) | one 2 KB tile per slot; no population mask (tiled law, 2026-09-21) |
| `Pred B under A` | `MaskOp::Pred { under }`; quack `lower` gates every conjunct on the accumulator (`emit_gated`) | as tiled |
| strided predicates under a gate | `exec.rs` `run_pred`: no `_under` twin, so the full kernel runs and is then ANDed | as tiled, and B is evaluated on rejected rows |
| `lower_fused` | one slot per comparison, ternlog skeleton, never gated | as tiled |

What the Cognitive Shader Driver has is a different operation:
- `Quad8::fold_product` folds the Cartesian product of ONE object's occupancy
  coordinates without building it. That is a precedent for not materialising a
  product, but it is not an aligned intersection of two populations.
- `MergeMode::{Xor, Bundle, Superposition}` are commit semantics for deltas.
- The rotated-XOR fingerprint in `driver.rs` is a lossy aggregate.
- `thinking-engine` superposition multiplies f32 amplitudes, which A2 forbids
  for masks.

No CSD arm was run (arm G): none of these computes `popcount(A ∧ B)`.

## Results

### Correctness

Every arm equals the oracle:
- **Edge cases:** all-zero, all-one, disjoint, identical, one live bit, n = 100,
  n = 65,573.
- **A bitmap the terminal asks for** (`Keep(A ∧ B)`) is written in full and
  equals the oracle.
- **One intermediate feeding two folds:** `|A∧B| + |A∧¬B| = |A|`.
- **A terminal without a merge law** (`BlendI32` over a partial extent) is
  refused.

### Timings

Timings are ns per execution on this host, 64K rows. They are noisy run to run
(about ±10 %); two runs were taken and agree on every ordering below.

| arm | what | scattered A sel 0.001 / 0.01 / 0.1 / 0.5 / 0.9 |
|---|---|---|
| B | resident A ∧ B → Count (no slot) | 445 / 387 / 382 / 384 / 385 |
| A | Keep A, Keep B (16 KB population masks), then fold | ~18.5 k at every selectivity |
| C | Pred A, Pred B, And, Count | 18.7 k / 17.9 k / 18.2 k / 18.0 k / 18.0 k |
| D | Pred A, Pred B under A | 13.5 k / 19.6 k / 25.2 k / 25.1 k / 25.9 k |
| Dp | resident mask A gates Pred B | 3.9 k / 10.0 k / 15.8 k / 15.7 k / 16.2 k |
| H | per-row scalar oracle (reference only) | 14.6 k – 18.3 k |

- **Clustered A** (survivors in one run; dead-tile fraction 0.75 at sel ≤ 0.1):
  D 12.6 k / 12.3 k / 13.7 k / 19.8 k / 25.0 k against C 23.1 k / 18.2 k /
  17.6 k / 19.3 k / 18.8 k.
- **Quack:** `lower` tracks D and `lower_fused` tracks C at every selectivity.
- **E (`AndNot` / `Xor` / `Or` / majority over resident masks):** all take the
  no-slot lowering, 362–1213 ns.

### CE64 read in place from `MaterializedEdges`

`MatchFacet16Strided` on 16-byte windows equals the `CausalEdge64` accessors.
- **Can fire:** a care shifted by one bit disagrees (227 vs 282).
- **Can stay silent:** rewriting every bit outside care leaves the answer
  unchanged.
- **Anti-vacuity:** 282 of 65,536 rows.
- **Disable run:** placing edge 2's pattern in the wrong half made C answer 263
  against the accessors' 282.

The tenant sits at row bytes 48..80. Window 0 (edges 0 | 1) is in 64-byte
line 0 of the row and window 1 (edges 2 | 3) in line 1. Both windows stay
inside the tenant, so there is no RBAC widening.

| arm | ns (64K rows, 33.5 MB resident) |
|---|---|
| one strided predicate | 1.03 M |
| C: edge0.pearl ∧ edge2.epi5, two passes | 2.11 M |
| D: the same, B under A (no `_under` kernel) | 2.24 M |
| scalar reference, both fields in one row visit | 0.97 M |
| scalar reference, B read only on A's survivors | 0.79 M |
| two fields in the SAME window, two predicates | 2.47 M |
| the same two fields as ONE merged pattern | 1.17 M |

## Findings

1. **Fold-Join over aligned resident masks is already deforested.** `popcount`
   of any Boolean function of up to six resident planes runs with no slot and
   no bitmap, at about 0.4 µs per 64K rows. No new primitive is needed for the
   resident-mask ∧ resident-mask case.
2. **Mask deforestation at population scale is already the law.** The tiled
   executor writes no population-sized mask; its intermediates are one 2 KB tile
   per slot, and there were 0 allocations per execution in every arm. For i32
   predicates, the two-predicate chain (C) runs at about the per-row scalar
   loop's speed. Reading the lanes costs more than writing the tile masks, so
   fusing the predicates into the terminal has little left to save there.
3. **Gate or not, measured.** Gating B under A (D, and quack `lower`) beats
   the ungated chain only when A's dead-word fraction is high:
   - ~1.4× faster at dead-word ≥ 0.9;
   - about even near 0.5;
   - ~1.4× slower with no dead words.

   Quack picks one shape per entry point (`lower` always gates, `lower_fused`
   never does). This is the capstone's fuse-or-gate switch, and the measured
   crossover on this host is a dead-word fraction around 0.5. It is a pin for
   this host, not physics.
4. **Strided predicates pay one pass per predicate.** With a 512-byte stride,
   each predicate touches 64K separate cache lines, so cost scales with passes:
   - two predicates take 2.1× one;
   - **merging two field predicates in the same 16-byte window into one ternary
     pattern** gives the same answer at 1.17 ms instead of 2.47 ms. This is a
     lowering rewrite with no new primitive.

   The precondition is load-bearing: two patterns that disagree on a shared
   care bit have an empty conjunction (0 rows), while an unconditional OR-merge
   answered 8,273.
5. **Gating a strided predicate does not skip anything.** Without an `_under`
   kernel, D costs at least as much as C. The scalar reference that reads B only
   on A's survivors (12.6 % here) is about 20 % faster than reading both fields
   on every row. That is an upper bound on the potential, not a measurement of a
   kernel.

## Proposed rewrite contract (not implemented)

- **R1 (shipped, unchanged):** a Boolean chain over ≤ 6 resident planes feeding
  `Count`/`Any`/`Keep` folds directly.
- **R2 strided pattern merge (lowering only):**
  - **When:** `MatchFacet16Strided(L, p1, c1) ∧ MatchFacet16Strided(L, p2, c2)`,
    on the SAME lane and the same declared reading, where neither intermediate
    has another consumer.
  - **Rewrite:** to `MatchFacet16Strided(L, p1 | p2, c1 | c2)` when
    `(p1 ^ p2) & c1 & c2 == 0`, and to the empty mask otherwise. Never an
    unconditional OR.
  - **Exact for every terminal**, because it is a mask identity.
- **R3 gate selection:** gate a conjunct under the accumulator only when the
  gate's dead-word fraction is known and high; otherwise run it ungated. This
  needs the live count threaded with the aperture (seventeen rooms, room 6).
  The crossover is a measured pin, re-measured per host class.
- **Gaps named, not built (T1, `ndarray::simd`):**
  - G1 `ternary_match_strided16_to_mask_under`;
  - G2 a two-window / 32-byte strided match (one row visit for both CE64 pairs);
  - G3 tile skip in the executor (capstone point 3, now printed as a
    dead-tile fraction).

## Boundaries kept

- No CE64 instruction or NARS revision is folded; the strided predicates read
  raw bits.
- Pattern merge applies only within one lane under one declared reading. Two
  signed-register lanes bound under different `RegisterLaw`s never meet in a
  fold: `bind_signed_register` refuses `RegisterLawMismatch` (#1410) before any
  mask exists.
- A Fold-Join is an aligned-address intersection. A hop that resolves a
  different address is `Gather` / `ScatterOrU32`, not a Fold-Join.

## OPEN

- R2, R3 and G1–G3 are proposals; none is implemented.
- The D-RPF-0 threshold-as-union-of-patterns claim is not exercised by this
  probe.
- Timings come from one host class. The crossover in finding 3 must be
  re-measured before it becomes a planner constant.
