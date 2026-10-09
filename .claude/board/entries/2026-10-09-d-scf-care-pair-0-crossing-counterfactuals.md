# 2026-10-09 — D-SCF-CARE-PAIR-0: crossing counterfactuals don't pay; one folded chain relation does

**Status:** MEASURED, TEST-PINNED. Test-only probe; no library code changed.
**Probe:** `crates/cognitive-shader-driver/examples/crossword_crossing_care_probe.rs`
(`test = true`, 11 tests, picked up by the existing shader-driver CI step).
Runs over the unchanged `shared/crossword_core.rs`. Population passes are
`ndarray::simd::{mask_ternlog_popcount, mask_andnot_assign}` or the core's
`State::narrow`. No new primitive.

```text
cargo run --release -p cognitive-shader-driver --example crossword_crossing_care_probe
```

## Question

Can one certified local constraint relation make a later derivation
unnecessary? Asked two ways:
- **inside one puzzle:** counterfactuals at crossings, run before search;
- **across instances:** existential elimination of internal slots into a
  boundary relation that is reused on held-out puzzles.

## TRUTH — what existed (read before writing code)

| capability | where | status |
|---|---|---|
| `P(L,i,x)` populations, compiled crossings, fixed point | `shared/crossword_core.rs` (#1387) | used unchanged |
| cui-bono ablation of givens | `crossword_mask_propagation_probe.rs::ablate` | exists; removes givens, not letters |
| popcount / dom-wdeg slot focus | `crossword_attention_probe.rs` (#1398) | baseline here is popcount |
| fused AND-popcount without an intermediate | `ndarray::simd::mask_ternlog_popcount` (the D-RPF-9 fold-join shape) | gives `|D ∧ P(i,a) ∧ P(j,b)|` directly |
| Boolean `mxm` (existential elimination) | `lance-graph/.../blasgraph` `HdrSemiring::Boolean` | exists over 16 Kbit `BitVec` entries; far too heavy for a 31×31 letter relation, not used |
| `IndirectKnown` | `lance-graph-planner/src/pearl.rs::hydrate` | a topology stamp when `A→B` and `B→Y` bindings exist; it is not a proof and is not written here |

**Gap found in the core:** `propagate_token` is singleton-triggered. Only a
slot whose mask has one candidate left sends letters to its crossings. Letter
support of a slot with many candidates is never propagated. Every arm below
is measured against that gap.

## Arms (same puzzles, givens, popcount slot policy and op budget; identical answers asserted)

| arm | root | per node |
|---|---|---|
| Baseline | fixed point | fixed point |
| Support | + letter support at every crossing (zero population = certified exclusion) | the same, recomputed |
| Single | + one-cell counterfactual: assume a letter, settle, drop the copy | fixed point |
| Pair | Single + two-cell counterfactuals inside one slot; refutations kept as pairwise nogoods | + nogood propagation |
| PairZ | Pair, with zero-population pairs also stored as nogoods | as Pair |

Created NYT puzzles, 10 per side; prove = all givens, count fills to 2;
open = half the givens, count to 50. Answers are identical across arms. All
arms keep the planted solution and violate none of their nogoods (the
independent soundness oracle).

| side / work | arm | nodes | ops | ms |
|---|---|---|---|---|
| 5 prove | Baseline | 5,658 | 11,095 | **6.5** |
| | Support | 87 | 122,102 | 14.0 |
| | Single | 2,149 | 32,673 | 10.1 |
| | Pair | 1,421 | 132,198 | 38.5 |
| | PairZ | 239 | 163,590 | 44.3 |
| 5 open | Baseline | 686,948 | 1.38 M | **776** |
| | Support | 11,881 | 13.96 M | 1,619 |
| | Single | 721,784 | 1.45 M | 952 |
| | Pair | 720,520 | 2.18 M | 1,107 |
| | PairZ | 96,961 | 26.03 M | 8,917 |
| 7 prove | Baseline | 1,298 | 4,273 | **2.8** |
| | Support | 35 | 131,366 | 14.7 |
| | Single | 760 | 27,611 | 14.6 |
| | Pair | 496 | 159,618 | 96.0 |
| 7 open | Baseline | 31,073 | 110,962 | **70.7** |
| | Support | 890 | 2.44 M | 311 |
| | Single | 22,918 | 157,582 | 92.9 |
| | Pair | 22,513 | 682,754 | 416 |

(One op is one pass over a 290-word population. Wall time is one machine,
release, three runs; every node and exclusion count is identical between runs
and timings agree within about 15 %. The op column was corrected after review:
`refuted` had charged `settle`'s propagation twice. Nodes, exclusions, answers
and every conclusion are unchanged by the fix.)

**Where Single's exclusions come from** (5 prove / 5 open / 7 prove / 7 open):
- zero population, no counterfactual needed: 882 / 1,030 / 902 / 2,435;
- counterfactual refutation: 124 / 58 / 124 / 279;
- counterfactual worlds needed: 2,268 / 6,341 / 2,472 / 8,387.

Pair adds 782 / 1,063 / 273 / 749 pairwise nogoods. Only 34 / 40 / 46 / 17
unary exclusions follow from them by completeness.

## DARE — the reusable relation actually demonstrated

The core lets a word have internal letter variables only off NYT rules. In
**every created NYT puzzle, unchecked cells = 0**: every letter is shared by
an across and a down slot. So folding a word out leaves all its letters on
the boundary. The separator width is the word length, and the boundary
relation is the word list itself. Nothing is gained there.

On chain fragments, where the internal slots are crossed only by their two
neighbours, folding works:
- **Relation per slot.** `n(a,b) = |D ∧ P(L,i,a) ∧ P(L,j,b)|` is exact, with
  multiplicities. The mask arm equals the id-scan oracle on full and
  narrowed contexts.
- **Composition.** `N(x,y) = Σ_z n1(x,z) n2(z,y)` folds A(3) – B1(4: 0→3) –
  B2(5: 0→4) – C(6) to one 31×31 table with 557 supported pairs.
- **Key.** `ChainKey` is: language, dictionary fingerprint, links up to
  reversal (with a transpose flag), and no-repeat vacuity. Context is checked
  by the walk: every internal slot has exactly two crossings.
- **Held-out reuse.**
  - H1: a different board, a 7-letter A. Reused.
  - H2: the same chain walked from the other end. Reused through the
    transpose.
  - **1,000 / 1,000 queries per instance gave identical counts** against the
    core's own `count_fills` search.
- **Refusals.** Each refused key field has an isolating fixture, and forced
  reuse past the refusal is wrong:
  - H3: a third crossing on B2. Wrong counts.
  - H4: two internal slots of length 4, no-repeat binds. Overcount.
  - H5: T's own links with a 5-letter A. Only no-repeat separates it.
    Overcount.

| instance | search per query | lookup | validation (fingerprint / key only) | break-even N | SFCR(1000) |
|---|---|---|---|---|---|
| H1 | 63.6 µs | 0.018 µs | 802 µs / 0.6 µs | 36 | 27.95 |
| H2 | 162.0 µs | 0.016 µs | 818 µs / 0.7 µs | 15 | 70.75 |

`SFCR(N) = N · search / (construct 1.46 ms + validate + N · lookup)`.
Construction is the mask arm over `P_all`. Almost all of the validation cost
is the dictionary fingerprint. With a versioned dictionary generation the
key check is under 1 µs, and break-even falls to about 24 (H1) and 9 (H2).

**The Cartesian adversary.**
- `{(a,0),(b,1)}`: the product of the marginals gives 4 pairs, the relation
  has 2.
- Real slots:
  - `P_all(4)` on (0,3): 260 spurious pairs out of 600;
  - `P_all(5)` on (0,4): 249 out of 575;
  - the two-link fold: 18 spurious pairs. They land outside any sampled
    query, so a sampling check alone would have passed.
- The one-link fixture: every spurious pair the product admits is refuted by
  the search.

End words are fixed by the query, not folded, so `lookup` refuses one word
on both ends (equal ids are possible only when the ends share a length). This
is pinned on H6, which has two 6-letter ends. Found in review.

**Epistemics.** The relation's correctness rests on construction plus the
oracle equality. Read as CE64, it is a certified domain consequence (a proof
of each count), not `IndirectKnown × Supports`. No CE64 write was made.

## STOP

- **Single-crossing counterfactuals before search.** Slower on all four
  workloads: 0.19× to 0.81× of baseline wall time. About 90 % of what they
  remove is plain letter support. The counterfactual worlds add 0.9–5.5 %
  hits per world. On the side-5 open workload the root pruning gave MORE
  nodes than the baseline (721,784 vs 686,948). That is likely the cap-50
  enumeration order, not investigated.
- **Two-crossing pairwise nogoods.** 0.03× to 0.70× of baseline wall time. A
  nogood mostly refutes a pair that later propagation also refutes; node
  savings are 0–34 %.
- **Storing zero-population pairs as nogoods (PairZ).** A materialised copy of
  what letter support recomputes from the resident masks. It cuts nodes
  (5,658 → 239) but costs 5–11× the wall time of Baseline. **Duplicate.**
- **The fused mask arm for building a static pair relation.** 780 passes,
  110–170 µs, against a 25 µs walk of the candidate ids, which equals it.
  The fused form wins only where the candidates must not be enumerated.
- **Folding NYT-valid grids.** No internal letter variables exist (measured
  0), so there is nothing to eliminate.

## Gates

12 tests. Falsifiers in the test file:
- mask relation = id scan;
- marginal product invents pairs (synthetic, real, two-link);
- composition = core search;
- reuse accepts H1/H2 and refuses H3/H4/H5;
- forced reuse is wrong on H3, H4, H5;
- equal end words are not a completion (H6);
- every arm is sound and agrees;
- refuted worlds leave the state unchanged;
- single counterfactuals fire;
- splitting a pairwise nogood into unary exclusions, or deriving a unary
  exclusion from one failing partner, removes a planted letter (with the
  honest arm sound on the same puzzles).

Disable runs, anchors asserted, each red on its named test:
1. fold through marginals;
2. no reversal;
3. no-repeat flag ignored (twice);
4. third crossing allowed;
5. zero-population exclusion over live letters;
6. nogood bans the wrong letter;
7. unary exclusion without completeness;
8. the equal-end-word guard removed.

D3 and D4 were vacuous at first: H3/H4 were refused for other reasons, and
`ends()` picked the extra slot. H5 and a 7-letter H3 isolate the fields.

Restore needs no separate check. `refuted` takes `&State`, so a
counterfactual world cannot write back; the test pins it.

## OPEN

- Letter support recomputed in full per node cuts nodes 35–65× and still
  loses 2–5× on wall time. An incremental version (re-check only cells whose
  slot changed) is unmeasured. It is the only arm here whose search-side
  effect is large.
- Chain reuse inside a solver has no consumer. NYT grids have no chains, and
  where chains exist, the relation answers completion counts and
  existence. It does not prune a running search.
- One dictionary (COCA academic, English), sides 5 and 7, one machine.
