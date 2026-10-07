# 2026-10-07 — D-PUZZLE-0 step 2: crosswords on the same algebra as Sudoku

**Probes:** `cognitive-shader-driver/examples/crossword_population_fold_probe.rs`,
`sudoku_population_fold_probe.rs`, both over
`examples/shared/population_fold.rs`.
**Status:** MEASURED (this machine), TEST-PINNED (fence + 6 domain tests, 3
disable runs red).

## What "same algebra" means here, and how it is checked

The domain-free half moved out of the Sudoku probe into
`shared/population_fold.rs`: the propagation reading (given / forced /
entailed-not-forced / candidate), the five questions as `facts_population`
masks, the filter / histogram / decode folds, the group fold, the declaration
gate. Each domain file supplies only its claims, its law and its counters.

- **Fence:** a test fails if `sudoku`, `crossword`, `chess`, `digit`, `slot`,
  `cell`, `letter`, `grid`, `board` or `move` appears in the shared half.
  Disable run: one planted word → red.
- **Refactor checked by output:** the Sudoku probe's five counts are
  identical before and after the move (1,000,066 edges).

## The crossword domain

A 5×5 grid, two black squares, ten slots, every white square in two slots.
Fills are random over a 3-letter alphabet; the dictionary is the solution
words plus 10 distractors per length; two slots are given. An instance is kept
only with exactly one solution (backtracking). The law is the forced single: a
slot whose consistent dictionary words narrow to one is placed.

## Measured (1,000,052 edges, 22,455 instances)

| question | edges |
|---|---|
| every asserted claim | 224,550 |
| given | 44,910 |
| forced, chain known | 13,745 |
| entailed, not yet forced | 165,895 |
| live candidates | 775,502 |

Every count is equal three ways and to the instances' own counters. Every
instance holds exactly 10 asserted claims. The instance build takes 7.3 s
(rejection sampling with backtracking).

Timing, release, `avx2=true avx512f=false`, same 4-core shared Xeon, median of
7, three runs:

| path | crossword ns/edge | Sudoku ns/edge |
|---|---|---|
| one population, filter | 0.30 – 0.43 | 0.35 – 0.39 |
| 32-bin histogram | 1.67 – 1.75 | 0.93 – 1.07 |
| per-edge decode (oracle) | 3.9 – 4.1 | 3.8 – 4.0 |

The filter and the oracle cost the same in both domains: they read five bits
and do not care what the claim is about. The histogram costs more on the
crossword lane, where 77 % of edges share one code. Repeated increments of
one bin are the likely cause; this is a hypothesis, not measured.

## What differs, and it is the law, not the algebra

Forcing is rare in crosswords: 13,745 of 177,640 non-given slots, against
Sudoku's naked singles, which solve the base puzzle completely. A slot rarely
narrows to one word without stronger propagation (arc consistency over
crossings), so most true words stay `IndirectUnknown × Causes`. That is a
statement about the law. The questions and folds did not change.

## Next

Chess in `stockfish-rs` (GPL; shakmaty supplies legality), using the same
shared half. The open point is how it gets there. The module is an
example-local file in lance-graph and is not part of `lance-graph-contract`.
Copying it into stockfish-rs would make a second copy (a fence violation in
spirit); promoting it into the contract is a contract change. The decision is
open.
