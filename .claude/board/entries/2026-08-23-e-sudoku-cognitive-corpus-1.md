## 2026-08-23 — E-SUDOKU-COGNITIVE-CORPUS-1 — the first real corpus through the dispatch bridge: a real puzzle, warranted end-to-end, and TCP/TCF/CUR still collide under real data

**Status:** FINDING (measured — `PROBE-SUDOKU-COGNITIVE-CORPUS-1`, run
against a real, independently-solved puzzle). **Confidence:** High;
reproducible from the commit.

**The first real corpus through `PROBE-RECIPE-DISPATCH-BRIDGE-1` (#996).**
`crates/lance-graph-ogar/examples/sudoku_cognitive_corpus_probe.rs` runs the
Sudoku Wikipedia article's example puzzle through naked-single constraint
propagation (ordinary, correct Rust — not claimed to be "solved by the
recipe kernels"), with every elimination event ALSO dispatched through a
real `kernel(id)` — `TCP`(5)/`TCF`(20)/`CUR`(26), the exact three recipes
`PROBE-RECIPE-EXECUTION-1` (#995) found collapsing into one coarse
signature under a synthetic battery — chosen per event by which real peer
group (row/column/box) has the fewest remaining unsolved cells.

**The load-bearing check: PASS.** This puzzle solves completely by naked
singles alone — 51 assignments, 81/81 cells — and every single one is
warranted: verified against an INDEPENDENT full backtracking solve
(`backtracking_solve`, sharing no code with the propagation loop). 282
elimination events were dispatched through real kernels alongside the
solving, never driving it.

**The re-test: TCP/TCF/CUR still collide, even under real puzzle-derived
data.** 107/88/87 dispatches respectively, each producing only 2 distinct
(fired, Δconfidence-sign) signatures across the entire real corpus —
identical to the synthetic-battery result. This distinguishes the "genuine
no-op / coarse-signature-too-blunt" collision class from the "different
labels, actually-different-behaviour" class the session's own analysis
predicted might exist: real, varied, puzzle-derived candidate sets (2-9
candidates per cell, real entropy variation) did NOT separate these three —
the collision is a property of the recipes' effect at THIS signature
granularity, not an artifact of the synthetic battery being unrepresentative.

**The generalization:**

> A synthetic test battery and a real corpus answering the SAME collision
> question is itself informative regardless of the answer. Had real data
> separated TCP/TCF/CUR, that would have indicted the synthetic battery as
> too narrow. It did not — which instead strengthens the claim that these
> three recipes' effects, at this signature granularity, are genuinely
> close, not merely under-tested.

**Scope, honestly held:** naked-single propagation only (no hidden singles,
pairs, X-wing, …). The recipe dispatch is a measured side channel over real
state, not a claim that the kernels perform or could perform Sudoku-specific
reasoning — their pruning logic is generic and confidence-shaped, unrelated
to digit legality.

Cross-ref: `E-RECIPE-DISPATCH-BRIDGE-1` (the bridge this corpus runs
through), `E-RECIPE-EXECUTION-SEPARABILITY-1` (the collision this re-tests).
