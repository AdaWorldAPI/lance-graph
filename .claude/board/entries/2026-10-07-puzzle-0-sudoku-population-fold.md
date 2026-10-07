# 2026-10-07 — D-PUZZLE-0 step 1: Sudoku at scale on the population algebra

**Probe:** `cognitive-shader-driver/examples/sudoku_population_fold_probe.rs`.
**Status:** MEASURED (this machine), TEST-PINNED (7 tests, 3 disable runs red).

The question D-PUZZLE-0 opens: do Sudoku, crosswords and chess run on the same
`EpistemicState5` population algebra under their own domain laws? Step 1 is
Sudoku at scale, the hot path the other two would share:

```
one edge          raw5 coordinate (bits 59..63)
one question      facts_population(required) : u32
many edges        declaration checked once per lane → population filter → fold
```

## Corpus and reading

1,000,066 claim edges ("cell c holds digit d", eliminated claims are not
edges) from 8,790 symmetry copies of the Sudoku Wikipedia puzzle (digit
relabel, band/stack and within-band/stack permutations, transpose). Each copy
is snapshotted after a random number of naked-single placements. Each copy's
full solve is checked against the transformed base solution, independently of
propagation order.

The reading is a probe-local declaration under class `0x0906`, not canon:

| claim | coordinate | raw5 |
|---|---|---|
| given | `Direct × Causes` | 20 |
| placed by a naked single | `IndirectKnown × Causes` | 21 |
| true digit, not yet derived | `IndirectUnknown × Causes` | 22 |
| other surviving candidate | `Direct × Associated` | 4 |

Row three is the state the Cartesian layout kept on purpose: the value is
entailed (unique solution, checked) but its derivation has not been produced.
The 5-bit state has no "refuted" coordinate, so eliminations are absences, not
edges. `Unknown × Causes` (23) is never emitted here. As populations, the
three placement states miss exactly that code, and the lane holds none of it.

## Measured

Counts (equal by the population filter, the 32-bin histogram, the per-edge
decode through the declaration, and the solver's own counters):

| question | population | edges |
|---|---|---|
| every placed digit | `CAUSES` = 0x00F00000 | 711,990 |
| givens | `DIRECT \| CAUSES` | 263,700 |
| derived, chain known | `IND_KNOWN \| CAUSES` | 227,189 |
| entailed, chain not derived | `IND_UNKNOWN \| CAUSES` | 221,101 |
| live candidates | `(DIRECT \| ASSOCIATED) & !RELATED` | 288,076 |

Group fold: every copy holds exactly 81 placed-digit claims and 30 givens.

Timing, release, `avx2=true avx512f=false`, 4-core shared Xeon @ 2.80 GHz,
median of 7 per run, three runs:

| path | ns / edge |
|---|---|
| one population, filter fold | 0.35 – 0.39 |
| 32-bin histogram (answers every question) | 0.93 – 1.07 |
| per-edge decode + asserts (oracle) | 3.8 – 4.0 |

The filter is about 10× cheaper per question than decoding each edge. The
histogram costs about 2.7 filters and then answers any population in 32
additions. With three or more questions per lane it is the cheaper fold. These
are one machine's numbers; no SIMD is claimed (plain loops; whether the
compiler vectorized them was not checked).

## Falsifiers

- Three-way count agreement for every question, and histogram == filter for
  all 2,048 requirements over the declared facts.
- The four states partition the lane (disjoint populations whose union covers
  every edge).
- Group fold: 81 placed and 30 givens per copy; depths actually vary.
- An undeclared class is refused before any edge is read.
- Disable runs: the fold reading bits 58..62 (3 tests red); a gate admitting
  any class (1 red); "live candidates" without `!RELATED` (3 red).

## Next (not started)

- Crossword in lance-graph: claims "slot s holds word w" / "cell holds letter",
  the crossing constraint as the domain law, the same lane and folds.
- Chess in `stockfish-rs` (GPL-3.0; shakmaty supplies legality and must not
  enter Apache lance-graph). It consumes `lance-graph-contract`, which it
  already dev-depends on.
- What would falsify "same algebra": a domain whose question cannot be
  written as a fact conjunction, or whose fold needs per-domain code in the
  shared path.
