# 2026-10-07 — D-PUZZLE-0-XWORD-MASK (D-PUZZLE-0 step 3): crossword propagation as Cartesian-addressed masking

**Probe:** `cognitive-shader-driver/examples/crossword_mask_propagation_probe.rs`,
over `examples/shared/population_fold.rs` (unchanged).
**Status:** MEASURED (one machine, release, `avx2=true avx512f=false`),
TEST-PINNED (19 tests, 11 disable runs red).

## Question

Once a crossword is compiled, is propagation the Sudoku operation
(`mask → intersect → popcount → promote → expose → adjacent masks → repeat`),
with nothing crossword-shaped on the hot path? And which physical
representation of that one logic is best, on identical puzzles?

## Inventory (read-only, before any code)

| need | existing facility | verdict |
|---|---|---|
| cell (WHERE) | `Morton8x8::from_xy(col,row)`, u16 | used |
| slot position | `FacetTier{hi: slot, lo: offset}.as_u16()` | used |
| crossing | no stored relation (`morton8x8.rs`: "no stored edge"); equality of cell codes, compiled once into `cross[slot:offset]` and `occupant[cell]` lanes | used |
| word (WHAT) | DeepNSM-v2 `PaletteVocab::from_frequency_ranked`, `WordId = u16` | used |
| masks | `ndarray::simd::{mask_and_assign, popcount_batch_u64}` (already a dependency) | used |
| `MaskOp::Gather` (#1383) | foreign-key semijoin | not needed: elimination is an AND within one WordId space |
| letter | no canonical Moore symbol space: every Moore in the repo is a direction table, a grid-edge validity byte or a palette-law operand | declared probe-local `MooreSymbol8` |

No new primitive and no contract change. `deepnsm-v2` became a dev-dependency
of the driver (its `guid-v3-tail` is already a default contract feature).

## Encodings

- `MooreSymbol8` (u8): `0` unknown, `1..=26` a–z, `27` ä, `28` ö, `29` ü,
  `30` ß, `31..=255` reserved. Accented Latin letters fold to the base letter
  (`cliché` is spelled `cliche`); ä ö ü ß are letters of their own. A word with
  any other character keeps its `WordId` and is not a crossword word.
  Promoting this reading to a canonical value tenant is a contract decision,
  not taken here.
- English vocabulary: `academic_20k.csv`, lowercased, through
  `from_frequency_ranked`, exactly as `deepnsm-v2/examples/genre_shapes.rs`
  builds it: **18,555** ids (`vocab.rs` quotes 18,559 distinct surfaces; four
  collapse when lowercased), 17,548 crossword words.
- German vocabulary: no German vocabulary is committed and DeepNSM-v2 builds
  none. With `DEREKO_PATH`, the 20,000 most frequent DeReKo-2014 forms
  (lowercased, `NE` and non-alphabetic forms dropped, frequencies summed),
  19,835 crossword words. CC BY-NC 3.0; counts only here, nothing derived
  committed.
- Candidates: `P(L, i, x)` bitsets over a language's `WordId`s; a slot starts
  at `P_all(L)`. 16.4 MB (EN) / 17.7 MB (DE) of populations for L = 3..=21.
- State: CE64 bits 59..63 only, the shared GIVEN 20 / FORCED 21 / ENTAILED 22
  / CANDIDATE 4. Slot and word sit in separate content lanes.

## The four arms (same puzzles, same givens; equal fixed points asserted)

| arm | content | addressing | propagation |
|---|---|---|---|
| A token | `WordId` masks | `cross` | letter at offset `i` ANDs `P(L,j,x)` into the crossing slot |
| B Cartesian | `MooreSymbol8` board | `occupant` | symbols written; affected slots recompute from all their cells |
| C literal | strings | cell array | every unplaced slot re-scans vocabulary strings each round (steps 2/2b) |
| D hybrid | `WordId` masks + board | `occupant` | each new symbol ANDs one `P` into the cell's other slot |

## Measured

Filter, 2000 patterns (length + 1–3 revealed symbols):

| | EN | DE |
|---|---|---|
| mask AND chain + popcount | 670 ns/pattern, 9.7 ns/claim | 683 ns/pattern, 7.6 ns/claim |
| direct spell-lane scan | 41,769 ns/pattern | 39,585 ns/pattern |

Solve, median over puzzles of each arm's best of 5 warm runs (µs):

| | A token | D hybrid | B Cartesian | C literal | ANDs A / B | bytes A / B |
|---|---|---|---|---|---|---|
| EN 5×5 (30) | 2.6 | 2.6 | 4.2 | 5,229 | 10.7 / 19.1 | 23,220 / 2,382 |
| EN 7×7 (5) | 3.0 | 3.1 | 5.0 | 5,753 | 15.6 / 33.6 | 42,724 / 2,417 |
| DE 5×5 (30) | 2.6 | 2.7 | 4.5 | 6,017 | 10.3 / 17.1 | 25,060 / 2,566 |
| DE 7×7 (2) | 5.5 | 5.6 | 9.1 | 8,689 | 14.0 / 20.0 | 42,602 / 2,599 |

- A and D cost the same: routing letters through the Morton board adds no
  measurable time. B does about twice the ANDs (it recomputes) and is ~1.6×
  slower, but holds ~10× fewer bytes (one board instead of a full WordId
  mask per slot). C, the string method, is ~2,000× slower than A.
- The first run timed each arm once, cold, A first, and showed A at 16.6 µs
  against D at 6.9 µs with an equal AND count. That was cache warm-up, not
  representation; the table above is the corrected method.
- The shared fold on a 5×5 lane (~1.0M claims, EN and DE): every count equal
  three ways; filter 0.28–0.35 ns/edge, histogram ~2.0, decode oracle ~3.9 —
  the same costs as Sudoku and steps 2/2b.

Creation (budget 100,000 tried words per step, 30 s per size), NYT-rule grids:

| side | EN made / attempts | DE made / attempts |
|---|---|---|
| 5 | 30 / 30 | 30 / 30 |
| 7 | 5 / 175 | 2 / 163 |
| 9–21 | 0 (all misses are fill, none uniqueness) | 0 (same) |

Solving is not the limit at any size measured; **creation is**. With 18–20k
general vocabularies a random NYT-dense grid of side ≥ 9 is not filled within
the budget. The likely cause is vocabulary density (crosswords are built from
answer lists with many short fill words); not measured here.

Cui bono, each given removed in turn (fixed point re-run + uniqueness count):

| | givens | Redundant | ShortcutOnly | SilentlyNecessary | Necessary | out of budget |
|---|---|---|---|---|---|---|
| EN 5×5 | 118 | 32 | 37 | 0 | 49 | 0 |
| EN 7×7 | 51 | 39 | 6 | 0 | 6 | 0 |
| DE 5×5 | 98 | 21 | 29 | 0 | 48 | 0 |
| DE 7×7 | 7 | 0 | 5 | 0 | 1 | 1 |

`SilentlyNecessary` (a given propagation never used but search needs) did not
occur in the 273 ablations that finished within budget. The readout nominates; it does not prove cause: a
given necessary in one set can be replaced by another set.

## Falsifiers (disable-verified red)

Arm A ignores crossings (4 tests); letters a/b swapped (codebook, decode);
index drops offset 0 (5, incl. index-vs-scan); no promotion at popcount 1 (2);
first crossing only (every-crossing test); zero candidates not a contradiction;
language gate removed; word bits written into the edge (CE64 test); crossings
rediscovered from the cell lane (compiled-lanes test); ablation ignores
uniqueness, and ablation removes nothing (both ablation tests).

## Open

- Creation beyond 7×7 with these vocabularies.
- Whether `MooreSymbol8` (and a `MooreLearn8` companion) become canonical
  value tenants: a contract decision.
- The step-2b NYT-grid + crossword-answer mode (#1384, `94499c3`) was never run
  end to end; this probe supersedes its grid generator.
