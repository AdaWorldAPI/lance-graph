# 2026-10-07 — D-PUZZLE-0-XWORD-MASK (D-PUZZLE-0 step 3): crossword propagation as Cartesian-addressed masking

**Probe:** `cognitive-shader-driver/examples/crossword_mask_propagation_probe.rs`,
over `examples/shared/population_fold.rs` (unchanged).
**Status:** MEASURED (one machine, release, `avx2=true avx512f=false`),
TEST-PINNED (20 tests; 12 disable runs (D1–D8, D10–D13), listed below, all red).

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

Final run at `2bdb50b` (both review fixes in). One machine, release,
`avx2=true avx512f=false`.

Filter, 2000 patterns (length + 1–3 revealed symbols):

| | EN | DE |
|---|---|---|
| mask AND chain + popcount | 775 ns/pattern, 11.2 ns/claim | 543 ns/pattern, 6.0 ns/claim |
| direct spell-lane scan | 35,755 ns/pattern | 42,124 ns/pattern |

Solve, median over puzzles of each arm's best of 5 warm runs, rotated order,
one warm-up per arm; the literal arm's word list is built once per language,
outside the timing (µs):

| | A token | D hybrid | B Cartesian | C literal | ANDs A / B | bytes A / B |
|---|---|---|---|---|---|---|
| EN 5×5 (30) | 3.0 | 2.9 | 5.1 | 1,045 | 11.3 / 20.4 | 23,220 / 2,381 |
| DE 5×5 (30) | 2.3 | 2.3 | 3.7 | 986 | 7.6 / 11.4 | 25,060 / 2,568 |

- A and D cost the same: routing letters through the Morton board adds no
  measurable time. B does about twice the ANDs (it recomputes) and is ~1.6×
  slower, but holds ~10× fewer bytes (one board instead of a full WordId
  mask per slot). C, the string method, is ~350–430× slower than A.
- Two earlier timing methods were wrong and are superseded by the table
  above: timing each arm once, cold, A first (A showed 16.6 µs against D's
  6.9 µs at an equal AND count: cache warm-up), and timing the literal arm
  once including the rebuild of its word list (~5,200 µs; Codex review).
- The shared fold on a 5×5 lane (~1.0M claims, EN and DE): every count equal
  three ways; filter 0.34–0.35 ns/edge, histogram ~2.0, decode oracle ~3.6 —
  the same costs as Sudoku and steps 2/2b.

Creation (budget 100,000 tried words per step, 30 s per size), NYT-rule
grids, **no word used twice in one fill**:

| side | EN made / misses | DE made / misses |
|---|---|---|
| 5 | 30 / 1 | 30 / 4 |
| 7–21 | 0 (all misses are fill, none uniqueness) | 0 (same) |

The no-repeat rule (CodeRabbit review) changed the 7×7 result. Before it,
EN made 5 / 175 and DE 2 / 163 boards at 7×7; with it, none. Those fills
reused a word across slots, which a crossword does not allow. So with the
18–20k general vocabularies, creation stops at 5×5. Solving is not the limit
at any size measured; creation is. The likely cause is vocabulary density
(crosswords are built from answer lists with many short fill words); not
measured here.

Cui bono, each given removed in turn (fixed point re-run + uniqueness count),
5×5 only (no larger board was created):

| | givens | Redundant | ShortcutOnly | SilentlyNecessary | Necessary | out of budget |
|---|---|---|---|---|---|---|
| EN 5×5 | 116 | 28 | 40 | 0 | 48 | 0 |
| DE 5×5 | 74 | 14 | 25 | 0 | 35 | 0 |

Mean slots lost from the fixed point per removed given: EN 1.26, DE 0.20.
`SilentlyNecessary` (a given propagation never used but search needs) did not
occur in the 190 ablations. The readout nominates; it does not prove cause: a
given necessary in one set can be replaced by another set.

## Falsifiers (disable-verified red)

Arm A ignores crossings (4 tests); letters a/b swapped (codebook, decode);
index drops offset 0 (5, incl. index-vs-scan); no promotion at popcount 1 (2);
first crossing only (every-crossing test); zero candidates not a contradiction;
language gate removed; word bits written into the edge (CE64 test); crossings
rediscovered from the cell lane (compiled-lanes test); ablation ignores
uniqueness, and ablation removes nothing (both ablation tests); no-repeat
rule removed (fills-never-repeat test).

## Open

- Creation beyond 5×5 with these vocabularies (try a crossword answer list
  at runtime).
- Whether `MooreSymbol8` (and a `MooreLearn8` companion) become canonical
  value tenants: a contract decision.
- The step-2b NYT-grid + crossword-answer mode (#1384, `94499c3`) was never run
  end to end; this probe supersedes its grid generator.
