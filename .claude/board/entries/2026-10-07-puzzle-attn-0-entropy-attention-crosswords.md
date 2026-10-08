# 2026-10-07 — D-PUZZLE-ATTN-0: on crosswords, uniform entropy focuses search; frequency-weighted Shannon misdirects it

**Status:** TEST-PINNED (`crates/cognitive-shader-driver/examples/crossword_attention_probe.rs`, `test = true`; 8 tests, 7 run by the shader-driver `cargo test` step plus `dump_small` ignored; the same binary also runs the 2 shared `population_fold` fence tests, which is where the earlier "9" came from). MEASURED (`cargo run --release -p cognitive-shader-driver --example crossword_attention_probe`; 20 created puzzles and 20 empty NYT grids, sides 5 and 7; 200,000 nodes per run).

## DECISION

- **SCOPE:** crossword solving is a fixed point (#1387), so deduction needs no attention. This probe changes only the search's choice of the next slot once deduction stalls. Everything else is held fixed:
  - the puzzles;
  - the propagation, moved unchanged into `examples/shared/crossword_core.rs` (#1387's 20 tests pass on it);
  - the value order, most frequent word first.
- **Policies:**
  - fixed order;
  - seeded random;
  - widest first (negative control);
  - **popcount**: most-constrained first, i.e. entropy under a uniform prior, `log2(count)`;
  - **Shannon**: `H = -Σ p log2 p` with `p ∝` COCA-All frequency;
  - **dom/wdeg** (Boussemart et al. 2004): count divided by accumulated contradictions on the slot that ran dry, so earlier failures steer later choices.
- **Contradiction:** a slot with no candidate has no entropy under either reading and is never settled (pinned).

## Measured

| policy | prove: solved | prove: nodes | open: solved | open: nodes / to first | fill: solved |
|---|---|---|---|---|---|
| fixed | 16/20 | 852,535 | 7/20 | 2,858,517 / 2,488,660 | 5/20 |
| random | 15/20 | 1,049,801 | 5/20 | 3,151,741 / 2,572,628 | 0/20 |
| widest | 16/20 | 916,437 | 5/20 | 3,186,550 / 2,653,695 | 10/20 |
| **popcount** | **20/20** | **58,729** | **20/20** | 570,450 / 91,930 | 10/20 |
| Shannon | 19/20 | 348,578 | 16/20 | 1,156,793 / 550,562 | 10/20 |
| **dom/wdeg** | **20/20** | 53,892 | **20/20** | 539,520 / **40,938** | 10/20 |

(prove = count fills to 2 on created puzzles; open = half the givens, count to 50; fill = an empty grid.)

- **Entropy as focus of attention works, read as uniform entropy.** Popcount needs 14.5× fewer nodes than fixed order to prove uniqueness, and 5× fewer on the open puzzles, where fixed order finishes only 7 of 20.
- **Frequency-weighted Shannon is worse than popcount**: 5.9× the nodes on prove; on open, 2× the nodes and 4 puzzles unfinished. It picks a different slot on 36–50% of decisions.
  - The frequency prior measures how *predictable* a slot is; a search pays for how many candidates it must *try*.
  - So for attention in search, `log2(count)` is the right reading, not a proxy for something better.
  - The `ms` column overstates popcount: its runs also compute Shannon on every decision to count disagreements.
- **Earlier contradictions do direct later choices.** dom/wdeg uses about the same total nodes as popcount but reaches the first fill sooner: 33,017 vs 41,029 on prove, 40,938 vs 91,930 on open.
- **The contradiction rate does not improve within a search.** Per tenth of the search it stays flat at 0.73–0.85 for both popcount and dom/wdeg. The gain shows up as an earlier solution, not a lower failure rate.
- **Empty grids:** 10 of 20 stay unfilled within 200,000 nodes under every good policy. This is likely the vocabulary rather than attention; not measured.

## Gates

7 disable runs, each red on its named falsifier:
- popcount picks the first slot
- Shannon reads popcount
- an empty mask has H = 0
- a flat prior
- no failure weights
- no propagation
- a wall-clock seed

One falsifier was vacuous on first write: the dom/wdeg "differs from popcount" check compared whole stats, which differ anyway through popcount-only counters. It now compares the search path, and its disable goes red.

## OPEN

- Shannon may belong in *value* ordering (which word to try first) rather than slot choice. Untested: the value order here is fixed by frequency.
- Coverage is narrow: one frequency prior (COCA academic), one vocabulary, sides 5 and 7 only. Side 9 takes minutes per created puzzle.
- No CE64 register in this loop. The slot state is the mask (the evidence); entropy is derived from it.
