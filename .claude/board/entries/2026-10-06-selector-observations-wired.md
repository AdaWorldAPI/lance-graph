# 2026-10-06 — Selector follow-up 3: observations_pending wired to the quorum fold

Deliverable line: the D-GSO-6 selector (follow-up to #1365).

## MEASURED

`crates/cognitive-shader-driver/examples/recipe_selector_probe.rs`, 21
tests, 5 new disable runs red (20 in total across the follow-ups).

- `new_encounter` is renamed `revision_pending`: it holds while revising
  would change the horizon, which is no longer the same as a new encounter.
- `observations_pending` is read from the world: more verdicts have arrived
  than the `Quorum` has counted (corroborating + silent + conflicting). The
  `ObserveFold` recipe folds only the uncounted verdicts with
  `Quorum::observe`; the quorum is its only output.
- Three verdicts fold to `Quorum(1, 1, 1)` in one step and the fact clears.
  A late verdict reopens it and is folded once (`(1,0,0)` → `(1,0,1)`).
- A silent verdict settles: it is counted, though `speaking()` stays 0.
- Dropping the fold's write never rests.
- Arrival is the "do": the test writes verdicts; the selector never does.

Disable runs, each red: pending read off `speaking()`; fold a no-op;
re-folding from the start; supplied flag used instead of the derived one;
the drop flag ignored.

## OPEN

- The quorum is not linked to the horizon: a conflicting verdict does not
  become an encounter or a contradiction.
- Coverage is per tension bit: new evidence on a covered bit does not reopen
  the interrogation. Receipt- or generation-aware coverage is not built.
- `frontier_bounded` is the last stand-in fact.
