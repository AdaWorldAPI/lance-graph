# 2026-10-06 — Selector follow-up 3: observations and frontier wired; no stand-in fact left

Deliverable line: the D-GSO-6 selector (follow-up to #1365).

## MEASURED

`crates/cognitive-shader-driver/examples/recipe_selector_probe.rs`, 25
tests, 10 new disable runs red (25 in total across the follow-ups).

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
- `frontier_bounded` is read from the world: a frontier recorded by the
  `ProductInterrogation` recipe (`Quad8::fold_product`) exists for the
  current candidate space. Its count equals a brute-force count. A changed
  space (a "do") makes the record stale and reopens the fact; an empty
  frontier is still bounded.
- The wired cycle now reads all four facts from the world. For each of the
  16 states, a world built to read as that state replays to the same path
  and final world and settles.

Disable runs, each red: pending read off `speaking()`; fold a no-op;
re-folding from the start; supplied flag used instead of the derived one;
the drop flag ignored; frontier record not checked against the current
space; bounding a no-op; empty frontier read as unbounded; frontier count
inverted; `frontier_bounded` forced true.

## OPEN

- The quorum is not linked to the horizon: a conflicting verdict does not
  become an encounter or a contradiction.
- Coverage is per tension bit: new evidence on a covered bit does not reopen
  the interrogation. Receipt- or generation-aware coverage is not built.
- The frontier feeds no other recipe, and nothing derives the candidate
  space from the horizon.
