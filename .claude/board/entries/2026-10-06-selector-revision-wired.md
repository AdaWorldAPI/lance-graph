# 2026-10-06 — D-GSO-6 follow-up: the selector's revision fact is wired

## MEASURED

`crates/cognitive-shader-driver/examples/recipe_selector_probe.rs`, now 14
tests (9 new), 10 disable runs red. Builds on D-GSO-6 (#1358) and the Round 6
cycle (#1359).

- `new_encounter` is no longer supplied. It is read from the world: revising
  would still change a mask of the horizon (projected claims, a root, an
  inherited root, or a contradiction not yet in the tension). On all 65 536
  worlds of a 2-bit universe it equals whether `revise` changes those masks.
- A first version tested new roots only and dropped rootless revisions
  (codex review): a contradiction that withdraws a claim without a new root
  (`ContradictionPreserved`) was never applied. Now it is applied once.
- The `Revision` recipe runs `GadamerRevision::revise`; only
  `delta.resulting` reaches the next state.
- After one real revision the cycle rests (roots `0b101 → 0b111`). With the
  write dropped it selects `Revision` every step and never rests.
- An echo (same projection, roots held, no contradiction) selects nothing. A held claim gaining its first
  root (`IndependentConfirmation`) selects `Revision` once.
- Every (policy, stand-in state) start replays to the same path and the
  same final horizon.

Disable runs, each red: derived fact forced true; derived fact ignored in
favour of the supplied flag; revise result discarded; roots compared
against projected claims instead of roots; the bypass flag ignored; each of
the four mask clauses dropped; the roots-only predicate restored. The
roots-vs-claims disable was green until the held-claim case was added.

## FINDING

- **The selector's rest after revision now comes from revision's output.**
  It is no longer a stand-in.
- **`local_disagreement` cannot be read off the horizon.**
  `unresolved_tension` survives every revision, so a selector reading it
  would interrogate forever.

## OPEN

- `observations_pending`, `frontier_bounded` and `local_disagreement` are
  still stand-ins. The last needs a source that clears: tension new since
  the last interrogation, not tension held.
- The policy version still covers selection only (§13 digest).
