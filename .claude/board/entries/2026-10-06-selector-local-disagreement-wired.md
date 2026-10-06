# 2026-10-06 — D-GSO-6 follow-up 2: local disagreement wired as uncovered tension

## MEASURED

`crates/cognitive-shader-driver/examples/recipe_selector_probe.rs`, 17
tests, 15 disable runs red. Builds on #1362. Also carries the rootless-
revision fix from codex review of #1362, which merged before it was pushed:
`new_encounter` holds while revising would still change a horizon mask
(equal to "revise changes those masks" on all 65 536 worlds of a 2-bit
universe).

- `local_disagreement` is read from the world as
  `unresolved_tension \ interrogated`. `interrogated` is the coverage the
  `MooreInterrogation` recipe writes (by union); nothing else writes it.
- Interrogation clears the fact and leaves the tension unchanged.
- A revision that adds a contradiction reopens the fact: from a settled
  state the fusion world runs `[Revision, MooreInterrogation]` and rests.
  A contradiction already covered does not reopen it; one on a new bit does.
- Dropping the interrogation's write never rests, as dropping revision's
  write never rests.
- Every wired cycle rests within 5 steps and replays to the same path and
  final world.

Disable runs, each red: raw tension read without coverage; interrogation a
no-op; supplied flag used instead of the derived one; interrogation clearing
the tension; the drop flag ignored.

## FINDING

- **With both facts derived, a cycle is no longer one step per condition.**
  Revision can reopen local disagreement, so V1 runs `Revision` then
  `MooreInterrogation` where the stand-in transitions ran them once each in
  §11 order.

## OPEN

- `interrogated` is probe-local. `InterpretiveHorizon` has no coverage
  field; where coverage lives durably is undecided.
- The Moore recipe's fold (Palette hop over `Register128`, D-GSO-5 R2) is not
  linked to claim bits; only its coverage is wired. What an interrogation
  concludes about a tension bit is not modelled.
- `observations_pending` and `frontier_bounded` remain stand-ins.
