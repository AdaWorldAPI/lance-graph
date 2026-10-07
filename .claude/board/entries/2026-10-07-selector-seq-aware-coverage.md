# 2026-10-07 — Selector follow-up 4: sequence-aware coverage

Deliverable line: the D-GSO-6 selector (follow-up to #1368). Uses the D-GSO-8
sequence rule (#1366).

## MEASURED

`crates/cognitive-shader-driver/examples/recipe_selector_probe.rs`, 29
tests, 7 new disable runs red.

- `present(encounter, seq)` refuses any `seq` not greater than the last
  accepted one and leaves the world unchanged. An accepted encounter stamps
  each of its contradiction bits with `seq`.
- `MooreInterrogation` records, per bit, the evidence `seq` it covered.
  `local_disagreement` holds for a tension bit that was never covered, or
  whose newest evidence is newer than its coverage.
- Presenting the same contradiction again at a newer `seq`, with nothing
  else in the horizon changing, reopens the interrogation once
  (`revision_pending` stays false) and records the new `seq`. Re-presenting
  that `seq` is refused.
- Evidence on one bit does not reopen another bit.
- One encounter slot: once an encounter is accepted, a newer one is refused
  (`Refused::Unprocessed`, world unchanged) until the cycle has revised with
  it (`revision_pending` false). Codex review on #1380: replacing it would
  advance `last_seq` past an encounter that never revised the horizon.
- An encounter written without `present` carries `seq` 0 and never reopens a
  covered bit, so the earlier tests are unchanged.

Disable runs, each red: reuse of a `seq` allowed; the stale-coverage clause
removed; coverage `seq` not recorded; evidence not stamped; `>=` instead of
`>`; the unprocessed-encounter refusal removed; the encounter written before the refusal check. The last was green until
the refusal test presented a different encounter than the current one.

## OPEN

- Only encounters carry a `seq`. Verdict arrivals and space changes do not,
  so they are not on one durable event chronology yet.
- `evidence_seq` and `covered_seq` are probe-local, like `interrogated`.
