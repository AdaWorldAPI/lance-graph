# 2026-10-07 — D-CTX-7: a revision-gated writer for EpistemicState5 (candidate)

Addresses `ISS-NO-EVIDENCE-WRITER-FOR-EPISTEMIC-STATE` with one declared,
versioned candidate. Does not close it: the production decision stays open.

## MEASURED

`cognitive-shader-driver/examples/revision_epistemic_writer_probe.rs`:

- One `MailboxSoA` CE64 row tracks "pixel P is a material boundary" on a
  16 × 16 palette tile. A source is one Moore direction of P, read from the
  resident tile only. Each cycle: start-of-cycle copy through
  `MailboxSoaView` → encounters → `GadamerRevision` → declared transition →
  `write_row` in cycle k → read in k+1.
- Transition V1: 3 → 7 on ≥ 2 roots the revision admitted with
  `IncreaseEligible` from `Differs` readings; 7 → 3 on a
  `ContradictionPreserved` verdict on the claim (a complete Moore check that
  reads only `Same`). Bits 59..63 are written as one joint code (the #1370
  reading) through `with_spare` + `with_truth`; every write lands on a code
  `EPI_LAW` declares.
- Trajectory (render only, one source, two sources, tile rewritten + complete
  check): codes `3, 3, 3, 7, 3`; eligibility `OBS, OBS, OBS, OBS|STRATIFY, OBS`.
- 360 rendered witnesses (D-CTX-4) proposing the claim never move the code in
  either direction: they carry no independent root, so the revision never
  admits them. One direction repeated 50 times is one root. A `Same` reading
  is not support. A complete check of a boundary pixel (8 roots, 3 `Differs`)
  promotes alone. V2 (3 roots) refuses what V1 accepts. Only bits 59..63 move.
  Restart from `to_le_bytes` continues identically. 0 allocations per settle.
- 11 tests, 7 disable runs red. Two disables were green on the first draft and
  exposed vacuous gates: the render was kept out by the probe's own
  `supports = false` flag instead of by the revision (the flag now says the
  render supports the claim, so only the missing root keeps it out), and no
  fixture admitted more than one root per encounter (the complete-check test
  now separates "roots" from "encounters").

## OPEN

- Whether this transition, or any, belongs in production.
- Cross-cycle corroboration: the horizon dies with its cycle, so one source in
  cycle k and another in k+1 never combine (`sources_split_across_cycles_do_not_combine`).
  Carrying the source set needs state the register does not have.
- Demotion reads a `Suspend` verdict (`ContradictionPreserved`); promotion
  reads `IncreaseEligible`. Whether demotion should require more than one
  complete check is a policy question the probe does not answer.
