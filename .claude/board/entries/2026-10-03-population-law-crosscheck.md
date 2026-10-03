# Population-law cross-check: frontend witness (#1311) against a real-data fold experiment (2026-10-03)

**Status:** ANALYSIS + VERIFIED-IN-CODE; no code changed. Plan:
`.claude/plans/population-law-crosscheck-v1.md`. The experiment's headline counts were
re-derived from `crates/deepnsm/word_frequency/academic_20k.csv` (20,845 / 20,842 /
18,559; three n=2 keys).

- **Law that survives both:** execute over the population that already carries the
  multiplicity; reach other data only by functional reads; switch anchors instead of
  building target populations; a phase boundary is real only when a later pass needs,
  per row, a fold over several rows of an earlier pass.
- **#1311 revised:** a second fan-out is not itself a barrier. Re-anchoring onto a
  population that references the current anchor costs read depth (G1). The barrier is a
  carried fan-in aggregate (A → R1 → R2 over edge tables).
- **I→S:** the temporary key column is an API artifact (the key is a function of the
  result's coordinate; sinks keep no producing-key metadata; no coordinate-derived key
  form exists). The presence fold's dependency on a complete I is real, unless the
  source has an ordered resident projection.
- **Result as operand:** mask results already feed a later phase over the same rows
  (Quack `GroupPlan`). `i64`/record sinks are finalized in the K-space in host code by
  design (Quack HAVING). Missing: reading a produced K-space result per row from a
  later N-row pass. No `LaneRef::I64`; `PowerSums` exists only in the ndarray facade.
- **#1311 gaps:** G1 confirmed general; G2 split (re-anchor vs phase boundary); G3 and
  G5 simple missing operators; G4 mis-specified witness (by source, a missing
  `masked_sum_i32_via`).
- **⊘ Corrected 2026-10-03 (#1313):** resident data + reference register → fold →
  zero-copy projection. A pivot rotates the register; only a write materializes.
  Pair → spelling folds directly from O (no phase boundary). Grouped output is a
  projection, not a population. Equal row count does not align rows. The falsifier
  above (D-PLX-1) is withdrawn.
