# 2026-11-03 — D-BAG-FOLD-V4: attested pointer lens and candidate-gated execution

**Status:** MEASURED, lab only (`.claude/probes/2026-11-03-d-bag-fold/probe-v4/`, `REPORT-V4.md`). 21/21 checks. Not promoted; EXO-140 / EXO-46 unchanged; no DT64; mask-risc IR used unchanged; Quack unchanged.

- **A. Attested pointer lens:**
  - A presence witness minted from the source edge facts (production `digest_of` over child keys) refuses a plane misaligned at construction, a permuted plane, and a deletion bypassing generation and frame.
  - A pointer lens minted by witnessed `seek_from` and origin-checked at mint refuses duplicates, out-of-bounds pointers, value-chosen pointers and cross-ClassId pointers.
  - Production `FrameStore` and `OrderedLaneWitness` refuse foreign or stale frames and versions.
  - Remaining gap: a re-sealed forged table passes verification. Origin is attestable only at mint.
- **B. Candidate-gated execution.** One mask-risc program gating both `Gather`s by an in-program register (`Range ∧ T_in`) matches Quack's Arm L and the oracle. It scales with candidates, not lane size: 1.2× on the smallest lane, 4.7–46.6× on larger ones.
  - Quack cannot lower it: `gate_walk` gates only by one resident plane.
  - Presence verification (O(popcount)) costs more than the gated execution, so it must run per generation, not per query.
  - Presence mint re-sorts the source facts on every mutation and crosses 1 ms at ~14k rows.
- **OPEN, owner decisions:**
  - the EXO-140 / EXO-46 rulings;
  - the contract extensions (presence, uniqueness, pointer origin, lens);
  - register-gated `Gather` lowering in Quack;
  - where presence verification runs.
