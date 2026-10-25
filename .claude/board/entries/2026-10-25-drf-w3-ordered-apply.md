# D-DRF-W3 — ordered same-facet writes apply by declared stream position (lab)

**Status:** TEST-PINNED (lab). **Code:** `crates/drf-focus-probe/src/lib.rs`
(`Writeback::stage_at`, `flush`). **Tests:** `tests/integration/focus.rs`
`w3_ordered_*`, `w3_equal_positions_or_mixed_ordering_*`. Plan:
`.claude/plans/d-dr64-focus-cache-falsifiers-v1.md` §4 W3.

- **W3 PASS (ordered arm):** three noncommuting overwrites of one facet (positions 1, 2, 3)
  through three aliasing slots give the position-3 value in all 6 staging orders. 64 writes
  over 4 rows in a scrambled staging order match a scalar oracle (highest position per row).
- **W3 PASS (refusal arm):** different values at one declared position, or an unordered
  `stage` next to a positioned one, refuse with `Conflict`. Equal values coalesce in both.
  An unordered write is checked against every positioned value of its facet, not only its
  sorted neighbour (`stage 1`, `stage_at 1@5`, `stage_at 2@6` refuses; codex P2 on #1766).
- **Mechanism:** patches sort by `(ord, offset, pos)` and apply in that order, so the
  sort is the order of application; no second pass.
- **Disables, each red on its own test:** sort without `pos` (staging order wins); treat
  equal positions as ordered; treat an unordered write as ordered.

OPEN: positions are caller-declared `u64`s. Nothing yet ties them to a real stream
(CycleWriter sequence, WAL LSN); that is W4–W6.
