# EXO-298: per-node open reach is expressible with the existing Cube::reach

**Date:** 2026-11-17 · **Status:** MEASURED (test-pinned) · **Scope:** lance-graph-xlog tests only, no API change

sentinel-rs EXO-298 asks for `sentinel_csi::reason`'s per-node rule (open nodes are reached but never expanded) in its XLOG projection, which today answers `Unknown` for every unreached pair once anything is masked.

The rule needs no new lance-graph API. With stored sets `P={from}`, `T={to}`, `O` (open), `C = all ∧¬ O`:

- `R1 = reach(P ∧ C, C)`
- Reachable iff `to ∈ reach(R1, C ∨ T)`
- Unknown iff `(P ∨ reach(R1, O)) ∧ O` is non-empty; Unreachable otherwise.

`crates/lance-graph-xlog/tests/reach_open.rs` checks this against a breadth-first oracle written after `reason()`: 400 random graphs, each answer seen ≥ 20 times, plus six hand cases. Three disables fail it (no `T` in the step target, no `P` in the open hit, open nodes expanded). `from == to` is excluded: `reason()` never tests `from` against `to`.

Not answered here (sentinel-rs's side): which names get an open coordinate without giving hidden content one.

OPEN: none on the lance-graph side.
