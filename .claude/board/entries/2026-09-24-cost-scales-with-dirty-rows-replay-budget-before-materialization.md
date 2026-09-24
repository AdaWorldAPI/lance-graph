# 2026-09-24 — cycle cost scales with dirty rows; the replay budget before a materialization

**Status:** MEASURED (dirty-row sweep; commit floor) · MEASURED elsewhere (fold cost, #1245 / #1250) · OPEN (see below)
**D-ids:** none new — supplies the measured `C_materialize` term for D-WFL-ECON; evidence for D-HWV-1 / D-HWV-1a.

## The invariant (DECISION, operator 2026-09-24)

Population = 64k → reserve 64k slots, always. The 32 MiB reserved field is intentional
substrate, not overhead. NULL / empty reserved slots are not a defect. The defect would be
unchanged rows crossing the dirty/materialization boundary merely because a cycle ticked
(kanban NOOP rows copied, hashed, folded or rewritten). Target: cost ∝ dirty rows, never
∝ population.

## Measured — dirty-row sweep through the real path

`seal_cycle` → retry clone → `persist_cycle` artifact gate → freeze → `LanceCycleWriter::commit_cycle`,
lance 11, release, 4 cores, 3 runs each, after 64k rows were resident. Dirty rows carry a 512-B
payload and no move; NOOP rows cast nothing (as the only production producer,
`owner_adapter.rs:88-100`, does). Scratch binary outside the repo:
`/tmp/claude-0/cyclesink-probe/src/bin/dirtysweep.rs`.

| dirty | landings / image | rows emitted | seal ms | freeze ms | bytes written | new versions |
|---|---|---|---|---|---|---|
| 0 | 0 / 0 | 0 | 0.00 | 0.00 | 0 | 0 |
| 1 | 1 / 1 | 3 | 1.6–2.6 | 0.00 | ~6.0 KB | 1 |
| 10 | 10 / 10 | 21 | 1.7–2.1 | 0.01 | ~15.9 KB | 1 |
| 100 | 100 / 100 | 201 | 2.1–2.3 | 0.12 | ~111 KB | 1 |
| 1,000 | 1,000 / 1,000 | 2,001 | 5.1–7.9 | 1.2 | ~1.04 MB | 1 |
| 65,536 | 65,536 / 65,536 | 131,073 | 533–583 | 72–107 | ~67.8 MB | 1 |

Clone, fold and hash run only over submitted casts. Above a fixed floor of ~1.7 ms, 3 files and
one manifest per non-empty cycle, cost scales with dirty rows. An all-NOOP cycle is free:
`NoChange`, no sink call, no version.

## The replay budget (operator, 2026-09-24) — D-WFL-ECON with both terms measured

D-WFL-ECON already rules: *if thinking again is cheaper than remembering the answer, think
again* (`C_retain` vs `C_replay = ΣC_fold + …`). This entry supplies the materialization side.

- **One fold ≈ 1.7 ns** — MEASURED in #1245 (six-tier axis chain); #1250 measured 1.7–4.2 ns
  for a whole-facet cell (`waben-fold-execution-loop-v1.md:416-424`).
- **One materialization ≈ 1.7 ms** — MEASURED here: the fixed floor of a non-empty cycle commit
  (table above).
- **So one materialization buys ≈ 1,000,000 folds** (1.7 ms / 1.7 ns), or ≈ 400,000 at the
  whole-facet 4.2 ns. With deterministic replay, up to that many folds can be replayed from
  history before a materialization pays for itself — and every dirty row adds to the
  materialization side (≈ 4–9 µs per dirty row above the floor, from the 1k/64k rows).

Deterministic replay rests on the regeneration doctrine (`cycle_driver.rs:180-196`) and
coordinate-ordered recovery (`persist_sink.rs:656-664`).

## OPEN

- Which fold a cycle replays (axis chain vs whole-facet cell vs an alpha-stamp update) decides
  whether the budget is ~1M or ~400k; D-WFL-ECON's W6 owns that measurement.
- The per-commit floor grows with history, not population: one fragment per non-empty cycle;
  whether the manifest still carries the full fragment list (E2 recorded this for lance 9) is
  unverified for lance 11/12.
- Reading the current 64k field = latest image row per node across cycles; not measured.
- `BatchWriter::cast` accepts a payload with no move, so a producer could dirty a row on a kanban
  NOOP; production code does not, but nothing enforces it.
