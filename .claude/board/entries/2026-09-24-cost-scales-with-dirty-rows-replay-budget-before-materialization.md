# 2026-09-24 — cycle cost scales with dirty rows; the replay budget before a materialization

**Status:** MEASURED (dirty-row sweep) · WORKING-MODEL (replay budget; per-fold cost UNMEASURED) · OPEN (see below)
**D-ids:** none new — evidence for D-HWV-1 / D-HWV-1a and the seal audits of 2026-09-23/24.

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

## Working model — the replay budget (operator, 2026-09-24)

If replay is deterministic and one fold costs ~1.7 ns, then ~1,000,000 folds cost ~1.7 ms — the
same as the measured per-commit floor. So up to ~1M folds can be replayed from history for the
price of one materialization; materializing more often than that buys nothing.

- The ~1.7 ms commit floor is MEASURED (table above).
- The ~1.7 ns per fold is NOT measured. The only per-row fold figure on record is freeze at 64k,
  ~1.1–1.6 µs/row, and that includes the payload clone, the sort and a byte-wise FNV hash over
  512 B — not a pure fold. A pure fold (e.g. per-row max over stream coordinates, or an alpha
  stamp update) must be measured before this budget is used for a decision.
- "Deterministic replay" rests on the repo's regeneration doctrine (`cycle_driver.rs:180-196`)
  and coordinate-ordered recovery (`persist_sink.rs:656-664`); neither is a measurement of
  replay cost.

## OPEN

- Measure a pure fold per row at 64k (no clone, no hash) to confirm or correct ~1.7 ns.
- The per-commit floor grows with history, not population: one fragment per non-empty cycle;
  whether the manifest still carries the full fragment list (E2 recorded this for lance 9) is
  unverified for lance 11/12.
- Reading the current 64k field = latest image row per node across cycles; not measured.
- `BatchWriter::cast` accepts a payload with no move, so a producer could dirty a row on a kanban
  NOOP; production code does not, but nothing enforces it.
