# 2026-10-20 — ARM-B-PREPARE: Arm B's read policy above 1% of centers

**Status:** MEASURED (lab probe `crates/crepe-plan-probe`). Closes ledger (d) item ARM-B-PREPARE for uniform demand. Plan: `.claude/plans/crepe-rule-optimizer-spike-v1.md` §2f.

## What landed

- `addressed::plan_read` (pure, lane positions only) chooses per side between exact ranges, coalesced ranges and a full read; extra rows are dropped by the row's own leading value (co-addressed, no gather).
- `ReadPolicy::{EXACT, FULL, DEFAULT}`; `evaluate_addressed` keeps EXACT, `evaluate_addressed_with` takes a policy. `Lookup` reports the read and the ranges handed to Lance.
- `examples/addressed_read.rs` (`READ_N`, `READ_KS`, `READ_REPS`).

## Measured

- 50,000 rows: exact wins to about 3%, ties 4–6%, loses from 10% (50%: 35.1 vs 18–19 ms).
- 200,000 rows: exact wins to about 5%, ties 7–10% (50%: 128.5 vs 74–79 ms).
- DEFAULT (gap 16, full at 10%) is never measurably worse on either size. Coalescing is neutral at gap 16 and loses at gap 64 on uniform demand.
- 46 tests pass; 5 disable runs red.

## OPEN

- The crossover moves with table size; a size-aware switch is not modelled.
- Clustered demand (where coalescing should pay) is not measured.
- `evaluate_addressed` still defaults to EXACT; no caller uses DEFAULT by default.
