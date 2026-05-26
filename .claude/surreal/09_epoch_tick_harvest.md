# 09 — epoch_tick_harvest

**Repo:** lance-graph · **Branch:** claude/splat3d-cpu-simd-renderer-MAOO0
**Phase:** edge control loop (cold path).

## Scope
A **ractor edge actor**: at epoch cadence, take a lock-free snapshot, check
**goal-state solved?**, and on FLOW dispatch a container-write commit (task 04).

## Owns
- `crates/surreal_container/src/tick.rs`

## Depends on
04, 07.

## Guards — do NOT touch
- **Cold-path / edge only — NEVER per-cycle.** This is where ractor (async) lives.
- The solved-predicate read must be **lock-free** (snapshot/epoch read) — it must
  never stall a writer.
- **No reasoning here** — only `solved?` check + dispatch. (NARS is Phase 2.)
- Tick = harvest cadence, NOT a compute deadline.

## Acceptance
- Tick fires at the configured cadence.
- The snapshot read does not lock the writer (concurrent write proceeds).
- FLOW dispatches exactly one container-write per solved goal.
