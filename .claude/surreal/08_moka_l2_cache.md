# 08 — moka_l2_cache

**Repo:** lance-graph · **Branch:** claude/splat3d-cpu-simd-renderer-MAOO0
**Phase:** bounded working-set (Gap #1, "bounded" half — POC substitute for the NARS bag).

## Scope
A **bounded L2 RAM cache** (moka, size/weight-based eviction) as a read-through over
container reads (task 05).

## Owns
- `crates/surreal_container/src/cache.rs`

## Depends on
05.

## Guards — do NOT touch
- **Hard-bounded by capacity/weight** (this is the anti-unbounded-growth guarantee).
- Generic eviction (TinyLFU) only — **NOT** NARS-semantic forgetting (Phase 2).
- Read-through over task-05; no write-back that bypasses the writer (task 04).
- moka is the L2 tier — it is NOT the L1 focus tile (that's Phase 2) and NOT durable.

## Acceptance
- Cache size never exceeds configured capacity (eviction test under load).
- Read-through hit/miss correctness == direct read (parity).
