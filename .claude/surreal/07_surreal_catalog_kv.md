# 07 — surreal_catalog_kv

**Repo:** lance-graph · **Branch:** claude/splat3d-cpu-simd-renderer-MAOO0
**Phase:** control plane (catalog + data split).

## Scope
Per-record SurrealDB KV for the **small mutable control state**: goal-state,
pointer → container-id, edges.

## Owns
- `crates/surreal_container/src/catalog.rs`

## Depends on
01.

## Guards — do NOT touch
- **Small mutable control only — NO bulk data** (bulk lives in containers, task 04).
- Records *point into* containers by id; they do not embed container payload.
- This is the only place per-key mutation lives (LWW/superposition decided here
  separately, later — keep it tiny).

## Acceptance
- Goal-state record CRUD.
- Pointer record resolves to a container-id (roundtrip → task 05 fetch).
- Edge record store + scan by endpoint.
