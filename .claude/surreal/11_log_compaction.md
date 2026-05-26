# 11 — log_compaction

**Repo:** lance-graph · **Branch:** claude/splat3d-cpu-simd-renderer-MAOO0
**Phase:** event-sourcing caveat (bound replay/storage).

## Scope
Fold old container records into a **snapshot container**; bound container count and
fold/read cost over time.

## Owns
- `crates/surreal_container/src/compaction.rs`

## Depends on
06.

## Guards — do NOT touch
- **Fold-equivalence is mandatory**: `fold(snapshot) == fold(original containers)` —
  compaction must never change the recovered state.
- Append-only: write a new snapshot container; retain the originals until safe to GC
  (RCU/epoch — no reader pinned to them).
- Never lose committed state; never overwrite in place.

## Acceptance
- `compaction(containers)` → snapshot; `fold(snapshot) == fold(containers)`.
- Container count bounded after compaction (assert ≤ threshold).
- Originals only GC'd after no reader references them.
