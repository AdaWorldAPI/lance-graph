# 06 — read_time_fold

**Repo:** lance-graph · **Branch:** claude/splat3d-cpu-simd-renderer-MAOO0
**Phase:** merge semantics (non-destructive). Resolves Gap #2.

## Scope
Reconstruct state from append-only containers. **Default = LWW-latest.**
**Superposition behind a `trait Merge` hook — NOT wired** (Phase 2).

## Owns
- `crates/surreal_container/src/fold.rs`

## Depends on
05.

## Guards — do NOT touch
- Default merge = **LWW-latest** (latest container's view wins).
- Superposition = a `trait Merge` with an LWW default impl + a **stub** superposition
  impl behind a marker. **Do NOT wire holograph superposition** (Phase 2 / risk).
- **Non-destructive**: the fold never deletes/overwrites containers — all retained.

## Acceptance
- `fold(containers)` recovers the latest committed state (LWW).
- Non-destructive: all input containers still present after fold.
- `trait Merge` compiles with LWW default + a no-op superposition stub.
