# 04 — surreal_container_write  ⚠ hallucination-zone

**Repo:** lance-graph · **Branch:** claude/splat3d-cpu-simd-renderer-MAOO0
**Phase:** SurrealDB write path (container × epoch).

## Scope
Write **one epoch-container as ONE `surrealdb` kv-lance record** (key = container-id,
val = LE bytes from task 03). Append-only, container×epoch cadence. SurrealDB's own
single flusher emits one Lance fragment.

## Owns
- `crates/surreal_container/src/write.rs`

## Depends on
01, 03.

## Guards — do NOT touch
- **Import the container type from ndarray (task 02) — do NOT redefine it.**
- **One record per epoch** (NOT per element/row).
- **Append-only**: a new container-id per epoch — never update-in-place.
- Single writer; no merge logic here (fold is task 06).
- `surrealdb` as dependency only; never patch the fork.

## Acceptance
- Writing N epochs → N records → **exactly one Lance fragment per epoch** (no
  fragmentation; assert fragment count == epoch count).
- Append-only: no fragment/record rewrite (assert prior fragments unchanged).
