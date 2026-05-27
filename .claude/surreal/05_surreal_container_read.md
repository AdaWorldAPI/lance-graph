# 05 — surreal_container_read

**Repo:** lance-graph · **Branch:** claude/splat3d-cpu-simd-renderer-MAOO0
**Phase:** SurrealDB read path.

## Scope
`get` a container by id; **range-scan** container-ids to feed the fold (task 06).

## Owns
- `crates/surreal_container/src/read.rs`

## Depends on
03, 04.

## Guards — do NOT touch
- Read-only. Decode via the task-03 codec; do not re-implement decode.
- Import the container type from ndarray (task 02).
- No fold semantics here (task 06); no cache here (task 08).

## Acceptance
- Roundtrip: read a written container, decode → == original.
- Range-scan returns containers in id order (ascending), tombstones excluded.
