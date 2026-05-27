# 01 — deps_substrate

**Repo:** lance-graph · **Branch:** claude/splat3d-cpu-simd-renderer-MAOO0
**Phase:** POC foundation · everything builds on this.

## Scope
Pin **LanceDB 0.28 / Lance 6** across the workspace; add the `surreal_container`
crate; wire **embedded `surrealdb` with the `kv-lance` feature** (in-process, no
server); align transitive deps.

## Owns
- `Cargo.toml` (workspace dep pins) · `crates/surreal_container/Cargo.toml`
- `crates/surreal_container/src/lib.rs` (embedded Datastore init only)

## Depends on
none.

## Guards — do NOT touch
- Only the files above. Edit-only; the orchestrator compiles/tests centrally.
- **The `surrealdb` fork's `kv-lance` backend is OUT of PR-scope** — use `surrealdb`
  as a *dependency* only; do NOT patch fork internals.
- Lance 6 has API deltas vs what the fork's `kv-lance` uses (`MergeInsertBuilder`,
  `Dataset`). If the bump breaks the fork, **STOP and report** — do not invent API.
- Document (don't perform) the required fork-side Lance-6 bump + the transitive
  `lance-index → ndarray 0.16` constraint.

## Acceptance
- Workspace builds with Lance 6 / LanceDB 0.28 pinned.
- Embedded `surrealdb` Datastore opens with `kv-lance`; smoke test: put/get one record.
