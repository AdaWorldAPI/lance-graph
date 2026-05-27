# 10 — lockfree_handoff_ring

**Repo:** lance-graph · **Branch:** claude/splat3d-cpu-simd-renderer-MAOO0
**Phase:** the zone seam (internal core ↔ ractor edge).

## Scope
A **lock-free SPSC/MPSC ring** that **moves container ownership** between an
OS-thread producer (internal preemptive core) and the ractor edge (task 09).

## Owns
- `crates/surreal_container/src/ring.rs`

## Depends on
02.

## Guards — do NOT touch
- **Ownership MOVE only** — no shared `&mut`, no clone of bulk payload.
- **Non-blocking** (`try_send`/`try_recv`); never block the OS-thread producer.
- Payloads are `Send + 'static` (forces owned transfer — the compile-time borrow
  guarantee; a borrowed `&[u8]` view must NOT be sendable).
- This is the ONLY crossing between zones; nothing else shares state across it.

## Acceptance
- Move-across-ring preserves the container (ownership transferred, not aliased).
- Non-blocking under contention (no producer stall).
- No data race (single-owner-at-a-time test; `loom` if available).
