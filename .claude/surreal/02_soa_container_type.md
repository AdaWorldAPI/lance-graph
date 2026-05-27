# 02 — soa_container_type  ⚠ hallucination-zone

**Repo:** ndarray · **Branch:** claude/splat3d-cpu-simd-renderer-MAOO0
**Phase:** SoA primary-citizen core · **the ONE source of truth for the layout.**

## Scope
Define the LE **`#[repr(C)]`, pointer-free SoA *container* type** built ON the
existing `hpc::soa` (#156). Append-only semantics. This layout is defined **here,
once** — lance-graph imports it, never redefines it.

## Owns
- `ndarray/src/hpc/soa/container.rs` (+ re-export from `hpc::soa`)

## Depends on
none (anchor for 03/04/05).

## Guards — do NOT touch
- **Build ON `hpc::soa` (`SoaVec`/`soa_struct!`) — do NOT invent a new SoA system**
  (duplication = the #1 hallucination).
- **Pointer-free**: only inline scalars + intra-buffer offsets — no `Box`/`Vec`/ptr
  in the on-wire layout (must be `bytemuck::Pod`).
- `#[repr(C)]`, explicit alignment. Never touch SIMD dispatch.
- This is the **single layout definition** — if you feel the urge to define it in
  lance-graph too, STOP.

## Acceptance
- **Golden-byte test**: exact LE bytes of a known container == committed fixture.
- `bytemuck` cast roundtrip; compile-time `Pod` (pointer-free) proof.
- `static_assert` on `size_of`/`align_of`.
