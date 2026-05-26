# 03 — container_codec  ⚠ hallucination-zone

**Repo:** ndarray · **Branch:** claude/splat3d-cpu-simd-renderer-MAOO0
**Phase:** SoA primary-citizen core.

## Scope
Zero-serialization **encode/decode** of a container as raw LE bytes + a frame
(`u32` LE length + `xxhash`/`crc32` checksum) + SIMD-alignment handling on read.

## Owns
- `ndarray/src/hpc/soa/container_codec.rs`

## Depends on
02.

## Guards — do NOT touch
- **Import the container type from task 02 — do NOT redefine the layout.**
- No DB, no cache (those are lance-graph tasks).
- **Checksum is mandatory** — raw-LE has no "fails to decode" signal; a bit-flip must
  be caught by the checksum, not silently accepted.
- Read path: decode into a 64-byte-aligned buffer for SIMD, OR document the single
  aligned copy. Never touch SIMD dispatch.

## Acceptance
- `encode` → `decode` roundtrip == input.
- Corrupted byte → checksum **rejects** (test).
- Alignment test (decoded buffer usable by `F32x16` load without misalignment).
- Bytes produced here are the cross-crate parity fixture for task 12.
