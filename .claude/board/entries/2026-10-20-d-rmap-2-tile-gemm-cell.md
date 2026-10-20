# D-RMAP-2 tile arm: Boolean composition and triangles on one focus (2026-10-20)

**Status:** MEASURED (one cell of D-RMAP-2), probe only, one host. Record: `.claude/plans/d-rmap-2-context-tile-algebra-probe-v1.md` §5a. Code: `crates/mork-pathmap-probe/src/tile_gemm.rs`.

## Measured

- On a 256 × 256 focus, `ndarray::simd::bf16_tile_gemm_16x16_packed` gives walk counts equal to a triple-loop oracle exactly, and `count > 0` equals Boolean composition (1 %, 10 %, 50 %, skewed hubs). 5 guards disable-verified.
- The tier that ran is the F32x16 FMA polyfill (no AMX, no `avx512_bf16` on this host): `A·A` about 19 ms. The bit arm: composition 2.6–25 µs, walk counts by popcount 165–192 µs, triangles 1.9–72 µs. Staging bits to BF16 (16 KiB to 256 KiB) costs 150–400 µs, more than the whole bit composition.
- The polyfill allocates per output element (`ISS-BF16-POLYFILL-PER-ELEMENT-ALLOC`), so this host measures allocation, not GEMM.

## Open

- AMX and `VDPBF16PS` hosts; blocked tiles above one focus; the GraphBLAS arm.
