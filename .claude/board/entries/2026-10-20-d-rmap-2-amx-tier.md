# D-RMAP-2-AMX: the tile arm on the AMX tier (2026-10-20)

**Status:** MEASURED, probe only, one host, five runs. Record: `.claude/plans/d-rmap-2-context-tile-algebra-probe-v1.md` §5a-AMX. Code unchanged: `crates/mork-pathmap-probe/src/tile_gemm.rs`.

## Measured

- Host reports `AMX TDPBF16PS` (`amx_bf16`, `amx_tile`, `avx512_bf16`; 4 vCPU). The exactness tests pass with the packed entry on AMX.
- Walk counts: AMX `A·A` 58–67 µs against popcount 128–164 µs (about 2.2–2.5×), staging (132–470 µs) excluded; by arithmetic, staging is repaid after about 1.4–6.4 products on one staged map.
- Boolean composition and triangles: the bit arm still wins (6–60× and 1.9–36 µs against 41–323 µs).

## Open

- `VDPBF16PS`-only hosts; blocked tiles above one focus; the GraphBLAS arm; why the row-major entry is 5.6–6.1 ms here against about 19 ms on the §5a host.
