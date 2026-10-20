# 2026-10-20 — D-RMAP-2-AMX-RERUN — §5a-AMX re-measured on a second AMX host

**Status:** MEASURED, lab probe only. **Basis:** `crates/mork-pathmap-probe` at `main` `bc8012da`; `cargo test --release --test tile_gemm` (5 passed); five runs of `cargo run --release --example tile_gemm_cost`, each the example's medians of 7. Host flags `amx_bf16`, `amx_tile`, `amx_int8`, `avx512_bf16`; `Intel(R) Xeon(R) Processor @ 2.10GHz`, 4 vCPU; ndarray tier `AMX TDPBF16PS`.

Median of five, µs. The #1626 value is in brackets.

| Map | bit compose | popcount counts | AMX `A·A` | stage | threshold | bit triangles | BF16 triangles | row-major |
|---|---|---|---|---|---|---|---|---|
| 1 % | 2.1 (2.0) | 181 (164) | 88 (67) | 182 (132) | 61 (50) | 2.5 (1.9) | 57 (45) | 8 819 (5 842) |
| 10 % | 7.6 (6.0) | 184 (135) | 80 (58) | 222 (182) | 133 (105) | 10.8 (7.4) | 110 (94) | 7 789 (5 823) |
| 50 % | 34 (27) | 174 (137) | 84 (63) | 501 (470) | 141 (101) | 48 (36) | 422 (323) | 9 000 (6 107) |
| hubs | 3.7 (2.6) | 174 (128) | 82 (58) | 171 (143) | 133 (97) | 6.0 (3.7) | 63 (41) | 8 819 (5 599) |

- **The ratios hold:**
  - walk counts: AMX over popcount 2.06–2.31× (#1626: 2.2–2.5×);
  - staging repaid after 1.85–5.57 products (#1626: 1.4–6.4);
  - bit arm over the GEMM for composition 6.6–71×, for triangles 8.8–22.6×;
  - the threshold alone costs more than bit composition on every map.
- **Absolute times are 10–60 % higher in every cell.** That points to load on the shared VM, not a code change; it is not verified.
- The `bf16_bytes` column reads 393 216 (384 KiB): the #1625 fix is on `main`.
- **Arithmetic check of #1626's own table:** ratios 2.18–2.46×, staging repaid after 1.36–6.35 products, composition 6.1–59.6×, triangles 8.9–23.8×; every derived claim in its body matches.

OPEN: why the row-major entry (no AMX) is 6–11 ms on these hosts against about 19 ms on the §5a host.
