# 2026-11-11 — Cypher counts: DataFusion against the Quack lowering

**Status:** MEASURED (bench). `crates/lance-graph-benches/benches/cypher_df_vs_quack.rs`.
Same inputs as `graph_execution` (ring of n edges, `age = 20 + i % 60`), read through Lance;
every arm's answer equals a row-at-a-time oracle before timing. One machine, 4 cores,
criterion `--warm-up-time 1 --measurement-time 4`; medians.

| case | n | DataFusion | Quack compile + execute | Quack execute |
|---|---|---|---|---|
| filter `count(*)` | 100 | 1.34 ms | 4.47 µs | 175 ns |
| | 10k | 1.46 ms | 6.89 µs | 1.56 µs |
| | 1M | 2.17 ms | 297 µs | 268 µs |
| one-hop `count(*)` | 100 | 1.96 ms | 5.46 µs | 604 ns |
| | 10k | 2.19 ms | 38.1 µs | 33.9 µs |
| | 1M | 26.8 ms | 3.68 ms | 3.45 ms |
| two-hop `count(*)` | 1M | 43.7 ms | refused (`Refusal::Gap`) | — |

Intake, `person` at 1M: full scan 26.1 ms, projected scan of the two bound columns 9.19 ms.

**Filter:** 0.27 ns/row at 1M, which is the bandwidth of reading 4 MB of `age`. In the 1–25 µs
band at 100 and 10k rows.

**One-hop: not SIMD-shaped. 3.4 ns per edge at both 10k and 1M.** The program gathers the
node-live bit for `src` and then for `dst` (`MaskOp::Gather`). `ndarray::simd::mask_gather_u32`
and `mask_gather_u32_under` (`src/simd_masking_ops.rs:1131`, `:1220`) have no backend dispatch:
both are scalar loops in the facade file, one bit extracted per row, and the gated one walks set
bits with `trailing_zeros`. With AVX-512 the same work is a 16-lane `vpgatherdd` on the source
words, a variable shift, and a compare into a 16-bit kmask written straight into the output word.
That would bring 10k into the 1–25 µs band. At 1M, two random bit reads per edge cannot fit in
25 µs (80 edges/ns); the floor is the gather's memory traffic.

**Open:** the SIMD gather in ndarray (`simd_avx512` arm plus a parity test against the scalar
form); two-hop needs a foreign-value sum before Quack can answer it.
