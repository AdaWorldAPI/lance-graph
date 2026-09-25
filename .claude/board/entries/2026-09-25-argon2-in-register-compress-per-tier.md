# 2026-09-25 — Argon2 in-register compress: routing won on v4, spills cancelled it on v3

**Status:** MEASURED · SHIPPED (password-hashes #3) · OPEN (v3 spill reduction; transposed block storage)

`compress_simd` in the AdaWorldAPI/password-hashes fork now keeps the block in
`ndarray::simd::U64x8` registers (8 `transpose8`, ndarray #332). Numbers and the
per-tier instruction census are in `.claude/knowledge/folding-doctrine.md` §6.5.

- v4: 111–115 ms → 91–95 ms (about 18 % over the gather/scatter SIMD path). 709
  instructions in `Argon2::compress`, 64 stack ops.
- v3: flat (118–124 → 121–128 ms). 2683 instructions, 478 stack ops: 16 live
  `U64x8` = 32 `ymm` against 16 architectural registers.

OPEN: on v3, running each pass as two halves of 4 permutations (16 live 4-lane
vectors, one `ymm` each) might remove the spills; unmeasured. Keeping blocks transposed would drop the 4
store transposes but changes what `Block::as_ref()` exposes; not decided.
