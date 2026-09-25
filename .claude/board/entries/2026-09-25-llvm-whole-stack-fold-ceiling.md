# 2026-09-25 — Whole-stack compilation ceiling: the ≤3-plane fold is already there; Keep and >3 planes are ~3× away

**Status:** MEASURED · ≤3-plane `Keep` fold LANDED 2026-09-25 (see `entries/2026-09-25-keep-fold.md`) · OPEN (a register-resident two-ternlog primitive for 4–5 planes)
**D-ids:** D-WFL-FUSE follow-up.

## What was asked
The operations are always the same, so how much would compiling the whole stack for a fixed chain buy? Where would LLVM inline and vectorise everything, and what is the gain?

## Probe
`examples/llvm_fold_probe.rs`. N = 1 048 576 rows (16 384 words), median of 41. Every arm is checked against a bit-serial oracle, and the inlined `Keep` output is checked word-for-word against the tiled one.

Arms:
- `tiled` — today's fallback, 256-word tile.
- `compiled` — `execute_compiled` on the folded program.
- `const_imm` — `mask_ternlog_popcount::<IMM>` with a literal immediate, as codegen would emit it.
- `inlined` — one plain loop with the chain written as literal Rust. A LAB ARM that bounds what whole-stack compilation can buy; production SIMD stays in `ndarray::simd`.

`target-cpu=native` (AVX-512F/BW/DQ/VL, **no** VPOPCNTDQ):

| chain | tiled | compiled | const_imm | inlined | tiled/inlined | Keep tiled | Keep inlined | ratio |
|---|---|---|---|---|---|---|---|---|
| `(a&b)\|!c` | 18.8 µs | 7.3 µs | 7.2 µs | 6.3 µs | 3.0× | 17.3 µs | 5.1 µs | 3.4× |
| `((a^b)&!c)\|(a&c)` | 24.0 µs | 7.3 µs | 7.2 µs | 6.3 µs | 3.8× | 23.4 µs | 5.1 µs | 4.6× |
| 4 planes `(a&b)\|(c&d)` | 20.8 µs | — | — | 7.1 µs | 2.9× | 20.2 µs | 6.4 µs | 3.2× |
| 5 planes `((a\|b)&c)^(d&!e)` | 26.2 µs | — | — | 9.2 µs | 2.8× | 25.7 µs | 7.4 µs | 3.5× |

Two runs agree within about 1 %.

## Readings
- **For ≤3-plane Count, the runtime fold is already at the ceiling.** `compiled` ≈ `const_imm` ≈ `inlined`, within about 15 %. Per-call overhead (validation, the 256-arm immediate match) is invisible at this size.
- **The gain is everywhere the fold declines: 2.8–4.6×.** That is `Keep` of any chain, and any chain over 4–5 planes. Both paths still run op by op and write every intermediate. The inlined loop reads each plane once and writes only the result.

## Probe corrections made along the way (so the numbers can be trusted)
- The first lab loop indexed five slices separately. Every load was bounds-checked, nothing vectorised, and the time was a flat 35 µs.
- The second version fixed that but passed each chain as a `fn` pointer from the table. That made every word an indirect call LLVM cannot inline, and the time was a flat 25.6 µs, for `Keep` as well.
- In both versions, identical time across different chains was the tell: the loop was measuring overhead, not the chain. The final loop uses literal closures, and an assert checks that they agree with the `fn` table.

## Open
- **Fold `Keep` for ≤3 planes.** One `ternlog_dispatch` pass straight into the `Out::Mask`, edge-masked at an unaligned extent. mask-risc only; the ceiling above says ~3–4×.
- **4–5 planes need a new ndarray primitive** that evaluates two ternlogs in registers (`tern(tern(a,b,c),d,e)`). The facade has no such primitive, and the production path may not use an autovectorised loop. It would be its own ndarray PR.
