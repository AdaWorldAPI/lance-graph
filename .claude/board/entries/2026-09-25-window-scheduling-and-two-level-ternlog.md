# 2026-09-25 — Mississippi-Queen windows lose; a two-level ternlog fold wins 1.7–1.8× for 4–5 planes

**Status:** MEASURED · OPEN (a production `Tern2` lowering)
**Probe:** `crates/lance-graph-mask-risc/examples/window_sched_probe.rs` (n = 2^20, `target-cpu=native`, median of 41, three runs within about 5 %). Every arm is asserted equal to the tiled result: Count by value, Keep word for word.

## The question
Can op-by-op execution get closer to the whole-stack ceiling if intermediates live only in a window that moves with the frontier? (Mississippi Queen: board tiles are laid ahead of the lead boat and picked up behind the last.) The candidates were fixed chunks (`as_chunks::<8>` as `U64x8` registers), variable slices, or `array_windows`.

## Numbers (Count, µs)

| chain | tiled T=256 | window K=1 | K=2 | K=4 | K=8 | tern2 T=8 | tern2 T=64 | tern2 T=256 | inlined (ceiling) |
|---|---|---|---|---|---|---|---|---|---|
| `(a&b)\|!c` | 18.2 | 26.8 | 23.1 | 20.5 | 19.7 | 31.0 | 11.3 | 11.3 | 6.2 |
| 4p `(a&b)\|(c&d)` | 21.5 | 29.0 | 28.7 | 24.9 | 24.2 | 31.0 | 12.5 | 12.6 | 7.2 |
| 5p `((a\|b)&c)^(d&!e)` | 26.5 | 34.0 | 30.6 | 27.2 | 24.6 | 32.0 | 14.6 | 14.4 | 9.2 |

Keep runs show the same order: the window at K=8 ≈ tiled, and smaller windows are worse.

## Classification (per the folding doctrine's cost classes)
- **The cost is INTERPRETATION, not ACCESS.** The 256-word scratch tile is already L1-resident (2 KB per slot). Shrinking the board to register width saves no memory traffic that was being paid, and it pays one `match` per op per window instead of per 256 words. So the moving window is already here, at the right size. A smaller board only adds overhead.
- **`array_windows` is not the tool.** Its windows OVERLAP (stride 1), so an elementwise Boolean op would recompute 7 of every 8 words. It fits only ops that read a neighbour word (run carries, cross-word shifts). None of the chain ops do.
- **What does win removes interpretation:** fewer facade calls per tile.
  - `t = tern1(a,b,c)` into a T-word chunk, then `popcount(tern2(t,d,e))`. That is two calls per tile, whatever the op count, and one slot.
  - It gives 1.7–1.8× on 4–5 planes using EXISTING kernels, and 1.6× even on the 3-plane chain against tiled. It is still 1.6–1.8× off the ceiling.
  - T=8 loses because per-call overhead returns. T=64..256 is the knee, which agrees with the tile sweep.

## Open — a `Tern2` lowering (mask-risc only, no new ndarray primitive)
- Interpret the chain symbolically over ≤5 leaves into a 32-bit truth table.
- Look for a simple disjoint decomposition `f = h(g(x,y,z), u, v)`. For a choice of the inner three leaves (10 choices), every one of the four `(u,v)` restrictions of `f` must lie in `{0, g, ¬g, 1}` for one common `g`. That yields `imm1 = g` and `imm2 = h`.
- Execute chunked at the scratch tile width: one slot, two facade calls per tile. Keep writes `tern2` straight into `Out::Mask`, with the edge rules of the ≤3-plane Keep fold.
- Chains that do not decompose keep the tiled path. The gap to the ceiling is still the ndarray kernel's loop shape (see `entries/2026-09-25-keep-fold.md`).
