## 2026-08-11 — E-THE-FRAME-WAS-ALREADY-SHIPPED-FOUR-TIMES-1

**Status:** FINDING `[G]` — read in source this session.

**The `mu + kσ` calibrate → band → roll-on-drift frame exists FOUR times in this
workspace, and I hand-wrote a fifth in Python after being corrected into it.**

| instance | metered quantity |
|---|---|
| `ndarray::hpc::cascade::Cascade` — `expose()` → `Band`, `observe`/`recalibrate` | Hamming distance |
| `perturbation_sim::rolling_floor::RollingFloor` — `threshold()` = *"the **confidence-interval** floor"*, `z()` = *"the **Jirak-honest** noise-floor units; significance via `n^(p/2−1)`, not IID"*, `band()` → Stable…Alarm, `preheat()` | mode instability |
| `helix::quantize::RollingFloor` — `quantize`/`bucket_center`/`observe`/`roll` | palette `[a,b]` |
| `thinking-engine::domino` — *"3σ top-K focus"* | table attention |
| **`probes/weather-p1/p1_ci_vs_floor.py` — MINE** | ERA5 bucket CI |

`perturbation_sim::RollingFloor` **is** the "corrected evaluation frame" that
took three correction sections (§12.10–§12.12) and an operator intervention to
reach — including the Jirak citation, in the doc comment.

**Also found in the same sweep, all `[G]`:**

- **`crates/perturbation-sim` is the applied instance of the whole stack** —
  `splat.rs` (Gaussian-splat **magnitude** side) + `sketch.rs` (Walsh/XOR **sign**
  side) is *literally* OGAR's two-algebra rule; `cascade_key.rs::morton48` is the
  OGAR production HHTL address; `hhtl.rs` derives `(HEEL,HIP,TWIG)` by **Cheeger
  bisection of the Laplacian**.
- **64×64 "Stockfish ergonomics" is an exact identity:** `64×64 = 4096 cells =
  4096 bit = 512 byte = 64 × u64` = **the CANON node stride**. A node's bits ARE a
  bitboard; `masked_popcount_batch(words, mask)` IS `popcount(attacks & targets)`;
  magic bitboards are the same LUT amortization as the `[a,b]` floor.
- **`symbiont/src/domino.rs` already proves the AMX path** — 4×4 Morton BF16
  tiles, **16 SoA boards per AMX 16×16 tile GEMM**, cascade feedback, real
  `TDPBF16PS` on Emerald Rapids.

**Rule:** **before writing a frame, grep the workspace for the frame.** The
`Consult, don't guess` ladder (card → knowledge doc → board → source) has no rung
for *"search the sibling crates for the thing you are about to build"* — four
misses in one document say it needs one. A measured number computed twice is not
a wrong number; it is a wasted one, and it hides the fact that the first
implementation already carried the caveats you were about to rediscover.

