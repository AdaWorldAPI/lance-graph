# D-APERTURE-16-0 — 4096 × u16 aperture scheduling over a known 64K self-space

**Status:** MEASURED, 2026-10-07. Probe: `crates/lance-graph-benches/examples/aperture16_probe.rs` (modes `a1 a2 occ a3 a4 a5 a6`).
**Machine:** 4-vCPU Xeon @ 2.80 GHz, AVX-512F. Per core: L1d 32 KiB, L2 1 MiB; L3 33 MiB shared. Load average 0.06 at start; release build with `debug = 0`; median of 101 runs (31 for `a2`).
**Correctness:** every route's answer is asserted equal to a reference before its time is printed. Only answers that passed were recorded.

## 0. Answer first

**The law survives; the representation does not.**

```
known self-space
→ remove impossible regions   (ordered extent → [lo, hi))
→ remove closed mask regions  (skip zero mask words)
→ visit only live ordinals    (set-bit walk)
→ touch payload bytes last
```

Every part of that rule is measured and holds. Its best execution unit is the **u64 mask word, not the u16 cell**. As a scheduling unit, the u16 cell is 1.6–2.8× slower on geometric mean, and slower almost everywhere. It earns its place in exactly one narrow role: finding full 16-runs inside a partial word, on data that contains them.

## 1. Current mask representation (read before building)

| what | where | shape |
|---|---|---|
| support mask | `lance-graph-mask-risc` `Planes::masks`, `Scratch` slots | `&[u64]`, row `i` = bit `i & 63` of word `i >> 6`, little-endian bit order, tail bits zero |
| mask iteration in fold terminals | ndarray `simd_masking_ops.rs`: `masked_sum_i32` (:813), `group_walk` (:1616), `mask_scatter_or_u32` (:1205) | **already** skip a zero word in one test, then walk set bits with `trailing_zeros`: the `u64-visit` route here, transcribed |
| `mask_gather_u32` | ndarray `simd_masking_ops.rs:1099` | full scan of `index`, a data-dependent branch per row, no gate (unchanged since D-GATED-GATHER-0) |
| `MaskOp::Gather` | mask-risc `ir.rs:269`, `exec.rs:1753` | `{ lane, foreign, dst }`; **no `under`**, unlike `MaskOp::Pred` (`ir.rs:227`) |
| `Pred::Range` / `execute_extent` | `ir.rs:180`, `exec.rs:1393`; tiles from `extent_tiles` (`exec.rs:1246`) | work proportional to the extent, edge words clipped by `clip` |
| `GroupReduce` / Via | `exec.rs:1920`: `masked_group_*_u32{,_via}` | all route through `group_walk`, so they are u64 set-bit walks |

So production already runs the u64-word schedule everywhere except `Gather`. This lab asks whether 16-self granularity would improve on it.

**Name collision.** `lance_graph_quack::SemanticAperture` (`quack/src/lib.rs:925`) already means a facet-key care mask, lowered to a bound. The execution idea measured here needs a different name, or the word comes to mean two things. This document uses "aperture" only for the brief's 16-self cell.

## 2. The zero-copy u16 view

`[u64; 1024]` and `[u16; 4096]` are the same 8 KiB. The probe reads cell `c` as `(words[c >> 2] >> ((c & 3) * 16)) as u16`. That is a shift of an existing word, with no allocation and no cast, and it gives the same bits on any endianness. On little-endian it is also exactly `align_to::<u16>()`; `views_agree()` checks all 4096 cells at start-up. **So the view is free, and its memory footprint is identical.** Any win therefore has to come from execution geometry (§7 of the brief), and no new type is needed to express it.

## 3. Configurations

- Universe: N = 65 536 selves. Exactly `round(d·N)` live selves per pattern.
- Densities: 100 / 75 / 50 / 25 / 10 / 5 / 1 / 0.1 / 0.01 / 0.001 / 0.0001 %. At N = 65 536 the last three are 7, 1 and 0 survivors, so they measure the empty-mask floor, not a density.
- Layouts, never averaged together:
  - uniform;
  - clustered (runs of 128–640);
  - one contiguous range;
  - tiny runs (2–4);
  - fully dense 16-islands in a sparse universe;
  - alternating bits;
  - one survivor per 16-cell;
  - one survivor per 64-bit word.
- Payload lanes: u8, u16, u32, u64, plus records of 16/32/64/128 bytes (every byte folded). Working sets run from 64 KiB to 8 MiB, which crosses L1, L2 and L3.
- Routes:

  | route | what it does |
  |---|---|
  | `scan-branch` | per row, branch on its bit |
  | `full-branchless` | load every row, AND with its bit |
  | `u64-visit` | production shape |
  | `u16-visit` | test each u16 cell |
  | `u64→u16` | u64 test, then u16 quarters |
  | `adapt16` | per cell: skip / dense 16 / set-bit |
  | `adapt64` | per word: skip / dense 64 / set-bit |
  | `adapt64>16` | u64 skip, dense full words, full u16 quarters dense, otherwise set-bit |
  | `adapt64q` | branch-free full-quarter detection, then one set-bit loop |
  | `ordinal-list` | materialise `Vec<u16>`, then fold; the build is timed and its bytes reported |

## 4. Correctness parity

Every route's answer is asserted against a reference before timing, for every pattern × payload × route in every mode, and the run aborts on a mismatch. All runs completed: 348 + 4 640 + 85 + 288 + 125 + 192 + 120 measured rows. The u16 view also agrees with `align_to::<u16>` on all 4096 cells (§2).

## 5. Density crossover — visitation only (A1)

Times in µs. Full table: `.claude/research/D-APERTURE-16-tables.md` §A1.

| layout | density | u64 visit | u16 visit | u64→u16 | per-row scan | ordinal list |
|---|---|---|---|---|---|---|
| uniform | 100 % | 46.3 | 47.2 | 54.5 | 53.2 | 56.2 |
| uniform | 50 % | 25.1 | 49.9 | 35.6 | 53.2 | 29.8 |
| uniform | 10 % | 4.7 | 21.9 | 7.5 | 53.3 | 6.3 |
| uniform | 1 % | 0.9 | 3.9 | 2.2 | 53.2 | 1.5 |
| uniform | 0 live | **0.5** | **3.6** | 0.7 | 53.2 | 0.5 |
| clustered | 50 % | 23.0 | 26.3 | 28.2 | 53.2 | 28.3 |
| islands16 | 50 % | 23.4 | 37.4 | 30.3 | 53.2 | 28.6 |

- **The empty-mask floor is 4096 vs 1024 tests: 3.6 vs 0.5 µs, a factor of 7.** The u16 cell visit has to test four times as many units, and nothing it saves offsets that.
- At mid density the u16 visit is 2–5× slower on scattered data: each cell's set-bit loop exits four times as often, and those exits are mispredicted.
- At 100 % the two routes converge, because the work is all set-bit visiting.
- The u64 → u16 hierarchy nearly recovers the floor (0.7 µs) but never beats the plain u64 walk.

## 6. Local occupancy crossover (`occ`)

Every one of the 4096 cells holds exactly k live bits. Times in µs (full table §OCC).

| payload | k = 1 | 4 | 8 | 10 | 12 | 15 | 16 |
|---|---|---|---|---|---|---|---|
| u32, u16 set-bit | 10.8 | 15.0 | 28.6 | 36.0 | 44.1 | 87.1 | 61.7 |
| u32, dense masked 16 | 46.4 | 34.1 | 34.1 | 34.1 | 34.1 | 43.0 | 34.5 |
| u32, u64 set-bit | 4.9 | 11.0 | 21.5 | 26.9 | 32.1 | 68.2 | 43.5 |
| u32, adapt16 (dense only at k = 16) | 13.4 | 17.4 | 31.5 | 39.1 | 81.1 | 61.5 | **10.8** |

- A dense masked 16-lane loop is flat in k, at about 33–34 µs for u8/u32/u64. It overtakes a u16 set-bit walk at k ≥ 10, and a u64 set-bit walk only at k ≥ 13.
- For 32-byte records it never wins. For 128-byte records the loop is memory-bound and the rows are noise.
- **The one large local win is the unmasked dense path at k = 16:** 10.5–11.6 µs against 42–67 µs for set-bit visiting, about 4–6×.
- So local density can choose an execution mode without changing the carrier. But only `cell == 0xFFFF` is a reliable switch; a popcount threshold is a narrow, payload-dependent band and not worth a branch.

## 7. Payload-width ladder (A2)

Times in µs. Geometric mean over all 464 pattern × payload cases, relative to `u64-visit`:

| route | geomean vs u64 visit | best | worst |
|---|---|---|---|
| u16-visit | 2.79 | 0.50 | 13.3 |
| u64→u16-visit | 1.38 | 0.47 | 4.67 |
| adapt16 | 2.07 | 0.16 | 11.3 |
| **adapt64** | **0.80** | 0.05 | 3.58 |
| adapt64>16 | **0.73** | 0.04 | 3.31 |
| adapt64q | 0.85 | 0.13 | 3.46 |
| ordinal-list | 1.31 | 0.32 | 3.17 |
| scan-branch | 14.6 | 0.68 | 181 |
| full-branchless | 17.8 | 0.96 | 922 |

By layout, for cases where `u64-visit` takes ≥ 2 µs (below that the numbers sit at the timer floor):

| layout | adapt64 | adapt64>16 | adapt64q | u16 visit |
|---|---|---|---|---|
| uniform | **0.79** | 0.99 | 1.02 | 1.57 |
| tiny runs | **0.86** | 0.96 | 1.01 | 1.69 |
| clustered | 0.39 | **0.36** | 0.46 | 1.59 |
| single range | **0.32** | 0.33 | 0.39 | 1.52 |
| islands16 | 0.80 | **0.41** | 0.49 | 1.56 |
| one-per-16 | **0.90** | 1.21 | 1.60 | 1.56 |
| one-per-64 | **1.06** | 1.43 | 1.80 | 2.16 |

What the ladder says:

- **A u16 granule helps only when there are full 16-runs that do not fill 64.** That is islands16 (0.41 vs 0.80) and partly clustered data. Everywhere else it costs.
- **The branch-free variant (`adapt64q`) does not remove that cost.** Searching for runs is O(live words) of comparisons, and on scattered data, which has no runs, the search is pure overhead: up to 2×, and 3.5× in one case (one-per-64, 128-byte records).
- `adapt64`, which only adds a dense path for a full word, has the best risk profile. It is never much worse than `u64-visit` (1.06 at worst per layout) and is 3× better on runs.

**Does the benefit grow with payload width?** It depends on density. Ratio `full-branchless / u64-visit`, uniform layout:

| density | u8 | u32 | u64 | rec32 | rec128 |
|---|---|---|---|---|---|
| 100 % | 1.1 | 1.1 | 1.2 | 1.0 | 1.0 |
| 10 % | 10.6 | 7.1 | 20.6 | 8.6 | 7.8 |
| 1 % | 49 | 49 | 38 | 51 | **153** |
| 0.1 % | 107 | 101 | 101 | 127 | **352** |

- **At ≤ 1 % the benefit grows with width:** saved bytes scale with W, while the skip costs a fixed 1024 tests.
- **At ≥ 10 % it does not.** Uniform survivors leave almost every 64-byte line live for lanes under 64 B (at 10 % density a 16-row u32 line is live with probability 1 − 0.9¹⁶ ≈ 81 %). Few bytes are saved, and the remaining win is compute.
- **Runtime tracks work actually left, not N** (correlation of `u64-visit` time with what remains):

  | payload | r with live rows | r with live payload lines |
  |---|---|---|
  | u8 | 0.96 | 0.71 |
  | u16 | 0.99 | 0.81 |
  | u32 | 0.99 | 0.87 |
  | u64 | 0.98 | 0.92 |
  | rec16 | 0.92 | 0.94 |
  | rec32 | 0.97 | **0.99** |
  | rec64 | 0.96 | 0.96 |
  | rec128 | 0.95 | 0.95 |

  For narrow lanes, rows predict runtime better. From 32-byte records upward, live lines predict it as well or better. That is the refinement "self first, bytes later" needs: **the bytes that matter are the touched lines, and a lane narrower than a line saves rows, not bytes.**

## 8. VIA (A3)

Routes: full scan, u64 gate, u16 gate, ordinal. `via[self] → target: u16`. Support (`target_mask |= 1`) and multiplicity (`K_next[target] += K[self]`) are timed separately. Each run includes clearing the target (8 KiB for support, 256 KiB for `K_next`).

| target | src density | support: full / u64 / u16 / ord (µs) | multiplicity: full / u64 / u16 / ord (µs) |
|---|---|---|---|
| uniform | 100 % | 109 / **80** / 85 / 138 | 180 / **131** / 136 / 234 |
| uniform | 50 % | 106 / **45** / 66 / 74 | 431 / **76** / 88 / 113 |
| uniform | 1 % | 194 / **3.8** / 6.7 / 3.5 | 154 / **19.8** / 23.7 / 20.7 |
| uniform | 0.01 % | 106 / 1.8 / 5.7 / 1.6 | 147 / 18.5 / 22.8 / 18.4 |
| hotspot | 100 % | **112** / 137 / 137 / 165 | 219 / **111** / 117 / 180 |
| many-to-one | 50 % | 136 / **76** / 111 / 154 | 433 / **114** / 136 / 165 |

- **The mask can schedule VIA with no selection vector.** The u64 gate wins or ties every case except one family: support at 100 % source density into concentrated targets. There a branch-free full scan wins: hotspot 112 vs 137 µs, zipf 123 vs 152, many-to-one 136 vs 158. Concentrated scattered writes defeat the visit loop, and at 100 % there is nothing to skip.
- The u16 gate is never better than the u64 gate.
- **Support and multiplicity behave differently at the floor.** At low density, support costs ~1.8 µs (clearing 8 KiB), but multiplicity costs ~18.5 µs regardless of the route, because clearing the dense 256 KiB `K_next` dominates. §11 removes that cost.

## 9. K propagation over CSR (A5)

`K_next(v) = Σ K(u)` for edges `u → v`, with fan-out 1, 4 or 16 (up to 1 M edges). Times in µs; full table §A5.

| target | fanout | src density | edge scan | src u64 gate | src u16 gate | ordinal |
|---|---|---|---|---|---|---|
| uniform | 1 | 100 % | 186 | **182** | 187 | 242 |
| uniform | 4 | 100 % | 767 | **375** | 382 | 477 |
| uniform | 16 | 100 % | 2984 | 1288 | **1271** | 1376 |
| uniform | 16 | 10 % | 2732 | **171** | 186 | 183 |
| uniform | 16 | 0.1 % | 2591 | **20.3** | 24.8 | 21.8 |
| hotspot | 16 | 100 % | 2857 | 1890 | 1939 | **980** |

- **Source gating beats the edge scan even at 100 % density** once fan-out ≥ 4: 2.0–2.8× for uniform, clustered and hotspot targets, but only 1.1–1.15× for many-to-one. It reads each source's K once and walks its CSR row contiguously, instead of testing a bit and re-reading `src_of_edge` per edge.
- **Not reproducible: the dense hotspot fan-out-16 row.** Its ordinal "win" (980 vs 1890 µs) reversed between runs: the first run gave 1711 vs 958. Concentrated writes into 64 hot targets make it unstable. Recorded as noise, not as a crossover.
- A clustered fan-out-4 "ordinal win" in the first run (372 vs 574 µs) also failed to reproduce: the second run gave 347 vs 300.
- At ≤ 10 % source density the u16 gate trails the u64 gate in every row. At 100 % the two are within run-to-run noise, and either can lead: uniform fan-out 16 gives 1271 vs 1288 µs, hotspot fan-out 1 gives 183 vs 213.

## 10. Coarse target histogram — the multiplicity aperture (A6)

The question: is a coarse target-side object worth an extra pass? Each route propagates K, clears `K_next` for the next round, and builds the next frontier mask (`K_next > 0`). Times in µs.

| target | src density | touched cells | exact + full consume | exact + cell-seen bitmap | u16 histogram pre-pass | **exact + target mask** |
|---|---|---|---|---|---|---|
| uniform | 100 % | 4096 | 325 | 386 | 283 | **210** |
| uniform | 10 % | 3279 | 137 | 65 | 97 | **37** |
| uniform | 1 % | 610 | 123 | 15.2 | 22.4 | **8.0** |
| uniform | 0.1 % | 66 | 119 | 6.9 | 14.4 | **6.3** |
| hotspot | 10 % | 617 | 181 | 54 | 92 | **41** |
| zipf | 10 % | 36 | 136 | **24** | 59 | 30 |
| many-to-one | 100 % | 16 | 241 | **156** | 299 | 274 |
| many-to-one | 10 % | 16 | 185 | **40** | 84 | 42 |

- **The u16 histogram pre-pass is KILLED (F3).** It is slower than a cell bitmap built during the same pass in 29 of 30 cases, by up to 2×. The one exception is uniform 100 %. There the histogram beat the cell bitmap (283 vs 386 µs), but the exact target mask beat it (210 µs). A second pass to count contributions cannot be repaid by any work it lets the consumer skip.
- The coarse target-cell bitmap (`target >> 4`, one OR per contribution) does pay, up to ~17× over "clear and scan everything" at sparse frontiers.
- **But the smallest object that delivers the saving is the EXACT next-frontier mask, written in the same pass.** Here K(u) ≥ 1 for every live u, so `K_next > 0` iff `v` was touched. That mask is the result the next round needs anyway, and walking its set bits is also the schedule for clearing `K_next`. It wins or ties every case except where the touched targets concentrate in a few cells: many-to-one at 100 % (274 vs 156 µs) and zipf at 10 % and 0.1 %. There the cell bitmap's 512 B walk is cheaper than the exact mask's 8 KiB walk.
- **Important:** the saving is not a coarse-granularity effect. It comes from not touching dead target ordinals when clearing and consuming, which is the same subtraction law applied on the target side.

## 11. Bounded extent × mask (A4)

An ordered i32 lane `ts = self / 4`, with a predicate `ts ∈ [a, b)`, `m(self)`, and a u32 payload. Times in µs; full table §A4.

| mask density | extent | extent selves | live in extent | full universe | extent only | mask only | extent + u16 | **extent + u64** |
|---|---|---|---|---|---|---|---|---|
| 100 % | 50 % | 32768 | 32768 | 104 | 29.2 | 73.4 | 25.5 | 25.4 |
| 10 % | 100 % | 52432 | 5259 | 100 | 46.5 | 16.8 | 8.1 | **5.5** |
| 10 % | 50 % | 32768 | 3321 | 101 | 29.1 | 15.7 | 5.1 | **3.0** |
| 1 % | 100 % | 52432 | 529 | 109 | 46.5 | 5.0 | 5.4 | **1.3** |
| 1 % | 10 % | 6552 | 65 | 137 | 9.5 | 8.1 | 1.1 | **0.3** |
| 0.1 % | 100 % | 52432 | 50 | 156 | 76.1 | 7.2 | 8.7 | **1.5** |

**Extent and mask are complementary, and they compose multiplicatively.** "Extent + u64 words" beats both single schedulers whenever both remove work, for example 3.0 µs vs 29.1 (extent only) and 15.7 (mask only).

- **F6 holds where the mask is dense.** At 100 % mask density the extent does all the work, and the mask adds nothing (25.4 vs 29.2 µs).
- When the extent is the whole universe, the mask does all the work.
- Extent + u16 cells is slower than extent + u64 words wherever the mask is sparse (5.4 vs 1.3 µs): the same 4× test count as §5.

## 12. "Known work subtraction" accounting

Every row in `a1`/`a2`/`a4` reports:

- **A1:** `live`, zero/full cells and words, and mask tests issued.
- **A2:** payload loads and touched lines.
- **A4:** universe, extent selves, live-in-extent and closed cells in the extent.

Removed work = universe − remaining, attributed in this order:

1. outside the extent;
2. closed words (A1's `zero_words`);
3. dead bits inside open words.

§7's correlations show runtime following *remaining* work (r ≈ 0.95–0.99), never N, which is constant. One correction to the model: **"closed regions" should be counted in u64 words, not u16 cells.** At 1 % uniform density, 3488 of 4096 cells are zero but only 542 of 1024 words are. The u16 view finds more closed regions, yet still loses (§5), because a closed cell is cheaper to skip as part of a word than as a cell of its own.

## 13. Cache and SIMD geometry

| lane | bytes per 16 selves | lines per 16 selves | selves per 64-byte line |
|---|---|---|---|
| u8 | 16 | ¼ | 64 |
| u16 | 32 | ½ | 32 |
| u32 | **64** | **1** (one AVX-512 register) | 16 |
| u64 | 128 | 2 | 8 |

A 16-self cell is line-aligned only for u32 lanes. For u8 lanes, the unit that matches a line is the 64-row u64 word.

Granularity does not change payload traffic. Every set-bit walk, u16 or u64, loads exactly the live rows, and therefore exactly the live lines. The `payload_lines` column is identical across them. So alignment alone buys nothing. Granularity only matters through the dense paths, where the full-cell path for u32 is one 64-byte line.

I did not inspect the generated assembly. Whether the dense loops auto-vectorise is **unverified**; their 4–6× speed-up over set-bit walking (§6) is consistent with vectorisation but does not prove it.

## 14. Falsifier outcomes

| # | falsifier | outcome |
|---|---|---|
| F1 | u16 aperture consistently slower than u64 set-bit visitation | **TRIGGERED.** Visitation: 7× at the floor, 2–5× mid-density, ≈1× at 100 %. Ladder geomean 2.79×. Keep the law, execute in u64 words. |
| F2 | popcount mode switching costs more than it saves | **TRIGGERED for thresholds. NOT triggered for "word or quarter is full".** The `== MAX` dense switch wins 3× on runs at ≤ 6 % cost. The general popcount crossover is a narrow, payload-dependent band (§6). Use one set-bit path plus a full-word dense path. |
| F3 | coarse target histogram does not amortise | **TRIGGERED. Kill `MultiplicityAperture`.** The exact same-pass target mask beats it, and so does the cell bitmap (§10). |
| F4 | ordinal lists win materially and reproducibly | **NOT triggered.** Geomean 1.31× slower. Every apparent win failed to reproduce or was sub-µs. No earned exception. |
| F5 | payload size does not affect the benefit | **PARTLY TRIGGERED.** True at ≥ 10 % density (lines are all live). False at ≤ 1 % (153–352× for 128-byte records). Refinement: count touched lines, not bytes. |
| F6 | bounded extent already removes most of the work | **TRIGGERED only for dense masks.** Otherwise extent and mask compose (§11). |

## 15. Final ruling

**KEEP**

- **The mask is the only carrier.** `[u64]` words are the schedule. Reading 16-bit cells is a free shift of the same bytes when a kernel wants it; there is no new type, no copy and no persisted view.
- **The subtraction law:** extent → zero words → set bits → payload. Measured on every operation tried: plain lane folds, VIA support, VIA multiplicity, CSR K propagation, and extent-plus-mask.
- **A full-word dense path** (`word == u64::MAX` → contiguous fold) inside set-bit walks.
- **The exact target mask produced in the same pass** as the schedule for the next round and for clearing `K_next`.
- **No selection vector**, which reconfirms the D-GATED-GATHER-0 ruling.

**BUILD** (each as its own PR with parity tests; nothing here is built yet)

- `mask_gather_u32` gated by a mask, walking u64 words and set bits. This is the D-GATED-GATHER-0 build, now with stronger support: VIA gating wins here too.
- A full-word dense fast path in the fold kernels' u64 walk (`group_walk`, `masked_sum_*`), to be validated first against the ndarray parity suite.
- An opt-in full-quarter (16-run) dense check, **only** if a workload with 16-aligned runs is measured to need it. On scattered data it costs 1.2–1.8×.
- For multi-hop K propagation: produce the next frontier mask in the same pass, and clear `K_next` by walking it. Not yet measured through mask-risc; this lab hand-rolled it.

**KILL**

- The u16 cell as the primary scheduling unit.
- `adapt16`, the per-cell skip/dense/set-bit policy (geomean 2.07×).
- Popcount-threshold mode switching.
- The `[u16; 4096]` coarse multiplicity histogram (`MultiplicityAperture`).
- Any new carrier type (`ApertureMask`, `SparseMask`, `SelectionVector`) and any new V4 opcode (`APERTURE`, `GATED_LOAD16`, `SPARSE_VIA`, `APERTURE_SUM`).

## 16. The eight questions

1. **Is `[u16; 4096]` a useful zero-copy aperture view of the 64K support mask?** It is zero-copy: a shift of the same 8 KiB, endianness-proof, and verified. As a *schedule* it is not useful. It is the wrong granularity to iterate, and only full 16-run detection uses it.
2. **Is 16-self granularity measurably better than 64-bit word scheduling?** No. It is 1.6–2.8× slower on geometric mean. It wins only on data with full 16-runs that do not fill a word (islands16: 0.41 vs 0.80).
3. **Does the benefit increase with payload width?** Only at low density, ≤ 1 %, up to 352×. At ≥ 10 % nearly every line is live and the ratio stays flat. Runtime tracks rows for lanes up to 16 B and lines from 32 B up.
4. **Can `mask(self)` reliably schedule LOAD and VIA without selection vectors?** Yes. The u64 gate wins or ties every reproducible case except one family: support at 100 % source density into concentrated targets (hotspot, zipf, many-to-one), where a branch-free full scan is 15–20 % faster. Nothing there is skippable, so this is not a selection-vector case either.
5. **Are bounded extent and aperture complementary?** Yes, and they multiply. The exception is a dense mask, where the extent alone suffices.
6. **Does a coarse target histogram help K / multiplicity propagation?** No; it is killed. What helps is the exact next-frontier mask written in the same pass, which removes the dense clearing and scanning (up to ~20×).
7. **Is any new carrier justified?** No.
8. **Is any new V4 opcode justified?** No. Every route here is a lowering of `LOAD` / `KEEP` / `VIA` / `SUM` / `GROUP_SUM`. The one real gap is executor-level: `Gather` has no `under`. That is a missing operand on an existing op, not a new opcode.

## 17. Limits of this lab

- Single machine, single thread, release build, hand-rolled routes.
- Only the "native count" in A1 runs through mask-risc. The rest are transcriptions of the production loop shape.
- N = 65 536 fits a u16 self. Larger universes were not tested, so the 4×-test argument against u16 cells is measured only here, though it can only grow with N.
- Sub-µs differences sit at the timer floor and are not interpreted.
- No assembly inspection (§13).
