# 2026-11-08 — DR G2L × ISA stream lab: one lane binding, two computation geometries

**Status:** MEASURED (lab, not production admission). Crate:
`crates/dr-isa-stream-lab`. Depends on AdaWorldAPI/ndarray#414
(`bf16_tile_gemm_16x16_packed_strided`).

**Question.** Can a source-witnessed DT64 lane binding be consumed by both an
ordered-cube fold and an AMX BF16 GEMM without a new canonical value layout,
and what is the minimum physical staging?

**Answer.** Yes. Source data is read in place (direct stream). The selection
needs one bounded staging step.

- A (the source's lanes) is loaded straight into AMX tiles from the slab when
  the demanded lanes sit in one allocation, at consecutive rows, with a
  uniform pitch padded to 32.
- B is the WHERE result itself, a 0/1 matrix, staged once per mask set.
- No DR or DT64 change was needed.
- One ndarray API gap blocked the direct read: the safe GEMM entry hardcoded
  A's row stride to K. ndarray#414 closes it.

## Shape

`C[p][m] = Σᵢ lane_p[i] · mask_m(i)`, i.e. every `SUM(lane p) WHERE mask m`
in one GEMM: M = 16 demanded lanes, N = 16 WHERE masks, K = the padded
population. ARM A computes the same 16×16 answer as a positional fold over
the 256×256 cube (`tile(i) = i >> 8`, `slot(i) = i & 255`). Both arms take one
`LaneBinding` and match an oracle computed from the fixture alone.

## Contract map

1. OGAR / ClassView. A 32-field view (`bind.rs`).
2. DT64. `BindStore::register` / `route` / `bind_lanes` give a `LaneBinding`
   (`lance-graph-dt64/src/lane.rs:28-140`, `:352`). It holds addresses only:
   `LaneAddr { space: CoordSpace, lane, scope }`.
3. Source. `LaneSource::borrow` returns the source's own `Lane<'a>`. Here that
   is `SlabLane { slab, row, pitch, n }`, so **the physical geometry rides on
   the source's lane type, not on DR**. `FrameStore` is owned by the data.
4. Consumer. ARM B checks in safe code that the 16 borrowed lanes share one
   allocation (`ptr::eq`), have equal pitch, sit at consecutive rows, and have
   a pitch of at least `round32(n)`. It then calls the strided GEMM with
   stride = pitch.
5. ndarray. `TILELOADD` takes the stride as an operand (`amx_matmul.rs`
   `tile_load`). Before #414, `amx_path` passed `k*2`.

**First missing guarantee.** The trait cannot prove that a source's
`SlabLane` geometry describes its real memory. That is the same obligation
`LaneSource::frames` already documents, and it stays the implementer's.

## Result matrix

| Claim | State |
|---|---|
| Both arms exact vs oracle at n = 256, 513, 4096, 65536 | proven in lab |
| One binding, 16 lane resolutions per arm, independent of n | proven in lab |
| Direct A (in place) on AMX, zero A bytes copied | proven in lab (AMX TDPBF16PS) |
| Lane in another CoordSpace refused at bind | existing DT64, exercised |
| …and read positionally it gives a wrong answer | proven (oracle catches it) |
| Stale frame refuses both arms; re-bind accepted | existing DT64, exercised |
| Masks / selection in another space refused | lab guard |
| Scattered lanes, permuted rows, unpadded pitch, < 16 lanes: direct refused, staged correct | lab guards |
| bf16 exactness: ints in [-256, 256] exact, 257 refused; decimals approximate, error ≤ 2⁻⁸ | lab |
| MKL comparator | **not executed** (MKL not installed) |
| Non-AMX tier | CI runner (tier printed in the log) |

Disable runs: removing each of the six lab guards fails exactly the test
that names it.

## Costs

Granite Rapids VM, 4 vCPU, AMX TDPBF16PS, rustc 1.99.0, lance-graph `7c301f5`,
ndarray `9a632f3`. Warm medians of 41 runs. Every timed result is checked
against the oracle.

n = 65,536, 16 lanes, 16 masks (524,717 selected bits):

| Stage | Cost |
|---|---|
| bind (cold, once) | 1.0 µs |
| ARM A: 16 WHERE counts (popcount terminal) | 2.2 µs |
| stage selection B (cold, once per mask set; 2 MiB) | 689 µs |
| ARM B whole lane, A in place | **152 µs** |
| ARM B whole lane, A staged (copy + zero-fill) | 362 µs |
| reference: per-32-block GEMM calls, B restaged per block | 2,212 µs |
| reference: ARM A per-bit sum fold (scalar) | 13,130 µs |

At 32 lanes, B is reused across both row tiles: 311 µs direct versus 726 µs
staged.

**Per-call overhead, measured.** AMX costs about 43 ns per 16×16×32 block in
steady state, plus about 105 ns fixed per call (tile config and release). A
GEMM of K = 32 per call therefore never amortizes. A consumer must make one
call per row tile with K = the population.

The two reference rows are not gates. They show where the cost goes when the
work leaves the fold: per-block calls, and a scalar per-bit gather for a sum
that is GEMM-shaped.

## Smallest justified interface change

- ndarray: `bf16_tile_gemm_16x16_packed_strided` (#414).
- DR and DT64: none.

A source that wants direct AMX consumption exposes a lane type carrying
`(allocation, row, pitch)`. DR keeps describing identity and row order only.

## Open

- B staging costs 689 µs cold. A sequential bit-to-VNNI expansion should be
  faster; not attempted.
- Gotcha 14: AMX tile state can corrupt on an oversubscribed VM. The numbers
  above were verified on every run, but this is not certification on
  dedicated silicon.
- MKL comparison not run.
- 32×128 and 64×64 layouts not compared.

## Handoff for report-rs #33

- **Available as lab primitives:**
  - the strided BF16 GEMM (once #414 merges);
  - the `SlabLane` geometry pattern;
  - a SUM-WHERE-for-16-masks GEMM recipe.
- **Lab-only:** everything in `dr-isa-stream-lab`; nothing is a production
  path.
- **Unchanged for #33:** its workbook gates. bf16 is a numeric realization;
  an exact decimal must not be treated as bf16.
