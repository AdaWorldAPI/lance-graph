# 2026-11-08 — Virtual CSR over the DR cube; aggregation as BF16 GEMM (tile, ndarray CBLAS)

**Status:** MEASURED (lab). This **fails the ms gate**: every aggregation row
is in milliseconds. Crate: `crates/dr-isa-stream-lab` (`vcsr.rs`, `spmm.rs`,
`bin/bench_vcsr.rs`). Stacked on the DR × ISA stream lab entry of the same
day. Depends on AdaWorldAPI/ndarray#414: the strided GEMM entry.

## What it is

**A virtual CSR.** Adjacency lives in the lanes' own 256 × 256 cube
coordinates.

- Each row tile lists its nonempty column tiles.
- Each (row tile, column tile) pair is one 256 × 256 bitmask block.
- There is no `row_ptr` or `col_idx` array. Row `i`'s neighbours are the set
  bits of block row `slot(i)` across the blocks of `tile(i)`.
- `ExplicitCsr` is the materialised form of the same edges. It is only an
  oracle, and the row-by-row test shows the two agree.

**Neighbour aggregation (plus-times).** `Y[p][i] = Σ_{j ∈ row(i)} lane_p[j]`
for 16 bound lanes, computed per block as BF16 GEMMs:

- **ndarray tile GEMM** (AMX TDPBF16PS on this host). A is read in place from
  the slab with stride = pitch. B is staged per 16-row group in VNNI layout,
  and empty groups are skipped.
- **ndarray's own** `backend::cblas_gemm_bf16bf16f32`: the CBLAS-compatible
  entry (AMX, then AVX-512 BF16, then scalar), with no external library. It
  takes dense `&[u16]` and no `lda`, so A is staged; one call per block. (An
  earlier revision of this lab used upstream `mkl_rt` instead. That arm was
  removed; the comparator is ndarray's own GEMM.)
- **Staged A**, for comparison.

All three are exact against the CSR oracle at n = 256, 513 (a partial last
tile) and 4096 for the tile paths. The tile paths are also exact at 65,536,
and ndarray's CBLAS entry is exact at 513 and 65,536.

**Not claimed:** blasgraph's `CsrStorage` holds `BitVec` values under HDR
semirings (XorBundle, HammingMin, …). A floating-point GEMM does not realise
those, and they were not wired: depending on `lance-graph` pulls in DataFusion
and Lance unconditionally.

## Measured

See `crates/dr-isa-stream-lab/BENCH.md`, which has every arm of both labs
in one table, from one run. For the graph rows, ndarray's CBLAS entry costs
581 ms, 4.7 s and 51 ms on the three graphs, because it repacks both
operands on every call (see BENCH.md, Reading).

## What the numbers say

1. **The cost of a block GEMM is set by block area, not by edges.** Each block
   costs 16 × 256 × 256 MACs regardless of content.
   - At ~101 edges per block (0.15% fill), the GEMM loses 2× to a scalar CSR
     loop over the same lanes.
   - At 25% fill (community structure), it wins 6.4×.
   - This is the same crossover the WHERE-mask GEMM showed from the other
     side: there B was ~50% dense, and the GEMM was the right tool.
2. **The block layout is only as sparse as the local order is local.** The
   same degree with one long-range edge per identity grows from 5,172 to
   41,916 blocks (8.1×) and 330 MB of bitmask. A virtual CSR therefore wants a
   locality-preserving G2L order for graph workloads. Nothing in DR provides
   that order yet; it is a property of how local ordinals are assigned.
3. **Reading A in place does not matter here.** Direct and staged A are within
   5–10%: per call, B dominates, not A.
4. **Per call, the community case costs ~3 µs** against the ~0.45 µs measured
   for a bare K = 256 GEMM. The likely cause is restaging B for every call
   (1,024 VNNI writes plus clears), though the adjacency is constant per graph
   and could be packed once. **Not measured separately.**
5. **ndarray's CBLAS entry is 3–45× slower than the strided tile path.** It
   repacks A and B on every call: a prepacked or strided entry is the gap.

## Production crate: not now (DECISION)

**DECISION:** no production crate for the virtual CSR or the graph
aggregation.

**BASIS (measured, BENCH.md):**

- Every graph-aggregation arm misses the gate, and the best one loses to a
  scalar CSR loop on the sparse graph.
- Even the dense community case is bounded below by its own work:
  16 × 65,536 × 256 MACs ≈ 268M. At the measured AMX rate (~43 ns per
  8,192-MAC block) that is ≈ 1.4 ms before any staging. So the batch cannot
  reach µs at this size, whichever arm runs it.
- The one arm that meets the gate is `SUM(lane) WHERE mask` over the whole
  universe: 172 µs per 16 lanes × 16 masks with A in place. That is an
  aggregation terminal over masks. If it is promoted, its home is the
  Quack / mask-risc terminal set, behind DT64 binding, not a new crate.

**SCOPE:** `crates/dr-isa-stream-lab` stays a lab.

**REVISIT WHEN** one of these is measured:

- each block's B packed once per graph version;
- a per-block dense/sparse choice;
- a locality-preserving G2L order;
- a prepacked or strided entry in ndarray's CBLAS GEMM;

and a graph arm then beats the scalar CSR loop on a sparse graph, or the
WHERE terminal is proposed for Quack.

## Open

- Pack each block's B once per graph version and re-measure; this would
  confirm or refute finding 4.
- A locality-preserving G2L order (for example a bandwidth-reducing order) and
  its effect on the block count.
- A sparse-block realization (CSR-in-block or mask-fold) for low-fill blocks,
  with the crossover chosen per block.
- The blasgraph HDR-semiring adapter: a named seam, not wired.
- Every number above misses the ms gate. The work is real dense block math,
  but the gate is not met and this is not a pass.
