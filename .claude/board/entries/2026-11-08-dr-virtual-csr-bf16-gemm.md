# 2026-11-08 — Virtual CSR over the DR cube; aggregation as BF16 GEMM (tile, MKL)

**Status:** MEASURED (lab). This **fails the ms gate**: every aggregation row
is in milliseconds. Crate: `crates/dr-isa-stream-lab` (`vcsr.rs`, `spmm.rs`,
`bin/bench_vcsr.rs`). Stacked on the DR × ISA stream lab entry of the same
day. Depends on AdaWorldAPI/ndarray#414: the strided GEMM entry, plus the
`BF16::slice_from_bits` view that lets MKL read the `u16` slab in place.

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
- **MKL** `cblas_gemm_bf16bf16f32`. One call per block, M=16, N=256, K=256.
  A is a strided `BF16` view of the same slab with `lda` = pitch, so nothing
  is copied. MKL comes from the PyPI wheel, run single-threaded.
- **Staged A**, for comparison.

All three are exact against the CSR oracle at n = 256, 513 (a partial last
tile) and 4096 for the tile paths. The tile paths are also exact at 65,536,
and MKL is exact at 513 and 65,536.

**Not claimed:** blasgraph's `CsrStorage` holds `BitVec` values under HDR
semirings (XorBundle, HammingMin, …). A floating-point GEMM does not realise
those, and they were not wired: depending on `lance-graph` pulls in DataFusion
and Lance unconditionally.

## Measured

AMX host, n = 65,536, 16 lanes. Every timed run was oracle-checked.

| Graph | nnz | Blocks | Tile, A in place | Tile, A staged | MKL, A in place | Scalar CSR |
|---|---|---|---|---|---|---|
| Local, degree 8, 1/16 long-range | 522K | 5,172 (7.9%) | 30.4 ms | 31.3 ms | 73.0 ms | 15.7 ms |
| Local, degree 8, all long-range | 523K | 41,916 (64%) | 105 ms | 116 ms | 503 ms | 12.3 ms |
| Community, degree 64 in own tile | 4.19M | 256 (0.4%) | **12.1 ms** | 11.8 ms | 24.9 ms | 77.1 ms |

Cold build of the virtual CSR: 20 ms, 158 ms and 71 ms respectively.

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
5. **MKL, single-threaded, is 2–5× slower than the tile path** on these
   16-row problems.

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
