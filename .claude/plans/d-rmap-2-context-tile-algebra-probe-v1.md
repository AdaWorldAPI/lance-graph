# D-RMAP-2: context declaration reuse and a GraphBLAS/BF16 tile probe

**Status: PROPOSAL / PREREGISTERED TEST DESIGN, 2026-10-19.** No runtime, ABI, dependency, opcode, CE64 interpretation or performance result is added. Extends [D-RMAP-1](d-rmap-implicit-relationship-register-v1.md) and its controlling [D-RMAP-1A ownership correction](d-rmap-separate-tables-thought-palettes-v1.md).


> **PRIOR-ART CORRECTION (2026-10-19, precedence over earlier wording):** The operator already reverse-engineered BlasGraph, MKL-like backends, BF16 16x16, AMX, VNNI and mask-GEMM interop in AdaWorldAPI/ndarray and AdaWorldAPI/lance-graph. **Do not start a new GraphBLAS/MKL implementation, import production FFI or redo settled kernel comparisons.** The unproven subject is the separate RelationshipMap tables, zero-copy index binding and thought-selected map palettes. Use existing native kernels and probes as references; any external GraphBLAS implementation is optional independent oracle only.

> **Partial answer (2026-10-19):** the first "updated test priority" item (bind a focus, compare direct indexed seek with a repeated local Boolean fold, focus churn, external-neighbour escapes, 1/8/64 thoughts) is measured in D-RMAP-1A §8 for in-memory warm fixtures. The GraphBLAS and BF16/INT8 arms here remain unrun.

## 0. Existing implementation and measurements: mandatory reuse

| Precedent | Source | What is already present |
| --- | --- | --- |
| Native BF16 16x16 tile | [ndarray BF16 tile GEMM](https://github.com/AdaWorldAPI/ndarray/blob/main/src/hpc/bf16_tile_gemm.rs) | AMX TDPBF16PS, AVX-512 BF16 and decoded FMA fallback |
| Native INT8/AMX and dispatch | [AMX matmul](https://github.com/AdaWorldAPI/ndarray/blob/main/src/hpc/amx_matmul.rs), [INT8 tile GEMM](https://github.com/AdaWorldAPI/ndarray/blob/main/src/hpc/int8_tile_gemm.rs), [AMX ISA primitives](https://github.com/AdaWorldAPI/ndarray/blob/main/src/hpc/amx_ops.rs) | Correctness-checked AMX/TILE paths and runtime gate |
| Pure Rust MKL-like backend | [ndarray MKL-shaped backend](https://github.com/AdaWorldAPI/ndarray/blob/main/src/backend/mkl.rs) | Already reverse-engineered CBLAS-style shape; do not confuse with a missing native kernel |
| Existing Boolean/tile experiment | [D-GTM evidence](https://github.com/AdaWorldAPI/ndarray/blob/main/.claude/plans/gemm-ternlog-mask-consolidation-v1.md) and [hex-trie GEMM probe](https://github.com/AdaWorldAPI/ndarray/blob/main/examples/hex_trie_vs_gemm_probe.rs) | Zero-heap-allocation mask path; crucial counterexamples and comparison limitations |
| Existing BlasGraph | [BlasGraph](../../crates/lance-graph/src/graph/blasgraph/mod.rs), [semiring](../../crates/lance-graph/src/graph/blasgraph/semiring.rs), [matrix](../../crates/lance-graph/src/graph/blasgraph/matrix.rs) | Shipped GraphBLAS-inspired CSR/CSC matrix operations; Boolean AND/OR and other semirings over HdrScalar/BitVec |
| Existing 256 x 256 tile probes | [H6b forms](../../crates/lance-graph-mask-risc/examples/holo_tile_forms_probe.rs), [H6c crossover](../../crates/lance-graph-mask-risc/examples/holo_tile_crossover_probe.rs) | Dense 8-KiB Boolean tile, Morton interval, product/hybrid forms and independent row oracle |

**Do not misuse D-GTM-0j:** it recorded mask against *dense f32* GEMM numbers of 745.8x at 0.02% and 297x at 100% density. Section 12.5 explicitly states these headline numbers are NOT evidence of mask superiority over a properly indexed sparse baseline: CSR SpMV was missing, and the f32 Boolean matrix was an unfair comparison. The measured positive was 0 bytes allocated in the hot mask step and a strong *prefix-local versus random* locality difference.

**Especially relevant to RelationshipMaps:** D-GTM-0l (section 13, the real long-range edge corpus) found that physical-address prefix tracts cannot economically represent diffuse global edges. At prefix width k=2, 257 tracts for 427 edges, only 1.66x compression; at k=3, 396 tracts, 1.08x. It does NOT disprove semantically minted HHTL locality, but it gives a concrete reason for a separately stored, globally indexed RelationshipMap that can be borrowed into a local focus.

**No false zero-copy:** BlasGraph GrBMatrix currently *owns* CSR and optional CSC and uses 16K-bit BitVec elements; it is not the proposed 256x256 bit-packed edge-presence tile. Its ndarray_bridge From reference implementation copies a 256-word array on the stack, which is no heap allocation but also not a literally borrowed buffer. Matrix transpose can clone index vectors. The proposed separate MapStore binder must prove real borrowed buffer/view lifetimes and charge cache/index materialization.

**Updated test priority:** first bind 1 MapId through separate RelationshipMap + FocusBinding + ThoughtMapPaletteBinding tables into existing mask-risc/Quack operands and compare direct indexed seek with repeated local Boolean fold. Next test 64..256 global maps and 1/8/64 overlapping thought palettes, focus churn, ACL/version witnesses, external-neighbor escapes and exact set/bag semantics. Reuse existing BlasGraph Boolean and ndarray BF16/INT8 kernels only for demonstrably eligible workload comparisons and measure missing cells, not settled kernel behavior. RelAlg remains logical; CE64 remains opaque.

## 1. Which existing forks actually help?

**AdaWorldAPI/lance-context (fork of lance-format/lance-context):**
- In crates/lance-context-core/src/generic_store.rs, GenericStore already accepts caller-declared SchemaSpec, persists it as schema metadata, refuses mismatched declarations on reopen, and reuses StorageBase for Lance persistence, versioning, projection, MemWAL and index maintenance.
- crates/lance-context-api/src/schema_spec.rs contains the declarative SchemaSpec / ColumnSpec vocabulary. This is a plausible precedent or candidate cold table owner for separately declared RelationshipMap, MapPalette and ThoughtMapPaletteBinding tables, not an existing thought-ontology context model.
- crates/lance-context-core/src/record.rs currently represents ContextRecord relationships as Vec<Relationship> with string target_id/relation and optional f32 weight, plus JSON-capable metadata. This is a boundary/ingestion view, NOT the hot numeric RelationMap. GenericStore's get/list row wrappers do not by themselves prove a zero-copy indexed native view. Prove native Arrow/Lance borrow semantics before claiming this reuse saves copies.

**AdaWorldAPI/petgraph:**
- The petgraph::visit traits GraphBase, IntoNeighbors, Visitable and IntoEdges offer a low-cost read-only facade for a borrowed, focus-bound relationship map, allowing existing BFS and graph oracle algorithms to operate without copying global facts into petgraph's GraphMap, Csr or MatrixGraph.
- MatrixGraph and GraphMap may serve as small independent oracles; their materialized adjacency representations must not replace canonical RelationshipMap tables. Petgraph neither declares typed thought context nor resolves palette/version/ACL witnesses itself.

**GraphBLAS and Intel oneMKL:**
- GraphBLAS provides masked, semiring-aware graph linear algebra, including Boolean OR/AND matrix-vector and matrix-matrix products. It is an appropriate exact Boolean reference.
- MKL BF16 x BF16 -> FP32 GEMM performs ARITHMETIC matrix multiplication, not the Boolean semiring. For binary 0/1 adjacency and k<=256, sums of products remain exact integers under FP32 accumulation, so (A @ B > 0) equals two-hop Boolean reachability. Threshold before repeating the hop; do not generalize to weighted NARS, independent evidence, negation, bag multiplicities, or unbounded recursive arithmetic products.
- Keep external C-based GraphBLAS/MKL implementations outside the production repo per no-C policy. If used, they are stand-alone independent reference/benchmark processes and do not add production dependencies.

## 2. Explicit storage, not false zero-copy accounting

| Representation | 16 x 16 | 256 x 256 per map | 64 maps | 256 maps |
| --- | ---: | ---: | ---: | ---: |
| Boolean bit-packed | 32 B | 8 KiB | 512 KiB | 2 MiB |
| Dense BF16 | 512 B | 128 KiB | 8 MiB | 32 MiB |
| Dense FP32 output | 1 KiB | 256 KiB | 16 MiB | 64 MiB |

A bit-packed 16 x 16 tile is 256 BITS, not 256 bytes. Staging it into BF16 uses 16x the operand bytes, before cache, palette, sorted indexes, conversion/output and version witnesses. If storage is already BF16, that conversion cost may be absent; report those cases separately.

The same thought-selected global MapIds and same focus are supplied to every candidate. The physical kernel sees an admitted tile/span, not DTO objects and not CE64 map fields. A map outside the local focus must preserve global external-neighbor IDs for the next frontier.

## 3. The semiring law to test

For relation A of source-to-middle and relation B of middle-to-target, exact Boolean composition is:

    C[i,j] = OR_k( A[i,k] AND B[k,j] )

One-hop frontier propagation:

    next[j] = OR_i( frontier[i] AND A[i,j] )

GraphBLAS computes these in a Boolean semiring. MKL BF16 arithmetic computes:

    counts[i,j] = SUM_k( bf16(A[i,k]) * bf16(B[k,j]) )
    C[i,j] = counts[i,j] > 0

That is a valid transform **only** under the stated Boolean binary-input and exact-count bounds. Arithmetic GEMM does not establish class inheritance, causal implication, semantic equivalence or provenance. A two-hop derived existence result is not a canonical fact until the typed rule permits the inference.

## 4. Pre-registered arms and workload matrix

1. **Direct global ordered map:** sealed RelationOrder/TrieMap seeks, no local dense tile, including index amortization.
2. **Existing ndarray/Mask-RISC Boolean bitsets:** 256-bit row masks, frontier row-OR, 16x16 bit tiles, scalar reference; direct co-addressed H14c fast path remains its own lawful arm.
3. **GraphBLAS:** Boolean lor_land masked mxv/mxm (independent oracle plus measured external implementation).
4. **oneMKL BF16:** binary adjacency staged into dense BF16, BF16xBF16->FP32 GEMM, threshold result, separately priced staging and kernel time.
5. **Optional oneMKL INT8xINT8->INT32:** arithmetic exact-control arm, if supported by hardware/library, never mistaken for Boolean semiring.

Matrix: focus sizes 16/64/256; map-bank sizes 1/8/64/128/256; 1/8/64 thoughts; relation densities 0.1/1/10/50/100%; clustered/random/skewed-hub topology; 1-hop/2-hop/multi-map 2-hop/bounded Boolean closure/64-query reuse; focus churn by 1/8/16/128/256 identities; cold/warm/hot index/cache. Reduced budget cells must be chosen before seeing results.

Report: total bound cost, sorted-index creation/maintenance, native scan time, rebind, page faults, bytes touched, BF16 staging/threshold, any cold device or CPU startup, density/fanout, ISA/AMX availability, and total time per logical output. Full end-to-end reuse break-even k is the verdict, not headline FLOPS.

## 5. Falsifiers

- Exact Boolean matrix-vector/matrix-matrix answers match independent bit enumeration for all eligible cells. Any single Boolean wrong result kills that BF16 lowering case.
- Cyclic Datalog triangle, walks-versus-distinct, bags, unknown facts, provenance and evidence roots do NOT get silently reduced to Boolean existence. Separate typed semantics or explicit refusal is required.
- Different thoughts may select overlapping but distinct MapPalettes, with map IDs, generations and frame scope validated. No per-thought copy of all maps; no CE64 bit repurposing.
- Changing map generation, focus-slot incarnation, ACL/tenant scope, Lance seal, compaction or thought slot refuses stale cached tiles.
- For sparse 1-hop, bitset or direct indexed seeks may defeat GEMM. For repeated dense 2-hop/all-pairs, tile GEMM may have a valid economic region. A loss in the full-cost test remains a loss.
- Do not add an MKL/GraphBLAS C dependency, hot JSON representation, new runtime/AST, unauthorized gather or hidden cross-coordinate index hop.

## 5a. First measured cell: Boolean composition and triangles on one 256 × 256 focus (2026-10-20)

> **Status: MEASURED, probe only, one host.** Code: `crates/mork-pathmap-probe/src/tile_gemm.rs`, `tests/tile_gemm.rs` (4 tests; 5 guards disable-verified), `examples/tile_gemm_cost.rs` (`cargo run --release --example tile_gemm_cost`). Release, warm, medians of 7, every timed answer checked against a triple-loop oracle. The GEMM arm is `ndarray::simd::bf16_tile_gemm_16x16_packed` through the facade, not a new kernel.

**Tier that ran: the F32x16 FMA polyfill** (`bf16_tile_gemm_tier()`). This host has `avx512_vnni` and `avx512bw` but neither AMX nor `avx512_bf16`. No tile-tier number exists here; the AMX and `VDPBF16PS` cells of §4 remain unmeasured. This is the hardware, not detection (ndarray `.claude/AMX_GOTCHAS.md` Gotcha 4: a wrong `arch_prctl` syscall number once made `amx_available()` false on real AMX hosts). Checked directly on this host: CPUID.7.0 EDX bits 22/24/25 (AMX-BF16/TILE/INT8) = 0, CPUID.7.1 EAX bit 5 (AVX512-BF16) = 0, XCR0 = `0xe7` (XTILECFG/XTILEDATA bits 17/18 clear), and `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` on syscall 158 returns `-95` (EOPNOTSUPP). The AMX status in the gotcha file was measured on an Emerald Rapids VM, which this one is not.

| Map | edges | triangles | bit compose | bit walk counts (popcount) | bit triangles | BF16 stage | BF16 `A·A` (packed / row-major) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 % | 658 | 23 | 3.6 µs | 175 µs | 1.9 µs | 155 µs | 19.6 / 19.4 ms |
| 10 % | 6 529 | 16 617 | 4.5 µs | 190 µs | 13.5 µs | 256 µs | 19.4 / 22.3 ms |
| 50 % | 32 776 | 2 099 676 | 25.2 µs | 192 µs | 71.8 µs | 400 µs | 19.5 / 18.9 ms |
| 1 % + 4 hubs | 2 706 | 20 273 | 2.6 µs | 165 µs | 5.8 µs | 150 µs | 19.7 / 19.4 ms |

- **§3's exactness claim holds as measured.** BF16 walk counts equal the oracle exactly (`assert_eq!`, not a tolerance), and `count > 0` equals Boolean composition, on all four maps.
- **On this host the bit arm wins every cell**, including walk counts. Popcount of `row_i(A) ∧ row_j(Aᵀ)` gives the same counts as the GEMM in about 170–190 µs, against about 19 ms. Staging alone (bits to dense BF16, 16 KiB to 256 KiB) costs more than the whole bit composition.
- **The 19 ms is the polyfill's shape, not GEMM's.** `ndarray::simd_ops::bf16_tile_gemm_16x16` allocates two decode buffers per call, then one strided B-column buffer and one horizontal reduce per output element: about 257 heap allocations per 16 × 16 call, 65 000 per 256 × 256 product. The packed entry on this tier also unpacks and allocates per call. Recorded as `ISS-BF16-POLYFILL-PER-ELEMENT-ALLOC`; the row-major entry is no faster, so the unpack is not the cost.
- **Two conclusions, kept apart.** (1) *Measured:* on this host's F32x16 polyfill tier, the bit arm wins every cell, and BF16 is exact. (2) *Not measured:* nothing here says anything about BF16 or AMX tile hardware performance. The polyfill's cost is dominated by allocation, so it is not even a lower bound on what `TDPBF16PS` or `VDPBF16PS` would cost. No layout decision may cite (1) as evidence about (2).
- **Not measured:** AMX and `VDPBF16PS` hosts; dense relations above one focus (blocked 256 × 256 tiles); weighted or evidence-bearing relations, which §3 excludes anyway.

## 5b. Tile GEMM is a consumer of prepared relationships, not a gather solution (operator, 2026-10-20)

> **Status: RULING-SHAPED CLARIFICATION, recorded as provenance.** Shared with D-RST (`relstore-quack-petgraph-traversal-v1.md` §5b), which carries the cost model and the falsifier F8.

Tile GEMM speeds up arithmetic once relationship operands are in a compatible dense layout. It does not resolve global identities, align coordinate domains, scatter, gather or traverse sparse data. Those happen before the kernel and are paid there. A kernel loads its operands from memory, but only from tiles that something already built.

**Three costs, always reported apart:** preparation (identity resolution, coordinate alignment, index traversal, gathering, packing, tile construction, sealing), execution (bit operations, SIMD, BF16/AMX GEMM on prepared operands) and maintenance (version checks, invalidation, incremental updates, repacking, rebuilds). A tile path wins only if `T_prepare + N·T_exec + M·T_maintain` falls below `N·T_traverse` for the measured reuse count `N` and invalidation count `M`. Break-even counts come from measured values, never from an assumed kernel speedup, and a warm kernel is never compared with a cold traversal.

**What §5a measured, in those terms.**
- *Execution:* bit composition, popcount counts, triangles and the BF16 product, each on operands already in one 256 × 256 focus.
- *Part of preparation:* the "BF16 stage" column (bits to dense BF16), and it alone exceeds the whole bit composition.
- *Not measured:* the preparation that obtains the focus (identity resolution and slot assignment, measured separately in D-RMAP-1A §8), maintenance (no invalidation was exercised), and any reuse count. So §5a ranks execution paths on prepared operands; it says nothing yet about an end-to-end relational operation.

**Boolean and arithmetic answers are different questions.** Reachability needs existence (`And`/`Or`); the BF16 product returns path multiplicities. §5a checked that `count > 0` equals Boolean composition. Using a count as existence is an explicit derivation step, never an implicit one. Exactness: the operands here are 0/1 and the accumulator is F32, so one product is exact up to 2^24. A count fed back as a BF16 operand (a further hop, or a weighted relation) is exact only up to 256, the BF16 significand bound. Such a path checks its counts against that bound and refuses beyond it.

**No new production kernel is added for this hypothesis.** The probe uses `ndarray::simd::bf16_tile_gemm_16x16{,_packed}` through the facade and the existing mask-risc/ndarray bit algebra. The Boolean tile already exists as a mask-risc presence plane and as the 256 × 256 tile in `rmap.rs`. Sparse data is never forced into tiles in order to use GEMM.

## 6. Recommendation

Use **lance-context GenericStore/SchemaSpec only as a candidate cold durable declaration precedent**, **petgraph::visit as an optional read-only compatibility/oracle facade**, **GraphBLAS as the Boolean algebra oracle**, and **MKL BF16 GEMM as an explicitly falsifiable physical tile accelerator**. The map bank remains in independent typed tables; OGAR provides vocabulary; BIND selects and borrows; RelAlg describes; existing Quack/R2IL/Mask-RISC remains the execution family; CE64 is opaque.

No physical kernel or storage layout is adopted by this plan. A later operator-approved implementation requires demonstrated cost advantage and exact semantics.
