# D-RMAP-2: context declaration reuse and a GraphBLAS/BF16 tile probe

**Status: PROPOSAL / PREREGISTERED TEST DESIGN, 2026-10-19.** No runtime, ABI, dependency, opcode, CE64 interpretation or performance result is added. Extends [D-RMAP-1](d-rmap-implicit-relationship-register-v1.md) and its controlling [D-RMAP-1A ownership correction](d-rmap-separate-tables-thought-palettes-v1.md).


> **PRIOR-ART CORRECTION (2026-10-19, precedence over earlier wording):** The operator already reverse-engineered BlasGraph, MKL-like backends, BF16 16x16, AMX, VNNI and mask-GEMM interop in AdaWorldAPI/ndarray and AdaWorldAPI/lance-graph. **Do not start a new GraphBLAS/MKL implementation, import production FFI or redo settled kernel comparisons.** The unproven subject is the separate RelationshipMap tables, zero-copy index binding and thought-selected map palettes. Use existing native kernels and probes as references; any external GraphBLAS implementation is optional independent oracle only.

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

## 6. Recommendation

Use **lance-context GenericStore/SchemaSpec only as a candidate cold durable declaration precedent**, **petgraph::visit as an optional read-only compatibility/oracle facade**, **GraphBLAS as the Boolean algebra oracle**, and **MKL BF16 GEMM as an explicitly falsifiable physical tile accelerator**. The map bank remains in independent typed tables; OGAR provides vocabulary; BIND selects and borrows; RelAlg describes; existing Quack/R2IL/Mask-RISC remains the execution family; CE64 is opaque.

No physical kernel or storage layout is adopted by this plan. A later operator-approved implementation requires demonstrated cost advantage and exact semantics.
