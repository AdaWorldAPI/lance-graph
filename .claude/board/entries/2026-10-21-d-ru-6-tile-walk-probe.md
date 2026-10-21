# 2026-10-21 — D-RU-6-LAB-1: graph_walking over co-addressed RMAP tiles, measured against walk_mask

**Status:** MEASURED, lab only. D-RU-6 stays **Queued/Lab**: ADMISSION B is not demonstrated for the binding path, and passing tests is not an admission. Plan: `.claude/plans/d-relalg-unified-v1.md` §8, §10, §14 (U6). References: D-RMAP-2 §5b (`d-rmap-2-context-tile-algebra-probe-v1.md`, tile kernels consume prepared relationships), #1642 (the `walk_mask` tuple oracle and fixtures).

## Benchmark requirement (operator, 2026-10-21)

Every D-RU-6 comparison reports these separately, never folded into one number:
1. tuple materialization (decoding to the canonical `Walk` / `NoWalk` sets);
2. count-only evaluation (no tuple built);
3. memory growth (resident, per engine, in a fresh process);
4. address binding: attestation, witness validation, identity resolution and focus binding, projection;
5. value-derived reads, by origin.

Correctness is the full tuple set against crepe and ascent wherever they run; Boolean reachability alone does not count. Above the oracle size the result is marked UNCHECKED and compared engine against engine only.

## What landed

`crates/crepe-plan-probe/examples/walk_tile.rs`, a separate example; the baseline `walk_mask.rs` is compiled in as a module, and it only gains `pub(crate)` on the reused items.
- **Binding.** The node ids form a frozen sorted codebook, with one `rmap::Focus` per 256 ids. The witness is `rmap::Witness`: facts digest, codebook seal, focus generation, scope.
- **Projection.** One block matrix of 256 × 256 tiles per edge length, `rmap::tile_addr` layout.
- **Recursion.** Semi-naive over length planes, `new &= !walk; walk |= new`. The hop is a Boolean tile product in predicated form: per coordinate `y`, `C |= broadcast(row y of A) & spread(bit y of each row of B)`, as one mask-risc `Ternlog` (0xF8) program. Every address is a loop coordinate.
- **Paging.** More than 256 nodes are paged over several foci; products run over all `(I, K, J)` tile triples, and cross-page edges are counted. A focus limit that the nodes exceed is refused (`IncompleteFocus`), never truncated. A scope that misses reached ids is refused.
- **Guards.** `admit` refuses any program containing `MaskOp::Gather` before execution. A stale witness (focus generation, facts, seal) is refused.

**Tests (23, of which 8 are the baseline's own):**
- tuple-set equality with ascent, crepe and walk_mask on mixed lengths {1,2,3,5,7}, cycles, alternative path lengths, zero-length and over-length edges, ids that are not nodes, 255/256/257/300 nodes across foci, the benchmark generator over two foci, and a sparse graph over three foci;
- a chain that needs ≥ 200 iterations (230 measured);
- a zero-length 200-ring that terminates;
- the tile product against `tile_gemm::BitMat::compose`;
- refusals.

**Disable runs** (each reverted afterwards; all red):

| Disable | Red |
|---|---|
| `admit` lets `Gather` through | the gather test |
| witness check removed | the stale test |
| escape refusal removed | the scope test |
| page cap removed | the cap test |
| length bound made exclusive | 6 tests |
| cross-page products dropped | 2 tests |
| spread reads `y ^ 1` | 8+ tests |
| decode shifts a length | 8+ tests |

CI runs `cargo test --example walk_tile`; the benchmark is not run in CI.

## Measured (release, 4 vCPU, load < 1.3, medians of 7)

| fixture | nodes | foci | mask count-only ms | tile count-only ms | tile/mask | mask + decode ms | tile + decode ms | tile/mask | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| edges(128) | 128 | 1 | 0.43 | 0.72 | 1.7× | 4.14 | 4.54 | 1.1× | equal (ascent, crepe) |
| edges(256) | 256 | 1 | 1.28 | 1.32 | 1.0× | 15.8 | 18.2 | 1.2× | equal |
| edges(512) | 512 | 2 | 3.07 | 8.95 | 2.9× | 68.3 | 74.7 | 1.1× | equal |
| edges(1024) | 1024 | 4 | 12.7 | 72.0 | 5.7× | 277 | 384 | 1.4× | UNCHECKED (tile = mask) |
| chain(230) | 230 | 1 | 7.43 | 27.6 | 3.7× | 16.8 | 33.6 | 2.0× | equal |
| mixed(256) | 256 | 1 | 8.65 | 80.0 | 9.3× | 298 | 568 | 1.9× | equal |

"Tile count-only" includes attestation, validation, binding, projection, recursion and counting. Its phases at edges(1024), in ms: attest 0.84, validate < 0.001, bind 0.80, project 1.74, recurse 67.6, count 0.16, decode 318.

**Memory.** Resident growth (VmHWM in a fresh process, MiB):

| | mask count-only | tile count-only | mask + decode | tile + decode |
|---|---:|---:|---:|---:|
| edges(512) | 3.61 | 0.66 | 13.9 | 10.9 |
| edges(1024) | 14.2 | 2.23 | 50.7 | 44.9 |
| mixed(256) | 0.65 | 1.30 | 57.1 | 57.5 |

At edges(1024), the mask planes take 19.1 MiB (three dense sets of 51 planes) and the tile walk takes 1.28 MiB, because only non-empty tiles exist.

**Value-derived reads at edges(1024):**

| Engine | Phase | Reads |
|---|---|---|
| walk_mask | execution (delta row at an adjacency bit) | 419,230 |
| tile | preparation (codebook lookups, 2 per edge) | 83,846 |
| tile | execution | 0 |
| tile | decoding (codebook decodes, 2 per output tuple) | 4,036,624 |

## Reading

- **Value-derived reads in execution: removed.** The tile recursion reads no position taken from a value, and `Gather` is refused before execution.
- **Speed: lost, everywhere but one cell.** Count-only, the tile path is 1.0× (edges(256), one focus, single edge length) to 9.3× slower (mixed lengths). End to end it is 1.1–2.0× slower; the gap narrows only because decoding (BTreeSet) dominates both engines.
- **Where the time goes.** The predicated product touches 3 × 1,024 words per contributing `y`, whatever the sparsity. walk_mask touches `w` words per set adjacency bit. Recursion is 94% of the tile path's count-only time at edges(1024).
- **Binding costs are small.** Attest, bind and project together are about 3.4 ms at edges(1024), about 4.7% of the count-only time. They are reported, not hidden.
- **Memory is the one gain.** Only non-empty tiles exist, so the count-only resident growth is 6.4× smaller at edges(1024). Decoding erases that.
- **Not measured:** a sparsity-aware co-addressed product (skipping whole 64-row blocks of `B` without addressing by value); AMX or BF16 tile products for this program (D-RMAP-2 §5a: the bit arm beats them on Boolean composition); identity-scale paging (65,536 nodes).

## ADMISSION B classification

| Step | Classification |
|---|---|
| attestation, codebook build | same-table sequential scans, counted |
| slot resolution | one frozen-codebook lookup per edge endpoint, keyed by the edge's own value. This is the shape the ruling admits as representation translation, but the witness is a probe-local digest (`rmap::Witness`), not the attested sorted key the ruling names. **Not demonstrated.** |
| edge tiles | projection of the same table at resolved slots |
| recursion | co-addressed. **Demonstrated** (refusal guard, disable-verified). |
| decoding | frozen-codebook decoding, counted |

**Verdict:** execution is gather-free; the complete preparation and binding path is not shown to satisfy ADMISSION B. D-RU-6 stays Queued/Lab. No production promotion.

## Open

- A contract-level attestation for the codebook (an `OrderedLaneWitness`-shaped seal), before any ADMISSION B claim.
- Whether a co-addressed product can use block sparsity of `B` to close the 1.0–9.3× count-only gap without a value-addressed read.
