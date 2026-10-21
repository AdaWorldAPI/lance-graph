# 2026-10-21 — D-RU-6-LAB-2: block-sparse tile recursion (arm C) against walk_mask and the dense tile path

**Status:** MEASURED, lab only. D-RU-6 stays **Queued/Lab**. Nothing is admitted. Builds on D-RU-6-LAB-1 (`entries/2026-10-21-d-ru-6-tile-walk-probe.md`, #1647). References: D-RMAP-2 §5b, #1642.

## Question

D-RU-6-LAB-1 found the dense tile product (arm B) 1.0–9.3× slower than walk_mask (arm A) count-only. Does a product that skips empty work, still addressed only by loop coordinates, close that gap anywhere end to end?

## What landed

`crates/crepe-plan-probe/examples/walk_sparse.rs`. It compiles `walk_tile.rs` as a module (which itself compiles `walk_mask.rs`), so all three arms run on the same graphs with the same oracles. `walk_tile.rs` only gains `pub(crate)` on reused items and a rustfmt pass; its 24 tests are unchanged.

Arm C reuses B's binding, witness and projection unchanged, and changes the product:
1. A tile triple runs only when its edge tile and its delta tile both exist. The delta holds only the facts that are new this iteration (semi-naive).
2. Each delta tile carries an occupancy map of `m × m` blocks (`m` ∈ {16, 32, 64, 256}; 64 for the main tables, fixed before any run). An empty block is skipped with one bit test.
3. Inside a block, a row OR runs only where the delta bit `(y, z)` and the precomputed row-occupancy bit of row `y` of the edge tile are both set. Output tiles are allocated on the first OR.
4. There is no broadcast, no spread and no per-coordinate `Ternlog`. mask-risc runs the tile update (`fresh = new & !walk`, `walk |= fresh`), two calls per changed tile. `admit` refuses `Gather` in both programs.
5. Row occupancy of the edge tiles is prepared once, outside the iteration loop. Binding is never redone.

**Address audit (Rust indexing, not only `MaskOp::Gather`).** Every index in the execution path is a loop coordinate: tile triples `(I, K, J)`, block cell `zb * nb + yb`, delta row `z`, inner coordinate `y = w * 64 + shift + b` from the word, shift and bit loops. There is no bit scan (`trailing_zeros`): the bit loop visits every coordinate of a non-empty segment and tests it. An occupancy bit only decides whether work runs.

**What that does not mean.** The set of row ORs C executes is exactly the set a bit scan would find. C differs from A in how a position is obtained, not in which work is done. Whether that satisfies the intent of ADMISSION B is an operator ruling, not something this probe shows.

**Tests (38, of which 24 are B's and A's):** tuple-set equality with ascent, crepe, A and B at every microtile side, on mixed lengths with cycles, alternative path lengths, a zero-length 200-ring with over-length edges, a 300-node chain across two foci (≥ 200 iterations), 255/256/257/300 nodes and a three-focus graph, edges at every 16/32/64 block boundary, and clustered, dense, cyclic and benchmark-generator fixtures; a row-OR bound that holds only for a semi-naive delta; refusals (stale witness, escaping scope, focus cap, gather in either update program); a wrong path length detected by the oracle comparison; segment coverage.

**Disable runs** (committed first, each reverted; all red):

| Disable | Red |
|---|---|
| delta block occupancy never refreshed | 9 tests |
| block skip inverted | 9 tests |
| inner coordinate off by one (`y ^ 1`) | 9 tests |
| row occupancy read from the wrong word | 9 tests |
| length bound made exclusive | 6 tests |
| witness check removed | the stale test |
| `admit` removed | the gather test |
| escape refusal removed | the scope test |
| delta keeps re-derived facts (naive) | the row-OR bound test. Correctness is unaffected, so only the bound catches it; a first version on a chain stayed green, because a chain has no alternative derivations |

CI runs `cargo test --example walk_sparse`; the benchmark is not run in CI.

## Measured (release, 4 vCPU, load ≈ 1.2, medians of 5)

Count-only includes attestation, binding, projection, preparation, recursion and counting. End to end replaces counting with decoding to the canonical tuple sets.

| fixture | nodes | foci | A count | B count | C count | C/A | C/B | A e2e | B e2e | C e2e | C/A e2e | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sparse(512), 1 out-edge | 512 | 2 | 22.0 | 94.9 | 20.1 | 0.91× | 0.21× | 64.9 | 146 | 69.5 | 1.07× | equal |
| clustered(512) | 512 | 2 | 21.8 | 213 | 48.8 | 2.24× | 0.23× | 323 | 554 | 388 | 1.20× | equal |
| dense(128), 10 % | 128 | 1 | 2.72 | 8.40 | 5.76 | 2.12× | 0.69× | 110 | 147 | 162 | 1.47× | equal |
| cyclic(300) | 300 | 2 | 181 | 3,828 | 679 | 3.75× | 0.18× | 788 | 4,530 | 1,917 | 2.43× | equal |
| chain(230) | 230 | 1 | 5.39 | 35.2 | 3.70 | **0.69×** | 0.11× | 12.3 | 41.6 | 10.1 | **0.82×** | equal |
| chain(700) | 700 | 3 | 105 | 270 | 31.3 | **0.30×** | 0.12× | 174 | 338 | 102 | **0.59×** | UNCHECKED (A = B = C) |
| mixed(256) | 256 | 1 | 9.09 | 95.0 | 91.5 | 10.1× | 0.96× | 341 | 552 | 538 | 1.58× | equal |
| edges(512) | 512 | 2 | 3.20 | 9.09 | 4.18 | 1.31× | 0.46× | 70.8 | 76.8 | 72.9 | 1.03× | equal |
| edges(1024) | 1024 | 4 | 14.3 | 70.0 | 29.9 | 2.10× | 0.43× | 310 | 378 | 331 | 1.07× | UNCHECKED (A = B = C) |

(ms; "equal" = A = B = C = ascent = crepe on the full tuple sets.)

**Phases of C** (ms): binding (attest + bind + project) is 0.02–3.1 ms and preparation 0.002–0.012 ms on every fixture; recursion is the rest of count-only. Decoding (shared by every arm, BTreeSet) is 6–1,205 ms and dominates end to end everywhere except the chains.

**Work counts.** C runs the same tile triples as B (e.g. 165,279 on cyclic(300)) but 2.3–474× fewer mask-risc calls (64,907 against 6,150,342 on cyclic(300)); C's calls are tile updates, B's are per-coordinate products, so the counts are of different operations. On sparse(512) and the chains, C's row ORs equal B's contributing coordinates exactly (7,158; 26,335; 225,650): neither does redundant inner work there, and C's gain is the per-coordinate broadcast and Ternlog it no longer runs.

**Microtile side** (recursion ms): m = 32 or 64 is best on 6 of 9 fixtures; m = 16 on edges(512) and edges(1024) (2.95 against 3.55; 24.6 against 26.3); m = 256 on mixed(256), by 0.4 ms. m = 256 (no block skipping) is worst on 4 of 9, up to 2.4× slower than the best (sparse(512)); m = 16 is worst on the other 5.

**Memory** (resident VmHWM growth in a fresh process, MiB; count-only arms bind, project and recurse without building tuples):

| fixture | A count | B count | C count | A + decode | B + decode | C + decode |
|---|---:|---:|---:|---:|---:|---:|
| sparse(512) | 4.89 | 3.95 | 3.75 | 9.67 | 8.62 | 8.54 |
| chain(230) | 0.31 | 0.37 | 0.46 | 0.89 | 1.00 | 1.08 |
| chain(700) | 9.12 | 2.02 | 1.91 | 20.0 | 12.8 | 12.7 |
| edges(1024) | 14.2 | 2.21 | 2.29 | 50.5 | 44.8 | 44.9 |
| cyclic(300) | 1.23 | 4.96 | 5.10 | 122 | 122 | 122 |
| dense(128) | 0.04 | 0.80 | 0.75 | 20.2 | 20.7 | 20.8 |

C's count-only memory equals B's within noise; it is below A where A's dense per-length planes are mostly empty, and above A on small dense or cyclic graphs.

**Value-derived reads.** Execution: A 98,012–5,375,274 (delta row at a scanned adjacency bit); B 0; C 0. Preparation for B and C: one codebook lookup per edge endpoint (458–83,846). Decoding: two codebook decodes per output tuple (105,800–9,180,000), the same in every arm.

## Reading

- **C against B:** faster on every fixture, 0.11–0.96× count-only. Skipping inactive triples, blocks and rows removes most of B's cost; the largest gains are on sparse, chained and cyclic graphs, the smallest on mixed(256), where most blocks are occupied.
- **C against A, end to end (the decision gate):** C wins on deep, low-fan-out recursion (chain(230) 0.82×, oracle-checked; chain(700) 0.59×, UNCHECKED), is near parity on sparse and benchmark-generator graphs (1.03–1.07×), and loses on dense, cyclic, clustered and mixed-length graphs (1.20–2.43×).
- **Why chains:** A sweeps its dense per-length planes every iteration, about 230–510 iterations here, while C touches only tiles and blocks that changed in the last iteration. That is the "unchanged frontier" skip (operator item 1), and it is where C's advantage comes from.
- **Why C loses on cyclic and dense graphs:** the frontier stays large, so C does A's work (the same row ORs) with extra tile, block and update bookkeeping.
- **Binding and preparation are not where the cost is** (≤ 3.1 ms everywhere, including the codebook lookups). The open cost of B and C is the attestation contract, not the measured binding time.
- **Decoding dominates end to end** for every arm outside the chains and is not addressed by any arm.

## Against the operator's gates

- A measured end-to-end region exists, preparation included, but it is narrow: long chains and other deep, low-fan-out recursion. Of the two fixtures in it, only chain(230) is oracle-checked; chain(700) is UNCHECKED.
- Outside that region C loses, so this is not a general replacement for walk_mask.
- Per the gate, the next step for this region would be a contract-level attestation of the frozen codebook (sorted-key and version witnesses, maintenance and invalidation accounting). Whether a region this narrow justifies that work is the operator's call; it is not started here.
- D-RU-6 stays Queued/Lab. No production admission.

## Open

- How wide the winning region is: a sweep over chain length, fan-out and number of foci, with oracles where they run.
- Whether the unchanged-frontier skip alone, ported into walk_mask (skipping delta planes and rows that did not change), gives A the same chain win without tiles or a codebook. That would remove the case for C in its only winning region.
- Decoding cost, shared by all arms.
