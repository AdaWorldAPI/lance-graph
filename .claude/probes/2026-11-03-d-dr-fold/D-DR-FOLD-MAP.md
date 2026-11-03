# D-DR-FOLD: three kinds of indirection, kept apart

**Status:** ARCHITECTURE MAP, source-grounded. Read-only pass: no code run, nothing
measured in this pass. Every number below is quoted from an earlier recorded
measurement, with its source. No production change. mask-risc unchanged. EXO-140 /
EXO-46 unchanged.

**Source SHAs.**

| what | SHA |
|---|---|
| tree read | lance-graph `fb5e2d35b10fc4940f5c78285bae246f90c7faad` (`origin/main` after #1940) |
| D-BAG-FOLD v3 / v4 measurements | `d651b70d4a3b2299c21f1d354ac02a3311280441` (+ #1938 / #1939) |
| H14d measurement | `802db7dfea88ecd33790a85d176910b62e2b902b` (entry `2026-10-16-d-holo-h14d-logic-mask-read.md`) |
| H14c-4 measurement | as recorded in `.claude/plans/d-holo-h14c-thought-first-v1.md` § Results (v4, 2026-10-15); the plan names no commit |
| D-DRMC-5, D-DYNBAG-1 | as recorded in their board entries (`2026-10-22-drmc-5-assoc-probe.md`, `2026-11-01-dynbag-1-n-compose.md`); no commit named |

**Classification.**

| label | meaning |
|---|---|
| PRODUCTION | a workspace-member crate and a caller in production |
| PRODUCTION CRATE, LAB CALLERS | a workspace member whose only callers are excluded crates |
| LAB | an excluded probe, or a recorded probe result |
| PROPOSED | named in a plan or working model, not built |
| BLOCKED | needs an owner decision first |
| UNCLASSIFIED | its legality under an existing rule is undecided |
| NOT VERIFIED | not checked in this pass |

## Three paths

1. **Physical indirection (Path A):** where a DR region's fragments live in storage,
   and how they become resident.
2. **Logical indirection (Path B):** how a global DR id maps to a local tile position.
3. **Relational evaluation (Path C):** how resident witnesses establish a relation
   without storing it.

Prefetch and lookahead belong to A. Mapping reuse belongs to B, and it is an
amortization, not a different hot execution algorithm. A tile sweep or fold belongs to C.

## Path A: storage, from fragments to resident tiles

| finding | class | source |
|---|---|---|
| No DR → physical-location translation. `lance-graph-dt64` depends only on `lance-graph-contract`; "DR routes, DT interprets … Nothing here holds a value." | PRODUCTION (absence) | `crates/lance-graph-dt64/Cargo.toml`, `src/dr.rs:1-20` |
| No I/O lookahead, readahead, coalescing under DR control, or I/O/fold overlap. No crate sets `fragment_readahead`, `batch_readahead`, `io_parallelism` or uses `take_rows`. | PRODUCTION (absence) | crate-wide grep; Read of `lance-graph/src/graph/versioned.rs` (plain `Dataset::open` + scans) |
| Lance 13 `with_row_addr_prefilter`: a scattered mask cuts rows decoded, not bytes fetched; a page-local mask cuts both (1 % uniform 1.00×, 1 % clustered 0.24×, 0.1 % uniform 0.77×). | LAB, H14c-4 | `.claude/plans/d-holo-h14c-thought-first-v1.md` § Results |
| Logic → address mask R → prefilter → terminal: rows scanned = \|R\| in 96/96 cells; bytes follow page locality; a 1,024- or 8,192-address aperture reads 0.01–0.12×. OPEN in that entry: which address order puts survivors on few pages. | LAB, H14d | board entry `2026-10-16-d-holo-h14d-logic-mask-read.md` |
| Ordered keys → `seek_from` ranges → row-address prefilter. A table that is not one contiguous fragment is refused (`NotContiguous`): physical fragmentation is refused, not scheduled. | LAB | `crates/crepe-plan-probe/src/addressed.rs:1-24` |
| The only "lookahead" in the tree is `ndarray::simd::prefetch_read_t0`, a CPU-cache prefetch. It is not physical I/O prefetch. | LAB | `crates/drf-focus-probe/src/lib.rs:370-406` |
| Lance reads run on the recovery clock and folds on the live clock; read once at replay, fold resident. | PROPOSED (working model; measured in one consumer) | `.claude/knowledge/split-tunnel-sparse-append-resident-fold.md` |

**Every `bytes_read` in this tree is what Lance requested from the local object
store, served from the operating-system page cache.** None is a physical disk read,
and none may be reported as one. No probe so far separates disk from page cache.

**The missing contract (physical read planning):** nothing turns a DR WHERE into an
ordered list of physical fragments or page ranges. It would need to state:

- which DR coordinates map to which Lance row addresses, under which dataset version
  (H14c-5: without stable row ids, compaction moved 3,510 of 3,510 `_rowid`s);
- how row addresses group into pages and fragments;
- how far ahead future requirements are known (lookahead distance);
- the read-coalescing and concurrency policy;
- that the read plan follows the DR address plan and never changes relational
  semantics.

No owner is named for it today (decision D4 below).

## Path B: addressing, from global DR id to a reusable local map

| finding | class | source |
|---|---|---|
| `LocalMap::prepare`: a sorted lookup index per tile. Refuses duplicates and over-width; lookups are counted. | PRODUCTION | `crates/lance-graph-dt64/src/dr.rs:556-641` |
| `MapTable`: an append-only per-tile history. An extension keeps positions, a replacement is a new generation, a removal is an event. `map(frame, tile)` returns a borrowed `TileMap`: zero allocation, replays the layout at the frame's version. `locate` resolves across listed tiles and refuses ambiguity. | PRODUCTION | `dr.rs:653-988` |
| `FrameStore` / `Frame`: issuer identity; `check` refuses any frame that is not the current seal. | PRODUCTION | `dr.rs:376-494` |
| `Cube` builds every relation's lanes once per frame (one lookup per id, at construction). Joins and terminals reuse them, and each terminal re-checks the frame. Presence edits (`stage`/`commit` of edge Add/Retire through a per-relation `DeltaLog`) keep the frame. Only a grown tile, which must extend (`Renumbered` otherwise), rebinds. | PRODUCTION CRATE, LAB CALLERS | `crates/lance-graph-xlog/src/cube.rs:1-35, 848-1000`; callers: `xlog-relcomp-probe`, `lance-graph-codegraph` (both excluded) |
| The free functions `lance_graph_xlog::{support, bag_count}` rebuild lanes and look every tuple up per call. `Cube` is their amortized form. | PRODUCTION CRATE, LAB CALLERS | `crates/lance-graph-xlog/src/lib.rs:255-297` |
| The D-BAG-FOLD v4 pointer lens is minted per key version and retired by the frame on any presence change (A4′, conservative). | LAB | `REPORT-V4.md` |

**`LocalMap` / `MapTable` already are the reusable coordinate mapping.** No second
binding store is needed or proposed. Reusing them is an amortization of prepare and
lookup cost; it does not change the hot execution algorithm.

**Hazard (BLOCKED, D3):** `MapTable::map` deliberately answers old frames (replay
by version). A map read therefore cannot detect stale presence. Cube-style reuse is
correct only if the cube is the single presence writer. This is the same condition
as D-BAG-FOLD v3 case 3b.

Foreign keys use the same contract: `MapTable::locate` plus `Cube`'s tiled mode
(`tile_ordinal * 256 + local`). Plan row MISSING-7 ("multi-tile global → (tile,
local) map", `d-dynbag-holographic-bags-v1.md:526`) looks at least partly closed by
this. Whether the plan was updated is NOT VERIFIED. Cross-tile composition is still
missing (D-DYNBAG-1 OPEN).

DuckDB comparison (SECONDHAND, not fetched in this pass): a selection vector, or a
dictionary vector, is a per-chunk index indirection over unchanged data, which is
conceptually `$i`. It is ephemeral per chunk. `MapTable` is versioned and reused
across queries.

## Path C: algebra, from resident tiles to a terminal

| finding | class | source |
|---|---|---|
| `Pred::Range{lo, hi}`: a contiguous range write, with no per-row compare. On an address-ordered lane a key prefix names a subtree, so its mask is a range. | PRODUCTION (frozen IR) | `crates/lance-graph-mask-risc/src/ir.rs:159-180` |
| `And`/`Or`/`AndNot`/`Not`/`Ternlog`: co-addressed conjunction, with no index lane. | PRODUCTION | `ir.rs:230-256` |
| `MaskOp::Gather{lane, foreign, under}` and `Pred::EqU32Via` are the only in-program cross-row reads. | PRODUCTION | `ir.rs:156-158, 257-282` |
| `ScatterOrU32` is terminal only: its scattered mask "is never an intermediate" (survival condition). `CountKeyRunsU32` is a run fold on a key-ordered lane, and yields only a count. | PRODUCTION | `ir.rs:322-389` |
| XLOG support `q(Y) :- r(X,Y), s(Y,Z)`: two `ScatterOrU32` programs, then `ndarray::simd::mask_and` outside mask-risc. No Gather, but an indexed scatter, and two program results combined. The answer is over Y only: the X and Z bindings are not preserved. | **LAB / UNCLASSIFIED** until the computed-mask survival rule and outer-coordinate binding preservation are established (D2) | `crates/lance-graph-xlog/src/lib.rs:16-35, 225-270` |
| XLOG bag count: per-side `GroupReduce Count`, then a checked i64 product. This is multiplicity, kept separate as EXO-108. | PRODUCTION CRATE, LAB CALLERS | `lib.rs:273-297` |
| `relcomp::compose` over DR64 64×64 Boolean registers: "ORs one row of B per set bit of A". It indexes by bit position inside a resident tile, with no pointer lane. It returns C as a transient 512 B register, which is the composed relation. Measured 50 ns sparse, 2.4 µs dense per tile pair. | LAB | `crates/dr-math-binding-probe/src/relcomp.rs:1-60, 212`; entries D-DRMC-5, D-DYNBAG-1 |
| `ncompose` ℕ-compose via `matmul_i8_to_i32` is bag-exact. About 21 µs per tile, attributed in that entry to per-call pack/alloc (HYPOTHESIS there). | LAB | `crates/xlog-relcomp-probe/src/ncompose.rs`; entry D-DYNBAG-1 |
| No vector×tile fold, where a 64/256-bit frontier r·S·T would reach the terminal without forming C. `relcomp`'s `Step::Push` is a stack push. No cross-tile compose. | absence | as above |

## Answers to the brief

**1. Implementations and callers.**
- Path A: absent in production; LAB probes only.
- Path B: PRODUCTION (`dt64`). `Cube` is a production crate with lab callers.
- Path C: the operators are PRODUCTION (mask-risc); the composition kernels are LAB.

**2. Connections between the paths.**
- A↔B: none. DR coordinates never become Lance row addresses in production. H14d's
  address mask is the lab prototype.
- B↔C: only `Cube`, whose lanes hold local positions.
- A↔C: none. Resident-tile reuse is a working model, not a component.

**3. Where a tile sweep can remove execution-time Gather, in principle:**
- every witness is on the same row axis: `And`/`Ternlog`, no lane;
- the cross-row relation is a contiguous range known at plan time: `Pred::Range`;
- both endpoints lie in one resident DR64/DR256 tile: bit-matrix compose, or a
  vector×tile fold;
- hierarchical ancestor relations. In the D-BAG-FOLD fixture b(d) and c(d) are key
  prefixes of d, so "d lies under some b ∈ R_in" is a union of contiguous ranges.

**4. Where Gather is required:**
- a data-dependent, non-contiguous foreign key;
- a relation spanning tiles (no cross-tile compose exists);
- an id domain wider than one tile with no tiled binding.

For the examined query shape (the D-BAG-FOLD existential) and the lowerings
available today, Gather is also required for the hierarchical case: the frozen IR has
no in-program op that expands a mask over ranges or reduces runs to a parent mask
(`ScatterOrU32` cannot feed another program; `CountKeyRunsU32` yields a count only).
**This statement covers that shape and those lowerings only, not query shapes in
general.**

**5. Binding reuse that exists:**
- `MapTable` / `TileMap`: borrowed, zero allocation, version replay;
- `Cube`: lanes built once, presence through `DeltaLog` without a remint;
- `LocalMap::lookups`, to prove that no lookup happens per query.

**6. Physical I/O:**
- present: the upstream Lance prefilter and scan statistics, used only in lab probes;
- absent: everything in the read-planning contract above;
- proposed: the split-tunnel working model; page-scale address clustering (H14d OPEN).

**7. The 83 % hypothesis: UNMEASURED.** Four denominators must be kept apart:
queries, relationships, candidate accesses, and cost.
- The only related numbers are locality, not avoidance (LAB v3): 47.6 % (17,476
  nodes) and 70.4 % (2,340 nodes) of `$i` reads land in the reading row's own 64-row
  word, and every pointer stays inside the candidate subtree.
- Falsifiers:
  - F1: uniform random foreign keys over N ids. Two endpoints share a 256-wide tile
    with probability ≈ 256/N, about 0.4 % at N = 65,536.
  - F2: sparse candidates. A sweep pays the same whole-lane passes as Arm L plus the
    expansion, so cost avoided can be about 0 even when every query avoids Gather.
  - F3: dense tiles. Compose cost grows with popcount(A).
  - F4: relations that span tiles have no sweep at all.

**8. Benchmark plan.** Carried out separately as the D-DR-FOLD lab comparison; see
its report.

**9. Owner decisions:**
- D1 (mask-risc owner): is an in-program range-expand or run-to-parent mask op
  admissible?
- D2 (#1791 / EXO-140 owner): is scatter-then-AND, as XLOG support does it, a
  demanded result or a forbidden intermediate, and must outer bindings be preserved?
- D3 (EXO-46 owner): may a coordinate binding outlive presence changes (the `Cube`
  split), and who is the single presence writer?
- D4 (DR / storage owner): who owns DR → Lance row-address translation and
  page-scale address clustering?
