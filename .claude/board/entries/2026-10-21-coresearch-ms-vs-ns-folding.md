# 2026-10-21 — Co-research: why the recent relalg / D-RU-6 probes measure milliseconds

**Status:** EXPLORATION MAP. Nothing is adopted; the operator chooses. D-RU-6 stays Queued/Lab.

## Question

The operator: "Milliseconds is not normal in a substrate with nanoseconds being the normal." Why do relalg and the last 5–15 PRs measure 0.4–5,000 ms? Is folding missing? Do we still use CSR, adjacency scans, gathers or tuple decoding where a fold belongs? Side question: should the join key carry a little-endian ordered-lane contract?

## Premise audit

- **Partly wrong.** Each operation already runs at nanosecond scale: walk_mask sweeps at 0.59 ns per word, mask-risc words cost about 0.2 ns, relalg about 0.34 ns per lane-row. The milliseconds come from running 1e5–1e8 of them.
- **Split.** Some work is intrinsic to the output a consumer asks for; some work no consumer asks for. Only the second is a missing fold.

## Where the milliseconds go

Each code path below was verified in the source, and the two main sites were re-read on the main thread. The numbers come from the board entries.

**1. Tuple decoding in every D-RU-6 arm.** This is the dominant avoidable cost.
- `tuples()` in `walk_mask.rs:232-249` and `walk_tile.rs:538-569` tests every column bit of every row and length. That costs O(n) per row, not O(set bits).
- Each set bit becomes a `BTreeSet` insert, and NoWalk is built as a full complement.
- edges(1024): 318 ms to decode against 0.16 ms count-only, about 2 M tuples at roughly 158 ns each. cyclic(300): about 1.2 s.
- The count-only path already exists (`walk_tile::counts`, `:529-535`). Only the correctness oracle needs the tuples.

**2. The crepe-plan-probe native terminal enumerates even for a count** (`native.rs:401-461`, reached from `pipeline.rs:189-214`).
- For every non-Exists answer it forms the per-slot cartesian product over sides that emit, clones and sorts it, then inserts into a `BTreeMap`.
- Sides that do not emit are already multiplied in rather than enumerated, so the fold exists only for those.
- Preparation copies side rows into owned Vecs: 3.5 ms at 50 k rows.

**3. Arm B (`walk_tile.rs:363-407`).** It spreads, broadcasts and runs a Ternlog over 1,024 words per contributing node, about 3,072 words each.
- cyclic(300): 6.15 M calls in 3,828 ms, about 0.2 ns per word.
- The cost per word is at the floor; the 256× amplification is not. `tile_gemm.rs:87-104` `BitMat::compose` does one 4-word OR per set bit (2.6–25 µs per 256² product), but it is used only as a test oracle.

**4. Arm A and A′.**
- Arm A gathers `delta[row(l2,y)]` at every adjacency bit, and sweeps all `lens·n·w` words every iteration: 230 M words at chain(700).
- A′ removes the sweep but still scans every adjacency row: 14.6 M words.

**5. Arm C.** It does the same row ORs as A′, plus allocation inside the hot loop: a fresh tile twice per changed tile per iteration, and `block_occ` and `next_occ` rebuilt every iteration.

**6. relalg D-RA-1, U1 and U2: not a fold gap.**
- Each query is one mask-risc program, with no tuples and no CSR. U2 runs 14 lanes over 1 M rows in 4.77 ms, about 0.34 ns per lane-row.
- This is intrinsic to scanning a million rows. The only lever is skipping rows: D-CMR-2 runs and extents, or the `under` gate.

**7. D-RMAP-DIR-1.** The milliseconds are in the baseline arm only, the dir-sim View walk. The fast arms run at 32–79 ns per user. The #1639 attest takes 7 ms and pays back after 2–3 queries.

**Unused fold primitives.**
- mask-risc offers `Pred{under}`, a fused `Ternlog` with an `Any` terminal, `CountKeyRunsU32` and `GroupPowerSumsI32`.
- lane-fold `Query::check` refuses `wants_rows`.
- `Refuse::PairList` is declared but raised nowhere in `lance-graph-lane-fold/src`, which is five files, all searched.

**Answer.** Folding is missing at the **terminal**, not in the core loop. The probes and the native terminal materialize tuples for consumers that only need a count, an existence check or a complement. The remaining milliseconds are intrinsic row scans, plus lab-only amplification and allocation in arms B and C.

## Exploration map

Verdicts combine three reviewers on the crosswalk: a bridge architect, a firewall and fit critic, and a falsifier designer. Every probe is pre-registered.

**Shared oracle rule for every D-RU-6 probe.**
- The D-RU-6 correctness rule (`d-relalg-unified-v1.md:205`) stays: full `Walk` / `NoWalk` tuple sets against crepe and ascent.
- Each fixture's oracle tuple sets are computed **once, untimed**, and encoded bijectively into reference planes. Timed arms compare planes word for word.
- One untimed tuple-set check per fixture keeps the decoder honest.
- Counts or fingerprints never replace the oracle.

| idea | verdict | firewall | probe and kill condition |
|---|---|---|---|
| **X1+X2+X3** demand-typed plane terminal: Exists, Count or Rows. Exists and Count end at a popcount or plane comparison; NoWalk is counted as `extent AND NOT reach`; only Rows decodes | **ADOPT-NOW** (lab probes; no production change) | PASS (law 1, doctrine §6.2, law 10) | P-X2: any count or plane mismatch kills it. Count-only above 5 % of decode wall time on 3 or more of 7 checked fixtures kills it. P-X3 adds a dangling-node fixture, so the extent mask is shown load-bearing (can-fire) and edges(512) stays unchanged (silent) |
| **X4** walk only set bits in `tuples()` when Rows is demanded | **ADOPT-NOW** | PASS. Decode reads only its own output plane, and law 10 lists "walk set bits". C's no-bit-scan audit covers execution, not decode | P-X4: under 2× decode speedup on 5 of 9 fixtures kills it, as does any tuple difference. Also kills if the time-vs-output intercept is not cut 5× or more |
| **X6** native Count(Bag) = Σ over joint mask of Π side degrees; Count(Set) as Σ Π per-slot distinct counts | **PROBE** | Bag: PASS. Set: CONFLICT. It is legal only when the joint-slot key is in the head; otherwise it is law 7's count-distinct-across-factors. Resolve demand and head-key presence in the semantic plan first (law 1) | P-X6 on the D-MPS-8 star (n = 50 k) plus a multi-rule union with overlap. Any mismatch kills it, and so does under 10× on the star. The overlap twin must still route to dedup |
| **X9** hoist allocation out of the hot loops of A′, B and C | **ADOPT-NOW** (probe hygiene) | PASS | P-X9: a counting global allocator must read 0 in the steady loop, and chain(700) must gain at least 15 % |
| **X5** factorized Rows output: `(x, length, row bitmask)` borrowed from the plane, with optional `expand()` | **PROBE** | PASS, with law 7 kept | P-X5 over `--sweep`: kills if the wall-time slope against tuple count is not at least 10× flatter, or if the expansion fails the oracle |
| **X8** `BitMat::compose` in arm B | **PROBE**, lab | CONFLICT: a per-bit write scatter inside one tile. Less severe than A's cross-table read, but it falls under the same open slot-binding ruling (`:234`) | P-X8: kills if under 3× on cyclic(300), or on any plane mismatch |
| **X10** C's update through mask-risc `Ternlog` + `Any` | **PROBE** | PASS (no new opcode) | P-X10: kills if under 1.3× on C's loop |
| **X11** push/pull per iteration | **PROBE**, lab | CONFLICT: a pull through reverse adjacency is a gather. A co-addressed pull exists as an inner product over the shared domain, popcount(rowₓ(A) ∧ row_z(δᵀ)) | P-X11 (P-SYS-4): kills if work per iteration does not decrease after the frontier peak, or if under 1.5× against A′ |
| **X12** zero-zone and one-bit-word skip | **PARK** | PASS | Already capstone points 3 and 9 |
| **X17** order contract on the join key | **PROBE** (correctness gate, contract) | PASS, conditionally: a separate join-key witness type with a numeric projection, never memcmp, never fused with a row-order digest. Correction: ADMISSION B requires an attested sorted key for **every** same-table key-range seek; frozen-codebook decoding is admitted separately, on an immutable mapping within the attested version, with no order requirement (`entries/2026-10-20-six-decisions.md` §1) | P-X17: binding a merge or seek join to a byte-ordered LE key (u16 1 vs 256) must be refused at bind. An equality or mask join on an identity key must bind without a witness |
| **X7** min-length or tropical plane | **SKIP** for D-RU-6 | CONFLICT: it changes the answer (law 3) | Only as a separately declared existence or min-length query |
| **X14** repeated squaring | **SKIP** | CONFLICT: it produces powers of two, not every length, and the source paper loses on sparse input | — |
| **X13** row skipping in relalg | **SKIP** (already in flight) | PASS | D-CMR-2 |
| **X15** Brie or B-tree relation storage | **SKIP** | TRAP: a stored second copy | — |
| **X16** differential dataflow or DBSP | **SKIP** as a dependency | CONFLICT: indexed second copies. The concept already lives in the dirty-row rule | — |

## On the little-endian join contract

**Equality and mask joins need no order contract.** They need identity-canonical keys plus extent bounds.

**Merge, seek and range joins need a total order.** ADMISSION B already demands an attested sorted key for every same-table key-range seek. Frozen-codebook decoding does not: it is admitted on an immutable mapping within the attested version, accounted for and never used as a relational lookup, so a codebook needs a mapping and version attestation, not an order witness (`entries/2026-10-20-six-decisions.md` §1; `plans/d-relalg-unified-v1.md:234`).

**The contract must be numeric order over the LE image.** LE bytes are not memcmp-ordered: u16 256 sorts before 1. That is how `OrderedLaneWitness` defines it. Today the witness covers only sealed facet lanes. So the gap is a **join-key witness type**, checked at bind time and only for carriers that need order. It is not an attestation on every join key.

## Not searched

- No outside source was read in full.
- SECONDHAND only: Factorized Databases, Beamer SC'12, Bagan–Durand–Grandjean / Segoufin, Umbra, Leapfrog Triejoin.
- Failed fetches: the Kuzu blog (DNS failure) and the LAGraph BFS template (404).
- Licences other than Soufflé's are from memory.
- Inside: ogar-loco and r2il were not searched for an existing demand-typed terminal. lance-graph-quack and cypher-quack were not inspected.
- The source of the finding that the two attestations must never be merged was not located in a bounded search.

## Candidate SPEC v1 for the 5+3

The ADOPT-NOW set, X1+X2+X3, X4 and X9, plus probe X6, form one coherent design: **a demand-typed terminal** with a plane oracle, in which tuples are built only for Rows. The operator chooses whether this goes to `/5plus3`.

PR #1657 | STATUS: open | OUTCOME: milliseconds attributed per path; folding is missing at the terminal, not the core loop | OPEN: which ideas to take forward; join-key witness type; X8/X11 under the open slot-binding ruling

## Correction (operator, 2026-10-21, after the map)

**Ruling: no millisecond result is acceptable anywhere.** A single operation must finish in nanoseconds; a batch takes microseconds at most.

1. **The tuple decode is the violation itself.** Making it faster (X4) does not satisfy the rule. X4 moves from ADOPT-NOW to **SKIP**, and the "decode cost" part of the 5+3 that followed was dropped.
2. **lance-graph-mask-risc is the reference folding algebra.** It stays untouched until tests prove a new algebra also runs in nanoseconds.
3. **The native count fold (X6) was implemented and reverted** (884abfd4..c00011ef, reverted in c49db8dc). On 50,000 centers it measured:

   | case | time |
   |---|---|
   | Count(Bag) | 620 µs |
   | Count(Bag), filtered | 1.2 ms |
   | Count(Set) | 1.6 ms |
   | Count(Set), filtered | 1.3 ms |

   It also kept the enumerating terminal, at 243–310 ms. The causes were scalar loops instead of the ndarray polyfill, and a per-slot sort for Set. Any rework must be SIMD through ndarray and must avoid a per-slot sort, which depends on a sorting contract.
4. **New open points (operator):**
   - An ndarray plug for the contract crate, plug-and-play like `hotplug.rs`. The zero-dependency rule in `lance-graph-contract/Cargo.toml` is ruled obsolete.
   - A sorting contract. mask-risc's `CountKeyRunsU32` checks key order at run time, and no attestation backs it.
   - Whether the substrate is sorted to begin with.
   - Whether a little-endian or sorting contract would slow folds down or prevent them.
   - Why Quack does not consume the contract set algebra. Today only relalg and `lance-graph/src/fold` consume it; quack uses the contract only for `OrderedLaneWitness`.
