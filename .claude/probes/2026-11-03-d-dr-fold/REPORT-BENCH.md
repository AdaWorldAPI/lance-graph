# D-DR-FOLD lab comparison: A (gated Arm L), B (amortized binding), C (vector×tile)

**Status:** MEASURED, lab only.
- Probe: `bench/`, its own workspace. Not production authorization.
- mask-risc is consumed unchanged. Arm C calls `ndarray::simd` (`mask_or_assign`, `popcount_batch_u64`) directly, outside mask-risc.
- EXO-140 / EXO-46 are unchanged. No storage I/O was measured here.

**Source:** lance-graph `fb5e2d35b` (+ #1941). ndarray `b3d7919662a3`. rustc 1.99.0, release. 4-vCPU shared Xeon @ 2.10 GHz.

```sh
cd .claude/probes/2026-11-03-d-dr-fold/bench
CARGO_PROFILE_DEV_DEBUG=0 cargo +1.99.0 run --release -q   # exits 1 on any failed check
```

**Checks: 43 passed, 0 failed** (three full runs).

**Noise.** Timings moved a median of 45 % and up to 94 % between two runs. Absolute µs values are therefore indicative only. The A/C ordering and the exact operation counts were identical in both runs, and only those are relied on below.

**No-ms rule.** A cell of 1 ms or more prints FAILED-GATE.

## Query and semantics

- **Query:** for a root `a`, count the distinct `d` with `∃b,c. R(a,b) ∧ S(b,c) ∧ T(c,d)`.
- **Semantics:** Boolean (set) for every arm.
- **Oracle:** an independent oracle by nested loops over global ids. The oracle never reads a row, a plane or a chunk.

**Binding.** All arms share one coordinate space. It is bound by the EXISTING `dt64::dr::{LocalMap, MapTable}`: W256 tiles, wide coordinate `tile*256 + local`. No second binding store.

| arm | exact operation | terminal |
|---|---|---|
| A | one mask-risc `Program`: `s0 = cand under T_in; s1 = Gather(ib→R_in) under s0; s0 = Gather(ic→S_in) under s1`. `cand` is `Pred::Range` (ordered layout) or `Pred::EqU32` on a root lane (shuffled layout) | `Count(s0)` |
| B | A's hot fold unchanged. The binding (map, pointer certification) and the presence witness are paid per generation under two lifetimes. **retire**: re-mint the pointers on every presence generation. **split**: pointers survive presence changes; only the witness is re-verified | as A |
| C | top-down vector×tile: `v_b = R[a]`, `v_c = v_b·S`, `v_d = v_c·T`. Each step ORs the 256-bit chunk of every set frontier row (`mask_or_assign`). Frontiers are two preallocated tile registers, cleared by their touched lists | `popcount(v_d)` (`popcount_batch_u64`) |

**Transient vs persistent.**
- A: two scratch masks of n bits.
- C: two tile registers, also n bits each. No `C = S·T` or other composed relation is formed or stored.
- Neither arm allocates in the hot path: 0 allocations per call in every cell, counted by a global allocator.

## Correctness (both layouts, three fixtures each)

| case | result |
|---|---|
| A = C = oracle for every root | PASS |
| ~10 % of presence deleted through the API; binding kept (split lifetime); parity for every root | PASS |
| stale presence witness refused; fresh witness verifies (A and C) | PASS |
| bit cleared bypassing generation and witness | refused by both arms (key-digest witness) |
| duplicate paths (43 extra T edges, d with two c parents) | A refuses to mint (`NotFunctional`: a pointer table needs a functional parent). C equals the set oracle, because OR dedups. Set 64 vs bag 85 on the same root: C is Boolean and drops multiplicity (EXO-108 stays separate) |
| scope change: a tile replaced through `MapTable::advance` | stale frame refused by A and C; a mint under the stale frame is refused |
| duplicate global | refused by the existing `LocalMap::prepare` (`Duplicate`) |
| disables | A without `Gather(ic→S_in)`: 115 vs 77. C with the T hop skipped: 20 vs 77. Both wrong, as required |

## Hot fold: sweep

The sweep covered 54 cells:

- layout: ordered / shuffled
- fanout: f ∈ {4, 8, 16}
- presence density: keep ∈ {10, 50, 90} %
- candidate density: roots ∈ {1, 16, 64}, so the query subtree is 1/roots of the table, at 85 to 279,616 rows

The raw CSV is in the run output.

- **C was faster than A in all 54 cells, in both runs.** A/C ranged from 2.6× to 490×.
- **Why, with exact counts:**
  - A starts at the d level. It loads one pointer per present T candidate and a second per survivor, whether or not the R/S chain is alive above it.
  - A also pays whole-lane word passes (predicate, gate, `Count`) proportional to all rows, not to the query.
  - C starts at `a` and expands only the live frontier.
  - Example, ordered, 64 roots, f=16, 90 %: A did 6,697 loads; C did 235 chunk ORs over 49 tiles.
  - The same cell shuffled: 6,697 loads vs 2,980 ORs over 1,196 tiles.
- **Candidate density:** at fixed f=16 / 10 %, A grows with table size (shuffled: 3.2 → 15 → 59 µs for 1 / 16 / 64 roots). That is the `EqU32` pass over every row. C stays at about 0.1–0.2 µs.
- **Locality and cross-tile:**
  - Ordered children share a chunk, so one OR covers up to f edges.
  - Shuffled children spread across tiles, so one OR covers about one edge. C's ORs grow about 10× and its tiles touched about 24× in that cell. Tiles touched counts all three frontiers (b, c, d); a first run omitted the b frontier.
  - C's chunk store also grows: 10.1 MB shuffled vs 1.8 MB ordered at 279,616 rows, against A's 3.4 MB of pointer lanes.
  - **Cross-tile data costs C memory, not latency, in this range.**

## Falsifier: where A wins

The falsifier puts dense R and S over a sparse T. C must expand every live b and c, while A touches only the few present T rows.

| layout | roots, f | keep R/S/T % | A loads | C ORs | A/C |
|---|---|---|---|---|---|
| ord | 1, 16 | 100/100/1 | 86 | 89 | **0.67** (A faster) |
| ord | 1, 32 | 100/100/1 | 696 | 493 | **0.69** |
| shuf | 1, 16 | 100/100/1 | 86 | 224 | **0.85** |
| shuf | 1, 32 | 100/100/1 | 696 | 1,303 | 1.00 (tie) |
| ord | 1, 16 | 100/100/5 | 438 | 193 | 1.25 |
| ord | 16, 16 | 100/100/1 | 100 | 93 | 4.11 |
| ord | 1, 32 | 5/100/100 | 32,768 | 0 | 1,691 |

**Break-even, measured:**
- A wins only when the d level is sparse (about 1 % present), the upper levels are dense, and the candidate range is the whole table.
- Once the query covers a fraction of the table (16 roots), A's whole-lane passes put it behind again (4.1×).
- C's advantage is largest when the chain dies early: a sparse R gives 0 ORs against 32,768 loads.

What decides it is the **evaluation direction** (top-down frontier vs d-anchored gather) plus **A's per-row lane term**. Tile geometry is secondary.

## B: amortization (64 roots, f=8, 50 %)

| layout | hot A µs | presence witness per generation µs | split policy per query at Q = 16 / 64 / 256 (measured loop) | retire policy |
|---|---|---|---|---|
| ord | 2.75 | 502 | 34.4 / 10.5 / 4.7 | pointer certification is FAILED-GATE (≥ 1 ms at 37,440 rows); amortized it needs Q in the thousands |
| shuf | 11.8 | 758 | 60.3 / 24.3 / 16.2 | as above |

- Derived (`hot + per-generation / Q`) and measured loops agree within run noise.
- Witness verification is O(popcount) with reverse lookups. It costs about the same as 180 (ord) or 64 (shuf) hot queries.
- Retiring the pointers on every presence change is the expensive policy. The split policy reaches near-hot cost at about 256 queries per generation.
- C has no pointer table to certify, but its tiles ARE the presence:
  - a deletion is a bit clear (tested);
  - an insertion changes the chunk structure (rebuild: `chunkbuild_us`, ≥ 1 ms at large sizes; not measured incrementally).
- A's pointers are presence-independent. A presence add is a bit set.
- At 37k+ rows, map preparation, certification, witness verification and chunk build all cross 1 ms. Each must be per generation and incremental, never per query.

## Recommendation

Next authorized work, from the four options:

1. **Existing ndarray tile algebra / lowering: YES, first.**
   - The vector×tile fold is built entirely from existing `ndarray::simd` words.
   - It beats gated Arm L in 54 of 54 sweep cells and keeps Boolean semantics.
   - It needs no new mask-risc op if it stays a terminal-side fold over resident tiles.
   - What is missing is a lowering that chooses evaluation direction (top-down frontier vs d-anchored gather) from presence density.
2. **New mask-risc capability: NOT yet.**
   - The comparison shows no op is needed to get the speed.
   - Whether the frontier fold should live inside mask-risc is a separate operator ruling (D1). The data does not force it.
3. **DR→Lance physical I/O planner: separate track, unchanged.**
   - Nothing here measured storage.
   - The H14c result stands: a scattered mask cuts rows decoded, not bytes; a clustered one cuts both.
   - The missing read-planning contract (D4) is still the gap.
4. **No new operator: holds for Gather itself.**
   - On the examined shape, Gather is not the bottleneck.
   - Evaluation direction and the whole-lane term are.

**Limitations:**
- Forest shapes, plus one DAG case.
- Arm C is one block-sparse representation (256-wide row chunks), not dense 256×256 tiles.
- Presence inserts into C's tiles were not measured.
- Single shared host, high run-to-run noise.
