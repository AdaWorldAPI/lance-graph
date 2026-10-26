# D-DR256-WHERE capstone — population-masked reads, thought amortization, DuckDB harvest

**Status:** P0 census DONE (this document). P1 lab measurement DONE (reported, not gated).
P2–P5 PROPOSED, NOT RUN, NOT AUTHORIZED. Documentation and one lab example only; no runtime,
ABI, planner, writer, WAL or ownership change.

**Board entry:** `entries/2026-10-26-dr256-where-capstone.md`.
**Predecessors (read, not restated):** #1788 (D-DQM crosswalk, Appendix A), #1789 (P1 chunk-touched,
FAIL), #1790 (D-DR256 global union lab), #1791 (on-demand / u16 / L1 lab), ndarray #377
(`mask_scatter_or_u16`).

## 0. Scope and firewall

**Goal.** One Quack optimization interface for eligible consumers, implemented through the existing
mask-risc / ndarray path, answering "which rows" (`WHERE`) and "read these properties over those rows"
with selection statistics and a witness.

**Firewall (unchanged by this document).**
- The scatter survival condition (`lance-graph-mask-risc/src/ir.rs:329-337`) stays in force. A
  `ScatterOrU32` mask is legal only as the demanded result, never an intermediate.
- Not touched: the 512-byte SoA ABI, CE64, HHTL16 basin storage, Thought-Git, MailboxSoA, CycleWriter,
  WAL, asynchronous write semantics.
- DataFusion is not extended (grace period).
- No new planner, IR, interpreter or executor. A strategy choice is a BIND/BUNDLE fact (fold laws 1–3).

## 1. Consolidated architecture

**Two universes, one algebra.**
- DR64 focus: a 64×64 local selection (512 B mask) plus a sealed u16 address map (8 KiB). Local,
  resident, the default.
- DR256 global: 65,536 addresses, an 8 KiB mask (1,024 u64 words, 256 blocks of 256 rows).

**WHERE is already a Quack program.** `Query { filter, agg }` (`lance-graph-quack/src/lib.rs:1301`).
`Filter` (`lib.rs:340`) lowers to mask-risc predicates and mask ops. `lower` / `lower_fused`
(`lib.rs:1383`, `:1397`) produce one `Program`. The terminal comes from `terminal_of`
(`lib.rs:2421`):
- `Agg::Rows → Terminal::Keep` returns the population mask (`PopulationSelection.optional_mask`).
- `Agg::Count`, `Agg::Any` return the count and existence.

**What may be reused.** A `Keep` mask is a legal projection-space mask: a later program may read it as a
plane (`Filter::Plane`). A `ScatterOrU32` mask may not (`ir.rs:329-337`). Every "compute once, read by
N thoughts" design in this document therefore reuses `Keep` output, never scatter output. The #1791
contract proposal (a witnessed derived population) is the only route to reusing a scattered mask, and it
is still not authorized.

**Strategy space, mapped to existing mechanisms.** Each strategy is a choice the consumer or Quack's
BIND step makes. None needs a new executor.

| strategy | meaning | existing mechanism | status |
|---|---|---|---|
| LOCAL | answer inside DR64 | Quack program over the 64×64 focus planes | ALREADY |
| REBIND | rebuild the local selection under a new map/seal | DRF `FocusKey` full witness (`drf-focus-probe`) | LAB |
| GLOBAL | build the DR256 mask, fold over it | `ScatterOrU32` as demanded result; reuse needs the #1791 contract | LAB / CONTRACT |
| BORROW | read SoA values in place under a mask | masked terminals (`MaskedSumI32`, `GroupReduce`, …) | ALREADY |
| GATHER | filter through a foreign key over a resident plane | `Filter::EqU32Via` / `Filter::Semijoin` → `MaskOp::Gather` (`lib.rs:2089-2093`) | ALREADY |
| STORAGE | witnessed range read instead of a sweep | `PrefixLowering::Bound { lo, hi, lane_version, lane_digest }` (`lib.rs:305`) | ALREADY |

## 2. Verified capability table (P0 census)

Classification: ALREADY / MISSING-ADMISSIBLE / LAB / REQUIRES-CONTRACT / NOT-APPLICABLE / REJECTED.
Every ALREADY row was checked against source in this session.

| capability | where | class | note |
|---|---|---|---|
| WHERE → population mask | `Agg::Rows → Terminal::Keep`, `lib.rs:2430`; `ir.rs:321` | ALREADY | |
| COUNT over the selection | `Terminal::Count` → `popcount_batch_u64`, `exec.rs:1788-1791` | ALREADY | |
| ANY over the selection | `Terminal::Any` → `mask_any`, `exec.rs:1793-1795` | ALREADY | No early exit in the generic tile loop (`any |=` per tile; ndarray `mask_any` ORs every word, `simd_masking_ops.rs:4288-4294`). Early exit exists only on the fused ternlog path (`exec.rs:1049`, `:1204`). |
| Predicate cost gated by a prior mask | `*_to_mask_under` twins, `exec.rs:553-576` | ALREADY | |
| Conjunction order by measured skip | `Filter::and_by_skip`, `lib.rs:752` | ALREADY | |
| Witnessed range vs sweep | `PrefixLowering`, `lib.rs:305` | ALREADY | |
| Exact fk semijoin | `EqU32Via`, `Semijoin → Gather`, `lib.rs:2089-2093` | ALREADY | |
| Single-property masked read | masked terminals in the `Terminal` enum | ALREADY | |
| Exact occupancy (active 64-row words, active 256-row blocks) | nowhere | MISSING-ADMISSIBLE | `touched_words` / `span_words` (`ir.rs:1885`, `:1892`) are extent word ranges, not occupancy. Smallest gap; see §4, §8. |
| Multi-property read over one selection with one shared schedule | nowhere | MISSING, NOT YET ADMISSIBLE | A `Query` carries one `Agg`. Legal composition exists (`Keep` R once, then one program per property over `Plane(R) ∧ Presence[p]`). P1 shows the gain is block skipping, not sharing R's walk (§8). |
| DR256 mask reused across thoughts | #1791 arm D | REQUIRES-CONTRACT | Scattered mask read by later programs = the forbidden shape. |
| u16 map read in place for scatter | ndarray `mask_scatter_or_u16` (#377) | ALREADY (ndarray), NOT EXPOSED in mask-risc | `LaneRef::U16` designed, not implemented (#1791). |
| DR64 vs DR256 switch | #1791 measurements | LAB | |
| Selection vector | — | REJECTED (fold law 5) | |

## 3. DR-aware benchmark profile

Every WHERE / masked-read probe uses this profile. Correctness is always a whole-mask or whole-answer
comparison against a BTreeSet (or equivalent) oracle, never a popcount alone.

- **Universe:** 65,536 rows (DR256) and 4,096 rows (DR64 focus).
- **Cardinalities:** 0, 1, 63, 64, 65, 255, 256, 257, 4,096, 32,768, 65,536. These sit on the 64-row
  word and 256-row block boundaries.
- **Geometries:** contiguous; uniform; clustered (64-row runs); HHTL prefix (inside one 4,096-row
  prefix); collisions (many map entries to one global address); many maps (64 registers); skewed
  presence (property density from 64/65 to 1/65); adversarial pages (one row per page).
- **The two 4,096 fixtures:** dense (64 words, 16 blocks) and scattered (one row per 16; 1,024 words,
  256 blocks). Same cardinality, opposite occupancy. Any predictor that cannot tell them apart is not
  a DR-aware predictor.
- **Property counts:** 1, 2, 4, 8, 16, 32, 64.
- **Thought counts N:** 1, 2, 4, 8, 64, compatible and incompatible (selection change, seal change).
- **Report:** ns per op, µs per batch. A workload above 1,000 µs is a failed gate and is reported only as
  "exceeds 1,000 µs". Two runs; ratios within one run only. CPU features printed.
- **Disable runs:** commit first; each guard disabled once; red-then-green or the guard is not evidence.

## 4. Minimal Quack WHERE / masked-read interface proposal

Conceptual shape (from the request), **not** a new public type:

```text
DR256::where(criteria) -> PopulationSelection {
    popcount, active_words, active_blocks, optional_mask, witness }
SelectionStats { popcount: u32, active_words_64: u16, active_blocks_256: u16 }
```

How each field is expressed today, without new ABI:

- `optional_mask`: `Query { filter: criteria, agg: Agg::Rows }` → `Terminal::Keep`.
- `popcount`: `popcount_batch_u64` over the `Keep` output (re-exported `ndarray::simd`), or
  `Agg::Count` when no mask is needed.
- `witness`: the existing `PrefixLowering::Bound` witness (`lane_version`, `lane_digest`) for range
  reads. For a DR256 mask derived through maps, the #1791 witness fields (source, scope, universe, every
  map id+seal, every register selection generation). The latter is a contract proposal, not available.
- `active_words`, `active_blocks`: **the one missing piece.** Smallest admissible form: a pure ndarray
  T1 fold, `mask_occupancy(words) -> (u64, u32, u32)`, with the usual backend + parity + disable
  discipline. Admissible only once P2 shows a consumer decision actually reads it (§7). It is not a
  mask-risc terminal: law 6 (no convenience opcode) holds while `Keep` + a T1 fold expresses it.
- Multi-property masked reads: one `Keep` program for R, then one program per property over
  `Plane(R) ∧ Presence[p]`. Logical reads = Σp popcount(R ∧ Presence[p]). No `Agg` widening.

## 5. Measured crossover criteria (DR64 ↔ DR256)

From #1791 (two runs, avx512f, 4 cores):
- `count` needs a global dedup; a DR256 mask built once pays from **N\* = 2** compatible thoughts on every
  fixture with a selection.
- Rebuilding the DR256 mask per thought at chaotic or full density fails the 1,000 µs gate at N = 64.
  Per-thought rebuild is never the policy at those densities.
- `any` stays LOCAL (3–21 ns per thought on 50 % dense X) unless the selection is empty or a hit is late;
  sparse X was not measured.
- Incompatible thoughts gain nothing from reuse (B ± 15 %); the witness check costs about 1 µs.
- Keeping one live 8 KiB mask does not slow the resident DR64 fold; rebuilding does (+11.8 % / +13 %
  clustered).

From P1 (§8), the block-skip criterion: a shared block schedule pays when **active_blocks / 256 is
small** and stops paying near full occupancy (B/S ≈ 0.73–1.02 at 256 of 256 blocks). Cardinality alone
does not predict it: the two 4,096 fixtures give B/S ≈ 0.24–0.33 (dense) and 0.63–0.95 (scattered).

**Still unmeasured:** the same criteria through the real Quack/mask-risc path (P2); sparse-X `any`;
hardware-counter confirmation of the L1 result.

## 6. DuckDB harvest (D1–D7)

Sources verified in this session (DuckDB docs and the cited research): zone maps are automatic per
row-group min/max pruning; ART index scans serve equality / IN / dynamic filters above
`MAX(2048, 0.001 × cardinality)` rows, single column; RPT+ is research on v1.3.0; CTEs are materialized
by default and inlined only when referenced once; the optimizer includes CSE and filter pushdown; the
recursive-CTE design separates reusable operator state from the per-epoch frontier and retains a build
only after a repeatability check.

| id | DuckDB technique | our counterpart | class |
|---|---|---|---|
| D1 | zone maps (per-block min/max pruning) | address-prefix `Range` with witnessed `lane_version` / `lane_digest` (`lib.rs:305`) | ALREADY for address ranges; per-block **value** min/max descriptor MISSING, admissible only if measured |
| D2 | predicate transfer / Bloom (RPT+) | exact fk semijoin `EqU32Via` / `Semijoin → Gather` | ALREADY (exact); Bloom NOT-APPLICABLE (addresses are exact) |
| D3 | selective late materialization | masked terminals read only survivors; per-property block skip measured in P1 | ALREADY (single property); shared block schedule LAB |
| D4 | CTE materialization / CSE | `Keep` → `Plane` reuse across programs | ALREADY for `Keep`; scatter reuse REQUIRES-CONTRACT (#1791) |
| D5 | recursive CTE: reusable state vs per-epoch frontier | compiled programs reused; per-thought selection separate (#1791 D) | LAB; retain-after-repeatability matches #1791's witness rule |
| D6 | adaptive filter ordering, index vs scan by threshold | `and_by_skip` (`lib.rs:752`); witnessed Range vs sweep | ALREADY; selection vector REJECTED (law 5) |
| D7 | dictionary / selection-vector execution | DR mask + sealed u16 map read in place | ALREADY (ndarray u16 scatter); exposing it in mask-risc REQUIRES `LaneRef::U16` |

A DuckDB component existing is not evidence that the matching DR path is faster; every MISSING row
needs a measurement before code.

## 7. PR ladder (dependency-ordered)

Each rung is its own PR and needs its own authorization. A rung that fails its falsifier stops the
ladder at that rung.

- **P0 — census.** DONE (this document).
- **P1 — occupancy and shared-schedule lab.** DONE, lab only (`drf-focus-probe/examples/where_occupancy.rs`).
- **P2 — the same questions through the real Quack path.** Build R with `Agg::Rows`, then one program per
  property over `Plane(R) ∧ Presence[p]`; measure against the profile in §3. Falsifier: if mask-risc's
  own skip already makes per-property programs track active blocks (B/S of P1 reproduced without a
  shared schedule), the shared-schedule gap closes as NOT NEEDED.
- **P3 — `mask_occupancy` in ndarray.** Only if P2 names a consumer decision that reads active words or
  blocks (strategy choice LOCAL/GLOBAL, or block-schedule on/off). Backends + parity check + disable
  test. Falsifier: a SIMD version must stay within a stated bound of `popcount_batch_u64` on the profile.
- **P4 — `LaneRef::U16` in mask-risc.** After ndarray #377 merges. Width as a bind fact; no new terminal.
- **P5 — witnessed derived population.** Only if the #1791 contract proposal is authorized. Until then,
  no cross-thought reuse of a scattered mask anywhere.

## 8. Implementation and test evidence (P1)

`crates/drf-focus-probe/examples/where_occupancy.rs`. Two runs, avx512f, 4 cores, median of 301.
Every fixture's popcount, active words and active blocks equal a BTreeSet oracle; every multi-property
sum and logical-read count equals a per-row oracle; both arms agree.
Disable runs (red then green): counting a block per 4 words regardless of content fails the active-blocks
oracle; dropping block 0 from arm B fails the arm-B sum oracle.

**Q1, occupancy (ns per 8 KiB mask).** `popcount_batch_u64` costs 99–119 ns on every fixture. The lab's
scalar one-pass occupancy (popcount + active words + active blocks) costs 340–380 ns. So the statistic
adds about 240 ns per mask in scalar form. A SIMD form is P3's question, not claimed here.

**Q2, multi-property reads (B = shared block schedule, S = P full passes; ratio B/S, run 1 / run 2).**

| fixture | active blocks | B/S at P=1 | B/S at P=64 |
|---|---|---|---|
| contiguous 256 | 1 | 0.045 / 0.048 | 0.017 / 0.022 |
| uniform 256 | 158 | 0.389 / 0.479 | 0.474 / 0.500 |
| clustered 4,096 | 56 | 0.355 / 0.329 | 0.317 / 0.368 |
| 4,096 dense (64 words) | 16 | 0.325 / 0.301 | 0.235 / 0.238 |
| 4,096 scattered (1 per 16) | 256 | 0.713 / 0.709 | 0.950 / 0.883 |
| uniform 32,768 | 256 | 0.994 / 0.993 | 0.880 / 0.729 |
| contiguous 65,536 | 256 | 0.841 / 1.020 | 0.973 / 0.944 |

Absolute times differ between runs by up to about 1.7× on this shared machine; only within-run ratios
are used.

Findings:
- The ratio is roughly flat in P. The gain comes from skipping dead 256-row blocks, not from sharing R's
  walk across properties. Finding R's active blocks once costs one pass over 1,024 words.
- S's cost already scales with density (2.0 µs at uniform 256, P=1, against 16.8 µs at full density), so
  ndarray's masked fold already pays less on sparse masks. The mechanism was not inspected.
- At P=64 over dense or full selections, both arms exceed 1,000 µs: failed gate. Dense whole-population
  reads of 64 properties are outside the fold budget at this universe size on either schedule.
- Limit: this is ndarray-level. mask-risc's own behaviour inside a tile is not exercised, so S may
  overstate what a Quack program pays. That is exactly P2.

## 9. Review blockers

1. **Scatter survival condition.** No design here reuses a `ScatterOrU32` mask. Any rung that would
   (P5) is blocked on authorization of the #1791 witnessed-derived-population contract.
2. **Witness.** A DR256 mask derived through maps has no lawful witness carrier today. `PrefixLowering`'s
   witness covers ranges only.
3. **S-INGRESS and the joint coordinate witness (#1788 A.9 / A.10).** Still operator decisions;
   untouched here.
4. **`mask-risc` pins ndarray `master`.** P4 waits for ndarray #377.
5. **Unmeasured on the real path.** No P1 number may be quoted as a Quack-path speed-up until P2 runs.

MIRROR
- BLIND SPOT: mask-risc's tile-internal dead-word handling in terminal folds was not read in full; P2
  measures it instead of assuming it.
- BIAS CHECK: the census first recorded `Any` as early-exit from an earlier session's note; reading
  `exec.rs:1793` showed it is not on the generic path. Treat other remembered properties the same way.
- STILL OPEN: whether any consumer has a compatible thought window of N ≥ 2 in practice.
