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

---

# Amendment A (2026-10-26) — DR64³ holographic binding, named intermediates, derived relational views

**Status:** GAP INVENTORY + STAGED PLAN, documentation only. Nothing here is implemented or authorized.
**Builds on, does not restate:** D-NFR-0 seam table S1–S17a (`.claude/plans/d-native-repr-contract-v1.md`,
#1753), D-RMAP-1A (RelationshipMap tables), D-DRAV-1 (map/value lifetimes), D-DRF-1 (focus lease), D-RPR-0
(perspective rebinding), D-QR-CAP (fixture matrix), H14c/H14d. Where a row below says "S*n*", the evidence is
that D-NFR row, re-checked in this session where marked "read".

## A.0 Premise, and how it is recorded

The operator direction of 2026-10-26 is: **DR #1 Identity** (selected identity population, sealed DR64 focus)
and **DR #2 Properties** (selected property vocabulary of ≤ 64 property IDs, ClassView-typed) are materialized
foundations; **DR #3 Identity × Properties** is the working relational view, bound and not necessarily
materialized; **Identity × Identity** (a named relation, optional declared inverse) and **Properties ×
Properties** (higher-order statistics) are optional, on-demand views.

Recorded as `WORKING-MODEL` (direction), not as verified fact. It gives a concrete reading of D-NFR's open
point NFR-OP-2 ("what the DR64³ third axis indexes"): DR #3 itself is a two-axis register over typed
(Identity, Property) axes, and the third axis indexes a **relation ordinal** for the Identity × Identity view.
Closing NFR-OP-2 formally is still the DR owner's decision.

## A.1 Implementation status (verified this session)

| item | status | source |
|---|---|---|
| canonical SoA identity rows | SHIPPED | 512-byte node, `canonical_node.rs` (unchanged here) |
| DR #1 sealed global→local focus map | LAB | S11 `dir-rmap-probe/src/focus.rs`; `drf-focus-probe` (`SealedMap`, `FocusKey`); no production crate defines either (closed search over `crates/*.rs`: only probe crates and one planner example) |
| DR #2 property vocabulary and presence | SHIPPED (contract) | `class_view.rs`: `FieldMask` (64 presence bits, "presence, NEVER semantics", C2), `WideFieldMask` past 64 |
| DR #2 typed property axis in a binding | LAB | S8 `dr-math-binding-probe`: `AxisKind::{Identity, Property, Community, Capability}` (read) |
| DR #3 Identity × Properties as a typed register | LAB | S8 + S9: `MethodSignature { row: Axis, column: Axis, truth, population }` over a borrowed `[u64; 64]` (read). DR64², no third axis |
| DR64³ third axis | PROPOSED, no type | S10; NFR-OP-2 open |
| frame (source × relation map × coordinates) with stale refusal | LAB | S6 `FrameStore` / `Frame`, compared by issue, never by value (read) |
| focus and scope as separate guards | LAB | S7 `Guards`, `bind_dr64` (read) |
| Quack binding membrane (names → numeric, once) | SHIPPED | S1 `lance-graph-quack/src/bind.rs:288` `bind(&dyn Binder) -> Result<Query, BindError>` (read) |
| prepared program reuse | SHIPPED | S2 `CompiledProgram` (`mask-risc/src/ir.rs:1427`), `execute_compiled` (`exec.rs:1469`) (read) |
| a bound `Query` knows its frame | MISSING | S5 × S6, D-NFR §4 (the smallest missing seam) |
| shared temporary planes across programs | ALREADY for `Keep` (`Filter::Plane`); FORBIDDEN for scatter output (`ir.rs:329-337`) | §1 above |
| native per-property masked read | ALREADY, one property per `Query` | `Agg` → masked terminals, `quack/src/lib.rs:2421-2466` (read) |
| grouped univariate / bivariate moments | SHIPPED in mask-risc, **not reachable from Quack** | `Terminal::GroupPowerSumsI32` (`ir.rs:465`), `GroupCrossPowerSumsI32` (`ir.rs:480`); `terminal_of` has no route to either (read). Sums are exact: `u128` / `i128` accumulators (`ndarray simd_masking_ops.rs:2194`, `:2448`) |
| relationship tables | PROPOSED / LAB | D-RMAP-1A PROPOSAL; D-RMAP-1A-P, D-MPS-8, D-RMAP-DIR-1 MEASURED probe only; D-RST-2 `RelationshipMapStore` Queued, lab only |
| prefetch of the next focus | LAB / NOT AUTHORIZED | D-DRF-1 lookahead (lab); D-RPR-1 not authorized |
| #1790, #1791, #1792, ndarray #377 | open drafts | none merged |

## A.2 Named intermediates — four lifetimes, existing carriers

| lifetime | existing carrier | invalidation | gap |
|---|---|---|---|
| 1. bound, unevaluated expression | a Quack `Query` (S1), or a `BoundDr64` (S8, lab) | `Query`: none today (S5 × S6 missing); `BoundDr64`: frame/focus/scope re-checked per borrow | D-NFR-1 adapter: `Binder` names the `FrameRevisions`, binding returns `Query` + sealed `Frame`, every run checks the frame before lowering |
| 2. prepared relational view | `CompiledProgram` + an S11 focus + a `BoundDr64` | as 1, plus the focus revision | none beyond D-NFR-1 for the Quack half; the DR half is lab |
| 3. computed mask or reduction | `Keep` output in caller-owned `Out::Mask` / `Scratch`; a `Value` | its lifetime is the caller's buffer; reuse only as a plane under the same frame | a scattered mask has no lawful reuse (#1791 proposal) |
| 4. explicitly materialized output | a consumer-owned buffer at the egress (`materialize*`-style naming) | owned by the consumer | none |

Rules: no new public type, no result store, no new versioning authority. Mutable guards are re-checked at
every use (D-NFR §3); a cached binding never caches a guard away. `quack.bind(identities, properties,
criteria)` is therefore **lifetime 1 + the D-NFR-1 frame**, not a new object.

## A.3 Population WHERE

Unchanged from §2 and §4: `Agg::Rows → Keep` returns the selection; `Count` / `Any` answer without
producing a global mask. The one missing statistic is occupancy (active words and blocks), which waits for P2.
`ScatterOrU32` stays a demanded result only.

## A.4 DR64 versus DR256: which strategy inputs exist today

| input | available? | where |
|---|---|---|
| exact survivor count | yes | `Agg::Count` / `popcount_batch_u64` |
| occupied 64-row words, 256-row blocks | no (lab only) | §8 P1; P3 gap |
| distinct canonical identities | partly | `CountKeyRunsU32` over a key-ordered lane; scatter-count over a map (#1790) |
| DR64 focus coverage | lab | S11 |
| legality and cost of rebinding | lab | S6/S7 guards; D-RMAP-1A-P incremental rebind (probe) |
| selected property count, widths, offsets | yes | `Binder` + `Planes` lanes + ClassView basis |
| SoA / cache / page locality | **no predictor** | P1 (#1789) FAILED: chunk-touched share does not predict Lance bytes; H14d: bytes follow page locality |
| compatible future readers N | lab | #1791 (N* = 2 for count) |
| materialization and invalidation cost | measured per case only | #1790, #1791, D-DRAV-AV1 (~2 µs map-only vs ~13 µs value gather) |

Consequence: no policy may switch on popcount, and none may claim a physical-locality predictor. A selection
of 4,096 identities fits one DR64 focus by count while costing 0.63–0.95 of a full pass if scattered (§8).

## A.5 Whole-population selected-property reads

| step | today | status |
|---|---|---|
| 1. resolve property IDs at BIND | `Binder` (S1) | ALREADY |
| 2. layout, presence, validity | layout: `Planes` lanes; ClassView presence (C2, structural); Quack validity planes (SQL 3VL, `Filter::sql_where`) | ALREADY, and they are **different concepts**: presence ≠ SQL validity |
| 3. admitted population mask | `Keep` | ALREADY |
| 4–5. walk live words, skip zero words/blocks | ndarray masked folds cost less on sparse masks (§8: 2.0 µs vs 16.8 µs); mask-risc tile behaviour not measured | P2 |
| 6. read payload only at surviving valid addresses | filter `R ∧ Presence[p]` + masked terminal, one property per `Query` | ALREADY (per property) |
| 7. reuse row-local reads across properties | no multi-`Agg` query | MISSING; P1 says the gain is block skipping, not sharing, so do not build it before P2 |
| 8. fused reductions | masked terminals; H14d fused Σ ≈ 5× under the explicit payload terminal | ALREADY |
| 9. materialize only at egress | lifetime 4 | ALREADY |
| source-side masked read | Lance 13 row-address prefilter (H14c/H14d) | MEASURED, gated on S-INGRESS (#1788) |

Open gap, not yet sized: **per-property presence planes over a population.** ClassView presence is per row
(`FieldMask` on the SoA row). H14c/H14d build presence planes `P_l` inside their probes. No production
transposition from per-row `FieldMask` to per-field population planes was found; the search was not closed,
so this is OPEN, not MISSING. Logical reads stay Σp popcount(R ∧ Presence[p]); H14d proves
"payload reads == Σ_l popcount(R ∧ P_l)" counted, and that logical pruning does not bound Lance bytes.

## A.6 Identity × Identity (named relation, declared inverse)

- **Precedent, all lab or proposal:** D-RMAP-1A (separate RelationshipMap tables, PROPOSAL); D-MPS-8 (native
  join over a versioned relationship table, leapfrog as one mask-risc `And`, MEASURED probe); D-RMAP-DIR-1
  (DR256 register: 256 × 256 bit matrix, `Carriers` "held by the register in both orientations",
  `dir-rmap-probe/src/compose.rs:153`; arm B a bidirectional CSR).
- **An inverse read is a physical access path, not a bit.** A bidirectional CSR is a maintained second index
  (a cost, D-HOLO H5). Without one, the inverse is a full sweep with the forward predicate, or a refusal.
- **Admission rule for an orientation bit:** only on a relation binding whose schema declares a valid inverse
  (e.g. `causes` / `caused_by`), and only where an eligible reverse access path exists in the same frame.
  The bit names the reading direction; it never asserts symmetry, causality, invertibility of an arbitrary
  predicate, or a reverse index that does not exist. Canonical edges are never duplicated to reverse a read.
- **Gap:** no typed inverse declaration was found in the contract (`schema.org inverseOf` appears only in an
  ontology hydrator, `hydrators/schemaorg.rs:31`); not a closed search, so OPEN.

## A.7 Properties × Properties (higher-order statistics)

- **Boolean co-occurrence (XᵀX entry a,b):** `Count` over `Plane(R) ∧ P_a ∧ P_b`, one program per requested
  pair (fused ternlog popcount). ALREADY. All-pairs is a reference arm only, never the default.
- **Numerical covariance / Pearson:** mask-risc `GroupCrossPowerSumsI32` folds `n, Σx, Σy, Σx², Σy², Σxy`
  exactly in one pass, both lanes read in place. Pairwise sample eligibility is the mask
  `R ∧ Valid[x] ∧ Valid[y]`. Correlation and covariance are consumer projections of those sums.
- **Gaps:** (a) no Quack `Agg` reaches the power-sum terminals; adding one is a front-end mapping to an existing
  terminal, not a new opcode (law 6 holds). (b) the terminals are grouped; an ungrouped request would need a
  constant key lane, which is a 256 KiB materialization over 65,536 rows. The smallest form needs measurement
  against that lane before it is proposed.
- Co-occurrence is never reported as correlation, and neither is evidence of causality.

## A.8 Prefetch and behaviour binding

Three things are prefetched separately and never implied by each other:

1. **sealed addresses / relationship bindings** — D-DRAV-AV1: map-only membership ~2 µs vs ~13 µs value
   gather (lab, #1776).
2. **selected property payload** — D-DRF-1 lookahead (lab); D-DRAV-AV2 (rehydrate exact snapshot or refuse)
   NOT RUN.
3. **disposable derived values** — D-DRF foreign cache, scoped exception undecided.

`lance_graph_contract::property::PrefetchDepth` is a schema-level property-visibility tier (Identity / Detail
/ Similar / Full) by property kind. It is not a physical prefetch and must not be cited as one.
`alpha_focus` is an attention mask, not DR focus (D-NFR note). Preparing the next perspective is D-RPR-1,
NOT AUTHORIZED. Later behaviour lowering reuses the same lifetime-1/2 bindings; no behaviour VM, ONNX
interpreter or dynamic dispatcher.

## A.9 DuckDB harvest, priority A/B/C

DuckDB references are the ones pinned in #1788 A.2 (`TryLateMaterialization`, `RowGroup::CheckZonemap`, at
`1641475`) plus the DuckDB documentation verified in §6. "Owner" is the layer a change would land in.

| technique | DuckDB ref | our equivalent | missing | semantic gate | falsifier | owner | decision |
|---|---|---|---|---|---|---|---|
| **A** projection pushdown | projection in scans | `Binder` resolves only named fields; masked terminals read only named lanes | nothing at the fold; Lance-side column projection under S-INGRESS | read set == requested property set | bytes read ∝ selected properties, not table width | Quack / source owner | KEEP |
| **A** zone-map pruning | `RowGroup::CheckZonemap` | witnessed `Range` on ordered lanes (S3) | a per-block **value** min/max statistic with a seal | statistic sealed with the lane version; stale → sweep | pruned blocks × answer parity vs sweep; gain on a real predicate stream | mask-risc + lane owner | ADAPT, only if measured |
| **A** selective late materialization | `TryLateMaterialization` | masked terminals over survivors; lifetime 4 | per-property block schedule (P2) | values read only at `R ∧ Valid[p]` | P2 B/S vs per-property programs | Quack / mask-risc | ADAPT (P2 first) |
| **A** adaptive predicate ordering | `AdaptiveFilter` | `and_by_skip` (`lib.rs:752`) | dead-word/dead-block measurements as its input | order never changes the answer (law 3) | shuffled-data control | Quack | KEEP |
| **A** CSE / prepared reuse | optimizer CSE, prepared statements | `Keep` → `Plane`; `CompiledProgram` | frame on the bound query (D-NFR-1) | same frame, same answer | stale frame refused | Quack + frame owner | ADAPT (D-NFR-1) |
| **A** filter propagation to source | dynamic filters, scan pushdown | Lance row-address prefilter (H14c/H14d) | S-INGRESS ruling; coordinate authority | rows scanned == `|R|` | bytes vs locality (P1 FAILED as predictor) | source owner | DEFER (S-INGRESS) |
| **A** operator / terminal fusion | fused pipelines | fused ternlog lowerings, fused Σ | none found | fused == explicit | H14d fused vs explicit | mask-risc | KEEP |
| **B** join-filter pushdown over SoA relations | join filters | `EqU32Via` / `Semijoin → Gather` (exact) | none for addressed keys | exact fk | — | Quack | KEEP |
| **B** index vs masked scan | ART threshold `MAX(2048, 0.001·N)` | witnessed `Range` vs sweep | none | — | — | Quack | KEEP |
| **B** sorted physical read batching | scan ordering | none at the fold; Lance ranges | page-scale clustering (H14d OPEN) | order never changes the answer | bytes at equal `|R|` | source owner | DEFER |
| **B** reusable state across thought frontiers | recursive-CTE operator state | `CompiledProgram`; #1791 D (lab) | #1791 contract for scattered masks | witness per window | #1791 N* | Quack + DR owner | DEFER (contract) |
| **B** exact selectivity feedback | adaptive filter stats | exact popcount; occupancy gap | occupancy (P3) | stats never change answers | P2 consumer decision | ndarray | ADAPT (P3) |
| **B** costed intermediate planes | CTE materialization | `Keep` → `Plane` | a cost rule | — | reuse vs recompute at N | Quack | DEFER (P2) |
| **B** factorized / shared sufficient statistics | factorized aggregates | `GroupPowerSumsI32` / `GroupCrossPowerSumsI32` | Quack `Agg` route; ungrouped form | exact i128/u128 sums | vs independent f64 oracle and exact integer oracle | Quack | ADAPT (PR E) |
| **C** Bloom filters | RPT+ (research) | exact masks / gathers | — | — | — | — | REJECT (exact wins on addressed keys) |
| **C** general join-order planning | optimizer | none | — | — | — | — | REJECT (no second planner) |
| **C** adaptive cardinality estimation | statistics | exact counts | — | — | — | — | REJECT (exact counts exist) |
| **C** compression-aware scheduling | compressed execution | — | — | — | — | — | DEFER |
| **C** dictionary-vector reuse | dictionary vectors | sealed u16 map + mask (#377) | mask-risc `LaneRef::U16` | — | #1791 | mask-risc | DEFER (P4) |
| **C** top-N / early-out | top-N, limit pushdown | `Any` early exit on fused path only | early exit on generic path | result semantics allow it | C7 / G8 measurement | mask-risc | DEFER |

No SelectionVector, row iterator, hash execution engine or copied tuple store is proposed.

## A.10 Benchmark geometry additions

On top of §3: number of DR64 focus bindings; duplicate map targets; physical cache lines and pages touched
(byte counts, not wall time, for storage); Identity × Properties narrow vs broad projections; Identity ×
Identity forward vs declared inverse (with and without a reverse access path); Properties × Properties
requested pairs vs all-pairs reference; empty / missing / nullable values (presence and SQL validity
separately); stale binding, changed focus, changed property generation. Preparation, source reads, native
execution, amortized reuse and materialization are reported separately. Historical 1 % / 25 % / 100 %
fixtures stay as controls.

## A.11 Distinctions kept

Materialized foundation vs derived view; relation support vs proven fact; named intermediate vs payload copy;
global membership vs adjacency; thought version vs binding/focus generation; source-side masked read vs
resident mask operation; disposable cache vs authoritative persistence; presence vs SQL validity; population
count vs Bag multiplicity (`bind_dr64` refuses Bag). A bounded focus does not make unseen contributors false:
expand lawfully, use an eligible global route, or refuse (S11 refuses `OutOfFocus`).

## A.12 PR ladder A–F, mapped to existing work (none authorized)

| PR | scope | existing D-id it is | depends on | first gate |
|---|---|---|---|---|
| A | this amendment: roles, status, lifetimes, owners | — (docs) | — | review |
| A′ | frame on a bound Quack query | **D-NFR-1** (do not open a second one) | NFR-OP-1 owner decision | second run after a revision is refused |
| B | population WHERE with occupancy | capstone P2 → P3 | — | P2 names a consumer decision |
| C | selected-property masked reads through Quack | capstone P2; D-DRAV-AV2 for payload residency | B's P2 | Quack-path B/S vs per-property programs; independent oracle; byte accounting |
| D | DR64/DR256 amortization | **D-NFR-5** + #1791 | #1791 contract; ndarray #377 for P4 | reuse saving measured with guards on every use |
| E | derived views: Identity × Properties typing; Identity × Identity inverse; Properties × Properties `Agg` route | D-RMAP-1A / D-MPS-8 / D-RMAP-DIR-2 for relations; new only for the power-sum `Agg` route | A′; ISS-ADMISSION-PREP-BINDING for relations | exact sums vs integer oracle; inverse refused without an access path |
| F | DuckDB experiments, one per demonstrated gap | A.9 ADAPT rows | P2 | each beats existing native execution or is rejected |

Coordination: EXO-50 is the append-only handoff; D-QR-CAP's fixture matrix (`QR64` / `QR256`, ClassId 7
basis, `WideFieldMask` 0/63/64/108) is reused as the fixture source rather than duplicated.

## A.13 Open points

- NFR-OP-1 (frame store owner), NFR-OP-2 (third axis), NFR-OP-3 (one witness type) — unchanged, operator / DR owner.
- Per-property presence planes over a population: production source not found (A.5).
- Typed inverse declaration for relations (A.6).
- The #1791 witnessed-derived-population contract, S-INGRESS and the joint coordinate witness (#1788).

---

# Amendment B (2026-10-26) — P2: WHERE integration audit and the Quack-path measurement

**Status:** P2 DONE, lab only (`crates/drf-focus-probe/examples/p2_quack_where.rs`), reported, not gated.
P3–P5 stay unauthorized. No new opcode, selection vector, temporary-table store or query cache.

## B.1 Binding store: what exists

- **Quack `Binder` → `Query` → `Program`** (read): `Draft::bind(&dyn Binder)` (`quack/src/bind.rs:288`)
  resolves table, fields and textual literals once and returns a numeric `Query` that holds no name. The
  bound filter here is `And([Plane(live), Cmp(Col(0), EqI32(1)), Cmp(Col(1), NeU32(5))])`. `lower` /
  `lower_fused` (`lib.rs:1383`, `:1397`) produce a mask-risc `Program`; `CompiledProgram` (`ir.rs:1427`)
  keeps the recognised lowering. The `Draft` surface binds only `=` / `<>` and returns COUNT or rows;
  ordered compares and sums are built on the numeric `Query`.
- **Binding store — classification, corrected 2026-10-26 (§D supersedes the earlier "what exists" wording):**
  - the generalized Quack Binding Store: PLANNED / UNBUILT. The token `BindStore` occurs in no file in the
    history of any of the 880 remote branches. It is defined by D-QSP (PROPOSAL; "do not put a behavior
    choice into the binding store") and D-NFR S5, with D-NFR-1 as the smallest seam;
  - Quack numeric binding: IMPLEMENTED (this section);
  - RelAlg named-intermediate reuse: IMPLEMENTED (#1649, #1653);
  - versioned DR population reuse: IMPLEMENTED IN LAB (#1666 `Pop`);
  - typed DR method binding: IMPLEMENTED IN LAB (#1701, on main via #1708);
  - reusable compiled mathematical recipes: IMPLEMENTED IN LAB (#1693, #1694 via #1696/#1698, #1707, #1740);
  - source-attested production integration: STILL MISSING.
- **Role of the binding store:** a DTO / ClassView-to-native-algebra binding authority over folding algebra.
  It owns reusable binding descriptors, their witnesses and lifetimes, and connects the existing lifetimes
  (§D.3). It is not a query-result cache, a temporary table, a population bitmap store, a behaviour chooser,
  a graph database or an evaluator.

## B.2 RelAlg `SetProgram` lets versus Quack `Keep → Plane`

| | RelAlg `SetProgram` (`contract/src/set_program.rs`, `relalg/src/program.rs`) | Quack `Keep → Plane` |
|---|---|---|
| reuse scope | one `run_program` call: each let computed once into scratch, read by every query of that call | as long as the caller keeps the 8 KiB buffer |
| materialization | one `n_rows/8` plane per let, **allocated per call** (`vec![vec![0; words]; lets]`) | one caller-owned buffer, allocated once |
| witness | `version: u64`, compared by value; mismatch refused (`ProgramError::VersionMismatch`, measured) | none: a kept plane carries no witness |
| shared-part discovery | `share()` lifts common conjunct subsets into lets | caller decides |
| semantics | two-valued set algebra (K3 rewrites guarded in `set_rules`) | `Filter::sql_where` gives K3 over validity planes |

## B.3 lance-graph-java `View.where()`

`View` (`java/.../View.java`) is an immutable, lazy predicate list:
- `where` crosses zero times and only narrows; the ABI's OR combine is unreachable from `View`.
- `count()` is one crossing; `sumOf` / `minOf` / `maxOf` are two (evaluate, reduce).
- `select()` returns a native generation-checked `Mask` handle.

The native side lowers the plan straight to one mask-risc `Program` (`lgj-abi/src/plan_lower.rs`), not
through Quack. AND terms are gated `under` the accumulator; OR is deliberately not gated.

**Reuse limit:** Java `Mask` offers `count`, `minus`, `ternlog` and `materializeRows`, but no masked sum.
So a Java consumer cannot reuse one selection across property reads; each `sumOf` re-evaluates the chain
(the QA pattern below). Closing that needs a surface change, which is out of scope.

**Crossing cost** (recorded in `lance-graph-java/bench/RESULTS.md`, not re-measured here; there is no JDK in
this container): bare downcall 21.9 ns, `lgj_mask_count` 117.9 ns, ~0.6 µs plan-evaluation floor; native
beats a Java scalar loop above ~16,384 rows.

## B.4 mask-risc skipping, as executed

- **Gated predicates skip.** `*_to_mask_under` writes zero and skips the compares for an empty gate word
  (`ndarray simd_masking_ops.rs:4979` doc + `pack_under`). Quack and Java both emit gates.
- **Masked folds skip reads, not words.** `masked_sum_i32` (`:815`) visits every word but reads values only
  under set bits.
- **Mask ops and the tile loop do not skip.** `execute_extent` runs every op on every tile (`exec.rs:1633`
  onward). There is no whole-tile or 256-row-block skip.
- **`Any`** exits early only on the fused ternlog path (§2).

## B.5 Capability crosswalk (semantics are NOT identical across rows)

| | Quack | RelAlg | Java `View` / lgj-abi | mask-risc |
|---|---|---|---|---|
| predicates | numeric `Cmp`: I32 6 ops, U32 eq/ne/match, U64 match, strided, `Range`, `EqU32Via`, `Semijoin`; `Draft` text surface `=` / `<>` only | `Atom`: I32 6 ops, U32 eq/ne | U32 eq/ne, I32 6 ops, ternary match U32 | `Pred` (all of these) |
| Boolean | And / Or / Not; `sql_where` K3 | And / Or / Not; normalize; symbolic decide | AND narrowing only (OR unreachable) | And / Or / Xor / AndNot / Not / Ternlog |
| terminals | Count, Any, All, Sum/Min/Max I32, Rows, Blend, ScatterOr, CountDistinctOrdered, GroupSum(Via), GroupReduce — **no power sums** | Count, Sum, Min, Max, Keep | count, sum, min, max, select | all of these + Group(Cross)PowerSums |
| reusable intermediate | numeric `Query`; `CompiledProgram`; caller-kept `Keep` plane | lets within one call | immutable `View`; native `Mask` handle | `CompiledProgram`; `Out::Mask` |
| witness on reuse | none (S5 × S6) | `version: u64` by value | handle generation (use-after-close), not data version | none |

## B.6 P2 measurement (two runs, avx512f, 4 cores; ns, median of 101; all arms equal a per-row oracle)

Fixed costs: `Draft::bind` 199–203 ns; `lower` 201–203 ns; `lower_fused` 357–360 ns. Occupancy (lab
scalar) 341–476 ns per R.

**Selection creation.**
- `Keep(R)` into an 8 KiB caller buffer costs ~20 µs on every fixture, empty included (some cells
  ~35–39 µs in one run only).
- The floor is the first predicate: its gate is the all-ones live plane, so it compares all 65,536 rows.
- COUNT without a mask costs the same: the predicate work dominates, not the 8 KiB write.

**Property reads, run 1 / run 2:**

| fixture | active blocks | P | QA (recompute R per property) | QB (Plane(R) reused) | NB (block schedule) | QB/QA | NB/QB |
|---|---|---|---|---|---|---|---|
| empty | 0 | 8 | 182,483 / 182,894 | 17,698 / 17,512 | 56 / 77 | 0.10 / 0.10 | 0.00 / 0.00 |
| contiguous 256 | 1 | 8 | 192,709 / 189,486 | 17,635 / 17,646 | 447 / 447 | 0.09 / 0.09 | 0.03 / 0.03 |
| uniform 256 | 161 | 8 | 198,764 / 185,288 | 19,985 / 17,967 | 7,237 / 7,022 | 0.10 / 0.10 | 0.36 / 0.39 |
| clustered 4096 | 57 | 8 | 213,772 / 193,851 | 26,742 / 26,688 | 6,923 / 6,950 | 0.13 / 0.14 | 0.26 / 0.26 |
| dense 4096 (64 words) | 16 | 8 | 347,040 / 190,299 | 35,964 / 22,944 | 7,359 / 6,313 | 0.10 / 0.12 | 0.20 / 0.28 |
| scattered 4096 (1/16) | 256 | 8 | 408,813 / 239,441 | 86,086 / 58,839 | 63,964 / 44,999 | 0.21 / 0.25 | 0.74 / 0.76 |
| uniform 32768 | 256 | 8 | 441,027 / 349,031 | 101,805 / 107,175 | 100,741 / 107,105 | 0.23 / 0.31 | 0.99 / 1.00 |
| full 65536 | 256 | 8 | 293,556 / 449,012 | 105,916 / 126,118 | 104,734 / 107,537 | 0.36 / 0.28 | 0.99 / 0.85 |

Full P = 1..64 tables are in the probe output. Other arms:
- `lower_fused` equals `lower` within noise for both QA and QB; the 2-plane conjunction has nothing to fuse.
- relalg `run_program` (one let) costs one selection plus P reads plus a per-call plane allocation. That is
  56–216 µs at P=8, between QB and QA. `run_recompute` equals QA.
- **Failed gate (above 1,000 µs):** QA at P = 64 on all 8 fixtures and at P = 32 on 5–6 of them; QB at
  P = 64 on uniform 32768 and full 65536 (both runs) and scattered 4096 (one run). Those shapes are outside
  the fold budget.

**Controls** (scattered 4096, all asserted):

| control | result |
|---|---|
| stale codebook | the old binding still answers (4,096 rows vs 4,051 fresh), **silently** |
| scope change (half the live words cleared) | a reused R keeps 4,051 rows vs 2,023 fresh, **silently** |
| selection change | 2,023 vs 2,086, **silently** |
| property change NOT read by R's criteria | R stays valid if its other witnesses hold; the value read through it equals the fresh oracle, because the read uses the current lane and no value cache exists |
| property change read by R's criteria (R2 = R ∧ `val_0 >= 0`; `--controls` run, seed `0x0D0D_F00D`) | reused R2 has 1,032 rows vs 31 fresh, and a `val_1` read through it differs, **silently** |
| relalg under a newer version | refused with `VersionMismatch { program: 7, table: 8 }` |

Disable runs (red, then restored): dropping the presence plane from QB fails the QB oracle; keeping the
codebook unchanged fails the stale-codebook control's non-vacuity assertion; skipping the `val_0` mutation
fails the predicate-dependent control's non-vacuity assertion.

**Invalidation semantics (corrected).** "No value cache, so a property change is harmless" was too broad:
- a property change that does not influence R keeps R valid, provided R's other witnesses still hold;
- a property change that affects R's criteria, validity, presence, relation support or coordinate
  interpretation invalidates R; it must be recomputed (measured above);
- a property change that affects only a later demanded value must be read under a valid current binding;
  an old payload is never reused.

A dependency witness must cover exactly the inputs that determine the result: not every property, and not
only the population's row count.

## B.7 Decision evidence

1. **Reusing R is already available and is the large win.** QB is 0.05–0.60 of QA across every fixture and P
   (0.07–0.39 at P = 1 and 8): keep
   R once, read properties through `Plane(R)`. It needs no new code; it needs a witness (below).
2. **Existing execution does NOT deliver P1's block skipping.** At sparse selections, NB is 0.02–0.39 of QB
   (e.g. empty: ~2.2 µs per property vs 54 ns; clustered 4096: 3.3 µs vs 0.86 µs). At 256 of 256 active
   blocks NB/QB is 0.63–1.08, so a block schedule gives no reliable gain there. The gap is real and confined to
   sparse selections.
3. **The occupancy statistic is cheap enough as scalar code** (341–476 ns, about 2 % of building R), and its
   use is now concrete: active blocks decide whether a block-gated read pays. A SIMD ndarray fold is NOT
   warranted on cost. Whether a fold or a statistic emitted alongside `Keep` is the right shape is P3's
   question.
4. **A shared-schedule specialization is warranted only as gating, not as a new opcode.** The measured gap
   is that mask ops and masked folds do not consult a gate the way predicates already do. The smallest
   candidate is the existing `_under` discipline extended to the `And` + masked-fold pair over a kept R.
   It must be measured before it is proposed (P3), and it must keep "a gate never changes an answer".
5. **A witness on a kept R is the blocking correctness gap.** Every reuse control ran silently stale through
   Quack. RelAlg already refuses by version value; `dr-math-binding-probe` refuses by seal identity (lab);
   #1666's `Pop` refuses by witness equality (lab; this was omitted when first written, see §C.2).
   The binding store (D-NFR-1 / D-QSP) is the owner. Until it exists, a kept R may be reused only inside one
   caller-controlled window where the caller can guarantee frame, scope and selection did not move.
6. **Java** cannot express QB today (`Mask` has no sum). Its `sumOf` is the QA pattern at ~0.6 µs crossing
   floor plus the chain. Not changed here.

## B.8 Compiler-input requirement (recorded, not built)

The binding store must preserve reusable, version-witnessed semantic bindings (the numeric `Query`, the
frame revisions, the method signature, the codebook generation) as inputs to a mathematical compiler:
Datalog / egglog / DialEgg certification (D-NFR S12, `relcomp.rs` egglog proofs) and versioned
ONNX/TFLite conformance annotations, lowering through R2IL to existing kernels. Two constraints follow from
this measurement:
- a binding must be reusable **without** materializing intermediates: lifetime 1 does that, and the QB
  pattern shows one kept plane is enough;
- it must not become a second execution engine: every arm here ran on the one mask-risc executor.

---

# Future mathematical compiler compatibility: DR³, BindStore and traceroute of thought

**Status:** architecture compatibility note (2026-10-26). P2 implementation scope is **unchanged** and lab-only.
Nothing here authorizes the compiler, a binding store, a proof store or any new runtime object.
**Authorities referenced, not restated:**
- D-DRMC-0 `dr-mathematical-compiler-v1.md` (the compiler charter: three contracts, cold vs hot, staged falsifiers);
- D-NFR-0 `d-native-repr-contract-v1.md` and D-NFC `d-native-first-dependency-correction-v1.md` (native-first);
- D-ONNX-RELALG-CAP `d-onnx-relalg-cap-v1.md` (ONNX as an import membrane);
- D-DR3 `d-dr3-holo-onnx-s3a-s3c-v1.md`;
- D-RPR-0 `recursive-perspective-rebinding-v1.md`;
- D-QSP (binding store proposal).

## C.1 Pipeline

The pipeline is D-DRMC-0's "Cold vs hot" diagram. This document adds nothing to it:

```text
Datalog / MeTTa / egglog / DialEgg      cold: rules, equality saturation, dependency analysis
        ▼
derivation + proof (traceroute)         cold: why, and which transformation was certified
        ▼
typed semantic binding                  frame / focus / scope guards; ClassView method; DR axes
        ▼
native lowering                         Quack / R2IL / method route
        ▼
mask-risc / ndarray::simd               hot: the one executor
        ▼
demanded result + read receipt
```

Reasoning stages are cold, preparatory or explicitly scheduled. No egglog, Datalog or DialEgg dependency
enters an ordinary DR64/DR256 fold; the runtime consumes admitted operations and bindings only.

## C.2 What exists (all lab unless stated) and what is planned

| capability | status | source |
|---|---|---|
| numeric binding (names → numbers, once) | SHIPPED | Quack `Draft::bind` (`bind.rs:288`) |
| prepared program | SHIPPED | `CompiledProgram` (`mask-risc ir.rs:1427`) |
| typed DR64 class/method operand binding with witnessed Identity / Property axes, frame / focus / scope guards | LAB, merged | D-DRMC-1, #1701, integrated into main by #1708; `dr-math-binding-probe` (`MethodSignature`, `FrameStore`, `bind_dr64`) |
| law identity ≠ compiled recipe ≠ data binding, reusable across ClassIds | LAB, merged | D-DRMC-3, #1703 (stacked; on main via #1708); `recipe.rs` (`LawShape`, `LawReference`, `RecipeCertificate`, `admit_recipe`) |
| certified recipe reused on a fresh frame without rerunning egglog | LAB, merged | #1740 / X8: `execute_certified` refuses `WrongTarget` / `NotCertifiedQuery` / `NotThisChain`; 0 egglog calls on the hit path; hot reuse 2,142 ns vs 7,216 ns plain eval; cold certify reported 15,696 µs (cold, not a gate) |
| version-witnessed reusable populations and method folds over a DR256 register | LAB, merged | #1666 `dir-rmap-probe/src/pop.rs`: `Pop { witness, tiles, kept }`; `and` / `and_not` / `or` refuse `Stale` on a witness mismatch; `repair` re-derives after edits; methods lowered through relalg `Rel::Plane` |
| zero-copy relation lanes; a side as an index map `$i` | LAB, merged | #1659 `mork-pathmap-probe` / `crepe-plan-probe` (`RelationView`, `Side`, `JoinDomain`) |
| logical traceroute of one derivation (premises → laws → saturation → extracted → proof → R2IL → value → oracle → verdict) | LAB | `egglog-r2il-probe/src/trace.rs` (`HOPS`) |
| replay capsule for E1 substitutions | LAB | Stage B `ReplayCapsule`: replay and audit evidence, **not** a proof authority (`dr-math-binding-probe/RECIPE_ADMISSION.md`) |
| generalized production binding store | PROPOSED | D-QSP; D-NFR S5 MISSING; smallest seam D-NFR-1 |
| independent proof checker | MISSING | #1740 "no independent proof checker yet"; D-DRMC-0 boundary 2 |
| passive proof reference in a contract | PENDING agreement | #1740 option B, to be agreed with CC1 before any contract PR |

**Correction to Amendment B.7(5):** a witnessed kept population already exists as a lab precedent. #1666's
`Pop` refuses a combination across versions by witness equality. P2's finding stands for Quack itself: a
`Keep` plane passed as `Plane(i)` carries no witness and ran silently stale in all three controls.

## C.3 Binding store and named-intermediate ownership

The binding store is the intended owner of named, reusable binding **descriptors**: their witnesses,
lifetimes and dependencies. It is not a data store or an evaluator, and it carries no behaviour choice
(D-QSP). The four lifetimes (A.2) keep separate owners:

| lifetime | today's carrier | owner |
|---|---|---|
| 1. bound expression | Quack `Query`; `BoundDr64`; relalg `SetExpr` / `SetProgram` | binding store (planned) |
| 2. prepared computation | `CompiledProgram`; `RecipeCertificate` + compiled steps | mask-risc for programs; the certified-compiler boundary for recipes (D-DRMC-0) |
| 3. computed intermediate | `Keep` plane in caller scratch; relalg let plane (one call); `Pop` (lab) | the caller's buffer, under the descriptor's witness |
| 4. materialized result | consumer egress buffer | consumer |

None of these is forced into one cache entry. A descriptor references a computed intermediate; it does not
own it.

## C.4 ONNX / TFLite annotation boundary

Per D-NFC A1/A2 and D-ONNX-RELALG-CAP:
- **They provide:** operator identity and version, type/shape constraints, numerical interpretation (quantization, rounding, saturation), eligible native mappings and differential-test references.
- **They do not own:** the internal graph, symbolic inference, graph identity or causal semantics.
- **Internal authority:** native typed expressions, R2IL routing and ndarray.
- **Not added:** a tensor AST, an ONNX runtime dependency, a generic dispatch engine or an ONNX → RelAlg graph.

## C.5 Derivation ≠ proof ≠ binding ≠ execution ≠ replay

| stage | answers | existing precedent | never mistaken for |
|---|---|---|---|
| derivation | why an expression was obtained | `egglog-r2il-probe` trace hops 1–4 | a proof |
| proof | which transformation was certified, under which assumptions | egglog `ProofStore` + `ProofId` (`relcomp.rs`), `RecipeCertificate` | an independent check (none yet) |
| binding | which data, identities, coordinates and versions | `Frame` / `Focus` / `Scope`, `Pop.witness`, `SetProgram.version` | a data copy |
| execution | which compiled operation ran and its result | mask-risc `Value`; `execute_certified` | a durable record |
| replay | whether the result can be reconstructed | `ReplayCapsule` (E1 only); oracle hops 7–9 | proof authority or a Thought-Git commit |

A future traceroute references these stages through immutable identities. It stores no intermediate
population, symbolic graph or duplicated SoA payload. Provenance and proof witnesses stay off the hot DR
register coordinates.

## C.6 Connection to DR256::WHERE

WHERE remains population selection plus an execution decision. Popcount, active words and blocks, and
witnessed geometry inform both query scheduling and later compiled-method scheduling. The DR64 vs DR256
choice is a physical strategy only: in P2 the same `Query { And[Plane(R), Plane(pres_p)], SumI32 }` produced
identical answers under every arm, including the non-Quack block schedule. That confirms a selected-property
read is expressible without binding the consumer to a physical strategy, for exact integer folds. Declared
numerical semantics (D-NFR S14, NFR-OP-4) govern any approximate fold.

## C.7 P2 compatibility checks (answers)

1. **Binding / binding-store ownership and relalg let reuse:** B.1, B.2, C.2. The binding store does not
   exist; Quack binds numerically; relalg lets are per call with a value-compared version.
2. **Do bound numeric queries keep their semantic identity and witnesses?** Partly.
   - A `Query` keeps its numeric meaning and is reusable.
   - It holds no ClassId / method / law identity; that lives in `MethodSignature` and `LawShape`, which are lab.
   - It holds no witness: three P2 controls ran silently stale.
3. **Do reusable intermediate planes keep the dependencies needed for invalidation?**
   - Quack `Keep` planes: no.
   - relalg let planes: a version value, for one call only.
   - #1666 `Pop`: yes, by witness equality (lab).
4. **Are selected-property reads independent of the DR64/DR256 strategy?** Yes, for exact integer folds (C.6).
5. **Missing hooks** (documented, not implemented; each with its existing owner):
   - a frame on a bound query — D-NFR-1, owner decision NFR-OP-1;
   - a dependency witness on a kept plane — the binding store, generalizing `Pop`-style witnesses; D-RMAP-1A-V for map / palette generations;
   - codebook generation in the frame — D-NFR-2;
   - a passive proof reference — D-DRMC, pending the CC1 agreement;
   - an independent proof checker — D-DRMC-0 boundary 2;
   - a Quack `Agg` route to the power-sum terminals — A.7.

No metadata field or runtime object was added to satisfy a future feature.

## C.8 Unresolved admission gates

NFR-OP-1/2/3, ISS-ADMISSION-PREP-BINDING, the #1791 witnessed-derived-population contract, S-INGRESS and
the joint coordinate witness (#1788), the independent proof checker, and the CC1 proof-reference agreement.
The scatter survival condition (`ir.rs:329-337`) is unchanged.

**P2 scope is unchanged:** lab-only measurement of existing Quack / mask-risc WHERE and multi-property
masked reads. P3–P5 remain unauthorized.

> Reason before materialization. Bind meaning to witnessed coordinates. Reuse the derivation and compiled
> mathematics. Execute against native DR/SoA data only when demanded. Materialize only what the result
> requires.

---

# D. Evidence reconciliation (2026-10-26): binding store, DR³, P2

Corrections already applied in place: B.1 (binding-store classification), B.6 (three-way invalidation
semantics + the predicate-dependent falsifier), B.7(5) (`Pop`), C.2 (correction note). This section adds
only the implementation spine, the missing seam and the authorization boundary.

**Placement legend:**
- PROD: a member of the production workspace (contract, quack, mask-risc, r2il);
- LAB: an own-workspace or excluded crate (relalg, dir-rmap-probe, crepe-plan-probe, mork-pathmap-probe,
  egglog-r2il-probe, dr-math-binding-probe).

All files listed below were confirmed present on `origin/main`.

## D.1 Implementation spine

| PR | source | stores / borrows | semantic identity kept | witness | lifetime / invalidation | executes through | status |
|---|---|---|---|---|---|---|---|
| #1649 | `contract/src/set_program.rs`, `relalg/src/program.rs` | `SetProgram { version, resident, lets, queries }`; one scratch plane per let | the set expression; lets are immutable | `version: u64`, by value | one `run_program` call; refused on version mismatch | relalg → mask-risc | type PROD (contract); executor LAB (relalg) |
| #1653 | `crepe-plan-probe/src/cse.rs` | shared sides laid out once per group; lets via `share` | per-rule answers equal the reference | side generations; version | per evaluation | relalg `run_program` | LAB. Measured: the gain is laying the core out once; lets save ops, not time |
| #1659 | `mork-pathmap-probe` (`RelationView`, `Side`, `JoinDomain`), `crepe-plan-probe` (`Store`) | borrows refcounted lane buffers (shared with Lance); builds only the index map `$i` | relation lanes | none beyond the caller | per prepare | folds (`bag_count`, `matching`) and ndarray compares | LAB |
| #1666 | `dir-rmap-probe/src/{qr.rs, pop.rs}` | DR256 register (256 slots, Warshall closure); `Pop { witness, tiles, kept }` | populations over one register version; ClassView methods bound to relalg queries | `Witness { version, users, groups, Weak<Snapshot> }`, compared by issue | `and` / `and_not` / `or` refuse `Stale`; `repair` re-derives after edits, never clears | relalg `Rel::Plane` → mask-risc | LAB |
| #1681 | `dir-rmap-probe/src/compose.rs`; `lance-graph-r2il` procedures | user → group → role composition; Bag and Distinct readings | Bag ≠ Distinct; NULL key = presence clear | one `Witness`; seal | refuses XOR, seal mismatch, stale binding, overflow | r2il `FoldDialect` → mask-risc | LAB (the r2il procedures are PROD) |
| #1685 | `dir-rmap-probe/src/compose.rs` | reads the resolved register (`Page::roles`, `Qr::role_slots`) instead of recomputing; the presence plane is borrowed | as #1681 | as #1681 | as #1681 | `ndarray::simd` AND + popcount | LAB |
| #1701 | `dr-math-binding-probe/src/lib.rs` | borrows a 512 B DR64 register (+ validity) | `MethodSignature { owner_class, method_ordinal, row/column Axis, truth, population }` | `Frame` (source, relation map, coordinates; by issue), `Focus`, `Scope`: three separate guards | re-checked per borrow; typed refusals | none (binding only) | LAB, on main via #1708 |
| #1703 | `dr-math-binding-probe/src/recipe.rs` | metadata: `LawShape`, `LawReference`, `CompiledReference`, `RecipeCertificate` | law ≠ compiled recipe ≠ data binding; a law is reusable across ClassIds | the binding is separate (`BoundDr64`) | admission refuses `NoCertifiedCompiler` for single-operand shapes | none | LAB, on main via #1708 |
| #1707 | `dr-math-binding-probe/src/relcomp.rs` | `Chain` of bound operands; `AdmittedLaw`; `RecipeCertificate` with egglog proof text | associativity of Boolean-Distinct composition only | one frame issue, focus and scope across the chain | commutation and Bag / K3 refused | the probe's compose kernel (not mask-risc) | LAB |
| #1740 | `relcomp.rs` (`execute_certified`) | certificate: `source_digest` + compiled steps | which query a certificate answers | the frame is checked per execution | refuses `WrongTarget`, `NotCertifiedQuery`, `NotThisChain`; 0 egglog calls on a hit | stored steps | LAB |
| #1693 | `egglog-r2il-probe/src/artifact.rs` | `ReplayCapsule`: premises, extraction, proof text, R2IL steps | the E1 exact-integer problem | none | `recheck` regenerates and compares text (not an independent checker) | R2IL scalar steps | LAB |
| #1694 | `egglog-r2il-probe/src/catalogue.rs` | `Catalogue { BTreeMap<SemanticKey, Entry> }` | `SemanticKey { goal, target, premises }` | `BindingWitness { source_generation, scope, coordinate_generation }` (caller-supplied), `TargetAbi` | `execute_or` falls back on a miss, a stale binding or a changed ABI; one entry per key (#1698 P1 gate) | R2IL | LAB; on main via #1696 / #1698 (#1694 itself merged into a feature branch) |
| #1696 / #1698 | integration PRs | — | — | — | #1698 adds dedicated CI and `REVIEW_GATES.md`; P0 gates open: exact → checked overflow, premise consistency | — | landed on main |
| (trace) | `egglog-r2il-probe/src/trace.rs` | nine hops, premises → verdict | derivation display | — | — | — | LAB; a trace is not a proof |

Production today: Quack numeric binding (`bind.rs`), `CompiledProgram`, mask-risc terminals including the
power sums, r2il procedures, and the contract's `SetProgram` type. Everything else in the table is lab.

## D.2 P2: what holds, what was incomplete

- **Holds** (Amendment B; not re-measured):
  - bind and lower ≈ 200 ns; `lower_fused` ≈ 360 ns;
  - building R ≈ 20 µs;
  - reusing R is 0.05–0.60 of recomputing it per property;
  - mask-risc skips dead words in gated predicates only;
  - block skipping pays at sparse selections and stops paying near full occupancy;
  - the failing P = 64 arms exceed 1,000 µs;
  - the Quack reuse controls run silently stale;
  - scalar occupancy is ≈ 2 % of building R.

  The P1 ndarray ratios are not a Quack result; B.6 is.
- **Incomplete, now corrected:**
  - #1666 `Pop` already refuses stale populations (B.7(5), C.2);
  - "a property change is harmless" became the three-way rule, with a measured falsifier (B.6).
- **The dependency rule against existing witnesses:**
  - `Pop.witness` and `Frame` are sound but whole-snapshot: any change refuses;
  - relalg's `version` is whole-table;
  - a bound Quack `Query` already names every input it reads (`Col` / `Mask` indices in its `Filter`).
    So a per-input witness needs no new semantic data, only a generation for each named input under a frame.

## D.3 The smallest missing production seam

Not a storage abstraction. In order:

1. **A frame on a bound `Query`** (D-NFR-1). Precondition: NFR-OP-1 decides who issues the production frame;
   not implicitly Quack. Frame, focus and scope stay three guards, never one revision number.
2. **A dependency witness for a kept plane:** the frame issue plus the generations of the inputs the bound
   filter already names. Owner: the binding store (D-QSP), informed by `Pop`. Whether `Pop.witness`,
   `Frame` and this become one type is NFR-OP-3. That is the reason not to add another `Pop`: the open
   question is the witness type, not the population type.
3. **The codebook generation inside the frame** (D-NFR-2): the stale-codebook control (B.6) is its falsifier.

Not seams, but deferred decisions with owners: DR³ physical contract (NFR-OP-2), cross-frame named
intermediates (D-QSP / D-NFR), #1791 scattered-mask reuse, S-INGRESS.

## D.4 D-DRMC alignment: one recipe, many witnessed bindings, no copied populations

- **Law identity** (`LawShape` / `LawReference`, #1703) excludes class, method and source version.
- **Recipe identity** (`RecipeCertificate`: law ref, compiled ref, `source_digest`, steps, proof text;
  #1707) names an executable sequence and target ABI.
- **Binding identity** (`Frame` / `Focus` / `Scope` + borrowed `rows`, #1701) is checked per execution.
- **#1740** runs the same certificate on fresh frames with 0 egglog calls. Each execution borrows that frame's
  operands; no population is copied into the certificate or the catalogue.
- **#1694's catalogue** keys recipes by `SemanticKey` + `TargetAbi`, with a separate `BindingWitness`. Its
  one-entry-per-key limit is a known gate (#1698 P1), not a design to copy.

So the binding store, when built, references recipes and bindings separately: a recipe hit never authorizes
stale operands, and a cached result is never keyed by the recipe alone.

## D.5 DuckDB implications after counting what exists

| DuckDB idea | existing answer | still useful? |
|---|---|---|
| CSE / reusable intermediates | relalg `share` + lets (#1649 / #1653), `Keep → Plane`, `Pop` | no new cache; #1653 measured that lets save ops, not time |
| prepared execution | `CompiledProgram`; `RecipeCertificate` (#1740) | yes, through the binding seam (D.3) |
| selective late materialization | masked terminals; reuse of a kept R (B.6) | the remaining gap is block gating at sparse selections (B.7(2)) |
| predicate transfer | exact semijoin / gather; register composition (#1681 / #1685) | no Bloom filter |
| adaptive filtering | `and_by_skip`; gated predicates | strategy choice only, never semantics |
| zone maps | witnessed `Range` | per-block value statistics only if source-sealed and measured |
| frontier reuse | #1740 recipe reuse; #1791 (lab) | gated on the #1791 contract |

## D.6 Authorization boundary

- **Ready for a separately approved implementation** (none authorized by this PR):
  - D-NFR-1 as a lab adapter, once NFR-OP-1 names the frame owner;
  - a Quack `Agg` route to the existing power-sum terminals;
  - a P3 measurement of gate-aware `And` + masked fold over a kept R.
- **Blocked:**
  - the production binding store (D-QSP);
  - cross-frame named intermediates;
  - #1791 scattered-mask reuse;
  - S-INGRESS / the joint coordinate witness;
  - the passive proof reference (CC1 agreement);
  - an independent proof checker;
  - the DR³ physical contract (NFR-OP-2);
  - witness unification (NFR-OP-3).

No new public ABI, persistent metadata, runtime cache or engine was added. P3–P5 remain unauthorized.

---

# Part E — D-NFR-1 ownership correction (2026-10-26)

**Principle: ownership is inherited from the source, not reconstructed in the query engine.** D.3(1) and D.6
asked NFR-OP-1 to name "who issues the production frame" before D-NFR-1 could start. That asked too much.
The source already owns its bytes, and Rust already enforces it inside a borrow. What remains open is
narrower and is stated in E.5.

## E.1 Authority and access, as built

```
owner (holds the bytes)                      reader (borrows them)
──────────────────────                       ─────────────────────
MailboxSoA<N>  ──impl MailboxSoaOwner──┐     &MailboxSoA ─ MailboxSoaView ─ entity_type / meta_raw / edges_raw
  write_row(&mut, row, cycle, cell)    │                                     │
  set_*(&mut) (12 raw setters)         │                                     ▼
  tick(&mut)                           │     LaneRef::{U32, U64} ─► Planes<'_> ─► mask_risc::execute ─► Value / Keep
Vec<NodeRow>   ────────────────────────┤     NodeRowPacket<'_> (SoaEnvelope) ─► LaneRef::Strided ─► same
                                       │
fleet (cycle_driver::MailboxFleet)     │     cognitive_pass(fleet: &F, …, think: FnMut(&F::Owner))
  owner_mut(&mut) for seal/write  ─────┘       └ the production borrow window (private fn, feature `cycle-driver`)

nominal:  SoaEnvelope::mailbox_owner() (default 0) ─► BatchWriter::cast(on_behalf, …) ─► resolve_owner cache
persisted: Lance version ◄─ QueryReference::at(v, rung)   (QueryReference::default() = u64::MAX, NOT a horizon)
```

Every arrow on the reader side is a `&` borrow: `Planes<'_>`, `LaneRef<'a>`, `StridedRef<'a>`,
`NodeRowPacket<'a>`, `MailboxSoaView`'s `&self` slices. A Quack `Query` lowers to a `Program` that reads only
what the caller lends it. No reader-side type owns operand bytes.

## E.2 Three guarantees, kept apart

- **A. Structural (compile time).** No owner `&mut` while any view or lane borrow is live; no view outlives
  the owner; no view is cloned into a second holder. Pinned by #1755's `compile_fail` E0502 / E0597 / E0599
  (`soa-view-probe/src/lib.rs`). Holds for every reader on the right of E.1, with nothing to check at run
  time. Limit, stated in that crate: interior mutability or an `unsafe` write beside the owner is not
  covered; neither exists for `NodeRow` or `MailboxSoA`'s columns.
- **B. Nominal (provenance).** `mailbox_owner()` stamps which mailbox owns the bytes; `cast(on_behalf, …)`
  routes a write on its behalf; `resolve_owner` caches a resolver's answer. None of this is authorization:
  `cast` accepts any id, and the cache is not an ACL. `cast_on_behalf` sits behind `with-planner`.
- **C. Persisted (snapshot).** Only a pinned `QueryReference::at(v, rung)` or a Lance version names a horizon
  a later reader can return to. `QueryReference::default()` is unbounded by design (F-QREF-STRICT).

## E.3 What P2's 1,032-vs-31 result shows, restated

B.6 control 4a kept `R2 = R ∧ (val_0 ≥ 0)` as a detached `Keep` mask, mutated `val_0`, and reused R2: 1,032
rows against 31 fresh. Between the two runs the lane borrow had ended, so A never applied. The result shows
that **a detached mask carries no dependency witness**. It does not show that a valid borrow can observe a
mutation; A rules that out at compile time.

## E.4 Falsifiers: borrow-scoped versus escaped

- **Borrow-scoped (exists, #1755).** The three `compile_fail` examples plus the compiling twin. A safe
  mutation cannot overlap the read; nothing new is needed.
- **Escaped (added here, commit `769d461d4`).**
  - `soa-view-probe/tests/falsifiers.rs::an_escaped_mask_is_not_witnessed_by_the_packet_cycle`. A mask
    computed through a `NodeRowPacket` escapes the borrow; the owner writes one classid; a new packet built
    with the same caller stamp (`cycle() == 5`) gives a different mask. The packet's `cycle` is what the
    caller passed to `new`, not a record of writes.
  - `cognitive-shader-driver/tests/escaped_reuse.rs`, three tests:
    - a raw setter (`set_entity_type`) changes the answer while `current_cycle` and every
      `last_write_cycle` stay put;
    - a same-cycle `write_row` is accepted and re-stamps the row with the cycle it already had, so it is
      equally invisible;
    - the twin: a write in a later cycle moves `current_cycle` and the written row's stamp, and leaves an
      unwritten row's stamp alone.
  - Disable runs, each red only in its own test: setter stamps → the setter test fails; `tick` a no-op → the
    twin fails; owner write removed → the packet test fails.

So the owner's existing stamps witness exactly one thing: **a `write_row` in a later cycle**. They cannot
witness a same-cycle write or a raw setter. An escaped mask therefore needs a rebind (recompute under a new
borrow) or a refusal from an owner-issued contract; it cannot be revalidated from `(cycle, last_write_cycle)`
as the mailbox stands.

## E.5 Classification of D-NFR-1, and the smallest remaining seam

- **Inside a borrow: a thin adapter, not a new authority.** Lending `MailboxSoaView` columns or
  `NodeRowPacket` bytes to `Planes` (as `LaneRef::U32` / `U64` / `Strided`) and running the bound `Query` in
  the same scope needs no frame, no lease and no guard: A already holds. Gaps that are type gaps, not
  authority gaps: `entity_type`/`class_id` are `u16` and `energy` is `f32`, and `LaneRef` has neither; the
  `u32` classid inside a `NodeRow` facet is reachable today through `Strided`. No Quack caller does this yet.
- **Across a borrow (escaped reuse): a missing contract, shown by counterexample (E.4).** The mailbox's
  write paths do not all advance a stamp a reader could hold. The smallest fix is on the OWNER side: make
  every mutating path advance one owner-issued generation (or route the raw setters through `write_row` and
  make a same-cycle write move a stamp). The query engine then keeps that generation beside a `Keep` and
  compares it at reuse. That is D.3(2)'s dependency witness, issued by the source rather than reconstructed
  in Quack. Not built here: it changes a write path, which this correction is not authorized to do.
- **Persisted reuse:** C only. A mask kept past a cycle seal needs `QueryReference::at`, never `default()`.

Neither half needs a `FrameStore`, a read lease, a cache, a new ABI or a source-generation type in Quack. The
lab `FrameStore` in `dr-math-binding-probe` stays a lab stand-in.

## E.6 NFR-OP-1, corrected

Was: "who owns the production frame store (S6): Quack, ClassView, or the source owner?" — OPEN. Corrected:

- **For in-borrow execution: answered.** The source owner, through the borrow. No frame store is needed.
- **For escaped reuse: narrowed, still OPEN.** The question is no longer who owns a frame store but whether
  the mailbox owner will issue a generation that every write path advances (E.5). Owner: the mailbox /
  write-path owner (`v3-mailbox-warden` scope), not Quack.
- **For persisted reuse: answered by C** (`QueryReference::at` / Lance version).
- **Delegated authority** (may this caller read this mailbox at all) stays open and is not a frame question.

## E.7 Cost

A in-borrow execution costs what the kernel costs: #1755 measured the packet path equal to direct ndarray,
and adding a per-operation frame check would re-check per call what the compiler has already proved. Escaped
reuse pays one integer comparison per kept mask at reuse, and only once the owner issues the generation. The
cost lands where the risk is (escape), never on the in-borrow path.

## E.8 What changed in this correction

- Two test files (E.4). No production code, ABI, writer or WAL change; no P3–P5; ScatterOr untouched.
- D.3(1) and D.6's first "ready" bullet read through E.5 / E.6: the in-borrow adapter no longer waits on
  NFR-OP-1.
- #1792's measurements (B.6) are unchanged; only their interpretation is restated (E.3).
