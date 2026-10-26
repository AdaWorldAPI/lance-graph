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
