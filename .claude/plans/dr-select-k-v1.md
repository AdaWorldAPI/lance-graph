# D-DR-SELECT-K — one versioned DR selection, bound once, reused by every fold

**Status:** PROPOSED, docs only. Read at `main` `54c9b98a4` (2026-11-12).
Hypothesis record: PR 2032 (`dr-widefieldmask-dt-classview-v1.md` §1a.1,
`dr-dt64-class-governance-v3.md` §2c.1). Coordination: EXO-253. Laws this
plan must keep: EXO-220 (locality), EXO-255 (mapping is versioned, content
needs its own witness), EXO-46 and EXO-140 (admission), EXO-114 (order).

Only D-DR-SELECT-K-1 (the lab) is a candidate for implementation, and it is
not authorized yet. Everything after it depends on what it measures.

## 1. Order of things

The choice of a physical representation is a DR addressing decision over a
deterministically bound version. It is not a storage or residency decision,
and it is not a query optimizer in front of DR.

```
                 DR
                  │
        deterministic version V
                  │
          ┌───────┴────────┐
          ▼                ▼
     index order         filter
          │                │
          └───────┬────────┘
                  ▼
          DR{dynamic}[K]
                  │
         reusable selection
                  │
          Quack / mask-RISC
                  │
               ndarray
```

- An index gives an order; a filter gives a subset. They are not exclusive
  and compose over the same witnessed version:
  `S_V = Filter_V(IndexOrder_V(N))`, then `DR{dynamic}[S_V] → Compute(K)`.
- That is one composition order. For `WHERE` plus Top-K the semantically
  correct order (eligibility before the limit) is kept, whatever order the
  physical execution uses.
- Lance may do the index or the filter work. DR binds the result once.
  Quack and mask-RISC consume the existing selection and compute only what
  remains. No further selection, no second read of the source, no second
  authority.
- Constraint that stays: folds never touch Lance. Intake and binding happen
  once per version and are reported in µs, not gated (the cold-cost
  exception of `CLAUDE.md`); folds are under the no-millisecond rule.

## 2. Ownership

- Canonical values stay with the source owner.
- `NodeGuid` stays the logical identity.
- DR's existing `MapTable` stays the authority for the versioned
  global-to-local numeric mapping.
- The source adapter owns the snapshot-scoped translation to Lance row
  identity or address. **Not shown yet:** how a `NodeGuid` resolves to the
  canonical global id and then to a row of one snapshot, using only
  existing authorities. Until that chain is demonstrated, D-DR-SELECT-K-2
  is blocked.
- A K-row Arrow batch is not zero copy because it is small. Where it is
  needed it is counted as bounded, derived staging owned by the source or
  the focus lease. It is never a second canonical owner of values. Borrow
  where borrowing is supported and shown to work.

## 3. Representation choice

The choice belongs to the versioned DR binding. It follows the
representation that already exists and the operation that comes next:

| view | chosen when |
|---|---|
| **IndexVector** | the order already exists and the consumer can use it |
| **SelectionVector** | a sparse sequence of local positions is cheaper |
| **FocusMask** | Boolean algebra and SIMD folds over the local population are cheaper |
| **implicit `ALL(K)`** | the upstream selection is already exact and complete |

One buffer may serve several roles. A further representation is allocated
only when its added benefit is measured. The choice is deterministic: the
same version, binding and next operation give the same view. If one view
wins everywhere, only that view is used.

> This replaces the input list of §1a.1's "Cold deterministic physical
> choice" paragraph and the size matrix of §2c.1's W7 addendum for planning
> purposes. Both stay in place as the recorded hypothesis of PR 2032.

## 4. D-DR-SELECT-K-1: the lab (not authorized yet)

**Question.** For one versioned DR selection, which existing physical
representation is cheaper for a given operation and a given reuse count?
The lab does not decide "cold Lance against resident DR".

A small, independent lab crate. No `StrategySelector`, cache, mapper,
executor or writer.

| dimension | values |
|---|---|
| N | 1,000,000 |
| K | 256 and 65,536 |
| reuse R | 1 and 100 |
| address pattern | clustered and random, where it matters |
| negative control | one case with 75% survivors |
| next operation | Boolean fold with count, aggregate over a lane, ordered Top-K |
| view | IndexVector, SelectionVector, FocusMask, implicit `ALL(K)` where valid |

**Setup.** One version V. The selection is produced once, by an index
order, a filter, or both composed, and bound once. Every view is taken
from that one binding; a view that has to be converted reports the
conversion.

**Oracle.** An independent row-at-a-time answer; every view equals it,
including the ordered `NodeGuid` sequence where order is requested.

**Reported per cell:** producing the selection (index, filter, bytes
copied), bind, conversion if any, fold, retained memory, and the total for
each R. Production and bind in µs, labelled "reported, not gates"; fold
terms under the no-millisecond rule.

**What "no second read" means.** mask-RISC keeps reading source-owned SoA
lanes; that is its operand access and it is allowed. What is forbidden is
running the completed Lance selection again, or scanning N through storage
a second time. The lab therefore keeps three separate counters: storage
accesses, binds, and operand reads. After the bind, the storage counter
does not move.

**Only admissible views compete.** The views are not always
interchangeable. A FocusMask keeps membership but not an arbitrary index
order or multiplicity. For each operation the lab compares only the views
that can lawfully answer it; for ordered Top-K the order stays witnessed,
so a mask alone is not an arm there.

**Pass or stop.** The lab passes if every view equals the oracle and each
number is attributable to one term. It does not have to produce a winner
per operation; "one view wins everywhere" is a valid result and ends the
strategy question.

## 5. Gates before D-DR-SELECT-K-2

1. The ownership chain `NodeGuid` → canonical global id → DR coordinate →
   row address of one Lance snapshot is established from existing
   authorities.
2. Exact WHERE results are told apart from candidate-only results, with
   residual filters and Top-K ordering covered.
3. A real workload exists where binding K and folding over it wins after
   producing the selection, staging and amortization are counted.
4. The Admission B, source, temporal, IAM and version witnesses are kept.

If these hold, D-DR-SELECT-K-2 may propose the smallest deterministic
handoff rule the measurements support.

## 6. Contingent, not planned

Each of these waits on §5 and gets its own plan if it is ever started:
binding the selected population in a lab (K-2), running the existing
Quack / RelAlg / mask-RISC over K-sized planes (K-3), temporal binding
(K-4), prefetch and lease (K-5), sparse write-through (K-6), an external
Arrow consumer (K-7).

## 7. Not verified

- Which Lance 14 scanner or index call returns a compact, exact survivor
  set, and at what copy cost. The lab's first task.
- Whether mask-RISC accepts an implicit `ALL(K)` without a materialized
  plane.
- The `NodeGuid` to row chain of §2.

## 8. Geometry roles (2026-11-12, amendment)

**Status:** the first table states what exists; the reading of the four
geometries as roles of one binding is a WORKING-MODEL, not a contract.
Background: PR 2032 (hypothesis), PR 2033 (this plan), PR 2035 (the lab).
Nothing in this section changes a namespace, a classid, an `F###` behaviour
address or `MethodId`.

| geometry | role | what exists | status |
|---|---|---|---|
| DR64 | native bounded focus, up to 64 positions | `Width::W64`, `Extent::DR64_CUBE`, `Axis::for_survivors` (≤ 64 → one W64 tile) in `lance-graph-dt64/src/dr.rs`; the 64 × 64 focus lease in `drf-focus-probe` | width rule SHIPPED (workspace member); focus lease LAB (excluded crate, "Not a production API") |
| DR256 | native bounded focus, up to 256 positions; the 256 × 256 slab | `Width::W256`, `Extent::DR256_SQ_X64`, `Extent::DR256_GLOBAL_X64`; slab planes in `dr256-ternlog-lab` | extent SHIPPED; slab planes LAB (a workspace member that declares itself a lab) |
| DR{dynamic} | a selection sized by its survivors, past one native tile or irregular | `Axis::for_survivors` (> 256 → `ceil(n / 256)` W256 tiles), `MapTable::locate`; views in `dr-select-k-lab` | width rule and locate SHIPPED (D-DYNBAG-4, partial); views LAB (PR 2035); cross-tile compose OPEN |
| DR256³ | three-axis relation geometry | the shape `Extent` (identity × property × predicate); no `Extent` constant has a 256-wide predicate axis; "No cube is allocated" (`dr256-ternlog-lab` module doc) | shape SHIPPED as a reservation that allocates nothing; a 256³ fold is PROPOSED and untested |

Read as roles of one deterministic binding:

- **A bounded focus that fits a native tile uses the native tile.** That is
  already the rule `Axis::for_survivors` encodes: "the narrow tile is used
  whenever the selection fits it".
- **DR{dynamic} is the escape hatch** for a population that is larger than
  a native focus or irregular. It is not a second engine and not a value
  owner.
- **DR256³ is used only when the operation needs a three-way relation**,
  and then §9 decides whether anything has to be expanded.

Order of things (§1) is unchanged; the focus step reads "native DR64 /
DR256, or DR{dynamic}".

## 9. Three masks are not a cube

`VPTERNLOGQ` combines three aligned bit vectors with one truth table. It
does not build or walk X × Y × Z.

| representation | positions addressed | one dense Boolean mask | can express |
|---|---|---|---|
| one DR256 axis | 256 | 32 B | a set of positions |
| three independent axes | 3 × 256 | 96 B | exactly the products A × B × C |
| three pairwise relations | 3 × 65,536 | 24 KiB | relations that are determined by their pairs; not every ternary relation |
| one full DR256³ relation | 16,777,216 | 2 MiB | any ternary relation |
| three full DR256³ operands | 3 × 16,777,216 | 6 MiB | a ternlog over three ternary relations |

A factorized selection `S(x,y,z) = A(x) ∧ B(y) ∧ C(z)` has
`count = popcount(A) × popcount(B) × popcount(C)` with no expansion, as
long as nothing else constrains the triple and no multiplicity is
attached. An arbitrary ternary relation needs explicit triples, an
implicit rule, bounded slices or the dense cube.

Checked with a small oracle
(`.claude/probes/2026-11-12-dr-geometry/three_axis_oracle.py`, exact
integers, no timing):

| case | result |
|---|---|
| factorized, popcounts 10, 12, 16 | count 1,920 = product; 38 coordinates visited; 96 B of masks against 2 MiB dense |
| `(x + y + z) mod 256 = 0` | count 65,536; all three axis projections full, so three masks give 16,777,216; all three pairwise projections full as well; 196,608 B as explicit triples against 2 MiB dense |
| the factorized set plus one cross-axis parity constraint | count 960; the product of popcounts (1,920) is wrong |

So: axis masks are right for a product and only for a product; a
cross-axis constraint, however small, takes the selection out of that
form; and a sparse ternary relation is cheaper as triples than as a cube
(here by a factor of 10).

Open, not answered by this plan:

1. Which existing operations need a 256³ coordinate domain at all.
2. Which of them are products, or pairwise, under DR's existing address
   and witness model.
3. Whether Quack / RelAlg / mask-RISC can do the lawful part without new
   execution semantics.
4. Which cases would need a Gather, a relation promoted to an operand, or
   a derived population that EXO-46 and EXO-140 do not admit.

No cube owner, dense cache or cube executor is proposed.

## 10. Next measurement (proposed, not authorized): D-DR-SELECT-K-1b

The strongest structural case first. N = 65,536 (one slab), an existing
sealed order on the key, top 64 and top 256:

```
version V → existing sealed order → first 64 / 256 positions
          → native DR64 / DR256 focus → existing fold
```

| quantity | top 64 | top 256 |
|---|---|---|
| share of N | 0.09765625% | 0.390625% |
| source positions as `u32` | 256 B | 1 KiB |
| local focus mask | 8 B | 32 B |
| mask over the whole population | 8 KiB | 8 KiB |

Arms, same answer required from each, order witnessed by the binding:

1. order index → positions → native focus → fold over the focus;
2. an N-wide mask of the same rows → fold over N (the existing path);
3. the positions as a dynamic view, where admissible.

R = 1 and R = 100; every copy and every staging step counted; no N-wide
mask is built on arm 1. Falsifier for the preference: arm 1 is not cheaper
than arm 2 on the total once staging is counted. The tie rule is the
existing `SealedOrder` contract (key, then frame row position).

## 11. Gap register

Only what is missing; nothing here is a request to build.

| gap | evidence | consequence |
|---|---|---|
| No admitted consumer folds over a list of positions. | mask-RISC A1: "a `SelectionVector` cannot be expressed in this vocabulary at all"; Quack: "no `SelectionVector`, anywhere … ELIMINATE — an index list is the materialisation the mask-native invariant forbids" (`lance-graph-quack/src/lib.rs`, matrix row R1). Not found in mask-RISC, Quack, `drf-focus-probe`, `dr256-ternlog-lab`, the planner's `temporal.rs` or ndarray's `simd_masking_ops.rs` (scopes and patterns in the PR). | The lab's positions arm is a scalar loop that measures an access pattern. The SelectionVector role of §3 conflicts with a standing rule of the two executors and needs a decision before any production use. |
| The gated predicate skips empty mask words only. | `MaskOp::Pred` doc: "the per-word gate test is still ∝ rows/64"; at about 1% random density about half the words are non-empty. `masked_sum_i32` already walks a sparse word bit by bit (at most 16 set bits). | A lawful candidate for the sparse case that stays inside the mask vocabulary: the same per-word sparse walk for a gated predicate. An ndarray question; not measured. |
| mask-RISC has no masked sum without a mask operand. | `Terminal::MaskedSumI32 { mask, lane }`; found in PR 2035. | `ALL(K)` needs one constant plane of K / 8 bytes for a sum. Count needs none. |
| Two order contracts. | `SealedOrder`: key, then frame row position (`TIE_KEY_THEN_FRAME_POSITION`); the plan asks for ordered `NodeGuid` results. | The lab declares both and admits a request only for the order the binding witnesses. Which one a consumer may rely on across versions is EXO-114. |
| `NodeGuid` → canonical global id → DR coordinate → snapshot row. | plan §2, §5 gate 1. | Unchanged. An in-memory lab with `u128` and `u32` does not settle it. |
| Which Lance call returns an exact survivor set. | plan §5 gate 2, §7. | Unchanged. |
| No `Extent` with three 256-wide axes; no 256³ fold. | `dr.rs`: the three constants. | §9 stays an analysis until an operation is named that needs it. |

## 12. What stays closed

D-DR-SELECT-K-2 to K-7 stay behind the four gates of §5. No strategy
selector, policy registry, cost framework, second focus type, planner or
cache is proposed anywhere in this plan.

## 13. DuckDB `SelectionVector` compatibility (2026-11-12, decided: D-DR-SELECT-K-SV)

**Status:** DECISION taken 2026-11-12 (D-DR-SELECT-K-SV, below). Nothing is
implemented. The decision admits the mechanism; production admission still
has to pass its gates.

> **DECISION (D-DR-SELECT-K-SV, 2026-11-12).** An explicit `SelectionVector`
> is admitted in two places: at the DuckDB-compatible boundary, and as a
> typed sparse operand inside the existing DR execution.
> **BASIS.** Matrix R1 excluded the mechanism and named its own falsifier;
> the falsifier fired in 12 of 12 cells (matrix §8b).
> **SCOPE, authorized.** (1) A narrow typed selection operand for the
> existing execution paths, first for independent `COUNT` and `SUM` folds
> over locally witnessed positions. (2) Borrowing existing index or position
> buffers: no new ownership, no allocation in the hot path, no
> value-dependent cross-table gather. (3) Carried through the existing Quack
> and mask-RISC semantics with the computation in `ndarray::simd`, with an
> independent oracle and backend parity tests. (4) An explicit DuckDB API
> adapter that materializes a position list only on request and counts the
> copy.
> **SCOPE, not authorized.** A new scalar executor; global position indexes
> as a second authority; a runtime optimizer; bypassing EXO-46 or EXO-140.
> **Limit.** A DuckDB position list may hold a permutation or repeated
> indexes; a focus mask is a set. The boundary never converts an arbitrary
> selection into a mask where that loses order or multiplicity.
> **Not needed for this decision.** The density threshold; it is a later
> performance calibration.
> **REVISIT WHEN.** The admission PR's parity or cost gates fail, or a
> selection operand turns out to need ownership the binding does not have.
> The first implementation is a separate lab / admission PR
> (D-DR-SELECT-K-SV-1).

**The requirement.** The aim is DuckDB-compatible semantics and, where
useful, API compatibility. DuckDB's `SelectionVector` is part of that
contract, so it cannot be excluded outright. Two levels are kept apart:

1. **Compatibility contract (boundary).** An explicit selection of `u32`
   positions is representable: identity when no buffer is set, explicit
   indices, order, slicing and composition, a writable buffer where the API
   hands one out, dictionary-style indirect access.
2. **Native execution (inside).** DR64 / DR256 focus, `WideFieldMask`,
   witnessed order index, borrowed sparse positions or implicit `ALL(K)`.
   No position list is built for native execution unless a consumer needs it.

**Checked against DuckDB `main` (fetched 2026-11-12, not pinned):**
`selection_vector.hpp` has `get_index` returning the index itself when no
buffer is set, `set_index`, `data()`, `Capacity()`, `Slice`, `SliceInPlace`
and `SelectFromBoolean`; the public `duckdb.h` declares
`duckdb_create_selection_vector`, `duckdb_destroy_selection_vector`,
`duckdb_selection_vector_get_data_ptr` (returns `sel_t *`),
`duckdb_slice_vector` and `duckdb_vector_copy_sel`. `vector.hpp` and
`ToUnifiedFormat` were not read here.

**What a mask cannot carry.** A mask is set membership in row order. It does
not carry an arbitrary permutation, repeated positions (bag multiplicity) or
dictionary indirection. These need a position form or an order witness, and
each needs its own parity test. NULL is separate again: DuckDB's
`ValidityMask` semantics hold at the boundary and are never equated with
absence or a zero fallback.

**Rules that do not change.**

- One authority. A compatibility selection refers to the existing versioned
  DR binding (§1, §2). No second row-id namespace, source owner, engine or
  persistent cache.
- Explicit materialization. A writable contiguous `sel_t *` is an owned,
  bounded copy with a stated lifetime and invalidation rule; every
  allocation and copy is counted, as in the lab.
- No second executor. Execution stays with Quack's algebra and
  mask-RISC / ndarray. The open question is whether their operand set admits
  a selection operand, not whether a parallel scalar path is added.
- The public C API is the compatibility target. DuckDB's internal C++ types
  are a separately version-pinned target, never an assumed stable ABI.

**Existing decisions this touches.**

| decision | where | state after 2026-11-12 |
|---|---|---|
| Matrix R1, `SelectionVector` | `duckdb-to-v3-translation-matrix-v1.md` | **ADAPT.** Its pre-registered falsifier was run and fired in all 12 cells (matrix §8b). No sparse threshold located yet. |
| Matrix R4, `DictionaryVector` | same | Parity check required; no automatic release. Its falsifier condition is met now that the requirement is adopted. |
| Matrix R7, `UnifiedVectorFormat` | same | Parity check required; no automatic release. `Planes` has no `sel` member. |
| Quack R1, mask-RISC A1 | `lance-graph-quack/src/lib.rs`, `lance-graph-mask-risc/src/lib.rs` | Unchanged in code until D-DR-SELECT-K-SV-1. Decided: a borrowed, frame-local position operand is admissible after its implementation gates. Their allocation, ownership and single-row-universe rules stay. |
| DR64 / DR256, DR{dynamic} | §8 | DR64 / DR256 stay the native default; DR{dynamic} carries the sparse and variable position view as the escape hatch. |

**Smallest adapter (to be designed, not built).** A boundary type that holds
a reference to one bound selection plus one of: nothing (identity), a
borrowed position slice, or an owned `u32` buffer created on request. It
answers `get_index`, slicing and the data pointer; it converts to the native
views through the conversions the lab already counts. It owns no rows and
runs no folds.

**Falsifier suite (none run except where stated).**

1. Identity selection without materializing indices.
2. Native Top-64 / Top-256 from a sealed index (§10, D-DR-SELECT-K-1b).
3. Sparse selection, repeated execution: consumption measured (matrix §8b);
   the total including construction is not.
4. Arbitrary permutation and repeated positions, per DuckDB operation that
   permits them.
5. Composed selections and dictionary slicing.
6. Selection mutation and buffer lifetime.
7. Validity / NULL parity.
8. Refusals: wrong frame, stale source, foreign coordinate, unauthorized
   gather.
9. Against both the existing mask kernel and a zero-word-skipping bit walk
   (done for the sum in matrix §8b; both lose to the list at 1/64 and below).

**Order of work.** First the contract (this section, decided),
then the adapter against the existing views with parity tests grounded in
DuckDB's own behaviour, then the sparse-kernel comparison at denser points
to locate the threshold. No optimizer and no selection engine.

## 14. D-DR-SELECT-K-SV-1: the first admission (2026-11-12)

**Status:** IN PR. Lab and admission level; not a production release.

**Shape chosen.** No field is added to `Planes` (it is built as a struct
literal in about 400 places). The admission is a separate, narrow entry
beside `execute_into`:

| layer | what was added | file |
|---|---|---|
| ndarray | `sum_i32_at`, `count_gt_i32_at`: folds over a borrowed `&[u32]` bag; out of range is `None`. One plain loop on every backend; no gather word. | `ndarray/src/simd_masking_ops.rs` (ndarray PR 423) |
| mask-RISC | `SelFold::{CountGtI32, SumI32}`, `execute_sel`, `reference_execute_sel`, refusals `SelOutOfRange` and `SelSumBound`. One delegation per fold, no allocation, validation shared with the oracle. | `crates/lance-graph-mask-risc/src/sel.rs`, `reference.rs` |
| Quack | `lower_sel(query, selection)`: the same query as for `lower`, with its selection plane supplied in position form. Two shapes admitted, everything else refused (`NotASelectionFold`). | `crates/lance-graph-quack/src/lib.rs` |

**Contracts.**

- Bag semantics, in list order. A row listed twice is folded twice. The list
  is never converted to a mask.
- One row universe. A position is a row of the frame (`0..n_rows`) or the
  fold is refused, naming the first offender. No foreign plane, no key.
- Overflow. `MaskedSumI32` is exact or refused (`n_rows` at most 2^32). The
  selection sum keeps that contract with the bound moved to the list, since
  a bag can be longer than the frame: at most 2^32 positions, else
  `SelSumBound`. Within the bound ndarray's wrapping accumulation cannot wrap.
- For a selection that is a set, the position fold and the mask program
  answer the same (tested in mask-RISC and through Quack).

**What this layer does not check.** mask-RISC knows a frame only as
`n_rows`, exactly as `Planes` does. Whether a position list was witnessed
for the version the lanes come from is the binding's check before the call
(in the lab: `Bound::bind` refusing a stale selection). `SealedOrder`, the
type that carries such a witness for an order index, lives in the lab crate
`dr256-ternlog-lab`, not in `lance-graph-dt64`. Connecting a witnessed
binding to `execute_sel` in a non-lab crate is the next gate, not done here.

**Not done.** The DuckDB API adapter; permutation-sensitive operations;
dictionary slicing (R4); a `sel` member for `Planes` (R7); a SIMD gather;
the density threshold; any use from production code.

## 15. D-DR-SELECT-K-SV-2: a selection witnessed with its lanes (2026-11-12)

**Status:** IN PR (Draft). Admission level. Two of the brief's properties are
BLOCKED and listed below; nothing here is a production release.

**Invariant.** A selection may execute only if its positions address the frame
that holds the borrowed value lanes. Equal row count, equal revision numbers,
an in-range position or a caller-supplied id do not establish that.

**Ownership and witness map (what exists, and who owns it).**

| property | owner (symbol, file) | state |
|---|---|---|
| frame identity, issuer, source / relation-map / coordinate revisions, reseal | `FrameStore::check`, `FrameRevisions`, `Frame` (no public constructor) in `crates/lance-graph-dt64/src/dr.rs` | SHIPPED |
| focus and scope equality against the bound values | `Guards::check` in `crates/lance-graph-dt64/src/dr.rs` | SHIPPED |
| lane issued by a source for its own frame, in one row order | `LaneSource`, `CoordSpace`, `LaneAddr`, `BindStore::bind_lanes`, `LaneBinding::check` in `crates/lance-graph-dt64/src/lane.rs` | SHIPPED |
| position list issued by the same source, same frame, same row order | `SelectionSource`, `SelAddr`, `LaneBinding::selected` in `crates/lance-graph-dt64/src/lane.rs` | NEW in this PR |
| the fold over borrowed positions | `execute_sel` in `crates/lance-graph-mask-risc/src/sel.rs` | SHIPPED (PR 2039) |
| the boundary that calls the fold once | `fold_bound_selection` in `crates/lance-graph-report/src/bound_sel.rs` | NEW in this PR |
| a reader `Scope` that carries permission | none. `Scope` is a plain `{ id }` value; `AbiLanes::unwitnessed` in `crates/lance-graph-report/src/lane_source.rs` is test-only for that reason; `crates/lance-graph-rbac/tests/scoped_lane_binding.rs` records the gap as findings | BLOCKED |
| a general row-order witness | `SealedOrder` in `crates/dr256-ternlog-lab/src/order_store.rs` is LAB; `OrderedLaneWitness` in `crates/lance-graph-contract/src/ordered_lane.rs` covers sealed facet key lanes only | BLOCKED for ordered consumers |
| a pinned historical snapshot | none; `FrameStore::check` accepts the current issue only | REFUSED (current-frame mode) |
| NodeGuid → global id → DR coordinate → Lance row | open (EXO-46, EXO-140) | OPEN, not touched |

**What was added.** One second socket on the owner that already issues lanes:
a `SelectionSource` names a position list (`selection_of`) for a frame and
lends it (`borrow_positions`). `LaneBinding::selected` runs the existing
`check`, then requires the selection's `CoordSpace` and `Scope` to equal the
binding's, then borrows the lane and the list. `fold_bound_selection` passes
both to `execute_sel` exactly once. No position store, id namespace, map
table, `Planes` field, Quack operator or executor was added, and nothing is
copied.

**Verified** (`crates/lance-graph-report/tests/bound_sel.rs`, 10 tests;
`tests/bound_sel_alloc.rs`, 1 test; oracle is plain indexing in the test):

- same frame: COUNT and SUM equal the oracle; operands are the source's own
  slices (pointer equality);
- Frame A positions against Frame B lanes with equal length and equal
  revisions: `ForeignFrame`, in all three pairings, before any borrow;
- source, relation-map or coordinate revision moved, or the frame resealed
  with unchanged revisions: refused, with the cause named;
- a selection in another row order with the same numbers: `SelectionSpace`;
- a selection issued for another reader scope, or not issued: refused;
- focus or scope changed: refused by the existing `Guards::check`;
- a position outside the lane: `SelOutOfRange` with the first offender;
- duplicates are folded twice; a reordered list gives the same COUNT and SUM;
- a thousand accepted and a thousand refused calls allocate 0 bytes.

**Disable runs.** Eight mutations of the binding path, each asserted to apply
once: 8 of 8 red. A ninth (asking the source for the selection at a freshly
sealed frame in place of the binding's) stayed green and is not counted: after
`check` passes, the store's current issue is the binding's frame, so the two
are the same value.

**Cost** (`crates/lance-graph-report/examples/bound_sel_cost.rs`, N = 65,536,
medians of 201, two runs in `.claude/probes/2026-11-12-dr-select-k-sv-2/`;
cold terms reported, not gates):

| K | produce Top-K (once per frame) | bind (once) | witness | fold SUM | witnessed fold | mask build | mask fold SUM |
|---|---|---|---|---|---|---|---|
| 64 | 138–144 µs | 233–250 ns | 139–164 ns | 64–75 ns | 170–171 ns | 145–147 ns | 2.0–2.5 µs |
| 256 | 141–142 µs | 233–246 ns | 139–140 ns | 138–139 ns | 243–245 ns | 290–292 ns | 2.0–2.1 µs |

Total for R = 1: 139–145 µs. For R = 100 with one witness: 146–156 µs; with a
witness per fold: 156–167 µs. Selection production is 90% or more of every
total and runs once per frame. The Top-K order is the source's own claim, no
order witness is consulted, and COUNT / SUM do not depend on it. The mask
column is the same rows as a set (a Top-K has no repeats); it is not compared
for bags. Position bytes: 4·K, borrowed; staging and conversion: none.

**Not verified / not supported.**

1. **Authorization (BLOCKED).** Scope equality is a consistency check. No
   shipped authority issues a `Scope` from an IAM decision, so a revoked
   permission with a matching scope id cannot be refused at this boundary.
   Owner: the `Scope` type in dt64 and the rbac crate
   (`authorize_scoped`); the governing plan
   `dr-dt64-class-governance-v3` holds contract changes until its council
   step W11. Smallest falsifier, once an issuer exists: revoke the grant,
   keep the scope id, expect a refusal before any borrow. Decision needed:
   who issues a `Scope` and what a binding holds as proof of it.
2. **Ordered consumers (BLOCKED).** No non-lab witness says a position list is
   in a stated order. Only COUNT and SUM are admitted, which need none.
3. **Historical snapshots (REFUSED).** A binding whose frame is no longer
   current is refused; nothing substitutes the current frame.
4. **Global identity.** The list holds local frame positions. Mapping from a
   NodeGuid or to a physical Lance row is not part of this step.
5. No production `SelectionSource` exists; the test fixture is the only
   implementor. R4, R7 and the DuckDB adapter are untouched.

## 16. D-DR-SELECT-K-SV-3: the DuckDB SelectionVector C API (2026-11-12)

**Status:** IN PR (Draft). LAB crate `crates/duckdb-selvec-compat`, own
workspace, not linked by anything. R4 and R7 stay UNDER REVIEW.

**Reference, pinned.** DuckDB tag `v1.5.6`, commit
`069cc9f9b5be802405797faecc284961b07c70ef`; release asset
`libduckdb-linux-amd64.zip`, sha256
`b845005f5132a7d8180057c35e14a7626632258782f871a90861b19c1c03841b`. Read:
`src/include/duckdb.h`, `src/main/capi/data_chunk-c.cpp`,
`src/include/duckdb/common/types/selection_vector.hpp`,
`src/common/types/selection_vector.cpp`. The header in the release asset is
byte-identical to the tagged source header.

**Verified upstream contract** (`idx_t = uint64_t`, `sel_t = uint32_t`, handle
= pointer to a one-pointer struct; all three marked stable since v1.5.6):

| function | upstream body | consequence |
|---|---|---|
| `duckdb_create_selection_vector(size)` | `new SelectionVector(size)`; storage `max(size, 1) * 4` bytes, uninitialised in release | non-null for size 0; the handle does not record its size |
| `duckdb_selection_vector_get_data_ptr(sel)` | `sel->data()` | same writable pointer until destroy; no bounds, no change notice |
| `duckdb_destroy_selection_vector(sel)` | `delete sel` | null is fine; double destroy and use after destroy are undefined |

**What was built.** The three functions as an owned, writable buffer
(`create_selection_vector`, `destroy_selection_vector`,
`selection_vector_get_data_ptr` in `crates/duckdb-selvec-compat/src/lib.rs`),
exported under DuckDB's names only with the opt-in feature `duckdb-symbols`.
Two Rust-side additions: `selection_from_positions` copies a borrowed native
list into a new buffer and returns a `CopyReceipt`; `positions` reads the
buffer back and refuses a length the handle does not own. `Planes`,
`Program`, `execute_sel`, `lower_sel` and the dt64 binding are unchanged.

**Three levels of claim.**

| level | what | evidence |
|---|---|---|
| ABI-tested | the three symbols, linked from C against the pinned `duckdb.h`, behave like DuckDB v1.5.6 on x86-64 Linux for the fixture below | `harness/run.sh`: one C program, two processes, identical output (9 lines); valgrind clean on the adapter |
| API-compatible by reading | signatures, types, size-0 rule, null destroy | source at the pinned commit |
| not compatible / not claimed | handles are NOT interchangeable with DuckDB's (theirs is a `SelectionVector *`); `duckdb_slice_vector` and `duckdb_vector_copy_sel` are not provided; allocation failure (abort here, C++ exception there); other platforms; any DuckDB C++ ABI | — |

Fixture: sizes 0, 1, 64, 256, 2048, 65,536; repeated and permuted positions
kept as written; a write through one pointer seen through a later one; two
live selections with separate storage; a pointer valid while another
selection is destroyed; 2000 create/destroy cycles; null destroy.
Out-of-range indices: the three functions have no index notion upstream and
check nothing, so there is no upstream behaviour to match. Here the consumer
refuses them (`execute_sel` → `SelOutOfRange`, tested after an external
write).

**Disable runs.** Six mutations, each asserted to apply once: 6 of 6 red in
`tests/contract.rs`; the three that touch the C surface are also red in the
DuckDB comparison (segfault, output diff, valgrind leak).

**Authority.** A buffer is caller-owned memory without frame, version, scope
or permission. A native selection reaches a writable pointer only by copy.
Positions read back are untrusted integers and never become a witnessed DR
selection; the crate has no dependency on dt64.

**Cost** (`examples/selvec_cost.rs`, medians of 201, two runs in
`.claude/probes/2026-11-12-dr-select-k-sv-3/`; 28 ns is the timer floor):

| K | native borrow | create + destroy | copy in + destroy | one read pass | range re-check | bytes copied | allocations |
|---|---|---|---|---|---|---|---|
| 64 | floor | 65–66 ns | 75–78 ns | 39 ns | 60–67 ns | 256 | 2 |
| 256 | floor | 66–67 ns | 82–83 ns | 55–56 ns | 195 ns | 1,024 | 2 |
| 65,536 | floor | 5.5–5.7 µs | 12.5–13.2 µs | 5.7–6.7 µs | 45–62 µs | 262,144 | 2 |

The native path pays none of this unless a C buffer is requested.

**R4 (`DictionaryVector`): measured upstream, not implemented.**
`harness/r4_upstream_probe.c` against the real library (output in the probe
folder):

| question | DuckDB v1.5.6 does |
|---|---|
| slice: view or copy? | view; `duckdb_vector_get_data` returns the same base pointer and the UNSELECTED values |
| permutation, repeats | kept (`7 3 3 0 5 7` reads `107 NULL NULL 100 105 107`) |
| NULL | validity is addressed through the selection |
| selection written after the slice | the vector follows the write: the slice shares the selection's storage |
| selection destroyed after the slice | the vector still reads correctly: shared ownership |
| nested slice | the second selection addresses the result of the first |
| `duckdb_vector_copy_sel(count, src_offset, dst_offset)` | `count` is an end index; `count - src_offset` entries are copied |

Gaps this opens: (1) an owned, single-owner buffer as built here cannot back
a slice, because DuckDB keeps the storage alive and aliased after
`destroy`; (2) a C caller of a sliced vector cannot see the selection
through `duckdb_vector_get_data`, so parity needs the copy path; (3) nested
composition and validity-through-selection have no counterpart in `Planes`.
Smallest falsifier for an R4 adapter: that probe file, run against it,
must print the same seven lines.

**R7 (`UnifiedVectorFormat`): read, not implemented.** It is internal C++
(`{ sel, data, validity, owned_sel, physical_type }`), not part of the C API.

| member | here today | gap |
|---|---|---|
| `data`, `physical_type` | `LaneRef::{I32, U32}` | other physical types |
| `sel` | the position slice passed beside `Planes` to `execute_sel` | not a member of `Planes`; COUNT / SUM only |
| `validity` | none per lane | NULL through a selection is not representable |
| `owned_sel` | none | no owned selection inside the access triple |

No `Planes` construction site was changed.

**Not done.** `duckdb_slice_vector`, `duckdb_vector_copy_sel`, dictionary or
unified-format compatibility, any authorization. The `Scope` issuer is not
touched.
