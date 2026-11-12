# Quack execution boundary: admission and materialisation lab (v1)

**Status:** LAB, measured 2026-11-12. No production API changed.
**D-id:** D-QUACK-BOUNDARY-LAB (`STATUS_BOARD.md`).
**Crate:** `crates/quack-boundary-lab` (workspace member, one CI line in `rust-test.yml`).
**Separate from:** `schema-wardrobe-ownership-audit-v1.md` (PR #2044), which audits the write side.

## 1. Question

A `Program` (`lance-graph-mask-risc/src/ir.rs`) names lanes and planes by
index. It carries no source, frame or row order, and that stays so: the
program is reusable algebra. The question is where a consumer establishes
that the operands it hands the executor are the ones the program was lowered
for, and what an execution writes besides the answer.

Decisions this lab works under (taken in session, 2026-11-12):

- `Want::Rows` stays the `Draft` default. No API change.
- `lower_group_by_auto` keeps its order: exact fold first, forest second.
- No source identity is added to `Program`.
- Resident-plane preconditions are to be checked at an admission boundary,
  not inside the physical IR; where no ratified authority exists the case
  stays BLOCKED.

## 2. What was read

In full, first to last non-test line: `lance-graph-mask-risc/src/{lib,ir,exec,reference,value,sel}.rs`;
`lance-graph-dt64/src/lane.rs`; `lance-graph-report/src/bound_sel.rs`;
`lance-graph-report/tests/bound_sel.rs`; `lance-graph-quack/src/{lib,bind}.rs`
(non-test, earlier the same day).

In part: `lance-graph-dt64/src/dr.rs` from `AxisKind` to the end of `Guards`
(`FrameRevisions`, `FrameStore`, `Frame`, `Scope`).

Not read: `lance-graph-dt64/src/lib.rs` (`BindStore` is used as the report
test uses it), the rest of `dr.rs` (`MapTable`, `CodeTable`),
`lance-graph-report/src/lane_source.rs`, `lance-graph-mask-risc/src/{fuse,ternlog_dispatch}.rs`,
every test module of the crates above, `lance-graph-cypher-quack`.

## 3. The seam that exists

`LaneBinding` (`lance-graph-dt64/src/lane.rs`) is the consumer-side
admission seam for lanes. One `LaneBinding::check` covers:

| what | how it is established |
|---|---|
| issuing source | `LaneSource::frames` must be the caller's `FrameStore` (`same_store`, identity) |
| current frame | `FrameStore::check`: a foreign, stale or re-sealed frame is refused and the moved revision named |
| class and method binding | route at the store's current version; basis witness; each bound field's `predicate_iri` |
| row order | `LaneAddr::space` against the request's `CoordSpace` |
| reader context | `LaneAddr::scope` against the request's `Scope` (consistency, not a grant) |
| required lanes | `borrow_all`: every bound lane, borrowed from the source after the one check |
| selection criteria | `LaneBinding::selected` for a source-held position list |

`fold_bound_selection` (`lance-graph-report/src/bound_sel.rs`) already uses
it for a position fold. Nothing uses it for a `Program`.

The lab's `admitted` function does: `borrow_all` → check the frame the
program was lowered under against the same source's store → `execute_into`
with `masks: &[]` and `Foreign::NONE`. The second step uses a lab-local pair
`Prepared { program, frame }`; the frame comes from `FrameStore::seal` and
has no public constructor.

## 4. Cases A to F (`tests/admission.rs`)

The program is `SUM(qty) WHERE row IN [1000, 9000) AND amount > 25`, lowered
once by `lance_graph_quack::lower`. Its range is in row ordinals. The oracle
(`src/oracle.rs`) names rows by value. Two sources hold the same 40,003
logical rows, one in id order and one reversed, with equal `CoordSpace`
numbers and equal frame revisions.

| case | unbound (negative control) | admitted |
|---|---|---|
| A correct source | equals the oracle | equals the oracle |
| B other source, same lane indices | executes; plausible and wrong | old binding: `Frame(ForeignFrame)`. Fresh binding over the other source with the old program: `Program(ForeignFrame)` |
| C same source after a coordinate revision | executes; wrong | old binding: `Frame(CoordinatesChanged)`. Rebound, old program: `Program(CoordinatesChanged)`. The by-value form prepared under the new frame is admitted and right |
| D kept mask of another program as a foreign plane | fresh: right, and indistinguishable from a resident plane. After the foreign table changed: still answers, with the old population | BLOCKED: no owner issues a foreign plane |
| E valid frame, another view / scope / row order | a caller mapping lanes by its own view sums another column | `MappingChanged`, `Scope`, `CoordinateSpace` |
| F NULL rows with stale non-zero bytes | naive two-valued run is wrong for the predicate and for its negation; `Filter::sql_where` over a caller-built validity plane is right | validity as a witnessed lane gated inside the program: right. The `sql_where` program: BLOCKED, it reads a plane |

In B the two sources agree on every number they carry. What tells them apart
is the identity of the store that sealed the frame.

## 5. Group lowering audit (`tests/group_audit.rs`)

`lower_group_by_auto` (`lance-graph-quack/src/lib.rs`):

| aggregate | lowering | carries a population mask between programs |
|---|---|---|
| `SumI32`, `Count`, `MinI32`, `MaxI32` | `Folded`, one program, K-slot sink | no |
| `Any`, `All`, `Rows` | `Forest`: one `Keep` program and K group programs | yes: `words_for(n_rows)` words, read back as `planes.masks[filter_plane]` |
| any aggregate with `groups == 0` | `Forest` with no group programs | the phase-1 `Keep` program is still in the plan; nothing reads its mask |
| `BlendI32` | `Err(LowerError::GroupedBlend)` | not lowered |

The fold and the forest give the same COUNT per group. An empty group is
reported through `normalize_group_sink`'s presence mask, not as a value.
A forest's phase-2 program has no admitted form: its input is a mask the
caller made.

## 6. Materialisation ledger (`tests/ledger.rs`)

40,003 rows, 8 groups. Bytes; no time was measured.

| workload | programs | passes | allocated | scratch | demanded sink | intermediate |
|---|---|---|---|---|---|---|
| 1a `COUNT WHERE amount > t` | 1 | 2 | 0 | 2,056 | 0 | 0 |
| 1b `SUM(qty) WHERE amount > t` | 1 | 2 | 0 | 2,056 | 0 | 0 |
| 2a grouped `COUNT` (folded) | 1 | 2 | 0 | 2,056 | 64 | 0 |
| 2b grouped `SUM` (folded) | 1 | 2 | 0 | 2,056 | 64 | 0 |
| 3 demanded `Keep` mask | 1 | 2 | 0 | 2,056 | 5,008 | 0 |
| grouped `ANY` (forest, unbound) | 9 | 18 | 0 | 4,112 | 8 | 5,008 |

- Scratch is one 256-word tile per slot plus one bitmap word, whatever the
  row count.
- The executor borrows `Planes` shared, so scratch and the sink are the only
  memory a run can write. Lane addresses and a checksum over every lane are
  equal before and after; the lanes the admitted path reads are the source's
  own allocations (pointer-equal).
- Every line allocates zero bytes, the forest too. The intermediate column is
  where the forest differs.
- For a predicate-built `Keep`, the tiled path writes each tile into scratch
  and copies it into the sink (`Terminal::Keep` arm of `execute_compiled`).
  The fused keep lowerings write the sink once; this workload does not take
  them because its mask comes from a lane predicate.

## 7. Disable runs

Each made after commit `9fb3517a5`, with the patch anchor asserted, the test
binaries built, and the tree restored afterwards.

| disabled | went red |
|---|---|
| `admitted` takes raw lanes instead of `borrow_all` | cases B, C, E |
| the program-frame check removed | cases B, C |
| one 64-byte allocation added inside `admitted` | the ledger |
| `lower_group_by_auto` never takes the semantic fold | both fold tests, the ledger |

## 8. Contracts that are not satisfied

1. **Mask planes have no issuer.** `LaneSource` hands out lanes and
   `SelectionSource` position lists. `Planes::masks` is whatever the caller
   passes. A validity plane, a liveness plane, a focus and every forest
   phase-2 input are therefore unadmitted. BLOCKED.
2. **Foreign planes have no residency admission.** `Filter::Semijoin`'s
   precondition (a resident plane, never another program's `Keep`) is a
   comment over a `u16`. Case D shows the executor accepts either. BLOCKED.
3. **Foreign lanes and the fk's coordinate domain.** Nothing establishes that
   an fk lane's values are positions in the foreign source's current row
   order. Not tested here.
4. **No pairing of a program with the frame it was lowered for.** `Prepared`
   is lab-local. `PrefixLowering::Bound` carries `lane_version` and
   `lane_digest` out of Quack and nothing consumes them.
5. **Field position to lane index.** `Col` is a lane index; a binding knows
   field positions. The lab binds every field and equates the two.
6. **`CoordSpace` and `Scope` are hand-assigned values.** No authority issues
   either. Scope equality is a consistency check and observes no revocation.
7. **Only the current frame is admitted.** No history read.
8. **The lab's frame check is stricter than a value-only program needs.** A
   program with no row-ordinal constant could survive a re-seal. Not
   explored.
9. **A `groups == 0` forest plan still contains a phase-1 `Keep`.**

None of these is closed by this lab, and none is closed by adding a field to
`Program`.

## 9. Out of scope here

DuckDB compatibility as a second engine; a native vector format; selection
or validity fields in `Planes`; a scope issuer; the write-side owner
provenance at `BatchWriter::cast` (the #2044 audit).
