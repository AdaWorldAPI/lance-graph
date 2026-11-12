# Quack execution boundary: admission and materialisation lab (v1)

**Status:** LAB, measured 2026-11-12. No production API changed.
**D-ids:** D-QUACK-BOUNDARY-LAB, D-QUACK-FIELD-OPERAND-PROBE (`STATUS_BOARD.md`).
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
`lance-graph-dt64/src/{lib,lane}.rs`; `lance-graph-report/src/{bound_sel,lane_source}.rs`;
`lance-graph-report/tests/bound_sel.rs`; `lance-graph-quack/src/{lib,bind}.rs`
(non-test, earlier the same day). `lance-graph-dt64/src/lib.rs` and
`lane_source.rs` were read after the lab was built; what they changed is in
section 8, items 6, 10 and 11.

In part: `lance-graph-dt64/src/dr.rs` from `AxisKind` to the end of `Guards`
(`FrameRevisions`, `FrameStore`, `Frame`, `Scope`).

Not read: the rest of `dr.rs` (`MapTable`, `CodeTable`),
`lance-graph-report/src/{batch,boundary}.rs`, `lance-graph-mask-risc/src/{fuse,ternlog_dispatch}.rs`,
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
with `masks: &[]` and `Foreign::NONE`. The second step uses a lab-local
`Prepared { program, frame, fields }`; the frame comes from
`FrameStore::seal` and has no public constructor, and `fields` are the bound
field positions the program's operands were numbered over (section 7).

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

## 7. Field to operand (D-QUACK-FIELD-OPERAND-PROBE, `tests/field_operand.rs`)

Three index spaces are involved and they are different numbers:

| index | owner | in the probe |
|---|---|---|
| field position in the `ClassView` basis | DT64 | `amount` 1, `quantity` 5 |
| lane id (`LaneAddr::lane`) | the source | 31 and 22 in one source, 5 and 4 in another |
| operand index (`Col`, into `Planes::lanes`) | whoever assembles the planes | 0 and 1 |

`LaneBinding::borrow_all` yields the bound lanes in ascending field order.
A consumer that assembles `Planes::lanes` in that order gives a bound field
the operand index equal to its **rank among the bound fields**. The lab's
`operand(binding, field)` computes that rank from `LaneBinding::fields`; it
returns `None` for an unbound field.

Results (fields 1 and 5 bound, `SUM(quantity) WHERE amount > 25`):

| falsifier | result |
|---|---|
| equal case (every field bound, identity ids) | still equals the oracle; rank equals position there |
| sparse, non-contiguous binding | equals the oracle with operands 0 and 1 |
| operands numbered by field position | positions 1 and 5 over two lanes: `LaneOutOfRange(5)` from the executor. Positions 0 and 1 fit and silently answer another question unbound; admitted, `Operands` refuses because the program's `fields` are not the bound ones |
| another lane numbering | same program text, same answer; one source's binding does not read the other (`ForeignFrame`) |
| changed basis, or lanes renumbered without a new frame | `MappingChanged` before execution; a store that never saw the basis refuses it (`PredicateBasisChanged`) |
| unbound field | no operand; `LaneBinding::lane` says `NotBound`; a position outside the basis does not bind (`FieldNotInBasis`) |

What exists and what is missing:

- **Exists.** The lane order (`borrow_all`), the bound set
  (`LaneBinding::fields`) and Quack's name-to-`Col` surface
  (`Binder::field` returning `BoundField::col`, `lance-graph-quack/src/bind.rs`).
  A `Binder` answered from a binding carries the translation (tested).
- **Missing, minimal.** One consumer-side function from a bound field to its
  rank, and the rule that planes are assembled in `borrow_all` order. Neither
  belongs in mask-RISC or `Planes`. Without a typed operand, a `Col` built
  any other way cannot be told apart; the lab's `fields` on `Prepared` only
  catches a program that declares another field set.
- **Blocked by item 1 of section 8.** `Draft::bind` always conjoins
  `Binder::live`, a mask plane, so no `Draft`-bound query is admissible
  through the lane socket today (tested: BLOCKED). The same comparison
  lowered without the plane is admitted and right.

### 7a. Control: two stores, one basis, other behaviour

Two `BindStore`s register the class over the same view with a method that
differs in its `reads`.

| question | observed |
|---|---|
| is the other store's `MethodRef` admitted | no: `BindingMismatch`. A clone with identical behaviour: `ForeignStore` |
| is a `LaneBinding` made through one store valid against the other | yes: `check` passes and the fold runs. Both route the position to the class and hold the same basis witness |

The second row is recorded, not judged. `MethodRef` vouches for one
behaviour binding of one store; `LaneBinding` vouches for a field and source
interpretation and is silent on behaviour. Whether a lane binding should be
portable between stores with equal bases is a design decision. Nothing here
couples method admission to a lane read.

## 7b. Disable runs

Each made after commit `9fb3517a5`, with the patch anchor asserted, the test
binaries built, and the tree restored afterwards.

| disabled | went red |
|---|---|
| `admitted` takes raw lanes instead of `borrow_all` | cases B, C, E |
| the program-frame check removed | cases B, C |
| one 64-byte allocation added inside `admitted` | the ledger |
| `lower_group_by_auto` never takes the semantic fold | both fold tests, the ledger |
| `operand` returns the field position instead of the rank | five of the seven field-operand tests |
| the `Operands` check removed | the position-numbered test |
| `operand` answers for unbound fields | the unbound-field test |

The ledger also went red on its own during this work: the first version of
the `Operands` check allocated 8 bytes per run. It now compares in place.

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
5. **Field position to operand index.** Answered in the lab (section 7): the
   operand is the field's rank among the bound fields. No production code
   computes it or fixes the plane order, and a `Col` is an untyped `u16`.
6. **`Scope` has no issuer; `CoordSpace` is whatever the source says.** A
   `Scope` is a plain `{ id }`. A `CoordSpace` is two numbers the source
   reports: `AbiLanes` (`lance-graph-report/src/lane_source.rs`) derives them
   from the batch's source id and generation, the lab assigns them by hand.
   Scope equality is a consistency check and observes no revocation.
7. **Only the current frame is admitted.** No history read.
8. **The lab's frame check is stricter than a value-only program needs.** A
   program with no row-ordinal constant could survive a re-seal. Not
   explored.
9. **A `groups == 0` forest plan still contains a phase-1 `Keep`.**

10. **The one non-lab `LaneSource` cannot be constructed in a library
    build.** `AbiLanes`'s only constructor, `unwitnessed`, is `#[cfg(test)]`,
    because nothing certifies which reader scope a batch serves. So the seam
    of section 3 has no production-constructible source today; it admits
    `I32` and `U32` lanes only and has no writer.
11. **A `LaneBinding` does not record which `BindStore` made it.** `check`
    accepts any store that routes the position to the class with the same
    basis witness (section 7a). `BindStore::admit` does check the issuing
    store and the binding digest. The two are different promises; whether
    the first should name its store is undecided.

None of these is closed by this lab, and none is closed by adding a field to
`Program`.

## 9. Out of scope here

DuckDB compatibility as a second engine; a native vector format; selection
or validity fields in `Planes`; a scope issuer; the write-side owner
provenance at `BatchWriter::cast` (the #2044 audit).
