# D-VAS implementation ledger v1

**Status:** IN PROGRESS. Implements the roadmap
`.claude/plans/virtual-address-superpowers-roadmap-v1.md` in the order R0 to
R6 (directive of 2026-11-13). Research baseline: PR #2060
(`d-virtual-address-algebra-1.md`, sections 11 to 18).
**Deliverables:** `D-VAS-A1`, `D-VAS-A2`, `D-VAS-C2`, `D-VAS-D1`,
`D-VAS-E0` in R1; `D-VAS-A3`, `D-VAS-B2` in R2; `D-VAS-C1`, `D-VAS-C3`,
`D-VAS-C4`, `D-VAS-C6` in R3; `D-VAS-D2`, `D-VAS-D3`, `D-VAS-D4`,
`D-VAS-E1`, `D-VAS-F` in R4 to R6.

Rules carried from the roadmap: ABI512, `NodeRow`, `VALUE_TENANTS` and the
`MailboxSoA<N>` layout stay as implemented; no second column owner, identity
map, executor or `FrameStore` substitute; an unsupported boundary is a
typed refusal with its blocker, never a passing fake.

## R1: E0 on the real owner

### Choices, each the smallest the code supports

1. **A1, publication = the owner's shared borrow.** Section 18 of the lab
   plan shows no mailbox stamp accounts for its writes. The owner now issues
   `MailboxSoA::published(&self) -> Published<'_, N>`
   (`cognitive-shader-driver/src/mailbox_soa.rs`): no new state, no layout
   change, not constructible outside the crate. The lane socket holds one
   `FrameStore` per `Published`. Rejected: a revision counter, because 19
   public columns can be assigned without it; making them private is an API
   change across at least four consumer crates and is left as the decision
   that would let a frame outlive a borrow.
2. **A2, a consumer-side bridge crate.** `cognitive-shader-driver` does not
   depend on DT64 and should not. `crates/lance-graph-mailbox-lanes` depends
   on both and implements `LaneSource` for `MailboxLanes<'_, N>`.
3. **C2, route-bound admission in DT64.** `BindStore::admit_routed(route, r)`
   and `Refusal::NotRoutedClass` (`lance-graph-dt64/src/lib.rs`), the
   comparison R2IL's private `check_route` makes.
4. **C1(a), direct class.** The position routes to a data class that carries
   its own `MethodBinding`s. No `0xF###` lens is involved in R1.
5. **D1.** The binding is made over the admitted method's `reads`
   (`LaneRequest::fields`); every other field is `NotBound`.

### Ledger

| Stage | Contract (file) | Independent oracle | Positive | Negative / disable | Status |
|---|---|---|---|---|---|
| A1 | `Published`, `MailboxSoA::published` (`mailbox_soa.rs`); `FrameStore::check` (`dr.rs:462`) | the owner's stamps and bytes | a fresh binding reads the new value | binding and frame of an earlier publication: `ForeignFrame`, across a `set_meta` that moves no stamp; two sockets over one publication do not share frames | INTEGRATION VERIFIED for one borrow; a frame that outlives the borrow is BLOCKED (no revision the owner moves) |
| A2 | `MailboxLanes: LaneSource` (`lance-graph-mailbox-lanes/src/lib.rs`) | `owner.meta[r].0`, `edges_raw`, `temporal`, generator functions | 40 heterogeneous rows per column, slice pointer equal to the owner's column | lane limited to the populated prefix; `f32`, `u16`, `u8`, `[u8; 12]` columns refused by name; a foreign address, row order or scope refused. Disables: all `N` rows lent; a lane number lending another column; the socket's space/scope check; its frame check | INTEGRATION VERIFIED for `u32` and `u64` columns; other kinds NOT SUPPORTED (`LaneRef` has no such lane) |
| C1(a) | `BindStore::resolve_at`, `Resolved::methods_on` | the registered bindings | the data class's method is the candidate ON the field | n/a | INTEGRATION VERIFIED (direct class) |
| C2 | `BindStore::admit_routed` | `routed_at` at both versions | own reference admitted | foreign `MethodRef` with the same `MethodId`; a re-route; another store's reference. Disable: comparison skipped | INTEGRATION VERIFIED |
| D1 | `LaneRequest::fields`, `LaneRefusal::NotBound` (`lane.rs`) | the mask | one bound field for `read_meta` | fields outside `reads`; a field outside the basis | INTEGRATION VERIFIED |
| E0 | the chain above | owner arrays | `tests/e0.rs`, 7 tests | stale publication, foreign source, wrong route, wrong basis, foreign method, undeclared field, position past the lane, another owner's row order, another scope | INTEGRATION VERIFIED on `MailboxSoA<64>` |

Disable runs (each anchor asserted once, each restored): route comparison
skipped, all rows lent, lane number lending another column: exit 101. Two
further disables (the socket's own space/scope check, its own frame check)
passed at first, because DT64 repeats both checks above the socket. A test
that asks the socket directly was added; both then exit 101.

### What R1 does not establish

- **Reader scope.** DR has no issuer of a `Scope`. The socket compares the
  scope the caller asserts. Authorization is out of the roadmap.
- **Column names.** `urn:mailbox:<column>` is this crate's naming; no
  contract names the mailbox's columns.
- **Replay.** No frame, binding or address survives a publication.
- **Cost.** Nothing in R1 is measured.
- Selections, `NodeGuid`, guards and effects: R2 to R4.

## R2: selection and identity

### A3, source-issued selections (done)

`MailboxLanes` implements `SelectionSource`. It issues an order by a column,
a filter by a column comparison and the concatenation of two issued
selections; a caller cannot hand in positions. The lists are views of one
publication and live in the socket. `dr256_ternlog_lab::OrderIndex` /
`SealedOrder` are not reused: they live in a lab crate and take `i32` keys.

| Stage | Contract (file) | Independent oracle | Positive | Negative / disable | Status |
|---|---|---|---|---|---|
| A3 | `MailboxLanes: SelectionSource`, `issue_order`, `issue_where`, `issue_concat` (`lance-graph-mailbox-lanes/src/lib.rs`); `LaneBinding::selected` (`lane.rs:463`) | a sort of `(value, row)` pairs, a filter and per-row counts over `owner.meta` | 48-row order equal to the oracle and really reordered; a filter keeping 24 rows in row order; a bag where kept rows occur twice; positions lent, not rebuilt | unissued name; another mailbox; a second socket over the same publication; a foreign frame or address asked directly; concatenation of an unissued name; the 65,537th name. Disables, all exit 101: sort removed, `<` as `<=`, concat deduplicating, frame check removed, name wrapping, space/scope check removed | INTEGRATION VERIFIED on `MailboxSoA<64>` |

Tests: `tests/selection.rs`, 3 tests, committed before the run, green on the
first run.

Not established: a selection that outlives its publication (same limit as
the frame); filters other than one unsigned column comparison; cost.

### B1, `NodeGuid` to a row: BLOCKED

Read in full: `lance-graph-contract/src/soa_view.rs` (the
`row_for_local_key` contract), `lance-graph/src/graph/mailbox_scan.rs`
1..375 (its one consumer, `match_node_by_local_key`),
`lance-graph-dt64/src/dr.rs` (`MapTable`, `LocalMap`).

- The declared seam is `MailboxSoaView::row_for_local_key(u64) ->
  Option<usize>`. Its doc names the authority: "an owner that stores keys
  (the in-RAM `MailboxSoA`, once it carries a `local_key` column) overrides
  this". `MailboxSoA` has no such column and does not override it.
- Closing it on the owner is a new column of `MailboxSoA<N>`: a layout
  change, which this roadmap excludes without a separate decision.
- The argument is `NodeGuid::local_key()`, the trailing 6 bytes read as the
  V1 `family ++ identity` tail. It discriminates only after the 10-byte
  prefix is resolved, and under the V3 4+12 facet it is a legacy reading.
- `MapTable` keys a `u64` inside a `u32` tile. No authority maps a GUID
  prefix to a tile, so a 128-bit identity cannot be bound through it either.

Nothing was built in place of it. Decision needed: a `local_key` (or full
key) column on `MailboxSoA`, or another owner of the key index.

### B2, version-bound coordinate (done, on `MapTable` ids)

`coords::Coordinates` (`lance-graph-mailbox-lanes/src/coords.rs`) pairs one
tile of DT64's existing `MapTable` at one sealed version with a lane
binding. The id is the map's own `u64`, not a `NodeGuid`. The source version
(publication) and the coordinate version (map) are separate: advancing the
map writes nothing to the mailbox and leaves the lane binding valid.

| Stage | Contract (file) | Independent oracle | Positive | Negative / disable | Status |
|---|---|---|---|---|---|
| B1 | `MailboxSoaView::row_for_local_key` (`soa_view.rs:126`) | n/a | n/a | n/a | BLOCKED: no key column on the owner |
| B2 | `Coordinates::bind`, `row_of`; `MapTable::map`, `Frame::same_issue` (`dr.rs`) | two permutations over `owner.meta` | 48 ids read through assignment 1, then through assignment 2, source bytes equal before and after | the first binding after the map advances: `StaleMapping` for all 48 ids; an unmapped id; a row number offered as an id; another table's frame; an unknown tile; a row past the lane; a repeated id. Disables, all exit 101: currency check removed, row bound removed, the current frame taken at read time | INTEGRATION VERIFIED for `MapTable` ids on `MailboxSoA<64>`; the map's producer for a mailbox does not exist (B1), the test fills it |

Tests: `tests/coords.rs`, 2 tests, committed before the run, green on the
first run.

### A4, typed selection folds: NOT SUPPORTED for the mailbox's lane kinds

- `lance_graph_mask_risc::execute_sel` (`sel.rs:72`) admits two folds,
  `CountGtI32` and `SumI32`, over an `i32` lane. Its kernels are
  `ndarray::simd::{count_gt_i32_at, sum_i32_at}`
  (`ndarray/src/simd_masking_ops.rs:801, 835`). No `u32` or `u64`
  selection kernel exists there.
- The mailbox lends `u32` and `u64` lanes, so the shipped witnessed fold
  (`lance_graph_report::bound_sel::fold_bound_selection`) refuses them with
  `ExecError::LaneKind` after the witness has held. That refusal is pinned
  on the real owner (`tests/selection.rs`,
  `the_shipped_fold_refuses_the_mailboxs_lane_kinds`).
- Lifting it needs `sum_u32_at`, `count_gt_u32_at` and a `u64` sum with a
  stated overflow bound in `ndarray::simd` (all three backends and a parity
  test, per that repo's consumer contract), then two `SelFold` variants. A
  scalar loop in this repo instead would be a second kernel beside the
  facade and was not written.

Until then a consumer reads selected mailbox values one by one through
`LaneBinding::selected`, which is what the A3 tests do.

## R3: behaviour provenance and WHEN

### C1, the lens association is the route (done; a second table not built)

`lance-graph-dt64` already carries the behaviour lens: a classid whose right
half is `0xF###` (`is_behaviour_lens`, `class_view.rs:1053`) has a basis of
its own, read by its full classid (`ClassView::lens_fields`, `basis_at`,
`lib.rs:1304`), and is registered and routed like any class. A position
routed to a lens has the lens's methods, basis and guard; `admit_routed`
refuses every other class. That is association (a) of the roadmap, and it is
versioned by the one log DT64 versions, the route log.

Association (b), a table `data class -> lens` with a version of its own, was
not built. It would let the behaviour of a classid change without a new
classid and a re-route, which D5 excludes ("a classid's behaviour never
changes once registered"; an override is `register_derived` plus a re-route,
`lib.rs:32-40`). Building it is a change to that rule, so it is reported as
an open decision, not implemented around.

Consequence: one position has one classid. "This data class, seen through
that lens, at the same position and time" is not expressible; route the
position to the lens (whose basis names the same mailbox columns) instead.

Tests: `tests/lens.rs`, 4 tests on a real `MailboxSoA<64>`. They passed on
their first run, because they exercise shipped DT64 code plus R1's
`admit_routed`. Disables in DT64 (anchors asserted, restored): admission
without the route comparison (2 of 4 fail), a lens read through its left half
(4 of 4 fail).

### C3, a guard lane is a field position (done)

`validate` (`lance-graph-dt64/src/lib.rs`) now walks a guard's residuals and
refuses a lane at or past the class's field count with
`Refusal::GuardLaneOutOfBasis { class, method, lane }`. `Sym` leaves name no
field and are not judged.

- Basis for the reading: every caller that registers a guard already uses
  lane = field position (`dt64-admission-probe`, `foundry-parity-lab`,
  `quack-boundary-lab`, `cg-beh-probe`, the lab of #2060).
  `grammar-constraints-probe` pinned the gap as a finding
  (`dt64_does_not_validate_guard_lanes`); that test is re-pinned to the
  refusal.
- Not enforced: that a guard's fields are inside `reads`. `reads` has no
  stated relation to the guard in the contract, and one probe relies on
  their being equal, none on a rule.
- Tests: `lance-graph-dt64/tests/guard_contract.rs` (2 tests), committed
  before the change; the negative failed first. Disable (check switched
  off): exit 101.
- Dependents run after the change: DT64 (120 tests), R2IL, quack-boundary-lab,
  virtual-address-lab, cg-beh-probe, dt64-admission-probe,
  foundry-parity-lab, grammar-constraints-probe, hidden-state-twins-probe,
  cg-defuse-probe: all green.

### C4, `SetExpr` to a Quack filter (done, library)

`lance_graph_quack::set::lower_set(&SetExpr, Logic)` returns `All`, `Empty`
or a `SetFilter`: the normalized set with the lanes and planes it names. The
caller says which lanes it supplies (`two_valued(lanes)`); column `i` of the
filter reads `lanes[i]`. The reading of a residual lane and of a `Sym` is the
one `lance-graph`'s `fold::set_lower` already uses. A DT64 `Guard` is
`{ expr, logic }`, so a guard lowers by passing its two fields; Quack does
not depend on DT64.

Evidence (`lance-graph-quack/tests/set_filter.rs`, 5 tests, committed against
a stub first, all 5 failed there):

- 1500 random expressions (lanes of both kinds, planes, constants, depth 4)
  over 197 rows: Quack through `lower` and through `lower_fused` equals a
  scalar evaluator written in the test, bit for bit.
- The same 1500 through `set_lower::SetPlan::unsplit(..).keep(..)` equal the
  scalar evaluator. `set_lower.rs` is compiled into the test binary from its
  own source file (`#[path]`), because the crate that owns it also builds the
  storage stack; its own 6 unit tests run in that binary too.
- Disables (anchors asserted, restored): `!=` lowered as `==` (2 tests fail);
  column = lane number instead of the supplied position (1 fails).

On the mailbox (`lance-graph-mailbox-lanes/src/when.rs`): `when(lanes,
store, view, request, guard)` binds the guard's fields through DT64 under
the publication's frame, has the socket lend them, lets Quack lower and
mask-risc run, and returns a `Fired` mask that carries its binding.

- `tests/when.rs`, 5 tests on a real `MailboxSoA<64>`, committed against a
  stub first (4 failed; the fifth pins C3). 400 random guards over `meta`
  and `last_active_cycle` equal a scalar evaluator over the owner's arrays,
  row by row, with rows past the populated prefix holding values that would
  satisfy most guards.
- Disables: lanes lent in reverse order (2 fail), `holds` without the
  binding check (1 fails), plane not refused (1 fails).
- Not established: that the guard reads ONLY its fields (the binding is made
  over them, but no test would notice a wider one); cost.

### C5, mask planes: BLOCKED

`when` refuses a guard that names a `Sym` with `PlaneUnissued`. No authority
issues a plane with a frame, a row order and a population:

- `MailboxSoA` has no plane column; `LaneSource` and `SelectionSource` are
  the only source sockets in `lane.rs`.
- The library readings of `Sym(i)` disagree: plane index `i` in
  `set_lower.rs:450` and in `lower_set`; a producer or lexicon digest in
  `hidden-state-twins-probe` and `grammar-constraints-probe`; a field in
  `foundry-parity-lab`'s `guard_fields`. A socket would have to pick one.
- `dr256-ternlog-lab`'s `bound_planes` says of itself that it is not an
  authority and issues nothing.

Needed: a decision on what `Sym(i)` names, then a third source socket.
`lower_set` already carries a plane leaf through unchanged, so the lowering
does not need to change when that exists.

### C6, UNKNOWN: done in the library, NOT SUPPORTED on the mailbox

`SetFilter::where_true(lanes, nullable)` applies Quack's `sql_where` to a set
normalized with `normalize_kleene`. A Kleene set is never handed out as a
two-valued filter, and a two-valued normal form is never rewritten for NULLs.

- Evidence: 1500 random expressions with about one NULL in four per lane
  equal an independent Kleene evaluator on the TRUE rows, through both
  lowerings; the NULLs changed the answer on more than 300 of them.
- Disables: without `sql_where`; Kleene input normalized two-valued. Each
  fails the Kleene test.
- A method does not fire on UNKNOWN: the filter keeps TRUE rows only, which
  is what DT64's `Guard` doc states.
- On the mailbox: refused with `ValidityUnstated`. The mailbox states no
  validity for its columns; "every populated row is valid" would be this
  crate's assertion, and it would make every Kleene guard two-valued.

### R3 ledger

| Stage | Contract (file) | Independent oracle | Positive | Negative / disable | Status |
|---|---|---|---|---|---|
| C1 | route log, `is_behaviour_lens`, `lens_fields`, `admit_routed` (`lance-graph-dt64/src/lib.rs`) | owner arrays, registered bindings | a routed lens supplies its method, basis and guard on `MailboxSoA<64>` | unrouted lens with the same basis and `MethodId`; the data class; a route captured earlier; a re-route; a lens unknown by its full classid. 2 disables | INTEGRATION VERIFIED for a routed lens; a `data class -> lens` table NOT BUILT (conflicts with D5) |
| C3 | `validate`, `Refusal::GuardLaneOutOfBasis` (`lance-graph-dt64/src/lib.rs`) | the declared field count | guards inside the basis register | a lane one past the basis, nested, both kinds, lane 300. 1 disable | SOURCE VERIFIED (DT64 + 9 dependent crates) |
| C4 | `quack::set::lower_set`; `mailbox_lanes::when::when` | scalar evaluator; `set_lower`; owner arrays | 1500 random sets; 400 random guards on the real owner | wrong comparison, wrong column, reversed lanes, stale answer, wrong kind, lane outside the basis. 5 disables | INTEGRATION VERIFIED for `u32` equality guards on `MailboxSoA<64>`; library parity LAB VERIFIED |
| C5 | none | n/a | n/a | `PlaneUnissued` pinned | BLOCKED: no plane issuer, three readings of `Sym` |
| C6 | `SetFilter::where_true`, `Filter::sql_where` | independent Kleene evaluator | 1500 random sets with NULLs | 2 disables; logic cannot be swapped | LAB VERIFIED in Quack; NOT SUPPORTED on the mailbox (`ValidityUnstated`) |

### What R3 does not establish

- A signed or ordered comparison on a mailbox column: its lendable columns
  are `u32` and `u64`, a residual is `I32` (six comparisons) or `U32`
  (equality), and mask-risc refuses the kind. So a mailbox guard is a
  Boolean combination of `u32` equalities.
- Any cost. The mask is one `Vec<u64>` per evaluation, the lowering is done
  per call.
- A guard over a selection as a bag: the mask is membership.

## R4: DO

### D2, the executor is R2IL's binding seam, under a gate (done)

`lance_graph_r2il::binding` is the one place that runs a `MethodBinding`'s
effects (`prepare` / `run`, through `FoldDialect` and mask-risc). It refused
every guarded binding. It now has a second pair:

- `prepare_gated` accepts a guarded, two-valued binding and marks the plan.
- `run_gated(.., &Gate { plane, guard })` restricts every fold of the plan
  (`POP_COUNT`, `SUM`, `MIN`, `MAX`, `ANY`, `KEY_RUNS`) to the rows of a
  resident plane, by the dialect's own `IntAnd` (`call_shared`, so a
  predicate a later step reads again is not rewritten). No opcode, kernel or
  evaluator is added; the seam still does not evaluate WHEN.
- `run` refuses a gated plan (`GateRequired`); `run_gated` refuses an
  ungated one (`NotGuarded`) and a gate evaluated for another guard
  (`GateMismatch`, compared against the `GuardWitness` the method reference
  carries). `ALL` is refused in a gated plan: `ALL(m AND gate)` asks whether
  every row of the table holds. The native path (`prepare_native`) refuses a
  gated plan.
- Precedent for a kept filter presented as a plane to a second program over
  the same table: `lance_graph_quack::GroupPlan`.
- Not checked, because nothing in R2IL can check it: that the plane's bits
  are that guard's rows over these planes under this frame. The caller that
  evaluated the guard vouches for it; on the mailbox that caller is `act`,
  which evaluates and executes under one publication.

Tests: `lance-graph-r2il/tests/integration/binding.rs`, 2 added, committed
against stubs first (both failed). Oracle: scalar loops over the lane
slices. Four disables (folds not gated, any gate accepted, a gated plan run
ungated, `ALL` prepared): each exit 101.

### D3, write-back through the owner's mutator (done)

`lance_graph_mailbox_lanes::act`:

- `act(lanes, store, view, ctx, method)` admits by route, evaluates WHEN,
  binds operands over `reads`, runs the effects gated, and returns `Intent`s:
  column, value, fired rows as one bit per row, and the owner's cycle at
  publication. It writes nothing.
- `apply(&mut owner, &intent)` hands each cell to `MailboxSoA::write_row`
  with the intent's cycle, so the owner's own gate refuses a write made for
  a cycle it has left (`Stale`). A cell that already holds the value is not
  written. An intent of another mailbox is refused.
- The publication ends before the write by construction: `apply` needs
  `&mut`, so no frame, binding or lent lane of the read survives it.

Semantics chosen, because DT64 states ON as a field mask and WHEN as a guard
over the population and nothing more: the guard's rows are where the method
fires, and an effect's scalar is written to its target field at every fired
row. A reading that fires only on the changed row is not implemented.

Limits: writable columns are `meta` and `temporal`; the two cycle stamps are
refused (`NotWritable`); `edges` is not wired. Mailbox lanes are `u32` and
`u64`, so the executable effects are `IntEqual` predicates, Boolean
combinations and counts; `SUM`, `MIN`, `MAX` need an `i32` lane.

### D4, ON from a real change, bounded (done)

`cascade(owner, store, view, ctx, changed, max_rounds)`: a round publishes
once, runs the methods ON the changed columns in ascending `MethodId` order
against that one publication, then applies their intents. The columns the
owner really changed are the next round's ON. It stops when a round changes
no cell (`settled`) or at `max_rounds` (`settled == false`). A write that
leaves a cell as it was triggers nothing.

## R5: E1 on the real owner

`lance-graph-mailbox-lanes/tests/e1.rs`, 4 tests on a real `MailboxSoA<64>`,
committed before the `act` module existed. All four passed on the first run
of the implementation, including ledger values written beforehand.

Oracle: `Model`, the two methods written out by hand over plain vectors. It
shares no code with DT64, Quack, mask-risc, R2IL or the socket.

| Ledger | What is compared | Result |
|---|---|---|
| guard | per round: which methods fired, on which rows; rows scanned | equal to the model in all 3 rounds; 40 rows per guard program |
| operands | lanes the socket lent per method | guard fields + 1 declared read (3 for `tally`, 2 for `mark`); 0 before admission |
| effects | value computed, cells changed, steps, programs, owner stamps | equal to the model; 2 steps and 1 program per method; every changed cell stamped with the write's cycle |

Final state: the owner's `meta`, `last_active_cycle` and `temporal` equal the
model's cell for cell; rows past the populated prefix are untouched.

Seven disables on `act.rs` (anchors asserted, restored): folds gated by every
row; `apply` writing every row; `apply` with the owner's current cycle;
`apply` for any mailbox; every write counted as a change; admission without
the route; an unchanged cell written anyway. Each exit 101.

Not in E1, each for a stated blocker: `NodeGuid` (B1), mask-plane guards
(C5), three-valued guards on the mailbox (C6), typed selection folds (A4).
E1 therefore starts from a source-local row, not from a `NodeGuid`.

## R6: cost (measured, not optimised)

`lance-graph-mailbox-lanes/examples/cost.rs`, release build, 40 rows of a
`MailboxSoA<64>`, median of 9 batches, allocations counted by a counting
global allocator. Two runs on the session's machine; the second in brackets
where it differs by more than 5 %.

| step | ns per call | allocations | bytes allocated |
|---|---|---|---|
| direct: sum 40 `meta` values from the owner's array | 2.7 | 0 | 0 |
| scalar interpreter: tally (guard, count, 40 rows) | 11.9 | 0 | 0 |
| eager: copy 40 rows into structs, then tally | 50.0 | 1 | 640 |
| `MailboxLanes::new` | 117 (102) | 3 | 80 |
| `admit_routed` | 692 | 17 | 2888 |
| `bind_lanes`, one field | 336 (361) | 3 | 72 |
| checked lane borrow + sum 40 values | 81 | 0 | 0 |
| WHEN: lower + bind + run, 2 fields | 2061 | 31 | 3767 |
| `act`: admission + WHEN + operands + gated DO | 5505 (6167) | 89 | 11140 |
| `apply`, one intent (reset subtracted) | 72 | 0 | 0 |
| `cascade`, 3 rounds, 2 methods (reset subtracted) | 18639 | 285 | 35751 |

What the numbers say:

- Reading a lent lane costs a check (about 80 ns per borrow here) and then
  is the owner's array: 0 allocations, 0 bytes copied.
- Everything above that is fixed cost per call, not per row: at 40 rows
  `act` is about 500 times the hand-written interpreter. It prepares cold on
  every call (guard lowered, lanes bound, plan prepared, dialect scratch
  allocated). Stage F's intent, normalise once and check once per
  population, is NOT implemented; this table is the baseline for it.
- An intent carries 8 bytes of rows for 64 rows.

A first version of the last two rows built an owner per call (about 50 us and
398 KB); the differences came out negative and were discarded. The rows above
reuse one owner and subtract a 9.5 ns reset.

Not measured: larger populations (a mailbox is 64 rows here), other
machines, the cost split inside `act`.

## Ledger, R4 to R6

| Stage | Contract (file) | Independent oracle | Positive | Negative / disable | Status |
|---|---|---|---|---|---|
| D2 | `prepare_gated`, `run_gated`, `Gate` (`lance-graph-r2il/src/binding.rs`) | scalar loops over the lanes | gated count and sum; all-rows and no-rows gates | no gate, wrong gate, ungated plan, absent plane, `ALL`, not permitted. 4 disables | SOURCE VERIFIED in R2IL; INTEGRATION VERIFIED through `act` |
| D3 | `act`, `apply`, `Intent` (`lance-graph-mailbox-lanes/src/act.rs`); `MailboxSoA::write_row` | `Model` | cells changed equal the model, stamped | stale cycle, foreign mailbox, stamp column, repeat apply. 4 disables | INTEGRATION VERIFIED on `MailboxSoA<64>` for `meta` and `temporal` |
| D4 | `cascade` | `Model::round` | 3 rounds, settles | bound of 2 reported unsettled; a column no method is ON. 2 disables | INTEGRATION VERIFIED |
| E1 | the chain | `Model` | `tests/e1.rs`, 4 tests, three ledgers | 7 disables | INTEGRATION VERIFIED from a source-local row; from a `NodeGuid` BLOCKED (B1) |
| F | `examples/cost.rs` | direct array, scalar interpreter, eager copy | measured, release | n/a | MEASURED; normalise-once NOT IMPLEMENTED |

## R7: review of #2064 (P0), the read-to-write boundary

### P0a, a stale intent (fixed)

Finding: `act` returns an `Intent` after its publication has ended; `apply`
checked the mailbox and, through `write_row`, the cycle. `set_meta` and the
public columns change data without moving the cycle
(`virtual-address-lab/tests/publication_boundary.rs`), so an intent could land
on data its guard and operands never read.

Not used: a source revision, a mutation log, or a copy of the read columns.
The mailbox has none of them and none was added.

Boundary chosen: a write is made only inside one exclusive borrow of the owner
that also covers the read it rests on.

- `cascade` already does this: `&mut MailboxSoA` is held from a round's read to
  its writes. It now writes through the private `write`.
- `apply(owner, store, view, ctx, intent)` treats the intent as what the
  caller expects. Under its own `&mut` it runs the intent's method again
  (`Intent` carries its `MethodRef`) and writes only when one of the writes the
  method asks for now is the same write: method, mailbox, column, value, rows.
  Otherwise `ApplyRefusal::Stale { column }`, or `NotReproduced(ActRefusal)`
  when the method no longer runs. The cycle stays the owner's check in
  `write_row`.

Consequence: a change the method does not read leaves the intent valid, and a
change that happens to reproduce the same write does too. `apply` costs one
`act` more than before.

Test (`tests/e1.rs`, `an_intent_does_not_outlive_what_it_read`), committed
before the fix and failing on it: a row joins the guard (`set_meta`, no
`tick`), a row leaves it, an operand value changes (public column). Each is
refused with `Stale` and the owner's columns and write stamps are equal before
and after. Silence twin: a `set_sigma` on a guard row, which the class does not
read, and the intent lands on every fired row.

Disable runs, anchors asserted, all exit 101: the `Stale` comparison switched
off; `value` dropped from the comparison; `rows` dropped from it.

### P0b, a recognised column as scratch (fixed)

Finding: `writable` mapped every `ColumnRefusal` to "scratch field", so a
method targeting `urn:mailbox:sigma` (a column of the owner the socket cannot
lend) ran and had its scalar filed under `unwritten`.

Fix: `ColumnRefusal::UnsupportedKind` refuses the method before anything is
read (`ActRefusal::UnsupportedColumn { field, kind }`). Only
`ColumnRefusal::Unknown`, an IRI that names no mailbox column, is scratch.

Test (`tests/writable.rs`), committed before the fix and failing on it: the
`sigma` target is refused with 0 lanes lent; a `urn:lab:scratch` target runs,
asks for no write and reports its scalar. One disable run, exit 101.

### P0c, the write-cycle stamp (review finding, fixed)

`write_row` stamps `last_write_cycle` on every accepted write. `apply` counts
the cells whose row carried another stamp (`Applied::stamped`) and `cascade`
adds `Column::LastWriteCycle` to the next round when that is non-zero.
`tests/stamp.rs`, 2 tests, one disable run.

### Cost after P0 (release, 40 rows, median of 9 batches, this machine)

| step | ns per call | allocations | bytes |
|---|---|---|---|
| `act` | 6407 | 90 | 11620 |
| `apply`, one intent (re-evaluates) | 6480 | 93 | 11700 |
| `cascade`, 3 rounds | 19218 | 287 | 37223 |

`apply` was 72 ns when it only wrote. `act` grew by one allocation and 480
bytes: the intent's copy of its method reference.

### Ledger, R7 on #2064

| Item | Contract (file) | Oracle | Positive | Negative / disable | Status |
|---|---|---|---|---|---|
| P0a | `apply`, `ApplyRefusal::Stale` (`act.rs`) | owner columns and stamps before and after | an unread change leaves the intent valid | guard joins, guard leaves, operand changes: refused, zero writes. 3 disables | INTEGRATION VERIFIED on `MailboxSoA<64>` |
| P0b | `writable`, `ActRefusal::UnsupportedColumn` | lanes lent, scalar by hand | an unknown IRI is scratch | `sigma` target refused before any read. 1 disable | INTEGRATION VERIFIED |
| P0c | `Applied::stamped`, `cascade` | methods by hand | a method ON the stamp fires in round 2 | rows already stamped trigger nothing. 1 disable | INTEGRATION VERIFIED |

### P0d, an intent of a cycle the owner has left (merge gate, fixed)

After `tick` with nothing else changed, `apply` reproduced the same rows and
value and answered `Ok`: with `refused: Some(Stale)` when a cell would change
and with a plain `Ok(unchanged)` when every cell already held the value. Now
`ApplyRefusal::Cycle { intent, owner }`, before anything is evaluated or
written. The cycle is the owner's write protocol; it is not used as a
statement about source freshness. `cascade` is unchanged. Test committed first
and failing (`a_reproduced_intent_of_an_earlier_cycle_is_refused`); one
disable run, exit 101.

Not established by R7: that two methods of one round do not read what the
other writes (a round reads one publication and writes afterwards, in
ascending `MethodId` order, by design); anything about writers outside this
module that hold their own `&mut`.
