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

## R8: the constant `Method` on the real mailbox path (review P1 to P4)

### P1, convergence gate (done, as a test; no production code)

Before this, two things were proven apart: the address law of
`virtual_address_lab::Method { selection, orientation }` on the lab `Grid`, and
ON / WHEN / DO on a real `MailboxSoA<64>` from DT64 bindings. `act` does not
consume a `Method`, so E1 was no proof of the constant-method address path.

`crates/virtual-address-lab/tests/mailbox_method.rs`, 4 tests on a real
`MailboxSoA<64>`:

```text
publication -> socket-issued selection (issue_order) -> const Method (6 bytes)
  -> Method::field (no lane) -> ON: methods_on(field)
  -> LaneBinding::selected -> Method::cell -> source row, one value
  -> WHEN: when(..), Fired::holds(row) -> DO: act(..) only where it holds
```

- Addressing oracle: the tile `[field][ordinal]` built from the owner's arrays
  and a sort written in the test, then moved (lanes reversed, positions
  reversed, transposed) per recipe name. It calls no `Recipe` code.
- Behaviour oracle: the two methods by hand.
- Sweep: 8 orientations x 4 fields x 40 ordinals = 1280 view cells; 320 have
  no address (the class's scratch field), 640 have a candidate method.
- Ledger (`behaviour_is_resolved_before_anything_is_read`): the field and the
  ON candidates cost 0 lent lanes; a cell with no candidate lends 1 (its
  value); a candidate whose guard is false at the row lends 1 + the guard's 2
  fields and runs no effect; a fired one lends 1 + 2 + 2 + 1.
- The tests passed on their first run: they exercise existing code. Three
  disable runs, anchors asserted, exit 101: `Recipe::resolve` ignoring the
  position flip; `Recipe::lane_at` ignoring the lane flip; `Fired::holds`
  always true.

The composition lives in the test. No function in a production crate takes a
`Method`; `virtual-address-lab` is a lab crate.

### P2, address carriers

| Carrier | Enters the chain | Evidence | Status |
|---|---|---|---|
| ordinal of a socket-issued selection (source-local SoA) | `Method::cell` -> `LaneBinding::selected` | `a_constant_method_addresses_and_behaves_on_the_real_mailbox` | INTEGRATION VERIFIED |
| id of DT64's `MapTable` at a sealed version | `Coordinates::row_of` -> `LaneBinding::lane` | `a_versioned_coordinate_enters_the_same_chain`: 40 ids x 3 fields, equal to the oracle and to the same row reached by ordinal | INTEGRATION VERIFIED; the test fills the map, no authority does |
| V3 `FacetCascade` / strided 12-byte payload | none | `tests/i4x24.rs` reads 24 x i4 in place from a `[NodeRow]` slab and a mailbox style lane through mask-risc `StridedRef`, unbound | LAB VERIFIED as a read; NOT CONNECTED to ON / WHEN / DO |
| `NodeGuid` | none | `row_for_local_key` answers `None` | optional, BLOCKED (B1), not required by any test here |

Why the strided carrier does not enter: `MailboxLanes` refuses the `[u8; 12]`
style columns (`ColumnRefusal::UnsupportedKind`), no `LaneSource` exists over a
`[NodeRow]` slab, and a guard residual (`Res::U32` / `Res::I32`) lowers to a
predicate over a plain lane, which mask-risc refuses on a strided one. Closing
it needs a strided lane kind in a socket and a guard form that names a facet
position; neither was invented.

No code added in R1 to R8 reads a `NodeGuid`, so no V1 `family ++ identity`
reading is applied to V3 bytes by it.

### P3, the semantic boundaries, as the code has them

| Question | Answer | Where |
|---|---|---|
| source write vs mapping change | a write goes through the owner (`write_row`, a setter, a public column) and ends every binding of the publication; a re-orientation (`Method::then`) or a map advance changes which row a view cell or id denotes, moves no byte and leaves the lane binding valid | `reorienting_is_not_a_write_and_a_write_is_not_a_reorientation`; `tests/coords.rs` |
| which changed field makes a method eligible | a column the owner really changed in the previous round (target column, or the write stamp), matched to the class field with that IRI, then `methods_on(field)` | `cascade` |
| which population WHEN evaluates | the populated prefix of the publication, all rows, as a membership mask; not a selection and not a bag | `when` |
| which rows DO affects | the rows the guard keeps; the effect's scalar is one value, written at each of them | `act`, module doc |
| does a routed `0xF###` lens have data-class provenance | only its route: the position routes to the lens, which carries its own basis and methods. No `data class -> lens` relation is stored or checked | R3, C1 |
| who issues `Sym` planes | nobody; `when` refuses (`PlaneUnissued`) | R3, C5 BLOCKED |
| who issues nullable validity | nobody for the mailbox; `when` refuses a Kleene guard (`ValidityUnstated`) | R3, C6 |

### P4, optimisation: NOT STARTED

Correctness first, as asked. Baseline after R7: `act` 6.4 us and 90
allocations per call at 40 rows, every call cold. Not done: profiling of the
allocation sources, a prepared plan and guard program reused across calls
with explicit invalidation by publication and route, a warm measurement, and
the cost of the `Method` path itself. `apply` now costs one `act`; a reused
preparation would cut both.

## R9: coordinate x content independence, and a read-only projection

Terms: a *publication witness* is the borrow-scoped frame of one
`MailboxLanes` socket; it is not a historical content version. A *mapping
version* is a sealed `MapTable` frame. The mailbox's cycle is its own write
protocol and is used by no check below.

```text
Read(S, g, Vm, Vc) = Interpret_Vc( S[ Resolve_Vm(g) ] )
```

### Gate 1, the 2 x 2 matrix (test only; existing code)

`crates/lance-graph-mailbox-lanes/tests/independence.rs`, 5 tests on a real
`MailboxSoA<64>`. Oracle per quadrant: `row = assignment[Vm](g)`, then
`value = contents[S][row]`, both written in the test.

| Contents | Mapping | Observed |
|---|---|---|
| same | same | same row and value, twice |
| same | advanced while the publication is live | the new assignment's row and that row's value; the owner's bytes equal before and after; the lane binding stays valid |
| changed by `set_meta`, no `tick` | same (still current) | the same row, the new value, under a new publication |
| changed | advanced again | both compose |

Negatives, each a refusal: a coordinate binding of an earlier map version
(`StaleMapping`) and of another table (`ForeignMap`); a lane binding after its
publication ended; a binding made for one routed class after the position was
routed to another; a second socket over the same owner; another owner; an
unrouted position; another reader scope. The negative law is its own test: with
the map version unchanged, the coordinate resolves to the same row, the value
there differs from the one read earlier, and the earlier read's only witness
(its lane binding) is refused.

The tests passed on their first run except for a fixture error of mine: two
classids of one concept share one `ClassView` basis (`view_class` is the
concept half), so the second interpretation needed another concept. Two
disable runs, anchors asserted, exit 101: the `StaleMapping` comparison off;
the `temporal` lane lending `meta`. A third patch did not apply (its anchor
occurs twice) and is no evidence.

Already proven before R9, not repeated: unchanged contents under two map
versions (`tests/coords.rs`); selection addresses ending with their socket
(`tests/selection.rs`).

### Gate 2, a read-only projection (test composition; one refusal added)

`ClassView IN published rows WHEN guard DO fold`, composed from what exists:
`act` with a method whose effect targets a class field that names no mailbox
column returns its scalar in `Acted::unwritten` (and
`receipt.output.values`), with no `Intent`. `tests/writable.rs`:

- the population and the scalar equal a fold written by hand; the receipt
  names the method; 2 lanes lent; the owner is equal before and after; the
  same fold targeting `meta` is a write request (silence twin);
- refused: a position that does not route to the method's class (0 lanes
  lent), no execution permit, a reader scope other than the socket's.

No new output API was added. A `project` function was drafted and removed: it
only re-shaped `act`'s result.

The population is DISTINCT: one bit per row of the populated prefix. It is not
a bag.

### Multiplicity (one production change)

Finding: a `MethodBinding` declared with `Population::Bag` ran through `act` on
the one-bit-per-row gate and returned the set reading. Nothing read
`population`.

Fix: `act` refuses it before anything is lent
(`ActRefusal::BagPopulation`). Test committed first and failing
(`a_bag_method_does_not_run_on_a_membership_gate`): the socket issues a bag in
which every guard row occurs twice (`issue_concat`), the bag fold by hand is
twice the set fold, the distinct method returns the set fold, the bag method
is refused. One disable run, exit 101.

Not done: a duplicate-sensitive fold over a socket-issued bag. The shipped
selection fold refuses the mailbox's `u32` / `u64` lanes (A4), R2IL's binding
seam takes no selection operand, and R2IL itself does not read
`population` either (only `act` refuses now).

### Ledger, R9

| Item | Contract | Oracle | Positive | Negative / disable | Status |
|---|---|---|---|---|---|
| independence | `Coordinates::row_of`, `LaneBinding::lane`, `MailboxLanes` | assignments and values in the test | 4 quadrants x 40 ids | 7 refusals; 2 disables | INTEGRATION VERIFIED, no new code |
| mapping does not validate a value | same | same | same row, other value | old binding refused | INTEGRATION VERIFIED |
| read-only projection | `act`, `Acted::unwritten`, `Receipt` | fold by hand | rows and scalar equal, no intent, owner unchanged | route, permit, scope | INTEGRATION VERIFIED for a DISTINCT population, by test composition |
| bag declared | `ActRefusal::BagPopulation` | bag fold by hand | distinct method runs | bag method refused. 1 disable | INTEGRATION VERIFIED (refusal); bag execution NOT SUPPORTED |

Not established: historical replay (no source contents are retained); any
reuse of a computed result past its publication; authorization (the scope
check compares two caller-asserted values); a general bag API; cost.

## R10: selection is not execution population

Finding: an address can be resolved through a socket-issued selection
(`Method::cell` -> `LaneBinding::selected`), and `act` runs an admitted method;
but `act` takes no selection and runs over the populated prefix. The #2065 test
addresses a cell through a selection and then calls population-wide `when` /
`act`. A correct scalar over the wrong population is not Bag correctness.

`tests/population.rs`: four rows, all in the guard; the selection `[1, 1, 3]`
issued by the socket (`issue_where`, `issue_where`, `issue_concat`). Three
counts by hand, and what each path returns (the scalar output, not a row
count):

| reading | by hand | path | returns |
|---|---|---|---|
| whole source | 4 | `act` | 4 |
| distinct selected | 2 | `act_over` (new) | 2 |
| selection as a bag | 3 | none | a bag-declared method is refused, with and without a selection |

`act_over(lanes, store, view, ctx, method, selection)`: the population is the
guard's rows that occur in the selection, each once. It narrows the gate plane
R2IL already runs under; no executor, operand form or fold is added, and
`act` shares its body. Refused: a method that writes a mailbox column
(`NotReadOnly`: `apply` re-evaluates over the whole prefix and could not
reproduce a selected write), an unguarded method (`SelectionNeedsGuard`: an
ungated plan has no plane to narrow), a method with no declared read
(`SelectionNeedsRead`), a selection the socket did not issue. `Acted::over`
records which population a result was computed over.

Selection provenance, exactly:

- established: the positions are taken inside `act_over` through
  `LaneBinding::selected` on the operand binding, so DT64's joint witness
  (same source, same publication frame, same row order and scope) is checked
  in the same borrow the fold runs in; a caller cannot hand in positions or a
  plane;
- missing: R2IL's `Gate` checks the guard's witness, not the plane's bits, and
  its `Receipt` names no selection. The narrowed plane is vouched for by this
  module, as the guard's own plane already was. A selection witness inside
  R2IL's gate does not exist and was not invented.

Test committed before `act_over` existed. Two disable runs, exit 101.

| Item | Contract | Oracle | Positive | Negative / disable | Status |
|---|---|---|---|---|---|
| `act` population | `act` | count by hand | whole source, 4 | not 2, not 3 | INTEGRATION VERIFIED (characterised) |
| distinct selected | `act_over`, `Acted::over` | count by hand | 2; the guard still applies (a selected row outside it is not counted) | writes, no guard, unknown selection. 2 disables | INTEGRATION VERIFIED, read-only, DISTINCT |
| bag multiplicity | none | 3 by hand | n/a | refused | NOT SUPPORTED |

Levels kept apart: `MethodBinding.on` is a field activation mask; a selected
ordinal is an address within an issued selection; the guard mask is Boolean
membership; a bag has declared multiplicity; DO computes over the population
that was explicitly bound. "ON as contextual Bag binding" is a target, not a
contract in the code.

## R11: multiplicity without materialization (characterised; no bag path admitted)

Baseline `main` at `9ef465089` (#2064, #2065, #2066 merged). No production
code changed in R11: tests, one measurement example, this section.

### Scope check of an unguarded method that reads nothing (R10 gap, closed)

Such a method is expressible and admitted: `scratch := IntAdd(2, 3)`, `when:
None`, empty `reads` (`tests/population.rs`, methods `CONST`, `CONST_TO_META`).
It binds no lane, so only `act`'s own `check_scope` compares scopes.
`an_unguarded_method_that_reads_nothing_still_checks_the_reader_scope`: a
foreign scope is refused with `ScopeMismatch` for both the scratch and the
`meta` target (no scalar, no intent); the socket's own scope returns 5 and one
intent for the `meta` target. Disable control (the `check_scope` line in
`act_in` removed, anchor asserted once): the test fails, exit 101. Consistency
of two caller-asserted values, not authorization.

### The selection-fold substrate, as read

1. Operations that keep repeated positions: `mask_risc::execute_sel` with
   `SelFold::CountGtI32` and `SelFold::SumI32`, each one call of
   `ndarray::simd::{count_gt_i32_at, sum_i32_at}`. Nothing else takes a
   position list. ndarray's facade has exactly these two `*_at` folds.
2. Lane kinds: `i32` only (`validate_sel_frame` checks `LaneKind::I32`). The
   mailbox lends `u32` and `u64`; the refusal is pinned in
   `tests/selection.rs` (R2, A4).
3. From a `MethodBinding`: not reachable. R2IL's binding seam runs effects as
   `Program`s through `FoldDialect` over `Planes`; it takes no position list
   and does not read `population`. `fold_bound_selection` is called with a
   `BoundFold` chosen by the caller, not with an admitted method.
4. WHEN with a selection: `quack::lower_sel` admits one predicate shape,
   `Plane(selection) AND column > t` on one `i32` lane, as part of the fold.
   A guard already evaluated to a kept-row mask cannot be combined with a
   position list: `execute_sel` reads no mask plane (stated in its doc).
5. Witness: `LaneBinding::selected` checks frame, route, basis, row order
   and scope for the lane and the positions together, in one borrow. It does
   not establish that positions are rows of the lane (the fold checks),
   authorization or history.
6. Receipt: `execute_sel` returns a bare `Value`. R2IL's `Receipt` comes only
   from `run` / `run_gated`.

### Falsifier (oracle by hand; calls nothing of the implementation)

`a_bag_count_weighs_each_occurrence_by_the_guard_and_has_no_path`, real
`MailboxSoA`, selections issued by the socket:

| fixture | selection | whole (`act`) | distinct (`act_over`) | bag (oracle only) |
|---|---|---|---|---|
| all rows satisfy WHEN | `[1, 1, 3]` | 4 | 2 | 3 |
| row 3 fails WHEN | `[1, 1, 3]` | 3 | 1 | 2 |
| all satisfy | `[1, 3, 1]` (other order) | 4 | 2 | 3 |
| row 3 fails | `[1, 3, 1]` | 3 | 1 | 2 |
| all satisfy | `[]` | 4 | 0 | 0 |
| all satisfy | `[1, 1]` | 4 | 1 | 2 |

`act` and `act_over` match the first two columns in every case. The bag
column has no path; `ActRefusal::BagPopulation` stays, with and without a
selection.

### Stop rule applied: the first missing capability

A guarded, multiplicity-preserving COUNT over the mailbox's lanes needs, per
listed occurrence, the answer "does the guard keep this row". The guard's
answer exists as one bit per row (typed evaluation over `u32` lanes, already
admitted). What is missing is a fold that reads that bit at each listed
position:

- first missing capability: `count_set_at(bits: &[u64], n_rows, positions:
  &[u32]) -> Option<u64>`, the number of listed positions whose bit is set,
  `None` for a position outside the frame;
- owner: `ndarray::simd` (`simd_masking_ops.rs`, beside `count_gt_i32_at`);
  mask-risc may only delegate (law L3), so no fold is added there first;
- then, in order: `SelFold::CountInPlane { mask }` in mask-risc with its
  reference oracle; `Plane(selection) AND Plane(kept)` with `Agg::Count` in
  `quack::lower_sel`; a way for R2IL to run a `Population::Bag` method's
  COUNT over a bound position list and return its `Receipt`. The last one is
  a semantic decision (below), not a kernel.
- falsifier for the kernel: the six rows above with the kept mask of each
  fixture (expected 3, 2, 3, 2, 0, 2), an out-of-range position (`None`), and
  the same positions deduplicated giving the distinct column, so a kernel
  that builds a mask from the list fails.

Nothing of this was implemented. No scalar loop in production, no `i32`
shadow column, no reinterpretation of the Boolean gate as a bag.

### Provenance of a result, as the contracts stand

| fact | held by | for how long |
|---|---|---|
| source publication | `MailboxLanes` (borrow), `Receipt::frame` | the borrow; the receipt keeps the frame revisions as values |
| selection identity | `Acted::over` (socket-local `u16`) | the socket; meaningless after it |
| mapping and route version | `ActContext::route`, checked by `prepare` / `run` | the call |
| ClassView / interpretation | `resolve_at`, `R2ilLens` | the call |
| method and guard | `Receipt::method` (`MethodRef` with guard witness) | in the receipt |
| population semantics | `MethodBinding::population`; not in the receipt | registration |
| scope | `check_scope`, lane and selection checks | the call |

The receipt does not prove selection membership: R2IL checks the guard's
witness, not the gate plane's bits. No receipt field was added.

### First P4 baseline (release, 40 rows, median of 9 batches, this machine)

`cargo run --release -p lance-graph-mailbox-lanes --example cost`. A size
class, not a comparison across machines.

| step | ns per call | allocations | bytes allocated |
|---|---|---|---|
| socket: `MailboxLanes::new` | 100.8 | 3 | 80 |
| DT64 `admit_routed` | 766.2 | 17 | 2888 |
| WHEN: lower + bind + run (2 fields) | 2716.0 | 31 | 3767 |
| `act`, writing method (admission to gated DO) | 5990.7 | 90 | 11624 |
| socket: open + issue 2 selections + concat | 769.6 | 14 | 1048 |
| `act`, read-only method, whole source | 3641.6 | 45 | 4393 |
| `act_over`, read-only, selection's distinct rows | 3978.8 | 47 | 4409 |
| `execute_sel` `CountGtI32`, SYNTHETIC `i32` lane, 54 positions | 47.3 | 0 | 0 |

- Source bytes copied: none on any path (lanes are borrowed). `act_over`
  allocates one membership mask and one narrowed mask (2 allocations, 16
  bytes at 40 rows) beyond `act`.
- Every call prepares anew: admission, guard lowering and R2IL `prepare` are
  inside the `act` numbers. There is no warm, prepared execution to measure
  separately; none was added.
- The `execute_sel` row is the shipped kernel on a synthetic `i32` lane. It
  is not mailbox execution and not an admitted method.
- Bag execution: not measured; it does not exist.

### Ledger, R11

| Item | Contract | Oracle | Positive | Negative / disable | Status |
|---|---|---|---|---|---|
| scope, unguarded read-free method | `act_in` `check_scope` | hand | own scope: 5, intent for `meta` | foreign scope refused; disable exit 101 | INTEGRATION VERIFIED |
| whole / distinct counts, six selections | `act`, `act_over` | `guarded_oracles` by hand | all six rows | bag method refused in all six | INTEGRATION VERIFIED |
| selection folds keep repeats | `execute_sel`, `fold_bound_selection` | `reference_execute_sel` (shipped) | `i32` lanes | mailbox lanes refused (`tests/selection.rs`) | SOURCE VERIFIED; LAB only for `i32` |
| guarded bag COUNT on the mailbox | none | hand: 3, 2, 3, 2, 0, 2 | n/a | `BagPopulation` | NOT SUPPORTED; first missing: `count_set_at` in `ndarray::simd` |

### Decisions needed

1. Whether `count_set_at` is added to `ndarray::simd` (another repository).
2. What a `Population::Bag` method means in R2IL: which effect words may run
   over a position list, and whether a mask-producing step (`IntEqual`) is
   then evaluated per row and counted per occurrence. Until stated, `act`
   refuses bag methods.
3. Whether a result must carry its selection beyond the socket. Today it
   does not, and no reuse across publications is claimed.

## R12: a guarded bag COUNT, admitted (two gates)

Decisions taken for R12: `count_set_at` goes into `ndarray::simd`; a bag
method means exactly a COUNT of the selected occurrences that satisfy WHEN
and the effect's own mask; a result names its population only inside the
publication.

### Gate A, the kernel (ndarray #424, merged as `bd9a7ba`)

`ndarray::simd::count_set_at(bits, n_rows, positions) -> Option<u64>`: the
number of listed positions whose bit is set in an LSB-first row mask. A
repeated position counts per occurrence. `None` for a mask whose length is
not `n_rows.div_ceil(64)` and for a position not below `n_rows`. Tail bits
are never read. A plain loop on every backend; nothing was measured there.
Tests against a stub first (3 failed), then the body; three disable runs
(row bound, mask length, bit order), exit 101 each; parity group `0x87x`,
`native` on AVX-512 and AVX2. The cross parity arms were not run.

### Gate B, the admitted path (this repository)

1. **mask-risc.** `SelFold::CountSet { plane }`: one delegation to
   `count_set_at`, with `reference_execute_sel` as its oracle. Refusals, in
   order: missing plane (`PlaneOutOfRange`), a plane not `words_for(n_rows)`
   long (`LenMismatch`), a position that is not a row (`SelOutOfRange`, first
   offender). A dirty tail is not read and not refused. `tests/sel_set.rs`
   (5 tests) was committed against a stub (5 failed).
2. **R2IL.** `FoldDialect::count_at(positions, stack)`: the mask on the stack
   is written to the dialect's `KEEP` sink (`Terminal::Keep`, the existing
   path) and folded over the positions by `execute_sel`. `binding::run_bag`:
   `run_gated`'s checks (issuer, guards, route, permit, gate witness), then
   every `PopCount` step runs `count_at` after `gate_fold` has ANDed the
   gate into the step's own mask. A plan with any other fold is refused
   whole (`ExecRefusal::BagFold { effect, opcode }`). The result comes back
   in the normal `Receipt`.
3. **`act_over`.** A `Population::Bag` method is admitted when a selection is
   bound, it writes no mailbox column and it has a two-valued guard. The
   positions are the ones `LaneBinding::selected` lends with the operand
   lanes. `act` (no selection) still refuses with `BagPopulation`.

The effect's mask and the guard stay two conditions: `gate_fold` ANDs them
per step; neither replaces the other.

### Falsifier (real `MailboxSoA`, socket-issued selections, oracle by hand)

| fixture | selection | whole (`act`) | distinct (`act_over`) | bag (`act_over`, bag method) |
|---|---|---|---|---|
| all rows satisfy WHEN | `[1, 1, 3]` | 4 | 2 | 3 |
| row 3 fails WHEN | `[1, 1, 3]` | 3 | 1 | 2 |
| all satisfy | `[1, 3, 1]` | 4 | 2 | 3 |
| row 3 fails | `[1, 3, 1]` | 3 | 1 | 2 |
| all satisfy | `[]` | 4 | 0 | 0 |
| all satisfy | `[1, 1]` | 4 | 1 | 2 |

Effect mask independent of WHEN (`WHEN meta == 2`, effect `active == 5`,
`active = [0, 5, 0, 6]`, selection `[1, 1, 3]`):

| owner | bag | distinct | excluded by |
|---|---|---|---|
| all `meta == 2` | 2 | 1 | row 3: effect mask |
| row 1 `meta = 7` | 0 | 0 | row 1: WHEN (its effect mask holds) |
| row 3 `meta = 7` | 2 | 1 | row 3: effect mask and WHEN |

Refused: a bag method with `ANY` (`BagFold`), a bag method writing `meta`
(`NotReadOnly`), an unguarded bag method (`SelectionNeedsGuard`), a
selection the socket did not issue (`Lane`), a foreign scope
(`ScopeMismatch`), an empty permit (`NotPermitted`), and `act` without a
selection (`BagPopulation`). A bag run returns no intent and leaves the four
owner columns as they were; it lends three lanes (guard field, operand
field, the lane `selected` borrows with the positions).

Disable runs, anchors asserted once each, exit 101 each:

| disabled | tests failing |
|---|---|
| `count_at` (the mask counted, not the occurrences) | 4 |
| the gate not ANDed into a bag count | 1 (row 1 fails WHEN, effect holds) |
| a bag method admitted without a selection | 2 |
| other folds admitted over a bag | 1 |
| positions not taken from the witnessed selection | 4 |

### Provenance, as it stands

- Established inside `act_over`'s borrow: source, publication frame, route,
  basis, row order and scope for the operand lanes and the positions
  together (`LaneBinding::selected`); the method and its guard witness
  (`run_bag`); the permit.
- Vouched for by the caller, not checked by R2IL: that the gate plane's bits
  are the guard's rows, and that `positions` is a selection of the source
  the planes come from.
- Not in the `Receipt`: the selection and the population. `Acted::over` is a
  socket-local number and is no witness after the socket. No reuse across
  publications and no historical replay is claimed.

### Cost (release, 40 rows, median of 9 batches, this machine)

| step | ns per call | allocations | bytes allocated |
|---|---|---|---|
| `act`, read-only, whole source | 3294.9 | 45 | 4393 |
| `act_over`, read-only, distinct rows | 3882.9 | 47 | 4409 |
| `act_over`, bag method, 54 listed occurrences | 3687.8 | 48 | 4417 |
| `execute_sel` `CountGtI32`, SYNTHETIC `i32` lane, 54 positions | 47.4 | 0 | 0 |

A bag run allocates one more buffer than a distinct run (the `KEEP` sink,
8 bytes at 40 rows). No source bytes are copied. Every call prepares anew;
there is no warm path. The times moved by up to a factor of 1.5 between two
runs of this example on the same machine (`act`, writing method: 5990.7 ns
in R11, 8913.3 ns now), so they are a size class only. The last row is not
mailbox execution.

### Ledger, R12

| Item | Contract | Oracle | Positive | Negative / disable | Status |
|---|---|---|---|---|---|
| kernel | `ndarray::simd::count_set_at` | `&[bool]` walk | 4-row table, word boundaries | 3 disables | SOURCE VERIFIED (ndarray), parity on 2 arms |
| position fold over a mask | `SelFold::CountSet` | `reference_execute_sel`, `&[bool]` walk, mask path for a set | 5 tests | refusals, stub-first | LAB VERIFIED |
| bag COUNT of an admitted method | `run_bag`, `count_at`, `act_over` | hand tables above | 4/2/3, 3/1/2, independent effect mask | 7 refusals, 5 disables | INTEGRATION VERIFIED, read-only, guarded, COUNT only |
| other bag folds, bag writes, bag without selection | typed refusals | n/a | n/a | tested | NOT SUPPORTED |
| selection in the receipt, reuse past the publication | none | n/a | n/a | n/a | NOT SUPPORTED |

## R13: where a read-only `act` spends its time (measurement only)

No production change. `examples/cost.rs` now times each stage of one
read-only `act` alone, with the inputs `act` hands it (release, 40 rows,
median of 9 batches, this machine; a size class).

| stage | ns | allocations | reusable without the publication? |
|---|---|---|---|
| `admit_routed` | 188.0 | 5 | yes, while the route version holds |
| `resolve_at` (class and fields) | 184.9 | 0 | yes, while route and schema revision hold |
| WHEN: lower + bind + run, 1 field | 1043.4 | 14 | the lowering only; bind and run are per publication |
| operands: `bind_lanes` + 1 lane | 548.8 | 3 | no: bound to the socket's frame |
| R2IL `prepare_gated` | 615.9 | 8 | no: `Prepared` carries `Guards { frame, .. }` of one socket |
| R2IL `run_gated` (new dialect) | 533.2 | 7 | no |
| whole `act` | 3698.6 | 45 | |

The stages sum to about 3110 ns; the rest of `act` is its own vectors (lane
table, gate copy, fired rows, intents).

Findings:

1. No stage dominates. The largest, WHEN, is itself three parts; its
   lane binding costs about what the operand binding costs (549 ns), so the
   guard's lowering is at most about 400 ns of it. That split was not
   measured separately.
2. Every `MailboxLanes::new` has a frame store of its own, and R2IL's
   `Prepared` plan checks `Guards::frame` on every run. So a prepared plan is
   valid for one socket only. Reuse across publications would need a change
   of what an R2IL plan is bound to; that is a contract decision, not an
   optimisation.
3. What can be kept across publications without touching a contract:
   admission, class resolution and the lowered guard program, each
   re-validated by the route check `run` already makes. Upper bound of the
   saving: about 190 + 185 + 400 = 775 ns of 3700 (about 20 %).
4. Inside one socket (several selections or methods over one publication)
   `prepare_gated` can be kept too: about 1400 ns of 3700 (about 38 %).
5. 45 allocations per call. Several are fixed-size vectors `act` builds per
   call; removing them needs no new API. Not measured which ones matter.

Not done: no prepared-plan cache, no warm entry point. Status: MEASURED.
