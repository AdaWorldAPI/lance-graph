# D-VAS implementation ledger v1

**Status:** IN PROGRESS. Implements the roadmap
`.claude/plans/virtual-address-superpowers-roadmap-v1.md` in the order R0 to
R6 (directive of 2026-11-13). Research baseline: PR #2060
(`d-virtual-address-algebra-1.md`, sections 11 to 18).
**Deliverables:** `D-VAS-A1`, `D-VAS-A2`, `D-VAS-C2`, `D-VAS-D1`,
`D-VAS-E0` in R1; the remaining `D-VAS-*` ids in R2 to R6.

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

### A4

Not started.

## R3 to R6

Not started. Order: C1(b) + C3 to C6, then D2 to D4, then E1, then F.
