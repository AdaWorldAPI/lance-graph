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

## R2 to R6

Not started. Order: A3 + B1 + B2 + A4, then C1(b) + C3 to C6, then D2 to
D4, then E1, then F.
