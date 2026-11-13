# Virtual-address superpowers: implementation roadmap v1

**Status:** PROPOSED / DOCS ONLY. Nothing here is authorized. Every stage
names a decision that is the operator's before any code outside a lab.
**Evidence base:** `.claude/plans/d-virtual-address-algebra-1.md` sections 11
to 17, at `c460ae03a` (PR #2060, lab only). This roadmap adds no finding; it
orders the seams that plan reports as BLOCKED, ABSENT or NOT DONE.
**Date:** 2026-11-13.

## 1. The target, in one line

```text
X ; Y(X)  ->  bound ClassView / F### behaviour  ->  ON  ->  WHEN  ->  demanded DO
```

Source-owned memory, virtual properties, behaviour-aware reads: a value is
read only when an admitted method demands it, from the bytes the owner
already holds, and no virtual row or relation is built first.

Four sentences the roadmap must keep true at every stage:

- Ontology is invariant. Coordinates are versioned. Storage is owned. Views
  are derived.
- `Read(S, x, f) = S[f(x)]`: a pure address function, then one source read.
- Admission is not part of the address function.
- An unsupported boundary is reported BLOCKED. It is never replaced by a lab
  authority.

## 2. Where the lab stands

| capability | state | on |
|---|---|---|
| recipe algebra (8 orientations), composition, inverse cancellation | LAB VERIFIED | pure |
| address function resolves field AND position | LAB VERIFIED | `Grid`, mailbox |
| one scalar read from the owner's own column | LAB VERIFIED | `MailboxSoA<64>` |
| `Method { selection, orientation }` as a constant | LAB VERIFIED | `Grid` |
| ON and admission on the address, routed lens | LAB VERIFIED | `Grid` |
| WHEN, two-valued lane comparisons | LAB VERIFIED, lab-local translation | `Grid` |
| demanded operands limited to `reads` | LAB VERIFIED | `Grid` |
| `NodeGuid` -> source-local row | BLOCKED | mailbox |
| mailbox as lane / frame / selection source | BLOCKED | mailbox |
| data class <-> behaviour lens | BLOCKED | DT64 |
| mask-plane guard (`Sym`), Kleene guard | BLOCKED / NOT SUPPORTED | all |
| library seam `Guard -> Quack filter` | ABSENT | Quack |
| DO executed, write-back | NOT DONE | all |
| cost of any path but the scalar read | NOT MEASURED | all |

The pattern: everything pure is shown; everything that needs an AUTHORITY
(identity, frame, selection, lens, mask plane) is shown only on a fixture
that issues its own.

## 3. Stages

Each stage: what it delivers, the existing authority it extends, its gate
(a test that must fail when the stage's guard is disabled), and the decision
it needs first. Stages inside one track are ordered; tracks A to D are
independent of each other until stage E.

### Track A: the owner becomes a source

The mailbox owns the bytes and lends columns, but it is not a `LaneSource`,
holds no frame and issues no selection.

- **A1 Frame.** The mailbox declares a revision that moves on every write a
  reader could observe. Today `set_meta` moves neither `current_cycle` nor
  `last_write_cycle`. Extends: `FrameStore`. Gate: a binding sealed before a
  write is refused after it; a write that moves no stamp fails the test.
  Decision: which mailbox clock is the frame (cycle, per-row stamp, both).
- **A2 Lanes.** `impl LaneSource for` a mailbox view: `lane_of` by the
  predicate DT64 resolved, `borrow` lending the existing arrays. Needs lane
  kinds beyond `I32`: the owner lends `u32`, `u64`, `f32`, `u16`,
  `[u8; 12]`. Extends: `LaneSource`, `LaneRef`. Gate: the lent slice has the
  owner's address; a lane under another frame is refused. Decision: which
  crate may depend on `lance-graph-dt64` (`cognitive-shader-driver` does not
  today); whether `[u8; 12]` is lent as `LaneRef::Strided`.
- **A3 Selection.** `impl SelectionSource`: the owner issues position lists
  in its own row order, under its frame. Extends: `SelectionSource`,
  `OrderIndex` / `SealedOrder`. Gate: a caller-supplied list cannot be paired
  with a binding; a selection of an earlier frame is refused. Decision: where
  an order index lives (the owner has no home for one).
- **A4 Folds.** `execute_sel` admits `i32` only. Typed selection folds for
  the kinds A2 lends, or a stated refusal per kind. Extends: mask-RISC `sel`.
  Gate: independent oracle per kind; no widening without an overflow bound.
  Decision: which kinds, and whether this is a mask-RISC change at all.

Exit of track A: sections 11.3's five boundaries are closed for the mailbox
and the 64-identity selected-property test, which was never run, runs on the
real owner.

### Track B: identity

- **B1 `NodeGuid` -> coordinate.** `row_for_local_key` answers `None`. A
  `local_key` is basin-local only after its prefix is resolved. Deliver the
  authority that binds a canonical `NodeGuid` to a CURRENT source-local
  position, versioned with the frame of A1. Gate: two identities never
  resolve to one position; a position read through a stale mapping is
  refused; no second key storage is introduced. Decision: which existing
  authority owns this mapping (the open production question of #2060).
- **B2 Version-bound virtual property.** `VirtualProperty` carries no source
  version. With A1 and B1: `Identity x ClassView(Property) x AddressRecipe x
  SourceVersion` as one admitted value. Gate: the same descriptor under a
  moved frame is refused.

### Track C: behaviour provenance and WHEN

- **C1 Lens association.** No contract links a data class to a separate
  `0xF###` lens; the only authority is the route. Either (a) state that
  behaviour is always the routed classid's own and route to lenses, or (b)
  add a DT64 association `data class -> behaviour lens` with its own version.
  Gate: an unrelated lens with a compatible basis never supplies a method
  (the lab test, moved to the contract). Decision: (a) or (b).
- **C2 Route-bound admission.** `BindStore::admit` checks the store, not a
  route; R2IL has the route comparison privately. One public admission that
  takes the route. Gate: a reference to a registered, unrouted class is
  refused for the position.
- **C3 Guard contract.** DT64's `validate` does not look inside a guard, and
  a guard's `lane` is a `Planes` index in the only library lowering. Decide
  and enforce what a guard lane IS (a class field position), and validate
  guards at registration. Gate: a guard naming a field outside the basis is
  refused at `register`.
- **C4 `Guard -> Quack` seam.** A library function, replacing the
  example-local and lab-local translations. `lower_sel` is not this. Gate:
  parity with `set_lower` and with a scalar oracle on random guards.
- **C5 Mask-plane socket.** `Guard::Sym(i)` needs a producer that issues a
  plane with a frame, a row order and a population. Extends: the source
  sockets (a third one beside lanes and selections). Gate: a plane of another
  population or frame is refused. Decision: whether sources issue planes at
  all, or guards over planes stay out of scope.
- **C6 UNKNOWN.** Kleene guards need validity planes (C5) and a statement of
  what a method does on UNKNOWN (the contract says it does not fire). Gate:
  TRUE / FALSE / UNKNOWN against an independent Kleene evaluator.

### Track D: DO

- **D1 Operand boundary.** Make "a binding over `reads` only" the stated way
  operands are obtained, so a method cannot read past its declaration. The
  mechanism exists (`LaneRequest::fields`, `NotBound`). Gate: the lab test,
  on the real owner after track A.
- **D2 Execution.** R2IL's `prepare` refuses guarded and three-valued
  bindings. Either lift that with C4 / C6, or name the executor that runs a
  guarded method's effects. No new executor is implied by this roadmap.
  Decision: which existing executor.
- **D3 Write-back.** An effect's target is written under the owner's rules
  (one writer per mailbox, write on behalf of the owner), moves the frame of
  A1, and leaves a bounded receipt. Gate: no write without the owner's stamp;
  a reader's binding sealed before the write is refused after it.
- **D4 ON from a real change.** ON is a field mask; today the changed cell is
  given by the test. Connect it to the write of D3 so a DO can trigger the
  next ON, with a stated termination rule. Decision: cascade depth and order.

### Stage E: join

Tracks A to D on one owner: `NodeGuid` (B1) -> coordinate under a frame (A1)
-> routed behaviour (C1, C2) -> ON -> WHEN over the owner's lanes or planes
(C3 to C6) -> operands over `reads` (D1) -> effect and write-back (D2, D3).
Gate: the full chain against a scalar interpreter of the same class, with
the three ledgers (guard, operands, effects) and a count of values read
before the first admitted method (must be 0).

### Stage F: cost

Nothing but the scalar read is measured (7.5 ns with a 3-recipe chain
normalised per read, against 0.44 ns direct). Before any performance claim:

- normalise once per method, not per read;
- one binding check per population, not per cell (the lab re-runs the whole
  check on every cell);
- guard and operand loops through `ndarray::simd` via mask-RISC, never
  through raw intrinsics;
- report per-operation costs in nanoseconds and batch costs in microseconds;
  a millisecond result is a failed gate, not a number to report.

The "1 to 4 cycles" figure stays unclaimed until an ISA kernel runs on this
path and is measured.

## 4. Order and dependencies

```text
A1 -> A2 -> A3 -> A4          C1 -> C2
 |                            C3 -> C4 -> C5 -> C6
 +--> B1 -> B2                D1 (needs A2) -> D2 (needs C4) -> D3 (needs A1) -> D4
                 \____________ E ____________/ -> F
```

Smallest useful first cut: **A1 + A2 + C1(a) + C2 + D1**. It gives a routed,
frame-checked, reads-bounded property read on the real mailbox with no guard,
and needs one decision (the mailbox clock) and one dependency edge.

## 5. Decisions needed before any stage starts

| # | decision | stage |
|---|---|---|
| 1 | which mailbox clock is the source frame | A1 |
| 2 | which crate carries the `LaneSource` impl and its DT64 dependency | A2 |
| 3 | where a source-issued order index lives | A3 |
| 4 | which authority binds `NodeGuid` to a current coordinate | B1 |
| 5 | behaviour = routed classid, or an explicit data-class -> lens contract | C1 |
| 6 | a guard lane is a class field position (and is validated) | C3 |
| 7 | do sources issue mask planes | C5 |
| 8 | which existing executor runs a guarded method | D2 |
| 9 | whether a position-reversing recipe is ever admitted on an owner without a selection | policy, #2060 open item |

## 6. Not in this roadmap

- A layout change. ABI512, `NodeRow`, `VALUE_TENANTS` and `MailboxSoA<N>`
  stay as implemented.
- A third recipe axis. The algebra is 2-D (fields x ordinals, 8
  orientations); the 48 cube symmetries are not needed for any stage above.
- A new storage owner, executor, `MapTable` or permission issuer.
- Authorization. `Scope` equality is a consistency check; who grants a scope
  is a separate open item and no stage here closes it.

## 7. Unread, so not relied on

`dr.rs` (searched only), the DT64 behaviour plans (`d-dt64-6-*`),
`.mermaid/05-methods.md`, `.claude/v3/soa_layout/*.md`, the mailbox-kanban
executor crates. A stage that touches one of them starts by reading it; a
finding there may reorder this roadmap.
