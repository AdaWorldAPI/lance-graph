# cycle-loop-closure-driver-v1 — the loop-closure driver that makes the persist_sink cycle/WAL seam load-bearing

> **Status:** PLANNED / CONJECTURE — design only. The **CONTROL loop** this
> driver closes is the deliverable; the **durability leg stays the
> contract-probe fake** until the concrete `LanceShardSink` lands (the
> `compile+test green ≠ storage proven` Ladybug rule). Each claim below is
> probe-gated; nothing here is shipped.
> **Date:** 2026-08-02.
> **Scope:** documentation-only architectural ruling. Records the *missing
> seam* — the driver that turns the already-merged `persist_sink` cycle/WAL
> bootstrap into a running loop at 64k concurrency — and the deliverables +
> falsifiers that gate its construction. Changes **no** Rust code, tests,
> public APIs, `persist_sink.rs`, or `temporal.rs`.
> **Owns (narrowly):** "why the loop is open today", the closed-loop
> seal→step→think→cast shape, the writer-fires-inline correctness point, the
> D-MBX-A6-P4a…f deliverables, the home + dep-direction decision, and the 64k
> mechanics as they bear on the control loop.
> **Does NOT own (cross-refs, never re-specifies):**
> - The cycle/WAL seam itself (the OUT/durability half + two-dimensional
>   temporal model) → `persistence-cycle-wal-bootstrap-v1.md`.
> - Horizontal temporal-stream detail (the version-range read, ±5 window) →
>   `temporal-markov-and-style-classes-v1.md`.
> - Per-row `write_row` cycle-gate + the 16k-per-prefix scale framing →
>   `mailbox-cycle-aware-write-contract-v1.md`.
> - Deliverable tracking → `.claude/board/STATUS_BOARD.md`
>   (D-MBX-A6-P1…P3e shipped, D-MBX-9-IN scheduler contract, D-V3-W2a board
>   tenant gated, D-V3-W2b supervisor kanban_actor shipped, D2 symbiont
>   kanban_loop slice shipped).
> - The reshape rulings → `.claude/board/EPIPHANIES.md`
>   `E-THE-DURABLE-UNIT-IS-THE-CYCLE-NOT-THE-CAST-ONE-WAL-WRITE-PER-SWEEP-1`,
>   `E-KANBANMOVE-IS-THE-PARCEL-ADDRESS-STEP-IS-THE-DELIVERY-SCAN-1`,
>   `E-SUBSTRATE-IS-THE-SCHEDULER`.
>
> This is the loop-closure companion to the persistence bootstrap plan, not a
> competing architecture. The driver mints **no** new semantic / temporal /
> rung / witness / branch / ancestry types — it composes existing organs.

---

## 1. The gap — `persist_sink` has ZERO production callers today (the loop is OPEN)

The organs all exist and are merged. The loop is **not** closed.

`lance_graph_planner::persist_sink` — `persist_cycle`, the `WalSink` trait
(`commit_cycle` / `scan_sealed` / `versions`), `recover_and_apply` — has, as of
this plan, **zero production callers** (verified by grep). It is a load-bearing
seam with nothing standing on it. The cycle can be frozen, one WAL write can be
made, one sealed `DatasetVersion` can be returned — but *nothing calls it in a
running loop*, so the version it seals never fans out to advance any mailbox,
and no finished thought ever casts the next cycle's intent.

Everything the loop needs is already built, in five separate crates:

- **persist** — `persist_sink::persist_cycle(sink, frame, casts)` freezes a
  `CycleFrame { cycle, base_version }` + `Vec<SweepSlot>` into one WAL write and
  returns a sealed `DatasetVersion`. Recovery: `recover_and_apply(owner, sealed,
  applied_through)`.
- **schedule (sync)** — `lance_graph_contract::scheduler::VersionScheduler::on_version`
  + `NextPhaseScheduler` (forward-arc Planning→CognitiveWork→Evaluation→Commit,
  Libet −550 µs stamp on the Planning→CognitiveWork Σ-crossing, `None` on
  absorbing).
- **schedule (async subscription)** —
  `lance_graph::graph::scheduler::LanceVersionScheduler::drive_once` /
  `drive_at_latest` — for READING a version you did **not** write (opens the
  Lance dataset per call).
- **apply** — `lance_graph_supervisor::kanban_actor::KanbanActor<O>` (the
  ractor actor whose State IS the owner; applies via `try_advance_phase`) +
  the free fns `drive_version_tick` / `drive_scheduled_tick`.
- **emit** — `lance_graph_planner::owner_adapter::{rebind_bootstrap,
  emit_bootstrap_intent}` turns a finished thought's Outcome into the next
  cycle's intent cast via `batch_writer::BatchWriter::cast(on_behalf, moves,
  payload)`.

The **shipped slice that proves the shape** is
`symbiont::kanban_loop::SymbiontBoard` (D2): it impls `MailboxSoaView` +
`MailboxSoaOwner` over a `Vec<NodeRow>` and its `step(&NextPhaseScheduler)`
drives `version_tick → on_version → try_advance_phase` synchronously, with a
`u32` tick standing in for the real Lance version. The driver **generalizes
that slice** to (a) the real sealed `DatasetVersion` from `persist_cycle` and
(b) a mailbox fleet instead of one `SymbiontBoard`.

The gap, stated once: **no crate composes persist → schedule → apply → emit into
a running cycle.** That composition is the driver. It is a control-loop, not a
new subsystem.

---

## 2. The closed-loop shape — seal → step → think → cast

The driver closes exactly this loop:

```
  collect the fleet's staged BatchWriter casts   → Vec<SweepSlot>
        │
        ▼
  persist_cycle(sink, CycleFrame{cycle, base_version=Vn}, casts)
        │                                    (one WAL write, freeze-before-I/O)
        ▼
  sealed DatasetVersion  Vn+1
        │
        ▼
  fan the step across the mailbox fleet          ← writer fires INLINE (§3)
    NextPhaseScheduler::on_version(view_i, Vn+1, exec)   (sync, pure)
        │
        ▼
  try_advance_phase per mailbox                  ← the KanbanStep (KanbanActor)
        │
        ▼
  CognitiveWork runs the thought                 ← pluggable callback (§5.4 seam)
        │                                          produces an Outcome
        ▼
  owner_adapter: Outcome → emit_bootstrap_intent → BatchWriter::cast(on_behalf=owner)
        │
        └──────────────── back to collect (next cycle Vn+2) ───────────────┘
```

The KanbanMove is the parcel-address; the step is the delivery-scan
(`E-KANBANMOVE-IS-THE-PARCEL-ADDRESS-STEP-IS-THE-DELIVERY-SCAN-1`). The durable
unit is the cycle, not the cast
(`E-THE-DURABLE-UNIT-IS-THE-CYCLE-NOT-THE-CAST-ONE-WAL-WRITE-PER-SWEEP-1`). The
substrate — the sealed version table — IS the scheduler
(`E-SUBSTRATE-IS-THE-SCHEDULER`).

The driver adds **no** node types, **no** move types, **no** scheduler types. It
is glue: it collects the casts the fleet already staged, calls a function that
already exists, and routes the result back through an adapter that already
exists.

---

## 3. Correctness — the writer fires the step INLINE and SYNCHRONOUSLY (not 64k `drive_once`)

This is the load-bearing correctness point.

**The driver WROTE the version** — `persist_cycle` returned `Vn+1`. Because the
driver already holds the version it committed, it fires the fan-step **inline
and synchronously**:

```
NextPhaseScheduler::on_version(view_i, Vn+1, exec)   // sync pure fn, per mailbox
  → try_advance_phase(mailbox_i)                     // per mailbox
```

`on_version` is a **sync pure function** (board D2 line: the writer knows the
version it committed and fires the update inline, no async). The sweep is a
straight-line pass over the mailbox SoA — one sync call per mailbox, then the
per-mailbox `try_advance_phase`. There is **no second dataset read** between the
seal and the step.

The driver does **NOT** fan 64k async `drive_once` / `drive_at_latest` calls.
`LanceVersionScheduler::drive_once` / `drive_at_latest` are the **subscription**
variant — async precisely because they READ a version they did NOT write, and
**each opens the Lance dataset**. Fanning 64k of those across a fleet would be
64k dataset opens to re-read a version the driver already has in hand: wrong,
and quadratically wrong at scale.

The rule, stated for the record:

> **Async is ONLY (a) the `persist_cycle` I/O leg and (b) the subscription
> drive path (a reader that did not write the version). The writer-side
> fan-out is sync `on_version` + `try_advance_phase`, inline, no dataset
> re-read.**

The subscription path (`drive_at_latest`) remains the correct tool for a
*separate* reader process that observes sealed versions it did not produce —
that reader legitimately opens the dataset because it has no other handle to the
version. That is a different actor from the driver and out of scope here.

---

## 4. Deliverables (probe-first — a falsifier per deliverable)

Per the workspace falsifiability rule: *what input makes this fail?* Every
deliverable names its falsifier; anti-vacuity and can-it-fire / can-it-stay-
silent twins are called out where the naive assertion would be vacuous.

| ID | Deliverable | Falsifier (what input makes it fail) |
|---|---|---|
| **D-MBX-A6-P4a** | **CycleDriver skeleton** — drain the fleet's `BatchWriter` casts into `Vec<SweepSlot>` → `persist_cycle` → sealed `DatasetVersion`. | N staged casts produce **exactly one** WAL write + **exactly one** version (reuse `persist_sink`'s amortization probe at the driver level — assert `commit_cycle` invoked once, not N times). |
| **D-MBX-A6-P4b** | **fleet fan-step** — sealed version → sync `on_version` sweep over the mailbox SoA → `try_advance_phase` per mailbox (via `KanbanActor` / owned sweep). | (i) **anti-vacuity:** a fleet with a *mix* of phases — each mailbox advances by exactly one **legal forward-arc** step AND assert different mailboxes took **different** steps (not lockstep-blind); (ii) an **absorbing** mailbox fires nothing (`on_version` → `None`); (iii) assert **NO second dataset read** between seal and step (proves the writer fires inline per §3, not via a re-read `drive_once`). |
| **D-MBX-A6-P4c** | **loop closure** — a CognitiveWork Outcome → `owner_adapter::emit_bootstrap_intent` → `BatchWriter::cast` into the NEXT cycle → appears in Vn+1's collected casts (round-trip). | An Outcome cast in cycle N is **present in cycle N+1's collected casts** AND advances the owner **one step further** (not merely enqueued — actually collected and applied next cycle). |
| **D-MBX-A6-P4d** | **wait-free-emit guard** — a mailbox whose neighbour has NOT completed still advances (no synchronous neighbour wait). | **can-it-fire:** construct a fleet where mailbox B is mid-thought and mailbox A completes — A **still steps in the same cycle**; assert **no barrier** / no neighbour wait blocked A. |
| **D-MBX-A6-P4e** | **recovery composition** — `recover_and_apply` replays the owner's pending tail after a mid-loop stop, idempotent with the watermark. | Stop mid-loop, re-drive, assert **no double-apply** (reuse the `persist_sink` watermark probe at the driver level — `applied_through` gates the replay so a re-applied slot is a no-op). |
| **D-MBX-A6-P4f** *(SCALE, gated on W2a)* | **16k / 64k mailboxes fan in one cycle** within the cycle budget. | **MEASURED**, labelled a **scale gate, not a correctness claim**: 16k/64k mailboxes fan in one cycle within the ~0.5–2.5 s/cycle budget; **log what was measured, never a silent cap**. |

**Sequencing:** P4a (collect+seal) and P4b (fan-step) are the spine; P4c closes
the round-trip; P4d and P4e are the wait-free + recovery guards on the spine;
P4f is the scale gate, deferred with W2a (§6).

---

## 5. Home, dependency direction, and the WalSink-fake honesty

### 5.1 HOME — `lance-graph-supervisor` (with a stated fallback)

**Decision:** the driver lives in **`lance-graph-supervisor`** — the structural
fleet owner. It already owns `KanbanActor<O>` + the owner-apply surface
(`try_advance_phase`, `drive_version_tick`, `drive_scheduled_tick`), which is
exactly the "apply" leg of the loop. Putting the control-loop next to the apply
surface keeps the fan-step where the fleet ownership already is.

The supervisor crate currently deps **only** `lance-graph-contract` (NOT
planner). The driver requires the planner's `persist_sink`, `owner_adapter`, and
`batch_writer`, so the wiring task adds a **`lance-graph-planner` path-dep** to
supervisor. This is safe: **planner does NOT dep supervisor** (verify with
`cargo tree` before landing), so there is no cycle.

**Fallback (stated as a decision, not left open):** if adding the planner dep to
supervisor surfaces a cycle (e.g. planner gains a supervisor dep in the
meantime), the driver instead lives **in `lance-graph-planner`** alongside
`persist_sink` / `owner_adapter` / `batch_writer`, and reaches the apply surface
through the contract's `VersionScheduler` + `MailboxSoaOwner` traits rather than
the concrete `KanbanActor`. The control-loop shape (§2, §3) is identical either
way; only the crate boundary moves.

### 5.2 Dependency direction

```
lance-graph-supervisor ──(new path-dep)──► lance-graph-planner
                       ──(existing)──────► lance-graph-contract
lance-graph-planner    ──(existing)──────► lance-graph-contract
  (planner does NOT dep supervisor — no cycle; verify via cargo tree)
```

### 5.3 The WalSink-fake honesty

`WalSink` has **no concrete sink yet** — the concrete `LanceShardSink` is
deferred, gated on crash falsifiers (per `persistence-cycle-wal-bootstrap-v1.md`
§4). So the driver initially wires against the **same in-process fake / MemWAL
slice** the `persist_sink` probes use. This is honest and deliberate:

> **The driver closes the CONTROL loop; the durability leg stays the
> contract-probe fake until `LanceShardSink` lands.** "Control loop closed,
> durability leg still fake" is the accurate status — `compile+test green ≠
> storage proven` (the Ladybug rule). The P4a…e falsifiers all pass against
> the fake sink because they probe the *control* invariants (one seal, one
> version, inline fan, round-trip, watermark idempotence), none of which need
> real crash durability. Only P4f-real-durability would need the concrete
> sink, and P4f as specified is a fan-out **scale** measurement, not a
> durability claim.

### 5.4 CognitiveWork execution is a pluggable seam (NOT designed here)

The **thought body** — what CognitiveWork actually runs (shader / StyleStrategy
P3a/P3b) — is a **pluggable callback**, not re-specified in this plan. The
driver's job is to **FIRE the Planning→CognitiveWork step** and, after the
thought produces an Outcome, **route that Outcome through `owner_adapter` into
the next cycle's casts**. Treat thought execution as a seam:
`Fn(&Owner) -> Outcome` (or the equivalent trait object). Do **not** design the
shader here.

---

## 6. 64k mechanics + the W2a scale gate

**Scale framing** (per `mailbox-cycle-aware-write-contract-v1.md`): one basin =
one prefix table = **16k mailboxes**; **64k = ~4 basins** = the sweep target.
The fan-step (§3) is a straight pass over the fleet SoA, so 64k mailboxes = one
sync sweep of ~4 prefix tables, not 64k async operations.

**Why `persist_sink`'s guarantees make 64k concurrent casts safe:**

- **freeze-before-I/O** — the cycle is frozen on a detached snapshot before the
  WAL append, so 64k concurrent casts collect into one immutable `Vec<SweepSlot>`
  without a live mutable SoA borrow crossing the I/O.
- **one WAL write per sweep** — 64k casts amortize into a single `commit_cycle`,
  so the durable-write cost is O(1) in cycles, not O(64k) in casts.
- **sealed read horizon** — every mailbox in the sweep reads exactly one sealed
  predecessor `Vn`; the open cycle (`Vn+1` accumulating) is excluded, so it is
  safe to read `Vn` while `Vn+1` accumulates the next 64k casts.

**W2a scale gate (D-V3-W2a, board-as-tenant, currently GATED/deferred):** the
driver targets the **existing `MailboxSoaView::phase()` surface today** and
adopts the per-mailbox board **tenant column** (kanban board as `ValueTenant`)
when W2a un-gates. W2a is a **scale / cleanliness gate** — the fan-out becomes a
tenant *column read* instead of per-mailbox structs — **NOT a hard blocker** for
the control-loop shape. The loop closes on the `phase()` surface now; W2a makes
the 64k fan cheaper and cleaner later. This is exactly why **P4f is gated on
W2a** and labelled a scale gate, while P4a…e are not.

---

## 7. Constraints / scope exclusions

- Do **not** introduce or document **cohort internals**, participant-count
  encoding, bitmap layout, or actor-neighbour firing dependencies — separate
  cohort architecture work.
- Do **not** invent new **semantic, temporal, rung, witness, branch, or
  ancestry** types. Reuse `KanbanMove` / `DatasetVersion` / `SweepSlot` /
  `CycleFrame` / `BatchWriter` / `NextPhaseScheduler` / `KanbanActor` /
  `owner_adapter` / `recover_and_apply` **verbatim**.
- The driver is a **control-loop composing existing organs** — it mints no new
  subsystem.
- **Persistence stays storage-only**; the concrete `LanceShardSink` stays
  **deferred** (gated on crash falsifiers). The driver wires the fake sink.
- Do **NOT** modify `persist_sink.rs` or `temporal.rs`.
- Do **not** design the CognitiveWork shader / StyleStrategy — thought execution
  is a pluggable seam (§5.4).
- **Status discipline:** the driver is **PLANNED / CONJECTURE** (design), not
  shipped; each claim is probe-gated, promoted to FINDING only when its falsifier
  runs green.

---

## 8. Status snapshot

| Aspect | State |
|---|---|
| `persist_sink` cycle/WAL seam (`persist_cycle` / `WalSink` / `recover_and_apply`) | **SHIPPED** (D-MBX-A6-P1…P3e) — but **ZERO production callers** (the loop is open) |
| `VersionScheduler` + `NextPhaseScheduler` (sync `on_version`) | **SHIPPED** contract (D-MBX-9-IN) |
| `KanbanActor<O>` + owner-apply (`try_advance_phase`) | **SHIPPED** (D-V3-W2b) |
| `owner_adapter` + `BatchWriter` (Outcome → next-cycle cast) | **SHIPPED** (planner) |
| `symbiont::kanban_loop::SymbiontBoard` (the shape-proving slice) | **SHIPPED** (D2) — `u32` tick placeholder for the real version |
| **CycleDriver** (P4a…f — closes seal→step→think→cast) | **PLANNED / CONJECTURE** — this plan; probe-gated |
| Home = `lance-graph-supervisor` + new planner path-dep (fallback: planner) | **DECIDED** (§5.1) — verify no cycle via `cargo tree` |
| Durability leg (concrete `LanceShardSink`, real crash durability) | **DEFERRED** — driver wires the contract-probe fake; control loop closes regardless |
| Board-as-tenant fan-out (D-V3-W2a) | **GATED** — driver uses `phase()` today; P4f scale gate adopts the tenant column when W2a un-gates |

The organs exist; the loop does not. This plan is the record of the one seam
that makes the merged persistence bootstrap load-bearing — and of the honest
boundary that the control loop closes now while the durability leg stays a fake
until the crash falsifiers earn the concrete sink.
