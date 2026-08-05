# cycle-loop-closure-driver-v1 — the loop-closure driver that makes the persist_sink cycle/WAL seam load-bearing

> **Status:** IMPLEMENTED (slice, PR #879) — updated 2026-08-02 after the
> grain-of-salt review round. `lance-graph-supervisor::cycle_driver` (feature
> `cycle-driver`) ships P4a–P4f as **control-loop contract probes**: retry-safe
> seal, restart-stable stream positions (`position_base` durable cursor),
> watermark-coupled normal apply, pre-seal ≤1-move/owner partition
> (`HeldIntent`/`restage_held`), Hold-as-reschedule, prefix-preserving apply
> errors. **Honesty ledger:** control-loop contract PROVEN · actor-owned
> production wiring NOT proven (`MailboxFleet` HashMap = probe/registry fleet;
> `KanbanActor` bridging open) · cognitive-shader-driver/MailboxSoA thought NOT
> proven (the MUL gate is real, its qualia inputs extractor-fed) · **durability
> stays the contract-probe fake** until the concrete `LanceShardSink` lands (the
> `compile+test green ≠ storage proven` Ladybug rule).
> **Date:** 2026-08-02.
> **Scope:** documentation-only architectural ruling. Records the *missing
> seam* — the driver that turns the already-merged `persist_sink` cycle/WAL
> bootstrap into a running loop at 64k concurrency — and the deliverables +
> falsifiers that gate its construction. Changes **no** Rust code, tests,
> public APIs, `persist_sink.rs`, or `temporal.rs`.
> **Owns (narrowly):** "why the loop is open today", the closed-loop
> seal→step→think→cast shape, the SPARSE sealed-transition application rule +
> the writer-fires-inline correctness point, the D-MBX-A6-P4a…f deliverables,
> the home + dep-direction decision, the D-MBX crate-responsibility map (§9),
> the adjacent-crates doctrine (§10), the subagent guardrail (§11), and the 64k
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
running loop*, so the version it seals never triggers any sealed owner's step,
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
  sealed DatasetVersion Vn+1  +  the sealed PAIRED-TRANSITION SET
        │                        (only the cycle's SweepSlots carrying a paired_move —
        │                         a SPARSE subset, NOT the whole fleet)
        ▼
  supervisor iterates ONLY the sealed paired transitions   ← writer fires INLINE (§3)
    for each sealed (owner, paired_move):
        resolve owner → try_advance_phase(paired_move.to)   ← the KanbanStep (KanbanActor)
    all UNREPRESENTED owners remain BYTE-IDENTICAL (untouched)
        │
        ▼
  owners entering CognitiveWork run the thought  ← pluggable callback (§5.4 seam)
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

## 3. Correctness — SPARSE sealed-transition application, fired INLINE (a version is NOT permission to advance every mailbox)

Two load-bearing correctness points. The first is the one the earlier draft of
this plan got **wrong** and this revision fixes.

### 3.1 A new version is global knowledge; Kanban mutation is SPARSE and owner-specific

> **`DatasetVersion` is global knowledge. Kanban mutation is sparse and
> owner-specific. A new version is NEVER, by itself, permission to advance every
> mailbox.**

The **rejected** model (the earlier draft, corrected here): "sealed
`DatasetVersion` → fan `NextPhaseScheduler::on_version` across the fleet →
advance every non-absorbing mailbox." That is wrong — it makes almost the entire
fleet dirty every cycle (every mailbox's phase changes), which **violates the
sparse-cycle ruling** (`persistence-cycle-wal-bootstrap-v1.md` §2 /
`E-COMPLETE-CYCLE-IS-PHYSICALLY-SPARSE-NOT-A-FULL-REWRITE-1`). A global clock
tick is not a fleet-wide step signal.

The **correct** production loop:

- **Mailboxes think concurrently over the sealed `Vn`.**
- **Owners that produce material updates emit fire-and-forget intent** — a
  `SweepSlot` carrying a `paired_move` — through `BatchWriter::cast` (via
  `owner_adapter`). The intended move is decided at **intent time** by the
  planner (StyleStrategy / `owner_adapter`, optionally using the
  `NextPhaseScheduler` forward-arc as the lowering policy). It is *proposed*, not
  yet authoritative.
- **Planner** collects + coalesces the sparse casts, freezes one cycle, performs
  **one WAL transaction**, receives the sealed `DatasetVersion Vn+1`, and exposes
  the **sealed paired-transition set** (the SweepSlots that carry a `paired_move`).
- **Supervisor** iterates **only the sealed paired transitions**, resolves the
  corresponding owner for each, and applies **one legal transition** to each
  **represented** owner via `try_advance_phase(paired_move.to)`. **All
  unrepresented owners remain byte-identical** — they are not touched, not
  re-serialized, not swept.
- **Owners entering CognitiveWork** run the native thought body, produce
  Outcomes, and route them through `owner_adapter` into the next cycle.

So the phase mutation is driven by the owner **having produced a sealed intent**,
never by the version tick fanning to everyone. `NextPhaseScheduler::on_version`
is the intent-time **lowering policy** (which move is legal for an owner that
produced an update), *not* an apply-time fan across the fleet. The symbiont
`SymbiontBoard.step` (D2) advances every board per tick — that is the SLICE
shape-prover, **not** the production apply rule.

**Interim conservative rule (record it):** *at most one durable Kanban phase
transition per owner per sealed cycle.* Multiple data updates for the same owner
may coalesce (per-row, per `persist_sink`), but additional **state-dependent
phase transitions wait for the next sealed horizon** — an owner advances at most
one Rubicon edge per seal.

### 3.2 The writer fires the sealed transitions INLINE and SYNCHRONOUSLY (not 64k `drive_once`)

**The supervisor/driver WROTE the version** — `persist_cycle` returned `Vn+1`
and the sealed paired-transition set is already in hand. So it applies those
transitions **inline and synchronously** — a straight iteration over the sparse
sealed set, one `try_advance_phase` per represented owner. There is **no second
dataset read** between the seal and the step.

The driver does **NOT** fan async `drive_once` / `drive_at_latest` calls.
`LanceVersionScheduler::drive_once` / `drive_at_latest` are the **subscription**
variant — async precisely because they READ a version they did NOT write, and
**each opens the Lance dataset**. Using them here would re-read a version the
driver already holds; across a fleet that is 64k dataset opens for nothing.

> **Async is ONLY (a) the `persist_cycle` I/O leg and (b) the subscription drive
> path (a reader that did not write the version). The writer-side application of
> the sealed sparse transitions is inline `try_advance_phase`, no dataset
> re-read.**

The subscription path (`drive_at_latest`) remains the correct tool for a
*separate external reader* that observes sealed versions it did not produce (it
legitimately opens the dataset because it has no other handle to the version).
That reader is a different actor from the driver and out of scope here
(D-MBX-9-IN external-reader implementation, `lance-graph`).

---

## 4. Deliverables (probe-first — a falsifier per deliverable)

Per the workspace falsifiability rule: *what input makes this fail?* Every
deliverable names its falsifier; anti-vacuity and can-it-fire / can-it-stay-
silent twins are called out where the naive assertion would be vacuous.

| ID | Deliverable | Falsifier (what input makes it fail) |
|---|---|---|
| **D-MBX-A6-P4a** | **supervisor drains planner casts and calls `persist_cycle`** — collect the fleet's staged `BatchWriter` casts into `Vec<SweepSlot>` → `persist_cycle` → sealed `DatasetVersion` + the sealed paired-transition set. | N staged casts produce **exactly one** WAL write + **exactly one** version (reuse `persist_sink`'s amortization probe at the driver level — assert `commit_cycle` invoked once, not N times). |
| **D-MBX-A6-P4b** | **supervisor applies ONLY the sealed sparse transition set** — iterate the cycle's sealed `paired_move` SweepSlots, resolve each owner, apply one legal `try_advance_phase`; leave every unrepresented owner byte-identical. | **The sparse falsifier:** 64k registered mailboxes; **17** owners have sealed paired transitions → **exactly those 17 owners advance** → **all other owner rows remain byte-identical** → **no second dataset read** → **one `DatasetVersion`**. (Anti-vacuity: assert the untouched set is the other 64k−17, not merely that 17 advanced; assert a mix of legal edges, not lockstep.) |
| **D-MBX-A6-P4c** | **owners entering CognitiveWork run the thought and cast the next intent** — the represented owner runs the pluggable thought body, produces an Outcome, routes it via `owner_adapter::emit_bootstrap_intent` → `BatchWriter::cast` into the NEXT cycle. | An Outcome cast in cycle N is **present in cycle N+1's collected casts** AND, when N+1 seals, advances that owner **one further legal step** (round-trip: not merely enqueued — collected and applied next cycle). |
| **D-MBX-A6-P4d** | **one completed owner never waits synchronously for an unrelated owner** — the emit path is wait-free; a finished owner casts + advances without blocking on a neighbour. | **can-it-fire:** a fleet where owner B is mid-thought and owner A completes — A **still emits + (if sealed) advances in the same cycle**; assert **no barrier / no neighbour wait** blocked A. |
| **D-MBX-A6-P4e** | **supervisor composes planner recovery, applies only unreplayed moves** — on a mid-loop restart, `recover_and_apply` replays the owner's pending tail, idempotent with the durable watermark. | Stop mid-loop, re-drive, assert **no double-apply** (reuse the `persist_sink` watermark probe — `applied_through` gates the replay so an already-applied slot is a no-op; represented owners advance once, unrepresented untouched). |
| **D-MBX-A6-P4f** *(SCALE, gated on W2a)* | **measure sparse routing + cycle cost at 16k / 64k** — the cost of resolving + applying the sealed sparse set (NOT a full sweep) at fleet scale. | **MEASURED**, labelled a **scale gate, not a correctness claim**: at 16k/64k registered mailboxes with a realistic sparse dirty fraction, measure sealed-set routing + apply within the ~0.5–2.5 s/cycle budget; **log the dirty fraction + what was measured, never a silent cap**. |

**Sequencing:** P4a (collect+seal) and P4b (apply the sparse sealed set) are the
spine; P4c closes the round-trip; P4d and P4e are the wait-free + recovery guards
on the spine; P4f is the sparse-routing scale gate, deferred with W2a (§6).

---

## 5. Home, dependency direction, and the WalSink-fake honesty

### 5.1 HOME — `lance-graph-supervisor` (with a stated fallback)

**Decision:** the driver lives in **`lance-graph-supervisor`** — the structural
fleet owner. It already owns `KanbanActor<O>` + the owner-apply surface
(`try_advance_phase`, `drive_version_tick`, `drive_scheduled_tick`), which is
exactly the "apply" leg of the loop. Putting the control-loop next to the apply
surface keeps the sparse sealed-transition apply where the fleet ownership
already is.

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
> version, inline sparse apply, round-trip, watermark idempotence), none of which need
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
one prefix table = **16k mailboxes**; **64k = ~4 basins** = the registered-fleet
size. **The apply cost scales with the SPARSE sealed-transition set, not the
fleet.** The supervisor iterates only the cycle's sealed `paired_move` SweepSlots
(§3.1) — 17 dirty owners cost 17 `try_advance_phase` calls, not a 64k sweep — and
the other ~64k owners are never touched. This is the whole point of the
sparse-cycle ruling: registration is 64k; mutation is the dirty subset.

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
when W2a un-gates. W2a is a **scale / cleanliness gate** — resolving the sealed
owners becomes a tenant *column read* instead of per-mailbox structs — **NOT a
hard blocker** for the control-loop shape. The loop closes on the `phase()`
surface now; W2a makes owner-resolution over the fleet cheaper and cleaner later.
This is exactly why **P4f is gated on
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
| `persist_sink` cycle/WAL seam (`persist_cycle` / `WalSink` / `recover_and_apply`) | **SHIPPED** (D-MBX-A6-P1…P3e) — first caller: `cycle_driver` (PR #879) |
| `VersionScheduler` + `NextPhaseScheduler` (sync `on_version`) | **SHIPPED** contract (D-MBX-9-IN) |
| `KanbanActor<O>` + owner-apply (`try_advance_phase`) | **SHIPPED** (D-V3-W2b) |
| `owner_adapter` + `BatchWriter` (Outcome → next-cycle cast) | **SHIPPED** (planner) |
| `symbiont::kanban_loop::SymbiontBoard` (the shape-proving slice) | **SHIPPED** (D2) — `u32` tick placeholder for the real version |
| **CycleDriver** (P4a…f — closes seal→step→think→cast) | **IMPLEMENTED (slice, PR #879)** — 19 falsifiers green incl. retry-safe seal, restart-stable positions, watermark-coupled apply, pre-seal held-move partition, Hold-reschedule. Actor-owned wiring + shader/SoA thought + durability remain open (header ledger) |
| Home = `lance-graph-supervisor` + new planner path-dep (fallback: planner) | **DECIDED** (§5.1) — verify no cycle via `cargo tree` |
| Durability leg (concrete `LanceShardSink`, real crash durability) | **DEFERRED** — driver wires the contract-probe fake; control loop closes regardless |
| Board-as-tenant owner-resolution (D-V3-W2a) | **GATED** — driver uses `phase()` today; P4f scale gate adopts the tenant column when W2a un-gates |

The organs exist; the loop does not. This plan is the record of the one seam
that makes the merged persistence bootstrap load-bearing — and of the honest
boundary that the control loop closes now while the durability leg stays a fake
until the crash falsifiers earn the concrete sink.

---

## 9. D-MBX crate-responsibility map (the production spine — ratified)

The canonical ownership map. Verified against `Cargo.toml` deps 2026-08-02 (see
§5.2). The distinction that must stay explicit: **MailboxSoA type/layout home =
`cognitive-shader-driver`; exclusive runtime ownership = `lance-graph-supervisor`;
decision + persistence-contract home = `lance-graph-planner`.**

| Crate | Owns | Explicitly does NOT own |
|---|---|---|
| **lance-graph-contract** | Canonical shared types: `KanbanColumn` / `KanbanMove`, `DatasetVersion`, the `VersionScheduler` traits, `MailboxSoaView` / `MailboxSoaOwner`, legal Rubicon transitions. Zero-dep. | fleet ownership; persistence implementation; any thought body. |
| **cognitive-shader-driver** | The canonical **MailboxSoA layout**; native cognition / shader / thinking machinery; thinking atoms, thinking styles, SoA columns — **defines the anatomy**. | the production fleet **lifecycle** (it does not run the runtime loop; its optional `lance-graph-planner` dep is debug/serve DTOs, not fleet ownership). |
| **lance-graph-planner** | **Decides what should happen**: StyleStrategy + StrategyOutcome, the intended `KanbanMove`, `owner_adapter`, `BatchWriter` cast/intents, cycle collection + coalescing, `persist_cycle` / `WalSink` contract, recovery + temporal projection **contracts**. | **never directly mutates a supervisor-owned MailboxSoA**; never depends on supervisor. |
| **lance-graph-supervisor** | **Production runtime owner of MailboxSoA instances**: `KanbanActor` state IS the owner; authoritative phase mutation; the production cycle-loop composition (**D-MBX-A6-P4**); applies only the **sealed sparse transitions**; fires CognitiveWork; returns Outcomes to planner/`owner_adapter`. | it consumes planner; planner never consumes it. |
| **lance-graph** | The actual Lance dataset + `DatasetVersion` substrate; the external-reader version subscription (`LanceVersionScheduler`); the future concrete `LanceShardSink` / physical persistence. | it is a **storage substrate, not a cognitive fleet owner**. |

**D-id allocation (audited):** D-MBX-A1..A5 → `cognitive-shader-driver` (+contract
support); D-MBX-A6-P1/P2 → `lance-graph-contract`; D-MBX-A6-P3a..P3e →
`lance-graph-planner`; **D-MBX-A6-P4a..P4f → `lance-graph-supervisor`** (one-way
dep on planner); D-MBX-9-IN contract → `lance-graph-contract`; D-MBX-9-IN
external-reader impl → `lance-graph`; D-V3-W2b `KanbanActor` →
`lance-graph-supervisor`.

**Dependency direction (verified 2026-08-02, §5.2):**
`lance-graph-supervisor → lance-graph-planner → lance-graph-contract`;
`lance-graph-planner → lance-graph-contract`. **Planner must not depend on
supervisor.** Currently supervisor deps only contract (+ callcenter); the
`supervisor → planner` edge is the **planned P4 wiring** and is acyclic (planner's
dep closure never reaches supervisor). `cognitive-shader-driver` has an optional
(feature-gated) planner dep for debug/serve DTOs — not fleet ownership, not a
cycle.

---

## 10. Adjacent-crates doctrine — bystanders, basements, and adapters (NOT owners)

The production spine (§9) is a straight railway track:
**contract defines · planner proposes and seals · supervisor owns and applies ·
shader thinks · Lance persists.** Adjacent crates **observe, adapt, or provide
optional capabilities** — none is crowned emperor of the hippocampus.

### 10.1 `symbiont` — golden-image + bystander research laboratory

**Allowed:** full-stack compile/link golden image; integration + scale probes;
brainstorming + falsification playground; possible AST-arm experiments
(Elixir-shaped syntax without an Elixir runtime, SurrealQL DDL/expression AST,
OGAR adapter composition); a possible second research leg for
`lance-graph-arm-discovery`, grammar heuristics, time-series observation,
cross-system hypothesis generation.

**Forbidden:** authoritative MailboxSoA owner; production D-MBX scheduler;
production Kanban lifecycle; production WAL owner; independent version authority;
second source of truth for cognition; a **required dependency** of planner,
supervisor, or cognitive-shader-driver.

`SymbiontBoard` (D2) impls `MailboxSoaView`/`MailboxSoaOwner` and its
`step()`-advances-every-board loop is **probe-only and intentionally local** — a
shape-prover, never the production owner. Any reusable production logic found in
symbiont is classified as *probe-only-and-local* or *candidate for later
extraction into its canonical D-MBX crate* — **not extracted in this task**.
Symbiont may observe the brain, suggest patterns to it, and test combinations
around it; it must not become the alien twin driving the hands.

### 10.2 `rs-graph-llm` — optional capability basement

**Useful:** agentic-coding-shaped demonstrations; sparse LLM assistance; ticket
orchestration; Rig integration; OpenClaw / tool-use adapters; optional
CognitiveWork capability providers; human-in-the-loop workflow façades.

**Forbidden:** authoritative MailboxSoA storage; authoritative Kanban state; a
second planning lifecycle; a second WAL or version ledger; mirrored live
cognition state; a **required dependency** of D-MBX core crates.

Composition shape: `application / MedCare composition layer` sits **above** both
the D-MBX production runtime and the optional `rs-graph-llm` / Rig capabilities —
NOT `lance-graph core → rs-graph-llm → duplicated session/Kanban/storage state`.
When `rs-graph-llm` invokes D-MBX thinking it is a **client / capability
provider**; when D-MBX invokes an LLM/tool capability the result returns as an
**Outcome or evidence input**. `rs-graph-llm` **never owns the standing wave**
(consistent with the workspace rule that rig is membrane-tier, not a brain crate).

### 10.3 `ogar-*` universal adapters — AST / declaration / adapter basement

**Allowed:** source AST ingestion; Elixir / Ruby / Python / SQL / SurrealQL
adaptation; Class / ActionDef declaration surfaces; code generation; cold-path
capability descriptions; schema + behaviour translation.

**Forbidden:** live MailboxSoA ownership; an independent D-MBX scheduler;
duplicate runtime Kanban state; standing-wave persistence.

OGAR may **describe available behaviour**; it does not own the living cognitive
cycle.

---

## 11. Subagent anti-drift guardrail (paste into every D-MBX worker brief)

Before any subagent changes D-MBX code, it MUST answer:

1. Is this **shared vocabulary, planning, runtime ownership, cognition, or
   storage**?
2. Which **canonical crate** (§9) owns that responsibility?
3. Does the change create a **second** owner / scheduler / WAL / Kanban lifecycle
   / Session state / `DatasetVersion` authority / MailboxSoA representation?
4. Is **symbiont** being used as a production dependency merely because it already
   links many crates?
5. Is **rs-graph-llm** being allowed to mirror or own live SoA state merely
   because it already has workflow/session abstractions?
6. Could the change be an **adapter, callback, Outcome, or trait seam** instead of
   importing a whole neighbouring runtime?

**STOP and report (do not proceed) when:**

- planner would depend on supervisor;
- a D-MBX **core** crate would depend on symbiont or rs-graph-llm;
- a **second** Kanban phase field appears;
- a **second** cycle/version counter appears;
- a Session snapshot becomes authoritative over MailboxSoA;
- **a `DatasetVersion` tick advances every owner** (the sparse-cycle violation
  this revision fixed — §3.1);
- SurrealDB JSON becomes the live cognition representation;
- a **production type is first declared inside symbiont**.

The desired result is a straight railway track, not a grand unification:
**contract defines · planner proposes and seals · supervisor owns and applies ·
shader thinks · Lance persists**; adjacent crates observe, adapt, or provide
optional capabilities.

---

## 12. Arm BLW — the Bible lens wave: 64k thoughts firing at once, four stances, measurable Horizontverschmelzung

> **Status:** PLANNED / CONJECTURE. Operator-directed 2026-08-04. Adds **no new
> subsystem** — the lens is a thought body in the §5.4 pluggable seam, the
> corpus is the shipped KJV bake, the four stances are **per-verse binary
> projections of** the shipped B6 panel (see §12.3a — the panel's own outputs are
> a ranking, a partition, a lift list and a concept→count map, **none of which is
> a per-verse binary**), and the fusion read is `temporal.rs`'s existing
> version-range surface. §7's
> exclusions hold verbatim: `persist_sink.rs` and `temporal.rs` are **not
> modified**, only consumed.

### 12.1 The shape, and the one architectural decision

**ONE 64k SoA bake of the whole Bible. Not 1+1+4.**

The operator offered two shapes — six parallel SoAs (base + Gadamer + four
lenses), or one sealed series read as a time series. **The six-SoA shape is
rejected, and not on cost grounds:** four lens SoAs would be *copies of the same
64k verse rows* differing only in which stance reads them. A stance read is a
**projection**, not a cross-input derivation of a higher KIND, so the zero-copy
law's ELEVATED carve-out (the `Locus::Quorum` precedent) does **not** cover it —
the `zero-copy-warden` verdict for that shape is **MATERIALIZES**. The 6× memory
is the smaller objection; the law is the real one.

So:

```
one KJV bake  →  ONE tenant, ONE MailboxSoA, 64k verse ROWS  (§12.1a′)
                      │
        cycle Vn:  sparse sealed set (§3) — a ROW-level dirty set inside the
                   one owner, never an owner-level set (owners are tenants)
                      │
        CognitiveWork body (§5.4 seam) = apply stance L to the owner's slice
                      │
        Outcome → owner_adapter::emit_bootstrap_intent → cast into Vn+1
                      │
        seal Vn  ────────────────────────────────────────────┐
                                                             │
   FOUR STANCES ARE FOUR READS OF THE SEALED SERIES, NOT FOUR BAKES
   Hegel(Aufhebung) · Nietzsche(genealogy-flip) · Kant(critique) ·
   Wittgenstein(meaning-as-use)   — all shipped in probe_eyes_opened.rs
```

The lens does not own a mailbox, does not add a node type, and does not change
the stride. It is a function over an owner's arena slice, dispatched through the
seam the driver already exposes.

#### 12.1a′ RETRACTION (operator-ruled 2026-08-04) — §12.1a below is WRONG. An owner is a TENANT, not a shard.

**§12.1a "tiled the bake across 64 owners" is retracted in full.** It is not a
sizing mistake, it is a **category error**, and the canon already said so:
*"one mailbox = one kanban board as **tenant**"* (`CLAUDE.md` §V3 rulings), and
one `MailboxSoA` is moved into exactly **one** `KanbanActor` which is its **sole
mutator** (`E-CE64-MB-4`; `tests/w2b_real_owner_probe.rs` — the SoA is *moved*
into `Actor::spawn`, and that move is the compile-time proof of no aliasing).

So an owner is not a unit you can *multiply to taste*. Splitting the Bible
across 64 mailboxes does not shard a corpus — it **fabricates 63 additional
tenants**, i.e. 64 separate kanban boards for one book. That is a topology
invention dressed up as a memory fix.

**The corpus is ONE tenant.** Its 64k verses are **ROWS inside that one owner's
SoA**, and "64k thoughts firing at the same time" is data-parallelism **over
rows within the owner's slice** — exactly what §12.1's own diagram says
(*"CognitiveWork body = apply stance L to **the owner's slice**"*), and exactly
what the data-flow rule already prescribes: SIMD reads borrowed row slices,
reasoning works on owned `Copy` microcopies, write-back is gated — **no
`&mut self` during computation**.

**What was actually wrong in my two "independent grounds":**

1. *"A sparse sealed set is a sparse set of owners, so one SoA cannot express
   it."* The observation is true and the conclusion is a non-sequitur: **this
   arm does not need owner-sparseness at all.** That mechanic is already proven
   at 64k in `cycle_driver.rs:1098`. The Bible arm's sparseness is **row-level,
   inside one owner**. I solved a problem that belonged to a different layer.
2. *The 384 MiB / ~5.1 MiB-stack figures.* **⊘ ALSO RETRACTED (operator,
   same day) — I measured the wrong object.** The canonical row is
   `NODE_ROW_STRIDE = 512` bytes, const-asserted
   `size_of::<NodeRow>() == 512` (`canonical_node.rs:735, :787`). **The whole
   Bible bake at canon is 65,536 × 512 B = 32 MiB** — trivially resident, no
   tiling, no `#[ignore]`, CI runs the full corpus. The 6,144 B/row I measured
   is `MailboxSoA`'s content/topic/angle hot planes, **12× the canonical node
   row**, which I silently treated as the corpus cost. So there was never any
   memory pressure to solve, and every conclusion drawn from it — tiling,
   the CI/full-scale split, the 24 GiB D-BLW-4 figure — was answering a problem
   that did not exist.

   **The lesson is sharper than the one I first wrote.** I said "measuring a
   real constraint does not license an arbitrary answer to it." True, but it
   let me keep believing the measurement. The actual failure is upstream:
   **I never checked what the number was a number OF.** A figure computed from
   the wrong struct is not a weaker fact, it is not a fact at all — and it is
   more dangerous than no figure, because arithmetic feels like evidence.

   **Open question, deliberately not resolved here** (asked, not concluded —
   this axis has already been wrong three times today): `MailboxSoA` carries
   6,144 B/row against a 512 B/row canon. Is that a deliberate hot working set
   layered above the canonical row, or a divergence from it? Recorded in
   `ISSUES.md`, not answered.

**Consequences, binding:**
- The tiling in §12.1a and everything downstream of it is void. `FULL_TILES`,
  `CI_TILES`, "64 tiles × `MailboxSoA<1024>`" and the `w_slot = tile_index`
  saturation note are all retracted.
- **D-BLW-4's "≥4,096 owners" axis is void** — see §12.3a′. Owner count is a
  deployment-topology property, not a scale knob, so both the inherited
  threshold *and* my "measure it with 4,096 lightweight owners" reply were
  category errors. The 24 GiB figure I derived is meaningless: you would never
  have 4,096 owners for one corpus.
- §12.1a is kept below **only** as the retracted record (append-only canon:
  regrade in place, never delete).

#### 12.1a Correction (2026-08-04): "ONE MailboxSoA" was wrong twice — the bake is TILED **[⊘ RETRACTED — see §12.1a′ above; an owner is a tenant, not a shard]**

The first draft of this section wrote *"64k verse-owners in ONE MailboxSoA"*.
That is corrected in place, on two independent grounds. **The anti-6× ruling
above is untouched** — tiling is a *partition of one corpus*, not a second
projection of it, so the `zero-copy-warden` verdict that rejected the six-SoA
shape does not reach it.

1. **It contradicted the very next line of its own diagram.** A sparse sealed
   transition set is a sparse set of *owners*. One `MailboxSoA` **is** one
   owner, so a single-SoA shape cannot express "17 dirty, not 64k" at all —
   its dirty set is always 0 or 1. The sparse-cycle mechanic (§3), which is
   the whole point of the driver, requires many owners.
2. **It was not constructible.** `MailboxSoA<N>`
   (`cognitive-shader-driver/src/mailbox_soa.rs:58`) allocates `content` +
   `topic` + `angle` as `3 × N × WORDS_PER_FP × 8 B` with
   `WORDS_PER_FP = 256` (ibid.:39, :322-324) = **6,144 B/row**. That is the
   designed hot layout ("~6 KB/thought", ibid.:136-141), not an accident — so
   65,536 verse rows cost **384 MiB of identity planes no matter how they are
   tiled**. Tiling does not reduce that total; it is a fact about the corpus
   size and must be stated wherever a 64k bake is proposed. What tiling *does*
   fix is the second half: `MailboxSoA::new` builds `Self { … }` **by value**,
   and the fixed-size columns hand-sum to ~82 B/row (excluding struct
   padding — this is a sum of the declared array types, not a measured
   `size_of`), so `MailboxSoA<65536>` is a ~5.1 MiB stack temporary against a
   2 MiB default spawned/tokio-worker thread stack. Whether that temporary is
   elided is an optimization detail and not something to build on.
   `MailboxSoA<1024>` is ~82 KiB of stack and 6 MiB of planes per tile.

**Resolved shape: 64 tiles × `MailboxSoA<1024>` = 65,536 verse rows.** Note the
`w_slot < 64` constraint (ibid.:293-296) is exactly saturated at 64 tiles —
`w_slot = tile_index` uses the full 6-bit W field with nothing to spare, so a
corpus larger than 64 tiles needs a second W-dimension, not a wider field.

**Consequence for D-BLW-1's falsifier:** 384 MiB is too heavy for routine CI, so
the falsifier runs at a tractable tile count in CI and the full 64-tile run is a
separate `#[ignore]`d test carrying the byte figure in its reason string. A
`#[ignore]`d test that is never actually run is a claim without a measurement —
the full-scale run must be executed centrally at least once and its result
recorded, or D-BLW-1 is not closed.

### 12.2 Gadamer, mechanically: a priori and hindsight are the SAME data, two reads

Horizontverschmelzung needs no third mode. The sealed version series supports
both readings the operator named, and `temporal.rs` already distinguishes them:

| Gadamer | mechanically | temporal.rs surface |
|---|---|---|
| **a priori** — *Vorurteil*, the prejudice that is the **condition** of understanding, not its defect | the horizon is the **prior sealed version `Vn`**, read at plan-evaluation time and fed into cycle `Vn+1`'s thought | single-version read, `QueryReference::at(Vn, rung)` — **filter** |
| **hindsight** — *wirkungsgeschichtliches Bewusstsein*, fusion recognised after the fact | the horizon is a **version RANGE** `Vn..Vm`, deinterlaced at read time | range read + deinterlace — **cascade** |

**Nothing is chosen at bake time.** One series, two reads, per
`E-MARKOV-TEMPORAL-STREAM-1` (the trajectory lives on the sorted stream; any
width, per-reader rung, replayable). This is why the time-series shape is not
merely cheaper than 1+1+4 — it is the only one where the a-priori and hindsight
readings are *the same object*.

> **Precision note (2026-08-04), verified against source before use in this
> arm:** the surfaces exist as named — `QueryReference::at(ref_version, rung)`
> (`lance-graph-planner/src/temporal.rs:167`) and `deinterlace(rows, v_ref,
> deps)` (ibid.:346) — but `deinterlace` is `-> Vec<R>` and `.cloned()`s the
> admitted rows (ibid.:351-364). It is a **filtered selection with clone**, not
> a zero-copy projection. The stream doctrine's "zero copies" is therefore
> dropped from the sentence above, and **no D-BLW-3 result line may claim the
> hindsight read is zero-copy.** In this arm the cloned rows are small per-verse
> verdict records, so the cost is a selection over lightweight rows and not a
> copy of the substrate — which is why this is a *wording* correction and not a
> blocker. `temporal.rs` is **not** modified (§12.5); the inaccuracy is recorded
> where it is consumed, not patched where it is defined.

### 12.3 Deliverables

| ID | Deliverable | Falsifier (what input makes it fail) |
|---|---|---|
| **D-BLW-1** | **one 64k KJV SoA + the lens body in the §5.4 seam** — verse-owners registered in one `MailboxSoA`; `CognitiveWork` dispatches a stance over the owner's slice; Outcome round-trips via `emit_bootstrap_intent`. | Reuse P4a/P4b/P4c verbatim at KJV scale: N casts → **one** WAL write + **one** version; only the sealed sparse set advances (**anti-vacuity:** assert the untouched remainder is byte-identical, not merely that the dirty set moved); an Outcome cast in `Vn` is collected and applied in `Vn+1`. |
| **D-BLW-2** | **the four stances as reads over the sealed version** — Hegel / Nietzsche / Kant / Wittgenstein each produce a per-verse binary verdict from `at(Vn)`. | **The discrimination twin — the gate most likely to fail.** Pairwise `jc::stats::binary_association` over verses: (a) *can-discriminate* — at least one lens pair has κ materially **below** 1 on a non-trivial share of units; (b) *can-agree* — at least one pair has κ materially **above** 0. Four lenses that rank everything identically carry exactly as much information as one (the `closed_class_guess` 150/150 defect); four that agree nowhere are noise, not perspectives. **Report the full table — counts + BOTH marginals + `p_o`/`p_e` — never bare κ** (`BinaryAssociation` exists precisely because κ and φ are uninterpretable without marginals). |
| **D-BLW-3** | **Horizontverschmelzung as a measured trajectory** — pairwise lens agreement tracked across the sealed series `V1..Vn`, under both the a-priori (single-version) and hindsight (range) reads. | **Fusion must MOVE.** If pairwise κ between two lenses is flat across the series, no horizons merged and the word is decoration. **Kill condition:** flat κ ⇒ the claim regrades to *"four independent stance reads over a shared corpus"* — still true, still useful, **not Gadamer**. The two reads must also be compared: if the a-priori and hindsight trajectories are identical, the distinction is not doing work and should be dropped rather than narrated. |
| **D-BLW-4** *(scale)* | **64k concurrent thought bodies** — the parallelism claim, at KJV scale. | Inherits W2's **pre-registered, non-adjustable** thresholds: median of ≥5 runs after one discarded warm-up; ≥2× speedup at ≥4,096 owners with ≥100 µs bodies. **Kill:** failure regrades claim (a) to *"64k-scale **sequential** sparse cycles"* — still true, different claim. |

### 12.3a Adjudicated design (2026-08-04) — four premises of §12.3 were wrong, verified in source

A D-BLW-2 design pass checked §12.3's premises against the code. Four did not
survive. Each was **re-verified independently before being recorded here**; the
line numbers are the checks, not the report.

**(1) Hegel is constant-false on the TSV path — so the cheap route is dead.**
`reason_whole_book.rs:92-96` observes every triple at `TruthValue::new(1.0, 0.9)`,
and `BeliefArena::revise_at` sets `depth = (b.truth.frequency − new.frequency).abs()`
(`belief.rs:194`). Uniform frequency ⟹ `depth ≡ 0.0` ⟹ `Belief.contradiction`
never leaves `0.0` ⟹ `contradiction_ranking`'s `> 0.05` filter is empty for the
whole book. **Consequence:** the "re-derive the four binaries from arena + TSV"
option is not cheap-but-lossy, it is *impossible for two of four lenses*.

**(2) Negation never reaches the inbound leg, so extending the TSV cannot fix
(1).** `deepnsm_v2::Spo` is three `WordId`s with no polarity field, and `not`
carries PoS `x` → `Pos::Other` → skipped by the FSM. `Provenance.negated` is the
sole input to the Nietzsche stance, and negation is the only source of the
low-frequency emissions that create contradiction depth at all. Adding TSV
columns would mean porting the clause machine into the inbound leg — which
`E-DEEPNSM-V2-IS-INBOUND-LEG-REASONING-LIVES-IN-LANCE-GRAPH-1` forbids.

**(3) The obvious Kant bit is a tautology.** `RungLift.quale = modal * staunen_at`
(`probe_eyes_opened.rs:465`) and the panel's ablated value is
`UNIFORM_MODAL(0.5) * l.staunen_at` (ibid.:616, 624). So
`quale > ablated ⟺ (modal − 0.5)·staunen_at > 0`, and **both** shipped modals
(0.85, 0.70) exceed 0.5 — the bit is true for every verse holding any lift.
That is the `closed_class_guess` 150/150 defect, caught *before* it was written.
**The Kant binary is therefore rank-based:** true iff the verse holds a lift the
graded ordering ranks strictly higher than the uniform-modal ordering does.
Ranking is relative, so promotions and demotions balance and the positive rate
cannot reach 1 by construction. **Mandatory companion:** report
`binary_association(kant, modal_only)` where `modal_only[i] = ∃ lift at i with
modal > 0.7`; κ ≥ 0.95 means the lens is a re-labelled verb detector and **the
result line must say so** rather than present it as a stance.

**(4) `QueryReference::at` is a reader PIN, not a data read — but D-BLW-3 is NOT
blocked.** The design pass concluded D-BLW-3 was blocked because
`at(ref_version, rung) -> Self` (`temporal.rs:167`) returns a coordinate and
nothing materializes a `BeliefArena` from a `LanceVersion`. The first half is
right; **the conclusion is not, and this plan overrides it.** D-BLW-3 never needed
arena reconstruction: `deinterlace(rows, v_ref, deps)` (`temporal.rs:346`) takes
**caller-supplied rows** over the public, externally-implementable
`trait DeinterlaceRow` (`temporal.rs:318`) with `NoDeps` (`temporal.rs:271`)
already provided. So the harness emits one lightweight **per-(verse, version)
verdict row** as the sealed series is produced, implements `DeinterlaceRow` on it
(`lance_version()` = the sealing version), and gets **both** reads off the real
surface: a-priori = `deinterlace` at `QueryReference::at(Vn, rung)`; hindsight =
the same over a version range. Nothing is reconstructed and nothing in
`temporal.rs` is modified (§12.5 holds).

**Placement ruling: lift the machinery into the library.** `stream` / `Interner` /
`ReadOut` / `Provenance` / `RungLift` / `FlipKind` / `contradiction_ranking` /
`stance_panel` move from `probe_eyes_opened.rs` into
`lance_graph_planner::nars::stance`; the probe keeps its `main()` and imports
instead of defining. **The lift's own falsifier is that the probe's B1–B6 asserts
stay byte-for-byte green** — if they move, the lift changed behaviour. Rationale:
options (2) and (1) are dead, and a fresh re-statement in the BLW module would
create a *second, divergent* definition of four stances.

**Known scope not yet paid (do not discover this late):** the labelled verse
parser `parse_kjv_genesis` hard-stops on a Genesis-specific end marker
(`probe_eyes_opened.rs:802`), while `bible_wave`'s splitter runs the whole book
but keeps **no** chapter:verse label. The BLW module needs labelled verses for
the whole book, so generalizing the parser is real work, not a config change.

**Pre-registered discrimination-twin thresholds** (fixed here, before any run,
**non-adjustable after** — a miss is a miss). Six pairs over the four lenses:
- **can-discriminate:** ∃ a pair with `kappa = Some(k)`, `k ≤ 0.80`, **and**
  `(n01 + n10) ≥ 0.05·N`. `0.80` is the Landis–Koch floor of the "almost
  perfect" band — an external convention that predates this corpus and so cannot
  have been fitted to it. The count clause supplies §12.3's "non-trivial share"
  on *counts*, because κ can fall well below 1 on a handful of discordant cells
  when the marginals are lopsided.
- **can-agree:** ∃ a pair with `k ≥ 0.20` **and** both positive rates in
  `[0.05, 0.95]`. `0.20` is the Landis–Koch slight/fair boundary; the marginal
  guard is what stops two near-constant lenses "agreeing" on a sea of `false`.
- **corpus floor:** `N ≥ 1,000` verses, so the 5 % disagreement floor is ≥ 50
  discordant cells. Below that the marginals are too noisy to read and the twin
  is not reported at all. The 13-verse inline fixture is **far** below this and
  must never be used to claim the twin.
- The two halves MAY be satisfied by different pairs; if one pair satisfies both,
  that is reported explicitly — it means one pair is doing all the work.

**Degeneracy assertions (a κ that is printable but meaningless must be visible):**
compute each lens's positive rate *before* pairing and assert `0 < rate < 1`; any
lens outside `[0.01, 0.99]` is stamped `DEGENERATE`, **excluded from both halves'
∃-quantifier, and the exclusion printed** — never silent. A pair with
`expected_agreement > 0.95` is stamped `UNSTABLE` and cannot satisfy *can-agree*.
`binary_association` returning `None` is a **KILL** naming the pair, never a
skipped row. Assert the six tables are not all identical. `kappa`/`phi` print as
`undefined(p_e=1)` / `undefined(constant)` when `None` — **never `0.0`, never
blank, never omitted.**

**Two diagnostics that must ship with the numbers, because they are directions of
known bias, not hypotheticals:** (a) `stream` normalizes all personal pronouns to
one corpus-wide referent, so statements from distant books collide and revise
against each other — report the share of Hegel-positive verses whose triggering
statement is that referent; (b) `Stamp::source(id) = 1 << (id % 64)` saturates
after ~64 distinct sources, after which observations route to CHOICE rather than
revision, **suppressing** contradiction on exactly the hub statements (a) inflates
— report the count of beliefs with a saturated stamp.

#### 12.3a′ D-BLW-4's AXIS IS OWNERS — and that is void (operator-ruled 2026-08-04)

**The paragraph that stood here is retracted.** It said D-BLW-4's "≥4,096
owners" threshold was unmeetable at 24 GiB and should therefore be measured with
4,096 *lightweight* owners. Both halves are category errors, and the second is
the worse one: it kept owner-count as the axis and merely made the owners cheap.

**Owner count is not a scale knob.** An owner is a **tenant** — one mailbox, one
kanban board, one `KanbanActor` that is its sole mutator (`CLAUDE.md` §V3;
`E-CE64-MB-4`). "4,096 owners" therefore means *4,096 tenants*, which for one
corpus is not a big configuration — it is a **fabricated deployment**. The
24 GiB figure I derived from it is meaningless: nobody would ever hold 4,096
owners for one book, so its cost was never the constraint.

**The real axis is rows inside one owner.** The arm's claim — *"64k thoughts
firing at the same time"* — is data-parallelism over the **verse rows of a single
owner's slice**, which is what §12.1's diagram said all along
(*"apply stance L to **the owner's slice**"*). So D-BLW-4 measures:

> concurrent vs sequential evaluation of **N row-level thought bodies within one
> owner**, where the reads are borrowed row slices, the reasoning is on owned
> `Copy` microcopies, and write-back is **gated** — never `&mut self` during
> computation (`.claude/rules/data-flow.md`, `borrow-strategy.md`).

That axis needs no fabricated tenants, costs one SoA, and is the thing the arm
actually claims. The inherited A2/W2 protocol (median of ≥5 runs after one
discarded warm-up; a can-fire *and* a can-stay-silent half) carries over
unchanged — **only the unit being scaled changes, from owners to rows.** The
per-row work threshold and the row count are re-pinned when the harness is
written, and pre-registered before it runs.

**Kill condition, restated:** if row-level concurrency does not beat sequential
under the pre-registered protocol, claim (a) regrades to *"64k-scale
**sequential** row evaluation"* — still true, different claim.

**New dependency, declare it:** `crates/jc` is workspace-EXCLUDED and currently
has **zero** consumers anywhere in the workspace. The twin harness is the first,
as a `[dev-dependencies]` path edge from `lance-graph-planner`. Do **not** invert
it — hosting the harness inside `jc` would drag the planner's whole dep tree into
a crate whose constitution is zero-dep, and §12.5 keeps `jc` the untouched oracle.

#### 12.3a″ MEASURED RESULT (2026-08-04): D-BLW-2 is a STRUCTURAL KILL on the TSV path

Built and **run** against the real export (`/tmp/kjv_spo.tsv`, 40,767 triples
over 20,022 distinct verses from the whole-book run). The twin did not miss a
threshold — **it has no pair to test.**

**§12.3a undercounted the unreachable stances: it is 3 of 4, not 2.**

| stance | verdict | why |
|---|---|---|
| **Hegel** | reachable, **DEGENERATE** | positive rate **0.000000** — exactly as §12.3a point 1 predicted (uniform `TruthValue::new(1.0, _)` ⟹ `revise_at`'s `\|f₁−f₂\|` depth is always 0) |
| **Nietzsche** | **UNREACHABLE** | needs `Provenance.negated`; no TSV column, no `Spo` field. Owner: `deepnsm-v2` |
| **Kant** | **UNREACHABLE** ← *new, not in §12.3a* | needs `RungLift`, minted only inside `stance::stream()`'s complementizer window over **labelled raw verse text**; flat `(s,p,o,verse)` triples do not preserve clause nesting. Owner: `deepnsm-v2` (the consuming machinery is already here — the missing piece is the INPUT) |
| **Wittgenstein** | reachable, **REDUCED and DEGENERATE** | only 2 of the panel's 6 game categories survive (`Inh-subj`/`Inh-obj`; `rel-*`/`impl-*` need the same unreachable inputs as Kant), and the surviving bit fires on **99.61 %** of verses |

**Measured pair (the only one formable):** Hegel × Wittgenstein-reduced —
`n00=78 n01=19944 n10=0 n11=0`, N=20022, rates `0.0000 / 0.9961`,
`p_o=0.0039 p_e=0.0039`, κ=0.0000, φ=`undefined(constant)`. Both lenses
DEGENERATE ⟹ **0 eligible pairs** ⟹ both ∃-quantifiers are false *by
construction*, not by measurement.

**The degeneracy machinery earned its place.** Wittgenstein-reduced firing on
99.61 % of verses is the `closed_class_guess` 150/150 shape — a bit that carries
no information — and the harness **excluded and printed it** instead of
reporting a stance. Without §12.3a's `[0.01, 0.99]` band this run would have
produced a κ table that looked like a result.

**What D-BLW-2 actually needs, stated once:** the four stances require
`stance::stream()` over **labelled verse text**, which the TSV does not carry.
Either the inbound leg exports verse text alongside its triples, or the reasoning
layer receives verses directly. **That is a seam change in `deepnsm-v2`** — the
inbound leg owns text — and it is the one prerequisite for D-BLW-2, D-BLW-3
(whose verdict rows are these same binaries), and any four-stance claim at
corpus scale. **Do not attempt the twin again until it lands.**

#### 12.3c THE INSTRUMENT WAS WRONG — texture, not κ (operator-ruled 2026-08-04)

**κ over per-verse binaries is retired.** §12.3a's twin measures how often two
lenses *coincide*, which discards what a stance is. Two lenses can agree on a
verse for opposite reasons and κ scores that as agreement. The clean falsifier
of the whole approach: **nihilism and sarcasm are both negative** — any sign,
threshold, or boolean collapses them — yet Nietzsche's negation and
Schopenhauer's are different gestures (one revalues, one refuses). An
instrument that cannot separate those was never measuring four horizons.

The measured KILL in §12.3a″ stands as a fact about the SPO path; it was simply
obtained with the wrong instrument. **The 99.61 % firing rate was the tell** — a
bit that fires on nearly everything is not a degenerate *lens*, it is a wrong
*projection* of a lens.

**Root cause, and it is mine:** I chose per-verse binaries because binaries feed
κ, then measured the binaries. The instrument selected the representation
instead of the phenomenon selecting the instrument.

**The right carrier already exists and is already proven** —
`CausalWitnessFacet` (`lance-graph-contract/src/causal_witness.rs:201`),
`#[repr(transparent)]` over `[u8; 12]` = **24 × i4 loci, each a signed −8..+7
delta to an antecedent row**. Everything this arm needs is in that one register:

| locus | meaning | serves |
|---|---|---|
| 0–3 | `Temporal` / `Kausal` / `Modal` / `Lokal` | TEKAMOLO frame |
| 4–6 | `SMeaning` / `PMeaning` / `OMeaning` | SPO grounding plane |
| **7** | **`Antecedent`** — *"relativPronomen → its antecedent"* | the relative-pronoun binder |
| **8** | **`BasinAnchor`** — *"binds me to my AriGraph basin (`part_of:is_a`, L1)"* | AriGraph tenant + episodic basin |
| 9/10 | `SupportedBy` (hi_chain) / `Supports` (lo_chain) | evidence topology |
| 11 | `RunbookEvidence` | which of the 34 recipes fired |
| **12** | **`QualiaReference`** — *"the event that set my current texture"* | qualia |
| 13 | `MeaningLevel` | rung-content ladder 0–4 |

**Texture = binding topology, not polarity.** A stance's reading of a verse is
*which loci it binds, to what signed distance, in what pattern*. Nihilism and
sarcasm then separate structurally rather than by sign: a sarcastic reading binds
`QualiaReference` to a **distant** antecedent that contradicts the local
`SMeaning` (the said and the meant point apart); a nihilistic reading **collapses
`Supports`/`SupportedBy`** (nothing grounds anything) while leaving the local
meaning loci intact. Same sign, different graph. κ cannot see this; the register
carries it natively.

**Two falsifiers replace the twin, and neither is a threshold I choose.** BOTH
are runnable — see the ⊘⊘ correction on the first, whose data was in a Release
all along:

1. **Cross-language texture agreement — ⊘⊘ THE RETRACTION WAS ALSO WRONG. The
   data exists; I searched two places and called that "does not exist".**

   **Available and verified on disk** (`v0.1.0-codebooks-2026-07-26`, published
   2026-07-26 from a prior session of mine — the release body even cites its own
   board entry `E-CODEBOOK-LICENSE-REGIMES-ONE-ASSET-EACH-1`):

   | asset | contents |
   |---|---|
   | `pd-texts-bundle.tar.gz` | **4 PD source lanes verbatim** — `bible_luther1545.json` (9.1 MB), `bible_elberfelder1905.json` (9.3 MB, contemporary German), `bible_bkr.json` (10.3 MB, Czech), `bible_tischendorf.json` (2.3 MB, Greek) |
   | `rosetta-pd-bundle.tar.gz` | 3 non-English lane codebooks + **`versification_map.tsv` (3,568 rows: lane, book, chapter, offset, kjv_verse_count, lane_verse_count, confidence)** |
   | `rosetta-gpl-bundle.tar.gz` | `codebook_kjv.tsv` + alignments **en-de (13,016) / en-cs (12,032) / en-el (4,594)** |

   So the falsifier is **RUNNABLE across five lanes** (KJV + Luther1545 +
   Elberfelder1905 + BKR + Tischendorf), and the versification map is precisely
   the organ a per-verse cross-lane comparison needs — chapter-level offsets with
   a stated per-row `confidence`, so lane divergence is *addressable* rather than
   assumed away.

   > **⊘ CORRECTION to the sentence above (2026-08-04, same day, before any
   > cross-lane run — original retained per append-only canon).** The map IS the
   > right organ, but **`confidence` is not what I said it was**, and consuming
   > it as written would have produced a flag that reads as diligence and carries
   > no information.
   >
   > The generator's own report (`rosetta-pd/versification_report.md`, § Method)
   > states it: candidate offsets are only `(-1, 0, +1)`; score = fraction of KJV
   > anchor tokens (capitalized, non-sentence-initial, `len>=4`, stoplist-filtered)
   > + digit runs fuzzy-matching the shifted lane verse, falling back to a
   > verse-length ratio when a chapter has no anchor signal; and
   > **`confidence` = best-score − second-best-score.** It is a **margin between
   > candidate shifts**, *not* a measure of alignment quality. `0.0` means the
   > three shifts tied — which is the ordinary outcome for an anchor-poor chapter
   > (Genesis 1, "In the beginning God created…", carries almost no capitalized
   > non-sentence-initial anchors), not a defect in the alignment.
   >
   > Measured on the asset before writing this note: rows whose KJV and lane
   > verse counts match exactly have mean confidence **0.3036**; rows whose counts
   > *disagree* have mean **0.2783** — indistinguishable. **480 rows read exactly
   > `0.0` while their verse counts match perfectly.** The column does not
   > separate good alignment from bad, so it cannot serve as a trust gate.
   >
   > Gating on it would have flagged **214/1189** luther1545, **199/1189**
   > elberfelder1905 and **584/1189** bkr chapters as suspect — half the Czech
   > lane. That is the can-it-stay-silent defect from `CLAUDE.md`'s falsifiability
   > rule exactly: *a guard that fires on everything carries as much information
   > as one that never fires.*
   >
   > **What IS addressable, both mechanically checkable and both genuinely rare:**
   >
   > | signal | rows | meaning |
   > |---|---|---|
   > | `offset != 0` | **47 / 3,567** (43 are `+1`, 4 are `-1`) | apply the shift — this IS the alignment |
   > | `kjv_verse_count != lane_verse_count` | **6 / 3,567** | a verse with no counterpart — drop the pair or report it, never pad |
   >
   > So alignment is **identity for 98.7 % of chapters** and must be reported as
   > the near-trivial step it is, not implied to be hard. The offsets concentrate
   > where the manifest already says: luther1545 36 chapters, **33 of them in
   > Psalms** (the Psalm-title convention); elberfelder1905 3; bkr 8. The
   > manifest's own rule governs — **versification is PER EDITION, not per
   > tradition** — so "German shifts Psalms" must not be generalized from
   > luther1545 to elberfelder1905.
   >
   > If `confidence` is reported at all, it is labelled *offset-decision margin
   > (anchor-poor chapters read `0.0` by construction)* — never *alignment
   > confidence*.
   >
   > **Also load-bearing for any cross-lane arm:** `bible_tischendorf.json` is
   > **Greek NT only** (books 40+, and minified to a single line so `wc -l` reads
   > `0`). It has no Old Testament, so no OT-inclusive claim may pool the Greek
   > lane. Lanes are therefore English + German×2 + Czech, with Greek on the NT
   > half only.

   **Genuinely absent, and only these:** Latin Vulgate and Aramaic/Peshitta. Any
   claim naming those remains unavailable; the five lanes above do not.

   > **⊘⊘⊘ THIRD CORRECTION, SAME AXIS, SAME DAY (2026-08-04) — Vulgate and
   > Peshitta are NOT absent. They are Public Domain and now fetched.** The
   > sentence above is wrong for the third time in one arc, and the defect is
   > *identical each time*: **my negative existence claim was only as wide as the
   > one container I happened to look in.** First I claimed corpora without
   > checking; then I checked `/tmp` plus a 4-level `find` and declared them
   > nonexistent; then I checked the *release bundle* and declared these two
   > nonexistent. The bundle is a **licence-partitioned subset**, not a census —
   > its own MANIFEST says so ("one-asset-per-regime law", deliberately excluding
   > NC-licensed lanes). Absent-from-the-bundle never meant absent-from-the-source.
   >
   > Re-queried `api.getbible.net/v2/translations.json` (the same API the shipped
   > `fetch_greek_lane.py` uses) — 117 translations. Fetched 2026-08-04 with the
   > licence gate re-verified **verbatim at fetch time**, receipt at
   > `/tmp/lanes/pd-texts-v2/FETCH_RECEIPT.json` (per-lane sha256):
   >
   > | lane | language | books | verses | licence (verbatim) |
   > |---|---|---|---|---|
   > | `vulgate` (Vulgata Clementina) | Latin | 73 | 35,809 | `Public Domain` |
   > | `peshitta` (Peshitta NT) | Syriac | 27 | 7,956 | `Public Domain` |
   > | `aleppo` (Aleppo Codex) | Hebrew | 39 | 23,188 | `Public Domain` |
   > | `codex` (Westminster Leningrad) | Hebrew | 39 | 23,213 | `Public Domain` |
   >
   > **Refused on licence, and they stay refused** (the one-asset-per-regime law
   > binds this fetch too): `lxx` (*Copyrighted; Free non-commercial*),
   > `textusreceptus` and `westcotthort` (*CC BY-NC-SA 4.0*), `modernhebrew`
   > (empty licence field — unstated, therefore excluded). Note the cost of that
   > refusal honestly: **the LXX is the natural Greek lane for the Old Testament**,
   > so Genesis has no PD Greek lane and the OT Greek arm is licence-blocked, not
   > merely unbuilt.
   >
   > **The lane set is therefore 9 lanes / 7 languages** — English (KJV), German
   > ×2, Czech, Greek (NT), Latin (whole), Syriac (NT), Hebrew ×2 (OT) — which is
   > what the arm was told it had at the outset. Every "absent" claim I made was
   > a search-depth artifact.

### 12.6 Pre-registered anchors — what the texture instrument must reproduce

**Status: PRE-REGISTRATION. Nothing here is measured.** These are targets with
**externally known answers**, written down *before* the instrument exists,
precisely because I now know the answers and that is a contamination risk. Fixing
them in advance converts my knowledge from contamination into a **control**: the
instrument either reproduces a split stated here first, or it does not.

The subject is **awareness**, not morality — the Genesis 3 material read as
*blindness vs sight* and *nakedness as mortality-awareness*, with the temptation
resolving as **"be careful what you wish for"**: the burden delivered is
awareness of one's own finitude, and it is universal rather than penal.

#### A1 — the awareness minimal pair (within one language, one book)

| | KJV bake index | text |
|---|---|---|
| **before** | `55` (Gen 2:25) | *"And they were both naked, the man and his wife, and were not ashamed."* |
| **after** | `62` (Gen 3:7) | *"And the eyes of them both were opened, and they knew that they were naked…"* |

**The fact is identical in both — Hebrew `ערומים` / `עירמם`, Latin `nudus` /
`nudos`, German `nackend` in both.** What changes is *knowing* (`וידעו` /
`cognovissent` / `wurden gewahr`). Nothing in the world changed; **awareness
changed** — and the sight that opens delivers knowledge of a *lack*, which is the
blindness/sight inversion stated exactly.

**Why this is the sharpest control available:** a lexical or polarity instrument
sees "naked" in both and scores them *similar*. **If the texture instrument
cannot separate index 55 from index 62, it is not measuring awareness** — and
that is a KILL of the instrument, not of the reading.

#### A2 — "be careful what you wish for" (proposition held constant)

| | verse | content |
|---|---|---|
| **promise** | Gen 3:5 (serpent) | *"your eyes shall be opened, and ye shall be as gods, knowing good and evil"* |
| **confirmation** | Gen 3:22 (God) | *"the man is become as one of us, to know good and evil"* |

**The serpent's promise is confirmed by God. It was true.** Hebrew
`והייתם כאלהים ידעי טוב ורע` → `הן האדם היה כאחד ממנו לדעת טוב ורע`; Latin
`eritis sicut dii, scientes bonum et malum` → `quasi unus ex nobis factus est,
sciens bonum et malum`. **Any instrument that scores the serpent as a liar by
polarity is wrong on the text.**

Here **proposition, lexis and polarity are ALL held constant** and only the frame
differs (future/desired/tempter vs perfect/achieved/alarmed). So **only topology
can separate them** — this is the strongest form of the §12.3c sarcasm signature
(*the said and the meant point apart*) because the said is literally the same
sentence. Gen 3:22's alarm is about the tree of **life** (*"lest he… eat, and
live for ever"*) and Gen 3:19 states mortality as what he **already is**
(*"dust thou art"*), not as a new penalty — so the text itself locates the change
in **awareness of mortality**, not in mortality.

#### A3 — Erbsünde as a rebound relative pronoun (the cross-language falsifier)

Romans 5:12, final clause, **measured across six lanes on disk** (this table is
the one *observation* in §12.6; the prediction it grounds is A3′ below):

| lane | final clause | binding |
|---|---|---|
| Greek (Tischendorf) | `ἐφ’ ᾧ πάντες ἥμαρτον` | causal idiom |
| **Latin (Vulgate)** | **`in quo omnes peccaverunt`** | **relative → antecedent** |
| **Czech (BKR)** | **`v němž všickni zhřešili`** | **relative → antecedent** |
| German (Luther 1545) | `dieweil sie alle gesündiget haben` | causal |
| German (Elberfelder 1905) | `weil sie alle gesündigt haben` | causal |
| Syriac (Peshitta) | `ܒܗܝ ܕܟܠܗܘܢ ܚܛܘ` (*b-hāy d-*) | causal |
| English (KJV) | *"for that all have sinned"* | causal |

**The mechanism is sharper than "mistranslation".** Greek `ἐφ’ ᾧ` *does* contain
a relative pronoun (ᾧ), but as a fixed **conjunctional idiom** meaning *inasmuch
as* (cf. 2 Cor 5:4, Phil 3:12, Phil 4:10). The Vulgate rendered it
morpheme-for-morpheme, converting an idiom into a **referential** relative — and
thereby **opened an antecedent slot the Greek never had open**. Augustine bound
it to `unum hominem`. **The doctrine grew into a slot a translation opened.**

That is **locus 7 `Antecedent`** exactly — unbound in one lane, distance-bound in
another — which is why the arm's carrier is the register and not a polarity bit.

**A3′ — the pre-registered prediction (stated before any instrument runs):** a
texture instrument reading binding topology must report `Antecedent` **bound at
distance** for `vulgate` and `bkr`, and **unbound** for `tischendorf`,
`luther1545`, `elberfelder1905`, `peshitta`, `kjv`. Reproducing a 2-vs-5 split it
was not told about is evidence; producing any other partition is a KILL.

**The unpredicted datum is the Czech.** BKR (Bible kralická, 1579–93) is a
Protestant translation from the originals, yet `v němž` follows the **Vulgate's**
binding rather than the Greek's. I did not predict it and did not plant it — it
came out of the fetch. It is the reason A3 is worth running: the *interesting*
lanes are the ones that cross the confessional line, and no polarity instrument
could ever surface that.

**Honest status of A3, stated so it cannot be quietly upgraded.** The corpus is
in hand and the phenomenon is now *visible*, but **detection is not built**: this
repo has no morphological parser for Latin, Greek, Syriac or Hebrew, and
hand-writing a `in quo`/`v němž` matcher is precisely the hand-rolling this arm
was corrected away from. So A3 is a **falsifier waiting for an instrument**, not a
result — and its value is that its answer is *already known from philology*, so
it can grade an instrument rather than be graded by one.

#### A4 — the instrument frame, and what is NOT claimed

The connotative-meaning frame for these readings is the **semantic differential**
(Osgood) — bipolar scales, **multi-axis by construction**, which is the structural
reason a single κ destroyed the signal in §12.3a: collapsing a multi-axis
connotative space to one coincidence scalar discards every axis that separates
the stances. **No semantic-differential implementation exists in this repo** — a
sweep found the term only in one knowledge doc, and none in code. It is named
here as the frame the texture register is standing in for, **not** as a shipped
capability.

**Architectural anchors recorded for this arm (operator-directed, not yet
measured):** WordNet's hypernym hierarchy read **as** the HHTL cascade rather
than as a corpus indexed by it; **CLAM/CHAODA** as the clustered-hierarchical
anomaly arm over that cascade; and **HHTL + helix as torque** — HHTL supplying the
lever arm (tier depth) and the helix phase the angular displacement. That last one
is not decoration: it is the mechanical statement of the §12.3c distinction —
**sarcasm is torque** (a real lever arm displaced through a large angle: said and
meant point apart) while **nihilism is a collapsed lever arm** (`Supports` /
`SupportedBy` collapse, so no torque is possible at any angle). Same sign,
different mechanics — which is the whole reason polarity could never separate
them. All four remain CONJECTURE until a probe runs.

   **The defect, stated plainly because it is the fifth instance today.** I
   checked `/tmp` and ran a 4-level `find`, then wrote "it does not exist". A
   negative existence claim is only as wide as the search behind it, and mine was
   two places deep on a repo whose whole data convention is *code-in-repo,
   data-in-Releases* — documented in `crates/deepnsm-v2/data/README.md`, which I
   had already read this session to find the cam96 artifacts. **The right search
   was the one I had already performed once for a different asset.**

   Original (wrong) retraction text retained below per append-only canon. I wrote this falsifier claiming "the corpus exists
   in Greek (LXX), Latin (Vulgate), German (Luther), English (KJV), Czech and
   Aramaic" **without checking that it does.** It does not. The only Bible corpus
   present is `/tmp/pg10.txt` (English KJV, uncommitted). PROBE-BABEL-STANCES'
   "lanes" are **hand-authored `LaneLex` fixtures** — a handful of
   `surface`/`root`/`morph`/`prag` entries per lane in the probe's own source
   (`probe_babel_stances.rs:363+`) — **not corpora**. A texture comparison needs
   the same verse in each language; six lexical fixtures cannot supply it.
   **This falsifier is BLOCKED on data acquisition** (someone must supply the
   parallel texts) and must not be cited as available until it is. The reasoning
   below is retained because it is sound *once the texts exist*; only its
   availability was false. Original text follows.

   ~~The corpus exists in Greek (LXX), Latin (Vulgate), German (Luther), English
   (KJV), Czech and Aramaic.~~ A stance that is real should carry **related texture across lanes**;
   one that is an artifact of English tokenization will not survive translation.
   This is structural, not a cutoff I pick. PROBE-BABEL-STANCES already found the
   shape of the failure mode — the pragmatic channel reading as coherent
   antiphase across verified lanes, i.e. inherited calque rather than independent
   convergence — so that probe's CHECK-row discipline carries over verbatim: an
   unverified lane is **reported, never gating**.
2. **The horizon as a Pearl rung-3 intervention, not a κ delta.** §12.3b's
   fixed-verse-set control becomes: hold the verse set fixed, read it from
   horizon `Vk` and from `Vm > k`, and measure **which loci REBIND** — a change
   in binding topology under an intervention on the horizon. Fusion is loci
   rebinding, not a coefficient moving. The sample-growth confound §12.3b
   identified is still removed the same way (fixed unit set).

**Carried forward unchanged:** the claim ceiling (§12.4 — overlap/structure, never
validity; no p-values under domain correlation), the degeneracy discipline (a
texture that is identical on every verse is the 99.61 % defect in a new costume
and must be *excluded and printed*, never reported as a stance), and the
`crates/jc` additive constraint — `jc` is untouched, and it is simply not the
instrument here.

**Open, honest:** which language lanes have committed, loadable codebooks versus
which were CHECK-only in PROBE-BABEL-STANCES must be established **by reading the
data on disk**, not assumed — a lane that cannot be loaded cannot be claimed.

### 12.3b D-BLW-3 design — the confound, and the controlled comparison that removes it

**The naive trajectory does not measure fusion.** §12.3's D-BLW-3 row says
"fusion must MOVE: if pairwise κ between two lenses is flat across the series,
no horizons merged". True as a *kill* condition — but the converse does **not**
hold, and that is the trap. As the series seals, each `Vn` contains more verses
than `Vn-1`, so a κ computed at each version is computed on a **growing sample**.
κ will drift for that reason alone. **A κ that moves because N grew is not
Horizontverschmelzung; it is arithmetic.** Reporting a moving trajectory as
fusion would be the D-BLW-2 Kant tautology one level up — a number that cannot
help but move, presented as though it discovered something.

**The controlled comparison.** Hold the verse set **FIXED** at the first `k`
verses and compute the four per-verse binaries **twice**:

| reading | arena state used | verse set |
|---|---|---|
| **a priori** (*Vorurteil*) | as sealed at `Vk` — what a reader could know then | first `k` |
| **hindsight** (*wirkungsgeschichtlich*) | as sealed at `Vm`, `m > k` | first `k` — **the same verses** |

Same lenses, same units, same `N`, **same text** — the only thing that differs
is the horizon the reading is performed from. A κ difference between those two
readings cannot be a sample-growth artifact, because the sample is identical by
construction. That difference *is* the fusion signal: later knowledge re-reading
earlier material. This is also why the a-priori/hindsight split is not
decoration here — it is the control.

Mechanically the binaries must be **recomputed** against the later arena, not
carried forward; a verse's Hegel bit can flip when a statement it emitted is
contradicted a thousand verses later, and that flip is the whole phenomenon.

**Row shape (this is why D-BLW-3 was never blocked).** The harness emits one
lightweight **per-(verse, version, lens)** verdict row and implements the public
`DeinterlaceRow` trait on it (`temporal.rs:318` — `subject()` = the
book-qualified verse ref, `lance_version()` = the sealing version,
`knowable_from()` = the version the verse entered the corpus, `hlc_tick()`
defaulted). Both reads then come off the **real** surface:
`deinterlace(&rows, &QueryReference::at(V, rung), &NoDeps)` (`temporal.rs:346`,
`NoDeps` at `:271`). Nothing reconstructs an arena from a version; nothing in
`temporal.rs` is modified (§12.5 holds). Note `deinterlace` **clones** the
admitted rows — per §12.2's precision note, no result line may call this
zero-copy; the rows are small per-verse verdict records, which is why the cost
is acceptable, not absent.

**Pre-registered thresholds (fixed here, before any run, non-adjustable):**
- **fusion-moves (can-fire):** ∃ a lens pair and a fixed prefix `k` with
  `|κ_hindsight(k, m) − κ_apriori(k)| ≥ 0.10`, both κ defined (not `None`), and
  `k ≥ 1000` (the same corpus floor as §12.3a — below it the marginals are too
  noisy to read).
- **the distinction must earn its keep (can-stay-silent's twin):** if for EVERY
  pair and EVERY prefix the two readings differ by `< 0.01`, then the a-priori /
  hindsight distinction is **doing no work and must be DROPPED from the
  write-up rather than narrated** — §12.3's own instruction, made numeric.
- **Why these numbers, from already-pinned ones rather than freshly invented:**
  `0.10` is one-fifth of the `0.20 … 0.80` span between the two twin thresholds
  already pre-registered in §12.3a — a movement big enough to matter inside the
  band structure those thresholds define. `0.01` is the reporting precision floor
  (κ is printed to two decimals); a difference below it is not distinguishable
  from rounding.
- **KILL:** flat under the controlled comparison ⇒ the claim regrades to *"four
  independent stance reads over a shared corpus"* — still true, still useful,
  **not Gadamer**. Print the regrade; do not adjust the threshold.

**Reporting:** the same full-table discipline as §12.3a — per pair, per prefix,
both readings' counts, both marginals, `p_o`, `p_e`, κ, φ, and the signed
difference. Never a bare κ, never a bare difference.

**Claim ceiling, tightened for this deliverable.** The permitted statement is
that *the later horizon reads the same verses **differently***. It is **NOT**
permitted to say the later horizon reads them **better**, **more truly**, or
**more completely** — that is a validity claim, it needs an external criterion,
and it is D3b, which stays blocked (§12.4). "Fusion" here names a measured
change in overlap between two projections, nothing more.

### 12.4 Claim ceiling (carried from the D3a/D3b split — do not re-cross it)

κ and φ between two lens projections measure **overlap**, not validity. A
Horizontverschmelzung measured this way is a **reliability-class** statement:
*these two horizons agree more (or less) than chance, and that agreement moved.*
It is **not** evidence that the fusion produced *better* understanding — that is
D3b, and D3b stays blocked on an external criterion and a criterion-appropriate
held-out score. Reliability is not validity (plan C3); the #888 board correction
exists because this exact line was crossed once already.

**Significance:** `jc::stats` p-values are classical **independent-sample**
p-values. Verses within a book are domain-correlated, so they do **not** apply
unmodified here — any significance claim over this corpus needs its own
justified dependence model, named at the claim site (C4, as corrected).

### 12.5 What this arm must NOT do

- **No sixth SoA, no lens-owned mailbox, no stance node type.** A lens is a read.
- **No modification of `persist_sink.rs` or `temporal.rs`** (§7) — the range read
  uses the surface that already exists.
- **No new statistics.** `binary_association` / `cohen_kappa` / `phi` ship in
  `jc::stats`; the `jc` additive constraint continues to hold — those functions
  are the independent reference frame this arm is measured against, and are not
  to be "improved" while being used as the oracle.
- **No fusion or validity claim** before D3b (§12.4).

### 12.7 D-BLW-2 MEASURED RESULT — the texture rewrite is a KILL, on κ's own axis

**Status: MEASURED (2026-08-04).** `examples/blw_texture.rs`, 2,000-verse KJV
prefix, 1 s wall. Full 31,102-verse run exceeded a 10-minute budget — the harness
documents why in its own source (`stance::stream` calls
`staunen(Snapshot::of(arena, 0.0))` **once per rung lift**, each an
O(arena-size) scan, and the harness runs `stream` twice), so this is a known
superlinear cost, not a crash. **All numbers below are from the bounded run.**

**The verdict: the carrier changed and the instrument did not.** §12.3c retired κ
for collapsing a multi-axis phenomenon into one coincidence scalar. The
replacement uses a 24-locus register — and then **writes three loci**. Verified
against source, not the harness's self-report: all seven `.with(Locus::…)` call
sites write only `Antecedent` (5 sites, every stance), `Quorum` (Hegel only) and
`Modal` (Kant only). **Only `Antecedent` is shared between any two stances, so
`agreement_count` is bounded at 1 of 24 by construction** — a binary coincidence
measure rebuilt inside a richer type. The harness states the ceiling honestly and
in advance, which is to its credit; it is still the same defect one level down.

| pair | mean `agreement_count` | distribution |
|---|---|---|
| Hegel × Wittgenstein | 0.0825 | `{0: 1835, 1: 165}` |
| Hegel × Nietzsche | 0.0505 | `{0: 1899, 1: 101}` |
| Kant × Wittgenstein | 0.0105 | `{0: 1979, 1: 21}` |
| Nietzsche × Wittgenstein | 0.0100 | `{0: 1980, 1: 20}` |
| Hegel × Kant | 0.0070 | `{0: 1986, 1: 14}` |
| Nietzsche × Kant | 0.0015 | `{0: 1997, 1: 3}` |

Per-locus bind rate — **21 of 24 loci read exactly 0.0000 for every stance, by
construction**: Hegel `Antecedent .3650 / Quorum .3235`; Wittgenstein
`Antecedent .8815`; Nietzsche `Antecedent .0570`; Kant
`Antecedent .0265 / Modal .0355`.

**Second, independent defect — the four stances are not four comparable reads.**
Bind rates: Wittgenstein **1763/2000 = 88.2 %**, Hegel 732 (36.6 %), Nietzsche 114
(5.7 %), Kant 72 (3.6 %). One near-constant, one moderate, two near-silent. An
88 % firing rate is the same degenerate tell as the 99.61 % that killed §12.3a″ —
close enough to a constant that its "agreement" with anything is mostly its own
prevalence.

**What did work, and it is worth keeping.** The §12.3b fixed-verse-set control
behaved exactly as designed: holding verses 0..1000 constant and moving only the
horizon (`Vk`=1000 → `Vm`=2000) produced real rebinding — Wittgenstein 127/1000,
Hegel 113/1000 (`antecedent` 94, `quorum` 95), Nietzsche 48/1000, Kant 6/1000.
Sample growth is excluded by construction, so **this movement is not the artefact
§12.3b was built to exclude.** But it is almost entirely `Antecedent` rebinding —
one axis again.

**Consequence for the arm, stated as a KILL and not softened.** D-BLW-2's
instrument does not separate the stances by texture; it reports co-occurrence of a
single locus. **The register was necessary and is not sufficient — the binding
rules ARE the instrument.** A rewrite must populate the loci that carry the
distinction the arm exists to make (`Supports`/`SupportedBy` collapse for
nihilism; `QualiaReference` distance from `SMeaning`/`PMeaning`/`OMeaning` for
sarcasm — the torque-vs-collapsed-lever-arm pair in §12.6 A4), and must report
those as **two quantities, never averaged**.

**Two further defects in the shipped harness, both now false or wrong:**
1. It prints *"CROSS-LANGUAGE FALSIFIER: BLOCKED — no parallel-text corpus is on
   disk."* **False as of this session** — 9 PD lanes / 7 languages are on disk
   (§12.3c ⊘⊘⊘). The line must go.
2. It **bypasses the post-#879 substrate entirely.** Grep count for
   `batch_writer|BatchWriter|KanbanStep|kanban|owner_adapter|MailboxSoA|SoaEnvelope`
   in the file is **0**; its whole import surface is `causal_witness` +
   `nars::stance` + `BeliefArena`. So it is a free-standing loop over a TSV — no
   tenant, no verses-as-rows, no `KanbanStep` advance, no batch-writer casts —
   and therefore **cannot be evidence for any substrate claim**, only for the
   stance functions. **D-BLW-1 remains unbuilt**, and this harness standing in
   for it is precisely the substitution D-BLW-1 was scoped to prevent.

### 12.8 D-BLW-3 RESULT (2026-08-04) — measured, per the pre-registered rules

**BUILT + RUN GREEN** as `examples/blw_fusion.rs` (re-scoped per the design
note's B1: two rank projections A/B + inert control Z over the tenant's own
rows; stances are NOT inputs). All gates passed on the real corpus (2,000
verses, 8 sealed cycles, 27,000 verdict rows, incremental seating P1, rank
criterion P2 at q=0.25).

| pre-registered rule | measured |
|---|---|
| §3.1 band (0.20/0.80, reused Landis–Koch) | **IN/IN** — Strict κ=0.4933, Aware κ=0.4619, full BinaryAssociation tables printed |
| §3.3 movement at V_pin (≥0.10) | Δκ = −0.031 → **middle ground** (0.01 ≤ \|Δκ\| < 0.10): reported, **no fusion verdict claimed** |
| §5.3 drop (<0.01 everywhere) | **DROP does not fire** — max \|Δκ\| over the 8 horizons = 0.485 |
| C5 signed churn | one-directional at V_pin (A: +66/−0; B: +184/−0) — accumulation-shaped, printed, never averaged into Δκ |
| controls | Z byte-identical (plumbing zero); G1 three-way extensional identity (Aware≡Retro≡Strict@V8); G4 both tails + real-data silent arm (the design's "~90 % god" premise measured 0.1285 — fixture replaced with constant-by-construction tails); G5/G6/G7 green |

**The headline, exactly as large as the measurement:** the a-priori/hindsight
gap **moves toward zero overall, with a small rebound at V6/V7** —
*(⊘ 2026-08-05 regrade: measured under a GROWING reference pool — the
trajectory is a cohort-relative rank effect until the A/B/C decomposition
runs; see the E-entry's regrade and D-BLW-3b below. The numbers stand; the
fusion ATTRIBUTION is CONJECTURE.)* Δκ: −0.485 (V1),
−0.251, −0.079, −0.031, ≈0.000, +0.011, +0.017, 0.000 (V8, identical by
construction); Hamming(A): 152→123→94→66→53→37→21→0. The distinction does
real work early in the series and dissolves as horizons merge. That is a
trajectory-shaped observation over eight reported points — **no trend claim,
no fusion verdict at V_pin, no substrate-exercise claim** (under this corpus
`deinterlace` reduces to filter+sort; the finding lives in the rank
criterion; the permitted claim is first `DeinterlaceRow` implementor and
first `deinterlace` caller). §12.4's D3b validity gate stays closed.

### 12.9 D-BLW-5 PROPOSED (2026-08-04, operator) — the OBSERVER-EFFECT loop: the jc measurement fed back into awareness

> Operator framing, verbatim intent: *a scientific version of
> Horizontverschmelzung is the jc-crate loop — information about the
> correlation of a dataset, when fed into the awareness, influences the
> correlation. The observer effect.*

**Status: DESIGNED / CONJECTURE — queued behind PROBE-IGNITION. Nothing here
is measured.**

**What it adds over §12.8:** D-BLW-3 measured FIRST-ORDER fusion — horizons
merge by sharing data (pool growth; Δκ −0.485 → 0). This probe measures
SECOND-ORDER fusion — horizons merge by sharing the MEASUREMENT of each
other. The Click's own arrow is the hook: `awareness.revise(key, outcome)` →
`global_context += fact` → *reshapes NEXT cycle's F landscape*. Here the
injected fact IS a jc statistic about the cohort.

**The four-arm design (pre-registered SHAPE; numbers pinned at build time,
before any run):**

| arm | injection | pre-registered expectation |
|---|---|---|
| **T** (true) | `shape₀ × true rank₀` derived from sealed S₀ (§12.9a payload — never the raw statistic), injected as an ELEVATED-rung fact | the observable: S₁ − S₀. Fire iff it clears the floor. |
| **F+ / F−** (false) | the TRUE shape₀ with a FALSE rank: equal-magnitude opposite shifts applied in **logit(rank) space** (symmetric by construction, no boundary clipping); anchors whose true rank falls outside a pinned eligibility band [δ, 1−δ] are EXCLUDED from the F-arms, never clipped | the DIRECTION test: S₁ tracking the injected RANK = anchoring on testimony over evidence (Gadamer's prejudice-structure, measurable; Goodhart's shadow); S₁ correcting TOWARD truth against the injection = evidence-dominance; value-invariant movement = mere perturbation. |
| **P** (placebo) | same shape, permuted content, zero information | **must not move** — if placebo moves S₁, the instrument measures injection mechanics, not information, and the observer-effect claim dies. |
| **N** (null instrument, free) | the same T-injection against the §12.8 bloom-rank criterion | **must stay frozen BY CONSTRUCTION** — that criterion has no awareness term, so any movement there is a plumbing leak that voids the run (G2's pattern, one level up). |

**Mechanical prerequisite, stated honestly:** the §12.8 instrument CANNOT
exhibit the effect — popcount-rank has no awareness input, which is exactly
what makes it arm N. The observed reader must be awareness-coupled: the
belief-arena side (NARS revision — the injected statistic participates as a
belief and interacts via support/contradiction) or a MUL-qualia-coupled
criterion. Choosing which is the probe's first design decision.

**Rulings that bind:**
- `crates/jc` stays the ORACLE — it measures S₀ and S₁ and is never modified
  and never fed its own output as input (the edge is one-way; the LOOP runs
  through the system's awareness, not through jc).
- C6 anti-circularity is not violated — it is INSTRUMENTED: C6 forbids the
  witness gating the slice it was computed on because that is a self-proving
  loop; this probe deliberately closes that loop and MEASURES it instead of
  using it for admission. Nothing downstream may gate on S₁.
- The injected statistic is stored under the ELEVATED carve-out (statistic-
  as-witness, higher-rung derivation) and must be rung-marked so the reader
  knows it is meta, not corpus.
- C4: no p-values; the paired contrast + placebo + null-instrument arms ARE
  the inference. C2 naming; full tables, never bare κ.

**Kill conditions, pre-accepted:** placebo moves ⇒ instrument invalid
(reported, not tuned); T-arm silent at every floor ⇒ "awareness does not
reflect this statistic" is the finding — a true and useful null; F-arms
tracking injected values ⇒ the anchoring finding stands even if T is silent
(testimony-dominance is itself the discovery).

#### 12.9a Payload refinement (operator, 2026-08-04 same day — refines the arm table's injection column in place)

> Full doctrine (TFPN arms + Gadamer/Goodhart readings + falsification
> regimen): `.claude/knowledge/observer-effect-tfpn-doctrine.md`. This
> subsection is the plan-side delta.

1. **Not the correlation — distribution × Prozentrang.** The injected fact
   is never the raw association scalar (a scalar is trivially echoable, so
   the Goodhart/anchoring fixed point would be built into the instrument).
   The preserving payload is (a) the distribution SHAPE of the statistic
   over the *prior* pool — palette256/HDR-bucketed census via the
   Belichtungsmesser machinery (banded exposure + popcount-stacking early
   exit + CI thresholds + preheat/rolling-floor; anchors:
   `ndarray::hpc::cascade::{expose→Band, recalibrate}`,
   `ndarray::hpc::statistics::percentile`) and (b) the **Prozentrang** —
   the percentile rank of the observed association within that prior shape.
2. **The single-measurement law.** A measurement burns the state it
   measured: S₀ is one-shot at V₀, sealed. Post-injection the system that
   produced S₀ no longer exists — the instrument's next run is S₁ at V₁, a
   NEW one-shot of a DIFFERENT system, never a "remeasure". The only
   carry-forward from V₀ is shape₀ × rank₀, frozen.
3. **temporal.rs × sensor = the meta channel.** Hindsight blindness
   (Strict-rung version-gated reads, the D-BLW-3 `no_hindsight_*`
   precedent) × the shape sensor, riding as META only (ELEVATED
   rung-marked), never corpus, never recomputed-and-back-dated. This is
   what makes the probe viable *without* remeasurement.
4. **Arm-table deltas:** T injects shape₀ × true rank₀; F± inject the TRUE
   shape₀ with a FALSE rank (shifted high/low on the bounded rank axis —
   cleaner than fabricating a whole table); P's zero-information envelope
   choice is pinned at build time (note: uniform-shape + median-rank is NOT
   empty — it asserts "nothing unusual"); N unchanged + gains the
   pool-drift-baseline duty (its own V₀-vs-V₁ shape drift, awareness-free).
5. **New guard:** the remeasure guard — append-only measurement ledger
   keyed `(statistic-id, arm, cohort, metric, version)` (scope-qualified so
   independent arms/cohorts/metrics at the same version never collide);
   recompute at a sealed key ERRORS, with can-fire + can-stay-silent tests.

### 12.10 PROBE-ARC-TORQUE family PROPOSED (2026-08-04, operator) — torque of an arc, translator stray, author bias

**Status: PROPOSED / CONJECTURE throughout. Queued behind PROBE-IGNITION and
D-BLW-5. Nothing here is measured. Machinery anchors verified in source
where marked FINDING.**

The operator's underlying question, three stages of one instrument: can the
TORQUE of an arc be measured and embedded using HHTL (WordNet) × Helix
(Fisher-2z hydratable cosine replacement) — and can that instrument then
measure translation variance (where did the translator stray; what mindset
does a version carry vs the Greek/Aramaic sources) and author bias (does a
non-canonical book match any canonical author).

#### Stage A — the torque estimator (single corpus)

- **Torque MAGNITUDE is purely metric** [derivation, not yet run]: per-step
  torque about an anchor = `|r × F| = 2 × area of the triangle
  (anchor, p_k, p_{k+1})` — Heron's formula from THREE pairwise distances
  (anchor→p_k, p_k→p_{k+1}, anchor→p_{k+1}). HHTL path distance is 3
  tier-table lookups O(1), so per-step torque is O(1) table reads. Total
  |torque| of the arc = the area swept by the lever. The radial sign
  (approach vs recede) is also free from distances (c < a vs c > a).
- **CHIRALITY (clockwise/counter-clockwise) is NOT metric** — it needs a
  frame. The shipped carrier: `ndarray::hpc::splat3d::helix_orient` [FINDING
  — verified in source]: RVQ-on-the-sphere direction codes, decode
  **Fisher-2z normalized**, comparable in O(1) LUT without materializing the
  vector; measured 1–3 B at 4.87°/0.97°/0.073°, compare-without-
  materialization Pearson 0.9917 / Spearman 0.9924.
- **Embedding coordinate: Fisher 2z = ln((1+r)/(1−r)) = logit((1+r)/2)** —
  the variance-stabilized (Var ≈ const, independent of ρ), evidence-additive
  (log-odds) coordinate for cosine-valued quantities; hydratable back via
  `tanh(z)`. Equal-width palette256 buckets in 2z-space ≈ equal-information
  buckets, where raw-cosine buckets starve the tails (near ±1 — exactly
  where near-synonyms/antonyms live). This is what makes the cosine
  REPLACEMENT properly HDR.
- **Pre-registered falsifiers:**
  - F1 radial-vs-tangential: on WordNet, a hypernym chain THROUGH the anchor
    is radial ⇒ torque ≈ 0 (can-stay-silent); a co-hyponym walk at constant
    depth AROUND a common-hypernym anchor circulates ⇒ torque > pinned floor
    (can-fire). If the estimator does not separate these, it dies.
  - F2 clamp accounting: quantized distance tables can violate the triangle
    inequality ⇒ Heron's radicand can go negative. Clamp AND COUNT; a clamp
    rate above a pinned ceiling invalidates the estimator at that codec tier
    (feeds PROBE-CLAM-VS-HELIX-RESIDUE, task #66).
  - F3 additivity inertness: accumulate-in-2z vs accumulate-in-r must
    DIFFER on real arcs — else the Fisher machinery is decoration.
  - F4 hydration round-trip: z → tanh → r within helix residue precision.

#### Stage B — translation variance (the Erbsünde exemplar)

- Units = verse-aligned parallel versions (canonical alignment is the
  paired structure jc needs). Readers = per-version "arc passes near anchor
  C" through ONE shared multilingual space (BGE-M3/XLM-R lens covers
  English/German/Greek; **Koine-vs-modern-Greek drift is a caveat; Aramaic
  coverage thin — deferred**). Agreement per anchor = jc full tables (C2).
- **The floor is intra-language variance:** same-language translation pairs
  (e.g. multiple public-domain English versions) ARE the placebo arm — a
  cross-language deviation counts as a STRAY only where its Prozentrang
  against the intra-language deviation distribution clears a pre-pinned
  rank (§12.9a payload law applied verbatim).
- **Ground-truth falsifier with a known answer:** Romans 5:12 — the Vulgate
  "in quo" vs Greek "eph' hō" divergence, the historically documented stray
  that fed the Erbsünde doctrine. Pre-registered: the detector must rank
  that locus high between Greek-faithful and Vulgate-descended renderings
  AND stay silent between two Greek-faithful renderings. Note Erbsünde
  itself is an ANCHOR concept, not a token — the German text says Sünde;
  the doctrine name lives in confessional literature.
- **Translator MINDSET** = the systematic (non-zero-mean) component of the
  deviation field after floor subtraction — Gadamer's Vorurteil as a
  measured object. TFPN mapping: T = source-text arc; F± = the translations
  (historical injections whose direction is MEASURED, not fabricated); P =
  intra-language pairs; N = a lens-free co-occurrence criterion (the lens
  has its own training-distribution horizon; found strays must survive the
  lens-free null or they are lens artifacts).

#### Stage C — author bias + attribution (the hypothesis proof)

- Per-author systematic torque field over the undisputed corpus; floor =
  intra-author variance across that author's books.
- **The synoptic confound, handled:** literary dependence (Matthew/Luke
  copying Mark) makes shared TEXT look like shared MIND. Bias is therefore
  measured on the REDACTIONAL layer — the deviations from the shared
  source — not on the shared text (redaction criticism as measurement;
  the Stage-B stray logic reused unchanged).
- **Ground-truth falsifiers, all INSIDE the canon** (no external corpus
  needed to validate the instrument):
  - G1 Luke–Acts must MATCH (consensus single author) — can-stay-silent;
  - G2 Mark 16:9-20 (the long ending) must SEPARATE from Mark 1:1–16:8
    (consensus interpolation) — can-fire;
  - G3 the Pericope Adulterae (John 7:53–8:11) must SEPARATE from John
    (consensus interpolation) — can-fire;
  - G4 Revelation vs the fourth gospel must SEPARATE (the famous
    stylometric split) — else the instrument is blunter than classical
    stylometry;
  - G5 Hebrews must NOT match undisputed Paul (modern consensus vs
    patristic attribution).
- Only after G1–G5: non-canonical books nearest-author matched in bias
  space, reported as distribution shape × Prozentrang per candidate author
  — never a bare "matches X" scalar. Attribution outputs stay CONJECTURE;
  classical function-word stylometry (Mosteller–Wallace lineage) is the
  prior-art baseline the torque feature-space must beat or complement —
  both reported.

**Doctrine that binds all three stages:** §12.9a single-measurement law +
shape×rank payload; `observer-effect-tfpn-doctrine.md` falsification
regimen; jc one-way oracle; C4 no p-values; full tables always.

#### 12.10a The Rosetta architecture (operator, 2026-08-05) — universal meaning space, per-language torque, living×dead hydration

One pattern, two instantiations. A Rosetta Stone = a paired inscription of
the SAME meaning in two systems; the shared inscription calibrates the map
between them; the per-system residual IS that system's torque signature.

**R1 — language × language (the two Babel codebooks).** The two language
codebooks (PROBE-BABEL-STANCES slice 2, SHIPPED — the existing two-Rosetta-
stone precedent) span a universal meaning space in which each language's
ROUTE to a shared anchor is its torque. The operator's example pair: Czech
reaches dying through aspectual PREFIXES (stem + prefix = concept + a
morphologically compositional rotation), German through NOMINALIZATION
(der Tod — the same anchor rotated into a substantive). Per-language
torque field = the language's mindset — the SAME estimator as Stage B's
translator mindset and Stage C's author bias, one level up. Pre-registered
pair for the can-fire/can-stay-silent twins: a known morphologically
divergent anchor (aspectual-verb family vs nominalization) must show
distinct signatures; a structurally parallel cognate pair must not.

**R2 — living × dead (Jina hydration over the WordNet spine).** WordNet is
the DEAD SPINE: static taxonomy = HHTL addresses + lever arms, CLAM
neighborhoods (`ndarray::hpc::clam` build/`rho_nn`) + CHAODA anomaly
detection (clam.rs Phase 4) — structure without life: no frequencies, no
coverage of KJV archaisms/names, no meaning axes beyond taxonomy. Jina is
the HYDRATION (API key present in env — verified, presence only): for the
alignment set (tokens BOTH systems know), fit the projection ONCE and seal
it (single-measurement law: the alignment is version-stamped; hydrations
are stamped against that alignment version); for Bible-specific tokens the
spine lacks, hydrate THROUGH the sealed projection:
  1. **frequency** — gated against the in-tree COCA 20k ground truth
     (`ndarray/src/hpc/jina/weights/coca_academic_20k.csv`:
     `word, PoS, COCA_All, …`);
  2. **POS** — same gate;
  3. **orthogonal meaning** — the component of the Jina embedding in the
     orthogonal complement of the WordNet-explained subspace: the axes the
     taxonomy structurally cannot express (register, affect, era).

**Hydration falsifiers (pre-registered):**
- H1 hydration gate: Jina-derived frequency/POS on a HELD-OUT slice of the
  COCA overlap must clear a pinned rank-correlation floor BEFORE any
  Bible-tail hydration is trusted. Fail ⇒ the living source is not
  admissible on this spine.
- H2 CHAODA quarantine: a hydrated Bible-specific token that lands as a
  manifold outlier under CHAODA is QUARANTINED, not silently projected —
  the sealed projection is valid only on the manifold the Rosetta
  calibrated.
- H3 the R1 torque twins (above).

**What this buys the staged probes:** Stage A gets lever arms for tokens
WordNet cannot address (the KJV tail); Stage B gets the universal anchor
space in which translator torque is per-VERSION while language torque is
per-LANGUAGE — separable because R1 measures the language signature on
non-biblical text, so Stage B can subtract it; Stage C inherits both.

#### 12.10b Jina → helix: the cosine>helix transcode, three routes (operator, 2026-08-05)

The direction arrow is the design: **cosine is measured once and DEMOTED
into a hydratable code; helix codes are the runtime carrier.** Jina is
called at bake/seal time only (membrane — the API key never enters the hot
path, per compilation-vs-runtime doctrine); everything downstream runs on
codes. Which machinery applies depends on WHAT is encoded:

1. **Pairwise cosines → the palette256 cosine replacement, buckets in
   Fisher-2z space, back-hydratable** (the direct `cosine>helix`; operator-
   confirmed naming 2026-08-05: "palette256, Fisher-z back hydratable —
   the cosine replacement"). The torque estimator consumes specific
   pairwise cosines, not embeddings. Compute each needed cosine ONCE from
   Jina vectors at the sealed alignment version, transform to Fisher 2z,
   encode place/residue (1-byte palette256 bucket in 2z + optional residue
   byte — equal-information buckets). All later comparison/accumulation is
   LUT + integer adds; the vector is never materialized again; hydrate
   back via tanh only at the boundary that needs a float.
   `helix_orient`'s own doc calls itself "the same RVQ machinery as
   palette256, on S² instead of the line" — this route is the line
   version. Single-measurement law applies verbatim. Synergy: a 256×256
   table over the 2z-palette codes gives O(1) pairwise compose/distance on
   coded cosines — the stack's recurring structure (bgz17 palette
   distance/compose tables, helix DistanceLut, attention-as-lookup).
2. **Whole vectors → Cam96 preferred, Base17-palette as the coarse tier,
   NEVER helix_orient** (the category-error guard). `helix_orient`'s
   codebook is the golden-spiral template on S² (2 DOF); a 1024-dim Jina
   vector cannot enter it. Per-vector compression is ALREADY SHIPPED for
   Jina twice, at two precision tiers: the Jina-trained **Cam96** Bible
   codebook (12-axis, 96-bit code — operator 2026-08-05: "probably more
   exact", plausible on axes×bits grounds: 12 subspaces vs 1 palette
   index) and `ndarray::hpc::jina::codec`'s F16 2048D → Base17 (34 B) →
   Palette (1 B, O(1) `JinaPalette::distance`). "Probably" stays a
   HYPOTHESIS until measured: the rank-preservation gate below runs BOTH
   tiers against f32 cosines on the same held-out set and reports both ρ
   per byte spent — the tier choice is then a read from the table, not a
   guess.
3. **Per-step plane angles → helix Signed360** (the chirality carrier
   Stage A needs). Each arc step spans a 2-plane (lever × step,
   Gram-Schmidt from Jina vectors); the signed angular increment in that
   plane is a scalar angle → Signed360/residue, 1–3 B per step. Valid at
   ANY ambient dimension because the plane is always 2D.

**Pre-registered gate:** rank preservation — helix-2z-coded cosines must
preserve pair ranking vs f32 cosines above a pinned ρ floor on a held-out
set. The S² precedent measured Pearson 0.9917 / Spearman 0.9924
(helix_orient header); the LINE version must be re-anchored, never assumed
from the sphere's numbers.

> **DEFERRED pointer (operator, 2026-08-05):** `ogar-blockly` (OGAR main —
> the 256-slot `(function:value)` call palette, LaneShape carvings, one
> 512-byte node per function body) as the storage substrate for the
> elixir-syntax thinking-template recipes (256:256 rails; the alternative
> to the planner/JITson route) is FEASIBLE-ASSESSED but the crate is ~3–5
> days from finished. Full plan entry lands when it does. Mandatory read
> before that design starts: `.claude/v3/knowledge/persona-vs-rung-ladder.md`
> (the recipe codebook binds to the 144 verbs + 34 tactics, never the
> adjective-36). Open encoding question carried: StepMask vs the 180-call
> Pairs cap.

### 12.11 D-IGN-B PROPOSED (2026-08-05, operator) — ignition starts the REAL lenses

**The directive:** ignition must be a *simple start* for Gadamer
Horizontverschmelzung or the four lenses — not an abstract style bit over
fixture bodies. Sequencing: PROBE-IGNITION (in build) proves the MECHANICS
(cast → scan → seal → advance, fixture qualia, §5 "no semantic claim");
D-IGN-B is the stage behind it that swaps the fixture thought body for the
shipped instruments, reusing the probe's fleet/scan/loop scaffolding.

**Arming vocabulary (probe-defined, sidesteps Q1 entirely):**
`z ∈ {0 = unarmed, 1 = Hegel, 2 = Nietzsche, 3 = Kant, 4 = Wittgenstein,
5 = Fusion}` — six ordinals fit the 6-bit `MetaWord::thinking()` field with
no MetaWord→PlanContext bridge needed: the CognitiveWork dispatch reads
`thinking()` directly and selects the lens body. The 36-style bridge stays
an explicitly open non-goal (design-note Q1; persona-vs-rung-ladder is the
mandatory read before any such bridge).

**The thought bodies — all shipped, nothing invented:**
- z=1..4: the shared four-stance machinery (`lance-graph-planner/src/nars/*`
  — the stance streaming + readout records extracted for
  `probe_eyes_opened`), run over the owner's POPULATED rows through
  cycle_driver's pluggable thought seam (D-BLW-1 precedent: the lens body
  is already wired into the 5.4 seam).
- z=5: the D-BLW-3 two-projection read (Strict rung-0 vs Aware rung-5 at
  the owner's sealed horizon — `blw_fusion.rs`'s proven machinery); the
  observable is that owner's gap read.

**Pre-registered observable SHAPE (numbers pinned at build time):**
- can-fire: two cohorts armed with DIFFERENT lenses over byte-identical
  rows produce NON-identical readouts — the lens axis is load-bearing
  (G2c's pattern lifted from reliability bits to instrument readouts);
- can-stay-silent: two cohorts armed with the SAME lens over byte-identical
  rows produce bit-identical readouts; the unarmed cohort produces none;
- the ignition property itself is inherited, not re-proven: arming is a
  write, discovery is a scan, work happens only after seal→apply — the
  mechanics probe's G1–G11 already gate that layer.

**Placement:** `lance-graph-supervisor/tests/` (same forced placement:
only the supervisor sees both `run_cycle` and the planner). **Build gate:**
starts only after PROBE-IGNITION passes central gates; shares its corpus
loader and MemWal provenance.

**Not claimed:** no fusion verdict, no stance-validity claim, no
parallelism, no durability — this stage adds exactly one fact to the
tree: *a cast-and-scan ignition starts real, shipped cognition, and which
cognition is selected by the armed bits alone.*
