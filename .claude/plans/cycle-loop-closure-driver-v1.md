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
