# Orchestration boundary — primitives stand, the loop's seams are unbuilt (v1, 2026-06-02)

## READ BY: truth-architect, integration-lead, anyone touching the cognitive/orchestration loop

> **STATUS: VERIFIED by a full-surface council (5-research + 3-brutal), 2026-06-02.**
> Every attribution below is grounded in file:line across **surreal + ractor + Lance + the
> contract** — not asserted from lance-graph alone (that scope error caused the `d5f5aa6`
> duplication this doc was born to prevent). The surreal repo double-nests:
> real paths are `surrealdb/surrealdb/core/...`.

## Thesis (corrected — it cuts against BOTH extremes)

The orchestration loop — *resolved understanding → committed action* (version → trigger → kanban →
actor) — is built from **real, standing infra PRIMITIVES**: the Lance store + native versioning,
SurrealDB LIVE → `Notification`, the ractor actor framework, and the contract's kanban DAG +
the `route_against` resolver atom. **lance-graph must rebuild none of these primitives.**

**But the loop is NOT wired end-to-end.** The integrating SEAMS — `Notification → KanbanMove`, a real
`MailboxSoaOwner` impl, surreal's *consumption* of the contract, the kv-lance "transparent view" — are
**unbuilt/scaffold**. lance-graph's own `EPIPHANIES.md:225` already names this: **"the SHOCK… the
integrating seam was never built."**

So the boundary is **not "build vs don't-build."** It is **primitive-vs-seam** and **spec-vs-wiring**:

| | the prior-council error (→ `d5f5aa6`) | the v0 doc's overclaim | the evidence |
|---|---|---|---|
| framing | "the wire is absent *here* → **build it in lance-graph**" | "the loop is **infra already standing**, builds none" | **primitives live; loop-seams unbuilt** ("the SHOCK") |

The `d5f5aa6` revert was still right (`rubicon_transition` was a *new standalone fn* in the wrong crate,
duplicating surreal's trigger concept). The opposite — "it's all done, do nothing" — is **also** wrong:
real seams are unbuilt. They are just **not lance-graph-crate-internal logic to invent**; they are
trait-impls + an adapter (below), and **building them is out of scope for this doc** — document the
boundary; wiring is a separate, explicit decision.

## The boundary (owners) — verified file:line + honest live/scaffold

| layer | owner | grounding (verified) | live? |
|---|---|---|---|
| store + native MVCC versioning | **Lance** via surreal `kvs/lance` | `surrealdb/surrealdb/core/src/kvs/lance/` (12 files, ~6.2K LOC); `impl Transactable for Transaction` (`mod.rs:532`); version via `Dataset::checkout_version` (`mod.rs:688`); merged PR #29; 59 unit + 8 integ tests | **LIVE** |
| trigger (record-write event) | **SurrealDB LIVE** | generic `Notification{action,record,result}` (`types/src/notification.rs:58`); fired inline from all 8 mutation paths via `process_table_lives` (`core/src/doc/lives.rs:29→:317`); tested (`core/tests/live.rs`) | **LIVE** |
| kanban: types + transition DAG | **`lance-graph-contract`** (NOT surreal) | `KanbanColumn` + Rubicon DAG `next_phases`/`can_transition_to` (`kanban.rs:32,90,102`); `KanbanMove` ≤16 B (`kanban.rs:112,177`); `VersionScheduler::on_version` (`scheduler.rs:46`); `MailboxSoaView`/`MailboxSoaOwner::try_advance_phase` (`soa_view.rs:28,118`); 13 own tests | **LIVE in lance-graph** |
| actor + mailbox + lifecycle | **ractor** (upstream framework) | `Actor` trait + `pre_start`/`post_start`/`post_stop`/`handle` (`ractor/src/actor.rs:124…`); loop `processing_loop`/`process_message` (`actor.rs:792,858`); 4-port mpsc mailbox (`actor_properties.rs:81`); FSM `ActorStatus{Unstarted..Stopped}` (`actor_cell.rs:40`); **0 cognitive/lance coupling** in either ractor repo | **LIVE** (framework only — NOT the loop's driver; see § What DRIVES the loop) |
| the NARS resolver | **lance-graph** (`route_against`) | `causal-edge/src/syllogism.rs:254`; `DominoStep{Settle,Fork,Escalate,Terminal}` (`:287`); thresholds `0.25`/`0.60`/`0.50` (`:304`); first live caller of `figure`/`syllogize`; **UNIQUE** (≠ `thinking-engine/domino.rs` cascade, ≠ the `layered.rs:46` edge-type shadow) | **ATOM** — 6 router tests, **0 production callers** |

### The seams between those owners — UNBUILT (this is "the SHOCK")

| seam | what it would connect | whose | state |
|---|---|---|---|
| `Notification → KanbanMove` adapter | trigger → kanban | contract-side or a consumer crate (NOT surreal — it stays generic) | **absent** (`EPIPHANIES.md:225`) |
| surreal *consumes* the contract | kanban spec ↔ the trigger | surreal | **scaffold**: declares dep + `pub use … as lance_graph` (`core/src/lib.rs:105`), **0 import sites**, feature off by default (`core/Cargo.toml:18`) |
| real `MailboxSoaOwner` on `MailboxSoa` | scheduler proposal → `try_advance_phase` | lance-graph (own type + own contract trait) | **BUILT in-RAM (2026-06-02)**: `impl MailboxSoaView + MailboxSoaOwner for MailboxSoA<N>`; the in-process loop runs (`NextPhaseScheduler::on_version` → `try_advance_phase`, driving test green). Remaining: a resolver-aware scheduler policy (route_against picks the Evaluation fork) + the surreal *external* trigger (fork-blocked) |
| `surreal_container : MailboxSoaView` (the kv-lance "view") | store → mailbox view | surreal-consumer | **BLOCKED(C)** — kv-lance fork dep not wired (`lib.rs:124`) |

> **Caveat the table must carry:** a ractor actor crate already lives *inside* this workspace —
> `crates/lance-graph-supervisor/` (`use ractor::{Actor,…}`, feature `supervisor`). It is the
> callcenter fan-out supervisor (does **not** call `route_against`/`advance_phase`), so "the actor
> layer is purely external ractor" is false: the *integration* is vendored here, the *framework* is
> upstream ractor.

> **Orphaned (2026-06-02):** jan reports the parallel surreal session that owned the INBOUND seams
> (per § Shared door of `le-domino-cognition-v1.md`, "Producer B") is **dead — the wiring was never
> built.** So the surreal-side seam-rows above (contract consumption, the `Notification→KanbanMove`
> adapter, the `surreal_container` view) are **orphaned, not in-flight**: no owner, OQ-11.6 unresolved.
> Now a workspace decision (adopt the surreal-side wiring here / leave documented-as-orphan /
> reassign) — tracked in `ISSUES.md` `ORCH-SURREAL-INBOUND-ORPHANED`. lance-graph's own half is now
> **BUILT** (2026-06-02, jan authorized "adopt"): `MailboxSoA` implements `MailboxSoaView +
> MailboxSoaOwner`, and the loop runs **in-RAM** via `NextPhaseScheduler` (in-process version tick, no
> surreal needed) — the in-process driver IS the consumer, so this is *not* the consumer-less
> `d5f5aa6` trap. **Only the surreal EXTERNAL trigger remains orphaned/fork-blocked** (`Notification →
> on_version` + the `surreal_container` view, OQ-11.6). The vertical probe (`a26c58b`) + the in-RAM
> driving test hold the path validated.

## What DRIVES the loop — the "surrealdb hot route," not a tokio/ractor message loop

The cognitive phase-transition driver is **substrate-native**, not a dedicated actor message route.
A `SurrealQL→ractor` message route was evaluated and is **not built** (grep: the only `SurrealQl` is
`ExecTarget::SurrealQl`, a routing *tag* on `KanbanMove` — `kanban.rs:143`, `scheduler.rs:203`). The
decision is already recorded in `EPIPHANIES.md` (`E-VERSION-ARC-IS-THE-KANBAN`,
`E-RACTOR-WANTS-TOKIO-NOT-GRPC`, the INBOUND-scheduler finding):

- **OUTBOUND (free):** `MailboxSoaOwner::advance_phase` commit = one Lance version = one `KanbanMove`.
  No separate kanban mechanism — the `Dataset::versions()` arc IS the kanban.
- **INBOUND (the driver):** surreal's LIVE/scheduled query over that version arc fires the next
  `try_advance_phase` tick (push, not poll — the GitHub-CI-subscription homology). **The mailbox does
  not run its own tick loop; surreal = clock + planner-dispatch.** The planner emits `KanbanMove`s; the
  mailbox is a pure state machine. `ExecTarget::SurrealQl` makes it literal: a scheduled SurrealQL query
  is BOTH the trigger AND a valid execution backend.
- **Rejected as more expensive:** a dedicated in-process **tokio/ractor message loop** to drive each
  transition (the D-MBX-8 "Σ10-commit→ractor-START" in-process planner loop). The version-arc route
  gets the same coordination for free off the commit + LIVE subscription that already fire — so no
  per-transition dispatch infra is built.
- **ractor-on-tokio is NOT thereby abandoned:** local ractor is a `Box<dyn Any>` pointer-move over
  Tokio mpsc (zero-serialize — the *cheap* in-process transport; gRPC is the expensive lab-only one).
  It stays the right tool where genuine in-process message-passing happens (`lance-graph-supervisor` /
  osint / ontology already use it) — it is just **not the phase-transition driver.**

Consequence for the seam-rows: the version→tick INBOUND wiring is **surreal-side** (gated on the
`surreal_container` fork, OQ-11.6 / BLOCKED(C)); the `MailboxSoaOwner` impl the tick advances is
lance-graph's own-type/own-trait wiring. **Neither is a tokio/ractor message route** — so R3's "the
seam = an `impl Actor` holding `MailboxSoA` as State" overstates ractor's role in the *driving* loop.

## What the loop MEANS — FSM → Rubicon → free-will gate (MUL)

The FSM is the *mechanism*; the **Rubicon** is *which* FSM; the **MUL** (Meta-Uncertainty Layer) is what
gates its one irreversible edge. The whole vertical is already typed — not metaphor.

- **Rubicon (Heckhausen).** `KanbanColumn{Planning → CognitiveWork → Evaluation → {Commit | Plan | Prune}}`
  IS the Rubicon model of action phases: pre-decisional deliberation (Planning), the crossing
  (Evaluation), post-decisional volition (Commit) — with the Libet **−550 ms** anchor stamped on the
  move (`KanbanMove.libet_offset_us`). `Evaluation.next_phases() = [Commit, Plan, Prune]` is the fork;
  `Commit`/`Prune` absorb, `Plan` re-deliberates.
- **Free will = the veto at the crossing, not the initiation.** The object-level NARS resolver
  (`route_against → DominoStep`) already *pushed* toward an action — it fires from the (f,c) diff before
  the meta-layer is consulted (the readiness potential). Free will enters as the MUL's power to
  **override** that push: `MulAssessment.free_will_modifier: f64` (`mul.rs:59` — "0.0 = fully
  constrained, 1.0 = fully autonomous") + `is_unskilled_overconfident()` (`mul.rs:384` — "used by the
  gate as a **veto hint**"). That veto is Libet's "free won't" — the Evaluation→Prune edge.
- **The gate arbitrates TWO confidences (the crux).**
  - *Object-level — NARS reasoning confidence:* the conclusion's **(f, c)**. *Am I confident IN this
    inference?* `route_against` routes on the pairwise (f,c) diff.
  - *Meta-level — Dunning-Kruger self-competence:* `DkPosition{MountStupid, ValleyOfDespair,
    SlopeOfEnlightenment, Plateau}` from `felt_competence` vs `demonstrated_competence` (`mul.rs:403`).
    *Am I confident in my COMPETENCE — and is that calibrated?*
  - *Trust = the calibration between them:* `TrustTexture{Calibrated, Overconfident, Uncertain,
    Underconfident}` (the gap felt − demonstrated; Overconfident = felt ≫ demonstrated).
  The gate (`MulProvider::gate_check → GateDecision{Flow, Hold, Block}`) fires the Rubicon transition
  from BOTH: `Flow → Commit` (cross — confident AND calibrated), `Hold → Plan` (re-deliberate —
  reduced autonomy), `Block → Prune` (veto — `Uncertain` trust, or `MountStupid`/`Overconfident`).
  The decisive case: **high NARS-c but DK-overconfident-and-miscalibrated → NOT Commit.** Acting on
  confident-but-incompetent reasoning is exactly what the veto blocks.
- **The humility chain makes it φ-bounded.** `free_will_modifier = dk_factor × trust × complexity × flow`
  (`mul.rs:343`), `dk_factor` = 0.3 (MountStupid) … 1.0 (Plateau). Full autonomy (1.0) is reachable
  *only* by a calibrated expert; overconfidence multiplicatively shrinks it — the CLAUDE.md "φ-1 =
  permanent humility" ceiling, so the veto is always reachable. `MulThresholdProfile` tightens the gate
  by context (`MEDICAL` trust ≥ 0.85 vs `CALLCENTER` ≥ 0.55) — same gate, stricter where stakes are.

**So determinism and freedom coexist.** The FSM is deterministic (replay = audit = the moat); free will
is the deterministic-but-meta **veto**: the loop can always decline to cross its own Rubicon when it
detects, about itself, that it doesn't actually know. Object-level says "go"; the meta-level keeps the
right to say "not yet."

## The two "elegance" claims — corrected (both were wishful)

1. **"Transparent view = SurrealDB *is* Lance, no transcode" — FALSE on no-transcode.** Values are
   `revision::to_vec`-encoded to **opaque** `key:Binary/val:Binary` blobs (`kvs/tx.rs:475` →
   `kvs/key.rs:77`; the schema is 4 generic columns `key,val,version,tombstone`, `schema.rs:50`). A
   SurrealDB record is **not** a Lance row; Lance is one of **six** peer `Transactable` blob-KV backends
   (rocksdb/tikv/surrealkv/lance/mem/indxdb), all fed the same encoded bytes. Typed/columnar access to
   record *fields* is the **unbuilt** "typed columns" extension (`schema.rs:25`, "not part of the POC").
   *True* part: one physical store, no separate cold tier, no CDC/reconciler.
2. **"LIVE fires on the Lance version change → the kanban update" — FALSE as written.** LIVE fires on
   the **surreal record write** (`lives.rs:29`, key = `self.changed()`), with **no** Lance-version→LIVE
   link in code (the Lance `Timeline` is a separate **read-only** surface, unused until a consumer,
   `kvs/lance/timeline.rs`). And no "kanban update" happens — surreal emits a generic `Notification`;
   the `Notification→KanbanMove` adapter is the unbuilt seam above. **What is genuinely free: versioning**
   (Timeline over `Dataset::versions()`), not the kanban trigger.

## The lesson (why this doc exists) — and its mirror

This session built — then reverted (`d5f5aa6`) — a `rubicon_transition` kanban-wire + a `MailboxSoA`
actor-step. Both **duplicated infra**, and the cause was a 5+3 *build*-council **scoped to lance-graph's
crates only** → it found "the wire is absent *here*" and mis-read that as "build it here."

> **RULE 1:** verify the FULL surface (surreal + ractor + Lance) before scoping a build.
> **"Absent in crate X" ≠ "build it in crate X."** The thing may already live next door.
>
> **RULE 2 (the mirror, learned here):** "the primitive exists next door" ≠ "the loop is wired."
> Don't swing from false *absence* to false *completeness*. Mark **live vs scaffold** honestly, or the
> next session builds on a loop that doesn't run (cf. `EPIPHANIES.md:225`, "the SHOCK").

## Open questions — now answered by the council

1. **Each owner verified?** Yes — table above, with file:line + live/scaffold. *Live:* Lance store +
   versioning, surreal LIVE/Notification, ractor framework, contract kanban DAG, `route_against` (as an
   atom). *Scaffold/absent:* surreal's contract consumption, the `Notification→KanbanMove` adapter, a
   real `MailboxSoaOwner` impl, the kv-lance "view," and any Lance-version→LIVE link.
2. **Does lance-graph own anything beyond contract + `route_against`?** Split by *role*. **(a) In the
   loop:** `route_against` (resolver atom) **+ the loop's *contract surface*** — which is more than
   passive types: it carries the Rubicon transition DAG (`can_transition_to`/`next_phases`) and the
   view/owner/scheduler airgap traits (the loop's *spec*; implementations deliberately elsewhere). Plus
   it **hosts** `lance-graph-supervisor` (vendored ractor integration). **(b) Separately:** lance-graph
   remains the query/codec/semantic spine (Cypher/datafusion/SPO/AriGraph, planner,
   bgz17/deepnsm/bgz-tensor) — orthogonal to loop control-flow. lance-graph needs to **build no new
   store/trigger/actor infra**; to make the loop *run* it would only **wire** existing traits (a real
   `MailboxSoaOwner` on `MailboxSoa`; unblock the kv-lance view) — a separate, explicit decision, not
   this doc's action.
3. **Is the loop fully assembled?** **No.** Primitives stand; the integrating seams are unbuilt (the
   seam-rows above). The gaps are surreal-side (consume the contract + wire `VersionScheduler` to LIVE)
   and a consumer/lance-graph-side trait-impl — **confirmed not a rebuild of any primitive.**

## Council provenance

Full-surface verification, 5 read-only Opus agents, one per layer against the owning repo:
**R1** Lance store + transparent-view (REFUTED "no transcode"); **R2** LIVE→kanban trigger (REFUTED the
trigger/kanban conflation + "fires on Lance version"); **R3** ractor (VERIFIED, 0 coupling); **R4**
contract SOT + surreal consumption (PARTIAL — owned, not consumed); **R5** lance-graph's role +
`route_against` (VERIFIED unique atom; caught the kanban-DAG-is-the-contract's correction +
`lance-graph-supervisor`). Then a 3× brutal pass: opposite-direction false confidence · wishful
elegance claims · monument/EPIPHANIES-rederivation. Naming fix surfaced: the scheduler trait is
`VersionScheduler` / `NextPhaseScheduler`, not `Scheduler` (correct in `le-domino-cognition-v1.md`).
