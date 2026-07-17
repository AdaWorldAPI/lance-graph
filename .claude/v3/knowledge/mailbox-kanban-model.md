# Mailbox-Kanban Model — executors, ahead updates, delegation cache

> READ BY: kanban-executor-engineer, mailbox-warden, integration-lead, and
> any session touching lance-graph-planner strategies, lance-graph-supervisor,
> symbiont, or a batch-writer path.

## Status: FINDING (operator-ruled 2026-07-02 — E-MAILBOX-KANBAN-NO-COLLAPSEGATE)

---

## The unit: one Mailbox = one Kanban board (as a TENANT)

Every mailbox carries its own Kanban board. The board is a **tenant** —
per-mailbox state, a sibling of the per-row `KanbanTenant` — never a global
singleton. Phases (Tier 2 of `docs/architecture/soa-three-tier-model.md`):

```
Planning → CognitiveWork → Evaluation → Commit → Plan → Prune
```

`MailboxSoaOwner::advance_phase(to)` is the SOLE mutator. Everything else
proposes; the owner disposes.

## The two executor arms + the structural owner

| Arm | Where | Role |
|---|---|---|
| **Arm #1 — planner** | `lance-graph-planner/src/strategy/style_strategy.rs` | The D-MBX-A6 seam: the deferred `Outcome → Candidate/KanbanMove` adapter. Strategy outcomes (converged, cycle_count, gate verdicts) become kanban moves. |
| **Arm #2 — symbiont** | `crates/symbiont` (`kanban_loop.rs` = POC) | SurrealDB-on-kv-lance executor: kanban updates as KV transactions on the same Lance substrate. Gated on the AdaWorldAPI surrealdb fork `kv-lance` feature. |
| **Structural owner** | `lance-graph-supervisor/kanban_actor.rs` | ractor actor per mailbox. **ractor is SOLELY the compile-time ownership guarantee** (name the role — the attestation must stay authoritative to the compiler; operator, 2026-07-17) — it spawns and proves single-ownership via move semantics (`KanbanActor<O>` with `type State = O`; the owner MOVES in at `pre_start`); it is NOT a data-plane bus and **not for messaging — it is slow** (operator, 2026-07-02). **It MAY serve as a HELPER where it makes sense** — spawn, supervision, occasional control RPC like the serialized Advance/MulAdvance (the codex #578 atomicity mechanism) — always keeping the speed difference in mind: nothing on the hot path may wait on ractor latency; hot-path dispatch belongs to the D-V3-W2e-probed ExecTarget. |

## The trigger: kanbanstep, never an ack (operator-ruled, 2026-07-10/17)

**The canonical kanban advance is the in-stream synchronous step
("kanbanstep")**: the writer/stream fires `on_version →
try_advance_phase(&mut)` inline with the version it already holds —
auto-progression on the Rubicon **aktionale** phase (Heckhausen's model
supplies the goalstate; Libet timing governs planning and
Bewerten→Commit; 550→200 ms window). The cognitive-shader-driver
**can't stop thinking**: cycles progress continuously over
StreamDto → BusDto → PerturbationDto; the SPO 2³ rung ladder amortizes
ONE SPO cache load over ≤8 back-to-back L1-resident cycles emitting
`CausalEdge64` NARS candidates — 40 ns-class operations.

**An ack NEVER paces a cycle.** A durability ack is 10–400 µs away —
pacing a 40 ns thinking op on it stalls 250×–10,000× and destroys the
ladder amortization (and the Morton-tile cascade streaming with it).
The ack's only job is durability bookkeeping (`unacked()` crash-replay).
The ONLY tier where waiting on an ack is a feature is the **external
ontology membrane**: lance-graph-callcenter's Supabase-realtime
transcode shape (tickets, business-logic SLA) + OGAR actionhandler
queues. Tier test before any advance-path design: *SLA/audit obligation
(ticket) → SLA gate; cycle budget (thought) → kanbanstep stream.* Never
cross-route. History + severity: `E-KANBANSTEP-IS-THE-TRIGGER-1`,
`E-ACK-VIOLATION-REGRADE-1` (the ack-gated advance was a hard
architecture violation, operator-reversed — do not rebuild it).

## The ahead-firing batch writer

The batch writer is where thinking is masked behind persistence:

1. Consumer/engine produces a commit (`BusDto`) + the envelope names its
   owner (`SoaEnvelope::mailbox_owner()`).
2. Batch writer **casts** the write:
   `cast(on_behalf = envelope.mailbox_owner(), payload = BusDto)`.
3. **At cast time** (not at write-landed time) it checks the **delegation
   cache**: does the caster hold ownership, need delegation, or already have
   it? Mismatch (cast id ≠ envelope stamp) is the cache-logic case.
4. It fires the **AHEAD kanban update immediately on cast** — the kanban
   board reflects intent before the write lands. No waiting.
5. Lance's columnar I/O writes the LE bytes from the in-place backing store
   (zero-copy; the store never serializes).

## Execution speed tiers + the planner's two natures (operator, 2026-07-02)

**The planner is expected too slow for sub-microsecond handling.** The
hot dispatch path is probed, not assumed: **D-V3-W2e benchmarks
rs-graph-llm (graph-flow) execution vs SurrealQL-on-kv-lance** on the
same kanban/thinking workload; the winner owns the sub-µs `ExecTarget`
(the enum already encodes the split: Native / Jit / SurrealQl / Elixir —
kanban.rs). The planner arm remains the SLOW/plan path (strategy
selection, elevation, MUL) either way.

**Not all planner methods route through DataFusion.** The strategy
surface has two natures:

1. **Query strategies** — DataFusion-routed (parse → plan → execute).
2. **Resonance-based thinking** — the thinking style (the elixir
   low-code compiled template) MATCHING against the **Gestalt resonance
   of the object** (the perspectival `awareness_dto::ResonanceDto` —
   "what does the object say about itself", classid → ClassView), to
   reason at **rung level X** (`RungLevel`/`RungElevator`). No SQL, no
   DataFusion — a resonance match dispatches a compiled template at a
   rung. This is the V3-native dispatch and must never be forced through
   the DataFusion mold.

## Standing async plans + the 550 ms budget

Thinking cycles follow a **standing async plan** (the compiled template —
see `compiled-templates.md`) whether or not an update arrived. They never
block on being called. The 64k–256k SoA prioritizes / load-balances the
cycle work within a **550 ms net budget** (minus load delays). Elevation
(planner `elevation/`, L0→L5) is the cost model that smells resistance and
is the natural budget allocator.

## What the arms still need (gap list — see INTEGRATION-PLAN W2)

- D-MBX-A6 adapter emit (Outcome→KanbanMove) implemented, not deferred.
- Per-mailbox kanban board carried as a TENANT (type + lane).
- The ahead-firing batch writer itself (cast pairing + delegation cache).
- symbiont arm unblocked by surrealdb `kv-lance` fork coordinates.

Cross-ref: `v3-substrate-primer.md` §1–2, `write-on-behalf.md`,
`.claude/v3/soa_layout/tenants.md`, board `E-MAILBOX-KANBAN-NO-COLLAPSEGATE`.
