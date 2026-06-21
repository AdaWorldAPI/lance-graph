# Capstone OUT-leg wiring — closing S2 / S3 / S4 / run-NaN (v1)

> **Status:** PLAN (2026-06-21). The execution spec for the four seams the
> Wave-0 census (`capstone-cognitive-loop-wiring-nan-census-v1.md` §4.1, codex-
> corrected) measured as **not consumed end-to-end**. Grounded in the actual
> code read this session (`orchestration.rs`, `soa_view.rs`, `scheduler.rs`,
> `canonical_node.rs`), not from memory.
>
> **Why a plan, not code:** every seam's *consumed* implementation lives in a
> disk-heavy consumer crate (planner / shader-driver / callcenter pull
> lance+datafusion ≈ 14–18 GB) or in `symbiont` (the cognitive-compilation
> session is active there). Landing contract-side **test-only** stubs would
> reproduce the very overclaim codex corrected on PR #572 ("a handler exists" ≠
> "the seam is consumed"). This plan is the honest unblock: each seam's enabler
> + consumer site + test + blocker, sequenced.

---

## The cross-cutting blockers (name them once)

1. **Disk ceiling.** A consumer-crate build (planner / shader-driver / callcenter)
   pulls `lance`+`datafusion`+`arrow` ≈ 14–18 GB of `target/`. Confirmed this
   session: a full `surrealdb-core --features kv-lance` test build exhausted the
   device twice. Any seam whose consumer is a workspace member needs disk
   headroom (clean slate ≈ 14 GB free) before its build/test is runnable here.
2. **`symbiont` ownership.** The cognitive-compilation session (PR #571,
   `cognitive-stack` golden image) is active in `symbiont`. The run-NaN harness
   (`symbiont::kanban_loop::run_to_absorbing`) lives there — coordinate or wait
   to avoid a two-session collision on one crate.

These two gate WHEN, not WHETHER. The specs below are ready to execute the
moment a surface frees.

---

## S2 — MUL→phase seam gets a real owner-side consumer

**Census state:** GAP. `NodeRow::mul_phase_step` (gate→phase) is test-only;
`sigma-tier-router` consumes `gate_decision_i4` for tier dispatch, not phase.

**Enabler (contract, tiny):** `MailboxSoaView` exposes qualia as a column —
add `fn qualia(&self) -> &[crate::qualia::QualiaI4_16D] { &[] }` (default,
non-breaking, the deferral at `soa_view.rs:157` "add when the first consumer
needs it"). **Land it IN THE SAME PR as the consumer** — a defaulted method
nothing calls is unused surface (the codex lesson).

**Consumer site:** `cognitive-shader-driver/src/mailbox_soa.rs` owner loop. On
the `Planning→CognitiveWork` crossing, per row:
`gate_decision_i4(view.qualia()[row], mantissa)` → `KanbanColumn::advance_on_gate`
→ `owner.try_advance_phase(to)`. (The column path uses `gate_decision_i4`
directly on the qualia column; `NodeRow::mul_phase_step` stays the single-node
convenience wrapper. Reconcile naming so the SAME gate logic is the one path.)

**Test:** shader-driver unit test — a known flow-vs-mismatch qualia column
advances exactly the expected rows; integer i4 gate ⇒ no NaN.

**Blocker:** shader-driver build (disk). **Decision unaffected** by Decision B.

---

## S3 — version→move gets the LIVE subscription (not a synthetic tick)

**Census state:** PARTIAL. `symbiont::kanban_loop` exercises `on_version` from a
synthetic `u32` tick (`self.cycle`); the lowering is proven, the live source is
open.

**Enabler:** none — and crucially the **live scheduler ALSO already exists**:
`lance-graph::graph::scheduler::LanceVersionScheduler<S = NextPhaseScheduler>`
with `drive_once` / `drive_at_latest` over `VersionedGraph::versions()`
(`crates/lance-graph/src/graph/scheduler.rs`). **Do NOT create a second one in
callcenter** (codex #573 — that would duplicate the surface and is the
"propose a type that already exists" tax).

**Consumer site (the ACTUAL gap):** the consumer that calls
`scheduler.drive_at_latest(view, exec)` and **applies** the returned
`Option<KanbanMove>` via `owner.try_advance_phase(move.to)`, **and suppresses
no-op ticks** (a `versions()` poll with no new version yields no apply). The
scheduler→move lowering is done; the missing work is the apply + de-dup loop in
whatever crate owns the live `VersionedGraph` + the mailbox owner.

**Test:** 2-version dataset → exactly one forward-arc apply; re-poll with no new
version → no apply (no-op tick suppressed).

**Blocker:** that consumer crate's build (lance) + disk. (Scheduler exists; this
is consumption only.)

---

## S4 — envelope route gets a `Kanban` handler (resolve-then-reject → accept)

**Census state:** GAP. `kanban.*` resolves to `StepDomain::Kanban` then every
bridge impl returns `DomainUnavailable` (`orchestration_impl.rs:55-57`,
`codec_bridge.rs:38-40`). No handler.

**Enabler (contract):** none — `StepDomain::Kanban` + `from_step_type("kanban")`
+ `OrchestrationBridge` trait all exist.

**Consumer site:** route `kanban.*` to the **owning ractor actor**. There is NO
owner-registry, NO interior-mutability-over-owner, and NO no-owner case —
**operator override (codex's S4 framing was wrong)**: *every SoA mailbox is
ALWAYS owned by its ractor actor* (mailbox-as-owner). Grounding:
`symbiont/src/kanban_loop.rs:19` "ractor is the runtime ownership … a
structural/dummy owner" in tests, a real `ractor::Actor` in prod (cf.
`lance-graph-supervisor::CallcenterSupervisor: Actor`); `SymbiontBoard`
**is** a `MailboxSoaOwner`. So an "absent owner / graceful `DomainUnavailable`"
test describes a state that cannot exist — it is void.

The wire: a `kanban.advance` step is delivered to the owning ractor actor (a
`cast`/message). The actor — which already holds the SoA `&mut` and processes
one message at a time, i.e. the compile-time single-writer / no-race guarantee
(CLAUDE.md mailbox-as-owner, E-CE64-MB-4) — applies `try_advance_phase`. The
`route()` is just the delivery edge (resolve `kanban.*` → the mailbox's actor
address); the **owner advances itself**, the bridge does not hold it. **Decision
B preserved**: identity = the actor address (the mailbox), never a `UnifiedStep`
field. (`PlannerAwareness::route` can still ACCEPT a `kanban.*` step for the
non-actor/in-process path, but the advance is the actor's, not a bridge
registry's.)

**Test:** the owning board/actor receives a kanban-advance → its phase
advances. This is exactly what `symbiont::kanban_loop` already does with the
structural/dummy owner (in-RAM); the prod path swaps the in-RAM call for a
ractor `cast` — **same ownership, same `try_advance_phase`**. No no-owner test.

**Blocker:** the actor host crate build (symbiont / a ractor consumer) → disk +
(if symbiont) the cognitive-compilation session's ownership.

---

## run-NaN — the measurement (Wave-0's third metric)

**Census state:** HYPOTHESIS. `symbiont::kanban_loop::run_to_absorbing
(&NextPhaseScheduler)` is the runnable harness.

**Work:** instrument one `run_to_absorbing` cycle; count valid vs
`NaN`/`None`/default-fallback over the observable outputs (energy column,
emitted `KanbanMove`s, phase trail). Record the real run-NaN% against the
~100% present / ~14% wired static baseline. **Caveat (codex):** the harness
drives a synthetic tick, so it measures the lowering/owner path, not a live
subscription — pair with S3 for a true live-cycle number.

**Blocker:** symbiont build (disk) **and** the cognitive-compilation session's
ownership of `symbiont` — coordinate first.

---

## Sequencing (lowest blast radius first)

1. **S4** — deliver `kanban.*` to the **owning ractor actor**, which advances
   itself via `try_advance_phase` (operator: every SoA is ractor-owned; no
   owner-registry, no no-owner case). The structural/dummy owner already does
   this in `symbiont::kanban_loop`; the wire swaps the in-RAM call for a ractor
   `cast`. Host-crate build.
2. **S2** — the `qualia()` enabler + shader-driver owner loop + test (one PR).
3. **S3** — the consumer that drives the EXISTING
   `lance-graph::graph::scheduler::LanceVersionScheduler::drive_at_latest` and
   applies the returned `KanbanMove` + suppresses no-op ticks (codex #573: do
   not add a second scheduler).
4. **run-NaN** — instrument `symbiont` (after coordinating with the other
   session); ideally after S3 so the cycle is live, not synthetic.

Each is independently shippable; none needs a `UnifiedStep` change (Decision B
holds throughout). The gate is disk headroom + symbiont coordination, not design.

## Cross-refs
`capstone-cognitive-loop-wiring-nan-census-v1.md` §4.1 (the corrected census);
`orchestration.rs` (`OrchestrationBridge`/`StepDomain::Kanban`); `soa_view.rs:157`
(the `qualia()` deferral); `scheduler.rs` (`NextPhaseScheduler`); `canonical_node.rs`
(`mul_phase_step`/`advance_on_gate`); EPIPHANIES `E-S4-ENVELOPE-STAYS-POINTERLESS`.
