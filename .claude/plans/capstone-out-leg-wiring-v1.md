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

**Consumer site:** an **owner-carrying bridge**, NOT a `PlannerAwareness`
branch. Codex #573 (verified): `PlannerAwareness` holds only
`strategies`/`selector` (`lib.rs:99-103`), `route(&self, &mut UnifiedStep)` is
`&self`, and `UnifiedStep` is pointer-free (`orchestration.rs:343-355`) — so a
`PlannerAwareness::route` branch can ONLY mark status `Completed` (accept); it
**cannot** reach an owner to `try_advance_phase`, and cannot exercise the
no-owner error case. The real S4 mechanism:
- a dedicated `KanbanBridge` that **holds the mailbox owner** behind interior
  mutability (`RwLock<dyn MailboxSoaOwner>` or a registry keyed by mailbox —
  needed because `route(&self)` must not take `&mut self` during compute, per
  the no-`&mut`-during-compute data-flow rule). Its `route()` accepts only
  `StepDomain::Kanban`, parses the sub-op after the `kanban.` prefix, and
  applies the move via the held owner; an unregistered/absent owner ⇒ graceful
  `OrchestrationError` (the fail-closed half).
- **Decision B preserved**: `UnifiedStep` gains NO field — the identity reaches
  the owner via the bridge's HELD state (mailbox registry), not via the step.
The "single `PlannerAwareness` branch" is demoted to an accept-only stub; it is
NOT the smallest *true* wire (it can't advance a phase).

**Test:** register an owner in the bridge; route `step_type:"kanban.advance"` →
owner phase advanced + status `Completed`; route with no owner registered →
graceful Err (no panic).

**Blocker:** the bridge's host crate build (datafusion+lance via planner, or
shader-driver) → disk.

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

1. **S4** — the owner-carrying `KanbanBridge` (holds the mailbox owner via
   interior mutability) + test. NOT a `PlannerAwareness` branch (codex #573:
   that can only accept, not advance). Smallest *true* wire; host-crate build.
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
