# OGIT-G end-to-end smoke test

**Sprint-3 deliverable -- W11 (this spec) -> engineer pickup once Tier-2 lands.**
**Pattern coverage:** A + B + C + E + F end-to-end (J + D have their own
unit-test surfaces; covered by W7 / W9 specs).
**Dependencies:** all Tier-1 (PR-A-1, PR-B-1, PR-C-1) AND all Tier-2
(PR-E-1, PR-F-1) must be merged before this test compiles.

---

## Goal

Validate the full architecture in a single integration test that exercises
the entire OGIT-G dispatch path:
**manifest -> registry -> supervisor -> consumer actor -> bridge -> audit**.

If it passes, the core "object speaks for itself" loop is shipped: a
Healthcare message dispatched by `g` index lands in the right consumer
actor, processes against a typed `ContextBundle`, and emits an
RBAC-policy-driven audit record without a single hand-coded routing table.

If it fails, the failure mode tells us *which pattern* regressed:

| Failure point | Regressed pattern | Owning PR |
|---|---|---|
| `OntologyRegistry::seed_from_manifests()` panics | E (manifest binding) | PR-E-1 |
| Registry returns no `consumer_pointer` | B (ContextBundle surface) | PR-B-1 |
| Supervisor doesn't spawn an actor for the active G | F (ractor port) | PR-F-1 |
| Dispatch hits a hand-coded match arm | C (GenericBridge) | PR-C-1 |
| `g` slot doesn't round-trip on the audit record | A (SPO-G u32) | PR-A-1 |

That's the diagnostic value: one smoke test, five-pattern fan-out.

---

## Test scenario: synthetic Healthcare consumer dispatch

Test binary compiles in:

- `lance-graph` (substrate -- SPO-G quad store)
- `lance-graph-ontology` (registry -- DOLCE root + Healthcare leaf)
- `lance-graph-callcenter` (supervisor + GenericBridge)
- `medcare-rs` (compile-in flag -- supplies `MedCareActor` + `HubSpoMessage`)

Expected behaviour, asserted step-by-step:

1. **Build script** reads `modules/dolce/manifest.yaml` +
   `modules/medcare/manifest.yaml`. Generated `OGIT::*` u32 constants
   emit (`OGIT::DOLCE_V1`, `OGIT::HEALTHCARE_V1`).
2. **`OntologyRegistry::seed_from_manifests()`** populates 2 active
   bundles: DOLCE (inert root) + Healthcare (inherits from DOLCE, has a
   `consumer_pointer`).
3. **`CallcenterSupervisor::pre_start`** spawns one `ractor` actor per
   active G *with* a `consumer_pointer`. DOLCE has none; Healthcare gets
   one `MedCareActor`.
4. **Test sends** `SupervisorMsg::DispatchToG { g: HEALTHCARE_V1.0,
   msg: GetPatientById { id: 42 }, reply }`.
5. **Supervisor routes** by `g` -> `MedCareActor`; actor replies with
   mocked `Patient { id: 42 }`.
6. **`GenericBridge::should_emit`** fires per RBAC policy from
   `medcare/manifest.yaml`.
7. **Audit emits** to `LanceAuditSink` per
   `stack_profile.audit_retention_days = 3650`. Sink is in test mode;
   test drains its in-memory buffer.

---

## Files to touch

| File | Change |
|---|---|
| `crates/lance-graph-callcenter/tests/integration/ogit_g_smoke.rs` | **NEW** -- primary smoke (~200 LOC) |
| `crates/lance-graph-callcenter/tests/integration/sub_inert_g.rs` | **NEW** -- inert-G error |
| `crates/lance-graph-callcenter/tests/integration/sub_cross_g_isolation.rs` | **NEW** -- mailbox isolation |
| `crates/lance-graph-callcenter/tests/integration/sub_actor_panic_restart.rs` | **NEW** -- supervisor restart |
| `crates/lance-graph-callcenter/tests/ui/duplicate_g_manifest.rs` | **NEW** -- `trybuild` compile-time error |
| `crates/lance-graph-callcenter/Cargo.toml` | dev-deps: `ractor`, `tokio`, `trybuild`, `medcare-rs = { path = "../../../MedCare-rs", features = ["compile-in-medcare"] }` |

Tests live behind `#[cfg(feature = "ogit-g-smoke")]`; CI enables the
feature on the integration job.

---

## Test sketch (~200 LOC)

```rust
// tests/integration/ogit_g_smoke.rs
use std::sync::Arc;
use lance_graph_callcenter::{
    CallcenterSupervisor, SupervisorMsg, test_support::LanceAuditSink,
};
use lance_graph_ontology::{OntologyRegistry, OGIT};
use medcare_rs::{HubSpoMessage, Patient};
use ractor::Actor;
use tokio::sync::oneshot;

#[tokio::test(flavor = "current_thread")]
async fn ogit_g_end_to_end_healthcare_dispatch() {
    // 1. Build registry from manifests (Pattern E + B).
    let registry = Arc::new(
        OntologyRegistry::seed_from_manifests().expect("seed manifests"),
    );

    // 2. Verify 2 active bundles + the inheritance edge.
    let dolce = registry.resolve(OGIT::DOLCE_V1.0).expect("dolce");
    let healthcare = registry.resolve(OGIT::HEALTHCARE_V1.0).expect("healthcare");
    assert_eq!(dolce.domain_name, "dolce");
    assert_eq!(healthcare.inherits_from, Some(OGIT::DOLCE_V1.0));
    assert!(healthcare.consumer_pointer.is_some(), "healthcare dispatches");
    assert!(dolce.consumer_pointer.is_none(), "dolce is inert root");

    // 3. Spawn supervisor (Pattern F).
    let (sup, handle) = Actor::spawn(
        Some("test_callcenter".into()),
        CallcenterSupervisor::default(),
        registry.clone(),
    ).await.expect("supervisor spawn");

    // 4. Dispatch a Healthcare message (Pattern C dispatcher).
    let (reply_tx, reply_rx) = oneshot::channel();
    sup.cast(SupervisorMsg::DispatchToG {
        g: OGIT::HEALTHCARE_V1.0,
        msg: HubSpoMessage::GetPatientById { id: 42, reply: reply_tx },
        reply: None,
    }).expect("cast");

    let patient: Patient = reply_rx.await.expect("reply");
    assert_eq!(patient.id, 42);

    // 5. Audit emit fired (Pattern C bridge -> Pattern A SPO-G round-trip).
    let audit = LanceAuditSink::test_drain();
    assert_eq!(audit.len(), 1);
    assert_eq!(audit[0].g, OGIT::HEALTHCARE_V1.0);
    assert_eq!(audit[0].action, "GetPatientById");
    assert_eq!(audit[0].subject_id, 42);

    sup.stop(None);
    handle.await.expect("clean shutdown");
}
```

---

## Sub-tests (each <50 LOC)

- **`inert_g_returns_error`** -- dispatch to `OGIT::DOLCE_V1.0` (or
  `OGIT::FMA_V1.0` if PR-D-1 landed) returns `SupervisorError::InertG`.
  Asserts no panic, right error variant, zero audit records emitted.
- **`cross_g_isolation`** -- spawn supervisor with both Healthcare + a
  synthetic `NoopActor` for SMB. Dispatch a Healthcare message; assert
  the SMB actor's mailbox depth is unchanged. Proves per-G routing
  doesn't leak across slots.
- **`actor_panic_supervisor_restart`** -- send `HubSpoMessage::PanicNow`;
  test-mode `MedCareActor` panics. Assert ractor's `OneForOne` strategy
  restarts the child, a subsequent `GetPatientById` succeeds, audit sink
  shows two records (failed dispatch + successful retry).
- **`manifest_validation_at_build`** (separate `trybuild` ui-test) --
  two fixture manifests both declare `g: 7`. Build script must reject
  with "duplicate OGIT slot 7 declared by `dolce` and `medcare`". Sibling
  `.stderr` snapshot. Proves Pattern E's compile-time guarantee holds.

---

## What this DOESN'T test

Spelled out so nobody mistakes the smoke for full coverage:

- **Pattern D hydration** -- PR-D-1's tests cover the FMA OWL hydrator.
  Smoke uses hand-coded seed bundles, not a hydrator pull.
- **Pattern J K-NN proximity** -- PR-J-1's tests cover INT4-32D atom
  similarity. Dispatch path here is `g`-keyed exact-match.
- **gRPC service trait** -- the lab-mode binary tests gRPC (see
  `lab-vs-canonical-surface.md`). Smoke goes through the canonical
  `OrchestrationBridge` / `SupervisorMsg::DispatchToG` path.
- **Anatomy demo** (FMA + DICOM end-to-end) -- PR-ANATOMY-* covers that
  vertical. Healthcare here keeps surface area manageable.
- **Real Lance dataset writes** -- `LanceAuditSink` is in test mode
  (in-memory). A separate `lance-graph` integration test covers on-disk.

---

## Dependencies

All Tier-1 + Tier-2 PRs:

- PR-A-1 (W2) -- `SpoQuad` with `g: u32`. Audit record's `g` field
  round-trips through this.
- PR-B-1 (W3) -- `OntologyRegistry::resolve(g) -> Option<&ContextBundle>`.
  Supervisor calls this to find `consumer_pointer`.
- PR-C-1 (W4) -- `GenericBridge::should_emit(g, action)`. Audit emit is
  policy-driven, not hand-coded.
- PR-E-1 (W5) -- `manifest.yaml` schema + build script. `OGIT::*`
  constants exist at compile time because of this.
- PR-F-1 (W6) -- `CallcenterSupervisor` ractor port. Per-G actor spawn
  + dispatch loop lives here.

Plus `medcare-rs` needing a `compile-in-medcare` feature flag and a
test-mode `MedCareActor`. W8's consumer-template covers this if the
feature isn't already in MedCare-rs.

---

## Acceptance criteria

- [ ] Smoke test runs in CI on the `integration` job; not flaky --
      run 100x in CI as a one-shot soak before declaring stable.
- [ ] Sub-tests cover inert-G error, cross-G mailbox isolation, actor
      panic + supervisor restart.
- [ ] `trybuild` ui-test covers the compile-time duplicate-G error.
- [ ] Audit trail emit verified end-to-end (drain sink, assert count
      + `g` + action shape).
- [ ] No `#[ignore]` or skip flags anywhere.
- [ ] Smoke runtime: **<5 s** primary, <2 s per sub-test.
- [ ] All five pattern letters (A + B + C + E + F) have at least one
      assertion that would fail if that pattern regressed (the failure
      table at the top of this spec).

---

## Effort

**Small.** ~200 LOC primary + ~100 LOC across four sub-tests + mock
setup. **~1 engineer-day after Tier-2 lands.**

Breakdown:

- Primary smoke -- 200 LOC
- 3 runtime sub-tests -- ~120 LOC
- 1 `trybuild` ui-test + `.stderr` -- 30 LOC
- `LanceAuditSink::test_drain()` helper -- 30 LOC (in
  `lance-graph-callcenter/src/test_support.rs`, behind `#[cfg(any(test,
  feature = "test-support"))]`)
- Fixture manifests for `dolce` + `medcare` if not provided by PR-E-1 --
  50 LOC YAML
- CI wiring -- ~20 lines

The largest single risk is fixture manifests drifting out of sync with
PR-E-1's schema; recommend the smoke depend on the same fixture files
PR-E-1 ships, not its own copies.

---

## Open questions for the engineer

1. **`tokio::test` or ractor's built-in test runtime?** Recommend
   `tokio::test(flavor = "current_thread")` -- ractor's `Actor::spawn`
   is `async fn` and tokio is the simplest harness. If PR-F-1 ships a
   `ractor::test_runtime` helper, switch to that.
2. **`LanceAuditSink::test_drain()` -- thread-local or static?**
   Recommend thread-local with a `Mutex<Vec<AuditRecord>>` so
   `--test-threads=4` doesn't cross-contaminate. PR-C-1's audit sink
   design owns this decision; this spec defers.
3. **Which inert G does `inert_g_returns_error` target?** If PR-D-1 has
   landed and seeded `OGIT::FMA_V1`, prefer that. Otherwise
   `OGIT::DOLCE_V1.0` (always present, always inert).
4. **`cross_g_isolation` -- real `smb-office-rs` or synthetic
   `NoopActor`?** Recommend synthetic noop -- the test asserts a
   *negative* (mailbox unchanged), and the real SMB consumer pulls in
   its full dependency tree for no benefit.
5. **Soak count for "not flaky" -- 100x or 1000x?** Recommend 100x in
   CI as the gate, 1000x in a nightly job. Anything failing 1/1000 in
   nightly is a P1 bug, not a flaky test.

---

## Cross-references

- Tier-1 + Tier-2 spec docs (W2, W3, W4, W5, W6)
- `.claude/specs/sprint-3-execution-plan.md` (W1 master)
- `.claude/specs/sprint-3-pr-graph.md` (W10 sequencing -- this smoke is
  the last node in the topological sort)
- `.claude/specs/consumer-crate-template.md` (W8 -- structures
  `medcare-rs`'s `compile-in-medcare` feature)
- `.claude/plans/ogit-g-context-bundle-v1.md` (Tier-1 master plan)
- `.claude/plans/compile-time-consumer-binding-v1.md` (Tier-2 master)
- `.claude/board/TECH_DEBT.md` -- TD-OGIT-G-SLOT-1 through
  TD-RACTOR-SUPERVISOR-5 (smoke is integration proof for all five)
