# Agent W11 — Sprint-3 log

**Role:** Worker Agent W11 of Sprint-3 (12 + meta CCA2A).
**Branch:** `claude/tier-1-implementation-specs` of `AdaWorldAPI/lance-graph`.
**Tier:** Sprint-3 validation track.
**Deliverable anchor:** end-to-end OGIT-G smoke test design (covers
Patterns A + B + C + E + F end-to-end).

---

## Deliverable

`.claude/specs/ogit-g-smoke-test.md` — design spec for the integration
test that proves the Tier-1 + Tier-2 architecture works as a single
loop. Engineer picks this up *after* PR-A-1, PR-B-1, PR-C-1, PR-E-1,
PR-F-1 are all merged.

## Status

**DONE — spec drafted and pushed to branch via pygithub.**

## Decisions logged

1. **Healthcare picked as the smoke vertical, not Anatomy.** Healthcare
   has one consumer + one inheritance edge (Healthcare -> DOLCE), which
   keeps the surface area small. FMA + DICOM (the Anatomy demo) is
   richer but pulls in PR-D-1 + PR-ANATOMY-*; those have their own
   integration tests. Smoke stays tight.
2. **One primary test + four sub-tests, not one mega-test.** Primary
   covers happy-path five-pattern fan-out; sub-tests isolate failure
   modes (inert G error, cross-G isolation, actor panic + restart,
   compile-time duplicate-G error). Each sub-test <50 LOC so failure
   diagnosis is fast.
3. **`trybuild` for the duplicate-G ui-test.** Compile-time errors need
   a snapshot harness, not a runtime assertion. Sibling `.stderr`
   snapshot proves Pattern E's compile-time guarantee.
4. **`LanceAuditSink` in test mode with an in-memory drain.** Don't write
   to real Lance datasets in the smoke — a separate integration test on
   the lance-graph crate covers on-disk semantics. Smoke focuses on
   "did the emit shape pop out correctly" not "did the bytes land in
   parquet."
5. **Tests gated behind `#[cfg(feature = "ogit-g-smoke")]`.** Keeps the
   optional `medcare-rs` compile-in dependency off the default build
   matrix. CI's integration job enables the feature.
6. **Failure-mode table at the top of the spec.** One smoke test, five
   pattern letters — the table tells engineers which pattern broke
   when the assertion fires. Diagnostic value > test count.
7. **Soak budget: 100x in CI gate, 1000x nightly.** "Not flaky" is a
   binary claim with a number behind it. Anything failing 1/1000 in
   nightly is a P1 bug, not test flakiness — recorded in the open
   questions.

## Dependency call-out

W11 is the *last* node in the Sprint-3 PR topological sort (per W10's
graph). It depends on EVERY Tier-1 + Tier-2 spec landing first:

- PR-A-1 (W2): needed for the `g: u32` field on `AuditRecord`.
- PR-B-1 (W3): needed for `OntologyRegistry::resolve(g)`.
- PR-C-1 (W4): needed for `GenericBridge::should_emit(g, action)`.
- PR-E-1 (W5): needed for `OGIT::*` constants from the build script.
- PR-F-1 (W6): needed for `CallcenterSupervisor` + `SupervisorMsg`.

Plus a transient on `medcare-rs` having `compile-in-medcare` — W8's
consumer-template spec covers that as a worked example.

## Cross-worker handover

- **W1 (master plan):** references this smoke as the Week-4 validation
  deliverable. Smoke runtime <5 s is the gate.
- **W2, W3, W4, W5, W6 (Tier-1 + Tier-2 specs):** every type / trait
  referenced in the test sketch must exist in the sister specs. If any
  shape drifts, the smoke compile-fails and the regression is caught
  before merge.
- **W8 (consumer template):** the smoke is the validation harness for
  W8's "<30 LOC per new consumer" claim — once W8's hubspo-rs dry-run
  lands, run this smoke against hubspo too as the proof.
- **W9 (PR-D-1 FMA hydrator):** open question 3 in the spec defers
  inert-G choice to whether PR-D-1 has landed. Either inert G works;
  FMA is preferred when available.
- **W10 (PR graph):** smoke is the last topological node; W10's graph
  shows the fan-in.

## Files written this session

- `.claude/specs/ogit-g-smoke-test.md` (spec, ~11.5 KB)
- `.claude/board/sprint-log-3/agents/agent-W11.md` (this log)

## Next handover

Engineer pickup, but **only after Tier-2 lands**. Until PR-F-1 is
merged, the smoke can't compile (no `CallcenterSupervisor` symbol).
Open questions 1-5 have recommendations; engineer should confirm
choices 1 (test runtime) + 2 (audit sink storage) with the PR-F-1 /
PR-C-1 authors before coding starts.
