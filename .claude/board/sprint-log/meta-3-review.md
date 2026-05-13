# Meta-3 Review — MedCareMembraneGate (Round 3, Stage 3)

**Reviewer:** Meta agent 3 of 3 (final review pass)
**Scope:** medcare-rs/crates/medcare-realtime/src/gate.rs + tests/{integration,regulatory}.rs
**Method:** read W9-W12 commits + log entries; cross-check against
PR #29 reference impl + Meta-1/Meta-2 carry-forwards + v1 boundary
honesty.

> **Tone:** brutally honest. This is the last review pass before sprint
> closure. Findings either block ship or document v1 limits explicitly
> for future sessions. No filler.

---

## Verdict

**Ship Round 3 + close sprint.** Zero CRITICAL findings. Two HIGH
findings are honest documentation gaps that the W12 author has
already partially captured; meta surfaces them more sharply. Two
MEDIUM findings deferred to follow-up sprints. The v1 surface is
the right shape; it's just smaller than ambition.

| # | Severity | Finding | Action |
|---|---|---|---|
| 1 | HIGH | Action operations (Operation::Act) unreachable via gate — gate routes only Read/Write | Doc note in gate.rs module head + tests/regulatory.rs; orchestration layer is the right home for action gating |
| 2 | HIGH | BtM Escalate "v1 limitation documented" tests will silently pass even if the gate IS later updated to return Escalate — the assertion is `is_allowed()` | Tighten future-spec assertions: explicit `assert_eq!(decision, AccessDecision::Allow)` so future Escalate flip is a real test failure |
| 3 | MEDIUM | Three name paths for `Policy` (rbac, gate, lib) | Choose one canonical; document the others as legacy aliases |
| 4 | MEDIUM | No bench harness validates 20-200 ns gate decision claim | Add to backlog; gate-bench follow-up sprint |
| 5 | LOW | Module-head TD-MEMBRANE-FIRST-VS-ANY caveat carries forward but no test exercises the divergence case (predicate-specific RLS) | Backlog: write a unit test once a real divergence case is identified |

---

## HIGH #1 — Action operations unreachable via gate

**Finding.** `MedCareMembraneGate::should_emit` and `evaluate` route
`gate_commit: bool` to `Operation::Read { depth }` (false) or
`Operation::Write { predicate }` (true). Neither path reaches
`Operation::Act { action }`.

This means actions like Diagnosis.classify/finalize/retract,
Prescription.issue/renew/revoke, Anamnese.append (the ONLY mutation
path for Anamnese per BMV-Ä §57), Ueberweisung.send/accept/decline,
Patient.merge/anonymize/delete cannot be gated through
`MedCareMembraneGate`. The orchestration layer must call
`medcare_rbac::Policy::evaluate(role, entity, Operation::Act { action })`
directly.

**This is intentional from PR #29's design** — the upstream
`MembraneGate` trait shape is `(commit: bool)` only. But it's a
substantial v1 limit that medcare's append-only Anamnese semantic
relies on.

**Action.** Add explicit doc note to gate.rs module head explaining
the action-routing constraint. Document orchestration layer as the
right home for action gating.

---

## HIGH #2 — BtM Escalate "limitation documented" tests are too weak

**Finding.** W12's regulatory tests for the BtM/finalize/anonymize
limitation use `decision.is_allowed()`. If a future commit lands the
Escalate wrapping, `decision` becomes `Escalate { reason: "..." }`,
which is NOT `is_allowed()`, so the test FAILS. That's the desired
flip — but the failure message will be cryptic.

**Solution.** Tighten the assertion:

```rust
// FUTURE: when row-context lands, this should become
// AccessDecision::Escalate { reason: "BtM second signature required" }
assert_eq!(decision, AccessDecision::Allow);
```

This applies to all three v1-limitation tests (BtM, finalize/retract,
anonymize). Not blocking — the loose assertions still flip when future
changes land, just with cryptic failure messages.

---

## MEDIUM #3 — Three name paths for `Policy`

Same `Policy` type reachable via:
- `medcare_rbac::policy::Policy` (canonical home)
- `medcare_realtime::gate::Policy` (re-exported via `pub use`)
- `medcare_realtime::Policy` (lib.rs crate-root re-export)

Compilation-equivalent. Cognition-confusing. Recommended canonical:
`medcare_realtime::Policy` (crate-root) — same as smb-realtime's
pattern. Backlog; doc-only update.

---

## MEDIUM #4 — No bench harness for 20-200 ns claim

Gate doc claims "decisions run at L1 inner speed (~20-200 ns)". v1
has zero benchmarks validating this.

**Solution.** Backlog item: `gate-bench-v1` follow-up adding
`criterion`-based microbenchmarks. Targets: <500 ns p99 for current
sync impl.

---

## LOW #5 — TD-MEMBRANE-FIRST-VS-ANY untested

PR #29 caveat #3 carries forward in module-head docs but writes no
test. Backlog: when a real divergence case is identified, write a
regression test. v1 doesn't have such a case.

---

## Sprint-wide closure assessment

**Round 1 (medcare-rbac):** Solid. 26 tests, 2 CRITICAL fixes applied
in revision-2.

**Round 2 (medcare-realtime skeleton):** Solid. 5 tests, 1 CRITICAL
casing fix + HIPAA-grade values applied in W7-revision-2.

**Round 3 (MedCareMembraneGate):** Solid. 33 tests across gate.rs +
integration.rs + regulatory.rs.

**Total tests:** 64 across all three crates (medcare-rbac 26 +
medcare-realtime 38).

---

## Verdict reaffirmed

**Ship.** v1 surface is correct and honest about its limits. Two HIGH
findings are documentation/test-clarity issues, not correctness
issues. Three MEDIUM/LOW findings go to backlog.

POLICY-1 / MEMBRANE-GATE-1 medcare-side seam is **CLOSED** for v1.
Topology I-1 / I-2 / I-3 / I-4 invariants are upheld. PR #29's three
TD caveats are honestly carried forward to the medcare side.
