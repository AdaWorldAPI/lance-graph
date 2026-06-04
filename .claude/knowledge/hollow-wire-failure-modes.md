# Hollow-wire failure modes — DRAFT-INERT, sealed-but-shadowed, and the silent default

**READ BY:** integration-lead, truth-architect, code-reviewer, anyone wiring a new emitter / sink / adapter between cognitive-shader-driver and the lance-graph hot path.

**Status:** preventive documentation, generalized from bardioc B0 + B1-step2 brutal-fix rounds.

**Companion to:** `.claude/knowledge/lab-vs-canonical-surface.md` (the discipline rule). This doc names the failure mode the discipline rule prevents.

---

## 1. The pattern in one sentence

A new emitter / sink / adapter is type-correctly defined, mock-tested, and exported, but never plugged into the production hot path — the wiring looks complete; payloads are hollow. The build is green; the system produces nothing.

## 2. Three symptoms

### Symptom A — DRAFT-INERT

The new type has a clear "DRAFT" or "INERT" annotation in its docstring. Maintainers know it's not wired yet. CI passes because the OLD path is still in use. End-to-end fixtures produce default values (empty `Vec`, `None`, `0`) from the new path; the OLD path keeps producing real data via a shadow leg.

Canonical bardioc example: `services/java-ogit/.../AuditWriter.java` declared "DRAFT/INERT" in Javadoc; the entire ClickHouse wire flush path was commented out. Brutal-fix 01 caught it pre-merge. The whole audit-write subsystem looked wired (type-checks, tests against mocks) but no data ever reached ClickHouse.

### Symptom B — sealed-but-shadowed

The new types are exported and importable. Library users reach for them and assume they're hot. Internally, the boot path / `main` loop / supervision tree still wires the OLD stub instance. The new type sits sealed in the module — present, exported, never instantiated where it matters.

Canonical bardioc example: substrate-b returned empty `Vec<Span>` because `cognitive-shader-driver::lab` defined `LanceShaderSink` (fully implemented) but the boot path imported the local stub. `KanbanShaderSink` was fully implemented but nothing imported it. `dual_emitter_with_lance` exists but is never called. Three independent brutal rounds converged on this.

### Symptom C — feature-flag mismatch

A feature flag (`feature = "x-canonical"` or `[features] x = ["dep:foo"]`) ostensibly switches the canonical path on. The factory function is feature-gated. The factory is never called from the boot path — even with the feature on, the OLD path runs. Cargo check passes both with and without the feature; observed behavior is identical.

## 3. The triple-smell that names hollow wire

When all three are true at once, the wire is hollow:

1. `cargo check` passes.
2. End-to-end fixtures produce **silent default values** (empty `Vec`, `None`, `0`, all zeros).
3. `git diff` against any sibling consumer that has a real wire shows zero divergences in the new path.

That triple is the smell. The build is green, the test suite is green, and yet the output is hollow. The CI gate cannot catch this because the new path is structurally consistent; only an integration-fixture that asserts **non-default observable output** can.

## 4. Why it happens

Architecture changes ship in stages: (a) define new types, (b) migrate one site, (c) "complete migration" lands in a follow-up PR that never lands. The follow-up sits in a branch labeled "post-merge cleanup" forever.

Specific cognitive failure modes:

- **Reviewer focuses on the new code.** The reviewer reads the new file, the new traits, the new tests. The call sites that still target the OLD code don't appear in the diff because they didn't change.
- **Tests target the new modules in isolation.** Unit tests pass because they construct the new type directly. Integration tests pass because they pass mocks. The boot path is never test-touched.
- **Agents (Claude Code or human) write the "wire" but not the "boot integration".** The instruction was "add the new emitter" — interpreted as "create the type", not "make it run".
- **Feature flags are aspirational.** The flag promises a switch; the switch is never thrown.

## 5. The detection checklist (bardioc brutal pattern)

For every new emitter / sink / adapter, ask three questions:

1. **Where is `use ...`?** Find every file that imports the new type. If zero, hollow.
2. **Where is the construction site?** Find every `T::new(...)` or `T::default()`. If zero non-test sites, hollow.
3. **Where is the boot-path activation?** Find the path from `main` / `Application::start` / supervision tree to the construction site. If the boot path still constructs the OLD type, hollow.

If any of those three is missing or stuck at OLD, file the case and escalate. For deferred-to-next-round disclosures: track until closed. If still in the same wire after 2 review cycles, escalate from "deferred" to "blocking".

## 6. How to fix it

Six-step recovery, in order:

1. **Name the call sites.** Enumerate every place the OLD path is constructed; produce a TODO list.
2. **Migrate one canonical site first.** Pick the highest-traffic site; flip the construction; assert non-default output via fixture.
3. **Add an integration test that DIES on default output.** If the new path returns empty Vec, the test fails. Not a unit test on the new type; an integration test on the boot path.
4. **Delete the OLD type after the last call site is migrated.** Don't deprecate; delete. Compiler-driven migration prevents stragglers.
5. **Audit feature-flag claims.** Every `feature = "x"` that "enables" something must have a test that proves the feature actually changes observed behavior.
6. **Track to closure.** Add the migration to a tracking doc (e.g., `LATEST_STATE.md` "Pending migrations" section); status moves through `Queued → In progress → Migrated → Verified → OLD-deleted`.

## 7. Relationship to `lab-vs-canonical-surface.md`

The lab-vs-canonical discipline says: the canonical consumer surface is `UnifiedStep` via `OrchestrationBridge`; the REST / gRPC server + per-op Wire DTOs are LAB-ONLY scaffolding. Hollow-wire is the failure mode that occurs when a consumer wires the LAB transport (it compiles, types export, mocks pass) but never plugs the canonical bridge into the hot path. The lab transport is exported, importable, sometimes even mock-tested — and the hot path runs through the old stub.

Hollow-wire is most likely to occur exactly at the canonical-vs-lab boundary, because the lab surface is the easy path to import and the canonical bridge requires deeper integration. The discipline doc says what to do; this doc names what goes wrong when the discipline is violated, so reviewers and consumers have a vocabulary for the failure.

## 8. Bardioc case studies (provenance)

Two large-scale cases prove this is a structural risk, not a series of incidents:

**Case 1 (B0, S1 era):** `services/java-ogit/.../AuditWriter.java` declared DRAFT-INERT in Javadoc; the entire ClickHouse wire flush path was commented out. The audit-write subsystem looked wired (type-checked, mock-tested) but no data ever reached ClickHouse. Provenance: `bardioc/.agent-logs/brutal/b0/01-methodology.log`.

**Case 2 (B1-step2 + post-merge):** substrate-b returned empty `Vec<Span>` because it used a local stub not the `substrate-b-telemetry` path-dep. `KanbanShaderSink` was fully implemented but nothing imported it. `dual_emitter_with_lance` exists but is never called. Three independent brutals converged on this. Provenance: `bardioc/.agent-logs/brutal/b1-step2/{01,02,03}/` + post-merge sweep 03.

Treated as a **structural risk** in `bardioc/.claude/TECH_DEBT.md` provenance section.

## 9. Knowledge Activation

This doc is MANDATORY READ BY: integration-lead, truth-architect, code-reviewer, and any agent wiring a new emitter / sink / adapter between cognitive-shader-driver and lance-graph.

It sits in `.claude/knowledge/` alongside `lab-vs-canonical-surface.md`:
- `lab-vs-canonical-surface.md` is the **discipline** (the rule).
- `hollow-wire-failure-modes.md` is the **failure mode** (what goes wrong when the discipline is violated).

Together they bound the canonical-vs-lab surface: the discipline tells you what to do; this catalogue tells you the shape of the bug that surfaces when you don't.

---

## Cross-references

- `.claude/knowledge/lab-vs-canonical-surface.md` — the discipline this doc complements
- `cognitive-shader-driver/src/lib.rs:46` — the canonical-API moduledoc rule
- `bardioc/.agent-logs/brutal/b0/01-methodology.log` — DRAFT-INERT discovery
- `bardioc/.agent-logs/brutal/b1-step2/{01,02,03}` — sealed-but-shadowed discovery
- `bardioc/.agent-logs/brutal/b1-post-merge/03-hollow-wire-followup.log` — synthesis
- `bardioc/.claude/TECH_DEBT.md` — provenance + "How to read this file" sections
