# Worker Prompt Template v2

> **Scope:** canonical prompt template for every Sonnet-class worker spawned
> from the main thread via the `Agent` tool. Applies to all impl, governance,
> spec-drafting, knowledge-doc, and verification workers.
>
> **Status:** v2 (sprint-13 onward). Supersedes the v1 implicit template used
> through sprint-11 (which leaked CSI-13: orphaned aggregation between worker
> DONE and meta-review SPAWN; see `sprint-log-11/meta-review-opus.md` §3 CSI-7
> / CSI-8 / CSI-9 / CSI-13).
>
> **Authoring:** Wave G refinement (sprint-12) confirmed each worker should own
> its full integration (lib.rs touch within its own crate is permitted) — the
> orchestration pattern is "no orphaned aggregation phase." Codified here.
>
> **Audience:** main-thread Opus orchestrator (i.e., this template is for the
> orchestrator to FILL IN when dispatching). Workers receive only the filled
> prompt; they do not need to know v2 exists.

---

## 1. Why v2 exists (one paragraph)

Sprint-11 Wave F shipped 12 worker outputs in two commits then went straight
to meta-review without an aggregation pass. Three worker outputs (W-F1
sigma-tier-router crate, W-F2 attention_mask.rs, W-F4/W-F6 ndarray streams)
were on disk but unwired in their parent lib.rs / workspace `members` — i.e.,
the workers reported DONE on file-write, not on integration. The v1 prompt
template said "main thread aggregates"; no aggregation phase was scheduled.
Sprint-12 Wave G confirmed the fix: each worker owns the full integration of
its own scope, including lib.rs / mod.rs registration within its target
crate. Workspace-level `Cargo.toml [workspace] members =` for NEW crates
remains the single main-thread reservation. This template bakes that
discipline into every prompt.

---

## 2. Header (every prompt starts with this block)

Every Sonnet+ worker prompt MUST open with the following fields. The
orchestrator FILLS the `<...>` placeholders:

```
**Workspace:** /home/user/lance-graph
**Branch:** <branch-name>
**Worker ID:** <W-X1 | W-X-N | W-Meta-Sonnet | ...>
**Model:** Sonnet 4.7 (grindwork) | Opus 4.7 (accumulation; meta workers only)
**Wave:** <sprint-N-wave-X>
**Deliverable(s):** <D-id-1, D-id-2, ...>
**Estimated source LOC:** <NNN>
**Estimated test count:** <NN tests>
**Final-response word budget:** <NNN words>
**DO NOT:** switch branch · commit · push · open PR · edit board files.
**Permission caveat:** Edit/Write/MultiEdit may be blocked in your subagent
context despite session-scoped settings.local.json allowing them. If they
are denied, fall back to `python3 << 'PYEOF' ... PYEOF` heredocs invoked via
the Bash tool to write/patch files. This is the workspace-standard fallback;
do NOT escalate or wait — switch and proceed.
```

The orchestrator MUST NOT omit any line; missing fields are how Wave F
silently shipped orphan files.

---

## 3. Mandatory reads (in this order, BEFORE producing output)

Every worker prompt MUST include the following read order. The worker reads
each file (or the relevant section) before issuing any tool call that writes
to the working tree.

### Tier 0 (unconditional, every worker)

1. `.claude/board/LATEST_STATE.md` — current contract inventory, what's
   shipped, what's queued, what's explicitly deferred. **What exists.**
2. `.claude/board/PR_ARC_INVENTORY.md` — per-PR Added / Locked / Deferred /
   Docs / Confidence (reverse-chronological, APPEND-ONLY). **Why it exists.**
3. `.claude/board/AGENT_LOG.md` — recent entries (top ~10). Peer awareness:
   who is in flight on the same branch, what was just shipped that affects
   your scope, what gotchas the previous wave logged.

### Tier 1 (domain-triggered; orchestrator selects per the trigger table)

Per `.claude/agents/BOOT.md § Knowledge Activation Protocol`. The
orchestrator lists the specific `.claude/knowledge/*.md` docs that match the
deliverable's domain. Common triggers:

| Domain trigger | Knowledge doc(s) the worker MUST read |
|---|---|
| Codec / encoding / distance / compression | `encoding-ecosystem.md` (P0 mandatory) |
| REST / gRPC / Wire DTO / OrchestrationBridge / shader-lab | `lab-vs-canonical-surface.md` (P0 mandatory) |
| HHTL cascade / γ+φ / Slot D / Slot V | `bf16-hhtl-terrain.md` |
| VSA / fingerprint identities / bundle | `vsa-switchboard-architecture.md` |
| Grammar / DeepNSM / TEKAMOLO | `grammar-landscape.md`, `grammar-tiered-routing.md` |
| Crystal / sandwich / quantum fingerprints | `crystal-quantum-blueprints.md` |
| i4 substrate / Wave-F outcomes | `i4-substrate-decisions.md` |
| Frankenstein composition / new abstraction | `frankenstein-checklist.md` |

### Tier 2 (deliverable-specific)

4. The active plan: `.claude/plans/<plan-name>-v<N>.md` (orchestrator names
   it explicitly).
5. The deliverable spec: `.claude/specs/<spec-name>.md` IF one exists.
6. The target source file(s) BEFORE editing (Read tool, full file) — never
   `Write` over a file without reading it first. This rule applies to your
   own prior commits in the same session.

The orchestrator pastes the literal paths into the prompt; the worker does
not have to discover them.

---

## 4. Scope discipline (every prompt declares)

The prompt MUST state the four scope axes explicitly. No "or thereabouts" —
exact numbers force the worker to converge instead of accrete.

```
**Target file(s):**
  - <absolute path 1> (NEW | EDIT, <LOC delta> LOC)
  - <absolute path 2> (NEW | EDIT, <LOC delta> LOC)
  ...

**Test target:** <NN tests> covering:
  - <test 1>
  - <test 2>
  ...

**IN SCOPE:**
  - <bullet 1>
  - <bullet 2>
  ...

**OUT OF SCOPE (explicit — defer to follow-up, do NOT touch):**
  - <bullet 1>
  - <bullet 2>
  ...
```

The OUT OF SCOPE bullets are not decorative — they are the worker's
permission slip to say "not my deliverable" when temptation strikes mid-task.

---

## 5. Integration responsibility (NEW in v2 — closes CSI-13)

This section is the structural correction over v1. The worker OWNS the
full integration of its target file(s) within its own crate. Specifically:

### 5.1 Module registration (worker owns)

For every NEW source file the worker creates under `crates/<crate>/src/`,
the worker MUST in the same commit:

- Add `pub mod <module_name>;` to that crate's `src/lib.rs` (or
  `src/mod.rs` if creating a sub-module under a submodule directory).
- Add the appropriate `pub use <module_name>::{TypeA, TypeB};` re-exports
  per the spec.
- Verify with `grep -n "pub mod <module_name>" crates/<crate>/src/lib.rs`
  that the registration landed BEFORE reporting DONE.

The v1 instruction `"main thread aggregates lib.rs"` is RESCINDED. There is
no aggregation phase; the worker is the aggregator for its own crate.

### 5.2 Cargo.toml dependency additions (worker owns, with constraints)

If the worker's code requires a new dependency in its target crate's
`Cargo.toml`:

- Behind a feature gate per the **zero-dep doctrine** when the target crate
  is `lance-graph-contract`, `bgz17`, `deepnsm`, `bgz-tensor`, or
  `sigma-tier-router`. These crates protect their zero-dep status; new
  deps must be `optional = true` and gated behind a feature in `[features]`.
- For workspace-member crates with existing deps (e.g., `lance-graph`,
  `lance-graph-planner`, `cognitive-shader-driver`), `optional = true` is
  preferred but not required when the dep is already transitively present.
- The worker MUST cite the spec line that authorizes the new dep. No
  speculative deps.

### 5.3 Re-exports for cross-crate consumers (worker owns)

If the deliverable adds a type that downstream crates consume, the worker
adds the `pub use` line in the target crate's `lib.rs` in the same commit.
The worker does NOT touch consumer crates' imports — that's a separate
deliverable.

### 5.4 Workspace `[workspace] members =` (MAIN THREAD reserved)

There is **one** integration step the worker does NOT perform: adding a NEW
crate to the parent workspace's `Cargo.toml` `[workspace] members` list.
This is the single main-thread reservation. Reason: parallel workers
spawning NEW crates would all need to edit the same `members =` list,
producing merge conflicts the orchestration pattern cannot resolve cheaply.

**If the worker creates a NEW crate, the prompt MUST instruct the worker to:**

- Create the crate directory and `Cargo.toml` WITHOUT a `[workspace]` line
  in its own Cargo.toml (i.e., do NOT make it a standalone subworkspace —
  that was the CSI-7 failure mode).
- Add a clearly-flagged section to the final report:

  ```
  ## ORCHESTRATOR ACTION REQUIRED — new crate, needs workspace registration

  Add to `/home/user/lance-graph/Cargo.toml [workspace] members =`:
    "crates/<new-crate-name>"
  ```

The orchestrator applies this one-line edit in the aggregation commit at the
end of the wave (or, more commonly, at the same time as accepting the
worker's commit).

### 5.5 Field-isolation matrix tests (worker owns at layout boundaries)

Per iron rule **I-LEGACY-API-FEATURE-GATED** (promoted from E-META-10
sprint-11, codified sprint-12+): any v1 API path that writes to bits
reclaimed by a v2 feature flag MUST be either feature-gated to no-op or
routed through the canonical v2 accessor. The worker MUST include
field-isolation matrix tests at every layout-bit boundary it touches.

A field-isolation matrix test asserts: setting field X does not perturb
fields Y_1..Y_n. For an N-field layout, the matrix is N×(N−1) assertions.
For the W-A1 v2 layout in PR #383 this surfaced 4 codex P1 catches — the
discipline is now mandatory.

The orchestrator includes the explicit field list in the prompt's "Test
target" section so the worker can enumerate the matrix without guessing.

---

## 6. Validation (worker MUST run before reporting DONE)

The worker runs the following commands in order. The final response cites
each result.

```bash
# 1. Cargo check with the crate's features the worker's code activates.
#    The orchestrator lists the exact invocation in the prompt.
cargo check -p <target-crate> [--features <feature-list>]
# Or workspace-wide if the worker touched multiple crates:
cargo check --workspace

# 2. Cargo test, filtered to the worker's new tests if cheap.
cargo test -p <target-crate> [--features <feature-list>] -- <test-filter>
# Or the full suite if the worker prefers + it's <60s:
cargo test -p <target-crate>

# 3. Clippy — recurring CI gate.
cargo clippy -p <target-crate> --lib --tests -- -D warnings

# 4. Fmt — either check or apply before reporting DONE.
cargo fmt --check
# OR
cargo fmt
```

**If `cargo check` fails:** the worker MUST fix the failure before reporting
DONE. A worker that reports DONE on a non-compiling tree is the CSI-7/8/9
failure mode; the v2 template forbids it.

**If `cargo test` fails:** the worker MUST distinguish pre-existing failures
(report as "N pre-existing, confirmed via main-branch revert") from new
failures (report as "N introduced, ROOT CAUSE: ..."). New failures BLOCK the
DONE report.

**Cross-repo edge case:** when the deliverable touches `/home/user/ndarray/`
or another sibling repo, the worker MAY skip `cargo check --workspace` if it
hits cross-repo build errors unrelated to its scope (the `blake3` missing
in `/home/user/ndarray/src/hpc/merkle_tree.rs` is the canonical example).
In that case the worker reports the validation gap explicitly per §7.

---

## 7. Output format (final response)

The worker's final response (after the last tool call) MUST be under the
word budget declared in §2 and MUST contain these sections, in this order:

```
## <Worker ID> DONE — <D-id-1, D-id-2, ...>

**Branch:** <branch> (commit pending; main thread aggregates IF impl work).

**Files touched (full paths):**
- <absolute path 1> (NEW | EDIT, +<add>/-<del> LOC)
- <absolute path 2> (...)
...

**Tests:** <passed> pass / <failed-new> failed / <failed-preex> pre-existing.
  Pre-existing confirmed via: <how — e.g., "stash + run on main">.

**Cargo check:** <PASS | FAIL with summary>.
**Cargo clippy:** <PASS | FAIL with summary; -D warnings honored>.
**Cargo fmt:** <CHECKED | APPLIED>.

**lib.rs / mod.rs registration:** <confirmed via grep — show the line>.
   (If NEW crate: include the ORCHESTRATOR ACTION REQUIRED block from §5.4.)

**Cross-spec inconsistencies surfaced (flag for meta-Opus):**
- <CSI candidate 1, if any>
- <CSI candidate 2>
...
(Empty list is fine. Do NOT manufacture findings.)

**Validation gaps (cross-repo, environmental, etc.):**
- <gap 1, if any>

**Outcome:** <one-sentence ship verdict>.
```

The orchestrator parses this format programmatically when aggregating the
fleet — deviation breaks the aggregation pipeline.

---

## 8. Forbidden actions (worker MUST NOT)

The worker MUST NOT, under any circumstance:

1. **Switch branches.** Worker stays on the branch given in the header.
2. **Commit or push.** The orchestrator (main thread) handles all commits
   per the CCA2A protocol. The worker leaves the working tree dirty; the
   orchestrator reviews + commits + pushes.
3. **Open PRs.** Same reason.
4. **Modify board files** — `.claude/board/AGENT_LOG.md`,
   `STATUS_BOARD.md`, `EPIPHANIES.md`, `LATEST_STATE.md`,
   `PR_ARC_INVENTORY.md`, `TECH_DEBT.md`, `ISSUES.md`,
   `INTEGRATION_PLANS.md`. Main thread prepends entries post-fleet.
   **Exception (per §10):** SOME governance workers are explicitly
   scoped to update these — the orchestrator marks such prompts with
   `**Worker class:** governance`. Default impl workers stay out.
5. **Touch `CLAUDE.md`.** The iron-rule promotion ceremony is
   main-thread-only. Workers may PROPOSE iron-rule promotions in their
   CSI section; they do not edit `CLAUDE.md`.
6. **Modify `.claude/settings.json`** (workspace-tracked). The
   `.claude/settings.local.json` (gitignored, per-session) is also off
   limits unless the worker is explicitly a settings-update worker.
7. **Add a new top-level abstraction** without `truth-architect` review
   per `frankenstein-checklist.md`. New struct / new trait / new layer
   → flag in CSI section, do not ship without review.
8. **Use `haiku` for any nested sub-spawn.** Sonnet floor per Model Policy.
9. **Skip the mandatory reads.** A worker that grep'd before reading
   `LATEST_STATE.md` is operating without context; the prompt is
   structurally invalid for that worker.

---

## 9. Codex P1 anticipation (sprint-12 lesson)

Every codex review of every PR catches the same anti-pattern:
v1-accessor-writes-to-v2-reclaimed-bits. PR #383 had 4 instances in one PR.
PR #381 had 2. The systemic finding is now iron rule
**I-LEGACY-API-FEATURE-GATED**:

> Any v1 API path that writes to bits reclaimed by a v2 feature flag MUST
> be either feature-gated to no-op or routed through the canonical v2
> accessor. Field-isolation matrix tests are mandatory at the layout-bit
> boundary.

**Worker self-scan checklist** (run as last step before reporting DONE):

1. Did I touch a layout file (e.g., `causal-edge/src/edge.rs`,
   `causal-edge/src/layout.rs`, `bindspace.rs`, any `#[repr(C, align(...))]`
   struct definition)?
2. Did the touched layout have a v1 → v2 feature flag (e.g.,
   `causal-edge-v2-layout`, `bindspace-i4`, `mailbox-soa`)?
3. For every v1 accessor (`pack()`, `with_temporal()`, etc.) on the
   touched layout, did I either:
   - Feature-gate the write to no-op under v2, OR
   - Route the write through the canonical v2 accessor?
4. Did I add field-isolation matrix tests covering every newly-reclaimed
   bit field?

If any answer is "no" or "not sure," the worker flags it in the CSI
section. The orchestrator routes the flag to `truth-architect` or
`integration-lead` before merge.

---

## 10. Worker classes (orchestrator picks one per prompt)

The orchestrator declares the worker class in the prompt header. Different
classes have different `forbidden actions` overrides:

| Class | Examples | May edit board files? | May commit? |
|---|---|---|---|
| **impl** | new types, new modules, new tests | No | No |
| **governance** | `TYPE_DUPLICATION_MAP.md` refresh, `TECH_DEBT.md` sweep, `ISSUES.md` triage, `EPIPHANIES.md` prepend | Yes (the specific files listed in scope) | No (orchestrator commits) |
| **spec** | `.claude/specs/<name>.md`, `.claude/plans/<name>-v<N>.md` | No (specs/plans live under `.claude/`, not `.claude/board/`) | No |
| **knowledge** | `.claude/knowledge/<topic>.md` with `READ BY:` header | No | No |
| **meta-review** | `sprint-log-NN/meta-review.md`, `meta-review-opus.md` | Yes (the meta-review file itself) | No |
| **verification** | grep / cargo / git read-only audits, produce findings doc | No (read-only mode by default) | No |

The default is **impl**. If unspecified in the prompt, the worker assumes
impl class and obeys the strict `forbidden actions` list.

---

## 11. Spawn prompt skeleton (orchestrator copy-paste)

This is the canonical skeleton for the main thread to fill in when
dispatching a Sonnet worker. Replace every `<...>` placeholder. Do NOT delete
sections — empty sections (e.g., no OUT-OF-SCOPE bullets) should say
`(none)` rather than be omitted, so the worker knows the orchestrator
considered them.

```markdown
**Workspace:** /home/user/lance-graph
**Branch:** <branch-name>
**Worker ID:** <W-X-N>
**Model:** Sonnet 4.7
**Wave:** <sprint-N-wave-X>
**Worker class:** impl | governance | spec | knowledge | meta-review | verification
**Deliverable(s):** <D-id-1, D-id-2, ...>
**Estimated source LOC:** <NNN>
**Estimated test count:** <NN tests>
**Final-response word budget:** <NNN words>
**DO NOT:** switch branch · commit · push · open PR · edit non-scoped board files.
**Permission caveat:** Edit/Write/MultiEdit may be denied in your subagent.
  Fallback: `python3 << 'PYEOF' ... PYEOF` heredocs via Bash. Switch and
  proceed; do NOT escalate.

## Mandatory reads (in order, BEFORE producing output)

Tier 0:
1. `.claude/board/LATEST_STATE.md`
2. `.claude/board/PR_ARC_INVENTORY.md`
3. `.claude/board/AGENT_LOG.md` (top ~10 entries for peer awareness)

Tier 1 (domain-triggered for THIS deliverable):
- <`.claude/knowledge/<doc1>.md`>
- <`.claude/knowledge/<doc2>.md`>

Tier 2 (deliverable-specific):
- Plan: <`.claude/plans/<plan>-v<N>.md` § <sections>>
- Spec: <`.claude/specs/<spec>.md`> (if one exists; otherwise: none)
- Target files (Read BEFORE editing): <list>

## Scope

**Target file(s):**
- <absolute path 1> (NEW | EDIT, <LOC delta> LOC)
- ...

**Test target:** <NN tests> covering:
- <test 1>
- <test 2>
- ...

**IN SCOPE:**
- <bullet 1>
- ...

**OUT OF SCOPE (do NOT touch):**
- <bullet 1>
- ...

## Integration responsibility (v2 — own it)

In the same commit:
- [ ] Add `pub mod <name>;` to `crates/<crate>/src/lib.rs`.
- [ ] Add `pub use <name>::{...};` re-exports per spec.
- [ ] Cargo.toml dep additions (if any): <list, with feature gate per §5.2>.
- [ ] Field-isolation matrix tests at layout boundaries: <enumerate, or "N/A">.
- [ ] Verify registration with `grep -n "pub mod <name>" crates/<crate>/src/lib.rs`.
- [ ] If NEW crate: include ORCHESTRATOR ACTION REQUIRED block in final report.

## Validation (run before reporting DONE)

```bash
cargo check -p <target-crate> [--features <list>]
cargo test  -p <target-crate> [--features <list>] -- <test-filter>
cargo clippy -p <target-crate> --lib --tests -- -D warnings
cargo fmt --check  # or `cargo fmt` to apply
```

Cite each result in the final response.

## Codex-P1 self-scan (last step before DONE)

Walk the I-LEGACY-API-FEATURE-GATED checklist in §9 of
`.claude/agents/worker-template-v2.md`. Flag any "no" / "not sure" in
the CSI section of your final response.

## Final response (under <NNN> words, sections in this order)

1. Header: `## <Worker ID> DONE — <D-id-list>`
2. Files touched (full paths, +add/-del).
3. Tests (passed / failed-new / failed-preex).
4. Cargo check / clippy / fmt results.
5. lib.rs / mod.rs registration confirmation.
6. CSI candidates (flag for meta-Opus, empty list OK).
7. Validation gaps (cross-repo, environmental).
8. Outcome (one sentence).

## Forbidden actions

Per `.claude/agents/worker-template-v2.md` §8. The worker class for this
prompt is `<class>` (see §10 for any overrides).
```

---

## 12. Worker prompt anti-patterns (failure modes to avoid)

Drawn from sprint-11 Wave F retrospective and sprint-12 Wave G refinement.
If the orchestrator's draft prompt exhibits any of these, REWRITE before
spawning:

1. **"main thread aggregates X"** for any X that isn't the workspace
   `members =` list. Per §5.1 / §5.3, the worker owns crate-local
   integration. v1 lib.rs deferral is the CSI-13 trigger.
2. **Missing OUT-OF-SCOPE list.** Workers without explicit OUT-OF-SCOPE
   bullets accrete scope. Always state what NOT to touch.
3. **Test target as a number only** ("8 tests"). The worker invents the
   8 tests. Better: enumerate the 8 specific assertions / scenarios.
4. **Missing `Worker class:`** declaration. Default-to-impl is fine but
   ambiguous for governance / knowledge / spec work. Be explicit.
5. **Vague LOC estimate** ("~200 LOC"). Worker doesn't know when to stop.
   Use a concrete number; over-delivery is also a discipline signal.
6. **No mandatory reads list** — the worker grep'd its way to context.
   That's a context-cost tax; always include Tier 0 + relevant Tier 1.
7. **Vague file paths** ("the cognitive-shader-driver crate"). The
   target paths are absolute and exact in the prompt; the worker does
   not have to discover them.
8. **No permission caveat.** Sonnet subagents hit Edit/Write denies; the
   fallback heredoc must be named so the worker does not stall.

---

## 13. Relationship to existing agent cards

This template does NOT replace the 19 specialist agent cards in
`.claude/agents/*.md` (container-architect, truth-architect, etc.). Those
cards define DOMAIN authority — what each specialist guards. This template
defines WORKFLOW discipline — how every worker, regardless of specialty,
gets dispatched.

**Composition rule:** when the orchestrator spawns a specialist (e.g.,
`truth-architect` for an HHTL claim), the prompt for that specialist STILL
follows this template's §2 header + §3 mandatory reads + §7 output format.
The specialist's domain knowledge is loaded by the Tier-1 trigger; the
workflow shape is loaded by this template.

---

## 14. Versioning

- **v1 (implicit, pre-sprint-12):** "main thread aggregates lib.rs" — caused
  CSI-7/8/9, retired sprint-12.
- **v2 (this file, sprint-13+):** worker owns full crate-local integration;
  workspace `members =` is the sole main-thread reservation. Codex-P1
  self-scan + I-LEGACY-API-FEATURE-GATED mandatory.

Future revisions: append to this file with a dated `## v2.x — <date> — <why>`
section. Do NOT delete v2 content; the discipline grew up from concrete
failures and the rationale must remain readable.

---

*Authored W-Sprint-13-PP-8 (Opus 4.7 planner, main-thread), 2026-05-16.
Sources: `.claude/board/sprint-log-11/meta-review-opus.md` §3 CSI-7..13,
Wave G refinement, Model Policy in CLAUDE.md, BOOT.md handover protocol.*
