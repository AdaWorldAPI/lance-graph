# KNOWLEDGE: Autoattended Multi-Agent Pattern (project-agnostic)

## READ BY:
- Any session that plans a multi-agent wave with ≥4 parallel workers
- Meta-orchestrator agents (PP-13 brutally-honest-tester, PP-14 convergence-architect, PP-15 baton-handoff-auditor, PP-16 preflight-drift-auditor)
- Any session that spawns a worker via the Agent tool
- Sprint retrospectives / post-mortems after a worker quota burn
- Any session adopting the pattern in a NEW repo (use §9 adoption checklist)

## P0 TRIGGERS:
- ≥6 independent work-slices → adopt this pattern (see §0)
- Worker quota-burn observed → re-read §11 ("scope-tightening on retry beats 'try harder'")
- About to spawn N parallel agents that touch the same file → re-read §5 Rule 1 (unique-file write discipline) BEFORE spawning
- About to skip `isolation: "worktree"` on a worker → re-read §5 Rule 2

---

> **Reference implementation:** `AdaWorldAPI/WoA` `.claude/v0.1/CLAUDE-CONTEXT.md`
> ([link](https://github.com/AdaWorldAPI/WoA/blob/main/.claude/v0.1/CLAUDE-CONTEXT.md)).
> That document is project-specific (Flask/MySQL/gunicorn deploy). This
> file distills the **transferable patterns** so any project — Rust,
> Python, TypeScript, Go — can adopt them. Domain-specific bits
> (toolchain, paths, conventions) appear as **adapter sections** at the
> end; the core loop is universal.

---

## 0. When to use this pattern

Adopt this when **all of the following** are true:

- A sprint has **≥6 independent work-slices** that can run in parallel
- Each slice can be specified clearly enough that a sub-agent can execute it without you mid-stream
- The cost of a meta-review pass is less than the cost of one rolled-back PR
- The work-tree can be partitioned so slices don't fight over the same files (or you can enforce worktree-isolation per agent)

If a sprint has ≤3 slices: just do it serially. If slices share files: serialize or pre-bake the shared scaffolding before spawning workers.

---

## 1. The orchestrator loop

The pattern is six steps. Everything else is detail.

```
1. Plan        → orchestrator plans the N-way slice partition
2. Sprint      → N parallel worker agents (1 sprint = N clusters end-to-end)
3. Meta review → 1+ honest-reviewer agents — bug hunt, NOT status report
4. Fix P0s     → orchestrator applies fixes + verifies (compile/test/lint)
5. Commit + PR → orchestrator (one PR per slice OR one combined PR per batch)
6. Repeat      → orchestrator plans the next sprint
```

**Iron rule:** the orchestrator is the only one allowed to write outside their assigned slice. Workers stay in their lane. Meta reviewers are read-only (they file findings; the orchestrator applies fixes).

---

## 2. Sprint sizing — the magic numbers

Empirically the sweet spot is **12 worker agents per batch** for codebases the size of a small monolith (~5-20 KLOC of changes per sprint). Smaller batches lose parallelism; larger batches exceed the meta-reviewer's attention and lose the "honest" property.

| Batch size | Worker model | Use when |
|---|---|---|
| 3-6 | Sonnet | Tight scope, well-defined slices, low risk of cross-cutting concerns |
| 7-12 | Sonnet | The standard sprint shape — most work fits here |
| >12 | Sonnet, split | Pre-fan into 2 batches with a meta-pass between them |
| 1-2 (planning) | Opus | Architecture / design / cross-cutting decisions |
| 1 (meta-review) | Sonnet or Opus | Per the 4-savant taxonomy below |

A 12-agent wave costs roughly **250-350k input tokens** (slice briefings + reference reads) plus **30-50k for meta**. Weigh against serial-time before dispatching.

---

## 3. The 4-savant scope partition (meta-reviewer taxonomy)

The single biggest meta-quality lever: **partition meta scope so no command is owned twice**. The four savants come from progressive refinement of "what kind of mistake is each one looking for?".

| Savant | When it runs | What it looks for | Owns commands |
|---|---|---|---|
| **PP-13 brutally-honest-tester** | POST-IMPL (after workers commit) | within-crate / within-module post-impl gates — does the actual code compile, lint clean, pass tests, satisfy the spec? | full canonical toolchain tier-1+2+3 (see §4 below) |
| **PP-14 convergence-architect** | PRE-PLAN (before partition) | divergent ideation, latent shared infrastructure, cross-slice synergies that no single worker sees | NO compile/test gates; surface-inspection only (e.g. `cargo doc`, `cargo tree`, `cargo expand`) + WebSearch / paper-search for cross-pollination |
| **PP-15 baton-handoff-auditor** | DURING-IMPL (after each slice lands) | cross-boundary contracts — does slice A's output match slice B's input expectation? DTO mismatches, missing handoffs, naming drift, type-state batons | cross-boundary commands: workspace-level cargo (`cargo check --workspace`, `cargo tree --workspace`, `cargo metadata`), public-API diff, cross-repo `git log`, cross-symbol grep |
| **PP-16 preflight-drift-auditor** | PRE-SPAWN (after plan, before worker fan-out) | spec-vs-code drift, hand-waved scope, dropped requirements, old symbols still referenced in plans/specs | git + grep only — `git log master`, `git show`, `git diff main...HEAD`, list-pull-requests, grep old symbols across `.claude/plans/` + `.claude/specs/` |

**Hand-off discipline:** each savant's prompt has an explicit "non-use → route to PP-X" line. A finding that crosses scope gets handed cleanly to the owner instead of duplicated. PP-13 owns `cargo mutants` + `cargo tarpaulin` because those are post-impl gates; PP-14 doesn't because they're not ideation tools.

### Belegte P0 finds from this pattern (anonymized)

- **Wave L** — closure-attribute import forgotten in the consolidation pass. Caught by PP-15 (cross-boundary symbol grep). Would have crashed the invoice flow.
- **Wave M** — `before_request` middleware imported but never registered. 3 security guards silently disabled. Caught by PP-13 (smoke-test against actual app boot).
- **Wave O** — Jinja autoescape on a render site: HTML logo string came out as `&lt;img&gt;` text. Caught by PP-13 (template render assertion).
- **Sprint 12 PR #18** — `RequireSuperAdmin` chained through a loader that rejected `tenant_id IS NULL`, blocking the very users the route was for. Caught by PP-15 (cross-extractor flow inspection).

**Iron rule:** P0 fixes land BEFORE the next sprint PR opens. Never let a "merged" state race against your fix-push.

---

## 4. The canonical toolchain (Rust example — adapt per language)

For **PP-13** post-impl gates. Three tiers; tier 1 runs every PR, tiers 2+3 opt-in.

### Tier 1 — every PR

| Command | Purpose |
|---|---|
| `cargo clippy --all-targets --all-features -- -D warnings -D clippy::pedantic -D clippy::nursery` | ~600 lints, strict tier |
| `cargo fmt --check` | rustfmt format gate |
| `cargo audit` | RustSec advisory scan |
| `cargo deny check` | license + dep + advisory + bans (the closest single-binary "ruff-ish" multi-axis check) |

### Tier 2 — quality / maintenance

| Command | Purpose |
|---|---|
| `cargo machete` | unused-dep detector |
| `cargo geiger` | unsafe scan (project-rule: every `unsafe` needs `// SAFETY:`) |
| `cargo semver-checks check-release` | public-API SemVer compat |
| `cargo spellcheck` | comments + docs |
| `cargo public-api` | surface diff (catches silent API drift) |

### Tier 3 — heavier / opt-in (all stable)

| Command | Purpose |
|---|---|
| `kani` | bounded model checker on stable (`#[kani::proof]` harnesses) |
| `loom` | concurrency model checker (lib, not CLI) |
| `cargo mutants` | mutation testing |
| `cargo-tarpaulin` | coverage |

### Adapter for other languages

| Language | Tier-1 equivalents |
|---|---|
| Python | `ruff check`, `ruff format --check`, `pip-audit`, `mypy --strict` |
| TypeScript | `eslint --max-warnings 0`, `prettier --check`, `npm audit --omit=dev`, `tsc --noEmit --strict` |
| Go | `golangci-lint run`, `gofmt -l`, `govulncheck ./...`, `go vet -all` |
| Generic | `<lint> --no-warnings`, `<formatter> --check`, `<advisory-scan>`, `<dep-policy>` |

The tier-1 invariant is the same in every language: **every PR proves it satisfies a no-warning gate before the orchestrator opens it**.

---

## 5. Worker-agent iron rules

These apply regardless of language. Hard-won from belegte P0s.

### Rule 1 — unique-file write discipline

Each agent writes to a **unique new file** in its slice. Never two agents at the same existing file simultaneously. Append-conflicts are guaranteed if you violate this.

When multiple slices must touch the same merge-zone (e.g. a top-level routes registry), declare it explicitly as **append-only**: each agent appends one line, the orchestrator resolves order in a final consolidation commit. Never let workers `git push` to a shared zone simultaneously.

### Rule 2 — worktree branches start from `origin/main`, not the working branch

A worktree branch started from a stale local working-branch will silently miss commits from sibling agents and overwrite them. Always:

```bash
git fetch origin <main-or-integration-branch>
git checkout -b agent/N <fresh-origin-ref>
```

If your harness supports `isolation: "worktree"`, use it — each agent gets its own clean tree.

### Rule 3 — atomic consolidation pass

After all workers return, the orchestrator does **one** consolidation commit that:
1. Pulls in each slice's commits in dependency order
2. Resolves the append-only merge zones (`src/routes/mod.rs`, `app.py` imports, etc.)
3. Re-runs tier-1 gates against the combined state
4. Files one PR (or N PRs if the slices are truly independent)

Never 12 mini-commits at the shared registry file. That's the path to "I merged your changes and now nothing builds".

### Rule 4 — pre-wave check (helper-call audit)

Before fanning out, scan each cluster for **uncommitted dependencies**:

```bash
# Adapt the regex to your language
grep -oE '[a-z_][a-z0-9_]+\(' <cluster-files> | sort -u | \
  comm -23 - <(grep -oE 'pub fn [a-z_][a-z0-9_]+' <imported-modules>/*.rs | sed 's/pub fn //;s/$/(/')
```

Clusters with non-zero unresolved-helper count need a **helper-hoist** committed before workers dispatch. Otherwise N workers each invent the same helper N different ways. (See PP-14 convergence-architect's job description.)

### Rule 5 — chunking discipline

Files >150 lines: write in chunks via `tee -a`, commit per chunk, push per chunk. NEVER one 500-line heredoc + commit + push in the same turn. The wrapping connection drops mid-serialize and you lose work. This is the load-bearing iron rule of the entire pattern.

### Rule 6 — quoting source in commit messages

For ports / behavior-preserving rewrites: every commit message quotes the source file + line range for the function being ported. Reviewers can `grep -rn "Source: " src/` to find every port. Without this, behavior-drift creeps in invisibly.

---

## 6. Memory-files pattern (the autopilot foundation)

The orchestrator and every worker share **three persistent context files** the harness loads at session start. Project-agnostic shapes:

| File | Mutability | Purpose |
|---|---|---|
| `CONTEXT.md` (or `CLAUDE.md`) | rare (architectural decisions only) | what the project is, hard rules, stack decisions, glossary |
| `JOURNAL.md` (append-only) | per-session append | dated lessons (`L<N>:` format). Never edit past entries; only append |
| `TODO.md` | hand-curated only | what's actively prioritized (the human owns this; agents read but don't write) |

**Critical rule:** a compaction summary is **not** a substitute for reading these files. Belegte: 2h lost on a bug whose diagnosis pattern was already in JOURNAL L40. Every session: `cat` the full files, never `grep`/`head`/`tail`. The first 30 seconds of context-loading saves the next 2 hours.

### Multi-file board pattern (optional extension)

For projects that outgrow a single JOURNAL: introduce a `board/` directory with 6-8 themed files:

- `Stand.md` — current snapshot, "what's running right now". **The only mutable board file.** Regenerated per session.
- `Übersicht.md` / `Index.md` — top-level navigation index
- `Goldstaub.md` — append-only lessons mirror (cross-cut view of the JOURNAL)
- `Altlasten.md` — known debts, with PR-link per line
- `Architektur_Vereinfachung.md` + `_Erledigt.md` — planned simplifications + done-mirror (plan shrinks, done grows)
- `Integrationsplan.md` — forward-looking sequencing of upcoming waves
- `Ideen.md` — captured-but-deferred ideas (Iron rule: improvement seen → here, NOT into the code)

The shape doesn't matter as much as the **single-mutable-file invariant**: agents know exactly one place to update status; everywhere else is append-only or human-edited.

---

## 7. Reference indexes — the autopilot data layer

For repos large enough that "read the source" is too expensive per worker: pre-generate JSON/MD indexes that workers consult instead of re-grepping.

| Index | Content |
|---|---|
| `routing_table.json` | every route with body, helpers used, models touched, templates rendered. Reach this BEFORE the source — it's faster + complete |
| `program_structure.json` | module → class → method tree with body LOC counts |
| `classes_index.json` / `functions_index.json` / `modules_index.json` | symbol lookup |
| `transcode_phase_plan.json` | per-route, transcode-readiness score (1A_trivial / 1B_simple / 2_moderate / 3_needs_SoC) |
| `tenant_scope_audit.md` (or equivalent security audit) | 0-leak proof: every route classified against the security invariant |
| `dead_route_audit.md` | JS-fetch-aware dead-code analysis (catches `fetch('/path')` in templates that static analyzers miss) |

**Iron rule:** re-harvest after every wave that changes the routing/symbol surface. Stale indexes are worse than no indexes — the autopilot trusts them blindly.

### Harvest tools

Keep harvest scripts under `.<harness>/v<N>/tools/`. Three core ones:

- `harvest_routes.py` (or equivalent) — walks `@route` / `@app.get` / Router::route() and dumps the JSON
- `harvest_program_structure.py` — AST walk producing class/function/module trees
- `harvest_deps.py` — call-graph + import-graph for cycle detection

Anything else (dead-route audit, tenant-scope audit, transcode-phase classifier) builds on these three.

---

## 8. The bare-`url_for` / cross-module-reference trap

A worth-its-own-section gotcha that bit production:

After moving a function into a new module/blueprint, **callers that referenced it by bare name break silently**. Flask's `url_for('login')` resolves to the current blueprint context; if `login` moved to `misc_clean_c.login` and the caller is in `system_pages.index`, you get a `BuildError` on every render.

Equivalent traps in other languages:
- Python imports: `from .login import login_view` → after a refactor, `login_view` lives elsewhere; the import succeeds (re-exported) but the symbol's behavior changed
- TypeScript: barrel-export `index.ts` resolves the wrong symbol after rename
- Rust: `use crate::login::*` glob-imports survive a function move but stop importing what you expected

**Counter-pattern:** post-move, search every callsite:

```bash
# Pick the right grep regex per language
grep -rE "url_for\(['\"]<func>['\"]" templates/ src/
grep -rE "\b<func>\(" src/
```

Every cross-module callsite needs requalification. Smoke-test every template/page that calls the moved function.

---

## 9. Adoption checklist for a new project

To bootstrap this pattern in a new repo:

1. **Memory files** — create `CONTEXT.md` (or `CLAUDE.md`), seed JOURNAL.md, TODO.md
2. **Iron rules** — write your 6-8 hard rules into CONTEXT.md (chunking, behavior-preservation, etc.)
3. **Toolchain tiers** — pick your tier-1 commands per §4; document which run per PR vs opt-in
4. **Append-only zones** — identify the 2-4 shared files agents will touch (module registry, app router, integration test list) and declare them in CONTEXT.md
5. **Harvest tools** — write routes/symbol harvesters once; cache outputs under `.claude/reference/` or `.<harness>/reference/`
6. **Savant prompts** — draft the 4 meta-reviewer system prompts (PP-13/14/15/16 templates) with the explicit "non-use → route to PP-X" lines
7. **First small sprint** — try the pattern on a 3-slice sprint first; measure orchestrator overhead, then scale to 12

The first wave often costs more (orchestrator learning + harvest tooling). By wave 3 the throughput multiplier is 6-8x serial.

---

## 10. What this pattern is NOT

- Not a substitute for **a real spec.** Workers need slice briefings tight enough that they don't have to ask questions mid-stream. The orchestrator's plan is where the engineering happens.
- Not a substitute for **code review.** Meta-reviewers catch ~80-90% of P0s but the 10-20% they miss are the ones a human PR-reviewer catches. Keep both.
- Not appropriate for **prototyping.** When the spec is fluid, parallel agents diverge. Use this for ports, refactors, mechanical fan-out — not for exploration.
- Not "free." Token cost is real (~300k per 12-agent wave). Worth it for week-scale work, not for hour-scale work.

---

## 11. Belegte session evidence

The pattern as documented here produced, in a single session against `AdaWorldAPI/woa-rs`:

- 18 merged PRs (#1–#18)
- 4 phase batches (entities, parity tests, deploy, route slices)
- 15 route slices shipped end-to-end (Phase 1.B Batches 1+2)
- 0 P0 bugs landed to main (codex bot + PP-13/14/15/16 caught all 5 attempted)
- 1 quota-induced partial-batch failure (B6/B10/B11) recovered via redispatch with tighter scope

The redispatch story is itself a meta-lesson: **scope-tightening on retry** beats "try harder" on the original scope. When a worker hits a budget cap, the answer is half the slice and a stricter brief — not the same slice with a "be faster" note.

---

## 12. Cross-references

- Reference implementation: [`AdaWorldAPI/WoA` `.claude/v0.1/CLAUDE-CONTEXT.md`](https://github.com/AdaWorldAPI/WoA/blob/main/.claude/v0.1/CLAUDE-CONTEXT.md)
- Multi-file board pattern source: [`AdaWorldAPI/WoA` `.claude/board/`](https://github.com/AdaWorldAPI/WoA/tree/main/.claude/board)
- Harvest tool examples: [`AdaWorldAPI/WoA` `.claude/v0.2/tools/`](https://github.com/AdaWorldAPI/WoA/tree/main/.claude/v0.2/tools)
- Reference index examples: [`AdaWorldAPI/WoA` `.claude/reference/`](https://github.com/AdaWorldAPI/WoA/tree/main/.claude/reference)
- woa-rs Phase 1.B v2 sprint plan: [`SPRINT-PHASE1B-PLAN-v2.md`](.claude/v0.2/SPRINT-PHASE1B-PLAN-v2.md) — concrete worked example of the 12-slice partition with savant findings folded in

---

## 13. lance-graph workspace cross-links

This workspace's existing knowledge docs and conventions that overlap with the pattern above:

- `.claude/skills/cca2a/SKILL.md` — explanation of the CCA2A pattern used in this workspace. Complementary to this doc: CCA2A is the lance-graph-specific A2A blackboard substrate (Layer 1 in `CLAUDE.md` § "Agent-to-Agent (A2A) Orchestration"); the autoattended pattern here covers the Layer-2 session-orchestration shape.
- `.claude/board/AGENT_LOG.md` — workspace's Layer-2 blackboard (matches §6 multi-file-board pattern: the AGENT_LOG is the append-only journal; `LATEST_STATE.md` is the mutable Stand.md).
- `.claude/board/LATEST_STATE.md` + `PR_ARC_INVENTORY.md` — these are the mandatory cold-start reads per `CLAUDE.md` § "Session Start — MANDATORY READS". They serve the same role as `CONTEXT.md` + `JOURNAL.md` in §6.
- `.claude/agents/BOOT.md` — the workspace's 19-specialist + 5-meta-agent ensemble. The 4-savant taxonomy in §3 here generalises the meta-agent role; this workspace's `truth-architect`, `integration-lead`, `workspace-primer`, `adk-coordinator`, `adk-behavior-monitor` map roughly to the PP-13/14/15/16 + an additional ADK-behavior savant.
- CLAUDE.md § "Model Policy (P0)" — codifies the grindwork-vs-accumulation split that §2 of this doc encodes in the per-batch-size model table.

## 14. Adoption notes for lance-graph (concrete deltas vs the generic pattern)

- **Worktree isolation:** when spawning Agent subagents in this workspace, pass `isolation: "worktree"` for any worker that will write to shared crates (lance-graph, ndarray, contract). Without it, the orchestrator and worker share a working tree and mid-flight state is indistinguishable from exited-without-commit state (see Sprint-13 W-I1 retry incident).
- **Tier-1 commands for lance-graph:** the workspace uses Rust 1.95 stable. The `cargo clippy --all-targets --all-features -- -D warnings -D clippy::pedantic -D clippy::nursery` invocation in §4 is the right call here; some `-D` lints may need carve-outs documented in `Cargo.toml`'s `[lints]` table per crate.
- **Append-only zones in lance-graph:** `crates/lance-graph-contract/src/lib.rs` (re-export list), `crates/lance-graph/Cargo.toml` (`[features]` table), `crates/lance-graph/src/graph/mod.rs` (graph backend registry). Workers touching these must declare them in their slice brief.
- **Harvest tools:** lance-graph already has some — `docs/UNIFIED_INVENTORY.md`, `docs/TYPE_DUPLICATION_MAP.md`, `docs/SEMIRING_ALGEBRA_SURFACE.md`. These map to §7's "reference indexes" role and should be re-harvested after any wave that adds a new type or semiring.
- **5th savant for SIMD invariants:** lance-graph (and any consumer of the ndarray fork) extends the project-agnostic 4-savant taxonomy in §3 with a workspace-specific `simd-savant` whose card lives at `.claude/agents/simd-savant.md`. Its scope is **the SIMD source-of-truth invariant — all SIMD must come from `ndarray::simd` via the polyfill (`simd.rs` + `simd_ops.rs` > `simd_{type}.rs` per-arch)**. It runs PRE-SPAWN (verifies worker brief routes through the polyfill), DURING-IMPL (commit-level grep for raw `_mm*` / `vld1q_*` outside `ndarray/src/simd_*`), and PRE-MERGE (gate on any PR with SIMD code). 8-entry AP-SIMD-N anti-pattern catalogue covers the common violations. Hand-offs: UB/OOB → PP-13; missing primitive → file `TD-NDARRAY-SIMD-<NAME>` and route to ndarray maintainer; spec-drift → PP-16; cross-crate aliasing → PP-15. Workspace-specific because SIMD source-of-truth depends on having a polyfill repo to be the source — not all projects do, so this savant is an adapter, not a transferable §3 slot.

### Rule 7 — judgment agents READ, don't grep (added 2026-06-01)

When a review/council/grounding agent's **verdict** depends on a type's semantics, its brief MUST instruct it to **READ the relevant files in full** (the Read tool), NOT grep/sed/head/tail. A fragment out of context produces a confident-but-wrong judgment — **grep is for LOCATING, reading is for JUDGING.** Belegt: 4 successive wrong plasticity framings (CausalEdge64-lens / per-plane-axis / Heel-vs-PlasticityState / Heel-compose) all came from narrating off grep fragments; the EW64 council's R3 found a grep-induced mis-citation (`edge.rs:750`). Cross-ref EPIPHANIES `E-READ-NOT-GREP`.
