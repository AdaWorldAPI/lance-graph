# Sprint Log 3 — Tier-1 Implementation Specs (12 + meta)

> **Branch:** `claude/tier-1-implementation-specs`
> **Off-main commit:** 4f7082a398ed
> **Pattern:** CCA2A — append-only per-agent logs at `agents/agent-W{N}.md`; meta review at `meta-1-review.md`; sprint summary at `sprint-summary.md`.
> **Protocol upgrades over sprint-2:**
> - **pygithub direct REST with quote-stripped GITHUB_TOKEN** (avoid MCP throttling)
> - Pre-verify branch + repo before write
> - Pre-written SPRINT_LOG (this file) so agents see coordination state from turn 0
> - Each agent prompt includes the canonical pattern letter table inline

## Sprint manifest

**Goal:** convert the 11 TD entries (TD-OGIT-G-SLOT-1 through TD-DEEPNSM-NSM-COLLAPSE-11 from `.claude/board/TECH_DEBT.md`) into PR-ready implementation specs. After this sprint, an engineer can pick any spec and start coding.

**Branch state pre-sprint:** `main` HEAD `4f7082a398ed` includes PR #358 (sprint-2 architecture synthesis) + PR #359 (tier-0 canonical pattern letter fix).

## Worker roster

| Agent | Deliverable | Output path | Target size |
|---|---|---|---|
| W1 | Sprint-3 master execution plan | `.claude/specs/sprint-3-execution-plan.md` | ~15 KB |
| W2 | PR-A-1 spec: SPO-G u32 slot (TD-OGIT-G-SLOT-1) | `.claude/specs/pr-a-1-spo-g-u32-slot.md` | ~10 KB |
| W3 | PR-B-1 spec: ContextBundle typed surface (TD-CONTEXT-BUNDLE-2) | `.claude/specs/pr-b-1-context-bundle.md` | ~10 KB |
| W4 | PR-C-1 spec: GenericBridge + ConsumerPointer (TD-GENERIC-BRIDGE-3) | `.claude/specs/pr-c-1-generic-bridge.md` | ~10 KB |
| W5 | PR-E-1 spec: /modules/<name>/manifest.yaml (TD-MANIFEST-MODULES-4) | `.claude/specs/pr-e-1-manifest-modules.md` | ~12 KB |
| W6 | PR-F-1 spec: ractor supervisor port (TD-RACTOR-SUPERVISOR-5) | `.claude/specs/pr-f-1-ractor-supervisor.md` | ~12 KB |
| W7 | PR-J-1 spec: INT4-32D thinking atoms (TD-INT4-32D-ATOMS-6) | `.claude/specs/pr-j-1-int4-32d-atoms.md` | ~8 KB |
| W8 | Consumer crate template: hubspo-rs scaffolding guide | `.claude/specs/consumer-crate-template.md` | ~10 KB |
| W9 | PR-D-1 spec: FMA OWL hydrator (PR-ANATOMY-1) | `.claude/specs/pr-d-1-fma-owl-hydrator.md` | ~10 KB |
| W10 | Sprint-3 PR sequencing + dependency graph | `.claude/specs/sprint-3-pr-graph.md` | ~6 KB |
| W11 | End-to-end OGIT-G smoke test design | `.claude/specs/ogit-g-smoke-test.md` | ~8 KB |
| W12 | Trivia bundle: TD-9 + TD-10 + TD-11 quick PRs | `.claude/specs/trivia-prs-bundle.md` | ~6 KB |
| M1 | Meta review (main thread) | `.claude/board/sprint-log-3/meta-1-review.md` | ~6 KB |

## Coordination notes

- **Each agent owns distinct file paths** — no merge conflicts expected
- **pygithub-first** for all writes (avoid MCP throttling); each agent strips token quotes
- **Append-only** for per-agent logs at `.claude/board/sprint-log-3/agents/agent-W{N}.md`
- **Canonical pattern letters** from W1 master (sprint-2): A SPO-G u32 / B ContextBundle / C GenericBridge / D Meta-Structure Hydration / E Compile-Time Consumer Binding / F ractor Supervisor / G Best-Practice Thinking / H Switchable Cognitive Vessel (SHIPPED) / I Implicit Cognition (SHIPPED) / J INT4-32D Atoms / K Circular Compilation / L SPO-Chain Narrative / M Wave-Particle Bimodal / N Fingerprint-as-Codebook (SHIPPED) / O Phenomenological Memory (SHIPPED)
