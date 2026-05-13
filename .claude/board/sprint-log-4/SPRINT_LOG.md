# Sprint Log 4 — Tier-2 D-SDR Follow-up + FMA Convergence Specs (12 + meta)

> **Branch:** `claude/lance-datafusion-integration-gv0BF`
> **Date:** 2026-05-13
> **Pattern:** CCA2A — append-only per-agent logs at `agents/agent-W{N}.md`; meta review at `meta-1-review.md` + `meta-2-review.md`; sprint summary at `sprint-summary.md`.
> **Protocol upgrades over sprint-3:**
> - **Sonnet-class workers** (medium context) — opus reserved for meta synthesis
> - Each worker `tee -a`'s into its own scoped log file
> - Meta agents have read-visibility across all per-agent logs
> - Pre-written SPRINT_LOG (this file) so agents see coordination state from turn 0
> - Each agent prompt includes the 11 TD-row table + 3 ideas inline

## Sprint manifest

**Goal:** Convert today's 11 TD entries (2026-05-13 batch from `.claude/board/TECH_DEBT.md`) + the FMA-heart-click smoke-test demo anchor into PR-ready implementation specs. After this sprint, an engineer can pick any spec and start coding the D-SDR Tier-2 follow-up wave.

**Branch state pre-sprint:** 9 commits ahead of `main` — D-SDR-3/4/5 code + lockfile + knowledge inbox + 4 governance harvest commits.

## TD inventory being addressed

| TD-ID | Priority | One-liner |
|---|---|---|
| TD-Q2-STUBS-DEDUP-1 | P0 | q2 local `lance-graph` + `q2-ndarray` stubs must be re-exports before FMA demo compiles |
| TD-API-DRIFT-MIDFLIGHT-1 | P0 | D-SDR-1..5 broke consumer migrations mid-air; needs deprecation path |
| TD-SUPER-DOMAIN-SUBCRATES-1 | P1 | medcare-analytics + medcare-bridge + smb-bridge + hubspot/hiro/woa not yet super-domain specialised |
| TD-SIMD-CALLCENTER-BATCH-PATHS-1 | P2 | callcenter batch paths scalar-loop where `ndarray::simd` is canonical |
| TD-THINKING-ENGINE-UNWIRED-1 | P1 | 582 KB cognitive substrate dormant; §16-§19 scaffolded clean-room instead of composed |
| TD-SDR-PR-FOLLOWUP-1 | P0 | 5 commits stacked on merged main, no follow-up PR opened |
| TD-SDR-CONSUMER-PUSH-1 | P0 | medcare-rs + smb-office-rs UnifiedBridge wirings committed locally, NOT pushed |
| TD-SDR-AUDIT-PERSIST-1 | P1 | UnifiedAuditEvent emits to in-memory chain only; no Lance/JSONL sink |
| TD-SDR-FAMILY-HYDRATION-1 | P2 | `FAMILY_TO_SUPER_DOMAIN` reverse lookup all-`Unknown` until TTL hydration |
| TD-SDR-SLOT-TRUNC-1 | P1 | `owl_from_schema_ptr` silently truncates 16-bit entity_type_id to 8-bit slot |
| TD-SDR-BRIDGE-ERR-AUDIT-1 | P2 | `BridgeError` short-circuits before audit emission — no probe-detection signal |

## Worker roster

| Agent | Deliverable | Output path | Target size |
|---|---|---|---|
| W1 | Sprint-4 master execution plan + FMA demo manifest | `.claude/specs/sprint-4-execution-plan.md` | ~12 KB |
| W2 | Q2 stubs dedup spec (TD-Q2-STUBS-DEDUP-1) | `.claude/specs/td-q2-stubs-dedup.md` | ~8 KB |
| W3 | D-SDR API deprecation playbook (TD-API-DRIFT-MIDFLIGHT-1) | `.claude/specs/td-api-drift-deprecation.md` | ~10 KB |
| W4 | Super-domain subcrate cascade (TD-SUPER-DOMAIN-SUBCRATES-1) | `.claude/specs/td-super-domain-subcrates.md` | ~12 KB |
| W5 | SIMD callcenter batch retrofit (TD-SIMD-CALLCENTER-BATCH-PATHS-1) | `.claude/specs/td-simd-callcenter-batch.md` | ~8 KB |
| W6 | thinking-engine UnifiedBridge wire-up (TD-THINKING-ENGINE-UNWIRED-1) | `.claude/specs/td-thinking-engine-wire.md` | ~12 KB |
| W7 | D-SDR PR follow-up + consumer push release plan (TD-SDR-PR-FOLLOWUP-1 + TD-SDR-CONSUMER-PUSH-1) | `.claude/specs/td-sdr-pr-release.md` | ~8 KB |
| W8 | Audit sink spec — Lance + JSONL (TD-SDR-AUDIT-PERSIST-1) | `.claude/specs/td-sdr-audit-persist.md` | ~10 KB |
| W9 | Family hydration + reverse-lookup TTL (TD-SDR-FAMILY-HYDRATION-1) | `.claude/specs/td-sdr-family-hydration.md` | ~8 KB |
| W10 | Slot widen u16 + bridge-err audit fix (TD-SDR-SLOT-TRUNC-1 + TD-SDR-BRIDGE-ERR-AUDIT-1) | `.claude/specs/td-sdr-slot-and-bridgeerr.md` | ~8 KB |
| W11 | FMA heart-click end-to-end smoke test (75K OWL → q2 3D render) | `.claude/specs/fma-heart-click-smoke.md` | ~12 KB |
| W12 | Cross-repo PR sequencing graph (stalwart + spear + lance-graph + q2 + medcare-rs + smb-office-rs) | `.claude/specs/sprint-4-pr-graph.md` | ~6 KB |
| M1 | Meta review — per-worker assessment | `.claude/board/sprint-log-4/meta-1-review.md` | ~6 KB |
| M2 | Meta synthesis — cross-spec coherence + governance updates | `.claude/board/sprint-log-4/meta-2-review.md` | ~5 KB |

## Coordination notes

- **Each agent owns distinct file paths** — no merge conflicts expected
- **Append-only** per-agent logs at `.claude/board/sprint-log-4/agents/agent-W{N}.md` via `tee -a`
- **Meta read-visibility:** M1/M2 read ALL `agents/agent-W*.md` + shipped specs before reviewing
- **D-SDR shorthand:** D-SDR = Data-SuperDomain-Routing; refers to the §13.1 PolicyRewriter chain shipped via PRs #355-#363
- **OGIT axes:** SuperDomain × OGIT-basin × OWL-leaf × DOLCE-leaf (partially orthogonal, not strictly nested)
- **FMA = Foundational Model of Anatomy** — 75K-entity OWL ontology; canonical smoke-test for OGIT↔OSINT↔Palantir/Neo4j↔q2 route
