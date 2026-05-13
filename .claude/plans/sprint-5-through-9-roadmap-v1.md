# Sprint 5–9 Roadmap — D-SDR Follow-up to FMA Convergence to Compliance Certification

> **Status:** v1 plan, 2026-05-13 evening
> **Branch:** `claude/lance-datafusion-integration-gv0BF`
> **Predecessor:** PR #363 merged at `421e71e` (D-SDR-1 + D-SDR-2 + canonical spec)
> **Current branch state:** 16 commits ahead of `main` — D-SDR-3/4/5 + sprint-log-4 governance/specs
> **Sprint-4 retrospective:** see `.claude/board/EPIPHANIES.md` 2026-05-13 duplication-audit entry; sprint-4 specs partially duplicated prior plan corpus. Sprint-5+ worker prompts must include mandatory `.claude/plans/` read-order as hard precondition.

---

## Worker-prompt template fix (applies to ALL sprints)

Before spawning any worker on an architectural spec, the prompt MUST include:

```
Mandatory read-order BEFORE drafting:
1. ls .claude/plans/ | head -40
2. Read all plans matching: grep -l "<your topic>" .claude/plans/
3. Read these by default:
   - .claude/plans/unified-ogit-architecture-v1.md (15-pattern master)
   - .claude/plans/super-domain-rbac-tenancy-v1.md (canonical RBAC/tenancy)
   - .claude/plans/anatomy-realtime-v1.md (proof-of-vision FMA demo)
4. Your spec must be a DELTA against the relevant plan, not a fresh draft.
   Cite §X of the plan you're extending, or explain why you're not.
```

This single change is the highest-leverage fix from sprint-4 retrospective.

---

## Sprint 5 — Substrate close-out (1 week)

**Goal:** ship PR-A (D-SDR-3/4/5 follow-up) + the 3 surgical fixes that unblock everything downstream.

**Worker roster (12 + 2 meta) — each owns ONE file:**

| Agent | File scope | Deliverable |
|---|---|---|
| W1 | `.claude/specs/sprint-5-execution-plan.md` | Sprint-5 master plan + PR-A body draft + sprint-4 retrospective summary |
| W2 | `.claude/specs/s5-pr-a-d-sdr-followup.md` | PR-A body (cite `super-domain-rbac-tenancy-v1.md §13`; SHAs `2c3e87d`/`1d0157f`/`dabd510`/`dc9e081`) |
| W3 | `.claude/specs/s5-pr-b-medcare-consumer-push.md` | medcare-rs UnifiedBridge wiring PR body + pre-push verify commands |
| W4 | `.claude/specs/s5-pr-c-smb-office-consumer-push.md` | smb-office-rs UnifiedBridge wiring PR body + pre-push verify commands |
| W5 | `.claude/specs/s5-pr-d1-slot-u16-widen.md` | Slot u8→u16 widen + `slot_u8()` deprecated shim + 65k-id round-trip test |
| W6 | `.claude/specs/s5-pr-d2-bridge-err-audit.md` | `authorize_*` refactor `?` → `match`; invert test at line 690; probe-detection dashboard SQL |
| W7 | `.claude/specs/s5-pr-d3a-lance-audit-sink.md` | LanceAuditSink with Arrow schema (u64 FNV-1a merkle) + Hive partitioning |
| W8 | `.claude/specs/s5-pr-d3b-jsonl-audit-sink.md` | JsonlAuditSink + CompositeSink + `audit-verify` CLI for chain replay |
| W9 | `.claude/specs/s5-pr-d4-family-hydration.md` | `new_hydrated()` + `SuperDomain::Unhydrated=255` + 1-hour TTL refresh + sanity gate |
| W10 | `.claude/specs/s5-pr-d5-compat-shim.md` | `compat_v0_4` shim layer for D-SDR API drift + 30-day auto-deletion CI lint |
| W11 | `.claude/specs/s5-ci-matrix.md` | Per-PR green-gates; cross-repo zipball build matrix; rollback SLA |
| W12 | `.claude/specs/s5-handover-to-s6.md` | Sprint-5 retrospective + sprint-6 setup checklist + mandatory-reads inventory |
| M1 | `.claude/board/sprint-log-5/meta-1-review.md` | Per-worker brutal review |
| M2 | `.claude/board/sprint-log-5/meta-2-review.md` | Cross-spec synthesis + governance updates |

**Each worker scratchpad:** `.claude/board/sprint-log-5/agents/agent-W{N}.md` via `tee -a` append-only.

**Critical mandatory-reads in every W{N} prompt:** the 12-step list from `EPIPHANIES.md` 2026-05-13 duplication-audit entry.

**Gates:** PR-A (W2) must merge before PR-B (W3) / PR-C (W4); PR-D1 (W5) before PR-D3a (W7).

---

## Sprint 6 — Tier-2 composable wiring (2 weeks)

**Goal:** wire thinking-engine into UnifiedBridge + cascade super-domain subcrates. All workers MUST extend prior plans, not redraft.

**Worker roster (12 + 2 meta):**

| Agent | File scope | Deliverable | Prior plan to extend |
|---|---|---|---|
| W1 | `.claude/specs/sprint-6-execution-plan.md` | Master plan + dependency graph | All below |
| W2 | `.claude/specs/s6-pr-e1-medcare-specialisation.md` | MedCare super-domain finalisation | `super-domain-rbac-tenancy-v1.md §14` + `foundry-roadmap §LF-3` |
| W3 | `.claude/specs/s6-pr-e2-smb-retrofit.md` | smb-office-rs retrofit to UnifiedBridge generic | `super-domain-rbac-tenancy-v1.md §14` |
| W4 | `.claude/specs/s6-pr-e3-woa-extraction.md` | woa-rs extraction from q2/geo | `super-domain-rbac-tenancy-v1.md §14` |
| W5 | `.claude/specs/s6-pr-e4-hiro-scaffold.md` | hiro-rs new repo (Networking/SRE template) | `super-domain-rbac-tenancy-v1.md §14` |
| W6 | `.claude/specs/s6-pr-e5-hubspot-scaffold.md` | hubspot-rs new repo (CRM template) | `super-domain-rbac-tenancy-v1.md §14` |
| W7 | `.claude/specs/s6-pr-f1-thinking-engine-wire.md` | `lance-graph-cognition-bridge` crate; collapse D-SDR-13/15/17 | `jc-pillars-runtime-wiring-v1.md` + `compile-time-consumer-binding-v1.md §F` |
| W8 | `.claude/specs/s6-pr-g1-manifest-modules.md` | `/modules/<name>/manifest.yaml` build-script + codegen | `compile-time-consumer-binding-v1.md §E` |
| W9 | `.claude/specs/s6-pr-g2-ractor-supervisor.md` | `CallcenterSupervisor` ractor port | `compile-time-consumer-binding-v1.md §F` |
| W10 | `.claude/specs/s6-conformance-tests.md` | Workspace-wide `consumer-crates-conformance` CI gate | `super-domain-rbac-tenancy-v1.md §14` |
| W11 | `.claude/specs/s6-cross-repo-pr-graph.md` | PR sequencing across 5 super-domain repos | `foundry-roadmap §LF-3..LF-8` |
| W12 | `.claude/specs/s6-handover-to-s7.md` | Sprint-6 retro + sprint-7 setup | (output of all above) |
| M1 | `.claude/board/sprint-log-6/meta-1-review.md` | Per-worker review |
| M2 | `.claude/board/sprint-log-6/meta-2-review.md` | Cross-spec synthesis |

**Gates:** W2..W6 land sequentially per foundry-roadmap order. W7 + W8/W9 parallel-track.

---

## Sprint 7 — FMA convergence + Tier-4 perf (1 week)

**Goal:** ship the FMA proof-of-vision demo per `anatomy-realtime-v1.md`. SIMD callcenter retrofit in parallel.

**Worker roster (12 + 2 meta):**

| Agent | File scope | Deliverable | Prior plan to extend |
|---|---|---|---|
| W1 | `.claude/specs/sprint-7-execution-plan.md` | Master plan | `anatomy-realtime-v1.md` |
| W2 | `.claude/specs/s7-pr-h1-lance-graph-rdf-crate.md` | New border crate: RDF ingest + OntologyContextId + SemanticQuad | `lance-graph-rdf-fma-snomed-v1.md` |
| W3 | `.claude/specs/s7-pr-h2-fma-owl-ingest.md` | FMA OWL Tier-2 ingest (full graph, BioPortal source) | `lance-graph-rdf-fma-snomed-v1.md` |
| W4 | `.claude/specs/s7-pr-h2b-fma-csv-quick.md` | FMA CSV Tier-1 ingest (preview/bootstrap) | (new — sprint-4 W11 patch) |
| W5 | `.claude/specs/s7-pr-h3-q2-cypher-wire.md` | q2 graph-notebook Cypher cell → UnifiedBridge auth chain | `anatomy-realtime-v1.md` |
| W6 | `.claude/specs/s7-pr-h4-heart-click-test.md` | Heart-click integration test (5 golden inputs) + Pillar 6 EWA propagation assertion | `anatomy-realtime-v1.md` + `jc-pillars §6` |
| W7 | `.claude/specs/s7-pr-h5-simd-callcenter.md` | `vsa_udfs.rs` scalar→`ndarray::hpc::{vsa,bitwise}` retrofit | (sprint-4 W5 spec, corrected) |
| W8 | `.claude/specs/s7-pr-h6-drug-knowledge-crosswalk.md` | MedCare drug-knowledge-2026-05-05 pivot integration | (MedCare-rs release) |
| W9 | `.claude/specs/s7-pr-h7-snomed-ingest.md` | SNOMED-CT ingest (companion to FMA, license-gated) | `lance-graph-rdf-fma-snomed-v1.md` |
| W10 | `.claude/specs/s7-pr-h8-radlex-ingest.md` | RadLex radiology imaging labels ingest | `lance-graph-rdf-fma-snomed-v1.md` |
| W11 | `.claude/specs/s7-cross-crate-dep-graph.md` | sprint-7 dependency graph + parallel-track ordering | (output of W2..W10) |
| W12 | `.claude/specs/s7-handover-to-s8.md` | Sprint-7 retro + sprint-8 setup + FMA demo recording protocol | — |
| M1 | `.claude/board/sprint-log-7/meta-1-review.md` | Per-worker review |
| M2 | `.claude/board/sprint-log-7/meta-2-review.md` | Cross-spec synthesis |

**Gates:** W2 → W3 → W5 → W6 sequential. W4 + W7 + W8 + W9 + W10 parallel.

---

## Sprint 8 — TTL namespaces + compliance certification (2 weeks)

**Goal:** address D-SDR-6..D-SDR-39 scope deferred by PR #363. Hardens product for customer compliance audits.

**Worker roster (12 + 2 meta):**

| Agent | File scope | Deliverable |
|---|---|---|
| W1 | `.claude/specs/sprint-8-execution-plan.md` | Master plan + compliance test matrix |
| W2 | `.claude/specs/s8-pr-i1-ttl-namespaces.md` | TTL namespace registry + per-tenant entity-type ranges |
| W3 | `.claude/specs/s8-pr-i2-hipaa-cert.md` | HIPAA cert surface: BAA audit export + encryption-at-rest gate |
| W4 | `.claude/specs/s8-pr-i3-sox-cert.md` | SOX: financial-data segregation + dual-control audit emissions |
| W5 | `.claude/specs/s8-pr-i4-gdpr-cert.md` | GDPR: right-to-erasure tombstones + data-minimization predicate rewrites |
| W6 | `.claude/specs/s8-pr-i5-osint-lanceprobe.md` | `osint_edge_traversal.rs` → production; LanceProbe M5/M6 unblock |
| W7 | `.claude/specs/s8-pr-i6-federation-phase2.md` | LanceDB transparent encrypted view for cross-tenant queries |
| W8 | `.claude/specs/s8-audit-replay-cli.md` | Deep-dive: `audit-verify` CLI v2 with selective tenant/date-range replay |
| W9 | `.claude/specs/s8-baa-audit-export-schema.md` | BAA-ready audit export format (CSV/JSON/PARQUET) + chain-of-custody proof |
| W10 | `.claude/specs/s8-encryption-at-rest-gate.md` | Encryption-at-rest enforcement at Lance write boundary + KMS integration sketch |
| W11 | `.claude/specs/s8-compliance-test-matrix.md` | Per-regime test matrix + customer-audit-readiness checklist |
| W12 | `.claude/specs/s8-handover-to-s9.md` | Sprint-8 retro + sprint-9 setup |
| M1 | `.claude/board/sprint-log-8/meta-1-review.md` | Per-worker review |
| M2 | `.claude/board/sprint-log-8/meta-2-review.md` | Cross-spec synthesis |

**Gates:** W2 (TTL) first. W3..W6 parallel. W7 last (depends on encryption surface from W10).

---

## Sprint 9 — Q2 cockpit + holographic cinematic (1-2 weeks, parallelizable)

**Goal:** ship the holographic UX + WOW cinematic. Sales-asset budget; can run parallel to sprints 5-8.

**Worker roster (12 + 2 meta):**

| Agent | File scope | Deliverable |
|---|---|---|
| W1 | `.claude/specs/sprint-9-execution-plan.md` | Master plan + storyboard outline |
| W2 | `.claude/specs/s9-pr-j1-q2-splat-shader.md` | q2 frontend Gaussian-splat WGSL/GLSL fragment shader (additive blending + bloom + cyan palette) |
| W3 | `.claude/specs/s9-pr-j2-render-frame-highlight.md` | `RenderFrame::highlight` SoA column + Pillar 6 Σ-displacement write-back |
| W4 | `.claude/specs/s9-pr-j3-fma-canonical-pose.md` | T-pose canonical-pose seeder; procedural RenderFrame initial state |
| W5 | `.claude/specs/s9-pr-j4-layer-toggle.md` | Per-system layer toggle UI + `SuperDomain::Healthcare` family slice filter |
| W6 | `.claude/specs/s9-pr-j5-cinematic-prerender-tool.md` | `crates/fma-cinematic` offline tool: trajectory + Floyd-Steinberg dither + Lance schema |
| W7 | `.claude/specs/s9-pr-j6-cinematic-player.md` | q2 cinematic player: Lance read + AVX2 nibble unpack + canvas blit |
| W8 | `.claude/specs/s9-pr-j7-release-artifact.md` | `intro-v1.lance` release artifact + q2 session-start hook |
| W9 | `.claude/specs/s9-camera-trajectory-storyboard.md` | Hand-curated 30-90 s camera path: full-body → cardiovascular → heart |
| W10 | `.claude/specs/s9-cyan-palette-curation.md` | 16-color cyan/teal/white palette design + accessibility check |
| W11 | `.claude/specs/s9-audio-cue-design.md` | Web Audio API integration: hover/click cues + ambient drone for hologram |
| W12 | `.claude/specs/s9-demo-recording-script.md` | Demo recording script + sales-pitch sync points |
| M1 | `.claude/board/sprint-log-9/meta-1-review.md` | Per-worker review |
| M2 | `.claude/board/sprint-log-9/meta-2-review.md` | Cross-spec synthesis |

**Gates:** W2..W5 (live UX) before live recording starts. W6..W8 + W9..W11 parallel; W12 last.

**Owner:** demo / marketing budget, NOT engineering critical path.

---

## Agent dispatch pattern (applies to every sprint)

Per `sprint-log-2/3/4` precedent (see `.claude/board/sprint-log-*/SPRINT_LOG.md`):

1. **Pre-write** `.claude/board/sprint-log-{N}/SPRINT_LOG.md` with worker roster + mandatory-reads inventory
2. **Spawn 12 worker agents in parallel** (single main-thread Agent tool call with 12 invocations). All sonnet (medium context). Each prompt:
   - States the worker ID + file scope + deliverable
   - Embeds the 12-step `.claude/plans/` mandatory-read list
   - Instructs `tee -a .claude/board/sprint-log-{N}/agents/agent-W{X}.md` for milestones
   - Reminds: "Write tool IS pre-allowed for *.md; retry once if denied"
   - Reminds: "DO NOT git commit/push — main thread aggregates"
3. **Wait for completion notifications** (background async)
4. **Spawn meta agents** (M1 + M2, opus model) after all workers complete. M1 reads all worker logs + specs; M2 cross-synthesizes.
5. **Main thread commits + pushes** all artifacts in one batch (workers' specs, scratchpads, meta reviews, SPRINT_LOG.md).
6. **Update governance** per board-hygiene rule: LATEST_STATE.md / PR_ARC_INVENTORY.md / STATUS_BOARD.md / AGENT_LOG.md.

**Permission gotcha (from sprint-4 retro):** sonnet workers occasionally bail at first Write denial. Prompt template includes explicit "RETRY ONCE if denied" instruction. `Bash(tee:*)` is bulletproof fallback; `Write(**/*.md)` is preferred when it works.

---

## Cumulative sizing

| Sprint | Calendar | Est. total LOC | Critical-path-blocking? |
|---|---|---|---|
| Sprint 5 | 1 week | ~1350 + 0 (existing commits) | Yes — blocks all downstream |
| Sprint 6 | 2 weeks | ~3300 | Yes — Tier-2 wiring |
| Sprint 7 | 1 week | ~2450 | Yes — FMA proof-of-vision |
| Sprint 8 | 2 weeks | ~3600 | Customer compliance (per sales pressure) |
| Sprint 9 | 1-2 weeks | ~1700 | Sales asset, parallelizable |

**Critical-path total (sprints 5+6+7):** 4 weeks, ~7100 LOC. Sprint 8 + 9 are independently scheduleable.

---

## Open questions for human reviewer

1. **Sprint 6 PR-E3..E5 (woa-rs / hiro-rs / hubspot-rs new repos):** confirm these get separate `adaworldapi/<name>-rs` repos vs subcrates of an existing one. Affects CI matrix.
2. **Sprint 8 priority:** does customer pipeline contain a HIPAA-blocking deal that pulls sprint 8 forward of sprint 7? If yes, swap.
3. **Sprint 9 budget approval:** demo team have ~1 week to spend, or held until sales asks?
4. **MedCare-rs drug-knowledge-2026-05-05 release:** is this a sprint-7 PR-H6 must-have or a sprint-8+ enrichment?
5. **Sprint 5 PR-D5 (compat shim):** consumer repos already merged or do they require a coordinated cutover? Affects PR-A acceptance criteria.

---

## Mandatory governance updates per merged PR

Every PR in this roadmap must update (in the same commit, per `CLAUDE.md` board-hygiene rule):
- `.claude/board/LATEST_STATE.md` — contract inventory + shipped table
- `.claude/board/PR_ARC_INVENTORY.md` — PREPEND entry with Added / Locked / Deferred / Docs / Confidence
- `.claude/board/STATUS_BOARD.md` — D-id row status flip
- `.claude/board/AGENT_LOG.md` — if agent-driven, PREPEND completion entry
