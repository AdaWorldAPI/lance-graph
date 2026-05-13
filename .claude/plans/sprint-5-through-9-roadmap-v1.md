# Sprint 5–9 Roadmap — D-SDR Follow-up to FMA Convergence to Compliance Certification

> **Status:** v1 plan, 2026-05-13 evening
> **Branch:** `claude/lance-datafusion-integration-gv0BF`
> **Predecessor:** PR #363 merged at `421e71e` (D-SDR-1 + D-SDR-2 + canonical spec)
> **Current branch state:** 16 commits ahead of `main` — D-SDR-3/4/5 + sprint-log-4 governance/specs
> **Sprint-4 retrospective:** see `.claude/board/EPIPHANIES.md` 2026-05-13 duplication-audit entry; sprint-4 specs partially duplicated prior plan corpus.
> **Sprint structure:** each sprint runs **12 worker agents (W1-W12) + 2 meta agents (M1 per-worker review, M2 cross-spec synthesis)**, CCA2A pattern. Each worker owns exactly one spec file. Append-only per-agent scratchpad at `.claude/board/sprint-log-N/agents/agent-W{N}.md` via `tee -a`. Sonnet workers, Opus meta.

---

## Worker-prompt template (MANDATORY for every worker in every sprint)

```
You are worker W{N} in sprint-log-{S}. CWD: /home/user/lance-graph. CCA2A pattern.

Your deliverable: ONE file at .claude/specs/<scoped-path>.md (~10 KB).
Your scratchpad: .claude/board/sprint-log-{S}/agents/agent-W{N}.md (tee -a, append-only).

Mandatory read-order BEFORE drafting (DO NOT skip — sprint-4 duplication retrospective):
1. ls .claude/plans/ | head -40
2. grep -l "<your topic keywords>" .claude/plans/   # find domain-relevant plans
3. Read by default:
   - .claude/plans/unified-ogit-architecture-v1.md (15-pattern master, A-O)
   - .claude/plans/super-domain-rbac-tenancy-v1.md (canonical RBAC, 1387 lines)
   - .claude/plans/anatomy-realtime-v1.md (FMA proof-of-vision)
4. Read your sprint's specific predecessor plans (listed in your row below)

Your spec must be a DELTA against the relevant plan(s), not a fresh draft.
Cite the §X of the plan you're extending, or explain why you're not.

Permission note: Write(**/*.md) + Bash(tee -a:*) pre-allowed. RETRY ONCE if first call denied.
Do NOT git commit/push — main thread aggregates.

Report <120 words: spec byte size + plans cited + key delta-vs-prior.
```

---

## Sprint 5 — Substrate close-out (1 week)

**Goal:** ship PR-A (D-SDR-3/4/5 follow-up) + surgical fixes that unblock everything.

**Worker roster (each owns 1 file):**

| W | Deliverable | Output file | Prior plan to extend |
|---|---|---|---|
| W1 | Sprint-5 execution plan + retrospective integration | `.claude/specs/sprint-5-execution-plan.md` | this roadmap; sprint-4 retrospective in EPIPHANIES |
| W2 | PR-A body + cherry-pick strategy for D-SDR-3/4/5 commits | `.claude/specs/pr-a-d-sdr-followup.md` | `super-domain-rbac-tenancy-v1.md` §13 (D-SDR-3/4/5 named there) |
| W3 | PR-B medcare-rs push + UnifiedBridge wiring | `.claude/specs/pr-b-medcare-push.md` | `foundry-roadmap-unified-smb-medcare-v1.md` LF-3 |
| W4 | PR-C smb-office-rs push + UnifiedBridge wiring | `.claude/specs/pr-c-smb-office-push.md` | same as W3 |
| W5 | PR-D1 slot u8→u16 widen spec | `.claude/specs/pr-d1-slot-widen.md` | `super-domain-rbac-tenancy-v1.md` §3 (OwlIdentity u16) |
| W6 | PR-D2 bridge-error audit emission spec | `.claude/specs/pr-d2-bridge-err-audit.md` | `super-domain-rbac-tenancy-v1.md` §13 (audit chain) |
| W7 | PR-D3a LanceAuditSink spec (Arrow schema + partitioning) | `.claude/specs/pr-d3a-lance-audit-sink.md` | `anatomy-realtime-v1.md` (LanceAuditSink named as substrate) |
| W8 | PR-D3b JsonlAuditSink + CompositeSink + verify CLI spec | `.claude/specs/pr-d3b-jsonl-and-verify.md` | same as W7 |
| W9 | PR-D4 family hydration + TTL refresh spec | `.claude/specs/pr-d4-family-hydration.md` | `super-domain-rbac-tenancy-v1.md` §6 (FAMILY_TO_SUPER_DOMAIN) |
| W10 | PR-D5 compat shim layer (`compat_v0_4`) + auto-deletion lint | `.claude/specs/pr-d5-compat-shim.md` | sprint-4 W3 spec extension |
| W11 | CI matrix + green-gate criteria per PR | `.claude/specs/sprint-5-ci-matrix.md` | `.github/workflows/` inventory |
| W12 | Sprint-5 PR dependency graph + sprint-6 handover | `.claude/specs/sprint-5-pr-graph.md` | sprint-4 W12 spec extension |
| M1 | Per-worker brutal review | `.claude/board/sprint-log-5/meta-1-review.md` | (reads all worker outputs) |
| M2 | Cross-spec synthesis + governance updates | `.claude/board/sprint-log-5/meta-2-review.md` | same |

**Estimated total LOC across PRs:** ~1350 + 0 (commits exist for PR-A/B/C).

---

## Sprint 6 — Tier-2 composable wiring (2 weeks)

**Goal:** wire thinking-engine + cascade super-domain subcrates + Pattern E/F (manifest + ractor). All DELTA against prior plans.

**Worker roster:**

| W | Deliverable | Output file | Prior plan to extend |
|---|---|---|---|
| W1 | Sprint-6 execution plan | `.claude/specs/sprint-6-execution-plan.md` | sprint-5 retrospective |
| W2 | PR-E1 MedCare super-domain finalisation | `.claude/specs/pr-e1-medcare-super-domain.md` | `super-domain-rbac-tenancy-v1.md` §14 |
| W3 | PR-E2 smb-office UnifiedBridge retrofit | `.claude/specs/pr-e2-smb-retrofit.md` | same §14 |
| W4 | PR-E3 woa-rs extraction from q2/geo | `.claude/specs/pr-e3-woa-rs-extract.md` | same §14 + `q2-foundry-integration-v1.md` |
| W5 | PR-E4 hiro-rs scaffold (Networking/SRE template) | `.claude/specs/pr-e4-hiro-rs-scaffold.md` | same §14 |
| W6 | PR-E5 hubspot-rs scaffold (CRM template) | `.claude/specs/pr-e5-hubspot-rs-scaffold.md` | same §14 |
| W7 | PR-F1 thinking-engine UnifiedBridge wire-up | `.claude/specs/pr-f1-thinking-engine-wire.md` | `jc-pillars-runtime-wiring-v1.md` + ERRATUM |
| W8 | PR-G1 manifest module build-script + codegen | `.claude/specs/pr-g1-manifest-modules.md` | `compile-time-consumer-binding-v1.md` Pattern E |
| W9 | PR-G2 CallcenterSupervisor ractor port | `.claude/specs/pr-g2-ractor-supervisor.md` | same plan, Pattern F |
| W10 | Cross-crate registry conformance test design | `.claude/specs/sprint-6-conformance-test.md` | sprint-6 W1 |
| W11 | Cross-repo PR sequencing graph (5+ repos in this sprint) | `.claude/specs/sprint-6-pr-graph.md` | sprint-5 W12 |
| W12 | Sprint-6 retrospective template + sprint-7 handover | `.claude/specs/sprint-6-retrospective.md` | sprint-5 retrospective |
| M1 | Per-worker brutal review | `.claude/board/sprint-log-6/meta-1-review.md` | (reads all worker outputs) |
| M2 | Cross-spec synthesis + super-domain coherence audit | `.claude/board/sprint-log-6/meta-2-review.md` | same |

**Estimated total LOC:** ~3300.

---

## Sprint 7 — FMA convergence + Tier-4 perf (1 week)

**Goal:** ship the FMA proof-of-vision demo. SIMD callcenter retrofit in parallel. Cinematic stays out (sprint 9).

**Worker roster:**

| W | Deliverable | Output file | Prior plan to extend |
|---|---|---|---|
| W1 | Sprint-7 execution plan | `.claude/specs/sprint-7-execution-plan.md` | sprint-6 retrospective |
| W2 | PR-H1 lance-graph-rdf border crate (Turtle/N-Quads/RDF-XML/OWL) | `.claude/specs/pr-h1-lance-graph-rdf.md` | `lance-graph-rdf-fma-snomed-v1.md` §1-§3 |
| W3 | PR-H2a FMA OWL full ingest pipeline (Tier-2) | `.claude/specs/pr-h2a-fma-owl-ingest.md` | `anatomy-realtime-v1.md` + `lance-graph-rdf-fma-snomed-v1.md` |
| W4 | PR-H2b FMA CSV quick-bootstrap ingest (Tier-1) | `.claude/specs/pr-h2b-fma-csv-quick.md` | same |
| W5 | PR-H3 q2 graph-notebook Cypher cell wire | `.claude/specs/pr-h3-q2-cypher-cell.md` | `q2-foundry-integration-v1.md` + sprint-4 W11 spec |
| W6 | PR-H4 heart-click integration test (5 golden inputs) | `.claude/specs/pr-h4-heart-click-test.md` | `anatomy-realtime-v1.md` + Pillar 6 EWA-Sandwich |
| W7 | PR-H5 SIMD callcenter batch retrofit (vsa_udfs.rs) | `.claude/specs/pr-h5-simd-callcenter.md` | sprint-4 W5 spec + `ndarray::hpc::{vsa,bitwise}` |
| W8 | PR-H6 MedCare drug-knowledge-2026-05-05 crosswalk | `.claude/specs/pr-h6-drug-knowledge-crosswalk.md` | `lance-graph-rdf-fma-snomed-v1.md` (named-graph pattern) |
| W9 | SNOMED-CT companion ingest (license-gated) | `.claude/specs/pr-h7-snomed-ingest.md` | `lance-graph-rdf-fma-snomed-v1.md` |
| W10 | RadLex companion ingest | `.claude/specs/pr-h8-radlex-ingest.md` | same |
| W11 | Cross-ontology context-id coherence test | `.claude/specs/sprint-7-context-id-coherence.md` | `lance-graph-rdf-fma-snomed-v1.md` |
| W12 | Sprint-7 retrospective + sprint-8 handover | `.claude/specs/sprint-7-retrospective.md` | sprint-6 retrospective |
| M1 | Per-worker brutal review | `.claude/board/sprint-log-7/meta-1-review.md` | (reads all worker outputs) |
| M2 | Cross-spec synthesis + FMA demo coherence audit | `.claude/board/sprint-log-7/meta-2-review.md` | same |

**Estimated total LOC:** ~2450 (excludes SNOMED/RadLex which are license-gated).

---

## Sprint 8 — TTL namespaces + compliance certification (2 weeks)

**Goal:** address D-SDR-6..D-SDR-39 explicitly deferred by PR #363. Hardens product for customer compliance audits.

**Worker roster:**

| W | Deliverable | Output file | Prior plan to extend |
|---|---|---|---|
| W1 | Sprint-8 execution plan + compliance scope decision | `.claude/specs/sprint-8-execution-plan.md` | sprint-7 retrospective |
| W2 | PR-I1 TTL namespace registry + runtime overlay | `.claude/specs/pr-i1-ttl-namespaces.md` | `super-domain-rbac-tenancy-v1.md` §6 + `lance-graph-ontology-v5.md` |
| W3 | PR-I2 HIPAA compliance surface (audit replay + BAA export) | `.claude/specs/pr-i2-hipaa-cert.md` | `super-domain-rbac-tenancy-v1.md` §16 (Zone 3) |
| W4 | PR-I3 SOX compliance (dual-control + financial segregation) | `.claude/specs/pr-i3-sox-cert.md` | same §16 |
| W5 | PR-I4 GDPR compliance (right-to-erasure + tombstones) | `.claude/specs/pr-i4-gdpr-cert.md` | same §16 |
| W6 | PR-I5 OSINT compliance + LanceProbe M5/M6 unblock | `.claude/specs/pr-i5-osint-cert.md` | `super-domain-rbac-tenancy-v1.md` §18 + `2026-05-06-splat-osint-ingestion-v1.md` |
| W7 | PR-I6 Federation Phase 2 (LanceDB transparent encrypted view) | `.claude/specs/pr-i6-federation-phase2.md` | `super-domain-rbac-tenancy-v1.md` §13 (federation A+B+C) |
| W8 | Audit replay + verify CLI deep-dive | `.claude/specs/pr-i2b-audit-verify-cli.md` | sprint-5 W8 |
| W9 | Encryption-at-rest gate (Argon2 backfill on login) | `.claude/specs/pr-i2c-encryption-at-rest.md` | `super-domain-rbac-tenancy-v1.md` §18 (Argon2 correction) |
| W10 | BAA-ready audit export schema + redaction surface | `.claude/specs/pr-i2d-baa-export-schema.md` | sprint-8 W3 |
| W11 | Cross-regime compliance test matrix (HIPAA × SOX × GDPR × OSINT) | `.claude/specs/sprint-8-compliance-matrix.md` | sprint-8 W3..W6 |
| W12 | Sprint-8 retrospective + sprint-9 handover | `.claude/specs/sprint-8-retrospective.md` | sprint-7 retrospective |
| M1 | Per-worker brutal review | `.claude/board/sprint-log-8/meta-1-review.md` | (reads all worker outputs) |
| M2 | Cross-spec synthesis + compliance coherence audit | `.claude/board/sprint-log-8/meta-2-review.md` | same |

**Estimated total LOC:** ~3600. **Out-of-scope:** ITAR-EAR, PCI-DSS (deferred until customer asks).

---

## Sprint 9 — Q2 cockpit + holographic cinematic (1-2 weeks, parallelizable earlier)

**Goal:** ship the holographic UX + the WOW cinematic. Sales-asset budget; can run parallel to any sprint 5-8.

**Worker roster:**

| W | Deliverable | Output file | Prior plan to extend |
|---|---|---|---|
| W1 | Sprint-9 execution plan + storyboard kickoff | `.claude/specs/sprint-9-execution-plan.md` | sprint-8 retrospective |
| W2 | PR-J1 q2 frontend Gaussian-splat fragment shader (WGSL/GLSL) | `.claude/specs/pr-j1-q2-splat-shader.md` | `q2-foundry-integration-v1.md` + sprint-4 IDEAS sci-fi entry |
| W3 | PR-J2 RenderFrame highlight column + Pillar 6 write-back | `.claude/specs/pr-j2-renderframe-highlight.md` | `ndarray/src/hpc/renderer.rs` + `crates/jc/src/ewa_sandwich.rs` |
| W4 | PR-J3 FMA canonical-pose seeder (T-pose, head up, arms out) | `.claude/specs/pr-j3-fma-pose-seeder.md` | sprint-7 PR-H2 outputs |
| W5 | PR-J4 per-system layer toggle (skeletal/cardio/nervous) | `.claude/specs/pr-j4-layer-toggle.md` | `super-domain-rbac-tenancy-v1.md` §6 (family slice filter) |
| W6 | PR-J5 fma-cinematic offline prerender tool + Lance schema | `.claude/specs/pr-j5-fma-cinematic-tool.md` | sprint-4 IDEAS execution-path entry (16-color LanceDB) |
| W7 | PR-J6 q2 cinematic player (AVX2 nibble unpack + canvas blit) | `.claude/specs/pr-j6-q2-cinematic-player.md` | same |
| W8 | PR-J7 release artifact bundle (`intro-v1.lance`, ~150 MB) | `.claude/specs/pr-j7-release-artifact.md` | bgz7 release pattern (`v0.1.0-bgz-data`) |
| W9 | Camera trajectory storyboard for intro-v1 (30s @ 30fps) | `.claude/specs/sprint-9-intro-storyboard.md` | sprint-9 W1 |
| W10 | Color palette curation (cyan-demoscene-v1 16-color) | `.claude/specs/sprint-9-palette-curation.md` | sprint-9 W6 |
| W11 | Audio cue layer (Web Audio API on highlight peak) | `.claude/specs/sprint-9-audio-cues.md` | sprint-9 W3 |
| W12 | Sprint-9 retrospective + demo recording script | `.claude/specs/sprint-9-demo-script.md` | sprint-8 retrospective |
| M1 | Per-worker brutal review | `.claude/board/sprint-log-9/meta-1-review.md` | (reads all worker outputs) |
| M2 | Cross-spec synthesis + sales-asset coherence audit | `.claude/board/sprint-log-9/meta-2-review.md` | same |

**Estimated total LOC:** ~1700. **Owner:** demo / marketing budget, NOT engineering critical path.

---

## Cumulative sizing

| Sprint | Workers | Calendar | Est. PR LOC | Critical path? |
|---|---|---|---|---|
| Sprint 5 | 12 + 2 meta | 1 week | ~1350 | Yes — blocks downstream |
| Sprint 6 | 12 + 2 meta | 2 weeks | ~3300 | Yes — Tier-2 wiring |
| Sprint 7 | 12 + 2 meta | 1 week | ~2450 | Yes — FMA proof-of-vision |
| Sprint 8 | 12 + 2 meta | 2 weeks | ~3600 | Customer compliance dep |
| Sprint 9 | 12 + 2 meta | 1-2 weeks | ~1700 | Parallelizable any time |

**Critical-path total (5+6+7):** 4 weeks, ~7100 LOC, 36 workers + 6 meta.
**Total roadmap:** 70 agents (60 workers + 10 meta), ~12,400 LOC, 7-9 calendar weeks.

---

## Open questions for human reviewer

1. **Sprint 6 PR-E3..E5 (woa-rs / hiro-rs / hubspot-rs new repos):** separate `adaworldapi/<name>-rs` repos vs subcrates of an existing one? Affects CI matrix + worker W4/W5/W6 prompts.
2. **Sprint 8 priority:** does customer pipeline contain a HIPAA-blocking deal that pulls sprint 8 forward of sprint 7? If yes, swap.
3. **Sprint 9 budget approval:** demo team have ~1 week to spend, or held until sales asks?
4. **MedCare-rs drug-knowledge-2026-05-05:** sprint-7 PR-H6 must-have or sprint-8+ enrichment?
5. **Sprint 5 PR-D5 (compat shim):** consumer repos already migrated or do they require coordinated cutover? Affects PR-A acceptance.

---

## Mandatory governance updates per merged PR

Every PR must update (in the same commit, per `CLAUDE.md` board-hygiene rule):
- `.claude/board/LATEST_STATE.md` — contract inventory + shipped table
- `.claude/board/PR_ARC_INVENTORY.md` — PREPEND with Added/Locked/Deferred/Docs/Confidence
- `.claude/board/STATUS_BOARD.md` — D-id row status flip
- `.claude/board/AGENT_LOG.md` — if agent-driven, PREPEND completion entry
