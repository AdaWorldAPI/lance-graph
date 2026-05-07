# Integration Plans — Versioned Index

> **APPEND-ONLY.** Every integration plan ever authored for this
> workspace, versioned, with status. New plans append to the top
> as new entries. Superseded plans stay — they are the design arc.
>
> Governance: same rule as `PR_ARC_INVENTORY.md`. The **Status**
> field is the only mutable field per entry. Supersedure is marked
> by adding a new top entry that references the prior version; the
> prior entry is NOT deleted.

---

## APPEND-ONLY RULE

1. **New plans PREPEND** a new section at the top.
2. **Old plan entries are IMMUTABLE** except the **Status** line.
3. **Supersedure:** if plan vN is replaced by vN+1, prepend a new
   entry for vN+1 that cites vN; update vN's **Status** to
   "Superseded by vN+1".
4. **Corrections** to plan scope during its lifetime: append a
   `**Correction (YYYY-MM-DD):**` line to the entry; do not edit
   the original scope line.
5. **Retire but never delete.** When a plan is complete or
   abandoned, update Status and move on. The entry stays.

**Per-entry format:**

- **Plan name + version**
- **Author + date**
- **Scope** — one-sentence goal (immutable)
- **Path** — workspace file location
- **Deliverables** — D-id list (immutable)
- **Status** — **mutable**: Active / Shipped / Superseded / Deferred / Abandoned
- **Confidence** — **mutable**: Working / Partial / Broken — see PR #N


---

## palantir-parity-cascade-v2 — Foundry/Gotham parity capstone + DTO ladder (authored 2026-05-07)

- **Plan:** `.claude/plans/palantir-parity-cascade-v2.md`
- **Companion knowledge:** `.claude/knowledge/soa-dto-dependency-ledger.md` (the SoA DTO entropy ledger; ships with this plan).
- **Author + date:** main thread (Opus 4.7 1M), 2026-05-07 (immediately after PR #352 merge).
- **Status:** Active.
- **Scope:** Integration capstone over 4 prior Foundry parity docs (`q2-foundry-integration-v1`, `lf-integration-mapping-v1`, `foundry-consumer-parity-v1`, `medcare-foundry-vision`) and v1 cascade Pillar 0. **Pillar 0 carry-forward**: Foundry parity IS SoA-as-canon parity — Column H (`EntityTypeId: u16`, PR #272 SHIPPED) is already the Foundry Object Type bridge; v2 just makes the SoA carry the Foundry-equivalent shape, NOT duplicate the table set. **DTO ladder finding (2026-05-07 audit)**: `StreamDto`, `ResonanceDto`, `BusDto` all live in `thinking-engine::dto.rs` (Tiers 0/1/2), upstream of contract. 22 DTOs classified across 3 buckets: 9 bare-metal, 7 SoA-glue, 6 bridge-projection (3 OPEN re-classifications). **Business Logic ↔ Thinking-style ↔ OGIT triangle**: each business operation has 3 faces (`thinking_style: ThinkingStyle` dispatch, `ogit_verb: TTL`, `ogit_entities[]: TTL`); v2 D-PARITY-V2-2 ships the routing table.
- **Originating context:** main-thread requests 2026-05-07: (a) updated roadmap with Foundry/Gotham parity; (b) SoA DTO dependency-graph / entropy ledger to classify bare-metal vs SoA-glue (StreamDto, BusDto, ResonanceDto); (c) cognitive-shader-driver internal vs lance-graph-callcenter external O(1) mapping; (d) Business Logic Thinking-style OGIT mapping later.
- **Resolves ledger rows:** none directly. **Hardens** v1 D-CASCADE-V1-7 (codec cascade column population) via the explicit ledger entry tracking the "OPEN" status of each cascade column.
- **Branch:** `claude/create-graph-ontology-crate-gkuJG`. PR target: `AdaWorldAPI/lance-graph` base=`main`.
- **Confidence (2026-05-07):** Pre-execution. Pillar 0 carry-forward is right per existing PR #272 (Column H is the bridge already). Top-3 ranked: D-PARITY-V2-1 (DTO ledger — ships with this plan), D-PARITY-V2-2 (triangle ledger — ships with this plan), D-PARITY-V2-3 (BusDto bridge into engine_bridge.rs).
- **Cross-plan deps:** v1 D-CASCADE-V1-2 (`SchemaPtr.context_id`) → v2 D-PARITY-V2-4 (`Schema::ObjectView`); v1 D-CASCADE-V1-7 (codec cascade columns) → v2 D-PARITY-V2-12 (`SchemaPtr.thinking_style`); v5 D-9 (`MulThresholdProfile`) → v2 D-PARITY-V2-12 (column extension).
- **Foundry parity status snapshot:** SHIPPED — Column H (PR #272), audit trail, RBAC/RLS, PostgREST. IN PROGRESS — Q2 cockpit. QUEUED — LF-12 Pipeline DAG, LF-20 FunctionSpec, LF-22/23 ObjectView/Notification, LF-50 ModelRegistry.
- **Out of v2 scope:** CRDT scenario branching (Column F already exists; UI affordance is v3), Foundry Marketplace/Compass, Foundry Code Repositories, Vertex/Workshop UX (covered by `q2-foundry-integration-v1`), Foundry-export-format ingest.

---

## ogit-cascade-supabase-callcenter-v1 — OGIT SPO-G + Supabase realtime + Zone 1/2/3 (authored 2026-05-07)

- **Plan:** `.claude/plans/ogit-cascade-supabase-callcenter-v1.md`
- **Author + date:** main thread (Opus 4.7 1M), 2026-05-07.
- **Status:** Active.
- **Scope:** 15 deliverables across `lance-graph-callcenter`, `lance-graph-ontology`, AdaWorldAPI/OGIT (extension fork), and a future `lance-graph-rdf` consumer. Pillar 0 (the holy-grail click): `OntologyRegistry` IS the SoA; per-domain schema (Healthcare, WorkOrder, SMB, CallCenter, Medical) IS the DTO + name→row index. Codec cascade per row: identity Vsa16kF32 → CAM-PQ 6 B → Base17 34 B → palette key 4 B → Scent 1 B → qualia/meta/edge columns. Every step O(1). Pillar 1: OGIT as universal SPO-G lingua franca with `ontology_context_id: u32` per named graph. Pillar 2: Zone 1 (BindSpace, no Serialize) / Zone 2 (Arrow scalar membrane, BBB invariant) / Zone 3 (Supabase RPC, REST, transcode — the only emission point). Pillar 3: smb-bridge + medcare-bridge collapse to 2-line projections over `OntologyRegistry::enumerate(ns)`. Pillar 4: BioPortal arsenal — 10 namespace stubs under `OGIT/NTO/Medical/{ICD10CM,RxNorm,LOINC,FMA,RadLex,SNOMED,MONDO,HPO,DRON,CHEBI}/` carrying provenance + license + size, with full ingestion gated on `lance-graph-rdf-fma-snomed-v1`.
- **Originating context:** main-thread question 2026-05-07: *"should the lance-graph-ontology be the SoA and the schema the DTO + index?"* — answered YES, with the codec cascade chain making it content-addressable through every encoding tier (the holy grail). User-supplied references: `MedCare-rs/.MYSQL/Struktur.sql` (104 tables, 5 dominant prefixes) and `MedCare-rs/releases/tag/bioportal-ontologies-2026-05-05` (25 bundles, ~2.4 GB).
- **Resolves ledger rows:** none directly. **Hardens** v5's D-9 (`MulThresholdProfile` becomes `ontology_context_id`-aware, so medical thresholds are stricter than callcenter thresholds). **Locks down** the BBB membrane doctrine from `callcenter-membrane-v1.md` § 10.9 with a `cert-officer` static check (D-CASCADE-V1-1).
- **Branch:** `claude/create-graph-ontology-crate-gkuJG` (continues the v4/v5 thread). PR target: `AdaWorldAPI/lance-graph` base=`main`. OGIT-fork PRs land under the same branch on the OGIT-fork side.
- **Confidence (2026-05-07):** Pre-execution. Pillar 0 is the only architectural commitment that admits no rollback — and it is right per the existing `LazyLock<&OntologyRegistry>` pattern in `lance-graph-ontology/src/bridges/`. Top-3 ranked: D-CASCADE-V1-1, D-CASCADE-V1-2, D-CASCADE-V1-3 (no upstream blockers).
- **Cross-plan deps:** v5 D-9 (`MulThresholdProfile`), `lance-graph-rdf-fma-snomed-v1` (`SemanticQuad`), `supabase-subscriber-v1` (DM-4 watcher / DM-6 drain), `callcenter-membrane-v1` § 10.9 (BBB iron rule).
- **Out of v1 scope (deferrals):** full SNOMED CT import (license-gated; BioPortal release ships only 666 KB partial), full DRON / CHEBI import (size unclear-payoff; revisit after D-CASCADE-V1-11 measures cascade), n8n-rs / crewai-rust consumption of new SoA columns (separate plan), bgz-tensor attention layer integration (orthogonal).

---

## lance-graph-ontology-v5 — post-merge follow-ons (authored 2026-05-07)

- **Plan:** `.claude/plans/lance-graph-ontology-v5.md`
- **Author + date:** integration-lead (Opus 4.7 1M), 2026-05-07
- **Status:** Active
- **Scope:** Picks up where v4 (`claude/create-graph-ontology-crate-gkuJG`, OGIT#1 merged) left off. 15 deliverables ranked by leverage / cost: D-ONTO-V5-1 (dcterms:source provenance, closes TTL-PROBE-5), D-ONTO-V5-2 (`arigraph::SpoBridge::promote_to_spo`, closes SPO-1), D-ONTO-V5-3 (Healthcare TTL transcode), D-ONTO-V5-4 (smb-ontology export-only, NOT migration — brutal-honest reversal, ratified by main thread 2026-05-07), D-ONTO-V5-5 (q2 TTL transcode), D-ONTO-V5-6/7 (MySQL/MSSQL `SchemaSource` impls), D-ONTO-V5-8 (customer admin form, owned by woa-rs surface), D-ONTO-V5-9 (ontology-aware MUL trust thresholds — registry as namespace-keyed lookup), D-ONTO-V5-10 (callcenter-bridge, deferred until SUBJECT-DTO-1 lands), D-ONTO-V5-11 (woa-rs 80/20 binary cut), D-ONTO-V5-12 (MUL publishers — Brier/damage/sandbox), D-ONTO-V5-13 (hydration parallelism), D-ONTO-V5-14 (Lance dictionary load probe), D-ONTO-V5-15 (in-memory → Lance-backed cutover).
- **Originating context:** v4 OGIT#1 merge (15 entities + 12 verbs in `NTO/WorkOrder/`, master); 36 ontology tests pass; cognitive-shader-driver wired (read-only registry attachment).
- **Resolves ledger rows:** TTL-PROBE-5 (D-ONTO-V5-1), SPO-1 (D-ONTO-V5-2 70+245). Partial leverage on MUL-ASSESS-1 (registry as namespace-keyed threshold table). No leverage on TRUST-1 / FLOW-1 / COMPASS-1 / PARSER-1 (out of scope; the ontology crate has no influence on enum consolidation or the cypher cold/hot split).
- **Branch:** `claude/onto-v5-<D-id>` per deliverable; OGIT-fork PRs per namespace transcode. Upstream `almatoai/OGIT` is never PR'd (ratified 2026-05-07).
- **Confidence (2026-05-07):** Pre-execution. Plan reviews v4's outputs as FINDING-grade and v5's deferrals as honestly-deferred (not punted). Next-3 ranked: D-ONTO-V5-1, D-ONTO-V5-9, D-ONTO-V5-2.
- **Cross-ref:** `.claude/RECON_ONTOLOGY_CRATE.md`, `.claude/DECISION_SPO_ARIGRAPH.md`, `.claude/knowledge/ontology-registry.md`, `sql-spo-ontology-bridge-v1.md` (partially superseded), `foundry-roadmap-unified-smb-medcare-v1.md` (adjacent).
- **Ratifications (main-thread, 2026-05-07):** Q1 smb-ontology export-only — RATIFIED (consistent with v4 "preserved as native fallback"; not a contradiction). Q2 D-9 above D-2 ordering — RATIFIED (registry has zero behavioral consumer until V5-9 lands; SPO L1/L2 cache works without the bridge fn today). Q3 `MulThresholdProfile` location — RATIFIED in `lance-graph-contract` (zero-dep canonical home; co-located with `MulAssessment`). Q4 OGIT-fork upstream PR rule — RATIFIED (AdaWorldAPI/OGIT extension fork only; never PR back to almatoai/OGIT).

---

## splat-osint-ingestion-v1 — Splat contract + EWA OSINT bridge (authored 2026-05-06)

- **Plan:** `.claude/plans/2026-05-06-splat-osint-ingestion-v1.md`
- **Author + date:** Claude (for Jan), 2026-05-06
- **Status:** Active (PR 1+2 of 6 in flight)
- **Scope:** SPLAT-1 ledger row Aspirational -> Wired (x1). Materialise SplatChannel/CamPlaneSplat/SplatPlaneSet/CamSplatCertificate in lance-graph-contract; demonstrate EWA-sandwich Sigma-push-forward as neo4j-edge-traversal substitute via crates/jc/examples/osint_edge_traversal.rs.
- **Originating question:** q2 PR #35 review
- **Resolves ledger rows:** SPLAT-1 (entropy 4 -> 2, Aspirational -> Wired stage 1).
- **Branch:** claude/splat-osint-ingestion
- **Confidence (2026-05-06):** Working (math certified by Pillar 6 PR #289).

---

## v1 — Grammar + Foundry Follow-up (authored 2026-04-29)

**Author:** main thread (Opus 4.7), session 2026-04-29
**Status:** Active
**Scope:** Wire the stubs and scaffolds shipped in PRs #275-#283 to existing tissue. Six explicit `stub`/`skeleton`/`placeholder`/`unimplemented!` markers in the merged code (verified by grep) name what remains. 13 PRs across two parallel tracks (6 Foundry + 6 Grammar) sharing one keystone (LF-12 Pipeline DAG). All deliverables target `main` directly; no stacking PRs (avoids the merge-order orphaning that bit #281/#283 → #284/#285).
**Path:** `.claude/plans/grammar-foundry-followup-v1.md`
**Deliverables:** PR-S1 (Pipeline DAG keystone), PR-F1..F6 (Foundry: PolicyRewriter UDF wrap, Encrypt+DP, Lance audit, PostgREST dispatch, audit_from_plan, dn_path scent), PR-G1..G6 (Grammar: Triangle causality, Disambiguator wiring, ContextChain fp, verb_table seed, AriGraph unbundle, Animal Farm real run).
**Cross-refs:**
- `lf-integration-mapping-v1.md` — LF-12 keystone rationale (PR-S1)
- `foundry-roadmap.md` — original PR-1..PR-5 (PR-1/PR-2 shipped as #278/#280; PR-3..PR-5 ship as PR-F1..F4 here)
- `integration-plan-grammar-crystal-arigraph.md` — original AriGraph follow-up (now ships as PR-G5)
- `grammar-landscape.md` — case inventories that PR-G4 consumes
**Open decisions:** (1) PR-F2 encryption key management (KMS? in-process? user-supplied?); (2) PR-G6 Animal Farm text licensing; (3) PR-F6 bgz-tensor → callcenter dep; (4) PR-G4 ownership.

---

## v1 — LF Integration Mapping (authored 2026-04-25)

**Author:** main thread (Opus 4.7 1M), session 2026-04-25 (branch claude/scenario-world-facade)
**Status:** Active
**Scope:** Comprehensive mapping of all 41 LF + 4 W chunks shipped or queued across the lance-graph workspace. Mirrors the SMB-side foundry-parity-checklist; producer-side companion. Documents Tier 1 (8/8 LF + 4/4 W shipped) + Tier 2 (28 chunks across 8 stages, ~38% shipped, sequencing for next 10 chunks). Includes Stage 7 redesign notes (LF-71 column rejected; LF-73/74/75 added wiring NARS counterfactual / Chronos-method palette forecast / Apache-Temporal-method deterministic replay).
**Path:** `.claude/plans/lf-integration-mapping-v1.md`
**Companions:** `.claude/agents/scenario-world.md`, `docs/ScenarioWorldCounterfactual.md`
**Cross-ref:** `smb-office-rs/docs/foundry-parity-checklist.md` (consumer mirror)

---

## v1 — Q2 Foundry-Equivalent Integration (authored 2026-04-24)

**Author:** main thread (Opus 4.7 1M), session 2026-04-24
**Status:** Proposed
**Scope:** Q2 = user interface (Gotham/Workshop/Vertex equivalent) + SMB = first tenant testbed. 4 phases: demo-able → operational → intelligent → fly. Firefly stack: Ballista + Dragonfly + GEL.
**Path:** `.claude/plans/q2-foundry-integration-v1.md`
**Deliverables:** Q2-1.1..1.7 (Phase 1 MVP), Q2-2.1..2.7 (Phase 2 workflows), Q2-3.1..3.7 (Phase 3 reasoning), Q2-4.1..4.7 (Phase 4 scale). 28 deliverables total.
**Foundation:** 8 PRs merged this session (#253-#260) provide substrate.
**Key differentiators vs Palantir Foundry:** active inference as dispatch, NARS truth as primary data, CausalEdge64 Pearl 2³ masks, JIT-compiled lenses, zero-dep contract crate.

---

## v1 — Supabase Subscriber Wire-up (authored 2026-04-24)

**Author:** sonnet agent, session 2026-04-24 (branch claude/supabase-subscriber-wire-up)
**Scope:** Flip `LanceMembrane::subscribe()` from Phase-A stub to a live `tokio::sync::watch::Receiver<CognitiveEventRow>` wired to `LanceVersionWatcher`; ship `DrainTask` scaffold.
**Path:** `.claude/plans/supabase-subscriber-v1.md`
**Deliverables:** DM-4a swap Subscription type, DM-4b `version_watcher.rs`, DM-4c uncomment `pub mod version_watcher`, DM-6a `drain.rs` scaffold, DM-6b uncomment `pub mod drain`.

**Status (2026-04-24):** In PR. All deliverables in branch `claude/supabase-subscriber-wire-up`.

**Confidence (2026-04-24):** FINDING — 17 tests pass (13 without realtime, 17 with; 4 new tests in `version_watcher.rs`, 1 new `subscribe_receives_on_project` in `lance_membrane.rs`). Zero regressions.

**Correction (2026-05-06):** The `tokio::sync::watch::Receiver` choice violates I-2 (tokio outbound only) per `SINGLE_BINARY_TOPOLOGY.md`. Sync substitute is `ArcSwap<u64>` + `event_listener::Event`, polled on a `std::thread`. WATCHER-1 entropy-ledger row carries the corrected spec.

---

## v1 — Unified Integration: PersonaHub × ONNX × Archetype × MM-CoT × RoleDB (authored 2026-04-23)

**Author:** main-thread session 2026-04-23
**Scope:** Integrate four upstream systems (PersonaHub compression, ONNX persona classifier @ L4/L5, Archetype ECS adapter, MM-CoT stage split) into the lance-graph cognitive substrate without adding new architectural layers — each maps onto existing contract types.
**Path:** `.claude/plans/unified-integration-v1.md`
**Deliverables:** DU-0 PersonaHub 56-bit compression, DU-1 ONNX persona classifier (replaces Chronos proposal), DU-2 Archetype ECS bridge crate, DU-3 RoleDB DataFusion VSA UDFs, DU-4 MM-CoT `rationale_phase: bool` in `CognitiveEventRow`, DU-5 board hygiene.

**Status (2026-04-23):** Active. No deliverables shipped yet. Plan written and committed (commit `468357d`). Architectural ground truth in `callcenter-membrane-v1.md` §§ 15–17.

**Confidence (2026-04-23):** CONJECTURE — all integration mappings grounded in repo evidence and upstream docs; no code shipped beyond plan.

**Correction (2026-04-23):** Chronos (Amazon) proposal superseded by ONNX classifier for DU-1. Chronos predicts 1D style scalar; ONNX classifier predicts full 288-class `(ExternalRole × ThinkingStyle)` product. ONNX infra already justified by Jina v5 ONNX on disk.

---

## v1 — Callcenter Membrane: Supabase-shape over Lance + DataFusion (authored 2026-04-22)

**Author:** main-thread session 2026-04-22
**Scope:** Assimilate the design and ergonomics of the Supabase callcenter surface into a new crate (`lance-graph-callcenter`) that sits entirely outside the canonical cognitive substrate, backed by Lance + DataFusion, enforcing the BBB (blood-brain barrier) at compile time via the Arrow type system — Phoenix channel realtime + PostgREST query surface without PostgreSQL.
**Path:** `.claude/plans/callcenter-membrane-v1.md` (254 lines)
**Deliverables:** DM-0 `ExternalMembrane` + `CommitFilter` in contract, DM-1 callcenter crate skeleton, DM-2 `LanceMembrane::project()` + compile-time leak test, DM-3 `CommitFilter → DataFusion Expr`, DM-4 `LanceVersionWatcher`, DM-5 `PhoenixServer`, DM-6 `DrainTask`, DM-7 `JwtMiddleware + RLS rewriter`, DM-8 `PostgRestHandler`, DM-9 end-to-end test.

**Status (2026-04-22):** Active. DM-0 and DM-1 shipped in this session. DM-2 through DM-9 queued.

**Confidence (2026-04-22):** CONJECTURE on the full architecture (grounded in Arrow BBB analysis + repo evidence; no DM-2+ implementation shipped). DM-0/DM-1 are working stubs; Arrow compile-time BBB enforcement verified structurally, awaiting DM-2 compile-time leak test.

**Correction (2026-05-06):** The framing "callcenter sits *outside* the canonical cognitive substrate" was read by some sessions as "separate process". Per `SINGLE_BINARY_TOPOLOGY.md`, callcenter is in-process Layer 2, sync, zero-copy over Layer 1 BindSpace. DM-5 / DM-8 are the only L3 (post-tokio) components in this plan.

---

## v1 — Categorical-Algebraic Inference (authored 2026-04-21)

**Author:** main-thread session 2026-04-21
**Scope:** Meta-architecture document proving that parsing (Kan extension), disambiguation (free-energy minimization), learning (NARS revision), memory (AriGraph commit), and awareness (method-call history) are one algebraic operation — element-wise XOR on role-indexed slices of a 10K binary VSA vector — viewed through five lenses. Grounded in Shaw 2501.05368 (category theory) + 13 supporting papers. Does not replace elegant-herding-rocket — extends it with the categorical foundation.
**Path:** `.claude/plans/categorical-algebraic-inference-v1.md` (496 lines)
**Deliverables:** This plan produces no NEW D-ids. It grounds the existing D2/D3/D5/D7/D8/D10 deliverables from elegant-herding-rocket in the categorical-algebraic framework and establishes the five-lens litmus + object-does-the-work test as architectural invariants.

**Status (2026-04-21):** Active. Companion to elegant-herding-rocket-v1, not a replacement.

**Confidence (2026-04-21):** CONJECTURE on the Kan-extension-IS-free-energy equivalence. FINDING on all other claims (grounded in shipped code + paper proofs).

---

## v1 — Codec Sweep via Lab Infra, JIT-first (authored 2026-04-20)

**Author:** main-thread session 2026-04-20
**Scope:** Operationalise PR #220's "What's Needed to Fix" list (wider codebook / residual PQ / Hadamard pre-rotation / OPQ) as a parameter sweep through the lab endpoint, with every codec candidate difference expressed as a JIT-compiled kernel rather than a cargo rebuild — one upfront API hardening rebuild, unlimited candidates afterwards.
**Path:** `.claude/plans/codec-sweep-via-lab-infra-v1.md` (396 lines)
**Deliverables:** D0.1 `CodecParams` in `WireCalibrate`, D0.2 `WireTokenAgreement` endpoint (I11 cert gate), D0.3 `WireSweep` streaming endpoint + Lance append, D0.4 surface freeze. D1.1 `CodecKernelCache` via `JitCompiler`, D1.2 rotation primitives (Identity / Hadamard / OPQ) as JIT kernels, D1.3 residual PQ via JIT composition. D2.1 reference-model loader, D2.2 decode-and-compare loop, D2.3 handler wiring. D3.1 server-side sweep handler, D3.2 curl-driven client. D4.1 DataFusion over Lance log, D4.2 Pareto frontier notebook. D5 graduation bridge (fires only on candidate passing all gates).

**Status (2026-04-20):** Active. Plan authored; no deliverables shipped yet. Depends on merge of PR #224 (three-part lab-surface framing + I11 measurability invariant) for the architectural grounding.

**Confidence (2026-04-20):** Pre-execution. Risk hot-spots: (a) JIT compile cost for residual PQ composition — needs measurement; (b) token-agreement harness load time on ref model — may dominate latency for small sweeps; (c) Lance append concurrency under streaming writes. Plan assumes these are tractable; D0 surface freeze is deliberate to prevent iterating on the DTO shape mid-sweep.

---

## v1 — Elegant Herding Rocket (authored 2026-04-19)

**Author:** main-thread session 2026-04-19
**Scope:** DeepNSM as full parser via Grammar Triangle wiring + Markov ±5 SPO+TEKAMOLO bundling + NARS-tested grammar thinking styles + coreference resolution + story-context bridge + ONNX arc emergence.
**Path:** `.claude/plans/elegant-herding-rocket-v1.md` (2,085 lines)
**Deliverables:** D0 landscape doc, D2 FailureTicket emission, D3 Triangle bridge, D4 ContextChain reasoning, D5 Markov ±5 bundler, D6 role keys, D7 grammar thinking styles, D8 story context + contradictions, D9 ONNX arc export, D10 Animal Farm validation harness, D11 bundle-perturb emergence.

**Status (2026-04-19):** Active. Phase 1 (D0 + D4 + D6) shipped in PR #210.

**Confidence (2026-04-19):** Phase 1 working (125 tests passing).
Phases 2–4 queued.

**Phases:**

- **Phase 1 — SHIPPED** (PR #210, merged): D0 landscape doc + D4
  ContextChain reasoning ops + D6 role keys. 125 tests passing.
- **Phase 2 — QUEUED:** D2 FailureTicket emission + D3 Triangle
  bridge + D5 Markov bundler + D7 grammar thinking styles.
  Estimate ~930 LOC, one PR.
- **Phase 3 — QUEUED:** D8 story-context/contradictions + D10
  Animal Farm validation harness.
- **Phase 4 — FUTURE:** D9 ONNX arc export + D11 bundle-perturb
  emergence interface.

---

## How to use this file

1. **Starting a new session:** check top entry. If Status is Active,
   that's the current plan. Read it at
   `.claude/plans/<plan-file>.md`.
2. **Proposing a new plan:** prepend a new v entry; move prior
   plan's Status to Superseded.
3. **Tracking deliverable progress:** use
   `.claude/board/STATUS_BOARD.md` for the cross-deliverable
   view (which D-ids are in which phase / PR).
4. **User requests / open threads** that aren't yet a plan: capture
   in `.claude/knowledge/OPEN_PROMPTS.md`.

## Cross-references

- **`SINGLE_BINARY_TOPOLOGY.md`** — canonical architecture reference
  (three layers, four invariants: single-binary, tokio-outbound-only,
  BBB compile-time-enforced, per-row vs per-cadence gates distinct).
  **READ FIRST** before proposing any new "membrane" / "transcode" /
  "subscriber" / "external surface" plan.
- **`STATUS_BOARD.md`** — deliverable-level status (D0 / D2 / D3 / …
  across all plans).
- **`OPEN_PROMPTS.md`** — outstanding user questions / threads that
  aren't yet scoped into a plan.
- **`PR_ARC_INVENTORY.md`** — shipped-PR decision history.
- **`LATEST_STATE.md`** — current-state snapshot.

## 2026-04-20 — cam-pq-production-wiring-v1
**Status:** DRAFT
**Plan:** `.claude/plans/cam-pq-production-wiring-v1.md`
**Scope:** Wire CAM-PQ as default codec for argmax-regime tensors.
**Deliverables:** D1-D7 (classifier, calibration, storage, decode, validation, E2E, fallback).
**Driver:** ICC 0.9999 at 6 B/row on Qwen3-8B (PR #218 bench).
**Effort:** ~8 person-days.
**Confidence:** HIGH.

---

## v1 — BindSpace Columns E/F/G/H (authored 2026-04-26)

**Author:** main thread (Opus 4.7 1M), session 2026-04-26
**Status:** Active
**Scope:** Extend BindSpace SoA from 4 → 8 column families. Column H (EntityTypeId, Foundry Object Type). Column E (OntologyDelta, per-cycle structural learning). Column F (AwarenessColumn, BF16-mantissa-inline per-word epistemic annotation). Column G (ModelRef, ONNX style_oracle binding). Total overhead +5.9% per row (6212→6578 bytes), still fits L3 cache.
**Path:** `.claude/plans/bindspace-columns-v1.md`
**Companions:** EPIPHANIES.md 2026-04-26 (4 entries), TD-AWARENESS-INLINE-1, TD-PALETTE-SENTINEL
**Scientific review:** 7 SOUND, 7 CAUTION, 0 WRONG (Jirak/Pearl/NARS/Kleyko/Shaw cross-check)
**Deliverables:** D-H1..4 (Phase 1), D-E1..6 (Phase 2), D-F1..9 (Phase 3), D-G1..5 (Phase 4). 24 total.
**Cross-ref:** LF integration mapping v1 (Stages 2/5/6), Q2 Foundry plan (Vertex parity), soa-review.md §semantic kernel

---

## v1 — Foundry Consumer Parity: Shared Ontology for SMB + MedCare (authored 2026-04-26)

**Author:** main thread (Opus 4.7 1M), session 2026-04-26
**Status:** Active
**Scope:** Map the shared Foundry parity surface consumed by both smb-office-rs and medcare-rs. Resolve 5 callcenter UNKNOWNs (consumer-validated). Document the DataFusion/SQL groundtruth pattern. Identify shared build priorities (DM-8 PostgREST is P-0 for both). Ontology unification: one contract shape, two domain-specific instances.
**Path:** `.claude/plans/foundry-consumer-parity-v1.md`
**Cross-ref:** `smb-office-rs/docs/foundry-parity-checklist.md` (45 LF chunks); `medcare-rs` callcenter-as-owner architecture; `q2-foundry-integration-v1.md`; `lf-integration-mapping-v1.md`; `callcenter-membrane-v1.md` (UNKNOWNs resolved)

## 2026-05-07 — Status annotation: `sql-spo-ontology-bridge-v1` partially superseded

**Status:** Active (partially superseded by `lance-graph-ontology` crate, 2026-05-07)
**Note:** The `SchemaExpander` proposed in `sql-spo-ontology-bridge-v1` already shipped in earlier work, and the new `lance-graph-ontology` crate (commit `4cf9a26`, branch `claude/create-graph-ontology-crate-gkuJG`) consumes it as its sole bridge surface. The plan's Phase 4 (NARS cold sink) and `promote_to_spo` writer bridge remain owned by the original plan. Recon + decision for the new crate: `.claude/RECON_ONTOLOGY_CRATE.md` + `.claude/DECISION_SPO_ARIGRAPH.md` (prior commit `edef321`). Federated two-layer cache (Option B): SPO + ARiGraph triplet_graph are not duplicates by design; entropy-ledger rows 70 + 245 cite the L1/L2 cache pair. APPEND-ONLY annotation; original plan entry not edited.
