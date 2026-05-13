# PR Arc — Architectural Decision History

> **Auto-loaded at session start.** Every merged PR, its meta, and
> the decisions it locked in. Read BEFORE proposing anything — a new
> proposal that contradicts a decision in this arc is a 30-turn
> rediscovery tax waiting to happen.
>
> ## APPEND-ONLY RULE (MANDATORY)
>
> 1. **New PRs PREPEND** a new section at the top (most-recent first).
> 2. **Old PR sections are IMMUTABLE HISTORY.** Never rewrite or
>    delete a past PR's Added / Locked / Deferred / Docs entries.
> 3. **The ONE exception: Confidence annotations.** Each PR section
>    may have a `**Confidence (YYYY-MM-DD):**` line that IS updatable.
>    Use it to record: "working", "partial", "superseded by PR #N",
>    "broken — see PR #N for fix". This is the only mutable field.
> 4. **Corrections append.** If a Locked claim turns out wrong,
>    append a `**Correction (YYYY-MM-DD from PR #N):**` line to the
>    same entry — do not edit the original Locked line. Both stay.
> 5. **Reversals are their own PR entry.** If a later PR explicitly
>    undoes a decision, the later entry documents the reversal; the
>    earlier entry's Confidence line references it. Both remain in
>    the arc.
>
> The arc is the historical record. Rewriting it destroys the
> "why was this decided that way" context that prevents future
> rediscovery. Every entry stays.
>
> **Format:** reverse chronological. Each PR carries:
> - **Added** — new types / modules / LOC (immutable)
> - **Locked** — conventions / invariants / patterns (immutable)
> - **Deferred** — explicit parks (immutable)
> - **Docs** — knowledge files produced (immutable)
> - **Confidence (YYYY-MM-DD):** — the ONLY mutable field

---

## #365 — specs(sprint-5-6): 13-worker parallel batch + Opus meta review (merged 2026-05-13)

**Confidence (2026-05-13):** governance-only PR, no `.rs` / `Cargo.toml` changes. CI green (format / clippy / build / test / coverage — no code touched). **Status:** Merged to `main`.

**Added:**
- **13 PR-ready specs at `.claude/specs/`** (~300 KB total):
  - W1 `pr-d3a-lance-audit-sink.md` (27 KB, B-grade) — Arrow 12-column schema with `FixedSizeBinary(3)` owl_identity, super_domain × date partitioning, buffered emit + flush-at-1024/5s.
  - W2 `pr-d3b-jsonl-and-verify.md` (27 KB, **A-grade**) — JsonlAuditSink + CompositeSink + verify CLI (3 subcommands, exit codes 0-3, owl_identity as 6-char lowercase hex).
  - W3 `pr-d4-family-hydration.md` (16 KB, B-grade) — TTL hydration of FAMILY_TO_SUPER_DOMAIN via `parse_family_registry()` (new parser entry per W3 OQ-1 recommendation).
  - W4 `sprint-5-ci-matrix.md` (21 KB, B-grade) — 6 blocking gates + target matrix; ndarray#142 SIGILL gate rules R-HW-1..4.
  - W5 `sprint-5-pr-graph.md` (16 KB, **A-grade**) — Sprint-5 retrospective + 4-PR adjacent-landings dependency graph + sprint-6 unblock map.
  - W6 `pr-e1-medcare-super-domain.md` (26 KB, B-grade) — MedCare finalisation gap analysis, ~900 LOC across 6 deliverables (E1-1..E1-6).
  - W7 `pr-e2-smb-retrofit.md` (11 KB, B-grade) — 5-site bypass inventory in smb-office-rs, 3-batch incremental retrofit plan.
  - W8 `pr-e3-woa-rs-extract.md` (27 KB, B-grade) — 3-subcrate woa-rs extraction (woa-rbac/realtime/analytics), `WorkOrderBilling` super_domain, ~950 LOC.
  - W9 `pr-f1-thinking-engine-wire.md` (16 KB, B-grade) — `CognitiveBridgeGate` trait in thinking-engine + `UnifiedBridgeGate<B>` wrapper in callcenter; 3 cross-tenant op categories gated, pure math stays pure.
  - W10 `pr-g1-manifest-modules.md` (27 KB, C-grade) — build.rs YAML→Rust codegen for consumer manifests; **needs §4.3 rewrite** from `phf::Map` to sorted-slice + binary search per zero-dep invariant.
  - W11 `pr-g2-ractor-supervisor.md` (25 KB, C-grade) — Per-G actor topology, one-for-one supervision, ractor 0.14; **needs separate `LifecycleAuditEvent`** to keep `AuthOp` byte-layout stable.
  - W12 `sprint-6-conformance-test.md` (26 KB, **A-grade**) — Cross-crate `assert_consumer_conformance<B: NamespaceBridge>()` harness with 10 contract assertions (A1-A10).
  - W13 `pr-ogit-ttl-smb-hydration.md` (35 KB, post-meta addendum) — OGIT/NTO/SMB TTL deliverable bridging from `smb-office-rs:main:.claude/board/OGIT_TTL_INVENTORY.md`; 3 §E recommended answers (use `ogit.SMB.bson:` sub-namespace, `ogit:marking` per-property triples, no custom semantic types).
- **24 KB Opus meta review** at `.claude/board/sprint-log-5-6/meta-review.md` (M1 per-worker + M2 cross-spec synthesis combined). 8 sections incl. per-spec critical defects, cross-spec contradictions (CC-1..CC-N), dependency graph, sequencing recommendation, coverage gaps, open-question triage, code-review readiness verdict, sprint-5/6 cohesion synthesis.
- **14 A2A scratchpads** at `.claude/board/sprint-log-5-6/agents/agent-W{1..13,META}.md` (append-only blackboard, one per worker via `tee -a`).
- **`.claude/settings.json`** — `Write/Edit(.claude/board/sprint-log-5-6/**)` allowlist entries (initial worker batch hit permission denial; respawn batch after fix landed clean).

**Locked:**
- **CCA2A 12+1+1 pattern works at scale.** 12 parallel Sonnet workers + 1 post-meta Sonnet worker + 1 Opus meta agent in one sprint produced ~300 KB of PR-ready specs in under an hour wall-clock. Worker-prompt template from sprint-5-through-9-roadmap-v1.md held — mandatory 12-step read-order prevented duplication. 3 workers needed respawns (W1/W4/W8) for permission reasons, root-caused to missing settings.json entries — locked: every new sprint-log-N directory needs explicit `Write/Edit/Bash(tee -a)` allowlist entries before workers spawn.
- **Spec-quality grading scale (A/B/C/D/F)** established by Opus meta — to be reused across future sprint meta reviews.
- **PR-and-merge-first philosophy for spec corpora:** C-grade specs ship as-is into the spec PR; their fixes happen in the implementation PR alongside the actual code. Saves a meta-iteration round-trip.

**Deferred:**
- **4 blocking OQs** (PR body checkboxes — user decision needed before sprint-7 implementation workers fire):
  - OQ-1 (W3): TTL family-registry parser entry — W3 recommends new `parse_family_registry()` API.
  - OQ-2 (W10): `phf::Map` vs sorted-slice + binary search — meta recommends sorted-slice (zero-dep invariant).
  - OQ-3 (W6): `medcare_rbac::Role` migration to canonical `RoleGroup` vs bridge — affects E1-1 LOC ±30%.
  - OQ-4 (W13 §E.1): OGIT/NTO/SMB BSON namespace — W13 recommends `ogit.SMB.bson:` sub-namespace.
- Sprint-6 W5/W6 (hiro-rs / hubspot-rs scaffolds) — blocked on repo-creation decision (separate repos vs monorepo subcrates).
- Sprint-5 W2 (PR-A spec) absorbed into PR #364; W3/W4 absorbed into MedCare-rs#112 + smb-office-rs#31 commits; W5/W10 made moot by widening choice in PR #364.

**Docs:**
- `.claude/specs/` — 13 new PR-ready specs.
- `.claude/board/sprint-log-5-6/` — SPRINT_LOG roster + meta-review + 14 agent scratchpads.

---

## #364 — D-SDR-3/4/5 + sprint-log-4 governance + sprint 5-9 roadmap + codex P1/P2 fixes (merged 2026-05-13)

**Confidence (2026-05-13):** merged clean, all 5 CI checks green on `c8176cb`. Codex review threads auto-marked Outdated by GitHub after the surgical fixes shipped pre-merge. **Status:** Merged to `main`. **Adjacent landings (2026-05-13):** MedCare-rs#112 (PR-B, UnifiedBridge<MedcareBridge> + medcare-rbac + medcare-realtime substrate, +2963 LOC across 17 files) and smb-office-rs#31 (PR-C, UnifiedBridge<OgitBridge> wiring, +111 LOC) both **merged** the same day, closing the sprint-5 cross-repo coordinated landing for D-SDR-5's `UnifiedBridge` surface. Substrate this PR shipped is now consumed end-to-end by both MedCare and smb-office.

**Added:**
- **D-SDR-3** (`2c3e87d`, ~300 LOC): `OgitFamilyTable` + `FamilyEntry` per-family codebook (inline label + schema + verbs per `super-domain-rbac-tenancy-v1.md §3.3`).
- **D-SDR-4** (`1d0157f`, ~460 LOC): merkle-chained `UnifiedAuditEvent` log for `UnifiedBridge`. `AuditMerkleRoot = u64` FNV-1a.
- **D-SDR-5** (`dc9e081`, ~300 LOC): wire `authorize_*` through `Policy::evaluate` chain with audit emission on every decision.
- **Codex P1 surgical fix** (`3208743`): widen `OwlIdentity` slot u8 → u16. Layout becomes `{ family: u8, slot: u16 }` = 3 bytes on-wire. `OgitFamilyTable` migrates from `[Option<FamilyEntry>; 256]` to sparse `HashMap<u16, FamilyEntry>`. `UnifiedAuditEvent::canonical_bytes` grows 25 → 26 bytes (`owl` slice [13..16); op/decision/role_hash offsets shift by 1). New test `slot_keyspace_distinguishes_high_ids` locks the invariant. `to_canonical_bytes() -> [u8; 3]` replaces `raw()`.
- **Codex P2 surgical fix** (`e23ce89`): `emit_audit` stamps `super_domain` from `self.audit_chain.super_domain()` instead of the all-`Unknown` static `FAMILY_TO_SUPER_DOMAIN` lookup.
- **CI build fix** (`a3c753f`): enable `ndarray/hpc-extras` feature so `blake3` resolves in the workspace build.
- **Sprint-log-4** governance corpus (~280 KB): 12 worker specs at `.claude/specs/`, 2 meta reviews at `.claude/board/sprint-log-4/meta-{1,2}-review.md`, sprint summary + per-worker scratchpads.
- **Sprint-5-through-9 roadmap** at `.claude/plans/sprint-5-through-9-roadmap-v1.md` (70 agents = 60 workers + 10 meta across 5 sprints).
- `Cargo.lock` updated post hpc-extras opt-in (`c8176cb`).

**Locked:**
- **OwlIdentity canonical wire form = 3 bytes** `[family, slot_lo, slot_hi]`. Any cross-language emitter (Rust / C#) MUST use `OwlIdentity::to_canonical_bytes()`. The old 2-byte packed `u16` layout is gone; no compat shim because no on-disk audit log exists outside test fixtures at this commit.
- **`UnifiedAuditEvent::canonical_bytes` is 26 bytes**, owl at `[13..16)`. Wire-format breaking for any persisted audit log.
- **`OgitFamilyTable` is sparse** (`HashMap<u16, FamilyEntry>`); the "256-slot dense array" framing in prior doc comments is replaced by "sparse map".
- **Audit events take super_domain from the configured `AuditChain.super_domain()`**, not from a static family→domain table. `FAMILY_TO_SUPER_DOMAIN`'s purpose narrows to a fallback / future hydration mechanism.
- **Sprint-5+ worker prompts have a mandatory 12-step `.claude/plans/` read-order** as hard precondition (per sprint-4 retrospective: worker specs duplicated existing plan corpus when read-order was advisory).

**Deferred:**
- TTL namespaces, full compliance certification, federation Phase 2, drift bridge LanceProbe M5/M6 — owned by sprints 6/8 per roadmap.
- **PR-B medcare-rs UnifiedBridge wiring**: commits exist locally on `claude/lance-datafusion-integration-gv0BF` in `MedCare-rs` repo (already pushed to remote integration branch, no PR opened yet).
- **PR-C smb-office-rs UnifiedBridge wiring**: same shape, commits already on remote integration branch in `smb-office-rs`, no PR opened yet.
- **Per-namespace u8 slot allocation in `RegistryState::append`**: declined this session — widening to u16 carrier in `3208743` is the chosen fix path. Per-namespace allocation would require widening `BindSpace.entity_type` from bare u16 to carry `(namespace_id, entity_type_id)` and rewriting `enumerate_first_with_entity_type_id` (currently relies on global uniqueness, breaks silently under per-namespace allocation — two known callers in `cascade_cols_test.rs:80` + `cognitive-shader-driver/src/driver.rs:312`). Tracked in TECH_DEBT.

**Docs:**
- `.claude/plans/sprint-5-through-9-roadmap-v1.md` (the 60-worker + 10-meta map).
- `.claude/board/sprint-log-4/` (full sprint corpus).
- `.claude/specs/` (12 PR-scoped specs for sprint-5 deliverables).
- `EPIPHANIES.md` 2026-05-13 entries (sprint-4 duplication-audit, 14+ FINDING/CORRECTION/CONJECTURE entries on OGIT axes, super-domain subcrates, API drift, FMA convergence).

**Correction (2026-05-13):** Sprint-4 specs partially duplicated existing `.claude/plans/` content despite the advisory read-order — see EPIPHANIES 2026-05-13 duplication-audit. Sprint-5+ enforces the read-order as a hard precondition in the worker-prompt template.

---

## #354 — gov: #353 post-merge + adjacent-landings (#109, OGIT#2, woa-rs#2) (merged 2026-05-07)

**Confidence (2026-05-07):** governance-only PR, no plan / knowledge / code changes. Append-only board hygiene confirmed working — merged cleanly, no past entries edited. **Status:** Merged to `main` as `a6797ad`.

**Added:**
- `.claude/board/PR_ARC_INVENTORY.md` — full Added/Locked/Deferred entry for #353 prepended.
- `.claude/board/LATEST_STATE.md` — `#353` row prepended; "Last updated" advanced.

**Locked:**
- **Append-only board hygiene works in practice** — the prepend pattern survived 4 sequential PR landings (#352, #353, #354, plus prior splat-osint) without any past-entry mutation. Confidence-line-only mutability policy is durable.
- **Cross-repo coordinated landing pattern** is documented as a 5-PR-in-a-day recipe: lance-graph plans → OGIT TTL → woa-rs/medcare-rs consumer integration → lance-graph governance close-out.

**Deferred:** none — pure governance.

**Docs:** none added — only board updates.

**Resolves ledger rows:** none. **Closes the governance loop** for the #352 → #353 → #354 sequence.

---

## #353 — plan: palantir-parity-cascade v2 + SoA DTO entropy ledger + #352 post-merge governance (merged 2026-05-07)

**Confidence (2026-05-07):** plan-only, pre-execution. Pillar 0 carry-forward (Foundry parity IS SoA-as-canon parity) is the architectural anchor; v2 integrates 4 prior Foundry parity docs without duplicating. SoA DTO ledger formalizes 22 DTOs across 4 tiers as the canonical classification artifact. **Status:** Merged to `main` as `4d0c2d9`.

**Added:**
- `.claude/plans/palantir-parity-cascade-v2.md` (262 lines) — integration capstone over `q2-foundry-integration-v1`, `lf-integration-mapping-v1`, `foundry-consumer-parity-v1`, `medcare-foundry-vision`, and v1 cascade Pillar 0. 15 D-PARITY-V2 deliverables. Top-3 ship with the plan: V2-1 (DTO ledger), V2-2 (triangle ledger — not yet), V2-3 (BusDto bridge — not yet).
- `.claude/knowledge/soa-dto-dependency-ledger.md` (210 lines) — append-only entropy ledger. 22 DTOs classified: 9 bare-metal, 7 SoA-glue, 6 bridge-projection (3 OPEN reclassifications). Codec cascade column status: all 8 OPEN today (registry uses `(bridge_id, public_name)` tuples + `ogit_uri` hashing per 2026-05-07 audit). Internal vs external O(1) mapping diagrams. Probe queue with pass criteria for D-CASCADE-V1-1/7/11 + D-PARITY-V2-3/10. Maintenance protocol attached.
- `.claude/board/PR_ARC_INVENTORY.md` + `.claude/board/LATEST_STATE.md` — post-merge governance for #352 (`8e2f088`).
- `.claude/board/INTEGRATION_PLANS.md` — v2 capstone prepend.

**Locked:**
- **v2 Pillar 0 carry-forward:** Foundry parity IS SoA-as-canon parity. Column H (`EntityTypeId = u16`, PR #272 SHIPPED) is already the Foundry Object Type bridge; v2 makes the SoA carry the Foundry-equivalent shape, NOT duplicate the table set.
- **DTO ledger maintenance protocol:** every PR adding `*Dto`/`*Row`/`*Filter`/`*Intent`/`*Event`/`*Step`/`*Slot` types prepends a row. CI gate D-PARITY-V2-10 (planned) enforces.
- **`ResonanceDto` IS the SoA**, not a glue layer (per the 2026-05-07 audit; `thinking-engine::dto.rs:59`, 4096-element ripple field).
- **Business Logic ↔ Thinking-style ↔ OGIT triangle** is a routing artifact (D-PARITY-V2-2), NOT a new schema column.
- **Three-tier classification doctrine:** bare-metal may carry `serde::Serialize` (Zone 3 only); SoA-glue must NOT carry `serde::Serialize` (projections break the SIMD sweep); bridge-projection must own no data (only `LazyLock<&Registry>`).

**Deferred (immutable parks):**
- All 15 D-PARITY-V2 code implementations except V2-1 (ledger ships with the plan).
- Q2 cockpit panels (D-PARITY-V2-7/11/15) — depend on lance-graph workspace + Q2 repo simultaneously; cross-repo sync needed.
- `lance-graph-models` crate scaffold (D-PARITY-V2-8) — independent but unscheduled.
- Helix-equivalent causal-histogram operator (D-PARITY-V2-14) — out of v2 scope.

**Docs:**
- `.claude/plans/palantir-parity-cascade-v2.md` — capstone with §"Self-bootstrapping prompt".
- `.claude/knowledge/soa-dto-dependency-ledger.md` — entropy ledger.

**Resolves ledger rows:** none directly. **Hardens** v1 D-CASCADE-V1-7 (codec cascade column population) via explicit OPEN status tracking per column.

**Adjacent consumer landings (not in this PR):**
- **MedCare-rs #109** (merged 2026-05-07): `?source=lance` toggle on `GET /api/patient/{id}` exercises per-request `RlsRewriter` + `ColumnMaskRewriter` attachment from `lance-graph-callcenter::rls` and `policy::ColumnMaskRewriter`. Validates the Zone 2 → Zone 3 path the v1/v2 plans rely on. Note: PR #109 documents that `ColumnMaskRewriter` has NO `::new()` method — constructed via struct literal `{ registry, actor_role }` (verified at `policy.rs:111-114, 464, 565, 672`). Consider a `// classification:` doc-comment audit for the DTO ledger now that consumer-side construction patterns are known.
- **OGIT fork branch** (`claude/create-graph-ontology-crate-gkuJG`, not yet PR'd): post-merge follow-on adds 24 predicate fills to NTO/WorkOrder/{Order,Customer,Article}.ttl + bootstraps NTO/Healthcare/ with 7 entities + 7 enums (846 lines). Closes the entity-level + per-attribute gaps the woa-bridge and medcare-bridge needed for O(1) migration. v5 D-1 (dcterms:source) extended from entity-level to per-attribute level in this work; medcare-bridge previously failed at hydrate with `UnknownNamespace("Healthcare")` — now resolvable.

---

## #352 — plan: lance-graph-ontology v5 + ogit-cascade v1 (merged 2026-05-07)

**Confidence (2026-05-07):** plan-only, pre-execution. Pillar 0 (SoA-as-canon) is the architectural anchor; Pillars 1-4 are mechanical consequences. Top-3 deliverables locked for both v5 and v1 cascade. Foundry/Gotham parity prior art confirmed extensive (Q2 = Gotham UI equivalent per `q2-foundry-integration-v1.md`; Column H EntityTypeId = Foundry Object Type bridge per PR #272 SHIPPED; LF-12/20/22/23/50 already mapped in `lf-integration-mapping-v1.md`). v2 roadmap will integrate, not duplicate. **Status:** Merged to `main` as `8e2f088`.

**Added:**
- `.claude/plans/lance-graph-ontology-v5.md` (177 lines) — 15 deliverables ranked by leverage/cost, picking up where v4 (OGIT#1 merged) left off.
- `.claude/plans/ogit-cascade-supabase-callcenter-v1.md` (209 lines) — Pillar 0 (SoA-as-canon) + Pillar 1 (OGIT SPO-G lingua franca) + Pillar 2 (Zone 1/2/3 BBB tightening) + Pillar 3 (smb/medcare bridge collapse to 2-line projections) + Pillar 4 (BioPortal arsenal stubs); 15 D-CASCADE deliverables.
- `.claude/board/INTEGRATION_PLANS.md` — prepended with both v5 and v1 cascade entries (append-only governance honored).

**Locked:**
- **v5 ratifications (4):** smb-ontology export-only forever; D-ONTO-V5-9 above D-ONTO-V5-2 (registry has zero behavioral consumer until V5-9 lands); `MulThresholdProfile` lives in `lance-graph-contract`; `AdaWorldAPI/OGIT` is extension-fork-only — never PR back to `almatoai/OGIT`.
- **v1 Pillar 0 (the holy-grail click):** `OntologyRegistry` IS the SoA; per-domain schema IS the DTO + name→row index. Bridges hold `LazyLock<&OntologyRegistry>`, project columns through scoped views.
- **v1 Pillar 2 (Zone 1/2/3 BBB tightening):** Zone 3 is the only outbound emission point. `serde::Serialize` is **denied on Zone 1 / Zone 2 types** via `cert-officer` static check (D-CASCADE-V1-1).
- **v1 codec cascade per row** (target state — NOT YET WIRED, see Deferred): identity `Vsa16kF32` (64 KB) → CAM-PQ `[u8; 6]` → Base17 `[u8; 34]` → palette key `u32` → Scent `u8` + qualia `[f32; 18]` + meta `MetaWord` + edge `CausalEdge64`. Every step `O(1)`.
- **v1 `ontology_context_id: u32` per named graph** (D-CASCADE-V1-2; consistent with `lance-graph-rdf-fma-snomed-v1` §Core).
- **v1 BioPortal arsenal scope:** 10 namespace stubs under `OGIT/NTO/Medical/{ICD10CM, RxNorm, LOINC, FMA, RadLex, SNOMED, MONDO, HPO, DRON, CHEBI}/`. Full ingestion gated on `lance-graph-rdf-fma-snomed-v1`.

**Deferred (immutable parks):**
- All 30 D-* code implementations across both plans (D-ONTO-V5-1..15 + D-CASCADE-V1-1..15).
- **Codec cascade column population in `OntologyRegistry`** — current state has NO `cam_pq_code`/`base17`/`palette_key`/`scent` columns; uses `(bridge_id, public_name)` tuples + `ogit_uri` hashing for indexing (per agent audit, `registry.rs:33-86`). D-CASCADE-V1-7 is the wiring deliverable.
- Full SNOMED CT, DRON, CHEBI imports (license / size-payoff gated).
- bgz-tensor attention layer integration with codec cascade (orthogonal).
- n8n-rs / crewai-rust consumption of new SoA columns (separate plan).
- **Thinking-style OGIT mapping** — user request 2026-05-07; queued as v2 follow-on.
- TRUST-1 / FLOW-1 / COMPASS-1 / PARSER-1 ledger rows — explicit deferrals.

**Docs:**
- `.claude/plans/lance-graph-ontology-v5.md` — v5 plan with §7 self-bootstrapping prompt.
- `.claude/plans/ogit-cascade-supabase-callcenter-v1.md` — v1 cascade plan with §"Self-bootstrapping prompt" + cross-reference table.

---

## splat-osint-ingestion: SPLAT-1 stage 0->1 + EWA OSINT bridge (2026-05-06)

**Confidence (2026-05-06):** high (math certified by Pillar 6 — PR #289 EWA-Sandwich PSD-preserving 10 000/10 000 hops; PR #286 Koestenberger-Stark 1.467x tightness; PR #287 Dueker-Zoubouloglou Hilbert-CLT). **Status:** In PR (branch `claude/splat-osint-ingestion`).

**Added:**
- `crates/lance-graph-contract/src/splat.rs` (new module) — `SplatChannel` (6 variants: Support / Contradiction / Forecast / Counterfactual / Style / Source), `CamPlaneSplat` (q8 amplitude / width / theta_accept + 16-byte witness identity + 8-byte `replay_ref`), `SplatPlaneSet` (6 channel planes = 12 KB), `AwarenessPlane16K` (256 x u64 = 2 KB pressure tile), `CamSplatCertificate` (q8 pressure measurements + replay decision), `SplatDecision` (Proceed / RequireExactReplay / PrefetchOnly / ScenarioOnly / Drop), `TriadicProjection`, `ReasoningWitness64`. 10+ unit tests.
- `crates/jc/examples/osint_edge_traversal.rs` (new example) — EWA-Sandwich Sigma-push-forward demo for an OSINT 5-hop chain. Side-by-side vs naive convolution. Pillar-6-certified neo4j-edge-hydration substitute.
- `.claude/plans/2026-05-06-splat-osint-ingestion-v1.md` — plan doc tracking PRs 1-6 of the gaussian-splat-cam-plane-workaround.md sequence (D-SPLAT-1 through D-SPLAT-7).
- Board hygiene: `INTEGRATION_PLANS.md` prepend; `LATEST_STATE.md` Contract Inventory adds `splat`; `ARCHITECTURE_ENTROPY_LEDGER.md` SPLAT-1 row Aspirational -> Wired stage 1 (entropy 4 -> 2); `STATUS_BOARD.md` new section with D-SPLAT-1..7 rows.

**Locked:**
- **Splat plane width = 16 384 bits** (matches `Vsa16kF32` and `Binary16K` carriers). `AwarenessPlane16K` = 256 x u64 = 2 KB.
- **q8 amplitudes everywhere on the hot path** — no `f32`/`f64` fields in `CamPlaneSplat`, `SplatPlaneSet`, or `CamSplatCertificate`. Float accumulation, if it ever appears, lives behind calibration paths, not the deposition kernel.
- **I-VSA-IDENTITIES preserved** — splats POINT TO content via 16-byte witness identity + 8-byte `replay_ref`. The 6 channel planes are addressable by content identity, never by anonymous superposition of content bits.
- **Zero-dep contract preserved** — `lance-graph-contract` keeps its zero external-crate-dep invariant.
- **No serde on types** — wire formats are explicit per CLAUDE.md Workspace Convention 5.
- **Click P-1 method discipline** — `CamPlaneSplat::pressure_q8()`, `SplatPlaneSet::deposit(&CamPlaneSplat)`, `CamSplatCertificate::decide() -> SplatDecision`. No free functions on the carrier state.
- **Pillar-6 / Pillar-7 inheritance** — PR 2 inherits PR #289 PSD-preservation guarantee; D-SPLAT-4 (queued) consumes `MergeMode::AlphaFrontToBack` from PR #324.

**Deferred (PRs 3-6 of the doc-sequence):**
- **PR 3 (D-SPLAT-3):** `witness_to_splat()` deterministic conversion — `(factor_a, factor_b, projection, ReasoningWitness64, sigma_idx, ThetaDecision, replay_ref) -> CamPlaneSplat` under fixed codebooks + seeds.
- **PR 3 (D-SPLAT-4):** Splat deposition into BindSpace columns via `MergeMode::AlphaFrontToBack` lanes (q8 / bit-tile accumulation per Pillar-7 sink mode).
- **PR 4 (D-SPLAT-5):** `PlanarSplatBundle4096` with local (8-16) / short (64) / medium (512) / long (4096) cycle bands.
- **PR 5 (D-SPLAT-6):** Semantic-CAM-distance integration — survivor tile selection compares against splatted pressure planes, not raw Hamming over anonymous bits.
- **PR 6 (D-SPLAT-7):** Replay fallback — when `CamSplatCertificate` is insufficient (e.g. high support AND high contradiction), load exact 4096-cycle ThoughtCycleSoA replay slice.

**Docs:**
- `.claude/knowledge/gaussian-splat-cam-plane-workaround.md` (already-existing; not modified by this PR).
- `.claude/plans/2026-05-06-splat-osint-ingestion-v1.md` (new this PR).
- Companion: `.claude/plans/tetrahedral-epiphany-splat-integration-v1.md` (SPOW tetrahedron axis; not modified).

**Resolves ledger rows:**
- SPLAT-1 (Section A of `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md`): Aspirational -> Wired (x1, stage 1). Entropy 4 -> 2.

---

## #243 — D5+D7 categorical-algebraic inference architecture (2026-04-21)

**Confidence (2026-04-21):** Working. 175/175 contract, 63/63 deepnsm (grammar-10k).

**Added:**
- `contract::grammar::thinking_styles` — `GrammarStyleConfig`, `GrammarStyleAwareness` (NARS-revised `HashMap<ParamKey, TruthValue>`), `revise_truth`, `ParseOutcome` (5 polarities), `divergence_from(prior)` (KL term). 490 LOC, 12 tests.
- `contract::grammar::free_energy` — `FreeEnergy` (likelihood + KL → total), `Hypothesis` (role fillers + Pearl 2³ mask), `Resolution` (Commit / Epiphany / FailureTicket), `from_ranked` classifier, `HOMEOSTASIS_FLOOR` / `EPIPHANY_MARGIN` / `FAILURE_CEILING`. 347 LOC, 7 tests.
- `contract::grammar::role_keys` — `RoleKey::bind/unbind/recovery_margin` (slice-masked XOR), `Vsa10k` type alias, `VSA_ZERO`, `vsa_xor`, `vsa_similarity`, `word_slice_mask` helper. +295 LOC, +14 tests (5-role lossless superposition verified).
- `deepnsm::content_fp` — 10K-dim content fingerprints from COCA vocab ranks (SplitMix64). 98 LOC, 5 tests. Feature-gated: `grammar-10k`.
- `deepnsm::markov_bundle` — `MarkovBundler` (±5 ring buffer, role-key bind, braiding via `vsa_permute`, XOR-superpose, `WeightingKernel`). 250 LOC, 8 tests.
- `deepnsm::trajectory` — `Trajectory` (Think carrier): `role_bundle`, `mean_recovery_margin`, `ambient_similarity`, `free_energy`, `resolve`. 298 LOC, 4 tests.
- `CLAUDE.md` § The Click (P-1): top-of-file architecture diagram + 3 simplicity invariants + shader-cant-resist + thinking-is-a-struct + tissue-not-storage + grammar-of-awareness + 2 litmus tests.
- `.claude/plans/categorical-algebraic-inference-v1.md` (496 lines): meta-architecture proving 5 operations are 1 algebraic substrate, grounded in 8-paper proof chain.

**Locked:**
- `RoleKey::bind` is slice-masked XOR (categorically optimal per Shaw 2501.05368 Kan extension theorem). Not a design choice — a theorem consequence.
- `FreeEnergy = (1 - likelihood) + KL` where likelihood = mean role recovery margin, KL = `awareness.divergence_from(prior)`. Three thresholds: F<0.2 commit, ΔF<0.05 epiphany, F>0.8 escalate.
- NARS revision asymptotes at φ-1 ≈ 0.618 (golden ratio confidence ceiling). Feature, not bug. Permanent epistemic humility.
- Markov = XOR of braided sentence VSAs. No HMM. No transition matrix. No weights.
- Thinking is a struct (not a service, not a function). The DTO carries cognition as identity.
- AriGraph/episodic/CAM-PQ are thinking tissue (organs of Think), not storage services.
- Object-does-the-work test: free function on carrier's state = reject. Method on carrier = accept.
- Five-lens test: every new type serves Parsing / Free-Energy / NARS / Memory / Awareness or is drift.

**Deferred:**
- Steps 4-8 of the 8-step wiring sequence (pipeline, AriGraph commit, global context, awareness revision, KL feedback). Three PRs to close the loop.
- D10 Animal Farm benchmark (the AGI test: chapter-10 accuracy > chapter-1 accuracy).
- Cross-lingual bundling (needs parallel corpora).
- ONNX arc model (D9, D11).

**Docs:**
- `.claude/knowledge/paper-landscape-grammar-parsing.md` — 14 papers in 3 tiers.
- `.claude/knowledge/session-2026-04-21-categorical-click.md` — session handover with 12 critical insights + 7 anti-patterns.
- `.claude/board/EPIPHANIES.md` — 12 new epiphanies with "why this dilutes" warnings.
- `.claude/board/INTEGRATION_PLANS.md` — `categorical-algebraic-inference-v1` entry prepended.

---

## #225 — Codec-sweep plan + D0.6/D0.7 CodecParams types (merged 2026-04-20)

**Confidence (2026-04-20):** Working. 147/147 contract suite passing (133 prior + 14 new).

**Added:**
- `.claude/plans/codec-sweep-via-lab-infra-v1.md` (~1,800 lines) — JIT-first codec sweep plan operationalising PR #220's "What's Needed to Fix" list through the lab endpoint. One upfront Wire-surface rebuild, unlimited JIT-kernel candidates afterwards.
- 9 starter YAML configs under Appendix A (controls + four #220 fixes + composite + cross-product grid).
- `.claude/board/INTEGRATION_PLANS.md` — prepended `codec-sweep-via-lab-infra-v1` entry per APPEND-ONLY rule.
- `contract::cam::LaneWidth` {F32x16, U8x64, F64x8, BF16x32} — mirrors `ndarray::simd::*` lane types.
- `contract::cam::Distance` {AdcU8, AdcI8} — split per CODING_PRACTICES gap 5 (sign-handling / bipolar cancellation).
- `contract::cam::Rotation` {Identity, Hadamard{dim}, Opq{matrix_blob_id, dim}} + `is_matmul()`.
- `contract::cam::ResidualSpec` {depth, centroids}.
- `contract::cam::CodecParams` + `kernel_signature() -> u64` + `is_matmul_heavy() -> bool`.
- `contract::cam::CodecParamsBuilder` — fluent API (CODING_PRACTICES gap 3 remediation).
- `contract::cam::CodecParamsError` {ZeroDimension, OpqRequiresBf16, HadamardDimNotPow2, CalibrationEqualsMeasurement}.
- 14 new `codec_params_tests` covering builder defaults + each validation + kernel_signature stability + matmul-heavy detection.

**Locked:**
- **Six rules A-F bind every JIT-emitted kernel in the codec sweep:**
  - Rule A: tensor access via stdlib `slice::array_windows::<N>()` + `ndarray::simd::*` loaders
  - Rule B: SIMD exclusively via `ndarray::simd::*` / `simd_amx::*` / `hpc::amx_matmul::*` / `hpc::simd_caps::*`
  - Rule C: polyfill hierarchy AMX → AVX-512 VNNI → AVX-512 baseline → AVX-2, **no consumer-visible scalar tier**
  - Rule D: JSON / YAML / REST configuration only
  - Rule E: Wire surface IS the SIMD surface (object-oriented, `LaneWidth` explicit, methods not scalar bags, 64-byte-aligned decode)
  - Rule F: **Serialisation at REST edge only; never inside**
- **Iron rule:** SoA never scalarises without ndarray. If a kernel runs scalar on the SoA path, the SoA invariant is broken.
- **Intel AMX** (not Apple) — `ndarray::simd_amx::amx_available()` + `ndarray::hpc::amx_matmul::{tile_dpbusd, tile_dpbf16ps, vnni_pack_bf16}` on Sapphire Rapids+ via stable inline asm (rust-lang #126622 keeps AMX intrinsics nightly).
- **Precision-ladder validation fires BEFORE JIT compile.** OPQ rotation requires BF16x32 lane. Hadamard dim must be 2^k.
- **Overfit guard typed-error-rejects the PR #219 pattern.** `CalibrationEqualsMeasurement` refuses to emit ICC when `calibration_rows == measurement_rows`.
- **Kernel signature excludes seed.** Seed changes calibration sample but not IR — cached kernels stay hot across seeds.
- **Zero ndarray changes.** "Everything the sweep needs is already in ndarray" — user directive, enforced.
- **Zero serde in the contract.** YAML/JSON deserialisation belongs to the consumer crate.

**Deferred:**
- D0.1 (`WireCalibrate` extension), D0.2 (`WireTokenAgreement`), D0.3 (`WireSweep`), D0.5 (`auto_detect`) — next PR.
- D1.1-D1.3 (JIT codec kernels), D2.1-D2.3 (token-agreement harness), D3.1-D3.2 (sweep driver + Lance logger), D4.1-D4.2 (frontier analysis), D5 (graduation bridge) — later PRs.

**Docs:**
- Plan references: `.claude/knowledge/lab-vs-canonical-surface.md`, `cam-pq-unified-pipeline.md`, `codec-findings-2026-04-20.md`, `rotation_vs_error_correction.md`, `encoding-ecosystem.md`.

**Decisions for future PRs to respect:**
- When testing a codec candidate: reconstruction error → reconstruction ICC on held-out rows → **token agreement**. The cert gate is token agreement, not synthetic ICC (PR #219 lesson).
- Adding a new codec candidate is authoring a YAML file. Zero Rust changes. Zero rebuilds.
- `CodecParams::kernel_signature` is the JIT cache key. Adding unrelated fields to `CodecParams` must NOT change what goes into the signature.

---

## #224 — Lab = API+Planner+JIT, thinking harvest, I11 measurability (merged 2026-04-20)

**Confidence (2026-04-20):** Working. Docs-only PR, no build impact.

**Added:**
- `.claude/knowledge/lab-vs-canonical-surface.md` extended with three load-bearing sections:
  - "Why the Lab Surface Exists (positive purpose)" — three-part stack (API + Planner + JIT), not just quarantine scaffolding.
  - "The third purpose — thinking harvest (the AGI magic bullet)" — REST/Cypher → `{rows, thinking_trace}` externalises planner's 36-style / 13-verb / NARS trace for log/replay/revision.
  - I11 invariant: measurable stack, not a black box. Every layer L0→L4 emits harvest-ready trace.

**Locked:**
- **Codec cert is token agreement, not synthetic ICC.** PR #219's 0.9998 was overfit-on-training; PR #220's 0.195 was reconstruction-only. Real cert gate is decoded codec's top-k tokens matching Passthrough.
- **Three-part lab stack:** REST/gRPC API (curl entry, no rebuild) × Planner (real dispatch path, not toy bench) × JIT (runtime kernel swap, no relink). All three together = unlimited candidates measured via real dispatch.
- **Thinking harvest = AGI magic bullet.** An AGI that cannot observe its own reasoning cannot revise it. REST/Cypher injection + JIT + planner closes that loop outside the binary.
- **I11 — measurable stack.** Every layer's trace is harvest-ready through the lab surface. Proposed changes that shrink trace for perf/simplicity are rejected.
- **Two allowed edges** for serialisation: REST/gRPC ingress (JSON/protobuf in, once per request), REST/gRPC response (JSON/protobuf out, once per response). **No internal serde between layers.** Lance append is the one persistent egress.

**Deferred:** Actual ONNX story-arc training, actual token-agreement harness implementation — all after Phase 0 Wire surface hardens.

**Docs:**
- `lab-vs-canonical-surface.md` — now the canonical cross-cutting invariant doc (I1-I11 + six rules A-F in PR #225).

**Decisions for future PRs to respect:**
- Never propose a codec cert claim based on reconstruction ICC alone. Always measure token agreement.
- The three-part stack is the iteration testbed AND the observability port — both uses share the same binary.

---

## #223 — LAB-ONLY firewall + AGI-as-SoA + I1-I10 (merged 2026-04-20)

**Confidence (2026-04-20):** Working. Docs-only PR, no build impact.

**Added:**
- `.claude/knowledge/lab-vs-canonical-surface.md` (NEW) — MANDATORY pre-read for REST/gRPC/Wire DTO/OrchestrationBridge/codec-research work. Three sections:
  - The One-Line Rule: `cognitive-shader-driver` IS the unified API; Wire DTOs are lab quarantine.
  - AGI = (topic, angle, thinking, planner) = struct-of-arrays consuming `cognitive-shader-driver`. The four AGI axes map to the four BindSpace SoA columns.
  - 10 cross-cutting architecture invariants I1-I10 (below).
- `CLAUDE.md` P0 rule: read this doc BEFORE any REST/gRPC/Wire DTO/endpoint/shader-lab work.

**Locked (Invariants I1-I10):**
- **I1** — BindSpace read-only `Arc<[u64; 256 * N]>`; writes cross the CollapseGate airgap via `MergeMode::{Xor, Bundle}`.
- **I2** — Canonical SIMD import is `ndarray::simd::*`. Never `ndarray::hpc::simd_avxNNN::*` reach-through.
- **I3** — Layer temporal budgets: L0 sub-ns, L1 ns zero-copy, L2 ns, L3 µs, L4 ms.
- **I4** — Temperature hierarchy Hot (BindSpace HDR) → Warm (CAM-PQ) → Cold (DataFusion scalar joins) → Frozen (metadata). Cold narrows first; HDR semirings fire only on survivors.
- **I5** — Thinking IS an `AdjacencyStore`. 36 styles at τ-prefix 0x0D. One engine, two graphs.
- **I6** — Weights are seeds. GGUF hydrates into palette + `Fingerprint<256>` + FisherZTable + holographic residual + `CausalEdge64`. Inference = Hamming cascade + palette lookup, no matmul.
- **I7** — Per-cycle cascade budget ~2.3ms/1M rows with monotone narrowing (topic → angle → causality → qualia → exact).
- **I8** — 4096 address surface = 16 prefix × 256 slots. `Addr(u16)`. Prefix `0x0D` is thinking styles.
- **I9** — Three DTO families (doctrinal, not yet shipped): StreamDto (pre-parse) / ResonanceDto (active sweep) / BusDto (post-collapse). Field ≠ sweep ≠ bus.
- **I10** — HEEL / HIP / BRANCH / TWIG / LEAF progressive precision hierarchy. bgz17 IS HEEL — not LEAF identity.

**Locked (framing):**
- Claude Code sessions in this workspace **never** write a parallel `struct Agi { topic, angle, thinking, planner }`. Those ARE the BindSpace SoA columns. Wrapping them in a new struct breaks SIMD sweep.
- Extend by **column**, not by layer. New AGI capability = new BindSpace column, not a new trait / endpoint / DTO family.
- REST endpoints (`/v1/shader/*`) are LAB-ONLY. Adding `/v1/shader/<new>` is the Kahneman-Tversky System-1 easy path; extending `OrchestrationBridge` / adding a `StepDomain` variant is the System-2 correct move.

**Deferred:** Thinking harvest subsection, I11 measurability invariant, codec-sweep plan — PR #224 / #225.

**Docs:**
- `lab-vs-canonical-surface.md` (NEW, 429 lines).

**Decisions for future PRs to respect:**
- Never add a per-op REST endpoint as "the API." The canonical consumer surface is `UnifiedStep` via `OrchestrationBridge`.
- Never bypass `ndarray::simd::*` to reach `hpc::simd_avxNNN::*`. That's a private backend, not a consumer surface.
- AGI is NOT a new crate. AGI is the already-shipped BindSpace + ShaderDriver + OrchestrationBridge interpreted through the four-axis lens.

---

## #210 — Phase 1 grammar + knowledge docs (merged 2026-04-19)

**Confidence (2026-04-19):** Working. All tests green at merge time.

**Added:**
- `contract::grammar::context_chain` — coherence_at / total_coherence / replay_with_alternative / disambiguate / DisambiguationResult / WeightingKernel {Uniform, MexicanHat, Gaussian} (+396 LOC, 8 tests).
- `contract::grammar::role_keys` — 47 canonical role keys addressed as contiguous `[start:stop]` slices over 10,000 VSA dims. FNV-64 + per-dim LCG deterministic generation. `Tense` enum (12 variants). `finnish_case_key / tense_key / nars_inference_key` lookups (+404 LOC, 7 tests).

**Locked:**
- **Role-key VSA addressing uses contiguous slices**, not scattered bits. Subject=[0..2000), Predicate=[2000..4000), Object=[4000..6000), Modifier=[6000..7500), Context=[7500..9000), TEKAMOLO slots=[9000..9900), Finnish cases=[9840..9910), tenses=[9910..9970), NARS inferences=[9970..10000).
- **All role-key slices are disjoint**; binding into one slice does not contaminate another.
- **ContextChain coherence is Hamming-based** on the Binary16K variant, graceful zero-score on other variants (zero-dep constraint).
- **Mexican-hat weight:** `(1 - 2x^2) · exp(-2x^2)` where `x = d / MARKOV_RADIUS`. Monotone on d=0..5.
- **DISAMBIGUATION_MARGIN_THRESHOLD = 0.1** — below this the `escalate_to_llm` flag fires.

**Deferred:**
- CausalityFlow 3→9 slot extension (modal/local/instrument + beneficiary/goal/source).
- Phase 2 work: D2 FailureTicket emission, D3 Triangle bridge, D5 Markov bundler, D7 grammar thinking styles.
- All of Phase 3/4.

**Docs:**
- `grammar-landscape.md` (429 lines)
- `linguistic-epiphanies-2026-04-19.md` (466 lines, E13-E27)
- `fractal-codec-argmax-regime.md` (256 lines, orthogonal thread)

**Decisions for future PRs to respect:**
- Finnish object marking uses Nominative/Genitive/Partitive, NOT Latinate Accusative (except personal pronouns).
- Russian 6 cases include Instrumental (not omitted).
- Each language gets its native case terminology.
- Never spawn Haiku subagents.
- Explore subagents → Sonnet, `general-purpose` grindwork → Sonnet, accumulation → Opus.

---

## #209 — sandwich layout + bipolar cells (merged 2026-04-19)

**Confidence (2026-04-19):** Working. All tests green at merge time.

**Added:**
- `CrystalFingerprint::Structured5x5` uses sandwich layout: 3,125 cells in middle (dims 3437..6562), 5 quorum floats (6562..6567), quorum sentinel (6567), plus leading/trailing role-binding space.
- Bipolar cell encoding: `u8 0..=255 → f32 [-1, 1]` via `v/127.5 - 1.0`.
- Lossless bundle/unbundle between Structured5x5 ↔ Vsa10kF32 sandwich.
- Codex-review fixes: Binary16K aliasing, i8 /128 clamp, `quorum: None` sentinel.

**Locked:**
- **VSA operations stay in `ndarray::hpc::vsa`** (bind, unbind, bundle, permute, similarity, hamming, sequence, clean). DO NOT duplicate in contract.
- **10K f32 Vsa10kF32 (40 KB) is lossless under linear sum**, not a wire-only format; lancedb natively handles 10K VSA.
- **Signed 5^5 bipolar is lossless**; unsigned / bitpacked binary is lossy via saturation.
- **CAM-PQ projection is distance-preserving** (lossless across form transitions).
- **VSA convention is `[start:stop]` contiguous slices**, not scattered bits.
- `Structured5x5` is the native rich form; `Vsa10kF32` is native storage (not passthrough).

**Deferred:**
- PhaseTag types (ladybug-rs owns them).
- Crystal4K 41:1 compression persistence (ladybug-rs owns it).
- ladybug-rs quantum 9-op set port.

**Docs:**
- `crystal-quantum-blueprints.md` (existing, cross-referenced)
- Cross-repo-harvest H1-H14 (Born rule, phase tag, interference, Grammar Triangle ≡ ContextCrystal(w=1), NSM ≡ SPO axes, FP_WORDS=160, Mexican-hat, Int4State, Glyph5B, Crystal4K, teleport F=1, 144-verb, Three Mountains).

---

## #208 — grammar + crystal + AriGraph unbundle (merged 2026-04-19)

**Confidence (2026-04-19):** Working. All tests green at merge time.

**Added:**
- `contract::grammar/` module (6 files): FailureTicket, PartialParse, CausalAmbiguity, TekamoloSlots/TekamoloSlot, WechselAmbiguity/WechselRole, FinnishCase, NarsInference (7 variants), ContextChain (ring buffer), LOCAL_COVERAGE_THRESHOLD = 0.9, MARKOV_RADIUS = 5.
- `contract::crystal/` module (7 files): Crystal trait, CrystalKind, TruthValue, CrystalFingerprint (Binary16K / Structured5x5 / Vsa10kI8 / Vsa10kF32), SentenceCrystal / ContextCrystal / DocumentCrystal / CycleCrystal / SessionCrystal.
- `lance-graph::graph::arigraph::episodic`: unbundle_hardened / unbundle_targeted / rebundle_cold with ndarray::hpc::bitwise::hamming_batch_raw SIMD dispatch under `ndarray-hpc` feature.
- `UNBUNDLE_HARDNESS_THRESHOLD = 0.8` synchronized in contract + arigraph.

**Locked:**
- **AriGraph lives in-tree** at `lance-graph/src/graph/arigraph/` (not a standalone crate). 4696 LOC transcoded from Python AdaWorldAPI/AriGraph.
- **Crystals unbundle when hardness ≥ 0.8.** Rebundle for cold entries.
- **FailureTicket carries SPO × 2³ × TEKAMOLO × Wechsel decomposition** plus coverage + attempted_inference + recommended_next.
- **Finnish 15 cases, Russian 6 cases, Turkish 6 cases** + agglutinative chain, German 4 cases, Japanese particles — each in native terminology.

**Deferred:**
- DeepNSM emission of FailureTicket (D2, Phase 2).
- Grammar Triangle bridge into DeepNSM (D3, Phase 2).

**Docs:**
- `integration-plan-grammar-crystal-arigraph.md` (E1-E12 epiphanies).
- `crystal-quantum-blueprints.md` (Crystal vs Quantum modes).
- `endgame-holographic-agi.md` (5-layer stack).

---

## #207 — session capstone + Wikidata plan (merged 2026-04-18)

**Confidence (2026-04-19):** Working. All tests green at merge time.

**Added:**
- `session-capstone-2026-04-18.md` — 8 epiphanies (E1-E8), Sleeping Beauties (SB1-7), Missing Bridges (MB1-5), Known Brittle (KB1-5), priority map.
- `wikidata-spo-nars-at-scale.md` — 1.2B triples → 14.4 GB scale demo plan.

**Locked:**
- **§7 addendum correction:** AriGraph is SHIPPED, not deferred. Invalidates capstone's DD2 and MB3.
- **4-pillar inheritance** (NARS + thinking + qualia + proprioception) is compile-time contract.
- **CMYK (17D experienced) vs RGB (18D observed) qualia** — dim 17 = classification_distance.
- **Vocabulary IS semantics** — glyph names use pretraining-derived associations on purpose.

---

## #206 — state classification pillars (merged 2026-04-18)

**Confidence (2026-04-19):** Working. All tests green at merge time.

**Added:**
- `contract::qualia` — 17-D QualiaVector + 17→11 projection.
- `contract::proprioception` — 7 StateAnchor (Intake/Focused/Rest/Flow/Observer/Balanced/Baseline), 11-D state vector, ProprioceptionAxes, StateClassifier trait, DefaultClassifier, softmax `hydrate()`.
- `contract::world_map` — WorldMapDto + WorldMapRenderer trait + DefaultRenderer.
- `world_model.rs` extended: `qualia`, `axes`, `proprioception`, `cycle_fingerprint`, `timestamp`, `cycle_index`, `is_self_recognised()`, `is_liminal()`.
- `cognitive-shader-driver::sigma_rosetta` — 64 glyphs (Σ11) × 144 verbs (Σ12 = 12 roots × 12 tenses), SigmaState, TriangleGestalt, QuadTriangleGestalt, InteractionKinematic (Σ13 dome), QPL ↔ sigma 17D bridge.
- REST (`serve` feature) + gRPC (`grpc` feature) endpoints.
- `examples/villager_ai.rs` — Pumpkin NPC AI reference.

**Locked:**
- **Proprioception in the contract** = ontological self-recognition by compilation.
- **WorldMapRenderer pattern** — drop-in framing trait keeps contract vocabulary neutral.
- **Pumpkin / villager AI framing** for externally-visible examples (corporate disguise).
- **`proprioception::axes` named fields** (drive / arousal / dominance / …) — public API.
- **No code comments or PR descriptions** reveal the disguise scrubbing. Everything personal goes to ada-rs.

---

## #205 — engine_bridge + CMYK/RGB qualia (merged 2026-04-18)

**Confidence (2026-04-19):** Working. All tests green at merge time.

**Added:**
- `cognitive-shader-driver::engine_bridge` — ingest_codebook_indices, dispatch_from_top_k, EngineBusBridge, persist_cycle, UNIFIED_STYLES[12] 3-way mapping (ThinkingStyle ↔ StyleSelector ↔ p64 StyleParams).
- CMYK vs RGB qualia decomposition — 17D experienced vs 18D observed, classification_distance as dim 17.

**Locked:**
- **12 UNIFIED_STYLES** are the canonical style inventory (3-way mapping must stay aligned).
- **Named emotion archetypes** (fear/anger/sadness/joy/surprise/disgust) live in engine_bridge as classification references.

---

## #204 — cognitive-shader-driver crate + Shader DTOs (merged 2026-04-18)

**Confidence (2026-04-19):** Working. All tests green at merge time.

**Added:**
- New crate `cognitive-shader-driver` (24 tests).
- `contract::cognitive_shader` — ShaderDispatch / ShaderResonance / ShaderBus / ShaderCrystal, MetaWord (u32 packed: thinking 6 + awareness 4 + nars_f 8 + nars_c 8 + free_e 6), CognitiveShaderDriver trait, ShaderSink commit-adapter.
- `auto_style` — 18D qualia → style ordinal.
- 630K LOC ladybug-rs import into `lance-graph-cognitive` (grammar, spo, learning, world, search, fabric, spectroscopy, container_bs, core_full).
- `crates/holograph` imported from RedisGraph, 10K→16K migration.
- `contract::container` — Container (16K fingerprint) + CogRecord (4KB = meta + content).
- `contract::collapse_gate` — GateDecision, MergeMode.

**Locked:**
- **Shader IS the driver** (role reversal from thinking-engine-first).
- **MetaWord packing layout** — thinking(6) + awareness(4) + nars_f(8) + nars_c(8) + free_e(6).
- **BindSpace struct-of-arrays** — FingerprintColumns (4 planes × 256 u64), EdgeColumn, QualiaColumn (18 f32), MetaColumn (u32).
- **`ShaderBus::cycle_fingerprint: [u64; 256]`** IS `Container` IS `CrystalFingerprint::Binary16K` (same 2 KB backing).
- **No serde in types** (debug-only); wire formats explicit.

**Docs:**
- `cognitive-shader-architecture.md` (canonical architecture reference).

---

## How to Use This File

1. **Opening a session on this workspace:** read the top 3 PRs
   (most recent). That covers ~90 % of what you need to know about
   current state.
2. **Before proposing a new type:** grep this file for the type
   name. If it's listed under Added, stop and read the source.
3. **Before proposing a convention:** grep for the topic. If it's
   listed under Locked, your proposal needs explicit justification
   to overturn it.
4. **When a PR merges:** prepend a new section at the top of this
   file. Old PRs stay — they are the arc.

This file is the fastest bootstrap available for a new session on
this workspace. Load it, then load 1-2 knowledge docs as the domain
triggers, then start working. Target: 3-5 turn cold start, not 30.

---

## 2026-05-05 BACKFILL — PRs #244–#335 (retrofitted from PR descriptions)

> Convention waiver: this section is APPENDED at the bottom of the file rather than PREPENDED, because governance permits only `tee -a` writes. Entries within this section are newest-first by PR number. Header dated; future PR_ARC entries should resume the standard PREPEND-at-top convention once a Write/Edit channel is restored, or continue this backfill section.

---

### #335 — Claude/thought cycle soa integration plan (merged 2026-05-05)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:** Two new knowledge docs: `.claude/knowledge/gaussian-splat-cam-plane-workaround.md` and `.claude/knowledge/entropy-budget-codebook-superposition.md`; 12-commit PR with 5835 additions across 12 files (full body is a bare file list — no template sections present).
**Locked:** —
**Deferred:** —
**Docs:** `.claude/knowledge/gaussian-splat-cam-plane-workaround.md`, `.claude/knowledge/entropy-budget-codebook-superposition.md`

---

### #330 — docs: add Cursor Cloud specific instructions to AGENTS.md (merged 2026-05-01)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:** `AGENTS.md` `## Cursor Cloud specific instructions` section documenting ndarray sibling path requirement, CI-gated check commands, excluded-crate fmt-drift inventory, bgz-tensor pre-existing failures.
**Locked:** ndarray must be cloned to `/ndarray` for path deps to resolve; 5 bgz-tensor failures are known/not-CI-gated.
**Deferred:** —
**Docs:** `AGENTS.md`

---

### #329 — style: apply rustfmt to contract lib.rs + python bindings (Tier-A drift) (merged 2026-05-01) [infra/format]

Tier-A rustfmt drift sweep: `lance-graph-contract/src/lib.rs` (sigma_propagation module order), `lance-graph-python/src/catalog.rs`, `lance-graph-python/src/graph.rs`. No semantic change.

---

### #328 — ci(test): add lance-graph-contract unit tests to the test gate (merged 2026-05-01) [infra/format]

Adds `cargo test --manifest-path crates/lance-graph-contract/Cargo.toml --lib` step to `rust-test.yml` so contract-crate logic regressions trip CI before merge.

---

### #327 — style(shader-driver): drop double-space alignment in bindspace.rs comments (merged 2026-05-01) [infra/format]

Two-line rustfmt drift fix in `cognitive-shader-driver/src/bindspace.rs` introduced by PR #323.

---

### #326 — fix(sigma-propagation): use non-identity seed in log_norm_growth_negative test (merged 2026-05-01)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:** Corrected test `log_norm_growth_negative_when_m_attenuates` to seed at `4·I` (not `I`) so attenuation actually reduces log-norm; comment documents the `seed = I` trap for future readers.
**Locked:** `log_norm_growth` measures signed change in log-Frobenius distance from identity; seeding at `I` makes growth structurally non-negative regardless of M.
**Deferred:** Extending workspace test job to cover `lance-graph-contract` beyond clippy (see PR #328).
**Docs:** —

---

### #325 — chore(toolchain): bump pin from 1.94.0 to 1.94.1 (merged 2026-04-30) [infra/format]

`rust-toolchain.toml` channel bumped to `1.94.1` to match sibling repos; policy comment added.

---

### #324 — feat(shader-driver): Pillar-7 α-front-to-back-merge sink mode (B5) (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `MergeMode::AlphaFrontToBack` (= 3) in `lance-graph-contract::collapse_gate`, `ALPHA_SATURATION_THRESHOLD = 0.99`
- `ShaderHit::confidence_to_alpha()` helper
- `AlphaComposite` carrier + `ShaderCrystal::alpha_composite` field
- `ShaderDispatch.merge_override` + `alpha_saturation_override`
- Stage [7] in `ShaderDriver::dispatch()` dispatches on effective MergeMode; Kerbl-2023 EWA loop replaces top-K only when `AlphaFrontToBack` selected
**Locked:** Existing Bundle / Xor / Superposition paths bit-exact unchanged; edits local to stage [7].
**Deferred:** —
**Docs:** —

---

### #323 — feat(cognitive-shader-driver): add Σ-codebook-index column to FingerprintColumns (B2) (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `FingerprintColumns.sigma: Box<[u8]>` (1 byte/row, index into 256-entry Σ codebook)
- `FingerprintColumns::zeros(len)` allocates sigma alongside existing planes
- `sigma_at(row)` / `write_sigma(row, idx)` accessors
- `BindSpace::byte_footprint` updated to 71777 (+1)
**Locked:** Σ codebook itself not loaded here (B3 concern); no public API breaks.
**Deferred:** B3 codebook static + boot-load-from-disk; B4 shader-driver Σ-propagate in dispatch stage.
**Docs:** —

---

### #322 — feat(contract): promote EWA-Sandwich Σ-propagation kernel to lance-graph-contract (B1) (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `crates/lance-graph-contract/src/sigma_propagation.rs` (~520 LOC): `Spd2` (2×2 SPD packed), `ewa_sandwich`, `ewa_inverse`, `log_norm_growth`, `pillar_5plus_bound`
- `pub mod sigma_propagation` in `lib.rs`
- 12 unit tests; 13 total (one was broken at merge, fixed by #326)
**Locked:** `crates/jc/src/ewa_sandwich.rs` unchanged (proof harness, zero-deps, regression-certificate posture preserved); contract module is canonical production surface.
**Deferred:** Hardware backends (AMX/MKL via ndarray); BindSpace integration (B2/B3/B4).
**Docs:** —

---

### #321 — fix: 10 pre-existing test failures (cosine_distance, arigraph orchestration, parse_triplets) (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `cosine_distance()` restored `1.0 -` inversion (SIMD helper returns similarity, not distance)
- `GraphSensorium::suggested_bias()` Stagnant condition moved before Explore (was unreachable)
- `switch_mode` clears `quality_window` on regime change to prevent stale-evidence restore failure
- `XaiClient::parse_triplets` argument order fixed: `Triplet::new(s, o, r, t)` not `(s, r, o, t)`
**Locked:** 846/846 `lance-graph` unit tests pass post-fix.
**Deferred:** Type-deduplication of `GraphSensorium` across `orchestrator.rs` and `sensorium.rs` (pre-existing tech debt, orthogonal).
**Docs:** —

---

### #320 — ci: declare rustfmt + clippy as pinned-toolchain components (merged 2026-04-30) [infra/format]

`rust-toolchain.toml` gains `components = ["rustfmt", "clippy"]` so pinned channel installs them at bootstrap; fixes `cargo-fmt not installed for 1.94.0` CI failure.

---

### #319 — fix(transcode): per-month day-validity in parse_iso_date_to_days (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:** Per-month max-day validation + Gregorian leap-year rule (century rule: div-by-100-not-400 is NOT leap) in `parse_iso_date_to_days`; rejects April-31, Feb-30, 1900-02-29, etc. 2 new tests.
**Locked:** Howard Hinnant `civil_to_days` itself is correct; gate inputs before calling it.
**Deferred:** `Date(Month)` / `Date(Year)` precision parsing (round-4).
**Docs:** —

---

### #316 — feat(transcode): round-3 typed-value resolver for triples_to_batch (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `triples_to_batch_with_resolver(soa, triples, resolver)` — closure `Fn(&str) -> Option<Vec<u8>>` maps `object_label` to bytes; typed Arrow scalars emitted per `SemanticType`
- `parse_iso_date_to_days` (Howard Hinnant civil_to_days, public-domain)
- `TranscodeError::ParseFailure { column, reason }` for Required-column parse failures
- Type mapping: `Currency→Float32`, `Date→Date32`, `CustomerId/InvoiceNumber→UInt64`, rest `Utf8`
- 21 tests total (+8 new)
**Locked:** Required column parse failure → typed error (not silent null); Optional column parse failure → null. `triples_to_batch` (round-1 lenient-Utf8) unchanged for callers without resolver.
**Deferred:** `Date(Month)`/`Date(Year)` precisions; `Geo`/`File`/`Image` typed reconstruction; async resolver; `FixedSizeListF32`/`FixedSizeBinary` wide-payload resolver.
**Docs:** —

---

### #315 — ci: revert ndarray-branch pin — PR #115 has landed on master (merged 2026-04-30) [infra/format]

Removes temporary `ref: claude/continue-lance-graph-ndarray-Ld786` CI pin from `rust-test.yml` and `style.yml` (4 occurrences); ndarray PR #115 merged 2026-04-30 07:01 UTC.

---

### #314 — docs(vision): clear post-F1 staleness items (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:** `.claude/medcare-foundry-vision.md` §1–§4 staleness cleared: DRAFT header removed, §2/§3/§4 forward-tense rewritten with actual PR anchors (#278, #280, #284, #302), latency benchmark explicitly split from parity (shipped) vs benchmark (not started).
**Locked:** No latency/throughput numbers claimed; tone rule ("brutally honest, no hype") preserved.
**Deferred:** —
**Docs:** `.claude/medcare-foundry-vision.md`

---

### #313 — feat(transcode): Phase-2-B triples_to_batch (ExpandedTriple stream → RecordBatch) (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `triples_to_batch(soa, &[ExpandedTriple]) → RecordBatch` — N subjects → N rows lex-sorted by `subject_label`
- `round1_lenient_schema(soa)` — all body columns nullable `Utf8`
- `TranscodeError::{EntityTypeMismatch, BadSubjectLabel}` variants
- 7 new tests (19 total in `transcode::zerocopy`)
**Locked:** `object_label` is FNV-1a encoded so round-1 keeps all body as `Utf8`; typed value reconstruction is round-3. Undeclared predicates silently dropped (BBB outer-view rule).
**Deferred:** Typed-value reconstruction (round-3, PR #316); async SpoStore reader; fingerprint→entity_id side-table (consumer-side state).
**Docs:** —

---

### #312 — feat(transcode): Phase-2-A pushdown classification (Inexact for recognised filters) (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:** `OntologyTableProvider::supports_filters_pushdown` classifies `entity_type=`, `entity_id=`, `predicate=`, `nars_frequency>`, `nars_confidence>` as `Inexact`; unknown columns / undeclared entity types as `Unsupported`; symmetric `lit op col` handled. 7 new tests (11 total).
**Locked:** Classification is `Inexact` (not `Exact`) until Phase-2-B SpoStore scan replaces MemTable delegate. DataFusion must still apply filter as residual.
**Deferred:** Phase-2-B: replace MemTable scan with custom `ExecutionPlan` walking SpoStore; flip to `Exact` once trusted.
**Docs:** —

---

### #311 — docs(vision): mark F1 shipped, restate next deliverable as F2 RBAC wiring (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:** `.claude/medcare-foundry-vision.md` §7 rewritten: "F1 has shipped" with concrete cross-links (MedCareV2 #1/#2/#3, medcare-rs #71, lance-graph #309); F2 RBAC+audit on read path named as next posture.
**Locked:** §1–§6 unchanged; tone rule preserved.
**Deferred:** F1 latency benchmark (correctness shipped; benchmark not started).
**Docs:** `.claude/medcare-foundry-vision.md`

---

### #310 — feat(transcode): r2 fixes — typed Arrow + codec_route + partial writes + CachedOntology + route validation (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `arrow_type_for_semantic`: `Currency→Float32`, `Date→Date32`, `CustomerId/InvoiceNumber→UInt64` (was all `Utf8`)
- `CachedOntology` upstream with `Arc<Ontology>` + eagerly-projected DTOs per locale
- `validate_route(route, ontology) → Result<(), String>` + 4 tests
- `from_columns_partial` — allows missing Optional/Free columns; Required + undeclared still rejected
- `route_for_column` reads `OuterColumn.codec_route` (was heuristic `route_tensor`)
**Locked:** `route_for_column` reads contract's own field — transcode layer can never disagree with schema author's intent.
**Deferred:** Phase 4 (NARS cold sink); Phase 5 (BindSpace → outer-DTO direction).
**Docs:** —

---

### #309 — feat(callcenter::transcode): outer ↔ inner ontology mapper + parallelbetrieb (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `lance-graph-callcenter::transcode` submodule with 5 modules: `zerocopy`, `cam_pq_decode`, `spo_filter`, `ontology_table`, `parallelbetrieb`
- `OuterColumn`/`OuterSchema`/`OwnedColumn`/`from_columns` (zerocopy, refuses undeclared columns)
- `CamPqDecoder` trait + `PassthroughDecoder` for `CodecRoute::{Skip, Passthrough}`
- `SpoFilterTranslator`: SQL filter terms → `SpoLookup` via `fnv1a`
- `OntologyTableProvider`: DataFusion `TableProvider` over `(Ontology, entity_type)` backed by `MemTable`
- `DriftEvent`/`DriftKind`/`Reconciler` trait (parallelbetrieb, MySQL↔DataFusion reconciler)
- 26 tests; `async-trait = "0.1"` dep added
**Locked:** `parallelbetrieb` is explicitly a transitional bandaid; no Foundry primitive in that module; no silent reconciliation.
**Deferred:** Phase 2-B (SpoStore reader replacing MemTable scan); Phase 4 (NARS cold sink); Phase 5 (BindSpace → outer-DTO reverse path).
**Docs:** —

---

### #308 — feat: bilingual ontology DTO surface + bgz-tensor workspace inclusion (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `Locale`/`Label`/`OntologyBuilder.locale+label` fields in `lance-graph-contract::ontology`
- `lance-graph-callcenter::ontology_dto`: `OntologyDto`, `EntityTypeDto`, `PropertyDto`, `LinkTypeDto`, `ActionTypeDto` + `OntologyDto::from_ontology(ontology, locale)`
- `smb_ontology()` (Customer/Invoice/TaxDeclaration) and `medcare_ontology()` (Patient/Diagnosis/LabResult/Prescription) canonical examples
- `bgz-tensor` moved from `exclude` to workspace `members` with `ndarray_compat.rs` shim
- 194/200 bgz-tensor tests (6 pre-existing failures in experimental paths)
**Locked:** `OntologyDto::from_ontology` is the single external projection function; bilingual labels travel with the ontology.
**Deferred:** OntologyDelta column on BindSpace (Q3); DM-8b Lance-backed PostgREST; AU-1 AuditEntry shape unification; TT-1 `scan_as_of`; ndarray SIMD dtype gaps.
**Docs:** —

---

### #307 — refactor: dedup FNV-1a — one canonical hash::fnv1a in lance-graph-contract (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:** `crates/lance-graph-contract/src/hash.rs` — `const fn fnv1a(bytes) → u64` + `fnv1a_str` convenience + 4 tests. 8 call sites updated; 2 copies remain in `thinking-engine` and `holograph` (don't depend on contract, annotated).
**Locked:** Canonical FNV vectors pinned: `""→0xcbf29ce484222325`, `"a"→0xaf63dc4c8601ec8c`, `"foobar"→0x85944171f73967e8`.
**Deferred:** —
**Docs:** —

---

### #306 — feat(G4): verb_table tense modulation (Quirk CGEL grounded) (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- 12 `VerbFamily` base priors across 4 semantic categories (Change/Action/State/Discovery)
- `tense_modifier(Tense) → SlotPriorDelta` — Quirk et al. CGEL §4.21–4.27 grounded; 7 tense modifiers
- `SlotPrior::combine(delta)` with `[0.0, 1.0]` clamp
- `Tense::ALL` const array in `role_keys.rs`
- 144 cells now have 144 unique values (was 12 broadcast)
**Locked:** Tense modulation is linguistically grounded (Quirk CGEL cited); Perfect/Imperative priors differ from Present.
**Deferred:** —
**Docs:** —

---

### #305 — feat(G3): DisambiguateOpts builder + deepnsm caller wiring real fingerprint (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `DisambiguateOpts` builder replaces 4-method explosion; legacy methods `#[deprecated]`
- `crates/deepnsm/src/disambiguator_glue.rs`: `sign_binarize_to_binary16k(&[f32]) → Box<[u64; 256]>` + `disambiguate_with_trajectory` (MarkovBundler→ContextChain bridge)
- `sign_binarize`: f32 bundle → 16,384 bits (v≥0.0 → 1) packed into 256 u64 words → `CrystalFingerprint::Binary16K`
**Locked:** `Binary16K` is an enum variant of `CrystalFingerprint`, not a newtype; sign-binarization happens in deepnsm (not contract) to preserve zero-dep invariant.
**Deferred:** —
**Docs:** —

---

### #304 — feat(G1): Pearl 2³ causality footprint with PAD-model qualia mapping (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `compute_pearl_mask()` derives 3-bit mask from SPO triple (S=bit2, P=bit1, O=bit0), matches `causal-edge::pearl::CausalMask` repr
- PAD-model qualia footprint replaces neutral 0.5 placeholder; Agency←Dominance, Activity←Activation, Affection←Arousal
- `#[cfg(feature = "grammar-triangle")]` removed from core Pearl mask code
**Locked:** Pearl mask uses 3-bit u8 without importing `CausalEdge64` — deepnsm dep tree stays clean.
**Deferred:** —
**Docs:** —

---

### #303 — feat(F6): FNV-1a scent with scent_u64 accessor + birthday collision tests (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `scent()`: FNV-1a hash of canonical hex path, folded to u8; replaces XOR-fold stub
- `scent_u64()`: full 64-bit FNV-1a digest (CAM-PQ Phase C downstream needs unfolded bits)
- FNV-1a inline (no crate dep); `scent_stub()` deprecated alias preserved
- `lance_membrane.rs` migrated to `scent()`; 10 tests
**Locked:** FNV-1a inline; `scent_u64()` fold-matches `scent()`.
**Deferred:** —
**Docs:** —

---

### #302 — feat(F3): LanceAuditSink with temporal timestamps + full schema round-trip (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `LanceAuditSink` implementing `AuditSink` — writes `AuditEntry` to Lance dataset via Arrow RecordBatch append
- Temporal timestamp: `DataType::Timestamp(Millisecond, Some("UTC"))` for DataFusion temporal predicates
- Full schema: `tenant_id`, `actor_id`, `statement_hash`, `timestamp`, `action`, `rls_predicates_added`, `rewritten_plan`
- `scan_back(n)` uses `scanner.limit(Some(n), Some(skip))` (O(1), not full-scan)
- Feature-gated behind `audit-log`; 14 tests
**Locked:** Lance v4 `Scanner::limit(Option, Option)` verified at source line 1344.
**Deferred:** —
**Docs:** —

---

### #301 — feat(F1): ColumnMaskRewriter with full-tree expression walk + Hash UDF hard-fail (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `ColumnMaskRewriter` as DataFusion `OptimizerRule` — `LogicalPlan::map_expressions` + `Expr::transform_down` covers Filter/Projection/Aggregate/Join/Sort
- `NotYetWiredHashUdf` (ScalarUDFImpl) binds at plan time, errors loudly at execute — no silent placeholder
- Truncate via `substr(col, 1, n)` (DataFusion built-in unicode substr)
- `TreeNodeRecursion::Jump` after Column→ScalarFunction wrap prevents infinite recursion
- 15 policy tests, 3 failing-first tests proving the WHERE/JOIN/GROUP BY leak existed
**Locked:** Full-tree walk is security-critical; initial impl only rewrote Projection (leaked through Filter/Aggregate).
**Deferred:** —
**Docs:** —

---

### #300 — feat(LF-12): Pipeline DAG with StepId derivation + OrchestrationBridge adapter (merged 2026-04-30)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `PipelineDag` (482 LOC): Kahn's algorithm topological executor with `depends_on` DAG edges
- `UnifiedStep::id()` computes FNV-1a over `step_id` bytes (eliminates `id: 0` landmine across 4 callers)
- `execute_via_bridge<B>(&self, bridge: &B)` wires PipelineDag into canonical contract pattern
- Cycle detection (multi-node + self-loop); `PipelineError::{MissingDependency, CycleDetected, StepFailed, DuplicateStepId}`
- `depends_on: Vec<StepId>` added to `UnifiedStep`; 12 tests
**Locked:** Synchronous-only executor; async fan-out is explicit follow-up.
**Deferred:** Async fan-out executor.
**Docs:** —

---

### #299 — revert #294/#295/#296 + clean on top (merged 2026-04-29)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source. REVERT PR — reverts #294, #295, #296.

**Added:**
- Reverts: #296 (COCA-Bundle idea — premise false: CAM_PQ IS COCA-based, one pipeline), #295 (data-available followup — inherited wrong routing from #294), #294 (probe-queue routing assessment — M1 wrongly routed to bgz-tensor/CHAODA; P2-P4 wrongly routed to standalone bgz-tensor calibrate instead of shader-lab WireSweep)
- Clean replacement content in `bf16-hhtl-terrain.md`: M1→`polarquant_hip_probe.rs`+`turboquant_correction_probe.rs`; P2-P4→shader-lab WireSweep; architecture notes (CAM_PQ=COCA, ICC family heel, CascadeConfig, jitson JIT)
- `EPIPHANIES.md` FINDING: existing lab infra covers M1/P2-P4
- `IDEAS.md` Open: inverted-pyramid awareness streaming via CausalEdge64
**Locked:** CAM_PQ IS COCA-based (not separate); P2-P4 belong in shader-lab WireSweep, not standalone jc.
**Deferred:** —
**Docs:** `.claude/knowledge/bf16-hhtl-terrain.md`, `.claude/board/EPIPHANIES.md`, `.claude/board/IDEAS.md`

---

### #296 — ideas: COCA-Bundle vs Jina-CLAM bucket comparison (Probe candidate) (merged 2026-04-29) — REVERTED by #299

**Confidence (2026-05-05):** Retrofitted from PR description — REVERTED by PR #299. Content removed from main.

**Added:** `IDEAS.md` Open entry for COCA-Bundle vs Jina-CLAM bucket comparison probe candidate.
**Locked:** —
**Deferred:** —
**Docs:** `.claude/board/IDEAS.md`

---

### #295 — docs(probe-queue): followup — release assets ARE available for P2/P3/P4 (merged 2026-04-29) — REVERTED by #299

**Confidence (2026-05-05):** Retrofitted from PR description — REVERTED by PR #299. Content removed from main.

**Added:** `bf16-hhtl-terrain.md` updated with concrete download/probe sequence using release assets; P2/P3/P4 status changed from "needs production data" to "data available".
**Locked:** —
**Deferred:** —
**Docs:** `.claude/knowledge/bf16-hhtl-terrain.md`

---

### #294 — docs(probe-queue): assess P2/P3/P4 routing — honest "needs production data" (merged 2026-04-29) — REVERTED by #299

**Confidence (2026-05-05):** Retrofitted from PR description — REVERTED by PR #299. Content removed from main.

**Added:** `bf16-hhtl-terrain.md` probe routing table: M1→bgz-tensor/CHAODA, P1→`jc` (PASS), P2-P4→bgz-tensor calibrate feature. `EPIPHANIES.md` FINDING.
**Locked:** —
**Deferred:** —
**Docs:** `.claude/knowledge/bf16-hhtl-terrain.md`, `.claude/board/EPIPHANIES.md`

---

### #293 — jc: drain Probe P1 (γ-phase-offset ranking discrimination) → PASS (merged 2026-04-29)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `crates/jc/src/probe_p1_gamma_phase.rs` (~290 LOC, 11 tests)
- `crates/jc/examples/probe_p1.rs`
- `bf16-hhtl-terrain.md` P1 status updated: NOT RUN → PASS
- `EPIPHANIES.md` FINDING; `IDEAS.md` triple-entry (Open status flipped + Implemented appended)
**Locked:** P1 PASS confirms γ+φ pre-rank discrete selector VALID: min Spearman ρ = -0.963 (Dupain-Sós signature); three production crates (`bgz-tensor::gamma_phi`, `gamma_calibration`, `projection`) rest on this axiom.
**Deferred:** P2/P3/P4 remain open (now re-routed per #299).
**Docs:** `.claude/knowledge/bf16-hhtl-terrain.md`, `.claude/board/EPIPHANIES.md`, `.claude/board/IDEAS.md`

---

### #292 — docs(board): posthoc-correct PRs #290 #291 — re-file via canonical board mechanism (merged 2026-04-29)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- CONJECTURE banners added to `IDEA_JOURNAL_2026_04_29_STREAMING_HYDRATION.md` and `IDEA_JOURNAL_2026_04_29_FUTURE_PILLARS.md`
- `IDEAS.md` 5 new Open entries: Safetensor-Streaming, Family-Bounds fractal, Pillar 7 (LIKELY-REDISCOVERY), Pillar 8 Adaptive Densification, Pillar 9 SH-Coefficients (TOUCHES PRODUCTION CODE)
- `EPIPHANIES.md` 2 new entries: CORRECTION (board/probe-queue discipline skipped) + FINDING (Pillars 5+/5++/6 close concentration family)
**Locked:** Pillar 7 LIKELY-REDISCOVERY — `bgz-tensor::cascade.rs` may already cover front-to-back α-blending; pre-implementation read mandatory. Pillar 9 TOUCHES PRODUCTION CODE — hold until explicit architecture decision.
**Deferred:** —
**Docs:** `.claude/board/IDEAS.md`, `.claude/board/EPIPHANIES.md`

---

### #291 — docs: idea journal — proposed application pillars 7/8/9 captured (merged 2026-04-29)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:** `.claude/IDEA_JOURNAL_2026_04_29_FUTURE_PILLARS.md` (~270 LOC) — Pillars 7/8/9 with concrete PASS criteria, effort estimates, reuse inventory, sequencing options.
**Locked:** —
**Deferred:** —
**Docs:** `.claude/IDEA_JOURNAL_2026_04_29_FUTURE_PILLARS.md`

---

### #290 — docs: idea journal — streaming-hydration + fractal-codec captured before dilution (merged 2026-04-29)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:** `.claude/IDEA_JOURNAL_2026_04_29_STREAMING_HYDRATION.md` (~170 LOC) — Idea 1 (Safetensor streaming as n-dimensional meaning accumulation) and Idea 2 (family-bounds as global fractal coding), explicitly separated to prevent Ada Hammer-sucht-Nagel failure mode.
**Locked:** —
**Deferred:** —
**Docs:** `.claude/IDEA_JOURNAL_2026_04_29_STREAMING_HYDRATION.md`

---

### #289 — jc: Pillar 6 — EWA-Sandwich Σ-push-forward (cant-stop-thinking math foundation) (merged 2026-04-29)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `crates/jc/src/ewa_sandwich.rs` (~440 LOC, 7 tests)
- Pillar 6: `M·Σ·Mᵀ` sandwich preserves PSD by construction (10000/10000 hops); log-norm concentration tightness 1.467× KS log-normal-corrected bound
**Locked:** Multi-hop path propagation `Σ_path = M_path · Σ_0 · M_pathᵀ` preserves SPD cone at any depth; bounded geometric multiplicative error (not O(n) arithmetic).
**Deferred:** Pillar 7 (Front-to-Back α-Akkumulation), Pillar 8 (Adaptive Densification), Higher-dim SPD (3×3), real-stream CV-bound validation.
**Docs:** —

---

### #288 — jc: Σ-Codebook Viability Probe — empirically rules out CausalEdge64 8→16 byte expansion (merged 2026-04-29)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `crates/jc/src/sigma_codebook_probe.rs` (~370 LOC, 6 tests)
- `crates/jc/examples/sigma_probe.rs`
- Result: R²=0.9949 at k=256 — CODEBOOK VIABLE; 8→16 byte CausalEdge64 expansion ruled out
**Locked:** CausalEdge64 stays 8 bytes; HighHeelBGZ 240-edges/2KB hard limit preserved; Σ-Codebook Option A (3.5 KB workspace-wide + 1-byte sidecar) or Option C (SchemaSidecar Block 14/15).
**Deferred:** `CausalEdgeTensor` design (caller choice: 9-byte sidecar or SchemaSidecar). Not a Pillar — diagnostic probe, separate category.
**Docs:** —

---

### #287 — jc: Pillar 5++ — Düker-Zoubouloglou Hilbert-space CLT (closes the concentration family) (merged 2026-04-29)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `crates/jc/src/dueker_zoubouloglou.rs` (~280 LOC, 6 tests)
- Pillar 5++: Breuer-Major Theorem 2.1 verified — bundle-of-N-fingerprints (AR(1) in ℝ^16384) converges to Gaussian limit in ℓ²; empirical trace 49101.2 vs predicted 49152.0 (0.103% error)
**Locked:** Substrate fingerprint dimension d=16384 certified: bundle-of-N partial sums obey Hilbert-space CLT with explicit closed-form limit covariance; Düker-Zoubouloglou 2024 (arXiv:2405.11452).
**Deferred:** Operator G ≠ identity (Hermite rank ≥ 2); Pillar 5++ Application Section 6 neural-operator CLT.
**Docs:** —

---

### #286 — jc: Pillar 5+ — Köstenberger-Stark concentration on Hadamard 2×2 SPD (merged 2026-04-29)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `crates/jc/src/koestenberger.rs` (~370 LOC, 8 tests)
- Pillar 5+: Theorem 1 (Köstenberger-Stark arXiv:2307.06057) verified — inductive mean on 2×2 SPD; measured 96.9% of predicted ceiling (tightness 0.969×)
**Locked:** Foundation for `CausalEdgeTensor` Σ-aggregation on PSD manifold (non-iid, with Huber-ε contamination tolerance); certifies architecture BEFORE production edge code.
**Deferred:** `CausalEdgeTensor` itself; `propagate()` in `holograph::resonance`; Pillar 5++ (Düker-Zoubouloglou).
**Docs:** —

---

### #285 — Re-land #283 unlocks (Quantum, Disambiguator, verb_table, animal-farm harness) — orphaned by merge order (merged 2026-04-29)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- YAML robustness + `Instrument` variant + persistence + 144-cell verb taxonomy table
- `Trajectory` audit-hash bridge + generalised `Disambiguator` trait + `PhaseTag`/`HolographicMode` (Quantum mode) + Animal Farm forward-validation harness
- `verb_table`, `disambiguator`, `trajectory_audit`, `quantum_mode` modules wired; `u128::MAX as f32 → infinity` overflow fix
**Locked:** —
**Deferred:** —
**Docs:** —

---

### #284 — Re-land #281 unlocks (PolicyRewriter, DomainProfile) — orphaned by merge order (merged 2026-04-29)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- Generalised `PolicyRewriter` trait + `ColumnMaskRewriter` (epiphany E1)
- `DomainProfile` with HIPAA-grade thresholds + verb taxonomy seam (E5) + `Display` impl on `StepDomain`
- `policy` module wired + `trajectory-audit` feature stub
**Locked:** —
**Deferred:** —
**Docs:** —

---

### #282 — fix: Grammar/Markov hardening — slice unification, kernel wiring, parser tests, triangle distance (merged 2026-04-29)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- Slice coordinate unification: `markov_bundle.rs` imports from `role_keys` exclusively (canonical 2000/2000/2000/1500/1500, not equal-partition 3277)
- Integration test `integration_role_alignment.rs` (8 tests — the slice alignment gate)
- `rotate_right` post-bundle rotation removed (was corrupting role-slice alignment)
- `coherence_at_with_kernel(i, kernel)` + `total_coherence_with_kernel(kernel)` wired
- Bundle normalization (divide by `sum(|weights|)`)
- NSM-prime ID set replaced heuristic with explicit `NSM_PRIME_IDS: HashSet`
- `compute_classification_distance` normalized Hamming over qualia fingerprint (was 0.0 stub)
- `role_candidates` parameterized with explicit `threshold` + `top_k`
**Locked:** Subject/Predicate/Object/TEKAMOLO slice start:stop coordinates must come from `role_keys` canonical allocation; post-bundle rotation is forbidden.
**Deferred:** ASCII→unicode restore on grammar-landscape.md; end-to-end coref test with ±5 trajectory.
**Docs:** —

---

### #280 — fix: Foundry hardening — sealed RLS, VecDeque audit, URL decode, Plugin handshake (merged 2026-04-29)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `RegistryMode::Sealed` as default; unregistered TableScans return DataFusionError::Plan; fail-open requires explicit `RlsPolicyRegistry::fail_open("reason")`
- `RlsContext::new` validates non-empty `tenant_id`/`actor_id`; `new_unchecked` preserved for system contexts
- Audit ring: `Vec::remove(0)` → `VecDeque::pop_front()` (O(1))
- FNV-1a hash replacing `DefaultHasher` (cross-build deterministic)
- PostgREST `%XX`/`+` URL decoding in filter values/select/order
- `GATE_DAMPING_FACTOR = 0.5` separates `gate_f` from `free_e`
- `Acquire`/`Release` atomics on `current_scent` and `current_rationale_phase`
- `Plugin` trait with `name()`, `depends_on()`, `seal()` for boot-time prerequisite verification
- Table name validation rejects path traversal + non-alnum characters
- `AuditEntry.rewritten_plan: Option<String>` for retroactive policy enforcement
- 58 tests
**Locked:** Sealed RLS registry is the default; deny-by-default contract from `foundry-roadmap.md`.
**Deferred:** Integration test: sealed RLS + audit log captures rewritten plan.
**Docs:** —

---

### #279 — feat: DeepNSM grammar parser — Markov ±5 bundler, role keys, thinking styles (merged 2026-04-29)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- D0: `grammar-landscape.md` knowledge doc (case inventories, Triangle overview, Markov ±5, 144 verb taxonomy, caveats)
- D4: `ContextChain` reasoning ops (`coherence_at()`, `total_coherence()`, `replay_with_alternative()`, `disambiguate()`, `WeightingKernel` Uniform/MexicanHat/Gaussian with Ricker wavelet)
- D6: `RoleKeySlice` with 13 SPO+TEKAMOLO const slices in 16384-dim VSA space, `LazyLock` arrays for Finnish cases/tenses/NARS inference keys, FNV-64a seeding
- D7: `GrammarStyleConfig` + `GrammarStyleAwareness` with NARS revision lifecycle, `ParamKey`/`ParseOutcome`, zero-dep YAML reader, 12 starter YAML configs mapped to `ThinkingStyle` enum
- D5: `MarkovBundler` with role-indexed VSA bundling (ring buffer, Mexican-hat weighting) + `Trajectory` struct
- D2+D3: `ticket_emit` + `triangle_bridge`
- New features: `contract-ticket`, `grammar-triangle`; 53-60 deepnsm tests
**Locked:** 16384-dim VSA layout (not 10000 from spec) per LF-2 migration.
**Deferred:** WeightingKernel::MexicanHat zero-crossing verification; end-to-end coref test; ASCII→unicode restore.
**Docs:** `crates/deepnsm/`, `crates/lance-graph-contract/src/grammar/`

---

### #278 — feat: Foundry parity — RLS rewriter, audit log, PostgREST, with_registry (merged 2026-04-29)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- LF-3/DM-7: `RlsPolicyRegistry` as DataFusion `OptimizerRule` — tenant/actor predicate injection on every TableScan
- LF-90: `AuditSink` trait + `InMemoryAuditSink` ring buffer with poison recovery
- DM-8: PostgREST-shape handler stub — `parse_path()` + `EchoHandler` dispatcher (20 tests, no HTTP deps)
- `LanceMembrane::with_registry()` builder
- `StepDomain::Medcare` variant
- `.claude/foundry-roadmap.md` + `.claude/medcare-foundry-vision.md` drafts
- New features: `audit-log`, `postgrest`, `membrane-plugins-rls`, `membrane-plugins-audit`; 35 tests
**Locked:** —
**Deferred:** Manual review of RLS predicate injection on multi-table JOINs; PostgREST filter parsing edge cases (nested paths, unicode table names).
**Docs:** `.claude/foundry-roadmap.md`, `.claude/medcare-foundry-vision.md`

---

### #277 — plan: unified Foundry roadmap for SMB + MedCare consumers (corrects PR #276 data-model framing) (merged 2026-04-28)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:** `.claude/plans/foundry-roadmap-unified-v1.md` (~180 LOC) correcting #276's Binary16K-centric framing. Per-tenant scale (1k–50k entities) → SPO+ontology+Vsa16kF32 hot-path; Binary16K for OSINT-scale only; CAM-PQ at 1M+ aggregated rows.
**Locked:** Data-model must use FormatBestPractices.md §5 scale decision matrix. LF-3/DM-7 RLS rewriter is critical path unblocking both consumers.
**Deferred:** —
**Docs:** `.claude/plans/foundry-roadmap-unified-v1.md`

---

### #276 — plan: Foundry Consumer Parity — shared ontology for SMB + MedCare + UNKNOWN resolutions (merged 2026-04-28)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source. Framing corrected by #277.

**Added:** `.claude/plans/foundry-consumer-parity-v1.md` (186 LOC); resolved 5 callcenter UNKNOWNs: UNKNOWN-2 (Phoenix+Rust), UNKNOWN-3 (no pgwire), UNKNOWN-4 (actor_id=String/JWT sub), UNKNOWN-5 (single root Lance URI env var), §8 PostgREST (CONFIRMED, DM-8 unblocked).
**Locked:** —
**Deferred:** —
**Docs:** `.claude/plans/foundry-consumer-parity-v1.md`

---

### #275 — feat: add lancedb 0.27.2 + pin lance =4.0.0 for exact version compat (merged 2026-04-28)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:** `lancedb = "=0.27.2"` optional dep behind `lancedb-sdk` feature (NOT default); all lance crates pinned to `=4.0.0` exact (lancedb requires exact match).
**Locked:** `lancedb-sdk` not in default features; `=4.0.0` exact pins required for lancedb compat.
**Deferred:** Arrow 58 still blocked (lance 4.0.0 pins `arrow = "^57"`; needs lance 5+).
**Docs:** —

---

### #274 — fix: F-01 identity-tear race + F-08 bounds check + F-09 poison recovery (merged 2026-04-27)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- F-01 CRITICAL: Single `RwLock<ActorState { role, faculty, expert }>` replacing 3 independent locks; identity triple always consistent
- F-09 HIGH: All lock sites use `.unwrap_or_else(|e| e.into_inner())` — poison recovery
- F-08 HIGH: `assert!(cursor < bs.len)` bounds check in `push_typed()` with overflow count
**Locked:** —
**Deferred:** F-10 (actor_id = expert as u64 semantic fix) — requires schema change to `ExternalIntent`; deferred to own commit with downstream coordination.
**Docs:** —

---

### #273 — feat: bump lance 2→4 + datafusion 51→52 + deltalake 0.30→0.31 (merged 2026-04-27)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:** lance 2.0.1→4.0.0, datafusion 51→52, deltalake 0.30→0.31 bumps across workspace; `NamespaceError::invalid_input()` 1-arg fix; `DeltaTableProvider::try_new(snapshot, log_store, scan_config)` migration. Arrow stays at 57.
**Locked:** Arrow 58 blocked until lance 5+; `deltalake 0.32` needs arrow 58 (incompatible).
**Deferred:** `auth-rls` xz2/liblzma collision re-test after merge.
**Docs:** —

---

### #272 — feat: Column H — EntityTypeId on BindSpace (Phase 1 of 4) (merged 2026-04-27)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- `EntityTypeId = u16` type alias + `entity_type_id(ontology, name) → u16` function in `contract::ontology` (1-based, 0 = untyped)
- `entity_type: Box<[u16]>` field on `BindSpace` SoA (+2 bytes/row)
- `BindSpaceBuilder::push_typed()` writes entity_type; `push()` defaults to 0 (backward compat)
- 4 tests (total 261 contract)
**Locked:** 1-based indexing; 0 = untyped sentinel.
**Deferred:** Dispatch-time type binding (Phase 2 — requires novel-pattern-detection logic D-E3); `entity_type_id()` O(N) scan (HashMap cache flagged as future optimization).
**Docs:** —

---

### #271 — plan: BindSpace Columns E/F/G/H — 4→8 SoA integration plan (merged 2026-04-27)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:** `.claude/plans/bindspace-columns-v1.md` (457 LOC, 24 deliverables across 4 phases): Column H (`EntityTypeId u16`), E (`OntologyDelta 32B`), F (`AwarenessColumn [u8; 256]`), G (`ModelRef u32`). Total overhead +366 B/row (+5.9%), 26.2 MB total. Scientific cross-check: 7 SOUND / 7 CAUTION / 0 WRONG.
**Locked:** Build order H→E→F→G has genuine dependency logic. Pearl rung gating (B2) embedded in struct layout.
**Deferred:** Phase 3 (Column F) needs proof-of-concept before full 9-deliverable plan; Phase 4 (Column G) blocked on LF-50/52. No migration path for existing BindSpace consumers documented.
**Docs:** `.claude/plans/bindspace-columns-v1.md`, `.claude/board/INTEGRATION_PLANS.md`

---

### #270 — ci: remove typos spell-check job (too many false positives) (merged 2026-04-26) [infra/format]

Removes `crate-ci/typos` spell-check job from `style.yml`; `cargo fmt --check` remains. Spelling discipline moved to code-review.

---

### #269 — feat: Distance trait + SIMD Hamming/cosine wiring + PaletteDistanceTable + Dockerfile docs (merged 2026-04-26)

**Confidence (2026-05-05):** Retrofitted from PR description — not re-verified against current source.

**Added:**
- SIMD Hamming: `cognitive-shader-driver/src/driver.rs:178` now calls `ndarray::hpc::bitwise::hamming_distance_raw()` (~8-16× speedup AVX-512 VPOPCNTDQ); DataFusion UDF + graph fingerprint Hamming delegated to ndarray
- CI `RUSTFLAGS`: all 4 workflows get `-C target-cpu=x86-64-v3` (AVX2); Dockerfile gets same env var
- `Dockerfile.md` (118 LOC): three-tier build strategy, SIMD dispatch, RUSTFLAGS vs `.cargo/config.toml` override behavior
- `Distance` trait (`distance()`, `similarity()`, `similarity_z()`) + `fisher_z_inverse()` + `mean_similarity_fisher()`; scalar impls for `[u64; 256]`, `[u8; 6]`, `[u8; 3]`; 11 tests
- SIMD cosine/dot in `vector_ops.rs` (4 scalar loops → `cosine_f32_to_f64_simd`/`dot_f64_simd`)
- `bgz17 Palette::build_distance_table()` → 256×256 u16 table (128 KB, L2-resident); `edge_distance(a,b)` O(1)
- `EPIPHANIES.md` Distance dispatch FINDING; `TECH_DEBT.md` TD-DIST-1/2/3 opened and marked PAID same session
**Locked:** Type-intrinsic dispatch (`fp_a.distance(&fp_b)`) — no `dyn`, no enum match; FisherZ inverse for safe averaging across SoA columns.
**Deferred:** —
**Docs:** `Dockerfile.md`, `.claude/board/EPIPHANIES.md`, `.claude/board/TECH_DEBT.md`


---

## (open / pending merge) — feat(lance-graph-ontology): scaffold OGIT-canonical ontology spine (2026-05-07)

(Per APPEND-ONLY rule: PR sections are reverse-chronological; this dated entry is the new top-of-arc entry. Reverse-chronologically newest, even though it sits at the file end under tee-a governance.)

**Confidence (2026-05-07):** High. 28 tests passing (16 inline + 12 integration). Builds without `protoc` because Lance persistence is feature-gated.

**Branch:** `claude/create-graph-ontology-crate-gkuJG`
**Commit:** `4cf9a26` (prior recon + SPO-1 decision: `edef321`)

**Added:**
- New workspace member `crates/lance-graph-ontology/` (~3000 LOC). Cargo.toml with feature-gated `lance-cache` so the crate compiles without `protoc` (lance-encoding's build-script otherwise requires it).
- `src/lib.rs` public surface; modules `error`, `namespace`, `proposal`, `semantic_types`, `ttl_parse`, `foundry_map`, `registry`, `bridge`, `schema_source`.
- Public types: `OntologyRegistry`, `NamespaceBridge` (trait), `NamespaceId`, `OgitUri`, `SchemaPtr`, `SchemaKind`, `MappingProposal`, `MappingProposalKind`, `MappingRow`, `MappingHandle`, `HydrationReport`, `HydrationFailure`, `BridgeError`, `Error`, `SchemaSource` (trait), `EntityRef`, `EdgeRef`, `OntologyAssembler`, `SemanticTypeMap`, `TtlSource`.
- Default tenant bridges `bridges::WoaBridge`, `bridges::MedcareBridge`, `bridges::OgitBridge` (thin scoped views over the shared registry, ~20 LOC each per the v4 plan).
- `src/semantic_types.toml`: declarative OGIT-attribute → SemanticType map (the only TOML in the crate; ontology data itself is TTL).
- `src/lance_cache.rs` (feature-gated `lance-cache`): `LanceWriter` for runtime dictionary persistence.
- Phase 3 (scaffold), Phase 4 (TTL hydration), Phase 5 (tenant bridges) of the v4 plan.

**Locked:**
- **OGIT TTL is the canonical ontology source.** Lance is the runtime dictionary cache, not the source of truth.
- **Tenant bridges are thin scoped views** over the shared `OntologyRegistry`, not independent ontology multiplication.
- **Lance persistence is feature-gated** under `lance-cache`; the default compile path requires no `protoc`.
- **Federated two-layer cache (Option B) for SPO + ARiGraph**, per `.claude/DECISION_SPO_ARIGRAPH.md` (entropy-ledger rows 70 + 245: SPO + ARiGraph triplet_graph are not duplicates by design — they are an L1/L2 cache pair). The ontology crate is agnostic; it produces `Ontology` values; consumers route via `SchemaExpander`. Does NOT close SPO-1 — `promote_to_spo` bridge work remains separately owned.
- **`SchemaExpander` consumer point** (already shipped in earlier work) is the one bridge surface the ontology crate writes through; the prior `sql-spo-ontology-bridge-v1` plan's `SchemaExpander` proposal is therefore partially superseded (the expander shipped, the bridge plan's surface is now produced).

**Deferred:**
- Lance feature-gated compile path requires `protoc` to actually exercise the `lance-cache` feature; default compile path stays clean. Activating `lance-cache` in CI is deferred pending a `protoc` install step or a vendored protobuf descriptor.
- SPO-1 closure (`promote_to_spo` writer bridge between `arigraph::triplet_graph` and `spo::store`) — owned separately, not by this crate.
- Phases 6-7 of the v4 plan (canonical TTL emission for WoA / Healthcare into `AdaWorldAPI/OGIT/NTO/`; Cypher integration test routing around PARSER-1 stub via `lance_graph::parser::parse_cypher_query`).
- Tenant rosters beyond WoA / MedCare / OGIT.

**Docs:**
- `.claude/RECON_ONTOLOGY_CRATE.md` (Phase 1 recon, commit `edef321`).
- `.claude/DECISION_SPO_ARIGRAPH.md` (SPO-1 decision, commit `edef321`).
- This board update (LATEST_STATE.md table + Inventory; INTEGRATION_PLANS.md status annotation on `sql-spo-ontology-bridge-v1`; EPIPHANIES.md SPO-1 disposition entry; AGENT_LOG.md run entry).
