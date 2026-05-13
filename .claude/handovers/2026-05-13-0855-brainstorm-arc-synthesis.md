# Synthesis — Brainstorming Arc, Compressed Insights, Verdicts, Outlook, Next Steps

**Date:** 2026-05-13 08:55 UTC
**Span:** 2026-05-01 → 2026-05-13 (the long session that started with Gaussian-Splat + EWA-Sandwich and converged on Super-Domain RBAC + UnifiedBridge)
**Companion files:**
- Formal status handover: `.claude/handovers/2026-05-13-0852-d-sdr-tier-a-complete-tier-b-and-beyond-pending.md`
- Active master spec: `.claude/plans/super-domain-rbac-tenancy-v1.md` (§1-§19)
- Scrubbed main-window transcripts: `.claude/transcript/` (79 jsonl + zip)
- Governance: `.claude/board/{LATEST_STATE,STATUS_BOARD,PR_ARC_INVENTORY,INTEGRATION_PLANS,EPIPHANIES,IDEAS}.md`

This document captures the **collaborative-architecting trajectory** the session traveled — what we explored, what we settled on, what we deliberately deferred, what we discovered was wrong, and what the concrete next steps are. The formal handover next to this one is the "what shipped"; this one is the "why we ended up here and what to do next."

---

## 1. The arc in one paragraph

The session started inside a math-certification pass for the **cognitive substrate**: Pillars 5+ (Köstenberger-Stark PSD-cone concentration), 5++ (Düker-Zoubouloglou Hilbert-space CLT at d=16384), and **6 (EWA-Sandwich Σ-push-forward `M·Σ·Mᵀ`)** — closing the concentration family for substrate aggregation across scalar / Σ-tensor / Hilbert-space / multi-hop edge propagation. That certification unlocked treating **splat-shaped reasoning** (Gaussian splatting, EWA sandwich) as a viable substitute for Neo4j-style edge traversal, which gave the **splat-osint-ingestion-v1** plan its math backbone. From there the session widened: (a) DTO-ladder audit produced the **palantir-parity-cascade-v2** plan with Foundry/Gotham parity as SoA-as-canon, (b) the OGIT spine got framed as Zone 1/2/3 over Lance + Supabase realtime (**ogit-cascade-supabase-callcenter-v1**), and (c) those two threads converged onto **Super-Domain RBAC + multi-tenancy v1** as the single load-bearing spec — 4-level OGIT hierarchy, 6-byte-per-row identity, RBAC + multi-tenant Chinese walls, Foundry-parity enforcement. §13-§18 of that spec then absorbed empirical reality (LanceProbe IS the drift bridge, the "3DES" is broken-single-DES, MedCareV2 is overlay-not-rewrite) and §19 nailed down build invariants (rust 1.94.1 / lance =4.0.0 / lancedb 0.27.2 / `ndarray::simd` canonical SIMD path). Tier A (D-SDR-1..5) shipped/committed; consumer wirings staged locally; Tier B+ on hold for MCP scope expansion.

---

## 2. Brainstorming threads (chronological)

### 2.1 Math substrate certification (May 1-5)

**Question:** is non-iid aggregation in d=16384 actually sound, or are we hand-waving the CLT?

**Verdict (FINDING):** Three proof-in-code pillars merged in succession:

| Pillar | What it certifies | Tightness |
|---|---|---|
| 5 (Jirak 2016) | Scalar weakly-dependent aggregation | rate `n^(p/2-1)` for `p ∈ (2,3]` |
| **5+ (Köstenberger-Stark, PR #286)** | PSD-cone Hadamard-space concentration for non-iid Σ aggregation | 0.969× — bound is **hit**, not just respected |
| **5++ (Düker-Zoubouloglou, PR #287)** | Hilbert-space CLT for AR(1) Gaussian process at d=16384 | 0.103% relative error — two orders below tolerance |
| **6 (EWA-Sandwich, PR #289)** | Σ push-forward `M·Σ·Mᵀ` along multi-hop paths | PSD-preserved 10000/10000 hops, CV tightness 1.467× |

Plus PR #288 (Σ-codebook viability probe, R²=0.9949) **ruled out** the proposed `CausalEdge64` 8→16 byte expansion — the 256-entry codebook with 1-byte sidecar is sufficient.

**Outlook:** every aggregation pattern in the cognitive substrate now sits on certified ground. The "cant-stop-thinking" loop has its mathematical backbone. This unblocks Gaussian-splat-shaped reasoning being treated as a first-class alternative to Neo4j-style edge traversal.

### 2.2 Splat as Neo4j-substitute (May 5-6)

**Question:** can EWA-sandwich Σ-push-forward replace neo4j-edge-traversal for OSINT reasoning, while preserving deterministic recall + uncertainty quantification?

**Verdict (Working):** Yes. Plan `2026-05-06-splat-osint-ingestion-v1.md` was authored; PR 1+2 of 6 in flight on `claude/splat-osint-ingestion`. SPLAT-1 ledger row dropped from entropy 4 → 2 (Aspirational → Wired stage 1).

**Concrete deliverables (D-SPLAT-1..7):**

| # | Title | Status |
|---|---|---|
| D-SPLAT-1 | `SplatChannel`, `CamPlaneSplat`, `SplatPlaneSet`, `AwarenessPlane16K`, `CamSplatCertificate`, `SplatDecision`, `TriadicProjection`, `ReasoningWitness64` + 10 unit tests in `lance-graph-contract::splat` | In PR (branch `claude/splat-osint-ingestion`) |
| D-SPLAT-2 | `crates/jc/examples/osint_edge_traversal.rs` — EWA-Sandwich Σ-push-forward demo for 5-hop chain, side-by-side vs naive convolution | In PR |
| D-SPLAT-3 | `witness_to_splat()` deterministic conversion | In PR (branch `claude/phase-3b-witness-to-splat`) |
| D-SPLAT-4 | Splat deposition into BindSpace columns via `MergeMode::AlphaFrontToBack` lanes | Queued |
| D-SPLAT-5 | `PlanarSplatBundle4096` with local/short/medium/long bands | Queued |
| D-SPLAT-6 | Semantic-CAM-distance integration — survivor tile selection vs splatted pressure planes | Queued |
| D-SPLAT-7 | Replay fallback — exact 4096-cycle `ThoughtCycleSoA` replay slice when certificate insufficient | Queued |

**Outlook:** D-SPLAT-1 + D-SPLAT-2 land first; the remaining 5 ride on the same branch sequence. The substrate pillars (5+/5++/6) certify the math, so the implementation risk is mechanical not theoretical.

### 2.3 DTO ladder + Foundry parity (May 7)

**Question:** what classifies as "bare-metal DTO" vs "SoA-glue" vs "bridge-projection"? Does our existing surface match Foundry's Object/Link/Action shape, or do we need a parallel table set?

**Verdict (FINDING, PR #352 baseline):** **Column H (`EntityTypeId: u16`, PR #272 already shipped)** IS the Foundry Object Type bridge. Foundry parity = SoA-as-canon parity. We do NOT duplicate the Foundry table set; we make the SoA carry the Foundry-equivalent shape.

**22-DTO audit:**
- 9 bare-metal (Tier 0)
- 7 SoA-glue (Tier 1)
- 6 bridge-projection (Tier 2)
- `StreamDto`, `ResonanceDto`, `BusDto` all live in `thinking-engine::dto.rs` — UPSTREAM of contract.

**Business Logic ↔ Thinking-style ↔ OGIT triangle:** each business op has 3 faces — `thinking_style: ThinkingStyle` dispatch, `ogit_verb: TTL`, `ogit_entities[]: TTL`. Plan `palantir-parity-cascade-v2.md`, D-PARITY-V2-2 ships the routing table.

**Outlook:** v2 D-PARITY-V2-1 (DTO ledger) and D-PARITY-V2-2 (triangle ledger) ship with the plan; D-PARITY-V2-3 wires `BusDto` into `engine_bridge.rs`. Foundry parity status snapshot — **SHIPPED:** Column H, audit trail, RBAC/RLS, PostgREST. **IN PROGRESS:** Q2 cockpit. **QUEUED:** LF-12 Pipeline DAG, LF-20 FunctionSpec, LF-22/23 ObjectView/Notification, LF-50 ModelRegistry.

### 2.4 OGIT spine as Zone 1/2/3 (May 7)

**Verdict:** OGIT SPO-G basins + Supabase realtime + Zone 1/2/3 boundary placement is the cleanest framing of the externalization surface. Plan `ogit-cascade-supabase-callcenter-v1.md`. This eventually fed §16 of the super-domain spec ("Zone 3 placement collapses determinism rules from 12 → 3").

### 2.5 Super-Domain RBAC convergence (May 13)

**The integrating move.** The DTO ladder + OGIT spine + Zone 1/2/3 + drift-detection threads all needed a single load-bearing structure to compose against. The answer was the **4-level OGIT hierarchy** of §1-§2:

```
MetaAnchors (Foundry ObjectType, OWL upper class, DOLCE marker, Wikidata QID)
  │
  ├─ SuperDomain (1 byte, 8 starters, 256 cap, activation root)
  │
  ├─ OgitBasin (1 byte family — Level-2 pointer)
  │
  └─ WithinBasinSlot (1 byte slot — Level-3 per-row identity)
```

Plus `TenantId: u32` for multi-tenant Chinese walls. **6 bytes per row total.**

**Verdict:** the entire integration plan becomes a function of three composable primitives:
1. **`UnifiedBridge<B: NamespaceBridge>`** — composes ontology bridge + RBAC policy + tenant tag (D-SDR-1, SHIPPED in #363).
2. **`SuperDomain` reverse-lookup table** — `FAMILY_TO_SUPER_DOMAIN: [SuperDomain; 256]` (D-SDR-2, SHIPPED in #363).
3. **Inline per-family codebook** `OgitFamilyTable + FamilyEntry` — label + schema + verbs INLINE, NOT sidecar — one cache line per slot (D-SDR-3, committed `2c3e87d`, unmerged).

Plus the merkle-chained audit log (D-SDR-4, `1d0157f`) and the wired 4-stage authorize (D-SDR-5, `dc9e081`).

### 2.6 §13 — Composes onto shipped PolicyRewriter (May 13)

**The ~30% LOC reduction lever.** D-SDR-5 doesn't introduce a parallel enforcement path. It composes onto the already-shipped `lance-graph-callcenter::policy::PolicyRewriter` chain (RowFilter + ColumnMask + RowEncryption + DifferentialPrivacy + Audit). **Consequence:** the spec's whole Tier A surface drops ~30% in LOC by reusing the existing chain.

**Verdict:** preserved as iron rule of the spec — D-SDR-13..17 (merkle salt, audit chain, DP role, encrypted view, hard-lock matrix) all compose as additional `OptimizerRule` slots, never as standalone authorization paths.

### 2.7 §15→§16→§17→§18 — empirical reality (May 13)

The most consequential reframing of the session: a chain of corrections that compressed scope by ~60%.

| Original framing (§15) | Reality (§16-§18) |
|---|---|
| 3 implementations under one contract, drift bridge as sustained production infrastructure | **Drift bridge bounded to Phase 2-3 cutover window only**, retires to CI gate after Phase 4 |
| 12 cross-language byte-determinism rules | **6 concrete canonicalization rules** (date / decimal / bool / soft-delete / pwd / timestamp); further collapsed by Zone 3 wire-format placement |
| Custom Protobuf IDL for cross-language | **Arrow Flight SQL** + Substrait extension types (D-SDR-20 superseded by §17.3) |
| 3-cipher 3DES chain → AES-GCM rewrap | **Single broken 3DES**: 128-bit truncated key + ECB-equivalent + zero IV + hardcoded password table. Affects ONLY `u_pwd` column. **Carry ciphertext forward; Argon2-backfill on login.** No AES-GCM rewrap. |
| MedCareV2 reshapeable freely | **MedCareV2 is overlay-only** (LanceProbe on top of copy of MedCare); DO NOT refactor |
| MySQLAdapterBridge as third drift partner | **DROPPED** — D-SDR-24 superseded by D-SDR-27 one-shot import (~80 LOC) |
| ~5-10 D-SDR deliverables in Tier F | **Concrete Rust-side gap = 5 endpoints** (D-SDR-35..39) + reduced D-SDR-27 = ~700 LOC + tests |
| Arrow Flight SQL is the immediate path | **HTTP+JSON over JWT is M2-M6** (matching what LanceProbe already targets); Flight SQL is Phase 5+ |
| LanceProbe is a concept to design | **LanceProbe already exists** (8 components, M1 scaffolded); the drift bridge **is** LanceProbe |

**Outlook:** the Tier F deliverable count dropped from ~12 nominal items to **5 concrete endpoints** (D-SDR-35..39) + 1 reduced import tool (D-SDR-27) + ~700 LOC. The scope-reduction is the most important payoff of the §18 empirical pass.

### 2.8 §19 — Build invariants nailed down (May 13)

**Pinned:** `rust 1.94.1` (stable, no `#![feature(...)]`), `lance =4.0.0`, `lancedb 0.27.2`. SIMD policy: `ndarray::simd` is canonical — `LazyLock<Tier>` dispatch already shipped; just import. Per §19.7 correction the Tier A LOC reduction story did NOT materialize (per-row hot path is scalar, not batch), but the canonical-path rule stands.

**Clippy gate:** `cargo clippy -- -D warnings` is the merge gate. Every `unsafe` block needs a `// SAFETY:` comment. OpenBLAS and MKL mutually exclusive.

### 2.9 Codex P2 fix (the canonical-name reality check, May 13)

**Verdict:** Policy must evaluate against the **canonical OGIT entity type** (`row.ogit_uri.name()`, e.g. `Order`), NOT the bridge-side alias (`public_name`, e.g. `WorkOrder`). Without this, consumer-facing aliases couple to policy authorship and one policy doesn't span multiple bridges resolving to the same canonical type.

Fix in commit `421e71e` (in #363). Two regression tests pin the contract:
- `unified_bridge_evaluates_policy_against_canonical_entity_type` (alias `WorkOrder` honored when policy keyed on canonical `Order`)
- `unified_bridge_does_not_honor_alias_keyed_policy` (deliberately rejects alias-keyed policies — they would never match across bridges)

---

## 3. Compressed insights (the load-bearing claims)

1. **6 bytes per row is the right size for OGIT addressing.** `TenantId: u32 + OwlIdentity: u16`. High byte of `OwlIdentity` = `OgitFamily` basin; low byte = within-basin slot. Bitmask predicates compose in DataFusion's single masked-predicate path.
2. **Inline codebook, not sidecar.** Label URI + schema + verbs + characteristics + DOLCE marker + provenance live INLINE in `FamilyEntry`, one cache line per occupied slot. Sub-microsecond O(1) lookup. No join.
3. **Reverse lookup `[SuperDomain; 256]`** = 256 bytes total. Static. Currently all-`Unknown` until TTL hydration (D-SDR-3b future work).
4. **Merkle audit chain is local per super-domain.** Per-super-domain `merkle_salt` makes cross-domain audit logs unlinkable (Healthcare ↔ OSINT hard-lock barrier). FNV-1a 64-bit chain. Already implemented in `AuditChain` (D-SDR-4).
5. **PolicyRewriter is the enforcement spine.** Every `authorize_*` decision composes onto the shipped chain — no parallel paths.
6. **DataFusion `OptimizerRule` is the right abstraction slot** for RLS / column mask / row encryption / DP / audit. The shipped `PolicyRewriter` chain occupies it; the spec's later D-SDR's slot into the same surface.
7. **Arrow Flight SQL > custom Protobuf IDL** for cross-language. Substrait extension types carry `OwlIdentity` + `MerkleRoot` + `SuperDomain`. HTTP+JSON over JWT is the immediate path; Flight SQL is Phase 5+.
8. **"3DES" is broken-single-DES.** Don't rewrap. Carry the ciphertext forward; Argon2-backfill on login. Affects (likely) only `u_pwd`.
9. **LanceProbe IS the drift bridge.** Don't design it; wire to it. 5 endpoints + Argon2 fallback.
10. **Cognitive aggregation has math certification.** Pillars 5/5+/5++/6 cover scalar / Σ-tensor / Hilbert-space / multi-hop. Splat-shaped reasoning is on certified ground.
11. **Codex's review catches matter.** P2 fix (canonical-name vs alias) prevented a leak that would have shipped without it; the regression tests are the contract.
12. **Stable Rust only.** No `#![feature(...)]`. `ndarray::simd` canonical SIMD path. Workspace pins are non-negotiable.

---

## 4. Verdicts on disputed framings

| Question | Resolution |
|---|---|
| Should `OgitFamilyTable` carry a sidecar map for label/schema/verbs? | **NO.** Inline, one cache line per slot. SGO meta (>256 entries) is excluded from runtime addressing. |
| Should we design custom cross-language IDL? | **NO.** Arrow Flight SQL + Substrait extension types. |
| Should drift detection run as sustained production infrastructure? | **NO.** Phase 2-3 cutover window only; retires to CI gate. |
| Should we design a 3DES → AES-GCM rewrap pipeline? | **NO.** Argon2-backfill on login. Single-cipher legacy. |
| Should MedCareV2 be refactored to fit the new shape? | **NO.** Overlay only; do not touch. |
| Should `UnifiedBridge::authorize_*` short-circuit before audit on `BridgeError`? | **YES, for now** (D-SDR-5 minimum). Bad input names aren't auth decisions; revisit if probing detection becomes a need. |
| Should the actor role be stored as `&'static str` in audit records? | **NO.** Cache an FNV-1a digest at construction; store the `u64` hash in records. The `&'static str` doesn't fit a `Copy` audit record. |
| Should Policy evaluate against the canonical OGIT name or the bridge alias? | **Canonical OGIT name** (`row.ogit_uri.name()`). Codex P2 fix. |
| Should every audit field be Copy? | **YES.** `UnifiedAuditEvent` is `Copy`. `RecordingSink` test uses `*event`, not `event.clone()`. |
| Should we ship a new SIMD scalar fallback per DTO method? | **NO.** Per-row hot path stays scalar; batch path delegates to `ndarray::simd`. |
| Should we open D-SDR-3..5 as separate PRs? | **NO.** Single follow-up PR; board hygiene updates in the same commit. |
| Should the K-anonymity federation policy ship now? | **NO.** Deferred (D-SDR-12) until a customer demands it. |

---

## 5. Outlook (what shape the system is converging to)

```
Cypher / SPARQL / Gremlin / SQL
        │
        ▼ (single DataFusion LogicalPlan)
PolicyRewriter chain:
  ├─ RlsRewriter             (RowFilter)
  ├─ ColumnMaskRewriter      (per-role redaction)
  ├─ RowEncryptionRewriter   (per-tenant DEK / LanceDB transparent encrypted view)
  ├─ DifferentialPrivacyRewriter (researcher role, ε-bounded noise, k-anonymity floor)
  └─ AuditRewriter           (one chained merkle event per row decision)
        │
        ▼
LanceDB scan
  ├─ tenant_id: u32          ─┐
  ├─ owl_id: u16             ─┤  6-byte identity
  ├─ merkle_root: u64        ─┘  (cleartext-beside-ciphertext)
  └─ payload: encrypted

Cross-language access:
  Rust direct          → DataFusion in-process
  C# via HTTP+JSON/JWT → M2-M6 LanceProbe milestones (Phase 1-4)
  C# via Flight SQL    → Phase 5+ (Substrait extension types for OwlIdentity + MerkleRoot + SuperDomain)

Drift bounded:
  Phase 2-3 only.
  D-SDR-35..39 endpoints in medcare-rs.
  Retires post-Argon2-backfill cutover.
```

Three orthogonal extensions live alongside this:

- **Cognitive substrate** (Pillars 5+/5++/6, BindSpace SoA, splat-osint pipeline) — separate research track; doesn't gate the RBAC path.
- **Foundry parity DTO ladder** (palantir-parity-cascade-v2) — SoA-as-canon; ships D-PARITY-V2-1..12 separately.
- **OGIT TTL namespaces** (`AdaWorldAPI/OGIT` fork) — D-SDR-6 + D-SDR-7 blocked on MCP scope expansion.

---

## 6. Implementation steps (priority-ordered)

### Phase 0 — Land what's already done (this week)

1. **Open follow-up PR** for D-SDR-3 + D-SDR-4 + D-SDR-5 on `AdaWorldAPI/lance-graph`. Five commits stacked clean on merged `main`:
   - `3e94a27` knowledge: E1-E6 splat + formal-grounding epiphanies inbox
   - `2c3e87d` D-SDR-3 per-family codebook table
   - `1d0157f` D-SDR-4 merkle-chained audit log
   - `dabd510` Cargo.lock after D-SDR-4
   - `dc9e081` D-SDR-5 authorize_* wiring with chained audit emission
2. **Update board governance in the same PR** (Mandatory Board-Hygiene Rule):
   - `LATEST_STATE.md` — add `unified_bridge`, `super_domain`, `family_table`, `unified_audit` to Contract Inventory.
   - `STATUS_BOARD.md` — D-SDR-1..2 Shipped (PR #363), D-SDR-3..5 In PR (this PR), D-SDR-6..39 Queued.
   - `PR_ARC_INVENTORY.md` PREPEND — entry for #363 (Added: D-SDR-1..2 + Codex fix) and the follow-up PR (Added: D-SDR-3..5).
   - `INTEGRATION_PLANS.md` PREPEND — status correction line on `super-domain-rbac-tenancy-v1`: "Tier A complete (Codex P2 fix + D-SDR-3..5 land in follow-up PR)."
   - `EPIPHANIES.md` PREPEND — `2026-05-13 — FINDING: Tier A (D-SDR-1..5) shipped against shipped PolicyRewriter chain (§13.1 thesis confirmed)`.
3. **Push + open consumer PRs** in parallel (already committed locally):
   - `AdaWorldAPI/medcare-rs` — `31e999b feat(medcare-analytics): wire UnifiedBridge<MedcareBridge> for F2 RBAC entry-point auth`
   - `AdaWorldAPI/smb-office-rs` — `342f601 feat(smb-bridge): wire UnifiedBridge<OgitBridge> for F2 RBAC entry-point auth`

### Phase 1 — Self-contained next deliverables (next ~2 weeks)

4. **D-SDR-13** — per-super-domain HKDF in `TenantContext::encryption_key`. Wires the `merkle_salt: u64` field already on `AuditChain`. Self-contained, ~80 LOC + 4 tests.
5. **D-SDR-17** — hard-lock partner matrix as static table + predicate-time check in `authorize_*`. ~60 LOC + 4 tests. Healthcare ↔ OSINT + 3 others (matrix to be enumerated as part of this deliverable).
6. **D-SDR-10** — `JsonLinesAuditSink` implementing `UnifiedAuditSink`. D-SDR-4 already exposed the trait. ~80 LOC + 1 integration test.
7. **D-SDR-14** — `AuditEntry` updated JSON schema + replay-time `verify_chain` integration. Builds on D-SDR-10. ~120 LOC + 6 tests.

### Phase 2 — Meta-bridge harvest (Tier F, ~2-4 weeks)

8. **D-SDR-18** — archaeology pass on `medcare_bridge.rs`/`sharepoint_bridge.rs`/`woa_bridge.rs`. Extract fix-commits as named tests. ~1 day.
9. **D-SDR-19** — `MetaBridge` trait + `BridgeFromRegistry` extension. ~150 LOC.
10. **D-SDR-21 + D-SDR-22** — `MedCare-rs` + `smb-office-rs` retrofit (zero behavior change).

### Phase 3 — LanceProbe wiring (Tier H, gated on MCP scope expansion)

11. **D-SDR-37** — `_dto_contracts.md` for 5 pilot endpoints + 40+ future routes. Doc-only, **blocks M2**. ~300 lines markdown.
12. **D-SDR-35** + **D-SDR-36** — `/api/__parity/csharp` ingest + `/api/__parity` dashboard. Unblocks M5. ~270 LOC + 7 tests.
13. **D-SDR-38** — `legacy-tripledes-fallback` feature flag + Argon2 backfill-on-login. **Blocks M5a.** ~180 LOC + 6 tests.
14. **D-SDR-27** — reduced-scope MySQL+3DES → LanceDB one-shot import. ~80 LOC + 2 tests.
15. **D-SDR-39** — `/api/__parity/telemetry` endpoint. Blocks M6. ~80 LOC + 2 tests.
16. **D-SDR-30** — 3DES key destroy step, post-Argon2-backfill (Phase M6).

### Phase 4 — Cross-language convergence (Tier from §17, Phase 5+)

17. **D-SDR-31** — Arrow Flight SQL endpoint exposing the DataFusion catalog (lancedb 0.27.2).
18. **D-SDR-32** — Substrait dialect mapping for DataFusion → C# client.
19. **D-SDR-33** — Substrait extension types for `OwlIdentity` + `MerkleRoot` + `SuperDomain`.
20. **D-SDR-34** — Cross-language schema fixtures + golden tests.
21. **D-SDR-23** — C# `MedCareV2` aligned to `MetaBridge` via Flight SQL client.

### Phase 5 — Phase-2 federation (deferred, customer-demand-gated)

22. **D-SDR-12** — `FederationPolicy::KAnonymityAggregate` (HLL cardinality gate).
23. **D-SDR-15** — `PolicyKind::DifferentialPrivacy` for `researcher` role (ε-bounded noise + k-anonymity floor).
24. **D-SDR-16** — `EncryptedViewAggregate` federation policy (LanceDB transparent encrypted view).

### Phase 6 — OGIT TTL namespaces (gated on MCP scope expansion)

25. **D-SDR-6** — `OGIT/NTO/Hiro/{entities,verbs}/*.ttl` PR on the OGIT fork.
26. **D-SDR-7** — `OGIT/NTO/HubSpot/{entities,verbs}/*.ttl` PR.
27. **D-SDR-8** — `/home/user/hiro-rs` new crate. ~150 LOC.
28. **D-SDR-9** — `/home/user/hubspot-rs` new crate. ~150 LOC.
29. **D-SDR-11** — Compliance regime certification doc (HIPAA §164.312 / SOX §404 / PCI-DSS Reqs 3+7+10). Doc-only.

### Orthogonal tracks (parallel to the above)

- **splat-osint-ingestion-v1** — D-SPLAT-3..7 ride the 6-PR sequence on `claude/phase-3b-witness-to-splat` + successors.
- **palantir-parity-cascade-v2** — D-PARITY-V2-3..12 land independently.
- **ogit-cascade-supabase-callcenter-v1** — Zone 1/2/3 realtime integration.

---

## 7. Things to NOT redo (anti-patterns recorded)

- **Don't design custom Protobuf IDL** (Arrow Flight SQL replaces; §17.6).
- **Don't design 3DES → AES-GCM rewrap** (Argon2 backfill on login; §18.5/§18.6).
- **Don't reshape MedCareV2** (overlay-only; §18.9).
- **Don't introduce a parallel enforcement path** alongside `PolicyRewriter` (§13.1).
- **Don't add `#![feature(...)]`** (stable-only; §19.1).
- **Don't sidecar codebooks** (inline per cache line; §3.3).
- **Don't pass `&'static str` actor names through audit records** (cache an FNV-1a digest; D-SDR-5).
- **Don't evaluate Policy against bridge-side aliases** (canonical OGIT name only; Codex P2).
- **Don't use `MySQLAdapterBridge`** as a sustained drift partner (D-SDR-24 dropped).
- **Don't expand `CausalEdge64` 8→16 bytes** (PR #288 R²=0.9949 ruled it out).
- **Don't write scalar SIMD fallbacks per DTO** (`ndarray::simd` is the canonical path).

---

## 8. Open questions still standing

1. Which columns besides `u_pwd` call `EncryptMessage()`/`DecryptMessage()` in `MySQL_Connect.cs`? Grep needed. Likely few/none.
2. DTO contracts for the 40+ routes in `FUTURE_STACK_ADMIN.md §4` — D-SDR-37 needs to enumerate.
3. `MedCare-rs/docs/AUTH_LEGACY_TRIPLEDES_MIGRATION.md` is DRAFT — what blocks promotion to Active? Likely D-SDR-38 + human security review.
4. Hard-lock partner matrix completeness — defer enumeration until the billing/ticket basins are concretized.
5. Per-super-domain DP epsilon defaults — Phase 2 federation territory; not blocking.
6. SGO meta exclusion from runtime — confirmed in §3.3; addressable domain ≤ 256 entries per family. Re-check when a basin approaches the cap.

---

## 9. Cross-reference index

- **Active spec:** `.claude/plans/super-domain-rbac-tenancy-v1.md`
- **Companion plans:**
  - `palantir-parity-cascade-v2.md` (DTO ladder + Foundry/Gotham parity)
  - `2026-05-06-splat-osint-ingestion-v1.md` (splat + EWA bridge)
  - `ogit-cascade-supabase-callcenter-v1.md` (Zone 1/2/3 + Supabase realtime)
  - `lance-graph-ontology-v5.md` (post-merge follow-ons)
  - `tetrahedral-epiphany-splat-integration-v1.md` (splat into BindSpace)
  - `thought-cycle-soa-awareness-integration-v1.md` (awareness SoA)
- **Knowledge:**
  - `.claude/knowledge/encoding-ecosystem.md` (MANDATORY before codec work)
  - `.claude/knowledge/lab-vs-canonical-surface.md` (MANDATORY before REST/Wire DTO)
  - `.claude/knowledge/vsa-switchboard-architecture.md` (three-layer VSA framing)
  - `.claude/knowledge/soa-dto-dependency-ledger.md` (DTO ladder ledger)
- **Governance:**
  - `.claude/board/{LATEST_STATE,STATUS_BOARD,PR_ARC_INVENTORY,INTEGRATION_PLANS,EPIPHANIES,IDEAS}.md`
- **Scrubbed transcripts:** `.claude/transcript/` (79 jsonl + 4.5 MB zip, 2026-05-01 → 2026-05-13).
- **Companion handover:** `.claude/handovers/2026-05-13-0852-d-sdr-tier-a-complete-tier-b-and-beyond-pending.md`.

---

**End of synthesis.**

The natural next move is Phase 0 step 1: open the follow-up PR for D-SDR-3..5 with board-hygiene updates folded in, then push the two consumer PRs. After that, Phase 1 (D-SDR-13/17/10/14) is fully self-contained and can ship in any order.
