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

## v1 — Super-Domain RBAC + Multi-Tenancy (authored 2026-05-13)

**Author:** main thread (Opus 4.7 1M), session 2026-05-13 (branch `claude/lance-datafusion-integration-gv0BF`)
**Status:** Active
**Scope:** 4-level addressing hierarchy (meta-anchors → super domain → OGIT basin → within-basin slot) with explicit byte-sized DTOs, RBAC + multi-tenant Chinese walls wired onto the super-domain boundary. 6 bytes per row (4-byte `TenantId` + 2-byte `OwlIdentity`), inline per-family codebook with label+schema+verbs, single masked DataFusion predicate enforces tenant + super-domain + role + slot in one vector pass. Foundry-parity selling point at the enforcement surface, sub-microsecond hot path. Locks the 2-consumer ticket-system constraint (`hiro-rs` absorbs OSLC-* off-label, `hubspot-rs` is fresh basin) and collapses 4 OSLC-* namespaces into a single Hiro basin with provenance lineage.
**Path:** `.claude/plans/super-domain-rbac-tenancy-v1.md`
**Deliverables:** D-SDR-1..D-SDR-12 (Tier A DTOs / Tier B TTL namespaces / Tier C consumer crates / Tier D compliance + audit / Tier E cross-tenant federation Phase 2)
**Substrate:** Builds on shipped `lance-graph-ontology::namespace::SchemaPtr`, `bridges::OgitBridge` + `BridgeFromRegistry`, `holograph::dntree::WellKnown` (promoted to `SuperDomain` enum), `lance-graph-callcenter::dn_path::DnPath` compression chain, `bgz-tensor::HhtlDEntry` bit-packed-hierarchy pattern, `lance-graph-contract::cam` CAM-PQ codec contract.
**Cross-ref:** `palantir-parity-cascade-v2.md` (this spec adds the enforcement surface), `lance-graph-ontology-v5.md` (this spec sits above v5; v5 unchanged), `GLUE_LAYER_OGIT_TO_OWL_SPEC.md` (source for OWL property characteristics bitfield).
**Open questions:** Foundry ObjectType cross-walk targets, Wikidata QID mappings, audit format choice (JSON Lines / CloudEvents / OTel), DEK rotation cadence, escalation UX, HPO/MONDO multi-member confirmation, slot 0xFF schema-only convention.

**Correction (2026-05-13):** §13 refinements added (same session). (a) Enforcement composes onto shipped `lance-graph-callcenter::policy::PolicyRewriter` chain + `PolicyKind` taxonomy (RowFilter/ColumnMask/RowEncryption/DifferentialPrivacy/Audit) rather than introducing parallel path — ~30% Tier A LOC reduction. (b) Cross-tenant federation upgraded to A+B+C all accepted; Option C (`EncryptedViewAggregate`) viable now via LanceDB transparent encrypted views, not 2027+ R&D. (c) Audit chain integrity built-in via `MerkleRoot::from_fingerprint` + `ClamPath` from `graph/spo/merkle.rs` (the merkle/DN-path mixing already shipped). (d) Hard-lock requirement formalized: Healthcare ↔ OSINT (and 3 other pairs) get 3 layers of defense — predicate + per-super-domain merkle salt + super-domain-scoped HKDF key derivation. (e) `researcher` role hardened to anonymized-projection-only with k-anonymity floor + DP noise injection on aggregates. New deliverables D-SDR-13..17 added. Open questions on audit format + cross-tenant federation RESOLVED; new open questions on hard-lock partner matrix + per-super-domain DP epsilon + merkle salt rotation cadence.

**Correction (2026-05-13, third commit):** §18 empirical reality check added after pygithub REST inspection of `AdaWorldAPI/MedCareV2` + `AdaWorldAPI/MedCare-rs@claude/csharp-handoff-docs-L3DF0`. Major findings: (a) The §15-§17 drift bridge concept is already designed and partially scaffolded as `MedCareV2/MedCare_2.0/LanceProbe/` (M1 complete; M2-M6 pending Rust-side endpoints). 8 LanceProbe components (ParityClient/ParityWitness/DriftSink/etc.) map nearly 1:1 to the spec's DTOs. (b) MedCareV2 is overlay-only (copy of MedCare + LanceProbe additions) — cannot be reshaped freely as I assumed; "do NOT refactor" is the explicit constraint. (c) CRITICAL crypto correction: the "3DES" in MedCare's `Crypt.cs:438-451` uses 128-bit truncated key + zero IV + ECB-equivalent + non-standard MD5+RC2 KDF + 62-entry hardcoded password array — cryptographically equivalent to single DES (broken). The migration is NOT 3DES→AES-GCM rewrap; it's Argon2-backfill-on-login per existing `MedCare-rs/docs/AUTH_LEGACY_TRIPLEDES_MIGRATION.md` plan. (d) Only the `u_pwd` column on `praxis_mitarbeiter` uses the 3DES path; rest of the schema is plaintext. D-SDR-27 scope reduces from "decrypt-rewrap pipeline" to "carry ciphertext forward, Argon2-backfill on first login." (e) §15.2 abstract 12-rule determinism table replaced by 6 concrete canonicalization rules from `CSHARP_HANDOFF_PROMPT.md` lines 93-104 (date / decimal / bool / soft-delete / pwd / timestamp). (f) §17.3 Arrow Flight SQL convergence is aspirational end-state; immediate path is HTTP+JSON over JWT (what LanceProbe already targets); Flight SQL is Phase 5+ migration. (g) New deliverables D-SDR-35..39 for medcare-rs side: parity ingest endpoint, dashboard, DTO contracts doc, TripleDES fallback feature flag, telemetry endpoint. M5 is blocked until these land. Resolved 7 prior open questions (audit format, federation, DEK rotation, hard-lock matrix scope, DP epsilon, MedCareV2 reshape, 3DES inventory). 3 new open questions: other columns calling EncryptMessage in MySQL_Connect.cs, DTO contracts for 40+ planned routes, AUTH_LEGACY_TRIPLEDES_MIGRATION.md DRAFT-to-Active blockers.

**Correction (2026-05-13, second commit):** §14-§17 refinements added (same session). (§14) Meta-bridge extracted from shipped medcare_bridge.rs + sharepoint_bridge.rs harvest, not designed clean-room. New bridges hubspot_bridge.rs + hiro_bridge.rs added as templates; woa_bridge.rs retrofit. Tier F (D-SDR-18..20, 23) + Tier G (D-SDR-21..22) deliverables. (§15) Drift detection initially framed as production parallelbetrieb infrastructure with 12 cross-language determinism rules — substantially refined by §16+§17. (§16) Pre-prod posture corrected per user clarification: nothing in production yet, single 3DES cipher (not 3-cipher chain), one-shot import tool not persistent infrastructure. Zone 3 boundary placement collapses determinism rules from 12 to ~3 (decimal + timestamp + FP aggregate). MerkleRoot-cleartext-beside-ciphertext insight: drift bridge compares without ever decrypting in steady-state production, so encryption uses random nonces (no need for AES-GCM-SIV). MedCare MySQL Struktur reality check (104 tables, all VARCHAR/TEXT/DATETIME, app-layer 3DES not at-rest, schema is purely clinical with billing/tickets in separate WoA/Hiro databases). New deliverables D-SDR-27..30. (§17) Convergence on LanceDB+DataFusion SQL as unified persistence; both Rust (in-process) and C# (Arrow Flight SQL gRPC) clients hit the same DataFusion logical plan layer. Custom Protobuf IDL (D-SDR-20) SUPERSEDED by Arrow Flight SQL — Substrait extension types for OwlIdentity/MerkleRoot/SuperDomain. Drift bridge bounded to Phase 2-3 cutover window, then retires to CI gate. New deliverables D-SDR-31..34. Dropped scope: MySQLAdapterBridge (D-SDR-24), persistent production drift infra, multi-trustee key escrow, C-ABI FFI option, custom Protobuf IDL. §18 deferred pending MCP scope expansion to AdaWorldAPI/MedCare + AdaWorldAPI/MedCareV2 for 3DES column inventory + transcoded shape grep.

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
