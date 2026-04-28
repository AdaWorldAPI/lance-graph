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
