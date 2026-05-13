# Ideas Log — Open + Implemented + Integration (triple-entry, append-only)

> **Append-only ledger** for every architectural idea, speculative
> design, "what if we tried X" moment. Ideas accumulate here
> whether or not they're ready to ship. When one gets implemented,
> it moves from Open → Implemented → Integration (a row linking
> the idea to the plan entry that scheduled it + the PR that
> shipped it).
>
> **Purpose:** a speculation has nowhere else to live until it's
> scoped into a plan. This file is the speculation surface. Ideas
> die or graduate here; nothing is lost.

---

## Triple-entry discipline

Every idea moves through three ledger sections in this file:

1. **Open Ideas** — speculative; captured when proposed.
2. **Implemented Ideas** — idea became real; row appended with PR
   anchor + integration-plan D-id reference.
3. **Integration Plan Update Log** — the paired "what the plan
   changed when this idea landed" row, citing the specific
   `INTEGRATION_PLANS.md` version bump or `STATUS_BOARD.md` row
   flip triggered by the idea.

The row in Open is NEVER moved; its Status flips. The Implemented
row is a NEW append that cites the Open anchor. The Integration
row is a THIRD append that cites both.

This is **triple-entry bookkeeping** — three sections, same idea,
cross-linked. The cost is a bit more writing; the benefit is that
every shipped idea has an audit trail from speculation → code →
plan consequence.

---

## Rejected / Deferred

Ideas that don't graduate go into a fourth section:

4. **Rejected / Deferred Ideas** — with `**Rationale:**` and cross-
   ref to the original Open entry. The Open row's Status flips to
   `Rejected YYYY-MM-DD` or `Deferred to <when>`.

Deferred ideas can later reactivate — append a new Open entry
citing the Deferred one; Deferred row's Status flips to `Reactivated
YYYY-MM-DD <new-entry>`.

---

## Governance

- **Append-only.** Never delete a row.
- **Mutable fields:** `**Status:**` line, `**Rationale:**` line (if
  added later with more context).
- **`permissions.ask` on Edit** (same rule as other bookkeeping
  files). Write for appends stays unprompted.

## Cross-references

- `EPIPHANIES.md` — if an idea came from an epiphany, both entries
  cross-reference each other.
- `INTEGRATION_PLANS.md` — the plan version that incorporated the
  idea.
- `STATUS_BOARD.md` — the D-id status row that reflects the idea's
  shipping status.
- `PR_ARC_INVENTORY.md` — the PR that landed the code.
- `ISSUES.md` — if implementing an idea surfaced a bug, both rows
  link.

---

## Kanban Format (priority + scope on every entry)

Every idea carries:
- **Priority** — `P0` must-ship-this-phase / `P1` next-phase / `P2`
  eventual / `P3` speculative.
- **Scope** — which agent / deliverable / domain: `@<agent-name>`,
  `D<N>` (plan D-id), `domain:<grammar|codec|arigraph|infra|...>`.

Ticket tag on each entry: `[P2 @family-codec-smith D7 domain:grammar]`.
Agents filter by `@`-mention or domain to see what's theirs.

## Open Ideas

(Prepend new ideas here with today's date. Format:)

## 2026-05-13 — CORRECTION-OF previous same-day splat row: split into two distinct ideas (arch + render)

Earlier this session conflated EWA-Sandwich with a Gaussian-splat anatomical renderer. Per user 2026-05-13 follow-up + source confirmation: EWA-Sandwich is **Pillar 6** of the JC pillars framework — Σ push-forward `M·Σ·Mᵀ` for multi-hop edge propagation in the SPD cone. Already implemented at `crates/jc/src/ewa_sandwich.rs` (450 LOC) + `crates/lance-graph-contract/src/sigma_propagation.rs` (488 LOC) + `crates/jc/examples/osint_edge_traversal.rs` + `crates/jc/examples/splat_perturbationslernen.rs`. Not a new idea — an existing certified pillar. See EPIPHANIES 2026-05-13 CORRECTION-OF entry.

## 2026-05-13 — RECONCILIATION: Amiga-demoscene prerender + live 60fps renderer compose as cutscene-plus-gameplay, NOT competing alternatives

Today's prior thrashing wrongly framed prerender vs live as either/or. **Both are right, for different phases.** Classic AAA game pattern: prerendered cinematics for WOW + live engine for interaction. The FMA holographic demo gets the same split:

| Phase | Substrate | Frames | Role |
|---|---|---|---|
| **Intro cinematic** | Prerendered frame stream (demoscene-grade) | 900-18000 (30-300s @ 30-60fps) | Camera fly-through full-body → cardiovascular → heart; layer reveals; system showcases. Hand-crafted, every frame perfect. Zero runtime compute. THIS IS THE WOW. |
| **Hand-off** | Last prerender frame == first live frame | 1 | Alignment contract: prerender exit camera + entity positions match live `RenderFrame` initial state. |
| **Live interaction** | `ndarray::hpc::renderer` 60fps double-buffer + Pillar 6 propagation | indefinite | User rotates, hovers, clicks. EWA-splat live render. Heart-click triggers Pillar 6 wave. |
| **Transition cinematics** | Short prerendered clips | 60-300 (1-5s) | "Switch to nervous system" → peel-transition clip → hand back to live in new layer state. Library of ~20 precomputed clips. |
| **Outro / share** | Capture live session to splat stream | variable | User clicks "share this view" → record N seconds of live render → publish. |

**Build implications:**
- Prerender pipeline: offline tool that runs the live renderer through scripted camera trajectories at higher per-frame compute (path tracing, accumulation, anti-aliasing) and dumps frames to a streamable format. ~500 LOC tool. Probably new crate `crates/fma-cinematic` or extension to a render binary.
- Handoff contract: `RenderFrame::from_prerender_final_state(stream)` constructor that seeds positions/velocities/camera from the prerender's last frame metadata. ~100 LOC.
- Transition clip library: 10-20 hand-curated clips for system switches, organ-zoom, ghost-mode toggle. Stored as `.splat-stream` files in a release artifact (similar to the bgz7 Qwen3.5 release pattern).
- Storage format candidates: (a) raw Arrow batches of `RenderFrame` snapshots, (b) MP4 with custom video codec, (c) `.splat`/`.ply` 3DGS native, (d) custom temporal-delta codec exploiting the front/back buffer's small per-frame deltas.

**Lesson learned:** when conjecturing an alternative substrate, ask "is this a different phase of the same UX, or actually a competing implementation?" The prerender vs live thrash today was a false dichotomy that wasted three correction cycles in EPIPHANIES. Cross-ref EPIPHANIES 2026-05-13 unification entry (one kernel, three Jacobians) — same lesson at the math layer.

## 2026-05-13 — Sci-fi presentation vision: transparent holographic human-body projection for q2 (Tony Stark / Star Trek sickbay aesthetic) — emerges naturally from the unified Σ-push-forward kernel

The sci-fi UX target for the FMA heart-click demo: a transparent, glowing, layered holographic projection of the human body the user can rotate, peel, and probe. **The technical substrate already produces this look** without a separate volumetric renderer — the unified Gaussian-splat + EWA-Sandwich kernel (see EPIPHANIES 2026-05-13 unification entry) gives every component for free:

| Sci-fi property | Substrate that delivers it |
|---|---|
| Soft glow, semi-transparency | Per-node 3D Gaussian splat → EWA projection has built-in alpha falloff (no separate transparency shader) |
| Volumetric / cloud-like body | Additive blending of 75K splats in frag shader (`ndarray::hpc::renderer` front-buffer → q2 WebGPU) |
| Pulsing scan wave on heart-click | Pillar 6 Σ push-forward along anatomy edges; readout = per-node Σ-displacement; render as bloom intensity |
| Peelable layers (skeletal / cardiovascular / nervous) | `SuperDomain::Healthcare` family slice filter on the SPO subset feeding the render frame |
| Real-time rotation, 60fps | Already canonical (`cached_splat(DT_60)`, double-buffer atomic swap) |
| Click-to-probe with audible feedback | UnifiedBridge<MedcareBridge> auth + audit chain (W11 spec) + thinking-engine intent classification (W6 spec) → q2 surfaces SPO neighbors + drug-knowledge crosswalk |
| Cyan/teal Stark palette | q2 frontend concern (CSS / shader uniform); no backend impact |

**What's actually new versus what's wiring:**
- **NEW (small):** q2 frontend shader — additive Gaussian-splat fragment shader with bloom + cyan palette; ~200 LOC of WGSL/GLSL
- **NEW (small):** FMA → RenderFrame seeder that picks initial 3D positions from anatomical "canonical pose" (one-shot offline job; head up, arms out, T-pose; ~300 LOC)
- **NEW (tiny):** edge highlight protocol — when Pillar 6 propagates Σ outward from heart, the per-node Σ-displacement is written to a `highlight: Vec<f32>` SoA column in RenderFrame; shader reads it as bloom intensity
- **WIRING ONLY:** ndarray::hpc::renderer (exists), Pillar 6 EWA-Sandwich (exists), UnifiedBridge auth (exists post D-SDR-5), MedcareBridge specialisation (W4 sprint-4 spec), thinking-engine intent (W6 sprint-4 spec), drug-knowledge crosswalk (MedCare-rs 2026-05-05 release)

**Sprint-5 candidate PRs (in order):**
1. FMA canonical-pose seeder → `RenderFrame` (lance-graph or new fma-render crate)
2. `highlight: Vec<f32>` column addition to RenderFrame + Σ-displacement write-back from Pillar 6 (ndarray)
3. q2 frontend Gaussian-splat shader (additive blending, bloom, cyan palette)
4. Heart-click integration test: click → Cypher → SPO neighbors → Pillar 6 Σ propagation → highlight column update → next render frame shows pulse wave

Open: (a) per-system layer toggle (skeletal/cardiovascular/etc) ergonomics in q2 cockpit; (b) audio cue layer — Web Audio API triggered on highlight peak? (c) hover-vs-click semantics — hover preview should be free since renderer streams 60fps anyway.

## 2026-05-13 — CORRECTION-OF the just-above 3DGS-prerender row: ndarray ALREADY ships the 60fps SIMD double-buffer renderer (`ndarray::hpc::renderer`) — no prerender needed for FMA heart-click

Per user-supplied source pointer to `ndarray/src/hpc/renderer.rs:1` ("SIMD-accelerated double-buffer renderer for SPO graph visualization … hardware-acceleration mothership for q2 cockpit / Palantir Gotham / Neo4j-style visual rendering"). Front/back LazyLock<RwLock<RenderFrame>> swap via AtomicUsize, F32x16::mul_add force integration, `cached_splat(DT_60)` canonical-tick optimization, SoA layout (positions+velocities+charges+fingerprints). Sprint-5 FMA pickup: seed `RenderFrame` from FMA SPO triples (positions from layout algorithm; fingerprints from VSA encoding); run force-directed integration at 60fps; q2 reads `front` buffer via REST/SSE; heart-click = Cypher → SPO neighbor query → frame update. The Tier-3 prerender escape hatch is DEFERRED — only worth doing if 75K-entity live integration is measured to fail. See EPIPHANIES 2026-05-13 ndarray-renderer FINDING entry.

## 2026-05-13 — Separate-and-orthogonal: 3D Gaussian-Splat prerender buffer as Tier-3 FMA render path for q2 (Amiga-demoscene escape hatch)

Distinct from EWA-Sandwich (which is graph covariance math, see correction above). Prerender 900–18,000 camera-fly-through frames of the 75K-entity FMA anatomy as a 3DGS scene; stream from buffer to q2; heart-click = seek-in-buffer, not live 75K-entity render. SPO graph still drives click semantics + audit + drug-knowledge crosswalk; splat buffer is the visual layer only. Open: 3DGS vs surfels vs point-cloud; prerender job ownership (CI nightly vs one-shot); buffer format (.splat/.ply/Arrow temporal codec); crate home — probably new `crates/lance-graph-render-buffer/` rather than reusing `jc` since the math overlap is only the kernel name.

## 2026-05-13 — Super-domain subcrate scaffolding cascade: finalize MedCare migration → smb-bridge retrofit → woa-rs extraction → hiro-rs / hubspot-rs new

**Status:** Open
**Priority:** P1 (sequenced after Pattern E+F+cognition cascade lands, which provides the manifest schema each subcrate registers under)
**Scope:** crate:medcare-analytics crate:medcare-bridge crate:medcare-realtime crate:smb-bridge crate:woa-rs crate:hiro-rs crate:hubspot-rs D-SDR-8 D-SDR-9 D-SDR-21 D-SDR-22 domain:super-domain domain:consumer-scaffolding

Captures the 2026-05-13 super-domain-as-subcrate finding. Per-`SuperDomain` enum variant gets its own specialised subcrate; Tier C (D-SDR-8/9) is not "consumer crate scaffolding" generically — it's specifically **super-domain subcrate scaffolding**.

**Sequenced PRs:**

**PR 1 — MedCare migration finalization (Healthcare super-domain proof case)** — touches `MedCare-rs/crates/medcare-analytics + medcare-bridge + medcare-realtime`.

1. Push `medcare-analytics/src/unified_bridge_wiring.rs` (commit `31e999b`, currently local-only).
2. Deprecate `medcare-analytics/src/column_mask_bridge.rs` in favour of `unified_bridge_wiring.rs` + `UnifiedBridge::with_audit_chain(SuperDomain::Healthcare, salt, JsonLinesAuditSink::healthcare())`.
3. Decide: keep `medcare-bridge` as a separate crate (current) or fold its `MedcareBridge` mapper into a `medcare-analytics::bridge` submodule? **Recommended:** fold + re-export from `medcare-analytics::bridge`, keeping `medcare-bridge` as a `pub use medcare_analytics::bridge::*` re-export shim for downstream consumers during the migration window.
4. Publish a single `medcare-rs::healthcare` re-export (`pub use medcare_analytics::bridge::*; pub use medcare_realtime::*` etc.) that downstream consumers import as one symbol.
5. Add the `/modules/healthcare/manifest.yaml` entry (per Pattern E) declaring `(G=Healthcare, version=V1, entity_types=..., rbac_policy=medcare_policy, action_capabilities=..., stack_profile=hipaa, actor_type=HealthcareActor, thinking_styles=[Clinical, Diagnostic, Procedural])`. Build-script ingests it into the compile-time `MODULES` table.
6. ~250 LOC + 2 deprecation comments + 1 manifest YAML + 4 integration tests.

**PR 2 — smb-bridge retrofit (WorkOrderBilling super-domain)** — touches `smb-office-rs/crates/smb-bridge`.

1. Push commit `342f601` (currently local-only).
2. Same migration shape as PR 1 but smaller (smb-bridge is already a single-crate consumer). Replace `auth-rls` standalone path with `UnifiedBridge::with_audit_chain(SuperDomain::WorkOrderBilling, ...)`.
3. Add `/modules/work-order-billing/manifest.yaml` entry.
4. ~150 LOC + 1 manifest + 3 integration tests.

**PR 3 — woa-rs extraction (existing WoaBridge → super-domain subcrate)** — moves `lance-graph-ontology::bridges::woa_bridge.rs` out into a new `/home/user/woa-rs` subcrate behind `MetaBridge` (post-D-SDR-19). Add `/modules/work-order-app/manifest.yaml`. ~200 LOC migration + 3 tests.

**PR 4 — hiro-rs new subcrate (TicketTool/Hiro super-domain slot, D-SDR-8 refined)** — `HiroBridge::from_registry()` + absorbs OSLC-* with provenance lineage per §5. Manifest declares `super_domain=TicketTool`. Composes against Pattern E+F+cognition cascade. ~150 LOC + 1 manifest + 3 tests.

**PR 5 — hubspot-rs new subcrate (TicketTool/HubSpot super-domain slot, D-SDR-9 refined)** — `HubspotBridge::from_registry()` + CRM vocabulary. Manifest declares `super_domain=TicketTool` (note: TicketTool basin holds multiple slot variants; D-SDR-7 OGIT TTL fork PR distinguishes Hiro vs HubSpot entity types). ~150 LOC + 1 manifest + 3 tests.

**Sequencing rationale:** PR 1 (medcare) must finalize before PR 4/5 (new subcrates), otherwise hiro-rs / hubspot-rs scaffold against a half-migrated pattern and accumulate a second-order dedup row in the entropy ledger. PR 2/3 (smb-bridge + woa-rs) can interleave with PR 1 since they touch independent repos.

**Dependency on Pattern E+F+cognition cascade:** each PR's manifest entry consumes the schema defined in D-MANIFEST-MODULES-4. The cascade ideally lands first; if scheduling forces overlap, the manifest files in this idea can be authored as drafts and the build-script rejects them with a clear error until D-MANIFEST-MODULES-4 lands.

**Open sub-questions:**

- For the Hiro / HubSpot slot distinction within `SuperDomain::TicketTool` — does the enum need to split (e.g. `TicketToolHiro` + `TicketToolHubspot` as separate variants) or is one variant sufficient with the slot disambiguating? Recommended: keep one variant; D-SDR-7 TTL fork-PR holds the entity-type-level distinction.
- For the WorkOrderBilling / WoA / smb-bridge overlap — both smb-bridge and woa-rs declare `super_domain=WorkOrderBilling`. Is that the right factoring, or should WoA get its own `SuperDomain::WorkOrderApp` variant? Recommended: one super-domain (WorkOrderBilling), two consumer subcrates (smb + woa) that the manifest disambiguates via `actor_type`.
- For MedCareV2 (C# overlay, per §18): does it consume `medcare-rs::healthcare` via Arrow Flight SQL (Phase 5+) or via HTTP+JSON (Tier H M2-M6)? Recommended: HTTP+JSON first (D-SDR-35..39), Flight SQL when Phase 5 starts.

Cross-ref: `EPIPHANIES.md` 2026-05-13 super-domain subcrate finding; `EPIPHANIES.md` 2026-05-13 Pattern E+F+cognition cascade (this idea's prerequisite); `TECH_DEBT.md` TD-SUPER-DOMAIN-SUBCRATES-1 (today) + TD-SDR-CONSUMER-PUSH-1 (PR 1/2 are this row's payoff); spec `super-domain-rbac-tenancy-v1` §3.4 + §3.6 + §3.7 + §4 + §8 Tier C; `MedCare-rs/crates/medcare-analytics/src/unified_bridge_wiring.rs`; `smb-office-rs/crates/smb-bridge/src/unified_bridge_wiring.rs`.

---

## 2026-05-13 — Pattern E+F+cognition cascade: ship manifest + ractor supervisor + thinking-engine bridge together (3-PR sequence)

**Status:** Open
**Priority:** P0 (architectural — every later D-SDR ships against the wrong substrate until this lands)
**Scope:** @callcenter-membrane @truth-architect crate:lance-graph-callcenter crate:thinking-engine D-MANIFEST-MODULES-4 D-RACTOR-SUPERVISOR-5 D-SDR-13 D-SDR-15 D-SDR-17 D-SDR-19 domain:cognition domain:topology domain:auth

Captures the 2026-05-13 two-paths-converging finding (`EPIPHANIES.md`). The wire-thinking-engine idea below carried Path A only; Path B (ractor supervisor) is the other half. Ship them as a single cascade, in order:

**PR 1 — D-MANIFEST-MODULES-4** (PostNuke `/modules/<name>/manifest.yaml` + build-script). Generates compile-time `MODULES: [ConsumerEntry; N]` static carrying `(G, version, entity_types, rbac_policy, action_capabilities, stack_profile, actor_type, thinking_styles)` per consumer. Zero edits to `lance-graph-contract` after this. ~250 LOC + build-script + 3 manifest entries (medcare, smb, woa) + 4 tests.

**PR 2 — D-RACTOR-SUPERVISOR-5** (`crates/lance-graph-callcenter/src/supervisor.rs`, ~400 LOC). `CallcenterSupervisor` ractor consuming the compile-time `MODULES` table, spawning each active consumer on boot, routing typed messages to the right addr — all in ractor sync mode (I-2: tokio outbound only / sync ractor inbound). 8-arm handler mapped 1:1 from `cognitive-shader-driver/src/grpc.rs` (dispatch / ingest / qualia / styles / health / tensors / calibrate / probe). Per-consumer crash isolation + restart strategy. ~5 integration tests covering boot, dispatch, crash-restart, and the I-2 BBB seam.

**PR 3 — `cognition_bridge`** (`crates/lance-graph-callcenter/src/cognition_bridge.rs`, ~300 LOC). Composes Path A (thinking-engine substrate) against Path B (the per-consumer actor address from PR 2). Exposes `RoleProjection::for_role`, `ActorPersona::from_jwt`, `AwarenessFrame::project` on the actor's handler boundary. `UnifiedBridge::authorize_*` extension `with_cognition(...)` builder makes the cognitive surface optional; audit events carry `awareness_root: u64` in addition to `merkle_root`. ~5 integration tests covering each authorize op × Allow/Deny/Escalate.

**Net deliverable collapse:** D-MANIFEST-MODULES-4 + D-RACTOR-SUPERVISOR-5 + D-SDR-13 + D-SDR-15 + D-SDR-17 (originally 5× scaffolded clean-room ≈ ~830 LOC + 23 tests) → 3-PR cascade ≈ ~950 LOC + 14 tests **composed against thinking-engine** instead of duplicating it. Architectural payoff: `lance-graph-callcenter` becomes the telephony-switching supervisor its name has promised since day one, cognitive substrate has a runtime home, and adding a new consumer = drop a manifest + add a Cargo dep + ~30 LOC glue.

**Sub-questions to resolve in PR 1's review:**

- Does the manifest build-script live in `lance-graph-callcenter/build.rs` or a new `lance-graph-modules` crate? (Probably the latter to keep callcenter's compile graph clean.)
- Does the `thinking_styles: Vec<ThinkingStyleId>` manifest field reference contract-canonical (36) styles or the planner's 12-ord projection? (Probably contract-canonical; `ord_to_thinking_style` driver.rs:677 handles the down-projection.)
- For BBB enforcement: does the `actor_type` enum gate which ractor handler shape the consumer gets (compile-time-typed dispatch), or does the supervisor dispatch dynamically with a trait object? (Sketch suggests compile-time-typed; let's confirm.)
- For Pattern F sync-mode invariant: how do we test the I-2 seam? Probably a compile-fail test asserting `tokio::spawn` cannot be called inside a handler body. `crates/lance-graph-callcenter/tests/zone_serialize_check.rs` is the prior-art template for compile-fail invariant tests.

**Open follow-ups (out of cascade scope, queued for after merge):**

- D-SDR-25 (DriftDetectionBridge) composes against `thinking-engine::ground_truth` + `cronbach` once PR 3 lands.
- D-SDR-26 (determinism rules) composes against `thinking-engine::reencode_safety` (x256-proven byte-determinism).
- D-PARITY-V2-3..12 (DTO ladder rest) composes against `thinking-engine::tensor_bridge` + `meaning_axes` + `superposition`.

Cross-ref: `EPIPHANIES.md` 2026-05-13 two-paths-converging finding; `TECH_DEBT.md` TD-RACTOR-SUPERVISOR-5 + TD-MANIFEST-MODULES-4 + TD-THINKING-ENGINE-UNWIRED-1; `.claude/plans/compile-time-consumer-binding-v1.md` Pattern E+F design; `.claude/plans/anatomy-realtime-v1.md` W11 gate; `.claude/handovers/2026-05-13-0855-brainstorm-arc-synthesis.md` §6 priority-ordered next steps (this cascade promotes to Phase 0.5 — before D-SDR-13/15/17).

---

## 2026-05-13 — Wire `thinking-engine` into UnifiedBridge — collapse D-SDR-13/15/17 into one bridge module

**Status:** Open
**Priority:** P1 (highest leverage in the workspace per `EPIPHANIES.md` 2026-05-13 thinking-engine finding)
**Scope:** @callcenter-membrane @truth-architect crate:thinking-engine crate:lance-graph-callcenter D-SDR-13 D-SDR-15 D-SDR-17 D-SDR-19 domain:auth domain:cognition

The thinking-engine crate (48 modules, 16,211 LOC, 582 KB) is shipped and indexed in `CLAUDE.md § Thinking Engine` but consumed by zero callcenter-side code. The §16-§19 spec's outstanding D-SDR deliverables map cleanly onto its existing modules (see the table in the 2026-05-13 epiphany).

**Proposed wiring (single PR ~300 LOC, ~3-5 integration tests):**

1. **New module `lance-graph-callcenter::cognition_bridge`** — thin adapter exposing:
   - `RoleProjection::for_role(actor_role: &str) -> Vsa16kF32` — wraps `thinking_engine::role_tables::*`
   - `ActorPersona::from_jwt(claims: &JwtClaims) -> PersonaCard` — wraps `thinking_engine::persona::*`
   - `AwarenessFrame::project(decision: &AccessDecision, persona: &PersonaCard) -> AwarenessDto` — wraps `thinking_engine::awareness_dto`
2. **`UnifiedBridge::authorize_*` extension** — optional `with_cognition(cognition_bridge: Arc<CognitionBridge>)` builder. When set, the audit event carries an `awareness_root: u64` (FNV-1a of `AwarenessDto::canonical_bytes`) in addition to `merkle_root`. Backward-compatible: noop bridge stays default.
3. **`Policy::evaluate` extension** — receives the `RoleProjection` fingerprint alongside `actor_role: &str`. Allows role permissions to be authored against canonical role fingerprints (cross-tenant role aliasing) without disturbing the existing canonical-name pathway. Policy evaluator uses cosine resonance against the codebook when string match misses.
4. **Hard-lock matrix (D-SDR-17) implementation** — leverages `osint_bridge.rs` from thinking-engine for the OSINT-side projection that the Healthcare ↔ OSINT crypto barrier needs to recognise. Static partner table lives in `lance-graph-callcenter::super_domain::HARD_LOCK_PARTNERS`.
5. **DP role (D-SDR-15)** — leverages `contrastive_learner.rs` + `cronbach.rs` from thinking-engine for the ε-bounded noise + k-anonymity floor primitives.

**Net deliverable collapse:** D-SDR-13 + D-SDR-15 + D-SDR-17 (originally 3× ~80-150 LOC = ~310 LOC + 13 tests) → 1× cognition-bridge PR (~300 LOC + 5 tests) that composes the thinking-engine substrate. Net LOC savings ~10-15%, but the **architectural** gain is much larger: every downstream D-SDR (Tier F MetaBridge, Tier H LanceProbe endpoints) gets the cognitive surface for free instead of re-scaffolding it.

**Open sub-questions:**
- Does `thinking-engine::contract_bridge` already expose the right shape, or does it need a new trait fan-out?
- Which of the 48 modules belong in the "Layer 2 role catalogue" per `I-VSA-IDENTITIES`, and which are "Layer 3 content stores" that should stay behind a YAML registry?
- Does the `cognitive-shader-driver` runtime expect `thinking-engine` to live on the **internal SoA side** of the BBB? If yes, the CognitionBridge needs to mediate through the BBB seam, not call `thinking-engine` directly.

Cross-ref: `EPIPHANIES.md` 2026-05-13 thinking-engine finding; `TECH_DEBT.md` TD-THINKING-ENGINE-UNWIRED-1; `.claude/handovers/2026-05-13-0855-brainstorm-arc-synthesis.md`; `CLAUDE.md § Thinking Engine`; `.claude/knowledge/lab-vs-canonical-surface.md`.

---

(Prepend new ideas here with today's date. Format:)

```
## YYYY-MM-DD — <short title>
**Status:** Open
**Priority:** P0 | P1 | P2 | P3
**Scope:** @<agent> D<N> domain:<tag>

<one paragraph: what the idea is, rough scope, why it matters>

Cross-ref: <epiphany entry / plan D-id / related knowledge doc>
```

---

## Implemented Ideas

(When an Open idea ships, APPEND here with same title + PR anchor.)

## 2026-04-29 — Probe P1: γ-phase-offset ranking discrimination (from 2026-04-29)
**Status:** Implemented 2026-04-29 via PR (this PR)
**Result:** PASS — min Spearman ρ = -0.963 across pairs of γ-offsets

Drained Probe P1 from `bf16-hhtl-terrain.md` Probe Queue (NOT RUN → PASS).
Tests Constraint C3's "VALID — pre-rank discrete selector" regime: 4
γ-phase offsets at stride 1/(4φ) on a 256-entry codebook produce
meaningfully different rankings. Pairwise Spearman ρ shows expected
gradient: adjacent offsets co-monotonic (+0.51), maximum-spaced offsets
near-anti-monotonic (-0.96). Dupain-Sós discrepancy property empirically
confirmed in synthetic regime; γ+φ encoding strategy in `bgz-tensor` is
grounded.

Cross-ref: `crates/jc/src/probe_p1_gamma_phase.rs`,
`.claude/knowledge/bf16-hhtl-terrain.md` Probe Queue P1 (now PASS),
`.claude/board/EPIPHANIES.md` 2026-04-29 FINDING entry.

```
## YYYY-MM-DD — <same title as Open entry> (from YYYY-MM-DD)
**Status:** Implemented YYYY-MM-DD via PR #NNN
**Shipped as:** D<N> in integration plan v<K>
**PR:** #NNN (commit SHA)

<verbatim original Open paragraph>

Cross-ref: <same + PR link + plan D-id>
```

The original Open entry's Status flips to `Implemented YYYY-MM-DD`.

---

## Integration Plan Update Log

(When an idea triggers a plan change — version bump, D-id status
move, new deliverable — APPEND here. This is the third-entry row.)

```
## YYYY-MM-DD — Plan consequence of <idea title> (from YYYY-MM-DD)
**Trigger idea:** <idea title> (YYYY-MM-DD)
**Plan change:** <version bump / D-id flip / deliverable added>
**Plan entry:** `INTEGRATION_PLANS.md` v<K> entry or new v<K+1> entry
**Status board update:** <D-id> → <new Status>

<one paragraph: what the plan documented differently after this idea>
```

---

## Rejected / Deferred Ideas

(Ideas that don't graduate go here.)

```
## YYYY-MM-DD — <same title as Open entry> (from YYYY-MM-DD)
**Status:** Rejected YYYY-MM-DD  |  Deferred to <when / trigger>
**Rationale:** <short explanation>

<original Open paragraph>

Cross-ref: <original + any related>
```

---

## How to use this file

**When a new architectural idea surfaces** — prepend to **Open
Ideas** with today's date. One paragraph. If it needs more, create
a knowledge doc and link.

**When an Open idea ships** — APPEND to **Implemented Ideas**; flip
Open Status to `Implemented YYYY-MM-DD`. Then APPEND to
**Integration Plan Update Log** with the plan consequence.

**When an Open idea is rejected** — APPEND to **Rejected /
Deferred Ideas** with Rationale; flip Open Status.

**When a deferred idea reactivates** — prepend a NEW Open entry
citing the deferred one; flip the deferred entry's Status to
`Reactivated YYYY-MM-DD <new-entry>`.

Nothing is lost. Every idea has a trail from speculation to
disposition.

## 2026-04-29 — Inverted-pyramid awareness streaming via CausalEdge64 through SPO+COCA→CAM_PQ
**Status:** Open
**Priority:** P2
**Scope:** @savant-research cognitive-shader-driver thinking-engine domain:streaming domain:awareness

When weight rows stream through the inverted pyramid (L4 16384² → L1 64²),
can the BF16 mantissa awareness (Column F `AwarenessColumn`, per
`bindspace-columns-v1.md`) flow through CausalEdge64 (Column D) at each
fold step — so awareness-annotated edges emit without a separate pass?

SPO 2³ + COCA → CAM_PQ is one pipeline (CAM_PQ Semantic CLAM trains
from COCA vectors). The question is not "which encoding wins" but whether
the awareness sidecar (BF16 mantissa quality → u8 per word) survives
the pyramid compression and produces meaningful CausalEdge64 updates
(frequency/confidence/Pearl 2³ mask) at each resolution level.

Routes through `shader-lab` Lab infra. Test infrastructure exists:
`polarquant_hip_probe.rs`, `turboquant_correction_probe.rs`, Phase 0
DTOs (`WireSweep`, `WireCalibrate`, `WireTokenAgreement`).

Cross-ref: `bindspace-columns-v1.md` (Column D/F), `causal-edge/src/edge.rs`,
`BGZ_HHTL_D.md`, `codec-sweep-via-lab-infra-v1.md`.

## 2026-04-29 — Probe P1: γ-phase-offset ranking discrimination
**Status:** Implemented 2026-04-29 (this PR)
**Priority:** P1
**Scope:** @savant-research jc bgz-tensor domain:probe-queue domain:codec

Execute Probe P1 from `bf16-hhtl-terrain.md` queue (status: NOT RUN). Tests
Constraint C3 directly: γ+φ as pre-rank discrete selector should produce
*different* rankings for different offsets on the same base codebook. If
yes (ρ between rankings differs by >0.01 across offsets) — γ+φ pre-rank
selector is VALID, Dupain-Sós discrepancy property holds. If no (ρ identical)
— γ+φ joins the dead post-rank regime as a DEAD axis.

Implementation form: jc-style probe (pure Rust, zero deps, ~250 lines).
Synthesize plausible 256-entry codebook, apply 4 γ-phase-offset shifts,
rank-by-distance under each, compute pairwise Spearman ρ. PASS if any
two offsets produce ρ < 0.99 (rankings meaningfully differ). FAIL if all
pairwise ρ > 0.999 (offsets are no-ops).

Result feeds back into `bf16-hhtl-terrain.md` Probe Queue as P1 status
update (NOT RUN → PASS or FAIL). On FAIL, downstream consequence: γ+φ
encoding strategy needs revision; CONJECTURE label on existing γ-related
architecture stays.

Cross-ref: `.claude/knowledge/bf16-hhtl-terrain.md` Probe Queue P1,
Constraint C3, `crates/bgz-tensor/src/gamma_phi.rs`,
`crates/bgz-tensor/src/gamma_calibration.rs`.
## 2026-04-29 — Safetensor-Streaming als ndimensionale Bedeutungsakkumulation
**Status:** Open
**Priority:** P2
**Scope:** @savant-research @palette-engineer bgz-tensor learning domain:hydration domain:cascade

Stream a safetensor (1B–70B params) tile-by-tile through the existing
HHTL cascade instead of loading into memory. Per tile: Hadamard-rotate
(`fractal_descriptor`), extract Σ, propagate via EWA-sandwich (PR #289),
accumulate in `holograph::width_16k::SchemaSidecar` Block 14/15. Estimated
3.8 min for 7B model based on Pillar 6 measured 2 ms/sandwich latency.
**CONJECTURE** — depends on Probe M2 / P3 (4096 terminal buckets correlate
with COCA vocabulary?) being PASS before tile-streaming approach is
guaranteed information-preserving.

Cross-ref: `IDEA_JOURNAL_2026_04_29_STREAMING_HYDRATION.md` (full context),
`bf16-hhtl-terrain.md` probe queue P3, `cognitive-shader-architecture.md`
(weights-as-seeds doctrine).

## 2026-04-29 — Family-Bounds als globale fraktale Codierung (Hypothesis Test)
**Status:** Open
**Priority:** P3
**Scope:** @savant-research bgz-tensor domain:fractal domain:hypothesis-test

Hypothesis: gesamtheit aller HighHeelBGZ family bounds bildet selbst-
ähnliche Hierarchie kodierbar als Fraktal mit on-demand decoding statt
vollständiger Materialisierung. **CONJECTURE** — `fractal_descriptor`
misst Selbst-Ähnlichkeit *pro Row*, nicht *global*. Vorbedingung:
Diagnostik-Probe ob globale Fraktalität existiert. PASS-Kriterium:
Hurst ≠ 0.5, fraktale Dim > 1, Spektrum-Breite > 0 auf der Verteilung
der family bounds. FAIL: Idee verworfen, lokale per-Row-Fraktalität ist
nicht globale Eigenschaft.

Cross-ref: `IDEA_JOURNAL_2026_04_29_STREAMING_HYDRATION.md`,
`fractal-codec-argmax-regime.md`, `endgame-holographic-agi.md`.

## 2026-04-29 — Pillar 7 Front-to-Back α-Akkumulation (LIKELY-REDISCOVERY)
**Status:** Open
**Priority:** P3
**Scope:** @savant-research jc bgz-tensor domain:cascade domain:probe

Apply 3DGS front-to-back α-blending with early-termination (`if α_acc > 0.95: break`)
to HHTL cascade. KS Pillar 5+ would certify that omitted sources fall
within concentration bound. **CONJECTURE / LIKELY-REDISCOVERY** —
`bgz-tensor::cascade` already implements HHTL (HEEL/HIP/TWIG/LEAF) with
metric-induced sparsity, which is a form of early-termination already.
Re-filing this pillar specifically should investigate whether it adds
α-blending novelty over existing cascade or duplicates known terrain.
Read `cascade.rs` + `attention.rs` headers BEFORE building.

Cross-ref: `IDEA_JOURNAL_2026_04_29_FUTURE_PILLARS.md`,
`crates/bgz-tensor/src/cascade.rs`, `crates/bgz-tensor/BGZ_HHTL_D.md`.

## 2026-04-29 — Pillar 8 Adaptive Densification für Σ-Codebook
**Status:** Open
**Priority:** P2
**Scope:** @palette-engineer @family-codec-smith jc bgz-tensor domain:codebook domain:adaptive

3DGS-style split (high error + many edges) and prune (low assignment count)
operations on the Σ-codebook from PR #288 (R² = 0.9949). Total k=256 stays
constant; codebook adapts to actual edge distribution online. **CONJECTURE** —
heuristic could oscillate vs converge. Pre-condition: probe must demonstrate
monotonic R² improvement over 50 densification passes. Builds on the
already-merged sigma_codebook_probe.

Cross-ref: `IDEA_JOURNAL_2026_04_29_FUTURE_PILLARS.md`, PR #288
(sigma_codebook_probe), KS Pillar 5+ for convergence guarantee.

## 2026-04-29 — Pillar 9 SH-Koeffizienten als Thinking-Style-Manifold
**Status:** Open
**Priority:** P3
**Scope:** @cognitive-shader-driver learning bgz-tensor domain:cognitive-style domain:architecture

Replace categorical thinking_style (analytical/creative/focused) with
continuous SH-coefficient manifold evaluated against query view-direction.
DZ Pillar 5++ already certifies the underlying Hilbert-space CLT.
**CONJECTURE — TOUCHES PRODUCTION CODE.** Would modify
`learning::cognitive_styles` and `awareness_dto::ResonanceDto::ThinkingStyle`.
Pre-condition: explicit architecture decision required before any
implementation — not a pure-math pillar like 5+/5++/6, but an actual
substrate behavior change. Hold until that decision is made.

Cross-ref: `IDEA_JOURNAL_2026_04_29_FUTURE_PILLARS.md`,
`cognitive-shader-architecture.md`, DZ Pillar 5++ (PR #287).

## 2026-04-19 — FP_WORDS = 256 (supersede the 160 plan)
**Status:** Open
**Priority:** P1
**Scope:** @container-architect @truth-architect ndarray domain:vsa domain:codec

Current: `FP_WORDS = 157`  (10,048 bits, 5-word remainder on AVX-512)
Planned (H6 harvest): `FP_WORDS = 160`  (10,240 bits, SIMD-clean)
**Proposed: `FP_WORDS = 256`**  (16,384 bits, cache-line-perfect, matches `Container<[u64; 256]>`)

**Why 256 over 160:**

- LanceDB `FixedSizeList<UInt8, 2048>` = 2 KB per row = 16,384 bits already.
  Padding 157 → 256 in Container currently wastes 99 u64 per fingerprint (62%).
- Container primitive is already `[u64; 256]`; unifying `FP_WORDS` with it
  means zero padding, zero remainder loops at any SIMD level, cache-line
  alignment guaranteed (2 KB / 64 B = 32 cache lines, every level clean).
- VSA capacity: Plate's bound rises ~1.6× (bundled-items-per-fingerprint
  capacity ~1,500 → ~2,400 at error < 1%).
- No rebake of stored fingerprints needed — Container was already 256 wide.

**Cost:** ~30 LOC in `ndarray::hpc::vsa` constants + test updates;
docs shift "10k VSA" language → "16k VSA". Plate's capacity math re-tune.

**Supersedes:** TECH_DEBT entry "FP_WORDS = 157 (not 160); SIMD remainder
loops remain" — the 160 plan was the right direction, 256 is the correct
destination.

**Cross-ref:** `.claude/knowledge/cross-repo-harvest-2026-04-19.md` H6,
`.claude/board/TECH_DEBT.md` FP_WORDS entry. Container layout in
`lance-graph-contract::cam::Container`.

## 2026-04-19 — CORRECTION-OF 2026-04-19 FP_WORDS = 256 (supersede the 160 plan)
**Status:** Open
**Priority:** P1
**Scope:** @container-architect @truth-architect ndarray domain:vsa domain:codec

The prior entry conflated **two distinct substrates** and used
"10,000-D binary VSA" framing that must be eliminated from the workspace.

### Two substrates (never collapse them again)

1. **Hamming binary fingerprint** — `Container<[u64; 256]>` = 16,384
   BITS = **2 KB**. For popcount-Hamming queries. **Not VSA.** FP_WORDS
   going from 157 → 256 applies here.

2. **VSA superposition substrate** — 16,384 DIMENSIONS × float.
   For bind / bundle / permute / unbind. **Never binary.**

   | Encoding | Bytes / fingerprint | LanceDB column |
   |---|---|---|
   | `Vsa16kF32` (lossless baseline) | **64 KB** | `FixedSizeList<Float32, 16384>` |
   | `Vsa16kBF16` | **32 KB** | `FixedSizeList<BFloat16, 16384>` |
   | `Vsa16k` u8 × 5-lane | **80 KB** | struct of 5 × `FixedSizeList<UInt8, 16384>` |
   | `Vsa16k` BF16 × 5-lane | **160 KB** | struct of 5 × `FixedSizeList<BFloat16, 16384>` |

   Current `Vsa10kF32` = 10,000 × f32 = 40 KB is the legacy narrower
   size. Move to 16,384-D.

### Governance: ban "10,000 binary" framing

**There shall be zero occurrences of "10,000-D binary VSA" / "10,000-bit
VSA" in any `.claude/*`, knowledge doc, skill doc, or board file.**
Those phrases collapse two distinct objects. When writing about:

- Binary fingerprint: say "16,384-bit Hamming fingerprint" / "2 KB
  Container" — never "VSA".
- VSA substrate: say "16,384-D float VSA (64 KB lossless / 80 KB u8-5-lane
  / 160 KB BF16-5-lane)" — never "binary", never "10k".

### Tasks (follow-up PR, not this one)

1. Rename `CrystalFingerprint::Vsa10kF32` → `Vsa16kF32` and
   `Vsa10kI8` → `Vsa16kI8` in `lance-graph-contract::crystal`.
2. Re-address role-key slices from [0..10000) → [0..16384) in
   `lance-graph-contract::grammar::role_keys`. Maintain disjoint
   slices; scale each segment proportionally (e.g., SUBJECT 2000 → 3200).
3. Update storage contracts to `FixedSizeList<Float32, 16384>` and
   the 5-lane struct variant. LanceDB needs no patching — both are
   native.
4. Sweep 21 lance-graph + 7 ndarray files for "10,000" / "Vsa10k*"
   / "10 000-D" / "10K VSA" → rename or reclassify. Exclude
   legitimate uses (query limits, sample counts, dollar amounts,
   speedup ratios, scale factors).

**Supersedes:** 2026-04-19 IDEAS entry "FP_WORDS = 256 (supersede the
160 plan)" — that entry was correct for the binary Hamming substrate
but mislabeled the VSA as "16,384 bits". The VSA dimension is 16,384
FLOAT, not bits.

## 2026-04-19 — REFINEMENT-OF 2026-04-19 CORRECTION-OF FP_WORDS scope
**Status:** Open
**Priority:** P2
**Scope:** @integration-lead domain:vsa

Scope the "no 10000-D VSA" ban to the three contexts where it is
LEGITIMATELY in use and must be preserved:

1. **Grammar prototype** — `lance-graph-contract::grammar::{role_keys,
   context_chain}`. Role-key slices `[0..10000)` are shipped in PR #210.
   Rename to 16384-D is a follow-up that must re-scale all slice
   boundaries proportionally; until that PR lands, 10,000-D addressing
   stays in grammar docs.
2. **Quantum prototype** — `CrystalFingerprint::Vsa10kF32` holographic
   residual mode (`crystal-quantum-blueprints.md`). Quantum-mode docs
   keep 10,000-D naming until the rename PR.
3. **Ladybug-rs / bighorn fresh imports** — PRs #200-203 brought the
   cognitive stack + CognitiveShader + BindSpace at 10,000-D. Known
   memory cost (see TECH_DEBT "Ladybug 10000-D memory blowup"). Do not
   rewrite these imports; migrate as part of the ladybug → contract
   consolidation PR.

**Elsewhere** (epiphanies, session handovers, OSINT plans, calibration
docs, prompts not in the above scopes): strip 10,000-D / Vsa10k*
references — they propagate the legacy substrate into contexts where
only 16,384-D is relevant.

**Files in-scope (keep as-is):**
- `.claude/plans/elegant-herding-rocket-v1.md` (grammar + quantum)
- `.claude/knowledge/crystal-quantum-blueprints.md` (quantum)
- `.claude/knowledge/integration-plan-grammar-crystal-arigraph.md` (grammar)
- `.claude/knowledge/linguistic-epiphanies-2026-04-19.md` (grammar)
- `.claude/knowledge/endgame-holographic-agi.md` (quantum / holographic)
- `.claude/prompts/session_ndarray_migration_inventory.md` (i8 10000D
  transient accumulation layer is the ladybug-import artifact)
- `.claude/board/PR_ARC_INVENTORY.md` (historical record of #208-#210)

**Files out-of-scope (sweep-candidate for rename / restatement):**
- `.claude/board/LATEST_STATE.md` — snapshot says `Vsa10kI8/F32` in
  CrystalFingerprint; append correction row naming the target
  (`Vsa16kI8/F32`) when rename PR lands.
- `.claude/prompts/session_deepnsm_cam.md` — "10,000 bits each
  (= Base17 compatible)" is a binary-VSA confusion; correct.
- `.claude/board/EPIPHANIES.md` — "10,000-D f32 VSA is lossless under
  linear sum" entry from another session — keep as historical record;
  append correction that the target is 16,384-D.

**Acknowledges:** the prior CORRECTION-OF entry framed the ban as
workspace-wide; it is not. Three scopes preserve 10,000-D legitimately
until the coordinated rename PR lands.

## 2026-04-19 — REFINEMENT-2 HDC substrate is FP16 / BF16, not FP32
**Status:** Open
**Priority:** P1
**Scope:** @container-architect @truth-architect domain:vsa domain:codec domain:memory

The prior CORRECTION-OF + REFINEMENT-OF entries assumed f32 as the
HDC baseline. Correction: **HDC superposition substrate is FP16 (or
BF16), not FP32.** Bundle-accumulation magnitudes stay in half-precision
range; f32 only buys ceremony.

**Size table (corrected):**

| Substrate | Per-row bytes |
|---|---|
| 1,024-D Jina v3 (FP16) | 2 KB |
| 1,536-D OpenAI text-embedding-3-small (FP16) | 3 KB |
| 3,072-D Upstash Vector cap (FP16) | 6 KB |
| **10,000-D HDC (current, FP16)** | **20 KB** |
| 10,000-D HDC (legacy f32 naming) | 40 KB ← `Vsa10kF32` today |
| **16,384-D HDC target (FP16)** | **32 KB** |
| 16,384-D HDC × u8 5-lane | 80 KB |
| 16,384-D HDC × BF16 5-lane | 160 KB |

**Revised memory math for the ladybug 700-1100 MB blowup:**

- At f32 (40 KB/row): observed 700-1,100 MB = ~17-27 K live rows.
- **Had it been FP16 (20 KB/row): same population = 350-550 MB.**
- 16k × FP16 (32 KB/row): same population = 560-880 MB — **cheaper
  than the current f32 state**, not worse.

The 16k rename is memory-positive IF paired with f32 → FP16 migration.
Without the precision drop, 16k × f32 (64 KB/row) does inflate the
problem. The coupled change is the right design.

**Architectural constraint (why LanceDB, not a vector-db SaaS):**

Commercial managed vector DBs cap at ≤ 3072 dimensions (Upstash).
Pinecone, Weaviate, Qdrant — all optimize for 768-3072 dense
embeddings. HDC substrate at 16,384-D is an order-of-magnitude wider
and cannot live in those systems. LanceDB's `FixedSizeList<BFloat16,
16384>` is the only viable column type across OSS + managed
offerings. **This is why lance-graph is The Spine, not a plug-in.**

**Updated rename scope for the follow-up PR:**

1. Type rename: `CrystalFingerprint::Vsa10kF32` → `Vsa16kBF16`
   (not `Vsa16kF32`). f32 variant retires.
2. Role-key slices re-address `[0..10000)` → `[0..16384)`.
3. Storage contract: `FixedSizeList<BFloat16, 16384>` as the canonical
   HDC column; 5-lane struct for multi-representation workloads.
4. Compute: preserve f32 accumulation internally where numerical
   stability matters (unbundle / unbind hot path), round-trip via BF16
   for storage.

**Supersedes:** prior CORRECTION-OF entry's "Vsa16kF32 (lossless
baseline): 64 KB" line. The lossless baseline is BF16 at 32 KB.

## 2026-04-19 — lance-graph-cognitive refactor: dedup + merge + excise
**Status:** Open
**Priority:** P2
**Scope:** @integration-lead @container-architect domain:cognitive domain:refactor

26,240 LOC across 11 modules, yesterday's ladybug-rs harvest staged
here. Not wholesale duplicate of other crates — it's the complementary
cognitive layer sitting above `lance-graph::graph::spo` (store) and
`lance-graph-contract::{grammar,crystal}` (primitives). Needs targeted
cleanup, not deletion.

**Keep in place (canonical cognitive-layer impl):**

- `grammar/` — `GrammarTriangle` (NSM × Causality × Qualia); plan D3
  explicitly calls it (`.claude/plans/elegant-herding-rocket-v1.md`).
- `spo/` — Crystal layer: `sentence_crystal`, `context_crystal`,
  `gestalt`, `meta_resonance`, `cognitive_codebook`. Sits on top of
  the SPO store + contract's `CrystalFingerprint` enum.
- `spectroscopy/` — detector 511 + features 408 LOC, standalone
  unique cognitive-spectroscopy work.

**Merge into active crates (if feasible):**

- `search/temporal.rs` (187 LOC) → `lance-graph-planner::strategy`
  (temporal search as a strategy, not a separate module).
- `cypher_bridge.rs` → check overlap with `lance-graph::parser`;
  merge or retire.

**Inspect and decide DTO vs excise:**

- `fabric/` — protocol surface? If yes → move to contract. If no →
  keep or excise.
- `world/` — world model, likely DTO. If yes → move to contract
  (parallel to `state_classification_pillars` already there).
- `container_bs/` — BindSpace container. If DTO → move to contract
  OR let ada-rs consume through contract. If stub → excise.

**Excise:**

- `learning/` — empty stub inside lance-graph-cognitive (distinct
  from the standalone `crates/learning/` DTO crate which is a
  different thing). Delete.
- `wip` feature-flagged modules — finish or excise, not both.
- `core_full/` — catch-all; decompose into themed modules or
  migrate contents into the modules above.

**Cost:** ~1 week refactor PR. Zero functional change; contract
compliance improves, dependency graph tightens. Contract-adoption
rule from CLAUDE.md (§Current Status In-Progress) is the governing
principle: public surface through contract, implementations behind
traits.

**Cross-ref:** TECH_DEBT "ladybug-rs retired — ada-rs + lance-graph
exclusively" (2026-04-19). Active plan:
`.claude/plans/elegant-herding-rocket-v1.md` D3 depends on
lance-graph-cognitive's grammar module staying put.

## 2026-04-19 — CORRECTION-OF 2026-04-19 lance-graph-cognitive refactor
**Status:** Open

Remove all ada-rs mentions from the prior entry — ada-rs is documented
only in ada-rs, not here.

Correction: the contract surface for cognitive DTOs **already exists** —
it shipped in PR #206 (Pumpkin NPC framed: state classification pillars
+ shader-driver endpoints). The lance-graph-cognitive refactor is about
cleaning up yesterday's messy imports against that EXISTING contract, not
creating new traits.

Replace "let ada-rs consume through contract" → "exposed through
existing contract from PR #206". Replace "move to contract" in
fabric/world/container_bs bullets → "check if PR #206 contract already
covers it; if yes, delete the import; if no, extend contract via Pumpkin
framing".

## 2026-04-19 — Fractal round-trip codec: phase+magnitude preservation
**Status:** Open (research)
**Priority:** P3
**Scope:** @cascade-architect domain:codec domain:fractal

Follow-on to the fractal-leaf CORRECTION (EPIPHANIES 2026-04-19).
The unsolved codec problem:

**Encode both phase and magnitude in fractal form so that decode is
a usable round-trip (not just a statistical twin).**

Pure fractal parameters (D, w, H, σ) reconstruct a *statistical twin* —
same shape, different bits. That's argmax-usable for random queries
(Meyer cardiac-FD analogy), but loses exact inner products. Two rows
with same (D, w, H) produce indistinguishable argmax rankings, which
is a feature for compression but means per-row identity is gone.

Round-trip requires pinning enough reference points that fractal
interpolation fills between them faithfully. Candidate recipe:

1. Hadamard-rotate row → coefficients c[0..n).
2. Sample at 17 golden-step positions → Base17 anchors (34 bytes).
3. Compute fractal params of the full sequence → Descriptor (7 bytes).
4. Decode: generate fractal interpolation that matches (D, w, H) AND
   passes through the Base17 anchor points with correct signs +
   magnitudes. Fractal-interpolation-between-samples.
5. Inverse Hadamard → reconstructed row.

This binds the existing workspace primitives (Base17 golden-step,
Stacked samples, fractal descriptor) into a single round-trip codec
where:
- Base17 carries the PHASE ANCHORS (sign + coarse magnitude at 17
  golden positions).
- FractalDescriptor carries the SHAPE (D, w, σ, H) for interpolation.
- Combined: 34 + 7 = 41 bytes/row, self-similar reconstruction between
  anchors, exact at anchors.

Open research questions:
- Does fractal interpolation actually converge to something close to
  the original between anchor points? Iterated Function System theory
  says yes for self-similar sequences; empirical for Qwen3 unknown.
- Phase half (sign-sequence fractal) still needs its own probe.
- How to parameterize the sign flips between anchors without storing
  them bit-by-bit? Barnsley fern-style IFS over sign space?

All gated behind `lab` feature until the round-trip math works.
Not a production codec priority until the two unmeasured probes
(sign-sequence fractal CoV, fractal-interp-between-samples fidelity)
return positive.

Cross-ref: fractal-codec-argmax-regime.md, EPIPHANIES 2026-04-19
CORRECTION, PR #216 magnitude-only half.

## 2026-04-19 — Fractal codec validation path: use codec_rnd_bench + ICC_3_1
**Status:** Open (operational)
**Priority:** P2
**Scope:** @cascade-architect domain:codec domain:psychometry

Existing infrastructure (no new tooling needed):

**`crates/bgz-tensor/src/quality.rs`** (shipped):
- `spearman` · `pearson` · `kendall_tau` · `icc_3_1` · `cronbach_alpha`
- `mae` · `rmse` · `top_k_recall` · `bias_variance`

**`crates/thinking-engine/examples/codec_rnd_bench.rs`** (shipped):
- Loads 128 rows from safetensors
- Computes ground-truth pairwise cosines
- Runs each registered codec through the 10-metric suite
- Outputs markdown table (see `bench_qwen3_tts_62codecs.md` / `bench_gemma4_e2b_62codecs.md`)

**Correct fractal validation** (replaces the hand-rolled CoV probe):

1. Implement `FractalCodec::decode(anchors: Base17, desc: FractalDescriptor) -> Vec<f32>`
   - Fractal interpolation between 17 golden-step anchor points
   - Shape constrained by (D_mag, w, H_mag, D_phase, σ)
   - IFS / wavelet-interp / similar — this is the "genius" piece
2. Register as `FractalCodec(41 B)` candidate in codec_rnd_bench.rs
3. Run:
   ```
   cargo run --release --features lab \
     --manifest-path crates/thinking-engine/Cargo.toml \
     --example codec_rnd_bench -- /path/to/Qwen3-8B/shard.safetensors
   ```

Output: markdown row with ICC_3_1 + Cronbach's α + Spearman ρ + Pearson r
+ top-5 recall vs ground truth. Direct comparison against the existing
67-codec sweep (I8-Hadamard leader at 9 B, adaptive codec, etc.).

**Gates:**
- ICC_3_1 ≥ 0.95 on k_proj @ 41 B/row → fractal codec beats I8-Hadamard on
  argmax-rank reliability (real argmax-wall crack, measurable).
- ICC ∈ [0.85, 0.95] → useful hybrid layer, not standalone winner.
- ICC < 0.85 → fractal codec inferior; the unpublished negative.

All gated behind `lab` feature. Bench-only, never main. Endpoint already
has ICC / Cronbach / Spearman — no new dependencies. The only missing
code is the decode function.

Cross-ref: EPIPHANIES 2026-04-19 fractal-leaf CORRECTION.
`crates/bgz-tensor/src/quality.rs` lines 47/279/362. `codec_rnd_bench.rs`
for the bench structure + existing codec registration pattern.

## 2026-04-19 — Zipper codec: phase + magnitude multiplexed in single bgz17 container
**Status:** Open (architecture correction)
**Priority:** P2
**Scope:** @container-architect @cascade-architect domain:codec domain:phi

Supersedes prior "triple-channel matryoshka" proposal. Per user +
existing `.claude/knowledge/phi-spiral-reconstruction.md` § "family
zipper" concept: the bgz17 container was always designed to carry
phase-only in ~48-64 active bits of 16384. The "halo" (~16,320 bits)
is not waste — it's available storage for a MAGNITUDE stream
interleaved at a different φ-stride.

**Corrected architecture — single-container zipper:**

| Stream | Stride | Positions carried | Role |
|---|---|---|---|
| Phase | round(N / φ) ≈ N·0.618 | ~48-64 | bgz17 container active bits |
| Magnitude | round(N / φ²) ≈ N·0.382 | ~48-64 | magnitude samples in the halo |
| Halo-remainder | unused positions | ~16,200 | structural / ECC / future |

Both strides are maximally-irrational → neither locks into Hadamard
butterfly frequencies → both get the anti-moiré ("X-Trans sensor")
property. Their coincidences are themselves at φ-ratios so mutual
aliasing is "hidden moiré" — dispersed below visibility.

**Zeckendorf property:** every integer has a unique non-adjacent
Fibonacci decomposition. Two non-adjacent Fibonacci indices give
naturally-non-colliding strides — the zipper is not hand-tuned, it's
mathematical.

**Truncation hierarchy (matryoshka property preserved):**

- Read phase stride only → Base17-level coarse codec (34 B signal)
- Read phase + magnitude strides → dual-stream decoder (~70 B signal)
- Read halo remainder for ECC → error-corrected reconstruction

Each level is a valid decode — no separate encoder/decoder pair, just
different depths of the stride-aware reader on the same container.

**Consequences (advantages over 3-channel):**

- Storage: 1 container (16384 bits / 2 KB), not 3 separate fields.
- Halo density: ~0.3% → ~0.6% signal (2× utilization).
- Decoder: one stride-aware reader, not 3 parallel readers.
- Matches existing bgz17 workspace design (family-zipper was the
  intended completion).

**Implementation path:**

1. `bgz17::zipper_encode(row)` — extract phase stream (existing)
   + magnitude stream (new, at φ² stride) → pack into 16384-bit
   container.
2. `bgz17::zipper_decode(container, level)` — stride-aware reader;
   `level` = {Phase, PhaseAndMag, Full}.
3. Wire `ZipperCodec` as `CodecCandidate` in `codec_rnd_bench.rs`.
   Measure ICC_3_1 at each truncation level against Qwen3 q_proj.
4. Gate behind `lab` feature until ICC gates pass.

**Predicted gate:**

- Zipper phase-only (Base17 equivalent): ~same as current Base17
  ICC 0.024 on q_proj (it's the same encoding, just re-addressed).
- Zipper phase+mag: hopefully > 0.3 — if magnitude stream carries
  independent discriminative info vs phase alone, the blend doesn't
  destroy signal (unlike the fractal-magnitude blend that produced
  ICC −0.49). Key test: magnitude stream bits must correlate with
  ground truth differences, not halo noise.

If zipper phase+mag achieves ICC ≥ 0.8 on q_proj at 2 KB/row → near-
lossless codec. If ~0.3-0.5 → useful hybrid. If ≤ 0.1 → the halo
positions also lack per-row discrimination and the "magnitude in halo"
hypothesis fails empirically (which would be a third negative,
narrowing the codec design space further).

Cross-ref: `.claude/knowledge/phi-spiral-reconstruction.md`
§ "family zipper". EPIPHANIES 2026-04-19 fractal-leaf NEGATIVE
entries. IDEAS 2026-04-19 "Fractal round-trip codec" (superseded by
this — single-container zipper is cheaper than triple-channel).
bgz17 crate as the substrate.

---

## 2026-05-05 — Future-work items extracted from PRs #244–#335

> Items below are ONLY those the PR author EXPLICITLY named as future work, "could do", "follow-up", "next PR", or "out of scope". No inference. Each item cites the PR.

---

### IDEA-B1-HARDWARE-BACKENDS — AMX/MKL hardware backends for sigma_propagation (PR #322)

**Status:** Open 2026-05-05
**Priority:** P3
**Source:** PR #322 explicit "What this PR does NOT do"
**Author's words:** "No hardware backends (AMX/MKL via ndarray #119/#121). That's B1.5 follow-up."

---

### IDEA-PILLAR7-ALPHA-ACCUMULATION — Pillar 7: Front-to-Back α-accumulation with Early-Termination (PRs #289, #291)

**Status:** Open 2026-05-05 (partially implemented as B5 in PR #324)
**Priority:** P2
**Source:** PR #289 (Pillar 6 out-of-scope section), PR #291 (idea journal)
**Author's words (PR #289):** "Pillar 7: Front-to-Back α-Akkumulation mit Early-Termination — direkte Anwendung von Pillar 6 + Pillar 5+ auf HHTL-Cascade-Beschleunigung. 60-90% Compute-Ersparnis."
**Note:** PR #324 shipped AlphaFrontToBack MergeMode (B5); Pillar 7 proof-in-code still deferred.

---

### IDEA-PILLAR8-ADAPTIVE-DENSIFICATION — Pillar 8: Adaptive Densification for online Σ-codebook learning (PRs #288, #291)

**Status:** Open 2026-05-05
**Priority:** P2
**Source:** PR #291 (idea journal), PR #288 (sigma codebook probe)
**Author's words (PR #291):** "Pillar 8 — Adaptive Densification für Online-Codebook-Lernen. Wendet an: KS Theorem 1 + sigma_codebook_probe (#288, R²=0.9949). Codebook wird selbst-verbessernd ohne Container-Wachstum. Split/Prune-Mechanik, ~250-300 Zeilen. Risiko: Split/Prune-Heuristik könnte oszillieren."

---

### IDEA-PILLAR9-SH-THINKING-MANIFOLD — Pillar 9: SH-coefficients as continuous Thinking-Style manifold (PR #291)

**Status:** Open 2026-05-05 — HOLD until explicit architecture decision (touches production code)
**Priority:** P3
**Source:** PR #291 (idea journal), PR #292 (TOUCHES PRODUCTION CODE tag added)
**Author's words (PR #291):** "Pillar 9 — SH-Koeffizienten als kontinuierliche Thinking-Style-Achse. Wendet an: Düker-Zoubouloglou Hilbert-Raum CLT. Substrat-Impact: kontinuierliche Thinking-Style-Mannigfaltigkeit statt kategorial. Berührt produktiven Code (`learning::cognitive_styles`) — braucht explizites Go-Ahead VOR Implementierung."

---

### IDEA-SAFETENSOR-STREAMING — Safetensor streaming as n-dimensional meaning accumulation (PR #290)

**Status:** Open 2026-05-05
**Priority:** P2
**Source:** PR #290 (idea journal)
**Author's words:** "Modelle (1B–70B params) Tile-für-Tile durch die Pipeline streamen statt vollständig laden. Pro Tile: Hadamard-rotieren, Σ extrahieren, EWA-Sandwich propagieren, in SchemaSidecar Block 14/15 akkumulieren. 7B-Modell ≈ 3.8 min Streaming-Zeit."

---

### IDEA-FRACTAL-CODEC — Family-Bounds as global fractal coding/decoding (PR #290)

**Status:** Open 2026-05-05 — CONJECTURE, requires diagnostic probe first
**Priority:** P3
**Source:** PR #290 (idea journal), PR #292 (CONJECTURE tag)
**Author's words:** "Das gesamte Substrat wird on-demand fraktal dekodiert statt vollständig materialisiert. Voraussetzung: globale Selbst-Ähnlichkeit der family bounds. Status: spekulativ. Globale Fraktalität ist eine Hypothese, kein gemessener Fakt."

---

### IDEA-INVERTED-PYRAMID-AWARENESS — Inverted-pyramid awareness streaming via CausalEdge64 (PR #299)

**Status:** Open 2026-05-05
**Priority:** P2
**Source:** PR #299 (replacement IDEAS.md entry after revert)
**Author's words:** "Open: inverted-pyramid awareness streaming via CausalEdge64 durch SPO+COCA→CAM_PQ pipeline."

---

### IDEA-CAUSAL-EDGE-TENSOR-SIDECAR — CausalEdgeTensor as 9-byte sidecar (CausalEdge64 + 1 byte Σ index) (PR #288)

**Status:** Open 2026-05-05
**Priority:** P2
**Source:** PR #288 (sigma codebook probe conclusion)
**Author's words:** "Mit diesem Probe-Resultat kann jetzt `CausalEdgeTensor`-Variante als 9-Byte-Sidecar (`CausalEdge64` + 1 Byte Σ-Codebook-Index) entworfen werden, ODER äquivalent über Schemasidecar Block 14/15. Caller-Wahl, beide architektonisch tragbar."

---

### IDEA-PILLAR5PP-OPERATOR-G — Pillar 5++ with Hermite rank ≥ 2 (operator G ≠ identity) (PR #287)

**Status:** Open 2026-05-05
**Priority:** P3
**Source:** PR #287 (out-of-scope section)
**Author's words:** "Operator G ≠ identity (Hermite rank ≥ 2) — kann als Erweiterungs-Test in einem späteren PR."

---

### IDEA-PROPAGATE-HOLOGRAPH-RESONANCE — propagate() in holograph::resonance (Gauss-convolution operator) (PRs #286, #287, #289)

**Status:** Open 2026-05-05
**Priority:** P2
**Source:** PRs #286, #287, #289 (each names this as out-of-scope)
**Author's words (PR #289):** "`propagate()` in `holograph::resonance` — orthogonal zur Encoding-Frage; wartet auf Architektur-Entscheidung."

---

### IDEA-ASYNC-PIPELINE-DAG — Async fan-out executor for PipelineDag (PR #300)

**Status:** Open 2026-05-05
**Priority:** P2
**Source:** PR #300
**Author's words:** "Synchronous-only executor; async fan-out is an explicit follow-up (documented in module doc)."

---

### IDEA-POLICY-HASH-UDF — policy_hash_v1 UDF registration (PR #301)

**Status:** Open 2026-05-05
**Priority:** P2
**Source:** PR #301
**Author's words:** "`NotYetWiredHashUdf` binds at plan time, returns `NotImplemented('policy_hash_v1 UDF not yet registered')` at execute. Plans build; execution fails loud."

---

### IDEA-TRANSCODE-GEO-FILE-IMAGE — Geo/File/Image typed reconstruction in triples_to_batch (PR #316)

**Status:** Open 2026-05-05
**Priority:** P3
**Source:** PR #316
**Author's words:** "`Geo` / `File` / `Image` typed reconstruction — round-4 candidates (collapse to `Utf8` today)."

---

### IDEA-TRANSCODE-ASYNC-RESOLVER — Async resolver for triples_to_batch_with_resolver (PR #316)

**Status:** Open 2026-05-05
**Priority:** P3
**Source:** PR #316
**Author's words:** "Async resolver — round-5 (for resolvers that hit a remote store)."

---

### IDEA-PILLAR5PLUS-HIGHER-DIM-SPD — Higher-dim SPD (3×3, n×n) for Pillar 6 logic (PR #289)

**Status:** Open 2026-05-05
**Priority:** P3
**Source:** PR #289
**Author's words:** "Higher-dim SPD (3×3, n×n) — Pillar 6 Logik erweitert sich monoton."

---

### IDEA-FMT-TIER-B-STANDALONE — Per-crate rustfmt.toml overrides + mass-reformat (PR #329)

**Status:** Open 2026-05-05
**Priority:** P3
**Source:** PR #329 (workspace-wide audit)
**Author's words:** "Path A (low): Add per-crate `rustfmt.toml` overrides where authors want one-line accessors / table-aligned literals [...] and then run `cargo fmt --write` per crate. Lets the author preferences coexist with `cargo fmt`. Path B (high): Decide on one canonical style for the whole repo, mass-rewrite, and add `cargo fmt --check` to CI for every crate. [...] Both should be a maintainer / `truth-architect` decision, not an autonomous agent's."

