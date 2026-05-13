# PR-F1 Spec: thinking-engine → UnifiedBridge Wiring

> **Sprint:** sprint-log-5-6 / S6-W7
> **Worker:** W9
> **Target crate:** `crates/thinking-engine/` (consumer) + `crates/lance-graph-callcenter/src/unified_bridge.rs` (target surface)
> **Status:** SPEC — not implementation. v1.
> **LOC estimate:** ~316 LOC new + ~40 LOC edits
> **Constraint:** Non-destructive — pure cognitive ops (encoding, distance, qualia computation) stay untouched; UnifiedBridge gate is added only on ops that cross a tenant boundary.

---

## 0. One-sentence thesis

`thinking-engine` today is a callable surface operating in isolation; PR-F1 adds a `CognitiveBridgeGate` trait injection point so any cognitive op that reads or writes cross-tenant data (shared retrieval index, persona switch touching another tenant's qualia corpus, cross-tenant reranker call) is intercepted, authorized through `UnifiedBridge`, and audited with a `UnifiedAuditEvent` before the op proceeds.

---

## 1. Boundary analysis — which ops cross a UnifiedBridge boundary?

### 1.1 Ops that STAY PURE (no UnifiedBridge needed)

These are intra-tenant or stateless-math ops. No auth or audit required.

| Op / file | Why pure |
|---|---|
| `engine.rs` / `signed_engine.rs` / `bf16_engine.rs` — encode sentence to embedding | Reads only the caller's own input text + model weights (shared, read-only, not tenant data) |
| `l4_bridge.rs` — XOR bind peaks to L4Experience | Pure math on local distance table; no cross-tenant index |
| `bridge.rs` — spiral address to table index + coarse distance | Pure geometry; no retrieval |
| `qualia.rs` — `Qualia17D::from_convergence()` | Pure math from convergence snapshots; all local |
| `cognitive_stack.rs` — `ThinkingStyle::params()` | Static config lookup; no tenant context |
| `cronbach.rs` / `ground_truth.rs` / `reencode_safety.rs` | Calibration math; local only |
| `prime_fingerprint.rs` / `spiral_segment.rs` | VSA perturbation; no retrieval |
| `pooling.rs` / `composite_engine.rs` / `dual_engine.rs` | Intra-engine composition; local |

### 1.2 Ops that CROSS a UnifiedBridge boundary (require auth + audit)

#### Category A — Cross-tenant retrieval via sensor lens

| File | Op | Why cross-tenant |
|---|---|---|
| `jina_lens.rs` | Encode + nearest-neighbor lookup in shared embedding index | Shared Jina v5 index may contain embeddings from multiple tenants; querying with one tenant's data can surface another tenant's documents |
| `bge_m3_lens.rs` | Same pattern for BGE-M3 multilingual index | Same shared-index concern |
| `reranker_lens.rs` | Cross-encoder reranking against candidate pool | Candidates may span tenants if pool was assembled cross-tenant |

Gate rule: `lens.retrieve(query_fp, k)` must pass through `CognitiveBridgeGate::authorize_retrieval(tenant_id, entity_type, depth)` before the ANN call. If no gate is configured, falls through via `PassthroughGate` (intra-tenant default).

#### Category B — Persona switch that reads another tenant's qualia corpus

| File | Op | Why cross-tenant |
|---|---|---|
| `persona.rs` | `PersonaProfile::switch_mode(PersonaMode)` when mode references a shared archetype corpus | The archetype registry (`agi_lego_party_canonical.yaml`) is shared; switching persona mode that loads a different tenant's archetype slot is a cross-tenant read |
| `cognitive_stack.rs` | `CognitiveStack::set_style(ThinkingStyle)` when style YAML is loaded from a shared registry | Same concern once YAML loading moves online |

Gate rule: `CognitiveBridgeGate::authorize_persona_switch(tenant_id, persona_mode)` fires before the switch commits. In the current state where archetypes are `'static` YAML, this gate is a no-op via `PassthroughGate`; the injection point exists so a future online registry can use it.

#### Category C — Cross-tenant cognitive_stack coordination (L6 delegation, L8 integration)

When L6 delegates to multiple lenses scoped to different tenants, the integration result at L8 aggregates cross-tenant evidence. Gate rule: `CognitiveBridgeGate::authorize_cognitive_op(tenant_id, op_kind: CognitiveOpKind)` fires at L6 fan-out and L8 integration.

---

## 2. BindSpace columns affected

Per CLAUDE.md AGI-as-glove rule: AGI = (topic, angle, thinking, planner) = the four `BindSpace` columns.

| BindSpace column | How PR-F1 affects it |
|---|---|
| **`FingerprintColumns` (topic / angle / content)** | Cross-tenant retrieval (Category A) writes the result into `content` (retrieved fingerprint) and `topic` (query fingerprint). Auth gate fires before write. No column shape change. |
| **`FingerprintColumns.sigma`** (u8 Sigma-codebook index, B2/PR#323) | Retrieval path through jina/bge/reranker sensors writes the Sigma index (each retrieved document carries a Sigma index from shared codebook per Pillar 6 / R=0.9949 at k=256). Auth gate fires before sigma write, ensuring cross-tenant Sigma propagation is audited. No column shape change. |
| **`QualiaColumn`** (18xf32 per row) | Persona switch (Category B) rewrites the qualia vector: PersonaMode::Work -> guardian archetype qualia, Personal -> catalyst archetype qualia. Auth gate fires before qualia write. No column shape change. |
| **`MetaColumn`** (MetaWord packed u32, thinking-style bits) | `set_style(ThinkingStyle)` writes style bits into MetaWord. When style YAML is loaded from a shared registry this is a cross-tenant op. Auth gate fires before MetaWord commit. No column shape change. |
| **`EdgeColumn`** (CausalEdge64) | L6/L8 coordination (Category C) may route edges across tenants in a future multi-tenant delegation graph. Gate injection point present; no column shape change today. |

Column shape policy: PR-F1 does NOT add new BindSpace columns. The four existing planes + sigma are sufficient. The gate adds authorization logic at write time, not new storage.

---

## 3. Trait / API shape on the thinking-engine side

### 3.1 `CognitiveBridgeGate` trait (new file: `src/bridge_gate.rs`, ~70 LOC)

```rust
/// Injection point for cross-tenant authorization in the cognitive pipeline.
/// Production impl: UnifiedBridgeGate<B> (in lance-graph-callcenter).
/// Default impl: PassthroughGate (in thinking-engine) — unconditionally allows.
/// All methods synchronous. The cognitive pipeline is not async.
pub trait CognitiveBridgeGate: Send + Sync {
    fn authorize_retrieval(
        &self,
        tenant_id: u32,
        entity_type: &str,
        depth: u8,
    ) -> CognitiveAuthResult;

    fn authorize_persona_switch(
        &self,
        tenant_id: u32,
        mode: u8,  // PersonaMode ordinal — avoids coupling to thinking-engine enum
    ) -> CognitiveAuthResult;

    fn authorize_cognitive_op(
        &self,
        tenant_id: u32,
        op_kind: CognitiveOpKind,
    ) -> CognitiveAuthResult;
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CognitiveOpKind {
    L6Delegation   = 1,
    L8Integration  = 2,
    QualiaWrite    = 3,
    MetaWordCommit = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CognitiveAuthResult { Allow, Deny, Escalate }

/// Default gate — unconditionally allows. Zero overhead.
pub struct PassthroughGate;
impl CognitiveBridgeGate for PassthroughGate {
    fn authorize_retrieval(&self, _: u32, _: &str, _: u8) -> CognitiveAuthResult { CognitiveAuthResult::Allow }
    fn authorize_persona_switch(&self, _: u32, _: u8) -> CognitiveAuthResult { CognitiveAuthResult::Allow }
    fn authorize_cognitive_op(&self, _: u32, _: CognitiveOpKind) -> CognitiveAuthResult { CognitiveAuthResult::Allow }
}
```

### 3.2 `UnifiedBridgeGate` (production, in callcenter)

New file `crates/lance-graph-callcenter/src/cognitive_bridge_gate.rs` (~80 LOC). Wraps `UnifiedBridge<B>`. Cross-tenant Chinese-wall check (mismatched TenantId -> Deny) fires before policy evaluation, consistent with UnifiedBridge section 3.8. Delegates to `authorize_read("Document"|"Persona")` and `authorize_act("CognitiveStack", action_str)`. Audit events emitted automatically by `UnifiedBridge::emit_audit()`.

`super_domain` on emitted events: comes from the `AuditChain` configured at `UnifiedBridgeGate` construction time. thinking-engine does not hard-code `super_domain` values.

### 3.3 Injection into sensors and persona

- `JinaLens`, `BgeM3Lens`, `RerankerLens`: add `gate: Arc<dyn CognitiveBridgeGate>` field, default `Arc::new(PassthroughGate)`. `retrieve()` calls `gate.authorize_retrieval(...)` before ANN call. `Deny` or `Escalate` returns `Err(CognitiveBridgeError::Denied)` without touching the shared index.
- `PersonaProfile::switch_mode()`: add `gate: &dyn CognitiveBridgeGate` param. Calls `authorize_persona_switch(...)` before committing.

---

## 4. Audit emission

The production `UnifiedBridgeGate` delegates to `UnifiedBridge::authorize_read()` / `authorize_act()`, which already emit `UnifiedAuditEvent` via the `AuditChain` (D-SDR-5, PR #364). No new audit path needed in thinking-engine.

| Op | Auth method called | Audit fired by |
|---|---|---|
| Cross-tenant retrieval (jina/bge/reranker) | `authorize_read(entity_type, depth)` | Existing `UnifiedBridge::emit_audit()` (D-SDR-5) |
| Persona switch | `authorize_read("Persona", PrefetchDepth::Detail)` | Same |
| L6/L8 cognitive op | `authorize_act("CognitiveStack", op_name)` | Same |

Coverage: Every cross-tenant retrieval/persona/coordination op emits 1 `UnifiedAuditEvent` with merkle-chained root. Pure math ops (encode, distance, qualia compute, l4 learn) emit zero.

---

## 5. LOC estimate

| File | Type | LOC |
|---|---|---|
| `crates/thinking-engine/src/bridge_gate.rs` | New | ~70 |
| `crates/thinking-engine/src/jina_lens.rs` | Edit | ~25 |
| `crates/thinking-engine/src/bge_m3_lens.rs` | Edit | ~20 |
| `crates/thinking-engine/src/reranker_lens.rs` | Edit | ~20 |
| `crates/thinking-engine/src/persona.rs` | Edit | ~15 |
| `crates/thinking-engine/src/cognitive_stack.rs` | Edit | ~25 |
| `crates/thinking-engine/src/lib.rs` | Edit | ~3 |
| `crates/lance-graph-callcenter/src/cognitive_bridge_gate.rs` | New | ~80 |
| `crates/lance-graph-callcenter/src/lib.rs` | Edit | ~3 |
| Tests (unit + integration) | New | ~55 |
| **TOTAL** | | **~316 LOC** |

Day-scale PR (2-3 hours). No math changes. No BindSpace column shape changes. No new contract types (CognitiveBridgeGate stays in thinking-engine until follow-up promotes to lance-graph-contract).

---

## 6. DELTA vs jc-pillars-runtime-wiring-v1.md + ERRATUM

### 6.1 Pillars governance-wrapped by PR-F1 (not new math, existing math gated)

- **Pillar 6 (EWA-sandwich Sigma-propagation, B1/B2/PR#322/PR#323):** `FingerprintColumns.sigma` is written during retrieval (Category A). PR-F1 ensures cross-tenant retrieval is gated before the sigma write, so cross-tenant Sigma propagation carries a merkle-chained `UnifiedAuditEvent`. Governance wrap around Pillar 6 wiring (B1/B2 shipped the math; PR-F1 adds the auth gate on the write path).

- **Pillar 5b (Pearl 2^3, L2/L3):** The shared embedding index is indexed under Pearl 2^3 masks in the BindSpace. Cross-tenant retrieval could return rows from a different Pearl-mask context. The gate prevents unauthorized cross-mask-context reads.

### 6.2 Pillars DEFERRED (not touched by PR-F1)

| Pillar | What's deferred | Why |
|---|---|---|
| **Pillar 1** (substrate, d>=10K bundle associativity) | CI gate promotion (P0) | Separate P0 deliverable |
| **Pillar 3** (phi-Weyl 144-verb collocation) | COCA-4096 predicate vocabulary replacing L2 regex | Lives in cognitive-shader-driver/convergence.rs |
| **Pillar 5** (Jirak Berry-Esseen) | Sup-error instrumentation on real traffic (P3) | PR-F1 wires auth; instrumentation is separate |
| **Pillar 6 math promotion** (full L3 traversal Sigma-propagation) | P3 of jc-pillars plan | B1/B2 shipped; P3 is the next step |
| **Pillars 2 + 4** (Cartan-Kuranishi + gamma+phi preconditioner) | Deferred per JC lib.rs | Unchanged |

### 6.3 Layer mapping (per ERRATUM)

The ERRATUM corrects layer attribution. PR-F1 operates at **L4** (thinking-styles + sensors + persona):
- **L1** (`nars_engine.rs`) — already correct three-plane Index regime; no change
- **L2** (`convergence.rs`) — bitmap / COCA-4096 concern; not this PR
- **L3** (`cognitive-shader-driver/bindspace.rs`) — receives gate-guarded sigma write
- **L4** (`thinking-engine` sensors + persona) — receives `CognitiveBridgeGate` injection

### 6.4 What PR-F1 adds to the plan

jc-pillars-runtime-wiring-v1.md focuses on L1-L3 math wiring across P0-P6. L4 auth was never explicitly assigned to any phase. PR-F1 is a new phase **P-auth** — the governance analog to the math phases P0-P6. The non-destructive principle applies: `PassthroughGate` default means no behavior change until an `UnifiedBridgeGate` is injected.

---

## 7. Risk register

| Risk | Mitigation |
|---|---|
| Circular dep: callcenter -> thinking-engine is OK; reverse is NOT | `CognitiveBridgeGate` trait lives in thinking-engine (no callcenter dep). `UnifiedBridgeGate` lives in callcenter. Direction: callcenter -> thinking-engine only. |
| PassthroughGate silently allows what should be denied | Contract test: `UnifiedBridgeGate` with mismatched `tenant_id` MUST return Deny before policy evaluation (Chinese-wall check §3.8) |
| Audit chain mutex contention at high retrieval throughput | Current: single `Mutex<AuditChain>` per `UnifiedBridge::emit_audit()` call. Acceptable for single-threaded cognitive sessions. Batched sink is future work. |
| `PrefetchDepth::from_u8()` missing from contract | Add `prefetch_from_u8(u8) -> PrefetchDepth` helper in `cognitive_bridge_gate.rs` via match arm. ~5 LOC. |

---

## 8. Acceptance criteria

- [ ] `cargo test -p thinking-engine` passes with `PassthroughGate` default (no behavior change on any existing test)
- [ ] `cargo test -p lance-graph-callcenter` passes with new `cognitive_bridge_gate` tests
- [ ] Integration test: cross-tenant retrieval (TenantId mismatch) -> `Err(CognitiveBridgeError::Denied)` + 1 `UnifiedAuditEvent` with `decision = Deny`
- [ ] Integration test: same-tenant retrieval -> `Ok(fingerprint)` + 1 `UnifiedAuditEvent` with `decision = Allow`
- [ ] Pure math ops (encode, qualia compute, l4 learn) -> zero `UnifiedAuditEvent`s emitted
- [ ] No `lance-graph-callcenter` dep in `crates/thinking-engine/Cargo.toml`
- [ ] No new BindSpace column shapes (gate logic only on existing write paths)
- [ ] `super_domain` on emitted events matches the `AuditChain` configured at `UnifiedBridgeGate` construction

---

## 9. Cross-references

- `crates/lance-graph-callcenter/src/unified_bridge.rs` — target surface (D-SDR-5, PR #364)
- `crates/lance-graph-callcenter/src/unified_audit.rs` — `UnifiedAuditEvent`, `AuditChain`, `AuthOp`, `AuthDecision`
- `crates/lance-graph-callcenter/src/super_domain.rs` — `SuperDomain` enum (Healthcare / WorkOrderBilling / OSINT / ...)
- `crates/cognitive-shader-driver/src/bindspace.rs` — `FingerprintColumns.sigma` (B2, PR #323); `QualiaColumn`; `MetaColumn`; `EdgeColumn`
- `crates/lance-graph-contract/src/sigma_propagation.rs` — `Spd2`, `ewa_sandwich` (B1, PR #322)
- `crates/thinking-engine/src/{jina_lens,bge_m3_lens,reranker_lens,persona,cognitive_stack}.rs` — surfaces receiving gate injection
- `.claude/plans/jc-pillars-runtime-wiring-v1.md` + ERRATUM — canonical JC pillar wiring plan (L1-L4 layer attribution corrected by ERRATUM)
- `.claude/plans/oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` — BindSpace column doctrine
- `.claude/board/LATEST_STATE.md` — UnifiedBridge D-SDR-5 + FingerprintColumns.sigma inventory
- `.claude/plans/super-domain-rbac-tenancy-v1.md` — SuperDomain + TenantId + AuditChain spec
