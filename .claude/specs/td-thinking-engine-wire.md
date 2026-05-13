# TD-THINKING-ENGINE-UNWIRED-1: Wire `thinking-engine` into `UnifiedBridge`

> **Status:** Spec (open — no PR yet)
> **Priority:** P1 — architectural debt; every downstream D-SDR pays clean-room cost until closed
> **Date:** 2026-05-13
> **Author:** W6 (sprint-log-4)
> **Deliverable type:** Implementation spec (~12 KB)
> **Estimated PR size:** ~300 LOC + 5 integration tests (single PR `cognition-bridge`)
> **Branch:** `claude/lance-datafusion-integration-gv0BF`

---

## 1. Problem statement

`crates/thinking-engine/` ships 48 modules / 16 211 LOC / 582 KB of Rust covering
precision-tier engines, cognition, calibration, and bridge algebra. It is indexed in
`CLAUDE.md § Thinking Engine` and cited by six plans but consumed by **zero
callcenter-side code**.

The §16–§19 spec deliverables (D-SDR-13 / D-SDR-15 / D-SDR-17) were scaffolded as new
clean-room types rather than composed onto the existing engine. This creates a growing
duplication surface: every downstream D-SDR that ships clean-room widens it further and
creates a future dedup pass.

**The debt is the wiring gap, not the substrate.** One ~300 LOC bridge module collapses
three independent D-SDR scaffolds into composed thinking-engine delegates.

---

## 2. Compose-not-rebuild diff

For each of D-SDR-13 / D-SDR-15 / D-SDR-17 the table maps the clean-room scaffolding
to the equivalent thinking-engine type that should back it.

| D-SDR | Clean-room concept | thinking-engine type | Module |
|---|---|---|---|
| **D-SDR-13** HKDF per-super-domain key derivation | `DriftSignal` — tenant isolation primitive that needed a per-SD identity vector to seed HKDF | `thinking_engine::ghosts::GhostField` + `Ghost { atom, intensity, created_at }` | `crates/thinking-engine/src/ghosts.rs` |
| **D-SDR-13** cont. | Role-keyed identity fingerprint for HKDF context | `thinking_engine::role_tables::LayerTables` (gate-modulated BF16 distance table rows as role projection vectors) | `crates/thinking-engine/src/role_tables.rs` |
| **D-SDR-13** cont. | Actor persona snapshot alongside HKDF seed | `thinking_engine::persona::CognitiveBaseline` + `PersonaMode` | `crates/thinking-engine/src/persona.rs` |
| **D-SDR-15** DifferentialPrivacy role | `QualiaSignal` — epsilon-bounded noise scaffold needed a calibrated noise magnitude source | `thinking_engine::qualia::Qualia17D` (`tension` dim 2 = 1 − convergence_speed; `entropy` dim 8 = Shannon entropy of energy) — both are calibrated noise-magnitude proxies | `crates/thinking-engine/src/qualia.rs` |
| **D-SDR-15** cont. | k-anonymity floor check via calibration metric | `thinking_engine::cronbach::CronbachAlpha` (internal-consistency alpha as k-anonymity proxy: alpha >= 0.7 means k >= threshold) | `crates/thinking-engine/src/cronbach.rs` |
| **D-SDR-15** cont. | epsilon-bounded noise injection into query results | `thinking_engine::contrastive_learner::ContrastiveLearner::update_pair` (alpha-scaled cosine delta as controllable noise magnitude) | `crates/thinking-engine/src/contrastive_learner.rs` |
| **D-SDR-17** Hard-lock partner matrix | `HomeostasisGate` — static crypto barrier between partner tenants | `thinking_engine::cognitive_stack::GateState` (`Flow` / `Hold` / `Block` thresholds on SD of candidate scores) as the homeostasis enforcement gate | `crates/thinking-engine/src/cognitive_stack.rs` |
| **D-SDR-17** cont. | OSINT-side projection for Healthcare↔OSINT barrier | `thinking_engine::osint_bridge::OsintThinkingBridge` — wraps codebook + cosine table; `think()` / `similarity()` already produce cross-tenant projection scores | `crates/thinking-engine/src/osint_bridge.rs` |
| **D-SDR-17** cont. | Drift observation to detect barrier violation | `thinking_engine::ghosts::Ghost` — a ghost whose `intensity` decays below threshold = no recent barrier-crossing signal (implicit GhostObservation pattern) | `crates/thinking-engine/src/ghosts.rs` |

**Net savings:** ~310 LOC + 13 tests scaffolded clean-room → ~140 LOC + 7 tests composed
against thinking-engine primitives. Approximately 45% LOC reduction per TECH_DEBT.md
Payoff estimate.

---

## 3. `UnifiedBridge<T>::cognition()` — new field and method

### 3.1 Field addition

```rust
// crates/lance-graph-callcenter/src/unified_bridge.rs

use thinking_engine::cognitive_stack::{GateState, MetaCognition, ThinkingStyle};

pub struct UnifiedBridge<B: NamespaceBridge> {
    bridge:      Arc<B>,
    policy:      Arc<dyn Policy>,
    tenant:      TenantId,
    audit_sink:  Arc<dyn UnifiedAuditSink>,
    audit_chain: Mutex<AuditChain>,
    /// NEW — optional cognitive stack. None = noop (backward compatible).
    cognition:   Option<Arc<Mutex<CognitionHandle>>>,
}

/// Thin handle wrapping the thinking-engine types needed by UnifiedBridge.
pub struct CognitionHandle {
    pub stack:       thinking_engine::cognitive_stack::MetaCognition,
    pub ghost_field: thinking_engine::ghosts::GhostField,
    pub persona:     thinking_engine::persona::PersonaMode,
}
```

### 3.2 Builder method

```rust
impl<B: NamespaceBridge> UnifiedBridge<B> {
    /// Attach a cognitive stack. When set, every authorize_* call funnels
    /// through `cognition_handle.stack.record()` before the policy decision.
    /// Noop (default) = no cognitive overhead.
    pub fn with_cognition(mut self, handle: Arc<Mutex<CognitionHandle>>) -> Self {
        self.cognition = Some(handle);
        self
    }

    /// Return a reference to the cognitive stack if wired.
    pub fn cognition(&self) -> Option<Arc<Mutex<CognitionHandle>>> {
        self.cognition.clone()
    }
}
```

### 3.3 Hot-path integration

Every `authorize_*` method gains a pre-decision cognitive observation:

```rust
pub fn authorize_read(
    &self,
    public_name: &str,
    depth: PrefetchDepth,
) -> Result<EntityRef, BridgeError> {
    // existing resolve + policy::evaluate logic unchanged
    let decision = self.inner_authorize(public_name, depth, Operation::Read)?;

    // NEW: cognitive observation (zero-cost when cognition = None)
    if let Some(ref cog) = self.cognition {
        let predicted_confidence = decision_to_confidence(&decision);
        let was_correct = matches!(decision, AccessDecision::Allow(_));
        let mut handle = cog.lock().unwrap();
        handle.stack.record(predicted_confidence, was_correct);
        // ghost tick for decay; imprint on next explicit resonance cycle
        handle.ghost_field.tick();
    }

    self.emit_audit(public_name, Operation::Read, &decision, depth);
    decision_to_result(decision, public_name)
}
```

`authorize_write` and `authorize_act` follow the same pattern. The `UnifiedAuditEvent`
gains an optional `awareness_root: Option<u64>` = FNV-1a of
`MetaCognition::brier_score().to_bits()` when cognition is wired, otherwise `None`.
This is **backward compatible** — consumers that do not call `with_cognition` see no
change.

---

## 4. Hook surfaces — three integration points

### 4.1 `tensor_bridge::EmbeddingOutput` → SpoQuad ingest

**File:** `crates/thinking-engine/src/tensor_bridge.rs`

`EmbeddingOutput` is the unified embedding type across F32 / I8 / U8 / Candle. For
SpoQuad ingest from an OWL ontology entity (e.g. FMA heart node), the hook is:

```rust
// In the cognition_bridge module:
fn spo_quad_from_embedding(
    output: &EmbeddingOutput,
    subject_uri: &str,
    predicate: &str,
) -> SpoQuad {
    let vector: Vec<f32> = output.to_f32();
    // project into VSA role-indexed space via role_tables::gate_modulate
    // then wrap as SpoQuad
    SpoQuad { subject: subject_uri.into(), predicate: predicate.into(),
               object_vector: vector }
}
```

The hook surface is `EmbeddingOutput::to_f32()` — already public, no new API needed.

### 4.2 `cognitive_stack::MetaCognition::record` → audit-event drift

**File:** `crates/thinking-engine/src/cognitive_stack.rs`

`MetaCognition` tracks Brier score calibration error across authorize calls. The drift
signal is: `brier_score() > 0.25` after N calls => insert a `UnifiedAuditEvent` with
`auth_decision = AuthDecision::Escalate` and reason `"cognition_drift"`.

```rust
// In unified_bridge.rs after each authorize_* call:
if let Some(ref cog) = self.cognition {
    let handle = cog.lock().unwrap();
    if handle.stack.brier_score() > 0.25 && handle.stack.n_predictions() > 10 {
        self.emit_drift_event("cognition_drift", handle.stack.brier_score());
    }
}
```

The mutation point is `MetaCognition::record(predicted_confidence, was_correct)`. The
drift event feeds into the audit chain (merkle-linked) so it is auditable downstream.

### 4.3 `composite_engine::CompositeEngine` → Cypher-cell intent classification (FMA demo)

**File:** `crates/thinking-engine/src/composite_engine.rs`

`CompositeEngine` runs multiple lenses (Jina v3 BF16 / Reranker BF16 / BGE-M3 u8) and
superimposes results. For Cypher-cell intent classification in the FMA demo, the hook
is:

```rust
// In the cognition_bridge module, called from the FMA demo path:
fn classify_cypher_intent(
    engine: &CompositeEngine,
    distance_table: &[u8],
    resonance_input: &ResonanceDto,
) -> IntentClass {
    let result: CompositeResult = engine.compose(resonance_input, distance_table);
    let (top_atom, score, n_lenses) = result.superposed.first()
        .copied()
        .unwrap_or((0, 0.0, 0));
    IntentClass { atom: top_atom, confidence: score, lens_agreement: n_lenses }
}
```

`CompositeEngine::compose` already exists (internal call); expose it as a single public
method. `superposed` is sorted by composite score descending, so `first()` is the intent
atom. `centroid_labels.rs` maps the atom index to a Cypher template string.

---

## 5. `lance-graph-cognition-bridge` type-bridge crate

### 5.1 Rationale

Consumers of `UnifiedBridge` (medcare-rs, smb-office-rs) must not take a direct
`thinking-engine` dep: the engine pulls in ONNX / candle feature flags that inflate
compile time and binary size. The bridge crate re-exports the minimal surface at a
stable ABI boundary.

Alternative: extend `lance-graph-contract` with a `cognition` feature. Rejected —
`lance-graph-contract` is the zero-dep DTO layer per `CONSUMER_WIRING_INSTRUCTIONS.md`.
Adding a thinking-engine dep would violate that invariant.

### 5.2 Proposed crate structure

```
crates/lance-graph-cognition-bridge/
  Cargo.toml          — dep: thinking-engine (default-features = false, features = ["cognition"])
  src/
    lib.rs            — re-exports and CognitionHandle
    role_projection.rs — RoleProjection::for_role(&str) -> [f32; 16_384]
    actor_persona.rs   — ActorPersona::from_jwt(claims) -> PersonaCard
    awareness_frame.rs — AwarenessFrame::project(decision, persona) -> AwarenessDto
    drift_signal.rs    — DriftSignal (wraps GhostField + brier_score threshold)
    dp_role.rs         — DpRoleGuard (wraps ContrastiveLearner + CronbachAlpha)
    hard_lock.rs       — HardLockMatrix (wraps OsintThinkingBridge + partner table)
```

### 5.3 Cargo.toml sketch

```toml
[package]
name    = "lance-graph-cognition-bridge"
version = "0.1.0"
edition = "2021"

[dependencies]
thinking-engine      = { path = "../thinking-engine", default-features = false,
                         features = ["cognition"] }
lance-graph-contract = { path = "../lance-graph-contract" }

[features]
default     = []
calibration = ["thinking-engine/calibration"]
```

### 5.4 Required Cargo.toml edit in thinking-engine

`thinking-engine` currently has no `cognition` feature gate. The cognition-bridge PR
must add one to exclude ONNX / candle from the default compile path:

```toml
[features]
default    = ["cognition"]
cognition  = []            # cognitive_stack, ghosts, persona, qualia, world_model, role_tables
calibration = ["dep:candle-core"]
onnx        = ["dep:ort"]
```

This is a prerequisite for Step 0 in the delivery sequence (§9).

---

## 6. Risk register — `CognitiveStack` mutability + Send/Sync

### 6.1 Mutability hazard

`MetaCognition` and `GhostField` both require `&mut self` on mutating calls.
`CompositeEngine` takes `&self` (immutable after build).

`UnifiedBridge<B>` wraps the handle in `Arc<Mutex<CognitionHandle>>`. This is correct
but introduces lock contention on the hot path. **Mitigation:** use `parking_lot::Mutex`
(already used by several lance-graph crates). Alternatively, make `CognitionHandle`
thread-local and only flush Brier score to the shared handle on audit emission
(amortised over N calls). The per-thread approach is preferred for low-latency
authorize paths.

### 6.2 Send + Sync requirements

| Type | Send | Sync | Wrapper |
|---|---|---|---|
| `MetaCognition` | yes (all fields f32 / u64) | yes | `Arc<T>` sufficient |
| `GhostField` | yes | no (interior mutation) | `Arc<Mutex<T>>` required |
| `CompositeEngine` | yes | yes (read-only after build) | `Arc<T>` sufficient |
| `OsintThinkingBridge` | yes | yes | `Arc<T>` sufficient |
| `ContrastiveLearner` | yes | no (update_pair mut) | `Arc<Mutex<T>>` per guard |

`CognitionHandle` bundles `GhostField` + `MetaCognition`, so the whole handle is
`Send + !Sync`. `Arc<Mutex<CognitionHandle>>` is the correct wrapper.

### 6.3 Per-tenant instantiation requirement

Each `UnifiedBridge` instance is already per-tenant (carries `TenantId`). The cognitive
stack MUST follow the same boundary: **one `CognitionHandle` per `UnifiedBridge`
instance**. A shared `CognitionHandle` across tenants would allow Brier score leakage:
tenant A's authorization pattern trains the ghost field that biases tenant B's atom
weights, creating a timing side-channel.

Enforcement: `CognitionHandle` must NOT implement `Clone`. The builder
`with_cognition(Arc<Mutex<CognitionHandle>>)` forces the caller to construct one per
tenant. Add a crate-level doc comment explicitly warning against sharing handles across
`TenantId` boundaries.

---

## 7. Cross-flags

### 7.1 Cross-flag with W4 — Super-domain subcrate cascade (TD-SUPER-DOMAIN-SUBCRATES-1)

W4's spec proposes per-super-domain subcrates for medcare-analytics, medcare-bridge,
smb-bridge, hubspot/hiro/woa. Each subcrate gets its own `UnifiedBridge` instance.

**Coordination requirement:** each subcrate's `UnifiedBridge` call to `.with_cognition()`
must receive its own `CognitionHandle`. Provide `CognitionHandle::for_super_domain(sd:
SuperDomain) -> Self` as the canonical constructor, seeding `GhostField` decay rate and
`MetaCognition` calibration window from per-super-domain defaults:

- `SuperDomain::HealthcareAnalytics` — decay_rate: 0.95 (slow decay for compliance),
  Brier window: 20 calls
- `SuperDomain::WorkOrderBilling` — decay_rate: 0.85 (standard), Brier window: 10 calls
- `SuperDomain::SmallBusinessOffice` — decay_rate: 0.75 (fast decay), Brier window: 50
  calls

Shared state MUST NOT cross super-domain boundaries per §6.3 above.

### 7.2 Cross-flag with W11 — FMA heart-click smoke test

W11's FMA demo flow: user clicks heart node in q2 3D render → intent classifies to SPO
query → Lance returns anatomy subgraph. The classification step uses
`composite_engine::CompositeEngine` (§4.3 hook surface).

**Coordination requirement:** W11 must call
`lance_graph_cognition_bridge::classify_cypher_intent(composite, table, resonance)` —
not thinking-engine directly. The `CompositeResult::superposed` atom index maps to a
Cypher template via `centroid_labels.rs` + `codebook_index.rs` already in
thinking-engine.

W11 and W6 must agree on the `IntentClass` type shape. Proposed canonical form:

```rust
// In lance-graph-cognition-bridge::src/lib.rs (public export)
pub struct IntentClass {
    pub atom:             u16,          // superposed codebook atom
    pub confidence:       f32,          // composite score (0.0–1.0)
    pub lens_agreement:   usize,        // number of lenses that agree
    pub cypher_template:  &'static str, // from centroid_labels lookup
}
```

The `cypher_template` field is a key design decision for W11: it must cover the FMA
anatomy SPO query shape (`MATCH (n:AnatomyNode)-[r]->(m) WHERE n.fma_id = $id`).

---

## 8. Open questions

1. **Does `thinking-engine::contract_bridge` already expose the right shape?**
   `contract_bridge.rs` exports `CascadeConfig`, `FastBusDto`, and
   `contract_style_to_engine`. It is a contract → engine adapter, not an engine →
   callcenter adapter. The new `lance-graph-cognition-bridge` crate is the reverse
   direction. `contract_bridge` should not be extended to avoid a circular dependency
   (engine → callcenter → engine).

2. **Which modules belong in "Layer 2 role catalogue" per I-VSA-IDENTITIES?**
   `role_tables.rs`, `persona.rs`, `ghosts.rs`, and `cognitive_stack.rs` feel like
   Layer 2 (role-keyed identity fingerprints). `qualia.rs`, `world_model.rs`, and
   `composite_engine.rs` feel like Layer 3 (content-level reasoning). The cognition-
   bridge crate must expose only Layer-2 types in its public surface to avoid coupling
   the RBAC hot path to content-level reasoning. Needs architecture-owner confirmation.

3. **Does the cognitive-shader-driver runtime expect thinking-engine on the internal
   SoA side of the BBB?**
   If yes, `CognitionHandle` must mediate through the BBB seam and not call
   thinking-engine directly from `UnifiedBridge`. This would require an additional
   indirection layer and changes the PR scope. Needs confirmation before coding begins.

4. **Can `GhostField::tick()` be called without a preceding `imprint()`?**
   The current `GhostField` API creates ghosts via `imprint(resonant_atoms, ...)` before
   `tick()` decays them. The `authorize_*` hot path does not always have `resonant_atoms`
   available. Is tick()-only (pure decay) safe, or does it corrupt field state? This
   needs a unit test in the cognition-bridge PR (test: construct GhostField, call
   tick() N times without imprint, assert ghosts Vec is empty and cycle counter
   advances).

5. **What is the correct N threshold for Brier-score drift emission?**
   The spec proposes N > 10. Ten is arbitrary. The correct value depends on the
   calibration window size and the false-positive tolerance for `cognition_drift` audit
   events per super-domain. Should this be a `CognitionHandle::for_super_domain`
   parameter (different N per SD) or a single workspace constant? A Monte Carlo
   calibration run against the medcare-rs test fixture would yield the right N; absent
   that, default to N = 20 with a `#[doc(hidden)]` constant that super-domain
   constructors can override.

---

## 9. Delivery sequence

| Step | What | Crate(s) touched |
|---|---|---|
| 0 | Add `cognition` feature gate to thinking-engine `Cargo.toml` | `thinking-engine` |
| 1 | Create `crates/lance-graph-cognition-bridge/` with 6 modules per §5.2 | new crate |
| 2 | Add `CognitionHandle` struct + `with_cognition()` builder to `unified_bridge.rs` | `lance-graph-callcenter` |
| 3 | Wire `MetaCognition::record()` into all three `authorize_*` hot paths | `lance-graph-callcenter` |
| 4 | Wire drift emission on `brier_score() > threshold` | `lance-graph-callcenter` |
| 5 | Expose `CompositeEngine::compose()` as single public method | `thinking-engine` |
| 6 | 5 integration tests: (a) noop default backward compat, (b) cognition wired + allow records correctly, (c) cognition wired + deny emits drift event, (d) two TenantId instances share no CognitionHandle state, (e) Brier score threshold triggers escalate audit | `lance-graph-cognition-bridge` |
| — | Update TECH_DEBT.md D-SDR-13 / D-SDR-15 / D-SDR-17 rows to cite composed modules | `.claude/board/TECH_DEBT.md` |

Steps 0–6 ship as a single PR named `cognition-bridge` (~300 LOC production + ~120 LOC
tests). Closes D-SDR-13 + D-SDR-15 + D-SDR-17 as composed-not-scaffolded.

---

## 10. References

- `CLAUDE.md § Thinking Engine` — module inventory + LOC census
- `.claude/board/TECH_DEBT.md` 2026-05-13 — TD-THINKING-ENGINE-UNWIRED-1 full entry
- `.claude/board/EPIPHANIES.md` 2026-05-13 — thinking-engine wiring epiphany
- `.claude/board/IDEAS.md` 2026-05-13 — proposed PR structure and wiring detail
- `.claude/plans/super-domain-rbac-tenancy-v1.md` §3.9 — UnifiedBridge Tier A design
- `crates/lance-graph-callcenter/src/unified_bridge.rs` — current struct definition
- `crates/thinking-engine/src/cognitive_stack.rs` — MetaCognition, GateState, ThinkingStyle
- `crates/thinking-engine/src/ghosts.rs` — GhostField, Ghost, GhostType
- `crates/thinking-engine/src/qualia.rs` — Qualia17D, DIMS_17D (17 calibrated dimensions)
- `crates/thinking-engine/src/composite_engine.rs` — CompositeEngine, CompositeResult
- `crates/thinking-engine/src/l4_bridge.rs` — commit_to_l4 (SpoQuad ingest hook)
- `crates/thinking-engine/src/osint_bridge.rs` — OsintThinkingBridge (hard-lock surface)
- `crates/thinking-engine/src/contrastive_learner.rs` — ContrastiveLearner (DP noise)
- `crates/thinking-engine/src/role_tables.rs` — LayerTables, gate_modulate (HKDF seed)
- Sprint-log-4 W4 spec `.claude/specs/td-super-domain-subcrates.md` (cross-flag §7.1)
- Sprint-log-4 W11 spec `.claude/specs/fma-heart-click-smoke.md` (cross-flag §7.2)
