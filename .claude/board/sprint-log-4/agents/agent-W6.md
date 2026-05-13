# agent-W6 — TD-THINKING-ENGINE-UNWIRED-1

## Start

**2026-05-13** — W6 begin. Deliverable: `.claude/specs/td-thinking-engine-wire.md`

## Recon

- Confirmed `crates/thinking-engine/src/` has 50+ modules (~582 KB)
- Key cognitive types:
  - `ghosts.rs` → `GhostField` / `Ghost` / `GhostType` — Friston free-energy prediction cache / drift priors
  - `qualia.rs` → `Qualia17D` / `ConvergenceSnapshot` — 17D computable feeling (tension, drift, clarity)
  - `cognitive_stack.rs` → `ThinkingStyle`, `GateState`, `RungLevel`, `MetaCognition` — homeostasis gate
  - `composite_engine.rs` → `CompositeEngine` / `CompositeResult` — multi-lens intent classify
  - `l4_bridge.rs` → `commit_to_l4()` — L3→L4 XOR-bind learning signal
  - `tensor_bridge.rs` → `EmbeddingOutput` — unified embedding across frameworks
  - `world_model.rs` → `WorldModelDto` / `SelfState` — situational awareness
- `UnifiedBridge<B>` lives in `crates/lance-graph-callcenter/src/unified_bridge.rs`
  - Has `authorize_read`, `authorize_write`, `authorize_act` — audit-emitting hot path
  - Fields: `bridge`, `policy`, `tenant`, `audit_sink`, `audit_chain`
  - No `CognitiveStack` field currently
- §16-§19 D-SDR scaffolding: `D-SDR-13` (drift), `D-SDR-15` (qualia), `D-SDR-17` (homeostasis) — wired as clean-room types

## Output

Spec written to `.claude/specs/td-thinking-engine-wire.md` (~12 KB)

## Status: DONE

## Spec written — 2026-05-13

Spec file: `.claude/specs/td-thinking-engine-wire.md`
Size: ~12 KB confirmed.

### Key decisions recorded
- Compose-not-rebuild table: D-SDR-13 → ghosts.rs/role_tables.rs/persona.rs; D-SDR-15 → qualia.rs/cronbach.rs/contrastive_learner.rs; D-SDR-17 → cognitive_stack.rs/osint_bridge.rs/ghosts.rs
- UnifiedBridge<B> gains `cognition: Option<Arc<Mutex<CognitionHandle>>>` field (backward compat noop default)
- Three hook surfaces: tensor_bridge::EmbeddingOutput::to_f32() for SpoQuad ingest; MetaCognition::record() for drift; CompositeEngine::compose() for FMA intent classification
- New crate: `lance-graph-cognition-bridge` (not extending lance-graph-contract which is zero-dep)
- Risk: CognitionHandle must NOT implement Clone; one per TenantId; park_lot::Mutex or thread-local amortisation
- W4 cross-flag: CognitionHandle::for_super_domain(sd) seeds decay_rate + Brier window per SD
- W11 cross-flag: IntentClass { atom, confidence, lens_agreement, cypher_template } is the coordination type
- 5 open questions documented

## Status: COMPLETE
