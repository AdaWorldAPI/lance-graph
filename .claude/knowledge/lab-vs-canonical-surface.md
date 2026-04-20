# LAB-ONLY vs Canonical Surface — Anti-Hallucination Guard

> READ BY: all agents touching cognitive-shader-driver, REST/gRPC
> endpoints, Wire DTOs, OrchestrationBridge, UnifiedStep, codec research,
> planner integration, or anything that looks like "external API"

## Status: FINDING (architectural invariant — do not drift)

Date: 2026-04-20
Related PRs: #220, #221

---

## The One-Line Rule

**`cognitive-shader-driver` IS the unified API.** Its public surface —
the super DTO family `ShaderDispatch` / `ShaderHit` / `MetaWord` /
`BindSpace*` columns / `ShaderDriver` / `ShaderSink`, plus
`UnifiedStep` + `OrchestrationBridge` for cross-domain composition —
is the ONLY canonical consumer surface. REST, gRPC, per-op Wire DTOs,
the shader-lab binary, codec research endpoints, and every
`Wire*Request` type are LAB-ONLY scaffolding. Everything compiles into
one binary (`shader-lab`); consumers never see the scaffolding.

## Two Canonical Layers (both live in `cognitive-shader-driver`'s re-exports)

1. **Per-cycle super DTO** (the hot path every consumer walks):
   `ShaderDispatch` → `ShaderDriver` → `ShaderHit` + `MetaWord`
   emission through `ShaderSink`. This is the unified dispatch shape
   for one cognitive cycle. `BindSpace` columns are the substrate;
   `MetaFilter` + `StyleSelector` configure the sweep.

2. **Cross-domain composition** (when one cycle is part of a larger
   orchestration): `UnifiedStep` routed through
   `OrchestrationBridge` by `StepDomain` (Crew / Ladybug / N8n /
   LanceGraph / Ndarray / Thinking / Query / Semantic / Persistence /
   Inference / Learning). Every domain implements the same trait;
   research is just one domain.

Both layers are zero-dep (defined in `lance-graph-contract`,
re-exported by `cognitive-shader-driver`). Both are always compiled.
Both are consumer-facing.

## What "LAB-ONLY" Means Concretely

These modules are gated behind `--features lab` (or its components:
`serve` / `grpc` / `with-planner` / `with-engine`) and are NOT part of
the canonical library surface:

| Module | Purpose | LAB-only reason |
|---|---|---|
| `cognitive-shader-driver/src/wire.rs` | Per-op REST/proto DTOs (WireDispatch, WireCalibrate, WireProbe, WireTensors, WirePlan, WireRunbook) | Test-transport convenience; canonical DTO is `UnifiedStep` |
| `cognitive-shader-driver/src/serve.rs` | Axum REST server with `/v1/shader/*` handlers | HTTP transport for Claude Code backend testing only |
| `cognitive-shader-driver/src/grpc.rs` | tonic gRPC service | Same as REST, different wire format |
| `cognitive-shader-driver/src/codec_research.rs` | Backing logic for calibrate/probe/tensors ops | Research is one consumer, not the canonical architecture |
| `cognitive-shader-driver/src/codec_bridge.rs` | `CodecResearchBridge: OrchestrationBridge` for StepDomain::Ndarray | The research consumer's bridge impl — canonical trait, lab-only consumer |
| `cognitive-shader-driver/src/planner_bridge.rs` | Per-op WirePlan shortcut → PlannerAwareness | `PlannerAwareness` already impls OrchestrationBridge directly; this is a lab test adapter only |

The canonical surface — all of it re-exported from
`cognitive-shader-driver` so consumers depend on one crate:

| Surface (via `cognitive-shader-driver` re-export) | Role |
|---|---|
| `ShaderDispatch`, `ShaderHit`, `ShaderBus`, `ShaderSink`, `NullSink`, `ShaderCrystal`, `ShaderResonance`, `MetaWord`, `MetaFilter`, `MetaSummary`, `ColumnWindow`, `EmitMode`, `RungLevel`, `StyleSelector`, `CognitiveShaderDriver` | Per-cycle super DTO family (from `contract::cognitive_shader`) |
| `BindSpace`, `BindSpaceBuilder`, `EdgeColumn`, `FingerprintColumns`, `MetaColumn`, `QualiaColumn` | L1 BindSpace columns |
| `CognitiveShaderBuilder`, `ShaderDriver` | L2 driver (the shader IS the driver) |
| `engine_bridge::{UnifiedStyle, UNIFIED_STYLES, unified_style, ingest_codebook_indices, dispatch_from_top_k, write_qualia_17d, read_qualia_17d, persist_cycle, EngineBusBridge}` | Engine bridge utilities |
| `GateDecision`, `MergeMode` (from `contract::collapse_gate`) | L3 CollapseGate protocol |
| `OrchestrationBridge`, `UnifiedStep`, `StepDomain`, `BridgeSlot` (from `contract::orchestration`) | Cross-domain composition |
| `Blackboard`, `BlackboardEntry`, `ExpertCapability` (from `contract::a2a_blackboard`) | Layer-1 runtime A2A bus |

## 5-Layer Stack Mapping (per `docs/INTEGRATION_PLAN_CS.md`)

```
L4 Planner strategies        → lance-graph-planner (canonical crate)
L3 CollapseGate              → contract::collapse_gate (canonical DTO)
L2 CognitiveShader           → contract::cognitive_shader + driver::{driver,engine_bridge}
L1 BindSpace                 → driver::bindspace (struct-of-arrays columns)
L0 ndarray SIMD              → ndarray crate (F32x16 / U8x64 / F16x32 / F64x8)
```

Every layer has its canonical surface in a zero-dep or feature-free
module. The lab transport (REST/gRPC) is **orthogonal to the stack** —
it's a debug probe, not a layer.

## The Failure Mode This Doc Prevents

Future sessions reading this crate see `/v1/shader/dispatch`,
`/v1/shader/calibrate`, `/v1/shader/probe`, `/v1/shader/plan` and think:

> "There's a REST API. I'll add `/v1/shader/<new_thing>` for this new
> feature."

**That is the hallucination.** The REST endpoints are not the canonical
API. Adding more of them does not extend the consumer surface — it only
bloats the lab transport.

### The cognitive traps (name them to dodge them)

- **Kahneman-Tversky System-1 easy route.** Pattern-matching on "REST
  server exists → add REST endpoint" is fast and feels productive. The
  System-2 correct move is slower: read this doc, extend
  `OrchestrationBridge` / add a new `StepDomain` variant in the
  contract, implement the trait in the relevant consumer crate, and
  let the lab transport dispatch the new `step_type` automatically
  with zero new endpoint code.
- **Dunning-Kruger confidence.** A session that sees axum + tonic +
  `Wire*Request` types assumes it understands the architecture. It
  does not. The architecture is `OrchestrationBridge`. The HTTP layer
  is a test harness. Confidence about "the API" without reading this
  doc is the confidence zone where hallucinations ship.
- **Availability bias.** Because REST examples in training data are
  everywhere, the easy path is "write more REST." Our stack is
  deliberately different: one trait, one `UnifiedStep`, N bridges.

## The Decision Procedure (paste this into the session's thinking)

Before adding ANY new endpoint, handler, or Wire DTO, answer in order:

1. **Does the canonical bridge already handle this?** Check
   `StepDomain` variants (Crew / Ladybug / N8n / LanceGraph / Ndarray /
   future) and existing `OrchestrationBridge` impls. If an existing
   bridge can dispatch the new operation via a new `step_type` string,
   **stop — no new endpoint is needed**. The lab transport already
   routes `UnifiedStep` through whichever bridge claims the domain.

2. **Is a new StepDomain needed?** If yes, add the variant in
   `lance-graph-contract::orchestration::StepDomain`. Implement
   `OrchestrationBridge` for the consumer in that consumer's crate
   (e.g. the way `PlannerAwareness` does it in
   `lance-graph-planner/src/orchestration_impl.rs`). **Do not add a
   new `/v1/<new>` endpoint.**

3. **Is this purely a lab test convenience?** If and only if the
   answer is unambiguously yes (e.g., "I want a curl-friendly
   per-op shortcut for debugging"), add it under `src/wire.rs` +
   `src/serve.rs` or `src/grpc.rs` and document that it routes
   through the canonical bridge internally. Mark it `LAB-ONLY` in
   the module doc.

4. **Never**:
   - Publish a per-op endpoint as "the API" in PR descriptions or
     knowledge docs. Canonical is `UnifiedStep` via the bridge.
   - Add a per-op DTO outside `wire.rs`.
   - Introduce a new REST handler that has its own dispatch logic.
     Handlers must construct a `UnifiedStep` (or delegate to an
     existing bridge) and return the bridge's result.
   - Expose HTTP/gRPC types (axum / tonic / prost) from any
     crate-root re-export without the feature gate.

## Feature Gate Reference

```toml
# crates/cognitive-shader-driver/Cargo.toml
[features]
default       = []                                          # canonical library only
with-engine   = ["dep:thinking-engine"]                     # lab: engine consumer
with-planner  = ["dep:lance-graph-planner"]                 # lab: planner consumer
serve         = ["dep:serde", "dep:serde_json",
                 "dep:axum", "dep:tokio"]                    # lab: REST transport
grpc          = ["dep:prost", "dep:tonic",
                 "dep:tonic-build", "dep:tokio"]             # lab: gRPC transport
lab           = ["serve", "grpc", "with-engine",
                 "with-planner"]                             # umbrella: the shader-lab binary
```

All the lab features coexist — REST and gRPC are siblings, both under
the lab umbrella. Neither is a subset of the other.

## Related Invariants

- The whole lab surface compiles into ONE binary (`shader-lab`). There
  is no service split; consumers embed the library.
- The canonical bridge has no runtime cost when lab features are off —
  `OrchestrationBridge` is a plain trait; `UnifiedStep` is a plain
  struct; `StepDomain` is a plain enum.
- `CodecResearchBridge` is a consumer like any other — it happens to
  need the lab features because it only exists to back the research
  Wire DTOs, but the trait it implements is canonical.

## Where to Read Next

- `crates/lance-graph-contract/src/orchestration.rs` — the canonical trait + DTO.
- `crates/lance-graph-planner/src/orchestration_impl.rs` — the reference consumer impl.
- `crates/cognitive-shader-driver/src/codec_bridge.rs` — the lab-only consumer impl for research.
- `.claude/knowledge/cam-pq-unified-pipeline.md` — the pipeline doc that produced this invariant.
- `.claude/knowledge/cognitive-shader-architecture.md` — the broader shader-as-driver architecture.

---

**If you are a future session and this doc is loaded, you are on the
correct path. If you are about to add a REST handler without having
read this section, stop and re-read the Decision Procedure above.**
