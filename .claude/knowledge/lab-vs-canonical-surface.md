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

## AGI IS the struct-of-arrays (per `docs/HISTORICAL_CONTEXT.md` Era 8)

**The structural identity** — commit this to muscle memory:

```
struct-of-arrays  ⇄  cognitive-shader-driver  ⇄  UnifiedStep + OrchestrationBridge
   (BindSpace)         (per-cycle dispatch)        (cross-cycle composition)

AGI = (topic, angle, thinking, planner)
    = struct-of-arrays consuming cognitive-shader-driver
```

AGI is **not a new crate**, not a new `struct Agi`, not a new API. AGI
is the semantic interpretation of the already-shipped `BindSpace`
columns, driven by the already-shipped `ShaderDriver`, composed across
cycles by the already-shipped `OrchestrationBridge`.

### The four AGI axes = four SoA columns

Era 8's `CognitiveRecord` (HISTORICAL_CONTEXT.md lines 176-198) lists
11 fields in 4 semantic groups. The four AGI-perspective axes are the
consumer-visible distillation:

| AGI axis | BindSpace column (shipped) | Era 8 `CognitiveRecord` field(s) | 5D-stream dimension |
|---|---|---|---|
| **topic** — "what about" | `fingerprints: FingerprintColumns` | `topic: Fingerprint<256>` (+ `fingerprint` for content identity) | D2: Context |
| **angle** — "from whose view" | `qualia: QualiaColumn` (18×f32) | `angle: Fingerprint<256>` + `qualia: [f32; 18]` | D3: Perspective |
| **thinking** — dispatch + modulation | `meta: MetaColumn` (packed u32 MetaWord) | `shader_mask: u8` (active layers) + style bits | dispatch gate |
| **planner** — "why/how", causal composition | `edges: EdgeColumn` (CausalEdge64) | `edge: CausalEdge64` + `rung: u8` | D4: Causality |

D1 (Content = raw Fingerprint) and D5 (Time = temporal index) are
carried inside the `meta` column's packed bits and the per-row
position index respectively — so the 5D stream still walks through
the same 4 SoA columns.

### The per-cycle contract (what ShaderDriver actually does)

`ShaderDriver::dispatch(ShaderDispatch)` reads the row-window from
`BindSpace`, processes the 5D stream, and emits `ShaderHit` + `MetaWord`
through the sink:

```
per cycle:   CognitiveShader processes 5D stream over a row window
per branch:  independent causal trajectory (CausalEdge64 on edges column)
per merge:   NARS evidence accumulation via CausalEdge64 revision
per dream:   offline consolidation (prune / merge / permute-XOR-bind)
```

### Migration status (per `docs/BINDSPACE_MIGRATION_GAP.md`)

The shipped BindSpace is a 4-column SoA that **compresses** the Era 8
11-field CognitiveRecord into the 4 AGI axes. Era 8's distinct
`topic` + `angle` + `fingerprint` (three Fingerprint<256> columns) are
currently folded into a single `fingerprints` column + the 18D qualia
projection. This is a deliberate distillation, not a loss:

- `fingerprints` carries row identity (content = topic-fingerprint).
- `qualia` carries the perspective (angle = 18D phenomenal coords).
- The full Era-8 schema lands if/when the migration in
  `BINDSPACE_MIGRATION_GAP.md` ships (Container/CogRecord canonicalisation
  in `lance-graph-contract`). That extension **adds columns to the same
  SoA**; it does not add a new crate or a new dispatch API.

### Consequences for future sessions

- **Never** write a parallel `struct Agi { topic, angle, thinking, planner }`.
  Those ARE the BindSpace columns. Wrapping them in a new struct
  breaks the SoA layout and the SIMD sweep.
- **Never** introduce an `agi` or `cognitive_record` crate that re-
  exports the axes. `cognitive-shader-driver` already IS the unified
  API; extend it by adding columns, not by wrapping it.
- **Extend by column, not by layer.** New AGI capability = new
  BindSpace column (with its SoA allocation + meta-prefilter + sink
  emission) consumed by the existing `ShaderDriver`. Not a new trait,
  not a new endpoint, not a new DTO family.
- **Era 8's "weights are seeds" doctrine**: weights don't compress
  into parameters; they seed the holographic BindSpace. If a future
  session proposes "train an AGI model," the correct framing is "hydrate
  new BindSpace rows from seed weights and let ShaderDriver dispatch
  cycles against them." No gradient-descent training loop lives on
  the canonical surface.

## Architecture Invariants (cross-cutting; cite these, do not drift)

The AGI-as-SoA identity above doesn't stand alone. A small set of
load-bearing invariants from the wider architecture docs compose
with it. Treat these as non-negotiable; cite the ref when extending.

### I1. BindSpace is read-only; writes cross the `CollapseGate` airgap

Ref: `docs/INTEGRATION_PLAN_CS.md`.

- BindSpace columns are `Arc<[u64; 256 * N]>` — shared, read-only,
  mmap-friendly. Shader kernels never hold `&mut` into a column.
- Writers hold owned `Copy` microcopies (`CausalEdge64`, `Band`,
  `TruthValue`, `ThinkingStyle` — all ≤ 16 B, stack-only).
- Deltas cross the airgap through `contract::collapse_gate::{GateDecision,
  MergeMode}`. `MergeMode::Xor` = single-writer (XOR self-inverse);
  `MergeMode::Bundle` = multi-writer majority; superposition =
  preserve ambiguity for next cycle.
- No locks, no `&mut` during compute. Ordering doesn't matter (XOR
  commutative + associative); rollback is free.

If a session proposes "let the shader mutate a column in place" or
"let the REST handler write directly," stop — handlers construct a
`UnifiedStep`, the bridge yields a delta, the gate commits.

### I2. Canonical SIMD import surface: `ndarray::simd::*`

Ref: `docs/INTEGRATION_PLAN_CS.md` §"Architecture separation".

```rust
// Correct — stable public surface:
use ndarray::simd::{F32x16, U8x64, F16x32, Fingerprint, MultiLaneColumn, array_window};

// Wrong — reaches into private implementation:
use ndarray::hpc::fingerprint::Fingerprint;
use ndarray::hpc::simd_avx512::F32x16;
```

Anything not re-exported from `ndarray::simd::*` is internal. The
`hpc::*` path is free to refactor; consumers never reach it.

### I3. Layer temporal budgets

Ref: `docs/INTEGRATION_PLAN_CS.md` §"5-Layer Stack".

| Layer | Scope | Budget |
|---|---|---|
| L4 Planner | per query | milliseconds |
| L3 CollapseGate | per commit cycle | microseconds |
| L2 CognitiveShader | per step | nanoseconds |
| L1 BindSpace | per lane read | nanoseconds, zero-copy |
| L0 ndarray SIMD | per instruction | sub-nanosecond |

L0/L1 kernels never allocate. "Push fingerprint similarity into the
planner" violates the budget and is rejected.

### I4. Temperature hierarchy — cold narrows, then algebra fires

Ref: `docs/SEMIRING_ALGEBRA_SURFACE.md` §3 "Cold Path Numbing Effect".

| Path | Substrate | Semiring | Role |
|---|---|---|---|
| Hot | BindSpace HDR sweep | XorBundle / HammingMin / Resonance | full 16 Kbit, SIMD popcount |
| Warm | CAM-PQ cascade | CamPqAdc | 6-byte codes, 500 M candidates/s |
| Cold | DataFusion columnar joins | Boolean | scalar columns only — no fingerprint data |
| Frozen | `metadata.rs` | none | pure CRUD skeleton |

Cold joins narrow the candidate set first on scalar columns; HDR
semirings fire only on the narrow survivors. A full HDR semiring over
all rows is what the cold/warm pre-filters exist to prevent.

### I5. Thinking IS an `AdjacencyStore` — one engine, two graphs

Ref: `docs/THINKING_PIPELINE.md` §"Layer 3: ThinkingGraph".

- 36 thinking styles are nodes in a real `AdjacencyStore` (CSR/CSC +
  NARS-weighted edges). Styles live at τ-prefix `0x0D` in the
  `Addr(u16)` space (`docs/METADATA_SCHEMA_INVENTORY.md` §1E).
- Cognitive verbs (`EXPLORE` / `FANOUT` / `HYPOTHESIS` / …) call the
  same `batch_adjacent()` / `adjacent_truth_propagate()` /
  `adjacent_incoming()` the data queries use.
- **One engine, two graphs** — thinking is another client of the
  planner's adjacency substrate, not a meta-layer above it.

"Add a thinking engine that lives outside the planner" is wrong;
extend the topology, the verbs, or the NARS edge set.

### I6. Weights are seeds — hydrate-then-cascade, not matmul

Ref: `docs/COGNITIVE_SHADER_HYDRATION.md`, Era 8.

Each GGUF weight matrix hydrates (build-time) into:
1. 256 archetypes (bgz17 palette) + `Fingerprint<256>` per archetype.
2. 256×256 FisherZTable (64 KB, L1-resident).
3. Holographic residual (phase + magnitude slots).
4. `CausalEdge64` wiring (`S(row) × P(role) × O(col)`).

After bake, inference = Hamming cascade + palette lookup + XOR
compose. No matmul, no FP inner loop. "Training a model" on the
canonical surface means hydrating new BindSpace rows; no
gradient-descent loop lives on the canonical surface.

### I7. Per-cycle cascade budget (monotone narrowing)

Ref: `docs/COGNITIVE_SHADER_HYDRATION.md` §"Why struct-of-arrays".

```
sweep topic[]     → 50 000 survivors   (~2 ms    Hamming)
sweep angle[]     →  5 000 survivors   (~0.2 ms  Hamming)
sweep causality[] →    500 survivors   (~0.05 ms CausalEdge64 filter)
sweep qualia[]    →     50 survivors   (scalar, 18-D range)
exact on 50 → palette lookup → CausalEdge64 output
```

Intersect = bitmap AND over per-column hit masks. A proposed dispatch
order that breaks monotone narrowing (scoring qualia before Hamming,
for instance) loses the 99 % reject gate — reject the proposal.

### I8. 4096 address surface = 16 prefix × 256 slots

Ref: `docs/METADATA_SCHEMA_INVENTORY.md` §1A.

- `Addr(u16)` = 8-bit prefix : 8-bit slot.
- Prefix `0x0D` = thinking style templates (τ addresses).
- Each slot holds one `[u64; 256]` = 16 384-bit fingerprint.
- This IS the BindSpace substrate before column extension.

### I9. Three DTO families above SoA (doctrinal — not yet shipped)

Ref: `docs/integrated-architecture-map.md` §5.

| DTO | Holds | Stage |
|---|---|---|
| `StreamDto` | arrival-order inputs (text, SPO seeds, anchors) | pre-parse |
| `ResonanceDto` | superposition field (topic / angle / hypothesis) | active sweep |
| `BusDto` | explicit thought (anchor + mask + edges + style) | post-collapse → CognitiveShader |

**Field ≠ sweep ≠ bus.** The field is searchable terrain (10 K
`[i8/i16]`); the sweep is the cheap XOR/bundle collapse into a
lookup vector; the bus is explicit execution through p64 /
CognitiveShader. Do not conflate them.

### I10. HEEL / HIP / BRANCH / TWIG / LEAF — progressive precision

Ref: `docs/integrated-architecture-map.md` §3.

| Level | Role | Maps to |
|---|---|---|
| HEEL | coarse basin routing | bgz17, Base17 |
| HIP | family sharpening | 16 384 palette, i16 |
| BRANCH | contradiction / signed split | polarity residuals |
| TWIG | local prototype neighborhood | CLAM ripple |
| LEAF | exact member | full fingerprint |

Each level narrows; never skip levels. bgz17 IS HEEL — do not ask
it to also be LEAF identity.

---

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

