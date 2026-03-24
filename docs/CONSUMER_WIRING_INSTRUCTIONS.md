# Consumer Wiring Instructions: lance-graph-contract

> **For the other session**: ladybug-rs, crewai-rust, n8n-rs integration
>
> **Branch**: `claude/unified-query-planner-aW8ax` (all repos)
>
> **Source of truth**: `lance-graph-contract` crate in `adaworldapi/lance-graph`

---

## What Was Built (This Session)

### ndarray (adaworldapi/ndarray)
- `src/hpc/cam_pq.rs` — CAM-PQ codec (encode, decode, DistanceTables, PackedDatabase, k-means training)
- 14 tests passing

### lance-graph (adaworldapi/lance-graph)

#### lance-graph-contract (NEW — zero-dep trait crate)
| Module | Types | Trait |
|--------|-------|-------|
| `thinking` | ThinkingStyle (36), StyleCluster (6), PlannerCluster (4), FieldModulation (7D), ScanParams | `ThinkingStyleProvider` |
| `mul` | SituationInput, MulAssessment, TrustQualia, DkPosition, FlowState, GateDecision | `MulProvider` |
| `nars` | InferenceType (5), QueryStrategy (5), SemiringChoice (5) | — |
| `plan` | ThinkingContext, PlanResult, PlanError, StrategySelector, QueryFeatures | **`PlannerContract`** |
| `cam` | CamByte, CamStrategy, DistanceTableProvider | `CamCodecContract`, `IvfContract` |
| `jit` | JitTemplate, KernelHandle, JitError | `JitCompiler`, `StyleRegistry` |
| `orchestration` | StepDomain (5), UnifiedStep, StepStatus, OrchestrationError | **`OrchestrationBridge`** |

#### lance-graph cam_pq/ (NEW)
- `udf.rs` — Real DataFusion ScalarUDF (`cam_distance`, `cam_heel_distance`)
- `storage.rs` — Arrow schema + RecordBatch for FIXED_SIZE_BINARY(6)
- `ivf.rs` — IVF coarse partitioning (k-means, probe, merge)
- `jitson_kernel.rs` — JITSON templates for cascade/full ADC scan

#### lance-graph-planner physical/ (NEW)
- `cam_pq_scan.rs` — CamPqScanOp physical operator (FullAdc/Cascade/IvfCascade)
- `ScanStrategy::CamPq` variant in IR

---

## What Each Consumer Must Do

### 1. CREWAI-RUST — Replace `ThinkingStyle` enum

**Current**: `src/persona/thinking_style.rs` defines its own 36-style enum + 23D vectors

**Change**: Import from contract instead. Delete the local enum.

```toml
# crewai-rust/Cargo.toml
[dependencies]
lance-graph-contract = { git = "https://github.com/adaworldapi/lance-graph.git", branch = "claude/unified-query-planner-aW8ax" }
```

```rust
// src/persona/thinking_style.rs — REPLACE the enum, keep the vectors
use lance_graph_contract::thinking::{ThinkingStyle, StyleCluster, FieldModulation};

// Keep STYLE_VECTORS (23D sparse vectors) — implement ThinkingStyleProvider
impl ThinkingStyleProvider for CrewaiStyleProvider {
    fn style_vector(&self, style: ThinkingStyle) -> SparseVec {
        STYLE_VECTORS[style as usize].clone()
    }
    fn default_modulation(&self, style: ThinkingStyle) -> FieldModulation {
        // Map from 23D axes to 7D FieldModulation
        let v = &STYLE_VECTORS[style as usize];
        FieldModulation {
            resonance_threshold: find_dim(v, 0).unwrap_or(0.5), // depth
            fan_out: (find_dim(v, 9).unwrap_or(0.5) * 8.0) as usize, // openness
            depth_bias: find_dim(v, 1).unwrap_or(0.5), // somatic
            breadth_bias: find_dim(v, 9).unwrap_or(0.5), // openness
            noise_tolerance: find_dim(v, 7).unwrap_or(0.3), // playfulness
            speed_bias: find_dim(v, 8).unwrap_or(0.5), // urgency
            exploration: find_dim(v, 11).unwrap_or(0.3), // wonder
        }
    }
    fn select_from_assessment(&self, assessment: &MulAssessment) -> ThinkingStyle {
        // Existing logic from jit_link.rs
    }
}
```

**Delete**: The local `ThinkingStyle` enum, `StyleCluster` enum, `tau()` method.
**Keep**: `STYLE_VECTORS`, `STYLE_TEXTURES`, `STYLE_TO_TAU_ARRAY` (these are crewai-specific data).

**Also wire**: `StepDomain` from `lance_graph_contract::orchestration` into `src/contract/router.rs`:

```rust
// src/contract/router.rs — replace local StepDomain with contract's
use lance_graph_contract::orchestration::StepDomain;
// Delete the local StepDomain enum
```

---

### 2. N8N-RS — Replace `ThinkingMode` + Wire JIT Contract

**Current**: `n8n-contract/src/thinking_mode.rs` defines its own `InferenceType`, `QueryPlan`, `ThinkingMode`

**Change**: Import `InferenceType` from contract. Keep `ThinkingMode` (it adds parameters).

```toml
# n8n-rust/crates/n8n-contract/Cargo.toml
[dependencies]
lance-graph-contract = { git = "https://github.com/adaworldapi/lance-graph.git", branch = "claude/unified-query-planner-aW8ax" }
```

```rust
// n8n-contract/src/thinking_mode.rs — REPLACE InferenceType, keep ThinkingMode
use lance_graph_contract::nars::InferenceType;
// Delete the local InferenceType enum

// ThinkingMode stays (it has cam_top_k, beam_width etc. — those are n8n-specific params)
pub struct ThinkingMode {
    pub inference_type: InferenceType,  // ← now from contract
    pub cam_top_k: usize,
    // ... rest stays the same
}
```

**Wire JIT contract**: `CompiledStyleRegistry` should implement `StyleRegistry`:

```rust
// n8n-contract/src/compiled_style.rs
use lance_graph_contract::jit::{StyleRegistry, JitTemplate, KernelHandle, JitError};
use lance_graph_contract::thinking::ThinkingStyle;

impl StyleRegistry for CompiledStyleRegistry {
    fn get_kernel(&self, style: ThinkingStyle) -> Result<KernelHandle, JitError> {
        let tau = style.tau();
        // Look up in existing kernel cache by tau address
        self.cache.get(&tau).copied().ok_or(JitError::CompileFailed(
            format!("No kernel for τ=0x{:02X}", tau)
        ))
    }

    fn warm_cache(&self) -> Result<(), JitError> {
        for style in ThinkingStyle::ALL {
            let template = self.template_for(style);
            let kernel = self.compiler.compile(&template)?;
            self.cache.insert(style.tau(), kernel);
        }
        Ok(())
    }

    fn template_for(&self, style: ThinkingStyle) -> JitTemplate {
        let modulation = style.default_modulation();  // from FieldModulation::default per cluster
        let scan_params = modulation.to_scan_params();
        JitTemplate {
            json: self.generate_jitson(style),  // existing logic
            tau_address: style.tau(),
            scan_params,
        }
    }
}
```

**Replace routing**: `crew_router.rs` and `ladybug_router.rs` should delegate to `OrchestrationBridge`:

```rust
// n8n-contract/src/executors.rs
use lance_graph_contract::orchestration::{OrchestrationBridge, StepDomain, UnifiedStep};

// Instead of HTTP proxies, use the bridge trait:
fn execute_step(bridge: &dyn OrchestrationBridge, step: &mut UnifiedStep) {
    bridge.route(step)  // In single binary = direct fn call. Multi-process = Flight.
}
```

---

### 3. LADYBUG-RS — Wire PlannerContract + OrchestrationBridge

**Current**: 5 Cypher paths (none complete), no lance-graph dep

**Change**: Add contract dep. Create `src/query/planner.rs`. Wire into HybridEngine.

```toml
# ladybug-rs/Cargo.toml
[dependencies]
lance-graph-contract = { git = "https://github.com/adaworldapi/lance-graph.git", branch = "claude/unified-query-planner-aW8ax" }

# Optional: full planner (for single-binary mode)
[dependencies.lance-graph-planner]
git = "https://github.com/adaworldapi/lance-graph.git"
branch = "claude/unified-query-planner-aW8ax"
optional = true

[features]
planning = ["lance-graph-planner"]
```

**New file**: `src/query/planner.rs`

```rust
use lance_graph_contract::plan::{PlannerContract, PlanResult, PlanError, StrategySelector};
use lance_graph_contract::mul::SituationInput;
use lance_graph_contract::thinking::ThinkingStyle;

/// Planner wrapper for ladybug-rs.
///
/// When `planning` feature is enabled, uses lance-graph-planner directly.
/// When disabled, uses a stub that returns minimal plans.
pub struct LadybugPlanner {
    #[cfg(feature = "planning")]
    inner: Box<dyn PlannerContract>,
    #[cfg(not(feature = "planning"))]
    _phantom: (),
}

impl LadybugPlanner {
    #[cfg(feature = "planning")]
    pub fn new() -> Self {
        Self {
            inner: Box::new(lance_graph_planner::PlannerAwareness::new()),
        }
    }

    #[cfg(not(feature = "planning"))]
    pub fn new() -> Self {
        Self { _phantom: () }
    }

    pub fn plan(&self, query: &str, situation: &SituationInput) -> Result<PlanResult, PlanError> {
        #[cfg(feature = "planning")]
        { self.inner.plan_full(query, situation) }

        #[cfg(not(feature = "planning"))]
        {
            Ok(PlanResult {
                mul: None,
                thinking: None,
                strategies_used: vec!["stub".into()],
                free_will_modifier: 1.0,
                compass_score: None,
            })
        }
    }
}
```

**Wire into HybridEngine** (`src/query/hybrid.rs`):

```rust
use crate::query::planner::LadybugPlanner;

impl HybridEngine {
    pub fn with_planner(mut self, planner: LadybugPlanner) -> Self {
        self.planner = Some(planner);
        self
    }

    pub async fn execute(&self, query: &str) -> Result<...> {
        // NEW: plan before execute
        if let Some(planner) = &self.planner {
            let situation = SituationInput::default(); // or from session context
            let plan_result = planner.plan(query, &situation)?;
            // Use plan_result.thinking to parameterize execution
        }
        // ... existing execution logic
    }
}
```

**Delete dead code**:
- `src/query/cypher.rs` (P1, 1560 lines) — transpiler, never executes
- `src/query/lance_parser/` (P3, 5532 lines) — orphaned lance-graph copy

**Implement OrchestrationBridge**:

```rust
// src/orchestration/bridge.rs
use lance_graph_contract::orchestration::*;
use lance_graph_contract::plan::ThinkingContext;

pub struct LadybugOrchestrationBridge {
    // References to subsystem handlers
}

impl OrchestrationBridge for LadybugOrchestrationBridge {
    fn route(&self, step: &mut UnifiedStep) -> Result<(), OrchestrationError> {
        let domain = StepDomain::from_step_type(&step.step_type)
            .ok_or(OrchestrationError::RoutingFailed(
                format!("Unknown step type: {}", step.step_type)
            ))?;

        match domain {
            StepDomain::Crew => {
                // In single binary: call crewai-rust handler directly
                // In multi-process: Arrow Flight to crewai-rust
            }
            StepDomain::Ladybug => {
                // Local: BindSpace operations
            }
            StepDomain::LanceGraph => {
                // Local: lance-graph query execution
            }
            StepDomain::N8n => {
                // Delegate to n8n-rs workflow engine
            }
            StepDomain::Ndarray => {
                // Direct SIMD: ndarray HPC call
            }
        }
        Ok(())
    }

    fn resolve_thinking(&self, style: ThinkingStyle, inference_type: InferenceType) -> ThinkingContext {
        // Use contract types to build context
    }

    fn domain_available(&self, domain: StepDomain) -> bool {
        match domain {
            StepDomain::Ladybug | StepDomain::LanceGraph => true,
            StepDomain::Crew => cfg!(feature = "vendor-crewai"),
            StepDomain::N8n => cfg!(feature = "vendor-n8n"),
            StepDomain::Ndarray => true,
        }
    }
}
```

---

## Thinking Style Reconciliation

| crewai-rust (36 styles) | lance-graph-planner (12 → PlannerCluster) | contract (canonical) |
|---|---|---|
| Logical, Analytical, Critical, Systematic, Methodical, Precise | Analytical, Convergent, Systematic → **Convergent** | ThinkingStyle::Analytical etc. → PlannerCluster::Convergent |
| Creative, Imaginative, Innovative, Artistic, Poetic, Playful | Creative, Divergent, Exploratory → **Divergent** | ThinkingStyle::Creative etc. → PlannerCluster::Divergent |
| Empathetic, Compassionate, Supportive, Nurturing, Gentle, Warm | Focused, Diffuse, Peripheral → **Attention** | ThinkingStyle::Empathetic etc. → PlannerCluster::Attention |
| Direct, Concise, Efficient, Pragmatic, Blunt, Frank | (mapped to Convergent) | ThinkingStyle::Direct etc. → PlannerCluster::Convergent |
| Curious, Exploratory, Questioning, Investigative, Speculative, Philosophical | (mapped to Divergent) | ThinkingStyle::Curious etc. → PlannerCluster::Divergent |
| Reflective, Contemplative, Metacognitive, Wise, Transcendent, Sovereign | Intuitive, Deliberate, Metacognitive → **Speed** | ThinkingStyle::Reflective etc. → PlannerCluster::Speed |

**Key**: crewai-rust keeps all 36 styles for behavioral richness. The planner collapses to 4 clusters for cost model decisions. The contract holds both mappings.

---

## Migration Checklist

### crewai-rust
- [ ] Add `lance-graph-contract` dep
- [ ] Replace local `ThinkingStyle` enum with contract's
- [ ] Keep `STYLE_VECTORS` + `STYLE_TEXTURES` (crewai-specific data)
- [ ] Implement `ThinkingStyleProvider` trait
- [ ] Replace local `StepDomain` with contract's
- [ ] Wire `OrchestrationBridge` into StepRouter

### n8n-rs
- [ ] Add `lance-graph-contract` dep to `n8n-contract`
- [ ] Replace local `InferenceType` with contract's
- [ ] Implement `StyleRegistry` for `CompiledStyleRegistry`
- [ ] Replace HTTP proxies in crew_router/ladybug_router with `OrchestrationBridge`
- [ ] Wire τ addresses from contract into kernel cache keys

### ladybug-rs
- [ ] Add `lance-graph-contract` dep
- [ ] Create `src/query/planner.rs` wrapping `PlannerContract`
- [ ] Wire planner into `HybridEngine` (parse → plan → execute)
- [ ] Delete `src/query/cypher.rs` (P1 dead code, 1560 lines)
- [ ] Delete `src/query/lance_parser/` (P3 dead code, 5532 lines)
- [ ] Implement `OrchestrationBridge` in `src/orchestration/bridge.rs`
- [ ] Optionally add `lance-graph-planner` as feature-gated dep

---

## What NOT to Do

1. **Do NOT** copy types from the contract — import them
2. **Do NOT** add lance-graph-planner as a hard dep in ladybug-rs — use feature gate
3. **Do NOT** delete crewai-rust's 23D vectors or style textures — they're behavioral data, not duplicated types
4. **Do NOT** change n8n-rs ThinkingMode struct — it adds parameters on top of InferenceType
5. **Do NOT** merge the repositories — the contract crate IS the coupling point
6. **Do NOT** add serde to the contract — keep it zero-dep. Consumers add serde derives via wrapper types if needed.
