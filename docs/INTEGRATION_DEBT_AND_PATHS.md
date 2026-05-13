# Integration Debt & Paths Forward: Strengths, Weaknesses, Epiphanies

> *Everything that's broken, everything that's brilliant, and how to connect the dots.*

## 1. STRENGTHS (What We Have That Nobody Else Does)

### S1. The Codec Stack Is Mathematically Complete
The Pareto frontier has 3 points. We have codecs that hit all 3:
- Full planes (ρ=1.000) → `Fingerprint<256>`
- 57-bit level (ρ=0.982) → ZeckBF17
- 8-bit level (ρ=0.937) → Scent byte / PaletteEdge

**No other graph database has a proven-optimal progressive compression chain for hyperdimensional graph data.** FalkorDB, Neo4j, KuzuDB — none of them have anything like this.

### S2. The Contract Crate Exists
`lance-graph-contract` is a zero-dependency trait crate with canonical types for:
- 36 thinking styles with τ addresses
- MUL assessment with Dunning-Kruger + compass
- CAM-PQ codec contracts
- JIT compilation contracts
- Orchestration bridge
- NARS inference types

**This is the architectural golden ticket.** Once consumers adopt it, type duplication drops from 40+ copies to zero.

### S3. Polyglot Parsing Is Real
lance-graph-planner strategy #1-4 can parse Cypher, Gremlin, SPARQL, and GQL → same IR.
Auto-detection works. Affinity scoring is correct.
**No other Rust graph engine supports 4 query languages through a single planner.**

### S4. The Elevation Model Is Novel
L0:Point → L1:Scan → L2:Cascade → L3:Batch → L4:IVF → L5:Async
Start cheap, escalate on observed resistance. **Nobody else does this** — they either plan upfront (expensive for simple queries) or use a fixed execution path (suboptimal for complex queries).

### S5. ndarray Has 880 Passing Tests
The HPC layer is battle-tested: BLAS L1/L2/L3, CLAM tree, cascade search, BNN, fingerprint operations, arrow bridge. This is production-grade infrastructure.

### S6. bgz17 Has 121 Passing Tests with Zero Dependencies
The palette semiring codec compiles standalone with zero external deps. It's the most portable, most tested codec in the stack.

### S7. MKL Backend Exists
ndarray has real Intel MKL FFI with cblas_sgemm, cblas_dgemm, VML, DFTI. When `intel-mkl` feature is enabled, every BLAS operation accelerates automatically.

### S8. The Thinking Pipeline Has Real Tests
```rust
#[test] fn auto_selects_dp_join_for_multi_hop_pattern()
#[test] fn auto_selects_sigma_scan_for_fingerprint_query()
#[test] fn collapse_gate_filters_low_resonance_results()
#[test] fn truth_propagation_accumulates_during_traversal()
#[test] fn resonance_mode_uses_thinking_style_affinity()
```
These pass. The planner logic works.

## 2. WEAKNESSES (What's Broken or Missing)

### W1. ndarray Build Is Broken (exit 101)
The most critical weakness. Without ndarray building, the entire foundation is unstable.
**Root cause**: Unknown — needs investigation (likely jitson/cranelift dep or SIMD compilation issue).

### W2. The Contract Crate Has Zero Consumers
`lance-graph-contract` exists but **nobody depends on it**:
- lance-graph-planner defines its own ThinkingStyle (12 variants, not 36)
- n8n-rs defines its own ThinkingMode
- crewai-rust defines its own styles
- ladybug-rs has no thinking types at all

**This is the #1 integration debt.** The contract was created to solve duplication, but the adoption step was never taken.

### W3. lance-graph-planner Is Self-Contained (Disconnected)
The planner has 16 strategies but:
- Parse strategies do regex on query text, not actual parsing
- Physical operators exist but can't execute (no DataFusion session)
- Elevation tracks levels but has no timing feedback loop
- NARS truth propagation works in tests but can't connect to real SPO store

**It's a well-architected prototype with no runtime integration.**

### W4. lance-graph → ndarray Dependency Missing
`ndarray_bridge.rs` is a standalone mirror of `Fingerprint<256>`. The actual `ndarray` crate is not a dependency of lance-graph. This means:
- No shared SIMD dispatch
- No shared CLAM tree
- No shared CAM-PQ codec
- Duplicate popcount/hamming implementations

### W5. bgz17 Still Excluded from Workspace
Cargo.toml still has `exclude = ["crates/bgz17"]`. Phase 3 integration has not started.

### W6. Three Copies of ZeckF64
ndarray, blasgraph, neighborhood — all have their own. Bug fixes must be applied to all three.

### W7. KuzuDB Is Empty
The kuzudb repo is a bare `.git` on our branch. The adjacency synergy with Kuzu's column-grouped CSR is theoretical, not practical. The planner's `DPJoinEnum` strategy is inspired by Kuzu but has no actual Kuzu code.

### W8. FieldModulation Threshold Divergence
Three different `to_scan_params()` implementations produce different thresholds from the same conceptual resonance value:
- Contract: `threshold = resonance × 1000`
- Planner: `threshold = resonance × 2000 + 100`
- n8n-rs: goes through jitson directly

**This is a semantic bug**, not just a duplication issue. A "resonance 0.5" query means different things depending on which path it takes.

### W9. No End-to-End Test
There is no single test that:
1. Takes a Cypher query
2. Runs it through the thinking pipeline
3. Selects strategies
4. Plans
5. Executes against real data
6. Returns results

Each layer is tested in isolation. The integration has never been tested.

## 3. EPIPHANIES (Insights That Change Everything)

### E1. "Orchestration IS Graph" (from docs/ORCHESTRATION_IS_GRAPH.md)
The capstone insight: the orchestration system (n8n-rs workflow DAGs, crewai-rust agent graphs, ladybug-rs BindSpace) **IS a graph problem**. lance-graph should be its own query engine AND its own orchestration substrate. The `WorkflowDAG` strategy (#15) and the `OrchestrationBridge` contract both point to this: **routing is traversal, scheduling is semiring mxv, thinking is node attributes**.

### E2. The Scent Byte IS a Boolean Lattice Adjacency
ZeckF64 byte 0's 7 bits form a boolean lattice: 19 legal states out of 128 possible. This means the scent byte doesn't just encode distance — it encodes **structural relationship type** across S/P/O dimensions. The 85% built-in error detection is a free integrity check.

### E3. PQ Lookup Tables ARE Matrix Multiplication
CAM-PQ distance = sum of 6 table lookups = dot product of 6-element vectors with precomputed distance columns. MKL `cblas_sgemv` on the 1536-float precomputed table against batch-encoded candidates = **vectorized batch ADC with zero custom SIMD code**. This connects CAM-PQ directly to ndarray's BLAS stack.

### E4. The Contract Crate Is the Rosetta Stone
If every consumer depends on `lance-graph-contract`, then:
- crewai-rust agents can directly set `ThinkingStyle` → planner uses it
- n8n-rs workflows can call `PlannerContract::plan_full()` → same-binary, zero-serde
- ladybug-rs can call `OrchestrationBridge::route()` → unified step dispatch
- ndarray implements `JitCompiler` → contract tells consumers how to compile styles

### E5. The 17th Dimension IS the Pythagorean Comma
From ZeckBF17: "16 = 2⁴ would alias. The 17th dimension IS the Pythagorean comma." This isn't just a cute observation — it explains why Base17 preserves information that Base16 doesn't. The prime dimensionality prevents subspace aliasing.

### E6. Dynamic Elevation = Self-Regulating Query Engine
The elevation model (`should_elevate()` trigger on resistance) means the engine learns where each query type lands on the cost spectrum without pre-classification. Combined with the learning loop (`elevation/learning.rs`), this creates a **self-regulating** query engine that adapts to workload patterns.

### E7. The Cold Path SHOULD Numb Thinking
DataFusion columnar joins on metadata.rs skeleton intentionally operate without semiring algebra. This is correct because:
1. Structural matching (label = "Person", age > 30) doesn't need hyperdimensional vectors
2. The semiring fires only AFTER joins narrow the candidate set
3. This prevents exponential blowup: N×N join with 2KB per element = disaster
4. The codec stack (CAM-PQ: 6 bytes, scent: 1 byte) enables pre-filtering before the cold path even runs

### E8. GQL Parser Completes the Surface Map
With GQL at prefix 0x02:0x80-0xFF in BindSpace AND `GqlParse` strategy in the planner, the full surface map is: Lance(0x00), SQL(0x01), Cypher(0x02:low), GQL(0x02:high), GraphQL(0x03), NARS(0x04). **Every prefix has a parser strategy.** The address space is self-describing.

## 4. INTEGRATION PATHS (Prioritized)

### Path 1: Fix Foundation (URGENT)
```
Priority: P0 (blocks everything)
Effort:   1 day
Risk:     Low

1. Fix ndarray build (exit 101)
   - Investigate: cargo build --features jitson 2>&1
   - Likely: missing cranelift dep or SIMD compile flag
2. Fix 2 doctest failures
   - crystal_encoder.rs line 251
   - udf_kernels.rs line 200
3. Verify: cargo test (880 tests pass)
```

### Path 2: Contract Adoption (HIGH VALUE)
```
Priority: P1 (highest ROI)
Effort:   3 days
Risk:     Medium (API changes in consumers)

1. lance-graph-planner: replace own ThinkingStyle with contract's
   - Change: src/thinking/style.rs imports from contract
   - Change: All 12-variant references → 36-variant contract enum
   - Fix: FieldModulation threshold divergence (pick ONE formula)

2. Add lance-graph-contract as dep in:
   - lance-graph-planner/Cargo.toml
   - n8n-rs/n8n-rust/crates/n8n-contract/Cargo.toml
   - crewai-rust/Cargo.toml

3. Replace duplicated types in each consumer:
   - n8n-rs: ThinkingMode → contract InferenceType + ThinkingStyle
   - crewai-rust: own 36 styles → contract ThinkingStyle

4. Implement contract traits:
   - PlannerContract in lance-graph-planner
   - JitCompiler in ndarray jitson engine
   - OrchestrationBridge in lance-graph-planner (routes to all subsystems)
```

### Path 3: Wire lance-graph → ndarray (FOUNDATION)
```
Priority: P1 (parallel with Path 2)
Effort:   2 days
Risk:     Medium (dependency management)

1. Add ndarray as optional dep in lance-graph core:
   ndarray = { path = "../../ndarray", optional = true, default-features = false }
   [features]
   ndarray-hpc = ["dep:ndarray"]

2. Replace NdarrayFingerprint with actual ndarray::Fingerprint<256>
   - File: crates/lance-graph/src/graph/blasgraph/ndarray_bridge.rs
   - Change: standalone struct → re-export + From impls

3. Dedup ZeckF64: make ndarray canonical, thin wrapper in lance-graph
   - Delete: crates/lance-graph/src/graph/neighborhood/zeckf64.rs
   - Change: crates/lance-graph/src/graph/blasgraph/zeckf64.rs → re-export

4. Wire CAM-PQ: ndarray codec + lance-graph UDF + planner operator
   - ndarray impl: CamCodecContract from contract
   - lance-graph UDF: call ndarray impl via feature flag
   - planner: CamPqScanOp calls contract trait
```

### Path 4: Wire Planner to Core (INTEGRATION)
```
Priority: P2 (after P1 paths)
Effort:   5 days
Risk:     High (complex integration)

1. Wire CypherParse strategy to lance-graph's actual nom parser
   - Replace regex feature detection with real AST analysis
   - Preserve affinity scoring but use parsed features

2. Wire physical operators to DataFusion session
   - ScanOp → DataFusion TableScan
   - CamPqScanOp → DataFusion custom PhysicalExec
   - CollapseOp → DataFusion FilterExec with gate state

3. Wire elevation to execution timing
   - Wrap physical operators with timing measurement
   - Call should_elevate() between pipeline stages
   - Implement actual level transitions (retry at higher level)

4. Wire planner adjacency to actual AdjacencyStore
   - Build AdjacencyStore from lance-graph MetadataStore edges
   - Connect DPJoinEnum to real adjacency data
   - Connect TruthPropagation to real SPO truth values
```

### Path 5: Move bgz17 into Workspace (CODEC)
```
Priority: P2 (parallel with Path 4)
Effort:   2 days
Risk:     Low (bgz17 has 121 passing tests, 0 deps)

1. Move bgz17 from exclude to members in workspace Cargo.toml
2. Add bgz17-codec feature flag to lance-graph core
3. Wire plane_to_base17() bridge function
4. Wire PaletteSemiring as a planner semiring option
5. Add palette-accelerated path in FalkorCompat
```

### Path 6: N8N-RS Orchestration Contract (VISION)
```
Priority: P3 (after core integration)
Effort:   5 days
Risk:     High (cross-repo coordination)

1. n8n-rs n8n-contract depends on lance-graph-contract
2. Replace crew_router.rs + ladybug_router.rs with OrchestrationBridge
3. Replace CompiledStyleRegistry with contract StyleRegistry
4. Wire thinking mode dispatch through planner instead of HTTP routing
5. In single-binary mode: direct function calls (zero-serde)
6. In multi-process mode: Arrow Flight for cross-process calls
```

### Path 7: Adjacency Unification (FUTURE)
```
Priority: P4 (research-grade)
Effort:   Weeks
Risk:     High (architectural change)

1. Define AdjacencyView trait in contract crate
2. Implement for AdjacencyStore, GrBMatrix, PaletteCsr, ScentCsr
3. Unify traversal interface across all adjacency models
4. Enable cascade: ScentCsr → PaletteCsr → GrBMatrix → AdjacencyStore
5. Wire MKL batch ADC via PQ distance tables (E3)
6. Evaluate KuzuDB column-grouped CSR model for cold-path joins
```

## 5. RISK REGISTER

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| ndarray build broken | CRITICAL | CERTAIN | P0: fix before anything else |
| Arrow/DataFusion version mismatch | HIGH | LIKELY | Pin arrow=57, datafusion=51 across all repos |
| Contract adoption breaks consumers | MEDIUM | POSSIBLE | Feature-flag: consumers can opt-in gradually |
| Planner integration causes regressions | MEDIUM | POSSIBLE | Keep lance-graph core tests green at all times |
| FieldModulation semantic divergence | HIGH | CERTAIN | P1: pick ONE threshold formula, document it |
| bgz17 workspace inclusion breaks build | LOW | UNLIKELY | 0 deps, 121 tests already pass |
| KuzuDB adjacency synergy remains theoretical | LOW | CERTAIN | Proceed without it — our CSR is sufficient |
| jitson/cranelift dep missing from disk | HIGH | POSSIBLE | Check wasmtime fork availability |

## 6. STRATEGIC CONNECTIONS (Connecting the Dots)

### The Single Binary Vision
```
One binary containing:
  ladybug-rs (brain) + ndarray (foundation) + lance-graph-planner (intelligence) +
  lance-graph core (query engine) + bgz17 (codec) + crewai-rust (agents) +
  n8n-rs contract (orchestration)

All connected via:
  lance-graph-contract (zero-dep trait crate)

Communication:
  Internal: direct fn calls (zero-serde, zero-copy)
  External: Arrow Flight DoAction for cross-process
  User: Cypher/GQL/Gremlin/SPARQL through polyglot planner
```

### The COCA 4096 Wordlist Connection
The 4096 surface addresses in BindSpace × the COCA (Corpus of Contemporary American English) 4096 most frequent words = **one word per address**. Each word gets a 16Kbit fingerprint formed from its usage context. This turns the BindSpace into a **semantic CAM** (Content-Addressable Memory) where word lookup is O(1) by address and similarity search is O(n) by Hamming distance with scent pre-filter.

### The Fujitsu × Sensor × Zeckendorf Connection
The no-NaN integer-only codec stack means the entire hot path is:
- SIMD-friendly (no branch misprediction on NaN checks)
- Deterministic (same input → same output, always)
- Sensor-grade (can run on embedded/edge hardware without FPU)
- Zeckendorf-encoded (progressive precision, bandwidth-proportional quality)

This makes the system suitable for **edge deployment** where floating-point is expensive or unavailable.

### The BlasGraph × MKL × Neighborhood Connection
When ndarray has MKL enabled:
1. CAM-PQ batch ADC → cblas_sgemv (E3)
2. Palette distance matrix → cblas (256×17 L1 distance)
3. Neighborhood scent SpMV → mkl_sparse_s_mv
4. BLAS L3 gemm → matrixmultiply or MKL depending on feature

**All of these accelerate without any code changes** — the backend trait dispatch in ndarray handles it transparently.

---

## 10. POST-PR60 UPDATE (2026-03-29)

### Debt Items Resolved

| # | Item | Resolution | PR |
|---|------|-----------|-----|
| **W-resolved** | GraphSensorium only in q2 | **MIGRATED** to `arigraph/sensorium.rs` (539L) + `contract/sensorium.rs` (236L) | #59 |
| **W-resolved** | MetaOrchestrator only in q2 | **MIGRATED** to `arigraph/orchestrator.rs` (1562L) | #59 |
| **W-resolved** | Compilation errors in migration | **FIXED**: f32/f64 casts, TruthValue serde, EpisodicMemory capacity() | this session |

### New Assets from PR #59-60

| Asset | Lines | What It Adds |
|-------|-------|-------------|
| `contract/sensorium.rs` | 236 | GraphSignals, GraphBias, HealingType, SensoriumProvider + OrchestratorProvider traits |
| `arigraph/sensorium.rs` | 539 | GraphSensorium::from_graph(), compute(), suggested_bias(), diagnose_healing() |
| `arigraph/orchestrator.rs` | 1562 | MetaOrchestrator, MUL assessment, NARS RL topology, temperature, auto-heal |
| `session_unified_26_epiphanies.md` | 385 | 26 epiphanies cross-referenced, 11 paths, 5 agents, QA savant protocol |
| `session_master_integration.md` | ~1000 | 18 epiphanies → 6 layers → 42 hours estimate |
| `session_epiphany_integration.md` | ~800 | 14 epiphanies + priorities |
| `session_integration_plan.md` | ~600 | 9 phases, 40 hours, DeepNSM+AriGraph+SIMD+psychometrics |

### New Epiphanies (S1-S18, from session_unified_26_epiphanies.md)

- **S1**: Everything is a lookup table (SIMD dispatch, PQ, attention, NARS revision)
- **S2**: SPO decomposition IS attention (bgz-tensor connection)
- **S3**: SimilarityTable unifies scoring (1 table calibrates all distances)
- **S7**: CausalEdge64 = 8-byte everything (SPO + NARS + Pearl + plasticity + temporal)
- **S9**: Psychometric validation (measure meaning, not pattern)
- **S10**: Friston free energy (entropy = exploration fuel)
- **S15**: Bias as rotation vector (VSA unbind to debias)
- **S18**: NARS contradiction = awareness compass for navigation

### Updated Dependency Verification Matrix (from session_unified_26_epiphanies.md)

```
Path 1 (Foundation):     ndarray builds? ✓  tests pass? ✓
Path 2 (Contract):       sensorium traits added ✓  first impl (AriGraph) ✓  others? BLOCKED
Path 3 (ndarray wire):   ndarray dep wired ✓  From impls ✓  ZeckF64 dedup? NOT STARTED
Path 4 (Planner wire):   classify_query wired ✓  full pipeline? NOT STARTED
Path 5 (bgz17):          121 tests ✓  in workspace? NOT STARTED
Path 8 (DeepNSM↔Ari):   deepnsm compiles (38 tests) ✓  entity resolution? NOT STARTED
Path 9 (Causal↔Ari):     causal-edge exists ✓  palette assignment? NOT STARTED
Path 10 (Psychometrics): CONCEPT DOCUMENTED, NOT STARTED
Path 11 (Cartography):   VISION DOCUMENTED, NOT STARTED
```

*This document is the map. The territory is the code. Walk the code with this map and you'll never get lost.*
