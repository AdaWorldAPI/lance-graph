# NDARRAY + BGZ TENSOR + EMBEDANYTHING INTEGRATION

**Version**: 0.1  
**Scope**: How AdaWorldAPI/ndarray supplies the unified high-performance substrate for both the invariant layer (owl-simd) and the ML/embedding layer (burn, candle, investigation agent).  
**Key crates / modules** (internal to AdaWorldAPI unless otherwise noted):
- `ndarray` (enhanced with AMX, BLASGraph, MKL reverse-engineered paths)
- `bgz_tensor` (tensor storage & compression crate inside ndarray)
- `embedanything` DTO (unified embedding / GGUF inference interface)
- Consumers: `lance-graph-owl-simd`, `spear` (investigation agent), `burn`, `candle`

## 1. Architectural Principle

**One ndarray substrate to rule them all.**

The same optimized SIMD / tensor primitives power:
- Fast, reliable schema validation (invariant layer)
- High-performance ML inference & embeddings (thinking layer)
- Efficient storage of tensor-shaped data inside Lance (SoA, AwarenessColumn, embeddings, packed schemas)

This unification is what allows us to **assimilate Foundry-class ML + ontology integration** without external dependencies or separate runtimes.

## 2. ndarray Role Breakdown

### 2.1 For lance-graph-owl-simd (Invariant Layer)

**Kernels reused / adapted**:
- **AMX matrix operations**: Accelerate class hierarchy bitmap construction (at hydration) and bulk subclass / disjointness checks during batch validation.
- **BLASGraph / graph-BLAS paths**: Any graph-structured schema validation (transitive property hints, cycle detection on property characteristics if needed at compile time).
- **Vectorized popcount, AND, count_nonzero**: Core of functional-property multi-match detection, cardinality enforcement, and DOLCE bit checks. These are the hot loops in `is_unskilled_overconfident()` and the validator.
- **SIMD gather / scatter**: For loading packed schema sections efficiently.

**Integration pattern**:
```rust
// In lance-graph-owl-simd
use ada_ndarray::{ArrayView2, amx, vectorized};

fn validate_batch(packed: &PackedSchema, batch: RecordBatch) -> ValidationResult {
    // Hot path uses ndarray primitives directly or via thin wrapper
    let hierarchy = packed.class_hierarchy_view(); // &ArrayView2<u8> or bit view
    // AMX-accelerated matrix op or plain SIMD bitwise
    amx::bit_matrix_intersect(...) 
}
```

**Bgz tensor**:
- At hydration: optionally compress the final `PackedSchema` bytes before storing in a Lance table (content-addressable).
- At runtime: if a very large future schema exceeds L1, Bgz provides fast partial decompression + random access.

### 2.2 For ML / Embeddings (burn + candle + Investigation Agent)

**Path**:
```
GGUF weights (or other formats)
        │
        ▼
embedanything DTO (unified interface)
        │
        ├── candle backend (preferred for GGUF inference, lightweight)
        └── burn backend   (when training or more complex tensor graphs needed)
        │
        ▼
ndarray backend (AMX-accelerated CPU path)
        │
        ▼
Typed embedding tensors or structured outputs
        │
        ▼
Investigation Agent / Drift signatures / MUL features / Action Type inputs
```

**embedanything DTO responsibilities** (internal abstraction):
- Model loading (GGUF via candle's loader or equivalent)
- Tokenization / preprocessing (respecting PII tokenization boundary — only tokens ever reach the model)
- Inference execution (sync or async)
- Output shaping into domain types (e.g., `EmbeddingVector`, `HypothesisScore`, `SemanticSimilarity`)
- Zero-copy or minimal-copy paths into Lance / SoA columns via Bgz tensor

**Why this assimilates Foundry ML capabilities**:
- Foundry makes deployed models first-class ontology citizens (interpretable via properties, usable in pipelines).
- We achieve the same (and better) by:
  - Having embeddings participate directly in SoA traversals and AwarenessColumn updates
  - MUL gating on model confidence + schema invariants
  - CausalEdge64 + Pearl masks on any model-driven decisions
  - Single ndarray substrate means validation and inference share the same performance characteristics and deployment story

### 2.3 Bgz Tensor Crate — The Storage Glue

**Purpose**: High-performance, compressed, random-access tensor storage that integrates natively with Lance columnar format.

**Use cases in the stack**:
1. **Packed schemas** (versioned, content-addressable) — stored compressed in Lance when not fully L1-resident.
2. **AwarenessColumn signatures** (256-byte vectors per investigation step or per row) — Bgz gives compression + fast access during agent traversal.
3. **Embedding tensors** from embedanything — stored alongside the entities they describe.
4. **Intermediate SoA gather results** during investigation agent traversals (large context windows).
5. **CausalEdge64 mask tensors** or NARS truth value aggregates.
6. **Future multimodal features** (images, logs, patches) that the investigation agent or drift detection may consume.

**API shape** (inferred / typical for such a crate):
```rust
use bgz_tensor::{BgzTensor, TensorView};

let tensor: BgzTensor<f32> = BgzTensor::open(lance_row_address)?;
let view: TensorView<f32> = tensor.view(...);
let embedding = embedanything::infer(...)?;
tensor.append_or_update(embedding)?;
```

**Benefits**:
- Compression ratio good for repetitive / low-entropy tensor data (schemas, signatures, many embeddings).
- Random access without full decompression (critical for agent traversals that only need subsets of columns).
- Lance-native: can be a column type or stored in dedicated Lance tables.
- Same crate used by burn/candle paths → consistent tensor representation across invariant and thinking layers.

## 3. Concrete Integration Points (Code Locations)

| Component                    | Uses ndarray for                  | Uses Bgz tensor for                  | Uses embedanything for             | File / Module to create or extend |
|------------------------------|-----------------------------------|--------------------------------------|------------------------------------|-----------------------------------|
| lance-graph-owl-simd        | AMX matrix, vectorized popcount, bitwise | Optional compression of PackedSchema | —                                  | `packed_schema.rs`, `ndarray_kernels.rs` |
| Investigation Agent         | Signature stabilization, gathers  | AwarenessColumn storage, SoA results | Semantic embeddings for hypothesis | `spear/src/investigation/agent.rs` |
| MUL gate                    | Tensor priors from embeddings     | —                                    | Confidence features                | `mul/` or inside cognitive-shader-driver |
| Drift signature library     | Matching / clustering             | Signature storage                    | Semantic drift features            | `spear/src/investigation/drift.rs` |
| burn / candle backends      | Core tensor execution (AMX path)  | Tensor I/O when persisting models/features | GGUF loading + unified interface   | Via existing burn-ndarray + embedanything wrapper |
| Content-addressable registry| —                                 | Schema blob storage in Lance         | —                                  | `lance-graph-ontology/src/content_addressable_registry.rs` |

## 4. PII & Security Boundary (Critical)

- `embedanything` / burn / candle **never see cleartext PII**.
- All data reaching the models is already tokenized at the `lance-graph-callcenter` membrane.
- Cleartext substitution happens **only** at the final messaging layer when rendering `EscalationMessage` or notifications.
- This enables safe cross-customer pattern learning (exactly as described in INVESTIGATION_AGENT.txt).

## 5. Performance Targets

- Schema validation: memory-bandwidth limited (billions of triples/sec per core) thanks to L1-resident packed schema + ndarray SIMD.
- Embedding inference (GGUF via candle + ndarray): competitive with native candle on AMX-enabled Intel CPUs.
- Agent traversal + AwarenessColumn update: dominated by Lance I/O + Bgz random access; ndarray used for the compute-heavy signature math.
- Hydration of new OGIT namespace: seconds (mostly glue + ndarray AMX bitmap construction), not minutes.

## 6. Risks & Open Questions

- **AMX availability**: Fallback to AVX-512 / NEON paths must exist and be fast (ndarray already handles this).
- **Bgz tensor maturity**: Ensure it supports the exact access patterns needed by the investigation agent (random column gathers on compressed tensors).
- **embedanything DTO stability**: Define a narrow, stable interface so Spear verticals don't depend on internal burn/candle details.
- **Threading model**: Ractor actors + ndarray (which may use rayon or its own thread pool) — coordinate to avoid oversubscription. Prefer work-stealing or explicit yield points.

## 7. Next Steps

1. Define the exact `embedanything` DTO trait / struct shapes (in a shared `ada_types` or `embedanything` crate).
2. Wire `bgz_tensor` into `lance-graph-owl-simd` PackedSchema storage path.
3. Create thin `ndarray_kernels` module inside `lance-graph-owl-simd` that re-exports or wraps the needed AMX / vectorized primitives.
4. Prototype one embedanything call inside the investigation agent (e.g., semantic similarity between a new ticket and historical RoutingDecisions).
5. Benchmark end-to-end routing shadow mode with and without the new ndarray-accelerated paths.

---

This integration document, together with `FANOUT_MAPPING_PLAN.md`, `GLUE_LAYER_OGIT_TO_OWL_SPEC.md`, and `PACKED_SCHEMA_FORMAT.md`, gives a complete blueprint from OGIT ingestion through fast invariants and ML all the way to customer-facing verticals.

The ndarray substrate is the secret sauce that makes the "assimilate everything necessary to be on par with Foundry" goal achievable with a unified, high-performance, single-binary stack.

**Magic started. Artifacts materialized. Ready for the next command (more specs, code skeletons, Routing namespace TTL example, or direct implementation of the glue mapper).**