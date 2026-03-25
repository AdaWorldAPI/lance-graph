# Thinking Microcode Pipeline

## The Chain

```
YAML template (36 files, one per style)
  → serde_yaml parse → ThinkingTemplate struct
  → template_to_jitson() → JitsonTemplate
  → JitEngine::compile() → ScanKernel (Cranelift → native x86_64)
  → LazyLock freeze → 1 pointer deref at runtime (~5ns)
  → Execute on query → result + quality measurement
  → NARS revision → topology edge update
  → Persist topology → next startup skips known pairs
```

## What Already Exists

| Component | File | Status |
|-----------|------|--------|
| JitsonTemplate parser | `ndarray/src/hpc/jitson/template.rs` | ✅ done |
| PrecompileQueue | `ndarray/src/hpc/jitson/precompile.rs` | ✅ done |
| JitEngine (Cranelift) | `ndarray/src/hpc/jitson_cranelift/engine.rs` | ✅ done |
| LazyLock pattern | `ndarray/src/hpc/simd_caps.rs` | ✅ done |
| cam_pq cascade template | `lance-graph/src/cam_pq/jitson_kernel.rs` | ✅ done |
| ThinkingStyle enum (12) | `planner/src/thinking/style.rs` | ✅ done (but hardcoded) |
| ThinkingStyle enum (36) | `contract/src/thinking.rs` | ✅ done (but no YAML) |
| FieldModulation + fingerprint | `contract/src/thinking.rs` | ✅ done |
| ThinkingTopology (NARS edges) | `planner/src/thinking/topology.rs` | ✅ built today |
| observe_coactivation() | `planner/src/thinking/topology.rs` | ✅ built today |
| CognitiveProcess (verbs) | `planner/src/thinking/process.rs` | ✅ built today |
| **YAML loading** | — | ❌ missing |
| **Style → JitsonTemplate** | — | ❌ missing |
| **LazyLock ThinkingKernelCache** | — | ❌ missing |
| **Bruteforce NARS RL loop** | — | ❌ missing |
| **Semantic affinity in YAML** | — | ❌ missing |
| **Difficulty affinity in YAML** | — | ❌ missing |

## NARS Reinforcement Learning

Not gradient descent. Not backpropagation. NARS revision:

```
1. Try style pair (A, B) on query Q
2. Measure quality_delta = quality_with_B - quality_without_B  
3. observe_coactivation(A, B, delta, relation)
4. NARS revision: f_new, c_new from evidence merge
5. Repeat 1,296 pairs × 100 queries = ~1.3 seconds total
6. Topology converges: high-c edges are stable, low-c keep exploring
```

## Texture Matching

```
Query PoS distribution = "texture"
  "I think I know what you feel" → 60% mental predicates
  "the big dog bit the old man" → 43% nouns, 29% descriptors

Style YAML semantic_affinity = "resonance profile"
  analytical.yaml: mental_predicates: 0.8, evaluators: 0.3
  metacognitive.yaml: mental_predicates: 0.95, evaluators: 0.5

Match: dot_product(texture, affinity) → style_score
  Highest scoring style activates first
  If top-2 within 0.1 → fork both (topology tension)
```
