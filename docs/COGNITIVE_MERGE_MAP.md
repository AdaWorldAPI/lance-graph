# Cognitive Module Merge Map

> Where lance-graph-cognitive imports overlap with existing lance-graph code,
> and how to merge them cleanly.

## Merge Targets (overlap exists, extend don't duplicate)

| Cognitive import | Existing lance-graph | Action |
|---|---|---|
| grammar/qualia.rs (18D) | contract/mul.rs (partial) | Extend MUL with 18D qualia coords |
| learning/scm.rs | planner/nars/ + cache/nars_engine | Wire to contract NarsTruth |
| spo/gestalt.rs | planner/collapse_gate | Extend with GREEN/AMBER/RED/BLUE |
| learning/cognitive_styles.rs | contract/thinking.rs (36 styles) | Add RL adaptation layer |

## Clean Additions (no overlap)

| Module | What it adds |
|---|---|
| learning/dream.rs | Offline prune/merge/permute-XOR-bind consolidation |
| learning/quantum_ops.rs | Fingerprint operator algebra |
| spo/sentence_crystal.rs | Text → 5D crystal → fingerprint |
| spo/context_crystal.rs | 5×5×5 SPO×qualia×temporal cube |
| spo/cognitive_codebook.rs | 16-bit bucket + 48-bit hash unified encoding |
| spo/meta_resonance.rs | Cleanup memory for noisy retrieval |
| grammar/causality.rs | Linguistic WHO→DID→WHAT→WHY (not graph-level causal) |
| grammar/nsm.rs | Natural Semantic Metalanguage primes |
| grammar/triangle.rs | SPO × causality × qualia vertex |
| world/counterfactual.rs | What-if reasoning |

## No Conflict (different levels)

| Cognitive | lance-graph | Why no conflict |
|---|---|---|
| grammar/causality (linguistic) | causal-edge (graph-level) | Different abstraction level |
| spo/ crystal extensions | lance-graph/graph/spo/ store | Extensions, not replacement |
| cognitive_styles (RL) | contract/thinking (static) | RL extends static styles |

## Dead References to Fix

| Pattern | In | Replace with |
|---|---|---|
| `crate::core::Fingerprint` | all cognitive modules | `ndarray::hpc::fingerprint::Fingerprint<256>` |
| `crate::core::rustynum_accel::*` | gestalt.rs | `ndarray::hpc::bitwise::*` |
| `crate::core::VsaOps` | quantum_ops.rs | direct XOR/bundle on `Fingerprint<256>` |
| `crate::FINGERPRINT_BITS` | many | `crate::FINGERPRINT_BITS` (already added to lib.rs) |
| `crate::FINGERPRINT_U64` | many | `crate::FINGERPRINT_U64` (already added to lib.rs) |
| `crate::simd::*` (holograph) | holograph internals | `ndarray::simd::*` (F32x16, etc.) |
| `rustynum_accel::simd_level` | gestalt.rs | `ndarray::hpc::simd_caps::simd_caps()` |

## Wiring to Existing Crates

| Cognitive module | Should wire to |
|---|---|
| gestalt.rs (per-plane Hamming) | blasgraph SPO semiring |
| sentence_crystal.rs (Jina embedding) | bgz-tensor projection |
| cognitive_codebook.rs (Base17) | bgz17 palette |
| scm.rs (NARS edges) | causal-edge CausalEdge64 |
| meta_resonance.rs (cleanup) | holograph resonance VectorField |
| dream.rs (XOR-bind recombine) | ndarray Fingerprint<256> XOR |
