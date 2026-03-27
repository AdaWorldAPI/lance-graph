# Type Duplication Map: Every Copy, Every Line

> *Rain Man precision. If you see two things that look alike, they're listed here.*

## 1. Fingerprint / BitVec (16,384-bit binary vector)

The foundational type — 256 × u64 words = 2048 bytes = 16,384 bits. **Four copies exist.**

| # | Location | Type Name | Definition | Notes |
|---|----------|-----------|-----------|-------|
| 1 | `ndarray/src/hpc/fingerprint.rs:21` | `Fingerprint<const N: usize>` | `pub words: [u64; N]` | **CANONICAL.** Const-generic, supports 128/256/1024. |
| 2 | `lance-graph/crates/lance-graph/src/graph/blasgraph/types.rs` | `BitVec` | `pub type BitVec = [u64; VECTOR_WORDS]` where VECTOR_WORDS=256 | BlasGraph internal. Raw array, no struct. |
| 3 | `lance-graph/crates/lance-graph/src/graph/blasgraph/ndarray_bridge.rs:28` | `NdarrayFingerprint` | `pub words: [u64; VECTOR_WORDS]` | **Standalone mirror** of ndarray's `Fingerprint<256>`. No dependency. |
| 4 | `ladybug-rs/src/storage/bind_space.rs` | `[u64; FINGERPRINT_WORDS]` | inline in BindNode struct | FINGERPRINT_WORDS = 256. Same layout. |

**Impact**: Zero-copy conversion is possible (same memory layout), but no `From` impls exist.
**Fix**: ndarray `Fingerprint<256>` becomes canonical. All others get `From<&Fingerprint<256>>` or become type aliases.

## 2. ZeckF64 (8-byte progressive SPO edge encoding)

**Three copies, one of which is internal duplication within lance-graph.**

| # | Location | Function | Constants | Notes |
|---|----------|----------|----------|-------|
| 1 | `ndarray/src/hpc/zeck.rs:55` | `zeckf64_from_distances(ds, dp, d_o) -> u64` | D_MAX=16384, THRESHOLD=8192 | "Ported from lance-graph" — comment at line 14 |
| 2 | `lance-graph/crates/lance-graph/src/graph/blasgraph/zeckf64.rs:58` | `zeckf64(a, b, sign, threshold) -> u64` | D_MAX=16384, DEFAULT_THRESHOLD=8192 | Takes BitVec triples directly |
| 3 | `lance-graph/crates/lance-graph/src/graph/neighborhood/zeckf64.rs` | Same as #2 | Same | **Exact duplicate** of blasgraph/zeckf64.rs |

**Impact**: Bug fixes must be applied to all three. ndarray version has extra batch + top-k ops.
**Fix**: ndarray version is canonical (has batch ops). lance-graph re-exports or thin-wraps.

## 3. CLAM Tree (Divisive Hierarchical Clustering)

**Three copies — but with different roles.**

| # | Location | Purpose | Full impl? |
|---|----------|---------|-----------|
| 1 | `ndarray/src/hpc/clam.rs` | Full CLAM tree: build, search, rho_nn, knn_brute, LFD. 46 tests. | **YES — CANONICAL** |
| 2 | `lance-graph/crates/lance-graph/src/graph/blasgraph/clam_neighborhood.rs` | Conjecture test only: do CLAM radii land on Pareto levels? | Metrics only, no tree build |
| 3 | `lance-graph/crates/lance-graph/src/graph/neighborhood/clam.rs` | **Exact duplicate** of #2 | Same metrics only |

**Impact**: ladybug-rs already calls `ndarray::hpc::clam::ClamTree` directly. lance-graph copies are test stubs.
**Fix**: lance-graph copies stay as conjecture validators. No dedup needed — they serve different purposes.

## 4. Base17 (Golden-Step 17D Encoding)

**Two copies.**

| # | Location | Type | Notes |
|---|----------|------|-------|
| 1 | `lance-graph/crates/bgz17/src/base17.rs` | `Base17 { dims: [i16; 17] }` | **ORIGINAL.** Full VSA ops: xor_bind, bundle, permute |
| 2 | `ndarray/src/hpc/bgz17_bridge.rs:33` | `Base17 { dims: [i16; 17] }` | "Self-contained port" — same struct, same encode logic |

Plus related types:
| 2a | `ndarray/src/hpc/bgz17_bridge.rs:39` | `SpoBase17 { subject, predicate, object }` | Triple of Base17, 102 bytes |
| 2b | `ndarray/src/hpc/bgz17_bridge.rs:46` | `PaletteEdge { s_idx, p_idx, o_idx }` | 3-byte compressed SPO |

**Impact**: Semantically identical but no `From` impls between them.
**Fix**: bgz17 crate becomes canonical when moved into workspace. ndarray bridge becomes a re-export.

## 5. CAM-PQ (6-Byte Product Quantization Codec)

**Three definitions — contract, ndarray implementation, and planner operator.**

| # | Location | What | Notes |
|---|----------|------|-------|
| 1 | `lance-graph-contract/src/cam.rs:17` | `CamByte` enum, `CamCodecContract` trait | **CONTRACT** — the single source of truth for the interface |
| 2 | `ndarray/src/hpc/cam_pq.rs:41` | `CamByte` enum, `CamCodebook`, `CamFingerprint`, encode/decode/train | **IMPLEMENTATION** — full codec with codebook training |
| 3 | `lance-graph-planner/src/physical/cam_pq_scan.rs` | `CamPqScanOp`, `CamPqStrategy` | **OPERATOR** — physical plan node, uses contract interface |

**Status**: The CamByte enum is defined in both contract and ndarray. They're identical.
**Fix**: ndarray should `pub use lance_graph_contract::cam::CamByte` when contract is a dep.

## 6. ThinkingStyle (Cognitive Mode Selector)

**Four definitions that the contract was specifically created to unify.**

| # | Location | Variants | Notes |
|---|----------|----------|-------|
| 1 | `lance-graph-contract/src/thinking.rs:23` | 36 styles, 6 clusters, τ addresses 0x20-0xC5 | **CANONICAL.** Zero-dep. |
| 2 | `lance-graph-planner/src/thinking/style.rs:16` | 12 styles, 4 clusters, τ addresses 0x20-0xC0 | **PARALLEL.** Planner's own, not using contract yet! |
| 3 | `n8n-rs/n8n-rust/crates/n8n-contract/src/thinking_mode.rs:36` | ThinkingMode { inference_type, cam_top_k, beam_width, ... } | Different struct, maps to 5 InferenceTypes |
| 4 | `n8n-rs/n8n-rust/crates/n8n-contract/src/compiled_style.rs:38` | CompiledStyle { kernel, params, name, tau } | JIT-compiled version, references jitson |

**Critical**: The contract crate (copy #1) exists specifically to replace copies #2-4, but **none of them actually depend on it yet.**

## 7. FieldModulation (Thinking Style → Scan Parameters)

**Three definitions.**

| # | Location | Fields | Notes |
|---|----------|--------|-------|
| 1 | `lance-graph-contract/src/thinking.rs:160` | 7D: resonance_threshold, fan_out, depth_bias, breadth_bias, noise_tolerance, speed_bias, exploration | **CANONICAL** |
| 2 | `lance-graph-planner/src/thinking/style.rs:151` | Same 7D fields, slightly different defaults | **PARALLEL** — should import from contract |
| 3 | `n8n-rs/n8n-contract/src/compiled_style.rs` | References jitson ScanParams directly | Different abstraction level |

**to_scan_params()** implementations:
- Contract: `threshold = resonance × 1000`
- Planner: `threshold = resonance × 2000 + 100`
- n8n-rs: goes through jitson directly

**Impact**: Three different threshold mappings from the same conceptual modulation. This is a **semantic divergence**, not just a copy.

## 8. NARS InferenceType

**Three copies.**

| # | Location | Variants | Notes |
|---|----------|----------|-------|
| 1 | `lance-graph-contract/src/nars.rs:12` | Deduction, Induction, Abduction, Revision, Synthesis | **CANONICAL** |
| 2 | `lance-graph-planner/src/thinking/nars_dispatch.rs:7` | NarsInferenceType: same 5 variants | **PARALLEL** |
| 3 | `n8n-rs/n8n-contract/src/thinking_mode.rs:37` | ThinkingMode.inference_type: InferenceType enum | **PARALLEL** |

## 9. SemiringChoice (Which Algebra to Use)

**Three definitions, one in contract, one in planner, one in planner-nars.**

| # | Location | Variants | Notes |
|---|----------|----------|-------|
| 1 | `lance-graph-contract/src/nars.rs:58` | Boolean, HammingMin, NarsTruth, XorBundle, CamPqAdc | **CANONICAL** |
| 2 | `lance-graph-planner/src/thinking/semiring_selection.rs:12` | SemiringChoice { semiring: SemiringType, rationale } | Wrapper with rationale string |
| 3 | `lance-graph-planner/src/ir/logical_op.rs` | SemiringType enum | IR-level representation |

**Plus the actual semiring implementations** (see SEMIRING_ALGEBRA_SURFACE.md):
- `lance-graph/blasgraph/semiring.rs`: 7 HdrSemirings
- `lance-graph/spo/semiring.rs`: TruthSemiring
- `bgz17/palette_semiring.rs`: SpoPaletteSemiring
- `lance-graph-planner/physical/accumulate.rs`: TruthPropagatingSemiring

## 10. Cypher Parser + AST

**Two copies.**

| # | Location | Parser | AST | Tests |
|---|----------|--------|-----|-------|
| 1 | `lance-graph/crates/lance-graph/src/parser.rs` + `ast.rs` | 1,932L nom | 544L | 44+ tests | **CANONICAL** |
| 2 | `ladybug-rs/src/query/lance_parser/` (parser.rs + ast.rs + error.rs + mod.rs + semantic.rs) | "Stolen from lance-graph" (comment at line 1) | Same AST | — |

**Impact**: ladybug-rs copy was made before lance-graph was a dep. Now obsolete per user's statement.
**Fix**: ladybug-rs can either depend on lance-graph directly or keep the frozen copy as P3 parser.

## 11. TruthValue (NARS Belief Strength)

**Four definitions.**

| # | Location | Fields | Notes |
|---|----------|--------|-------|
| 1 | `lance-graph-planner/src/nars/truth.rs:11` | `frequency: f32, confidence: f32` | Planner's own |
| 2 | `lance-graph/src/graph/spo/truth.rs` | TruthValue enum (Open/Normal/Strong) + frequency/confidence | Different shape — enum + fields |
| 3 | `ndarray/src/hpc/bf16_truth.rs` | BF16Weights + TruthValue | BF16-specific representation |
| 4 | `ndarray/src/hpc/causality.rs` | NarsTruthValue | Another variant |

**Impact**: Four different representations of the same concept. The contract's `nars.rs` doesn't define TruthValue at all — it only defines InferenceType and SemiringChoice.

## 12. CSR Adjacency (Compressed Sparse Row)

**Five implementations.**

| # | Location | Type Name | Edge Weight | Purpose |
|---|----------|-----------|-------------|---------|
| 1 | `lance-graph-planner/src/adjacency/csr.rs:12` | `AdjacencyStore` | u64 edge IDs → EdgeProperties | **Kuzu-style columnar CSR/CSC** |
| 2 | `lance-graph/blasgraph/neighborhood_csr.rs:17` | `ScentCsr` | `u8` scent bytes | BFS/PageRank on scent vectors |
| 3 | `lance-graph/blasgraph/sparse.rs` | `CsrStorage<T>` | Generic T | GraphBLAS sparse matrix |
| 4 | `bgz17/palette_csr.rs:20` | `PaletteCsr` | Archetype assignments | O(k²) palette search |
| 5 | `ndarray` (indirect via `binding_matrix.rs`) | Dense 256³ matrix | u64 popcount | Not CSR, but adjacency-like |

**Impact**: Five different adjacency representations with no shared interface.
**Epiphany**: All of these could implement a single `AdjacencyView` trait from the contract crate.

## Summary: Duplication Heat Map

```
TYPE                  COPIES    SEVERITY    FIX COMPLEXITY
─────────────────────────────────────────────────────────
Fingerprint/BitVec       4      HIGH        Medium (same layout, needs From impls)
ZeckF64                  3      HIGH        Easy (ndarray canonical, re-export)
ThinkingStyle         4(!)      CRITICAL    Hard (contract exists but nobody uses it)
FieldModulation          3      HIGH        Medium (semantic divergence in thresholds)
InferenceType            3      HIGH        Easy (contract canonical)
CLAM                     3      LOW         None needed (different purposes)
Base17                   2      MEDIUM      Easy (bgz17 canonical when in workspace)
CamByte                  2      LOW         Easy (re-export from contract)
Cypher Parser            2      MEDIUM      Debatable (stolen copy may be frozen P3)
TruthValue               4      HIGH        Hard (different representations)
CSR Adjacency            5      MEDIUM      Hard (different purposes, shared trait?)
SemiringChoice           3      MEDIUM      Easy (contract → planner re-export)
```
