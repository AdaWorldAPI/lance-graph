# Type Duplication Map: Every Copy, Every Line

> *Rain Man precision. If you see two things that look alike, they're listed here.*

## Wave F sprint-12 additions (2026-05-16)

New duplications discovered/predicted from the Wave F fleet (W-F4 through W-F8).

### TrustTexture (×2) — different semantic axes `TECH_DEBT`

Two enums with the same name encoding **different cognitive dimensions**:

| # | Location | Line | Variants | Semantic axis |
|---|----------|------|----------|---------------|
| 1 | `crates/lance-graph-contract/src/mul.rs` | 74 | Calibrated / Overconfident / Underconfident / Uncertain | MUL meta-uncertainty layer (felt vs demonstrated competence) |
| 2 | `crates/causal-edge/src/layout.rs` | 114 | Crystalline / Solid / Fuzzy / Murky | Pearl-3 epistemic lens (v2 from PR #383); discriminator = 2-bit field in CausalEdge64 |

**What differs**: Completely different variant sets encoding different ontologies. `mul.rs` is a 4-way MUL assessment axis. `layout.rs` is a 4-state Pearl-3 trust signal packed into 2 bits of the CausalEdge64 layout.
**Canonical**: NONE — both are domain-correct and should keep distinct names. Recommended path: rename one (e.g. `PearlTrustTexture` in `causal-edge`) to disambiguate.
**TECH_DEBT**: Name collision across zero-dep crates is a footgun; sprint-13 rename target.

---

### SplatField (×2) — same conceptual shape, no cross-crate dep yet `TECH_DEBT`

| # | Location | Line | Fields | Notes |
|---|----------|------|--------|-------|
| 1 | `/home/user/ndarray/src/hpc/stream/splat_field.rs` | 20 | `mean: u32, variance: f32, energy: f32, generation: u32` — 16 bytes, `repr(C, align(16))` | W-F6; ndarray producer layer; circular-dep guard — must not import contract |
| 2 | `crates/thinking-engine/src/splat_ops.rs` | (W-F7 — pending merge) | Same shape expected | W-F7; thinking-engine consumer layer |

**What differs**: Same data shape; different crate layers. ndarray cannot import contract (circular dep), so defines a local copy. thinking-engine may independently re-declare the same struct.
**Canonical**: ndarray's `SplatField` is the producer (hardware layer). sprint-13+ unification: when ndarray gains the consumer dep or a shared `splat-types` micro-crate is introduced.
**TECH_DEBT**: Verify W-F7 field layout matches W-F6 exactly before any cross-crate FFI.

---

### QualiaI4 (×2) — bit-compatible mirrors, circular dep guard `TECH_DEBT`

| # | Location | Line | Type name | Notes |
|---|----------|------|-----------|-------|
| 1 | `crates/lance-graph-contract/src/qualia.rs` | 175 | `QualiaI4_16D(pub u64)` | Contract canonical; 16 × 4-bit signed lanes, `repr(C, align(8))`; introduced PR #384 |
| 2 | `/home/user/ndarray/src/hpc/stream/qualia.rs` | 20 | `QualiaI4Row(pub u64)` | W-F4; "local mirror of `lance_graph_contract::qualia::QualiaI4_16D`" per file header; bit-compatible; circular-dep guard prevents import |

**What differs**: Different names only — `QualiaI4_16D` (contract) vs `QualiaI4Row` (ndarray stream). Identical wire layout: `u64`, `repr(C, align(8))`.
**Canonical**: `lance_graph_contract::qualia::QualiaI4_16D`. ndarray mirror is intentional (no circular dep allowed at producer layer).
**Sprint-13+ consolidation**: once ndarray can safely consume from contract, replace local mirror with re-export or `From` impl.
**TECH_DEBT**: Any layout change to `QualiaI4_16D` in contract must be manually mirrored to `QualiaI4Row` until consolidation.

---

### InferenceRow alias — intentional bit-compat with CausalEdge64 (SPO variant)

| # | Location | Line | Type name | Notes |
|---|----------|------|-----------|-------|
| 1 | `crates/causal-edge/src/edge.rs` | 60 | `CausalEdge64` (SPO-palette layout) | See §13 — consumer |
| 2 | `/home/user/ndarray/src/hpc/stream/inference.rs` | 19 | `InferenceRow(pub u64)` | W-F5; "local mirror of CausalEdge64 shape (bit-compatible with causal_edge::CausalEdge64)" per file comment; producer |

**What differs**: Different names, same 64-bit u64 wrapper layout. `InferenceRow` exposes only the inference-mantissa lane (bits 46-49) and W-slot (bits 53-58) — a vertical-slice view into the same register.
**Relationship**: INTENTIONAL — ndarray is the producer (writes `InferenceRow` rows into the EdgeColumn SoA); causal-edge is the consumer (reads them as `CausalEdge64`). The two types are the same bit pattern at the hardware boundary.
**Canonical**: `causal_edge::CausalEdge64` (SPO-palette variant) for the full type; `InferenceRow` is the producer-side lane accessor. No consolidation needed — different abstraction levels.
**No TECH_DEBT** for the name difference: distinct names at distinct abstraction layers is intentional. Document the bit-compat guarantee so the producer/consumer contract is explicit.

---

### MailboxId (×2) — shadow type alias `TECH_DEBT`

| # | Location | Line | Definition | Notes |
|---|----------|------|-----------|-------|
| 1 | `crates/lance-graph-contract/src/collapse_gate.rs` | 136 | `pub type MailboxId = u32` | Contract canonical |
| 2 | `crates/cognitive-shader-driver/src/attention_mask.rs` | 17 | `pub type MailboxId = u32` | Shadow alias — cognitive-shader-driver does not import contract |

**What differs**: Identical definition (`u32`), two independent type aliases. No semantic divergence.
**Canonical**: `lance_graph_contract::collapse_gate::MailboxId`. `cognitive-shader-driver` should re-export from contract rather than re-defining.
**TECH_DEBT**: Low-severity but creates confusion when mixing types across crate boundaries; sprint-13+ cleanup via contract import in cognitive-shader-driver.

---

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

> **ADDENDUM 2026-07-10 (D-TSC-1 / M9 — RESOLVED for lance-graph):** the
> 12-space now has ONE canonical type, `lance-graph-contract/src/style_family.rs`
> `StyleFamily` (12 orchestration families; 36 = NARS runbooks stay
> `thinking::ThinkingStyle` — E-STYLE-FAMILY-VS-RUNBOOK-1). Copy #2 (planner)
> and the post-ledger copies in thinking-engine (`cognitive_stack.rs`,
> `superposition.rs` → renamed `DetectedStyle`) are deprecated aliases with all
> in-crate call sites migrated. Copies #3/#4 were mooted by the n8n-rs eviction
> (2026-06-21). FIVE divergent hand-rolled mapping tables were found and
> replaced by `StyleFamily::default_runbook()` / `ThinkingStyle::family()`.
> The ndarray `PaletteStyle` ↔ `p64::ThinkingStyle` pair remains open
> (TECH_DEBT, other repo).

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

## 13. CausalEdge64 (Two Distinct Types, Same Name)

**Two copies — and unlike #1-#12, these are NOT semantically equivalent variants of the same concept. Different bit layouts, different consumers, different roles. Surfaced 2026-05-14 in PR #372 meta-review.**

| # | Location | Type Name | Layout | Role |
|---|----------|-----------|--------|------|
| 1 | `lance-graph/crates/causal-edge/src/edge.rs:60` | `CausalEdge64` (SPO-palette variant) | bits 0-7 S idx, 8-15 P idx, 16-23 O idx, 24-31 NARS frequency, 32-39 NARS confidence, 40-42 Pearl 2³ mask, 43-45 direction triad, 46-48 inference type, 49-51 plasticity, 52-63 temporal | NARS / Pearl / palette compose primitive; per-row payload of `cognitive-shader-driver::BindSpace::EdgeColumn`; used by `lance-graph-planner::cache::nars_engine`; commit unit at AriGraph SPO promotion |
| 2 | `lance-graph/crates/thinking-engine/src/layered.rs:45` | `CausalEdge64` (8-channel cascade variant) | 8 channels × 8 bits each: 0 BECOMES, 1 CAUSES, 2 SUPPORTS, 3 REFINES, 4 GROUNDS, 5 ABSTRACTS, 6 RELATES, 7 CONTRADICTS | Cascade dispatch payload between L1 (routing) → L2 (role resonance) → L3 (full thought) tier engines; emitted by `TierEngine::emit_causal_edges`; consumed by `TierEngine::apply_edges` |

**Critical distinction — NOT a candidate for direct dedup:**

These represent **different cognitive operations** rendered into the same u64 register width with the same name by coincidence (and lack of cross-crate review). The SPO variant carries (S, P, O, truth, mask, …) — self-contained statements. The 8-channel variant carries (target u16 separately) + 8 cognitive operators — strength vectors over a separate addressing. Source/target are **not** in the u64 for the 8-channel variant (`layered.rs:88-89`).

**Drift origin:** `lance-graph/crates/lance-graph-planner/src/cache/convergence.rs:18-22` documents the intended unification ("CausalEdge64 intended for hot-path convergence wiring") via `#[allow(unused_imports)]` — wiring started, never finished. The 8-channel variant was reinvented locally in thinking-engine rather than imported from `causal-edge`.

**Impact**: Same name in two crates is a footgun. Sprint-10 v2 work (`causaledge64-mailbox-rename-soa-v1`) targets the SPO-palette variant only; the 8-channel variant is out of scope. Future use of `CausalEdge64` must qualify by crate to avoid confusion.

**Fix**: Reunification per Option R-3 (recommended) — keep both at their respective tiers; transcode 8-channel → SPO at the thinking-engine L3 commit boundary. Mapping per `.claude/knowledge/causal-edge-64-synergies-and-pr-trajectory.md` §6.3:
- BECOMES → `InferenceType::Synthesis` (transformative resonance)
- CAUSES → `InferenceType::Deduction` (forward chain)
- SUPPORTS → `InferenceType::Revision` positive (truth corroboration)
- REFINES → `InferenceType::Abduction` (specialization)
- GROUNDS → `InferenceType::Synthesis` (foundational basis — grouped with BECOMES + RELATES per canonical source; whether GROUNDS deserves a dedicated `InferenceType::Abduction` mapping for "foundational basis = abductive justification" semantics is an open design question for the sprint-12+ transcoder spec)
- ABSTRACTS → `InferenceType::Induction` (upward generalization)
- RELATES → `InferenceType::Synthesis` (lateral relation)
- CONTRADICTS → `InferenceType::Revision` negative (refutation, lowers c)

Full design: `.claude/knowledge/causal-edge-64-spo-variant.md` + `.claude/knowledge/causal-edge-64-thinking-engine-variant.md` + `.claude/knowledge/causal-edge-64-synergies-and-pr-trajectory.md` + `.claude/knowledge/cognitive-shader-driver-thinking-engine-reunification.md`.

---

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
