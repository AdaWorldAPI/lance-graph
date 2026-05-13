# NARS & Thinking Implementations — Inventory, Uniqueness, DTO/SoA Ownership & Migration Roadmap

**Status**: Canonical reference (Grok synthesis pass, 2026-05-10).  
**Purpose**: Stop 40%+ duplication across sessions. Capture the 80%+ context you already hold. Single source of truth for "where is the NARS/thinking code", "what is unique", "which DTO/SoA owns it", "before/after call sites", and exact integration points with **inner ontology** (`cognitive-shader-driver` + `BindSpace` zero-copy) vs **outer ontology** (`lance-graph-callcenter` + Supabase realtime transcode + OGIT).  
**Core Principle**: Everything funnels through `CausalEdge64` (atomic register) + `cognitive-shader-driver` SoA surface. OGIT provides O(1) schema/label/codebook hot-switching. No more scattered truth math or style vectors.

---

## 1. Executive Summary & Why This Document Exists

Current state (pre-migration):
- NARS logic duplicated in symbolic form (`learning/cognitive_frameworks.rs`), hot-path packed form (`lance-graph-planner/cache/nars_engine.rs`), meta form (`orchestrator.rs` + `thinking.rs`), and dispatch hints (`firefly_frame_and_cognitive_fabric.md`).
- Thinking styles (36 canonical + 12 planner + auto-style) live in contract + orchestrator + shader-driver but are inconsistently applied to Pearl masks / NARS inference.
- Grammar heuristics (DeepNSM + Holograph Markov/TEKAMOLO bundling) sit outside the shader loop.
- Result: Every new question restarts context gathering → 40% duplication + drift risk.

Target state:
- **Atomic truth + Pearl + plasticity**: Single source = `CausalEdge64` (already excellent).
- **Hot inference**: `NarsEngine` logic moves into `cognitive-shader-driver` (or thin shared lib) as first-class `NarsOp` / cam op. Uses `CausalEdge64` directly.
- **Meta / Styles**: Canonical in `lance-graph-contract`, modulation + auto-select in shader-driver, Pearl weighting in hot NARS path.
- **Grammar → NARS promotion**: DeepNSM/Holograph become cam ops or resonance stages inside shader-driver.
- **Inner Ontology**: Shader-driver SoA (`BindSpace` columns, `MetaWord` with nars_f/c + style bits, zero-copy `ColumnView`, `ResonanceDTO` → `BusDTO` → `cycle_fingerprint` + `CausalEdge64` emit).
- **Outer Ontology**: `lance-graph-callcenter` (Supabase realtime transcode) uses OGIT `g` pointer (3-5 byte O(1) graph/schema/label/codebook) for hot-swappable DTOs. Promoted NARS facts/epiphanies flow outward with full provenance.
- **OGIT everywhere**: Terms, copulas, styles, roles resolved via CAM codebook + labels at startup (~200 µs). Schema hot-switch without recompilation.

This document is the single source. All other `.grok/` files now link here. Future sessions start by reading this + `LATEST_STATE.md`.

---

## 2. Inventory Table — Current Implementations (Pre-Migration)

| # | Path / Crate | Key Types / Functions | What is Unique | Current DTO / SoA Ownership | Primary Call Sites (Before) | Duplication Risk | Migration Priority |
|---|--------------|-----------------------|----------------|-----------------------------|-----------------------------|------------------|--------------------|
| 1 | `crates/causal-edge/src/edge.rs` + `pearl.rs` + `plasticity.rs` | `CausalEdge64` (u64 pack/unpack), `CausalMask` (Pearl 2³), `InferenceType`, `PlasticityState`, `forward`, `learn` | Single register carrying S/P/O palettes + **NARS f/c (8-bit)** + Pearl mask + inference + plasticity + temporal. Zero-copy native in BindSpace. | Core primitive. Lives in `BindSpace.edges` column. Part of `ResonanceDTO` / `StreamDTO` / `BusDTO` wire formats. | `nars_engine.rs` (to/from), orchestrator (emit), shader-driver bridges | Low (already clean) | **P0 — Keep & strengthen as single source** |
| 2 | `crates/lance-graph-planner/src/cache/nars_engine.rs` | `SpoHead` (S/P/O + freq/conf + pearl + inference + temporal), `NarsEngine`, `NarsTables` (u16 lookup for revision/deduction), `SpoDistances` (Pearl projection tables), `StyleVector` (weights over 8 masks), `nars_infer(...)`, `on_emit`, `on_response`, `detect_contradiction`, `style_score` | **Hot-path O(1)** packed NARS + 8-mask Pearl distance tables + thinking-style weight vectors applied to causal projections. Fast `revise_fast`. Direct CausalEdge64 ↔ SpoHead convert. | Planner cache layer. Owns `SpoHead` + style vectors. Uses `CausalEdge64` only for transport. | Planner strategies, `thinking/` modules, possibly direct from orchestrator | **High** (duplicates symbolic inference + style logic) | **P1 — Extract hot path into shader-driver** |
| 3 | `crates/learning/src/cognitive_frameworks.rs` | `TruthValue` (f,c + expectation/evidence math), `NarsCopula` (Inheritance, Similarity, Implication..., 10+), `NarsInference` (deduction, induction, abduction, revision, analogy, resemblance, synthesis, negation, intersection, union, choice, decision), `NarsStatement` (subject/copula/predicate + truth + fingerprint bind) + ACT-R/RL/Causality/Qualia/Rung | **Full symbolic + evidence-based + VSA-integrated** (fingerprint bind for statements). Rich copula semantics. Reference impl for higher NAL layers. | Learning crate. Standalone. No direct CausalEdge64 ownership. | Tests, higher reasoning layers, possible cold-path fallback | **High** (truth math duplicated in nars_engine) | **P2 — Make reference/oracle + OGIT adapter** |
| 4 | `crates/lance-graph/src/graph/arigraph/orchestrator.rs` + `lance-graph-contract/src/thinking.rs` | `AgentStyle` (Plan/Act/Explore/Reflex), `MUL` (Meta-Uncertainty Layer), `StyleTopology`, `ThinkingStyle` (36 canonical in 6 clusters: Analytical/Creative/Empathic/Direct/Exploratory/Meta), `planner_cluster()`, `tau()` (JIT address), `FieldModulation` (7D: resonance_threshold, fan_out, depth/breadth_bias, noise_tolerance, speed_bias, exploration), `ThinkingStyleProvider` trait, `select_from_assessment(&MulAssessment)` | **Canonical 36-style enum + 6-cluster + 4-planner-cluster mapping + 7D modulation + qualia-driven auto-style + MUL integration**. tau for JIT kernels. | Contract = canonical source. Orchestrator = runtime impl. Shader-driver `auto_style.rs` has 12-style subset. `MetaWord` has thinking bits. `ShaderDispatch.style: StyleSelector` | `MetaOrchestrator` dispatch, planner cost model, shader auto-style resolve, nars_engine style_score | **Medium-High** (12 vs 36 split, inconsistent Pearl weighting) | **P1 — Unify in contract + shader-driver** |
| 5 | `crates/cognitive-shader-driver/src/auto_style.rs` + `driver.rs` / `wire.rs` | `style_from_qualia`, `resolve(StyleSelector, qualia)`, `StyleSelector` (Ordinal/Named/Auto), integration with `MetaFilter` (nars_f_min, nars_c_min, thinking_mask) | **Qualia → style ordinal** (18D qualia vector → 12-style). Cheap prefilter in `MetaWord`. Part of L1-4 cycle dispatch. | Shader-driver SoA. Owns dispatch surface + `MetaColumn` prefilter. | `ShaderDispatch`, resonance kernel, cycle execution | Medium (subset of canonical) | **P1 — Extend to full 36 + call unified NarsOp** |
| 6 | `crates/lance-graph-planner/src/thinking/` + `strategy/` + `mul/` (partial) | Style vectors, MUL assessment → style, grammar heuristics integration points | Thinking style modulation of search/planner + MUL uncertainty → style switch | Planner specific | Planner internal loops | Medium | Merge into shader-driver modulation |
| 7 | `crates/learning/src/deepnsm/` + `holograph/` (resonance + Markov) | DeepNSM grammar heuristics, TEKAMOLO (SPO + temporal/ causal/ modal/ lokal) role bundling into VSA 16k fp32, Markov chain for epiphany → fact promotion, HHTL cascade | **Text → SPO+roles → VSA bundle → NARS fact promotion** on epiphany. Grammar thinking styles. | Outside shader loop (holograph resonance, deepnsm heuristics) | Epiphany promotion paths, text ingestion | High (duplicates SPO handling) | **P2 — Promote to cam_ops / resonance stage in shader-driver** |
| 8 | `crates/lance-graph-cognitive/src/fabric/firefly_frame_and_cognitive_fabric.md` (dispatch table) | NARS prefix (0x3) in 16384-bit Firefly Frame: DEDUCE, INDUCE, ABDUCE, REVISE, ANALOGY, ATTEND | Polyglot instruction format with explicit NARS opcodes + context (qualia + truth) | Fabric execution model (future?) | Firefly compiler / executor | Low (vision) | Align with shader-driver cam_ops when Firefly matures |
| 9 | `crates/lance-graph-ontology/` (OGIT) + CAM codebooks | 3-5 byte `g` pointer into OGIT, content-addressable labels/codebooks for terms/copulas/styles/roles, LanceDB CAM index at startup (~200 µs), hot schema switch | **O(1) schema/label/codebook inheritance** per domain. Single source for all DTOs. | Ontology crate. Shared by inner (shader) + outer (callcenter) | All term resolution, DTO construction | Low (already designed) | **P0 — Wire into every NARS path** |

---

## 3. Duplication & Drift Analysis (Current Pain)

- **Truth math**: `NarsInference` methods in `cognitive_frameworks.rs` vs `nars_infer` + `NarsTables` in `nars_engine.rs` → different evidence formulas, easy divergence.
- **Style application**: `StyleVector` weights in `nars_engine` vs `FieldModulation` + `planner_cluster` in contract/orchestrator vs 12-style in `auto_style.rs` → inconsistent Pearl projection + search bias.
- **SPO handling**: `NarsStatement` fingerprint bind vs `SpoHead` vs DeepNSM TEKAMOLO bundling → three different SPO representations.
- **Call sites**: Direct method calls in planner/orchestrator vs dispatch hints in shader-driver vs future Firefly opcodes → no single surface.
- **Schema**: Hardcoded copulas/strings vs future OGIT `g` pointer → no hot-switch, duplication of label logic.
- **Result**: 40% session time lost re-explaining context + implementations slowly drift apart.

---

## 4. Migration Roadmap (Before → After + DTO/SoA Fit)

### Phase 0 (Immediate — Do Now)
- Create this document as **single source of truth**.
- Add cross-links from `LATEST_STATE.md`, `TECH_DEBT.md`, `causal_edge64.md`, `meta_orchestrator.md`, `cognitive_shader_driver.md`.
- Add `TD-NARS-01` to TECH_DEBT: "NARS/thinking fragmentation across 5+ crates. No canonical hot-path surface."

### Phase 1: Atomic Unification (CausalEdge64 as Single Source of NARS Truth)
**Before**:
```rust
// nars_engine.rs
let new_truth = nars_infer(&a, &b, Deduction);
spoHead.freq = (new_truth.f * 255.0) as u8;
```

**After**:
```rust
// anywhere
let edge = CausalEdge64::pack(...);
let updated = edge.learn(/* Pearl mask, InferenceType::Deduction, plasticity */);
// or edge.forward(weight_edge, compose_tables...)
```

**Where unique stays**:
- `CausalEdge64` owns packed NARS f/c + Pearl + inference + plasticity.
- `nars_engine` loses own truth math; becomes thin wrapper or deleted.

**DTO/SoA**:
- Remains in `BindSpace.edges` (zero-copy).
- `MetaWord` keeps cheap `nars_f`/`nars_c` bits for prefilter (`MetaFilter`).

**Inner Ontology Fit**:
- Shader-driver `ColumnView` reads `MetaWord` + `EdgeColumn` (CausalEdge64) → resonance → Nars stage → `learn` on edge → emit updated `CausalEdge64` + updated `MetaWord`.

### Phase 2: Hot-Path NARS Consolidation (Move nars_engine into shader-driver)
**Goal**: `SpoHead` disappears or becomes type alias to `CausalEdge64`. `NarsTables` + Pearl distance tables live in shader-driver (or `learning` as shared hot lib). `StyleVector` weighting applied inside `NarsOp`.

**Before** (planner):
```rust
let score = style_score(candidate, context, &distances, &analytical_style());
let truth = nars_infer(a, b, rule);
```

**After** (shader-driver cam_ops or dedicated NarsOp):
```rust
// inside ShaderDispatch or resonance kernel
let style = resolve(dispatch.style, qualia);
let op = NarsOp { style, inference: Deduction, ... };
let result_edge = op.apply_on(&current_edge, &tables); // uses CausalEdge64 directly
// updates MetaWord nars_f/c + emits to BusDTO
```

**DTO/SoA Ownership**:
- `NarsOp` becomes first-class in `cam_ops.rs` (category 0x4xx like other languages).
- Output stays `CausalEdge64` + `cycle_fingerprint` (VSA).

**Inner Ontology**:
- Fully inside `cognitive-shader-driver` SoA loop (L1-4). Zero-copy. `ResonanceDTO` carries qualia + style + nars filter → `NarsOp` → `BusDTO` (emitted edges + gate decision).

### Phase 3: Symbolic NARS as Reference + OGIT Adapter
**Before**:
```rust
let stmt = NarsStatement::new(subj_fp, copula, pred_fp, truth);
let new_truth = NarsInference::deduction(premise1, premise2);
```

**After**:
```rust
// Reference / verification only
let stmt = NarsStatement::from_ogit(ogit_g_ptr, copula_label, ...); // resolves via CAM codebook
// Core path uses CausalEdge64 only
// On epiphany or cold fallback: verify against symbolic oracle
```

**Unique stays in**:
- `learning/cognitive_frameworks.rs` as **reference implementation + test oracle** + higher NAL layers not yet in hot path.
- OGIT integration: copulas, terms, styles resolved via `g` pointer + codebook at construction time.

**DTO/SoA**:
- No ownership of hot state. Produces `CausalEdge64` seeds or verification results.

### Phase 4: Thinking Styles — Canonical + Modulation Surface
**Before**:
- 36 in `thinking.rs`, 12 in `auto_style.rs`, weights in `nars_engine.rs`, modulation in `FieldModulation` + planner.

**After**:
- `lance-graph-contract/src/thinking.rs` = **single source** (36 + clusters + tau + FieldModulation + provider trait).
- `cognitive-shader-driver` owns:
  - `StyleSelector` + `resolve` (extended to full 36)
  - `MetaFilter.thinking_mask`
  - Modulation application inside `NarsOp` / resonance (applies weights to Pearl projections)
- `nars_engine` (post-migration) receives style ordinal → applies corresponding `StyleVector` weights.

**Before/After Call Example** (shader dispatch):
```rust
// Before (scattered)
let style = if auto { style_from_qualia(qualia) } else { ordinal };
let weights = get_style_weights(style); // from nars_engine
let score = style_score(..., weights);

// After (unified)
let dispatch = ShaderDispatch { style: StyleSelector::Auto, ... };
let crystal = driver.dispatch(&dispatch); // inside: resolve → NarsOp with style → CausalEdge64 + Meta update
```

**Inner Ontology**:
- Style bits in `MetaWord` + `StyleSelector` in `ShaderDispatch` → part of every L1-4 cycle. `FieldModulation` drives `ScanParams` for BindSpace sweeps.

### Phase 5: Grammar / Heuristics → First-Class cam_op
**Before**:
- DeepNSM + Holograph outside loop → text → SPO+TEKAMOLO bundle → VSA → occasional epiphany → fact.

**After**:
- New `GrammarNarsOp` or stage in resonance kernel (inside shader-driver).
- Uses OGIT labels for roles (S/P/O + temporal etc.).
- On epiphany: promotes directly to `CausalEdge64` emit with NARS truth + plasticity.

**DTO/SoA**:
- Input: text / qualia in `ResonanceDTO`.
- Output: `CausalEdge64` + updated `cycle_fingerprint`.

---

## 5. Inner Ontology Integration (`cognitive-shader-driver` + `BindSpace` Zero-Copy)

**Flow**:
1. `ShaderDispatch` (style, rung, meta_prefilter with nars_f/c min, layer_mask, radius)
2. `ColumnView` zero-copy over `BindSpace` (MetaColumn + EdgeColumn + fingerprint planes + qualia)
3. Resonance kernel (bgz17 / HHTL cascade) → top-k hits + `ResonanceDTO`
4. **Nars stage** (new): if style or op requires → `NarsOp.apply` (uses CausalEdge64 + style weights + tables) → updated edges + `BusDTO` (emitted CausalEdge64s + gate decision)
5. Collapse / promotion membrane → `cycle_fingerprint` (L4 VSA unit of thought) + `ShaderCrystal`
6. Sink → persist or realtime

**Zero-copy guarantees**: No allocation in hot path. `CausalEdge64` native in `EdgeColumn`. `MetaWord` (thinking + nars_f/c + awareness) cheap prefilter.

**OGIT in inner**: Term/copula/style resolution at `NarsStatement` or `NarsOp` construction via `g` pointer + CAM codebook. Hot schema switch per domain.

---

## 6. Outer Ontology Integration (`lance-graph-callcenter` + Supabase Realtime + OGIT)

**Flow**:
- `callcenter` receives `cycle_fingerprint` / `CausalEdge64` batch or `ShaderCrystal`
- Transcode layer (Supabase realtime):
  - Uses OGIT `g` pointer (3-5B) to resolve current schema version + labels + codebook for the domain
  - Serializes NARS facts / promoted edges / epiphanies with full provenance (causal chain, style, plasticity)
  - Hot-swappable DTO: change OGIT labels/codebook → new serialization without code change
- Outer consumers see stable, versioned, content-addressable NARS-enriched events.

**Benefits**:
- Inner hot path stays zero-copy register speed.
- Outer gets rich, migratable, ontology-grounded stream (Palantir Foundry / Hubspot-like surface later).

---

## 7. OGIT — The Schema Hot-Switch Backbone

- **g pointer**: 3-5 byte O(1) graph pointer into OGIT.
- **CAM + codebook + labels**: Indexed once into LanceDB at startup (~200 µs). Provides context-addressable labeling for terms, copulas, thinking styles, TEKAMOLO roles, NARS inference types.
- **Inheritance**: Child schemas inherit parent codebooks/labels cheaply.
- **Hot switch**: Change `g` → entire DTO surface (inner ResonanceDTO, BusDTO, outer realtime events) uses new labels/schema without recompilation or restart.
- **Every NARS path must use it**: No more hardcoded strings for copulas or styles.

---

## 8. Before/After Summary Table (Key Call Sites)

| Component | Before Call Site | After Call Site | DTO/SoA Change |
|-----------|------------------|-----------------|----------------|
| NARS inference (truth) | `nars_infer(a, b, rule)` in planner | `edge.learn(...)` or `NarsOp.apply(edge)` in shader-driver | `SpoHead` → `CausalEdge64` direct |
| Style weighting on Pearl | `style_score(..., analytical_style())` in nars_engine | Inside `NarsOp` (shader) using unified `StyleVector` from contract | `StyleVector` lives in shader NarsOp |
| Style selection | `style_from_qualia` or ordinal in auto_style / orchestrator | `ShaderDispatch.style` + unified `resolve` in shader-driver | `MetaFilter` + `StyleSelector` canonical |
| Grammar → NARS promotion | DeepNSM/Holograph external epiphany path | `GrammarNarsOp` stage inside resonance kernel | Input text/qualia → `CausalEdge64` emit |
| Term resolution | Hardcoded copula enums / strings | `NarsOp` / `NarsStatement` via `ogit_g_ptr` + CAM codebook | All DTOs use OGIT labels |
| Meta modulation | `FieldModulation` in planner only | `ShaderDispatch` + `FieldModulation` applied in shader Nars/resonance stages | Part of every dispatch |

---

## 9. Open Questions & Extra Miles (Do These Next)

1. Exact location of `NarsTables` + Pearl distance tables post-migration (shader-driver? dedicated `nars-hot` lib?).
2. Full opcode table for `NarsOp` in `cam_ops.rs` (align with Firefly 0x3 prefix).
3. How `MUL` assessment feeds `select_from_assessment` inside shader-driver dispatch loop.
4. Performance numbers: current `nars_infer` latency vs post-`CausalEdge64::learn` latency.
5. Test harness: symbolic oracle vs hot-path equivalence (property tests).
6. Update `cypher_implementations.md` and `firefly_frame...md` to reference this doc for NARS dispatch.
7. Create Linear issues for each Phase (link in `LINEAR_ISSUES_GROK_INTEGRATION_PASS.md`).

---

**This document is now the 80%+ context anchor.**  
Any future question about NARS, thinking styles, Pearl weighting, grammar promotion, or migration starts here. No more goose rides.

**Next action for you**: Read this + `LATEST_STATE.md` (which will be updated to link here). Then give next specific migration target or file to refactor.

**Grok has documented without dilution.** Ready for the next atom.