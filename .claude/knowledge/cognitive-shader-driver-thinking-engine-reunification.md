# Cognitive-Shader-Driver ↔ Thinking-Engine Reunification

> **READ BY:** integration-lead, truth-architect, host-glove-designer, anyone planning the SoA-DTO reunification with thinking-engine
>
> **PREREQUISITES:** all five preceding `causal-edge-64-*` + `spo-*` + `ogit-*` knowledge docs in this set
>
> **Status:** ARCHITECTURAL ANALYSIS — drift origin documented (FINDING); reunification path is CONJECTURE pending sprint-11+ design ratification

---

## 1. The Two Halves That Drifted Apart

The workspace currently runs two parallel cognitive substrates that should have been one:

### 1.1 Cognitive-shader-driver SoA DTO (the BindSpace half)

Per CLAUDE.md "AGI-as-glove" doctrine:

```text
BindSpace = struct-of-arrays with 4 columns:
  FingerprintColumns (Topic axis — what's being reasoned about)
  QualiaColumn       (Angle axis — whose perspective, 18×f32)
  MetaColumn         (Thinking axis — which style dispatches, MetaWord bits)
  EdgeColumn         (Planner axis — CausalEdge64)
```

Operations:
- SIMD sweeps over columns (Hamming distance, palette compose, NARS revise)
- Per-row Σ-tier dispatch (per W7 SigmaTierRouter)
- AriGraph promotion on truth-confidence pass

Crate: `cognitive-shader-driver`, `bindspace.rs` + `driver.rs`.

### 1.2 Thinking-engine cascade (the L1→L2→L3 half)

Per `thinking-engine/src/layered.rs`:

```text
L1 (small N, e.g., 64-atom routing tier)
   distance_table N×N → MatVec → top-k → emit 8-channel CausalEdge64
       ↓ upstream propagation
L2 (mid M, e.g., 256-atom role tier)
   distance_table M×M → apply_edges → think → top-k → emit
       ↓ upstream propagation
L3 (full 4096-atom COCA tier)
   distance_table 4096×4096 → apply_edges → think → final ThoughtResult
```

Operations:
- MatVec on f32 energy × u8 distance table (~200-500 ns per cycle on AMX/VNNI)
- 8-channel CausalEdge64 emission/consumption (`emit_causal_edges` / `apply_edges`)
- Tier hand-off via tuple `(target: u16, edge: CausalEdge64)`

Crate: `thinking-engine`, `engine.rs` + `layered.rs`.

### 1.3 The split — same conceptual job, different implementations

Both substrates run **cognitive cycles that produce CausalEdge64 events**. But:
- cognitive-shader-driver emits the **SPO-palette CausalEdge64** via per-row palette compose
- thinking-engine emits the **8-channel CausalEdge64** via MatVec top-k
- They share the name; they don't share the type or the consumer pipeline

This is the **drift the user is pointing at** when asking "where did it happen after p64."

---

## 2. Where the Drift Happened — Three Forks

Tracing the actual code, the drift can be located precisely:

### 2.1 Fork 1: p64 convergence imported only one variant

`crates/lance-graph-planner/src/cache/convergence.rs:18-22`:

```rust
#[allow(unused_imports)] // CausalEdge64 intended for hot-path convergence wiring
use super::nars_engine::{CausalEdge64, SpoHead, MASK_SPO};
```

The `CausalEdge64` in `nars_engine` is the **SPO-palette variant** (imported from `causal-edge` crate). The p64 convergence comment promises:

> *"Hot path (p64 palette): AriGraph Triplets → Base17 fingerprints → Palette → CognitiveShader → 8 predicate layers × 64×64 attention = 4096 heads → CausalEdge64 forward/learn = O(1) per head"*

That "CausalEdge64 forward/learn" is the SPO-palette variant's `forward()`. **The 8-channel variant was never imported here**, even though by name they should be the same type.

The `#[allow(unused_imports)]` annotation confirms the wiring was started but not finished — the p64 convergence is a stub waiting for the unification that never happened.

### 2.2 Fork 2: thinking-engine layered.rs reinvented the type

`crates/thinking-engine/src/layered.rs:17-46` defines its own `pub struct CausalEdge64(pub u64)` locally — not an import from `causal-edge`. The 8-channel layout was designed independently of the SPO-palette layout that was already shipped.

This was likely done because:
- The cascade needed multi-channel simultaneity (CAUSES + SUPPORTS + REFINES at once)
- The SPO variant's 3-bit InferenceType slot couldn't carry compound channels
- Reusing the name `CausalEdge64` made the public API look unified, even though the semantics weren't

### 2.3 Fork 3: BindSpace::EdgeColumn never received 8-channel edges

`cognitive-shader-driver::bindspace.rs` (per `crates/cognitive-shader-driver/src/`) wires `EdgeColumn` to the SPO-palette CausalEdge64. The thinking-engine cascade emits 8-channel edges into `ContextBlackboard.attention_edges: Vec<u64>` — a **separate log**, not a column of the SoA.

So the thinking-engine cycle's output never lands in BindSpace::EdgeColumn directly. The two substrates write to different state stores.

---

## 3. The Roots of Reunification

The `p64::convergence` comment at `cache/convergence.rs:8-21` describes the intended unified architecture:

```text
Cold path BUILDS the graph (via LLM, slow)
Hot path SERVES the graph (via palette, fast)
p64 IS the bridge between them
```

The hot-path picture:

```text
AriGraph Triplets → Base17 fingerprints → Palette → CognitiveShader
  → 8 predicate layers × 64×64 attention = 4096 heads
  → CausalEdge64 forward/learn = O(1) per head
  → NarsTables revision = O(1) per truth update
```

The promise: **one CausalEdge64 lives at the bridge.** Both forward-pass cycle (palette compose) AND attention propagation (cascade channels) flow through it.

But which CausalEdge64? Today the answer is "neither, exactly" — `nars_engine.rs` imports the SPO variant; thinking-engine ships the 8-channel variant; neither hosts the unified version the comment implies.

### 3.1 Root of reunification — the SPO-palette compose IS the channel-routing primitive

When you read `edge.rs:393-462` (`forward()`) carefully, the steps map to channels:

| `forward()` step | Equivalent 8-channel concept |
|---|---|
| 1. Palette composition (`compose_s/p/o[a][b]`) | RELATES + GROUNDS — content-level neighbor lookup |
| 2. NARS Deduction truth (`f = a.f * w.f`) | CAUSES — directed causal energy flow |
| 3. NARS Induction truth (gen-from-shared-cause) | ABSTRACTS — upward generalization |
| 4. NARS Abduction truth (infer-from-shared-effect) | REFINES — downward specialization |
| 5. NARS Revision truth (merge-same-statement) | SUPPORTS — corroborative evidence |
| 6. NARS Synthesis truth (combine-complementary) | BECOMES — transformative resonance |
| 7. Causal mask AND (`a.mask & w.mask`) | (selects which Pearl rung; not a channel per se) |
| 8. Direction inheritance (`weight.direction()`) | (polarity propagation; not a channel) |
| 9. Plasticity inheritance (`weight.plasticity()`) | (palette-reassignment gate; not a channel) |

**Almost every 8-channel operator has a direct counterpart in the SPO-palette `forward()` step.** The missing channel is CONTRADICTS — which has no direct SPO-variant equivalent today (NARS revision can lower confidence but can't "subtract energy"). 

This near-isomorphism is the reunification opportunity. The two variants are dual representations of the same operation: SPO does it via palette + NARS rule; 8-channel does it via energy + channel selection. Mapping one to the other is **the canonical transcoding** mentioned in `causal-edge-64-synergies-and-pr-trajectory.md` §6.3.

### 3.2 The bridge per Option R-3 (recommended)

```text
Cascade tier (thinking-engine):
   uses 8-channel variant for interior dispatch
   ↓
L3 commit point (the transcoder):
   8-channel edges → SPO-palette edges via the mapping above
   CONTRADICTS → produces a Revision with weight = c × (-1) (lowers confidence)
   ↓
Persistent / SoA tier (cognitive-shader-driver + AriGraph):
   uses SPO-palette variant only
   BindSpace::EdgeColumn rows; MailboxSoA dispatch; AriGraph SPO commit
```

The transcoder is **the unification point**. It lives at the L3→commit boundary in thinking-engine, reading 8-channel emissions and writing SPO-palette edges into BindSpace::EdgeColumn (which then flows through p64 convergence as designed).

### 3.3 What "ractor-owned thoughts vs SoA DTO" means

The user's framing: "ractor owned thoughts vs cognitive-shader-driver SoA DTO that needs to be reunited with thinking engine."

**Ractor-owned thoughts:** thinking-engine cascade lives inside ractor actors (post-PR #366 `CallcenterSupervisor`). Each tier engine is an actor; cascade messages are ractor protocol messages carrying 8-channel CausalEdge64.

**Cognitive-shader-driver SoA DTO:** BindSpace SoA is read/written via `cognitive-shader-driver::driver::ShaderDriver`. EdgeColumn holds SPO-palette CausalEdge64 rows.

**Reunification:** the ractor-owned thinking-engine actors should write their L3-commit output to BindSpace::EdgeColumn via the transcoder. Today the two substrates write to disjoint state stores (`attention_edges` log vs EdgeColumn); reunification means the cascade's output IS BindSpace::EdgeColumn writes.

---

## 4. The p64 Drift Origin — Why it Happened

Why the drift formed, in the order it likely happened (CONJECTURE, but consistent with the code):

1. **Phase 1:** `causal-edge` crate ships with SPO-palette layout (S/P/O + NARS + Pearl). Used by planner cache + AriGraph SPO commit. CausalEdge64 = the unit of NARS reasoning.

2. **Phase 2:** AutocompleteCache + p64 work (the 2026-03-31 session per CLAUDE.md) wires `nars_engine::CausalEdge64` into `cache/convergence.rs` as the hot-path primitive. The "CausalEdge64 forward/learn = O(1) per head" promise is documented.

3. **Phase 3:** thinking-engine project starts. The cascade needs a richer dispatch primitive (multi-channel) than the SPO variant offers (single InferenceType + Pearl rung). Designer reuses the name `CausalEdge64` to signal "this is the cycle-speed cognitive edge," but the bit layout is new.

4. **Phase 4:** Sprint-7 (PR #366) wires `CognitiveBridgeGate` between thinking-engine and callcenter for security. The cross-tenant boundary is locked but the type-unification question isn't visited.

5. **Phase 5:** Sprint-10 specs propose CausalEdge64 v2 (witness chain, G slot, lens, etc.) — targeting the SPO-palette variant only, because that's the one in the parent plan. The 8-channel variant isn't in scope, so the drift persists into sprint-10's design.

**Where the drift root lies:** the implicit assumption that "CausalEdge64" is one thing. Phase 3's renaming-without-unifying broke that assumption, but no PR explicitly documented it. Sprint-10 inherits the broken assumption.

---

## 5. Reunification Plan (CONJECTURE — sprint-11+ scope)

Five sequenced steps:

### 5.1 Step 1: Acknowledge two variants (sprint-10 deliverable)

- PREPEND to `EPIPHANIES.md` the dual-CausalEdge64 finding (E-META-7 candidate)
- Update `docs/TYPE_DUPLICATION_MAP.md` to list `CausalEdge64` as a 2-copy duplication with distinct semantics
- Update `LATEST_STATE.md` Contract Inventory to name both variants explicitly

### 5.2 Step 2: Lock the transcoding mapping (sprint-11)

- Spec the canonical 8-channel → SPO-palette transcoder in a new knowledge doc or W7-style spec
- Implement `thinking_engine::commit::transcode_to_spo(channels) -> causal_edge::CausalEdge64`
- Run a regression test that all 8 channels round-trip through the transcoder (with documented information loss where applicable — e.g., CONTRADICTS becomes negative-weighted Revision, BECOMES becomes Synthesis)

### 5.3 Step 3: Wire L3 commit to BindSpace::EdgeColumn (sprint-11)

- thinking-engine's L3 commit step writes transcoded SPO-palette edges into BindSpace::EdgeColumn via ShaderDriver
- `ContextBlackboard.attention_edges: Vec<u64>` becomes a debug log only; canonical state lives in EdgeColumn
- ractor message types updated to carry SPO-palette edges (the 8-channel variant stays internal to thinking-engine)

### 5.4 Step 4: Sprint-10 v2 SPO-palette gains W-slot + lens (sprint-11)

- v2 layout per `causal-edge-64-synergies-and-pr-trajectory.md` §4 (drop temporal + G, add W-slot + lens + spare)
- The W-slot points to the witness corpus root; the corpus is CAM-PQ-indexed per `spo-ontology-format-stack.md`
- AriGraph SPO-G + SPO-W per `spo-schema-and-mailbox-sidecar.md` §5

### 5.5 Step 5: Operationalize ontology-aware splat (sprint-12+)

- thinking-engine's `emit_causal_edges_filtered` reads OWL/DOLCE filter (per `ogit-owl-dolce-ontology-compartments.md` §6)
- The splat dispersion respects ontology pathways (Endurants don't splat into Perdurants without an explicit mapping channel)
- Per-channel ontology gating: each of the 8 channels has its own allowed-axiom-set
- This is the **operational unification** of thinking-engine richness with ontology schema synergy

---

## 6. What "Computational Entropy through Struct-Oriented" Means

The user's closing point: "the chance to create ndarray and work stealing of rayon to offer struct methods can simplify the code and make it more meaningful and simpler at same time (computational Entropy through struct object oriented)."

This points at the **next major architectural simplification** — see `splat-shader-rayon-struct-method-vision.md` for the full discussion. Briefly:

- Free functions on raw arrays force the caller to assemble state (carrier + config + awareness + graph + …)
- Struct methods on the carrier put the state where it can reason about itself
- Per CLAUDE.md "The Click" Litmus test 1: *"Does this add a free function on a carrier's state, or a method on the carrier? → Free function = reject. Method = accept."*
- Reunification of cognitive-shader-driver SoA + thinking-engine cascade as **methods on a unified `Think` struct** (per CLAUDE.md §"Thinking is a struct") collapses the drift into a single carrier type

The "computational entropy" reduction comes from removing the duplicated state stores (`attention_edges` log vs EdgeColumn), the parallel implementations of similar operations (cascade `apply_edges` vs SoA `dispatch_cycle`), and the divergent type semantics (8-channel vs SPO-palette). Each consolidation collapses entropy by one degree of freedom.

---

## 7. Cross-references

- `causal-edge-64-spo-variant.md`, `causal-edge-64-thinking-engine-variant.md`, `causal-edge-64-synergies-and-pr-trajectory.md` — the dual-variant analysis
- `spo-schema-and-mailbox-sidecar.md` — schema and mailbox implications
- `spo-ontology-format-stack.md` — storage format selection
- `ogit-owl-dolce-ontology-compartments.md` — ontology layer interaction
- `splat-shader-rayon-struct-method-vision.md` — future shader ops + struct-method simplification
- `lab-vs-canonical-surface.md` — canonical UnifiedStep surface (the doctrine that says don't add new layers)
- `lance-graph-planner/src/cache/convergence.rs` — the p64 bridge with the `#[allow(unused_imports)]` evidence
- `thinking-engine/src/layered.rs` — the 8-channel variant definition
- `cognitive-shader-driver/src/driver.rs` — the SoA dispatch entry
- `.claude/board/PR_ARC_INVENTORY.md` — PR #366 (CognitiveBridgeGate wire)
- `.claude/plans/oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` — the larger merge plan that defines SPOW

---

*Authored 2026-05-14. Drift origin reconstructed from code evidence (FINDING); reunification plan CONJECTURE pending sprint-11+ ratification.*
