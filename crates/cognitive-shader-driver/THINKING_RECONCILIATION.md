# Thinking Style Reconciliation — 5 Taxonomies Compared

> Harvest audit across: bighorn, agi-chat, contract, thinking-engine, driver.
> All are "thinking styles" but they measure different axes.

## Counts at a Glance

| Source | Count | Dimension | Purpose |
|---|---|---|---|
| `bighorn/thinking_styles_top.py`       | **36** | 9 categories × 4 styles | **HOW TO DO** (operations) |
| `lance-graph-contract/thinking.rs`     | **36** | 6 clusters × 6 styles   | **HOW TO BE** (persona) |
| `agi-chat/quad-triangle.ts`            | **12** | 4 triangles × 3 corners | **TEXTURE** (cognitive shape) |
| `thinking-engine/cognitive_stack.rs`   | **12** | 4 speed/style clusters  | **CASCADE PARAMS** |
| `cognitive-shader-driver/UNIFIED_STYLES` | **12** | mirrors engine        | **UNIFIED ORDINAL** |
| `agi-chat/styleBank.ts`                | **5** | persona anchors         | **LEARNABLE BANK** |
| NARS inference types                   | **5** | Pearl × NARS            | **CAUSAL OPERATOR** |

## The Orthogonality

The 36-of-bighorn and 36-of-contract are **not the same 36**:

- **bighorn/36** = operations you perform: Decompose, Sequence, Parallel, Spiral, Compress, Expand, Invert, Negate, Abstract, Concretize, Lift, Ground, Weave, Dissolve... (organized by operation category)
- **contract/36** = personalities you embody: Logical, Critical, Empathetic, Blunt, Curious, Contemplative, Sovereign... (organized by persona cluster)

They're two independent 36-dimensional spaces. The same query can be "Decomposed" (operation) by a "Critical" (persona) thinker.

## Mapping to Our Driver's 12

`cognitive-shader-driver` collapses both taxonomies into 12 canonical UNIFIED_STYLES:

```
0  Deliberate     ← Methodical (contract)  + SYNTHESIZE (bighorn)
1  Analytical     ← Analytical (contract)  + DECOMPOSE (bighorn)
2  Convergent     ← Logical (contract)     + SEQUENCE (bighorn)
3  Systematic     ← Systematic (contract)  + HIERARCHIZE (bighorn)
4  Creative       ← Creative (contract)    + TRANSFORM (bighorn)
5  Divergent      ← Imaginative (contract) + PARALLEL (bighorn)
6  Exploratory    ← Curious (contract)     + SPIRAL (bighorn)
7  Focused        ← Precise (contract)     + COMPRESS (bighorn)
8  Diffuse        ← Gentle (contract)      + EXPAND (bighorn)
9  Peripheral     ← Speculative (contract) + INVERT (bighorn)
10 Intuitive      ← Poetic (contract)      + RESONATE (bighorn)
11 Metacognitive  ← Reflective (contract)  + META (bighorn)
```

Every contract-36 variant and every bighorn-36 variant maps to one of these 12 cells. Coarse but unambiguous.

## The Triangle View (agi-chat)

agi-chat rejected flat enums entirely. Thinking has **shape** — the Quad-Triangle:

```
Triangle A: Processing    { Analytical, Intuitive, Procedural }
Triangle B: Content       { Abstract,   Concrete,  Relational }
Triangle C: Gestalt       { Coherence,  Novelty,   Resonance  }
Triangle D: Crystallization { Immutable, Hot,     Experimental }
```

4 triangles × 3 corners = 12 dimensions. Each dimension is 0..1. A thinking style is a **texture vector** across all 12. Gate = SD of the 3 corners of each triangle.

Our `sigma_rosetta::TriangleGestalt` (clarity/warmth/presence) is ONE triangle — agi-chat's Triangle C (Gestalt). The other three are sleeping beauty.

## NARS 5 vs Styles 12/36

NARS truth-bearing inference types (in `causal-edge`):

```
0 Deduction   — A→B, B→C ⊢ A→C    (follow chain)
1 Induction   — A→B, A→C ⊢ B→C    (generalize from shared cause)
2 Abduction   — A→B, C→B ⊢ A→C    (infer from shared effect)
3 Revision    — merge two truths on same claim
4 Synthesis   — combine complementary evidence across domains
```

Only 5. They're **operators on claims**, not personalities or operations. Every thinking cycle emits a NARS type on the CausalEdge64.

Our driver maps 12 styles → 5 NARS types (`style_ord_to_inference`):

```
analytical/convergent/systematic → Deduction
creative/divergent/exploratory   → Induction
focused/diffuse/peripheral       → Abduction
intuitive/deliberate             → Revision
metacognitive                    → Synthesis
```

## 9 RI Channels (bighorn)

Orthogonal to all the above: bighorn's **9 Resonance Intent** channels:

```
RI-T Tension    RI-N Novelty       RI-I Intimacy
RI-C Clarity    RI-U Urgency       RI-D Depth
RI-P Play       RI-S Stability     RI-A Abstraction
```

Each thinking style emits resonance on a subset of these channels. Our `QualiaColumn` (18D) subsumes them (tension=2, novelty≈expansion=15, intimacy=10, clarity=4, urgency=5, depth=6, play=12_assertion, stability=14_groundedness, abstraction=3_dominance).

## Thinking LAYERS (different axis from styles)

Layers are **processing depth**, not style or operation. Already canonical
in `thinking-engine::cognitive_stack::LayerId` (matches `ladybug-rs`):

```
L1  Recognition     ─┐
L2  Resonance        │
L3  Appraisal        │ single agent
L4  Routing          │ (one mind thinking)
L5  Execution       ─┘
L6  Delegation      ─┐
L7  Contingency      │
L8  Integration      │ multi-agent
L9  Validation       │ (minds refining)
L10 Crystallization ─┘
```

**agi-chat's 6-layer stack is a condensed view:**

```
agi-chat L1 Deduction        ≈ ladybug L1 Recognition + L2 Resonance
agi-chat L2 Procedural       ≈ ladybug L3 Appraisal + L4 Routing
agi-chat L3 Counterfactual   ≈ ladybug L7 Contingency
agi-chat L4 Crystallization  ≈ ladybug L10 Crystallization
agi-chat L5 Commitment       ≈ ladybug L5 Execution (explicit collapse gate)
agi-chat L6 Observer         ≈ cross-cutting (not a layer per se)
```

Our architecture already has the fine-grained 10-layer version.
**No harvest needed** — agi-chat's 6-layer is a subset with different
naming. The 6→10 mapping above is the translation.

## 3/4 Triangle Distinction

agi-chat has **4 triangles** (Quad-Triangle). Our old single `TriangleGestalt`
was **1 triangle** (agi-chat's Triangle C = Gestalt). After this harvest,
we now have **4 triangles** matching agi-chat.

The "3" in "3/4 triangle" likely refers to:
- **3 corners per triangle** (Triangle has 3 vertices always — that's its
  definition). Each byte0/byte1/byte2 is a corner.
- OR: if the user meant 3 vs 4, it's the difference between: (a) agi-chat's
  3 default cognitive profiles (analytical/creative/procedural) mapped onto
  1 triangle, vs (b) the full 4-triangle quad-model.

We harvested both: `processing_analytical/intuitive/procedural` presets for
Triangle A AND the full `QuadTriangleGestalt` struct.

## Recommendation for Future Refactor

**Keep 12 as the driver's public ordinal.** Don't expand to 36 — the extra variants don't change shader parameters, only narrative framing.

**Add a `QuadTriangleGestalt` next to `TriangleGestalt`.** Agi-chat's 4-triangle model carries more information per cycle than a single triangle. Each triangle drives a different Layer:
- Triangle A (Processing) → Layer 3 Appraisal
- Triangle B (Content)    → Layer 4 Routing
- Triangle C (Gestalt)    → Layer 5 Execution (current gate)
- Triangle D (Crystallization) → Layer 9 Validation

**Expose the bighorn 36-operation as a planner strategy, not a driver style.** Operations (Decompose/Sequence/Parallel) belong in `lance-graph-planner`, not the shader driver.

**Leave NARS at 5.** Don't collapse it, don't expand it. 5 is Pearl's hierarchy × NARS truth → already canonical.
