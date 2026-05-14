# CausalEdge64 — SPO-Palette Variant (causal-edge crate)

> **READ BY:** truth-architect, integration-lead, palette-engineer, family-codec-smith, certification-officer, anyone touching `NarsTables` / `lance-graph-planner::cache` / `cognitive-shader-driver::BindSpace::EdgeColumn`
>
> **PAIRED WITH:** `causal-edge-64-thinking-engine-variant.md` (the OTHER `CausalEdge64` type in this workspace; see also `causal-edge-64-synergies-and-pr-trajectory.md` for cross-comparison)
>
> **Status:** FINDING (verified against shipped source 2026-05-14)

---

## 1. Identity

**Type:** `causal_edge::CausalEdge64` (the "SPO-palette variant", "NARS variant", "Pearl variant").

**Definition site:** `crates/causal-edge/src/edge.rs:60`

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CausalEdge64(pub u64);
```

**Crate:** `causal-edge` (in workspace `members` per CLAUDE.md; sibling of `lance-graph` core; depended on by `lance-graph-planner`).

**One-line role:** the 64-bit causal **neuron** — one register holds a complete SPO causal proposition with NARS truth, Pearl rung, propagated direction, NARS inference type, palette plasticity, and a temporal index.

**Doctrine line** (from `edge.rs:1-3`):
> *"CausalEdge64: the atomic causal unit. One u64. One register. One read. Full causal edge with epistemic state."*

---

## 2. Bit Layout

From `edge.rs:44-72`:

```text
[ 0:  7]  S palette index           u8   (256 subject archetypes)
[ 8: 15]  P palette index           u8   (256 predicate archetypes)
[16: 23]  O palette index           u8   (256 object archetypes)
[24: 31]  NARS frequency            u8   (f = val/255 ∈ [0, 1])
[32: 39]  NARS confidence           u8   (c = val/255 ∈ [0, 1])
[40: 42]  Causal mask               3b   (Pearl's 2³ — see CausalMask enum)
[43: 45]  Direction triad           3b   (sign(palette[idx].dim0) per S,P,O — propagated)
[46: 48]  Inference type            3b   (NARS rule: Deduction/Induction/Abduction/Revision/Synthesis + 3 reserved)
[49: 51]  Plasticity flags          3b   (hot/cold per S,P,O plane — palette-reassignment gate)
[52: 63]  Temporal index            12b  (4096 time slots — coarse cycle bucket)

Total                               64b  (zero unused bits)
```

**Constants** (`edge.rs:63-76`):

```rust
const S_SHIFT:        u32 = 0;
const P_SHIFT:        u32 = 8;
const O_SHIFT:        u32 = 16;
const FREQ_SHIFT:     u32 = 24;
const CONF_SHIFT:     u32 = 32;
const CAUSAL_SHIFT:   u32 = 40;
const DIR_SHIFT:      u32 = 43;
const INFER_SHIFT:    u32 = 46;
const PLAST_SHIFT:    u32 = 49;
const TEMPORAL_SHIFT: u32 = 52;

const BYTE_MASK:   u64 = 0xFF;
const BITS3_MASK:  u64 = 0b111;
const BITS12_MASK: u64 = 0xFFF;
```

---

## 3. Field Semantics

### 3.1 S / P / O palette indices (bits 0-23)

Three 8-bit indices into three separate 256-entry palette codebooks. **Not role keys** — role identity is given by *which* 8-bit slot the index sits in (S is bits 0-7, P is 8-15, O is 16-23). Role keys (per CLAUDE.md "The Click") live elsewhere as 4K-bit Vsa16kF32 slices.

Addresses 256³ ≈ 16M (S, P, O) content tuples per edge.

### 3.2 NARS frequency + confidence (bits 24-39)

NARS truth value (Wang 2013 NAL semantics):

- `f = positive_evidence / total_evidence ∈ [0, 1]`
- `c = total_evidence / (total_evidence + k) ∈ [0, 1]` where `k = 1` (NARS personality constant)

Accessors at `edge.rs:152-220`:

```rust
pub fn frequency_u8(self) -> u8       { (self.0 >> FREQ_SHIFT) as u8 }
pub fn confidence_u8(self) -> u8      { (self.0 >> CONF_SHIFT) as u8 }
pub fn frequency(self) -> f32         { /* val/255 */ }
pub fn confidence(self) -> f32        { /* val/255 */ }
pub fn expectation(self) -> f32       { /* c × (f − 0.5) + 0.5 */ }
pub fn evidence_weight(self) -> f32   { /* c / (1 − c) */ }
```

### 3.3 Causal Mask (bits 40-42) — Pearl's 2³ Ladder

From `pearl.rs:11-49`:

```text
0b000 (None) → No planes active. Aggregate prior.
0b001 (O)    → Object only. Outcome marginal.
0b010 (P)    → Predicate only. Intervention marginal.
0b011 (PO)   → Predicate + Object. Level 2: Intervention P(Y|do(X)).
0b100 (S)    → Subject only. Entity marginal.
0b101 (SO)   → Subject + Object. Level 1: Association P(Y|X).
0b110 (SP)   → Subject + Predicate. Confounder detection.
0b111 (SPO)  → All planes active. Level 3: Counterfactual P(Y_x|X',Y').
```

Accessors at `edge.rs:221-278`:

```rust
pub fn causal_mask(self) -> CausalMask
pub fn set_causal_mask(&mut self, m: CausalMask)
pub const fn matches_causal(&self, query_mask: u8) -> bool
pub fn matches_causal_mask(&self, query_mask: CausalMask) -> bool
pub fn s_active(self) -> bool
pub fn p_active(self) -> bool
pub fn o_active(self) -> bool
```

Pearl-rung is **load-bearing** for predicate-pushdown paths (cycle-speed projection filters).

### 3.4 Direction triad (bits 43-45) — propagated polarity

`sign(palette[s_idx].dim0)`, same for P and O. NOT a static cache — `forward()` at `edge.rs:457` propagates direction from the *weight* edge, not from current palette state. Comment at line 457:

```rust
weight.direction(), // TODO: recompute from composed palette dim0 signs
```

That TODO marks direction as **load-bearing inference state**, not a derived value. Dropping direction requires either recomputing from palette (loses composition history) or accepting an information loss.

Accessors at `edge.rs:286-310`:

```rust
pub fn direction(self) -> u8
pub fn set_direction(&mut self, d: u8)
pub fn s_pathological(self) -> bool
pub fn p_pathological(self) -> bool
pub fn o_pathological(self) -> bool
```

### 3.5 Inference Type (bits 46-48) — NARS rule applied

From `edge.rs:9-26`:

```rust
pub enum InferenceType {
    Deduction = 0,   // A→B, B→C ⊢ A→C. Follow the chain.
    Induction = 1,   // A→B, A→C ⊢ B→C. Generalize from shared cause.
    Abduction = 2,   // A→B, C→B ⊢ A→C. Infer from shared effect.
    Revision = 3,    // Merge two truth values for the same statement.
    Synthesis = 4,   // Combine complementary evidence across domains.
    Reserved5 = 5,
    Reserved6 = 6,
    Reserved7 = 7,
}
```

5 used + 3 reserved. Read in `forward()` at `edge.rs:406-439` — selects the NARS rule that produces the output truth values.

### 3.6 Plasticity (bits 49-51) — per-plane palette-reassignment gate

From `plasticity.rs:6-92`:

```rust
pub struct PlasticityState(u8);

pub const ALL_FROZEN: Self = Self(0b000); // Established clinical pattern.
pub const ALL_HOT:    Self = Self(0b111); // New/uncertain edge.
pub const S_HOT:      Self = Self(0b001); // Only S-plane plastic.
pub const P_HOT:      Self = Self(0b010);
pub const O_HOT:      Self = Self(0b100);

pub fn s_hot(self) -> bool
pub fn p_hot(self) -> bool
pub fn o_hot(self) -> bool
pub fn freeze_s(self) -> Self
pub fn heat_s(self) -> Self
// ... plus per-plane freeze_p/o + heat_p/o
```

**Distinct from NARS confidence:** plasticity gates archetype reassignment per plane; confidence gates evidence weighting for revision. High `c` typically correlates with `ALL_FROZEN` but not always (e.g., clinical-pattern lock vs. exploratory mode).

### 3.7 Temporal index (bits 52-63) — 4096-slot cycle bucket

12-bit coarse time bucket. Read in `forward()` at `edge.rs:446`:

```rust
let t_out = self.temporal().max(weight.temporal());
```

**This is the field that is genuinely redundant** with (a) AriGraph `Triplet.timestamp: u64` (at commit time) and (b) `SpoWitnessChain` position (per-cycle order). It is the leading candidate for v2 bit-reclaim.

---

## 4. Key Methods

### 4.1 `pack()` — full constructor

`edge.rs:84-110`:

```rust
pub fn pack(
    s_idx: u8, p_idx: u8, o_idx: u8,
    frequency: u8, confidence: u8,
    causal_mask: CausalMask,
    direction: u8,
    inference: InferenceType,
    plasticity: PlasticityState,
    temporal: u16,
) -> Self
```

### 4.2 `forward()` — the BNN-style cognitive composition

`edge.rs:393-462` — composes two edges (self = activation, weight = learned edge) into a new edge:

1. **Palette composition** (lines 401-403): three 256×256 compose-table lookups (compose_s, compose_p, compose_o) — one per plane. O(1) per plane.
2. **NARS truth propagation** (lines 406-439): switch on `weight.inference_type()` → applies Deduction / Induction / Abduction / Revision / Synthesis truth formula → produces `(f_out, c_out)`.
3. **Causal mask AND** (lines 442-443): `mask_out = self.causal_mask() & weight.causal_mask()` — only planes active in both survive.
4. **Temporal max** (line 446): `t_out = self.temporal().max(weight.temporal())`.
5. **Inherit plasticity** from weight (line 459).

This IS the cycle-speed cognitive forward pass — pure register operations + three 256×256 table lookups. Cache-friendly when the compose tables are L2-resident.

### 4.3 `learn()` — evidence-driven revision

`edge.rs:464+` (Learning section). Applies NARS revision rule to merge observed evidence with existing truth; updates plasticity flags based on confidence transitions.

### 4.4 Distance accessors

`edge.rs:370-384` (per-plane Hamming distance via palette distance tables):

```rust
pub fn distance_masked(&self, other: &Self,
                       s_dm: &[u8; 65536],
                       p_dm: &[u8; 65536],
                       o_dm: &[u8; 65536],
                       mask: u8) -> u32
```

Filtered by 3-bit mask — only sums the planes active in the mask. O(1) per plane.

---

## 5. Consumers

### 5.1 `lance-graph-planner::cache::nars_engine`

Per CLAUDE.md "Session: AutocompleteCache + p64 Convergence":

> *`nars_engine.rs`: SpoHead, Pearl 2³, NarsTables (causal-edge hot path), StyleVectors*

The planner re-exports CausalEdge64 from the causal-edge crate, packages it into `NarsTables` (precomputed NARS lookup tables), and consumes it in the AutocompleteCache.

### 5.2 `cognitive-shader-driver::BindSpace::EdgeColumn`

Per CLAUDE.md "AGI-as-glove" doctrine:

> *Planner (why/how, causal composition) = a write to `EdgeColumn` (`CausalEdge64`). Never a new bridge.*

`EdgeColumn` is one of the 4 BindSpace columns in the SoA: `FingerprintColumns / QualiaColumn / MetaColumn / EdgeColumn`. The SPO-palette `CausalEdge64` is the per-row entry in `EdgeColumn`.

### 5.3 AriGraph SPO commit path

The `spo_bridge::promote_to_spo()` gate (per W5 spec for sprint-10) commits CausalEdge64-grounded inferences into AriGraph as `Triplet` rows when truth confidence passes a `TruthGate` threshold. The CausalEdge64's S/P/O palette indices map to AriGraph subject/predicate/object entity references.

### 5.4 `p64::convergence`

Per `crates/lance-graph-planner/src/cache/convergence.rs:1-23`:

```text
Cold path (columns/rows):
  AriGraph TripletGraph → SPO strings → DataFusion → Arrow

Hot path (p64 palette):
  AriGraph Triplets → Base17 fingerprints → Palette → CognitiveShader
    → 8 predicate layers × 64×64 attention = 4096 heads
    → CausalEdge64 forward/learn = O(1) per head
    → NarsTables revision = O(1) per truth update

Convergence:
  Cold path BUILDS the graph (via LLM, slow)
  Hot path SERVES the graph (via palette, fast)
  p64 IS the bridge between them
```

This is the canonical convergence — the SPO-variant CausalEdge64 is the unit of "O(1) per head" at the hot path.

---

## 6. Hot-Path Role

**Zone-1 (cycle-speed):** YES. CausalEdge64 is the per-row payload of `BindSpace::EdgeColumn`; SoA sweeps over the column run `forward()` and `learn()` at O(1) per row + three 256×256 table lookups. Cache-resident; ~50-200 ns per row at typical CPU speeds when tables are L2-hot.

**Zone-2 (SPO-as-3D-vector ANN search):** Indirect — CausalEdge64's `distance_masked()` is the per-edge primitive that the `blasgraph` + `neighborhood` cascade (HEEL → HIP → TWIG → LEAF) ranks edges by. The S/P/O palette indices are the addresses; the distance is computed via per-plane 256×256 palette-distance tables.

**Zone-3 (DataFusion cold path):** Indirect — when triples promote to AriGraph SPO, they become `Triplet` rows that DataFusion can query columnwise. The CausalEdge64's truth values flow into the persisted truth columns. But the edge itself is not read directly by DataFusion.

---

## 7. Cross-references

- **Paired with:** `causal-edge-64-thinking-engine-variant.md` (the OTHER `CausalEdge64` type — same name, different bit layout, different semantics, different consumers).
- **Synergies + drift analysis:** `causal-edge-64-synergies-and-pr-trajectory.md`.
- **AGI-as-glove doctrine:** `lab-vs-canonical-surface.md` (the 4 BindSpace columns; CausalEdge64 = EdgeColumn).
- **Encoding stack:** `encoding-ecosystem.md` (Full → ZeckBF17 → BGZ17 → CAM-PQ → Scent → CausalEdge64).
- **NARS semantics:** `lance-graph-planner/src/cache/nars_engine.rs` (NarsTables, SpoHead, MASK_SPO).
- **Pearl ladder:** `causal-edge/src/pearl.rs` (CausalMask enum).
- **Plasticity:** `causal-edge/src/plasticity.rs` (PlasticityState).
- **p64 convergence:** `lance-graph-planner/src/cache/convergence.rs` (the bridge where this variant lives in the hot path).
- **Sprint-10 v2 layout proposal:** `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` (parent plan §3); `.claude/specs/pr-ce64-mb-2-causaledge64-v2.md` (W2 spec); `.claude/specs/pr-ce64-mb-2-pal8-nars-regression.md` (W3 spec).
- **SPOW tetrahedron (witness as 4th vertex):** `.claude/plans/oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §8.

---

## 8. The v2 Bit-Reclaim Decision

This variant is the target of the sprint-10 CausalEdge64-v2 work (PR-CE64-MB-2). The bit-reclaim debate (CSI-1 in `.claude/board/sprint-log-10/meta-review.md`) concerns which bits to drop to make room for new fields (G-slot, W-slot, truth-band lens). Current meta-review recommendation: **drop temporal (12b) + drop G_slot-from-edge (5b) = 17b freed**; keep direction/plasticity/inference (load-bearing dispatch payload per the corrected hot-path analysis).

See `causal-edge-64-synergies-and-pr-trajectory.md` §4 for the full bit-reclaim trade analysis.

---

*Last verified: 2026-05-14 against shipped `crates/causal-edge/src/edge.rs` + `pearl.rs` + `plasticity.rs` + `convergence.rs`.*
