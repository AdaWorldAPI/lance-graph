> **⊘ SUPERSEDED 2026-07-10 (D-TSC-1 / M9).** This pre-eviction doc still
> cites crewai-rust/n8n-rs as live consumers (evicted 2026-06-21) and predates
> the family/runbook split (E-STYLE-FAMILY-VS-RUNBOOK-1). Canonical: contract
> `style_family.rs` (12 families) + `thinking.rs` (36 runbooks). Kept for
> history; do not cite as source of truth.

# Thinking Styles — Canonical 36 Styles in 6 Clusters

**Source of Truth**: `crates/lance-graph-contract/src/thinking.rs` (commit b4b43399)

This module defines the **single source of truth** for thinking styles used across the entire AdaWorldAPI ecosystem:
- `lance-graph-planner`
- `crewai-rust`
- `n8n-rs`
- `ladybug-rs`

Nobody re-defines these types.

## Reconciliation History

- **crewai-rust**: Originally 36 styles / 6 clusters (Analytical, Creative, Empathic, Direct, Exploratory, Meta)
- **lance-graph-planner**: Previously 12 styles / 4 clusters (Convergent, Divergent, Attention, Speed)

**Now canonical**: 36 styles / 6 clusters. The planner's coarser 12-style view (and the 12-style auto-detection in `cognitive-shader-driver`) maps to cluster representatives + attention/speed extras. The planner internally uses `ThinkingStyle::planner_cluster()` for cost modeling.

## The 36 Canonical Styles

```rust
#[repr(u8)]
pub enum ThinkingStyle {
    // Analytical Cluster (τ 0x40-0x4F)
    Logical = 0, Analytical = 1, Critical = 2, Systematic = 3, Methodical = 4, Precise = 5,

    // Creative Cluster (τ 0xA0-0xAF)
    Creative = 6, Imaginative = 7, Innovative = 8, Artistic = 9, Poetic = 10, Playful = 11,

    // Empathic Cluster (τ 0x80-0x8F)
    Empathetic = 12, Compassionate = 13, Supportive = 14, Nurturing = 15, Gentle = 16, Warm = 17,

    // Direct Cluster (τ 0x60-0x6F)
    Direct = 18, Concise = 19, Efficient = 20, Pragmatic = 21, Blunt = 22, Frank = 23,

    // Exploratory Cluster (τ 0x20-0x2F)
    Curious = 24, Exploratory = 25, Questioning = 26, Investigative = 27, Speculative = 28, Philosophical = 29,

    // Meta Cluster (τ 0xC0-0xCF)
    Reflective = 30, Contemplative = 31, Metacognitive = 32, Wise = 33, Transcendent = 34, Sovereign = 35,
}
```

### Cluster Mapping

| Cluster     | τ Base | Styles (u8)     | PlannerCluster |
|-------------|---------|-----------------|----------------|
| Analytical  | 0x40    | 0–5            | Convergent     |
| Creative    | 0xA0    | 6–11           | Divergent      |
| Empathic    | 0x80    | 12–17          | Attention      |
| Direct      | 0x60    | 18–23          | Convergent     |
| Exploratory | 0x20    | 24–29          | Divergent      |
| Meta        | 0xC0    | 30–35          | Speed          |

**PlannerCluster** (coarser 4-way model for cost decisions):
- **Convergent**: Analytical + Direct → depth-first, precise
- **Divergent**: Creative + Exploratory → breadth-first, exploratory
- **Attention**: Empathic → focus allocation, peripheral awareness
- **Speed**: Meta → System 1 vs System 2 deliberation speed

## Core Methods on ThinkingStyle

- `cluster(&self) -> StyleCluster`
- `planner_cluster(&self) -> PlannerCluster`
- `tau(&self) -> u8` : JIT macro address for `n8n-rs` CompiledStyleRegistry and scan kernel lookup
- `ALL: [ThinkingStyle; 36]` : Canonical ordered array

## FieldModulation — The 7D Cognitive Control Surface

```rust
pub struct FieldModulation {
    pub resonance_threshold: f64,  // 0.0=exact → 1.0=broad
    pub fan_out: usize,            // alternatives per decision
    pub depth_bias: f64,           // 0.0=breadth → 1.0=depth
    pub breadth_bias: f64,
    pub noise_tolerance: f64,      // 0.0=strict → 1.0=tolerant
    pub speed_bias: f64,           // 0.0=thorough → 1.0=fast
    pub exploration: f64,          // 0.0=exploit → 1.0=explore
}
```

### Derived Types

- `ScanParams`: SIMD parameters (threshold, top_k, prefetch_ahead, filter_mask) for ndarray / jitson / BindSpace bgz17 scans
- `to_fingerprint(&self) -> [u8; 7]`: Compact 7-byte encoding (prefix 0x0D in BindSpace Meta columns)

**Default** provided with balanced values.

## ThinkingStyleProvider Trait

```rust
pub trait ThinkingStyleProvider: Send + Sync {
    fn style_vector(&self, style: ThinkingStyle) -> SparseVec;  // 23D sparse (crewai-rust)
    fn default_modulation(&self, style: ThinkingStyle) -> FieldModulation;
    fn select_from_assessment(&self, assessment: &MulAssessment) -> ThinkingStyle;
}
```

- `crewai-rust`: Full 23D vectors + rich implementations
- `lance-graph-planner`: Simplified 7D projection
- `select_from_assessment`: Core of adaptive meta-cognition (MUL → style switch)

## Integration with the Rest of the Stack

### 1. Cognitive Shader Driver & L1-4 Cycle (ShaderDispatch)
- `StyleSelector` enum in `cognitive_shader.rs` (Ordinal(u8), Named(&str), Auto)
- `auto_style.rs`: `resolve(sel, qualia: &[f32]) -> u8` — rule-based mapping from 18D qualia vector or explicit selection to 0–11 style ordinal (cluster reps + extras)
- Modulates: resonance sweep (bgz17), cycle energy, promotion membrane, collapse gate, emitted CausalEdge64
- `MetaWord.thinking: u6` (in packed u32 prefilter) likely encodes current style for cheap BindSpace filtering before fingerprint loads

**Note**: `auto_style` currently hardcodes 12 styles (0–11). Future work: map from canonical 36.

### 2. NARS + CausalEdge64 + Pearl Masks (nars_engine.rs)
- Thinking styles used as **weight vectors over the 8 Pearl causal projections** (association → counterfactual)
- Canonical 36 styles enable *style-specific* causal attention (e.g. `Metacognitive` heavily weights SPO counterfactuals; `Direct` favors association)
- `CausalEdge64` + style modulation = per-cycle causal reasoning flavored by current thinking style
- MUL-driven style selection feeds directly into NARS inference type and truth revision

### 3. Planner Cost Model & Search Strategy
- `PlannerCluster` drives convergent vs divergent search bias, fan-out, depth/breadth
- `FieldModulation` knobs directly control the planner's decision process and are converted to `ScanParams` for hot-path BindSpace queries

### 4. MUL (Meta-Uncertainty Layer) & Meta-Awareness
- `select_from_assessment(MulAssessment)` is the key hook for the "system can't stop thinking" loop
- Style switching based on uncertainty, contradiction detection, or entropy provides the meta-cognitive layer on top of NARS + resonance

### 5. BindSpace, CAM, HHTL Cascade
- `to_fingerprint()` stores style context persistently in SoA columns
- Style-modulated `ScanParams` tune the HHTL (HighHeelBGZ) cascade and CAM execution
- Enables style-aware top-k, resonance, and promotion membrane behavior

## Epiphanies & High-Signal Insights

1. **The 7D Modulation Surface is the "Cognitive Shader Program"**: Instead of hard-coded behavior, every thinking style carries a portable 7D parameter vector that reconfigures the entire L1-4 resonance + NARS + planner pipeline on the fly. This is extremely powerful for adaptive AGI-like behavior.

2. **Style + Pearl = Causal Reasoning Flavors**: The fusion of 36 thinking styles with 8 Pearl masks in `CausalEdge64` creates a rich space of causal inference *personalities*. Metacognitive style naturally does more counterfactual thinking; Exploratory does more speculative abduction, etc.

3. **Qualia → Style → Modulation is Closed-Loop Meta-Cognition**: The shader reads qualia (from previous cycle or sensors), auto-selects style, applies modulation, runs resonance + NARS, emits new CausalEdges and updated qualia. This is the "can't stop thinking" engine.

4. **Low-Entropy Propagation**: τ(u8) + ordinal(u8) + 7-byte fingerprint + 6-bit MetaWord + planner_cluster(u2) = multiple cheap encodings for hot path (20-200ns) and cold path.

5. **Reconciliation Success**: Moving the full 36 here eliminates duplication and drift between crewai agents and the graph planner/shader. All components now share the same mental ontology.

## Technical Debt & Open Items

- **12 vs 36 Gap**: `auto_style.rs` and planner's internal 12-style view need an explicit mapping or derivation from the canonical 36 (e.g. one representative per cluster + dedicated Attention/Speed styles).
- **StyleSelector Range**: Currently `Ordinal(n) => n % 12`. Should support full 0-35 range when using canonical `ThinkingStyle`.
- **23D Vectors**: Live only in crewai-rust. Consider a default static table or proc-macro generation here for standalone use / testing.
- **Cross-Crate Testing**: Good unit tests in this file, but missing integration tests that exercise `ThinkingStyleProvider` impls + shader dispatch + NARS together.
- **Documentation Sync**: Ensure all downstream crates (crewai-rust etc.) point to this contract module.

## Alignment with Overall Architecture Vision

This module is central to the **inner ontology (Zone 1 hot path)**:
- Powers the cognitive-shader-driver SoA L1-4 cycle
- Feeds NARS via style-modulated Pearl projections
- Uses BindSpace zero-copy + CAM for style context
- MUL + style selection = meta-awareness layer
- OGIT / COCA codebooks can ground style names for outer ontology (Zone 2/3) and Palantir Foundry-like surfaces

**Potential**: With this in place, the next high-leverage work is wiring style modulation directly into the hot-path `cognitive-shader-driver` dispatch and extending `nars_engine` to use per-style Pearl weight vectors from the canonical set.

---

*Generated from analysis of thinking.rs + cross-references to cognitive_shader.rs, auto_style.rs, nars_engine.rs, and prior architecture docs.*
*Last updated: 2026-05-10*
