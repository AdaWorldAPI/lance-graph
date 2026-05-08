# MetaOrchestrator + Thinking Styles

**Location**: `crates/lance-graph/src/graph/arigraph/orchestrator.rs`

## Core Concept

The `MetaOrchestrator` is the **meta-cognitive controller** of the system.

It decides *which thinking style* to use at every step and can switch between learned (NARS topology) and hardcoded modes based on self-assessment.

## Four Thinking Styles (`AgentStyle`)

| Style    | Cognitive Mode          | Typical Use                     |
|----------|-------------------------|---------------------------------|
| `Plan`   | Analytical / Convergent | Deep sequential reasoning       |
| `Act`    | Focused / Deductive     | Precise single-action selection |
| `Explore`| Exploratory / Divergent | Lateral expansion, discovery    |
| `Reflex` | Metacognitive / Revision| Learning from mistakes, healing |

## Key Mechanisms

### 1. MUL — Meta-Uncertainty Layer
- Detects Dunning-Kruger position (`MountStupid`, `ValleyOfDespair`, etc.)
- Assesses Trust Texture, Flow State, Compass override
- Produces `free_will_modifier` that scales trust in learned topology

### 2. StyleTopology (NARS at meta level)
- 4×4 directed graph of style transitions
- Each edge carries a `TruthValue` (NARS frequency + confidence)
- `record_outcome()` performs NARS revision on transitions

### 3. Adaptive vs Hardcoded Fallback
- Starts in **Adaptive** mode (uses learned topology)
- Falls back to `Plan → Act → Explore → Reflex` when efficiency drops
- Automatically restores adaptive mode when performance recovers

## Connection to CausalEdge64

The orchestrator produces `StepResult` containing the chosen style. This style can (and should) influence:
- Which `InferenceType` is emphasized
- Which `CausalMask` planes are prioritized
- Plasticity decisions

---

*This is one of the strongest metacognition implementations seen in cognitive architectures.*