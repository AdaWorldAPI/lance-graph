# Thinking Pipeline Architecture

## Overview

The thinking pipeline gives the lance-graph query planner the ability to
REASON about queries using cognitive orchestration. 36 thinking styles are
nodes in a real `AdjacencyStore`. 13 cognitive verbs traverse this graph.
NARS truth values on edges encode empirically learned relationships.

## Why It Exists

The planner had 36 thinking styles and 13 orchestration methods that were
never used. `select_from_mul()` was a switch statement that always picked
Analytical. The styles were dead code.

Now: the styles are graph nodes with NARS-weighted edges. Activating one
style materializes its entire cognitive neighborhood. The neighborhood's
shape (coherence, tension, evidence mass) drives planning decisions
without any external "meta-awareness" module.

## Architecture

```
┌────────────────────────────────────────────────────────┐
│  plan_autonomous(query)                                 │
│    │                                                    │
│    ├── estimate_query_complexity(query)  → f64           │
│    ├── auto-generate SituationInput                      │
│    └── plan_with_topology(query, &situation)             │
│          │                                               │
│          ├── MUL assessment → seed style                 │
│          ├── ThinkingTopology.activate(seed)              │
│          │     → CognitiveCluster {                      │
│          │         activations,   // co-fire styles      │
│          │         contradictions, // fork targets       │
│          │         deepeners,     // second-pass styles  │
│          │         validators,    // meta-observers      │
│          │         shape: ClusterShape {                  │
│          │           coherence,    // high = confident   │
│          │           tension,      // high = fork        │
│          │           meta_coverage, // validators active │
│          │           evidence_mass, // topology maturity │
│          │         }                                     │
│          │       }                                       │
│          │                                               │
│          ├── if cluster.should_fork():                    │
│          │     execute_forks(query, fork_styles)          │
│          │     → parallel plans from contradiction styles │
│          │                                               │
│          ├── primary plan with cluster's blended modulation│
│          │                                               │
│          └── learn: observe_plan_quality()                │
│                → NARS revision updates topology edges    │
│                → topology self-organizes over time        │
└────────────────────────────────────────────────────────┘
```

## The Three Layers

### Layer 1: ThinkingTopology (topology.rs)

12×12 adjacency matrix for the planner's cost model.

**Nodes:** 12 thinking styles (Analytical, Creative, Focused, etc.)
**Edges:** StyleRelation (Activates, Contradicts, Deepens, Validates) + NARS TruthValue
**Activation:** Kuzu-style batch — touch one style → neighborhood materializes
**Learning:** NARS revision from empirical co-activation results
**Expansion:** 2-hop with deduction attenuation when cluster can't resolve

The topology is seeded with structural priors (c=0.3) and learns from experience.
Edges can flip: enough negative evidence turns Activates → Contradicts.

### Layer 2: CognitiveProcess (process.rs)

13 verbs as composable traversal programs on the full 36-style topology.

**Verbs modify CognitiveState:**
- `active_styles` — which of 36 are lit up, with activation strength
- `modulation` — 7D field params (resonance, fan_out, depth, breadth, noise, speed, exploration)
- `beliefs` — accumulated NARS-weighted beliefs at sigma stages (Ω→Δ→Φ→Θ→Λ)
- `tensions` — style pairs that disagree (the disagreement IS information)
- `entropy` — spine/flesh/connectivity of traversed region
- `trail` — full execution trace for cockpit rendering

**Programs compose verbs with branching:**
```rust
CognitiveProcess::new("deep_analysis")
    .then(Explore)
    .then(Fanout)
    .then(Hypothesis)
    .then(Deepen)
    .branch_if_tense(0.4,
        CognitiveProcess::new("resolve_tension")
            .then(Counterfactual)
            .then(Synthesis),
    )
    .then(Review)
    .then(Application)
```

### Layer 3: ThinkingGraph (graph.rs)

36 thinking styles as a REAL `AdjacencyStore` — same CSR/CSC, same edge
properties, same batch operations as data graph queries.

**The key insight:** cognitive verbs call the same code that data queries use.
`batch_adjacent()`, `adjacent_truth_propagate()`, `adjacent_incoming()`.
One engine, two graphs.

```
Data query:     store.batch_adjacent(&[node_42])
Cognitive verb: thinking_graph.store.batch_adjacent(&[Analytical_idx])
                ↑ same function, different graph
```

## The 13 Cognitive Verbs

| Verb | Graph Operation | Sigma Stage |
|------|----------------|-------------|
| EXPLORE | batch_adjacent → truth_propagate → detect tensions | Ω observe |
| FANOUT | k × batch_adjacent (breadth-first) | Ω observe |
| HYPOTHESIS | strongest activation → generate Delta belief | Δ insight |
| COUNTERFACT | negate belief, boost Contradicts edges | Φ belief |
| SYNTHESIS | semiring.add (NARS revision) across beliefs | Θ integrate |
| ABDUCTION | adjacent_incoming (CSC backward traversal) | Δ insight |
| REVIEW | activate Meta cluster (30-35), reduce noise | Φ belief |
| APPLICATION | project to Lambda trajectory | Λ trajectory |
| EXTRAPOLATE | extend trajectory, attenuate confidence | Λ trajectory |
| INDUCTION | strengthen edges between co-active styles | Φ belief |
| INTERRELATE | filter Bridges edges, cross-cluster activation | Θ integrate |
| DEEPEN | filter Deepens edges only, narrow beam | Φ belief |
| MODULATE | JIT-adjust modulation from topology entropy | Φ belief |

## Topology Entropy

The health of the region being traversed determines verb selection:

```
spine_density:  ratio of high-confidence edges (c > 0.5)
flesh_ratio:    ratio of low-confidence edges (c < 0.2)
connectivity:   ratio of bidirectional edges
missing_edges:  expected edges that don't exist
entropy:        Shannon entropy over confidence distribution [0,1]
```

- `needs_exploration()` → high entropy or many missing edges → EXPLORE/FANOUT
- `is_stable()` → high spine, low entropy → DEEPEN/APPLICATION
- Programs use entropy for branching: `branch_if_uncertain(0.6, subprocess)`

## Persistence

Topology serializes to 1,440 bytes (12×12 × 10 bytes per edge entry).
Load/save via `planner.save_topology()` / `planner.load_topology(&bytes)`.
JSON export via `planner.topology_json()` for cockpit rendering.

## Connection Points

### → DeepNSM-CAM

`ThinkingGraph.with_semantics(NsmEncoder)` attaches the semantic layer.
Cognitive verbs then reason about word MEANING via distance matrix lookups
and calibrated similarity scores. See `docs/DEEPNSM_ARCHITECTURE.md`.

### → Elevation System

When a cognitive process can't converge (RepeatUntilStable hits max iterations),
it maps to ElevationLevel escalation:
- PaletteEdge (3 bytes) → CognitiveProcess quick_assess
- CAM-6 → CognitiveProcess deep_analysis
- Base17 → CognitiveProcess scientific_method
- Full 96D → expand topology → meta_calibration

### → Cockpit

`VerbTrace` in each CognitiveState step gives:
- Which styles are active (with activation strength)
- How modulation changed
- Current entropy
- Beliefs produced

Render as animated 36-node force graph:
- Node size = activation strength
- Edge color = relation type (green=Activates, red=Contradicts, blue=Deepens, yellow=Validates)
- Edge thickness = NARS confidence
- Background = entropy (dark=stable, bright=uncertain)
- Pulse = currently executing verb
