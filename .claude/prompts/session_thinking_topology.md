# SESSION: Thinking Topology + Cognitive Process Engine

## What Was Built

Three layers replacing flat `select_from_mul()` switch statement with
adjacency-graph-driven cognitive orchestration using real lance-graph infrastructure.

### Layer 1: topology.rs (904 lines, 13 tests)

12×12 NARS-weighted adjacency graph over the planner's thinking styles.

- `ThinkingTopology` — 12×12 matrix with `StyleEdge` containing `StyleRelation` + `TruthValue`
- `CognitiveCluster` — Kuzu-style batch activation: touch one style → entire neighborhood materializes
- `ClusterShape` — coherence/tension/meta_coverage/evidence_mass derived from topology structure
- NARS learning: `observe_coactivation()` uses revision to update edge truth from empirical results
- `expand()` — 2-hop neighborhood expansion with NARS deduction attenuation
- Cluster shape IS the MUL input — no separate assessment construction needed
- `should_fork()` — high tension + mature topology → fork parallel plans

### Layer 2: process.rs (1,308 lines, 17 tests)

13 cognitive verbs as composable traversal programs on a 36-node style topology.

**Verbs:**
```
EXPLORE      → activate 1-hop, detect tensions
FANOUT       → k-hop breadth-first exploration
HYPOTHESIS   → abductive leap: Omega → Delta belief
COUNTERFACT  → negate belief, activate contradiction edges
SYNTHESIS    → NARS revision across beliefs (semiring.add)
ABDUCTION    → trace backward via CSC (incoming edges)
REVIEW       → activate Meta cluster (30-35), reduce noise
APPLICATION  → project to Lambda trajectory
EXTRAPOLATE  → extend trajectory forward
INDUCTION    → strengthen edges between co-active styles
INTERRELATE  → cross-cluster bridge activation
DEEPEN       → follow only Deepens edges, narrow beam
MODULATE     → JIT-adjust modulation from entropy
```

**Programs (composable verb sequences):**
- `quick_assess()` — EXPLORE → REVIEW → APPLICATION
- `deep_analysis()` — EXPLORE → FANOUT → HYPOTHESIS → DEEPEN → [if tense: COUNTERFACT → SYNTHESIS] → REVIEW → APPLICATION
- `creative_divergence()` — EXPLORE → FANOUT → INTERRELATE → HYPOTHESIS → [if uncertain: FANOUT → EXPLORE] → SYNTHESIS → REVIEW
- `scientific_method()` — EXPLORE → HYPOTHESIS → COUNTERFACT → INDUCTION → [repeat: DEEPEN → REVIEW] → SYNTHESIS → APPLICATION
- `abductive_trace()` — EXPLORE → ABDUCTION → FANOUT → HYPOTHESIS → [if tense: INTERRELATE → SYNTHESIS] → REVIEW → EXTRAPOLATE
- `meta_calibration()` — EXPLORE → REVIEW → MODULATE → [repeat: EXPLORE → INDUCTION → MODULATE]
- `select_process()` — auto-select from query complexity + topology entropy

**Branching constructs:**
- `BranchIfTense` — fork if contradiction tension exceeds threshold
- `BranchIfUncertain` — fork if topology entropy exceeds threshold
- `RepeatUntilStable` — loop subprocess until entropy converges

**Entropy spine:**
- `TopologyEntropy` — spine_density, flesh_ratio, connectivity, missing_edges, entropy
- `needs_exploration()` — high entropy or many missing edges
- `is_stable()` — high spine density, low entropy

### Layer 3: graph.rs (840 lines, 16 tests)

The kill shot: 36 thinking styles as a REAL `AdjacencyStore` with NARS edge properties.
Every cognitive verb calls the SAME infrastructure as data graph queries.

**Key difference from process.rs shadow matrix:**
- `ThinkingGraph.store` IS an `AdjacencyStore` — real CSR/CSC
- Edge properties have `truth_f`, `truth_c` columns + `relation` column
- Verbs call `store.batch_adjacent()` — Kuzu's columnar batch primitive
- SYNTHESIS uses `TruthPropagatingSemiring.add()` — real NARS revision
- ABDUCTION uses `store.adjacent_incoming()` — real CSC backward traversal
- INTERRELATE filters by `EdgeKind::Bridges` — relation-aware graph ops
- Entropy computed from real edge confidence values

**Verb → Graph Operation mapping:**
```
EXPLORE       → batch_adjacent() + adjacent_truth_propagate()
FANOUT        → k × batch_adjacent()
ABDUCTION     → adjacent_incoming() (CSC backward)
SYNTHESIS     → semiring.add() = NARS revision
COUNTERFACT   → batch_adjacent() → filter Contradicts → boost
INTERRELATE   → batch_adjacent() → filter Bridges cross-cluster
DEEPEN        → batch_adjacent() → filter Deepens only
```

### Additional files:

- `persistence.rs` (199 lines, 4 tests) — binary serialize/deserialize (1440 bytes) + JSON export
- `mod.rs` (163 lines) — `orchestrate_with_topology()` replaces flat dispatch
- `api.rs` (+148 lines, 9 tests) — `plan_with_topology()`, `plan_autonomous()`, save/load

## Total Output

```
thinking/
├── style.rs          (252)   12 base styles, modulation, scan params
├── topology.rs       (904)   12×12 NARS adjacency, activate(), expand(), learn
├── process.rs       (1308)   36×36 graph + 13 verbs + composable programs
├── graph.rs          (840)   ThinkingGraph on real AdjacencyStore
├── persistence.rs    (199)   binary serialize/deserialize + JSON
├── sigma_chain.rs    (245)   Ω→Δ→Φ→Θ→Λ epistemic lifecycle
├── nars_dispatch.rs  (131)   query shape → NARS inference type
├── semiring_selection(113)   auto-select semiring from context
└── mod.rs            (163)   orchestrate + orchestrate_with_topology
                     ─────
                     4,155 lines, 60 tests
```

## Branch

`claude/unified-query-planner-aW8ax` in AdaWorldAPI/lance-graph

## API Surface

```rust
// CC just calls this — topology does the thinking
planner.plan_autonomous(query)

// Or with explicit situation
planner.plan_with_topology(query, &situation)

// Topology persists across sessions
let bytes = planner.save_topology();   // 1440 bytes
planner.load_topology(&bytes)?;

// Inspect
planner.topology_json()                // for cockpit
planner.topology_stats()               // edge_count, evidence, tensions
```

## Connection to DeepNSM-CAM

The ThinkingGraph + CognitiveProcess pipeline is WHERE DeepNSM plugs in.
When verbs reason about concepts, DeepNSM provides the MEANING layer:

```
ThinkingGraph.with_semantics(NsmEncoder)
  → SYNTHESIS uses triple_similarity for merge decisions
  → COUNTERFACTUAL negates predicate plane
  → INTERRELATE finds cross-domain analogies via subject similarity
  → MODULATE shifts thinking style based on content PoS
  → DEEPEN resolves words into NSM prime decomposition
```

See: `.claude/prompts/session_deepnsm_cam.md`
See: `AdaWorldAPI/DeepNSM/.claude/prompts/session_deepnsm_cam.md`
