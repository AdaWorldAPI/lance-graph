# META INTEGRATION PLAN
# Knowledge Graph + Thinking + Semantic Understanding

> One graph. Three memory layers. Thirteen cognitive verbs. 4,096 words.
> Zero learned weights. Zero GPU. Zero LLM calls in the hot path.
> The graph IS the memory. The memory IS the knowledge. The knowledge IS the thinking.

## 1. THE VISION

A cognitive graph engine that:
- Reads text and understands meaning (DeepNSM: 12-bit tokens, PoS parsing, SPO triples)
- Accumulates knowledge as graph structure (AriGraph: semantic + episodic memory)
- Reasons about what it knows (ThinkingGraph: 13 cognitive verbs, NARS truth propagation)
- Learns what works (topology: NARS revision on style edges, empirical self-organization)
- Operates entirely in CPU cache (9MB runtime, 8MB distance matrix, fits L2/L3)

This is not a chatbot. It's a knowledge engine that thinks about graph queries.

## 2. THE STACK (bottom to top)

```
LAYER 8: Microcode       36 YAML templates → JIT → LazyLock → NARS RL
LAYER 7: Application     query → plan → execute → result
LAYER 6: AriGraph        observe → retrieve → plan → decide (AriGraph transcode)
LAYER 5: Cognitive Verbs  EXPLORE/HYPOTHESIS/SYNTHESIS/... (13 verbs, composable)
LAYER 4: ThinkingGraph   36 styles × NARS adjacency × topology learning
LAYER 3: DeepNSM         tokenize → parse → encode → similarity → context
LAYER 2: bgz17           SimilarityTable, Base17, SpoBase17, distance calibration
LAYER 1: cam_pq          6-byte fingerprints, stroke cascade, AVX-512
LAYER 0: AdjacencyStore  CSR/CSC, batch_adjacent, truth_propagate, semirings
```

Every layer calls the layer below. No layer skips. No lateral dependencies.
The entire stack compiles to one binary. Zero serialization between layers.

## 3. THE THREE MEMORY LAYERS

### Semantic Memory (persistent knowledge)
```
What:   4,096 concept nodes + SPO relation edges
Where:  AdjacencyStore with NARS truth values on edges
How:    Observation → parse → add/revise triples
Growth: Bounded by vocabulary (4K nodes), unbounded edges
Query:  batch_adjacent() + truth_propagate() + SimilarityTable
```

### Episodic Memory (temporal experiences)
```
What:   Episode nodes linked to semantic triples + temporal chain
Where:  ContextWindow (±5 sentence ring buffer, 10K-bit VSA bundles)
How:    Each sentence → BitVec → push into window → link to semantic
Growth: Ring buffer (fixed 11 slots), older episodes evicted
Query:  contextualize(word) = word XOR context → disambiguated
```

### Thinking Memory (meta-cognitive topology)
```
What:   36 ThinkingStyle nodes + NARS-weighted activation edges
Where:  ThinkingGraph (AdjacencyStore, same CSR as semantic layer)
How:    Each query → activate style → cluster → fork/merge → learn
Growth: Fixed topology (36 nodes), edge weights evolve via NARS revision
Query:  activate(style) → CognitiveCluster → shape → decision
```

### Cross-Layer Connections
```
Episodic → Semantic:   EXTRACTED edges (which triples came from which episode)
Semantic → Thinking:   MODULATE verb (content PoS drives thinking style selection)
Thinking → Semantic:   Cognitive verbs (EXPLORE/SYNTHESIS/etc) operate on semantic graph
Context  → All:        ContextWindow disambiguates words across all layers
```

## 4. THE DATA FLOW

```
Raw Text
  │
  ▼
LAYER 3: DeepNSM Tokenizer ─────────────────────────────────────────────
  │  "the big dog bit the old man"
  │  → [(1,a), (156,j), (671,n), (2943,v), (1,a), (174,j), (95,n)]
  │  12-bit tokens + 4-bit PoS = 16 bits per word. O(n) hash lookup.
  │
  ▼
LAYER 3: DeepNSM Parser ────────────────────────────────────────────────
  │  → SPO(dog:671, bite:2943, man:95)
  │  → Mod(big:156 → dog:671), Mod(old:174 → man:95)
  │  6-state PoS FSM. O(n). No regex. No LLM.
  │
  ├─────────────────┐
  ▼                 ▼
LAYER 6: AriGraph   LAYER 3: DeepNSM Context
  Observe            ContextWindow.push(sentence_bitvec)
  │                  ±5 sentences, O(1) update
  │                  Disambiguates all subsequent lookups
  ▼
LAYER 0+2: Semantic Graph Update ────────────────────────────────────────
  │  AdjacencyStore.add_edge(671, 95, predicate=2943)
  │  If edge exists: NARS revision (merge new evidence with old)
  │  If edge new: TruthValue::new(1.0, 0.3) — moderate initial confidence
  │  Episode node created, linked to triple via EXTRACTED edge
  │
  ▼
LAYER 6: AriGraph Retrieve ──────────────────────────────────────────────
  │  Goal: "find animals that attack people"
  │  → tokenize → seed nodes: [animal:1247, attack:423, people:62]
  │  → batch_adjacent(seed_nodes) → multi-hop BFS
  │  → score: distance_matrix[neighbor][goal_word] → SimilarityTable → f32
  │  → rank by relevance × NARS confidence
  │  → top-k triples as knowledge context
  │
  ▼
LAYER 4: ThinkingGraph ──────────────────────────────────────────────────
  │  Goal PoS analysis: predicate "attack" → action domain
  │  → select ThinkingStyle: Pragmatic (action predicates)
  │  → activate(Pragmatic) → CognitiveCluster
  │  → cluster shape: coherence, tension, meta_coverage
  │  → should_fork()? → parallel exploration of alternative styles
  │
  ▼
LAYER 5: Cognitive Process ──────────────────────────────────────────────
  │  select_process(complexity, entropy) → deep_analysis()
  │  
  │  EXPLORE:  batch_adjacent on active styles → activate neighbors
  │  FANOUT:   k-hop expansion → broad perspective
  │  HYPOTHESIS: strongest activation → Delta belief
  │            "animals that attack = predatory behavior"
  │  DEEPEN:   NSM decompose "attack" → [do, bad, touch, move]
  │            deeper understanding of the concept
  │  [if tense]:
  │    COUNTERFACTUAL: negate predicate → "animals that protect"
  │    SYNTHESIS: merge hypothesis + counterfactual → Theta belief
  │  REVIEW:   Meta cluster validates → reduce noise
  │  APPLICATION: project to Lambda → actionable result
  │
  ▼
LAYER 7: Result ─────────────────────────────────────────────────────────
  │  Retrieved triples + cognitive trace + confidence scores
  │  Thinking style used, verbs executed, entropy change
  │  All deterministic, all traceable, all in ~18μs
```

## 5. KNOWLEDGE ACCUMULATION

The system gets smarter over time through three mechanisms:

### 5a. Semantic Accumulation (NARS revision)
```
Observation 1: "dogs bite people"     → edge(dog→people, pred=bite, f=1.0, c=0.3)
Observation 2: "big dogs bite people" → NARS revision: c increases to 0.46
Observation 3: "dogs bite children"   → new edge(dog→children, pred=bite, f=1.0, c=0.3)
                                        existing edge confidence grows
Observation 10: same triple           → c = 0.83 (high confidence, well-established fact)

The graph doesn't just store facts — it WEIGHS them by evidence.
```

### 5b. Thinking Accumulation (topology learning)
```
Query 1: analytical thinking about "think" → Metacognitive co-activated → both succeed
  → observe_coactivation(Analytical, Metacognitive, +0.7, Activates)
  → edge truth revised: confidence increases

Query 50: pattern emerges: Analytical + Metacognitive always co-succeed on mental predicates
  → edge truth: f=0.92, c=0.81
  → future queries about mental predicates automatically co-activate both

Query 100: Creative contradicted Analytical on an abstract query, and Creative was BETTER
  → observe_coactivation(Analytical, Creative, +0.3, Contradicts)
  → Contradicts edge frequency shifts toward Activates
  → topology self-reorganizes: Creative is now an ally, not an opponent

The thinking topology LEARNS which cognitive strategies work together.
```

### 5c. Contextual Accumulation (episodic grounding)
```
Document 1 (financial article): "bank" → context weighted toward financial
  → ContextWindow carries financial genre signature
  → all "bank" lookups in this document resolve to financial sense
  → episodic nodes linked to financial semantic triples

Document 2 (nature article): "bank" → context shifts to geographic
  → ContextWindow rotates out financial, rotates in nature
  → all "bank" lookups resolve to geographic sense
  → episodic nodes linked to geographic semantic triples

The context doesn't accumulate indefinitely — it slides.
But the SEMANTIC GRAPH accumulates both senses as separate edges.
Context selects which edges are relevant NOW.
```

## 6. CROSS-REPO EXECUTION MAP

```
REPO                  COMPONENT              SESSION PROMPT                      STATUS
────                  ─────────              ──────────────                      ──────
AdaWorldAPI/ndarray
                      cam_pq.rs              session_bgz17_similarity.md         codec done
                      simd_compat            session_simd_surgery.md             in progress

AdaWorldAPI/lance-graph
  crates/bgz17/       SimilarityTable        session_bgz17_similarity.md         TO BUILD
  crates/lance-graph/
    cam_pq/           UDF + storage + IVF    CAM_PQ_SPEC.md                      done
  crates/lance-graph-planner/
    thinking/
      topology.rs     12×12 NARS adjacency   session_thinking_topology.md        BUILT TODAY
      process.rs      13 cognitive verbs     session_thinking_topology.md        BUILT TODAY
      graph.rs        ThinkingGraph (36-node) session_thinking_topology.md       BUILT TODAY
    arigraph/         AriGraph transcode     session_arigraph_transcode.md       TO BUILD

AdaWorldAPI/DeepNSM
  word_frequency/     COCA data + CAM-PQ     —                                   PUSHED TODAY
  nsm_tokenizer.rs    12-bit tokenizer       session_deepnsm_cam.md              TO BUILD
  nsm_parser.rs       PoS FSM parser         session_deepnsm_cam.md              TO BUILD
  nsm_encoder.rs      SPO + VSA encoder      session_deepnsm_cam.md              TO BUILD
  nsm_similarity.rs   SimilarityTable        session_deepnsm_cam.md              TO BUILD
  nsm_context.rs      ±5 context window      session_deepnsm_cam.md              TO BUILD
  nsm_build.rs        build pipeline         session_deepnsm_cam.md              TO BUILD

AdaWorldAPI/AriGraph
  (Python source)     reference impl         session_arigraph_transcode.md       REFERENCE ONLY
```

## 7. BUILD ORDER (critical path)

```
Phase 1 — Foundations (can run in parallel)
  ├── bgz17 SimilarityTable          (ndarray + lance-graph)
  ├── DeepNSM D7: build pipeline     (DeepNSM)
  └── thinking topology tests        (lance-graph planner branch)

Phase 2 — Core DeepNSM (sequential)
  ├── D1: nsm_tokenizer              (depends: D7 artifacts)
  ├── D4: nsm_similarity             (depends: D7 + bgz17 SimilarityTable)
  ├── D2: nsm_parser                 (depends: D1)
  └── D3: nsm_encoder                (depends: D1 + D2 + D4)

Phase 3 — Context + Integration
  ├── D8: nsm_context (±5 window)    (depends: D3)
  ├── D5: ThinkingGraph integration  (depends: D3 + thinking/graph.rs)
  └── D6: DataFusion UDFs            (depends: D3 + cam_pq/udf.rs)

Phase 4 — AriGraph Assembly
  ├── AriGraphStore                  (depends: all of Phase 2+3)
  ├── observe pipeline               (depends: D1 + D2 + D3)
  ├── retrieve pipeline              (depends: D4 + AdjacencyStore)
  └── plan pipeline                  (depends: cognitive verbs + D5)

Phase 5 — Hardening
  ├── Persistence (topology save/load, semantic graph checkpoints)
  ├── Cockpit visualization (topology force graph, cognitive traces)
  └── Benchmarks (vs AriGraph Python, vs transformer baselines)
```

## 8. THE RECURSIVE STRUCTURE

The most elegant property: the engine uses ITSELF to reason about ITSELF.

```
Data queries:     text → DeepNSM → SPO triples → semantic graph → batch_adjacent
Thinking queries: style → ThinkingGraph → batch_adjacent → cognitive verbs
Both use:         THE SAME AdjacencyStore, THE SAME truth_propagate, THE SAME semirings

The planner plans data queries using the SAME graph operations
that data queries use on the data graph.
The thinking topology learns using the SAME NARS revision
that the semantic graph uses to accumulate knowledge.

One engine. One code path. Two graphs (data + thinking).
The engine inhabits both.
```

## 9. WHAT THIS ENABLES

### Query Understanding
```sql
-- The planner doesn't just execute this Cypher — it UNDERSTANDS it
MATCH (a:Person)-[:KNOWS]->(b:Person)
WHERE a.name CONTAINS 'think'
RETURN b.name

-- DeepNSM parses the query itself:
-- SPO(person, know, person) where subject contains "think"
-- "think" → mental predicate → ThinkingStyle: Metacognitive
-- Metacognitive activates in topology → co-activates Reflective
-- EXPLORE verb finds: KNOWS edges are relational → social domain
-- MODULATE: shift to Empathetic style (social context)
-- Plan: use relationship-aware traversal, not just pattern match
```

### Analogical Reasoning
```sql
-- "Find relationships similar to doctor→patient"
MATCH (a)-[r1]->(b), (c)-[r2]->(d)
WHERE nsm_predicate_similarity(r1.type, r2.type) > 0.9
  AND nsm_subject_similarity(a.label, c.label) < 0.3
RETURN a, r1, b, c, r2, d

-- DeepNSM + ThinkingGraph:
-- INTERRELATE verb bridges across semantic domains
-- "doctor treats patient" ↔ "mechanic treats car"
-- predicate_sim(treat, treat) = 1.0
-- subject_sim(doctor, mechanic) = 0.28
-- → cross-domain analogy detected and returned
```

### Self-Improving Planning
```
Query 1-100: planner uses Analytical style for everything
  → topology learns: Analytical + Systematic co-succeed on structural queries
  → topology learns: Analytical + Creative co-succeed on exploratory queries
  → topology learns: Metacognitive validates everything (universal)

Query 101: complex query arrives
  → planner activates Analytical
  → topology says: co-activate Systematic (structural)
  → topology also says: fork Creative (exploratory, high tension)
  → both plans execute in parallel
  → Creative finds a better path
  → topology updates: Creative gets stronger activation edge

Query 200: similar complex query
  → topology automatically co-activates Creative (learned from Q101)
  → no fork needed — the topology already knows

The system LEARNED which thinking strategies work for which queries.
No gradient descent. No training loop. NARS revision on graph edges.
```

## 10. DOCUMENTATION INDEX

```
docs/
  META_INTEGRATION_PLAN.md           ← THIS DOCUMENT
  deepnsm_cam_architecture.md       ← Technical reference (15 sections)
  DEEPNSM_ARCHITECTURE.md           ← High-level vision (3 replacements)
  THINKING_PIPELINE.md              ← Thinking pipeline architecture
  THINKING_MICROCODE.md             ← YAML templates + JIT + LazyLock + NARS RL

.claude/
  DEEPNSM_CAM_REFERENCE.md          ← Quick reference for CC sessions
  knowledge/
    deepnsm_integration_map.md       ← How DeepNSM connects to bgz17 + cam_pq
    thinking_microcode.md            ← YAML + JIT + LazyLock + NARS RL pipeline
  prompts/
    session_deepnsm_cam.md           ← DeepNSM deliverables (8 deliverables, 24 tests)
    session_arigraph_transcode.md    ← AriGraph → Rust transcode (4 deliverables)
    session_thinking_topology.md     ← ThinkingGraph + cognitive verbs
    session_bgz17_similarity.md      ← SimilarityTable spec
    CAM_PQ_SPEC.md                   ← CAM-PQ codec integration
    session_unified_vector_search.md ← Vector search wiring
```

## 11. THE ONE-LINE SUMMARY

```
AriGraph's insight (LLM builds knowledge graph from experience)
  + DeepNSM's replacement (distributional lookup replaces LLM inference)
  + bgz17's calibration (exact similarity without GPU cosine)
  + cam_pq's compression (6 bytes per concept, stroke cascade search)
  + ThinkingGraph's meta-cognition (cognitive verbs on adjacency topology)
  + NARS truth propagation (evidence-weighted reasoning on all edges)
  ─────────────────────────────────────────────────────────────────────
  = A cognitive graph engine that reads, understands, remembers, reasons,
    and learns — in 9MB of cache, at 18μs per step, with zero API calls.
```
