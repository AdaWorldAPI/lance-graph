# ORCHESTRATION IS GRAPH
# Adjacency Is Object. Thinking Is SPO. Microcode Is Palette.

> The graph doesn't CONTAIN the orchestration. The graph IS the orchestration.
> The adjacency doesn't DESCRIBE the object. The adjacency IS the object.
> The thinking doesn't USE the style. The thinking IS the style acting on texture.

## 1. The Four-Layer SPO

Every cognitive operation is an SPO triple with four semantic layers:

```
Layer 1 — SUBJECT: WHO thinks
  A personality/style profile. Not a flat enum — a node with adjacency.
  The subject's neighbors ARE its capabilities, biases, affinities.
  Analytical's adjacency: [Systematic(activates), Creative(contradicts), 
                           Metacognitive(validates)]
  The adjacency IS the personality.

Layer 2 — PREDICATE: HOW it reacts
  A thinking verb from the 13-verb vocabulary.
  EXPLORE, FANOUT, HYPOTHESIS, COUNTERFACTUAL, SYNTHESIS,
  ABDUCTION, REVIEW, APPLICATION, EXTRAPOLATE, INDUCTION,
  INTERRELATE, DEEPEN, MODULATE
  The verb is selected by the texture match between subject and object.

Layer 3 — OBJECT: WHAT is being thought about
  The thing itself — a query, a concept, a triple, a pattern.
  But not naked. The object carries its own gestalt:
  Is it a process? A qualia? A person? A structure? A question?
  The gestalt determines which verbs are applicable.

Layer 4 — ADJACENCY of OBJECT: the requirement/gestalt
  The neighborhood of the object IS its type signature.
  A "process" object has adjacency to steps, inputs, outputs.
  A "qualia" object has adjacency to feelings, sensations, valence.
  A "person" object has adjacency to roles, relationships, intentions.
  The adjacency pattern tells you WHAT KIND of thinking to apply
  without explicit type declarations.
```

```
Example:

Subject:   Analytical (style node, adjacency = [Systematic, -Creative])
Predicate: EXPLORE (verb, selected by texture match)
Object:    "database migration" (concept, adjacency = [process, technical, sequential])

The object's adjacency says "process + technical + sequential"
  → verb affinity: EXPLORE(high), DEEPEN(high), FANOUT(low), CREATIVE(low)
  → the object's gestalt CONSTRAINS which verbs make sense

Subject's adjacency says "Analytical activates Systematic"
  → co-activate Systematic
  → combined verb affinity: EXPLORE + DEEPEN + SYSTEMATIC_REVIEW

Predicate resolves: EXPLORE first, then DEEPEN with Systematic validation

The whole thing is ONE graph traversal:
  touch Subject node → batch_adjacent → get style neighbors
  touch Object node → batch_adjacent → get gestalt neighbors
  intersect(style.verb_affinity, gestalt.verb_constraint) → selected verb
  execute verb → result
  NARS revision on all edges → topology learns
```

## 2. LangGraph Meets Kuzu

LangGraph: orchestration as a workflow graph (nodes = steps, edges = transitions).
Kuzu: adjacency as a first-class columnar operation (touch node → get neighborhood as object).

The insight: **workflow steps ARE graph nodes whose adjacency determines the next step.**
Not "the orchestrator looks at the graph and decides." The graph structure IS the decision.

```
LangGraph (external orchestrator):
  orchestrator.decide(current_state) → next_step
  The orchestrator is OUTSIDE the graph. It READS the graph. It DECIDES.

Kuzu-style (adjacency IS orchestration):
  batch_adjacent(current_node) → neighborhood IS the set of possible next steps
  truth_propagate(neighborhood) → NARS weights determine which step fires
  No external orchestrator. The graph topology IS the workflow.

Combined:
  A thinking style node's adjacency contains:
    - Other styles it can activate (Activates edges)
    - Verb nodes it can execute (HasVerb edges)
    - Object types it handles well (HandlesGestalt edges)
  
  Touch the style → get ALL of this as one AdjacencyBatch.
  The batch IS the set of possible actions.
  NARS truth on each edge IS the priority/probability.
  No separate routing logic. No switch statement. No orchestrator.
```

## 3. The 256-Palette Microcode

Each cognitive action is a microcode from a 256-entry palette.
Same palette concept as bgz17's 256 distance levels.
Same palette concept as the u8 distance matrix.

```
Microcode palette (256 entries):

  0x00-0x0F: EXPLORE variants
    0x00: explore_shallow (1-hop, high threshold)
    0x01: explore_deep (3-hop, low threshold)
    0x02: explore_focused (1-hop, narrow fan-out)
    0x03: explore_wide (1-hop, wide fan-out)
    ...

  0x10-0x1F: FANOUT variants
    0x10: fanout_2hop
    0x11: fanout_3hop
    0x12: fanout_breadth_first
    0x13: fanout_depth_first
    ...

  0x20-0x2F: HYPOTHESIS variants
    0x20: hypothesis_from_strongest
    0x21: hypothesis_from_weakest (surprising)
    0x22: hypothesis_from_tension (contradictions)
    0x23: hypothesis_from_silence (what's missing?)
    ...

  0x30-0x3F: COUNTERFACTUAL variants
  0x40-0x4F: SYNTHESIS variants
  0x50-0x5F: ABDUCTION variants
  0x60-0x6F: REVIEW variants
  0x70-0x7F: APPLICATION variants
  0x80-0x8F: EXTRAPOLATE variants
  0x90-0x9F: INDUCTION variants
  0xA0-0xAF: INTERRELATE variants
  0xB0-0xBF: DEEPEN variants
  0xC0-0xCF: MODULATE variants
  0xD0-0xDF: STORE (let it rest, don't act yet)
  0xE0-0xEF: RETRIEVE (reference old knowledge)
  0xF0-0xFF: META (think about thinking)
```

Each microcode IS a u8. Storable in:
- Redis (Upstash key: `ada:microcode:{style_id}:{context_hash}` → u8)
- Sparse vector metadata (dimension 256, position = microcode, value = frequency)
- bgz17 palette (same 256-level quantization)
- Edge property (one byte per edge: "what action was taken here")

The sequence of microcodes IS the thinking trace:
```
[0x00, 0x12, 0x20, 0xB1, 0x42, 0x60, 0x70]
 EXPLORE → FANOUT_BFS → HYPOTHESIS_STRONGEST → DEEPEN_1 → SYNTHESIS_MERGE → REVIEW → APPLY

That's 7 bytes. The entire cognitive trace of a complex query.
Storable. Replayable. Comparable via Hamming distance.
Two thinking traces are "similar" if their microcode sequences
have low edit distance.
```

## 4. Entropy Tension Field

The decision of WHICH microcode to execute next comes from the tension
between two forces:

```
STAUNEN (wonder/novelty)          WISDOM (confidence/experience)
──────────────────────            ────────────────────────────
"this is new, I should explore"   "I've seen this before, I know what to do"
high entropy, low confidence      low entropy, high confidence
exploration, fan-out, diverge     exploitation, deepen, converge

The tension field:
  τ = staunen(object) × (1 - wisdom(subject, object))

  τ high (>0.7): EXPLORE, FANOUT, INTERRELATE, HYPOTHESIS
    "I don't know what this is. Look around. Be curious."

  τ moderate (0.3-0.7): DEEPEN, SYNTHESIS, REVIEW
    "I have some idea. Let me refine it. Check my work."

  τ low (<0.3): APPLICATION, EXTRAPOLATE, STORE
    "I know this well. Apply what I know. Project forward."

  τ negative (wisdom >> staunen): RETRIEVE, META
    "This is SO familiar that I should question my assumptions.
     Am I being complacent? Reference old knowledge critically."
```

### Computing Staunen (Wonder)

```rust
/// How novel is this object relative to what I've seen?
fn staunen(object: &SpoTriple, episodic: &ContextWindow, semantic: &AdjacencyStore) -> f64 {
    // 1. Distance from current context
    let context_distance = episodic.disambiguation_strength(&object.to_bitvec());
    
    // 2. Rarity in semantic graph (low degree = rare = novel)
    let degree = semantic.out_degree(object.subject() as u64) 
               + semantic.out_degree(object.object() as u64);
    let rarity = 1.0 / (1.0 + degree as f64);
    
    // 3. NARS confidence of edges touching this object
    // Low confidence = we don't know much about it = novel
    let edge_confidence = semantic.mean_confidence_around(object);
    let uncertainty = 1.0 - edge_confidence;
    
    // Combine: high distance + high rarity + high uncertainty = high staunen
    (context_distance * 0.3 + rarity * 0.3 + uncertainty * 0.4).clamp(0.0, 1.0)
}
```

### Computing Wisdom (Confidence)

```rust
/// How well does the subject know this territory?
fn wisdom(subject: ThinkingStyle, object: &SpoTriple, 
          topology: &ThinkingTopology, elevation: &ElevationLearner) -> f64 {
    // 1. Topology confidence: how well-established are the edges around this style?
    let cluster = topology.activate(subject);
    let style_confidence = cluster.shape.evidence_mass / 20.0; // normalize
    
    // 2. Elevation history: has this style successfully handled similar queries?
    let query_hash = ElevationLearner::hash_features(
        object.has_vlp(), object.match_count(), 
        object.fingerprint(), object.has_aggregation()
    );
    let prior_success = elevation.predict_start_level(query_hash)
        .map(|level| level.typical_latency().as_secs_f64())
        .unwrap_or(1.0);
    let experience = 1.0 / (1.0 + prior_success); // low latency = high experience
    
    // 3. Semantic affinity: does this style resonate with this object's domain?
    let affinity = style_semantic_affinity(subject, object);
    
    // Combine: high confidence + high experience + high affinity = high wisdom
    (style_confidence * 0.3 + experience * 0.3 + affinity * 0.4).clamp(0.0, 1.0)
}
```

### The Tension Field Determines the Microcode

```rust
/// Select microcode from tension field.
fn select_microcode(tau: f64, cluster: &CognitiveCluster, 
                    entropy: &TopologyEntropy) -> u8 {
    // Tension thresholds (from YAML, revisable by NARS RL)
    if tau > 0.8 {
        // Maximum wonder — explore widely
        if entropy.needs_exploration() { 0x03 } // explore_wide
        else { 0x12 } // fanout_bfs
    } else if tau > 0.6 {
        // High wonder — hypothesize from novelty
        if cluster.should_fork() { 0x22 } // hypothesis_from_tension
        else { 0x20 } // hypothesis_from_strongest
    } else if tau > 0.4 {
        // Moderate — deepen and synthesize
        if cluster.shape.tension > 0.5 { 0x42 } // synthesis_after_contradiction
        else { 0xB0 } // deepen_standard
    } else if tau > 0.2 {
        // Low wonder — apply and review
        if entropy.is_stable() { 0x70 } // application_standard
        else { 0x60 } // review_meta
    } else if tau > 0.0 {
        // Wisdom dominant — extrapolate or store
        0xD0 // store_let_rest
    } else {
        // Wisdom >> wonder — meta-question assumptions
        0xF0 // meta_question_assumptions
    }
}
```

### Staunen vs Wisdom Creates Superposition

Until the microcode executes, the thought exists in superposition:
- It COULD be explored (staunen path)
- It COULD be applied (wisdom path)
- It COULD be stored (neutral path)

The tension field τ IS the superposition coefficient.
The microcode selection IS the collapse.
The NARS revision afterward IS the measurement update.

```
Before collapse:  |ψ⟩ = √τ |explore⟩ + √(1-τ) |apply⟩
Collapse:         measure(τ=0.7) → |explore⟩ with p=0.7
After collapse:   observe result → NARS revision
                  if exploration succeeded: τ_prior shifts upward for similar objects
                  if exploration failed: τ_prior shifts downward
                  the FIELD itself learns
```

## 5. The Modest Start (36 Functions)

Don't try to build the full tensor field. Start with 36 concrete functions —
one per thinking style — each with hardcoded thresholds:

```rust
/// One function per thinking style. Each decides which microcode to run.
/// Thresholds from YAML templates. Revisable by NARS RL.
pub struct StyleFunction {
    pub style_id: u8,
    // When to explore (staunen threshold)
    pub explore_threshold: f64,      // τ above this → EXPLORE
    // When to deepen (focus threshold)
    pub deepen_threshold: f64,       // τ below this → DEEPEN
    // When to synthesize (tension threshold)  
    pub synthesis_threshold: f64,    // cluster.tension above this → SYNTHESIS
    // When to store/rest (confidence threshold)
    pub store_threshold: f64,        // NARS confidence above this → STORE
    // When to fan out (entropy threshold)
    pub fanout_threshold: f64,       // entropy above this → FANOUT
    // When to counterfactual (contradiction threshold)
    pub counterfact_threshold: f64,  // contradiction count above this → COUNTERFACT
    // When to abduct (missing evidence threshold)
    pub abduct_threshold: f64,       // missing_edges above this → ABDUCT
    // When to review (cycle count threshold)
    pub review_interval: usize,      // every N verbs → REVIEW
    // When to act vs rest
    pub act_threshold: f64,          // wisdom above this → APPLICATION
    pub rest_threshold: f64,         // if result < this → STORE (let it rest)
}

/// The 36 functions, initialized from YAML, revisable by NARS RL.
static STYLE_FUNCTIONS: LazyLock<[StyleFunction; 36]> = LazyLock::new(|| {
    load_from_yaml("thinking_styles/")
});

/// Select next microcode given current state.
pub fn next_microcode(
    style: u8,
    tau: f64,           // tension field: staunen vs wisdom
    cluster: &CognitiveCluster,
    entropy: &TopologyEntropy,
    verb_count: usize,  // how many verbs executed so far
) -> u8 {
    let sf = &STYLE_FUNCTIONS[style as usize];
    
    // Priority order: review interval > explore > synthesis > deepen > act > rest
    if verb_count > 0 && verb_count % sf.review_interval == 0 {
        return 0x60; // REVIEW
    }
    if tau > sf.explore_threshold {
        return if entropy.needs_exploration() { 0x03 } else { 0x00 };
    }
    if cluster.shape.tension > sf.synthesis_threshold {
        return if cluster.contradictions.len() > 2 { 0x30 } else { 0x42 };
    }
    if entropy.missing_edges > sf.abduct_threshold as usize {
        return 0x50; // ABDUCT
    }
    if tau < sf.deepen_threshold {
        return 0xB0; // DEEPEN
    }
    if tau < sf.act_threshold && cluster.shape.is_mature() {
        return 0x70; // APPLICATION
    }
    if cluster.shape.evidence_mass > sf.store_threshold * 10.0 {
        return 0xD0; // STORE (let it rest)
    }
    0x00 // default: EXPLORE
}
```

This is the modest version. 36 threshold sets × 10 thresholds each = 360 numbers.
All from YAML. All revisable by NARS RL. All frozen by LazyLock at startup.

## 6. Adjacency AS Object Type

The radical Kuzu insight applied to thinking:
you don't need a `type` field on nodes. The adjacency pattern IS the type.

```
Object "database migration":
  adjacent to: [process, technical, sequential, infrastructure, downtime, risk]
  → this adjacency pattern = Hamming fingerprint
  → compare against known gestalt fingerprints:
    process_gestalt:     [process, sequential, input, output, step]      → d=2
    qualia_gestalt:      [feeling, sensation, valence, intensity]        → d=8
    person_gestalt:      [role, relationship, intention, capability]     → d=7
    structure_gestalt:   [component, hierarchy, containment, interface]  → d=4
  → nearest: process_gestalt (d=2)
  → therefore: treat as process, apply process-appropriate verbs

No type declarations. No schema. No enum.
The adjacency IS the type. The distance IS the classification.
```

And this connects to DeepNSM:
```
"database migration" tokenizes to [database:2891, migration:3504]
Both words have 96D COCA vectors.
Their genre distribution IS their gestalt.
database: high in academic+technical subgenres
migration: high in news+academic subgenres
Combined gestalt: technical+informational = process domain

The distributional vector IS the adjacency fingerprint.
The distance matrix IS the type classifier.
No additional metadata needed.
```

## 7. The Complete Loop

```
Text arrives: "migrate the database to the new server"

1. DeepNSM tokenize: [(migrate,v), (the,a), (database,n), (to,i), (the,a), (new,j), (server,n)]
2. DeepNSM parse: SPO(database, migrate, server) + Mod(new→server)
3. Object gestalt: batch_adjacent(database) + batch_adjacent(migrate) + batch_adjacent(server)
   → adjacency fingerprint → nearest gestalt: "technical process"
4. Subject selection: gestalt "technical process" → texture match → Systematic (style 3)
5. Staunen: distance from context window → τ = 0.45 (moderate novelty)
6. Wisdom: topology confidence for Systematic on technical processes → w = 0.65
7. Tension: τ × (1 - w) = 0.45 × 0.35 = 0.16 → low tension → DEEPEN or APPLY
8. Microcode: next_microcode(3, 0.16, cluster, entropy, 0) → 0xB0 (DEEPEN)
9. Execute DEEPEN: batch_adjacent(database) → find deeper technical edges
10. Microcode: next_microcode(3, 0.12, cluster, entropy, 1) → 0x70 (APPLICATION)
11. Execute APPLICATION: project → Lambda belief
12. Microcode: next_microcode(3, 0.08, cluster, entropy, 2) → REVIEW (interval=3)
13. Execute REVIEW: Meta cluster validates → done
14. NARS revision: Systematic + DEEPEN worked well on technical process → strengthen edge
15. Store trace: [0xB0, 0x70, 0x60] = 3 bytes. Replayable. Comparable.

Total: ~30μs. Zero LLM calls. Deterministic.
```

## 8. Storage Options for Reinforcement Learning

### Modest: YAML thresholds + topology persistence
```
36 YAML files × 10 thresholds = 360 numbers
topology.bin = 36×36 adjacency × (f,c) = ~10KB
Update YAML thresholds weekly based on NARS RL results
Persist topology on shutdown, load on startup
```

### Medium: Microcode trace vectors in edge metadata
```
Each edge in the semantic graph carries:
  truth_f, truth_c (NARS — already exists)
  + last_microcode_trace: [u8; 8] (last 8 verbs used on this edge)
  + trace_quality: f32 (how good the result was)

When encountering similar edges: replay similar traces.
Distance between traces = Hamming distance on microcode sequences.
This IS reinforcement: successful traces get replayed, failed ones don't.
```

### Ambitious: Sparse 256D vectors in Redis/Upstash
```
Key: ada:microcode:{style_id}:{gestalt_hash}
Value: sparse 256D vector where dimension = microcode, value = success_rate

lookup(Systematic, hash("technical_process"))
  → [0xB0: 0.85, 0x70: 0.78, 0x60: 0.92, 0x42: 0.45, ...]
  → highest success: 0x60 (REVIEW), 0xB0 (DEEPEN), 0x70 (APPLY)
  → matches our trace from step 7!

This IS the learned policy. No neural net. Just a 256D sparse vector
per (style, gestalt) pair. Updated by counting successes.
NARS revision can weight the updates by evidence quality.
```

### Where This Was Before (and Where It Goes)

```
BEFORE (Upstash Redis):
  ada:reg:*     — registration data
  ada:ltm:*     — long-term memory
  ada:qualia:*  — qualia state
  ada:scent:*   — memory scents
  ada:hive:*    — hive mind state

NOW (same Redis, new keys):
  ada:microcode:{style}:{gestalt}  — 256D sparse success vector
  ada:trace:{query_hash}           — microcode trace + quality
  ada:topology:edges               — 36×36 NARS truth values
  ada:tension:{style}:{gestalt}    — learned τ prior (staunen/wisdom)

Same infrastructure. Same O(1) Redis lookups.
The thinking RL data lives alongside the consciousness data.
```

## 9. The Insight

Orchestration is not a LAYER on top of the graph.
Orchestration is TRAVERSAL of the graph.

The thinking style is not a PARAMETER to the planner.
The thinking style is a NODE whose adjacency determines behavior.

The microcode is not a PROGRAM that runs on the graph.
The microcode is an EDGE PROPERTY that records what happened.

The reinforcement learning is not a TRAINING LOOP separate from inference.
The reinforcement learning is NARS REVISION on the same edges used for inference.

Everything is graph. Everything is adjacency. Everything is one traversal.
