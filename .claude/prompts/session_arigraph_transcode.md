# SESSION: AriGraph Transcode — Episodic+Semantic Memory Graph in Rust

## Mission

Transcode AriGraph (AdaWorldAPI/AriGraph, Python+OpenAI) to lance-graph (Rust+DeepNSM).
Replace every LLM call with a deterministic O(1) operation. Same graph architecture,
same memory types, zero API cost, bit-exact results.

## Source: AdaWorldAPI/AriGraph

```
AriGraph/
├── agents/
│   ├── parent_agent.py        ← GPTagent: every operation calls OpenAI
│   └── llama_agent.py         ← LLama variant (same API pattern)
├── graphs/
│   ├── parent_graph.py        ← TripletGraph: list of (subj, obj, {label}) triples
│   ├── contriever_graph.py    ← Contriever embeddings for retrieval
│   ├── hypergraph.py          ← Hyper-relational extensions
│   └── lazy_graph.py          ← Lazy evaluation variant
├── src/
│   ├── contriever.py          ← Embedding model for retrieval
│   └── utils.py               ← Distance utilities
├── prompts/                   ← LLM prompts for parsing, planning, deciding
├── pipeline_arigraph.py       ← Main pipeline: observe → parse → update → retrieve → plan → act
└── envs/                      ← TextWorld game environments
```

## READ FIRST

```bash
# AriGraph source (understand what we're replacing)
cat graphs/parent_graph.py                    # TripletGraph: add_triplets, get_associated, BFS
cat agents/parent_agent.py                    # GPTagent: item_processing_scores (LLM entity extraction)
cat graphs/contriever_graph.py                # Embedding-based retrieval
cat pipeline_arigraph.py                      # Full cognitive loop

# lance-graph target (what we're building on)
cat crates/lance-graph-planner/src/thinking/graph.rs     # ThinkingGraph: real AdjacencyStore
cat crates/lance-graph-planner/src/thinking/process.rs   # 13 cognitive verbs
cat crates/lance-graph-planner/src/adjacency/csr.rs      # CSR batch_adjacent
cat crates/lance-graph-planner/src/adjacency/propagate.rs # NARS truth propagation

# DeepNSM (the GPU/LLM replacement layer)
# See: .claude/prompts/session_deepnsm_cam.md
# See: docs/deepnsm_cam_architecture.md
```

## THE CORE TRANSCODE

### 1. TripletGraph → AdjacencyStore

AriGraph stores triples as a Python list. Every operation is O(n) scan.
lance-graph stores them as CSR/CSC with O(1) adjacency lookup.

```
ARIGRAPH (Python)                    LANCE-GRAPH (Rust)
─────────────────                    ──────────────────
self.triplets = []                   AdjacencyStore::from_edges()
  list of (subj, obj, {label})         CSR + CSC + EdgeProperties

add_triplets(triplets):              store.add_edge(src, dst, props):
  for t in triplets:                   CSR insert + NARS truth on edge
    if t not in self.triplets:         dedup via adjacency lookup O(1)
      self.triplets.append(t)          not O(n) scan

get_associated_triplets(items, k):   store.batch_adjacent(&node_ids):
  BFS over Python list                 CSR batch lookup, vectorized
  O(n × k) per query                  O(degree × k) per query
  returns string representations       returns AdjacencyBatch

delete_triplets(triplets):           store.remove_edge(src, dst):
  list.remove() — O(n)                CSR compaction — O(degree)

str(triplet):                        SpoTriple::new(s, p, o):
  "subj, label, obj" string           36-bit packed u64
```

### 2. GPTagent → DeepNSM Tokenizer+Parser

AriGraph calls OpenAI for EVERY observation parse. ~$0.001/call, ~100ms latency.
DeepNSM parses with a 6-state PoS FSM. ~0 cost, ~1μs latency.

```
ARIGRAPH (Python + OpenAI)           DEEPNSM (Rust, no LLM)
──────────────────────────           ──────────────────────

item_processing_scores(obs, plan):   nsm_tokenizer::tokenize(text):
  prompt = "extract entities..."       O(n) hash lookup, 12-bit indices
  response = openai.create(prompt)     no API call
  entities = ast.literal_eval(resp)    returns Vec<Token> with PoS
  return {entity: score}               score = word frequency (from COCA)
  COST: $0.001, 100ms                  COST: $0, 1μs

# AriGraph parse (in pipeline):     nsm_parser::parse(tokens):
  prompt = "extract triplets..."       6-state PoS FSM
  response = openai.create(prompt)     no API call
  triplets = parse_json(response)      returns SentenceStructure
  COST: $0.003, 200ms                  COST: $0, 500ns

# AriGraph relevance scoring:       distance_matrix[word][goal_word]:
  prompt = "score relevance..."        O(1) lookup, 8MB matrix
  response = openai.create(prompt)     SimilarityTable → calibrated f32
  scores = parse_scores(response)      no API call
  COST: $0.001, 100ms                  COST: $0, 15ns
```

### 3. Contriever Embeddings → CAM-PQ / Distance Matrix

AriGraph uses Contriever (a transformer) for semantic retrieval.
DeepNSM uses the precomputed distance matrix + CAM-PQ fingerprints.

```
ARIGRAPH (Contriever)                DEEPNSM (distance matrix)
─────────────────────                ────────────────────────

embed(text) → 768D vector            word_rank → 12-bit index
  transformer inference, GPU           hash lookup, O(1)
  ~50ms per text                       ~100ns per word

cosine(embed_a, embed_b) → f32      distance_matrix[a][b] → u8
  768 multiplies + sum                 1 memory access
  ~1μs                                 ~5ns

retrieve_k_nearest(query, k):        batch_adjacent() + top-k sort:
  encode query, scan all embeddings    CSR lookup + SimilarityTable
  O(n × 768)                          O(degree × k)
```

### 4. Episodic Memory → ContextWindow

AriGraph stores episodic vertices as timestamped observations linked to semantic triples.
DeepNSM's ContextWindow provides the same temporal grounding via VSA bundles.

```
ARIGRAPH                             DEEPNSM
────────                             ───────

episodic_vertex(observation_t):      ContextWindow.push(sentence_bitvec):
  store full text                      store 10K-bit VSA bundle
  link to extracted triples            XOR-in new, XOR-out oldest
  timestamp                            ring buffer position = time

episodic_edge(episode → semantic):   contextualize(word):
  "this triple was learned at t=42"    word XOR context → disambiguated
  LLM decides relevance                distributional shift IS relevance

retrieve_episodes(items, k):         ContextWindow.context():
  BFS from items through edges         current bundle captures ±5 sentences
  LLM ranks relevance                  hamming distance = relevance
  O(n) scan                            O(1) lookup
```

### 5. Planning & Decision → Cognitive Verbs

AriGraph's planning loop calls the LLM for each decision.
The cognitive verb pipeline makes the same decisions structurally.

```
ARIGRAPH (LLM planning)              LANCE-GRAPH (cognitive verbs)
───────────────────────              ────────────────────────────

plan(goal, knowledge):               CognitiveProcess::deep_analysis():
  prompt = "given knowledge, plan"      EXPLORE → FANOUT → HYPOTHESIS
  response = openai.create(prompt)      → DEEPEN → [COUNTERFACTUAL → SYNTHESIS]
  plan = parse(response)                → REVIEW → APPLICATION
  COST: $0.005, 300ms                   COST: $0, ~10μs

decide(plan, observation):           ClusterShape.should_fork():
  prompt = "given plan+obs, act"       topology tension → fork or commit
  response = openai.create(prompt)     MUL gate check → proceed/sandbox/compass
  action = parse(response)             COST: $0, ~1μs
  COST: $0.003, 200ms

# Total per step:                    # Total per step:
  ~5 LLM calls × $0.002 = $0.01       0 LLM calls
  ~500ms latency                       ~15μs latency
  non-deterministic                    deterministic
```

## GRAPH SCHEMA (Cypher)

```cypher
// ═══ SEMANTIC LAYER ═══
// Concept nodes: the 4,096-word vocabulary
CREATE (c:Concept {
    rank: 671,           // 12-bit index
    word: "dog",
    pos: "n",
    cam: [0x29, 0xF0, ...],  // 6-byte CAM-PQ (OOV path)
    freq: 319042,        // COCA frequency
    disp: 0.93           // dispersion
})

// Relation edges: SPO triples with NARS truth values
CREATE (c1:Concept)-[:RELATION {
    predicate_rank: 2943,     // "bite"
    predicate_word: "bite",
    spo_packed: 0x29FB7F05F,  // 36-bit SPO fingerprint
    truth_f: 0.8,             // NARS frequency (how often observed)
    truth_c: 0.6,             // NARS confidence (evidence weight)
    episode_count: 3,         // number of episodes containing this triple
    first_seen: datetime(),
    last_seen: datetime()
}]->(c2:Concept)

// ═══ EPISODIC LAYER ═══
// Episode nodes: timestamped observations
CREATE (e:Episode {
    id: 42,
    timestamp: datetime(),
    text: "the big dog bit the old man",
    context_vsa: <binary>,    // 10K-bit VSA bundle
    window_hash: 0xABCD       // hash of ±5 context
})

// Links: episode → semantic (which triples were extracted)
CREATE (e)-[:EXTRACTED {
    confidence: 0.9,
    position: 0               // which triple in the sentence
}]->(rel)

// Temporal chain
CREATE (e)-[:FOLLOWS]->(e_prev:Episode)

// ═══ THINKING LAYER ═══
// ThinkingStyle nodes (36 styles from contract)
CREATE (s:ThinkingStyle {
    id: 1,
    name: "Analytical",
    cluster: "Analytical",
    tau: 0x41
})

// Style adjacency with NARS truth (learned topology)
CREATE (s1:ThinkingStyle)-[:ACTIVATES {
    truth_f: 0.85,
    truth_c: 0.7,
    evidence_count: 47
}]->(s2:ThinkingStyle)

// Style → SemanticField binding (MODULATE verb)
CREATE (s)-[:HANDLES {
    truth_f: 0.9,
    truth_c: 0.6
}]->(f:SemanticField {name: "mental_predicates", primes: [53,39,68,136,56,184]})
```

## THREE LAYERS, ONE ENGINE

```
All three layers use the SAME infrastructure:

AdjacencyStore:    CSR/CSC for all three layer types
batch_adjacent():  works on Concept, Episode, and ThinkingStyle nodes
truth_propagate(): NARS semiring on Relation, EXTRACTED, and ACTIVATES edges
SimilarityTable:   calibrates distances across all layers
cognitive verbs:   EXPLORE/SYNTHESIS/etc operate on any node type

The graph IS the memory.
The memory IS the knowledge.
The knowledge IS the thinking.
No separate systems. One graph. Three views.
```

## DELIVERABLES

### D1: AriGraphStore (Rust struct wrapping three AdjacencyStores)

```rust
pub struct AriGraphStore {
    /// Semantic memory: Concept nodes + Relation edges
    pub semantic: AdjacencyStore,
    /// Episodic memory: Episode nodes + temporal/extraction edges
    pub episodic: AdjacencyStore,
    /// Thinking topology: ThinkingStyle nodes + activation edges
    pub thinking: ThinkingGraph,  // already built

    /// Cross-layer edges: episode → semantic, style → semantic field
    pub cross_links: AdjacencyStore,

    /// DeepNSM encoder (tokenizer + parser + distance matrix)
    pub nsm: NsmEncoder,
    /// Episode counter
    pub episode_count: u64,
}
```

### D2: Observation Pipeline (replaces GPTagent)

```rust
impl AriGraphStore {
    /// Process an observation: tokenize → parse → update graph → link episode.
    /// This replaces AriGraph's full LLM pipeline with O(n) deterministic ops.
    pub fn observe(&mut self, text: &str, timestamp: u64) -> ObservationResult {
        // 1. Tokenize (O(n), hash lookup)
        let tokens = self.nsm.vocab.tokenize(text);

        // 2. Parse (O(n), 6-state FSM)
        let structure = nsm_parser::parse(&tokens);

        // 3. Update semantic graph (O(triples), NARS revision on existing edges)
        for triple in &structure.triples {
            self.update_semantic(triple);
        }

        // 4. Create episode node, link to semantic
        let episode_id = self.create_episode(text, timestamp, &structure);

        // 5. Update context window
        let sentence_vec = self.nsm.encode_sentence(&structure);
        self.nsm.context.push(&sentence_vec.vsa);

        ObservationResult { episode_id, triples: structure.triples.len() }
    }

    /// Update semantic graph: add or revise triple.
    fn update_semantic(&mut self, triple: &SpoTriple) {
        // Check if edge already exists
        let existing = self.semantic.edge_between(
            triple.subject() as u64, triple.object() as u64
        );
        match existing {
            Some(edge_id) => {
                // NARS revision: merge new evidence with existing
                let old_truth = self.semantic.edge_properties.truth_value(edge_id).unwrap();
                let observation = TruthValue::new(1.0, 0.3); // each observation = moderate evidence
                let revised = TruthValue::new(old_truth.0, old_truth.1).revise(&observation);
                // Update edge truth values
                self.semantic.update_truth(edge_id, revised);
            }
            None => {
                // New edge: add with initial truth
                self.semantic.add_edge(
                    triple.subject() as u64,
                    triple.object() as u64,
                    TruthValue::new(1.0, 0.3),
                );
            }
        }
    }
}
```

### D3: Retrieval (replaces Contriever + LLM ranking)

```rust
impl AriGraphStore {
    /// Retrieve relevant knowledge for a goal/query.
    /// Replaces: AriGraph's get_associated_triplets + LLM relevance scoring.
    pub fn retrieve(&self, query: &str, k: usize, hops: usize) -> Vec<RetrievedTriple> {
        // 1. Tokenize query
        let tokens = self.nsm.vocab.tokenize(query);
        let query_words: Vec<u16> = tokens.iter().map(|t| t.rank).collect();

        // 2. Find seed nodes: query words → concept nodes
        let seed_ids: Vec<u64> = query_words.iter().map(|&r| r as u64).collect();

        // 3. Multi-hop BFS via batch_adjacent (Kuzu primitive)
        let mut frontier = seed_ids.clone();
        let mut all_triples = Vec::new();

        for _hop in 0..hops {
            let batch = self.semantic.batch_adjacent(&frontier);

            // 4. Score each adjacent triple by distance to query words
            for i in 0..batch.num_sources() {
                let source = batch.source_ids[i] as u16;
                let targets = batch.targets_for(i);
                let edge_ids = batch.edge_ids_for(i);

                for (target, edge_id) in targets.iter().zip(edge_ids.iter()) {
                    // Relevance = semantic similarity to ANY query word
                    let relevance = query_words.iter()
                        .map(|&qw| self.nsm.similarity_table.similarity(
                            self.nsm.distance_matrix.get(qw, *target as u16)
                        ))
                        .fold(0.0_f32, f32::max);

                    let truth = self.semantic.edge_properties.truth_value(*edge_id)
                        .unwrap_or((0.5, 0.1));

                    all_triples.push(RetrievedTriple {
                        subject: source,
                        object: *target as u16,
                        edge_id: *edge_id,
                        relevance,
                        confidence: truth.1,
                    });
                }

                frontier = batch.targets.clone();
            }
        }

        // 5. Sort by relevance × confidence, return top-k
        all_triples.sort_by(|a, b|
            (b.relevance * b.confidence)
                .partial_cmp(&(a.relevance * a.confidence))
                .unwrap()
        );
        all_triples.truncate(k);
        all_triples
    }
}
```

### D4: Planning (replaces LLM plan generation)

```rust
impl AriGraphStore {
    /// Plan: given goal + retrieved knowledge, produce action sequence.
    /// Replaces: AriGraph's LLM planning with cognitive process execution.
    pub fn plan(&self, goal: &str, knowledge: &[RetrievedTriple]) -> Plan {
        // 1. Parse goal into SPO
        let goal_tokens = self.nsm.vocab.tokenize(goal);
        let goal_structure = nsm_parser::parse(&goal_tokens);

        // 2. Select cognitive process based on complexity + entropy
        let complexity = estimate_complexity(&goal_structure);
        let entropy = self.thinking.topology_entropy();
        let process = programs::select_process(complexity, &entropy);

        // 3. Initialize thinking state from goal
        let start_style = self.select_thinking_style(&goal_structure);
        let mut state = self.thinking.initial_state(start_style);

        // 4. Inject retrieved knowledge as initial beliefs
        for triple in knowledge {
            state.beliefs.push(Belief {
                source_style: start_style,
                content: format!("{}->{}", triple.subject, triple.object),
                frequency: triple.relevance as f64,
                confidence: triple.confidence as f64,
                stage: SigmaStage::Omega,
            });
        }

        // 5. Execute cognitive process
        self.thinking.execute_process(&process, &mut state);

        // 6. Extract plan from Lambda-stage beliefs
        let actions: Vec<Action> = state.beliefs.iter()
            .filter(|b| b.stage == SigmaStage::Lambda)
            .map(|b| Action { description: b.content.clone(), confidence: b.confidence })
            .collect();

        Plan { actions, thinking_trace: state.trail }
    }

    /// Select initial ThinkingStyle from goal content.
    fn select_thinking_style(&self, goal: &SentenceStructure) -> u8 {
        // Mental predicates → Metacognitive (32)
        // Action predicates → Pragmatic (21)
        // Evaluative predicates → Empathetic (12)
        // Default → Analytical (1)
        if let Some(triple) = goal.triples.first() {
            let pred_pos = self.nsm.vocab.pos(triple.predicate());
            match pred_pos {
                PoS::Verb => {
                    let pred_rank = triple.predicate();
                    // Check NSM field of predicate
                    let nearest_prime = self.nsm.nearest_prime(pred_rank);
                    // Mental primes: think(53), know(39), want(68), feel(136)
                    if [53, 39, 68, 136, 56, 184].contains(&nearest_prime.0) {
                        return 32; // Metacognitive
                    }
                    // Action primes: do(15), move(227), touch(1157)
                    if [15, 227, 1157].contains(&nearest_prime.0) {
                        return 21; // Pragmatic
                    }
                    1 // Analytical (default for verbs)
                }
                _ => 1
            }
        } else {
            1 // Analytical
        }
    }
}
```

## COST COMPARISON

```
Operation            AriGraph (LLM)              lance-graph (DeepNSM)
──────────           ──────────────              ─────────────────────
Parse observation    $0.003, 200ms, stochastic   $0, 1μs, deterministic
Extract entities     $0.001, 100ms, stochastic   $0, 1μs, deterministic
Score relevance      $0.001, 100ms, stochastic   $0, 15ns, deterministic
Retrieve knowledge   $0.002, 150ms, stochastic   $0, 5μs, deterministic
Plan actions         $0.005, 300ms, stochastic   $0, 10μs, deterministic
─────────────────────────────────────────────────────────────────────
Per step total       $0.012, 850ms               $0, ~18μs
Per 1000 steps       $12, 14 minutes             $0, 18ms
Per 1M steps         $12,000, 10 days            $0, 18 seconds
```

## TESTS

1. Observation creates semantic triple + episodic node
2. Repeated observation → NARS revision increases confidence
3. Retrieval returns triples relevant to query (sorted by similarity)
4. Multi-hop retrieval (k=2) reaches 2-hop neighbors
5. Plan selects appropriate cognitive process for complexity
6. ThinkingStyle selection matches goal content PoS
7. Episode temporal ordering preserved
8. Context window disambiguates repeated observations
9. Cross-layer links (episode→semantic) traversable via batch_adjacent
10. Full pipeline: text → observe → retrieve → plan → deterministic

## OUTPUT

Branch: `feat/arigraph-transcode`
New files: `crates/lance-graph-planner/src/arigraph/mod.rs`,
           `crates/lance-graph-planner/src/arigraph/store.rs`,
           `crates/lance-graph-planner/src/arigraph/observe.rs`,
           `crates/lance-graph-planner/src/arigraph/retrieve.rs`,
           `crates/lance-graph-planner/src/arigraph/plan.rs`
Dependencies: DeepNSM (tokenizer, parser, encoder, similarity, context)
              ThinkingGraph (cognitive verbs, topology)
              AdjacencyStore (CSR/CSC, batch_adjacent, truth_propagate)
Reference: AdaWorldAPI/AriGraph (Python source)
