# lance-graph-wikidata: Streaming Wikidata → AriGraph Hydration

> Extract, index, and reason over Wikidata's 110M entities in 255 MB.
> Streaming SPARQL — no dump, no batch, live knowledge.

## Architecture

```
Wikidata SPARQL endpoint (live, free)
  │
  ▼ streaming query (1000 triples/request)
lance-graph-wikidata::sparql::stream()
  │
  ▼ parse JSON → (Q-number, P-label, Q-number)
lance-graph-wikidata::extract::to_spo()
  │
  ▼ entity label → DeepNSM tokenize → COCA rank
lance-graph-wikidata::hydrate::to_codebook()
  │  COCA 4096: direct rank (O(1))
  │  Scientific 20K: route to COCA centroids
  │  OOV: hash to nearest COCA centroid
  │
  ▼ codebook centroid IDs
lance-graph-wikidata::index::add_triple()
  │  entity_index[Q-number] = centroid (u16)
  │  AriGraph.add_triplets() → SPO store
  │  NARS.revise_with_evidence() → truth update
  │
  ▼ semantic grounding (128 KB table, 5,676 q/s)
  │  semantic_table[centroid_S][centroid_O] > 0.6?
  │  → YES: grounded triple (high confidence)
  │  → NO: weak triple (low confidence, evict candidate)
  │
  ▼ AriGraph knowledge graph (445 MB working set)
     .get_associated(entity, steps=3) → multi-hop retrieval
     .infer_deductions() → derive new triples
     .detect_contradictions() → find conflicts
     LRU eviction when > 700 MB
```

## Crate Structure

```
crates/lance-graph-wikidata/
  Cargo.toml
  src/
    lib.rs          — pub mod sparql, extract, hydrate, index, budget
    sparql.rs       — SPARQL endpoint streaming client
    extract.rs      — JSON → SPO triple conversion
    hydrate.rs      — Entity label → COCA centroid mapping
    index.rs        — Entity index + AriGraph integration
    budget.rs       — Memory budget manager (700 MB cap)
```

## Dependencies

```toml
[package]
name = "lance-graph-wikidata"
version = "0.1.0"
edition = "2021"

[dependencies]
# Core graph
lance-graph = { path = "../lance-graph" }
lance-graph-planner = { path = "../lance-graph-planner" }

# Thinking engine (codebook + semantic table)
thinking-engine = { path = "../thinking-engine" }

# DeepNSM (COCA vocabulary + tokenizer)
deepnsm = { path = "../deepnsm" }

# HTTP client for SPARQL
ureq = { version = "3", features = ["json"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

## API

### Streaming

```rust
use lance_graph_wikidata::{WikidataStream, HydrationConfig};

let config = HydrationConfig {
    memory_budget_mb: 700,
    codebook_path: "data/context-spine-v1.0/",
    sparql_batch_size: 1000,
    eviction_threshold: 0.3,  // NARS confidence below this = evict
};

let mut stream = WikidataStream::new(config)?;

// Stream by topic
stream.hydrate_topic("gene editing", 10_000)?;  // 10K triples about gene editing
stream.hydrate_topic("quantum computing", 10_000)?;

// Stream by entity
stream.hydrate_entity("Q7187", 1000)?;  // Q7187 = Gene
stream.hydrate_entity("Q944", 1000)?;   // Q944 = Quantum mechanics

// Stream everything (background, rate-limited)
stream.hydrate_all(|progress| {
    println!("{} entities, {} triples, {} MB",
        progress.entities, progress.triples, progress.memory_mb);
})?;

// Query the hydrated graph
let results = stream.graph().get_associated(&["CRISPR"], 3);
for triplet in results {
    println!("{}", triplet.to_string_repr());
}
```

### SPARQL Queries

```rust
// Topic-based: find all triples about a subject
pub fn query_topic(topic: &str, limit: usize) -> String {
    format!(r#"
        SELECT ?item ?itemLabel ?prop ?propLabel ?value ?valueLabel WHERE {{
            ?item rdfs:label "{topic}"@en .
            ?item ?prop ?value .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }} LIMIT {limit}
    "#)
}

// Entity neighborhood: all triples within N hops
pub fn query_entity(qid: &str, hops: usize) -> String {
    format!(r#"
        SELECT ?s ?sLabel ?p ?pLabel ?o ?oLabel WHERE {{
            wd:{qid} ?p1 ?mid .
            ?mid ?p2 ?o .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }} LIMIT 1000
    "#)
}

// Property-based: all instances of a relation
pub fn query_property(pid: &str, limit: usize) -> String {
    format!(r#"
        SELECT ?s ?sLabel ?o ?oLabel WHERE {{
            ?s wdt:{pid} ?o .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }} LIMIT {limit}
    "#)
}
```

### Hydration Pipeline

```rust
pub struct HydrationResult {
    pub entities_added: usize,
    pub triples_added: usize,
    pub triples_grounded: usize,    // semantic score > 0.6
    pub triples_rejected: usize,    // semantic score < 0.2
    pub triples_uncertain: usize,   // 0.2 < score < 0.6
    pub memory_used_mb: usize,
    pub evictions: usize,
}

pub fn hydrate_triple(
    entity_index: &mut EntityIndex,
    graph: &mut TripletGraph,
    nars: &mut NarsEngine,
    semantic_table: &[f32],
    codebook_idx: &[u16],
    coca_vocab: &Vocabulary,
    triple: &RawTriple,
) -> Result<TripleStatus> {
    // 1. Resolve entity labels to COCA ranks
    let s_rank = coca_vocab.lookup(&triple.subject_label)?;
    let o_rank = coca_vocab.lookup(&triple.object_label)?;
    
    // 2. Map COCA ranks to codebook centroids
    let s_cent = codebook_idx[s_rank as usize];
    let o_cent = codebook_idx[o_rank as usize];
    
    // 3. Semantic grounding
    let sem_score = semantic_table[s_cent as usize * 256 + o_cent as usize];
    
    // 4. Add to AriGraph with NARS truth
    let truth = TruthValue::new(sem_score, 0.5);  // initial confidence
    let triplet = Triplet::with_truth(
        &triple.subject_label, &triple.object_label,
        &triple.property_label, truth, triple.timestamp,
    );
    
    graph.add_triplets(&[triplet]);
    nars.revise(&triple.subject_label, &triple.property_label, sem_score);
    
    // 5. Index entity → centroid mapping
    entity_index.insert(triple.subject_qid, s_cent);
    entity_index.insert(triple.object_qid, o_cent);
    
    Ok(if sem_score > 0.6 { TripleStatus::Grounded }
       else if sem_score < 0.2 { TripleStatus::Rejected }
       else { TripleStatus::Uncertain })
}
```

### Memory Budget Manager

```rust
pub struct BudgetManager {
    max_bytes: usize,          // 700 MB
    entity_index_bytes: usize, // grows with entities
    graph_bytes: usize,        // grows with triples
    cache_bytes: usize,        // scientific routing cache
}

impl BudgetManager {
    pub fn can_add_triple(&self) -> bool {
        self.total() + 20 < self.max_bytes  // 20 bytes per NARS triple
    }
    
    pub fn evict_lowest_confidence(&mut self, graph: &mut TripletGraph, threshold: f32) {
        // Remove triples with NARS confidence < threshold
        let before = graph.len();
        graph.triplets.retain(|t| t.truth.confidence >= threshold);
        let evicted = before - graph.len();
        self.graph_bytes -= evicted * 20;
    }
    
    pub fn total(&self) -> usize {
        self.entity_index_bytes + self.graph_bytes + self.cache_bytes
            + 35_000_000  // fixed: tables + codebooks
    }
}
```

## Seed Topics (bootstrapping)

```rust
const SEED_TOPICS: &[(&str, usize)] = &[
    // Science (high-value for SPO grounding)
    ("gene editing", 5000),
    ("quantum computing", 5000),
    ("machine learning", 5000),
    ("climate change", 5000),
    
    // Technology
    ("programming language", 5000),
    ("computer science", 5000),
    ("artificial intelligence", 5000),
    
    // General knowledge
    ("country", 10000),
    ("city", 10000),
    ("person", 10000),
    ("organization", 5000),
    
    // Relations (high-connectivity)
    ("P31", 50000),   // instance-of
    ("P279", 20000),  // subclass-of
    ("P17", 20000),   // country
    ("P131", 20000),  // located-in
    ("P106", 10000),  // occupation
];
// Total: ~180K seed triples = ~3.6 MB
// Bootstraps the graph with high-connectivity entities
```

## Railway Deployment

```dockerfile
# Add to existing Dockerfile.railway:
ADD https://github.com/AdaWorldAPI/lance-graph/releases/download/v1.0.0-context-spine/context-spine-v1.0.tar.gz /tmp/
RUN tar xzf /tmp/context-spine-v1.0.tar.gz -C /app/data/ && rm /tmp/*.tar.gz

# Wikidata hydration runs at startup (background, rate-limited)
ENV WIKIDATA_BUDGET_MB=700
ENV WIKIDATA_SEED_TOPICS="gene editing,quantum computing,machine learning"
ENV WIKIDATA_SPARQL_DELAY_MS=100
```

## Size Estimates

```
Bootstrap (seed topics):     ~4 MB (180K triples)
After 1 hour crawling:      ~50 MB (2.5M triples)
After 24 hours:             ~400 MB (20M triples)
Steady state (700 MB cap):  ~22M triples, evicting lowest confidence

Entity coverage at steady state:
  ~5M unique entities indexed (u16 centroids)
  ~22M relationship triples (NARS truth)
  ~130M tokens worth of world knowledge
```
