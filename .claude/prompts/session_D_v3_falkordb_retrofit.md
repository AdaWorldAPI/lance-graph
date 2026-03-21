# SESSION D v3: FalkorDB Retrofit — Reality Check

## CONTEXT

**Repo:** `AdaWorldAPI/lance-graph` branch from `main`
**Depends on:** Sessions A+B+C complete
**External:** `FalkorDB/falkordb-rs-next-gen` (read-only reference)

FalkorDB uses SuiteSparse:GraphBLAS (C FFI) with scalar semirings over CSC matrices.
We have pure-Rust GraphBLAS (blasgraph) over 16Kbit HDR vectors + bgz17 palette
compression. This session verifies they produce the same results on the same queries.

**Goal:** NOT a production fork. A reality check: same Cypher, same results, measured.

## READ FIRST

```bash
# Our stack
cat crates/lance-graph/src/graph/spo/store.rs      # SpoStore: the triple store
cat crates/lance-graph/src/graph/spo/truth.rs       # TruthValue, TruthGate
cat crates/lance-graph/src/graph/spo/semiring.rs    # HammingMin
cat crates/lance-graph/src/graph/metadata.rs        # MetadataStore (Cypher→DataFusion)
cat crates/bgz17/src/router.rs                      # Query routing
cat crates/bgz17/KNOWLEDGE.md                       # Architecture reference
cat /mnt/user-data/outputs/bgz17_container_mapping.md  # Container word map

# FalkorDB (clone and read)
git clone https://github.com/FalkorDB/falkordb-rs-next-gen.git /tmp/falkordb
find /tmp/falkordb/src -name "*.rs" | sort | head -30
cat /tmp/falkordb/Cargo.toml
```

Key questions while reading FalkorDB:
1. Where does GrB_mxm get called? → our grb_mxm
2. Where are per-reltype CSC matrices built? → our TypedGraph
3. Where does query planner emit matrix ops? → our blasgraph_planner
4. Where are results materialized to rows? → our MetadataStore.to_datasets()
5. Where are property values joined? → our DataFusion cold path

## DELIVERABLE 1: Architecture Map (falkordb_architecture_map.md)

Before code, produce a mapping document:

```
FalkorDB Component          Our Equivalent                  Gap?
─────────────────          ──────────────                  ────
GrB_Matrix (CSC, scalar)   GrBMatrix (CSR+CSC, BitVec)    Session A ✓
GrB_mxm (SuiteSparse C)    grb_mxm (pure Rust)            Same algebra
960 scalar semirings        7 HDR + PaletteSemiring        Session A+B ✓
Query parser (C)            parser.rs (Rust, nom)          Already have
Query planner               blasgraph_planner.rs           Session A ✓
Per-reltype matrices        TypedGraph + TypedPaletteGraph Sessions A+B ✓
Label masks                 TypedGraph.labels               Session A ✓
Property storage            MetadataStore → RecordBatch     Already have
TruthGate                   NOT IN FALKORDB                 We have more ✓
Palette compression         NOT IN FALKORDB                 We have more ✓
HHTL progressive search     NOT IN FALKORDB                 We have more ✓
Container 256-word struct   NOT IN FALKORDB                 We have more ✓
Redis server protocol       Not needed (library mode)       N/A
```

## DELIVERABLE 2: FalkorCompat Shim (new: falkor_compat.rs)

Executes FalkorDB-style queries through our stack:

```rust
pub struct FalkorCompat {
    /// MetadataStore for cold-path DataFusion queries (Entry 1).
    pub metadata: MetadataStore,
    /// TypedGraph for blasgraph semiring traversal (Entry 2, BitVec).
    pub typed_graph: TypedGraph,
    /// TypedPaletteGraph for palette-accelerated traversal (Entry 2, bgz17).
    pub palette_graph: Option<TypedPaletteGraph>,
    /// SpoStore for truth-gated fingerprint queries.
    pub spo_store: SpoStore,
    /// Containers for reading W4-7 (truth), W16-31 (edges), W112-125 (bgz17).
    pub containers: Vec<[u64; 256]>,
}

impl FalkorCompat {
    /// Execute Cypher through DataFusion (Entry 1 cold path).
    /// Same path as upstream lance-graph. No bgz17 involved.
    pub async fn query_datafusion(&self, cypher: &str) -> Result<RecordBatch> {
        self.metadata.query(cypher).await
    }

    /// Execute Cypher through blasgraph semirings (Entry 2, BitVec hot path).
    /// Uses TypedGraph + grb_mxm + HammingMin semiring.
    pub fn query_blasgraph(&self, cypher: &str, gate: TruthGate) -> Result<Vec<SpoHit>> {
        let plan = parse_and_plan(cypher)?;
        let matrix = compile_to_blasgraph(&plan, &self.typed_graph, &HdrSemiring::HammingMin)?;
        let candidates = materialize_positions(&matrix);
        filter_with_truth_gate(candidates, gate, &self.spo_store)
    }

    /// Execute Cypher through bgz17 palette (Entry 2, palette hot path).
    /// Uses TypedPaletteGraph + compose_table + palette distance.
    pub fn query_palette(&self, cypher: &str, gate: TruthGate) -> Result<Vec<SearchResult>> {
        let plan = parse_and_plan(cypher)?;
        let pal = self.palette_graph.as_ref().ok_or(Error::NoPalette)?;
        let candidates = compile_to_palette(&plan, pal)?;
        filter_with_truth_gate_and_distance(candidates, gate, &self.containers)
    }

    /// Route automatically: structural → blasgraph, similarity → bgz17.
    pub fn query_routed(&self, cypher: &str, gate: TruthGate) -> Result<QueryResult> {
        let plan = parse_and_plan(cypher)?;
        match classify_query(&plan) {
            QueryClass::PureTraversal => self.query_blasgraph(cypher, gate),
            QueryClass::Similarity => self.query_palette(cypher, gate),
            QueryClass::Hybrid => {
                let structural = self.query_blasgraph(cypher, TruthGate::OPEN)?;
                let positions: Vec<usize> = structural.iter().map(|h| h.position).collect();
                refine_with_palette(positions, self.palette_graph.as_ref().unwrap(), gate)
            }
        }
    }
}
```

## DELIVERABLE 3: Reality Check Test Suite

THE SAME social graph from both SpoStore (spo_redisgraph_parity tests)
AND MetadataStore (metadata.rs tests): Jan→Ada→Max via KNOWS/CREATES/HELPS.

```rust
/// Build the test graph through ALL THREE backends.
fn build_social_graph() -> FalkorCompat {
    let mut compat = FalkorCompat::new();

    // Add to MetadataStore (Entry 1: DataFusion)
    compat.metadata.add_node(NodeRecord::new(1, "Person").with_prop("name", "Jan"));
    compat.metadata.add_node(NodeRecord::new(2, "Person").with_prop("name", "Ada"));
    compat.metadata.add_node(NodeRecord::new(3, "Person").with_prop("name", "Max"));
    compat.metadata.add_edge(EdgeRecord::new(1, 2, "KNOWS"));
    compat.metadata.add_edge(EdgeRecord::new(2, 3, "KNOWS"));

    // Add to SpoStore (Entry 2: fingerprint queries)
    let jan = label_fp("Jan"); let ada = label_fp("Ada"); let max = label_fp("Max");
    let knows = label_fp("KNOWS");
    compat.spo_store.insert(
        dn_hash("jan-knows-ada"),
        &SpoBuilder::build_edge(&jan, &knows, &ada, TruthValue::new(0.9, 0.8)),
    );
    // ... etc

    // Build containers with bgz17 annex
    // Encode planes → Base17 → palette → write W112-125

    compat
}

#[tokio::test]
async fn test_1_datafusion_matches_blasgraph() {
    let compat = build_social_graph();
    let df = compat.query_datafusion("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name").await?;
    let bg = compat.query_blasgraph("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name", TruthGate::OPEN)?;
    // Both should return Jan→Ada, Ada→Max
    assert_eq!(df.num_rows(), bg.len());
}

#[test]
fn test_2_palette_matches_blasgraph_ranking() {
    let compat = build_social_graph();
    let bg = compat.query_blasgraph("...", TruthGate::OPEN)?;
    let pal = compat.query_palette("...", TruthGate::OPEN)?;
    // Top-k ordering should agree (ρ > 0.95)
}

#[test]
fn test_3_truth_gate_filters_correctly() {
    let compat = build_social_graph();
    // Jan-KNOWS-Ada has truth (0.9, 0.8), expectation 0.82 → passes STRONG
    // Max-HELPS-Jan has truth (0.6, 0.3), expectation 0.53 → fails STRONG
    let strong = compat.query_routed("MATCH ...", TruthGate::STRONG)?;
    // Only strong edges survive
}

#[test]
fn test_4_two_hop_chain() {
    let compat = build_social_graph();
    // Jan→Ada→Max via KNOWS (2-hop)
    // blasgraph: KNOWS × KNOWS via grb_mxm
    // palette: compose_table[jan_pal][ada_pal] → result_pal
    // Both should find Max
}

#[test]
fn test_5_performance_comparison() {
    // Build larger graph (1000 nodes, 10000 edges)
    // Run same query 100× through each backend
    // Print:
    //   DataFusion (cold): X ms
    //   blasgraph (BitVec): X ms
    //   palette (bgz17): X ms
    //   Expected: palette 100-10,000× faster than BitVec
}
```

## DELIVERABLE 4: Palette Semiring Variants (falkor_semirings.rs)

Map FalkorDB's most-used semirings to palette compose tables:

```rust
pub enum FalkorSemiring {
    MinPlus,     // shortest path: compose=distance_add, reduce=min
    OrAnd,       // reachability: compose=AND, reduce=OR
    XorBundle,   // HDR composition: compose=xor_bind, reduce=bundle
}

impl FalkorSemiring {
    pub fn build_compose_table(&self, pal: &Palette) -> Vec<u8>;
}
```

## DELIVERABLE 5: Benchmark Document (falkordb_benchmark.md)

```
Query Type          DataFusion    blasgraph    bgz17       Speedup
                    (cold SQL)    (BitVec)     (palette)   (pal/bg)
─────────────       ──────────    ─────────    ─────────   ────────
1-hop traversal     ? ms          ? ms         ? ms        ?×
2-hop traversal     ? ms          ? ms         ? ms        ?×
3-hop chain         ? ms          ? ms         ? ms        ?×
KNN similarity      N/A           ? ms         ? ms        ?×
TruthGate filter    ? ms          ? ms         ? ms        ?×

Storage per node:
  DataFusion:  properties only (varies)
  blasgraph:   2KB BitVec per plane × 3 = 6KB
  bgz17:       102B Base17 + 3B palette = 105B (in container W112-125)
  FalkorDB:    scalar per matrix entry (~16 bytes per relationship)
```

## CRITICAL NOTES

1. **Don't build FalkorDB from source.** Use Docker: `docker run -p 6379:6379 falkordb/falkordb:latest`
   Connect via `falkordb-rs` client for comparison queries if desired.

2. **The compose table (64KB) is the novel contribution.** If it works for
   2-hop traversal with ρ > 0.9, that's publishable. Test this carefully.

3. **TruthGate is our advantage over FalkorDB.** FalkorDB has no concept of
   edge confidence filtering. Our queries can say "only trust edges with
   confidence > 0.8" — FalkorDB can't.

4. **The container annex (W112-125) means bgz17 is FREE to use.** The
   Cascade already samples W112 at stride-16. No extra columns. No extra
   storage. Just filled reserved words in the existing 2KB container.

5. **Focus on correctness first.** If 2-hop results differ between blasgraph
   and palette backends, fix that before measuring speed.

## OUTPUT

Branch: `feat/falkordb-retrofit`
Files: `falkor_compat.rs`, `falkor_semirings.rs`, `falkordb_benchmark.md`
Run: `cargo test --features bgz17-codec -- --nocapture`
