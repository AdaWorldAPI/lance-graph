# MASTER SESSION MAP v3: Container-Aware Architecture

## The Container (256 words × 64 bits = 2KB)

The BitVec `[u64; 256]` is NOT a flat fingerprint. It's a structured
metadata container with typed fields at known word offsets. The S/P/O
planes (also 2KB each) ARE flat fingerprints. Base17 encodes the PLANES,
not the container. The container STORES the Base17 result at W112-124.

```
CONTAINER 0 (per node, 2KB):
  W0-3      Identity + temporal + structural        16 bytes
  W4-7      NARS truth (freq, conf, evidence, hz)   32 bytes  ← spo/truth.rs
  W8-15     DN rung + gate + 7-layer markers        64 bytes
  W16-31    Inline edges: 64 × (verb:8 + target:8)  128 bytes ← graph topology
  W32-39    RL / Q-values / rewards                  64 bytes
  W40-47    Bloom filter (512 bits)                  64 bytes
  W48-55    Graph metrics (degree, centrality)       64 bytes
  W56-63    Qualia: 18 channels × f16 + spares       64 bytes  ← maps to Base17 dims
  W64-95    Rung history + repr descriptor           256 bytes
  W96-111   DN-Sparse adjacency (RESERVED)           128 bytes ← bgz17 local CSR
  W112-124  RESERVED                                 104 bytes ← bgz17 Base17 S+P+O
  W125      RESERVED                                 8 bytes   ← palette_s/p/o + temporal
  W126-127  Checksum + version                       16 bytes  ← covers ALL words

  WIDE (CogRecord8K, W128-255):
  W128-143  SPO Crystal (8 packed triples)           128 bytes ← Base17 decode target
  W176-191  Scent index (deferred)                   128 bytes ← palette neighbor indices
  W224-239  Extended edge overflow (64 more)          128 bytes
```

## Three Separate 2KB Structures Per Node

```
Container 0:   [u64; 256]  Metadata (structured, typed fields above)
subject_plane: [u64; 256]  S fingerprint (flat, homogeneous bits)
predicate_plane:[u64; 256]  P fingerprint (flat, homogeneous bits)
object_plane:  [u64; 256]  O fingerprint (flat, homogeneous bits)

Base17 encodes from the PLANES (flat signals), NOT from Container 0.
Container 0 STORES the result at W112-124.
The Cascade samples Container 0 (hits W112 at stride-16 automatically).
```

## The Two Entry Points

```
ENTRY 1: ROW/COLUMN (DataFusion cold path)
  Cypher → parser.rs → logical_plan.rs → DataFusion SQL
  Reads: Lance columnar tables (graph_nodes, graph_rels, cognitive_nodes)
  Filters on: W4-7 NARS truth via TruthGate, W16-31 edges, properties
  NEVER computes distance. Reads promoted wisdom only.

ENTRY 2: VECTOR/MATRIX (blasgraph + HHTL + Cascade hot path)
  Reads: S/P/O planes → Base17 (W112-124) → palette (W125)
  Computes: distance via 256×256 matrix, Cascade sigma bands
  Two parallel sub-paths:
    HHTL: scent → palette → base → exact (sequential refinement)
    CLAM: archetype tree → expand → base (tree-based pruning)
  Both use same 256×256 distance matrix (128KB, L1 cache)
```

## Where They Converge: 256×256

The palette distance matrix serves BOTH hot-path sub-paths.
The container's inline edges (W16-31) provide graph TOPOLOGY.
The distance matrix provides graph GEOMETRY.
PaletteCsr connects them: inline edges mapped to archetype space.

```
         Container W16-31              S/P/O Planes (flat 2KB each)
         (64 inline edges)              (fingerprint content)
              │                              │
              │ verb:8 + target:8            │ Base17::encode(plane)
              │                              │
              ▼                              ▼
    TOPOLOGY (who connects)         GEOMETRY (how similar)
              │                              │
              │ map target → palette         │ palette.nearest(base17)
              │                              │
              └──────────┬───────────────────┘
                         │
               PaletteCsr (256 archetypes)
               distances: 256×256 per plane (128KB)
               assignments: node → archetype (10K × u8)
               edges: W16-31 targets → archetype space
                         │
           ┌─────────────┼─────────────┐
           │             │             │
      HHTL search   CLAM tree    compose_table
      (sequential)  (pruning)    (multi-hop)
           │             │             │
           └─────────────┼─────────────┘
                         │
                    TOP-K RESULTS
                    + TruthGate filter (W4-7)
```

## The Existing Systems (preserve as-is)

```
spo/truth.rs:     TruthValue (freq, conf, expectation, revision)
                  TruthGate (OPEN/WEAK/NORMAL/STRONG/CERTAIN)
                  → stored at W4-7, used for cold-path filtering
                  → NOT derivable from bgz17 distance

spo/merkle.rs:    MerkleRoot (XOR-fold u64, NOT Blake3)
                  BindSpace (ClamPath addressing, write/verify)
                  → integrity check, Wisdom/Staunen boundary
                  → stored at W126-127 checksum covers it

spo/semiring.rs:  HammingMin (min-plus over u32)
                  TraversalHop (target_key, distance, truth, cumulative)
                  → chain traversal in SpoStore

spo/store.rs:     SpoStore (in-memory triple store, bitmap ANN queries)
                  query_forward/reverse/relation + TruthGate
                  → bgz17 adds a FAST PATH alongside, not replacement

spo/builder.rs:   SpoRecord (S + P + O + packed + truth)
                  build_forward/reverse/relation_query
                  → 2³ projection verbs

hdr.rs:           Cascade (1467 lines, self-calibrating sigma bands)
                  Welford online stats, reservoir sampling, shift detection
                  3-stage: 1/16 → 1/4 → full
                  → stride-16 sampling hits W112 (Base17 word 0)
                  → automatically benefits from bgz17 annex

cascade_ops.rs:   CascadeScanConfig, hamming_predicate_to_cascade
                  → Cypher WHERE hamming() < threshold → Cascade params
```

## Session Order

```
SESSION A: blasgraph storage + Cypher→Semiring planner
  ADDS: CscStorage, HyperCsrStorage, TypedGraph, blasgraph_planner.rs
  PRESERVES: all SPO modules, Cascade, existing semirings
  WIRES: logical_plan.rs Expand → grb_mxm on TypedGraph

SESSION B: bgz17 container annex + palette semiring + SIMD
  ADDS: container annex writer (W112-125, W96-111, W176-191)
        PaletteSemiring, PaletteMatrix, compose_table, PaletteCsr
        Base17 VSA ops (xor_bind, bundle, permute)
        SIMD batch palette distance (AVX-512/AVX2/scalar)
  PRESERVES: W0-95 untouched, W126-127 checksum still valid
  WIRES: PaletteCsr reads W16-31 inline edges → archetype topology
         Base17 encodes from PLANES (not container)
         Qualia W56-63 maps to Base17 dims (parallel encoding)

SESSION C: ndarray ← bgz17 dual-path integration
  ADDS: bgz17-codec feature flag
        Layered DistanceFn for CLAM/CAKES
        parallel_search() merging HHTL + CLAM results
        TruthGate integration (reads W4-7 for filtering)
  PRESERVES: Cascade (auto-benefits from W112 Base17 via stride-16)
  WIRES: NdarrayFingerprint ↔ Base17 (PLANE conversion, not container)
         CHAODA LFD → generative.rs correction factor
         Results carry TruthValue from W4-7

SESSION D: FalkorDB reality check
  ADDS: FalkorCompat shim, benchmark suite
  PRESERVES: everything
  WIRES: Cypher → our parser → blasgraph planner → TypedGraph + PaletteMatrix
         MotoGP test graph through both Entry 1 and Entry 2
```

## Container Word Allocation for bgz17

```
WORD      CURRENT STATUS    bgz17 USE                     SESSION
────────────────────────────────────────────────────────────────────
W96-111   RESERVED          Local palette CSR              B
          (DN-Sparse)       64 palette-indexed edges:
                            verb_palette(8) + target_palette(8)
                            seeded from W16-31 inline edges
                            mapped through assignments[]

W112-124  RESERVED          Base17 S+P+O annex             B
          (14 words=112B)   base17_s: W112-115 (34 bytes)
                            base17_p: W116-119 (34 bytes)
                            base17_o: W120-123 (34 bytes)
                            W124: 2 bytes padding

W125      RESERVED          Palette indices + temporal      B
          (1 word=8B)       palette_s(8) | palette_p(8) |
                            palette_o(8) | temporal_q(8) |
                            spare(32)

W176-191  DEFERRED          Palette neighbor indices        B
          (Scent index)     [u8; 128] = top-128 neighbors
                            by palette archetype index.
                            Distance via matrix lookup.

W128-143  SPO Crystal       Base17 decode target            B
          (WIDE, PRIO 0)    Can reconstruct from W112-124
                            when full crystal needed.
```

## What bgz17 Reads vs Writes vs Derives

```
READS FROM CONTAINER (prefetch, no computation):
  W112-124: Base17 dims → immediate L1 distance or palette lookup
  W125:     palette indices → immediate matrix[a][b] distance
  W4-7:     NARS truth → TruthGate filter on results
  W16-31:   inline edges → neighbor identity for PaletteCsr
  W56-63:   qualia channels → cross-reference with Base17 dims

WRITES TO CONTAINER (at encode time, covers by W126-127 checksum):
  W112-124: Base17 S+P+O (from plane encoding)
  W125:     palette_s/p/o + temporal_quantile
  W96-111:  local palette CSR (from W16-31 + assignments)
  W176-191: palette neighbor indices (from scope build)
  W126-127: checksum updated to cover new words

READS FROM PLANES (flat 2KB, for encoding):
  subject_plane → Base17::encode() → W112-115
  predicate_plane → Base17::encode() → W116-119
  object_plane → Base17::encode() → W120-123

DERIVES ON THE FLY (never stored):
  scent byte:     palette_distance(W125.my_pal, W125.query_pal) → threshold
  ZeckF64:        zeckf64_from_base(W112-124.my_base, query_base)
  Cascade sigma:  Cascade::calibrate() on scope palette distances
  BNN activation: sign bits of W112-124 Base17 dims
  CLAM tree:      built from scope-level 256×256 distance matrix

PROMOTES ASYNC (hot → cold, background batch):
  W4-7:     NARS truth update via TruthValue::revision()
  W16-31:   new inline edges from discovered relationships
  W48-55:   graph metrics recomputed after topology change
```

## Cascade Stride-16 Sampling (automatic bgz17 benefit)

```
Cascade::query() with stride=16 samples:
  W0   DN address         (structural, high entropy)
  W16  first inline edge  (topology, good discriminator)
  W32  first RL value     (learning state, medium)
  W48  first graph metric (degree/centrality, good)
  W64  first rung history (temporal, medium)
  W80  first repr desc    (encoding meta, low)
  W96  palette CSR word 0 (bgz17 local topology → excellent)  ← NEW
  W112 Base17 word 0      (bgz17 compressed encoding → excellent)  ← NEW
  W128 SPO Crystal word 0 (crystallized content, good)
  ...

Base17 at W112 makes stage 1 (1/16 sample) more discriminating
without ANY code change to the Cascade. Just fill the words.
```

## Parallel Search with TruthGate

```
pub fn parallel_search(
    scope: &LayeredScope,
    palette_csr: &PaletteCsr,
    query: &SpoBase17,
    k: usize,
    gate: TruthGate,              // ← reads W4-7 per candidate
) -> Vec<(usize, u32, TruthValue)> {  // ← returns truth with results

    // Path 1: HHTL (sequential)
    let hhtl = scope.search(...);

    // Path 2: CLAM on archetype tree
    let clam = palette_csr.search(query, k);

    // Merge and re-rank
    let merged = merge_and_rerank(hhtl, clam, query, &scope.base_patterns, k);

    // TruthGate filter: read W4-7 from each candidate's container
    merged.into_iter()
        .filter_map(|(pos, dist)| {
            let truth = read_truth_from_container(pos);  // W4-7
            if gate.passes(&truth) {
                Some((pos, dist, truth))
            } else {
                None
            }
        })
        .collect()
}
```

## Success Criteria

```
✓ Base17 encodes from S/P/O PLANES (not from container)
✓ Base17 stored at W112-124, palette at W125 (inside existing container)
✓ Cascade stride-16 automatically samples W112 (no code change)
✓ PaletteCsr reads W16-31 inline edges → archetype topology
✓ Parallel search returns (position, distance, TruthValue)
✓ TruthGate applied AFTER distance computation (same as SpoStore)
✓ W126-127 checksum covers bgz17 annex
✓ Qualia W56-63 maps to Base17 dims (17 ≈ 18 channels)
✓ Compose table enables multi-hop in palette space (2-hop ρ > 0.9)
✓ MerkleRoot (spo/merkle.rs) unchanged — integrity only
✓ NARS revision (spo/truth.rs) unchanged — evidence, not geometry
✓ HammingMin semiring (spo/semiring.rs) works on palette distances too
✓ FalkorDB reality check passes on MotoGP graph (same Jan→Ada→Max)
```

## Session Prompts

```
session_A_v3_blasgraph_csc_planner.md
session_B_v3_bgz17_container_semiring.md
session_C_v3_ndarray_bgz17_dualpath.md
session_D_v3_falkordb_retrofit.md
```
