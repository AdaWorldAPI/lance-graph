# Semiring Algebra Surface: Every Ring, Every Repo

> *A semiring (S, ⊕, ⊗, 0, 1) has additive monoid (S, ⊕, 0) and multiplicative monoid (S, ⊗, 1)
> where ⊗ distributes over ⊕. Graph algorithms ARE semiring computations.*

## 1. Complete Semiring Inventory

### 1A. BlasGraph HDR Semirings (lance-graph/blasgraph/semiring.rs)

Operate on `BitVec = [u64; 256]` (16,384-bit hyperdimensional vectors).

| # | Name | ⊗ (Multiply) | ⊕ (Add) | 0 (Zero) | Use Case | Code |
|---|------|-------------|---------|----------|----------|------|
| 1 | **XorBundle** | XOR | Majority vote (bundle) | all-zeros | Path composition, creative association | `BinaryOp::Xor` / `MonoidOp::BundleMonoid` |
| 2 | **BindFirst** | XOR | First non-empty | all-zeros | BFS traversal (first path found) | `BinaryOp::Xor` / `MonoidOp::First` |
| 3 | **HammingMin** | XOR (→ popcount) | Min popcount | all-zeros | Shortest path (Hamming distance) | `BinaryOp::Xor` / `MonoidOp::MinPopcount` |
| 4 | **SimilarityMax** | XOR (→ similarity) | Max popcount | all-zeros | Best match (highest similarity) | `BinaryOp::Xor` / `MonoidOp::MinPopcount` (inverted) |
| 5 | **Resonance** | XOR bind | Best density | all-zeros | Query expansion (creative search) | `BinaryOp::Xor` / `MonoidOp::BestDensity` |
| 6 | **Boolean** | AND | OR | all-zeros | Reachability (can A reach B?) | `BinaryOp::And` / `MonoidOp::Or` |
| 7 | **XorField** | XOR | XOR | all-zeros | GF(2) field algebra (linear codes) | `BinaryOp::Xor` / `MonoidOp::XorField` |

**Matrix operations**: `GrBMatrix::mxm(A, B, semiring)` and `GrBMatrix::mxv(A, v, semiring)`
**Sparse formats**: COO, CSR, CSC, HyperCSR — chosen by sparsity pattern.

### 1B. SPO Truth Semiring (lance-graph/spo/semiring.rs)

Operates on `(frequency: f32, confidence: f32, hamming: u32)` tuples.

| # | Name | ⊗ (Multiply) | ⊕ (Add) | Use Case |
|---|------|-------------|---------|----------|
| 1 | **HammingMin** | Track min hamming distance along edge | Min of accumulated distances | Nearest-neighbor path cost |

**TraversalHop**: `{ fingerprint_hash: u64, distance: u32, truth: TruthValue }`
Used in SPO triple store for truth-value-aware traversal.

### 1C. Palette Semiring (bgz17/palette_semiring.rs)

Operates on `palette_index: u8` (256 archetypes) with precomputed 256×256 distance table.

| # | Name | ⊗ (Multiply) | ⊕ (Add) | Use Case |
|---|------|-------------|---------|----------|
| 1 | **PaletteCompose** | `palette.nearest(a.xor_bind(b))` | Min distance | Path composition via archetype algebra |

**SpoPaletteSemiring**: Three parallel semirings (one per S/P/O plane) running simultaneously.
**Distance table**: Precomputed L1 distance on i16[17] base patterns. 256×256 = 65K entries.

### 1D. Planner Truth-Propagating Semiring (lance-graph-planner/physical/accumulate.rs)

Operates on `SemiringValue::Truth { frequency: f32, confidence: f32 }`.

| # | Name | ⊗ (Multiply) | ⊕ (Add) | Use Case |
|---|------|-------------|---------|----------|
| 1 | **TruthPropagating** | Deduction: f = f_A × f_B, c = f_A × c_A × f_B × c_B | Revision: weighted evidence merge | NARS inference along graph edges |

**Deduction** (⊗): Following an edge reduces both frequency and confidence.
**Revision** (⊕): Merging paths increases confidence with more evidence.

### 1E. Planner IR Semiring Types (lance-graph-planner/ir/logical_op.rs)

Abstract semiring identifiers used in the logical plan IR:

| Variant | Maps To | Selected By |
|---------|---------|------------|
| `SemiringType::Boolean` | BlasGraph Boolean OR DataFusion | Default |
| `SemiringType::Tropical` | min-plus on edge weights | shortestPath queries |
| `SemiringType::XorBundle` | BlasGraph XorBundle | RESONATE queries, creative styles |
| `SemiringType::HammingMin` | BlasGraph HammingMin | Fingerprint WHERE clauses |
| `SemiringType::TruthPropagating` | Planner TruthPropagatingSemiring | NARS revision/synthesis |

### 1F. Contract Semiring Choices (lance-graph-contract/nars.rs)

| Variant | Intended Consumer | Maps To |
|---------|------------------|---------|
| `SemiringChoice::Boolean` | Standard queries | BlasGraph Boolean |
| `SemiringChoice::HammingMin` | Distance queries | BlasGraph HammingMin |
| `SemiringChoice::NarsTruth` | NARS reasoning | Planner TruthPropagating |
| `SemiringChoice::XorBundle` | Creative/resonance | BlasGraph XorBundle |
| `SemiringChoice::CamPqAdc` | Compressed search | CAM-PQ distance tables |

## 2. Semiring Composition (How They Stack)

```
Query: "MATCH (a)-[:CAUSES*1..5]->(b) WHERE RESONATE(a.fp, $q, 0.7) RETURN a, b"

THINKING ORCHESTRATION:
  Style = Analytical → resonance_threshold = 0.85
  NARS type = Abduction (multi-hop causal query)
  Semiring auto-select → XorBundle (RESONATE keyword detected)

ELEVATION STACK:
  L0: BindSpace lookup for $q → fingerprint
  L1: Scan neighborhood with scent pre-filter (ScentCsr, u8 semiring)
  L2: CAM-PQ cascade on survivors (CamPqAdc semiring on 6-byte codes)
  L3: Full XorBundle mxv on remaining candidates (BlasGraph semiring)

RESULT ACCUMULATION:
  path_a via XorBundle: bind(a, CAUSES, intermediate, CAUSES, b)
  path_b via XorBundle: bind(a, CAUSES, other_intermediate, CAUSES, b)
  merge via BundleMonoid: majority_vote(path_a, path_b) → superposition
```

## 3. The Cold Path Numbing Effect

The user's insight: *"the cold path of joins of columns and rows should numb the thinking"*

This means:

| Path | Semiring | Temperature | What Happens |
|------|----------|-------------|-------------|
| **Hot** (BindSpace, fingerprint scan) | XorBundle, HammingMin, Resonance | Hot | Full HDR vectors, SIMD popcount, creative association |
| **Warm** (CAM-PQ cascade) | CamPqAdc | Warm | 6-byte compressed codes, 500M candidates/sec |
| **Cold** (DataFusion columnar joins) | Boolean | Cold | Arrow RecordBatch joins, standard SQL semantics |
| **Frozen** (metadata skeleton) | None | Frozen | Pure CRUD on metadata.rs, no semiring needed |

**The new awareness codecs (CAM-PQ, BGZ17) create a middle ground**: warm-path pre-filtering that
reduces the hot-path workload by 99%+ before the semiring algebra fires.

The cold path "numbs" thinking because:
1. DataFusion joins operate on scalar columns (node_id, label, properties)
2. No fingerprint data flows through cold joins
3. The semiring only fires AFTER the join narrows the candidate set
4. This is correct: you want columnar efficiency for structural matching, then algebraic richness for semantic matching

## 4. Semiring × Adjacency Matrix

| Adjacency Type | Compatible Semirings | Optimal Use |
|---------------|---------------------|-------------|
| **AdjacencyStore** (planner CSR/CSC) | Boolean, Tropical, TruthPropagating | Standard graph queries, NARS reasoning |
| **GrBMatrix** (blasgraph sparse) | All 7 HDR semirings | Hyperdimensional graph algebra |
| **ScentCsr** (u8 edge weights) | HammingMin (approximate) | Fast pre-filter, BFS, PageRank |
| **PaletteCsr** (palette indices) | PaletteCompose | Compressed scope search |
| **Dense matrix** (binding_matrix) | XorBundle | Full binding popcount analysis |

## 5. Missing Semiring Wiring

| Gap | What's Missing | Impact |
|-----|---------------|--------|
| Planner → BlasGraph | Planner selects `SemiringType::XorBundle` but can't call `GrBMatrix::mxm()` | Plan exists but can't execute on HDR vectors |
| CAM-PQ → Semiring | CamPqAdc is a choice but no semiring trait impl for CAM-PQ distance | Can't compose CAM-PQ search with graph traversal |
| Palette → Planner | PaletteSemiring exists in bgz17 but planner doesn't know about it | Can't route to palette-accelerated traversal |
| TruthPropagating → SPO | Planner has TruthPropagatingSemiring, SPO store has TruthSemiring — different impls | Potential semantic mismatch in truth accumulation |
