# KNOWLEDGE.md — bgz17 Crate Reference

## What This Is

**bgz17** = **b**las**g**raph + **z**eck**17**. A unified distance codec that
compresses 49,152-byte SPO planes to 3 bytes per edge via palette indexing,
with precomputed 256×256 distance matrices for O(1) lookup.

## The Layered Distance Codec

```
Layer 0: Scent (1 byte)      — Hamming on 7-bit lattice,   ρ=0.937  ⚠️ NOT metric-safe
Layer 1: Palette (3 bytes)   — matrix[s][s'] + matrix[p][p'] + matrix[o][o'],  ρ≈0.965  ✓ metric-safe
Layer 2: ZeckBF17 (102 bytes)— i16[17] L1 per plane,       ρ=0.992  ✓ metric-safe
Layer 3: Full planes (6 KB)  — exact Hamming,               ρ=1.000  ✓ metric-safe

95%+ of searches terminate at Layer 0-1 (CAKES triangle inequality).
Layer 2 for decision-boundary cases. Layer 3 almost never loaded.
```

## Metric Safety (CRITICAL for CAKES correctness)

CAKES DFS sieve requires triangle inequality: d(a,c) ≤ d(a,b) + d(b,c).

**Palette (Layer 1):** L1 on i16[17]. IS a metric. Safe for CAKES pruning.
**Base (Layer 2):** L1 on i16[17]. IS a metric. Safe for CAKES pruning.
**Scent (Layer 0):** Hamming on 7-bit Boolean lattice. NOT a metric.
  The 19-pattern constraint means some "distances" violate triangle inequality.
  Use ONLY as heuristic pre-filter (HEEL stage). NEVER for CAKES bounds.

Production search path:
  HEEL (Scent, heuristic, 10K → 200) → CAKES sieve (Palette, metric-safe, 200 → k)

`distance_adaptive()` guarantees Palette-minimum precision.
`distance_heuristic()` returns Scent — caller must NOT use for CAKES bounds.

## Critical Insight: L2 BitVec (ρ=0.834) is WRONG baseline

The integrated 16Kbit BitVec bundles S⊕P⊕O into ONE vector, DESTROYING
plane separation. It scores LOWER than the 1-byte scent because the scent's
7 bits encode WHICH planes are close while the BitVec can't.

bgz17 preserves plane separation at every layer (three separate indices,
three separate matrices, three separate base patterns). It compares against
ρ=0.937 (scent) not ρ=0.834 (integrated BitVec).

## Module Map

```
CORE CODEC:
  base17.rs           228  — i16[17] base pattern, golden-step encode/decode, L1 distance
  palette.rs          256  — k-means codebook (≤256 entries), palette encode/decode
  distance_matrix.rs  163  — precomputed 256×256 u16 pairwise L1 matrices (128 KB, L1 cache)
  scope.rs            227  — scope builder: raw planes → bgz17 scope ready for search

SEARCH:
  layered.rs          284  — the 4-layer search codec: scent → palette → base → exact
  bridge.rs           480  — Bgz17Distance trait, metric safety contract, CAKES/HHTL wiring
  router.rs           240  — query routing: bgz17 vs blasgraph vs ndarray cascade fallback
  prefetch.rs         448  — software prefetch pipeline + LFD correction (Zend Optimizer)
  clam_bridge.rs      777  — concrete DistanceFn injection into CLAM/CAKES

ALGEBRA:
  scalar_sparse.rs    149  — scalar CSR with standard + min-plus (tropical) semiring SpMV
  tripartite.rs       171  — cross-plane S×P×O reasoning via scalar sparse matrices
  generative.rs       225  — arXiv:2602.03505 LFD correction, anomaly→layer routing
```

## Synergy Map: bgz17 × ndarray × blasgraph

### Encode Path

```
ndarray HPC                         lance-graph blasgraph
──────────                          ─────────────────────

Plane(i8[16384])                    BitVec(u64[256])
     ↓                                   ↓
Base17(i16[17])                     ZeckF64(u64, 8 bytes)
     ↓                                   ↓
Palette(u8)                         Scent(u8, byte 0)
```

### Search Path — HHTL with bgz17 LEAF Replacement

```
HEEL (10KB)   scent XOR popcount ← bgz17 Layer 0 (same, heuristic)
HIP  (500KB)  scent XOR popcount ← bgz17 Layer 0 (same, heuristic)
TWIG (500KB)  scent XOR popcount ← bgz17 Layer 0 (same, heuristic)
                    ↓ 50 candidates
              ┌────── DECISION POINT ──────┐
              │                             │
  LEAF L1     │  OLD: BitVec Hamming 16Kbit │  ← KILL THIS
              │  ρ=0.834, 2KB, WRONG        │
              │                             │
              │  NEW: bgz17 palette lookup  │  ← USE THIS
              │  ρ=0.965, 3 bytes, O(1)     │
              │                             │
  LEAF L2     │  bgz17 Base17 L1 (top 10)  │
              │  ρ=0.992, 102 bytes          │
              │                             │
  LEAF L3     │  exact S+P+O (top 3)       │
              │  ρ=1.000, 6KB per edge       │
              └─────────────────────────────┘
```

### Three Synergies

**Synergy 1: bgz17 replaces integrated BitVec in LEAF L1**

```
BEFORE: LEAF L1 loads 2KB BitVec, computes 16K-bit Hamming  → ρ=0.834
AFTER:  LEAF L1 loads 3 bytes, does 3 matrix lookups         → ρ=0.965
Speedup:  ~10,000× (cache load vs popcount chain)
Storage:  680× smaller
```

**Synergy 2: CLAM tree uses bgz17 palette as its metric**

ClamTree takes `DistanceFn = fn(&[u8], &[u8]) -> u64` (function pointer, zero-cost).
bgz17 palette L1 IS a valid metric (triangle inequality verified, 0 violations).
clam_bridge.rs resolves 99.4% at palette, 0.6% at base.

**Synergy 3: panCAKES XOR-diff compresses Base17 layer**

panCAKES stores points as XOR-diffs from cluster center.
Applied to Base17: cluster center = palette entry (34B shared),
diff = ~5-8 dimensions × 3 bytes = 15-24 bytes instead of 102.

### Query Routing (router.rs)

```
SearchType::Knn / Range  → bgz17 Layered (if ≥32 edges, else ndarray cascade)
SearchType::Bfs / Sssp   → blasgraph grb_mxm (multi-hop needs BitVec XOR)
SearchType::SemiringOp   → blasgraph (XOR_BUNDLE, RESONANCE, BIND_FIRST)
SearchType::AnomalyDetect→ CLAM+CHAODA (if tree exists, else ndarray cascade)
```

### Fallback Signals

| Situation | Detection | Fallback |
|---|---|---|
| Scent prune insufficient | <30% eliminated at Layer 0 | Skip HEEL, brute-force palette |
| Palette too coarse | >10% collision rate | Escalate to Layer 2 (Base17) |
| CLAM tree too shallow | depth < 3 | ndarray cascade Stroke 1-3 |
| High anomaly | CHAODA score > 0.75 | Load full planes (Layer 3) |
| Multi-hop reasoning | BFS/SSSP query | blasgraph grb_mxm |
| New scope bootstrap | < 32 edges | ndarray cascade until palette buildable |

### SIMD Integration

| ndarray Hot Path | With bgz17 | Speedup |
|---|---|---|
| cascade Stroke 1 (128B prefix) | scent byte XOR (1B) | ~128× |
| cascade Stroke 2 (2KB full) | palette lookup (3B) | ~10,000× |
| clam_search::rho_nn | palette on 3B edges | ~10,000× |
| clam_compress::hamming_to_compressed | XOR-diff on 102B Base17 | ~20× |
| hamming_distance_raw (2KB) | NOT REPLACED (Layer 3 fallback) | 1× |
| grb_mxm inner product | NOT REPLACED (needs BitVec XOR) | N/A |

### Key Insight: bgz17 IS ndarray's Stroke 1+2 Replacement

```
ndarray Cascade          bgz17 Layered           Relationship
───────────────         ──────────────           ────────────
Stroke 1 (prefix)   →   Layer 0 (scent)          SAME (both heuristic)
Stroke 2 (full Ham) →   Layer 1 (palette)         bgz17 WINS (O(1) vs O(n))
Stroke 3 (precision)→   Layer 2 (base17)          bgz17 WINS (102B vs 2KB)
                    →   Layer 3 (exact planes)     SAME (both fallback)
```

## How blasgraph Connects

blasgraph's `GrBMatrix` stores `BitVec`-valued edges with 7 semirings.
bgz17's `ScalarCsr` stores scalar-valued edges with 2 semirings
(standard multiply-add and min-plus tropical).

The 256×256 distance matrix IS a materialized semiring result:
```
BEFORE: distance = hamming(bitvec_a, bitvec_b)     → ~16K bit ops
AFTER:  distance = matrix[palette_a][palette_b]     → 1 cache load
```

The tripartite cross-plane matrices (SP, SO, PO) generalize the scent
byte's 3 pairwise bits to continuous distances at palette resolution.
blasgraph's `grb_mxm` with HammingMin semiring computes multi-hop paths;
bgz17's `spmv_min_plus` computes shortest paths in palette space.

bgz17 CANNOT replace blasgraph for: multi-hop BFS/SSSP, semiring algebra
(XOR_BUNDLE, RESONANCE, BIND_FIRST), or any operation needing BitVec XOR
as the multiplicative operator. These fall back to blasgraph via router.rs.

## Storage Per Scope (10K edges)

```
Component          Size        Layer
─────────────────────────────────────
scent              10 KB       0  (always hot)
palette indices    30 KB       1  (3 bytes × 10K)
codebooks          26 KB       1  (3 × 8.7 KB, amortized)
distance matrices  384 KB      1  (3 × 128 KB)
base patterns      1.0 MB      2  (102 bytes × 10K)

Total (compact):   ~450 KB     Layers 0-1 only
Total (full):      ~1.45 MB    Layers 0-2
Full planes:       ~480 MB     Layer 3

Compression (compact): 1,067:1
Compression (full):    331:1
```

## How To Run

```bash
cd crates/bgz17
cargo test -- --nocapture
```
