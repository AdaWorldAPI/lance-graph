# KNOWLEDGE.md — bgz17 Crate Reference

## What This Is

**bgz17** = **b**las**g**raph + **z**eck**17**. A unified distance codec that
compresses 49,152-byte SPO planes to 3 bytes per edge via palette indexing,
with precomputed 256×256 distance matrices for O(1) lookup.

## The Layered Distance Codec

```
Layer 0: Scent (1 byte)      — Hamming on 7-bit lattice,   ρ=0.937
Layer 1: Palette (3 bytes)   — matrix[s][s'] + matrix[p][p'] + matrix[o][o'],  ρ≈0.965
Layer 2: ZeckBF17 (102 bytes)— i16[17] L1 per plane,       ρ=0.992
Layer 3: Full planes (6 KB)  — exact Hamming,               ρ=1.000

95%+ of searches terminate at Layer 0-1 (CAKES triangle inequality).
Layer 2 for decision-boundary cases. Layer 3 almost never loaded.
```

## Critical Insight: L2 BitVec (ρ=0.834) is WRONG baseline

The integrated 16Kbit BitVec bundles S⊕P⊕O into ONE vector, DESTROYING
plane separation. It scores LOWER than the 1-byte scent because the scent's
7 bits encode WHICH planes are close while the BitVec can't.

bgz17 preserves plane separation at every layer (three separate indices,
three separate matrices, three separate base patterns). It compares against
ρ=0.937 (scent) not ρ=0.834 (integrated BitVec).

## Module Map

```
base17.rs           — i16[17] base pattern, golden-step encode/decode, L1 distance
palette.rs          — k-means codebook (≤256 entries), palette encode/decode
distance_matrix.rs  — precomputed 256×256 u16 pairwise L1 matrices (128 KB, fits L1 cache)
tripartite.rs       — cross-plane S×P×O reasoning via scalar sparse matrices
scalar_sparse.rs    — scalar CSR with standard and min-plus (tropical) semiring SpMV
layered.rs          — the 4-layer search codec: scent → palette → base → exact
scope.rs            — scope builder: raw planes → bgz17 scope ready for search
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
