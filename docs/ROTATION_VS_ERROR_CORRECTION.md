# Rotation vs Error Correction: SPO Codebook & Graph Reasoning

> Why bgz17/CAM-PQ uses Euler-Gamma rotation with Fibonacci codebooks
> instead of post-quantization error correction. How the SPO-triple
> structure eliminates the class of problems that TurboQuant/QJL solves.
>
> Scope: lance-graph CAM-PQ, HHTL cascade, SPO codebook, BF16→FP32 hydration
>
> Cross-reference: ndarray/docs/ROTATION_VS_ERROR_CORRECTION.md (kernel perspective)

## 1. What TurboQuant Does and Why We Don't Need It

Google TurboQuant (ICLR 2026) solves KV-cache compression for LLMs:

```
Problem:  LLM attention stores key-value pairs at FP32/FP16
          Long context → cache grows → GPU memory exhausted
Solution: Compress to 3 bits, correct the error with 1-bit QJL
Target:   H100 GPU, KV-cache is a flat key→value store
```

bgz17/CAM-PQ solves a different problem:

```
Problem:  Knowledge graph with nodes, edges, truth values
          Need to search, reason, and infer — not just retrieve
Solution: 4097-entry SPO codebook at 12 bits, cascading resolution
Target:   CPU (AVX2/NEON) or NPU (INT8), graph IS the data structure
```

The fundamental difference: TurboQuant compresses vectors.
bgz17 encodes knowledge graph topology.

## 2. The SPO Codebook: 3 × 12 Bit

A bgz17 fingerprint is not a compressed vector. It is an SPO address:

```
┌──────────┬──────────┬──────────┐
│ Subject  │ Predicate│ Object   │
│ 12 bit   │ 12 bit   │ 12 bit   │
│ (HEEL)   │ (HIP)    │ (TWIG)   │
└──────────┴──────────┴──────────┘
     = 36 bits = 4.5 bytes per triple

4097 codebook entries per dimension (12 bit + 1 for null/unknown)
Total addressable space: 4097³ ≈ 68.7 billion unique SPO triples
```

Each 12-bit code maps to a Fibonacci-spaced position in the concept space.
The codebook is **not trained** on data — it is defined by Fibonacci number
theory. Same codebook for Wikidata, aiwar, or any future dataset.

### Contrast with TurboQuant

```
TurboQuant codebook: implicitly defined by PolarQuant rotation
  → data-oblivious (good)
  → but entries have no semantic meaning
  → code 0x1A3 has no interpretation

bgz17 codebook: defined by Fibonacci geometry
  → data-oblivious (same)
  → but entries ARE semantic coordinates
  → HEEL code 0x1A3 = specific category in the concept tree
  → the code IS the meaning, not a compressed approximation of it
```

## 3. Resolution Modes

The same SPO content can be represented at three resolution levels:

```
Mode          Representation         Size/triple    Operation
────────────  ─────────────────────  ─────────────  ──────────────────
CAM/bgz17     3 × 12-bit SPO index  4.5 bytes      Table lookup (vpshufb)
Hamming       3 × 16 kbit fingerp.  6,144 bytes    XOR + popcount scan
FP32          3 × 32-bit values     12 bytes       Full precision reasoning
```

The HHTL cascade traverses these levels:

```
HEEL (12-bit):   Category-level match. 90% rejected. O(1) table lookup.
HIP  (12-bit):   Relationship-level match. 90% of survivors rejected.
TWIG (12-bit):   Contextual refinement. 90% again.
LEAF (variable):  Final candidates hydrated to BF16 → FP32.

1,000,000 candidates → HEEL: 100,000 → HIP: 10,000 → TWIG: 1,000 → LEAF: ~50
Only 50 candidates reach FP32. The rest were resolved at INT8.
```

TurboQuant has one resolution: 3-bit compressed, period.
No cascading, no progressive refinement, no early rejection.

## 4. Rotation: Euler-Gamma vs PolarQuant

### The Normalization Problem

Standard PQ (FAISS) stores per-block normalization constants:
```
block = [v₁, v₂, ..., v_k]
min_val = min(block)
max_val = max(block)
scale = (max_val - min_val) / (2^bits - 1)
quantized = round((v - min_val) / scale)

Storage: quantized values + min_val (FP32) + scale (FP32)
Overhead: 64 bits per block = 1-2 bits/value at small block sizes
```

### PolarQuant's Solution

```
Rotate with randomized Hadamard → distribution becomes uniform
Convert to polar: radius + angles
Angles are concentrated → quantize without per-block constants
Overhead: zero
```

### bgz17's Solution

```
Euler-Gamma rotation (Fujifilm X-Sensor pattern):
  → Bundle rotation with γ ≈ 0.5772 spacing
  → Distribution equalized without Hadamard matrix
  → No separate normalization step

Fibonacci encoding:
  → Codebook IS the grid — fixed positions at σ intervals
  → No min/max/scale per block
  → No per-block constants at all
  → Overhead: zero (same as PolarQuant, different mechanism)
```

### Why Euler-Gamma Instead of Hadamard

Hadamard rotation requires:
- Generating/storing the rotation matrix (or using a structured one)
- Matrix-vector multiply to rotate each input
- This is FP32 arithmetic — needs GPU or FP SIMD

Euler-Gamma rotation requires:
- Reordering the bundle dimensions by γ-spacing
- Fibonacci encoding of the reordered values
- This is integer reindexing + table lookup — runs on INT8/NPU

The rotation is baked into the encoding step. No separate matrix multiply.

## 5. Error Correction: QJL vs "No Error"

### Why TurboQuant Needs QJL

PolarQuant quantizes uniformly within each angular bucket.
Values near bucket boundaries get rounded → systematic bias.
The bias accumulates across dimensions → attention scores drift.

QJL fixes this: project the residual, keep the sign, correct the score.
Cost: 1 bit per value. Benefit: unbiased attention.

### Why bgz17 Does Not Need Error Correction

The codebook entries sit at 1/4σ discrete positions.
Qualia (semantically distinct concepts) sit at 3σ separation.

```
3σ in Gaussian normal distribution:
  99.73% of values fall within ±3σ of the mean
  P(wrong code assignment) = 0.13%
  For comparison: medical test at 99% accuracy = "very good"
  bgz17 code assignment = 99.87% = 7× more accurate

Within one code:
  3σ / (1/4σ) = 12 discrete positions
  Each position is exact — no interpolation, no rounding
  The engine knows the exact diff value between adjacent positions
```

There is no residual error because there is no continuous-to-discrete
mapping. The codebook IS discrete. The input is mapped to the nearest
discrete coordinate, and the distance to that coordinate is known exactly.

Analogy: latitude 48°31'22" is not a rounded float. It IS the position
at arc-second resolution. Going to 48°31'23" is exactly 1 arc-second
movement, not an approximation.

## 6. Nodes and Edges in Metadata

This is the capability TurboQuant does not address at all.

A bgz17 fingerprint carries graph topology:

```
Subject  (12 bit) = which node
Predicate(12 bit) = which edge type  
Object   (12 bit) = which target node

Plus metadata:
  - Truth value: frequency × confidence (NARS)
  - Provenance: source document/date
  - Edge weight: deduced/abduced/induced
```

A query returns not "distance: 0.23" but:
```
"Palantir DEVELOPS Gotham" (confidence: 0.94, source: contract 2023)
  → "Gotham DEPLOYED_BY US DoD" (confidence: 0.87, source: press 2024)
  → inferred: "Palantir CONNECTED_TO US DoD" (confidence: 0.82)
```

The path through the graph IS the explanation.
TurboQuant returns a compressed vector. bgz17 returns a logical chain.

## 7. BF16 → FP32 Hydration Path

After HHTL cascade reduces candidates to ~50, hydration begins:

```
INT8 CAM fingerprint (6 bytes)
  → BF16 intermediate (10-bit mantissa preserves upper signal bits)
  → FP32 final resolution (full precision for SPO reasoning)

This is zoom-in, not error recovery.
The INT8 representation was exact at INT8 resolution.
BF16 adds precision that was not stored, not precision that was lost.
FP32 adds more.

Contrast with TurboQuant:
  3-bit compressed → decompress to FP16/FP32 → the gap is error
  QJL corrects the error
  
bgz17:
  INT8 exact → BF16 extends → FP32 extends → no error to correct
```

## 8. Hardware Implications

| Operation | TurboQuant | bgz17 |
|---|---|---|
| Rotation | FP32 Hadamard (GPU matmul) | INT8 reindexing (table lookup) |
| Quantization | Uniform scalar (GPU) | Fibonacci position (table lookup) |
| Distance | FP polar arithmetic (GPU) | INT8 table lookup (vpshufb/vtbl) |
| Error correction | QJL sign bits (GPU) | Not needed |
| Cascade | None (single stage) | HHTL 4-stage (90% rejection each) |
| Hydration | Decompress to FP16 | Zoom-in: INT8 → BF16 → FP32 |
| Hardware | H100 GPU (€25,000) | Any AVX2 CPU or INT8 NPU (€75) |

## 9. What We Can Learn from TurboQuant

Despite not needing TurboQuant's error correction, two ideas are worth noting:

### 9.1 Polar Decomposition Before Fibonacci Encoding

PolarQuant separates radius (signal strength) from angles (meaning).
bgz17's HEEL already captures category (≈ meaning) while Fibonacci upper
bits capture magnitude (≈ signal strength). But explicitly applying
polar decomposition before Fibonacci encoding could improve codebook
utilization — the radius becomes a natural HEEL value, angles map to
HIP/TWIG/LEAF.

**Status**: Architecturally compatible. Not yet implemented.
**Estimate**: ~300 LOC in lance-graph cam_pq.rs.

### 9.2 Squeeze-and-Excitation for HHTL

MobileNetV3's channel attention: weight subspaces by query relevance.
Applied to HHTL: not all cascade stages are equally informative for
every query. A surveillance query should weight HEEL (category) higher
than LEAF (fine detail). The cascade could skip uninformative stages.

This is routing, not arithmetic: "which table to query first?"
rather than "how to weight the distance?"

**Status**: Design documented, not yet implemented.
**Estimate**: ~50 LOC in lance-graph cam_pq.rs (weighted distance tables).

---

*Document created: 2026-03-26*
*Cross-reference: ndarray/docs/ROTATION_VS_ERROR_CORRECTION.md (kernel perspective)*
*Related: lance-graph/.claude/DEEP_ADJACENT_EXPLORATION.md*
*Related: lance-graph/.claude/FALKORDB_ANALYSIS.md*
*Related: lance-graph/docs/WIKIDATA_HHTL_TILES.md*
