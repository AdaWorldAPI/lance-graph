# Codec Compression Atlas: From 16,384 Bits to 1 Byte

> *The complete compression chain. Every codec, every ratio, every trade-off.*

## 1. The Pareto Frontier (Empirical, from ndarray benchmarks)

Only **3 points** on the precision-compression Pareto frontier:

| Bits | Spearman ρ (random) | ρ (structured) | Compression | Bytes/node |
|------|-------------------|----------------|-------------|------------|
| 8 | 0.937 | 0.899 | 6,144× | 1 |
| 57 | 0.982 | 0.986 | 862× | ~7 |
| 49,152 | 1.000 | 1.000 | 1× | 6,144 |

Everything between these points is **dominated** — worse compression AND worse precision.
This is the mathematical foundation for the entire codec stack.

## 2. The Complete Compression Chain

```
LAYER 0: FULL PLANES (Reference)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Type:     Fingerprint<256> = [u64; 256]
  Size:     2,048 bytes per plane (16,384 bits)
  SPO:      6,144 bytes per triple (3 planes)
  ρ:        1.000 (reference)
  Location: ndarray/src/hpc/fingerprint.rs
            lance-graph/blasgraph/types.rs (BitVec)
            ladybug-rs/storage/bind_space.rs (BindNode)
  Schema:   cognitive_nodes.lance: FixedSizeBinary(2048) per plane

         │ ZeckBF17 encode: golden-step octave averaging
         │ 17 base dims × 964 octaves → i16[17] base + u8[14] envelope
         ▼

LAYER 1: ZECKBF17 (Research Codec)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Type:     { base: [i16; 17], envelope: [u8; 14] }
  Size:     48 bytes per plane (34 base + 14 envelope)
  SPO:      144 bytes per triple
  ρ:        ~0.982 (matches Pareto 57-bit level)
  Compress: 341:1 per plane, 424:1 per SPO edge (with 116-byte encoding)
  Location: lance-graph/crates/lance-graph-codec-research/src/zeckbf17.rs
  WHY 17:   17 is PRIME → no non-trivial subspace decomposition
            Golden step = round(17/φ) = 11. gcd(11,17)=1 → visits ALL 17 residues
            16 = 2⁴ would alias. The 17th dimension IS the Pythagorean comma.

         │ Drop envelope, keep base only
         ▼

LAYER 2: BASE17 / BGZ17 (Production Codec)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Type:     Base17 { dims: [i16; 17] }
  Size:     34 bytes per plane
  SPO:      102 bytes per triple (SpoBase17)
  ρ:        ~0.965 (Layer 1 of HHTL cascade)
  Compress: 482:1 per plane
  Location: lance-graph/crates/bgz17/src/base17.rs (ORIGINAL)
            ndarray/src/hpc/bgz17_bridge.rs (PORT)
  Algebra:  xor_bind(a, b) on i16[17] — dimension-wise XOR of sign bits
            bundle(a, b) — element-wise average (majority vote analog)
            L1 distance for similarity (maps to palette_distance)
  Palette:  256 archetypes via k-means on Base17 patterns
            PaletteEdge = { s_idx: u8, p_idx: u8, o_idx: u8 } = 3 bytes/edge
            PaletteCsr for O(k²) scope search

         │ Palette assignment → 3 bytes per node
         │ OR CAM-PQ encoding → 6 bytes per vector
         ▼

LAYER 3A: PALETTE EDGE (Graph Codec)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Type:     PaletteEdge { s_idx: u8, p_idx: u8, o_idx: u8 }
  Size:     3 bytes per edge
  ρ:        ~0.937 (matches Pareto 8-bit level, 3× redundancy for SPO)
  Compress: 2,048× per edge
  Location: bgz17/src/palette.rs + palette_csr.rs
            ndarray/src/hpc/bgz17_bridge.rs:46
  Algebra:  Distance via precomputed 256×256 L1 table (65K entries)
            Compose: palette.nearest(palette[a].xor_bind(palette[b]))
            SpoPaletteSemiring: three semirings, one per S/P/O plane

LAYER 3B: CAM-PQ (Vector Codec)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Type:     CamFingerprint = [u8; 6]
  Size:     6 bytes per vector (48 bits)
  ρ:        varies by domain (trained codebook)
  Compress: 170× for 256D, 682× for 1024D
  Speed:    500M candidates/sec (AVX-512 VPGATHERDD)
  Location: ndarray/src/hpc/cam_pq.rs (IMPLEMENTATION)
            lance-graph-contract/src/cam.rs (CONTRACT)
            lance-graph-planner/src/physical/cam_pq_scan.rs (OPERATOR)
            lance-graph-planner/src/api.rs:418 (CamSearch API)
  Layout:   FAISS PQ6x8 compatible — 6 subspaces × 256 centroids
  Bytes:    HEEL / BRANCH / TWIG_A / TWIG_B / LEAF / GAMMA
  Cascade:  Stroke 1 (HEEL only):   1 byte → 90% rejected
            Stroke 2 (HEEL+BRANCH): 2 bytes → 90% of survivors rejected
            Stroke 3 (full 6-byte):  6 bytes → precise ranking
            = 99% rejection before full ADC computation
  Tables:   6×256 = 1,536 floats = 6KB → fits in L1 cache
  Strategy: <100K: brute force Hamming on raw fingerprints
            100K-10M: CamPqScanOp::FullAdc (6 lookups/candidate)
            10M+: CamPqScanOp::Cascade (stroke 1→2→3)
            100M+: IVF probe → Cascade per partition

         │ ZeckF64: extract scent byte (byte 0 of 8-byte encoding)
         ▼

LAYER 4: ZECKF64 (Edge Codec)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Type:     u64 (8 bytes)
  Size:     8 bytes per edge
  ρ:        ~0.98 at full 8 bytes, ~0.94 at byte 0 only
  Location: ndarray/src/hpc/zeck.rs (CANONICAL + batch ops)
            lance-graph/blasgraph/zeckf64.rs (DUPLICATE)
            lance-graph/neighborhood/zeckf64.rs (DUPLICATE)
  Layout:
    BYTE 0 — THE SCENT (94% precision alone):
      bit 7: sign (causality direction)
      bit 6: SPO close? (all three)
      bit 5: _PO close? (predicate + object)
      bit 4: S_O close? (subject + object)
      bit 3: SP_ close? (subject + predicate)
      bit 2: __O close? (object only)
      bit 1: _P_ close? (predicate only)
      bit 0: S__ close? (subject only)
    BYTES 1-7: distance quantiles per band (0-255 each)
  Boolean lattice: SP_ close ⟹ S__ close AND _P_ close
    19 of 128 patterns are legal → 85% built-in error detection
  Progressive reading:
    Read byte 0:     ~94% precision, 1 byte/edge
    Read bytes 0-1:  ~96% precision, 2 bytes/edge
    Read bytes 0-7:  ~98% precision, 8 bytes/edge

         │ Extract byte 0 only
         ▼

LAYER 5: SCENT BYTE (Ultimate Compression)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Type:     u8
  Size:     1 byte per edge
  ρ:        0.937 (matches Pareto 8-bit level exactly)
  Compress: 6,144× from full SPO plane
  Location: Used as edge weight in ScentCsr
            Stored in neighborhoods.lance: FixedSizeBinary(10000)
  NOTE:     NOT metric-safe for CAKES pruning (boolean lattice, not distance)
            Good as heuristic pre-filter only (HHTL Layer 0)
```

## 3. The HHTL Search Cascade (How Codecs Compose)

```
Query arrives → HHTL 4-stage progressive loading:

STAGE      CODEC           BYTES/NODE   ρ       METRIC-SAFE?   WHAT IT DOES
─────────────────────────────────────────────────────────────────────────────
HEEL       Scent (ZeckF64  1            0.937   NO             Heuristic pre-filter
           byte 0)                                              Reject 95% of nodes

HIP        PaletteEdge     3            0.965   YES (for CAKES) CLAM ball-tree pruning
           (Base17 palette)                                     ArchetypeTree O(k²)

TWIG       Base17           102 (SPO)   0.992   YES            Full L1 distance
           (i16[17] × 3)                                       Precise ranking

LEAF       Full planes      6,144 (SPO) 1.000   YES            Ground truth
           (Fingerprint<256>×3)                                Verify top-K
```

**95%+ of searches terminate at HEEL+HIP (4 bytes per candidate).**

## 4. The No-NaN Constraint

All codecs in this stack are **integer-only** or **fixed-point**:

| Codec | Type | NaN possible? | Why |
|-------|------|--------------|-----|
| Fingerprint | u64 words | NO | Binary, XOR algebra |
| ZeckBF17 | i16 + u8 | NO | Fixed-point (scale=256) |
| Base17 | i16 | NO | Fixed-point integer |
| PaletteEdge | u8 indices | NO | Index into codebook |
| CAM-PQ | u8 codes | NO | Index into codebook |
| ZeckF64 | u8 bytes in u64 | NO | Integer quantiles |
| Scent | u8 | NO | Boolean lattice byte |

**The only floats in the system are:**
- CAM-PQ distance tables (6×256 f32, precomputed once per query)
- BF16 truth values (bf16_truth.rs — explicit NaN handling via BF16Weights)
- NARS truth frequency/confidence (f32, clamped to [0,1])

This is the "no NaN Fujitsu x sensor semantic Zeckendorf codec" — every hot-path computation is integer arithmetic, SIMD-friendly, branch-free.

## 5. Codec Decision Matrix

| Use Case | Codec | Why |
|----------|-------|-----|
| Cold storage (Lance files) | Full planes (16Kbit) | Lossless reference |
| Hot search (BindSpace) | Full planes + scent pre-filter | L1 cache resident per node |
| Batch scan (>1M nodes) | CAM-PQ 6-byte | 500M candidates/sec via AVX-512 |
| Graph traversal (BFS/PageRank) | ScentCsr (1 byte/edge) | Minimal memory, 94% ρ |
| Scope search (<10K nodes) | PaletteCsr + Base17 | O(k²) with k=256 archetypes |
| Edge encoding (SPO triples) | ZeckF64 (8 byte/edge) | Progressive precision |
| Research / offline analysis | ZeckBF17 (48 byte/plane) | Best precision per byte |
| Cross-system transfer | Base17 (34 byte/plane) | Small, algebraic, lossless-ish |

## 6. Compression Ratio Summary Table

| From → To | Ratio | Bytes | ρ | Method |
|-----------|-------|-------|---|--------|
| Full → Full | 1:1 | 6,144 (SPO) | 1.000 | Identity |
| Full → ZeckBF17 | 42:1 | 144 (SPO) | 0.982 | Octave averaging |
| Full → Base17 | 60:1 | 102 (SPO) | 0.965 | Drop envelope |
| Full → PaletteEdge | 2,048:1 | 3 | 0.937 | k-means palette |
| Full → CAM-PQ | 170-682:1 | 6 | varies | PQ codebook |
| Full → ZeckF64 | 768:1 | 8 (edge) | 0.98 | Progressive bands |
| Full → Scent | 6,144:1 | 1 (edge) | 0.937 | Byte 0 of ZeckF64 |

## 7. Codec Location Cross-Reference

| Codec | ndarray | lance-graph | bgz17 | codec-research | ladybug-rs |
|-------|---------|-------------|-------|---------------|------------|
| Fingerprint<256> | `hpc/fingerprint.rs` | `blasgraph/types.rs` | `container.rs` | — | `bind_space.rs` |
| ZeckF64 | `hpc/zeck.rs` | `blasgraph/zeckf64.rs` + `neighborhood/zeckf64.rs` | — | — | — |
| Base17 | `hpc/bgz17_bridge.rs` | — | `base17.rs` | — | — |
| ZeckBF17 | — | — | — | `zeckbf17.rs` | — |
| PaletteEdge | `hpc/bgz17_bridge.rs` | — | `palette.rs` | — | — |
| PaletteCsr | — | — | `palette_csr.rs` | — | — |
| CAM-PQ | `hpc/cam_pq.rs` | `planner/physical/cam_pq_scan.rs` | — | — | — |
| ScentCsr | — | `blasgraph/neighborhood_csr.rs` | — | — | — |
| HDR Cascade | `hpc/cascade.rs` | `blasgraph/cascade_ops.rs` | — | — | `search/hdr_cascade.rs` |
| SimilarityTable | — | — | `(NEW) bgz17/similarity_table.rs` | — | — |
