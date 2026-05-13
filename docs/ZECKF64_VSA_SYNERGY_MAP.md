# ZeckF64 / Fibonacci-VSA / Neighborhood Synergy Map

> *Every variant, every difference, every synergy. Do not delete any file until this map says why.*

## 1. THE FIVE ZECKF64-FAMILY IMPLEMENTATIONS

### 1A. neighborhood/zeckf64.rs (443 lines, 12 tests)
**File**: `crates/lance-graph/src/graph/neighborhood/zeckf64.rs`
**Role**: The richest implementation. Graph topology codec for SPO edge encoding.

```rust
// Input: BitVec triples (16,384-bit)
pub fn zeckf64(a: (&BitVec, &BitVec, &BitVec), b: (&BitVec, &BitVec, &BitVec)) -> u64
pub fn zeckf64_from_distances(ds: u32, dp: u32, d_o: u32) -> u64

// Extraction
pub fn scent(edge: u64) -> u8           // byte 0
pub fn resolution(edge: u64, byte_n: u8) -> u8  // bytes 1-7
pub fn get_sign(edge: u64) -> bool      // bit 7 (causality)
pub fn set_sign(edge: u64, sign: bool) -> u64

// Distance metrics (UNIQUE to this file)
pub fn zeckf64_distance(a: u64, b: u64) -> u32           // L1 on all 8 bytes
pub fn zeckf64_scent_distance(a: u64, b: u64) -> u32     // L1 on byte 0 only
pub fn zeckf64_scent_hamming_distance(a: u64, b: u64) -> u32  // Hamming on byte 0
pub fn zeckf64_progressive_distance(a: u64, b: u64, n: u8) -> u32  // L1 on bytes 0..=n

// Boolean lattice validation (UNIQUE to this file)
pub fn is_legal_scent(byte0: u8) -> bool
pub fn count_legal_patterns() -> usize   // returns 19
```

**Unique features NOT in any other copy**:
- `zeckf64_distance()` — L1 across all 8 bytes
- `zeckf64_scent_distance()` — L1 on byte 0 (amplitude-sensitive, range 0-255)
- `zeckf64_scent_hamming_distance()` — Hamming on byte 0 (bit-pattern-sensitive, range 0-8)
- `zeckf64_progressive_distance()` — L1 on bytes 0..=n (progressive precision)
- `is_legal_scent()` — validates boolean lattice (19 of 128 patterns)
- `count_legal_patterns()` — proves there are exactly 19
- `set_sign()` / `get_sign()` — causality direction bit
- 12 tests including lattice legality, symmetry, progressive monotonicity

### 1B. blasgraph/zeckf64.rs (~100 lines, referenced in tests)
**File**: `crates/lance-graph/src/graph/blasgraph/zeckf64.rs`
**Role**: Encoding-only. Takes BitVec triples or pre-computed distances.

```rust
pub fn zeckf64(a: (&BitVec, &BitVec, &BitVec), b: (&BitVec, &BitVec, &BitVec), sign: bool, threshold: u32) -> u64
pub fn zeckf64_from_distances(ds: u32, dp: u32, d_o: u32) -> u64
```

**Key differences from neighborhood version**:
- Takes explicit `sign` parameter (neighborhood defaults to 0)
- Takes explicit `threshold` parameter (neighborhood uses const `D_MAX/2`)
- **No distance metrics** (no `zeckf64_distance`, no `zeckf64_scent_distance`)
- **No lattice validation** (no `is_legal_scent`)
- **No progressive reading** (no `zeckf64_progressive_distance`)
- Stricter compound validation in some paths

### 1C. ndarray/hpc/zeck.rs (~200 lines)
**File**: `/home/user/ndarray/src/hpc/zeck.rs`
**Role**: Ported from neighborhood. Adds batch + top-k ops.

```rust
pub fn zeckf64_from_distances(ds: u32, dp: u32, d_o: u32) -> u64

// UNIQUE to ndarray version:
// - Batch operations (zeckf64_batch)
// - Top-k operations
// - Integration with ndarray::hpc::bitwise::hamming_distance_raw
```

**Key differences**:
- Header says "Ported from lance-graph/crates/lance-graph/src/graph/neighborhood/zeckf64.rs"
- Has `zeckf64_from_distances()` only (not the full BitVec version)
- Adds batch + top-k (not in either lance-graph version)
- **Missing**: all distance metrics, lattice validation, sign ops, progressive reading

### 1D. deepnsm/spo.rs — The 36-bit Cousin
**File**: `crates/deepnsm/src/spo.rs`
**Role**: NOT ZeckF64 but structurally parallel. 36-bit SPO triple encoding.

```rust
pub struct SpoTriple { packed: u64 }  // [S:12][P:12][O:12] = 36 bits
pub fn distance(&self, other: &SpoTriple, matrix: &WordDistanceMatrix) -> u32
pub fn distance_per_role(&self, other, matrix) -> (u8, u8, u8)
pub fn similarity(&self, other, matrix, table) -> f32
```

**Relationship to ZeckF64**:
- Same SPO triple concept but different encoding
- ZeckF64: encodes **edge distance** between two nodes (8 bytes, progressive)
- SpoTriple: encodes **the triple itself** (36 bits, vocabulary ranks)
- ZeckF64 distance = Hamming on 16Kbit planes → quantile bytes
- SpoTriple distance = lookup in 4096² u8 matrix → CDF calibrated similarity
- **Synergy**: SpoTriple could be the SOURCE for ZeckF64 encoding in the semantic domain

### 1E. codec-research/zeckbf17.rs — The Research Codec
**File**: `crates/lance-graph-codec-research/src/zeckbf17.rs`
**Role**: Golden-step octave compression research. Not ZeckF64 encoding but uses same naming.

```
i8[16384] per plane → i16[17] base + u8[14] envelope = 48 bytes
Compression: 341:1 per plane
```

**Relationship to ZeckF64**:
- ZeckBF17 compresses **fingerprint planes** (16Kbit → 48 bytes)
- ZeckF64 compresses **edge relationships** (two planes → 8 bytes)
- The "Zeck" prefix comes from Zeckendorf representation concept
- ZeckBF17 uses golden-step octave averaging; ZeckF64 uses boolean lattice bands
- Both are progressive: ZeckBF17 has base+envelope, ZeckF64 has byte 0..7

## 2. THE THREE VSA SYSTEMS

### 2A. ndarray/hpc/vsa.rs — 10,000-bit Working Memory
**Dimensions**: 10,000 bits (157 u64 words, 1,250 bytes)
**Domain**: General-purpose VSA for cognitive working memory

```rust
pub struct VsaVector { pub words: [u64; 157] }
pub struct VsaAccumulator { pub values: Vec<i16> }

// Operations:
bind/unbind:  XOR (self-inverse, O(n))
bundle:       majority vote via i16 accumulator
clean:        iterative similarity search against codebook
permute:      cyclic shift for sequence encoding
```

**Design choice**: 10,000 bits because Johnson-Lindenstrauss bound for 4096 items at ε=0.1 requires ~10,000 dimensions. The theoretical minimum for near-orthogonality.

### 2B. deepnsm/encoder.rs — 512-bit Semantic Encoding
**Dimensions**: 512 bits (8 u64 words, 64 bytes)
**Domain**: Natural language sentence encoding for 4,096-word vocabulary

```rust
pub struct VsaVec { data: [u64; 8] }  // 512 bits
pub struct RoleVectors { ... }         // 6 fixed role vectors (S, P, O, NEG_S, NEG_P, NEG_O)

// Operations:
bind:     XOR (word ⊕ role → role-tagged representation)
bundle:   majority vote (superposition of bindings → sentence vector)
unbind:   bundle ⊕ role → recover approximate word
permute:  NOT USED (roles are fixed, not positional)
```

**Design choice**: 512 bits because the vocabulary is only 4,096 words. JL bound for 4096 items at ε=0.1 ≈ 500 dimensions. 512 = 8 × 64 (cache-aligned).

**Word order sensitivity**: "dog bites man" ≠ "man bites dog" because `XOR(dog, ROLE_SUBJECT) ≠ XOR(dog, ROLE_OBJECT)`. Role vectors enforce asymmetry.

### 2C. ndarray/hpc/holo.rs — Phase-Space Holographic
**Dimensions**: Same as container (2048 bytes = 16,384 bits for CogRecord)
**Domain**: Phase-space operations with Fibonacci-spaced carriers

```rust
// NOT binary VSA — phase-space (each byte = angle 0-255 → 0°-360°)
pub fn phase_bind_i8(a: &[u8], b: &[u8]) -> Vec<u8>    // addition mod 256
pub fn phase_unbind_i8(a: &[u8], b: &[u8]) -> Vec<u8>   // subtraction mod 256
pub fn phase_inverse_i8(v: &[u8]) -> Vec<u8>             // 256 - v[i]

// Fibonacci-spaced carrier frequencies
// Carrier ops: encode/decode in spectral domain
// Focus ops: 3D spatial gating (8×8×32)
```

**Critical difference from binary VSA**:
- Binary VSA: XOR bind, Hamming distance, majority vote
- Phase VSA: modular addition bind, circular distance, weighted average
- Phase preserves **spatial locality** (nearby phases → nearby meanings)
- Binary XOR is **maximally decorrelated** (no locality preservation)

## 3. THE FIBONACCI CONNECTION

### 3A. Fibonacci in Zeckendorf Representation
The "Zeck" in ZeckF64 refers to the **Zeckendorf representation** — every positive integer can be uniquely represented as a sum of non-consecutive Fibonacci numbers. This is conceptual ancestry, not implementation detail:
- Fibonacci numbers define the "natural" precision levels
- Progressive reading (byte 0 → byte 0-7) mirrors progressive Fibonacci approximation
- The 19 legal scent patterns form a **monotone boolean lattice** similar to how Fibonacci representations form a subset of binary representations (no consecutive 1s)

### 3B. Fibonacci in Golden-Step Octave Traversal
The golden step = round(17/φ) = 11 where φ = (1+√5)/2 ≈ 1.618.
- gcd(11, 17) = 1 → visits all 17 residues (full coverage)
- **NOT** Fibonacci mod 17 (which only visits 13 of 17 residues — missing {6,7,10,11})
- The golden RATIO step is correct; the Fibonacci SEQUENCE step is wrong
- This was a documented correction in `codec-research/zeckbf17.rs`

### 3C. Fibonacci in Holographic Carriers
`ndarray/hpc/holo.rs` uses Fibonacci-spaced frequency encoding:
- Carrier frequencies at Fibonacci intervals (1, 1, 2, 3, 5, 8, 13, 21, ...)
- Spectral distance between carriers → natural harmonic spacing
- Connection to music theory: Fibonacci ratios approximate just intonation intervals

## 4. THE METRIC DIVERGENCE (Critical Finding)

The neighborhood/ and blasgraph/ cascade systems use the **same stage names** (HEEL/HIP/TWIG/LEAF) but **different distance metrics**:

| Stage | neighborhood/ | blasgraph/ | bgz-tensor/ |
|-------|--------------|------------|-------------|
| HEEL | **L1** on scent byte: `\|a - b\|`, range [0, 255] | **Hamming** on scent byte: `(a ^ b).count_ones()`, range [0, 8] | Plane agreement check |
| HIP | L1 on 2nd-hop scent | Hamming on 2nd-hop | Palette lookup |
| TWIG | L1 on 3rd-hop scent | Hamming on 3rd-hop | Base17 L1 distance |
| LEAF | Full BitVec Hamming | Full BitVec Hamming | f32 exact dot product |

**Why this matters**:
- L1 on scent (amplitude): treats 0b01111111 and 0b10000000 as distance 1 (close!)
- Hamming on scent (structure): treats 0b01111111 and 0b10000000 as distance 8 (maximally different!)
- The boolean lattice validation (`is_legal_scent`) only exists in neighborhood/
- CLAM ball-tree pruning requires a **metric** — L1 is a metric, Hamming is a metric, but they give **different pruning radii**

**neighborhood/zeckf64.rs has BOTH metrics**: `zeckf64_scent_distance()` (L1) and `zeckf64_scent_hamming_distance()` (Hamming). This is the only file that acknowledges both are valid.

## 5. COMPLETE CODEC FAMILY TREE

```
                        Zeckendorf Representation (conceptual)
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              ZeckF64          ZeckBF17         Golden-Step
           (edge codec)    (plane codec)    (dimension traversal)
                    │               │               │
         ┌─────────┼─────────┐     │         ┌─────┼─────┐
         │         │         │     │         │     │     │
   neighborhood  blasgraph  ndarray codec-   bgz17 ndarray bgz-
   /zeckf64.rs  /zeckf64   /zeck  research  base17 bridge  tensor
                                  /zeckbf17        bridge  proj

Binary VSA Family:
         ┌─────────────────────────────┐
         │                             │
   ndarray/vsa.rs              deepnsm/encoder.rs
   10,000-bit                  512-bit
   General purpose             Semantic (COCA 4096)
         │                             │
   ndarray/hdc.rs              deepnsm/codebook.rs
   XOR bind/bundle             CAM-PQ distance
   rotate/permute              4096² matrix

Phase VSA Family (DIFFERENT ALGEBRA):
         │
   ndarray/holo.rs
   Phase-space (mod 256)
   Fibonacci carriers
   Circular distance

Fingerprint Family:
   ndarray/fingerprint.rs    Fingerprint<256>  16,384-bit   CANONICAL
   lance-graph/types.rs      BitVec            16,384-bit   alias
   lance-graph/ndarray_bridge NdarrayFingerprint 16,384-bit  mirror
   ladybug-rs/bind_space.rs  [u64; 256]        16,384-bit   inline

Attention Family (NEW from PR 49):
   bgz-tensor/attention.rs   AttentionTable    256×256 u16   distance
   bgz-tensor/attention.rs   ComposeTable      256×256 u8    composition
   bgz-tensor/projection.rs  Base17            [i16; 17]     weight projection
```

## 6. SYNERGY MATRIX

| Feature | neighborhood/ | blasgraph/ | ndarray/zeck | deepnsm | bgz-tensor | holo |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| Encodes SPO edges | ✓ | ✓ | ✓ | — | — | — |
| Encodes SPO triples | — | — | — | ✓ | — | — |
| Encodes weight matrices | — | — | — | — | ✓ | — |
| Progressive precision | ✓ (byte 0→7) | ✓ (byte 0→7) | ✓ (byte 0→7) | — | ✓ (HHTL) | — |
| Boolean lattice validation | ✓ | — | — | — | — | — |
| L1 distance on encoded | ✓ | — | — | ✓ (CDF) | ✓ (Base17) | — |
| Hamming distance on encoded | ✓ | ✓ | — | — | — | — |
| Batch operations | — | — | ✓ | — | — | — |
| Top-k operations | — | — | ✓ | — | ✓ | — |
| Causality sign bit | ✓ | ✓ | — | — | — | — |
| Configurable threshold | — | ✓ | — | — | — | — |
| CDF calibrated similarity | — | — | — | ✓ | — | — |
| Compose table (multi-hop) | — | — | — | — | ✓ | — |
| XOR bind | — | — | — | ✓ | — | — |
| Phase bind (mod 256) | — | — | — | — | — | ✓ |
| Fibonacci carriers | — | — | — | — | — | ✓ |
| Golden-step octaves | — | — | — | — | ✓ | — |
| Zero dependencies | — | — | — | ✓ | ✓ | — |
| Tests | 12 | ref | port | 6 | 5 | 3 |

## 7. WHY EACH FILE EXISTS (Do Not Delete Rationale)

### neighborhood/zeckf64.rs — THE REFERENCE IMPLEMENTATION
- **Only file** with boolean lattice validation (`is_legal_scent`, `count_legal_patterns`)
- **Only file** with progressive distance metrics (`zeckf64_progressive_distance`)
- **Only file** with both L1 AND Hamming scent distance
- **Only file** with full sign bit API (`set_sign`, `get_sign`)
- **12 tests** covering lattice legality, symmetry, progressive monotonicity, boundary cases
- **Used by**: `neighborhood/search.rs` HHTL cascade (HEEL stage)
- **Verdict**: KEEP. This is the most complete implementation. Other versions are subsets.

### blasgraph/zeckf64.rs — THE PARAMETERIZED ENCODER
- **Only file** with configurable threshold and explicit sign parameter
- **Used by**: `blasgraph/` typed graph construction and SPO store integration
- Different API contract: caller provides sign and threshold, not baked in
- **Verdict**: KEEP. Different API surface for blasgraph layer. Could become a thin wrapper.

### ndarray/hpc/zeck.rs — THE BATCH ENGINE
- **Only file** with batch + top-k operations
- Integrates with ndarray's SIMD `hamming_distance_raw`
- **Used by**: ladybug-rs (via ndarray path dep) for high-throughput encoding
- **Verdict**: KEEP. Batch ops are unique. Could import core logic from neighborhood.

### deepnsm/spo.rs — THE SEMANTIC TRIPLE
- Different concept: encodes triples, not edges between triples
- 12-bit vocabulary ranks, not 16Kbit fingerprint distances
- 4096² distance matrix, not Hamming on bit planes
- **Verdict**: KEEP. Not a ZeckF64 variant — it's a different codec for a different domain.

### bgz-tensor/attention.rs — THE ATTENTION COMPILER
- Replaces transformer matmul with table lookup
- AttentionSemiring = distance table + compose table
- Uses Base17 projection, not ZeckF64 encoding
- **Verdict**: KEEP. Not a ZeckF64 variant — it's an attention codec.

## 8. INTEGRATION PATHS (What Could Be Unified)

### Path A: Core Logic Extraction
Extract the shared core (encoding + quantile) into a single function:
```rust
// Shared across ALL ZeckF64 variants:
fn zeckf64_core(ds: u32, dp: u32, d_o: u32, sign: u8) -> u64
```
Then each variant adds its unique surface:
- neighborhood: distance metrics, lattice validation, progressive reading
- blasgraph: configurable threshold, explicit sign
- ndarray: batch, top-k, SIMD integration

### Path B: HpcBackend Dispatch (from HPC_BACKEND_DISPATCH_PLAN.md)
```rust
// All call sites:
hpc().zeckf64_from_distances(ds, dp, d_o)  // core encoding
hpc().zeckf64_batch(&distances)            // batch (ndarray-only)

// Unique operations stay in their modules:
neighborhood::zeckf64::is_legal_scent(byte0)           // lattice validation
neighborhood::zeckf64::zeckf64_progressive_distance()  // progressive metrics
blasgraph::zeckf64::zeckf64(..., sign, threshold)      // parameterized API
```

### Path C: DeepNSM → ZeckF64 Bridge (NEW SYNERGY)
SpoTriple.distance_per_role() returns `(u8, u8, u8)` — three per-role distances.
These could feed directly into `zeckf64_from_distances(ds as u32, dp as u32, d_o as u32)`:
```rust
// Bridge: semantic triple distance → graph edge encoding
let (ds, dp, d_o) = triple_a.distance_per_role(&triple_b, &matrix);
let edge = zeckf64_from_distances(ds as u32 * 64, dp as u32 * 64, d_o as u32 * 64);
// Scale factor 64: maps u8 distance range [0,255] to ZeckF64 range [0,16384]
```
This would let DeepNSM semantic distances flow into the graph codec stack.

### Path D: Similarity Table as Universal Calibrator
`deepnsm/similarity.rs::SimilarityTable` (256-entry CDF lookup, 1KB) could calibrate
ANY distance metric — not just deepnsm's u8 matrix distances:
```rust
// Universal calibration:
let scent_sim = similarity_table.lookup(zeckf64_scent_distance(a, b));
let hamming_sim = similarity_table.lookup(hamming_distance(fp_a, fp_b) / 64);
let cam_sim = similarity_table.lookup(cam_pq_distance(tables, cam) as u32);
```
The CDF-based approach is **distribution-free** — it adapts to whatever distance distribution the data has, without assuming Gaussian or any parametric form.
