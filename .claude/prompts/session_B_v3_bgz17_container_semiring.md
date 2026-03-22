# SESSION B v3: bgz17 Container Annex + Palette Semiring + SIMD

## STATUS: ⚡ PARTIAL

**DONE (PR #28, container.rs, 728 lines, 15 tests):**
Annex pack/unpack (W112-125), wide checksum (W254), SPO crystal (W128-143),
extended edges (W224-239), seal_wide_meta, has_bgz17_annex. Skip Deliverable 1.

**REMAINING:** Deliverables 2-7 below.

## CONTEXT

**Repo:** `AdaWorldAPI/lance-graph` branch from `main`
**Crate:** `crates/bgz17/` (3,743 lines, 13 modules, 61 tests)
**Depends on:** Session A's TypedGraph (for TypedPaletteGraph conversion)

The node container is `[u64; 256]` — structured metadata, NOT a flat fingerprint.
Base17 encodes from S/P/O PLANES (flat 2KB each), not from the container.
The Base17 result + palette indices are STORED in container W112-125.
The Cascade's stride-16 sampling automatically hits W112 (first Base17 word).

## READ FIRST

```bash
cat crates/bgz17/KNOWLEDGE.md
cat /mnt/user-data/outputs/bgz17_container_mapping.md   # Word-by-word mapping

# Understand what we write INTO
cat crates/lance-graph/src/graph/blasgraph/columnar.rs   # Container schemas
cat crates/lance-graph/src/graph/neighborhood/storage.rs # Lance table schemas

# Understand what we preserve
cat crates/lance-graph/src/graph/spo/truth.rs            # TruthValue (W4-7)
cat crates/lance-graph/src/graph/spo/merkle.rs           # MerkleRoot (W126-127)
cat crates/lance-graph/src/graph/spo/builder.rs          # SpoRecord: packed bitmap
cat crates/lance-graph/src/graph/blasgraph/hdr.rs        # Cascade (stride-16)
```

## DELIVERABLE 1: Container Annex Writer (new: container_annex.rs)

Write/read bgz17 data into the reserved words of the existing container.

```rust
/// Container word offsets for bgz17 data.
pub const W_BASE17_START: usize = 112;  // W112-124: Base17 S+P+O (102 bytes)
pub const W_PALETTE: usize = 125;       // W125: palette_s/p/o + temporal_q
pub const W_PALETTE_CSR_START: usize = 96;  // W96-111: local palette CSR
pub const W_PALETTE_NEIGHBORS: usize = 176; // W176-191: palette neighbor indices (WIDE)

/// Write Base17 + palette indices into container reserved words.
/// Updates W112-125. Does NOT touch W0-111 or W126-127.
/// Caller must recompute W126 checksum after this call.
pub fn write_bgz17_annex(
    container: &mut [u64; 256],
    base17: &SpoBase17,
    palette: &PaletteEdge,
    temporal_quantile: u8,
) { ... }

/// Read Base17 + palette indices from container.
pub fn read_bgz17_annex(
    container: &[u64; 256],
) -> (SpoBase17, PaletteEdge, u8) { ... }

/// Write local palette CSR into W96-111.
/// Maps W16-31 inline edges through palette assignments.
/// Each entry: verb_palette(8) + target_palette(8) = 16 bits.
/// 16 words × 4 entries/word = 64 palette-indexed edges.
pub fn write_palette_csr(
    container: &mut [u64; 256],
    inline_edges: &[(u8, u8)],      // verb + target from W16-31
    assignments: &[u8],              // node → palette archetype
) { ... }

/// Write palette neighbor indices into W176-191 (WIDE container only).
/// Top-128 neighbors by palette archetype index.
pub fn write_palette_neighbors(
    container: &mut [u64; 256],
    neighbor_palette_indices: &[u8],  // up to 128 entries
) { ... }
```

**CRITICAL:** The checksum at W126 must be recomputed after writing the annex.
Read the existing checksum algorithm from spo/merkle.rs — it's XOR-fold rotate-multiply.

## DELIVERABLE 2: PaletteSemiring (new: palette_semiring.rs)

The 256×256 distance matrix defines a semiring over palette indices.

```rust
pub struct PaletteSemiring {
    pub distance_matrix: DistanceMatrix,
    /// compose[a * 256 + b] = palette index of path(a → b).
    pub compose_table: Vec<u8>,  // 256 × 256 = 64KB
}
```

The compose table is built from palette codebooks via Base17 XOR bind:
```rust
pub fn build_compose(pal: &Palette) -> Vec<u8> {
    let mut table = vec![0u8; 256 * 256];
    for a in 0..pal.len() {
        for b in 0..pal.len() {
            let composed = pal.entries[a].xor_bind(&pal.entries[b]);
            table[a * 256 + b] = pal.nearest(&composed);
        }
    }
    table
}
```

Required:
- Implement trait compatible with blasgraph Semiring (Session A's TypedGraph)
- `palette_mxm(a: &PaletteMatrix, b: &PaletteMatrix, compose: &[u8]) -> PaletteMatrix`
- compose(a, identity) = a (identity test)
- compose is associative

## DELIVERABLE 3: PaletteMatrix (new: palette_matrix.rs)

Sparse matrix of 3-byte `PaletteEdge` entries (not 2KB BitVec).

```rust
pub struct PaletteMatrix {
    pub nrows: usize, pub ncols: usize,
    pub row_ptr: Vec<usize>, pub col_idx: Vec<usize>,
    pub vals: Vec<PaletteEdge>,  // 3 bytes each
}
```

Required:
- `from_typed_graph(graph: &TypedGraph, palettes: &SpoPalettes) -> Self`
- `mxm(a, b, compose_table) -> PaletteMatrix`
- `to_distance_csr(dm: &SpoDistanceMatrices) -> ScalarCsr`

## DELIVERABLE 4: PaletteCsr (new: palette_csr.rs)

The 256×256 distance matrix reinterpreted as compressed scope CSR.
Seeds from container W16-31 inline edges mapped through palette assignments.

```rust
pub struct PaletteCsr {
    pub distances: SpoDistanceMatrices,
    pub assignments: Vec<u8>,                // node → archetype
    pub archetype_members: Vec<Vec<usize>>,  // archetype → node list
    pub edge_topology: Vec<Vec<(u8, u8)>>,   // per-archetype: [(verb_pal, target_pal)]
    pub k: usize,
}

impl PaletteCsr {
    /// Build from scope + inline edges.
    /// Reads W16-31 from each node's container for graph topology.
    pub fn from_scope_with_edges(
        scope: &Bgz17Scope,
        containers: &[[u64; 256]],  // access to W16-31 per node
    ) -> Self;

    /// CLAM tree on 256 archetypes: O(256²) not O(N²).
    pub fn build_archetype_tree(&self) -> ArchetypeTree;

    /// Search merging topology (inline edges) and geometry (distance matrix).
    pub fn search(&self, query: &SpoBase17, k: usize) -> Vec<(usize, u32)>;
}
```

## DELIVERABLE 5: Base17 VSA Operations (base17.rs additions)

```rust
impl Base17 {
    /// XOR bind: path composition in hyperdimensional space.
    pub fn xor_bind(&self, other: &Base17) -> Base17;

    /// Bundle: majority vote (set union).
    pub fn bundle(patterns: &[&Base17]) -> Base17;

    /// Permute: cyclic shift (sequence encoding).
    pub fn permute(&self, shift: usize) -> Base17;
}
```

These make Base17 a complete VSA algebra matching BitVec XOR/majority/rotate.

## DELIVERABLE 6: SIMD Batch Palette Distance (new: simd.rs)

AVX-512 VGATHERDPS for 16 matrix lookups per instruction.

```rust
pub fn batch_palette_distance(dm: &[u16], query: u8, candidates: &[u8], out: &mut [u16]) {
    match detect_simd() {
        SimdLevel::Avx512 => unsafe { avx512_gather(...) },
        SimdLevel::Avx2 => unsafe { avx2_gather(...) },
        SimdLevel::Scalar => scalar_lookup(...),
    }
}
```

Plus `batch_spo_distance`: 3× gather (S+P+O), sum results.
Plus software prefetch hints for matrix rows.

## DELIVERABLE 7: PaletteResolution Auto-Select

```rust
pub enum PaletteResolution {
    Full256,    // 128KB, ρ=0.992
    Half128,    // 32KB, ρ=0.965
    Quarter64,  // 8KB, ρ=0.738
}

impl PaletteResolution {
    pub fn auto_select(edge_count: usize) -> Self;
}
```

## TESTS

1. Container annex roundtrip: write→read→compare Base17 + palette
2. W126 checksum covers annex (corrupt W112, checksum detects)
3. compose_table self-consistency: compose(a, identity) = a
4. PaletteMatrix mxm: 2-hop matches manual computation
5. PaletteCsr from_scope_with_edges: reads W16-31 correctly
6. xor_bind is its own inverse: a XOR b XOR b = a
7. SIMD batch matches scalar for all inputs
8. TypedPaletteGraph KNOWS² ranks same as BitVec (ρ > 0.9)

## OUTPUT

Branch: `feat/bgz17-container-semiring`
Run: `cd crates/bgz17 && cargo test -- --nocapture`
