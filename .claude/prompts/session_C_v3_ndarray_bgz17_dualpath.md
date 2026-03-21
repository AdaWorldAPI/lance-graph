# SESSION C v3: ndarray ← bgz17 Dual-Path + TruthGate + Cascade

## CONTEXT

**Repo:** `AdaWorldAPI/lance-graph` branch from `main`
**Crates:** `bgz17/` + `lance-graph/src/graph/blasgraph/ndarray_bridge.rs`
**Depends on:** Session A (TypedGraph, CSC), Session B (container annex, PaletteCsr, SIMD)

Two parallel hot paths share the same 256×256 distance matrix.
Results carry TruthValue from W4-7. TruthGate filters AFTER distance.
The Cascade (hdr.rs) automatically benefits from Base17 at W112 via stride-16.

## READ FIRST

```bash
cat crates/lance-graph/src/graph/blasgraph/ndarray_bridge.rs  # NdarrayFingerprint
cat crates/lance-graph/src/graph/neighborhood/clam.rs         # CLAM tree
cat crates/lance-graph/src/graph/neighborhood/search.rs       # k-NN search
cat crates/lance-graph/src/graph/blasgraph/hdr.rs             # Cascade (1467 lines)
cat crates/lance-graph/src/graph/spo/truth.rs                 # TruthValue, TruthGate
cat crates/bgz17/src/bridge.rs                                # Bgz17Distance trait
cat crates/bgz17/src/clam_bridge.rs                           # ClamTree wiring
cat crates/bgz17/src/prefetch.rs                              # Prefetch pipeline
```

## DELIVERABLE 1: bgz17 as lance-graph dependency

```toml
# In crates/lance-graph/Cargo.toml
[features]
bgz17-codec = ["bgz17"]
[dependencies]
bgz17 = { path = "../bgz17", optional = true }
```

Move bgz17 from workspace `exclude` to `members`.

## DELIVERABLE 2: NdarrayFingerprint ↔ Base17 Bridge (ndarray_bridge.rs)

Base17 encodes from PLANES (flat 2KB), not from the container.

```rust
#[cfg(feature = "bgz17-codec")]
impl NdarrayFingerprint {
    /// Convert flat 16384-bit fingerprint PLANE to Base17.
    /// The plane is a flat signal — golden-step octave averaging applies.
    /// Do NOT call this on the container (which has typed fields at known offsets).
    pub fn plane_to_base17(&self) -> bgz17::base17::Base17 {
        let mut acc = vec![0i8; 16384];
        for w in 0..256 {
            for bit in 0..64 {
                acc[w * 64 + bit] = if (self.words[w] >> bit) & 1 == 1 { 1 } else { -1 };
            }
        }
        bgz17::base17::Base17::encode(&acc)
    }

    /// Convert three flat PLANES to SpoBase17.
    pub fn planes_to_spo(s: &Self, p: &Self, o: &Self) -> bgz17::base17::SpoBase17 {
        bgz17::base17::SpoBase17 {
            subject: s.plane_to_base17(),
            predicate: p.plane_to_base17(),
            object: o.plane_to_base17(),
        }
    }
}
```

## DELIVERABLE 3: Layered DistanceFn (new: layered_distance.rs)

Single function pointer for CLAM/CAKES. Reads palette from W125 of container.

```rust
/// Build a distance function that reads palette indices from containers.
///
/// CLAM passes node indices. The function reads W125 from each node's
/// container for the palette indices, then does matrix lookup.
/// Metric-safe: palette L1 satisfies triangle inequality.
#[cfg(feature = "bgz17-codec")]
pub fn build_palette_distance_fn(
    containers: &[[u64; 256]],
    distance_matrices: &bgz17::distance_matrix::SpoDistanceMatrices,
) -> impl Fn(usize, usize) -> u64 {
    let dm = distance_matrices.clone();
    let pal_cache: Vec<(u8, u8, u8)> = containers.iter()
        .map(|c| {
            let w = c[125]; // W125: palette_s | palette_p | palette_o
            (w as u8, (w >> 8) as u8, (w >> 16) as u8)
        })
        .collect();

    move |a: usize, b: usize| -> u64 {
        let (as_, ap, ao) = pal_cache[a];
        let (bs, bp, bo) = pal_cache[b];
        dm.spo_distance(as_, ap, ao, bs, bp, bo) as u64
    }
}
```

## DELIVERABLE 4: CLAM Tree Build with Palette (clam.rs modification)

```rust
#[cfg(feature = "bgz17-codec")]
impl ClamTree {
    /// Build CLAM tree using palette distance from containers.
    /// Reads W125 from each container for palette indices.
    /// O(1) distance via matrix lookup instead of O(16K) Hamming.
    pub fn build_from_containers(
        containers: &[[u64; 256]],
        dm: &bgz17::distance_matrix::SpoDistanceMatrices,
        max_leaf_size: usize,
    ) -> Self {
        let dist_fn = build_palette_distance_fn(containers, dm);
        Self::build_with_fn(containers.len(), max_leaf_size, dist_fn)
    }
}
```

## DELIVERABLE 5: Parallel Search with TruthGate (new: parallel_search.rs)

Both paths run simultaneously. Results carry TruthValue. TruthGate filters.

```rust
/// Parallel search: HHTL + CLAM, merged, TruthGate filtered.
///
/// HHTL reads W125 palette indices from neighborhoods.lance (sequential).
/// CLAM reads W125 from containers via palette_distance_fn (tree pruning).
/// Results merged, re-ranked with Base17 L1 (W112-124).
/// TruthGate reads W4-7 from container for final filtering.
#[cfg(feature = "bgz17-codec")]
pub fn parallel_search(
    scope: &bgz17::layered::LayeredScope,
    palette_csr: &bgz17::palette_csr::PaletteCsr,  // from Session B
    containers: &[[u64; 256]],
    query: &bgz17::base17::SpoBase17,
    k: usize,
    gate: TruthGate,
) -> Vec<SearchResult> {
    // Path 1: HHTL (sequential, reads palette from scope)
    let hhtl = scope.search(/* scent, palette, base, limits */);

    // Path 2: CLAM on archetype tree (tree-based, reads palette from containers)
    let clam = palette_csr.search(query, k * 2);

    // Merge: union, deduplicate, re-rank with Base17 L1
    let merged = merge_and_rerank(hhtl, clam, query, k * 2);

    // TruthGate filter: read W4-7 from each candidate's container
    merged.into_iter()
        .filter_map(|(pos, dist)| {
            let truth = read_container_truth(containers, pos);  // W4-7
            if gate.passes(&truth) {
                Some(SearchResult { position: pos, distance: dist, truth })
            } else {
                None
            }
        })
        .take(k)
        .collect()
}

/// Read NARS TruthValue from container W4-7.
fn read_container_truth(containers: &[[u64; 256]], pos: usize) -> TruthValue {
    let w4 = containers[pos][4];
    let freq = f32::from_bits(w4 as u32);
    let conf = f32::from_bits((w4 >> 32) as u32);
    TruthValue::new(freq, conf)
}

pub struct SearchResult {
    pub position: usize,
    pub distance: u32,
    pub truth: TruthValue,
}
```

## DELIVERABLE 6: Cascade Auto-Benefit Verification

The Cascade at stride-16 now hits W112 (Base17 word 0).
No code change needed. Just verify the benefit empirically:

```rust
#[test]
fn test_cascade_benefits_from_base17_annex() {
    // Build 1000 containers with and without Base17 at W112
    // Run Cascade::query() stage 1 (stride-16) on both
    // Measure: containers WITH Base17 should have better
    // discrimination (more candidates rejected at stage 1)

    let mut containers_empty: Vec<[u64; 256]> = vec![[0u64; 256]; 1000];
    let mut containers_filled: Vec<[u64; 256]> = vec![[0u64; 256]; 1000];

    // Fill W0-95 identically in both (random metadata)
    // Fill W112-125 in containers_filled only (Base17 + palette)

    // Run Cascade stage 1 on both sets
    // Compare rejection rates: filled should reject more at stage 1
}
```

## DELIVERABLE 7: CHAODA LFD from Palette

```rust
/// Compute LFD using palette distances (from containers, not full planes).
#[cfg(feature = "bgz17-codec")]
pub fn lfd_from_palette(parent_radius: u16, child_radii: &[u16]) -> f32 {
    let n = child_radii.len() as f32;
    if n < 2.0 || parent_radius == 0 { return 1.0; }
    let avg = child_radii.iter().map(|&r| r as f32).sum::<f32>() / n;
    if avg < 1.0 { return 1.0; }
    n.ln() / (parent_radius as f32 / avg).ln()
}
```

Feed into `generative.rs` for Bayesian distance correction (arXiv:2602.03505).

## DELIVERABLE 8: SIMD Dispatch (ndarray_bridge.rs addition)

```rust
pub enum SimdLevel { Scalar, Avx2, Avx512 }
pub fn detect_simd() -> SimdLevel;
pub fn batch_palette_distance(dm: &[u16], query: u8, candidates: &[u8], out: &mut [u16]);
```

Runtime detection, fallback chain. Used by parallel_search for batch scoring.

## TESTS

1. plane_to_base17: roundtrip sign-bit fidelity > 0.5 (NOT container, PLANE)
2. build_palette_distance_fn: reads W125 correctly, self-distance = 0
3. CLAM tree from containers: valid tree (all nodes reachable)
4. parallel_search: HHTL + CLAM merged results ⊇ either alone
5. TruthGate integration: STRONG gate filters low-confidence candidates
6. Cascade stage-1 discrimination improves with Base17 at W112
7. LFD from palette: reasonable values (1.0-10.0)
8. SIMD dispatch: correct level detected, batch matches scalar

## OUTPUT

Branch: `feat/ndarray-bgz17-dualpath`
Run: `cargo test --features bgz17-codec -- --nocapture`
