# HPC Backend Dispatch Plan: LazyLock, One Decision, Zero If-Then

> *Compilation status (2026-03-28): lance-graph compiles cleanly with AND without ndarray.*
> - `cargo check -p lance-graph` (default, ndarray-hpc ON): **OK** (4 warnings, 0 errors)
> - `cargo check -p lance-graph --no-default-features` (ndarray-hpc OFF): **OK** (4 warnings, 0 errors)
> - `cargo check -p lance-graph-planner -p lance-graph-contract`: **OK** (55 warnings, 0 errors)
> - Rust 1.94.0 pinned via `rust-toolchain.toml`

## 1. The Problem

ndarray is now an optional default dependency of lance-graph via the `ndarray-hpc` feature flag.
Without a dispatch pattern, every call site that differs between ndarray and standalone would need:

```rust
// BAD: army of if-then scattered across 50+ files
#[cfg(feature = "ndarray-hpc")]
fn hamming(a: &[u8], b: &[u8]) -> u64 {
    ndarray::hpc::bitwise::hamming_distance_raw(a, b)
}
#[cfg(not(feature = "ndarray-hpc"))]
fn hamming(a: &[u8], b: &[u8]) -> u64 {
    standalone_hamming(a, b) // duplicate implementation
}
```

This creates maintenance hell — every new operation needs two paths, and forgetting a `#[cfg]` is a silent bug.

## 2. The Solution: LazyLock Trait Dispatch

One trait. One `LazyLock`. One decision point at startup. Zero `#[cfg]` at call sites.

```rust
// crates/lance-graph/src/hpc_backend.rs

use std::sync::LazyLock;

/// HPC backend operations.
///
/// ndarray implements this with SIMD dispatch (avx512/avx2/sse4).
/// Standalone implements this with portable Rust.
///
/// Call sites use `hpc()` which returns a &'static reference — no branching.
pub trait HpcBackend: Send + Sync + 'static {
    // ─── Fingerprint ops ──────────────────────────
    /// Hamming distance between two byte slices.
    fn hamming_distance(&self, a: &[u8], b: &[u8]) -> u64;
    /// Population count of a byte slice.
    fn popcount(&self, data: &[u8]) -> u64;
    /// XOR two equal-length byte slices into `out`.
    fn xor_into(&self, a: &[u8], b: &[u8], out: &mut [u8]);

    // ─── CLAM tree ops ────────────────────────────
    /// Build a CLAM tree from a flat dataset.
    /// Returns (reorder_indices, cluster_offsets, cluster_radii, lfd_values).
    fn clam_build(
        &self,
        data: &[u8],
        vector_len: usize,
        count: usize,
        max_leaf: usize,
    ) -> ClamResult;
    /// ρ-nearest neighbor search in a CLAM tree.
    fn clam_rho_nn(
        &self,
        tree_data: &[u8],
        vector_len: usize,
        count: usize,
        query: &[u8],
        rho: f64,
        k: usize,
    ) -> Vec<(usize, u64)>;

    // ─── ZeckF64 ──────────────────────────────────
    /// Encode ZeckF64 from pre-computed Hamming distances.
    fn zeckf64_from_distances(&self, ds: u32, dp: u32, d_o: u32) -> u64;
    /// Batch ZeckF64 encode.
    fn zeckf64_batch(&self, distances: &[(u32, u32, u32)]) -> Vec<u64>;

    // ─── CAM-PQ ───────────────────────────────────
    /// Precompute distance tables for a query vector.
    fn cam_pq_precompute(&self, query: &[f32], codebook: &[Vec<Vec<f32>>]) -> [[f32; 256]; 6];
    /// ADC distance for one 6-byte CAM fingerprint.
    fn cam_pq_distance(&self, tables: &[[f32; 256]; 6], cam: &[u8; 6]) -> f32;
    /// Batch ADC distance for N candidates.
    fn cam_pq_distance_batch(&self, tables: &[[f32; 256]; 6], cams: &[[u8; 6]]) -> Vec<f32>;

    // ─── Sigma gate (statistical significance) ────
    /// Sigma gate check: is the distance statistically significant?
    fn sigma_gate_check(&self, distance: u64, mean: f64, std: f64) -> bool;

    // ─── Backend info ─────────────────────────────
    /// Backend name for diagnostics.
    fn name(&self) -> &'static str;
    /// Whether AVX-512 is available.
    fn has_avx512(&self) -> bool;
}

/// CLAM tree build result.
pub struct ClamResult {
    pub reorder: Vec<usize>,
    pub offsets: Vec<usize>,
    pub radii: Vec<u64>,
    pub lfd: Vec<f64>,
}

// ═══════════════════════════════════════════════════════════
// LazyLock singleton — ONE decision point, NEVER repeated
// ═══════════════════════════════════════════════════════════

/// Get the HPC backend. First call initializes; all subsequent calls are
/// a single pointer dereference with no branching.
///
/// ```rust
/// use lance_graph::hpc_backend::hpc;
///
/// let d = hpc().hamming_distance(&a, &b);
/// let p = hpc().popcount(&data);
/// let results = hpc().cam_pq_distance_batch(&tables, &cams);
/// ```
#[inline(always)]
pub fn hpc() -> &'static dyn HpcBackend {
    &*HPC_BACKEND
}

static HPC_BACKEND: LazyLock<Box<dyn HpcBackend>> = LazyLock::new(|| {
    #[cfg(feature = "ndarray-hpc")]
    {
        Box::new(NdarrayBackend)
    }
    #[cfg(not(feature = "ndarray-hpc"))]
    {
        Box::new(StandaloneBackend)
    }
});
```

## 3. The Two Implementations

### 3A. NdarrayBackend (feature = "ndarray-hpc")

```rust
// Only compiled when ndarray-hpc feature is active
#[cfg(feature = "ndarray-hpc")]
struct NdarrayBackend;

#[cfg(feature = "ndarray-hpc")]
impl HpcBackend for NdarrayBackend {
    fn hamming_distance(&self, a: &[u8], b: &[u8]) -> u64 {
        ndarray::hpc::bitwise::hamming_distance_raw(a, b)
    }
    fn popcount(&self, data: &[u8]) -> u64 {
        ndarray::hpc::bitwise::popcount_raw(data)
    }
    fn xor_into(&self, a: &[u8], b: &[u8], out: &mut [u8]) {
        ndarray::hpc::bitwise::xor_raw(a, b, out)
    }
    fn clam_build(&self, data: &[u8], vector_len: usize, count: usize, max_leaf: usize) -> ClamResult {
        let config = ndarray::hpc::clam::BuildConfig { max_leaf_size: max_leaf, ..Default::default() };
        let tree = ndarray::hpc::clam::ClamTree::build_with_config(data, vector_len, count, &config)
            .expect("CLAM build failed");
        ClamResult {
            reorder: tree.reorder_indices().to_vec(),
            offsets: tree.cluster_offsets().to_vec(),
            radii: tree.cluster_radii().to_vec(),
            lfd: tree.lfd_values().to_vec(),
        }
    }
    fn clam_rho_nn(&self, tree_data: &[u8], vector_len: usize, count: usize,
                    query: &[u8], rho: f64, k: usize) -> Vec<(usize, u64)> {
        ndarray::hpc::clam::rho_nn(tree_data, vector_len, count, query, rho, k)
    }
    fn zeckf64_from_distances(&self, ds: u32, dp: u32, d_o: u32) -> u64 {
        ndarray::hpc::zeck::zeckf64_from_distances(ds, dp, d_o)
    }
    fn zeckf64_batch(&self, distances: &[(u32, u32, u32)]) -> Vec<u64> {
        distances.iter().map(|&(ds, dp, d_o)|
            ndarray::hpc::zeck::zeckf64_from_distances(ds, dp, d_o)
        ).collect()
    }
    fn cam_pq_precompute(&self, query: &[f32], codebook: &[Vec<Vec<f32>>]) -> [[f32; 256]; 6] {
        // Delegate to ndarray CAM-PQ implementation
        ndarray::hpc::cam_pq::precompute_tables(query, codebook)
    }
    fn cam_pq_distance(&self, tables: &[[f32; 256]; 6], cam: &[u8; 6]) -> f32 {
        ndarray::hpc::cam_pq::adc_distance(tables, cam)
    }
    fn cam_pq_distance_batch(&self, tables: &[[f32; 256]; 6], cams: &[[u8; 6]]) -> Vec<f32> {
        ndarray::hpc::cam_pq::adc_distance_batch(tables, cams)
    }
    fn sigma_gate_check(&self, distance: u64, mean: f64, std: f64) -> bool {
        let z = (distance as f64 - mean) / std;
        z.abs() > 2.0 // 2-sigma threshold
    }
    fn name(&self) -> &'static str { "ndarray-hpc (AdaWorldAPI fork)" }
    fn has_avx512(&self) -> bool {
        ndarray::hpc::simd_caps::simd_caps().avx512f
    }
}
```

### 3B. StandaloneBackend (fallback, no deps)

```rust
#[cfg(not(feature = "ndarray-hpc"))]
struct StandaloneBackend;

#[cfg(not(feature = "ndarray-hpc"))]
impl HpcBackend for StandaloneBackend {
    fn hamming_distance(&self, a: &[u8], b: &[u8]) -> u64 {
        // Existing standalone implementation from ndarray_bridge.rs
        a.iter().zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones() as u64)
            .sum()
    }
    fn popcount(&self, data: &[u8]) -> u64 {
        data.iter().map(|&b| b.count_ones() as u64).sum()
    }
    fn xor_into(&self, a: &[u8], b: &[u8], out: &mut [u8]) {
        for i in 0..out.len().min(a.len()).min(b.len()) {
            out[i] = a[i] ^ b[i];
        }
    }
    // ... standalone CLAM, ZeckF64, CAM-PQ from existing code in blasgraph/
    fn name(&self) -> &'static str { "standalone (portable Rust)" }
    fn has_avx512(&self) -> bool { false }
}
```

## 4. Call Site Migration

### Before (hypothetical if-then army):
```rust
// graph/blasgraph/ndarray_bridge.rs — 80 lines of standalone popcount
// graph/blasgraph/zeckf64.rs — 120 lines of standalone ZeckF64
// graph/neighborhood/zeckf64.rs — EXACT DUPLICATE
// Every call site: #[cfg(feature = "ndarray-hpc")] { ... } else { ... }
```

### After (one function call):
```rust
use crate::hpc_backend::hpc;

// Anywhere in lance-graph:
let distance = hpc().hamming_distance(&a_bytes, &b_bytes);
let popcount = hpc().popcount(&fingerprint_bytes);
let zeck = hpc().zeckf64_from_distances(ds, dp, d_o);
let tables = hpc().cam_pq_precompute(&query, &codebook);
let results = hpc().cam_pq_distance_batch(&tables, &cam_data);
let tree = hpc().clam_build(&data, vec_len, count, 32);

// No #[cfg], no if-then, no branching, no duplication.
// hpc() is &'static — one pointer deref, same as a global variable.
```

## 5. Migration Steps

### Step 1: Create `hpc_backend.rs` (1 file)
- Define `HpcBackend` trait
- Define `LazyLock<Box<dyn HpcBackend>>` singleton
- Implement `NdarrayBackend` (delegates to `ndarray::hpc::*`)
- Implement `StandaloneBackend` (extracts from existing `ndarray_bridge.rs`)

### Step 2: Wire existing call sites (~10 files)
- `graph/blasgraph/ndarray_bridge.rs` → standalone impl moves into `StandaloneBackend`
- `graph/blasgraph/zeckf64.rs` → `hpc().zeckf64_from_distances()`
- `graph/neighborhood/zeckf64.rs` → **DELETE** (duplicate, replaced by `hpc()`)
- Any future code just calls `hpc().method()`

### Step 3: Wire existing consumers in ladybug-rs
- `ladybug-rs/src/graph/spo/store.rs:905` currently calls `ndarray::hpc::clam::ClamTree` directly
- After: `lance_graph::hpc_backend::hpc().clam_build(...)` — same function, but through the spine
- OR: ladybug-rs keeps its direct ndarray dep (it's already a path dep) and lance-graph also has it

### Step 4: Future CAM-PQ UDF wiring
- `lance-graph/src/cam_pq/udf.rs` → `hpc().cam_pq_precompute()` + `hpc().cam_pq_distance_batch()`
- DataFusion UDF calls `hpc()` — gets ndarray SIMD or standalone, transparently

## 6. Performance Characteristics

| Operation | ndarray Backend | Standalone Backend | Ratio |
|-----------|----------------|-------------------|-------|
| `hamming_distance(2KB)` | ~50ns (AVX-512 VPOPCNTDQ) | ~2μs (scalar popcount) | 40× |
| `popcount(2KB)` | ~25ns (AVX-512) | ~1μs (scalar) | 40× |
| `cam_pq_distance_batch(1M)` | ~2ms (VPGATHERDD) | ~50ms (scalar) | 25× |
| `clam_build(100K)` | ~100ms (SIMD distance) | ~4s (scalar) | 40× |
| `zeckf64_from_distances` | ~5ns (same algorithm) | ~5ns (same algorithm) | 1× |
| `hpc()` dispatch overhead | ~1ns (LazyLock deref) | ~1ns (LazyLock deref) | — |

The dispatch overhead is **one pointer dereference** — same cost as ndarray's `simd_caps()` pattern.
This is negligible compared to any actual operation.

## 7. Design Decisions

### Why LazyLock, not static dispatch?
Static dispatch (`#[cfg]` on every call) is zero-cost but creates an army of conditional compilation.
LazyLock dispatch costs ~1ns per call (one pointer deref through `&dyn HpcBackend`) which is
negligible for operations that take 50ns-100ms. The code clarity gain is enormous.

### Why not a global trait object behind a feature flag?
That's what we're doing — `LazyLock<Box<dyn HpcBackend>>` IS a global trait object initialized
once. The `LazyLock` ensures thread-safe initialization without `unsafe`.

### Why not two separate crate features with different `mod` declarations?
That approach requires `#[cfg]` on every `mod` and `use` statement. The trait approach keeps
all module structure identical regardless of feature — only the `LazyLock::new()` closure differs.

### Why keep StandaloneBackend at all?
- CI without ndarray (faster builds for planner-only changes)
- Wasm targets (ndarray's SIMD won't compile to wasm32)
- Embedded targets (no std BLAS)
- Downstream consumers who want lance-graph without the full ndarray tree

### Precedent: ndarray's own `simd_caps()` singleton
ndarray uses exactly this pattern for SIMD capability detection:
```rust
static CAPS: LazyLock<SimdCaps> = LazyLock::new(SimdCaps::detect);
pub fn simd_caps() -> SimdCaps { *CAPS }
```
Our `hpc()` follows the same principle: detect once, dispatch forever.

## 8. File Impact Summary

| File | Action | Lines Changed |
|------|--------|--------------|
| `src/hpc_backend.rs` | NEW | ~300 (trait + 2 impls + LazyLock) |
| `src/lib.rs` | ADD `pub mod hpc_backend;` | 1 |
| `graph/blasgraph/ndarray_bridge.rs` | REFACTOR: extract standalone impl → StandaloneBackend | -80, +10 (thin re-export) |
| `graph/blasgraph/zeckf64.rs` | REFACTOR: call `hpc().zeckf64_from_distances()` | -100, +5 |
| `graph/neighborhood/zeckf64.rs` | DELETE (duplicate) | -120 |
| `graph/blasgraph/ops.rs` | REFACTOR: `hpc().hamming_distance()` | ~5 lines |
| Any future files | `use crate::hpc_backend::hpc;` | 1 per file |
