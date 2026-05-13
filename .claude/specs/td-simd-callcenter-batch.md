# TD-SIMD-CALLCENTER-BATCH-PATHS-1 — Callcenter Batch Path SIMD Retrofit

**TD-ID:** TD-SIMD-CALLCENTER-BATCH-PATHS-1  
**Priority:** P2  
**Sprint:** sprint-log-4 (2026-05-13)  
**Worker:** W5  
**Branch:** `claude/lance-datafusion-integration-gv0BF`

---

## 1. Problem Statement

The `lance-graph-callcenter` crate ships five DataFusion scalar UDFs that perform VSA (Vector Symbolic Architecture) operations over `FixedSizeBinary(2048)` fingerprints (`[u64; 256]` = 16 Kbit). Every hot-path operation — XOR-bind, XOR-bundle, Hamming-distance, top-k — executes a scalar `for` loop over 256 `u64` words per fingerprint. When DataFusion invokes these UDFs over a batch of N rows, the loop runs N × 256 iterations with no vectorisation.

The sibling repo `/home/user/ndarray` already ships SIMD-accelerated primitives in `ndarray::hpc::bitwise` and `ndarray::hpc::vsa`, auto-dispatched via a `LazyLock<SimdCaps>` singleton (`simd_caps()`). §19.2 of the architecture spec designates these as the canonical batch-compute path (Path C of the three-path convergence: Path A = thinking-engine cognition, Path B = ractor supervisor runtime, Path C = ndarray::simd compute).

---

## 2. Scalar-Loop Inventory

### 2.1 grep results — scalar loops in lance-graph-callcenter/src/

```
vsa_udfs.rs:62:    for (i, &w) in words.iter().enumerate() {
```

All substantive scalar loops live inside the private helper functions in `crates/lance-graph-callcenter/src/vsa_udfs.rs`. The other loop sites (`dn_path.rs:75`, `transcode/zerocopy.rs:361/478/640`, `unified_audit.rs:299`, `audit.rs:660/844`, `postgrest.rs:449`) iterate over struct fields or audit records — not compute-hot and out of scope.

### 2.2 Hot scalar sites (all in `vsa_udfs.rs`)

| Function | Line (approx) | Pattern | Cost per UDF call |
|---|---|---|---|
| `bytes_to_words` | ~55 | `chunks_exact(8).enumerate().take(256)` | 256 scalar byte→u64 loads |
| `words_to_bytes` | ~62 | `for (i, &w) in words.iter().enumerate()` | 256 scalar u64→byte stores |
| `bundle_op` | ~115 | `for i in 0..FP_WORDS { out[i] = wa[i] ^ wb[i]; }` | 256 scalar XOR |
| `hamming_dist_op` | ~125 | `.zip().map(|(x,y)| (x^y).count_ones()).sum()` | 256 scalar XOR+popcount |
| `braid_at_op` | ~133 | `for i in 0..FP_WORDS { ... }` | 256 scalar rotates |
| `top_k_op` | ~143 | `.iter().enumerate().map(...)` | 256 popcount + alloc + sort |

Note: `unbind_op` reads a slice and calls `sum()` — also scalar but only touches 32 words (role slice), lower priority.

### 2.3 Not yet present — batch-level loops

The UDF `invoke_with_args` functions iterate over Arrow rows with `(0..len).map(|i| ...)`. Each iteration calls one of the above helpers. This is the outer batch loop — N iterations × 256 inner iterations = O(N·256) unvectorised work.

---

## 3. ndarray::simd Surface

### 3.1 Core primitives (available today in `/home/user/ndarray`)

All dispatch through a single `LazyLock<SimdCaps>` singleton in `ndarray::hpc::simd_caps`:

```rust
// Detect once, dispatch forever (~1ns deref vs ~3ns per-call feature detection)
use ndarray::hpc::simd_caps::simd_caps;  // returns SimdCaps (Copy struct)
```

**Bitwise batch ops** (`ndarray::hpc::bitwise`):

```rust
// Single-pair Hamming — dispatches AVX2/NEON/Scalar automatically
pub fn hamming_distance_raw(a: &[u8], b: &[u8]) -> u64

// Batch query-vs-database — zero allocation, SIMD-accelerated
// database: num_rows * row_bytes contiguous bytes
pub fn hamming_batch_raw(
    query: &[u8], database: &[u8],
    num_rows: usize, row_bytes: usize,
) -> Vec<u64>

// Top-k nearest by Hamming — O(n) partial sort
pub fn hamming_top_k_raw(
    query: &[u8], database: &[u8],
    num_rows: usize, row_bytes: usize, k: usize,
) -> (Vec<usize>, Vec<u64>)

// Raw popcount
pub fn popcount_raw(a: &[u8]) -> u64
```

**VSA semantic ops** (`ndarray::hpc::vsa`):

```rust
use ndarray::hpc::vsa::{VsaVector, vsa_bind, vsa_bundle, vsa_similarity, vsa_hamming};

// XOR bind (= unbind, self-inverse)
pub fn vsa_bind(a: &VsaVector, b: &VsaVector) -> VsaVector
pub fn vsa_unbind(bundle: &VsaVector, key: &VsaVector) -> VsaVector  // alias

// Majority-vote bundle (N-ary, allocation via VsaAccumulator)
pub fn vsa_bundle(items: &[VsaVector]) -> VsaVector

// Normalized Hamming similarity [0.0, 1.0]
pub fn vsa_similarity(a: &VsaVector, b: &VsaVector) -> f32

// Raw Hamming distance (delegates to bitwise::hamming_distance_raw)
pub fn vsa_hamming(a: &VsaVector, b: &VsaVector) -> u32
```

**SIMD types** (`ndarray::simd`):

```rust
// F32x16: dispatches AVX512(16 lanes) -> AVX2(8) -> NEON(4) -> Scalar(1)
use ndarray::simd::F32x16;
// U8x64: same dispatch, used for XOR+popcount
use ndarray::simd::U8x64;
```

### 3.2 Dispatch mechanism

```
simd_caps()
  +-- LazyLock<SimdCaps>  (first call: runtime cpuid; subsequent: pointer deref)
       +-- caps.avx512f -> Avx512 tier (16 x f32 / 512-bit)
       +-- caps.avx2    -> Avx2 tier   (8 x f32 / 256-bit)
       +-- caps.neon    -> NeonDotProd or Neon (4 x f32 / 128-bit)
       +-- fallback     -> Scalar tier (1 lane)
```

The `ndarray::hpc::bitwise` functions call `dispatch_hamming` / `dispatch_popcount` which branch on a frozen `SimdDispatch` table built from `simd_caps()` — zero per-call branching after warm-up.

### 3.3 Note on naming vs. prompt spec

The prompt references `vsa_cosine_batch`, `vsa_bundle_simd`, `vsa_bind_simd_inplace`, and `Vsa16kF32`. These names do NOT exist in the current ndarray codebase. The actual canonical names are:

- `vsa_bundle` (majority-vote, operates on `VsaVector = [u64; 256]`, Binary16K)
- `vsa_bind` / `vsa_unbind` (XOR, inplace variant does not yet exist)
- `vsa_similarity` / `vsa_hamming` (dispatches to `hamming_distance_raw`)
- The binary fingerprint type is `VsaVector` (16 Kbit) not `Vsa16kF32`

For f32 distance (cosine over float embeddings), use `ndarray::hpc::heel_f64x8::cosine_f32_to_f64_simd`.

---

## 4. Migration Recipe

### 4.1 Feature-flag guard

Add to `lance-graph-callcenter/Cargo.toml`:

```toml
[features]
ndarray-hpc = ["dep:ndarray-crate"]  # gates all SIMD paths; fallback = scalar-correct

[dependencies]
ndarray-crate = { path = "../../path/to/ndarray", optional = true, package = "ndarray" }
```

**Note:** The exact path and package name must be resolved against the workspace Cargo.toml. The feature name `ndarray-hpc` follows the architecture spec §19.2 naming.

---

### 4.2 Site A — `bundle_op` (XOR bundle)

**Before (scalar, `vsa_udfs.rs` approx line 115):**

```rust
fn bundle_op(a: &[u8], b: &[u8]) -> [u64; FP_WORDS] {
    let wa = bytes_to_words(a);
    let wb = bytes_to_words(b);
    let mut out = [0u64; FP_WORDS];
    for i in 0..FP_WORDS {
        out[i] = wa[i] ^ wb[i];   // scalar XOR, 256 iterations
    }
    out
}
```

**After (SIMD via ndarray):**

```rust
#[cfg(feature = "ndarray-hpc")]
fn bundle_op(a: &[u8], b: &[u8]) -> [u64; FP_WORDS] {
    use ndarray_crate::hpc::vsa::{VsaVector, vsa_bind};
    let va = VsaVector::from_bytes(a);
    let vb = VsaVector::from_bytes(b);
    vsa_bind(&va, &vb).words  // dispatches U8x64 XOR (AVX2: 32-byte chunks)
}

#[cfg(not(feature = "ndarray-hpc"))]
fn bundle_op(a: &[u8], b: &[u8]) -> [u64; FP_WORDS] {
    // scalar fallback unchanged
    let wa = bytes_to_words(a);
    let wb = bytes_to_words(b);
    let mut out = [0u64; FP_WORDS];
    for i in 0..FP_WORDS { out[i] = wa[i] ^ wb[i]; }
    out
}
```

---

### 4.3 Site B — `hamming_dist_op` + batch path

**Before (scalar per-row in `invoke_with_args`):**

```rust
// In HammingDistUdf::invoke_with_args -- N rows, each calls hamming_dist_op
let results: Vec<Option<u32>> = (0..len)
    .map(|i| match (&a_fp[i], &b_fp[i]) {
        (Some(a), Some(b)) => Some(hamming_dist_op(a, b)),  // scalar inner loop
        _ => None,
    })
    .collect();
```

Where `hamming_dist_op` is:

```rust
fn hamming_dist_op(a: &[u8], b: &[u8]) -> u32 {
    let wa = bytes_to_words(a); let wb = bytes_to_words(b);
    wa.iter().zip(wb.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}
```

**After (SIMD via ndarray, per-row):**

```rust
#[cfg(feature = "ndarray-hpc")]
fn hamming_dist_op(a: &[u8], b: &[u8]) -> u32 {
    ndarray_crate::hpc::bitwise::hamming_distance_raw(a, b) as u32
}
```

**After (SIMD via ndarray, contiguous-buffer batch path):**

```rust
// When Arrow buffer is contiguous (no nulls splitting the backing buffer):
let database_buf: &[u8] = fps_b.value_data();  // num_rows * FP_BYTES contiguous bytes
let query: &[u8] = fps_a.value(0);             // single query row
let distances = ndarray_crate::hpc::bitwise::hamming_batch_raw(
    query, database_buf, len, FP_BYTES as usize,
);
// Amortises dispatch overhead across entire Arrow batch -- one call for N rows
```

---

## 5. Benchmark Plan

### 5.1 Benchmark location

Create `crates/lance-graph-callcenter/benches/callcenter_simd.rs`.

### 5.2 Benchmark groups

```rust
// cargo bench --bench callcenter_simd --features ndarray-hpc

fn bench_hamming_scalar(c: &mut Criterion) { /* existing scalar loop */ }
fn bench_hamming_simd(c: &mut Criterion)   { /* hamming_distance_raw */ }
fn bench_hamming_batch_scalar(c: &mut Criterion) { /* N=1024 row loop */ }
fn bench_hamming_batch_simd(c: &mut Criterion)   { /* hamming_batch_raw */ }
fn bench_bundle_scalar(c: &mut Criterion) { /* existing bundle_op */ }
fn bench_bundle_simd(c: &mut Criterion)   { /* vsa_bind via VsaVector */ }
```

### 5.3 Assertion target

On an AVX2 host (Intel/AMD x86_64 with AVX2):

| Operation | Scalar baseline | SIMD target | Required multiplier |
|---|---|---|---|
| Single hamming (2048 bytes) | ~400 ns | <= 100 ns | >= 4x |
| Batch hamming (N=1024 rows) | ~400 us | <= 80 us | >= 5x |
| XOR-bundle (2048 bytes) | ~300 ns | <= 75 ns | >= 4x |

CI assertion via criterion threshold:

```toml
[[bench]]
name = "callcenter_simd"
harness = false
required-features = ["ndarray-hpc"]
```

```rust
// In bench: assert SIMD throughput >= 4x scalar or fail
let scalar_ns = measure_scalar();
let simd_ns   = measure_simd();
assert!(scalar_ns / simd_ns >= 4,
    "AVX2 SIMD speedup {:.1}x < required 4x", scalar_ns as f64 / simd_ns as f64);
```

### 5.4 Platform matrix

| Platform | Tier | Expected speedup |
|---|---|---|
| x86_64 + AVX2 | `Avx2` (8 x f32) | >= 4x vs scalar |
| x86_64 + AVX-512 | `Avx512` (16 x f32) | >= 8x vs scalar |
| aarch64 + NEON | `Neon` / `NeonDotProd` | >= 2x vs scalar |
| i686 / WASM | `Scalar` | 1x (no regression) |

---

## 6. Feature-Flag Fallback

The `ndarray-hpc` feature gates all SIMD paths. When absent, every `#[cfg(not(feature = "ndarray-hpc"))]` block retains the existing scalar implementation unchanged.

### 6.1 Correctness invariants for fallback

- `bundle_op` scalar: XOR of 256 u64 words — identical output to `vsa_bind`
- `hamming_dist_op` scalar: `count_ones(XOR)` summation — identical output to `hamming_distance_raw`
- `bytes_to_words` / `words_to_bytes` remain shared helpers (compiler auto-vectorises `chunks_exact`)

### 6.2 Test guard

```rust
#[test]
fn hamming_scalar_simd_parity() {
    let a = vec![0xABu8; FP_BYTES as usize];
    let b = vec![0xCDu8; FP_BYTES as usize];
    let scalar = hamming_dist_op(&a, &b);
    #[cfg(feature = "ndarray-hpc")]
    {
        let simd = ndarray_crate::hpc::bitwise::hamming_distance_raw(&a, &b) as u32;
        assert_eq!(scalar, simd, "scalar/SIMD parity");
    }
}
```

### 6.3 Feature flag interaction with existing features

The `ndarray-hpc` feature is orthogonal to all existing callcenter features (`query`, `persist`, `realtime`, `audit-log`, etc.). It may be combined freely. The `full` feature set should NOT auto-enable `ndarray-hpc` — that choice belongs to the consumer (avoids pulling ndarray into users who don't need HPC batch).

---

## 7. Cross-Flag: W6 Coordination (Shared Dispatch Cache)

W6's spec covers the thinking-engine (TD-THINKING-ENGINE-UNWIRED-1), which also uses `ndarray::simd` for f32 cosine/bind operations over Vsa16k carriers. Both consumers will call `simd_caps()` — the `LazyLock` ensures detection runs exactly once per process regardless of how many crates call it.

**Shared pattern:**

```rust
// Both W5 (callcenter) and W6 (thinking-engine) should use the same singleton
// -- no per-crate detection copies needed
use ndarray_crate::hpc::simd_caps::simd_caps;
// First caller (any crate) pays ~50us detection cost; all subsequent: ~1ns deref
```

**Recommended coordination:**

- Both crates depend on the same ndarray path dep version/rev
- Do NOT create per-crate SIMD dispatch tables — use ndarray's `SIMD_DISPATCH` singleton
- Document in `CLAUDE.md` under `§ SIMD_DISPATCH`: "callcenter + thinking-engine both consume the same frozen dispatch table via ndarray dep"

---

## 8. Implementation Sequence

1. Add `ndarray-hpc` feature + optional ndarray dep to `Cargo.toml`
2. Wrap `bytes_to_words` / `words_to_bytes` — verify compiler auto-vectorises via godbolt (likely already does with `chunks_exact`)
3. Retrofit `bundle_op` -> `vsa_bind` (Site A above)
4. Retrofit `hamming_dist_op` -> `hamming_distance_raw` (Site B above)
5. Add batch-level aggregation for `HammingDistUdf` using `hamming_batch_raw` (contiguous buffer path)
6. Retrofit `top_k_op` -> `hamming_top_k_raw` (O(n) partial sort + SIMD distances)
7. Write bench + parity tests
8. CI: add `--features ndarray-hpc` run to `.github/workflows`

---

## 9. Open Questions

1. **ndarray workspace path:** The ndarray crate lives at `/home/user/ndarray` (separate repo). Is it published to crates.io or consumed as a git dep? The Cargo.toml path dep must be resolved; a published version would be preferred for CI portability.

2. **VsaVector vs raw `[u64; 256]`:** The callcenter uses `[u64; 256]` (matching `lance-graph-contract::crystal::Fingerprint<256>`). `VsaVector` wraps the same layout. Does the conversion `VsaVector::from_bytes` -> `.words` add a copy overhead that negates small-batch SIMD gains? For N < ~16 rows, the `LazyLock` deref amortises poorly — need profiling to find the crossover batch size.

3. **Majority-vote bundle deferred:** `bundle_op` is currently XOR (Phase A), not majority vote (Phase B). The ndarray `vsa_bundle` implements majority vote (VsaAccumulator i16 path). For Phase B migration, the UDF signature and semantics change. This spec covers only Phase A XOR -> `vsa_bind` SIMD retrofit. Phase B majority-vote migration is a separate TD.

4. **Arrow buffer contiguity guarantee:** `hamming_batch_raw` requires contiguous byte layout. Arrow `FixedSizeBinaryArray::value_data()` returns a contiguous `&[u8]` buffer IFF no nulls or offsets split the backing buffer. Need to verify this holds for DataFusion's columnar execution — if not, fall back to per-row `hamming_distance_raw` calls.

---

## Appendix: Actual Scalar Loops (verbatim, for review)

File: `crates/lance-graph-callcenter/src/vsa_udfs.rs`

```rust
// bundle_op scalar (retrofitable to vsa_bind)
fn bundle_op(a: &[u8], b: &[u8]) -> [u64; FP_WORDS] {
    let wa = bytes_to_words(a);
    let wb = bytes_to_words(b);
    let mut out = [0u64; FP_WORDS];
    for i in 0..FP_WORDS {         // 256 iterations
        out[i] = wa[i] ^ wb[i];
    }
    out
}

// hamming_dist_op scalar (retrofitable to hamming_distance_raw)
fn hamming_dist_op(a: &[u8], b: &[u8]) -> u32 {
    let wa = bytes_to_words(a);
    let wb = bytes_to_words(b);
    wa.iter()
        .zip(wb.iter())
        .map(|(x, y)| (x ^ y).count_ones())  // 256 iterations
        .sum()
}

// words_to_bytes scalar (line 62 -- likely auto-vectorised already)
fn words_to_bytes(words: &[u64; FP_WORDS]) -> [u8; FP_BYTES as usize] {
    let mut out = [0u8; FP_BYTES as usize];
    for (i, &w) in words.iter().enumerate() {   // 256 iterations
        out[i * 8..(i + 1) * 8].copy_from_slice(&w.to_le_bytes());
    }
    out
}
```

---

*Spec written by W5 (sprint-log-4, 2026-05-13). See `.claude/board/sprint-log-4/agents/agent-W5.md` for research log.*
