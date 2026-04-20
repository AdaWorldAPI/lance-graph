//! **LAB-ONLY.** D1.1 — `CodecKernelCache`: JIT kernel cache keyed by
//! `CodecParams::kernel_signature()`.
//!
//! The structural layer of Phase 1 — independent of the underlying
//! Cranelift / jitson implementation. This module defines the cache
//! semantics; D1.2 (rotation primitives), D1.3 (residual composition),
//! and D1.1b (actual Cranelift IR emission) plug into it.
//!
//! The insight this module captures: **kernel signature and sweep grid
//! axis are the same object viewed from two sides** (EPIPHANIES 2026-04-20
//! "D0.3 sweep grid IS the JIT cache warmer"). Every unique
//! `(subspaces, centroids, residual_depth, rotation_kind, distance,
//! lane_width)` tuple maps to exactly one `kernel_signature()` — so the
//! grid traversal order determines how fast the cache warms.
//!
//! ## Design — generic over handle type
//!
//! `CodecKernelCache<H>` is generic over `H: Clone` so this scaffold can
//! host:
//!
//! - **Production:** `H = KernelHandle` from `lance-graph-contract::jit`
//!   (raw function pointer to Cranelift-emitted code).
//! - **Stub / testing:** `H = StubKernel` (deterministic fake — what the
//!   kernel WOULD be, without compilation).
//! - **Future variants:** e.g., a GPU-kernel handle when/if that lands.
//!
//! The cache itself doesn't know or care what a kernel IS — it only
//! manages the `kernel_signature() → H` map with concurrent read-many /
//! single-writer semantics. Per ndarray/.claude/rules/data-flow.md:
//! "No `&mut self` during computation" — cache uses interior mutability.

use lance_graph_contract::cam::{CodecParams, CodecParamsError};
use std::collections::HashMap;
use std::sync::RwLock;

/// JIT kernel cache keyed by `CodecParams::kernel_signature()`.
///
/// Generic over kernel handle type. Concurrent-safe via `RwLock`; multiple
/// readers can hit cache simultaneously; exactly one writer at a time for
/// insert.
pub struct CodecKernelCache<H: Clone> {
    cache: RwLock<HashMap<u64, H>>,
    compile_count: RwLock<u64>,
    hit_count: RwLock<u64>,
}

impl<H: Clone> CodecKernelCache<H> {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            compile_count: RwLock::new(0),
            hit_count: RwLock::new(0),
        }
    }

    /// Get the kernel for `params`, compiling if missing.
    ///
    /// The `compile` closure runs **only on cache miss**; for the typical
    /// sweep where overlapping grid tuples share a kernel signature, most
    /// calls are zero-cost cache reads.
    ///
    /// Returns a cloned handle — the caller drives the kernel; the cache
    /// retains its own copy indefinitely.
    pub fn get_or_compile<F>(&self, params: &CodecParams, compile: F) -> H
    where
        F: FnOnce() -> H,
    {
        let sig = params.kernel_signature();
        // Fast path: read-lock check for cache hit.
        if let Some(h) = self.cache.read().unwrap().get(&sig).cloned() {
            *self.hit_count.write().unwrap() += 1;
            return h;
        }
        // Slow path: compile + insert. Double-check inside write-lock to
        // prevent duplicate compilation under concurrent misses.
        let mut w = self.cache.write().unwrap();
        if let Some(h) = w.get(&sig).cloned() {
            *self.hit_count.write().unwrap() += 1;
            return h;
        }
        let h = compile();
        w.insert(sig, h.clone());
        *self.compile_count.write().unwrap() += 1;
        h
    }

    /// Same as `get_or_compile` but with a fallible compile closure.
    pub fn try_get_or_compile<F>(
        &self,
        params: &CodecParams,
        compile: F,
    ) -> Result<H, CodecParamsError>
    where
        F: FnOnce() -> Result<H, CodecParamsError>,
    {
        let sig = params.kernel_signature();
        if let Some(h) = self.cache.read().unwrap().get(&sig).cloned() {
            *self.hit_count.write().unwrap() += 1;
            return Ok(h);
        }
        let mut w = self.cache.write().unwrap();
        if let Some(h) = w.get(&sig).cloned() {
            *self.hit_count.write().unwrap() += 1;
            return Ok(h);
        }
        let h = compile()?;
        w.insert(sig, h.clone());
        *self.compile_count.write().unwrap() += 1;
        Ok(h)
    }

    /// Number of unique kernels in the cache (= unique signatures seen).
    pub fn len(&self) -> usize {
        self.cache.read().unwrap().len()
    }

    /// `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of `compile()` invocations — one per unique signature.
    pub fn compile_count(&self) -> u64 {
        *self.compile_count.read().unwrap()
    }

    /// Number of cache hits (compile closure NOT invoked).
    pub fn hit_count(&self) -> u64 {
        *self.hit_count.read().unwrap()
    }

    /// Cache hit ratio: `hit_count / (hit_count + compile_count)`.
    /// Returns 0.0 when no calls have been made.
    pub fn hit_ratio(&self) -> f64 {
        let hits = self.hit_count() as f64;
        let compiles = self.compile_count() as f64;
        let total = hits + compiles;
        if total < 0.5 { 0.0 } else { hits / total }
    }

    /// Check whether a specific signature is cached without calling compile.
    pub fn has_signature(&self, signature: u64) -> bool {
        self.cache.read().unwrap().contains_key(&signature)
    }

    /// Clear the cache (and reset counters). Useful for test isolation.
    pub fn clear(&self) {
        self.cache.write().unwrap().clear();
        *self.compile_count.write().unwrap() = 0;
        *self.hit_count.write().unwrap() = 0;
    }
}

impl<H: Clone> Default for CodecKernelCache<H> {
    fn default() -> Self {
        Self::new()
    }
}

/// Deterministic stub kernel handle — for testing the cache without
/// invoking the real Cranelift / jitson compilation path.
///
/// Captures what the kernel WOULD be (the signature it was compiled for +
/// whether AMX would be used). D1.1b's Cranelift path replaces the
/// stub with a real `KernelHandle`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StubKernel {
    /// `CodecParams::kernel_signature()` this stub represents.
    pub signature: u64,
    /// `params.is_matmul_heavy()` at compile time — drives Tier-1 AMX dispatch.
    pub is_matmul_heavy: bool,
    /// SIMD tier name this stub claims ("amx" | "vnni" | "avx512" | "avx2").
    /// Never "scalar" on a SoA path — iron rule.
    pub backend: &'static str,
}

impl StubKernel {
    /// Build a stub from the current `CodecParams`, selecting a tier label
    /// under the assumption that AMX is available for matmul-heavy paths.
    /// The actual per-process capability query is
    /// `ndarray::simd_amx::amx_available()`; this stub pretends it's true.
    pub fn from_params(params: &CodecParams) -> Self {
        Self {
            signature: params.kernel_signature(),
            is_matmul_heavy: params.is_matmul_heavy(),
            backend: if params.is_matmul_heavy() { "amx" } else { "avx512" },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::cam::{CodecParamsBuilder, LaneWidth, Rotation};

    #[test]
    fn cache_starts_empty() {
        let c: CodecKernelCache<StubKernel> = CodecKernelCache::new();
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
        assert_eq!(c.compile_count(), 0);
        assert_eq!(c.hit_count(), 0);
        assert_eq!(c.hit_ratio(), 0.0);
    }

    #[test]
    fn first_call_compiles_second_is_cache_hit() {
        let c: CodecKernelCache<StubKernel> = CodecKernelCache::new();
        let p = CodecParamsBuilder::new().centroids(1024).build().unwrap();

        let k1 = c.get_or_compile(&p, || StubKernel::from_params(&p));
        let k2 = c.get_or_compile(&p, || panic!("must not recompile on cache hit"));

        assert_eq!(k1, k2);
        assert_eq!(c.compile_count(), 1);
        assert_eq!(c.hit_count(), 1);
        assert_eq!(c.len(), 1);
        assert_eq!(c.hit_ratio(), 0.5);
    }

    #[test]
    fn different_params_produce_different_kernels() {
        let c: CodecKernelCache<StubKernel> = CodecKernelCache::new();
        let p1 = CodecParamsBuilder::new().centroids(256).build().unwrap();
        let p2 = CodecParamsBuilder::new().centroids(1024).build().unwrap();

        let k1 = c.get_or_compile(&p1, || StubKernel::from_params(&p1));
        let k2 = c.get_or_compile(&p2, || StubKernel::from_params(&p2));

        assert_ne!(k1.signature, k2.signature);
        assert_eq!(c.compile_count(), 2);
        assert_eq!(c.hit_count(), 0);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn seed_changes_do_not_invalidate_cache() {
        // CodecParams::kernel_signature() excludes `seed` (PR #225).
        // Same IR-shaping fields → same signature → cache hit.
        let c: CodecKernelCache<StubKernel> = CodecKernelCache::new();
        let p1 = CodecParamsBuilder::new().seed(1).build().unwrap();
        let p2 = CodecParamsBuilder::new().seed(2).build().unwrap();

        let k1 = c.get_or_compile(&p1, || StubKernel::from_params(&p1));
        let k2 = c.get_or_compile(&p2, || panic!("seed change must not invalidate cache"));

        assert_eq!(k1, k2);
        assert_eq!(c.compile_count(), 1);
        assert_eq!(c.hit_count(), 1);
    }

    #[test]
    fn matmul_heavy_params_select_amx_backend_in_stub() {
        let opq = CodecParamsBuilder::new()
            .lane_width(LaneWidth::BF16x32)
            .rotation(Rotation::Opq { matrix_blob_id: 42, dim: 4096 })
            .build()
            .unwrap();
        let identity = CodecParamsBuilder::new().build().unwrap();

        let k_opq = StubKernel::from_params(&opq);
        let k_id = StubKernel::from_params(&identity);

        assert_eq!(k_opq.backend, "amx");
        assert!(k_opq.is_matmul_heavy);
        assert_eq!(k_id.backend, "avx512");
        assert!(!k_id.is_matmul_heavy);
    }

    #[test]
    fn clear_resets_cache_and_counters() {
        let c: CodecKernelCache<StubKernel> = CodecKernelCache::new();
        let p = CodecParamsBuilder::new().build().unwrap();
        c.get_or_compile(&p, || StubKernel::from_params(&p));
        c.get_or_compile(&p, || StubKernel::from_params(&p));

        assert_eq!(c.len(), 1);
        assert_eq!(c.compile_count(), 1);
        assert_eq!(c.hit_count(), 1);

        c.clear();

        assert_eq!(c.len(), 0);
        assert_eq!(c.compile_count(), 0);
        assert_eq!(c.hit_count(), 0);
        assert!(c.is_empty());
    }

    #[test]
    fn try_get_or_compile_propagates_errors() {
        let c: CodecKernelCache<StubKernel> = CodecKernelCache::new();
        let p = CodecParamsBuilder::new().build().unwrap();
        let result: Result<StubKernel, _> = c.try_get_or_compile(&p, || {
            Err(CodecParamsError::ZeroDimension { field: "test" })
        });
        assert!(result.is_err());
        // Failed compile doesn't populate cache.
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn has_signature_checks_without_compiling() {
        let c: CodecKernelCache<StubKernel> = CodecKernelCache::new();
        let p = CodecParamsBuilder::new().centroids(512).build().unwrap();
        let sig = p.kernel_signature();

        assert!(!c.has_signature(sig));
        c.get_or_compile(&p, || StubKernel::from_params(&p));
        assert!(c.has_signature(sig));
    }

    #[test]
    fn sweep_grid_warms_cache_deterministically() {
        // Simulate the D0.3 insight: a sweep grid with 4 distinct kernel
        // signatures + 1 repeat (seed difference) compiles exactly 4 kernels.
        let c: CodecKernelCache<StubKernel> = CodecKernelCache::new();
        let candidates: Vec<CodecParams> = vec![
            CodecParamsBuilder::new().centroids(256).build().unwrap(),
            CodecParamsBuilder::new().centroids(512).build().unwrap(),
            CodecParamsBuilder::new().centroids(1024).build().unwrap(),
            CodecParamsBuilder::new().centroids(256).seed(999).build().unwrap(), // same sig as first
            CodecParamsBuilder::new()
                .lane_width(LaneWidth::BF16x32)
                .rotation(Rotation::Opq { matrix_blob_id: 1, dim: 4096 })
                .build().unwrap(),
        ];

        for p in &candidates {
            c.get_or_compile(p, || StubKernel::from_params(p));
        }

        // 4 unique signatures (seed=999 collides with the first).
        assert_eq!(c.len(), 4);
        assert_eq!(c.compile_count(), 4);
        assert_eq!(c.hit_count(), 1);
        assert!((c.hit_ratio() - 0.2).abs() < 1e-9);
    }
}
