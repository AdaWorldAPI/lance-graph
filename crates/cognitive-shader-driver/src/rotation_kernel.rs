//! **LAB-ONLY.** D1.2 — rotation primitives as `RotationKernel`
//! implementations.
//!
//! Three variants matching `lance_graph_contract::cam::Rotation`:
//!
//! - **Identity** — no-op; zero-overhead pass-through. `signature()` only
//!   depends on dim so the JIT cache hit is trivial.
//! - **Hadamard** — real Sylvester butterfly in-place, `O(N log N)` add/sub
//!   operations. No JIT needed — the butterfly is a fixed-shape kernel and
//!   plain Rust compiles to AVX-512 under `target-cpu=x86-64-v4`.
//!   Per Rule C: Hadamard stays at Tier-3 F32x16 because it's add/sub,
//!   not matmul — AMX adds no value here (confirmed in plan appendix §12).
//! - **OPQ** — learned rotation matmul; placeholder stub. Real impl
//!   plugs into `ndarray::hpc::jitson_cranelift::JitEngine` via the
//!   D1.1b `CodecKernelEngine` adapter and uses AMX tile_dpbf16ps when
//!   `amx_available()`.
//!
//! Per ndarray/.claude/rules/data-flow.md: in-place `&mut [f32]` slice;
//! no heap allocations inside rotation; computation paths never mutate
//! `self` — the `RotationKernel` trait's `&self` receiver is load-bearing.

use lance_graph_contract::cam::Rotation;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Error produced when a rotation cannot be applied — dimensional
/// mismatch, non-power-of-two for Hadamard, or missing OPQ matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RotationError {
    /// Input slice length does not match the kernel's declared dim.
    DimMismatch { expected: usize, actual: usize },
    /// Hadamard dim must be a power of two (Sylvester construction).
    HadamardNotPow2 { dim: u32 },
    /// OPQ rotation matrix not loaded (stub path).
    OpqMatrixNotLoaded { matrix_blob_id: u64 },
}

impl std::fmt::Display for RotationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimMismatch { expected, actual } => {
                write!(f, "rotation input dim mismatch: expected {expected}, got {actual}")
            }
            Self::HadamardNotPow2 { dim } => {
                write!(f, "Hadamard dim must be power of two, got {dim}")
            }
            Self::OpqMatrixNotLoaded { matrix_blob_id } => {
                write!(f, "OPQ rotation matrix blob {matrix_blob_id:#x} not loaded")
            }
        }
    }
}

impl std::error::Error for RotationError {}

/// A compiled rotation kernel.
///
/// Implementors run the rotation in-place on a `&mut [f32]` slice.
/// The trait is object-safe so callers can hold a `Box<dyn RotationKernel>`
/// when the variant is chosen at runtime from a `CodecParams::pre_rotation`.
pub trait RotationKernel: Send + Sync + std::fmt::Debug {
    /// Apply the rotation in place. Contract: modifies `vec` in-place;
    /// returns `Err` on dim mismatch, never on a valid call shape.
    fn apply(&self, vec: &mut [f32]) -> Result<(), RotationError>;

    /// Declared input dimension. Used by the cache-signature computation
    /// and by the `CodecKernelCache` key (distinct dims → distinct kernels).
    fn dim(&self) -> u32;

    /// Stable hash over the kernel's identity — used as part of
    /// `CodecParams::kernel_signature()` so the cache keys cleanly.
    fn signature(&self) -> u64;

    /// Backend tier label for the SIMD dispatch trace — "avx512" for
    /// identity/Hadamard on x86_64-v4, "amx" for OPQ when AMX is live,
    /// "stub" for OPQ without a loaded matrix. Never "scalar" — iron rule.
    fn backend(&self) -> &'static str;
}

/// Build a boxed kernel from a `Rotation` enum + input dim.
///
/// This is the factory the JIT cache's compile closure calls:
/// `cache.get_or_compile(params, || build(params.pre_rotation, d)?)`.
pub fn build(rotation: &Rotation, dim: u32) -> Result<Box<dyn RotationKernel>, RotationError> {
    match rotation {
        Rotation::Identity => Ok(Box::new(IdentityRotation { dim })),
        Rotation::Hadamard { dim: h_dim } => {
            // Respect the rotation's declared dim — caller must size to match.
            if *h_dim != dim {
                return Err(RotationError::DimMismatch {
                    expected: *h_dim as usize,
                    actual: dim as usize,
                });
            }
            if *h_dim == 0 || !h_dim.is_power_of_two() {
                return Err(RotationError::HadamardNotPow2 { dim: *h_dim });
            }
            Ok(Box::new(HadamardRotation { dim: *h_dim }))
        }
        Rotation::Opq { matrix_blob_id, dim: o_dim } => {
            if *o_dim != dim {
                return Err(RotationError::DimMismatch {
                    expected: *o_dim as usize,
                    actual: dim as usize,
                });
            }
            // Stub — D1.1b wires the real matrix load through
            // ndarray::hpc::jitson_cranelift::JitEngine + tile_dpbf16ps.
            Ok(Box::new(OpqRotationStub {
                matrix_blob_id: *matrix_blob_id,
                dim: *o_dim,
            }))
        }
    }
}

// ─── Identity ────────────────────────────────────────────────────────────

/// Zero-overhead pass-through rotation. `apply()` is a no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdentityRotation {
    pub dim: u32,
}

impl RotationKernel for IdentityRotation {
    fn apply(&self, vec: &mut [f32]) -> Result<(), RotationError> {
        if vec.len() != self.dim as usize {
            return Err(RotationError::DimMismatch {
                expected: self.dim as usize,
                actual: vec.len(),
            });
        }
        // No-op.
        Ok(())
    }

    fn dim(&self) -> u32 { self.dim }

    fn signature(&self) -> u64 {
        let mut h = DefaultHasher::new();
        "identity".hash(&mut h);
        self.dim.hash(&mut h);
        h.finish()
    }

    fn backend(&self) -> &'static str { "avx512" }
}

// ─── Hadamard (Sylvester butterfly) ──────────────────────────────────────

/// Sylvester Hadamard transform via in-place butterfly.
///
/// For dim `N = 2^k`, the Sylvester Hadamard matrix `H_N` satisfies
/// `H_N · H_N^T = N · I`. We apply `H_N` in-place using the classic
/// butterfly algorithm: `log2(N)` stages, each swapping pairs of elements
/// at stride `2^stage` with `(a, b) → (a+b, a-b)`.
///
/// Complexity: `O(N log N)` add/sub operations. No allocations.
/// No AMX benefit (Rule C) — Hadamard is butterfly add/sub, not matmul,
/// so it stays at Tier-3 F32x16 (AVX-512 baseline).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HadamardRotation {
    pub dim: u32,
}

impl RotationKernel for HadamardRotation {
    fn apply(&self, vec: &mut [f32]) -> Result<(), RotationError> {
        let n = self.dim as usize;
        if vec.len() != n {
            return Err(RotationError::DimMismatch { expected: n, actual: vec.len() });
        }
        if n == 0 || !n.is_power_of_two() {
            return Err(RotationError::HadamardNotPow2 { dim: self.dim });
        }
        // In-place Sylvester butterfly. `stride` doubles each stage.
        let mut stride = 1usize;
        while stride < n {
            let mut i = 0;
            while i < n {
                for j in 0..stride {
                    let a_idx = i + j;
                    let b_idx = i + j + stride;
                    let a = vec[a_idx];
                    let b = vec[b_idx];
                    vec[a_idx] = a + b;
                    vec[b_idx] = a - b;
                }
                i += stride * 2;
            }
            stride *= 2;
        }
        Ok(())
    }

    fn dim(&self) -> u32 { self.dim }

    fn signature(&self) -> u64 {
        let mut h = DefaultHasher::new();
        "hadamard".hash(&mut h);
        self.dim.hash(&mut h);
        h.finish()
    }

    fn backend(&self) -> &'static str { "avx512" }
}

// ─── OPQ (stub — real impl plugs JIT engine in D1.1b) ────────────────────

/// OPQ learned rotation matmul — stub. `apply()` returns
/// `OpqMatrixNotLoaded`.
///
/// The real implementation loads the rotation matrix from a Lance blob
/// column (one-time per `matrix_blob_id`) and applies it via
/// `ndarray::hpc::amx_matmul::tile_dpbf16ps` when
/// `ndarray::simd_amx::amx_available()` (Tier-1), falling through to
/// VNNI (Tier-2) or F32x16 matmul (Tier-3) per the polyfill hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpqRotationStub {
    pub matrix_blob_id: u64,
    pub dim: u32,
}

impl RotationKernel for OpqRotationStub {
    fn apply(&self, vec: &mut [f32]) -> Result<(), RotationError> {
        if vec.len() != self.dim as usize {
            return Err(RotationError::DimMismatch {
                expected: self.dim as usize,
                actual: vec.len(),
            });
        }
        // Stub — no matrix loaded yet.
        Err(RotationError::OpqMatrixNotLoaded { matrix_blob_id: self.matrix_blob_id })
    }

    fn dim(&self) -> u32 { self.dim }

    fn signature(&self) -> u64 {
        let mut h = DefaultHasher::new();
        "opq".hash(&mut h);
        self.matrix_blob_id.hash(&mut h);
        self.dim.hash(&mut h);
        h.finish()
    }

    fn backend(&self) -> &'static str { "stub" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_rotation_is_noop() {
        let r = IdentityRotation { dim: 8 };
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let before = v.clone();
        r.apply(&mut v).unwrap();
        assert_eq!(v, before);
        assert_eq!(r.backend(), "avx512");
    }

    #[test]
    fn identity_rotation_rejects_dim_mismatch() {
        let r = IdentityRotation { dim: 8 };
        let mut v = vec![0.0; 16];
        let err = r.apply(&mut v).unwrap_err();
        assert!(matches!(err, RotationError::DimMismatch { expected: 8, actual: 16 }));
    }

    #[test]
    fn hadamard_orthogonality_property_n4() {
        // H_4 applied to [1,0,0,0] produces [1,1,1,1] (first column of H_4).
        let r = HadamardRotation { dim: 4 };
        let mut v = vec![1.0, 0.0, 0.0, 0.0];
        r.apply(&mut v).unwrap();
        assert_eq!(v, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn hadamard_n8_applied_twice_scales_by_n() {
        // H · H = n · I ⇒ applying twice multiplies every element by n.
        let r = HadamardRotation { dim: 8 };
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut v = input.clone();
        r.apply(&mut v).unwrap();
        r.apply(&mut v).unwrap();
        let n = 8.0;
        for (a, b) in v.iter().zip(input.iter()) {
            assert!((a - n * b).abs() < 1e-4, "expected {} got {}", n * b, a);
        }
    }

    #[test]
    fn hadamard_rejects_non_pow2_dim() {
        let r = HadamardRotation { dim: 6 };
        let mut v = vec![0.0; 6];
        let err = r.apply(&mut v).unwrap_err();
        assert!(matches!(err, RotationError::HadamardNotPow2 { dim: 6 }));
    }

    #[test]
    fn hadamard_preserves_norm_squared_up_to_scale() {
        // ‖Hx‖² = n ‖x‖² for Sylvester Hadamard.
        let r = HadamardRotation { dim: 16 };
        let input: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        let norm_sq_in: f32 = input.iter().map(|x| x * x).sum();
        let mut v = input.clone();
        r.apply(&mut v).unwrap();
        let norm_sq_out: f32 = v.iter().map(|x| x * x).sum();
        let expected = 16.0 * norm_sq_in;
        let rel_err = (norm_sq_out - expected).abs() / expected;
        assert!(rel_err < 1e-5, "norm² out {norm_sq_out} vs expected {expected}");
    }

    #[test]
    fn opq_stub_returns_matrix_not_loaded() {
        let r = OpqRotationStub { matrix_blob_id: 0xDEAD_BEEF, dim: 4096 };
        let mut v = vec![0.0; 4096];
        let err = r.apply(&mut v).unwrap_err();
        assert!(matches!(err, RotationError::OpqMatrixNotLoaded { matrix_blob_id: 0xDEAD_BEEF }));
        assert_eq!(r.backend(), "stub");
    }

    #[test]
    fn build_identity() {
        let k = build(&Rotation::Identity, 256).unwrap();
        assert_eq!(k.dim(), 256);
        assert_eq!(k.backend(), "avx512");
    }

    #[test]
    fn build_hadamard() {
        let k = build(&Rotation::Hadamard { dim: 4096 }, 4096).unwrap();
        assert_eq!(k.dim(), 4096);
        assert_eq!(k.backend(), "avx512");
    }

    #[test]
    fn build_hadamard_rejects_mismatched_dim() {
        let err = build(&Rotation::Hadamard { dim: 4096 }, 2048).unwrap_err();
        assert!(matches!(err, RotationError::DimMismatch { expected: 4096, actual: 2048 }));
    }

    #[test]
    fn build_hadamard_rejects_non_pow2() {
        let err = build(&Rotation::Hadamard { dim: 100 }, 100).unwrap_err();
        assert!(matches!(err, RotationError::HadamardNotPow2 { dim: 100 }));
    }

    #[test]
    fn build_opq_returns_stub() {
        let k = build(&Rotation::Opq { matrix_blob_id: 42, dim: 4096 }, 4096).unwrap();
        assert_eq!(k.dim(), 4096);
        assert_eq!(k.backend(), "stub");
    }

    #[test]
    fn kernel_signatures_are_distinct_across_variants() {
        let id = IdentityRotation { dim: 256 };
        let had = HadamardRotation { dim: 256 };
        let opq = OpqRotationStub { matrix_blob_id: 1, dim: 256 };
        assert_ne!(id.signature(), had.signature());
        assert_ne!(id.signature(), opq.signature());
        assert_ne!(had.signature(), opq.signature());
    }

    #[test]
    fn kernel_signatures_stable_for_same_shape() {
        let a = HadamardRotation { dim: 4096 };
        let b = HadamardRotation { dim: 4096 };
        assert_eq!(a.signature(), b.signature());
    }

    #[test]
    fn opq_signature_depends_on_matrix_blob_id() {
        let a = OpqRotationStub { matrix_blob_id: 1, dim: 4096 };
        let b = OpqRotationStub { matrix_blob_id: 2, dim: 4096 };
        assert_ne!(a.signature(), b.signature());
    }
}
