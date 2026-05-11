//! InferenceBackend — runtime-switchable dispatch across codec/inference paths.
//!
//! Design principle: every research path coexists as a backend variant. Nothing
//! is killed. The R&D bench runs all backends against the same input and produces
//! a comparison table. Deprecation is data-driven (bench results), not
//! opinion-driven.
//!
//! Python scripts are REFERENCE IMPLEMENTATIONS for cross-checking and HF
//! download. The Rust backends here are the canonical inference paths.
//!
//! ## Backend families (two axes)
//!
//! **Axis 1 — full-path vs leaf-only quantization:**
//!
//! | Approach | What it quantizes | Per-row cost | Quality model |
//! |---|---|---|---|
//! | Full-path QJL/PolarQuant | Entire row → JL-projected sign+magnitude | ~20 B | Inner-product preservation via Lindenstrauss |
//! | Leaf-only (I8 hybrid) | HEEL+HIP location (6 bit) + i8 JLQ on residual only | 9 B | Location finds neighborhood; JLQ corrects fine-grained |
//! | Passthrough | No quantization | 2×n_cols B | Exact |
//!
//! **Axis 2 — reconstruction-grade vs signature-grade:**
//!
//! | Grade | What you can do with the output | Backends |
//! |---|---|---|
//! | Reconstruction | Feed into f32 GEMM, get exact-ish logits | SafetensorsRaw, BurnFwd, CandleFwd, HhtlF32+SlotL |
//! | Signature | Compare pairwise (cosine/Hamming), route via cascade | RaBitQ, SpiralEncoding, CodecCascade, Base17 |
//! | Hybrid | Location-grade routing + reconstruction-grade residual | I8Hybrid (HEEL+HIP + JLQ leaf) |
//!
//! The bench must test ALL combinations because we don't yet know which
//! (family × grade) cell wins per regime. That's the point of the R&D.

/// Encoded state produced by a backend. Opaque to the consumer —
/// only the backend that produced it can score/reconstruct from it.
pub enum EncodedState {
    /// Raw f32 rows (passthrough / candle / burn forward pass output)
    F32Rows(Vec<Vec<f32>>),
    /// Binary sign-quantized (RaBitQ: binary[] + norm + dot_correction per row)
    RaBitQ {
        encodings: Vec<bgz17::rabitq_compat::RaBitQEncoding>,
        rotation: bgz17::rabitq_compat::OrthogonalMatrix,
    },
    /// Spiral signature (K anchors × 17 dims per row)
    Spiral {
        encodings: Vec<highheelbgz::rehydrate::SpiralEncoding>,
    },
    /// HEEL+HIP location + i8 JLQ leaf residual (I8 hybrid from invariant I8)
    I8Hybrid {
        /// 6-bit location address per row (basin 2b + family 4b)
        locations: Vec<u8>,
        /// Per-location-bin f32 centroid
        centroids: Vec<Vec<f32>>,
        /// 8 × i8 JL-projected residual per row
        leaves: Vec<[i8; 8]>,
        /// Shared Hadamard rotation matrix (seeded, deterministic)
        rotation_seed: u64,
        /// Per-row magnitude (BF16 as u16)
        magnitudes: Vec<u16>,
    },
    /// Base17 + HhtlDTensor (cascade lookup grade, not reconstruction)
    HhtlD {
        tensor: bgz_tensor::hhtl_d::HhtlDTensor,
    },
    /// f32 CLAM palette + optional SlotL
    HhtlF32 {
        tensor: bgz_tensor::hhtl_f32::HhtlF32Tensor,
    },
    /// Codec cascade state (hhtl_cache routing decisions precomputed)
    Cascade {
        cache: bgz_tensor::hhtl_cache::HhtlCache,
        assignments: Vec<u8>,
    },
}

/// What a backend can do.
pub trait InferenceBackend: Send + Sync {
    fn name(&self) -> &str;

    /// Encode rows from f32 source. Each backend stores what it needs.
    fn encode(&self, rows: &[Vec<f32>], n_cols: usize) -> EncodedState;

    /// Pairwise score between two encoded rows (cosine-like, higher = more similar).
    /// Returns None if the backend doesn't support pairwise scoring (reconstruction-only).
    fn score(&self, state: &EncodedState, i: usize, j: usize) -> Option<f64>;

    /// Reconstruct row i to f32.
    /// Returns None if the backend is signature-only (no reconstruction).
    fn reconstruct(&self, state: &EncodedState, i: usize, n_cols: usize) -> Option<Vec<f32>>;

    /// Per-row byte cost (excluding shared overhead).
    fn bytes_per_row(&self) -> usize;

    /// Shared overhead bytes (palette, rotation matrix, SVD basis — amortised over row count).
    fn shared_overhead_bytes(&self, n_rows: usize, n_cols: usize) -> usize;

    /// Which quality grade this backend operates at.
    fn grade(&self) -> BackendGrade;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendGrade {
    /// Can reconstruct rows for f32 GEMM inference
    Reconstruction,
    /// Can compare pairwise (cosine/Hamming) but not reconstruct
    Signature,
    /// Location-grade routing + reconstruction-grade residual
    Hybrid,
}

// ═════════════════════════════════════════════════════════════════════
// Backend implementations (stubs — each gets filled in the R&D bench)
// ═════════════════════════════════════════════════════════════════════

pub struct PassthroughBackend;
pub struct RaBitQBackend { pub dim: usize }
pub struct SpiralBackend { pub k: usize, pub start: u32, pub stride: u32 }
pub struct I8HybridBackend { pub n_bins: usize }
pub struct HhtlF32Backend { pub palette_k: usize }
pub struct CascadeBackend { pub palette_k: usize }
pub struct Base17SignatureBackend;

/// Registry: all available backends for the R&D bench.
/// Feature-gated where deps are heavy.
pub fn all_backends(n_cols: usize) -> Vec<Box<dyn InferenceBackend>> {
    vec![
        Box::new(PassthroughBackend),
        Box::new(RaBitQBackend { dim: n_cols }),
        Box::new(SpiralBackend { k: 8, start: 0, stride: 3 }),
        Box::new(I8HybridBackend { n_bins: 64 }),
        Box::new(HhtlF32Backend { palette_k: 256 }),
        Box::new(CascadeBackend { palette_k: 256 }),
        Box::new(Base17SignatureBackend),
    ]
}
