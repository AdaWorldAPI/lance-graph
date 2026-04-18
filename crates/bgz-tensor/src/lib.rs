//! # bgz-tensor: Metric-Algebraic Tensor Codec
//!
//! Not quantization — computation compilation.
//!
//! ## The Thesis
//!
//! Transformer weight quantization (TurboQuant, GPTQ, AWQ, GGUF Q4_K_M) makes
//! the NUMBERS smaller but keeps the OPERATION unchanged: multiply, accumulate,
//! activate. You're compressing operands but running the same matmul.
//!
//! bgz-tensor replaces the operation. Weight matrices are projected through
//! golden-step folding into a 17-dimensional metric space with algebraic
//! structure (distance + compose). Every attention score becomes a u16 table
//! lookup. Every multi-hop composition becomes a u8 table lookup. The matmul
//! collapses into addressing.
//!
//! ## Architecture
//!
//! ```text
//! Weight matrix W (d_model × d_head)         64 MB per matrix
//!   │
//!   ▼ projection.rs: golden-step folding
//! Base17 patterns (d_head × 34 bytes)         136 KB (470× smaller)
//!   │
//!   ▼ palette.rs: CLAM manifold clustering
//! 256 archetypes + assignments                8.5 KB codebook + N bytes indices
//!   │
//!   ▼ attention.rs: precompute ALL pairs
//! Distance table (256 × 256 × u16)           128 KB (fits L1 cache)
//! Compose table (256 × 256 × u8)             64 KB
//!   │
//!   ▼ cascade.rs: HHTL progressive elimination
//! Inference: 95% of pairs skipped at Layer 0-1
//! Remaining 5%: one table lookup each
//! ```
//!
//! ## Comparison
//!
//! | | TurboQuant/GPTQ/AWQ | bgz-tensor |
//! |---|---|---|
//! | What's compressed | Weight values | Weight computation |
//! | Inference kernel | matmul (cuBLAS) | table lookup (L1 cache) |
//! | Bits per weight | 2-4 bits | 8 bits (palette index) |
//! | Bytes per matrix | 4-8 MB (Q4_K_M) | 12.5 KB (palette + indices) |
//! | Attention score | O(d) multiply-adds | O(1) table lookup |
//! | Multi-hop | Stack attention layers | O(1) compose table |
//! | Sparsity | Learned (BigBird etc.) | Metric-induced (triangle ineq) |
//! | Hardware | GPU (tensor cores) | CPU (L1 cache) |
//!
//! ## Quality Targets
//!
//! Measured as Pearson correlation ρ between palette-compiled attention
//! scores and ground-truth dot-product attention:
//!
//! - ρ > 0.95 → paper quality (publishable)
//! - ρ > 0.99 → product quality (deployable)
//!
//! bgz17 achieves ρ = 0.992 for general distance preservation.
//! The question is whether attention-specific distance (dot product similarity)
//! preserves as well as generic L1 distance.

pub mod adaptive_codec;
pub mod attention;
pub mod belichtungsmesser;
pub mod cascade;
pub mod codebook4096;
pub mod codebook_calibrated;
pub mod euler_fold;
pub mod fisher_z;
pub mod gamma_calibration;
pub mod gamma_phi;
pub mod had_cascade;
pub mod holographic_residual;
pub mod hdr_belichtung;
pub mod hhtl_cache;
pub mod hhtl_d;
pub mod hhtl_f32;
pub mod jina;
pub mod neuron_hetero;
pub mod palette;
pub mod projection;
pub mod shared_palette;
pub mod slot_l;
pub mod quality;
pub mod similarity;
pub mod stacked;
pub mod stacked_n;
pub mod turboquant_kv;
pub mod variance;
pub mod xor_adaptive;

#[cfg(feature = "hydrate")]
pub mod manifest;
pub mod matryoshka;

// ─── Re-exports ──────────────────────────────────────────────────────────────

pub use attention::{AttentionSemiring, AttentionTable, CompiledHead, ComposeTable};
pub use belichtungsmesser::{Band, Belichtungsmesser};
pub use codebook4096::{Codebook4096, CodebookIndex};
pub use stacked::{StackedBF16x4, SearchKey17, VedicCascadeConfig};
pub use stacked_n::{StackedN, ClamCodebook};
pub use hdr_belichtung::{PaletteCascade, NdarrayCascade, NdarrayBand, ShiftAlert};
pub use hhtl_d::{HhtlDEntry, HhtlDTensor, HeelBasin, HhtlDMeta};
pub use cascade::{CascadeConfig, CascadeLevel, CascadeStats};
pub use fisher_z::{FisherZTable, FamilyGamma};
pub use had_cascade::{HadCascadeTensor, HadCascadeRow, TensorRegime};
pub use palette::WeightPalette;
pub use projection::{Base17, Base17Fz};
pub use quality::QualityReport;
pub use similarity::SimilarityTable;
pub use variance::RoleVarianceReport;
pub use matryoshka::{MatryoshkaRow, SvdBasis, BandProfile, BandPrecision};
pub use turboquant_kv::{TurboQuantKvCache, TurboQuantEntry, KvCacheStats};
