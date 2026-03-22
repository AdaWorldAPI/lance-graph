//! # bgz17: Blasgraph-ZeckBF17 Unified Distance Codec
//!
//! Palette-indexed SPO search via HHTL. Combines:
//! - **blasgraph**: GraphBLAS semiring algebra on 16Kbit BitVecs
//! - **ZeckBF17**: 17-dimensional golden-step octave compression (i16 base)
//! - **SPO**: three-plane Subject/Predicate/Object decomposition
//! - **HHTL**: Heel→Hip→Twig→Leaf progressive search cascade
//!
//! ## The Architecture
//!
//! Each knowledge edge has three SPO planes. Each plane's 16,384-dimensional
//! accumulator is compressed to an i16[17] base pattern (34 bytes, ρ=0.992).
//! The 256 most common base patterns per plane form a **palette** (codebook).
//! Each edge becomes 3 bytes (u8 index per S/P/O). Distance between edges
//! is a single lookup in a precomputed 256×256 matrix per plane.
//!
//! ## Layered Distance Codec
//!
//! ```text
//! Layer 0: Scent (1 byte)     — Boolean lattice, ρ=0.937
//! Layer 1: Palette (3 bytes)  — matrix lookup, ρ=0.965 (k=128) to 0.992 (k=256)
//! Layer 2: ZeckBF17 (102 bytes) — full i16[17] base L1, ρ=0.992
//! Layer 3: Full planes (6 KB)  — exact Hamming, ρ=1.000
//! ```
//!
//! 95%+ of searches terminate at Layer 0-1 (CAKES triangle inequality pruning).
//! Layer 2 used for decision-boundary cases. Layer 3 almost never loaded.

pub mod base17;
pub mod palette;
pub mod distance_matrix;
pub mod tripartite;
pub mod layered;
pub mod scalar_sparse;
pub mod scope;
pub mod bridge;
pub mod generative;
pub mod prefetch;
pub mod clam_bridge;
pub mod router;
pub mod container;
pub mod palette_semiring;
pub mod palette_matrix;
pub mod palette_csr;
pub mod simd;
pub mod rabitq_compat;

/// Maximum palette size per plane.
pub const MAX_PALETTE_SIZE: usize = 256;

/// Base dimensionality (prime, golden-step covers all residues).
pub const BASE_DIM: usize = 17;

/// Full accumulator dimensionality.
pub const FULL_DIM: usize = 16384;

/// Golden-ratio step for dimension traversal.
pub const GOLDEN_STEP: usize = 11;

/// Fixed-point scale for i16 base encoding.
pub const FP_SCALE: f64 = 256.0;

/// Precision levels for distance computation.
///
/// ## Metric Safety
///
/// CAKES DFS sieve requires a true metric (triangle inequality must hold).
/// - **Scent**: NOT metric-safe. The 19-pattern Boolean lattice breaks
///   triangle inequality. Use ONLY as heuristic pre-filter (HEEL stage).
/// - **Palette**: Metric-safe. L1 on i16[17] is a metric. Safe for CAKES pruning.
/// - **Base**: Metric-safe. L1 on i16[17] is a metric. Safe for CAKES pruning.
/// - **Exact**: Metric-safe. Hamming distance is a metric.
///
/// Rule: any function that feeds CAKES `delta_minus` / `delta_plus` bounds
/// MUST use Palette or higher. Scent is for heuristic pre-filtering only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Precision {
    /// Layer 0: scent byte Hamming (1 byte, ρ=0.937).
    /// ⚠️ NOT metric-safe — Boolean lattice breaks triangle inequality.
    /// Use ONLY for heuristic pre-filtering (HEEL stage), never for CAKES pruning.
    Scent,
    /// Layer 1: palette matrix lookup (3 bytes, ρ=0.965).
    /// ✓ Metric-safe — L1 satisfies triangle inequality.
    Palette,
    /// Layer 2: full i16[17] base L1 (102 bytes, ρ=0.992).
    /// ✓ Metric-safe.
    Base,
    /// Layer 3: exact Hamming on full planes (6 KB, ρ=1.000).
    /// ✓ Metric-safe.
    Exact,
}
