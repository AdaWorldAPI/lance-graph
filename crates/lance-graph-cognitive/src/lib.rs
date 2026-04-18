//! lance-graph-cognitive — cognitive extensions from ladybug-rs.
//!
//! Optional crate. Not required by core lance-graph or bgz-tensor.
//! Imported from AdaWorldAPI/ladybug-rs, adapted to use ndarray +
//! holograph + lance-graph-contract as foundations instead of
//! ladybug-rs's own core types.
//!
//! # Modules
//!
//! - **grammar**: causality flow, 18D qualia, grammar triangle, NSM primes
//! - **learning**: quantum ops, structural causal models, dream consolidation
//! - **spo**: cognitive codebook, sentence crystal, context crystal, gestalt
//! - **world**: counterfactual reasoning, world state
//!
//! # Note on imports
//!
//! These modules were imported from ladybug-rs and may reference
//! `crate::core::Fingerprint` or other ladybug-rs types. They need
//! adaptation passes to wire to ndarray/holograph types. Modules that
//! don't compile yet are behind `#[cfg(feature = "wip")]` — the
//! Cargo.toml exposes them as opt-in until adaptation is complete.

// ── Width constants (unified 16K production) ──────────────────────
// All imported modules reference these via `crate::FINGERPRINT_BITS` etc.
// Matches ndarray::hpc::fingerprint::VectorWidth::W16K.

/// Fingerprint width in bits (16,384 = production).
pub const FINGERPRINT_BITS: usize = 16_384;
/// Fingerprint width in u64 words (256).
pub const FINGERPRINT_U64: usize = 256;
/// Fingerprint width in bytes (2,048).
pub const FINGERPRINT_BYTES: usize = 2_048;

/// Re-export ndarray's const-generic Fingerprint as the canonical type.
pub type Fingerprint = ndarray::hpc::fingerprint::Fingerprint<256>;

/// Dense embedding vector (ladybug-rs compat).
pub type Embedding = Vec<f32>;

/// Error type (ladybug-rs compat).
#[derive(Debug)]
pub enum Error {
    InvalidFingerprint { expected: usize, got: usize },
    DimensionMismatch { expected: usize, got: usize },
    Other(String),
}
pub type Result<T> = std::result::Result<T, Error>;

/// Stub for Container (not yet ported — see BINDSPACE_MIGRATION_GAP.md).
pub mod container {
    pub type Container = crate::Fingerprint;
    pub mod record {
        #[derive(Clone, Debug)]
        pub struct CogRecord {
            pub content: super::Container,
            pub meta: super::Container,
        }
        impl CogRecord {
            pub fn new(content: super::Container, meta: super::Container) -> Self {
                Self { content, meta }
            }
        }
    }
}

/// Stub for storage/BindSpace (not yet ported).
pub mod storage {
    pub mod bind_space {
        pub type Addr = u64;
        pub struct BindSpace;
    }
}

/// Compatibility module mirroring ladybug-rs `crate::core::*`.
/// Modules reference `crate::core::Fingerprint`, `crate::core::rustynum_accel::*`, etc.
pub mod core {
    pub use super::Fingerprint;
    pub use super::Embedding;
    pub const DIM: usize = super::FINGERPRINT_BITS;
    pub const DIM_U64: usize = super::FINGERPRINT_U64;

    /// Compatibility shim for rustynum SIMD acceleration.
    /// Maps to ndarray::hpc::bitwise.
    pub mod rustynum_accel {
        pub fn hamming_distance(a: &[u8], b: &[u8]) -> u64 {
            ndarray::hpc::bitwise::hamming_distance_raw(a, b)
        }
        pub fn slice_hamming(a: &[u64], b: &[u64]) -> u64 {
            let a_bytes: Vec<u8> = a.iter().flat_map(|w| w.to_le_bytes()).collect();
            let b_bytes: Vec<u8> = b.iter().flat_map(|w| w.to_le_bytes()).collect();
            ndarray::hpc::bitwise::hamming_distance_raw(&a_bytes, &b_bytes)
        }
        pub fn batch_hamming(query: &[u8], database: &[u8], vec_len: usize) -> Vec<u64> {
            let n = database.len() / vec_len;
            (0..n).map(|i| {
                let start = i * vec_len;
                ndarray::hpc::bitwise::hamming_distance_raw(query, &database[start..start + vec_len])
            }).collect()
        }
        pub fn simd_level() -> &'static str { "ndarray" }
    }

    /// VSA operations trait (ladybug-rs compat).
    pub trait VsaOps: Sized {
        fn bind(&self, other: &Self) -> Self;
        fn unbind(&self, other: &Self) -> Self;
        fn bundle(items: &[Self]) -> Self;
        fn permute(&self, positions: i32) -> Self;
    }
}

// Modules imported from ladybug-rs (adaptation needed)
// These are source-imported, not yet fully wired to ndarray/holograph.
// Compile-gating happens at the module level — each mod.rs controls
// what's publicly exposed.

// Grammar triangle: SPO × causality × qualia
pub mod grammar;

// Learning: moved to standalone crate `crates/learning/` (optional dep)
// 16 modules, 300K+ LOC. Use: `learning = { path = "../learning" }`

// SPO extensions: cognitive codebook, crystals, gestalt
#[cfg(feature = "wip")]
pub mod spo;

// World model: counterfactual reasoning
pub mod world;
