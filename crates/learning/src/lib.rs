//! Learning engine — 4096 CAM operations, cognitive styles, quantum ops,
//! structural causal models, dream consolidation, RL feedback.
//!
//! Standalone crate. Optional dependency for lance-graph.
//! Imported from AdaWorldAPI/ladybug-rs/src/learning/.

// ── Width constants ──────────────────────────────────────────────
pub const FINGERPRINT_BITS: usize = 16_384;
pub const FINGERPRINT_U64: usize = 256;
pub const FINGERPRINT_BYTES: usize = 2_048;
pub type Fingerprint = ndarray::hpc::fingerprint::Fingerprint<256>;
pub type Embedding = Vec<f32>;

// ── Error types ──────────────────────────────────────────────────
#[derive(Debug)]
pub enum Error {
    InvalidFingerprint { expected: usize, got: usize },
    DimensionMismatch { expected: usize, got: usize },
    Other(String),
}
pub type Result<T> = std::result::Result<T, Error>;

// ── Container/BindSpace stubs ────────────────────────────────────
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

pub mod storage {
    pub mod bind_space {
        pub type Addr = u64;
        pub struct BindSpace;
    }
}

// ── Core compat shim ─────────────────────────────────────────────
pub mod core {
    pub use super::Fingerprint;
    pub use super::Embedding;
    pub const DIM: usize = super::FINGERPRINT_BITS;
    pub const DIM_U64: usize = super::FINGERPRINT_U64;

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

    pub trait VsaOps: Sized {
        fn bind(&self, other: &Self) -> Self;
        fn unbind(&self, other: &Self) -> Self;
        fn bundle(items: &[Self]) -> Self;
        fn permute(&self, positions: i32) -> Self;
    }
}

// ── Modules (wip until rustynum refs are fully replaced) ─────────
#[cfg(feature = "wip")]
mod blackboard;
#[cfg(feature = "wip")]
mod cam_ops;
#[cfg(feature = "wip")]
mod causal_bridge;
#[cfg(feature = "wip")]
mod causal_ops;
#[cfg(feature = "wip")]
mod cognitive_frameworks;
#[cfg(feature = "wip")]
mod cognitive_styles;
#[cfg(feature = "wip")]
mod concept;
#[cfg(feature = "wip")]
mod dream;
#[cfg(feature = "wip")]
mod feedback;
#[cfg(feature = "wip")]
mod moment;
#[cfg(feature = "wip")]
mod quantum_ops;
#[cfg(feature = "wip")]
mod resonance;
#[cfg(feature = "wip")]
mod rl_ops;
#[cfg(feature = "wip")]
mod scm;
#[cfg(feature = "wip")]
mod session;
