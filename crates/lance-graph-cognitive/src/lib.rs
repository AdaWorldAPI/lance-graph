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

// Modules imported from ladybug-rs (adaptation needed)
// These are source-imported, not yet fully wired to ndarray/holograph.
// Compile-gating happens at the module level — each mod.rs controls
// what's publicly exposed.

// Grammar triangle: SPO × causality × qualia
#[cfg(feature = "wip")]
pub mod grammar;

// Learning: quantum ops, SCM, dream consolidation
#[cfg(feature = "wip")]
pub mod learning;

// SPO extensions: cognitive codebook, crystals, gestalt
#[cfg(feature = "wip")]
pub mod spo;

// World model: counterfactual reasoning
#[cfg(feature = "wip")]
pub mod world;
