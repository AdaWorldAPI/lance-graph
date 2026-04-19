//! Crystal — polymorphic semantic crystallizations (shared interface).
//!
//! A Crystal is a structured semantic object that accumulates truth (NARS
//! revision), hardens over time, and supports bundle/unbundle operations.
//!
//! ## Relationship to existing crystal/quantum crates
//!
//! Implementations live in the siblings: `ladybug-rs`,
//! `ada-consciousness`, and `bighorn` already ship crystal/quantum
//! crates. This module is the **contract surface** — the trait and
//! layout they all implement against. No logic lives here; only shared
//! types, sandwich-layout constants, and the [`Crystal`] trait.
//!
//! Downstream VSA algebra (bind / bundle / permute / similarity) is
//! the canonical `ndarray::hpc::vsa` module on binary 10K vectors.
//! Contract-level types ([`CrystalFingerprint`]) carry the storage
//! format; the consumer crate picks the VSA operator.
//!
//! ## Crystal hierarchy
//!
//! ```text
//! SentenceCrystal   — one parsed sentence, triples + tekamolo slots
//! ContextCrystal    — Markov ±5 window around a sentence
//! DocumentCrystal   — full document, composed of sentence crystals
//! CycleCrystal     — one cognitive cycle (observe → act → feedback)
//! SessionCrystal   — full conversation / agent session
//! ```
//!
//! All crystals share the [`Crystal`] trait: hardness, revision count,
//! crystallized-at timestamp, and a polymorphic [`CrystalFingerprint`].

pub mod fingerprint;
pub mod sentence;
pub mod context;
pub mod document;
pub mod cycle;
pub mod session;

pub use fingerprint::{CrystalFingerprint, Structured5x5, Quorum5D};
pub use sentence::SentenceCrystal;
pub use context::ContextCrystal;
pub use document::DocumentCrystal;
pub use cycle::CycleCrystal;
pub use session::SessionCrystal;

/// The kind of crystal — used for dispatch and policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CrystalKind {
    Sentence,
    Context,
    Document,
    Cycle,
    Session,
}

/// NARS truth value — frequency (evidence ratio) + confidence (sample size).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TruthValue {
    pub frequency: f32,
    pub confidence: f32,
}

impl TruthValue {
    pub const fn new(frequency: f32, confidence: f32) -> Self {
        Self { frequency, confidence }
    }
}

/// Common trait across all crystal kinds.
pub trait Crystal {
    fn kind(&self) -> CrystalKind;

    /// Hardness ∈ [0, 1]. Accumulates via NARS revision as evidence stacks.
    /// Crosses the unbundling threshold (~0.8) → promote to individually
    /// addressable facts in episodic memory.
    fn hardness(&self) -> f32;

    /// Number of NARS revisions that have folded into this crystal.
    fn revision_count(&self) -> u32;

    /// Crystallization timestamp (Unix seconds).
    fn crystallized_at(&self) -> u64;

    /// The polymorphic fingerprint carrying this crystal's semantic content.
    fn fingerprint(&self) -> &CrystalFingerprint;

    /// NARS truth value at the current revision.
    fn truth(&self) -> TruthValue;
}

/// Threshold above which a crystal is considered "hardened" — ready to
/// unbundle from the young bundled form into individually addressable facts.
pub const UNBUNDLE_HARDNESS_THRESHOLD: f32 = 0.8;
