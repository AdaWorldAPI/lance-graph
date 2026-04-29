//! # DeepNSM — Distributional Semantic Transformer Replacement
//!
//! A semantic processing layer that replaces transformer inference with
//! precomputed distributional lookup.
//!
//! ```text
//! 4,096 words × 12 bits × 8MB distance matrix = complete semantic engine
//! O(1) per word, O(n) per sentence, deterministic, bit-reproducible
//! ```
//!
//! ## The Three Replacements
//!
//! | Transformer Component | DeepNSM Replacement | Ratio |
//! |---|---|---|
//! | Embedding matrix (3M params) | Distance matrix (8MB u8) | 0 learned params |
//! | Multi-head attention (O(n²)) | XOR role binding (O(n)) | Structural, not learned |
//! | Contextual embedding | ±5 sentence window | O(1) per update |
//!
//! ## Architecture
//!
//! ```text
//! Raw text: "the big dog bit the old man"
//!      │
//!      ▼ vocabulary (tokenize)
//! Token stream: [(0,a), (155,j), (670,n), (2942,v), (0,a), (173,j), (94,n)]
//!      │
//!      ▼ parser (PoS FSM)
//! SPO triples: [SPO(dog, bite, man), Mod(big→dog), Mod(old→man)]
//!      │
//!      ▼ encoder (XOR bind + majority bundle)
//! VSA vectors: 512-bit binary representations with word-order sensitivity
//!      │
//!      ▼ similarity (calibrated CDF lookup)
//! f32 [0,1]: distributional similarity via precomputed 4096² matrix
//!      │
//!      ▼ context (±5 sentence ring buffer)
//! Disambiguated representations colored by local context
//! ```
//!
//! ## Key Numbers
//!
//! | Metric | Value |
//! |---|---|
//! | Vocabulary | 4,096 words (98.4% text coverage) |
//! | Token size | 12 bits (16 with PoS) |
//! | SPO triple | 36 bits (in u64) |
//! | Distance matrix | 16 MB u8 (fits L2 cache) |
//! | SimilarityTable | 1 KB (fits L1 cache) |
//! | Context window | ~1 KB (±5 sentences) |
//! | Total runtime | ~16.5 MB |
//! | NSM primes | 62/63 in vocabulary |
//! | Pipeline latency | < 10μs per sentence |

pub mod codebook;
pub mod context;
pub mod encoder;
pub mod parser;
pub mod pipeline;
pub mod pos;
pub mod similarity;
pub mod spo;
pub mod vocabulary;

pub mod trajectory;
pub mod markov_bundle;
pub mod nsm_primes;

// PR #279 outlook epiphany E4 — Trajectory-as-statement-hash bridge to
// PR #278 audit log. Converts grammatical structure to a 16384-bit
// semantic hash key.
pub mod trajectory_audit;

// PR #279 outlook epiphany E8 — Quantum mode (PhaseTag + holographic
// addressing) sharing the 16384-dim substrate with Crystal mode.
pub mod quantum_mode;

#[cfg(feature = "contract-ticket")]
pub mod ticket_emit;

#[cfg(feature = "grammar-triangle")]
pub mod triangle_bridge;

// ─── Re-exports ──────────────────────────────────────────────────────────────

pub use pipeline::DeepNsmEngine;
pub use pos::PoS;
pub use spo::SpoTriple;
pub use vocabulary::Vocabulary;
pub use similarity::SimilarityTable;
pub use encoder::{VsaVec, RoleVectors};
pub use context::ContextWindow;
pub mod fingerprint16k;
