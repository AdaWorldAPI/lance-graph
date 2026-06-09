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

pub mod markov_bundle;
pub mod nsm_primes;
pub mod trajectory;

// E-ENGLISH-BIFURCATES — two SEPARATE faculties (don't fuse them):
//   arcs (Broca/projection): basin/literal decomposition of the MarkovBundler wave.
//   comprehension (Wernicke): literal sentence resolution + fact/story router,
//     tokenless, reading SentenceStructure — NOT the projection band.
// Hippocampus (episodic story-arc + consolidation) is downstream/agnostic.
// See .claude/knowledge/english-fact-story-bifurcation-grail-v1.md.
pub mod arcs;
pub mod comprehension;

// E-ARCUATE-CONDUCTION: the arcuate fasciculus — owns the MarkovBundler
// producer + the ±5 ContextChain ring and slides the projection into it, so
// the Broca↔Wernicke cable carries signal. Separate seam; NOT wired into
// pipeline.rs's live ContextWindow (that coexistence is a distinct decision).
pub mod arcuate;

// Loose-end-#2 closer (PR-G3): glue from MarkovBundler::role_bundle()
// → ContextChain::disambiguate_with(.., DisambiguateOpts {
// sentinel_fp }). Closes the "real fp" honesty gap by giving the
// contract crate a caller that actually constructs a non-zero
// `CrystalFingerprint::Binary16K` from an f32 trajectory bundle.
pub mod disambiguator_glue;

// PR #279 outlook epiphany E4 — Trajectory-as-statement-hash bridge to
// PR #278 audit log. Converts grammatical structure to a 16384-bit
// semantic hash key.
pub mod trajectory_audit;

// PR #279 outlook epiphany E8 — Quantum mode (PhaseTag + holographic
// addressing) sharing the 16384-dim substrate with Crystal mode.
pub mod quantum_mode;

#[cfg(feature = "contract-ticket")]
pub mod ticket_emit;

// PR-G1: module always compiled — Pearl mask computation and
// analyze_without_triangle are core. Only GrammarTriangle-dependent
// code inside is gated behind #[cfg(feature = "grammar-triangle")].
pub mod triangle_bridge;

// ─── Re-exports ──────────────────────────────────────────────────────────────

pub use context::ContextWindow;
pub use encoder::{RoleVectors, VsaVec};
pub use pipeline::DeepNsmEngine;
pub use pos::PoS;
pub use similarity::SimilarityTable;
pub use spo::SpoTriple;
pub use vocabulary::Vocabulary;
pub mod fingerprint16k;

// ── DeepNSM reader — sentence-level AriGraph reader ──────────────────────
// Left-corner state machine: expectation + evidence → episodic SPO + next state.
// Five modules; writer order is dependency order.
pub mod cam64; // Cam64 — 8-lane reading-state locality key (NOT the truth)
pub mod episodic_spo; // EpisodicSpoFrame — the auditable witness rows
pub mod morphology; // MorphFlags — heuristic tense/voice/clause flags
pub mod reader_state;
pub mod window; // SentenceWindow — ±5 exact entity tracking for coreference // ReadingState + step() — the left-corner transition
                // Signed discrete reading-crystal: P64 meaning field + Crystal4096 coordinate.
                // Bridge from DeepNSM grammar reader → holograph bitpacked resonance substrate.
                // Integer-only hot path; floats remain only in EpisodicSpoFrame quality fields.
pub mod signed_crystal;
// SentenceTransformer64 — deterministic state-transition transformer.
// Maps grammar/NSM/discourse → P64 native meaning field → Cam4096 codebook address.
// "Transformer" = state-transition automaton, NOT neural self-attention.
// P64 is the native address space; Cam4096 is its deterministic 12-bit locality key.
pub mod sentence_transformer64;
// L1 local geometry for Crystal4096: neighbors_4096(), chebyshev/manhattan distance,
// NeighborhoodMetric (Manhattan / Chebyshev / LaneCompatible). No floats.
// blasgraph (L2) will consume these for frontier propagation in v2.
pub mod crystal_neighborhood;
