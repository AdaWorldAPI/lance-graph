//! Grammar — tiered routing types and reference tables.
//!
//! Local grammar matches (deepnsm + rule tables) handle 90-99% of parsing
//! at sub-10µs. The remainder produces a structured [`FailureTicket`] that
//! an LLM fallback can resolve surgically — never the full sentence, only
//! the ambiguous slots.
//!
//! ## Modules
//!
//! - [`ticket`]        — [`FailureTicket`], partial parse, causal ambiguity.
//! - [`tekamolo`]      — Temporal/Kausal/Modal/Lokal adverbial slots.
//! - [`wechsel`]       — Dual-role tokens (prepositions, pronouns, etc.).
//! - [`finnish`]       — 15 Finnish cases as direct slot lookups.
//! - [`inference`]     — NARS inference × thinking style routing table.
//! - [`context_chain`] — Markov ±5 replay for disambiguation.

pub mod ticket;
pub mod tekamolo;
pub mod wechsel;
pub mod finnish;
pub mod inference;
pub mod context_chain;
pub mod role_keys;
pub mod thinking_styles;

pub use ticket::{FailureTicket, PartialParse, CausalAmbiguity};
pub use tekamolo::{TekamoloSlots, TekamoloSlot};
pub use wechsel::{WechselAmbiguity, WechselRole};
pub use finnish::{FinnishCase, finnish_case_for_suffix};
pub use inference::{NarsInference, inference_to_style_cluster};
pub use context_chain::{
    ContextChain, DisambiguationResult, ReplayDirection, ReplayRequest,
    WeightingKernel, CHAIN_LEN, DISAMBIGUATION_MARGIN_THRESHOLD, MARKOV_RADIUS,
};
pub use role_keys::*;
pub use thinking_styles::{
    CoveragePolicy, GrammarStyleAwareness, GrammarStyleConfig, MarkovPolicy,
    MorphologyPolicy, MorphologyTableId, NarsPriorityChain, ParamKey,
    ParseOutcome, ReplayStrategy, SpoCausalPolicy, TekamoloPolicy, revise_truth,
};

/// Coverage of a local parse — if below [`LOCAL_COVERAGE_THRESHOLD`],
/// the ticket is emitted for LLM fallback.
pub const LOCAL_COVERAGE_THRESHOLD: f32 = 0.9;
