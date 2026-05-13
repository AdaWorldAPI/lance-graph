//! # 16M RISC Thought Engine
//!
//! 4096² distance table × energy vector = thought.
//!
//! ```text
//! A perturbation enters.
//! 4096 thought-atoms activate simultaneously.
//! Each atom composes with all 4096 others = 16M paths.
//! Interference: similar paths reinforce, contradictions cancel.
//! After N cycles, 3-5 peaks dominate.
//! The strongest peak crystallizes into explicit thought.
//! ```
//!
//! Models are SENSORS. The matrix is the BRAIN. DTOs are the BUS.

pub mod auto_detect;
pub mod awareness_dto;
pub mod bf16_engine;
pub mod bge_m3_lens;
pub mod branching;
pub mod bridge;
pub mod builder;
pub mod centroid_labels;
pub mod codebook_index;
pub mod cognitive_stack;
pub mod cognitive_trace;
pub mod composite_engine;
pub mod contract_bridge;
pub mod contrastive_learner;
pub mod cronbach;
pub mod domino;
pub mod dto;
pub mod dual_engine;
pub mod engine;
pub mod f32_engine;
pub mod ghosts;
pub mod ground_truth;
pub mod jina_lens;
pub mod l4;
pub mod l4_bridge;
pub mod layered;
#[cfg(feature = "tokenizer")]
pub mod lookup;
pub mod meaning_axes;
pub mod osint_bridge;
pub mod persona;
pub mod pooling;
pub mod prime_fingerprint;
pub mod qualia;
pub mod reencode_safety;
pub mod reranker_lens;
pub mod role_tables;
pub mod semantic_chunker;
pub mod sensor;
pub mod signed_domino;
pub mod signed_engine;
pub mod silu_correction;
pub mod spiral_segment;
pub mod superposition;
pub mod tensor_bridge;
#[cfg(feature = "tokenizer")]
pub mod tokenizer_registry;
pub mod world_model;
// ripple.rs deleted: wave simulation wrong, replaced by VSA bundle in prime_fingerprint.rs

// PR-F1 — CognitiveBridgeGate: cross-tenant authorization injection point.
// No lance-graph-callcenter dep. PassthroughGate is the standalone default.
pub mod bridge_gate;
