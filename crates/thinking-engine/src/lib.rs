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

pub mod engine;
pub mod dto;
pub mod sensor;
pub mod layered;
pub mod branching;
pub mod bridge;
pub mod codebook_index;
pub mod l4;
#[cfg(feature = "tokenizer")]
pub mod lookup;
pub mod qualia;
pub mod domino;
pub mod jina_lens;
pub mod bge_m3_lens;
pub mod reranker_lens;
pub mod centroid_labels;
pub mod superposition;
pub mod cognitive_trace;
pub mod ghosts;
pub mod cognitive_stack;
pub mod meaning_axes;
pub mod contract_bridge;
pub mod persona;
pub mod awareness_dto;
pub mod world_model;
pub mod silu_correction;
pub mod signed_engine;
pub mod dual_engine;
pub mod l4_bridge;
pub mod composite_engine;
pub mod signed_domino;
pub mod pooling;
pub mod builder;
pub mod auto_detect;
pub mod semantic_chunker;
pub mod tensor_bridge;
#[cfg(feature = "tokenizer")]
pub mod tokenizer_registry;
