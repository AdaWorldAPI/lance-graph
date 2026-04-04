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
