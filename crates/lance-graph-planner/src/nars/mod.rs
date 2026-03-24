//! # P1: NARS Schema + Thinking
//!
//! **NARS Schema** = truth values stored as edge properties in the adjacency store.
//! **NARS Thinking** = inference rules that OPERATE on adjacent truth values.
//!
//! Each inference type = a specific semiring for adjacent_truth_propagate().

pub mod truth;
pub mod inference;

pub use truth::TruthValue;
pub use inference::NarsInference;
