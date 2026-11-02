//! Codegraph queries and structural diagnostics over the ruff_charon store
//! artifacts (`*.rows.tsv`), folded through the xlog cube. This crate reads
//! the artifact format itself and has no dependency on ruff_charon.
#![forbid(unsafe_code)]

pub mod artifact;
pub mod graph;

pub use graph::{Codegraph, IntakeRefusal, Pos, RunStatus};
