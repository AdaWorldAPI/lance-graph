//! The fold path: mask-risc and the two frontends that lower onto it.
//!
//! - [`mask_risc`]: the bulk execution engine. Programs over resident mask
//!   planes and value lanes; compiled once, run over a row extent.
//! - [`quack`]: the set/aggregate frontend. A `Query` (filter + aggregate)
//!   lowers to one mask-risc program.
//! - [`r2il`]: the procedure frontend. R2IL calls emit mask-risc programs
//!   through `FoldDialect`; NAL operations are procedures over it.
//!
//! These are re-exports, not wrappers: core adds no second algebra over them.
//! A capability that is missing here is missing in one of the three crates.

pub use lance_graph_mask_risc as mask_risc;
pub use lance_graph_quack as quack;
pub use lance_graph_r2il as r2il;
