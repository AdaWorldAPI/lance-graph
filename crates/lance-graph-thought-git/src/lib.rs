//! D-TGB-7: thought git as a shared, admitted capability
//! (`.claude/plans/thought-git-basics-v1.md`, `thought-git-lifecycle-contract-v1.md`).
//!
//! W0 is the canonical resident state; W1..=63 are reusable resident thought
//! spaces, generation-qualified so a reused slot is never read as the thought
//! that lived in it before. Branching, changing, diffing, squashing,
//! discarding and replaying are resident operations; only an admitted change
//! crosses the durable boundary, and it lands through the one canonical writer
//! (`LanceCycleWriter`), never a second one.
//!
//! This crate is filled by migration from `crates/thought-git-probe`, one slice
//! at a time, each slice moving code rather than copying it:
//!
//! - [`slots`]: the slot pool (D-TGB-1), moved in slice S1.
//! - [`spaces`]: the resident thought spaces (D-HL-5/7): sparse per-slot XOR
//!   deltas over one borrowed base, merged on read, with saccades as alpha
//!   overlay claims. Moved in slice S2 from
//!   `cognitive-shader-driver/examples/thought_layers_probe.rs`.
//! - `oracle` and `fixtures` (feature `oracle`): the from-scratch reference and
//!   the pre-registered fixtures (D-TGB-0). They are the independent oracle
//!   every implementation is checked against, not execution, so they are off
//!   by default.
//!
//! The givens layer, the squash gate,
//! the receipt and the landing follow in later slices; the inventory is on
//! EXO-379.

#[cfg(feature = "oracle")]
pub mod fixtures;
#[cfg(feature = "oracle")]
pub mod oracle;
pub mod slots;
pub mod spaces;
