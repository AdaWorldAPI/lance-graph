//! **LAB, D-DR-SELECT-K-1.** One versioned selection, bound once, read
//! through the physical views that already exist.
//!
//! ```text
//! Source (version V, N rows)         owns the values; the only "storage"
//!   select_filter / select_ordered   produces the selection, once
//!   take                             stages K rows, counted
//! Bound (K local rows)               the one binding: rows, order witness,
//!                                    staged K-local lanes
//! Residual                           a further predicate over K, held as
//!                                    ALL, positions or a K-bit mask
//! ops                                count, sum, ordered top-T over K
//! ```
//!
//! Plan: `.claude/plans/dr-select-k-v1.md` §4. The question is which
//! admissible view is cheaper for an operation and a reuse count; it is not
//! "storage against resident".
//!
//! Three counters are kept apart ([`Counters`]): storage accesses, binds and
//! operand rows read by folds. After the bind the storage counter does not
//! move; a fold reads the staged K-local lanes, which is operand access.
//!
//! `u128` stands in for `NodeGuid`. Nothing here is a contract type.

#![forbid(unsafe_code)]

pub mod bind;
pub mod counters;
pub mod ops;
pub mod oracle;
pub mod source;

pub use bind::{Bound, Order, Residual, View};
pub use counters::Counters;
pub use ops::{admissible, Inadmissible, Op, PreparedFolds};
pub use source::{KeyRange, Selection, Source, Version};
