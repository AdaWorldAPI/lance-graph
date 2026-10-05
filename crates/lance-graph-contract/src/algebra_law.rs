//! Planner-facing algebra metadata: the LAW a fold obeys, never the fold.
//!
//! A planner deciding whether it may regroup partial aggregates, permute
//! them, deduplicate them, or retract one needs to know which algebraic laws
//! hold. It does NOT need, and must not get, the operation itself.
//! [`AlgebraLaw`] carries the answers to those questions and nothing else.
//!
//! ```text
//! executor state ──describes──► AlgebraLaw ◄──reads── planner
//!      │
//!      └── owns identity value, merge, accumulator representation,
//!          physical fold
//! ```
//!
//! # Invariant: metadata, not execution
//!
//! Nothing in this module merges, folds, executes, or holds an identity
//! VALUE. There is no `merge(i64, i64)` here and there must never be one:
//! the accumulator behind a law may be a scalar today and a bitmap, vector,
//! structured statistic or mutation state later, and an executing trait in
//! the contract would freeze that representation. The concrete identity and
//! merge stay with the executor that owns the state (e.g.
//! `lance_graph_report::plan::FoldState::{identity, merge}`).
//!
//! This module also carries no generation or version: source authority is a
//! separate pattern (`SourceRef { id, generation }` in the report crate) and
//! a law is a property of the operation, not of any source it reads.
//!
//! Zero dependencies, `Copy`, no allocation.

/// The algebraic laws a fold's MERGE obeys, as the planner may rely on them.
///
/// Every flag has exactly one planner-facing meaning. A flag set to `true`
/// is a licence the planner may act on; when in doubt an implementation
/// reports `false`, which only forfeits a rewrite and is never unsound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AlgebraLaw {
    /// `merge(merge(a, b), c) == merge(a, merge(b, c))` under the operation's
    /// real execution semantics over its legal state domain.
    ///
    /// Licenses regrouping: partial aggregation, tree-shaped reduction,
    /// splitting a population into independently folded parts.
    pub associative: bool,

    /// `merge(a, b) == merge(b, a)` under legal state semantics.
    ///
    /// Licenses permuting independent partials (combining them in arrival
    /// order rather than a fixed order).
    pub commutative: bool,

    /// `merge(a, a) == a`.
    ///
    /// This is a property of merging two equal STATES. It is not a statement
    /// about duplicate INPUT rows (a duplicate row still changes a count),
    /// and it does not by itself license deduplicating contributions.
    pub idempotent: bool,

    /// `true` iff the result of combining contributions depends on their
    /// SEMANTIC contribution order.
    ///
    /// This is about the algebra only. It does NOT mean the source lane must
    /// be physically ordered; physical or source-order requirements belong to
    /// source / route / witness metadata (e.g. an ordered-lane attestation),
    /// never here. An order-insensitive operation reports `false`.
    pub ordered: bool,

    /// `true` iff the planner may REMOVE a previously accumulated
    /// contribution and recover the exact legal retained state without
    /// replaying the source population.
    ///
    /// Stronger than "a mathematical inverse exists": it claims an exposed,
    /// tested retraction capability on the executor side. Addition having
    /// subtraction does not set this flag; only a shipped remove path does.
    pub invertible: bool,

    /// Which KIND of identity element the merge has. A category, never a
    /// value: the concrete identity stays with the executor.
    pub identity_kind: IdentityKind,
}

/// The kind of identity element a merge has — a semantic category only.
///
/// This deliberately carries no `i64`, `u64`, `f32`, bytes or generic
/// accumulator value. The executor owns the concrete identity in whatever
/// representation its state uses; the planner only learns which kind of
/// "empty" an empty partial contributes. The enum names only the kinds the
/// currently described algebras need.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IdentityKind {
    /// The additive zero: an empty partial contributes nothing to a total
    /// (COUNT, SUM).
    Zero,
    /// The top of the state's order: the seed of a meet such as MIN. An
    /// empty MIN stays at this seed.
    Top,
    /// The bottom of the state's order: the seed of a join such as MAX, or
    /// "absent" for a presence fold such as EXISTS (OR over `false < true`).
    Bottom,
}

/// Implemented by an EXECUTOR's operator type to describe the law its own
/// merge obeys. The implementing crate keeps the merge; this only reports.
pub trait AlgebraDescriptor {
    /// The law this operator's merge obeys.
    fn algebra_law(&self) -> AlgebraLaw;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn law_is_plain_copyable_metadata() {
        let law = AlgebraLaw {
            associative: true,
            commutative: true,
            idempotent: false,
            ordered: false,
            invertible: false,
            identity_kind: IdentityKind::Zero,
        };
        let copy = law;
        assert_eq!(law, copy);
        // Three identity kinds, all distinct — Bottom is not a renamed Zero.
        assert_ne!(IdentityKind::Zero, IdentityKind::Bottom);
        assert_ne!(IdentityKind::Top, IdentityKind::Bottom);
        // Small enough to be a register microcopy, no hidden payload.
        assert!(core::mem::size_of::<AlgebraLaw>() <= 8);
    }

    /// Source invariant: the non-test part of this module defines no
    /// executing operation. A plain text scan, not reflection.
    #[test]
    fn module_defines_no_execution() {
        let src = include_str!("algebra_law.rs");
        let body = src.split("#[cfg(test)]").next().unwrap();
        for forbidden in [
            "fn merge",
            "fn fold",
            "fn execute",
            "fn identity",
            "fn combine",
        ] {
            assert!(
                !body.contains(forbidden),
                "algebra_law must not define `{forbidden}`"
            );
        }
    }
}
