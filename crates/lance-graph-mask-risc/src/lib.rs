//! # `lance-graph-mask-risc` — the mask RISC above `ndarray::simd`
//!
//! **STATUS: EXECUTOR LANDED (PR3).** [`ir`] is the vocabulary, [`exec`] the
//! borrowing evaluator (one facade delegation per op over caller-owned
//! [`exec::Scratch`]), [`reference`] the row-at-a-time oracle that never
//! touches `ndarray`, [`fuse`] the Boolean-tree → ternlog fuser, and
//! [`ternlog_dispatch`] the GENERATED 256-arm bridge from a runtime
//! immediate to the const-generic facade word. The differential suite
//! (`tests/differential.rs`) diffs executor against oracle on every backend;
//! `tests/no_alloc.rs` pins the zero-allocation law. Still absent, named:
//! `hop` (PR5), the strided operand family (an `ndarray` T1 gap), and the
//! Cypher `mask_lower` seam (the Cypher plan's Wave 1 consumes this crate).
//!
//! A tiny mechanical evaluator and fuser for Boolean programs over
//! **borrowed resident bit-planes** (one `u64` per 64 rows, LSB-first, tail
//! bits zero — `ndarray::simd`'s normative mask order and the contract's
//! `AlphaMask` order alike).
//!
//! ```text
//! DuckDB execution semantics            (source material only — never a dependency)
//!         ↓ reverse-engineered
//! this crate's op vocabulary            (ir.rs: predicates, mask algebra, terminals)
//!         ↓ borrows
//! V3 resident SoA / caller-owned scratch (exec.rs: nothing allocated per row, ever)
//!         ↓ lowers to
//! ndarray::simd primitives              (the ISA membrane; five compile-time realizations)
//! ```
//!
//! ## The four laws this crate is built to make structural
//!
//! 1. **The plan describes, the executor borrows, ndarray computes, the
//!    caller owns memory.** Inputs are `&[u64]` / `&[i32]` / … borrowed from
//!    whoever owns the address space (a mailbox, an `AlphaMask`, an lgj
//!    `RowStore`); temporaries live in a caller-supplied `Scratch`; the
//!    executor never allocates. There is no per-row object, no hidden
//!    rowset, no second row-index universe — a `SelectionVector` cannot be
//!    expressed in this vocabulary at all.
//! 2. **Masks choose admissibility; magnitude is a separate reduction over
//!    survivors** (`E-TOPOLOGY-MASKS-MAGNITUDE-COMPOSE-NEVER-COLLAPSE-1`).
//!    [`Terminal::MaskedSumI32`] and friends reduce a *value plane* under
//!    the final mask; a mask bit is never a weight.
//! 3. **`TERNLOG` is semantics, never a hardware assumption.** The fuser
//!    (`fuse`) turns any Boolean subtree over three leaves into one
//!    [`MaskOp::Ternlog`] by evaluating its truth table; how a given
//!    immediate is realized on AVX-512 / AVX2 / NEON / WASM / scalar is
//!    entirely `ndarray`'s business (the polyfill law). **This crate contains
//!    no `cfg(target_feature)`, no ISA cost model, no fallback chain.**
//! 4. **Reference semantics are independent.** `reference` evaluates the
//!    same program one row at a time in plain Rust with no `ndarray` — the
//!    oracle every executor is diffed against, on every backend.
//!
//! ## Which DuckDB semantics this vocabulary emulates (exactly, and only these)
//!
//! Comparison of `i32`/`u32` lanes against a constant (`=`, `<>`, `<`, `<=`,
//! `>`, `>=`, signed for `i32`, exact bitwise for `u32`), two-valued
//! conjunction/disjunction/negation of predicate results (there is no NULL:
//! absence in the V3 substrate is a zero-fallback, never a validity bit, so
//! DuckDB's three-valued AND/OR collapses to Boolean algebra), `CASE`-shaped
//! conditional selection ([`Terminal::BlendI32`]), and the aggregates
//! `COUNT`, `EXISTS`, `MIN`, `MAX`, `SUM` (the latter widened to `i64` —
//! carry-safe for every `i32` input up to [`ir::MASKED_SUM_I32_MAX_ROWS`] rows,
//! beyond which an executor rejects rather than wraps — DuckDB's never-wrap
//! contract, with the bound stated instead of assumed).
//! Nothing else — no strings, no dictionaries, no ORDER BY, no bag-semantics
//! joins.

#![forbid(unsafe_code)]

// `hop` (src_mask → edge lane → dst_mask) is PR5's and is declared when it
// lands — a `mod` line for a file that is not on disk made this crate fail to
// build once, which both the DuckDB matrix and the Cypher lowering plan
// recorded.
pub mod exec;
pub mod fuse;
pub mod ir;
pub mod reference;
pub mod ternlog_dispatch;
pub mod value;

pub use exec::{execute, materialize_rows, Scratch};
pub use fuse::{fuse, fuse_program, ternlog_imm, BoolExpr, FuseError, Fused};
pub use ir::{LaneRef, MaskOp, Operand, Planes, Pred, Program, Terminal, MASKED_SUM_I32_MAX_ROWS};
pub use reference::{reference_execute, reference_scratch};
pub use ternlog_dispatch::{ternlog_dispatch, ternlog_dispatch_assign};
pub use value::{ExecError, LaneKind, Value};

/// Number of `u64` words a mask over `n_rows` occupies.
#[inline]
pub fn words_for(n_rows: usize) -> usize {
    n_rows.div_ceil(64)
}

#[cfg(test)]
mod tests {
    use super::words_for;

    /// FAILS IF: `words_for` floors instead of ceils, or counts a word for
    /// zero rows — the boundaries a `/ 64` or a `+ 1` would each get wrong.
    #[test]
    fn words_for_rounds_up_to_whole_words_and_zero_rows_need_none() {
        assert_eq!(words_for(0), 0);
        assert_eq!(words_for(1), 1);
        assert_eq!(words_for(63), 1);
        assert_eq!(words_for(64), 1);
        assert_eq!(words_for(65), 2);
        assert_eq!(words_for(128), 2);
        assert_eq!(words_for(129), 3);
    }
}
