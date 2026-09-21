//! # `lance-graph-mask-risc` — the mask RISC above `ndarray::simd`
//!
//! **STATUS: EXECUTOR LANDED (PR3).** [`ir`] is the vocabulary, [`exec`] the
//! borrowing evaluator (one facade delegation per op over caller-owned
//! [`exec::Scratch`]), [`reference`] the row-at-a-time oracle that never
//! touches `ndarray`, [`fuse`] the Boolean-tree → ternlog fuser, and
//! [`ternlog_dispatch`] the GENERATED 256-arm bridge from a runtime
//! immediate to the const-generic facade word. The differential suite
//! (`tests/differential.rs`) diffs executor against oracle, a claim that is
//! **backend-independent by construction** — this crate names no ISA (A3,
//! enforced by `the_crate_names_no_isa`), so there is no per-realization
//! behaviour here to cover. `ndarray` IS the SIMD polyfill: which backend a
//! word lowers to is its question, answered by its own parity tests, and
//! running this suite under another realization would test ndarray through a
//! proxy rather than test this crate. `tests/no_alloc.rs` pins the
//! zero-allocation law. `hop` (PR5) LANDED as [`MaskOp::Gather`] (the fk
//! semijoin, over [`ir::Foreign`]) and [`Terminal::ScatterOrU32`] (the
//! one-to-many hop back); [`Terminal::GroupSumI32`] is the one-terminal
//! `GROUP BY … SUM`. Still absent, named: a strided `Operand` — the gap is
//! in THIS IR, not in T1: `ndarray::simd` already ships
//! `ternary_match_strided_to_mask`, `eq_u32_strided_to_mask` and
//! `masked_strided_group_sum`, and nothing here can name a `(base, stride,
//! group)` source; a via-key group-sum ([`ndarray::simd::masked_group_sum_i32_via`],
//! `SUM(...) GROUP BY partner.country`); and the Cypher `mask_lower` seam
//! (the Cypher plan's Wave 1 consumes this crate).
//!
//! One duplication is filed rather than resolved here: `lgj-abi` already
//! carries its own runtime-immediate → const-generic ternlog bridge
//! (`simd_mask_ternlog_assign_dyn`, ABI minor ≥ 11). The two agree on the
//! index convention, but PR4 must pick ONE owner — either lgj-abi delegates
//! to [`ternlog_dispatch`] or this module is scoped to the evaluator.
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
//! ## The four ARCHITECTURAL laws (A1-A4)
//!
//! These four are the crate's doctrine. They are NOT the same list as
//! [`exec`]'s **L1-L5**, which are the narrower *structural* laws — the ones
//! a test enforces. The two lists overlap but do not share numbering, and
//! naming them apart is the point: A3 and L2 are both "no ISA", so a bare
//! "law 3" or "law 2" is ambiguous unless the prefix is written. Cite `A3`
//! for the doctrine and `L2` for the test that holds it up
//! (`the_crate_names_no_isa`). Mapping: A1 → L1 + L5, A3 → L2, A4 → L4;
//! A2 has no structural counterpart, and L3 (one delegation per op) has no
//! architectural one — it is `[claimed, unverified]` with no instrument.
//!
//! A1. **The plan describes, the executor borrows, ndarray computes, the
//!    caller owns memory.** Inputs are `&[u64]` / `&[i32]` / … borrowed from
//!    whoever owns the address space (a mailbox, an `AlphaMask`, an lgj
//!    `RowStore`); temporaries live in a caller-supplied `Scratch`; the
//!    executor never allocates. There is no per-row object, no hidden
//!    rowset, no second row-index universe — a `SelectionVector` cannot be
//!    expressed in this vocabulary at all.
//! A2. **Masks choose admissibility; magnitude is a separate reduction over
//!    survivors** (`E-TOPOLOGY-MASKS-MAGNITUDE-COMPOSE-NEVER-COLLAPSE-1`).
//!    [`Terminal::MaskedSumI32`] and friends reduce a *value plane* under
//!    the final mask; a mask bit is never a weight.
//! A3. **`TERNLOG` is semantics, never a hardware assumption.** The fuser
//!    (`fuse`) turns any Boolean subtree over three leaves into one
//!    [`MaskOp::Ternlog`] by evaluating its truth table; how a given
//!    immediate is realized on AVX-512 / AVX2 / NEON / WASM / scalar is
//!    entirely `ndarray`'s business (the polyfill law). **This crate contains
//!    no `cfg(target_feature)`, no ISA cost model, no fallback chain.**
//! A4. **Reference semantics are independent.** `reference` evaluates the
//!    same program one row at a time in plain Rust with no `ndarray` — the
//!    oracle every executor is diffed against, on whichever backend the test
//!    binary is built for (see the status note above; not all five).
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

pub use exec::{execute, execute_into, materialize_rows, scratch_words_for, Scratch};
pub use fuse::{fuse, fuse_program, ternlog_imm, BoolExpr, FuseError, Fused};
pub use ir::{
    Foreign, ForeignPlane, LaneRef, MaskOp, Operand, Planes, Pred, Program, Terminal,
    MASKED_SUM_I32_MAX_ROWS, MAX_SCRATCH_SLOTS,
};
pub use reference::{
    reference_execute, reference_execute_into, reference_scratch, reference_scratch_with_foreign,
};
pub use ternlog_dispatch::{ternlog_dispatch, ternlog_dispatch_assign};
pub use value::{ExecError, LaneKind, Out, Value};

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
