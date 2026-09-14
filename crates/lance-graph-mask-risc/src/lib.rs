//! # `lance-graph-mask-risc` — the mask RISC above `ndarray::simd`
//!
//! A tiny mechanical evaluator and fuser for Boolean programs over
//! **borrowed resident bit-planes** (one `u64` per 64 rows, LSB-first, tail
//! bits zero — `ndarray::simd`'s normative mask order and the contract's
//! [`AlphaMask`](lance_graph_contract::alpha::AlphaMask) order alike).
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
//!    `RowStore`); temporaries live in a caller-supplied [`Scratch`]; the
//!    executor never allocates. There is no per-row object, no hidden
//!    rowset, no second row-index universe — a `SelectionVector` cannot be
//!    expressed in this vocabulary at all.
//! 2. **Masks choose admissibility; magnitude is a separate reduction over
//!    survivors** (`E-TOPOLOGY-MASKS-MAGNITUDE-COMPOSE-NEVER-COLLAPSE-1`).
//!    [`Terminal::MaskedSumI32`] and friends reduce a *value plane* under
//!    the final mask; a mask bit is never a weight.
//! 3. **`TERNLOG` is semantics, never a hardware assumption.** The fuser
//!    ([`fuse`]) turns any Boolean subtree over three leaves into one
//!    [`MaskOp::Ternlog`] by evaluating its truth table; how a given
//!    immediate is realized on AVX-512 / AVX2 / NEON / WASM / scalar is
//!    entirely `ndarray`'s business (the polyfill law). **This crate contains
//!    no `cfg(target_feature)`, no ISA cost model, no fallback chain.**
//! 4. **Reference semantics are independent.** [`reference`] evaluates the
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
//! carry-safe for every `i32` input, matching DuckDB's never-wrap contract).
//! Nothing else — no strings, no dictionaries, no ORDER BY, no bag-semantics
//! joins.

#![forbid(unsafe_code)]

// Only the IR exists in this skeleton. `exec` (the borrowing executor),
// `fuse` (predicate → one ternlog chain), `hop` (src_mask → edge lane →
// dst_mask), `reference` (the scalar oracle) and `ternlog_table` (the
// generated truth tables) are PR3's deliverables and are declared when they
// land — a `mod` line for a file that is not on disk made this crate fail to
// build, which both the DuckDB matrix and the Cypher lowering plan recorded.
pub mod ir;

pub use ir::{LaneRef, MaskOp, Operand, Planes, Pred, Program, Terminal};

/// Number of `u64` words a mask over `n_rows` occupies.
#[inline]
pub fn words_for(n_rows: usize) -> usize {
    n_rows.div_ceil(64)
}
