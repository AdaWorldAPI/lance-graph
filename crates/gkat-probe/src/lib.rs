//! # D-GKAT-0 — PROBE: bounded GKAT checking of structured Rubicon flows
//!
//! **This is a research fixture, not a verifier.** It is not used by any
//! runtime, has no dependencies, and executes nothing. It answers three
//! questions about small structured flows:
//!
//! 1. **Equivalence** ([`automaton::equivalent`]): do two flows have the same
//!    guarded traces? Decided by bisimulation of normalized guarded Thompson
//!    automata (Smolka et al., *Guarded Kleene Algebra with Tests*, POPL 2020,
//!    arXiv:1907.05920, §§4–5).
//! 2. **Safety** ([`automaton::commit_safe`]): is every `RequestCommit`
//!    immediately preceded by an atom with a valid revision and MUL
//!    authorization? This is a separate check: a flow equivalent to an unsafe
//!    flow is unsafe.
//! 3. **Refusal** ([`flow`]): constructs outside the admitted fragment
//!    (unguarded choice, `GoTo`, `GoBack`, more than [`expr::MAX_TESTS`]
//!    primitive tests, undeclared projections) are refused with a typed
//!    [`expr::Refusal`], never translated.
//!
//! ## Proof boundary
//!
//! * **What the paper proves.** For GKAT expressions over a fixed finite set
//!   of atoms, two expressions denote the same guarded strings iff their
//!   normalized Thompson automata are bisimilar (§5), and the check runs in
//!   `O(n α(n))` for a fixed atom set (Theorem 5.10). With an unbounded
//!   number of tests the problem is co-NP-hard (Proposition 5.11).
//! * **What this crate does.** It implements that procedure over an
//!   explicitly enumerated atom set of at most 256 atoms. The implementation
//!   is **tested, not proven**: every verdict in the test suite is
//!   cross-checked against [`oracle`], an independent interpreter that
//!   exhaustively explores all runs up to a **stated bound** on the number of
//!   actions. That bound makes the oracle a one-directional check (see its
//!   module documentation); it is not a completeness argument.
//! * **What is outside.** The paper's relational model interprets actions as
//!   uninterpreted state relations. Versions, snapshot and codebook staleness,
//!   SQL three-valued logic beyond a declared projection, bag counts, side
//!   effects and the Lance write are not modelled. `RequestCommit` is an
//!   intent; the write belongs to the supervisor / `cycle_driver`.
//! * **No timings.** Under the workspace rule against millisecond results,
//!   the probe reports counts (atoms, states, merged pairs), never times.

pub mod automaton;
pub mod expr;
pub mod flow;
pub mod oracle;
pub mod rubicon;

pub use automaton::{Equivalence, Safety, commit_safe, equivalent};
pub use expr::{Action, Atom, Atoms, BExp, Exp, MAX_TESTS, Refusal};
