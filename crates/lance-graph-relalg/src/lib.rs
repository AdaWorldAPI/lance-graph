//! `lance-graph-relalg` — relational algebra over coordinate populations
//! (D-RA-1, `.claude/plans/relalg-v1.md`).
//!
//! A PROTOTYPE for D-CMR (`.claude/plans/d-cmr-concept-chain-normalization-v1.md`):
//! rearrange the description of a query with proven identities before anything
//! runs, and lower only the irreducible remainder.
//!
//! # What it holds
//!
//! - [`Rel`]: a set of rows of one table, described — `All`, `Empty`, a
//!   resident plane, a comparison on one lane, `And` / `Or` / `Not`. In D-CMR
//!   terms every comparison is a residual (`Res`): the lanes are not a proven
//!   row ordering, so no row interval (`Runs`) is formed here.
//! - [`normalize()`]: the rewrites, to a fixpoint (see [`mod@normalize`]'s
//!   module doc). The interval work is constraint propagation over VALUE intervals of
//!   one lane — narrow bounds, detect `∅` — not row intervals.
//! - [`symbolic`]: a description that normalizes to `Empty` (or `All`, for a
//!   count) is answered without running a program.
//! - [`run`]: lowering into R2IL calls over
//!   [`FoldDialect`](lance_graph_r2il::FoldDialect), which builds and runs one
//!   mask-risc program per fold. This crate never loops over rows.
//!
//! # What it is not
//!
//! Not a third frontend syntax beside Quack's `Filter` and NAL's R2IL
//! programs, and not the home of the rule catalogue: D-CMR puts the set
//! algebra in `lance-graph-contract`. The rules and tests here are the
//! evidence for that move; recursion (`μ`) and the hop (preimage) are not
//! built.

pub mod ir;
pub mod join_body;
pub mod lower;
pub mod normalize;

pub use ir::{Atom, Cmp, Measure, Query, Rel};
pub use lower::{run, run_as_written, Answer, RelError, Run};
pub use normalize::{normalize, symbolic};
