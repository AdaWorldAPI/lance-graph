//! `lance-graph-relalg` — relational algebra over coordinate populations
//! (D-RA-1, `.claude/plans/relalg-v1.md`).

pub mod ir;
pub mod lower;
pub mod normalize;

pub use ir::{Atom, Cmp, Measure, Query, Rel};
pub use lower::{run, run_as_written, Answer, RelError, Run};
pub use normalize::{normalize, symbolic};
