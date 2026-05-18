//! # four-repo-demo
//!
//! Minimal end-to-end integration demo for the lance-graph workspace.
//!
//! ## What this crate demonstrates
//!
//! Three of the four glue crates described in
//! `.claude/plans/integration-plan.md` are exercised together:
//!
//! | Glue | Crate | Demonstrated here |
//! |------|-------|-------------------|
//! | #4 | `cognitive-shader-actor` | [`SumShader`] wrapped as a `ractor::Actor` via `CognitiveShaderActor` |
//! | IR  | `lance-graph-contract::ir` | 3-node `OperatorTree` (RangeScan→Filter→CognitiveApply) |
//! | #4 contract | `lance-graph-contract::actor` | `SupervisableShader` implemented by [`SumShader`] |
//!
//! ## What is NOT demonstrated (and why)
//!
//! See [`README.md`] for the full list.  In brief:
//!
//! - **Glue #1 `surrealdb-ractor`** — requires a running SurrealDB instance
//!   with the `surrealdb-ractor` crate wired in.  That crate lives in the
//!   `AdaWorldAPI/surrealdb` repo and is not yet integrated.
//! - **Glue #2 `lance-graph-tikv-provider`** — requires a live TiKV cluster
//!   (`make tikv-up`).  The provider crate is scaffolded but not built here.
//! - **Glue #3 `sea-orm-ractor`** — requires a running Postgres and the
//!   `sea-orm-ractor` crate from `AdaWorldAPI/sea-orm`.
//!
//! [`README.md`]: ../README.md
//! [`SumShader`]: crate::cognitive::SumShader

/// Running-sum [`SupervisableShader`] implementation.
///
/// [`SupervisableShader`]: lance_graph_contract::actor::SupervisableShader
pub mod cognitive;

/// Planner IR demo — builds a 3-node [`OperatorTree`] and computes cardinality.
///
/// [`OperatorTree`]: lance_graph_contract::ir::OperatorTree
pub mod planner;

pub use cognitive::SumShader;
pub use planner::{build_demo_tree, node_count, print_plan};
