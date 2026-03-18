// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Neighborhood Vector Search — Heel / Hip / Twig / Leaf
//!
//! The primary search path for lance-graph. Each node stores a neighborhood
//! vector of [`ZeckF64`] edge encodings (one `u64` per scope neighbor).
//! Search proceeds via L1 distance on these vectors, progressively loading
//! finer resolution bytes only when needed.
//!
//! ## ZeckF64 Encoding (8 bytes per edge)
//!
//! - **Byte 0 (scent):** 7 SPO band classifications + sign bit.
//!   Boolean lattice with ~40% built-in error detection.
//!   Alone achieves ρ ≈ 0.94 rank correlation with exact distance.
//!
//! - **Bytes 1–7 (resolution):** Distance quantiles within each SPO band
//!   (0 = identical, 255 = maximally different). Progressive: reading
//!   more bytes monotonically improves precision.
//!
//! ## Search Cascade
//!
//! | Stage | Name | Vectors Loaded | Explored | Latency |
//! |-------|------|---------------|----------|---------|
//! | 1     | HEEL | 1 × 10KB     | 10K      | ~20 µs  |
//! | 2     | HIP  | 50 × 10KB    | ~50K     | ~500 µs |
//! | 3     | TWIG | 50 × 10KB    | ~200K    | ~500 µs |
//! | 4     | LEAF | 50 cold loads | 50       | ~100 µs |

pub mod clam;
pub mod scope;
pub mod search;
pub mod sparse;
pub mod storage;
pub mod zeckf64;

pub use clam::{
    analyze_pareto_convergence, measure_cluster_radii, ParetoAnalysis, RadiusObservation,
};
pub use scope::{NeighborhoodVector, ScopeBuilder, ScopeMap};
pub use search::{HeelResult, SearchCascade, SearchConfig};
pub use sparse::ScentCsr;
pub use storage::{
    cognitive_nodes_schema, deserialize_scope_node_ids, neighborhoods_schema, scopes_schema,
    serialize_scent, serialize_resolution, serialize_scope_node_ids,
};
pub use zeckf64::{
    resolution, scent, zeckf64, zeckf64_distance, zeckf64_scent_distance,
    zeckf64_scent_hamming_distance,
};
