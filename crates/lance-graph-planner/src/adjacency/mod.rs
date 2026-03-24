//! # P0: Adjacency Substrate
//!
//! **THE GROUND.** Everything else operates ON this.
//!
//! Kuzu's killer feature: adjacency is a first-class columnar operation.
//! When you say "traverse A→B→C," you don't walk pointers. You batch all A's
//! adjacent edges into a vector, intersect with B's adjacency vector, then
//! intersect with C's. Vectorized. One pass.
//!
//! ## What lives here
//! - CSR/CSC compressed adjacency lists (columnar, morsel-compatible)
//! - `batch_adjacent()` — the core primitive
//! - `intersect_adjacent()` — Kuzu's WCO join primitive
//! - Edge properties (NARS truth values live here)
//! - `adjacent_fingerprint_distance()` — similarity ON adjacent pairs only
//! - `adjacent_truth_propagate()` — semiring ops ON the adjacency batch
//!
//! ## Future: VSA Focus-of-Awareness
//! Each node's adjacency pattern encoded as a 10K Hamming vector.
//! High similarity with query = "this node's neighborhood is relevant."
//! Prefilter before actual traversal. Attention for graphs.

pub mod csr;
pub mod batch;
pub mod properties;
pub mod distance;
pub mod propagate;

pub use csr::AdjacencyStore;
pub use batch::AdjacencyBatch;
pub use properties::EdgeProperties;
