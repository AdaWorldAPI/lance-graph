//! # GraphBLAS for HDR - Sparse XOR Adjacency with Arrow Backend
//!
//! A Rust implementation of the GraphBLAS API using hyperdimensional
//! computing primitives. Instead of numeric linear algebra, we use:
//!
//! - **XOR binding** for matrix "multiplication"
//! - **Majority bundling** for "addition"
//! - **Hamming distance** for comparison
//! - **Sparse Arrow storage** for efficient graph representation
//!
//! ## GraphBLAS Mapping to HDR
//!
//! ```text
//! GraphBLAS Operation    HDR Equivalent
//! ─────────────────────  ──────────────────────────────────
//! C = A ⊕.⊗ B            XOR-bind traversal with bundle accumulator
//! mxm (matrix multiply)   Multi-hop binding: A ⊗ B
//! vxm (vector × matrix)   Query expansion via binding
//! reduce                   Bundle all row/column vectors
//! apply                    Per-element transformation
//! eWiseAdd                 Elementwise bundle (majority)
//! eWiseMult                Elementwise bind (XOR)
//! ```
//!
//! ## Semirings for HDR
//!
//! | Name | "Multiply" | "Add" | Use Case |
//! |------|------------|-------|----------|
//! | XOR_BUNDLE | XOR | Majority | Path composition |
//! | XOR_FIRST | XOR | First | Traversal |
//! | HAMMING_MIN | Hamming | Min | Shortest path |
//! | BIND_RESONANCE | Bind | Best match | Query expansion |

mod descriptor;
mod matrix;
mod ops;
mod semiring;
#[cfg(feature = "datafusion-storage")]
mod sparse;
pub mod types;
mod vector;

pub use descriptor::{Descriptor, GrBDesc};
pub use matrix::GrBMatrix;
pub use ops::*;
pub use semiring::{HdrSemiring, Semiring};
#[cfg(feature = "datafusion-storage")]
pub use sparse::{CooStorage, CsrStorage, SparseFormat};
pub use types::*;
pub use vector::GrBVector;

use crate::{HdrError, Result};

/// GraphBLAS info codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum GrBInfo {
    Success = 0,
    NoValue = 1,
    InvalidValue = 2,
    InvalidIndex = 3,
    DomainMismatch = 4,
    DimensionMismatch = 5,
    OutputNotEmpty = 6,
    OutOfMemory = 7,
    InvalidObject = 8,
    NullPointer = 9,
}

impl From<GrBInfo> for Result<()> {
    fn from(info: GrBInfo) -> Self {
        match info {
            GrBInfo::Success => Ok(()),
            GrBInfo::NoValue => Err(HdrError::Query("No value".into())),
            _ => Err(HdrError::Query(format!("GraphBLAS error: {:?}", info))),
        }
    }
}

/// Initialize the GraphBLAS context
pub fn grb_init() -> GrBInfo {
    // In Rust, no special initialization needed
    GrBInfo::Success
}

/// Finalize the GraphBLAS context
pub fn grb_finalize() -> GrBInfo {
    GrBInfo::Success
}

/// Get library version
pub fn grb_version() -> (u32, u32, u32) {
    (2, 0, 0) // GraphBLAS 2.0 compatible
}
