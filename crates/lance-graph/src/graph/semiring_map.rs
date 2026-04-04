// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Semiring map: maps standard GraphBLAS semirings to our HDR equivalents.
//!
//! GraphBLAS (SuiteSparse) exposes ~960 scalar semirings over CSC matrices.
//! Our stack uses 7 HDR semirings over 16Kbit BitVec matrices.
//! This module maps the three most important GraphBLAS semirings to our
//! equivalents, proving that the same algebraic structure applies.
//!
//! Methods harvested from GraphBLAS literature. No external database dependency.

use crate::graph::blasgraph::semiring::HdrSemiring;

/// Standard GraphBLAS semiring variants mapped to our HDR/palette operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphBlasSemiring {
    /// Shortest path: compose=distance_add, reduce=min.
    /// GraphBLAS: `GrB_MIN_PLUS_INT64`.
    /// Ours: `HdrSemiring::HammingMin`.
    MinPlus,

    /// Reachability: compose=AND, reduce=OR.
    /// GraphBLAS: `GrB_LOR_LAND_BOOL`.
    /// Ours: `HdrSemiring::Boolean`.
    OrAnd,

    /// HDR composition: compose=xor_bind, reduce=bundle.
    /// Novel — not in standard GraphBLAS.
    XorBundle,
}

impl GraphBlasSemiring {
    /// Convert to our native HDR semiring.
    pub fn to_hdr_semiring(&self) -> HdrSemiring {
        match self {
            GraphBlasSemiring::MinPlus => HdrSemiring::HammingMin,
            GraphBlasSemiring::OrAnd => HdrSemiring::Boolean,
            GraphBlasSemiring::XorBundle => HdrSemiring::XorBundle,
        }
    }

    /// Standard GraphBLAS name for this semiring.
    pub fn graphblas_name(&self) -> &str {
        match self {
            GraphBlasSemiring::MinPlus => "GrB_MIN_PLUS (shortest path)",
            GraphBlasSemiring::OrAnd => "GrB_LOR_LAND (reachability)",
            GraphBlasSemiring::XorBundle => "HDR_XOR_BUNDLE (novel, not in standard GraphBLAS)",
        }
    }

    /// Our native semiring name.
    pub fn hdr_name(&self) -> &str {
        match self {
            GraphBlasSemiring::MinPlus => "HAMMING_MIN",
            GraphBlasSemiring::OrAnd => "BOOLEAN",
            GraphBlasSemiring::XorBundle => "XOR_BUNDLE",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semiring_mapping() {
        assert_eq!(GraphBlasSemiring::MinPlus.to_hdr_semiring(), HdrSemiring::HammingMin);
        assert_eq!(GraphBlasSemiring::OrAnd.to_hdr_semiring(), HdrSemiring::Boolean);
        assert_eq!(GraphBlasSemiring::XorBundle.to_hdr_semiring(), HdrSemiring::XorBundle);
    }

    #[test]
    fn test_graphblas_names() {
        assert!(GraphBlasSemiring::MinPlus.graphblas_name().contains("GrB_MIN_PLUS"));
        assert!(GraphBlasSemiring::OrAnd.graphblas_name().contains("GrB_LOR_LAND"));
        assert!(GraphBlasSemiring::XorBundle.graphblas_name().contains("novel"));
    }
}
