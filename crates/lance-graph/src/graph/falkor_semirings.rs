// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Maps FalkorDB's most-used semirings to our palette compose tables.
//!
//! FalkorDB (via SuiteSparse:GraphBLAS) exposes ~960 scalar semirings over
//! CSC matrices. Our stack uses 7 HDR semirings over 16Kbit BitVec matrices.
//! This module maps the three most important FalkorDB semirings to our
//! equivalents, proving that the same algebraic structure applies.

use crate::graph::blasgraph::semiring::HdrSemiring;

/// FalkorDB semiring variants mapped to our HDR/palette operations.
///
/// Each variant documents how FalkorDB's scalar semiring maps to our
/// hyperdimensional vector semiring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalkorSemiring {
    /// Shortest path: compose=distance_add, reduce=min.
    ///
    /// FalkorDB: `GrB_MIN_PLUS_INT64` over scalar weights.
    /// Our equivalent: `HdrSemiring::HammingMin` — XOR multiply (Hamming
    /// distance accumulates), min-popcount reduce (shortest path wins).
    MinPlus,

    /// Reachability: compose=AND, reduce=OR.
    ///
    /// FalkorDB: `GrB_LOR_LAND_BOOL` over boolean scalars.
    /// Our equivalent: `HdrSemiring::Boolean` — AND multiply, OR add.
    /// Identical algebra, different element type (BitVec vs bool).
    OrAnd,

    /// HDR composition: compose=xor_bind, reduce=bundle.
    ///
    /// NOT in FalkorDB — this is our novel contribution.
    /// Composes hyperdimensional vectors via XOR binding and accumulates
    /// via majority-vote bundling. Enables semantic path composition
    /// that FalkorDB cannot express.
    XorBundle,
}

impl FalkorSemiring {
    /// Convert to our native HDR semiring.
    pub fn to_hdr_semiring(&self) -> HdrSemiring {
        match self {
            FalkorSemiring::MinPlus => HdrSemiring::HammingMin,
            FalkorSemiring::OrAnd => HdrSemiring::Boolean,
            FalkorSemiring::XorBundle => HdrSemiring::XorBundle,
        }
    }

    /// Human-readable name matching FalkorDB conventions.
    pub fn falkor_name(&self) -> &str {
        match self {
            FalkorSemiring::MinPlus => "GrB_MIN_PLUS (shortest path)",
            FalkorSemiring::OrAnd => "GrB_LOR_LAND (reachability)",
            FalkorSemiring::XorBundle => "HDR_XOR_BUNDLE (novel, not in FalkorDB)",
        }
    }

    /// Our native semiring name.
    pub fn hdr_name(&self) -> &str {
        match self {
            FalkorSemiring::MinPlus => "HAMMING_MIN",
            FalkorSemiring::OrAnd => "BOOLEAN",
            FalkorSemiring::XorBundle => "XOR_BUNDLE",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semiring_mapping() {
        assert_eq!(
            FalkorSemiring::MinPlus.to_hdr_semiring(),
            HdrSemiring::HammingMin
        );
        assert_eq!(
            FalkorSemiring::OrAnd.to_hdr_semiring(),
            HdrSemiring::Boolean
        );
        assert_eq!(
            FalkorSemiring::XorBundle.to_hdr_semiring(),
            HdrSemiring::XorBundle
        );
    }

    #[test]
    fn test_falkor_names() {
        assert!(FalkorSemiring::MinPlus.falkor_name().contains("GrB_MIN_PLUS"));
        assert!(FalkorSemiring::OrAnd.falkor_name().contains("GrB_LOR_LAND"));
        assert!(FalkorSemiring::XorBundle.falkor_name().contains("novel"));
    }
}
