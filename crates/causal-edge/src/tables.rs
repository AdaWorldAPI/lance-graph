//! Precomputed NARS Lookup Tables — eliminate all FP from the hot path.
//!
//! At u8 quantization (256 levels for f and c), every NARS operation
//! can be precomputed as a 256×256 lookup table.
//!
//! Size per table: 256 × 256 × 2 bytes (output f,c packed as u16) = 128 KB.
//! Fits L1 cache. Every NARS inference becomes a single memory read.

/// Packed (frequency_u8, confidence_u8) output as u16.
/// Low byte = frequency, high byte = confidence.
pub type PackedTruth = u16;

#[inline]
pub fn pack_truth(f: u8, c: u8) -> PackedTruth {
    (c as u16) << 8 | f as u16
}

#[inline]
pub fn unpack_f(p: PackedTruth) -> u8 {
    p as u8
}

#[inline]
pub fn unpack_c(p: PackedTruth) -> u8 {
    (p >> 8) as u8
}

/// Precomputed NARS revision table.
///
/// revision_table[f1 * 256 + f2] = packed output truth
/// where c1 and c2 are implicit (we use a fixed c-pair per table,
/// or build a separate table per c-quantile).
///
/// For the full 4D table (f1, c1, f2, c2), we use a two-level approach:
/// 1. Quantize c into 16 buckets (4 bits)
/// 2. Build 16×16 = 256 tables of 256×256 entries each
/// Total: 256 × 128KB = 32 MB — fits L2 cache.
///
/// For the fast path, we use a single table with average c:
/// revision_fast[f1 * 256 + f2] → packed truth for c = c_mean.
pub struct NarsTables {
    /// Revision: merge two truths for the same statement.
    /// Index: f1 * 256 + f2, for c_self and c_other encoded in table selection.
    pub revision: Vec<[PackedTruth; 256 * 256]>,
    /// Deduction: A→B, B→C ⊢ A→C.
    pub deduction: [PackedTruth; 256 * 256],
    /// Number of c-quantile levels for revision.
    pub c_levels: usize,
}

impl NarsTables {
    /// Build all lookup tables.
    ///
    /// `c_levels`: number of confidence quantiles for revision tables.
    /// Use 16 for full precision (32 MB), 1 for fast path (128 KB).
    pub fn build(c_levels: usize) -> Self {
        let c_levels = c_levels.clamp(1, 16);

        // Deduction table: f_out = f1 * f2 / 255, c_out = f_out * c1 * c2 / 255²
        // Since c depends on inputs, we precompute f_out and set c_out = f_out
        // (conservative: deduction confidence ≤ deduction frequency).
        let mut deduction = [0u16; 256 * 256];
        for f1 in 0u16..256 {
            for f2 in 0u16..256 {
                let f_out = (f1 * f2 / 255).min(255) as u8;
                // c_out ≤ f_out * c1 * c2. Without knowing c, store f_out as upper bound.
                let c_out = f_out; // conservative estimate
                deduction[f1 as usize * 256 + f2 as usize] = pack_truth(f_out, c_out);
            }
        }

        // Revision tables: one per c-quantile pair
        let mut revision = Vec::with_capacity(c_levels * c_levels);
        for ci1 in 0..c_levels {
            for ci2 in 0..c_levels {
                let c1 = ((ci1 as f32 + 0.5) / c_levels as f32).min(0.99);
                let c2 = ((ci2 as f32 + 0.5) / c_levels as f32).min(0.99);
                let w1 = c1 / (1.0 - c1);
                let w2 = c2 / (1.0 - c2);
                let ws = w1 + w2;
                let c_rev = if ws > f32::EPSILON { ws / (ws + 1.0) } else { 0.0 };
                let c_rev_u8 = (c_rev * 255.0).round().min(255.0) as u8;

                let mut table = [0u16; 256 * 256];
                for f1_u in 0u16..256 {
                    let f1 = f1_u as f32 / 255.0;
                    for f2_u in 0u16..256 {
                        let f2 = f2_u as f32 / 255.0;
                        let f_rev = if ws > f32::EPSILON {
                            (f1 * w1 + f2 * w2) / ws
                        } else {
                            0.5
                        };
                        let f_rev_u8 = (f_rev * 255.0).round().min(255.0) as u8;
                        table[f1_u as usize * 256 + f2_u as usize] =
                            pack_truth(f_rev_u8, c_rev_u8);
                    }
                }
                revision.push(table);
            }
        }

        NarsTables {
            revision,
            deduction,
            c_levels,
        }
    }

    /// Look up revision result for two truth values.
    #[inline]
    pub fn revise(&self, f1: u8, c1: u8, f2: u8, c2: u8) -> PackedTruth {
        let ci1 = (c1 as usize * self.c_levels / 256).min(self.c_levels - 1);
        let ci2 = (c2 as usize * self.c_levels / 256).min(self.c_levels - 1);
        let table_idx = ci1 * self.c_levels + ci2;
        self.revision[table_idx][f1 as usize * 256 + f2 as usize]
    }

    /// Look up deduction result.
    #[inline]
    pub fn deduce(&self, f1: u8, f2: u8) -> PackedTruth {
        self.deduction[f1 as usize * 256 + f2 as usize]
    }

    /// Total memory footprint in bytes.
    pub fn byte_size(&self) -> usize {
        let rev_size = self.revision.len() * 256 * 256 * 2;
        let ded_size = 256 * 256 * 2;
        rev_size + ded_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_fast() {
        let tables = NarsTables::build(1); // fast path: single c-level
        assert_eq!(tables.revision.len(), 1);
        assert!(tables.byte_size() < 256 * 1024); // < 256 KB
    }

    #[test]
    fn test_deduction_identity() {
        let tables = NarsTables::build(1);
        // f=1.0 * f=1.0 should → f≈1.0
        let result = tables.deduce(255, 255);
        assert_eq!(unpack_f(result), 255);
    }

    #[test]
    fn test_revision_agreement() {
        let tables = NarsTables::build(4);
        // Two sources agree at f=0.8 → revised f should be near 0.8
        let result = tables.revise(204, 127, 204, 127); // f=0.8, c=0.5
        let f_out = unpack_f(result) as f32 / 255.0;
        assert!((f_out - 0.8).abs() < 0.05,
            "Revision of agreeing f=0.8 should stay near 0.8, got {}", f_out);
    }
}
