//! Distribution-free similarity scoring via empirical CDF.
//!
//! Maps raw u8 distances to calibrated f32 similarity in [0.0, 1.0].
//! Built from the EXACT 4096² distance distribution (8.4M unique pairs).
//! Not sampled. Not approximated.
//!
//! 512 bytes total. Fits in L1 cache.
//! `similarity(distance) = 1.0 - CDF(distance)` → O(1) lookup.

/// 256-entry lookup table: distance bucket → calibrated similarity.
///
/// Built from corpus statistics. Resolution: 256 levels spanning [0, max_distance].
/// `table[d]` gives the fraction of word pairs MORE DISTANT than `d`.
/// This is `1 - CDF(d)`, which IS the similarity score.
///
/// # Interpretation
/// - `similarity = 1.0` → identical (distance = 0)
/// - `similarity = 0.5` → median distance (50th percentile)
/// - `similarity = 0.0` → maximally distant (beyond all observed pairs)
pub struct SimilarityTable {
    /// 256 similarity values, indexed by u8 distance directly.
    table: [f32; 256],
}

impl SimilarityTable {
    /// Build from the exact distribution of an 8-bit distance matrix.
    ///
    /// Scans ALL upper-triangle entries of the matrix to build the
    /// empirical CDF, then inverts it: `similarity(d) = 1.0 - CDF(d)`.
    ///
    /// This is the primary construction method for DeepNSM.
    pub fn from_distance_matrix(matrix: &super::spo::WordDistanceMatrix) -> Self {
        let k = super::spo::WordDistanceMatrix::K;
        let mut histogram = [0u64; 256];
        let mut total = 0u64;

        // Count all unique pairs (upper triangle)
        for i in 0..k {
            for j in (i + 1)..k {
                let d = matrix.get(i as u16, j as u16) as usize;
                histogram[d] += 1;
                total += 1;
            }
        }

        Self::from_histogram(&histogram, total)
    }

    /// Build from a precomputed distance histogram.
    ///
    /// `histogram[d]` = number of pairs with distance `d`.
    /// `total` = sum of all histogram entries.
    pub fn from_histogram(histogram: &[u64; 256], total: u64) -> Self {
        let mut table = [0.0f32; 256];

        if total == 0 {
            // Degenerate: all similarities = 0.5
            for entry in table.iter_mut() {
                *entry = 0.5;
            }
            return Self { table };
        }

        // Compute CDF: cumulative fraction of pairs with distance ≤ d
        let mut cumulative = 0u64;
        let total_f = total as f64;

        for d in 0..256 {
            cumulative += histogram[d];
            let cdf = cumulative as f64 / total_f;
            // Similarity = 1.0 - CDF (fraction of pairs MORE distant)
            table[d] = (1.0 - cdf) as f32;
        }

        // Ensure table[0] is close to 1.0 (identical = max similarity)
        // The CDF at d=0 is usually very small, so 1-CDF ≈ 1.0
        // But if distance 0 is common, this naturally adjusts.

        Self { table }
    }

    /// Build from parametric model (mean μ, standard deviation σ).
    /// Sigmoid approximation: `similarity(d) = 1 / (1 + exp((d - μ) / σ))`.
    /// Use when exact histogram is unavailable.
    pub fn from_stats(mu: f32, sigma: f32) -> Self {
        let mut table = [0.0f32; 256];
        let sigma = sigma.max(1.0);

        for d in 0..256 {
            let z = (d as f32 - mu) / sigma;
            table[d] = 1.0 / (1.0 + z.exp());
        }

        Self { table }
    }

    /// Look up similarity for a raw u8 distance. O(1).
    #[inline]
    pub fn lookup_u8(&self, distance: u8) -> f32 {
        self.table[distance as usize]
    }

    /// Look up similarity for a summed distance (e.g., SPO sum of 3 roles).
    /// Clamps to [0, 255] range before lookup.
    #[inline]
    pub fn lookup(&self, distance: u32) -> f32 {
        let clamped = distance.min(255) as usize;
        self.table[clamped]
    }

    /// Look up similarity for a summed distance, with role-count scaling.
    /// Divides by `n_roles` before lookup (e.g., 3 for SPO triples).
    #[inline]
    pub fn lookup_averaged(&self, distance: u32, n_roles: u32) -> f32 {
        let avg = distance / n_roles.max(1);
        self.lookup(avg)
    }

    /// Get the raw table for serialization.
    pub fn as_slice(&self) -> &[f32; 256] {
        &self.table
    }

    /// Build from a raw 256-entry f32 slice.
    pub fn from_slice(data: &[f32; 256]) -> Self {
        Self { table: *data }
    }

    /// Byte size (always 1024 bytes = 256 × f32).
    pub const BYTE_SIZE: usize = 256 * 4;

    /// Serialize to bytes (little-endian f32).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Self::BYTE_SIZE);
        for &val in &self.table {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Deserialize from bytes (little-endian f32).
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::BYTE_SIZE {
            return None;
        }
        let mut table = [0.0f32; 256];
        for i in 0..256 {
            let offset = i * 4;
            let val = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            table[i] = val;
        }
        Some(Self { table })
    }
}

impl core::fmt::Debug for SimilarityTable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "SimilarityTable {{ [0]={:.3}, [64]={:.3}, [128]={:.3}, [192]={:.3}, [255]={:.3} }}",
            self.table[0], self.table[64], self.table[128], self.table[192], self.table[255]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parametric_sigmoid() {
        let table = SimilarityTable::from_stats(128.0, 30.0);

        // Distance 0 should be high similarity
        assert!(table.lookup_u8(0) > 0.95);

        // Distance at mean should be ~0.5
        let mid = table.lookup_u8(128);
        assert!(mid > 0.45 && mid < 0.55, "mid = {}", mid);

        // Distance 255 should be low similarity
        assert!(table.lookup_u8(255) < 0.05);

        // Monotonically decreasing
        for d in 1..256 {
            assert!(table.table[d] <= table.table[d - 1] + 1e-6);
        }
    }

    #[test]
    fn from_uniform_histogram() {
        let mut hist = [0u64; 256];
        // Uniform distribution
        for h in hist.iter_mut() {
            *h = 100;
        }
        let total = 25600;

        let table = SimilarityTable::from_histogram(&hist, total);

        // Distance 0: similarity ≈ 1.0 - (100/25600) ≈ 0.996
        assert!(table.lookup_u8(0) > 0.99);

        // Distance 128: similarity ≈ 0.5
        let mid = table.lookup_u8(128);
        assert!(mid > 0.49 && mid < 0.51, "mid = {}", mid);

        // Distance 255: similarity ≈ 0.0
        assert!(table.lookup_u8(255) < 0.01);
    }

    #[test]
    fn serialization_roundtrip() {
        let table = SimilarityTable::from_stats(100.0, 25.0);
        let bytes = table.to_bytes();
        assert_eq!(bytes.len(), SimilarityTable::BYTE_SIZE);

        let restored = SimilarityTable::from_bytes(&bytes).unwrap();
        for i in 0..256 {
            assert!((table.table[i] - restored.table[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn lookup_clamping() {
        let table = SimilarityTable::from_stats(128.0, 30.0);

        // Large distance should clamp to 255
        let result = table.lookup(10000);
        assert_eq!(result, table.lookup_u8(255));
    }
}
