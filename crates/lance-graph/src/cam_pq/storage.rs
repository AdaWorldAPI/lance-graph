//! Lance storage schema for CAM-PQ fingerprints and codebooks.
//!
//! # Tables
//!
//! ## `vectors` — Main data table
//! ```sql
//! CREATE TABLE vectors (
//!     id          BIGINT PRIMARY KEY,
//!     cam         FIXED_SIZE_BINARY(6),     -- 48-bit CAM fingerprint
//!     metadata    VARCHAR,
//!     timestamp   TIMESTAMP
//! );
//! ```
//!
//! ## `cam_codebook` — Codebook table (trained once, immutable)
//! ```sql
//! CREATE TABLE cam_codebook (
//!     subspace    TINYINT,                   -- 0-5 (HEEL through GAMMA)
//!     centroid_id TINYINT UNSIGNED,           -- 0-255
//!     vector      FIXED_SIZE_LIST(FLOAT, N),  -- centroid vector (D/6 dims)
//!     label       VARCHAR                     -- semantic label (CLAM mode)
//! );
//! ```
//!
//! In Lance columnar format, the `cam` column is 6 bytes per row.
//! 1 million vectors = 6MB. 1 billion = 6GB. Fits in RAM.

/// CAM column name in Lance tables.
pub const CAM_COLUMN: &str = "cam";

/// CAM fingerprint size in bytes.
pub const CAM_SIZE: usize = 6;

/// Codebook table name.
pub const CODEBOOK_TABLE: &str = "cam_codebook";

/// Schema for the CAM vectors table.
#[derive(Debug, Clone)]
pub struct CamVectorSchema {
    /// Name of the table.
    pub table_name: String,
    /// Name of the CAM column.
    pub cam_column: String,
    /// Name of the ID column.
    pub id_column: String,
}

impl Default for CamVectorSchema {
    fn default() -> Self {
        Self {
            table_name: "vectors".into(),
            cam_column: CAM_COLUMN.into(),
            id_column: "id".into(),
        }
    }
}

/// Schema for the codebook table.
#[derive(Debug, Clone)]
pub struct CamCodebookSchema {
    /// Name of the codebook table.
    pub table_name: String,
    /// Number of subspaces (always 6).
    pub num_subspaces: usize,
    /// Number of centroids per subspace (always 256).
    pub num_centroids: usize,
    /// Dimension per subspace (D/6).
    pub subspace_dim: usize,
}

impl CamCodebookSchema {
    pub fn new(total_dim: usize) -> Self {
        Self {
            table_name: CODEBOOK_TABLE.into(),
            num_subspaces: 6,
            num_centroids: 256,
            subspace_dim: total_dim / 6,
        }
    }

    /// Total rows in the codebook table: 6 × 256 = 1536.
    pub fn total_rows(&self) -> usize {
        self.num_subspaces * self.num_centroids
    }

    /// Codebook size in bytes: 1536 × (D/6) × 4.
    pub fn codebook_bytes(&self) -> usize {
        self.total_rows() * self.subspace_dim * 4
    }
}

/// Storage statistics for a CAM-PQ dataset.
#[derive(Debug, Clone)]
pub struct CamStorageStats {
    /// Number of vectors stored.
    pub num_vectors: u64,
    /// Storage for CAM column (bytes).
    pub cam_bytes: u64,
    /// Codebook size (bytes).
    pub codebook_bytes: u64,
    /// Compression ratio vs raw vectors.
    pub compression_ratio: f64,
}

impl CamStorageStats {
    pub fn compute(num_vectors: u64, total_dim: usize) -> Self {
        let cam_bytes = num_vectors * CAM_SIZE as u64;
        let codebook = CamCodebookSchema::new(total_dim);
        let codebook_bytes = codebook.codebook_bytes() as u64;
        let raw_bytes = num_vectors * total_dim as u64 * 4; // f32
        let compression_ratio = if cam_bytes + codebook_bytes > 0 {
            raw_bytes as f64 / (cam_bytes + codebook_bytes) as f64
        } else {
            0.0
        };

        CamStorageStats {
            num_vectors,
            cam_bytes,
            codebook_bytes,
            compression_ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_stats_1m_256d() {
        let stats = CamStorageStats::compute(1_000_000, 256);
        assert_eq!(stats.cam_bytes, 6_000_000); // 6MB
        // Raw: 1M × 256 × 4 = 1024MB. Compressed: ~6MB + codebook
        assert!(stats.compression_ratio > 100.0);
    }

    #[test]
    fn test_storage_stats_1b_1024d() {
        let stats = CamStorageStats::compute(1_000_000_000, 1024);
        assert_eq!(stats.cam_bytes, 6_000_000_000); // 6GB
        assert!(stats.compression_ratio > 500.0);
    }

    #[test]
    fn test_codebook_schema() {
        let schema = CamCodebookSchema::new(1024);
        assert_eq!(schema.total_rows(), 1536);
        assert_eq!(schema.subspace_dim, 170); // 1024/6 ≈ 170
        // ~1MB codebook
        assert!(schema.codebook_bytes() < 2_000_000);
    }
}
