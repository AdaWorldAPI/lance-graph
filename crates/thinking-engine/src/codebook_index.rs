//! Token → centroid codebook assignment.
//!
//! For a model with V tokens and a K×K distance table, this stores
//! one u16 per token: which distance-table row that token maps to.
//!
//! Built offline by finding the nearest weight row (cosine similarity)
//! for each token embedding. Replaces the `token_id % K` placeholder.

use std::path::Path;
use std::io::{Read, Write};

/// Maps each token_id to a distance-table row index.
#[derive(Clone, Debug)]
pub struct CodebookIndex {
    /// `indices[token_id]` = distance table row (u16).
    indices: Vec<u16>,
    /// Number of rows in the distance table (K).
    table_size: u16,
    /// Model identifier (e.g. "bge-m3").
    model: String,
}

impl CodebookIndex {
    /// Create a new codebook index.
    ///
    /// # Panics
    /// Panics if any index in `indices` is >= `table_size`.
    pub fn new(indices: Vec<u16>, table_size: u16, model: String) -> Self {
        for (i, &idx) in indices.iter().enumerate() {
            assert!(
                idx < table_size,
                "index[{}] = {} >= table_size {}",
                i, idx, table_size
            );
        }
        Self { indices, table_size, model }
    }

    /// Look up the distance-table row for a single token.
    ///
    /// Out-of-range token_ids wrap via modulo (same as the old placeholder,
    /// but only as a safety net — real vocab should always be in range).
    pub fn lookup(&self, token_id: u32) -> u16 {
        let idx = token_id as usize;
        if idx < self.indices.len() {
            self.indices[idx]
        } else {
            // Safety fallback — should not happen with correct vocab
            (token_id % self.table_size as u32) as u16
        }
    }

    /// Batch lookup for multiple tokens.
    pub fn lookup_many(&self, token_ids: &[u32]) -> Vec<u16> {
        token_ids.iter().map(|&id| self.lookup(id)).collect()
    }

    /// Save as raw binary: V × 2 bytes (little-endian u16 per token).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(path)?;
        for &idx in &self.indices {
            file.write_all(&idx.to_le_bytes())?;
        }
        file.flush()
    }

    /// Load from raw binary (V × 2 bytes, little-endian u16).
    pub fn load(path: &Path, table_size: u16, model: String) -> std::io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        if buf.len() % 2 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "file size not a multiple of 2",
            ));
        }
        let indices: Vec<u16> = buf
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        // Validate all indices are in range
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= table_size {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("index[{}] = {} >= table_size {}", i, idx, table_size),
                ));
            }
        }
        Ok(Self { indices, table_size, model })
    }

    /// Number of tokens in the index (vocab size).
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// The distance table size (K).
    pub fn table_size(&self) -> u16 {
        self.table_size
    }

    /// Model name.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// How many unique centroid rows are actually used.
    pub fn unique_centroids(&self) -> usize {
        let mut seen = vec![false; self.table_size as usize];
        for &idx in &self.indices {
            seen[idx as usize] = true;
        }
        seen.iter().filter(|&&b| b).count()
    }

    /// Distribution: for each centroid, how many tokens map to it.
    pub fn centroid_counts(&self) -> Vec<u32> {
        let mut counts = vec![0u32; self.table_size as usize];
        for &idx in &self.indices {
            counts[idx as usize] += 1;
        }
        counts
    }

    /// Raw access to the indices slice.
    pub fn indices(&self) -> &[u16] {
        &self.indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_lookup() {
        let indices = vec![0u16, 5, 3, 7, 1];
        let ci = CodebookIndex::new(indices.clone(), 8, "test".into());
        assert_eq!(ci.len(), 5);
        assert_eq!(ci.lookup(0), 0);
        assert_eq!(ci.lookup(1), 5);
        assert_eq!(ci.lookup(2), 3);
        assert_eq!(ci.lookup(3), 7);
        assert_eq!(ci.lookup(4), 1);
    }

    #[test]
    fn test_lookup_out_of_range_fallback() {
        let indices = vec![0u16, 1, 2];
        let ci = CodebookIndex::new(indices, 8, "test".into());
        // token_id 10 is out of range — falls back to 10 % 8 = 2
        assert_eq!(ci.lookup(10), 2);
    }

    #[test]
    fn test_lookup_many() {
        let indices = vec![10u16, 20, 30, 40];
        let ci = CodebookIndex::new(indices, 64, "test".into());
        let result = ci.lookup_many(&[0, 2, 3]);
        assert_eq!(result, vec![10, 30, 40]);
    }

    #[test]
    fn test_unique_centroids() {
        let indices = vec![3u16, 3, 5, 3, 5, 7];
        let ci = CodebookIndex::new(indices, 8, "test".into());
        assert_eq!(ci.unique_centroids(), 3); // {3, 5, 7}
    }

    #[test]
    fn test_centroid_counts() {
        let indices = vec![0u16, 0, 1, 0, 2, 1];
        let ci = CodebookIndex::new(indices, 4, "test".into());
        let counts = ci.centroid_counts();
        assert_eq!(counts, vec![3, 2, 1, 0]);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = std::env::temp_dir().join("codebook_index_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_index.u16");

        let indices = vec![0u16, 99, 512, 1023, 42, 7];
        let ci = CodebookIndex::new(indices.clone(), 1024, "bge-m3".into());
        ci.save(&path).unwrap();

        // Check file size
        let meta = std::fs::metadata(&path).unwrap();
        assert_eq!(meta.len(), 12); // 6 tokens × 2 bytes

        let loaded = CodebookIndex::load(&path, 1024, "bge-m3".into()).unwrap();
        assert_eq!(loaded.len(), 6);
        assert_eq!(loaded.indices(), indices.as_slice());
        assert_eq!(loaded.table_size(), 1024);

        // Clean up
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "index[2] = 8 >= table_size 8")]
    fn test_new_rejects_out_of_range() {
        let indices = vec![0u16, 1, 8]; // 8 >= table_size 8
        CodebookIndex::new(indices, 8, "test".into());
    }

    #[test]
    fn test_load_rejects_bad_index() {
        let dir = std::env::temp_dir().join("codebook_index_bad_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad_index.u16");

        // Write an index value that exceeds table_size
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&0u16.to_le_bytes()).unwrap();
        f.write_all(&100u16.to_le_bytes()).unwrap(); // 100 >= table_size 8
        drop(f);

        let result = CodebookIndex::load(&path, 8, "test".into());
        assert!(result.is_err());

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn test_is_empty() {
        let ci = CodebookIndex::new(vec![], 8, "test".into());
        assert!(ci.is_empty());
        assert_eq!(ci.len(), 0);
    }
}
