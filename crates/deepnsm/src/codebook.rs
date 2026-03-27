//! CAM-PQ codebook loading and distance computation.
//!
//! Product Quantization over 96D distributional vectors from COCA subgenre frequencies.
//! 6 subspaces × 16 dimensions × 256 centroids = 96KB codebook.
//! 5,050 words × 6 bytes = 30KB CAM codes.
//!
//! Distance between any two words: 6 × 16 = 96 float operations via centroid lookup.
//! No original vectors needed at runtime.

use std::path::Path;

/// Number of PQ subspaces.
pub const NUM_SUBSPACES: usize = 6;
/// Dimensions per subspace.
pub const SUB_DIM: usize = 16;
/// Number of centroids per subspace.
pub const NUM_CENTROIDS: usize = 256;
/// Total original dimensions.
pub const TOTAL_DIM: usize = NUM_SUBSPACES * SUB_DIM; // 96
/// Size of the codebook in f32 values.
pub const CODEBOOK_SIZE: usize = NUM_SUBSPACES * NUM_CENTROIDS * SUB_DIM; // 24,576

/// CAM-PQ codebook: [6][256][16] f32 centroids.
pub struct Codebook {
    /// Flat storage: codebook[subspace * 256 * 16 + centroid * 16 + dim]
    centroids: Vec<f32>,
    /// Normalization mean (96 values, one per original dimension).
    pub mean: Vec<f32>,
    /// Normalization std (96 values).
    pub std: Vec<f32>,
}

/// CAM codes for the vocabulary: one 6-byte fingerprint per word.
pub struct CamCodes {
    /// Flat storage: codes[word_index * 6 + subspace]
    data: Vec<u8>,
    /// Number of words.
    pub count: usize,
}

impl Codebook {
    /// Load codebook from binary file.
    ///
    /// Format: [6][256][16] × f32, little-endian.
    /// Total: 24,576 × 4 = 98,304 bytes.
    pub fn load_binary(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("Failed to read codebook {}: {}", path.display(), e))?;

        let expected = CODEBOOK_SIZE * 4;
        if bytes.len() < expected {
            return Err(format!(
                "Codebook too small: {} bytes, expected {}",
                bytes.len(),
                expected
            ));
        }

        let centroids: Vec<f32> = bytes[..expected]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(Codebook {
            centroids,
            mean: Vec::new(),
            std: Vec::new(),
        })
    }

    /// Load codebook with normalization params from JSON.
    /// (Parses only the codebook and normalization arrays.)
    pub fn load_json(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        // Simple JSON parsing for the arrays we need
        let mean = extract_f32_array(&content, "\"mean\"")
            .ok_or("Failed to extract mean array")?;
        let std_vals = extract_f32_array(&content, "\"std\"")
            .ok_or("Failed to extract std array")?;
        let centroids = extract_codebook_array(&content)
            .ok_or("Failed to extract codebook array")?;

        Ok(Codebook {
            centroids,
            mean: mean,
            std: std_vals,
        })
    }

    /// Get centroid vector for a subspace and centroid index.
    #[inline]
    pub fn centroid(&self, subspace: usize, centroid_idx: u8) -> &[f32] {
        let offset = subspace * NUM_CENTROIDS * SUB_DIM + centroid_idx as usize * SUB_DIM;
        &self.centroids[offset..offset + SUB_DIM]
    }

    /// Compute L2 distance between two CAM codes via codebook lookup.
    #[inline]
    pub fn distance(&self, a: &[u8; 6], b: &[u8; 6]) -> f32 {
        let mut dist_sq = 0.0f32;
        for s in 0..NUM_SUBSPACES {
            let ca = self.centroid(s, a[s]);
            let cb = self.centroid(s, b[s]);
            for d in 0..SUB_DIM {
                let diff = ca[d] - cb[d];
                dist_sq += diff * diff;
            }
        }
        dist_sq.sqrt()
    }

    /// Compute squared L2 distance (avoids sqrt for ranking).
    #[inline]
    pub fn distance_sq(&self, a: &[u8; 6], b: &[u8; 6]) -> f32 {
        let mut dist_sq = 0.0f32;
        for s in 0..NUM_SUBSPACES {
            let ca = self.centroid(s, a[s]);
            let cb = self.centroid(s, b[s]);
            for d in 0..SUB_DIM {
                let diff = ca[d] - cb[d];
                dist_sq += diff * diff;
            }
        }
        dist_sq
    }

    /// Build precomputed distance table for one subspace.
    /// Returns a 256×256 f32 table of squared distances between all centroid pairs.
    pub fn subspace_distance_table(&self, subspace: usize) -> Vec<f32> {
        let mut table = vec![0.0f32; NUM_CENTROIDS * NUM_CENTROIDS];
        for i in 0..NUM_CENTROIDS {
            let ci = self.centroid(subspace, i as u8);
            for j in (i + 1)..NUM_CENTROIDS {
                let cj = self.centroid(subspace, j as u8);
                let mut d = 0.0f32;
                for dim in 0..SUB_DIM {
                    let diff = ci[dim] - cj[dim];
                    d += diff * diff;
                }
                table[i * NUM_CENTROIDS + j] = d;
                table[j * NUM_CENTROIDS + i] = d;
            }
        }
        table
    }

    /// Number of centroids loaded.
    pub fn len(&self) -> usize {
        self.centroids.len() / SUB_DIM
    }
}

impl CamCodes {
    /// Load CAM codes from binary file.
    ///
    /// Format: N × 6 bytes, one 6-byte code per word.
    pub fn load(path: &Path) -> Result<Self, String> {
        let data = std::fs::read(path)
            .map_err(|e| format!("Failed to read CAM codes {}: {}", path.display(), e))?;

        if data.len() % 6 != 0 {
            return Err(format!(
                "CAM codes file size {} not divisible by 6",
                data.len()
            ));
        }

        let count = data.len() / 6;
        Ok(CamCodes { data, count })
    }

    /// Get the 6-byte CAM code for a word by index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<[u8; 6]> {
        if index >= self.count {
            return None;
        }
        let offset = index * 6;
        let mut code = [0u8; 6];
        code.copy_from_slice(&self.data[offset..offset + 6]);
        Some(code)
    }

    /// Number of words with CAM codes.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

// ─── Simple JSON parsing helpers ─────────────────────────────────────────────
// No serde dependency — hand-parse the specific structure we need.

/// Extract a float array following a key in JSON.
fn extract_f32_array(json: &str, key: &str) -> Option<Vec<f32>> {
    let key_pos = json.find(key)?;
    let after_key = &json[key_pos + key.len()..];
    let bracket_start = after_key.find('[')?;
    let bracket_end = after_key[bracket_start..].find(']')?;
    let array_str = &after_key[bracket_start + 1..bracket_start + bracket_end];

    let values: Vec<f32> = array_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

/// Extract the nested codebook array from JSON.
/// Structure: "codebook": [[[ ... ], ...], ...]
fn extract_codebook_array(json: &str) -> Option<Vec<f32>> {
    let key = "\"codebook\"";
    let key_pos = json.find(key)?;
    let after_key = &json[key_pos + key.len()..];

    // Find the start of the codebook array
    let start = after_key.find("[[[")?;
    let codebook_str = &after_key[start..];

    // Parse all numbers between [[[ and ]]]
    let mut values = Vec::with_capacity(CODEBOOK_SIZE);
    let mut num_buf = String::new();
    let mut in_number = false;

    for ch in codebook_str.chars() {
        match ch {
            '-' | '.' | '0'..='9' | 'e' | 'E' | '+' => {
                num_buf.push(ch);
                in_number = true;
            }
            _ => {
                if in_number {
                    if let Ok(val) = num_buf.parse::<f32>() {
                        values.push(val);
                    }
                    num_buf.clear();
                    in_number = false;
                }
            }
        }

        // Stop after we have enough values
        if values.len() >= CODEBOOK_SIZE {
            break;
        }
    }

    if values.len() >= CODEBOOK_SIZE {
        values.truncate(CODEBOOK_SIZE);
        Some(values)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_layout() {
        // Verify constants
        assert_eq!(TOTAL_DIM, 96);
        assert_eq!(CODEBOOK_SIZE, 24576);
        assert_eq!(CODEBOOK_SIZE * 4, 98304); // matches codebook_pq.bin size
    }

    #[test]
    fn parse_f32_array() {
        let json = r#"{"mean": [1.0, 2.5, 3.7]}"#;
        let arr = extract_f32_array(json, "\"mean\"").unwrap();
        assert_eq!(arr.len(), 3);
        assert!((arr[0] - 1.0).abs() < 1e-6);
        assert!((arr[1] - 2.5).abs() < 1e-6);
    }
}
