//! BGZ-HHTL-D: Slot D (16-bit tree address) + Slot V (BF16 value).
//!
//! This is the branching cascade encoding — each weight row is encoded
//! as a 4-byte pair: WHERE it lives in the HHTL tree (Slot D) and
//! WHAT its residual scale is (Slot V).
//!
//! ```text
//! Slot D bit layout (u16):
//!   bits 15..14 = HEEL basin      (2 bits, 4 states: Q/K, V, Gate, FFN)
//!   bits 13..10 = HIP family      (4 bits, 16 families within basin)
//!   bits  9..2  = TWIG centroid   (8 bits, 256 centroids within family)
//!   bit      1  = BRANCH polarity (sign of residual)
//!   bit      0  = reserved
//!
//! Slot V (u16 BF16):
//!   BF16 residual magnitude from the centroid.
//!   rehydrate: centroid_f32 + polarity * bf16_to_f32(slot_v) * gamma
//!
//! Total: 4 bytes per row. A 2048×6144 weight matrix (12M elements)
//! compresses to 2048 × 4 = 8192 bytes (8 KB) + shared palette.
//! ```
//!
//! The palette (HHTL cache) is shared across all rows of the same role.
//! The per-row encoding is just the 4-byte (Slot D, Slot V) pair.

use crate::hhtl_cache::HhtlCache;
use crate::projection::Base17;
use crate::stacked_n::{bf16_to_f32, f32_to_bf16};

/// Basin classification from stride/role.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HeelBasin {
    /// Query/Key/Output projections (stride=3)
    QK = 0,
    /// Value projection (stride=5)
    V = 1,
    /// Gate projection (stride=8)
    Gate = 2,
    /// FFN Up/Down projections (stride=2,4)
    FFN = 3,
}

impl HeelBasin {
    pub fn from_role(role: &str) -> Self {
        let r = role.to_lowercase();
        if r.contains("q_proj") || r.contains("k_proj") || r.contains("o_proj") {
            HeelBasin::QK
        } else if r.contains("v_proj") {
            HeelBasin::V
        } else if r.contains("gate") {
            HeelBasin::Gate
        } else {
            HeelBasin::FFN
        }
    }

    pub fn bits(self) -> u16 {
        (self as u16) << 14
    }
}

/// A single BGZ-HHTL-D encoded row: 4 bytes total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HhtlDEntry {
    /// Slot D: 16-bit tree address.
    pub slot_d: u16,
    /// Slot V: BF16 residual magnitude.
    pub slot_v: u16,
}

impl HhtlDEntry {
    pub const BYTE_SIZE: usize = 4;

    /// Construct from components.
    pub fn new(basin: HeelBasin, hip_family: u8, twig_centroid: u8, polarity: bool, residual_bf16: u16) -> Self {
        let d = basin.bits()
            | ((hip_family as u16 & 0x0F) << 10)
            | ((twig_centroid as u16) << 2)
            | if polarity { 2 } else { 0 };
        HhtlDEntry { slot_d: d, slot_v: residual_bf16 }
    }

    /// Extract HEEL basin (bits 15..14).
    #[inline]
    pub fn heel_basin(&self) -> u8 {
        (self.slot_d >> 14) as u8
    }

    /// Extract HIP family (bits 13..10).
    #[inline]
    pub fn hip_family(&self) -> u8 {
        ((self.slot_d >> 10) & 0x0F) as u8
    }

    /// Extract TWIG centroid index (bits 9..2).
    #[inline]
    pub fn twig_centroid(&self) -> u8 {
        ((self.slot_d >> 2) & 0xFF) as u8
    }

    /// Extract BRANCH polarity (bit 1).
    #[inline]
    pub fn polarity(&self) -> bool {
        (self.slot_d & 2) != 0
    }

    /// Residual magnitude as f32.
    #[inline]
    pub fn residual_f32(&self) -> f32 {
        bf16_to_f32(self.slot_v)
    }

    /// Serialize to 4 bytes (little-endian).
    pub fn to_le_bytes(&self) -> [u8; 4] {
        let mut buf = [0u8; 4];
        buf[0..2].copy_from_slice(&self.slot_d.to_le_bytes());
        buf[2..4].copy_from_slice(&self.slot_v.to_le_bytes());
        buf
    }

    /// Deserialize from 4 bytes (little-endian).
    pub fn from_le_bytes(bytes: &[u8; 4]) -> Self {
        HhtlDEntry {
            slot_d: u16::from_le_bytes([bytes[0], bytes[1]]),
            slot_v: u16::from_le_bytes([bytes[2], bytes[3]]),
        }
    }
}

/// A complete BGZ-HHTL-D encoded tensor: palette + per-row entries.
#[derive(Clone, Debug)]
pub struct HhtlDTensor {
    /// Role name (e.g., "talker_q_proj").
    pub role: String,
    /// Basin for this tensor.
    pub basin: HeelBasin,
    /// Shared HHTL cache (palette + distance table + route table).
    pub cache: HhtlCache,
    /// Per-row entries: one (Slot D, Slot V) per weight row.
    pub entries: Vec<HhtlDEntry>,
    /// Original shape: [n_rows, n_cols].
    pub original_shape: [usize; 2],
    /// Gamma profile for rehydration.
    pub gamma_meta: [f32; 4],
    /// Fisher z i8 pairwise cosine table (k×k + 8 bytes gamma).
    /// The centroid routes into this table. The table IS the cosine value.
    /// None for legacy files without Fisher z.
    pub fisher_z: Option<crate::fisher_z::FisherZTable>,
}

impl HhtlDTensor {
    /// Encode a weight matrix (f32 rows) into BGZ-HHTL-D.
    ///
    /// Pipeline per row:
    ///   f32[cols] → Base17 (golden-step fold)
    ///     → nearest centroid in palette → twig_centroid (u8)
    ///     → residual magnitude → Slot V (BF16)
    ///     → HIP family from centroid clustering → hip_family (u8)
    ///     → basin from role → heel_basin (u8)
    pub fn encode(
        role: &str,
        rows_f32: &[Vec<f32>],
        cache: &HhtlCache,
        hip_assignments: &[u8],  // centroid → HIP family (16-way)
    ) -> Self {
        let basin = HeelBasin::from_role(role);
        let n_rows = rows_f32.len();
        let n_cols = if n_rows > 0 { rows_f32[0].len() } else { 0 };

        let mut entries = Vec::with_capacity(n_rows);

        for row in rows_f32 {
            // Step 1: Project to Base17
            let base17 = Base17::from_f32(row);

            // Step 2: Find nearest centroid
            let (centroid_idx, l1_dist) = cache.nearest(&base17);

            // Step 3: Compute residual magnitude
            // Residual = L1 distance normalized by centroid magnitude
            let centroid = &cache.palette.entries[centroid_idx as usize];
            let centroid_mag: f64 = centroid.dims.iter()
                .map(|&d| (d as f64).abs())
                .sum::<f64>()
                .max(1.0);
            let residual = l1_dist as f64 / centroid_mag;
            let residual_f32 = residual as f32;

            // Step 4: Polarity from sign of dominant residual dimension
            let polarity = {
                let mut max_dim = 0i32;
                for i in 0..17 {
                    let diff = base17.dims[i] as i32 - centroid.dims[i] as i32;
                    if diff.abs() > max_dim.abs() {
                        max_dim = diff;
                    }
                }
                max_dim >= 0
            };

            // Step 5: HIP family from pre-computed clustering
            let hip_family = if (centroid_idx as usize) < hip_assignments.len() {
                hip_assignments[centroid_idx as usize]
            } else {
                0
            };

            entries.push(HhtlDEntry::new(
                basin,
                hip_family,
                centroid_idx,
                polarity,
                f32_to_bf16(residual_f32),
            ));
        }

        // Build Fisher z table from representative rows (centroid-nearest)
        let fisher_z = if !rows_f32.is_empty() {
            let k = cache.palette.entries.len();
            let mut reps: Vec<Vec<f32>> = vec![Vec::new(); k];
            let mut rep_dists: Vec<u32> = vec![u32::MAX; k];
            for (i, row) in rows_f32.iter().enumerate() {
                let b17 = Base17::from_f32(row);
                let (ci, dist) = cache.nearest(&b17);
                let ci = ci as usize;
                if ci < k && dist < rep_dists[ci] {
                    reps[ci] = row.clone();
                    rep_dists[ci] = dist;
                }
            }
            // Fill empty centroids with palette to_f32
            for ci in 0..k {
                if reps[ci].is_empty() {
                    reps[ci] = cache.palette.entries[ci].to_f32(n_cols);
                }
            }
            Some(crate::fisher_z::FisherZTable::build(&reps, k))
        } else {
            None
        };

        HhtlDTensor {
            role: role.to_string(),
            basin,
            cache: cache.clone(),
            entries,
            original_shape: [n_rows, n_cols],
            gamma_meta: cache.gamma_meta,
            fisher_z,
        }
    }

    /// Total encoded size: entries only (palette is shared).
    pub fn entries_byte_size(&self) -> usize {
        self.entries.len() * HhtlDEntry::BYTE_SIZE
    }

    /// Serialize entries to flat byte vector (for safetensors storage).
    pub fn entries_to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.entries.len() * 4);
        for entry in &self.entries {
            buf.extend_from_slice(&entry.to_le_bytes());
        }
        buf
    }

    /// Deserialize entries from flat bytes.
    pub fn entries_from_bytes(bytes: &[u8]) -> Vec<HhtlDEntry> {
        let n = bytes.len() / 4;
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let chunk: [u8; 4] = [
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ];
            entries.push(HhtlDEntry::from_le_bytes(&chunk));
        }
        entries
    }

    /// Compression ratio vs original BF16 storage.
    pub fn compression_ratio(&self) -> f64 {
        let original_bytes = self.original_shape[0] * self.original_shape[1] * 2; // BF16
        let encoded_bytes = self.entries_byte_size(); // 4 bytes per row (NOT per element)
        original_bytes as f64 / encoded_bytes as f64
    }

    /// Pairwise cosine between two rows via Fisher z table lookup.
    ///
    /// O(1): one i8 read + one tanh decode.
    /// Returns None if Fisher z table is not available.
    #[inline]
    pub fn cosine_lookup(&self, row_a: usize, row_b: usize) -> Option<f32> {
        let fz = self.fisher_z.as_ref()?;
        let ca = self.entries.get(row_a)?.twig_centroid();
        let cb = self.entries.get(row_b)?.twig_centroid();
        Some(fz.lookup_f32(ca, cb))
    }

    /// Total byte size including Fisher z table.
    pub fn total_byte_size(&self) -> usize {
        let entries = self.entries_byte_size();
        let fz = self.fisher_z.as_ref().map(|t| t.byte_size()).unwrap_or(0);
        entries + fz
    }
}

/// Build 16-way HIP family assignments from a palette via binary splits.
///
/// Takes 256 centroids, recursively splits by farthest-pair into 16 groups.
/// Returns [u8; 256] mapping centroid index → HIP family (0..15).
pub fn build_hip_families(palette: &[Base17]) -> Vec<u8> {
    let k = palette.len();
    let mut assignments = vec![0u8; k];

    if k == 0 {
        return assignments;
    }

    // Recursive binary split: 4 levels → 16 families
    fn split(indices: &[usize], palette: &[Base17], assignments: &mut [u8], family_base: u8, depth: u8) {
        if depth == 4 || indices.len() <= 1 {
            for &idx in indices {
                assignments[idx] = family_base;
            }
            return;
        }

        // Find farthest pair (poles)
        let mut max_dist = 0u32;
        let mut pole_a = indices[0];
        let mut pole_b = indices[0];

        for &i in indices {
            for &j in indices {
                let d = palette[i].l1(&palette[j]);
                if d > max_dist {
                    max_dist = d;
                    pole_a = i;
                    pole_b = j;
                }
            }
        }

        // Split by nearer pole
        let mut left = Vec::new();
        let mut right = Vec::new();
        for &idx in indices {
            let da = palette[idx].l1(&palette[pole_a]);
            let db = palette[idx].l1(&palette[pole_b]);
            if da <= db {
                left.push(idx);
            } else {
                right.push(idx);
            }
        }

        let half = 1u8 << (3 - depth); // 8, 4, 2, 1
        split(&left, palette, assignments, family_base, depth + 1);
        split(&right, palette, assignments, family_base + half, depth + 1);
    }

    let all: Vec<usize> = (0..k).collect();
    split(&all, palette, &mut assignments, 0, 0);

    assignments
}

// ═══════════════════════════════════════════════════════════════════════════
// Safetensors serialization support
// ═══════════════════════════════════════════════════════════════════════════

/// Metadata for a BGZ-HHTL-D encoded safetensor.
/// Stored as JSON in the safetensors header's `__metadata__` field.
#[derive(Clone, Debug)]
pub struct HhtlDMeta {
    pub encoding: String,   // "bgz-hhtl-d"
    pub version: u32,       // 1
    pub original_model: String,
    pub n_roles: usize,
    pub palette_k: usize,
    pub total_entries: usize,
    pub total_bytes: usize,
    pub compression_ratio: f64,
}

impl HhtlDMeta {
    pub fn to_json_map(&self) -> std::collections::HashMap<String, String> {
        let mut m = std::collections::HashMap::new();
        m.insert("encoding".into(), self.encoding.clone());
        m.insert("version".into(), self.version.to_string());
        m.insert("original_model".into(), self.original_model.clone());
        m.insert("n_roles".into(), self.n_roles.to_string());
        m.insert("palette_k".into(), self.palette_k.to_string());
        m.insert("total_entries".into(), self.total_entries.to_string());
        m.insert("total_bytes".into(), self.total_bytes.to_string());
        m.insert("compression_ratio".into(), format!("{:.1}", self.compression_ratio));
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hhtl_d_entry_roundtrip() {
        let entry = HhtlDEntry::new(HeelBasin::Gate, 7, 42, true, 0x3C00); // BF16 1.0
        let bytes = entry.to_le_bytes();
        let decoded = HhtlDEntry::from_le_bytes(&bytes);

        assert_eq!(decoded.heel_basin(), HeelBasin::Gate as u8);
        assert_eq!(decoded.hip_family(), 7);
        assert_eq!(decoded.twig_centroid(), 42);
        assert!(decoded.polarity());
        assert!((decoded.residual_f32() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn hhtl_d_entry_bit_layout() {
        // Verify bit positions are correct
        let entry = HhtlDEntry::new(HeelBasin::V, 0xF, 0xFF, false, 0);
        assert_eq!(entry.heel_basin(), 1); // V = 1
        assert_eq!(entry.hip_family(), 0xF);
        assert_eq!(entry.twig_centroid(), 0xFF);
        assert!(!entry.polarity());
    }

    #[test]
    fn hhtl_d_entry_byte_size() {
        assert_eq!(HhtlDEntry::BYTE_SIZE, 4);
    }

    #[test]
    fn entries_bulk_roundtrip() {
        let entries = vec![
            HhtlDEntry::new(HeelBasin::QK, 3, 100, true, 0x4000),
            HhtlDEntry::new(HeelBasin::FFN, 12, 200, false, 0x3F80),
        ];

        let bytes: Vec<u8> = entries.iter()
            .flat_map(|e| e.to_le_bytes())
            .collect();
        let decoded = HhtlDTensor::entries_from_bytes(&bytes);

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].twig_centroid(), 100);
        assert_eq!(decoded[1].hip_family(), 12);
    }

    #[test]
    fn hip_families_produce_16_groups() {
        // Synthetic palette: 256 Base17 entries
        let palette: Vec<Base17> = (0..256).map(|i| {
            let mut dims = [0i16; 17];
            dims[0] = (i as i16) * 100;
            dims[1] = ((i as i16) % 17) * 50;
            dims[2] = ((i as i16) / 17) * 30;
            Base17 { dims }
        }).collect();

        let families = build_hip_families(&palette);
        assert_eq!(families.len(), 256);

        let max_family = *families.iter().max().unwrap();
        assert!(max_family < 16, "should produce at most 16 families, got max {}", max_family);

        // At least 8 families should be used
        let used: std::collections::HashSet<u8> = families.iter().copied().collect();
        assert!(used.len() >= 8, "should use at least 8 families, got {}", used.len());
    }

    #[test]
    fn project_row_nonzero() {
        let row: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.01 - 10.0)).collect();
        let b17 = Base17::from_f32(&row);
        let mag: i64 = b17.dims.iter().map(|&d| (d as i64).abs()).sum();
        assert!(mag > 0, "projection should be nonzero");
    }
}
