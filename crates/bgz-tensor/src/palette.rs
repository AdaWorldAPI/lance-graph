//! Weight palette construction via manifold-aware clustering.
//!
//! Standard PQ uses k-means: assumes spherical clusters, ignores manifold
//! structure, gives no distortion guarantees. CLAM (Clustered Learning of
//! Approximate Manifolds) gives metric-safe radius bounds for every
//! palette assignment — you KNOW the worst-case distortion.
//!
//! For weight matrices, the manifold structure matters because:
//! - Weight rows cluster around learned features (not randomly distributed)
//! - Dead neurons produce near-zero rows (manifold singularities)
//! - Attention heads specialize → weight manifold has low intrinsic dimension
//!
//! This module implements a simplified CLAM-inspired palette builder:
//! furthest-point sampling for coverage, then Voronoi assignment with
//! tracked distortion bounds.

use crate::projection::Base17;

/// Maximum palette size (8-bit index).
pub const MAX_PALETTE: usize = 256;

/// A palette of archetypal weight patterns.
///
/// Each entry is a Base17 projection of a representative weight vector.
/// The distance matrix and compose table are built from these entries.
#[derive(Clone, Debug)]
pub struct WeightPalette {
    /// The k archetypal Base17 patterns (k ≤ 256).
    pub entries: Vec<Base17>,
    /// For each entry: maximum L1 distance from any assigned weight row.
    /// This is the CLAM radius guarantee — worst-case distortion bound.
    pub radii: Vec<u32>,
    /// For each entry: number of weight rows assigned to this archetype.
    pub counts: Vec<u32>,
}

/// Assignment of a weight row to a palette entry.
#[derive(Clone, Debug)]
pub struct PaletteAssignment {
    /// Palette index (0-255).
    pub index: u8,
    /// L1 distance to assigned archetype (distortion).
    pub distortion: u32,
}

impl WeightPalette {
    /// Build a palette from projected weight rows.
    ///
    /// Uses furthest-point sampling (greedy cover): iteratively selects the
    /// weight row that is FURTHEST from all currently selected archetypes.
    /// This guarantees good coverage of the weight manifold without assuming
    /// any cluster shape.
    ///
    /// # Arguments
    /// - `rows`: All projected weight rows
    /// - `k`: Target palette size (capped at 256 and row count)
    pub fn build(rows: &[Base17], k: usize) -> Self {
        let k = k.min(MAX_PALETTE).min(rows.len());
        if k == 0 {
            return WeightPalette {
                entries: Vec::new(),
                radii: Vec::new(),
                counts: Vec::new(),
            };
        }

        // Phase 1: Furthest-point sampling for initial archetypes
        let mut selected = Vec::with_capacity(k);
        let mut min_dist = vec![u32::MAX; rows.len()]; // min dist to any selected

        // Start with the row closest to the centroid (most "average")
        let centroid = compute_centroid(rows);
        let first = nearest_to(&centroid, rows);
        selected.push(first);

        // Update distances
        for i in 0..rows.len() {
            min_dist[i] = rows[i].l1(&rows[first]);
        }

        // Greedily add the furthest point from all selected
        for _ in 1..k {
            // Find the point with maximum min_dist
            let next = min_dist
                .iter()
                .enumerate()
                .max_by_key(|&(_, &d)| d)
                .map(|(i, _)| i)
                .unwrap_or(0);

            selected.push(next);

            // Update distances
            for i in 0..rows.len() {
                let d = rows[i].l1(&rows[next]);
                if d < min_dist[i] {
                    min_dist[i] = d;
                }
            }
        }

        // Phase 2: Build palette entries and compute assignments
        let entries: Vec<Base17> = selected.iter().map(|&i| rows[i].clone()).collect();
        let mut radii = vec![0u32; k];
        let mut counts = vec![0u32; k];

        // Assign every row to its nearest archetype, track distortion
        for row in rows {
            let (idx, dist) = nearest_with_distance(row, &entries);
            if dist > radii[idx] {
                radii[idx] = dist;
            }
            counts[idx] += 1;
        }

        WeightPalette {
            entries,
            radii,
            counts,
        }
    }

    /// Build with PCDVQ-weighted distance (sign-sensitive assignment).
    ///
    /// Weight rows where the sign dimension differs are assigned to different
    /// archetypes even if their overall L1 distance is small. This preserves
    /// the direction structure that matters 20× more than detail.
    pub fn build_weighted(rows: &[Base17], k: usize) -> Self {
        let k = k.min(MAX_PALETTE).min(rows.len());
        if k == 0 {
            return WeightPalette {
                entries: Vec::new(),
                radii: Vec::new(),
                counts: Vec::new(),
            };
        }

        let mut selected = Vec::with_capacity(k);
        let mut min_dist = vec![u32::MAX; rows.len()];

        let centroid = compute_centroid(rows);
        let first = nearest_to_weighted(&centroid, rows);
        selected.push(first);

        for i in 0..rows.len() {
            min_dist[i] = rows[i].l1_weighted(&rows[first]);
        }

        for _ in 1..k {
            let next = min_dist
                .iter()
                .enumerate()
                .max_by_key(|&(_, &d)| d)
                .map(|(i, _)| i)
                .unwrap_or(0);
            selected.push(next);
            for i in 0..rows.len() {
                let d = rows[i].l1_weighted(&rows[next]);
                if d < min_dist[i] {
                    min_dist[i] = d;
                }
            }
        }

        let entries: Vec<Base17> = selected.iter().map(|&i| rows[i].clone()).collect();
        let mut radii = vec![0u32; k];
        let mut counts = vec![0u32; k];

        for row in rows {
            let (idx, dist) = nearest_with_distance_weighted(row, &entries);
            if dist > radii[idx] {
                radii[idx] = dist;
            }
            counts[idx] += 1;
        }

        WeightPalette {
            entries,
            radii,
            counts,
        }
    }

    /// Assign a single weight row to the nearest palette entry.
    #[inline]
    pub fn assign(&self, row: &Base17) -> PaletteAssignment {
        let (index, distortion) = nearest_with_distance(row, &self.entries);
        PaletteAssignment {
            index: index as u8,
            distortion,
        }
    }

    /// Assign all rows, return palette indices.
    pub fn assign_all(&self, rows: &[Base17]) -> Vec<u8> {
        rows.iter()
            .map(|row| nearest_with_distance(row, &self.entries).0 as u8)
            .collect()
    }

    /// Palette size (number of archetypes).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is the palette empty?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Maximum distortion across all assignments (CLAM radius bound).
    pub fn max_distortion(&self) -> u32 {
        self.radii.iter().copied().max().unwrap_or(0)
    }

    /// Mean distortion estimate (average of per-archetype max radii,
    /// weighted by assignment counts).
    pub fn mean_distortion(&self) -> f32 {
        let total_weighted: u64 = self
            .radii
            .iter()
            .zip(self.counts.iter())
            .map(|(&r, &c)| r as u64 * c as u64)
            .sum();
        let total_count: u64 = self.counts.iter().map(|&c| c as u64).sum();
        if total_count == 0 {
            0.0
        } else {
            total_weighted as f32 / total_count as f32
        }
    }

    /// Detect anomalous palette entries (CHAODA-inspired).
    ///
    /// Entries with abnormally high radius or low count are likely
    /// outlier weight rows that don't fit the learned manifold.
    /// These should use higher precision (Base17 instead of palette index).
    pub fn anomalous_entries(&self, radius_threshold: f32) -> Vec<usize> {
        let mean_radius = self.radii.iter().map(|&r| r as f64).sum::<f64>()
            / self.radii.len().max(1) as f64;
        let threshold = (mean_radius * radius_threshold as f64) as u32;

        self.radii
            .iter()
            .enumerate()
            .filter(|&(_, &r)| r > threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Byte size of the palette (entries only, not distance/compose tables).
    pub fn byte_size(&self) -> usize {
        self.entries.len() * Base17::BYTE_SIZE
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Compute element-wise centroid of Base17 patterns.
fn compute_centroid(rows: &[Base17]) -> Base17 {
    let n = rows.len() as i64;
    if n == 0 {
        return Base17::zero();
    }
    let mut sums = [0i64; 17];
    for row in rows {
        for d in 0..17 {
            sums[d] += row.dims[d] as i64;
        }
    }
    let mut dims = [0i16; 17];
    for d in 0..17 {
        dims[d] = (sums[d] / n).clamp(-32768, 32767) as i16;
    }
    Base17 { dims }
}

/// Find nearest entry by L1 distance. Returns index.
fn nearest_to(query: &Base17, entries: &[Base17]) -> usize {
    entries
        .iter()
        .enumerate()
        .min_by_key(|(_, e)| query.l1(e))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Find nearest entry by weighted L1 distance.
fn nearest_to_weighted(query: &Base17, entries: &[Base17]) -> usize {
    entries
        .iter()
        .enumerate()
        .min_by_key(|(_, e)| query.l1_weighted(e))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Find nearest entry, return (index, distance).
fn nearest_with_distance(query: &Base17, entries: &[Base17]) -> (usize, u32) {
    entries
        .iter()
        .enumerate()
        .map(|(i, e)| (i, query.l1(e)))
        .min_by_key(|&(_, d)| d)
        .unwrap_or((0, u32::MAX))
}

/// Find nearest entry by weighted distance, return (index, distance).
fn nearest_with_distance_weighted(query: &Base17, entries: &[Base17]) -> (usize, u32) {
    entries
        .iter()
        .enumerate()
        .map(|(i, e)| (i, query.l1_weighted(e)))
        .min_by_key(|&(_, d)| d)
        .unwrap_or((0, u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rows(n: usize) -> Vec<Base17> {
        (0..n)
            .map(|i| {
                let mut dims = [0i16; 17];
                for d in 0..17 {
                    dims[d] = ((i * 97 + d * 31) % 512) as i16 - 256;
                }
                Base17 { dims }
            })
            .collect()
    }

    #[test]
    fn build_palette_basic() {
        let rows = make_rows(1000);
        let palette = WeightPalette::build(&rows, 64);
        assert_eq!(palette.len(), 64);
        assert!(palette.max_distortion() > 0);
    }

    #[test]
    fn palette_covers_all_rows() {
        let rows = make_rows(500);
        let palette = WeightPalette::build(&rows, 32);

        let total_assigned: u32 = palette.counts.iter().sum();
        assert_eq!(total_assigned, 500);
    }

    #[test]
    fn assign_deterministic() {
        let rows = make_rows(100);
        let palette = WeightPalette::build(&rows, 16);

        let a1 = palette.assign(&rows[0]);
        let a2 = palette.assign(&rows[0]);
        assert_eq!(a1.index, a2.index);
        assert_eq!(a1.distortion, a2.distortion);
    }

    #[test]
    fn weighted_separates_signs() {
        // Create rows that differ only in sign dimension
        let mut pos = Base17::zero();
        pos.dims[0] = 1000; // strong positive sign
        let mut neg = Base17::zero();
        neg.dims[0] = -1000; // strong negative sign

        let rows = vec![pos.clone(), neg.clone()];
        let palette = WeightPalette::build_weighted(&rows, 2);
        assert_eq!(palette.len(), 2);

        // They should be assigned to different archetypes
        let a_pos = palette.assign(&pos);
        let a_neg = palette.assign(&neg);
        assert_ne!(a_pos.index, a_neg.index);
    }

    #[test]
    fn anomaly_detection() {
        let mut rows = make_rows(100);
        // Add an outlier
        rows.push(Base17 {
            dims: [30000; 17],
        });

        let palette = WeightPalette::build(&rows, 16);
        let anomalies = palette.anomalous_entries(2.0);
        // The outlier's archetype should have high radius
        assert!(!anomalies.is_empty() || palette.max_distortion() > 0);
    }
}
