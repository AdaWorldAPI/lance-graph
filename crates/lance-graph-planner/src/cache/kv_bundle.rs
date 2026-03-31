//! KV Bundle: VSA superposition store for attention cache.
//!
//! Fixed-size i16 arrays. Bundle/unbundle in O(1).
//! The holographic property: every fragment contains the whole.
//! 4096 attention heads × 17 dims = 69632 i16 values = 136 KB.
//!
//! Three resolution levels (HHTL):
//!   HEEL:  8×8   = 64 entries   (512 bytes, routing decisions)
//!   HIP:   64×64 = 4096 entries (32 KB, attention topology)
//!   TWIG:  256×256 = 65536 entries (512 KB, fine-grain)

/// Type alias: HeadPrint is ndarray's Base17 (17 × i16 = 34 bytes).
/// All downstream code continues to use the `HeadPrint` name unchanged.
pub use ndarray::hpc::bgz17_bridge::Base17 as HeadPrint;

const BASE_DIM: usize = 17;

/// Bundle: weighted addition (majority vote analog for i16).
pub fn bundle_into(source: &HeadPrint, target: &mut HeadPrint, weight_self: f32, weight_new: f32) {
    let total = weight_self + weight_new;
    for d in 0..BASE_DIM {
        let old = target.dims[d] as f32 * weight_self;
        let new = source.dims[d] as f32 * weight_new;
        target.dims[d] = ((old + new) / total).round() as i16;
    }
}

/// Unbundle: subtract out (XOR analog for i16).
pub fn unbundle_from(source: &HeadPrint, target: &mut HeadPrint) {
    for d in 0..BASE_DIM {
        target.dims[d] = target.dims[d].wrapping_sub(source.dims[d]);
    }
}

/// Attention matrix at HIP level: 64×64 = 4096 heads.
/// Each cell is a HeadPrint representing the attention between head i and head j.
/// Interdependent: head[i][j] influences head[i+1][k] through the residual stream.
#[derive(Clone)]
pub struct AttentionMatrix {
    /// 64×64 heads, row-major. heads[i*64+j] = attention from head i to head j.
    pub heads: Vec<HeadPrint>,
    /// Resolution: 64 for HIP, 256 for TWIG.
    pub resolution: usize,
    /// Bundle of ALL heads (the "gestalt" — holographic summary).
    pub gestalt: HeadPrint,
    /// Number of updates applied.
    pub epoch: u32,
}

impl AttentionMatrix {
    pub fn new_hip() -> Self {
        Self {
            heads: vec![HeadPrint::zero(); 64 * 64],
            resolution: 64,
            gestalt: HeadPrint::zero(),
            epoch: 0,
        }
    }

    pub fn new_twig() -> Self {
        Self {
            heads: vec![HeadPrint::zero(); 256 * 256],
            resolution: 256,
            gestalt: HeadPrint::zero(),
            epoch: 0,
        }
    }

    /// Get attention head at (row, col).
    pub fn get(&self, row: usize, col: usize) -> &HeadPrint {
        &self.heads[row * self.resolution + col]
    }

    /// Set attention head and update gestalt.
    pub fn set(&mut self, row: usize, col: usize, head: HeadPrint) {
        let idx = row * self.resolution + col;
        // Unbundle old from gestalt
        let old = self.heads[idx].clone();
        unbundle_from(&old, &mut self.gestalt);
        // Bundle new into gestalt
        bundle_into(&head, &mut self.gestalt, self.epoch as f32, 1.0);
        self.heads[idx] = head;
        self.epoch += 1;
    }

    /// Surprise: how different is a new head from the gestalt?
    /// High = unexpected = high free energy = attend to this.
    pub fn surprise(&self, head: &HeadPrint) -> f32 {
        let max_l1 = (BASE_DIM as u32 * 65535) as f32;
        self.gestalt.l1(head) as f32 / max_l1
    }

    /// Topic shift: how different are two rows (two perspectives)?
    pub fn row_divergence(&self, row_a: usize, row_b: usize) -> f32 {
        let max_l1 = (BASE_DIM as u32 * 65535) as f32;
        let mut total = 0u64;
        for col in 0..self.resolution {
            total += self.get(row_a, col).l1(self.get(row_b, col)) as u64;
        }
        total as f32 / (self.resolution as f32 * max_l1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_headprint_bundle_unbundle() {
        let a = HeadPrint {
            dims: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170],
        };
        let b = HeadPrint {
            dims: [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165],
        };

        // Bundle a and b into a target with equal weight
        let mut target = HeadPrint::zero();
        bundle_into(&a, &mut target, 0.0, 1.0); // first item: target becomes a
        assert_eq!(target, a);

        bundle_into(&b, &mut target, 1.0, 1.0); // second item: average of a and b
        for d in 0..BASE_DIM {
            let expected = ((a.dims[d] as f32 + b.dims[d] as f32) / 2.0).round() as i16;
            assert_eq!(target.dims[d], expected, "dim {d} mismatch");
        }

        // Unbundle b from target: should shift back toward a
        let before_unbundle = target.clone();
        unbundle_from(&b, &mut target);
        // After unbundle, each dim should be before - b
        for d in 0..BASE_DIM {
            let expected = before_unbundle.dims[d].wrapping_sub(b.dims[d]);
            assert_eq!(target.dims[d], expected, "unbundle dim {d} mismatch");
        }
    }

    #[test]
    fn test_attention_matrix_hip() {
        let mut mat = AttentionMatrix::new_hip();
        assert_eq!(mat.heads.len(), 64 * 64);
        assert_eq!(mat.resolution, 64);

        let head = HeadPrint {
            dims: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        };
        mat.set(3, 7, head.clone());
        assert_eq!(mat.get(3, 7), &head);
        assert_eq!(mat.epoch, 1);

        // Setting another head increments epoch
        let head2 = HeadPrint {
            dims: [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        };
        mat.set(10, 20, head2.clone());
        assert_eq!(mat.get(10, 20), &head2);
        assert_eq!(mat.epoch, 2);
    }

    #[test]
    fn test_surprise() {
        let mut mat = AttentionMatrix::new_hip();

        // Surprise of zero head against zero gestalt should be 0
        let zero = HeadPrint::zero();
        assert_eq!(mat.surprise(&zero), 0.0);

        // Set some heads to shift the gestalt, then check surprise
        let head = HeadPrint {
            dims: [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
        };
        mat.set(0, 0, head.clone());

        // Surprise of the same head should be low (gestalt moved toward it)
        let s_same = mat.surprise(&head);

        // Surprise of opposite head should be higher
        let opposite = HeadPrint {
            dims: [-100, -200, -300, -400, -500, -600, -700, -800, -900, -1000, -1100, -1200, -1300, -1400, -1500, -1600, -1700],
        };
        let s_opposite = mat.surprise(&opposite);
        assert!(
            s_opposite > s_same,
            "Opposite head should be more surprising: {s_opposite} vs {s_same}"
        );
    }

    #[test]
    fn test_row_divergence() {
        let mut mat = AttentionMatrix::new_hip();

        // Two identical rows should have zero divergence
        assert_eq!(mat.row_divergence(0, 1), 0.0);

        // Set row 0 col 0 to something non-zero
        let head = HeadPrint {
            dims: [1000; BASE_DIM],
        };
        mat.set(0, 0, head);

        // Now row 0 and row 1 should diverge
        let div = mat.row_divergence(0, 1);
        assert!(div > 0.0, "Rows should diverge after setting head: {div}");

        // Row 0 with itself should still be zero
        assert_eq!(mat.row_divergence(0, 0), 0.0);
    }
}
