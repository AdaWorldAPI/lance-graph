//! SPO (Subject-Predicate-Object) triple encoding.
//!
//! 36 bits per triple: 12 bits subject + 12 bits predicate + 12 bits object.
//! Packed in a u64 (upper 28 bits zero). This is the atomic unit of meaning
//! in the DeepNSM semantic engine.
//!
//! Distance between two triples = 3 matrix lookups (one per role).
//! Total: 4 memory accesses for full similarity. < 15ns.

/// A semantic triple: Subject(who) - Predicate(what) - Object(whom).
///
/// Packed as `[S:12][P:12][O:12]` in 36 bits of a u64.
/// Each component is a 12-bit vocabulary rank (0-4095).
///
/// # Examples
/// ```
/// # use deepnsm::spo::SpoTriple;
/// let triple = SpoTriple::new(671, 2943, 95); // "dog bites man"
/// assert_eq!(triple.subject(), 671);
/// assert_eq!(triple.predicate(), 2943);
/// assert_eq!(triple.object(), 95);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpoTriple {
    /// Packed: [S:12][P:12][O:12] = 36 bits, stored in u64 (upper 28 bits zero).
    packed: u64,
}

/// Sentinel value for "no role" (e.g., intransitive verb has no object).
pub const NO_ROLE: u16 = 0xFFF; // 4095 — last valid index, but reserved as sentinel

impl SpoTriple {
    /// Create a new triple from three 12-bit vocabulary ranks.
    ///
    /// # Panics
    /// Debug-asserts that all ranks are < 4096.
    #[inline]
    pub fn new(subject: u16, predicate: u16, object: u16) -> Self {
        debug_assert!(subject < 4096 && predicate < 4096 && object < 4096);
        Self {
            packed: ((subject as u64) << 24) | ((predicate as u64) << 12) | object as u64,
        }
    }

    /// Create with optional object (intransitive verbs).
    #[inline]
    pub fn transitive(subject: u16, predicate: u16, object: u16) -> Self {
        Self::new(subject, predicate, object)
    }

    /// Create without object (intransitive: "the dog runs").
    #[inline]
    pub fn intransitive(subject: u16, predicate: u16) -> Self {
        Self::new(subject, predicate, NO_ROLE)
    }

    /// Extract subject rank (bits 24-35).
    #[inline]
    pub fn subject(&self) -> u16 {
        ((self.packed >> 24) & 0xFFF) as u16
    }

    /// Extract predicate rank (bits 12-23).
    #[inline]
    pub fn predicate(&self) -> u16 {
        ((self.packed >> 12) & 0xFFF) as u16
    }

    /// Extract object rank (bits 0-11).
    #[inline]
    pub fn object(&self) -> u16 {
        (self.packed & 0xFFF) as u16
    }

    /// Does this triple have an object?
    #[inline]
    pub fn has_object(&self) -> bool {
        self.object() != NO_ROLE
    }

    /// Raw packed u64 value.
    #[inline]
    pub fn as_u64(&self) -> u64 {
        self.packed
    }

    /// Construct from raw packed u64.
    #[inline]
    pub fn from_u64(packed: u64) -> Self {
        debug_assert!(packed < (1u64 << 36));
        Self { packed }
    }

    /// Distance to another triple via distance matrix.
    /// Sum of per-role distances: 3 matrix lookups.
    #[inline]
    pub fn distance(&self, other: &SpoTriple, matrix: &WordDistanceMatrix) -> u32 {
        let ds = matrix.get(self.subject(), other.subject()) as u32;
        let dp = matrix.get(self.predicate(), other.predicate()) as u32;
        let d_o = if self.has_object() && other.has_object() {
            matrix.get(self.object(), other.object()) as u32
        } else {
            0
        };
        ds + dp + d_o
    }

    /// Per-role distances: (subject_dist, predicate_dist, object_dist).
    #[inline]
    pub fn distance_per_role(
        &self,
        other: &SpoTriple,
        matrix: &WordDistanceMatrix,
    ) -> (u8, u8, u8) {
        (
            matrix.get(self.subject(), other.subject()),
            matrix.get(self.predicate(), other.predicate()),
            if self.has_object() && other.has_object() {
                matrix.get(self.object(), other.object())
            } else {
                0
            },
        )
    }

    /// Similarity via SimilarityTable. Per-role similarities combined.
    #[inline]
    pub fn similarity(
        &self,
        other: &SpoTriple,
        matrix: &WordDistanceMatrix,
        table: &super::similarity::SimilarityTable,
    ) -> f32 {
        let d = self.distance(other, matrix);
        table.lookup(d)
    }
}

impl core::fmt::Debug for SpoTriple {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "SPO({}, {}, {})",
            self.subject(),
            self.predicate(),
            self.object()
        )
    }
}

impl core::fmt::Display for SpoTriple {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.has_object() {
            write!(
                f,
                "({} → {} → {})",
                self.subject(),
                self.predicate(),
                self.object()
            )
        } else {
            write!(f, "({} → {})", self.subject(), self.predicate())
        }
    }
}

// ─── Distance Matrix (4096² u8) ─────────────────────────────────────────────

/// Precomputed 4096×4096 symmetric distance matrix.
///
/// Each entry is a u8 palette-quantized distance (256 levels).
/// Upper triangle stored in row-major order.
/// Total: 4096 × 4096 = 16MB (full) or ~8MB (upper triangle).
/// Fits in L2 cache.
///
/// For the DeepNSM tokenizer, distances are computed from 96D COCA
/// subgenre frequency vectors via the CAM-PQ codebook.
pub struct WordDistanceMatrix {
    /// Full symmetric matrix: data[a * 4096 + b] = distance(a, b).
    data: Vec<u8>,
}

impl WordDistanceMatrix {
    /// Vocabulary size.
    pub const K: usize = 4096;

    /// Create from flat data. Must be exactly K×K bytes.
    pub fn from_flat(data: Vec<u8>) -> Self {
        assert_eq!(data.len(), Self::K * Self::K);
        Self { data }
    }

    /// Look up distance between two vocabulary ranks. O(1).
    #[inline]
    pub fn get(&self, a: u16, b: u16) -> u8 {
        self.data[a as usize * Self::K + b as usize]
    }

    /// Build from a distance function over vocabulary entries.
    /// Calls `dist_fn(i, j)` for all pairs, must return u8.
    pub fn build<F>(dist_fn: F) -> Self
    where
        F: Fn(u16, u16) -> u8,
    {
        let mut data = vec![0u8; Self::K * Self::K];
        for i in 0..Self::K as u16 {
            for j in (i + 1)..Self::K as u16 {
                let d = dist_fn(i, j);
                data[i as usize * Self::K + j as usize] = d;
                data[j as usize * Self::K + i as usize] = d;
            }
        }
        Self { data }
    }

    /// Build from CAM-PQ codes and codebook.
    /// This is the primary construction method for DeepNSM.
    pub fn build_from_cam(
        cam_codes: &[[u8; 6]],
        codebook: &[f32], // [6][256][16] flat
    ) -> Self {
        let k = cam_codes.len().min(Self::K);

        // First pass: compute all raw distances (f32)
        // to find the distribution for palette quantization
        let pair_count = k * (k - 1) / 2;
        let mut raw_distances = Vec::with_capacity(pair_count);

        for i in 0..k {
            for j in (i + 1)..k {
                let d = cam_l2_distance(&cam_codes[i], &cam_codes[j], codebook);
                raw_distances.push(d);
            }
        }

        // Find min/max for quantization
        let min_d = raw_distances.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_d = raw_distances
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let range = (max_d - min_d).max(1e-6);

        // Quantize to u8 and fill matrix
        let mut data = vec![0u8; Self::K * Self::K];
        let mut idx = 0;
        for i in 0..k {
            for j in (i + 1)..k {
                let d = raw_distances[idx];
                let quantized = (((d - min_d) / range) * 255.0).round() as u8;
                data[i * Self::K + j] = quantized;
                data[j * Self::K + i] = quantized;
                idx += 1;
            }
        }

        Self { data }
    }

    /// Byte size of the matrix.
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Access raw data slice.
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
}

/// Compute L2 distance between two CAM-PQ codes via codebook lookup.
///
/// Each code is 6 bytes (one centroid index per subspace).
/// Codebook is [6][256][16] f32 (6 subspaces, 256 centroids, 16 dims each).
///
/// Total: 6 × 16 = 96 dimension comparisons via precomputed centroids.
fn cam_l2_distance(a: &[u8; 6], b: &[u8; 6], codebook: &[f32]) -> f32 {
    let mut dist_sq = 0.0f32;
    for s in 0..6 {
        let base = s * 256 * 16;
        let offset_a = base + a[s] as usize * 16;
        let offset_b = base + b[s] as usize * 16;
        for d in 0..16 {
            let diff = codebook[offset_a + d] - codebook[offset_b + d];
            dist_sq += diff * diff;
        }
    }
    dist_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack() {
        let t = SpoTriple::new(671, 2943, 95);
        assert_eq!(t.subject(), 671);
        assert_eq!(t.predicate(), 2943);
        assert_eq!(t.object(), 95);
        assert!(t.has_object());
    }

    #[test]
    fn intransitive() {
        let t = SpoTriple::intransitive(671, 100);
        assert_eq!(t.subject(), 671);
        assert_eq!(t.predicate(), 100);
        assert!(!t.has_object());
    }

    #[test]
    fn max_values() {
        let t = SpoTriple::new(4095, 4095, 4094);
        assert_eq!(t.subject(), 4095);
        assert_eq!(t.predicate(), 4095);
        assert_eq!(t.object(), 4094);
    }

    #[test]
    fn zero_triple() {
        let t = SpoTriple::new(0, 0, 0);
        assert_eq!(t.as_u64(), 0);
    }

    #[test]
    fn roundtrip_u64() {
        let t = SpoTriple::new(1234, 567, 89);
        let packed = t.as_u64();
        let t2 = SpoTriple::from_u64(packed);
        assert_eq!(t, t2);
    }

    #[test]
    fn display() {
        let t = SpoTriple::new(671, 2943, 95);
        assert_eq!(format!("{}", t), "(671 → 2943 → 95)");

        let t2 = SpoTriple::intransitive(671, 100);
        assert_eq!(format!("{}", t2), "(671 → 100)");
    }
}
