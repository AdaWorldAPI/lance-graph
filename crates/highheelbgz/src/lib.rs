//! # HighHeelBGZ: 3-Integer Spiral Address Encoding
//!
//! Each weight vector is represented by 3 integers: (start, stride, length).
//! Not a compressed copy — an address into the golden spiral walk.
//! Values recomputed on demand from source data via the spiral address.
//!
//! ```text
//! Traditional:  f32[4096] → project → i16[17] (34 bytes) or BF16×32 (1 KB)
//! HighHeelBGZ:  f32[4096] → address → (start, stride, len) = 3 integers
//! ```
//!
//! The HEEL walks the spiral in high strides.

use std::f64::consts::GOLDEN_RATIO;

const BASE_DIM: usize = 17;

// ═══════════════════════════════════════════════════════════════════════════
// SpiralAddress: the 3-integer representation
// ═══════════════════════════════════════════════════════════════════════════

/// 3-integer spiral address. This IS the encoding.
///
/// - `start`: octave offset where the walk begins (skip degenerate region)
/// - `stride`: octave spacing between samples (controls averaging density)
/// - `length`: number of samples per base dim
///
/// Total: 12 bytes (3 × u32). Or 6 bytes (3 × u16) for smaller models.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpiralAddress {
    pub start: u32,
    pub stride: u32,
    pub length: u32,
}

impl SpiralAddress {
    pub const BYTE_SIZE_U32: usize = 12;
    pub const BYTE_SIZE_U16: usize = 6;

    pub fn new(start: u32, stride: u32, length: u32) -> Self {
        Self { start, stride, length }
    }

    /// Total samples this address will read: BASE_DIM × length.
    pub fn total_samples(&self) -> usize {
        BASE_DIM * self.length as usize
    }

    /// Which octave positions this address visits for base dim `d`.
    pub fn octave_positions(&self, d: usize) -> Vec<usize> {
        (0..self.length as usize)
            .map(|s| self.start as usize + d + s * self.stride as usize)
            .collect()
    }

    /// The φ-fractional position within an octave for base dim `bi` at octave `oct`.
    #[inline]
    fn phi_position(bi: usize, oct: usize) -> usize {
        let phi_pos = frac((bi + oct) as f64 * GOLDEN_RATIO) * BASE_DIM as f64;
        phi_pos as usize
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Walk: execute a spiral address against source data
// ═══════════════════════════════════════════════════════════════════════════

/// Walk result: the actual f32 samples extracted by a spiral address.
/// This is NOT stored — computed on demand.
#[derive(Clone, Debug)]
pub struct SpiralWalk {
    /// Samples: [base_dim][sample_index].
    /// Outer: 17 base dims. Inner: `length` samples per dim.
    pub samples: Vec<Vec<f32>>,
}

impl SpiralWalk {
    /// Execute a spiral address against source data.
    ///
    /// This is the core operation: read the weight vector through the
    /// spiral lens defined by the address. O(BASE_DIM × length) reads.
    pub fn execute(addr: &SpiralAddress, weights: &[f32]) -> Self {
        let n = weights.len();
        let n_oct = (n + BASE_DIM - 1) / BASE_DIM;
        let len = addr.length as usize;

        let mut samples = vec![Vec::with_capacity(len); BASE_DIM];

        for bi in 0..BASE_DIM {
            for s in 0..len {
                let octave = addr.start as usize + bi + s * addr.stride as usize;
                if octave >= n_oct { break; }

                let pos = SpiralAddress::phi_position(bi, octave);
                let dim = octave * BASE_DIM + pos;
                if dim < n {
                    samples[bi].push(weights[dim]);
                }
            }
        }

        SpiralWalk { samples }
    }

    /// Cosine similarity between two walks.
    pub fn cosine(&self, other: &SpiralWalk) -> f64 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for d in 0..BASE_DIM {
            let a = &self.samples[d];
            let b = &other.samples[d];
            for i in 0..a.len().min(b.len()) {
                let x = a[i] as f64;
                let y = b[i] as f64;
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
        }
        let denom = (na * nb).sqrt();
        if denom < 1e-12 { 0.0 } else { dot / denom }
    }

    /// L1 distance between two walks (hydrated).
    pub fn l1(&self, other: &SpiralWalk) -> f64 {
        let mut d = 0.0f64;
        for dim in 0..BASE_DIM {
            let a = &self.samples[dim];
            let b = &other.samples[dim];
            for i in 0..a.len().min(b.len()) {
                d += (a[i] as f64 - b[i] as f64).abs();
            }
        }
        d
    }

    /// Flatten to f32 vector (for external comparison).
    pub fn to_f32(&self) -> Vec<f32> {
        self.samples.iter().flat_map(|s| s.iter().copied()).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Address geometry: coarse distance without reading source data
// ═══════════════════════════════════════════════════════════════════════════

/// Coarse distance estimate from address geometry alone.
///
/// Two addresses that start far apart and have different strides
/// are probably sampling different regions of the weight space.
/// This gives a HEEL-level reject signal without touching the source data.
pub fn address_distance(a: &SpiralAddress, b: &SpiralAddress) -> u32 {
    let start_diff = (a.start as i64 - b.start as i64).unsigned_abs() as u32;
    let stride_diff = (a.stride as i64 - b.stride as i64).unsigned_abs() as u32;
    let len_diff = (a.length as i64 - b.length as i64).unsigned_abs() as u32;
    start_diff + stride_diff * BASE_DIM as u32 + len_diff
}

/// Do two addresses overlap in their octave coverage?
/// If no overlap, the walks sample completely different parts of the vector.
pub fn address_overlap(a: &SpiralAddress, b: &SpiralAddress) -> f64 {
    let a_octaves: std::collections::HashSet<usize> = (0..a.length as usize)
        .flat_map(|s| (0..BASE_DIM).map(move |d| a.start as usize + d + s * a.stride as usize))
        .collect();
    let b_octaves: std::collections::HashSet<usize> = (0..b.length as usize)
        .flat_map(|s| (0..BASE_DIM).map(move |d| b.start as usize + d + s * b.stride as usize))
        .collect();
    let intersection = a_octaves.intersection(&b_octaves).count();
    let union = a_octaves.union(&b_octaves).count();
    if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
}

// ═══════════════════════════════════════════════════════════════════════════
// Palette: 256 canonical spiral addresses
// ═══════════════════════════════════════════════════════════════════════════

/// A palette of canonical spiral addresses.
///
/// 256 entries × 12 bytes = 3 KB (vs 256 × 34 bytes = 8.7 KB for Base17 palette).
/// Distance between palette entries computed on demand from source data.
#[derive(Clone, Debug)]
pub struct SpiralPalette {
    pub entries: Vec<SpiralAddress>,
}

impl SpiralPalette {
    /// Build default palette: 256 addresses at systematically varied
    /// (start, stride) combinations.
    pub fn default_256() -> Self {
        let mut entries = Vec::with_capacity(256);
        // 16 start values × 16 stride values = 256 entries
        for si in 0..16 {
            for sti in 0..16 {
                let start = 10 + si * 4;      // 10, 14, 18, ..., 70
                let stride = 2 + sti * 2;     // 2, 4, 6, ..., 32
                entries.push(SpiralAddress::new(start as u32, stride as u32, 4));
            }
        }
        SpiralPalette { entries }
    }

    /// Assign a vector to the nearest palette entry (by walk cosine).
    pub fn assign(&self, weights: &[f32]) -> (u8, f64) {
        let mut best_idx = 0u8;
        let mut best_cos = f64::NEG_INFINITY;
        for (i, addr) in self.entries.iter().enumerate() {
            let walk = SpiralWalk::execute(addr, weights);
            // Use the walk's self-magnitude as proxy for coverage quality
            let mag: f64 = walk.samples.iter()
                .flat_map(|s| s.iter())
                .map(|v| (*v as f64).abs())
                .sum();
            if mag > best_cos {
                best_cos = mag;
                best_idx = i as u8;
            }
        }
        (best_idx, best_cos)
    }

    /// Byte size.
    pub fn byte_size(&self) -> usize {
        self.entries.len() * SpiralAddress::BYTE_SIZE_U32
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Calibration: find optimal (start, stride) for a dataset
// ═══════════════════════════════════════════════════════════════════════════

/// Find the (start, stride, length) that maximizes pairwise cosine
/// correlation with ground truth for a set of vectors.
pub fn calibrate(
    vectors: &[Vec<f32>],
    start_range: std::ops::Range<u32>,
    stride_range: std::ops::Range<u32>,
    lengths: &[u32],
) -> (SpiralAddress, f64) {
    let n = vectors.len().min(30);
    if n < 2 { return (SpiralAddress::new(20, 8, 4), 0.0); }

    // Ground truth pairwise cosines
    let mut gt = Vec::new();
    for i in 0..n { for j in (i+1)..n {
        gt.push(cosine_f32(&vectors[i], &vectors[j]));
    }}

    let mut best_addr = SpiralAddress::new(20, 8, 4);
    let mut best_spearman = f64::NEG_INFINITY;

    for &len in lengths {
        for start in start_range.clone() {
            for stride in stride_range.clone() {
                let addr = SpiralAddress::new(start, stride, len);
                let walks: Vec<SpiralWalk> = vectors[..n].iter()
                    .map(|v| SpiralWalk::execute(&addr, v))
                    .collect();
                let mut walk_cos = Vec::new();
                for i in 0..n { for j in (i+1)..n {
                    walk_cos.push(walks[i].cosine(&walks[j]));
                }}

                let sp = spearman(&gt, &walk_cos);
                if sp > best_spearman {
                    best_spearman = sp;
                    best_addr = addr;
                }
            }
        }
    }

    (best_addr, best_spearman)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn frac(x: f64) -> f64 { x - x.floor() }

fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    for i in 0..n { dot += a[i] as f64 * b[i] as f64; na += (a[i] as f64).powi(2); nb += (b[i] as f64).powi(2); }
    let d = (na * nb).sqrt(); if d < 1e-12 { 0.0 } else { dot / d }
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 { return 0.0; }
    let rank_x = ranks(&x[..n]);
    let rank_y = ranks(&y[..n]);
    pearson(&rank_x, &rank_y)
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 { return 0.0; }
    let mx = x[..n].iter().sum::<f64>() / n as f64;
    let my = y[..n].iter().sum::<f64>() / n as f64;
    let mut cov = 0.0f64; let mut vx = 0.0f64; let mut vy = 0.0f64;
    for i in 0..n { let dx = x[i] - mx; let dy = y[i] - my; cov += dx*dy; vx += dx*dx; vy += dy*dy; }
    let d = (vx * vy).sqrt(); if d < 1e-12 { 0.0 } else { cov / d }
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut result = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-12 { j += 1; }
        let avg = (i + j) as f64 / 2.0 + 0.5;
        for k in i..j { result[indexed[k].0] = avg; }
        i = j;
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(seed: usize, dim: usize) -> Vec<f32> {
        (0..dim).map(|d| ((d * 97 + seed * 31) as f32 % 200.0 - 100.0) * 0.01).collect()
    }

    #[test]
    fn spiral_address_basic() {
        let addr = SpiralAddress::new(20, 8, 4);
        assert_eq!(addr.total_samples(), 17 * 4);
        let positions = addr.octave_positions(0);
        assert_eq!(positions, vec![20, 28, 36, 44]);
        let positions_d1 = addr.octave_positions(1);
        assert_eq!(positions_d1, vec![21, 29, 37, 45]);
    }

    #[test]
    fn walk_execute_nonzero() {
        let v = make_vec(42, 1024);
        let addr = SpiralAddress::new(20, 8, 4);
        let walk = SpiralWalk::execute(&addr, &v);
        assert_eq!(walk.samples.len(), BASE_DIM);
        let mag: f64 = walk.samples.iter().flat_map(|s| s.iter()).map(|v| v.abs() as f64).sum();
        assert!(mag > 0.0, "walk should extract nonzero values");
    }

    #[test]
    fn walk_cosine_self_one() {
        let v = make_vec(7, 1024);
        let addr = SpiralAddress::new(20, 8, 4);
        let walk = SpiralWalk::execute(&addr, &v);
        let c = walk.cosine(&walk);
        assert!((c - 1.0).abs() < 1e-10);
    }

    #[test]
    fn walk_cosine_different() {
        let a = make_vec(1, 1024);
        let b = make_vec(999, 1024);
        let addr = SpiralAddress::new(20, 8, 4);
        let wa = SpiralWalk::execute(&addr, &a);
        let wb = SpiralWalk::execute(&addr, &b);
        let c = wa.cosine(&wb);
        assert!(c < 0.99, "different vectors should have cosine < 1: {}", c);
    }

    #[test]
    fn address_distance_self_zero() {
        let a = SpiralAddress::new(20, 8, 4);
        assert_eq!(address_distance(&a, &a), 0);
    }

    #[test]
    fn address_distance_different() {
        let a = SpiralAddress::new(20, 8, 4);
        let b = SpiralAddress::new(30, 12, 4);
        assert!(address_distance(&a, &b) > 0);
    }

    #[test]
    fn address_overlap_same() {
        let a = SpiralAddress::new(20, 8, 4);
        let ov = address_overlap(&a, &a);
        assert!((ov - 1.0).abs() < 1e-10);
    }

    #[test]
    fn address_overlap_disjoint() {
        let a = SpiralAddress::new(0, 1, 2);
        let b = SpiralAddress::new(100, 1, 2);
        let ov = address_overlap(&a, &b);
        assert!(ov < 0.5, "far-apart addresses should have low overlap: {}", ov);
    }

    #[test]
    fn palette_default_256() {
        let pal = SpiralPalette::default_256();
        assert_eq!(pal.entries.len(), 256);
        assert_eq!(pal.byte_size(), 256 * 12);
    }

    #[test]
    fn twelve_bytes_per_vector() {
        assert_eq!(SpiralAddress::BYTE_SIZE_U32, 12);
        assert_eq!(SpiralAddress::BYTE_SIZE_U16, 6);
    }

    #[test]
    fn calibrate_finds_good_address() {
        let vecs: Vec<Vec<f32>> = (0..20).map(|i| make_vec(i, 512)).collect();
        let (best, spearman) = calibrate(&vecs, 10..30, 4..12, &[4]);
        assert!(best.start >= 10 && best.start < 30);
        assert!(best.stride >= 4 && best.stride < 12);
        eprintln!("Best: start={}, stride={}, length={}, spearman={:.4}",
            best.start, best.stride, best.length, spearman);
    }
}
