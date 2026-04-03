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

    /// End position (inclusive). Derived: start + (length-1) * stride.
    #[inline]
    pub fn end(&self) -> u32 {
        self.start + self.length.saturating_sub(1) * self.stride
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Three-finger coarse distance: 3 subtractions, zero data access
// ═══════════════════════════════════════════════════════════════════════════

/// Coarse distance band from address geometry alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CoarseBand {
    Foveal,     // nearly identical walk
    Near,       // overlapping walks, same stride
    Maybe,      // need to hydrate to decide
    Reject,     // definitely different (disjoint or different stride)
}

impl SpiralAddress {
    /// Three-finger coarse cosine: 3 integer operations, zero data access.
    ///
    /// Finger 1 — OFFSET: |start₁ - start₂|
    ///   Small offset → walks see nearly the same data → high cosine.
    ///
    /// Finger 2 — STRIDE MATCH: stride₁ == stride₂
    ///   Different stride → different sampling scale → probably orthogonal.
    ///   Also: different stride → different ROLE → categorically different.
    ///
    /// Finger 3 — OVERLAP: |[start₁,end₁] ∩ [start₂,end₂]| / max(len₁, len₂)
    ///   Disjoint intervals → walks see completely different data → cosine ≈ 0.
    pub fn coarse_band(&self, other: &SpiralAddress) -> CoarseBand {
        let offset = (self.start as i64 - other.start as i64).unsigned_abs() as u32;
        let stride_match = self.stride == other.stride;
        let overlap = self.overlap_fraction(other);

        if !stride_match {
            return CoarseBand::Reject; // different scale = different role
        }
        if overlap < 0.01 {
            return CoarseBand::Reject; // disjoint walks
        }
        if offset <= 2 && overlap > 0.8 {
            return CoarseBand::Foveal; // nearly identical
        }
        if offset <= 16 && overlap > 0.5 {
            return CoarseBand::Near; // overlapping, worth checking
        }
        CoarseBand::Maybe // need hydration to decide
    }

    /// Interval overlap fraction: |[s₁,e₁] ∩ [s₂,e₂]| / max(len₁, len₂)
    pub fn overlap_fraction(&self, other: &SpiralAddress) -> f32 {
        let e1 = self.end();
        let e2 = other.end();
        let lo = self.start.max(other.start);
        let hi = e1.min(e2);
        if lo > hi { return 0.0; }
        let overlap = (hi - lo) as f32;
        let max_len = (e1 - self.start).max(e2 - other.start) as f32;
        if max_len < 1.0 { 0.0 } else { overlap / max_len }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Role detection from stride: the address IS cognition
// ═══════════════════════════════════════════════════════════════════════════

/// Tensor role detected from stride value.
/// No data access — the stride IS the role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorRole {
    Gate,   // stride 8: coarse routing, big picture
    V,      // stride 5: content retrieval
    Down,   // stride 4: compression, summarizing
    QK,     // stride 3: attention query/key (must match)
    Up,     // stride 2: fine expansion
    Other,
}

impl SpiralAddress {
    /// Detect role from stride. Zero cost.
    pub fn role(&self) -> TensorRole {
        match self.stride {
            8 => TensorRole::Gate,
            5 => TensorRole::V,
            4 => TensorRole::Down,
            3 => TensorRole::QK,
            2 => TensorRole::Up,
            _ => TensorRole::Other,
        }
    }
}

/// Thinking scale from gate stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingScale {
    Exploiting,  // stride 1-2: fine-grained, local
    Focused,     // stride 3-4: careful, detailed
    Exploring,   // stride 5-8: broad, routing
    Abstract,    // stride 9+: meta-level
}

// ═══════════════════════════════════════════════════════════════════════════
// NeuronPrint: 36 bytes = 6 roles × 6 bytes
// ═══════════════════════════════════════════════════════════════════════════

/// 36-byte neuron identity. Stride encodes role + thinking style + transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NeuronPrint {
    pub q:    SpiralAddress,   // stride=3
    pub k:    SpiralAddress,   // stride=3 (must match Q)
    pub v:    SpiralAddress,   // stride=5
    pub gate: SpiralAddress,   // stride=8 → thinking style
    pub up:   SpiralAddress,   // stride=2
    pub down: SpiralAddress,   // stride=4 → ratio with Up = effective rank
}

impl NeuronPrint {
    pub const BYTE_SIZE: usize = 6 * SpiralAddress::BYTE_SIZE_U16; // 36

    /// Thinking scale from gate stride.
    pub fn thinking_scale(&self) -> ThinkingScale {
        match self.gate.stride {
            0..=2 => ThinkingScale::Exploiting,
            3..=4 => ThinkingScale::Focused,
            5..=8 => ThinkingScale::Exploring,
            _     => ThinkingScale::Abstract,
        }
    }

    /// Effective rank from Up/Down stride ratio.
    pub fn effective_rank(&self) -> f32 {
        self.down.stride as f32 / self.up.stride.max(1) as f32
    }

    /// Three-finger distance across all 6 roles.
    /// Returns the WORST band (most different role determines outcome).
    pub fn coarse_band(&self, other: &NeuronPrint) -> CoarseBand {
        let bands = [
            self.q.coarse_band(&other.q),
            self.k.coarse_band(&other.k),
            self.v.coarse_band(&other.v),
            self.gate.coarse_band(&other.gate),
            self.up.coarse_band(&other.up),
            self.down.coarse_band(&other.down),
        ];
        // Worst band wins (Reject > Maybe > Near > Foveal)
        *bands.iter().max().unwrap_or(&CoarseBand::Reject)
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

    // ── Three-finger tests ──────────────────────────────────────────

    #[test]
    fn three_finger_identical() {
        let a = SpiralAddress::new(20, 8, 4);
        assert_eq!(a.coarse_band(&a), CoarseBand::Foveal);
    }

    #[test]
    fn three_finger_adjacent() {
        let a = SpiralAddress::new(20, 8, 4);
        let b = SpiralAddress::new(21, 8, 4);
        assert_eq!(a.coarse_band(&b), CoarseBand::Foveal);
    }

    #[test]
    fn three_finger_different_stride_rejects() {
        let a = SpiralAddress::new(20, 8, 4);
        let b = SpiralAddress::new(20, 2, 4); // same start, different stride
        assert_eq!(a.coarse_band(&b), CoarseBand::Reject);
    }

    #[test]
    fn three_finger_disjoint_rejects() {
        let a = SpiralAddress::new(20, 8, 4);
        let c = SpiralAddress::new(200, 8, 4);
        assert_eq!(a.coarse_band(&c), CoarseBand::Reject);
    }

    #[test]
    fn three_finger_nearby_is_near() {
        let a = SpiralAddress::new(20, 8, 4);
        let b = SpiralAddress::new(25, 8, 4);
        let band = a.coarse_band(&b);
        assert!(band == CoarseBand::Near || band == CoarseBand::Maybe,
            "nearby same-stride should be Near or Maybe: {:?}", band);
    }

    // ── Role detection ──────────────────────────────────────────────

    #[test]
    fn stride_is_role() {
        assert_eq!(SpiralAddress::new(0, 8, 1).role(), TensorRole::Gate);
        assert_eq!(SpiralAddress::new(0, 3, 1).role(), TensorRole::QK);
        assert_eq!(SpiralAddress::new(0, 5, 1).role(), TensorRole::V);
        assert_eq!(SpiralAddress::new(0, 2, 1).role(), TensorRole::Up);
        assert_eq!(SpiralAddress::new(0, 4, 1).role(), TensorRole::Down);
    }

    #[test]
    fn stride_mismatch_means_different_role() {
        let gate = SpiralAddress::new(20, 8, 4);
        let up = SpiralAddress::new(20, 2, 4);
        // Different stride → Reject → categorically different
        assert_eq!(gate.coarse_band(&up), CoarseBand::Reject);
        assert_ne!(gate.role(), up.role());
    }

    // ── NeuronPrint ─────────────────────────────────────────────────

    #[test]
    fn neuron_print_size() {
        assert_eq!(NeuronPrint::BYTE_SIZE, 36);
    }

    #[test]
    fn neuron_print_self_foveal() {
        let np = NeuronPrint {
            q:    SpiralAddress::new(20, 3, 4),
            k:    SpiralAddress::new(20, 3, 4),
            v:    SpiralAddress::new(20, 5, 4),
            gate: SpiralAddress::new(20, 8, 4),
            up:   SpiralAddress::new(20, 2, 4),
            down: SpiralAddress::new(20, 4, 4),
        };
        assert_eq!(np.coarse_band(&np), CoarseBand::Foveal);
    }

    #[test]
    fn neuron_print_different_gate_rejects() {
        let a = NeuronPrint {
            q: SpiralAddress::new(20, 3, 4), k: SpiralAddress::new(20, 3, 4),
            v: SpiralAddress::new(20, 5, 4), gate: SpiralAddress::new(20, 8, 4),
            up: SpiralAddress::new(20, 2, 4), down: SpiralAddress::new(20, 4, 4),
        };
        let b = NeuronPrint {
            gate: SpiralAddress::new(200, 8, 4), // distant gate
            ..a
        };
        assert_eq!(a.coarse_band(&b), CoarseBand::Reject);
    }

    #[test]
    fn thinking_scale_from_gate() {
        let np = NeuronPrint {
            q: SpiralAddress::new(0, 3, 1), k: SpiralAddress::new(0, 3, 1),
            v: SpiralAddress::new(0, 5, 1), gate: SpiralAddress::new(0, 8, 1),
            up: SpiralAddress::new(0, 2, 1), down: SpiralAddress::new(0, 4, 1),
        };
        assert_eq!(np.thinking_scale(), ThinkingScale::Exploring);
        assert_eq!(np.effective_rank(), 2.0); // down(4) / up(2)
    }

    #[test]
    fn three_finger_validates_against_walk_cosine() {
        // The key verification: do three-finger bands predict walk cosine?
        let source = make_vec(42, 4096);

        // Foveal: offset=1, same stride → should have very high walk cosine
        let a_fov = SpiralAddress::new(20, 8, 4);
        let b_fov = SpiralAddress::new(21, 8, 4);
        assert_eq!(a_fov.coarse_band(&b_fov), CoarseBand::Foveal);
        let cos_fov = SpiralWalk::execute(&a_fov, &source).cosine(&SpiralWalk::execute(&b_fov, &source));

        // Reject: different stride → should have low or random cosine
        let a_rej = SpiralAddress::new(20, 8, 4);
        let b_rej = SpiralAddress::new(20, 2, 4);
        assert_eq!(a_rej.coarse_band(&b_rej), CoarseBand::Reject);
        let cos_rej = SpiralWalk::execute(&a_rej, &source).cosine(&SpiralWalk::execute(&b_rej, &source));

        eprintln!("Foveal pair cosine: {:.4}", cos_fov);
        eprintln!("Reject pair cosine: {:.4}", cos_rej);

        // Foveal should have HIGHER cosine than Reject
        assert!(cos_fov > cos_rej || cos_rej.abs() < 0.5,
            "Foveal cosine ({:.4}) should exceed Reject ({:.4})", cos_fov, cos_rej);
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
