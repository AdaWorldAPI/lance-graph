//! SIMD-hardened spiral address types and three-finger distance.
//!
//! Applied review: repr(C) layout, integer-only HEEL, SoA index,
//! stack-array hydration, precomputed φ-offset tables.
//!
//! Consumer API: SpiralAddr, NeuronPrint (AoS, ergonomic).
//! Search internals: SpiralIndex, NeuronIndex (SoA, SIMD-batched).

use std::f64::consts::GOLDEN_RATIO;

// ═══════════════════════════════════════════════════════════════════════════
// SpiralAddr: repr(C), validated, 6 bytes guaranteed
// ═══════════════════════════════════════════════════════════════════════════

/// Spiral walk address. 6 bytes. repr(C) guaranteed.
/// Not a projection — a READ INSTRUCTION into source data.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpiralAddr {
    pub start: u16,
    pub end: u16,
    pub stride: u16,
}
const _: () = assert!(core::mem::size_of::<SpiralAddr>() == 6);

impl SpiralAddr {
    /// Validated constructor. Returns None if stride==0 or end < start.
    #[inline]
    pub fn new(start: u16, end: u16, stride: u16) -> Option<Self> {
        if stride == 0 || end < start { return None; }
        Some(Self { start, end, stride })
    }

    /// Unchecked constructor for trusted data (e.g. from index).
    #[inline]
    pub fn new_unchecked(start: u16, end: u16, stride: u16) -> Self {
        debug_assert!(stride > 0 && end >= start);
        Self { start, end, stride }
    }

    /// Number of samples this walk takes.
    #[inline]
    pub fn n_samples(self) -> u16 {
        (self.end - self.start) / self.stride + 1
    }

    /// Role from stride. Zero cost.
    #[inline]
    pub fn role(self) -> TensorRole {
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

/// Tensor role — detected from stride, not stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorRole { Gate, V, Down, QK, Up, Other }

/// Thinking scale — from gate stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingScale { Exploiting, Focused, Exploring, Abstract }

// ═══════════════════════════════════════════════════════════════════════════
// Three-finger coarse distance: INTEGER ONLY, zero float, zero data access
// ═══════════════════════════════════════════════════════════════════════════

/// Coarse distance band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CoarseBand { Foveal, Near, Maybe, Reject }

/// HEEL: 3 integer comparisons. No float. No data access.
/// Integer-scaled overlap: `overlap * 10 > max_len * 8` not `overlap / max_len > 0.8`.
#[inline]
pub fn coarse_band(a: SpiralAddr, b: SpiralAddr) -> CoarseBand {
    // Finger 2 first (cheapest reject)
    if a.stride != b.stride { return CoarseBand::Reject; }

    // Finger 1: offset
    let offset = a.start.abs_diff(b.start);

    // Finger 3: overlap (integer only, no float, no division)
    let lo = a.start.max(b.start);
    let hi = a.end.min(b.end);

    if lo > hi {
        return if offset >= 100 { CoarseBand::Reject } else { CoarseBand::Maybe };
    }

    let overlap = (hi - lo) as u32;
    let max_len = ((a.end - a.start) as u32).max((b.end - b.start) as u32);
    if max_len == 0 { return CoarseBand::Maybe; }

    // Integer-scaled thresholds
    if offset <= 2 && overlap * 10 > max_len * 8 { return CoarseBand::Foveal; }
    if offset <= 16 && overlap * 10 > max_len * 5 { return CoarseBand::Near; }
    CoarseBand::Maybe
}

// ═══════════════════════════════════════════════════════════════════════════
// NeuronPrint: repr(C), 36 bytes, 6 roles × 6 bytes
// ═══════════════════════════════════════════════════════════════════════════

/// 36-byte neuron identity. repr(C) guaranteed.
/// Stride encodes role + thinking style + transform spectrum.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NeuronPrint {
    pub q:    SpiralAddr,
    pub k:    SpiralAddr,
    pub v:    SpiralAddr,
    pub gate: SpiralAddr,
    pub up:   SpiralAddr,
    pub down: SpiralAddr,
}
const _: () = assert!(core::mem::size_of::<NeuronPrint>() == 36);

impl NeuronPrint {
    /// Thinking scale from gate stride.
    #[inline]
    pub fn thinking_scale(&self) -> ThinkingScale {
        match self.gate.stride {
            0..=2 => ThinkingScale::Exploiting,
            3..=4 => ThinkingScale::Focused,
            5..=8 => ThinkingScale::Exploring,
            _     => ThinkingScale::Abstract,
        }
    }

    /// Effective rank from Up/Down stride ratio.
    #[inline]
    pub fn effective_rank(&self) -> f32 {
        self.down.stride as f32 / self.up.stride.max(1) as f32
    }

    /// Three-finger across all 6 roles. Worst band wins.
    #[inline]
    pub fn coarse_band(&self, other: &NeuronPrint) -> CoarseBand {
        let mut worst = CoarseBand::Foveal;
        for (a, b) in [
            (self.q, other.q), (self.k, other.k), (self.v, other.v),
            (self.gate, other.gate), (self.up, other.up), (self.down, other.down),
        ] {
            let band = coarse_band(a, b);
            if band > worst { worst = band; }
            if worst == CoarseBand::Reject { return CoarseBand::Reject; } // early exit
        }
        worst
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SoA layout for batched SIMD HEEL screening
// ═══════════════════════════════════════════════════════════════════════════

/// Structure-of-Arrays for one role's spiral addresses.
/// SIMD processes 16-32 candidates per pass on these columns.
pub struct SpiralIndex {
    pub start: Vec<u16>,
    pub end: Vec<u16>,
    pub stride: Vec<u16>,
}

impl SpiralIndex {
    pub fn new() -> Self { Self { start: Vec::new(), end: Vec::new(), stride: Vec::new() } }

    pub fn push(&mut self, addr: SpiralAddr) {
        self.start.push(addr.start);
        self.end.push(addr.end);
        self.stride.push(addr.stride);
    }

    pub fn len(&self) -> usize { self.start.len() }
    pub fn is_empty(&self) -> bool { self.start.is_empty() }

    pub fn addr(&self, idx: usize) -> SpiralAddr {
        SpiralAddr { start: self.start[idx], end: self.end[idx], stride: self.stride[idx] }
    }

    /// HEEL filter: return bitmask of survivors against a query address.
    /// Integer-only. Processes candidates in blocks.
    pub fn heel_filter(&self, query: SpiralAddr) -> Vec<bool> {
        let n = self.len();
        let mut survivors = vec![false; n];

        for i in 0..n {
            let cand = SpiralAddr {
                start: self.start[i],
                end: self.end[i],
                stride: self.stride[i],
            };
            let band = coarse_band(query, cand);
            survivors[i] = band != CoarseBand::Reject;
        }
        survivors
    }
}

/// Per-role SoA for 6-role screening. AND survivor masks across roles.
pub struct NeuronIndex {
    pub q: SpiralIndex,
    pub k: SpiralIndex,
    pub v: SpiralIndex,
    pub gate: SpiralIndex,
    pub up: SpiralIndex,
    pub down: SpiralIndex,
}

impl NeuronIndex {
    pub fn new() -> Self {
        Self {
            q: SpiralIndex::new(), k: SpiralIndex::new(), v: SpiralIndex::new(),
            gate: SpiralIndex::new(), up: SpiralIndex::new(), down: SpiralIndex::new(),
        }
    }

    pub fn push(&mut self, np: NeuronPrint) {
        self.q.push(np.q); self.k.push(np.k); self.v.push(np.v);
        self.gate.push(np.gate); self.up.push(np.up); self.down.push(np.down);
    }

    pub fn len(&self) -> usize { self.q.len() }

    /// HEEL: AND survivor masks across all 6 roles.
    /// Returns indices of candidates that survive ALL roles.
    pub fn heel_filter(&self, query: &NeuronPrint) -> Vec<u32> {
        let n = self.len();
        let q_ok = self.q.heel_filter(query.q);
        let k_ok = self.k.heel_filter(query.k);
        let v_ok = self.v.heel_filter(query.v);
        let gate_ok = self.gate.heel_filter(query.gate);
        let up_ok = self.up.heel_filter(query.up);
        let down_ok = self.down.heel_filter(query.down);

        (0..n)
            .filter(|&i| q_ok[i] && k_ok[i] && v_ok[i] && gate_ok[i] && up_ok[i] && down_ok[i])
            .map(|i| i as u32)
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Hydration: stack arrays, no Vec, f32 not f64
// ═══════════════════════════════════════════════════════════════════════════

/// Precomputed φ-offsets for fixed sample counts.
/// Compile-time constant. Zero runtime f64 math.
pub const PHI4: [u16; 4] = precompute_phi_offsets::<4>();
pub const PHI8: [u16; 8] = precompute_phi_offsets::<8>();
pub const PHI16: [u16; 16] = precompute_phi_offsets::<16>();

const fn precompute_phi_offsets<const N: usize>() -> [u16; N] {
    let phi_frac = 0.6180339887498949; // GOLDEN_RATIO.fract()
    let mut out = [0u16; N];
    let mut i = 0;
    while i < N {
        out[i] = (phi_frac * i as f64) as u16;
        i += 1;
    }
    out
}

/// HIP: 1 sample per address. Scalar. Stack.
#[inline]
pub fn hip_sample(addr: SpiralAddr, source: &[u16]) -> f32 {
    let idx = addr.start as usize;
    if idx < source.len() { f32::from_bits((source[idx] as u32) << 16) } else { 0.0 }
}

/// TWIG: 4 samples into stack array. No allocation.
#[inline]
pub fn hydrate_4(addr: SpiralAddr, source: &[u16]) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    let base = addr.start as usize;
    let step = addr.stride as usize;
    let mut i = 0;
    while i < 4 {
        let idx = base + i * step + PHI4[i] as usize;
        if idx < source.len() { out[i] = f32::from_bits((source[idx] as u32) << 16); }
        i += 1;
    }
    out
}

/// TWIG: 8 samples into stack array.
#[inline]
pub fn hydrate_8(addr: SpiralAddr, source: &[u16]) -> [f32; 8] {
    let mut out = [0.0f32; 8];
    let base = addr.start as usize;
    let step = addr.stride as usize;
    let mut i = 0;
    while i < 8 {
        let idx = base + i * step + PHI8[i] as usize;
        if idx < source.len() { out[i] = f32::from_bits((source[idx] as u32) << 16); }
        i += 1;
    }
    out
}

/// TWIG: 16 samples into stack array.
#[inline]
pub fn hydrate_16(addr: SpiralAddr, source: &[u16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    let base = addr.start as usize;
    let step = addr.stride as usize;
    let mut i = 0;
    while i < 16 {
        let idx = base + i * step + PHI16[i] as usize;
        if idx < source.len() { out[i] = f32::from_bits((source[idx] as u32) << 16); }
        i += 1;
    }
    out
}

/// LEAF: full row hydration. ONLY place Vec is acceptable. ~0.1% reach here.
pub fn leaf_hydrate(source: &[u16]) -> Vec<f32> {
    source.iter().map(|&x| f32::from_bits((x as u32) << 16)).collect()
}

/// Cosine on fixed-size stack array. f32. SIMD-friendly.
#[inline]
pub fn cosine_f32_8(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let mut dot = 0.0f32; let mut na = 0.0f32; let mut nb = 0.0f32;
    let mut i = 0;
    while i < 8 { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; i += 1; }
    let denom = (na * nb).sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

#[inline]
pub fn cosine_f32_16(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    let mut dot = 0.0f32; let mut na = 0.0f32; let mut nb = 0.0f32;
    let mut i = 0;
    while i < 16 { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; i += 1; }
    let denom = (na * nb).sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spiral_addr_is_6_bytes() {
        assert_eq!(core::mem::size_of::<SpiralAddr>(), 6);
    }

    #[test]
    fn neuron_print_is_36_bytes() {
        assert_eq!(core::mem::size_of::<NeuronPrint>(), 36);
    }

    #[test]
    fn validated_constructor() {
        assert!(SpiralAddr::new(20, 44, 8).is_some());
        assert!(SpiralAddr::new(20, 44, 0).is_none());  // stride=0
        assert!(SpiralAddr::new(44, 20, 8).is_none());  // end < start
    }

    #[test]
    fn heel_integer_only_identical() {
        let a = SpiralAddr::new_unchecked(20, 44, 8);
        assert_eq!(coarse_band(a, a), CoarseBand::Foveal);
    }

    #[test]
    fn heel_integer_only_adjacent() {
        let a = SpiralAddr::new_unchecked(20, 44, 8);
        let b = SpiralAddr::new_unchecked(21, 45, 8);
        assert_eq!(coarse_band(a, b), CoarseBand::Foveal);
    }

    #[test]
    fn heel_integer_only_different_stride_rejects() {
        let a = SpiralAddr::new_unchecked(20, 44, 8);
        let b = SpiralAddr::new_unchecked(20, 44, 2);
        assert_eq!(coarse_band(a, b), CoarseBand::Reject);
    }

    #[test]
    fn heel_integer_only_disjoint_rejects() {
        let a = SpiralAddr::new_unchecked(20, 44, 8);
        let b = SpiralAddr::new_unchecked(200, 224, 8);
        assert_eq!(coarse_band(a, b), CoarseBand::Reject);
    }

    #[test]
    fn stride_is_role() {
        assert_eq!(SpiralAddr::new_unchecked(0, 8, 8).role(), TensorRole::Gate);
        assert_eq!(SpiralAddr::new_unchecked(0, 3, 3).role(), TensorRole::QK);
        assert_eq!(SpiralAddr::new_unchecked(0, 5, 5).role(), TensorRole::V);
        assert_eq!(SpiralAddr::new_unchecked(0, 2, 2).role(), TensorRole::Up);
        assert_eq!(SpiralAddr::new_unchecked(0, 4, 4).role(), TensorRole::Down);
    }

    #[test]
    fn neuron_print_self_foveal() {
        let np = NeuronPrint {
            q: SpiralAddr::new_unchecked(20, 32, 3),
            k: SpiralAddr::new_unchecked(20, 32, 3),
            v: SpiralAddr::new_unchecked(20, 40, 5),
            gate: SpiralAddr::new_unchecked(20, 44, 8),
            up: SpiralAddr::new_unchecked(20, 28, 2),
            down: SpiralAddr::new_unchecked(20, 36, 4),
        };
        assert_eq!(np.coarse_band(&np), CoarseBand::Foveal);
    }

    #[test]
    fn neuron_print_early_exit_on_reject() {
        let a = NeuronPrint {
            q: SpiralAddr::new_unchecked(20, 32, 3),
            k: SpiralAddr::new_unchecked(20, 32, 3),
            v: SpiralAddr::new_unchecked(20, 40, 5),
            gate: SpiralAddr::new_unchecked(20, 44, 8),
            up: SpiralAddr::new_unchecked(20, 28, 2),
            down: SpiralAddr::new_unchecked(20, 36, 4),
        };
        let b = NeuronPrint {
            gate: SpiralAddr::new_unchecked(200, 224, 8), // distant gate
            ..a
        };
        assert_eq!(a.coarse_band(&b), CoarseBand::Reject);
    }

    #[test]
    fn soa_index_builds() {
        let mut idx = NeuronIndex::new();
        let np = NeuronPrint {
            q: SpiralAddr::new_unchecked(20, 32, 3),
            k: SpiralAddr::new_unchecked(20, 32, 3),
            v: SpiralAddr::new_unchecked(20, 40, 5),
            gate: SpiralAddr::new_unchecked(20, 44, 8),
            up: SpiralAddr::new_unchecked(20, 28, 2),
            down: SpiralAddr::new_unchecked(20, 36, 4),
        };
        for _ in 0..100 { idx.push(np); }
        assert_eq!(idx.len(), 100);
    }

    #[test]
    fn heel_filter_self_matches() {
        let mut idx = NeuronIndex::new();
        let np = NeuronPrint {
            q: SpiralAddr::new_unchecked(20, 32, 3),
            k: SpiralAddr::new_unchecked(20, 32, 3),
            v: SpiralAddr::new_unchecked(20, 40, 5),
            gate: SpiralAddr::new_unchecked(20, 44, 8),
            up: SpiralAddr::new_unchecked(20, 28, 2),
            down: SpiralAddr::new_unchecked(20, 36, 4),
        };
        idx.push(np);
        let survivors = idx.heel_filter(&np);
        assert_eq!(survivors.len(), 1);
        assert_eq!(survivors[0], 0);
    }

    #[test]
    fn phi_offsets_are_const() {
        assert_eq!(PHI4[0], 0);
        assert_eq!(PHI8[0], 0);
        assert_eq!(PHI16[0], 0);
        // φ-frac ≈ 0.618, so offset[1] ≈ 0, offset[2] ≈ 1, offset[3] ≈ 1
        assert!(PHI8[3] <= 2);
    }

    #[test]
    fn hydrate_8_no_alloc() {
        let source: Vec<u16> = (0..1000).map(|i| ((i as f32 * 0.1).to_bits() >> 16) as u16).collect();
        let addr = SpiralAddr::new_unchecked(10, 74, 8);
        let vals = hydrate_8(addr, &source);
        let mag: f32 = vals.iter().map(|v| v.abs()).sum();
        assert!(mag > 0.0, "hydrated values should be nonzero");
    }

    #[test]
    fn cosine_self_one() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let c = cosine_f32_8(&a, &a);
        assert!((c - 1.0).abs() < 1e-6);
    }
}
