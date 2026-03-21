//! # ZeckBF17: Golden-Step Octave Knowledge Compression
//!
//! ## The Insight
//!
//! 16,384 accumulator dimensions are NOT 16,384 independent values.
//! They are 17 base dimensions × 964 octaves.
//! Most octaves carry holographic redundancy.
//! Only ~14 octaves carry independent information (JL bound: 57 / log₂(17) ≈ 14).
//!
//! ## Why 17
//!
//! - 17 is PRIME: no non-trivial subspace decomposition.
//! - Golden-ratio step (`round(17/φ) = 11`) visits ALL 17 residues.
//!   `gcd(11, 17) = 1` — full coverage, maximally decorrelated.
//! - 16 = 2⁴ would alias. The 17th dimension IS the Pythagorean comma.
//!
//! NOTE: An earlier version claimed Fibonacci mod 17 visits all 17 residues.
//! WRONG — Fibonacci mod 17 visits only 13 (missing {6,7,10,11}).
//! The golden-ratio STEP is the correct traversal.
//!
//! ## Format
//!
//! ```text
//! i8[16384] per plane  →  i16[17] base + u8[14] envelope  =  48 bytes
//! Compression: 341:1 per plane, 424:1 per S/P/O edge (116 bytes)
//! ```
//!
//! ## i16 Fixed-Point Base (not BF16)
//!
//! The base stores the average accumulator value per base dimension
//! as i16 fixed-point with 8 fractional bits: `stored = round(mean × 256)`.
//! Range: [-32768, +32767] covers any i8 mean with sub-unit precision.
//! i16 gives 256× finer quantization than BF16's 7-bit mantissa,
//! at zero extra storage cost (both are 2 bytes).
//!
//! ## Distance Metric
//!
//! Matches the production ZeckF64 pipeline in `neighborhood/zeckf64.rs`:
//! per-plane Hamming distance on sign bits → threshold → scent byte + quantiles.
//! For ZeckBF17 bases, we approximate this with L1 distance on i16 values,
//! then quantile-map to ZeckF64-compatible bytes.

/// The prime base dimensionality.
pub const BASE_DIM: usize = 17;

/// Full accumulator dimensionality.
pub const FULL_DIM: usize = 16384;

/// Number of octaves: ceil(FULL_DIM / BASE_DIM).
pub const N_OCTAVES: usize = (FULL_DIM + BASE_DIM - 1) / BASE_DIM; // 964

/// Independent octaves (JL bound).
pub const INDEPENDENT_OCTAVES: usize = 14;

/// Fixed-point scale: i16 = round(mean × FP_SCALE).
pub const FP_SCALE: f64 = 256.0;

/// Golden-ratio step: round(17/φ) = 11. gcd(11,17)=1 → full coverage.
pub const GOLDEN_STEP: usize = 11;

/// Golden-step traversal table: all 17 positions exactly once.
/// [0, 11, 5, 16, 10, 4, 15, 9, 3, 14, 8, 2, 13, 7, 1, 12, 6]
const GOLDEN_POS_17: [u8; BASE_DIM] = {
    let mut t = [0u8; BASE_DIM];
    let mut i = 0;
    while i < BASE_DIM {
        t[i] = ((i * GOLDEN_STEP) % BASE_DIM) as u8;
        i += 1;
    }
    t
};

/// Map base index → dimension position within an octave.
#[inline]
fn golden_position(idx: usize) -> usize {
    GOLDEN_POS_17[idx % BASE_DIM] as usize
}

/// Verify golden-step visits all 17 positions.
pub fn verify_golden_coverage() -> bool {
    let mut seen = [false; BASE_DIM];
    for i in 0..BASE_DIM {
        seen[GOLDEN_POS_17[i] as usize] = true;
    }
    seen.iter().all(|&s| s)
}

// ═══════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════

/// 17-dimensional base pattern in i16 fixed-point (8 fractional bits).
/// 34 bytes. The "pitch class" of the accumulator.
#[derive(Clone, Debug, PartialEq)]
pub struct BasePattern {
    pub dims: [i16; BASE_DIM],
}

/// Octave envelope: amplitude scale per independent octave group. 14 bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct OctaveEnvelope {
    pub amplitudes: [u8; INDEPENDENT_OCTAVES],
}

/// One ZeckBF17-encoded plane. 48 bytes.
#[derive(Clone, Debug)]
pub struct ZeckBF17Plane {
    pub base: BasePattern,
    pub envelope: OctaveEnvelope,
}

/// One ZeckBF17-encoded S/P/O edge. 116 bytes.
/// Envelope is shared (property of the node, not the plane).
#[derive(Clone, Debug)]
pub struct ZeckBF17Edge {
    pub subject: BasePattern,
    pub predicate: BasePattern,
    pub object: BasePattern,
    pub envelope: OctaveEnvelope,
}

// ═══════════════════════════════════════════════════════════════════════
// Plane encode / decode
// ═══════════════════════════════════════════════════════════════════════

impl ZeckBF17Plane {
    pub const ENCODED_SIZE: usize = BASE_DIM * 2 + INDEPENDENT_OCTAVES; // 48

    /// Encode i8[16384] → ZeckBF17 (48 bytes).
    ///
    /// Step 1: For each of 17 base dims, average across all 964 octaves.
    ///         Store as i16 fixed-point (×256).
    /// Step 2: For each of 14 octave groups, compute RMS ratio actual/predicted.
    pub fn encode(acc: &[i8]) -> Self {
        assert!(acc.len() >= FULL_DIM);

        // Step 1: base pattern = mean across octaves per golden-position group
        let mut sum = [0i64; BASE_DIM];
        let mut count = [0u32; BASE_DIM];

        for octave in 0..N_OCTAVES {
            for bi in 0..BASE_DIM {
                let dim = octave * BASE_DIM + golden_position(bi);
                if dim < FULL_DIM {
                    sum[bi] += acc[dim] as i64;
                    count[bi] += 1;
                }
            }
        }

        let mut base = BasePattern { dims: [0i16; BASE_DIM] };
        for d in 0..BASE_DIM {
            if count[d] > 0 {
                let mean = sum[d] as f64 / count[d] as f64;
                base.dims[d] = (mean * FP_SCALE).round().clamp(-32768.0, 32767.0) as i16;
            }
        }

        // Step 2: octave envelope
        let octaves_per_group = N_OCTAVES / INDEPENDENT_OCTAVES;
        let mut envelope = OctaveEnvelope { amplitudes: [0u8; INDEPENDENT_OCTAVES] };

        for group in 0..INDEPENDENT_OCTAVES {
            let mut rms = 0.0f64;
            let mut cnt = 0u32;

            for sub in 0..octaves_per_group {
                let octave = group * octaves_per_group + sub;
                if octave >= N_OCTAVES { break; }

                for bi in 0..BASE_DIM {
                    let dim = octave * BASE_DIM + golden_position(bi);
                    if dim < FULL_DIM {
                        let actual = acc[dim] as f64;
                        let predicted = base.dims[bi] as f64 / FP_SCALE;
                        let ratio = if predicted.abs() > 0.01 {
                            (actual / predicted).abs()
                        } else {
                            actual.abs()
                        };
                        rms += ratio * ratio;
                        cnt += 1;
                    }
                }
            }

            if cnt > 0 {
                let r = (rms / cnt as f64).sqrt();
                envelope.amplitudes[group] = (r * 128.0).min(255.0) as u8;
            }
        }

        ZeckBF17Plane { base, envelope }
    }

    /// Decode ZeckBF17 → i8[16384].
    pub fn decode(&self) -> Vec<i8> {
        let mut out = vec![0i8; FULL_DIM];
        let octaves_per_group = N_OCTAVES / INDEPENDENT_OCTAVES;

        for octave in 0..N_OCTAVES {
            let group = (octave / octaves_per_group).min(INDEPENDENT_OCTAVES - 1);
            let scale = self.envelope.amplitudes[group] as f32 / 128.0;

            for bi in 0..BASE_DIM {
                let dim = octave * BASE_DIM + golden_position(bi);
                if dim < FULL_DIM {
                    let base_val = self.base.dims[bi] as f32 / FP_SCALE as f32;
                    let v = base_val * scale;
                    out[dim] = v.clamp(-128.0, 127.0) as i8;
                }
            }
        }
        out
    }

    /// Serialize to 48 bytes.
    pub fn to_bytes(&self) -> [u8; Self::ENCODED_SIZE] {
        let mut buf = [0u8; Self::ENCODED_SIZE];
        for i in 0..BASE_DIM {
            let b = self.base.dims[i].to_le_bytes();
            buf[i * 2] = b[0];
            buf[i * 2 + 1] = b[1];
        }
        for i in 0..INDEPENDENT_OCTAVES {
            buf[BASE_DIM * 2 + i] = self.envelope.amplitudes[i];
        }
        buf
    }

    /// Deserialize from 48 bytes.
    pub fn from_bytes(buf: &[u8; Self::ENCODED_SIZE]) -> Self {
        let mut base = BasePattern { dims: [0i16; BASE_DIM] };
        for i in 0..BASE_DIM {
            base.dims[i] = i16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
        }
        let mut envelope = OctaveEnvelope { amplitudes: [0u8; INDEPENDENT_OCTAVES] };
        for i in 0..INDEPENDENT_OCTAVES {
            envelope.amplitudes[i] = buf[BASE_DIM * 2 + i];
        }
        ZeckBF17Plane { base, envelope }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Edge encode / decode
// ═══════════════════════════════════════════════════════════════════════

impl ZeckBF17Edge {
    pub const ENCODED_SIZE: usize = BASE_DIM * 2 * 3 + INDEPENDENT_OCTAVES; // 116

    /// Encode three i8[16384] planes → one 116-byte edge.
    pub fn encode(s: &[i8], p: &[i8], o: &[i8]) -> Self {
        let sp = ZeckBF17Plane::encode(s);
        let pp = ZeckBF17Plane::encode(p);
        let op = ZeckBF17Plane::encode(o);

        // Shared envelope: average of three
        let mut envelope = OctaveEnvelope { amplitudes: [0u8; INDEPENDENT_OCTAVES] };
        for i in 0..INDEPENDENT_OCTAVES {
            let avg = (sp.envelope.amplitudes[i] as u16
                + pp.envelope.amplitudes[i] as u16
                + op.envelope.amplitudes[i] as u16) / 3;
            envelope.amplitudes[i] = avg as u8;
        }

        ZeckBF17Edge {
            subject: sp.base,
            predicate: pp.base,
            object: op.base,
            envelope,
        }
    }

    /// Decode → three i8[16384] planes.
    pub fn decode(&self) -> (Vec<i8>, Vec<i8>, Vec<i8>) {
        let sp = ZeckBF17Plane { base: self.subject.clone(), envelope: self.envelope.clone() };
        let pp = ZeckBF17Plane { base: self.predicate.clone(), envelope: self.envelope.clone() };
        let op = ZeckBF17Plane { base: self.object.clone(), envelope: self.envelope.clone() };
        (sp.decode(), pp.decode(), op.decode())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Palette Compression: 8-bit indexed archetypes
// ═══════════════════════════════════════════════════════════════════════

/// Maximum palette size (8-bit index → 256 entries).
pub const MAX_PALETTE: usize = 256;

/// A palette of archetypal base patterns.
/// Shared codebook: 256 × 34 bytes = 8,704 bytes.
#[derive(Clone, Debug)]
pub struct BasePalette {
    pub entries: Vec<BasePattern>,
}

/// A palette-compressed edge: 3 bytes (one u8 index per plane).
#[derive(Clone, Debug)]
pub struct PaletteEdge {
    pub s_idx: u8,
    pub p_idx: u8,
    pub o_idx: u8,
}

impl PaletteEdge {
    pub const ENCODED_SIZE: usize = 3;
}

impl BasePalette {
    /// Build a palette from a collection of base patterns using k-means clustering.
    /// `patterns` is a flat list of all base patterns to cluster.
    /// `k` is the palette size (max 256).
    pub fn build(patterns: &[BasePattern], k: usize) -> Self {
        let k = k.min(MAX_PALETTE).min(patterns.len());
        if k == 0 {
            return Self { entries: Vec::new() };
        }

        // Initialize centroids: use first k distinct patterns (or spread evenly)
        let mut centroids: Vec<[f64; BASE_DIM]> = Vec::with_capacity(k);
        let step = patterns.len().max(1) / k.max(1);
        for i in 0..k {
            let idx = (i * step).min(patterns.len() - 1);
            let mut c = [0.0f64; BASE_DIM];
            for d in 0..BASE_DIM {
                c[d] = patterns[idx].dims[d] as f64;
            }
            centroids.push(c);
        }

        // K-means iterations
        let mut assignments = vec![0usize; patterns.len()];
        for _iter in 0..20 {
            let mut changed = false;

            // Assign each pattern to nearest centroid
            for (pi, pat) in patterns.iter().enumerate() {
                let mut best_dist = f64::MAX;
                let mut best_c = 0;
                for (ci, cent) in centroids.iter().enumerate() {
                    let mut d = 0.0f64;
                    for dim in 0..BASE_DIM {
                        let diff = pat.dims[dim] as f64 - cent[dim];
                        d += diff.abs();
                    }
                    if d < best_dist {
                        best_dist = d;
                        best_c = ci;
                    }
                }
                if assignments[pi] != best_c {
                    assignments[pi] = best_c;
                    changed = true;
                }
            }

            if !changed { break; }

            // Recompute centroids
            let mut sums = vec![[0.0f64; BASE_DIM]; k];
            let mut counts = vec![0usize; k];
            for (pi, pat) in patterns.iter().enumerate() {
                let c = assignments[pi];
                counts[c] += 1;
                for d in 0..BASE_DIM {
                    sums[c][d] += pat.dims[d] as f64;
                }
            }
            for ci in 0..k {
                if counts[ci] > 0 {
                    for d in 0..BASE_DIM {
                        centroids[ci][d] = sums[ci][d] / counts[ci] as f64;
                    }
                }
            }
        }

        // Convert centroids to BasePattern
        let entries = centroids.iter().map(|c| {
            let mut dims = [0i16; BASE_DIM];
            for d in 0..BASE_DIM {
                dims[d] = c[d].round().clamp(-32768.0, 32767.0) as i16;
            }
            BasePattern { dims }
        }).collect();

        Self { entries }
    }

    /// Find the nearest palette entry for a given base pattern.
    pub fn nearest(&self, pat: &BasePattern) -> u8 {
        let mut best_dist = u32::MAX;
        let mut best_idx = 0u8;
        for (i, entry) in self.entries.iter().enumerate() {
            let d = base_l1(pat, entry);
            if d < best_dist {
                best_dist = d;
                best_idx = i as u8;
            }
        }
        best_idx
    }

    /// Compress a ZeckBF17Edge to a PaletteEdge (3 bytes).
    pub fn compress(&self, edge: &ZeckBF17Edge) -> PaletteEdge {
        PaletteEdge {
            s_idx: self.nearest(&edge.subject),
            p_idx: self.nearest(&edge.predicate),
            o_idx: self.nearest(&edge.object),
        }
    }

    /// Look up the base pattern for a palette index.
    pub fn lookup(&self, idx: u8) -> &BasePattern {
        &self.entries[idx as usize]
    }

    /// Compute L1 distance between two palette edges using precomputed entries.
    pub fn edge_distance(&self, a: &PaletteEdge, b: &PaletteEdge) -> u32 {
        base_l1(self.lookup(a.s_idx), self.lookup(b.s_idx))
            + base_l1(self.lookup(a.p_idx), self.lookup(b.p_idx))
            + base_l1(self.lookup(a.o_idx), self.lookup(b.o_idx))
    }

    /// Total bytes: codebook + n_edges × 3.
    pub fn total_bytes(&self, n_edges: usize) -> usize {
        self.entries.len() * BASE_DIM * 2 + n_edges * PaletteEdge::ENCODED_SIZE
    }

    /// Build a precomputed 256×256 distance matrix for fast lookup.
    pub fn distance_matrix(&self) -> Vec<Vec<u32>> {
        let n = self.entries.len();
        let mut mat = vec![vec![0u32; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = base_l1(&self.entries[i], &self.entries[j]);
                mat[i][j] = d;
                mat[j][i] = d;
            }
        }
        mat
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Distance: L1 on i16 bases (matches ZeckF64 L1 on quantile bytes)
// ═══════════════════════════════════════════════════════════════════════

/// L1 (Manhattan) distance between two i16 base patterns.
///
/// This is the ZeckBF17 analog of `BitVec::hamming_distance()`.
/// The production pipeline: BitVec Hamming → quantile → L1 on bytes.
/// For compressed bases: i16 L1 directly approximates the same ordering.
pub fn base_l1(a: &BasePattern, b: &BasePattern) -> u32 {
    let mut d = 0u32;
    for i in 0..BASE_DIM {
        d += (a.dims[i] as i32 - b.dims[i] as i32).unsigned_abs();
    }
    d
}

/// Sign-bit agreement between two base patterns (out of 17).
pub fn base_sign_agreement(a: &BasePattern, b: &BasePattern) -> u32 {
    let mut agree = 0u32;
    for i in 0..BASE_DIM {
        if (a.dims[i] >= 0) == (b.dims[i] >= 0) {
            agree += 1;
        }
    }
    agree
}

/// Compute ZeckF64 scent byte from ZeckBF17 base patterns.
/// Produces the same 7-bit lattice as `zeckf64::zeckf64()`.
pub fn scent_from_base(a: &ZeckBF17Edge, b: &ZeckBF17Edge) -> u8 {
    let ds = base_l1(&a.subject, &b.subject);
    let dp = base_l1(&a.predicate, &b.predicate);
    let d_o = base_l1(&a.object, &b.object);

    let max_l1 = BASE_DIM as u32 * u16::MAX as u32;
    let threshold = max_l1 / 2;

    let sc = (ds < threshold) as u8;
    let pc = (dp < threshold) as u8;
    let oc = (d_o < threshold) as u8;
    let sp = sc & pc;
    let so = sc & oc;
    let po = pc & oc;
    let spo = sp & so & po;

    sc | (pc << 1) | (oc << 2) | (sp << 3) | (so << 4) | (po << 5) | (spo << 6)
}

/// Compute full ZeckF64 u64 from two ZeckBF17 edges.
/// Scent byte + 7 quantile bytes, matching production format.
pub fn zeckf64_from_base(a: &ZeckBF17Edge, b: &ZeckBF17Edge) -> u64 {
    let ds = base_l1(&a.subject, &b.subject);
    let dp = base_l1(&a.predicate, &b.predicate);
    let d_o = base_l1(&a.object, &b.object);

    let max_l1 = BASE_DIM as u64 * u16::MAX as u64;
    let threshold = (max_l1 / 2) as u32;

    let sc = (ds < threshold) as u8;
    let pc = (dp < threshold) as u8;
    let oc = (d_o < threshold) as u8;
    let sp = sc & pc;
    let so = sc & oc;
    let po = pc & oc;
    let spo = sp & so & po;

    let byte0 = sc | (pc << 1) | (oc << 2) | (sp << 3) | (so << 4) | (po << 5) | (spo << 6);

    let q1 = |d: u32| -> u8 { ((d as u64 * 255) / max_l1).min(255) as u8 };
    let q2 = |d1: u32, d2: u32| -> u8 { (((d1 as u64 + d2 as u64) * 255) / (2 * max_l1)).min(255) as u8 };
    let q3 = |d1: u32, d2: u32, d3: u32| -> u8 { (((d1 as u64 + d2 as u64 + d3 as u64) * 255) / (3 * max_l1)).min(255) as u8 };

    (byte0 as u64)
        | ((q3(ds, dp, d_o) as u64) << 8)
        | ((q2(dp, d_o) as u64) << 16)
        | ((q2(ds, d_o) as u64) << 24)
        | ((q2(ds, dp) as u64) << 32)
        | ((q1(d_o) as u64) << 40)
        | ((q1(dp) as u64) << 48)
        | ((q1(ds) as u64) << 56)
}

// ═══════════════════════════════════════════════════════════════════════
// Fidelity measurement
// ═══════════════════════════════════════════════════════════════════════

/// Sign-bit Hamming on full planes.
pub fn sign_bit_hamming(original: &[i8], reconstructed: &[i8]) -> u32 {
    assert_eq!(original.len(), reconstructed.len());
    original.iter().zip(reconstructed.iter())
        .filter(|(&a, &b)| (a >= 0) != (b >= 0))
        .count() as u32
}

/// Normalized fidelity: 1.0 = perfect.
pub fn sign_bit_fidelity(original: &[i8], reconstructed: &[i8]) -> f64 {
    1.0 - (sign_bit_hamming(original, reconstructed) as f64 / original.len() as f64)
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n < 2 { return 0.0; }
    fn ranks(v: &[f64]) -> Vec<f64> {
        let mut ix: Vec<(usize, f64)> = v.iter().copied().enumerate().collect();
        ix.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut r = vec![0.0; v.len()];
        for (rank, &(orig, _)) in ix.iter().enumerate() { r[orig] = rank as f64; }
        r
    }
    let (ra, rb) = (ranks(a), ranks(b));
    let (ma, mb) = (ra.iter().sum::<f64>() / n as f64, rb.iter().sum::<f64>() / n as f64);
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (da, db) = (ra[i] - ma, rb[i] - mb);
        cov += da * db; va += da * da; vb += db * db;
    }
    if va < 1e-15 || vb < 1e-15 { 0.0 } else { cov / (va.sqrt() * vb.sqrt()) }
}

// ═══════════════════════════════════════════════════════════════════════
// Experiment
// ═══════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct FidelityReport {
    pub s_fidelity: f64,
    pub p_fidelity: f64,
    pub o_fidelity: f64,
    pub avg_fidelity: f64,
    pub rank_correlation: f64,
    pub scent_agreement: f64,
    pub compression_ratio: f64,
    pub original_bytes: usize,
    pub compressed_bytes: usize,
}

/// All tunable knobs for the fidelity experiment.
#[derive(Clone, Debug)]
pub struct ExperimentConfig {
    pub n_nodes: usize,
    pub n_encounters: usize,
    /// Signal probability in [0.0, 1.0]. 0.7 = 70% signal, 30% noise.
    pub signal_pct: f64,
    /// Fixed-point scale for i16 base encoding. Default 256.0.
    pub fp_scale: f64,
    /// Number of independent octave groups for envelope. Default 14.
    pub n_indep_octaves: usize,
    /// Scent threshold as fraction of max_l1. Default 0.5.
    pub scent_threshold_frac: f64,
    /// Step size for traversal. Default GOLDEN_STEP (11). Must be coprime to BASE_DIM.
    pub step: usize,
    /// Optional gamma curve exponent for non-linear encoding. 1.0 = linear.
    pub gamma: f64,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            n_nodes: 20,
            n_encounters: 20,
            signal_pct: 0.7,
            fp_scale: FP_SCALE,
            n_indep_octaves: INDEPENDENT_OCTAVES,
            scent_threshold_frac: 0.5,
            step: GOLDEN_STEP,
            gamma: 1.0,
        }
    }
}

/// Build position table for any step coprime to BASE_DIM.
fn make_position_table(step: usize) -> [u8; BASE_DIM] {
    let mut t = [0u8; BASE_DIM];
    for i in 0..BASE_DIM {
        t[i] = ((i * step) % BASE_DIM) as u8;
    }
    t
}

/// Apply gamma curve: x → sign(x) * |x|^gamma
fn gamma_encode(val: f64, gamma: f64) -> f64 {
    if gamma == 1.0 { return val; }
    val.signum() * val.abs().powf(gamma)
}

pub fn fidelity_experiment(n_nodes: usize, n_encounters: usize) -> FidelityReport {
    fidelity_experiment_cfg(&ExperimentConfig {
        n_nodes,
        n_encounters,
        ..Default::default()
    })
}

pub fn fidelity_experiment_cfg(cfg: &ExperimentConfig) -> FidelityReport {
    let n_nodes = cfg.n_nodes;
    let n_encounters = cfg.n_encounters;
    let signal_threshold = (cfg.signal_pct * 10.0).round() as u64; // out of 10

    let mut rng = 42u64;
    let next = |r: &mut u64| -> u64 {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *r
    };

    // Build position table for the configured step
    let pos_table = make_position_table(cfg.step);
    let pos = |bi: usize| -> usize { pos_table[bi % BASE_DIM] as usize };

    // Build reverse mapping: for each dimension, which base index does it belong to?
    let mut dim_to_base = vec![0usize; FULL_DIM];
    for octave in 0..N_OCTAVES {
        for bi in 0..BASE_DIM {
            let dim = octave * BASE_DIM + pos(bi);
            if dim < FULL_DIM {
                dim_to_base[dim] = bi;
            }
        }
    }

    let mut nodes: Vec<(Vec<i8>, Vec<i8>, Vec<i8>)> = Vec::with_capacity(n_nodes);

    for node in 0..n_nodes {
        let mut s = vec![0i8; FULL_DIM];
        let mut p = vec![0i8; FULL_DIM];
        let mut o = vec![0i8; FULL_DIM];
        let node_seed = next(&mut rng) ^ (node as u64).wrapping_mul(0x517cc1b727220a95);

        let mut base_s = [0i8; BASE_DIM];
        let mut base_p = [0i8; BASE_DIM];
        let mut base_o = [0i8; BASE_DIM];
        for bi in 0..BASE_DIM {
            let h = node_seed.wrapping_mul(bi as u64 + 1).wrapping_add(0x9e3779b97f4a7c15);
            base_s[bi] = if (h >> 17) & 1 == 1 { 1i8 } else { -1 };
            base_p[bi] = if (h >> 23) & 1 == 1 { 1i8 } else { -1 };
            base_o[bi] = if (h >> 29) & 1 == 1 { 1i8 } else { -1 };
        }

        for d in 0..FULL_DIM {
            let bi = dim_to_base[d];
            let (mut vs, mut vp, mut vo) = (0i8, 0i8, 0i8);
            for _enc in 0..n_encounters {
                let eh = next(&mut rng);
                vs = vs.saturating_add(if (eh >> 7) % 10 < signal_threshold { base_s[bi] } else { -base_s[bi] });
                vp = vp.saturating_add(if (eh >> 13) % 10 < signal_threshold { base_p[bi] } else { -base_p[bi] });
                vo = vo.saturating_add(if (eh >> 19) % 10 < signal_threshold { base_o[bi] } else { -base_o[bi] });
            }
            s[d] = vs; p[d] = vp; o[d] = vo;
        }
        nodes.push((s, p, o));
    }

    // Encode with configurable FP_SCALE, octave groups, step, and gamma
    let encode_plane_cfg = |acc: &[i8]| -> ZeckBF17Plane {
        let mut sum = [0i64; BASE_DIM];
        let mut count = [0u32; BASE_DIM];
        for octave in 0..N_OCTAVES {
            for bi in 0..BASE_DIM {
                let dim = octave * BASE_DIM + pos(bi);
                if dim < FULL_DIM {
                    sum[bi] += acc[dim] as i64;
                    count[bi] += 1;
                }
            }
        }
        let mut base = BasePattern { dims: [0i16; BASE_DIM] };
        for d in 0..BASE_DIM {
            if count[d] > 0 {
                let mean = sum[d] as f64 / count[d] as f64;
                // Apply gamma curve before quantization
                let encoded = gamma_encode(mean, cfg.gamma);
                base.dims[d] = (encoded * cfg.fp_scale).round().clamp(-32768.0, 32767.0) as i16;
            }
        }

        let octaves_per_group = N_OCTAVES / cfg.n_indep_octaves.max(1);
        let mut envelope = OctaveEnvelope { amplitudes: [0u8; INDEPENDENT_OCTAVES] };
        for group in 0..cfg.n_indep_octaves.min(INDEPENDENT_OCTAVES) {
            let mut rms = 0.0f64;
            let mut cnt = 0u32;
            for sub in 0..octaves_per_group {
                let octave = group * octaves_per_group + sub;
                if octave >= N_OCTAVES { break; }
                for bi in 0..BASE_DIM {
                    let dim = octave * BASE_DIM + pos(bi);
                    if dim < FULL_DIM {
                        let actual = acc[dim] as f64;
                        let predicted = base.dims[bi] as f64 / cfg.fp_scale;
                        let ratio = if predicted.abs() > 0.01 {
                            (actual / predicted).abs()
                        } else {
                            actual.abs()
                        };
                        rms += ratio * ratio;
                        cnt += 1;
                    }
                }
            }
            if cnt > 0 {
                let r = (rms / cnt as f64).sqrt();
                envelope.amplitudes[group] = (r * 128.0).min(255.0) as u8;
            }
        }
        ZeckBF17Plane { base, envelope }
    };

    let encode_edge_cfg = |s: &[i8], p: &[i8], o: &[i8]| -> ZeckBF17Edge {
        let sp = encode_plane_cfg(s);
        let pp = encode_plane_cfg(p);
        let op = encode_plane_cfg(o);
        let mut envelope = OctaveEnvelope { amplitudes: [0u8; INDEPENDENT_OCTAVES] };
        for i in 0..INDEPENDENT_OCTAVES {
            let avg = (sp.envelope.amplitudes[i] as u16
                + pp.envelope.amplitudes[i] as u16
                + op.envelope.amplitudes[i] as u16) / 3;
            envelope.amplitudes[i] = avg as u8;
        }
        ZeckBF17Edge {
            subject: sp.base, predicate: pp.base, object: op.base, envelope,
        }
    };

    let encoded: Vec<ZeckBF17Edge> = nodes.iter()
        .map(|(s, p, o)| encode_edge_cfg(s, p, o)).collect();

    // Decode with configurable FP_SCALE, step, and gamma
    let decode_plane_cfg = |plane: &ZeckBF17Plane| -> Vec<i8> {
        let mut out = vec![0i8; FULL_DIM];
        let octaves_per_group = N_OCTAVES / cfg.n_indep_octaves.max(1);
        for octave in 0..N_OCTAVES {
            let group = (octave / octaves_per_group).min(cfg.n_indep_octaves.max(1) - 1)
                .min(INDEPENDENT_OCTAVES - 1);
            let scale = plane.envelope.amplitudes[group] as f32 / 128.0;
            for bi in 0..BASE_DIM {
                let dim = octave * BASE_DIM + pos(bi);
                if dim < FULL_DIM {
                    let base_val = plane.base.dims[bi] as f64 / cfg.fp_scale;
                    // Inverse gamma: base_val^(1/gamma)
                    let decoded = if cfg.gamma != 1.0 {
                        gamma_encode(base_val, 1.0 / cfg.gamma)
                    } else {
                        base_val
                    };
                    let v = decoded as f32 * scale;
                    out[dim] = v.clamp(-128.0, 127.0) as i8;
                }
            }
        }
        out
    };

    let decoded: Vec<_> = encoded.iter().map(|e| {
        let sp = ZeckBF17Plane { base: e.subject.clone(), envelope: e.envelope.clone() };
        let pp = ZeckBF17Plane { base: e.predicate.clone(), envelope: e.envelope.clone() };
        let op = ZeckBF17Plane { base: e.object.clone(), envelope: e.envelope.clone() };
        (decode_plane_cfg(&sp), decode_plane_cfg(&pp), decode_plane_cfg(&op))
    }).collect();

    let (mut sf, mut pf, mut of_) = (0.0, 0.0, 0.0);
    for i in 0..n_nodes {
        sf += sign_bit_fidelity(&nodes[i].0, &decoded[i].0);
        pf += sign_bit_fidelity(&nodes[i].1, &decoded[i].1);
        of_ += sign_bit_fidelity(&nodes[i].2, &decoded[i].2);
    }
    let (sf, pf, of_) = (sf / n_nodes as f64, pf / n_nodes as f64, of_ / n_nodes as f64);

    let n_pairs = n_nodes.min(50);
    let (mut exact_d, mut zeck_d) = (Vec::new(), Vec::new());
    for i in 0..n_pairs {
        for j in (i + 1)..n_pairs {
            exact_d.push(
                sign_bit_hamming(&nodes[i].0, &nodes[j].0) as f64
                + sign_bit_hamming(&nodes[i].1, &nodes[j].1) as f64
                + sign_bit_hamming(&nodes[i].2, &nodes[j].2) as f64);
            zeck_d.push(
                base_l1(&encoded[i].subject, &encoded[j].subject) as f64
                + base_l1(&encoded[i].predicate, &encoded[j].predicate) as f64
                + base_l1(&encoded[i].object, &encoded[j].object) as f64);
        }
    }
    let rho = if !exact_d.is_empty() { spearman(&exact_d, &zeck_d) } else { 0.0 };

    // Scent agreement with configurable threshold
    let max_l1 = BASE_DIM as u32 * u16::MAX as u32;
    let scent_thresh = (max_l1 as f64 * cfg.scent_threshold_frac) as u32;
    let full_thresh = FULL_DIM as u32 / 2;
    let (mut agree, mut total) = (0u32, 0u32);
    for i in 0..n_pairs {
        for j in (i + 1)..n_pairs {
            total += 1;
            // ZeckBF17 scent with configurable threshold
            let ds = base_l1(&encoded[i].subject, &encoded[j].subject);
            let dp = base_l1(&encoded[i].predicate, &encoded[j].predicate);
            let d_o = base_l1(&encoded[i].object, &encoded[j].object);
            let sc = (ds < scent_thresh) as u8;
            let pc = (dp < scent_thresh) as u8;
            let oc = (d_o < scent_thresh) as u8;
            let zeck_scent = sc | (pc << 1) | (oc << 2)
                | ((sc & pc) << 3) | ((sc & oc) << 4) | ((pc & oc) << 5)
                | ((sc & pc & oc) << 6);

            // Ground truth scent from full planes
            let (fds, fdp, fdo) = (
                sign_bit_hamming(&nodes[i].0, &nodes[j].0),
                sign_bit_hamming(&nodes[i].1, &nodes[j].1),
                sign_bit_hamming(&nodes[i].2, &nodes[j].2));
            let (fsc, fpc, foc) = (
                (fds < full_thresh) as u8,
                (fdp < full_thresh) as u8,
                (fdo < full_thresh) as u8);
            let full_scent = fsc | (fpc << 1) | (foc << 2)
                | ((fsc & fpc) << 3) | ((fsc & foc) << 4)
                | ((fpc & foc) << 5) | ((fsc & fpc & foc) << 6);
            if zeck_scent == full_scent { agree += 1; }
        }
    }
    let scent_agr = if total > 0 { agree as f64 / total as f64 } else { 0.0 };

    let orig = n_nodes * FULL_DIM * 3;
    let comp = n_nodes * ZeckBF17Edge::ENCODED_SIZE;

    FidelityReport {
        s_fidelity: sf, p_fidelity: pf, o_fidelity: of_,
        avg_fidelity: (sf + pf + of_) / 3.0,
        rank_correlation: rho, scent_agreement: scent_agr,
        compression_ratio: orig as f64 / comp as f64,
        original_bytes: orig, compressed_bytes: comp,
    }
}

impl std::fmt::Display for FidelityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\n{}", "═".repeat(60))?;
        writeln!(f, "  ZeckBF17 FIDELITY (i16 base, L1 distance)")?;
        writeln!(f, "{}", "═".repeat(60))?;
        writeln!(f, "  Sign-bit fidelity:  S={:.4}  P={:.4}  O={:.4}  avg={:.4}",
            self.s_fidelity, self.p_fidelity, self.o_fidelity, self.avg_fidelity)?;
        writeln!(f, "  Rank correlation ρ: {:.4}", self.rank_correlation)?;
        writeln!(f, "  Scent agreement:    {:.1}%", self.scent_agreement * 100.0)?;
        writeln!(f, "  Compression:        {:.0}:1 ({} → {} bytes)",
            self.compression_ratio, self.original_bytes, self.compressed_bytes)?;
        writeln!(f, "{}", "═".repeat(60))?;
        if self.rank_correlation > 0.937 {
            writeln!(f, "  ✓ BEATS SCENT: ρ > 0.937")?;
        } else if self.rank_correlation > 0.90 {
            writeln!(f, "  ✓ EXCELLENT: ρ > 0.90")?;
        } else if self.rank_correlation > 0.50 {
            writeln!(f, "  ? PARTIAL: ρ > 0.50")?;
        } else {
            writeln!(f, "  ✗ POOR: ρ < 0.50")?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_step_coverage() {
        assert!(verify_golden_coverage(), "Must visit all 17 residues");
    }

    #[test]
    fn test_golden_step_values() {
        assert_eq!(GOLDEN_POS_17[0], 0);
        assert_eq!(GOLDEN_POS_17[1], 11);
        assert_eq!(GOLDEN_POS_17[2], 5);
        assert_eq!(GOLDEN_POS_17[3], 16);
        let mut sorted: Vec<u8> = GOLDEN_POS_17.to_vec();
        sorted.sort();
        assert_eq!(sorted, (0..17).collect::<Vec<u8>>());
    }

    #[test]
    fn test_i16_captures_subunit() {
        // mean = 0.2 → i16 = 51 (0.2 * 256). BF16 would store 0.0.
        let mut acc = vec![0i8; FULL_DIM];
        for i in 0..FULL_DIM {
            acc[i] = if i % 5 < 3 { 1 } else { -1 }; // mean ≈ 0.2
        }
        let plane = ZeckBF17Plane::encode(&acc);
        // At least some base dims should capture the positive bias
        let positive = plane.base.dims.iter().filter(|&&d| d > 0).count();
        assert!(positive > 0, "i16 should capture subunit positive bias");
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut acc = vec![0i8; FULL_DIM];
        for i in 0..FULL_DIM {
            acc[i] = if i % 3 == 0 { 50 } else if i % 3 == 1 { -30 } else { 10 };
        }
        let fid = sign_bit_fidelity(&acc, &ZeckBF17Plane::encode(&acc).decode());
        println!("Roundtrip fidelity: {:.4}", fid);
        assert!(fid > 0.5);
    }

    #[test]
    fn test_byte_serialization() {
        let mut acc = vec![42i8; FULL_DIM];
        acc[0] = -100;
        let enc = ZeckBF17Plane::encode(&acc);
        let rec = ZeckBF17Plane::from_bytes(&enc.to_bytes());
        assert_eq!(enc.base, rec.base);
        assert_eq!(enc.envelope, rec.envelope);
    }

    #[test]
    fn test_self_scent_all_close() {
        let edge = ZeckBF17Edge::encode(&vec![50i8; FULL_DIM], &vec![-30i8; FULL_DIM], &vec![10i8; FULL_DIM]);
        assert_eq!(scent_from_base(&edge, &edge) & 0x7F, 0x7F);
    }

    #[test]
    fn test_zeckf64_from_base_self() {
        let edge = ZeckBF17Edge::encode(&vec![50i8; FULL_DIM], &vec![-30i8; FULL_DIM], &vec![10i8; FULL_DIM]);
        let z = zeckf64_from_base(&edge, &edge);
        assert_eq!(z as u8 & 0x7F, 0x7F);
        for b in 1..=7u8 { assert_eq!((z >> (b * 8)) as u8, 0); }
    }

    #[test]
    fn test_base_l1_self_zero() {
        let a = BasePattern { dims: [100, -200, 50, 0, 127, -128, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0] };
        assert_eq!(base_l1(&a, &a), 0);
    }

    #[test]
    fn test_sizes() {
        assert_eq!(ZeckBF17Edge::ENCODED_SIZE, 116);
        assert_eq!(ZeckBF17Plane::ENCODED_SIZE, 48);
    }

    #[test]
    fn test_compression_ratio() {
        let r = fidelity_experiment(10, 20);
        println!("{}", r);
        assert!(r.compression_ratio > 100.0);
    }

    #[test]
    fn test_fidelity_vs_encounters() {
        println!("\n{:>12} {:>10} {:>10} {:>10}", "encounters", "fidelity", "ρ(rank)", "scent%");
        for enc in [5, 10, 20, 50, 100] {
            let r = fidelity_experiment(20, enc);
            println!("{:>12} {:>10.4} {:>10.4} {:>10.1}%",
                enc, r.avg_fidelity, r.rank_correlation, r.scent_agreement * 100.0);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 1: Signal ratio × encounters
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_sweep_signal_ratio() {
        println!("\n{:═<80}", "═ SWEEP: signal_pct × encounters ");
        println!("{:>8} {:>6} {:>10} {:>10} {:>10}",
            "sig%", "enc", "fidelity", "ρ(rank)", "scent%");
        println!("{}", "─".repeat(50));

        for sig in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] {
            for enc in [5, 10, 20, 50] {
                let r = fidelity_experiment_cfg(&ExperimentConfig {
                    n_nodes: 20,
                    n_encounters: enc,
                    signal_pct: sig,
                    ..Default::default()
                });
                let marker = if r.rank_correlation > 0.937 { " ★" }
                    else if r.rank_correlation > 0.90 { " ●" }
                    else { "" };
                println!("{:>7.0}% {:>6} {:>10.4} {:>10.4} {:>9.1}%{}",
                    sig * 100.0, enc, r.avg_fidelity, r.rank_correlation,
                    r.scent_agreement * 100.0, marker);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 2: FP_SCALE
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_sweep_fp_scale() {
        println!("\n{:═<80}", "═ SWEEP: fp_scale ");
        println!("{:>10} {:>10} {:>10} {:>10}",
            "fp_scale", "fidelity", "ρ(rank)", "scent%");
        println!("{}", "─".repeat(45));

        for scale in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 4096.0, 16384.0] {
            let r = fidelity_experiment_cfg(&ExperimentConfig {
                n_nodes: 20,
                n_encounters: 20,
                fp_scale: scale,
                ..Default::default()
            });
            let marker = if r.rank_correlation > 0.937 { " ★" }
                else if r.rank_correlation > 0.90 { " ●" }
                else { "" };
            println!("{:>10.0} {:>10.4} {:>10.4} {:>9.1}%{}",
                scale, r.avg_fidelity, r.rank_correlation,
                r.scent_agreement * 100.0, marker);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 3: Independent octaves
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_sweep_indep_octaves() {
        println!("\n{:═<80}", "═ SWEEP: n_indep_octaves ");
        println!("{:>10} {:>10} {:>10} {:>10}",
            "octaves", "fidelity", "ρ(rank)", "scent%");
        println!("{}", "─".repeat(45));

        for oct in [1, 2, 3, 4, 5, 7, 10, 14, 17, 20, 28, 48, 96] {
            let r = fidelity_experiment_cfg(&ExperimentConfig {
                n_nodes: 20,
                n_encounters: 20,
                n_indep_octaves: oct,
                ..Default::default()
            });
            let bytes = BASE_DIM * 2 + oct.min(INDEPENDENT_OCTAVES);
            let marker = if r.rank_correlation > 0.937 { " ★" }
                else if r.rank_correlation > 0.90 { " ●" }
                else { "" };
            println!("{:>10} {:>10.4} {:>10.4} {:>9.1}% ({} bytes){}",
                oct, r.avg_fidelity, r.rank_correlation,
                r.scent_agreement * 100.0, bytes, marker);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 4: Scent threshold fraction
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_sweep_scent_threshold() {
        println!("\n{:═<80}", "═ SWEEP: scent_threshold_frac ");
        println!("{:>10} {:>10} {:>10}",
            "thresh", "scent%", "ρ(rank)");
        println!("{}", "─".repeat(35));

        for frac in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90] {
            let r = fidelity_experiment_cfg(&ExperimentConfig {
                n_nodes: 30,
                n_encounters: 20,
                scent_threshold_frac: frac,
                ..Default::default()
            });
            let marker = if r.scent_agreement > 0.90 { " ★★" }
                else if r.scent_agreement > 0.70 { " ★" }
                else if r.scent_agreement > 0.50 { " ●" }
                else { "" };
            println!("{:>9.3} {:>9.1}% {:>10.4}{}",
                frac, r.scent_agreement * 100.0, r.rank_correlation, marker);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 5: Node count effect on rank correlation stability
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_sweep_node_count() {
        println!("\n{:═<80}", "═ SWEEP: n_nodes (stability) ");
        println!("{:>8} {:>10} {:>10} {:>10}",
            "nodes", "fidelity", "ρ(rank)", "scent%");
        println!("{}", "─".repeat(43));

        for nodes in [5, 10, 15, 20, 30, 40, 50] {
            let r = fidelity_experiment_cfg(&ExperimentConfig {
                n_nodes: nodes,
                n_encounters: 20,
                ..Default::default()
            });
            println!("{:>8} {:>10.4} {:>10.4} {:>9.1}%",
                nodes, r.avg_fidelity, r.rank_correlation,
                r.scent_agreement * 100.0);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 6: Fractal self-similarity — does ρ stabilize across scales?
    // Tests whether the compression invariant has fractal structure:
    // the SAME ρ at 100 nodes as at 10 nodes means scale-free.
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_sweep_fractal_invariant() {
        println!("\n{:═<80}", "═ FRACTAL INVARIANT: ρ across scale × signal × encounters ");
        println!("{:>6} {:>6} {:>6} {:>10} {:>10} {:>10}",
            "nodes", "enc", "sig%", "fidelity", "ρ(rank)", "Δρ_prev");
        println!("{}", "─".repeat(55));

        for sig in [0.60, 0.70, 0.80] {
            let mut prev_rho = 0.0f64;
            for (nodes, enc) in [(5, 5), (10, 10), (20, 20), (30, 30), (40, 40), (50, 50)] {
                let r = fidelity_experiment_cfg(&ExperimentConfig {
                    n_nodes: nodes,
                    n_encounters: enc,
                    signal_pct: sig,
                    ..Default::default()
                });
                let delta = if prev_rho > 0.0 { r.rank_correlation - prev_rho } else { 0.0 };
                let marker = if delta.abs() < 0.02 && prev_rho > 0.0 { " ◆" } else { "" };
                println!("{:>6} {:>6} {:>5.0}% {:>10.4} {:>10.4} {:>+10.4}{}",
                    nodes, enc, sig * 100.0, r.avg_fidelity, r.rank_correlation, delta, marker);
                prev_rho = r.rank_correlation;
            }
            println!("{}", "─".repeat(55));
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 7a: Mantissa-matching fractal — FP_SCALE × scent_threshold
    // The hypothesis: when scale matches data magnitude, the L1
    // distribution's shape (its "mantissa") naturally aligns with the
    // threshold, giving near-perfect scent agreement. The dead angles
    // are scale/threshold combos where the L1 histogram straddles the
    // threshold boundary (bimodal split).
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_sweep_mantissa_fractal() {
        println!("\n{:═<80}", "═ MANTISSA FRACTAL: scale × threshold → scent shape ");
        println!("{:>8} {:>8} {:>10} {:>10} {:>10}",
            "scale", "thresh", "scent%", "ρ(rank)", "fidelity");
        println!("{}", "─".repeat(50));

        let mut fractal_map: Vec<(f64, f64, f64)> = Vec::new(); // (scale, thresh, scent)

        for scale in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 4096.0] {
            let mut best_scent = 0.0f64;
            let mut best_thresh = 0.0f64;
            for thresh in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50] {
                let r = fidelity_experiment_cfg(&ExperimentConfig {
                    n_nodes: 25,
                    n_encounters: 30,
                    signal_pct: 0.70,
                    fp_scale: scale,
                    scent_threshold_frac: thresh,
                    ..Default::default()
                });
                fractal_map.push((scale, thresh, r.scent_agreement));
                if r.scent_agreement > best_scent {
                    best_scent = r.scent_agreement;
                    best_thresh = thresh;
                }
            }
            let marker = if best_scent > 0.90 { " ★★" }
                else if best_scent > 0.50 { " ★" }
                else { "" };
            println!("{:>8.0} {:>8.4} {:>9.1}% {:>10}  (best thresh){}", scale, best_thresh,
                best_scent * 100.0, "", marker);
        }

        // Analyze: does optimal_threshold ~ C / scale? (fractal self-similarity)
        println!("\n  SCALE-THRESHOLD RELATIONSHIP:");
        println!("  {:>8} {:>10} {:>12}", "scale", "opt_thresh", "scale*thresh");
        println!("  {}", "─".repeat(34));

        let mut products = Vec::new();
        let mut prev_scale = 0.0f64;
        for scale in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 4096.0] {
            let best = fractal_map.iter()
                .filter(|(s, _, _)| (*s - scale).abs() < 0.1)
                .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
            if let Some(&(s, t, scent)) = best {
                if scent > 0.3 {
                    let product = s * t;
                    products.push(product);
                    println!("  {:>8.0} {:>10.4} {:>12.4}", s, t, product);
                }
            }
        }

        if products.len() >= 3 {
            let mean = products.iter().sum::<f64>() / products.len() as f64;
            let cv = (products.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
                / products.len() as f64).sqrt() / mean.abs().max(1e-10);
            println!("\n  scale × opt_thresh:  mean={:.4}  CV={:.4}", mean, cv);
            if cv < 0.20 {
                println!("  ★ FRACTAL INVARIANT: scale × threshold ≈ {:.2} (constant mantissa)", mean);
            } else if cv < 0.50 {
                println!("  ● QUASI-FRACTAL: scale × threshold moderately stable");
            } else {
                println!("  ○ NON-FRACTAL: no simple scale-threshold relationship");
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 7b: Dead angle analysis — where does scent collapse?
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_sweep_dead_angles() {
        println!("\n{:═<80}", "═ DEAD ANGLE ANALYSIS: scent collapse boundaries ");
        println!("{:>8} {:>10} {:>10} {:>10} {:>10}",
            "scale", "thresh_lo", "thresh_hi", "scent_lo", "scent_hi");
        println!("{}", "─".repeat(52));

        for scale in [4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0] {
            let thresholds = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90];
            let mut scents: Vec<(f64, f64)> = Vec::new();
            for &thresh in &thresholds {
                let r = fidelity_experiment_cfg(&ExperimentConfig {
                    n_nodes: 25,
                    n_encounters: 30,
                    signal_pct: 0.70,
                    fp_scale: scale,
                    scent_threshold_frac: thresh,
                    ..Default::default()
                });
                scents.push((thresh, r.scent_agreement));
            }

            // Find the steepest drop (dead angle boundary)
            let mut max_drop = 0.0f64;
            let mut drop_lo = 0.0f64;
            let mut drop_hi = 0.0f64;
            let mut scent_lo = 0.0f64;
            let mut scent_hi = 0.0f64;
            for i in 1..scents.len() {
                let drop = (scents[i-1].1 - scents[i].1).abs();
                if drop > max_drop {
                    max_drop = drop;
                    drop_lo = scents[i-1].0;
                    drop_hi = scents[i].0;
                    scent_lo = scents[i-1].1;
                    scent_hi = scents[i].1;
                }
            }
            let marker = if max_drop > 0.5 { " ⚡" }
                else if max_drop > 0.2 { " △" }
                else { "" };
            println!("{:>8.0} {:>10.4} {:>10.4} {:>9.1}% {:>9.1}%{}",
                scale, drop_lo, drop_hi, scent_lo * 100.0, scent_hi * 100.0, marker);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 7c: Combined sweet spot — grid search for optimal config
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_sweep_sweet_spot() {
        println!("\n{:═<80}", "═ SWEET SPOT SEARCH ");
        println!("{:>6} {:>6} {:>8} {:>8} {:>10} {:>10} {:>10}",
            "sig%", "enc", "scale", "thresh", "fidelity", "ρ(rank)", "scent%");
        println!("{}", "─".repeat(68));

        let mut best_rho = 0.0f64;
        let mut best_scent = 0.0f64;
        let mut best_rho_cfg = String::new();
        let mut best_scent_cfg = String::new();
        let mut best_combined = 0.0f64;
        let mut best_combined_cfg = String::new();

        for sig in [0.60, 0.70, 0.80] {
            for enc in [10, 20, 50] {
                for scale in [64.0, 256.0, 1024.0] {
                    for thresh in [0.01, 0.05, 0.10, 0.20, 0.50] {
                        let r = fidelity_experiment_cfg(&ExperimentConfig {
                            n_nodes: 20,
                            n_encounters: enc,
                            signal_pct: sig,
                            fp_scale: scale,
                            scent_threshold_frac: thresh,
                            ..Default::default()
                        });

                        let combined = r.rank_correlation * 0.6 + r.scent_agreement * 0.4;

                        if r.rank_correlation > best_rho {
                            best_rho = r.rank_correlation;
                            best_rho_cfg = format!("sig={:.0}% enc={} scale={} thresh={}", sig*100.0, enc, scale, thresh);
                        }
                        if r.scent_agreement > best_scent {
                            best_scent = r.scent_agreement;
                            best_scent_cfg = format!("sig={:.0}% enc={} scale={} thresh={}", sig*100.0, enc, scale, thresh);
                        }
                        if combined > best_combined {
                            best_combined = combined;
                            best_combined_cfg = format!("sig={:.0}% enc={} scale={} thresh={}", sig*100.0, enc, scale, thresh);
                        }

                        // Only print notable results
                        if r.rank_correlation > 0.93 && r.scent_agreement > 0.30 {
                            println!("{:>5.0}% {:>6} {:>8.0} {:>8.3} {:>10.4} {:>10.4} {:>9.1}% ★",
                                sig*100.0, enc, scale, thresh, r.avg_fidelity,
                                r.rank_correlation, r.scent_agreement * 100.0);
                        }
                    }
                }
            }
        }

        println!("\n{}", "═".repeat(68));
        println!("  BEST ρ(rank):  {:.4}  @ {}", best_rho, best_rho_cfg);
        println!("  BEST scent%:   {:.1}%  @ {}", best_scent * 100.0, best_scent_cfg);
        println!("  BEST combined: {:.4}  @ {}", best_combined, best_combined_cfg);
        println!("{}", "═".repeat(68));
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 8: Dynamic compression — does ρ track encounters non-linearly?
    // Looking for power-law / log-law convergence (fractal signature).
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_sweep_convergence_curve() {
        println!("\n{:═<80}", "═ CONVERGENCE CURVE: ρ vs encounters (looking for fractal scaling) ");
        println!("{:>8} {:>10} {:>10} {:>10} {:>12}",
            "enc", "fidelity", "ρ(rank)", "ln(enc)", "ρ/ln(enc)");
        println!("{}", "─".repeat(55));

        let mut rho_values: Vec<(f64, f64)> = Vec::new();

        for enc in [3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200] {
            let r = fidelity_experiment_cfg(&ExperimentConfig {
                n_nodes: 30,
                n_encounters: enc,
                signal_pct: 0.70,
                ..Default::default()
            });
            let ln_enc = (enc as f64).ln();
            let ratio = if ln_enc > 0.0 { r.rank_correlation / ln_enc } else { 0.0 };
            rho_values.push((ln_enc, r.rank_correlation));
            println!("{:>8} {:>10.4} {:>10.4} {:>10.3} {:>12.4}",
                enc, r.avg_fidelity, r.rank_correlation, ln_enc, ratio);
        }

        // Test multiple fractal scaling models:
        // Model A: ρ ~ ln(enc)           → ρ/ln(enc) = const
        // Model B: ρ ~ 1 - C/enc^α       → (1-ρ)*enc^α = const (saturation)
        // Model C: ρ ~ 1 - C*exp(-λ*enc) → ln(1-ρ) ~ -λ*enc (exponential)
        // Model D: self-similar Δρ        → Δρ/Δ(ln(enc)) = const

        let valid: Vec<(f64, f64)> = rho_values.iter()
            .filter(|(_, rho)| *rho > 0.5 && *rho < 1.0)
            .copied().collect();

        if valid.len() >= 4 {
            // Model A: ρ/ln(enc)
            let ratios_a: Vec<f64> = valid.iter().map(|(ln_e, rho)| rho / ln_e).collect();
            let mean_a = ratios_a.iter().sum::<f64>() / ratios_a.len() as f64;
            let cv_a = (ratios_a.iter().map(|r| (r - mean_a).powi(2)).sum::<f64>()
                / ratios_a.len() as f64).sqrt() / mean_a.abs().max(1e-10);

            // Model B: (1-ρ)*enc^0.5 = const (power-law saturation)
            let ratios_b: Vec<f64> = valid.iter()
                .map(|(ln_e, rho)| (1.0 - rho) * ln_e.exp().sqrt())
                .collect();
            let mean_b = ratios_b.iter().sum::<f64>() / ratios_b.len() as f64;
            let cv_b = (ratios_b.iter().map(|r| (r - mean_b).powi(2)).sum::<f64>()
                / ratios_b.len() as f64).sqrt() / mean_b.abs().max(1e-10);

            // Model B2: (1-ρ)*enc = const (inverse-linear saturation)
            let ratios_b2: Vec<f64> = valid.iter()
                .map(|(ln_e, rho)| (1.0 - rho) * ln_e.exp())
                .collect();
            let mean_b2 = ratios_b2.iter().sum::<f64>() / ratios_b2.len() as f64;
            let cv_b2 = (ratios_b2.iter().map(|r| (r - mean_b2).powi(2)).sum::<f64>()
                / ratios_b2.len() as f64).sqrt() / mean_b2.abs().max(1e-10);

            // Model C: ln(1-ρ)/enc = -λ (exponential decay)
            let ratios_c: Vec<f64> = valid.iter()
                .map(|(ln_e, rho)| (1.0 - rho).ln() / ln_e.exp())
                .collect();
            let mean_c = ratios_c.iter().sum::<f64>() / ratios_c.len() as f64;
            let cv_c = (ratios_c.iter().map(|r| (r - mean_c).powi(2)).sum::<f64>()
                / ratios_c.len() as f64).sqrt() / mean_c.abs().max(1e-10);

            // Model D: self-similar — Δρ per log-step
            let mut deltas = Vec::new();
            for i in 1..valid.len() {
                let d_rho = valid[i].1 - valid[i-1].1;
                let d_ln = valid[i].0 - valid[i-1].0;
                if d_ln.abs() > 0.01 {
                    deltas.push(d_rho / d_ln);
                }
            }
            let mean_d = if !deltas.is_empty() { deltas.iter().sum::<f64>() / deltas.len() as f64 } else { 0.0 };
            let cv_d = if !deltas.is_empty() && mean_d.abs() > 1e-10 {
                (deltas.iter().map(|r| (r - mean_d).powi(2)).sum::<f64>()
                    / deltas.len() as f64).sqrt() / mean_d.abs()
            } else { 999.0 };

            println!("\n  FRACTAL SCALING MODEL COMPARISON (lower CV = better fit):");
            println!("  ─────────────────────────────────────────────────────");
            println!("  Model A  ρ/ln(enc) = const         CV = {:.4}", cv_a);
            println!("  Model B  (1-ρ)·√enc = const        CV = {:.4}", cv_b);
            println!("  Model B2 (1-ρ)·enc = const          CV = {:.4}", cv_b2);
            println!("  Model C  ln(1-ρ)/enc = -λ           CV = {:.4}", cv_c);
            println!("  Model D  Δρ/Δln(enc) = const        CV = {:.4}", cv_d);

            let models = [("A: ρ~ln(enc)", cv_a), ("B: ρ~1-C/√enc", cv_b),
                ("B2: ρ~1-C/enc", cv_b2), ("C: ρ~1-Ce^{-λn}", cv_c),
                ("D: self-similar Δρ", cv_d)];
            let best = models.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
            println!("\n  ◆ BEST FIT: {} (CV={:.4})", best.0, best.1);

            if best.1 < 0.15 {
                println!("  ★ STRONG fractal/scaling invariant detected");
            } else if best.1 < 0.30 {
                println!("  ● MODERATE scaling regularity");
            } else {
                println!("  ○ WEAK scaling — ρ likely saturates non-uniformly");
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 9: Step comparison — Golden(11) vs Quinten vs all coprime
    // Golden: round(17/φ) = 11  (golden ratio)
    // Quinten: round(17×7/12) = 10 (circle of fifths: 7 semitones / 12)
    // Fibonacci: fib sequence mod 17 (but only visits 13 residues!)
    // All steps 1-16 are coprime to 17 since 17 is prime.
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_sweep_step_comparison() {
        println!("\n{:═<80}", "═ STEP COMPARISON: all coprime steps 1-16 ");
        println!("{:>6} {:>12} {:>10} {:>10} {:>10} {:>8}",
            "step", "name", "fidelity", "ρ(rank)", "scent%", "spread");
        println!("{}", "─".repeat(62));

        let step_names = |s: usize| -> &'static str {
            match s {
                1 => "sequential",
                2 => "minor 2nd",
                3 => "minor 3rd",
                5 => "perfect 4th",
                6 => "tritone-",
                7 => "perfect 5th",
                8 => "minor 6th",
                10 => "quinten",
                11 => "GOLDEN φ",
                12 => "major 7th",
                _ => "",
            }
        };

        let mut results: Vec<(usize, f64, f64, f64)> = Vec::new();

        for step in 1..=16 {
            // Verify coverage
            let table = make_position_table(step);
            let mut seen = [false; BASE_DIM];
            for &p in &table { seen[p as usize] = true; }
            let coverage = seen.iter().filter(|&&s| s).count();
            assert_eq!(coverage, BASE_DIM, "step {} must cover all 17 (prime!)", step);

            // Measure spread: how evenly distributed are consecutive positions?
            let mut gaps: Vec<usize> = Vec::new();
            for i in 0..BASE_DIM {
                let next_pos = table[(i + 1) % BASE_DIM] as usize;
                let curr_pos = table[i] as usize;
                let gap = if next_pos >= curr_pos { next_pos - curr_pos } else { BASE_DIM + next_pos - curr_pos };
                gaps.push(gap);
            }
            let mean_gap = gaps.iter().sum::<usize>() as f64 / gaps.len() as f64;
            let spread_var = gaps.iter().map(|&g| (g as f64 - mean_gap).powi(2)).sum::<f64>() / gaps.len() as f64;
            let spread_cv = spread_var.sqrt() / mean_gap;

            let r = fidelity_experiment_cfg(&ExperimentConfig {
                n_nodes: 25,
                n_encounters: 20,
                step,
                ..Default::default()
            });

            results.push((step, r.avg_fidelity, r.rank_correlation, r.scent_agreement));

            let marker = if step == GOLDEN_STEP { " ★φ" }
                else if step == 10 { " ★Q" }
                else if step == 7 { " ★5" }
                else if r.rank_correlation > 0.99 { " ●" }
                else { "" };

            println!("{:>6} {:>12} {:>10.4} {:>10.4} {:>9.1}% {:>7.3}{}",
                step, step_names(step), r.avg_fidelity, r.rank_correlation,
                r.scent_agreement * 100.0, spread_cv, marker);
        }

        // Find best step
        let best = results.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();
        println!("\n  BEST STEP: {} (ρ={:.4})", best.0, best.2);

        // Check if golden step is truly optimal or if spread matters
        let golden_rho = results.iter().find(|r| r.0 == GOLDEN_STEP).unwrap().2;
        let max_rho = best.2;
        if (golden_rho - max_rho).abs() < 0.01 {
            println!("  ◆ Golden step (11) is within 0.01 of best — confirms φ optimality");
        } else {
            println!("  ○ Step {} beats golden by {:.4}", best.0, max_rho - golden_rho);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 10: Dorian mode — non-uniform stepping
    // Instead of a fixed step, use mode-based intervals:
    // Dorian:   W H W W W H W = [2,1,2,2,2,1,2] (sum=12)
    // Phrygian: H W W W H W W = [1,2,2,2,1,2,2]
    // We adapt these to base-17 by scaling intervals proportionally.
    // Also test: IV-VI pattern (Dorian characteristic tones)
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_sweep_dorian_modes() {
        println!("\n{:═<80}", "═ DORIAN MODE & NON-UNIFORM STEPPING ");

        // Build mode-based position tables
        // A "mode" gives a sequence of intervals that sums to BASE_DIM
        // and visits all positions exactly once.
        let modes: Vec<(&str, Vec<usize>)> = vec![
            ("uniform-11", vec![11; 17]),  // golden step baseline
            ("uniform-10", vec![10; 17]),  // quinten baseline
            // Dorian scaled to 17: [2,1,2,2,2,1,2] repeated ~2.4x
            ("dorian", {
                let pattern = [2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2];
                pattern.to_vec()
            }),
            // Phrygian: starts on half-step
            ("phrygian", {
                let pattern = [1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2];
                pattern.to_vec()
            }),
            // Fibonacci intervals: [1,1,2,3,5,8,13,4,0,4,4,8,12,3,15,1,16]
            // (fib(i) mod 17 as interval)
            ("fibonacci", {
                let mut fibs = vec![0usize; BASE_DIM];
                let (mut a, mut b) = (1usize, 1usize);
                for i in 0..BASE_DIM {
                    fibs[i] = a % BASE_DIM;
                    let next = (a + b) % BASE_DIM;
                    a = b; b = next;
                }
                // Ensure all intervals > 0
                fibs.iter().map(|&f| if f == 0 { 1 } else { f }).collect()
            }),
            // IV-VI emphasis: larger steps at positions 4,6 (Dorian characteristic)
            ("dorian-IV-VI", {
                let mut intervals = vec![2; BASE_DIM];
                intervals[3] = 5;  // IV: leap
                intervals[5] = 5;  // VI: leap
                intervals[9] = 1;  // compensate
                intervals[13] = 1; // compensate
                intervals
            }),
        ];

        println!("{:>14} {:>10} {:>10} {:>10} {:>8}",
            "mode", "fidelity", "ρ(rank)", "scent%", "covers?");
        println!("{}", "─".repeat(56));

        for (name, intervals) in &modes {
            // Build position table from intervals
            let mut table = [0u8; BASE_DIM];
            let mut pos_val = 0usize;
            for i in 0..BASE_DIM {
                table[i] = (pos_val % BASE_DIM) as u8;
                pos_val = (pos_val + intervals[i % intervals.len()]) % BASE_DIM;
            }

            // Check coverage
            let mut seen = [false; BASE_DIM];
            for &p in &table { seen[p as usize] = true; }
            let coverage = seen.iter().filter(|&&s| s).count();

            if coverage < BASE_DIM {
                println!("{:>14} {:>10} {:>10} {:>10} {:>8}",
                    name, "—", "—", "—", format!("{}/17", coverage));
                continue;
            }

            // For uniform modes, use the step directly
            let step = if intervals.windows(2).all(|w| w[0] == w[1]) {
                intervals[0]
            } else {
                GOLDEN_STEP // non-uniform modes use golden for L1 comparison
            };

            let r = fidelity_experiment_cfg(&ExperimentConfig {
                n_nodes: 25,
                n_encounters: 20,
                step,
                ..Default::default()
            });

            let marker = if name.contains("11") { " (baseline)" } else { "" };
            println!("{:>14} {:>10.4} {:>10.4} {:>9.1}% {:>7}/17{}",
                name, r.avg_fidelity, r.rank_correlation,
                r.scent_agreement * 100.0, coverage, marker);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 11: Gamma curve encoding
    // Like Photoshop gamma: compresses dynamic range non-linearly.
    // γ < 1: expands small values (more precision near zero)
    // γ > 1: expands large values (more precision at extremes)
    // γ = 0.4545 ≈ 1/2.2: sRGB gamma (perceptual uniformity)
    // γ = 1/√2 ≈ 0.707: sqrt(2) gamma
    // γ = 1/e ≈ 0.368: Euler gamma
    // We also test Fibonacci-ratio gammas: φ, 1/φ, φ², 1/φ²
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_sweep_gamma_curves() {
        println!("\n{:═<80}", "═ GAMMA CURVE ENCODING ");
        println!("{:>8} {:>14} {:>10} {:>10} {:>10}",
            "γ", "name", "fidelity", "ρ(rank)", "scent%");
        println!("{}", "─".repeat(56));

        let phi: f64 = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let euler_gamma = 0.5772156649; // Euler-Mascheroni constant

        let gammas: Vec<(f64, &str)> = vec![
            (0.25, "¼"),
            (1.0 / std::f64::consts::E, "1/e"),
            (euler_gamma, "γ_Euler"),
            (1.0 / phi.powi(2), "1/φ²"),
            (0.4545, "sRGB"),
            (1.0 / phi, "1/φ"),
            (1.0 / 2.0_f64.sqrt(), "1/√2"),
            (2.0 / 3.0, "⅔ quinten"),
            (1.0, "LINEAR"),
            (phi - 1.0, "φ-1"),
            (2.0_f64.sqrt(), "√2"),
            (phi, "φ"),
            (2.0, "square"),
            (phi.powi(2), "φ²"),
            (std::f64::consts::E, "e"),
            (3.0, "cube"),
        ];

        let mut best_rho = 0.0f64;
        let mut best_name = "";
        let mut best_gamma = 0.0f64;

        for &(gamma, name) in &gammas {
            let r = fidelity_experiment_cfg(&ExperimentConfig {
                n_nodes: 25,
                n_encounters: 20,
                gamma,
                ..Default::default()
            });

            if r.rank_correlation > best_rho {
                best_rho = r.rank_correlation;
                best_name = name;
                best_gamma = gamma;
            }

            let marker = if name == "LINEAR" { " (baseline)" }
                else if r.rank_correlation > 0.995 { " ★★" }
                else if r.rank_correlation > 0.99 { " ★" }
                else { "" };

            println!("{:>8.4} {:>14} {:>10.4} {:>10.4} {:>9.1}%{}",
                gamma, name, r.avg_fidelity, r.rank_correlation,
                r.scent_agreement * 100.0, marker);
        }

        println!("\n  BEST GAMMA: γ={:.4} ({}) → ρ={:.4}", best_gamma, best_name, best_rho);
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 12: Fibonacci carrier × gamma — combined
    // Use Fibonacci-based step (8 = fib(6) mod 17) with gamma curves
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_sweep_fib_carrier_gamma() {
        println!("\n{:═<80}", "═ FIBONACCI CARRIER × GAMMA ");
        println!("{:>6} {:>8} {:>10} {:>10} {:>10}",
            "step", "γ", "fidelity", "ρ(rank)", "scent%");
        println!("{}", "─".repeat(48));

        let phi: f64 = (1.0 + 5.0_f64.sqrt()) / 2.0;
        // Fibonacci numbers mod 17: 1,1,2,3,5,8,13,4,0,4,4,8,12,3,15,1,16,...
        // Non-zero coprime: 1,2,3,5,8,13,4 → steps: 8 (fib), 13 (fib), 5 (fib)
        let fib_steps = [5, 8, 13, GOLDEN_STEP]; // Fibonacci + golden
        let gammas = [1.0 / phi, 2.0 / 3.0, 1.0, phi, 2.0_f64.sqrt()];

        let mut best = (0usize, 0.0f64, 0.0f64, 0.0f64); // step, gamma, rho, scent

        for &step in &fib_steps {
            for &gamma in &gammas {
                let r = fidelity_experiment_cfg(&ExperimentConfig {
                    n_nodes: 25,
                    n_encounters: 20,
                    step,
                    gamma,
                    ..Default::default()
                });

                let combined = r.rank_correlation * 0.7 + r.scent_agreement * 0.3;
                if combined > best.2 * 0.7 + best.3 * 0.3 {
                    best = (step, gamma, r.rank_correlation, r.scent_agreement);
                }

                let marker = if step == GOLDEN_STEP && gamma == 1.0 { " (baseline)" }
                    else if r.rank_correlation > 0.995 { " ★★" }
                    else { "" };

                println!("{:>6} {:>8.4} {:>10.4} {:>10.4} {:>9.1}%{}",
                    step, gamma, r.avg_fidelity, r.rank_correlation,
                    r.scent_agreement * 100.0, marker);
            }
        }

        println!("\n  BEST: step={} γ={:.4} → ρ={:.4} scent={:.1}%",
            best.0, best.1, best.2, best.3 * 100.0);
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 13: Psychometric convergence
    // Does the encoding reach a FIXED POINT where encode→decode→encode
    // produces the same base pattern? If so, the encoding has reached
    // its natural attractor — the psychometric convergence point.
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_psychometric_convergence() {
        println!("\n{:═<80}", "═ PSYCHOMETRIC CONVERGENCE: encode→decode→encode fixed point ");
        println!("{:>6} {:>10} {:>10} {:>10} {:>10}",
            "iter", "Δ_base", "fidelity", "ρ(rank)", "converged?");
        println!("{}", "─".repeat(50));

        // Generate a test signal
        let mut rng = 42u64;
        let next = |r: &mut u64| -> u64 {
            *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *r
        };

        let mut signal = vec![0i8; FULL_DIM];
        for d in 0..FULL_DIM {
            let h = next(&mut rng);
            signal[d] = (((h >> 8) % 201) as i16 - 100) as i8; // range [-100, 100]
        }

        let mut current = signal.clone();
        let mut prev_base: Option<BasePattern> = None;
        let mut converge_iter = 0;

        for iter in 0..20 {
            let encoded = ZeckBF17Plane::encode(&current);
            let decoded = encoded.decode();

            // Measure change in base pattern
            let delta = if let Some(ref pb) = prev_base {
                base_l1(pb, &encoded.base)
            } else {
                u32::MAX
            };

            let fid = sign_bit_fidelity(&signal, &decoded);

            let converged = delta == 0 && iter > 0;
            if converged && converge_iter == 0 {
                converge_iter = iter;
            }

            println!("{:>6} {:>10} {:>10.4} {:>10} {:>10}",
                iter,
                if iter == 0 { "—".to_string() } else { format!("{}", delta) },
                fid,
                "",
                if converged { "★ FIXED POINT" } else { "" }
            );

            prev_base = Some(encoded.base);
            current = decoded;

            if converged { break; }
        }

        if converge_iter > 0 {
            println!("\n  ◆ PSYCHOMETRIC CONVERGENCE at iteration {}", converge_iter);
        } else {
            println!("\n  ○ No convergence in 20 iterations");
        }

        // Now test convergence across different gammas
        println!("\n  GAMMA × CONVERGENCE:");
        println!("  {:>8} {:>10} {:>10}", "γ", "converge@", "final_fid");
        println!("  {}", "─".repeat(32));

        let phi: f64 = (1.0 + 5.0_f64.sqrt()) / 2.0;
        for gamma in [0.5, 1.0 / phi, 2.0 / 3.0, 1.0, phi, 2.0, 3.0] {
            let mut curr = signal.clone();
            let mut conv_at = 0;
            let mut final_fid = 0.0;
            let mut pb: Option<BasePattern> = None;

            for iter in 0..20 {
                // Encode with gamma
                let mut sum_arr = [0i64; BASE_DIM];
                let mut cnt_arr = [0u32; BASE_DIM];
                for octave in 0..N_OCTAVES {
                    for bi in 0..BASE_DIM {
                        let dim = octave * BASE_DIM + golden_position(bi);
                        if dim < FULL_DIM {
                            sum_arr[bi] += curr[dim] as i64;
                            cnt_arr[bi] += 1;
                        }
                    }
                }
                let mut base = BasePattern { dims: [0i16; BASE_DIM] };
                for d in 0..BASE_DIM {
                    if cnt_arr[d] > 0 {
                        let mean = sum_arr[d] as f64 / cnt_arr[d] as f64;
                        let encoded = gamma_encode(mean, gamma);
                        base.dims[d] = (encoded * FP_SCALE).round().clamp(-32768.0, 32767.0) as i16;
                    }
                }

                let delta = if let Some(ref p) = pb { base_l1(p, &base) } else { u32::MAX };

                // Decode with inverse gamma
                let mut out = vec![0i8; FULL_DIM];
                for octave in 0..N_OCTAVES {
                    for bi in 0..BASE_DIM {
                        let dim = octave * BASE_DIM + golden_position(bi);
                        if dim < FULL_DIM {
                            let bv = base.dims[bi] as f64 / FP_SCALE;
                            let dec = gamma_encode(bv, 1.0 / gamma);
                            out[dim] = dec.clamp(-128.0, 127.0) as i8;
                        }
                    }
                }

                final_fid = sign_bit_fidelity(&signal, &out);
                pb = Some(base);
                curr = out;

                if delta == 0 && iter > 0 {
                    conv_at = iter;
                    break;
                }
            }

            let marker = if conv_at > 0 && conv_at <= 3 { " ★" }
                else if conv_at > 0 { " ●" }
                else { "" };

            println!("  {:>8.4} {:>10} {:>10.4}{}",
                gamma,
                if conv_at > 0 { format!("{}", conv_at) } else { "—".to_string() },
                final_fid,
                marker);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 14: Palette compression — 8-bit indexed archetypes
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_palette_compression() {
        println!("\n{:═<80}", "═ PALETTE COMPRESSION: archetype codebook ");

        let mut rng = 42u64;
        let next = |r: &mut u64| -> u64 {
            *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *r
        };

        // Build reverse mapping
        let mut dim_to_base = vec![0usize; FULL_DIM];
        for octave in 0..N_OCTAVES {
            for bi in 0..BASE_DIM {
                let dim = octave * BASE_DIM + golden_position(bi);
                if dim < FULL_DIM { dim_to_base[dim] = bi; }
            }
        }

        // Generate nodes with octave structure
        let n_nodes = 100;
        let n_encounters = 30;
        let mut nodes: Vec<(Vec<i8>, Vec<i8>, Vec<i8>)> = Vec::new();
        for node in 0..n_nodes {
            let mut s = vec![0i8; FULL_DIM];
            let mut p = vec![0i8; FULL_DIM];
            let mut o = vec![0i8; FULL_DIM];
            let node_seed = next(&mut rng) ^ (node as u64).wrapping_mul(0x517cc1b727220a95);

            let mut base_s = [0i8; BASE_DIM];
            let mut base_p = [0i8; BASE_DIM];
            let mut base_o = [0i8; BASE_DIM];
            for bi in 0..BASE_DIM {
                let h = node_seed.wrapping_mul(bi as u64 + 1).wrapping_add(0x9e3779b97f4a7c15);
                base_s[bi] = if (h >> 17) & 1 == 1 { 1i8 } else { -1 };
                base_p[bi] = if (h >> 23) & 1 == 1 { 1i8 } else { -1 };
                base_o[bi] = if (h >> 29) & 1 == 1 { 1i8 } else { -1 };
            }
            for d in 0..FULL_DIM {
                let bi = dim_to_base[d];
                let (mut vs, mut vp, mut vo) = (0i8, 0i8, 0i8);
                for _enc in 0..n_encounters {
                    let eh = next(&mut rng);
                    vs = vs.saturating_add(if (eh >> 7) % 10 < 7 { base_s[bi] } else { -base_s[bi] });
                    vp = vp.saturating_add(if (eh >> 13) % 10 < 7 { base_p[bi] } else { -base_p[bi] });
                    vo = vo.saturating_add(if (eh >> 19) % 10 < 7 { base_o[bi] } else { -base_o[bi] });
                }
                s[d] = vs; p[d] = vp; o[d] = vo;
            }
            nodes.push((s, p, o));
        }

        // Encode to ZeckBF17 first
        let edges: Vec<ZeckBF17Edge> = nodes.iter()
            .map(|(s, p, o)| ZeckBF17Edge::encode(s, p, o)).collect();

        // Collect all base patterns for clustering
        let mut all_patterns: Vec<BasePattern> = Vec::new();
        for e in &edges {
            all_patterns.push(e.subject.clone());
            all_patterns.push(e.predicate.clone());
            all_patterns.push(e.object.clone());
        }

        // Compute ground truth distances (from full planes)
        let n_pairs = n_nodes.min(50);
        let mut exact_d: Vec<f64> = Vec::new();
        for i in 0..n_pairs {
            for j in (i + 1)..n_pairs {
                exact_d.push(
                    sign_bit_hamming(&nodes[i].0, &nodes[j].0) as f64
                    + sign_bit_hamming(&nodes[i].1, &nodes[j].1) as f64
                    + sign_bit_hamming(&nodes[i].2, &nodes[j].2) as f64);
            }
        }

        // ZeckBF17 baseline (no palette)
        let mut zeck_d: Vec<f64> = Vec::new();
        for i in 0..n_pairs {
            for j in (i + 1)..n_pairs {
                zeck_d.push(
                    base_l1(&edges[i].subject, &edges[j].subject) as f64
                    + base_l1(&edges[i].predicate, &edges[j].predicate) as f64
                    + base_l1(&edges[i].object, &edges[j].object) as f64);
            }
        }
        let rho_baseline = spearman(&exact_d, &zeck_d);

        println!("\n  Baseline (ZeckBF17, no palette):");
        println!("    ρ={:.4}  bytes/edge={}  total={}",
            rho_baseline, ZeckBF17Edge::ENCODED_SIZE, n_nodes * ZeckBF17Edge::ENCODED_SIZE);

        // Sweep palette sizes
        println!("\n{:>8} {:>10} {:>10} {:>12} {:>12} {:>10}",
            "palette", "ρ(rank)", "Δρ", "bytes/edge", "total_bytes", "ratio");
        println!("{}", "─".repeat(68));

        for palette_size in [2, 4, 8, 16, 32, 64, 128, 256] {
            let palette = BasePalette::build(&all_patterns, palette_size);

            // Compress edges
            let pal_edges: Vec<PaletteEdge> = edges.iter()
                .map(|e| palette.compress(e)).collect();

            // Compute palette distances
            let mut pal_d: Vec<f64> = Vec::new();
            for i in 0..n_pairs {
                for j in (i + 1)..n_pairs {
                    pal_d.push(palette.edge_distance(&pal_edges[i], &pal_edges[j]) as f64);
                }
            }

            let rho = spearman(&exact_d, &pal_d);
            let delta_rho = rho - rho_baseline;
            let total = palette.total_bytes(n_nodes);
            let orig = n_nodes * FULL_DIM * 3;
            let ratio = orig as f64 / total as f64;
            let bytes_per_edge = total as f64 / n_nodes as f64;

            let marker = if rho > 0.937 { " ★" }
                else if rho > 0.90 { " ●" }
                else { "" };

            println!("{:>8} {:>10.4} {:>+10.4} {:>12.1} {:>12} {:>9.0}:1{}",
                palette_size, rho, delta_rho, bytes_per_edge,
                total, ratio, marker);
        }

        // Count unique palette indices actually used
        let palette_256 = BasePalette::build(&all_patterns, 256);
        let pal_edges_256: Vec<PaletteEdge> = edges.iter()
            .map(|e| palette_256.compress(e)).collect();
        let mut used_s = std::collections::HashSet::new();
        let mut used_p = std::collections::HashSet::new();
        let mut used_o = std::collections::HashSet::new();
        for pe in &pal_edges_256 {
            used_s.insert(pe.s_idx);
            used_p.insert(pe.p_idx);
            used_o.insert(pe.o_idx);
        }
        println!("\n  Palette utilization (k=256, {} edges):", n_nodes);
        println!("    S: {}/256 entries used", used_s.len());
        println!("    P: {}/256 entries used", used_p.len());
        println!("    O: {}/256 entries used", used_o.len());
        let total_unique: std::collections::HashSet<u8> = used_s.union(&used_p)
            .chain(used_o.iter()).copied().collect();
        println!("    Total unique: {}/256", total_unique.len());

        // Effective bits per plane
        let effective_bits = (total_unique.len() as f64).log2();
        println!("    Effective bits: {:.1} per plane → {:.1} bits/edge",
            effective_bits, effective_bits * 3.0);
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 15: Palette + scent threshold co-optimization
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_palette_scent() {
        println!("\n{:═<80}", "═ PALETTE × SCENT THRESHOLD ");

        let mut rng = 42u64;
        let next = |r: &mut u64| -> u64 {
            *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *r
        };

        let mut dim_to_base = vec![0usize; FULL_DIM];
        for octave in 0..N_OCTAVES {
            for bi in 0..BASE_DIM {
                let dim = octave * BASE_DIM + golden_position(bi);
                if dim < FULL_DIM { dim_to_base[dim] = bi; }
            }
        }

        let n_nodes = 50;
        let mut nodes: Vec<(Vec<i8>, Vec<i8>, Vec<i8>)> = Vec::new();
        for node in 0..n_nodes {
            let mut s = vec![0i8; FULL_DIM];
            let mut p = vec![0i8; FULL_DIM];
            let mut o = vec![0i8; FULL_DIM];
            let node_seed = next(&mut rng) ^ (node as u64).wrapping_mul(0x517cc1b727220a95);
            let mut base_s = [0i8; BASE_DIM];
            let mut base_p = [0i8; BASE_DIM];
            let mut base_o = [0i8; BASE_DIM];
            for bi in 0..BASE_DIM {
                let h = node_seed.wrapping_mul(bi as u64 + 1).wrapping_add(0x9e3779b97f4a7c15);
                base_s[bi] = if (h >> 17) & 1 == 1 { 1 } else { -1 };
                base_p[bi] = if (h >> 23) & 1 == 1 { 1 } else { -1 };
                base_o[bi] = if (h >> 29) & 1 == 1 { 1 } else { -1 };
            }
            for d in 0..FULL_DIM {
                let bi = dim_to_base[d];
                let (mut vs, mut vp, mut vo) = (0i8, 0i8, 0i8);
                for _enc in 0..30 {
                    let eh = next(&mut rng);
                    vs = vs.saturating_add(if (eh >> 7) % 10 < 7 { base_s[bi] } else { -base_s[bi] });
                    vp = vp.saturating_add(if (eh >> 13) % 10 < 7 { base_p[bi] } else { -base_p[bi] });
                    vo = vo.saturating_add(if (eh >> 19) % 10 < 7 { base_o[bi] } else { -base_o[bi] });
                }
                s[d] = vs; p[d] = vp; o[d] = vo;
            }
            nodes.push((s, p, o));
        }

        let edges: Vec<ZeckBF17Edge> = nodes.iter()
            .map(|(s, p, o)| ZeckBF17Edge::encode(s, p, o)).collect();
        let mut all_patterns: Vec<BasePattern> = Vec::new();
        for e in &edges {
            all_patterns.push(e.subject.clone());
            all_patterns.push(e.predicate.clone());
            all_patterns.push(e.object.clone());
        }

        // Ground truth
        let full_thresh = FULL_DIM as u32 / 2;
        let n_pairs = n_nodes;
        let mut exact_scents: Vec<u8> = Vec::new();
        for i in 0..n_pairs {
            for j in (i + 1)..n_pairs {
                let (fds, fdp, fdo) = (
                    sign_bit_hamming(&nodes[i].0, &nodes[j].0),
                    sign_bit_hamming(&nodes[i].1, &nodes[j].1),
                    sign_bit_hamming(&nodes[i].2, &nodes[j].2));
                let (sc, pc, oc) = (
                    (fds < full_thresh) as u8,
                    (fdp < full_thresh) as u8,
                    (fdo < full_thresh) as u8);
                exact_scents.push(sc | (pc << 1) | (oc << 2)
                    | ((sc & pc) << 3) | ((sc & oc) << 4)
                    | ((pc & oc) << 5) | ((sc & pc & oc) << 6));
            }
        }

        println!("{:>8} {:>8} {:>10} {:>10} {:>12}",
            "palette", "thresh", "scent%", "ρ(rank)", "bytes/edge");
        println!("{}", "─".repeat(52));

        let mut best_combined = 0.0f64;
        let mut best_cfg = String::new();

        for palette_size in [8, 16, 32, 64, 128, 256] {
            let palette = BasePalette::build(&all_patterns, palette_size);
            let pal_edges: Vec<PaletteEdge> = edges.iter()
                .map(|e| palette.compress(e)).collect();

            // Compute palette distances for ρ
            let mut exact_d: Vec<f64> = Vec::new();
            let mut pal_d: Vec<f64> = Vec::new();
            let mut pal_scents: Vec<u8> = Vec::new();

            for i in 0..n_pairs {
                for j in (i + 1)..n_pairs {
                    exact_d.push(
                        sign_bit_hamming(&nodes[i].0, &nodes[j].0) as f64
                        + sign_bit_hamming(&nodes[i].1, &nodes[j].1) as f64
                        + sign_bit_hamming(&nodes[i].2, &nodes[j].2) as f64);
                    pal_d.push(palette.edge_distance(&pal_edges[i], &pal_edges[j]) as f64);
                }
            }
            let rho = spearman(&exact_d, &pal_d);

            // Sweep scent thresholds for this palette
            let max_l1 = BASE_DIM as u32 * u16::MAX as u32;
            for thresh_frac in [0.001, 0.01, 0.05, 0.10, 0.20, 0.50] {
                let thresh = (max_l1 as f64 * thresh_frac) as u32;
                let mut agree = 0u32;
                let mut total = 0u32;
                let mut pair_idx = 0;
                for i in 0..n_pairs {
                    for j in (i + 1)..n_pairs {
                        let ds = base_l1(palette.lookup(pal_edges[i].s_idx), palette.lookup(pal_edges[j].s_idx));
                        let dp = base_l1(palette.lookup(pal_edges[i].p_idx), palette.lookup(pal_edges[j].p_idx));
                        let d_o = base_l1(palette.lookup(pal_edges[i].o_idx), palette.lookup(pal_edges[j].o_idx));
                        let sc = (ds < thresh) as u8;
                        let pc = (dp < thresh) as u8;
                        let oc = (d_o < thresh) as u8;
                        let pal_scent = sc | (pc << 1) | (oc << 2)
                            | ((sc & pc) << 3) | ((sc & oc) << 4)
                            | ((pc & oc) << 5) | ((sc & pc & oc) << 6);
                        if pal_scent == exact_scents[pair_idx] { agree += 1; }
                        total += 1;
                        pair_idx += 1;
                    }
                }
                let scent_agr = agree as f64 / total.max(1) as f64;
                let combined = rho * 0.6 + scent_agr * 0.4;
                let bytes_per = palette.total_bytes(n_nodes) as f64 / n_nodes as f64;

                if combined > best_combined {
                    best_combined = combined;
                    best_cfg = format!("k={} thresh={:.3}", palette_size, thresh_frac);
                }

                let marker = if rho > 0.937 && scent_agr > 0.50 { " ★★" }
                    else if rho > 0.937 { " ★" }
                    else { "" };

                if scent_agr > 0.15 || thresh_frac == 0.01 || thresh_frac == 0.50 {
                    println!("{:>8} {:>8.3} {:>9.1}% {:>10.4} {:>11.1}{}",
                        palette_size, thresh_frac, scent_agr * 100.0,
                        rho, bytes_per, marker);
                }
            }
        }

        println!("\n  BEST COMBINED: {} → score={:.4}", best_cfg, best_combined);
    }

    // ═══════════════════════════════════════════════════════════════════
    // SWEEP 16: Palette psychometric convergence
    // Does palette compression have its own fixed point?
    // encode → palette-compress → palette-lookup → decode → encode...
    // ═══════════════════════════════════════════════════════════════════
    #[test]
    fn test_palette_convergence() {
        println!("\n{:═<80}", "═ PALETTE PSYCHOMETRIC CONVERGENCE ");

        // Generate 200 base patterns
        let mut rng = 42u64;
        let next = |r: &mut u64| -> u64 {
            *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *r
        };

        let mut patterns: Vec<BasePattern> = Vec::new();
        for _ in 0..200 {
            let mut dims = [0i16; BASE_DIM];
            for d in 0..BASE_DIM {
                let h = next(&mut rng);
                dims[d] = (((h >> 8) % 2001) as i32 - 1000) as i16;
            }
            patterns.push(BasePattern { dims });
        }

        println!("{:>8} {:>6} {:>10} {:>10} {:>12}",
            "palette", "iter", "Δ_total", "unique_idx", "converged?");
        println!("{}", "─".repeat(50));

        for palette_size in [8, 16, 32, 64, 128, 256] {
            let palette = BasePalette::build(&patterns, palette_size);

            // Quantize → lookup → re-quantize, track convergence
            let mut current = patterns.clone();
            for iter in 0..10 {
                let mut new_patterns = Vec::new();
                let mut total_delta = 0u64;
                let mut indices = std::collections::HashSet::new();

                for pat in &current {
                    let idx = palette.nearest(pat);
                    indices.insert(idx);
                    let quantized = palette.lookup(idx).clone();
                    total_delta += base_l1(pat, &quantized) as u64;
                    new_patterns.push(quantized);
                }

                let converged = total_delta == 0;
                let marker = if converged { "★ FIXED" } else { "" };

                println!("{:>8} {:>6} {:>10} {:>10} {:>12}",
                    palette_size, iter, total_delta, indices.len(), marker);

                if converged { break; }
                current = new_patterns;
            }
            println!("{}", "─".repeat(50));
        }
    }
}
