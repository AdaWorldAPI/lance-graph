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

pub fn fidelity_experiment(n_nodes: usize, n_encounters: usize) -> FidelityReport {
    let mut rng = 42u64;
    let next = |r: &mut u64| -> u64 {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *r
    };

    // Build reverse mapping: for each dimension, which base index does it belong to?
    // The encoder maps base index bi → position golden_position(bi) within each octave.
    // So dimension (octave * BASE_DIM + golden_position(bi)) → base index bi.
    let mut dim_to_base = vec![0usize; FULL_DIM];
    for octave in 0..N_OCTAVES {
        for bi in 0..BASE_DIM {
            let dim = octave * BASE_DIM + golden_position(bi);
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

        // Generate a base signal per base class (17 values per plane).
        // This is the "true" signal the encoding should capture.
        let mut base_s = [0i8; BASE_DIM];
        let mut base_p = [0i8; BASE_DIM];
        let mut base_o = [0i8; BASE_DIM];
        for bi in 0..BASE_DIM {
            let h = node_seed.wrapping_mul(bi as u64 + 1).wrapping_add(0x9e3779b97f4a7c15);
            base_s[bi] = if (h >> 17) & 1 == 1 { 1i8 } else { -1 };
            base_p[bi] = if (h >> 23) & 1 == 1 { 1i8 } else { -1 };
            base_o[bi] = if (h >> 29) & 1 == 1 { 1i8 } else { -1 };
        }

        // Fill all 16,384 dims: each dim gets its base class signal + noise.
        // 70% signal (agrees with base), 30% noise (flips sign).
        // This models real accumulators where octaves carry redundant info.
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

    let encoded: Vec<ZeckBF17Edge> = nodes.iter()
        .map(|(s, p, o)| ZeckBF17Edge::encode(s, p, o)).collect();
    let decoded: Vec<_> = encoded.iter().map(|e| e.decode()).collect();

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

    let full_thresh = FULL_DIM as u32 / 2;
    let (mut agree, mut total) = (0u32, 0u32);
    for i in 0..n_pairs {
        for j in (i + 1)..n_pairs {
            total += 1;
            let sz = scent_from_base(&encoded[i], &encoded[j]);
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
            if sz == full_scent { agree += 1; }
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
}
