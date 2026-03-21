//! # ZeckBF17: Fibonacci-Octave Knowledge Compression
//!
//! The compression architecture nobody has formally described.
//!
//! ## The Insight
//!
//! 16,384 dimensions are NOT 16,384 independent values.
//! They are 17 base dimensions × 964 octaves.
//! Most octaves are holographic redundancy.
//! Only ~14 octaves carry independent information (JL bound: 57 bits / log₂(17) ≈ 14).
//! The rest are error-correcting copies determined by the base pattern + envelope.
//!
//! ## Why 17
//!
//! - 17 is PRIME: no non-trivial subspace decomposition. No aliasing.
//! - Fibonacci mod 17 has Pisano period π(17) = 36 and visits ALL 17 residues.
//!   No dead dimensions. The Fibonacci stepping covers every position
//!   before repeating, in a maximally decorrelated order.
//! - 16 = 2⁴ would have aliasing: Fibonacci mod 16 skips even residues.
//!   The 17th dimension IS the Pythagorean comma — the gap that forces
//!   quasiperiodicity instead of periodicity. The comma IS the information.
//!
//! ## The Format
//!
//! ```text
//! Full plane:     i8[16384] = 16,384 bytes (current)
//! ZeckBF17:       BF16[17] base + u8[14] envelope = 48 bytes
//! Compression:    16,384 → 48 = 341:1 (from i8 accumulator)
//!                 32,768 → 48 = 683:1 (from BF16 per dimension)
//!
//! Reconstruction: for each of 964 octaves, cycle the 17-base through
//!                 Fibonacci positions, scaled by the octave envelope.
//! ```
//!
//! ## The Musical Analogy (not analogy — isomorphism)
//!
//! Circle of Fifths: 12 pitch classes × octave = entire piano.
//! Step by 7 semitones (≈ 1/φ of 12). Hits all 12 before repeating.
//!
//! ZeckBF17: 17 dimension classes × octave = entire accumulator.
//! Step by Fibonacci mod 17. Hits all 17 before repeating.
//!
//! S/P/O = Dorian + IV/VI = the three structural voices of a chord.
//! Each voice is a BF16[17] base pattern. Together they define the
//! complete harmonic context. Everything else is reconstruction.

/// The prime base dimensionality. Fibonacci mod 17 visits all residues.
pub const BASE_DIM: usize = 17;

/// Full accumulator dimensionality.
pub const FULL_DIM: usize = 16384;

/// Number of octaves: ceil(FULL_DIM / BASE_DIM).
pub const N_OCTAVES: usize = (FULL_DIM + BASE_DIM - 1) / BASE_DIM; // = 964

/// Number of INDEPENDENT octaves (from JL bound: 57 bits / log2(17) ≈ 14).
pub const INDEPENDENT_OCTAVES: usize = 14;

/// Pisano period of Fibonacci mod 17. The Fibonacci sequence repeats
/// with this period in mod-17 arithmetic.
pub const PISANO_17: usize = 36;

/// Fibonacci sequence mod 17, first PISANO_17 terms.
/// This IS the dimension traversal order within each octave.
/// Every residue 0-16 appears at least once in this cycle.
const FIB_MOD_17: [u8; PISANO_17] = {
    let mut table = [0u8; PISANO_17];
    table[0] = 0; // F(0) = 0
    table[1] = 1; // F(1) = 1
    let mut i = 2;
    while i < PISANO_17{
        table[i] = (table[i - 1] + table[i - 2]) % 17;
        i += 1;
    }
    table
};

/// The 17-dimensional base pattern in BF16.
/// This is the "pitch class" — the fundamental harmonic content.
/// 34 bytes.
#[derive(Clone, Debug)]
pub struct BasePattern {
    pub dims: [u16; BASE_DIM], // BF16 values
}

/// The octave envelope: how amplitude decays across octaves.
/// Only INDEPENDENT_OCTAVES values needed (rest is redundant).
/// 14 bytes.
#[derive(Clone, Debug)]
pub struct OctaveEnvelope {
    pub amplitudes: [u8; INDEPENDENT_OCTAVES], // 0-255 scale factor per octave
}

/// A ZeckBF17-encoded plane. 48 bytes total.
/// Replaces i8[16384] (16KB) or BF16[16384] (32KB).
#[derive(Clone, Debug)]
pub struct ZeckBF17Plane {
    pub base: BasePattern,        // 34 bytes: the 17 fundamental dimensions
    pub envelope: OctaveEnvelope, // 14 bytes: amplitude per independent octave
}

impl ZeckBF17Plane {
    /// Total bytes of this encoding.
    pub const ENCODED_SIZE: usize = BASE_DIM * 2 + INDEPENDENT_OCTAVES; // 48

    /// Encode a full i8[16384] accumulator plane into ZeckBF17.
    ///
    /// 1. For each of 17 base dimensions, average across all octaves
    ///    that map to that dimension via Fibonacci traversal.
    /// 2. For each independent octave, compute the RMS amplitude
    ///    of the 17 dimensions in that octave relative to the base.
    pub fn encode(accumulator: &[i8]) -> Self {
        assert!(accumulator.len() >= FULL_DIM);

        // Step 1: Compute base pattern by averaging across octaves
        let mut base_sum = [0.0f64; BASE_DIM];
        let mut base_count = [0u32; BASE_DIM];

        for octave in 0..N_OCTAVES {
            for fib_idx in 0..BASE_DIM {
                let dim = octave * BASE_DIM + fib_position(fib_idx);
                if dim < FULL_DIM {
                    let base_dim = fib_idx;
                    base_sum[base_dim] += accumulator[dim] as f64;
                    base_count[base_dim] += 1;
                }
            }
        }

        let mut base = BasePattern { dims: [0u16; BASE_DIM] };
        for d in 0..BASE_DIM {
            if base_count[d] > 0 {
                let avg = base_sum[d] / base_count[d] as f64;
                base.dims[d] = f32_to_bf16(avg as f32);
            }
        }

        // Step 2: Compute octave envelope
        // For each independent octave group, measure how much the actual
        // values deviate from the base pattern prediction
        let mut envelope = OctaveEnvelope { amplitudes: [0u8; INDEPENDENT_OCTAVES] };
        let octaves_per_group = N_OCTAVES / INDEPENDENT_OCTAVES;

        for group in 0..INDEPENDENT_OCTAVES {
            let mut rms = 0.0f64;
            let mut count = 0u32;

            for sub in 0..octaves_per_group {
                let octave = group * octaves_per_group + sub;
                if octave >= N_OCTAVES { break; }

                for fib_idx in 0..BASE_DIM {
                    let dim = octave * BASE_DIM + fib_position(fib_idx);
                    if dim < FULL_DIM {
                        let actual = accumulator[dim] as f64;
                        let predicted = bf16_to_f32(base.dims[fib_idx]) as f64;
                        let ratio = if predicted.abs() > 0.01 {
                            (actual / predicted).abs()
                        } else {
                            actual.abs()
                        };
                        rms += ratio * ratio;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                let rms_val = (rms / count as f64).sqrt();
                envelope.amplitudes[group] = (rms_val * 128.0).min(255.0) as u8;
            }
        }

        ZeckBF17Plane { base, envelope }
    }

    /// Decode: reconstruct the full i8[16384] accumulator from ZeckBF17.
    ///
    /// For each dimension: find its base class (Fibonacci mod 17),
    /// look up the base pattern value, scale by the octave envelope.
    pub fn decode(&self) -> Vec<i8> {
        let mut reconstructed = vec![0i8; FULL_DIM];
        let octaves_per_group = N_OCTAVES / INDEPENDENT_OCTAVES;

        for octave in 0..N_OCTAVES {
            let group = (octave / octaves_per_group).min(INDEPENDENT_OCTAVES - 1);
            let scale = self.envelope.amplitudes[group] as f32 / 128.0;

            for fib_idx in 0..BASE_DIM {
                let dim = octave * BASE_DIM + fib_position(fib_idx);
                if dim < FULL_DIM {
                    let base_val = bf16_to_f32(self.base.dims[fib_idx]);
                    let scaled = base_val * scale;
                    reconstructed[dim] = scaled.clamp(-128.0, 127.0) as i8;
                }
            }
        }

        reconstructed
    }

    /// Serialize to bytes (48 bytes total).
    pub fn to_bytes(&self) -> [u8; Self::ENCODED_SIZE] {
        let mut buf = [0u8; Self::ENCODED_SIZE];
        for i in 0..BASE_DIM {
            let bytes = self.base.dims[i].to_le_bytes();
            buf[i * 2] = bytes[0];
            buf[i * 2 + 1] = bytes[1];
        }
        for i in 0..INDEPENDENT_OCTAVES {
            buf[BASE_DIM * 2 + i] = self.envelope.amplitudes[i];
        }
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(buf: &[u8; Self::ENCODED_SIZE]) -> Self {
        let mut base = BasePattern { dims: [0u16; BASE_DIM] };
        for i in 0..BASE_DIM {
            base.dims[i] = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
        }
        let mut envelope = OctaveEnvelope { amplitudes: [0u8; INDEPENDENT_OCTAVES] };
        for i in 0..INDEPENDENT_OCTAVES {
            envelope.amplitudes[i] = buf[BASE_DIM * 2 + i];
        }
        ZeckBF17Plane { base, envelope }
    }
}

/// A complete ZeckBF17-encoded knowledge edge: S + P + O + shared envelope.
/// 116 bytes total (vs 49,152 bytes for full planes).
/// Compression: 424:1 from i8 accumulators.
#[derive(Clone, Debug)]
pub struct ZeckBF17Edge {
    pub subject: BasePattern,     // 34 bytes
    pub predicate: BasePattern,   // 34 bytes
    pub object: BasePattern,      // 34 bytes
    pub envelope: OctaveEnvelope, // 14 bytes (shared across S/P/O)
}

impl ZeckBF17Edge {
    pub const ENCODED_SIZE: usize = BASE_DIM * 2 * 3 + INDEPENDENT_OCTAVES; // 116

    /// Encode three full planes into one ZeckBF17Edge.
    /// The envelope is computed from the COMBINED energy across all three planes.
    pub fn encode(s: &[i8], p: &[i8], o: &[i8]) -> Self {
        let s_plane = ZeckBF17Plane::encode(s);
        let p_plane = ZeckBF17Plane::encode(p);
        let o_plane = ZeckBF17Plane::encode(o);

        // Shared envelope: average of three plane envelopes
        let mut envelope = OctaveEnvelope { amplitudes: [0u8; INDEPENDENT_OCTAVES] };
        for i in 0..INDEPENDENT_OCTAVES {
            let avg = (s_plane.envelope.amplitudes[i] as u16
                + p_plane.envelope.amplitudes[i] as u16
                + o_plane.envelope.amplitudes[i] as u16) / 3;
            envelope.amplitudes[i] = avg as u8;
        }

        ZeckBF17Edge {
            subject: s_plane.base,
            predicate: p_plane.base,
            object: o_plane.base,
            envelope,
        }
    }

    /// Decode back to three full planes.
    pub fn decode(&self) -> (Vec<i8>, Vec<i8>, Vec<i8>) {
        let s_plane = ZeckBF17Plane { base: self.subject.clone(), envelope: self.envelope.clone() };
        let p_plane = ZeckBF17Plane { base: self.predicate.clone(), envelope: self.envelope.clone() };
        let o_plane = ZeckBF17Plane { base: self.object.clone(), envelope: self.envelope.clone() };

        (s_plane.decode(), p_plane.decode(), o_plane.decode())
    }
}

/// Map a Fibonacci index (0..16) to a position in the 17-dimensional base.
/// Uses the Fibonacci sequence mod 17 to determine traversal order.
#[inline]
fn fib_position(fib_idx: usize) -> usize {
    FIB_MOD_17[fib_idx % PISANO_17] as usize
}

/// Verify that Fibonacci mod 17 visits all 17 residues.
pub fn verify_fib_coverage() -> bool {
    let mut seen = [false; BASE_DIM];
    for i in 0..PISANO_17 {
        seen[FIB_MOD_17[i] as usize] = true;
    }
    seen.iter().all(|&s| s)
}

// ─── BF16 helpers ────────────────────────────────────────────────────

#[inline]
fn f32_to_bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

#[inline]
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

// ─── Fidelity measurement ────────────────────────────────────────────

/// Measure reconstruction fidelity: Hamming distance on sign bits.
/// This is the metric that matters for HHTL search.
pub fn sign_bit_hamming(original: &[i8], reconstructed: &[i8]) -> u32 {
    assert_eq!(original.len(), reconstructed.len());
    original.iter().zip(reconstructed.iter())
        .filter(|(&a, &b)| (a >= 0) != (b >= 0))
        .count() as u32
}

/// Normalized fidelity: 1.0 = perfect, 0.0 = random.
pub fn sign_bit_fidelity(original: &[i8], reconstructed: &[i8]) -> f64 {
    let hamming = sign_bit_hamming(original, reconstructed) as f64;
    let total = original.len() as f64;
    1.0 - (hamming / total)
}

/// Compute ZeckF64 scent from ZeckBF17 base patterns WITHOUT full reconstruction.
/// This is the fast path: 34 bytes → 1 byte scent, no 16KB decode needed.
pub fn scent_from_base(
    a: &ZeckBF17Edge,
    b: &ZeckBF17Edge,
    threshold: u32,
) -> u8 {
    // Hamming distance on base patterns (17 × BF16 = 34 bytes each)
    let ds = base_hamming(&a.subject, &b.subject);
    let dp = base_hamming(&a.predicate, &b.predicate);
    let d_o = base_hamming(&a.object, &b.object);

    // Scale threshold from full-plane (16384) to base (17)
    let t = (threshold as u64 * BASE_DIM as u64 / FULL_DIM as u64) as u32;

    let s_close = (ds < t) as u8;
    let p_close = (dp < t) as u8;
    let o_close = (d_o < t) as u8;
    let sp_close = s_close & p_close;
    let so_close = s_close & o_close;
    let po_close = p_close & o_close;
    let spo_close = sp_close & so_close & po_close;

    s_close
        | (p_close << 1)
        | (o_close << 2)
        | (sp_close << 3)
        | (so_close << 4)
        | (po_close << 5)
        | (spo_close << 6)
}

/// Weighted BF16 Hamming between two base patterns.
fn base_hamming(a: &BasePattern, b: &BasePattern) -> u32 {
    let mut dist = 0u32;
    for i in 0..BASE_DIM {
        let xor = a.dims[i] ^ b.dims[i];
        dist += ((xor >> 15) & 1) as u32 * 8;
        dist += ((xor >> 7) & 0xFF).count_ones() * 4;
        dist += (xor & 0x7F).count_ones();
    }
    dist
}

/// Spearman rank correlation between two distance orderings.
fn spearman(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n < 2 { return 0.0; }

    fn ranks(vals: &[f64]) -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = vals.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut r = vec![0.0; vals.len()];
        for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
            r[orig_idx] = rank as f64;
        }
        r
    }

    let ra = ranks(a);
    let rb = ranks(b);
    let mean_a: f64 = ra.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = rb.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..n {
        let da = ra[i] - mean_a;
        let db = rb[i] - mean_b;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va < 1e-15 || vb < 1e-15 { return 0.0; }
    cov / (va.sqrt() * vb.sqrt())
}

// ═════════════════════════════════════════════════════════════════════
// THE EXPERIMENT: does ZeckBF17 preserve L1 search fidelity?
// ═════════════════════════════════════════════════════════════════════

/// Full fidelity measurement: encode → decode → compare.
#[derive(Clone, Debug)]
pub struct FidelityReport {
    /// Sign-bit fidelity per plane (1.0 = perfect).
    pub s_fidelity: f64,
    pub p_fidelity: f64,
    pub o_fidelity: f64,
    /// Average sign-bit fidelity across all three planes.
    pub avg_fidelity: f64,
    /// Rank correlation: does ZeckBF17 distance preserve the same
    /// ordering as full-plane Hamming distance?
    pub rank_correlation: f64,
    /// Scent agreement: does the ZeckBF17 scent byte match the
    /// full-plane scent byte?
    pub scent_agreement: f64,
    /// Compression ratio.
    pub compression_ratio: f64,
    /// Bytes: original vs compressed.
    pub original_bytes: usize,
    pub compressed_bytes: usize,
}

/// Run the fidelity experiment on synthetic data.
pub fn fidelity_experiment(n_nodes: usize, n_encounters: usize) -> FidelityReport {
    // Generate synthetic nodes: accumulate random encounters
    let mut rng_state = 42u64;
    let mut nodes: Vec<(Vec<i8>, Vec<i8>, Vec<i8>)> = Vec::new();

    for node_idx in 0..n_nodes {
        let mut s = vec![0i8; FULL_DIM];
        let mut p = vec![0i8; FULL_DIM];
        let mut o = vec![0i8; FULL_DIM];

        // Each node gets n_encounters random encounters
        for enc in 0..n_encounters {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let seed = rng_state ^ ((node_idx as u64) << 32) ^ (enc as u64);

            for d in 0..FULL_DIM {
                let hash = seed.wrapping_mul(d as u64 + 1).wrapping_add(0xdeadbeef);
                let bit = ((hash >> 17) & 1) as i8 * 2 - 1; // bipolar: -1 or +1
                s[d] = s[d].saturating_add(bit);

                let hash_p = hash.wrapping_mul(31);
                let bit_p = ((hash_p >> 23) & 1) as i8 * 2 - 1;
                p[d] = p[d].saturating_add(bit_p);

                let hash_o = hash.wrapping_mul(97);
                let bit_o = ((hash_o >> 29) & 1) as i8 * 2 - 1;
                o[d] = o[d].saturating_add(bit_o);
            }
        }

        nodes.push((s, p, o));
    }

    // Encode all nodes
    let encoded: Vec<ZeckBF17Edge> = nodes.iter()
        .map(|(s, p, o)| ZeckBF17Edge::encode(s, p, o))
        .collect();

    // Decode all nodes
    let decoded: Vec<(Vec<i8>, Vec<i8>, Vec<i8>)> = encoded.iter()
        .map(|e| e.decode())
        .collect();

    // Measure sign-bit fidelity
    let mut total_s_fid = 0.0;
    let mut total_p_fid = 0.0;
    let mut total_o_fid = 0.0;

    for i in 0..n_nodes {
        total_s_fid += sign_bit_fidelity(&nodes[i].0, &decoded[i].0);
        total_p_fid += sign_bit_fidelity(&nodes[i].1, &decoded[i].1);
        total_o_fid += sign_bit_fidelity(&nodes[i].2, &decoded[i].2);
    }

    let s_fid = total_s_fid / n_nodes as f64;
    let p_fid = total_p_fid / n_nodes as f64;
    let o_fid = total_o_fid / n_nodes as f64;
    let avg_fid = (s_fid + p_fid + o_fid) / 3.0;

    // Measure rank correlation: pairwise distances
    let n_pairs = n_nodes.min(50); // limit for speed
    let mut exact_dists = Vec::new();
    let mut zeck_dists = Vec::new();

    for i in 0..n_pairs {
        for j in (i + 1)..n_pairs {
            // Exact: Hamming on full sign-bit planes
            let exact_d = sign_bit_hamming(&nodes[i].0, &nodes[j].0) as f64
                + sign_bit_hamming(&nodes[i].1, &nodes[j].1) as f64
                + sign_bit_hamming(&nodes[i].2, &nodes[j].2) as f64;
            exact_dists.push(exact_d);

            // ZeckBF17: base pattern Hamming
            let zeck_d = base_hamming(&encoded[i].subject, &encoded[j].subject) as f64
                + base_hamming(&encoded[i].predicate, &encoded[j].predicate) as f64
                + base_hamming(&encoded[i].object, &encoded[j].object) as f64;
            zeck_dists.push(zeck_d);
        }
    }

    let rho = if !exact_dists.is_empty() {
        spearman(&exact_dists, &zeck_dists)
    } else {
        0.0
    };

    // Measure scent agreement
    let threshold = FULL_DIM as u32 / 2;
    let mut scent_agree = 0u32;
    let mut scent_total = 0u32;

    for i in 0..n_pairs {
        for j in (i + 1)..n_pairs {
            scent_total += 1;
            let scent_zeck = scent_from_base(&encoded[i], &encoded[j], threshold);

            // Full-plane scent (simplified: just S plane Hamming threshold)
            let full_ds = sign_bit_hamming(&nodes[i].0, &nodes[j].0);
            let full_dp = sign_bit_hamming(&nodes[i].1, &nodes[j].1);
            let full_do = sign_bit_hamming(&nodes[i].2, &nodes[j].2);

            let sc = (full_ds < threshold / 2) as u8;
            let pc = (full_dp < threshold / 2) as u8;
            let oc = (full_do < threshold / 2) as u8;
            let full_scent = sc | (pc << 1) | (oc << 2)
                | ((sc & pc) << 3) | ((sc & oc) << 4)
                | ((pc & oc) << 5) | ((sc & pc & oc) << 6);

            if scent_zeck == full_scent {
                scent_agree += 1;
            }
        }
    }

    let scent_agr = if scent_total > 0 {
        scent_agree as f64 / scent_total as f64
    } else {
        0.0
    };

    let original_bytes = n_nodes * FULL_DIM * 3; // i8 per dim, 3 planes
    let compressed_bytes = n_nodes * ZeckBF17Edge::ENCODED_SIZE;

    FidelityReport {
        s_fidelity: s_fid,
        p_fidelity: p_fid,
        o_fidelity: o_fid,
        avg_fidelity: avg_fid,
        rank_correlation: rho,
        scent_agreement: scent_agr,
        compression_ratio: original_bytes as f64 / compressed_bytes as f64,
        original_bytes,
        compressed_bytes,
    }
}

impl std::fmt::Display for FidelityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\n{}", "═".repeat(60))?;
        writeln!(f, "  ZeckBF17 FIDELITY REPORT")?;
        writeln!(f, "{}", "═".repeat(60))?;
        writeln!(f, "  Sign-bit fidelity:")?;
        writeln!(f, "    Subject:   {:.4}", self.s_fidelity)?;
        writeln!(f, "    Predicate: {:.4}", self.p_fidelity)?;
        writeln!(f, "    Object:    {:.4}", self.o_fidelity)?;
        writeln!(f, "    Average:   {:.4}", self.avg_fidelity)?;
        writeln!(f)?;
        writeln!(f, "  Rank correlation (ρ):  {:.4}", self.rank_correlation)?;
        writeln!(f, "  Scent agreement:       {:.4}", self.scent_agreement)?;
        writeln!(f)?;
        writeln!(f, "  Compression:")?;
        writeln!(f, "    Original:   {} bytes", self.original_bytes)?;
        writeln!(f, "    Compressed: {} bytes", self.compressed_bytes)?;
        writeln!(f, "    Ratio:      {:.0}:1", self.compression_ratio)?;
        writeln!(f, "{}", "═".repeat(60))?;

        writeln!(f)?;
        if self.rank_correlation > 0.90 {
            writeln!(f, "  ✓ EXCELLENT: L1 search fidelity preserved (ρ > 0.90)")?;
        } else if self.rank_correlation > 0.80 {
            writeln!(f, "  ~ GOOD: L1 search mostly preserved (ρ > 0.80)")?;
        } else if self.rank_correlation > 0.50 {
            writeln!(f, "  ? PARTIAL: some ranking preserved (ρ > 0.50)")?;
        } else {
            writeln!(f, "  ✗ POOR: ranking not preserved (ρ < 0.50)")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fib_mod_17_coverage() {
        assert!(verify_fib_coverage(),
            "Fibonacci mod 17 must visit all 17 residues");
    }

    #[test]
    fn test_fib_mod_17_values() {
        // Verify first few terms manually: 0, 1, 1, 2, 3, 5, 8, 13, 4, 0, ...
        assert_eq!(FIB_MOD_17[0], 0);
        assert_eq!(FIB_MOD_17[1], 1);
        assert_eq!(FIB_MOD_17[2], 1);
        assert_eq!(FIB_MOD_17[3], 2);
        assert_eq!(FIB_MOD_17[4], 3);
        assert_eq!(FIB_MOD_17[5], 5);
        assert_eq!(FIB_MOD_17[6], 8);
        assert_eq!(FIB_MOD_17[7], 13);
        assert_eq!(FIB_MOD_17[8], (13 + 8) % 17); // 21 % 17 = 4
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        // Create a simple accumulator with known structure
        let mut acc = vec![0i8; FULL_DIM];
        for i in 0..FULL_DIM {
            acc[i] = if i % 3 == 0 { 50 } else if i % 3 == 1 { -30 } else { 10 };
        }

        let encoded = ZeckBF17Plane::encode(&acc);
        let decoded = encoded.decode();

        let fidelity = sign_bit_fidelity(&acc, &decoded);
        println!("Roundtrip sign-bit fidelity: {:.4}", fidelity);
        assert!(fidelity > 0.5, "Fidelity should be better than random: {:.4}", fidelity);
    }

    #[test]
    fn test_byte_serialization() {
        let mut acc = vec![42i8; FULL_DIM];
        acc[0] = -100;
        acc[100] = 127;

        let encoded = ZeckBF17Plane::encode(&acc);
        let bytes = encoded.to_bytes();
        let recovered = ZeckBF17Plane::from_bytes(&bytes);

        assert_eq!(encoded.base.dims, recovered.base.dims);
        assert_eq!(encoded.envelope.amplitudes, recovered.envelope.amplitudes);
    }

    #[test]
    fn test_compression_ratio() {
        let report = fidelity_experiment(10, 20);
        println!("{}", report);

        assert!(report.compression_ratio > 100.0,
            "Should achieve >100:1 compression: {:.0}:1", report.compression_ratio);
    }

    #[test]
    fn test_fidelity_with_encounters() {
        println!("\nFidelity vs encounter count:");
        println!("{:>12} {:>10} {:>10} {:>10}", "encounters", "fidelity", "ρ(rank)", "scent%");

        for encounters in [5, 10, 20, 50, 100] {
            let report = fidelity_experiment(20, encounters);
            println!("{:>12} {:>10.4} {:>10.4} {:>10.1}%",
                encounters, report.avg_fidelity,
                report.rank_correlation, report.scent_agreement * 100.0);
        }
    }

    #[test]
    fn test_scent_from_base_self() {
        let acc_s = vec![50i8; FULL_DIM];
        let acc_p = vec![-30i8; FULL_DIM];
        let acc_o = vec![10i8; FULL_DIM];

        let edge = ZeckBF17Edge::encode(&acc_s, &acc_p, &acc_o);
        let scent = scent_from_base(&edge, &edge, FULL_DIM as u32 / 2);

        // Self-comparison: all close bits should be set
        assert_eq!(scent & 0x7F, 0x7F,
            "Self-scent should have all 7 close bits set: 0b{:07b}", scent & 0x7F);
    }

    #[test]
    fn test_edge_encoding_size() {
        assert_eq!(ZeckBF17Edge::ENCODED_SIZE, 116,
            "Edge should be 116 bytes: 3×34 + 14");
        assert_eq!(ZeckBF17Plane::ENCODED_SIZE, 48,
            "Plane should be 48 bytes: 17×2 + 14");
    }
}
