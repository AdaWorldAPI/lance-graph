//! Euler-gamma holographic fold: N similar vectors → 1 container.
//!
//! Multiple similar vectors (same CLAM family) are folded into ONE
//! StackedN container via Euler-gamma rotation at 3σ-separated angles.
//!
//! This is NOT a compression trick — it IS the architecture:
//! - Centroid absorbs mean → zero-centered residuals (normalization step 1: FREE)
//! - γ rotation spreads energy ergodically on torus (normalization step 2: FREE)
//! - √(n+γ) bounds magnitude without per-block constants (normalization step 3: FREE)
//! - Inverse rotation recovers individual members (holographic readout)
//!
//! Recovery quality depends on SNR = √(d×SPD / N_members).
//! At SPD=32, d=17: SNR(N=6) ≈ 9.5 → expected Pearson ~0.96

// cosine_f32_slice reserved for future fold quality measurement
#[allow(unused_imports)]
use crate::stacked_n::{bf16_to_f32, cosine_f32_slice, f32_to_bf16, StackedN};

/// Euler-Mascheroni constant γ ≈ 0.5772156649...
/// Irrational + transcendental → ergodic on torus → no aliasing between members.
const EULER_GAMMA: f64 = std::f64::consts::EULER_GAMMA;

/// Golden step for dim coupling: 11 mod 17 visits all residues.
const GOLDEN_STEP: usize = 11;

/// Number of base dimensions.
const BASE_DIM: usize = 17;

// ═══════════════════════════════════════════════════════════════════════════
// CLAM Family: group similar vectors
// ═══════════════════════════════════════════════════════════════════════════

/// A CLAM family: a group of similar vectors with shared centroid.
#[derive(Clone, Debug)]
pub struct ClamFamily {
    /// Family centroid (element-wise mean of all members), as f32.
    pub centroid_f32: Vec<f32>,
    /// Indices of members in the original vector array.
    pub member_indices: Vec<usize>,
}

/// Group vectors into CLAM families by cosine similarity.
///
/// Uses greedy seed + assign: pick the most distant unseen vector as next seed,
/// assign all vectors within `cos_threshold` of nearest seed.
pub fn clam_group(vectors: &[Vec<f32>], cos_threshold: f64) -> Vec<ClamFamily> {
    let n = vectors.len();
    if n == 0 {
        return Vec::new();
    }

    let mut assigned = vec![false; n];
    let mut families: Vec<ClamFamily> = Vec::new();

    // Precompute norms for fast cosine
    let norms: Vec<f64> = vectors
        .iter()
        .map(|v| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt())
        .collect();

    loop {
        // Find furthest unassigned vector from all existing seeds
        let seed = if families.is_empty() {
            0 // first seed = first vector
        } else {
            let mut best_idx = None;
            let mut best_min_cos = f64::INFINITY;
            for i in 0..n {
                if assigned[i] {
                    continue;
                }
                let min_cos = families
                    .iter()
                    .map(|f| fast_cosine(&vectors[i], &f.centroid_f32, norms[i]))
                    .fold(f64::INFINITY, f64::min);
                if min_cos < best_min_cos {
                    best_min_cos = min_cos;
                    best_idx = Some(i);
                }
            }
            match best_idx {
                Some(i) => i,
                None => break, // all assigned
            }
        };

        if assigned[seed] {
            break;
        }

        // Assign seed + all unassigned vectors within threshold
        let mut members = vec![seed];
        assigned[seed] = true;

        for i in 0..n {
            if assigned[i] {
                continue;
            }
            let cos = fast_cosine(&vectors[i], &vectors[seed], norms[i]);
            if cos >= cos_threshold {
                members.push(i);
                assigned[i] = true;
            }
        }

        // Compute centroid
        let dim = vectors[0].len();
        let mut centroid = vec![0.0f64; dim];
        for &mi in &members {
            for (d, &v) in vectors[mi].iter().enumerate() {
                centroid[d] += v as f64;
            }
        }
        let n_m = members.len() as f64;
        let centroid_f32: Vec<f32> = centroid.iter().map(|&c| (c / n_m) as f32).collect();

        families.push(ClamFamily {
            centroid_f32,
            member_indices: members,
        });
    }

    families
}

/// Fast cosine: uses precomputed norm for one vector.
fn fast_cosine(a: &[f32], b: &[f32], norm_a: f64) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        dot += a[i] as f64 * b[i] as f64;
        nb += (b[i] as f64).powi(2);
    }
    let denom = norm_a * nb.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Euler-Gamma Fold
// ═══════════════════════════════════════════════════════════════════════════

/// A folded family: centroid + holographic container.
///
/// Storage: centroid (1×SPD×17×2 bytes) + folded (1×SPD×17×2 bytes) + member_count.
/// N members compressed into 2× the space of 1 member.
#[derive(Clone, Debug)]
pub struct FoldedFamily {
    /// Centroid encoded as StackedN.
    pub centroid: StackedN,
    /// Holographic container: all member residuals folded via γ-rotation.
    pub folded: StackedN,
    /// Number of members folded in.
    pub n_members: usize,
    /// SPD used for encoding.
    pub spd: usize,
    /// Original vector dimensionality (for recovery).
    pub original_dim: usize,
}

impl FoldedFamily {
    /// Byte size of the folded representation.
    pub fn byte_size(&self) -> usize {
        self.centroid.byte_size() + self.folded.byte_size() + 8 // + metadata
    }

    /// Compression ratio vs storing all members separately.
    pub fn compression_ratio(&self) -> f64 {
        let member_bytes = self.n_members * self.centroid.byte_size();
        member_bytes as f64 / self.byte_size() as f64
    }
}

/// Fold a family of similar f32 vectors into a holographic container.
///
/// 1. Compute centroid (element-wise mean)
/// 2. Subtract centroid → zero-centered residuals (normalization: FREE)
/// 3. Encode residuals as StackedN
/// 4. Euler-gamma rotate each member by γ×i with √(i+γ) radius
/// 5. Sum all rotated members → one container
pub fn euler_gamma_fold(members: &[Vec<f32>], spd: usize) -> FoldedFamily {
    let n = members.len();
    let dim = members[0].len();

    // Step 1: centroid
    let mut centroid_f64 = vec![0.0f64; dim];
    for m in members {
        for (d, &v) in m.iter().enumerate() {
            centroid_f64[d] += v as f64;
        }
    }
    let n_f64 = n as f64;
    let centroid_f32: Vec<f32> = centroid_f64.iter().map(|&c| (c / n_f64) as f32).collect();
    let centroid_enc = StackedN::from_f32(&centroid_f32, spd);

    // Step 2: residuals (member - centroid)
    let residuals: Vec<Vec<f32>> = members
        .iter()
        .map(|m| m.iter().zip(&centroid_f32).map(|(&a, &b)| a - b).collect())
        .collect();

    // Step 3: encode residuals as StackedN
    let encoded_residuals: Vec<StackedN> = residuals
        .iter()
        .map(|r| StackedN::from_f32(r, spd))
        .collect();

    // Step 4+5: Euler-gamma fold
    let total_samples = BASE_DIM * spd;
    let mut folded_data = vec![0.0f64; total_samples];

    for (i, enc) in encoded_residuals.iter().enumerate() {
        let angle = EULER_GAMMA * i as f64;
        let radius = ((i as f64) + EULER_GAMMA).sqrt(); // √(n+γ)
        let cos_a = angle.cos() * radius;
        let sin_a = angle.sin() * radius;

        for d in 0..BASE_DIM {
            let d_next = (d + GOLDEN_STEP) % BASE_DIM;
            for s in 0..spd {
                let idx = d * spd + s;
                let idx_next = d_next * spd + s;
                let v = bf16_to_f32(enc.data[idx]) as f64;
                let v_next = bf16_to_f32(enc.data[idx_next]) as f64;
                let rotated = v * cos_a + v_next * sin_a;
                folded_data[idx] += rotated;
            }
        }
    }

    // Convert folded f64 → BF16
    let folded_bf16: Vec<u16> = folded_data.iter().map(|&v| f32_to_bf16(v as f32)).collect();

    let folded = StackedN {
        samples_per_dim: spd,
        data: folded_bf16,
    };

    FoldedFamily {
        centroid: centroid_enc,
        folded,
        n_members: n,
        spd,
        original_dim: dim,
    }
}

/// Recover member j from a folded family.
///
/// Inverse-rotate by -γ×j: member j's signal reinforces,
/// other members' signals average toward noise floor.
/// Then add centroid back to get the recovered vector.
pub fn euler_gamma_unfold(family: &FoldedFamily, member_index: usize) -> Vec<f32> {
    let spd = family.spd;

    // Inverse rotation
    let angle = -(EULER_GAMMA * member_index as f64);
    let radius = ((member_index as f64) + EULER_GAMMA).sqrt();
    let inv_radius = 1.0 / radius;
    let cos_a = angle.cos() * inv_radius;
    let sin_a = angle.sin() * inv_radius;

    let mut recovered_f64 = vec![0.0f64; BASE_DIM * spd];

    for d in 0..BASE_DIM {
        let d_next = (d + GOLDEN_STEP) % BASE_DIM;
        for s in 0..spd {
            let idx = d * spd + s;
            let idx_next = d_next * spd + s;
            let v = bf16_to_f32(family.folded.data[idx]) as f64;
            let v_next = bf16_to_f32(family.folded.data[idx_next]) as f64;
            let unrotated = v * cos_a + v_next * sin_a;
            recovered_f64[idx] = unrotated;
        }
    }

    // Recovered residual as StackedN
    let recovered_bf16: Vec<u16> = recovered_f64
        .iter()
        .map(|&v| f32_to_bf16(v as f32))
        .collect();
    let recovered = StackedN {
        samples_per_dim: spd,
        data: recovered_bf16,
    };

    // Add centroid back: hydrate both and sum
    let centroid_f32 = family.centroid.hydrate_f32();
    let residual_f32 = recovered.hydrate_f32();

    centroid_f32
        .iter()
        .zip(residual_f32.iter())
        .map(|(&c, &r)| c + r)
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Full pipeline: vectors → families → fold → unfold → measure
// ═══════════════════════════════════════════════════════════════════════════

/// Results from a fold-unfold cycle.
#[derive(Clone, Debug)]
pub struct FoldResult {
    pub n_members: usize,
    pub spd: usize,
    /// Pearson correlation between original and recovered, per member.
    pub pearson_per_member: Vec<f64>,
    /// Mean Pearson across all members.
    pub mean_pearson: f64,
    /// Min Pearson (worst-case member).
    pub min_pearson: f64,
    /// Compression ratio (N members → 2 containers).
    pub compression_ratio: f64,
    /// SNR = √(d × SPD / N) theoretical.
    pub theoretical_snr: f64,
}

/// Run the gate test: fold N members, measure recovery quality.
pub fn gate_test(members: &[Vec<f32>], spd: usize) -> FoldResult {
    let n = members.len();
    let family = euler_gamma_fold(members, spd);

    let mut pearsons = Vec::with_capacity(n);
    for (j, member) in members.iter().enumerate().take(n) {
        let recovered = euler_gamma_unfold(&family, j);

        // Compute Pearson between original and recovered
        // (on the hydrated StackedN representation, not raw f32)
        let orig_enc = StackedN::from_f32(member, spd);
        let orig_f32 = orig_enc.hydrate_f32();

        let r = crate::quality::pearson(
            &orig_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            &recovered.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        );
        pearsons.push(r);
    }

    let mean = pearsons.iter().sum::<f64>() / n as f64;
    let min = pearsons.iter().cloned().fold(f64::INFINITY, f64::min);
    let snr = ((BASE_DIM as f64) * (spd as f64) / (n as f64)).sqrt();

    FoldResult {
        n_members: n,
        spd,
        pearson_per_member: pearsons,
        mean_pearson: mean,
        min_pearson: min,
        compression_ratio: family.compression_ratio(),
        theoretical_snr: snr,
    }
}

/// NeuronPrint 6D fold: fold 6 role vectors into 1 container.
pub fn neuronprint_fold_test(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    spd: usize,
) -> FoldResult {
    let members = vec![
        q.to_vec(),
        k.to_vec(),
        v.to_vec(),
        gate.to_vec(),
        up.to_vec(),
        down.to_vec(),
    ];
    gate_test(&members, spd)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate N vectors with controlled similarity (cosine ≈ base_cos).
    fn make_similar_vectors(n: usize, dim: usize, base_cos: f64) -> Vec<Vec<f32>> {
        // Start with a base vector, perturb slightly
        let base: Vec<f32> = (0..dim)
            .map(|d| ((d as f32 * 0.01).sin() * 0.5 + (d as f32 * 0.003).cos() * 0.3))
            .collect();

        let noise_scale = (1.0 - base_cos).sqrt() as f32;

        (0..n)
            .map(|i| {
                base.iter()
                    .enumerate()
                    .map(|(d, &b)| {
                        let noise = ((d * 97 + i * 31) as f32 % 100.0 - 50.0) * 0.01 * noise_scale;
                        b + noise
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn fold_unfold_two_members() {
        let members = make_similar_vectors(2, 1024, 0.95);
        let result = gate_test(&members, 32);
        eprintln!(
            "N=2: mean_pearson={:.4}, min={:.4}, SNR={:.1}, ratio={:.1}×",
            result.mean_pearson,
            result.min_pearson,
            result.theoretical_snr,
            result.compression_ratio
        );
        assert!(
            result.mean_pearson > 0.5,
            "N=2 should recover well: {:.4}",
            result.mean_pearson
        );
    }

    #[test]
    fn fold_unfold_four_members() {
        let members = make_similar_vectors(4, 1024, 0.95);
        let result = gate_test(&members, 32);
        eprintln!(
            "N=4: mean_pearson={:.4}, min={:.4}, SNR={:.1}, ratio={:.1}×",
            result.mean_pearson,
            result.min_pearson,
            result.theoretical_snr,
            result.compression_ratio
        );
        assert!(
            result.mean_pearson > 0.3,
            "N=4 should recover: {:.4}",
            result.mean_pearson
        );
    }

    #[test]
    fn fold_unfold_six_members_neuronprint() {
        // The NeuronPrint gate test: 6 roles folded into 1 container
        let members = make_similar_vectors(6, 1024, 0.90);
        let result = gate_test(&members, 32);
        eprintln!(
            "N=6 (NeuronPrint): mean_pearson={:.4}, min={:.4}, SNR={:.1}, ratio={:.1}×",
            result.mean_pearson,
            result.min_pearson,
            result.theoretical_snr,
            result.compression_ratio
        );
        // This is the GATE: does 6-fold work?
    }

    #[test]
    fn clam_grouping_basic() {
        let mut vectors = Vec::new();
        // Group A: similar to each other
        for i in 0..10 {
            vectors.push(
                (0..100)
                    .map(|d| (d as f32 + i as f32 * 0.1) * 0.01)
                    .collect(),
            );
        }
        // Group B: different direction
        for i in 0..10 {
            vectors.push(
                (0..100)
                    .map(|d| -(d as f32 + i as f32 * 0.1) * 0.01)
                    .collect(),
            );
        }

        let families = clam_group(&vectors, 0.8);
        assert!(
            families.len() >= 2,
            "should find at least 2 families: {}",
            families.len()
        );
        eprintln!(
            "Found {} families from {} vectors",
            families.len(),
            vectors.len()
        );
        for (i, f) in families.iter().enumerate() {
            eprintln!("  Family {}: {} members", i, f.member_indices.len());
        }
    }

    #[test]
    fn compression_ratio_increases_with_n() {
        let members = make_similar_vectors(16, 512, 0.95);

        for n in [2, 4, 8, 16] {
            let result = gate_test(&members[..n], 32);
            eprintln!(
                "N={:>2}: ratio={:.1}×, pearson={:.4}, SNR={:.1}",
                n, result.compression_ratio, result.mean_pearson, result.theoretical_snr
            );
        }
    }

    #[test]
    fn gate_test_sweep() {
        // THE gate test: sweep N from 2 to 16
        let members = make_similar_vectors(16, 1024, 0.92);
        println!("\n=== GATE TEST: Euler-gamma fold recovery ===");
        println!("N  │ Mean ρ │ Min ρ  │ SNR  │ Ratio");
        println!("───┼────────┼────────┼──────┼──────");
        for n in [2, 4, 6, 8, 12, 16] {
            let result = gate_test(&members[..n], 32);
            println!(
                "{:>2} │ {:>.4} │ {:>.4} │ {:>4.1} │ {:>4.1}×",
                n,
                result.mean_pearson,
                result.min_pearson,
                result.theoretical_snr,
                result.compression_ratio
            );
        }
    }
}
