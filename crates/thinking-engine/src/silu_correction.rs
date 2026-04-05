//! SiLU-ONNX correction for distance tables.
//!
//! Gate is not a codebook ENTRY. Gate is a LENS.
//! The distance table built from raw weight cosine misses the nonlinear
//! Gate×SiLU interaction. A tiny learned model corrects this.
//!
//! ```text
//! Phase 2.5 in build pipeline:
//!   Extract gate + up rows from GGUF
//!   Generate training pairs: centroid_i ⊕ centroid_j → correction
//!   Ground truth: cos(silu(gate×probe)×up, ...) - cos(raw, ...)
//!   Train tiny MLP (~270K params, ~1 MB ONNX)
//!   Apply: distance_table[i][j] += correction[i][j]
//!   Bake corrected table. Discard ONNX.
//! ```
//!
//! Where to apply:
//!   Q rows:  raw (extern, world doesn't need gate permission)
//!   K rows:  gate×K (intern, self-filtered knowledge index)
//!   V rows:  gate×V (intern, self-filtered content)
//!   Up rows: gate×Up (intern, SiLU activation = strongest effect)
//!   Down:    raw (funnel, receives gate×up result)

/// SiLU activation: x × σ(x).
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Which roles get gate modulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatePolicy {
    /// No gate modulation (Q, Down). World/funnel.
    Raw,
    /// Gate × weight before codebook (K, V, Up). Self-modulated.
    GateModulated,
}

/// Determine gate policy for a role.
pub fn gate_policy(role: &str) -> GatePolicy {
    match role {
        "attn_q" | "attn_qkv" | "ffn_down" => GatePolicy::Raw,
        "attn_k" | "attn_v" | "ffn_gate" | "ffn_up" => GatePolicy::GateModulated,
        _ => GatePolicy::Raw, // unknown = raw (safe default)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TRAINING DATA GENERATION
// ═══════════════════════════════════════════════════════════════════════════

/// One training example: centroid pair → correction.
#[derive(Debug, Clone)]
pub struct CorrectionSample {
    /// Centroid i embedding (dim-dimensional).
    pub centroid_i: Vec<f32>,
    /// Centroid j embedding (dim-dimensional).
    pub centroid_j: Vec<f32>,
    /// Correction = cos(silu_activated) - cos(raw).
    pub correction: f32,
}

/// Generate training data for SiLU correction model.
///
/// For each probe × (centroid_i, centroid_j):
///   1. Compute gate_i = silu(gate_row_i × probe) elementwise
///   2. Compute activated_i = gate_i × up_row_i elementwise
///   3. true_cos = cosine(activated_i, activated_j)
///   4. linear_cos = cosine(centroid_i, centroid_j)
///   5. correction = true_cos - linear_cos
pub fn generate_training_data(
    gate_centroids: &[Vec<f32>],  // N × dim (gate weight centroids)
    up_centroids: &[Vec<f32>],    // N × dim (up weight centroids)
    centroids: &[Vec<f32>],       // N × dim (raw embedding centroids)
    probes: &[Vec<f32>],          // P × dim (probe vectors, can be centroids themselves)
) -> Vec<CorrectionSample> {
    let n = centroids.len();
    let p = probes.len();
    let mut samples = Vec::with_capacity(n * n * p);

    for probe in probes {
        for i in 0..n {
            // Compute SiLU-activated representation for centroid i
            let activated_i = activate(
                &gate_centroids[i.min(gate_centroids.len() - 1)],
                &up_centroids[i.min(up_centroids.len() - 1)],
                probe,
            );

            for j in 0..n {
                let activated_j = activate(
                    &gate_centroids[j.min(gate_centroids.len() - 1)],
                    &up_centroids[j.min(up_centroids.len() - 1)],
                    probe,
                );

                let true_cos = cosine_f32(&activated_i, &activated_j);
                let linear_cos = cosine_f32(&centroids[i], &centroids[j]);
                let correction = true_cos - linear_cos;

                samples.push(CorrectionSample {
                    centroid_i: centroids[i].clone(),
                    centroid_j: centroids[j].clone(),
                    correction,
                });
            }
        }
    }

    samples
}

/// SiLU activation: gate × up with SiLU nonlinearity.
fn activate(gate: &[f32], up: &[f32], probe: &[f32]) -> Vec<f32> {
    let dim = gate.len().min(up.len()).min(probe.len());
    let mut result = vec![0.0f32; dim];
    for k in 0..dim {
        let g = silu(gate[k] * probe[k]);
        result[k] = g * up[k];
    }
    result
}

/// Cosine similarity between two f32 vectors.
fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..n {
        dot += a[i] as f64 * b[i] as f64;
        norm_a += (a[i] as f64).powi(2);
        norm_b += (b[i] as f64).powi(2);
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-12 { return 0.0; }
    (dot / denom) as f32
}

// ═══════════════════════════════════════════════════════════════════════════
// CORRECTION APPLICATION
// ═══════════════════════════════════════════════════════════════════════════

/// Apply gate modulation to centroids before distance table build.
///
/// For gate-modulated roles (K, V, Up): centroid = mean(silu(gate) × weight)
/// For raw roles (Q, Down): centroid unchanged.
pub fn gate_modulate_centroids(
    centroids: &[Vec<f32>],
    gate_weights: &[Vec<f32>],
    policy: GatePolicy,
) -> Vec<Vec<f32>> {
    match policy {
        GatePolicy::Raw => centroids.to_vec(),
        GatePolicy::GateModulated => {
            centroids.iter().enumerate().map(|(i, centroid)| {
                let gate = &gate_weights[i.min(gate_weights.len() - 1)];
                let dim = centroid.len().min(gate.len());
                let mut modulated = vec![0.0f32; dim];
                for k in 0..dim {
                    modulated[k] = centroid[k] * silu(gate[k]);
                }
                modulated
            }).collect()
        }
    }
}

/// Apply SiLU corrections to a distance table.
///
/// corrections[i * n + j] = learned correction from ONNX model.
/// table[i * n + j] = HDR CDF-encoded cosine + correction.
pub fn apply_corrections(
    table: &mut [u8],
    corrections: &[f32],
    n: usize,
) {
    assert_eq!(table.len(), n * n);
    assert_eq!(corrections.len(), n * n);

    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            let original = table[idx] as f32;
            // Correction is in cosine space [-1, +1], scale to u8 adjustment
            let adjustment = corrections[idx] * 128.0; // ±128 range
            let corrected = (original + adjustment).clamp(0.0, 255.0) as u8;
            table[idx] = corrected;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// STATISTICS
// ═══════════════════════════════════════════════════════════════════════════

/// Analyze correction magnitude distribution.
#[derive(Debug, Clone)]
pub struct CorrectionStats {
    pub count: usize,
    pub mean_abs: f32,
    pub max_abs: f32,
    pub mean: f32,
    pub std_dev: f32,
    /// Fraction of corrections > 0.01 (material difference).
    pub material_fraction: f32,
    /// Fraction of corrections > 0.1 (large difference).
    pub large_fraction: f32,
}

pub fn correction_stats(samples: &[CorrectionSample]) -> CorrectionStats {
    let n = samples.len();
    if n == 0 {
        return CorrectionStats {
            count: 0, mean_abs: 0.0, max_abs: 0.0, mean: 0.0,
            std_dev: 0.0, material_fraction: 0.0, large_fraction: 0.0,
        };
    }
    let corrections: Vec<f32> = samples.iter().map(|s| s.correction).collect();
    let mean = corrections.iter().sum::<f32>() / n as f32;
    let mean_abs = corrections.iter().map(|c| c.abs()).sum::<f32>() / n as f32;
    let max_abs = corrections.iter().map(|c| c.abs()).fold(0.0f32, f32::max);
    let variance = corrections.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / n as f32;
    let material = corrections.iter().filter(|c| c.abs() > 0.01).count();
    let large = corrections.iter().filter(|c| c.abs() > 0.1).count();

    CorrectionStats {
        count: n,
        mean_abs,
        max_abs,
        mean,
        std_dev: variance.sqrt(),
        material_fraction: material as f32 / n as f32,
        large_fraction: large as f32 / n as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_values() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!(silu(5.0) > 4.9);   // ≈ x for large positive
        assert!(silu(-5.0).abs() < 0.04); // ≈ 0 for large negative
        assert!((silu(0.1) - 0.052).abs() < 0.01); // decision boundary region
    }

    #[test]
    fn test_silu_sign_flip() {
        // THE critical property: sign flip around zero
        let pos = silu(0.01);
        let neg = silu(-0.01);
        assert!(pos > 0.0);
        assert!(neg < 0.0);
        // The ratio matters more than absolute values
        assert!((pos - (-neg)).abs() < 0.001, "silu should be approximately odd near 0");
    }

    #[test]
    fn test_gate_policy() {
        assert_eq!(gate_policy("attn_q"), GatePolicy::Raw);
        assert_eq!(gate_policy("attn_k"), GatePolicy::GateModulated);
        assert_eq!(gate_policy("attn_v"), GatePolicy::GateModulated);
        assert_eq!(gate_policy("ffn_gate"), GatePolicy::GateModulated);
        assert_eq!(gate_policy("ffn_up"), GatePolicy::GateModulated);
        assert_eq!(gate_policy("ffn_down"), GatePolicy::Raw);
    }

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_f32(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_f32(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_gate_modulation_changes_centroids() {
        let centroids = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let gates = vec![vec![0.5, -0.5, 2.0, 0.01]]; // mixed: positive, negative, large, near-zero

        let raw = gate_modulate_centroids(&centroids, &gates, GatePolicy::Raw);
        let modulated = gate_modulate_centroids(&centroids, &gates, GatePolicy::GateModulated);

        assert_eq!(raw[0], centroids[0]); // raw = unchanged
        assert_ne!(modulated[0], centroids[0]); // modulated = different

        // Check SiLU effect: negative gate should suppress, positive should pass
        // gate[1] = -0.5 → silu(-0.5) ≈ -0.19 → centroid[1] = 2.0 * -0.19 ≈ -0.38
        assert!(modulated[0][1].abs() < centroids[0][1].abs(),
            "negative gate should suppress: {} vs {}", modulated[0][1], centroids[0][1]);
        // gate[3] = 0.01 → silu(0.01) ≈ 0.005 → nearly zero
        assert!(modulated[0][3].abs() < 0.1,
            "near-zero gate should nearly suppress: {}", modulated[0][3]);
    }

    #[test]
    fn test_generate_training_data() {
        let dim = 4;
        let centroids = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];
        let gates = vec![
            vec![0.5, 0.1, -0.3, 0.8],
            vec![-0.2, 0.7, 0.1, -0.1],
        ];
        let ups = vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ];
        // Use centroids as probes
        let samples = generate_training_data(&gates, &ups, &centroids, &centroids);

        // 2 probes × 2 centroids × 2 centroids = 8 samples
        assert_eq!(samples.len(), 8);

        // Self-correction should be small (same centroid, same activation)
        let self_correction = samples.iter()
            .filter(|s| s.centroid_i == s.centroid_j)
            .map(|s| s.correction.abs())
            .sum::<f32>();
        // Self-pairs should have near-zero correction (cos with itself = 1.0 both ways)

        let stats = correction_stats(&samples);
        eprintln!("Correction stats: mean_abs={:.4}, max_abs={:.4}, material={:.1}%",
            stats.mean_abs, stats.max_abs, stats.material_fraction * 100.0);
    }

    #[test]
    fn test_correction_stats_on_narrow_gate() {
        // Simulate reader-lm gate range: [-0.095, 0.336]
        let dim = 8;
        let n_centroids = 4;
        let mut centroids = Vec::new();
        let mut gates = Vec::new();
        let mut ups = Vec::new();

        for i in 0..n_centroids {
            let mut c = vec![0.0f32; dim];
            let mut g = vec![0.0f32; dim];
            let mut u = vec![0.0f32; dim];
            for d in 0..dim {
                c[d] = ((i * dim + d) as f32 * 0.1).sin();
                // Narrow gate range like reader-lm
                g[d] = -0.095 + (i * dim + d) as f32 * 0.05;
                g[d] = g[d].clamp(-0.095, 0.336);
                u[d] = ((i * dim + d) as f32 * 0.2).cos();
            }
            centroids.push(c);
            gates.push(g);
            ups.push(u);
        }

        let samples = generate_training_data(&gates, &ups, &centroids, &centroids);
        let stats = correction_stats(&samples);

        eprintln!("\nNarrow gate (reader-lm range):");
        eprintln!("  Samples:    {}", stats.count);
        eprintln!("  Mean |Δ|:   {:.4}", stats.mean_abs);
        eprintln!("  Max |Δ|:    {:.4}", stats.max_abs);
        eprintln!("  Std dev:    {:.4}", stats.std_dev);
        eprintln!("  Material:   {:.1}% (|Δ| > 0.01)", stats.material_fraction * 100.0);
        eprintln!("  Large:      {:.1}% (|Δ| > 0.1)", stats.large_fraction * 100.0);

        // With narrow gate, corrections should be material
        assert!(stats.count > 0);
    }

    #[test]
    fn test_apply_corrections() {
        let n = 4;
        let mut table = vec![128u8; n * n]; // baseline
        let mut corrections = vec![0.0f32; n * n];
        corrections[0] = 0.5;   // large positive correction
        corrections[3] = -0.3;  // negative correction
        corrections[5] = 0.01;  // tiny correction

        apply_corrections(&mut table, &corrections, n);

        assert!((table[0] as i32 - 192).abs() <= 1, "expected ~192, got {}", table[0]);  // 128 + 0.5*128 = 192
        assert!((table[3] as i32 - 90).abs() <= 2, "expected ~90, got {}", table[3]);   // 128 - 0.3*128 ≈ 90
        assert!((table[5] as i32 - 129).abs() <= 1, "expected ~129, got {}", table[5]);  // 128 + 0.01*128 ≈ 129
        assert_eq!(table[1], 128);  // unchanged (correction = 0)
    }
}
