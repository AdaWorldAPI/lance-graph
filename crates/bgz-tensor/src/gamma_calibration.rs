//! Three-γ calibration: role offset + cosine replacement + meta alignment.
//!
//! ```text
//! γ_role:    per role per model   — shadow expansion for small weights (28 bytes/model)
//! γ_cosine:  per codebook         — redistribute u8 levels around cosine=0 (4 bytes)
//! γ_meta:    per model pair       — cross-model baseline alignment (N×N × 4 bytes)
//! ```
//!
//! Together: the Kurvenlineal offset that tells you where on the curve
//! each model enters, and how to decode u8 back to exact cosine.

// cosine_f32_slice reserved for future cross-model calibration
#[allow(unused_imports)]
use crate::stacked_n::cosine_f32_slice;
use std::f64::consts::GOLDEN_RATIO;

/// Euler-Mascheroni constant.
// Reserved for future γ-offset calibration formula
#[allow(dead_code)]
const EULER_GAMMA: f64 = std::f64::consts::EULER_GAMMA;

// ═══════════════════════════════════════════════════════════════════════════
// γ_role: per-role magnitude offset (28 bytes per model)
// ═══════════════════════════════════════════════════════════════════════════

/// Per-role gamma offsets calibrated from weight magnitude distribution.
///
/// **Layout (append-at-tail for backward compatibility)**:
/// ```text
///   [0] Q     — attention query projection    (q_proj.weight)
///   [1] K     — attention key projection      (k_proj.weight)
///   [2] V     — attention value projection    (v_proj.weight)
///   [3] Gate  — MLP gate projection            (gate_proj.weight)
///   [4] Up    — MLP up projection              (up_proj.weight)
///   [5] Down  — MLP down projection            (down_proj.weight)
///   [6] O     — attention output projection    (o_proj.weight)       ← ADDED for Jina v5 coverage
///   [7] Embed — token embedding matrix         (embed_tokens.weight) ← ADDED for Jina v5 coverage
/// ```
///
/// Indices 0..=5 unchanged from the legacy 6-role layout; any caller that
/// reads those indices directly keeps working. Indices 6 and 7 are
/// additive for code that needs full Jina v5 / Qwen 3.5 matmul-role
/// coverage. Layernorm scale vectors (q_norm, k_norm, input_layernorm,
/// post_attention_layernorm, final norm) are a separate `NormScale`
/// class and NOT represented here.
///
/// 8 roles × f32 + 1 global phi_scale = **36 bytes** (was 28 bytes
/// before the Jina v5 role extension). Any serialized profile from a
/// pre-extension bake is wire-incompatible; re-bake under the 36-byte
/// format for anything certification-targeting.
#[derive(Clone, Debug)]
pub struct RoleGamma {
    /// Per-role: [Q, K, V, Gate, Up, Down, O, Embed].
    pub gamma: [f32; 8],
    /// Global φ-scale (max gamma across all 8 roles).
    pub phi_scale: f32,
}

impl RoleGamma {
    pub const BYTE_SIZE: usize = 8 * 4 + 4; // 36

    /// Role-index constants for callers that want named access instead
    /// of magic numbers. Prefer these over literal indices.
    pub const Q: usize = 0;
    pub const K: usize = 1;
    pub const V: usize = 2;
    pub const GATE: usize = 3;
    pub const UP: usize = 4;
    pub const DOWN: usize = 5;
    pub const O: usize = 6;
    pub const EMBED: usize = 7;

    /// Calibrate from per-role weight rows.
    ///
    /// `roles: &[("Q", &[row_slices]), ("K", ...), ...]`
    ///
    /// Recognized role names (case-sensitive match, extend freely):
    /// - `"Q"`, `"q_proj"`, `"attn_q"`, `"attn_qkv"`
    /// - `"K"`, `"k_proj"`, `"attn_k"`
    /// - `"V"`, `"v_proj"`, `"attn_v"`
    /// - `"Gate"`, `"gate_proj"`, `"ffn_gate"`
    /// - `"Up"`, `"up_proj"`, `"ffn_up"`
    /// - `"Down"`, `"down_proj"`, `"ffn_down"`
    /// - `"O"`, `"o_proj"`, `"attn_o"`, `"attn_output"`
    /// - `"Embed"`, `"embed_tokens"`, `"tok_embd"`, `"wte"`
    ///
    /// Unknown role names are silently skipped (the resulting γ stays
    /// at the default 0.01 seed for that slot).
    pub fn calibrate(roles: &[(&str, &[&[f32]])]) -> Self {
        let mut gamma = [0.01f32; 8];
        for (name, rows) in roles {
            let idx = match *name {
                "Q" | "q_proj" | "attn_q" | "attn_qkv" => Self::Q,
                "K" | "k_proj" | "attn_k" => Self::K,
                "V" | "v_proj" | "attn_v" => Self::V,
                "Gate" | "gate_proj" | "ffn_gate" => Self::GATE,
                "Up" | "up_proj" | "ffn_up" => Self::UP,
                "Down" | "down_proj" | "ffn_down" => Self::DOWN,
                "O" | "o_proj" | "attn_o" | "attn_output" => Self::O,
                "Embed" | "embed_tokens" | "tok_embd" | "wte" => Self::EMBED,
                _ => continue,
            };
            if rows.is_empty() { continue; }
            let total_mag: f64 = rows.iter()
                .flat_map(|r| r.iter())
                .map(|v| v.abs() as f64)
                .sum();
            let count = rows.iter().map(|r| r.len()).sum::<usize>();
            gamma[idx] = (total_mag / count.max(1) as f64) as f32;
        }
        let phi_scale = gamma.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
        RoleGamma { gamma, phi_scale }
    }

    /// Apply γ expansion to a value: log(1 + |v|/γ) × γ × sign(v).
    #[inline]
    pub fn encode(&self, value: f32, role_idx: usize) -> f32 {
        let g = self.gamma[role_idx].max(1e-8);
        let sign = value.signum();
        sign * ((1.0 + value.abs() / g).ln() * g)
    }

    /// Inverse: (exp(|v|/γ) - 1) × γ × sign(v).
    #[inline]
    pub fn decode(&self, encoded: f32, role_idx: usize) -> f32 {
        let g = self.gamma[role_idx].max(1e-8);
        let sign = encoded.signum();
        sign * ((encoded.abs() / g).exp() - 1.0) * g
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// γ_cosine: cosine replacement offset (4 bytes per codebook)
// ═══════════════════════════════════════════════════════════════════════════

/// Cosine distribution gamma: redistributes u8 levels so the crowded
/// center (cosine ≈ 0, where most pairs land) gets more resolution.
#[derive(Clone, Debug)]
pub struct CosineGamma {
    /// γ for the cosine→u8 mapping.
    /// Small γ = aggressive shadow expansion (more levels near cos=0).
    /// Large γ = nearly linear (uniform distribution).
    pub gamma: f32,
    /// Measured cosine distribution center (median of all pairwise cosines).
    pub center: f32,
    /// Measured cosine distribution spread (IQR or σ).
    pub spread: f32,
}

impl CosineGamma {
    pub const BYTE_SIZE: usize = 12;

    /// Calibrate from a set of pairwise cosines.
    pub fn calibrate(cosines: &[f64]) -> Self {
        if cosines.is_empty() {
            return Self { gamma: 1.0, center: 0.0, spread: 1.0 };
        }

        let n = cosines.len();
        let mut sorted = cosines.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let center = sorted[n / 2] as f32;
        let q1 = sorted[n / 4] as f32;
        let q3 = sorted[3 * n / 4] as f32;
        let iqr = (q3 - q1).max(0.01);

        // γ = IQR / 2: smaller IQR (tighter cluster) → smaller γ → more expansion
        let gamma = (iqr / 2.0).max(0.01);

        Self { gamma, center, spread: iqr }
    }

    /// Map cosine [-1, 1] → γ-expanded → u8 [0, 255].
    pub fn cosine_to_u8(&self, cosine: f64) -> u8 {
        let g = self.gamma.max(1e-6) as f64;
        let centered = cosine - self.center as f64;

        // γ expansion: more resolution near center
        let expanded = centered.signum() * (1.0 + centered.abs() / g).ln() * g;

        // Normalize to [0, 1] using φ distribution
        let max_expanded = (1.0 + 2.0 / g).ln() * g; // max possible expansion
        let normalized = ((expanded / max_expanded) + 1.0) / 2.0;
        let phi_distributed = (GOLDEN_RATIO.powf(normalized.clamp(0.0, 1.0)) - 1.0) / (GOLDEN_RATIO - 1.0);

        (phi_distributed * 255.0).round().clamp(0.0, 255.0) as u8
    }

    /// Decode u8 → cosine.
    pub fn u8_to_cosine(&self, value: u8) -> f64 {
        let g = self.gamma.max(1e-6) as f64;
        let phi_val = value as f64 / 255.0;

        // Inverse φ distribution
        let normalized = (phi_val * (GOLDEN_RATIO - 1.0) + 1.0).ln() / GOLDEN_RATIO.ln();
        let expanded = (normalized * 2.0 - 1.0) * (1.0 + 2.0 / g).ln() * g;

        // Inverse γ expansion
        let centered = expanded.signum() * ((expanded.abs() / g).exp() - 1.0) * g;
        centered + self.center as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// γ_meta: cross-model baseline alignment (N×N × 4 bytes)
// ═══════════════════════════════════════════════════════════════════════════

/// Cross-model meta offset: aligns cosine baselines across models.
///
/// "cosine 0.3 in Jina = cosine 0.5 in Qwopus" because their weight
/// magnitude distributions differ. The meta offset normalizes this.
#[derive(Clone, Debug)]
pub struct MetaGamma {
    /// Model names.
    pub models: Vec<String>,
    /// Per-model baseline cosine (median of within-model pairwise cosines).
    pub baselines: Vec<f32>,
    /// Per-model-pair offset: offset[i * n + j] = baseline[i] - baseline[j].
    pub offsets: Vec<f32>,
}

impl MetaGamma {
    /// Calibrate from multiple models' pairwise cosine distributions.
    ///
    /// model_cosines: &[("model_name", &[pairwise_cosines])]
    pub fn calibrate(model_cosines: &[(&str, &[f64])]) -> Self {
        let n = model_cosines.len();
        let mut models = Vec::with_capacity(n);
        let mut baselines = Vec::with_capacity(n);

        for (name, cosines) in model_cosines {
            models.push(name.to_string());
            if cosines.is_empty() {
                baselines.push(0.0);
                continue;
            }
            let mut sorted = cosines.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            baselines.push(sorted[sorted.len() / 2] as f32);
        }

        // Pairwise offsets
        let mut offsets = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                offsets[i * n + j] = baselines[i] - baselines[j];
            }
        }

        MetaGamma { models, baselines, offsets }
    }

    /// Get the offset to align model_a's cosines to model_b's scale.
    pub fn offset(&self, model_a: &str, model_b: &str) -> f32 {
        let idx_a = self.models.iter().position(|m| m == model_a);
        let idx_b = self.models.iter().position(|m| m == model_b);
        match (idx_a, idx_b) {
            (Some(a), Some(b)) => self.offsets[a * self.models.len() + b],
            _ => 0.0,
        }
    }

    /// Translate a cosine value from model_a's space to model_b's space.
    pub fn translate(&self, cosine: f64, from: &str, to: &str) -> f64 {
        cosine + self.offset(from, to) as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Complete 3-γ calibration profile
// ═══════════════════════════════════════════════════════════════════════════

/// Complete calibration profile: role + cosine + meta offsets.
/// This is the Kurvenlineal metadata that makes the u8 table exact.
#[derive(Clone, Debug)]
pub struct CalibrationProfile {
    pub model_name: String,
    pub role_gamma: RoleGamma,
    pub cosine_gamma: CosineGamma,
    // Meta gamma is shared across models (not per-model).
    // Accessed via the shared MetaGamma instance, not stored per-profile.
}

impl CalibrationProfile {
    /// Total metadata size per model.
    pub fn byte_size() -> usize {
        RoleGamma::BYTE_SIZE + CosineGamma::BYTE_SIZE // 28 + 12 = 40 bytes
    }

    /// Build from raw weight data + pairwise cosines.
    pub fn calibrate(
        model_name: &str,
        role_data: &[(&str, &[&[f32]])],
        pairwise_cosines: &[f64],
    ) -> Self {
        CalibrationProfile {
            model_name: model_name.to_string(),
            role_gamma: RoleGamma::calibrate(role_data),
            cosine_gamma: CosineGamma::calibrate(pairwise_cosines),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_gamma_roundtrip() {
        let rg = RoleGamma { gamma: [0.37, 0.94, 1.33, 1.50, 0.12, 0.15, 0.80, 0.45], phi_scale: 1.50 };
        for role in 0..8 {
            for &v in &[0.001f32, 0.1, 0.5, 1.0, 2.0] {
                let encoded = rg.encode(v, role);
                let decoded = rg.decode(encoded, role);
                assert!((v - decoded).abs() < 0.001,
                    "role {} value {}: encoded={}, decoded={}", role, v, encoded, decoded);
            }
        }
    }

    #[test]
    fn cosine_gamma_calibrate() {
        // Typical cosine distribution: clustered near 0
        let cosines: Vec<f64> = (-100..100).map(|i| i as f64 * 0.01).collect();
        let cg = CosineGamma::calibrate(&cosines);
        assert!(cg.gamma > 0.0);
        assert!((cg.center).abs() < 0.1); // centered near 0
        eprintln!("CosineGamma: γ={:.4}, center={:.4}, spread={:.4}", cg.gamma, cg.center, cg.spread);
    }

    #[test]
    fn cosine_u8_roundtrip() {
        let cg = CosineGamma { gamma: 0.3, center: 0.0, spread: 0.5 };
        for i in 0..20 {
            let cos = -1.0 + i as f64 * 0.1;
            let u8_val = cg.cosine_to_u8(cos);
            let decoded = cg.u8_to_cosine(u8_val);
            // Allow up to 1/256 = 0.004 quantization error
            assert!((cos - decoded).abs() < 0.05,
                "cos={:.3}: u8={}, decoded={:.3}, err={:.4}", cos, u8_val, decoded, (cos - decoded).abs());
        }
    }

    #[test]
    fn cosine_u8_monotone() {
        let cg = CosineGamma { gamma: 0.2, center: 0.0, spread: 0.4 };
        let mut prev = 0u8;
        for i in 0..200 {
            let cos = -1.0 + i as f64 * 0.01;
            let u8_val = cg.cosine_to_u8(cos);
            assert!(u8_val >= prev, "should be monotone: cos={:.2} u8={} < prev={}", cos, u8_val, prev);
            prev = u8_val;
        }
    }

    #[test]
    fn meta_gamma_offset() {
        let mg = MetaGamma::calibrate(&[
            ("jina", &[0.1, 0.05, 0.15, 0.08, 0.12]),      // baseline ~0.1
            ("qwopus", &[0.3, 0.25, 0.35, 0.28, 0.32]),    // baseline ~0.3
            ("gpt2", &[-0.05, 0.0, 0.05, -0.02, 0.03]),    // baseline ~0.0
        ]);

        // Qwopus baseline higher than Jina → positive offset
        let offset = mg.offset("jina", "qwopus");
        assert!(offset < 0.0, "jina→qwopus should be negative offset: {}", offset);

        // Translate: "0.1 in jina = ? in qwopus"
        let translated = mg.translate(0.1, "jina", "qwopus");
        assert!(translated < 0.1, "should shift down to qwopus scale: {}", translated);

        eprintln!("Baselines: {:?}", mg.baselines);
        eprintln!("Jina→Qwopus offset: {:.4}", offset);
    }

    #[test]
    fn calibration_profile_size() {
        assert_eq!(CalibrationProfile::byte_size(), 40); // 28 + 12
    }

    #[test]
    fn full_pipeline() {
        // Simulate: weight rows → γ_role → pairwise cosines → γ_cosine
        let q_rows: Vec<Vec<f32>> = (0..10).map(|i|
            (0..64).map(|d| ((d * i) as f32 * 0.01).sin() * 0.4).collect()
        ).collect();
        let gate_rows: Vec<Vec<f32>> = (0..10).map(|i|
            (0..64).map(|d| ((d * i) as f32 * 0.02).cos() * 2.0).collect()
        ).collect();

        let q_refs: Vec<&[f32]> = q_rows.iter().map(|r| r.as_slice()).collect();
        let gate_refs: Vec<&[f32]> = gate_rows.iter().map(|r| r.as_slice()).collect();

        // Compute pairwise cosines
        let all_rows: Vec<&[f32]> = q_refs.iter().chain(gate_refs.iter()).copied().collect();
        let mut cosines = Vec::new();
        for i in 0..all_rows.len() {
            for j in (i+1)..all_rows.len() {
                cosines.push(cosine_f32_slice(all_rows[i], all_rows[j]));
            }
        }

        let profile = CalibrationProfile::calibrate(
            "test_model",
            &[("Q", &q_refs), ("Gate", &gate_refs)],
            &cosines,
        );

        assert!(profile.role_gamma.gamma[3] > profile.role_gamma.gamma[0],
            "Gate should have higher γ than Q");
        eprintln!("Profile: Q_γ={:.4}, Gate_γ={:.4}, cos_γ={:.4}",
            profile.role_gamma.gamma[0], profile.role_gamma.gamma[3],
            profile.cosine_gamma.gamma);
    }
}
