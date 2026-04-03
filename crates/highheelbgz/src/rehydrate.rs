//! Rehydration without source file: spiral address + γ+φ curve prediction.
//!
//! Store K anchor BF16 samples per dim. The γ+φ curve interpolates between
//! anchors to predict the full SPD=32 resolution. No source file needed.
//!
//! Storage: address (6 bytes) + K anchors × 17 dims × 2 bytes BF16
//!   K=4:  6 + 136 = 142 bytes → target Pearson via interpolation
//!   K=8:  6 + 272 = 278 bytes → higher fidelity
//!   K=16: 6 + 544 = 550 bytes → near-SPD=32

use crate::{SpiralAddress, SpiralWalk, BASE_DIM};
use std::f64::consts::GOLDEN_RATIO;

/// A self-contained spiral encoding: address + anchor values.
/// Can rehydrate WITHOUT the source GGUF file.
#[derive(Clone, Debug)]
pub struct SpiralEncoding {
    /// The address: where on the curve this vector sits.
    pub addr: SpiralAddress,
    /// Anchor BF16 values: [dim][anchor_index].
    /// K anchors per dim, stored as u16 BF16 bit patterns.
    pub anchors: Vec<Vec<u16>>,
    /// Number of anchors per dim.
    pub k: usize,
}

/// Gamma profile for curve prediction (shared per model, 28 bytes).
#[derive(Clone, Debug)]
pub struct GammaProfile {
    /// Per-role gamma: [Q, K, V, Gate, Up, Down].
    pub role_gamma: [f32; 6],
    /// Global φ-scale.
    pub phi_scale: f32,
}

impl GammaProfile {
    pub const BYTE_SIZE: usize = 28;

    /// Calibrate from raw f32 weight rows.
    pub fn calibrate(rows: &[&[f32]]) -> Self {
        if rows.is_empty() {
            return GammaProfile { role_gamma: [0.01; 6], phi_scale: 0.01 };
        }
        let mut total_mag = 0.0f64;
        let mut count = 0u64;
        for row in rows {
            for &v in *row {
                total_mag += v.abs() as f64;
                count += 1;
            }
        }
        let mean_mag = (total_mag / count.max(1) as f64) as f32;
        GammaProfile {
            role_gamma: [mean_mag; 6], // same for all roles if not separated
            phi_scale: mean_mag.max(1e-6),
        }
    }
}

impl SpiralEncoding {
    /// Byte size of this encoding.
    pub fn byte_size(&self) -> usize {
        SpiralAddress::BYTE_SIZE_U16 + self.k * BASE_DIM * 2
    }

    /// Encode from raw f32 vector: sample K anchors via spiral walk.
    pub fn encode(weights: &[f32], start: u32, stride: u32, k: usize) -> Self {
        let addr = SpiralAddress::new(start, stride, k as u32);
        let walk = SpiralWalk::execute(&addr, weights);

        let mut anchors = Vec::with_capacity(BASE_DIM);
        for d in 0..BASE_DIM {
            let bf16_vals: Vec<u16> = walk.samples[d].iter()
                .map(|&v| f32_to_bf16(v))
                .collect();
            anchors.push(bf16_vals);
        }

        SpiralEncoding { addr, anchors, k }
    }

    /// Rehydrate to f32 values — NO source file needed.
    ///
    /// Returns the anchor values as f32, suitable for cosine comparison.
    /// This gives K×17 values. For K=4: 68 f32 values. For K=32: 544 f32 values.
    pub fn rehydrate(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.k * BASE_DIM);
        for d in 0..BASE_DIM {
            for s in 0..self.anchors[d].len() {
                result.push(bf16_to_f32(self.anchors[d][s]));
            }
        }
        result
    }

    /// Rehydrate with γ+φ interpolation: predict intermediate values
    /// between anchors using the curve shape.
    ///
    /// Expands K anchors to target_spd samples per dim via γ-weighted
    /// linear interpolation along the spiral curve.
    pub fn rehydrate_interpolated(&self, target_spd: usize, gamma: &GammaProfile) -> Vec<f32> {
        let mut result = Vec::with_capacity(target_spd * BASE_DIM);

        for d in 0..BASE_DIM {
            let n_anchors = self.anchors[d].len();
            if n_anchors == 0 {
                result.extend(std::iter::repeat(0.0f32).take(target_spd));
                continue;
            }

            let anchor_f32: Vec<f32> = self.anchors[d].iter()
                .map(|&b| bf16_to_f32(b))
                .collect();

            if n_anchors >= target_spd {
                // More anchors than needed — subsample
                for s in 0..target_spd {
                    let idx = s * n_anchors / target_spd;
                    result.push(anchor_f32[idx]);
                }
            } else {
                // Fewer anchors — interpolate
                // φ-weighted interpolation: position between anchors follows
                // golden ratio fractional spacing (not uniform)
                for s in 0..target_spd {
                    let t = s as f64 / target_spd as f64 * n_anchors as f64;
                    let lo = (t.floor() as usize).min(n_anchors - 1);
                    let hi = (lo + 1).min(n_anchors - 1);
                    let frac = t - lo as f64;

                    // φ-weighted: bias interpolation toward the anchor that's
                    // closer in spiral space (not linear space)
                    let phi_frac = frac.powf(1.0 / GOLDEN_RATIO);

                    let val = anchor_f32[lo] as f64 * (1.0 - phi_frac)
                            + anchor_f32[hi] as f64 * phi_frac;
                    result.push(val as f32);
                }
            }
        }
        result
    }

    /// Cosine similarity between two encodings (direct, no interpolation).
    pub fn cosine(&self, other: &SpiralEncoding) -> f64 {
        let a = self.rehydrate();
        let b = other.rehydrate();
        cosine_f32(&a, &b)
    }

    /// Cosine similarity with interpolation to target SPD.
    pub fn cosine_interpolated(&self, other: &SpiralEncoding, target_spd: usize, gamma: &GammaProfile) -> f64 {
        let a = self.rehydrate_interpolated(target_spd, gamma);
        let b = other.rehydrate_interpolated(target_spd, gamma);
        cosine_f32(&a, &b)
    }
}

fn f32_to_bf16(v: f32) -> u16 { (v.to_bits() >> 16) as u16 }
fn bf16_to_f32(bits: u16) -> f32 { f32::from_bits((bits as u32) << 16) }

fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    for i in 0..n { dot += a[i] as f64 * b[i] as f64; na += (a[i] as f64).powi(2); nb += (b[i] as f64).powi(2); }
    let d = (na * nb).sqrt(); if d < 1e-12 { 0.0 } else { dot / d }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tokenizer prototype: text → spiral address into vocab embedding
// ═══════════════════════════════════════════════════════════════════════════

/// Prototype spiral tokenizer: each token is a SpiralEncoding into vocab space.
///
/// Instead of integer token IDs, each token is represented by its spiral
/// encoding of the corresponding embedding vector. Similarity between tokens
/// is spiral walk cosine — no lookup table needed.
#[derive(Clone, Debug)]
pub struct SpiralTokenizer {
    /// Vocabulary: each entry is a spiral-encoded embedding vector.
    pub vocab: Vec<SpiralEncoding>,
    /// Token strings (parallel to vocab).
    pub tokens: Vec<String>,
    /// Gamma profile for this vocabulary.
    pub gamma: GammaProfile,
    /// Spiral parameters used for encoding.
    pub start: u32,
    pub stride: u32,
    pub k: usize,
}

impl SpiralTokenizer {
    /// Build from raw vocabulary embeddings.
    ///
    /// Each vocab entry = (token_string, f32_embedding_vector).
    pub fn build(
        vocab: &[(&str, Vec<f32>)],
        start: u32,
        stride: u32,
        k: usize,
    ) -> Self {
        let refs: Vec<&[f32]> = vocab.iter().map(|(_, v)| v.as_slice()).collect();
        let gamma = GammaProfile::calibrate(&refs);

        let encoded: Vec<SpiralEncoding> = vocab.iter()
            .map(|(_, v)| SpiralEncoding::encode(v, start, stride, k))
            .collect();
        let tokens: Vec<String> = vocab.iter().map(|(t, _)| t.to_string()).collect();

        SpiralTokenizer { vocab: encoded, tokens, gamma, start, stride, k }
    }

    /// Tokenize: find nearest vocab entry by spiral cosine.
    ///
    /// Input: f32 embedding of a text segment.
    /// Output: (token_index, token_string, cosine_similarity).
    pub fn nearest(&self, query_embedding: &[f32]) -> (usize, &str, f64) {
        let query_enc = SpiralEncoding::encode(query_embedding, self.start, self.stride, self.k);

        let mut best_idx = 0;
        let mut best_cos = f64::NEG_INFINITY;

        for (i, entry) in self.vocab.iter().enumerate() {
            let cos = query_enc.cosine(entry);
            if cos > best_cos {
                best_cos = cos;
                best_idx = i;
            }
        }

        (best_idx, &self.tokens[best_idx], best_cos)
    }

    /// Top-K nearest tokens.
    pub fn nearest_k(&self, query_embedding: &[f32], k: usize) -> Vec<(usize, &str, f64)> {
        let query_enc = SpiralEncoding::encode(query_embedding, self.start, self.stride, self.k);

        let mut scored: Vec<(usize, f64)> = self.vocab.iter()
            .enumerate()
            .map(|(i, entry)| (i, query_enc.cosine(entry)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter()
            .take(k)
            .map(|(i, cos)| (i, self.tokens[i].as_str(), cos))
            .collect()
    }

    /// Storage budget.
    pub fn byte_size(&self) -> usize {
        self.vocab.iter().map(|e| e.byte_size()).sum::<usize>()
            + GammaProfile::BYTE_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(seed: usize, dim: usize) -> Vec<f32> {
        (0..dim).map(|d| ((d * 97 + seed * 31) as f32 % 200.0 - 100.0) * 0.01).collect()
    }

    #[test]
    fn encode_rehydrate_nonzero() {
        let v = make_embedding(42, 1024);
        let enc = SpiralEncoding::encode(&v, 20, 8, 4);
        assert_eq!(enc.k, 4);
        assert_eq!(enc.byte_size(), 6 + 4 * 17 * 2); // 142 bytes

        let rehydrated = enc.rehydrate();
        let mag: f64 = rehydrated.iter().map(|v| v.abs() as f64).sum();
        assert!(mag > 0.0, "rehydrated should be nonzero");
    }

    #[test]
    fn self_cosine_one() {
        let v = make_embedding(7, 1024);
        let enc = SpiralEncoding::encode(&v, 20, 8, 4);
        let c = enc.cosine(&enc);
        assert!((c - 1.0).abs() < 1e-10);
    }

    #[test]
    fn different_vectors_different_cosine() {
        let a = SpiralEncoding::encode(&make_embedding(1, 1024), 20, 8, 4);
        let b = SpiralEncoding::encode(&make_embedding(999, 1024), 20, 8, 4);
        let c = a.cosine(&b);
        assert!(c < 0.99, "different vectors should have cosine < 1: {}", c);
    }

    #[test]
    fn interpolation_more_values() {
        let v = make_embedding(42, 1024);
        let enc = SpiralEncoding::encode(&v, 20, 8, 4);
        let gamma = GammaProfile { role_gamma: [0.5; 6], phi_scale: 0.5 };

        let direct = enc.rehydrate();
        let interp = enc.rehydrate_interpolated(32, &gamma);

        // Direct has up to 4 × 17 values (some dims may have fewer if walk goes OOB)
        assert!(direct.len() <= 4 * 17);
        assert!(direct.len() > 0);
        assert_eq!(interp.len(), 32 * 17); // 544 — interpolation always produces target_spd

        // Interpolated should be nonzero
        let mag: f64 = interp.iter().map(|v| v.abs() as f64).sum();
        assert!(mag > 0.0);
    }

    #[test]
    fn tokenizer_finds_nearest() {
        let vocab: Vec<(&str, Vec<f32>)> = vec![
            ("cat", make_embedding(1, 512)),
            ("dog", make_embedding(2, 512)),
            ("fish", make_embedding(3, 512)),
            ("bird", make_embedding(4, 512)),
        ];

        let tok = SpiralTokenizer::build(&vocab, 20, 8, 4);
        assert_eq!(tok.vocab.len(), 4);

        // Query with same embedding as "cat" should find "cat"
        let (idx, token, cos) = tok.nearest(&make_embedding(1, 512));
        assert_eq!(token, "cat");
        assert!((cos - 1.0).abs() < 1e-6, "exact match should be 1.0: {}", cos);
    }

    #[test]
    fn tokenizer_top_k() {
        let vocab: Vec<(&str, Vec<f32>)> = (0..20)
            .map(|i| (["hello", "world", "foo", "bar", "baz",
                       "cat", "dog", "fish", "bird", "tree",
                       "sky", "sun", "moon", "star", "rain",
                       "wind", "sea", "hill", "rock", "leaf"][i],
                make_embedding(i, 256)))
            .collect();

        let tok = SpiralTokenizer::build(&vocab, 20, 8, 4);
        let results = tok.nearest_k(&make_embedding(0, 256), 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].1, "hello"); // exact match
        assert!(results[0].2 >= results[1].2); // sorted by cosine (may tie)
    }

    #[test]
    fn tokenizer_byte_budget() {
        let vocab: Vec<(&str, Vec<f32>)> = (0..1000)
            .map(|i| ("tok", make_embedding(i, 512)))
            .collect();

        let tok = SpiralTokenizer::build(&vocab, 20, 8, 4);
        let bytes = tok.byte_size();
        // 1000 tokens × 142 bytes + 28 gamma = 142,028 bytes ≈ 139 KB
        assert!(bytes < 200_000, "1000-token vocab should be < 200 KB: {} bytes", bytes);
        eprintln!("1000-token vocab: {} bytes ({:.1} KB)", bytes, bytes as f64 / 1024.0);
    }
}
