//! Tensor bridge: unified output type across frameworks.
//!
//! EmbedAnything has ModelOutput = Tensor | ndarray::Array.
//! We bridge between:
//!   - candle Tensor (calibration forward pass)
//!   - f32 slices (runtime, distance tables)
//!   - i8 slices (signed tables)
//!
//! The bridge enables: calibration output → runtime input
//! without manual conversion at every boundary.

/// Unified embedding output from any source.
#[derive(Clone, Debug)]
pub enum EmbeddingOutput {
    /// Raw f32 slice (from distance table rows, energy vectors).
    F32(Vec<f32>),
    /// Signed i8 (from signed distance tables).
    I8(Vec<i8>),
    /// Quantized u8 (from HDR CDF-encoded tables).
    U8(Vec<u8>),
    /// Candle tensor (only with calibration feature).
    #[cfg(feature = "calibration")]
    Tensor(candle_core::Tensor),
}

impl EmbeddingOutput {
    /// Convert to f32 regardless of source.
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            EmbeddingOutput::F32(v) => v.clone(),
            EmbeddingOutput::I8(v) => v.iter().map(|&x| x as f32 / 127.0).collect(),
            EmbeddingOutput::U8(v) => v.iter().map(|&x| (x as f32 - 128.0) / 127.0).collect(),
            #[cfg(feature = "calibration")]
            EmbeddingOutput::Tensor(t) => {
                t.flatten_all()
                    .and_then(|t| t.to_vec1::<f32>())
                    .unwrap_or_default()
            }
        }
    }

    /// Convert to i8 (signed quantization).
    pub fn to_i8(&self) -> Vec<i8> {
        match self {
            EmbeddingOutput::I8(v) => v.clone(),
            EmbeddingOutput::F32(v) => v.iter()
                .map(|&x| (x * 127.0).round().clamp(-128.0, 127.0) as i8)
                .collect(),
            EmbeddingOutput::U8(v) => v.iter()
                .map(|&x| (x as i16 - 128) as i8)
                .collect(),
            #[cfg(feature = "calibration")]
            EmbeddingOutput::Tensor(t) => {
                self.to_f32().iter()
                    .map(|&x| (x * 127.0).round().clamp(-128.0, 127.0) as i8)
                    .collect()
            }
        }
    }

    /// Convert to u8 (unsigned, centered at 128).
    pub fn to_u8(&self) -> Vec<u8> {
        match self {
            EmbeddingOutput::U8(v) => v.clone(),
            EmbeddingOutput::F32(v) => v.iter()
                .map(|&x| ((x * 127.0 + 128.0).round().clamp(0.0, 255.0)) as u8)
                .collect(),
            EmbeddingOutput::I8(v) => v.iter()
                .map(|&x| (x as i16 + 128) as u8)
                .collect(),
            #[cfg(feature = "calibration")]
            EmbeddingOutput::Tensor(_) => {
                self.to_f32().iter()
                    .map(|&x| ((x * 127.0 + 128.0).round().clamp(0.0, 255.0)) as u8)
                    .collect()
            }
        }
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        match self {
            EmbeddingOutput::F32(v) => v.len(),
            EmbeddingOutput::I8(v) => v.len(),
            EmbeddingOutput::U8(v) => v.len(),
            #[cfg(feature = "calibration")]
            EmbeddingOutput::Tensor(t) => t.elem_count(),
        }
    }

    /// Cosine similarity between two outputs.
    pub fn cosine(&self, other: &EmbeddingOutput) -> f32 {
        let a = self.to_f32();
        let b = other.to_f32();
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na > 1e-10 && nb > 1e-10 { dot / (na * nb) } else { 0.0 }
    }
}

/// Batch of embeddings from a model forward pass or codebook lookup.
#[derive(Clone, Debug)]
pub struct EmbeddingBatch {
    pub embeddings: Vec<EmbeddingOutput>,
    pub source: String,
}

impl EmbeddingBatch {
    pub fn new(source: &str) -> Self {
        Self { embeddings: Vec::new(), source: source.into() }
    }

    pub fn push(&mut self, emb: EmbeddingOutput) {
        self.embeddings.push(emb);
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Pairwise cosine similarity matrix.
    pub fn pairwise_cosines(&self) -> Vec<Vec<f32>> {
        let n = self.embeddings.len();
        let mut matrix = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n {
                let cos = self.embeddings[i].cosine(&self.embeddings[j]);
                matrix[i][j] = cos;
                matrix[j][i] = cos;
            }
        }
        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_roundtrip() {
        let data = vec![0.5f32, -0.3, 0.0, 1.0, -1.0];
        let emb = EmbeddingOutput::F32(data.clone());
        assert_eq!(emb.to_f32(), data);
        assert_eq!(emb.dim(), 5);
    }

    #[test]
    fn u8_to_i8_conversion() {
        // u8=0 → i8=-128, u8=128 → i8=0, u8=255 → i8=127
        let emb = EmbeddingOutput::U8(vec![0, 128, 255]);
        let i8_out = emb.to_i8();
        assert_eq!(i8_out[0], -128);
        assert_eq!(i8_out[1], 0);
        assert_eq!(i8_out[2], 127);
    }

    #[test]
    fn i8_to_u8_conversion() {
        let emb = EmbeddingOutput::I8(vec![-128, 0, 127]);
        let u8_out = emb.to_u8();
        assert_eq!(u8_out[0], 0);
        assert_eq!(u8_out[1], 128);
        assert_eq!(u8_out[2], 255);
    }

    #[test]
    fn cosine_identical() {
        let a = EmbeddingOutput::F32(vec![1.0, 0.0, 0.0]);
        let b = EmbeddingOutput::F32(vec![1.0, 0.0, 0.0]);
        assert!((a.cosine(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = EmbeddingOutput::F32(vec![1.0, 0.0]);
        let b = EmbeddingOutput::F32(vec![0.0, 1.0]);
        assert!(a.cosine(&b).abs() < 1e-6);
    }

    #[test]
    fn cosine_cross_type() {
        // f32 vs u8 — same values, different encoding
        let a = EmbeddingOutput::F32(vec![0.5, -0.5]);
        let b = EmbeddingOutput::U8(vec![191, 64]); // ~0.496, ~-0.504
        let cos = a.cosine(&b);
        assert!(cos > 0.99, "cross-type cosine should be high: {}", cos);
    }

    #[test]
    fn batch_pairwise() {
        let mut batch = EmbeddingBatch::new("test");
        batch.push(EmbeddingOutput::F32(vec![1.0, 0.0]));
        batch.push(EmbeddingOutput::F32(vec![0.0, 1.0]));
        batch.push(EmbeddingOutput::F32(vec![1.0, 1.0]));

        let matrix = batch.pairwise_cosines();
        assert_eq!(matrix.len(), 3);
        assert!((matrix[0][0] - 1.0).abs() < 1e-6); // self = 1
        assert!(matrix[0][1].abs() < 1e-6); // orthogonal
        assert!(matrix[0][2] > 0.5); // [1,0] vs [1,1] = cos(45°) ≈ 0.707
    }
}
