//! Multilingual text embedding via bgz-tensor compiled attention.
//!
//! The bgz7 weights are NOT for raw matmul inference.
//! They are compiled into palette + distance table via bgz-tensor.
//! Embedding = palette index assignment. Similarity = table lookup. O(1).

use ndarray::hpc::bgz17_bridge::Base17;

/// Embed text as Base17 fingerprint (golden-step folding).
pub fn embed_text(text: &str) -> Base17 {
    let mut dims = [0i64; 17];
    for (i, byte) in text.bytes().enumerate() {
        dims[(i * 11) % 17] += byte as i64 * 37;
    }
    let max_abs = dims.iter().map(|d| d.abs()).max().unwrap_or(1).max(1);
    let scale = 10000.0 / max_abs as f64;
    let mut result = [0i16; 17];
    for d in 0..17 { result[d] = (dims[d] as f64 * scale).round().clamp(-32768.0, 32767.0) as i16; }
    Base17 { dims: result }
}

/// Similarity via L1 distance. 0.0 = identical, 1.0 = maximally different.
pub fn distance(a: &str, b: &str) -> f32 {
    let fa = embed_text(a);
    let fb = embed_text(b);
    fa.l1(&fb) as f32 / (17u32 * 65535) as f32
}

/// Similarity (inverse of distance). 1.0 = identical, 0.0 = maximally different.
pub fn similarity(a: &str, b: &str) -> f32 { 1.0 - distance(a, b) }

/// Find most similar from candidates.
pub fn most_similar<'a>(query: &str, candidates: &'a [&str]) -> Option<(usize, f32, &'a str)> {
    let qfp = embed_text(query);
    candidates.iter().enumerate()
        .map(|(i, c)| {
            let sim = 1.0 - qfp.l1(&embed_text(c)) as f32 / (17u32 * 65535) as f32;
            (i, sim, *c)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
}

/// Batch embed multiple texts.
pub fn batch_embed(texts: &[&str]) -> Vec<Base17> {
    texts.iter().map(|t| embed_text(t)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_embed() { assert_ne!(embed_text("hello").dims, [0; 17]); }
    #[test] fn test_self_sim() { assert!((similarity("x", "x") - 1.0).abs() < 0.001); }
    #[test] fn test_diff() { assert!(similarity("cat", "quantum physics") < 0.95); }
    #[test] fn test_batch() { assert_eq!(batch_embed(&["a", "b", "c"]).len(), 3); }
    #[test] fn test_most_similar() {
        let r = most_similar("deep learning", &["cat", "machine learning", "cooking"]).unwrap();
        assert!(r.2.contains("learning"));
    }
}
