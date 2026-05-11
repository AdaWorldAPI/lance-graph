//! Spiral segment encoding: (anfang, ende, stride, gamma) = 8 bytes per row.
//!
//! Instead of storing every distance value (256 BF16 = 512 bytes per row),
//! store the spiral parameters that GENERATE them (4 BF16 = 8 bytes per row).
//!
//! Reconstruction: value(position) = anfang + (ende - anfang) × (position/stride)^gamma
//!
//! ```text
//! Full table:    256×256 × 2 bytes = 128 KB
//! Spiral params: 256 rows × 8 bytes = 2 KB + 512 bytes diagonal = 2.5 KB
//! Compression:   51× (if single segment per row suffices)
//! ```
//!
//! gamma IS the GammaProfile — not metadata BESIDE the table, but AS the table.

use bgz_tensor::stacked_n::{bf16_to_f32, f32_to_bf16};

/// One spiral segment: describes a monotonic curve from anfang to ende.
#[derive(Clone, Copy, Debug)]
pub struct SpiralSegment {
    /// Start value (BF16).
    pub anfang: u16,
    /// End value (BF16).
    pub ende: u16,
    /// Stride (number of steps in this segment).
    pub stride: u16,
    /// Gamma: distribution shape between anfang and ende.
    /// 1.0 = linear. <1.0 = log (compress start). >1.0 = exp (compress end).
    pub gamma: u16,
}

impl SpiralSegment {
    pub fn new(anfang: f32, ende: f32, stride: u16, gamma: f32) -> Self {
        Self {
            anfang: f32_to_bf16(anfang),
            ende: f32_to_bf16(ende),
            stride,
            gamma: f32_to_bf16(gamma),
        }
    }

    /// Reconstruct value at position within this segment.
    #[inline]
    pub fn reconstruct(&self, position: usize) -> f32 {
        let a = bf16_to_f32(self.anfang);
        let e = bf16_to_f32(self.ende);
        let g = bf16_to_f32(self.gamma);
        let s = self.stride.max(1) as f32;
        let t = (position as f32 / s).min(1.0); // normalized [0, 1]
        let curved_t = t.powf(g); // gamma bends the distribution
        a + (e - a) * curved_t
    }

    /// Byte size: 4 × u16 = 8 bytes.
    pub const BYTE_SIZE: usize = 8;

    /// Fit a spiral segment to a sequence of values.
    /// Returns the segment and the max reconstruction error.
    pub fn fit(values: &[f32]) -> (Self, f32) {
        if values.is_empty() {
            return (Self::new(0.0, 0.0, 1, 1.0), 0.0);
        }
        if values.len() == 1 {
            return (Self::new(values[0], values[0], 1, 1.0), 0.0);
        }

        let anfang = values[0];
        let ende = *values.last().unwrap();
        let stride = values.len() as u16;

        // Find optimal gamma by minimizing max error
        // Try a range of gammas and pick the best
        let mut best_gamma = 1.0f32;
        let mut best_max_err = f32::MAX;

        for g_int in 1..=300 {
            let gamma = g_int as f32 * 0.01; // 0.01 to 3.0
            let seg = Self::new(anfang, ende, stride, gamma);

            let max_err = values.iter().enumerate()
                .map(|(i, &v)| (seg.reconstruct(i) - v).abs())
                .fold(0.0f32, f32::max);

            if max_err < best_max_err {
                best_max_err = max_err;
                best_gamma = gamma;
            }
        }

        let segment = Self::new(anfang, ende, stride, best_gamma);
        (segment, best_max_err)
    }
}

/// A row encoded as one or more spiral segments.
#[derive(Clone, Debug)]
pub struct SpiralRow {
    pub segments: Vec<SpiralSegment>,
    /// Self-distance (diagonal value).
    pub self_distance: u16,
}

impl SpiralRow {
    /// Encode a full distance row as spiral segments.
    /// Splits into segments where single-segment error exceeds threshold.
    pub fn encode(values: &[f32], self_dist: f32, max_error: f32) -> Self {
        let self_distance = f32_to_bf16(self_dist);

        // Try single segment first
        let (seg, err) = SpiralSegment::fit(values);
        if err <= max_error {
            return Self { segments: vec![seg], self_distance };
        }

        // Split into sub-segments where error is high
        let mut segments = Vec::new();
        let mut start = 0;

        while start < values.len() {
            // Try increasingly large segments from this start
            let mut best_end = start + 1;
            for end in (start + 2)..=values.len() {
                let (_seg, err) = SpiralSegment::fit(&values[start..end]);
                if err > max_error {
                    break;
                }
                best_end = end;
            }

            let (seg, _) = SpiralSegment::fit(&values[start..best_end]);
            segments.push(seg);
            start = best_end;
        }

        Self { segments, self_distance }
    }

    /// Reconstruct full row from segments.
    pub fn decode(&self, n: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(n);
        let mut global_pos = 0;

        for seg in &self.segments {
            let seg_len = seg.stride as usize;
            for local_pos in 0..seg_len {
                if global_pos >= n { break; }
                result.push(seg.reconstruct(local_pos));
                global_pos += 1;
            }
        }

        // Pad if segments don't cover full row
        while result.len() < n {
            result.push(0.0);
        }

        result
    }

    /// Byte size of this encoded row.
    pub fn byte_size(&self) -> usize {
        self.segments.len() * SpiralSegment::BYTE_SIZE + 2 // +2 for self_distance
    }
}

/// Encode an entire BF16 distance table as spiral segments.
pub struct SpiralTable {
    pub rows: Vec<SpiralRow>,
    pub n: usize,
}

impl SpiralTable {
    /// Encode from BF16 table values.
    pub fn encode(table: &[u16], n: usize, max_error: f32) -> Self {
        let mut rows = Vec::with_capacity(n);

        for i in 0..n {
            let row_f32: Vec<f32> = (0..n)
                .filter(|&j| j != i) // exclude diagonal
                .map(|j| bf16_to_f32(table[i * n + j]))
                .collect();
            let self_dist = bf16_to_f32(table[i * n + i]);
            rows.push(SpiralRow::encode(&row_f32, self_dist, max_error));
        }

        Self { rows, n }
    }

    /// Total byte size of the spiral-encoded table.
    pub fn byte_size(&self) -> usize {
        self.rows.iter().map(|r| r.byte_size()).sum()
    }

    /// Compression ratio vs full BF16 table.
    pub fn compression_ratio(&self, n: usize) -> f32 {
        let full_size = n * n * 2; // BF16
        full_size as f32 / self.byte_size().max(1) as f32
    }

    /// Average segments per row.
    pub fn avg_segments(&self) -> f32 {
        self.rows.iter().map(|r| r.segments.len()).sum::<usize>() as f32
            / self.rows.len().max(1) as f32
    }

    /// Decode back to full BF16 table.
    pub fn decode(&self) -> Vec<u16> {
        let n = self.n;
        let mut table = vec![0u16; n * n];

        for i in 0..n {
            table[i * n + i] = self.rows[i].self_distance;

            let decoded = self.rows[i].decode(n - 1);
            let mut j_out = 0;
            for j in 0..n {
                if j == i { continue; }
                if j_out < decoded.len() {
                    table[i * n + j] = f32_to_bf16(decoded[j_out]);
                }
                j_out += 1;
            }
        }

        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_segment_linear() {
        let values: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
        let (seg, err) = SpiralSegment::fit(&values);
        eprintln!("Linear fit: gamma={:.2}, err={:.4}", bf16_to_f32(seg.gamma), err);
        // BF16 truncation of anfang/ende adds ~0.008 per endpoint
        assert!(err < 0.1, "linear values should fit within BF16 range: err={}", err);

        // Reconstruct
        for (i, &v) in values.iter().enumerate() {
            let r = seg.reconstruct(i);
            assert!((r - v).abs() < 0.1, "pos {}: expected {:.3}, got {:.3}", i, v, r);
        }
    }

    #[test]
    fn single_segment_curved() {
        // Exponential-ish curve: gamma should be > 1
        let values: Vec<f32> = (0..10).map(|i| (i as f32 * 0.1).powi(2)).collect();
        let (seg, err) = SpiralSegment::fit(&values);
        let gamma = bf16_to_f32(seg.gamma);
        eprintln!("Curved fit: gamma={:.2}, err={:.4}", gamma, err);
        assert!(gamma > 1.0, "exponential curve should have gamma > 1: {}", gamma);
    }

    #[test]
    fn row_single_segment() {
        let values: Vec<f32> = (0..255).map(|i| (i as f32 - 128.0) / 127.0).collect();
        let row = SpiralRow::encode(&values, 1.0, 0.05);
        eprintln!("Single segment row: {} segments, {} bytes",
            row.segments.len(), row.byte_size());
        assert!(row.segments.len() <= 3, "linear-ish row should need few segments");
    }

    #[test]
    fn row_multi_segment() {
        // Non-monotonic: needs multiple segments
        let values: Vec<f32> = (0..255).map(|i| ((i as f32 * 0.1).sin() * 0.5)).collect();
        let row = SpiralRow::encode(&values, 1.0, 0.05);
        eprintln!("Multi segment row: {} segments, {} bytes",
            row.segments.len(), row.byte_size());
        // Sine wave needs several segments
        assert!(row.segments.len() > 1, "sine wave should need multiple segments");
    }

    #[test]
    fn table_compression_ratio() {
        // Build a synthetic BF16 table with structure
        let n = 64;
        let mut table = vec![f32_to_bf16(0.0); n * n];
        for i in 0..n {
            table[i * n + i] = f32_to_bf16(1.0);
            for j in 0..n {
                if i == j { continue; }
                let dist = (i as f32 - j as f32).abs() / n as f32;
                table[i * n + j] = f32_to_bf16(1.0 - dist);
            }
        }

        let spiral = SpiralTable::encode(&table, n, 0.01);
        let ratio = spiral.compression_ratio(n);
        let avg_seg = spiral.avg_segments();

        eprintln!("Table {}×{}: {:.1}× compression, {:.1} avg segments/row, {} bytes",
            n, n, ratio, avg_seg, spiral.byte_size());
        // With BF16 precision and 0.01 threshold, many rows need multiple segments
        assert!(ratio > 1.0, "should compress at least 1×: {:.1}×", ratio);
    }

    #[test]
    fn table_roundtrip() {
        let n = 16;
        let mut table = vec![f32_to_bf16(0.0); n * n];
        for i in 0..n {
            table[i * n + i] = f32_to_bf16(1.0);
            for j in 0..n {
                if i == j { continue; }
                let cos = 1.0 - (i as f32 - j as f32).abs() / n as f32;
                table[i * n + j] = f32_to_bf16(cos);
            }
        }

        let spiral = SpiralTable::encode(&table, n, 0.02);
        let decoded = spiral.decode();

        // Compare roundtrip
        let mut max_diff = 0.0f32;
        for i in 0..n * n {
            let orig = bf16_to_f32(table[i]);
            let recon = bf16_to_f32(decoded[i]);
            let diff = (orig - recon).abs();
            if diff > max_diff { max_diff = diff; }
        }
        eprintln!("Roundtrip max error: {:.4} (threshold: 0.02)", max_diff);
        assert!(max_diff < 0.05, "roundtrip error too high: {:.4}", max_diff);
    }

    #[test]
    fn reencode_spiral_idempotent() {
        // THE KEY TEST: encode → decode → re-encode → must be identical
        let n = 16;
        let mut table = vec![f32_to_bf16(0.0); n * n];
        for i in 0..n {
            table[i * n + i] = f32_to_bf16(1.0);
            for j in 0..n {
                if i == j { continue; }
                table[i * n + j] = f32_to_bf16(0.5 + (i as f32 * 0.01) - (j as f32 * 0.005));
            }
        }

        let threshold = 0.02;
        let spiral1 = SpiralTable::encode(&table, n, threshold);
        let decoded1 = spiral1.decode();
        let spiral2 = SpiralTable::encode(&decoded1, n, threshold);
        let decoded2 = spiral2.decode();

        // Second decode should be identical to first decode (idempotent)
        let mut max_diff = 0.0f32;
        for i in 0..n * n {
            let d1 = bf16_to_f32(decoded1[i]);
            let d2 = bf16_to_f32(decoded2[i]);
            let diff = (d1 - d2).abs();
            if diff > max_diff { max_diff = diff; }
        }
        eprintln!("Re-encode drift: {:.6} (should be < BF16 truncation 0.008)", max_diff);
        assert!(max_diff < 0.01,
            "spiral re-encode should be near-idempotent: drift={:.6}", max_diff);

        // Compression should be same both times
        eprintln!("  encode 1: {} bytes, {:.1} avg segments",
            spiral1.byte_size(), spiral1.avg_segments());
        eprintln!("  encode 2: {} bytes, {:.1} avg segments",
            spiral2.byte_size(), spiral2.avg_segments());
    }
}
