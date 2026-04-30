//! Local implementations of functions that bgz-tensor needs but that ndarray
//! either doesn't expose publicly or doesn't implement.
//!
//! - `wht_f32`: Walsh-Hadamard Transform (ndarray's fft module only has FFT/IFFT)
//! - `dequantize_i8_to_f32`: inverse of ndarray's `quantize_f32_to_i8`
//! - `quantize_f32_to_i2` / `dequantize_i2_to_f32`: 2-bit quantization (not in ndarray)
//! - `kmeans` / `squared_l2`: exist in ndarray::hpc::cam_pq but are private

// Re-export QuantParams from ndarray so callers can use a single import path.
pub use ndarray::hpc::quantized::QuantParams;

/// Walsh-Hadamard Transform (in-place, unnormalized).
///
/// `data` must have power-of-two length. Standard butterfly.
pub fn wht_f32(data: &mut [f32]) {
    let n = data.len();
    assert!(n.is_power_of_two(), "WHT requires power-of-two length, got {n}");
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

/// Dequantize i8 codes back to f32 using the QuantParams from `quantize_f32_to_i8`.
///
/// ndarray's symmetric i8 quantization uses `code = round(val / scale)`,
/// so dequantization is `val ≈ code * scale`.
pub fn dequantize_i8_to_f32(codes: &[i8], params: &QuantParams, n: usize) -> Vec<f32> {
    codes
        .iter()
        .take(n)
        .map(|&c| c as f32 * params.scale)
        .collect()
}

/// 2-bit symmetric quantization: maps values to {-1, 0, +1} packed 4 per byte.
///
/// Encoding: 2-bit signed value per element, stored in LSB-first order within
/// each byte. Bit pattern: 0b11 = -1, 0b00 = 0, 0b01 = +1.
pub fn quantize_f32_to_i2(data: &[f32]) -> (Vec<u8>, QuantParams) {
    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let abs_max = min_val.abs().max(max_val.abs());
    let scale = if abs_max > 0.0 { 1.0 / abs_max } else { 1.0 };

    let mut packed = vec![0u8; (data.len() + 3) / 4];
    for (i, &v) in data.iter().enumerate() {
        let q = (v * scale).round().clamp(-1.0, 1.0) as i8;
        let bits = (q & 0x03) as u8;
        packed[i / 4] |= bits << ((i % 4) * 2);
    }
    (packed, QuantParams {
        scale: abs_max,
        zero_point: 0,
        min_val,
        max_val,
    })
}

/// Dequantize 2-bit packed codes back to f32.
pub fn dequantize_i2_to_f32(packed: &[u8], params: &QuantParams, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let byte = packed[i / 4];
        let bits = (byte >> ((i % 4) * 2)) & 0x03;
        let val = match bits {
            0b11 => -1.0f32,
            0b01 => 1.0,
            _ => 0.0,
        };
        out.push(val * params.scale);
    }
    out
}

/// Simple k-means clustering (Lloyd's algorithm).
///
/// Mirrors the private `kmeans` in `ndarray::hpc::cam_pq`.
pub fn kmeans(data: &[Vec<f32>], k: usize, dim: usize, iterations: usize) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return Vec::new();
    }
    // Seed centroids from the first k data points.
    let mut centroids: Vec<Vec<f32>> = data.iter().take(k).cloned().collect();
    while centroids.len() < k {
        centroids.push(vec![0.0; dim]);
    }

    for _ in 0..iterations {
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];

        for point in data {
            let mut best = 0;
            let mut best_d = f32::MAX;
            for (c, centroid) in centroids.iter().enumerate() {
                let d: f32 = point.iter().zip(centroid).map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            counts[best] += 1;
            for (s, &p) in sums[best].iter_mut().zip(point) {
                *s += p;
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / counts[c] as f32;
                }
            }
        }
    }
    centroids
}

/// Squared Euclidean distance between two slices.
pub fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}
