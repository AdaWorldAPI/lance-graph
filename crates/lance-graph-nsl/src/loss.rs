//! NSL's graph loss, `lib/distances.pairwise_distance_wrapper`, for the
//! configurations the oracle covers: L2 (`mean_squared_error`), L1
//! (`absolute_difference`) and cosine (`cosine_distance` after
//! `l2_normalize`), reduced by `SUM_BY_NONZERO_WEIGHTS`, with one weight per
//! (source, target) pair broadcast over the feature axis.
//!
//! Accumulates in f64 and returns f32. TensorFlow sums in f32 in its own
//! order, so results agree to a few ULP, not bit for bit.

/// Which distance NSL's `DistanceConfig` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    L2,
    L1,
    Cosine,
}

/// `sum_over_axis`: `None` averages over feature elements, `Last` sums along
/// the feature axis (required for cosine).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    None,
    Last,
}

/// Weighted graph loss over `pairs` rows of `dim` features each.
/// `src` and `tgt` are row-major `pairs * dim`; `w` has one entry per pair.
pub fn graph_loss(src: &[f32], tgt: &[f32], w: &[f32], dim: usize, d: Distance, axis: Axis) -> f32 {
    assert_eq!(src.len(), tgt.len());
    assert_eq!(src.len(), w.len() * dim);
    let nonzero = w.iter().filter(|&&x| x != 0.0).count();
    if nonzero == 0 {
        return 0.0;
    }
    let mut total = 0.0f64;
    for (i, &wi) in w.iter().enumerate() {
        let (s, t) = (&src[i * dim..(i + 1) * dim], &tgt[i * dim..(i + 1) * dim]);
        let row = match d {
            Distance::L2 => s
                .iter()
                .zip(t)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum::<f64>(),
            Distance::L1 => s
                .iter()
                .zip(t)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                .sum::<f64>(),
            Distance::Cosine => {
                let norm = |v: &[f32]| {
                    v.iter()
                        .map(|&x| f64::from(x).powi(2))
                        .sum::<f64>()
                        .max(1e-12)
                        .sqrt()
                };
                let dot: f64 = s
                    .iter()
                    .zip(t)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                1.0 - dot / (norm(s) * norm(t))
            }
        };
        total += f64::from(wi) * row;
    }
    let present = match (d, axis) {
        (Distance::Cosine, _) => nonzero as f64,
        (_, Axis::None) => (nonzero * dim) as f64,
        (_, Axis::Last) => nonzero as f64,
    };
    (total / present) as f32
}
