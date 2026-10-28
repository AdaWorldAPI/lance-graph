//! NSL's graph loss, `lib/distances.pairwise_distance_wrapper`, for the
//! configurations the oracle covers: L2 (`mean_squared_error`), L1
//! (`absolute_difference`), cosine (`cosine_distance` after `l2_normalize`),
//! and KL / Jensen-Shannon divergence after `transform_fn=SOFTMAX` along the
//! last axis, reduced by `SUM_BY_NONZERO_WEIGHTS`, with one weight per
//! (source, target) pair broadcast over the feature axis.
//!
//! KL is NSL's `_kl_divergence_fn`: `p*ln(p+1e-7) - p*ln(q+1e-7)` summed over
//! the class axis, with P = softmax(sources) and Q = softmax(targets). JS uses
//! M = (P+Q)/2 and halves both KL terms.
//!
//! Accumulates in f64 and returns f32. TensorFlow sums in f32 in its own
//! order, so results agree to a few ULP, not bit for bit.

/// Which distance NSL's `DistanceConfig` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    L2,
    L1,
    Cosine,
    /// `kl_divergence` with `transform_fn=SOFTMAX`.
    Kl,
    /// `jensen_shannon_divergence` with `transform_fn=SOFTMAX`.
    Js,
}

/// NSL's epsilon in `_kl_divergence_fn`, as the f32 TensorFlow adds.
pub const KL_EPS: f64 = 1e-7f32 as f64;

/// Softmax of one row with the max subtracted first; probabilities are
/// rounded to f32, the precision TensorFlow holds them in.
pub fn softmax_row(x: &[f32]) -> Vec<f64> {
    let m = x
        .iter()
        .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
    let e: Vec<f64> = x.iter().map(|&v| (f64::from(v) - m).exp()).collect();
    let z: f64 = e.iter().sum();
    e.iter().map(|&v| f64::from((v / z) as f32)).collect()
}

/// NSL's per-row divergence on two probability rows.
pub fn divergence_row(p: &[f64], q: &[f64], d: Distance) -> f64 {
    let kl = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| x * (x + KL_EPS).ln() - x * (y + KL_EPS).ln())
            .sum()
    };
    match d {
        Distance::Kl => kl(p, q),
        Distance::Js => {
            let m: Vec<f64> = p.iter().zip(q).map(|(&x, &y)| 0.5 * (x + y)).collect();
            0.5 * kl(p, &m) + 0.5 * kl(q, &m)
        }
        _ => panic!("divergence_row needs Kl or Js"),
    }
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
            Distance::Kl | Distance::Js => divergence_row(&softmax_row(s), &softmax_row(t), d),
        };
        total += f64::from(wi) * row;
    }
    let present = match (d, axis) {
        (Distance::Cosine | Distance::Kl | Distance::Js, _) => nonzero as f64,
        (_, Axis::None) => (nonzero * dim) as f64,
        (_, Axis::Last) => nonzero as f64,
    };
    (total / present) as f32
}
