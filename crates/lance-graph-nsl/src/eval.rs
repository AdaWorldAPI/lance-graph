//! Evaluation of quantized NSL paths against the frozen TensorFlow oracle.
//!
//! Not a production codec. The 9-bit lane here mirrors ndarray
//! `hpc::codec9` (`EpiLane9`, `Codec9::encode_touched`): magnitude u8 plus a
//! presence bit, +0 canonical, ties away from zero, saturating at 255. The
//! difference is a scale chosen at runtime from the data, where Codec9 fixes
//! it at compile time.

use crate::loss::{divergence_row, graph_loss, softmax_row, Axis, Distance, KL_EPS};

const LOSS_GOLDEN: &str = include_str!("../data/nsl_loss_golden.txt");

pub(crate) struct Case {
    pub name: String,
    pub dist: Distance,
    pub axis: Axis,
    pub dim: usize,
    pub src: Vec<f32>,
    pub tgt: Vec<f32>,
    pub w: Vec<f32>,
    pub tf: f32,
}

fn f32_of(hex: &str) -> f32 {
    f32::from_bits(u32::from_str_radix(hex, 16).expect("hex"))
}

pub(crate) fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    let mut cur: Option<Case> = None;
    for line in LOSS_GOLDEN.lines().filter(|l| !l.starts_with('#')) {
        let mut tok = line.split(' ');
        let head = tok.next().unwrap();
        let vals = || line.split(' ').skip(1).map(f32_of);
        match head {
            "case" => {
                let t: Vec<&str> = tok.collect();
                cur = Some(Case {
                    name: t[0].into(),
                    dist: match t[1] {
                        "l2" => Distance::L2,
                        "l1" => Distance::L1,
                        "cosine" => Distance::Cosine,
                        "kl_divergence" => Distance::Kl,
                        "jensen_shannon_divergence" => Distance::Js,
                        o => panic!("distance {o}"),
                    },
                    axis: if t[2] == "none" {
                        Axis::None
                    } else {
                        Axis::Last
                    },
                    dim: t[5].parse().unwrap(),
                    src: vec![],
                    tgt: vec![],
                    w: vec![],
                    tf: 0.0,
                });
            }
            "src" => cur.as_mut().unwrap().src.extend(vals()),
            "tgt" => cur.as_mut().unwrap().tgt.extend(vals()),
            "w" => cur.as_mut().unwrap().w.extend(vals()),
            "tf" => cur.as_mut().unwrap().tf = f32_of(tok.next().unwrap()),
            "end" => out.push(cur.take().unwrap()),
            o => panic!("bad loss golden line {o}"),
        }
    }
    out
}

pub(crate) fn ulps(a: f32, b: f32) -> u32 {
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32
}

/// A sign-magnitude lane with `levels` magnitudes (255 for the 9-bit lane,
/// 127 for int8 Sym), quantized with a given scale, ties away from zero.
fn quantize(x: f32, scale: f64, levels: i32) -> i32 {
    let q = (f64::from(x) / scale).round() as i64; // f64::round is ties away
    q.clamp(-i64::from(levels), i64::from(levels)) as i32
}

fn shared_scale(a: &[f32], b: &[f32], levels: i32) -> f64 {
    let m = a
        .iter()
        .chain(b)
        .fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
    if m == 0.0 {
        1.0
    } else {
        m / f64::from(levels)
    }
}

/// Graph loss with both sides quantized under one shared scale, computed in
/// integer code space and rescaled once. Cosine needs no rescale.
pub(crate) fn loss_shared(c: &Case, levels: i32, quantize_weights: bool) -> f32 {
    if matches!(c.dist, Distance::Kl | Distance::Js) {
        return loss_softmax_table(c, levels, None);
    }
    let s = shared_scale(&c.src, &c.tgt, levels);
    let qa: Vec<i32> = c.src.iter().map(|&x| quantize(x, s, levels)).collect();
    let qb: Vec<i32> = c.tgt.iter().map(|&x| quantize(x, s, levels)).collect();
    let w: Vec<f64> =
        c.w.iter()
            .map(|&x| {
                if quantize_weights {
                    f64::from(quantize(x, 1.0 / 255.0, 255)) / 255.0
                } else {
                    f64::from(x)
                }
            })
            .collect();
    let nonzero = w.iter().filter(|&&x| x != 0.0).count();
    let mut total = 0.0f64;
    for (i, &wi) in w.iter().enumerate() {
        let (a, b) = (
            &qa[i * c.dim..(i + 1) * c.dim],
            &qb[i * c.dim..(i + 1) * c.dim],
        );
        let row = match c.dist {
            Distance::L2 => {
                a.iter()
                    .zip(b)
                    .map(|(&x, &y)| i64::from(x - y).pow(2))
                    .sum::<i64>() as f64
                    * s
                    * s
            }
            Distance::L1 => {
                a.iter()
                    .zip(b)
                    .map(|(&x, &y)| i64::from((x - y).abs()))
                    .sum::<i64>() as f64
                    * s
            }
            Distance::Cosine => {
                // sign-aware MAC: |x|*|y| added or subtracted by sign XOR
                let dot: i64 = a
                    .iter()
                    .zip(b)
                    .map(|(&x, &y)| {
                        let p = i64::from(x.unsigned_abs()) * i64::from(y.unsigned_abs());
                        if (x < 0) ^ (y < 0) {
                            -p
                        } else {
                            p
                        }
                    })
                    .sum();
                let n = |v: &[i32]| {
                    (v.iter().map(|&x| i64::from(x).pow(2)).sum::<i64>() as f64)
                        .max(1e-12)
                        .sqrt()
                };
                1.0 - dot as f64 / (n(a) * n(b))
            }
            Distance::Kl | Distance::Js => unreachable!("handled above"),
        };
        total += wi * row;
    }
    let present = match (c.dist, c.axis) {
        (Distance::Cosine, _) | (_, Axis::Last) => nonzero as f64,
        (_, Axis::None) => (nonzero * c.dim) as f64,
    };
    (total / present) as f32
}

/// Softmax of one row of input codes, shaped like TFLM's int8 softmax
/// (`reference/softmax.h`): the row max is subtracted in code space, so the
/// largest logit always reads table[0] = exp(0) = 1 exactly (the needle), and
/// every other code reads one entry of a 256-or-511-entry exp table keyed by
/// `q - q_max`. One row sum, one division. With `out` set, the probabilities
/// are written back on an output grid of `out` steps (TFLM int8 output: 256,
/// scale 1/256; int16 output: 32768).
///
/// TFLM-shaped, not TFLM byte parity: the table here is f64 and the
/// division is exact; TFLM uses a fixed-point table and a reciprocal.
pub(crate) fn softmax_table_row(codes: &[i32], scale: f64, out: Option<u32>) -> Vec<f64> {
    let qmax = *codes.iter().max().expect("non-empty row");
    let e: Vec<f64> = codes
        .iter()
        .map(|&q| (scale * f64::from(q - qmax)).exp())
        .collect();
    let z: f64 = e.iter().sum();
    e.iter()
        .map(|&v| {
            let p = v / z;
            match out {
                None => p,
                Some(n) => (p * f64::from(n)).round().min(f64::from(n - 1)) / f64::from(n),
            }
        })
        .collect()
}

/// KL/JS graph loss on quantized logits: both sides share one input scale,
/// softmax runs through [`softmax_table_row`], the divergence is NSL's.
pub(crate) fn loss_softmax_table(c: &Case, levels: i32, out: Option<u32>) -> f32 {
    let s = shared_scale(&c.src, &c.tgt, levels);
    let q = |v: &[f32]| {
        v.iter()
            .map(|&x| quantize(x, s, levels))
            .collect::<Vec<i32>>()
    };
    let nonzero = c.w.iter().filter(|&&x| x != 0.0).count();
    if nonzero == 0 {
        return 0.0;
    }
    let mut total = 0.0f64;
    for (i, &wi) in c.w.iter().enumerate() {
        let r = i * c.dim..(i + 1) * c.dim;
        let p = softmax_table_row(&q(&c.src[r.clone()]), s, out);
        let qq = softmax_table_row(&q(&c.tgt[r]), s, out);
        total += f64::from(wi) * divergence_row(&p, &qq, c.dist);
    }
    (total / nonzero as f64) as f32
}

/// Graph loss (any distance) with the target coded as source + residual:
/// source rows on one scale, the residual `tgt - src` on its own scale, both
/// `levels` magnitudes. Dequantized and computed by the f32 reference (for
/// KL/JS the target codes are off the source grid, so no exp table).
pub(crate) fn loss_residual(c: &Case, levels: i32) -> f32 {
    let s = shared_scale(&c.src, &[], levels);
    let d: Vec<f32> = c.src.iter().zip(&c.tgt).map(|(&a, &b)| b - a).collect();
    let sd = shared_scale(&d, &[], levels);
    let a: Vec<f32> = c
        .src
        .iter()
        .map(|&x| (f64::from(quantize(x, s, levels)) * s) as f32)
        .collect();
    let b: Vec<f32> = a
        .iter()
        .zip(&d)
        .map(|(&x, &y)| (f64::from(x) + f64::from(quantize(y, sd, levels)) * sd) as f32)
        .collect();
    graph_loss(&a, &b, &c.w, c.dim, c.dist, c.axis)
}

/// Mirror of `bgz-tensor::fisher_z::FamilyGamma` (lance-graph
/// `crates/bgz-tensor/src/fisher_z.rs` at deb0588a6): z = atanh of the cosine
/// clamped to +-0.9999, fitted min/range, i8 code = trunc(norm*254 - 127).
/// Copied rather than linked: bgz-tensor pulls ndarray and holograph, and this
/// crate stays dependency-free. Same f32 arithmetic, same truncation.
#[derive(Clone, Copy)]
pub(crate) struct FamilyGamma {
    z_min: f32,
    z_range: f32,
}

fn atanh_clamp(x: f32) -> f32 {
    let c = x.clamp(-0.9999, 0.9999);
    0.5 * ((1.0 + c) / (1.0 - c)).ln()
}

impl FamilyGamma {
    pub(crate) fn from_cosines(cos: &[f32]) -> Self {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &c in cos {
            let z = atanh_clamp(c);
            lo = lo.min(z);
            hi = hi.max(z);
        }
        FamilyGamma {
            z_min: lo,
            z_range: (hi - lo).max(1e-10),
        }
    }
    pub(crate) fn encode(&self, c: f32) -> i8 {
        let n = (atanh_clamp(c) - self.z_min) / self.z_range;
        (n * 254.0 - 127.0).clamp(-128.0, 127.0) as i8
    }
    pub(crate) fn decode(&self, v: i8) -> f32 {
        ((f32::from(v) + 127.0) / 254.0 * self.z_range + self.z_min).tanh()
    }
}

/// The two shapes an 8-byte envelope (two f32) can carry over z = atanh(c).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Shape {
    /// `a = z_min`, `b = z_range`, codes spread linearly over z.
    ZLinear,
    /// Gumbel probability-integral transform over z: `a = mu`, `b = beta`,
    /// fitted by moments, `beta = sd * sqrt(6) / pi` signed by the skew,
    /// `mu = mean - EULER_GAMMA * beta`. Codes are equal-probability bins, so
    /// a Gumbel-shaped z lands on a flat code histogram (the deskew).
    ZGumbel,
}

/// Two f32 of metadata (8 bytes) plus `levels` codes: 255 is the i8 reach of
/// [`FamilyGamma`], 511 is the numeric reach of a 9-bit SM9 lane.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Envelope {
    pub shape: Shape,
    pub a: f32,
    pub b: f32,
    pub levels: u32,
}

fn gumbel_cdf(t: f64) -> f64 {
    (-(-t).exp()).exp()
}

impl Envelope {
    pub(crate) fn fit(shape: Shape, vals: &[f32], levels: u32) -> Self {
        let z: Vec<f64> = vals.iter().map(|&v| f64::from(atanh_clamp(v))).collect();
        let (a, b) = match shape {
            Shape::ZLinear => {
                let lo = z.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (lo, (hi - lo).max(1e-10))
            }
            Shape::ZGumbel => {
                let n = z.len() as f64;
                let m = z.iter().sum::<f64>() / n;
                let var = z.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n;
                let skew = z.iter().map(|&x| (x - m).powi(3)).sum::<f64>() / n;
                let beta = (var.sqrt() * 6f64.sqrt() / std::f64::consts::PI).max(1e-10)
                    * if skew < 0.0 { -1.0 } else { 1.0 };
                (m - std::f64::consts::EULER_GAMMA * beta, beta)
            }
        };
        Envelope {
            shape,
            a: a as f32,
            b: b as f32,
            levels,
        }
    }

    pub(crate) fn encode(&self, c: f32) -> u32 {
        let z = f64::from(atanh_clamp(c));
        let (a, b, top) = (f64::from(self.a), f64::from(self.b), self.levels - 1);
        match self.shape {
            Shape::ZLinear => (((z - a) / b) * f64::from(top))
                .round()
                .clamp(0.0, f64::from(top)) as u32,
            Shape::ZGumbel => {
                let g = gumbel_cdf((z - a) / b);
                let u = if b > 0.0 { g } else { 1.0 - g };
                ((u * f64::from(self.levels)) as u32).min(top)
            }
        }
    }

    pub(crate) fn decode(&self, code: u32) -> f32 {
        let (a, b, l) = (f64::from(self.a), f64::from(self.b), f64::from(self.levels));
        let z = match self.shape {
            Shape::ZLinear => a + f64::from(code) / (l - 1.0) * b,
            Shape::ZGumbel => {
                let u = (f64::from(code) + 0.5) / l;
                let g = if b > 0.0 { u } else { 1.0 - u };
                a + b * -(-g.ln()).ln()
            }
        };
        z.tanh() as f32
    }
}

/// Ceiling for any deskew, not an 8-byte envelope: `levels` equal-count bins
/// from the sorted data, each decoded to its median.
pub(crate) struct Quantiles {
    bounds: Vec<f32>,
    reps: Vec<f32>,
}

impl Quantiles {
    pub(crate) fn fit(vals: &[f32], levels: usize) -> Self {
        let mut v = vals.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
        let bounds = (1..levels).map(|i| at(i as f64 / levels as f64)).collect();
        let reps = (0..levels)
            .map(|i| at((i as f64 + 0.5) / levels as f64))
            .collect();
        Quantiles { bounds, reps }
    }
    pub(crate) fn encode(&self, c: f32) -> usize {
        self.bounds.partition_point(|&b| b < c)
    }
    pub(crate) fn decode(&self, code: usize) -> f32 {
        self.reps[code]
    }
}

/// Affine 8-bit control for [`FamilyGamma`]: the same fitted range, spent
/// uniformly on the cosine instead of on atanh(cosine). Rounds to nearest.
#[derive(Clone, Copy)]
pub(crate) struct Affine8 {
    lo: f32,
    span: f32,
}

impl Affine8 {
    pub(crate) fn from_values(v: &[f32]) -> Self {
        let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        Affine8 {
            lo,
            span: (hi - lo).max(1e-10),
        }
    }
    pub(crate) fn encode(&self, x: f32) -> u8 {
        (((x - self.lo) / self.span) * 255.0)
            .round()
            .clamp(0.0, 255.0) as u8
    }
    pub(crate) fn decode(&self, c: u8) -> f32 {
        self.lo + f32::from(c) / 255.0 * self.span
    }
}

/// A palette256 over unit rows: centroid index = the needle (which row),
/// pair similarity = the Fisher-z i8 LUT (how close two needles are).
/// Same index reads 1.0 from the address, never from the table.
pub(crate) struct Palette {
    pub centroids: Vec<Vec<f32>>,
    pub lut_f32: Vec<f32>,
    pub lut_fz: Vec<i8>,
    pub gamma: FamilyGamma,
    pub lut_u8: Vec<u8>,
    pub affine: Affine8,
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .max(1e-24)
        .sqrt();
    v.iter().map(|&x| (f64::from(x) / n) as f32).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum::<f64>() as f32
}

impl Palette {
    /// Spherical k-means, `k` centroids, deterministic init from the first
    /// `k` training rows, fixed iteration count. Empty clusters keep their
    /// previous centroid.
    pub(crate) fn train(rows: &[f32], dim: usize, k: usize, iters: usize) -> Self {
        let data: Vec<Vec<f32>> = rows.chunks(dim).map(unit).collect();
        let mut cents: Vec<Vec<f32>> = data[..k].to_vec();
        for _ in 0..iters {
            let mut acc = vec![vec![0.0f64; dim]; k];
            let mut cnt = vec![0usize; k];
            for r in &data {
                let j = Self::nearest(&cents, r);
                cnt[j] += 1;
                for (a, &x) in acc[j].iter_mut().zip(r) {
                    *a += f64::from(x);
                }
            }
            for j in 0..k {
                if cnt[j] > 0 {
                    let v: Vec<f32> = acc[j].iter().map(|&x| x as f32).collect();
                    cents[j] = unit(&v);
                }
            }
        }
        let mut lut_f32 = vec![0.0f32; k * k];
        let mut off = Vec::with_capacity(k * (k - 1) / 2);
        for i in 0..k {
            for j in 0..k {
                let c = dot(&cents[i], &cents[j]);
                lut_f32[i * k + j] = c;
                if i < j {
                    off.push(c);
                }
            }
        }
        let gamma = FamilyGamma::from_cosines(&off);
        let affine = Affine8::from_values(&off);
        Palette {
            lut_fz: lut_f32.iter().map(|&c| gamma.encode(c)).collect(),
            lut_u8: lut_f32.iter().map(|&c| affine.encode(c)).collect(),
            lut_f32,
            gamma,
            affine,
            centroids: cents,
        }
    }

    fn nearest(cents: &[Vec<f32>], r: &[f32]) -> usize {
        let mut best = (f32::NEG_INFINITY, 0);
        for (j, c) in cents.iter().enumerate() {
            let s = dot(c, r);
            if s > best.0 {
                best = (s, j);
            }
        }
        best.1
    }

    pub(crate) fn index(&self, row: &[f32]) -> usize {
        Self::nearest(&self.centroids, &unit(row))
    }

    /// Pair cosine between two needles under a given LUT reading.
    pub(crate) fn pair(&self, a: usize, b: usize, lut: Lut) -> f64 {
        if a == b {
            return 1.0;
        }
        let k = self.centroids.len();
        f64::from(match lut {
            Lut::F32 => self.lut_f32[a * k + b],
            Lut::FisherZ => self.gamma.decode(self.lut_fz[a * k + b]),
            Lut::Affine => self.affine.decode(self.lut_u8[a * k + b]),
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Lut {
    F32,
    FisherZ,
    Affine,
}

/// NSL cosine graph loss where each row's cosine is read from the palette:
/// 1 - pair(index(source), index(target)).
pub(crate) fn loss_palette_cosine(c: &Case, p: &Palette, lut: Lut) -> f32 {
    let nonzero = c.w.iter().filter(|&&x| x != 0.0).count();
    if nonzero == 0 {
        return 0.0;
    }
    let mut total = 0.0f64;
    for (i, &wi) in c.w.iter().enumerate() {
        let r = i * c.dim..(i + 1) * c.dim;
        let (a, b) = (p.index(&c.src[r.clone()]), p.index(&c.tgt[r]));
        total += f64::from(wi) * (1.0 - p.pair(a, b, lut));
    }
    (total / nonzero as f64) as f32
}

/// Graph loss with every row quantized under its own scale, dequantized, and
/// computed in f32 by the reference. Not integer-only.
pub(crate) fn loss_rowscale(c: &Case, levels: i32) -> f32 {
    let deq = |v: &[f32]| -> Vec<f32> {
        v.chunks(c.dim)
            .flat_map(|row| {
                let s = shared_scale(row, &[], levels);
                row.iter()
                    .map(move |&x| (f64::from(quantize(x, s, levels)) * s) as f32)
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    graph_loss(&deq(&c.src), &deq(&c.tgt), &c.w, c.dim, c.dist, c.axis)
}

/// Cosine similarity between the f32 L2/L1 gradient w.r.t. the sources and
/// the gradient computed from shared-scale codes.
pub(crate) fn grad_cosine(c: &Case, levels: i32) -> f64 {
    let s = shared_scale(&c.src, &c.tgt, levels);
    let g = |a: f64, b: f64| match c.dist {
        Distance::L2 => a - b,
        _ => (a - b).signum(),
    };
    let (mut dot, mut nf, mut nq) = (0.0, 0.0, 0.0);
    for (i, (&a, &b)) in c.src.iter().zip(&c.tgt).enumerate() {
        let wi = f64::from(c.w[i / c.dim]);
        let gf = wi * g(f64::from(a), f64::from(b));
        let gq = wi
            * g(
                f64::from(quantize(a, s, levels)) * s,
                f64::from(quantize(b, s, levels)) * s,
            );
        dot += gf * gq;
        nf += gf * gf;
        nq += gq * gq;
    }
    if nf == 0.0 || nq == 0.0 {
        0.0
    } else {
        dot / (nf * nq).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_f32_reference_matches_tensorflow_within_a_few_ulp() {
        let cs = cases();
        assert_eq!(cs.len(), 16);
        let mut worst = 0;
        let mut checked = 0;
        for c in cs
            .iter()
            .filter(|c| !matches!(c.dist, Distance::Kl | Distance::Js))
        {
            let got = graph_loss(&c.src, &c.tgt, &c.w, c.dim, c.dist, c.axis);
            let u = ulps(got, c.tf);
            eprintln!("{:16} tf {:>14e} ref {:>14e} ulp {u}", c.name, c.tf, got);
            worst = worst.max(u);
            checked += 1;
        }
        assert_eq!(checked, 10);
        assert!(worst <= 4, "worst {worst} ulp");
    }

    /// Sum of |p ln(p+eps)| + |p ln(q+eps)| over all terms, weighted and
    /// averaged like the loss: the magnitude TensorFlow's f32 subtraction
    /// cancels. One f32 rounding of each term is eps32 times this.
    fn divergence_magnitude(c: &Case) -> f64 {
        let n = c.w.iter().filter(|&&x| x != 0.0).count() as f64;
        let mut m = 0.0;
        for (i, &w) in c.w.iter().enumerate() {
            let r = i * c.dim..(i + 1) * c.dim;
            let (p, q) = (softmax_row(&c.src[r.clone()]), softmax_row(&c.tgt[r]));
            let mix: Vec<f64> = p.iter().zip(&q).map(|(&a, &b)| 0.5 * (a + b)).collect();
            let t = |a: &[f64], b: &[f64]| -> f64 {
                a.iter()
                    .zip(b)
                    .map(|(&x, &y)| (x * (x + KL_EPS).ln()).abs() + (x * (y + KL_EPS).ln()).abs())
                    .sum()
            };
            m += f64::from(w)
                * match c.dist {
                    Distance::Kl => t(&p, &q),
                    _ => 0.5 * (t(&p, &mix) + t(&q, &mix)),
                };
        }
        m / n
    }

    /// KL/JS are a difference of near-equal f32 terms, so TensorFlow's own
    /// value carries a cancellation error of order eps32 * magnitude, which
    /// at small divergences is thousands of ULP of the result. The f64
    /// reference must land inside that budget, not within a few ULP.
    #[test]
    fn the_divergence_reference_lands_inside_tensorflows_f32_cancellation_budget() {
        let mut checked = 0;
        for c in cases()
            .iter()
            .filter(|c| matches!(c.dist, Distance::Kl | Distance::Js))
        {
            let got = graph_loss(&c.src, &c.tgt, &c.w, c.dim, c.dist, c.axis);
            let budget = f64::from(f32::EPSILON) * divergence_magnitude(c);
            let ratio = (f64::from(got) - f64::from(c.tf)).abs() / budget;
            eprintln!(
                "{:16} tf {:>14e} ref {:>14e} ulp {:>6} |ref-tf|/budget {ratio:.3}",
                c.name,
                c.tf,
                got,
                ulps(got, c.tf)
            );
            assert!(ratio <= 1.0, "{} ratio {ratio}", c.name);
            checked += 1;
        }
        assert_eq!(checked, 6);
    }

    #[test]
    fn the_toy_case_matches_the_hand_value() {
        let c = cases().into_iter().find(|c| c.name == "toy_00a").unwrap();
        let t = 1.0f64.tanh();
        let want = (0.9f32 as f64 + 0.4f32 as f64) * t * t / 4.0;
        assert!((f64::from(c.tf) - want).abs() < 1e-7, "{} vs {want}", c.tf);
    }

    #[test]
    fn zero_weights_add_no_loss_and_are_not_counted() {
        let c = cases()
            .into_iter()
            .find(|c| c.name == "l2_zerow_d32")
            .unwrap();
        let w0 = vec![0.0f32; c.w.len()];
        assert_eq!(graph_loss(&c.src, &c.tgt, &w0, c.dim, c.dist, c.axis), 0.0);
    }

    /// Report, not a gate: relative error of each quantized path vs TF.
    #[test]
    #[ignore]
    fn quantized_loss_probe() {
        eprintln!(
            "{:16} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9}",
            "case", "sm9", "sm9+w9", "int8", "sm9_row", "int8_row", "gcos9", "gcos8"
        );
        for c in cases() {
            let rel = |x: f32| ((f64::from(x) - f64::from(c.tf)) / f64::from(c.tf)).abs();
            let gc = |l| {
                if c.dist == Distance::Cosine {
                    f64::NAN
                } else {
                    grad_cosine(&c, l)
                }
            };
            eprintln!(
                "{:16} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e} {:>9.6} {:>9.6}",
                c.name,
                rel(loss_shared(&c, 255, false)),
                rel(loss_shared(&c, 255, true)),
                rel(loss_shared(&c, 127, false)),
                rel(loss_rowscale(&c, 255)),
                rel(loss_rowscale(&c, 127)),
                gc(255),
                gc(127)
            );
        }
    }

    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn n(&mut self) -> f64 {
            let (a, b) = (self.u().max(1e-300), self.u());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        }
    }

    fn synth(r: &mut Rng, pairs: usize, dim: usize, noise: f64, dist: Distance) -> Case {
        let src: Vec<f32> = (0..pairs * dim).map(|_| r.n() as f32).collect();
        let tgt: Vec<f32> = src
            .iter()
            .map(|&x| (f64::from(x) + noise * r.n()) as f32)
            .collect();
        let w = (0..pairs).map(|_| (0.5 + 0.5 * r.u()) as f32).collect();
        let axis = if dist == Distance::Cosine {
            Axis::Last
        } else {
            Axis::None
        };
        Case {
            name: String::new(),
            dist,
            axis,
            dim,
            src,
            tgt,
            w,
            tf: 0.0,
        }
    }

    fn pct(mut v: Vec<f64>, p: f64) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f64 * p).round() as usize]
    }

    /// Report, not a gate: 400 seeds per row, reference = the f32 path that
    /// matches TF within 3 ULP. Median and p95 relative loss error, and p05
    /// gradient cosine.
    #[test]
    #[ignore]
    fn quantized_loss_sweep() {
        eprintln!(
            "{:7} {:>4} {:>6} | {:>17} {:>17} {:>17} {:>17} | {:>8} {:>8}",
            "dist",
            "dim",
            "noise",
            "sm9 med/p95",
            "int8 med/p95",
            "sm9row med/p95",
            "int8row med/p95",
            "gc9 p05",
            "gc8 p05"
        );
        eprintln!("(last column: int8 source + int8 residual, med/p95)");
        let mut r = Rng(0x9E37_79B9_7F4A_7C15);
        for dist in [Distance::L2, Distance::L1, Distance::Cosine] {
            for dim in [32usize, 128] {
                for noise in [0.5, 0.14, 0.05, 0.01, 0.001] {
                    let (mut e9, mut e8, mut r9, mut r8, mut g9, mut g8) =
                        (vec![], vec![], vec![], vec![], vec![], vec![]);
                    let mut res8 = vec![];
                    for _ in 0..400 {
                        let c = synth(&mut r, 16, dim, noise, dist);
                        let f = f64::from(graph_loss(&c.src, &c.tgt, &c.w, c.dim, c.dist, c.axis));
                        let rel = |x: f32| ((f64::from(x) - f) / f).abs();
                        e9.push(rel(loss_shared(&c, 255, false)));
                        e8.push(rel(loss_shared(&c, 127, false)));
                        r9.push(rel(loss_rowscale(&c, 255)));
                        r8.push(rel(loss_rowscale(&c, 127)));
                        res8.push(rel(loss_residual(&c, 127)));
                        if dist != Distance::Cosine {
                            g9.push(grad_cosine(&c, 255));
                            g8.push(grad_cosine(&c, 127));
                        }
                    }
                    let mp = |v: &Vec<f64>| {
                        format!("{:.1e}/{:.1e}", pct(v.clone(), 0.5), pct(v.clone(), 0.95))
                    };
                    let gp = |v: &Vec<f64>| {
                        if v.is_empty() {
                            "-".into()
                        } else {
                            format!("{:.5}", pct(v.clone(), 0.05))
                        }
                    };
                    eprintln!(
                        "{:7} {:>4} {:>6} | {:>17} {:>17} {:>17} {:>17} | {:>8} {:>8} | {:>17}",
                        format!("{dist:?}"),
                        dim,
                        noise,
                        mp(&e9),
                        mp(&e8),
                        mp(&r9),
                        mp(&r8),
                        gp(&g9),
                        gp(&g8),
                        mp(&res8)
                    );
                }
            }
        }
    }

    /// Report, not a gate: storing input features quantized (SM9, one scale
    /// per row) and running a fixed random tanh layer, as NSL computes logits
    /// from the copied neighbor features. The loss itself stays f32.
    #[test]
    #[ignore]
    fn quantized_feature_storage_probe() {
        let mut r = Rng(0xD1B5_4A32_D192_ED03);
        let (din, dout, pairs) = (64usize, 16usize, 16usize);
        eprintln!(
            "{:>6} | {:>17} {:>17}",
            "noise", "sm9 in med/p95", "int8 in med/p95"
        );
        for noise in [0.5, 0.14, 0.05, 0.01, 0.001] {
            let mut errs = [vec![], vec![]];
            for _ in 0..200 {
                let wmat: Vec<f64> = (0..din * dout)
                    .map(|_| r.n() / (din as f64).sqrt())
                    .collect();
                let model = |x: &[f32]| -> Vec<f32> {
                    x.chunks(din)
                        .flat_map(|row| {
                            (0..dout)
                                .map(|o| {
                                    (0..din)
                                        .map(|i| f64::from(row[i]) * wmat[i * dout + o])
                                        .sum::<f64>()
                                        .tanh() as f32
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect()
                };
                let c = synth(&mut r, pairs, din, noise, Distance::L2);
                let f = f64::from(graph_loss(
                    &model(&c.src),
                    &model(&c.tgt),
                    &c.w,
                    dout,
                    Distance::L2,
                    Axis::None,
                ));
                for (k, levels) in [255, 127].into_iter().enumerate() {
                    let deq = |v: &[f32]| -> Vec<f32> {
                        v.chunks(din)
                            .flat_map(|row| {
                                let s = shared_scale(row, &[], levels);
                                row.iter()
                                    .map(move |&x| (f64::from(quantize(x, s, levels)) * s) as f32)
                                    .collect::<Vec<_>>()
                            })
                            .collect()
                    };
                    let q = f64::from(graph_loss(
                        &model(&deq(&c.src)),
                        &model(&deq(&c.tgt)),
                        &c.w,
                        dout,
                        Distance::L2,
                        Axis::None,
                    ));
                    errs[k].push(((q - f) / f).abs());
                }
            }
            let mp =
                |v: &Vec<f64>| format!("{:.1e}/{:.1e}", pct(v.clone(), 0.5), pct(v.clone(), 0.95));
            eprintln!("{:>6} | {:>17} {:>17}", noise, mp(&errs[0]), mp(&errs[1]));
        }
    }

    use crate::{Adjacency, Example, FeatureTable, Graph};

    /// SM9 code of an edge weight in [0, 1] at scale 1/255 (Codec9 with a
    /// unit calibration), written back as the f64 the code decodes to.
    fn on_grid(w: f64) -> f64 {
        f64::from(quantize(w as f32, 1.0 / 255.0, 255)) / 255.0
    }

    fn order(tsv: &str, seeds: &[Example], nbrs: &[Example]) -> Vec<Vec<u32>> {
        let g = Graph::from_tsv(tsv).unwrap();
        let t = FeatureTable::new(seeds, nbrs).unwrap();
        let a = Adjacency::build(&g, &t);
        (0..a.seeds())
            .map(|s| a.neighbors(s, None).iter().map(|x| x.row).collect())
            .collect()
    }

    fn ex(id: &str) -> Example {
        Example {
            id: id.into(),
            tag: "t".into(),
            x: 0.0,
        }
    }

    #[test]
    fn sm9_weights_keep_nsl_order_on_grid_and_can_break_it_off_grid() {
        let (seeds, nbrs) = ([ex("A")], [ex("B"), ex("C"), ex("D")]);
        // on grid: distinct codes 200, 210, 255 -> order unchanged (silent case)
        let on = "A\tB\t0.7843137254901961\nA\tC\t0.8235294117647058\nA\tD\t1.0\n";
        let quantized = |tsv: &str| {
            tsv.lines()
                .map(|l| {
                    let c: Vec<&str> = l.split('\t').collect();
                    format!(
                        "{}\t{}\t{:.17}\n",
                        c[0],
                        c[1],
                        on_grid(c[2].parse().unwrap())
                    )
                })
                .collect::<String>()
        };
        assert_eq!(
            order(on, &seeds, &nbrs),
            order(&quantized(on), &seeds, &nbrs)
        );
        // off grid: 0.8000 and 0.8015 share code 204; NSL puts C first, the
        // code tie falls to the ID and puts D first (can-fire case)
        let off = "A\tC\t0.8015\nA\tD\t0.8000\n";
        assert_eq!(on_grid(0.8015), on_grid(0.8));
        assert_ne!(
            order(off, &seeds, &nbrs),
            order(&quantized(off), &seeds, &nbrs)
        );
    }

    /// Report, not a gate: NSL build_graph-shaped weights (similarity >= 0.8,
    /// written with %f), power-law degrees. How often does SM9-quantizing the
    /// weights change which neighbors are selected, or their order?
    #[test]
    #[ignore]
    fn sm9_weight_order_probe() {
        let mut r = Rng(0xA076_1D64_78BD_642F);
        let (n_seed, n_nbr) = (2000usize, 4000usize);
        let seeds: Vec<Example> = (0..n_seed).map(|i| ex(&format!("s{i:05}"))).collect();
        let nbrs: Vec<Example> = (0..n_nbr).map(|i| ex(&format!("n{i:05}"))).collect();
        for (label, lo) in [("sim>=0.8", 0.8), ("sim>=0.95", 0.95)] {
            let (mut exact, mut quant) = (String::new(), String::new());
            let mut max_rel = 0.0f64;
            for s in 0..n_seed {
                let deg = 1 + (64usize >> (r.u() * 7.0) as usize);
                for _ in 0..deg {
                    let d = (r.u() * n_nbr as f64) as usize;
                    let w = format!("{:.6}", lo + (1.0 - lo) * r.u())
                        .parse::<f64>()
                        .unwrap();
                    exact.push_str(&format!("s{s:05}\tn{d:05}\t{w:.6}\n"));
                    quant.push_str(&format!("s{s:05}\tn{d:05}\t{:.17}\n", on_grid(w)));
                    max_rel = max_rel.max((on_grid(w) - w).abs() / w);
                }
            }
            let (a, b) = (order(&exact, &seeds, &nbrs), order(&quant, &seeds, &nbrs));
            let mut line = format!("{label:10} max weight rel err {max_rel:.2e} |");
            for k in [1usize, 4, 16, usize::MAX] {
                let cut = |v: &Vec<u32>| v.iter().take(k).copied().collect::<Vec<_>>();
                let (mut set, mut ord) = (0, 0);
                for (x, y) in a.iter().zip(&b) {
                    let (cx, cy) = (cut(x), cut(y));
                    let (mut sx, mut sy) = (cx.clone(), cy.clone());
                    sx.sort_unstable();
                    sy.sort_unstable();
                    set += usize::from(sx != sy);
                    ord += usize::from(cx != cy);
                }
                let k = if k == usize::MAX {
                    "all".to_string()
                } else {
                    k.to_string()
                };
                line += &format!(
                    " k={k}: set {:.1}% order {:.1}% |",
                    100.0 * set as f64 / n_seed as f64,
                    100.0 * ord as f64 / n_seed as f64
                );
            }
            eprintln!("{line}");
        }
    }

    /// Report, not a gate: does subtracting the expected quantization bias
    /// (two independent uniform errors, 2 * s^2 / 12 per element) rescue L2?
    #[test]
    #[ignore]
    fn l2_bias_correction_probe() {
        let mut r = Rng(0x2545_F491_4F6C_DD1D);
        eprintln!(
            "{:>4} {:>6} | {:>17} {:>17}",
            "dim", "noise", "sm9 med/p95", "corrected med/p95"
        );
        for dim in [32usize, 128] {
            for noise in [0.14, 0.05, 0.01, 0.001] {
                let (mut raw, mut cor) = (vec![], vec![]);
                for _ in 0..400 {
                    let c = synth(&mut r, 16, dim, noise, Distance::L2);
                    let f = f64::from(graph_loss(&c.src, &c.tgt, &c.w, c.dim, c.dist, c.axis));
                    let q = f64::from(loss_shared(&c, 255, false));
                    let s = shared_scale(&c.src, &c.tgt, 255);
                    let bias = 2.0 * s * s / 12.0; // per element, before the mean
                    raw.push(((q - f) / f).abs());
                    cor.push(((q - bias - f) / f).abs());
                }
                let mp = |v: &Vec<f64>| {
                    format!("{:.1e}/{:.1e}", pct(v.clone(), 0.5), pct(v.clone(), 0.95))
                };
                eprintln!(
                    "{:>4} {:>6} | {:>17} {:>17}",
                    dim,
                    noise,
                    mp(&raw),
                    mp(&cor)
                );
            }
        }
    }

    #[test]
    fn the_softmax_table_puts_the_max_at_one_and_keeps_exact_ratios() {
        let (codes, s) = ([3, -40, 17, 17, 0], 0.05);
        let p = softmax_table_row(&codes, s, None);
        // ratios against the max are exactly the table entries exp(s*(q-qmax))
        for (i, &q) in codes.iter().enumerate() {
            let want = (s * f64::from(q - 17)).exp();
            assert!((p[i] / p[2] - want).abs() < 1e-15, "{i}");
        }
        // and it equals the f64 softmax of the dequantized codes
        let deq: Vec<f32> = codes.iter().map(|&q| (f64::from(q) * s) as f32).collect();
        for (a, b) in p.iter().zip(softmax_row(&deq)) {
            assert!((a - b).abs() < 1e-7);
        }
        // the max is subtracted in code space: logits whose exp overflows f64
        // (6 * 127 = 762 > 709) still give a finite row
        let big = softmax_table_row(&[127, 120], 6.0, None);
        assert!(big.iter().all(|x| x.is_finite()), "{big:?}");
    }

    #[test]
    fn residual_coding_keeps_a_near_duplicate_loss_that_endpoint_coding_loses() {
        let mut r = Rng(0x1357_9BDF_2468_ACE0);
        let c = synth(&mut r, 16, 32, 0.001, Distance::L2);
        let f = f64::from(graph_loss(&c.src, &c.tgt, &c.w, c.dim, c.dist, c.axis));
        let rel = |x: f32| ((f64::from(x) - f) / f).abs();
        // can-fire: both endpoints on one int8 scale lose the residual
        assert!(rel(loss_shared(&c, 127, false)) > 1.0);
        // silent: source + residual, each int8, keep it within 1%
        assert!(rel(loss_residual(&c, 127)) < 1e-2);
    }

    #[test]
    fn the_int8_output_grid_rounds_small_probabilities_to_zero() {
        // can-fire: a probability below 1/512 lands on code 0
        let p = softmax_table_row(&[127, -127], 0.05, Some(256));
        assert_eq!(p[1], 0.0);
        // the top code saturates at 255/256, as an int8 output (zp -128) does
        assert_eq!(p[0], 255.0 / 256.0);
        assert!(p.iter().all(|&x| (x * 256.0).fract() == 0.0));
        // silent twin: without an output grid nothing reaches zero
        assert!(softmax_table_row(&[127, -127], 0.05, None)[1] > 0.0);
    }

    #[test]
    fn the_family_gamma_mirror_round_trips_like_bgz_tensor() {
        // bgz-tensor fisher_z::tests::encode_decode_identity, same inputs
        let g = FamilyGamma::from_cosines(&[-0.3, -0.1, 0.0, 0.1, 0.5]);
        for c in [-0.3f32, -0.1, 0.0, 0.1, 0.5] {
            assert!((c - g.decode(g.encode(c))).abs() < 0.02, "{c}");
        }
        assert_eq!(g.encode(-0.3), -127);
        // truncation, not rounding: -5.78 encodes to -5 (rounding gives -6)
        assert_eq!(g.encode(0.1), -5);
    }

    #[test]
    fn a_palette_needle_reads_one_from_the_address_not_the_table() {
        let mut r = Rng(0x0BAD_5EED_1234_5678);
        let rows: Vec<f32> = (0..64 * 8).map(|_| r.n() as f32).collect();
        let p = Palette::train(&rows, 8, 16, 4);
        let k = p.centroids.len();
        for a in 0..k {
            assert_eq!(p.pair(a, a, Lut::FisherZ), 1.0);
            // the table's own diagonal is not 1: atanh is clamped at 0.9999
            assert!(f64::from(p.gamma.decode(p.lut_fz[a * k + a])) < 1.0);
        }
        // off the diagonal the table is read
        assert_eq!(p.pair(0, 1, Lut::F32), f64::from(p.lut_f32[1]));
    }

    /// Report, not a gate: KL/JS graph loss with logits quantized (shared
    /// scale), softmax through the TFLM-shaped table, divergence in f64.
    /// Arms: input codes only (in9 / in8), plus an int8 output grid (o256,
    /// TFLM int8 softmax) or an int16 output grid (o32768). Golden cases are
    /// scored against TF; the sweep against the f64 reference.
    #[test]
    #[ignore]
    fn softmax_divergence_probe() {
        let arms: [(&str, i32, Option<u32>); 5] = [
            ("in9", 255, None),
            ("in8", 127, None),
            ("in9+o256", 255, Some(256)),
            ("in8+o256", 127, Some(256)),
            ("in8+o32768", 127, Some(32768)),
        ];
        eprintln!("golden (vs TF; TF's own cancellation budget shown as rel):");
        for c in cases()
            .iter()
            .filter(|c| matches!(c.dist, Distance::Kl | Distance::Js))
        {
            let tf = f64::from(c.tf);
            let mut line = format!(
                "{:17} budget {:.1e} |",
                c.name,
                f64::from(f32::EPSILON) * divergence_magnitude(c) / tf
            );
            for (n, l, o) in arms {
                let q = f64::from(loss_softmax_table(c, l, o));
                line += &format!(" {n} {:.1e}", ((q - tf) / tf).abs());
            }
            let q = f64::from(loss_residual(c, 127));
            line += &format!(" in8+res8 {:.1e}", ((q - tf) / tf).abs());
            eprintln!("{line}");
        }
        eprintln!("sweep, 400 seeds, 16 pairs x 10 classes, logits ~ 2N(0,1), median/p95 rel err:");
        let mut r = Rng(0x5851_F42D_4C95_7F2D);
        for dist in [Distance::Kl, Distance::Js] {
            for noise in [0.5, 0.14, 0.05, 0.01, 0.001] {
                let mut errs = vec![vec![]; arms.len()];
                let mut res = vec![];
                for _ in 0..400 {
                    let mut c = synth(&mut r, 16, 10, noise, dist);
                    c.axis = Axis::Last;
                    for x in c.src.iter_mut().chain(c.tgt.iter_mut()) {
                        *x *= 2.0;
                    }
                    let f = f64::from(graph_loss(&c.src, &c.tgt, &c.w, c.dim, c.dist, c.axis));
                    for (k, (_, l, o)) in arms.iter().enumerate() {
                        let q = f64::from(loss_softmax_table(&c, *l, *o));
                        errs[k].push(((q - f) / f).abs());
                    }
                    let q = f64::from(loss_residual(&c, 127));
                    res.push(((q - f) / f).abs());
                }
                let mut line = format!("{:4} noise {:>6} |", format!("{dist:?}"), noise);
                for (k, (n, _, _)) in arms.iter().enumerate() {
                    line += &format!(
                        " {n} {:.1e}/{:.1e}",
                        pct(errs[k].clone(), 0.5),
                        pct(errs[k].clone(), 0.95)
                    );
                }
                line += &format!(
                    " in8+res8 {:.1e}/{:.1e}",
                    pct(res.clone(), 0.5),
                    pct(res.clone(), 0.95)
                );
                eprintln!("{line}");
            }
        }
    }

    /// Clustered rows: `classes` unit prototypes, each row a prototype plus
    /// `spread` noise. Targets add `noise` on top, as NSL neighbors do.
    fn clustered(r: &mut Rng, protos: &[Vec<f64>], pairs: usize, spread: f64, noise: f64) -> Case {
        let dim = protos[0].len();
        let (mut src, mut tgt) = (vec![], vec![]);
        for _ in 0..pairs {
            let p = &protos[(r.u() * protos.len() as f64) as usize];
            for &x in p {
                let v = x + spread * r.n();
                src.push(v as f32);
                tgt.push((v + noise * r.n()) as f32);
            }
        }
        let w = (0..pairs).map(|_| (0.5 + 0.5 * r.u()) as f32).collect();
        Case {
            name: String::new(),
            dist: Distance::Cosine,
            axis: Axis::Last,
            dim,
            src,
            tgt,
            w,
            tf: 0.0,
        }
    }

    /// Report, not a gate: NSL cosine graph loss read from a palette256
    /// (needle per row, pair cosine from the LUT) vs the f32 reference.
    /// LUT readings: f32 (codebook error only), Fisher-z i8 (FamilyGamma), and
    /// an affine i8 control with the same fitted range. "needle%" is the share
    /// of pairs whose two rows land on the same centroid, which read 1.0.
    #[test]
    #[ignore]
    fn palette_cosine_probe() {
        let mut r = Rng(0x94D0_49BB_1331_11EB);
        let (dim, classes, k) = (32usize, 64usize, 256usize);
        let protos: Vec<Vec<f64>> = (0..classes)
            .map(|_| (0..dim).map(|_| r.n() / (dim as f64).sqrt()).collect())
            .collect();
        for spread in [0.05, 0.15] {
            let train = clustered(&mut r, &protos, 8192, spread, 0.0);
            let pal = Palette::train(&train.src, dim, k, 12);
            eprintln!(
                "spread {spread}: gamma z_min {:.3} z_range {:.3}",
                pal.gamma.z_min, pal.gamma.z_range
            );
            for noise in [0.14, 0.05, 0.01, 0.001] {
                let mut errs = [vec![], vec![], vec![]];
                let mut needle = 0.0;
                for _ in 0..200 {
                    let c = clustered(&mut r, &protos, 16, spread, noise);
                    let f = f64::from(graph_loss(&c.src, &c.tgt, &c.w, c.dim, c.dist, c.axis));
                    for (j, lut) in [Lut::F32, Lut::FisherZ, Lut::Affine]
                        .into_iter()
                        .enumerate()
                    {
                        let q = f64::from(loss_palette_cosine(&c, &pal, lut));
                        errs[j].push(((q - f) / f).abs());
                    }
                    needle += (0..16)
                        .filter(|&i| {
                            let rr = i * dim..(i + 1) * dim;
                            pal.index(&c.src[rr.clone()]) == pal.index(&c.tgt[rr])
                        })
                        .count() as f64
                        / 16.0;
                }
                let mp = |v: &Vec<f64>| {
                    format!("{:.1e}/{:.1e}", pct(v.clone(), 0.5), pct(v.clone(), 0.95))
                };
                eprintln!(
                    "  noise {:>6} needle {:>5.1}% | lut_f32 {} fisher_z {} affine {}",
                    noise,
                    100.0 * needle / 200.0,
                    mp(&errs[0]),
                    mp(&errs[1]),
                    mp(&errs[2])
                );
            }
            // LUT encoding error alone, over the off-diagonal pairs
            let kk = pal.centroids.len();
            let (mut ez, mut ea, mut ez_hi, mut ea_hi) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let (mut n, mut nhi) = (0usize, 0usize);
            for i in 0..kk {
                for j in (i + 1)..kk {
                    let c = f64::from(pal.lut_f32[i * kk + j]);
                    let dz = (f64::from(pal.gamma.decode(pal.lut_fz[i * kk + j])) - c).abs();
                    let da = (f64::from(pal.affine.decode(pal.lut_u8[i * kk + j])) - c).abs();
                    ez = ez.max(dz);
                    ea = ea.max(da);
                    n += 1;
                    if c > 0.9 {
                        ez_hi = ez_hi.max(dz);
                        ea_hi = ea_hi.max(da);
                        nhi += 1;
                    }
                }
            }
            eprintln!(
                "  LUT max abs err: fisher_z {ez:.2e} affine {ea:.2e} | cos>0.9 ({nhi}/{n} pairs): fisher_z {ez_hi:.2e} affine {ea_hi:.2e}"
            );
        }
    }

    /// Report, not a gate: the edge-weight encodings against NSL's order on
    /// build_graph-shaped weights. SM9 at a fixed 1/255 scale (Codec9 unit
    /// calibration), affine u8 fitted to the observed range, and FamilyGamma
    /// (Fisher-z i8) fitted to the observed range.
    #[test]
    #[ignore]
    fn weight_encoding_order_probe() {
        let mut r = Rng(0xA076_1D64_78BD_642F);
        let (n_seed, n_nbr) = (2000usize, 4000usize);
        let seeds: Vec<Example> = (0..n_seed).map(|i| ex(&format!("s{i:05}"))).collect();
        let nbrs: Vec<Example> = (0..n_nbr).map(|i| ex(&format!("n{i:05}"))).collect();
        for (label, lo) in [("sim>=0.8", 0.8), ("sim>=0.95", 0.95)] {
            let mut edges = vec![];
            for s in 0..n_seed {
                let deg = 1 + (64usize >> (r.u() * 7.0) as usize);
                for _ in 0..deg {
                    let d = (r.u() * n_nbr as f64) as usize;
                    let w = format!("{:.6}", lo + (1.0 - lo) * r.u())
                        .parse::<f64>()
                        .unwrap();
                    edges.push((s, d, w));
                }
            }
            let ws: Vec<f32> = edges.iter().map(|e| e.2 as f32).collect();
            let fz = FamilyGamma::from_cosines(&ws);
            let af = Affine8::from_values(&ws);
            let tsv = |enc: &dyn Fn(f64) -> f64| -> String {
                edges
                    .iter()
                    .map(|&(s, d, w)| format!("s{s:05}\tn{d:05}\t{:.17}\n", enc(w)))
                    .collect()
            };
            let exact = order(&tsv(&|w| w), &seeds, &nbrs);
            type Enc<'a> = Box<dyn Fn(f64) -> f64 + 'a>;
            let encs: [(&str, Enc); 3] = [
                ("sm9_1/255", Box::new(on_grid)),
                (
                    "affine_u8",
                    Box::new(|w| f64::from(af.decode(af.encode(w as f32)))),
                ),
                (
                    "fisher_z_i8",
                    Box::new(|w| f64::from(fz.decode(fz.encode(w as f32)))),
                ),
            ];
            for (name, enc) in encs.iter() {
                let b = order(&tsv(enc.as_ref()), &seeds, &nbrs);
                let codes = {
                    let mut v: Vec<u64> = edges.iter().map(|e| enc(e.2).to_bits()).collect();
                    v.sort_unstable();
                    v.dedup();
                    v.len()
                };
                let mut line = format!("{label:10} {name:12} codes {codes:>3} |");
                for k in [1usize, 4, usize::MAX] {
                    let cut = |v: &Vec<u32>| v.iter().take(k).copied().collect::<Vec<_>>();
                    let (mut set, mut ord) = (0, 0);
                    for (x, y) in exact.iter().zip(&b) {
                        let (cx, cy) = (cut(x), cut(y));
                        let (mut sx, mut sy) = (cx.clone(), cy.clone());
                        sx.sort_unstable();
                        sy.sort_unstable();
                        set += usize::from(sx != sy);
                        ord += usize::from(cx != cy);
                    }
                    let kn = if k == usize::MAX {
                        "all".into()
                    } else {
                        k.to_string()
                    };
                    line += &format!(
                        " k={kn}: set {:.1}% order {:.1}% |",
                        100.0 * set as f64 / n_seed as f64,
                        100.0 * ord as f64 / n_seed as f64
                    );
                }
                eprintln!("{line}");
            }
        }
    }

    fn gumbel_sample(r: &mut Rng, n: usize, mu: f64, beta: f64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let g = -(-r.u().max(1e-300).ln()).ln();
                (mu + beta * g).tanh() as f32
            })
            .collect()
    }

    fn entropy_bits(codes: impl Iterator<Item = u64>) -> f64 {
        let mut h = std::collections::HashMap::<u64, usize>::new();
        let mut n = 0usize;
        for c in codes {
            *h.entry(c).or_default() += 1;
            n += 1;
        }
        h.values()
            .map(|&k| {
                let p = k as f64 / n as f64;
                -p * p.log2()
            })
            .sum()
    }

    #[test]
    fn euler_gamma_puts_the_gumbel_location_where_the_sample_has_it() {
        let mut r = Rng(0xE0E0_1234_5678_9ABC);
        let v = gumbel_sample(&mut r, 50_000, 0.6, 0.25);
        let e = Envelope::fit(Shape::ZGumbel, &v, 255);
        // mean of mu + beta*G is mu + EULER_GAMMA*beta; the fit undoes it
        assert!((f64::from(e.a) - 0.6).abs() < 0.01, "mu {}", e.a);
        assert!((f64::from(e.b) - 0.25).abs() < 0.01, "beta {}", e.b);
    }

    #[test]
    fn the_gumbel_envelope_flattens_a_gumbel_sample_and_not_a_flat_one() {
        let mut r = Rng(0x2468_1357_ACE0_BDF1);
        let ent = |v: &[f32], sh: Shape| {
            let e = Envelope::fit(sh, v, 255);
            entropy_bits(v.iter().map(|&c| u64::from(e.encode(c))))
        };
        // can-fire: skewed (Gumbel) z, the Gumbel envelope uses the codes evenly
        let g = gumbel_sample(&mut r, 50_000, 0.6, 0.25);
        let (lin, gum) = (ent(&g, Shape::ZLinear), ent(&g, Shape::ZGumbel));
        assert!(gum > 7.9 && gum > lin + 0.5, "gumbel {gum} linear {lin}");
        // silent: flat z, the linear envelope is already flat and wins
        let flat: Vec<f32> = (0..50_000)
            .map(|_| (0.2 + 1.0 * r.u()).tanh() as f32)
            .collect();
        let (lin, gum) = (ent(&flat, Shape::ZLinear), ent(&flat, Shape::ZGumbel));
        assert!(lin > gum, "flat: linear {lin} gumbel {gum}");
    }

    #[test]
    fn a_left_skewed_sample_flips_the_sign_of_beta_and_stays_monotone() {
        let mut r = Rng(0x0F0F_F0F0_1111_2222);
        let v: Vec<f32> = gumbel_sample(&mut r, 20_000, 0.6, 0.25)
            .iter()
            .map(|&c| (1.2 - f64::from(atanh_clamp(c))).tanh() as f32)
            .collect();
        let e = Envelope::fit(Shape::ZGumbel, &v, 511);
        assert!(e.b < 0.0, "beta {}", e.b);
        let mut s = v.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(s.windows(2).all(|w| e.encode(w[0]) <= e.encode(w[1])));
        assert!(s
            .windows(2)
            .all(|w| e.decode(e.encode(w[0])) <= e.decode(e.encode(w[1]))));
    }

    fn embed_edges(r: &mut Rng, thr: f64) -> Vec<(usize, usize, f64)> {
        let (dim, clusters, per) = (32usize, 200usize, 20usize);
        let unit = |v: Vec<f64>| {
            let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v.into_iter().map(|x| x / n).collect::<Vec<f64>>()
        };
        let protos: Vec<Vec<f64>> = (0..clusters)
            .map(|_| unit((0..dim).map(|_| r.n()).collect()))
            .collect();
        let pt = |p: &Vec<f64>, r: &mut Rng| unit(p.iter().map(|&x| x + 0.08 * r.n()).collect());
        let nbrs: Vec<Vec<f64>> = (0..clusters * per)
            .map(|i| pt(&protos[i / per], r))
            .collect();
        let mut out = vec![];
        for s in 0..2000 {
            let c = (r.u() * clusters as f64) as usize;
            let sv = pt(&protos[c], r);
            for (j, nb) in nbrs.iter().enumerate().skip(c * per).take(per) {
                let cos: f64 = sv.iter().zip(nb).map(|(a, b)| a * b).sum();
                if cos >= thr {
                    out.push((s, j, format!("{cos:.6}").parse().unwrap()));
                }
            }
        }
        out
    }

    /// Report, not a gate: can the 8-byte envelope deskew the value
    /// distribution, and what does a 9-bit (SM9-reach) code add over i8?
    /// Data: uniform build_graph weights, cosine weights of clustered
    /// embeddings (build_graph-shaped, sim >= 0.8), and the off-diagonal pair
    /// cosines of a palette512 (the palette512:palette512 LUT).
    #[test]
    #[ignore]
    fn deskew_envelope_probe() {
        let mut r = Rng(0x7F4A_7C15_9E37_79B9);
        let (n_seed, n_nbr) = (2000usize, 4000usize);
        let seeds: Vec<Example> = (0..n_seed).map(|i| ex(&format!("s{i:05}"))).collect();
        let nbrs: Vec<Example> = (0..n_nbr).map(|i| ex(&format!("n{i:05}"))).collect();
        let mut uniform = vec![];
        for s in 0..n_seed {
            let deg = 1 + (64usize >> (r.u() * 7.0) as usize);
            for _ in 0..deg {
                let d = (r.u() * n_nbr as f64) as usize;
                uniform.push((
                    s,
                    d,
                    format!("{:.6}", 0.8 + 0.2 * r.u()).parse::<f64>().unwrap(),
                ));
            }
        }
        let embed = embed_edges(&mut r, 0.8);
        let train = {
            let protos: Vec<Vec<f64>> = (0..64)
                .map(|_| (0..32).map(|_| r.n() / 32f64.sqrt()).collect())
                .collect();
            clustered(&mut r, &protos, 8192, 0.05, 0.0)
        };
        let pal = Palette::train(&train.src, 32, 512, 12);
        let k = pal.centroids.len();
        let lut: Vec<(usize, usize, f64)> = (0..k)
            .flat_map(|i| ((i + 1)..k).map(move |j| (i, j)))
            .map(|(i, j)| (i, j, f64::from(pal.lut_f32[i * k + j])))
            .collect();
        for (label, edges, ordered) in [
            ("uniform w>=0.8", &uniform, true),
            ("embed w>=0.8", &embed, true),
            ("lut512 pairs", &lut, false),
        ] {
            let vals: Vec<f32> = edges.iter().map(|e| e.2 as f32).collect();
            let mut sorted = vals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p90 = sorted[sorted.len() * 9 / 10];
            let m = vals.iter().map(|&x| f64::from(x)).sum::<f64>() / vals.len() as f64;
            let sk = {
                let z: Vec<f64> = vals.iter().map(|&v| f64::from(atanh_clamp(v))).collect();
                let zm = z.iter().sum::<f64>() / z.len() as f64;
                let sd = (z.iter().map(|x| (x - zm).powi(2)).sum::<f64>() / z.len() as f64).sqrt();
                z.iter().map(|x| ((x - zm) / sd).powi(3)).sum::<f64>() / z.len() as f64
            };
            eprintln!("{label}: n {} mean {m:.4} z-skew {sk:.2}", vals.len());
            let fg = FamilyGamma::from_cosines(&vals);
            let env = |sh, l| Envelope::fit(sh, &vals, l);
            let (l8, g8, l9, g9) = (
                env(Shape::ZLinear, 255),
                env(Shape::ZGumbel, 255),
                env(Shape::ZLinear, 511),
                env(Shape::ZGumbel, 511),
            );
            let (q8, q9) = (Quantiles::fit(&vals, 255), Quantiles::fit(&vals, 511));
            type Enc<'a> = Box<dyn Fn(f32) -> (u64, f32) + 'a>;
            let encs: Vec<(&str, Enc)> = vec![
                (
                    "fisher_z_i8",
                    Box::new(|c| {
                        let e = fg.encode(c);
                        (e as u8 as u64, fg.decode(e))
                    }),
                ),
                (
                    "zlin_8",
                    Box::new(|c| {
                        let e = l8.encode(c);
                        (u64::from(e), l8.decode(e))
                    }),
                ),
                (
                    "zgumbel_8",
                    Box::new(|c| {
                        let e = g8.encode(c);
                        (u64::from(e), g8.decode(e))
                    }),
                ),
                (
                    "quantile_8*",
                    Box::new(|c| {
                        let e = q8.encode(c);
                        (e as u64, q8.decode(e))
                    }),
                ),
                (
                    "zlin_9",
                    Box::new(|c| {
                        let e = l9.encode(c);
                        (u64::from(e), l9.decode(e))
                    }),
                ),
                (
                    "zgumbel_9",
                    Box::new(|c| {
                        let e = g9.encode(c);
                        (u64::from(e), g9.decode(e))
                    }),
                ),
                (
                    "quantile_9*",
                    Box::new(|c| {
                        let e = q9.encode(c);
                        (e as u64, q9.decode(e))
                    }),
                ),
            ];
            let exact = ordered.then(|| {
                let tsv: String = edges
                    .iter()
                    .map(|&(s, d, w)| format!("s{s:05}\tn{d:05}\t{w:.17}\n"))
                    .collect();
                order(&tsv, &seeds, &nbrs)
            });
            for (name, enc) in &encs {
                let ent = entropy_bits(vals.iter().map(|&c| enc(c).0));
                let errs: Vec<f64> = vals
                    .iter()
                    .map(|&c| (f64::from(enc(c).1) - f64::from(c)).abs())
                    .collect();
                let mean = errs.iter().sum::<f64>() / errs.len() as f64;
                let top: Vec<f64> = vals
                    .iter()
                    .zip(&errs)
                    .filter(|(v, _)| **v >= p90)
                    .map(|(_, e)| *e)
                    .collect();
                let topm = top.iter().sum::<f64>() / top.len() as f64;
                let mut line = format!(
                    "  {name:12} entropy {ent:.2} b | abs err mean {mean:.2e} top10% {topm:.2e} max {:.2e}",
                    errs.iter().copied().fold(0.0, f64::max)
                );
                if let Some(a) = &exact {
                    let tsv: String = edges
                        .iter()
                        .map(|&(s, d, w)| {
                            format!("s{s:05}\tn{d:05}\t{:.17}\n", f64::from(enc(w as f32).1))
                        })
                        .collect();
                    let b = order(&tsv, &seeds, &nbrs);
                    for kk in [1usize, 4] {
                        let ch = a
                            .iter()
                            .zip(&b)
                            .filter(|(x, y)| x.iter().take(kk).ne(y.iter().take(kk)))
                            .count();
                        line +=
                            &format!(" | top{kk} order {:.1}%", 100.0 * ch as f64 / n_seed as f64);
                    }
                }
                eprintln!("{line}");
            }
        }
        eprintln!("(* quantile = ceiling, needs 255/511 floats of metadata, not 8 bytes)");
    }

    /// Report, not a gate: palette256 vs palette512 needles on the same
    /// clustered pairs. Share of pairs reading the address 1.0, and the NSL
    /// cosine loss error with an exact (f32) pair LUT.
    #[test]
    #[ignore]
    fn palette512_needle_probe() {
        let mut r = Rng(0x94D0_49BB_1331_11EB);
        let protos: Vec<Vec<f64>> = (0..64)
            .map(|_| (0..32).map(|_| r.n() / 32f64.sqrt()).collect())
            .collect();
        let train = clustered(&mut r, &protos, 8192, 0.05, 0.0);
        let p256 = Palette::train(&train.src, 32, 256, 12);
        let p512 = Palette::train(&train.src, 32, 512, 12);
        for noise in [0.14, 0.05, 0.01] {
            let (mut n2, mut n5, mut e2, mut e5) = (0.0, 0.0, vec![], vec![]);
            for _ in 0..200 {
                let c = clustered(&mut r, &protos, 16, 0.05, noise);
                let f = f64::from(graph_loss(&c.src, &c.tgt, &c.w, c.dim, c.dist, c.axis));
                for (p, n, e) in [(&p256, &mut n2, &mut e2), (&p512, &mut n5, &mut e5)] {
                    *n += (0..16)
                        .filter(|&i| {
                            let rr = i * 32..(i + 1) * 32;
                            p.index(&c.src[rr.clone()]) == p.index(&c.tgt[rr])
                        })
                        .count() as f64
                        / 16.0;
                    e.push(((f64::from(loss_palette_cosine(&c, p, Lut::F32)) - f) / f).abs());
                }
            }
            eprintln!(
                "noise {noise:>5}: needle k256 {:.1}% k512 {:.1}% | loss rel err med k256 {:.2} k512 {:.2}",
                100.0 * n2 / 200.0,
                100.0 * n5 / 200.0,
                pct(e2, 0.5),
                pct(e5, 0.5)
            );
        }
    }
}
