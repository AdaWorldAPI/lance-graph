//! Evaluation of quantized NSL paths against the frozen TensorFlow oracle.
//!
//! Not a production codec. The 9-bit lane here mirrors ndarray
//! `hpc::codec9` (`EpiLane9`, `Codec9::encode_touched`): magnitude u8 plus a
//! presence bit, +0 canonical, ties away from zero, saturating at 255. The
//! difference is a scale chosen at runtime from the data, where Codec9 fixes
//! it at compile time.

use crate::loss::{graph_loss, Axis, Distance};

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
        };
        total += wi * row;
    }
    let present = match (c.dist, c.axis) {
        (Distance::Cosine, _) | (_, Axis::Last) => nonzero as f64,
        (_, Axis::None) => (nonzero * c.dim) as f64,
    };
    (total / present) as f32
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
        assert_eq!(cs.len(), 10);
        let mut worst = 0;
        for c in &cs {
            let got = graph_loss(&c.src, &c.tgt, &c.w, c.dim, c.dist, c.axis);
            let u = ulps(got, c.tf);
            eprintln!("{:16} tf {:>14e} ref {:>14e} ulp {u}", c.name, c.tf, got);
            worst = worst.max(u);
        }
        assert!(worst <= 4, "worst {worst} ulp");
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
        let mut r = Rng(0x9E37_79B9_7F4A_7C15);
        for dist in [Distance::L2, Distance::L1, Distance::Cosine] {
            for dim in [32usize, 128] {
                for noise in [0.5, 0.14, 0.05, 0.01, 0.001] {
                    let (mut e9, mut e8, mut r9, mut r8, mut g9, mut g8) =
                        (vec![], vec![], vec![], vec![], vec![], vec![]);
                    for _ in 0..400 {
                        let c = synth(&mut r, 16, dim, noise, dist);
                        let f = f64::from(graph_loss(&c.src, &c.tgt, &c.w, c.dim, c.dist, c.axis));
                        let rel = |x: f32| ((f64::from(x) - f) / f).abs();
                        e9.push(rel(loss_shared(&c, 255, false)));
                        e8.push(rel(loss_shared(&c, 127, false)));
                        r9.push(rel(loss_rowscale(&c, 255)));
                        r8.push(rel(loss_rowscale(&c, 127)));
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
                        "{:7} {:>4} {:>6} | {:>17} {:>17} {:>17} {:>17} | {:>8} {:>8}",
                        format!("{dist:?}"),
                        dim,
                        noise,
                        mp(&e9),
                        mp(&e8),
                        mp(&r9),
                        mp(&r8),
                        gp(&g9),
                        gp(&g8)
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
}
