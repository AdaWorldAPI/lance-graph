//! `gen <outdir>`: deterministic generator for fx_cases.txt and fc_cases.txt (Amendment 6.4 lab probe).
//!
//! fx_cases.txt:  `S a b` | `R x e` | `M x mult shift`        (read by the C harness `tflm64 fx`)
//! fc_cases.txt:  blocks `FC ...` / `IN` / `WS` / `W` / `B`    (read by `tflm64 fc`)
//! All f32 bit patterns are written as `0x%08x`.
//!
//! Scale choices (stated, per the spec):
//!   in_scale  cycles over IN_SCALES = [0.0471, 0.0198, 0.0833, 0.0125]
//!   w_scale   cycles over W_BASES   = [0.0071, 0.0134, 0.0029]; per-channel c uses base*(1 + c/16), per-tensor base
//!   out_scale = in_scale * w_scale[0] * (74*73*sqrt(in_dim)) / 64   (f32)
//!     -> one standard deviation of the accumulator (uniform a in -128..127, w in -127..127) maps to ~64 output
//!        codes, so random rows cover the whole int8 range and extreme rows (all -128 / all 127) saturate.
//!     real multiplier = in*w/out ~ 0.012..0.4 < 1 for every in_dim in the list.
//!   half-tie fixtures use in_scale 0.1875 (3/16) and w_scale 0.375 (3/8): their product 9/64 is exact in f32, so
//!     out_scale = 2*(in*w) / 4*(in*w) is exact and the real multiplier is exactly 0.5 / 0.25.

use int8sem::*;
use std::fmt::Write as _;
use std::fs;

const S_SET: [i32; 11] = [
    i32::MIN,
    i32::MIN + 1,
    -(1 << 30), // -2^31/2
    -(1 << 15),
    -1,
    0,
    1,
    1 << 15,
    1 << 30,
    i32::MAX - 1,
    i32::MAX,
];

const IN_ZPS: [i32; 4] = [0, -128, 37, -5];
const OUT_ZPS: [i32; 3] = [-128, 0, 11];
const IN_DIMS: [usize; 14] = [1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 256, 257];
const IN_SCALES: [f32; 4] = [0.0471, 0.0198, 0.0833, 0.0125];
const W_BASES: [f32; 3] = [0.0071, 0.0134, 0.0029];
const OUT_DIM: usize = 16;
const BATCHES: usize = 16;

fn rand_mix(rng: &mut Rng) -> i32 {
    match rng.below(4) {
        0 | 1 => rng.next_i32(),
        2 => rng.next_i32() >> rng.below(32) as u32,
        _ => {
            let b = S_SET[rng.below(S_SET.len() as u64) as usize] as i64;
            (b + rng.range_i32(-3, 3) as i64).clamp(i32::MIN as i64, i32::MAX as i64) as i32
        }
    }
}

fn gen_fx(rng: &mut Rng) -> (String, [usize; 3]) {
    let mut s = String::new();
    let mut cnt = [0usize; 3];

    // S: all boundary combos + 200,000 random pairs.
    for &a in &S_SET {
        for &b in &S_SET {
            let _ = writeln!(s, "S {} {}", a, b);
            cnt[0] += 1;
        }
    }
    for _ in 0..200_000 {
        let (a, b) = (rand_mix(rng), rand_mix(rng));
        let _ = writeln!(s, "S {} {}", a, b);
        cnt[0] += 1;
    }

    // R: every e in 0..=31; boundary x, every half tie +-(odd * 2^(e-1)) +- {0,1}, +-(2^e)+-{0,1}; 50,000 random per e.
    for e in 0..=31u32 {
        let mut xs: Vec<i64> = vec![i32::MIN as i64, i32::MIN as i64 + 1, -1, 0, 1, i32::MAX as i64];
        let mut bases: Vec<i64> = vec![1i64 << e];
        if e >= 1 {
            for j in 0..4i64 {
                bases.push((2 * j + 1) * (1i64 << (e - 1))); // odd multiples of 2^(e-1): exact half ties
            }
        }
        for &b in &bases {
            for sign in [-1i64, 1] {
                for d in [-1i64, 0, 1] {
                    xs.push(sign * b + d);
                }
            }
        }
        for x in xs {
            if x >= i32::MIN as i64 && x <= i32::MAX as i64 {
                let _ = writeln!(s, "R {} {}", x, e);
                cnt[1] += 1;
            }
        }
        for _ in 0..50_000 {
            let _ = writeln!(s, "R {} {}", rand_mix(rng), e);
            cnt[1] += 1;
        }
    }

    // M: x boundary x multipliers x shifts -31..=30 (skip x << max(shift,0) overflowing i32: TFLM's int32 multiply
    // would be UB there; ndarray wraps), plus 200,000 random.
    let xset: Vec<i32> = vec![
        i32::MIN,
        i32::MIN + 1,
        -(1 << 30),
        -(1 << 15),
        -12345,
        -1,
        0,
        1,
        12345,
        1 << 15,
        1 << 30,
        i32::MAX - 1,
        i32::MAX,
    ];
    let mut mset: Vec<i32> = vec![0, 1, 1 << 30, i32::MAX - 1, i32::MAX, 1_073_741_824 + 12_345];
    for _ in 0..8 {
        mset.push((rng.next_u32() >> 1) as i32);
    }
    let ok = |x: i32, sh: i32| -> bool {
        let v = (x as i64) << sh.max(0);
        v >= i32::MIN as i64 && v <= i32::MAX as i64
    };
    for &x in &xset {
        for &m in &mset {
            for sh in -31..=30i32 {
                if ok(x, sh) {
                    let _ = writeln!(s, "M {} {} {}", x, m, sh);
                    cnt[2] += 1;
                }
            }
        }
    }
    let mut n = 0;
    while n < 200_000 {
        let x = rand_mix(rng);
        let m = if rng.below(8) == 0 { 1_073_741_824 + rng.range_i32(-20_000, 20_000) } else { (rng.next_u32() >> 1) as i32 };
        let sh = rng.range_i32(-31, 30);
        if ok(x, sh) {
            let _ = writeln!(s, "M {} {} {}", x, m, sh);
            n += 1;
            cnt[2] += 1;
        }
    }
    (s, cnt)
}

/// Activations: row 0 all -128, row 1 all 127, row 2 cycles through all 256 codes (j mod 256 - 128), rest random.
fn make_input(rng: &mut Rng, batches: usize, in_dim: usize) -> Vec<i8> {
    let mut v = vec![0i8; batches * in_dim];
    for r in 0..batches {
        for j in 0..in_dim {
            v[r * in_dim + j] = match r {
                0 => -128,
                1 => 127,
                2 => ((j % 256) as i32 - 128) as i8,
                _ => rng.range_i32(-128, 127) as i8,
            };
        }
    }
    v
}

struct Spec {
    id: String,
    ext: bool,
    pc: bool,
    batches: usize,
    in_dim: usize,
    in_scale: f32,
    w_base: f32,
    in_zp: i32,
    out_zp: i32,
    act_min: i32,
    act_max: i32,
}

fn build(rng: &mut Rng, sp: Spec, input: Vec<i8>) -> FcCase {
    let ws_f: Vec<f32> =
        (0..OUT_DIM).map(|c| if sp.pc { sp.w_base * (1.0 + 0.0625 * c as f32) } else { sp.w_base }).collect();
    // Conforming: -127..=127. Extended: >= 10% (random 15%, then topped up deterministically) are -128.
    let mut w = vec![0i8; OUT_DIM * sp.in_dim];
    for x in w.iter_mut() {
        *x = rng.range_i32(-127, 127) as i8;
    }
    if sp.ext {
        for x in w.iter_mut() {
            if rng.below(100) < 15 {
                *x = -128;
            }
        }
        let need = (w.len() + 9) / 10;
        let mut have = w.iter().filter(|&&x| x == -128).count();
        while have < need {
            let idx = rng.below(w.len() as u64) as usize;
            if w[idx] != -128 {
                w[idx] = -128;
                have += 1;
            }
        }
    }
    let bias: Vec<i32> = (0..OUT_DIM).map(|_| rng.range_i32(-4000, 4000)).collect();
    let std_acc = 74.0f64 * 73.0 * (sp.in_dim as f64).sqrt();
    let out_scale = ((sp.in_scale as f64) * (ws_f[0] as f64) * std_acc / 64.0) as f32;
    FcCase {
        id: sp.id,
        per_channel: sp.pc,
        batches: sp.batches,
        in_dim: sp.in_dim,
        out_dim: OUT_DIM,
        in_scale_bits: sp.in_scale.to_bits(),
        in_zp: sp.in_zp,
        out_scale_bits: out_scale.to_bits(),
        out_zp: sp.out_zp,
        act_min: sp.act_min,
        act_max: sp.act_max,
        input,
        ws: ws_f.iter().map(|f| f.to_bits()).collect(),
        w,
        bias,
    }
}

fn gen_fc(rng: &mut Rng) -> Vec<FcCase> {
    let mut cases: Vec<FcCase> = Vec::new();
    let mut ctr = 0usize;
    for ext in [false, true] {
        let pre = if ext { "X" } else { "C" };
        // (a)/(b) full cartesian: in_dim x in_zp x out_zp x per_channel
        for &in_dim in &IN_DIMS {
            for &izp in &IN_ZPS {
                for &ozp in &OUT_ZPS {
                    for pc in [false, true] {
                        let sp = Spec {
                            id: format!("{}{}_pc{}_izp{}_ozp{}", pre, in_dim, pc as i32, izp, ozp),
                            ext,
                            pc,
                            batches: BATCHES,
                            in_dim,
                            in_scale: IN_SCALES[ctr % IN_SCALES.len()],
                            w_base: W_BASES[ctr % W_BASES.len()],
                            in_zp: izp,
                            out_zp: ozp,
                            act_min: -128,
                            act_max: 127,
                        };
                        ctr += 1;
                        let input = make_input(rng, BATCHES, in_dim);
                        cases.push(build(rng, sp, input));
                    }
                }
            }
        }
        // narrow clamp cases (-100/90)
        for (in_dim, pc) in [(33usize, false), (33, true), (64, false), (64, true)] {
            let sp = Spec {
                id: format!("{}N{}_pc{}", pre, in_dim, pc as i32),
                ext,
                pc,
                batches: BATCHES,
                in_dim,
                in_scale: IN_SCALES[ctr % IN_SCALES.len()],
                w_base: W_BASES[ctr % W_BASES.len()],
                in_zp: 37,
                out_zp: 11,
                act_min: -100,
                act_max: 90,
            };
            ctr += 1;
            let input = make_input(rng, BATCHES, in_dim);
            cases.push(build(rng, sp, input));
        }
        // exhaustive activation cases: in_dim 1, 256 batch rows, row r carries code r-128
        for pc in [false, true] {
            for &izp in &IN_ZPS {
                let sp = Spec {
                    id: format!("{}E_pc{}_izp{}", pre, pc as i32, izp),
                    ext,
                    pc,
                    batches: 256,
                    in_dim: 1,
                    in_scale: IN_SCALES[ctr % IN_SCALES.len()],
                    w_base: W_BASES[ctr % W_BASES.len()],
                    in_zp: izp,
                    out_zp: 0,
                    act_min: -128,
                    act_max: 127,
                };
                ctr += 1;
                let input: Vec<i8> = (0..256).map(|r| (r as i32 - 128) as i8).collect();
                cases.push(build(rng, sp, input));
            }
        }
    }

    // (c) HALF-TIE fixtures. in_dim 1, weight 1, per-tensor. in_scale 3/16, w_scale 3/8 (exact products).
    let (ins, wsc) = (0.1875f32, 0.375f32);
    let prod = ins * wsc; // 9/64 exact
    let fams: [(&str, f32); 3] = [("half", 2.0 * prod), ("quarter", 4.0 * prod), ("bias", 2.0 * prod)];
    for (fam, out_scale) in fams {
        for &ozp in &[0i32, 3, -3] {
            let (input, bias, in_zp): (Vec<i8>, Vec<i32>, i32) = match fam {
                // real multiplier 1/2, a odd => acc odd => acc/2 = k + .5 (positive and negative acc)
                "half" => ((0..BATCHES as i32).map(|r| (2 * r - 15) as i8).collect(), vec![0; OUT_DIM], 0),
                // real multiplier 1/4, a = 4j+2 => acc/4 = k + .5
                "quarter" => ((0..BATCHES as i32).map(|r| (4 * (r - 8) + 2) as i8).collect(), vec![0; OUT_DIM], 0),
                // real multiplier 1/2, acc = (a - 2) + bias[c] with a even, bias odd => odd
                _ => (
                    (0..BATCHES as i32).map(|r| (2 * (r - 8)) as i8).collect(),
                    (0..OUT_DIM as i32).map(|c| 2 * c - 15).collect(),
                    2,
                ),
            };
            cases.push(FcCase {
                id: format!("T_{}_ozp{}", fam, ozp),
                per_channel: false,
                batches: BATCHES,
                in_dim: 1,
                out_dim: OUT_DIM,
                in_scale_bits: ins.to_bits(),
                in_zp,
                out_scale_bits: out_scale.to_bits(),
                out_zp: ozp,
                act_min: -128,
                act_max: 127,
                input,
                ws: vec![wsc.to_bits(); OUT_DIM],
                w: vec![1i8; OUT_DIM],
                bias,
            });
        }
    }
    cases
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: gen <outdir>");
        std::process::exit(2);
    }
    let dir = std::path::Path::new(&args[1]);
    fs::create_dir_all(dir).expect("create outdir");
    let mut rng = Rng::new(SEED);

    let (fx, cnt) = gen_fx(&mut rng);
    fs::write(dir.join("fx_cases.txt"), fx).expect("write fx_cases.txt");

    let cases = gen_fc(&mut rng);
    let mut s = String::new();
    for c in &cases {
        c.write_to(&mut s);
    }
    fs::write(dir.join("fc_cases.txt"), s).expect("write fc_cases.txt");

    let mut by_suite = std::collections::BTreeMap::new();
    for c in &cases {
        *by_suite.entry(c.suite()).or_insert(0usize) += c.batches * c.out_dim;
    }
    eprintln!(
        "gen: fx lines S={} R={} M={}; fc cases={} cells by suite={:?}",
        cnt[0],
        cnt[1],
        cnt[2],
        cases.len(),
        by_suite
    );
}
