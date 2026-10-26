//! `arms <dir> [artifacts.bin]`: differential report for the Amendment 6.4 probe.
//!
//! Reads <dir>/fx_cases.txt + tflm_fx.txt and <dir>/fc_cases.txt + tflm_fc.txt, runs the ndarray arms, the lab
//! oracle arms and the negative controls, and writes a hand-built JSON document to stdout and <dir>/arms64.json.
//! Nothing here is a production copy except `subtract_i8_to_u8_bias` (int8sem::lib, cited there).

use int8sem::*;
use ndarray::hpc::codec8::{
    Asym, Calibration, Codec8, Codec8Error, NaNAtMin, OnnxQ, Stores, Sym, TfliteAct, TfliteWeight, TiesAway, I8, U8,
};
use ndarray::simd::{
    multiply_by_quantized_multiplier_i32, quantize_multiplier, requantize_i32_to_i8, rounding_divide_by_pot_i32,
    srdhm_i32,
};
use std::collections::BTreeMap;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};

const DEFAULT_ARTIFACT: &str = "/tmp/claude-0/-home-user/08f9a4f9-dac0-54c1-91f7-ba3c8cb3e6cb/scratchpad/artifacts.bin";
/// First TANH SourceParity table: byte offset 14 + 64, 256 bytes, biased layout (spec).
const F2_TABLE_OFFSET: usize = 14 + 64;
const MAX_EXAMPLES: usize = 5;

fn read_opt(p: &std::path::Path) -> Option<String> {
    fs::read_to_string(p).ok()
}

fn not_run(why: &str) -> J {
    J::s(&format!("NOT RUN: {}", why))
}

// ───────────────────────────── meta ─────────────────────────────

fn meta() -> J {
    J::obj()
        .set("avx2", J::Bool(std::is_x86_feature_detected!("avx2")))
        .set("avx512f", J::Bool(std::is_x86_feature_detected!("avx512f")))
        .set("avx512vnni", J::Bool(std::is_x86_feature_detected!("avx512vnni")))
        .set("avxvnni", J::Bool(std::is_x86_feature_detected!("avxvnni")))
        .set("amx_available", J::Bool(ndarray::hpc::amx_matmul::amx_available()))
        .set("debug_assertions", J::Bool(cfg!(debug_assertions)))
        .set("feature_rd_vnni_dot", J::Bool(cfg!(feature = "rd")))
        .set(
            "note",
            J::s("debug_assertions=true means i32 overflow panics (lab bias copy, ndarray debug_asserts); release wraps"),
        )
}

// ───────────────────────────── 1. fx parity ─────────────────────────────

fn fx_parity(cases: &Option<String>, tflm: &Option<String>) -> J {
    let (Some(c), Some(t)) = (cases, tflm) else {
        return not_run("fx_cases.txt or tflm_fx.txt absent");
    };
    let (cl, tl) = match (parse_fx(c), parse_fx(t)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => return not_run(&format!("parse: {}", e)),
    };
    let mut total: BTreeMap<char, u64> = BTreeMap::new();
    let mut bad: BTreeMap<char, u64> = BTreeMap::new();
    let mut misaligned: BTreeMap<char, u64> = BTreeMap::new();
    let mut ex: BTreeMap<char, Vec<J>> = BTreeMap::new();
    for (i, cs) in cl.iter().enumerate() {
        *total.entry(cs.kind).or_default() += 1;
        let Some(tt) = tl.get(i) else {
            *misaligned.entry(cs.kind).or_default() += 1;
            continue;
        };
        let n = cs.args.len();
        if tt.kind != cs.kind || tt.args.len() != n + 1 || tt.args[..n] != cs.args[..] {
            *misaligned.entry(cs.kind).or_default() += 1;
            continue;
        }
        let tr = tt.args[n] as i32;
        let Some(nr) = fx_ndarray(cs.kind, &cs.args) else {
            *misaligned.entry(cs.kind).or_default() += 1;
            continue;
        };
        if nr != tr {
            *bad.entry(cs.kind).or_default() += 1;
            let e = ex.entry(cs.kind).or_default();
            if e.len() < MAX_EXAMPLES {
                e.push(
                    J::obj()
                        .set("args", J::Arr(cs.args.iter().map(|&x| J::i(x)).collect()))
                        .set("tflm", J::i(tr))
                        .set("ndarray", J::i(nr)),
                );
            }
        }
    }
    let mut o = J::obj();
    o.put("lines_cases", J::us(cl.len()));
    o.put("lines_tflm", J::us(tl.len()));
    for k in ['S', 'R', 'M'] {
        let name = match k {
            'S' => "S_srdhm",
            'R' => "R_rounding_divide_by_pot",
            _ => "M_multiply_by_quantized_multiplier",
        };
        o.put(
            name,
            J::obj()
                .set("cases", J::u(*total.get(&k).unwrap_or(&0)))
                .set("mismatches", J::u(*bad.get(&k).unwrap_or(&0)))
                .set("misaligned_or_missing_lines", J::u(*misaligned.get(&k).unwrap_or(&0)))
                .set("first_counterexamples", J::Arr(ex.remove(&k).unwrap_or_default())),
        );
    }
    o
}

// ───────────────────────────── 2. dot-product arms ─────────────────────────────

#[derive(Default)]
struct ArmStat {
    ran: u64,
    not_run: u64,
    cells: u64,
    bad: u64,
    reasons: BTreeMap<String, u64>,
}

impl ArmStat {
    fn to_j(&self, first: Vec<J>) -> J {
        let mut r = J::obj();
        for (k, v) in &self.reasons {
            r.put(k, J::u(*v));
        }
        J::obj()
            .set("cases_ran", J::u(self.ran))
            .set("cases_not_run", J::u(self.not_run))
            .set("not_run_reasons", r)
            .set("cells", J::u(self.cells))
            .set("mismatches_vs_i64_oracle", J::u(self.bad))
            .set("first_mismatches", J::Arr(first))
    }
}

fn dot_arms(cases: &[FcCase]) -> J {
    // key: suite -> arm -> stat
    let mut stats: BTreeMap<String, BTreeMap<&'static str, ArmStat>> = BTreeMap::new();
    let mut firsts: BTreeMap<(String, &'static str), Vec<J>> = BTreeMap::new();
    let mut tiers: BTreeMap<String, BTreeMap<&'static str, u64>> = BTreeMap::new();
    for c in cases {
        let suite = c.suite();
        if suite == "ties" {
            continue; // ties suite is only used by the FC four-arm comparison
        }
        let wt = transpose(&c.w, c.out_dim, c.in_dim);
        let d = DotInputs { a: &c.input, w_rows: &c.w, wt: &wt, m: c.batches, n: c.out_dim, k: c.in_dim };
        let orc = oracle(&d);
        *tiers
            .entry(suite.to_string())
            .or_default()
            .entry(matmul_tier(d.m, d.n, d.k))
            .or_default() += 1;
        for arm in DOT_ARMS {
            let st = stats.entry(suite.to_string()).or_default().entry(arm).or_default();
            match run_dot_arm(arm, &d) {
                ArmOut::NotRun(why) => {
                    st.not_run += 1;
                    let key = why.split(" m=").next().unwrap_or(&why).to_string();
                    *st.reasons.entry(key).or_default() += 1;
                }
                ArmOut::Cells(v) => {
                    st.ran += 1;
                    st.cells += v.len() as u64;
                    for (i, (&got, &want)) in v.iter().zip(orc.iter()).enumerate() {
                        if got != want {
                            st.bad += 1;
                            let f = firsts.entry((suite.to_string(), arm)).or_default();
                            if f.len() < MAX_EXAMPLES {
                                f.push(
                                    J::obj()
                                        .set("case", J::s(&c.id))
                                        .set("cell", J::us(i))
                                        .set("arm", J::i(got))
                                        .set("oracle", J::i(want)),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    let mut o = J::obj();
    for (suite, arms) in &stats {
        let mut so = J::obj();
        let mut tj = J::obj();
        if let Some(t) = tiers.get(suite) {
            for (k, v) in t {
                tj.put(k, J::u(*v));
            }
        }
        so.put("matmul_i8_to_i32_tier_by_case_count", tj);
        for (arm, st) in arms {
            let j = st.to_j(firsts.remove(&(suite.clone(), *arm)).unwrap_or_default());
            so.put(arm, j);
        }
        o.put(suite, so);
    }
    o
}

// ───────────────────────────── 3. extreme pairs ─────────────────────────────

fn extreme_pairs() -> J {
    const PAIRS: [(i8, i8); 8] =
        [(-128, -128), (-128, -127), (-128, 127), (-1, -128), (0, -128), (1, -128), (127, -128), (127, 127)];
    let mut rows = Vec::new();
    for k in [1usize, 64] {
        // k = 1 -> 1x1 (never AMX); k = 64 -> 16x16 so the AMX tier is eligible on this host.
        let (m, n) = if k == 1 { (1usize, 1usize) } else { (16, 16) };
        for (a, w) in PAIRS {
            let av = vec![a; m * k];
            let wr = vec![w; n * k];
            let wt = transpose(&wr, n, k);
            let d = DotInputs { a: &av, w_rows: &wr, wt: &wt, m, n, k };
            let orc = oracle(&d);
            let mut arms = J::obj();
            for arm in DOT_ARMS {
                let j = match run_dot_arm(arm, &d) {
                    ArmOut::NotRun(why) => not_run(&why),
                    ArmOut::Cells(v) => J::obj()
                        .set("cell0", J::i(v[0]))
                        .set("all_cells_equal_oracle", J::Bool(v == orc)),
                };
                arms.put(arm, j);
            }
            rows.push(
                J::obj()
                    .set("a", J::i(a))
                    .set("w", J::i(w))
                    .set("k", J::us(k))
                    .set("shape_mnk", J::Arr(vec![J::us(m), J::us(n), J::us(k)]))
                    .set("matmul_tier", J::s(matmul_tier(m, n, k)))
                    .set("oracle", J::i(orc[0]))
                    .set("arms", arms),
            );
        }
    }
    J::Arr(rows)
}

// ───────────────────────────── 4. FC four arms ─────────────────────────────

/// Per-case precomputation shared by the four arms and the negative controls.
struct Calc {
    mult: Vec<i32>,
    shift: Vec<i32>,
    q_source: &'static str,
    /// exact accumulator per cell (i64): bias + sum (a - in_zp) * w          [arm B/D stage]
    acc_b: Vec<i64>,
    /// matmul_i8_to_i32(a, W^T) per cell (None if the arm failed)            [arm C stage]
    mm: Option<Vec<i64>>,
    colsum: Vec<i64>,
    /// ndarray quantize_multiplier(real) vs TFLM FCQ: number of differing channels
    fcq_diff_channels: usize,
}

fn real_multiplier(c: &FcCase, ch: usize) -> f64 {
    // Same expression TFLM's GetQuantizedConvolutionMultipler uses: double(in)*double(w)/double(out).
    (f32::from_bits(c.in_scale_bits) as f64) * (f32::from_bits(c.ws[ch]) as f64) / (f32::from_bits(c.out_scale_bits) as f64)
}

fn compute_calc(c: &FcCase, tflm: Option<&TflmFc>) -> Calc {
    let n = c.out_dim;
    let nd: Vec<(i32, i32)> = (0..n).map(|ch| quantize_multiplier(real_multiplier(c, ch))).collect();
    let (mult, shift, q_source, diff) = match tflm {
        Some(t) if t.have_q && (t.mult.len() == n || t.mult.len() == 1) && t.shift.len() == t.mult.len() => {
            // A single (mult, shift) pair is broadcast to all channels (per-tensor).
            let bc = |v: &Vec<i32>| -> Vec<i32> { (0..n).map(|i| if v.len() == 1 { v[0] } else { v[i] }).collect() };
            let (m, sh) = (bc(&t.mult), bc(&t.shift));
            let diff = (0..n).filter(|&i| (m[i], sh[i]) != nd[i]).count();
            (m, sh, "tflm_FCQ", diff)
        }
        _ => (nd.iter().map(|x| x.0).collect(), nd.iter().map(|x| x.1).collect(), "ndarray_quantize_multiplier", 0),
    };
    let mut acc_b = vec![0i64; c.batches * n];
    let mut colsum = vec![0i64; n];
    for ch in 0..n {
        colsum[ch] = c.w[ch * c.in_dim..(ch + 1) * c.in_dim].iter().map(|&x| x as i64).sum();
    }
    for r in 0..c.batches {
        for ch in 0..n {
            let mut s = c.bias[ch] as i64;
            for p in 0..c.in_dim {
                s += (c.input[r * c.in_dim + p] as i64 - c.in_zp as i64) * c.w[ch * c.in_dim + p] as i64;
            }
            acc_b[r * n + ch] = s;
        }
    }
    let wt = transpose(&c.w, n, c.in_dim);
    let d = DotInputs { a: &c.input, w_rows: &c.w, wt: &wt, m: c.batches, n, k: c.in_dim };
    let mm = match run_dot_arm("matmul_i8_to_i32", &d) {
        ArmOut::Cells(v) => Some(v),
        ArmOut::NotRun(_) => None,
    };
    Calc { mult, shift, q_source, acc_b, mm, colsum, fcq_diff_channels: diff }
}

fn fits_i32(x: i64) -> bool {
    x >= i32::MIN as i64 && x <= i32::MAX as i64
}

/// ndarray's multiply_by_quantized_multiplier_i32 body (simd_fixedpoint.rs:95-101) with a pluggable rounding divide.
fn mulq_custom(x: i32, mult: i32, shift: i32, rdp: &dyn Fn(i32, i32) -> i32) -> i32 {
    let left = shift.max(0) as u32;
    let right = (-shift).max(0);
    rdp(srdhm_i32(x.wrapping_mul(1i32.wrapping_shl(left)), mult), right)
}

fn rdp_floor(x: i32, e: i32) -> i32 {
    x >> e
}

fn rdp_ties_even(x: i32, e: i32) -> i32 {
    if e == 0 {
        return x;
    }
    let q = x >> e;
    let mask = ((1i64 << e) - 1) as i32;
    let rem = x & mask;
    let half = 1i32 << (e - 1);
    if rem > half {
        q + 1
    } else if rem == half {
        q + (q & 1)
    } else {
        q
    }
}

#[derive(Clone, Copy)]
enum BVariant {
    Baseline,
    FloorShift,
    TiesEven,
    CastBeforeClamp,
}

/// Arm B for one cell. Returns (byte, scaled_before_offset, value_after_offset).
fn arm_b_cell(c: &FcCase, calc: &Calc, idx: usize, ch: usize, var: BVariant) -> Result<(i8, i32, i32), String> {
    let acc = calc.acc_b[idx];
    if !fits_i32(acc) {
        return Err(format!("acc {} does not fit i32", acc));
    }
    let (m, s) = (calc.mult[ch], calc.shift[ch]);
    // ndarray contract (simd_fixedpoint.rs:96-97): multiplier >= 0, shift in -31..=30.
    if m < 0 || !(-31..=30).contains(&s) {
        return Err(format!("multiplier/shift outside ndarray contract: {} {}", m, s));
    }
    let x = acc as i32;
    let scaled = match var {
        BVariant::Baseline | BVariant::CastBeforeClamp => multiply_by_quantized_multiplier_i32(x, m, s),
        BVariant::FloorShift => mulq_custom(x, m, s, &rdp_floor),
        BVariant::TiesEven => mulq_custom(x, m, s, &rdp_ties_even),
    };
    let v = scaled.wrapping_add(c.out_zp);
    let byte = match var {
        // F7: wrapping `as i8` BEFORE the clamp
        BVariant::CastBeforeClamp => ((v as i8) as i32).clamp(c.act_min, c.act_max) as i8,
        _ => v.clamp(c.act_min, c.act_max) as i8,
    };
    Ok((byte, scaled, v))
}

/// Arm C for the whole case: acc = bias + matmul - in_zp*colsum (i64, checked into i32), then
/// requantize_i32_to_i8 once per batch row (channels = out_dim). `fold=false` is negative control F8b.
/// Returns (bytes, acc per cell).
fn arm_c(c: &FcCase, calc: &Calc, fold: bool) -> Result<(Vec<i8>, Vec<i64>), String> {
    let mm = calc.mm.as_ref().ok_or_else(|| "matmul_i8_to_i32 not run".to_string())?;
    let n = c.out_dim;
    if calc.mult.iter().any(|&m| m < 0) || calc.shift.iter().any(|s| !(-31..=30).contains(s)) {
        return Err("multiplier/shift outside ndarray contract".to_string());
    }
    let mut accs = vec![0i64; c.batches * n];
    let mut bytes = vec![0i8; c.batches * n];
    for r in 0..c.batches {
        let mut row = vec![0i32; n];
        for ch in 0..n {
            let mut a = c.bias[ch] as i64 + mm[r * n + ch];
            if fold {
                a -= c.in_zp as i64 * calc.colsum[ch];
            }
            if !fits_i32(a) {
                return Err(format!("acc_c {} does not fit i32", a));
            }
            accs[r * n + ch] = a;
            row[ch] = a as i32;
        }
        let mut out_row = vec![0i8; n];
        requantize_i32_to_i8(&row, &calc.mult, &calc.shift, c.out_zp, c.act_min, c.act_max, &mut out_row);
        bytes[r * n..(r + 1) * n].copy_from_slice(&out_row);
    }
    Ok((bytes, accs))
}

/// Arm D for one cell: exact rational, rounded half away from zero, + out_zp, clamp.
fn arm_d_cell(c: &FcCase, calc: &Calc, idx: usize, ch: usize) -> Result<(i8, i64), String> {
    let q = exact_scaled_round_away(calc.acc_b[idx], c.in_scale_bits, c.ws[ch], c.out_scale_bits)?;
    let v = (q + c.out_zp as i64).clamp(c.act_min as i64, c.act_max as i64);
    Ok((v as i8, q))
}

#[derive(Default)]
struct PairStats {
    cells: u64,
    a_missing: u64,
    b_err: u64,
    c_err: u64,
    d_err: u64,
    ab: u64,
    bc: u64,
    ac: u64,
    ad: u64,
    bd: u64,
    acc_bc: u64,
    ex: BTreeMap<&'static str, Vec<J>>,
}

impl PairStats {
    fn to_j(&self) -> J {
        let mut ex = J::obj();
        for (k, v) in &self.ex {
            ex.put(k, J::Arr(v.iter().map(clone_j).collect()));
        }
        J::obj()
            .set("cells", J::u(self.cells))
            .set("A_missing_cells", J::u(self.a_missing))
            .set("B_error_cells", J::u(self.b_err))
            .set("C_error_cells", J::u(self.c_err))
            .set("D_error_cells", J::u(self.d_err))
            .set("A_ne_B", J::u(self.ab))
            .set("B_ne_C", J::u(self.bc))
            .set("A_ne_C", J::u(self.ac))
            .set("A_ne_D", J::u(self.ad))
            .set("B_ne_D", J::u(self.bd))
            .set("acc_B_ne_acc_C", J::u(self.acc_bc))
            .set("first_counterexamples", ex)
    }
}

fn clone_j(j: &J) -> J {
    match j {
        J::Null => J::Null,
        J::Bool(b) => J::Bool(*b),
        J::Int(i) => J::Int(*i),
        J::Str(s) => J::Str(s.clone()),
        J::Arr(v) => J::Arr(v.iter().map(clone_j).collect()),
        J::Obj(v) => J::Obj(v.iter().map(|(k, x)| (k.clone(), clone_j(x))).collect()),
    }
}

fn fc_arms(cases: &[FcCase], calcs: &[Calc], tflm: &Option<std::collections::HashMap<String, TflmFc>>) -> J {
    let mut buckets: BTreeMap<String, PairStats> = BTreeMap::new();
    let mut tflm_status = BTreeMap::<&'static str, u64>::new();
    let mut q_src = BTreeMap::<&'static str, u64>::new();
    let mut fcq_diff_total = 0u64;
    for (c, calc) in cases.iter().zip(calcs.iter()) {
        let t = tflm.as_ref().and_then(|m| m.get(&c.id));
        let a_ok = match t {
            Some(t) if t.have_out && t.prepare_ok && t.invoke_ok && t.out.len() == c.batches * c.out_dim => {
                *tflm_status.entry("ok").or_default() += 1;
                true
            }
            Some(_) => {
                *tflm_status.entry("prepare_or_invoke_failed_or_wrong_len").or_default() += 1;
                false
            }
            None => {
                *tflm_status.entry("no_output_for_case").or_default() += 1;
                false
            }
        };
        *q_src.entry(calc.q_source).or_default() += 1;
        fcq_diff_total += calc.fcq_diff_channels as u64;
        let cres = arm_c(c, calc, true);
        let suite = c.suite();
        let keys = [suite.to_string(), format!("{}/pc{}", suite, c.per_channel as i32)];
        for r in 0..c.batches {
            for ch in 0..c.out_dim {
                let idx = r * c.out_dim + ch;
                let a = if a_ok {
                    // Narrow-clamp cases: TFLM ran with ActNone (-128..127); clamp(clamp(x)) == clamp(x) so the
                    // post-clamp below is exactly what a narrow act range would have produced.
                    let t = t.unwrap();
                    Some((t.out[idx] as i32).clamp(c.act_min, c.act_max) as i8)
                } else {
                    None
                };
                let b = arm_b_cell(c, calc, idx, ch, BVariant::Baseline);
                let cc = match &cres {
                    Ok((bytes, accs)) => Some((bytes[idx], accs[idx])),
                    Err(_) => None,
                };
                let d = arm_d_cell(c, calc, idx, ch);
                let mut ex_to_add: Vec<(&'static str, J)> = Vec::new();
                let mk = |why: &str| -> J {
                    J::obj()
                        .set("pair", J::s(why))
                        .set("case", J::s(&c.id))
                        .set("batch", J::us(r))
                        .set("channel", J::us(ch))
                        .set("in_zp", J::i(c.in_zp))
                        .set("out_zp", J::i(c.out_zp))
                        .set("bias", J::i(c.bias[ch]))
                        .set("mult", J::i(calc.mult[ch]))
                        .set("shift", J::i(calc.shift[ch]))
                        .set("acc_B", J::i(calc.acc_b[idx]))
                        .set("acc_C", cc.map(|x| J::i(x.1)).unwrap_or(J::Null))
                        .set("scaled_B_before_offset", b.as_ref().map(|x| J::i(x.1)).unwrap_or(J::Null))
                        .set("exact_rounded_D_before_offset", d.as_ref().map(|x| J::i(x.1)).unwrap_or(J::Null))
                        .set("A_tflm", a.map(|x| J::i(x)).unwrap_or(J::Null))
                        .set("B_scalar", b.as_ref().map(|x| J::i(x.0)).unwrap_or(J::Null))
                        .set("C_simd", cc.map(|x| J::i(x.0)).unwrap_or(J::Null))
                        .set("D_exact", d.as_ref().map(|x| J::i(x.0)).unwrap_or(J::Null))
                };
                // (counter selector, name)
                let mut inc = [false; 5]; // ab, bc, ac, ad, bd
                let (mut a_miss, mut b_e, mut c_e, mut d_e, mut accd) = (false, false, false, false, false);
                if a.is_none() {
                    a_miss = true;
                }
                if b.is_err() {
                    b_e = true;
                }
                if cc.is_none() {
                    c_e = true;
                }
                if d.is_err() {
                    d_e = true;
                }
                let bb = b.as_ref().ok().map(|x| x.0);
                let dd = d.as_ref().ok().map(|x| x.0);
                if let (Some(a), Some(b)) = (a, bb) {
                    if a != b {
                        inc[0] = true;
                        ex_to_add.push(("A_ne_B", mk("A!=B")));
                    }
                }
                if let (Some(b), Some((c_, _))) = (bb, cc) {
                    if b != c_ {
                        inc[1] = true;
                        ex_to_add.push(("B_ne_C", mk("B!=C")));
                    }
                }
                if let (Some(a), Some((c_, _))) = (a, cc) {
                    if a != c_ {
                        inc[2] = true;
                        ex_to_add.push(("A_ne_C", mk("A!=C")));
                    }
                }
                if let (Some(a), Some(d_)) = (a, dd) {
                    if a != d_ {
                        inc[3] = true;
                        ex_to_add.push(("A_ne_D", mk("A!=D")));
                    }
                }
                if let (Some(b), Some(d_)) = (bb, dd) {
                    if b != d_ {
                        inc[4] = true;
                        ex_to_add.push(("B_ne_D", mk("B!=D")));
                    }
                }
                if let Some((_, acc_c)) = cc {
                    if acc_c != calc.acc_b[idx] {
                        accd = true;
                        ex_to_add.push(("acc_B_ne_acc_C", mk("acc_B!=acc_C")));
                    }
                }
                for key in &keys {
                    let ps = buckets.entry(key.clone()).or_default();
                    ps.cells += 1;
                    ps.a_missing += a_miss as u64;
                    ps.b_err += b_e as u64;
                    ps.c_err += c_e as u64;
                    ps.d_err += d_e as u64;
                    ps.ab += inc[0] as u64;
                    ps.bc += inc[1] as u64;
                    ps.ac += inc[2] as u64;
                    ps.ad += inc[3] as u64;
                    ps.bd += inc[4] as u64;
                    ps.acc_bc += accd as u64;
                    for (name, j) in &ex_to_add {
                        let v = ps.ex.entry(*name).or_default();
                        if v.len() < MAX_EXAMPLES {
                            v.push(clone_j(j));
                        }
                    }
                }
            }
        }
    }
    let mut o = J::obj();
    let mut ts = J::obj();
    for (k, v) in &tflm_status {
        ts.put(k, J::u(*v));
    }
    let mut qs = J::obj();
    for (k, v) in &q_src {
        qs.put(k, J::u(*v));
    }
    o.put("cases", J::us(cases.len()));
    o.put("tflm_case_status", ts);
    o.put("multiplier_source_case_count", qs);
    o.put("fcq_vs_ndarray_quantize_multiplier_differing_channels", J::u(fcq_diff_total));
    o.put(
        "note_narrow_clamp",
        J::s("TFLM runs with activation None (-128..127); for act range -100/90 cases the TFLM bytes are post-clamped (equivalent)."),
    );
    o.put(
        "note_arms",
        J::s("A=TFLM kernel bytes; B=scalar i64 acc -> ndarray multiply_by_quantized_multiplier_i32; C=matmul_i8_to_i32 - in_zp*colsum -> requantize_i32_to_i8; D=exact rational, ties away"),
    );
    let mut bj = J::obj();
    for (k, v) in &buckets {
        bj.put(k, v.to_j());
    }
    o.put("buckets", bj);
    o
}

// ───────────────────────────── 5. codec / index checks ─────────────────────────────

/// value = (code - 0) * 1.0
struct UnitCal;
impl Calibration for UnitCal {
    const SCALE: f32 = 1.0;
    const ZERO_POINT: i32 = 0;
}
struct CalM3;
impl Calibration for CalM3 {
    const SCALE: f32 = 1.0;
    const ZERO_POINT: i32 = -3;
}
struct CalP3;
impl Calibration for CalP3 {
    const SCALE: f32 = 1.0;
    const ZERO_POINT: i32 = 3;
}

fn codec_index(artifact: &str) -> J {
    let mut raw_seen = [false; 256];
    let mut bia_seen = [false; 256];
    let (mut raw_inv_bad, mut bia_inv_bad) = (0u32, 0u32);
    let (mut asym_store_bad, mut sym_store_bad, mut asym_load_bad) = (0u32, 0u32, 0u32);
    for q in -128i32..=127 {
        let r = (q as i8) as u8; // raw mapping: two's-complement byte
        raw_seen[r as usize] = true;
        if (r as i8) as i32 != q {
            raw_inv_bad += 1;
        }
        let b = (q + 128) as u8; // biased mapping
        bia_seen[b as usize] = true;
        if b as i32 - 128 != q {
            bia_inv_bad += 1;
        }
        // ndarray/src/hpc/codec8.rs:169-187: Stores<Asym|Sym> for U8: store = (code + 128) as u8
        let sa: u8 = <U8 as Stores<Asym>>::store(q);
        let ss: u8 = <U8 as Stores<Sym>>::store(q);
        if sa != b {
            asym_store_bad += 1;
        }
        if ss != b {
            sym_store_bad += 1;
        }
        let la: i32 = <U8 as Stores<Asym>>::load(b);
        if la != q {
            asym_load_bad += 1;
        }
    }
    let named: Vec<J> = [-128i32, -1, 0, 127]
        .iter()
        .map(|&q| {
            J::obj()
                .set("code", J::i(q))
                .set("raw_u8", J::i((q as i8) as u8))
                .set("biased_u8", J::i((q + 128) as u8))
        })
        .collect();
    let mut o = J::obj()
        .set("raw_bijective_256", J::Bool(raw_seen.iter().all(|&x| x)))
        .set("biased_bijective_256", J::Bool(bia_seen.iter().all(|&x| x)))
        .set("raw_inverse_failures", J::i(raw_inv_bad))
        .set("biased_inverse_failures", J::i(bia_inv_bad))
        .set("biased_vs_Codec8_U8_Asym_store_mismatches", J::i(asym_store_bad))
        .set("biased_vs_Codec8_U8_Sym_store_mismatches", J::i(sym_store_bad))
        .set("Codec8_U8_Asym_load_inverse_failures", J::i(asym_load_bad))
        .set("named_codes", J::Arr(named));

    // F2: the 6.2 TANH SourceParity table indexed raw instead of biased.
    let f2 = match fs::read(artifact) {
        Err(e) => not_run(&format!("artifact {} unreadable: {}", artifact, e)),
        Ok(bytes) if bytes.len() < F2_TABLE_OFFSET + 256 => not_run("artifact shorter than offset 78 + 256"),
        Ok(bytes) => {
            let t = &bytes[F2_TABLE_OFFSET..F2_TABLE_OFFSET + 256];
            let mut changed = 0u32;
            for q in -128i32..=127 {
                let bi = (q + 128) as usize;
                let ri = ((q as i8) as u8) as usize;
                if t[bi] != t[ri] {
                    changed += 1;
                }
            }
            let listed: Vec<J> = [-128i32, -1, 0, 127]
                .iter()
                .map(|&q| {
                    let bi = (q + 128) as usize;
                    let ri = ((q as i8) as u8) as usize;
                    J::obj()
                        .set("code", J::i(q))
                        .set("biased_index", J::us(bi))
                        .set("raw_index", J::us(ri))
                        .set("value_via_biased", J::i(t[bi]))
                        .set("value_via_raw", J::i(t[ri]))
                        .set("changes", J::Bool(t[bi] != t[ri]))
                })
                .collect();
            J::obj()
                .set("table_offset", J::us(F2_TABLE_OFFSET))
                .set("codes_whose_byte_changes_when_indexed_raw", J::i(changed))
                .set("of", J::i(256))
                .set("named", J::Arr(listed))
        }
    };
    o.put("F2_tanh_table_raw_vs_biased", f2);
    o
}

// ───────────────────────────── 6. negative controls ─────────────────────────────

fn maddubs_dot(a_u8: &[u8], w: &[i8]) -> i64 {
    // VPMADDUBSW: adjacent u8*i8 products summed pairwise with SIGNED 16-bit SATURATION; the following
    // VPMADDWD(.., 1) adds the i16 pairs into i32 lanes, i.e. a plain sum of the saturated pair values.
    let mut tot = 0i64;
    let mut i = 0;
    while i < a_u8.len() {
        let p0 = a_u8[i] as i32 * w[i] as i32;
        let p1 = if i + 1 < a_u8.len() { a_u8[i + 1] as i32 * w[i + 1] as i32 } else { 0 };
        tot += (p0 + p1).clamp(-32768, 32767) as i64;
        i += 2;
    }
    tot
}

fn f5() -> J {
    let fixtures: [(i8, i8, usize); 8] =
        [(127, 127, 2), (127, 127, 64), (127, 127, 128), (-128, 127, 64), (0, 127, 64), (1, 127, 64), (2, 127, 64), (64, 127, 64)];
    let mut rows = Vec::new();
    for (a, w, k) in fixtures {
        let av = vec![a; k];
        let wv = vec![w; k];
        let a_u8 = a_to_u8(&av);
        let colsum: i64 = wv.iter().map(|&x| x as i64).sum();
        let emu = maddubs_dot(&a_u8, &wv) - 128 * colsum;
        let orc = a as i64 * w as i64 * k as i64;
        rows.push(
            J::obj()
                .set("a", J::i(a))
                .set("a_u8", J::i(a_u8[0]))
                .set("w", J::i(w))
                .set("k", J::us(k))
                .set("oracle", J::i(orc))
                .set("vpmaddubsw_corrected", J::i(emu))
                .set("differs", J::Bool(emu != orc)),
        );
    }
    J::obj().set("fixtures", J::Arr(rows))
}

fn f9_case(k: usize, a_val: i8, w_val: i8) -> J {
    let (m, n) = (1usize, 16usize);
    let a = vec![a_val; k];
    let a_u8 = a_to_u8(&a);
    let wt = vec![w_val; k * n];
    let orc = a_val as i64 * w_val as i64 * k as i64;
    let true_unsigned = (a_val as i64 + 128) * w_val as i64 * k as i64;
    let colsum_true = w_val as i64 * k as i64;
    let colsum128 = 128 * colsum_true;
    let mut o = J::obj()
        .set("k", J::us(k))
        .set("a", J::i(a_val))
        .set("a_u8", J::i(a_u8[0]))
        .set("w", J::i(w_val))
        .set("oracle_signed_result_i64", J::i(orc))
        .set("oracle_fits_i32", J::Bool(fits_i32(orc)))
        .set("true_unsigned_accumulator_i64", J::i(true_unsigned))
        .set("unsigned_accumulator_fits_i32", J::Bool(fits_i32(true_unsigned)))
        .set("colsum_i64", J::i(colsum_true))
        .set("colsum_fits_i32", J::Bool(fits_i32(colsum_true)))
        .set("128_x_colsum_i64", J::i(colsum128))
        .set("128_x_colsum_fits_i32", J::Bool(fits_i32(colsum128)));
    if !(std::is_x86_feature_detected!("avx512vnni") && std::is_x86_feature_detected!("avx512f")) {
        o.put("kernel", not_run("avx512vnni absent"));
        return o;
    }
    let mut raw = vec![0i32; m * n];
    // SAFETY: avx512vnni + avx512f verified just above.
    unsafe { ndarray::hpc::int8_tile_gemm::int8_gemm_vpdpbusd_zmm(&a_u8, &wt, &mut raw, m, n, k) };
    let raw0 = raw[0];
    o.put("kernel_raw_lane0_i32", J::i(raw0));
    o.put("kernel_raw_equals_true_unsigned_mod_2_32", J::Bool(raw0 == true_unsigned as i32));
    // Shadow of the lab bias step with checked_* (which step overflows):
    let cs_i32 = colsum_true as i32; // truncating; only meaningful when colsum_fits_i32
    let step_mul = cs_i32.checked_mul(128);
    o.put("shadow_checked_128_mul_colsum", step_mul.map(|x| J::i(x)).unwrap_or_else(|| J::s("OVERFLOW")));
    let step_sub = step_mul.and_then(|bias| raw0.checked_sub(bias));
    o.put("shadow_checked_raw_minus_bias", step_sub.map(|x| J::i(x)).unwrap_or_else(|| J::s("OVERFLOW (or earlier step overflowed)")));
    // Verbatim lab copy of the production bias step (panics on overflow iff debug_assertions/overflow-checks).
    let mut c_v = raw.clone();
    let verb = catch_unwind(AssertUnwindSafe(|| {
        subtract_i8_to_u8_bias(&mut c_v, &wt, m, n, k);
        c_v[0]
    }));
    o.put(
        "verbatim_subtract_i8_to_u8_bias",
        match verb {
            Ok(v) => J::obj().set("panicked", J::Bool(false)).set("lane0", J::i(v)).set("equals_oracle", J::Bool(v as i64 == orc)),
            Err(_) => J::obj().set("panicked", J::Bool(true)),
        },
    );
    let mut c_w = raw.clone();
    subtract_i8_to_u8_bias_wrapping(&mut c_w, &wt, m, n, k);
    o.put(
        "wrapping_variant",
        J::obj().set("lane0", J::i(c_w[0])).set("equals_oracle_when_oracle_fits", J::Bool(fits_i32(orc) && c_w[0] as i64 == orc)),
    );
    o
}

fn enc_act(zp: i32, x: f32, even: bool) -> i8 {
    let mut o = [0i8; 1];
    let r = if even {
        match zp {
            -3 => OnnxQ::encode_checked::<CalM3>(&[x], &mut o),
            3 => OnnxQ::encode_checked::<CalP3>(&[x], &mut o),
            _ => OnnxQ::encode_checked::<UnitCal>(&[x], &mut o),
        }
    } else {
        match zp {
            -3 => TfliteAct::encode_checked::<CalM3>(&[x], &mut o),
            3 => TfliteAct::encode_checked::<CalP3>(&[x], &mut o),
            _ => TfliteAct::encode_checked::<UnitCal>(&[x], &mut o),
        }
    };
    r.expect("encode of a non-NaN value");
    o[0]
}

/// Source order (tflite-micro micro_utils.h:40: round(value / scale) + zero_point), then clamp.
fn order1(x: f32, zp: i32, even: bool) -> i8 {
    let r = if even { x.round_ties_even() } else { x.round() };
    (r as i32 + zp).clamp(-128, 127) as i8
}
/// Zero point moved across the rounding: round(x + zp), then clamp.
fn order2(x: f32, zp: i32, even: bool) -> i8 {
    let y = x + zp as f32;
    let r = if even { y.round_ties_even() } else { y.round() };
    (r as i32).clamp(-128, 127) as i8
}

fn f8a() -> J {
    let zps = [-3i32, 0, 3];
    let xs: Vec<f32> = (-131i32..=130).map(|k| k as f32 + 0.5).collect(); // every half integer in [-130.5, 130.5]
    let table_x = [-2.5f32, -1.5, -0.5, 0.5, 1.5, 2.5];
    let mut o = J::obj().set("half_integers_tested", J::us(xs.len())).set("zero_points", J::Arr(zps.iter().map(|&z| J::i(z)).collect()));
    for (mode, even) in [("TiesAway(TfliteAct)", false), ("TiesEven(OnnxQ)", true)] {
        let (mut diff, mut total, mut codec_vs_o1) = (0u32, 0u32, 0u32);
        let mut by_zp: BTreeMap<i32, u32> = BTreeMap::new();
        for &zp in &zps {
            for &x in &xs {
                total += 1;
                if order1(x, zp, even) != order2(x, zp, even) {
                    diff += 1;
                    *by_zp.entry(zp).or_default() += 1;
                }
                if enc_act(zp, x, even) != order1(x, zp, even) {
                    codec_vs_o1 += 1;
                }
            }
        }
        let mut tab = Vec::new();
        for &zp in &zps {
            for &x in &table_x {
                tab.push(
                    J::obj()
                        .set("x", J::s(&format!("{:+.1}", x)))
                        .set("zp", J::i(zp))
                        .set("round_then_add_order1", J::i(order1(x, zp, even)))
                        .set("add_then_round_order2", J::i(order2(x, zp, even)))
                        .set("codec8_encoder", J::i(enc_act(zp, x, even))),
                );
            }
        }
        let mut bz = J::obj();
        for (z, c) in &by_zp {
            bz.put(&z.to_string(), J::i(*c));
        }
        o.put(
            mode,
            J::obj()
                .set("pairs_tested", J::i(total))
                .set("order1_vs_order2_differing", J::i(diff))
                .set("differing_by_zp", bz)
                .set("codec8_encoder_vs_order1_mismatches", J::i(codec_vs_o1))
                .set("table", J::Arr(tab)),
        );
    }
    o
}

fn negative_controls(cases: &[FcCase], calcs: &[Calc]) -> J {
    let mut o = J::obj();

    // F1: drop the 128*colsum correction from the zmm arm.
    let f1 = if !(std::is_x86_feature_detected!("avx512vnni") && std::is_x86_feature_detected!("avx512f")) {
        not_run("avx512vnni absent")
    } else {
        let mut per: BTreeMap<&'static str, (u64, u64)> = BTreeMap::new();
        for c in cases.iter().filter(|c| c.suite() != "ties") {
            let wt = transpose(&c.w, c.out_dim, c.in_dim);
            let a_u8 = a_to_u8(&c.input);
            let mut raw = vec![0i32; c.batches * c.out_dim];
            // SAFETY: avx512vnni + avx512f verified above.
            unsafe {
                ndarray::hpc::int8_tile_gemm::int8_gemm_vpdpbusd_zmm(&a_u8, &wt, &mut raw, c.batches, c.out_dim, c.in_dim)
            };
            let mut fixed = raw.clone();
            subtract_i8_to_u8_bias(&mut fixed, &wt, c.batches, c.out_dim, c.in_dim);
            let e = per.entry(c.suite()).or_default();
            e.0 += raw.len() as u64;
            e.1 += raw.iter().zip(fixed.iter()).filter(|(a, b)| a != b).count() as u64;
        }
        let mut j = J::obj();
        for (k, (t, ch)) in per {
            j.put(k, J::obj().set("cells", J::u(t)).set("cells_changed", J::u(ch)).set("inert", J::Bool(ch == 0)));
        }
        j
    };
    o.put("F1_drop_128_colsum_correction_zmm_arm", f1);

    // F3 / F7: arm-B mutations, counted against the baseline arm B.
    let mut f3: BTreeMap<String, [u64; 5]> = BTreeMap::new(); // cells, floor, ties_even, identity_bad, cast_before_clamp
    for (c, calc) in cases.iter().zip(calcs.iter()) {
        let e = f3.entry(c.suite().to_string()).or_default();
        for r in 0..c.batches {
            for ch in 0..c.out_dim {
                let idx = r * c.out_dim + ch;
                let Ok((base, scaled, _)) = arm_b_cell(c, calc, idx, ch, BVariant::Baseline) else { continue };
                e[0] += 1;
                if let Ok((b, _, _)) = arm_b_cell(c, calc, idx, ch, BVariant::FloorShift) {
                    e[1] += (b != base) as u64;
                }
                if let Ok((b, _, _)) = arm_b_cell(c, calc, idx, ch, BVariant::TiesEven) {
                    e[2] += (b != base) as u64;
                }
                // baseline identity: mulq_custom with ndarray's own rounding_divide_by_pot == the library fn
                let x = calc.acc_b[idx] as i32;
                let id = mulq_custom(x, calc.mult[ch], calc.shift[ch], &rounding_divide_by_pot_i32);
                e[3] += (id != scaled) as u64;
                if let Ok((b, _, _)) = arm_b_cell(c, calc, idx, ch, BVariant::CastBeforeClamp) {
                    e[4] += (b != base) as u64;
                }
            }
        }
    }
    let mut f3j = J::obj();
    let mut f7j = J::obj();
    let mut idj = J::obj();
    for (suite, v) in &f3 {
        f3j.put(
            suite,
            J::obj()
                .set("cells", J::u(v[0]))
                .set("floor_shift_x_gt_gt_e_cells_changed", J::u(v[1]))
                .set("ties_even_cells_changed", J::u(v[2])),
        );
        f7j.put(suite, J::obj().set("cells", J::u(v[0])).set("cells_changed", J::u(v[4])).set("inert", J::Bool(v[4] == 0)));
        idj.put(suite, J::u(v[3]));
    }
    o.put("F3_rounding_divide_by_pot_replaced", f3j);
    o.put("F3_baseline_identity_mismatches_mulq_custom_vs_library", idj);
    o.put("F7_cast_i8_before_clamp_arm_B", f7j);

    // F4: Asym code -128 fed to Codec8<I8, Sym, TiesAway, NaNAtMin>::decode_checked.
    {
        type SymNaN = Codec8<I8, Sym, TiesAway, NaNAtMin>;
        let src = [-128i8, -127, 0, 127];
        let mut dst = [0.0f32; 4];
        let r: Result<(), Codec8Error> = SymNaN::decode_checked::<UnitCal>(&src, &mut dst);
        o.put(
            "F4_asym_minus128_into_Sym_NaNAtMin_decode",
            J::obj()
                .set("input_codes", J::s("[-128,-127,0,127]"))
                .set("result", J::s(&format!("{:?}", r)))
                .set("decoded", J::s(&format!("{:?}", dst)))
                .set("minus128_decoded_as_NaN", J::Bool(r.is_ok() && dst[0].is_nan())),
        );
    }

    // F6: TfliteWeight::decode_checked on weights containing -128.
    {
        let (mut ok_c, mut err_c, mut ok_x, mut err_x) = (0u32, 0u32, 0u32, 0u32);
        let mut first: Option<J> = None;
        for c in cases {
            let mut dst = vec![0.0f32; c.w.len()];
            let r = TfliteWeight::decode_checked::<UnitCal>(&c.w, &mut dst);
            let has_m128 = c.w.contains(&-128);
            match (c.suite(), r.is_ok()) {
                ("extended", true) => ok_x += 1,
                ("extended", false) => err_x += 1,
                (_, true) => ok_c += 1,
                (_, false) => err_c += 1,
            }
            if has_m128 && first.is_none() {
                first = Some(
                    J::obj()
                        .set("case", J::s(&c.id))
                        .set("result", J::s(&format!("{:?}", r)))
                        .set("first_minus128_index", J::us(c.w.iter().position(|&x| x == -128).unwrap_or(0))),
                );
            }
        }
        o.put(
            "F6_TfliteWeight_decode_checked_with_minus128",
            J::obj()
                .set("conforming_and_ties_cases_Ok", J::i(ok_c))
                .set("conforming_and_ties_cases_Err", J::i(err_c))
                .set("extended_cases_Ok", J::i(ok_x))
                .set("extended_cases_Err", J::i(err_x))
                .set("first_case_with_minus128", first.unwrap_or(J::Null)),
        );
    }

    o.put("F8a_zero_point_across_rounding_float_encode", f8a());

    // F8b: omit the in_zp fold (-in_zp*colsum) in arm C.
    {
        let mut per_zp: BTreeMap<i32, (u64, u64)> = BTreeMap::new();
        for (c, calc) in cases.iter().zip(calcs.iter()) {
            if let (Ok((good, _)), Ok((bad, _))) = (arm_c(c, calc, true), arm_c(c, calc, false)) {
                let e = per_zp.entry(c.in_zp).or_default();
                e.0 += good.len() as u64;
                e.1 += good.iter().zip(bad.iter()).filter(|(a, b)| a != b).count() as u64;
            }
        }
        let mut j = J::obj();
        for (z, (t, ch)) in per_zp {
            j.put(&format!("in_zp={}", z), J::obj().set("cells", J::u(t)).set("cells_changed", J::u(ch)).set("inert", J::Bool(ch == 0)));
        }
        o.put("F8b_omit_in_zp_fold_arm_C", j);
    }

    // F5: VPMADDUBSW emulation (fixtures + across all conforming/extended cells).
    {
        let mut j = f5();
        let (mut cells, mut bad) = (0u64, 0u64);
        for c in cases.iter().filter(|c| c.suite() != "ties") {
            for r in 0..c.batches {
                let a_u8 = a_to_u8(&c.input[r * c.in_dim..(r + 1) * c.in_dim]);
                for ch in 0..c.out_dim {
                    let w = &c.w[ch * c.in_dim..(ch + 1) * c.in_dim];
                    let colsum: i64 = w.iter().map(|&x| x as i64).sum();
                    let emu = maddubs_dot(&a_u8, w) - 128 * colsum;
                    let orc: i64 = c.input[r * c.in_dim..(r + 1) * c.in_dim].iter().zip(w).map(|(&a, &b)| a as i64 * b as i64).sum();
                    cells += 1;
                    bad += (emu != orc) as u64;
                }
            }
        }
        j.put("all_cells_conforming_and_extended", J::obj().set("cells", J::u(cells)).set("mismatches", J::u(bad)));
        o.put("F5_vpmaddubsw_i16_saturation_emulation", j);
    }

    // F9: overflow stress through the real zmm kernel + lab bias.
    o.put(
        "F9_overflow_stress",
        J::obj()
            .set("i_m1_n16_k70000_a127_w127", f9_case(70_000, 127, 127))
            .set("ii_m1_n16_k140000_a0_w127", f9_case(140_000, 0, 127)),
    );
    o
}

// ───────────────────────────── main ─────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: arms <dir> [artifacts.bin]");
        std::process::exit(2);
    }
    let dir = std::path::PathBuf::from(&args[1]);
    let artifact = args.get(2).map(|s| s.as_str()).unwrap_or(DEFAULT_ARTIFACT).to_string();

    let fx_c = read_opt(&dir.join("fx_cases.txt"));
    let fx_t = read_opt(&dir.join("tflm_fx.txt"));
    let fc_txt = read_opt(&dir.join("fc_cases.txt"));
    let fc_t = read_opt(&dir.join("tflm_fc.txt"));

    let mut top = J::obj();
    top.put("meta", meta());
    top.put("fx_parity_ndarray_vs_tflm", fx_parity(&fx_c, &fx_t));

    match fc_txt {
        None => {
            for k in ["dot_arms", "fc_four_arms", "negative_controls"] {
                top.put(k, not_run("fc_cases.txt absent"));
            }
            top.put("extreme_pairs", extreme_pairs());
        }
        Some(txt) => match parse_fc_cases(&txt) {
            Err(e) => {
                for k in ["dot_arms", "fc_four_arms", "negative_controls"] {
                    top.put(k, not_run(&format!("fc_cases parse: {}", e)));
                }
                top.put("extreme_pairs", extreme_pairs());
            }
            Ok(cases) => {
                let tflm = match &fc_t {
                    Some(t) => match parse_tflm_fc(t) {
                        Ok(m) => Some(m),
                        Err(e) => {
                            eprintln!("tflm_fc.txt parse error: {}", e);
                            None
                        }
                    },
                    None => None,
                };
                let calcs: Vec<Calc> = cases.iter().map(|c| compute_calc(c, tflm.as_ref().and_then(|m| m.get(&c.id)))).collect();
                top.put("dot_arms", dot_arms(&cases));
                top.put("extreme_pairs", extreme_pairs());
                top.put("fc_four_arms", fc_arms(&cases, &calcs, &tflm));
                top.put("negative_controls", negative_controls(&cases, &calcs));
            }
        },
    }
    top.put("codec_index_checks", codec_index(&artifact));

    let json = top.render();
    println!("{}", json);
    if let Err(e) = fs::write(dir.join("arms64.json"), &json) {
        eprintln!("could not write arms64.json: {}", e);
    }
}
