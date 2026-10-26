//! int8sem: shared helpers for the Amendment 6.4 lab probe (numerical meaning of 8-bit codes).
//!
//! Lab only. Part 1: PRNG + tiny JSON writer. Part 2: exchange-format types and parsers.
//! Part 3: exact rational requantisation + lab copies. Part 4: the dot-product arms.

#[cfg(not(target_arch = "x86_64"))]
compile_error!("int8sem is an x86_64-only lab probe (AMX / AVX-512 VNNI / AVX-VNNI arms)");

use std::collections::HashMap;
use std::fmt::Write as _;
use std::panic::{catch_unwind, AssertUnwindSafe};

// ───────────────────────────── Part 1: PRNG + JSON ─────────────────────────────

/// Fixed seed so `gen` is deterministic.
pub const SEED: u64 = 0x6A09_E667_F3BC_C908;

/// xorshift64* PRNG (no external deps).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
    pub fn next_i32(&mut self) -> i32 {
        self.next_u32() as i32
    }
    /// Uniform-ish in 0..n (modulo bias is irrelevant for a lab generator).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
    /// Inclusive range.
    pub fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        let span = (hi as i64 - lo as i64 + 1) as u64;
        (lo as i64 + self.below(span) as i64) as i32
    }
}

/// Minimal JSON value (hand-written writer, no serde).
pub enum J {
    Null,
    Bool(bool),
    Int(i64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    pub fn s(x: &str) -> J {
        J::Str(x.to_string())
    }
    pub fn i<T: Into<i64>>(x: T) -> J {
        J::Int(x.into())
    }
    pub fn u(x: u64) -> J {
        J::Int(x as i64)
    }
    pub fn us(x: usize) -> J {
        J::Int(x as i64)
    }
    pub fn obj() -> J {
        J::Obj(Vec::new())
    }
    /// Builder: push a key on an object (no-op on non-objects).
    pub fn set(mut self, k: &str, v: J) -> J {
        if let J::Obj(ref mut items) = self {
            items.push((k.to_string(), v));
        }
        self
    }
    /// In-place push on an object.
    pub fn put(&mut self, k: &str, v: J) {
        if let J::Obj(items) = self {
            items.push((k.to_string(), v));
        }
    }
    pub fn render(&self) -> String {
        let mut s = String::new();
        self.write(&mut s);
        s
    }
    fn write(&self, out: &mut String) {
        match self {
            J::Null => out.push_str("null"),
            J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            J::Int(i) => {
                let _ = write!(out, "{}", i);
            }
            J::Str(s) => {
                out.push('"');
                for ch in s.chars() {
                    match ch {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        c if (c as u32) < 0x20 => {
                            let _ = write!(out, "\\u{:04x}", c as u32);
                        }
                        c => out.push(c),
                    }
                }
                out.push('"');
            }
            J::Arr(v) => {
                out.push('[');
                for (n, x) in v.iter().enumerate() {
                    if n > 0 {
                        out.push(',');
                    }
                    x.write(out);
                }
                out.push(']');
            }
            J::Obj(v) => {
                out.push('{');
                for (n, (k, x)) in v.iter().enumerate() {
                    if n > 0 {
                        out.push(',');
                    }
                    J::Str(k.clone()).write(out);
                    out.push(':');
                    x.write(out);
                }
                out.push('}');
            }
        }
    }
}

// ───────────────────────────── Part 2: exchange formats ─────────────────────────────

/// Parse `0x1f2e3d4c` or `1f2e3d4c` (hex). Bit patterns in the exchange files are written `0x%08x`.
pub fn parse_hex_u32(s: &str) -> Option<u32> {
    let t = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    u32::from_str_radix(t, 16).ok()
}

/// One fully-connected test block (fc_cases.txt).
#[derive(Clone)]
pub struct FcCase {
    pub id: String,
    pub per_channel: bool,
    pub batches: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub in_scale_bits: u32,
    pub in_zp: i32,
    pub out_scale_bits: u32,
    pub out_zp: i32,
    pub act_min: i32,
    pub act_max: i32,
    /// batches*in_dim, row-major [batch][in]
    pub input: Vec<i8>,
    /// out_dim f32 bit patterns (per-tensor: all equal)
    pub ws: Vec<u32>,
    /// out_dim*in_dim, row-major [out][in]
    pub w: Vec<i8>,
    /// out_dim int32 bias
    pub bias: Vec<i32>,
}

impl FcCase {
    /// Suite by id prefix: `X...` extended (weights include -128), `T...` half-tie fixtures,
    /// everything else (`C...`, `E...`) conforming.
    pub fn suite(&self) -> &'static str {
        match self.id.as_bytes().first() {
            Some(b'X') => "extended",
            Some(b'T') => "ties",
            _ => "conforming",
        }
    }
    /// True when act_min/act_max is narrower than the int8 range (TFLM with ActNone cannot clamp this;
    /// the arms post-clamp the TFLM bytes, which is equivalent: clamp(clamp(x,-128,127),lo,hi)==clamp(x,lo,hi)).
    pub fn narrow(&self) -> bool {
        self.act_min != -128 || self.act_max != 127
    }

    pub fn write_to(&self, s: &mut String) {
        let _ = writeln!(
            s,
            "FC {} {} {} {} {} 0x{:08x} {} 0x{:08x} {} {} {}",
            self.id,
            self.per_channel as i32,
            self.batches,
            self.in_dim,
            self.out_dim,
            self.in_scale_bits,
            self.in_zp,
            self.out_scale_bits,
            self.out_zp,
            self.act_min,
            self.act_max
        );
        s.push_str("IN");
        for v in &self.input {
            let _ = write!(s, " {}", v);
        }
        s.push('\n');
        s.push_str("WS");
        for v in &self.ws {
            let _ = write!(s, " 0x{:08x}", v);
        }
        s.push('\n');
        s.push_str("W");
        for v in &self.w {
            let _ = write!(s, " {}", v);
        }
        s.push('\n');
        s.push_str("B");
        for v in &self.bias {
            let _ = write!(s, " {}", v);
        }
        s.push('\n');
    }
}

fn parse_nums<T: std::str::FromStr>(toks: &[&str], what: &str) -> Result<Vec<T>, String> {
    let mut v = Vec::with_capacity(toks.len());
    for t in toks {
        v.push(t.parse::<T>().map_err(|_| format!("bad {} token '{}'", what, t))?);
    }
    Ok(v)
}

/// Parse fc_cases.txt (blocks FC / IN / WS / W / B).
pub fn parse_fc_cases(text: &str) -> Result<Vec<FcCase>, String> {
    let mut out: Vec<FcCase> = Vec::new();
    let mut cur: Option<FcCase> = None;
    for (ln, line) in text.lines().enumerate() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }
        let err = |m: &str| format!("fc_cases line {}: {}", ln + 1, m);
        match toks[0] {
            "FC" => {
                if let Some(c) = cur.take() {
                    out.push(c);
                }
                if toks.len() != 12 {
                    return Err(err("FC header needs 12 tokens"));
                }
                let p = |i: usize| -> Result<i64, String> {
                    toks[i].parse::<i64>().map_err(|_| err("bad int in header"))
                };
                cur = Some(FcCase {
                    id: toks[1].to_string(),
                    per_channel: p(2)? != 0,
                    batches: p(3)? as usize,
                    in_dim: p(4)? as usize,
                    out_dim: p(5)? as usize,
                    in_scale_bits: parse_hex_u32(toks[6]).ok_or_else(|| err("bad in_scale_bits"))?,
                    in_zp: p(7)? as i32,
                    out_scale_bits: parse_hex_u32(toks[8]).ok_or_else(|| err("bad out_scale_bits"))?,
                    out_zp: p(9)? as i32,
                    act_min: p(10)? as i32,
                    act_max: p(11)? as i32,
                    input: Vec::new(),
                    ws: Vec::new(),
                    w: Vec::new(),
                    bias: Vec::new(),
                });
            }
            "IN" | "WS" | "W" | "B" => {
                let c = cur.as_mut().ok_or_else(|| err("data line before FC header"))?;
                match toks[0] {
                    "IN" => c.input = parse_nums::<i8>(&toks[1..], "IN")?,
                    "WS" => {
                        c.ws = toks[1..]
                            .iter()
                            .map(|t| parse_hex_u32(t).ok_or_else(|| err("bad WS hex")))
                            .collect::<Result<Vec<u32>, String>>()?
                    }
                    "W" => c.w = parse_nums::<i8>(&toks[1..], "W")?,
                    _ => c.bias = parse_nums::<i32>(&toks[1..], "B")?,
                }
            }
            other => return Err(err(&format!("unknown record '{}'", other))),
        }
    }
    if let Some(c) = cur.take() {
        out.push(c);
    }
    for c in &out {
        if c.input.len() != c.batches * c.in_dim
            || c.w.len() != c.out_dim * c.in_dim
            || c.ws.len() != c.out_dim
            || c.bias.len() != c.out_dim
        {
            return Err(format!("case {}: inconsistent lengths", c.id));
        }
    }
    Ok(out)
}

/// TFLM output for one case (tflm_fc.txt): `FCOUT id prepare_ok invoke_ok <bytes>` and
/// `FCQ id <mults> | <shifts>`.
#[derive(Clone, Default)]
pub struct TflmFc {
    pub prepare_ok: bool,
    pub invoke_ok: bool,
    pub out: Vec<i8>,
    pub mult: Vec<i32>,
    pub shift: Vec<i32>,
    pub have_out: bool,
    pub have_q: bool,
}

fn tok_bool(t: &str) -> bool {
    matches!(t, "1" | "true" | "ok" | "OK" | "True")
}

pub fn parse_tflm_fc(text: &str) -> Result<HashMap<String, TflmFc>, String> {
    let mut m: HashMap<String, TflmFc> = HashMap::new();
    for (ln, line) in text.lines().enumerate() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }
        match toks[0] {
            "FCOUT" => {
                if toks.len() < 4 {
                    return Err(format!("tflm_fc line {}: short FCOUT", ln + 1));
                }
                let e = m.entry(toks[1].to_string()).or_default();
                e.prepare_ok = tok_bool(toks[2]);
                e.invoke_ok = tok_bool(toks[3]);
                e.out = parse_nums::<i8>(&toks[4..], "FCOUT")?;
                e.have_out = true;
            }
            "FCQ" => {
                if toks.len() < 3 {
                    return Err(format!("tflm_fc line {}: short FCQ", ln + 1));
                }
                let bar = toks
                    .iter()
                    .position(|t| *t == "|")
                    .ok_or_else(|| format!("tflm_fc line {}: FCQ without '|'", ln + 1))?;
                let e = m.entry(toks[1].to_string()).or_default();
                e.mult = parse_nums::<i32>(&toks[2..bar], "FCQ mult")?;
                e.shift = parse_nums::<i32>(&toks[bar + 1..], "FCQ shift")?;
                e.have_q = true;
            }
            _ => {}
        }
    }
    Ok(m)
}

/// One fx line: `S a b` / `R x e` / `M x mult shift` (cases) or the same plus the result (TFLM output).
pub struct FxLine {
    pub kind: char,
    pub args: Vec<i64>,
}

pub fn parse_fx(text: &str) -> Result<Vec<FxLine>, String> {
    let mut v = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }
        let kind = toks[0].chars().next().unwrap_or('?');
        let mut args = Vec::with_capacity(toks.len() - 1);
        for t in &toks[1..] {
            args.push(t.parse::<i64>().map_err(|_| format!("fx line {}: bad int '{}'", ln + 1, t))?);
        }
        v.push(FxLine { kind, args });
    }
    Ok(v)
}

/// The ndarray result for an fx case (inputs must satisfy the contracts the generator respects).
/// Sources: ndarray/src/simd_fixedpoint.rs:35 (srdhm_i32), :74 (rounding_divide_by_pot_i32), :95
/// (multiply_by_quantized_multiplier_i32).
pub fn fx_ndarray(kind: char, a: &[i64]) -> Option<i32> {
    use ndarray::simd::{multiply_by_quantized_multiplier_i32, rounding_divide_by_pot_i32, srdhm_i32};
    match (kind, a.len()) {
        ('S', 2) => Some(srdhm_i32(a[0] as i32, a[1] as i32)),
        ('R', 2) => Some(rounding_divide_by_pot_i32(a[0] as i32, a[1] as i32)),
        ('M', 3) => Some(multiply_by_quantized_multiplier_i32(a[0] as i32, a[1] as i32, a[2] as i32)),
        _ => None,
    }
}

// ───────────────────────────── Part 3: exact rational + lab copies ─────────────────────────────

/// (mantissa, exponent) with value = mantissa * 2^exponent, positive finite f32 bit pattern.
pub fn f32_parts(bits: u32) -> Result<(u128, i32), String> {
    if bits >> 31 == 1 {
        return Err(format!("negative scale 0x{:08x}", bits));
    }
    let e = ((bits >> 23) & 0xff) as i32;
    let f = bits & 0x7f_ffff;
    if e == 0xff {
        return Err(format!("non-finite scale 0x{:08x}", bits));
    }
    if e == 0 {
        Ok((f as u128, -149)) // subnormal / zero
    } else {
        Ok((((f | 0x80_0000) as u128), e - 150)) // (1.f)*2^(e-127) = (f|2^23) * 2^(e-150)
    }
}

/// Exact `acc * (in_scale * w_scale / out_scale)` rounded half AWAY from zero, using only integer
/// arithmetic on the f32 bit patterns (no f64 rounding anywhere). Returns the rounded integer.
pub fn exact_scaled_round_away(acc: i64, in_bits: u32, w_bits: u32, out_bits: u32) -> Result<i64, String> {
    let (mi, ei) = f32_parts(in_bits)?;
    let (mw, ew) = f32_parts(w_bits)?;
    let (mo, eo) = f32_parts(out_bits)?;
    if mo == 0 {
        return Err("out_scale is zero".to_string());
    }
    if mi == 0 || mw == 0 || acc == 0 {
        return Ok(0);
    }
    // |acc| < 2^63, mi, mw < 2^24  =>  num < 2^111.
    let mut num: u128 = (acc.unsigned_abs() as u128) * mi * mw;
    let mut den: u128 = mo;
    let e = ei + ew - eo;
    if e >= 0 {
        if e > 16 {
            return Err(format!("binary exponent {} too large for u128", e));
        }
        num <<= e as u32;
    } else {
        let s = (-e) as u32;
        if s > 100 {
            return Err(format!("binary exponent -{} too small for u128", s));
        }
        den <<= s;
    }
    let (q, r) = (num / den, num % den);
    let q = if r >= den - r { q + 1 } else { q }; // ties away from zero (magnitude domain)
    if q > i64::MAX as u128 {
        return Err("rounded value exceeds i64".to_string());
    }
    let q = q as i64;
    Ok(if acc < 0 { -q } else { q })
}

/// VERBATIM COPY of ndarray/src/hpc/amx_matmul.rs:1019-1031 (private `subtract_i8_to_u8_bias`).
/// Panics on i32 overflow in a debug build, wraps in release.
pub fn subtract_i8_to_u8_bias(c: &mut [i32], b_i8: &[i8], m: usize, n: usize, k: usize) {
    let mut colsum = vec![0i32; n];
    for p in 0..k {
        for j in 0..n {
            colsum[j] += b_i8[p * n + j] as i32;
        }
    }
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] -= 128 * colsum[j];
        }
    }
}

/// Same arithmetic as above but with explicit wrapping ops (never panics) — used by the F9 shadow so the
/// observation does not depend on the build profile. NOT a production copy.
pub fn subtract_i8_to_u8_bias_wrapping(c: &mut [i32], b_i8: &[i8], m: usize, n: usize, k: usize) {
    let mut colsum = vec![0i32; n];
    for p in 0..k {
        for j in 0..n {
            colsum[j] = colsum[j].wrapping_add(b_i8[p * n + j] as i32);
        }
    }
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = c[i * n + j].wrapping_sub(128i32.wrapping_mul(colsum[j]));
        }
    }
}

// ───────────────────────────── Part 4: dot-product arms ─────────────────────────────

pub fn a_to_u8(a: &[i8]) -> Vec<u8> {
    // Same shift as amx_matmul.rs:964 / :975 / :986: (v as i32 + 128) as u8.
    a.iter().map(|&v| (v as i32 + 128) as u8).collect()
}

pub fn transpose(w_rows: &[i8], n: usize, k: usize) -> Vec<i8> {
    let mut t = vec![0i8; n * k];
    for c in 0..n {
        for p in 0..k {
            t[p * n + c] = w_rows[c * k + p];
        }
    }
    t
}

/// Inputs for the dot arms: activations a (m×k), weights as rows (n×k) and as W^T (k×n).
pub struct DotInputs<'a> {
    pub a: &'a [i8],
    pub w_rows: &'a [i8],
    pub wt: &'a [i8],
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

pub enum ArmOut {
    Cells(Vec<i64>),
    NotRun(String),
}

/// i64 oracle: sum_p a[i][p] * w[c][p], cell index i*n + c.
pub fn oracle(d: &DotInputs) -> Vec<i64> {
    let mut o = vec![0i64; d.m * d.n];
    for i in 0..d.m {
        for c in 0..d.n {
            let mut s = 0i64;
            for p in 0..d.k {
                s += d.a[i * d.k + p] as i64 * d.w_rows[c * d.k + p] as i64;
            }
            o[i * d.n + c] = s;
        }
    }
    o
}

pub const DOT_ARMS: [&str; 9] = [
    "scalar_i32",
    "uncorrected_u8xi8",
    "corrected_lab",
    "matmul_i8_to_i32",
    "vpdpbusd_zmm",
    "vpdpbusd_ymm",
    "amx_tiled",
    "simd_int_ops_dot_i8",
    "vnni_dot_u8_i8_corrected",
];

/// Which tier `matmul_i8_to_i32` takes for this shape (mirrors amx_matmul.rs:962-1005).
pub fn matmul_tier(m: usize, n: usize, k: usize) -> &'static str {
    if ndarray::hpc::amx_matmul::amx_available() && m % 16 == 0 && n % 16 == 0 && k % 64 == 0 {
        return "amx";
    }
    if std::is_x86_feature_detected!("avx512vnni") {
        return "avx512vnni_zmm";
    }
    if std::is_x86_feature_detected!("avxvnni") {
        return "avxvnni_ymm";
    }
    "scalar"
}

fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic".to_string()
    }
}

/// Run one named arm. Panics inside the arm are caught and reported as NotRun("PANIC: ...").
pub fn run_dot_arm(name: &str, d: &DotInputs) -> ArmOut {
    match catch_unwind(AssertUnwindSafe(|| run_dot_arm_inner(name, d))) {
        Ok(o) => o,
        Err(p) => ArmOut::NotRun(format!("PANIC: {}", panic_msg(p))),
    }
}

fn run_dot_arm_inner(name: &str, d: &DotInputs) -> ArmOut {
    let (m, n, k) = (d.m, d.n, d.k);
    match name {
        "scalar_i32" => {
            // Lab loop identical to ndarray/src/hpc/amx_matmul.rs:991-998 (tier 4); `wrapping_add` so the
            // lab arm behaves like the release build in any profile (values here never reach the wrap).
            let mut c = vec![0i32; m * n];
            for i in 0..m {
                for p in 0..k {
                    let av = d.a[i * k + p] as i32;
                    for j in 0..n {
                        c[i * n + j] = c[i * n + j].wrapping_add(av * d.wt[p * n + j] as i32);
                    }
                }
            }
            ArmOut::Cells(c.into_iter().map(|x| x as i64).collect())
        }
        "uncorrected_u8xi8" => {
            let mut c = vec![0i64; m * n];
            for i in 0..m {
                for c_ in 0..n {
                    let mut s = 0i64;
                    for p in 0..k {
                        s += (d.a[i * k + p] as i64 + 128) * d.w_rows[c_ * k + p] as i64;
                    }
                    c[i * n + c_] = s;
                }
            }
            ArmOut::Cells(c)
        }
        "corrected_lab" => {
            let ArmOut::Cells(mut c) = run_dot_arm_inner("uncorrected_u8xi8", d) else {
                return ArmOut::NotRun("uncorrected failed".into());
            };
            for c_ in 0..n {
                let colsum: i64 = (0..k).map(|p| d.w_rows[c_ * k + p] as i64).sum();
                for i in 0..m {
                    c[i * n + c_] -= 128 * colsum;
                }
            }
            ArmOut::Cells(c)
        }
        "matmul_i8_to_i32" => {
            use ndarray::hpc::amx_matmul::matmul_i8_to_i32; // amx_matmul.rs:951 (pub fn)
            use ndarray::{Array2, ArrayView2};
            let lhs = ArrayView2::from_shape((m, k), d.a).map_err(|e| format!("{:?}", e));
            let rhs = ArrayView2::from_shape((k, n), d.wt).map_err(|e| format!("{:?}", e));
            let (lhs, rhs) = match (lhs, rhs) {
                (Ok(l), Ok(r)) => (l, r),
                (Err(e), _) | (_, Err(e)) => return ArmOut::NotRun(format!("shape: {}", e)),
            };
            let mut out = Array2::<i32>::zeros((m, n));
            if let Err(e) = matmul_i8_to_i32(lhs, rhs, out.view_mut()) {
                return ArmOut::NotRun(format!("MatmulError: {:?}", e));
            }
            ArmOut::Cells(out.iter().map(|&x| x as i64).collect())
        }
        "vpdpbusd_zmm" => {
            if !(std::is_x86_feature_detected!("avx512vnni") && std::is_x86_feature_detected!("avx512f")) {
                return ArmOut::NotRun("avx512vnni/avx512f absent".into());
            }
            let a_u8 = a_to_u8(d.a);
            let mut c = vec![0i32; m * n];
            // SAFETY: avx512vnni + avx512f verified at runtime just above.
            // int8_tile_gemm.rs:150 `pub unsafe fn int8_gemm_vpdpbusd_zmm(a_u8,b_i8,c,m,n,k)`, B is row-major k×n.
            unsafe { ndarray::hpc::int8_tile_gemm::int8_gemm_vpdpbusd_zmm(&a_u8, d.wt, &mut c, m, n, k) };
            subtract_i8_to_u8_bias(&mut c, d.wt, m, n, k);
            ArmOut::Cells(c.into_iter().map(|x| x as i64).collect())
        }
        "vpdpbusd_ymm" => {
            if !(std::is_x86_feature_detected!("avxvnni") && std::is_x86_feature_detected!("avx2")) {
                return ArmOut::NotRun("avxvnni/avx2 absent".into());
            }
            let a_u8 = a_to_u8(d.a);
            let mut c = vec![0i32; m * n];
            // SAFETY: avxvnni + avx2 verified at runtime just above. int8_tile_gemm.rs:250 (same signature).
            unsafe { ndarray::hpc::int8_tile_gemm::int8_gemm_vpdpbusd_ymm(&a_u8, d.wt, &mut c, m, n, k) };
            subtract_i8_to_u8_bias(&mut c, d.wt, m, n, k);
            ArmOut::Cells(c.into_iter().map(|x| x as i64).collect())
        }
        "amx_tiled" => {
            if !ndarray::hpc::amx_matmul::amx_available() {
                return ArmOut::NotRun("amx_available() == false".into());
            }
            // int8_tile_gemm.rs:357 `pub fn int8_gemm_amx_tiled(a_u8,b_i8,c,m,n,k)`; asserts (L368-384 as read):
            // lengths, m%16==0, n%16==0, k%64==0, amx_available(). B is row-major k×n (the serial/rb drivers read b_i8[kk*n + j_tile ..], row-major).
            if m % 16 != 0 || n % 16 != 0 || k % 64 != 0 {
                return ArmOut::NotRun(format!("shape m={} n={} k={} not 16/16/64-aligned", m, n, k));
            }
            let a_u8 = a_to_u8(d.a);
            let mut c = vec![0i32; m * n];
            ndarray::hpc::int8_tile_gemm::int8_gemm_amx_tiled(&a_u8, d.wt, &mut c, m, n, k);
            subtract_i8_to_u8_bias(&mut c, d.wt, m, n, k);
            ArmOut::Cells(c.into_iter().map(|x| x as i64).collect())
        }
        "simd_int_ops_dot_i8" => {
            // ndarray/src/simd_int_ops.rs:201 (pub fn, `pub mod simd_int_ops` at lib.rs:468 under std).
            let mut c = vec![0i64; m * n];
            for i in 0..m {
                for c_ in 0..n {
                    c[i * n + c_] =
                        ndarray::simd_int_ops::dot_i8(&d.a[i * k..(i + 1) * k], &d.w_rows[c_ * k..(c_ + 1) * k]) as i64;
                }
            }
            ArmOut::Cells(c)
        }
        "vnni_dot_u8_i8_corrected" => vnni_dot_cells(d),
        other => ArmOut::NotRun(format!("unknown arm {}", other)),
    }
}

#[cfg(feature = "rd")]
fn vnni_dot_cells(d: &DotInputs) -> ArmOut {
    // ndarray/src/simd_runtime/vnni_dot.rs:52 `pub fn vnni_dot_u8_i8(a: &[u8], b: &[i8]) -> i32`,
    // re-exported at ndarray/src/simd.rs:827 (cfg runtime-dispatch).
    let (m, n, k) = (d.m, d.n, d.k);
    let a_u8 = a_to_u8(d.a);
    let mut c = vec![0i64; m * n];
    for i in 0..m {
        for c_ in 0..n {
            let row = &d.w_rows[c_ * k..(c_ + 1) * k];
            let colsum: i64 = row.iter().map(|&x| x as i64).sum();
            let raw = ndarray::simd::vnni_dot_u8_i8(&a_u8[i * k..(i + 1) * k], row) as i64;
            c[i * n + c_] = raw - 128 * colsum;
        }
    }
    ArmOut::Cells(c)
}

#[cfg(not(feature = "rd"))]
fn vnni_dot_cells(_d: &DotInputs) -> ArmOut {
    ArmOut::NotRun("built without feature rd (ndarray/runtime-dispatch)".into())
}
