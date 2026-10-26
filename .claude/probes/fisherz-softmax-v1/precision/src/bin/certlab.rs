//! Amendment 6.1 Rust certification lab (R0 parity, R1 arms, R2 seams, R3 timing).
//!
//! Input: the canonical fixture written by `cert2.py OUT FIXTURE` (never regenerated
//! here, so Rust and Python see the same bytes). Per case, 14 lines:
//!   case cid beta_hex n d m mu
//!   x[256] f64 hex
//!   q[n]
//!   V[n*d] f64 hex (element-major)
//!   rowmax table "W:e" x256   (W = certified F24 weight relative to x[m], -1 = absent;
//!                              e = certified per-entry error, eps = e / 2^8)
//!   unary table  "W:e" x256   (relative to the codebook maximum x[mu])
//!   ref y codes | ref p codes   (40-digit mpmath reference, from Python)
//!   7 arm lines: arm rowclass yflags pflags ycodes... pcodes...
//!     (flags: B = certified by the bound, S = exact shortcut, x = other resolver;
//!      codes = Python's FINAL codes after its certified resolver)
//!
//! All certification arithmetic is exact integers with checked operations; any
//! overflow is a refusal (counted), never a wrap. No floating point is used to
//! decide a certification. f64 appears only to (a) decompose V exactly into
//! mantissa/exponent and (b) the uncertified f64 table generator compared against
//! the certified table.
//!
//! The interval resolver is NOT implemented in Rust (no multiprecision dependency):
//! outputs needing it are DEFERRED to the Python reference result, and counted.
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::time::Instant;

const F: u32 = 24;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Res { Bound, Shortcut, ExactRational, GroupMean, Deferred }

struct Case { cid: usize, beta: f64, n: usize, d: usize, m: usize, mu: usize, x: Vec<f64>, q: Vec<u8>,
    v: Vec<f64>, wrow: Vec<(i64, i64)>, wuni: Vec<(i64, i64)>, yref: Vec<i32>, pref: Vec<i32>,
    arms: Vec<(String, String, String, String, Vec<Option<i32>>, Vec<Option<i32>>)> }

fn hx(s: &str) -> f64 { f64::from_bits(u64::from_str_radix(s, 16).expect("hex")) }
fn we(s: &str) -> (i64, i64) { let mut it = s.split(':'); (it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap()) }
fn opt(s: &str) -> Option<i32> { if s == "None" { None } else { Some(s.parse().unwrap()) } }

fn read(path: &str) -> Vec<Case> {
    let f = BufReader::new(std::fs::File::open(path).expect("fixture"));
    let lines: Vec<String> = f.lines().map(|l| l.unwrap()).collect();
    let mut out = vec![];
    for chunk in lines.chunks(14) {
        let h: Vec<&str> = chunk[0].split_whitespace().collect();
        assert_eq!(h[0], "case");
        let (n, d) = (h[3].parse::<usize>().unwrap(), h[4].parse::<usize>().unwrap());
        let refs: Vec<&str> = chunk[6].split(" | ").collect();
        let mut arms = vec![];
        for l in &chunk[7..14] {
            let t: Vec<&str> = l.split_whitespace().collect();
            let yc = t[4..4 + d].iter().map(|s| opt(s)).collect();
            let pc = t[4 + d..4 + d + n].iter().map(|s| opt(s)).collect();
            arms.push((t[0].to_string(), t[1].to_string(), t[2].to_string(), t[3].to_string(), yc, pc));
        }
        out.push(Case { cid: h[1].parse().unwrap(), beta: hx(h[2]), n, d, m: h[5].parse().unwrap(), mu: h[6].parse().unwrap(),
            x: chunk[1].split_whitespace().map(hx).collect(), q: chunk[2].split_whitespace().map(|s| s.parse().unwrap()).collect(),
            v: chunk[3].split_whitespace().map(hx).collect(), wrow: chunk[4].split_whitespace().map(we).collect(),
            wuni: chunk[5].split_whitespace().map(we).collect(),
            yref: refs[0].split_whitespace().map(|s| s.parse().unwrap()).collect(),
            pref: refs[1].split_whitespace().map(|s| s.parse().unwrap()).collect(), arms });
    }
    out
}

/// Exact decomposition v = m * 2^e (m signed), for finite v.
fn decompose(v: f64) -> (i128, i32) {
    assert!(v.is_finite());
    if v == 0.0 { return (0, 0); }
    let b = v.to_bits(); let sign = b >> 63; let ex = ((b >> 52) & 0x7ff) as i32; let fr = b & ((1u64 << 52) - 1);
    let (m, e) = if ex == 0 { (fr, -1074) } else { (fr | (1u64 << 52), ex - 1075) };
    (if sign == 1 { -(m as i128) } else { m as i128 }, e)
}

/// 128x128 -> 256-bit product, compared with a 128-bit value: returns a*b <= c (le) / a*b < c (lt).
fn mul_cmp(a: u128, b: u128, c: u128, strict: bool) -> bool {
    let (a1, a0) = (a >> 64, a & u64::MAX as u128); let (b1, b0) = (b >> 64, b & u64::MAX as u128);
    let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
    let mid = (p00 >> 64) + (p01 & u64::MAX as u128) + (p10 & u64::MAX as u128);
    let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    let lo = (p00 & u64::MAX as u128) | ((mid & u64::MAX as u128) << 64);
    if hi != 0 { return false; }
    if strict { lo < c } else { lo <= c }
}

#[derive(Default)]
struct Widths { s: u32, n: u32, a_sum: u32, t: u32, overflow_refusals: u64 }
fn bits(x: u128) -> u32 { 128 - x.leading_zeros() }

fn code_y_exact(num: i128, den: u128) -> i32 { // round-away(num/den), clamp
    let a = num.unsigned_abs(); let r = (2 * a + den) / (2 * den);
    let r = r.min(1000) as i32; let r = if num < 0 { -r } else { r }; r.clamp(-128, 127)
}
fn code_p_exact(w: u128, s: u128) -> i32 { (((512 * w + s) / (2 * s)) as i64 - 128).clamp(-128, 127) as i32 }

struct Row { yhat: Vec<Option<i32>>, ycert: Vec<bool>, phat: Vec<Option<i32>>, pcert: Vec<bool> }

/// Arm A + ledger-G bound in exact integers. w: certified F24 weights per element;
/// e: per-element error numerators at grid g (eps = e / 2^g).
fn bound(cs: &Case, w: &[u128], e: &[u128], g: u32, wd: &mut Widths) -> Row {
    let (n, d) = (cs.n, cs.d);
    let s: u128 = w.iter().sum(); wd.s = wd.s.max(bits(s));
    let se: u128 = e.iter().sum();
    let den = (s << g).checked_sub(se).filter(|&x| x > 0); // S > E
    let mut row = Row { yhat: vec![], ycert: vec![], phat: vec![], pcert: vec![] };
    if s == 0 {
        row.yhat = vec![None; d]; row.ycert = vec![false; d]; row.phat = vec![None; n]; row.pcert = vec![false; n]; return row;
    }
    for k in 0..d {
        let parts: Vec<(i128, i32)> = (0..n).map(|i| decompose(cs.v[i * d + k])).collect();
        let e0 = parts.iter().filter(|p| p.0 != 0).map(|p| p.1).min().unwrap_or(-1).min(-1);
        let mut ok = true;
        let mm: Vec<i128> = parts.iter().map(|&(m, ex)| {
            let sh = (ex - e0) as u32; if sh >= 127 { ok = false; return 0 }
            m.checked_mul(1i128 << sh).unwrap_or_else(|| { ok = false; 0 })
        }).collect();
        let mut nsum: i128 = 0;
        for i in 0..n { match (w[i] as i128).checked_mul(mm[i]).and_then(|p| nsum.checked_add(p)) { Some(v) => nsum = v, None => ok = false } }
        let t = s.checked_shl((-e0) as u32).filter(|_| (bits(s) + (-e0) as u32) < 127);
        if !ok || t.is_none() { wd.overflow_refusals += 1; row.yhat.push(None); row.ycert.push(false); continue; }
        let t = t.unwrap(); wd.n = wd.n.max(bits(nsum.unsigned_abs())); wd.t = wd.t.max(bits(t));
        row.yhat.push(Some(code_y_exact(nsum, t)));
        let Some(den) = den else { row.ycert.push(false); continue };
        // R = 2 * sum e_i |M_i S - N|
        let mut r: u128 = 0; let mut ok2 = true;
        for i in 0..n {
            if e[i] == 0 { continue; }
            let a = mm[i].checked_mul(s as i128).and_then(|x| x.checked_sub(nsum)).map(|x| x.unsigned_abs());
            match a.and_then(|a| a.checked_mul(e[i])).and_then(|x| r.checked_add(x)) { Some(v) => r = v, None => ok2 = false }
        }
        let r = r.checked_mul(2);
        if !ok2 || r.is_none() { wd.overflow_refusals += 1; row.ycert.push(false); continue; }
        let r = r.unwrap(); wd.a_sum = wd.a_sum.max(bits(r));
        // relevant boundaries h/2, h odd in [-255, 253]; nearest below-or-equal and above yhat = nsum/t
        let two_n = nsum * 2; let ti = t as i128;
        let hf = two_n.div_euclid(ti); // floor(2y)
        let hlo = if hf.rem_euclid(2) == 0 { hf - 1 } else { hf };
        let mut cands: Vec<i128> = vec![];
        if hlo > 253 { cands.push(253); } else if hlo < -255 { cands.push(-255); } else { cands.push(hlo); if hlo + 2 <= 253 { cands.push(hlo + 2); } }
        let mut hit = false;
        for &h in &cands {
            let x = (h * ti - two_n).unsigned_abs(); // |b - y| * 2t
            let b_le_y = h * ti <= two_n; let b_eq_y = h * ti == two_n;
            let strict = if h > 0 { b_le_y } else { b_eq_y || !b_le_y };
            if mul_cmp(x, den, r, strict) { hit = true; }
        }
        row.ycert.push(!hit);
    }
    for i in 0..n {
        row.phat.push(Some(code_p_exact(w[i], s)));
        let Some(den) = den else { row.pcert.push(false); continue };
        let rp = 512 * (e[i] * s + w[i] * se);
        let hf = (512 * w[i]) / s; // floor(512 p)
        let hlo = if hf % 2 == 0 { hf as i128 - 1 } else { hf as i128 };
        let mut cands = vec![];
        if hlo >= 1 { cands.push(hlo.min(509)); }
        if hlo + 2 <= 509 && hlo < 509 { cands.push((hlo + 2).max(1)); }
        let mut hit = false;
        for &h in &cands {
            let hs = h as u128 * s; let pw = 512 * w[i];
            let x = hs.abs_diff(pw); let strict = hs <= pw; // b <= p -> strict
            if mul_cmp(x, den, rp, strict) { hit = true; }
        }
        row.pcert.push(!hit);
    }
    row
}

fn key(v: f64) -> u64 { if v == 0.0 { 0 } else { v.to_bits() } }

/// Rust-side exact resolvers; anything else is DEFERRED to the Python reference result.
fn resolve(cs: &Case, row: &Row, py_y: &[Option<i32>], py_p: &[Option<i32>]) -> (Vec<(i32, Res)>, Vec<(i32, Res)>) {
    let (n, d) = (cs.n, cs.d);
    let xs: Vec<f64> = cs.q.iter().map(|&a| cs.x[a as usize]).collect();
    let uniform = xs.iter().all(|&v| key(v) == key(xs[0]));
    let mut ys = vec![]; let mut ps = vec![];
    for k in 0..d {
        if row.ycert[k] { ys.push((row.yhat[k].unwrap(), Res::Bound)); continue; }
        let (c, how) = exact_y(cs, &xs, k, uniform);
        match c { Some(c) => ys.push((c, how)), None => ys.push((py_y[k].expect("python resolved"), Res::Deferred)) }
    }
    for i in 0..n {
        if row.pcert[i] { ps.push((row.phat[i].unwrap(), Res::Bound)); }
        else if uniform { ps.push((code_p_exact(1, n as u128), Res::ExactRational)); }
        else { ps.push((py_p[i].expect("python resolved"), Res::Deferred)); }
    }
    (ys, ps)
}

/// Exact y: single representative -> mean; equal group means -> that mean; else None.
fn exact_y(cs: &Case, xs: &[f64], k: usize, uniform: bool) -> (Option<i32>, Res) {
    let (n, d) = (cs.n, cs.d);
    let parts: Vec<(i128, i32)> = (0..n).map(|i| decompose(cs.v[i * d + k])).collect();
    let e0 = parts.iter().filter(|p| p.0 != 0).map(|p| p.1).min().unwrap_or(-1).min(-1);
    let mm: Vec<i128> = parts.iter().map(|&(m, ex)| m << ((ex - e0) as u32)).collect();
    let unit: u128 = 1u128 << ((-e0) as u32);
    if uniform {
        let sum: i128 = mm.iter().sum();
        return (Some(code_y_exact(sum, unit * n as u128)), Res::ExactRational);
    }
    let mut groups: Vec<(u64, i128, i128)> = vec![]; // key, sum M, count
    for i in 0..n {
        let kk = key(xs[i]);
        match groups.iter_mut().find(|g| g.0 == kk) { Some(g) => { g.1 += mm[i]; g.2 += 1 } None => groups.push((kk, mm[i], 1)) }
    }
    let (s0, c0) = (groups[0].1, groups[0].2);
    if groups.iter().all(|g| g.1 * c0 == s0 * g.2) {
        return (Some(code_y_exact(s0, unit * c0 as u128)), Res::GroupMean);
    }
    (None, Res::Deferred)
}

fn shortcut(cs: &Case) -> Option<(Vec<i32>, i32)> {
    let xs: Vec<f64> = cs.q.iter().map(|&a| cs.x[a as usize]).collect();
    if !xs.iter().all(|&v| key(v) == key(xs[0])) { return None; }
    let ys = (0..cs.d).map(|k| exact_y(cs, &xs, k, true).0.unwrap()).collect();
    Some((ys, code_p_exact(1, cs.n as u128)))
}

/// Uncertified f64 generator, for comparison with the certified table only.
fn f64_row(cs: &Case, m: usize) -> Vec<i64> {
    (0..256).map(|a| if cs.x[a] > cs.x[m] { -1 } else { ((cs.beta * (cs.x[a] - cs.x[m])).exp() * (1u64 << F) as f64).round() as i64 }).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cases = read(&args[1]);
    let mut wd = Widths::default();
    let (mut f64_dis, mut f64_dis_half) = (0u64, 0u64);
    let arm_spec: [(&str, bool, &str, bool); 7] = [("base", false, "half", false), ("zero", false, "zero", false), ("pe", false, "pe", false),
        ("first", false, "half", true), ("first_pe", false, "pe", true), ("unary", true, "half", false), ("unary_pe", true, "pe", false)];
    println!("arm flag_mismatch_y flag_mismatch_p rowclass_mismatch hat_code_mismatch_certified final_vs_ref_wrong misses deferred_outputs rows_shortcut rows_bound rows_escalated");
    for (ai, &(arm, unary, err, first)) in arm_spec.iter().enumerate() {
        let (mut fy, mut fp, mut rcm, mut hm, mut wrong, mut miss, mut deferred) = (0, 0, 0, 0, 0, 0, 0);
        let (mut r_sc, mut r_b, mut r_e) = (0, 0, 0);
        for cs in &cases {
            let (pa, prc, pyf, ppf, pyc, ppc) = &cs.arms[ai]; assert_eq!(pa, arm);
            if ai == 0 { // table parity, once per case
                let fr = f64_row(cs, cs.m);
                for a in 0..256 { if cs.wrow[a].0 >= 0 { f64_dis += (fr[a] != cs.wrow[a].0) as u64; } }
                let _ = &mut f64_dis_half;
            }
            let (ycls, pcls, ycodes, pcodes, rc): (Vec<Res>, Vec<Res>, Vec<i32>, Vec<i32>, &str);
            let sc = if first { shortcut(cs) } else { None };
            if let Some((yv, pv)) = sc {
                ycls = vec![Res::Shortcut; cs.d]; pcls = vec![Res::Shortcut; cs.n]; ycodes = yv; pcodes = vec![pv; cs.n]; rc = "shortcut";
            } else {
                let tab = if unary { &cs.wuni } else { &cs.wrow };
                let w: Vec<u128> = cs.q.iter().map(|&a| tab[a as usize].0 as u128).collect();
                let (e, g): (Vec<u128>, u32) = match err {
                    "half" => (vec![1; cs.n], 1),
                    "zero" => (cs.q.iter().map(|&a| if tab[a as usize].1 == 0 { 0 } else { 1 }).collect(), 1),
                    _ => (cs.q.iter().map(|&a| tab[a as usize].1 as u128).collect(), 8),
                };
                let row = bound(cs, &w, &e, g, &mut wd);
                let all = row.ycert.iter().all(|&b| b) && row.pcert.iter().all(|&b| b);
                rc = if all { "bound" } else { "escalated" };
                for k in 0..cs.d { if row.ycert[k] && row.yhat[k] != pyc[k] { hm += 1 } }
                for i in 0..cs.n { if row.pcert[i] && row.phat[i] != ppc[i] { hm += 1 } }
                let (ys, ps) = resolve(cs, &row, pyc, ppc);
                ycls = ys.iter().map(|t| t.1).collect(); pcls = ps.iter().map(|t| t.1).collect();
                ycodes = ys.iter().map(|t| t.0).collect(); pcodes = ps.iter().map(|t| t.0).collect();
            }
            let fl = |c: &[Res]| c.iter().map(|r| match r { Res::Bound => 'B', Res::Shortcut => 'S', _ => 'x' }).collect::<String>();
            fy += (fl(&ycls) != *pyf) as u64; fp += (fl(&pcls) != *ppf) as u64;
            let pyrc = if prc == "UNRESOLVED" { "escalated" } else { prc.as_str() };
            rcm += (rc != pyrc) as u64;
            match rc { "shortcut" => r_sc += 1, "bound" => r_b += 1, _ => r_e += 1 }
            for k in 0..cs.d { if ycodes[k] != cs.yref[k] { wrong += 1; if matches!(ycls[k], Res::Bound | Res::Shortcut) { miss += 1 } } if ycls[k] == Res::Deferred { deferred += 1 } }
            for i in 0..cs.n { if pcodes[i] != cs.pref[i] { wrong += 1; if matches!(pcls[i], Res::Bound | Res::Shortcut) { miss += 1 } } if pcls[i] == Res::Deferred { deferred += 1 } }
        }
        println!("{arm} {fy} {fp} {rcm} {hm} {wrong} {miss} {deferred} {r_sc} {r_b} {r_e}");
    }
    println!("cases {} f64_table_disagreements {} widths_bits S={} N={} R={} T={} overflow_refusals={}",
             cases.len(), f64_dis, wd.s, wd.n, wd.a_sum, wd.t, wd.overflow_refusals);
    if args.len() > 2 && args[2] == "--bench" { bench(&cases); }
    let _ = cases.iter().map(|c| (c.cid, c.mu)).count();
}

fn bench(cases: &[Case]) {
    // the 4096-element rows; ns/element per stage, median of 7 timed batches
    let rows: Vec<&Case> = cases.iter().filter(|c| c.n == 4096 && c.arms[3].1 != "shortcut").take(6).collect();
    let tie: &Case = cases.iter().find(|c| c.n == 4096 && c.arms[3].1 == "shortcut").unwrap();
    let med = |f: &mut dyn FnMut() -> u64, elems: usize| -> f64 {
        let mut v: Vec<f64> = (0..7).map(|_| { let t = Instant::now(); let mut acc = 0u64; for _ in 0..20 { acc = acc.wrapping_add(f()); } black_box(acc); t.elapsed().as_nanos() as f64 / (20 * elems) as f64 }).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[3]
    };
    let mut wd = Widths::default();
    for cs in &rows {
        let w: Vec<u128> = cs.q.iter().map(|&a| cs.wrow[a as usize].0 as u128).collect();
        let e1 = vec![1u128; cs.n]; let epe: Vec<u128> = cs.q.iter().map(|&a| cs.wrow[a as usize].1 as u128).collect();
        let t_gen = med(&mut || f64_row(cs, cs.m).iter().map(|&x| x as u64).sum(), 256) ;
        let t_sc = med(&mut || shortcut(cs).is_some() as u64, cs.n);
        let t_sum = med(&mut || { let s: u128 = w.iter().sum(); let mut n: i128 = 0; for k in 0..cs.d { for i in 0..cs.n { n = n.wrapping_add(w[i] as i128 * decompose(cs.v[i * cs.d + k]).0); } } (s as u64) ^ (n as u64) }, cs.n);
        let t_half = med(&mut || bound(cs, &w, &e1, 1, &mut wd).pcert.iter().filter(|&&b| b).count() as u64, cs.n);
        let t_pe = med(&mut || bound(cs, &w, &epe, 8, &mut wd).pcert.iter().filter(|&&b| b).count() as u64, cs.n);
        println!("bench case {} {} table_gen_ns_per_entry {:.2} shortcut_check {:.2} int_sums {:.2} bound_half_full_row {:.2} bound_pe_full_row {:.2}",
                 cs.cid, cs.arms[0].1, t_gen, t_sc, t_sum, t_half, t_pe);
    }
    let t_tie = med(&mut || shortcut(tie).unwrap().1 as u64, tie.n);
    println!("bench shortcut_resolve_tie_row case {} ns/elem {:.2}", tie.cid, t_tie);
}
