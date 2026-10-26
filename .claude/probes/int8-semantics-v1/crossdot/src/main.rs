//! int8 dot products against a widened oracle: Σ i64(a)·i64(w).
//! Arms (each only where it exists for the build target):
//!   scalar   ndarray::simd_int_ops::dot_i8(&[i8], &[i8]) -> i32   (src/simd_int_ops.rs:201, wrapping i32)
//!   neon     ndarray::simd_neon::dot_i8x16_neon(&[i8;16], &[i8;16]) (aarch64; vmull_s8 + vpaddlq_s16)
//!   wasm     ndarray::simd_wasm::wasm32_simd::dot_i8_to_i32_wasm(&[i8], &[i8], k) (wasm32 + simd128)
//! Prints one line per (arm, fixture): `ARM name fixture cases mismatches first_counterexample`.

fn oracle(a: &[i8], w: &[i8]) -> i64 { a.iter().zip(w).map(|(&x, &y)| i64::from(x) * i64::from(y)).sum() }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn i8_full(&mut self) -> i8 { (self.next() >> 56) as u8 as i8 }
    fn i8_narrow(&mut self) -> i8 { loop { let v = self.i8_full(); if v != -128 { return v; } } }
}

const PAIRS: [(i8, i8); 8] = [(-128, -128), (-128, -127), (-128, 127), (-1, -128), (0, -128), (1, -128), (127, -128), (127, 127)];
const LENS: [usize; 16] = [1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 255, 256, 257, 4096];

fn fixtures() -> Vec<(String, Vec<i8>, Vec<i8>)> {
    let mut out = Vec::new();
    for &(a, w) in &PAIRS {
        for &n in &[1usize, 16, 64] { out.push((format!("pair({a},{w})x{n}"), vec![a; n], vec![w; n])); }
    }
    let mut r = Rng(0x9E3779B97F4A7C15);
    for &n in &LENS {
        for rep in 0..50 {
            let a: Vec<i8> = (0..n).map(|_| r.i8_full()).collect();
            let wc: Vec<i8> = (0..n).map(|_| r.i8_narrow()).collect();
            let we: Vec<i8> = (0..n).map(|_| r.i8_full()).collect();
            out.push((format!("conforming_n{n}_{rep}"), a.clone(), wc));
            out.push((format!("extended_n{n}_{rep}"), a.clone(), we));
            // cancellation: w then -w halves (with -128 kept unpaired when it appears)
            let mut c = a.clone(); c.extend(a.iter().map(|&x| x.wrapping_neg()));
            let mut cw = vec![37i8; n]; cw.extend(vec![37i8; n]);
            out.push((format!("cancel_n{n}_{rep}"), c, cw));
        }
    }
    out
}

fn report(arm: &str, f: &dyn Fn(&[i8], &[i8]) -> Option<i64>) {
    let mut by: std::collections::BTreeMap<String, (usize, usize, String)> = Default::default();
    for (name, a, w) in fixtures() {
        let first = name.split('_').next().unwrap();
        let group = if first.starts_with("pair") { first.split('x').next().unwrap() } else { first }.to_string();
        let Some(got) = f(&a, &w) else { continue };
        let want = oracle(&a, &w);
        let e = by.entry(group).or_insert((0, 0, String::new()));
        e.0 += 1;
        if got != want { e.1 += 1; if e.2.is_empty() { e.2 = format!("{name}: got {got} want {want}"); } }
    }
    for (g, (n, m, ce)) in by { println!("ARM {arm} {g} cases={n} mismatches={m} {ce}"); }
}

fn main() {
    println!("TARGET {} {}", std::env::consts::ARCH, if cfg!(target_feature = "simd128") { "simd128" } else { "" });
    report("scalar_simd_int_ops_dot_i8", &|a, w| Some(i64::from(ndarray::simd_int_ops::dot_i8(a, w))));
    // the doc claim "with 4M lanes the sum stays well within i32::MAX" (simd_int_ops.rs:195-196): worst lane product is
    // (-128)*(-128) = 16384, so 131_072 such lanes reach 2^31
    for &n in &[131_071usize, 131_072, 140_000, 4_000_000] {
        let a = vec![-128i8; n];
        let got = i64::from(ndarray::simd_int_ops::dot_i8(&a, &a));
        println!("SCALAR_ACC_LIMIT n={n} got={got} want={} exact={}", oracle(&a, &a), got == oracle(&a, &a));
    }
    #[cfg(target_arch = "x86_64")]
    {
        // simd_avx2::dot_i8(&[u8], &[u8]) -> i64 reads each byte as i8 (src/simd_avx2.rs:411-421, raw bit pattern).
        report("avx2_dot_i8_raw_bytes", &|a, w| {
            let ra: Vec<u8> = a.iter().map(|&x| x as u8).collect();
            let rw: Vec<u8> = w.iter().map(|&x| x as u8).collect();
            Some(ndarray::simd_avx2::dot_i8(&ra, &rw))
        });
        // the same function handed Codec8 U8 (biased, code + 128) storage: a layout mismatch, not a numeric one
        report("avx2_dot_i8_fed_codec8_biased_u8", &|a, w| {
            let ba: Vec<u8> = a.iter().map(|&x| (i32::from(x) + 128) as u8).collect();
            let bw: Vec<u8> = w.iter().map(|&x| (i32::from(x) + 128) as u8).collect();
            Some(ndarray::simd_avx2::dot_i8(&ba, &bw))
        });
    }
    #[cfg(target_arch = "aarch64")]
    report("neon_dot_i8x16_neon", &|a, w| {
        if a.len() % 16 != 0 { return None; }
        let mut s = 0i64;
        for (ca, cw) in a.chunks_exact(16).zip(w.chunks_exact(16)) {
            // SAFETY: aarch64 always has NEON; both chunks are exactly 16 bytes.
            s += i64::from(unsafe { ndarray::simd_neon::dot_i8x16_neon(ca.try_into().unwrap(), cw.try_into().unwrap()) });
        }
        Some(s)
    });
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    report("wasm_dot_i8_to_i32_wasm", &|a, w| Some(i64::from(ndarray::simd_wasm::wasm32_simd::dot_i8_to_i32_wasm(a, w, a.len()))));
}
