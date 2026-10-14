//! Coverage orientation: the thought x identity coverage matrix (65,536 rows x
//! 64 thoughts) stored per thought (C: 64 masks of 65,536 bits) or per row
//! (R: one 64-bit written-by word per row), measured on four count queries.
//!
//! Pre-registered in `.claude/plans/ce64-hypothesis-layers-v1.md`
//! ("Coverage orientation: per-thought masks or per-row words").
//!
//! - Q-ID: distinct NodeGuid covered by a thought in T, in the aperture.
//! - Q-TH: distinct thoughts in T covering a row in the aperture.
//! - Q-PAIR: (NodeGuid, thought) observations in the aperture.
//! - Q-PERSP: thoughts in T covering one row.
//!
//! An aperture is a contiguous row range, as a Morton prefix gives.
//! No crate types; builds with plain `rustc -O`.
//! Run: `cargo run --release --example coverage_orientation_probe` (tier printed).
//! Tests: `cargo test --example coverage_orientation_probe`.

use std::hint::black_box;
use std::time::Instant;

const ROWS: usize = 65_536;
const WORDS: usize = ROWS / 64;
const THOUGHTS: usize = 64;

fn tier() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "v4 (avx512f)"
    } else if cfg!(target_feature = "avx2") {
        "v3 (avx2): timings not reportable"
    } else {
        "below v3: timings not reportable"
    }
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Per-row orientation: bit t of `w[r]` is set when thought t covers row r.
struct RowWords {
    w: Vec<u64>,
}

/// Per-thought orientation: bit r of `m[t]` is set when thought t covers row r.
struct ThoughtMasks {
    m: Vec<[u64; WORDS]>,
}

/// Independent coverage with probability `d_ppm` per million.
fn fixture(d_ppm: u64, seed: u64) -> RowWords {
    let mut w = vec![0u64; ROWS];
    for (r, word) in w.iter_mut().enumerate() {
        for t in 0..THOUGHTS {
            let h = mix(seed ^ ((r as u64) << 8) ^ t as u64);
            if h % 1_000_000 < d_ppm {
                *word |= 1 << t;
            }
        }
    }
    RowWords { w }
}

fn transpose_rc(r: &RowWords) -> ThoughtMasks {
    let mut m = vec![[0u64; WORDS]; THOUGHTS];
    for (row, &word) in r.w.iter().enumerate() {
        let mut b = word;
        while b != 0 {
            let t = b.trailing_zeros() as usize;
            b &= b - 1;
            m[t][row / 64] |= 1 << (row % 64);
        }
    }
    ThoughtMasks { m }
}

fn transpose_cr(c: &ThoughtMasks) -> RowWords {
    let mut w = vec![0u64; ROWS];
    for (t, mask) in c.m.iter().enumerate() {
        for (wi, &word) in mask.iter().enumerate() {
            let mut b = word;
            while b != 0 {
                let r = b.trailing_zeros() as usize;
                b &= b - 1;
                w[wi * 64 + r] |= 1 << t;
            }
        }
    }
    RowWords { w }
}

/// Aperture: rows `lo..hi`, both multiples of 64 here (prefix ranges are).
#[derive(Clone, Copy)]
struct Aperture {
    lo: usize,
    hi: usize,
}

impl Aperture {
    fn words(self) -> (usize, usize) {
        (self.lo / 64, self.hi.div_ceil(64))
    }
    /// Mask of word `wi` restricted to the aperture.
    fn word_mask(self, wi: usize) -> u64 {
        let s = wi * 64;
        let lo = self.lo.max(s) - s;
        let hi = self.hi.min(s + 64) - s;
        if hi <= lo {
            return 0;
        }
        let top = if hi == 64 { u64::MAX } else { (1u64 << hi) - 1 };
        top & !((1u64 << lo) - 1)
    }
}

// ---------- R: per-row words ----------

fn r_id(r: &RowWords, t: u64, a: Aperture) -> u64 {
    r.w[a.lo..a.hi].iter().filter(|&&w| w & t != 0).count() as u64
}

fn r_th(r: &RowWords, t: u64, a: Aperture) -> u64 {
    let mut or = 0u64;
    for &w in &r.w[a.lo..a.hi] {
        or |= w;
    }
    (or & t).count_ones() as u64
}

fn r_pair(r: &RowWords, t: u64, a: Aperture) -> u64 {
    r.w[a.lo..a.hi]
        .iter()
        .map(|&w| (w & t).count_ones() as u64)
        .sum()
}

fn r_persp(r: &RowWords, t: u64, row: usize) -> u32 {
    (r.w[row] & t).count_ones()
}

// ---------- C: per-thought masks ----------

/// The thoughts of `t`, gathered once per query.
fn sel(t: u64) -> ([u8; THOUGHTS], usize) {
    let mut out = [0u8; THOUGHTS];
    let mut n = 0;
    let mut b = t;
    while b != 0 {
        out[n] = b.trailing_zeros() as u8;
        b &= b - 1;
        n += 1;
    }
    (out, n)
}

fn c_id(c: &ThoughtMasks, t: u64, a: Aperture) -> u64 {
    let (ts, n) = sel(t);
    let (w0, w1) = a.words();
    let mut total = 0u64;
    for wi in w0..w1 {
        let mut or = 0u64;
        for &th in &ts[..n] {
            or |= c.m[th as usize][wi];
        }
        total += (or & a.word_mask(wi)).count_ones() as u64;
    }
    total
}

fn c_th(c: &ThoughtMasks, t: u64, a: Aperture) -> u64 {
    let (ts, n) = sel(t);
    let (w0, w1) = a.words();
    let mut count = 0;
    for &th in &ts[..n] {
        let m = &c.m[th as usize];
        if (w0..w1).any(|wi| m[wi] & a.word_mask(wi) != 0) {
            count += 1;
        }
    }
    count
}

fn c_pair(c: &ThoughtMasks, t: u64, a: Aperture) -> u64 {
    let (ts, n) = sel(t);
    let (w0, w1) = a.words();
    let mut total = 0u64;
    for &th in &ts[..n] {
        let m = &c.m[th as usize];
        for wi in w0..w1 {
            total += (m[wi] & a.word_mask(wi)).count_ones() as u64;
        }
    }
    total
}

fn c_persp(c: &ThoughtMasks, t: u64, row: usize) -> u32 {
    let (wi, bit) = (row / 64, row % 64);
    let mut b = t;
    let mut n = 0;
    while b != 0 {
        let th = b.trailing_zeros() as usize;
        b &= b - 1;
        n += ((c.m[th][wi] >> bit) & 1) as u32;
    }
    n
}

// ---------- oracle ----------

fn covers(r: &RowWords, row: usize, th: usize) -> bool {
    (r.w[row] >> th) & 1 == 1
}

fn o_id(r: &RowWords, t: u64, a: Aperture) -> u64 {
    (a.lo..a.hi)
        .filter(|&row| (0..THOUGHTS).any(|th| (t >> th) & 1 == 1 && covers(r, row, th)))
        .count() as u64
}

fn o_th(r: &RowWords, t: u64, a: Aperture) -> u64 {
    (0..THOUGHTS)
        .filter(|&th| (t >> th) & 1 == 1 && (a.lo..a.hi).any(|row| covers(r, row, th)))
        .count() as u64
}

fn o_pair(r: &RowWords, t: u64, a: Aperture) -> u64 {
    let mut n = 0;
    for row in a.lo..a.hi {
        for th in 0..THOUGHTS {
            if (t >> th) & 1 == 1 && covers(r, row, th) {
                n += 1;
            }
        }
    }
    n
}

// ---------- timing ----------

/// Median of 5 runs, ns per call; each run repeats `f` until at least
/// 2 ms have passed.
fn ns_per_call<F: FnMut() -> u64>(mut f: F) -> f64 {
    let mut v: Vec<f64> = (0..5)
        .map(|_| {
            let t = Instant::now();
            let mut n = 0u64;
            let mut acc = 0u64;
            while n < 4 || t.elapsed().as_micros() < 2_000 {
                acc ^= f();
                n += 1;
            }
            black_box(acc);
            t.elapsed().as_nanos() as f64 / n as f64
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[2]
}

fn thought_sel(k: usize, seed: u64) -> u64 {
    if k >= THOUGHTS {
        return u64::MAX;
    }
    let mut t = 0u64;
    let mut i = 0;
    while (t.count_ones() as usize) < k {
        t |= 1 << (mix(seed ^ i) % THOUGHTS as u64);
        i += 1;
    }
    t
}

fn main() {
    println!("tier: {}", tier());
    println!(
        "d\t|T|\taperture\tquery\tR_ns\tC_ns\tR_ns_row\tC_ns_row\tanswer"
    );
    let probes: Vec<usize> = (0..4096).map(|i| (mix(i as u64) as usize) % ROWS).collect();
    for &d_ppm in &[10_000u64, 100_000, 500_000] {
        let r = fixture(d_ppm, 0x5EED);
        let c = transpose_rc(&r);
        let rc = ns_per_call(|| transpose_rc(&r).m[0][0]) / ROWS as f64;
        let cr = ns_per_call(|| transpose_cr(&c).w[0]) / ROWS as f64;
        println!(
            "# d = {}%: transpose R->C {:.2} ns/row, C->R {:.2} ns/row",
            d_ppm / 10_000,
            rc,
            cr
        );
        for &k in &[1usize, 8, 64] {
            let t = thought_sel(k, 0xC0DE);
            for &rows in &[ROWS, 4096, 256] {
                // A prefix range in the middle of the tile.
                let lo = (ROWS / 2 / rows) * rows;
                let a = Aperture { lo, hi: lo + rows };
                type Q = (&'static str, fn(&RowWords, u64, Aperture) -> u64,
                          fn(&ThoughtMasks, u64, Aperture) -> u64);
                let qs: [Q; 3] = [("ID", r_id, c_id), ("TH", r_th, c_th), ("PAIR", r_pair, c_pair)];
                for (name, rf, cf) in qs {
                    let ans = rf(&r, t, a);
                    assert_eq!(ans, cf(&c, t, a), "{name}");
                    let rn = ns_per_call(|| rf(black_box(&r), black_box(t), a));
                    let cn = ns_per_call(|| cf(black_box(&c), black_box(t), a));
                    println!(
                        "{}%\t{}\t{}\t{}\t{:.1}\t{:.1}\t{:.3}\t{:.3}\t{}",
                        d_ppm / 10_000,
                        k,
                        rows,
                        name,
                        rn,
                        cn,
                        rn / rows as f64,
                        cn / rows as f64,
                        ans
                    );
                }
            }
            let rn = ns_per_call(|| {
                probes.iter().map(|&p| r_persp(&r, black_box(t), p) as u64).sum()
            }) / probes.len() as f64;
            let cn = ns_per_call(|| {
                probes.iter().map(|&p| c_persp(&c, black_box(t), p) as u64).sum()
            }) / probes.len() as f64;
            println!(
                "{}%\t{}\t1\tPERSP\t{:.2}\t{:.2}\t-\t-\t-",
                d_ppm / 10_000,
                k,
                rn,
                cn
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apertures() -> Vec<Aperture> {
        vec![
            Aperture { lo: 0, hi: ROWS },
            Aperture { lo: 4096, hi: 8192 },
            Aperture { lo: 32_768, hi: 33_024 },
            Aperture { lo: 100, hi: 3_000 },
        ]
    }

    /// FAILS IF: any query in either orientation differs from the oracle.
    #[test]
    fn both_orientations_match_the_oracle() {
        for &d in &[10_000u64, 300_000] {
            let r = fixture(d, 7);
            let c = transpose_rc(&r);
            for &k in &[1usize, 8, 64] {
                let t = thought_sel(k, 3);
                for a in apertures() {
                    assert_eq!(r_id(&r, t, a), o_id(&r, t, a));
                    assert_eq!(c_id(&c, t, a), o_id(&r, t, a));
                    assert_eq!(r_th(&r, t, a), o_th(&r, t, a));
                    assert_eq!(c_th(&c, t, a), o_th(&r, t, a));
                    assert_eq!(r_pair(&r, t, a), o_pair(&r, t, a));
                    assert_eq!(c_pair(&c, t, a), o_pair(&r, t, a));
                }
                for row in [0, 63, 64, 4_097, ROWS - 1] {
                    let want = (0..THOUGHTS)
                        .filter(|&th| (t >> th) & 1 == 1 && covers(&r, row, th))
                        .count() as u32;
                    assert_eq!(r_persp(&r, t, row), want);
                    assert_eq!(c_persp(&c, t, row), want);
                }
            }
        }
    }

    /// FAILS IF: the transposes are not inverse to each other.
    #[test]
    fn transposes_round_trip() {
        let r = fixture(100_000, 11);
        assert_eq!(transpose_cr(&transpose_rc(&r)).w, r.w);
    }

    /// FAILS IF: the three equivalence keys stop giving three different
    /// numbers on one fixture (pair > id > thoughts), i.e. a query collapsed
    /// into another.
    #[test]
    fn the_keys_count_different_things() {
        let r = fixture(100_000, 5);
        let t = thought_sel(8, 9);
        let a = Aperture { lo: 0, hi: 4096 };
        let (id, th, pair) = (r_id(&r, t, a), r_th(&r, t, a), r_pair(&r, t, a));
        assert_eq!(th, 8);
        assert!(id > th && pair > id, "id {id} th {th} pair {pair}");
    }

    /// FAILS IF: the aperture does not bound the count (a narrower range
    /// must give a strictly smaller ID count on a 10 % fixture).
    #[test]
    fn the_aperture_bounds_the_count() {
        let r = fixture(100_000, 13);
        let c = transpose_rc(&r);
        let t = thought_sel(8, 2);
        let full = c_id(&c, t, Aperture { lo: 0, hi: ROWS });
        let part = c_id(&c, t, Aperture { lo: 128, hi: 4_000 });
        assert!(part > 0 && part * 4 < full, "part {part} full {full}");
        assert_eq!(part, o_id(&r, t, Aperture { lo: 128, hi: 4_000 }));
    }

    /// FAILS IF: the thought selection is ignored (a one-thought selection
    /// must count fewer pairs than all 64).
    #[test]
    fn the_selection_bounds_the_count() {
        let r = fixture(100_000, 17);
        let a = Aperture { lo: 0, hi: ROWS };
        assert!(r_pair(&r, thought_sel(1, 1), a) * 8 < r_pair(&r, u64::MAX, a));
    }
}
