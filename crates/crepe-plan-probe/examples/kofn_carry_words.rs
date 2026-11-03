//! D-DRAM-P3-5 (D-DRAM-3): k-of-n threshold over n masks, built from the two
//! shipped ndarray carry words only, against native per-lane counting.
//!
//! Lab only. No public type, no new opcode. The only ndarray words used are
//! `half_add_mask` and `full_add_mask`; `U24x5` / `Abacus` are lab fixtures
//! and are not imported.
//!
//! Question: does the bit-plane carry fold (the Abakus register model) answer
//! "at least k of these n masks are set, per lane" faster than native
//! per-lane counting, and does it agree exactly with an independent oracle?
//!
//! Arms, all producing one result mask word per 64 lanes:
//! - `lane`  native: for each of the 64 lanes, count the n bits, compare to k.
//! - `bytes` native: 64 `u8` counters per word, one add per mask, compare.
//! - `simd`  native on the facade: one `U8x64` of counters, per mask
//!   `saturating_add(mask_blend(m, 0, 1))`, then `cmpgt_mask(k - 1)`.
//! - `carry` the bit-plane counter: `full_add_mask` adds two masks into
//!   column 0, `half_add_mask` ripples the carry up; a constant-k comparator
//!   over the planes gives the result. The planes stay in registers.
//! - `or` / `and`: native Boolean folds, timed only for k = 1 and k = n,
//!   where k-of-n must equal them exactly (the silence twin).
//!
//! The oracle walks the set bits of every mask with `trailing_zeros` into a
//! per-lane `u32` count, so it shares no mechanism with any arm.
//!
//! Run: `cd crates/crepe-plan-probe && cargo run --release --example kofn_carry_words`

use ndarray::simd::{U8x64, full_add_mask, half_add_mask};
use std::hint::black_box;
use std::time::Instant;

const WORDS: usize = 1024; // 65,536 lanes per mask
const REPS: usize = 31;
const MAX_PLANES: usize = 8;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[derive(Clone, Copy)]
enum Workload {
    Dense,
    Random,
    SparseOneLane,
}

impl Workload {
    fn name(self) -> &'static str {
        match self {
            Workload::Dense => "dense p=7/8",
            Workload::Random => "random p=1/2",
            Workload::SparseOneLane => "sparse one lane",
        }
    }
    fn word(self, r: &mut Rng) -> u64 {
        match self {
            Workload::Dense => r.next() | r.next() | r.next(),
            Workload::Random => r.next(),
            Workload::SparseOneLane => 1u64 << (r.next() % 64),
        }
    }
}

/// `masks[i][w]`: input mask i, word w.
fn make(n: usize, wl: Workload, seed: u64) -> Vec<Vec<u64>> {
    let mut r = Rng(seed);
    (0..n)
        .map(|_| (0..WORDS).map(|_| wl.word(&mut r)).collect())
        .collect()
}

fn planes_for(n: usize) -> usize {
    (usize::BITS - n.leading_zeros()) as usize
}

// ── oracle ──────────────────────────────────────────────────────────────────

fn oracle(masks: &[Vec<u64>], k: usize) -> Vec<u64> {
    let mut counts = vec![0u32; WORDS * 64];
    for m in masks {
        for (w, &word) in m.iter().enumerate() {
            let mut x = word;
            while x != 0 {
                let j = x.trailing_zeros() as usize;
                counts[w * 64 + j] += 1;
                x &= x - 1;
            }
        }
    }
    let mut out = vec![0u64; WORDS];
    for (i, &c) in counts.iter().enumerate() {
        if c as usize >= k {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
    out
}

// ── native arms ─────────────────────────────────────────────────────────────

fn arm_lane(masks: &[Vec<u64>], k: usize, out: &mut [u64]) {
    for (w, o) in out.iter_mut().enumerate() {
        let mut r = 0u64;
        for j in 0..64 {
            let mut c = 0usize;
            for m in masks {
                c += ((m[w] >> j) & 1) as usize;
            }
            r |= ((c >= k) as u64) << j;
        }
        *o = r;
    }
}

fn arm_bytes(masks: &[Vec<u64>], k: usize, out: &mut [u64]) {
    let k = k as u8;
    for (w, o) in out.iter_mut().enumerate() {
        let mut cnt = [0u8; 64];
        for m in masks {
            let x = m[w];
            for (j, c) in cnt.iter_mut().enumerate() {
                *c += ((x >> j) & 1) as u8;
            }
        }
        let mut r = 0u64;
        for (j, &c) in cnt.iter().enumerate() {
            r |= ((c >= k) as u64) << j;
        }
        *o = r;
    }
}

fn arm_simd(masks: &[Vec<u64>], k: usize, out: &mut [u64]) {
    let (zero, one) = (U8x64::splat(0), U8x64::splat(1));
    let km1 = U8x64::splat((k - 1) as u8);
    for (w, o) in out.iter_mut().enumerate() {
        let mut cnt = zero;
        for m in masks {
            cnt = cnt.saturating_add(U8x64::mask_blend(m[w], zero, one));
        }
        *o = cnt.cmpgt_mask(km1);
    }
}

fn arm_or(masks: &[Vec<u64>], out: &mut [u64]) {
    for (w, o) in out.iter_mut().enumerate() {
        *o = masks.iter().fold(0, |a, m| a | m[w]);
    }
}

fn arm_and(masks: &[Vec<u64>], out: &mut [u64]) {
    for (w, o) in out.iter_mut().enumerate() {
        *o = masks.iter().fold(!0, |a, m| a & m[w]);
    }
}

// ── the carry-word arm ──────────────────────────────────────────────────────

/// Adds `carry` (weight 2^from) into the planes, rippling up.
#[inline(always)]
fn ripple(p: &mut [u64; MAX_PLANES], b: usize, from: usize, mut carry: u64) {
    let mut j = from;
    while carry != 0 && j < b {
        let (s, c) = half_add_mask(p[j], carry);
        p[j] = s;
        carry = c;
        j += 1;
    }
}

/// Per lane: plane value >= k, for a constant k, MSB first.
#[inline(always)]
fn ge_const(p: &[u64; MAX_PLANES], b: usize, k: usize) -> u64 {
    let mut eq = !0u64;
    let mut gt = 0u64;
    for j in (0..b).rev() {
        if (k >> j) & 1 == 1 {
            eq &= p[j];
        } else {
            gt |= eq & p[j];
        }
    }
    gt | eq
}

fn arm_carry(masks: &[Vec<u64>], k: usize, out: &mut [u64]) {
    let n = masks.len();
    let b = planes_for(n);
    for (w, o) in out.iter_mut().enumerate() {
        let mut p = [0u64; MAX_PLANES];
        let mut i = 0;
        while i + 1 < n {
            let (s, c) = full_add_mask(p[0], masks[i][w], masks[i + 1][w]);
            p[0] = s;
            ripple(&mut p, b, 1, c);
            i += 2;
        }
        if i < n {
            ripple(&mut p, b, 0, masks[i][w]);
        }
        *o = ge_const(&p, b, k);
    }
}

// ── timing ──────────────────────────────────────────────────────────────────

fn time(mut f: impl FnMut(&mut [u64])) -> f64 {
    let mut out = vec![0u64; WORDS];
    let mut v = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        f(black_box(&mut out));
        black_box(&out);
        v.push(t.elapsed().as_nanos() as f64 / WORDS as f64);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[REPS / 2]
}

fn main() {
    let mut failures = 0usize;
    println!(
        "k-of-n over n masks, {WORDS} words ({} lanes), median of {REPS}, ns per 64-lane word",
        WORDS * 64
    );
    println!(
        "{:<16} {:>3} {:>3} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10}",
        "workload", "n", "k", "lane", "bytes", "simd", "carry", "or/and", "simd/carry"
    );
    for wl in [Workload::Dense, Workload::Random, Workload::SparseOneLane] {
        for n in [3usize, 8, 32] {
            let masks = make(n, wl, 0x5EED ^ n as u64);
            for k in [1usize, n.div_ceil(2), n] {
                let want = oracle(&masks, k);
                let mut got = vec![0u64; WORDS];
                for (name, f) in [
                    ("lane", arm_lane as fn(&[Vec<u64>], usize, &mut [u64])),
                    ("bytes", arm_bytes),
                    ("simd", arm_simd),
                    ("carry", arm_carry),
                ] {
                    f(&masks, k, &mut got);
                    if got != want {
                        failures += 1;
                        println!(
                            "FAIL {name} {} n={n} k={k}: disagrees with oracle",
                            wl.name()
                        );
                    }
                }
                // Silence twin: k = 1 is OR, k = n is AND, exactly.
                let bool_ns = if k == 1 || k == n {
                    let mut b = vec![0u64; WORDS];
                    if k == 1 {
                        arm_or(&masks, &mut b);
                    } else {
                        arm_and(&masks, &mut b);
                    }
                    arm_carry(&masks, k, &mut got);
                    if got != b {
                        failures += 1;
                        println!(
                            "FAIL twin {} n={n} k={k}: carry != {}",
                            wl.name(),
                            if k == 1 { "OR" } else { "AND" }
                        );
                    }
                    Some(time(|o| {
                        if k == 1 {
                            arm_or(&masks, o)
                        } else {
                            arm_and(&masks, o)
                        }
                    }))
                } else {
                    None
                };
                // Anti-vacuity: a middle k on random masks must split the lanes.
                if matches!(wl, Workload::Random) && k == n.div_ceil(2) {
                    let ones: u32 = want.iter().map(|w| w.count_ones()).sum();
                    let frac = ones as f64 / (WORDS * 64) as f64;
                    if !(0.05..=0.95).contains(&frac) {
                        failures += 1;
                        println!("FAIL vacuity random n={n} k={k}: result density {frac:.3}");
                    }
                }
                let t_lane = time(|o| arm_lane(&masks, k, o));
                let t_bytes = time(|o| arm_bytes(&masks, k, o));
                let t_simd = time(|o| arm_simd(&masks, k, o));
                let t_carry = time(|o| arm_carry(&masks, k, o));
                let b = bool_ns.map_or("-".to_string(), |x| format!("{x:.2}"));
                println!(
                    "{:<16} {n:>3} {k:>3} {t_lane:>9.2} {t_bytes:>9.2} {t_simd:>9.2} {t_carry:>9.2} {b:>9} {:>9.2}x",
                    wl.name(),
                    t_simd / t_carry
                );
            }
        }
    }
    if failures == 0 {
        println!("all arms agree with the oracle; twins k=1 == OR and k=n == AND hold");
    } else {
        println!("{failures} FAILURES");
        std::process::exit(1);
    }
}
