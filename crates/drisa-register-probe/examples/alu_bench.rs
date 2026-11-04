//! D-DRISA-3 bench: the ALU instructions over 65,536 + 37 DR-aligned rows,
//! carry words against native baselines on the same facade. Median of 31
//! runs, ns per 64-row word. Answers are checked equal before timing.
//!
//! Run: cargo run --release --manifest-path crates/drisa-register-probe/Cargo.toml --example alu_bench

use std::hint::black_box;
use std::time::Instant;

use drisa_register_probe::alu::{bag_at_least, kofn, kofn_bytes, kofn_kleene, three_of_five_sop};
use ndarray::simd::U8x64;

const ROWS: usize = 65_536 + 37;

fn rng(mut z: u64) -> impl FnMut() -> u64 {
    move || {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }
}

fn masks(n: usize, seed: u64) -> Vec<Vec<u64>> {
    let mut r = rng(seed);
    let words = ROWS.div_ceil(64);
    (0..n).map(|_| (0..words).map(|_| r()).collect()).collect()
}

fn time<T>(mut f: impl FnMut() -> T) -> f64 {
    let mut v: Vec<f64> = (0..31)
        .map(|_| {
            let t = Instant::now();
            black_box(f());
            t.elapsed().as_nanos() as f64 / ROWS.div_ceil(64) as f64
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[15]
}

/// Kleene k-of-n with two byte counters (TRUE, and TRUE-or-UNKNOWN).
fn kleene_bytes(t: &[&[u64]], u: &[&[u64]], k: usize, presence: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let (zero, one) = (U8x64::splat(0), U8x64::splat(1));
    let km1 = U8x64::splat((k - 1) as u8);
    let words = ROWS.div_ceil(64);
    let mut yes = Vec::with_capacity(words);
    let mut unk = Vec::with_capacity(words);
    for w in 0..words {
        let (mut ct, mut cp) = (zero, zero);
        for (ti, ui) in t.iter().zip(u) {
            ct = ct.saturating_add(U8x64::mask_blend(ti[w], zero, one));
            cp = cp.saturating_add(U8x64::mask_blend(ti[w] | ui[w], zero, one));
        }
        let live = presence[w]
            & if w == words - 1 {
                (1u64 << (ROWS % 64)) - 1
            } else {
                !0
            };
        let sure = ct.cmpgt_mask(km1) & live;
        yes.push(sure);
        unk.push(cp.cmpgt_mask(km1) & live & !sure);
    }
    (yes, unk)
}

/// Bag sum with one byte counter: multiplicity lo + 2·hi.
fn bag_bytes(lo: &[&[u64]], hi: &[&[u64]], k: usize, presence: &[u64]) -> Vec<u64> {
    let (zero, one, two) = (U8x64::splat(0), U8x64::splat(1), U8x64::splat(2));
    let km1 = U8x64::splat((k - 1) as u8);
    let words = ROWS.div_ceil(64);
    (0..words)
        .map(|w| {
            let mut c = zero;
            for (l, h) in lo.iter().zip(hi) {
                c = c.saturating_add(U8x64::mask_blend(l[w], zero, one));
                c = c.saturating_add(U8x64::mask_blend(h[w], zero, two));
            }
            let live = presence[w]
                & if w == words - 1 {
                    (1u64 << (ROWS % 64)) - 1
                } else {
                    !0
                };
            c.cmpgt_mask(km1) & live
        })
        .collect()
}

fn main() {
    let words = ROWS.div_ceil(64);
    let mut presence = vec![!0u64; words];
    presence[words - 1] = (1u64 << (ROWS % 64)) - 1;
    println!("D-DRISA-3: {ROWS} rows, ns per 64-row word, median of 31");
    println!(
        "{:<34} {:>10} {:>10} {:>10}",
        "instruction", "carry", "native", "carry/native"
    );

    // A7 negative control: a 2-input AND.
    let m2 = masks(2, 1);
    let r2: Vec<&[u64]> = m2.iter().map(|v| v.as_slice()).collect();
    let and = |p: &[u64]| -> Vec<u64> { (0..words).map(|w| r2[0][w] & r2[1][w] & p[w]).collect() };
    assert_eq!(kofn(&r2, 2, &presence, ROWS), and(&presence));
    let (c, n) = (
        time(|| kofn(&r2, 2, &presence, ROWS)),
        time(|| and(&presence)),
    );
    println!(
        "{:<34} {c:>10.2} {n:>10.2} {:>9.1}x",
        "2-of-2 vs native AND",
        c / n
    );

    // 3-of-5 vs its 10-term Boolean expansion and vs byte counters.
    let m5 = masks(5, 2);
    let r5: Vec<&[u64]> = m5.iter().map(|v| v.as_slice()).collect();
    let a5: [&[u64]; 5] = [r5[0], r5[1], r5[2], r5[3], r5[4]];
    assert_eq!(
        kofn(&r5, 3, &presence, ROWS),
        three_of_five_sop(&a5, &presence, ROWS)
    );
    assert_eq!(
        kofn(&r5, 3, &presence, ROWS),
        kofn_bytes(&r5, 3, &presence, ROWS)
    );
    let c = time(|| kofn(&r5, 3, &presence, ROWS));
    let s = time(|| three_of_five_sop(&a5, &presence, ROWS));
    let b = time(|| kofn_bytes(&r5, 3, &presence, ROWS));
    println!(
        "{:<34} {c:>10.2} {s:>10.2} {:>9.1}x",
        "3-of-5 vs 10-term SOP",
        c / s
    );
    println!(
        "{:<34} {c:>10.2} {b:>10.2} {:>9.1}x",
        "3-of-5 vs U8x64 counters",
        c / b
    );

    // k-of-n at n = 16 and 24.
    for n_in in [16usize, 24] {
        let m = masks(n_in, 3 + n_in as u64);
        let r: Vec<&[u64]> = m.iter().map(|v| v.as_slice()).collect();
        let k = n_in / 2;
        assert_eq!(
            kofn(&r, k, &presence, ROWS),
            kofn_bytes(&r, k, &presence, ROWS)
        );
        let c = time(|| kofn(&r, k, &presence, ROWS));
        let b = time(|| kofn_bytes(&r, k, &presence, ROWS));
        println!(
            "{:<34} {c:>10.2} {b:>10.2} {:>9.1}x",
            format!("{k}-of-{n_in} vs U8x64 counters"),
            c / b
        );
    }

    // Kleene k-of-n, two counters each side.
    for n_in in [8usize, 24] {
        let t = masks(n_in, 10 + n_in as u64);
        let mut u = masks(n_in, 20 + n_in as u64);
        for (ui, ti) in u.iter_mut().zip(&t) {
            for (x, y) in ui.iter_mut().zip(ti) {
                *x &= !*y; // UNKNOWN disjoint from TRUE
            }
        }
        let tr: Vec<&[u64]> = t.iter().map(|v| v.as_slice()).collect();
        let ur: Vec<&[u64]> = u.iter().map(|v| v.as_slice()).collect();
        let k = n_in / 2;
        assert_eq!(
            kofn_kleene(&tr, &ur, k, &presence, ROWS),
            kleene_bytes(&tr, &ur, k, &presence)
        );
        let c = time(|| kofn_kleene(&tr, &ur, k, &presence, ROWS));
        let b = time(|| kleene_bytes(&tr, &ur, k, &presence));
        println!(
            "{:<34} {c:>10.2} {b:>10.2} {:>9.1}x",
            format!("Kleene {k}-of-{n_in} vs U8x64"),
            c / b
        );
    }

    // Bag sum of 5 two-bit multiplicities.
    let lo = masks(5, 40);
    let hi = masks(5, 41);
    let lr: Vec<&[u64]> = lo.iter().map(|v| v.as_slice()).collect();
    let hr: Vec<&[u64]> = hi.iter().map(|v| v.as_slice()).collect();
    assert_eq!(
        bag_at_least(&lr, &hr, 8, &presence, ROWS),
        bag_bytes(&lr, &hr, 8, &presence)
    );
    let c = time(|| bag_at_least(&lr, &hr, 8, &presence, ROWS));
    let b = time(|| bag_bytes(&lr, &hr, 8, &presence));
    println!(
        "{:<34} {c:>10.2} {b:>10.2} {:>9.1}x",
        "Bag sum>=8 of 5x2-bit vs U8x64",
        c / b
    );
}
