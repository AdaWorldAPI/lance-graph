//! D-DRISA-3 (EXO-153): arithmetic over DR-aligned mask planes with the two
//! shipped ndarray carry words only (`half_add_mask`, `full_add_mask`). LAB
//! ONLY: no public type, no new word, no `U24x5` / `Abacus` import.
//!
//! Each `u64` word is one significance column for 64 independent rows. A
//! counter of `b` planes holds a per-row count in `0..2^b`. Three
//! instruction shapes are tested against a per-row scalar oracle:
//! - [`kofn`]: at least `k` of `n` Boolean inputs (two-valued);
//! - [`kofn_kleene`]: the same under Kleene logic, where an input may be
//!   UNKNOWN for a row: TRUE iff the TRUE count reaches `k`, FALSE iff even
//!   counting every UNKNOWN as true does not, UNKNOWN otherwise;
//! - [`bag_at_least`]: Bag multiplicity, each input contributing a 2-bit
//!   multiplicity (0..=3) per row; the per-row SUM is compared, which is not
//!   the support count.
//!
//! Every result is ANDed with the population's presence plane: an absent row
//! is never TRUE (nor UNKNOWN), even for `k = 0`, and tail bits past `n_rows`
//! are zero.
//!
//! Baselines for the same instructions: [`kofn_bytes`] (one `U8x64` byte
//! counter per 64 rows on the same facade) and [`three_of_five_sop`] (the
//! sum-of-products Boolean expansion over native mask words).

use ndarray::simd::{full_add_mask, half_add_mask, U8x64};

/// Widest counter: counts up to 255.
pub const MAX_PLANES: usize = 8;

fn planes_for(max_count: usize) -> usize {
    (usize::BITS - max_count.leading_zeros()) as usize
}

fn tail(n_rows: usize, w: usize) -> u64 {
    let rows = n_rows.saturating_sub(w * 64);
    if rows >= 64 {
        !0
    } else {
        (1u64 << rows) - 1
    }
}

/// Add `x` (weight `2^from`) into a counter of `b` planes.
#[inline(always)]
fn ripple(p: &mut [u64; MAX_PLANES], b: usize, from: usize, mut x: u64) {
    let mut j = from;
    while x != 0 && j < b {
        let (s, c) = half_add_mask(p[j], x);
        p[j] = s;
        x = c;
        j += 1;
    }
}

/// Per row: counter >= `k`. `k` larger than the counter can hold is false.
#[inline(always)]
fn ge_const(p: &[u64; MAX_PLANES], b: usize, k: usize) -> u64 {
    if k >= 1 << b {
        return 0;
    }
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

/// Sum of `n` single-bit inputs at word `w`, into `b` planes.
#[inline(always)]
fn count_word(inputs: &[&[u64]], w: usize, b: usize) -> [u64; MAX_PLANES] {
    let mut p = [0u64; MAX_PLANES];
    let mut i = 0;
    while i + 1 < inputs.len() {
        let (s, c) = full_add_mask(p[0], inputs[i][w], inputs[i + 1][w]);
        p[0] = s;
        ripple(&mut p, b, 1, c);
        i += 2;
    }
    if i < inputs.len() {
        ripple(&mut p, b, 0, inputs[i][w]);
    }
    p
}

/// At least `k` of the `inputs` hold, per present row.
pub fn kofn(inputs: &[&[u64]], k: usize, presence: &[u64], n_rows: usize) -> Vec<u64> {
    let b = planes_for(inputs.len()).max(1);
    assert!(b <= MAX_PLANES, "n too large for an 8-plane counter");
    (0..n_rows.div_ceil(64))
        .map(|w| {
            let p = count_word(inputs, w, b);
            ge_const(&p, b, k) & presence[w] & tail(n_rows, w)
        })
        .collect()
}

/// Kleene k-of-n. Input `i` is TRUE where `t[i]` is set, UNKNOWN where `u[i]`
/// is set (the two must be disjoint), FALSE elsewhere. Returns `(true,
/// unknown)` result masks; a present row in neither is FALSE.
pub fn kofn_kleene(
    t: &[&[u64]],
    u: &[&[u64]],
    k: usize,
    presence: &[u64],
    n_rows: usize,
) -> (Vec<u64>, Vec<u64>) {
    assert_eq!(t.len(), u.len());
    let b = planes_for(t.len()).max(1);
    let words = n_rows.div_ceil(64);
    let mut yes = Vec::with_capacity(words);
    let mut unk = Vec::with_capacity(words);
    for w in 0..words {
        let pt = count_word(t, w, b);
        // "possible": TRUE or UNKNOWN, counted once each.
        let mut pp = [0u64; MAX_PLANES];
        for (ti, ui) in t.iter().zip(u) {
            ripple(&mut pp, b, 0, ti[w] | ui[w]);
        }
        let live = presence[w] & tail(n_rows, w);
        let sure = ge_const(&pt, b, k) & live;
        let could = ge_const(&pp, b, k) & live;
        yes.push(sure);
        unk.push(could & !sure);
    }
    (yes, unk)
}

/// Bag: input `i` contributes multiplicity `lo[i] + 2·hi[i]` (0..=3) per row.
/// True where the per-row SUM reaches `k`.
pub fn bag_at_least(
    lo: &[&[u64]],
    hi: &[&[u64]],
    k: usize,
    presence: &[u64],
    n_rows: usize,
) -> Vec<u64> {
    assert_eq!(lo.len(), hi.len());
    let b = planes_for(3 * lo.len()).max(1);
    assert!(b <= MAX_PLANES, "sum too large for an 8-plane counter");
    (0..n_rows.div_ceil(64))
        .map(|w| {
            let mut p = [0u64; MAX_PLANES];
            for (l, h) in lo.iter().zip(hi) {
                ripple(&mut p, b, 0, l[w]);
                ripple(&mut p, b, 1, h[w]);
            }
            ge_const(&p, b, k) & presence[w] & tail(n_rows, w)
        })
        .collect()
}

/// Baseline: the same k-of-n with one `U8x64` byte counter per 64 rows on the
/// ndarray facade. `k = 0` is all present rows; `n < 255`.
pub fn kofn_bytes(inputs: &[&[u64]], k: usize, presence: &[u64], n_rows: usize) -> Vec<u64> {
    let (zero, one) = (U8x64::splat(0), U8x64::splat(1));
    (0..n_rows.div_ceil(64))
        .map(|w| {
            let live = presence[w] & tail(n_rows, w);
            if k == 0 {
                return live;
            }
            if k > inputs.len() {
                return 0;
            }
            let mut c = zero;
            for m in inputs {
                c = c.saturating_add(U8x64::mask_blend(m[w], zero, one));
            }
            c.cmpgt_mask(U8x64::splat((k - 1) as u8)) & live
        })
        .collect()
}

/// Baseline: at least 3 of 5 as the 10-term sum of products over native mask
/// words.
pub fn three_of_five_sop(m: &[&[u64]; 5], presence: &[u64], n_rows: usize) -> Vec<u64> {
    (0..n_rows.div_ceil(64))
        .map(|w| {
            let [a, b, c, d, e] = [m[0][w], m[1][w], m[2][w], m[3][w], m[4][w]];
            let r = (a & b & c)
                | (a & b & d)
                | (a & b & e)
                | (a & c & d)
                | (a & c & e)
                | (a & d & e)
                | (b & c & d)
                | (b & c & e)
                | (b & d & e)
                | (c & d & e);
            r & presence[w] & tail(n_rows, w)
        })
        .collect()
}
