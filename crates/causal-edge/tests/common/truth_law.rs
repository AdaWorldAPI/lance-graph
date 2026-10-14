//! The exact NAL truth laws over `u8` codes, written independently of the
//! crate (harvested from lance-graph #1544, D-NALP-4).
//!
//! A code `x` stands for `x / 255`. Each law is computed in exact integer
//! arithmetic and rounded once, half up, back into a code. Formulas (horizon
//! `k = 1`):
//! - deduction: `f = f1·f2`, `c = f1·f2·c1·c2`;
//! - induction: `f = f2`, `c = w / (w + 1)`, `w = f1·c1·c2`;
//! - abduction: `f = f1`, `c = w / (w + 1)`, `w = f2·c1·c2`.

#![allow(dead_code)]

/// `round(p / q)`, half up, for `p ≥ 0`, `q > 0`.
pub fn round_div(p: u64, q: u64) -> u8 {
    let r = (2 * p + q) / (2 * q);
    u8::try_from(r).expect("a law value lies in [0, 1]")
}

const K1: u64 = 255;
const K3: u64 = 255 * 255 * 255;

/// `w = w_num / 255³`, `c = w / (w + 1) = w_num / (w_num + 255³)`.
fn w2c(w_num: u64) -> u8 {
    round_div(255 * w_num, w_num + K3)
}

/// `(f, c)` codes of deduction.
pub fn deduction(f1: u8, c1: u8, f2: u8, c2: u8) -> (u8, u8) {
    let ff = u64::from(f1) * u64::from(f2);
    (
        round_div(ff, K1),
        round_div(ff * u64::from(c1) * u64::from(c2), K3),
    )
}

/// `(f, c)` codes of induction.
pub fn induction(f1: u8, c1: u8, f2: u8, c2: u8) -> (u8, u8) {
    (f2, w2c(u64::from(f1) * u64::from(c1) * u64::from(c2)))
}

/// `(f, c)` codes of abduction.
pub fn abduction(f1: u8, c1: u8, f2: u8, c2: u8) -> (u8, u8) {
    (f1, w2c(u64::from(f2) * u64::from(c1) * u64::from(c2)))
}

/// A two-premise law, for iteration.
pub type Law = fn(u8, u8, u8, u8) -> (u8, u8);
