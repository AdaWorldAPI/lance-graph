//! The exact per-row NAL truth laws over `u8` codes (W4 oracle).
//!
//! A code `x` stands for the value `x / 255`. Each law is computed in exact
//! integer arithmetic and rounded ONCE, half up, back into a code — the one
//! rounding `causal_edge::isa::truth` performs when it packs (`(v *
//! 255).round()`), without its intermediate `f32` rounding. This is the
//! contract a lane-producing realization must meet; it is test code, never a
//! production path (the production law runs in mask-risc).
//!
//! Formulas (horizon `k = 1`, `causal_edge::isa::truth` and the planner's
//! `TruthValue`):
//! - deduction: `f = f1·f2`, `c = f1·f2·c1·c2`;
//! - analogy: `f = f1·f2`, `c = c1·c2·f2`;
//! - induction: `f = f2`, `c = w / (w + 1)`, `w = f1·c1·c2`;
//! - abduction: `f = f1`, `c = w / (w + 1)`, `w = f2·c1·c2`.
//!
//! Which premise is "1" and which is "2" is the syllogism's business (W5);
//! these are two-input laws.

#![allow(dead_code)]

/// `round(p / q)`, half up, for `p ≥ 0`, `q > 0`.
pub fn round_div(p: u64, q: u64) -> u8 {
    let r = (2 * p + q) / (2 * q);
    u8::try_from(r).expect("a law value lies in [0, 1]")
}

const K1: u64 = 255;
const K2: u64 = 255 * 255;
const K3: u64 = 255 * 255 * 255;

fn w2c(w_num: u64) -> u8 {
    // w = w_num / 255³, c = w / (w + 1) = w_num / (w_num + 255³), code = 255·c.
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

/// `(f, c)` codes of analogy.
pub fn analogy(f1: u8, c1: u8, f2: u8, c2: u8) -> (u8, u8) {
    let ff = u64::from(f1) * u64::from(f2);
    let c = u64::from(c1) * u64::from(c2) * u64::from(f2);
    (round_div(ff, K1), round_div(c, K2))
}

/// `(f, c)` codes of induction.
pub fn induction(f1: u8, c1: u8, f2: u8, c2: u8) -> (u8, u8) {
    (f2, w2c(u64::from(f1) * u64::from(c1) * u64::from(c2)))
}

/// `(f, c)` codes of abduction.
pub fn abduction(f1: u8, c1: u8, f2: u8, c2: u8) -> (u8, u8) {
    (f1, w2c(u64::from(f2) * u64::from(c1) * u64::from(c2)))
}

/// A rule, for iteration.
pub type Law = fn(u8, u8, u8, u8) -> (u8, u8);

/// The four NAL-1 laws by name.
pub const LAWS: [(&str, Law); 4] = [
    ("deduction", deduction),
    ("analogy", analogy),
    ("induction", induction),
    ("abduction", abduction),
];
