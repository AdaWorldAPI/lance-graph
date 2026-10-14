//! Exhaustive check of the exact truth codes, over every input
//! (`(f1, c1, f2, c2)` ∈ 256⁴ = 2³² rows per law).
//!
//! For deduction, induction and abduction it counts the rows where
//! - `truth::code` (what the register paths pack) differs from the
//!   independent oracle in `tests/common/truth_law.rs` (must be 0), and
//! - the `f32` functions, packed with `(v · 255).round()`, differ from it
//!   (the error the exact codes remove).
//!
//! Run: `cargo run --release -p causal-edge --example truth_exact_probe`.

#[path = "../tests/common/truth_law.rs"]
mod truth_law;

use causal_edge::isa::{truth, Opcode};
use std::thread;

type Law = truth_law::Law;

const LAWS: [(&str, Opcode, Law, Law); 3] = [
    (
        "deduction",
        Opcode::Deduction,
        truth::code::deduction,
        truth_law::deduction,
    ),
    (
        "induction",
        Opcode::Induction,
        truth::code::induction,
        truth_law::induction,
    ),
    (
        "abduction",
        Opcode::Abduction,
        truth::code::abduction,
        truth_law::abduction,
    ),
];

fn f32_path(op: Opcode, f1: u8, c1: u8, f2: u8, c2: u8) -> (u8, u8) {
    let d = |x: u8| f32::from(x) / 255.0;
    let (f, c) = op.truth(d(f1), d(c1), d(f2), d(c2));
    (truth::pack(f), truth::pack(c))
}

fn main() {
    println!(
        "tier: avx512f={} avx2={} (scalar probe; exactness, not speed)",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    println!("rows per law: 2^32 = {}", 1u64 << 32);
    println!("{:<10} {:>14} {:>14}", "law", "code != law", "f32 != law");
    for (name, op, code, oracle) in LAWS {
        let threads = 8u16;
        let (code_off, f32_off) = thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    s.spawn(move || {
                        let (mut a, mut b) = (0u64, 0u64);
                        let mut f1 = t;
                        while f1 < 256 {
                            let f1b = f1 as u8;
                            for c1 in 0..=255u8 {
                                for f2 in 0..=255u8 {
                                    for c2 in 0..=255u8 {
                                        let want = oracle(f1b, c1, f2, c2);
                                        a += u64::from(code(f1b, c1, f2, c2) != want);
                                        b += u64::from(f32_path(op, f1b, c1, f2, c2) != want);
                                    }
                                }
                            }
                            f1 += threads;
                        }
                        (a, b)
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().expect("worker"))
                .fold((0, 0), |x, y| (x.0 + y.0, x.1 + y.1))
        });
        println!("{name:<10} {code_off:>14} {f32_off:>14}");
    }
}
