//! **W0B — the mask-ABI differential at the R2IL seam.**
//!
//! `ogar-r2il`'s [`CallMask`] carries its own Boolean algebra — `and` / `or`
//! / `xor` / `and_not` / `not` / `count` — over inline `u64` words.
//! `lance-graph-mask-risc` carries the same algebra as the ONE evaluator
//! above `ndarray::simd`. `lance-graph-quack`'s own manifest states the rule
//! this probe exists to test against: *"the masking algebra is reached
//! THROUGH mask-risc, never beside it."*
//!
//! So this asks exactly one question: **do the two agree bit-identically
//! over the same borrowed words?** `CallMask` is the oracle here and stays
//! the oracle — nothing is deleted, nothing delegates, no dependency
//! direction is decided. That decision needs the differential green first.
//!
//! # What this does NOT prove
//!
//! It is not a Quack↔R2IL bridge, and the two sides are not two views of one
//! population:
//!
//! ```text
//! CallMask     population = call slots inside ONE body   N <= 180
//! mask-risc    population = rows of one projection       N ~= 64K
//! ```
//!
//! Same algebra, different index spaces. `n_rows` below is a CallMask's
//! `len()` precisely so the comparison is apples-to-apples on the narrow
//! side; feeding a body's call mask to a row-population consumer would be a
//! category error, not an optimisation. Deriving a genuine row predicate
//! from a body is W0C.
//!
//! # Kill condition
//!
//! A disagreement is the FINDING, not a bug to align away. Do not "fix"
//! either side until it is settled which semantics is correct — the
//! difference is the information.

use lance_graph_mask_risc::{
    execute, words_for, MaskOp, Operand, Planes, Program, Scratch, Terminal, Value,
};
use ogar_loco::LaneShape;
use ogar_r2il::CallMask;

/// Every shape, with the call population `ogar-loco` derives for it
/// (`CONTENT_SLOTS × calls_per_lane` = 30 × 6 / 4 / 3).
const SHAPES: [(LaneShape, u32, usize); 3] = [
    (LaneShape::Pairs, 180, 3),
    (LaneShape::Triples, 120, 2),
    (LaneShape::Quads, 90, 2),
];

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

/// A seeded mask plus every word/population boundary the shape can express.
///
/// The explicit edges are the point: `63/64` and `127/128` are the word
/// seams, and `len-1` is the straddling word's last real bit. A purely
/// random fixture hits them only by luck, and the tail is exactly where two
/// complement implementations diverge.
fn seeded(shape: LaneShape, seed: u64, density: u64) -> CallMask {
    let mut m = CallMask::empty(shape);
    let mut s = seed;
    for i in 0..m.len() {
        if lcg(&mut s) % 100 < density {
            m.set(i);
        }
    }
    for edge in [0u32, 1, 62, 63, 64, 65, 126, 127, 128, 129] {
        if edge < m.len() && edge % 3 != 0 {
            m.set(edge);
        }
    }
    if m.len() >= 2 {
        m.set(m.len() - 2);
        m.set(m.len() - 1);
    }
    m
}

/// Run a one-op program over borrowed CallMask words and read the result back.
fn run_mask_op(op_of: impl Fn(Operand, Operand) -> MaskOp, a: &CallMask, b: &CallMask) -> Vec<u64> {
    let n_rows = a.len() as usize;
    let program = Program::new(
        vec![op_of(Operand::Plane(0), Operand::Plane(1))],
        Terminal::Keep {
            mask: Operand::Scratch(0),
        },
    );
    let planes: [&[u64]; 2] = [a.words(), b.words()];
    let planes = Planes {
        n_rows,
        masks: &planes,
        lanes: &[],
    };
    let mut scratch = Scratch::for_program(&program, n_rows).expect("scratch");
    let v = execute(&program, &planes, &mut scratch, None).expect("execute");
    assert_eq!(
        v,
        Value::Mask(Operand::Scratch(0)),
        "Keep must report the slot it kept"
    );
    scratch.slot(0).expect("slot 0").to_vec()
}

/// The word count the two sides must independently agree on.
///
/// Not a formality: `CallMask::words()` slices to `len.div_ceil(64)` and
/// mask-risc sizes a scratch slot with `words_for(n_rows)`. If those ever
/// disagreed, every comparison below would be between slices of different
/// length and the `assert_eq!` would report a shape mismatch rather than a
/// semantic one.
#[test]
fn both_sides_span_the_same_words() {
    for (shape, want_len, want_words) in SHAPES {
        let m = CallMask::all(shape);
        assert_eq!(m.len(), want_len, "{shape:?}: population");
        assert_eq!(
            m.words().len(),
            want_words,
            "{shape:?}: CallMask slice width"
        );
        assert_eq!(
            words_for(m.len() as usize),
            want_words,
            "{shape:?}: mask-risc words_for disagrees with the CallMask slice"
        );
    }
}

#[test]
fn and_or_xor_andnot_agree_bit_for_bit() {
    for (shape, _, _) in SHAPES {
        for (sa, sb) in [(1u64, 2u64), (0xDEAD, 0xBEEF), (7, 7)] {
            for (da, db) in [(50u64, 50u64), (3, 97), (100, 0), (0, 0), (100, 100)] {
                let a = seeded(shape, sa, da);
                let b = seeded(shape, sb, db);

                for (name, expect, op) in [
                    (
                        "and",
                        a.and(&b),
                        (|x, y| MaskOp::And { a: x, b: y, dst: 0 })
                            as fn(Operand, Operand) -> MaskOp,
                    ),
                    ("or", a.or(&b), |x, y| MaskOp::Or { a: x, b: y, dst: 0 }),
                    ("xor", a.xor(&b), |x, y| MaskOp::Xor { a: x, b: y, dst: 0 }),
                    ("and_not", a.and_not(&b), |x, y| MaskOp::AndNot {
                        a: x,
                        b: y,
                        dst: 0,
                    }),
                ] {
                    let got = run_mask_op(op, &a, &b);
                    assert_eq!(
                        got,
                        expect.words(),
                        "{shape:?} {name}: seeds ({sa},{sb}) density ({da},{db}) \
                         -- CallMask and mask-risc disagree. THIS IS THE FINDING: \
                         settle which semantics is correct before aligning either side."
                    );
                }
            }
        }
    }
}

/// `not` is the one with a tail obligation on both sides, so it gets its own
/// test: `CallMask::not` clears per-word against `len`, mask-risc's `Not`
/// documents "(tail cleared)" against `n_rows`. They agree only if both
/// clear against the same population — which is what this measures.
#[test]
fn not_agrees_including_the_tail() {
    for (shape, _, _) in SHAPES {
        for (seed, density) in [(1u64, 0u64), (2, 50), (3, 100), (4, 1)] {
            let a = seeded(shape, seed, density);
            let n_rows = a.len() as usize;
            let program = Program::new(
                vec![MaskOp::Not {
                    a: Operand::Plane(0),
                    dst: 0,
                }],
                Terminal::Keep {
                    mask: Operand::Scratch(0),
                },
            );
            let planes: [&[u64]; 1] = [a.words()];
            let planes = Planes {
                n_rows,
                masks: &planes,
                lanes: &[],
            };
            let mut scratch = Scratch::for_program(&program, n_rows).expect("scratch");
            execute(&program, &planes, &mut scratch, None).expect("execute");
            assert_eq!(
                scratch.slot(0).expect("slot 0"),
                a.not().words(),
                "{shape:?} not: seed {seed} density {density} -- tail handling differs"
            );
        }
    }
}

#[test]
fn count_agrees_with_the_count_terminal() {
    for (shape, _, _) in SHAPES {
        for (seed, density) in [(1u64, 0u64), (2, 13), (3, 50), (4, 99), (5, 100)] {
            let a = seeded(shape, seed, density);
            let n_rows = a.len() as usize;
            let program = Program::new(
                vec![],
                Terminal::Count {
                    mask: Operand::Plane(0),
                },
            );
            let planes: [&[u64]; 1] = [a.words()];
            let planes = Planes {
                n_rows,
                masks: &planes,
                lanes: &[],
            };
            let mut scratch = Scratch::for_program(&program, n_rows).expect("scratch");
            let v = execute(&program, &planes, &mut scratch, None).expect("execute");
            assert_eq!(
                v,
                Value::Count(a.count() as usize),
                "{shape:?} count: seed {seed} density {density}"
            );
        }
    }
}

/// The four binary ops as TERNLOG immediates.
///
/// `CallMask` has no three-input op, so this is the direction the agreement
/// actually matters in: if mask-risc's arbitrary-immediate form reproduces
/// all four of CallMask's binary ops, then CallMask's algebra is a SUBSET of
/// mask-risc's, not a sibling of it — which is the evidence the ownership
/// decision (deferred out of this probe) will need.
///
/// The immediates are DERIVED from each op's own truth table, never written
/// by hand. Hand-writing them is how the first draft of this test failed:
/// `and` was given `0b1010_0000` (bits 7 and 5) where only bit 7 is the
/// conjunction, and the failure looked like a substrate disagreement rather
/// than an arithmetic slip in the fixture.
fn ternlog_imm_for(f: impl Fn(bool, bool) -> bool) -> u8 {
    let mut imm = 0u8;
    for idx in 0u8..8 {
        // VPTERNLOG index convention: (a << 2) | (b << 1) | c.
        let a = idx & 0b100 != 0;
        let b = idx & 0b010 != 0;
        let c = idx & 0b001 != 0;
        // `c` is bound to `a`'s plane below, so triples with `a != c` are
        // unreachable and their bits are left zero — one canonical immediate
        // per function rather than the four that would also work.
        if a == c && f(a, b) {
            imm |= 1 << idx;
        }
    }
    imm
}

#[test]
fn ternlog_reproduces_callmask_s_binary_ops() {
    let and = ternlog_imm_for(|x, y| x && y);
    let or = ternlog_imm_for(|x, y| x || y);
    let xor = ternlog_imm_for(|x, y| x != y);
    let andnot = ternlog_imm_for(|x, y| x && !y);

    // Pinned, so a change to the index convention fails HERE with the
    // derivation visible, not inside a mask comparison.
    assert_eq!(and, 0b1000_0000, "and = index 7 only");
    assert_eq!(or, 0b1010_0100, "or = indices 2, 5, 7");
    assert_eq!(xor, 0b0010_0100, "xor = indices 2, 5");
    assert_eq!(andnot, 0b0010_0000, "and_not = index 5 only");
    // The four must be distinct, or the test could pass with one op's
    // immediate standing in for another's.
    let mut seen = [and, or, xor, andnot];
    seen.sort_unstable();
    assert!(
        seen.windows(2).all(|w| w[0] != w[1]),
        "two ops derived the same immediate: {seen:?}"
    );

    for (shape, _, _) in SHAPES {
        for (sa, sb, da, db) in [(1u64, 2u64, 50u64, 50u64), (9, 4, 7, 93), (5, 5, 100, 0)] {
            let a = seeded(shape, sa, da);
            let b = seeded(shape, sb, db);
            let n_rows = a.len() as usize;

            for (name, imm, expect) in [
                ("and", and, a.and(&b)),
                ("or", or, a.or(&b)),
                ("xor", xor, a.xor(&b)),
                ("and_not", andnot, a.and_not(&b)),
            ] {
                let program = Program::new(
                    vec![MaskOp::Ternlog {
                        imm,
                        a: Operand::Plane(0),
                        b: Operand::Plane(1),
                        c: Operand::Plane(0),
                        dst: 0,
                    }],
                    Terminal::Keep {
                        mask: Operand::Scratch(0),
                    },
                );
                let planes: [&[u64]; 2] = [a.words(), b.words()];
                let planes = Planes {
                    n_rows,
                    masks: &planes,
                    lanes: &[],
                };
                let mut scratch = Scratch::for_program(&program, n_rows).expect("scratch");
                execute(&program, &planes, &mut scratch, None).expect("execute");
                assert_eq!(
                    scratch.slot(0).expect("slot 0"),
                    expect.words(),
                    "{shape:?} ternlog imm {imm:#010b} must reproduce CallMask::{name}"
                );
            }
        }
    }
}

/// **Anti-vacuity.** Every comparison above is between two computed masks;
/// if the fixtures were degenerate — all-zero, all-one, or `a == b` — most
/// of the ops would coincide and the differential would pass while proving
/// almost nothing.
#[test]
fn the_fixtures_actually_discriminate() {
    for (shape, _, _) in SHAPES {
        let a = seeded(shape, 1, 50);
        let b = seeded(shape, 2, 50);
        assert_ne!(a.words(), b.words(), "{shape:?}: fixtures are identical");
        assert!(a.count() > 0, "{shape:?}: a is empty");
        assert!(a.count() < a.len(), "{shape:?}: a is full");
        assert_ne!(
            a.and(&b).words(),
            a.or(&b).words(),
            "{shape:?}: and == or, so the ops cannot be told apart"
        );
        assert_ne!(
            a.xor(&b).words(),
            a.and_not(&b).words(),
            "{shape:?}: xor == and_not, so the ops cannot be told apart"
        );
    }
}
