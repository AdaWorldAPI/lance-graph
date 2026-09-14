//! D-MRX-5: the executor against the scalar oracle, on every backend this
//! binary is built for. Seeded planes at eight row counts; every `Pred` with
//! and without a gate; every two-input op in every aliasing shape; `Not`; all
//! 256 ternlog immediates; every `Terminal`. Values AND scratch contents are
//! compared, so a wrong word that a terminal happens to hide is still caught.

use lance_graph_mask_risc::exec::{execute, Scratch};
use lance_graph_mask_risc::reference::{reference_execute, reference_scratch};
use lance_graph_mask_risc::{
    words_for, LaneRef, MaskOp, Operand, Planes, Pred, Program, Terminal, Value,
};

const ROWS: [usize; 8] = [0, 1, 63, 64, 65, 130, 1000, 65_536];

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

struct Fixture {
    n: usize,
    masks: Vec<Vec<u64>>,
    i32s: Vec<i32>,
    u32s: Vec<u32>,
    u64s: Vec<u64>,
    /// A SECOND, distinct `i32` lane (index 3). With `then == els` a blend
    /// fixture is vacuous — `out` equals lane 0 under every mask, so swapping
    /// the branches in either implementation changes nothing.
    i32s_b: Vec<i32>,
}

impl Fixture {
    fn new(n: usize, seed: u64) -> Self {
        let mut s = seed ^ (n as u64);
        let words = words_for(n);
        let mut masks = Vec::new();
        for density in [2u32, 1, 3] {
            let mut m: Vec<u64> = (0..words)
                .map(|_| {
                    let mut w = lcg(&mut s);
                    for _ in 0..density {
                        w &= lcg(&mut s);
                    }
                    w
                })
                .collect();
            if !n.is_multiple_of(64) {
                m[words - 1] &= (1u64 << (n % 64)) - 1;
            }
            masks.push(m);
        }
        let mut i32s: Vec<i32> = (0..n).map(|_| (lcg(&mut s) % 2000) as i32 - 1000).collect();
        let mut u32s: Vec<u32> = (0..n).map(|_| (lcg(&mut s) % 64) as u32).collect();
        // the equality needles (`== 3`, `== 5`) are rare by construction; plant
        // one hit so the can-it-fire half of every predicate is real at n >= 8
        let mut u64s: Vec<u64> = (0..n).map(|_| lcg(&mut s) & 0xFFFF).collect();
        if n > 8 {
            i32s[7] = 3;
            u32s[7] = 5;
            u32s[8] = 0b10_0000;
            u64s[8] = 0x1000;
        }
        let i32s_b = (0..n).map(|_| -1 - (lcg(&mut s) % 500) as i32).collect();
        Self {
            n,
            masks,
            i32s,
            u32s,
            u64s,
            i32s_b,
        }
    }

    /// Execute `p` on BOTH sides over this fixture and assert they agree —
    /// the returned [`Value`], and every scratch slot word for word.
    ///
    /// `name` is only ever read out of an assertion message: when a shape
    /// fails somewhere inside a generated sweep of hundreds of programs, the
    /// index alone does not say which one.
    fn run(&self, p: &Program, name: &str) {
        let masks: Vec<&[u64]> = self.masks.iter().map(|m| m.as_slice()).collect();
        let lanes = [
            LaneRef::I32(&self.i32s),
            LaneRef::U32(&self.u32s),
            LaneRef::U64(&self.u64s),
            LaneRef::I32(&self.i32s_b),
        ];
        let planes = Planes {
            n_rows: self.n,
            masks: &masks,
            lanes: &lanes,
        };
        let mut scratch = Scratch::for_program(p, self.n).expect("addressable");
        let mut out_exec = vec![0i32; self.n];
        let mut out_ref = vec![0i32; self.n];
        let got = execute(p, &planes, &mut scratch, Some(&mut out_exec));
        let want = reference_execute(p, &planes, Some(&mut out_ref));
        assert_eq!(got, want, "{name} @ n={}: value", self.n);
        assert_eq!(out_exec, out_ref, "{name} @ n={}: blend output", self.n);
        if let Ok(slots) = reference_scratch(p, &planes) {
            for (i, want_words) in slots.iter().enumerate() {
                let got_words = scratch.slot(i as u16).unwrap_or(&[]);
                assert_eq!(
                    got_words,
                    want_words.as_slice(),
                    "{name} @ n={}: scratch slot {i}",
                    self.n
                );
            }
        }
    }

    fn count(&self, p: &Program) -> usize {
        let masks: Vec<&[u64]> = self.masks.iter().map(|m| m.as_slice()).collect();
        let lanes = [
            LaneRef::I32(&self.i32s),
            LaneRef::U32(&self.u32s),
            LaneRef::U64(&self.u64s),
            LaneRef::I32(&self.i32s_b),
        ];
        let planes = Planes {
            n_rows: self.n,
            masks: &masks,
            lanes: &lanes,
        };
        match reference_execute(p, &planes, None) {
            Ok(Value::Count(c)) => c,
            other => panic!("expected a count, got {other:?}"),
        }
    }
}

const P0: Operand = Operand::Plane(0);
const P1: Operand = Operand::Plane(1);
const P2: Operand = Operand::Plane(2);
const S0: Operand = Operand::Scratch(0);
const S1: Operand = Operand::Scratch(1);

fn all_preds() -> Vec<Pred> {
    vec![
        Pred::GtI32 { lane: 0, t: 600 },
        Pred::LtI32 { lane: 0, t: -600 },
        Pred::GeI32 { lane: 0, t: 700 },
        Pred::LeI32 { lane: 0, t: -700 },
        Pred::EqI32 { lane: 0, v: 3 },
        Pred::NeI32 { lane: 0, v: 3 },
        Pred::EqU32 { lane: 1, v: 5 },
        Pred::NeU32 { lane: 1, v: 5 },
        Pred::MatchU32 {
            lane: 1,
            pattern: 0b10_0000,
            care: 0b11_1000,
        },
        Pred::MatchU64 {
            lane: 2,
            pattern: 0x1000,
            care: 0xF000,
        },
    ]
}

/// FAILS IF: any predicate, gated or not, disagrees with the row-at-a-time
/// oracle on any backend — or the fixtures are vacuous (the `survivors * 3 <
/// n_rows` guard on the non-`Ne` predicates at `n_rows >= 3`).
#[test]
fn every_predicate_with_and_without_a_gate() {
    for n in ROWS {
        let f = Fixture::new(n, 11);
        for pred in all_preds() {
            for under in [None, Some(P0), Some(S1)] {
                let mut ops = Vec::new();
                if under == Some(S1) {
                    ops.push(MaskOp::Not { a: P1, dst: 1 });
                }
                ops.push(MaskOp::Pred {
                    pred,
                    under,
                    dst: 0,
                });
                let p = Program::new(ops, Terminal::Keep { mask: S0 });
                f.run(&p, &format!("{pred:?} under {under:?}"));
                let is_ne = matches!(pred, Pred::NeI32 { .. } | Pred::NeU32 { .. });
                if n >= 3 && !is_ne && under.is_none() {
                    let c = Program::new(
                        vec![MaskOp::Pred {
                            pred,
                            under: None,
                            dst: 0,
                        }],
                        Terminal::Count { mask: S0 },
                    );
                    let survivors = f.count(&c);
                    assert!(
                        survivors * 3 < n,
                        "{pred:?} @ n={n}: {survivors} survivors is not a selective fixture"
                    );
                    assert!(
                        survivors > 0 || n < 9,
                        "{pred:?} @ n={n}: the predicate never fires"
                    );
                }
                // ...and the GATE must bind: a gate that admits everything the
                // predicate already admits proves nothing about gating.
                if n >= 64 && under == Some(P0) {
                    let gated = Program::new(
                        vec![MaskOp::Pred {
                            pred,
                            under,
                            dst: 0,
                        }],
                        Terminal::Count { mask: S0 },
                    );
                    let ungated = Program::new(
                        vec![MaskOp::Pred {
                            pred,
                            under: None,
                            dst: 0,
                        }],
                        Terminal::Count { mask: S0 },
                    );
                    assert!(
                        f.count(&gated) < f.count(&ungated),
                        "{pred:?} @ n={n}: the gate removes nothing"
                    );
                }
            }
        }
    }
}

/// FAILS IF: a two-input op is wrong in any aliasing shape — `dst == a`,
/// `dst == b`, `dst == a == b`, or no alias — including the non-commutative
/// `AndNot` with `dst` on the right (the re-indexed in-place ternlog).
#[test]
fn every_two_input_op_in_every_aliasing_shape() {
    type Mk = fn(Operand, Operand, u16) -> MaskOp;
    let ops: [(&str, Mk); 4] = [
        ("And", |a, b, dst| MaskOp::And { a, b, dst }),
        ("Or", |a, b, dst| MaskOp::Or { a, b, dst }),
        ("Xor", |a, b, dst| MaskOp::Xor { a, b, dst }),
        ("AndNot", |a, b, dst| MaskOp::AndNot { a, b, dst }),
    ];
    for n in ROWS {
        let f = Fixture::new(n, 23);
        for (name, mk) in ops {
            let setup = [MaskOp::Not { a: P0, dst: 0 }, MaskOp::Not { a: P1, dst: 1 }];
            let shapes: [(&str, MaskOp); 6] = [
                ("plain", mk(P0, P1, 0)),
                ("dst==a", mk(S0, P2, 0)),
                ("dst==b", mk(P2, S0, 0)),
                ("dst==a==b", mk(S0, S0, 0)),
                ("scratch both", mk(S0, S1, 0)),
                ("a==b, no alias", mk(P1, P1, 0)),
            ];
            for (shape, op) in shapes {
                let mut v = setup.to_vec();
                v.push(op);
                let p = Program::new(v, Terminal::Keep { mask: S0 });
                f.run(&p, &format!("{name} {shape}"));
            }
        }
    }
}

/// FAILS IF: `Not` breaks the tail law in either shape.
#[test]
fn not_in_both_shapes() {
    for n in ROWS {
        let f = Fixture::new(n, 5);
        f.run(
            &Program::new(
                vec![MaskOp::Not { a: P0, dst: 0 }],
                Terminal::Keep { mask: S0 },
            ),
            "Not plain",
        );
        f.run(
            &Program::new(
                vec![MaskOp::Not { a: P0, dst: 0 }, MaskOp::Not { a: S0, dst: 0 }],
                Terminal::Keep { mask: S0 },
            ),
            "Not in place",
        );
    }
}

/// FAILS IF: any of the 256 immediates is wrong in the plain shape or with
/// `dst == a` — including every ODD immediate's tail clear (a whole-word
/// read of the scratch would show phantom bits the oracle never sets).
#[test]
fn all_256_immediates_plain_and_in_place() {
    for n in [0, 1, 65, 130] {
        let f = Fixture::new(n, 31);
        for imm in 0..=255u8 {
            f.run(
                &Program::new(
                    vec![MaskOp::Ternlog {
                        imm,
                        a: P0,
                        b: P1,
                        c: P2,
                        dst: 0,
                    }],
                    Terminal::Keep { mask: S0 },
                ),
                &format!("ternlog {imm:#04x} plain"),
            );
            f.run(
                &Program::new(
                    vec![
                        MaskOp::Not { a: P0, dst: 0 },
                        MaskOp::Ternlog {
                            imm,
                            a: S0,
                            b: P1,
                            c: P2,
                            dst: 0,
                        },
                    ],
                    Terminal::Keep { mask: S0 },
                ),
                &format!("ternlog {imm:#04x} dst==a"),
            );
        }
    }
}

/// FAILS IF: the immediate re-indexing is wrong for any aliasing map — every
/// way `dst` can appear among `(a, b, c)`, for a spread of asymmetric tables.
#[test]
fn ternlog_every_aliasing_map() {
    // 0xFF / 0x81 / 0xE7 carry BOTH bit 0 and bit 7: they are the only shapes
    // that reach `ternlog_self`'s fill-ones arm under the `x,x,x` alias, and
    // without one of them that arm can be deleted with every test still green.
    let imms = [
        0x01u8, 0x0C, 0x30, 0x40, 0x80, 0xA8, 0xCA, 0xE2, 0xFE, 0x96, 0x35, 0x5A, 0xFF, 0x81, 0xE7,
    ];
    for n in [63, 130, 1000] {
        let f = Fixture::new(n, 41);
        for imm in imms {
            let setup = [MaskOp::Not { a: P0, dst: 0 }, MaskOp::Not { a: P1, dst: 1 }];
            let shapes: [(&str, [Operand; 3]); 8] = [
                ("x,b,c", [S0, P1, P2]),
                ("a,x,c", [P1, S0, P2]),
                ("a,b,x", [P1, P2, S0]),
                ("x,x,c", [S0, S0, P2]),
                ("x,b,x", [S0, P2, S0]),
                ("a,x,x", [P2, S0, S0]),
                ("x,x,x", [S0, S0, S0]),
                ("x,s1,s1", [S0, S1, S1]),
            ];
            for (shape, [a, b, c]) in shapes {
                let mut v = setup.to_vec();
                v.push(MaskOp::Ternlog {
                    imm,
                    a,
                    b,
                    c,
                    dst: 0,
                });
                f.run(
                    &Program::new(v, Terminal::Keep { mask: S0 }),
                    &format!("ternlog {imm:#04x} {shape}"),
                );
            }
        }
    }
}

/// FAILS IF: any terminal disagrees with the oracle, including the empty
/// (`n = 0`, `n = 1`) cases and the `Option` terminals over an empty mask.
#[test]
fn every_terminal() {
    for n in ROWS {
        let f = Fixture::new(n, 59);
        let pred = MaskOp::Pred {
            pred: Pred::GtI32 { lane: 0, t: 0 },
            under: Some(P0),
            dst: 0,
        };
        let empty = MaskOp::AndNot {
            a: P0,
            b: P0,
            dst: 1,
        };
        let terminals = [
            Terminal::Count { mask: S0 },
            Terminal::Any { mask: S0 },
            Terminal::Any { mask: S1 },
            Terminal::All { mask: S0 },
            Terminal::All { mask: S1 },
            Terminal::MaskedSumI32 { mask: S0, lane: 0 },
            Terminal::MaskedMinI32 { mask: S0, lane: 0 },
            Terminal::MaskedMaxI32 { mask: S0, lane: 0 },
            Terminal::MaskedMinI32 { mask: S1, lane: 0 },
            Terminal::MaskedMaxI32 { mask: S1, lane: 0 },
            Terminal::BlendI32 {
                mask: S0,
                then: 0,
                els: 3,
            },
            Terminal::Keep { mask: P2 },
        ];
        for t in terminals {
            f.run(&Program::new(vec![pred, empty], t), &format!("{t:?}"));
        }
    }
}
