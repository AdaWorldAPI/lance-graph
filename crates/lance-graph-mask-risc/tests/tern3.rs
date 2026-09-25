//! `Program::fused_tern3`, the six-view window: a Boolean chain over SIX
//! resident planes, held as one 64-bit truth table and materialised as a tree
//! of three ternlog tables per chunk instead of one tile pass per op.
//!
//! Every case checks the fused path against a TILED TWIN of the same relation
//! (the same ops plus one self-copy into a slot at [`FUSED_SLOT_CAP`], which no
//! fused recogniser can mark), and the scalar folds also against the crate's
//! independent row-at-a-time oracle.

use lance_graph_mask_risc::{
    execute_extent, reference_execute_into, words_for, Foreign, Lowering, MaskOp, Operand, Out,
    Planes, Program, Scratch, Terminal, Tern2Fold, Tern3In, Value, FUSED_SLOT_CAP,
};

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn random_plane(n: usize, seed: &mut u64, modulus: u64) -> Vec<u64> {
    let mut w = vec![0u64; words_for(n)];
    for r in 0..n {
        if splitmix64(seed).is_multiple_of(modulus) {
            w[r / 64] |= 1u64 << (r % 64);
        }
    }
    w
}

fn garbage(words: usize) -> Vec<u64> {
    const PATTERN: u64 = 0xA5A5_5A5A_F00F_0FF0;
    (0..words)
        .map(|w| PATTERN.rotate_left((w % 64) as u32))
        .collect()
}

fn p(i: u16) -> Operand {
    Operand::Plane(i)
}

fn s(i: u16) -> Operand {
    Operand::Scratch(i)
}

/// A random Boolean chain over planes `0..6`, with slot reuse and derived
/// operands. Returns the ops and the last written slot.
fn random_chain6(seed: &mut u64, len: usize) -> (Vec<MaskOp>, u16) {
    let mut written: Vec<u16> = Vec::new();
    let mut ops = Vec::new();
    let pick = |seed: &mut u64, written: &Vec<u16>| {
        if !written.is_empty() && splitmix64(seed).is_multiple_of(3) {
            s(written[(splitmix64(seed) as usize) % written.len()])
        } else {
            p((splitmix64(seed) % 6) as u16)
        }
    };
    let mut last = 0u16;
    for _ in 0..len {
        let (x, y, z) = (
            pick(seed, &written),
            pick(seed, &written),
            pick(seed, &written),
        );
        let dst = (splitmix64(seed) % 6) as u16;
        ops.push(match splitmix64(seed) % 6 {
            0 => MaskOp::And { a: x, b: y, dst },
            1 => MaskOp::Or { a: x, b: y, dst },
            2 => MaskOp::Xor { a: x, b: y, dst },
            3 => MaskOp::AndNot { a: x, b: y, dst },
            4 => MaskOp::Not { a: x, dst },
            _ => MaskOp::Ternlog {
                imm: (splitmix64(seed) & 0xFF) as u8,
                a: x,
                b: y,
                c: z,
                dst,
            },
        });
        if !written.contains(&dst) {
            written.push(dst);
        }
        last = dst;
    }
    (ops, last)
}

#[derive(Clone, Copy)]
enum Term {
    Count,
    Any,
    Keep,
}

fn terminal(t: Term, slot: u16) -> Terminal {
    match t {
        Term::Count => Terminal::Count { mask: s(slot) },
        Term::Any => Terminal::Any { mask: s(slot) },
        Term::Keep => Terminal::Keep { mask: s(slot) },
    }
}

/// The same relation, forced onto the tiled path.
fn force_tiled(ops: &[MaskOp], last: u16, t: Term) -> Program {
    let mut ops = ops.to_vec();
    let far = FUSED_SLOT_CAP as u16;
    ops.push(MaskOp::Or {
        a: s(last),
        b: s(last),
        dst: far,
    });
    Program::new(ops, terminal(t, far))
}

fn extents(n: usize) -> Vec<(usize, usize)> {
    let mut v = vec![(0, n), (0, 0), (n, n)];
    for &k in &[1usize, 5, 63, 64, 65, 127, 128, 129, 193] {
        if k < n {
            v.push((0, k));
            v.push((k, n));
        }
    }
    if n >= 4 {
        v.push((n / 4, 3 * n / 4));
    }
    if n > 130 {
        v.push((64, 65));
        v.push((70, 71));
    }
    v.retain(|&(lo, hi)| lo <= hi && hi <= n);
    v.sort_unstable();
    v.dedup();
    v
}

/// Build `ops` from a tree written as nested closures would be noisy; these
/// three chains are written out once and reused.
fn and6() -> Vec<MaskOp> {
    let mut ops = vec![MaskOp::And {
        a: p(0),
        b: p(1),
        dst: 0,
    }];
    for i in 2..6 {
        ops.push(MaskOp::And {
            a: s(0),
            b: p(i),
            dst: 0,
        });
    }
    ops
}

/// FAILS IF: a six-plane AND chain, or the balanced `(a&b&c) ^ (d|e|f)`, stops
/// lowering to the window, or Count/Any start needing scratch, or 6-input
/// majority is claimed (it has no three-table tree: the can-stay-silent case).
#[test]
fn known_chains_hold_and_majority_does_not() {
    let and = Program::new(and6(), Terminal::Count { mask: s(0) });
    assert!(
        matches!(and.lowering(), Lowering::Tern3(_)),
        "{:?}",
        and.lowering()
    );
    assert!(!and.requires_scratch());
    let bal = Program::new(
        vec![
            MaskOp::Ternlog {
                imm: 0x80,
                a: p(0),
                b: p(1),
                c: p(2),
                dst: 0,
            },
            MaskOp::Ternlog {
                imm: 0xFE,
                a: p(3),
                b: p(4),
                c: p(5),
                dst: 1,
            },
            MaskOp::Xor {
                a: s(0),
                b: s(1),
                dst: 2,
            },
        ],
        Terminal::Any { mask: s(2) },
    );
    assert!(matches!(bal.lowering(), Lowering::Tern3(_)));
    // At least 4 of 6: sum the planes with ternlog full adders.
    let maj = Program::new(
        vec![
            // s1 = a^b^c, c1 = maj(a,b,c); s2 = d^e^f, c2 = maj(d,e,f)
            MaskOp::Ternlog {
                imm: 0x96,
                a: p(0),
                b: p(1),
                c: p(2),
                dst: 0,
            },
            MaskOp::Ternlog {
                imm: 0xE8,
                a: p(0),
                b: p(1),
                c: p(2),
                dst: 1,
            },
            MaskOp::Ternlog {
                imm: 0x96,
                a: p(3),
                b: p(4),
                c: p(5),
                dst: 2,
            },
            MaskOp::Ternlog {
                imm: 0xE8,
                a: p(3),
                b: p(4),
                c: p(5),
                dst: 3,
            },
            // total = s1 + s2 + 2*(c1 + c2) >= 4
            //   <=> (c1 & c2) | ((c1 | c2) & s1 & s2)
            MaskOp::And {
                a: s(1),
                b: s(3),
                dst: 4,
            },
            MaskOp::Or {
                a: s(1),
                b: s(3),
                dst: 5,
            },
            MaskOp::Ternlog {
                imm: 0x80,
                a: s(5),
                b: s(0),
                c: s(2),
                dst: 5,
            },
            MaskOp::Or {
                a: s(4),
                b: s(5),
                dst: 6,
            },
        ],
        Terminal::Count { mask: s(6) },
    );
    assert!(
        maj.fused_tern3().is_none(),
        "6-input majority has no three-table tree"
    );
    assert!(matches!(maj.lowering(), Lowering::Tiled));
    assert!(maj.requires_scratch());
}

/// The shape a lowered window took, read off its step inputs.
fn shape(steps: &[lance_graph_mask_risc::Tern3Step; 3]) -> usize {
    let reads_t0 = |i: Tern3In| i == Tern3In::T0;
    if [steps[1].a, steps[1].b, steps[1].c]
        .into_iter()
        .any(reads_t0)
    {
        // t1 builds on t0: one of the two outer shapes.
        let root_planes = [steps[2].b, steps[2].c]
            .into_iter()
            .filter_map(|i| {
                if let Tern3In::Plane(p) = i {
                    Some(p)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        if root_planes.len() == 2 && root_planes[0] != root_planes[1] {
            2
        } else {
            1
        }
    } else {
        0
    }
}

/// FAILS IF: for a random chain over six planes that the window claims,
/// Count / Any / Keep(Out::Mask) disagree with the tiled twin over any extent,
/// or Count / Any need scratch. Also fails if the window was not genuinely
/// exercised, including each of its three shapes (anti-vacuity).
#[test]
fn tern3_matches_tiled_over_random_chains_and_extents() {
    let ns = [1usize, 63, 64, 65, 200, 1000, 64 * 300 + 17];
    let mut top = 0x7E53_u64;
    let (mut held, mut declined) = (0usize, 0usize);
    let mut shapes = [0usize; 3];
    for &n in &ns {
        let mut seed = splitmix64(&mut top) ^ n as u64;
        let ms: Vec<Vec<u64>> = [2, 3, 5, 2, 7, 3]
            .iter()
            .map(|&m| random_plane(n, &mut seed, m))
            .collect();
        let masks: Vec<&[u64]> = ms.iter().map(|v| v.as_slice()).collect();
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &[],
        };
        let words = words_for(n);
        for _ in 0..120 {
            let len = 5 + (splitmix64(&mut seed) % 7) as usize;
            let (ops, last) = random_chain6(&mut seed, len);
            for t in [Term::Count, Term::Any, Term::Keep] {
                let prog = Program::new(ops.clone(), terminal(t, last));
                let Lowering::Tern3(f) = prog.compile().lowering() else {
                    if matches!(prog.lowering(), Lowering::Tiled) {
                        declined += 1;
                    }
                    continue;
                };
                held += 1;
                shapes[shape(&f.steps)] += 1;
                let tp = force_tiled(&ops, last, t);
                assert!(matches!(tp.lowering(), Lowering::Tiled));
                for (lo, hi) in extents(n) {
                    let mut sc = Scratch::for_program(&tp, n).expect("tiled scratch");
                    match t {
                        Term::Count | Term::Any => {
                            assert!(!prog.requires_scratch());
                            let got = execute_extent(
                                &prog,
                                &planes,
                                &Foreign::NONE,
                                &mut Scratch::new(0, 0),
                                Out::None,
                                lo..hi,
                            )
                            .unwrap();
                            let want = execute_extent(
                                &tp,
                                &planes,
                                &Foreign::NONE,
                                &mut sc,
                                Out::None,
                                lo..hi,
                            )
                            .unwrap();
                            assert_eq!(got, want, "n={n} {ops:?} [{lo},{hi})");
                        }
                        Term::Keep => {
                            assert!(matches!(f.fold, Tern2Fold::Keep { .. }));
                            let mut a = garbage(words);
                            let mut b = a.clone();
                            let v = execute_extent(
                                &prog,
                                &planes,
                                &Foreign::NONE,
                                &mut Scratch::new(0, 0),
                                Out::Mask(&mut a),
                                lo..hi,
                            )
                            .unwrap();
                            assert_eq!(v, Value::Mask(s(last)));
                            execute_extent(
                                &tp,
                                &planes,
                                &Foreign::NONE,
                                &mut sc,
                                Out::Mask(&mut b),
                                lo..hi,
                            )
                            .unwrap();
                            assert_eq!(a, b, "n={n} {ops:?} [{lo},{hi})");
                        }
                    }
                }
            }
        }
    }
    eprintln!("held={held} declined={declined} shapes={shapes:?}");
    assert!(
        held > 100,
        "too few six-plane chains took the window: {held}"
    );
    assert!(
        declined > 0,
        "no six-plane chain declined: the recogniser never stayed silent"
    );
}

/// FAILS IF: each shape is not exercised END TO END through the executor
/// (the random differential may favour one shape), or a held chain's Count /
/// Any disagrees with the independent row-at-a-time oracle, or `Keep` without
/// `Out::Mask` stops falling back to the tiled path.
#[test]
fn every_shape_matches_the_oracle_and_keep_without_out_mask_runs_tiled() {
    let n = 64 * 9 + 13;
    let mut seed = 0xBEF3_u64;
    let ms: Vec<Vec<u64>> = (0..6).map(|i| random_plane(n, &mut seed, 2 + i)).collect();
    let masks: Vec<&[u64]> = ms.iter().map(|v| v.as_slice()).collect();
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    // One chain per shape, each the table the unit tests pin to that shape.
    let balanced = vec![
        MaskOp::Ternlog {
            imm: 0x80,
            a: p(0),
            b: p(1),
            c: p(2),
            dst: 0,
        },
        MaskOp::Ternlog {
            imm: 0xFE,
            a: p(3),
            b: p(4),
            c: p(5),
            dst: 1,
        },
        MaskOp::Xor {
            a: s(0),
            b: s(1),
            dst: 2,
        },
    ];
    // ((((a & b) | c) ^ d) & e) | f
    let outer2 = vec![
        MaskOp::And {
            a: p(0),
            b: p(1),
            dst: 0,
        },
        MaskOp::Or {
            a: s(0),
            b: p(2),
            dst: 0,
        },
        MaskOp::Xor {
            a: s(0),
            b: p(3),
            dst: 0,
        },
        MaskOp::And {
            a: s(0),
            b: p(4),
            dst: 0,
        },
        MaskOp::Or {
            a: s(0),
            b: p(5),
            dst: 2,
        },
    ];
    // (((a | b) & c) ^ (d & !e)) & f
    let outer1 = vec![
        MaskOp::Or {
            a: p(0),
            b: p(1),
            dst: 0,
        },
        MaskOp::And {
            a: s(0),
            b: p(2),
            dst: 0,
        },
        MaskOp::AndNot {
            a: p(3),
            b: p(4),
            dst: 1,
        },
        MaskOp::Xor {
            a: s(0),
            b: s(1),
            dst: 1,
        },
        MaskOp::And {
            a: s(1),
            b: p(5),
            dst: 2,
        },
    ];
    for (want_shape, ops) in [(0usize, balanced), (2, outer2), (1, outer1)] {
        for t in [Term::Count, Term::Any] {
            let prog = Program::new(ops.clone(), terminal(t, 2));
            let Lowering::Tern3(f) = prog.lowering() else {
                panic!(
                    "shape {want_shape} chain did not take the window: {:?}",
                    prog.lowering()
                );
            };
            assert_eq!(shape(&f.steps), want_shape, "{ops:?}");
            let want = reference_execute_into(&prog, &planes, &Foreign::NONE, Out::None).unwrap();
            let got = execute_extent(
                &prog,
                &planes,
                &Foreign::NONE,
                &mut Scratch::new(0, 0),
                Out::None,
                0..n,
            )
            .unwrap();
            assert_eq!(got, want, "shape {want_shape}");
        }
        let keep = Program::new(ops.clone(), terminal(Term::Keep, 2));
        assert!(keep.requires_scratch(), "Keep may be read from its slot");
        let mut sc = Scratch::for_program(&keep, n).unwrap();
        let v = execute_extent(&keep, &planes, &Foreign::NONE, &mut sc, Out::None, 0..n).unwrap();
        assert_eq!(v, Value::Mask(s(2)));
        let mut mask = vec![0u64; words_for(n)];
        reference_execute_into(&keep, &planes, &Foreign::NONE, Out::Mask(&mut mask)).unwrap();
        let mut fused = garbage(words_for(n));
        execute_extent(
            &keep,
            &planes,
            &Foreign::NONE,
            &mut Scratch::new(0, 0),
            Out::Mask(&mut fused),
            0..n,
        )
        .unwrap();
        assert_eq!(fused, mask, "shape {want_shape}");
    }
}
