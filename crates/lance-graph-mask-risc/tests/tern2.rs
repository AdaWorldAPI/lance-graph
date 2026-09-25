//! `Program::fused_tern2`: a Boolean chain over FOUR or FIVE resident planes,
//! interpreted as one 32-bit truth table and split as `h(g(x, y, z), u, v)`,
//! then executed as two ternlog passes per chunk instead of one tile pass per
//! op.
//!
//! Every case checks the fused path against a TILED TWIN of the same relation
//! (the same ops plus one self-copy into a slot at [`FUSED_SLOT_CAP`], which no
//! fused recogniser can mark), and the scalar folds also against the crate's
//! independent row-at-a-time oracle.

use lance_graph_mask_risc::{
    execute_extent, reference_execute_into, words_for, Foreign, Lowering, MaskOp, Operand, Out,
    Planes, Program, Scratch, Terminal, Tern2Fold, Value, FUSED_SLOT_CAP,
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

/// A random Boolean chain over planes `0..5`, with slot reuse and derived
/// operands. Returns the ops and the last written slot.
fn random_chain(seed: &mut u64, len: usize) -> (Vec<MaskOp>, u16) {
    let mut written: Vec<u16> = Vec::new();
    let mut ops = Vec::new();
    let pick = |seed: &mut u64, written: &Vec<u16>| {
        if !written.is_empty() && splitmix64(seed).is_multiple_of(3) {
            s(written[(splitmix64(seed) as usize) % written.len()])
        } else {
            p((splitmix64(seed) % 5) as u16)
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

/// FAILS IF: the probe's two multi-plane chains stop decomposing, or the
/// recogniser claims a chain it cannot split. Five-input majority has three
/// distinct non-constant restrictions for every choice of inner triple, so no
/// simple disjoint decomposition exists; it is the can-stay-silent case.
#[test]
fn known_chains_decompose_and_majority_does_not() {
    // (a & b) | (c & d)
    let four = Program::new(
        vec![
            MaskOp::And {
                a: p(0),
                b: p(1),
                dst: 0,
            },
            MaskOp::And {
                a: p(2),
                b: p(3),
                dst: 1,
            },
            MaskOp::Or {
                a: s(0),
                b: s(1),
                dst: 2,
            },
        ],
        Terminal::Count { mask: s(2) },
    );
    assert!(matches!(four.lowering(), Lowering::Tern2(_)));
    assert!(!four.requires_scratch());
    // ((a | b) & c) ^ (d & !e)
    let five = Program::new(
        vec![
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
                dst: 2,
            },
        ],
        Terminal::Any { mask: s(2) },
    );
    assert!(matches!(five.lowering(), Lowering::Tern2(_)));
    let maj = Program::new(
        vec![
            // carry of a+b+c
            MaskOp::Ternlog {
                imm: 0xE8,
                a: p(0),
                b: p(1),
                c: p(2),
                dst: 0,
            },
            // parity of a+b+c
            MaskOp::Ternlog {
                imm: 0x96,
                a: p(0),
                b: p(1),
                c: p(2),
                dst: 1,
            },
            // >=3 of 5  ==  carry & (parity | d | e)  |  !carry & parity & d & e
            MaskOp::Ternlog {
                imm: 0xFE,
                a: s(1),
                b: p(3),
                c: p(4),
                dst: 2,
            },
            MaskOp::And {
                a: s(0),
                b: s(2),
                dst: 3,
            },
            MaskOp::Ternlog {
                imm: 0x80,
                a: s(1),
                b: p(3),
                c: p(4),
                dst: 4,
            },
            MaskOp::AndNot {
                a: s(4),
                b: s(0),
                dst: 4,
            },
            MaskOp::Or {
                a: s(3),
                b: s(4),
                dst: 5,
            },
        ],
        Terminal::Count { mask: s(5) },
    );
    assert!(
        maj.fused_tern2().is_none(),
        "5-input majority has no simple disjoint decomposition"
    );
    assert!(matches!(maj.lowering(), Lowering::Tiled));
    assert!(maj.requires_scratch());
}

/// FAILS IF: for a random chain over up to five planes that the recogniser
/// splits, Count / Any / Keep(Out::Mask) disagree with the tiled twin over any
/// extent, or Count / Any need scratch. Also fails if too few of the split
/// chains genuinely read four or five planes (anti-vacuity).
#[test]
fn tern2_matches_tiled_over_random_chains_and_extents() {
    let ns = [1usize, 63, 64, 65, 200, 1000, 64 * 300 + 17];
    let mut top = 0x7E52_u64;
    let (mut split, mut declined, mut multi) = (0usize, 0usize, 0usize);
    for &n in &ns {
        let mut seed = splitmix64(&mut top) ^ n as u64;
        let ms: Vec<Vec<u64>> = [2, 3, 5, 2, 7]
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
        for _ in 0..40 {
            let len = 2 + (splitmix64(&mut seed) % 6) as usize;
            let (ops, last) = random_chain(&mut seed, len);
            for t in [Term::Count, Term::Any, Term::Keep] {
                let prog = Program::new(ops.clone(), terminal(t, last));
                let Lowering::Tern2(f) = prog.compile().lowering() else {
                    if !matches!(
                        prog.lowering(),
                        Lowering::Ternlog(_) | Lowering::TernlogKeep(_)
                    ) {
                        declined += 1;
                    }
                    continue;
                };
                split += 1;
                let distinct = {
                    let mut v = [f.x, f.y, f.z, f.u, f.v];
                    v.sort_unstable();
                    v.windows(2).filter(|w| w[0] != w[1]).count() + 1
                };
                if distinct >= 4 {
                    multi += 1;
                }
                let tp = force_tiled(&ops, last, t);
                assert!(matches!(tp.lowering(), Lowering::Tiled));
                for (lo, hi) in extents(n) {
                    let mut sc = Scratch::for_program(&tp, n).expect("tiled scratch");
                    match t {
                        Term::Count | Term::Any => {
                            assert!(!prog.requires_scratch());
                            let mut none = Scratch::new(0, 0);
                            let got = execute_extent(
                                &prog,
                                &planes,
                                &Foreign::NONE,
                                &mut none,
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
                            let mut none = Scratch::new(0, 0);
                            let v = execute_extent(
                                &prog,
                                &planes,
                                &Foreign::NONE,
                                &mut none,
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
    // Short random chains almost always decompose (measured: none declined),
    // so the can-stay-silent half lives in the majority test above. What must
    // hold here is that the split path was genuinely exercised on 4-5 planes.
    eprintln!("split={split} multi={multi} declined={declined}");
    assert!(multi > 50, "too few genuinely 4-5-plane splits: {multi}");
}

/// FAILS IF: a split chain's Count / Any disagrees with the independent
/// row-at-a-time oracle, or `Keep` without `Out::Mask` stops falling back to
/// the tiled path (it must still run, from scratch, and agree).
#[test]
fn tern2_matches_the_oracle_and_keep_without_out_mask_runs_tiled() {
    let n = 64 * 9 + 13;
    let mut seed = 0xBEEF_u64;
    let ms: Vec<Vec<u64>> = (0..5).map(|i| random_plane(n, &mut seed, 2 + i)).collect();
    let masks: Vec<&[u64]> = ms.iter().map(|v| v.as_slice()).collect();
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let ops = vec![
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
        // An ODD root table: the dead tail must still be kept out.
        MaskOp::Ternlog {
            imm: 0x69,
            a: s(0),
            b: s(1),
            c: p(3),
            dst: 2,
        },
    ];
    for t in [Term::Count, Term::Any] {
        let prog = Program::new(ops.clone(), terminal(t, 2));
        assert!(matches!(prog.lowering(), Lowering::Tern2(_)));
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
        assert_eq!(got, want);
    }
    let keep = Program::new(ops, terminal(Term::Keep, 2));
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
    assert_eq!(fused, mask);
}
