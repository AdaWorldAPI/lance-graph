//! Selection folds ([`execute_sel`]) against the independent oracle, against
//! the mask path for the selections a mask can express, and on the inputs a
//! mask cannot express (a bag, an unsorted list).

use lance_graph_mask_risc::{
    execute_into, execute_sel, reference_execute_sel, words_for, ExecError, Foreign, LaneKind,
    LaneRef, MaskOp, Operand, Out, Planes, Pred, Program, Scratch, SelFold, Terminal, Value,
    SEL_SUM_I32_MAX_POSITIONS,
};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

const LENS: [usize; 7] = [0, 1, 63, 64, 65, 130, 5000];
const THRESHOLDS: [i32; 6] = [i32::MIN, -1, 0, 1, i32::MAX - 1, i32::MAX];

fn lane(n: usize, rng: &mut Rng) -> Vec<i32> {
    (0..n)
        .map(|i| match i % 11 {
            0 => i32::MIN,
            1 => i32::MAX,
            2 => 0,
            _ => rng.next() as i32,
        })
        .collect()
}

/// A sorted, duplicate-free selection of about a third of the rows.
fn a_set(n: usize, rng: &mut Rng) -> Vec<u32> {
    (0..n as u32)
        .filter(|_| rng.next().is_multiple_of(3))
        .collect()
}

fn folds() -> Vec<SelFold> {
    let mut f = vec![SelFold::SumI32 { lane: 0 }];
    f.extend(
        THRESHOLDS
            .iter()
            .map(|&t| SelFold::CountGtI32 { lane: 0, t }),
    );
    f
}

/// FAILS IF: the executor and the row-at-a-time oracle disagree on any fold
/// for a set, the same set reversed, a bag with every row doubled, a bag of
/// one row repeated, or the empty list.
#[test]
fn every_fold_equals_the_oracle_on_sets_and_bags() {
    let mut rng = Rng(0x5E1_0001);
    let mut nonzero = 0;
    for n in LENS {
        let values = lane(n, &mut rng);
        let lanes = [LaneRef::I32(&values)];
        let planes = Planes {
            n_rows: n,
            masks: &[],
            lanes: &lanes,
        };
        let set = a_set(n, &mut rng);
        let reversed: Vec<u32> = set.iter().rev().copied().collect();
        let doubled: Vec<u32> = set.iter().flat_map(|&p| [p, p]).collect();
        let repeated: Vec<u32> = if n == 0 {
            vec![]
        } else {
            vec![(n - 1) as u32; 9]
        };
        for list in [&set, &reversed, &doubled, &repeated, &Vec::new()] {
            for fold in folds() {
                let want = reference_execute_sel(fold, &planes, list);
                let got = execute_sel(fold, &planes, list);
                assert_eq!(got, want, "n={n} fold={fold:?} len={}", list.len());
                assert!(got.is_ok(), "an in-frame list is never refused");
                if !matches!(got, Ok(Value::Count(0)) | Ok(Value::SumI64(0))) {
                    nonzero += 1;
                }
            }
        }
    }
    // Anti-vacuity: the corpus is not all empty answers.
    assert!(nonzero > 100, "only {nonzero} non-zero answers");
}

/// FAILS IF: for a selection a mask CAN express (sorted, duplicate-free) the
/// position fold answers differently from the mask path over the same rows.
/// This is the "a view change does not change the answer" half.
#[test]
fn a_set_answers_like_its_mask() {
    let mut rng = Rng(0x5E1_0002);
    for n in LENS {
        let values = lane(n, &mut rng);
        let set = a_set(n, &mut rng);
        let mut mask = vec![0u64; words_for(n)];
        for &p in &set {
            mask[p as usize / 64] |= 1u64 << (p % 64);
        }
        let lanes = [LaneRef::I32(&values)];
        let masks: [&[u64]; 1] = [&mask];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &lanes,
        };

        let sum = Program::new(
            vec![],
            Terminal::MaskedSumI32 {
                mask: Operand::Plane(0),
                lane: 0,
            },
        );
        let mut scratch = Scratch::for_program(&sum, n).unwrap();
        let by_mask = execute_into(&sum, &planes, &Foreign::NONE, &mut scratch, Out::None);
        assert_eq!(
            execute_sel(SelFold::SumI32 { lane: 0 }, &planes, &set),
            by_mask
        );

        for t in THRESHOLDS {
            let count = Program::new(
                vec![MaskOp::Pred {
                    pred: Pred::GtI32 { lane: 0, t },
                    under: Some(Operand::Plane(0)),
                    dst: 0,
                }],
                Terminal::Count {
                    mask: Operand::Scratch(0),
                },
            );
            let mut scratch = Scratch::for_program(&count, n).unwrap();
            let by_mask = execute_into(&count, &planes, &Foreign::NONE, &mut scratch, Out::None);
            assert_eq!(
                execute_sel(SelFold::CountGtI32 { lane: 0, t }, &planes, &set),
                by_mask,
                "n={n} t={t}"
            );
        }
    }
}

/// FAILS IF: a repeated position is folded once (set semantics), or list
/// order changes an answer. A mask can express neither input, so the mask
/// test above cannot cover this.
#[test]
fn a_bag_keeps_multiplicity_and_ignores_order() {
    let values = [5i32, -7, 100, i32::MAX, i32::MIN, 0];
    let lanes = [LaneRef::I32(&values)];
    let planes = Planes {
        n_rows: values.len(),
        masks: &[],
        lanes: &lanes,
    };
    let run = |fold, list: &[u32]| execute_sel(fold, &planes, list).unwrap();
    let sum = SelFold::SumI32 { lane: 0 };
    let gt = |t| SelFold::CountGtI32 { lane: 0, t };

    assert_eq!(run(sum, &[2]), Value::SumI64(100));
    assert_eq!(run(sum, &[2, 2, 2]), Value::SumI64(300));
    assert_eq!(run(gt(99), &[2, 2, 2]), Value::Count(3));
    assert_eq!(run(gt(100), &[2, 2, 2]), Value::Count(0), "strict");
    assert_eq!(run(sum, &[3, 3]), Value::SumI64(2 * i64::from(i32::MAX)));
    assert_eq!(run(sum, &[4, 4]), Value::SumI64(2 * i64::from(i32::MIN)));
    assert_eq!(run(sum, &[0, 2, 5]), run(sum, &[5, 0, 2]));
    assert_eq!(run(gt(0), &[0, 2, 5]), run(gt(0), &[5, 0, 2]));
    assert_eq!(run(gt(i32::MAX), &[0, 1, 2, 3, 4, 5]), Value::Count(0));
}

/// FAILS IF: a position outside the frame is folded, read as zero or
/// skipped, or the refusal names the wrong place; or the executor and the
/// oracle refuse differently.
#[test]
fn a_position_outside_the_frame_is_refused_and_named() {
    let values = [1i32, 2, 3, 4];
    let lanes = [LaneRef::I32(&values)];
    let planes = Planes {
        n_rows: 4,
        masks: &[],
        lanes: &lanes,
    };
    // The last row is a row; the frame's length is the first non-row.
    assert!(execute_sel(SelFold::SumI32 { lane: 0 }, &planes, &[3]).is_ok());
    let cases: [(&[u32], usize, u32); 4] = [
        (&[4, 0, 1], 0, 4),
        (&[0, 4, 1], 1, 4),
        (&[0, 1, u32::MAX], 2, u32::MAX),
        (&[0, 9, 4], 1, 9), // the FIRST offender is named
    ];
    for (list, at, position) in cases {
        for fold in folds() {
            let want = Err(ExecError::SelOutOfRange { at, position });
            assert_eq!(execute_sel(fold, &planes, list), want, "{list:?} {fold:?}");
            assert_eq!(reference_execute_sel(fold, &planes, list), want);
        }
    }
}

/// FAILS IF: a fold runs against a lane that is missing, of another kind, or
/// not as long as the frame; or the two sides disagree on the refusal.
#[test]
fn a_wrong_lane_is_refused_before_anything_runs() {
    let ints = [1i32, 2, 3];
    let keys = [1u32, 2, 3];
    let short = [1i32, 2];
    let lanes = [
        LaneRef::I32(&ints),
        LaneRef::U32(&keys),
        LaneRef::I32(&short),
    ];
    let planes = Planes {
        n_rows: 3,
        masks: &[],
        lanes: &lanes,
    };
    let both = |fold, want: ExecError| {
        assert_eq!(execute_sel(fold, &planes, &[0]), Err(want));
        assert_eq!(reference_execute_sel(fold, &planes, &[0]), Err(want));
    };
    for make in [
        (|lane| SelFold::SumI32 { lane }) as fn(u16) -> SelFold,
        |lane| SelFold::CountGtI32 { lane, t: 0 },
    ] {
        both(make(3), ExecError::LaneOutOfRange(3));
        both(
            make(1),
            ExecError::LaneKind {
                lane: 1,
                expected: LaneKind::I32,
                found: LaneKind::U32,
            },
        );
        both(
            make(2),
            ExecError::LenMismatch {
                what: "lane",
                expected: 3,
                found: 2,
            },
        );
        // A wrong lane wins over a bad position: nothing is read first.
        assert_eq!(
            execute_sel(make(3), &planes, &[99]),
            Err(ExecError::LaneOutOfRange(3))
        );
    }
    assert!(execute_sel(SelFold::SumI32 { lane: 0 }, &planes, &[0, 1, 2]).is_ok());
}

/// The sum bound is on the LIST, because a bag may be longer than the frame.
/// A list over the bound cannot be allocated in a test, so this pins the
/// constant and the relation the refusal is built on.
#[test]
fn the_sum_bound_is_the_masked_sum_bound() {
    assert_eq!(SEL_SUM_I32_MAX_POSITIONS, 1usize << 32);
    assert_eq!(
        SEL_SUM_I32_MAX_POSITIONS,
        lance_graph_mask_risc::MASKED_SUM_I32_MAX_ROWS
    );
    // 2^32 copies of either extreme fit an i64 exactly.
    let n = SEL_SUM_I32_MAX_POSITIONS as i128;
    assert!(n * i128::from(i32::MAX) <= i128::from(i64::MAX));
    assert!(n * i128::from(i32::MIN) >= i128::from(i64::MIN));
}
