//! `SelFold::CountSet`: how many listed positions have their bit set in a
//! mask plane. Executor against the oracle, against an independent `&[bool]`
//! walk, against the mask path, and on the refusals.

use lance_graph_mask_risc::{
    execute_into, execute_sel, reference_execute_sel, words_for, ExecError, Foreign, LaneRef,
    MaskOp, Operand, Out, Planes, Program, Scratch, SelFold, Terminal, Value,
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

fn pack(rows: &[bool]) -> Vec<u64> {
    let mut words = vec![0u64; words_for(rows.len())];
    for (i, &b) in rows.iter().enumerate() {
        if b {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    words
}

fn count_set(plane: u16) -> SelFold {
    SelFold::CountSet { plane }
}

/// Runs both the executor and the oracle and asserts they agree.
fn both(fold: SelFold, planes: &Planes<'_>, list: &[u32]) -> Result<Value, ExecError> {
    let got = execute_sel(fold, planes, list);
    let want = reference_execute_sel(fold, planes, list);
    assert_eq!(got, want, "executor and oracle disagree on {list:?}");
    got
}

/// FAILS IF: a position is counted once instead of per occurrence, list order
/// matters, an empty list is not zero, or a plane's bits are read from the
/// wrong plane.
#[test]
fn the_four_row_fixture_counts_occurrences() {
    let all = [0b1111u64];
    let some = [0b0111u64];
    let zero = [0u64];
    let masks: [&[u64]; 3] = [&all, &some, &zero];
    let planes = Planes {
        n_rows: 4,
        masks: &masks,
        lanes: &[],
    };
    let table: [(&[u32], usize, usize); 4] = [
        (&[1, 1, 3], 3, 2),
        (&[1, 3, 1], 3, 2),
        (&[], 0, 0),
        (&[1, 1], 2, 2),
    ];
    for (list, a, b) in table {
        assert_eq!(both(count_set(0), &planes, list), Ok(Value::Count(a)));
        assert_eq!(both(count_set(1), &planes, list), Ok(Value::Count(b)));
    }
    let seven = [0u32, 1, 1, 2, 3, 3, 3];
    assert_eq!(both(count_set(2), &planes, &seven), Ok(Value::Count(0)));
    assert_eq!(both(count_set(0), &planes, &seven), Ok(Value::Count(7)));
    assert_eq!(both(count_set(0), &planes, &[1, 3]), Ok(Value::Count(2)));
}

/// FAILS IF: the executor and the oracle disagree, either disagrees with a
/// count taken directly from the bool rows, or a doubled list is not counted
/// twice.
#[test]
fn the_executor_equals_the_oracle_and_a_row_reference() {
    let mut rng = Rng(0x5E7_0001);
    let mut nonzero = 0;
    for n in [0usize, 1, 63, 64, 65, 127, 128, 129, 5000] {
        let rows: Vec<bool> = (0..n).map(|_| rng.next().is_multiple_of(2)).collect();
        let words = pack(&rows);
        let masks: [&[u64]; 1] = [&words];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &[],
        };
        let set: Vec<u32> = (0..n as u32)
            .filter(|_| !rng.next().is_multiple_of(3))
            .collect();
        let reversed: Vec<u32> = set.iter().rev().copied().collect();
        let doubled: Vec<u32> = set.iter().flat_map(|&p| [p, p]).collect();
        let mut single = 0;
        for list in [&set, &reversed, &doubled] {
            let mut k = 0usize;
            for &p in list {
                if rows[p as usize] {
                    k += 1;
                }
            }
            let got = both(count_set(0), &planes, list);
            assert_eq!(got, Ok(Value::Count(k)), "n={n} len={}", list.len());
            if k != 0 {
                nonzero += 1;
            }
            if std::ptr::eq(list, &set) {
                single = k;
            }
            if std::ptr::eq(list, &doubled) {
                assert_eq!(k, 2 * single, "n={n}: doubled must be twice the set");
            }
        }
    }
    // Anti-vacuity: the corpus is not all zero answers.
    assert!(nonzero > 10, "only {nonzero} non-zero answers");
}

/// FAILS IF: for a selection a mask CAN express (a set) the position fold
/// counts differently from the mask path (selection AND bits, then Count).
#[test]
fn a_set_answers_like_the_mask_path() {
    let mut rng = Rng(0x5E7_0002);
    for n in [1usize, 63, 64, 65, 130, 5000] {
        let rows: Vec<bool> = (0..n).map(|_| rng.next().is_multiple_of(2)).collect();
        let bits = pack(&rows);
        let set: Vec<u32> = (0..n as u32)
            .filter(|_| rng.next().is_multiple_of(3))
            .collect();
        let mut sel = vec![0u64; words_for(n)];
        for &p in &set {
            sel[p as usize / 64] |= 1u64 << (p % 64);
        }
        let masks: [&[u64]; 2] = [&bits, &sel];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &[],
        };
        let prog = Program::new(
            vec![MaskOp::And {
                a: Operand::Plane(0),
                b: Operand::Plane(1),
                dst: 0,
            }],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let mut scratch = Scratch::for_program(&prog, n).unwrap();
        let by_mask = execute_into(&prog, &planes, &Foreign::NONE, &mut scratch, Out::None);
        assert_eq!(by_mask, both(count_set(0), &planes, &set), "n={n}");
    }
}

/// FAILS IF: a position that is not a row is counted (even when a mask word
/// covers it), the first offender is misnamed, the tail bits of the last word
/// are read, a missing or malformed plane is accepted, or the executor and
/// the oracle refuse differently.
#[test]
fn non_rows_and_malformed_planes_are_refused() {
    let ones = [u64::MAX, u64::MAX];
    let masks: [&[u64]; 1] = [&ones];
    let planes = Planes {
        n_rows: 65,
        masks: &masks,
        lanes: &[],
    };
    assert_eq!(
        both(count_set(0), &planes, &[0, 63, 64]),
        Ok(Value::Count(3))
    );
    let bad: [(&[u32], usize, u32); 5] = [
        (&[65, 0, 1], 0, 65),
        (&[0, 65, 1], 1, 65),
        (&[0, 1, 65], 2, 65),
        (&[0, 1, 127], 2, 127),
        (&[0, 1, u32::MAX], 2, u32::MAX),
    ];
    for (list, at, position) in bad {
        assert_eq!(
            both(count_set(0), &planes, list),
            Err(ExecError::SelOutOfRange { at, position }),
            "{list:?}"
        );
    }

    // A dirty tail is not read and not refused.
    for tail in [u64::MAX, 1] {
        let words = [u64::MAX, tail];
        let masks: [&[u64]; 1] = [&words];
        let planes = Planes {
            n_rows: 65,
            masks: &masks,
            lanes: &[],
        };
        assert_eq!(
            both(count_set(0), &planes, &[64, 64, 3]),
            Ok(Value::Count(3))
        );
    }
    let clean = [u64::MAX, 0];
    let masks: [&[u64]; 1] = [&clean];
    let planes = Planes {
        n_rows: 65,
        masks: &masks,
        lanes: &[],
    };
    assert_eq!(
        both(count_set(0), &planes, &[64, 64, 3]),
        Ok(Value::Count(1))
    );

    // A plane that is not there.
    assert_eq!(
        both(count_set(1), &planes, &[0]),
        Err(ExecError::PlaneOutOfRange(1))
    );

    // A plane one word short and one word long, with and without a list.
    let short = [u64::MAX];
    let long = [0u64; 3];
    for (words, found) in [(&short[..], 1usize), (&long[..], 3)] {
        let masks: [&[u64]; 1] = [words];
        let planes = Planes {
            n_rows: 65,
            masks: &masks,
            lanes: &[],
        };
        for list in [&[0u32, 1][..], &[][..]] {
            assert_eq!(
                both(count_set(0), &planes, list),
                Err(ExecError::LenMismatch {
                    what: "plane",
                    expected: 2,
                    found
                })
            );
        }
    }

    // The empty frame.
    let empty: [u64; 0] = [];
    let masks: [&[u64]; 1] = [&empty];
    let planes = Planes {
        n_rows: 0,
        masks: &masks,
        lanes: &[],
    };
    assert_eq!(both(count_set(0), &planes, &[]), Ok(Value::Count(0)));
    assert_eq!(
        both(count_set(0), &planes, &[0]),
        Err(ExecError::SelOutOfRange {
            at: 0,
            position: 0
        })
    );
}

/// FAILS IF: the fold reads a lane: a missing lane, or a lane of the wrong
/// length, changes the answer or is refused.
#[test]
fn the_lanes_are_not_read() {
    let words = [0b1011u64];
    let masks: [&[u64]; 1] = [&words];
    let list = [0u32, 1, 1, 2, 3];
    let none = Planes {
        n_rows: 4,
        masks: &masks,
        lanes: &[],
    };
    let wrong_len = [7i32, 8];
    let lanes = [LaneRef::I32(&wrong_len)];
    let with = Planes {
        n_rows: 4,
        masks: &masks,
        lanes: &lanes,
    };
    let a = both(count_set(0), &none, &list);
    let b = both(count_set(0), &with, &list);
    assert_eq!(a, Ok(Value::Count(4)));
    assert_eq!(a, b);
}
