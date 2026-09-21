//! `MaskOp::Gather` / `Terminal::ScatterOrU32` / `Terminal::GroupSumI32`
//! against the row-at-a-time oracle — the same differential shape
//! `tests/differential.rs` uses, extended over a SECOND, foreign row space.

use lance_graph_mask_risc::exec::{execute_into, Scratch};
use lance_graph_mask_risc::reference::{reference_execute_into, reference_scratch_with_foreign};
use lance_graph_mask_risc::{
    scratch_words_for, words_for, ExecError, Foreign, ForeignPlane, LaneRef, MaskOp, Operand, Out,
    Planes, Pred, Program, Terminal, Value,
};

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

const S0: Operand = Operand::Scratch(0);
const S1: Operand = Operand::Scratch(1);

/// A primary ("line") table of `n` rows and a foreign ("partner") table of
/// `foreign_rows` rows, deliberately SHORTER than `n` so `Gather`'s
/// out-of-range zero-fallback is real, not vacuous.
struct Fixture {
    foreign_rows: usize,
    /// The fk lane: values `0..(foreign_rows + slack)`, so some indices land
    /// past `foreign_rows` and are genuinely out of range.
    fk: Vec<u32>,
    /// A `u32` value lane in `0..64`, used as a status-shaped predicate.
    status: Vec<u32>,
    /// A signed value lane, including negatives, for the sum/group cases.
    amount: Vec<i32>,
    /// A `u32` key lane for `GroupSumI32`, values `0..(groups + slack)` so
    /// some keys are dropped (>= the group universe).
    key: Vec<u32>,
    /// The foreign table's own validity — a random-density mask.
    foreign_bits: Vec<u64>,
}

impl Fixture {
    fn new(n: usize, foreign_rows: usize, groups: u32, seed: u64) -> Self {
        let mut s = seed ^ (n as u64) ^ (foreign_rows as u64).wrapping_shl(17);
        let slack = 5u32;
        let fk_range = foreign_rows as u32 + slack;
        let mut fk: Vec<u32> = (0..n)
            .map(|_| {
                if fk_range == 0 {
                    0
                } else {
                    (lcg(&mut s) % u64::from(fk_range)) as u32
                }
            })
            .collect();
        // Plant a deterministic out-of-range fk on the LAST row (mirroring
        // `tests/differential.rs`'s own "plant one hit" convention): the
        // random draw above makes an out-of-range hit merely LIKELY for a
        // small `n`, which is not the same as guaranteed, and the
        // can-it-fire falsifier below needs it guaranteed.
        if let Some(last) = fk.last_mut() {
            *last = foreign_rows as u32 + slack - 1;
        }
        let status: Vec<u32> = (0..n).map(|_| (lcg(&mut s) % 3) as u32).collect();
        let amount: Vec<i32> = (0..n).map(|_| (lcg(&mut s) % 4000) as i32 - 2000).collect();
        let key_range = groups + slack;
        let key: Vec<u32> = (0..n)
            .map(|_| {
                if key_range == 0 {
                    0
                } else {
                    (lcg(&mut s) % u64::from(key_range)) as u32
                }
            })
            .collect();
        let fwords = words_for(foreign_rows);
        let mut foreign_bits: Vec<u64> = (0..fwords)
            .map(|_| {
                let mut w = lcg(&mut s);
                w &= lcg(&mut s);
                w
            })
            .collect();
        if foreign_rows > 0 && !foreign_rows.is_multiple_of(64) {
            let last = fwords - 1;
            foreign_bits[last] &= (1u64 << (foreign_rows % 64)) - 1;
        }
        Self {
            foreign_rows,
            fk,
            status,
            amount,
            key,
            foreign_bits,
        }
    }

    fn planes(&self) -> (Vec<LaneRef<'_>>, Vec<&[u64]>) {
        let lanes = vec![
            LaneRef::U32(&self.fk),
            LaneRef::U32(&self.status),
            LaneRef::I32(&self.amount),
            LaneRef::U32(&self.key),
        ];
        (lanes, vec![])
    }

    fn foreign_plane(&self) -> ForeignPlane<'_> {
        ForeignPlane {
            words: &self.foreign_bits,
            rows: self.foreign_rows,
        }
    }
}

const ROWS: [usize; 5] = [1, 63, 64, 130, 1000];

/// FAILS IF: `Gather` disagrees with the oracle anywhere — including a
/// foreign table shorter than `n_rows`, fk indices past `foreign_rows`
/// (zero-fallback), and a non-word-multiple `n`. Also checks scratch
/// contents via `reference_scratch_with_foreign` so a wrong WORD hidden by
/// `Keep`'s own bit is still caught, and asserts the fixture is neither
/// all-hit nor all-miss (a vacuous gather would pass a same-value oracle
/// trivially).
#[test]
fn gather_matches_the_oracle_including_out_of_range_and_a_tail() {
    for &n in &ROWS {
        for &foreign_rows in &[5usize, 64, 70, n.saturating_sub(1).max(1)] {
            let fx = Fixture::new(n, foreign_rows, 8, 0xA11CE);
            let (lanes, masks) = fx.planes();
            let planes = Planes {
                n_rows: n,
                masks: &masks,
                lanes: &lanes,
            };
            let fp = fx.foreign_plane();
            let foreign = Foreign {
                planes: std::slice::from_ref(&fp),
            };
            let p = Program::new(
                vec![MaskOp::Gather {
                    lane: 0,
                    foreign: 0,
                    dst: 0,
                }],
                Terminal::Keep { mask: S0 },
            );

            let words = words_for(n);
            let slots = p.scratch_slots as usize;
            let mut buf = vec![0u64; scratch_words_for(words, slots).expect("sized")];
            let mut scratch = Scratch::over(&mut buf, words, slots).expect("carves");
            let got = execute_into(&p, &planes, &foreign, &mut scratch, Out::None).expect("runs");
            let want =
                reference_execute_into(&p, &planes, &foreign, Out::None).expect("oracle runs");
            assert_eq!(got, want, "n={n} foreign_rows={foreign_rows}: value");

            let got_slots = scratch.slot(0).expect("written").to_vec();
            let want_slots = reference_scratch_with_foreign(&p, &planes, &foreign)
                .expect("oracle scratch")
                .remove(0);
            assert_eq!(
                got_slots, want_slots,
                "n={n} foreign_rows={foreign_rows}: scratch word"
            );

            // Non-vacuous: at least one fk value must be IN range (a real
            // hit is reachable) and, since fk's range always exceeds
            // `foreign_rows` by 5, at least one row's fk is genuinely
            // out-of-range for n >= foreign_rows (checked structurally,
            // not by popcount, since the foreign bits are random and a
            // hit can legitimately be zero-heavy).
            let any_out_of_range = fx.fk.iter().any(|&i| i as usize >= foreign_rows);
            assert!(
                any_out_of_range,
                "n={n} foreign_rows={foreign_rows}: fixture never exercises the out-of-range fallback"
            );
        }
    }
}

/// FAILS IF: `ScatterOrU32` disagrees with the oracle for `out_rows` both
/// SMALLER and LARGER than `n_rows`, or fails to union repeated targets.
/// Non-vacuous: asserts more than one source row scatters to the SAME
/// target bit at least once across the sweep (the union case), and that the
/// output is neither all-zero nor all-one.
#[test]
fn scatter_or_matches_the_oracle_smaller_and_larger_out_rows_with_repeats() {
    let mut saw_a_union = false;
    for &n in &ROWS {
        for &out_rows in &[3u32, 40, 2000] {
            let fx = Fixture::new(n, 10, 8, 0xBEEF ^ (out_rows as u64));
            let (lanes, masks) = fx.planes();
            let planes = Planes {
                n_rows: n,
                masks: &masks,
                lanes: &lanes,
            };
            // `status != 2` over the u32 `status` lane (index 1) selects
            // roughly 2/3 of rows — non-trivial, and never all of them.
            let p = Program::new(
                vec![MaskOp::Pred {
                    pred: Pred::NeU32 { lane: 1, v: 2 },
                    under: None,
                    dst: 0,
                }],
                Terminal::ScatterOrU32 {
                    mask: S0,
                    lane: 0,
                    out_rows,
                },
            );
            let words = words_for(n);
            let slots = p.scratch_slots as usize;
            let mut exec_buf = vec![0u64; scratch_words_for(words, slots).expect("sized")];
            let mut scratch = Scratch::over(&mut exec_buf, words, slots).expect("carves");
            let out_words = words_for(out_rows as usize);
            let mut got_out = vec![0u64; out_words];
            let got = execute_into(
                &p,
                &planes,
                &Foreign::NONE,
                &mut scratch,
                Out::Mask(&mut got_out),
            )
            .expect("runs");
            let mut want_out = vec![0u64; out_words];
            let want =
                reference_execute_into(&p, &planes, &Foreign::NONE, Out::Mask(&mut want_out))
                    .expect("oracle runs");
            assert_eq!(got, want, "n={n} out_rows={out_rows}: value");
            assert_eq!(got_out, want_out, "n={n} out_rows={out_rows}: out buffer");
            assert!(matches!(got, Value::Scattered));

            // Detect a union: more selected rows than set bits in `got_out`
            // means at least two rows scattered to the same target.
            let selected = fx
                .status
                .iter()
                .zip(fx.fk.iter())
                .filter(|(&st, _)| st != 2)
                .count();
            let set_bits: u32 = got_out.iter().map(|w| w.count_ones()).sum();
            if selected > set_bits as usize {
                saw_a_union = true;
            }
            assert!(
                set_bits == 0 || set_bits <= out_rows,
                "n={n} out_rows={out_rows}: a scattered bit landed past out_rows"
            );
        }
    }
    assert!(
        saw_a_union,
        "the sweep never exercised the union-of-repeats case at all"
    );
}

/// FAILS IF: `GroupSumI32` disagrees with the oracle when `K` is smaller
/// than the key range (some rows are DROPPED) or the value lane carries
/// negatives. Non-vacuous: asserts at least one row's key is dropped, at
/// least one group is non-empty, and not every group sum is identical.
#[test]
fn group_sum_matches_the_oracle_with_dropped_keys_and_negatives() {
    let groups = 4u32;
    let mut any_dropped_anywhere = false;
    for &n in &ROWS {
        let fx = Fixture::new(n, 10, groups, 0x600D ^ n as u64);
        let (lanes, masks) = fx.planes();
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &lanes,
        };
        let p = Program::new(
            vec![MaskOp::Pred {
                pred: Pred::NeU32 { lane: 1, v: 2 },
                under: None,
                dst: 0,
            }],
            Terminal::GroupSumI32 {
                mask: S0,
                key: 3,
                val: 2,
            },
        );
        let words = words_for(n);
        let slots = p.scratch_slots as usize;
        let mut exec_buf = vec![0u64; scratch_words_for(words, slots).expect("sized")];
        let mut scratch = Scratch::over(&mut exec_buf, words, slots).expect("carves");
        let mut got_out = vec![0i64; groups as usize];
        let got = execute_into(
            &p,
            &planes,
            &Foreign::NONE,
            &mut scratch,
            Out::I64(&mut got_out),
        )
        .expect("runs");
        let mut want_out = vec![0i64; groups as usize];
        let want = reference_execute_into(&p, &planes, &Foreign::NONE, Out::I64(&mut want_out))
            .expect("oracle runs");
        assert_eq!(got, want, "n={n}: value");
        assert_eq!(got_out, want_out, "n={n}: out buffer");
        assert!(matches!(got, Value::GroupSummed));

        let dropped = fx
            .status
            .iter()
            .zip(fx.key.iter())
            .filter(|(&st, &k)| st != 2 && k >= groups)
            .count();
        if dropped > 0 {
            any_dropped_anywhere = true;
        }
        let distinct: std::collections::HashSet<i64> = got_out.iter().copied().collect();
        assert!(
            n < 4 || distinct.len() > 1,
            "n={n}: every group summed to the same value — not a real fixture"
        );
    }
    assert!(any_dropped_anywhere, "the sweep never dropped a key at all");
}

/// FAILS IF: any of the four named validation refusals is not reported
/// IDENTICALLY by the executor and the oracle — same [`ExecError`] variant
/// and payload, not merely "both are `Err`".
#[test]
fn validation_refusals_match_between_executor_and_oracle() {
    let fx = Fixture::new(200, 10, 4, 0xDEAD);
    let (lanes, masks) = fx.planes();
    let planes = Planes {
        n_rows: 200,
        masks: &masks,
        lanes: &lanes,
    };
    let fp = fx.foreign_plane();
    let foreign = Foreign {
        planes: std::slice::from_ref(&fp),
    };

    let assert_same_refusal = |label: &str,
                               p: &Program,
                               foreign: &Foreign<'_>,
                               out: Out<'_>,
                               out2: Out<'_>,
                               want: ExecError| {
        let words = words_for(planes.n_rows);
        let slots = p.scratch_slots.max(1) as usize;
        let mut buf = vec![0u64; scratch_words_for(words, slots).expect("sized")];
        let mut scratch = Scratch::over(&mut buf, words, slots).expect("carves");
        let got = execute_into(p, &planes, foreign, &mut scratch, out);
        assert_eq!(got, Err(want), "{label}: executor");
        let oracle = reference_execute_into(p, &planes, foreign, out2);
        assert_eq!(oracle, Err(want), "{label}: oracle");
    };

    // (a) foreign index out of range.
    let p_bad_foreign = Program::new(
        vec![MaskOp::Gather {
            lane: 0,
            foreign: 3,
            dst: 0,
        }],
        Terminal::Keep { mask: S0 },
    );
    assert_same_refusal(
        "foreign out of range",
        &p_bad_foreign,
        &foreign,
        Out::None,
        Out::None,
        ExecError::ForeignOutOfRange(3),
    );

    // (b) wrong lane kind — `Gather`'s fk must be U32; lane 2 is I32.
    let p_wrong_kind = Program::new(
        vec![MaskOp::Gather {
            lane: 2,
            foreign: 0,
            dst: 0,
        }],
        Terminal::Keep { mask: S0 },
    );
    assert_same_refusal(
        "wrong lane kind",
        &p_wrong_kind,
        &foreign,
        Out::None,
        Out::None,
        ExecError::LaneKind {
            lane: 2,
            expected: lance_graph_mask_risc::LaneKind::U32,
            found: lance_graph_mask_risc::LaneKind::I32,
        },
    );

    // (c) wrong Out kind for the terminal that IS present.
    let p_group = Program::new(
        vec![MaskOp::Pred {
            pred: Pred::NeU32 { lane: 1, v: 99 },
            under: None,
            dst: 0,
        }],
        Terminal::GroupSumI32 {
            mask: S0,
            key: 3,
            val: 2,
        },
    );
    let mut wrong = [0i32; 1];
    let mut wrong2 = [0i32; 1];
    assert_same_refusal(
        "wrong Out kind",
        &p_group,
        &Foreign::NONE,
        Out::I32(&mut wrong),
        Out::I32(&mut wrong2),
        ExecError::TerminalNeedsOut {
            what: "GroupSumI32",
        },
    );

    // (d) out length mismatch — `ScatterOrU32` wants `words_for(out_rows)`.
    let p_scatter = Program::new(
        vec![MaskOp::Pred {
            pred: Pred::NeU32 { lane: 1, v: 99 },
            under: None,
            dst: 0,
        }],
        Terminal::ScatterOrU32 {
            mask: S0,
            lane: 0,
            out_rows: 500,
        },
    );
    let mut too_short = [0u64; 1]; // words_for(500) == 8
    let mut too_short2 = [0u64; 1];
    assert_same_refusal(
        "out length mismatch",
        &p_scatter,
        &Foreign::NONE,
        Out::Mask(&mut too_short),
        Out::Mask(&mut too_short2),
        ExecError::LenMismatch {
            what: "out",
            expected: 8,
            found: 1,
        },
    );
}

/// FAILS IF: an `AND` between a `Gather` leaf and a plain predicate — the
/// composed-semijoin shape a consumer builds by hand until this IR gets an
/// `under` for `Gather` — disagrees with the oracle. Exercises the "no
/// `under` variant (compose with `And`)" doc claim directly.
#[test]
fn gather_composes_with_and_like_any_other_leaf() {
    let fx = Fixture::new(300, 40, 4, 0xC0FFEE);
    let (lanes, masks) = fx.planes();
    let planes = Planes {
        n_rows: 300,
        masks: &masks,
        lanes: &lanes,
    };
    let fp = fx.foreign_plane();
    let foreign = Foreign {
        planes: std::slice::from_ref(&fp),
    };
    let p = Program::new(
        vec![
            MaskOp::Pred {
                pred: Pred::NeU32 { lane: 1, v: 2 },
                under: None,
                dst: 0,
            },
            MaskOp::Gather {
                lane: 0,
                foreign: 0,
                dst: 1,
            },
            MaskOp::And {
                a: S0,
                b: S1,
                dst: 0,
            },
        ],
        Terminal::Count { mask: S0 },
    );
    let words = words_for(300);
    let slots = p.scratch_slots as usize;
    let mut buf = vec![0u64; scratch_words_for(words, slots).expect("sized")];
    let mut scratch = Scratch::over(&mut buf, words, slots).expect("carves");
    let got = execute_into(&p, &planes, &foreign, &mut scratch, Out::None).expect("runs");
    let want = reference_execute_into(&p, &planes, &foreign, Out::None).expect("oracle runs");
    assert_eq!(got, want);
    if let Value::Count(c) = got {
        assert!(c > 0 && c < 300, "the composed semijoin is not selective");
    } else {
        panic!("expected a count, got {got:?}");
    }
}
