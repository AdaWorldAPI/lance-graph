//! Keyed partial-sink merge: the keyed `i64` folds over partial extents.
//!
//! `GroupSumI32`, `GroupSumViaI32` and `GroupReduce { Count, MinI32, MaxI32,
//! SumSymI32 }` run over any absolute extent into a FRESH, re-seeded
//! `Out::I64`, and partial sinks combine with `Terminal::merge_group_sink`,
//! which applies each fold's own law (`GroupFold::merge`).
//!
//! The gates:
//! - split composition: `full(rows) == merge(fold(part_1), …, fold(part_n))`
//!   for one, two and many uneven partitions, empty partitions, cuts inside
//!   a word, on word edges and on tile edges, in several merge orders AND
//!   groupings (the laws are associative; these folds are also commutative);
//! - the whole run equals the row-at-a-time oracle, so "full" is not just
//!   the executor agreeing with itself;
//! - the `_sym` SUM law's can-fire cases: `⊥ ⊕ x`, `x ⊕ ⊥`, `⊥ ⊕ ⊥`, and a
//!   real zero sum that must stay distinct from `⊥`;
//! - finalize once, after the merge: merging FINALIZED slots is shown to
//!   give a wrong answer, so the raw-only domain is load-bearing;
//! - refusals: no keyed `i64` sink, or two different sink lengths (a length
//!   check only — same length is not the same destination universe).

use lance_graph_mask_risc::exec::{execute_extent, Scratch};
use lance_graph_mask_risc::{
    reference_execute_into, scratch_words_for, ExecError, Foreign, GroupFold, GroupKey, LaneRef,
    Operand, Out, Planes, Program, Terminal, Value,
};

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

fn plane(n: usize, set: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut w = vec![0u64; n.div_ceil(64)];
    for r in (0..n).filter(|&r| set(r)) {
        w[r / 64] |= 1 << (r % 64);
    }
    w
}

fn bit(p: &[u64], r: usize) -> bool {
    p[r / 64] >> (r % 64) & 1 == 1
}

/// The identity a fresh accumulator starts from: what the executor seeds.
fn identity(t: &Terminal) -> i64 {
    match *t {
        Terminal::GroupReduce { fold, .. } => fold.seed(),
        _ => 0,
    }
}

/// `t` over one extent, into a FRESH sink the caller owns. The sink starts
/// dirty on purpose: the terminal must seed it, never add to stale state.
fn part(
    t: &Terminal,
    planes: &Planes<'_>,
    foreign: &Foreign<'_>,
    tile_words: usize,
    groups: usize,
    ext: std::ops::Range<usize>,
) -> Vec<i64> {
    let p = Program::new(vec![], *t);
    let slots = p.scratch_slots as usize;
    let mut buf = vec![0u64; scratch_words_for(tile_words, slots).expect("sized")];
    let mut s = Scratch::over(&mut buf, tile_words, slots).expect("carves");
    let mut sink = vec![0x5A5A_5A5A_i64; groups];
    let v = execute_extent(&p, planes, foreign, &mut s, Out::I64(&mut sink), ext)
        .expect("a partial extent is admitted");
    assert!(matches!(v, Value::GroupSummed | Value::GroupReduced));
    sink
}

/// Merge `parts` in `order`, left to right, from the identity.
fn merge_linear(t: &Terminal, parts: &[Vec<i64>], order: &[usize]) -> Vec<i64> {
    let mut acc = vec![identity(t); parts[0].len()];
    for &i in order {
        t.merge_group_sink(&mut acc, &parts[i])
            .expect("same terminal, same K");
    }
    acc
}

/// Merge `parts` as a balanced tree — a different GROUPING of the same
/// operands, which only an associative law survives.
fn merge_tree(t: &Terminal, parts: &[Vec<i64>]) -> Vec<i64> {
    match parts.len() {
        1 => parts[0].clone(),
        k => {
            let (l, r) = parts.split_at(k / 2);
            let mut acc = merge_tree(t, l);
            t.merge_group_sink(&mut acc, &merge_tree(t, r))
                .expect("same terminal, same K");
            acc
        }
    }
}

/// FAILS IF: a keyed i64 terminal over a partial extent counts a row outside
/// it (an unclipped edge word), re-counts a row two extents both claim, adds
/// to a stale sink instead of re-seeding it, or its partials do not merge
/// back to the whole under the fold's law — in any order or grouping, for
/// every fold and key address, at two tile widths.
///
/// Anti-vacuity (asserted at the end): some group must be non-empty in two
/// partials (else the merge never combines), some group must be non-empty in
/// exactly one partial of a multi-part split (else "absent on one side" is
/// never exercised), some group must be empty in the whole (else the seed is
/// never merged), some partition must contain an empty extent, and some cut
/// must split a 64-row word with selected rows on both sides.
#[test]
fn keyed_i64_partials_merge_to_the_whole() {
    let mut spans_two = false;
    let mut only_one = false;
    let mut empty_group = false;
    let mut empty_extent = false;
    let mut split_live_word = false;
    for n in [1317usize, 4096 + 37] {
        let mut seed = 0x6E_E0 ^ n as u64;
        let sel: Vec<bool> = (0..n).map(|_| !lcg(&mut seed).is_multiple_of(3)).collect();
        let pl = plane(n, |r| sel[r]);
        // Keys 0..=5 with 6 = past the universe (dropped). Key 4 only occurs
        // in the first 90 rows, so most splits see it in one extent only;
        // group 6 of a K = 7 universe is never named, so it stays empty.
        let key: Vec<u32> = (0..n)
            .map(|r| {
                let k = (lcg(&mut seed) % 7) as u32;
                if k == 4 && r >= 90 {
                    0
                } else {
                    k
                }
            })
            .collect();
        let val: Vec<i32> = (0..n)
            .map(|i| match i % 17 {
                3 => i32::MIN,
                9 => i32::MAX,
                _ => (lcg(&mut seed) % 200_001) as i32 - 100_000,
            })
            .collect();
        let fk: Vec<u32> = (0..n).map(|_| (lcg(&mut seed) % 8) as u32).collect(); // 7 = past table
        let hi: Vec<u32> = (0..n).map(|_| (lcg(&mut seed) % 3) as u32).collect();
        let lo: Vec<u32> = (0..n).map(|_| (lcg(&mut seed) % 5) as u32).collect(); // 4 = >= stride
        let table: Vec<u32> = vec![0, 4, 2, 9, 1, 3, 5]; // 9 = second-hop drop
        let masks: [&[u64]; 1] = [&pl];
        let lanes = [
            LaneRef::U32(&key),
            LaneRef::I32(&val),
            LaneRef::U32(&fk),
            LaneRef::U32(&hi),
            LaneRef::U32(&lo),
        ];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &lanes,
        };
        let flanes = [LaneRef::U32(&table)];
        let foreign = Foreign {
            planes: &[],
            lanes: &flanes,
        };

        // One partition; two-way cuts inside a word, on word edges (64) and
        // on the 2-word tile edge (128); cuts at 0 / n leave an EMPTY
        // extent; many uneven random partitions, duplicates included (also
        // empty extents).
        let mut partitions: Vec<Vec<usize>> = vec![vec![0, n]];
        for k in [0, 1, 63, 64, 65, 127, 128, 129, n / 2, n - 1, n] {
            partitions.push(vec![0, k, n]);
        }
        for _ in 0..12 {
            let mut cuts: Vec<usize> = (0..2 + lcg(&mut seed) % 6)
                .map(|_| (lcg(&mut seed) as usize) % (n + 1))
                .collect();
            cuts.push(0);
            cuts.push(n);
            cuts.sort_unstable();
            partitions.push(cuts);
        }
        for cuts in &partitions {
            empty_extent |= cuts.windows(2).any(|w| w[0] == w[1]);
            split_live_word |= cuts[1..cuts.len() - 1].iter().any(|&c| {
                c % 64 != 0
                    && (c - c % 64..c).any(|r| bit(&pl, r))
                    && (c..(c - c % 64 + 64).min(n)).any(|r| bit(&pl, r))
            });
        }

        let mask = Operand::Plane(0);
        let addrs = [
            (GroupKey::Lane(0), 7usize),
            (GroupKey::Via { fk: 2, key: 0 }, 7),
            (
                GroupKey::Pair {
                    hi: 3,
                    lo: 4,
                    stride: 4,
                },
                13,
            ),
        ];
        let mut terminals: Vec<(Terminal, usize)> = vec![
            (
                Terminal::GroupSumI32 {
                    mask,
                    key: 0,
                    val: 1,
                },
                7,
            ),
            (
                Terminal::GroupSumViaI32 {
                    mask,
                    fk: 2,
                    key: 0,
                    val: 1,
                },
                7,
            ),
        ];
        for (gk, groups) in addrs {
            for fold in [
                GroupFold::Count,
                GroupFold::MinI32(1),
                GroupFold::MaxI32(1),
                GroupFold::SumSymI32(1),
            ] {
                terminals.push((
                    Terminal::GroupReduce {
                        mask,
                        key: gk,
                        fold,
                    },
                    groups,
                ));
            }
        }

        let words = n.div_ceil(64);
        for (t, groups) in &terminals {
            let (t, groups) = (*t, *groups);
            // The oracle knows nothing about tiles, extents or merges.
            let mut want = vec![0i64; groups];
            reference_execute_into(
                &Program::new(vec![], t),
                &planes,
                &foreign,
                Out::I64(&mut want),
            )
            .expect("oracle");
            empty_group |= matches!(t, Terminal::GroupReduce { .. })
                && want.iter().any(|&v| v == identity(&t));
            for tile_words in [2usize, words] {
                let whole = part(&t, &planes, &foreign, tile_words, groups, 0..n);
                assert_eq!(whole, want, "{t:?} tile={tile_words}: whole run vs oracle");
                for cuts in &partitions {
                    let spans: Vec<_> = cuts.windows(2).map(|w| w[0]..w[1]).collect();
                    let parts: Vec<Vec<i64>> = spans
                        .iter()
                        .map(|e| part(&t, &planes, &foreign, tile_words, groups, e.clone()))
                        .collect();
                    let id = identity(&t);
                    for g in 0..groups {
                        let live = parts.iter().filter(|p| p[g] != id && p[g] != 0).count();
                        spans_two |= live > 1;
                        only_one |= live == 1 && parts.len() > 1;
                    }
                    let k = parts.len();
                    let mut shuffled: Vec<usize> = (0..k).collect();
                    for i in (1..k).rev() {
                        shuffled.swap(i, (lcg(&mut seed) as usize) % (i + 1));
                    }
                    for order in [
                        (0..k).collect::<Vec<_>>(),
                        (0..k).rev().collect(),
                        (0..k).map(|i| (i + 1) % k).collect(),
                        shuffled,
                    ] {
                        let at =
                            format!("n={n} {t:?} tile={tile_words} cuts={cuts:?} order={order:?}");
                        assert_eq!(merge_linear(&t, &parts, &order), whole, "linear {at}");
                    }
                    assert_eq!(
                        merge_tree(&t, &parts),
                        whole,
                        "tree n={n} {t:?} tile={tile_words} cuts={cuts:?}"
                    );
                }
            }
        }
    }
    assert!(spans_two, "some group must be live in two partials");
    assert!(only_one, "some group must be live in exactly one partial");
    assert!(empty_group, "some group must be empty in the whole");
    assert!(empty_extent, "some partition must contain an empty extent");
    assert!(split_live_word, "some cut must split a live 64-row word");
}

/// FAILS IF: the `_sym` SUM merges with plain `+`, reads its empty marker as
/// `0`, or lets a real zero sum collapse into "empty". Each case is built so
/// it CAN fire: a group present on one side only, a group empty on both, and
/// a group whose two partials cancel to a present `0`.
#[test]
fn sym_sum_merge_law_can_fire() {
    let f = GroupFold::SumSymI32(0);
    let empty = f.seed();
    assert_eq!(f.merge(empty, 5), 5, "⊥ ⊕ x = x");
    assert_eq!(f.merge(5, empty), 5, "x ⊕ ⊥ = x");
    assert_eq!(f.merge(empty, empty), empty, "⊥ ⊕ ⊥ = ⊥");
    assert_eq!(f.merge(7, -7), 0, "a real zero sum");
    assert!(!f.is_empty_slot(f.merge(7, -7)), "a real zero is NOT empty");
    // What the forbidden `+` would have produced, so the law is not vacuous:
    assert_ne!(empty.wrapping_add(5), 5);
    assert_ne!(empty.wrapping_add(empty), empty);

    // The same cases through execution: rows 0..64 and 64..128 are two
    // extents. Group 0: +7 in A, −7 in B → present 0. Group 1: only in A.
    // Group 2: only in B. Group 3: never named → empty.
    let n = 128;
    let key: Vec<u32> = (0..n as u32)
        .map(|r| match r {
            0 | 64 => 0,
            1..=5 => 1,
            70..=72 => 2,
            _ => 9, // past the universe
        })
        .collect();
    let val: Vec<i32> = (0..n as i32)
        .map(|r| match r {
            0 => 7,
            64 => -7,
            _ => r,
        })
        .collect();
    let pl = plane(n, |_| true);
    let masks: [&[u64]; 1] = [&pl];
    let lanes = [LaneRef::U32(&key), LaneRef::I32(&val)];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };
    let t = Terminal::GroupReduce {
        mask: Operand::Plane(0),
        key: GroupKey::Lane(0),
        fold: GroupFold::SumSymI32(1),
    };
    let a = part(&t, &planes, &Foreign::NONE, 1, 4, 0..64);
    let b = part(&t, &planes, &Foreign::NONE, 1, 4, 64..n);
    assert_eq!(a, vec![7, 15, empty, empty]);
    assert_eq!(b, vec![-7, empty, 213, empty]);
    let mut acc = a.clone();
    t.merge_group_sink(&mut acc, &b).expect("merge");
    let whole = part(&t, &planes, &Foreign::NONE, 1, 4, 0..n);
    assert_eq!(acc, whole);
    assert_eq!(acc, vec![0, 15, 213, empty]);
    assert!(!GroupFold::SumSymI32(1).is_empty_slot(acc[0]));
}

/// Finalize the way a consumer does (cf. quack's `normalize_group_sink`):
/// an empty slot becomes `None` (SQL `NULL`), everything else its value.
fn finalize(fold: GroupFold, slot: i64) -> Option<i64> {
    (!fold.is_empty_slot(slot)).then_some(slot)
}

/// FAILS IF: finalize-after-merge is wrong, OR merging finalized slots
/// happens to be right — the second half pins that the raw-only domain of
/// `merge` is load-bearing. A consumer that coalesces `NULL` to `0` before
/// merging gets `MIN(0, 5) = 0` for a group whose true minimum is `5`.
#[test]
fn finalize_once_after_merge_never_merge_finalized() {
    let n = 128;
    let key: Vec<u32> = (0..n as u32)
        .map(|r| if r == 100 { 0 } else { 9 })
        .collect();
    let val: Vec<i32> = vec![5; n];
    let pl = plane(n, |_| true);
    let masks: [&[u64]; 1] = [&pl];
    let lanes = [LaneRef::U32(&key), LaneRef::I32(&val)];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };
    for fold in [GroupFold::MinI32(1), GroupFold::SumSymI32(1)] {
        let t = Terminal::GroupReduce {
            mask: Operand::Plane(0),
            key: GroupKey::Lane(0),
            fold,
        };
        let a = part(&t, &planes, &Foreign::NONE, 1, 1, 0..64); // group 0 empty here
        let b = part(&t, &planes, &Foreign::NONE, 1, 1, 64..n); // value 5 here
        let mut raw = a.clone();
        t.merge_group_sink(&mut raw, &b).expect("merge");
        assert_eq!(
            finalize(fold, raw[0]),
            Some(5),
            "{fold:?}: finalize(merge(raw))"
        );

        // The wrong order: finalize each partial, coalesce NULL to 0, merge.
        let coalesce =
            |s: &[i64]| -> Vec<i64> { s.iter().map(|&v| finalize(fold, v).unwrap_or(0)).collect() };
        let mut fin = coalesce(&a);
        t.merge_group_sink(&mut fin, &coalesce(&b)).expect("merge");
        if matches!(fold, GroupFold::MinI32(_)) {
            assert_eq!(
                fin[0], 0,
                "merge(finalized) is WRONG for MIN, not merely different"
            );
            assert_ne!(finalize(fold, fin[0]), Some(5));
        }
    }
}

/// FAILS IF: the merge accepts a terminal with no keyed i64 sink, or two
/// sinks of different LENGTHS — or writes before refusing. This is a length
/// check only: same length != same destination universe, and nothing here
/// can tell (TD-KEYED-SINK-MERGE-IDENTITY-1).
#[test]
fn merge_refuses_unmergeable_terminals_and_mismatched_sink_lengths() {
    let mut acc = vec![1i64, 2, 3];
    let count = Terminal::Count {
        mask: Operand::Plane(0),
    };
    assert_eq!(
        count.merge_group_sink(&mut acc, &[1, 1, 1]),
        Err(ExecError::ExtentUnsupported {
            what: "merge_group_sink"
        })
    );
    let sum = Terminal::GroupSumI32 {
        mask: Operand::Plane(0),
        key: 0,
        val: 1,
    };
    assert_eq!(
        sum.merge_group_sink(&mut acc, &[1, 1]),
        Err(ExecError::LenMismatch {
            what: "merge_group_sink",
            expected: 3,
            found: 2
        })
    );
    assert_eq!(acc, vec![1, 2, 3], "a refusal writes nothing");
    assert_eq!(sum.merge_group_sink(&mut acc, &[1, 1, 1]), Ok(()));
    assert_eq!(acc, vec![2, 3, 4]);
}

/// FAILS IF: the `_sym` law were associative even where a present total
/// wraps onto ⊥ — i.e. if the row bound were NOT what makes the merge
/// lawful. Two present partials of `i64::MIN + 1` plus `-1`: grouped one way
/// the inner sum is exactly `⊥` and absorbs, grouped the other it is not.
/// Real partials cannot produce these totals: their rows stay within
/// `GROUP_SUM_SYM_MAX_ROWS`, which is why that bound is stated over the
/// TOTAL rows of all merged partials, never per partial.
#[test]
fn sym_sum_bound_is_load_bearing() {
    let f = GroupFold::SumSymI32(0);
    let (a, b, c) = (i64::MIN + 1, i64::MIN + 1, -1);
    assert_eq!(f.merge(b, c), f.seed(), "a present total landed on ⊥");
    assert_ne!(f.merge(f.merge(a, b), c), f.merge(a, f.merge(b, c)));
}

/// FAILS IF: a fold's merge is not a monoid with its seed as identity —
/// checked directly on extreme values, not only through execution.
/// Commutativity is checked too; it holds for THESE folds, and is a property
/// of each fold, not of the merge contract (an ordered-segment fold such as
/// a key-run carry may be associative without it).
#[test]
fn fold_merge_laws_hold_on_extremes() {
    let xs = [
        i64::MIN + 1,
        -1_000_000_007,
        -1,
        0,
        1,
        i64::from(i32::MAX),
        i64::from(i32::MIN),
        i64::MAX - 1,
    ];
    for fold in [
        GroupFold::Count,
        GroupFold::MinI32(0),
        GroupFold::MaxI32(0),
        GroupFold::SumSymI32(0),
    ] {
        // The `_sym` SUM is a monoid only while no PRESENT total reaches ⊥.
        // `GROUP_SUM_SYM_MAX_ROWS` guarantees that for real partials: every
        // total of at most 2^32 − 1 rows of i32 lies in ±(2^63 − 1). Here the
        // domain is |x| ≤ 2^61, so no sum of three operands can reach ⊥.
        // Outside it the law breaks; `sym_sum_bound_is_load_bearing` pins how.
        let in_domain = |x: i64| match fold {
            GroupFold::SumSymI32(_) => x.unsigned_abs() <= 1 << 61,
            _ => true,
        };
        let mut domain: Vec<i64> = xs.iter().copied().filter(|&x| in_domain(x)).collect();
        domain.push(fold.seed());
        for &a in &domain {
            assert_eq!(
                fold.merge(fold.seed(), a),
                a,
                "{fold:?}: left identity at {a}"
            );
            assert_eq!(
                fold.merge(a, fold.seed()),
                a,
                "{fold:?}: right identity at {a}"
            );
            for &b in &domain {
                assert_eq!(
                    fold.merge(a, b),
                    fold.merge(b, a),
                    "{fold:?}: commutes {a} {b}"
                );
                for &c in &domain {
                    assert_eq!(
                        fold.merge(fold.merge(a, b), c),
                        fold.merge(a, fold.merge(b, c)),
                        "{fold:?}: associates {a} {b} {c}"
                    );
                }
            }
        }
    }
}
