//! Crate-level tests for the extracted [`FoldDialect`].
//!
//! The reference tests, disable falsifiers and allocation measurements stay
//! in `r2il-mask-abi-probe/tests/row_bridge.rs`, which now runs against this
//! crate. These tests pin what the extraction itself must keep: the R2IL-band
//! ordinals still name the opcodes `ogar_r2il` declares, every named refusal
//! is still produced, and a fold still answers against a scalar reference.

use lance_graph_mask_risc::{words_for, Foreign, LaneRef, Planes};
use lance_graph_r2il::{
    Addr, FoldDialect, FoldError, Val, INT_ADD, INT_AND, INT_EQUAL, INT_LESS, INT_LESS_EQUAL,
    INT_MULT, INT_NOT, INT_OR, INT_SUB, INT_S_DIV, INT_S_LESS, INT_XOR, LOAD, MAX_SLOTS, POP_COUNT,
};
use ogar_loco::FnIndex;
use ogar_r2il::{
    R2ILFn, ALL, ANY, BLEND, GROUP_SUM, KEEP, KEY_RUNS, R2IL_BASE, RANGE, SCATTER_OR, SUM, VIA,
};

/// The R2IL-band ordinal of `f`, as an index into `R2ILFn::MNEMONICS`.
fn mnemonic(f: FnIndex) -> &'static str {
    R2ILFn::MNEMONICS[usize::from(f.0 - R2IL_BASE)]
}

/// FAILS IF: a locally spelled R2IL-band ordinal stops naming the opcode
/// `ogar_r2il` declares at that position. `ogar_r2il` stays the only
/// authority; this crate only addresses into it.
#[test]
fn local_r2il_ordinals_name_the_ogar_r2il_opcodes() {
    let pinned = [
        (LOAD, "Load"),
        (INT_SUB, "IntSub"),
        (INT_AND, "IntAnd"),
        (INT_OR, "IntOr"),
        (INT_XOR, "IntXor"),
        (INT_ADD, "IntAdd"),
        (INT_MULT, "IntMult"),
        (INT_S_DIV, "IntSDiv"),
        (INT_NOT, "IntNot"),
        (INT_EQUAL, "IntEqual"),
        (INT_LESS, "IntLess"),
        (INT_S_LESS, "IntSLess"),
        (INT_LESS_EQUAL, "IntLessEqual"),
        (POP_COUNT, "PopCount"),
    ];
    for (f, name) in pinned {
        assert_eq!(mnemonic(f), name, "{f:?}");
    }
}

const N: usize = 1_000;

/// One I32 lane holding `i % 50`.
fn lane() -> Vec<i32> {
    (0..N).map(|i| (i % 50) as i32).collect()
}

fn call(
    d: &mut FoldDialect<'_>,
    f: FnIndex,
    v: [u8; 3],
    stack: &mut Vec<Val>,
) -> Result<(), FoldError> {
    d.call(f, v, stack)
}

fn i32_lane(idx: u16) -> Val {
    Val::Address(Addr::Lane {
        idx,
        kind: lance_graph_mask_risc::LaneKind::I32,
    })
}

/// FAILS IF: any byte the crate doc lists as refused stops being refused,
/// or is refused under a different name.
#[test]
fn every_named_refusal_is_still_produced() {
    let values = lane();
    let lanes = [LaneRef::I32(&values)];
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign::NONE;

    // Fold-band bytes that are not wired, and one R2IL-band one. `IntOr` is
    // wired since D-SQLP-2; `IntXor`, `ANY`, `ALL`, `KEY_RUNS`, `IntAdd`,
    // `IntMult` and `IntSDiv` since D-RFT-1.
    let int_rem = FnIndex(R2IL_BASE + 14);
    assert_eq!(mnemonic(int_rem), "IntRem");
    for f in [SCATTER_OR, BLEND, int_rem] {
        let mut d = FoldDialect::new(&planes, &foreign);
        let mut stack = vec![Val::Scalar(1), Val::Scalar(2)];
        assert_eq!(
            call(&mut d, f, [0; 3], &mut stack),
            Err(FoldError::Unimplemented(f))
        );
    }

    for f in [INT_LESS, INT_LESS_EQUAL] {
        let mut d = FoldDialect::new(&planes, &foreign);
        let mut stack = vec![i32_lane(0), Val::Scalar(3)];
        assert_eq!(
            call(&mut d, f, [0; 3], &mut stack),
            Err(FoldError::NoUnsignedLaneCompare(f))
        );
    }

    let mut d = FoldDialect::new(&planes, &foreign);
    let mut stack = vec![i32_lane(0), i32_lane(0)];
    assert_eq!(
        call(&mut d, INT_SUB, [0; 3], &mut stack),
        Err(FoldError::LaneVersusLane(INT_SUB))
    );

    let mut d = FoldDialect::new(&planes, &foreign);
    let mut stack = vec![Val::Scalar(0)];
    assert_eq!(
        call(&mut d, LOAD, [4, 0, 0], &mut stack),
        Err(FoldError::UnknownLoadSpace(4))
    );

    // KEEP on a dialect that was lent no sink.
    let mut d = FoldDialect::new(&planes, &foreign);
    let mut stack = vec![];
    call(&mut d, RANGE, [0, 10, 0], &mut stack).expect("RANGE emits a predicate");
    assert_eq!(
        call(&mut d, KEEP, [0; 3], &mut stack),
        Err(FoldError::NoKeepSink)
    );

    // A slot from an earlier fold is refused once a fold has run.
    let mut d = FoldDialect::new(&planes, &foreign);
    let mut stack = vec![];
    call(&mut d, RANGE, [0, 10, 0], &mut stack).expect("RANGE");
    let stale = stack[0];
    call(&mut d, POP_COUNT, [0; 3], &mut stack).expect("POP_COUNT folds");
    let mut stack = vec![stale];
    assert_eq!(
        call(&mut d, POP_COUNT, [0; 3], &mut stack),
        Err(FoldError::StaleSlot)
    );

    // ANY, ALL and IntXor are wired: a scalar where a mask belongs is a shape
    // refusal, not Unimplemented.
    for f in [ANY, ALL] {
        let mut d = FoldDialect::new(&planes, &foreign);
        let mut stack = vec![Val::Scalar(1)];
        assert_eq!(
            call(&mut d, f, [0; 3], &mut stack),
            Err(FoldError::WrongOperandKind(f))
        );
    }
    let mut d = FoldDialect::new(&planes, &foreign);
    let mut stack = vec![i32_lane(0), Val::Scalar(1)];
    assert_eq!(
        call(&mut d, INT_XOR, [0; 3], &mut stack),
        Err(FoldError::WrongOperandKind(INT_XOR))
    );

    // Scalar arithmetic: checked, never wrapped; a lane has no scalar reading.
    for (f, a, b, want) in [
        (INT_S_DIV, 7, 0, FoldError::DivisionByZero(INT_S_DIV)),
        (INT_S_DIV, i64::MIN, -1, FoldError::Overflow(INT_S_DIV)),
        (INT_ADD, i64::MAX, 1, FoldError::Overflow(INT_ADD)),
        (INT_MULT, i64::MAX, 2, FoldError::Overflow(INT_MULT)),
        (INT_SUB, -2, i64::MAX, FoldError::Overflow(INT_SUB)),
        (INT_SUB, i64::MIN, 1, FoldError::Overflow(INT_SUB)),
    ] {
        let mut d = FoldDialect::new(&planes, &foreign);
        let mut stack = vec![Val::Scalar(a), Val::Scalar(b)];
        assert_eq!(call(&mut d, f, [0; 3], &mut stack), Err(want), "{f:?}");
    }
    for f in [INT_ADD, INT_MULT, INT_S_DIV] {
        let mut d = FoldDialect::new(&planes, &foreign);
        let mut stack = vec![i32_lane(0), i32_lane(0)];
        assert_eq!(
            call(&mut d, f, [0; 3], &mut stack),
            Err(FoldError::LaneVersusLane(f))
        );
        let mut d = FoldDialect::new(&planes, &foreign);
        let mut stack = vec![i32_lane(0), Val::Scalar(1)];
        assert_eq!(
            call(&mut d, f, [0; 3], &mut stack),
            Err(FoldError::WrongOperandKind(f))
        );
    }
    // KEY_RUNS takes a mask and a U32 key lane; an I32 lane is a shape refusal.
    let mut d = FoldDialect::new(&planes, &foreign);
    let mut stack = vec![];
    call(&mut d, RANGE, [0, 10, 0], &mut stack).expect("RANGE");
    stack.push(i32_lane(0));
    assert_eq!(
        call(&mut d, KEY_RUNS, [0; 3], &mut stack),
        Err(FoldError::WrongOperandKind(KEY_RUNS))
    );

    // VIA and GROUP_SUM are wired: a wrong operand kind is a shape refusal,
    // not Unimplemented.
    let mut d = FoldDialect::new(&planes, &foreign);
    let mut stack = vec![Val::Scalar(0), i32_lane(0)];
    assert_eq!(
        call(&mut d, VIA, [0; 3], &mut stack),
        Err(FoldError::WrongOperandKind(VIA))
    );
    let mut d = FoldDialect::new(&planes, &foreign);
    let mut stack = vec![Val::Scalar(0), Val::Scalar(0), Val::Scalar(0)];
    assert_eq!(
        call(&mut d, GROUP_SUM, [0; 3], &mut stack),
        Err(FoldError::WrongOperandKind(GROUP_SUM))
    );
}

/// FAILS IF: branching on a population stops being recorded, or branching
/// on a scalar starts being recorded.
#[test]
fn branching_on_a_population_poisons_and_a_scalar_does_not() {
    let values = lane();
    let lanes = [LaneRef::I32(&values)];
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign::NONE;

    let d = FoldDialect::new(&planes, &foreign);
    assert!(d.truthy(&Val::Scalar(7)));
    assert!(!d.truthy(&Val::Scalar(0)));
    assert_eq!(d.poison(), None);

    let d = FoldDialect::new(&planes, &foreign);
    assert!(!d.truthy(&Val::Slot { slot: 0, epoch: 0 }));
    assert_eq!(d.poison(), Some(FoldError::BranchOnPopulation));
}

/// FAILS IF: `SUM(x | x < t)` stops matching a scalar reference, runs more
/// than one fold, or emits more than one physical pass.
#[test]
fn a_masked_sum_matches_the_scalar_reference() {
    let values = lane();
    let lanes = [LaneRef::I32(&values)];
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign::NONE;
    let t = 17;
    let want: i64 = values
        .iter()
        .filter(|&&x| x < t)
        .map(|&x| i64::from(x))
        .sum();
    assert!(want > 0, "the fixture must select something");

    let mut d = FoldDialect::new(&planes, &foreign);
    let mut stack = vec![Val::Scalar(0)];
    call(&mut d, LOAD, [1, 0, 0], &mut stack).expect("LOAD I32 lane 0");
    stack.push(Val::Scalar(i64::from(t)));
    call(&mut d, INT_S_LESS, [0; 3], &mut stack).expect("x < t");
    stack.push(Val::Scalar(0));
    call(&mut d, LOAD, [1, 0, 0], &mut stack).expect("LOAD I32 lane 0");
    call(&mut d, SUM, [0; 3], &mut stack).expect("SUM folds");

    assert_eq!(stack, vec![Val::Scalar(want)]);
    assert_eq!(d.programs_run(), 1);
    assert_eq!(d.facade_passes(), 1);
    assert!(d.last_program().is_some());
}

/// FAILS IF: `KEEP` stops writing the selected rows into the caller's sink.
#[test]
fn keep_writes_the_selection_into_the_lent_sink() {
    let values = lane();
    let lanes = [LaneRef::I32(&values)];
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign::NONE;
    let mut sink = vec![0u64; words_for(N)];
    {
        let mut d = FoldDialect::with_keep(&planes, &foreign, &mut sink);
        let mut stack = vec![];
        call(&mut d, RANGE, [3, 70, 0], &mut stack).expect("RANGE");
        call(&mut d, KEEP, [0; 3], &mut stack).expect("KEEP folds");
        assert!(stack.is_empty(), "KEEP pushes nothing");
    }
    let kept: Vec<usize> = (0..N)
        .filter(|&r| sink[r / 64] >> (r % 64) & 1 == 1)
        .collect();
    assert_eq!(kept, (3..70).collect::<Vec<_>>());
}

/// FAILS IF: the cross-moment fold stops matching a row loop per group, folds
/// rows outside the mask, or drops a group.
#[test]
fn cross_power_sums_match_the_row_loop_per_group() {
    use lance_graph_mask_risc::CrossPowerSums;
    let xs = lane();
    let ys: Vec<i32> = (0..N).map(|i| (i % 7) as i32 - 3).collect();
    let keys: Vec<u32> = (0..N).map(|i| (i % 3) as u32).collect();
    let lanes = [LaneRef::I32(&xs), LaneRef::I32(&ys), LaneRef::U32(&keys)];
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign::NONE;
    let mut sink = [CrossPowerSums::default(); 3];
    {
        let mut d = FoldDialect::with_cross_sums(&planes, &foreign, &mut sink);
        let mut stack = vec![];
        call(&mut d, RANGE, [40, 200, 0], &mut stack).expect("RANGE");
        d.fold_cross_power_sums(2, 0, 1, &mut stack).expect("folds");
        assert!(stack.is_empty(), "the sink is the result");
        assert_eq!(d.programs_run(), 1);
    }
    for (g, got) in sink.iter().enumerate() {
        let rows = (40..200).filter(|&i| keys[i] as usize == g);
        let (mut n, mut sx, mut sy, mut sxy) = (0u64, 0i64, 0i64, 0i128);
        for i in rows {
            n += 1;
            sx += i64::from(xs[i]);
            sy += i64::from(ys[i]);
            sxy += i128::from(xs[i]) * i128::from(ys[i]);
        }
        assert_eq!(
            (got.n, got.sum_x, got.sum_y, got.sum_xy),
            (n, sx, sy, sxy),
            "group {g}"
        );
        assert!(n > 0);
    }
    // A scalar where the mask belongs is refused, not folded.
    let mut sink = [CrossPowerSums::default(); 3];
    let mut d = FoldDialect::with_cross_sums(&planes, &foreign, &mut sink);
    assert_eq!(
        d.fold_cross_power_sums(2, 0, 1, &mut vec![Val::Scalar(1)]),
        Err(FoldError::WrongOperandKind(GROUP_SUM))
    );
}

/// FAILS IF: the scratch buffer starts growing with the row count. It is
/// sized by the tile, so 1,000 and 1,048,576 rows get the same buffer once
/// both exceed one tile's worth only through the tile cap.
#[test]
fn scratch_is_sized_by_the_tile_not_the_rows() {
    let small: Vec<i32> = vec![0; 1_000];
    let large: Vec<i32> = vec![0; 1 << 20];
    let foreign = Foreign::NONE;
    let lanes_s = [LaneRef::I32(&small)];
    let lanes_l = [LaneRef::I32(&large)];
    let ps = Planes {
        n_rows: small.len(),
        masks: &[],
        lanes: &lanes_s,
    };
    let pl = Planes {
        n_rows: large.len(),
        masks: &[],
        lanes: &lanes_l,
    };
    let s = FoldDialect::new(&ps, &foreign).scratch_len();
    let l = FoldDialect::new(&pl, &foreign).scratch_len();
    let cap = lance_graph_mask_risc::TILE_WORDS * MAX_SLOTS;
    assert!(
        l <= cap + MAX_SLOTS.div_ceil(64) + 1,
        "large = {l}, cap = {cap}"
    );
    assert!(
        l < large.len() / 64,
        "scratch must stay below one word per 64 rows"
    );
    assert!(s <= l);
}

/// FAILS IF: a fold wider than `MAX_SLOTS` stops being refused (or is
/// silently truncated), or a fold of exactly `MAX_SLOTS` stops running.
#[test]
fn a_fold_wider_than_max_slots_is_refused_not_truncated() {
    let values = lane();
    let lanes = [LaneRef::I32(&values)];
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign::NONE;

    // `width` RANGE predicates, each its own slot, then count the last one.
    let fold = |width: usize| {
        let mut d = FoldDialect::new(&planes, &foreign);
        let mut stack = vec![];
        for i in 0..width {
            call(&mut d, RANGE, [i as u8, 100, 0], &mut stack).expect("RANGE");
        }
        let r = call(&mut d, POP_COUNT, [0; 3], &mut stack);
        (r, stack)
    };

    let (ok, stack) = fold(MAX_SLOTS);
    assert_eq!(ok, Ok(()), "a fold of exactly MAX_SLOTS slots runs");
    let last = (MAX_SLOTS - 1) as i64;
    assert_eq!(stack.last(), Some(&Val::Scalar(100 - last)));

    let (wide, _) = fold(MAX_SLOTS + 1);
    assert!(
        matches!(
            wide,
            Err(FoldError::Exec(
                lance_graph_mask_risc::ExecError::ScratchBufferTooSmall { .. }
            ))
        ),
        "a fold wider than MAX_SLOTS must be refused, got {wide:?}"
    );
}

/// FAILS IF: `call_owned` stops releasing the slots an `IntOr` / `IntNot`
/// consumed (so a chain of `k` ranges needs `2k - 1` slots again), or if plain
/// `call` starts releasing them too. Plain `call` must keep the old numbering:
/// its caller may hold copies of a `Val::Slot`, so it is refused past
/// `MAX_SLOTS` exactly as before.
#[test]
fn call_owned_reuses_consumed_slots_and_call_does_not() {
    let values = lane();
    let lanes = [LaneRef::I32(&values)];
    let planes = Planes {
        n_rows: N,
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign::NONE;
    let width = 4 * MAX_SLOTS;

    // `[0,1) | [1,2) | … | [width-1, width)`, negated twice: `width` rows.
    let run = |owned: bool| {
        let mut d = FoldDialect::new(&planes, &foreign);
        let mut stack = vec![];
        let op = |d: &mut FoldDialect<'_>, f: FnIndex, v: [u8; 3], s: &mut Vec<Val>| {
            if owned {
                d.call_owned(f, v, s)
            } else {
                call(d, f, v, s)
            }
        };
        for i in 0..width {
            op(&mut d, RANGE, [i as u8, i as u8 + 1, 0], &mut stack).expect("RANGE");
            if i > 0 {
                op(&mut d, INT_OR, [0; 3], &mut stack).expect("OR");
            }
        }
        op(&mut d, INT_NOT, [0; 3], &mut stack).expect("NOT");
        op(&mut d, INT_NOT, [0; 3], &mut stack).expect("NOT");
        let r = call(&mut d, POP_COUNT, [0; 3], &mut stack);
        let slots = d.last_program().map(|p| p.scratch_slots);
        (r, stack.last().copied(), slots)
    };

    let (r, top, slots) = run(true);
    assert_eq!(r, Ok(()));
    assert_eq!(top, Some(Val::Scalar(width as i64)));
    assert_eq!(
        slots,
        Some(3),
        "an owned chain needs the accumulator, one atom and one result"
    );

    let (r, _, _) = run(false);
    assert!(
        matches!(
            r,
            Err(FoldError::Exec(
                lance_graph_mask_risc::ExecError::ScratchBufferTooSmall { .. }
            ))
        ),
        "plain call must keep one slot per result, got {r:?}"
    );
}
