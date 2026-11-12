//! Audit of the existing `lower_group_by_auto`: which grouped aggregates
//! fold to ONE program, which stay a two-phase forest, and what the forest
//! hands from one program to the next.
//!
//! The grouped query is `… WHERE amount > T AND region <> SKIP GROUP BY
//! region`, so one region is empty and an empty group's encoding is covered.

use lance_graph_mask_risc::{
    words_for, Foreign, GroupFold, Out, Program, Scratch, Terminal, Value,
};
use lance_graph_quack::{
    lower_group_by, lower_group_by_auto, normalize_group_sink, Agg, Cmp, Col, Filter, GroupBy,
    GroupLowering, LowerError,
};
use quack_boundary_lab::{
    admitted, oracle, unbound, Prepared, Refused, RowOrder, Source, World, AMOUNT, QTY, REGION,
    REGIONS, ROWS,
};

const T: i32 = 25;
const SKIP: u32 = 3;
const K: usize = REGIONS as usize;

fn col(field: u8) -> Col {
    Col(u16::from(field))
}

fn grouped(agg: Agg, groups: u32) -> GroupBy {
    GroupBy {
        filter: Filter::and([
            Filter::cmp(col(AMOUNT), Cmp::GtI32(T)),
            Filter::cmp(col(REGION), Cmp::NeU32(SKIP)),
        ]),
        key: col(REGION),
        groups,
        agg,
    }
}

fn scratch(p: &Program) -> Scratch<'static> {
    Scratch::for_program(p, ROWS).unwrap()
}

/// Run a folded group program through the admitted path into a K-slot sink.
fn run_folded(program: &Program, w: &World, s: &Source) -> Vec<i64> {
    let binding = w.bind(s).unwrap();
    let p = Prepared::under(program.clone(), s);
    let mut sink = vec![0i64; K];
    let v = admitted(
        &p,
        w,
        &w.view,
        &binding,
        s,
        &mut scratch(program),
        Out::I64(&mut sink),
    );
    assert!(
        matches!(v, Ok(Value::GroupSummed | Value::GroupReduced)),
        "got {v:?}"
    );
    sink
}

/// FAILS IF: SUM, COUNT, MIN or MAX takes the two-phase path, a folded
/// result differs from the oracle, or an empty group reads as a value.
#[test]
fn sum_count_min_max_fold_to_one_program_and_match_the_oracle() {
    let w = World::new();
    let s = Source::new(RowOrder::ById);
    let want = oracle::by_region_amount_gt(&s, T, SKIP);
    assert_eq!(want[SKIP as usize], (0, 0, None, None), "one empty group");
    assert!(want.iter().filter(|g| g.0 > 100).count() == K - 1);

    for agg in [
        Agg::SumI32(col(QTY)),
        Agg::Count,
        Agg::MinI32(col(QTY)),
        Agg::MaxI32(col(QTY)),
    ] {
        let lowered = lower_group_by_auto(&grouped(agg, REGIONS), 0).unwrap();
        let GroupLowering::Folded { program, groups } = lowered else {
            panic!("{agg:?} must fold, got a forest")
        };
        assert_eq!(groups, REGIONS);
        // One program, and it reads no plane: nothing crosses a boundary.
        let mut sink = run_folded(&program, &w, &s);
        match (agg, program.terminal) {
            (Agg::SumI32(_), Terminal::GroupSumI32 { .. }) => {
                let sums: Vec<i64> = want.iter().map(|g| g.1).collect();
                assert_eq!(sink, sums);
            }
            (_, Terminal::GroupReduce { fold, .. }) => {
                let present = normalize_group_sink(fold, &mut sink);
                let reached: Vec<bool> = (0..K).map(|g| present[0] >> g & 1 == 1).collect();
                let expect: Vec<bool> = want.iter().map(|g| g.0 > 0).collect();
                assert_eq!(reached, expect, "{agg:?}");
                let values: Vec<i64> = want
                    .iter()
                    .map(|g| match fold {
                        GroupFold::Count => g.0,
                        GroupFold::MinI32(_) => g.2.map_or(0, i64::from),
                        GroupFold::MaxI32(_) => g.3.map_or(0, i64::from),
                        GroupFold::SumSymI32(_) => g.1,
                    })
                    .collect();
                assert_eq!(sink, values, "{agg:?}");
            }
            (agg, terminal) => panic!("{agg:?} lowered to {terminal:?}"),
        }
    }
}

/// FAILS IF: an aggregate with no grouped terminal is folded anyway, the
/// forest's shape changes, its answer differs from the oracle, or the forest
/// can be run without handing a population mask from one program to the next.
#[test]
fn an_unfoldable_aggregate_is_a_forest_that_carries_a_population_mask() {
    let w = World::new();
    let s = Source::new(RowOrder::ById);
    for agg in [Agg::Any, Agg::All, Agg::Rows] {
        let lowered = lower_group_by_auto(&grouped(agg, REGIONS), 0).unwrap();
        let GroupLowering::Forest(plan) = lowered else {
            panic!("{agg:?} has no grouped terminal and must stay a forest")
        };
        assert_eq!(plan.groups.len(), K);
        assert!(matches!(plan.filter.terminal, Terminal::Keep { .. }));
    }

    let GroupLowering::Forest(plan) = lower_group_by_auto(&grouped(Agg::Any, REGIONS), 0).unwrap()
    else {
        unreachable!()
    };
    // Phase 1 writes the kept filter: one bit per row of the population.
    let mut kept = vec![0u64; words_for(ROWS)];
    unbound(
        &plan.filter,
        &s.raw_lanes(),
        &[],
        &Foreign::NONE,
        &mut scratch(&plan.filter),
        Out::Mask(&mut kept),
    )
    .unwrap();
    assert_eq!(kept.len() * 8, 5_008, "the intermediate, in bytes");
    // Phase 2 reads it back as a plane, once per group.
    let want = oracle::by_region_amount_gt(&s, T, SKIP);
    for (g, program) in plan.groups.iter().enumerate() {
        let got = unbound(
            program,
            &s.raw_lanes(),
            &[&kept],
            &Foreign::NONE,
            &mut scratch(program),
            Out::None,
        );
        assert_eq!(got, Ok(Value::Bool(want[g].0 > 0)), "group {g}");
    }

    // The same phase-2 program has no admitted form: its input is a mask the
    // caller made, and no owner issues it.
    let binding = w.bind(&s).unwrap();
    let p = Prepared::under(plan.groups[0].clone(), &s);
    assert_eq!(
        admitted(
            &p,
            &w,
            &w.view,
            &binding,
            &s,
            &mut scratch(&p.program),
            Out::None
        ),
        Err(Refused::Blocked(
            "mask plane: the lane socket issues lanes, not planes"
        ))
    );
}

/// FAILS IF: the fold and the two-phase forest disagree on a query both can
/// answer. The fold is a replacement only because it is exact.
#[test]
fn the_fold_and_the_forest_agree_where_both_exist() {
    let w = World::new();
    let s = Source::new(RowOrder::ById);
    let g = grouped(Agg::Count, REGIONS);
    let GroupLowering::Folded { program, .. } = lower_group_by_auto(&g, 0).unwrap() else {
        panic!("COUNT folds")
    };
    let folded = run_folded(&program, &w, &s);

    let plan = lower_group_by(&g, 0).unwrap();
    let mut kept = vec![0u64; words_for(ROWS)];
    unbound(
        &plan.filter,
        &s.raw_lanes(),
        &[],
        &Foreign::NONE,
        &mut scratch(&plan.filter),
        Out::Mask(&mut kept),
    )
    .unwrap();
    let forest: Vec<i64> = plan
        .groups
        .iter()
        .map(|program| {
            match unbound(
                program,
                &s.raw_lanes(),
                &[&kept],
                &Foreign::NONE,
                &mut scratch(program),
                Out::None,
            ) {
                Ok(Value::Count(n)) => n as i64,
                other => panic!("a group COUNT, got {other:?}"),
            }
        })
        .collect();
    assert_eq!(folded, forest);
    assert!(forest.iter().sum::<i64>() > 1_000);
}

/// FAILS IF: an empty group universe folds (there is no sink to name), or a
/// grouped blend lowers at all.
#[test]
fn an_empty_universe_is_a_forest_with_no_groups_and_a_blend_is_refused() {
    let lowered = lower_group_by_auto(&grouped(Agg::SumI32(col(QTY)), 0), 0).unwrap();
    let GroupLowering::Forest(plan) = lowered else {
        panic!("K = 0 has no group sink and must not fold")
    };
    assert!(plan.groups.is_empty());
    // The plan still carries a phase-1 Keep program. Run as written it would
    // write a population mask no group program reads.
    assert!(matches!(plan.filter.terminal, Terminal::Keep { .. }));

    let blend = Agg::BlendI32 {
        then: col(QTY),
        els: col(AMOUNT),
    };
    assert_eq!(
        lower_group_by_auto(&grouped(blend, REGIONS), 0),
        Err(LowerError::GroupedBlend)
    );
}
