//! Cases A to F: one unchanged Quack-lowered program, run unbound (the
//! negative control) and admitted (lanes through one `LaneBinding` check),
//! each compared with a scalar oracle that names rows by value.
//!
//! The program under test is
//! `SUM(qty) WHERE row IN [LO, HI) AND amount > T`. Its range is in ROW
//! ORDINALS of the lane order it was lowered for (a source in id order), so
//! its meaning is `LO <= id < HI` there and something else in any other
//! order. The program text never changes between cases.

use lance_graph_dt64::dr::FrameRefusal;
use lance_graph_dt64::{CoordSpace, LaneRefusal};
use lance_graph_mask_risc::{
    words_for, ExecError, Foreign, ForeignPlane, LaneRef, Out, Program, Scratch, Value,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, ForeignPlane as QForeignPlane, Mask, Query};
use quack_boundary_lab::{
    admitted, oracle, unbound, Prepared, Refused, RowOrder, Source, View, World, AMOUNT, DISCOUNT,
    DISC_VALID, ID, OTHER_READER, PARTNER, PARTNERS, QTY, READER, ROWS,
};

const LO: u32 = 1_000;
const HI: u32 = 9_000;
const T: i32 = 25;

fn col(field: u8) -> Col {
    Col(u16::from(field))
}

/// `SUM(qty) WHERE row IN [LO, HI) AND amount > T`, lowered once.
fn ranged_program() -> Program {
    lower(&Query {
        filter: Filter::and([
            Filter::cmp(col(ID), Cmp::Range { lo: LO, hi: HI }),
            Filter::cmp(col(AMOUNT), Cmp::GtI32(T)),
        ]),
        agg: Agg::SumI32(col(QTY)),
    })
    .expect("the ranged program lowers")
}

/// The same question asked by VALUE: `SUM(qty) WHERE LO <= id < HI AND
/// amount > T`. Correct in any row order.
fn by_value_program() -> Program {
    lower(&Query {
        filter: Filter::and([
            Filter::cmp(col(ID), Cmp::GeI32(LO as i32)),
            Filter::cmp(col(ID), Cmp::LtI32(HI as i32)),
            Filter::cmp(col(AMOUNT), Cmp::GtI32(T)),
        ]),
        agg: Agg::SumI32(col(QTY)),
    })
    .expect("the by-value program lowers")
}

fn scratch(p: &Program) -> Scratch<'static> {
    Scratch::for_program(p, ROWS).expect("the fixture's scratch is addressable")
}

fn want(s: &Source) -> i64 {
    oracle::sum_qty_id_in_amount_gt(s, LO as i32, HI as i32, T)
}

fn run_unbound(p: &Program, s: &Source) -> Result<Value, ExecError> {
    unbound(
        p,
        &s.raw_lanes(),
        &[],
        &Foreign::NONE,
        &mut scratch(p),
        Out::None,
    )
}

/// A. FAILS IF: a correctly bound source is refused, the admitted answer
/// differs from the oracle or from the unbound run over the same lanes, or
/// the fixture is trivial.
#[test]
fn a_correctly_bound_source_answers_the_oracle() {
    let w = World::new();
    let s = Source::new(RowOrder::ById);
    let binding = w.bind(&s).expect("the source binds");
    let p = Prepared::under(ranged_program(), &s);
    let got = admitted(
        &p,
        &w,
        &w.view,
        &binding,
        &s,
        &mut scratch(&p.program),
        Out::None,
    );
    let oracle = want(&s);
    assert!(oracle > 1_000, "the query selects a real population");
    assert_eq!(got, Ok(Value::SumI64(oracle)));
    assert_eq!(run_unbound(&p.program, &s), Ok(Value::SumI64(oracle)));
    // The oracle is order-free: the value form agrees with it too.
    assert_eq!(
        run_unbound(&by_value_program(), &s),
        Ok(Value::SumI64(oracle))
    );
}

/// B. FAILS IF: the unbound run over another source's lanes is refused or
/// happens to be right (then the control proves nothing), or the admitted
/// path reads the other source under any pairing.
#[test]
fn b_a_wrong_source_computes_unbound_and_is_refused_admitted() {
    let w = World::new();
    let s1 = Source::new(RowOrder::ById);
    let s2 = Source::new(RowOrder::Reversed);
    // Equal row count, lane kinds, coordinate numbers and frame revisions;
    // even the same multiset of rows.
    assert_eq!(s1.space(), s2.space());
    assert_eq!(s1.frame_store().revisions(), s2.frame_store().revisions());
    assert_eq!(want(&s1), want(&s2), "both hold the same logical rows");
    assert_ne!(s1.checksum(), s2.checksum(), "in another row order");

    let p = Prepared::under(ranged_program(), &s1);

    // Negative control: the physical program runs and is wrong.
    let wrong = run_unbound(&p.program, &s2);
    let Ok(Value::SumI64(wrong)) = wrong else {
        panic!("the unbound run must execute, got {wrong:?}")
    };
    assert!(wrong > 1_000, "a plausible answer, not an empty one");
    assert_ne!(wrong, want(&s2), "and it is not the answer");

    // The binding made over s1 does not read s2.
    let b1 = w.bind(&s1).unwrap();
    let got = admitted(
        &p,
        &w,
        &w.view,
        &b1,
        &s2,
        &mut scratch(&p.program),
        Out::None,
    );
    assert_eq!(
        got,
        Err(Refused::Binding(LaneRefusal::Frame(
            FrameRefusal::ForeignFrame
        )))
    );

    // A fresh, valid binding over s2 does not admit a program lowered under
    // s1's frame, although every number the two sources carry is equal.
    let b2 = w.bind(&s2).unwrap();
    let got = admitted(
        &p,
        &w,
        &w.view,
        &b2,
        &s2,
        &mut scratch(&p.program),
        Out::None,
    );
    assert_eq!(got, Err(Refused::Program(FrameRefusal::ForeignFrame)));

    // The value form, prepared under s2, is admitted and right.
    let pv = Prepared::under(by_value_program(), &s2);
    let got = admitted(
        &pv,
        &w,
        &w.view,
        &b2,
        &s2,
        &mut scratch(&pv.program),
        Out::None,
    );
    assert_eq!(got, Ok(Value::SumI64(want(&s2))));
}

/// C. FAILS IF: a coordinate revision of the same source leaves the old
/// binding or the old program admissible, or the unbound control stays right.
#[test]
fn c_a_coordinate_revision_refuses_the_old_binding_and_the_old_program() {
    let w = World::new();
    let mut s = Source::new(RowOrder::ById);
    let old_binding = w.bind(&s).unwrap();
    let p = Prepared::under(ranged_program(), &s);
    let before = want(&s);
    assert_eq!(
        admitted(
            &p,
            &w,
            &w.view,
            &old_binding,
            &s,
            &mut scratch(&p.program),
            Out::None
        ),
        Ok(Value::SumI64(before))
    );

    s.reorder(12_345);
    assert_eq!(want(&s), before, "a reorder changes no logical row");

    // Negative control.
    let Ok(Value::SumI64(wrong)) = run_unbound(&p.program, &s) else {
        panic!("the unbound run must execute")
    };
    assert!(wrong > 1_000);
    assert_ne!(wrong, before);

    // The old binding names the frame that moved.
    assert_eq!(
        admitted(
            &p,
            &w,
            &w.view,
            &old_binding,
            &s,
            &mut scratch(&p.program),
            Out::None
        ),
        Err(Refused::Binding(LaneRefusal::Frame(
            FrameRefusal::CoordinatesChanged
        )))
    );

    // Rebinding is lawful; the program lowered under the old order is not.
    let binding = w.bind(&s).expect("the revised source binds again");
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
        Err(Refused::Program(FrameRefusal::CoordinatesChanged))
    );

    // The value form, prepared under the new frame, is admitted and right.
    let pv = Prepared::under(by_value_program(), &s);
    assert_eq!(
        admitted(
            &pv,
            &w,
            &w.view,
            &binding,
            &s,
            &mut scratch(&pv.program),
            Out::None
        ),
        Ok(Value::SumI64(before))
    );

    // A binding that asks for the OLD row order is refused by the lanes.
    let old_space = CoordSpace {
        population: s.space().population,
        coordinates: s.space().coordinates - 1,
    };
    assert!(matches!(
        w.bind_as(&s, READER, old_space),
        Err(LaneRefusal::CoordinateSpace { .. })
    ));
}

/// The partners table of case D: one `tier` lane, addressed by row.
fn partner_tiers() -> Vec<i32> {
    (0..PARTNERS as i64)
        .map(|i| ((i * 29) % 9) as i32)
        .collect()
}

/// D. FAILS IF: the executor can tell a derived kept mask from a resident
/// plane (it cannot, and the test would then be describing something else),
/// a detached mask stays right, or the admitted path accepts a foreign plane.
#[test]
fn d_a_detached_kept_mask_is_accepted_unbound_and_blocked_admitted() {
    const TIER: i32 = 5;
    let w = World::new();
    let s = Source::new(RowOrder::ById);
    let mut tier = partner_tiers();

    // Program 1 over PARTNERS: the kept mask `tier > TIER`.
    let keep = lower(&Query {
        filter: Filter::cmp(Col(0), Cmp::GtI32(TIER)),
        agg: Agg::Rows,
    })
    .unwrap();
    let mut kept = vec![0u64; words_for(PARTNERS)];
    let partner_lanes = [LaneRef::I32(&tier)];
    let v = unbound(
        &keep,
        &partner_lanes,
        &[],
        &Foreign::NONE,
        &mut Scratch::for_program(&keep, PARTNERS).unwrap(),
        Out::Mask(&mut kept),
    );
    assert!(matches!(v, Ok(Value::Mask(_))));

    // Program 2 over ORDERS: the semijoin through `partner`.
    let semi = lower(&Query {
        filter: Filter::semijoin(col(PARTNER), QForeignPlane(0)),
        agg: Agg::Count,
    })
    .unwrap();
    let run_semi = |kept: &[u64], s: &Source| {
        let planes = [ForeignPlane {
            words: kept,
            rows: PARTNERS,
        }];
        let foreign = Foreign {
            planes: &planes,
            lanes: &[],
        };
        unbound(
            &semi,
            &s.raw_lanes(),
            &[],
            &foreign,
            &mut scratch(&semi),
            Out::None,
        )
    };

    // Fresh, the derived mask is taken for a resident plane and the answer
    // is right. Nothing at this layer distinguishes the two.
    let fresh = oracle::count_orders_with_partner_tier_gt(&s, &tier, TIER);
    assert!(fresh > 1_000);
    assert_eq!(run_semi(&kept, &s), Ok(Value::Count(fresh)));

    // The partners change; the mask is now detached from its source.
    for t in tier.iter_mut().step_by(3) {
        *t = 8 - *t;
    }
    let now = oracle::count_orders_with_partner_tier_gt(&s, &tier, TIER);
    assert_ne!(now, fresh, "the revision moved the answer");
    assert_eq!(
        run_semi(&kept, &s),
        Ok(Value::Count(fresh)),
        "the stale mask still answers, with the old population"
    );

    // Admitted: no owner issues a foreign plane, so the program is not run.
    let binding = w.bind(&s).unwrap();
    let p = Prepared::under(semi.clone(), &s);
    assert_eq!(
        admitted(
            &p,
            &w,
            &w.view,
            &binding,
            &s,
            &mut scratch(&semi),
            Out::None
        ),
        Err(Refused::Blocked(
            "foreign plane: no owner-issued residency admission"
        ))
    );
}

/// E. FAILS IF: under a valid frame, another class view, another reader
/// scope or another selection row order is admitted.
#[test]
fn e_a_valid_frame_with_another_view_scope_or_row_order_is_refused() {
    let w = World::new();
    let mut s = Source::new(RowOrder::ById);
    let binding = w.bind(&s).unwrap();
    let p = Prepared::under(ranged_program(), &s);
    let run = |view: &View, s: &Source| {
        admitted(
            &p,
            &w,
            view,
            &binding,
            s,
            &mut scratch(&p.program),
            Out::None,
        )
    };
    assert!(run(&w.view, &s).is_ok(), "the frame is valid");

    // Another interpretation: a view that names `quantity` where the
    // binding bound `amount`.
    let other = View::swapped(AMOUNT, QTY);
    assert_eq!(
        run(&other, &s),
        Err(Refused::Binding(LaneRefusal::MappingChanged {
            field: AMOUNT
        }))
    );

    // Negative control for the same mistake made without a binding: a caller
    // that reads lanes by its own view's positions sums another column.
    let by_other_view = lower(&Query {
        filter: Filter::cmp(col(AMOUNT), Cmp::GtI32(T)),
        agg: Agg::SumI32(col(QTY)),
    })
    .unwrap();
    let mut swapped = s.raw_lanes();
    swapped.swap(usize::from(AMOUNT), usize::from(QTY));
    let got = unbound(
        &by_other_view,
        &swapped,
        &[],
        &Foreign::NONE,
        &mut scratch(&by_other_view),
        Out::None,
    );
    assert!(matches!(got, Ok(Value::SumI64(_))));
    assert_ne!(got, Ok(Value::SumI64(oracle::sum_qty_amount_gt(&s, T))));

    // Selection criteria: another reader scope, another row order.
    assert!(matches!(
        w.bind_as(&s, OTHER_READER, s.space()),
        Err(LaneRefusal::Scope { .. })
    ));
    let other_population = CoordSpace {
        population: s.space().population + 1,
        coordinates: s.space().coordinates,
    };
    assert!(matches!(
        w.bind_as(&s, READER, other_population),
        Err(LaneRefusal::CoordinateSpace { .. })
    ));

    // The source now serves another reader: the bound mapping moved.
    s.serve(OTHER_READER);
    assert!(matches!(
        run(&w.view, &s),
        Err(Refused::Binding(LaneRefusal::MappingChanged { .. }))
    ));
}

/// F. FAILS IF: a NULL row's stale payload takes part in an answer, in a
/// predicate or in its negation; or the fixture's stale bytes are inert (the
/// naive runs would then agree with the oracle and prove nothing).
#[test]
fn f_stale_bytes_under_null_never_reach_the_answer() {
    const D: i32 = 100;
    let w = World::new();
    let s = Source::new(RowOrder::ById);
    let binding = w.bind(&s).unwrap();
    let gt = Filter::cmp(col(DISCOUNT), Cmp::GtI32(D));
    let valid = Filter::cmp(col(DISC_VALID), Cmp::EqU32(1));

    for negated in [false, true] {
        let want = oracle::count_discount_gt_3vl(&s, D, negated);
        assert!(want > 1_000);
        let leaf = if negated {
            Filter::negate(gt.clone())
        } else {
            gt.clone()
        };

        // Naive two-valued evaluation reads the stale bytes and is wrong.
        let naive = lower(&Query {
            filter: leaf.clone(),
            agg: Agg::Count,
        })
        .unwrap();
        let got = run_unbound(&naive, &s);
        assert!(matches!(got, Ok(Value::Count(_))));
        assert_ne!(got, Ok(Value::Count(want)), "negated={negated}");

        // Admitted: validity is a witnessed LANE, gated inside the program.
        let gated = lower(&Query {
            filter: Filter::and([valid.clone(), leaf.clone()]),
            agg: Agg::Count,
        })
        .unwrap();
        let p = Prepared::under(gated, &s);
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
            Ok(Value::Count(want)),
            "negated={negated}"
        );

        // Quack's own three-valued rewrite over a validity PLANE is right
        // too, on the unbound path: the plane is the caller's, unwitnessed.
        let plane = s.validity_plane();
        let rewritten = lower(&Query {
            filter: leaf.sql_where(&[(col(DISCOUNT), Mask(0))]),
            agg: Agg::Count,
        })
        .unwrap();
        let got = unbound(
            &rewritten,
            &s.raw_lanes(),
            &[&plane],
            &Foreign::NONE,
            &mut scratch(&rewritten),
            Out::None,
        );
        assert_eq!(got, Ok(Value::Count(want)), "negated={negated}");

        // That plane has no admission seam: the same program is blocked.
        let p = Prepared::under(rewritten, &s);
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
}
