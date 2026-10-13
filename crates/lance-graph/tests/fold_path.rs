//! D-FOLD-CORE-b: the fold path is a core capability.
//!
//! `lance-graph` reaches mask-risc, Quack and R2IL through `lance_graph::fold`
//! with no extra dependency or feature. Each test runs a query through one
//! frontend and checks it against a plain row loop over the same lane.

use lance_graph::fold::mask_risc::{self, Foreign, LaneRef, Planes, Pred, Scratch, Value};
use lance_graph::fold::quack::{self, Agg, Cmp, Col, Filter, Query};
use lance_graph::fold::r2il::{procs, FoldDialect, Val};

/// A deterministic lane whose values cover `0..1000` in scrambled row order,
/// so a value predicate selects scattered rows, not a row range.
fn lane() -> Vec<i32> {
    (0..4096i32)
        .map(|i| (i.wrapping_mul(7919)) % 1000)
        .collect()
}

/// A range of ROWS (`Cmp::Range` is a row range on the ordered axis, not a
/// value range).
fn rows(lo: u32, hi: u32) -> Filter {
    Filter::Cmp(Col(0), Cmp::Range { lo, hi })
}

/// FAILS IF: core cannot lower a Quack query and run it on mask-risc, the two
/// row ranges are not intersected, or the value predicate is not applied
/// inside the intersection.
#[test]
fn quack_through_core_equals_the_row_loop() {
    let v = lane();
    let lanes = [LaneRef::I32(&v)];
    let planes = Planes {
        n_rows: v.len(),
        masks: &[],
        lanes: &lanes,
    };
    let q = Query {
        filter: Filter::and([
            rows(100, 700),
            rows(300, 900),
            Filter::Cmp(Col(0), Cmp::GeI32(500)),
        ]),
        agg: Agg::Count,
    };
    let program = quack::lower_fused(&q).expect("lowers");
    let mut scratch = Scratch::for_program(&program, v.len()).expect("scratch");
    let got = mask_risc::execute(&program, &planes, &mut scratch, None).expect("runs");

    let want = (300..700).filter(|&r| v[r] >= 500).count();
    // Anti-vacuity: the value predicate cuts inside the row intersection.
    assert!(want > 0 && want < 400, "want = {want}");
    assert_eq!(got, Value::Count(want));
}

/// FAILS IF: core cannot run an R2IL procedure over mask-risc: a row range
/// counted through the fold dialect must equal its width.
#[test]
fn r2il_through_core_counts_a_row_range() {
    let v = lane();
    let lanes = [LaneRef::I32(&v)];
    let planes = Planes {
        n_rows: v.len(),
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign {
        planes: &[],
        lanes: &[],
    };
    let (lo, hi) = (130u32, 3001u32);
    let mut d = FoldDialect::new(&planes, &foreign);
    let mut stack: Vec<Val> = Vec::new();
    d.emit_pred(Pred::Range { lo, hi }, &mut stack);
    procs::count(&mut d, &mut stack).expect("count");
    assert!(d.poison().is_none());
    match stack.as_slice() {
        [Val::Scalar(n)] => assert_eq!(*n, i64::from(hi - lo)),
        other => panic!("expected one scalar, got {other:?}"),
    }
}
