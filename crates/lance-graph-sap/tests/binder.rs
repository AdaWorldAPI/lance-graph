//! D-XGP-1: the CATS batch behind Quack's generic `Binder`.
//!
//! The question each test asks: does resolving fields and literals BY NAME,
//! through `lance_graph_quack::bind`, reach exactly the lanes and values the
//! shipped `CatsQuery` reaches through hard-coded `Col` constants?
mod common;
use common::*;
use lance_graph_mask_risc::{execute_into, words_for, Foreign, Out, Planes, Scratch, Value};
use lance_graph_quack::bind::{table, BindError, Binder, FieldKind, TableId};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Query};
use lance_graph_sap::bind::{ACTIVITY, EMPLOYEE, HOURS, WORK_DAY};
use lance_graph_sap::binder::{live_words, CatsBinder, TABLE};
use lance_graph_sap::query::CatsQuery;

const T: TableId = TableId(0);

/// Mixed employees, days and activities; the same shape `tests/fold.rs` uses.
#[allow(clippy::needless_range_loop)] // Same test-only multi-column fixture as tests/fold.rs.
fn mixed(n: usize) -> [Vec<Option<&'static str>>; 23] {
    let mut input = fixture(n);
    for i in 0..n {
        input[5][i] = Some(if i % 3 == 0 { "00000007" } else { "00000042" });
        input[9][i] = Some(if i % 5 == 0 {
            "2026-08-31T23:59:59Z"
        } else {
            "2026-09-01T12:34:56Z"
        });
        input[11][i] = Some(if i % 2 == 0 { "DEV" } else { "OPS" });
        input[10][i] = Some(if i % 2 == 0 { "8.50" } else { "0.125" });
    }
    input
}

#[test]
fn names_resolve_to_the_lanes_the_shipped_query_hard_codes() {
    let batch = bind(&mixed(3));
    let b = CatsBinder::new(&batch);
    assert_eq!(b.table(TABLE), Some(T));
    assert_eq!(b.table("users"), None);

    let f = b.field(T, "employee_number").unwrap();
    assert_eq!(
        (f.col, f.kind, f.validity),
        (Col(EMPLOYEE as u16), FieldKind::Code, None)
    );
    // The C# alias is the same field, as in `CatsSchema::resolve`.
    assert_eq!(b.field(T, "EmployeeNumber"), Some(f));

    let f = b.field(T, "hours_logged").unwrap();
    assert_eq!((f.col, f.kind), (Col(HOURS as u16), FieldKind::I32));
    let f = b.field(T, "activity_type").unwrap();
    assert_eq!((f.col, f.kind), (Col(ACTIVITY as u16), FieldKind::Code));
    // The derived calendar-day lens is a lane the bind produced.
    let f = b.field(T, "work_day").unwrap();
    assert_eq!((f.col, f.kind), (Col(WORK_DAY as u16), FieldKind::I32));
}

#[test]
fn literals_resolve_through_the_fields_own_domain_and_mint_nothing() {
    let batch = bind(&mixed(4));
    let b = CatsBinder::new(&batch);
    let emp = Col(EMPLOYEE as u16);
    let act = Col(ACTIVITY as u16);
    // PERNR is NUMC, not a dictionary: the literal is the number.
    assert_eq!(b.code(T, emp, "00000042"), Some(42));
    assert_eq!(b.code(T, emp, "42"), None, "PERNR requires eight digits");
    // Dictionary fields: first-occurrence codes, 1-based (0 is NULL).
    assert_eq!(b.code(T, act, "DEV"), Some(1));
    assert_eq!(b.code(T, act, "OPS"), Some(2));
    let groups = batch.activity_groups();
    assert_eq!(b.code(T, act, "QA"), None);
    assert_eq!(
        batch.activity_groups(),
        groups,
        "an unknown literal minted nothing"
    );
}

#[test]
fn a_bound_draft_selects_exactly_the_oracles_rows() {
    // An empty batch has observed no activity, so the literal has no code.
    let empty = bind(&mixed(0));
    assert!(matches!(
        table(TABLE)
            .where_eq("activity_type", "DEV")
            .bind(&CatsBinder::new(&empty)),
        Err(BindError::UnknownValue { .. })
    ));
    for n in [1, 2, 63, 64, 65, 131, 4097] {
        let batch = bind(&mixed(n));
        let b = CatsBinder::new(&batch);
        let q = table(TABLE)
            .where_eq("employee_number", "00000042")
            .where_eq("activity_type", "DEV")
            .rows()
            .bind(&b)
            .unwrap();
        let program = lower(&q).unwrap();
        let live = live_words(n);
        let lanes = batch.lanes();
        let planes = Planes {
            n_rows: n,
            masks: &[live.as_slice()],
            lanes: &lanes,
        };
        let mut scratch = Scratch::for_program(&program, n).unwrap();
        let mut mask = vec![0; words_for(n)];
        execute_into(
            &program,
            &planes,
            &Foreign::NONE,
            &mut scratch,
            Out::Mask(&mut mask),
        )
        .unwrap();
        for i in 0..n {
            let kept = (mask[i / 64] >> (i % 64)) & 1 != 0;
            assert_eq!(kept, i % 3 != 0 && i % 2 == 0, "n={n}, row={i}");
        }
    }
}

#[test]
fn folding_over_bound_lanes_equals_the_shipped_cats_query() {
    for n in [2, 65, 4097] {
        let batch = bind(&mixed(n));
        let b = CatsBinder::new(&batch);
        // Selection by name through the generic frontend.
        let bound = table(TABLE)
            .where_eq("employee_number", "00000042")
            .rows()
            .bind(&b)
            .unwrap();
        // `Draft` has no range or grouped fold (Eq/Ne, Count/Rows only), so
        // those are written against the lanes the binder resolved.
        let day = b.field(T, "work_day").unwrap().col;
        let q = Query {
            filter: Filter::and([
                bound.filter,
                Filter::cmp(day, Cmp::GeI32(20260901)),
                Filter::cmp(day, Cmp::LeI32(20260930)),
            ]),
            agg: Agg::GroupSumI32 {
                key: b.field(T, "activity_type").unwrap().col,
                val: b.field(T, "hours_logged").unwrap().col,
            },
        };
        let program = lower(&q).unwrap();
        let live = live_words(n);
        let lanes = batch.lanes();
        let planes = Planes {
            n_rows: n,
            masks: &[live.as_slice()],
            lanes: &lanes,
        };
        let mut scratch = Scratch::for_program(&program, n).unwrap();
        let mut by_name = vec![0; batch.activity_groups() as usize];
        let v = execute_into(
            &program,
            &planes,
            &Foreign::NONE,
            &mut scratch,
            Out::I64(&mut by_name),
        )
        .unwrap();
        assert_eq!(v, Value::GroupSummed);

        let mut shipped =
            CatsQuery::prepare(&batch, "00000042", "2026-09-01", "2026-09-30").unwrap();
        let mut expected = vec![0; shipped.groups()];
        shipped.execute_into(&mut expected).unwrap();
        assert_eq!(by_name, expected, "n={n}");
        assert!(
            expected.iter().any(|s| *s != 0),
            "anti-vacuity: something was summed"
        );
    }
}

#[test]
fn unknown_names_and_values_are_bind_errors_never_defaults() {
    let batch = bind(&mixed(3));
    let b = CatsBinder::new(&batch);
    let err = |d: lance_graph_quack::bind::Draft| d.bind(&b).unwrap_err();
    assert_eq!(
        err(table("timesheet").count()),
        BindError::UnknownTable("timesheet".into())
    );
    assert!(matches!(
        err(table(TABLE).where_eq("invented", "x")),
        BindError::UnknownField { .. }
    ));
    assert!(matches!(
        err(table(TABLE).where_eq("activity_type", "QA")),
        BindError::UnknownValue { .. }
    ));
    assert!(matches!(
        err(table(TABLE).where_eq("hours_logged", "eight")),
        BindError::KindMismatch { .. }
    ));
}

#[test]
fn optional_and_instant_fields_are_refused_not_read_through_sentinels() {
    // CATS stores NULL as a sentinel (code 0, u32::MAX, timestamp 0), not as a
    // validity plane. Binding an optional field without one would let `<>`
    // keep NULL rows, so the binder refuses it.
    let mut input = mixed(3);
    input[6][1] = None; // customer_number is optional
    let batch = bind(&input);
    let b = CatsBinder::new(&batch);
    assert_eq!(b.field(T, "customer_number"), None);
    assert_eq!(b.field(T, "approver_employee_num"), None);
    // U64 instants have no `FieldKind`; the derived `work_day` is the bindable lens.
    assert_eq!(b.field(T, "work_date_utc"), None);
    assert!(matches!(
        table(TABLE)
            .where_eq("customer_number", "0000000123")
            .bind(&b),
        Err(BindError::UnknownField { .. })
    ));
}

/// The billable lens is a VERIFIED conversion (plan §C.6.2): on real rows with
/// every documented indicator plus an undocumented one, `billable = true` and
/// `billable = false` select exactly the rows the DTO's documentation says,
/// and the undocumented value is in neither set.
#[test]
fn billable_lens_selects_exactly_the_documented_rows_and_never_an_unknown_one() {
    use lance_graph_sap::bind::{billable_code, BILLABLE, BILLABLE_UNKNOWN};
    const VALUES: [&str; 4] = ["Billable", "Non-Billable", "Internal Cost", "Pro bono"];
    for n in [4, 65, 4097] {
        let mut input = fixture(n);
        for (i, v) in input[12].iter_mut().enumerate() {
            *v = Some(VALUES[i % 4]);
        }
        let batch = bind(&input);
        let b = CatsBinder::new(&batch);
        let f = b.field(T, "billable").unwrap();
        assert_eq!((f.col, f.kind), (Col(BILLABLE as u16), FieldKind::Code));
        assert_eq!(
            b.code(T, f.col, "maybe"),
            None,
            "only true/false are literals"
        );

        let live = live_words(n);
        let lanes = batch.lanes();
        let planes = Planes {
            n_rows: n,
            masks: &[live.as_slice()],
            lanes: &lanes,
        };
        let mut kept = [Vec::new(), Vec::new()];
        for (k, literal) in ["false", "true"].iter().enumerate() {
            let q = table(TABLE)
                .where_eq("billable", *literal)
                .rows()
                .bind(&b)
                .unwrap();
            let program = lower(&q).unwrap();
            let mut scratch = Scratch::for_program(&program, n).unwrap();
            let mut mask = vec![0; words_for(n)];
            execute_into(
                &program,
                &planes,
                &Foreign::NONE,
                &mut scratch,
                Out::Mask(&mut mask),
            )
            .unwrap();
            kept[k] = (0..n)
                .filter(|i| (mask[i / 64] >> (i % 64)) & 1 != 0)
                .collect();
        }
        // Stated from the DTO's documentation, NOT read from `BILLING_VALUES`:
        // VALUES[0] "Billable" is billable; [1] and [2] are not; [3] is unknown.
        let expected = |want: bool| -> Vec<usize> {
            (0..n)
                .filter(|i| {
                    if want {
                        i % 4 == 0
                    } else {
                        i % 4 == 1 || i % 4 == 2
                    }
                })
                .collect()
        };
        assert_eq!(kept[1], expected(true), "n={n}");
        assert_eq!(kept[0], expected(false), "n={n}");
        // Anti-vacuity: both sets are non-empty and the unknown rows are excluded.
        assert!(!kept[0].is_empty() && !kept[1].is_empty());
        let unknown = (0..n).filter(|i| i % 4 == 3).count();
        assert!(unknown > 0);
        assert_eq!(kept[0].len() + kept[1].len(), n - unknown);
        assert_eq!(billable_code("Pro bono"), BILLABLE_UNKNOWN);
    }
}
