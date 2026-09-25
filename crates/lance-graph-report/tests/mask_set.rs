//! The SET coordinate: a many-to-many axis whose members are resident masks.
//!
//! A row lands in every member whose mask holds it — zero, one or several —
//! so this dimension's cells do not sum to the selected population. Every
//! plan here is checked against the per-row oracle, which places a row in the
//! cartesian product of its per-dimension memberships.

mod common;

use common::*;
use lance_graph_report::*;

const A: FieldId = FieldId(0);
const V: FieldId = FieldId(100);
const BASE: MaskId = MaskId(10);
const TAGS: u32 = 5;

fn tags() -> CoordSpec {
    CoordSpec::MaskSet {
        base: BASE,
        count: TAGS,
    }
}

/// `n` rows, an ordinal field A (domain 3), a value field, and `TAGS` masks
/// where each row carries each tag with probability 1/3 — so rows with no
/// tag and rows with several both occur.
fn tagged(n: usize, seed: u64) -> Fixture {
    let mut fx = synthetic(n, &[3], seed);
    let mut r = Rng(seed.wrapping_mul(7) | 1);
    let words = n.div_ceil(64);
    for t in 0..TAGS {
        let mut w = vec![0u64; words];
        for i in 0..n {
            if r.below(3) == 0 {
                w[i / 64] |= 1 << (i % 64);
            }
        }
        fx.masks.push((MaskId(BASE.0 + t), w));
    }
    fx
}

fn membership(fx: &Fixture, i: usize) -> usize {
    (0..TAGS)
        .filter(|&t| {
            let w = &fx.masks.iter().find(|(m, _)| m.0 == BASE.0 + t).unwrap().1;
            w[i / 64] >> (i % 64) & 1 == 1
        })
        .count()
}

/// The fixture really exercises multi-membership: some rows carry no tag and
/// some carry several. Without both, the tests below prove nothing about sets.
fn assert_fixture_is_a_set(fx: &Fixture) {
    let counts: Vec<usize> = (0..fx.n).map(|i| membership(fx, i)).collect();
    assert!(counts.contains(&0), "some row must carry no tag");
    assert!(
        counts.iter().any(|&c| c >= 2),
        "some row must carry several tags"
    );
}

#[test]
fn a_tag_set_alone_counts_each_row_in_every_tag_it_carries() {
    let fx = tagged(3_000, 21);
    assert_fixture_is_a_set(&fx);
    let batch = fx.batch();
    let plan = with_four(ReportPlan::over(src()).axis(tags(), AxisRole::Row), V);
    let (res, _) = plan.execute(&batch, &PlannerPolicy::default()).unwrap();
    assert_matches_oracle(&fx, &plan, &res, V);

    // The set semantics, stated as numbers: the per-tag counts add up to the
    // number of (row, tag) memberships, which exceeds the tagged rows.
    let m = &res.measures()[0];
    let per_tag: i64 = (0..TAGS)
        .map(|t| match res.value(m, &[], &[t], &[]) {
            CellValue::Int(c) => c,
            other => panic!("tag {t}: {other:?}"),
        })
        .sum();
    let memberships: usize = (0..fx.n).map(|i| membership(&fx, i)).sum();
    let tagged_rows = (0..fx.n).filter(|&i| membership(&fx, i) > 0).count();
    assert_eq!(per_tag, memberships as i64);
    assert!(
        per_tag > tagged_rows as i64,
        "a row in two tags counts twice"
    );
}

#[test]
fn a_tag_set_crossed_with_an_ordinal_field_matches_the_oracle_dense_and_sparse() {
    let fx = tagged(4_000, 33);
    assert_fixture_is_a_set(&fx);
    let batch = fx.batch();
    let plan = with_four(
        ReportPlan::over(src()).pivot(&[tags()], &[CoordSpec::Field(A)]),
        V,
    );
    let dense = PlannerPolicy::default();
    let sparse = PlannerPolicy {
        dense_cell_budget: 1,
        ..PlannerPolicy::default()
    };
    for policy in [dense, sparse] {
        let pp = plan.explain(&batch, &policy).unwrap();
        // The ordinal field is the fold key; the set is never chosen for it.
        let key = pp.fold_key.expect("the ordinal field keys the fold");
        assert_eq!(pp.dims[key].coord, CoordSpec::Field(A));
        let (res, st) = plan.execute(&batch, &policy).unwrap();
        assert_eq!(res.space().is_sparse(), policy == sparse);
        assert_matches_oracle(&fx, &plan, &res, V);
        assert!(st.population_scans >= u64::from(TAGS));
    }
}

#[test]
fn a_selection_narrows_every_tag_cell() {
    let fx = tagged(3_000, 45);
    let batch = fx.batch();
    let plan = with_four(
        ReportPlan::over(src())
            .filter(Selection::cmp(A, CmpOp::Eq, Scalar::Ordinal(1)))
            .axis(tags(), AxisRole::Row),
        V,
    );
    let (res, _) = plan.execute(&batch, &PlannerPolicy::default()).unwrap();
    assert_matches_oracle(&fx, &plan, &res, V);
}

#[test]
fn a_missing_tag_mask_is_refused_not_read_as_empty() {
    let mut fx = tagged(500, 57);
    fx.masks.retain(|(m, _)| m.0 != BASE.0 + 3);
    let batch = fx.batch();
    let plan = ReportPlan::over(src())
        .axis(tags(), AxisRole::Row)
        .measure(Measure::count());
    assert_eq!(
        plan.execute(&batch, &PlannerPolicy::default()).unwrap_err(),
        ReportError::UnknownMask(MaskId(BASE.0 + 3))
    );
}

#[test]
fn an_empty_tag_set_is_refused() {
    let fx = tagged(500, 69);
    let batch = fx.batch();
    let plan = ReportPlan::over(src())
        .axis(
            CoordSpec::MaskSet {
                base: BASE,
                count: 0,
            },
            AxisRole::Row,
        )
        .measure(Measure::count());
    assert_eq!(
        plan.explain(&batch, &PlannerPolicy::default()).unwrap_err(),
        ReportError::EmptyMaskSet(BASE)
    );
}

#[test]
fn explain_names_the_set_and_its_members() {
    let fx = tagged(500, 81);
    let batch = fx.batch();
    let plan = ReportPlan::over(src())
        .axis(tags(), AxisRole::Row)
        .measure(Measure::count());
    let text = plan
        .explain(&batch, &PlannerPolicy::default())
        .unwrap()
        .to_string();
    assert!(text.contains("M10..M15"), "{text}");
    assert!(text.contains("mask set (5 resident masks"), "{text}");
}
