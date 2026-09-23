//! The AGNOSTIC validation suite: synthetic dimensions, no domain nouns.
//! TEST 1–9 of the capstone, plus the planner-knob inertness twins.

mod common;

use common::*;
use lance_graph_mask_risc::Program;
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Query};
use lance_graph_report::*;

const A: FieldId = FieldId(0);
const B: FieldId = FieldId(1);
const C: FieldId = FieldId(2);
const V: FieldId = FieldId(100);
const W: FieldId = FieldId(101);

fn fa() -> CoordSpec {
    CoordSpec::Field(A)
}
fn fb() -> CoordSpec {
    CoordSpec::Field(B)
}
fn fc() -> CoordSpec {
    CoordSpec::Field(C)
}

// ── TEST 1 — AXIS ROTATION ──────────────────────────────────────────────
#[test]
fn t1_rotation_is_metadata_only_and_reads_the_same_cells() {
    let fx = synthetic(5_000, &[4, 7], 11);
    let batch = fx.batch();
    let plan = with_four(ReportPlan::over(src()).pivot(&[fa()], &[fb()]), V);
    let (res, stats) = plan.execute(&batch, &PlannerPolicy::default()).unwrap();
    assert_matches_oracle(&fx, &plan, &res, V);
    let scans_before = stats.population_scans;

    // Rotation through the RESULT (no batch in reach) and through the PLAN.
    let rot = res.rotate();
    let rot_plan = plan.clone().rotate();
    assert_eq!(
        rot_plan.physical_key(),
        plan.physical_key(),
        "roles are not part of the fold"
    );
    let rot2 = res.reinterpret(&rot_plan).unwrap();
    for r in [&rot, &rot2] {
        assert!(
            std::sync::Arc::ptr_eq(r.space(), res.space()),
            "cell payload shared, not copied"
        );
        assert_eq!(r.space().payload_addr(), res.space().payload_addr());
    }
    // rows=A,cols=B cell (a,b) == rows=B,cols=A cell (b,a), every cell.
    let m = &res.measures()[1];
    for a in 0..4 {
        for b in 0..7 {
            assert_eq!(res.value(m, &[], &[a], &[b]), rot.value(m, &[], &[b], &[a]));
            assert_eq!(
                res.value(m, &[], &[a], &[b]),
                rot2.value(m, &[], &[b], &[a])
            );
        }
    }
    assert_eq!(rot.row_keys().len(), 7);
    assert_eq!(rot.column_keys().len(), 4);
    // No re-aggregation: the only scans are the original execution's.
    assert_eq!(scans_before, stats.population_scans);
}

// ── TEST 2 — THREE-DIMENSIONAL ROLE PERMUTATION ─────────────────────────
#[test]
fn t2_three_dim_role_permutation_changes_only_view_metadata() {
    let fx = synthetic(8_000, &[3, 4, 5], 12);
    let batch = fx.batch();
    let p1 = with_four(
        ReportPlan::over(src())
            .axis(fa(), AxisRole::Row)
            .axis(fb(), AxisRole::Column)
            .axis(fc(), AxisRole::Page),
        V,
    );
    let (r1, _) = p1.execute(&batch, &PlannerPolicy::default()).unwrap();
    assert_matches_oracle(&fx, &p1, &r1, V);
    let p2 = p1.clone().view(&[fc(), fb()], &[fa()], &[]);
    assert_eq!(p2.physical_key(), p1.physical_key());
    let r2 = r1.reinterpret(&p2).unwrap();
    assert!(std::sync::Arc::ptr_eq(r1.space(), r2.space()));
    assert_eq!(r2.view().rows.len(), 2);
    assert!(r2.view().pages.is_empty());
    let m = &r1.measures()[0];
    for a in 0..3 {
        for b in 0..4 {
            for c in 0..5 {
                // view 1: page=c, row=a, col=b   view 2: row=(c,b), col=a
                assert_eq!(
                    r1.value(m, &[c], &[a], &[b]),
                    r2.value(m, &[], &[c, b], &[a])
                );
            }
        }
    }
    // Removing a presented axis from the view merges over it — still no fold.
    let r3 = r1.with_roles(&[fa()], &[], &[]).unwrap();
    assert!(std::sync::Arc::ptr_eq(r1.space(), r3.space()));
    let or = fx.oracle(&p1, Some(V));
    let want: i64 = or.iter().filter(|(k, _)| k[0] == 2).map(|(_, v)| v.0).sum();
    assert_eq!(r3.value(m, &[], &[2], &[]), CellValue::Int(want));
}

// ── TEST 3 — RANGE WITHOUT MASK ─────────────────────────────────────────
#[test]
fn t3_range_count_and_sum_write_no_mask() {
    let fx = synthetic(20_000, &[3], 13);
    let batch = fx.batch();
    let range = Selection::Range(RowRange {
        lo: 1_234,
        hi: 17_001,
    });
    let plan = with_four(ReportPlan::over(src()).filter(range.clone()), V);
    let pp = plan.explain(&batch, &PlannerPolicy::default()).unwrap();
    assert_eq!(
        pp.extent,
        Some(RowRange {
            lo: 1_234,
            hi: 17_001
        })
    );
    assert_eq!(pp.selection, SelectionCarrier::ValidityPlane);
    assert!(pp
        .first_pass
        .iter()
        .all(|p| p.ops.is_empty() && !p.requires_scratch()));
    let (res, st) = plan.execute(&batch, &PlannerPolicy::default()).unwrap();
    assert_eq!(st.tile_mask_ops, 0, "no mask op of any size");
    assert_eq!(st.mask_materializations, 0);
    assert_eq!(st.scratch_bytes_peak, 0);
    assert_matches_oracle(&fx, &plan, &res, V);

    // Can-fire twin: the SAME range, but not top-level (inside an OR), must
    // fall back to a tile-local range write — proving the extent path is a
    // decision, not a constant.
    let never = Selection::cmp(V, CmpOp::Gt, Scalar::Int(10_000));
    let plan2 = with_four(ReportPlan::over(src()).filter(range.or(never)), V);
    let pp2 = plan2.explain(&batch, &PlannerPolicy::default()).unwrap();
    assert_eq!(pp2.extent, None);
    assert_ne!(pp2.selection, SelectionCarrier::ValidityPlane);
    let (res2, st2) = plan2.execute(&batch, &PlannerPolicy::default()).unwrap();
    assert!(st2.tile_mask_ops > 0);
    assert_eq!(
        res2.grand_total(&res2.measures()[1]),
        res.grand_total(&res.measures()[1])
    );
}

// ── TEST 4 — MASK REUSE ─────────────────────────────────────────────────
fn sparse_selection() -> Selection {
    Selection::cmp(A, CmpOp::Eq, Scalar::Ordinal(3))
        .and(Selection::cmp(V, CmpOp::Gt, Scalar::Int(40)))
        .and_not(Selection::cmp(W, CmpOp::Lt, Scalar::Int(100)))
}

#[test]
fn t4_planner_reuses_one_mask_only_when_the_selection_is_reevaluated_enough() {
    let fx = synthetic(30_000, &[10, 6], 14);
    let batch = fx.batch();
    // Anti-vacuity: the selection keeps a small, non-empty minority.
    let kept = (0..fx.n)
        .filter(|&i| fx.selected(&sparse_selection(), i))
        .count();
    assert!(kept > 0 && kept * 20 < fx.n, "kept {kept}");

    // One scalar measure: one program, re-evaluation is cheaper → tile-local.
    let one = ReportPlan::over(src()).filter(sparse_selection());
    let pp = one.explain(&batch, &PlannerPolicy::default()).unwrap();
    assert_eq!(pp.selection, SelectionCarrier::TileLocal);
    let (_, st) = one.execute(&batch, &PlannerPolicy::default()).unwrap();
    assert_eq!(st.mask_materializations, 0);

    // Four measures × 6 partition passes: reuse ONE mask.
    let many = with_four(
        ReportPlan::over(src())
            .filter(sparse_selection())
            .pivot(&[fb()], &[fa()]),
        V,
    );
    let pp = many.explain(&batch, &PlannerPolicy::default()).unwrap();
    assert_eq!(pp.selection, SelectionCarrier::ReusedMask);
    let (r_reuse, st) = many.execute(&batch, &PlannerPolicy::default()).unwrap();
    assert_eq!(st.mask_materializations, 1);
    assert_eq!(st.mask_bytes, (fx.n.div_ceil(64) * 8) as u64);
    assert_matches_oracle(&fx, &many, &r_reuse, V);

    // Inertness twin: raising the threshold silences reuse; results identical.
    let no_reuse = PlannerPolicy {
        reuse_mask_min_programs: u64::MAX,
        ..PlannerPolicy::default()
    };
    let (r_eval, st2) = many.execute(&batch, &no_reuse).unwrap();
    assert_eq!(st2.mask_materializations, 0);
    assert!(
        st2.tile_mask_ops > st.tile_mask_ops,
        "re-evaluation costs more mask ops"
    );
    for m in r_reuse.measures() {
        assert_eq!(r_reuse.grand_total(m), r_eval.grand_total(m));
    }
}

// ── TEST 5 — SPARSE COORDINATE PRODUCT ──────────────────────────────────
#[test]
fn t5_high_cardinality_sparse_product_is_never_allocated_densely() {
    const D: u32 = 1_000_000;
    let n = 20_000;
    let mut r = Rng(15);
    let a_members: Vec<u32> = (0..40).map(|_| r.below(u64::from(D)) as u32).collect();
    let b_members: Vec<u32> = (0..30).map(|_| r.below(u64::from(D)) as u32).collect();
    let fx = Fixture {
        n,
        ords: vec![
            (
                A,
                (0..n).map(|_| a_members[r.below(40) as usize]).collect(),
                D,
            ),
            (
                B,
                (0..n).map(|_| b_members[r.below(30) as usize]).collect(),
                D,
            ),
        ],
        vals: vec![(V, (0..n).map(|_| r.below(100) as i32 - 50).collect())],
        masks: vec![],
    };
    let batch = fx.batch();
    let plan = with_four(ReportPlan::over(src()).pivot(&[fa()], &[fb()]), V);
    let pp = plan.explain(&batch, &PlannerPolicy::default()).unwrap();
    assert!(
        matches!(pp.accumulator, Accumulator::Sparse { product } if product == 1_000_000_000_000)
    );
    let (res, st) = plan.execute(&batch, &PlannerPolicy::default()).unwrap();
    let observed = fx.oracle(&plan, Some(V)).len();
    assert!(observed <= 1_200 && observed > 100, "observed {observed}");
    assert!(res.space().is_sparse());
    assert_eq!(res.space().stored_cells(), observed);
    assert_eq!(
        st.accumulator_bytes,
        (observed * 8 * 4) as u64,
        "result scales with observed cells"
    );
    // The only non-result buffers are ONE-dimensional domains, never the product.
    assert!(st.domain_buffer_bytes <= u64::from(D) * 8 * 4);
    assert_eq!(st.discovery_scans, 1);
    assert_matches_oracle(&fx, &plan, &res, V);
    // Absent vs empty: an unobserved coordinate is absent.
    assert_eq!(
        res.space().state(&[
            a_members[0],
            (0..D).find(|x| !b_members.contains(x)).unwrap()
        ]),
        CellState::Absent
    );
}

// ── TEST 6 — DENSE SMALL PRODUCT (+ the dense budget's inertness twin) ──
#[test]
fn t6_tiny_product_is_dense_and_the_budget_decides_not_the_result() {
    let fx = synthetic(4_000, &[3, 5], 16);
    let batch = fx.batch();
    let plan = with_four(ReportPlan::over(src()).pivot(&[fa()], &[fb()]), V);
    let pp = plan.explain(&batch, &PlannerPolicy::default()).unwrap();
    assert_eq!(pp.accumulator, Accumulator::Dense { cells: 15 });
    let (dense, st) = plan.execute(&batch, &PlannerPolicy::default()).unwrap();
    assert_eq!(st.accumulator_bytes, 15 * 8 * 4);
    assert_eq!(st.discovery_scans, 0);
    assert_matches_oracle(&fx, &plan, &dense, V);

    let tight = PlannerPolicy {
        dense_cell_budget: 14,
        ..PlannerPolicy::default()
    };
    let (sparse, _) = plan.execute(&batch, &tight).unwrap();
    assert!(
        sparse.space().is_sparse(),
        "lowering the budget flips the strategy"
    );
    for m in dense.measures() {
        for a in 0..3 {
            for b in 0..5 {
                let d = dense.value(m, &[], &[a], &[b]);
                let s = sparse.value(m, &[], &[a], &[b]);
                // Empty (dense) and absent (sparse) both read NULL for every
                // measure except COUNT, where empty is a real 0.
                if m.kind == MeasureKind::Count && s == CellValue::Null {
                    assert_eq!(d, CellValue::Int(0));
                } else {
                    assert_eq!(d, s);
                }
            }
        }
    }
}

// ── TEST 7 — DERIVED COORDINATE ─────────────────────────────────────────
#[test]
fn t7_derived_bucket_folds_without_a_materialized_lane() {
    let fx = synthetic(25_000, &[4], 17);
    let batch = fx.batch();
    // W ∈ [0,1000); buckets of 150 from 100 → [100,250) … 5 buckets up to 850;
    // values below 100 and at/above 850 are OUTSIDE the domain.
    let bucket = CoordSpec::Bucket {
        field: W,
        origin: 100,
        width: 150,
        count: 5,
    };
    let plan = with_four(
        ReportPlan::over(src()).pivot(std::slice::from_ref(&bucket), &[fa()]),
        V,
    );
    let pp = plan.explain(&batch, &PlannerPolicy::default()).unwrap();
    let d = pp.dims.iter().find(|d| d.coord == bucket).unwrap();
    assert!(matches!(d.provider, Provider::DerivedBucket { .. }));
    assert!(
        pp.fold_key.is_some_and(|k| pp.dims[k].coord == fa()),
        "the ordinal lane is the fold key"
    );
    let (res, st) = plan.execute(&batch, &PlannerPolicy::default()).unwrap();
    assert_eq!(st.mask_materializations, 0);
    assert_eq!(
        st.accumulator_bytes,
        5 * 4 * 8 * 4,
        "result-sized: 5×4 cells × 4 states"
    );
    assert_matches_oracle(&fx, &plan, &res, V);
    // Anti-vacuity: rows outside every bucket exist and are excluded.
    let outside = (0..fx.n)
        .filter(|&i| fx.coord(std::slice::from_ref(&bucket), i).is_none())
        .count();
    assert!(outside > fx.n / 10);
    assert_eq!(
        res.grand_total(&res.measures()[0]),
        CellValue::Int((fx.n - outside) as i64)
    );
}

// ── TEST 8 — TERMINAL MATERIALIZATION ───────────────────────────────────
// JSON and CSV here; HTML / Typst go through OGAR's composition path and are
// exercised by `lance-graph-report-ogar` (the same Grid, another emitter).
#[test]
fn t8_json_and_csv_differ_only_at_the_terminal() {
    use lance_graph_report::boundary::{CamLabels, Catalog, MemKv};
    use lance_graph_report::render::Terminal;
    let fx = synthetic(3_000, &[3, 4], 18);
    let batch = fx.batch();
    let plan = with_four(ReportPlan::over(src()).pivot(&[fa()], &[fb()]), V);
    let (res, _) = plan.execute(&batch, &PlannerPolicy::default()).unwrap();
    let (cam, kv, cat) = (CamLabels::default(), MemKv::default(), Catalog::default());
    let t = Terminal {
        cam: &cam,
        kv: &kv,
        catalog: &cat,
    };
    let addr = res.space().payload_addr();
    let grid = t.grid(&res);
    let (json, js) = t.json(&res);
    let (csv, cs) = t.csv(&res);
    assert_eq!(
        res.space().payload_addr(),
        addr,
        "rendering never touches the cells"
    );
    assert_eq!(js.cells, 3 * 4 * 4);
    assert_eq!(cs.cells, js.cells);
    // The grid keeps identity beside the labels (back-trace).
    assert_eq!(grid.column_keys, res.column_keys());
    assert_eq!(grid.pages[0].1[2].key, vec![2]);
    // Every CSV value line equals the grid value it came from, in order.
    let lines: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(lines.len(), 48);
    let mut k = 0;
    for row in &grid.pages[0].1 {
        for c in &row.cells {
            for v in c {
                let want = match v {
                    CellValue::Int(i) => i.to_string(),
                    CellValue::Real(r) => format!("{r}"),
                    CellValue::Null => String::new(),
                };
                assert!(
                    lines[k].ends_with(&format!(",{want}")),
                    "{} vs {want}",
                    lines[k]
                );
                assert!(json.contains(&want) || want.is_empty());
                k += 1;
            }
        }
    }
}

// ── TEST 9 — PLAN EQUIVALENCE (native Rust ≡ hand-built Quack) ──────────
#[test]
fn t9_native_plan_lowers_to_exactly_the_programs_quack_would_write() {
    let fx = synthetic(2_000, &[6, 9], 19);
    let batch = fx.batch();
    // "WHERE V >= 0 GROUP BY A, B  SUM(V)" — A (6) is a partition, B (9)
    // the fold key (wider). Pass 0 is A = 0.
    let plan = ReportPlan::over(src())
        .filter(Selection::cmp(V, CmpOp::Ge, Scalar::Int(0)))
        .pivot(&[fa()], &[fb()])
        .measure(Measure::of(MeasureKind::Sum, V));
    let pp = plan
        .explain(
            &batch,
            &PlannerPolicy {
                reuse_mask_min_programs: u64::MAX,
                ..PlannerPolicy::default()
            },
        )
        .unwrap();
    // Lanes: F0 → 0, F1 → 1, F100 → 2 (fixture order).
    let where_ = Filter::Cmp(Col(2), Cmp::GeI32(0));
    let pass0 = Filter::And(vec![where_, Filter::Cmp(Col(0), Cmp::EqU32(0))]);
    let quack: Vec<Program> = [
        Agg::GroupReduce {
            key: lance_graph_quack::GroupAddr::Local(Col(1)),
            agg: lance_graph_quack::GroupAgg::Count,
        },
        Agg::GroupSumI32 {
            key: Col(1),
            val: Col(2),
        },
    ]
    .into_iter()
    .map(|agg| {
        lower(&Query {
            filter: pass0.clone(),
            agg,
        })
        .unwrap()
    })
    .collect();
    assert_eq!(pp.first_pass, quack);
    // The same semantic request built in a different order normalizes to
    // the same physical key and the same programs.
    let plan_b = ReportPlan::over(src())
        .measure(Measure::of(MeasureKind::Sum, V))
        .axis(fb(), AxisRole::Column)
        .axis(fa(), AxisRole::Row)
        .filter(Selection::cmp(V, CmpOp::Ge, Scalar::Int(0)));
    assert_eq!(plan_b.physical_key(), plan.physical_key());
    assert_eq!(
        plan_b
            .explain(
                &batch,
                &PlannerPolicy {
                    reuse_mask_min_programs: u64::MAX,
                    ..PlannerPolicy::default()
                }
            )
            .unwrap()
            .first_pass,
        quack
    );
}

// ── knob inertness: pass and domain budgets can fire, and stay silent ───
#[test]
fn pass_budget_refuses_instead_of_running_slowly_and_is_silent_when_met() {
    let fx = synthetic(1_000, &[20, 30], 20);
    let batch = fx.batch();
    let plan = ReportPlan::over(src())
        .pivot(&[fa()], &[fb()])
        .measure(Measure::count());
    let tight = PlannerPolicy {
        pass_budget: 19,
        ..PlannerPolicy::default()
    };
    assert!(matches!(
        plan.execute(&batch, &tight),
        Err(ReportError::PassBudget {
            passes: 20,
            budget: 19
        })
    ));
    let ok = PlannerPolicy {
        pass_budget: 20,
        ..PlannerPolicy::default()
    };
    assert!(plan.execute(&batch, &ok).is_ok());
}

#[test]
fn domain_budget_moves_the_fold_key() {
    let fx = synthetic(1_000, &[20, 30], 21);
    let batch = fx.batch();
    let plan = ReportPlan::over(src())
        .pivot(&[fa()], &[fb()])
        .measure(Measure::count());
    let wide = plan.explain(&batch, &PlannerPolicy::default()).unwrap();
    assert_eq!(wide.dims[wide.fold_key.unwrap()].coord, fb());
    let narrow = plan
        .explain(
            &batch,
            &PlannerPolicy {
                domain_buffer_budget: 25,
                ..PlannerPolicy::default()
            },
        )
        .unwrap();
    assert_eq!(narrow.dims[narrow.fold_key.unwrap()].coord, fa());
    let (x, _) = plan.execute(&batch, &PlannerPolicy::default()).unwrap();
    let (y, _) = plan
        .execute(
            &batch,
            &PlannerPolicy {
                domain_buffer_budget: 25,
                ..PlannerPolicy::default()
            },
        )
        .unwrap();
    for a in 0..20 {
        for b in 0..30 {
            let m = &x.measures()[0];
            assert_eq!(x.value(m, &[], &[a], &[b]), y.value(m, &[], &[a], &[b]));
        }
    }
}

#[test]
fn stale_source_generation_fails_closed() {
    let fx = synthetic(100, &[2], 22);
    let batch = fx.batch();
    let plan = ReportPlan::over(SourceRef {
        id: SourceId(1),
        generation: 0,
    })
    .measure(Measure::count());
    assert_eq!(
        plan.execute(&batch, &PlannerPolicy::default()).unwrap_err(),
        ReportError::StaleSource
    );
}

#[test]
fn top_k_orders_presentation_without_touching_identity() {
    let fx = synthetic(5_000, &[8, 3], 23);
    let batch = fx.batch();
    let plan = ReportPlan::over(src())
        .pivot(&[fa()], &[fb()])
        .measure(Measure::of(MeasureKind::Sum, W));
    let (res, _) = plan.execute(&batch, &PlannerPolicy::default()).unwrap();
    let top = res
        .reinterpret(&plan.clone().top_k(TopK {
            measure: 0,
            k: 3,
            descending: true,
        }))
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(res.space(), top.space()));
    let keys = top.row_keys();
    assert_eq!(keys.len(), 3);
    let m = &res.measures()[0];
    let tot = |k: &Vec<u32>| match res.row_total(m, k) {
        CellValue::Int(i) => i,
        _ => i64::MIN,
    };
    let mut all: Vec<i64> = res.row_keys().iter().map(tot).collect();
    all.sort_unstable_by(|a, b| b.cmp(a));
    assert_eq!(keys.iter().map(tot).collect::<Vec<_>>(), all[..3].to_vec());
    // The same member reads the same value whatever its presented position.
    for k in &keys {
        assert_eq!(top.row_total(m, k), res.row_total(m, k));
    }
}

#[test]
fn mean_is_derived_from_sum_and_count_and_merges_correctly_in_totals() {
    let fx = synthetic(6_000, &[5], 24);
    let batch = fx.batch();
    let plan = ReportPlan::over(src())
        .axis(fa(), AxisRole::Row)
        .measure(Measure::of(MeasureKind::Mean, V))
        .measure(Measure::of(MeasureKind::Sum, V))
        .measure(Measure::count());
    let pp = plan.explain(&batch, &PlannerPolicy::default()).unwrap();
    assert_eq!(
        pp.states,
        vec![FoldState::Count, FoldState::Sum(V)],
        "MEAN adds no third fold"
    );
    let (res, _) = plan.execute(&batch, &PlannerPolicy::default()).unwrap();
    let (sum, cnt): (i64, i64) = (0..fx.n)
        .map(|i| (i64::from(fx.vals[0].1[i]), 1))
        .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    assert_eq!(
        res.grand_total(&res.measures()[0]),
        CellValue::Real(sum as f64 / cnt as f64)
    );
}
