//! The REFERENCE workload — replaceable acceptance data, never ontology.
//!
//! Business nouns (region / month / revenue / memo) exist ONLY in this file,
//! as catalog names and label text at the boundary. The crate under test
//! sees FieldIds and ordinals.
//!
//! Covers: raw → KV / CAM canonicalization; the fold touching neither CAM nor
//! KV; label resolution proportional to presented members; label rename not
//! changing aggregate identity; late hydration of a detail export; Quack
//! parity (the same answer from hand-built Quack programs) with SQL NULL
//! semantics for an empty cell.

mod common;

use common::Rng;
use lance_graph_contract::content_store::ContentSink;
use lance_graph_mask_risc::{execute_into, Foreign, LaneRef, Out, Planes, Scratch};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Query};
use lance_graph_report::boundary::{BoundaryCounters, CamLabels, Catalog, MemKv};
use lance_graph_report::render::Terminal;
use lance_graph_report::*;

const REGIONS: [&str; 8] = [
    "North", "South", "East", "West", "Central", "Coast", "Hills", "Isles",
];
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

struct World {
    batch: AbiBatch,
    cam: CamLabels,
    kv: MemKv,
    cat: Catalog,
    raw_region: Vec<&'static str>,
    raw_month: Vec<&'static str>,
    year: Vec<i32>,
    revenue: Vec<i32>,
}

/// Ingest: raw text → CAM ordinals / KV references, ONCE. Nothing textual
/// reaches the batch.
fn ingest(n: usize) -> World {
    let mut r = Rng(0x5eed);
    let mut raw_region = Vec::with_capacity(n);
    let mut raw_month = Vec::with_capacity(n);
    let mut year = Vec::with_capacity(n);
    let mut revenue = Vec::with_capacity(n);
    let mut memo = Vec::with_capacity(n);
    for i in 0..n {
        let reg = REGIONS[r.below(8) as usize];
        let y = if reg == "Isles" {
            2025
        } else {
            2025 + r.below(2) as i32
        };
        raw_region.push(reg);
        raw_month.push(MONTHS[r.below(12) as usize]);
        year.push(y);
        revenue.push(r.below(10_000) as i32);
        memo.push(format!("order {i} note {}", r.below(1 << 20)));
    }
    let (fr, fm, fy, fv, fo) = (FieldId(1), FieldId(2), FieldId(3), FieldId(4), FieldId(5));
    let cat = Catalog::default()
        .with("region", fr)
        .with("month", fm)
        .with("year", fy)
        .with("revenue", fv)
        .with("memo", fo);
    let mut cam = CamLabels::default();
    let mut kv = MemKv::default();
    // Canonical label order = first appearance; seed months in calendar order.
    for m in MONTHS {
        cam.intern(fm, m, &mut kv);
    }
    for g in REGIONS {
        cam.intern(fr, g, &mut kv);
    }
    let region = cam.canonicalize(fr, raw_region.iter().copied(), &mut kv);
    let month = cam.canonicalize(fm, raw_month.iter().copied(), &mut kv);
    let memo_refs = kv.persist(memo.iter().map(|s| s.as_bytes()));
    let batch = AbiBatch::new(SourceId(9), 4, n)
        .with_column(Column::coordinate(fr, region, cam.domain(fr)))
        .unwrap()
        .with_column(Column::coordinate(fm, month, cam.domain(fm)))
        .unwrap()
        .with_column(Column::value(fy, LaneData::I32(year.clone().into())))
        .unwrap()
        .with_column(Column::value(fv, LaneData::I32(revenue.clone().into())))
        .unwrap()
        .with_column(Column::value(fo, LaneData::U64(memo_refs)))
        .unwrap();
    World {
        batch,
        cam,
        kv,
        cat,
        raw_region,
        raw_month,
        year,
        revenue,
    }
}

/// What a UI / SQL adapter does: resolve names ONCE, then build an id-only plan.
fn sales_plan(w: &World) -> ReportPlan {
    let f = |n| w.cat.field(n).unwrap();
    ReportPlan::over(SourceRef {
        id: SourceId(9),
        generation: 4,
    })
    .filter(Selection::cmp(f("year"), CmpOp::Eq, Scalar::Int(2026)))
    .pivot(
        &[CoordSpec::Field(f("region"))],
        &[CoordSpec::Field(f("month"))],
    )
    .measure(Measure::of(MeasureKind::Sum, f("revenue")))
    .measure(Measure::count())
}

fn snapshot(c: &BoundaryCounters) -> [u64; 5] {
    [
        &c.kv_puts,
        &c.kv_derefs,
        &c.cam_lookups,
        &c.cam_insertions,
        &c.cam_resolutions,
    ]
    .map(BoundaryCounters::get)
}

#[test]
fn fold_touches_neither_cam_nor_kv_and_labels_resolve_per_presented_member() {
    let w = ingest(100_000);
    let plan = sales_plan(&w);
    let (cam0, kv0) = (snapshot(&w.cam.counters), snapshot(&w.kv.counters));
    let (res, _) = plan.execute(&w.batch, &PlannerPolicy::default()).unwrap();
    assert_eq!(snapshot(&w.cam.counters), cam0, "the fold did no CAM work");
    assert_eq!(snapshot(&w.kv.counters), kv0, "the fold did no KV work");

    let t = Terminal {
        cam: &w.cam,
        kv: &w.kv,
        catalog: &w.cat,
    };
    let (json, _) = t.json(&res);
    let cam1 = snapshot(&w.cam.counters);
    let kv1 = snapshot(&w.kv.counters);
    // 8 row members + 12 column members, each resolved once — not 100 000.
    assert_eq!(cam1[4] - cam0[4], 20);
    assert_eq!(kv1[1] - kv0[1], 20);
    assert!(json.contains("\"North\"") && json.contains("\"Dec\""));
    assert!(json.contains("SUM(revenue)"));
}

#[test]
fn quack_parity_and_null_semantics_for_the_reference_pivot() {
    let w = ingest(50_000);
    let plan = sales_plan(&w);
    let (res, _) = plan.execute(&w.batch, &PlannerPolicy::default()).unwrap();
    let (sum_m, cnt_m) = (&res.measures()[0], &res.measures()[1]);

    // Hand-built Quack: per region r, `year = 2026 AND region = r`,
    // GROUP BY month SUM(revenue) — executed straight on mask-risc.
    let n = w.batch.n_rows();
    let mut alpha = vec![u64::MAX; n.div_ceil(64)];
    if !n.is_multiple_of(64) {
        *alpha.last_mut().unwrap() = (1u64 << (n % 64)) - 1;
    }
    let mref: Vec<&[u64]> = vec![&alpha];
    let region: Vec<u32> = w
        .raw_region
        .iter()
        .map(|s| REGIONS.iter().position(|x| x == s).unwrap() as u32)
        .collect();
    let month: Vec<u32> = w
        .raw_month
        .iter()
        .map(|s| MONTHS.iter().position(|x| x == s).unwrap() as u32)
        .collect();
    let lanes = [
        LaneRef::U32(&region),
        LaneRef::U32(&month),
        LaneRef::I32(&w.year),
        LaneRef::I32(&w.revenue),
    ];
    let planes = Planes {
        n_rows: w.batch.n_rows(),
        masks: &mref,
        lanes: &lanes,
    };
    for r in 0..8u32 {
        let q = Query {
            filter: Filter::And(vec![
                Filter::Cmp(Col(2), Cmp::EqI32(2026)),
                Filter::Cmp(Col(0), Cmp::EqU32(r)),
            ]),
            agg: Agg::GroupSumI32 {
                key: Col(1),
                val: Col(3),
            },
        };
        let prog = lower(&q).unwrap();
        let mut out = vec![0i64; 12];
        let mut s = Scratch::for_program(&prog, w.batch.n_rows()).unwrap();
        execute_into(&prog, &planes, &Foreign::NONE, &mut s, Out::I64(&mut out)).unwrap();
        for m in 0..12u32 {
            // Independent per-row oracle, from the RAW strings.
            let rows: Vec<usize> = (0..w.year.len())
                .filter(|&i| {
                    w.year[i] == 2026
                        && w.raw_region[i] == REGIONS[r as usize]
                        && w.raw_month[i] == MONTHS[m as usize]
                })
                .collect();
            let oracle_sum: i64 = rows.iter().map(|&i| i64::from(w.revenue[i])).sum();
            assert_eq!(out[m as usize], oracle_sum, "quack vs oracle");
            let got = res.value(sum_m, &[], &[r], &[m]);
            if rows.is_empty() {
                // SQL: SUM over no rows is NULL; COUNT is 0. Quack's
                // coalescing terminal reads 0 — the report layer pins NULL
                // from the COUNT state, identically for every adapter.
                assert_eq!(got, CellValue::Null);
                assert_eq!(res.value(cnt_m, &[], &[r], &[m]), CellValue::Int(0));
            } else {
                assert_eq!(got, CellValue::Int(out[m as usize]), "report vs quack");
            }
        }
    }
    // Anti-vacuity: the empty region (Isles never occurs in 2026) is empty.
    assert_eq!(res.row_total(sum_m, &[7]), CellValue::Null);
    assert!(matches!(res.row_total(sum_m, &[0]), CellValue::Int(x) if x > 0));
}

#[test]
fn renaming_a_label_changes_no_aggregate_identity() {
    let mut w = ingest(20_000);
    let plan = sales_plan(&w);
    let (res, _) = plan.execute(&w.batch, &PlannerPolicy::default()).unwrap();
    let key = plan.physical_key();
    let addr = res.space().payload_addr();
    let fr = w.cat.field("region").unwrap();
    assert!(w.cam.rename(fr, 0, "Nordland", &mut w.kv));
    assert_eq!(
        plan.physical_key(),
        key,
        "the plan holds ordinals, not labels"
    );
    let again = res.reinterpret(&plan).unwrap();
    assert_eq!(
        again.space().payload_addr(),
        addr,
        "cached aggregate still valid"
    );
    let t = Terminal {
        cam: &w.cam,
        kv: &w.kv,
        catalog: &w.cat,
    };
    let (json, _) = t.json(&again);
    assert!(json.contains("\"Nordland\"") && !json.contains("\"North\""));
    assert_eq!(w.cam.ordinal(fr, "Nordland"), Some(0));
}

#[test]
fn detail_export_filters_ids_first_and_hydrates_only_kept_rows() {
    let w = ingest(40_000);
    let f = |n| w.cat.field(n).unwrap();
    // An adapter resolves the label literal to its ordinal at the boundary.
    let coast = w.cam.ordinal(f("region"), "Coast").unwrap();
    let plan = ReportPlan::over(SourceRef {
        id: SourceId(9),
        generation: 4,
    })
    .filter(Selection::cmp(
        f("region"),
        CmpOp::Eq,
        Scalar::Ordinal(coast),
    ))
    .filter(Selection::cmp(f("revenue"), CmpOp::Ge, Scalar::Int(9_950)));
    let kept = (0..w.year.len())
        .filter(|&i| w.raw_region[i] == "Coast" && w.revenue[i] >= 9_950)
        .count();
    assert!(kept > 0 && kept * 100 < w.year.len(), "kept {kept}");
    let (cam0, kv0) = (snapshot(&w.cam.counters), snapshot(&w.kv.counters));
    let t = Terminal {
        cam: &w.cam,
        kv: &w.kv,
        catalog: &w.cat,
    };
    let (csv, st) = t
        .detail_csv(&plan, &w.batch, &[f("region"), f("revenue"), f("memo")])
        .unwrap();
    assert_eq!(csv.lines().count(), kept + 1);
    assert_eq!(st.cells as usize, kept * 3);
    // One CAM resolution per kept row (region), one KV deref per kept row for
    // the label text plus one for the memo — proportional to OUTPUT rows.
    assert_eq!(snapshot(&w.cam.counters)[4] - cam0[4], kept as u64);
    assert_eq!(snapshot(&w.kv.counters)[1] - kv0[1], 2 * kept as u64);
    assert!(csv
        .lines()
        .skip(1)
        .all(|l| l.starts_with("Coast,") && l.contains(",order ")));
}

#[test]
fn raw_values_live_in_kv_once_and_duplicates_dedup() {
    let mut kv = MemKv::default();
    let refs = kv.persist(["a", "b", "a"].iter().map(|s| s.as_bytes()));
    assert_eq!(
        refs[0], refs[2],
        "content-addressed: one identity per value (S10)"
    );
    assert_ne!(refs[0], refs[1]);
    let id = kv.put_str("a");
    assert_eq!(id.0, refs[0]);
}
