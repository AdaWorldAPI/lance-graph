//! The convergence proof: reports, objects and documents compose through ONE
//! OGAR machinery. Synthetic nouns only (A, B, M, objects O1/O2).
//!
//! W3 rotation through composition · W4 mixed composition · W5 recombination
//! by reference · W6 Live vs Revision · W7 terminal render (Typst via OGAR).

use std::cell::Cell;
use std::sync::Arc;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::content_store::{ContentId, ContentSink, ContentStore};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::selection::{NamedView, RailGraph, ViewId, ViewRegistry};
use lance_graph_report::boundary::{CamLabels, Catalog, MemKv};
use lance_graph_report::render::Terminal;
use lance_graph_report::{
    AbiBatch, AxisRole, Column, CoordSpec, FieldId, LaneData, Measure, MeasureKind, PlannerPolicy,
    ReportPlan, ReportResult, SourceId, SourceRef,
};
use lance_graph_report_ogar::{emit_typst, Orientation, ReportSource};
use ogar_doc_ir::compose::{
    parse_ogar_uri, DocCompose, DocNode, NodeId, ObjectSlot, SnapshotRef, DOC_COMPOSE_VERSION,
};
use ogar_doc_ir::resolve::{
    resolve_doc, DocObjectSource, ResolvedBlock, ResolvedGrid, SlotOutcome,
};

const A: FieldId = FieldId(1);
const B: FieldId = FieldId(2);
const M: FieldId = FieldId(3);
const OBJ: ClassId = 10;
const REPORT: ClassId = 11;

// ── an existing object world: two objects whose field values live in KV ──

struct ObjGraph;
impl RailGraph for ObjGraph {
    type Key = u32;
    fn class_of(&self, _k: u32) -> ClassId {
        OBJ
    }
    fn present_mask(&self, _k: u32) -> WideFieldMask {
        WideFieldMask::from(0b11)
    }
    fn rail_target(&self, _k: u32, _p: u8) -> Option<u32> {
        None
    }
}

struct Objects<'k> {
    graph: ObjGraph,
    refs: Vec<[ContentId; 2]>,
    kv: &'k MemKv,
    view: ViewId,
    derefs: Cell<u64>,
}
impl DocObjectSource for Objects<'_> {
    type Graph = ObjGraph;
    fn graph(&self) -> &ObjGraph {
        &self.graph
    }
    fn lookup(
        &self,
        t: &ogar_doc_ir::compose::ObjectRef,
        _m: &ogar_doc_ir::compose::ResolutionMode,
    ) -> Option<u32> {
        let i: u32 = t.id.strip_prefix('o')?.parse().ok()?;
        ((i as usize) < self.refs.len() && t.class == "obj").then_some(i)
    }
    fn value_of(&self, k: u32, p: u8) -> Option<String> {
        self.derefs.set(self.derefs.get() + 1);
        let id = *self.refs.get(k as usize)?.get(p as usize)?;
        self.kv
            .resolve(id)
            .map(|b| String::from_utf8_lossy(b).into_owned())
    }
    fn view_by_name(&self, name: &str) -> Option<ViewId> {
        (name == "obj.card").then_some(self.view)
    }
    fn view_for_class(&self, _c: ClassId) -> Option<ViewId> {
        None
    }
}

struct Labels;
impl ClassView for Labels {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        use std::sync::OnceLock;
        static F: OnceLock<Vec<FieldRef>> = OnceLock::new();
        if class == OBJ {
            F.get_or_init(|| {
                vec![
                    FieldRef::new("o:title", "title"),
                    FieldRef::new("o:body", "body"),
                ]
            })
        } else {
            &[]
        }
    }
    fn template(&self, _c: ClassId) -> DisplayTemplate {
        DisplayTemplate::Card
    }
    fn dolce_category_id(&self, _c: ClassId) -> u8 {
        0
    }
}

// ── an analytical result: fold(M) over coordinates {A, B} ──

fn report(
    cam: &mut CamLabels,
    kv: &mut MemKv,
    generation: u32,
    n: usize,
    seed: u64,
) -> ReportResult {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let a_raw: Vec<String> = (0..n).map(|_| format!("a{}", next() % 4)).collect();
    let b_raw: Vec<String> = (0..n).map(|_| format!("b{}", next() % 7)).collect();
    let m: Vec<i32> = (0..n).map(|_| (next() % 100) as i32).collect();
    for i in 0..4 {
        cam.intern(A, &format!("a{i}"), kv);
    }
    for i in 0..7 {
        cam.intern(B, &format!("b{i}"), kv);
    }
    let a = cam.canonicalize(A, a_raw.iter().map(String::as_str), kv);
    let b = cam.canonicalize(B, b_raw.iter().map(String::as_str), kv);
    let batch = AbiBatch::new(SourceId(3), generation, n)
        .with_column(Column::coordinate(A, a, 4))
        .unwrap()
        .with_column(Column::coordinate(B, b, 7))
        .unwrap()
        .with_column(Column::value(M, LaneData::I32(m.into())))
        .unwrap();
    let plan = ReportPlan::over(SourceRef {
        id: SourceId(3),
        generation,
    })
    .axis(CoordSpec::Field(A), AxisRole::Row)
    .axis(CoordSpec::Field(B), AxisRole::Column)
    .measure(Measure::of(MeasureKind::Sum, M));
    plan.execute(&batch, &PlannerPolicy::default()).unwrap().0
}

fn slot(uri: &str, view: &str, fallback: Option<SnapshotRef>) -> DocNode {
    let (target, resolution) = parse_ogar_uri(uri).unwrap();
    DocNode::ObjectSlot {
        slot: ObjectSlot {
            target,
            class_view: view.into(),
            field_mask: 0,
            wide_mask_words: vec![],
            resolution,
            fallback,
        },
    }
}

/// Document ├ Section "S1" [slot…] … — built from (heading, slot) pairs.
fn compose(sections: Vec<(&str, DocNode)>) -> DocCompose {
    let mut nodes = vec![DocNode::Document { children: vec![] }];
    let mut kids = Vec::new();
    for (h, s) in sections {
        let hid = nodes.len() as u32;
        nodes.push(DocNode::Text { text: h.into() });
        let sid = nodes.len() as u32;
        nodes.push(s);
        let sec = nodes.len() as u32;
        nodes.push(DocNode::Section {
            heading: Some(NodeId(hid)),
            children: vec![NodeId(sid)],
        });
        kids.push(NodeId(sec));
    }
    nodes[0] = DocNode::Document { children: kids };
    DocCompose {
        version: DOC_COMPOSE_VERSION.into(),
        nodes,
        root: NodeId(0),
    }
}

fn grids(doc: &ogar_doc_ir::resolve::ResolvedDoc) -> Vec<&ResolvedGrid> {
    doc.blocks
        .iter()
        .filter_map(|b| match b {
            ResolvedBlock::Slot(s) => match &s.outcome {
                SlotOutcome::Grid { grid, .. } => Some(grid),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

struct World {
    cam: CamLabels,
    kv: MemKv,
    cat: Catalog,
    refs: Vec<[ContentId; 2]>,
}

fn world() -> World {
    let (cam, mut kv) = (CamLabels::default(), MemKv::default());
    // Two existing objects; their bodies live in KV, addressed by ContentId.
    let refs = vec![
        [kv.put_str("First object"), kv.put_str("body of O0")],
        [kv.put_str("Second object"), kv.put_str("body of O1")],
    ];
    World {
        cam,
        kv,
        cat: Catalog::default().with("M", M),
        refs,
    }
}

#[test]
fn rotation_through_composition_rescans_nothing_and_shares_the_cells() {
    let mut w = world();
    let r = report(&mut w.cam, &mut w.kv, 1, 20_000, 7);
    let payload = r.space().payload_addr();
    let mut reg = ViewRegistry::new();
    let obj_view = reg.register(NamedView::new(
        OBJ,
        WideFieldMask::from(0b11),
        DisplayTemplate::Card,
    ));
    let objects = Objects {
        graph: ObjGraph,
        refs: w.refs.clone(),
        kv: &w.kv,
        view: obj_view,
        derefs: Cell::new(0),
    };
    let t = Terminal {
        cam: &w.cam,
        kv: &w.kv,
        catalog: &w.cat,
    };
    let mut src = ReportSource::new(&objects, REPORT, t);
    src.publish(7, r.clone());
    src.register_grid_view(
        &mut reg,
        "r.a_by_b",
        Orientation {
            rows: vec![CoordSpec::Field(A)],
            columns: vec![CoordSpec::Field(B)],
        },
    );
    src.register_grid_view(
        &mut reg,
        "r.b_by_a",
        Orientation {
            rows: vec![CoordSpec::Field(B)],
            columns: vec![CoordSpec::Field(A)],
        },
    );

    let doc = compose(vec![
        (
            "View 1",
            slot("ogar://lance-graph/report/7@live", "r.a_by_b", None),
        ),
        (
            "View 2",
            slot("ogar://lance-graph/report/7@live", "r.b_by_a", None),
        ),
    ]);
    let resolved = resolve_doc(&doc, &src, &Labels, &reg, 4).unwrap();
    let g = grids(&resolved);
    assert_eq!(g.len(), 2);
    let (v1, v2) = (g[0], g[1]);
    assert_eq!((v1.rows.keys.len(), v1.columns.keys.len()), (4, 7));
    assert_eq!((v2.rows.keys.len(), v2.columns.keys.len()), (7, 4));
    for a in 0..4 {
        for b in 0..7 {
            assert_eq!(
                v1.cell(a, b, 0),
                v2.cell(b, a, 0),
                "same cell, rotated view"
            );
        }
    }
    // Identity survives: labels are CAM text, keys are the ordinals.
    assert_eq!(v1.rows.labels[2], vec!["a2".to_string()]);
    assert_eq!(v1.rows.keys[2], vec![2]);
    // Zero population work: the source holds no batch, the aggregate payload
    // is the one computed above, and projections are counted.
    assert_eq!(r.space().payload_addr(), payload);
    assert_eq!(src.stats.grids_projected.get(), 2);
    assert_eq!(
        Arc::strong_count(r.space()),
        2,
        "held by `r` and the published entry only"
    );
}

#[test]
fn mixed_recombination_stores_references_and_resolves_only_at_render() {
    let mut w = world();
    let small = report(&mut w.cam, &mut w.kv, 1, 1_000, 8);
    let big = report(&mut w.cam, &mut w.kv, 1, 200_000, 9);
    let mut reg = ViewRegistry::new();
    let obj_view = reg.register(NamedView::new(
        OBJ,
        WideFieldMask::from(0b11),
        DisplayTemplate::Card,
    ));
    let objects = Objects {
        graph: ObjGraph,
        refs: w.refs.clone(),
        kv: &w.kv,
        view: obj_view,
        derefs: Cell::new(0),
    };

    // Document ├ projection(O0) ├ projection(R) └ projection(O1)
    let doc_for = |id: u64| {
        compose(vec![
            ("Object A", slot("ogar://app/obj/o0@live", "obj.card", None)),
            (
                "Analysis",
                slot(
                    &format!("ogar://lance-graph/report/{id}@live"),
                    "r.a_by_b",
                    None,
                ),
            ),
            ("Object B", slot("ogar://app/obj/o1@live", "obj.card", None)),
        ])
    };
    // The composition is references + projection choices: its bytes do not
    // depend on how many rows fed the report, nor on the objects' bodies.
    let (d1, d2) = (
        serde_json::to_string(&doc_for(1)).unwrap(),
        serde_json::to_string(&doc_for(2)).unwrap(),
    );
    assert_eq!(d1.len(), d2.len());
    assert!(
        !d1.contains("body of") && !d1.contains("a0"),
        "no object or report content copied in"
    );

    let kv_before =
        w.kv.counters
            .kv_derefs
            .load(std::sync::atomic::Ordering::Relaxed);
    let t = Terminal {
        cam: &w.cam,
        kv: &w.kv,
        catalog: &w.cat,
    };
    let mut src = ReportSource::new(&objects, REPORT, t);
    src.publish(1, small);
    src.publish(2, big.clone());
    src.register_grid_view(
        &mut reg,
        "r.a_by_b",
        Orientation {
            rows: vec![CoordSpec::Field(A)],
            columns: vec![CoordSpec::Field(B)],
        },
    );
    assert_eq!(
        objects.derefs.get(),
        0,
        "building the composition dereferenced nothing"
    );
    assert_eq!(
        w.kv.counters
            .kv_derefs
            .load(std::sync::atomic::Ordering::Relaxed),
        kv_before
    );

    let resolved = resolve_doc(&doc_for(2), &src, &Labels, &reg, 4).unwrap();
    let kinds: Vec<&str> = resolved
        .blocks
        .iter()
        .filter_map(|b| match b {
            ResolvedBlock::Slot(s) => Some(match s.outcome {
                SlotOutcome::Resolved { .. } => "object",
                SlotOutcome::Grid { .. } => "grid",
                _ => "other",
            }),
            _ => None,
        })
        .collect();
    assert_eq!(kinds, ["object", "grid", "object"]);
    // Resolution dereferenced exactly the two objects' two fields, and the
    // report's labels: 4 row + 7 column members (+ one KV read each).
    assert_eq!(objects.derefs.get(), 4);
    let kv_after =
        w.kv.counters
            .kv_derefs
            .load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(kv_after - kv_before, 4 + 11);

    // W7: one OGAR emitter renders all three kinds.
    let typ = emit_typst(&resolved);
    assert!(typ.contains("First object") && typ.contains("Second object"));
    assert!(typ.contains("columns: 8"), "row header + 7 columns");
    assert!(typ.contains("= Analysis"));
}

#[test]
fn live_and_revision_resolve_through_ogar_resolution_modes() {
    let mut w = world();
    let g1 = report(&mut w.cam, &mut w.kv, 1, 5_000, 11);
    let g2 = report(&mut w.cam, &mut w.kv, 2, 5_000, 12); // the source moved on
    let mut reg = ViewRegistry::new();
    let obj_view = reg.register(NamedView::new(
        OBJ,
        WideFieldMask::from(0b11),
        DisplayTemplate::Card,
    ));
    let objects = Objects {
        graph: ObjGraph,
        refs: w.refs.clone(),
        kv: &w.kv,
        view: obj_view,
        derefs: Cell::new(0),
    };
    let t = Terminal {
        cam: &w.cam,
        kv: &w.kv,
        catalog: &w.cat,
    };
    let mut src = ReportSource::new(&objects, REPORT, t);
    src.publish(5, g1.clone());
    src.publish(5, g2.clone());
    src.register_grid_view(
        &mut reg,
        "r.a_by_b",
        Orientation {
            rows: vec![CoordSpec::Field(A)],
            columns: vec![CoordSpec::Field(B)],
        },
    );

    let doc = compose(vec![
        (
            "Live",
            slot("ogar://lance-graph/report/5@live", "r.a_by_b", None),
        ),
        (
            "Pinned",
            slot("ogar://lance-graph/report/5@revision:1", "r.a_by_b", None),
        ),
        (
            "Gone",
            slot("ogar://lance-graph/report/5@revision:9", "r.a_by_b", None),
        ),
        (
            "Snapshot",
            slot(
                &format!("ogar://lance-graph/report/5@sha256:{}", "ab".repeat(32)),
                "r.a_by_b",
                Some(SnapshotRef {
                    content_sha256: [0xAB; 32],
                }),
            ),
        ),
    ]);
    let resolved = resolve_doc(&doc, &src, &Labels, &reg, 4).unwrap();
    let g = grids(&resolved);
    assert_eq!(g.len(), 2);
    assert_ne!(
        g[0].cells, g[1].cells,
        "live and pinned are different results"
    );
    let m = &g1.measures()[0];
    let pinned_cell = match g1.value(m, &[], &[0], &[0]) {
        lance_graph_report::CellValue::Int(i) => i.to_string(),
        _ => String::new(),
    };
    assert_eq!(
        g[1].cell(0, 0, 0),
        Some(pinned_cell.as_str()),
        "revision:1 is the generation-1 result"
    );
    let outcomes: Vec<&SlotOutcome> = resolved
        .blocks
        .iter()
        .filter_map(|b| match b {
            ResolvedBlock::Slot(s) => Some(&s.outcome),
            _ => None,
        })
        .collect();
    assert_eq!(
        *outcomes[2],
        SlotOutcome::Unresolvable,
        "an unknown revision fails closed"
    );
    assert!(
        matches!(outcomes[3], SlotOutcome::Fallback { .. }),
        "snapshot answers from the slot's fallback"
    );
}

#[test]
fn a_label_rename_changes_the_render_not_the_aggregate() {
    let mut w = world();
    let r = report(&mut w.cam, &mut w.kv, 1, 5_000, 13);
    let payload = r.space().payload_addr();
    assert!(w.cam.rename(A, 0, "renamed-a0", &mut w.kv));
    let mut reg = ViewRegistry::new();
    let obj_view = reg.register(NamedView::new(
        OBJ,
        WideFieldMask::from(0b11),
        DisplayTemplate::Card,
    ));
    let objects = Objects {
        graph: ObjGraph,
        refs: w.refs.clone(),
        kv: &w.kv,
        view: obj_view,
        derefs: Cell::new(0),
    };
    let t = Terminal {
        cam: &w.cam,
        kv: &w.kv,
        catalog: &w.cat,
    };
    let mut src = ReportSource::new(&objects, REPORT, t);
    src.publish(1, r.clone());
    src.register_grid_view(
        &mut reg,
        "r.a_by_b",
        Orientation {
            rows: vec![CoordSpec::Field(A)],
            columns: vec![CoordSpec::Field(B)],
        },
    );
    let doc = compose(vec![(
        "R",
        slot("ogar://lance-graph/report/1@live", "r.a_by_b", None),
    )]);
    let resolved = resolve_doc(&doc, &src, &Labels, &reg, 4).unwrap();
    let g = grids(&resolved);
    assert_eq!(g[0].rows.labels[0], vec!["renamed-a0".to_string()]);
    assert_eq!(
        g[0].rows.keys[0],
        vec![0],
        "identity is the ordinal, unchanged"
    );
    assert_eq!(r.space().payload_addr(), payload);
}
