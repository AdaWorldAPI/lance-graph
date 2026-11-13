//! D-VAS-A3: the mailbox socket issues its own position lists, under its
//! publication's frame and in the mailbox's row order.
//!
//! Oracle: sorts, filters and counts written here over the owner's public
//! arrays. Real `MailboxSoA<64>`.

use causal_edge::CausalEdge64;
use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{FrameRefusal, Scope};
use lance_graph_dt64::{
    view_class, BindStore, ClassSchema, Extent, LaneBinding, LaneRefusal, LaneRequest, LaneSource,
    RouteAt, SchemaRev, SelectionSource,
};
use lance_graph_mailbox_lanes::{Column, Keep, MailboxLanes, SelectionRefusal};
use lance_graph_mask_risc::LaneRef;

const N: usize = 64;
type Owner = MailboxSoA<N>;
const ROWS: usize = 48;
const CLASS: u32 = 0x0A1A_0003;
const POSITION: u16 = 11;
const READER: Scope = Scope { id: 3 };
const META: u8 = 0;
const EDGES: u8 = 1;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn owner(id: u32, rows: usize) -> Box<Owner> {
    let mut m: Box<Owner> = Box::new(MailboxSoA::new(id, 3, 1.0));
    m.set_populated(rows);
    for r in 0..rows {
        m.set_meta(r, MetaWord(mix(u64::from(id) << 32 | r as u64) as u32));
        m.set_edge(r, CausalEdge64(mix(1_000 + r as u64)));
    }
    m
}

struct View(Vec<FieldRef>);

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == view_class(CLASS) {
            &self.0
        } else {
            &[]
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

struct World {
    store: BindStore,
    view: View,
    route: RouteAt,
}

fn world() -> World {
    let view = View(vec![
        FieldRef::new(Column::Meta.iri(), "meta"),
        FieldRef::new(Column::Edges.iri(), "edges"),
    ]);
    let mut store = BindStore::new();
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: 2,
                extent: Extent::DR256_SQ_X64,
            },
            vec![],
        )
        .unwrap();
    let version = store.route(POSITION, CLASS).unwrap();
    World {
        store,
        view,
        route: RouteAt {
            position: POSITION,
            version,
        },
    }
}

fn bind(w: &World, lanes: &MailboxLanes<'_, N>) -> LaneBinding {
    let frame = lanes.seal();
    w.store
        .bind_lanes(
            &w.view,
            lanes.frame_store(),
            lanes,
            LaneRequest {
                route: w.route,
                rev: SchemaRev(1),
                fields: &WideFieldMask::from_positions(&[META, EDGES]),
                frame: &frame,
                scope: READER,
                selection: lanes.space(),
            },
        )
        .unwrap()
}

fn selected<'s>(
    w: &World,
    b: &LaneBinding,
    lanes: &'s MailboxLanes<'_, N>,
    sel: u16,
) -> Result<(&'s [u32], &'s [u32]), LaneRefusal> {
    let (lane, positions) = b.selected(&w.store, &w.view, lanes.frame_store(), lanes, META, sel)?;
    let LaneRef::U32(values) = lane else {
        panic!("a u32 lane")
    };
    Ok((values, positions))
}

/// FAILS IF: an issued order is not the owner's rows sorted by the column,
/// a filter keeps or drops the wrong row, a concatenation loses a repeat, or
/// the positions are copied per read.
#[test]
fn order_filter_and_bag_are_the_owners_rows() {
    let m = owner(7, ROWS);
    let w = world();
    let mut lanes = MailboxLanes::new(m.published(), READER);
    let median = {
        let mut v: Vec<u32> = (0..ROWS).map(|r| m.meta[r].0).collect();
        v.sort_unstable();
        u64::from(v[ROWS / 2])
    };
    let order = lanes.issue_order(Column::Meta).unwrap();
    let low = lanes.issue_where(Column::Meta, Keep::Lt(median)).unwrap();
    let bag = lanes.issue_concat(low, order).unwrap();
    assert_eq!((order, low, bag), (0, 1, 2));
    let b = bind(&w, &lanes);

    // Order: an independent sort of (value, row) pairs.
    let mut pairs: Vec<(u32, u32)> = (0..ROWS as u32)
        .map(|r| (m.meta[r as usize].0, r))
        .collect();
    pairs.sort_unstable();
    let expect: Vec<u32> = pairs.iter().map(|p| p.1).collect();
    let (values, positions) = selected(&w, &b, &lanes, order).unwrap();
    assert_eq!(positions, expect);
    assert_ne!(positions, (0..ROWS as u32).collect::<Vec<_>>(), "reordered");
    assert!(positions
        .windows(2)
        .all(|p| values[p[0] as usize] < values[p[1] as usize]));
    assert_eq!(values.as_ptr().cast::<u8>(), m.meta.as_ptr().cast::<u8>());
    let (_, again) = selected(&w, &b, &lanes, order).unwrap();
    assert_eq!(again.as_ptr(), positions.as_ptr(), "lent, not rebuilt");

    // Filter: row order, both outcomes present.
    let kept: Vec<u32> = (0..ROWS as u32)
        .filter(|&r| u64::from(m.meta[r as usize].0) < median)
        .collect();
    let (_, positions) = selected(&w, &b, &lanes, low).unwrap();
    assert_eq!(positions, kept);
    assert_eq!(kept.len(), ROWS / 2);

    // Bag: every kept row twice, every other row once, in list order.
    let (values, positions) = selected(&w, &b, &lanes, bag).unwrap();
    assert_eq!(positions.len(), kept.len() + ROWS);
    assert_eq!(&positions[..kept.len()], kept);
    assert_eq!(&positions[kept.len()..], expect);
    for r in 0..ROWS as u32 {
        let n = positions.iter().filter(|&&p| p == r).count();
        assert_eq!(n, if kept.contains(&r) { 2 } else { 1 }, "row {r}");
    }
    let sum: u64 = positions
        .iter()
        .map(|&p| u64::from(values[p as usize]))
        .sum();
    let oracle: u64 = (0..ROWS)
        .map(|r| u64::from(m.meta[r].0) * if kept.contains(&(r as u32)) { 2 } else { 1 })
        .sum();
    assert_eq!(sum, oracle);

    // The other comparisons, against the same oracle shape.
    for keep in [
        Keep::Ge(median),
        Keep::Eq(u64::from(m.meta[5].0)),
        Keep::Ne(u64::from(m.meta[5].0)),
    ] {
        let sel = lanes.issue_where(Column::Meta, keep).unwrap();
        let b = bind(&w, &lanes);
        let (_, positions) = selected(&w, &b, &lanes, sel).unwrap();
        let expect: Vec<u32> = (0..ROWS as u32)
            .filter(|&r| {
                let v = u64::from(m.meta[r as usize].0);
                match keep {
                    Keep::Ge(t) => v >= t,
                    Keep::Eq(t) => v == t,
                    Keep::Ne(t) => v != t,
                    Keep::Lt(t) => v < t,
                }
            })
            .collect();
        assert_eq!(positions, expect, "{keep:?}");
    }
}

/// FAILS IF: a selection is read under another publication, another
/// mailbox, an unissued name, or a foreign address.
#[test]
fn a_selection_belongs_to_its_publication() {
    let (a, other) = (owner(7, ROWS), owner(8, ROWS));
    let w = world();
    let mut la = MailboxLanes::new(a.published(), READER);
    let sel = la.issue_order(Column::Meta).unwrap();
    let b = bind(&w, &la);
    assert!(selected(&w, &b, &la, sel).is_ok());
    assert_eq!(
        selected(&w, &b, &la, 9).err(),
        Some(LaneRefusal::SelectionUnbound { sel: 9 })
    );

    // Another mailbox, and a second socket over the same publication.
    let mut lb = MailboxLanes::new(other.published(), READER);
    lb.issue_order(Column::Meta).unwrap();
    let again = MailboxLanes::new(a.published(), READER);
    for foreign in [&lb, &again] {
        assert_eq!(
            selected(&w, &b, foreign, sel).err(),
            Some(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
        );
    }

    // Asked directly: another source's frame, another source's address.
    assert!(la.selection_of(sel, &la.seal()).is_some());
    assert_eq!(la.selection_of(sel, &lb.seal()), None);
    let addr = la.selection_of(sel, &la.seal()).unwrap();
    assert!(la.borrow_positions(&addr).is_some());
    assert!(lb.borrow_positions(&addr).is_none());
    let mut stranger = addr;
    stranger.scope = Scope { id: 99 };
    assert!(la.borrow_positions(&stranger).is_none());

    assert_eq!(
        la.issue_concat(sel, 77),
        Err(SelectionRefusal::Unknown { sel: 77 })
    );
    assert_eq!(la.frames().revisions(), la.frame_store().revisions());
}

/// FAILS IF: the 65,537th selection is issued under a name already taken.
#[test]
fn selection_names_are_never_reissued() {
    let m = owner(7, 4);
    let mut lanes = MailboxLanes::new(m.published(), READER);
    let first = lanes.issue_where(Column::Meta, Keep::Ne(0)).unwrap();
    for n in 1..=u16::MAX {
        let keep = Keep::Eq(u64::from(m.meta[usize::from(n) % 4].0));
        assert_eq!(lanes.issue_where(Column::Meta, keep), Ok(n));
    }
    assert_eq!(
        lanes.issue_order(Column::Meta),
        Err(SelectionRefusal::NamesExhausted)
    );
    let frame = lanes.seal();
    let positions = |sel| {
        let addr = lanes.selection_of(sel, &frame).unwrap();
        lanes.borrow_positions(&addr).unwrap().to_vec()
    };
    assert_eq!(positions(first), [0, 1, 2, 3]);
    assert_eq!(positions(u16::MAX), [3]);
}

/// Review finding. FAILS IF: a selection address issued by one socket
/// resolves to another socket's selection of the same name.
#[test]
fn a_selection_address_belongs_to_one_socket() {
    let m = owner(7, ROWS);
    let mut first = MailboxLanes::new(m.published(), READER);
    let mut twin = MailboxLanes::new(m.published(), READER);
    let order = first.issue_order(Column::Meta).unwrap();
    let filter = twin.issue_where(Column::Meta, Keep::Lt(1 << 31)).unwrap();
    assert_eq!(order, filter, "both sockets start at name 0");
    let addr = first.selection_of(order, &first.seal()).unwrap();
    assert_eq!(first.borrow_positions(&addr).unwrap().len(), ROWS);
    assert!(twin.borrow_positions(&addr).is_none());
    assert_ne!(Some(addr), twin.selection_of(filter, &twin.seal()));
}

/// D-VAS-A4, as it stands. The shipped selection fold (`execute_sel`,
/// through `fold_bound_selection`) admits `i32` lanes only, and the mailbox
/// lends `u32` and `u64`. FAILS IF: a fold over a mailbox lane returns a
/// value instead of the typed refusal, or the witness itself is refused.
#[test]
fn the_shipped_fold_refuses_the_mailboxs_lane_kinds() {
    use lance_graph_mask_risc::ExecError;
    use lance_graph_report::bound_sel::{fold_bound_selection, BoundFold, BoundSelRefusal};
    let m = owner(7, ROWS);
    let w = world();
    let mut lanes = MailboxLanes::new(m.published(), READER);
    let sel = lanes.issue_order(Column::Meta).unwrap();
    let b = bind(&w, &lanes);
    for field in [META, EDGES] {
        // The witness holds: lane and positions are both lent.
        assert!(b
            .selected(&w.store, &w.view, lanes.frame_store(), &lanes, field, sel)
            .is_ok());
        for fold in [BoundFold::SumI32, BoundFold::CountGtI32 { t: 0 }] {
            let refused = fold_bound_selection(
                &b,
                &w.store,
                &w.view,
                lanes.frame_store(),
                &lanes,
                field,
                sel,
                fold,
            );
            assert!(
                matches!(
                    refused,
                    Err(BoundSelRefusal::Fold(ExecError::LaneKind { .. }))
                ),
                "field {field}: {refused:?}"
            );
        }
    }
}
