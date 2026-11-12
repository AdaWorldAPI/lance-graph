//! D-DR-SELECT-K-SV-2: a selection is folded only under the joint witness of
//! the lanes it is read against. Every refusal fires on a realistic input,
//! before anything is borrowed; an accepted fold equals a row-at-a-time
//! oracle that calls nothing of the implementation.

use std::cell::Cell;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{
    Focus, Frame, FrameRefusal, FrameRevisions, FrameStore, GuardRefusal, Guards, Scope,
};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRefusal, LaneRequest,
    LaneSource, RouteAt, SchemaRev, SelAddr, SelectionSource,
};
use lance_graph_mask_risc::{ExecError, LaneKind, LaneRef, Value};
use lance_graph_report::{fold_bound_selection, BoundFold, BoundSelRefusal};

const CLASS: u32 = 0x0C1A_0001;
const VIEW_CLASS: ClassId = 0x0C1A;
const POSITION: u16 = 5;
const ROWS: usize = 500;
/// Field 0 and 1 are `i32` lanes; field 2 is a `u32` lane.
const AMOUNT: u8 = 0;
const QTY: u8 = 1;
const CODE: u8 = 2;

const REVS: FrameRevisions = FrameRevisions {
    source: 1,
    relation_map: 1,
    coordinates: 1,
};
const ORDERS: CoordSpace = CoordSpace {
    population: 1,
    coordinates: 1,
};
/// Another population of the SAME length: its row `i` is not ORDERS' row `i`.
const CUSTOMERS: CoordSpace = CoordSpace {
    population: 2,
    coordinates: 1,
};
// Hand-assigned consistency values. They carry no authority.
const READER: Scope = Scope { id: 7 };
const OTHER_READER: Scope = Scope { id: 8 };

struct View(Vec<FieldRef>);

impl View {
    fn three() -> View {
        View(vec![
            FieldRef::new("ord:amount", "Amount"),
            FieldRef::new("ord:quantity", "Quantity"),
            FieldRef::new("ord:code", "Code"),
        ])
    }
}

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == VIEW_CLASS {
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

struct Sel {
    space: CoordSpace,
    scope: Scope,
    positions: Vec<u32>,
}

/// A source owner: its lanes, its selections, and the store they live with.
struct Source {
    frames: FrameStore,
    scope: Scope,
    amount: Vec<i32>,
    qty: Vec<i32>,
    code: Vec<u32>,
    sels: Vec<Sel>,
    lane_borrows: Cell<usize>,
    position_borrows: Cell<usize>,
}

// Selection ids of the fixture.
const EVERY_THIRD: u16 = 0;
const BAG: u16 = 1;
const REVERSED: u16 = 2;
const OUT_OF_FRAME: u16 = 3;
const OF_CUSTOMERS: u16 = 4;
const FOR_OTHER_READER: u16 = 5;
const EMPTY: u16 = 6;
const MISSING: u16 = 99;

impl Source {
    /// `salt` changes the VALUES only; shape, revisions and ids are equal.
    fn new(salt: i32) -> Source {
        let third: Vec<u32> = (0..ROWS as u32).filter(|p| p % 3 == 1).collect();
        let sel = |space, scope, positions| Sel {
            space,
            scope,
            positions,
        };
        Source {
            frames: FrameStore::new(REVS),
            scope: READER,
            amount: (0..ROWS as i32)
                .map(|r| (r * 37 + salt) % 401 - 200)
                .collect(),
            qty: (0..ROWS as i32).map(|r| (r * 11 + salt) % 17).collect(),
            code: (0..ROWS as u32).collect(),
            sels: vec![
                sel(ORDERS, READER, third.clone()),
                sel(ORDERS, READER, vec![4, 4, 4, 9, 9, 250]),
                sel(ORDERS, READER, third.iter().rev().copied().collect()),
                sel(ORDERS, READER, vec![0, 1, ROWS as u32, 2, 9999]),
                sel(CUSTOMERS, READER, third.clone()),
                sel(ORDERS, OTHER_READER, third),
                sel(ORDERS, READER, Vec::new()),
            ],
            lane_borrows: Cell::new(0),
            position_borrows: Cell::new(0),
        }
    }

    fn borrows(&self) -> (usize, usize) {
        (self.lane_borrows.get(), self.position_borrows.get())
    }
}

impl LaneSource for Source {
    type Lane<'a> = LaneRef<'a>;

    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        frame: &Frame,
    ) -> Option<LaneAddr> {
        // Only this source's current frame addresses its lanes.
        self.frames.check(frame).ok()?;
        (class == CLASS && rev == SchemaRev(1) && field <= CODE).then_some(LaneAddr {
            space: ORDERS,
            lane: u16::from(field),
            scope: self.scope,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<LaneRef<'_>> {
        self.lane_borrows.set(self.lane_borrows.get() + 1);
        match addr.lane {
            0 => Some(LaneRef::I32(&self.amount)),
            1 => Some(LaneRef::I32(&self.qty)),
            2 => Some(LaneRef::U32(&self.code)),
            _ => None,
        }
    }

    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}

impl SelectionSource for Source {
    fn selection_of(&self, sel: u16, frame: &Frame) -> Option<SelAddr> {
        self.frames.check(frame).ok()?;
        let s = self.sels.get(usize::from(sel))?;
        Some(SelAddr {
            space: s.space,
            sel,
            scope: s.scope,
        })
    }

    fn borrow_positions(&self, addr: &SelAddr) -> Option<&[u32]> {
        self.position_borrows.set(self.position_borrows.get() + 1);
        let s = self.sels.get(usize::from(addr.sel))?;
        (s.space == addr.space && s.scope == addr.scope).then_some(s.positions.as_slice())
    }
}

struct World {
    store: BindStore,
    view: View,
    source: Source,
    binding: LaneBinding,
}

fn bind(store: &BindStore, view: &View, source: &Source, version: RouteAt) -> LaneBinding {
    let fields = WideFieldMask::from_positions(&[AMOUNT, QTY, CODE]);
    let frame = source.frames.seal();
    store
        .bind_lanes(
            view,
            &source.frames,
            source,
            LaneRequest {
                route: version,
                rev: SchemaRev(1),
                fields: &fields,
                frame: &frame,
                scope: READER,
                selection: ORDERS,
            },
        )
        .expect("the fixture binds")
}

fn world() -> World {
    let mut store = BindStore::new();
    let view = View::three();
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: 3,
                extent: Extent::DR64_CUBE,
            },
            Vec::new(),
        )
        .unwrap();
    let version = store.route(POSITION, CLASS).unwrap();
    let source = Source::new(0);
    let binding = bind(
        &store,
        &view,
        &source,
        RouteAt {
            position: POSITION,
            version,
        },
    );
    World {
        store,
        view,
        source,
        binding,
    }
}

fn run(
    w: &World,
    source: &Source,
    field: u8,
    sel: u16,
    fold: BoundFold,
) -> Result<Value, BoundSelRefusal> {
    fold_bound_selection(
        &w.binding,
        &w.store,
        &w.view,
        &source.frames,
        source,
        field,
        sel,
        fold,
    )
}

const SUM: BoundFold = BoundFold::SumI32;
const GT0: BoundFold = BoundFold::CountGtI32 { t: 0 };

/// The oracle: plain indexing over the fixture's own vectors.
fn oracle_sum(lane: &[i32], positions: &[u32]) -> i64 {
    let mut s = 0i64;
    for &p in positions {
        s += i64::from(lane[p as usize]);
    }
    s
}

fn oracle_gt(lane: &[i32], positions: &[u32], t: i32) -> usize {
    let mut c = 0;
    for &p in positions {
        if lane[p as usize] > t {
            c += 1;
        }
    }
    c
}

fn frame_refusal(r: Result<Value, BoundSelRefusal>) -> Option<FrameRefusal> {
    match r {
        Err(BoundSelRefusal::Binding(LaneRefusal::Frame(f))) => Some(f),
        _ => None,
    }
}

/// FAILS IF: a selection and a lane of one source under its current frame are
/// refused, or fold to anything but the oracle's answer; or the adapter reads
/// anything but the source's own lane and list, or reads either twice.
#[test]
fn a_selection_and_a_lane_of_one_frame_fold_to_the_oracle() {
    let w = world();
    let s = &w.source;
    for (field, lane) in [(AMOUNT, &s.amount), (QTY, &s.qty)] {
        let positions = &s.sels[usize::from(EVERY_THIRD)].positions;
        assert!(positions.len() > 100, "the selection is not trivial");
        assert_eq!(
            run(&w, s, field, EVERY_THIRD, SUM),
            Ok(Value::SumI64(oracle_sum(lane, positions)))
        );
        for t in [-200, -1, 0, 5, 199, 200] {
            assert_eq!(
                run(&w, s, field, EVERY_THIRD, BoundFold::CountGtI32 { t }),
                Ok(Value::Count(oracle_gt(lane, positions, t))),
                "field {field} t {t}"
            );
        }
    }
    assert_ne!(run(&w, s, AMOUNT, EVERY_THIRD, SUM), Ok(Value::SumI64(0)));
    assert_eq!(run(&w, s, AMOUNT, EMPTY, SUM), Ok(Value::SumI64(0)));

    // One lane borrow and one position borrow per fold: no second read.
    let before = s.borrows();
    run(&w, s, AMOUNT, EVERY_THIRD, SUM).unwrap();
    let after = s.borrows();
    assert_eq!((after.0 - before.0, after.1 - before.1), (1, 1));
}

/// FAILS IF: a repeated position is folded once, or list order changes a
/// COUNT or SUM. Nothing may turn the list into a set on the way.
#[test]
fn a_bag_keeps_multiplicity_and_order_does_not_matter() {
    let w = world();
    let s = &w.source;
    let bag = &s.sels[usize::from(BAG)].positions;
    assert_eq!(bag, &[4, 4, 4, 9, 9, 250]);
    assert_eq!(
        run(&w, s, AMOUNT, BAG, SUM),
        Ok(Value::SumI64(oracle_sum(&s.amount, bag)))
    );
    let once: Vec<u32> = vec![4, 9, 250];
    assert_ne!(
        oracle_sum(&s.amount, bag),
        oracle_sum(&s.amount, &once),
        "the fixture distinguishes a bag from its set"
    );
    assert_eq!(
        run(&w, s, QTY, BAG, BoundFold::CountGtI32 { t: -1 }),
        Ok(Value::Count(6)),
        "every occurrence is counted"
    );
    for fold in [SUM, GT0, BoundFold::CountGtI32 { t: 50 }] {
        assert_eq!(
            run(&w, s, AMOUNT, REVERSED, fold),
            run(&w, s, AMOUNT, EVERY_THIRD, fold)
        );
    }
}

/// FAILS IF: positions of one source are folded over the lanes of another
/// source of equal length, equal revisions and equal lane numbers — whichever
/// store the caller pairs it with. This is the "equal `n_rows` is not
/// evidence" case.
#[test]
fn positions_of_one_frame_are_never_read_against_another_sources_lanes() {
    let w = world();
    let twin = Source::new(13);
    assert_eq!(twin.amount.len(), w.source.amount.len());
    assert_eq!(twin.frames.revisions(), w.source.frames.revisions());
    assert_ne!(twin.amount, w.source.amount, "the twin holds other values");

    // The twin with its own store: the binding's frame is foreign to it.
    assert_eq!(
        frame_refusal(run(&w, &twin, AMOUNT, EVERY_THIRD, SUM)),
        Some(FrameRefusal::ForeignFrame)
    );
    // The twin paired with the bound source's store: the source does not own it.
    let mixed = fold_bound_selection(
        &w.binding,
        &w.store,
        &w.view,
        &w.source.frames,
        &twin,
        AMOUNT,
        EVERY_THIRD,
        SUM,
    );
    assert_eq!(frame_refusal(mixed), Some(FrameRefusal::ForeignFrame));
    // The bound source paired with the twin's store.
    let mixed = fold_bound_selection(
        &w.binding,
        &w.store,
        &w.view,
        &twin.frames,
        &w.source,
        AMOUNT,
        EVERY_THIRD,
        SUM,
    );
    assert_eq!(frame_refusal(mixed), Some(FrameRefusal::ForeignFrame));
    assert_eq!(twin.borrows(), (0, 0), "nothing of the twin was read");
    assert_eq!(w.source.borrows(), (0, 0), "a refusal borrows nothing");
}

/// FAILS IF: a binding survives its frame. Each revision that moves is named,
/// and re-sealing at unchanged numbers is still a new frame.
#[test]
fn a_frame_that_moved_refuses_the_fold_and_names_what_moved() {
    let cases = [
        (
            FrameRevisions { source: 2, ..REVS },
            FrameRefusal::SourceChanged,
        ),
        (
            FrameRevisions {
                relation_map: 2,
                ..REVS
            },
            FrameRefusal::MapChanged,
        ),
        (
            FrameRevisions {
                coordinates: 2,
                ..REVS
            },
            FrameRefusal::CoordinatesChanged,
        ),
        (REVS, FrameRefusal::FrameResealed),
    ];
    for (next, want) in cases {
        let mut w = world();
        assert!(run(&w, &w.source, AMOUNT, EVERY_THIRD, SUM).is_ok());
        let read = w.source.borrows();
        w.source.frames.advance(next);
        assert_eq!(
            frame_refusal(run(&w, &w.source, AMOUNT, EVERY_THIRD, SUM)),
            Some(want)
        );
        assert_eq!(w.source.borrows(), read, "the refusal borrowed nothing");
    }
}

/// FAILS IF: a selection in another row order is read against the lanes. The
/// two populations have the same length, so only the row-order witness can
/// tell them apart: local coordinates of two populations are not
/// interchangeable integers.
#[test]
fn a_selection_in_another_row_order_is_refused() {
    let w = world();
    let s = &w.source;
    assert_eq!(
        s.sels[usize::from(OF_CUSTOMERS)].positions,
        s.sels[usize::from(EVERY_THIRD)].positions,
        "the same numbers, meaning other rows"
    );
    assert_eq!(
        run(&w, s, AMOUNT, OF_CUSTOMERS, SUM),
        Err(BoundSelRefusal::Binding(LaneRefusal::SelectionSpace {
            sel: OF_CUSTOMERS,
            bound: ORDERS,
            selection: CUSTOMERS,
        }))
    );
    assert_eq!(s.borrows(), (0, 0));
}

/// FAILS IF: a selection or a lane serving another reader scope is read, or
/// a missing selection is read as empty. Scope here is a consistency check,
/// not a grant; see the module docs of `bound_sel`.
#[test]
fn another_scope_or_a_missing_selection_is_refused() {
    let w = world();
    let s = &w.source;
    assert_eq!(
        run(&w, s, AMOUNT, FOR_OTHER_READER, SUM),
        Err(BoundSelRefusal::Binding(LaneRefusal::SelectionScope {
            sel: FOR_OTHER_READER,
            reader: READER,
            selection: OTHER_READER,
        }))
    );
    assert_eq!(
        run(&w, s, AMOUNT, MISSING, SUM),
        Err(BoundSelRefusal::Binding(LaneRefusal::SelectionUnbound {
            sel: MISSING
        }))
    );
    assert_eq!(s.borrows(), (0, 0));

    // The source now serves its lanes to another reader: the bound mapping moved.
    let mut w = world();
    w.source.scope = OTHER_READER;
    assert!(matches!(
        run(&w, &w.source, AMOUNT, EVERY_THIRD, SUM),
        Err(BoundSelRefusal::Binding(LaneRefusal::MappingChanged {
            field: AMOUNT
        }))
    ));
    assert_eq!(w.source.borrows(), (0, 0));
}

/// The focus and the reader scope of a READ are `Guards`' checks, made
/// against the same store, before the fold. `LaneBinding` carries no focus.
/// FAILS IF: a changed focus or scope passes the existing guard.
#[test]
fn the_existing_guards_refuse_a_changed_focus_or_scope() {
    let w = world();
    let guards = Guards {
        frame: w.binding.frame().clone(),
        focus: Focus { revision: 3 },
        scope: READER,
    };
    let frames = &w.source.frames;
    assert_eq!(
        guards.check(frames, (Focus { revision: 3 }, READER)),
        Ok(())
    );
    assert_eq!(
        guards.check(frames, (Focus { revision: 4 }, READER)),
        Err(GuardRefusal::FocusChanged)
    );
    assert_eq!(
        guards.check(frames, (Focus { revision: 3 }, OTHER_READER)),
        Err(GuardRefusal::ScopeChanged)
    );
}

/// FAILS IF: a position that is not a row of the lane is folded, skipped or
/// read as zero, or the refusal names another place than the first.
#[test]
fn a_position_outside_the_lane_is_refused_and_named() {
    let w = world();
    assert_eq!(
        run(&w, &w.source, AMOUNT, OUT_OF_FRAME, SUM),
        Err(BoundSelRefusal::Fold(ExecError::SelOutOfRange {
            at: 2,
            position: ROWS as u32,
        }))
    );
    assert_eq!(
        run(&w, &w.source, AMOUNT, OUT_OF_FRAME, GT0),
        Err(BoundSelRefusal::Fold(ExecError::SelOutOfRange {
            at: 2,
            position: ROWS as u32,
        }))
    );
}

/// FAILS IF: a field that was not bound, or whose lane is not `i32`, folds.
#[test]
fn an_unbound_field_or_a_lane_of_another_kind_is_refused() {
    let w = world();
    assert_eq!(
        run(&w, &w.source, 3, EVERY_THIRD, SUM),
        Err(BoundSelRefusal::Binding(LaneRefusal::NotBound { field: 3 }))
    );
    assert_eq!(
        run(&w, &w.source, CODE, EVERY_THIRD, SUM),
        Err(BoundSelRefusal::Fold(ExecError::LaneKind {
            lane: 0,
            expected: LaneKind::I32,
            found: LaneKind::U32,
        }))
    );
}

/// FAILS IF: the adapter copies. The lane and the list it folds are the
/// source's own allocations.
#[test]
fn the_witnessed_operands_are_the_sources_own() {
    let w = world();
    let s = &w.source;
    let (lane, positions) = w
        .binding
        .selected(&w.store, &w.view, &s.frames, s, AMOUNT, BAG)
        .unwrap();
    let LaneRef::I32(values) = lane else {
        panic!("the amount lane is i32")
    };
    assert!(std::ptr::eq(values, s.amount.as_slice()));
    assert!(std::ptr::eq(
        positions,
        s.sels[usize::from(BAG)].positions.as_slice()
    ));
}
