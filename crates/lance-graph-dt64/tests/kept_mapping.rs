//! D-DRDT-SPLIT-S7 over value lanes: a kept mapping is a pure address
//! binding. Over changed contents with unchanged positions the assignment is
//! unchanged and the values follow the source; over changed positions the
//! mapping no longer matches the source and the read is refused (plan
//! `dr-widefieldmask-dt-classview-v1.md` §1: "A pure address / property
//! binding stays valid across a content change while its mapping and its
//! interpretation stay valid").

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Frame, FrameStore, LocalMap, MapTable, Scope, Width};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRefusal, LaneRequest,
    LaneSource, RouteAt, SchemaRev,
};

const CLASS: u32 = 0x5E70_0001;
const VIEW_CLASS: ClassId = 0x5E70;
const POSITION: u16 = 6;
const READER: Scope = Scope { id: 2 };
const TILE: u32 = 0;

struct View(Vec<FieldRef>);

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

/// The owner's lanes, by row; the row order is named by `order`.
struct Owner {
    names: Vec<i32>,
    mails: Vec<i32>,
    order: u64,
}

/// A source over the owner's lanes, under the table's own frames.
struct Source<'a> {
    owner: &'a Owner,
    frames: &'a FrameStore,
}

impl LaneSource for Source<'_> {
    type Lane<'s>
        = &'s [i32]
    where
        Self: 's;
    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        (class == CLASS && rev == SchemaRev(1) && field < 2).then_some(LaneAddr {
            space: CoordSpace {
                population: 1,
                coordinates: self.owner.order,
            },
            lane: u16::from(field),
            scope: READER,
        })
    }
    fn borrow(&self, addr: &LaneAddr) -> Option<&[i32]> {
        (addr.space.coordinates == self.owner.order).then_some(())?;
        match addr.lane {
            0 => Some(&self.owner.names),
            1 => Some(&self.owner.mails),
            _ => None,
        }
    }
    fn frames(&self) -> &FrameStore {
        self.frames
    }
}

fn table() -> MapTable {
    let mut t = MapTable::new(1, 1);
    let ids: Vec<u64> = (1000..1004).collect();
    t.advance(
        1,
        1,
        vec![(TILE, Some(LocalMap::prepare(Width::W256, &ids).unwrap()))],
    );
    t
}

fn store() -> (BindStore, View, RouteAt) {
    let view = View(vec![
        FieldRef::new("dir:displayName", "Name"),
        FieldRef::new("dir:mail", "Mail"),
    ]);
    let mut store = BindStore::new();
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: 2,
                extent: Extent::DR64_CUBE,
            },
            Vec::new(),
        )
        .unwrap();
    let version = store.route(POSITION, CLASS).unwrap();
    (
        store,
        view,
        RouteAt {
            position: POSITION,
            version,
        },
    )
}

fn bind(
    store: &BindStore,
    view: &View,
    route: RouteAt,
    t: &MapTable,
    owner: &Owner,
) -> LaneBinding {
    let source = Source {
        owner,
        frames: t.frames(),
    };
    let frame = t.seal();
    store
        .bind_lanes(
            view,
            t.frames(),
            &source,
            LaneRequest {
                route,
                rev: SchemaRev(1),
                fields: &WideFieldMask::from_positions(&[0, 1]),
                frame: &frame,
                scope: READER,
                selection: CoordSpace {
                    population: 1,
                    coordinates: owner.order,
                },
            },
        )
        .unwrap()
}

/// Every bound lane through the kept mapping `id`, copied out for the test.
fn replay(
    store: &BindStore,
    view: &View,
    t: &MapTable,
    owner: &Owner,
    id: lance_graph_dt64::KeptId,
) -> Result<Vec<Vec<i32>>, LaneRefusal> {
    let frame = t.seal();
    let kept = t
        .kept::<LaneBinding>(&frame, id)
        .expect("kept under this version");
    let source = Source {
        owner,
        frames: t.frames(),
    };
    kept.borrow_all(store, view, t.frames(), &source)?
        .map(|item| item.map(|(_, lane)| lane.to_vec()))
        .collect()
}

/// FAILS IF: a kept mapping does not follow a content change at unchanged
/// positions (it serves old values, or refuses although nothing it names
/// moved), or still reads after the positions changed.
#[test]
fn a_kept_mapping_follows_contents_and_refuses_moved_positions() {
    let mut t = table();
    let (store, view, route) = store();
    let mut owner = Owner {
        names: vec![10, 11, 12, 13],
        mails: vec![20, 21, 22, 23],
        order: 1,
    };
    let id = {
        let binding = bind(&store, &view, route, &t, &owner);
        binding.keep_in(&mut t).unwrap()
    };
    assert_eq!(
        replay(&store, &view, &t, &owner, id).unwrap(),
        [vec![10, 11, 12, 13], vec![20, 21, 22, 23]]
    );

    // Contents change, positions do not: the same mapping reads the new values.
    owner.mails[2] = 99;
    assert_eq!(
        replay(&store, &view, &t, &owner, id).unwrap(),
        [vec![10, 11, 12, 13], vec![20, 21, 99, 23]],
        "the values follow the source"
    );

    // The rows are reordered: the mapping no longer matches the source.
    owner.names.reverse();
    owner.mails.reverse();
    owner.order = 2;
    assert!(matches!(
        replay(&store, &view, &t, &owner, id),
        Err(LaneRefusal::MappingChanged { .. })
    ));
}
