//! The store-network class: its field basis (the DT64 menu), its route, and
//! binding the fields a question demands through the DT64 lane socket.

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Extent, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, LaneBinding, LaneRefusal, LaneRequest, LaneSource, RouteAt,
    SchemaRev,
};

use crate::fixture::{CATEGORIES, EQUIPMENT, ITEMS, SKILLS};
use crate::source::Network;

/// The lab's class (lab-local id; no OGAR concept is minted for it).
pub const CLASS: u32 = 0x5116_0001;
/// The ClassView key of the class.
pub const VIEW_CLASS: ClassId = 0x5116;
/// DR position the class is routed at.
pub const POSITION: u16 = 7;
/// The scope the canonical lanes serve. A consistency witness, NOT an
/// authorization (locality contract, boundary 5).
pub const READER: Scope = Scope { id: 9 };
/// The canonical row order.
pub const SPACE_A: CoordSpace = CoordSpace {
    population: 64,
    coordinates: 1,
};
/// The same 64 stores in another row order.
pub const SPACE_REVERSED: CoordSpace = CoordSpace {
    population: 64,
    coordinates: 2,
};

pub const STOCK0: usize = 0;
pub const KNOWN0: usize = STOCK0 + CATEGORIES;
pub const COST0: usize = KNOWN0 + CATEGORIES;
pub const STAFF0: usize = COST0 + CATEGORIES;
pub const EQUIP0: usize = STAFF0 + SKILLS;
pub const OFFERED0: usize = EQUIP0 + EQUIPMENT;
pub const SLOTS: usize = OFFERED0 + ITEMS;
pub const MINUTES: usize = SLOTS + 1;
pub const STORE_ID: usize = MINUTES + 1;
/// Fields of the class: 211, so the class needs a 256-wide property axis.
pub const FIELD_COUNT: usize = STORE_ID + 1;

/// What a field is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Stock(usize),
    Known(usize),
    Cost(usize),
    Staff(usize),
    Equipment(usize),
    Offered(usize),
    DeliverySlots,
    DeliveryMinutes,
    StoreId,
}

/// The kind of field `f`.
pub fn field(f: u8) -> Kind {
    let f = usize::from(f);
    match f {
        _ if f < KNOWN0 => Kind::Stock(f - STOCK0),
        _ if f < COST0 => Kind::Known(f - KNOWN0),
        _ if f < STAFF0 => Kind::Cost(f - COST0),
        _ if f < EQUIP0 => Kind::Staff(f - STAFF0),
        _ if f < OFFERED0 => Kind::Equipment(f - EQUIP0),
        _ if f < SLOTS => Kind::Offered(f - OFFERED0),
        SLOTS => Kind::DeliverySlots,
        MINUTES => Kind::DeliveryMinutes,
        _ => Kind::StoreId,
    }
}

fn predicate(k: Kind) -> String {
    match k {
        Kind::Stock(i) => format!("ops:stock/{i}"),
        Kind::Known(i) => format!("ops:stockKnown/{i}"),
        Kind::Cost(i) => format!("ops:unitCost/{i}"),
        Kind::Staff(i) => format!("ops:staff/{i}"),
        Kind::Equipment(i) => format!("ops:equipment/{i}"),
        Kind::Offered(i) => format!("ops:offers/{i}"),
        Kind::DeliverySlots => "ops:deliverySlots".into(),
        Kind::DeliveryMinutes => "ops:deliveryMinutes".into(),
        Kind::StoreId => "ops:storeId".into(),
    }
}

/// The ClassView of the store-network class.
pub struct View(Vec<FieldRef>);

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

/// Store, view and route of the class.
pub struct World {
    pub store: BindStore,
    pub view: View,
    pub route: RouteAt,
}

impl Default for World {
    fn default() -> World {
        World::new()
    }
}

impl World {
    pub fn new() -> World {
        let view = View(
            (0..FIELD_COUNT)
                .map(|f| FieldRef::new(predicate(field(f as u8)), format!("F{f}")))
                .collect(),
        );
        let mut store = BindStore::new();
        store
            .register(
                &view,
                CLASS,
                ClassSchema {
                    schema_rev: SchemaRev(1),
                    field_count: FIELD_COUNT,
                    extent: Extent::DR256_SQ_X64,
                },
                Vec::new(),
            )
            .expect("the network class registers");
        let version = store.route(POSITION, CLASS).expect("the class routes");
        World {
            store,
            view,
            route: RouteAt {
                position: POSITION,
                version,
            },
        }
    }

    /// Cold: bind exactly `fields` for `reader` under `selection`, at the
    /// source's current frame.
    pub fn bind(
        &self,
        source: &Network,
        fields: &[u8],
        selection: CoordSpace,
        reader: Scope,
    ) -> Result<LaneBinding, LaneRefusal> {
        let mask = WideFieldMask::from_positions(fields);
        let frame = source.frames().seal();
        self.store.bind_lanes(
            &self.view,
            source.frames(),
            source,
            LaneRequest {
                route: self.route,
                rev: SchemaRev(1),
                fields: &mask,
                frame: &frame,
                scope: reader,
                selection,
            },
        )
    }
}
