//! The class, its view and route, and binding demanded fields through DT64.

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::Scope;
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneBinding, LaneRefusal, LaneRequest, LaneSource,
    RouteAt, SchemaRev,
};

use crate::source::Socket;

/// The lab's class.
pub const CLASS: u32 = 0x5115_0001;
/// The ClassView key of the class.
pub const VIEW_CLASS: ClassId = 0x5115;
/// DR position the class is routed at.
pub const POSITION: u16 = 5;
/// The reader scope every lane serves.
pub const READER: Scope = Scope { id: 9 };
/// The row order of the canonical population.
pub const SPACE_A: CoordSpace = CoordSpace {
    population: 1,
    coordinates: 1,
};
/// Another row order of the same population.
pub const SPACE_OTHER: CoordSpace = CoordSpace {
    population: 1,
    coordinates: 2,
};
/// Fields of the class.
pub const FIELDS: usize = 32;

/// A ClassView: field `f` is `isa:f{f}`.
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

impl World {
    pub fn new() -> World {
        World::with_fields(FIELDS, "isa:f")
    }

    /// The class with `count` fields named `{prefix}{f}`. Two worlds with
    /// different prefixes bind the class over different bases.
    pub fn with_fields(count: usize, prefix: &str) -> World {
        World::with_extent(count, prefix, Extent::DR64_CUBE)
    }

    /// The class with `count` fields under `extent` (DR256 property axes are
    /// needed past 64 fields).
    pub fn with_extent(count: usize, prefix: &str, extent: Extent) -> World {
        let mut store = BindStore::new();
        let view = View(
            (0..count)
                .map(|f| FieldRef::new(format!("{prefix}{f}"), format!("F{f}")))
                .collect(),
        );
        store
            .register(
                &view,
                CLASS,
                ClassSchema {
                    schema_rev: SchemaRev(1),
                    field_count: count,
                    extent,
                },
                Vec::new(),
            )
            .expect("the lab class registers");
        let version = store.route(POSITION, CLASS).expect("the lab class routes");
        World {
            store,
            view,
            route: RouteAt {
                position: POSITION,
                version,
            },
        }
    }

    /// Cold: bind exactly `fields` under `selection`'s row order, at the
    /// source's current frame.
    pub fn bind(
        &self,
        socket: &Socket<'_>,
        fields: &[u8],
        selection: CoordSpace,
    ) -> Result<LaneBinding, LaneRefusal> {
        let mask = WideFieldMask::from_positions(fields);
        let frame = socket.frames().seal();
        self.store.bind_lanes(
            &self.view,
            socket.frames(),
            socket,
            LaneRequest {
                route: self.route,
                rev: SchemaRev(1),
                fields: &mask,
                frame: &frame,
                scope: READER,
                selection,
            },
        )
    }
}

impl Default for World {
    fn default() -> World {
        World::new()
    }
}
