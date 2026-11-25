//! A tile source and the class it binds through, shared by the tests and the
//! bench. One class with one Codec9 field `x`.

use lance_graph_contract::class_view::{ClassId, ClassView};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneSource, PresenceSource, RouteAt,
    SchemaRev,
};

pub const CLASS: u32 = 0x0D2C_0001;
pub const POSITION: u16 = 5;
pub const READER: Scope = Scope { id: 7 };
pub const ROWS: CoordSpace = CoordSpace {
    population: 1,
    coordinates: 1,
};
pub const FIELD: u8 = 0;

pub struct View {
    class: ClassId,
    fields: Vec<FieldRef>,
}

impl View {
    pub fn new() -> View {
        View {
            class: (CLASS >> 16) as ClassId,
            fields: vec![FieldRef::new("t:x".to_string(), "x")],
        }
    }
}

impl Default for View {
    fn default() -> Self {
        Self::new()
    }
}

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == self.class {
            &self.fields
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

/// One DR tile's Codec9 lane: `u8` magnitudes, padded to whole blocks with
/// untouched rows, and the polarity words. It owns its frame store.
pub struct Tile {
    pub mags: Vec<u8>,
    pub pos: Vec<u64>,
    frames: FrameStore,
    revision: u64,
}

impl Tile {
    /// A tile over codewords `0..512` (magnitude in the low byte, polarity
    /// bit 8), padded with untouched rows to a multiple of 64.
    pub fn from_codes(codes: &[u16], revision: u64) -> Tile {
        let blocks = codes.len().div_ceil(64);
        let mut mags = vec![0u8; blocks * 64];
        let mut pos = vec![0u64; blocks];
        for (i, &c) in codes.iter().enumerate() {
            mags[i] = (c & 0xff) as u8;
            pos[i / 64] |= u64::from(c >> 8 & 1) << (i % 64);
        }
        Tile {
            mags,
            pos,
            frames: FrameStore::new(FrameRevisions {
                source: revision,
                relation_map: 1,
                coordinates: 1,
            }),
            revision,
        }
    }

    /// Relay the tile with new codes: the store advances, so every part
    /// folded before is stale.
    pub fn relay(&mut self, codes: &[u16]) {
        let next = Tile::from_codes(codes, self.revision + 1);
        self.mags = next.mags;
        self.pos = next.pos;
        self.revision += 1;
        self.frames.advance(FrameRevisions {
            source: self.revision,
            relation_map: 1,
            coordinates: 1,
        });
    }

    pub fn blocks(&self) -> usize {
        self.pos.len()
    }
}

impl LaneSource for Tile {
    type Lane<'a> = &'a [u8];
    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        _field: u8,
        basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        (class == CLASS && rev == SchemaRev(1) && basis.predicate_iri == "t:x").then_some(
            LaneAddr {
                space: ROWS,
                lane: 0,
                scope: READER,
            },
        )
    }
    fn borrow(&self, addr: &LaneAddr) -> Option<&[u8]> {
        (addr.lane == 0).then_some(self.mags.as_slice())
    }
    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}

impl PresenceSource for Tile {
    fn borrow_presence(&self, addr: &LaneAddr) -> Option<&[u64]> {
        (addr.lane == 0).then_some(self.pos.as_slice())
    }
}

/// A store with the class registered and routed at [`POSITION`].
pub fn store() -> (BindStore, RouteAt) {
    let mut store = BindStore::new();
    store
        .register(
            &View::new(),
            CLASS,
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: 1,
                extent: Extent::DR64_CUBE,
            },
            Vec::new(),
        )
        .expect("register");
    let version = store.route(POSITION, CLASS).expect("route");
    (
        store,
        RouteAt {
            position: POSITION,
            version,
        },
    )
}

/// The value of codeword `c`, or `None` for untouched.
pub fn value_of(c: u16) -> Option<i64> {
    let (m, p) = (i64::from(c & 0xff), c >> 8 & 1 == 1);
    match (p, m) {
        (false, 0) => None,
        (true, m) => Some(m),
        (false, m) => Some(-m),
    }
}
