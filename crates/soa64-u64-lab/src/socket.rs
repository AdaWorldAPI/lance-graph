//! The DT64 lane socket as the coordinate-space witness for [`SoaU64`].
//!
//! A [`U64Source`] says which of its own lanes holds each field and in whose
//! row order. [`bind_demanded`] joins that with the class route and a sealed
//! frame into a [`LaneBinding`] (cold, addresses only); the sweeps then borrow
//! each demanded lane once and fold positionally (hot, no per-element check).

use std::cell::Cell;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRefusal, LaneRequest,
    LaneSource, RouteAt, SchemaRev,
};

use crate::arms::{SoaU64, K};
use crate::fixture::{value, Fixture, FIELDS};

/// The 64-field class.
pub const CLASS: u32 = 0x5064_0001;
/// The ClassView key of the class.
pub const VIEW_CLASS: ClassId = 0x5064;
/// DR position the class is routed at.
pub const POSITION: u16 = 3;
/// The reader scope every lane serves by default.
pub const READER: Scope = Scope { id: 7 };
/// The row order of the canonical (source-owner sealed) population.
pub const SPACE_A: CoordSpace = CoordSpace {
    population: 1,
    coordinates: 1,
};
/// Another population's row order.
pub const SPACE_OTHER: CoordSpace = CoordSpace {
    population: 2,
    coordinates: 1,
};

/// A 64-field ClassView: field `f` is `soa64:f{f}`.
pub struct View(Vec<FieldRef>);

impl View {
    pub fn sixty_four() -> View {
        View(
            (0..FIELDS)
                .map(|f| FieldRef::new(format!("soa64:f{f}"), format!("F{f}")))
                .collect(),
        )
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

/// Store, view, sealed frame and route of the class.
pub struct World {
    pub store: BindStore,
    pub view: View,
    pub frames: FrameStore,
    pub frame: Frame,
    pub route: RouteAt,
}

impl World {
    pub fn new() -> World {
        let mut store = BindStore::new();
        store
            .register(
                CLASS,
                ClassSchema {
                    schema_rev: SchemaRev(1),
                    field_count: FIELDS,
                    extent: Extent::DR64_CUBE,
                },
                Vec::new(),
            )
            .unwrap();
        let version = store.route(POSITION, CLASS).unwrap();
        let frames = FrameStore::new(FrameRevisions {
            source: 1,
            relation_map: 1,
            coordinates: 1,
        });
        let frame = frames.seal();
        World {
            store,
            view: View::sixty_four(),
            frames,
            frame,
            route: RouteAt {
                position: POSITION,
                version,
            },
        }
    }
}

impl Default for World {
    fn default() -> World {
        World::new()
    }
}

/// Source owner of a [`SoaU64`]: lane `f` holds field `f`; each lane declares
/// its row order in `spaces`. The counters record what was actually borrowed.
pub struct U64Source<'a> {
    pub soa: &'a SoaU64,
    pub spaces: [CoordSpace; FIELDS],
    pub scope: Scope,
    /// Number of `borrow` calls.
    pub borrows: Cell<u64>,
    /// Bit `f` set once lane `f` has been borrowed.
    pub borrowed: Cell<u64>,
}

impl<'a> U64Source<'a> {
    /// All lanes in [`SPACE_A`], serving [`READER`].
    pub fn new(soa: &'a SoaU64) -> U64Source<'a> {
        U64Source {
            soa,
            spaces: [SPACE_A; FIELDS],
            scope: READER,
            borrows: Cell::new(0),
            borrowed: Cell::new(0),
        }
    }
}

impl<'s> LaneSource for U64Source<'s> {
    type Lane<'b>
        = &'b [u64]
    where
        Self: 'b;

    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        if class != CLASS || rev != SchemaRev(1) || usize::from(field) >= FIELDS {
            return None;
        }
        Some(LaneAddr {
            space: self.spaces[usize::from(field)],
            lane: u16::from(field),
            scope: self.scope,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<&[u64]> {
        let f = usize::from(addr.lane);
        if f >= FIELDS {
            return None;
        }
        self.borrows.set(self.borrows.get() + 1);
        self.borrowed.set(self.borrowed.get() | (1u64 << f));
        Some(self.soa.lane(f))
    }
}

/// Cold: bind exactly the demanded `fields` under `selection`'s row order.
pub fn bind_demanded(
    world: &World,
    source: &U64Source<'_>,
    fields: &[u8],
    selection: CoordSpace,
) -> Result<LaneBinding, LaneRefusal> {
    let mask = WideFieldMask::from_positions(fields);
    world.store.bind_lanes(
        &world.view,
        &world.frames,
        source,
        LaneRequest {
            route: world.route,
            rev: SchemaRev(1),
            fields: &mask,
            frame: &world.frame,
            scope: READER,
            selection,
        },
    )
}

/// SUM(B) where A == K through the socket: lanes 0 and 1 are borrowed once
/// each (cold), then folded positionally (hot, no per-element check).
pub fn sum_where_via_socket(
    world: &World,
    binding: &LaneBinding,
    source: &U64Source<'_>,
) -> Result<u128, LaneRefusal> {
    let a = binding.lane(&world.store, &world.view, &world.frames, source, 0)?;
    let b = binding.lane(&world.store, &world.view, &world.frames, source, 1)?;
    let mut s = 0u128;
    for (&x, &y) in a.iter().zip(b) {
        if x == K {
            s += y as u128;
        }
    }
    Ok(s)
}

/// Fold `acc = acc.wrapping_add(v ^ f)` over every row of each listed field,
/// borrowing each lane once.
pub fn fold_via_socket(
    world: &World,
    binding: &LaneBinding,
    source: &U64Source<'_>,
    fields: &[u8],
) -> Result<u128, LaneRefusal> {
    let mut acc = 0u64;
    for &f in fields {
        let lane = binding.lane(&world.store, &world.view, &world.frames, source, f)?;
        let fx = u64::from(f);
        for &v in lane {
            acc = acc.wrapping_add(v ^ fx);
        }
    }
    Ok(acc as u128)
}

/// Adversarial lane: field 1 in the reverse row order. It belongs to
/// [`SPACE_OTHER`]; reading it positionally against field 0 is a gather
/// disguised as a scan.
pub fn reordered_b(fx: &Fixture) -> Vec<u64> {
    (0..fx.n).rev().map(|i| value(fx.seed, i, 1)).collect()
}
