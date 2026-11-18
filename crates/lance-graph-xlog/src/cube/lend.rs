//! A cube relation lent through DT64's lane socket (D-XLOG-W7-2 step 4a).
//!
//! A relation is two lanes of equal length and a live plane. Row `i` is the
//! tuple `x[i] : y[i]`, read in place: no pair is built. [`CubeLanes`] lends
//! those two lanes and the live plane as their presence, so a caller reaches
//! them through `LaneBinding::borrow_all` / `borrow_all_present` like any
//! other class's fields.
//!
//! The socket owns nothing. The lanes are the cube's own effective lanes
//! ([`Cube::lanes`]); the frames are the map table's, the store the cube's
//! frame was sealed from.

use lance_graph_contract::ontology::FieldRef;
use lance_graph_dt64::dr::{Frame, FrameStore, MapTable, Scope};
use lance_graph_dt64::{CoordSpace, LaneAddr, LaneSource, PresenceSource, SchemaRev};
use lance_graph_mask_risc::LaneRef;

use super::{Cube, RelId};

/// One relation of a cube as a lane source: field `x` is column 0, field `y`
/// is column 1, named by the predicate the class view resolved.
pub struct CubeLanes<'a> {
    cube: &'a Cube,
    table: &'a MapTable,
    rel: RelId,
    class: u32,
    /// The predicate IRIs of column 0 and column 1.
    predicates: [&'a str; 2],
    scope: Scope,
}

impl<'a> CubeLanes<'a> {
    /// Lend relation `rel` of `cube` as the two fields of `class` whose
    /// predicates are `predicates`, to readers of `scope`. `table` is the map
    /// table the cube was built from.
    pub fn new(
        cube: &'a Cube,
        table: &'a MapTable,
        rel: RelId,
        class: u32,
        predicates: [&'a str; 2],
        scope: Scope,
    ) -> Self {
        Self {
            cube,
            table,
            rel,
            class,
            predicates,
            scope,
        }
    }

    /// The row order of the relation as lent now: which cube, which
    /// relation, and that relation's state. It moves with every logged edge
    /// and every refold, so a binding made before one is refused after it
    /// even though the cube's frame did not advance. Two cubes over one map
    /// table never share a space, so a binding made over one is refused over
    /// the other.
    pub fn space(&self) -> CoordSpace {
        let generation = self.cube.base_generation(self.rel).unwrap_or(u64::MAX);
        let logged = self.cube.delta_len(self.rel) as u64;
        CoordSpace {
            population: self
                .cube
                .id
                .wrapping_mul(1 << 32)
                .wrapping_add(u64::from(self.rel) + 1),
            coordinates: generation.wrapping_mul(1 << 32).wrapping_add(logged),
        }
    }

    /// Whether `frame` is the frame the cube reads under, still current.
    fn current(&self, frame: &Frame) -> bool {
        let frames = self.table.frames();
        frames.check(frame).is_ok()
            && frames.check(self.cube.frame()).is_ok()
            && self.cube.check_frame(self.table).is_ok()
    }
}

impl LaneSource for CubeLanes<'_> {
    type Lane<'s>
        = LaneRef<'s>
    where
        Self: 's;

    fn lane_of(
        &self,
        class: u32,
        _rev: SchemaRev,
        _field: u8,
        basis: &FieldRef,
        frame: &Frame,
    ) -> Option<LaneAddr> {
        if class != self.class || !self.current(frame) {
            return None;
        }
        let lane = self
            .predicates
            .iter()
            .position(|p| *p == basis.predicate_iri)?;
        Some(LaneAddr {
            space: self.space(),
            lane: lane as u16,
            scope: self.scope,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<LaneRef<'_>> {
        if addr.space != self.space() || addr.scope != self.scope {
            return None;
        }
        let rel = self.cube.lanes(self.rel).ok()?;
        rel.cols
            .get(usize::from(addr.lane))
            .map(|lane| LaneRef::U32(lane))
    }

    fn frames(&self) -> &FrameStore {
        self.table.frames()
    }
}

impl PresenceSource for CubeLanes<'_> {
    fn borrow_presence(&self, addr: &LaneAddr) -> Option<&[u64]> {
        if addr.space != self.space() || addr.scope != self.scope || addr.lane > 1 {
            return None;
        }
        Some(&self.cube.lanes(self.rel).ok()?.live)
    }
}
