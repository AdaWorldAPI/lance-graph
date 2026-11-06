//! The property-to-lane socket and its joint version witness (D-LANESOCK-1,
//! `.claude/plans/d-lanesock-dr-dt64-lane-socket-v1.md`).
//!
//! A [`ClassView`] gives a class its field basis and a [`WideFieldMask`]
//! names the requested fields; neither is a physical address. The source
//! owner says, through [`LaneSource`], which of its own lanes holds a field
//! and in whose row order. [`BindStore::bind_lanes`] joins that answer with
//! the class route and the source frame into a [`LaneBinding`], which holds
//! addresses only, never a value.
//!
//! The class route and the frame are checked as a pair: the route against
//! this store's own route log, the frame against the [`FrameStore`] that
//! issued it. Neither clock stands in for the other.
//!
//! A lane whose row order differs from the selection's row order is refused
//! ([`LaneRefusal::CoordinateSpace`]): no mapping between coordinate spaces
//! is admitted here, so reading one through the other would be a gather.

use lance_graph_contract::class_view::{ClassView, WideFieldMask};

use crate::dr::{Frame, FrameRefusal, FrameStore, RouteVersion, Scope};
use crate::{BindStore, Refusal, SchemaRev};

/// Whose row order a lane is in: a source population and the revision of
/// that population's row order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoordSpace {
    pub population: u64,
    pub coordinates: u64,
}

/// A source-owned lane: which lane, in whose row order, for which reader
/// scope. An address; it carries no value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneAddr {
    pub space: CoordSpace,
    pub lane: u16,
    pub scope: Scope,
}

/// The socket a source owner implements. `lane_of` returning `None` means
/// the field is unbound, which is refused, never read as empty. `borrow`
/// hands out the source's own lane; it must not copy, reorder or allocate a
/// lane.
pub trait LaneSource {
    /// The borrowed lane (for example a mask-RISC `LaneRef`).
    type Lane<'a>
    where
        Self: 'a;

    fn lane_of(&self, class: u32, rev: SchemaRev, field: u8, frame: &Frame) -> Option<LaneAddr>;

    fn borrow(&self, addr: &LaneAddr) -> Option<Self::Lane<'_>>;
}

/// A DR position and the route version it was resolved at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteAt {
    pub position: u16,
    pub version: RouteVersion,
}

/// What [`BindStore::bind_lanes`] is asked to bind.
#[derive(Debug, Clone, Copy)]
pub struct LaneRequest<'a> {
    pub route: RouteAt,
    pub rev: SchemaRev,
    pub fields: &'a WideFieldMask,
    pub frame: &'a Frame,
    /// The reader's scope; every lane must serve exactly this scope.
    pub scope: Scope,
    /// The row order of the selection the lanes will be read under.
    pub selection: CoordSpace,
}

/// Why a lane binding was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaneRefusal {
    /// The class, route, schema revision or field basis (see [`Refusal`]).
    Bind(Refusal),
    /// The position routes to another class (or none) than it did at bind.
    RouteChanged {
        position: u16,
        bound: u32,
        now: Option<u32>,
    },
    /// The frame is foreign, stale or re-sealed.
    Frame(FrameRefusal),
    /// A requested field lies outside the class basis.
    FieldNotInBasis { class: u32, field: u8 },
    /// The source declares no lane for the field.
    Unbound { class: u32, field: u8 },
    /// The lane's row order is not the selection's.
    CoordinateSpace {
        field: u8,
        selection: CoordSpace,
        lane: CoordSpace,
    },
    /// The lane serves another reader scope.
    Scope {
        field: u8,
        reader: Scope,
        lane: Scope,
    },
    /// The source maps the field to another lane than it did at bind.
    MappingChanged { field: u8 },
    /// The source no longer holds the lane.
    Vanished { field: u8 },
    /// The field was not part of the binding.
    NotBound { field: u8 },
}

impl From<Refusal> for LaneRefusal {
    fn from(r: Refusal) -> LaneRefusal {
        LaneRefusal::Bind(r)
    }
}

/// A validated set of lane addresses for one class under one route and one
/// frame. Holds addresses only.
#[derive(Debug, Clone)]
pub struct LaneBinding {
    class: u32,
    rev: SchemaRev,
    route: RouteAt,
    frame: Frame,
    scope: Scope,
    selection: CoordSpace,
    fields: WideFieldMask,
    lanes: Vec<(u8, LaneAddr)>,
}

impl BindStore {
    /// Cold: resolve the class at `req.route`, check the basis at `req.rev`,
    /// check the frame, and take one lane per requested field from `source`.
    /// Every condition in [`LaneRefusal`] refuses.
    pub fn bind_lanes<V: ClassView + ?Sized, S: LaneSource>(
        &self,
        view: &V,
        frames: &FrameStore,
        source: &S,
        req: LaneRequest<'_>,
    ) -> Result<LaneBinding, LaneRefusal> {
        frames.check(req.frame).map_err(LaneRefusal::Frame)?;
        let class = self
            .routed_at(req.route.version, req.route.position)?
            .ok_or(Refusal::Unrouted {
                position: req.route.position,
                version: req.route.version,
            })?;
        current_route(self, req.route.position, class)?;
        let resolved = self.resolve(view, class, req.rev)?;
        let basis = resolved.fields.len();
        let mut lanes = Vec::with_capacity(req.fields.count() as usize);
        for field in 0..=u8::MAX {
            if !req.fields.has(field) {
                continue;
            }
            if usize::from(field) >= basis {
                return Err(LaneRefusal::FieldNotInBasis { class, field });
            }
            let addr = source
                .lane_of(class, req.rev, field, req.frame)
                .ok_or(LaneRefusal::Unbound { class, field })?;
            admit_lane(field, &addr, req.scope, req.selection)?;
            lanes.push((field, addr));
        }
        Ok(LaneBinding {
            class,
            rev: req.rev,
            route: req.route,
            frame: req.frame.clone(),
            scope: req.scope,
            selection: req.selection,
            fields: req.fields.clone(),
            lanes,
        })
    }
}

/// The position must still route to `class` at the store's current version.
fn current_route(store: &BindStore, position: u16, class: u32) -> Result<(), LaneRefusal> {
    let now = store.routed_at(store.current_version(), position)?;
    if now == Some(class) {
        Ok(())
    } else {
        Err(LaneRefusal::RouteChanged {
            position,
            bound: class,
            now,
        })
    }
}

fn admit_lane(
    field: u8,
    addr: &LaneAddr,
    scope: Scope,
    selection: CoordSpace,
) -> Result<(), LaneRefusal> {
    if addr.space != selection {
        return Err(LaneRefusal::CoordinateSpace {
            field,
            selection,
            lane: addr.space,
        });
    }
    if addr.scope != scope {
        return Err(LaneRefusal::Scope {
            field,
            reader: scope,
            lane: addr.scope,
        });
    }
    Ok(())
}

impl LaneBinding {
    pub fn class(&self) -> u32 {
        self.class
    }

    pub fn rev(&self) -> SchemaRev {
        self.rev
    }

    pub fn route(&self) -> RouteAt {
        self.route
    }

    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    pub fn selection(&self) -> CoordSpace {
        self.selection
    }

    /// The bound fields.
    pub fn fields(&self) -> &WideFieldMask {
        &self.fields
    }

    /// The bound lane of `field`.
    pub fn addr(&self, field: u8) -> Option<&LaneAddr> {
        self.lanes.iter().find(|(f, _)| *f == field).map(|(_, a)| a)
    }

    /// Hot: everything the bind checked, again, without allocating. The frame
    /// against its store; the route against the store's CURRENT version (a
    /// re-route since the bind is refused); the basis at the bound revision;
    /// and every lane against what the source maps now.
    pub fn check<V: ClassView + ?Sized, S: LaneSource>(
        &self,
        store: &BindStore,
        view: &V,
        frames: &FrameStore,
        source: &S,
    ) -> Result<(), LaneRefusal> {
        frames.check(&self.frame).map_err(LaneRefusal::Frame)?;
        current_route(store, self.route.position, self.class)?;
        store.resolve(view, self.class, self.rev)?;
        for (field, addr) in &self.lanes {
            let field = *field;
            match source.lane_of(self.class, self.rev, field, &self.frame) {
                Some(current) if current == *addr => {}
                Some(_) => return Err(LaneRefusal::MappingChanged { field }),
                None => {
                    return Err(LaneRefusal::Unbound {
                        class: self.class,
                        field,
                    })
                }
            }
            admit_lane(field, addr, self.scope, self.selection)?;
        }
        Ok(())
    }

    /// [`LaneBinding::check`], then borrow `field`'s lane from the source.
    pub fn lane<'s, V: ClassView + ?Sized, S: LaneSource>(
        &self,
        store: &BindStore,
        view: &V,
        frames: &FrameStore,
        source: &'s S,
        field: u8,
    ) -> Result<S::Lane<'s>, LaneRefusal> {
        self.check(store, view, frames, source)?;
        let addr = self.addr(field).ok_or(LaneRefusal::NotBound { field })?;
        source.borrow(addr).ok_or(LaneRefusal::Vanished { field })
    }
}
