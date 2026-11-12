//! The property-to-lane socket and its joint version witness (D-LANESOCK-1,
//! `.claude/plans/d-lanesock-dr-dt64-lane-socket-v1.md`).
//!
//! A [`ClassView`] gives a class its field basis and a [`WideFieldMask`]
//! names the requested fields; neither is a physical address. The source
//! owner says, through [`LaneSource`], which of its own lanes holds a field
//! and in whose row order. [`BindStore::bind_lanes`] joins that answer with
//! the class route and the source frame into a [`LaneBinding`], which holds
//! addresses and the bound predicate identities, never a value.
//!
//! The class route and the frame are checked as a pair: the route against
//! this store's own route log, the frame against the [`FrameStore`] that
//! issued it. Neither clock stands in for the other.
//!
//! A lane whose row order differs from the selection's row order is refused
//! ([`LaneRefusal::CoordinateSpace`]): no mapping between coordinate spaces
//! is admitted here, so reading one through the other would be a gather.

use lance_graph_contract::class_view::{ClassView, WideFieldMask};
use lance_graph_contract::ontology::FieldRef;

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
///
/// `basis` is the field DT64 resolved at `field` through the [`ClassView`] it
/// validated. A source maps that field, never a position it read from a
/// ClassView of its own: two views can agree on class, revision and field
/// count and still order or name their fields differently.
pub trait LaneSource {
    /// The borrowed lane (for example a mask-RISC `LaneRef`).
    type Lane<'a>
    where
        Self: 'a;

    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        basis: &FieldRef,
        frame: &Frame,
    ) -> Option<LaneAddr>;

    fn borrow(&self, addr: &LaneAddr) -> Option<Self::Lane<'_>>;

    /// The [`FrameStore`] that lives with the data this source borrows from.
    ///
    /// The source owns this pairing, not the caller: an implementation derives
    /// the store from the same owner its lanes come from and never takes one
    /// from outside. [`BindStore::bind_lanes`] and [`LaneBinding::check`]
    /// refuse `Frame(ForeignFrame)` when the caller's `frames` is any other
    /// store, so a binding made over one source cannot be checked or read
    /// against another source whose lane numbers happen to match.
    ///
    /// The trait cannot prove that the returned store issued this source's
    /// bytes; that remains the implementer's obligation.
    fn frames(&self) -> &FrameStore;
}

/// A source-owned selection: which of the source's position lists, in whose
/// row order, for which reader scope. An address; it carries no position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelAddr {
    pub space: CoordSpace,
    pub sel: u16,
    pub scope: Scope,
}

/// The second socket of a source owner: selections in position form
/// (D-DR-SELECT-K-SV-2).
///
/// A selection is a list of row positions of the SAME source, in the row
/// order its [`SelAddr::space`] names. The source is its only issuer: a
/// caller cannot pair positions of its own with a binding, because
/// [`LaneBinding::selected`] takes them from here and nowhere else. The list
/// is a bag in list order; the socket never sorts or deduplicates it.
///
/// `selection_of` returning `None` means the source holds no such selection
/// under `frame`, which is refused, never read as empty.
/// `borrow_positions` hands out the source's own list; it must not copy,
/// reorder, deduplicate or allocate.
///
/// As for [`LaneSource::frames`], the trait cannot prove that the positions
/// were produced from the bytes the frame seals; that is the implementer's
/// obligation. What the socket does establish is that lanes and positions
/// come from one owner, under one frame of that owner's store, in one row
/// order and for one reader scope.
pub trait SelectionSource: LaneSource {
    fn selection_of(&self, sel: u16, frame: &Frame) -> Option<SelAddr>;

    fn borrow_positions(&self, addr: &SelAddr) -> Option<&[u32]>;
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
    /// The store binds the class over another basis than the one this binding
    /// was made against. A store never changes a bound class (D5), so this
    /// fires only when the binding is checked against ANOTHER store; it says
    /// nothing about the view; a view that changed at a bound position is
    /// `PredicateChanged`.
    StoreBasisMismatch { class: u32 },
    /// The view names another predicate at a bound position than the one the
    /// binding was made over. Compared per bound field by `predicate_iri`,
    /// without hashing; a label-only rename does not fire it, and a change
    /// at a position the binding did not bind does not either.
    PredicateChanged { class: u32, field: u8 },
    /// The source maps the field to another lane than it did at bind.
    MappingChanged { field: u8 },
    /// The source no longer holds the lane.
    Vanished { field: u8 },
    /// The field was not part of the binding.
    NotBound { field: u8 },
    /// The source holds no selection `sel` under the binding's frame.
    SelectionUnbound { sel: u16 },
    /// The selection's positions are in another row order than the one the
    /// lanes were bound under. Reading the lanes through them would be a
    /// read through another coordinate space, so it is refused.
    SelectionSpace {
        sel: u16,
        bound: CoordSpace,
        selection: CoordSpace,
    },
    /// The selection serves another reader scope than the binding's.
    SelectionScope {
        sel: u16,
        reader: Scope,
        selection: Scope,
    },
    /// The source no longer holds the selection's positions.
    SelectionVanished { sel: u16 },
}

impl From<Refusal> for LaneRefusal {
    fn from(r: Refusal) -> LaneRefusal {
        LaneRefusal::Bind(r)
    }
}

/// A validated set of lane addresses for one class under one route and one
/// frame. Holds addresses and, per bound field, the predicate identity it was
/// bound over. Never a value.
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
    /// Parallel to `lanes`: the `predicate_iri` each bound field had in the
    /// witnessed basis at bind. An identity, never a value. The hot check
    /// compares the view's field at each bound position against it.
    predicates: Vec<Box<str>>,
    /// The witness of the basis the class was bound with when this binding
    /// was made: one `u128`, compared on the hot path without hashing.
    basis: crate::BasisDigest,
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
        source_owns(frames, source)?;
        frames.check(req.frame).map_err(LaneRefusal::Frame)?;
        let class = self
            .routed_at(req.route.version, req.route.position)?
            .ok_or(Refusal::Unrouted {
                position: req.route.position,
                version: req.route.version,
            })?;
        current_route(self, req.route.position, class)?;
        let resolved = self.resolve(view, class, req.rev)?;
        let basis = self
            .bound_basis(class)
            .ok_or(Refusal::UnknownClass(class))?;
        let mut lanes = Vec::with_capacity(req.fields.count() as usize);
        let mut predicates = Vec::with_capacity(req.fields.count() as usize);
        for field in 0..=u8::MAX {
            if !req.fields.has(field) {
                continue;
            }
            let Some(field_ref) = resolved.fields.get(usize::from(field)) else {
                return Err(LaneRefusal::FieldNotInBasis { class, field });
            };
            let addr = source
                .lane_of(class, req.rev, field, field_ref, req.frame)
                .ok_or(LaneRefusal::Unbound { class, field })?;
            admit_lane(field, &addr, req.scope, req.selection)?;
            lanes.push((field, addr));
            predicates.push(Box::<str>::from(field_ref.predicate_iri.as_str()));
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
            predicates,
            basis,
        })
    }
}

/// The caller's `frames` must be the store the source's data lives with.
fn source_owns<S: LaneSource + ?Sized>(frames: &FrameStore, source: &S) -> Result<(), LaneRefusal> {
    if frames.same_store(source.frames()) {
        Ok(())
    } else {
        Err(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
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

    /// Hot: the bind's checks again, without allocating and without hashing
    /// the basis. The frame against its store; the route against the store's
    /// CURRENT version (a re-route since the bind is refused); the store's
    /// basis witness against the one recorded at bind (one `u128`); the
    /// view's basis at the bound revision by revision and field count; each
    /// bound position's `predicate_iri` against the one recorded at bind (a
    /// string compare, no hashing); and every lane against what the source
    /// maps now.
    ///
    /// NOT checked here: positions the binding did not bind. A same-count
    /// change confined to unbound positions passes this check and is refused
    /// by `BindStore::resolve`.
    pub fn check<V: ClassView + ?Sized, S: LaneSource>(
        &self,
        store: &BindStore,
        view: &V,
        frames: &FrameStore,
        source: &S,
    ) -> Result<(), LaneRefusal> {
        source_owns(frames, source)?;
        frames.check(&self.frame).map_err(LaneRefusal::Frame)?;
        current_route(store, self.route.position, self.class)?;
        // Revision and count here: the whole-basis witness was checked when
        // the binding was made (bind_lanes resolves through the witnessed
        // `resolve`), and re-hashing the basis would put schema hashing on
        // the hot path. The bound positions are compared below, each against
        // the predicate recorded at bind, so a reorder that moves a bound
        // field is refused even when the source maps lanes by position.
        let (fields, bound) = store.resolve_counted(view, self.class, self.rev)?;
        if bound != self.basis {
            return Err(LaneRefusal::StoreBasisMismatch { class: self.class });
        }
        for ((field, addr), bound_iri) in self.lanes.iter().zip(&self.predicates) {
            let field = *field;
            let field_ref = fields
                .get(usize::from(field))
                .ok_or(LaneRefusal::FieldNotInBasis {
                    class: self.class,
                    field,
                })?;
            match source.lane_of(self.class, self.rev, field, field_ref, &self.frame) {
                Some(current) if current == *addr => {}
                Some(_) => return Err(LaneRefusal::MappingChanged { field }),
                None => {
                    return Err(LaneRefusal::Unbound {
                        class: self.class,
                        field,
                    })
                }
            }
            if field_ref.predicate_iri.as_str() != &**bound_iri {
                return Err(LaneRefusal::PredicateChanged {
                    class: self.class,
                    field,
                });
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

    /// [`LaneBinding::check`] once, then `field`'s lane AND selection `sel`'s
    /// positions, both borrowed from the one `source`.
    ///
    /// This is the joint witness of D-DR-SELECT-K-SV-2: the positions may be
    /// read against the lane because one check covers both. The frame is this
    /// source's store's current seal (a foreign, stale or re-sealed frame is
    /// refused, as is a caller-chosen store); the route, the basis and every
    /// bound lane are as they were at the bind; and the selection is one the
    /// source itself holds under THIS binding's frame, in THIS binding's row
    /// order ([`LaneRefusal::SelectionSpace`]) and for THIS binding's reader
    /// scope ([`LaneRefusal::SelectionScope`]). Every refusal is returned
    /// before a lane or a position is borrowed. Nothing is allocated, copied
    /// or reordered.
    ///
    /// NOT established here, by design:
    ///
    /// - That each position is a row of the lane. This crate does not know a
    ///   lane's length; the consumer that folds (`execute_sel`) refuses a
    ///   position outside the frame.
    /// - Authorization. [`Scope`] equality is a consistency check between
    ///   the reader, the lanes and the selection. It is not a grant: no
    ///   authority in this crate issues a `Scope`, so a revoked permission is
    ///   not observed here.
    /// - History. Like [`LaneBinding::check`], this admits the current frame
    ///   only; a selection of an earlier frame is refused, never read
    ///   against today's lanes.
    pub fn selected<'s, V: ClassView + ?Sized, S: SelectionSource>(
        &self,
        store: &BindStore,
        view: &V,
        frames: &FrameStore,
        source: &'s S,
        field: u8,
        sel: u16,
    ) -> Result<(S::Lane<'s>, &'s [u32]), LaneRefusal> {
        self.check(store, view, frames, source)?;
        let lane = self.addr(field).ok_or(LaneRefusal::NotBound { field })?;
        let selection = source
            .selection_of(sel, &self.frame)
            .ok_or(LaneRefusal::SelectionUnbound { sel })?;
        if selection.space != self.selection {
            return Err(LaneRefusal::SelectionSpace {
                sel,
                bound: self.selection,
                selection: selection.space,
            });
        }
        if selection.scope != self.scope {
            return Err(LaneRefusal::SelectionScope {
                sel,
                reader: self.scope,
                selection: selection.scope,
            });
        }
        let lane = source.borrow(lane).ok_or(LaneRefusal::Vanished { field })?;
        let positions = source
            .borrow_positions(&selection)
            .ok_or(LaneRefusal::SelectionVanished { sel })?;
        Ok((lane, positions))
    }

    /// [`LaneBinding::check`] once, then every bound lane, in ascending field
    /// order.
    ///
    /// Calling [`LaneBinding::lane`] per field re-runs the whole check on
    /// each call, and the check walks every bound lane, so reading all N
    /// lanes that way does N² source lookups. This does N. Because the one
    /// check covers every bound lane, any refusal (frame, route, basis,
    /// mapping, scope, row order) is returned before a single lane is
    /// borrowed. A lane the source no longer holds yields
    /// `Err(Vanished { field })` at its position; the iterator does not stop
    /// there. Nothing is allocated.
    ///
    /// The returned lanes are as current as the check: like `lane`, this
    /// holds no lock on the source.
    pub fn borrow_all<'b, 's, V: ClassView + ?Sized, S: LaneSource>(
        &'b self,
        store: &BindStore,
        view: &V,
        frames: &FrameStore,
        source: &'s S,
    ) -> Result<BoundLanes<'b, 's, S>, LaneRefusal> {
        self.check(store, view, frames, source)?;
        Ok(BoundLanes {
            lanes: self.lanes.iter(),
            source,
        })
    }
}

/// Every lane of a checked [`LaneBinding`], borrowed one at a time from the
/// source, in ascending field order. Made by [`LaneBinding::borrow_all`].
pub struct BoundLanes<'b, 's, S: LaneSource> {
    lanes: core::slice::Iter<'b, (u8, LaneAddr)>,
    source: &'s S,
}

impl<'s, S: LaneSource> Iterator for BoundLanes<'_, 's, S> {
    type Item = Result<(u8, S::Lane<'s>), LaneRefusal>;

    fn next(&mut self) -> Option<Self::Item> {
        let (field, addr) = self.lanes.next()?;
        let field = *field;
        Some(
            self.source
                .borrow(addr)
                .map(|lane| (field, lane))
                .ok_or(LaneRefusal::Vanished { field }),
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.lanes.size_hint()
    }
}

impl<S: LaneSource> ExactSizeIterator for BoundLanes<'_, '_, S> {}
