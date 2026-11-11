//! S4 / EXO-243: applying a scoped authorization decision to a DT64 lane
//! binding. Test-only: no production code changes.
//!
//! Field half: `requested ∩ ScopedDecision::visible_fields()` is the mask that
//! goes to `bind_lanes`. Row half: a SKETCH (`mod row_plane`) of a per-row
//! plane prepared from a `ScopeSpec`; no production type does this yet.
//! Tests named `finding_*` pin today's behaviour, gaps included.

use std::cell::Cell;

use lance_graph_contract::class_view::{ClassId, ClassView, FieldMask, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::property::PrefetchDepth;
use lance_graph_contract::rbac::{ActorId, ClassRbac, Operation, RoleId, ScopePath, ScopeSpec};
use lance_graph_dt64::dr::{Focus, Frame, FrameRevisions, FrameStore, GuardRefusal, Guards, Scope};
use lance_graph_dt64::{
    view_class, BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRefusal,
    LaneRequest, LaneSource, RouteAt, SchemaRev,
};
use lance_graph_rbac::access::AccessDecision;
use lance_graph_rbac::authorize::{authorize_scoped, ScopedDecision};

const CLASS: u32 = 0x0C1A_0001;
const POSITION: u16 = 5;
const ROWS: usize = 1000;
const FIELDS: usize = 12;
const WORDS: usize = 16;
const POP: CoordSpace = CoordSpace {
    population: 1,
    coordinates: 1,
};
// Hand-assigned consistency witnesses. They carry no authority.
const SCOPE_A: Scope = Scope { id: 0xA };
const SCOPE_B: Scope = Scope { id: 0xB };

struct View(Vec<FieldRef>);

impl View {
    fn twelve() -> View {
        View(
            (0..FIELDS)
                .map(|i| FieldRef::new(format!("med:f{i:02}"), format!("F{i:02}")))
                .collect(),
        )
    }
}

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

/// The source owner's data: 12 lanes of 1000 rows, per-row tenant and path.
struct Data {
    lanes: Vec<Vec<i32>>,
    row_tenant: Vec<u64>,
    row_path: Vec<ScopePath>,
    frames: FrameStore,
    lookups: Cell<usize>,
    borrows: Cell<usize>,
}

impl Data {
    fn new() -> Data {
        Data {
            lanes: (0..FIELDS)
                .map(|f| (0..ROWS).map(|r| (f * 10_000 + r) as i32).collect())
                .collect(),
            row_tenant: (0..ROWS).map(|r| if r < 800 { 9 } else { 10 }).collect(),
            row_path: (0..ROWS)
                .map(|r| {
                    ScopePath::new(&[1 + (r % 4) as u64, ((r / 4) % 5) as u64]).expect("depth")
                })
                .collect(),
            frames: FrameStore::new(FrameRevisions {
                source: 1,
                relation_map: 1,
                coordinates: 1,
            }),
            lookups: Cell::new(0),
            borrows: Cell::new(0),
        }
    }
}

/// What the data owner hands one reader: a socket serving one scope over the
/// same data and the same `FrameStore`.
struct Socket<'d> {
    data: &'d Data,
    serves: Scope,
}

impl LaneSource for Socket<'_> {
    type Lane<'a>
        = &'a [i32]
    where
        Self: 'a;
    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        self.data.lookups.set(self.data.lookups.get() + 1);
        if class != CLASS || rev != SchemaRev(1) || usize::from(field) >= FIELDS {
            return None;
        }
        Some(LaneAddr {
            space: POP,
            lane: u16::from(field),
            scope: self.serves,
        })
    }
    fn borrow(&self, addr: &LaneAddr) -> Option<&[i32]> {
        self.data.borrows.set(self.data.borrows.get() + 1);
        self.data
            .lanes
            .get(usize::from(addr.lane))
            .map(Vec::as_slice)
    }
    fn frames(&self) -> &FrameStore {
        &self.data.frames
    }
}

/// alice: fields 0..=5, rows tenant 9 under [1]. bob: fields 4..=9, rows
/// tenant 9 under [2, 3]. carol: no Read grant. ghost: no roles.
struct Policy;

impl ClassRbac for Policy {
    fn actor_roles(&self, actor: ActorId<'_>) -> &[RoleId] {
        match actor {
            "alice" => &["role_a"],
            "bob" => &["role_b"],
            "carol" => &["role_c"],
            _ => &[],
        }
    }
    fn grant_permits(&self, role: RoleId, class: u32, op: &Operation<'_>) -> bool {
        class == CLASS
            && match role {
                "role_a" | "role_b" => matches!(op, Operation::Read { .. }),
                "role_c" => matches!(op, Operation::Act { .. }),
                _ => false,
            }
    }
    fn row_scope(&self, role: RoleId, _class: u32) -> Option<ScopeSpec> {
        let at = |seg: &[u64]| ScopeSpec {
            tenant: Some(9),
            path: ScopePath::new(seg).expect("depth"),
            ..ScopeSpec::default()
        };
        match role {
            "role_a" => Some(at(&[1])),
            "role_b" => Some(at(&[2, 3])),
            _ => None,
        }
    }
    fn field_mask(&self, role: RoleId, _class: u32) -> WideFieldMask {
        match role {
            "role_a" => WideFieldMask::from_positions(&[0, 1, 2, 3, 4, 5]),
            "role_b" => WideFieldMask::from_positions(&[4, 5, 6, 7, 8, 9]),
            "role_c" => WideFieldMask::from_positions(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            _ => WideFieldMask::EMPTY,
        }
    }
}

struct World {
    store: BindStore,
    view: View,
    data: Data,
    route: RouteAt,
    frame: Frame,
}

fn world() -> World {
    let mut store = BindStore::new();
    store
        .register(
            &View::twelve(),
            CLASS,
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: FIELDS,
                extent: Extent::DR64_CUBE,
            },
            Vec::new(),
        )
        .expect("register");
    let version = store.route(POSITION, CLASS).expect("route");
    let data = Data::new();
    let frame = data.frames.seal();
    World {
        store,
        view: View::twelve(),
        data,
        route: RouteAt {
            position: POSITION,
            version,
        },
        frame,
    }
}

fn read() -> Operation<'static> {
    Operation::Read {
        depth: PrefetchDepth::Detail,
    }
}

fn decide(actor: &str) -> ScopedDecision {
    authorize_scoped(&Policy, actor, CLASS, read())
}

fn requested() -> WideFieldMask {
    WideFieldMask::from_positions(&[0, 2, 4, 6, 8, 10, 11])
}

/// The answer to EXO-243 ask 1.
fn fields_for(d: &ScopedDecision) -> WideFieldMask {
    requested().intersect(&d.visible_fields())
}

fn bind_as(
    w: &World,
    serves: Scope,
    reader: Scope,
    fields: &WideFieldMask,
) -> Result<LaneBinding, LaneRefusal> {
    let sock = Socket {
        data: &w.data,
        serves,
    };
    w.store.bind_lanes(
        &w.view,
        &w.data.frames,
        &sock,
        LaneRequest {
            route: w.route,
            rev: SchemaRev(1),
            fields,
            frame: &w.frame,
            scope: reader,
            selection: POP,
        },
    )
}

#[derive(Debug, PartialEq)]
enum Blocked {
    Denied(AccessDecision),
    Lane(LaneRefusal),
}

/// The consumer recipe: decide first, touch the socket only on Allow.
fn bind_decided(
    w: &World,
    d: &ScopedDecision,
    serves: Scope,
    reader: Scope,
) -> Result<LaneBinding, Blocked> {
    if d.decision != AccessDecision::Allow {
        return Err(Blocked::Denied(d.decision.clone()));
    }
    bind_as(w, serves, reader, &fields_for(d)).map_err(Blocked::Lane)
}

fn sock(w: &World, serves: Scope) -> Socket<'_> {
    Socket {
        data: &w.data,
        serves,
    }
}

fn check(w: &World, b: &LaneBinding, serves: Scope) -> Result<(), LaneRefusal> {
    b.check(&w.store, &w.view, &w.data.frames, &sock(w, serves))
}

// ---------------------------------------------------------------- A. fields

#[test]
fn role_a_binds_exactly_the_requested_and_visible_fields() {
    let w = world();
    let d = decide("alice");
    assert_eq!(d.decision, AccessDecision::Allow);
    let fields = fields_for(&d);
    assert_eq!(fields, WideFieldMask::from_positions(&[0, 2, 4]));
    assert_eq!(fields.count(), 3);

    let b = bind_as(&w, SCOPE_A, SCOPE_A, &fields).expect("alice binds");
    assert_eq!(*b.fields(), fields);
    for f in 0..FIELDS as u8 {
        assert_eq!(b.addr(f).is_some(), matches!(f, 0 | 2 | 4), "field {f}");
    }
    assert_eq!(w.data.lookups.get(), 3, "no lookup for an excluded field");

    let s = sock(&w, SCOPE_A);
    let lane = b
        .lane(&w.store, &w.view, &w.data.frames, &s, 0)
        .expect("field 0");
    assert!(std::ptr::eq(lane, w.data.lanes[0].as_slice()));
    assert_eq!(lane.len(), ROWS);
    assert_eq!(
        b.lane(&w.store, &w.view, &w.data.frames, &s, 6).err(),
        Some(LaneRefusal::NotBound { field: 6 })
    );
    let order: Vec<u8> = b
        .borrow_all(&w.store, &w.view, &w.data.frames, &s)
        .expect("check")
        .map(|r| r.expect("lane").0)
        .collect();
    assert_eq!(order, vec![0, 2, 4]);
}

#[test]
fn role_b_gets_its_own_fields_and_the_overlap_is_one_field() {
    let w = world();
    let fa = fields_for(&decide("alice"));
    let fb = fields_for(&decide("bob"));
    assert_eq!(fb, WideFieldMask::from_positions(&[4, 6, 8]));
    assert_ne!(fa, fb);
    assert_ne!(fa, requested());
    assert_ne!(fb, requested());
    assert!(!fa.is_empty() && !fb.is_empty());
    assert_eq!(fa.intersect(&fb), WideFieldMask::from_positions(&[4]));
    for f in [10u8, 11] {
        assert!(
            !fa.has(f) && !fb.has(f),
            "field {f} is visible to no reader"
        );
    }
    let b = bind_as(&w, SCOPE_B, SCOPE_B, &fb).expect("bob binds");
    assert_eq!(b.fields().count(), 3);
}

#[test]
fn a_non_allow_decision_yields_no_bindable_field() {
    let w = world();
    let carol = decide("carol");
    assert_eq!(
        carol.decision,
        AccessDecision::Deny {
            reason: "insufficient read depth"
        }
    );
    let ghost = decide("ghost");
    assert_eq!(
        ghost.decision,
        AccessDecision::Deny {
            reason: "unknown role"
        }
    );
    for d in [&carol, &ghost] {
        assert_eq!(d.scope, None);
        assert!(fields_for(d).is_empty());
        assert_eq!(
            bind_decided(&w, d, SCOPE_A, SCOPE_A).err(),
            Some(Blocked::Denied(d.decision.clone()))
        );
    }
    assert_eq!(w.data.lookups.get(), 0);
    assert_eq!(w.data.borrows.get(), 0);

    // Anti-vacuity: the same recipe binds for an allowed reader.
    let b = bind_decided(&w, &decide("alice"), SCOPE_A, SCOPE_A).expect("alice binds");
    assert_eq!(b.fields().count(), 3);
    assert_eq!(w.data.lookups.get(), 3);
}

#[test]
fn finding_the_raw_field_mask_of_a_refused_decision_binds_every_requested_field() {
    // Why ask 1 says `visible_fields()`, never `field_mask`: dt64 sees only a
    // mask and cannot protect here.
    let w = world();
    let d = decide("carol");
    assert_eq!(d.field_mask, WideFieldMask::from(FieldMask::FULL));
    let leaked = requested().intersect(&d.field_mask);
    assert_eq!(leaked, requested());
    assert_eq!(leaked.count(), 7);
    let b = bind_as(&w, SCOPE_A, SCOPE_A, &leaked).expect("binds");
    assert_eq!(
        (0..FIELDS as u8).filter(|f| b.addr(*f).is_some()).count(),
        7
    );
}

#[test]
fn finding_bind_lanes_admits_an_empty_request() {
    // "Binds nothing" holds as zero lanes, but dt64 cannot tell a refused
    // reader from an empty request: branch on the decision (A3), not on the
    // bind result. Re-pin if an empty request ever becomes a refusal.
    let w = world();
    let b = bind_as(&w, SCOPE_A, SCOPE_A, &WideFieldMask::EMPTY).expect("empty binds");
    assert_eq!(b.fields().count(), 0);
    assert!((0..FIELDS as u8).all(|f| b.addr(f).is_none()));
    let s = sock(&w, SCOPE_A);
    let n = b
        .borrow_all(&w.store, &w.view, &w.data.frames, &s)
        .expect("check")
        .count();
    assert_eq!(n, 0);
    assert_eq!(check(&w, &b, SCOPE_A), Ok(()));
    assert_eq!(w.data.lookups.get(), 0);
}

// ------------------------------------------------- B. refused for the other

#[test]
fn role_b_cannot_bind_through_role_a_socket() {
    let w = world();
    let fb = fields_for(&decide("bob"));
    assert_eq!(
        bind_as(&w, SCOPE_A, SCOPE_B, &fb).err(),
        Some(LaneRefusal::Scope {
            field: 4,
            reader: SCOPE_B,
            lane: SCOPE_A
        })
    );
    assert!(bind_as(&w, SCOPE_B, SCOPE_B, &fb).is_ok());
}

#[test]
fn role_a_guards_are_refused_under_role_b_scope() {
    let w = world();
    let g = Guards {
        frame: w.frame.clone(),
        focus: Focus { revision: 1 },
        scope: SCOPE_A,
    };
    assert_eq!(
        g.check(&w.data.frames, (Focus { revision: 1 }, SCOPE_B)),
        Err(GuardRefusal::ScopeChanged)
    );
    assert_eq!(
        g.check(&w.data.frames, (Focus { revision: 1 }, SCOPE_A)),
        Ok(())
    );
}

#[test]
fn finding_a_lane_binding_alone_does_not_know_its_reader() {
    // `check` takes no reader and the binding has no `scope()` accessor, so
    // whoever holds alice's binding and her socket reads field 0, which is
    // not in bob's fields. Step S4b-1 turns this red-to-green.
    let w = world();
    let b_a = bind_decided(&w, &decide("alice"), SCOPE_A, SCOPE_A).expect("alice");
    assert!(!fields_for(&decide("bob")).has(0));
    let s = sock(&w, SCOPE_A);
    let lane = b_a
        .lane(&w.store, &w.view, &w.data.frames, &s, 0)
        .expect("transferable binding reads");
    assert!(std::ptr::eq(lane, w.data.lanes[0].as_slice()));
}

#[test]
fn role_a_binding_checked_through_role_b_socket_is_refused() {
    let w = world();
    let b_a = bind_decided(&w, &decide("alice"), SCOPE_A, SCOPE_A).expect("alice");
    assert_eq!(
        check(&w, &b_a, SCOPE_B),
        Err(LaneRefusal::MappingChanged { field: 0 })
    );
    assert_eq!(check(&w, &b_a, SCOPE_A), Ok(()));
}

#[test]
fn finding_a_copied_scope_id_binds_another_readers_fields() {
    // Bob copies the public `u64` id of alice's scope. Equality of two ids is
    // not a grant; a sealed scope (S4b-2) closes this.
    let w = world();
    assert!(!fields_for(&decide("bob")).has(0));
    let alices = fields_for(&decide("alice"));
    let b = bind_as(&w, SCOPE_A, SCOPE_A, &alices).expect("copied id admitted");
    let s = sock(&w, SCOPE_A);
    let lane = b
        .lane(&w.store, &w.view, &w.data.frames, &s, 0)
        .expect("reads alice-only field");
    assert!(std::ptr::eq(lane, w.data.lanes[0].as_slice()));
}

/// SKETCH (test-local). Interns (actor, class, decision) by VALUE and hands
/// out a serial id. Not a hash: two different keys never share an id.
struct Issuer {
    seen: Vec<(String, u32, ScopedDecision)>,
}

impl Issuer {
    fn scope_for(&mut self, actor: &str, class: u32, d: &ScopedDecision) -> Option<Scope> {
        if d.decision != AccessDecision::Allow {
            return None;
        }
        let at = match self
            .seen
            .iter()
            .position(|(a, c, s)| a == actor && *c == class && s == d)
        {
            Some(i) => i,
            None => {
                self.seen.push((actor.to_owned(), class, d.clone()));
                self.seen.len() - 1
            }
        };
        Some(Scope { id: at as u64 + 1 })
    }
}

#[test]
fn scope_ids_issued_per_decision_separate_readers_and_follow_a_changed_decision() {
    let w = world();
    let mut issuer = Issuer { seen: Vec::new() };
    let da = decide("alice");
    let a1 = issuer.scope_for("alice", CLASS, &da).expect("allow");
    assert_eq!(issuer.scope_for("alice", CLASS, &da), Some(a1));
    let b1 = issuer
        .scope_for("bob", CLASS, &decide("bob"))
        .expect("allow");
    assert_ne!(a1, b1);

    let mut changed = da.clone();
    changed.field_mask = changed
        .field_mask
        .union(&WideFieldMask::from_positions(&[7]));
    let a2 = issuer.scope_for("alice", CLASS, &changed).expect("allow");
    assert_ne!(a1, a2);

    let guards = Guards {
        frame: w.frame.clone(),
        focus: Focus { revision: 1 },
        scope: a1,
    };
    let now = |s| (Focus { revision: 1 }, s);
    assert_eq!(
        guards.check(&w.data.frames, now(a2)),
        Err(GuardRefusal::ScopeChanged)
    );
    let unchanged = issuer.scope_for("alice", CLASS, &da).expect("allow");
    assert_eq!(guards.check(&w.data.frames, now(unchanged)), Ok(()));

    for actor in ["carol", "ghost"] {
        assert_eq!(issuer.scope_for(actor, CLASS, &decide(actor)), None);
    }
}

// -------------------------------------------------------------- C. row half

mod row_plane {
    //! SKETCH, test-local. No production type prepares or stores a row plane
    //! yet (plan S7). This module fixes what such a plane must be witnessed
    //! by: the source frame it was prepared under, the ScopeSpec it was
    //! prepared from (compared by value), and the row order. A row plane
    //! depends on row CONTENT, so by EXO-255 it is recomputed when the source
    //! moves; it is never reused because the coordinates are unchanged.

    use super::*;
    use lance_graph_dt64::dr::FrameRefusal;

    pub struct RowPlane {
        words: [u64; WORDS],
        frame: Frame,
        spec: ScopeSpec,
        space: CoordSpace,
    }

    #[derive(Debug, PartialEq, Eq)]
    pub enum PlaneRefusal {
        Unscoped,
        PredicateNotInterpreted,
        Frame(FrameRefusal),
        ScopeChanged,
        CoordinateSpace,
    }

    pub fn prepare(
        data: &Data,
        frame: &Frame,
        scope: Option<ScopeSpec>,
        space: CoordSpace,
    ) -> Result<RowPlane, PlaneRefusal> {
        data.frames.check(frame).map_err(PlaneRefusal::Frame)?;
        let spec = scope.ok_or(PlaneRefusal::Unscoped)?;
        if spec.predicate_key != 0 {
            return Err(PlaneRefusal::PredicateNotInterpreted);
        }
        let mut words = [0u64; WORDS];
        for (r, (tenant, path)) in data.row_tenant.iter().zip(&data.row_path).enumerate() {
            if spec.tenant.is_none_or(|t| t == *tenant) && spec.admits(path) {
                words[r / 64] |= 1u64 << (r % 64);
            }
        }
        Ok(RowPlane {
            words,
            frame: frame.clone(),
            spec,
            space,
        })
    }

    impl RowPlane {
        pub fn admit(
            &self,
            frames: &FrameStore,
            scope_now: Option<ScopeSpec>,
            space_now: CoordSpace,
        ) -> Result<&[u64; WORDS], PlaneRefusal> {
            frames.check(&self.frame).map_err(PlaneRefusal::Frame)?;
            if scope_now != Some(self.spec) {
                return Err(PlaneRefusal::ScopeChanged);
            }
            if space_now != self.space {
                return Err(PlaneRefusal::CoordinateSpace);
            }
            Ok(&self.words)
        }
    }

    pub fn popcount(words: &[u64; WORDS]) -> u32 {
        words.iter().map(|w| w.count_ones()).sum()
    }

    pub fn bit(words: &[u64; WORDS], r: usize) -> bool {
        (words[r / 64] >> (r % 64)) & 1 == 1
    }

    /// Sum of `lane` over the rows set in `words`, in place; no row list.
    pub fn fold_sum(words: &[u64; WORDS], lane: &[i32]) -> i64 {
        let mut sum = 0i64;
        for (wi, &word) in words.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let idx = wi * 64 + w.trailing_zeros() as usize;
                sum += i64::from(lane[idx]);
                w &= w - 1;
            }
        }
        sum
    }
}

use row_plane::{bit, fold_sum, popcount, prepare, PlaneRefusal};

fn plane_for(w: &World, actor: &str) -> row_plane::RowPlane {
    prepare(&w.data, &w.frame, decide(actor).scope, POP).expect("plane")
}

#[test]
fn row_planes_admit_exactly_each_roles_rows() {
    let w = world();
    let pa = plane_for(&w, "alice");
    let pb = plane_for(&w, "bob");
    let a = pa
        .admit(&w.data.frames, decide("alice").scope, POP)
        .expect("a");
    let b = pb
        .admit(&w.data.frames, decide("bob").scope, POP)
        .expect("b");
    assert_eq!(popcount(a), 200);
    assert!(bit(a, 0) && bit(a, 796) && !bit(a, 1) && !bit(a, 800));
    assert_eq!(popcount(b), 40);
    assert!(bit(b, 13) && bit(b, 33) && bit(b, 53));
    let both: u32 = a.iter().zip(b).map(|(x, y)| (x & y).count_ones()).sum();
    assert_eq!(both, 0);
    assert_eq!(a[15] >> 40, 0);
    assert_eq!(b[15] >> 40, 0);
}

#[test]
fn finding_admits_alone_ignores_the_tenant() {
    let w = world();
    let spec = ScopeSpec {
        tenant: Some(9),
        ..ScopeSpec::default()
    };
    assert_eq!(w.data.row_tenant[800], 10);
    assert!(spec.admits(&w.data.row_path[800]));
    let plane = prepare(&w.data, &w.frame, Some(spec), POP).expect("plane");
    let words = plane.admit(&w.data.frames, Some(spec), POP).expect("admit");
    assert_eq!(popcount(words), 800);
    let by_admits_alone = w.data.row_path.iter().filter(|p| spec.admits(p)).count();
    assert_eq!(by_admits_alone, 1000);
}

#[test]
fn a_row_plane_is_refused_after_the_source_moves_and_recomputed() {
    let mut w = world();
    let scope = decide("alice").scope;
    let plane = plane_for(&w, "alice");
    assert_eq!(
        popcount(plane.admit(&w.data.frames, scope, POP).expect("ok")),
        200
    );

    w.data.row_path[0] = ScopePath::new(&[3, 0]).expect("depth");
    let _version = w.data.frames.advance(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    assert_eq!(
        plane.admit(&w.data.frames, scope, POP).err(),
        Some(PlaneRefusal::Frame(
            lance_graph_dt64::dr::FrameRefusal::SourceChanged
        ))
    );
    let fresh = w.data.frames.seal();
    let again = prepare(&w.data, &fresh, scope, POP).expect("recomputed");
    let words = again.admit(&w.data.frames, scope, POP).expect("admit");
    assert_eq!(popcount(words), 199);
    assert!(!bit(words, 0));

    // Silent half: a second world with no change and no advance.
    let w2 = world();
    let p2 = plane_for(&w2, "alice");
    let ok = p2.admit(&w2.data.frames, scope, POP).expect("unchanged");
    assert_eq!(popcount(ok), 200);
}

#[test]
fn a_row_plane_prepared_for_one_decision_is_refused_for_another() {
    let w = world();
    let plane = plane_for(&w, "alice");
    assert_eq!(
        plane.admit(&w.data.frames, decide("bob").scope, POP).err(),
        Some(PlaneRefusal::ScopeChanged)
    );
    assert!(plane
        .admit(&w.data.frames, decide("alice").scope, POP)
        .is_ok());
    let other = CoordSpace {
        population: 1,
        coordinates: 2,
    };
    assert_eq!(
        plane
            .admit(&w.data.frames, decide("alice").scope, other)
            .err(),
        Some(PlaneRefusal::CoordinateSpace)
    );
}

#[test]
fn an_unscoped_decision_and_a_reserved_predicate_prepare_no_plane() {
    let w = world();
    assert_eq!(decide("carol").scope, None);
    assert_eq!(decide("ghost").scope, None);
    assert_eq!(
        prepare(&w.data, &w.frame, None, POP).err(),
        Some(PlaneRefusal::Unscoped)
    );
    let reserved = ScopeSpec {
        predicate_key: 2,
        ..ScopeSpec::default()
    };
    assert_eq!(
        prepare(&w.data, &w.frame, Some(reserved), POP).err(),
        Some(PlaneRefusal::PredicateNotInterpreted)
    );
    let deny = prepare(&w.data, &w.frame, Some(ScopeSpec::DENY), POP).expect("deny prepares");
    let words = deny
        .admit(&w.data.frames, Some(ScopeSpec::DENY), POP)
        .expect("admit");
    assert_eq!(popcount(words), 0);
    assert!(prepare(&w.data, &w.frame, decide("alice").scope, POP).is_ok());
}

#[test]
fn role_a_reads_exactly_its_fields_and_rows() {
    let w = world();
    let da = decide("alice");
    let b = bind_decided(&w, &da, SCOPE_A, SCOPE_A).expect("alice");
    let plane = plane_for(&w, "alice");
    let rows = plane.admit(&w.data.frames, da.scope, POP).expect("plane");
    assert_eq!(popcount(rows), 200);
    let s = sock(&w, SCOPE_A);
    let sums: Vec<(u8, i64)> = b
        .borrow_all(&w.store, &w.view, &w.data.frames, &s)
        .expect("check")
        .map(|r| {
            let (f, lane) = r.expect("lane");
            (f, fold_sum(rows, lane))
        })
        .collect();
    assert_eq!(sums, vec![(0, 79_600), (2, 4_079_600), (4, 8_079_600)]);
    assert_eq!(w.data.borrows.get(), 3);

    let db = decide("bob");
    let bb = bind_decided(&w, &db, SCOPE_B, SCOPE_B).expect("bob");
    let pb = plane_for(&w, "bob");
    let brows = pb.admit(&w.data.frames, db.scope, POP).expect("plane");
    assert_eq!(popcount(brows), 40);
    let sb = sock(&w, SCOPE_B);
    let bob4: Vec<i64> = bb
        .borrow_all(&w.store, &w.view, &w.data.frames, &sb)
        .expect("check")
        .filter_map(|r| {
            let (f, lane) = r.expect("lane");
            (f == 4).then(|| fold_sum(brows, lane))
        })
        .collect();
    assert_eq!(bob4, vec![1_616_120]);
}

#[test]
fn finding_a_source_revision_move_refuses_the_address_binding_too() {
    // One `FrameRevisions.source` number is the only change signal a source
    // has, so advancing it for a content change also refuses address
    // bindings. Safe direction; re-pin if content gets its own witness.
    let mut w = world();
    let b = bind_decided(&w, &decide("alice"), SCOPE_A, SCOPE_A).expect("alice");
    let _version = w.data.frames.advance(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    assert_eq!(
        check(&w, &b, SCOPE_A),
        Err(LaneRefusal::Frame(
            lance_graph_dt64::dr::FrameRefusal::SourceChanged
        ))
    );
}
