//! dir-sim results through the DT64 lane socket, at every DR tile width.
//!
//! A `subtree` result over a version's users is held as a selection; its DR
//! axis follows its survivors (DR64, DR256, 2 × DR256, n × DR256). Fields are
//! read at the selected rows as virtual tuples through the version's view, and
//! compared with the view itself as the oracle.
#![cfg(feature = "dt64")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dir_sim::dt64::{DirLanes, SelectRefusal, F_SMTP, F_UPN, LANE_REV};
use lance_graph_dir_sim::{
    groups_where, subtree, GroupWhere, NodeKind, Observation, ObservedNode, UserOrdinal, ValueId,
    VersionStore,
};
use lance_graph_dt64::dr::{Axis, FrameRefusal, Scope, Width};
use lance_graph_dt64::{
    BindStore, ClassSchema, Extent, LaneBinding, LaneRefusal, LaneRequest, LaneSource, RouteAt,
};
use ogar_dir_core::{DirectoryScope, Dn128, Guid128};
use ogar_dir_sim::Attribute;

struct Counting;
thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|a| a.set(a.get() + 1));
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;
fn allocs() -> u64 {
    ALLOCS.with(Cell::get)
}

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));
const USER: u32 = 0x0D1B_0001;
const VIEW_CLASS: ClassId = 0x0D1B;
const POSITION: u16 = 3;
const READER: Scope = Scope { id: 7 };

struct Basis(Vec<FieldRef>);
impl ClassView for Basis {
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
fn basis() -> Basis {
    Basis(vec![
        FieldRef::new("dir:userPrincipalName", "UPN"),
        FieldRef::new("dir:primarySmtp", "Primary SMTP"),
    ])
}

fn user(u: usize) -> Guid128 {
    let mut b = [0u8; 16];
    b[0] = 1;
    b[1..3].copy_from_slice(&(u as u16).to_le_bytes());
    Guid128(b)
}

/// `inside` users under OU [1], `outside` users under OU [2].
fn observation(inside: usize, outside: usize) -> Observation {
    let mut o = Observation {
        scope: SCOPE,
        nodes: Vec::new(),
        members: Vec::new(),
    };
    for u in 0..inside + outside {
        let mut n =
            ObservedNode::user(&format!("u{u}@example.test"), &format!("m{u}@example.test"));
        n.dn = Some(Dn128::new(if u < inside { &[1] } else { &[2] }).unwrap());
        o.nodes.push((user(u), n));
    }
    o
}

fn store() -> BindStore {
    let mut s = BindStore::new();
    s.register(
        &basis(),
        USER,
        ClassSchema {
            schema_rev: LANE_REV,
            field_count: 2,
            extent: Extent::DR64_CUBE,
        },
        Vec::new(),
    )
    .unwrap();
    s
}

fn bind(
    store: &BindStore,
    lanes: &DirLanes<'_, '_>,
    route: RouteAt,
    frame: &lance_graph_dt64::dr::Frame,
) -> Result<LaneBinding, LaneRefusal> {
    store.bind_lanes(
        &basis(),
        lanes.frames(),
        lanes,
        LaneRequest {
            route,
            rev: LANE_REV,
            fields: &WideFieldMask::from_positions(&[F_UPN, F_SMTP]),
            frame,
            scope: READER,
            selection: lanes.space(),
        },
    )
}

fn inside_ou() -> Dn128 {
    Dn128::new(&[1]).unwrap()
}

/// The axis follows the survivors at each width, and every selected cell
/// equals the view's own attribute (the oracle) without any allocation on the
/// read path. Falsifier: a fixed axis, an axis from the population instead of
/// the survivors, or a lane read that ignores the row.
#[test]
fn every_tile_width_reads_the_selected_cells_as_the_view_does() {
    let cases = [
        (40, Axis::new(Width::W64, 1)),
        (64, Axis::new(Width::W64, 1)),
        (65, Axis::new(Width::W256, 1)),
        (200, Axis::new(Width::W256, 1)),
        (512, Axis::new(Width::W256, 2)),
        (1000, Axis::new(Width::W256, 4)),
    ];
    for (inside, want) in cases {
        let mut st = VersionStore::new();
        let v = st.observe("ad", 0, observation(inside, 7)).unwrap();
        let view = st.view(v).unwrap();
        let mut lanes = DirLanes::new(&view, v, USER, READER);
        let kept = subtree(&view, SCOPE, NodeKind::User, &inside_ou()).unwrap();
        assert_eq!(kept.len(), inside, "the fixture selects `inside` users");
        let picked = lanes.select(&kept).unwrap();
        assert_eq!(picked.axis, want, "{inside} survivors");
        assert_eq!(
            lanes.identity_axis(),
            Axis::for_survivors(inside + 7),
            "the population's axis is its own"
        );

        let mut bs = store();
        let version = bs.route(POSITION, USER).unwrap();
        let route = RouteAt {
            position: POSITION,
            version,
        };
        let frame = lanes.frames().seal();
        let b = bind(&bs, &lanes, route, &frame).unwrap();

        let basis = basis();
        let a0 = allocs();
        let mut agree = 0usize;
        for (f, attr) in [(F_UPN, Attribute::Upn), (F_SMTP, Attribute::PrimarySmtp)] {
            let (lane, rows) = b
                .selected(&bs, &basis, lanes.frames(), &lanes, f, picked.sel)
                .unwrap();
            assert_eq!(rows.len(), inside);
            for &r in rows {
                let g = view.user_guid(UserOrdinal(r as u16)).unwrap();
                if lane.get(r) == view.attr(&g, attr).map(|x| x.0) && lane.get(r).is_some() {
                    agree += 1;
                }
            }
        }
        assert_eq!(allocs() - a0, 0, "the read path allocates nothing");
        assert_eq!(agree, 2 * inside, "every selected cell equals the oracle");
    }
}

/// A re-observed version reads its overridden and created users through the
/// same socket, and a binding made over the previous version is refused
/// before a lane is borrowed. Falsifier: a lane over the base snapshot only
/// (the override or the created user reads wrong), or a shared frame store.
#[test]
fn a_reobserved_version_reads_its_overlay_and_refuses_the_old_binding() {
    let mut st = VersionStore::new();
    let v0 = st.observe("ad", 0, observation(3, 1)).unwrap();
    let mut next = observation(3, 1);
    next.nodes[1].1.primary_smtp = Some("moved@example.test".into());
    let mut created = ObservedNode::user("new@example.test", "new-mail@example.test");
    created.dn = Some(inside_ou());
    next.nodes.push((user(9), created));
    let v1 = st.reobserve("ad", 1, next).unwrap();
    assert_ne!(v0, v1);

    let mut bs = store();
    let version = bs.route(POSITION, USER).unwrap();
    let route = RouteAt {
        position: POSITION,
        version,
    };

    let old_view = st.view(v0).unwrap();
    let old = DirLanes::new(&old_view, v0, USER, READER);
    let old_binding = bind(&bs, &old, route, &old.frames().seal()).unwrap();

    let view = st.view(v1).unwrap();
    assert!(view.delta_len() > 0, "v1 is an overlay over v0's snapshot");
    let mut lanes = DirLanes::new(&view, v1, USER, READER);
    let kept = subtree(&view, SCOPE, NodeKind::User, &inside_ou()).unwrap();
    assert_eq!(kept.len(), 4, "three observed and one created user inside");
    let picked = lanes.select(&kept).unwrap();

    // The old binding against the new version: refused, nothing borrowed.
    assert_eq!(
        old_binding
            .selected(&bs, &basis(), lanes.frames(), &lanes, F_SMTP, picked.sel)
            .err(),
        Some(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    );

    let b = bind(&bs, &lanes, route, &lanes.frames().seal()).unwrap();
    let (lane, rows) = b
        .selected(&bs, &basis(), lanes.frames(), &lanes, F_SMTP, picked.sel)
        .unwrap();
    let read = |g: Guid128| {
        let r = view.user_ordinal(&g).unwrap().0 as u32;
        assert!(rows.contains(&r));
        lane.get(r)
            .and_then(|id| view.dicts().value(ValueId(id)))
            .map(str::to_string)
    };
    assert_eq!(read(user(1)).as_deref(), Some("moved@example.test"));
    assert_eq!(read(user(9)).as_deref(), Some("new-mail@example.test"));
    assert_eq!(read(user(0)).as_deref(), Some("m0@example.test"));
}

/// A result over groups is not a selection of users. Falsifier: admitting any
/// bitmap whatever its population.
#[test]
fn a_result_over_another_population_is_refused() {
    let mut o = observation(5, 0);
    for g in 0..3 {
        o.nodes.push((user(100 + g), ObservedNode::group()));
    }
    let mut st = VersionStore::new();
    let v = st.observe("ad", 0, o).unwrap();
    let view = st.view(v).unwrap();
    let mut lanes = DirLanes::new(&view, v, USER, READER);
    let groups = groups_where(&view, &GroupWhere::And(Vec::new()));
    assert_eq!(
        lanes.select(&groups),
        Err(SelectRefusal::NotUsers { rows: 3, users: 5 })
    );
    // Silence half: a users result of the same version is admitted.
    let users = subtree(&view, SCOPE, NodeKind::User, &inside_ou()).unwrap();
    assert!(lanes.select(&users).is_ok());
}

/// A basis that lists SMTP before UPN reads each field's own attribute, and a
/// predicate the directory has no lane for is unbound. Falsifier: lanes taken
/// by field position, which reads UPN values under the SMTP field.
#[test]
fn directory_lanes_bind_by_predicate_not_position() {
    struct Swapped(Vec<FieldRef>);
    impl ClassView for Swapped {
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
    let swapped = Swapped(vec![
        FieldRef::new("dir:primarySmtp", "Primary SMTP"),
        FieldRef::new("dir:userPrincipalName", "UPN"),
        FieldRef::new("dir:department", "Department"),
    ]);
    let mut st = VersionStore::new();
    let v = st.observe("ad", 0, observation(3, 0)).unwrap();
    let view = st.view(v).unwrap();
    let mut lanes = DirLanes::new(&view, v, USER, READER);
    let kept = subtree(&view, SCOPE, NodeKind::User, &inside_ou()).unwrap();
    let picked = lanes.select(&kept).unwrap();
    let mut bs = BindStore::new();
    bs.register(
        &swapped,
        USER,
        ClassSchema {
            schema_rev: LANE_REV,
            field_count: 3,
            extent: Extent::DR64_CUBE,
        },
        Vec::new(),
    )
    .unwrap();
    let version = bs.route(POSITION, USER).unwrap();
    let route = RouteAt {
        position: POSITION,
        version,
    };
    let frame = lanes.frames().seal();
    let req = |fields: &[u8]| LaneRequest {
        route,
        rev: LANE_REV,
        fields: Box::leak(Box::new(WideFieldMask::from_positions(fields))),
        frame: Box::leak(Box::new(frame.clone())),
        scope: READER,
        selection: lanes.space(),
    };
    let b = bs
        .bind_lanes(&swapped, lanes.frames(), &lanes, req(&[0, 1]))
        .unwrap();
    let (smtp, rows) = b
        .selected(&bs, &swapped, lanes.frames(), &lanes, 0, picked.sel)
        .unwrap();
    for &r in rows {
        let g = view.user_guid(UserOrdinal(r as u16)).unwrap();
        assert_eq!(
            smtp.get(r),
            view.attr(&g, Attribute::PrimarySmtp).map(|x| x.0)
        );
        assert_ne!(smtp.get(r), view.attr(&g, Attribute::Upn).map(|x| x.0));
    }
    assert_eq!(
        bs.bind_lanes(&swapped, lanes.frames(), &lanes, req(&[2]))
            .err(),
        Some(LaneRefusal::Unbound {
            class: USER,
            field: 2
        })
    );
}
