//! D-DIRSIM-LICENSE-DT64-1: a version's licence planes through the DT64 lane
//! and selection sockets, read in the row order of the version's directory
//! lanes. Every answer is compared with an independent scalar oracle built
//! from the fixture's own rule, never from the planes.
#![cfg(feature = "dt64")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::time::Instant;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dir_sim::dt64::{
    DirLanes, LicenseBasis, LicenseLanes, LicenseSourceRefusal, SelectRefusal, F_UPN, LANE_REV,
    LICENSE_REV,
};
use lance_graph_dir_sim::license::{
    Coverage, ExplainRefusal, LicenseError, LicensePlanes, Via, MAX_LICENSE_STATES,
};
use lance_graph_dir_sim::{
    subtree, NodeKind, Observation, ObservedNode, UserOrdinal, VersionStore,
};
use lance_graph_dt64::dr::{FrameRefusal, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneBinding, LaneRefusal, LaneRequest, LaneSource,
    RouteAt,
};
use ogar_az::license::{ingest_licenses_page, LicensesPage, PathState};
use ogar_dir_core::{DirectoryScope, Dn128, Guid128};

// ---- counting allocator: allocations and bytes ------------------------------

struct Counting;
thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|a| a.set(a.get() + 1));
        let _ = BYTES.try_with(|b| b.set(b.get() + l.size() as u64));
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
fn allocs() -> (u64, u64) {
    (ALLOCS.with(Cell::get), BYTES.with(Cell::get))
}

// ---- fixture ----------------------------------------------------------------

const TENANT: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));
const USER: u32 = 0x0D1B_0001;
const LICENSE: u32 = 0x0D1C_0001;
const USER_POS: u16 = 3;
const LICENSE_POS: u16 = 4;
const READER: Scope = Scope { id: 7 };
const USERS: usize = 512;
const INSIDE: usize = 5;
const SKUS: usize = 8;
const GROUPS: usize = 5;
const STATES: [&str; 4] = ["Active", "Disabled", "Error", "PendingInput"];

fn user(u: usize) -> Guid128 {
    let mut b = [0u8; 16];
    b[0] = 1;
    b[1..3].copy_from_slice(&(u as u16).to_le_bytes());
    Guid128(b)
}
fn sku(i: usize) -> String {
    format!("00000000-0000-4000-8000-{:012x}", 0x5100 + i)
}
fn group(i: usize) -> String {
    format!("00000000-0000-4000-8000-{:012x}", 0x6100 + i)
}

/// The fixture's rule, written once, read by both the pages and the oracle.
/// `None`: the property is absent. Each path is `(sku, group, state)`.
fn rule(u: usize) -> Option<Vec<(usize, Option<usize>, usize)>> {
    if u % 11 == 5 {
        return None;
    }
    if u % 13 == 6 {
        return Some(Vec::new());
    }
    let (s, st) = (u % SKUS, (u / SKUS) % STATES.len());
    let mut p = Vec::new();
    if u % 3 == 0 {
        p.push((s, None, st));
    }
    p.push((s, Some(u % GROUPS), st));
    if u % 7 == 0 {
        // A second assigning group for the same SKU in the same state.
        p.push((s, Some((u + 1) % GROUPS), st));
    }
    Some(p)
}

fn entry(u: usize) -> String {
    match rule(u) {
        None => format!(r#"{{"id":"{}"}}"#, user(u)),
        Some(paths) => {
            let items: Vec<String> = paths
                .iter()
                .map(|&(s, g, st)| match g {
                    Some(g) => format!(
                        r#"{{"skuId":"{}","state":"{}","assignedByGroup":"{}"}}"#,
                        sku(s),
                        STATES[st],
                        group(g)
                    ),
                    None => format!(r#"{{"skuId":"{}","state":"{}"}}"#, sku(s), STATES[st]),
                })
                .collect();
            format!(
                r#"{{"id":"{}","licenseAssignmentStates":[{}]}}"#,
                user(u),
                items.join(",")
            )
        }
    }
}

/// The users in four pages of 128, linked except the last, in `order`.
fn pages(order: [usize; 4], reverse_users: bool) -> Vec<LicensesPage> {
    order
        .iter()
        .map(|&p| {
            let mut us: Vec<usize> = (p * 128..(p + 1) * 128).collect();
            if reverse_users {
                us.reverse();
            }
            let body: Vec<String> = us.into_iter().map(entry).collect();
            let mut page =
                ingest_licenses_page(&format!(r#"{{"value":[{}]}}"#, body.join(","))).unwrap();
            page.next_link = (p < 3).then(|| format!("https://graph.example/users?page={}", p + 1));
            page
        })
        .collect()
}

fn inside_ou() -> Dn128 {
    Dn128::new(&[1]).unwrap()
}

fn observation() -> Observation {
    let mut o = Observation {
        scope: TENANT,
        nodes: Vec::new(),
        members: Vec::new(),
    };
    for u in 0..USERS {
        let mut n =
            ObservedNode::user(&format!("u{u}@example.test"), &format!("m{u}@example.test"));
        n.dn = Some(Dn128::new(if u < INSIDE { &[1] } else { &[2] }).unwrap());
        o.nodes.push((user(u), n));
    }
    o
}

struct UserBasis(Vec<FieldRef>);
impl ClassView for UserBasis {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == lance_graph_dt64::view_class(USER) {
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
fn user_basis() -> UserBasis {
    UserBasis(vec![
        FieldRef::new("dir:userPrincipalName", "UPN"),
        FieldRef::new("dir:primarySmtp", "Primary SMTP"),
    ])
}

/// The store, binding the users class and the licence class over `basis`.
fn store(basis: &LicenseBasis) -> (BindStore, RouteAt, RouteAt) {
    let mut bs = BindStore::new();
    bs.register(
        &user_basis(),
        USER,
        ClassSchema {
            schema_rev: LANE_REV,
            field_count: 2,
            extent: Extent::DR64_CUBE,
        },
        Vec::new(),
    )
    .unwrap();
    bs.register(
        basis,
        LICENSE,
        ClassSchema {
            schema_rev: LICENSE_REV,
            field_count: basis.len(),
            extent: Extent::DR64_CUBE,
        },
        Vec::new(),
    )
    .unwrap();
    let uv = bs.route(USER_POS, USER).unwrap();
    let lv = bs.route(LICENSE_POS, LICENSE).unwrap();
    (
        bs,
        RouteAt {
            position: USER_POS,
            version: uv,
        },
        RouteAt {
            position: LICENSE_POS,
            version: lv,
        },
    )
}

fn bind_codes(
    bs: &BindStore,
    basis: &LicenseBasis,
    lic: &LicenseLanes<'_>,
    route: RouteAt,
    fields: &[u8],
    scope: Scope,
    selection: CoordSpace,
) -> Result<LaneBinding, LaneRefusal> {
    let frame = lic.frames().seal();
    bs.bind_lanes(
        basis,
        lic.frames(),
        lic,
        LaneRequest {
            route,
            rev: LICENSE_REV,
            fields: &WideFieldMask::from_positions(fields),
            frame: &frame,
            scope,
            selection,
        },
    )
}

/// The oracle: does user `u` hold the code `(sku, via, state)`?
fn oracle_holds(u: usize, s: usize, group_path: bool, st: usize) -> bool {
    rule(u).is_some_and(|p| {
        p.iter()
            .any(|&(ps, pg, pst)| ps == s && pg.is_some() == group_path && pst == st)
    })
}

/// The code index of `(sku, via, state)` in `planes`, found by content.
fn code(planes: &LicensePlanes, s: usize, group_path: bool, st: usize) -> usize {
    let want = Guid128::parse(&sku(s)).unwrap();
    let state = match STATES[st] {
        "Active" => PathState::Active,
        "Disabled" => PathState::Disabled,
        "Error" => PathState::Error,
        other => PathState::Other(other.into()),
    };
    let via = if group_path { Via::Group } else { Via::Direct };
    planes
        .states()
        .iter()
        .position(|c| c.sku == want && c.via == via && c.state == state)
        .unwrap()
}

/// The version's row (user ordinal) of fixture user `u`. Ordinals follow the
/// directory's own order, not the fixture's numbering.
fn row(view: &lance_graph_dir_sim::View<'_>, u: usize) -> u32 {
    u32::from(view.user_ordinal(&user(u)).unwrap().0)
}

/// The oracle's users, as sorted rows of `view`.
fn rows_of(view: &lance_graph_dir_sim::View<'_>, users: impl Iterator<Item = usize>) -> Vec<u32> {
    let mut r: Vec<u32> = users.map(|u| row(view, u)).collect();
    r.sort_unstable();
    r
}

// ---- Gate 2: native selection -----------------------------------------------

/// Select by licence state, intersect with a directory selection, resolve a
/// selected user's codes under the source frame, and keep holders apart from
/// paths. Every answer against the scalar oracle.
#[test]
fn licence_states_select_intersect_and_resolve_against_the_oracle() {
    let mut st = VersionStore::new();
    let v = st.observe("ad", 0, observation()).unwrap();
    let view = st.view(v).unwrap();
    let planes = LicensePlanes::build(&view, v, pages([0, 1, 2, 3], false), Vec::new()).unwrap();
    assert_eq!(
        planes.states().len(),
        MAX_LICENSE_STATES,
        "one full DR64 tile"
    );
    assert_eq!(planes.coverage(), Coverage::Complete);

    let mut dir = DirLanes::new(&view, v, USER, READER);
    let mut lic = LicenseLanes::over(&dir, &planes, LICENSE, READER).unwrap();
    let basis = LicenseBasis::of(&planes, LICENSE);
    let (bs, user_route, lic_route) = store(&basis);

    // 1. Users with (sku 3, group, Disabled): select, bind, read.
    let c = code(&planes, 3, true, 1);
    let field = c as u8;
    let picked = lic.select_state(c).unwrap();
    let b = bind_codes(&bs, &basis, &lic, lic_route, &[field], READER, dir.space()).unwrap();
    let (plane, rows) = b
        .selected(&bs, &basis, lic.frames(), &lic, field, picked.sel)
        .unwrap();
    let want = rows_of(&view, (0..USERS).filter(|&u| oracle_holds(u, 3, true, 1)));
    assert_eq!(rows, want.as_slice());
    assert!(!want.is_empty() && want.len() * 3 < USERS, "a real filter");
    assert!(rows.iter().all(|&r| plane.contains(r)));

    // 2. The directory's five users ∩ users holding (sku 0, group, Active).
    let ou = subtree(&view, TENANT, NodeKind::User, &inside_ou()).unwrap();
    assert_eq!(ou.len(), INSIDE);
    let c0 = code(&planes, 0, true, 0);
    let both = planes.with_state(c0).unwrap().and(&ou).unwrap();
    let in_dir = dir.select(&both).unwrap();
    let in_lic = lic.select(&both).unwrap();
    let want_both = rows_of(&view, (0..INSIDE).filter(|&u| oracle_holds(u, 0, true, 0)));
    assert!(
        !want_both.is_empty() && want_both.len() < INSIDE,
        "a real intersection"
    );
    let ub = bs
        .bind_lanes(
            &user_basis(),
            dir.frames(),
            &dir,
            LaneRequest {
                route: user_route,
                rev: LANE_REV,
                fields: &WideFieldMask::from_positions(&[F_UPN]),
                frame: &dir.frames().seal(),
                scope: READER,
                selection: dir.space(),
            },
        )
        .unwrap();
    let (upn, dir_rows) = ub
        .selected(&bs, &user_basis(), dir.frames(), &dir, F_UPN, in_dir.sel)
        .unwrap();
    assert_eq!(dir_rows, want_both.as_slice());
    for &r in dir_rows {
        let g = view.user_guid(UserOrdinal(r as u16)).unwrap();
        assert_eq!(
            upn.get(r),
            view.attr(&g, ogar_dir_sim::Attribute::Upn).map(|x| x.0)
        );
    }
    let b0 = bind_codes(
        &bs,
        &basis,
        &lic,
        lic_route,
        &[c0 as u8],
        READER,
        dir.space(),
    )
    .unwrap();
    let (p0, lic_rows) = b0
        .selected(&bs, &basis, lic.frames(), &lic, c0 as u8, in_lic.sel)
        .unwrap();
    assert_eq!(lic_rows, dir_rows, "one row order");
    assert!(lic_rows.iter().all(|&r| p0.contains(r)));

    // 3. One selected user's codes under the source frame, and the paths
    //    behind them from the evidence (u = 0: direct + two groups).
    let all: Vec<u8> = (0..basis.len() as u8).collect();
    let ball = bind_codes(&bs, &basis, &lic, lic_route, &all, READER, dir.space()).unwrap();
    let u0 = row(&view, 0);
    let mut held = 0;
    for item in ball.borrow_all(&bs, &basis, lic.frames(), &lic).unwrap() {
        let (f, plane) = item.unwrap();
        let c = &planes.states()[usize::from(f)];
        let s = (0..SKUS)
            .find(|&i| Guid128::parse(&sku(i)).unwrap() == c.sku)
            .unwrap();
        let st_i = STATES
            .iter()
            .position(|t| match &c.state {
                PathState::Active => *t == "Active",
                PathState::Disabled => *t == "Disabled",
                PathState::Error => *t == "Error",
                PathState::Other(x) => t == x,
            })
            .unwrap();
        let want = oracle_holds(0, s, c.via == Via::Group, st_i);
        assert_eq!(plane.contains(u0), want, "field {f}");
        held += usize::from(want);
    }
    assert_eq!(held, 2, "u0 holds the direct and the group code");
    let gc = code(&planes, 0, true, 0);
    let groups: Vec<Guid128> = planes
        .explain(UserOrdinal(u0 as u16), gc)
        .unwrap()
        .map(|p| p.assigned_by_group.unwrap())
        .collect();
    assert_eq!(
        groups,
        vec![
            Guid128::parse(&group(0)).unwrap(),
            Guid128::parse(&group(1)).unwrap()
        ],
        "one bit, two assigning groups, both recovered"
    );

    // 4. Holders versus assignment paths for sku 0.
    let sku0 = Guid128::parse(&sku(0)).unwrap();
    let want_holders = (0..USERS)
        .filter(|&u| rule(u).is_some_and(|p| p.iter().any(|x| x.0 == 0)))
        .count();
    let want_paths: usize = (0..USERS)
        .filter_map(rule)
        .map(|p| p.iter().filter(|x| x.0 == 0).count())
        .sum();
    assert_eq!(planes.holder_count(&sku0), want_holders);
    assert_eq!(planes.path_count(&sku0), want_paths);
    assert!(
        want_paths > want_holders,
        "the fixture has multi-path holders"
    );
}

/// A 65th distinct code is refused; 64 fit (the fixture above).
#[test]
fn the_65th_licence_state_code_is_refused() {
    let mut st = VersionStore::new();
    let v = st.observe("ad", 0, observation()).unwrap();
    let view = st.view(v).unwrap();
    let mut ps = pages([0, 1, 2, 3], false);
    let extra = ingest_licenses_page(&format!(
        r#"{{"value":[{{"id":"{}","licenseAssignmentStates":[{{"skuId":"{}","state":"Suspended"}}]}}]}}"#,
        user(5),
        sku(0)
    ))
    .unwrap();
    // u5's property was absent in the pages; list it in a page of its own.
    ps.last_mut().unwrap().next_link = Some("https://graph.example/users?page=4".into());
    let mut ps2: Vec<LicensesPage> = ps
        .into_iter()
        .map(|mut p| {
            p.users.retain(|x| x.user != user(5));
            p
        })
        .collect();
    ps2.push(extra);
    assert_eq!(
        LicensePlanes::build(&view, v, ps2, Vec::new()),
        Err(LicenseError::TooManyStates {
            found: 65,
            limit: 64
        })
    );
}

// ---- Gate 4: typed refusals -------------------------------------------------

#[test]
fn stale_foreign_and_mismatched_bindings_are_refused() {
    let mut st = VersionStore::new();
    let v0 = st.observe("ad", 0, observation()).unwrap();
    // The later revision exists before any view borrows the store.
    let v1 = st.reobserve("ad", 1, observation()).unwrap();
    let view0 = st.view(v0).unwrap();
    let planes0 = LicensePlanes::build(&view0, v0, pages([0, 1, 2, 3], false), Vec::new()).unwrap();
    let dir0 = DirLanes::new(&view0, v0, USER, READER);
    let lic0 = LicenseLanes::over(&dir0, &planes0, LICENSE, READER).unwrap();
    let basis0 = LicenseBasis::of(&planes0, LICENSE);
    let (bs, _, route) = store(&basis0);
    let c = code(&planes0, 2, false, 2) as u8;
    let b0 = bind_codes(&bs, &basis0, &lic0, route, &[c], READER, dir0.space()).unwrap();

    // Reordered pages: the same code at the same index, a new source frame.
    let planes_r = LicensePlanes::build(&view0, v0, pages([3, 1, 0, 2], true), Vec::new()).unwrap();
    assert_eq!(planes_r.states(), planes0.states());
    let lic_r = LicenseLanes::over(&dir0, &planes_r, LICENSE, READER).unwrap();
    assert_eq!(
        b0.check(&bs, &basis0, lic_r.frames(), &lic_r).err(),
        Some(LaneRefusal::Frame(FrameRefusal::ForeignFrame)),
        "a binding is never read against another source's planes"
    );
    let br = bind_codes(&bs, &basis0, &lic_r, route, &[c], READER, dir0.space()).unwrap();
    assert_eq!(br.addr(c).unwrap().lane, b0.addr(c).unwrap().lane);

    // A later observation revision: users 1..512 lose their direct paths
    // except for sku 7, so codes vanish and indices move.
    let view1 = st.view(v1).unwrap();
    let mut later = pages([0, 1, 2, 3], false);
    for p in &mut later {
        for u in &mut p.users {
            if let Some(paths) = &mut u.paths {
                paths.retain(|x| {
                    x.assigned_by_group.is_some() || x.sku == Guid128::parse(&sku(7)).unwrap()
                });
            }
        }
    }
    let planes1 = LicensePlanes::build(&view1, v1, later, Vec::new()).unwrap();
    assert!(planes1.states().len() < planes0.states().len());
    let dir1 = DirLanes::new(&view1, v1, USER, READER);
    let lic1 = LicenseLanes::over(&dir1, &planes1, LICENSE, READER).unwrap();
    // Stale frame / source generation.
    assert_eq!(
        b0.check(&bs, &basis0, lic1.frames(), &lic1).err(),
        Some(LaneRefusal::Frame(FrameRefusal::ForeignFrame))
    );
    // The old basis names a code the new planes do not have: unbound, never
    // read at the slot that now holds another code.
    let gone = code(&planes0, 2, false, 2) as u8;
    assert_eq!(
        bind_codes(&bs, &basis0, &lic1, route, &[gone], READER, dir1.space()).err(),
        Some(LaneRefusal::Unbound {
            class: LICENSE,
            field: gone
        })
    );
    assert_ne!(
        planes1.predicate_iri(usize::from(gone)),
        planes0.predicate_iri(usize::from(gone)),
        "another code now sits at that index"
    );
    // A code that survived resolves by identity to its NEW index.
    let kept_code = code(&planes0, 7, false, 0) as u8;
    let moved = bind_codes(
        &bs,
        &basis0,
        &lic1,
        route,
        &[kept_code],
        READER,
        dir1.space(),
    )
    .unwrap();
    assert_eq!(
        usize::from(moved.addr(kept_code).unwrap().lane),
        code(&planes1, 7, false, 0)
    );
    assert_ne!(
        usize::from(kept_code),
        code(&planes1, 7, false, 0),
        "it moved"
    );
    // The new codebook cannot be checked against a store bound over the old.
    let basis1 = LicenseBasis::of(&planes1, LICENSE);
    assert!(matches!(
        bind_codes(&bs, &basis1, &lic1, route, &[0], READER, dir1.space()),
        Err(LaneRefusal::Bind(_))
    ));

    // Mismatched population: v0's planes under v1's directory lanes.
    assert_eq!(
        LicenseLanes::over(&dir1, &planes0, LICENSE, READER).err(),
        Some(LicenseSourceRefusal::OtherVersion {
            planes: v0,
            lanes: v1
        })
    );
    // Foreign reader scope.
    assert!(matches!(
        bind_codes(
            &bs,
            &basis0,
            &lic0,
            route,
            &[c],
            Scope { id: 8 },
            dir0.space()
        ),
        Err(LaneRefusal::Scope { .. })
    ));
    // Wrong user row order.
    assert!(matches!(
        bind_codes(&bs, &basis0, &lic0, route, &[c], READER, dir1.space()),
        Err(LaneRefusal::CoordinateSpace { .. })
    ));
    // A field outside the class basis.
    assert!(matches!(
        bind_codes(&bs, &basis0, &lic0, route, &[200], READER, dir0.space()),
        Err(LaneRefusal::FieldNotInBasis { .. })
    ));
    // An incomplete collection is not served.
    let partial = LicensePlanes::build(
        &view0,
        v0,
        pages([0, 1, 2, 2], false)[..3].to_vec(),
        Vec::new(),
    )
    .unwrap();
    assert_eq!(partial.coverage(), Coverage::Partial);
    assert_eq!(
        LicenseLanes::over(&dir0, &partial, LICENSE, READER).err(),
        Some(LicenseSourceRefusal::Incomplete(Coverage::Partial))
    );
    // An exact path explanation for a user whose paths were not read.
    assert_eq!(
        planes0.explain(UserOrdinal(row(&view0, 5) as u16), 0).err(),
        Some(ExplainRefusal::NotRead)
    );
    // A selection of another source's row order is refused.
    let mut lic0m = LicenseLanes::over(&dir0, &planes0, LICENSE, READER).unwrap();
    assert_eq!(
        lic0m.select_state(64).err(),
        Some(SelectRefusal::NoSuchCode(64))
    );
}

/// A licence property that is not collected as planes (`assignedLicenses`)
/// has no lane: a basis naming it is unbound, never read as empty.
#[test]
fn an_uncollected_licence_property_has_no_lane() {
    struct Mixed(Vec<FieldRef>);
    impl ClassView for Mixed {
        fn fields(&self, class: ClassId) -> &[FieldRef] {
            if class == lance_graph_dt64::view_class(LICENSE) {
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
    let mut st = VersionStore::new();
    let v = st.observe("ad", 0, observation()).unwrap();
    let view = st.view(v).unwrap();
    let planes = LicensePlanes::build(&view, v, pages([0, 1, 2, 3], false), Vec::new()).unwrap();
    let dir = DirLanes::new(&view, v, USER, READER);
    let lic = LicenseLanes::over(&dir, &planes, LICENSE, READER).unwrap();
    let mixed = Mixed(vec![
        FieldRef::new(planes.predicate_iri(0).unwrap(), "code 0"),
        FieldRef::new("urn:graph:assignedLicenses:skuId", "effective SKUs"),
    ]);
    let mut bs = BindStore::new();
    bs.register(
        &mixed,
        LICENSE,
        ClassSchema {
            schema_rev: LICENSE_REV,
            field_count: 2,
            extent: Extent::DR64_CUBE,
        },
        Vec::new(),
    )
    .unwrap();
    let version = bs.route(LICENSE_POS, LICENSE).unwrap();
    let route = RouteAt {
        position: LICENSE_POS,
        version,
    };
    let req = |fields: &[u8]| {
        let mask = WideFieldMask::from_positions(fields);
        let frame = lic.frames().seal();
        bs.bind_lanes(
            &mixed,
            lic.frames(),
            &lic,
            LaneRequest {
                route,
                rev: LICENSE_REV,
                fields: &mask,
                frame: &frame,
                scope: READER,
                selection: dir.space(),
            },
        )
        .map(|_| ())
    };
    assert_eq!(req(&[0]), Ok(()), "the code field binds");
    assert_eq!(
        req(&[1]),
        Err(LaneRefusal::Unbound {
            class: LICENSE,
            field: 1
        })
    );
}

// ---- Gate 3: no-explosion proof and measured projection ----------------------

fn median(mut xs: Vec<u128>) -> u128 {
    xs.sort_unstable();
    xs[xs.len() / 2]
}

/// The whole operation over 512 users, 64 codes, 5 groups and 5 selected
/// users: phase timings (reported, not gated), the bits examined, and the
/// allocations and bytes of each phase. The query phases allocate a few
/// hundred bytes; a user × code × group table would be 512 × 64 × 5 cells.
/// Falsifier: any per-user object or Cartesian table on the query path
/// moves the byte count past the bound.
#[test]
fn the_512_user_projection_builds_no_cartesian_table() {
    const RUNS: usize = 21;
    let mut st = VersionStore::new();
    let v = st.observe("ad", 0, observation()).unwrap();
    let view = st.view(v).unwrap();
    let source = pages([2, 0, 3, 1], true);

    let mut build_ns = Vec::new();
    let mut planes = None;
    let (a, by) = allocs();
    for _ in 0..RUNS {
        let input = source.clone();
        let t = Instant::now();
        planes = Some(LicensePlanes::build(&view, v, input, Vec::new()).unwrap());
        build_ns.push(t.elapsed().as_nanos());
    }
    let planes = planes.unwrap();
    let (a2, by2) = allocs();
    let _ = (a2 - a, by2 - by);

    let mut dir = DirLanes::new(&view, v, USER, READER);
    let ou = subtree(&view, TENANT, NodeKind::User, &inside_ou()).unwrap();
    let c = code(&planes, 0, true, 0);

    // Binding, cold: source, basis, store, one bind.
    let (a0, b0) = allocs();
    let t = Instant::now();
    let mut lic = LicenseLanes::over(&dir, &planes, LICENSE, READER).unwrap();
    let basis = LicenseBasis::of(&planes, LICENSE);
    let (bs, _, route) = store(&basis);
    let binding = bind_codes(&bs, &basis, &lic, route, &[c as u8], READER, dir.space()).unwrap();
    let bind_ns = t.elapsed().as_nanos();
    let (a1, b1) = allocs();

    // Selection positions: state ∩ directory, held by both sources.
    let t = Instant::now();
    let both = planes.with_state(c).unwrap().and(&ou).unwrap();
    let in_lic = lic.select(&both).unwrap();
    let _in_dir = dir.select(&both).unwrap();
    let select_ns = t.elapsed().as_nanos();
    let (a2, b2) = allocs();

    // Warm reads: check, borrow, read every selected user's bit.
    let mut read_ns = Vec::with_capacity(RUNS);
    let mut examined = 0usize;
    let (a3, _) = allocs();
    for _ in 0..RUNS {
        let t = Instant::now();
        let (plane, rows) = binding
            .selected(&bs, &basis, lic.frames(), &lic, c as u8, in_lic.sel)
            .unwrap();
        let mut held = 0usize;
        for &r in rows {
            held += usize::from(plane.contains(r));
            examined += 1;
        }
        read_ns.push(t.elapsed().as_nanos());
        assert_eq!(held, rows.len());
    }
    let (a4, _) = allocs();

    // Terminal materialization: the selected users' UPNs as owned strings.
    let (a5, b5) = allocs();
    let t = Instant::now();
    let upns: Vec<String> = both
        .rows()
        .into_iter()
        .filter_map(|r| view.user_guid(UserOrdinal(r as u16)))
        .filter_map(|g| view.attr(&g, ogar_dir_sim::Attribute::Upn))
        .filter_map(|id| view.dicts().value(id).map(str::to_string))
        .collect();
    let terminal_ns = t.elapsed().as_nanos();
    let (a6, b6) = allocs();

    let want: Vec<String> = (0..INSIDE)
        .filter(|&u| oracle_holds(u, 0, true, 0))
        .map(|u| format!("u{u}@example.test"))
        .collect();
    assert_eq!(upns, want, "the oracle");
    assert_eq!(a4 - a3, 0, "warm reads allocate nothing");

    let plane_bytes = planes.states().len() * USERS.div_ceil(64) * 8;
    let cartesian_bits = USERS * MAX_LICENSE_STATES * GROUPS;
    let query_bytes = (b2 - b1) as usize;
    eprintln!(
        "D-DIRSIM-LICENSE-DT64-1 projection, {USERS} users, {} codes, {GROUPS} groups:",
        planes.states().len()
    );
    eprintln!(
        "  plane construction      median {:>8} ns ({RUNS} runs)",
        median(build_ns)
    );
    eprintln!(
        "  binding (cold)          {:>8} ns, {} allocs, {} B",
        bind_ns,
        a1 - a0,
        b1 - b0
    );
    eprintln!(
        "  selection positions     {:>8} ns, {} allocs, {} B",
        select_ns,
        a2 - a1,
        query_bytes
    );
    eprintln!(
        "  warm read               median {:>8} ns, 0 allocs, {} bits examined in {RUNS} runs",
        median(read_ns),
        examined
    );
    eprintln!(
        "  terminal (UPN strings)  {:>8} ns, {} allocs, {} B",
        terminal_ns,
        a6 - a5,
        b6 - b5
    );
    eprintln!(
        "  resident planes {plane_bytes} B; a user x code x group bit table would be {} B",
        cartesian_bits / 8
    );
    // The query path holds two 8-word bitmaps and two position lists: far
    // below even a bit-packed Cartesian table.
    assert!(
        query_bytes * 8 < cartesian_bits / 8,
        "{query_bytes} B on the query path"
    );
    assert_eq!(examined, RUNS * want.len());
}
