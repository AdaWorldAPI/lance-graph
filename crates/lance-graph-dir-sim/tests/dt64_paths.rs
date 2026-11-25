//! A user's location as bound DT64 coordinates: Path1..Path`n`.
//!
//! The version holds a location as a `Dn128` (level codes against an
//! `OuDictionary`), never as a DN string. [`UserBasis`] names UPN, primary SMTP
//! and the OU path levels; each level binds a lane that reads that level's
//! code of the user's location, cell by cell, through the version's view. No
//! DN is parsed or formatted and no path is copied out.
#![cfg(feature = "dt64")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dir_sim::dt64::{DirLanes, UserBasis, F_PATH1, F_UPN, LANE_REV};
use lance_graph_dir_sim::{Observation, ObservedNode, Rule, UserOrdinal, VersionStore, View};
use lance_graph_dt64::dr::Scope;
use lance_graph_dt64::{
    BindStore, ClassSchema, Extent, LaneBinding, LaneRefusal, LaneRequest, LaneSource, RouteAt,
};
use ogar_dir_core::{DirectoryScope, Dn128, Guid128};
use ogar_dir_sim::{Change, EvidenceRef, RuleId};

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
const POSITION: u16 = 3;
const READER: Scope = Scope { id: 7 };
const LEVELS: usize = 3;

fn user(u: u8) -> Guid128 {
    let mut b = [0u8; 16];
    b[0] = 1;
    b[1] = u;
    Guid128(b)
}

/// The fixture's locations, as level codes; `None` = no location.
fn locations() -> [(Guid128, Option<&'static [u8]>); 5] {
    [
        (user(0), Some(&[1, 0])),
        (user(1), Some(&[1, 1])),
        (user(2), Some(&[2])),
        (user(3), Some(&[])),
        (user(4), None),
    ]
}

fn observation() -> Observation {
    let mut o = Observation {
        scope: SCOPE,
        nodes: Vec::new(),
        members: Vec::new(),
    };
    for (g, at) in locations() {
        let mut n = ObservedNode::user(
            &format!("u{}@example.test", g.0[1]),
            &format!("m{}@example.test", g.0[1]),
        );
        n.dn = at.map(|c| Dn128::new(c).unwrap());
        o.nodes.push((g, n));
    }
    o
}

fn store(basis: &dyn ClassView, fields: usize) -> (BindStore, RouteAt) {
    let mut bs = BindStore::new();
    bs.register(
        basis,
        USER,
        ClassSchema {
            schema_rev: LANE_REV,
            field_count: fields,
            extent: Extent::DR64_CUBE,
        },
        Vec::new(),
    )
    .unwrap();
    let version = bs.route(POSITION, USER).unwrap();
    (
        bs,
        RouteAt {
            position: POSITION,
            version,
        },
    )
}

fn bind(
    bs: &BindStore,
    basis: &dyn ClassView,
    lanes: &DirLanes<'_, '_>,
    route: RouteAt,
    fields: &[u8],
) -> Result<LaneBinding, LaneRefusal> {
    bs.bind_lanes(
        basis,
        lanes.frames(),
        lanes,
        LaneRequest {
            route,
            rev: LANE_REV,
            fields: &WideFieldMask::from_positions(fields),
            frame: &lanes.frames().seal(),
            scope: READER,
            selection: lanes.space(),
        },
    )
}

/// The basis is UPN, SMTP, then Path1..Path`n`, and the store resolves it as
/// such. Falsifier: a basis that drops, reorders or misnames the path levels.
#[test]
fn the_user_basis_names_upn_smtp_and_the_path_levels() {
    let basis = UserBasis::new(USER, LEVELS);
    let (bs, _) = store(&basis, 2 + LEVELS);
    let r = bs.resolve(&basis, USER, LANE_REV).unwrap();
    let got: Vec<(&str, &str)> = r
        .fields
        .iter()
        .map(|f| (f.predicate_iri.as_str(), f.label.as_str()))
        .collect();
    assert_eq!(
        got,
        [
            ("dir:userPrincipalName", "UPN"),
            ("dir:primarySmtp", "Primary SMTP"),
            ("dir:ouPath/1", "Path1"),
            ("dir:ouPath/2", "Path2"),
            ("dir:ouPath/3", "Path3"),
        ]
    );
    assert_eq!(
        UserBasis::new(USER, 99).len(),
        2 + 16,
        "clamped to 16 levels"
    );
}

/// Each path lane reads its level of the user's location; a user with a
/// shallower location or none has no value there. Checked against the
/// fixture's own codes, with nothing allocated on the read path. Falsifier: a
/// lane that ignores the level (reads level 0 everywhere), reads the row's
/// ordinal, or invents a value past the location's depth.
#[test]
fn path_lanes_read_each_level_of_the_users_location() {
    let mut st = VersionStore::new();
    let v = st.observe("ad", 0, observation()).unwrap();
    let view = st.view(v).unwrap();
    let lanes = DirLanes::new(&view, v, USER, READER);
    let basis = UserBasis::new(USER, LEVELS);
    let (bs, route) = store(&basis, 2 + LEVELS);
    let path: Vec<u8> = (0..LEVELS as u8).map(|k| F_PATH1 + k).collect();
    let b = bind(&bs, &basis, &lanes, route, &path).unwrap();

    let want = locations();
    let mut checked = 0usize;
    let mut present = 0usize;
    let mut read_allocs = 0u64;
    for (k, &f) in path.iter().enumerate() {
        let lane = b.lane(&bs, &basis, lanes.frames(), &lanes, f).unwrap();
        let a0 = allocs();
        let cells: [Option<u32>; 5] = std::array::from_fn(|r| lane.get(r as u32));
        read_allocs += allocs() - a0;
        for row in 0..lanes.users_len() as u32 {
            let g = view.user_guid(UserOrdinal(row as u16)).unwrap();
            let (_, at) = want.iter().find(|(w, _)| *w == g).unwrap();
            let expect = at.and_then(|c| c.get(k)).map(|&c| u32::from(c));
            assert_eq!(cells[row as usize], expect, "user {} Path{}", g.0[1], k + 1);
            checked += 1;
            present += usize::from(expect.is_some());
        }
    }
    assert_eq!(read_allocs, 0, "reading the cells allocates nothing");
    assert_eq!(checked, 5 * LEVELS);
    // Anti-vacuity: values at level 1 and 2, and absences at every level.
    assert_eq!(present, 3 + 2, "three users at Path1, two at Path2");
}

struct MoveTo {
    node: Guid128,
    from: Option<Dn128>,
    to: Option<Dn128>,
}
impl Rule for MoveTo {
    fn id(&self) -> RuleId {
        RuleId {
            name: "MoveTo",
            version: 1,
        }
    }
    fn propose(&self, _: &View<'_>, _: &[EvidenceRef]) -> Vec<Change> {
        vec![Change::SetLocation {
            node: self.node,
            from: self.from,
            to: self.to,
        }]
    }
}

/// A simulated move is read through the lanes of the simulated version: the
/// coordinates are the version's, not the observation's. Falsifier: a lane
/// over the base snapshot that ignores the version's location overrides.
#[test]
fn path_lanes_follow_a_simulated_move() {
    let mut st = VersionStore::new();
    let observed = st.observe("ad", 0, observation()).unwrap();
    let to = Dn128::new(&[2, 5]).unwrap();
    let from = st.view(observed).unwrap().location(&user(0)).unwrap();
    assert_eq!(from, Some(Dn128::new(&[1, 0]).unwrap()));
    let sim = st
        .simulate(
            observed,
            &MoveTo {
                node: user(0),
                from,
                to: Some(to),
            },
            &[],
        )
        .unwrap();

    let basis = UserBasis::new(USER, LEVELS);
    let (bs, route) = store(&basis, 2 + LEVELS);
    let read = |v| {
        let view = st.view(v).unwrap();
        let lanes = DirLanes::new(&view, v, USER, READER);
        let b = bind(&bs, &basis, &lanes, route, &[F_PATH1, F_PATH1 + 1]).unwrap();
        let row = (0..lanes.users_len() as u32)
            .find(|&r| view.user_guid(UserOrdinal(r as u16)) == Some(user(0)))
            .unwrap();
        let at = |f| {
            b.lane(&bs, &basis, lanes.frames(), &lanes, f)
                .unwrap()
                .get(row)
        };
        (at(F_PATH1), at(F_PATH1 + 1))
    };
    assert_eq!(read(observed), (Some(1), Some(0)));
    assert_eq!(read(sim), (Some(2), Some(5)));
}

struct Fields(Vec<FieldRef>);
impl ClassView for Fields {
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

/// A path predicate outside 1..=16, or spelled another way, names no lane and
/// is refused at bind; the spellings that do name one still bind. Falsifier: a
/// lenient parser that maps `dir:ouPath/01` or `/17` onto a lane.
#[test]
fn only_the_canonical_path_predicates_bind() {
    let mut st = VersionStore::new();
    let v = st.observe("ad", 0, observation()).unwrap();
    let view = st.view(v).unwrap();
    let lanes = DirLanes::new(&view, v, USER, READER);
    for (iri, binds) in [
        ("dir:ouPath/1", true),
        ("dir:ouPath/16", true),
        ("dir:ouPath/0", false),
        ("dir:ouPath/17", false),
        ("dir:ouPath/01", false),
        ("dir:ouPath/+1", false),
        ("dir:ouPath/", false),
    ] {
        let basis = Fields(vec![
            FieldRef::new("dir:userPrincipalName", "UPN"),
            FieldRef::new(iri, "path"),
        ]);
        let (bs, route) = store(&basis, 2);
        let r = bind(&bs, &basis, &lanes, route, &[F_UPN, 1]);
        if binds {
            assert!(r.is_ok(), "{iri} binds: {:?}", r.err());
        } else {
            assert!(
                matches!(r, Err(LaneRefusal::Unbound { field: 1, .. })),
                "{iri} is refused, got {:?}",
                r.err()
            );
        }
    }
}
