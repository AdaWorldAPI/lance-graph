//! The cube against PostgreSQL in both directions, with the bijection checked.
//!
//! Two relations of Zammad's `User.find(k).permissions` (has-many-through),
//! seeded by zammad-rs's rule (`oracle/shadow-joins.pg.json`, "seeding"):
//!
//! - `roles_users(user_id, role_id)`: `(u, r)` when `u + r` is even, users
//!   1..=6, roles 1..=12, plus `(1, 3)` a second time;
//! - `permissions_roles(role_id, permission_id)`: `(r, p)` when `r * p % 3 ==
//!   0`, permissions 1..=20, plus `(2, 3)` and `(2, 6)` a second time.
//!
//! An id's global is `class << 24 | key` (user 1, role 2, permission 3), so a
//! coordinate names its class as well as its key.
//!
//! What is checked, each against PostgreSQL 16 through `pg-parity-harness`:
//!
//! 1. **Outbound** (`identity[index].key : identity[index].value`): each
//!    relation's two lanes are lent through ONE `borrow_all_present`, and each
//!    row is spelled `global(x[i]) : global(y[i])`, a pointer-table read per
//!    cell. It must equal `SELECT * FROM <relation>` as a multiset, duplicates
//!    included. The outbound read must make no counted map lookup.
//! 2. **Inbound:** PostgreSQL's own rows are encoded through `local(global)`,
//!    one counted lookup (a binary search over at most 256 ids) per cell, and
//!    must equal the lanes the cube lends, as a multiset.
//! 3. **Bijection:** `global(local(g)) == g` for every id, and
//!    `local(global(l)) == l` for every coordinate the map holds.
//! 4. **The join as a bag count:** `Cube::join_size` per user (bind on the
//!    user, join on the role) must equal PostgreSQL's `count(*)` of the
//!    same join, with no row built.
//! 5. The filtered form (`roles.active`, `permissions.active`) cannot be put
//!    to the cube today (EXO-327) and is recorded as UNSUPPORTED, not skipped.
//!
//! Disables, each of which must FAIL against PostgreSQL:
//! - **local as key:** a cell is spelled with its local coordinate, not
//!   `global(local)`;
//! - **swapped coordinate:** cells are decoded through a pointer table with two
//!   entries swapped (`global(local(g)) == g` broken for one pair);
//! - **relation read as a set:** the duplicate pairs are dropped before the
//!   cube is built, so the bag join counts existence.
//!
//! ```text
//! PGHOST=<socket dir> PGUSER=<user> PGDATABASE=postgres \
//!   cargo run -p lance-graph-xlog --example cube_pg_bijection
//! ```
//!
//! Without a server it prints `SKIP` and exits 0. A SKIP is not a pass.

use std::collections::BTreeSet;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{LocalMap, MapTable, Scope, Width};
use lance_graph_dt64::{BindStore, ClassSchema, Extent, LaneRequest, RouteAt, SchemaRev};
use lance_graph_mask_risc::LaneRef;
use lance_graph_xlog::cube::{Cube, CubeLanes};
use pg_parity_harness::{
    check, Answer, Case, Cell, Order, Outcome, Pg, Refusal, Report, Row, Rows, Skip,
};

const TILE: u32 = 7;
const READER: Scope = Scope { id: 4 };
const USER: u32 = 1;
const ROLE: u32 = 2;
const PERMISSION: u32 = 3;

/// One relation as a class: its two columns, their classes and predicates.
struct Rel {
    name: &'static str,
    columns: [&'static str; 2],
    classes: [u32; 2],
    class: u32,
    position: u16,
}

const ROLES_USERS: Rel = Rel {
    name: "roles_users",
    columns: ["user_id", "role_id"],
    classes: [USER, ROLE],
    class: 0x7F20_0001,
    position: 2,
};
const PERMISSIONS_ROLES: Rel = Rel {
    name: "permissions_roles",
    columns: ["role_id", "permission_id"],
    classes: [ROLE, PERMISSION],
    class: 0x7F21_0001,
    position: 3,
};

fn global(class: u32, key: u32) -> u32 {
    class << 24 | key
}

fn predicate(rel: &Rel, col: usize) -> String {
    format!("zammad:{}/{}", rel.name, rel.columns[col])
}

/// The two relations' keys, in zammad-rs's seeding order.
fn seeded() -> [Vec<(u32, u32)>; 2] {
    let mut ru: Vec<(u32, u32)> = (1..=6u32)
        .flat_map(|u| {
            (1..=12u32)
                .filter(move |r| (u + r) % 2 == 0)
                .map(move |r| (u, r))
        })
        .collect();
    ru.push((1, 3));
    let mut pr: Vec<(u32, u32)> = (1..=12u32)
        .flat_map(|r| {
            (1..=20u32)
                .filter(move |p| r * p % 3 == 0)
                .map(move |p| (r, p))
        })
        .collect();
    pr.extend([(2, 3), (2, 6)]);
    [ru, pr]
}

fn as_globals(rel: &Rel, keys: &[(u32, u32)]) -> Vec<(u32, u32)> {
    keys.iter()
        .map(|&(a, b)| (global(rel.classes[0], a), global(rel.classes[1], b)))
        .collect()
}

fn dedup(mut v: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    v.sort_unstable();
    v.dedup();
    v
}

struct View(Vec<(ClassId, Vec<FieldRef>)>);

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        self.0
            .iter()
            .find(|(c, _)| *c == class)
            .map_or(&[], |(_, f)| f.as_slice())
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Card
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

fn view_class(class: u32) -> ClassId {
    (class >> 16) as ClassId
}

/// How cells are spelled outbound; the two decode disables.
#[derive(Clone, Copy, PartialEq)]
enum Decode {
    Global,
    LocalAsKey,
    Swapped,
}

/// The relation's two lanes, lent through ONE `borrow_all_present`, handed to
/// `f` as `(x, y)` local coordinates, in place.
#[allow(clippy::too_many_arguments)]
fn with_lanes<R>(
    store: &BindStore,
    view: &View,
    cube: &Cube,
    t: &MapTable,
    rel_no: u32,
    rel: &Rel,
    route: RouteAt,
    f: impl FnOnce(&[u32], &[u32]) -> R,
) -> R {
    let p0 = predicate(rel, 0);
    let p1 = predicate(rel, 1);
    let source = CubeLanes::new(
        cube,
        t,
        rel_no,
        rel.class,
        [p0.as_str(), p1.as_str()],
        READER,
    );
    let frame = t.seal();
    let binding = store
        .bind_lanes(
            view,
            t.frames(),
            &source,
            LaneRequest {
                route,
                rev: SchemaRev(1),
                fields: &WideFieldMask::from_positions(&[0, 1]),
                frame: &frame,
                scope: READER,
                selection: source.space(),
            },
        )
        .expect("bind");
    let mut cols: [&[u32]; 2] = [&[], &[]];
    for item in binding
        .borrow_all_present(store, view, t.frames(), &source)
        .expect("check")
    {
        let (field, lane, _live) = item.expect("lane");
        let LaneRef::U32(values) = lane else {
            panic!("a relation lane is u32")
        };
        cols[usize::from(field)] = values;
    }
    f(cols[0], cols[1])
}

/// Outbound: each row of the lent lanes spelled `global(x[i]) : global(y[i])`,
/// checked against the column's class.
#[allow(clippy::too_many_arguments)]
fn outbound(
    store: &BindStore,
    view: &View,
    cube: &Cube,
    t: &MapTable,
    rel_no: u32,
    rel: &Rel,
    route: RouteAt,
    decode: Decode,
) -> Answer {
    let frame = t.seal();
    let map = t.map(&frame, TILE).expect("map");
    // The swapped-coordinate disable: the pointer table with two entries swapped.
    let swapped = |local: u8| match local {
        0 => map.global(1),
        1 => map.global(0),
        l => map.global(l),
    };
    let rows = with_lanes(store, view, cube, t, rel_no, rel, route, |x, y| {
        (0..x.len())
            .map(|i| {
                [x[i], y[i]]
                    .iter()
                    .enumerate()
                    .map(|(c, &local)| {
                        let local8 = u8::try_from(local).expect("a tile coordinate");
                        let g = match decode {
                            Decode::LocalAsKey => return Some(local.to_string()),
                            Decode::Global => map.global(local8),
                            Decode::Swapped => swapped(local8),
                        }
                        .expect("a coordinate the map holds");
                        let (class, key) = ((g >> 24) as u32, (g & 0xFF_FFFF) as u32);
                        Some(if class == rel.classes[c] {
                            key.to_string()
                        } else {
                            // A coordinate of another class in this column.
                            format!("class {class}:{key}")
                        })
                    })
                    .collect()
            })
            .collect()
    });
    Answer::Rows(Rows {
        types: vec!["integer".into(), "integer".into()],
        rows,
    })
}

fn cube_of(t: &MapTable, rels: &[Vec<(u32, u32)>; 2]) -> Cube {
    Cube::new(t, &t.seal(), TILE, &[&rels[0], &rels[1]]).expect("cube")
}

/// Per user, `Cube::join_size` of `roles_users[user = k] ⋈ permissions_roles`
/// on the role: a bag count, no row built.
fn join_counts(cube: &mut Cube, t: &MapTable) -> Answer {
    let rows = (1..=6u32)
        .map(|k| {
            let ru = cube.scan(0).expect("scan");
            let bound = cube.bind(t, ru, 0, global(USER, k)).expect("bind");
            let pr = cube.scan(1).expect("scan");
            let join = cube.join(bound, 1, pr, 0).expect("join");
            let n = cube.join_size(t, join).expect("join size");
            vec![Some(k.to_string()), Some(n.to_string())]
        })
        .collect();
    Answer::Rows(Rows {
        types: vec!["integer".into(), "bigint".into()],
        rows,
    })
}

fn cells_to_pairs(rows: &[Row]) -> Vec<(u32, u32)> {
    rows.iter()
        .map(|r| {
            let n = |c: &Cell| c.as_deref().expect("not NULL").parse::<u32>().expect("int");
            (n(&r[0]), n(&r[1]))
        })
        .collect()
}

fn main() {
    let pg = match Pg::connect() {
        Ok(pg) => pg,
        Err(Skip(why)) => {
            println!("SKIP: {why}");
            return;
        }
    };
    let keys = seeded();
    let rels = [&ROLES_USERS, &PERMISSIONS_ROLES];
    let globals: [Vec<(u32, u32)>; 2] =
        [as_globals(rels[0], &keys[0]), as_globals(rels[1], &keys[1])];

    // DR's map: every id of the three classes, one tile.
    let ids: BTreeSet<u64> = globals
        .iter()
        .flatten()
        .flat_map(|&(a, b)| [u64::from(a), u64::from(b)])
        .collect();
    let ids: Vec<u64> = ids.into_iter().collect();
    let mut t = MapTable::new(1, 1);
    t.advance(
        1,
        1,
        vec![(
            TILE,
            Some(LocalMap::prepare(Width::W256, &ids).expect("map")),
        )],
    );
    let mut cube = cube_of(&t, &globals);

    let view = View(
        rels.iter()
            .map(|r| {
                (
                    view_class(r.class),
                    (0..2)
                        .map(|c| FieldRef::new(predicate(r, c), r.columns[c]))
                        .collect(),
                )
            })
            .collect(),
    );
    let mut store = BindStore::new();
    for r in rels {
        store
            .register(
                &view,
                r.class,
                ClassSchema {
                    schema_rev: SchemaRev(1),
                    field_count: 2,
                    extent: Extent::DR64_CUBE,
                },
                Vec::new(),
            )
            .expect("register");
    }
    let routes: Vec<RouteAt> = rels
        .iter()
        .map(|r| RouteAt {
            position: r.position,
            version: store.route(r.position, r.class).expect("route"),
        })
        .collect();

    // PostgreSQL: the same seeding, plus the two classes the filter would need.
    let mut script = String::new();
    for (r, k) in rels.iter().zip(&keys) {
        script.push_str(&format!(
            "CREATE TABLE {} ({} int, {} int);\nINSERT INTO {} VALUES {};\n",
            r.name,
            r.columns[0],
            r.columns[1],
            r.name,
            k.iter()
                .map(|(a, b)| format!("({a}, {b})"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    script.push_str(
        "CREATE TABLE roles AS SELECT r AS id, r % 4 <> 0 AS active FROM generate_series(1, 12) r;\n\
         CREATE TABLE permissions AS SELECT p AS id, p % 5 <> 0 AS active FROM generate_series(1, 20) p;\n",
    );
    pg.script(&script)
        .unwrap_or_else(|e| panic!("PostgreSQL rejected the fixture: {e}"));

    let map = t.map(&t.seal(), TILE).expect("map");
    let mut report = Report::new();

    // 1. Outbound, per relation; no counted lookup on the way out.
    let before = map.lookups();
    for (no, r) in rels.iter().enumerate() {
        let sql = format!("SELECT {}, {} FROM {}", r.columns[0], r.columns[1], r.name);
        let case = Case {
            name: r.name,
            sql: &sql,
            order: Order::Unordered,
        };
        check(&pg, &mut report, &case, || {
            outbound(
                &store,
                &view,
                &cube,
                &t,
                no as u32,
                r,
                routes[no],
                Decode::Global,
            )
        });
    }
    let outbound_lookups = map.lookups() - before;

    // 2. Inbound: PostgreSQL's rows encoded through local(global), compared
    // with the lanes the cube lends. Only the encode is counted.
    let mut inbound_equal = true;
    let mut inbound_lookups = 0;
    let mut encoded_cells = 0;
    for (no, r) in rels.iter().enumerate() {
        let theirs = pg
            .query(&format!(
                "SELECT {}, {} FROM {}",
                r.columns[0], r.columns[1], r.name
            ))
            .expect("rows");
        let pairs = cells_to_pairs(&theirs.rows);
        let before = map.lookups();
        let mut encoded: Vec<(u32, u32)> = pairs
            .iter()
            .map(|&(a, b)| {
                let l = |class, key| {
                    u32::from(
                        map.local(u64::from(global(class, key)))
                            .expect("an id of the map"),
                    )
                };
                (l(r.classes[0], a), l(r.classes[1], b))
            })
            .collect();
        inbound_lookups += map.lookups() - before;
        encoded_cells += 2 * pairs.len();
        let mut lent: Vec<(u32, u32)> = with_lanes(
            &store,
            &view,
            &cube,
            &t,
            no as u32,
            r,
            routes[no],
            |x, y| x.iter().copied().zip(y.iter().copied()).collect(),
        );
        encoded.sort_unstable();
        lent.sort_unstable();
        inbound_equal &= encoded == lent;
    }

    // 3. Bijection over every id and every coordinate.
    let ids_round_trip = ids
        .iter()
        .all(|&g| map.local(g).ok().and_then(|l| map.global(l)) == Some(g));
    let coords_round_trip = (0..map.len())
        .all(|l| map.global(l as u8).and_then(|g| map.local(g).ok()) == Some(l as u8));

    // 4. The join as a bag count, per user.
    let join_sql = "SELECT u.id, count(pr.role_id) FROM generate_series(1, 6) u(id) \
                    LEFT JOIN roles_users ru ON ru.user_id = u.id \
                    LEFT JOIN permissions_roles pr ON pr.role_id = ru.role_id GROUP BY u.id";
    let join_case = Case {
        name: "join size per user",
        sql: join_sql,
        order: Order::Unordered,
    };
    let gate_join = check(&pg, &mut report, &join_case, || join_counts(&mut cube, &t));

    // 5. The filtered join: not expressible on a cube side today.
    report.record(
        "join size per user, active roles and permissions",
        Outcome::Unsupported(Refusal {
            feature: "join-side-property-filter",
            detail: "a cube side cannot filter on a property of the joined row (EXO-327)".into(),
        }),
    );

    // Disables.
    let ru_sql = "SELECT user_id, role_id FROM roles_users";
    let local_as_key = check(
        &pg,
        &mut report,
        &Case {
            name: "DISABLE local as key",
            sql: ru_sql,
            order: Order::Unordered,
        },
        || {
            outbound(
                &store,
                &view,
                &cube,
                &t,
                0,
                &ROLES_USERS,
                routes[0],
                Decode::LocalAsKey,
            )
        },
    );
    let swapped = check(
        &pg,
        &mut report,
        &Case {
            name: "DISABLE swapped coordinate",
            sql: ru_sql,
            order: Order::Unordered,
        },
        || {
            outbound(
                &store,
                &view,
                &cube,
                &t,
                0,
                &ROLES_USERS,
                routes[0],
                Decode::Swapped,
            )
        },
    );
    let as_sets = [dedup(globals[0].clone()), dedup(globals[1].clone())];
    let mut set_cube = cube_of(&t, &as_sets);
    let as_set = check(
        &pg,
        &mut report,
        &Case {
            name: "DISABLE relation read as a set",
            sql: join_sql,
            order: Order::Unordered,
        },
        || join_counts(&mut set_cube, &t),
    );

    println!("PostgreSQL {}", pg.version());
    println!(
        "ids {} (6 users, 12 roles, 20 permissions); roles_users {} rows, permissions_roles {} rows",
        ids.len(),
        keys[0].len(),
        keys[1].len()
    );
    println!(
        "outbound map lookups {outbound_lookups}; inbound map lookups {inbound_lookups} for {encoded_cells} encoded cells"
    );
    println!(
        "inbound equals the lent lanes: {inbound_equal}; bijection ids {ids_round_trip}, coordinates {coords_round_trip}"
    );
    println!("{}", report.render());

    let (pass, fail, unsupported, _) = report.counts();
    assert_eq!(
        report.outcomes()[..2]
            .iter()
            .filter(|(_, o)| *o == Outcome::Pass)
            .count(),
        2,
        "outbound equals PostgreSQL for both relations"
    );
    assert_eq!(gate_join, Outcome::Pass, "the bag join equals PostgreSQL");
    assert_eq!(outbound_lookups, 0, "outbound reads only the pointer table");
    assert!(
        inbound_equal,
        "PostgreSQL's rows encode to the cube's lanes"
    );
    assert_eq!(
        inbound_lookups, encoded_cells as u64,
        "inbound: one lookup per cell"
    );
    assert!(
        ids_round_trip && coords_round_trip,
        "the map is a bijection"
    );
    assert!(
        matches!(local_as_key, Outcome::Fail(_)),
        "local as key not detected"
    );
    assert!(
        matches!(swapped, Outcome::Fail(_)),
        "swapped coordinate not detected"
    );
    assert!(
        matches!(as_set, Outcome::Fail(_)),
        "relation as a set not detected"
    );
    assert_eq!((pass, fail, unsupported), (3, 3, 1));
}
