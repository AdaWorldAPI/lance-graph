//! The PostgreSQL row polyfill gate: two tables read through ONE DT64 check
//! each, answered as PostgreSQL answers them.
//!
//! Every answer is compared twice:
//! - with a model oracle built from the logical rows (never from the lanes),
//!   which always runs;
//! - with PostgreSQL through `pg-parity-harness`, when a server answers. A
//!   SKIP prints `SKIP`; with `PG_ROW_REQUIRE_PG=1` it fails the test. A SKIP
//!   is not a pass of the PostgreSQL half.
//!
//! The fixture holds what the register contract has to get right
//! (`.claude/knowledge/register-contract-reconciliation.md`):
//! - NULLs in integer, text, date and timestamp columns (absent = −0);
//! - a `NOT NULL DEFAULT ''` text column whose no-value is absent and spelled
//!   `''` (C3, D-REGC-10);
//! - negative integers, read through signed lanes;
//! - deleted rows that keep their old values (C3, C4);
//! - foreign keys to a deleted parent and to no parent (left join);
//! - rows stored in a shuffled order, so a position is not an id.
//!
//! The disable features each turn the gate red:
//! - `disable-pg-row-presence`: an absent cell is read as its value;
//! - `disable-pg-row-live`: deleted rows are read and kept by WHERE;
//! - `disable-pg-row-bag-present`: a left join's missing side is read at its
//!   placeholder position.

use std::collections::BTreeMap;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::pg_row::{AbsentAs, CodeText, LaneEncoding, PgColumn, PgType};
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRequest, LaneSource,
    PresenceSource, RouteAt, SchemaRev, SelAddr, SelectionSource,
};
use lance_graph_mask_risc::LaneRef;
use lance_graph_pg_row::{read_bag, BagSide, ColumnSpec, LiveSource, PgRowRefusal, PgRows, Reader};
use lance_graph_quack::{Cmp, Col, Filter};
use pg_parity_harness::{multiset_diff, Pg, Row, Skip};

const READER: Scope = Scope { id: 7 };

/// One logical column: name, PostgreSQL type, encoding, absent spelling.
#[derive(Clone, Copy)]
struct Def {
    name: &'static str,
    sql: &'static str,
    col: PgColumn,
}

const fn def(name: &'static str, sql: &'static str, ty: PgType, encoding: LaneEncoding) -> Def {
    Def {
        name,
        sql,
        col: PgColumn {
            ty,
            encoding,
            absent: AbsentAs::Null,
        },
    }
}

const TICKET: [Def; 7] = [
    def("id", "integer", PgType::Int4, LaneEncoding::Signed),
    def("title", "text", PgType::Text, LaneEncoding::TextCode),
    Def {
        name: "note",
        sql: "text NOT NULL DEFAULT ''",
        col: PgColumn {
            ty: PgType::Text,
            encoding: LaneEncoding::TextCode,
            absent: AbsentAs::EmptyText,
        },
    },
    def("open", "boolean", PgType::Bool, LaneEncoding::Bool),
    def("owner_id", "integer", PgType::Int4, LaneEncoding::Signed),
    def("priority", "integer", PgType::Int4, LaneEncoding::Signed),
    def(
        "created",
        "timestamp(3) without time zone",
        PgType::Timestamp(Some(3)),
        LaneEncoding::EpochSeconds,
    ),
];
const DUE: Def = def("due", "date", PgType::Date, LaneEncoding::EpochDays);

const AGENT: [Def; 2] = [
    def("id", "integer", PgType::Int4, LaneEncoding::Signed),
    def("name", "text", PgType::Text, LaneEncoding::TextCode),
];

/// A logical cell: `None` is NULL (or, for `note`, no value: `''`).
#[derive(Clone, Debug, PartialEq)]
enum V {
    Int(i32),
    Text(String),
    Bool(bool),
    Secs(u32),
    Days(u32),
}

struct Logical {
    defs: Vec<Def>,
    rows: Vec<Vec<Option<V>>>,
    deleted: Vec<bool>,
}

fn xorshift(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

fn tickets(agents: usize) -> Logical {
    let mut x = 0x7E57_u64;
    let titles = ["printer", "vpn", "", "login", "mail"];
    let notes = ["", "call back", "waiting"];
    let mut rows = Vec::new();
    let mut deleted = Vec::new();
    for id in 1..=300i32 {
        let r = xorshift(&mut x);
        let title = (r % 7 != 0).then(|| V::Text(titles[(r >> 3) as usize % titles.len()].into()));
        // `note` is NOT NULL DEFAULT '': no value is the empty string.
        let note = (r % 3 != 0).then(|| V::Text(notes[(r >> 5) as usize % notes.len()].into()));
        let open = Some(V::Bool(r >> 7 & 1 == 1));
        // Owners 1..=agents+3: some past the agents table (no parent).
        let owner = (r % 5 != 0).then(|| V::Int(1 + ((r >> 9) % (agents as u64 + 3)) as i32));
        let priority = (r % 11 != 0).then(|| V::Int(((r >> 11) % 9) as i32 - 4));
        let created = Some(V::Secs(1_600_000_000 + (r >> 13) as u32 % 100_000_000));
        let due = (r % 4 != 0).then(|| V::Days(18_000 + (r >> 17) as u32 % 4_000));
        rows.push(vec![
            Some(V::Int(id)),
            title,
            note,
            open,
            owner,
            priority,
            created,
            due,
        ]);
        deleted.push(r % 13 == 0);
    }
    let mut defs = TICKET.to_vec();
    defs.push(DUE);
    Logical {
        defs,
        rows,
        deleted,
    }
}

fn agents() -> Logical {
    let names = ["ana", "bo", "cy", "dee", "eli", "fay", "gus", "hal"];
    Logical {
        defs: AGENT.to_vec(),
        rows: (0..names.len())
            .map(|i| vec![Some(V::Int(i as i32 + 1)), Some(V::Text(names[i].into()))])
            .collect(),
        // Agent 3 is deleted: tickets owned by it get NULL on a left join.
        deleted: (0..names.len()).map(|i| i == 2).collect(),
    }
}

/// The source owner: lanes, presence and live planes, code tables, issued
/// selections. Rows are stored in a shuffled order.
struct Source {
    class: u32,
    space: CoordSpace,
    lanes: Vec<Lane>,
    presence: Vec<Vec<u64>>,
    live: Vec<u64>,
    codes: Vec<Vec<String>>,
    /// Logical row → stored position.
    position: Vec<u32>,
    selections: Vec<Vec<u32>>,
    frames: FrameStore,
}

enum Lane {
    U32(Vec<u32>),
    I32(Vec<i32>),
}

impl Source {
    fn new(class: u32, population: u64, t: &Logical, seed: u64) -> Source {
        let n = t.rows.len();
        let mut order: Vec<usize> = (0..n).collect();
        let mut x = seed;
        for i in (1..n).rev() {
            order.swap(i, (xorshift(&mut x) % (i as u64 + 1)) as usize);
        }
        let words = n.div_ceil(64);
        let mut lanes: Vec<Lane> = t
            .defs
            .iter()
            .map(|d| match d.col.encoding {
                LaneEncoding::Signed => Lane::I32(vec![0; n]),
                _ => Lane::U32(vec![0; n]),
            })
            .collect();
        let mut presence = vec![vec![0u64; words]; t.defs.len()];
        let mut live = vec![0u64; words];
        // Code 0 is a real string in every code table, so reading an absent
        // cell as its value shows up as that string.
        let mut codes: Vec<Vec<String>> = vec![vec!["<code 0>".to_owned()]; t.defs.len()];
        let mut position = vec![0u32; n];
        for (stored, &logical) in order.iter().enumerate() {
            position[logical] = stored as u32;
            if !t.deleted[logical] {
                live[stored / 64] |= 1 << (stored % 64);
            }
            for (f, cell) in t.rows[logical].iter().enumerate() {
                let Some(v) = cell else { continue };
                let raw = match v {
                    V::Int(i) => *i as u32,
                    V::Bool(b) => u32::from(*b),
                    V::Secs(s) | V::Days(s) => *s,
                    V::Text(s) => match codes[f].iter().position(|c| c == s) {
                        Some(c) => c as u32,
                        None => {
                            codes[f].push(s.clone());
                            (codes[f].len() - 1) as u32
                        }
                    },
                };
                match &mut lanes[f] {
                    Lane::U32(l) => l[stored] = raw,
                    Lane::I32(l) => l[stored] = raw as i32,
                }
                presence[f][stored / 64] |= 1 << (stored % 64);
            }
        }
        Source {
            class,
            space: CoordSpace {
                population,
                coordinates: 1,
            },
            lanes,
            presence,
            live,
            codes,
            position,
            selections: Vec::new(),
            frames: FrameStore::new(FrameRevisions {
                source: population,
                relation_map: 1,
                coordinates: 1,
            }),
        }
    }

    /// The source issues a selection of logical rows (the test stands in for
    /// the issuer, EXO-329).
    fn issue(&mut self, logical: &[usize]) -> u16 {
        self.selections
            .push(logical.iter().map(|&l| self.position[l]).collect());
        (self.selections.len() - 1) as u16
    }
}

impl LaneSource for Source {
    type Lane<'a> = LaneRef<'a>;
    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        (class == self.class && rev == SchemaRev(1) && usize::from(field) < self.lanes.len())
            .then_some(LaneAddr {
                space: self.space,
                lane: u16::from(field),
                scope: READER,
            })
    }
    fn borrow(&self, addr: &LaneAddr) -> Option<LaneRef<'_>> {
        self.lanes.get(usize::from(addr.lane)).map(|l| match l {
            Lane::U32(v) => LaneRef::U32(v),
            Lane::I32(v) => LaneRef::I32(v),
        })
    }
    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}

impl PresenceSource for Source {
    fn borrow_presence(&self, addr: &LaneAddr) -> Option<&[u64]> {
        self.presence.get(usize::from(addr.lane)).map(Vec::as_slice)
    }
}

impl SelectionSource for Source {
    fn selection_of(&self, sel: u16, _frame: &Frame) -> Option<SelAddr> {
        (usize::from(sel) < self.selections.len()).then_some(SelAddr {
            space: self.space,
            sel,
            scope: READER,
        })
    }
    fn borrow_positions(&self, addr: &SelAddr) -> Option<&[u32]> {
        self.selections
            .get(usize::from(addr.sel))
            .map(Vec::as_slice)
    }
}

impl LiveSource for Source {
    fn borrow_live(&self, space: CoordSpace) -> Option<&[u64]> {
        (space == self.space).then_some(self.live.as_slice())
    }
}

struct View {
    class: ClassId,
    fields: Vec<FieldRef>,
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

/// One table: its class, view, binding and source.
struct Bound {
    view: View,
    store: BindStore,
    binding: LaneBinding,
    source: Source,
}

fn bind(
    table: &str,
    class: u32,
    position: u16,
    t: &Logical,
    mut source: Source,
    sels: &[Vec<usize>],
) -> (Bound, Vec<u16>) {
    let issued = sels.iter().map(|s| source.issue(s)).collect();
    let view = View {
        class: (class >> 16) as ClassId,
        fields: t
            .defs
            .iter()
            .map(|d| FieldRef::new(format!("test:{table}/{}", d.name), d.name))
            .collect(),
    };
    let mut store = BindStore::new();
    store
        .register(
            &view,
            class,
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: t.defs.len(),
                extent: Extent::DR64_CUBE,
            },
            Vec::new(),
        )
        .expect("register");
    let version = store.route(position, class).expect("route");
    let frame = source.frames.seal();
    let fields: Vec<u8> = (0..t.defs.len() as u8).collect();
    let binding = store
        .bind_lanes(
            &view,
            &source.frames,
            &source,
            LaneRequest {
                route: RouteAt { position, version },
                rev: SchemaRev(1),
                fields: &WideFieldMask::from_positions(&fields),
                frame: &frame,
                scope: READER,
                selection: source.space,
            },
        )
        .expect("bind");
    (
        Bound {
            view,
            store,
            binding,
            source,
        },
        issued,
    )
}

fn specs<'c>(t: &Logical, codes: &'c [Vec<String>]) -> Vec<ColumnSpec<'c>> {
    t.defs
        .iter()
        .zip(codes)
        .map(|(d, c)| ColumnSpec {
            name: d.name,
            col: d.col,
            codes: (d.col.encoding == LaneEncoding::TextCode).then_some(c as &dyn CodeText),
        })
        .collect()
}

fn open<'s, 'c>(b: &'s Bound, cols: &'c [ColumnSpec<'c>]) -> Reader<'s, 'c> {
    Reader::open(
        &b.binding,
        &b.store,
        &b.view,
        &b.source.frames,
        &b.source,
        cols,
    )
    .expect("open")
}

// ---- the model oracle: PostgreSQL's text output of logical cells -------

fn text(d: &Def, v: &Option<V>) -> Option<String> {
    let mut s = String::new();
    let (present, raw) = match v {
        None => (false, 0),
        Some(V::Int(i)) => (true, *i as u32),
        Some(V::Bool(b)) => (true, u32::from(*b)),
        Some(V::Secs(x) | V::Days(x)) => (true, *x),
        Some(V::Text(t)) => return Some(t.clone()),
    };
    if !present {
        return (d.col.absent == AbsentAs::EmptyText).then(String::new);
    }
    lance_graph_contract::pg_row::write_cell(&d.col, true, raw, None, &mut s).expect("model");
    Some(s)
}

fn model_row(t: &Logical, r: usize) -> Row {
    t.defs
        .iter()
        .zip(&t.rows[r])
        .map(|(d, v)| text(d, v))
        .collect()
}

fn model_null_row(t: &Logical) -> Row {
    t.defs.iter().map(|d| text(d, &None)).collect()
}

fn int(t: &Logical, r: usize, f: usize) -> Option<i32> {
    match &t.rows[r][f] {
        Some(V::Int(i)) => Some(*i),
        _ => None,
    }
}

fn types(t: &Logical) -> Vec<String> {
    t.defs.iter().map(|d| d.col.ty.gdesc()).collect()
}

fn same(name: &str, expected: &[Row], ours: &PgRows) {
    if let Some(diff) = multiset_diff(expected, &ours.rows) {
        panic!("{name}: model oracle differs: {diff}");
    }
}

// ---- PostgreSQL ---------------------------------------------------------

fn sql_value(d: &Def, v: &Option<V>) -> String {
    match v {
        None if d.col.absent == AbsentAs::EmptyText => "''".into(),
        None => "NULL".into(),
        Some(V::Int(i)) => i.to_string(),
        Some(V::Bool(b)) => b.to_string(),
        Some(V::Text(t)) => format!("'{}'", t.replace('\'', "''")),
        Some(cell @ (V::Secs(_) | V::Days(_))) => {
            format!("'{}'", text(d, &Some(cell.clone())).unwrap())
        }
    }
}

fn create(name: &str, t: &Logical) -> String {
    let cols: Vec<String> = t
        .defs
        .iter()
        .map(|d| format!("{} {}", d.name, d.sql))
        .collect();
    let mut s = format!("CREATE TABLE {name} ({});\n", cols.join(", "));
    let live: Vec<String> = t
        .rows
        .iter()
        .zip(&t.deleted)
        .filter(|(_, &del)| !del)
        .map(|(r, _)| {
            let vals: Vec<String> = t.defs.iter().zip(r).map(|(d, v)| sql_value(d, v)).collect();
            format!("({})", vals.join(", "))
        })
        .collect();
    s.push_str(&format!(
        "INSERT INTO {name} VALUES {};\n",
        live.join(",\n")
    ));
    s
}

fn postgres(cases: &[(&str, String, &PgRows)], t: &Logical, a: &Logical) {
    let pg = match Pg::connect() {
        Ok(pg) => pg,
        Err(Skip(why)) => {
            assert!(
                std::env::var_os("PG_ROW_REQUIRE_PG").is_none(),
                "PG_ROW_REQUIRE_PG is set and PostgreSQL is unavailable: {why}"
            );
            println!("SKIP (PostgreSQL half): {why}");
            return;
        }
    };
    println!("PostgreSQL {}", pg.version());
    pg.script(&(create("ticket", t) + &create("agent", a)))
        .expect("create");
    for (name, sql, ours) in cases {
        let theirs = pg.query(sql).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(theirs.types, ours.types, "{name}: types");
        if let Some(diff) = multiset_diff(&theirs.rows, &ours.rows) {
            panic!("{name}: PostgreSQL differs: {diff}");
        }
        println!("PASS {name}: {} rows", ours.rows.len());
    }
}

// ---- the gate -------------------------------------------------------------

#[test]
fn polyfill_equals_postgresql() {
    let a = agents();
    let t = tickets(a.rows.len());
    let live: Vec<usize> = (0..t.rows.len()).filter(|&r| !t.deleted[r]).collect();
    // A bag of live tickets: every fifth, and every eleventh listed again.
    let mut bag: Vec<usize> = live.iter().copied().step_by(5).collect();
    bag.extend(live.iter().copied().step_by(11));

    let (tb, sels) = bind(
        "ticket",
        0x2B01_0001,
        3,
        &t,
        Source::new(0x2B01_0001, 11, &t, 0x51),
        &[bag.clone()],
    );
    let (ab, _) = bind(
        "agent",
        0x2B02_0001,
        4,
        &a,
        Source::new(0x2B02_0001, 12, &a, 0x52),
        &[],
    );
    let tcols = specs(&t, &tb.source.codes);
    let acols = specs(&a, &ab.source.codes);
    let tr = open(&tb, &tcols);
    let ar = open(&ab, &acols);

    // 1. SELECT * FROM ticket.
    let all = tr.rows_in(&tb.source.live).expect("all rows");
    let model_all: Vec<Row> = live.iter().map(|&r| model_row(&t, r)).collect();
    assert_eq!(all.types, types(&t));
    same("select-all", &model_all, &all);

    // 2. WHERE, three-valued: owner_id = 2 OR NOT (priority < 0).
    let filter = Filter::or([
        Filter::cmp(Col(4), Cmp::EqI32(2)),
        Filter::negate(Filter::cmp(Col(5), Cmp::LtI32(0))),
    ]);
    let kept = tr.keep_where(&filter).expect("where");
    let where_rows = tr.rows_in(&kept).expect("where rows");
    let k3 = |r: usize| {
        let owner = int(&t, r, 4).map(|o| o == 2);
        let not_neg = int(&t, r, 5).map(|p| p >= 0);
        owner == Some(true) || not_neg == Some(true)
    };
    let model_where: Vec<Row> = live
        .iter()
        .copied()
        .filter(|&r| k3(r))
        .map(|r| model_row(&t, r))
        .collect();
    same("where-k3", &model_where, &where_rows);
    assert_eq!(tr.count_where(&filter).expect("count"), model_where.len());
    let unknown = live
        .iter()
        .filter(|&&r| int(&t, r, 4).is_none() && int(&t, r, 5).is_none())
        .count();
    assert!(
        unknown > 0,
        "the fixture has rows where the filter is UNKNOWN"
    );

    // 3. A selection issued by the source: a bag with repeats.
    let (sr, positions) = Reader::open_selected(
        &tb.binding,
        &tb.store,
        &tb.view,
        &tb.source.frames,
        &tb.source,
        sels[0],
        &tcols,
    )
    .expect("open selected");
    let picked = sr.rows_at(positions).expect("bag rows");
    let model_bag: Vec<Row> = bag.iter().map(|&r| model_row(&t, r)).collect();
    assert!(bag.len() > bag.iter().collect::<std::collections::BTreeSet<_>>().len());
    same("selection-bag", &model_bag, &picked);

    // 4. LEFT JOIN agent ON agent.id = ticket.owner_id: a virtual tuple bag.
    // The agent side's positions are resolved once per distinct owner key;
    // a key with no live agent has no position and its present bit is 0.
    let agent_pos: BTreeMap<i32, u32> = (0..a.rows.len())
        .filter(|&r| !a.deleted[r])
        .map(|r| (int(&a, r, 0).unwrap(), ab.source.position[r]))
        .collect();
    let left: Vec<u32> = live.iter().map(|&r| tb.source.position[r]).collect();
    let mut right = vec![0u32; live.len()];
    let mut present = vec![0u64; live.len().div_ceil(64)];
    for (i, &r) in live.iter().enumerate() {
        if let Some(&p) = int(&t, r, 4).and_then(|o| agent_pos.get(&o)) {
            right[i] = p;
            present[i / 64] |= 1 << (i % 64);
        }
    }
    let missing = live.len()
        - present
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum::<usize>();
    assert!(missing > 0, "some tickets have no live owner");
    let joined = read_bag(&[
        BagSide {
            reader: &tr,
            positions: &left,
            present: None,
        },
        BagSide {
            reader: &ar,
            positions: &right,
            present: Some(&present),
        },
    ])
    .expect("left join");
    let model_join: Vec<Row> = live
        .iter()
        .map(|&r| {
            let mut row = model_row(&t, r);
            let owner = int(&t, r, 4);
            match (0..a.rows.len())
                .find(|&ar| !a.deleted[ar] && int(&a, ar, 0) == owner && owner.is_some())
            {
                Some(ar) => row.extend(model_row(&a, ar)),
                None => row.extend(model_null_row(&a)),
            }
            row
        })
        .collect();
    same("left-join", &model_join, &joined);

    // 5. A deleted row is refused by position, never read.
    let dead = (0..t.rows.len())
        .find(|&r| t.deleted[r])
        .expect("a deleted ticket");
    if !cfg!(feature = "disable-pg-row-live") {
        assert_eq!(
            tr.rows_at(&[tb.source.position[dead]]),
            Err(PgRowRefusal::DeadPosition {
                at: tb.source.position[dead]
            })
        );
    }

    // The fixture exercises what it claims.
    let empty_notes = all
        .rows
        .iter()
        .filter(|r| r[2].as_deref() == Some(""))
        .count();
    let null_titles = all.rows.iter().filter(|r| r[1].is_none()).count();
    let empty_titles = all
        .rows
        .iter()
        .filter(|r| r[1].as_deref() == Some(""))
        .count();
    let negatives = all
        .rows
        .iter()
        .filter(|r| r[5].as_deref().is_some_and(|p| p.starts_with('-')))
        .count();
    assert!(empty_notes > 0 && null_titles > 0 && empty_titles > 0 && negatives > 0);
    println!(
        "rows {} live {}, where {}, bag {}, left join {} ({} missing side), '' notes {}, NULL titles {}, '' titles {}",
        t.rows.len(),
        live.len(),
        model_where.len(),
        bag.len(),
        live.len(),
        missing,
        empty_notes,
        null_titles,
        empty_titles
    );

    let bag_ids: Vec<String> = bag
        .iter()
        .map(|&r| int(&t, r, 0).unwrap().to_string())
        .collect();
    postgres(
        &[
            ("select-all", "SELECT * FROM ticket".into(), &all),
            (
                "where-k3",
                "SELECT * FROM ticket WHERE owner_id = 2 OR NOT (priority < 0)".into(),
                &where_rows,
            ),
            (
                "selection-bag",
                format!(
                    "SELECT t.* FROM unnest(ARRAY[{}]) AS u(id) JOIN ticket t USING (id)",
                    bag_ids.join(",")
                ),
                &picked,
            ),
            (
                "left-join",
                "SELECT t.*, a.* FROM ticket t LEFT JOIN agent a ON a.id = t.owner_id".into(),
                &joined,
            ),
        ],
        &t,
        &a,
    );
}

/// FAILS IF: a read refuses nothing when it should.
#[test]
fn refusals() {
    let a = agents();
    let (ab, _) = bind(
        "agent",
        0x2B02_0001,
        4,
        &a,
        Source::new(0x2B02_0001, 12, &a, 0x52),
        &[],
    );
    let acols = specs(&a, &ab.source.codes);

    // Wrong arity.
    let short = &acols[..1];
    assert!(matches!(
        Reader::open(
            &ab.binding,
            &ab.store,
            &ab.view,
            &ab.source.frames,
            &ab.source,
            short
        ),
        Err(PgRowRefusal::Arity {
            columns: 1,
            lanes: 2
        })
    ));

    // A frame store that is not the source's own.
    let other = FrameStore::new(FrameRevisions {
        source: 99,
        relation_map: 1,
        coordinates: 1,
    });
    assert!(matches!(
        Reader::open(&ab.binding, &ab.store, &ab.view, &other, &ab.source, &acols),
        Err(PgRowRefusal::Lane(_))
    ));

    // A filter naming a plane, or a column past the table.
    let r = open(&ab, &acols);
    assert!(matches!(
        r.keep_where(&Filter::plane(lance_graph_quack::Mask(0))),
        Err(PgRowRefusal::FilterNotAdmitted(_))
    ));
    assert!(matches!(
        r.keep_where(&Filter::cmp(Col(9), Cmp::EqI32(1))),
        Err(PgRowRefusal::FilterNotAdmitted(_))
    ));

    // A position past the lanes.
    assert_eq!(r.rows_at(&[999]), Err(PgRowRefusal::Position { at: 999 }));

    // Bag sides of different length.
    assert!(matches!(
        read_bag(&[
            BagSide {
                reader: &r,
                positions: &[0, 1],
                present: None
            },
            BagSide {
                reader: &r,
                positions: &[0],
                present: None
            },
        ]),
        Err(PgRowRefusal::Bag(_))
    ));
}
