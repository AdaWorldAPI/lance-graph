//! D-QSP-VROW gate: one zammad-rs table, read through ONE
//! `LaneBinding::borrow_all_selected_present`, equals PostgreSQL.
//!
//! The table is Zammad's `ticket_states`, 146 rows as PostgreSQL 16 answered
//! them in zammad-rs's capture (`oracle/shadow-rows.pg.json`, size 1025). Five
//! columns are read:
//!
//! | column | kind | NULLs in the capture |
//! |---|---|---|
//! | `name` | text, all distinct | 0 |
//! | `next_state_id` | integer | all 146 |
//! | `default_create` | boolean | 0 |
//! | `note` | text, 7 distinct | 35 |
//! | `active` | boolean | 0 |
//!
//! The source holds each column as a `u32` lane with a presence plane (absent
//! = bit 0 and value 0, EXO-294), text as codes into the source's own code
//! table. Rows are stored in a shuffled order, so a row position is not an id.
//! The source issues one selection: a bag of ids, some listed twice. The read
//! is one check, then every bound lane, its presence plane and the witnessed
//! positions; cells are spelled out only at the end, for the comparison.
//!
//! PostgreSQL answers the same bag with `unnest(...) JOIN ticket_states`
//! (duplicates kept), compared as a multiset with types, NULLs and values.
//!
//! The four disables the D-QSP-VROW board row names, plus one more. Each must
//! go red: a FAIL against PostgreSQL, or a refusal before any lane is read.
//! - **local index** (FAIL): row `i` of the answer reads lane row `i`, not the
//!   witnessed position `positions[i]`;
//! - **presence ignored** (FAIL): an absent cell is read as its value 0, so
//!   NULL becomes `0` / `f` / the first code;
//! - **foreign frame** (refused): the read is checked against another owner's
//!   frame store;
//! - **swapped coordinate** (refused): the source issues the selection in
//!   another row order than the binding's;
//! - **selection read as a set** (FAIL, extra): the positions are sorted and
//!   deduplicated, so the repeated ids lose their multiplicity.
//!
//! Not measured here: allocations per chunk, bytes copied and fold time per
//! chunk at 256-4096 positions, and the wire encoding itself (cells are
//! spelled out as PostgreSQL text for the comparison, not sent on a wire).
//!
//! The fixture is not committed. Make it from a zammad-rs checkout:
//!
//! ```text
//! python3 - <<'EOF' > ticket_states.tsv
//! import json
//! d = json.load(open('oracle/shadow-rows.pg.json'))
//! rows, cols = {}, None
//! for c in d['sizes']['1025']['calls']:
//!     a = c.get('answer') or {}
//!     if c.get('table') != 'ticket_states' or c.get('select') != '*' \
//!             or 'rows' not in a or c.get('joins'):
//!         continue
//!     cols = a['columns']
//!     for r in a['rows']:
//!         rows[r[0]] = r
//! ix = {c: i for i, c in enumerate(cols)}
//! want = ['id', 'name', 'next_state_id', 'default_create', 'note', 'active']
//! cell = lambda v: r'\N' if v is None else ('t' if v else 'f') \
//!     if isinstance(v, bool) else str(v)
//! for k in sorted(rows):
//!     print('\t'.join(cell(rows[k][ix[c]]) for c in want))
//! EOF
//! VROW_TICKET_STATES=ticket_states.tsv PGHOST=<socket dir> PGUSER=<user> \
//!   PGDATABASE=postgres \
//!   cargo run -p lance-graph-r2il --example vrow_ticket_states_pg_parity
//! ```
//!
//! Without the fixture or without a server it prints `SKIP` and exits 0. A
//! SKIP is not a pass of the gate.

use std::cell::Cell as Counter;
use std::collections::BTreeMap;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Frame, FrameRefusal, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneRefusal, LaneRequest, LaneSource,
    PresenceSource, RouteAt, SchemaRev, SelAddr, SelectionSource,
};
use pg_parity_harness::{check, Answer, Case, Cell, Order, Outcome, Pg, Report, Rows, Skip};

const CLASS: u32 = 0x2A11_0001;
const VIEW_CLASS: ClassId = 0x2A11;
const POSITION: u16 = 3;
const READER: Scope = Scope { id: 5 };
const SPACE: CoordSpace = CoordSpace {
    population: 1,
    coordinates: 1,
};

/// The read columns, in field order.
const COLUMNS: [(&str, Kind); 5] = [
    ("name", Kind::Text),
    ("next_state_id", Kind::Int),
    ("default_create", Kind::Bool),
    ("note", Kind::Text),
    ("active", Kind::Bool),
];

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Text,
    Int,
    Bool,
}

struct View(Vec<FieldRef>);

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

/// The source owner: lanes, presence planes, code tables, one selection.
struct Source {
    lanes: Vec<Vec<u32>>,
    presence: Vec<Vec<u64>>,
    /// Per text column, code -> text. Read only when cells are spelled out.
    codes: Vec<Vec<String>>,
    selection: Vec<u32>,
    /// The row order the selection is issued in (the swapped-coordinate arm
    /// changes it).
    selection_space: Counter<CoordSpace>,
    frames: FrameStore,
    lookups: Counter<usize>,
}

impl LaneSource for Source {
    type Lane<'a> = &'a [u32];
    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        _frame: &Frame,
    ) -> Option<LaneAddr> {
        self.lookups.set(self.lookups.get() + 1);
        (class == CLASS && rev == SchemaRev(1) && usize::from(field) < self.lanes.len()).then_some(
            LaneAddr {
                space: SPACE,
                lane: u16::from(field),
                scope: READER,
            },
        )
    }
    fn borrow(&self, addr: &LaneAddr) -> Option<&[u32]> {
        self.lanes.get(usize::from(addr.lane)).map(Vec::as_slice)
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
        (sel == 0).then_some(SelAddr {
            space: self.selection_space.get(),
            sel,
            scope: READER,
        })
    }
    fn borrow_positions(&self, addr: &SelAddr) -> Option<&[u32]> {
        (addr.sel == 0).then_some(self.selection.as_slice())
    }
}

fn bit(p: &[u64], i: usize) -> bool {
    p[i / 64] >> (i % 64) & 1 == 1
}

/// One fixture row: id, then the five columns as PostgreSQL text (`None` =
/// NULL).
type Fixture = Vec<(u32, Vec<Cell>)>;

fn load(path: &str) -> Fixture {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    text.lines()
        .map(|line| {
            let mut f = line.split('\t');
            let id = f.next().expect("id").parse().expect("integer id");
            let cells: Vec<Cell> = f.map(|c| (c != r"\N").then(|| c.to_string())).collect();
            assert_eq!(cells.len(), COLUMNS.len(), "row {id}: column count");
            (id, cells)
        })
        .collect()
}

/// Rows stored in a fixed shuffled order; text as codes in first-seen order.
fn source(rows: &Fixture, picked: &[u32]) -> Source {
    let n = rows.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut x = 0x5EED_u64;
    for i in (1..n).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        order.swap(i, (x % (i as u64 + 1)) as usize);
    }
    let words = n.div_ceil(64);
    let mut lanes = vec![vec![0u32; n]; COLUMNS.len()];
    let mut presence = vec![vec![0u64; words]; COLUMNS.len()];
    let mut codes: Vec<Vec<String>> = vec![Vec::new(); COLUMNS.len()];
    let mut row_of = BTreeMap::new();
    for (row, &i) in order.iter().enumerate() {
        let (id, cells) = &rows[i];
        row_of.insert(*id, row as u32);
        for (f, (cell, (_, kind))) in cells.iter().zip(COLUMNS).enumerate() {
            let Some(text) = cell else { continue };
            lanes[f][row] = match kind {
                Kind::Int => text.parse().expect("integer"),
                Kind::Bool => u32::from(text == "t"),
                Kind::Text => match codes[f].iter().position(|c| c == text) {
                    Some(code) => code as u32,
                    None => {
                        codes[f].push(text.clone());
                        (codes[f].len() - 1) as u32
                    }
                },
            };
            presence[f][row / 64] |= 1 << (row % 64);
        }
    }
    Source {
        lanes,
        presence,
        codes,
        selection: picked.iter().map(|id| row_of[id]).collect(),
        selection_space: Counter::new(SPACE),
        frames: FrameStore::new(FrameRevisions {
            source: 1,
            relation_map: 1,
            coordinates: 1,
        }),
        lookups: Counter::new(0),
    }
}

/// How the read treats presence and the selection; the two disable arms.
#[derive(Clone, Copy)]
struct Arm {
    presence: bool,
    bag: bool,
    witnessed: bool,
}

const GATE: Arm = Arm {
    presence: true,
    bag: true,
    witnessed: true,
};

/// ONE `borrow_all_selected_present`, then the cells spelled out per position.
fn answer(w: &World, arm: Arm) -> Answer {
    let (lanes, positions) = w
        .binding
        .borrow_all_selected_present(&w.store, &w.view, &w.source.frames, &w.source, 0)
        .expect("the binding is current");
    let lanes: Vec<(u8, &[u32], &[u64])> = lanes.map(|l| l.expect("lane")).collect();
    let mut positions: Vec<u32> = positions.to_vec();
    if !arm.bag {
        positions.sort_unstable();
        positions.dedup();
    }
    if !arm.witnessed {
        positions = (0..positions.len() as u32).collect();
    }
    let rows = positions
        .iter()
        .map(|&p| {
            let p = p as usize;
            lanes
                .iter()
                .map(|&(field, lane, plane)| {
                    if arm.presence && !bit(plane, p) {
                        return None;
                    }
                    let f = usize::from(field);
                    let v = lane[p];
                    Some(match COLUMNS[f].1 {
                        Kind::Int => v.to_string(),
                        Kind::Bool => if v == 1 { "t" } else { "f" }.to_string(),
                        Kind::Text => w.source.codes[f]
                            .get(v as usize)
                            .cloned()
                            .unwrap_or_default(),
                    })
                })
                .collect()
        })
        .collect();
    Answer::Rows(Rows {
        types: COLUMNS
            .iter()
            .map(|(_, k)| match k {
                Kind::Text => "text",
                Kind::Int => "integer",
                Kind::Bool => "boolean",
            })
            .map(String::from)
            .collect(),
        rows,
    })
}

struct World {
    store: BindStore,
    view: View,
    source: Source,
    binding: lance_graph_dt64::LaneBinding,
}

fn main() {
    let Ok(path) = std::env::var("VROW_TICKET_STATES") else {
        println!("SKIP: VROW_TICKET_STATES is not set (see the header for the fixture)");
        return;
    };
    let pg = match Pg::connect() {
        Ok(pg) => pg,
        Err(Skip(why)) => {
            println!("SKIP: {why}");
            return;
        }
    };
    let rows = load(&path);

    // Every third id, and every seventh listed again: a bag with repeats.
    let ids: Vec<u32> = rows.iter().map(|(id, _)| *id).collect();
    let mut picked: Vec<u32> = ids.iter().copied().step_by(3).collect();
    picked.extend(ids.iter().copied().step_by(7));
    let repeats = picked.len()
        - picked
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len();
    assert!(repeats > 0, "the selection lists some ids twice");

    let view = View(
        COLUMNS
            .iter()
            .map(|&(c, _)| FieldRef::new(format!("zammad:ticket_states/{c}"), c))
            .collect(),
    );
    let mut store = BindStore::new();
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: COLUMNS.len(),
                extent: Extent::DR64_CUBE,
            },
            Vec::new(),
        )
        .expect("register");
    let version = store.route(POSITION, CLASS).expect("route");
    let source = source(&rows, &picked);
    let frame = source.frames.seal();
    let fields: Vec<u8> = (0..COLUMNS.len() as u8).collect();
    let binding = store
        .bind_lanes(
            &view,
            &source.frames,
            &source,
            LaneRequest {
                route: RouteAt {
                    position: POSITION,
                    version,
                },
                rev: SchemaRev(1),
                fields: &WideFieldMask::from_positions(&fields),
                frame: &frame,
                scope: READER,
                selection: SPACE,
            },
        )
        .expect("bind");
    let w = World {
        store,
        view,
        source,
        binding,
    };

    // PostgreSQL: the same table and the same bag of ids.
    let mut script = String::from(
        "CREATE TABLE ticket_states (id int PRIMARY KEY, name text, next_state_id int, \
         default_create boolean, note text, active boolean);\nINSERT INTO ticket_states VALUES\n",
    );
    let sql_cell = |c: &Cell, kind: Kind| match (c, kind) {
        (None, _) => "NULL".to_string(),
        (Some(t), Kind::Text) => format!("'{}'", t.replace('\'', "''")),
        (Some(t), Kind::Bool) => (t == "t").to_string(),
        (Some(t), Kind::Int) => t.clone(),
    };
    let values: Vec<String> = rows
        .iter()
        .map(|(id, cells)| {
            let vals: Vec<String> = cells
                .iter()
                .zip(COLUMNS)
                .map(|(c, (_, k))| sql_cell(c, k))
                .collect();
            format!("({id}, {})", vals.join(", "))
        })
        .collect();
    script.push_str(&values.join(",\n"));
    script.push_str(";\n");
    pg.script(&script)
        .unwrap_or_else(|e| panic!("PostgreSQL rejected the fixture: {e}"));
    let list: Vec<String> = picked.iter().map(u32::to_string).collect();
    let sql = format!(
        "SELECT t.name, t.next_state_id, t.default_create, t.note, t.active \
         FROM unnest(ARRAY[{}]) AS v(id) JOIN ticket_states t USING (id)",
        list.join(",")
    );

    let mut report = Report::new();
    w.source.lookups.set(0);
    let gate = check(
        &pg,
        &mut report,
        &Case {
            name: "ticket_states bag",
            sql: &sql,
            order: Order::Unordered,
        },
        || answer(&w, GATE),
    );
    let checks = w.source.lookups.get();
    let local_index = check(
        &pg,
        &mut report,
        &Case {
            name: "DISABLE local index",
            sql: &sql,
            order: Order::Unordered,
        },
        || {
            answer(
                &w,
                Arm {
                    witnessed: false,
                    ..GATE
                },
            )
        },
    );
    let no_presence = check(
        &pg,
        &mut report,
        &Case {
            name: "DISABLE presence ignored",
            sql: &sql,
            order: Order::Unordered,
        },
        || {
            answer(
                &w,
                Arm {
                    presence: false,
                    ..GATE
                },
            )
        },
    );
    let as_set = check(
        &pg,
        &mut report,
        &Case {
            name: "DISABLE selection read as a set",
            sql: &sql,
            order: Order::Unordered,
        },
        || answer(&w, Arm { bag: false, ..GATE }),
    );

    // Foreign frame: another owner's store, with a frame of its own.
    let foreign = FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: 1,
    });
    let foreign_frame = w
        .binding
        .borrow_all_selected_present(&w.store, &w.view, &foreign, &w.source, 0)
        .err();
    // Swapped coordinate: the selection is issued in another row order.
    let swapped = CoordSpace {
        population: 1,
        coordinates: 2,
    };
    w.source.selection_space.set(swapped);
    let swapped_coordinate = w
        .binding
        .borrow_all_selected_present(&w.store, &w.view, &w.source.frames, &w.source, 0)
        .err();
    w.source.selection_space.set(SPACE);

    println!("PostgreSQL {}", pg.version());
    println!(
        "ticket_states: {} rows, {} selected positions ({repeats} repeated ids), {} lane lookups for {} bound fields",
        rows.len(),
        picked.len(),
        checks,
        COLUMNS.len()
    );
    println!("{}", report.render());
    println!("REFUSED foreign frame: {foreign_frame:?}");
    println!("REFUSED swapped coordinate: {swapped_coordinate:?}");
    assert_eq!(gate, Outcome::Pass, "the gate: equal to PostgreSQL");
    assert_eq!(
        checks,
        COLUMNS.len(),
        "one check: one lookup per bound field"
    );
    assert!(
        matches!(local_index, Outcome::Fail(_)),
        "disable arm 'local index' was not detected"
    );
    assert_eq!(
        foreign_frame,
        Some(LaneRefusal::Frame(FrameRefusal::ForeignFrame)),
        "disable arm 'foreign frame' was not refused"
    );
    assert!(
        matches!(swapped_coordinate, Some(LaneRefusal::SelectionSpace { .. })),
        "disable arm 'swapped coordinate' was not refused"
    );
    assert!(
        matches!(no_presence, Outcome::Fail(_)),
        "disable arm 'presence ignored' was not detected"
    );
    assert!(
        matches!(as_set, Outcome::Fail(_)),
        "disable arm 'selection read as a set' was not detected"
    );
}
