//! The streaming cell writer (D-INTAKE-2G): `write_rows`, `write_rows_in` and
//! `write_bag` agree with `rows_at`, `rows_in` and `read_bag`, `CopyText` is
//! PostgreSQL's COPY text, and the writer allocates nothing per cell or row.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::pg_row::{AbsentAs, CodeText, LaneEncoding, PgColumn, PgType};
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRequest, LaneSource,
    PresenceSource, RouteAt, SchemaRev, SelAddr, SelectionSource,
};
use lance_graph_mask_risc::LaneRef;
use lance_graph_pg_row::sink::{CellSink, CollectRows, CopyText};
use lance_graph_pg_row::{read_bag, BagSide, ColumnSpec, LiveSource, PgRowRefusal, PgRows, Reader};

// ---- a counting allocator, per thread ------------------------------------

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.with(Cell::get)
}

// ---- the fixture ------------------------------------------------------------

const READER: Scope = Scope { id: 7 };

const fn col(ty: PgType, encoding: LaneEncoding) -> PgColumn {
    PgColumn {
        ty,
        encoding,
        absent: AbsentAs::Null,
    }
}

const TICKET: [(&str, PgColumn); 4] = [
    ("id", col(PgType::Int4, LaneEncoding::Signed)),
    ("title", col(PgType::Text, LaneEncoding::TextCode)),
    ("open", col(PgType::Bool, LaneEncoding::Bool)),
    (
        "created",
        col(PgType::Timestamp(Some(3)), LaneEncoding::EpochSeconds),
    ),
];

const AGENT: [(&str, PgColumn); 2] = [
    ("id", col(PgType::Int4, LaneEncoding::Signed)),
    ("name", col(PgType::Text, LaneEncoding::TextCode)),
];

const TICKETS: usize = 300;
const TRICKY: &str = "tab\there\\back\nline\r";

#[derive(Clone)]
enum V {
    Int(i32),
    Text(String),
    Bool(bool),
    Secs(u32),
}

fn ticket_rows() -> Vec<Vec<Option<V>>> {
    (0..TICKETS)
        .map(|i| {
            let title = match i {
                _ if i % 7 == 0 => None,
                _ if i % 5 == 1 => Some(V::Text(String::new())),
                _ if i % 5 == 2 => Some(V::Text(TRICKY.to_owned())),
                _ => Some(V::Text(["printer", "vpn", "login"][i % 3].to_owned())),
            };
            vec![
                Some(V::Int(i as i32 - 150)),
                title,
                Some(V::Bool(i % 2 == 0)),
                (i % 4 != 0).then(|| V::Secs(1_600_000_000 + i as u32 * 86_401)),
            ]
        })
        .collect()
}

fn agent_rows() -> Vec<Vec<Option<V>>> {
    ["ana", "bo", "cy", "dee", "eli", "fay", "gus", "hal"]
        .iter()
        .enumerate()
        .map(|(i, n)| vec![Some(V::Int(i as i32 + 1)), Some(V::Text((*n).to_owned()))])
        .collect()
}

fn ticket_dead(i: usize) -> bool {
    i % 13 == 5
}

enum Lane {
    U32(Vec<u32>),
    I32(Vec<i32>),
}

struct Source {
    class: u32,
    space: CoordSpace,
    lanes: Vec<Lane>,
    presence: Vec<Vec<u64>>,
    live: Vec<u64>,
    codes: Vec<Vec<String>>,
    frames: FrameStore,
}

impl Source {
    fn new(
        class: u32,
        population: u64,
        defs: &[(&str, PgColumn)],
        rows: &[Vec<Option<V>>],
        dead: impl Fn(usize) -> bool,
    ) -> Source {
        let n = rows.len();
        let words = n.div_ceil(64);
        let mut lanes: Vec<Lane> = defs
            .iter()
            .map(|(_, c)| match c.encoding {
                LaneEncoding::Signed => Lane::I32(vec![0; n]),
                _ => Lane::U32(vec![0; n]),
            })
            .collect();
        let mut presence = vec![vec![0u64; words]; defs.len()];
        let mut live = vec![0u64; words];
        // Code 0 is a real string, so reading an absent cell as its value
        // would show up as that string.
        let mut codes: Vec<Vec<String>> = vec![vec!["<code 0>".to_owned()]; defs.len()];
        for (at, row) in rows.iter().enumerate() {
            if !dead(at) {
                live[at / 64] |= 1 << (at % 64);
            }
            for (f, cell) in row.iter().enumerate() {
                let Some(v) = cell else { continue };
                let raw = match v {
                    V::Int(i) => *i as u32,
                    V::Bool(b) => u32::from(*b),
                    V::Secs(s) => *s,
                    V::Text(s) => match codes[f].iter().position(|c| c == s) {
                        Some(c) => c as u32,
                        None => {
                            codes[f].push(s.clone());
                            (codes[f].len() - 1) as u32
                        }
                    },
                };
                match &mut lanes[f] {
                    Lane::U32(l) => l[at] = raw,
                    Lane::I32(l) => l[at] = raw as i32,
                }
                presence[f][at / 64] |= 1 << (at % 64);
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
            frames: FrameStore::new(FrameRevisions {
                source: population,
                relation_map: 1,
                coordinates: 1,
            }),
        }
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
    fn selection_of(&self, _sel: u16, _frame: &Frame) -> Option<SelAddr> {
        None
    }
    fn borrow_positions(&self, _addr: &SelAddr) -> Option<&[u32]> {
        None
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

struct Bound {
    store: BindStore,
    binding: LaneBinding,
    source: Source,
    defs: Vec<(&'static str, PgColumn)>,
}

fn bind(
    table: &str,
    class: u32,
    position: u16,
    defs: &[(&'static str, PgColumn)],
    source: Source,
) -> Bound {
    let view = View {
        class: (class >> 16) as ClassId,
        fields: defs
            .iter()
            .map(|(name, _)| FieldRef::new(format!("test:{table}/{name}"), *name))
            .collect(),
    };
    let mut store = BindStore::new();
    store
        .register(
            &view,
            class,
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: defs.len(),
                extent: Extent::DR64_CUBE,
            },
            Vec::new(),
        )
        .expect("register");
    let version = store.route(position, class).expect("route");
    let frame = source.frames.seal();
    let fields: Vec<u8> = (0..defs.len() as u8).collect();
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
    Bound {
        store,
        binding,
        source,
        defs: defs.to_vec(),
    }
}

fn tickets() -> Bound {
    let source = Source::new(0x2B01_0001, 11, &TICKET, &ticket_rows(), ticket_dead);
    bind("ticket", 0x2B01_0001, 3, &TICKET, source)
}

fn agents() -> Bound {
    let source = Source::new(0x2B02_0001, 12, &AGENT, &agent_rows(), |_| false);
    bind("agent", 0x2B02_0001, 4, &AGENT, source)
}

fn specs(b: &Bound) -> Vec<ColumnSpec<'_>> {
    b.defs
        .iter()
        .zip(&b.source.codes)
        .map(|((name, col), codes)| ColumnSpec {
            name,
            col: *col,
            codes: (col.encoding == LaneEncoding::TextCode).then_some(codes as &dyn CodeText),
        })
        .collect()
}

fn open<'s, 'c>(b: &'s Bound, cols: &'c [ColumnSpec<'c>]) -> Reader<'s, 'c> {
    Reader::open(&b.binding, &b.store, &b.source.frames, &b.source, cols).expect("open")
}

/// Live ticket positions, a bag with repeats: every third, every seventh,
/// and position 0 three more times.
fn bag() -> Vec<u32> {
    let live: Vec<u32> = (0..TICKETS)
        .filter(|&i| !ticket_dead(i))
        .map(|i| i as u32)
        .collect();
    let mut bag: Vec<u32> = live.iter().copied().step_by(3).collect();
    bag.extend(live.iter().copied().step_by(7));
    bag.extend([0, 0, 0]);
    bag
}

macro_rules! setup {
    ($tb:ident, $tr:ident, $ar:ident) => {
        let $tb = tickets();
        let ab = agents();
        let tcols = specs(&$tb);
        let acols = specs(&ab);
        let $tr = open(&$tb, &tcols);
        let $ar = open(&ab, &acols);
    };
}

// ---- the reference COPY encoding, built from PgRows ----------------------------

fn reference_copy(rows: &PgRows) -> Vec<u8> {
    let mut out = String::new();
    for row in &rows.rows {
        let fields: Vec<String> = row
            .iter()
            .map(|cell| match cell {
                None => "\\N".to_owned(),
                Some(text) => text
                    .replace('\\', "\\\\")
                    .replace('\t', "\\t")
                    .replace('\n', "\\n")
                    .replace('\r', "\\r")
                    .replace('\u{8}', "\\b")
                    .replace('\u{b}', "\\v")
                    .replace('\u{c}', "\\f"),
            })
            .collect();
        out.push_str(&fields.join("\t"));
        out.push('\n');
    }
    out.into_bytes()
}

// ---- tests -------------------------------------------------------------------

#[test]
fn write_rows_equals_rows_at() {
    setup!(tb, tr, ar);
    let _ = (&tb, &ar);
    let bag = bag();
    let mut sink = CollectRows::new();
    let n = tr.write_rows(&bag, &mut sink).expect("write_rows");
    let expected = tr.rows_at(&bag).expect("rows_at");
    assert_eq!(n, bag.len());
    assert_eq!(sink.rows, expected.rows);
    // Not vacuous: the bag repeats positions and holds NULLs and ''.
    assert!(sink.rows.iter().any(|r| r[1].is_none()));
    assert!(sink.rows.iter().any(|r| r[1].as_deref() == Some("")));
    assert!(sink.rows.iter().any(|r| r[1].as_deref() == Some(TRICKY)));
    assert!(sink.rows.iter().any(|r| r[3].is_none()));
    assert!(sink
        .rows
        .iter()
        .any(|r| r[0].as_deref().is_some_and(|s| s.starts_with('-'))));
}

#[test]
fn write_rows_in_equals_rows_in() {
    setup!(tb, tr, ar);
    let _ = &ar;
    for mask in [
        tb.source.live.clone(),
        tb.source
            .live
            .iter()
            .map(|w| w & 0x4924_9249_2492_4924)
            .collect(),
    ] {
        let mut sink = CollectRows::new();
        let n = tr.write_rows_in(&mask, &mut sink).expect("write_rows_in");
        let expected = tr.rows_in(&mask).expect("rows_in");
        assert_eq!(n, expected.rows.len());
        assert_eq!(sink.rows, expected.rows);
        assert!(n > 20);
    }
    let short = [0u64];
    assert_eq!(
        tr.write_rows_in(&short, &mut CollectRows::new())
            .unwrap_err(),
        tr.rows_in(&short).unwrap_err()
    );
}

#[test]
fn write_bag_equals_read_bag() {
    setup!(tb, tr, ar);
    let _ = &tb;
    let a = bag();
    // Agent positions cycle; index i has an agent unless i % 3 == 0.
    let b: Vec<u32> = (0..a.len()).map(|i| (i % 8) as u32).collect();
    let mut present = vec![0u64; a.len().div_ceil(64)];
    for i in (0..a.len()).filter(|i| i % 3 != 0) {
        present[i / 64] |= 1 << (i % 64);
    }
    let sides = [
        BagSide {
            reader: &tr,
            positions: &a,
            present: None,
        },
        BagSide {
            reader: &ar,
            positions: &b,
            present: Some(&present),
        },
    ];
    let mut sink = CollectRows::new();
    let n = write_bag_via(&sides, &mut sink);
    let expected = read_bag(&sides).expect("read_bag");
    assert_eq!(n, a.len());
    assert_eq!(sink.rows, expected.rows);
    assert!(sink.rows.iter().all(|r| r.len() == 6));
    // The missing side is NULL on both of its columns, and only there.
    let missing = sink
        .rows
        .iter()
        .filter(|r| r[4].is_none() && r[5].is_none())
        .count();
    assert_eq!(missing, a.len().div_ceil(3));

    let mut none_sides = CollectRows::new();
    assert_eq!(
        lance_graph_pg_row::sink::write_bag(&[], &mut none_sides).unwrap_err(),
        read_bag(&[]).unwrap_err()
    );
    let short = [BagSide {
        reader: &tr,
        positions: &a,
        present: Some(&present[..1]),
    }];
    assert_eq!(
        lance_graph_pg_row::sink::write_bag(&short, &mut none_sides).unwrap_err(),
        read_bag(&short).unwrap_err()
    );
}

fn write_bag_via(sides: &[BagSide<'_, '_, '_>], sink: &mut impl CellSink) -> usize {
    lance_graph_pg_row::sink::write_bag(sides, sink).expect("write_bag")
}

#[test]
fn copy_text_equals_reference_encoding() {
    setup!(tb, tr, ar);
    let _ = (&tb, &ar);
    let bag = bag();
    let mut copy = CopyText::new();
    tr.write_rows(&bag, &mut copy).expect("write_rows");
    let rows = tr.rows_at(&bag).expect("rows_at");
    assert_eq!(copy.out, reference_copy(&rows));

    let text = String::from_utf8(copy.out.clone()).expect("utf8");
    // NULL is `\N`; '' is an empty field between two tabs; the tab, backslash,
    // newline and carriage return of the value are escaped.
    assert!(text.contains("\\N"));
    assert!(text.contains("\t\t"));
    assert!(text.contains("tab\\there\\\\back\\nline\\r"));
    assert!(!text.contains('\r'));
    let lines: Vec<&str> = text.split('\n').collect();
    assert_eq!(lines.len(), bag.len() + 1);
    assert_eq!(lines[bag.len()], "");
    // Position 0: id -150, title NULL, open true, created NULL.
    assert_eq!(lines[bag.len() - 1], "-150\t\\N\tt\t\\N");
    for line in &lines[..bag.len()] {
        assert_eq!(line.matches('\t').count(), 3, "{line:?}");
    }

    // A second use of the same sink, cleared, repeats the bytes.
    let first = copy.out.clone();
    copy.out.clear();
    tr.write_rows(&bag, &mut copy).expect("write_rows");
    assert_eq!(copy.out, first);
}

#[test]
fn dead_and_out_of_range_positions_are_refused_as_rows_at_refuses() {
    setup!(tb, tr, ar);
    let _ = (&tb, &ar);
    assert!(ticket_dead(5) && !ticket_dead(0) && !ticket_dead(1));
    for positions in [vec![0u32, 5, 1], vec![1, TICKETS as u32]] {
        let mut sink = CollectRows::new();
        let ours: PgRowRefusal = tr.write_rows(&positions, &mut sink).unwrap_err();
        assert_eq!(ours, tr.rows_at(&positions).unwrap_err());
        // Rows before the refusal reached the sink: refusal is not atomic.
        assert_eq!(sink.rows.len(), 1);
    }
    assert_eq!(
        tr.write_rows(&[5], &mut CollectRows::new()).unwrap_err(),
        PgRowRefusal::DeadPosition { at: 5 }
    );
    assert_eq!(
        tr.write_rows(&[TICKETS as u32], &mut CollectRows::new())
            .unwrap_err(),
        PgRowRefusal::Position { at: TICKETS as u32 }
    );
    let mut mask = tb.source.live.clone();
    mask[0] |= 1 << 5;
    assert_eq!(
        tr.write_rows_in(&mask, &mut CollectRows::new())
            .unwrap_err(),
        tr.rows_in(&mask).unwrap_err()
    );
}

#[derive(Default)]
struct CountingSink {
    rows: usize,
    cells: usize,
    nulls: usize,
}

impl CellSink for CountingSink {
    fn begin_row(&mut self) {
        self.rows += 1;
    }
    fn cell(&mut self, text: Option<&str>) {
        self.cells += 1;
        self.nulls += usize::from(text.is_none());
    }
}

#[test]
fn write_rows_allocates_nothing_per_cell_or_row() {
    setup!(tb, tr, ar);
    let _ = (&tb, &ar);
    let live: Vec<u32> = (0..TICKETS as u32)
        .filter(|&i| !ticket_dead(i as usize))
        .collect();
    let positions: Vec<u32> = (0..4096).map(|k| live[k % live.len()]).collect();

    let mut sink = CountingSink::default();
    let before = allocs();
    let n = tr.write_rows(&positions, &mut sink).expect("write_rows");
    let during = allocs() - before;
    assert_eq!(n, 4096);
    assert_eq!((sink.rows, sink.cells), (4096, 4096 * 4));
    assert!(sink.nulls > 0 && sink.nulls < sink.cells);
    assert!(during < 4, "{during} allocations over 4096 rows");

    // The counter counts: the materializing sink allocates per cell.
    let mut collect = CollectRows::new();
    let before = allocs();
    tr.write_rows(&positions, &mut collect).expect("write_rows");
    assert!(allocs() - before > 4096);
}
