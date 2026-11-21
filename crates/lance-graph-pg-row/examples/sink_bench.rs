//! `rows_at` against `write_rows` over a bag of 4096 positions in a table of
//! 10^5 rows × 4 columns, timed as the median of 11 runs.
//!
//! Run with `cargo run --release -p lance-graph-pg-row --example sink_bench`.
//! The `CopyText` bytes are checked against a reference encoding of the
//! `PgRows` before anything is timed.

use std::hint::black_box;
use std::time::Instant;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::pg_row::{AbsentAs, CodeText, LaneEncoding, PgColumn, PgType};
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRequest, LaneSource,
    PresenceSource, RouteAt, SchemaRev,
};
use lance_graph_mask_risc::LaneRef;
use lance_graph_pg_row::sink::{CellSink, CopyText};
use lance_graph_pg_row::{ColumnSpec, LiveSource, PgRows, Reader};

const ROWS: usize = 100_000;
const BAG: usize = 4096;
const RUNS: usize = 11;

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

#[derive(Clone)]
enum V {
    Int(i32),
    Text(String),
    Bool(bool),
    Secs(u32),
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

fn rows() -> Vec<Vec<Option<V>>> {
    let mut x = 0x7E57_u64;
    (0..ROWS)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let title = match x % 9 {
                0 => None,
                1 => Some(V::Text(String::new())),
                2 => Some(V::Text("tab\there\\back".to_owned())),
                k => Some(V::Text(
                    ["printer", "vpn", "login", "mail"][k as usize % 4].to_owned(),
                )),
            };
            vec![
                Some(V::Int(i as i32 - 50_000)),
                title,
                Some(V::Bool(x >> 9 & 1 == 1)),
                (!x.is_multiple_of(5))
                    .then(|| V::Secs(1_600_000_000 + (x >> 11) as u32 % 100_000_000)),
            ]
        })
        .collect()
}

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

struct CountingSink {
    cells: usize,
}

impl CellSink for CountingSink {
    fn cell(&mut self, _text: Option<&str>) {
        self.cells += 1;
    }
}

/// Median of `RUNS` timings of `f`, in nanoseconds.
fn median_ns(mut f: impl FnMut()) -> u128 {
    let mut times: Vec<u128> = (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_nanos()
        })
        .collect();
    times.sort_unstable();
    times[RUNS / 2]
}

fn main() {
    let source = Source::new(0x2B01_0001, 11, &TICKET, &rows(), |_| false);
    let b = bind("ticket", 0x2B01_0001, 3, &TICKET, source);
    let cols: Vec<ColumnSpec<'_>> = b
        .defs
        .iter()
        .zip(&b.source.codes)
        .map(|((name, col), codes)| ColumnSpec {
            name,
            col: *col,
            codes: (col.encoding == LaneEncoding::TextCode).then_some(codes as &dyn CodeText),
        })
        .collect();
    let reader =
        Reader::open(&b.binding, &b.store, &b.source.frames, &b.source, &cols).expect("open");

    let mut x = 0x51_u64;
    let positions: Vec<u32> = (0..BAG)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % ROWS as u64) as u32
        })
        .collect();

    let reference = reference_copy(&reader.rows_at(&positions).expect("rows_at"));
    let mut copy = CopyText::new();
    reader
        .write_rows(&positions, &mut copy)
        .expect("write_rows");
    assert_eq!(
        copy.out, reference,
        "CopyText differs from the reference encoding"
    );
    let bytes = copy.out.len();

    let at = median_ns(|| {
        black_box(reader.rows_at(black_box(&positions)).expect("rows_at"));
    });
    let mut counting = CountingSink { cells: 0 };
    let count = median_ns(|| {
        reader
            .write_rows(black_box(&positions), &mut counting)
            .expect("write_rows");
    });
    assert_eq!(counting.cells, RUNS * BAG * 4);
    let to_copy = median_ns(|| {
        copy.out.clear();
        reader
            .write_rows(black_box(&positions), &mut copy)
            .expect("write_rows");
        black_box(&copy.out);
    });
    assert_eq!(copy.out.len(), bytes);

    println!("{ROWS} rows x 4 columns, bag of {BAG}, median of {RUNS}");
    for (name, ns) in [
        ("rows_at", at),
        ("write_rows -> CountingSink", count),
        ("write_rows -> CopyText", to_copy),
    ] {
        println!(
            "{name:<28} {ns:>9} ns per call {:>8.1} ns per row",
            ns as f64 / BAG as f64
        );
    }
    println!("CopyText: {bytes} bytes per call");
}
