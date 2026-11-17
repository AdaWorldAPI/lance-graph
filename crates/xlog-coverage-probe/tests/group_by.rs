//! D-XLOG-W7-P0, the GROUP BY head (`w7-census-007bb19.tsv`).
//!
//! `sales_by_cat_region(Cat, Region, sum(Amount)) :- sales(Cat, Region,
//! Amount).` (xlog `e2e_integration_tests.rs`, literal 3) is the one wide head
//! in the census that reads a stored relation. It needs no tuple store: the
//! three `sales` lanes are read through ONE DT64 `LaneBinding::borrow_all`,
//! and the head is ONE mask-risc terminal, `GroupReduce` with the fused
//! two-column key `GroupKey::Pair { hi: Cat, lo: Region }` and
//! `GroupFold::SumSymI32(Amount)`. The symmetric sum keeps a group whose
//! values cancel to 0 apart from a group no row reached, which is Datalog's
//! rule: only groups with rows exist.
//!
//! Lab scope: the group universe is `(max Cat + 1) * stride` over the raw u32
//! values, `stride = max Region + 1`. A coordinate map that densifies the
//! values is not part of this.

use std::collections::BTreeMap;

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_dt64::dr::{Frame, FrameRevisions, FrameStore, Scope};
use lance_graph_dt64::{
    BindStore, ClassSchema, CoordSpace, Extent, LaneAddr, LaneBinding, LaneRequest, LaneSource,
    RouteAt, SchemaRev,
};
use lance_graph_mask_risc::{
    execute_into, Foreign, GroupFold, GroupKey, LaneRef, MaskOp, Operand, Out, Planes, Pred,
    Program, Scratch, Terminal,
};
use xlog_logic::{parse_program, Term};

const CLASS: u32 = 0x0C1C_0001;
const VIEW_CLASS: ClassId = 0x0C1C;
const POSITION: u16 = 4;
const ROWS: CoordSpace = CoordSpace {
    population: 1,
    coordinates: 1,
};
const READER: Scope = Scope { id: 3 };
const CAT: u8 = 0;
const REGION: u8 = 1;
const AMOUNT: u8 = 2;

/// The upstream program, verbatim.
const PROGRAM: &str = "
    sales(1, 10, 5).
    sales(1, 10, 7).
    sales(1, 11, 3).
    sales(2, 10, 2).

    sales_by_cat_region(Cat, Region, sum(Amount)) :- sales(Cat, Region, Amount).
";

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

fn view() -> View {
    View(
        ["Cat", "Region", "Amount"]
            .iter()
            .map(|l| FieldRef::new(format!("sales:{l}"), *l))
            .collect(),
    )
}

/// The owner of `sales`: its own lanes and frame store.
struct Sales {
    frames: FrameStore,
    cat: Vec<u32>,
    region: Vec<u32>,
    amount: Vec<i32>,
}

impl Sales {
    fn new(rows: &[(u32, u32, i32)]) -> Sales {
        Sales {
            frames: FrameStore::new(FrameRevisions {
                source: 1,
                relation_map: 1,
                coordinates: 1,
            }),
            cat: rows.iter().map(|r| r.0).collect(),
            region: rows.iter().map(|r| r.1).collect(),
            amount: rows.iter().map(|r| r.2).collect(),
        }
    }

    /// The `sales` facts of an XLOG program, as the source's lanes.
    fn from_program(src: &str) -> Sales {
        let p = parse_program(src).expect("program parses");
        let int = |t: &Term| match t {
            Term::Integer(v) => *v,
            other => panic!("non-integer fact term {other:?}"),
        };
        let rows: Vec<(u32, u32, i32)> = p
            .rules
            .iter()
            .filter(|r| r.is_fact() && r.head.predicate == "sales")
            .map(|r| {
                let t = &r.head.terms;
                (
                    u32::try_from(int(&t[0])).unwrap(),
                    u32::try_from(int(&t[1])).unwrap(),
                    i32::try_from(int(&t[2])).unwrap(),
                )
            })
            .collect();
        Sales::new(&rows)
    }
}

impl LaneSource for Sales {
    type Lane<'a> = LaneRef<'a>;

    fn lane_of(
        &self,
        class: u32,
        rev: SchemaRev,
        field: u8,
        _basis: &FieldRef,
        frame: &Frame,
    ) -> Option<LaneAddr> {
        self.frames.check(frame).ok()?;
        (class == CLASS && rev == SchemaRev(1) && field <= AMOUNT).then_some(LaneAddr {
            space: ROWS,
            lane: u16::from(field),
            scope: READER,
        })
    }

    fn borrow(&self, addr: &LaneAddr) -> Option<LaneRef<'_>> {
        match addr.lane {
            0 => Some(LaneRef::U32(&self.cat)),
            1 => Some(LaneRef::U32(&self.region)),
            2 => Some(LaneRef::I32(&self.amount)),
            _ => None,
        }
    }

    fn frames(&self) -> &FrameStore {
        &self.frames
    }
}

fn bind(source: &Sales, view: &View) -> (BindStore, LaneBinding) {
    let mut store = BindStore::new();
    store
        .register(
            view,
            CLASS,
            ClassSchema {
                schema_rev: SchemaRev(1),
                field_count: 3,
                extent: Extent::DR64_CUBE,
            },
            Vec::new(),
        )
        .unwrap();
    let version = store.route(POSITION, CLASS).unwrap();
    let frame = source.frames.seal();
    let fields = WideFieldMask::from_positions(&[CAT, REGION, AMOUNT]);
    let binding = store
        .bind_lanes(
            view,
            &source.frames,
            source,
            LaneRequest {
                route: RouteAt {
                    position: POSITION,
                    version,
                },
                rev: SchemaRev(1),
                fields: &fields,
                frame: &frame,
                scope: READER,
                selection: ROWS,
            },
        )
        .expect("sales binds");
    (store, binding)
}

/// The head, on the witnessed path: one `borrow_all`, one `GroupReduce`.
/// Returns `(cat, region, sum)` for every group a row reached, ascending.
fn grouped(source: &Sales) -> Vec<(u32, u32, i64)> {
    let view = view();
    let (store, binding) = bind(source, &view);
    let mut lanes = [LaneRef::U32(&[]); 3];
    for item in binding
        .borrow_all(&store, &view, &source.frames, source)
        .expect("the witness holds")
    {
        let (field, lane) = item.expect("every lane held");
        lanes[usize::from(field)] = lane;
    }
    let n = lanes[0].len();
    let max = |l: &[u32]| l.iter().copied().max().map_or(0, |m| m + 1);
    let stride = max(&source.region).max(1);
    let k = (max(&source.cat).max(1) * stride) as usize;
    let program = Program {
        ops: vec![MaskOp::Pred {
            pred: Pred::Range {
                lo: 0,
                hi: u32::try_from(n).unwrap(),
            },
            under: None,
            dst: 0,
        }],
        terminal: Terminal::GroupReduce {
            mask: Operand::Scratch(0),
            key: GroupKey::Pair {
                hi: u16::from(CAT),
                lo: u16::from(REGION),
                stride,
            },
            fold: GroupFold::SumSymI32(u16::from(AMOUNT)),
        },
        scratch_slots: 1,
    };
    let mut scratch = Scratch::for_program(&program, n).unwrap();
    let planes = Planes {
        n_rows: n,
        masks: &[],
        lanes: &lanes,
    };
    let mut sums = vec![0i64; k];
    execute_into(
        &program,
        &planes,
        &Foreign::NONE,
        &mut scratch,
        Out::I64(&mut sums),
    )
    .expect("the fold runs");
    let fold = GroupFold::SumSymI32(u16::from(AMOUNT));
    sums.iter()
        .enumerate()
        .filter(|(_, &v)| !fold.is_empty_slot(v))
        .map(|(g, &v)| (g as u32 / stride, g as u32 % stride, v))
        .collect()
}

/// The oracle: a plain group-by over the rows, no DT64 and no mask-risc.
fn oracle(rows: &[(u32, u32, i32)]) -> Vec<(u32, u32, i64)> {
    let mut m: BTreeMap<(u32, u32), i64> = BTreeMap::new();
    for &(c, r, a) in rows {
        *m.entry((c, r)).or_default() += i64::from(a);
    }
    m.into_iter().map(|((c, r), s)| (c, r, s)).collect()
}

/// The upstream program's facts, through the witnessed path, give exactly
/// the rows its own test asserts (`e2e_integration_tests.rs`):
/// `(1, 10, 12), (1, 11, 3), (2, 10, 2)`.
#[test]
fn the_upstream_program_gives_its_own_expected_rows() {
    let sales = Sales::from_program(PROGRAM);
    assert_eq!(sales.cat.len(), 4, "the four facts were read");
    assert_eq!(grouped(&sales), vec![(1, 10, 12), (1, 11, 3), (2, 10, 2)]);
}

/// On seeded data with negative amounts, groups that cancel to 0 and groups
/// no row reaches, the witnessed fold equals the oracle at widths 1..2000.
/// A group whose amounts cancel is present with sum 0; an unreached group
/// is absent.
#[test]
fn the_fold_equals_the_oracle_and_keeps_zero_sum_groups() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = |k: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % k
    };
    let (mut zero_sum, mut unreached) = (0, 0);
    for &n in &[1usize, 63, 64, 65, 300, 2000] {
        let mut rows: Vec<(u32, u32, i32)> = (0..n)
            .map(|_| {
                (
                    next(5) as u32,
                    // Region 3 is never drawn, so some groups are unreached.
                    [0u32, 1, 2, 4][next(4) as usize],
                    next(101) as i32 - 50,
                )
            })
            .collect();
        // A group whose amounts cancel exactly: category 5 is never drawn,
        // so only these two rows reach it.
        rows.extend([(5, 4, 9), (5, 4, -9)]);
        let want = oracle(&rows);
        assert_eq!(grouped(&Sales::new(&rows)), want, "n = {n}");
        zero_sum += want.iter().filter(|g| g.2 == 0).count();
        unreached += 6 * 5 - want.len();
    }
    // Anti-vacuity: zero-sum groups are present, and unreached ones exist.
    assert!(zero_sum >= 6, "zero-sum groups: {zero_sum}");
    assert!(unreached >= 30, "unreached groups: {unreached}");
}

/// A source whose frame moved is refused before any lane is read.
#[test]
fn a_moved_frame_is_refused_before_the_fold() {
    let mut sales = Sales::from_program(PROGRAM);
    let view = view();
    let (store, binding) = bind(&sales, &view);
    sales.frames.advance(FrameRevisions {
        source: 2,
        relation_map: 1,
        coordinates: 1,
    });
    assert!(binding
        .borrow_all(&store, &view, &sales.frames, &sales)
        .is_err());
}
