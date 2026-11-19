//! A registered method called over a cube relation's lent lanes.
//!
//! The method has no WHEN. What it computes is its DO: an ordered list of
//! effects, each a frozen word of the polyfill lens (`0xFFFF :: 0xF000`)
//! addressed by position, with its operands. A call does the work:
//! `Prepared::prepare` admits the reference, borrows the method's reads from
//! the cube through the lane socket and binds the words once; `run` executes
//! the bound plan over the cube's own words. Nothing is evaluated for a
//! method that is not called.

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Focus, Guards, LocalMap, MapTable, Scope, Width};
use lance_graph_dt64::{
    AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, LaneRefusal, LaneRequest,
    MethodBinding, MethodId, MethodRef, RouteAt, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mask_risc::{Foreign, LaneRef, Planes};
use lance_graph_r2il::binding::{
    prepare, run, ExecPermit, ExecRefusal, FieldLanes, Prepared as Plan, R2ilLens, Resolution,
};
use lance_graph_r2il::polyfill::{
    polyfill_arities, position_of, HARDCODED, POLYFILL, POLYFILL_LENS,
};
use lance_graph_r2il::FoldDialect;
use lance_graph_xlog::cube::CubeLanes;

pub const TILE: u32 = 5;
pub const CLASS: u32 = 0x7F0E_0001;
const CLASS_VIEW: ClassId = 0x7F0E;
pub const REV: SchemaRev = SchemaRev(1);
pub const POSITION: u16 = 2;
pub const READER: Scope = Scope { id: 9 };
const FOCUS: Focus = Focus { revision: 1 };
pub const X: &str = "lab:edge/x";
pub const Y: &str = "lab:edge/y";
/// Field positions in the view: `y` is listed before `x`.
pub const FIELD_Y: u8 = 0;
pub const FIELD_X: u8 = 1;
/// Fields from here on hold a method's intermediate values and its answer.
const SCRATCH: u8 = 2;
const N_FIELDS: usize = 8;
const LENS: R2ilLens = R2ilLens {
    lens: POLYFILL_LENS,
    schema_rev: REV,
};

pub struct View {
    class: Vec<FieldRef>,
    lens: Vec<FieldRef>,
}

impl View {
    fn new() -> Self {
        let mut class = vec![FieldRef::new(Y, "y"), FieldRef::new(X, "x")];
        class.extend(
            (class.len()..N_FIELDS)
                .map(|i| FieldRef::new(format!("lab:edge/t{i}"), format!("t{i}"))),
        );
        let lens = POLYFILL
            .iter()
            .map(|w| FieldRef::new(format!("lab:polyfill/{}", w.aliases[0]), w.aliases[0]))
            .collect();
        View { class, lens }
    }
}

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        match class {
            CLASS_VIEW => &self.class,
            HARDCODED => &self.lens,
            _ => &[],
        }
    }
    fn lens_fields(&self, lens: u32) -> Option<&[FieldRef]> {
        (lens == POLYFILL_LENS).then_some(self.lens.as_slice())
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Card
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

/// One tile holding `ids`.
pub fn table(ids: &[u32]) -> MapTable {
    let mut all: Vec<u64> = ids.iter().map(|&i| u64::from(i)).collect();
    all.sort_unstable();
    all.dedup();
    let mut t = MapTable::new(1, 1);
    t.advance(
        1,
        1,
        vec![(TILE, Some(LocalMap::prepare(Width::W256, &all).unwrap()))],
    );
    t
}

/// One effect: the word named `name` in any of the lens's languages, by the
/// position that name is an alias of.
fn word(target: u8, name: &str, args: Vec<Arg>) -> Effect {
    Effect {
        target,
        algebra: AlgebraRef {
            lens: POLYFILL_LENS,
            word: position_of(name).unwrap_or_else(|| panic!("{name:?} is no polyfill alias")),
        },
        args,
    }
}

/// The field a method leaves its answer in.
pub const ANSWER: u8 = SCRATCH + 4;

/// The method `from(src) and not to(dst)`, counting the edges out of `src`
/// that do not end in `dst`, as five words:
/// `=`(x, src), `=`(y, dst), NOT, AND, COUNT. The lanes hold tile
/// coordinates, so the two constants are the coordinates of the ids in
/// `table`'s current frame.
pub fn method(id: u32, table: &MapTable, src: u32, dst: u32) -> MethodBinding {
    let frame = table.seal();
    let map = table.map(&frame, TILE).unwrap();
    let code = |g: u32| i64::from(map.local(u64::from(g)).unwrap());
    let (from, to, not_to, both) = (SCRATCH, SCRATCH + 1, SCRATCH + 2, SCRATCH + 3);
    let effects = vec![
        word(from, "=", vec![Arg::Field(FIELD_X), Arg::Const(code(src))]),
        word(to, "=", vec![Arg::Field(FIELD_Y), Arg::Const(code(dst))]),
        word(not_to, "NOT", vec![Arg::Field(to)]),
        word(both, "AND", vec![Arg::Field(from), Arg::Field(not_to)]),
        word(ANSWER, "COUNT", vec![Arg::Field(both)]),
    ];
    let reads = WideFieldMask::from_positions(&[FIELD_X, FIELD_Y]);
    MethodBinding {
        id: MethodId(id),
        sig_rev: SigRev(1),
        name: format!("from {src} and not to {dst}"),
        on: reads.clone(),
        when: None,
        reads,
        effects,
        population: Population::Bag,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

pub struct World {
    pub store: BindStore,
    pub view: View,
    pub route: RouteAt,
}

pub fn world(methods: Vec<MethodBinding>) -> World {
    let mut store = BindStore::new();
    let view = View::new();
    store
        .declare_algebra(POLYFILL_LENS, REV, Extent::DR64_CUBE, polyfill_arities())
        .unwrap();
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: REV,
                field_count: N_FIELDS,
                extent: Extent::DR64_CUBE,
            },
            methods,
        )
        .unwrap();
    let version = store.route(POSITION, CLASS).unwrap();
    World {
        store,
        view,
        route: RouteAt {
            position: POSITION,
            version,
        },
    }
}

// The payloads are read through `Debug` only.
#[allow(dead_code)]
#[derive(Debug)]
pub enum MethodRefusal {
    /// The words could not be bound, or the reference was not admitted.
    Prepare(ExecRefusal),
    Lane(LaneRefusal),
    /// A field the method reads was not lent.
    NotLent(u8),
    Run(ExecRefusal),
    /// The method wrote no value to its answer field.
    NoAnswer,
}

/// A method made ready by one call's preparation.
pub struct Prepared<'s> {
    plan: Plan,
    lanes: Vec<LaneRef<'s>>,
    field_lanes: [Option<u16>; N_FIELDS],
    rows: usize,
}

impl<'s> Prepared<'s> {
    pub fn prepare(
        w: &World,
        source: &'s CubeLanes<'_>,
        table: &MapTable,
        r: &MethodRef,
    ) -> Result<Prepared<'s>, MethodRefusal> {
        let guards = Guards {
            frame: table.seal(),
            focus: FOCUS,
            scope: READER,
        };
        let plan = prepare(
            &w.store,
            &w.view,
            LENS,
            &Resolution {
                method: r,
                schema_rev: REV,
                route: (w.route.version, POSITION),
                guards: &guards,
            },
            table.frames(),
            (FOCUS, READER),
        )
        .map_err(MethodRefusal::Prepare)?;
        let reads = WideFieldMask::from_positions(&[FIELD_X, FIELD_Y]);
        let binding = w
            .store
            .bind_lanes(
                &w.view,
                table.frames(),
                source,
                LaneRequest {
                    route: w.route,
                    rev: REV,
                    fields: &reads,
                    frame: &guards.frame,
                    scope: READER,
                    selection: source.space(),
                },
            )
            .map_err(MethodRefusal::Lane)?;
        let mut lanes = Vec::new();
        let mut field_lanes = [None; N_FIELDS];
        let mut rows = 0;
        for lent in binding
            .borrow_all_present(&w.store, &w.view, table.frames(), source)
            .map_err(MethodRefusal::Lane)?
        {
            let (field, lane, _live) = lent.map_err(MethodRefusal::Lane)?;
            if let LaneRef::U32(values) = lane {
                rows = values.len();
            }
            field_lanes[usize::from(field)] = Some(lanes.len() as u16);
            lanes.push(lane);
        }
        for field in [FIELD_X, FIELD_Y] {
            if field_lanes[usize::from(field)].is_none() {
                return Err(MethodRefusal::NotLent(field));
            }
        }
        Ok(Prepared {
            plan,
            lanes,
            field_lanes,
            rows,
        })
    }

    /// One call: the value the method leaves in its answer field.
    pub fn run(&self, w: &World, table: &MapTable) -> Result<u64, MethodRefusal> {
        let planes = Planes {
            n_rows: self.rows,
            masks: &[],
            lanes: &self.lanes,
        };
        let foreign = Foreign::NONE;
        let mut dialect = FoldDialect::new(&planes, &foreign);
        let receipt = run(
            &self.plan,
            &w.store,
            table.frames(),
            (FOCUS, READER),
            &ExecPermit {
                methods: vec![self.plan.method().clone()],
            },
            FieldLanes {
                lanes: &self.field_lanes,
            },
            &mut dialect,
        )
        .map_err(MethodRefusal::Run)?;
        receipt
            .output
            .values
            .iter()
            .find(|(f, _)| *f == ANSWER)
            .map(|(_, v)| *v as u64)
            .ok_or(MethodRefusal::NoAnswer)
    }
}

/// The same count by walking the edges; shares no code with the cube.
pub fn oracle(edges: &[(u32, u32)], src: u32, dst: u32) -> u64 {
    edges.iter().filter(|&&(a, b)| a == src && b != dst).count() as u64
}
