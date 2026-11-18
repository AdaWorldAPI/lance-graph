//! A registered method executed over a cube relation's lent lanes.
//!
//! The store holds the method with its WHEN guard over the class's field
//! positions. `Prepared::prepare` does everything that depends on the binding
//! once: admit the reference, bind the method's reads through the lane
//! socket, borrow both lanes and the live plane, lower the guard. `run` only
//! executes the lowered program over the cube's own words.

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{LocalMap, MapTable, Scope, Width};
use lance_graph_dt64::{
    BindStore, ClassSchema, Extent, Guard, LaneRefusal, LaneRequest, MethodBinding, MethodId,
    MethodRef, Refusal, RouteAt, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mask_risc::{execute_into, Foreign, LaneRef, Out, Planes, Program, Scratch, Value};
use lance_graph_quack::set::{lower_set, SetLowered};
use lance_graph_quack::{lower, Agg, Filter, Mask, Query};
use lance_graph_xlog::cube::CubeLanes;

pub const TILE: u32 = 5;
pub const CLASS: u32 = 0x7F0E_0001;
const CLASS_VIEW: ClassId = 0x7F0E;
pub const REV: SchemaRev = SchemaRev(1);
pub const POSITION: u16 = 2;
pub const READER: Scope = Scope { id: 9 };
pub const X: &str = "lab:edge/x";
pub const Y: &str = "lab:edge/y";
/// Field positions in the view: `y` is listed before `x`.
pub const FIELD_Y: u16 = 0;
pub const FIELD_X: u16 = 1;

pub struct View(Vec<FieldRef>);

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == CLASS_VIEW {
            &self.0
        } else {
            &[]
        }
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

/// The method `from(src) and not to(dst)`: the edges out of `src` that do
/// not end in `dst`. The guard is stated over the class's lanes, which hold
/// tile coordinates, so its constants are the coordinates of the two ids in
/// `table`'s current frame.
pub fn method(id: u32, table: &MapTable, src: u32, dst: u32) -> MethodBinding {
    let frame = table.seal();
    let map = table.map(&frame, TILE).unwrap();
    let code = |g: u32| u32::from(map.local(u64::from(g)).unwrap());
    let guard = SetExpr::And(vec![
        SetExpr::eq_u32(FIELD_X, code(src)),
        SetExpr::ne_u32(FIELD_Y, code(dst)),
    ]);
    let reads = WideFieldMask::from_positions(&[FIELD_X as u8, FIELD_Y as u8]);
    MethodBinding {
        id: MethodId(id),
        sig_rev: SigRev(1),
        name: format!("from {src} and not to {dst}"),
        on: reads.clone(),
        when: Some(Guard::new(guard, Logic::TwoValued)),
        reads,
        effects: Vec::new(),
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
    let view = View(vec![FieldRef::new(Y, "y"), FieldRef::new(X, "x")]);
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: REV,
                field_count: 2,
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
    Admit(Refusal),
    Lane(LaneRefusal),
    NoGuard,
    /// The guard reduced to a constant, so there is no program to run.
    Constant,
    /// A field the guard names was not lent.
    NotLent(u16),
    Lower(String),
}

/// A method made ready once per binding.
pub struct Prepared<'s> {
    lanes: Vec<LaneRef<'s>>,
    masks: Vec<&'s [u64]>,
    rows: usize,
    program: Program,
    scratch: Scratch<'static>,
}

impl<'s> Prepared<'s> {
    pub fn prepare(
        w: &World,
        source: &'s CubeLanes<'_>,
        table: &MapTable,
        r: &MethodRef,
    ) -> Result<Prepared<'s>, MethodRefusal> {
        let bound = w.store.admit(r).map_err(MethodRefusal::Admit)?;
        let guard = bound.when.as_ref().ok_or(MethodRefusal::NoGuard)?;
        let set = match lower_set(&guard.expr, guard.logic) {
            Ok(SetLowered::Filter(set)) => set,
            Ok(_) => return Err(MethodRefusal::Constant),
            Err(e) => return Err(MethodRefusal::Lower(format!("{e:?}"))),
        };
        let frame = table.seal();
        let binding = w
            .store
            .bind_lanes(
                &w.view,
                table.frames(),
                source,
                LaneRequest {
                    route: w.route,
                    rev: REV,
                    fields: &bound.reads,
                    frame: &frame,
                    scope: READER,
                    selection: source.space(),
                },
            )
            .map_err(MethodRefusal::Lane)?;
        // Column `i` of the filter reads the lane of field `set.lanes()[i]`.
        let mut by_field: Vec<Option<LaneRef<'s>>> = vec![None; set.lanes().len()];
        let mut live: Option<&'s [u64]> = None;
        for lent in binding
            .borrow_all_present(&w.store, &w.view, table.frames(), source)
            .map_err(MethodRefusal::Lane)?
        {
            let (field, lane, plane) = lent.map_err(MethodRefusal::Lane)?;
            if let Some(at) = set.lanes().iter().position(|&l| l == u16::from(field)) {
                by_field[at] = Some(lane);
            }
            live = Some(plane);
        }
        let mut lanes = Vec::with_capacity(by_field.len());
        for (at, lane) in by_field.into_iter().enumerate() {
            lanes.push(lane.ok_or(MethodRefusal::NotLent(set.lanes()[at]))?);
        }
        let live = live.ok_or(MethodRefusal::NotLent(FIELD_X))?;
        let rows = match lanes[0] {
            LaneRef::U32(v) => v.len(),
            _ => return Err(MethodRefusal::Lower("a relation lane is u32".into())),
        };
        let filter = set
            .two_valued(set.lanes())
            .map_err(|e| MethodRefusal::Lower(format!("{e:?}")))?;
        let program = lower(&Query {
            filter: Filter::and([Filter::plane(Mask(0)), filter]),
            agg: Agg::Count,
        })
        .map_err(|e| MethodRefusal::Lower(format!("{e:?}")))?;
        let scratch = Scratch::for_program(&program, rows)
            .map_err(|e| MethodRefusal::Lower(format!("{e:?}")))?;
        Ok(Prepared {
            lanes,
            masks: vec![live],
            rows,
            program,
            scratch,
        })
    }

    /// How many live rows of the relation satisfy the guard.
    pub fn run(&mut self) -> u64 {
        let planes = Planes {
            n_rows: self.rows,
            masks: &self.masks,
            lanes: &self.lanes,
        };
        match execute_into(
            &self.program,
            &planes,
            &Foreign::NONE,
            &mut self.scratch,
            Out::None,
        ) {
            Ok(Value::Count(n)) => n as u64,
            other => panic!("a count terminal: {other:?}"),
        }
    }
}

/// The same count by walking the edges; shares no code with the cube.
pub fn oracle(edges: &[(u32, u32)], src: u32, dst: u32) -> u64 {
    edges.iter().filter(|&&(a, b)| a == src && b != dst).count() as u64
}
