//! `Arg::Present` on the prepared native path (EXO-296): an F000 effect reads
//! a field's presence plane as a mask operand.
//!
//! The three scopes are zammad-rs's, refused today as `NullableComparison`:
//!
//! - `unfinished`: `finished_at IS NULL AND started_at IS NOT NULL`;
//! - `running`: `unfinished AND dry_run = 0`;
//! - `root`: `parent IS NULL`.
//!
//! Under decision B (EXO-294) an absent value is presence bit 0 with value 0,
//! so `IS NULL` is `IntNot` of the presence plane and no Kleene logic is
//! involved. Each count is compared with a scalar oracle over the same rows,
//! at a row count that is not a multiple of 64, and `run_native` is measured
//! at zero allocations.

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::lane_codec::{CodeSpace, EpiLaneSpec, LaneWidth};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Focus, FrameRevisions, FrameStore, Guards, Scope};
use lance_graph_dt64::{
    AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, MethodBinding, MethodId, MethodRef,
    Refusal, RouteVersion, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mask_risc::{Foreign, LaneRef, Planes};
use lance_graph_r2il::binding::{
    prepare, run, ExecPermit, ExecRefusal, FieldLanes, FieldPresence, Prepared, R2ilLens,
    Resolution,
};
use lance_graph_r2il::binding_native::{
    prepare_native, prepare_native_present, run_native, NativeScratch,
};
use lance_graph_r2il::{FoldDialect, INT_AND, INT_EQUAL, INT_NOT, POP_COUNT};
use ogar_loco::{FnIndex, Vocabulary};
use ogar_r2il::{R2ILVocabulary, CONCEPT_R2IL_FOLD};

use crate::binding_native::allocs;

const CLASS: u32 = 0x7F0D_0001;
const CLASS_VIEW: ClassId = 0x7F0D;
const LENS: u32 = (CONCEPT_R2IL_FOLD as u32) << 16;
const REV: SchemaRev = SchemaRev(1);
const R2IL: R2ilLens = R2ilLens {
    lens: LENS,
    schema_rev: REV,
};

/// Four stored lanes, then computed fields.
const FIELDS: [&str; 10] = [
    "finished_at",
    "started_at",
    "dry_run",
    "parent",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
];
const F_FINISHED: u8 = 0;
const F_STARTED: u8 = 1;
const F_DRY: u8 = 2;
const F_PARENT: u8 = 3;
const LANES: [Option<u16>; 10] = [
    Some(0),
    Some(1),
    Some(2),
    Some(3),
    None,
    None,
    None,
    None,
    None,
    None,
];
/// Presence planes for the three nullable fields; `dry_run` has none.
const PRESENCE: [Option<u16>; 10] = [
    Some(0),
    Some(1),
    None,
    Some(2),
    None,
    None,
    None,
    None,
    None,
    None,
];

const W_NOT: u8 = 0;
const W_AND: u8 = 1;
const W_POP: u8 = 2;
const W_EQ: u8 = 3;
const WORD_OPS: [FnIndex; 4] = [INT_NOT, INT_AND, POP_COUNT, INT_EQUAL];

const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 1 };
const NOW: (Focus, Scope) = (FOCUS, SCOPE);
const POSITION: u16 = 3;
const REVS: FrameRevisions = FrameRevisions {
    source: 1,
    relation_map: 1,
    coordinates: 1,
};

struct Views {
    class: Vec<FieldRef>,
    lens: Vec<FieldRef>,
    /// The class's declared lane layout, if any.
    lanes: Option<EpiLaneSpec>,
}

fn refs(labels: &[&str], ns: &str) -> Vec<FieldRef> {
    labels
        .iter()
        .map(|l| FieldRef::new(format!("lab:{ns}/{l}"), (*l).to_string()))
        .collect()
}

impl ClassView for Views {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        match class {
            CLASS_VIEW => &self.class,
            c if c == CONCEPT_R2IL_FOLD => &self.lens,
            _ => &[],
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Card
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
    fn epi_lanes(&self, class: ClassId) -> Option<EpiLaneSpec> {
        (class == CLASS_VIEW).then_some(self.lanes).flatten()
    }
}

fn step(target: u8, word: u8, args: Vec<Arg>) -> Effect {
    Effect {
        target,
        algebra: AlgebraRef { lens: LENS, word },
        args,
    }
}

fn method(id: u32, effects: Vec<Effect>) -> MethodBinding {
    let reads = WideFieldMask::from_positions(&[F_FINISHED, F_STARTED, F_DRY, F_PARENT]);
    MethodBinding {
        id: MethodId(id),
        sig_rev: SigRev(1),
        name: format!("m{id}"),
        on: reads.clone(),
        when: None,
        reads,
        effects,
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

const UNFINISHED: u32 = 1;
const RUNNING: u32 = 2;
const ROOT: u32 = 3;
const NO_PLANE: u32 = 4; // Present(dry_run): no presence plane bound

/// `f4 = NOT present(finished_at)`, `f5 = f4 AND present(started_at)`.
fn unfinished_mask() -> Vec<Effect> {
    vec![
        step(4, W_NOT, vec![Arg::Present(F_FINISHED)]),
        step(5, W_AND, vec![Arg::Field(4), Arg::Present(F_STARTED)]),
    ]
}

fn methods() -> Vec<MethodBinding> {
    let mut unfinished = unfinished_mask();
    unfinished.push(step(6, W_POP, vec![Arg::Field(5)]));
    let mut running = unfinished_mask();
    running.extend([
        step(7, W_EQ, vec![Arg::Field(F_DRY), Arg::Const(0)]),
        step(8, W_AND, vec![Arg::Field(5), Arg::Field(7)]),
        step(9, W_POP, vec![Arg::Field(8)]),
    ]);
    vec![
        method(UNFINISHED, unfinished),
        method(RUNNING, running),
        method(
            ROOT,
            vec![
                step(4, W_NOT, vec![Arg::Present(F_PARENT)]),
                step(5, W_POP, vec![Arg::Field(4)]),
            ],
        ),
        method(
            NO_PLANE,
            vec![
                step(4, W_NOT, vec![Arg::Present(F_DRY)]),
                step(5, W_POP, vec![Arg::Field(4)]),
            ],
        ),
    ]
}

fn mref(s: &BindStore, id: u32) -> MethodRef {
    s.method_ref(CLASS, MethodId(id)).unwrap()
}

fn permit(s: &BindStore, id: u32) -> ExecPermit {
    ExecPermit {
        methods: vec![mref(s, id)],
    }
}

fn schema() -> ClassSchema {
    ClassSchema {
        schema_rev: REV,
        field_count: FIELDS.len(),
        extent: Extent::DR64_CUBE,
    }
}

fn arities() -> Vec<u8> {
    WORD_OPS
        .iter()
        .map(|&f| R2ILVocabulary.stack_arity(f).expect("OGAR gives an arity"))
        .collect()
}

fn views() -> Views {
    Views {
        class: refs(&FIELDS, "class"),
        lens: refs(
            &WORD_OPS
                .iter()
                .map(|&f| R2ILVocabulary.name(f).expect("OGAR names the opcode"))
                .collect::<Vec<_>>(),
            "r2il",
        ),
        lanes: None,
    }
}

/// Rows with values and presence; absent is presence 0 with value 0.
struct Data {
    finished: Vec<i32>,
    started: Vec<i32>,
    dry: Vec<i32>,
    parent: Vec<u32>,
    present: [Vec<u64>; 3],
}

impl Data {
    fn new(n: usize) -> Self {
        let finished_p: Vec<bool> = (0..n).map(|i| i % 3 != 0).collect();
        let started_p: Vec<bool> = (0..n).map(|i| i % 5 != 1).collect();
        let parent_p: Vec<bool> = (0..n).map(|i| i % 7 != 2).collect();
        let value = |p: &[bool], i: usize, v: i32| if p[i] { v } else { 0 };
        let bits = |p: &[bool]| {
            let mut w = vec![0u64; n.div_ceil(64)];
            for (i, _) in p.iter().enumerate().filter(|(_, &b)| b) {
                w[i / 64] |= 1 << (i % 64);
            }
            w
        };
        Self {
            finished: (0..n)
                .map(|i| value(&finished_p, i, 1000 + i as i32))
                .collect(),
            started: (0..n)
                .map(|i| value(&started_p, i, 10 + i as i32))
                .collect(),
            dry: (0..n).map(|i| i32::from(i % 4 == 0)).collect(),
            parent: (0..n)
                .map(|i| if parent_p[i] { 1 + (i as u32 % 9) } else { 0 })
                .collect(),
            present: [bits(&finished_p), bits(&started_p), bits(&parent_p)],
        }
    }
    fn n(&self) -> usize {
        self.dry.len()
    }
    fn present(&self, plane: usize, i: usize) -> bool {
        self.present[plane][i / 64] >> (i % 64) & 1 == 1
    }
    fn unfinished(&self, i: usize) -> bool {
        !self.present(0, i) && self.present(1, i)
    }
    fn count(&self, id: u32) -> i64 {
        (0..self.n())
            .filter(|&i| match id {
                UNFINISHED => self.unfinished(i),
                RUNNING => self.unfinished(i) && self.dry[i] == 0,
                ROOT => !self.present(2, i),
                _ => unreachable!(),
            })
            .count() as i64
    }
}

struct Fixture {
    views: Views,
    store: BindStore,
    version: RouteVersion,
    frames: FrameStore,
}

impl Fixture {
    fn new() -> Self {
        Self::with_lanes(None)
    }

    /// A fixture whose class declares `lanes` as its lane layout.
    fn with_lanes(lanes: Option<EpiLaneSpec>) -> Self {
        let mut store = BindStore::new();
        store
            .declare_algebra(LENS, REV, Extent::DR64_CUBE, arities())
            .unwrap();
        let views = Views { lanes, ..views() };
        store.register(&views, CLASS, schema(), methods()).unwrap();
        let version = store.route(POSITION, CLASS).unwrap();
        Self {
            views,
            store,
            version,
            frames: FrameStore::new(REVS),
        }
    }

    fn prepare(&self, id: u32) -> Prepared {
        self.try_prepare(id).expect("the fixture's methods prepare")
    }

    fn try_prepare(&self, id: u32) -> Result<Prepared, ExecRefusal> {
        let r = mref(&self.store, id);
        let g = Guards {
            frame: self.frames.seal(),
            focus: FOCUS,
            scope: SCOPE,
        };
        prepare(
            &self.store,
            &self.views,
            R2IL,
            &Resolution {
                method: &r,
                schema_rev: REV,
                route: (self.version, POSITION),
                guards: &g,
            },
            &self.frames,
            NOW,
        )
    }
}

/// Run `id` natively over `data` and return its one scalar and the
/// allocations of the hot run.
fn native_count(f: &Fixture, id: u32, data: &Data) -> Result<(i64, usize), ExecRefusal> {
    let lanes = [
        LaneRef::I32(&data.finished),
        LaneRef::I32(&data.started),
        LaneRef::I32(&data.dry),
        LaneRef::U32(&data.parent),
    ];
    let masks: [&[u64]; 3] = [&data.present[0], &data.present[1], &data.present[2]];
    let planes = Planes {
        n_rows: data.n(),
        masks: &masks,
        lanes: &lanes,
    };
    let plan = f.prepare(id);
    let np = prepare_native_present(
        &plan,
        &f.store,
        &f.frames,
        NOW,
        FieldLanes { lanes: &LANES },
        FieldPresence { planes: &PRESENCE },
        &planes,
    )?;
    let mut scratch = NativeScratch::for_plan(&np);
    let p = permit(&f.store, id);
    let before = allocs();
    let r = run_native(&np, &f.store, &f.frames, NOW, &p, &planes, &mut scratch)?;
    let hot = allocs() - before;
    assert_eq!(r.values().len(), 1, "one scalar output for {id}");
    Ok((r.values()[0].1, hot))
}

// Fails if a presence plane is read as anything but its own bits (inverted,
// another plane, the value lane), if the tail past `n` leaks into a count, or
// if the hot run allocates.
#[test]
fn nullable_scopes_count_through_the_presence_plane() {
    let f = Fixture::new();
    for n in [1, 63, 64, 65, 1_000, 16_385] {
        let data = Data::new(n);
        for id in [UNFINISHED, RUNNING, ROOT] {
            let (got, hot) = native_count(&f, id, &data).expect("runs natively");
            assert_eq!(got, data.count(id), "scope {id} at n = {n}");
            assert_eq!(hot, 0, "run_native allocated for scope {id}");
        }
    }
    // Anti-vacuity: the three scopes select different, non-trivial sets.
    let data = Data::new(1_000);
    let counts: Vec<i64> = [UNFINISHED, RUNNING, ROOT]
        .iter()
        .map(|&id| data.count(id))
        .collect();
    assert!(counts.iter().all(|&c| c > 0 && c < 1_000), "{counts:?}");
    assert!(
        counts[1] < counts[0],
        "dry_run filters some unfinished rows"
    );
}

#[test]
fn a_field_without_a_presence_plane_is_refused_by_name() {
    let f = Fixture::new();
    assert_eq!(
        native_count(&f, NO_PLANE, &Data::new(100)).unwrap_err(),
        ExecRefusal::PresenceUnbound {
            effect: 0,
            field: F_DRY
        }
    );
}

#[test]
fn a_presence_index_outside_the_planes_is_refused() {
    let f = Fixture::new();
    let data = Data::new(100);
    let lanes = [
        LaneRef::I32(&data.finished),
        LaneRef::I32(&data.started),
        LaneRef::I32(&data.dry),
        LaneRef::U32(&data.parent),
    ];
    let masks: [&[u64]; 1] = [&data.present[0]];
    let planes = Planes {
        n_rows: data.n(),
        masks: &masks,
        lanes: &lanes,
    };
    let err = prepare_native_present(
        &f.prepare(ROOT),
        &f.store,
        &f.frames,
        NOW,
        FieldLanes { lanes: &LANES },
        FieldPresence { planes: &PRESENCE },
        &planes,
    )
    .unwrap_err();
    assert_eq!(
        err,
        ExecRefusal::PresenceUnusable {
            effect: 0,
            field: F_PARENT,
            plane: 2
        }
    );
    // `prepare_native` takes no presence binding, so the same plan is refused
    // there too, not read as a value.
    let err = prepare_native(
        &f.prepare(ROOT),
        &f.store,
        &f.frames,
        NOW,
        FieldLanes { lanes: &LANES },
        &planes,
    )
    .unwrap_err();
    assert_eq!(
        err,
        ExecRefusal::PresenceUnbound {
            effect: 0,
            field: F_PARENT
        }
    );
}

#[test]
fn the_interpretive_run_refuses_a_presence_operand() {
    let f = Fixture::new();
    let data = Data::new(100);
    let lanes = [
        LaneRef::I32(&data.finished),
        LaneRef::I32(&data.started),
        LaneRef::I32(&data.dry),
        LaneRef::U32(&data.parent),
    ];
    let planes = Planes {
        n_rows: data.n(),
        masks: &[],
        lanes: &lanes,
    };
    let foreign = Foreign::NONE;
    let mut d = FoldDialect::new(&planes, &foreign);
    let err = run(
        &f.prepare(ROOT),
        &f.store,
        &f.frames,
        NOW,
        &permit(&f.store, ROOT),
        FieldLanes { lanes: &LANES },
        &mut d,
    )
    .unwrap_err();
    assert_eq!(
        err,
        ExecRefusal::PresenceUnbound {
            effect: 0,
            field: F_PARENT
        }
    );
}

/// Registration refuses a presence read of a field the method does not
/// declare it reads, and of a field an earlier step computed.
#[test]
fn registration_refuses_presence_of_an_undeclared_or_computed_field() {
    for (effects, position) in [
        // f4 is not a declared read.
        (vec![step(5, W_NOT, vec![Arg::Present(4)])], 4u8),
        // f4 is written first; its presence is still not a stored plane.
        (
            vec![
                step(4, W_NOT, vec![Arg::Present(F_PARENT)]),
                step(5, W_NOT, vec![Arg::Present(4)]),
            ],
            4u8,
        ),
    ] {
        let mut store = BindStore::new();
        store
            .declare_algebra(LENS, REV, Extent::DR64_CUBE, arities())
            .unwrap();
        let effect = effects.len() - 1;
        let err = store
            .register(&views(), CLASS, schema(), vec![method(99, effects)])
            .unwrap_err();
        assert_eq!(
            err,
            Refusal::UndeclaredRead {
                class: CLASS,
                method: MethodId(99),
                effect,
                position,
            }
        );
    }
}

/// The binding digest tells `Present(p)` from `Field(p)`: the same method
/// id in two stores, differing only in that one argument.
#[test]
fn present_and_field_are_different_bindings() {
    let digest = |arg: Arg| {
        let mut store = BindStore::new();
        store
            .declare_algebra(LENS, REV, Extent::DR64_CUBE, arities())
            .unwrap();
        let m = method(1, vec![step(4, W_NOT, vec![arg])]);
        store.register(&views(), CLASS, schema(), vec![m]).unwrap();
        mref(&store, 1).stamp.binding()
    };
    let present = digest(Arg::Present(F_PARENT));
    assert_eq!(present, digest(Arg::Present(F_PARENT)), "deterministic");
    assert_ne!(present, digest(Arg::Field(F_PARENT)));
}

/// A class layout of `width` lanes (the offsets are not read here).
fn layout(width: LaneWidth) -> EpiLaneSpec {
    EpiLaneSpec {
        width,
        space: CodeSpace::Sym,
        values: 32,
        presence: 96,
        lanes: 4,
    }
}

/// Run `ROOT` (`parent IS NULL`) over four rows whose `parent` presence plane
/// is `p`, the rows' values never read.
fn root_over(f: &Fixture, p: u64) -> i64 {
    let n = 4;
    let p = vec![p];
    let other = vec![0u64; 1];
    let i32s = vec![0i32; n];
    let u32s = vec![0u32; n];
    let lanes = [
        LaneRef::I32(&i32s),
        LaneRef::I32(&i32s),
        LaneRef::I32(&i32s),
        LaneRef::U32(&u32s),
    ];
    let masks: [&[u64]; 3] = [&other, &other, &p];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };
    let plan = f.prepare(ROOT);
    let np = prepare_native_present(
        &plan,
        &f.store,
        &f.frames,
        NOW,
        FieldLanes { lanes: &LANES },
        FieldPresence { planes: &PRESENCE },
        &planes,
    )
    .expect("the binding is admitted");
    let mut scratch = NativeScratch::for_plan(&np);
    let permit = permit(&f.store, ROOT);
    let r = run_native(
        &np,
        &f.store,
        &f.frames,
        NOW,
        &permit,
        &planes,
        &mut scratch,
    )
    .expect("runs");
    r.values()[0].1
}

/// Codec9 (`ndarray::hpc::codec9`, `lane_codec`): a 9-bit lane's presence bit
/// is also its sign. Rows `+7, -7, +0, -0` have P = `1, 0, 1, 0` and
/// magnitude `7, 7, 0, 0`; only `-0` is absent. Read from the plane alone,
/// `parent IS NULL` would count `-7` too: 2 rows instead of 1. A class that
/// declares 9-bit lanes is refused at preparation, by name.
#[test]
fn a_presence_operand_over_9_bit_lanes_is_refused() {
    let f = Fixture::with_lanes(Some(layout(LaneWidth::Bits9)));
    assert_eq!(
        f.try_prepare(ROOT).unwrap_err(),
        ExecRefusal::PresenceIsSign {
            effect: 0,
            field: F_PARENT
        }
    );
}

/// The twin: 8-bit lanes keep `P = 0 => value 0`, so the plane alone is
/// absent and the same four planes count as the contract says.
#[test]
fn a_presence_operand_over_8_bit_lanes_counts_from_the_plane() {
    let f = Fixture::with_lanes(Some(layout(LaneWidth::Bits8)));
    // Rows: present, absent, present, absent.
    assert_eq!(root_over(&f, 0b0101), 2);
    assert_eq!(root_over(&f, 0b1111), 0);
}

/// An undeclared layout is not refused: the binding cannot tell 8-bit from
/// 9-bit lanes, and reads the plane as the 8-bit rule says. The Codec9 rows
/// `+7, -7, +0, -0` then count 2, not the 1 their own contract gives. This
/// pins that reachable misread; declaring the layout is what removes it.
#[test]
fn an_undeclared_layout_reads_the_plane_as_8_bit() {
    let f = Fixture::new();
    assert_eq!(root_over(&f, 0b0101), 2);
}
