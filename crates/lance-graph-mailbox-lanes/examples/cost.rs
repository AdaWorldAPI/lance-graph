//! D-VAS-F: what the chain costs on a real `MailboxSoA<64>` (40 populated
//! rows), measured, with allocations counted.
//!
//! Run in release: `cargo run --release -p lance-graph-mailbox-lanes --example cost`.
//! A debug build measures the wrong thing. Times are wall-clock medians of
//! batches on whatever machine runs this; they are a size class, not a
//! benchmark result to compare across machines.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::cognitive_shader::MetaWord;
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Focus, Scope};
use lance_graph_dt64::{
    view_class, AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, Guard, LaneRequest,
    MethodBinding, MethodId, RouteAt, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mailbox_lanes::act::{act, act_over, apply, cascade, ActContext};
use lance_graph_mailbox_lanes::when::{when, WhenRequest};
use lance_graph_mailbox_lanes::{Column, Keep, MailboxLanes};
use lance_graph_mask_risc::{execute_sel, LaneRef, Planes, SelFold};
use lance_graph_r2il::binding::{ExecPermit, R2ilLens};
use lance_graph_r2il::polyfill::{AliasLanguage, POLYFILL, POLYFILL_LENS};

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards to the system allocator unchanged; the counters are
// atomics and touch no allocator state.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: same layout, passed through.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System.alloc` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const N: usize = 64;
type Owner = MailboxSoA<N>;
const POPULATED: usize = 40;
const CLASS: u32 = 0x0A1A_0003;
const POSITION: u16 = 11;
const READER: Scope = Scope { id: 3 };
const REV: SchemaRev = SchemaRev(1);
const META: u8 = 0;
const ACTIVE: u8 = 1;
const TEMPORAL: u8 = 2;
const MASK: u8 = 3;
/// Names no mailbox column.
const SCRATCH: u8 = 4;
const TALLY: MethodId = MethodId(1);
/// Read-only: ON active; WHEN `meta == 2`; DO `scratch := COUNT(active == 0)`.
const PROBE: MethodId = MethodId(3);
const MARK: MethodId = MethodId(2);

fn owner() -> Box<Owner> {
    let mut m: Box<Owner> = Box::new(MailboxSoA::new(7, 3, 1.0));
    m.set_populated(POPULATED);
    for r in 0..POPULATED {
        m.set_meta(r, MetaWord((r as u32 * 7 + 3) % 4));
        m.last_active_cycle[r] = (r as u32 * 5 + 1) % 3;
    }
    m.meta[5] = MetaWord(2);
    m
}

struct View {
    class: Vec<FieldRef>,
    lens: Vec<FieldRef>,
}

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == view_class(CLASS) {
            &self.class
        } else {
            &[]
        }
    }
    fn lens_fields(&self, lens: u32) -> Option<&[FieldRef]> {
        (lens == POLYFILL_LENS).then_some(&self.lens[..])
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

fn count_where(field: u8, v: i64, target: u8) -> Vec<Effect> {
    let word = |target, word, args| Effect {
        target,
        algebra: AlgebraRef {
            lens: POLYFILL_LENS,
            word,
        },
        args,
    };
    vec![
        word(MASK, 4, vec![Arg::Field(field), Arg::Const(v)]),
        word(target, 10, vec![Arg::Field(MASK)]),
    ]
}

fn method(id: MethodId, on: u8, when: SetExpr, effects: Vec<Effect>) -> MethodBinding {
    MethodBinding {
        id,
        sig_rev: SigRev(1),
        name: format!("m{}", id.0),
        on: WideFieldMask::from_positions(&[on]),
        when: Some(Guard::new(when, Logic::TwoValued)),
        reads: WideFieldMask::from_positions(&[ACTIVE]),
        effects,
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

fn tally_guard() -> SetExpr {
    SetExpr::And(vec![
        SetExpr::eq_u32(u16::from(META), 2),
        SetExpr::ne_u32(u16::from(ACTIVE), 1),
    ])
}

/// Median ns per call of `f` over `batches` batches of `per` calls, and the
/// allocations and allocated bytes of one call.
fn measure<T>(name: &str, per: u32, mut f: impl FnMut() -> T) -> f64 {
    let (a0, b0) = (
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    );
    black_box(f());
    let allocs = ALLOCS.load(Ordering::Relaxed) - a0;
    let bytes = BYTES.load(Ordering::Relaxed) - b0;
    let mut samples: Vec<f64> = (0..9)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..per {
                black_box(f());
            }
            t.elapsed().as_nanos() as f64 / f64::from(per)
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    let ns = samples[samples.len() / 2];
    println!("| {name} | {ns:.1} | {allocs} | {bytes} |");
    ns
}

fn main() {
    let field = |iri: &str| FieldRef::new(iri, iri);
    let view = View {
        class: vec![
            field(Column::Meta.iri()),
            field(Column::LastActiveCycle.iri()),
            field(Column::Temporal.iri()),
            field("urn:lab:mask"),
            field("urn:lab:scratch"),
        ],
        lens: POLYFILL
            .iter()
            .enumerate()
            .map(|(i, w)| {
                FieldRef::new(format!("urn:polyfill:{i}"), w.name_in(AliasLanguage::R2il))
            })
            .collect(),
    };
    let mut store = BindStore::new();
    store
        .declare_algebra(
            POLYFILL_LENS,
            REV,
            Extent::DR256_SQ_X64,
            POLYFILL.iter().map(|w| w.arity).collect(),
        )
        .unwrap();
    store
        .register(
            &view,
            CLASS,
            ClassSchema {
                schema_rev: REV,
                field_count: 5,
                extent: Extent::DR256_SQ_X64,
            },
            vec![
                method(TALLY, META, tally_guard(), count_where(ACTIVE, 0, TEMPORAL)),
                method(
                    MARK,
                    TEMPORAL,
                    SetExpr::eq_u32(u16::from(META), 1),
                    count_where(ACTIVE, 2, META),
                ),
                method(
                    PROBE,
                    ACTIVE,
                    SetExpr::eq_u32(u16::from(META), 2),
                    count_where(ACTIVE, 0, SCRATCH),
                ),
            ],
        )
        .unwrap();
    let version = store.route(POSITION, CLASS).unwrap();
    let route = RouteAt {
        position: POSITION,
        version,
    };
    let permit = ExecPermit {
        methods: vec![
            store.method_ref(CLASS, TALLY).unwrap(),
            store.method_ref(CLASS, MARK).unwrap(),
            store.method_ref(CLASS, PROBE).unwrap(),
        ],
    };
    let ctx = ActContext {
        route,
        rev: REV,
        scope: READER,
        focus: Focus { revision: 1 },
        lens: R2ilLens {
            lens: POLYFILL_LENS,
            schema_rev: REV,
        },
        permit: &permit,
    };
    let tally = store.method_ref(CLASS, TALLY).unwrap();
    let guard = Guard::new(tally_guard(), Logic::TwoValued);
    let m = owner();

    println!("rows = {POPULATED} of a MailboxSoA<{N}>; median of 9 batches\n");
    println!("| step | ns per call | allocations per call | bytes allocated per call |");
    println!("|---|---|---|---|");

    // Baselines: the owner's arrays, read directly.
    measure(
        "direct: sum 40 `meta` values from the owner's array",
        2_000_000,
        || (0..POPULATED).map(|r| u64::from(m.meta[r].0)).sum::<u64>(),
    );
    measure(
        "scalar interpreter: tally (guard, count, 40 rows)",
        2_000_000,
        || {
            let mut fired = 0u64;
            let mut value = 0u64;
            for r in 0..POPULATED {
                if m.meta[r].0 == 2 && m.last_active_cycle[r] != 1 {
                    fired |= 1 << r;
                    value += u64::from(m.last_active_cycle[r] == 0);
                }
            }
            (fired, value)
        },
    );
    measure(
        "eager: copy 40 rows into structs, then tally",
        1_000_000,
        || {
            let rows: Vec<(u32, u32, u64)> = (0..POPULATED)
                .map(|r| (m.meta[r].0, m.last_active_cycle[r], m.temporal[r]))
                .collect();
            rows.iter()
                .filter(|r| r.0 == 2 && r.1 != 1)
                .filter(|r| r.1 == 0)
                .count()
        },
    );

    // The chain, step by step, over one publication.
    let lanes = MailboxLanes::new(m.published(), READER);
    measure(
        "socket: `MailboxLanes::new` (one publication)",
        200_000,
        || MailboxLanes::new(m.published(), READER).rows(),
    );
    measure("DT64: `admit_routed`", 1_000_000, || {
        store.admit_routed(route, &tally).unwrap().id
    });
    let reads = WideFieldMask::from_positions(&[META]);
    let bind = || {
        let frame = lanes.seal();
        store
            .bind_lanes(
                &view,
                lanes.frame_store(),
                &lanes,
                LaneRequest {
                    route,
                    rev: REV,
                    fields: &reads,
                    frame: &frame,
                    scope: READER,
                    selection: lanes.space(),
                },
            )
            .unwrap()
    };
    measure("DT64: `bind_lanes` over one field", 200_000, bind);
    let binding = bind();
    measure("checked lane borrow + sum 40 values", 500_000, || {
        let LaneRef::U32(v) = binding
            .lane(&store, &view, lanes.frame_store(), &lanes, META)
            .unwrap()
        else {
            unreachable!()
        };
        v.iter().map(|&x| u64::from(x)).sum::<u64>()
    });
    let request = WhenRequest {
        route,
        rev: REV,
        scope: READER,
    };
    measure(
        "WHEN: lower + bind + run (2 fields, 40 rows)",
        50_000,
        || {
            when(&lanes, &store, &view, request, &guard)
                .unwrap()
                .count()
        },
    );
    let acted = measure(
        "act: admission + WHEN + operands + gated DO",
        20_000,
        || act(&lanes, &store, &view, &ctx, &tally).unwrap().rows.len(),
    );
    let intent = act(&lanes, &store, &view, &ctx, &tally).unwrap().intents;
    drop(lanes);

    // D-VAS-R11: the population paths. One socket, selections issued once.
    let probe = store.method_ref(CLASS, PROBE).unwrap();
    measure(
        "socket: open + issue 2 selections + concat (40 rows)",
        200_000,
        || {
            let mut lanes = MailboxLanes::new(m.published(), READER);
            let a = lanes
                .issue_where(Column::LastActiveCycle, Keep::Eq(0))
                .unwrap();
            let b = lanes
                .issue_where(Column::LastActiveCycle, Keep::Ge(0))
                .unwrap();
            lanes.issue_concat(a, b).unwrap()
        },
    );
    let mut lanes = MailboxLanes::new(m.published(), READER);
    let a = lanes
        .issue_where(Column::LastActiveCycle, Keep::Eq(0))
        .unwrap();
    let b = lanes
        .issue_where(Column::LastActiveCycle, Keep::Ge(0))
        .unwrap();
    let sel = lanes.issue_concat(a, b).unwrap();
    let lanes = lanes;
    measure("act: read-only method, whole source", 20_000, || {
        act(&lanes, &store, &view, &ctx, &probe).unwrap().unwritten
    });
    measure(
        "act_over: read-only method, selection's distinct rows",
        20_000,
        || {
            act_over(&lanes, &store, &view, &ctx, &probe, sel)
                .unwrap()
                .unwritten
        },
    );
    drop(lanes);
    // The shipped selection fold. It admits `i32` lanes only, so this lane is
    // SYNTHETIC: 40 `i32` values, not a mailbox column. It is not a
    // measurement of mailbox execution.
    let synthetic: Vec<i32> = (0..POPULATED as i32).map(|r| r % 3).collect();
    let listed: Vec<u32> = (0..POPULATED as u32).chain(0..14).collect();
    let synthetic_lanes = [LaneRef::I32(&synthetic)];
    let planes = Planes {
        n_rows: POPULATED,
        masks: &[],
        lanes: &synthetic_lanes,
    };
    measure(
        "synthetic i32 lane (not a mailbox column): `execute_sel` CountGtI32, 54 listed positions",
        2_000_000,
        || execute_sel(SelFold::CountGtI32 { lane: 0, t: 0 }, &planes, &listed).unwrap(),
    );

    // Write-back and the cascade mutate the owner. One owner is reused and
    // its two written columns are put back before each call; that reset is
    // measured on its own and subtracted. (Building an owner per call costs
    // about 50 us and 398 KB, which would hide everything measured here.)
    let mut o = owner();
    let (meta0, temporal0) = (o.meta, o.temporal);
    let reset = measure(
        "(reset: put two columns back, subtracted below)",
        1_000_000,
        || {
            o.meta = meta0;
            o.temporal = temporal0;
            o.populated()
        },
    );
    let applied = measure("reset + apply one intent", 500_000, || {
        o.meta = meta0;
        o.temporal = temporal0;
        apply(&mut o, &store, &view, &ctx, &intent[0])
            .unwrap()
            .changed
    });
    let whole = measure("reset + cascade (3 rounds, 2 methods)", 10_000, || {
        o.meta = meta0;
        o.temporal = temporal0;
        cascade(&mut o, &store, &view, &ctx, &[Column::Meta], 8)
            .unwrap()
            .rounds
            .len()
    });
    println!();
    assert!(
        applied > reset && whole > reset,
        "the reset must not dominate"
    );
    println!("apply alone: {:.1} ns", applied - reset);
    println!(
        "cascade alone: {:.1} ns ({:.1} ns per round)",
        whole - reset,
        (whole - reset) / 3.0
    );
    println!("act per row: {:.1} ns", acted / POPULATED as f64);
    println!(
        "row bits an intent carries: {} bytes (one bit per row), beside its method reference",
        POPULATED.div_ceil(64) * 8
    );
}
