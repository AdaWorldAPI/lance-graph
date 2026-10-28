//! D-DRCA-2 stage costs: `DR256::WHERE` × DR#3, each stage of the global
//! selection → local projection → fold path measured on its own, then the
//! cost of reusing one projected selection across N compatible reads.
//!
//! `cargo run --release --example dr256_where_stages`
//!
//! Lab only. Reported, not gated. Every number is the median of 21 timed
//! runs. Stages that are too small to time alone (admission, a fold, one
//! rotated read) are batched inside one call and reported per operation.
//!
//! The fixture (`world`) is built outside every timed region. Cold
//! preparation (encoding 65,536 rows into planes and lanes, the `LocalMap`,
//! the frame stores) is timed separately and is a once-per-frame cost; it is
//! reported, not part of any warm row.
//!
//! The reuse table compares three strategies for N compatible reads of one
//! selection: re-evaluating everything each time (fresh), evaluating once
//! and re-admitting (warm), and re-evaluating after the global owner moved
//! its source revision before every read (rebind-after-invalidation). The
//! answer of every strategy is asserted equal to an oracle count taken from
//! the world's facts.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::time::Instant;

use dr_math_binding_probe::hhtl16::{LocalMap, address, cell_rotated};
use dr_math_binding_probe::relcomp::Rows;
use dr_math_binding_probe::where_dr3::{
    Criteria, GLOBAL, GLOBAL_WORDS, LocalSelection, ReuseRefusal, Under, project, selected_column,
    selected_count,
};
use dr_math_binding_probe::{
    Axis, AxisKind, BoundDr64, Focus, Frame, FrameRevisions, FrameStore, Guards, MethodSignature,
    Operand, Refusal, Scope, bind_dr64,
};
use lance_graph_mask_risc::{
    Foreign, LaneRef, Out, Planes, Program, Scratch, execute_into, words_for,
};
use lance_graph_quack::bind::{Binder, BoundField, FieldKind, TableId, table};
use lance_graph_quack::{Agg, Col, Filter, Mask, Query, lower};

// ── allocation counter ────────────────────────────────────────────────────

struct Counting;
thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}
fn allocs() -> u64 {
    ALLOCS.with(Cell::get)
}
// SAFETY: forwards to `System` unchanged; only counts calls.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static GLOBAL_ALLOC: Counting = Counting;

// ── timing ────────────────────────────────────────────────────────────────

const RUNS: usize = 21;

/// Median ns per op of `f` (which performs `ops` operations per call) and the
/// allocations of one untimed call. Returns the first call's result so the
/// caller can fold it into a printed checksum.
fn measure(name: &str, ops: u64, mut f: impl FnMut() -> u64) -> u64 {
    let a0 = allocs();
    let first = black_box(f());
    let per_call_allocs = allocs() - a0;
    let mut t: Vec<f64> = (0..RUNS)
        .map(|_| {
            let s = Instant::now();
            black_box(f());
            s.elapsed().as_nanos() as f64 / ops as f64
        })
        .collect();
    t.sort_by(f64::total_cmp);
    println!(
        "{name:<44} {ops:>6} ops  {:>12.1} ns/op  {per_call_allocs:>5} allocs/call",
        t[RUNS / 2]
    );
    first
}

/// Median ns of one call of `f`.
fn median_ns(mut f: impl FnMut() -> u64, sink: &mut u64) -> f64 {
    *sink ^= black_box(f());
    let mut t: Vec<f64> = (0..RUNS)
        .map(|_| {
            let s = Instant::now();
            *sink ^= black_box(f());
            s.elapsed().as_nanos() as f64
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[RUNS / 2]
}

// ── the source, as plain values ───────────────────────────────────────────

const EMP: u32 = 21;
const DEPT: u32 = 22;
const OTHER: u32 = 9;
const EMP_CLASS: u16 = 21;

#[derive(Clone, Copy, Debug)]
struct Row {
    class: u32,
    sel: Option<i32>,
    live: bool,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn source(rng: &mut Rng) -> Vec<Row> {
    (0..GLOBAL)
        .map(|_| {
            let class = match rng.below(10) {
                0..=4 => EMP,
                5..=7 => DEPT,
                _ => OTHER,
            };
            let sel = (rng.below(7) != 0).then(|| rng.below(3) as i32);
            Row {
                class,
                sel,
                live: rng.below(20) != 0,
            }
        })
        .collect()
}

/// The oracle's three-valued verdict of `class = 'employee' AND sel = want`.
/// `None`: not in the live population (neither TRUE nor FALSE).
fn verdict(r: &Row, want: i32) -> Option<Option<bool>> {
    if !r.live {
        return None;
    }
    Some(if r.class != EMP {
        Some(false)
    } else {
        r.sel.map(|v| v == want)
    })
}

// ── the engine side: resident encoding + Quack ────────────────────────────

/// Planes: 0 = live, 1 = validity of `sel`, 2 = every address. Lanes: 0 =
/// class code (`U32`), 1 = `sel` (`I32`; payload at NULL rows is 1).
struct Resident {
    planes: [Vec<u64>; 3],
    class: Vec<u32>,
    sel: Vec<i32>,
}
const LIVE: Mask = Mask(0);
const VSEL: Mask = Mask(1);
const CLASS: Col = Col(0);
const SEL: Col = Col(1);

fn resident(src: &[Row], codebook: &Codebook) -> Resident {
    let mut planes: [Vec<u64>; 3] = std::array::from_fn(|_| vec![0u64; GLOBAL_WORDS]);
    let (mut class, mut sel) = (vec![0u32; GLOBAL], vec![1i32; GLOBAL]);
    for (g, r) in src.iter().enumerate() {
        let set = |p: &mut Vec<u64>| p[g / 64] |= 1 << (g % 64);
        if r.live {
            set(&mut planes[0]);
        }
        if let Some(v) = r.sel {
            set(&mut planes[1]);
            sel[g] = v;
        }
        set(&mut planes[2]);
        class[g] = codebook.encode(r.class);
    }
    Resident { planes, class, sel }
}

/// The consumer codebook: which `U32` code the class `employee` carries.
#[derive(Clone, Copy)]
struct Codebook {
    shift: u32,
}
impl Codebook {
    fn encode(&self, class: u32) -> u32 {
        class + self.shift
    }
}
impl Binder for Codebook {
    fn table(&self, name: &str) -> Option<TableId> {
        (name == "node").then_some(TableId(0))
    }
    fn live(&self, _t: TableId) -> Mask {
        LIVE
    }
    fn field(&self, _t: TableId, name: &str) -> Option<BoundField> {
        match name {
            "class" => Some(BoundField {
                col: CLASS,
                kind: FieldKind::Code,
                validity: None,
            }),
            "sel" => Some(BoundField {
                col: SEL,
                kind: FieldKind::I32,
                validity: Some(VSEL),
            }),
            _ => None,
        }
    }
    fn code(&self, _t: TableId, _f: Col, literal: &str) -> Option<u32> {
        (literal == "employee").then_some(self.encode(EMP))
    }
}

// ── frames, facts and registers ───────────────────────────────────────────

// Kept field for field identical to the test's fixture, so `world(SEED)`
// draws the same random sequence; the example reads only some fields.
#[allow(dead_code)]
struct World {
    src: Vec<Row>,
    emp_ids: Vec<u64>,
    dept_ids: Vec<u64>,
    skills: Vec<u64>,
    sites: Vec<u64>,
    has_skill: Vec<(u64, u64)>,
    certified: Vec<(u64, u64, Option<bool>)>,
    works_in: Vec<(u64, u64)>,
    located_at: Vec<(u64, u64)>,
}

/// 64 live employees and 64 live departments in shuffled source order, with
/// TRUE, FALSE and UNKNOWN employees all present.
fn world(seed: u64) -> World {
    let mut rng = Rng(seed);
    let src = source(&mut rng);
    let pick = |rng: &mut Rng, class: u32| -> Vec<u64> {
        let mut out = Vec::new();
        while out.len() < 64 {
            let g = rng.below(GLOBAL as u64);
            let r = src[g as usize];
            if r.live && r.class == class && !out.contains(&g) {
                out.push(g);
            }
        }
        out
    };
    let emp_ids = pick(&mut rng, EMP);
    let dept_ids = pick(&mut rng, DEPT);
    let skills: Vec<u64> = (0..40).map(|i| 900_000 + i).collect();
    let sites: Vec<u64> = (0..16).map(|i| 800_000 + i).collect();
    let (mut has_skill, mut certified, mut works_in, mut located_at) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for &e in &emp_ids {
        for _ in 0..rng.below(5) {
            let s = skills[rng.below(skills.len() as u64) as usize];
            if !has_skill.contains(&(e, s)) {
                has_skill.push((e, s));
                let c = match rng.below(3) {
                    0 => Some(true),
                    1 => Some(false),
                    _ => None,
                };
                certified.push((e, s, c));
            }
        }
        works_in.push((e, dept_ids[rng.below(64) as usize]));
    }
    for &d in &dept_ids {
        for _ in 0..1 + rng.below(2) {
            let t = sites[rng.below(16) as usize];
            if !located_at.contains(&(d, t)) {
                located_at.push((d, t));
            }
        }
    }
    World {
        src,
        emp_ids,
        dept_ids,
        skills,
        sites,
        has_skill,
        certified,
        works_in,
        located_at,
    }
}

fn pos(ids: &[u64], g: u64) -> u8 {
    ids.iter().position(|&x| x == g).unwrap() as u8
}

/// Registers are built from the oracle's positions, not through the map.
fn plane(facts: &[(u64, u64)], rows: &[u64], cols: &[u64]) -> Rows {
    let mut r = [0u64; 64];
    for &(a, b) in facts {
        r[pos(rows, a) as usize] |= 1 << pos(cols, b);
    }
    r
}

const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 7 };
const CRITERIA: Criteria = Criteria { id: 0xC1 };

fn axis(class: u16, kind: AxisKind) -> Axis {
    Axis { class, kind }
}
fn emp_id() -> Axis {
    axis(EMP_CLASS, AxisKind::Identity)
}
fn sig(ord: u16, row: Axis, col: Axis) -> MethodSignature {
    MethodSignature::presence(row.class, ord, row, col)
}
fn has_skill_sig() -> MethodSignature {
    sig(1, emp_id(), axis(EMP_CLASS, AxisKind::Property))
}

/// The global source's owner. `relation_map` is the codebook revision.
fn global_store(cb: &Codebook) -> FrameStore {
    FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: u64::from(cb.shift),
        coordinates: 1,
    })
}
/// The DR frame's owner, sealed under one map's coordinate assignment.
fn local_store(map: &LocalMap) -> FrameStore {
    FrameStore::new(FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: map.digest(),
    })
}
fn guards(st: &FrameStore, focus: Focus, scope: Scope) -> Guards {
    Guards {
        frame: st.seal(),
        focus,
        scope,
    }
}
fn bind<'a>(
    st: &FrameStore,
    g: &Guards,
    m: MethodSignature,
    rows: &'a Rows,
    validity: Option<&'a Rows>,
) -> BoundDr64<'a> {
    bind_dr64(st, m, m, g, (g.focus, g.scope), Operand { rows, validity }).unwrap()
}

const SEED: u64 = 0xD0CA_0002;

/// The fold budget for one batch (workspace rule: ns per op, µs per batch,
/// never milliseconds).
const GATE_NS: f64 = 1_000_000.0;
fn gated(ns: f64) -> String {
    if ns > GATE_NS {
        "> 1,000 (FAIL)".to_string()
    } else {
        format!("{:.1}", ns / 1_000.0)
    }
}
const WANT: i32 = 1;

// ── WHERE: lowered once, executed many times ──────────────────────────────

/// One lowered `Keep` program and its scratch, built once.
struct Kept {
    prog: Program,
    scratch: Scratch<'static>,
}
impl Kept {
    fn new(filter: Filter) -> Kept {
        let q = Query {
            filter,
            agg: Agg::Rows,
        };
        let prog = lower(&q).expect("lowers");
        let scratch = Scratch::for_program(&prog, GLOBAL).expect("carves");
        Kept { prog, scratch }
    }
    /// Execute into the caller's preallocated output; allocation-free.
    fn run(&mut self, inp: &Inputs<'_>, out: &mut [u64]) {
        let pl = Planes {
            n_rows: GLOBAL,
            masks: &inp.masks,
            lanes: &inp.lanes,
        };
        execute_into(
            &self.prog,
            &pl,
            &Foreign::NONE,
            &mut self.scratch,
            Out::Mask(out),
        )
        .expect("runs");
    }
}

/// The borrowed resident inputs of the engine.
struct Inputs<'a> {
    masks: Vec<&'a [u64]>,
    lanes: [LaneRef<'a>; 2],
}

/// The one WHERE `Keep` program of `class = 'employee' AND sel = WANT` and its
/// reusable output buffer.
struct Eng {
    kt: Kept,
    ot: Vec<u64>,
}
impl Eng {
    fn where_true(&mut self, inp: &Inputs<'_>) {
        self.kt.run(inp, &mut self.ot);
    }
}

/// The WHERE filter as Quack's own bind makes it (three-valued over `sel`).
fn filter(cb: &Codebook) -> Filter {
    table("node")
        .where_eq("class", "employee")
        .where_eq("sel", WANT)
        .bind(cb)
        .expect("binds")
        .filter
}

fn proj(
    map: &LocalMap,
    eng: &Eng,
    gst: &FrameStore,
    seal: &Frame,
    lst: &FrameStore,
    g: &Guards,
) -> LocalSelection {
    project(
        map,
        &eng.ot,
        Under {
            global_store: gst,
            global: seal,
            local_store: lst,
            guards: g,
            axis: emp_id(),
            method: has_skill_sig(),
            criteria: CRITERIA,
        },
    )
    .unwrap()
}

// ── the three reuse strategies ────────────────────────────────────────────

/// One full read: WHERE, project, admit, count.
#[allow(clippy::too_many_arguments)]
fn fresh_read(
    eng: &mut Eng,
    inp: &Inputs<'_>,
    map: &LocalMap,
    gst: &FrameStore,
    seal: &Frame,
    lst: &FrameStore,
    g: &Guards,
    b: &BoundDr64<'_>,
) -> u64 {
    eng.where_true(inp);
    let sel = proj(map, eng, gst, seal, lst, g);
    let a = sel.admit(gst, lst, b, CRITERIA).unwrap();
    u64::from(selected_count(b.rows, &a).unwrap())
}

#[allow(clippy::too_many_arguments)]
fn total_fresh(
    n: usize,
    eng: &mut Eng,
    inp: &Inputs<'_>,
    map: &LocalMap,
    gst: &FrameStore,
    seal: &Frame,
    lst: &FrameStore,
    g: &Guards,
    b: &BoundDr64<'_>,
) -> u64 {
    (0..n)
        .map(|_| fresh_read(eng, inp, map, gst, seal, lst, g, b))
        .sum()
}

#[allow(clippy::too_many_arguments)]
fn total_warm(
    n: usize,
    eng: &mut Eng,
    inp: &Inputs<'_>,
    map: &LocalMap,
    gst: &FrameStore,
    seal: &Frame,
    lst: &FrameStore,
    g: &Guards,
    b: &BoundDr64<'_>,
) -> u64 {
    eng.where_true(inp);
    let sel = proj(map, eng, gst, seal, lst, g);
    let mut acc = 0u64;
    for _ in 0..n {
        let a = sel.admit(gst, lst, b, CRITERIA).unwrap();
        acc += u64::from(selected_count(b.rows, black_box(&a)).unwrap());
    }
    acc
}

/// Before every read the global owner advances its source revision (same
/// data). The previous selection must be refused; WHERE, projection and
/// admission are then redone under the new seal.
#[allow(clippy::too_many_arguments)]
fn total_rebind(
    n: usize,
    eng: &mut Eng,
    inp: &Inputs<'_>,
    map: &LocalMap,
    gst: &mut FrameStore,
    lst: &FrameStore,
    g: &Guards,
    b: &BoundDr64<'_>,
    prev: &mut LocalSelection,
) -> u64 {
    let mut acc = 0u64;
    for _ in 0..n {
        let cur = gst.revisions();
        gst.advance(FrameRevisions {
            source: cur.source + 1,
            ..cur
        });
        assert_eq!(
            prev.admit(gst, lst, b, CRITERIA),
            Err(ReuseRefusal::Global(Refusal::SourceChanged)),
            "the old selection must be refused after the source moved"
        );
        let seal = gst.seal();
        eng.where_true(inp);
        let sel = proj(map, eng, gst, &seal, lst, g);
        let a = sel.admit(gst, lst, b, CRITERIA).unwrap();
        acc += u64::from(selected_count(b.rows, &a).unwrap());
        *prev = sel;
    }
    acc
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|m| m.trim().to_string())
        })
        .unwrap_or_else(|| "unknown (/proc/cpuinfo unreadable)".to_string())
}

fn main() {
    println!(
        "cpu: {} | avx512f: {} | runs per row: {RUNS}",
        cpu_model(),
        cfg!(target_feature = "avx512f")
    );
    println!();

    let mut sink = 0u64;
    let cb = Codebook { shift: 0 };
    let w = world(SEED);
    let res = resident(&w.src, &cb);

    // ── cold preparation (once per frame; not part of any warm row) ──────
    println!("== cold preparation (once per frame; reported, not gated) ==");
    sink ^= measure("resident: encode 65,536 rows", 1, || {
        let r = resident(black_box(&w.src), black_box(&cb));
        r.planes[0][0] ^ u64::from(r.class[1])
    });
    sink ^= measure("LocalMap::prepare (64 ids)", 1, || {
        LocalMap::prepare(black_box(&w.emp_ids)).unwrap().len() as u64
    });
    let emp = LocalMap::prepare(&w.emp_ids).unwrap();
    sink ^= measure("FrameStore::new + seal (local)", 1, || {
        let st = local_store(black_box(&emp));
        guards(&st, FOCUS, SCOPE).frame.revisions().coordinates
    });

    // ── bind + lower ─────────────────────────────────────────────────────
    println!();
    println!("== bind + lower ==");
    sink ^= measure("quack: table().where_eq x2 .bind()", 1, || {
        let bound = table("node")
            .where_eq("class", "employee")
            .where_eq("sel", WANT)
            .bind(black_box(&cb))
            .expect("binds");
        black_box(bound);
        1
    });
    let ftrue = filter(&cb);
    let q_true = Query {
        filter: ftrue.clone(),
        agg: Agg::Rows,
    };
    sink ^= measure("quack: lower(&query)", 1, || {
        black_box(lower(black_box(&q_true)).expect("lowers"));
        1
    });

    // ── engine state: lowered once, scratch and outputs preallocated ─────
    let masks: Vec<&[u64]> = res.planes.iter().map(Vec::as_slice).collect();
    let inp = Inputs {
        masks,
        lanes: [LaneRef::U32(&res.class), LaneRef::I32(&res.sel)],
    };
    let mut eng = Eng {
        kt: Kept::new(ftrue),
        ot: vec![0u64; words_for(GLOBAL)],
    };

    // ── WHERE evaluation ─────────────────────────────────────────────────
    println!();
    println!("== WHERE evaluation (program lowered once; execute_into only) ==");
    sink ^= measure("WHERE Keep (65,536 rows)", 1, || {
        eng.where_true(black_box(&inp));
        eng.ot[0] ^ eng.ot[GLOBAL_WORDS - 1]
    });
    eng.where_true(&inp);

    // ── frames, register, selection ──────────────────────────────────────
    let gst = global_store(&cb);
    let lst = local_store(&emp);
    let seal = gst.seal();
    let g = guards(&lst, FOCUS, SCOPE);
    let r = plane(&w.has_skill, &w.emp_ids, &w.skills);
    let b = bind(&lst, &g, has_skill_sig(), &r, None);

    // ── global -> local, admission, reads ────────────────────────────────
    println!();
    println!("== global -> local, admission, warm reads ==");
    sink ^= measure("project: global->local (64 probes)", 1, || {
        let sel = proj(
            black_box(&emp),
            black_box(&eng),
            black_box(&gst),
            black_box(&seal),
            black_box(&lst),
            black_box(&g),
        );
        u64::from(sel.probes())
    });
    let sel = proj(&emp, &eng, &gst, &seal, &lst, &g);
    const BATCH: u64 = 256;
    sink ^= measure("admit: LocalSelection::admit", BATCH, || {
        let mut acc = 0u64;
        for _ in 0..BATCH {
            let a = black_box(&sel)
                .admit(black_box(&gst), black_box(&lst), black_box(&b), CRITERIA)
                .unwrap();
            acc ^= black_box(a.truth());
        }
        acc
    });
    sink ^= measure("warm read: admit + selected_count", BATCH, || {
        let mut acc = 0u64;
        for _ in 0..BATCH {
            let a = black_box(&sel)
                .admit(black_box(&gst), black_box(&lst), black_box(&b), CRITERIA)
                .unwrap();
            acc += u64::from(selected_count(black_box(b.rows), black_box(&a)).unwrap());
        }
        acc
    });
    let a = sel.admit(&gst, &lst, &b, CRITERIA).unwrap();
    assert_ne!(a.truth(), 0, "the fixture must select some identity");
    let x = a.truth().trailing_zeros() as u8;
    sink ^= measure("rotation: cell_rotated via address(y, x)", 64, || {
        let mut acc = 0u64;
        for y in 0..64u8 {
            let code = address(black_box(y), black_box(x)).unwrap();
            acc += u64::from(cell_rotated(black_box(b.rows), code).unwrap());
        }
        acc
    });
    sink ^= measure("fold: selected_count", BATCH, || {
        let mut acc = 0u64;
        for _ in 0..BATCH {
            acc += u64::from(selected_count(black_box(b.rows), black_box(&a)).unwrap());
        }
        acc
    });
    sink ^= measure("fold: selected_column (one column)", 40, || {
        let mut acc = 0u64;
        for y in 0..40u8 {
            acc ^= selected_column(black_box(b.rows), black_box(&a), black_box(y)).unwrap();
        }
        acc
    });
    sink ^= measure("end-to-end fresh read", 1, || {
        fresh_read(
            &mut eng,
            black_box(&inp),
            black_box(&emp),
            &gst,
            &seal,
            &lst,
            &g,
            &b,
        )
    });

    // ── oracle ───────────────────────────────────────────────────────────
    let oracle = w
        .has_skill
        .iter()
        .filter(|&&(e, _)| verdict(&w.src[e as usize], WANT) == Some(Some(true)))
        .count() as u64;
    assert!(oracle > 0, "the oracle must select some fact");
    assert!(
        (oracle as usize) < w.has_skill.len(),
        "the selection must remove some fact"
    );
    let fresh1 = fresh_read(&mut eng, &inp, &emp, &gst, &seal, &lst, &g, &b);
    let warm1 = total_warm(1, &mut eng, &inp, &emp, &gst, &seal, &lst, &g, &b);
    assert_eq!(fresh1, oracle, "fresh answer equals the oracle");
    assert_eq!(warm1, oracle, "warm answer equals the oracle");

    // ── reuse table ──────────────────────────────────────────────────────
    println!();
    println!("== reuse: total µs for N compatible reads (median of {RUNS}) ==");
    println!("   a total above 1,000 µs is a failed gate: printed as such, no ratio drawn from it");
    println!(
        "{:>3} {:>16} {:>16} {:>16} {:>12} {:>10}",
        "N", "fresh µs", "warm µs", "rebind µs", "fresh/warm", "warm<fresh"
    );
    let mut gst_r = global_store(&cb);
    let seal_r = gst_r.seal();
    let mut prev = proj(&emp, &eng, &gst_r, &seal_r, &lst, &g);
    for &n in &[1usize, 2, 4, 8, 64] {
        let want = oracle * n as u64;
        let t_fresh = median_ns(
            || {
                let v = total_fresh(
                    n,
                    &mut eng,
                    black_box(&inp),
                    &emp,
                    &gst,
                    &seal,
                    &lst,
                    &g,
                    &b,
                );
                assert_eq!(v, want);
                v
            },
            &mut sink,
        );
        let t_warm = median_ns(
            || {
                let v = total_warm(
                    n,
                    &mut eng,
                    black_box(&inp),
                    &emp,
                    &gst,
                    &seal,
                    &lst,
                    &g,
                    &b,
                );
                assert_eq!(v, want);
                v
            },
            &mut sink,
        );
        let t_rebind = median_ns(
            || {
                let v = total_rebind(
                    n,
                    &mut eng,
                    black_box(&inp),
                    &emp,
                    &mut gst_r,
                    &lst,
                    &g,
                    &b,
                    &mut prev,
                );
                assert_eq!(v, want);
                v
            },
            &mut sink,
        );
        let (ratio, cmp) = if t_fresh > GATE_NS || t_warm > GATE_NS {
            ("-".to_string(), "-".to_string())
        } else {
            (
                format!("{:.2}x", t_fresh / t_warm),
                (t_warm < t_fresh).to_string(),
            )
        };
        println!(
            "{n:>3} {:>16} {:>16} {:>16} {ratio:>12} {cmp:>10}",
            gated(t_fresh),
            gated(t_warm),
            gated(t_rebind)
        );
    }

    println!();
    println!("checksum (anti-DCE): {sink:#018x}");
    println!(
        "note: microbenchmarks on a shared machine, reported, not gated. Ratios within one run \
         are the meaningful quantity; absolute ns move with the host. Cold preparation is a \
         once-per-frame cost and is not part of any warm row. The rebind column asserts the old \
         selection is refused (a cheap early-out) inside the timed total."
    );
}
