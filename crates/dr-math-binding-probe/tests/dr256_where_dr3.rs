//! D-DRCA-2: `DR256::WHERE` × DR#3.
//!
//! Global selection (Quack WHERE over 65,536 addresses, mask-risc `Keep`)
//! → global→local projection through a [`LocalMap`] → HHTL16 Cartesian read
//! of a bound DR64 register → fold.
//!
//! # The oracle
//!
//! `Source` holds the rows as plain Rust values: `class: u32`, `sel:
//! Option<i32>` (`None` = NULL), `live: bool`, indexed by global address.
//! The oracle answers every question from `Source`, the fact arrays and the
//! frame's source-order id list. It shares no code with Quack, mask-risc,
//! `hhtl16` or `where_dr3`: local positions are `position()` in the id list,
//! three-valued verdicts are a `match` on `Option`.
//!
//! The engine side encodes the same `Source` into resident lanes and planes
//! (a value lane plus a validity plane per nullable column) and asks Quack.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use dr_math_binding_probe::hhtl16::{self, LocalMap, address, cell, cell_rotated};
use dr_math_binding_probe::relcomp::{Chain, ComposeRefusal, RelExpr, Rows};
use dr_math_binding_probe::where_dr3::{
    self, Criteria, GLOBAL, GLOBAL_WORDS, LocalSelection, ProjectRefusal, ReuseRefusal, Under,
    project, selected_cell, selected_column, selected_column_k3, selected_count,
};
use dr_math_binding_probe::{
    Axis, AxisKind, BoundDr64, Focus, Frame, FrameRevisions, FrameStore, Guards, MethodSignature,
    Operand, Population, Refusal, Scope, Truth, bind_dr64,
};
use lance_graph_mask_risc::{Foreign, LaneRef, Out, Planes, Scratch, execute_into, words_for};
use lance_graph_quack::bind::{Binder, BoundField, FieldKind, TableId, table};
use lance_graph_quack::{Agg, Cmp, Col, Filter, Mask, Query, lower};

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

// ── the source, as plain values ───────────────────────────────────────────

const EMP: u32 = 21;
const DEPT: u32 = 22;
const OTHER: u32 = 9;
const EMP_CLASS: u16 = 21;
const DEPT_CLASS: u16 = 22;

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
/// class code (`U32`), 1 = `sel` (`I32`; payload at NULL rows is 1, so an
/// engine that reads NULL payloads keeps them).
struct Resident {
    planes: [Vec<u64>; 3],
    class: Vec<u32>,
    sel: Vec<i32>,
}
const LIVE: Mask = Mask(0);
const VSEL: Mask = Mask(1);
const ALL: Mask = Mask(2);
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

/// `Keep` of one filter through mask-risc.
fn keep(res: &Resident, filter: Filter) -> Vec<u64> {
    let q = Query {
        filter,
        agg: Agg::Rows,
    };
    let prog = lower(&q).expect("lowers");
    let masks: Vec<&[u64]> = res.planes.iter().map(Vec::as_slice).collect();
    let lanes = [LaneRef::U32(&res.class), LaneRef::I32(&res.sel)];
    let pl = Planes {
        n_rows: GLOBAL,
        masks: &masks,
        lanes: &lanes,
    };
    let mut out = vec![0u64; words_for(GLOBAL)];
    let mut s = Scratch::for_program(&prog, GLOBAL).expect("carves");
    execute_into(&prog, &pl, &Foreign::NONE, &mut s, Out::Mask(&mut out)).expect("runs");
    out
}

/// WHERE `class = 'employee' AND sel = want`: the TRUE rows, through Quack's
/// own `Draft::bind` (three-valued over the nullable `sel`), checked against
/// the same filter built by hand.
fn where_k3(res: &Resident, cb: &Codebook, want: i32) -> Vec<u64> {
    let bound = table("node")
        .where_eq("class", "employee")
        .where_eq("sel", want)
        .bind(cb)
        .expect("binds");
    let raw = Filter::and([
        Filter::Cmp(CLASS, Cmp::EqU32(cb.encode(EMP))),
        Filter::Cmp(SEL, Cmp::EqI32(want)),
    ]);
    let nullable = [(SEL, VSEL)];
    let t_direct = Filter::and([Filter::plane(LIVE), raw]).sql_where(&nullable);
    let t = keep(res, bound.filter);
    assert_eq!(
        t,
        keep(res, t_direct),
        "Draft::bind and the hand-built filter must agree"
    );
    t
}

/// WHERE `NOT (class = 'employee' AND sel = want)`: a second WHERE, evaluated
/// by the same three-valued `sql_where`. `NOT UNKNOWN` stays UNKNOWN, so a
/// NULL row is kept by neither this nor [`where_k3`].
fn where_not(res: &Resident, cb: &Codebook, want: i32) -> Vec<u64> {
    let raw = Filter::and([
        Filter::Cmp(CLASS, Cmp::EqU32(cb.encode(EMP))),
        Filter::Cmp(SEL, Cmp::EqI32(want)),
    ]);
    keep(
        res,
        Filter::and([Filter::plane(LIVE), Filter::negate(raw)]).sql_where(&[(SEL, VSEL)]),
    )
}

fn bit(m: &[u64], g: u64) -> bool {
    (m[(g >> 6) as usize] >> (g & 63)) & 1 == 1
}

// ── frames, facts and registers ───────────────────────────────────────────

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
/// TRUE, FALSE and UNKNOWN employees all present (asserted by the tests).
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

struct Maps {
    emp: LocalMap,
    dept: LocalMap,
    skill: LocalMap,
    site: LocalMap,
}
fn maps(w: &World) -> Maps {
    Maps {
        emp: LocalMap::prepare(&w.emp_ids).unwrap(),
        dept: LocalMap::prepare(&w.dept_ids).unwrap(),
        skill: LocalMap::prepare(&w.skills).unwrap(),
        site: LocalMap::prepare(&w.sites).unwrap(),
    }
}

/// Registers are built from the oracle's positions, not through the map, so
/// a wrong map cannot make a register agree with itself.
fn plane(facts: &[(u64, u64)], rows: &[u64], cols: &[u64]) -> Rows {
    let mut r = [0u64; 64];
    for &(a, b) in facts {
        r[pos(rows, a) as usize] |= 1 << pos(cols, b);
    }
    r
}
fn certified_planes(w: &World) -> (Rows, Rows) {
    let (mut val, mut known) = ([0u64; 64], [0u64; 64]);
    for &(e, s, c) in &w.certified {
        let (x, y) = (pos(&w.emp_ids, e), pos(&w.skills, s));
        if let Some(v) = c {
            known[x as usize] |= 1 << y;
            if v {
                val[x as usize] |= 1 << y;
            }
        }
    }
    (val, known)
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
fn certified_sig() -> MethodSignature {
    MethodSignature {
        truth: Truth::ThreeValued,
        ..sig(2, emp_id(), axis(EMP_CLASS, AxisKind::Property))
    }
}
fn works_in_sig() -> MethodSignature {
    sig(3, emp_id(), axis(DEPT_CLASS, AxisKind::Identity))
}
fn located_at_sig() -> MethodSignature {
    sig(
        4,
        axis(DEPT_CLASS, AxisKind::Identity),
        axis(DEPT_CLASS, AxisKind::Property),
    )
}
fn dept_attr_sig() -> MethodSignature {
    sig(
        5,
        axis(DEPT_CLASS, AxisKind::Identity),
        axis(DEPT_CLASS, AxisKind::Property),
    )
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

#[allow(clippy::too_many_arguments)]
fn select(
    map: &LocalMap,
    t: &[u64],
    gst: &FrameStore,
    gseal: &Frame,
    lst: &FrameStore,
    g: &Guards,
    axis: Axis,
    method: MethodSignature,
) -> Result<LocalSelection, ProjectRefusal> {
    project(
        map,
        t,
        Under {
            global_store: gst,
            global: gseal,
            local_store: lst,
            guards: g,
            axis,
            method,
            criteria: CRITERIA,
        },
    )
}

/// The oracle's in-frame (TRUE, FALSE, UNKNOWN) words for `ids`.
fn oracle_local(src: &[Row], ids: &[u64], want: i32) -> (u64, u64, u64) {
    let (mut t, mut f, mut u) = (0u64, 0u64, 0u64);
    for (l, &g) in ids.iter().enumerate() {
        match verdict(&src[g as usize], want) {
            Some(Some(true)) => t |= 1 << l,
            Some(Some(false)) => f |= 1 << l,
            Some(None) => u |= 1 << l,
            None => panic!("frame ids are live by construction"),
        }
    }
    (t, f, u)
}

const SEED: u64 = 0xD0CA_0002;
const WANT: i32 = 1;

// ── 1. global selection inside and outside the frame ──────────────────────

#[test]
fn global_selection_is_exact_and_only_frame_identities_are_projected() {
    let w = world(SEED);
    let cb = Codebook { shift: 0 };
    let res = resident(&w.src, &cb);
    let t = where_k3(&res, &cb, WANT);
    // Whole-mask comparison against the oracle, every one of 65,536 addresses.
    let (mut gt, mut gf, mut gu) = (0u32, 0u32, 0u32);
    for (g, r) in w.src.iter().enumerate() {
        let g64 = g as u64;
        let v = verdict(r, WANT);
        assert_eq!(bit(&t, g64), v == Some(Some(true)), "TRUE at {g}");
        gt += u32::from(v == Some(Some(true)));
        gf += u32::from(v == Some(Some(false)));
        gu += u32::from(v == Some(None));
    }
    let m = maps(&w);
    let (gst, lst) = (global_store(&cb), local_store(&m.emp));
    let (seal, g) = (gst.seal(), guards(&lst, FOCUS, SCOPE));
    let sel = select(&m.emp, &t, &gst, &seal, &lst, &g, emp_id(), has_skill_sig()).unwrap();
    let r = plane(&w.has_skill, &w.emp_ids, &w.skills);
    let b = bind(&lst, &g, has_skill_sig(), &r, None);
    let a = sel.admit(&gst, &lst, &b, CRITERIA).unwrap();
    let (ot, of, ou) = oracle_local(&w.src, &w.emp_ids, WANT);
    assert_eq!(a.truth(), ot);
    // Neither a FALSE nor an UNKNOWN identity is selected.
    assert_eq!(a.truth() & (of | ou), 0);
    assert_eq!(a.slots(), u64::MAX);
    // Anti-vacuity: all three verdicts occur in the frame, and the global
    // selection reaches far beyond it. Projection work is the frame, not the
    // universe.
    assert!(ot != 0 && of != 0 && ou != 0, "t={ot:x} f={of:x} u={ou:x}");
    assert!(gt > 1000 && gf > 1000 && gu > 1000);
    assert!(gt > ot.count_ones() * 10);
    assert_eq!(sel.probes(), 64);
    assert_eq!(m.emp.lookups(), 0, "projection never searches the map");
}

// ── 2. two tables, equal local ordinals ───────────────────────────────────

#[test]
fn equal_local_ordinals_of_two_tables_never_alias() {
    let w = world(SEED);
    let cb = Codebook { shift: 0 };
    let res = resident(&w.src, &cb);
    let t = where_k3(&res, &cb, WANT);
    let m = maps(&w);
    let gst = global_store(&cb);
    let seal = gst.seal();
    let (lemp, ldept) = (local_store(&m.emp), local_store(&m.dept));
    let (ge, gd) = (guards(&lemp, FOCUS, SCOPE), guards(&ldept, FOCUS, SCOPE));
    let se = select(
        &m.emp,
        &t,
        &gst,
        &seal,
        &lemp,
        &ge,
        emp_id(),
        has_skill_sig(),
    )
    .unwrap();
    let dept_axis = axis(DEPT_CLASS, AxisKind::Identity);
    let sd = select(
        &m.dept,
        &t,
        &gst,
        &seal,
        &ldept,
        &gd,
        dept_axis,
        dept_attr_sig(),
    )
    .unwrap();
    let re = plane(&w.has_skill, &w.emp_ids, &w.skills);
    let rd = plane(&w.located_at, &w.dept_ids, &w.sites);
    let be = bind(&lemp, &ge, has_skill_sig(), &re, None);
    let bd = bind(&ldept, &gd, dept_attr_sig(), &rd, None);
    let ae = se.admit(&gst, &lemp, &be, CRITERIA).unwrap();
    let ad = sd.admit(&gst, &ldept, &bd, CRITERIA).unwrap();
    // The employee selection occupies ordinals the department frame also has;
    // projected through the department map it selects no department.
    assert_ne!(ae.truth() & ad.slots(), 0);
    assert_eq!(ad.truth(), 0);
    // An employee selection cannot be read against a department-row register,
    // even when that register sits in the very frame the selection was made in.
    let bd_in_emp = bind(&lemp, &ge, dept_attr_sig(), &rd, None);
    assert_eq!(
        se.admit(&gst, &lemp, &bd_in_emp, CRITERIA),
        Err(ReuseRefusal::AxisMismatch)
    );
    // Nor against a register of the other frame.
    assert_eq!(
        se.admit(&gst, &lemp, &bd, CRITERIA),
        Err(ReuseRefusal::DifferentFrame)
    );
}

// ── 3. reads under the selection, both orientations and YZ rotation ───────

#[test]
fn selected_identity_by_property_matches_the_oracle_and_survives_rotation() {
    let w = world(SEED);
    let cb = Codebook { shift: 0 };
    let res = resident(&w.src, &cb);
    let t = where_k3(&res, &cb, WANT);
    let m = maps(&w);
    let (gst, lst) = (global_store(&cb), local_store(&m.emp));
    let (seal, g) = (gst.seal(), guards(&lst, FOCUS, SCOPE));
    let sel = select(&m.emp, &t, &gst, &seal, &lst, &g, emp_id(), has_skill_sig()).unwrap();
    let r = plane(&w.has_skill, &w.emp_ids, &w.skills);
    let b = bind(&lst, &g, has_skill_sig(), &r, None);
    let a = sel.admit(&gst, &lst, &b, CRITERIA).unwrap();
    let selected = |e: u64| verdict(&w.src[e as usize], WANT) == Some(Some(true));
    let holds = |e: u64, s: u64| w.has_skill.contains(&(e, s));
    let (mut cells, mut selected_facts, mut rotated_hits) = (0u32, 0u32, 0u32);
    for &e in &w.emp_ids {
        let x = pos(&w.emp_ids, e);
        for (y, &s) in w.skills.iter().enumerate() {
            let y = y as u8;
            let want = selected(e) && holds(e, s);
            assert_eq!(selected_cell(b.rows, &a, x, y).unwrap(), want);
            // YZ rotation: the rotated address (y, x) reads the same fact, and
            // the selection still applies to the identity coordinate.
            let rot =
                (a.truth() >> x) & 1 == 1 && cell_rotated(b.rows, address(y, x).unwrap()).unwrap();
            assert_eq!(rot, want, "rotated ({y}, {x})");
            rotated_hits += u32::from(rot);
            cells += u32::from(want);
        }
    }
    for (y, &s) in w.skills.iter().enumerate() {
        let col = selected_column(b.rows, &a, y as u8).unwrap();
        let expect = w
            .emp_ids
            .iter()
            .filter(|&&e| selected(e) && holds(e, s))
            .fold(0u64, |acc, &e| acc | 1 << pos(&w.emp_ids, e));
        assert_eq!(col, expect, "column {s}");
    }
    selected_facts += w.has_skill.iter().filter(|&&(e, _)| selected(e)).count() as u32;
    assert_eq!(selected_count(b.rows, &a).unwrap(), selected_facts);
    assert_eq!(cells, selected_facts);
    assert_eq!(rotated_hits, cells);
    // Anti-vacuity: the selection removes facts, and keeps some.
    assert!(selected_facts > 0 && (selected_facts as usize) < w.has_skill.len());
    // The fixture is not symmetric, so an identity "rotation" cannot pass.
    assert!((0..64u8).any(|x| {
        (0..64u8).any(|y| {
            cell(&r, address(x, y).unwrap()).unwrap() != cell(&r, address(y, x).unwrap()).unwrap()
        })
    }));
}

// ── 4. NULL / UNKNOWN: NOT UNKNOWN is UNKNOWN ─────────────────────────────

const NOT_CRITERIA: Criteria = Criteria { id: 0xC1_0001 };

#[test]
fn unknown_is_neither_selected_nor_negated_and_never_read_as_false() {
    let w = world(SEED);
    let cb = Codebook { shift: 0 };
    let res = resident(&w.src, &cb);
    let t = where_k3(&res, &cb, WANT);
    // The negation is its own WHERE through Quack's three-valued sql_where.
    let tn = where_not(&res, &cb, WANT);
    // Globally: the negated WHERE keeps exactly the oracle's FALSE rows, and
    // a NULL row is kept by neither WHERE.
    let mut gap = 0u32;
    for (g, r) in w.src.iter().enumerate() {
        let (g64, v) = (g as u64, verdict(r, WANT));
        assert_eq!(bit(&tn, g64), v == Some(Some(false)), "NOT at {g}");
        if v == Some(None) {
            assert!(!bit(&t, g64) && !bit(&tn, g64), "UNKNOWN at {g}");
            gap += 1;
        }
    }
    assert!(gap > 1000);
    let m = maps(&w);
    let (gst, lst) = (global_store(&cb), local_store(&m.emp));
    let (seal, g) = (gst.seal(), guards(&lst, FOCUS, SCOPE));
    let (val, known) = certified_planes(&w);
    let b = bind(&lst, &g, certified_sig(), &val, Some(&known));
    let sel = select(&m.emp, &t, &gst, &seal, &lst, &g, emp_id(), certified_sig()).unwrap();
    let a = sel.admit(&gst, &lst, &b, CRITERIA).unwrap();
    let under = |criteria| Under {
        global_store: &gst,
        global: &seal,
        local_store: &lst,
        guards: &g,
        axis: emp_id(),
        method: certified_sig(),
        criteria,
    };
    let neg = project(&m.emp, &tn, under(NOT_CRITERIA)).unwrap();
    let an = neg.admit(&gst, &lst, &b, NOT_CRITERIA).unwrap();
    // A selection is not reusable as its own negation.
    assert_eq!(
        neg.admit(&gst, &lst, &b, CRITERIA),
        Err(ReuseRefusal::CriteriaChanged)
    );
    // In the frame: the UNKNOWN slots are in neither selection, so the
    // complement of a selection is not its negation.
    let (ot, of, ou) = oracle_local(&w.src, &w.emp_ids, WANT);
    assert_eq!((a.truth(), an.truth()), (ot, of));
    assert_eq!(a.truth() & an.truth(), 0);
    assert_eq!(a.slots() & !(a.truth() | an.truth()), ou);
    assert_ne!(ou, 0);
    assert_ne!(a.slots() & !a.truth(), an.truth());
    // The NULL rows' payload is 1 = WANT, so an engine reading payloads at
    // NULL would have kept them: none is selected by either WHERE.
    for &e in &w.emp_ids {
        if w.src[e as usize].sel.is_none() {
            let x = pos(&w.emp_ids, e);
            assert_eq!((a.truth() | an.truth()) >> x & 1, 0);
        }
    }
    // Three-valued property over the selected identities, in the register's
    // own value + validity form.
    let check = |b: &BoundDr64<'_>| {
        let (mut u_cells, mut f_cells) = (0u32, 0u32);
        for (y, &s) in w.skills.iter().enumerate() {
            let (cv, ck) = selected_column_k3(b, &a, y as u8).unwrap();
            let (mut et, mut ef, mut eu) = (0u64, 0u64, 0u64);
            for &e in &w.emp_ids {
                if verdict(&w.src[e as usize], WANT) != Some(Some(true)) {
                    continue;
                }
                let bitpos = 1u64 << pos(&w.emp_ids, e);
                match w.certified.iter().find(|c| c.0 == e && c.1 == s) {
                    Some((_, _, Some(true))) => et |= bitpos,
                    Some((_, _, Some(false))) => ef |= bitpos,
                    // No fact, or a fact with unknown certification: UNKNOWN.
                    _ => eu |= bitpos,
                }
            }
            assert_eq!(cv, et, "TRUE, certified column {s}");
            assert_eq!(ck & !cv, ef, "FALSE, certified column {s}");
            assert_eq!(a.truth() & !ck, eu, "UNKNOWN, certified column {s}");
            u_cells += (a.truth() & !ck).count_ones();
            f_cells += (ck & !cv).count_ones();
        }
        assert!(u_cells > 0 && f_cells > 0);
    };
    check(&b);
    // A stale payload bit under an UNKNOWN cell of a selected identity is
    // never returned as TRUE.
    let (x, y) = w
        .emp_ids
        .iter()
        .filter(|&&e| verdict(&w.src[e as usize], WANT) == Some(Some(true)))
        .flat_map(|&e| w.skills.iter().map(move |&s| (e, s)))
        .find(|&(e, s)| {
            !matches!(
                w.certified.iter().find(|c| c.0 == e && c.1 == s),
                Some((_, _, Some(_)))
            )
        })
        .map(|(e, s)| (pos(&w.emp_ids, e), pos(&w.skills, s)))
        .unwrap();
    let mut dirty = val;
    dirty[x as usize] |= 1 << y;
    let bd = bind(&lst, &g, certified_sig(), &dirty, Some(&known));
    check(&bd);
    // The Boolean folds read payload bits as presence, so they refuse a
    // selection admitted against this three-valued register.
    assert_eq!(selected_count(b.rows, &a), None);
    assert_eq!(selected_column(b.rows, &a, 0), None);
    assert_eq!(selected_cell(b.rows, &a, 0, 0), None);
}

// ── 5. source change after WHERE: refuse, then rebind ─────────────────────

#[test]
fn source_change_after_where_refuses_reuse_and_rebind_matches() {
    let mut w = world(SEED);
    let cb = Codebook { shift: 0 };
    let res = resident(&w.src, &cb);
    let t = where_k3(&res, &cb, WANT);
    let m = maps(&w);
    let (mut gst, lst) = (global_store(&cb), local_store(&m.emp));
    let (seal, g) = (gst.seal(), guards(&lst, FOCUS, SCOPE));
    let sel = select(&m.emp, &t, &gst, &seal, &lst, &g, emp_id(), has_skill_sig()).unwrap();
    let r = plane(&w.has_skill, &w.emp_ids, &w.skills);
    let b = bind(&lst, &g, has_skill_sig(), &r, None);
    let before = sel.admit(&gst, &lst, &b, CRITERIA).unwrap();
    // Flip `sel` on every selected frame employee: none is TRUE any more.
    for &e in &w.emp_ids {
        if verdict(&w.src[e as usize], WANT) == Some(Some(true)) {
            w.src[e as usize].sel = Some(0);
        }
    }
    gst.advance(FrameRevisions {
        source: 2,
        ..gst.revisions()
    });
    assert_eq!(
        sel.admit(&gst, &lst, &b, CRITERIA),
        Err(ReuseRefusal::Global(Refusal::SourceChanged))
    );
    // Rebind: WHERE again, project again, under the new seal.
    let res2 = resident(&w.src, &cb);
    let t2 = where_k3(&res2, &cb, WANT);
    let seal2 = gst.seal();
    let sel2 = select(
        &m.emp,
        &t2,
        &gst,
        &seal2,
        &lst,
        &g,
        emp_id(),
        has_skill_sig(),
    )
    .unwrap();
    let after = sel2.admit(&gst, &lst, &b, CRITERIA).unwrap();
    assert_eq!(after.truth(), 0);
    assert_eq!(after.truth(), oracle_local(&w.src, &w.emp_ids, WANT).0);
    // The silent alternative: the old projection would still answer.
    assert_ne!(
        selected_count(b.rows, &before).unwrap(),
        selected_count(b.rows, &after).unwrap()
    );
}

// ── 6. ACL / scope and focus with identical coordinates ───────────────────

#[test]
fn scope_or_focus_change_refuses_with_identical_coordinates() {
    let w = world(SEED);
    let cb = Codebook { shift: 0 };
    let res = resident(&w.src, &cb);
    let t = where_k3(&res, &cb, WANT);
    let m = maps(&w);
    let (gst, lst) = (global_store(&cb), local_store(&m.emp));
    let (seal, g) = (gst.seal(), guards(&lst, FOCUS, SCOPE));
    let sel = select(&m.emp, &t, &gst, &seal, &lst, &g, emp_id(), has_skill_sig()).unwrap();
    let r = plane(&w.has_skill, &w.emp_ids, &w.skills);
    // Same frame issue, same coordinate numbers, same register bytes.
    let g_scope = Guards {
        scope: Scope { id: 8 },
        ..g.clone()
    };
    let b_scope = bind(&lst, &g_scope, has_skill_sig(), &r, None);
    assert_eq!(
        sel.admit(&gst, &lst, &b_scope, CRITERIA),
        Err(ReuseRefusal::ScopeChanged)
    );
    let g_focus = Guards {
        focus: Focus { revision: 2 },
        ..g.clone()
    };
    let b_focus = bind(&lst, &g_focus, has_skill_sig(), &r, None);
    assert_eq!(
        sel.admit(&gst, &lst, &b_focus, CRITERIA),
        Err(ReuseRefusal::FocusChanged)
    );
    // Silence twin: the unchanged register admits.
    let b = bind(&lst, &g, has_skill_sig(), &r, None);
    assert!(sel.admit(&gst, &lst, &b, CRITERIA).is_ok());
}

// ── 7. witnessed FK: selected employees' sites; FK change refuses ─────────

#[test]
fn fk_projection_over_selection_matches_and_a_changed_reference_refuses() {
    let mut w = world(SEED);
    let cb = Codebook { shift: 0 };
    let res = resident(&w.src, &cb);
    let t = where_k3(&res, &cb, WANT);
    let m = maps(&w);
    let gst = global_store(&cb);
    // One frame over both tables; its coordinates name the employee map, the
    // axis the selection is projected onto.
    let both = FrameRevisions {
        source: 1,
        relation_map: 1,
        coordinates: m.emp.digest(),
    };
    let mut lst = FrameStore::new(both);
    let (seal, g) = (gst.seal(), guards(&lst, FOCUS, SCOPE));
    let sel = select(&m.emp, &t, &gst, &seal, &lst, &g, emp_id(), works_in_sig()).unwrap();
    let rw = plane(&w.works_in, &w.emp_ids, &w.dept_ids);
    let rl = plane(&w.located_at, &w.dept_ids, &w.sites);
    let bw = bind(&lst, &g, works_in_sig(), &rw, None);
    let bl = bind(&lst, &g, located_at_sig(), &rl, None);
    let a = sel.admit(&gst, &lst, &bw, CRITERIA).unwrap();
    let chain = Chain::new(&lst, vec![bw.clone(), bl.clone()]).unwrap();
    let (es, _) = chain
        .eval(&RelExpr::comp(RelExpr::Leaf(0), RelExpr::Leaf(1)))
        .unwrap();
    let selected = |e: u64| verdict(&w.src[e as usize], WANT) == Some(Some(true));
    let at_site = |w: &World, e: u64, s: u64| {
        w.works_in
            .iter()
            .any(|&(a, d)| a == e && w.located_at.contains(&(d, s)))
    };
    let mut hits = 0u32;
    for &e in &w.emp_ids {
        for &s in &w.sites {
            let got = selected_cell(&es, &a, pos(&w.emp_ids, e), pos(&w.sites, s)).unwrap();
            assert_eq!(got, selected(e) && at_site(&w, e, s), "({e}, {s})");
            hits += u32::from(got);
        }
    }
    assert!(hits > 0);
    drop(chain);
    // The referenced property changes: one department moves to a new site.
    let (d0, _) = w.located_at[0];
    w.located_at.retain(|&(d, _)| d != d0);
    w.located_at.push((d0, w.sites[15]));
    lst.advance(FrameRevisions { source: 2, ..both });
    assert_eq!(
        sel.admit(&gst, &lst, &bw, CRITERIA),
        Err(ReuseRefusal::Local(Refusal::SourceChanged))
    );
    assert_eq!(
        Chain::new(&lst, vec![bw.clone(), bl.clone()]).err(),
        Some(ComposeRefusal::StaleFrame)
    );
    // Rebind both registers and the selection under the new seal.
    let g2 = guards(&lst, FOCUS, SCOPE);
    let rl2 = plane(&w.located_at, &w.dept_ids, &w.sites);
    let bw2 = bind(&lst, &g2, works_in_sig(), &rw, None);
    let bl2 = bind(&lst, &g2, located_at_sig(), &rl2, None);
    let sel2 = select(&m.emp, &t, &gst, &seal, &lst, &g2, emp_id(), works_in_sig()).unwrap();
    let a2 = sel2.admit(&gst, &lst, &bw2, CRITERIA).unwrap();
    let chain2 = Chain::new(&lst, vec![bw2, bl2]).unwrap();
    let (es2, _) = chain2
        .eval(&RelExpr::comp(RelExpr::Leaf(0), RelExpr::Leaf(1)))
        .unwrap();
    for &e in &w.emp_ids {
        for &s in &w.sites {
            let got = selected_cell(&es2, &a2, pos(&w.emp_ids, e), pos(&w.sites, s)).unwrap();
            assert_eq!(got, selected(e) && at_site(&w, e, s));
        }
    }
    let _ = m.site.len() + m.skill.len();
}

// ── 8. reseal and coordinate changes ──────────────────────────────────────

#[test]
fn reseal_and_coordinate_changes_refuse() {
    let w = world(SEED);
    let cb = Codebook { shift: 0 };
    let res = resident(&w.src, &cb);
    let t = where_k3(&res, &cb, WANT);
    let m = maps(&w);
    let gst = global_store(&cb);
    let seal = gst.seal();
    let mut lst = local_store(&m.emp);
    let g = guards(&lst, FOCUS, SCOPE);
    let sel = select(&m.emp, &t, &gst, &seal, &lst, &g, emp_id(), has_skill_sig()).unwrap();
    let r = plane(&w.has_skill, &w.emp_ids, &w.skills);
    let b = bind(&lst, &g, has_skill_sig(), &r, None);
    let before = sel.admit(&gst, &lst, &b, CRITERIA).unwrap();
    // Reseal at the same numbers.
    lst.advance(lst.revisions());
    assert_eq!(
        sel.admit(&gst, &lst, &b, CRITERIA),
        Err(ReuseRefusal::Local(Refusal::FrameResealed))
    );
    // A new coordinate assignment: the same ids in another order.
    let mut reordered = w.emp_ids.clone();
    reordered.reverse();
    let map2 = LocalMap::prepare(&reordered).unwrap();
    assert_ne!(map2.digest(), m.emp.digest());
    lst.advance(FrameRevisions {
        coordinates: map2.digest(),
        ..lst.revisions()
    });
    assert_eq!(
        sel.admit(&gst, &lst, &b, CRITERIA),
        Err(ReuseRefusal::Local(Refusal::CoordinatesChanged))
    );
    // Projecting through the old map under the new frame is refused.
    let g2 = guards(&lst, FOCUS, SCOPE);
    assert_eq!(
        select(
            &m.emp,
            &t,
            &gst,
            &seal,
            &lst,
            &g2,
            emp_id(),
            has_skill_sig()
        )
        .err(),
        Some(ProjectRefusal::MapNotFrame)
    );
    // Through the new map it projects, with the bits moved to the new
    // ordinals.
    let sel2 = select(&map2, &t, &gst, &seal, &lst, &g2, emp_id(), has_skill_sig()).unwrap();
    let r2 = plane(&w.has_skill, &reordered, &w.skills);
    let b2 = bind(&lst, &g2, has_skill_sig(), &r2, None);
    let a2 = sel2.admit(&gst, &lst, &b2, CRITERIA).unwrap();
    assert_eq!(a2.truth(), oracle_local(&w.src, &reordered, WANT).0);
    assert_eq!(
        selected_count(b2.rows, &a2).unwrap(),
        selected_count(&r, &before).unwrap()
    );
}

// ── 9. owner, codebook, criteria, schema ──────────────────────────────────

#[test]
fn foreign_owner_codebook_criteria_and_schema_refuse() {
    let w = world(SEED);
    let cb = Codebook { shift: 0 };
    let res = resident(&w.src, &cb);
    let t = where_k3(&res, &cb, WANT);
    let m = maps(&w);
    let mut gst = global_store(&cb);
    let lst = local_store(&m.emp);
    let (seal, g) = (gst.seal(), guards(&lst, FOCUS, SCOPE));
    let sel = select(&m.emp, &t, &gst, &seal, &lst, &g, emp_id(), has_skill_sig()).unwrap();
    let r = plane(&w.has_skill, &w.emp_ids, &w.skills);
    let b = bind(&lst, &g, has_skill_sig(), &r, None);
    // Another owner at identical numbers.
    let other = global_store(&cb);
    assert_eq!(
        sel.admit(&other, &lst, &b, CRITERIA),
        Err(ReuseRefusal::Global(Refusal::ForeignFrame))
    );
    // Other criteria.
    assert_eq!(
        sel.admit(&gst, &lst, &b, Criteria { id: 0xC2 }),
        Err(ReuseRefusal::CriteriaChanged)
    );
    // Another DR method over the same axis (DT schema moved).
    let other_method = sig(9, emp_id(), axis(EMP_CLASS, AxisKind::Property));
    let b_schema = bind(&lst, &g, other_method, &r, None);
    assert_eq!(
        sel.admit(&gst, &lst, &b_schema, CRITERIA),
        Err(ReuseRefusal::SchemaChanged)
    );
    // A new codebook: `employee` is re-encoded. The old projection refuses;
    // the stale binding, run silently, keeps nobody.
    let cb2 = Codebook { shift: 100 };
    gst.advance(FrameRevisions {
        relation_map: u64::from(cb2.shift),
        ..gst.revisions()
    });
    assert_eq!(
        sel.admit(&gst, &lst, &b, CRITERIA),
        Err(ReuseRefusal::Global(Refusal::MapChanged))
    );
    let res2 = resident(&w.src, &cb2);
    let t_stale = where_k3(&res2, &cb, WANT);
    let t_fresh = where_k3(&res2, &cb2, WANT);
    assert_eq!(t_fresh, t);
    assert!(t_stale.iter().all(|&x| x == 0));
    let seal2 = gst.seal();
    let sel2 = select(
        &m.emp,
        &t_fresh,
        &gst,
        &seal2,
        &lst,
        &g,
        emp_id(),
        has_skill_sig(),
    )
    .unwrap();
    assert!(sel2.admit(&gst, &lst, &b, CRITERIA).is_ok());
}

// ── 10. empty and full selections ─────────────────────────────────────────

#[test]
fn empty_and_full_global_selections_are_exact() {
    let w = world(SEED);
    let cb = Codebook { shift: 0 };
    let res = resident(&w.src, &cb);
    let m = maps(&w);
    let (gst, lst) = (global_store(&cb), local_store(&m.emp));
    let (seal, g) = (gst.seal(), guards(&lst, FOCUS, SCOPE));
    let r = plane(&w.has_skill, &w.emp_ids, &w.skills);
    let b = bind(&lst, &g, has_skill_sig(), &r, None);
    // Empty: no row has sel = 99. Every frame employee is FALSE or UNKNOWN.
    let t0 = where_k3(&res, &cb, 99);
    assert!(t0.iter().all(|&x| x == 0));
    let s0 = select(
        &m.emp,
        &t0,
        &gst,
        &seal,
        &lst,
        &g,
        emp_id(),
        has_skill_sig(),
    )
    .unwrap();
    let a0 = s0.admit(&gst, &lst, &b, CRITERIA).unwrap();
    assert_eq!(a0.truth(), oracle_local(&w.src, &w.emp_ids, 99).0);
    assert_eq!(selected_count(b.rows, &a0).unwrap(), 0);
    // Full: the every-address plane. All 65,536 TRUE, every slot selected.
    let full = keep(&res, Filter::plane(ALL));
    assert_eq!(
        full.iter().map(|w| w.count_ones()).sum::<u32>(),
        GLOBAL as u32
    );
    let s1 = select(
        &m.emp,
        &full,
        &gst,
        &seal,
        &lst,
        &g,
        emp_id(),
        has_skill_sig(),
    )
    .unwrap();
    let a1 = s1.admit(&gst, &lst, &b, CRITERIA).unwrap();
    assert_eq!(a1.truth(), u64::MAX);
    assert_eq!(
        selected_count(b.rows, &a1).unwrap() as usize,
        w.has_skill.len()
    );
    // Malformed inputs are refused, not read.
    assert_eq!(
        select(
            &m.emp,
            &full[..10],
            &gst,
            &seal,
            &lst,
            &g,
            emp_id(),
            has_skill_sig()
        )
        .err(),
        Some(ProjectRefusal::WrongWidth)
    );
    let far = LocalMap::prepare(&[GLOBAL as u64]).unwrap();
    let lfar = local_store(&far);
    let gfar = guards(&lfar, FOCUS, SCOPE);
    assert_eq!(
        select(
            &far,
            &full,
            &gst,
            &seal,
            &lfar,
            &gfar,
            emp_id(),
            has_skill_sig()
        )
        .err(),
        Some(ProjectRefusal::OutOfUniverse)
    );
}

// ── 11. #1792's shape: a detached Keep reused after a criteria input moved ─

#[test]
fn stale_keep_negative_control_is_refused_not_answered() {
    let mut w = world(SEED);
    let cb = Codebook { shift: 0 };
    let res = resident(&w.src, &cb);
    let t = where_k3(&res, &cb, WANT);
    // The detached Keep: kept outside any guard, then the criteria input
    // (`sel`) changes on half the selected rows of the whole universe.
    let detached = t.clone();
    let mut moved = 0u32;
    for (g, r) in w.src.iter_mut().enumerate() {
        if bit(&t, g as u64) && g % 2 == 0 {
            r.sel = Some(2);
            moved += 1;
        }
    }
    let res2 = resident(&w.src, &cb);
    let fresh = where_k3(&res2, &cb, WANT);
    let pop = |m: &[u64]| m.iter().map(|x| x.count_ones()).sum::<u32>();
    // Unguarded, the detached mask answers silently with the old population.
    assert_eq!(pop(&detached) - pop(&fresh), moved);
    assert!(moved > 100);
    // Guarded: the same mask, projected under the old seal, is refused once
    // the owner records the change.
    let m = maps(&w);
    let (mut gst, lst) = (global_store(&cb), local_store(&m.emp));
    let (seal, g) = (gst.seal(), guards(&lst, FOCUS, SCOPE));
    let sel = select(
        &m.emp,
        &detached,
        &gst,
        &seal,
        &lst,
        &g,
        emp_id(),
        has_skill_sig(),
    )
    .unwrap();
    let r = plane(&w.has_skill, &w.emp_ids, &w.skills);
    let b = bind(&lst, &g, has_skill_sig(), &r, None);
    gst.advance(FrameRevisions {
        source: 2,
        ..gst.revisions()
    });
    assert_eq!(
        sel.admit(&gst, &lst, &b, CRITERIA),
        Err(ReuseRefusal::Global(Refusal::SourceChanged))
    );
    // Projecting the detached mask under the NEW seal is not detected: the
    // seal attests the caller's claim, not the mask's provenance. Recorded
    // as the open limitation, asserted so a fix flips this test.
    let seal2 = gst.seal();
    assert!(
        select(
            &m.emp,
            &detached,
            &gst,
            &seal2,
            &lst,
            &g,
            emp_id(),
            has_skill_sig()
        )
        .is_ok()
    );
}

// ── 12. allocation: projection, admission and folds allocate nothing ──────

#[test]
fn projection_admission_and_folds_do_not_allocate() {
    let w = world(SEED);
    let cb = Codebook { shift: 0 };
    let res = resident(&w.src, &cb);
    let t = where_k3(&res, &cb, WANT);
    let m = maps(&w);
    let (gst, lst) = (global_store(&cb), local_store(&m.emp));
    let (seal, g) = (gst.seal(), guards(&lst, FOCUS, SCOPE));
    let r = plane(&w.has_skill, &w.emp_ids, &w.skills);
    let b = bind(&lst, &g, has_skill_sig(), &r, None);
    let before = allocs();
    let sel = select(&m.emp, &t, &gst, &seal, &lst, &g, emp_id(), has_skill_sig()).unwrap();
    let a = sel.admit(&gst, &lst, &b, CRITERIA).unwrap();
    let mut acc = selected_count(b.rows, &a).unwrap();
    for y in 0..40u8 {
        acc ^= selected_column(b.rows, &a, y).unwrap().count_ones();
    }
    assert_eq!(allocs(), before, "acc={acc}");
    let _ = hhtl16::LOCAL + where_dr3::GLOBAL;
    let _ = Population::Distinct;
}

// ── 13. a projection under a stale seal is refused before any read ────────

#[test]
fn projection_under_a_stale_global_seal_or_local_frame_is_refused() {
    let w = world(SEED);
    let cb = Codebook { shift: 0 };
    let res = resident(&w.src, &cb);
    let t = where_k3(&res, &cb, WANT);
    let m = maps(&w);
    let (mut gst, mut lst) = (global_store(&cb), local_store(&m.emp));
    let (seal, g) = (gst.seal(), guards(&lst, FOCUS, SCOPE));
    // Silence twin: current seals project.
    assert!(select(&m.emp, &t, &gst, &seal, &lst, &g, emp_id(), has_skill_sig()).is_ok());
    // The global owner moved between WHERE and projection.
    gst.advance(FrameRevisions {
        source: 2,
        ..gst.revisions()
    });
    assert_eq!(
        select(&m.emp, &t, &gst, &seal, &lst, &g, emp_id(), has_skill_sig()).err(),
        Some(ProjectRefusal::Global(Refusal::SourceChanged))
    );
    // Another owner's seal at identical numbers.
    let other = global_store(&cb);
    let seal_now = gst.seal();
    assert_eq!(
        select(
            &m.emp,
            &t,
            &other,
            &seal_now,
            &lst,
            &g,
            emp_id(),
            has_skill_sig()
        )
        .err(),
        Some(ProjectRefusal::Global(Refusal::ForeignFrame))
    );
    // The local frame moved.
    lst.advance(lst.revisions());
    assert_eq!(
        select(
            &m.emp,
            &t,
            &gst,
            &seal_now,
            &lst,
            &g,
            emp_id(),
            has_skill_sig()
        )
        .err(),
        Some(ProjectRefusal::Local(Refusal::FrameResealed))
    );
}
