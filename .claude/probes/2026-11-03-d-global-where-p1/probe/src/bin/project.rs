//! LAB: D-GLOBAL-WHERE P2 Part 5b, the projection probe (property matching).
//!
//! Lab only, not production. Standalone: it does not use the probe's lib.rs.
//!
//! Question: CSV-style property matching / virtual CustomObject selection is a
//! PROJECTION concern. What does it cost to carry 64 LOGICAL properties over
//! 32 PHYSICAL SoA lanes, and where does a packed layout stop being able to
//! represent what the logical layer can say?
//!
//! Logical properties p0..p63:
//!   p0..p15   full-range i32
//!   p16..p47  small u8 (0..=255), the packable ones
//!   p48..p63  sparse: present on ~2 % of rows (f(id,j) % 50 == 0), value
//!             0..=999; ABSENT is NOT 0 (SQL NULL: a compare on an absent
//!             value is unknown, so the row is not selected).
//!
//! Layouts (all built from the same logical formula `lv`):
//!   U unpacked: 64 i32 lanes + 16 presence planes. 64 lanes exceeds the
//!     32-lane budget, so U is the REFERENCE, not a candidate.
//!   P packed, 32 lanes: 16 i32 lanes (p0..p15, bytes aliased to U's first 16
//!     lanes and not charged twice), 8 u32 lanes holding 4 x u8 each
//!     (p16..p47), 8 u32 lanes holding 2 x u16 each (p48..p63). Sparse
//!     encoding in P: stored = value + 1 as u16, 0 = absent. Legal because
//!     values <= 999 fit; a value that would need stored > 65535 is refused at
//!     build time (`encode_sparse`). Because 0 means absent, a packed sparse
//!     compare can never match an absent row.
//!   S sparse: p0..p47 as unpacked i32 lanes (aliased to U's first 48 lanes),
//!     and each sparse property as one presence mask (words_for(n) u64) plus a
//!     COMPACT i32 value column over present rows only, in row order. The
//!     compact column is read by walking the presence bits, so absent rows are
//!     never visited and can never match.
//!
//! Evaluation: Quack `lower` + mask-risc `execute_into`, `Agg::Count`, plain
//! (non-fused) lowering. Quack has no ordered compare over an unsigned lane,
//! so a packed ORDERED compare (< or >=) needs a Rust extraction loop
//! (shift + mask + compare, 64 rows per output word) that writes a
//! preallocated mask plane which Quack then ANDs/ORs like any other plane.
//! A packed EQUALITY compare needs no loop: it lowers to `Cmp::MatchU32`
//! with pattern/care. The S sparse compare is also a Rust loop (over present
//! bits only). That loop cost is what this probe exists to measure against the
//! unpacked lane compare. Every extraction loop is counted as `ext_loops`.
//!
//! REFUSALS (never silently clamped; each is exercised by a check that fails
//! the run if it does not fire):
//!   R1 a constant outside a packed field's range under P.
//!   R2 (not a refusal, a parity law) a sparse compare never matches absent
//!      rows under any layout: count(p50 >= 0) == presence count, not N.
//!   R3 a sparse value that does not fit the P u16 encoding, refused when the
//!      lane is built (separate tiny fixture).
//!
//! ORACLE: an independent scalar row loop over the LOGICAL formula `lv`; it
//! never reads any layout. It is never timed. Every arm's Count must equal it
//! or the binary exits 1.
//!
//! NOT claimed: no ms-scale result is a measurement here: a warm value
//! >= 1000 us prints FAILED-GATE and is reported, not hidden; build/prep
//! values are COLD (reported, not gate). No claim that the layouts are
//! equivalent beyond what the oracle checks. No tuple, no row-id list, no
//! persisted intermediate in any timed path.
//!
//! Usage: `project <OUT> [--n 1000000|10000000]...` (default: both sizes).
//! Writes `<OUT>/project/n<N>/project.json`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, LaneRef, Out, Planes, Program, Scratch, Value,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Mask, Query};

// ---------------------------------------------------------------------------
// counting allocator
// ---------------------------------------------------------------------------

struct CountingAlloc;
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: pure pass-through to `System`; the atomic is the only addition.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_BYTES.fetch_add(layout.size(), Relaxed);
        // SAFETY: same layout and contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn alloc_bytes() -> usize {
    ALLOC_BYTES.load(Relaxed)
}

// ---------------------------------------------------------------------------
// logical data (the single source of truth; layouts and oracle both use `lv`)
// ---------------------------------------------------------------------------

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const N_PROPS: usize = 64;
const N_FULL: usize = 16;
const N_SMALL_END: usize = 48;
const MAX_EXT: usize = 16;

/// Same closed formula as where_k.rs (u64, no overflow).
#[inline]
fn f(id: u64, j: u64) -> u64 {
    let a = (id * 2_654_435_761 + j * 40_503) % 2_147_483_648;
    ((a * 1_103_515_245) % 2_147_483_648) % 1000
}

/// SplitMix64-style finalizer keyed by (SEED, id, j); gives full-range bits.
#[inline]
fn h(id: u64, j: u64) -> u64 {
    let mut z = SEED ^ id.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (j + 1).wrapping_mul(0xD1B5_4A32_D192_ED03);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Logical value of property `j` at row `id`; None = absent (SQL NULL).
#[inline]
fn lv(id: u64, j: usize) -> Option<i64> {
    let ju = j as u64;
    if j < N_FULL {
        Some(i64::from(h(id, ju) as u32 as i32))
    } else if j < N_SMALL_END {
        Some(((h(id, ju) >> 8) % 256) as i64)
    } else if f(id, ju) % 50 == 0 {
        Some(((h(id, ju) >> 16) % 1000) as i64)
    } else {
        None
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Full,
    Small,
    Sparse,
}
impl Kind {
    fn of(j: usize) -> Kind {
        if j < N_FULL {
            Kind::Full
        } else if j < N_SMALL_END {
            Kind::Small
        } else {
            Kind::Sparse
        }
    }
}

// ---------------------------------------------------------------------------
// predicate tree, refusals
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Op {
    Lt,
    Eq,
    Ge,
}
impl Op {
    fn name(self) -> &'static str {
        match self {
            Op::Lt => "<",
            Op::Eq => "=",
            Op::Ge => ">=",
        }
    }
}
#[inline]
fn holds(op: Op, v: i64, c: i64) -> bool {
    match op {
        Op::Lt => v < c,
        Op::Eq => v == c,
        Op::Ge => v >= c,
    }
}

#[derive(Clone)]
enum Pr {
    Leaf(usize, Op, i64),
    And(Vec<Pr>),
    Or(Vec<Pr>),
}

#[derive(Debug, Clone)]
enum Refusal {
    ConstOutsideI32 { prop: usize, c: i64 },
    ConstOutsidePackedU8 { prop: usize, c: i64 },
    ConstOutsidePackedU16 { prop: usize, c: i64 },
    SparseValueUnencodable { v: i64 },
    TooManyExtractions,
    LowerFailed,
}
impl Refusal {
    fn reason(&self) -> String {
        match self {
            Refusal::ConstOutsideI32 { prop, c } => {
                format!("p{prop}: constant {c} does not fit an i32 lane")
            }
            Refusal::ConstOutsidePackedU8 { prop, c } => {
                format!("p{prop}: constant {c} outside packed u8 range 0..=255")
            }
            Refusal::ConstOutsidePackedU16 { prop, c } => {
                format!("p{prop}: constant {c} outside packed u16 sparse range 0..=65534")
            }
            Refusal::SparseValueUnencodable { v } => {
                format!("sparse value {v} does not fit the u16 (value+1) encoding")
            }
            Refusal::TooManyExtractions => format!("more than {MAX_EXT} extraction planes"),
            Refusal::LowerFailed => "quack lowering failed".to_string(),
        }
    }
}

/// P sparse encoding: stored = value + 1 as u16, 0 = absent.
fn encode_sparse(v: i64) -> Result<u16, Refusal> {
    if v < 0 || v + 1 > 65_535 {
        return Err(Refusal::SparseValueUnencodable { v });
    }
    Ok((v + 1) as u16)
}

/// Tiny-fixture packer for two sparse properties in one P lane (R3).
fn pack_sparse_pair(a: Option<i64>, b: Option<i64>) -> Result<u32, Refusal> {
    let ea = a.map_or(Ok(0u16), encode_sparse)?;
    let eb = b.map_or(Ok(0u16), encode_sparse)?;
    Ok(u32::from(ea) | (u32::from(eb) << 16))
}

/// ORACLE: scalar row evaluation over the logical formula. Absent => false
/// (equivalent to SQL unknown here because the trees contain no NOT).
fn oracle_row(p: &Pr, id: u64) -> bool {
    match p {
        Pr::Leaf(j, op, c) => lv(id, *j).is_some_and(|v| holds(*op, v, *c)),
        Pr::And(c) => c.iter().all(|x| oracle_row(x, id)),
        Pr::Or(c) => c.iter().any(|x| oracle_row(x, id)),
    }
}
fn oracle_count(p: &Pr, n: usize) -> usize {
    (0..n as u64).filter(|&id| oracle_row(p, id)).count()
}

// ---------------------------------------------------------------------------
// layouts
// ---------------------------------------------------------------------------

struct Store {
    n: usize,
    live: Vec<u64>,
    // U (S aliases lanes 0..48, P aliases lanes 0..16)
    u_lanes: Vec<Vec<i32>>,
    u_pres: Vec<Vec<u64>>,
    // P
    p_small: Vec<Vec<u32>>,
    p_sp: Vec<Vec<u32>>,
    // S
    s_pres: Vec<Vec<u64>>,
    s_vals: Vec<Vec<i32>>,
}

fn build_u(n: usize) -> (Vec<Vec<i32>>, Vec<Vec<u64>>) {
    let words = words_for(n);
    let mut lanes: Vec<Vec<i32>> = (0..N_PROPS).map(|_| Vec::with_capacity(n)).collect();
    let mut pres = vec![vec![0u64; words]; 16];
    for id in 0..n {
        for (j, lane) in lanes.iter_mut().enumerate() {
            let v = lv(id as u64, j);
            lane.push(v.unwrap_or(0) as i32);
            if j >= N_SMALL_END && v.is_some() {
                pres[j - N_SMALL_END][id >> 6] |= 1u64 << (id & 63);
            }
        }
    }
    (lanes, pres)
}

fn build_p(n: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let mut small = vec![vec![0u32; n]; 8];
    let mut sp = vec![vec![0u32; n]; 8];
    for id in 0..n {
        for i in 0..32 {
            let v = lv(id as u64, N_FULL + i).expect("small is never absent") as u32;
            small[i / 4][id] |= v << (8 * (i % 4));
        }
        for s in 0..16 {
            let e = match lv(id as u64, N_SMALL_END + s) {
                Some(v) => u32::from(encode_sparse(v).expect("sparse values are <= 999")),
                None => 0,
            };
            sp[s / 2][id] |= e << (16 * (s % 2));
        }
    }
    (small, sp)
}

fn build_s(n: usize) -> (Vec<Vec<u64>>, Vec<Vec<i32>>) {
    let words = words_for(n);
    let mut pres = vec![vec![0u64; words]; 16];
    let mut vals: Vec<Vec<i32>> = (0..16).map(|_| Vec::new()).collect();
    for id in 0..n {
        for s in 0..16 {
            if let Some(v) = lv(id as u64, N_SMALL_END + s) {
                pres[s][id >> 6] |= 1u64 << (id & 63);
                vals[s].push(v as i32);
            }
        }
    }
    (pres, vals)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    U,
    P,
    S,
}
impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::U => "U",
            Arm::P => "P",
            Arm::S => "S",
        }
    }
}

struct ArmView<'a> {
    arm: Arm,
    lanes: Vec<LaneRef<'a>>,
}

fn view(arm: Arm, st: &Store) -> ArmView<'_> {
    let mut lanes: Vec<LaneRef<'_>> = Vec::new();
    match arm {
        Arm::U => lanes.extend(st.u_lanes.iter().map(|c| LaneRef::I32(c))),
        Arm::S => lanes.extend(st.u_lanes[..N_SMALL_END].iter().map(|c| LaneRef::I32(c))),
        Arm::P => {
            lanes.extend(st.u_lanes[..N_FULL].iter().map(|c| LaneRef::I32(c)));
            lanes.extend(st.p_small.iter().map(|c| LaneRef::U32(c)));
            lanes.extend(st.p_sp.iter().map(|c| LaneRef::U32(c)));
        }
    }
    ArmView { arm, lanes }
}

// ---------------------------------------------------------------------------
// compilation: logical predicate -> Quack filter + extraction specs
// ---------------------------------------------------------------------------

enum Spec {
    PackU8 { lane: usize, shift: u32, op: Op, c: u32 },
    PackU16 { lane: usize, shift: u32, op: Op, c: i64 },
    Sparse { s: usize, op: Op, c: i64 },
}

struct Ctx {
    arm: Arm,
    specs: Vec<Spec>,
    touched: Vec<u32>,
    planes: usize,
}
impl Ctx {
    fn touch(&mut self, id: u32) {
        if !self.touched.contains(&id) {
            self.touched.push(id);
        }
    }
}

fn qcmp(op: Op, c: i32) -> Cmp {
    match op {
        Op::Lt => Cmp::LtI32(c),
        Op::Eq => Cmp::EqI32(c),
        Op::Ge => Cmp::GeI32(c),
    }
}

fn push_ext(cx: &mut Ctx, spec: Spec) -> Result<Filter, Refusal> {
    if cx.specs.len() >= MAX_EXT {
        return Err(Refusal::TooManyExtractions);
    }
    cx.specs.push(spec);
    cx.planes += 1;
    Ok(Filter::plane(Mask(cx.specs.len() as _)))
}

fn leaf(cx: &mut Ctx, j: usize, op: Op, c: i64) -> Result<Filter, Refusal> {
    let kind = Kind::of(j);
    let as_i32 = |c: i64| i32::try_from(c).map_err(|_| Refusal::ConstOutsideI32 { prop: j, c });
    match (cx.arm, kind) {
        (_, Kind::Full) | (Arm::U, Kind::Small) | (Arm::S, Kind::Small) => {
            let c32 = as_i32(c)?;
            cx.touch(j as u32);
            Ok(Filter::cmp(Col(j as u16), qcmp(op, c32)))
        }
        (Arm::U, Kind::Sparse) => {
            let c32 = as_i32(c)?;
            cx.touch(j as u32);
            cx.planes += 1;
            Ok(Filter::and([
                Filter::cmp(Col(j as u16), qcmp(op, c32)),
                Filter::plane(Mask((1 + j - N_SMALL_END) as _)),
            ]))
        }
        (Arm::P, Kind::Small) => {
            if !(0..=255).contains(&c) {
                return Err(Refusal::ConstOutsidePackedU8 { prop: j, c });
            }
            let i = j - N_FULL;
            let (lane, shift) = (i / 4, 8 * (i % 4) as u32);
            cx.touch(1000 + lane as u32);
            if op == Op::Eq {
                let cmp = Cmp::MatchU32 {
                    pattern: (c as u32) << shift,
                    care: 0xFFu32 << shift,
                };
                return Ok(Filter::cmp(Col((N_FULL + lane) as u16), cmp));
            }
            push_ext(cx, Spec::PackU8 { lane, shift, op, c: c as u32 })
        }
        (Arm::P, Kind::Sparse) => {
            if !(0..=65_534).contains(&c) {
                return Err(Refusal::ConstOutsidePackedU16 { prop: j, c });
            }
            let s = j - N_SMALL_END;
            let (lane, shift) = (s / 2, 16 * (s % 2) as u32);
            cx.touch(2000 + lane as u32);
            if op == Op::Eq {
                // stored = value + 1, so an absent row (0) can never match.
                let cmp = Cmp::MatchU32 {
                    pattern: ((c + 1) as u32) << shift,
                    care: 0xFFFFu32 << shift,
                };
                return Ok(Filter::cmp(Col((N_FULL + 8 + lane) as u16), cmp));
            }
            push_ext(cx, Spec::PackU16 { lane, shift, op, c })
        }
        (Arm::S, Kind::Sparse) => {
            let s = j - N_SMALL_END;
            cx.touch(3000 + s as u32);
            push_ext(cx, Spec::Sparse { s, op, c })
        }
    }
}

fn compile(cx: &mut Ctx, p: &Pr) -> Result<Filter, Refusal> {
    match p {
        Pr::Leaf(j, op, c) => leaf(cx, *j, *op, *c),
        Pr::And(v) => {
            let fs = v.iter().map(|x| compile(cx, x)).collect::<Result<Vec<_>, _>>()?;
            Ok(Filter::and(fs))
        }
        Pr::Or(v) => {
            let fs = v.iter().map(|x| compile(cx, x)).collect::<Result<Vec<_>, _>>()?;
            Ok(Filter::or(fs))
        }
    }
}

// ---------------------------------------------------------------------------
// prepared program + evaluation
// ---------------------------------------------------------------------------

struct Prog {
    program: Program,
    scratch: Scratch<'static>,
}

struct Prep {
    prog: Prog,
    specs: Vec<Spec>,
    bufs: Vec<Vec<u64>>,
    lanes_touched: usize,
    planes: usize,
}

fn prepare(v: &ArmView<'_>, st: &Store, p: &Pr) -> Result<Prep, Refusal> {
    let mut cx = Ctx {
        arm: v.arm,
        specs: Vec::new(),
        touched: Vec::new(),
        planes: 0,
    };
    let filter = compile(&mut cx, p)?;
    let q = Query {
        filter,
        agg: Agg::Count,
    };
    let program = lower(&q).map_err(|_| Refusal::LowerFailed)?;
    let scratch = Scratch::for_program(&program, st.n).map_err(|_| Refusal::LowerFailed)?;
    let bufs = (0..cx.specs.len())
        .map(|_| vec![0u64; words_for(st.n)])
        .collect();
    Ok(Prep {
        prog: Prog { program, scratch },
        lanes_touched: cx.touched.len(),
        planes: cx.planes,
        specs: cx.specs,
        bufs,
    })
}

/// Extraction loop: one output word per 64 rows; shift + mask + compare.
fn fill_lane(buf: &mut [u64], lane: &[u32], shift: u32, wmask: u32, test: impl Fn(u32) -> bool) {
    for (w, chunk) in lane.chunks(64).enumerate() {
        let mut word = 0u64;
        for (b, &x) in chunk.iter().enumerate() {
            word |= u64::from(test((x >> shift) & wmask)) << b;
        }
        buf[w] = word;
    }
}

fn fill(buf: &mut [u64], sp: &Spec, st: &Store) {
    match *sp {
        Spec::PackU8 { lane, shift, op, c } => {
            let l = &st.p_small[lane];
            match op {
                Op::Lt => fill_lane(buf, l, shift, 0xFF, |e| e < c),
                Op::Ge => fill_lane(buf, l, shift, 0xFF, |e| e >= c),
                Op::Eq => fill_lane(buf, l, shift, 0xFF, |e| e == c),
            }
        }
        Spec::PackU16 { lane, shift, op, c } => {
            // e = value + 1 (0 = absent). value < c  <=>  0 < e <= c.
            let l = &st.p_sp[lane];
            let cu = c as u32;
            match op {
                Op::Lt => fill_lane(buf, l, shift, 0xFFFF, |e| e != 0 && e <= cu),
                Op::Ge => fill_lane(buf, l, shift, 0xFFFF, |e| e >= cu + 1),
                Op::Eq => fill_lane(buf, l, shift, 0xFFFF, |e| e == cu + 1),
            }
        }
        Spec::Sparse { s, op, c } => {
            let (pres, vals) = (&st.s_pres[s], &st.s_vals[s]);
            let mut k = 0usize;
            for (w, &pw) in pres.iter().enumerate() {
                let (mut m, mut out) = (pw, 0u64);
                while m != 0 {
                    let b = m.trailing_zeros();
                    if holds(op, i64::from(vals[k]), c) {
                        out |= 1u64 << b;
                    }
                    k += 1;
                    m &= m - 1;
                }
                buf[w] = out;
            }
        }
    }
}

fn eval(v: &ArmView<'_>, st: &Store, pr: &mut Prep) -> usize {
    let Prep {
        prog, specs, bufs, ..
    } = pr;
    for (i, sp) in specs.iter().enumerate() {
        fill(&mut bufs[i], sp, st);
    }
    let masks: [&[u64]; 17] = std::array::from_fn(|i| {
        if i == 0 {
            &st.live[..]
        } else if v.arm == Arm::U {
            &st.u_pres[i - 1][..]
        } else if i - 1 < bufs.len() {
            &bufs[i - 1][..]
        } else {
            &st.live[..]
        }
    });
    let planes = Planes {
        n_rows: st.n,
        masks: &masks,
        lanes: &v.lanes,
    };
    match execute_into(
        &prog.program,
        &planes,
        &Foreign::NONE,
        &mut prog.scratch,
        Out::None,
    )
    .expect("mask-risc runs")
    {
        Value::Count(c) => c,
        other => panic!("expected Count, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// timing
// ---------------------------------------------------------------------------

fn median_us(runs: usize, mut f: impl FnMut()) -> (f64, usize) {
    let t = Instant::now();
    f();
    let first = t.elapsed().as_nanos() as f64 / 1000.0;
    let runs = if first > 20_000.0 { runs.min(11) } else { runs };
    let mut v = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        f();
        v.push(t.elapsed().as_nanos() as f64 / 1000.0);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], runs)
}

fn jf(x: f64) -> String {
    format!("{x:.3}")
}

// ---------------------------------------------------------------------------
// cases
// ---------------------------------------------------------------------------

struct Case {
    id: String,
    k: usize,
    mix: &'static str,
    shape: &'static str,
    tree: Pr,
}

fn case_leaf(mix: &str, i: usize) -> Pr {
    // kind of the i-th leaf, then property + op + constant by kind.
    let kind = match mix {
        "full" => 0,
        "packed" => 1,
        "sparse" => 2,
        _ => i % 3,
    };
    let idx = if mix == "mixed" { i / 3 } else { i };
    match kind {
        0 => {
            let (op, c) = [(Op::Lt, i64::from(i32::MAX / 2)), (Op::Ge, 0), (Op::Lt, 0)][idx % 3];
            Pr::Leaf(idx % N_FULL, op, c)
        }
        1 => {
            let (op, c) = [(Op::Lt, 200), (Op::Ge, 20), (Op::Lt, 230)][idx % 3];
            Pr::Leaf(N_FULL + idx % 32, op, c)
        }
        _ => {
            let (op, c) = [(Op::Ge, 0), (Op::Lt, 800), (Op::Ge, 100)][idx % 3];
            Pr::Leaf(N_SMALL_END + idx % 16, op, c)
        }
    }
}

fn gen_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for k in [1usize, 4, 16] {
        for mix in ["full", "packed", "sparse", "mixed"] {
            let leaves: Vec<Pr> = (0..k).map(|i| case_leaf(mix, i)).collect();
            let tree = if k == 1 {
                leaves.into_iter().next().unwrap()
            } else {
                Pr::And(leaves)
            };
            cases.push(Case {
                id: format!("k{k}_{mix}_and"),
                k,
                mix,
                shape: if k == 1 { "single" } else { "and" },
                tree,
            });
        }
    }
    for k in [4usize, 16] {
        let leaves: Vec<Pr> = (0..k).map(|i| case_leaf("mixed", i)).collect();
        cases.push(Case {
            id: format!("k{k}_mixed_or"),
            k,
            mix: "mixed",
            shape: "or",
            tree: Pr::Or(leaves),
        });
    }
    cases
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

struct Totals {
    gate_failed: usize,
    failures: usize,
}

fn run_once(arm: Arm, st: &Store, p: &Pr) -> Result<usize, Refusal> {
    let v = view(arm, st);
    let mut pr = prepare(&v, st, p)?;
    Ok(eval(&v, st, &mut pr))
}

fn fmt_res(r: &Result<usize, Refusal>) -> String {
    match r {
        Ok(c) => c.to_string(),
        Err(e) => format!("REFUSED({})", e.reason()),
    }
}

/// Refusal and parity checks. Returns the JSON array body.
fn refusal_checks(st: &Store, n: usize, t: &mut Totals) -> String {
    let mut js = String::new();
    let (mut r1_fired, mut r3_fired) = (0usize, 0usize);

    // R1: constant outside a packed field's range. P refuses, U and S answer.
    let r1: [(&str, Pr); 4] = [
        ("p20 < 300", Pr::Leaf(20, Op::Lt, 300)),
        ("p20 = -1", Pr::Leaf(20, Op::Eq, -1)),
        ("p20 >= 256", Pr::Leaf(20, Op::Ge, 256)),
        ("p50 < 70000", Pr::Leaf(50, Op::Lt, 70_000)),
    ];
    for (label, tree) in &r1 {
        let (u, p, s) = (
            run_once(Arm::U, st, tree),
            run_once(Arm::P, st, tree),
            run_once(Arm::S, st, tree),
        );
        let o = oracle_count(tree, n);
        println!(
            "R1 {label:<12} U={} P={} S={} oracle={o}",
            fmt_res(&u),
            fmt_res(&p),
            fmt_res(&s)
        );
        let fired = p.is_err();
        let ok = fired && u.as_ref().is_ok_and(|c| *c == o) && s.as_ref().is_ok_and(|c| *c == o);
        if fired {
            r1_fired += 1;
        }
        if !ok {
            eprintln!("CHECK FAILED R1 {label}");
            t.failures += 1;
        }
        let _ = write!(
            js,
            "{{\"check\":\"R1\",\"query\":\"{label}\",\"U\":\"{}\",\"P\":\"{}\",\"S\":\"{}\",\"oracle\":{o}}},",
            fmt_res(&u),
            fmt_res(&p).replace('"', "'"),
            fmt_res(&s)
        );
    }
    // R1 silence twin: in-range constants must NOT refuse and must be exact.
    for (label, tree) in [
        ("p20 < 200", Pr::Leaf(20, Op::Lt, 200)),
        ("p20 = 255", Pr::Leaf(20, Op::Eq, 255)),
        ("p50 >= 100", Pr::Leaf(50, Op::Ge, 100)),
    ] {
        let p = run_once(Arm::P, st, &tree);
        let o = oracle_count(&tree, n);
        println!("R1-silent {label:<12} P={} oracle={o}", fmt_res(&p));
        if !p.as_ref().is_ok_and(|c| *c == o) {
            eprintln!("CHECK FAILED R1-silent {label}");
            t.failures += 1;
        }
    }

    // R2: a sparse compare never matches absent rows, in any layout.
    let present = (0..n as u64).filter(|&id| lv(id, 50).is_some()).count();
    let tree = Pr::Leaf(50, Op::Ge, 0);
    for arm in [Arm::U, Arm::P, Arm::S] {
        let r = run_once(arm, st, &tree);
        println!("R2 {} count(p50 >= 0)={} presence={present} N={n}", arm.name(), fmt_res(&r));
        let ok = r.as_ref().is_ok_and(|c| *c == present) && present < n;
        if !ok {
            eprintln!("CHECK FAILED R2 {}", arm.name());
            t.failures += 1;
        }
    }
    let _ = write!(js, "{{\"check\":\"R2\",\"presence\":{present},\"n\":{n}}},");

    // R3: out-of-range sparse value refused when the P lane is built.
    let bad = pack_sparse_pair(Some(70_000), Some(5));
    let neg = pack_sparse_pair(Some(1), Some(-3));
    let good = pack_sparse_pair(Some(999), None);
    println!("R3 70000 -> {:?}, -3 -> {:?}, (999,absent) -> {:?}", bad, neg, good);
    if bad.is_err() {
        r3_fired += 1;
    }
    if neg.is_err() {
        r3_fired += 1;
    }
    let good_ok = good.as_ref().is_ok_and(|w| *w == 1000);
    if r3_fired != 2 || !good_ok {
        eprintln!("CHECK FAILED R3 (fired {r3_fired}/2, valid-encodes {good_ok})");
        t.failures += 1;
    }
    let _ = write!(js, "{{\"check\":\"R3\",\"fired\":{r3_fired},\"valid_encodes\":{good_ok}}}");

    if r1_fired != r1.len() {
        eprintln!("CHECK FAILED: R1 refusal fired {r1_fired}/{} times", r1.len());
        t.failures += 1;
    }
    js
}

fn run_size(out_root: &str, n: usize) -> bool {
    println!("== N = {n} ==");
    let mut t = Totals {
        gate_failed: 0,
        failures: 0,
    };
    let words = words_for(n);
    let mut live = vec![0u64; words];
    for r in 0..n {
        live[r >> 6] |= 1u64 << (r & 63);
    }

    let t0 = Instant::now();
    let (u_lanes, u_pres) = build_u(n);
    let build_u_us = t0.elapsed().as_nanos() as f64 / 1000.0;
    let t0 = Instant::now();
    let (p_small, p_sp) = build_p(n);
    let build_p_us = t0.elapsed().as_nanos() as f64 / 1000.0;
    let t0 = Instant::now();
    let (s_pres, s_vals) = build_s(n);
    let build_s_us = t0.elapsed().as_nanos() as f64 / 1000.0;
    let compact: usize = s_vals.iter().map(Vec::len).sum();
    let st = Store {
        n,
        live,
        u_lanes,
        u_pres,
        p_small,
        p_sp,
        s_pres,
        s_vals,
    };
    let lane_b = n * 4;
    let bytes_u = 64 * lane_b + 16 * words * 8;
    let bytes_p = 32 * lane_b;
    let bytes_s = 48 * lane_b + 16 * words * 8 + compact * 4;
    println!(
        "COLD (reported, not gate) build us: U {build_u_us:.0} P {build_p_us:.0} S {build_s_us:.0} \
         (P aliases U's 16 full lanes, S aliases U's 48 lanes; those are not charged again)"
    );
    println!(
        "storage bytes: U {bytes_u} (64 lanes, over budget) P {bytes_p} (32 lanes) S {bytes_s} \
         (48 lanes + 16 presence + {compact} compact values)"
    );

    let checks_js = refusal_checks(&st, n, &mut t);

    let cases = gen_cases();
    let mut recs = String::new();
    for (ci, c) in cases.iter().enumerate() {
        let o = oracle_count(&c.tree, n);
        println!("case {:<16} K={:<2} mix={:<6} oracle count {o}", c.id, c.k, c.mix);
        for arm in [Arm::U, Arm::P, Arm::S] {
            let v = view(arm, &st);
            let t0 = Instant::now();
            let prep = prepare(&v, &st, &c.tree);
            let prep_us = t0.elapsed().as_nanos() as f64 / 1000.0;
            let mut pr = match prep {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("UNEXPECTED REFUSAL {} {}: {}", c.id, arm.name(), e.reason());
                    t.failures += 1;
                    continue;
                }
            };
            let got = eval(&v, &st, &mut pr);
            if got != o {
                eprintln!("MISMATCH {} {}: oracle {o}, got {got}", c.id, arm.name());
                t.failures += 1;
            }
            let b0 = alloc_bytes();
            std::hint::black_box(eval(&v, &st, &mut pr));
            let alloc = alloc_bytes() - b0;
            let (warm, runs) = median_us(31, || {
                std::hint::black_box(eval(&v, &st, &mut pr));
            });
            let gate = warm >= 1000.0;
            if gate {
                println!("FAILED-GATE n={n} {} {} warm {warm:.1} us", c.id, arm.name());
                t.gate_failed += 1;
            }
            println!(
                "  {} lanes {:>2} planes {:>2} ext_loops {:>2} prep {:.0} us (COLD) warm {:.2} us alloc {alloc} B",
                arm.name(),
                pr.lanes_touched,
                pr.planes,
                pr.specs.len(),
                prep_us,
                warm
            );
            if !recs.is_empty() {
                recs.push(',');
            }
            let _ = write!(
                recs,
                "\n{{\"layout\":\"{}\",\"case\":\"{}\",\"k\":{},\"mix\":\"{}\",\"shape\":\"{}\",\
\"count\":{got},\"oracle\":{o},\"prep_us_cold\":{},\"warm_median_us\":{},\"runs\":{runs},\
\"failed_gate\":{gate},\"alloc_bytes_eval\":{alloc},\"lanes_touched\":{},\"planes\":{},\"ext_loops\":{}}}",
                arm.name(),
                c.id,
                c.k,
                c.mix,
                c.shape,
                jf(prep_us),
                jf(warm),
                pr.lanes_touched,
                pr.planes,
                pr.specs.len()
            );
        }
        let _ = ci;
    }

    let dir = format!("{out_root}/project/n{n}");
    std::fs::create_dir_all(&dir).expect("create out dir");
    let _ = (u_lane_note(), Op::Eq.name());
    let js = format!(
        "{{\"n\":{n},\"seed\":{SEED},\"build_us_cold\":{{\"U\":{},\"P\":{},\"S\":{}}},\
\"storage_bytes\":{{\"U\":{bytes_u},\"P\":{bytes_p},\"S\":{bytes_s}}},\"s_compact_values\":{compact},\
\"checks\":[{checks_js}],\"records\":[{recs}\n]}}\n",
        jf(build_u_us),
        jf(build_p_us),
        jf(build_s_us)
    );
    std::fs::write(format!("{dir}/project.json"), js).expect("write project.json");
    println!(
        "N={n}: {} FAILED-GATE values (warm >= 1000 us, reported), {} failures",
        t.gate_failed, t.failures
    );
    t.failures == 0
}

fn u_lane_note() -> &'static str {
    "U needs 64 lanes + 16 presence planes; over the 32-lane budget"
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1].starts_with("--") {
        eprintln!("usage: project <OUT> [--n 1000000|10000000]...");
        std::process::exit(2);
    }
    let out = args[1].clone();
    let mut sizes: Vec<usize> = Vec::new();
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--n" && i + 1 < args.len() {
            sizes.push(args[i + 1].parse().expect("--n takes an integer"));
            i += 2;
        } else {
            eprintln!("unknown argument {}", args[i]);
            std::process::exit(2);
        }
    }
    if sizes.is_empty() {
        sizes = vec![1_000_000, 10_000_000];
    }
    let mut ok = true;
    for n in sizes {
        ok &= run_size(&out, n);
    }
    if !ok {
        println!("PROJECT FAILED: answer mismatch or a refusal did not fire");
        std::process::exit(1);
    }
    println!("PROJECT ALL ANSWERS MATCH THE ORACLE, ALL REFUSALS FIRED");
}
