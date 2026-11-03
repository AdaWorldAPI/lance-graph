//! LAB: D-GLOBAL-WHERE P2 Part 5, the criteria-aware WHERE probe (ours side).
//!
//! Lab only. Standalone: it does not use the probe's lib.rs. A K-criteria
//! `WHERE` over resident `i32` lanes c0..c15 (c15 nullable through a resident
//! validity plane), lowered by Quack (`lower` / `lower_fused`, optionally
//! `Filter::and_by_skip` ordered, `Filter::sql_where` for the nullable case)
//! and executed by mask-risc `execute_into`. No tuple, no row-ID list, no
//! persisted intermediate in any timed path. The scalar row loop below is the
//! ORACLE only (it is how `achieved` is computed and how every answer is
//! checked); it is never timed.
//!
//! Usage: `where_k <OUT> [--n 1000000|10000000]...` (default: both sizes).
//! Writes `<OUT>/where/n<N>/cases.json` and `<OUT>/where/n<N>/where_ours.json`.
//! Exit 1 on any answer mismatch against the oracle. A warm value >= 1000 us
//! prints FAILED-GATE (counted, reported, does not change the exit code: it is
//! a gate on the substrate path, not a correctness check).
//!
//! Column formula (identical to the SQL in duckdb/bench_where.py; u64, no
//! overflow because every operand is < 2^31 before each multiply):
//!   f(id, j) = (((id * 2654435761 + j * 40503) % 2147483648) * 1103515245
//!               % 2147483648) % 1000
//!   c_j = f(id, j) for j != 1;   c1 = (c0 + f(id, 1) % 20) % 1000
//!   c15 is NULL (validity bit clear) where f(id, 15) % 10 == 0; the value
//!   lane keeps f(id, 15) there as stale payload, which must never matter.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use lance_graph_mask_risc::{
    execute_into, tile_words_for, words_for, Foreign, LaneRef, Out, Planes, Program, Scratch,
    Value,
};
use lance_graph_quack::{lower, lower_fused, Agg, Cmp, Col, Filter, Mask, Query};

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
// deterministic data
// ---------------------------------------------------------------------------

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

#[inline]
fn f(id: u64, j: u64) -> u64 {
    let a = (id * 2_654_435_761 + j * 40_503) % 2_147_483_648;
    ((a * 1_103_515_245) % 2_147_483_648) % 1000
}

struct Data {
    n: usize,
    cols: Vec<Vec<i32>>,
    live: Vec<u64>,
    valid: Vec<u64>,
}

fn build_data(n: usize) -> Data {
    let words = words_for(n);
    let mut cols: Vec<Vec<i32>> = (0..16).map(|_| Vec::with_capacity(n)).collect();
    let mut valid = vec![0u64; words];
    for id in 0..n as u64 {
        let c0 = f(id, 0);
        for (j, col) in cols.iter_mut().enumerate() {
            let v = if j == 1 {
                (c0 + f(id, 1) % 20) % 1000
            } else {
                f(id, j as u64)
            };
            col.push(v as i32);
        }
        if f(id, 15) % 10 != 0 {
            let r = id as usize;
            valid[r >> 6] |= 1u64 << (r & 63);
        }
    }
    let mut live = vec![0u64; words];
    for r in 0..n {
        live[r >> 6] |= 1u64 << (r & 63);
    }
    Data {
        n,
        cols,
        live,
        valid,
    }
}

struct Env<'a> {
    n: usize,
    masks: [&'a [u64]; 2],
    lanes: Vec<LaneRef<'a>>,
}

impl<'a> Env<'a> {
    fn new(d: &'a Data) -> Env<'a> {
        Env {
            n: d.n,
            masks: [&d.live, &d.valid],
            lanes: d.cols.iter().map(|c| LaneRef::I32(c)).collect(),
        }
    }
    fn planes(&self) -> Planes<'_> {
        Planes {
            n_rows: self.n,
            masks: &self.masks,
            lanes: &self.lanes,
        }
    }
}

// ---------------------------------------------------------------------------
// predicate tree (shared shape with cases.json)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Op::Lt => "lt",
            Op::Le => "le",
            Op::Gt => "gt",
            Op::Ge => "ge",
            Op::Eq => "eq",
            Op::Ne => "ne",
        }
    }
    fn cmp(self, v: i32) -> Cmp {
        match self {
            Op::Lt => Cmp::LtI32(v),
            Op::Le => Cmp::LeI32(v),
            Op::Gt => Cmp::GtI32(v),
            Op::Ge => Cmp::GeI32(v),
            Op::Eq => Cmp::EqI32(v),
            Op::Ne => Cmp::NeI32(v),
        }
    }
    fn holds(self, x: i32, v: i32) -> bool {
        match self {
            Op::Lt => x < v,
            Op::Le => x <= v,
            Op::Gt => x > v,
            Op::Ge => x >= v,
            Op::Eq => x == v,
            Op::Ne => x != v,
        }
    }
}

#[derive(Clone)]
enum Node {
    Leaf { col: u16, op: Op, v: i32 },
    And(Vec<Node>),
    Or(Vec<Node>),
    Not(Box<Node>),
}

fn leaf(col: u16, op: Op, v: i32) -> Node {
    Node::Leaf { col, op, v }
}

fn leaf_count(n: &Node) -> usize {
    match n {
        Node::Leaf { .. } => 1,
        Node::And(c) | Node::Or(c) => c.iter().map(leaf_count).sum(),
        Node::Not(c) => leaf_count(c),
    }
}

fn tree_json(n: &Node) -> String {
    match n {
        Node::Leaf { col, op, v } => format!(
            "{{\"op\":\"cmp\",\"col\":{col},\"cmpop\":\"{}\",\"value\":{v}}}",
            op.name()
        ),
        Node::And(c) | Node::Or(c) => {
            let name = if matches!(n, Node::And(_)) { "and" } else { "or" };
            let parts: Vec<String> = c.iter().map(tree_json).collect();
            format!("{{\"op\":\"{name}\",\"args\":[{}]}}", parts.join(","))
        }
        Node::Not(c) => format!("{{\"op\":\"not\",\"arg\":{}}}", tree_json(c)),
    }
}

/// ORACLE: SQL three-valued evaluation of the tree at one row.
/// 0 = FALSE, 1 = TRUE, 2 = UNKNOWN.
fn eval3(n: &Node, row: usize, d: &Data) -> u8 {
    match n {
        Node::Leaf { col, op, v } => {
            if *col == 15 && (d.valid[row >> 6] >> (row & 63)) & 1 == 0 {
                return 2;
            }
            u8::from(op.holds(d.cols[*col as usize][row], *v))
        }
        Node::And(c) => {
            let mut unk = false;
            for x in c {
                match eval3(x, row, d) {
                    0 => return 0,
                    2 => unk = true,
                    _ => {}
                }
            }
            if unk {
                2
            } else {
                1
            }
        }
        Node::Or(c) => {
            let mut unk = false;
            for x in c {
                match eval3(x, row, d) {
                    1 => return 1,
                    2 => unk = true,
                    _ => {}
                }
            }
            if unk {
                2
            } else {
                0
            }
        }
        Node::Not(c) => match eval3(c, row, d) {
            0 => 1,
            1 => 0,
            _ => 2,
        },
    }
}

/// ORACLE: rows for which the WHERE clause is TRUE (scalar row loop).
fn oracle_count(n: &Node, d: &Data) -> usize {
    (0..d.n).filter(|&r| eval3(n, r, d) == 1).count()
}

// ---------------------------------------------------------------------------
// tree -> Quack filter (plain order, and_by_skip order)
// ---------------------------------------------------------------------------

fn to_filter(n: &Node) -> Filter {
    match n {
        Node::Leaf { col, op, v } => Filter::cmp(Col(*col), op.cmp(*v)),
        Node::And(c) => Filter::and(c.iter().map(to_filter)),
        Node::Or(c) => Filter::or(c.iter().map(to_filter)),
        Node::Not(c) => Filter::negate(to_filter(c)),
    }
}

/// Estimated survival fraction. Assumes uniform columns on 0..999 and
/// independent leaves; it is NOT measured from the data. It is the score
/// source for the `and_by_skip` ordering: the more selective child (the
/// smaller estimate) gets the higher skip score and moves to the front.
fn est(n: &Node) -> f64 {
    let s = match n {
        Node::Leaf { op, v, .. } => {
            let v = f64::from(*v);
            match op {
                Op::Lt => v / 1000.0,
                Op::Le => (v + 1.0) / 1000.0,
                Op::Gt => (999.0 - v) / 1000.0,
                Op::Ge => (1000.0 - v) / 1000.0,
                Op::Eq => 0.001,
                Op::Ne => 0.999,
            }
        }
        Node::And(c) => c.iter().map(est).product(),
        Node::Or(c) => 1.0 - c.iter().map(|x| 1.0 - est(x)).product::<f64>(),
        Node::Not(c) => 1.0 - est(c),
    };
    s.clamp(0.0, 1.0)
}

fn to_filter_skip(n: &Node) -> Filter {
    match n {
        Node::Leaf { .. } => to_filter(n),
        Node::And(c) => Filter::and_by_skip(
            c.iter()
                .map(|x| (((1.0 - est(x)) * 1_000_000.0) as u32, to_filter_skip(x))),
        ),
        Node::Or(c) => Filter::or(c.iter().map(to_filter_skip)),
        Node::Not(c) => Filter::negate(to_filter_skip(c)),
    }
}

fn finish(f: Filter, nullable: bool) -> Filter {
    if nullable {
        f.sql_where(&[(Col(15), Mask(1))])
    } else {
        f
    }
}

// ---------------------------------------------------------------------------
// prepared programs
// ---------------------------------------------------------------------------

struct Prog {
    program: Program,
    scratch: Scratch<'static>,
}

fn mk(filter: &Filter, agg: Agg, n: usize, fused: bool) -> Option<Prog> {
    let q = Query {
        filter: filter.clone(),
        agg,
    };
    let lowered = if fused { lower_fused(&q) } else { lower(&q) };
    let program = match lowered {
        Ok(p) => p,
        Err(e) => {
            eprintln!("lower failed (fused={fused}): {e:?}");
            return None;
        }
    };
    let scratch = Scratch::for_program(&program, n).ok()?;
    Some(Prog { program, scratch })
}

fn run_count(p: &mut Prog, planes: &Planes<'_>) -> usize {
    match execute_into(
        &p.program,
        planes,
        &Foreign::NONE,
        &mut p.scratch,
        Out::None,
    )
    .expect("mask-risc runs")
    {
        Value::Count(c) => c,
        other => panic!("expected Count, got {other:?}"),
    }
}

fn run_any(p: &mut Prog, planes: &Planes<'_>) -> bool {
    match execute_into(
        &p.program,
        planes,
        &Foreign::NONE,
        &mut p.scratch,
        Out::None,
    )
    .expect("mask-risc runs")
    {
        Value::Bool(b) => b,
        other => panic!("expected Bool, got {other:?}"),
    }
}

fn run_rows(p: &mut Prog, planes: &Planes<'_>, out: &mut [u64]) {
    execute_into(
        &p.program,
        planes,
        &Foreign::NONE,
        &mut p.scratch,
        Out::Mask(out),
    )
    .expect("mask-risc runs");
}

/// Search helper: count of a tree through plain `lower` (no oracle).
fn count_node(node: &Node, planes: &Planes<'_>, n: usize) -> usize {
    let mut p = mk(&to_filter(node), Agg::Count, n, false).expect("search lowers");
    run_count(&mut p, planes)
}

// ---------------------------------------------------------------------------
// timing
// ---------------------------------------------------------------------------

/// Median wall time in us over `runs` runs after one warm-up. If the warm-up
/// alone exceeds 20 ms the run count drops to 11 (recorded by the caller).
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
fn jo(x: Option<f64>) -> String {
    x.map_or("null".to_string(), jf)
}

// ---------------------------------------------------------------------------
// case generation
// ---------------------------------------------------------------------------

struct Case {
    id: String,
    k_group: &'static str,
    pop: &'static str,
    shape: String,
    adversarial: bool,
    nullable: bool,
    target: usize,
    tree: Node,
    achieved: usize,
}

fn k_group(k: usize) -> &'static str {
    match k {
        0 | 1 => "1",
        2..=4 => "2-4",
        5..=16 => "5-16",
        _ => "17+",
    }
}

/// Leaf `i` of a grid case: even leaves are `col < v`, odd leaves are
/// `col >= 1000 - v`; both have selectivity `v / 1000`.
fn grid_leaf(col: u16, i: usize, v: i32) -> Node {
    if i % 2 == 0 {
        leaf(col, Op::Lt, v)
    } else {
        leaf(col, Op::Ge, 1000 - v)
    }
}

fn grid_tree(shape: &str, k: usize, cols: &[u16], vs: &[i32]) -> Node {
    let leaves: Vec<Node> = (0..k)
        .map(|i| grid_leaf(cols[i % cols.len()], i, vs[i]))
        .collect();
    if k == 1 {
        return leaves.into_iter().next().unwrap();
    }
    match shape {
        "and" => Node::And(leaves),
        "or" => Node::Or(leaves),
        _ => {
            // mixed: AND of OR-pairs (an odd last leaf stands alone).
            let mut groups: Vec<Node> = Vec::new();
            let mut it = leaves.into_iter();
            while let Some(a) = it.next() {
                match it.next() {
                    Some(b) => groups.push(Node::Or(vec![a, b])),
                    None => groups.push(a),
                }
            }
            Node::And(groups)
        }
    }
}

fn lerr(c: usize, t: usize) -> f64 {
    ((c as f64 + 1.0).ln() - (t as f64 + 1.0).ln()).abs()
}

/// Search the per-leaf constants that bring the achieved count closest to
/// `target`: bisection over one shared constant (every grid tree is monotone
/// non-decreasing in it), then a greedy +-1 refinement per leaf. The count is
/// read from the engine under test; the final answer is re-checked against
/// the scalar oracle by the caller.
fn search(
    planes: &Planes<'_>,
    n: usize,
    shape: &str,
    k: usize,
    cols: &[u16],
    target: usize,
) -> Vec<i32> {
    let eval = |vs: &[i32]| count_node(&grid_tree(shape, k, cols, vs), planes, n);
    if eval(&vec![1000; k]) < target {
        return vec![1000; k];
    }
    let (mut lo, mut hi) = (0i32, 1000i32);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if eval(&vec![mid; k]) >= target {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    let mut best = lo;
    if lo > 0 {
        let c_hi = eval(&vec![lo; k]);
        let c_lo = eval(&vec![lo - 1; k]);
        if lerr(c_lo, target) < lerr(c_hi, target) {
            best = lo - 1;
        }
    }
    let mut vs = vec![best; k];
    let mut cur = eval(&vs);
    for _round in 0..2 {
        for i in 0..k {
            if lerr(cur, target) < 0.03 {
                return vs;
            }
            for dir in [1i32, -1] {
                for _ in 0..6 {
                    let nv = vs[i] + dir;
                    if !(0..=1000).contains(&nv) {
                        break;
                    }
                    let old = vs[i];
                    vs[i] = nv;
                    let c = eval(&vs);
                    if lerr(c, target) < lerr(cur, target) {
                        cur = c;
                    } else {
                        vs[i] = old;
                        break;
                    }
                }
            }
        }
    }
    vs
}

fn explicit(
    id: &str,
    pop: &'static str,
    adv: bool,
    nullable: bool,
    shape: &str,
    tree: Node,
) -> Case {
    Case {
        id: id.to_string(),
        k_group: k_group(leaf_count(&tree)),
        pop,
        shape: shape.to_string(),
        adversarial: adv,
        nullable,
        target: 0,
        tree,
        achieved: 0,
    }
}

fn gen_cases(d: &Data, planes: &Planes<'_>) -> Vec<Case> {
    let n = d.n;
    let mut cases: Vec<Case> = Vec::new();
    // ---- regular grid: K x population x shape ----
    let pops: [(&'static str, usize); 3] = [("tiny", 50), ("medium", n / 100), ("large", n / 2)];
    let mut ci = 0u64;
    for k in [1usize, 3, 8, 17, 24] {
        let shapes: &[&str] = if k == 1 {
            &["single"]
        } else {
            &["and", "or", "mixed"]
        };
        for &shape in shapes {
            for &(pop, target) in &pops {
                ci += 1;
                let mut rng = Rng(SEED ^ ci.wrapping_mul(0x1000_0000_01B3));
                let mut cols: Vec<u16> = (0..15).collect();
                for i in (1..cols.len()).rev() {
                    let j = rng.below(i as u64 + 1) as usize;
                    cols.swap(i, j);
                }
                let vs = search(planes, n, shape, k, &cols, target);
                cases.push(Case {
                    id: format!("k{k}_{pop}_{shape}"),
                    k_group: k_group(k),
                    pop,
                    shape: shape.to_string(),
                    adversarial: false,
                    nullable: false,
                    target,
                    tree: grid_tree(shape, k, &cols, &vs),
                    achieved: 0,
                });
            }
        }
    }
    // ---- adversarial ----
    let lt = |c: u16, v: i32| leaf(c, Op::Lt, v);
    cases.push(explicit(
        "adv_one_everything",
        "adv",
        true,
        false,
        "and",
        Node::And(vec![lt(4, 1000), lt(5, 10)]),
    ));
    cases.push(explicit(
        "adv_one_nothing",
        "adv",
        true,
        false,
        "and",
        Node::And(vec![lt(5, 1), lt(6, 900), lt(7, 900)]),
    ));
    cases.push(explicit(
        "adv_or_one_everything",
        "adv",
        true,
        false,
        "or",
        Node::Or(vec![lt(0, 1000), lt(3, 5)]),
    ));
    cases.push(explicit(
        "adv_or_one_nothing",
        "adv",
        true,
        false,
        "or",
        Node::Or(vec![lt(5, 0), lt(6, 1)]),
    ));
    cases.push(explicit(
        "adv_16_nonselective",
        "adv",
        true,
        false,
        "and",
        Node::And(
            (0..16)
                .map(|i| lt((i % 15) as u16, 985 + (i % 5) as i32))
                .collect(),
        ),
    ));
    let r = (n / 3) as u64;
    let mut collapse: Vec<Node> = (0..15u16)
        .map(|j| {
            let v = if j == 1 {
                (f(r, 0) + f(r, 1) % 20) % 1000
            } else {
                f(r, u64::from(j))
            };
            leaf(j, Op::Eq, v as i32)
        })
        .collect();
    collapse.push(leaf(0, Op::Ge, f(r, 0) as i32));
    cases.push(explicit(
        "adv_16_collapse_one",
        "adv",
        true,
        false,
        "and",
        Node::And(collapse),
    ));
    cases.push(explicit(
        "adv_correlated_2",
        "adv",
        true,
        false,
        "and",
        Node::And(vec![lt(0, 500), lt(1, 520)]),
    ));
    cases.push(explicit(
        "adv_correlated_5",
        "adv",
        true,
        false,
        "and",
        Node::And(vec![lt(0, 500), lt(1, 510), lt(0, 490), lt(1, 530), lt(0, 520)]),
    ));
    cases.push(explicit(
        "adv_contradictory_2",
        "adv",
        true,
        false,
        "and",
        Node::And(vec![lt(0, 100), leaf(0, Op::Ge, 900)]),
    ));
    let mut contra = vec![lt(0, 100), leaf(0, Op::Ge, 900)];
    contra.extend((0..15u16).map(|j| lt(1 + (j % 14), 900)));
    cases.push(explicit(
        "adv_contradictory_17",
        "adv",
        true,
        false,
        "and",
        Node::And(contra),
    ));
    // ---- nullable (through Filter::sql_where over the validity plane) ----
    cases.push(explicit(
        "null_and",
        "null",
        true,
        true,
        "and",
        Node::And(vec![lt(15, 500), lt(0, 500)]),
    ));
    cases.push(explicit(
        "null_not",
        "null",
        true,
        true,
        "not",
        Node::Not(Box::new(lt(15, 500))),
    ));
    cases.push(explicit(
        "null_or",
        "null",
        true,
        true,
        "or",
        Node::Or(vec![lt(15, 100), lt(2, 100)]),
    ));
    cases.push(explicit(
        "null_not_and",
        "null",
        true,
        true,
        "not",
        Node::Not(Box::new(Node::And(vec![lt(15, 500), lt(0, 500)]))),
    ));
    cases.push(explicit(
        "null_and_5",
        "null",
        true,
        true,
        "and",
        Node::And(vec![
            leaf(15, Op::Ge, 200),
            lt(0, 700),
            lt(2, 700),
            lt(3, 900),
            lt(15, 900),
        ]),
    ));
    cases
}

// ---------------------------------------------------------------------------
// measurement
// ---------------------------------------------------------------------------

struct Totals {
    gate_failed: usize,
    mismatches: usize,
}

/// Measure one filter variant: Count, Any, Rows. Returns the JSON object and
/// the Count median (for the best-order pick), or None when the variant does
/// not lower.
fn measure_variant(
    filter: &Filter,
    fused: bool,
    planes: &Planes<'_>,
    n: usize,
    achieved: usize,
    ctx: &str,
    t: &mut Totals,
) -> Option<(String, f64)> {
    let mut pc = mk(filter, Agg::Count, n, fused)?;
    let mut pa = mk(filter, Agg::Any, n, fused)?;
    let mut pr = mk(filter, Agg::Rows, n, fused)?;
    let words = words_for(n);
    let mut out = vec![0u64; words];

    // correctness first
    let c = run_count(&mut pc, planes);
    let a = run_any(&mut pa, planes);
    run_rows(&mut pr, planes, &mut out);
    // Only bits below n_rows are rows; a complemented mask may carry tail bits.
    let mut rows: usize = out.iter().map(|w| w.count_ones() as usize).sum();
    if n % 64 != 0 {
        if let Some(last) = out.last() {
            rows -= (last >> (n % 64)).count_ones() as usize;
        }
    }
    if c != achieved || a != (achieved > 0) || rows != achieved {
        eprintln!(
            "MISMATCH {ctx}: oracle {achieved}, count {c}, any {a}, rows(popcount of Keep) {rows}"
        );
        t.mismatches += 1;
    }

    // temp bytes: allocator delta of one warm eval of each terminal
    let b0 = alloc_bytes();
    std::hint::black_box(run_count(&mut pc, planes));
    let count_alloc = alloc_bytes() - b0;
    let b0 = alloc_bytes();
    std::hint::black_box(run_any(&mut pa, planes));
    let any_alloc = alloc_bytes() - b0;
    let b0 = alloc_bytes();
    run_rows(&mut pr, planes, &mut out);
    let rows_alloc = alloc_bytes() - b0;
    let scratch_bytes = pc.scratch.slots() * tile_words_for(n) * 8;

    let (count_us, rc) = median_us(31, || {
        std::hint::black_box(run_count(&mut pc, planes));
    });
    let (any_us, ra) = median_us(31, || {
        std::hint::black_box(run_any(&mut pa, planes));
    });
    let (rows_us, rr) = median_us(31, || {
        run_rows(&mut pr, planes, &mut out);
    });
    for (what, v) in [("count", count_us), ("any", any_us), ("rows", rows_us)] {
        if v >= 1000.0 {
            println!("FAILED-GATE {ctx} {what} warm {v:.1} us");
            t.gate_failed += 1;
        }
    }
    let json = format!(
        "{{\"count_us\":{},\"any_us\":{},\"rows_us\":{},\"runs\":[{rc},{ra},{rr}],\
\"slots\":{},\"scratch_bytes\":{scratch_bytes},\"alloc_bytes_eval\":{{\"count\":{count_alloc},\"any\":{any_alloc},\"rows\":{rows_alloc}}}}}",
        jf(count_us),
        jf(any_us),
        jf(rows_us),
        pc.scratch.slots()
    );
    Some((json, count_us))
}

fn lower_us(filter: &Filter, fused: bool) -> Option<f64> {
    let q = Query {
        filter: filter.clone(),
        agg: Agg::Count,
    };
    let go = |q: &Query| if fused { lower_fused(q) } else { lower(q) };
    if go(&q).is_err() {
        return None;
    }
    let (m, _) = median_us(31, || {
        let r = go(&q);
        std::hint::black_box(&r);
    });
    Some(m)
}

fn run_size(out_root: &str, n: usize) -> bool {
    println!("== N = {n} ==");
    let t0 = Instant::now();
    let data = build_data(n);
    println!(
        "columns built in {:.2} s (16 x i32 lanes + validity plane)",
        t0.elapsed().as_secs_f64()
    );
    let env = Env::new(&data);
    let planes = env.planes();

    let t0 = Instant::now();
    let mut cases = gen_cases(&data, &planes);
    // oracle + engine cross-check for every case
    let mut totals = Totals {
        gate_failed: 0,
        mismatches: 0,
    };
    for c in cases.iter_mut() {
        c.achieved = oracle_count(&c.tree, &data);
        let k = leaf_count(&c.tree);
        let ours = {
            let mut p = mk(&finish(to_filter(&c.tree), c.nullable), Agg::Count, n, false)
                .expect("case lowers");
            run_count(&mut p, &planes)
        };
        let met = if c.target == 0 {
            "-".to_string()
        } else {
            format!("{}", c.achieved <= c.target * 2 && c.achieved * 2 >= c.target)
        };
        println!(
            "case {:<24} K={k:<2} target {:>8} achieved {:>8} met={met}",
            c.id, c.target, c.achieved
        );
        if ours != c.achieved {
            eprintln!(
                "MISMATCH generation {}: oracle {} vs quack {}",
                c.id, c.achieved, ours
            );
            totals.mismatches += 1;
        }
    }
    println!(
        "generated {} cases in {:.2} s",
        cases.len(),
        t0.elapsed().as_secs_f64()
    );

    let dir = format!("{out_root}/where/n{n}");
    std::fs::create_dir_all(&dir).expect("create out dir");

    // cases.json
    let mut js = String::new();
    write!(js, "{{\"n\":{n},\"seed\":{SEED},\"cases\":[").unwrap();
    for (i, c) in cases.iter().enumerate() {
        if i > 0 {
            js.push(',');
        }
        let met = c.target == 0 || (c.achieved <= c.target * 2 && c.achieved * 2 >= c.target);
        write!(
            js,
            "\n{{\"id\":\"{}\",\"k\":{},\"k_group\":\"{}\",\"pop\":\"{}\",\"shape\":\"{}\",\
\"adversarial\":{},\"nullable\":{},\"target\":{},\"achieved\":{},\"target_met\":{met},\"tree\":{}}}",
            c.id,
            leaf_count(&c.tree),
            c.k_group,
            c.pop,
            c.shape,
            c.adversarial,
            c.nullable,
            c.target,
            c.achieved,
            tree_json(&c.tree)
        )
        .unwrap();
    }
    js.push_str("\n]}\n");
    std::fs::write(format!("{dir}/cases.json"), js).expect("write cases.json");

    // measurement
    let mut ours = String::new();
    write!(ours, "{{\"n\":{n},\"records\":[").unwrap();
    let mask_bytes = words_for(n) * 8;
    for (i, c) in cases.iter().enumerate() {
        let k = leaf_count(&c.tree);
        let plain = finish(to_filter(&c.tree), c.nullable);
        let skip = finish(to_filter_skip(&c.tree), c.nullable);
        let skip_same = plain == skip;
        let lower_plain = lower_us(&plain, false);
        let lower_fused_us = lower_us(&plain, true);
        let mut variants = String::new();
        let mut best: Option<(&str, f64)> = None;
        for (tag, filt, fused) in [
            ("plain", &plain, false),
            ("fused", &plain, true),
            ("skip", &skip, false),
            ("skip_fused", &skip, true),
        ] {
            let ctx = format!("n={n} case={} variant={tag}", c.id);
            if let Some((json, cus)) =
                measure_variant(filt, fused, &planes, n, c.achieved, &ctx, &mut totals)
            {
                if !variants.is_empty() {
                    variants.push(',');
                }
                write!(variants, "\"{tag}\":{json}").unwrap();
                if best.is_none_or(|(_, b)| cus < b) {
                    best = Some((tag, cus));
                }
            } else {
                if !variants.is_empty() {
                    variants.push(',');
                }
                write!(variants, "\"{tag}\":null").unwrap();
            }
        }
        let (best_tag, best_us) = best.map_or(("none", f64::NAN), |(t, u)| (t, u));
        println!(
            "measured {:<24} K={k:<2} achieved {:>8} lower plain {} fused {} us, best count {} = {:.2} us",
            c.id,
            c.achieved,
            jo(lower_plain),
            jo(lower_fused_us),
            best_tag,
            best_us
        );
        if i > 0 {
            ours.push(',');
        }
        write!(
            ours,
            "\n{{\"id\":\"{}\",\"k\":{k},\"k_group\":\"{}\",\"pop\":\"{}\",\"shape\":\"{}\",\
\"adversarial\":{},\"nullable\":{},\"achieved\":{},\"lower_us_plain\":{},\"lower_us_fused\":{},\
\"skip_same_as_plain\":{skip_same},\"best_count_variant\":\"{best_tag}\",\
\"representation\":{{\"count\":\"virtual\",\"any\":\"virtual\",\"rows\":\"transient\"}},\
\"rows_mask_bytes\":{mask_bytes},\"variants\":{{{variants}}}}}",
            c.id,
            c.k_group,
            c.pop,
            c.shape,
            c.adversarial,
            c.nullable,
            c.achieved,
            jo(lower_plain),
            jo(lower_fused_us),
        )
        .unwrap();
    }
    ours.push_str("\n]}\n");
    std::fs::write(format!("{dir}/where_ours.json"), ours).expect("write where_ours.json");
    println!(
        "N={n}: {} FAILED-GATE values (warm >= 1000 us, reported), {} mismatches",
        totals.gate_failed, totals.mismatches
    );
    totals.mismatches == 0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1].starts_with("--") {
        eprintln!("usage: where_k <OUT> [--n 1000000|10000000]...");
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
        println!("WHERE-K FAILED: answer mismatch");
        std::process::exit(1);
    }
    println!("WHERE-K ALL ANSWERS MATCH THE ORACLE");
}
