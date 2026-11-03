//! LAB: D-GLOBAL-WHERE P2, composition-scaling probe ("beyond 64K = repeated joins").
//!
//! Lab only. Nothing here is production, and nothing here is a compiled or run
//! result until the orchestrator builds and runs it.
//!
//! QUESTION (operator): beyond 65,536 identities the answer is NOT a new paging
//! engine. It is REPEATED JOINS / COMPOSITION with the same DR algebra and the
//! same addressing: a `WorkingSet` is one `MapTable` of <= 256 DR256 tiles (<=
//! 65,536 ids), and an edge into another set is `CROSS | noderef(set << 16 | coord)`.
//! This probe measures how cost scales with the number of compositions D,
//! independently of the selected cardinality |sel| and of the corpus size N.
//!
//! Independent variables, one at a time from a base point:
//!   * N in {1_000_000, 10_000_000}: must NOT drive any stage except the WHERE scan.
//!   * |sel| in {1_000, 50_000, 200_000, 1_000_000}: D = ceil(|sel| / 65,536).
//!   * D in {1, 2, 4, 8} at fixed |sel| = 50_000 (one selection split into more,
//!     smaller sets), so composition overhead is separated from volume.
//!   * partition: `contig` (chunks of the id-ordered candidates; community blocks
//!     stay together, few cross-set edges) and `scatter` (position mod D; nearly
//!     every edge crosses a set boundary, the worst case for routing).
//!
//! Graph: deterministic (lib `Rng` = SplitMix64), generated per source id, never
//! materialised corpus-wide: out-degree 4..12 (avg 8); 7/8 of the targets fall in
//! the source's own 1024-id block, 1/8 are uniform over N. The WHERE lane gives a
//! whole block the same high bits, so a threshold selects whole blocks (plus one
//! partial block). Edges whose target lies in no set are dropped (counted), as in
//! `Binding::prepare`. Per-set `CoordCsr` only; there is no dense adjacency.
//!
//! Query per case: A = `a < 300`, B = `b < 500` (virtual WHERE over lanes gathered
//! at preparation); W2 (EXISTS / COUNT of x in A with an edge into B), W6 (bag
//! count of 2-paths) and W5 (DISTINCT: y in B with an edge from A, 1 hop; "DISTINCT
//! reachable" is read as this lib workload). Each set k folds with A restricted to
//! set k (all CSRs visible, so cross targets resolve through `tgt_unpack`), then the
//! parts compose: EXISTS = OR, COUNT = `checked_sum` over disjoint contributions,
//! DISTINCT = exact union of the per-set reach masks (never a sum), bag
//! multiplicities preserved (`w6`, checked i64).
//!
//! Stages charged separately: discover (WHERE scan, N-BOUND BY DESIGN), partition,
//! fetch (edges of the set's sources from the generator, standing in for a global
//! relation index), bind (`WorkingSet::build`), route (arm index build + batch
//! routing of both endpoints), prep (CSR + presence + lanes + masks + fold plan),
//! fold (per set and total, warm), compose (warm), plus continuation (the D-th set
//! added to D-1 built sets vs a full rebuild) and `WorkingSet::extend` vs rebuild.
//! `StageTimes` carries Discover/Bind/Project(prep)/Fold; the lib has no slot for
//! the others, which use local timers.
//!
//! ORACLE: `lib::oracle` over the global edge list of the selected ids (generator
//! run again, edges kept only when the target is selected), outside every timed
//! region, never reading a working set. A composed answer that differs exits 1.
//!
//! REFUSALS (each exercised and checked to fire): an id in two sets -> `union_check`
//! Err; `checked_sum` overflow -> None (checked_sum does NOT see overlap, only
//! overflow; the overlap refusal is union_check, and the wrong sum is shown);
//! stale binding after a set is replaced -> `Binding::guard` Stale (and the arm-C
//! index refuses the new frame); > 65,536 ids into one set -> TooMany.
//!
//! NOT CLAIMED: absolute speed, that the generated graph resembles a real
//! corpus, that `fetch` equals a real relation-index read, that these numbers hold
//! at 30M, or anything about DuckDB. Cold stages (build/route/prep/continuation)
//! are reported, never gated. A warm fold >= 1000 us prints FAILED-GATE (counted,
//! reported; it does not change the exit code).
//!
//! Usage: `compose <OUT> [--n 1000000|10000000]... [--arm A|B|C]` (default: both
//! sizes, arm C). Writes `<OUT>/compose/n<N>/compose.json` (and
//! `<OUT>/compose/scaling.json` when two sizes ran). Exit 1 on any mismatch.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::Instant;

use d_global_where_p1_probe::*;
use lance_graph_dt64::dr::{Frame, MapRefusal};
use lance_graph_quack::{Agg, Cmp};

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

const A_LT: i32 = 300;
const B_LT: i32 = 500;
const CAP: usize = 65_536;

fn die(msg: String) -> ! {
    eprintln!("compose: FATAL {msg}");
    std::process::exit(1)
}

// ---------------------------------------------------------------------------
// deterministic corpus: WHERE lane, value lanes, edge generator
// ---------------------------------------------------------------------------

fn mix(x: u64, salt: u64) -> u64 {
    let mut r = Rng::new(x ^ salt.wrapping_mul(0xD6E8_FEB8_6659_FD93));
    r.next_u64()
}

/// WHERE lane: 21 hash bits of the id's 1024-id block, then the low 10 id bits.
fn where_val(id: u64) -> i32 {
    ((((mix(id >> 10, 0xB10C) & 0x1F_FFFF) << 10) | (id & 1023)) & 0x7FFF_FFFF) as i32
}
fn a_of(g: u64) -> i32 {
    (mix(g, 0xA) % 1000) as i32
}
fn b_of(g: u64) -> i32 {
    (mix(g, 0xB) % 1000) as i32
}

#[derive(Clone, Copy)]
struct Graph {
    n: u64,
}

impl Graph {
    fn edges_of(&self, s: u64, out: &mut Vec<(u64, u64)>) {
        let deg = 4 + (mix(s, 0x51) % 9) as usize;
        for k in 0..deg {
            let r = mix(s, 0x100 + k as u64);
            let t = if r & 7 != 0 {
                ((s & !1023) + ((r >> 8) % 1024)) % self.n
            } else {
                (r >> 8) % self.n
            };
            out.push((s, t));
        }
    }
}

fn fetch_edges(g: &Graph, members: &[u64]) -> Vec<(u64, u64)> {
    let mut out = Vec::with_capacity(members.len() * 8 + 8);
    for &s in members {
        g.edges_of(s, &mut out);
    }
    out
}

/// Oracle over the global edge list of `ids` (edges kept when the target is selected).
fn oracle_for(g: &Graph, ids: &[u64]) -> Oracle {
    let set: HashSet<u64> = ids.iter().copied().collect();
    let mut edges: Vec<(u64, u64)> = Vec::new();
    let mut tmp: Vec<(u64, u64)> = Vec::new();
    for &s in ids {
        tmp.clear();
        g.edges_of(s, &mut tmp);
        for &(a, b) in &tmp {
            if set.contains(&b) {
                edges.push((a, b));
            }
        }
    }
    oracle(&edges, &|x| a_of(x) < A_LT, &|x| b_of(x) < B_LT, &[])
}

// ---------------------------------------------------------------------------
// selection (discover) and partition
// ---------------------------------------------------------------------------

struct Sel {
    requested: usize,
    ids: Vec<u64>,
    scan_us: f64,
    extract_us: f64,
    oracle: Option<Oracle>,
}

fn discover(lane: &[i32], live: &[u64], n: usize, sel: usize) -> Sel {
    let mut tmp = lane.to_vec();
    let k = sel.min(n - 1);
    tmp.select_nth_unstable(k);
    let t = tmp[k];
    drop(tmp);
    let mut pr = Prepared::new(Cmp::LtI32(t), Agg::Rows, n);
    let mut out = vec![0u64; words_for(n)];
    let runs = if n >= 5_000_000 { 5 } else { 9 };
    let scan = time_runs(runs, || pr.keep(lane, live, &mut out));
    let mut ids: Vec<u64> = Vec::new();
    let ext = time_runs(3, || {
        ids.clear();
        for_each_bit(&out, |r| ids.push(r as u64));
    });
    if popcount(&out) as usize != ids.len() {
        die("discover: mask popcount != extracted ids".to_string());
    }
    Sel {
        requested: sel,
        ids,
        scan_us: stats(&scan).0,
        extract_us: stats(&ext).0,
        oracle: None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Part {
    Contig,
    Scatter,
}
impl Part {
    fn name(self) -> &'static str {
        match self {
            Part::Contig => "contig",
            Part::Scatter => "scatter",
        }
    }
}

fn partition(ids: &[u64], d: usize, part: Part) -> Vec<Vec<u64>> {
    match part {
        Part::Contig => {
            let per = ids.len().div_ceil(d);
            ids.chunks(per).map(|c| c.to_vec()).collect()
        }
        Part::Scatter => {
            let mut v: Vec<Vec<u64>> = (0..d).map(|_| Vec::with_capacity(ids.len() / d + 1)).collect();
            for (i, &g) in ids.iter().enumerate() {
                v[i % d].push(g);
            }
            v
        }
    }
}

// ---------------------------------------------------------------------------
// costs
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Cost {
    st: StageTimes,
    partition: f64,
    fetch: f64,
    route: f64,
}

fn lap<R>(slot: &mut f64, f: impl FnOnce() -> R) -> R {
    let t = Instant::now();
    let r = f();
    *slot += t.elapsed().as_nanos() as f64 / 1000.0;
    r
}

impl Cost {
    fn build_total(&self) -> f64 {
        self.fetch + self.route + self.st.us(Stage::Bind) + self.st.us(Stage::Project)
    }
}

fn med(cs: &[Cost], f: impl Fn(&Cost) -> f64) -> f64 {
    let v: Vec<f64> = cs.iter().map(f).collect();
    stats(&v).0
}

// ---------------------------------------------------------------------------
// routing across sets (arm A/B/C through the lib batch routers)
// ---------------------------------------------------------------------------

fn route_to(arm: Arm, ws: &WorkingSet, frame: &Frame, ix: Option<&CoordIndex>, ep: &[u64], out: &mut [u32]) -> RouteStats {
    match arm {
        Arm::A => route_batch_a_in(&ws.table, frame, &ws.tile_ids, ws.tw(), ep, out)
            .unwrap_or_else(|e| die(format!("route A: {e:?}"))),
        Arm::B => route_batch_b_in(&ws.table, frame, &ws.tile_ids, ws.tw(), ep, out)
            .unwrap_or_else(|e| die(format!("route B: {e:?}"))),
        Arm::C => route_batch_c(ix.unwrap_or_else(|| die("arm C index missing".to_string())), frame, ep, out)
            .unwrap_or_else(|e| die(format!("route C: {e:?}"))),
    }
}

/// Route both endpoints of `edges` (all sources are members of `order[0]`): the
/// own set first, then each remaining set for the still-unmapped targets. The
/// first set that maps an id owns it. Returns (routed pairs, dropped (src, dst gid)).
fn route_edges(
    arm: Arm,
    order: &[usize],
    sets: &[WorkingSet],
    frames: &[Frame],
    idx: &[CoordIndex],
    edges: &[(u64, u64)],
    calls: &mut u64,
) -> (Vec<(u32, u32)>, Vec<(u32, u64)>) {
    let own = order[0];
    let mut ep: Vec<u64> = Vec::with_capacity(edges.len() * 2);
    for &(s, d) in edges {
        ep.push(s);
        ep.push(d);
    }
    let mut out = vec![0u32; ep.len()];
    let st = route_to(arm, &sets[own], &frames[own], idx.get(own), &ep, &mut out);
    *calls += st.locate_calls;
    let mut dst = vec![UNROUTED; edges.len()];
    let mut rem: Vec<usize> = Vec::new();
    for e in 0..edges.len() {
        let (cs, cd) = (out[2 * e], out[2 * e + 1]);
        if cs >= COORD_AMBIGUOUS {
            die(format!("source {:#x} not mapped by its own set {own}", edges[e].0));
        }
        if cd == COORD_AMBIGUOUS {
            die(format!("target {:#x} ambiguous in set {own}", edges[e].1));
        }
        if cd == COORD_UNMAPPED {
            rem.push(e);
        } else {
            dst[e] = noderef(own, cd as usize);
        }
    }
    for &j in &order[1..] {
        if rem.is_empty() {
            break;
        }
        let sub: Vec<u64> = rem.iter().map(|&e| edges[e].1).collect();
        let mut so = vec![0u32; sub.len()];
        let st = route_to(arm, &sets[j], &frames[j], idx.get(j), &sub, &mut so);
        *calls += st.locate_calls;
        let mut next: Vec<usize> = Vec::new();
        for (t, &e) in rem.iter().enumerate() {
            match so[t] {
                COORD_UNMAPPED => next.push(e),
                COORD_AMBIGUOUS => die(format!("target {:#x} ambiguous in set {j}", edges[e].1)),
                c => dst[e] = noderef(j, c as usize),
            }
        }
        rem = next;
    }
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(edges.len());
    let mut dropped: Vec<(u32, u64)> = Vec::new();
    for e in 0..edges.len() {
        let s = noderef(own, out[2 * e] as usize);
        if dst[e] == UNROUTED {
            dropped.push((s, edges[e].1));
        } else {
            pairs.push((s, dst[e]));
        }
    }
    (pairs, dropped)
}

// ---------------------------------------------------------------------------
// preparation: per-set CSR + presence + lanes + masks; the fold plan
// ---------------------------------------------------------------------------

struct SetData {
    g: CoordCsr,
    v: Vec<u64>,
    la: Vec<i32>,
    lb: Vec<i32>,
    ma: Vec<u64>,
    mb: Vec<u64>,
}

fn prep_set(i: usize, ws: &WorkingSet, frame: &Frame, pairs: &[(u32, u32)]) -> SetData {
    let n = ws.n_coords();
    let g = CoordCsr::from_routed(i, n, pairs);
    let v = ws.valid_mask(frame);
    let la = ws.gather_lane(frame, a_of);
    let lb = ws.gather_lane(frame, b_of);
    let mut ma = vec![0u64; words_for(n)];
    let mut mb = vec![0u64; words_for(n)];
    virtual_where(&la, A_LT, &v, &mut ma);
    virtual_where(&lb, B_LT, &v, &mut mb);
    SetData { g, v, la, lb, ma, mb }
}

fn assemble(sets: &[WorkingSet], frames: Vec<Frame>, pairs: &[Vec<(u32, u32)>], stats: BindStats) -> Binding {
    let mut b = Binding {
        frames,
        graphs: Vec::new(),
        valid: Vec::new(),
        lane_a: Vec::new(),
        lane_b: Vec::new(),
        a: Vec::new(),
        b: Vec::new(),
        stats,
    };
    for i in 0..sets.len() {
        let sd = prep_set(i, &sets[i], &b.frames[i], &pairs[i]);
        b.graphs.push(sd.g);
        b.valid.push(sd.v);
        b.lane_a.push(sd.la);
        b.lane_b.push(sd.lb);
        b.a.push(sd.ma);
        b.b.push(sd.mb);
    }
    b
}

/// Reusable fold buffers, sized once per binding: A restricted to each set k, the
/// per-k reach masks, the union accumulator, the per-k parts.
struct Plan {
    ak: Vec<Vec<Vec<u64>>>,
    reach: Vec<Vec<Vec<u64>>>,
    uni: Vec<Vec<u64>>,
    tmp: Vec<Vec<u64>>,
    w2p: Vec<i64>,
    w6p: Vec<i64>,
    w5p: Vec<i64>,
    ovf: bool,
}

impl Plan {
    fn new(b: &Binding) -> Plan {
        let d = b.graphs.len();
        let sizes: Vec<usize> = b.graphs.iter().map(|g| g.n).collect();
        let zero = || sizes.iter().map(|&n| vec![0u64; words_for(n)]).collect::<Vec<_>>();
        let ak: Vec<Vec<Vec<u64>>> = (0..d)
            .map(|k| {
                (0..d)
                    .map(|si| if si == k { b.a[si].clone() } else { vec![0u64; words_for(sizes[si])] })
                    .collect()
            })
            .collect();
        Plan {
            ak,
            reach: (0..d).map(|_| zero()).collect(),
            uni: zero(),
            tmp: zero(),
            w2p: vec![0; d],
            w6p: vec![0; d],
            w5p: vec![0; d],
            ovf: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Composed {
    exists: bool,
    count: Option<i64>,
    bag: Option<i64>,
    distinct: i64,
    sum_distinct: Option<i64>,
}

/// The per-set folds (warm, allocation-free).
fn fold(b: &Binding, p: &mut Plan) {
    let g = &b.graphs;
    p.ovf = false;
    for k in 0..p.ak.len() {
        p.w2p[k] = w2(g, &p.ak[k], &b.b);
        match w6(g, &p.ak[k], None) {
            Some(v) => p.w6p[k] = v,
            None => {
                p.ovf = true;
                p.w6p[k] = 0;
            }
        }
        step(g, &p.ak[k], &mut p.reach[k]);
        p.w5p[k] = and_count(&p.reach[k], &b.b, &mut p.tmp);
    }
}

/// Combine the per-set parts with the proven laws (warm, allocation-free).
fn compose(b: &Binding, p: &mut Plan) -> Composed {
    let exists = p.w2p.iter().any(|&c| c > 0);
    let count = checked_sum(&p.w2p);
    let bag = if p.ovf { None } else { checked_sum(&p.w6p) };
    for u in p.uni.iter_mut() {
        u.fill(0);
    }
    for k in 0..p.reach.len() {
        for si in 0..p.uni.len() {
            ndarray::simd::mask_or_assign(&mut p.uni[si], &p.reach[k][si]);
        }
    }
    let distinct = and_count(&p.uni, &b.b, &mut p.tmp);
    let sum_distinct = checked_sum(&p.w5p);
    Composed { exists, count, bag, distinct, sum_distinct }
}

/// Monolithic reference folds over the same binding (untimed; cross-check only).
fn mono(b: &Binding) -> (i64, i64, Option<i64>) {
    let sizes: Vec<usize> = b.graphs.iter().map(|g| g.n).collect();
    let mut sc = FoldScratch::new(&sizes);
    (w2(&b.graphs, &b.a, &b.b), w5(&b.graphs, &b.a, &b.b, &mut sc), w6(&b.graphs, &b.a, None))
}

fn check(c: &Composed, o: &Oracle, m: (i64, i64, Option<i64>)) -> Vec<String> {
    let mut e = Vec::new();
    if c.count != Some(o.w2) {
        e.push(format!("COUNT {:?} != oracle {}", c.count, o.w2));
    }
    if c.exists != (o.w2 > 0) {
        e.push(format!("EXISTS {} != oracle {}", c.exists, o.w2 > 0));
    }
    if c.distinct != o.w5 {
        e.push(format!("DISTINCT {} != oracle {}", c.distinct, o.w5));
    }
    if c.bag != Some(o.w6) {
        e.push(format!("BAG {:?} != oracle {}", c.bag, o.w6));
    }
    if m.0 != o.w2 || m.1 != o.w5 || m.2 != Some(o.w6) {
        e.push(format!("monolithic fold {m:?} != oracle"));
    }
    e
}

// ---------------------------------------------------------------------------
// the built state: sets, routing indexes, routed pairs, dropped edges, binding
// ---------------------------------------------------------------------------

struct Built {
    arm: Arm,
    src_base: u64,
    sets: Vec<WorkingSet>,
    idx: Vec<CoordIndex>,
    pairs: Vec<Vec<(u32, u32)>>,
    dropped: Vec<Vec<(u32, u64)>>,
    bind: Binding,
    plan: Plan,
}

fn build(parts: &[Vec<u64>], arm: Arm, graph: &Graph, src_base: u64, cost: &mut Cost) -> Built {
    let d = parts.len();
    let sets: Vec<WorkingSet> = cost.st.time(Stage::Bind, || {
        parts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                WorkingSet::build(p, src_base + i as u64).unwrap_or_else(|e| die(format!("WorkingSet::build: {e:?}")))
            })
            .collect()
    });
    let edges: Vec<Vec<(u64, u64)>> = lap(&mut cost.fetch, || parts.iter().map(|p| fetch_edges(graph, p)).collect());
    let frames: Vec<Frame> = sets.iter().map(|s| s.seal()).collect();
    let mut calls = 0u64;
    let (idx, pairs, dropped) = lap(&mut cost.route, || {
        let idx: Vec<CoordIndex> = if arm == Arm::C {
            sets.iter()
                .zip(&frames)
                .map(|(ws, f)| CoordIndex::build(ws, f).unwrap_or_else(|e| die(format!("CoordIndex::build: {e:?}"))))
                .collect()
        } else {
            Vec::new()
        };
        let mut pairs = Vec::with_capacity(d);
        let mut dropped = Vec::with_capacity(d);
        for i in 0..d {
            let order: Vec<usize> = std::iter::once(i).chain((0..d).filter(|&j| j != i)).collect();
            let (p, dr) = route_edges(arm, &order, &sets, &frames, &idx, &edges[i], &mut calls);
            pairs.push(p);
            dropped.push(dr);
        }
        (idx, pairs, dropped)
    });
    let n_edges: usize = edges.iter().map(|e| e.len()).sum();
    let n_dropped: usize = dropped.iter().map(|e| e.len()).sum();
    let stats = BindStats { locate_calls: calls, endpoints: 2 * n_edges, dropped_edges: n_dropped };
    drop(edges);
    let bind = cost.st.time(Stage::Project, || assemble(&sets, frames, &pairs, stats));
    let plan = cost.st.time(Stage::Project, || Plan::new(&bind));
    Built { arm, src_base, sets, idx, pairs, dropped, bind, plan }
}

impl Built {
    /// Continuation: add one more set to an already-built composition. The new set
    /// routes its own sources against every set; the earlier sets' dropped targets
    /// are re-routed against the new set only; earlier CSRs that gained edges are
    /// rebuilt from their routed pairs. Nothing already routed is routed again.
    fn add_set(&mut self, ids: &[u64], graph: &Graph, cost: &mut Cost) {
        let d = self.sets.len();
        let ws = cost.st.time(Stage::Bind, || {
            WorkingSet::build(ids, self.src_base + d as u64).unwrap_or_else(|e| die(format!("add_set build: {e:?}")))
        });
        let frame = ws.seal();
        self.sets.push(ws);
        self.bind.frames.push(frame);
        let edges = lap(&mut cost.fetch, || fetch_edges(graph, ids));
        let mut gained = vec![0usize; d];
        let mut calls = 0u64;
        let (np, nd) = lap(&mut cost.route, || {
            if self.arm == Arm::C {
                let ix = CoordIndex::build(&self.sets[d], &self.bind.frames[d])
                    .unwrap_or_else(|e| die(format!("CoordIndex::build: {e:?}")));
                self.idx.push(ix);
            }
            let order: Vec<usize> = std::iter::once(d).chain(0..d).collect();
            let out = route_edges(self.arm, &order, &self.sets, &self.bind.frames, &self.idx, &edges, &mut calls);
            for i in 0..d {
                let dr = std::mem::take(&mut self.dropped[i]);
                if dr.is_empty() {
                    continue;
                }
                let ep: Vec<u64> = dr.iter().map(|&(_, g)| g).collect();
                let mut so = vec![0u32; ep.len()];
                let st = route_to(self.arm, &self.sets[d], &self.bind.frames[d], self.idx.get(d), &ep, &mut so);
                calls += st.locate_calls;
                let mut keep: Vec<(u32, u64)> = Vec::new();
                for (t, &(s_nr, g)) in dr.iter().enumerate() {
                    match so[t] {
                        COORD_UNMAPPED => keep.push((s_nr, g)),
                        COORD_AMBIGUOUS => die(format!("target {g:#x} ambiguous in new set")),
                        c => {
                            self.pairs[i].push((s_nr, noderef(d, c as usize)));
                            gained[i] += 1;
                        }
                    }
                }
                self.dropped[i] = keep;
            }
            out
        });
        self.pairs.push(np);
        self.dropped.push(nd);
        let (sets, bind, pairs) = (&self.sets, &mut self.bind, &self.pairs);
        cost.st.time(Stage::Project, || {
            let sd = prep_set(d, &sets[d], &bind.frames[d], &pairs[d]);
            bind.graphs.push(sd.g);
            bind.valid.push(sd.v);
            bind.lane_a.push(sd.la);
            bind.lane_b.push(sd.lb);
            bind.a.push(sd.ma);
            bind.b.push(sd.mb);
            for i in 0..d {
                if gained[i] > 0 {
                    bind.graphs[i] = CoordCsr::from_routed(i, sets[i].n_coords(), &pairs[i]);
                }
            }
            bind.stats.locate_calls += calls;
            bind.stats.endpoints += 2 * edges.len();
        });
        self.bind.stats.dropped_edges = self.dropped.iter().map(|v| v.len()).sum();
        self.plan = cost.st.time(Stage::Project, || Plan::new(&self.bind));
    }
}

// ---------------------------------------------------------------------------
// one case
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct CaseSpec {
    sel: usize,
    d: usize,
    part: Part,
}

#[derive(Clone)]
struct Res {
    sel_req: usize,
    sel_got: usize,
    d: usize,
    part: Part,
    discover_scan: f64,
    discover_extract: f64,
    partition: f64,
    fetch: f64,
    bind: f64,
    route: f64,
    prep: f64,
    fold_total: f64,
    fold_per_set: f64,
    compose: f64,
    disjoint: f64,
    cont: f64,
    rebuild: f64,
    retained: usize,
    peak: usize,
    per_set_bytes: usize,
    warm_calls: usize,
    warm_bytes: usize,
    edges: usize,
    dropped: usize,
    cross: usize,
    locate_calls: u64,
    ans: Composed,
    sum_distinct_wrong: bool,
    ok: bool,
}

impl Res {
    fn discover(&self) -> f64 {
        self.discover_scan + self.discover_extract
    }
    /// Everything except discover: the cost once the candidate ids are given.
    fn rest(&self) -> f64 {
        self.partition + self.fetch + self.bind + self.route + self.prep + self.fold_total + self.compose
    }
    fn key(&self) -> (usize, usize, Part) {
        (self.sel_req, self.d, self.part)
    }
}

fn run_case(sel: &mut Sel, spec: &CaseSpec, arm: Arm, graph: &Graph) -> Res {
    let reps = if sel.requested >= 500_000 {
        1
    } else if sel.requested >= 100_000 {
        3
    } else {
        5
    };
    let warm_runs = if sel.requested >= 500_000 { 5 } else { 11 };
    let mut costs: Vec<Cost> = Vec::new();
    let mut built: Option<Built> = None;
    let (mut retained, mut peak) = (0usize, 0usize);
    let mut dcount = spec.d;
    for _ in 0..reps {
        drop(built.take());
        let mut cost = Cost::default();
        let parts = lap(&mut cost.partition, || partition(&sel.ids, spec.d, spec.part));
        dcount = parts.len();
        let live0 = live_bytes();
        reset_peak();
        let b = build(&parts, arm, graph, 1000, &mut cost);
        retained = live_bytes().saturating_sub(live0);
        peak = peak_bytes().saturating_sub(live0);
        built = Some(b);
        costs.push(cost);
    }
    let mut b = built.unwrap_or_else(|| die("no build".to_string()));
    let d = dcount;

    // oracle (cached per selection, outside every timed region)
    let ids_len = sel.ids.len();
    if sel.oracle.is_none() {
        sel.oracle = Some(oracle_for(graph, &sel.ids));
    }
    let orc = sel.oracle.unwrap_or_else(|| die("oracle".to_string()));

    // disjointness guard (cold)
    let refs: Vec<&WorkingSet> = b.sets.iter().collect();
    let t = Instant::now();
    let uc = union_check(&refs);
    let disjoint = t.elapsed().as_nanos() as f64 / 1000.0;
    drop(refs);
    if uc != Ok(ids_len) {
        die(format!("union_check on a disjoint partition returned {uc:?}, expected Ok({ids_len})"));
    }

    // warm: fold, compose, allocations
    fold(&b.bind, &mut b.plan);
    let ans = compose(&b.bind, &mut b.plan);
    let errs = check(&ans, &orc, mono(&b.bind));
    for e in &errs {
        println!("  MISMATCH sel={} D={} {}: {e}", spec.sel, d, spec.part.name());
    }
    let (c0, y0) = (alloc_calls(), alloc_bytes());
    fold(&b.bind, &mut b.plan);
    black_box(compose(&b.bind, &mut b.plan));
    let (warm_calls, warm_bytes) = (alloc_calls() - c0, alloc_bytes() - y0);
    let fold_us = {
        let (bind, plan) = (&b.bind, &mut b.plan);
        stats(&time_runs(warm_runs, || fold(bind, plan))).0
    };
    let compose_us = {
        let (bind, plan) = (&b.bind, &mut b.plan);
        stats(&time_runs(warm_runs, || {
            black_box(compose(bind, plan));
        }))
        .0
    };

    let edges: usize = b.bind.graphs.iter().map(|g| g.edges()).sum();
    let cross: usize = b
        .bind
        .graphs
        .iter()
        .map(|g| g.tgt.iter().filter(|&&t| t & CROSS != 0).count())
        .sum();
    let dropped = b.bind.stats.dropped_edges;
    let locate_calls = b.bind.stats.locate_calls;

    // continuation: the D-th set given D-1 built vs a full rebuild (contig only)
    let (mut cont, mut rebuild) = (f64::NAN, f64::NAN);
    let mut cont_ok = true;
    if d >= 2 && spec.part == Part::Contig {
        let (mut cs, mut rs) = (Vec::new(), Vec::new());
        for r in 0..reps {
            let parts = partition(&sel.ids, spec.d, spec.part);
            let mut c0 = Cost::default();
            let mut base = build(&parts[..d - 1], arm, graph, 2000, &mut c0);
            let mut c1 = Cost::default();
            base.add_set(&parts[d - 1], graph, &mut c1);
            cs.push(c1.build_total());
            if r + 1 == reps {
                fold(&base.bind, &mut base.plan);
                let a2 = compose(&base.bind, &mut base.plan);
                let e2 = check(&a2, &orc, mono(&base.bind));
                for e in &e2 {
                    println!("  MISMATCH (continuation) sel={} D={d}: {e}", spec.sel);
                }
                cont_ok = e2.is_empty();
            }
            drop(base);
            let mut c2 = Cost::default();
            let full = build(&parts, arm, graph, 3000, &mut c2);
            rs.push(c2.build_total());
            drop(full);
        }
        cont = stats(&cs).0;
        rebuild = stats(&rs).0;
    }

    Res {
        sel_req: spec.sel,
        sel_got: ids_len,
        d,
        part: spec.part,
        discover_scan: sel.scan_us,
        discover_extract: sel.extract_us,
        partition: med(&costs, |c| c.partition),
        fetch: med(&costs, |c| c.fetch),
        bind: med(&costs, |c| c.st.us(Stage::Bind)),
        route: med(&costs, |c| c.route),
        prep: med(&costs, |c| c.st.us(Stage::Project)),
        fold_total: fold_us,
        fold_per_set: fold_us / d as f64,
        compose: compose_us,
        disjoint,
        cont,
        rebuild,
        retained,
        peak,
        per_set_bytes: retained / d,
        warm_calls,
        warm_bytes,
        edges,
        dropped,
        cross,
        locate_calls,
        ans,
        sum_distinct_wrong: ans.sum_distinct != Some(ans.distinct),
        ok: errs.is_empty() && cont_ok,
    }
}

// ---------------------------------------------------------------------------
// refusals and the WorkingSet::extend micro
// ---------------------------------------------------------------------------

struct Refusals {
    json: String,
    ok: bool,
}

fn refusals(cand: &[u64], arm: Arm, graph: &Graph) -> Refusals {
    let mut ok = true;
    let mut js = String::new();
    let mut note = |name: &str, fired: bool, paired: bool, detail: String, js: &mut String| {
        let pass = fired && paired;
        ok &= pass;
        println!("  refusal {name}: {} ({detail})", if pass { "FIRES" } else { "FAILED" });
        if !js.is_empty() {
            js.push(',');
        }
        write!(js, "\"{name}\":{{\"fires\":{fired},\"control_passes\":{paired},\"detail\":\"{detail}\"}}").unwrap();
    };

    // 1. more than 65,536 ids into one working set -> TooMany, never truncated
    let big: Vec<u64> = (0..(CAP as u64 + 1)).collect();
    let r_big = WorkingSet::build(&big, 9000);
    let fired = matches!(r_big, Err(WsError::Map(MapRefusal::TooMany)));
    let full = WorkingSet::build(&big[..CAP], 9001);
    let control = full.is_ok();
    let mut ext_fired = false;
    if let Ok(mut w) = full {
        ext_fired = matches!(w.extend(&[999_999_999]), Err(WsError::Map(MapRefusal::TooMany)));
    }
    note(
        "too_many",
        fired && ext_fired,
        control,
        format!("build(65537)->TooMany {fired}; extend(full set)->TooMany {ext_fired}; build(65536) ok {control}"),
        &mut js,
    );

    // 2. an id present in two sets: union_check refuses; the sum is shown wrong
    let c = &cand[..5000.min(cand.len())];
    let overlap_parts = vec![c[..3000].to_vec(), c[2000..5000].to_vec()];
    let mut cost = Cost::default();
    let mut ob = build(&overlap_parts, arm, graph, 7000, &mut cost);
    let refs: Vec<&WorkingSet> = ob.sets.iter().collect();
    let uc = union_check(&refs);
    drop(refs);
    let uc_fired = matches!(uc, Err(g) if c[2000..3000].binary_search(&g).is_ok());
    fold(&ob.bind, &mut ob.plan);
    let wrong = compose(&ob.bind, &mut ob.plan);
    let want = oracle_for(graph, c);
    let sum_wrong = wrong.count != Some(want.w2) && wrong.distinct != want.w5 && wrong.bag != Some(want.w6);
    let disjoint_parts = vec![c[..2500].to_vec(), c[2500..].to_vec()];
    let mut db = build(&disjoint_parts, arm, graph, 7100, &mut cost);
    let drefs: Vec<&WorkingSet> = db.sets.iter().collect();
    let duc = union_check(&drefs);
    drop(drefs);
    fold(&db.bind, &mut db.plan);
    let good = compose(&db.bind, &mut db.plan);
    let control = duc == Ok(5000) && check(&good, &want, mono(&db.bind)).is_empty();
    note(
        "id_in_two_sets",
        uc_fired && sum_wrong,
        control,
        format!(
            "union_check overlapping={uc:?}; naive composed COUNT {:?}/DISTINCT {}/BAG {:?} vs oracle {}/{}/{}; disjoint control union_check={duc:?} and composed==oracle",
            wrong.count, wrong.distinct, wrong.bag, want.w2, want.w5, want.w6
        ),
        &mut js,
    );

    // 3. checked_sum: overflow -> None (it does not see overlap; union_check does)
    let ovf = checked_sum(&[i64::MAX, 1]);
    let fine = checked_sum(&[1, 2, 3]);
    note(
        "checked_sum_overflow",
        ovf.is_none(),
        fine == Some(6),
        format!("checked_sum([MAX,1])={ovf:?}; checked_sum([1,2,3])={fine:?}"),
        &mut js,
    );

    // 4. stale binding / index after a set is replaced
    let s = &cand[..4000.min(cand.len())];
    let sp = vec![s[..2000].to_vec(), s[2000..].to_vec()];
    let mut sb = build(&sp, arm, graph, 7200, &mut cost);
    let refs: Vec<&WorkingSet> = sb.sets.iter().collect();
    let fresh_ok = sb.bind.guard(&refs, true).is_ok();
    drop(refs);
    let f0 = sb.sets[0].seal();
    let mut t0 = sb.sets[0].table.map(&f0, 0).expect("tile 0").globals().to_vec();
    t0.reverse();
    sb.sets[0].replace_tile(0, &t0).expect("replace tile");
    let refs: Vec<&WorkingSet> = sb.sets.iter().collect();
    let stale = sb.bind.guard(&refs, true);
    let off = sb.bind.guard(&refs, false);
    let nf = sb.sets[0].seal();
    let idx_stale = if arm == Arm::C {
        sb.idx[0].lookup(&nf, s[0]) == Err(CRefusal::StaleIndex)
    } else {
        true
    };
    let fired = matches!(stale, Err(BindError::Stale(0))) && idx_stale;
    note(
        "stale_binding",
        fired,
        fresh_ok && off.is_ok(),
        format!("guard after replace_tile={stale:?}; arm-C index refuses new frame {idx_stale}; guard before replace ok {fresh_ok}; guard disabled lets it through {}", off.is_ok()),
        &mut js,
    );

    Refusals { json: format!("{{{js}}}"), ok }
}

fn extend_micro(cand: &[u64]) -> (String, bool) {
    if cand.len() < 60_000 {
        return ("null".to_string(), true);
    }
    let (base, extra) = (&cand[..50_000], &cand[50_000..60_000]);
    let (mut ext, mut reb) = (Vec::new(), Vec::new());
    let mut ok = true;
    let mut kept = true;
    for _ in 0..11 {
        let mut ws = WorkingSet::build(base, 8000).unwrap_or_else(|e| die(format!("{e:?}")));
        let old = ws.seal();
        let c_before = ws.coord_a(&old, base[123]).ok();
        let t = Instant::now();
        let n = ws.extend(extra).unwrap_or_else(|e| die(format!("extend: {e:?}")));
        ext.push(t.elapsed().as_nanos() as f64 / 1000.0);
        let nf = ws.seal();
        kept &= c_before.is_some()
            && c_before == ws.coord_a(&nf, base[123]).ok()
            && ws.coord_a(&nf, extra[0]).ok() == Some(50_000)
            && !old.same_issue(&nf)
            && n > 0;
        let t = Instant::now();
        let w2s = WorkingSet::build(&cand[..60_000], 8001).unwrap_or_else(|e| die(format!("{e:?}")));
        reb.push(t.elapsed().as_nanos() as f64 / 1000.0);
        black_box(w2s);
    }
    ok &= kept;
    let (e, r) = (stats(&ext).0, stats(&reb).0);
    println!(
        "  WorkingSet::extend(+10000 onto 50000) {e:.1}us vs WorkingSet::build(60000) {r:.1}us; old coordinates kept and old frame stale: {kept}"
    );
    (
        format!("{{\"extend_us\":{e:.3},\"rebuild_us\":{r:.3},\"added\":10000,\"base\":50000,\"coords_kept_old_frame_stale\":{kept}}}"),
        ok,
    )
}

// ---------------------------------------------------------------------------
// reporting
// ---------------------------------------------------------------------------

fn jf(v: f64) -> String {
    if v.is_nan() {
        "null".to_string()
    } else {
        format!("{v:.3}")
    }
}
fn cell(v: f64) -> String {
    if v.is_nan() {
        "-".to_string()
    } else {
        format!("{v:.1}")
    }
}

type Get = fn(&Res) -> f64;
fn metrics() -> Vec<(&'static str, Get)> {
    vec![
        ("discover", |r| r.discover()),
        ("partition", |r| r.partition),
        ("fetch", |r| r.fetch),
        ("bind", |r| r.bind),
        ("route", |r| r.route),
        ("prep", |r| r.prep),
        ("fold_total", |r| r.fold_total),
        ("fold_per_set", |r| r.fold_per_set),
        ("compose", |r| r.compose),
        ("continuation", |r| r.cont),
        ("rebuild", |r| r.rebuild),
        ("rest_given_ids", |r| r.rest()),
    ]
}

fn res_json(r: &Res) -> String {
    let mut s = String::new();
    write!(
        s,
        "{{\"sel_requested\":{},\"sel_achieved\":{},\"d\":{},\"partition\":\"{}\",\
\"discover_scan_us\":{},\"discover_extract_us\":{},\"partition_us\":{},\"fetch_us\":{},\"bind_us\":{},\"route_us\":{},\"prep_us\":{},\
\"fold_total_us\":{},\"fold_per_set_us\":{},\"compose_us\":{},\"disjoint_check_us\":{},\"continuation_us\":{},\"rebuild_us\":{},\"rest_given_ids_us\":{},\
\"fold_gate\":\"{}\",\"warm_alloc_calls\":{},\"warm_alloc_bytes\":{},\"retained_bytes\":{},\"peak_bytes\":{},\"per_set_bytes\":{},\
\"edges\":{},\"dropped_edges\":{},\"cross_edges\":{},\"locate_calls\":{},\
\"answers\":{{\"exists\":{},\"count_w2\":{},\"distinct_w5\":{},\"bag_w6\":{},\"sum_of_per_set_distinct_is_wrong\":{}}},\"oracle_ok\":{}}}",
        r.sel_req,
        r.sel_got,
        r.d,
        r.part.name(),
        jf(r.discover_scan),
        jf(r.discover_extract),
        jf(r.partition),
        jf(r.fetch),
        jf(r.bind),
        jf(r.route),
        jf(r.prep),
        jf(r.fold_total),
        jf(r.fold_per_set),
        jf(r.compose),
        jf(r.disjoint),
        jf(r.cont),
        jf(r.rebuild),
        jf(r.rest()),
        gate_label(r.fold_per_set, false).trim(),
        r.warm_calls,
        r.warm_bytes,
        r.retained,
        r.peak,
        r.per_set_bytes,
        r.edges,
        r.dropped,
        r.cross,
        r.locate_calls,
        r.ans.exists,
        r.ans.count.map_or("null".to_string(), |v| v.to_string()),
        r.ans.distinct,
        r.ans.bag.map_or("null".to_string(), |v| v.to_string()),
        r.sum_distinct_wrong,
        r.ok
    )
    .unwrap();
    s
}

fn print_table(n: usize, rs: &[Res], gates: &mut u32) {
    println!("\n== N={n} (all times us; discover = WHERE scan + mask->ids, N-BOUND by design; rest = everything else given the ids) ==");
    println!(
        "{:>8} {:>2} {:<7} {:>9} {:>8} {:>9} {:>8} {:>9} {:>9} {:>9} {:>9} {:>8} {:>10} {:>10} {:>10}  gate",
        "sel", "D", "part", "discover", "partit", "fetch", "bind", "route", "prep", "fold/set", "fold_tot", "compose", "continue", "rebuild", "rest"
    );
    for r in rs {
        let g = gate_label(r.fold_per_set, false);
        if !g.is_empty() {
            *gates += 1;
        }
        println!(
            "{:>8} {:>2} {:<7} {:>9} {:>8} {:>9} {:>8} {:>9} {:>9} {:>9} {:>9} {:>8} {:>10} {:>10} {:>10} {}{}",
            r.sel_got,
            r.d,
            r.part.name(),
            cell(r.discover()),
            cell(r.partition),
            cell(r.fetch),
            cell(r.bind),
            cell(r.route),
            cell(r.prep),
            cell(r.fold_per_set),
            cell(r.fold_total),
            cell(r.compose),
            cell(r.cont),
            cell(r.rebuild),
            cell(r.rest()),
            g,
            if r.ok { "" } else { "  MISMATCH" }
        );
    }
    println!("  (route/prep/bind/fetch/partition/continuation/rebuild are COLD, reported not gated; fold gate is per set, warm)");
    for r in rs {
        println!(
            "  sel={:>8} D={:>2} {:<7} edges {:>9} cross {:>9} ({:>5.1}%) dropped {:>8} locate {:>10} warm allocs {} calls/{} B retained {:>10} B ({} B/set) peak {} B disjoint-check {:.1}us answers exists={} W2={:?} W5={} W6={:?} sum(per-set DISTINCT) wrong={}",
            r.sel_got,
            r.d,
            r.part.name(),
            r.edges,
            r.cross,
            100.0 * r.cross as f64 / r.edges.max(1) as f64,
            r.dropped,
            r.locate_calls,
            r.warm_calls,
            r.warm_bytes,
            r.retained,
            r.per_set_bytes,
            r.peak,
            r.disjoint,
            r.ans.exists,
            r.ans.count,
            r.ans.distinct,
            r.ans.bag,
            r.sum_distinct_wrong
        );
    }
}

fn slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let num: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

/// Slope in D at fixed N and fixed |sel| = 50,000 (OLS, us per additional set).
fn d_slopes(n: usize, rs: &[Res]) -> String {
    let mut js = String::new();
    println!("\n-- N={n}: slope in D at fixed |sel|=50,000 (us per added set; D8/D1 ratio) --");
    for part in [Part::Contig, Part::Scatter] {
        let pts: Vec<&Res> = rs
            .iter()
            .filter(|r| r.sel_req == 50_000 && (r.part == part || r.d == 1))
            .collect();
        if pts.len() < 3 {
            continue;
        }
        let xs: Vec<f64> = pts.iter().map(|r| r.d as f64).collect();
        let mut line = format!("   {:<7} D={:?}:", part.name(), pts.iter().map(|r| r.d).collect::<Vec<_>>());
        let mut one = String::new();
        for (name, get) in metrics() {
            if name == "discover" || name == "fold_per_set" {
                continue;
            }
            let ys: Vec<f64> = pts.iter().map(|r| get(r)).collect();
            if ys.iter().any(|v| v.is_nan()) {
                continue;
            }
            let sl = slope(&xs, &ys);
            let ratio = ys[ys.len() - 1] / ys[0].max(1e-9);
            write!(line, " {name} {sl:+.1}/set x{ratio:.1};").unwrap();
            if !one.is_empty() {
                one.push(',');
            }
            write!(one, "\"{name}\":{{\"slope_us_per_set\":{sl:.3},\"ratio_dmax_over_d1\":{ratio:.3}}}").unwrap();
        }
        println!("{line}");
        if !js.is_empty() {
            js.push(',');
        }
        write!(js, "\"{}\":{{{one}}}", part.name()).unwrap();
    }
    format!("{{{js}}}")
}

fn scaling(all: &[(usize, Vec<Res>)]) -> String {
    if all.len() < 2 {
        return "null".to_string();
    }
    let (lo, hi) = (&all[0], &all[all.len() - 1]);
    println!("\n== SCALING CHECK: ratio of stage cost, N={} over N={} (flat = ~1; a stage > 1.5x and > 100us is CORPUS-BOUND) ==", hi.0, lo.0);
    let mut js = String::new();
    let mut bound: Vec<String> = Vec::new();
    for rl in &lo.1 {
        let Some(rh) = hi.1.iter().find(|r| r.key() == rl.key()) else { continue };
        let mut line = format!("sel={:>8} D={:>2} {:<7}:", rl.sel_got, rl.d, rl.part.name());
        let mut one = String::new();
        for (name, get) in metrics() {
            let (a, b) = (get(rl), get(rh));
            if a.is_nan() || b.is_nan() {
                continue;
            }
            let ratio = b / a.max(1e-9);
            let flag = name != "discover" && ratio > 1.5 && a.max(b) > 100.0;
            if name == "discover" {
                write!(line, " {name} x{ratio:.1}(scan, N-bound expected);").unwrap();
            } else if flag {
                write!(line, " {name} x{ratio:.1} CORPUS-BOUND;").unwrap();
                bound.push(format!("{name}@sel={},D={},{}", rl.sel_got, rl.d, rl.part.name()));
            } else {
                write!(line, " {name} x{ratio:.2};").unwrap();
            }
            if !one.is_empty() {
                one.push(',');
            }
            write!(one, "\"{name}\":{{\"ratio\":{ratio:.3},\"corpus_bound\":{flag}}}").unwrap();
        }
        println!("{line}");
        if !js.is_empty() {
            js.push(',');
        }
        write!(js, "{{\"sel\":{},\"d\":{},\"partition\":\"{}\",\"stages\":{{{one}}}}}", rl.sel_got, rl.d, rl.part.name()).unwrap();
    }
    if bound.is_empty() {
        println!("   no stage other than discover tracks N");
    } else {
        println!("   CORPUS-BOUND findings (reported, not failures): {}", bound.join(", "));
    }
    format!("{{\"n_lo\":{},\"n_hi\":{},\"cases\":[{js}],\"corpus_bound\":{:?}}}", lo.0, hi.0, bound)
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn case_list() -> Vec<CaseSpec> {
    let mut v: Vec<CaseSpec> = Vec::new();
    for sel in [1_000usize, 50_000, 200_000, 1_000_000] {
        let d = sel.div_ceil(CAP);
        v.push(CaseSpec { sel, d, part: Part::Contig });
        if d >= 2 {
            v.push(CaseSpec { sel, d, part: Part::Scatter });
        }
    }
    for d in [2usize, 4, 8] {
        v.push(CaseSpec { sel: 50_000, d, part: Part::Contig });
        v.push(CaseSpec { sel: 50_000, d, part: Part::Scatter });
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1].starts_with("--") {
        eprintln!("usage: compose <OUT> [--n 1000000|10000000]... [--arm A|B|C]");
        std::process::exit(2);
    }
    let out = args[1].clone();
    let mut sizes: Vec<usize> = Vec::new();
    let mut arm = Arm::C;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--n" if i + 1 < args.len() => {
                sizes.push(args[i + 1].parse().expect("--n takes an integer"));
                i += 2;
            }
            "--arm" if i + 1 < args.len() => {
                arm = match args[i + 1].as_str() {
                    "A" => Arm::A,
                    "B" => Arm::B,
                    "C" => Arm::C,
                    other => die(format!("unknown arm {other}")),
                };
                i += 2;
            }
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    if sizes.is_empty() {
        sizes = vec![1_000_000, 10_000_000];
    }
    sizes.sort_unstable();

    let mut all: Vec<(usize, Vec<Res>)> = Vec::new();
    let mut failed = false;
    let mut gates = 0u32;
    for &n in &sizes {
        println!("\n######## N={n}  routing arm {} ########", arm.name());
        let graph = Graph { n: n as u64 };
        let lane: Vec<i32> = (0..n as u64).map(where_val).collect();
        let mut live = vec![u64::MAX; words_for(n)];
        if n % 64 != 0 {
            let last = live.len() - 1;
            live[last] = (1u64 << (n % 64)) - 1;
        }
        let mut sels: Vec<Sel> = Vec::new();
        for sel in [1_000usize, 50_000, 200_000, 1_000_000] {
            if sel < n {
                let s = discover(&lane, &live, n, sel);
                println!(
                    "discover |sel|={} achieved {}: scan {:.1}us + extract {:.1}us (N-bound by design)",
                    sel,
                    s.ids.len(),
                    s.scan_us,
                    s.extract_us
                );
                sels.push(s);
            }
        }
        drop(lane);
        let mut rs: Vec<Res> = Vec::new();
        for spec in case_list() {
            let Some(s) = sels.iter_mut().find(|s| s.requested == spec.sel) else { continue };
            println!("case sel={} D={} {} ...", spec.sel, spec.d, spec.part.name());
            let r = run_case(s, &spec, arm, &graph);
            failed |= !r.ok;
            rs.push(r);
        }
        print_table(n, &rs, &mut gates);
        let slopes = d_slopes(n, &rs);

        println!("\n-- refusals and WorkingSet::extend (N={n}) --");
        let cand = sels
            .iter()
            .find(|s| s.requested == 200_000)
            .map(|s| s.ids.clone())
            .unwrap_or_else(|| die("no 200k selection".to_string()));
        let rf = refusals(&cand, arm, &graph);
        let (ext_json, ext_ok) = extend_micro(&cand);
        failed |= !rf.ok || !ext_ok;

        let dir = format!("{out}/compose/n{n}");
        std::fs::create_dir_all(&dir).expect("create out dir");
        let cases: Vec<String> = rs.iter().map(res_json).collect();
        let js = format!(
            "{{\"n\":{n},\"arm\":\"{}\",\"graph\":\"avg out-degree 8, 7/8 targets in the source's 1024-id block, 1/8 uniform\",\
\"gate\":\"warm fold per set >= 1000us prints FAILED-GATE (reported)\",\"cases\":[{}],\"d_slopes\":{slopes},\"refusals\":{},\"extend_vs_rebuild\":{ext_json}}}\n",
            arm.name(),
            cases.join(","),
            rf.json
        );
        std::fs::write(format!("{dir}/compose.json"), js).expect("write compose.json");
        all.push((n, rs));
    }

    if all.len() >= 2 {
        let sc = scaling(&all);
        let dir = format!("{out}/compose");
        std::fs::create_dir_all(&dir).expect("create out dir");
        std::fs::write(format!("{dir}/scaling.json"), format!("{sc}\n")).expect("write scaling.json");
    }
    println!("\ncompose: {gates} case(s) printed FAILED-GATE on the warm per-set fold (reported, not an exit code)");
    if failed {
        println!("compose: FAILED (an answer differed from the oracle, or a refusal did not fire)");
        std::process::exit(1);
    }
    println!("compose: all composed answers equal the oracle; every refusal fired");
}
