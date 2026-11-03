//! LAB: shared helpers of the D-GLOBAL-WHERE P0 probe (falsify + bench).
//!
//! Lab only. Nothing here is production, and nothing is a new identity table:
//! identities live in `lance_graph_dt64::dr::{MapTable, LocalMap}`. The types
//! defined here are positional only (`Plane` is a bit matrix over DR
//! coordinates, `Prepared` is a lowered Quack program plus its scratch, `Rng`
//! and `CountingAlloc` are instruments). The id lists (`Vec<u64>`, edge
//! `(u32, u32)` lists) are FIXTURE and ORACLE inputs only; no timed fold reads
//! one.
//!
//! The global allocator is declared by each binary (`#[global_allocator]`),
//! not here; this crate only supplies the counting implementation and the
//! counters.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use lance_graph_dt64::dr::{Frame, LocateRefusal, MapTable, TileId, TileMap};
use lance_graph_mask_risc::{execute_into, Foreign, LaneRef, Out, Planes, Program, Scratch, Value};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Query};

// ---------------------------------------------------------------------------
// counting allocator (the binaries declare the #[global_allocator] static)
// ---------------------------------------------------------------------------

/// A pass-through allocator that counts bytes, calls, live bytes and the peak.
pub struct CountingAlloc;

static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: a pure pass-through to `System`; the atomics are the only addition.
// `realloc` and `alloc_zeroed` use the trait defaults, which go through
// `alloc` and `dealloc`, so they are counted too.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_BYTES.fetch_add(layout.size(), Relaxed);
        ALLOC_CALLS.fetch_add(1, Relaxed);
        let live = LIVE_BYTES.fetch_add(layout.size(), Relaxed) + layout.size();
        PEAK_BYTES.fetch_max(live, Relaxed);
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Relaxed);
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Total bytes handed out so far (never decreases).
pub fn alloc_bytes() -> usize {
    ALLOC_BYTES.load(Relaxed)
}
/// Total allocation calls so far (never decreases).
pub fn alloc_calls() -> usize {
    ALLOC_CALLS.load(Relaxed)
}
/// Bytes currently live.
pub fn live_bytes() -> usize {
    LIVE_BYTES.load(Relaxed)
}
/// Highest live-byte count since the last [`reset_peak`].
pub fn peak_bytes() -> usize {
    PEAK_BYTES.load(Relaxed)
}
/// Restart peak tracking from the current live bytes.
pub fn reset_peak() {
    PEAK_BYTES.store(LIVE_BYTES.load(Relaxed), Relaxed);
}

// ---------------------------------------------------------------------------
// deterministic data
// ---------------------------------------------------------------------------

/// The probe's seed.
pub const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64.
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Global ids in the universe (two DR256 axes' worth).
pub const UNIVERSE: usize = 65_536;
/// `u64` words of a mask over the universe.
pub const UNIVERSE_WORDS: usize = UNIVERSE / 64;
/// Selection A is `a < A_LT`.
pub const A_LT: i32 = 50;
/// Selection B is `b < B_LT`.
pub const B_LT: i32 = 300;

/// The two value lanes over the universe, uniform in `0..1000`.
pub struct Lanes {
    pub a: Vec<i32>,
    pub b: Vec<i32>,
}

pub fn gen_lanes(rng: &mut Rng) -> Lanes {
    let a = (0..UNIVERSE).map(|_| rng.below(1000) as i32).collect();
    let b = (0..UNIVERSE).map(|_| rng.below(1000) as i32).collect();
    Lanes { a, b }
}

/// `n` distinct pseudo-random global ids from the universe.
pub fn pick_domain(rng: &mut Rng, n: usize) -> Vec<u64> {
    let mut all: Vec<u32> = (0..UNIVERSE as u32).collect();
    for i in 0..n {
        let j = i + rng.below((UNIVERSE - i) as u64) as usize;
        all.swap(i, j);
    }
    all[..n].iter().map(|&g| u64::from(g)).collect()
}

/// Edge distributions of the bench.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dist {
    Sparse,
    Medium,
    Dense,
    Skewed,
}

impl Dist {
    pub const ALL: [Dist; 4] = [Dist::Sparse, Dist::Medium, Dist::Dense, Dist::Skewed];
    pub fn name(self) -> &'static str {
        match self {
            Dist::Sparse => "sparse",
            Dist::Medium => "medium",
            Dist::Dense => "dense",
            Dist::Skewed => "skewed",
        }
    }
}

/// A directed edge set over the domain `dom` (global ids), no self loops, no
/// duplicates. `dense_p` is the Bernoulli probability of the dense case.
/// FIXTURE/ORACLE input: a tuple list, never read by a timed fold.
pub fn gen_edges(rng: &mut Rng, dom: &[u64], dist: Dist, dense_p: f64) -> Vec<(u32, u32)> {
    let n = dom.len();
    let g = |i: usize| dom[i] as u32;
    let mut e: Vec<(u32, u32)> = Vec::new();
    match dist {
        Dist::Sparse => {
            for i in 0..n {
                let mut j = rng.below(n as u64) as usize;
                if j == i {
                    j = (j + 1) % n;
                }
                e.push((g(i), g(j)));
            }
        }
        Dist::Medium => {
            let k = 8.min(n - 1);
            for i in 0..n {
                let mut seen: Vec<usize> = Vec::with_capacity(k);
                while seen.len() < k {
                    let j = rng.below(n as u64) as usize;
                    if j != i && !seen.contains(&j) {
                        seen.push(j);
                        e.push((g(i), g(j)));
                    }
                }
            }
        }
        Dist::Dense => {
            for i in 0..n {
                for j in 0..n {
                    if i != j && rng.unit() < dense_p {
                        e.push((g(i), g(j)));
                    }
                }
            }
        }
        Dist::Skewed => {
            // node `perm[r]` has rank r; weight 1/(r+1); total degree ~ 8n.
            let mut perm: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.below(i as u64 + 1) as usize;
                perm.swap(i, j);
            }
            let mut cum: Vec<f64> = Vec::with_capacity(n);
            let mut total = 0.0f64;
            for r in 0..n {
                total += 1.0 / (r as f64 + 1.0);
                cum.push(total);
            }
            for r in 0..n {
                let w = 1.0 / (r as f64 + 1.0);
                let want = ((8.0 * n as f64) * w / total).round() as usize;
                let want = want.min(n - 1);
                let src = perm[r];
                let mut seen: HashSet<usize> = HashSet::new();
                let mut draws = 0usize;
                while seen.len() < want && draws < want * 20 + 50 {
                    draws += 1;
                    let u = rng.unit() * total;
                    let rank = cum.partition_point(|&c| c < u).min(n - 1);
                    let dst = perm[rank];
                    if dst != src && seen.insert(dst) {
                        e.push((g(src), g(dst)));
                    }
                }
            }
        }
    }
    e
}

// ---------------------------------------------------------------------------
// bit helpers
// ---------------------------------------------------------------------------

#[inline]
pub fn bit(m: &[u64], i: usize) -> bool {
    (m[i >> 6] >> (i & 63)) & 1 == 1
}
#[inline]
pub fn set_bit(m: &mut [u64], i: usize) {
    m[i >> 6] |= 1u64 << (i & 63);
}
#[inline]
pub fn clear_bit(m: &mut [u64], i: usize) {
    m[i >> 6] &= !(1u64 << (i & 63));
}
/// Set bits of a mask: `ndarray::simd::popcount_batch_u64`.
#[inline]
pub fn popcount(m: &[u64]) -> u64 {
    ndarray::simd::popcount_batch_u64(m)
}
/// Call `f` with the index of every set bit, in ascending order. No allocation.
#[inline]
pub fn for_each_bit(m: &[u64], mut f: impl FnMut(usize)) {
    for (w, &word) in m.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            f(w * 64 + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
}
/// A mask of `len` low local positions set (a selection tile's local mask).
pub fn prefix_mask(words: usize, len: usize) -> Vec<u64> {
    let mut m = vec![0u64; words];
    for i in 0..len {
        set_bit(&mut m, i);
    }
    m
}

// ---------------------------------------------------------------------------
// presence plane over DR coordinates (LAB)
// ---------------------------------------------------------------------------

/// A square bit matrix over `coords` DR coordinates: row `x` holds the set of
/// coordinates `y` with an edge `x -> y`. The wide coordinate is
/// `ordinal * tile_width + local`. Positional only; it holds no identity.
#[derive(Clone)]
pub struct Plane {
    pub rows: Vec<u64>,
    pub coords: usize,
    pub wpr: usize,
}

impl Plane {
    pub fn new(coords: usize) -> Plane {
        let wpr = coords.div_ceil(64);
        Plane {
            rows: vec![0u64; coords * wpr],
            coords,
            wpr,
        }
    }
    #[inline]
    pub fn row(&self, x: usize) -> &[u64] {
        &self.rows[x * self.wpr..(x + 1) * self.wpr]
    }
    #[inline]
    pub fn set(&mut self, x: usize, y: usize) {
        let wpr = self.wpr;
        set_bit(&mut self.rows[x * wpr..(x + 1) * wpr], y);
    }
    #[inline]
    pub fn clear(&mut self, x: usize, y: usize) {
        let wpr = self.wpr;
        clear_bit(&mut self.rows[x * wpr..(x + 1) * wpr], y);
    }
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> bool {
        bit(self.row(x), y)
    }
    /// The transposed plane (an address map written out, for the XOR check).
    pub fn transpose(&self) -> Plane {
        let mut t = Plane::new(self.coords);
        for x in 0..self.coords {
            for_each_bit(self.row(x), |y| t.set(y, x));
        }
        t
    }
}

/// The wide coordinate of `global` in the frame tiles `tiles` (each `tw`
/// positions wide): `locate`, then `ordinal * tw + local`.
pub fn coord_of(
    table: &MapTable,
    frame: &Frame,
    tiles: &[TileId],
    tw: usize,
    global: u64,
) -> Result<usize, LocateRefusal> {
    let (ordinal, local) = table.locate(frame, tiles, global)?;
    Ok(ordinal as usize * tw + usize::from(local))
}

/// Route a selection tile set into the R frame's coordinates: for every local
/// of every selection tile, local -> global (the pointer table), then
/// `locate` into the R tiles. Writes the domain mask into `out`.
pub fn route_selection(
    table: &MapTable,
    frame: &Frame,
    r_tiles: &[TileId],
    tw: usize,
    sel_tiles: &[TileId],
    out: &mut [u64],
) {
    out.fill(0);
    for &st in sel_tiles {
        let tm = table.map(frame, st).expect("selection tile readable");
        for l in 0..tm.len() {
            let g = tm.global(l as u8).expect("local in range");
            let c = coord_of(table, frame, r_tiles, tw, g).expect("id in R frame");
            set_bit(out, c);
        }
    }
}

/// Project a global Keep mask (one bit per universe id) into domain
/// coordinates: one probe per domain position, read through the R tiles'
/// `globals()` pointer tables.
pub fn project_keep(tiles: &[TileMap<'_>], tw: usize, keep: &[u64], out: &mut [u64]) {
    out.fill(0);
    for (t, tm) in tiles.iter().enumerate() {
        for (l, &g) in tm.globals().iter().enumerate() {
            if bit(keep, g as usize) {
                set_bit(out, t * tw + l);
            }
        }
    }
}

/// The virtual WHERE: test the predicate only at the domain's globals and
/// build the domain mask directly.
pub fn virtual_where(tiles: &[TileMap<'_>], tw: usize, lane: &[i32], lt: i32, out: &mut [u64]) {
    out.fill(0);
    for (t, tm) in tiles.iter().enumerate() {
        for (l, &g) in tm.globals().iter().enumerate() {
            if lane[g as usize] < lt {
                set_bit(out, t * tw + l);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Quack WHERE over the universe (prepared: lowered once, scratch reused)
// ---------------------------------------------------------------------------

/// One Quack query over one `i32` lane of `n_rows` rows, lowered once.
pub struct Prepared {
    program: Program,
    scratch: Scratch<'static>,
    n_rows: usize,
}

impl Prepared {
    /// `Query { filter: lane0 <cmp>, agg }` lowered and scratch-carved.
    pub fn new(cmp: Cmp, agg: Agg, n_rows: usize) -> Prepared {
        let q = Query {
            filter: Filter::cmp(Col(0), cmp),
            agg,
        };
        let program = lower(&q).expect("quack lowers");
        let scratch = Scratch::for_program(&program, n_rows).expect("scratch carves");
        Prepared {
            program,
            scratch,
            n_rows,
        }
    }

    /// `Agg::Rows`: the kept mask lands in `out` (`words_for(n_rows)` long).
    pub fn keep(&mut self, lane: &[i32], live: &[u64], out: &mut [u64]) {
        let masks: [&[u64]; 1] = [live];
        let lanes = [LaneRef::I32(lane)];
        let planes = Planes {
            n_rows: self.n_rows,
            masks: &masks,
            lanes: &lanes,
        };
        execute_into(
            &self.program,
            &planes,
            &Foreign::NONE,
            &mut self.scratch,
            Out::Mask(out),
        )
        .expect("mask-risc runs");
    }

    /// `Agg::Count`.
    pub fn count(&mut self, lane: &[i32], live: &[u64]) -> usize {
        let masks: [&[u64]; 1] = [live];
        let lanes = [LaneRef::I32(lane)];
        let planes = Planes {
            n_rows: self.n_rows,
            masks: &masks,
            lanes: &lanes,
        };
        match execute_into(
            &self.program,
            &planes,
            &Foreign::NONE,
            &mut self.scratch,
            Out::None,
        )
        .expect("mask-risc runs")
        {
            Value::Count(c) => c,
            other => panic!("expected Value::Count, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// timing
// ---------------------------------------------------------------------------

/// Run `f` `runs` times, returning each wall time in microseconds.
pub fn time_runs(runs: usize, mut f: impl FnMut()) -> Vec<f64> {
    let mut out = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        f();
        out.push(t.elapsed().as_nanos() as f64 / 1000.0);
    }
    out
}

/// `(median, p10, p90)` of a sample (sorts a copy).
pub fn stats(samples: &[f64]) -> (f64, f64, f64) {
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |p: f64| s[(((s.len() - 1) as f64) * p).round() as usize];
    (at(0.5), at(0.1), at(0.9))
}

// ---------------------------------------------------------------------------
// scalar oracle (HashSet-based; oracle only)
// ---------------------------------------------------------------------------

/// Answers of every workload from the edge list, with plain sets.
pub struct Oracle {
    pub w1: i64,
    pub w2: i64,
    pub w3: i64,
    pub w4: i64,
    pub w5: i64,
    pub w6: i64,
    pub w7i: i64,
    pub w8: i64,
    pub sel_a: usize,
    pub sel_b: usize,
}

pub fn adjacency(edges: &[(u32, u32)]) -> HashMap<u32, Vec<u32>> {
    let mut m: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(s, d) in edges {
        m.entry(s).or_default().push(d);
    }
    m
}

fn w2_over(lanes: &Lanes, adj: &HashMap<u32, Vec<u32>>) -> i64 {
    adj.iter()
        .filter(|&(&x, _)| lanes.a[x as usize] < A_LT)
        .filter(|&(_, ys)| ys.iter().any(|&y| lanes.b[y as usize] < B_LT))
        .count() as i64
}

pub fn oracle(lanes: &Lanes, dom: &[u64], edges: &[(u32, u32)], deleted: &[(u32, u32)]) -> Oracle {
    let adj = adjacency(edges);
    let w1 = lanes.a.iter().filter(|&&v| v < A_LT).count() as i64;
    let w2 = w2_over(lanes, &adj);
    let w3 = adj
        .iter()
        .filter(|&(&x, _)| lanes.a[x as usize] < A_LT)
        .filter(|&(_, ys)| {
            ys.iter().any(|y| {
                adj.get(y)
                    .is_some_and(|zs| zs.iter().any(|&z| lanes.b[z as usize] < B_LT))
            })
        })
        .count() as i64;
    let mut front: HashSet<u32> = adj
        .keys()
        .copied()
        .filter(|&x| lanes.a[x as usize] < A_LT)
        .collect();
    for _ in 0..3 {
        let mut next: HashSet<u32> = HashSet::new();
        for x in &front {
            if let Some(ys) = adj.get(x) {
                next.extend(ys.iter().copied());
            }
        }
        front = next;
    }
    let w4 = front
        .iter()
        .filter(|&&z| lanes.b[z as usize] < B_LT)
        .count() as i64;
    let mut reach: HashSet<u32> = HashSet::new();
    let mut w6 = 0i64;
    for (&x, ys) in &adj {
        if lanes.a[x as usize] < A_LT {
            for &y in ys {
                reach.insert(y);
                w6 += adj.get(&y).map_or(0, |zs| zs.len() as i64);
            }
        }
    }
    let w5 = reach.len() as i64;
    let sel_a = dom
        .iter()
        .filter(|&&g| lanes.a[g as usize] < A_LT)
        .count();
    let sel_b = dom
        .iter()
        .filter(|&&g| lanes.b[g as usize] < B_LT)
        .count();
    let w7i = dom
        .iter()
        .filter(|&&g| lanes.a[g as usize] < A_LT && lanes.b[g as usize] < B_LT)
        .count() as i64;
    let del: HashSet<(u32, u32)> = deleted.iter().copied().collect();
    let kept: Vec<(u32, u32)> = edges.iter().copied().filter(|e| !del.contains(e)).collect();
    let w8 = w2_over(lanes, &adjacency(&kept));
    Oracle {
        w1,
        w2,
        w3,
        w4,
        w5,
        w6,
        w7i,
        w8,
        sel_a,
        sel_b,
    }
}
