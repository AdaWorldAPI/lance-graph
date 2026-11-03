//! D-DR-FOLD lab comparison. Lab only, own workspace.
//!
//! Query (Boolean, set semantics): for a root `a`, the number of distinct `d`
//! with `∃b,c. R(a,b) ∧ S(b,c) ∧ T(c,d)`. On a forest b, c are functions of d;
//! the DAG case (duplicate paths) is tested separately.
//!
//! One coordinate space for every arm: rows `0..n`, bound by the EXISTING
//! `lance_graph_dt64::dr::{LocalMap, MapTable}` (W256 tiles, wide coordinate
//! `tile * 256 + local`). No second binding store.
//!
//! - A: candidate-gated Arm L. One mask-risc `Program` (frozen IR, unchanged):
//!   `s0 = cand under T_in; s1 = Gather(ib→R_in) under s0;
//!    s0 = Gather(ic→S_in) under s1; Count(s0)`.
//!   `cand` is `Pred::Range` on the ordered layout, `Pred::EqU32` on a root
//!   lane on the shuffled one.
//! - B: A, with the binding (map + pointer certification) prepared once and
//!   reused across queries and presence generations. Same hot fold as A; it
//!   measures amortization only.
//! - C: vector × tile over 256-wide row chunks, `ndarray::simd` directly,
//!   outside mask-risc. Frontiers are transient tile registers; no composed
//!   relation (no C = S·T) is formed or stored.
//!
//! No-ms rule: any cell of 1 ms or more prints FAILED-GATE.

use lance_graph_dt64::dr::{Frame, LocalMap, MapRefusal, MapTable, TileId, Width};
use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, ForeignPlane, LaneRef, MaskOp, Operand, Out, Planes, Pred,
    Program, Scratch, Terminal, Value,
};
use ndarray::simd::{mask_or_assign, popcount_batch_u64};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

// ------------------------------------------------------------- allocation count
struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size() as u64, Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(n as u64, Relaxed);
        System.realloc(p, l, n)
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;
fn alloc_snapshot() -> (u64, u64) {
    (ALLOCS.load(Relaxed), BYTES.load(Relaxed))
}

// ------------------------------------------------------------- fixture
const TILE: usize = 256;

/// A forest of `roots` trees, depth 3 below the root, fanout `f`. Global ids in
/// DFS order. `pres[g]` = the edge `parent(g) → g` is present (R at level 1,
/// S at level 2, T at level 3).
struct Fx {
    level: Vec<u8>,
    parent: Vec<u32>,
    pres: Vec<bool>,
    roots: Vec<u32>,
    /// Extra T edges (c, d) beyond the tree edge: duplicate paths (DAG case).
    extra_t: Vec<(u32, u32)>,
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn forest(roots: usize, f: usize, keep_pct: u64, seed: u64) -> Fx {
    forest_lv(roots, f, [keep_pct; 3], seed)
}

/// `keep[l-1]` = presence probability (%) of the relation whose child level is `l`.
fn forest_lv(roots: usize, f: usize, keep: [u64; 3], seed: u64) -> Fx {
    let mut rng = Lcg(seed);
    let (mut level, mut parent, mut pres, mut rs) = (vec![], vec![], vec![], vec![]);
    let mut push = |lv: u8, p: u32, rng: &mut Lcg| -> u32 {
        let g = level.len() as u32;
        level.push(lv);
        parent.push(p);
        pres.push(lv > 0 && rng.below(100) < keep[lv as usize - 1]);
        g
    };
    for _ in 0..roots {
        let a = push(0, u32::MAX, &mut rng);
        rs.push(a);
        for _ in 0..f {
            let b = push(1, a, &mut rng);
            for _ in 0..f {
                let c = push(2, b, &mut rng);
                for _ in 0..f {
                    push(3, c, &mut rng);
                }
            }
        }
    }
    Fx { level, parent, pres, roots: rs, extra_t: vec![] }
}

impl Fx {
    fn n(&self) -> usize {
        self.level.len()
    }
    /// Present edges of the relation whose child level is `lv`, as (parent, child).
    fn edges(&self, lv: u8) -> Vec<(u32, u32)> {
        let mut v: Vec<(u32, u32)> = (0..self.n() as u32)
            .filter(|&g| self.level[g as usize] == lv && self.pres[g as usize])
            .map(|g| (self.parent[g as usize], g))
            .collect();
        if lv == 3 {
            v.extend(self.extra_t.iter().copied());
        }
        v
    }
}

/// Independent oracle: nested loops over the edge sets, global ids only.
fn oracle(fx: &Fx, a: u32) -> u64 {
    let (r, s, t) = (fx.edges(1), fx.edges(2), fx.edges(3));
    let bs: BTreeSet<u32> = r.iter().filter(|e| e.0 == a).map(|e| e.1).collect();
    let cs: BTreeSet<u32> = s.iter().filter(|e| bs.contains(&e.0)).map(|e| e.1).collect();
    let ds: BTreeSet<u32> = t.iter().filter(|e| cs.contains(&e.0)).map(|e| e.1).collect();
    ds.len() as u64
}
/// Witness paths (b, c) per d, summed: the Bag count (EXO-108), for contrast only.
fn oracle_paths(fx: &Fx, a: u32) -> u64 {
    let (r, s, t) = (fx.edges(1), fx.edges(2), fx.edges(3));
    let mut n = 0u64;
    for &(x, b) in &r {
        if x != a { continue; }
        for &(b2, c) in &s {
            if b2 != b { continue; }
            n += t.iter().filter(|e| e.0 == c).count() as u64;
        }
    }
    n
}

// ------------------------------------------------------------- layout + binding
#[derive(Clone, Copy, PartialEq)]
enum Locality { Ordered, Shuffled }

struct Layout2 {
    row: Vec<u32>, // global → row
    inv: Vec<u32>, // row → global
}
fn layout(n: usize, loc: Locality, seed: u64) -> Layout2 {
    let mut inv: Vec<u32> = (0..n as u32).collect();
    if loc == Locality::Shuffled {
        let mut rng = Lcg(seed ^ 0xD15C);
        for i in (1..n).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            inv.swap(i, j);
        }
    }
    let mut row = vec![0u32; n];
    for (r, &g) in inv.iter().enumerate() {
        row[g as usize] = r as u32;
    }
    Layout2 { row, inv }
}

/// The binding: the existing MapTable, one W256 tile per 256 rows.
fn bind(l: &Layout2) -> Result<(MapTable, Frame, Vec<TileId>), MapRefusal> {
    let mut table = MapTable::new(1, 1);
    let mut changes = vec![];
    let mut tiles = vec![];
    for (t, chunk) in l.inv.chunks(TILE).enumerate() {
        let globals: Vec<u64> = chunk.iter().map(|&g| u64::from(g)).collect();
        changes.push((t as TileId, Some(LocalMap::prepare(Width::W256, &globals)?)));
        tiles.push(t as TileId);
    }
    table.advance(1, 1, changes);
    let frame = table.seal();
    Ok((table, frame, tiles))
}
/// Reverse lookup through the binding: which global sits at `row`.
fn global_at(table: &MapTable, frame: &Frame, row: u32) -> Option<u64> {
    let t = row as usize / TILE;
    let m = table.map(frame, t as TileId).ok()?;
    m.global((row as usize % TILE) as u8)
}
fn current(table: &MapTable, frame: &Frame) -> bool {
    frame.same_issue(&table.seal())
}

// ------------------------------------------------------------- witnesses
fn fnv(h: u64, x: u64) -> u64 {
    (h ^ x).wrapping_mul(0x0100_0000_01b3)
}
/// Presence witness minted from SOURCE FACTS: per relation, digest over sorted
/// child globals of the present edges, and the count.
#[derive(Clone, PartialEq, Debug)]
struct PresenceWitness { generation: u64, digest: [u64; 3], pop: [u64; 3] }
fn mint_presence(fx: &Fx, generation: u64) -> PresenceWitness {
    let mut digest = [0u64; 3];
    let mut pop = [0u64; 3];
    for lv in 1..=3u8 {
        let mut kids: Vec<u32> = fx.edges(lv).iter().map(|e| e.1).collect();
        kids.sort_unstable();
        kids.dedup();
        digest[lv as usize - 1] = kids.iter().fold(0xcbf2_9ce4_8422_2325, |h, &g| fnv(h, u64::from(g)));
        pop[lv as usize - 1] = kids.len() as u64;
    }
    PresenceWitness { generation, digest, pop }
}

#[derive(Debug, PartialEq)]
enum Refusal { StaleFrame, Generation, Presence(usize), PointerOrigin, NotFunctional }

// ------------------------------------------------------------- arm A
struct ArmA {
    pres: [Vec<u64>; 3], // R_in, S_in, T_in (planes over rows)
    ib: Vec<u32>,
    ic: Vec<u32>,
    ia: Vec<u32>,
    frame: Frame,
    n: usize,
}
impl ArmA {
    fn planes(fx: &Fx, l: &Layout2) -> [Vec<u64>; 3] {
        let n = fx.n();
        let mut p = [vec![0u64; words_for(n)], vec![0u64; words_for(n)], vec![0u64; words_for(n)]];
        for lv in 1..=3u8 {
            for (_, g) in fx.edges(lv) {
                let r = l.row[g as usize] as usize;
                p[lv as usize - 1][r / 64] |= 1 << (r % 64);
            }
        }
        p
    }
    /// Mint: build ib/ic/ia from the layout, then certify every pointer of
    /// every d row through the binding (reverse lookup), and refuse a d with
    /// two T parents (a pointer table needs a functional parent).
    fn mint(fx: &Fx, l: &Layout2, table: &MapTable, frame: Frame) -> Result<ArmA, Refusal> {
        if !current(table, &frame) { return Err(Refusal::StaleFrame); }
        if !fx.extra_t.is_empty() { return Err(Refusal::NotFunctional); }
        let n = fx.n();
        let mut ib: Vec<u32> = (0..n as u32).collect();
        let mut ic = ib.clone();
        let mut ia = ib.clone();
        for g in 0..n {
            let r = l.row[g] as usize;
            let mut root = g as u32;
            while fx.parent[root as usize] != u32::MAX { root = fx.parent[root as usize]; }
            ia[r] = l.row[root as usize];
            if fx.level[g] == 3 {
                let c = fx.parent[g];
                let b = fx.parent[c as usize];
                ic[r] = l.row[c as usize];
                ib[r] = l.row[b as usize];
                if global_at(table, &frame, ic[r]) != Some(u64::from(c))
                    || global_at(table, &frame, ib[r]) != Some(u64::from(b))
                {
                    return Err(Refusal::PointerOrigin);
                }
            }
        }
        Ok(ArmA { pres: Self::planes(fx, l), ib, ic, ia, frame, n })
    }
    fn program(&self, cand: Pred) -> Program {
        Program::new(
            vec![
                MaskOp::Pred { pred: cand, under: Some(Operand::Plane(0)), dst: 0 },
                MaskOp::Gather { lane: 0, foreign: 0, under: Some(Operand::Scratch(0)), dst: 1 },
                MaskOp::Gather { lane: 1, foreign: 1, under: Some(Operand::Scratch(1)), dst: 0 },
            ],
            Terminal::Count { mask: Operand::Scratch(0) },
        )
    }
    fn run(&self, prog: &Program, sc: &mut Scratch<'static>) -> u64 {
        let lanes = [LaneRef::U32(&self.ib), LaneRef::U32(&self.ic), LaneRef::U32(&self.ia)];
        let fps = [
            ForeignPlane { words: &self.pres[0], rows: self.n },
            ForeignPlane { words: &self.pres[1], rows: self.n },
        ];
        let foreign = Foreign { planes: &fps, lanes: &[] };
        let masks: [&[u64]; 1] = [&self.pres[2]];
        let pl = Planes { n_rows: self.n, masks: &masks, lanes: &lanes };
        match execute_into(prog, &pl, &foreign, sc, Out::None).expect("runs") {
            Value::Count(c) => c as u64,
            v => panic!("{v:?}"),
        }
    }
    /// Presence verification against a witness: O(popcount), globals at set bits.
    fn verify(&self, table: &MapTable, w: &PresenceWitness, generation: u64) -> Result<(), Refusal> {
        if !current(table, &self.frame) { return Err(Refusal::StaleFrame); }
        if w.generation != generation { return Err(Refusal::Generation); }
        for p in 0..3 {
            let mut gs = vec![];
            for (wi, &word) in self.pres[p].iter().enumerate() {
                let mut b = word;
                while b != 0 {
                    let r = (wi * 64 + b.trailing_zeros() as usize) as u32;
                    gs.push(global_at(table, &self.frame, r).ok_or(Refusal::Presence(p))? as u32);
                    b &= b - 1;
                }
            }
            gs.sort_unstable();
            let d = gs.iter().fold(0xcbf2_9ce4_8422_2325, |h, &g| fnv(h, u64::from(g)));
            if gs.len() as u64 != w.pop[p] || d != w.digest[p] { return Err(Refusal::Presence(p)); }
        }
        Ok(())
    }
    /// Indexed loads the gated program performs: |cand ∧ T| + |cand ∧ T ∧ R@ib|.
    fn loads(&self, cand: impl Fn(usize) -> bool) -> (u64, u64) {
        let get = |p: &[u64], i: usize| p[i / 64] >> (i % 64) & 1 == 1;
        let (mut l1, mut l2) = (0, 0);
        for i in 0..self.n {
            if cand(i) && get(&self.pres[2], i) {
                l1 += 1;
                if get(&self.pres[0], self.ib[i] as usize) { l2 += 1; }
            }
        }
        (l1, l2)
    }
}

// ------------------------------------------------------------- arm C
/// Present edges as 256-wide row chunks: for source row `r`, the chunks
/// `(target tile, 256-bit bits)` of its present children. A row-ordered
/// block-sparse tile store; the tiles are the presence, read in place.
struct Chunks { off: Vec<u32>, tile: Vec<u32>, bits: Vec<[u64; 4]> }
impl Chunks {
    fn build(fx: &Fx, l: &Layout2) -> Chunks {
        let n = fx.n();
        let mut kids: Vec<Vec<u32>> = vec![vec![]; n];
        for lv in 1..=3u8 {
            for (p, c) in fx.edges(lv) {
                kids[l.row[p as usize] as usize].push(l.row[c as usize]);
            }
        }
        let (mut off, mut tile, mut bits) = (vec![0u32], vec![], vec![]);
        for k in kids.iter_mut() {
            k.sort_unstable();
            k.dedup();
            let mut cur: Option<(u32, [u64; 4])> = None;
            for &c in k.iter() {
                let (t, local) = (c / TILE as u32, c as usize % TILE);
                match &mut cur {
                    Some((ct, b)) if *ct == t => b[local / 64] |= 1 << (local % 64),
                    _ => {
                        if let Some((ct, b)) = cur.take() { tile.push(ct); bits.push(b); }
                        let mut b = [0u64; 4];
                        b[local / 64] |= 1 << (local % 64);
                        cur = Some((t, b));
                    }
                }
            }
            if let Some((ct, b)) = cur { tile.push(ct); bits.push(b); }
            off.push(tile.len() as u32);
        }
        Chunks { off, tile, bits }
    }
}

/// A transient tile register: one 256-bit chunk per tile, cleared by its
/// touched list (cost ∝ tiles touched, not tiles total). Preallocated.
struct Acc { words: Vec<[u64; 4]>, mark: Vec<bool>, touched: Vec<u32> }
impl Acc {
    fn new(tiles: usize) -> Acc {
        Acc { words: vec![[0; 4]; tiles], mark: vec![false; tiles], touched: Vec::with_capacity(tiles) }
    }
    fn clear(&mut self) {
        for &t in &self.touched {
            self.words[t as usize] = [0; 4];
            self.mark[t as usize] = false;
        }
        self.touched.clear();
    }
    #[inline]
    fn or(&mut self, t: u32, b: &[u64; 4]) {
        if !self.mark[t as usize] {
            self.mark[t as usize] = true;
            self.touched.push(t);
        }
        mask_or_assign(&mut self.words[t as usize], b);
    }
}

#[derive(Default, Clone, Copy)]
struct CStats { chunk_ors: u64, tiles_touched: u64 }

struct ArmC { ch: Chunks, frame: Frame, acc_c: Acc, acc_d: Acc }
impl ArmC {
    fn new(fx: &Fx, l: &Layout2, frame: Frame) -> ArmC {
        let tiles = fx.n().div_ceil(TILE);
        ArmC { ch: Chunks::build(fx, l), frame, acc_c: Acc::new(tiles), acc_d: Acc::new(tiles) }
    }
    /// Expand one frontier tile register through the chunks of its set rows.
    fn hop(ch: &Chunks, from: &Acc, to: &mut Acc, st: &mut CStats) {
        for &t in &from.touched {
            let w = from.words[t as usize];
            for (k, &word) in w.iter().enumerate() {
                let mut b = word;
                while b != 0 {
                    let r = t as usize * TILE + k * 64 + b.trailing_zeros() as usize;
                    for i in ch.off[r]..ch.off[r + 1] {
                        to.or(ch.tile[i as usize], &ch.bits[i as usize]);
                        st.chunk_ors += 1;
                    }
                    b &= b - 1;
                }
            }
        }
    }
    /// v_b = R[a]; v_c = v_b · S; v_d = v_c · T; count = |v_d|. No C = S·T.
    fn fold(&mut self, a_row: u32, st: &mut CStats) -> u64 {
        self.acc_c.clear();
        self.acc_d.clear();
        let ch = &self.ch;
        // b frontier: the chunks of a's row (R edges), into acc_d as scratch.
        for i in ch.off[a_row as usize]..ch.off[a_row as usize + 1] {
            self.acc_d.or(ch.tile[i as usize], &ch.bits[i as usize]);
            st.chunk_ors += 1;
        }
        Self::hop(ch, &self.acc_d, &mut self.acc_c, st); // b → c
        // Count the b frontier's tiles before the register is reused for d.
        st.tiles_touched += self.acc_d.touched.len() as u64;
        self.acc_d.clear();
        Self::hop(ch, &self.acc_c, &mut self.acc_d, st); // c → d
        st.tiles_touched += (self.acc_c.touched.len() + self.acc_d.touched.len()) as u64;
        self.acc_d.touched.iter().map(|&t| popcount_batch_u64(&self.acc_d.words[t as usize])).sum()
    }
    fn verify(&self, table: &MapTable, w: &PresenceWitness, generation: u64, fx_level: &[u8], _l: &Layout2) -> Result<(), Refusal> {
        if !current(table, &self.frame) { return Err(Refusal::StaleFrame); }
        if w.generation != generation { return Err(Refusal::Generation); }
        let mut gs: [Vec<u32>; 3] = [vec![], vec![], vec![]];
        for i in 0..self.ch.tile.len() {
            let t = self.ch.tile[i] as usize;
            for (k, &word) in self.ch.bits[i].iter().enumerate() {
                let mut b = word;
                while b != 0 {
                    let r = (t * TILE + k * 64 + b.trailing_zeros() as usize) as u32;
                    let g = global_at(table, &self.frame, r).ok_or(Refusal::Presence(0))? as u32;
                    let lv = fx_level[g as usize] as usize;
                    gs[lv - 1].push(g);
                    b &= b - 1;
                }
            }
        }
        for (p, g) in gs.iter_mut().enumerate() {
            g.sort_unstable();
            g.dedup();
            let d = g.iter().fold(0xcbf2_9ce4_8422_2325, |h, &x| fnv(h, u64::from(x)));
            if g.len() as u64 != w.pop[p] || d != w.digest[p] { return Err(Refusal::Presence(p)); }
        }
        Ok(())
    }
    fn delete(&mut self, src_row: u32, dst_row: u32) {
        let t = dst_row / TILE as u32;
        let local = dst_row as usize % TILE;
        for i in self.ch.off[src_row as usize]..self.ch.off[src_row as usize + 1] {
            if self.ch.tile[i as usize] == t {
                self.ch.bits[i as usize][local / 64] &= !(1 << (local % 64));
            }
        }
    }
}

// ------------------------------------------------------------- timing
const REPS: usize = 51;
fn median_ns(mut f: impl FnMut()) -> f64 {
    let mut v: Vec<f64> = (0..REPS)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[REPS / 2]
}
fn us(ns: f64) -> String {
    if ns >= 1e6 { "FAILED-GATE".into() } else { format!("{:.2}", ns / 1e3) }
}

struct Tally(u32, u32);
impl Tally {
    fn check(&mut self, name: &str, ok: bool, detail: String) {
        if ok { self.0 += 1 } else { self.1 += 1 }
        println!("{} {:<64} {}", if ok { "PASS" } else { "FAIL" }, name, detail);
    }
}

fn cand_pred(loc: Locality, a_row: u32, size: u32) -> Pred {
    match loc {
        Locality::Ordered => Pred::Range { lo: a_row, hi: a_row + size },
        Locality::Shuffled => Pred::EqU32 { lane: 2, v: a_row },
    }
}

// ------------------------------------------------------------- correctness
fn correctness(t: &mut Tally) {
    println!("== correctness (A, C vs independent oracle)");
    for &loc in &[Locality::Ordered, Locality::Shuffled] {
        for &(roots, f, p) in &[(4usize, 4usize, 70u64), (8, 6, 30), (3, 8, 95)] {
            let mut fx = forest(roots, f, p, 7 + roots as u64);
            let l = layout(fx.n(), loc, 11);
            let (table, frame, _) = bind(&l).expect("binds");
            let mut a = ArmA::mint(&fx, &l, &table, frame.clone()).expect("mints");
            let mut c = ArmC::new(&fx, &l, frame.clone());
            let size = (1 + f + f * f + f * f * f) as u32;
            let mut sc = Scratch::new(words_for(fx.n()), 2);
            let tag = format!("{:?} r{roots} f{f} p{p}", if loc == Locality::Ordered { "ord" } else { "shuf" });
            let mut ok = true;
            for &root in &fx.roots {
                let ar = l.row[root as usize];
                let prog = a.program(cand_pred(loc, ar, size));
                let want = oracle(&fx, root);
                let got_a = a.run(&prog, &mut sc);
                let got_c = c.fold(ar, &mut CStats::default());
                ok &= got_a == want && got_c == want;
            }
            t.check(&format!("parity every root, {tag}"), ok, String::new());

            // Deletion through the API: presence changes, binding kept (split).
            let w0 = mint_presence(&fx, 1);
            t.check(&format!("presence witness verifies A, C, {tag}"),
                a.verify(&table, &w0, 1).is_ok() && c.verify(&table, &w0, 1, &fx.level, &l).is_ok(), String::new());
            let mut rng = Lcg(99);
            let mut dels = 0;
            for g in 0..fx.n() {
                if fx.level[g] > 0 && fx.pres[g] && rng.below(100) < 10 {
                    fx.pres[g] = false;
                    let (pr, cr) = (l.row[fx.parent[g] as usize], l.row[g] as usize);
                    a.pres[fx.level[g] as usize - 1][cr / 64] &= !(1 << (cr % 64));
                    c.delete(pr, cr as u32);
                    dels += 1;
                }
            }
            let w1 = mint_presence(&fx, 2);
            let mut ok = true;
            for &root in &fx.roots {
                let ar = l.row[root as usize];
                let prog = a.program(cand_pred(loc, ar, size));
                let want = oracle(&fx, root);
                ok &= a.run(&prog, &mut sc) == want && c.fold(ar, &mut CStats::default()) == want;
            }
            t.check(&format!("parity after {dels} deletions, binding reused, {tag}"), ok, String::new());
            t.check(&format!("stale presence witness refused (A, C), {tag}"),
                a.verify(&table, &w0, 2).is_err() && c.verify(&table, &w0, 2, &fx.level, &l).is_err(), String::new());
            t.check(&format!("fresh presence witness verifies (A, C), {tag}"),
                a.verify(&table, &w1, 2).is_ok() && c.verify(&table, &w1, 2, &fx.level, &l).is_ok(), String::new());
            // Bypass: a bit cleared without a generation or witness change.
            if let Some(g) = (0..fx.n()).find(|&g| fx.level[g] == 3 && fx.pres[g]) {
                let r = l.row[g] as usize;
                a.pres[2][r / 64] &= !(1 << (r % 64));
                c.delete(l.row[fx.parent[g] as usize], r as u32);
                t.check(&format!("bypass deletion refused by presence witness (A, C), {tag}"),
                    a.verify(&table, &w1, 2).is_err() && c.verify(&table, &w1, 2, &fx.level, &l).is_err(), String::new());
            }
        }
    }

    // Duplicate paths: d reachable through two c. A cannot express it; C dedups.
    let mut fx = forest(2, 4, 100, 3);
    let ds: Vec<u32> = (0..fx.n() as u32).filter(|&g| fx.level[g as usize] == 3).collect();
    let cs: Vec<u32> = (0..fx.n() as u32).filter(|&g| fx.level[g as usize] == 2).collect();
    for &d in ds.iter().step_by(3) {
        let pc = fx.parent[d as usize];
        let root = |c: u32| fx.parent[fx.parent[c as usize] as usize];
        let other = cs.iter().copied().find(|&c| c != pc && root(c) == root(pc));
        if let Some(c) = other { fx.extra_t.push((c, d)); }
    }
    let l = layout(fx.n(), Locality::Ordered, 1);
    let (table, frame, _) = bind(&l).unwrap();
    t.check("duplicate paths: A refuses (pointer table needs a functional parent)",
        matches!(ArmA::mint(&fx, &l, &table, frame.clone()), Err(Refusal::NotFunctional)), format!("{} extra T edges", fx.extra_t.len()));
    let mut c = ArmC::new(&fx, &l, frame);
    let mut ok = true;
    let mut set_vs_bag = String::new();
    for &root in &fx.roots {
        let want = oracle(&fx, root);
        ok &= c.fold(l.row[root as usize], &mut CStats::default()) == want;
        set_vs_bag = format!("set {} vs bag {} (root {root})", want, oracle_paths(&fx, root));
    }
    t.check("duplicate paths: C equals the set oracle (OR dedups)", ok, set_vs_bag);

    // Scope change: a tile replaced (reordered) → old frame refused by both arms.
    let fx = forest(2, 4, 80, 5);
    let l = layout(fx.n(), Locality::Ordered, 1);
    let (mut table, frame, _) = bind(&l).unwrap();
    let a = ArmA::mint(&fx, &l, &table, frame.clone()).unwrap();
    let c = ArmC::new(&fx, &l, frame.clone());
    let w = mint_presence(&fx, 1);
    let mut g0: Vec<u64> = l.inv[..TILE.min(fx.n())].iter().map(|&g| u64::from(g)).collect();
    g0.reverse();
    table.advance(1, 2, vec![(0, Some(LocalMap::prepare(Width::W256, &g0).unwrap()))]);
    t.check("scope change: stale frame refused by A and C",
        a.verify(&table, &w, 1) == Err(Refusal::StaleFrame) && c.verify(&table, &w, 1, &fx.level, &l) == Err(Refusal::StaleFrame), String::new());
    t.check("scope change: mint under the stale frame refused",
        matches!(ArmA::mint(&fx, &l, &table, frame), Err(Refusal::StaleFrame)), String::new());
    t.check("duplicate global refused by the existing LocalMap",
        LocalMap::prepare(Width::W256, &[1, 2, 1]).err() == Some(MapRefusal::Duplicate), String::new());
}

// ------------------------------------------------------------- disables
fn disables(t: &mut Tally) {
    println!("== disable runs (each must turn the answer wrong)");
    let fx = forest(4, 6, 60, 21);
    let l = layout(fx.n(), Locality::Ordered, 1);
    let (table, frame, _) = bind(&l).unwrap();
    let a = ArmA::mint(&fx, &l, &table, frame.clone()).unwrap();
    let mut c = ArmC::new(&fx, &l, frame);
    let size = (1 + 6 + 36 + 216) as u32;
    let mut sc = Scratch::new(words_for(fx.n()), 2);
    let root = fx.roots[1];
    let ar = l.row[root as usize];
    let want = oracle(&fx, root);
    // A without the S gather.
    let p = Program::new(
        vec![
            MaskOp::Pred { pred: cand_pred(Locality::Ordered, ar, size), under: Some(Operand::Plane(0)), dst: 0 },
            MaskOp::Gather { lane: 0, foreign: 0, under: Some(Operand::Scratch(0)), dst: 1 },
        ],
        Terminal::Count { mask: Operand::Scratch(1) },
    );
    let got = a.run(&p, &mut sc);
    t.check("A without Gather(ic→S_in) is wrong", got != want, format!("{got} vs {want}"));
    // C with one hop skipped (b frontier counted as d).
    c.acc_c.clear();
    c.acc_d.clear();
    for i in c.ch.off[ar as usize]..c.ch.off[ar as usize + 1] {
        let (tt, b) = (c.ch.tile[i as usize], c.ch.bits[i as usize]);
        c.acc_c.or(tt, &b);
    }
    let mut st = CStats::default();
    ArmC::hop(&c.ch, &c.acc_c, &mut c.acc_d, &mut st);
    let got: u64 = c.acc_d.touched.iter().map(|&tt| popcount_batch_u64(&c.acc_d.words[tt as usize])).sum();
    t.check("C with the T hop skipped is wrong", got != want, format!("{got} vs {want}"));
}

// ------------------------------------------------------------- sweep
fn sweep() {
    println!("== sweep (µs, median of {REPS}; loads/ors are exact counts, not estimates)");
    println!("loc,roots,f,keep%,rows,tiles,cand_rows,d_hits,A_hot_us,A_loads,C_hot_us,C_chunk_ors,C_tiles,C_chunks_total,C_bytes,A_lane_bytes,map_us,certify_us,witA_us,witC_us,chunkbuild_us,A_hot_allocs,C_hot_allocs");
    for &loc in &[Locality::Ordered, Locality::Shuffled] {
        for &f in &[4usize, 8, 16] {
            for &keep in &[10u64, 50, 90] {
                for &roots in &[1usize, 16, 64] {
                    let fx = forest(roots, f, keep, 1000 + f as u64 * 7 + keep);
                    let n = fx.n();
                    let l = layout(n, loc, 2024);
                    let map_ns = median_ns(|| { let _ = bind(&l).unwrap(); });
                    let (table, frame, tiles) = bind(&l).unwrap();
                    let cert_ns = median_ns(|| { let _ = ArmA::mint(&fx, &l, &table, frame.clone()).unwrap(); });
                    let a = ArmA::mint(&fx, &l, &table, frame.clone()).unwrap();
                    let chunk_ns = median_ns(|| { let _ = Chunks::build(&fx, &l); });
                    let mut c = ArmC::new(&fx, &l, frame.clone());
                    let w = mint_presence(&fx, 1);
                    let wa_ns = median_ns(|| { a.verify(&table, &w, 1).unwrap(); });
                    let wc_ns = median_ns(|| { c.verify(&table, &w, 1, &fx.level, &l).unwrap(); });
                    // Query root: the middle root.
                    let root = fx.roots[roots / 2];
                    let ar = l.row[root as usize];
                    let size = (1 + f + f * f + f * f * f) as u32;
                    let prog = a.program(cand_pred(loc, ar, size));
                    let mut sc = Scratch::new(words_for(n), 2);
                    let want = oracle(&fx, root);
                    assert_eq!(a.run(&prog, &mut sc), want);
                    let mut st = CStats::default();
                    assert_eq!(c.fold(ar, &mut st), want);
                    let (al, ab) = alloc_snapshot();
                    let a_ns = median_ns(|| { std::hint::black_box(a.run(&prog, &mut sc)); });
                    let (al2, _) = alloc_snapshot();
                    let c_ns = median_ns(|| { std::hint::black_box(c.fold(ar, &mut CStats::default())); });
                    let (al3, _) = alloc_snapshot();
                    let _ = ab;
                    let (l1, l2) = match loc {
                        Locality::Ordered => a.loads(|i| (ar as usize..(ar + size) as usize).contains(&i)),
                        Locality::Shuffled => a.loads(|i| a.ia[i] == ar),
                    };
                    let chunks_total = c.ch.tile.len();
                    let c_bytes = chunks_total * 36 + c.ch.off.len() * 4;
                    let a_bytes = 3 * n * 4;
                    println!(
                        "{},{roots},{f},{keep},{n},{},{size},{want},{},{},{},{},{},{chunks_total},{c_bytes},{a_bytes},{},{},{},{},{},{},{}",
                        if loc == Locality::Ordered { "ord" } else { "shuf" },
                        tiles.len(),
                        us(a_ns), l1 + l2, us(c_ns), st.chunk_ors, st.tiles_touched,
                        us(map_ns), us(cert_ns), us(wa_ns), us(wc_ns), us(chunk_ns),
                        (al2 - al) / REPS as u64, (al3 - al2) / REPS as u64,
                    );
                }
            }
        }
    }
}

/// B: amortization. Same hot fold as A; the binding (map + certify) and the
/// presence witness are paid per generation, under two lifetime policies:
/// retire = remint pointers on every presence generation (v4, conservative);
/// split  = pointers survive presence changes (the Cube split), only the
///          presence witness is re-verified.
fn amortization() {
    println!("== B amortization (measured components; per-query cost derived as hot + per-gen / Q)");
    println!("loc,roots,f,keep%,hot_A_us,certify_us,witA_us,Q,retire_per_query_us,split_per_query_us,measured_loop_split_per_query_us");
    for &loc in &[Locality::Ordered, Locality::Shuffled] {
        let (roots, f, keep) = (64usize, 8usize, 50u64);
        let fx = forest(roots, f, keep, 77);
        let n = fx.n();
        let l = layout(n, loc, 2024);
        let (table, frame, _) = bind(&l).unwrap();
        let cert = median_ns(|| { let _ = ArmA::mint(&fx, &l, &table, frame.clone()).unwrap(); });
        let a = ArmA::mint(&fx, &l, &table, frame.clone()).unwrap();
        let w = mint_presence(&fx, 1);
        let wit = median_ns(|| { a.verify(&table, &w, 1).unwrap(); });
        let size = (1 + f + f * f + f * f * f) as u32;
        let progs: Vec<Program> = fx.roots.iter().map(|&r| a.program(cand_pred(loc, l.row[r as usize], size))).collect();
        let mut sc = Scratch::new(words_for(n), 2);
        let hot = median_ns(|| { for p in &progs { std::hint::black_box(a.run(p, &mut sc)); } }) / roots as f64;
        for &q in &[1usize, 16, 64, 256, 4096] {
            let retire = hot + (cert + wit) / q as f64;
            let split = hot + wit / q as f64;
            // Measured: one generation = verify once, then q queries cycling over roots.
            let loop_ns = if q <= 256 {
                median_ns(|| {
                    a.verify(&table, &w, 1).unwrap();
                    for i in 0..q { std::hint::black_box(a.run(&progs[i % roots], &mut sc)); }
                }) / q as f64
            } else { f64::NAN };
            println!("{},{roots},{f},{keep},{},{},{},{q},{},{},{}",
                if loc == Locality::Ordered { "ord" } else { "shuf" },
                us(hot), us(cert), us(wit), us(retire), us(split),
                if loop_ns.is_nan() { "-".into() } else { us(loop_ns) });
        }
    }
}

/// Falsifier for 'the tile sweep always wins': dense R and S over a sparse T.
/// C (top-down) must expand every live b and c; A (d-anchored) touches only
/// the few present T rows. And the reverse skew, for contrast.
fn skew() {
    println!("== skew falsifier (µs, median of {REPS}); keep% per level R/S/T");
    println!("loc,roots,f,keep_R,keep_S,keep_T,rows,d_hits,A_hot_us,A_loads,C_hot_us,C_chunk_ors,C_tiles,A_over_C");
    for &loc in &[Locality::Ordered, Locality::Shuffled] {
        for &(roots, f) in &[(1usize, 16usize), (16, 16), (1, 32)] {
            for &keep in &[[100u64, 100, 1], [100, 100, 5], [100, 100, 20], [5, 100, 100], [100, 5, 100]] {
                let fx = forest_lv(roots, f, keep, 4242 + f as u64);
                let n = fx.n();
                let l = layout(n, loc, 2024);
                let (table, frame, _) = bind(&l).unwrap();
                let a = ArmA::mint(&fx, &l, &table, frame.clone()).unwrap();
                let mut c = ArmC::new(&fx, &l, frame);
                let root = fx.roots[roots / 2];
                let ar = l.row[root as usize];
                let size = (1 + f + f * f + f * f * f) as u32;
                let prog = a.program(cand_pred(loc, ar, size));
                let mut sc = Scratch::new(words_for(n), 2);
                let want = oracle(&fx, root);
                assert_eq!(a.run(&prog, &mut sc), want);
                let mut st = CStats::default();
                assert_eq!(c.fold(ar, &mut st), want);
                let a_ns = median_ns(|| { std::hint::black_box(a.run(&prog, &mut sc)); });
                let c_ns = median_ns(|| { std::hint::black_box(c.fold(ar, &mut CStats::default())); });
                let (l1, l2) = match loc {
                    Locality::Ordered => a.loads(|i| (ar as usize..(ar + size) as usize).contains(&i)),
                    Locality::Shuffled => a.loads(|i| a.ia[i] == ar),
                };
                println!("{},{roots},{f},{},{},{},{n},{want},{},{},{},{},{},{:.2}",
                    if loc == Locality::Ordered { "ord" } else { "shuf" }, keep[0], keep[1], keep[2],
                    us(a_ns), l1 + l2, us(c_ns), st.chunk_ors, st.tiles_touched, a_ns / c_ns);
            }
        }
    }
}

fn main() {
    let mut t = Tally(0, 0);
    correctness(&mut t);
    disables(&mut t);
    println!("== checks: {} passed, {} failed", t.0, t.1);
    sweep();
    skew();
    amortization();
    if t.1 > 0 { std::process::exit(1); }
}
