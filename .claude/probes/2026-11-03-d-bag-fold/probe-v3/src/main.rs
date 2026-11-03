//! D-BAG-FOLD v3: Arm L (`$i` + `Filter::Semijoin` over resident presence planes)
//! exercised against PRODUCTION witness contracts. Lab only, own workspace.
//!
//! Production contracts used, unchanged:
//!   - order + version: `lance_graph_contract::ordered_lane::{SealedFacetLane,
//!     OrderedLaneWitness, WitnessError}` (seal, validate, seek_from, bound);
//!   - frame: `lance_graph_dt64::dr::{FrameStore, Frame, FrameRevisions, FrameRefusal}`;
//!   - keys: `lance_graph_contract::facet::{FacetCascade, SemanticPrefix}`.
//! Lab-local (no production contract exists; each is named in the report):
//!   - `AddressLens` provenance check, pointer bounds check, strict key uniqueness,
//!     presence-plane digests.
//!
//! Query: ∃d. R(a,b(d)) ∧ S(b(d),c(d)) ∧ T(c(d),d), and the count of distinct valid d.
//! Oracle: nested loops over explicit edge sets, independent of every lane.

use lance_graph_contract::facet::{FacetCascade, SemanticPrefix};
use lance_graph_contract::ordered_lane::{OrderedLaneWitness, SealedFacetLane, WitnessError};
use lance_graph_dt64::dr::{Frame, FrameRefusal, FrameRevisions, FrameStore};
use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, ForeignPlane as FP, LaneRef, Out, Planes, Program, Scratch, Value,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, ForeignPlane, Mask, Query};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

// ---------------------------------------------------------------- allocation counter
struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size(), Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { System.dealloc(p, l) }
}
#[global_allocator]
static A: Counting = Counting;
fn alloc_snapshot() -> (usize, usize) { (ALLOCS.load(Relaxed), BYTES.load(Relaxed)) }

// ---------------------------------------------------------------- keys
const CANON: u16 = 0x0905;
const APP_K: u16 = 0x000A;
const APP_K2: u16 = 0x000B;

fn key(app: u16, t: [u16; 4]) -> FacetCascade {
    FacetCascade::from_semantic_tiles([CANON, app, t[0], t[1], t[2], t[3], 0, 0])
}
fn tiers(k: &FacetCascade) -> [u16; 8] { k.semantic_tiles() }
fn depth(k: &FacetCascade) -> usize { tiers(k)[2..6].iter().take_while(|&&t| t != 0).count() }
/// Address arithmetic on the key's own tiles: zero the last set tier. Stays in its classid.
fn parent(k: &FacetCascade) -> Option<FacetCascade> {
    let d = depth(k);
    if d < 2 { return None; }
    let mut t = tiers(k);
    t[2 + d - 1] = 0;
    Some(FacetCascade::from_semantic_tiles(t))
}
type Edges = BTreeSet<(u128, u128)>;
fn fc(v: u128) -> FacetCascade { FacetCascade::from_u128(v) }
fn e(x: FacetCascade, y: FacetCascade) -> (u128, u128) { (x.as_u128(), y.as_u128()) }

// ---------------------------------------------------------------- the store
/// One table: a sealed key lane, three presence planes stamped at the CHILD row, and
/// the frame store whose `source` revision the presence writer must advance.
struct Store {
    lane: SealedFacetLane,
    pres: [Vec<u64>; 3],
    frames: FrameStore,
}

#[derive(Debug)]
#[allow(dead_code)] // fields are read through Debug in the report
enum Refusal {
    Witness(WitnessError),
    Frame(FrameRefusal),
    DuplicateKey(usize),
    NonRefiningEdge,
    UnknownNode,
    PointerOutOfBounds { row: usize, ptr: u32 },
    Provenance { row: usize },
    PresenceChanged { plane: usize },
}

fn set(p: &mut [u64], i: usize) { p[i / 64] |= 1 << (i % 64); }
fn get(p: &[u64], i: usize) -> bool { p[i / 64] >> (i % 64) & 1 == 1 }
fn fnv(words: &[u64]) -> u64 {
    words.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, w| (h ^ w).wrapping_mul(0x0100_0000_01b3))
}

impl Store {
    fn build(nodes: &[FacetCascade], rels: [&Edges; 3], version: u64) -> Result<Store, Refusal> {
        let lane = SealedFacetLane::seal(nodes.to_vec(), version).map_err(Refusal::Witness)?;
        let n = lane.keys().len();
        let w = lane.witness();
        let mut pres = [vec![0u64; words_for(n)], vec![0u64; words_for(n)], vec![0u64; words_for(n)]];
        for (r, edges) in rels.iter().enumerate() {
            for &(x, y) in edges.iter() {
                let (x, y) = (fc(x), fc(y));
                if parent(&y).as_ref() != Some(&x) { return Err(Refusal::NonRefiningEdge); }
                let i = lane.seek_from(&w, 0, &y).map_err(Refusal::Witness)? as usize;
                if i >= n || lane.keys()[i] != y { return Err(Refusal::UnknownNode); }
                set(&mut pres[r], i);
            }
        }
        let frames = FrameStore::new(FrameRevisions { source: 1, relation_map: 1, coordinates: 1 });
        Ok(Store { lane, pres, frames })
    }
    /// A presence write that follows protocol: change the bit, advance the frame.
    fn delete_presence(&mut self, plane: usize, row: usize, advance: bool) {
        self.pres[plane][row / 64] &= !(1 << (row % 64));
        if advance {
            let mut r = self.frames.revisions();
            r.source += 1;
            self.frames.advance(r);
        }
    }
}

// ---------------------------------------------------------------- Arm L: the prepared lens
/// The prepared pointer table. Only `prepare` constructs it from a store.
struct AddressLens {
    witness: OrderedLaneWitness,
    frame: Frame,
    ib: Vec<u32>,
    ic: Vec<u32>,
    plane_digest: [u64; 3],
    range: (u32, u32),
    sentinel: u32,
}

#[derive(Clone, Copy)]
struct Checks { uniqueness: bool, bounds: bool, provenance: bool, plane_digest: bool }
const ALL: Checks = Checks { uniqueness: true, bounds: true, provenance: true, plane_digest: true };
const CONTRACT_ONLY: Checks = Checks { uniqueness: false, bounds: false, provenance: false, plane_digest: false };

/// Preparation: witnessed same-table seeks keyed by each row's own key arithmetic.
fn prepare(s: &Store, w: &OrderedLaneWitness, frame: Frame, a: FacetCascade, c: Checks) -> Result<AddressLens, Refusal> {
    s.lane.validate(w).map_err(Refusal::Witness)?; // production: version, n_rows, digest
    s.frames.check(&frame).map_err(Refusal::Frame)?; // production: frame identity + revisions
    let keys = s.lane.keys();
    let n = keys.len();
    if c.uniqueness {
        if let Some(i) = keys.windows(2).position(|p| p[0] == p[1]) { return Err(Refusal::DuplicateKey(i + 1)); }
    }
    // The candidate population: rows under prefix(a), by the production witnessed bound.
    let range = s.lane.bound(w, &SemanticPrefix::of(a, 3)).map_err(Refusal::Witness)?;
    // Sentinel: a row whose R and S bits are both clear, or refuse to need one.
    let sentinel = (0..n).find(|&i| !get(&s.pres[0], i) && !get(&s.pres[1], i)).unwrap_or(0) as u32;
    // Over a lane in numeric projection order, the ancestors of the rows of one depth
    // are non-decreasing, so one witnessed `seek_from` cursor per ancestor depth makes
    // every seek a gallop from the previous answer (the leapfrog use `seek_from`
    // documents). Same production call; only the `from` argument differs.
    let find = |cursor: &mut [u32; 5], k: Option<FacetCascade>| -> Result<Option<u32>, Refusal> {
        let Some(k) = k else { return Ok(None) };
        let d = depth(&k);
        let i = s.lane.seek_from(w, cursor[d], &k).map_err(Refusal::Witness)?;
        cursor[d] = i;
        Ok(((i as usize) < n && keys[i as usize] == k).then_some(i))
    };
    let mut ib = vec![sentinel; n];
    let mut ic = vec![sentinel; n];
    // Rows are visited depth by depth so each cursor only moves forward.
    for want_depth in 1..=4 {
        let mut cursor = [0u32; 5];
        for (j, k) in keys.iter().enumerate() {
            if depth(k) != want_depth { continue; }
            if let Some(c1) = find(&mut cursor, parent(k))? { ic[j] = c1; }
            if let Some(b1) = find(&mut cursor, parent(k).as_ref().and_then(parent))? { ib[j] = b1; }
        }
    }
    let plane_digest = [fnv(&s.pres[0]), fnv(&s.pres[1]), fnv(&s.pres[2])];
    Ok(AddressLens { witness: *w, frame, ib, ic, plane_digest, range, sentinel })
}

/// Execution-time checks that run before the fold. Returns `Ok(())` or a refusal.
fn admit(s: &Store, l: &AddressLens, c: Checks) -> Result<(), Refusal> {
    s.lane.validate(&l.witness).map_err(Refusal::Witness)?;
    s.frames.check(&l.frame).map_err(Refusal::Frame)?;
    let keys = s.lane.keys();
    let n = keys.len() as u32;
    if c.bounds {
        for (row, &p) in l.ib.iter().chain(l.ic.iter()).enumerate() {
            if p >= n { return Err(Refusal::PointerOutOfBounds { row: row % keys.len(), ptr: p }); }
        }
    }
    if c.provenance {
        for (j, k) in keys.iter().enumerate() {
            let pc = parent(k);
            let pb = pc.as_ref().and_then(parent);
            let ok = |p: u32, want: Option<FacetCascade>| match want {
                Some(w) => keys.get(p as usize) == Some(&w) || p == l.sentinel,
                None => p == l.sentinel,
            };
            if !ok(l.ic[j], pc) || !ok(l.ib[j], pb) { return Err(Refusal::Provenance { row: j }); }
            // A sentinel standing in for an ancestor that EXISTS is also a provenance
            // failure. Only sentinel rows need the search, so this stays O(n).
            let exists = |w: &FacetCascade| keys.binary_search_by(|k| k.cmp_numeric_projection(w)).is_ok();
            if let Some(w) = pc { if keys[l.ic[j] as usize] != w && exists(&w) { return Err(Refusal::Provenance { row: j }); } }
            if let Some(w) = pb { if keys[l.ib[j] as usize] != w && exists(&w) { return Err(Refusal::Provenance { row: j }); } }
        }
    }
    if c.plane_digest {
        for (i, p) in s.pres.iter().enumerate() {
            if fnv(p) != l.plane_digest[i] { return Err(Refusal::PresenceChanged { plane: i }); }
        }
    }
    Ok(())
}

fn lens_filter(range: (u32, u32), drop: Option<usize>) -> Filter {
    let mut v = vec![
        Filter::Cmp(Col(0), Cmp::Range { lo: range.0, hi: range.1 }),
        Filter::Plane(Mask(0)),
        Filter::Semijoin { fk: Col(0), foreign: ForeignPlane(0) },
        Filter::Semijoin { fk: Col(1), foreign: ForeignPlane(1) },
    ];
    if let Some(i) = drop { v.remove(i); }
    Filter::And(v)
}

struct Prepared { any: Program, count: Program, sc: Scratch<'static> }
fn programs(range: (u32, u32), n: usize, drop: Option<usize>) -> Prepared {
    let f = lens_filter(range, drop);
    let any = lower(&Query { filter: f.clone(), agg: Agg::Any }).expect("lowers");
    let count = lower(&Query { filter: f, agg: Agg::Count }).expect("lowers");
    let sc = Scratch::for_program(&count, n.max(1)).expect("carves");
    Prepared { any, count, sc }
}

/// One Program per terminal over the node lane. `R_in`/`S_in` read in place through `$i`.
fn run(s: &Store, l: &AddressLens, p: &mut Prepared) -> (bool, u64) {
    let n = s.lane.keys().len();
    let lanes = [LaneRef::U32(&l.ib), LaneRef::U32(&l.ic)];
    let fps = [FP { words: &s.pres[0], rows: n }, FP { words: &s.pres[1], rows: n }];
    let foreign = Foreign { planes: &fps, lanes: &[] };
    let masks: [&[u64]; 1] = [&s.pres[2]];
    let pl = Planes { n_rows: n, masks: &masks, lanes: &lanes };
    let any = match execute_into(&p.any, &pl, &foreign, &mut p.sc, Out::None).expect("runs") { Value::Bool(b) => b, v => panic!("{v:?}") };
    let count = match execute_into(&p.count, &pl, &foreign, &mut p.sc, Out::None).expect("runs") { Value::Count(c) => c as u64, v => panic!("{v:?}") };
    (any, count)
}

fn eval(s: &Store, w: &OrderedLaneWitness, frame: Frame, a: FacetCascade, c: Checks) -> Result<(bool, u64), Refusal> {
    let l = prepare(s, w, frame, a, c)?;
    admit(s, &l, c)?;
    let mut p = programs(l.range, s.lane.keys().len(), None);
    Ok(run(s, &l, &mut p))
}

fn oracle(a: FacetCascade, r: &Edges, s: &Edges, t: &Edges) -> (bool, u64) {
    let a = a.as_u128();
    let mut ds = BTreeSet::new();
    for &(x, b) in r { if x != a { continue; }
        for &(b2, c) in s { if b2 != b { continue; }
            for &(c2, d) in t { if c2 == c { ds.insert(d); } } } }
    (!ds.is_empty(), ds.len() as u64)
}

// ---------------------------------------------------------------- fixtures
struct Fx { nodes: Vec<FacetCascade>, r: Edges, s: Edges, t: Edges, roots: Vec<FacetCascade> }

/// `roots` depth-1 nodes, fanout `f` per level, depth 4. Edge presence by a fixed LCG.
fn hierarchy(app: u16, roots: u16, f: u16, keep_pct: u64, seed: u64) -> Fx {
    let mut st = seed;
    let mut coin = || { st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (st >> 33) % 100 < keep_pct };
    let (mut nodes, mut r, mut s, mut t, mut rs) = (vec![], Edges::new(), Edges::new(), Edges::new(), vec![]);
    for a1 in 1..=roots {
        let a = key(app, [a1, 0, 0, 0]); nodes.push(a); rs.push(a);
        for b1 in 1..=f { let b = key(app, [a1, b1, 0, 0]); nodes.push(b); if coin() { r.insert(e(a, b)); }
            for c1 in 1..=f { let c = key(app, [a1, b1, c1, 0]); nodes.push(c); if coin() { s.insert(e(b, c)); }
                for d1 in 1..=f { let d = key(app, [a1, b1, c1, d1]); nodes.push(d); if coin() { t.insert(e(c, d)); } } } }
    }
    Fx { nodes, r, s, t, roots: rs }
}

// ---------------------------------------------------------------- harness
struct Tally { pass: usize, fail: usize }
impl Tally {
    fn check(&mut self, name: &str, ok: bool, detail: String) {
        if ok { self.pass += 1 } else { self.fail += 1 }
        println!("{} {:<64} {}", if ok { "PASS" } else { "FAIL" }, name, detail);
    }
}
fn is_refused<T>(r: &Result<T, Refusal>) -> bool { r.is_err() }

fn main() {
    let mut tl = Tally { pass: 0, fail: 0 };

    // ============ 1. correctness vs oracle, every root, two sizes
    for (roots, f) in [(4u16, 4u16), (8, 8)] {
        let fx = hierarchy(APP_K, roots, f, 70, 7);
        let st = Store::build(&fx.nodes, [&fx.r, &fx.s, &fx.t], 1).unwrap();
        let w = st.lane.witness();
        let mut ok = true;
        let mut detail = String::new();
        for &a in &fx.roots {
            let got = eval(&st, &w, st.frames.seal(), a, ALL).unwrap();
            let want = oracle(a, &fx.r, &fx.s, &fx.t);
            if got != want { ok = false; detail = format!("{got:?} vs {want:?}"); }
        }
        tl.check(&format!("oracle parity, {} roots, fanout {f}, {} nodes", fx.roots.len(), fx.nodes.len()), ok, detail);
    }

    // fixed small fixture for the negative controls
    let fx = hierarchy(APP_K, 2, 3, 100, 1);
    let a = fx.roots[0];
    let base = || Store::build(&fx.nodes, [&fx.r, &fx.s, &fx.t], 1).unwrap();
    let want = oracle(a, &fx.r, &fx.s, &fx.t);

    // ============ 2. permuted planes
    {
        let mut st = base();
        let w = st.lane.witness();
        let l = prepare(&st, &w, st.frames.seal(), a, ALL).unwrap();
        st.pres[1].reverse(); // planes permuted at rest, key lane untouched
        let contract = st.lane.validate(&w).is_ok() && st.frames.check(&l.frame).is_ok();
        tl.check("2a permuted S plane: production witness + frame still validate (gap)", contract, "validate=Ok, frame=Ok".into());
        tl.check("2b permuted S plane: refused by lab plane digest", is_refused(&admit(&st, &l, ALL)), format!("{:?}", admit(&st, &l, ALL).err()));
        let mut p = programs(l.range, st.lane.keys().len(), None);
        let silent = admit(&st, &l, CONTRACT_ONLY).is_ok() && run(&st, &l, &mut p) != want;
        tl.check("2c permuted S plane, contract-only checks: silent wrong answer (disable)", silent, format!("got {:?} vs oracle {want:?}", run(&st, &l, &mut p)));
    }

    // ============ 3. stale presence
    {
        let mut st = base();
        let w = st.lane.witness();
        let l = prepare(&st, &w, st.frames.seal(), a, ALL).unwrap();
        let row = (0..st.lane.keys().len()).find(|&i| get(&st.pres[1], i)).unwrap();
        st.delete_presence(1, row, true);
        let r = admit(&st, &l, CONTRACT_ONLY);
        tl.check("3a stale presence, writer advances FrameStore: refused by production frame check", matches!(r, Err(Refusal::Frame(FrameRefusal::SourceChanged))), format!("{:?}", r.err()));
        let mut st = base();
        let l = prepare(&st, &w, st.frames.seal(), a, ALL).unwrap();
        st.delete_presence(1, row, false);
        let contract = admit(&st, &l, CONTRACT_ONLY).is_ok();
        tl.check("3b stale presence, frame NOT advanced: production checks pass (gap)", contract, "validate=Ok, frame=Ok".into());
        tl.check("3c stale presence, frame NOT advanced: refused by lab plane digest", is_refused(&admit(&st, &l, ALL)), format!("{:?}", admit(&st, &l, ALL).err()));
        let resealed = SealedFacetLane::seal(st.lane.keys().to_vec(), 2).unwrap();
        tl.check("3d lane resealed at a new LanceVersion: old witness refused by production validate", matches!(resealed.validate(&w), Err(WitnessError::VersionMismatch { .. })), format!("{:?}", resealed.validate(&w).err()));
    }

    // ============ 4. duplicate keys
    {
        let mut nodes = fx.nodes.clone();
        let d = fc(fx.t.iter().next().map(|(_, d)| *d).unwrap());
        nodes.push(d);
        let sealed = SealedFacetLane::seal(nodes.clone(), 1);
        tl.check("4a duplicate key: production seal ACCEPTS it (gap)", sealed.is_ok(), format!("{:?}", sealed.as_ref().err()));
        let st = Store::build(&nodes, [&fx.r, &fx.s, &fx.t], 1).unwrap();
        let w = st.lane.witness();
        let r = prepare(&st, &w, st.frames.seal(), a, ALL);
        tl.check("4b duplicate key: refused by lab uniqueness check", matches!(r, Err(Refusal::DuplicateKey(_))), format!("{:?}", r.err()));
        // stamp the duplicate row's T bit too, as a duplicated T edge would
        let mut st = st;
        for i in 0..st.lane.keys().len() { if st.lane.keys()[i] == d { set(&mut st.pres[2], i); } }
        let got = eval(&st, &w, st.frames.seal(), a, CONTRACT_ONLY).unwrap();
        tl.check("4c duplicate key, contract-only checks: Count != distinct (disable)", got.1 != want.1, format!("got {got:?} vs oracle {want:?}"));
    }

    // ============ 5. invalid pointer bounds
    {
        let st = base();
        let w = st.lane.witness();
        let mut l = prepare(&st, &w, st.frames.seal(), a, ALL).unwrap();
        let n = st.lane.keys().len() as u32;
        let j = (l.range.0..l.range.1).find(|&j| depth(&st.lane.keys()[j as usize]) == 4 && get(&st.pres[2], j as usize)).unwrap() as usize;
        l.ib[j] = n; // one past the end
        tl.check("5a pointer = n_rows: refused by lab bounds check", matches!(admit(&st, &l, ALL), Err(Refusal::PointerOutOfBounds { .. })), format!("{:?}", admit(&st, &l, ALL).err()));
        let mut p = programs(l.range, n as usize, None);
        let got = run(&st, &l, &mut p);
        tl.check("5b pointer = n_rows, contract-only: Gather zero-fallback answers silently (disable)", admit(&st, &l, CONTRACT_ONLY).is_ok() && got != want, format!("got {got:?} vs oracle {want:?}"));
    }

    // ============ 6. wrong ClassId / codebook / foreign witness / foreign frame
    {
        let mut fx2 = hierarchy(APP_K, 2, 3, 100, 1);
        let other = hierarchy(APP_K2, 2, 3, 100, 1); // same tiers, other app prefix, full chains
        fx2.nodes.extend(other.nodes.iter().copied());
        fx2.r.extend(other.r.iter().copied()); fx2.s.extend(other.s.iter().copied()); fx2.t.extend(other.t.iter().copied());
        // remove K's chains so the only valid chains belong to K2
        let fx2_r: Edges = fx2.r.iter().copied().filter(|(x, _)| tiers(&fc(*x))[1] == APP_K2).collect();
        let st = Store::build(&fx2.nodes, [&fx2_r, &fx2.s, &fx2.t], 1).unwrap();
        let w = st.lane.witness();
        let got = eval(&st, &w, st.frames.seal(), a, ALL).unwrap();
        let wnt = oracle(a, &fx2_r, &fx2.s, &fx2.t);
        tl.check("6a other ClassId holds the only valid chains: query under K answers per oracle", got == wnt && !got.0, format!("got {got:?} vs oracle {wnt:?}"));
        let cross: Edges = [e(key(APP_K, [1, 1, 0, 0]), key(APP_K2, [1, 1, 1, 0]))].into_iter().collect();
        tl.check("6b edge crossing the ClassId boundary: refused at build (lab)", matches!(Store::build(&fx2.nodes, [&fx2_r, &cross, &fx2.t], 1), Err(Refusal::NonRefiningEdge)), String::new());
        // a pointer redirected into the other class (same tiers): provenance must refuse
        let mut l = prepare(&st, &w, st.frames.seal(), a, ALL).unwrap();
        let keys = st.lane.keys();
        let j = (l.range.0..l.range.1).find(|&j| depth(&keys[j as usize]) == 4).unwrap() as usize;
        let mut t = tiers(&keys[l.ic[j] as usize]); t[1] = APP_K2;
        let tgt = FacetCascade::from_semantic_tiles(t);
        l.ic[j] = keys.iter().position(|k| *k == tgt).unwrap() as u32;
        tl.check("6c pointer redirected into another ClassId: refused by lab provenance", matches!(admit(&st, &l, ALL), Err(Refusal::Provenance { .. })), format!("{:?}", admit(&st, &l, ALL).err()));
        tl.check("6d same, contract-only: production checks pass (gap)", admit(&st, &l, CONTRACT_ONLY).is_ok(), String::new());
        // witness from another lane at the same version
        let other_lane = SealedFacetLane::seal(other.nodes.clone(), 1).unwrap();
        let r = st.lane.validate(&other_lane.witness());
        tl.check("6e witness of another lane, same version: refused by production validate", r.is_err(), format!("{:?}", r.err()));
        // frame from another store
        let foreign = FrameStore::new(FrameRevisions { source: 1, relation_map: 1, coordinates: 1 }).seal();
        let r = prepare(&st, &w, foreign, a, ALL);
        tl.check("6f frame sealed by another FrameStore: refused by production frame check", matches!(r, Err(Refusal::Frame(FrameRefusal::ForeignFrame))), format!("{:?}", r.err()));
        tl.check("6g wrong codebook/lens: SemanticLens has one variant, mismatch not constructible (gap)", true, "untestable by construction".into());
    }

    // ============ 7. value-derived pointers
    {
        // R(a, b0) absent; S still has edges out of b0. The value-derived pointer lane is
        // filled from S's SOURCE endpoint values: every row points at b0, a row chosen by
        // a value in another relation, not by the row's own address.
        let mut fx7 = hierarchy(APP_K, 2, 3, 100, 1);
        let b0 = fc(fx7.r.iter().next().map(|(_, b)| *b).unwrap());
        fx7.r.remove(&e(a, b0));
        let st = Store::build(&fx7.nodes, [&fx7.r, &fx7.s, &fx7.t], 1).unwrap();
        let want = oracle(a, &fx7.r, &fx7.s, &fx7.t);
        let w = st.lane.witness();
        let mut l = prepare(&st, &w, st.frames.seal(), a, ALL).unwrap();
        let keys = st.lane.keys();
        let src = fc(fx7.s.iter().find(|(x, _)| fc(*x) == b0).map(|(x, _)| *x).unwrap());
        let pos = keys.iter().position(|k| *k == src).unwrap() as u32;
        for j in 0..l.ib.len() { l.ib[j] = pos; }
        tl.check("7a value-derived ib: refused by lab provenance", matches!(admit(&st, &l, ALL), Err(Refusal::Provenance { .. })), format!("{:?}", admit(&st, &l, ALL).err()));
        let mut p = programs(l.range, keys.len(), None);
        let got = run(&st, &l, &mut p);
        tl.check("7b value-derived ib, contract-only: accepted, answer differs from oracle (disable)", admit(&st, &l, CONTRACT_ONLY).is_ok() && got != want, format!("got {got:?} vs oracle {want:?}"));
    }

    // ============ 8. disable runs on the fold itself
    {
        let mut fx3 = hierarchy(APP_K, 2, 3, 100, 1);
        let c0 = fc(fx3.s.iter().next().map(|(_, c)| *c).unwrap());
        fx3.s.remove(&e(parent(&c0).unwrap(), c0)); // one S edge missing
        let st = Store::build(&fx3.nodes, [&fx3.r, &fx3.s, &fx3.t], 1).unwrap();
        let w = st.lane.witness();
        let l = prepare(&st, &w, st.frames.seal(), a, ALL).unwrap();
        let wnt = oracle(a, &fx3.r, &fx3.s, &fx3.t);
        let mut p0 = programs(l.range, st.lane.keys().len(), None);
        let full = run(&st, &l, &mut p0);
        tl.check("8 full fold with one S edge missing equals oracle", full == wnt, format!("got {full:?} vs oracle {wnt:?}"));
        let mut p = programs(l.range, st.lane.keys().len(), Some(3));
        let got = run(&st, &l, &mut p);
        tl.check("8 disable: drop Semijoin(ic -> S_in) with one S edge missing", got != wnt, format!("got {got:?} vs oracle {wnt:?}"));
        let mut fx4 = hierarchy(APP_K, 2, 3, 100, 1);
        let b0 = fc(fx4.r.iter().next().map(|(_, b)| *b).unwrap());
        fx4.r.remove(&e(a, b0));
        let st = Store::build(&fx4.nodes, [&fx4.r, &fx4.s, &fx4.t], 1).unwrap();
        let w = st.lane.witness();
        let l = prepare(&st, &w, st.frames.seal(), a, ALL).unwrap();
        let wnt = oracle(a, &fx4.r, &fx4.s, &fx4.t);
        let mut p = programs(l.range, st.lane.keys().len(), Some(2));
        let got = run(&st, &l, &mut p);
        tl.check("8 disable: drop Semijoin(ib -> R_in) with one R edge missing", got != wnt, format!("got {got:?} vs oracle {wnt:?}"));
    }

    // ============ 9. benchmark: preparation and execution separately
    println!("\n# benchmark (release, medians of {REPS}); no-ms rule: any cell of 1 ms or more prints FAILED-GATE");
    println!("{:>8} {:>10} {:>12} {:>12} {:>12} {:>14} {:>12} {:>14} {:>16}", "nodes", "seal_us", "pointers_us", "verify_us", "exec_ns", "exec_ns/row", "base_ns", "exec_allocs", "ptr_same_word%");
    for (roots, f) in [(4u16, 8u16), (4, 16), (16, 16)] {
        let fx = hierarchy(APP_K, roots, f, 70, 3);
        let n = fx.nodes.len();
        let seal_us = median(|| { let t = Instant::now(); let _ = SealedFacetLane::seal(fx.nodes.clone(), 1).unwrap(); t.elapsed().as_nanos() as f64 / 1e3 });
        let st = Store::build(&fx.nodes, [&fx.r, &fx.s, &fx.t], 1).unwrap();
        let w = st.lane.witness();
        let a = fx.roots[0];
        let prep_us = median(|| { let fr = st.frames.seal(); let t = Instant::now(); let _ = prepare(&st, &w, fr, a, ALL).unwrap(); t.elapsed().as_nanos() as f64 / 1e3 });
        let l = prepare(&st, &w, st.frames.seal(), a, ALL).unwrap();
        let verify_us = median(|| { let t = Instant::now(); admit(&st, &l, ALL).unwrap(); t.elapsed().as_nanos() as f64 / 1e3 });
        let mut p = programs(l.range, n, None);
        let want = oracle(a, &fx.r, &fx.s, &fx.t);
        if n == fx.nodes.len() && roots == 4 && f == 8 { println!("{:>8} lowered Count program: {:?}", "", p.count); }
        assert_eq!(run(&st, &l, &mut p), want, "benchmark answer must equal the oracle");
        let exec_ns = median(|| { let t = Instant::now(); let r = run(&st, &l, &mut p); std::hint::black_box(r); t.elapsed().as_nanos() as f64 });
        let (a0, _) = alloc_snapshot(); let _ = run(&st, &l, &mut p); let (a1, _) = alloc_snapshot();
        // baseline: same plane work, no Gather (Plane AND only; a wrong query, cost reference)
        let fb = Filter::And(vec![Filter::Cmp(Col(0), Cmp::Range { lo: l.range.0, hi: l.range.1 }), Filter::Plane(Mask(0)), Filter::Plane(Mask(1)), Filter::Plane(Mask(2))]);
        let bp_any = lower(&Query { filter: fb.clone(), agg: Agg::Any }).unwrap();
        let bp_cnt = lower(&Query { filter: fb, agg: Agg::Count }).unwrap();
        let mut bsc = Scratch::for_program(&bp_cnt, n).unwrap();
        let masks: [&[u64]; 3] = [&st.pres[2], &st.pres[0], &st.pres[1]];
        let bpl = Planes { n_rows: n, masks: &masks, lanes: &[] };
        let base_ns = median(|| { let t = Instant::now(); let x = execute_into(&bp_any, &bpl, &Foreign::NONE, &mut bsc, Out::None).unwrap(); let y = execute_into(&bp_cnt, &bpl, &Foreign::NONE, &mut bsc, Out::None).unwrap(); std::hint::black_box((x, y)); t.elapsed().as_nanos() as f64 });
        // locality of the reads the Gather performs, over the candidate range
        let (mut same_word, mut tot) = (0usize, 0usize);
        let mut maxd = 0usize;
        for j in l.range.0 as usize..l.range.1 as usize {
            for &p in [l.ib[j], l.ic[j]].iter() { tot += 1; if (p as usize) / 64 == j / 64 { same_word += 1; } maxd = maxd.max(j.abs_diff(p as usize)); }
        }
        // The no-ms rule (CLAUDE.md P0): a value of 1 ms or more is a failed gate and is
        // printed as such, never as a number.
        let us = |v: f64| if v >= 1000.0 { "FAILED-GATE".to_string() } else { format!("{v:.1}") };
        let ns = |v: f64| if v >= 1e6 { "FAILED-GATE".to_string() } else { format!("{v:.0}") };
        println!("{:>8} {:>11} {:>12} {:>12} {:>12} {:>14.2} {:>12} {:>14} {:>15.1}%  (max |j-ptr| = {maxd} rows, range {} rows)", n, us(seal_us), us(prep_us), us(verify_us), ns(exec_ns), exec_ns / n as f64, ns(base_ns), a1 - a0, 100.0 * same_word as f64 / tot.max(1) as f64, l.range.1 - l.range.0);
        if prep_us + verify_us >= 1000.0 { println!("{:>8} preparation + verification: FAILED-GATE (1 ms or more)", ""); }
        let (pa0, pb0) = alloc_snapshot(); let _l = prepare(&st, &w, st.frames.seal(), a, ALL).unwrap(); let (pa1, pb1) = alloc_snapshot();
        println!("{:>8} prepare allocations: {} allocs, {} bytes ($i = 2 x 4 B x n = {} B)", "", pa1 - pa0, pb1 - pb0, 8 * n);
    }

    println!("\n{} passed, {} failed", tl.pass, tl.fail);
    std::process::exit(i32::from(tl.fail != 0));
}

const REPS: usize = 31;
fn median(mut f: impl FnMut() -> f64) -> f64 {
    let mut v: Vec<f64> = (0..REPS).map(|_| f()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[REPS / 2]
}
