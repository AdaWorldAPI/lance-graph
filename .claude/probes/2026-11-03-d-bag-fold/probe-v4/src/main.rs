//! D-BAG-FOLD v4. Two experiments, lab only, own workspace.
//!
//! A. An attested pointer lens: witnesses minted at construction that bind the
//!    presence planes to the sealed key lane BY KEY (not by bit pattern), the
//!    pointer table to its address-derived origin, and both to one lane version,
//!    one frame and one presence generation. Built on production types wherever
//!    one exists: `SealedFacetLane`, `OrderedLaneWitness`, `ordered_lane::digest_of`,
//!    `SemanticPrefix`, `lance_graph_dt64::dr::FrameStore`.
//! B. Candidate-gated execution: one hand-built mask-risc `Program` (frozen IR, used
//!    unchanged) whose two `Gather`s are gated by an in-program register holding
//!    `candidate range ∧ T_in`, compared with Arm L as Quack lowers it.
//!
//! Query: ∃d. R(a,b(d)) ∧ S(b(d),c(d)) ∧ T(c(d),d), answered as the count of
//! distinct valid d (keys unique). Oracle: nested loops over the edge sets.

use lance_graph_contract::facet::{FacetCascade, SemanticLens, SemanticPrefix};
use lance_graph_contract::ordered_lane::{digest_of, OrderedLaneWitness, SealedFacetLane, WitnessError};
use lance_graph_dt64::dr::{Frame, FrameRefusal, FrameRevisions, FrameStore};
use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, ForeignPlane as FP, LaneRef, MaskOp, Operand, Out, Planes, Pred,
    Program, Scratch, Terminal, Value,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, ForeignPlane, Mask, Query};
use std::collections::BTreeSet;
use std::time::Instant;

// ------------------------------------------------------------------ keys
const CANON: u16 = 0x0905;
const APP_K: u16 = 0x000A;
const APP_K2: u16 = 0x000B;
fn key(app: u16, t: [u16; 4]) -> FacetCascade {
    FacetCascade::from_semantic_tiles([CANON, app, t[0], t[1], t[2], t[3], 0, 0])
}
fn tiles(k: &FacetCascade) -> [u16; 8] { k.semantic_tiles() }
fn depth(k: &FacetCascade) -> usize { tiles(k)[2..6].iter().take_while(|&&t| t != 0).count() }
fn parent(k: &FacetCascade) -> Option<FacetCascade> {
    let d = depth(k);
    if d < 2 { return None; }
    let mut t = tiles(k);
    t[2 + d - 1] = 0;
    Some(FacetCascade::from_semantic_tiles(t))
}
fn classid(k: &FacetCascade) -> u32 { let t = tiles(k); (u32::from(t[0]) << 16) | u32::from(t[1]) }
type Edges = BTreeSet<(u128, u128)>;
fn fc(v: u128) -> FacetCascade { FacetCascade::from_u128(v) }
fn e(x: FacetCascade, y: FacetCascade) -> (u128, u128) { (x.as_u128(), y.as_u128()) }
fn set(p: &mut [u64], i: usize) { p[i / 64] |= 1 << (i % 64); }
fn get(p: &[u64], i: usize) -> bool { p[i / 64] >> (i % 64) & 1 == 1 }
fn fnv32(v: &[u32]) -> u64 { v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ u64::from(*x)).wrapping_mul(0x0100_0000_01b3)) }

// ------------------------------------------------------------------ store
struct Store {
    lane: SealedFacetLane,
    pres: [Vec<u64>; 3],
    frames: FrameStore,
    /// Presence generation, bumped by every mutation that goes through the API.
    generation: u64,
}

#[derive(Debug, PartialEq)]
#[allow(dead_code)] // read through Debug
enum Refusal {
    Witness(WitnessError),
    Frame(FrameRefusal),
    DuplicateKey,
    NonRefiningEdge,
    UnknownNode,
    Generation { witnessed: u64, now: u64 },
    PlaneNotAligned { plane: usize },
    PointerOutOfBounds,
    PointerOrigin,
    PointerTampered,
    ClassScope,
    LensMismatch,
}

/// How a plane's bits are stamped. `Misaligned` reproduces a construction bug:
/// every bit lands one row after the edge's child row.
#[derive(Clone, Copy, PartialEq)]
enum Stamp { Aligned, Misaligned }

impl Store {
    fn build(nodes: &[FacetCascade], rels: [&Edges; 3], stamp: Stamp) -> Result<Store, Refusal> {
        let lane = SealedFacetLane::seal(nodes.to_vec(), 1).map_err(Refusal::Witness)?;
        let n = lane.keys().len();
        let w = lane.witness();
        let mut pres = [vec![0u64; words_for(n)], vec![0u64; words_for(n)], vec![0u64; words_for(n)]];
        for (r, edges) in rels.iter().enumerate() {
            for &(x, y) in edges.iter() {
                let (x, y) = (fc(x), fc(y));
                if parent(&y).as_ref() != Some(&x) { return Err(Refusal::NonRefiningEdge); }
                let i = lane.seek_from(&w, 0, &y).map_err(Refusal::Witness)? as usize;
                if i >= n || lane.keys()[i] != y { return Err(Refusal::UnknownNode); }
                let at = if stamp == Stamp::Misaligned { (i + 1).min(n - 1) } else { i };
                set(&mut pres[r], at);
            }
        }
        let frames = FrameStore::new(FrameRevisions { source: 1, relation_map: 1, coordinates: 1 });
        Ok(Store { lane, pres, frames, generation: 1 })
    }
    /// The protocol write: change the bit, bump the generation, advance the frame.
    fn delete(&mut self, plane: usize, row: usize) {
        self.pres[plane][row / 64] &= !(1 << (row % 64));
        self.generation += 1;
        let mut r = self.frames.revisions();
        r.source += 1;
        self.frames.advance(r);
    }
}

// ------------------------------------------------------------------ Experiment A
/// Presence witness, minted from the SOURCE FACTS (the edge sets), never from the
/// planes. For each plane it records `digest_of` (production) over the sorted child
/// keys of that relation's edges, and the edge count. Verification recomputes the
/// same digest over the KEYS AT THE SET BITS of the plane. A plane stamped at the
/// wrong rows names other keys, so it fails, however the bits are arranged.
struct PresenceWitness {
    lane: OrderedLaneWitness,
    frame: Frame,
    generation: u64,
    keys: [u64; 3],
    pop: [u32; 3],
}

fn mint_presence(s: &Store, rels: [&Edges; 3], frame: Frame) -> Result<PresenceWitness, Refusal> {
    s.frames.check(&frame).map_err(Refusal::Frame)?;
    let keys = s.lane.keys();
    if keys.windows(2).any(|p| p[0] == p[1]) { return Err(Refusal::DuplicateKey); }
    let mut digests = [0u64; 3];
    let mut pop = [0u32; 3];
    for (r, edges) in rels.iter().enumerate() {
        let mut children: Vec<FacetCascade> = edges.iter().map(|&(_, y)| fc(y)).collect();
        children.sort_by(FacetCascade::cmp_numeric_projection);
        digests[r] = digest_of(&children);
        pop[r] = children.len() as u32;
    }
    Ok(PresenceWitness { lane: s.lane.witness(), frame, generation: s.generation, keys: digests, pop })
}

fn verify_presence(s: &Store, w: &PresenceWitness) -> Result<(), Refusal> {
    s.lane.validate(&w.lane).map_err(Refusal::Witness)?;
    s.frames.check(&w.frame).map_err(Refusal::Frame)?;
    if w.generation != s.generation { return Err(Refusal::Generation { witnessed: w.generation, now: s.generation }); }
    let keys = s.lane.keys();
    let mut at_bits: Vec<FacetCascade> = Vec::new();
    for (r, plane) in s.pres.iter().enumerate() {
        at_bits.clear();
        for (wi, &word) in plane.iter().enumerate() {
            let mut b = word;
            while b != 0 {
                let i = wi * 64 + b.trailing_zeros() as usize;
                // A bit past the last row names no key: the plane is not this lane's.
                if i >= keys.len() { return Err(Refusal::PlaneNotAligned { plane: r }); }
                at_bits.push(keys[i]);
                b &= b - 1;
            }
        }
        if at_bits.len() as u32 != w.pop[r] || digest_of(&at_bits) != w.keys[r] {
            return Err(Refusal::PlaneNotAligned { plane: r });
        }
    }
    Ok(())
}

/// Pointer witness: the pointer table plus what it was derived under.
struct PointerLens {
    lane: OrderedLaneWitness,
    frame: Frame,
    lens: SemanticLens,
    class: u32,
    range: (u32, u32),
    sentinel: u32,
    ib: Vec<u32>,
    ic: Vec<u32>,
    digest: u64,
}

/// Mint: witnessed seeks keyed by each row's own key arithmetic (`seek_from`, one
/// cursor per ancestor depth), then verify the origin of every pointer once by
/// recomputation, the bounds, and the class scope. Seals the table with a digest.
fn mint_pointers(s: &Store, w: &OrderedLaneWitness, frame: Frame, a: FacetCascade) -> Result<PointerLens, Refusal> {
    s.lane.validate(w).map_err(Refusal::Witness)?;
    s.frames.check(&frame).map_err(Refusal::Frame)?;
    let keys = s.lane.keys();
    let n = keys.len();
    if keys.windows(2).any(|p| p[0] == p[1]) { return Err(Refusal::DuplicateKey); }
    let prefix = SemanticPrefix::of(a, 3);
    if prefix.lens() != w.lens() { return Err(Refusal::LensMismatch); }
    let range = s.lane.bound(w, &prefix).map_err(Refusal::Witness)?;
    // Sentinel: a row with R and S both clear. Its use is checked below.
    let sentinel = (0..n).find(|&i| !get(&s.pres[0], i) && !get(&s.pres[1], i)).unwrap_or(0) as u32;
    let mut ib = vec![sentinel; n];
    let mut ic = vec![sentinel; n];
    for want in 1..=4 {
        let mut cursor = [0u32; 5];
        for (j, k) in keys.iter().enumerate() {
            if depth(k) != want { continue; }
            for (target, slot) in [(parent(k), &mut ic), (parent(k).as_ref().and_then(parent), &mut ib)] {
                let Some(t) = target else { continue };
                let d = depth(&t);
                let i = s.lane.seek_from(w, cursor[d], &t).map_err(Refusal::Witness)?;
                cursor[d] = i;
                if (i as usize) < n && keys[i as usize] == t { slot[j] = i; }
            }
        }
    }
    let lens = PointerLens { lane: *w, frame, lens: w.lens(), class: classid(&a), range, sentinel, ib, ic, digest: 0 };
    check_origin(keys, &lens)?;
    let digest = fnv32(&lens.ib) ^ fnv32(&lens.ic).rotate_left(1);
    Ok(PointerLens { digest, ..lens })
}

/// Origin, bounds and class scope, by recomputation from the keys alone.
fn check_origin(keys: &[FacetCascade], l: &PointerLens) -> Result<(), Refusal> {
    let n = keys.len() as u32;
    let exists = |t: &FacetCascade| keys.binary_search_by(|k| k.cmp_numeric_projection(t)).is_ok();
    for (j, k) in keys.iter().enumerate() {
        for (p, want) in [(l.ic[j], parent(k)), (l.ib[j], parent(k).as_ref().and_then(parent))] {
            if p >= n { return Err(Refusal::PointerOutOfBounds); }
            match want {
                Some(t) if keys[p as usize] == t => {
                    if (l.range.0..l.range.1).contains(&(j as u32)) && classid(&t) != l.class { return Err(Refusal::ClassScope); }
                }
                Some(t) if p == l.sentinel && !exists(&t) => {}
                None if p == l.sentinel => {}
                _ => return Err(Refusal::PointerOrigin),
            }
        }
    }
    Ok(())
}

/// Execution-time verification: O(1) witness and frame checks, then the table digest.
fn verify_pointers(s: &Store, l: &PointerLens) -> Result<(), Refusal> {
    s.lane.validate(&l.lane).map_err(Refusal::Witness)?;
    s.frames.check(&l.frame).map_err(Refusal::Frame)?;
    if l.lens != l.lane.lens() { return Err(Refusal::LensMismatch); }
    if fnv32(&l.ib) ^ fnv32(&l.ic).rotate_left(1) != l.digest { return Err(Refusal::PointerTampered); }
    Ok(())
}

// ------------------------------------------------------------------ Experiment B
/// Arm L exactly as Quack lowers it (Count terminal).
fn arm_l_quack(range: (u32, u32)) -> Program {
    let f = Filter::And(vec![
        Filter::Cmp(Col(0), Cmp::Range { lo: range.0, hi: range.1 }),
        Filter::Plane(Mask(0)),
        Filter::Semijoin { fk: Col(0), foreign: ForeignPlane(0) },
        Filter::Semijoin { fk: Col(1), foreign: ForeignPlane(1) },
    ]);
    lower(&Query { filter: f, agg: Agg::Count }).expect("lowers")
}

/// Candidate-gated: s0 = Range ∧ T; s1 = R@ib under s0; s0 = S@ic under s1; Count(s0).
/// `extra` repeats the two Gathers (same answer) to measure cost per resident witness.
/// `drop` removes one Gather for the disable controls (0 = R, 1 = S).
fn gated(range: (u32, u32), extra: usize, drop: Option<usize>) -> Program {
    let mut ops = vec![MaskOp::Pred { pred: Pred::Range { lo: range.0, hi: range.1 }, under: Some(Operand::Plane(0)), dst: 0 }];
    let mut cur = 0u16;
    for _ in 0..=extra {
        for (lane, foreign) in [(0u16, 0u16), (1, 1)] {
            if drop == Some(lane as usize) { continue; }
            let dst = 1 - cur;
            ops.push(MaskOp::Gather { lane, foreign, under: Some(Operand::Scratch(cur)), dst });
            cur = dst;
        }
    }
    Program::new(ops, Terminal::Count { mask: Operand::Scratch(cur) })
}

fn run(s: &Store, l: &PointerLens, prog: &Program, sc: &mut Scratch<'static>) -> u64 {
    let n = s.lane.keys().len();
    let lanes = [LaneRef::U32(&l.ib), LaneRef::U32(&l.ic)];
    let fps = [FP { words: &s.pres[0], rows: n }, FP { words: &s.pres[1], rows: n }];
    let foreign = Foreign { planes: &fps, lanes: &[] };
    let masks: [&[u64]; 1] = [&s.pres[2]];
    let pl = Planes { n_rows: n, masks: &masks, lanes: &lanes };
    match execute_into(prog, &pl, &foreign, sc, Out::None).expect("runs") { Value::Count(c) => c as u64, v => panic!("{v:?}") }
}

fn oracle(a: FacetCascade, r: &Edges, s: &Edges, t: &Edges) -> u64 {
    let a = a.as_u128();
    let mut ds = BTreeSet::new();
    for &(x, b) in r { if x != a { continue; }
        for &(b2, c) in s { if b2 != b { continue; }
            for &(c2, d) in t { if c2 == c { ds.insert(d); } } } }
    ds.len() as u64
}

// ------------------------------------------------------------------ fixtures
struct Fx { nodes: Vec<FacetCascade>, r: Edges, s: Edges, t: Edges, roots: Vec<FacetCascade> }
/// One root per entry of `fanouts`, depth 4, root `i` with fanout `fanouts[i]`.
fn forest(app: u16, fanouts: &[u16], keep_pct: u64, seed: u64) -> Fx {
    let mut st = seed;
    let mut coin = || { st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (st >> 33) % 100 < keep_pct };
    let (mut nodes, mut r, mut s, mut t, mut roots) = (vec![], Edges::new(), Edges::new(), Edges::new(), vec![]);
    for (ri, &f) in fanouts.iter().enumerate() {
        let a1 = ri as u16 + 1;
        let a = key(app, [a1, 0, 0, 0]); nodes.push(a); roots.push(a);
        for b1 in 1..=f { let b = key(app, [a1, b1, 0, 0]); nodes.push(b); if coin() { r.insert(e(a, b)); }
            for c1 in 1..=f { let c = key(app, [a1, b1, c1, 0]); nodes.push(c); if coin() { s.insert(e(b, c)); }
                for d1 in 1..=f { let d = key(app, [a1, b1, c1, d1]); nodes.push(d); if coin() { t.insert(e(c, d)); } } } }
    }
    Fx { nodes, r, s, t, roots }
}

struct Tally(usize, usize);
impl Tally {
    fn check(&mut self, name: &str, ok: bool, detail: String) {
        if ok { self.0 += 1 } else { self.1 += 1 }
        println!("{} {:<70} {}", if ok { "PASS" } else { "FAIL" }, name, detail);
    }
}

const REPS: usize = 101;
fn median(mut f: impl FnMut() -> f64) -> f64 {
    let mut v: Vec<f64> = (0..REPS).map(|_| f()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[REPS / 2]
}
fn show(v_ns: f64) -> String { if v_ns >= 1e6 { "FAILED-GATE".into() } else { format!("{:.1}", v_ns / 1e3) } }

fn main() {
    let mut tl = Tally(0, 0);
    println!("# A. attested pointer lens");

    let fx = forest(APP_K, &[3, 3], 100, 1);
    let a = fx.roots[0];
    let rels = [&fx.r, &fx.s, &fx.t];
    let want = oracle(a, &fx.r, &fx.s, &fx.t);

    // A1 happy path, and witnesses mint and verify
    {
        let st = Store::build(&fx.nodes, rels, Stamp::Aligned).unwrap();
        let pw = mint_presence(&st, rels, st.frames.seal()).unwrap();
        let l = mint_pointers(&st, &st.lane.witness(), st.frames.seal(), a).unwrap();
        let ok = verify_presence(&st, &pw).is_ok() && verify_pointers(&st, &l).is_ok();
        let mut sc = Scratch::for_program(&gated(l.range, 0, None), st.lane.keys().len()).unwrap();
        let got = run(&st, &l, &gated(l.range, 0, None), &mut sc);
        tl.check("A1 aligned store: both witnesses verify, gated answer = oracle", ok && got == want, format!("got {got} oracle {want}"));
    }
    // A2 plane misaligned AT CONSTRUCTION
    {
        let st = Store::build(&fx.nodes, rels, Stamp::Misaligned).unwrap();
        let pw = mint_presence(&st, rels, st.frames.seal()).unwrap();
        let r = verify_presence(&st, &pw);
        tl.check("A2 plane misaligned at construction: refused by key-bound presence witness", matches!(r, Err(Refusal::PlaneNotAligned { .. })), format!("{:?}", r.err()));
        let l = mint_pointers(&st, &st.lane.witness(), st.frames.seal(), a).unwrap();
        let mut sc = Scratch::for_program(&gated(l.range, 0, None), st.lane.keys().len()).unwrap();
        let got = run(&st, &l, &gated(l.range, 0, None), &mut sc);
        tl.check("A2' same store, presence witness skipped: silent wrong answer (disable)", got != want, format!("got {got} oracle {want}"));
    }
    // A3 permuted after construction
    {
        let mut st = Store::build(&fx.nodes, rels, Stamp::Aligned).unwrap();
        let pw = mint_presence(&st, rels, st.frames.seal()).unwrap();
        st.pres[1].reverse();
        tl.check("A3 plane permuted after construction: refused", matches!(verify_presence(&st, &pw), Err(Refusal::PlaneNotAligned { .. })), format!("{:?}", verify_presence(&st, &pw).err()));
    }
    // A4 mutation through the API: generation + frame
    {
        let mut st = Store::build(&fx.nodes, rels, Stamp::Aligned).unwrap();
        let pw = mint_presence(&st, rels, st.frames.seal()).unwrap();
        let l = mint_pointers(&st, &st.lane.witness(), st.frames.seal(), a).unwrap();
        let row = (0..st.lane.keys().len()).find(|&i| get(&st.pres[1], i)).unwrap();
        st.delete(1, row);
        let r = verify_presence(&st, &pw);
        tl.check("A4 deletion through the API: presence witness refused (frame)", matches!(r, Err(Refusal::Frame(FrameRefusal::SourceChanged))), format!("{:?}", r.err()));
        let r = verify_pointers(&st, &l);
        tl.check("A4' same deletion: pointer lens refused too (frame), though its keys are unchanged", matches!(r, Err(Refusal::Frame(FrameRefusal::SourceChanged))), format!("{:?}", r.err()));
    }
    // A5 mutation bypassing the API (no generation, no frame)
    {
        let mut st = Store::build(&fx.nodes, rels, Stamp::Aligned).unwrap();
        let pw = mint_presence(&st, rels, st.frames.seal()).unwrap();
        let row = (0..st.lane.keys().len()).find(|&i| get(&st.pres[1], i)).unwrap();
        st.pres[1][row / 64] &= !(1 << (row % 64));
        tl.check("A5 deletion bypassing generation and frame: refused by key digest", matches!(verify_presence(&st, &pw), Err(Refusal::PlaneNotAligned { plane: 1 })), format!("{:?}", verify_presence(&st, &pw).err()));
    }
    // A6 duplicate keys
    {
        let mut nodes = fx.nodes.clone();
        let d = fc(fx.t.iter().next().map(|(_, d)| *d).unwrap());
        nodes.push(d);
        let st = Store::build(&nodes, rels, Stamp::Aligned).unwrap();
        tl.check("A6 duplicate key: production seal accepts; presence mint refuses", matches!(mint_presence(&st, rels, st.frames.seal()), Err(Refusal::DuplicateKey)), String::new());
        tl.check("A6' duplicate key: pointer mint refuses", matches!(mint_pointers(&st, &st.lane.witness(), st.frames.seal(), a), Err(Refusal::DuplicateKey)), String::new());
    }
    // A7 pointer bounds / origin / tampering / class scope
    {
        let st = Store::build(&fx.nodes, rels, Stamp::Aligned).unwrap();
        let w = st.lane.witness();
        let l = mint_pointers(&st, &w, st.frames.seal(), a).unwrap();
        let keys = st.lane.keys();
        let n = keys.len() as u32;
        let j = (l.range.0..l.range.1).find(|&j| depth(&keys[j as usize]) == 4).unwrap() as usize;
        let mut bad = PointerLens { ib: l.ib.clone(), ic: l.ic.clone(), frame: st.frames.seal(), ..l };
        bad.ib[j] = n;
        tl.check("A7a pointer = n_rows after mint: refused (table digest)", matches!(verify_pointers(&st, &bad), Err(Refusal::PointerTampered)), String::new());
        tl.check("A7b same table offered as a fresh mint: refused (bounds)", matches!(check_origin(keys, &bad), Err(Refusal::PointerOutOfBounds)), String::new());
        let b0 = (0..n as usize).find(|&i| depth(&keys[i]) == 2 && keys[i] != keys[l.ib[j] as usize]).unwrap() as u32;
        let mut val = PointerLens { ib: vec![b0; l.ib.len()], ic: l.ic.clone(), frame: st.frames.seal(), ..l };
        val.digest = fnv32(&val.ib) ^ fnv32(&val.ic).rotate_left(1); // re-sealed by a forger
        tl.check("A7c value-chosen ib, re-sealed digest: verify passes (gap) ...", verify_pointers(&st, &val).is_ok(), String::new());
        tl.check("A7c' ... but the mint-time origin check refuses it", matches!(check_origin(keys, &val), Err(Refusal::PointerOrigin)), String::new());
    }
    {
        // class scope: a second class with identical tiers
        let mut fx2 = forest(APP_K, &[3, 3], 100, 1);
        let other = forest(APP_K2, &[3, 3], 100, 1);
        fx2.nodes.extend(other.nodes.iter().copied());
        let st = Store::build(&fx2.nodes, [&fx2.r, &fx2.s, &fx2.t], Stamp::Aligned).unwrap();
        let l = mint_pointers(&st, &st.lane.witness(), st.frames.seal(), a).unwrap();
        let keys = st.lane.keys();
        let j = (l.range.0..l.range.1).find(|&j| depth(&keys[j as usize]) == 4).unwrap() as usize;
        let mut t = tiles(&keys[l.ic[j] as usize]); t[1] = APP_K2;
        let tgt = FacetCascade::from_semantic_tiles(t);
        let mut x = PointerLens { ib: l.ib.clone(), ic: l.ic.clone(), frame: st.frames.seal(), ..l };
        x.ic[j] = keys.iter().position(|k| *k == tgt).unwrap() as u32;
        tl.check("A7d pointer into another ClassId, same tiers: refused at mint", matches!(check_origin(keys, &x), Err(Refusal::PointerOrigin | Refusal::ClassScope)), format!("{:?}", check_origin(keys, &x).err()));
    }
    // A8 frame, version, lens
    {
        let st = Store::build(&fx.nodes, rels, Stamp::Aligned).unwrap();
        let foreign = FrameStore::new(FrameRevisions { source: 1, relation_map: 1, coordinates: 1 }).seal();
        tl.check("A8a foreign frame: refused at mint (production FrameStore)", matches!(mint_pointers(&st, &st.lane.witness(), foreign, a), Err(Refusal::Frame(FrameRefusal::ForeignFrame))), String::new());
        let l = mint_pointers(&st, &st.lane.witness(), st.frames.seal(), a).unwrap();
        let resealed = SealedFacetLane::seal(st.lane.keys().to_vec(), 2).unwrap();
        tl.check("A8b lane resealed at a new version: pointer lens witness refused (production)", matches!(resealed.validate(&l.lane), Err(WitnessError::VersionMismatch { .. })), String::new());
        tl.check("A8c lens mismatch: not constructible (one SemanticLens variant)", SemanticLens::CanonHighTiles8 == l.lens, "gap: untestable".into());
    }

    println!("\n# B. candidate-gated execution");
    // B1 parity on every root, varied fanouts and edge loss, both programs
    {
        let fx = forest(APP_K, &[2, 3, 4, 6, 8], 70, 9);
        let st = Store::build(&fx.nodes, [&fx.r, &fx.s, &fx.t], Stamp::Aligned).unwrap();
        let w = st.lane.witness();
        let n = st.lane.keys().len();
        let mut ok = true;
        let mut detail = String::new();
        for &a in &fx.roots {
            let l = mint_pointers(&st, &w, st.frames.seal(), a).unwrap();
            let (pg, pq) = (gated(l.range, 0, None), arm_l_quack(l.range));
            let mut s1 = Scratch::for_program(&pg, n).unwrap();
            let mut s2 = Scratch::for_program(&pq, n).unwrap();
            let (g, q, o) = (run(&st, &l, &pg, &mut s1), run(&st, &l, &pq, &mut s2), oracle(a, &fx.r, &fx.s, &fx.t));
            if g != o || q != o { ok = false; detail = format!("gated {g} quack {q} oracle {o}"); }
        }
        tl.check(&format!("B1 gated = Quack Arm L = oracle on all {} roots ({n} rows)", fx.roots.len()), ok, detail);
        // no Gather gated by a resident plane in the gated program; both Gathers gated by a register
        let pg = gated((0, 1), 0, None);
        let regs = pg.ops.iter().filter(|o| matches!(o, MaskOp::Gather { under: Some(Operand::Scratch(_)), .. })).count();
        tl.check("B1' gated program: both Gathers gated by an in-program register", regs == 2, format!("{:?}", pg.ops));
    }
    // B2 disable controls
    {
        let mut fxr = forest(APP_K, &[3, 3], 100, 1);
        let b0 = fc(fxr.r.iter().next().map(|(_, b)| *b).unwrap());
        fxr.r.remove(&e(a, b0));
        let mut fxs = forest(APP_K, &[3, 3], 100, 1);
        let c0 = fc(fxs.s.iter().next().map(|(_, c)| *c).unwrap());
        fxs.s.remove(&e(parent(&c0).unwrap(), c0));
        for (name, f, drop) in [("drop Gather(ib -> R_in), one R edge missing", &fxr, 0usize), ("drop Gather(ic -> S_in), one S edge missing", &fxs, 1)] {
            let st = Store::build(&f.nodes, [&f.r, &f.s, &f.t], Stamp::Aligned).unwrap();
            let l = mint_pointers(&st, &st.lane.witness(), st.frames.seal(), a).unwrap();
            let o = oracle(a, &f.r, &f.s, &f.t);
            let n = st.lane.keys().len();
            let full = gated(l.range, 0, None);
            let mut sc = Scratch::for_program(&full, n).unwrap();
            let g = run(&st, &l, &full, &mut sc);
            let dp = gated(l.range, 0, Some(drop));
            let mut sd = Scratch::for_program(&dp, n).unwrap();
            let d = run(&st, &l, &dp, &mut sd);
            tl.check(&format!("B2 disable: {name}"), g == o && d != o, format!("full {g}, dropped {d}, oracle {o}"));
        }
    }

    // B3 scaling. Exec timings in µs; any value of 1 ms or more prints FAILED-GATE.
    println!("\n# B3 scaling (release, medians of {REPS}); µs; 1 ms or more prints FAILED-GATE");
    println!("{:>7} {:>8} {:>8} {:>9} {:>12} {:>11} {:>11} {:>12} {:>12} {:>9}", "lane", "cands", "witn", "mint_ptr", "mint_pres", "verify_ptr", "verify_pres", "armL_quack", "gated", "speedup");
    let mut rows = Vec::new();
    // full lane size: candidate root fixed (fanout 8), more filler roots
    for fill in [0usize, 7, 31] {
        let mut f = vec![8u16]; f.extend(std::iter::repeat_n(8u16, fill));
        rows.push((f, 0usize, 0usize));
    }
    // candidate population: one lane, query each root of different fanout
    for (i, _) in [2u16, 4, 8, 16].iter().enumerate() { rows.push((vec![2, 4, 8, 16, 16, 16], i, 0)); }
    // resident witnesses: repeat the Gather pair
    for extra in [1usize, 2] { rows.push((vec![8u16; 8], 0, extra)); }
    for (fanouts, ri, extra) in rows {
        let fx = forest(APP_K, &fanouts, 70, 5);
        let rels = [&fx.r, &fx.s, &fx.t];
        let st = Store::build(&fx.nodes, rels, Stamp::Aligned).unwrap();
        let w = st.lane.witness();
        let a = fx.roots[ri];
        let n = st.lane.keys().len();
        let mint_ptr = median(|| { let fr = st.frames.seal(); let t = Instant::now(); let _ = mint_pointers(&st, &w, fr, a).unwrap(); t.elapsed().as_nanos() as f64 });
        let mint_pres = median(|| { let fr = st.frames.seal(); let t = Instant::now(); let _ = mint_presence(&st, rels, fr).unwrap(); t.elapsed().as_nanos() as f64 });
        let l = mint_pointers(&st, &w, st.frames.seal(), a).unwrap();
        let pw = mint_presence(&st, rels, st.frames.seal()).unwrap();
        let verify_ptr = median(|| { let t = Instant::now(); verify_pointers(&st, &l).unwrap(); t.elapsed().as_nanos() as f64 });
        let verify_pres = median(|| { let t = Instant::now(); verify_presence(&st, &pw).unwrap(); t.elapsed().as_nanos() as f64 });
        let pq = if extra == 0 { arm_l_quack(l.range) } else {
            let mut v = vec![Filter::Cmp(Col(0), Cmp::Range { lo: l.range.0, hi: l.range.1 }), Filter::Plane(Mask(0))];
            for _ in 0..=extra { v.push(Filter::Semijoin { fk: Col(0), foreign: ForeignPlane(0) }); v.push(Filter::Semijoin { fk: Col(1), foreign: ForeignPlane(1) }); }
            lower(&Query { filter: Filter::And(v), agg: Agg::Count }).unwrap()
        };
        let pg = gated(l.range, extra, None);
        let mut sq = Scratch::for_program(&pq, n).unwrap();
        let mut sg = Scratch::for_program(&pg, n).unwrap();
        let o = oracle(a, &fx.r, &fx.s, &fx.t);
        assert_eq!(run(&st, &l, &pq, &mut sq), o, "Quack Arm L must equal the oracle");
        assert_eq!(run(&st, &l, &pg, &mut sg), o, "gated must equal the oracle");
        let tq = median(|| { let t = Instant::now(); std::hint::black_box(run(&st, &l, &pq, &mut sq)); t.elapsed().as_nanos() as f64 });
        let tg = median(|| { let t = Instant::now(); std::hint::black_box(run(&st, &l, &pg, &mut sg)); t.elapsed().as_nanos() as f64 });
        let cands = (l.range.0..l.range.1).filter(|&j| depth(&st.lane.keys()[j as usize]) == 4).count();
        println!("{:>7} {:>8} {:>8} {:>9} {:>12} {:>11} {:>11} {:>12} {:>12} {:>8.1}x", n, cands, 2 * (extra + 1) + 1, show(mint_ptr), show(mint_pres), show(verify_ptr), show(verify_pres), show(tq), show(tg), tq / tg);
    }

    println!("\n{} passed, {} failed", tl.0, tl.1);
    std::process::exit(i32::from(tl.1 != 0));
}
