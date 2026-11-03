//! D-BAG-FOLD v2: `∃d. R(a,b(d)) ∧ S(b(d),c(d)) ∧ T(c(d),d)` without storing any
//! composed relation. Lab probe, outside every workspace.
//!
//! Shape under test (ADMISSION B, `entries/2026-10-20-six-decisions.md` §1):
//!   PREPARE  counted, version-witnessed, same-table key seeks over an attested
//!            sorted key build a co-addressed PROJECTION: for every candidate d,
//!            the presence of each relation's edge re-addressed to d's row.
//!            Each plane is ONE relation's own presence; nothing composes here.
//!   EXECUTE  one Quack-lowered mask-risc Program: And(R', S', T') -> Any / Count.
//!            Row-aligned ops only. No Gather, no produced mask, no tuple.
//!   ORACLE   nested loops over the explicit edge sets, independent of the layout.
//!
//! Key: u64 = classid(32) | t1 | t2 | t3 | t4 (8 bits each, 0 = unused tier).
//! The witness is a probe-local mirror of `OrderedLaneWitness` (version, n_rows,
//! order-sensitive FNV digest of the key sequence) plus the version of the
//! presence planes, which no contract type covers today (see README).
use lance_graph_mask_risc::{execute_into, words_for, Foreign, Out, Planes, Scratch, Value};
use lance_graph_mask_risc::{ForeignPlane as FP, LaneRef};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, ForeignPlane, Mask, Query};
use std::collections::BTreeSet;

type Key = u64;
type Edges = BTreeSet<(Key, Key)>;

fn key(cls: u32, t: [u8; 4]) -> Key {
    (u64::from(cls) << 32) | (u64::from(t[0]) << 24) | (u64::from(t[1]) << 16) | (u64::from(t[2]) << 8) | u64::from(t[3])
}
fn depth(k: Key) -> usize { (0..4).take_while(|i| (k >> (24 - 8 * i)) & 0xFF != 0).count() }
/// Address arithmetic on a key's own bits; stays inside its classid.
fn parent(k: Key) -> Option<Key> {
    let d = depth(k);
    (d >= 2).then(|| k & !(0xFFu64 << (24 - 8 * (d - 1))))
}

#[derive(Debug, PartialEq)]
enum Refuse { StaleVersion, OrderNotAttested, DuplicateKey, NonRefiningEdge((Key, Key)), UnknownNode(Key) }

#[derive(Clone, Copy, Debug, PartialEq)]
struct Witness { key_version: u64, presence_version: u64, n_rows: usize, digest: u64 }

fn fnv(keys: &[Key]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for k in keys { for b in k.to_le_bytes() { h ^= u64::from(b); h = h.wrapping_mul(0x0100_0000_01b3); } }
    h
}

/// One table: a node lane plus three presence planes stamped at the CHILD row.
struct Store { key_version: u64, presence_version: u64, keys: Vec<Key>, pres: [Vec<bool>; 3] }

impl Store {
    /// Builds the layout from explicit edges. An edge the layout cannot place
    /// (child is not a one-tier extension of its source) is refused, never dropped.
    fn build(nodes: &[Key], rels: [&Edges; 3]) -> Result<Store, Refuse> {
        let mut keys = nodes.to_vec();
        keys.sort_unstable();
        let mut pres = [vec![false; keys.len()], vec![false; keys.len()], vec![false; keys.len()]];
        for (r, edges) in rels.iter().enumerate() {
            for &(x, y) in edges.iter() {
                if parent(y) != Some(x) { return Err(Refuse::NonRefiningEdge((x, y))); }
                let i = keys.binary_search(&y).map_err(|_| Refuse::UnknownNode(y))?;
                pres[r][i] = true;
            }
        }
        Ok(Store { key_version: 1, presence_version: 1, keys, pres })
    }
    /// The seal. Refuses a key lane that is not strictly increasing.
    fn seal(&self) -> Result<Witness, Refuse> {
        if self.keys.windows(2).any(|w| w[0] >= w[1]) {
            return Err(if self.keys.windows(2).any(|w| w[0] == w[1]) { Refuse::DuplicateKey } else { Refuse::OrderNotAttested });
        }
        Ok(Witness { key_version: self.key_version, presence_version: self.presence_version, n_rows: self.keys.len(), digest: fnv(&self.keys) })
    }
}

/// Prepared co-addressed projection over the candidate population.
struct Prepared { witness: Witness, planes: [Vec<u64>; 3], n: usize, seeks: usize }

/// `checked = false` skips every witness check: the deliberately broken arm.
fn prepare(s: &Store, w: &Witness, a: Key, checked: bool) -> Result<Prepared, Refuse> {
    if checked {
        if w.key_version != s.key_version || w.presence_version != s.presence_version || w.n_rows != s.keys.len() { return Err(Refuse::StaleVersion); }
        if fnv(&s.keys) != w.digest { return Err(Refuse::OrderNotAttested); }
    }
    // Candidate range: depth-4 keys under prefix(a), located by two seeks on the
    // attested sorted key (a's subtree is [a, a + 2^(8·(4-depth(a))) ) in key order).
    let span = 1u64 << (8 * (4 - depth(a)));
    let lo = s.keys.partition_point(|&k| k < a);
    let hi = s.keys.partition_point(|&k| k < a + span);
    let mut seeks = 2;
    let cand: Vec<usize> = (lo..hi).filter(|&i| depth(s.keys[i]) == 4).collect();
    let n = cand.len();
    let mut planes = [vec![0u64; words_for(n)], vec![0u64; words_for(n)], vec![0u64; words_for(n)]];
    let mut set = |p: usize, j: usize| planes[p][j / 64] |= 1 << (j % 64);
    for (j, &di) in cand.iter().enumerate() {
        let d = s.keys[di];
        let c = parent(d).expect("depth 4");
        let b = parent(c).expect("depth 3");
        let find = |k: Key| s.keys.binary_search(&k).ok(); // same-table seek, keyed by d's own address
        seeks += 2;
        if s.pres[2][di] { set(2, j); }
        if let Some(ci) = find(c) { if s.pres[1][ci] { set(1, j); } }
        if let Some(bi) = find(b) { if s.pres[0][bi] && parent(b) == Some(a) { set(0, j); } }
    }
    Ok(Prepared { witness: *w, planes, n, seeks })
}

/// One Program. Refuses a projection prepared at another version of the store.
fn execute(s: &Store, p: &Prepared, checked: bool) -> Result<(bool, u64), Refuse> {
    if checked && (p.witness.key_version != s.key_version || p.witness.presence_version != s.presence_version) { return Err(Refuse::StaleVersion); }
    let f = Filter::And(vec![Filter::Plane(Mask(0)), Filter::Plane(Mask(1)), Filter::Plane(Mask(2))]);
    let run = |agg| {
        let prog = lower(&Query { filter: f.clone(), agg }).expect("lowers");
        assert!(!format!("{prog:?}").contains("Gather"), "no Gather in the program");
        let masks: Vec<&[u64]> = p.planes.iter().map(|v| v.as_slice()).collect();
        let pl = Planes { n_rows: p.n, masks: &masks, lanes: &[] };
        let mut sc = Scratch::for_program(&prog, p.n.max(1)).expect("carves");
        execute_into(&prog, &pl, &Foreign::NONE, &mut sc, Out::None).expect("runs")
    };
    let any = match run(Agg::Any) { Value::Bool(b) => b, v => panic!("{v:?}") };
    let count = match run(Agg::Count) { Value::Count(c) => c as u64, v => panic!("{v:?}") };
    Ok((any, count))
}

fn pack(v: &[bool]) -> Vec<u64> {
    let mut w = vec![0u64; words_for(v.len())];
    for (i, &b) in v.iter().enumerate() { if b { w[i / 64] |= 1 << (i % 64); } }
    w
}

/// ARM L (zero-copy form). Preparation builds only pointer tables `$i`
/// (`ib[row]` = row of grandparent link b, `ic[row]` = row of parent link c),
/// by same-table seeks keyed by the row's own address. Execution is ONE Program
/// over the node lane: Range(subtree a) ∧ T_in ∧ Semijoin(ib → R_in) ∧
/// Semijoin(ic → S_in). R_in and S_in are the table's own RESIDENT planes, read
/// in place through `$i` by `MaskOp::Gather`. No presence bit is copied.
fn execute_lens(s: &Store, w: &Witness, a: Key) -> Result<(bool, u64), Refuse> {
    if w.key_version != s.key_version || w.presence_version != s.presence_version || w.n_rows != s.keys.len() { return Err(Refuse::StaleVersion); }
    if fnv(&s.keys) != w.digest { return Err(Refuse::OrderNotAttested); }
    let n = s.keys.len();
    let span = 1u64 << (8 * (4 - depth(a)));
    let lo = s.keys.partition_point(|&k| k < a) as u32;
    let hi = s.keys.partition_point(|&k| k < a + span) as u32;
    let find = |k: Option<Key>| k.and_then(|k| s.keys.binary_search(&k).ok());
    // A row with no such ancestor points at row 0 (the lane minimum); the
    // Range restricts to a's subtree, whose depth-4 rows always have both ancestors.
    let ib: Vec<u32> = s.keys.iter().map(|&k| find(parent(k).and_then(parent)).unwrap_or(0) as u32).collect();
    let ic: Vec<u32> = s.keys.iter().map(|&k| find(parent(k)).unwrap_or(0) as u32).collect();
    let t = pack(&s.pres[2]);
    let (r, sp) = (pack(&s.pres[0]), pack(&s.pres[1]));
    let f = Filter::And(vec![
        Filter::Cmp(Col(0), Cmp::Range { lo, hi }),
        Filter::Plane(Mask(0)),
        Filter::Semijoin { fk: Col(0), foreign: ForeignPlane(0) },
        Filter::Semijoin { fk: Col(1), foreign: ForeignPlane(1) },
    ]);
    let lanes = [LaneRef::U32(&ib), LaneRef::U32(&ic)];
    let fps = [FP { words: &r, rows: n }, FP { words: &sp, rows: n }];
    let foreign = lance_graph_mask_risc::Foreign { planes: &fps, lanes: &[] };
    let masks: Vec<&[u64]> = vec![&t];
    let pl = Planes { n_rows: n, masks: &masks, lanes: &lanes };
    let run = |agg| {
        let prog = lower(&Query { filter: f.clone(), agg }).expect("lowers");
        let mut sc = Scratch::for_program(&prog, n).expect("carves");
        execute_into(&prog, &pl, &foreign, &mut sc, Out::None).expect("runs")
    };
    let any = match run(Agg::Any) { Value::Bool(b) => b, v => panic!("{v:?}") };
    let count = match run(Agg::Count) { Value::Count(c) => c as u64, v => panic!("{v:?}") };
    Ok((any, count))
}

/// Independent oracle: nested loops over the three edge sets.
fn oracle(a: Key, r: &Edges, s: &Edges, t: &Edges) -> (bool, u64) {
    let mut ds = BTreeSet::new();
    for &(x, b) in r { if x != a { continue; }
        for &(b2, c) in s { if b2 != b { continue; }
            for &(c2, d) in t { if c2 == c { ds.insert(d); } } } }
    (!ds.is_empty(), ds.len() as u64)
}
/// The single-range reduction from v1, for contrast: T presence under prefix(a).
fn prefix_only(st: &Store, a: Key) -> bool {
    let span = 1u64 << (8 * (4 - depth(a)));
    st.keys.iter().zip(&st.pres[2]).any(|(&k, &p)| p && k >= a && k < a + span)
}

fn main() {
    let k = 0x0905_000A;
    let a = key(k, [1, 0, 0, 0]);
    let (b1, b2) = (key(k, [1, 2, 0, 0]), key(k, [1, 7, 0, 0]));
    let (c1, c2, c3) = (key(k, [1, 2, 3, 0]), key(k, [1, 7, 3, 0]), key(k, [1, 2, 8, 0]));
    let (d1, d2, d3) = (key(k, [1, 2, 3, 4]), key(k, [1, 7, 3, 4]), key(k, [1, 2, 8, 4]));
    let nodes = vec![a, b1, b2, c1, c2, c3, d1, d2, d3];
    let e = |v: &[(Key, Key)]| v.iter().copied().collect::<Edges>();
    let mut fails = 0;
    let mut check = |name: &str, got: Result<(bool, u64), Refuse>, want: (bool, u64)| {
        let ok = got.as_ref().map(|g| *g == want).unwrap_or(false);
        if !ok { fails += 1; }
        println!("{:<58} got={got:?} oracle={want:?} {}", name, if ok { "OK" } else { "MISMATCH" });
    };
    let run = |nodes: &[Key], r: &Edges, s: &Edges, t: &Edges| -> Result<(bool, u64), Refuse> {
        let st = Store::build(nodes, [r, s, t])?;
        let w = st.seal()?;
        let p = prepare(&st, &w, a, true)?;
        let v2 = execute(&st, &p, true)?;
        let lens = execute_lens(&st, &w, a)?;
        assert_eq!(v2, lens, "arm L (Gather through $i) disagrees with arm P (projection)");
        Ok(v2)
    };

    // 1. all three edges present (single path)
    let (r, s, t) = (e(&[(a, b1)]), e(&[(b1, c1)]), e(&[(c1, d1)]));
    check("1 all present", run(&nodes, &r, &s, &t), oracle(a, &r, &s, &t));
    // 2. each edge deleted independently; prefix-only shown for contrast
    for (name, rr, ss, tt) in [("2a R deleted", e(&[]), s.clone(), t.clone()), ("2b S deleted", r.clone(), e(&[]), t.clone()), ("2c T deleted", r.clone(), s.clone(), e(&[]))] {
        let st = Store::build(&nodes, [&rr, &ss, &tt]).unwrap();
        check(&format!("{name} (prefix-only says {})", prefix_only(&st, a)), run(&nodes, &rr, &ss, &tt), oracle(a, &rr, &ss, &tt));
    }
    // 3. three candidates, exactly one valid; decoys are broken at different hops
    //    d2: S(b2,c2), T(c2,d2) but no R(a,b2).  d3: R(a,b1), T(c3,d3) but no S(b1,c3).
    let (r, s, t) = (e(&[(a, b1)]), e(&[(b1, c1), (b2, c2)]), e(&[(c1, d1), (c2, d2), (c3, d3)]));
    check("3 three candidates, one valid", run(&nodes, &r, &s, &t), oracle(a, &r, &s, &t));
    // 4. same candidates, none valid
    let (r4, s4) = (e(&[(a, b1)]), e(&[(b2, c2)]));
    let st = Store::build(&nodes, [&r4, &s4, &t]).unwrap();
    check(&format!("4 three candidates, none valid (prefix-only says {})", prefix_only(&st, a)), run(&nodes, &r4, &s4, &t), oracle(a, &r4, &s4, &t));
    // two valid, to show Count is the number of distinct valid d (keys are unique)
    let r5 = e(&[(a, b1), (a, b2)]);
    check("3b three candidates, two valid", run(&nodes, &r5, &s, &t), oracle(a, &r5, &s, &t));

    // 5. permuted physical order against the witness
    let mut st = Store::build(&nodes, [&r, &s, &t]).unwrap();
    let w = st.seal().unwrap();
    st.keys.reverse();
    for p in st.pres.iter_mut() { p.reverse(); }
    println!("5 permuted rows, checked: prepare = {:?}", prepare(&st, &w, a, true).err());
    let unchecked = prepare(&st, &w, a, false).and_then(|p| execute(&st, &p, false));
    println!("5 permuted rows, UNCHECKED arm: {unchecked:?} vs oracle {:?}  (silent wrong answer when the check is removed)", oracle(a, &r, &s, &t));
    println!("5 sealing a permuted lane: {:?}", st.seal().err());

    // 6. duplicate terminal key
    let mut dup = nodes.clone(); dup.push(d1);
    let mut st = Store::build(&dup, [&r, &s, &t]).unwrap();
    // a T table holding the edge (c1,d1) twice stamps BOTH rows of d1
    for i in 0..st.keys.len() { if st.keys[i] == d1 { st.pres[2][i] = true; } }
    println!("6 duplicate d1, seal: {:?}", st.seal().err());
    let w_forged = Witness { key_version: 1, presence_version: 1, n_rows: st.keys.len(), digest: fnv(&st.keys) };
    let unchecked = prepare(&st, &w_forged, a, true).and_then(|p| execute(&st, &p, false));
    println!("6 duplicate d1, forged witness: {unchecked:?} vs oracle {:?}  (Count = rows, not distinct d)", oracle(a, &r, &s, &t));

    // 7. cross-ClassId: another class (same concept hi u16, other app lo u16), same tiers,
    //    a fully valid chain of its own; plus an edge crossing the class boundary.
    let k2 = 0x0905_000B;
    let (a2, b2x, c2x, d2x) = (key(k2, [1, 0, 0, 0]), key(k2, [1, 2, 0, 0]), key(k2, [1, 2, 3, 0]), key(k2, [1, 2, 3, 4]));
    let mut n7 = nodes.clone(); n7.extend([a2, b2x, c2x, d2x]);
    let (r7, s7, t7) = (e(&[(a, b1), (a2, b2x)]), e(&[(b2x, c2x)]), e(&[(c2x, d2x)]));
    // class K has R only; class K2 has a full chain. Query under a (class K) must say false.
    check("7 other class holds the only valid chain", run(&n7, &r7, &s7, &t7), oracle(a, &r7, &s7, &t7));
    let cross = e(&[(b1, c2x)]);
    println!("7 edge crossing the classid boundary: build = {:?}", Store::build(&n7, [&r7, &cross, &t7]).err().map(|_| "refused (NonRefiningEdge)"));

    // 8. version change after preparation: S(b1,c1) deleted at presence_version 2
    let mut st = Store::build(&nodes, [&r, &s, &t]).unwrap();
    let w = st.seal().unwrap();
    let p = prepare(&st, &w, a, true).unwrap();
    println!("8 before change: {:?}", execute(&st, &p, true));
    let ci = st.keys.binary_search(&c1).unwrap();
    st.pres[1][ci] = false; st.presence_version = 2;
    println!("8 after S(b1,c1) deleted, stale projection, checked: {:?}", execute(&st, &p, true).err());
    println!("8 stale projection, UNCHECKED: {:?} (silent stale true)", execute(&st, &p, false));
    let w2 = st.seal().unwrap();
    let p2 = prepare(&st, &w2, a, true).unwrap();
    let s8 = e(&[(b2, c2)]);
    check("8 re-prepared at version 2", execute(&st, &p2, true), oracle(a, &r, &s8, &t));
    println!("8 arm L under the stale witness: {:?}", execute_lens(&st, &w, a).err());
    check("8 arm L at version 2", execute_lens(&st, &w2, a), oracle(a, &r, &s8, &t));
    println!("preparation seeks for case 8: {}", p2.seeks);

    println!("\n{} mismatches against the oracle", fails);
    std::process::exit(i32::from(fails != 0));
}
